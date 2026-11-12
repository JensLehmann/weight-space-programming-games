"""Fast CPU tests for wsp.train_bc (teacher defaults, augmentation hook, BPTT, progressive, ckpt)."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wsp import actor as A
from wsp import train_bc as B
from wsp.envs import A_INDEX, build_start, obs_of
from wsp.queries import reviewed_start


@pytest.fixture
def short_T(monkeypatch):
    """8 tied iterations instead of 128 (CPU speed); deployment code paths are unchanged."""
    monkeypatch.setattr(A, "T_ITER", 8)
    if hasattr(A, "VARIANTS"):                # actor variants read T at construction
        monkeypatch.setitem(A.VARIANTS["full"], "T", 8)


def _toy_specs(n=2, cap=10):
    return [replace(reviewed_start("toy_reach", s, 0, "S-start"), cap=cap) for s in range(n)]


def _inputs(W=2, L=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randint(0, 16, (W, L, 64, 64), generator=g)
    mem0 = torch.rand(W, A.MEM, generator=g) * 2 - 1
    prev = torch.zeros(W, L, 8)
    prev[:, :, 2] = 1
    avail = torch.ones(W, L, 8, dtype=torch.bool)
    avail[:, :, 7] = False
    return obs, mem0, prev, avail


def _actor(seed=0):
    return A.Actor(A.random_params(seed, scale=0.05))


def test_forward_window_matches_actor_forward(short_T):
    actor = _actor()
    obs, mem0, prev, avail = _inputs()
    reset = torch.zeros(2, 3, dtype=torch.bool)
    reset[1, 2] = True
    with torch.no_grad():
        lg, ck, mems = B.forward_window(actor, obs, mem0, prev, avail, reset)
        m = mem0
        for t in range(3):
            m = torch.where(reset[:, t, None], torch.zeros_like(m), m)
            l1, c1, m = actor(obs[:, t], m, prev[:, t], avail[:, t])
            fin = torch.isfinite(l1)
            assert torch.allclose(lg[:, t][fin], l1[fin], atol=1e-5)
            assert torch.equal(torch.isfinite(lg[:, t]), fin)
            assert torch.allclose(ck[:, t], c1, atol=1e-5)
            assert torch.allclose(mems[:, t], m, atol=1e-6)
        l1, c1, m1 = actor(obs[:, 0], mem0, prev[:, 0], avail[:, 0])
        l2, c2, m2 = B.forward_step(actor, obs[:, 0], mem0, prev[:, 0], avail[:, 0], ckpt=3)
        assert torch.allclose(l1[torch.isfinite(l1)], l2[torch.isfinite(l2)], atol=1e-5)
        assert torch.allclose(m1, m2, atol=1e-6)


def test_checkpoint_and_progressive_preserve_values_and_grads(short_T):
    actor = _actor(1)
    obs = torch.randint(0, 16, (2, 64, 64), generator=torch.Generator().manual_seed(1))
    T = B.depth(actor)
    assert T == 8
    f3, h = actor.encode(obs)
    for kw in ({"ckpt": 3}, {"n_nograd": 5, "n_iter": T}, {"n_nograd": 2, "n_iter": T, "ckpt": 2}):
        f3b, hb = B.features(actor, obs, **kw)
        assert torch.allclose(h, hb, atol=1e-6) and torch.allclose(f3, f3b)
    if "noaxial" in getattr(A, "VARIANTS", {}):   # replica follows the operator-ablation variants
        ab = A.Actor(A.random_params(1, scale=0.05), variant="noaxial")
        assert torch.allclose(ab.encode(obs)[1], B.features(ab, obs, ckpt=4)[1], atol=1e-6)
    # checkpointing leaves gradients unchanged
    grads = []
    for kw in ({}, {"ckpt": 3}):
        actor.zero_grad()
        _, h2 = B.features(actor, obs, **kw) if kw else actor.encode(obs)
        h2.square().mean().backward()
        grads.append(actor.Wh.grad.clone())
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-8)
    # progressive: fewer iterations change the output, no-grad prefix still gives Wh a gradient
    actor.zero_grad()
    _, h3 = B.features(actor, obs, n_iter=5, n_nograd=3)
    h3.square().mean().backward()
    assert actor.Wh.grad.abs().sum() > 0
    rng = np.random.default_rng(0)
    draws = [B.progressive_depth(rng, 128) for _ in range(400)]
    full = sum(not d for d in draws)
    assert 120 < full < 280
    for d in draws:
        if d:
            assert 0 <= d["n_nograd"] < d["n_iter"] <= 128


def test_bptt_gradient_flows_through_memory(short_T):
    actor = _actor(2)
    obs, mem0, prev, avail = _inputs(W=1, L=4)
    mem0 = mem0.clone().requires_grad_(True)
    lg, _, _ = B.forward_window(actor, obs, mem0, prev, avail)
    lg[0, -1, :7].sum().backward()
    assert mem0.grad.abs().sum() > 0              # step 3 depends on the memory entering step 0
    assert actor.Wm.grad.abs().sum() > 0 and actor.Wma.grad.abs().sum() > 0


def test_make_windows():
    ep = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2])
    act = np.array([1, 1, -1, 1, 1, -1, -1, 2, -1, 3])
    idx, valid = B.make_windows(ep, act, 2)
    assert idx.shape[1] == 2
    covered = sorted(int(i) for i, v in zip(idx.ravel(), valid.ravel()) if v)
    assert covered == [0, 1, 2, 3, 4, 7, 8, 9]    # episode 1 has no labels -> dropped
    for row, v in zip(idx, valid):                 # windows never cross episodes
        assert len(set(ep[row[v]])) == 1
    idx, valid = B.make_windows(ep, act, 4, np.random.default_rng(0))
    covered = sorted(int(i) for i, v in zip(idx.ravel(), valid.ravel()) if v)
    assert covered == [0, 1, 2, 3, 4, 7, 8, 9]


class _FakeOracle:
    def __init__(self, first):
        self.first = first
        self.stats = {}

    def info(self, game):
        return B.PlanInfo(self.first, None)


def test_planner_teacher_defaults_and_no_reset_fallback():
    t = B.PlannerTeacher(oracle=_FakeOracle(None))
    assert t.budget_ms == 5000 and t.reset_fallback is False
    assert t.label(0, None, None) is None and t.counts["no_plan"] == 1
    t1 = B.PlannerTeacher(budget_ms=2000, reset_fallback=True, oracle=_FakeOracle(None))
    assert t1.label(0, None, None) == ("RESET", None, None)
    assert B.PlannerTeacher(oracle=_FakeOracle(("ACTION2", None, None))).label(0, None, None)[0] == "ACTION2"


def test_planner_oracle_real_planner_and_cache():
    from wsp.envs import EpisodeSpec
    game = build_start(EpisodeSpec("lys", 0, 0))
    o = B.PlannerOracle(budget_ms=5000)
    info = o.info(game)
    assert info.first is not None and info.first[0] in A_INDEX and info.dist and info.dist > 0
    assert o.info(game) == info and o.stats["cache_hits"] == 1 and o.stats["queries"] == 1
    toy = build_start(_toy_specs(1)[0])
    ti = B.PlannerOracle().info(toy)
    assert ti.dist == len(toy.direct_plan()) and ti.first[0] == toy.direct_plan()[0]


class _HalfTeacher:
    """Labels with the toy BFS plan, but refuses every other call (unlabelled states)."""
    budget_ms, reset_fallback = None, None

    def __init__(self):
        self.n = 0

    def begin(self, i): pass

    def label(self, i, game, obs):
        self.n += 1
        return None if self.n % 2 == 0 else (game.direct_plan()[0], None, None)

    def close(self): pass


def test_dagger_counts_and_skips_unlabelled(short_T, tmp_path):
    specs = _toy_specs(2, cap=8)
    buf = B.Buffer.new()
    st = B.collect(specs, A.Actor(B.init_params(0)), _HalfTeacher(), 0.0, buf, "cpu",
                   __import__("random").Random(0))
    assert st["unlabelled"] > 0 and len(buf) == st["visited"] - st["unlabelled"]
    buf2 = B.Buffer.new()
    st2 = B.collect(specs, A.Actor(B.init_params(0)), _HalfTeacher(), 0.0, buf2, "cpu",
                    __import__("random").Random(0), sequences=True)
    assert len(buf2) == st2["visited"] and sum(a < 0 for a in buf2.act) == st2["unlabelled"]
    assert len(set(buf2.ep)) == len(specs)
    res = B.dagger(specs, _HalfTeacher(), tmp_path / "d", seed=0, iters=1, epochs=1, batch=8)
    assert 0 < res["unlabelled_frac"] < 1
    log = json.loads((tmp_path / "d" / "dagger_log.json").read_text())
    assert all("unlabelled_frac" in e for e in log)
    assert (tmp_path / "d" / "student.npz").exists()


@pytest.mark.parametrize("opts", [
    {},
    {"augment": {"palette", "dihedral"}},
    {"bptt": 4},
    {"bptt": 4, "augment": {"palette", "dihedral"}, "grad_ckpt": 3},
    {"progressive": True},
    {"progressive": True, "bptt": 3, "grad_ckpt": 2, "max_samples": 12},
])
def test_dagger_short_runs(short_T, tmp_path, opts):
    specs = _toy_specs(2, cap=8)
    res = B.dagger(specs, B.PlannerTeacher(), tmp_path / "d", seed=1, iters=1, epochs=2,
                   batch=8, **opts)
    assert len(res["log"]) == 2 and all(np.isfinite(e["loss"]) for e in res["log"])
    assert res["unlabelled_frac"] == 0.0              # BFS teacher labels every toy state
    p = A.load_npz(tmp_path / "d" / "student.npz")
    p0 = B.init_params(1)
    assert any(not np.array_equal(p[k], p0[k]) for k in A.SCHEMA)
    cfg = json.loads((tmp_path / "d" / "train_config.json").read_text())
    assert cfg["bptt"] == opts.get("bptt") and cfg["teacher_budget_ms"] == 5000
    if "max_samples" in opts:
        assert res["log"][-1]["samples"] <= opts["max_samples"]


class _NeverTeacher(_HalfTeacher):
    def label(self, i, game, obs):
        return None


def test_dagger_survives_fully_unlabelled_iteration(short_T, tmp_path):
    res = B.dagger(_toy_specs(1, cap=4), _NeverTeacher(), tmp_path / "d", seed=0, iters=1, epochs=1)
    assert res["unlabelled_frac"] == 1.0 and all(np.isnan(e["loss"]) for e in res["log"])

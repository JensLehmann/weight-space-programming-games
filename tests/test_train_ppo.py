"""Fast CPU tests for wsp.train_ppo (recurrent PPO, shaping, structure-preserving options, eval)."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from wsp import actor as A
from wsp import train_ppo as P
from wsp.train_bc import PlannerOracle, init_params
from wsp.queries import reviewed_start


@pytest.fixture
def short_T(monkeypatch):
    monkeypatch.setattr(A, "T_ITER", 8)
    if hasattr(A, "VARIANTS"):
        monkeypatch.setitem(A.VARIANTS["full"], "T", 8)


def _toy_specs(n=2, cap=8):
    return [replace(reviewed_start("toy_reach", s, 0, "S-start"), cap=cap) for s in range(n)]


def test_eval_schedule():
    nxt = P.eval_schedule([0, 10_000, 40_000, 80_000, 160_000], 200_000)
    seq, x = [], -1
    for _ in range(8):
        x = nxt(x)
        seq.append(x)
    assert seq == [0, 10_000, 40_000, 80_000, 160_000, 200_000, 400_000, 600_000]
    assert nxt(12_345) == 40_000 and nxt(170_000) == 200_000
    assert P.eval_schedule([], None)(-1) == math.inf
    assert P.eval_schedule([], 100)(-1) == 100


def test_windows_cover_horizon():
    flat, valid = P._windows(H=10, E=3, L=4)
    assert flat.shape == (9, 4) and valid.sum() == 30
    assert sorted(flat[valid].tolist()) == list(range(30))
    for row, v in zip(flat, valid):                       # one env per window, consecutive steps
        r = row[v]
        assert len(set((r % 3).tolist())) == 1 and ((r[1:] - r[:-1]) == 3).all()


def test_planner_shaping_reward_toy():
    spec = _toy_specs(1, cap=50)[0]
    env = P.VecEnv([spec], 1, 0, shaped=PlannerOracle(200), gamma=0.9)
    d0 = len(env.games[0].direct_plan())
    assert env.phi[0] == -d0
    a = env.games[0].direct_plan()[0]
    r, done = env.step([(a, None, None)], np.zeros((1, A.MEM), np.float32))
    assert not done[0] and r[0] == pytest.approx(0.9 * -(d0 - 1) + d0)
    assert env.shaping["applied"] == 1 and env.shaping["uncertified"] == 0
    # terminal transitions use Phi(s') = 0
    env2 = P.VecEnv([replace(spec, cap=1)], 1, 0, shaped=PlannerOracle(200), gamma=0.9)
    d = len(env2.games[0].direct_plan())
    r, done = env2.step([("RESET", None, None)], np.zeros((1, A.MEM), np.float32))
    assert done[0] and r[0] == pytest.approx(d)


class _NoCert:
    stats = {}

    def distance(self, game):
        return None


def test_uncertified_steps_get_no_shaping():
    env = P.VecEnv(_toy_specs(1, cap=50), 1, 0, shaped=_NoCert())
    r, _ = env.step([("ACTION1", None, None)], np.zeros((1, A.MEM), np.float32))
    assert r[0] == 0 and env.shaping["uncertified"] == 1 and env.shaping["applied"] == 0


def test_policy_kl_and_margin():
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(5, 8, generator=g)
    av = torch.ones(5, 8, dtype=torch.bool)
    av[:, 0] = False
    lg = lg.masked_fill(~av, float("-inf"))
    ck = torch.randn(5, 64, 64, generator=g)
    assert torch.allclose(P.policy_kl(lg, ck, lg, ck, av), torch.zeros(5), atol=1e-6)
    assert (P.policy_kl(lg, ck, lg.flip(-1).masked_fill(~av, float("-inf")), ck * 2, av) > 0).all()
    assert P.action_margin(lg, av) >= 0
    bad = lg.clone()
    bad[0, 3] = float("nan")
    s, c, n = P.sanitize(bad, ck, av)
    assert n == 1 and torch.isfinite(s[av]).all() and torch.isneginf(s[~av]).all()


def _run(tmp_path, **kw):
    specs = _toy_specs(2, cap=8)
    base = dict(seed=0, total_steps=32, n_envs=2, horizon=8, minibatch=8, epochs=2, device="cpu",
                log_every=1)
    base.update(kw)
    res = P.ppo(specs, tmp_path / "p", **base)
    log = json.loads((tmp_path / "p" / "ppo_log.json").read_text())
    assert all("steps" in e and "solve_rate_window" in e for e in log)   # original schema kept
    for e in log:
        for k in ("entropy", "margin", "kl_init", "grad_norm", "nonfinite_rollout", "ratio0_maxdev"):
            assert k in e
        assert e["ratio0_maxdev"] < 1e-3                 # recomputed policy == rollout policy
    return res, log


@pytest.mark.parametrize("opts", [
    {},
    {"bptt": 4},
    {"bptt": 3, "grad_ckpt": 3},                        # window length not dividing the horizon
    {"shaping": "planner"},
    {"optimizer": "sgd", "lr": 1e-3},
    {"kl_to_init": 0.1, "ent": 0.0, "lr": 1e-3},
    {"bptt": 4, "kl_to_init": 0.1, "masked": True, "shaping": "planner", "critic_warmup": 16,
     "optimizer": "sgd", "lr": 1e-2, "critic_lr": 1e-3},
])
def test_ppo_short_runs(short_T, tmp_path, opts):
    res, log = _run(tmp_path, **opts)
    assert len(log) == 2
    p0 = init_params(0)
    assert any(not np.array_equal(res["params"][k], p0[k]) for k in A.SCHEMA)
    if opts.get("shaping") == "planner":
        assert all(e["shaping_uncertified_frac"] == 0.0 for e in log)
        assert log[-1]["planner_queries"] > 0
    if opts.get("kl_to_init"):
        assert "kl_train" in log[-1] and log[-1]["kl_init"] > 0


def test_ppo_masked_updates_keep_zero_entries(short_T, tmp_path):
    init = init_params(3)
    init["Wo"][:, ::2] = 0
    res, _ = _run(tmp_path, masked=True, init=init, lr=1e-2)
    for k in A.SCHEMA:
        z = init[k] == 0
        assert (res["params"][k][z] == 0).all(), k
    assert not np.array_equal(res["params"]["Wo"], init["Wo"])


def test_ppo_critic_warmup_freezes_actor(short_T, tmp_path):
    init = init_params(4)
    res, log = _run(tmp_path, init=init, critic_warmup=10_000, optimizer="sgd", lr=1e-2)
    assert all(np.array_equal(res["params"][k], A.validate(init)[k]) for k in A.SCHEMA)
    assert all(e["critic_warmup"] for e in log)
    res2, log2 = _run(tmp_path, init=init, critic_warmup=16)      # first update only
    assert [e["critic_warmup"] for e in log2] == [True, False]


def test_ppo_eval_checkpoints(short_T, tmp_path):
    specs = _toy_specs(2, cap=8)
    res, _ = _run(tmp_path, eval_specs={"S-start": specs, "Q": specs[:1]}, eval_at=[0, 16],
                  eval_every=None, eval_batch=4)
    ev = json.loads((tmp_path / "p" / "ppo_eval.json").read_text())
    at0 = [(e["set"], e["decoding"], e["actor"]) for e in ev if e["eval_steps"] == 0]
    assert sorted(at0) == sorted([("S-start", "greedy", "init"), ("S-start", "sampled", "init"),
                                  ("Q", "greedy", "init"), ("Q", "sampled", "init")])
    steps = sorted({e["eval_steps"] for e in ev})
    assert steps == [0, 16, 32]                    # listed checkpoint + final evaluation
    assert all("solve_rate" in e and e["n"] in (1, 2) for e in ev)
    assert res["evals"] == ev


def test_sampled_policy_runs(short_T):
    from wsp.envs import run_episodes
    actor = A.Actor(init_params(0))
    out = run_episodes(_toy_specs(2, cap=5), P.SampledPolicy(actor, "cpu", seed=0), keep_trace=True)
    assert all(r.actions == 5 or r.success for r in out)
    assert all(m[0] in ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4") or m[0] == "INVALID"
               for r in out for m in r.trace)

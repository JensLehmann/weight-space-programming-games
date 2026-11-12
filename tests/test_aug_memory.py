"""aug_memory='recompute' (wsp.train_bc): per-episode transforms with memory recomputed by the student.

* identity transform: recompute reproduces the stored-memory path (memories and one training epoch)
  when the buffer was collected by the same student weights: bitwise with batch-1 recompute
  (one episode per chunk, one frame per encoder call, as in collection), up to float32 rounding
  (~1e-7) with the batched defaults;
* non-identity transform: the recomputed memory entering every step equals a direct step-by-step
  rollout of the student over the independently transformed episode prefix.
"""
import copy
import functools
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from wsp import actor as A
from wsp import augment as G
from wsp import train_bc as B
from wsp.queries import reviewed_start


@pytest.fixture
def short_T(monkeypatch):
    """8 tied iterations instead of 128 (CPU speed); deployment code paths are unchanged."""
    monkeypatch.setattr(A, "T_ITER", 8)
    monkeypatch.setitem(A.VARIANTS["full"], "T", 8)


def _specs(n=3, cap=12):
    return [replace(reviewed_start("toy_reach", s, 0, "S-start"), cap=cap) for s in range(n)]


def _student(seed=0):
    return A.Actor(B.init_params(seed))


def _sequence_buffer(student, n=3, cap=12, beta=0.5, seed=0):
    """Sequence buffer collected by `student` (beta-mixture with the toy BFS teacher)."""
    buf = B.Buffer.new()
    B.collect(_specs(n, cap), student, B.PlannerTeacher(), beta, buf, "cpu", random.Random(seed),
              sequences=True)
    return buf


def _tensors(buf):
    return (torch.as_tensor(np.stack(buf.obs)), torch.as_tensor(np.stack(buf.avail)),
            torch.as_tensor(np.stack(buf.prev)), torch.as_tensor(np.array(buf.act)),
            torch.as_tensor(np.array(buf.cell)))


def _spans(ep):
    ep = np.asarray(ep)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    return list(zip(starts, np.r_[starts[1:], len(ep)]))


@pytest.mark.parametrize("chunk,enc", [(B.RECOMPUTE_CHUNK_FRAMES, B.RECOMPUTE_ENC_BATCH), (1, 1), (20, 7)])
def test_identity_recompute_equals_stored_memory(short_T, chunk, enc):
    student = _student(0)
    buf = _sequence_buffer(student)
    assert len(_spans(buf.ep)) == 3 and len(set(len(range(*s)) for s in _spans(buf.ep))) > 1
    obs, av, prev, act, cell = _tensors(buf)
    rng = np.random.default_rng(0)
    state = copy.deepcopy(rng.bit_generator.state)
    o2, av2, p2, y2, c2, m2 = B.transformed_episodes(student, obs, av, prev, act, cell, buf.ep, None,
                                                     rng, "cpu", chunk_frames=chunk, enc_batch=enc)
    assert rng.bit_generator.state == state                    # identity: no rng draws
    assert torch.equal(o2, obs) and torch.equal(av2, av) and torch.equal(p2, prev)
    assert torch.equal(y2, act) and torch.equal(c2, cell)
    stored = torch.as_tensor(np.stack(buf.mem))
    assert stored.abs().max() > 0.1                            # a non-trivial memory
    assert torch.allclose(m2, stored, atol=1e-6, rtol=0), float((m2 - stored).abs().max())
    if chunk == enc == 1:                                      # batch-1, as in collection: bitwise
        assert torch.equal(m2, stored)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("opts", [{"bptt": 4}, {"bptt": 4, "progressive": True}, {"bptt": None}])
def test_identity_recompute_reproduces_stored_training(short_T, monkeypatch, opts, exact):
    """Same student weights as at collection, one epoch: identical windows, memories and updates."""
    if exact:                                                  # batch-1 recompute -> bitwise
        monkeypatch.setattr(B, "transformed_episodes",
                            functools.partial(B.transformed_episodes, chunk_frames=1, enc_batch=1))
    student = _student(1)
    buf = _sequence_buffer(student, seed=1)
    out = {}
    for mode in B.AUG_MEMORY:
        s = copy.deepcopy(student)
        loss = B.train(s, buf, "cpu", 1, 8, 1e-3, np.random.default_rng(5), augment=None,
                       aux_rng=np.random.default_rng([5, 1]), aug_memory=mode, **opts)
        out[mode] = (loss, s.export())
    (l0, p0), (l1, p1) = out["stored"], out["recompute"]
    assert np.isfinite(l0) and abs(l0 - l1) < 1e-5
    p_init = student.export()
    assert any(not np.array_equal(p0[k], p_init[k]) for k in A.SCHEMA)
    for k in A.SCHEMA:
        assert np.allclose(p0[k], p1[k], atol=1e-6, rtol=0), (k, float(np.abs(p0[k] - p1[k]).max()))
        if exact:
            assert np.array_equal(p0[k], p1[k]), k
    if exact:
        assert l0 == l1


def _match_transform(orig, new):
    """All (e, colour map) with new == map(transform_grid(orig, e)) for every frame of an episode."""
    out = []
    for e in G.ELEMENTS:
        g = G.transform_grid(orig, e)
        cmap = {}
        for c in np.unique(g):
            vals = np.unique(new[g == c])
            if len(vals) != 1:
                break
            cmap[int(c)] = int(vals[0])
        else:
            if len(set(cmap.values())) == len(cmap):           # injective on the colours present
                out.append((e, cmap))
    return out


def _expected(o, av, pv, a, c, e, cmap):
    """Independent (NumPy reference) transform of one episode by D4 element e and colour map."""
    perm = G.action_perm(e)
    o2 = G.transform_grid(np.vectorize(cmap.get)(o), e)
    av2 = np.zeros_like(av); av2[:, perm] = av
    pv2 = np.zeros_like(pv); pv2[:, perm] = pv
    a2 = np.where(a >= 0, perm[np.maximum(a, 0)], a)
    c2 = np.array([-1 if x < 0 else (lambda rc: rc[0] * 64 + rc[1])(G.transform_cell(x // 64, x % 64, e))
                   for x in c])
    return o2, av2, pv2, a2, c2


@pytest.mark.parametrize("chunk", [B.RECOMPUTE_CHUNK_FRAMES, 1, 20])
def test_nonidentity_recompute_equals_direct_rollout(short_T, chunk):
    student = _student(2)
    buf = _sequence_buffer(student, n=4, seed=2)
    obs, av, prev, act, cell = _tensors(buf)
    o2, av2, p2, y2, c2, m2 = B.transformed_episodes(
        student, obs, av, prev, act, cell, buf.ep, {"palette", "dihedral"},
        np.random.default_rng(3), "cpu", chunk_frames=chunk, enc_batch=5)
    stored = np.stack(buf.mem)
    non_identity = 0
    for s, t in _spans(buf.ep):
        O, X = obs[s:t].numpy().astype(int), o2[s:t].numpy().astype(int)
        cands = _match_transform(O, X)
        ok = []
        for e, cmap in cands:
            exp = _expected(O, av[s:t].numpy(), prev[s:t].numpy(), act[s:t].numpy(), cell[s:t].numpy(),
                            e, cmap)
            got = (X, av2[s:t].numpy(), p2[s:t].numpy(), y2[s:t].numpy(), c2[s:t].numpy())
            if all(np.array_equal(x, y) for x, y in zip(exp, got)):
                ok.append((e, cmap, exp))
        assert ok, "episode not transformed by a single D4 element + palette permutation"
        e, cmap, (xo, xav, xpv, _, _) = ok[0]
        non_identity += int(e != G.IDENTITY or any(k != v for k, v in cmap.items()))
        # direct rollout of the student over the transformed prefix, from zero memory
        m = torch.zeros(1, A.MEM)
        with torch.no_grad():
            for k in range(t - s):
                assert torch.allclose(m2[s + k], m[0], atol=1e-6, rtol=0), (s, k)
                _, _, m = student(torch.as_tensor(xo[k][None]), m, torch.as_tensor(xpv[k][None]),
                                  torch.as_tensor(xav[k][None]))
        # memory stored at collection (untransformed frames) differs from the consistent one
        if e != G.IDENTITY or any(k != v for k, v in cmap.items()):
            assert np.abs(m2[s:t].numpy() - stored[s:t])[1:].max() > 1e-4
    assert non_identity >= 1


@pytest.mark.parametrize("opts", [
    {"bptt": 4, "augment": {"palette", "dihedral"}, "progressive": True},
    {"bptt": None, "augment": {"palette", "dihedral"}},
    {"bptt": 3, "augment": {"dihedral"}, "grad_ckpt": 2, "max_samples": 12},
])
def test_dagger_recompute_short_runs(short_T, tmp_path, opts):
    res = B.dagger(_specs(2, cap=8), B.PlannerTeacher(), tmp_path / "d", seed=1, iters=1, epochs=2,
                   batch=8, aug_memory="recompute", **opts)
    assert len(res["log"]) == 2 and all(np.isfinite(e["loss"]) for e in res["log"])
    assert all("train_s" in e for e in res["log"])
    cfg = json.loads((tmp_path / "d" / "train_config.json").read_text())
    assert cfg["aug_memory"] == "recompute" and cfg["bptt"] == opts["bptt"]
    if "max_samples" in opts:
        assert res["log"][-1]["samples"] <= opts["max_samples"]


def test_default_is_stored_and_recompute_needs_sequences(short_T, tmp_path):
    res = B.dagger(_specs(1, cap=4), B.PlannerTeacher(), tmp_path / "d", seed=0, iters=0, epochs=1,
                   batch=8, augment={"palette"})
    assert res["config"]["aug_memory"] == "stored"
    buf = B.Buffer.new()
    B.collect(_specs(1, cap=4), None, B.PlannerTeacher(), 1.0, buf, "cpu", random.Random(0))
    with pytest.raises(ValueError, match="sequence storage"):
        B.train(_student(), buf, "cpu", 1, 8, 1e-3, np.random.default_rng(0), aug_memory="recompute")
    with pytest.raises(ValueError, match="aug_memory"):
        B.train(_student(), buf, "cpu", 1, 8, 1e-3, np.random.default_rng(0), aug_memory="fresh")

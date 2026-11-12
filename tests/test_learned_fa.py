"""Learned free-architecture students (wsp.train_bc.LearnedFA): init, executor, training, recompute, saving."""
import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from wsp import augment as G
from wsp import freearch as FA
from wsp import train_bc as B
from wsp.queries import reviewed_start

M = 4


def tiny_arch(mem=M, T=3):
    """Small graph: conv features, axial/min/max ops, a tied loop, a memory MLP, logits and click."""
    g = [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},
        {"id": "c1", "op": "conv", "in": ["x"], "w": "W1", "b": "b1"},
        {"id": "f1", "op": "relu", "in": ["c1"]},
        {"id": "rm", "op": "rowmax", "in": ["f1"]},
        {"id": "ab", "op": "abs", "in": ["c1"]},
        {"id": "mx", "op": "max2", "in": ["f1", "rm"]},
        {"id": "mn", "op": "min2", "in": ["mx", "ab"]},
        {"id": "L", "op": "loop", "T": T, "state": [{"id": "h", "zeros": [4, 64, 64]}],
         "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh"},
                  {"id": "d", "op": "conv", "in": ["f1"], "w": "Wd", "b": "bd"},
                  {"id": "hs", "op": "add", "in": ["hc", "d"]},
                  {"id": "hn", "op": "tanh", "in": ["hs"]}],
         "next": {"h": "hn"}},
        {"id": "g1", "op": "gmax", "in": ["L.h"]},
        {"id": "g2", "op": "gmean", "in": ["mn"]},
        {"id": "g", "op": "vconcat", "in": ["g1", "g2"]},                       # 12
        {"id": "gate", "op": "dense", "in": ["g"], "w": "Wg", "b": "bg"},
        {"id": "gs", "op": "vsigmoid", "in": ["gate"]},
    ]
    if mem:
        g += [{"id": "m1", "op": "dense", "in": ["mem"], "w": "Wm"},
              {"id": "m2", "op": "dense", "in": ["g"], "w": "Wmg", "b": "bm"},
              {"id": "m3", "op": "dense", "in": ["aprev"], "w": "Wma"},
              {"id": "ms", "op": "vadd", "in": ["m1", "m2"]},
              {"id": "ms2", "op": "vadd", "in": ["ms", "m3"]},
              {"id": "mem_next", "op": "vtanh", "in": ["ms2"]},
              {"id": "zm", "op": "vconcat", "in": ["gs", "mem_next"]}]
    else:
        g += [{"id": "zm", "op": "vconcat", "in": ["gs", "g"]}]
    g += [{"id": "logits", "op": "dense", "in": ["zm"], "w": "Wo", "b": "bo"},
          {"id": "fh", "op": "concat", "in": ["f1", "L.h"]},
          {"id": "click", "op": "conv", "in": ["fh"], "w": "Wc", "b": "bc"}]
    return {"graph": g, "memory_size": mem}


def tiny_params(mem=M, seed=0):
    r = np.random.default_rng(seed)
    shapes = {"W1": (8, 19, 3, 3), "b1": (8,), "Wh": (4, 4, 3, 3), "Wd": (4, 8, 1, 1), "bd": (4,),
              "Wg": (12, 12), "bg": (12,), "Wc": (1, 12, 1, 1), "bc": (1,),
              "Wo": (8, 12 + (mem if mem else 12)), "bo": (8,)}
    if mem:
        shapes.update({"Wm": (mem, mem), "Wmg": (mem, 12), "bm": (mem,), "Wma": (mem, 8)})
    return {k: r.normal(0, 0.3, s).astype(np.float32) for k, s in shapes.items()}


def _net(mem=M, seed=0, bias="uniform"):
    arch, written = FA.validate(tiny_arch(mem), tiny_params(mem))
    return B.LearnedFA(arch, B.fa_init_params(arch, written, seed, bias=bias)), arch, written


def _window(W=2, L=3, seed=0, mem=M):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randint(0, 16, (W, L, 64, 64), generator=g)
    mem0 = torch.rand(W, mem, generator=g) * 2 - 1
    prev = torch.zeros(W, L, 8)
    prev[:, :, 2] = 1
    avail = torch.ones(W, L, 8, dtype=torch.bool)
    avail[:, :, 7] = False
    return obs, mem0, prev, avail


def _specs(n=3, cap=10):
    return [replace(reviewed_start("toy_reach", s, 0, "S-start"), cap=cap) for s in range(n)]


def test_init_scheme():
    arch, written = FA.validate(tiny_arch(), tiny_params())
    p0 = B.fa_init_params(arch, written, 0)
    pu = B.fa_init_params(arch, written, 0, bias="uniform")
    assert {k: v.shape for k, v in p0.items()} == {k: v.shape for k, v in written.items()}
    assert B._fa_recurrent_params(arch["graph"]) == {"Wh"}                  # Wd reads only f1
    for k, v in p0.items():
        if v.ndim == 1:
            assert not v.any() and pu[k].any()                              # zero / uniform biases
            continue
        assert np.array_equal(v, pu[k])                                     # weights identical
        bound = 1 / np.sqrt(np.prod(v.shape[1:])) * (0.1 if k == "Wh" else 1.0)
        assert np.abs(v).max() <= bound and np.abs(v).max() > 0.5 * bound
        assert not np.array_equal(v, B.fa_init_params(arch, written, 1)[k])
    assert np.abs(pu["bd"]).max() <= 1 / np.sqrt(8)                          # fan_in of Wd (8,1,1)
    net = B.LearnedFA(arch, p0)
    assert {n for n, _ in net.named_parameters()} == {"p_" + k for k in written} and not any(
        k.startswith("p_") for k in net._buffers)


@pytest.mark.parametrize("mem", [M, 0])
def test_executor_matches_freearchnet(mem):
    net, arch, _ = _net(mem)
    ref = FA.FreeArchNet(arch, net.export())
    obs, mem0, prev, avail = _window(mem=mem)
    with torch.no_grad():
        for t in range(3):
            a = net(obs[:, t], mem0, prev[:, t], avail[:, t])
            b = ref(obs[:, t], mem0, prev[:, t], avail[:, t])
            c = B.fa_forward_step(net, obs[:, t], mem0, prev[:, t], avail[:, t])
            for x, y, z in zip(a, b, c):
                assert torch.equal(x, y) and torch.equal(x, z)
        lg, ck, ms = B.fa_forward_window(net, obs, mem0, prev, avail)
        m = mem0
        for t in range(3):                                                   # window == step-by-step rollout
            l1, c1, m = ref(obs[:, t], m, prev[:, t], avail[:, t])
            fin = torch.isfinite(l1)
            assert torch.equal(torch.isfinite(lg[:, t]), fin)
            assert torch.allclose(lg[:, t][fin], l1[fin], atol=1e-5)
            assert torch.allclose(ck[:, t], c1, atol=1e-5) and torch.allclose(ms[:, t], m, atol=1e-6)
    if mem:
        assert [nd["id"] for nd in net.dynamic] == ["m1", "ms", "ms2", "mem_next", "zm", "logits"]
    else:
        assert not net.dynamic and ms.shape == (2, 3, 0)


def test_checkpoint_and_full_depth_options_preserve_values_and_grads():
    net, _, _ = _net()
    obs, mem0, prev, avail = _window()
    outs, grads = [], []
    for kw in ({}, {"ckpt": 2}, {"loops": {"L": (3, 0)}}, {"loops": {"L": (3, 0)}, "ckpt": 1}):
        net.zero_grad()
        lg, ck, ms = B.fa_forward_window(net, obs, mem0, prev, avail, **kw)
        (lg[torch.isfinite(lg)].square().mean() + ck.square().mean() + ms.square().mean()).backward()
        outs.append(lg.detach())
        grads.append({n: p.grad.clone() for n, p in net.named_parameters()})
    for o, gr in zip(outs[1:], grads[1:]):
        assert torch.equal(torch.isfinite(o), torch.isfinite(outs[0]))
        assert torch.allclose(o[torch.isfinite(o)], outs[0][torch.isfinite(o)], atol=1e-6)
        for n in gr:
            assert torch.allclose(gr[n], grads[0][n], rtol=1e-4, atol=1e-7), n
    # every parameter gets a gradient (no op in the executor blocks autograd)
    assert all(float(g.abs().sum()) > 0 for g in grads[0].values())
    # progressive: shorter loop / no-grad prefix changes the output but still trains the loop weight
    net.zero_grad()
    lg, _, _ = B.fa_forward_window(net, obs, mem0, prev, avail, loops={"L": (2, 1)})
    assert not torch.allclose(lg[torch.isfinite(lg)], outs[0][torch.isfinite(outs[0])])
    lg[torch.isfinite(lg)].square().mean().backward()
    assert net.P("Wh").grad.abs().sum() > 0
    rng = np.random.default_rng(0)
    draws = [B.fa_progressive(rng, {"a": 5, "b": 2}) for _ in range(300)]
    assert 90 < sum(not d for d in draws) < 210
    for d in draws:
        for lid, (n_iter, n0) in d.get("loops", {}).items():
            assert 0 <= n0 < n_iter <= {"a": 5, "b": 2}[lid]
    assert B.fa_progressive(rng, {}) == {}


def test_training_step_decreases_loss():
    net, _, _ = _net(bias="zero")                       # the default scheme
    obs, mem0, prev, avail = _window(W=4, L=4, seed=3)
    y = torch.randint(0, 7, (4, 4), generator=torch.Generator().manual_seed(1))
    cell = torch.where(y == 6, 33 * 64 + 21, -1)
    avail[:, :, 6] = True

    def loss_fn():
        lg, ck, _ = B.fa_forward_window(net, obs, mem0, prev, avail, ckpt=2)
        return B._bc_loss(lg.reshape(-1, 8), ck.reshape(-1, 64, 64), y.reshape(-1), cell.reshape(-1),
                          avail.reshape(-1, 8))

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    l0 = loss_fn()
    opt.zero_grad()
    l0.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    opt.step()
    with torch.no_grad():
        l1 = loss_fn()
    assert float(l1) < float(l0.detach())


def _buffer(net, n=3, cap=10, seed=0):
    buf = B.Buffer.new()
    B.collect(_specs(n, cap), net, B.PlannerTeacher(), 0.5, buf, "cpu", random.Random(seed), sequences=True,
              mem_size=net.M)
    return buf


def _tensors(buf):
    return (torch.as_tensor(np.stack(buf.obs)), torch.as_tensor(np.stack(buf.avail)),
            torch.as_tensor(np.stack(buf.prev)), torch.as_tensor(np.array(buf.act)),
            torch.as_tensor(np.array(buf.cell)))


def _spans(ep):
    ep = np.asarray(ep)
    s = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    return list(zip(s, np.r_[s[1:], len(ep)]))


@pytest.mark.parametrize("max_eps", [1, 2, None])
def test_fa_recompute_identity_equals_stored(max_eps):
    net, _, _ = _net(bias="uniform")
    buf = _buffer(net)
    stored = torch.as_tensor(np.stack(buf.mem))
    assert stored.shape[1] == M and stored.abs().max() > 0.05
    out = B.transformed_episodes(net, *_tensors(buf), buf.ep, None, np.random.default_rng(0), "cpu",
                                 max_episodes=max_eps)
    assert out[5].shape == stored.shape
    if max_eps == 1:                                    # batch 1 per step, as in collection: bitwise
        assert torch.equal(out[5], stored)
    assert torch.allclose(out[5], stored, atol=1e-6, rtol=0)


def _match(orig, new):
    out = []
    for e in G.ELEMENTS:
        g = G.transform_grid(orig, e)
        cmap = {}
        for c in np.unique(g):
            v = np.unique(new[g == c])
            if len(v) != 1:
                break
            cmap[int(c)] = int(v[0])
        else:
            if len(set(cmap.values())) == len(cmap):
                out.append((e, cmap))
    return out


@pytest.mark.parametrize("max_eps", [1, None])
def test_fa_recompute_nonidentity_equals_direct_rollout(max_eps):
    net, _, _ = _net(bias="uniform", seed=1)
    buf = _buffer(net, n=4, seed=1)
    obs, av, prev, act, cell = _tensors(buf)
    o2, av2, p2, y2, c2, m2 = B.transformed_episodes(net, obs, av, prev, act, cell, buf.ep, {"palette", "dihedral"},
                                                     np.random.default_rng(3), "cpu", max_episodes=max_eps)
    stored = np.stack(buf.mem)
    moved = 0
    for s, t in _spans(buf.ep):
        O, X = obs[s:t].numpy().astype(int), o2[s:t].numpy().astype(int)
        ok = False
        for e, cmap in _match(O, X):
            perm = G.action_perm(e)
            xav = np.zeros_like(av[s:t].numpy()); xav[:, perm] = av[s:t].numpy()
            xpv = np.zeros_like(prev[s:t].numpy()); xpv[:, perm] = prev[s:t].numpy()
            xo = G.transform_grid(np.vectorize(cmap.get)(O), e)
            if np.array_equal(xo, X) and np.array_equal(xav, av2[s:t].numpy()) and np.array_equal(xpv, p2[s:t].numpy()):
                ok = True
                break
        assert ok
        m = torch.zeros(1, M)
        with torch.no_grad():
            for k in range(t - s):                      # direct rollout over the transformed prefix
                assert torch.allclose(m2[s + k], m[0], atol=1e-6, rtol=0), (s, k)
                _, _, m = net(torch.as_tensor(xo[k][None]), m, torch.as_tensor(xpv[k][None]),
                              torch.as_tensor(xav[k][None]))
        moved += int(np.abs(m2[s:t].numpy() - stored[s:t]).max() > 1e-4)
    assert moved >= 1                                   # stored memory is not the consistent one


def test_dagger_fa_student_saves_loadable_artifact(tmp_path):
    arch, written = FA.validate(tiny_arch(), tiny_params())
    src = tmp_path / "writer"
    src.mkdir()
    np.savez_compressed(src / "round2.npz", **written)
    (src / "round2.graph.json").write_text(json.dumps(arch))
    out = tmp_path / "d"
    res = B.dagger(_specs(2, cap=8), B.PlannerTeacher(), out, seed=1, iters=1, epochs=2, batch=8,
                   augment={"palette", "dihedral"}, bptt=4, progressive=True, grad_ckpt=2,
                   aug_memory="recompute", student_arch=src / "round2.graph.json")
    assert len(res["log"]) == 2 and all(np.isfinite(e["loss"]) for e in res["log"])
    assert {p.name for p in out.iterdir()} >= {"student.graph.json", "student_fa.npz", "init_fa.npz",
                                               "train_config.json", "dagger_log.json", "dagger_summary.json"}
    assert not (out / "student.npz").exists()
    cfg = json.loads((out / "train_config.json").read_text())
    assert cfg["student_arch"].endswith("round2.graph.json") and cfg["student_bias_init"] == "zero"
    assert cfg["student_count"]["memory_size"] == M and cfg["student_loops"] == {"L": 3}
    arch2, p2 = B.load_fa_student(out)
    assert arch2 == arch and set(p2) == set(written)
    init = dict(np.load(out / "init_fa.npz"))
    assert any(not np.array_equal(p2[k], init[k]) for k in p2)            # trained
    assert all(np.array_equal(init[k], v) for k, v in B.fa_init_params(arch, written, 1).items())
    from wsp.session import Artifact, evaluate                              # the eval_artifacts path
    r = evaluate(Artifact("freearch", "", params=p2, graph=arch2), _specs(2, cap=8), device="cpu")
    assert len(r) == 2 and all(x.error is None for x in r)


def test_checkpointed_loops_do_not_leak_activations():
    """Activations must be freed by reference counting alone: a checkpoint closure that captured the
    whole environment (which later holds the loop outputs) formed a cycle that kept every
    minibatch's activations alive until the cyclic GC ran."""
    import gc
    import weakref
    net, _, _ = _net()
    obs, mem0, prev, avail = _window()
    gc.collect()
    gc.disable()
    try:
        refs = []
        for backward in (True, False):                  # nodes backward never reaches keep their
            lg, ck, ms = B.fa_forward_window(net, obs, mem0, prev, avail, ckpt=1)   # checkpoint state
            if backward:
                lg[torch.isfinite(lg)].sum().backward()
            refs.append(weakref.ref(ck._base))          # the graph's click tensor (downstream of L)
            del lg, ck, ms
        assert all(r() is None for r in refs)
    finally:
        gc.enable()


def _fa_src(tmp_path):
    arch, written = FA.validate(tiny_arch(), tiny_params())
    src = tmp_path / "writer"
    src.mkdir()
    np.savez_compressed(src / "round0.npz", **written)
    (src / "round0.graph.json").write_text(json.dumps(arch))
    return src / "round0.graph.json"


def test_init_probe_recorded_and_nan_detected(tmp_path):
    out = tmp_path / "d"
    B.dagger(_specs(2, cap=4), B.PlannerTeacher(), out, seed=0, iters=0, epochs=1, batch=8, bptt=4,
             aug_memory="recompute", student_arch=_fa_src(tmp_path))
    cfg = json.loads((out / "train_config.json").read_text())
    assert 0.0 <= cfg["init_dead_grad_frac"] < 0.5 and cfg["init_nan"] is False and cfg["init_probe_frames"] == 2
    assert "time_cap_h" not in cfg
    summ = json.loads((out / "dagger_summary.json").read_text())
    assert "truncated" not in summ                                 # no cap: summary unchanged
    net, _, _ = _net()
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1e30)                                          # overflows -> non-finite logits
    pr = B.fa_init_probe(net, _specs(4, cap=4), "cpu")
    assert pr["init_nan"] is True and pr["init_dead_grad_frac"] == 1.0
    assert all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize("fa", [True, False])
@pytest.mark.parametrize("trip,where", [(1, "collect"), (3, "train")])
def test_time_cap_stops_and_saves(tmp_path, monkeypatch, fa, trip, where):
    """Deterministic stop sites: the cap check reports 'time up' on its trip-th call (calls 1-2 are
    the two collection episodes of iteration 0, call 3 the first training epoch)."""
    calls = []
    monkeypatch.setattr(B, "_time_up", lambda deadline: deadline is not None and (calls.append(1) or len(calls) >= trip))
    out = tmp_path / "d"
    res = B.dagger(_specs(2, cap=6), B.PlannerTeacher(), out, seed=0, iters=3, epochs=2, batch=8, bptt=4,
                   aug_memory="recompute", time_cap_h=6.0, **({"student_arch": _fa_src(tmp_path)} if fa else {}))
    summ = json.loads((out / "dagger_summary.json").read_text())
    assert summ["truncated"] is True and summ["stopped_in"] == where and summ["stopped_iter"] == 0
    assert summ["iterations_completed"] == 0 == len(res["log"]) and summ["time_cap_h"] == 6.0
    assert summ["elapsed_h"] >= 0 and json.loads((out / "train_config.json").read_text())["time_cap_h"] == 6.0
    if fa:
        assert (out / "student_fa.npz").exists() and (out / "student.graph.json").exists()
        B.load_fa_student(out)
    else:
        assert (out / "student.npz").exists()
    assert json.loads((out / "dagger_log.json").read_text()) == []


def test_time_cap_wall_clock(tmp_path):
    """A real cap of 2 s on a run that would take much longer: stops, saves, flags truncated."""
    import time
    out = tmp_path / "d"
    t = time.time()
    res = B.dagger(_specs(2, cap=8), B.PlannerTeacher(), out, seed=0, iters=10_000, epochs=3, batch=8, bptt=4,
                   aug_memory="recompute", progressive=True, grad_ckpt=2, time_cap_h=2 / 3600,
                   student_arch=_fa_src(tmp_path))
    wall = time.time() - t
    summ = json.loads((out / "dagger_summary.json").read_text())
    assert summ["truncated"] is True and summ["stopped_in"] in ("collect", "train")
    assert summ["iterations_completed"] == len(res["log"]) < 10_001
    assert 2 / 3600 <= summ["elapsed_h"] and wall < 30
    assert (out / "student_fa.npz").exists() and (out / "student.graph.json").exists()
    arch, p = B.load_fa_student(out)
    assert all(np.isfinite(v).all() for v in p.values())


def test_time_cap_not_hit_marks_complete(tmp_path):
    out = tmp_path / "d"
    res = B.dagger(_specs(2, cap=4), B.PlannerTeacher(), out, seed=0, iters=1, epochs=1, batch=8, bptt=4,
                   aug_memory="recompute", time_cap_h=1.0)
    summ = json.loads((out / "dagger_summary.json").read_text())
    assert summ["truncated"] is False and summ["stopped_in"] is None and summ["iterations_completed"] == 2 == len(res["log"])
    assert (out / "student.npz").exists()

"""Free-architecture language: fixed-actor equivalence, torch vs NumPy reference, validation, budget, sandbox."""
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from wsp import actor as A
from wsp import freearch as FA
from wsp.envs import ACTIONS, avail_mask, new_game, obs_of, run_episodes
from wsp.manifest import RUNS


def _t(*arrays):
    return [torch.as_tensor(a) for a in arrays]


def _frames():
    out = []
    for fam, seed in (("esm", 3), ("ztl", 1), ("ruz", 0), ("mds", 2)):
        g = new_game(fam, seed)
        out.append((obs_of(g), avail_mask(g)))
    rng = np.random.default_rng(0)
    for _ in range(3):
        out.append((rng.integers(0, 16, (64, 64)), (rng.random(8) < 0.7) | (np.arange(8) == 1)))
    return out


def _close(a, b, tol=1e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    assert (np.isfinite(a) == np.isfinite(b)).all()
    scale = max(1.0, float(np.abs(a[fin]).max()) if fin.any() else 1.0)
    assert np.abs(a[fin] - b[fin]).max(initial=0) <= tol * scale, np.abs(a[fin] - b[fin]).max()


# ----------------------------------------------------------------------------- fixed actor
def test_fixed_actor_count():
    arch = FA.fixed_actor_arch()
    c = FA.count(arch, FA.fixed_actor_params(A.zero_params()))
    assert c["params"] == A.N_PARAMS == 57_289
    assert 1.30e9 < c["macs_per_step"] < 1.34e9          # ~1.32 GMAC (convs) + loop adds
    assert c["params"] <= FA.MAX_PARAMS and c["macs_per_step"] <= FA.MAX_MACS
    assert c["node_evals_per_step"] <= FA.MAX_EVALS
    assert {k: v.shape for k, v in FA.fixed_actor_params(A.zero_params()).items()} == FA.fixed_actor_shapes()


def _compare_fixed(p, frames, rng):
    arch = FA.fixed_actor_arch()
    fp = FA.fixed_actor_params(p)
    net, actor = FA.FreeArchNet(arch, fp).eval(), A.Actor(p).eval()
    for obs, av in frames:
        mem = rng.uniform(-1, 1, A.MEM).astype(np.float32)
        ap = np.eye(8, dtype=np.float32)[rng.integers(0, 8)]
        with torch.no_grad():
            l1, c1, m1 = net(*_t(obs[None], mem[None], ap[None], av[None]))
            l2, c2, m2 = actor(*_t(obs[None], mem[None], ap[None], av[None]))
        assert A.decode(l1, c1) == A.decode(l2, c2)
        assert torch.equal(c1, c2)                             # identical computation up to the click head
        _close(l1[0].numpy(), l2[0].numpy())
        _close(m1[0].numpy(), m2[0].numpy(), 1e-5)
        mv1, lg1, mn1 = FA.reference_step(arch, fp, obs, mem, ap, av)
        mv2, lg2, mn2 = A.reference_step(p, obs, mem, ap, av)
        assert mv1 == mv2 == A.decode(l2, c2)[0]
        _close(lg1, lg2)
        _close(mn1, mn2, 1e-5)


def test_fixed_actor_random_params():
    rng = np.random.default_rng(1)
    frames = _frames()
    for s in range(3):
        _compare_fixed(A.random_params(s, scale=0.05), frames, rng)


WRITTEN = sorted(Path(os.environ.get("WSP_RUNS", RUNS)).glob(
    "rev2_private/*/astra-6/weights_feedback_M_full_r0/session.json"))


def _written():
    out = []
    for sj in WRITTEN:
        rec = json.loads(sj.read_text())
        if rec.get("final_round") is not None:
            out.append((sj.parent, rec["family"], A.load_npz(sj.parent / f"round{rec['final_round']}.npz")))
    return out


@pytest.mark.skipif(not WRITTEN, reason="private written artifacts not available")
def test_fixed_actor_written_artifacts():
    """Written astra-6 weights: identical episode traces (S-start, first 12 episodes, <= 25 steps) with
    the Actor and the FA graph, and NumPy reference agreement on real frames."""
    from wsp.queries import load_sets
    from wsp.envs import build_start
    rng = np.random.default_rng(2)
    arts = _written()
    assert len(arts) >= 15
    for d, fam, p in arts:
        specs = [replace(s, cap=min(s.cap, 25)) for s in load_sets(fam, rev=2)["S-start"][:12]]
        r1 = run_episodes(specs, A.NeuralPolicy(A.Actor(p), "cpu"), batch=64)
        pol = FA.FreeArchPolicy(FA.FreeArchNet(FA.fixed_actor_arch(), FA.fixed_actor_params(p)), "cpu")
        r2 = run_episodes(specs, pol, batch=64)
        assert [r.trace for r in r1] == [r.trace for r in r2], d
        assert [r.success for r in r1] == [r.success for r in r2], d
        frames = []                                            # real frames along the Actor's trajectories
        for s, r in list(zip(specs, r1))[:3]:
            g = build_start(s)
            for a, row, col in r.trace[:2]:
                if a in ACTIONS:
                    g.step(a, row, col)
            frames.append((obs_of(g), avail_mask(g)))
        _compare_fixed(p, frames, rng)


# ----------------------------------------------------------------------------- random graphs
def random_net(seed: int, M: int):
    """A random valid network that exercises every op (small channel counts)."""
    rng = np.random.default_rng(seed)
    P, g = {}, []
    cnt = [0]

    def par(shape, scale):
        name = f"p{len(P)}"
        P[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
        return name

    def node(op, ins, **kw):
        cnt[0] += 1
        nid = kw.pop("id", f"n{cnt[0]}")
        g.append({"id": nid, "op": op, "in": ins, **kw})
        return nid

    x = node("concat", ["x0", "coords"])
    c = {}
    c[x] = 19
    a = node("conv", [x], w=par((6, 19, 3, 3), 0.3), b=par((6,), 0.1)); c[a] = 6
    b = node("conv", [x], w=par((6, 19, 5, 5), 0.2)); c[b] = 6
    k7 = node("conv", ["coords"], w=par((2, 3, 7, 7), 0.2), b=par((2,), 0.1)); c[k7] = 2
    u = [node(op, [a]) for op in ("relu", "tanh", "sigmoid", "abs")]
    bins = [node(op, [u[0], b]) for op in ("add", "sub", "mul", "max2", "min2")]
    one = node("slice", [k7], start=0, stop=1)                     # 1-channel broadcast
    bc1 = node("mul", [u[1], one])
    bc2 = node("sub", [one, u[2]])
    ax = [node(op, [u[3]]) for op in ("rowmax", "colmax", "rowsum", "colsum")]
    axs = node("tanh", [node("concat", ax)])
    cat = node("concat", [bins[0], bins[3], bc1, bc2, axs])        # 6*4 + 24 = 48
    f = node("conv", [cat], w=par((8, 48, 1, 1), 0.15), b=par((8,), 0.1))
    v0 = node("gmean", [f])
    v1 = node("gmax", [bins[4]])
    vb = node("dense", ["aprev"], w=par((8, 8), 0.5), b=par((8,), 0.1))
    vec = node("vconcat", [v0, v1, vb, "avail"] + (["mem"] if M else []))
    n_vec = 8 + 6 + 8 + 8 + M
    bcast = node("add", [f, node("vslice", [vec], start=0, stop=8)])   # vector (8,) acts as (8,1,1)
    bscal = node("max2", [f, node("vslice", [vec], start=3, stop=4)])  # length-1 vector broadcast
    sp = node("concat", [bcast, bscal])                                 # 16
    wl = par((16, 16, 3, 3), 0.08)
    ul = par((6, n_vec), 0.2)
    g.append({"id": "L", "op": "loop", "T": 5,
              "state": [{"id": "s", "init": sp}, {"id": "q", "zeros": [6]}],
              "body": [{"id": "lc", "op": "conv", "in": ["s"], "w": wl},
                       {"id": "la", "op": "add", "in": ["lc", sp]},
                       {"id": "ln", "op": "tanh", "in": ["la"]},
                       {"id": "lq", "op": "dense", "in": [vec], "w": ul},
                       {"id": "lq2", "op": "vadd", "in": ["lq", "q"]},
                       {"id": "lq3", "op": "vtanh", "in": ["lq2"]}],
              "next": {"s": "ln", "q": "lq3"}})
    gm = node("gmax", ["L.s"])
    h = node("vconcat", [gm, "L.q", vec])
    z = node("vrelu", [node("dense", [h], w=par((12, 16 + 6 + n_vec), 0.2), b=par((12,), 0.1))])
    zz = node("vmul", [node("vsigmoid", [z]), node("vtanh", [z])])
    node("dense", [zz], w=par((8, 12), 1.0), b=par((8,), 0.3), id="logits")
    node("conv", [node("concat", ["L.s", f])], w=par((1, 24, 1, 1), 0.5), b=par((1,), 0.1), id="click")
    if M:
        node("vtanh", [node("dense", [zz], w=par((M, 12), 0.5))], id="mem_next")
    return {"graph": g, "memory_size": M}, P


def test_random_graphs_torch_vs_numpy():
    frames = _frames()
    rng = np.random.default_rng(3)
    compared = 0
    for seed in range(4):
        M = (0, 5, 17, 64)[seed]
        arch, P = random_net(seed, M)
        used = {op["op"] for op in arch["graph"]} | {op["op"] for nd in arch["graph"] if nd["op"] == "loop"
                                                     for op in nd["body"]}
        assert used == set(FA.OPS)
        net = FA.FreeArchNet(arch, P).eval()
        obs = np.stack([f[0] for f in frames])
        av = np.stack([f[1] for f in frames])
        mem = rng.uniform(-1, 1, (len(frames), M)).astype(np.float32)
        ap = np.eye(8, dtype=np.float32)[rng.integers(0, 8, len(frames))]
        with torch.no_grad():
            lt, ct, mt = net(*_t(obs, mem, ap, av))
        moves = A.decode(lt, ct)
        for j in range(len(frames)):
            mv, lg, mn, env = FA.reference_step(arch, P, obs[j], mem[j], ap[j], av[j], return_env=True)
            _close(lt[j].numpy(), lg)
            _close(ct[j].numpy(), env["click"][0])
            _close(mt[j].numpy(), mn, 1e-5)
            fin = np.sort(lg[np.isfinite(lg)])
            cl = np.sort(env["click"][0].ravel())
            if (len(fin) < 2 or fin[-1] - fin[-2] > 1e-3) and (mv[0] != "ACTION6" or cl[-1] - cl[-2] > 1e-3):
                assert moves[j] == mv
                compared += 1
    assert compared >= 20


# ----------------------------------------------------------------------------- budget and validation
def _tiny(M=0, extra=None, params=None):
    g = [{"id": "v", "op": "dense", "in": ["avail"], "w": "W", "b": "b"},
         {"id": "logits", "op": "vadd", "in": ["v", "avail"]},
         {"id": "click", "op": "slice", "in": ["coords"], "start": 0, "stop": 1}]
    p = {"W": np.zeros((8, 8)), "b": np.eye(8)[1]}
    if M:
        g.append({"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wm"})
        p["Wm"] = np.zeros((M, M))
    g += extra or []
    p.update(params or {})
    return {"graph": g, "memory_size": M}, p


def test_budget_enforced():
    arch, p = _tiny(extra=[{"id": "q", "op": "conv", "in": ["x0"], "w": "K"}],
                    params={"K": np.zeros((365, 16, 5, 5))})
    FA.validate(arch, p)                                       # 146,072 parameters
    p["K"] = np.zeros((375, 16, 5, 5))
    assert FA.count(arch, p)["params"] == 150_000 + 64 + 8
    with pytest.raises(FA.FAError, match="parameters > budget"):
        FA.validate(arch, p)
    # MACs: a 16-channel 3x3 recurrent conv for 256 steps = 2.4e9 MACs
    arch, p = _tiny(extra=[{"id": "s", "op": "conv", "in": ["x0"], "w": "Ws"},
                           {"id": "L", "op": "loop", "T": 256, "state": [{"id": "h", "init": "s"}],
                            "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh"}], "next": {"h": "hc"}}],
                    params={"Ws": np.zeros((16, 16, 1, 1)), "Wh": np.zeros((16, 16, 3, 3))})
    c = FA.count(arch, p)
    assert c["macs_per_step"] == 16 * 16 * 4096 + 256 * 16 * 16 * 9 * 4096 + 64 + 8   # + dense, vadd
    with pytest.raises(FA.FAError, match="MACs per step > budget"):
        FA.validate(arch, p)
    arch["graph"][-1]["T"] = 128
    FA.validate(arch, p)


BAD = [
    (lambda a, p: a["graph"].append({"id": "q", "op": "softmax", "in": ["v"]}), "unknown op"),
    (lambda a, p: a["graph"].insert(0, {"id": "q", "op": "vrelu", "in": ["v"]}), "not defined before"),
    (lambda a, p: a["graph"].append({"id": "v", "op": "vrelu", "in": ["avail"]}), "already used"),
    (lambda a, p: a["graph"].append({"id": "x0", "op": "relu", "in": ["coords"]}), "already used"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "relu", "in": ["v"]}), "needs spatial"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "vrelu", "in": ["v"], "k": 3}), "unknown key"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "conv", "in": ["x0"], "w": "W"}), "needs (O, 16, k, k)"),
    (lambda a, p: (a["graph"].append({"id": "q", "op": "conv", "in": ["x0"], "w": "K"}),
                   p.update(K=np.zeros((2, 16, 4, 4)))), "k in"),
    (lambda a, p: (a["graph"].append({"id": "q", "op": "conv", "in": ["x0"], "w": "K", "b": "b"}),
                   p.update(K=np.zeros((2, 16, 3, 3)))), "bias 'b'"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "dense", "in": ["v"], "w": "nope"}), "not in params"),
    (lambda a, p: p.update(extra=np.zeros(3)), "unused parameter"),
    (lambda a, p: p.update(W=np.full((8, 8), np.nan)), "non-finite"),
    (lambda a, p: p.update(W=np.full((8, 8), 1e39)), "non-finite"),
    (lambda a, p: a["graph"].pop(1), "output 'logits'"),
    (lambda a, p: a["graph"].pop(2), "output 'click'"),
    (lambda a, p: a["graph"].__setitem__(2, {"id": "click", "op": "slice", "in": ["coords"], "start": 0,
                                            "stop": 2}), "must be spatial (1,64,64)"),
    (lambda a, p: a["graph"].append({"id": "mem_next", "op": "vrelu", "in": ["v"]}), "memory_size is 0"),
    (lambda a, p: a.__setitem__("memory_size", 300), "memory_size must be"),
    (lambda a, p: a.__setitem__("memory_size", 4), "output 'mem_next'"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "add", "in": ["x0", "coords"]}), "cannot broadcast"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "vadd", "in": ["v", "q0"]}), "not defined"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "add", "in": ["v", "avail"]}), "at least one input"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "slice", "in": ["x0"], "start": 3, "stop": 20}), "start < stop"),
    (lambda a, p: a["graph"].append({"id": "q.r", "op": "relu", "in": ["x0"]}), "must match"),
    (lambda a, p: a["graph"].append({"id": "L", "op": "loop", "T": 0, "state": [{"id": "h", "zeros": [8]}],
                                     "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}}),
     "T must be"),
    (lambda a, p: a["graph"].append({"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                     "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {}}),
     "'next' must map"),
    (lambda a, p: a["graph"].append({"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                     "body": [{"id": "hb", "op": "vslice", "in": ["h"], "start": 0, "stop": 2}],
                                     "next": {"h": "hb"}}), "but state"),
    (lambda a, p: a["graph"].append({"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                     "body": [{"id": "L2", "op": "loop", "T": 2, "state": [{"id": "h2", "zeros": [8]}],
                                               "body": [{"id": "hb", "op": "vrelu", "in": ["h2"]}],
                                               "next": {"h2": "hb"}}], "next": {"h": "h"}}), "cannot be nested"),
    (lambda a, p: a["graph"].extend([{"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                      "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}},
                                     {"id": "q", "op": "vrelu", "in": ["L"]}]), "is a loop"),
    (lambda a, p: a["graph"].extend([{"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                      "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}},
                                     {"id": "q", "op": "vrelu", "in": ["hb"]}]), "not defined"),
    (lambda a, p: a["graph"].append({"id": "L", "op": "loop", "T": 200, "state": [{"id": "h", "zeros": [8]}],
                                     "body": [{"id": f"b{i}", "op": "vrelu", "in": ["h"]} for i in range(60)],
                                     "next": {"h": "b0"}}), "node evaluations"),
    (lambda a, p: a["graph"].append({"id": "q", "op": "concat", "in": ["x0"] * 40}), "channels"),
    (lambda a, p: a["graph"].extend([{"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                      "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}},
                                     {"id": "L2", "op": "loop", "T": 2, "state": [{"id": "h2", "init": "L"}],
                                      "body": [{"id": "hb2", "op": "vrelu", "in": ["h2"]}], "next": {"h2": "hb2"}}]),
     "not a tensor"),
    (lambda a, p: a["graph"].extend([{"id": "L", "op": "loop", "T": 2, "state": [{"id": "h", "zeros": [8]}],
                                      "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}},
                                     {"id": "L2", "op": "loop", "T": 2, "state": [{"id": "h2", "zeros": [8]}],
                                      "body": [{"id": "hb2", "op": "vrelu", "in": ["h2"]}], "next": {"h2": "L"}}]),
     "not a tensor"),
    (lambda a, p: p.update(file=np.zeros(1)), "must match"),
]


def test_valid_vector_loop_and_count():
    arch, p = _tiny(M=4, extra=[{"id": "L", "op": "loop", "T": 3, "state": [{"id": "h", "init": "v"}],
                                 "body": [{"id": "hb", "op": "vrelu", "in": ["h"]}], "next": {"h": "hb"}},
                                {"id": "q", "op": "vrelu", "in": ["L.h"]}])
    FA.validate(arch, p)
    c = FA.count(arch, p)
    assert c["node_evals_per_step"] == 3 + 1 + 3 + 1 and c["memory_size"] == 4 and c["params"] == 64 + 8 + 16
    assert c["macs_per_step"] == 64 + 8 + 16


@pytest.mark.parametrize("i", range(len(BAD)))
def test_validation_errors(i):
    arch, p = _tiny()
    FA.validate(arch, p)
    mutate, msg = BAD[i]
    mutate(arch, p)
    with pytest.raises(FA.FAError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        FA.validate(arch, p)


# ----------------------------------------------------------------------------- sandbox, artifacts, policy
TRIVIAL = '''
import numpy as np

def build():
    graph = [
        {"id": "v", "op": "dense", "in": ["avail"], "w": "W", "b": "b"},   # constant preference
        {"id": "logits", "op": "vadd", "in": ["v", "avail"]},
        {"id": "click", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wm", "b": "bm"},
    ]
    b = np.zeros(8)
    b[1] = 1.0                                   # ACTION1
    return {"graph": graph, "memory_size": np.int64(3),
            "params": {"W": np.zeros((8, 8)), "b": b, "Wm": np.eye(3), "bm": np.ones(3)}}
'''


def test_sandbox_build_and_policy(tmp_path):
    from wsp.queries import reviewed_start
    from wsp.session import Artifact, evaluate, load_final, materialize
    art = materialize("freearch", TRIVIAL)
    assert art.error is None, art.error
    assert art.condition == "freearch" and art.graph["memory_size"] == 3
    assert art.audit["count"]["params"] == 64 + 8 + 9 + 3
    FA.save_artifact(art, tmp_path, 0)
    (tmp_path / "round0.py").write_text(TRIVIAL)
    (tmp_path / "session.json").write_text(json.dumps({"final_round": 0}))
    art2 = load_final(tmp_path, "freearch")
    assert art2.graph == art.graph and all(np.array_equal(art2.params[k], art.params[k]) for k in art.params)
    spec = reviewed_start("toy_reach", 154, 0, "t")        # toy target straight above the player
    res = evaluate(art2, [spec], device="cpu")
    assert res[0].success and res[0].actions == spec.ref_len and res[0].trace[0][0] == "ACTION1"
    pol = FA.policy(art2, "cpu")
    pol.begin([0])
    pol.act([0], np.zeros((1, 64, 64), np.int64), np.ones((1, 8), bool))
    assert torch.allclose(pol.mem[0], torch.ones(3))                 # mem_next = Wm mem + bm, passed through
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32


def test_sandbox_rejections():
    from wsp.session import materialize
    cases = {
        "import os\ndef build():\n    return {}": "audit: import os",
        "def build():\n    return {'graph': []}": "exactly the keys",
        "import numpy as np\ndef build():\n    return {'graph': [], 'memory_size': 0, 'params': {}}": "output 'logits'",
        TRIVIAL.replace("np.zeros((8, 8))", "np.zeros((8, 7))"): "needs (out, 8)",
        TRIVIAL.replace('"b": b,', '"b": b, "huge": np.zeros(200000),'): "unused parameter",
        TRIVIAL.replace("np.eye(3)", "np.full((3, 3), np.inf)"): "non-finite",
        TRIVIAL.replace('"start": 0', '"start": object()'): "unsupported type",
    }
    for src, msg in cases.items():
        art = materialize("freearch", src)
        assert art.params is None and art.error and msg in art.error, (msg, art.error)

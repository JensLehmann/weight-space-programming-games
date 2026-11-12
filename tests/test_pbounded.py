"""Bounded program player (pbounded): static rules, runtime guards and budget, equivalence with an FA network,
artifact round trip and evaluation, prompt invariants."""
import json

import numpy as np
import pytest
import torch

from wsp import _pb_runtime as RT
from wsp import freearch as FA
from wsp import pbounded as PB
from wsp.envs import ACTIONS, avail_mask, new_game, obs_of

HEAD = "import numpy as np\n\nMEMORY_SIZE = 2\nT = 8\nW = np.ones((8, 16))\n\n\n"
RET = "return np.zeros(8), np.zeros((64, 64)), memory"


def player(body, extra=""):
    return HEAD + extra + "def act(obs, available, memory, prev_action):\n" + "".join(
        "    " + line + "\n" for line in body.split("\n"))


# ----------------------------------------------------------------------------- static rules
def test_accepts_prompt_example():
    from wsp.prompts_pb import EXAMPLE
    spec = PB.analyse(EXAMPLE)
    assert spec["runtime"] == ["act", "neighbours"] and spec["memory_size"] == 8
    assert spec["globals"] == ["COLOURS", "MIX", "PREFER", "STEPS"] and spec["loops"] == [["act", 25, 32]]
    r = PB.run_trusted(EXAMPLE)
    c = r["count"]
    assert c["counts_equal"] and not PB.budget_errors(c)
    assert c["ops_per_call"] == 339 and c["constants"] == 16 + 128 + 128 + 1 + spec["literal_numbers"]


ACCEPTED = {
    "helpers, constants, methods": player(
        "x = (obs == 3).astype(np.float32)\nx += 1.0\ny = shift(x)[0:32, ::2].T\n"
        "z = np.where(y > 0.5, y, -y).sum(axis=0, keepdims=True)\n"
        "v = np.concatenate([z.ravel()[0:8], memory])\ns = W @ np.tanh(np.ones(16))\n"
        "return s + v[0:8] * 0.0, x - np.mean(x), np.clip(memory, -1.0, 1.0)",
        "def shift(a, k=1):\n    return np.roll(np.pad(a, 1)[1:65, 1:65], 1, axis=1)\n\n\n"),
    "numpy imported in a construction helper": HEAD + "def kernel():\n    import numpy\n    return numpy.ones(3)\n\n\n"
    "K3 = kernel()\n\n\ndef act(obs, available, memory, prev_action):\n    x = K3[0:2]\n    " + RET + "\n",
    "loop with module-level count, tuple targets": player(
        "h = obs * 0.0\nfor _ in range(T):\n    h = np.maximum(h, step(h))\na, b = h[0:2]\n"
        "return W @ np.ones(16) + a.mean(), h + b[None, :], memory",
        "def step(h):\n    return np.roll(h, -1, axis=0) * 0.5\n\n\n"),
}

REJECTED = {
    "if": (player("if obs.sum() > 3:\n    pass\n" + RET), "`if` is not allowed"),
    "while": (player("while True:\n    pass\n" + RET), "`while` is not allowed"),
    "nested loops": (player("for _ in range(4):\n    for _ in range(4):\n        pass\n" + RET),
                     "loops cannot be nested"),
    "nested through a call": (player("for _ in range(4):\n    y = h(obs)\n" + RET,
                                     "def h(x):\n    for _ in range(3):\n        x = x + 1\n    return x\n\n\n"),
                              "calls h(), which contains a loop"),
    "loop count 257": (player("for _ in range(257):\n    pass\n" + RET), "loop count 257 is outside 1..256"),
    "loop count from a local": (player("k = 3\nfor _ in range(k):\n    pass\n" + RET), "loops must be"),
    "loop variable used": (player("s = 0\nfor i in range(3):\n    s = s + i\n" + RET), "loop variable i"),
    ".item()": (player("v = obs.sum().item()\n" + RET), ".item() is not allowed"),
    ".tolist()": (player("v = obs.tolist()\n" + RET), ".tolist() is not allowed"),
    "indexing with data": (player("v = W[obs]\n" + RET), "indices and slice bounds must be constants"),
    "indexing with a local integer": (player("k = 2\nv = W[k]\n" + RET), "indices and slice bounds"),
    "import math": ("import math\n" + player(RET), "import math"),
    "import os": ("import os\n" + player(RET), "import os"),
    "import inside player code": (player("import numpy\n" + RET), "an import is not allowed in player code"),
    "from numpy import": ("from numpy import zeros\n" + player(RET), "only `import numpy as np`"),
    "comprehension": (player("v = [x for x in range(3)]\n" + RET), "comprehension"),
    "generator": (player("v = np.stack(list(x for x in range(3)))\n" + RET), "generator"),
    "and/or": (player("v = (obs > 0) and (obs < 3)\n" + RET), "`and` / `or`"),
    "not": (player("v = not obs\n" + RET), "`not`"),
    "conditional expression": (player("v = 1 if obs else 2\n" + RET), "conditional expression"),
    "try": (player("try:\n    v = 1\nexcept Exception:\n    v = 2\n" + RET), "`try`"),
    "recursion": (player("v = f(obs)\n" + RET, "def f(x):\n    return g(x)\n\n\ndef g(x):\n    return f(x)\n\n\n"),
                  "recursion"),
    "int()": (player("v = int(obs.sum())\n" + RET), "builtin int()"),
    "float()": (player("v = float(obs.sum())\n" + RET), "builtin float()"),
    "bool()": (player("v = bool(obs.sum())\n" + RET), "builtin bool()"),
    "len()": (player("v = len(obs)\n" + RET), "builtin len()"),
    "np.argmax": (player("v = np.argmax(obs)\n" + RET), "np.argmax() is not available"),
    "np.nonzero": (player("v = np.nonzero(obs)\n" + RET), "np.nonzero() is not available"),
    "one-argument np.where": (player("v = np.where(obs)\n" + RET), "np.where needs exactly three"),
    "assignment into an array": (player("v = np.zeros(8)\nv[0] = 1\n" + RET), "assignment into arrays"),
    ".shape": (player("v = obs.shape\n" + RET), "attribute .shape"),
    "in": (player("v = 3 in obs\n" + RET), "`in` and `is`"),
    "chained comparison": (player("v = 0 < obs < 3\n" + RET), "chained comparisons"),
    "lambda": (player("f = lambda x: x\n" + RET), "`lambda`"),
    "global": (player("global W\n" + RET), "`global`"),
    "print": (player("print(obs)\n" + RET), "builtin print()"),
    "early return": (player("return 1\n" + RET), "`return` is allowed only as the last"),
    "missing MEMORY_SIZE": ("import numpy as np\n\n\ndef act(obs, available, memory, prev_action):\n    " + RET,
                            "define MEMORY_SIZE"),
    "MEMORY_SIZE too large": (player(RET).replace("MEMORY_SIZE = 2", "MEMORY_SIZE = 257"), "MEMORY_SIZE must be"),
    "wrong act signature": (HEAD + "def act(obs, available, memory):\n    " + RET, "exactly four positional"),
}


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_static_accepts(name):
    PB.analyse(ACCEPTED[name])


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_static_rejects(name):
    src, msg = REJECTED[name]
    with pytest.raises(PB.PBError) as exc:
        PB.analyse(src)
    assert msg in str(exc.value), str(exc.value)


# ----------------------------------------------------------------------------- materialisation (sandbox)
def test_materialize_accepts_and_counts():
    from wsp.session import materialize
    for name, src in ACCEPTED.items():
        art = materialize("pbounded", src)
        assert art.error is None, (name, art.error)
        c = art.audit["count"]
        assert c["counts_equal"] and c["ops_per_call"] > 0 and c["memory_size"] == 2
    art = materialize("pbounded", ACCEPTED["loop with module-level count, tuple targets"])
    # obs * 0.0; 8 x (np.roll, * 0.5, np.maximum); h[0:2]; unpacking reads 2 rows; return line: 6 operations
    assert art.audit["count"]["ops_per_call"] == 1 + 8 * 3 + 1 + 2 + 6


def _loop_ops(n_iter, n_ops):
    body = "x = obs * 1.0\nfor _ in range(%d):\n" % n_iter + "".join(f"    x = x + {i}.5\n" for i in range(n_ops))
    return player(body + RET)


MATERIALIZE_REJECTED = {
    "operations over budget": (_loop_ops(256, 40), "budget: 10,243 operations per call > limit 10,000"),
    "operations within budget": (_loop_ops(240, 40), None),
    "element work over budget": (player("x = np.zeros((512, 64, 64))\nfor _ in range(256):\n    x = x + 1.0\n"
                                        "    x = x * 0.5\n    x = x - 0.25\n" + RET),
                                 "element operations per call > budget"),
    "array too large": (player("x = np.zeros((1024, 64, 64))\n" + RET), "4,194,304 entries > limit 2,097,152"),
    "constants over budget": (HEAD + "BIG = np.ones(150_001)\n\n\ndef act(obs, available, memory, prev_action):\n"
                              "    x = BIG[0:1]\n    " + RET, "constants > budget 150,000"),
    "non-finite constant": (HEAD + "BAD = np.array([1e39])\n\n\ndef act(obs, available, memory, prev_action):\n"
                            "    x = BAD[0:1]\n    " + RET, "non-finite values after the float32 cast"),
    "data-dependent shift": (player("x = np.roll(obs, obs.sum())\n" + RET), "shift must be a Python constant"),
    "data-dependent shape": (player("x = np.zeros(obs.max(axis=0))\n" + RET), "shape must be a Python constant"),
    "@ of two input-dependent arrays": (player("v = obs.mean(axis=0)\nx = v @ v\n" + RET),
                                        "at least one operand must be computed only from constants"),
    "wrong output shape": (player("return np.zeros(7), np.zeros((64, 64)), memory"), "scores must be"),
    "module-level object": (HEAD + "R = np.random.default_rng(0)\n\n\ndef act(obs, available, memory, "
                            "prev_action):\n    x = R\n    " + RET, "of type Generator"),
    "act replaced": (HEAD + "def act(obs, available, memory, prev_action):\n    " + RET + "\n\n\nact = 3\n",
                     "act was replaced"),
    "error in module-level code": (HEAD + "Q = 1 / 0\n\n\ndef act(obs, available, memory, prev_action):\n    " + RET,
                                   "ZeroDivisionError"),
}


@pytest.mark.parametrize("name", sorted(MATERIALIZE_REJECTED))
def test_materialize_runtime_rules(name):
    from wsp.session import materialize
    src, msg = MATERIALIZE_REJECTED[name]
    art = materialize("pbounded", src)
    if msg is None:
        assert art.error is None, art.error
        assert art.audit["count"]["ops_per_call"] == 1 + 240 * 40 + 2
    else:
        assert art.params is None and art.error and msg in art.error, art.error


def test_runtime_guards_in_process():
    RT.install(PB.CAPS)
    x = RT._input(np.arange(6.0).reshape(2, 3), np.float32)
    w = RT._const(np.ones((3, 3)))
    RT.counter().active = True
    try:
        for f in (lambda: bool(x), lambda: int(x[0, 0]), lambda: float(x[0, 0]), lambda: range(x[0, 0]),
                  lambda: [0, 1][x[0, 0]], lambda: x[x > 1], lambda: x.item(), lambda: x.argmax()):
            with pytest.raises(RT.PBRuntimeError):
                f()
        with pytest.raises(RT.PBRuntimeError):
            x.__setitem__(0, 1.0)
        with pytest.raises(RT.PBRuntimeError, match="in-place"):
            np.add(x, 1.0, out=x)
        with pytest.raises(RT.PBRuntimeError, match="not available"):
            RT.NP.argmax
        y = x @ w                                        # constant weight: allowed
        assert isinstance(y, RT.BArray) and y._data and y.dtype == np.float32
        with pytest.raises(RT.PBRuntimeError, match="only from constants"):
            x @ (x.T * 1.0)
        assert not RT.NP.zeros_like(x)._data             # values that do not depend on the inputs
        assert str(x) == repr(x) == "%s" % (x,) == "<array shape=(2, 3) dtype=float32>"   # no values as strings
        n0 = RT.counter().ops
        _ = (x + 1.0)[0] * 2.0
        assert RT.counter().ops == n0 + 3
        with pytest.raises(RT.PBRuntimeError, match="in-place"):
            w += 1.0
        with pytest.raises(ValueError, match="read-only"):
            w.view(np.ndarray)[0, 0] = 2.0               # constants are read-only
    finally:
        RT.counter().active = False


# ----------------------------------------------------------------------------- equivalence with an FA network
FA_TINY = {"memory_size": 4, "graph": [
    {"id": "e", "op": "conv", "in": ["x0"], "w": "We", "b": "be"},                       # 1x1, 16 -> 4
    {"id": "L", "op": "loop", "T": 6, "state": [{"id": "h", "init": "e"}],
     "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh", "b": "bh"},             # 3x3, 4 -> 4
              {"id": "hn", "op": "relu", "in": ["hc"]}],
     "next": {"h": "hn"}},
    {"id": "rm", "op": "rowmax", "in": ["L.h"]},
    {"id": "g1", "op": "gmax", "in": ["L.h"]},
    {"id": "g2", "op": "gmean", "in": ["rm"]},
    {"id": "v", "op": "vconcat", "in": ["g1", "g2", "mem", "aprev", "avail"]},          # 28
    {"id": "logits", "op": "dense", "in": ["v"], "w": "Wo", "b": "bo"},
    {"id": "m", "op": "dense", "in": ["v"], "w": "Wm"},
    {"id": "mem_next", "op": "vtanh", "in": ["m"]},
    {"id": "cs", "op": "concat", "in": ["L.h", "coords"]},                               # 7
    {"id": "click", "op": "conv", "in": ["cs"], "w": "Wc", "b": "bc"},                   # 1x1, 7 -> 1
]}

PARAMS_SRC = '''rng = np.random.default_rng(5)
We = rng.normal(0, 1, (4, 16, 1, 1))
be = rng.normal(0, 0.1, 4)
Wh = rng.normal(0, 0.4, (4, 4, 3, 3))
bh = rng.normal(0, 0.1, 4)
Wo = rng.normal(0, 1, (8, 28))
bo = rng.normal(0, 0.1, 8) + np.eye(8)[6] * 1.5
Wm = rng.normal(0, 0.5, (4, 28))
Wc = rng.normal(0, 1, (1, 7, 1, 1))
bc = np.zeros(1)
'''

PB_TINY = "import numpy as np\n\nMEMORY_SIZE = 4\nT = 6\n" + PARAMS_SRC + '''
E1 = We[:, :, 0, 0]                                    # 1x1 kernel as a matrix (4, 16)
WH = Wh.transpose(0, 2, 3, 1).reshape(4, 36)           # 3x3 kernel over windows ordered (row, col, channel)
WC = Wc[:, :, 0, 0]
BE = be.reshape(4, 1, 1)
BH = bh.reshape(4, 1, 1)
COLOURS = np.arange(16).reshape(16, 1, 1)
COORDS = np.stack([np.repeat(np.arange(64)[:, None] / 63, 64, 1), np.repeat(np.arange(64)[None] / 63, 64, 0),
                   np.ones((64, 64))])


def conv3(h):
    p = np.pad(h, ((0, 0), (1, 1), (1, 1)))
    win = np.concatenate([p[:, 0:64, 0:64], p[:, 0:64, 1:65], p[:, 0:64, 2:66],
                          p[:, 1:65, 0:64], p[:, 1:65, 1:65], p[:, 1:65, 2:66],
                          p[:, 2:66, 0:64], p[:, 2:66, 1:65], p[:, 2:66, 2:66]])
    return np.einsum("ok,khw->ohw", WH, win) + BH


def act(obs, available, memory, prev_action):
    x0 = (obs[None] == COLOURS).astype(np.float32)
    h = np.einsum("oc,chw->ohw", E1, x0) + BE
    for _ in range(T):
        h = np.maximum(conv3(h), 0.0)
    rm = h.max(axis=2, keepdims=True)
    v = np.concatenate([h.max(axis=(1, 2)), rm.mean(axis=(1, 2)), memory, prev_action, available])
    scores = Wo @ v + bo
    memory_next = np.tanh(Wm @ v)
    click = np.einsum("oc,chw->ohw", WC, np.concatenate([h, COORDS]))[0] + bc[0]
    return scores, click, memory_next
'''


def _fa_params():
    ns = {"np": np}
    exec(PARAMS_SRC, ns)
    return {k: ns[k] for k in ("We", "be", "Wh", "bh", "Wo", "bo", "Wm", "Wc", "bc")}


def _frames():
    out = []
    for seed in (3, 1, 0, 2):                            # instrumentation game toy_reach, then random frames
        g = new_game("toy_reach", seed)
        out.append((obs_of(g), avail_mask(g)))
    rng = np.random.default_rng(0)
    for _ in range(3):
        out.append((rng.integers(0, 16, (64, 64)), (rng.random(8) < 0.7) | (np.arange(8) == 1)))
    return out


def _onehot(move):
    a = np.zeros(8, np.float32)
    if move[0] in ACTIONS:
        a[ACTIONS.index(move[0])] = 1.0
    return a


def test_equivalent_to_fa_network():
    from wsp.session import materialize
    params = _fa_params()
    arch, p32 = FA.validate(FA_TINY, params)
    net = FA.FreeArchNet(arch, p32).eval()
    frames = _frames()
    # FA: NumPy reference and the torch executor, memory and previous action chained over the frames
    ref, mem, ap = [], np.zeros(4, np.float32), np.zeros(8, np.float32)
    tmem, tap = torch.zeros(1, 4), torch.zeros(1, 8)
    for obs, av in frames:
        move, logits, mem_next = FA.reference_step(arch, p32, obs, mem, ap, av)
        with torch.no_grad():
            tl, tc, tm = net(torch.as_tensor(obs[None]), tmem, tap, torch.as_tensor(av[None]))
        tmove = FA.A.decode(tl, tc)[0]
        assert tmove == move
        ref.append((move, logits, mem_next))
        mem, ap = mem_next.astype(np.float32), _onehot(move)
        tmem, tap = tm, torch.as_tensor(_onehot(tmove)[None])
    assert {m[0][0] for m in ref} >= {"ACTION6"}                           # the click rule is exercised
    # the bounded player, in process: same scores and memory
    act = PB.run_trusted(PB_TINY)["act"]
    mem, ap = np.zeros(4, np.float32), np.zeros(8, np.float32)
    for (obs, av), (move, logits, mem_ref) in zip(frames, ref):
        mv, scores, click, mem_next, ap_next = RT.step(act, 4, obs, av.astype(np.float32), mem, ap)
        assert mv == move
        fin = np.isfinite(logits)
        assert np.allclose(np.where(av, scores, -np.inf)[fin], logits[fin], rtol=1e-4, atol=1e-4)
        assert np.allclose(mem_next, mem_ref, atol=1e-5)
        mem, ap = mem_next, ap_next
    # the bounded player through the sandbox and the evaluation policy: identical decisions
    art = materialize("pbounded", PB_TINY)
    assert art.error is None, art.error
    pol = PB.policy(art)
    try:
        pol.begin([0])
        for (obs, av), (move, _, _) in zip(frames, ref):
            assert pol.act([0], obs[None], av[None])[0] == move
    finally:
        pol.close()


# ----------------------------------------------------------------------------- artifacts and evaluation
def test_artifact_round_trip_and_evaluation(tmp_path):
    from wsp.queries import reviewed_start
    from wsp.session import evaluate, load_final, materialize
    src = player("s = np.zeros(8) + np.eye(8)[1]\nreturn s, np.zeros((64, 64)), memory + 1.0")
    art = materialize("pbounded", src)
    assert art.error is None, art.error
    PB.save_artifact(art, tmp_path, 0)
    (tmp_path / "round0.py").write_text(src)
    (tmp_path / "session.json").write_text(json.dumps({"final_round": 0}))
    art2 = load_final(tmp_path, "pbounded")
    assert art2.graph == art.graph and all(np.array_equal(art2.params[k], art.params[k]) for k in art.params)
    spec = reviewed_start("toy_reach", 154, 0, "t")        # toy target straight above the player
    res = evaluate(art2, [spec])
    assert res[0].success and res[0].actions == spec.ref_len and res[0].trace[0][0] == "ACTION1"
    pol = PB.policy(art2)
    try:
        pol.begin([0])
        for k in range(3):
            pol._send(0, np.zeros((64, 64), np.int64), np.ones(8), outputs=True)
            out = pol._recv(0)
        assert out["memory_l2"] == pytest.approx(np.sqrt(2) * 3, abs=1e-3)   # memory_next fed back as computed
        assert out["scores"][1] == 1.0 and out["move"] == ["ACTION1", None, None]
    finally:
        pol.close()


# ----------------------------------------------------------------------------- prompt
def test_prompt_shares_fa_sections():
    from wsp.prompts import PYTHON_TASK, build_prompt_v2
    from wsp.prompts_fa import FA_SPEC, FA_TASK, build_prompt_fa
    from wsp.prompts_pb import PB_SPEC, PB_TASK, build_prompt_pb, feedback_message_pb
    obs = [("template 0, level 1, start", np.zeros((64, 64), int))]
    pb, fa = build_prompt_pb("## The game\n\nsection", obs), build_prompt_fa("## The game\n\nsection", obs)
    head_pb, head_fa = pb[: -len(PB_SPEC + "\n\n" + PB_TASK)], fa[: -len(FA_SPEC + "\n\n" + FA_TASK)]
    assert head_pb.split("\n", 1)[1] == head_fa.split("\n", 1)[1]                 # all but the title line
    assert head_pb.split("\n", 1)[0] == build_prompt_v2("x", "python", obs).split("\n", 1)[0]
    assert PYTHON_TASK not in pb and "network" not in PB_TASK
    for word in ("control", "free-architecture", "FA "):
        assert word not in PB_SPEC + PB_TASK
    fb = feedback_message_pb({"solved": 0, "n": 1, "rows": [{"i": 0, "seed": 0, "level": 0, "start": "fresh",
                                                             "result": "not solved", "actions": 3}]},
                             [{"i": 0, "why": "ran out of actions", "trace": ["ACTION1"], "step": 0,
                               "frame": np.zeros((64, 64), int), "internals": {"scores": [1.0]}}], 0, 3)
    assert "Player outputs at that frame" in fb and "Revise your act() program." in fb

"""Free-architecture (FA) actors: a static tensor-op graph language chosen by the writer.

A network ("arch") is JSON: {"graph": [node, ...], "memory_size": M}; its weights are a dict
name -> float32 array. Per step the graph reads the inputs x0 (16x64x64 one-hot colours), coords
(3x64x64: row/63, col/63, 1), mem (M), aprev (8, previous action one-hot), avail (8, 0/1) and
must define the outputs logits (8), click (1x64x64) and mem_next (M; omitted when M = 0).
Decoding is wsp.actor.decode (argmax; unavailable actions masked to -inf; non-finite available
logits -> invalid action, as in wsp.actor.NeuralPolicy).

`np_op` / `run_graph` / `reference_step` are the semantic definition (NumPy float32; shown to
the writer); `FreeArchNet` is the batched exact-FP32 torch executor and `FreeArchPolicy` mirrors
`wsp.actor.NeuralPolicy`. `validate` / `count` check references, shapes, parameters and budget.
Artifacts are stored as round<k>.npz (params) + round<k>.graph.json (arch).

Implementation choices:
  1 structural limits that bound evaluation time/memory: <= 1,000 nodes, <= 10,000 node evaluations per
    step (loop bodies count T times), <= 512 channels per spatial tensor and <= 4,096 summed over all
    spatial tensors, vectors <= 4,096; loops cannot be nested.
  2 biases are optional ("b" omitted = no bias); loop results are addressed as "<loop id>.<state id>";
    node and parameter ids are [A-Za-z_][A-Za-z0-9_]* (unique; "file"/"allow_pickle" reserved for npz);
    nodes must be listed in evaluation order (definition before use gives acyclicity); memory_size may be 0.
  3 broadcasting (vector (C,) as (C,1,1); 1-channel / length-1 operands) applies to all of
    add/sub/mul/max2/min2 (vadd/vmul: length-1 broadcast).
  4 MACs: conv O*C*k*k*4096, dense m*n, add/sub/mul/max2/min2 per output element, axial and global
    reductions per input element, vadd/vmul per element, other ops 0 (fixed actor: 1,331,914,880).
  5 the fixed actor in the language is computed with the same torch kernels as wsp.actor.Actor except the
    memory pre-activation, summed (Wm m + Wmz z) + (Wma a + bm) instead of ((Wm m + Wmz z) + Wma a) + bm
    (float32 rounding differences ~1e-7; decisions identical on all tests).
  6 the sandbox runs build() through _sandbox_runner_fa.py, which reuses the guarded namespace of
    _sandbox_runner.py; hard limit 120 s as for the weights build() (the prompt states 60 s).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import actor as A
from .envs import ACTIONS, CLICK

# --------------------------------------------------------------------------- budget and limits
MAX_PARAMS = 150_000
MAX_MACS = 1.5e9
MAX_MEM = 256
MAX_T = 256
MAX_NODES = 1_000            # all nodes incl. loop bodies
MAX_EVALS = 10_000           # node evaluations per step (a loop body counts T times)
MAX_CH = 512                 # channels of one spatial tensor
MAX_TOTAL_CH = 4_096         # channels summed over all spatial tensors (bodies counted once)
MAX_VEC = 4_096              # entries of one vector
KERNELS = (1, 3, 5, 7)
HW = 64 * 64

INPUTS = {"x0": ("s", 16), "coords": ("s", 3), "aprev": ("v", 8), "avail": ("v", 8)}  # + mem ("v", M)
OUTPUTS = ("logits", "click", "mem_next")
SPATIAL_UNARY = ("relu", "tanh", "sigmoid", "abs")
VECTOR_UNARY = ("vrelu", "vtanh", "vsigmoid")
SPATIAL_BINARY = ("add", "sub", "mul", "max2", "min2")
VECTOR_BINARY = ("vadd", "vmul")
AXIAL = ("rowmax", "colmax", "rowsum", "colsum")
POOL = ("gmax", "gmean")
OPS = (("conv", "dense", "concat", "vconcat", "slice", "vslice", "loop") + SPATIAL_UNARY + VECTOR_UNARY
       + SPATIAL_BINARY + VECTOR_BINARY + AXIAL + POOL)
_EXTRA_KEYS = {"conv": {"w", "b"}, "dense": {"w", "b"}, "slice": {"start", "stop"},
               "vslice": {"start", "stop"}, "loop": {"T", "state", "body", "next"}}
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
RESERVED_PARAM_NAMES = ("file", "allow_pickle")      # keyword names of np.savez_compressed


class FAError(ValueError):
    """Invalid network; the message is returned to the writer."""


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _fmt(shape) -> str:
    return f"spatial ({shape[1]},64,64)" if shape[0] == "s" else f"vector ({shape[1]},)"


# --------------------------------------------------------------------------- validation and counting
class _Ctx:
    def __init__(self, params):
        self.params = params
        self.used: set[str] = set()
        self.ids: set[str] = set()
        self.nodes = 0
        self.evals = 0
        self.macs = 0
        self.channels = 16 + 3          # input planes
        self.shapes: dict[str, tuple] = {}


def _param(ctx, nd, key, where, required=True):
    if key not in nd:
        if required:
            raise FAError(f"{where}: missing parameter reference '{key}'")
        return None
    name = nd[key]
    if not isinstance(name, str) or name not in ctx.params:
        raise FAError(f"{where}: parameter {name!r} ('{key}') is not in params")
    ctx.used.add(name)
    return ctx.params[name]


def _infer(ctx, nd, op, ins, where):
    """Output shape of a non-loop node and its MACs (one evaluation)."""
    n = len(ins)

    def need(k, kinds):                 # k inputs (None: at least one), all of kind "s" or "v"
        if (n < 1) if k is None else (n != k):
            raise FAError(f"{where}: needs {'at least 1' if k is None else k} input(s), got {n}")
        for i, s in enumerate(ins):
            if s[0] not in kinds:
                raise FAError(f"{where}: input {i} is {_fmt(s)}; {op} needs "
                              + ("spatial" if kinds == "s" else "vector") + " inputs")

    if op == "conv":
        need(1, "s")
        w = _param(ctx, nd, "w", where)
        c = ins[0][1]
        if w.ndim != 4 or w.shape[1] != c or w.shape[2] != w.shape[3] or w.shape[2] not in KERNELS:
            raise FAError(f"{where}: weight '{nd['w']}' has shape {w.shape}; needs (O, {c}, k, k) with k in "
                          f"{KERNELS} for an input with {c} channels")
        o = w.shape[0]
        b = _param(ctx, nd, "b", where, required=False)
        if b is not None and b.shape != (o,):
            raise FAError(f"{where}: bias '{nd['b']}' has shape {b.shape}; needs ({o},)")
        return ("s", o), o * c * w.shape[2] * w.shape[3] * HW
    if op == "dense":
        need(1, "v")
        w = _param(ctx, nd, "w", where)
        m = ins[0][1]
        if w.ndim != 2 or w.shape[1] != m:
            raise FAError(f"{where}: weight '{nd['w']}' has shape {w.shape}; needs (out, {m}) for an input of "
                          f"length {m}")
        b = _param(ctx, nd, "b", where, required=False)
        if b is not None and b.shape != (w.shape[0],):
            raise FAError(f"{where}: bias '{nd['b']}' has shape {b.shape}; needs ({w.shape[0]},)")
        return ("v", w.shape[0]), w.shape[0] * m
    if op in SPATIAL_UNARY:
        need(1, "s")
        return ins[0], 0
    if op in VECTOR_UNARY:
        need(1, "v")
        return ins[0], 0
    if op in SPATIAL_BINARY:
        if n != 2:
            raise FAError(f"{where}: needs 2 inputs, got {n}")
        if ins[0][0] != "s" and ins[1][0] != "s":
            raise FAError(f"{where}: at least one input must be spatial (use vadd/vmul for vectors)")
        ca, cb = ins[0][1], ins[1][1]
        if ca != cb and 1 not in (ca, cb):
            raise FAError(f"{where}: cannot broadcast {_fmt(ins[0])} with {_fmt(ins[1])} (channel counts must "
                          "match or one must be 1)")
        return ("s", max(ca, cb)), max(ca, cb) * HW
    if op in VECTOR_BINARY:
        need(2, "v")
        ca, cb = ins[0][1], ins[1][1]
        if ca != cb and 1 not in (ca, cb):
            raise FAError(f"{where}: cannot broadcast vectors of length {ca} and {cb}")
        return ("v", max(ca, cb)), max(ca, cb)
    if op in ("concat", "vconcat"):
        need(None, "s" if op == "concat" else "v")
        return (ins[0][0], sum(s[1] for s in ins)), 0
    if op in ("slice", "vslice"):
        need(1, "s" if op == "slice" else "v")
        a, b, c = nd.get("start"), nd.get("stop"), ins[0][1]
        if not (_int(a) and _int(b) and 0 <= a < b <= c):
            raise FAError(f"{where}: needs integers 0 <= start < stop <= {c}, got start={a!r} stop={b!r}")
        return (ins[0][0], b - a), 0
    if op in AXIAL:
        need(1, "s")
        return ins[0], ins[0][1] * HW
    if op in POOL:
        need(1, "s")
        return ("v", ins[0][1]), ins[0][1] * HW
    raise FAError(f"{where}: unknown op {op!r}")


def _check_size(ctx, shape, where):
    if shape[0] == "s":
        if shape[1] > MAX_CH:
            raise FAError(f"{where}: {shape[1]} channels > limit {MAX_CH}")
        ctx.channels += shape[1]
        if ctx.channels > MAX_TOTAL_CH:
            raise FAError(f"{where}: spatial channels summed over the graph exceed {MAX_TOTAL_CH}")
    elif shape[1] > MAX_VEC:
        raise FAError(f"{where}: vector length {shape[1]} > limit {MAX_VEC}")


def _new_id(ctx, nid, where):
    if not isinstance(nid, str) or not _ID.fullmatch(nid):
        raise FAError(f"{where}: id {nid!r} must match [A-Za-z_][A-Za-z0-9_]* (at most 64 characters)")
    if nid in ctx.ids or nid in INPUTS or nid == "mem":
        raise FAError(f"{where}: id {nid!r} is already used")
    ctx.ids.add(nid)


def _check_nodes(ctx, nodes, scope, mult, prefix, in_loop):
    if not isinstance(nodes, list):
        raise FAError(f"{prefix or 'graph'}: must be a list of nodes")
    for idx, nd in enumerate(nodes):
        where = f"{prefix}node {idx}"
        if not isinstance(nd, dict):
            raise FAError(f"{where}: must be a dict")
        nid, op = nd.get("id"), nd.get("op")
        where = f"{prefix}node {idx} {nid!r} ({op})"
        if op not in OPS:
            raise FAError(f"{where}: unknown op {op!r}; allowed: {', '.join(OPS)}")
        allowed = {"id", "op", "comment"} | _EXTRA_KEYS.get(op, set()) | ({"in"} if op != "loop" else set())
        extra = sorted(set(nd) - allowed)
        if extra:
            raise FAError(f"{where}: unknown key(s) {extra}; {op} takes {sorted(allowed - {'comment'})}")
        _new_id(ctx, nid, where)
        ctx.nodes += 1
        if ctx.nodes > MAX_NODES:
            raise FAError(f"more than {MAX_NODES} nodes")
        if op == "loop":
            if in_loop:
                raise FAError(f"{where}: loops cannot be nested")
            T = nd.get("T")
            if not _int(T) or not 1 <= T <= MAX_T:
                raise FAError(f"{where}: T must be an integer in 1..{MAX_T}, got {T!r}")
            states = nd.get("state")
            if not isinstance(states, list) or not states:
                raise FAError(f"{where}: 'state' must be a non-empty list of {{'id', 'init' | 'zeros'}}")
            inner = dict(scope)
            sids = []
            for j, st in enumerate(states):
                sw = f"{where} state {j}"
                if not isinstance(st, dict) or set(st) - {"id", "init", "zeros", "comment"} \
                        or ("init" in st) == ("zeros" in st):
                    raise FAError(f"{sw}: must be {{'id': name, 'init': node id}} or {{'id': name, 'zeros': shape}}")
                _new_id(ctx, st.get("id"), sw)
                if "init" in st:
                    src = st["init"]
                    if not isinstance(src, str) or src not in scope or scope[src] is None:
                        raise FAError(f"{sw}: init {src!r} is not a tensor defined before the loop")
                    shape = scope[src]
                else:
                    z = st["zeros"]
                    if isinstance(z, list) and len(z) == 3 and all(_int(v) for v in z) and z[1:] == [64, 64] \
                            and z[0] >= 1:
                        shape = ("s", z[0])
                    elif isinstance(z, list) and len(z) == 1 and _int(z[0]) and z[0] >= 1:
                        shape = ("v", z[0])
                    else:
                        raise FAError(f"{sw}: zeros must be [C, 64, 64] or [n], got {z!r}")
                _check_size(ctx, shape, sw)
                inner[st["id"]] = shape
                sids.append(st["id"])
            body = nd.get("body")
            if not isinstance(body, list) or not body:
                raise FAError(f"{where}: 'body' must be a non-empty list of nodes")
            _check_nodes(ctx, body, inner, mult * T, f"{nid}.body ", True)
            nxt = nd.get("next")
            if not isinstance(nxt, dict) or set(nxt) != set(sids):
                raise FAError(f"{where}: 'next' must map every state id {sids} to a body node id")
            for sid, src in nxt.items():
                if not isinstance(src, str) or src not in inner or inner[src] is None:
                    raise FAError(f"{where}: next[{sid!r}] = {src!r} is not a tensor defined in the loop")
                if inner[src] != inner[sid]:
                    raise FAError(f"{where}: next[{sid!r}] = {src!r} is {_fmt(inner[src])} but state {sid!r} is "
                                  f"{_fmt(inner[sid])}")
            for sid in sids:
                scope[f"{nid}.{sid}"] = inner[sid]
            scope[nid] = None                                 # the loop itself is not a tensor
            continue
        ins = nd.get("in")
        if not isinstance(ins, list) or not all(isinstance(i, str) for i in ins):
            raise FAError(f"{where}: 'in' must be a list of ids")
        shapes = []
        for i in ins:
            if i not in scope:
                raise FAError(f"{where}: input {i!r} is not defined before this node (nodes are evaluated in "
                              "list order; cycles are not allowed)")
            if scope[i] is None:
                raise FAError(f"{where}: {i!r} is a loop; use '{i}.<state id>' for its final state")
            shapes.append(scope[i])
        shape, macs = _infer(ctx, nd, op, shapes, where)
        _check_size(ctx, shape, where)
        ctx.macs += macs * mult
        ctx.evals += mult
        scope[nid] = shape
    return scope


def _float32_params(params) -> dict[str, np.ndarray]:
    if not isinstance(params, dict):
        raise FAError("params must be a dict name -> array")
    out = {}
    for k, v in params.items():
        if not isinstance(k, str) or not _ID.fullmatch(k) or k in RESERVED_PARAM_NAMES:
            raise FAError(f"parameter name {k!r} must match [A-Za-z_][A-Za-z0-9_]* (at most 64 characters) "
                          f"and not be one of {RESERVED_PARAM_NAMES}")
        try:
            with np.errstate(over="ignore"):
                a = np.asarray(v, dtype=np.float64).astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise FAError(f"parameter {k!r}: not a numeric array ({exc})") from None
        if not np.isfinite(a).all():
            raise FAError(f"parameter {k!r}: non-finite values after float32 cast")
        out[k] = a
    return out


def _normalise(arch) -> dict:
    if not isinstance(arch, dict) or set(arch) != {"graph", "memory_size"}:
        raise FAError("the network must be {'graph': [nodes], 'memory_size': M}")
    try:
        arch = json.loads(json.dumps(arch, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise FAError(f"graph is not JSON-serialisable: {exc}") from None
    M = arch["memory_size"]
    if not _int(M) or not 0 <= M <= MAX_MEM:
        raise FAError(f"memory_size must be an integer in 0..{MAX_MEM}, got {M!r}")
    return arch


def analyse(arch, params) -> tuple[dict, dict, dict]:
    """Check everything except the budget. Returns (arch as plain JSON, float32 params, info)."""
    arch = _normalise(arch)
    p32 = _float32_params(params)
    ctx = _Ctx(p32)
    M = arch["memory_size"]
    scope = dict(INPUTS)
    if M:
        scope["mem"] = ("v", M)
    _check_nodes(ctx, arch["graph"], scope, 1, "", False)
    want = {"logits": ("v", 8), "click": ("s", 1), "mem_next": ("v", M)}
    for name in OUTPUTS:
        if name == "mem_next" and M == 0:
            if "mem_next" in scope:
                raise FAError("memory_size is 0 but the graph defines mem_next")
            continue
        if name not in scope or name not in [nd.get("id") for nd in arch["graph"]]:
            raise FAError(f"output {name!r} must be a top-level node ({_fmt(want[name])})")
        if scope[name] != want[name]:
            raise FAError(f"output {name!r} is {_fmt(scope[name])} but must be {_fmt(want[name])}")
    unused = sorted(set(p32) - ctx.used)
    if unused:
        raise FAError(f"unused parameter(s) {unused[:10]}{' ...' if len(unused) > 10 else ''}")
    if ctx.evals > MAX_EVALS:
        raise FAError(f"{ctx.evals} node evaluations per step > limit {MAX_EVALS} (a loop body counts T times)")
    top = {}                                                   # graph order, loop states in place
    for nd in arch["graph"]:
        if nd["op"] == "loop":
            top.update({f"{nd['id']}.{st['id']}": scope[f"{nd['id']}.{st['id']}"] for st in nd["state"]})
        else:
            top[nd["id"]] = scope[nd["id"]]
    info = {"params": int(sum(a.size for a in p32.values())), "macs_per_step": int(ctx.macs),
            "nodes": ctx.nodes, "node_evals_per_step": ctx.evals, "spatial_channels": ctx.channels,
            "memory_size": M, "shapes": top}
    return arch, p32, info


def count(arch, params) -> dict:
    """{params, macs_per_step, nodes, node_evals_per_step, spatial_channels, memory_size}."""
    info = analyse(arch, params)[2]
    return {k: v for k, v in info.items() if k != "shapes"}


def budget_errors(info: dict) -> list[str]:
    errs = []
    if info["params"] > MAX_PARAMS:
        errs.append(f"{info['params']:,} parameters > budget {MAX_PARAMS:,}")
    if info["macs_per_step"] > MAX_MACS:
        errs.append(f"{info['macs_per_step']:,} MACs per step > budget {int(MAX_MACS):,}")
    return errs


def validate(arch, params) -> tuple[dict, dict[str, np.ndarray]]:
    """Return (arch as plain JSON, float32 params); raise FAError (with a writer-facing message)."""
    arch, p32, info = analyse(arch, params)
    errs = budget_errors(info)
    if errs:
        raise FAError("budget: " + "; ".join(errs))
    return arch, p32


def cast_report(params: dict) -> dict:
    rep = {}
    for k, v in params.items():
        a = np.asarray(v, dtype=np.float64)
        d = np.abs(a - a.astype(np.float32).astype(np.float64))
        rep[k] = {"changed": int((d > 0).sum()), "max_abs": float(d.max()) if d.size else 0.0}
    return rep


# --------------------------------------------------------------------------- NumPy reference (authoritative)
def np_op(nd, xs, P):
    """One node on one example. Spatial tensors are (C,64,64), vectors (n,); P = params (float32)."""
    op, x = nd["op"], xs[0]
    lift = lambda a: a[:, None, None] if a.ndim == 1 else a      # a vector (C,) acts as (C,1,1)
    if op == "conv":                                             # cross-correlation, zero padding k//2
        w = P[nd["w"]]
        k = w.shape[-1]
        if k == 1:
            y = np.einsum("oc,chw->ohw", w[:, :, 0, 0], x)
        else:
            xp = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
            win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))  # C,64,64,k,k
            y = np.einsum("chwij,ocij->ohw", win, w, optimize=True)
        return y + P[nd["b"]][:, None, None] if "b" in nd else y
    if op == "dense":
        y = P[nd["w"]] @ x
        return y + P[nd["b"]] if "b" in nd else y
    if op in ("relu", "vrelu"):
        return np.maximum(x, 0)
    if op in ("tanh", "vtanh"):
        return np.tanh(x)
    if op in ("sigmoid", "vsigmoid"):
        return 1 / (1 + np.exp(-x))
    if op == "abs":
        return np.abs(x)
    if op in ("add", "sub", "mul", "max2", "min2"):             # broadcasting; result (C,64,64)
        f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "max2": np.maximum, "min2": np.minimum}[op]
        return f(lift(x), lift(xs[1]))
    if op == "vadd":
        return x + xs[1]
    if op == "vmul":
        return x * xs[1]
    if op in ("concat", "vconcat"):
        return np.concatenate(xs, 0)
    if op in ("slice", "vslice"):
        return x[nd["start"]:nd["stop"]]
    if op == "rowmax":                                           # (c,i,j) -> max_j' x[c,i,j']
        return np.broadcast_to(x.max(2, keepdims=True), x.shape)
    if op == "colmax":                                           # (c,i,j) -> max_i' x[c,i',j]
        return np.broadcast_to(x.max(1, keepdims=True), x.shape)
    if op == "rowsum":
        return np.broadcast_to(x.sum(2, keepdims=True), x.shape)
    if op == "colsum":
        return np.broadcast_to(x.sum(1, keepdims=True), x.shape)
    if op == "gmax":
        return x.max((1, 2))
    if op == "gmean":
        return x.mean((1, 2))
    raise ValueError(op)


def run_graph(nodes, env, P):
    """Evaluate nodes in list order into env (id -> array)."""
    for nd in nodes:
        if nd["op"] != "loop":
            env[nd["id"]] = np_op(nd, [env[i] for i in nd["in"]], P)
            continue
        state = {s["id"]: env[s["init"]] if "init" in s else np.zeros(s["zeros"], np.float32)
                 for s in nd["state"]}
        for _ in range(nd["T"]):                                 # same parameters every iteration
            local = {**env, **state}
            run_graph(nd["body"], local, P)
            state = {sid: local[src] for sid, src in nd["next"].items()}   # all states update together
        for sid, v in state.items():
            env[nd["id"] + "." + sid] = v


def reference_step(arch, params, obs, mem, aprev, avail, return_env=False):
    """Exact semantics of one step (float32). Returns (action, row, col), masked logits, mem_next."""
    P = {k: np.asarray(v, np.float32) for k, v in params.items()}
    x0 = np.zeros((16, 64, 64), np.float32)
    x0[obs, np.arange(64)[:, None], np.arange(64)[None, :]] = 1.0
    coords = np.zeros((3, 64, 64), np.float32)
    coords[0] = np.arange(64)[:, None] / 63.0
    coords[1] = np.arange(64)[None, :] / 63.0
    coords[2] = 1.0
    env = {"x0": x0, "coords": coords, "mem": np.asarray(mem, np.float32),
           "aprev": np.asarray(aprev, np.float32), "avail": np.asarray(avail, np.float32)}
    with np.errstate(all="ignore"):
        run_graph(arch["graph"], env, P)
    logits = np.where(avail, env["logits"], -np.inf)
    mem_next = env["mem_next"] if arch["memory_size"] else np.zeros(0, np.float32)
    a = int(np.argmax(logits))                                   # first maximal index
    if not np.isfinite(logits[np.asarray(avail, bool)]).all():
        move = ("NONFINITE", None, None)                         # counted as an invalid action
    elif a == CLICK:
        cell = int(np.argmax(env["click"][0]))                   # first cell in row-major order
        move = (ACTIONS[a], cell // 64, cell % 64)
    else:
        move = (ACTIONS[a], None, None)
    return (move, logits, mem_next, env) if return_env else (move, logits, mem_next)


# --------------------------------------------------------------------------- torch executor
def exact_fp32() -> None:
    """Exact FP32 (TF32 convolutions/matmuls flip argmax decisions of high-gain written nets)."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _lift(t):
    return t[:, :, None, None] if t.dim() == 2 else t


class FreeArchNet(torch.nn.Module):
    """Batched executor; forward(obs (B,64,64) long, mem (B,M), aprev (B,8), avail (B,8) bool)
    -> (logits masked to -inf, click (B,64,64), mem_next (B,M)), like wsp.actor.Actor."""

    def __init__(self, arch: dict, params: dict):
        super().__init__()
        self.arch, p, self.info = analyse(arch, params)
        errs = budget_errors(self.info)
        if errs:
            raise FAError("budget: " + "; ".join(errs))
        self.M = self.arch["memory_size"]
        self.graph = self.arch["graph"]
        for k, v in p.items():
            self.register_buffer("p_" + k, torch.from_numpy(v.copy()))
        coords = torch.stack(torch.meshgrid(torch.arange(64.0) / 63, torch.arange(64.0) / 63, indexing="ij"))
        self.register_buffer("coords", torch.cat([coords, torch.ones(1, 64, 64)]), persistent=False)

    def P(self, name):
        return self._buffers["p_" + name]

    def _op(self, nd, xs):
        op, x = nd["op"], xs[0]
        if op == "conv":
            w = self.P(nd["w"])
            b = self.P(nd["b"]) if "b" in nd else None
            if w.shape[-1] == 1:                       # 1x1: same einsum as wsp.actor.Actor
                y = torch.einsum("oc,bchw->bohw", w[:, :, 0, 0], x)
                return y if b is None else y + b[None, :, None, None]
            return F.conv2d(x, w, b, padding=w.shape[-1] // 2)
        if op == "dense":
            y = x @ self.P(nd["w"]).T
            return y + self.P(nd["b"]) if "b" in nd else y
        if op in ("relu", "vrelu"):
            return F.relu(x)
        if op in ("tanh", "vtanh"):
            return torch.tanh(x)
        if op in ("sigmoid", "vsigmoid"):
            return torch.sigmoid(x)
        if op == "abs":
            return torch.abs(x)
        if op in SPATIAL_BINARY:
            a, b = _lift(x), _lift(xs[1])
            return {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "max2": torch.maximum,
                    "min2": torch.minimum}[op](a, b)
        if op == "vadd":
            return x + xs[1]
        if op == "vmul":
            return x * xs[1]
        if op in ("concat", "vconcat"):
            return torch.cat(xs, 1)
        if op in ("slice", "vslice"):
            return x[:, nd["start"]:nd["stop"]]
        if op == "rowmax":
            return x.amax(dim=3, keepdim=True).expand_as(x)
        if op == "colmax":
            return x.amax(dim=2, keepdim=True).expand_as(x)
        if op == "rowsum":
            return x.sum(dim=3, keepdim=True).expand_as(x)
        if op == "colsum":
            return x.sum(dim=2, keepdim=True).expand_as(x)
        if op == "gmax":
            return x.amax((2, 3))
        if op == "gmean":
            return x.mean((2, 3))
        raise ValueError(op)

    def _run(self, nodes, env, B, dev):
        for nd in nodes:
            if nd["op"] != "loop":
                env[nd["id"]] = self._op(nd, [env[i] for i in nd["in"]])
                continue
            state = {}
            for s in nd["state"]:
                state[s["id"]] = env[s["init"]] if "init" in s else torch.zeros(B, *s["zeros"], device=dev)
            for _ in range(nd["T"]):
                local = {**env, **state}
                self._run(nd["body"], local, B, dev)
                state = {sid: local[src] for sid, src in nd["next"].items()}
            for sid, v in state.items():
                env[nd["id"] + "." + sid] = v

    def run(self, obs, mem, aprev, avail) -> dict:
        """All node values (batched) for one step."""
        exact_fp32()
        B, dev = obs.shape[0], obs.device
        env = {"x0": F.one_hot(obs.long(), 16).permute(0, 3, 1, 2).float(),
               "coords": self.coords.expand(B, -1, -1, -1), "aprev": aprev, "avail": avail.float()}
        if self.M:
            env["mem"] = mem
        self._run(self.graph, env, B, dev)
        return env

    def forward(self, obs, mem, aprev, avail):
        env = self.run(obs, mem, aprev, avail)
        logits = env["logits"].masked_fill(~avail, float("-inf"))
        mem_next = env["mem_next"] if self.M else mem[:, :0]
        return logits, env["click"][:, 0], mem_next


class FreeArchPolicy(A.NeuralPolicy):
    """NeuralPolicy (same batching, decoding and non-finite rule) with an M-dim memory."""

    def __init__(self, net: FreeArchNet, device: str = "cpu"):
        super().__init__(net, device)

    def begin(self, slots):
        for i in slots:
            self.mem[i] = torch.zeros(self.actor.M, device=self.device)
            self.prev[i] = None


# --------------------------------------------------------------------------- the fixed actor in the language
def fixed_actor_arch(T=128):
    """The study's fixed actor (wsp.actor, 57,289 parameters) written in this language."""
    return {"memory_size": 64, "graph": [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},              # 19 channels
        {"id": "c1", "op": "conv", "in": ["x"], "w": "W1", "b": "b1"},    # 5x5 -> 32
        {"id": "f1", "op": "relu", "in": ["c1"]},
        {"id": "c2", "op": "conv", "in": ["f1"], "w": "W2", "b": "b2"},   # 3x3 -> 32
        {"id": "f2", "op": "relu", "in": ["c2"]},
        {"id": "rm", "op": "rowmax", "in": ["f2"]},
        {"id": "cm", "op": "colmax", "in": ["f2"]},
        {"id": "f2a", "op": "concat", "in": ["f2", "rm", "cm"]},          # 96
        {"id": "c3", "op": "conv", "in": ["f2a"], "w": "W3", "b": "b3"},  # 1x1 -> 32
        {"id": "f3", "op": "relu", "in": ["c3"]},
        {"id": "drive", "op": "conv", "in": ["f3"], "w": "Uh", "b": "bh"},  # 1x1 -> 16
        {"id": "H", "op": "loop", "T": T,                                 # H = relu(conv3x3(H) + drive)
         "state": [{"id": "h", "zeros": [16, 64, 64]}],
         "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh"},     # 3x3, no bias
                  {"id": "hs", "op": "add", "in": ["hc", "drive"]},
                  {"id": "hn", "op": "relu", "in": ["hs"]}],
         "next": {"h": "hn"}},
        {"id": "g1", "op": "gmax", "in": ["H.h"]},
        {"id": "g2", "op": "gmean", "in": ["H.h"]},
        {"id": "g3", "op": "gmax", "in": ["f3"]},
        {"id": "g4", "op": "gmean", "in": ["f3"]},
        {"id": "g", "op": "vconcat", "in": ["g1", "g2", "g3", "g4"]},     # 96
        {"id": "zl", "op": "dense", "in": ["g"], "w": "Wz", "b": "bz"},
        {"id": "z", "op": "vrelu", "in": ["zl"]},                         # 128
        {"id": "m1", "op": "dense", "in": ["mem"], "w": "Wm"},
        {"id": "m2", "op": "dense", "in": ["z"], "w": "Wmz"},
        {"id": "m3", "op": "dense", "in": ["aprev"], "w": "Wma", "b": "bm"},
        {"id": "m12", "op": "vadd", "in": ["m1", "m2"]},
        {"id": "m123", "op": "vadd", "in": ["m12", "m3"]},
        {"id": "mem_next", "op": "vtanh", "in": ["m123"]},                # 64
        {"id": "zm", "op": "vconcat", "in": ["z", "mem_next"]},           # 192
        {"id": "logits", "op": "dense", "in": ["zm"], "w": "Wo", "b": "bo"},
        {"id": "fh", "op": "concat", "in": ["f3", "H.h"]},                # 48
        {"id": "click", "op": "conv", "in": ["fh"], "w": "Wc", "b": "bc"},  # 1x1 -> 1
    ]}


FIXED_1X1 = ("W3", "Uh", "Wc")      # stored (out, in) in wsp.actor, (out, in, 1, 1) here


def fixed_actor_params(p: dict) -> dict[str, np.ndarray]:
    """wsp.actor parameters -> parameters of fixed_actor_arch() (same numbers)."""
    p = A.validate(p)
    return {k: (v[:, :, None, None] if k in FIXED_1X1 else v).copy() for k, v in p.items()}


def fixed_actor_shapes() -> dict[str, tuple]:
    return {k: (s + (1, 1) if k in FIXED_1X1 else s) for k, s in A.SCHEMA.items()}


# --------------------------------------------------------------------------- sandboxed build() and artifacts
RUNNER_FA = Path(__file__).with_name("_sandbox_runner_fa.py")


def run_fa_initializer(source: str, timeout: int = 120):
    """Run the writer's build() in the sandbox (same audit, allowlist, environment and limits as the
    weights build()). Returns (arch | None, params | None, error | None, audit, wall_s)."""
    from .sandbox import _env, _limits, audit
    au = audit(source, "build")
    if not au.ok:
        return None, None, "audit: " + "; ".join(au.violations[:10]), au, 0.0
    with tempfile.TemporaryDirectory(prefix="wsp_fa_") as tmp:
        src = Path(tmp) / "writer_source.py"
        src.write_text(source)
        out_npz, out_json = Path(tmp) / "out.npz", Path(tmp) / "out.json"
        work = Path(tmp) / "work"
        work.mkdir()
        t = time.time()
        cmd = [sys.executable, "-I", str(RUNNER_FA), str(src), str(out_npz), str(out_json)]
        try:
            proc = subprocess.run(cmd, cwd=work, env=_env(str(work)), capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits(timeout))
        except subprocess.TimeoutExpired:
            return None, None, f"timeout after {timeout}s", au, time.time() - t
        wall = time.time() - t
        if proc.returncode != 0 or not out_npz.exists() or not out_json.exists():
            return None, None, "runtime: " + proc.stderr.strip()[-1500:], au, wall
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        arch = json.loads(out_json.read_text())
        with np.load(out_npz) as z:                         # transport keys are "p_" + name
            params = {k[2:]: z[k] for k in z.files}
        if info.get("cpu_s", 0) > 30:
            au.flags.append(f"initializer used {info['cpu_s']:.0f}s CPU")
        return arch, params, None, au, wall


# Language controls: None = free architecture; "fixed_graph" = the fixed actor's exact graph
# (only weights free); "fixed_ops" = any graph using only the fixed actor's operations.
CONTROL = None
FIXED_OPS = frozenset({"add", "colmax", "concat", "conv", "dense", "gmax", "gmean", "loop", "relu", "rowmax",
                       "vadd", "vconcat", "vrelu", "vtanh"})


def _ops(nodes):
    out = set()
    for n in nodes:
        out.add(n.get("op"))
        if n.get("op") == "loop":
            out |= _ops(n.get("body", []))
    return out


def control_errors(arch32: dict) -> list[str]:
    if CONTROL == "fixed_graph":
        ref = _normalise(fixed_actor_arch())
        if (json.dumps(arch32["graph"], sort_keys=True) != json.dumps(ref["graph"], sort_keys=True)
                or arch32.get("memory_size") != ref.get("memory_size")):
            return ["this run requires exactly the fixed actor graph shown in the prompt (same nodes, ids, "
                    "operations, loop count and memory size); only parameter values may differ"]
    if CONTROL == "fixed_ops":
        bad = sorted(_ops(arch32["graph"]) - FIXED_OPS)
        if bad:
            return [f"this run allows only the fixed actor's operations {sorted(FIXED_OPS)}; not allowed: {bad}"]
    return []


def materialize(code: str):
    """Writer program -> session.Artifact (condition 'freearch')."""
    from .session import Artifact
    arch, params, err, au, wall = run_fa_initializer(code)
    if err:
        return Artifact("freearch", code, error=err, audit=au.to_json(), init_wall_s=wall)
    aj = au.to_json()
    try:
        arch32, p32, info = analyse(arch, params)
        aj["count"] = {k: v for k, v in info.items() if k != "shapes"}
        errs = budget_errors(info)
        if errs:
            raise FAError("budget: " + "; ".join(errs))
        cerrs = control_errors(arch32)
        if cerrs:
            raise FAError("; ".join(cerrs))
    except FAError as exc:
        return Artifact("freearch", code, error=f"invalid network: {exc}", audit=aj, init_wall_s=wall)
    except Exception as exc:          # validator bug or malformed structure: still an invalid artifact
        return Artifact("freearch", code, error=f"invalid network: {type(exc).__name__}: {exc}"[:1500],
                        audit=aj, init_wall_s=wall)
    aj["cast"] = cast_report(params)
    return Artifact("freearch", code, params=p32, audit=aj, init_wall_s=wall, graph=arch32)


def save_artifact(art, out_dir: Path, k: int) -> None:
    np.savez_compressed(out_dir / f"round{k}.npz", **art.params)
    (out_dir / f"round{k}.graph.json").write_text(json.dumps(art.graph))


def load_net(out_dir: Path, k: int) -> tuple[dict, dict]:
    arch = json.loads((Path(out_dir) / f"round{k}.graph.json").read_text())
    with np.load(Path(out_dir) / f"round{k}.npz") as z:
        params = {n: z[n] for n in z.files}
    return validate(arch, params)


def load_artifact(out_dir: Path, k: int, code: str = ""):
    from .session import Artifact
    arch, params = load_net(out_dir, k)
    return Artifact("freearch", code, params=params, graph=arch)


def policy(art, device: str) -> FreeArchPolicy:
    return FreeArchPolicy(FreeArchNet(art.graph, art.params), device)


# --------------------------------------------------------------------------- feedback internals
def _max_at(t) -> list:
    idx = int(torch.argmax(t))
    return [round(float(t.max()), 3), idx // 64, idx % 64]


def probe_internals(art, spec, trace, step, n_nodes: int = 3, max_ch: int = 32) -> dict | None:
    """Logits, per-channel spatial maxima [max, row, col] of the last spatial nodes (excluding concat/slice)
    and of the click map, and the memory norm at decision `step` of an episode (replayed with the network, CPU)."""
    from .envs import avail_mask, build_start, obs_of
    net = FreeArchNet(art.graph, art.params).eval()
    ops = {nd["id"]: nd["op"] for nd in net.graph}
    spatial = [k for k, s in net.info["shapes"].items()        # skip copies (concat/slice) and the click head
               if s[0] == "s" and k != "click" and ops.get(k) not in ("concat", "slice")]
    mem = torch.zeros(1, net.M)
    prev = torch.zeros(1, len(ACTIONS))
    g = build_start(spec)
    with torch.no_grad():
        for k in range(step + 1):
            o = torch.as_tensor(obs_of(g)[None])
            av = torch.as_tensor(avail_mask(g)[None])
            env = net.run(o, mem, prev, av)
            if k == step:
                logits = env["logits"].masked_fill(~av, float("-inf"))[0]
                nodes = {}
                for nid in spatial[-n_nodes:]:
                    t = env[nid][0]
                    key = nid if t.shape[0] <= max_ch else f"{nid}[0:{max_ch}] of {t.shape[0]}"
                    nodes[key] = [_max_at(t[c]) for c in range(min(max_ch, t.shape[0]))]
                return {"logits": [None if not np.isfinite(x) else round(float(x), 3) for x in logits.tolist()],
                        "channel_max_at": nodes, "click_max_at": _max_at(env["click"][0, 0]),
                        "memory_l2": round(float(env["mem_next"].norm()), 3) if net.M else 0.0}
            a, r, c = trace[k] if k < len(trace) else ("ACTION1", None, None)
            mem = env["mem_next"] if net.M else mem
            prev = torch.zeros(1, len(ACTIONS))
            if a in ACTIONS:
                prev[0, ACTIONS.index(a)] = 1
                g.step(a, r, c)
    return None


def net_shapes(arch, params) -> dict:
    """Top-level node (and loop-state output) shapes in graph order: id -> ('s', C) | ('v', n)."""
    return analyse(arch, params)[2]["shapes"]

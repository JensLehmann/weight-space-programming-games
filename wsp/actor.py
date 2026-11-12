"""Fixed actor architecture (one graph for every method), torch implementation + NumPy reference.

    x0  = [onehot16(obs), row/63, col/63, 1]                         19x64x64
    F1  = relu(conv5x5(x0; W1,b1))                                   32
    F2  = relu(conv3x3(F1; W2,b2))                                   32
    F3  = relu(conv1x1([F2, rowmax(F2), colmax(F2)]; W3,b3))         32   (axial broadcast)
    H   = 0;  repeat T=128: H = relu(conv3x3(H; Wh) + conv1x1(F3; Uh) + bh)   16  (tied)
    g   = [max(H), mean(H), max(F3), mean(F3)]                       96   (spatial pooling)
    z   = relu(Wz g + bz)                                            128
    m'  = tanh(Wm m + Wmz z + Wma a_prev + bm)                       64   (episode memory)
    act = argmax over available of  Wo [z, m'] + bo                  8
    click (if ACTION6) = argmax_{row,col} conv1x1([F3, H]; Wc, bc)   64x64
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .envs import ACTIONS, CLICK

T_ITER = 128
# Operator-ablation variants: same tensor schema, different fixed graph.
VARIANTS = {"full": {"axial": True, "T": 128}, "noaxial": {"axial": False, "T": 128}, "t8": {"axial": True, "T": 8}}
SCHEMA: dict[str, tuple[int, ...]] = {
    "W1": (32, 19, 5, 5), "b1": (32,),
    "W2": (32, 32, 3, 3), "b2": (32,),
    "W3": (32, 96), "b3": (32,),
    "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
    "Wz": (128, 96), "bz": (128,),
    "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,),
    "Wo": (8, 192), "bo": (8,),
    "Wc": (1, 48), "bc": (1,),
}
N_PARAMS = int(sum(np.prod(s) for s in SCHEMA.values()))
MEM = 64


def validate(params: dict) -> dict[str, np.ndarray]:
    """Return float32 copies; raise ValueError on missing/extra/misshaped/non-finite tensors."""
    if set(params) != set(SCHEMA):
        raise ValueError(f"tensor names mismatch: missing={sorted(set(SCHEMA)-set(params))} "
                         f"extra={sorted(set(params)-set(SCHEMA))}")
    out = {}
    for k, shape in SCHEMA.items():
        a = np.asarray(params[k])
        if a.shape != shape:
            raise ValueError(f"{k}: shape {a.shape} != {shape}")
        a32 = a.astype(np.float32)
        if not np.isfinite(a32).all():
            raise ValueError(f"{k}: non-finite values after float32 cast")
        out[k] = a32
    return out


def cast_report(params: dict) -> dict:
    """Entries changed by the float32 deployment cast (for intended float64 constructions)."""
    rep = {}
    for k in SCHEMA:
        a = np.asarray(params[k], dtype=np.float64)
        d = np.abs(a - a.astype(np.float32).astype(np.float64))
        rep[k] = {"changed": int((d > 0).sum()), "max_abs": float(d.max()) if d.size else 0.0}
    return rep


def save_npz(params: dict, path: Path) -> None:
    np.savez_compressed(path, **validate(params))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return validate({k: z[k] for k in z.files})


def to_bytes(params: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **validate(params))
    return buf.getvalue()


# --------------------------------------------------------------------------- torch implementation
class Actor(torch.nn.Module):
    def __init__(self, params: dict | None = None, variant: str = "full"):
        super().__init__()
        self.variant = variant
        self.axial = VARIANTS[variant]["axial"]
        self.T = VARIANTS[variant]["T"]
        for k, shape in SCHEMA.items():
            self.register_parameter(k, torch.nn.Parameter(torch.zeros(shape)))
        if params is not None:
            self.load(params)
        coords = torch.stack(torch.meshgrid(torch.arange(64.0) / 63, torch.arange(64.0) / 63,
                                            indexing="ij"))
        self.register_buffer("coords", torch.cat([coords, torch.ones(1, 64, 64)]), persistent=False)

    def load(self, params: dict) -> None:
        p = validate(params)
        with torch.no_grad():
            for k in SCHEMA:
                getattr(self, k).copy_(torch.from_numpy(p[k]))

    def export(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).detach().cpu().numpy().astype(np.float32) for k in SCHEMA}

    def encode(self, obs: torch.Tensor):
        """obs: (B,64,64) long -> F3 (B,32,64,64), H (B,16,64,64)."""
        B = obs.shape[0]
        x = F.one_hot(obs.long(), 16).permute(0, 3, 1, 2).float()
        x = torch.cat([x, self.coords.expand(B, -1, -1, -1)], 1)
        f1 = F.relu(F.conv2d(x, self.W1, self.b1, padding=2))
        f2 = F.relu(F.conv2d(f1, self.W2, self.b2, padding=1))
        if self.axial:
            rm = f2.amax(dim=3, keepdim=True).expand_as(f2)
            cm = f2.amax(dim=2, keepdim=True).expand_as(f2)
            f3 = F.relu(torch.einsum("oc,bchw->bohw", self.W3, torch.cat([f2, rm, cm], 1))
                        + self.b3[None, :, None, None])
        else:   # ablation: no row/column broadcasts (W3[:, 32:] unused)
            f3 = F.relu(torch.einsum("oc,bchw->bohw", self.W3[:, :32], f2) + self.b3[None, :, None, None])
        drive = torch.einsum("oc,bchw->bohw", self.Uh, f3) + self.bh[None, :, None, None]
        h = torch.zeros(B, 16, 64, 64, device=obs.device)
        if bool(self.Wh.abs().sum() > 0) or bool(drive.abs().sum() > 0):
            for _ in range(self.T):
                h = F.relu(F.conv2d(h, self.Wh, padding=1) + drive)
        return f3, h

    def forward(self, obs, mem, a_prev, avail):
        """Returns action logits (masked), click logits (B,64,64), new memory."""
        f3, h = self.encode(obs)
        g = torch.cat([h.amax((2, 3)), h.mean((2, 3)), f3.amax((2, 3)), f3.mean((2, 3))], 1)
        z = F.relu(g @ self.Wz.T + self.bz)
        mem2 = torch.tanh(mem @ self.Wm.T + z @ self.Wmz.T + a_prev @ self.Wma.T + self.bm)
        logits = torch.cat([z, mem2], 1) @ self.Wo.T + self.bo
        logits = logits.masked_fill(~avail, float("-inf"))
        click = (torch.einsum("oc,bchw->bohw", self.Wc, torch.cat([f3, h], 1))
                 + self.bc[None, :, None, None])[:, 0]
        return logits, click, mem2


def decode(logits: torch.Tensor, click: torch.Tensor) -> list[tuple]:
    """Deterministic argmax decoding; ties -> lowest action index / first row-major cell."""
    moves = []
    a_idx = torch.argmax(logits, 1).tolist()      # torch.argmax returns the first maximal index
    flat = torch.argmax(click.flatten(1), 1).tolist()
    for a, cell in zip(a_idx, flat):
        if a == CLICK:
            moves.append((ACTIONS[a], cell // 64, cell % 64))
        else:
            moves.append((ACTIONS[a], None, None))
    return moves


class NeuralPolicy:
    """Wraps an Actor for envs.run_episodes; memory/prev-action reset at each episode start."""

    def __init__(self, actor: Actor, device: str = "cpu"):
        self.nonfinite = 0
        self.actor = actor.to(device).eval()
        self.device = device
        self.mem: dict[int, torch.Tensor] = {}
        self.prev: dict[int, int | None] = {}

    def begin(self, slots):
        for i in slots:
            self.mem[i] = torch.zeros(MEM, device=self.device)
            self.prev[i] = None

    @torch.no_grad()
    def act(self, slots, obs, avail):
        dev = self.device
        o = torch.as_tensor(obs, device=dev)
        av = torch.as_tensor(avail, device=dev)
        mem = torch.stack([self.mem[i] for i in slots])
        a_prev = torch.zeros(len(slots), len(ACTIONS), device=dev)
        for j, i in enumerate(slots):
            if self.prev[i] is not None:
                a_prev[j, self.prev[i]] = 1.0
        logits, click, mem2 = self.actor(o, mem, a_prev, av)
        moves = decode(logits, click)
        # non-finite logits on available actions (e.g. overflow in high-gain written weights) are
        # reported as an invalid action rather than silently decoded by argmax
        bad = (~torch.isfinite(logits) & av).any(1) | torch.isnan(logits).any(1)
        for j in torch.nonzero(bad).flatten().tolist():
            moves[j] = ("NONFINITE", None, None)
            self.nonfinite += 1
        for j, i in enumerate(slots):
            self.mem[i] = mem2[j]
            self.prev[i] = ACTIONS.index(moves[j][0]) if moves[j][0] in ACTIONS else None
        return moves


# --------------------------------------------------------------------------- NumPy reference
def _conv(x, w, b=None, pad=0):
    """x (C,H,W), w (O,C,k,k) cross-correlation with zero padding, stride 1."""
    k = w.shape[-1]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(1, 2))  # C,H,W,k,k
    y = np.einsum("chwij,ocij->ohw", win, w, optimize=True)
    return y if b is None else y + b[:, None, None]


def reference_step(p: dict, obs: np.ndarray, mem: np.ndarray, a_prev: np.ndarray,
                   avail: np.ndarray, variant: str = "full"):
    """Exact semantics of one actor step (float32). Returns (action, row, col), logits, new mem."""
    p = {k: np.asarray(v, np.float32) for k, v in p.items()}
    relu = lambda v: np.maximum(v, 0)
    x = np.zeros((19, 64, 64), np.float32)
    x[obs, np.arange(64)[:, None], np.arange(64)[None, :]] = 1.0
    x[16] = np.arange(64)[:, None] / 63.0
    x[17] = np.arange(64)[None, :] / 63.0
    x[18] = 1.0
    f1 = relu(_conv(x, p["W1"], p["b1"], 2))
    f2 = relu(_conv(f1, p["W2"], p["b2"], 1))
    if VARIANTS[variant]["axial"]:
        rm = np.broadcast_to(f2.max(axis=2, keepdims=True), f2.shape)
        cm = np.broadcast_to(f2.max(axis=1, keepdims=True), f2.shape)
        f3 = relu(np.einsum("oc,chw->ohw", p["W3"], np.concatenate([f2, rm, cm])) + p["b3"][:, None, None])
    else:
        f3 = relu(np.einsum("oc,chw->ohw", p["W3"][:, :32], f2) + p["b3"][:, None, None])
    drive = np.einsum("oc,chw->ohw", p["Uh"], f3) + p["bh"][:, None, None]
    h = np.zeros((16, 64, 64), np.float32)
    for _ in range(VARIANTS[variant]["T"]):
        h = relu(_conv(h, p["Wh"], None, 1) + drive)
    g = np.concatenate([h.max((1, 2)), h.mean((1, 2)), f3.max((1, 2)), f3.mean((1, 2))])
    z = relu(p["Wz"] @ g + p["bz"])
    mem2 = np.tanh(p["Wm"] @ mem + p["Wmz"] @ z + p["Wma"] @ a_prev + p["bm"])
    logits = p["Wo"] @ np.concatenate([z, mem2]) + p["bo"]
    logits = np.where(avail, logits, -np.inf)
    a = int(np.argmax(logits))
    click = np.einsum("c,chw->hw", p["Wc"][0], np.concatenate([f3, h])) + p["bc"][0]
    if a == CLICK:
        cell = int(np.argmax(click))
        return (ACTIONS[a], cell // 64, cell % 64), logits, mem2
    return (ACTIONS[a], None, None), logits, mem2


def random_params(seed: int, scale: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in SCHEMA.items()}


def zero_params() -> dict[str, np.ndarray]:
    return {k: np.zeros(s, np.float32) for k, s in SCHEMA.items()}

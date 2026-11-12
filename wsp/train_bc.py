"""Behaviour cloning + DAgger into the fixed actor (BC* planner teacher, D = writer-player teacher).

The teacher is privileged at training time only (planner) or an ordinary program (writer player).
The student is the same fixed actor architecture. Memory is carried forward during collection.

Default behaviour (all options below off): memory is stored as an input and every labelled step
is an independent sample (no backprop through time). Options (all off by default):

* planner teacher (PlannerTeacher defaults): the planner gets the certification budget (5000 ms) and
  never falls back to RESET; states it cannot label are dropped from training and counted
  (``unlabelled_frac`` in the log). ``budget_ms`` and ``reset_fallback`` change this.
* ``augment``: set of kinds from ``wsp.augment`` ('palette', 'dihedral'), applied to sampled
  minibatches (one transformation per sample / per BPTT window).
* ``bptt=L``: trajectories are stored as sequences (including unlabelled steps, which still feed
  the memory) and trained over windows of length L (random per-epoch offsets), unrolled from the
  memory stored at the window start, so gradients flow through Wm/Wmz/Wma (and through z of
  earlier frames) across steps. The convolutional encoder does not depend on the memory, so all
  frames of a minibatch of windows are encoded in one batch; a minibatch holds ``batch // L``
  windows, i.e. the same number of frames (and activation memory) as a per-step minibatch.
* ``progressive``: Bansal et al. (2022) progressive loss for the tied block, simplified: with
  probability 0.5 per minibatch, sample n ~ U{0..T-1} and k ~ U{1..T-n}; run n tied iterations
  without gradient, then k with gradient, and compute the loss on the output after n+k <= T
  iterations (instead of Bansal's alpha-weighted sum of both losses). Otherwise the usual T
  iterations with gradient. Deployment is unchanged (T=128). BC/DAgger only: in PPO it would
  change the sampled policy and bias the importance ratio.
* ``grad_ckpt=s``: gradient checkpointing of the tied block in segments of s iterations
  (activation memory ~ (T/s + 2 s) instead of 2 T feature maps per frame; ~+1 forward of the block).
* ``aug_memory``: where the memory entering a training window (BPTT) or step (per-step) comes from.
  ``"stored"`` (default, as run): the memory stored at collection time, i.e. computed by the
  collection-time student on the untransformed frames (zero in iteration 0), while ``augment``
  transforms the window's frames independently per minibatch, so under augmentation the memory is
  inconsistent with the frames. ``"recompute"``: at every epoch, one transformation is sampled per
  buffer episode (via ``augment_batch``, so the remapping is identical) and applied to its frames,
  available-action masks, previous actions and labels; the memory sequence is recomputed by rolling
  the current student (full depth, no gradient) through each transformed episode from its start;
  windows/steps are then drawn from the transformed episodes with the recomputed memories. Memories
  are exact for the student at the start of the epoch. Needs sequence storage, so collection stores
  every step with an episode id (as for BPTT) also without ``bptt``.
* ``student_arch`` (learned free-architecture students): the student is a writer's FA graph
  (wsp.freearch) with its written weights discarded and re-initialised (``FA_INIT``), trained by the
  same DAgger learners (``LearnedFA``). ``progressive`` draws a depth per loop of the graph
  (``fa_progressive``), ``grad_ckpt`` checkpoints every loop; BPTT windows run the memory-independent
  part of the graph once for all frames and the memory-dependent part step by step.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import actor as A
from . import freearch as FA
from .augment import augment_batch
from .envs import ACTIONS, A_INDEX, EpisodeSpec, avail_mask, build_start, obs_of
from .queries import evaluate_state

CERTIFY_BUDGET_MS = 5000             # = queries.certify default
AUG_MEMORY = ("stored", "recompute")


def _time_up(deadline: float | None) -> bool:
    """Wall-clock cap (dagger time_cap_h): True once time.time() passes `deadline` (None: no cap)."""
    return deadline is not None and time.time() > deadline
RECOMPUTE_CHUNK_FRAMES = 4096        # padded frames per episode chunk on the device (recompute)
RECOMPUTE_ENC_BATCH = 256            # frames per no-grad encoder call (recompute)


def init_params(seed: int) -> dict[str, np.ndarray]:
    """theta_0 for learned baselines: PyTorch-style uniform fan-in init; tied block scaled 0.1."""
    rng = np.random.default_rng(seed)
    p = {}
    for k, shape in A.SCHEMA.items():
        if k.startswith("b"):
            p[k] = np.zeros(shape, np.float32)
            continue
        fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        bound = 1.0 / np.sqrt(fan_in)
        p[k] = rng.uniform(-bound, bound, shape).astype(np.float32)
    p["Wh"] *= 0.1
    return p


# --------------------------------------------------------------------------- planner queries
@dataclass(frozen=True)
class PlanInfo:
    first: tuple | None      # first move of (direct_plan or plan)
    dist: int | None         # tightest certified upper bound on the distance to go
    recovery: bool = False   # label taken from a (RESET-containing) recovery certificate
    cutoff: bool = False     # planner hit its budget
    error: bool = False


class PlannerOracle:
    """Planner queries with a per-process cache keyed by a state hash.

    Key: (game class, level, sha1(repr(planner snapshot state_key))) when the planner exposes a
    snapshot, else (family, seed, level, observation bytes). The instrumentation game `toy_reach`
    uses its own `direct_plan()`. Results that hit the budget are cached too (re-querying the
    same state with the same budget would mostly burn the budget again).
    """

    def __init__(self, budget_ms: float = CERTIFY_BUDGET_MS, cache_size: int = 500_000,
                 cache_cutoffs: bool = True):
        self.budget_ms = budget_ms
        self.cache_size = cache_size
        self.cache_cutoffs = cache_cutoffs
        self.cache: OrderedDict = OrderedDict()
        self.stats: Counter = Counter()

    @staticmethod
    def _obs_key(game) -> tuple:
        return (getattr(game, "family", type(game).__name__), getattr(game, "seed", None),
                getattr(game, "level", None), obs_of(game).astype(np.int8).tobytes())

    def _prepare(self, game):
        if hasattr(game, "direct_plan"):
            return self._obs_key(game), None, None
        try:
            from arc3_games.api import make_planner
            from arc3_games.games.public import PublicGame
            target = game.engine if isinstance(game, PublicGame) else game
            planner = make_planner(game)
            snap = planner.snapshot(target)
            digest = hashlib.sha1(repr(snap.state_key).encode()).hexdigest()
            return (type(target).__qualname__, snap.level_index, digest), planner, snap
        except Exception:
            self.stats["no_state_key"] += 1
            return self._obs_key(game), None, None

    def info(self, game) -> PlanInfo:
        key, planner, snap = self._prepare(game)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return hit
        self.stats["queries"] += 1
        t0 = time.perf_counter()
        try:
            info = self._query(game, planner, snap)
        except Exception:
            self.stats["errors"] += 1
            info = PlanInfo(None, None, error=True)
        self.stats["planner_ms"] += int((time.perf_counter() - t0) * 1000)
        self.stats["cutoffs"] += int(info.cutoff)
        if not info.error and (self.cache_cutoffs or not info.cutoff):
            self.cache[key] = info
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return info

    def _query(self, game, planner, snap) -> PlanInfo:
        if hasattr(game, "direct_plan"):                 # instrumentation toy game
            plan = game.direct_plan()
            if plan is None:
                return PlanInfo(None, None)
            return PlanInfo((plan[0], None, None) if plan else None, len(plan))
        r = (planner.evaluate(snap, time_budget_ms=self.budget_ms) if planner is not None
             else evaluate_state(game, time_budget_ms=self.budget_ms))
        plan = r.direct_plan or r.plan
        first = None
        if plan:
            p = plan[0]
            first = (p.action, p.row, p.col)
        cands = [x for x in (r.distance.upper_bound,
                             None if r.direct_plan is None else len(r.direct_plan),
                             None if r.plan is None else len(r.plan)) if x is not None]
        return PlanInfo(first, min(cands) if cands else None, recovery=bool(plan) and not r.direct_plan,
                        cutoff=r.stats.cutoff_reason is not None)

    def distance(self, game) -> int | None:
        return self.info(game).dist


class PlannerTeacher:
    """Family planner as teacher. Revision 2 defaults: certification budget, no RESET fallback
    (label() returns None for states the planner cannot solve within the budget)."""

    def __init__(self, budget_ms: float = CERTIFY_BUDGET_MS, reset_fallback: bool = False,
                 oracle: PlannerOracle | None = None):
        self.budget_ms = budget_ms
        self.reset_fallback = reset_fallback
        self.oracle = oracle or PlannerOracle(budget_ms)
        self.counts: Counter = Counter()

    def begin(self, i): pass

    def label(self, i, game, obs):
        info = self.oracle.info(game)
        self.counts["calls"] += 1
        if info.first is None:
            self.counts["no_plan"] += 1
            return ("RESET", None, None) if self.reset_fallback else None
        self.counts["recovery_labels"] += int(info.recovery)
        return info.first

    def stats(self) -> dict:
        return {"budget_ms": self.budget_ms, "reset_fallback": self.reset_fallback,
                **self.counts, **{f"oracle_{k}": v for k, v in self.oracle.stats.items()}}

    def close(self): pass


class PlayerTeacher:
    """Writer-produced Python player as teacher (fresh process per episode)."""

    def __init__(self, code: str):
        from .sandbox import PythonPlayerPolicy
        self.pol = PythonPlayerPolicy(code)

    def begin(self, i):
        self.pol.begin([i])

    def label(self, i, game, obs):
        mv = self.pol.act([i], obs[None], avail_mask(game)[None])[0]
        return tuple(mv)

    def stats(self) -> dict:
        return {}

    def close(self):
        self.pol.close()


# --------------------------------------------------------------------------- data
@dataclass
class Buffer:
    obs: list
    avail: list
    mem: list
    prev: list
    act: list
    cell: list
    ep: list = field(default_factory=list)     # episode id per step (sequence storage only)
    n_eps: int = 0

    FIELDS = ("obs", "avail", "mem", "prev", "act", "cell")

    @staticmethod
    def new():
        return Buffer([], [], [], [], [], [])

    def add(self, o, av, m, pv, a, cell, ep: int | None = None):
        self.obs.append(o.astype(np.int8)); self.avail.append(av); self.mem.append(m)
        self.prev.append(pv); self.act.append(a); self.cell.append(cell)
        if ep is not None:
            self.ep.append(ep)

    def select(self, keep) -> None:
        keep = list(keep)
        for f in self.FIELDS + (("ep",) if self.ep else ()):
            setattr(self, f, [getattr(self, f)[k] for k in keep])

    def __len__(self):
        return len(self.act)


def collect(specs, student: A.Actor | None, teacher, beta: float, buf: Buffer, device: str,
            rng: random.Random, sequences: bool = False, mem_size: int = A.MEM,
            deadline: float | None = None) -> dict:
    """Roll out a beta-mixture of teacher and student; label every visited state with the teacher.

    Steps the teacher cannot label (None, or an unavailable action) are not trained on and are
    counted. With `sequences`, every visited step is stored (act=-1 if unlabelled) with an
    episode id, so training can unroll the memory over contiguous windows. mem_size: the
    student's memory size (FA students: memory_size). deadline: wall-clock cap, checked before
    every episode; when passed, collection stops and the result has "truncated": True."""
    solved = visited = unlabelled = 0
    truncated = False
    for i, spec in enumerate(specs):
        if _time_up(deadline):
            truncated = True
            break
        game = build_start(spec)
        teacher.begin(i)
        ep = buf.n_eps
        buf.n_eps += 1
        mem = np.zeros(mem_size, np.float32)
        prev = np.zeros(len(ACTIONS), np.float32)
        for _ in range(spec.cap):
            obs = obs_of(game)
            av = avail_mask(game)
            t_move = teacher.label(i, game, obs)
            if t_move is None or t_move[0] not in A_INDEX or not av[A_INDEX[t_move[0]]]:
                t_move = None
            visited += 1
            unlabelled += t_move is None
            if student is not None:
                with torch.no_grad():
                    lg, ck, m2 = student(torch.as_tensor(obs[None], device=device),
                                         torch.as_tensor(mem[None], device=device),
                                         torch.as_tensor(prev[None], device=device),
                                         torch.as_tensor(av[None], device=device))
                s_move = A.decode(lg, ck)[0]
                mem_next = m2[0].cpu().numpy()
            else:
                s_move, mem_next = None, mem
            if t_move is not None:
                cell = -1 if t_move[0] != "ACTION6" else int(t_move[1]) * 64 + int(t_move[2])
                buf.add(obs, av, mem.copy(), prev.copy(), A_INDEX[t_move[0]], cell,
                        ep if sequences else None)
            elif sequences:
                buf.add(obs, av, mem.copy(), prev.copy(), -1, -1, ep)
            use_teacher = s_move is None or (t_move is not None and rng.random() < beta)
            move = t_move if use_teacher else s_move
            if move is None:
                break
            _, state, level, _, solved_now = game.step(move[0], move[1], move[2])
            mem = mem_next
            prev = np.zeros(len(ACTIONS), np.float32)
            prev[A_INDEX[move[0]]] = 1
            if solved_now or level > spec.level:
                solved += 1
                break
            if state == "GAME_OVER":
                break
    return {"episodes": len(specs), "solved": solved, "visited": visited, "unlabelled": unlabelled,
            "truncated": truncated}


# --------------------------------------------------------------------------- actor pieces
def depth(actor: A.Actor) -> int:
    """Deployed number of tied iterations of this actor (variant-aware)."""
    return int(getattr(actor, "T", A.T_ITER))


def tied_block(actor: A.Actor, drive: torch.Tensor, n_iter: int | None = None, n_nograd: int = 0,
               ckpt: int = 0) -> torch.Tensor:
    """H after n_iter (default: the actor's T) tied iterations; the first n_nograd without
    gradient; optional checkpointing of the gradient part in segments of `ckpt` iterations."""
    n_iter = depth(actor) if n_iter is None else n_iter
    h = torch.zeros(drive.shape[0], 16, 64, 64, device=drive.device, dtype=drive.dtype)
    if not (bool(actor.Wh.abs().sum() > 0) or bool(drive.abs().sum() > 0)):
        return h
    Wh = actor.Wh

    def run(h_, d_, n):
        for _ in range(n):
            h_ = F.relu(F.conv2d(h_, Wh, padding=1) + d_)
        return h_

    if n_nograd:
        with torch.no_grad():
            h = run(h, drive, n_nograd)
    rest = n_iter - n_nograd
    if ckpt and torch.is_grad_enabled():
        while rest > 0:
            s = min(ckpt, rest)
            h = checkpoint(run, h, drive, s, use_reentrant=False)
            rest -= s
        return h
    return run(h, drive, rest)


def features(actor: A.Actor, obs: torch.Tensor, n_iter: int | None = None, n_nograd: int = 0,
             ckpt: int = 0):
    """Actor.encode, optionally with a progressive split / checkpointed tied block.
    (The replica is used only when an option is on; tests check it against Actor.encode.)"""
    if not n_nograd and not ckpt and n_iter in (None, depth(actor)):
        return actor.encode(obs)
    B = obs.shape[0]
    x = F.one_hot(obs.long(), 16).permute(0, 3, 1, 2).float()
    x = torch.cat([x, actor.coords.expand(B, -1, -1, -1)], 1)
    f1 = F.relu(F.conv2d(x, actor.W1, actor.b1, padding=2))
    f2 = F.relu(F.conv2d(f1, actor.W2, actor.b2, padding=1))
    if getattr(actor, "axial", True):
        rm = f2.amax(dim=3, keepdim=True).expand_as(f2)
        cm = f2.amax(dim=2, keepdim=True).expand_as(f2)
        f3 = F.relu(torch.einsum("oc,bchw->bohw", actor.W3, torch.cat([f2, rm, cm], 1))
                    + actor.b3[None, :, None, None])
    else:                                   # operator-ablation variant without axial broadcasts
        f3 = F.relu(torch.einsum("oc,bchw->bohw", actor.W3[:, :32], f2) + actor.b3[None, :, None, None])
    drive = torch.einsum("oc,bchw->bohw", actor.Uh, f3) + actor.bh[None, :, None, None]
    return f3, tied_block(actor, drive, n_iter, n_nograd, ckpt)


def heads(actor: A.Actor, f3: torch.Tensor, h: torch.Tensor):
    """Memory-independent part of Actor.forward: z (B,128), click logits (B,64,64)."""
    g = torch.cat([h.amax((2, 3)), h.mean((2, 3)), f3.amax((2, 3)), f3.mean((2, 3))], 1)
    z = F.relu(g @ actor.Wz.T + actor.bz)
    click = (torch.einsum("oc,bchw->bohw", actor.Wc, torch.cat([f3, h], 1))
             + actor.bc[None, :, None, None])[:, 0]
    return z, click


def mem_step(actor: A.Actor, z, mem, a_prev, avail):
    mem2 = torch.tanh(mem @ actor.Wm.T + z @ actor.Wmz.T + a_prev @ actor.Wma.T + actor.bm)
    logits = torch.cat([z, mem2], 1) @ actor.Wo.T + actor.bo
    return logits.masked_fill(~avail, float("-inf")), mem2


def forward_step(actor: A.Actor, obs, mem, a_prev, avail, **enc):
    """Equivalent of actor(obs, mem, a_prev, avail) with encoder options (progressive, ckpt)."""
    if not enc or not any(enc.values()):
        return actor(obs, mem, a_prev, avail)
    z, click = heads(actor, *features(actor, obs, **enc))
    logits, mem2 = mem_step(actor, z, mem, a_prev, avail)
    return logits, click, mem2


def forward_window(actor: A.Actor, obs, mem0, prev, avail, reset=None, **enc):
    """Unroll the episodic memory over windows.

    obs (W,L,64,64), mem0 (W,64) memory entering step 0, prev (W,L,8), avail (W,L,8) bool,
    reset (W,L) bool: memory is zeroed before step t (new episode). Returns logits (W,L,8),
    click (W,L,64,64), mem (W,L,64) = memory after each step."""
    W, L = obs.shape[:2]
    f3, h = features(actor, obs.reshape(W * L, 64, 64), **enc)
    z, click = heads(actor, f3, h)
    z = z.reshape(W, L, -1)
    mem, logits, mems = mem0, [], []
    for t in range(L):
        if reset is not None:
            mem = torch.where(reset[:, t, None], torch.zeros_like(mem), mem)
        lg, mem = mem_step(actor, z[:, t], mem, prev[:, t], avail[:, t])
        logits.append(lg)
        mems.append(mem)
    return torch.stack(logits, 1), click.reshape(W, L, 64, 64), torch.stack(mems, 1)


def progressive_depth(rng: np.random.Generator, T: int = A.T_ITER, p: float = 0.5) -> dict:
    """Encoder kwargs for one minibatch of the progressive loss (empty = full depth T)."""
    if rng.random() >= p:
        return {}
    n = int(rng.integers(0, T))
    k = int(rng.integers(1, T - n + 1))
    return {"n_iter": n + k, "n_nograd": n}


# --------------------------------------------------------------------------- learned free-architecture students
FA_INIT = ("written values discarded (shapes kept); every parameter re-drawn: conv weights (O,C,k,k) and "
           "dense weights (out,in) ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = C*k*k resp. in (PyTorch's "
           "nn.Conv2d/nn.Linear default, kaiming_uniform_(a=sqrt(5)); the fixed actor's init_params), x0.1 for "
           "weights of recurrent loop-body nodes (nodes inside a loop body that read a loop state, directly "
           "or through earlier body nodes; as the fixed actor's tied block Wh), rng "
           "np.random.default_rng([seed, 2]), parameters in sorted name order; biases (1-d): ")
FA_BIAS_INIT = {"zero": "zero (as init_params)",
                "uniform": "U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with the fan_in of the node's weight (PyTorch's "
                           "nn.Conv2d/nn.Linear default), rng np.random.default_rng([seed, 3]), sorted name order "
                           "(weights identical to 'zero')"}
FA_PRECISION = ("training and DAgger rollouts use the process-default torch precision, as the fixed-actor "
                "learners (FreeArchNet.run's exact_fp32() switch is not called); evaluation is exact FP32")
RECOMPUTE_FA_EPISODES = 64           # episodes per chunk (= batch of one recompute step) for FA students


def load_fa_graph(path) -> tuple[dict, dict]:
    """<dir>/round<k>.graph.json (params in <dir>/round<k>.npz) -> validated (arch, written params)."""
    path = Path(path)
    m = re.fullmatch(r"round(\d+)\.graph\.json", path.name)
    if m is None:
        raise ValueError(f"--student-arch must be a round<k>.graph.json file, got {path}")
    return FA.load_net(path.parent, int(m.group(1)))


def _fa_loop_outer_refs(nd) -> set:
    """Ids a loop node reads from the enclosing scope (state inits and body inputs)."""
    local = {s["id"] for s in nd["state"]}
    refs = {s["init"] for s in nd["state"] if "init" in s}
    for b in nd["body"]:
        refs |= {i for i in b["in"] if i not in local}
        local.add(b["id"])
    return refs


def _fa_recurrent_params(graph) -> set:
    """Parameter names used by loop-body nodes that depend (within the body) on a loop state."""
    out = set()
    for nd in graph:
        if nd["op"] != "loop":
            continue
        dep = {s["id"] for s in nd["state"]}
        for b in nd["body"]:
            if set(b["in"]) & dep:
                dep.add(b["id"])
                out |= {b[k] for k in ("w", "b") if k in b}
    return out


def _fa_bias_fan_in(graph) -> dict:
    """bias name -> fan_in of the weight of the (first) conv/dense node using it."""
    out = {}
    for nd in graph:
        if nd["op"] == "loop":
            out = {**_fa_bias_fan_in(nd["body"]), **out}
        elif "b" in nd and nd["b"] not in out:
            out[nd["b"]] = nd["w"]
    return out


def fa_init_params(arch: dict, written: dict, seed: int, bias: str = "zero") -> dict[str, np.ndarray]:
    """Re-initialised parameters of an FA graph (scheme: FA_INIT + FA_BIAS_INIT[bias])."""
    if bias not in FA_BIAS_INIT:
        raise ValueError(f"bias init must be one of {sorted(FA_BIAS_INIT)}, got {bias!r}")
    rng = np.random.default_rng([seed, 2])
    rec = _fa_recurrent_params(arch["graph"])
    p = {}
    for k in sorted(written):
        shape = tuple(written[k].shape)
        if len(shape) == 1:
            p[k] = np.zeros(shape, np.float32)
            continue
        if len(shape) not in (2, 4):
            raise ValueError(f"FA parameter {k!r} has shape {shape}; expected 1-d, 2-d or 4-d")
        bound = 1.0 / np.sqrt(int(np.prod(shape[1:])))
        p[k] = rng.uniform(-bound, bound, shape).astype(np.float32) * np.float32(0.1 if k in rec else 1.0)
    if bias == "uniform":
        brng = np.random.default_rng([seed, 3])
        owner = _fa_bias_fan_in(arch["graph"])
        for k in sorted(owner):
            bound = 1.0 / np.sqrt(int(np.prod(written[owner[k]].shape[1:])))
            p[k] = brng.uniform(-bound, bound, written[k].shape).astype(np.float32)
    return p


class LearnedFA(FA.FreeArchNet):
    """FreeArchNet with trainable parameters (nn.Parameter 'p_<name>' instead of buffers).

    forward() (collection, recompute) is FreeArchNet's executor without the exact-FP32 switch
    (FA_PRECISION). Training goes through fa_forward_step / fa_forward_window (same ops, plus
    progressive loop depths and loop checkpointing). For windows, the graph is split into the
    memory-independent part (`static`: evaluated once for all W*L frames of a window batch) and
    the nodes that depend on `mem` (`dynamic`: unrolled step by step over the window)."""

    def __init__(self, arch: dict, params: dict):
        super().__init__(arch, params)
        for k in [k for k in self._buffers if k.startswith("p_")]:
            self.register_parameter(k, torch.nn.Parameter(self._buffers.pop(k)))
        self.loop_T = {nd["id"]: nd["T"] for nd in self.graph if nd["op"] == "loop"}
        self.loop_refs = {nd["id"]: sorted(_fa_loop_outer_refs(nd)) for nd in self.graph if nd["op"] == "loop"}
        dep = {"mem"} if self.M else set()
        self.static, self.dynamic, reads = [], [], set()
        for nd in self.graph:                       # graph order is a topological order
            ins = _fa_loop_outer_refs(nd) if nd["op"] == "loop" else set(nd["in"])
            if ins & dep:
                self.dynamic.append(nd)
                reads |= ins
                dep.add(nd["id"])
                if nd["op"] == "loop":
                    dep |= {f"{nd['id']}.{s['id']}" for s in nd["state"]}
            else:
                self.static.append(nd)
        self.dyn_ids = dep - {"mem"}                # memory-dependent values
        self.frontier = sorted(reads - dep)         # static values / inputs read by dynamic nodes

    def P(self, name):
        return self._parameters["p_" + name]

    def inputs(self, obs, aprev, avail) -> dict:
        B = obs.shape[0]
        return {"x0": F.one_hot(obs.long(), 16).permute(0, 3, 1, 2).float(),
                "coords": self.coords.expand(B, -1, -1, -1), "aprev": aprev, "avail": avail.float()}

    def run(self, obs, mem, aprev, avail) -> dict:
        env = self.inputs(obs, aprev, avail)
        if self.M:
            env["mem"] = mem
        self._run(self.graph, env, obs.shape[0], obs.device)
        return env

    def export(self) -> dict[str, np.ndarray]:
        return {k[2:]: v.detach().cpu().numpy().astype(np.float32) for k, v in self._parameters.items()}


def _fa_exec(net: LearnedFA, nodes, env: dict, B: int, dev, loops: dict | None = None, ckpt: int = 0):
    """FreeArchNet._run with training options: loops {loop id: (n_iter, n_nograd)} (progressive:
    the first n_nograd iterations without gradient, n_iter <= T in total; default (T, 0)) and
    ckpt = s (gradient checkpointing of every loop in segments of s iterations). Without options
    the same ops in the same order as FreeArchNet._run."""
    for nd in nodes:
        if nd["op"] != "loop":
            env[nd["id"]] = net._op(nd, [env[i] for i in nd["in"]])
            continue
        sids = [s["id"] for s in nd["state"]]
        state = tuple(env[s["init"]] if "init" in s else torch.zeros(B, *s["zeros"], device=dev)
                      for s in nd["state"])
        n_iter, n_nograd = (loops or {}).get(nd["id"], (nd["T"], 0))
        # The body reads only these outer values. Binding a snapshot (not `env`, which later also
        # holds the loop's outputs) avoids a reference cycle output -> checkpoint -> closure -> env
        # that would keep every minibatch's activations alive until Python's cyclic GC runs.
        outer = {k: env[k] for k in net.loop_refs[nd["id"]]}

        def run(n, *st, nd=nd, sids=sids, outer=outer):      # bound now: recomputed in backward
            cur = dict(zip(sids, st))
            for _ in range(n):
                local = {**outer, **cur}
                _fa_exec(net, nd["body"], local, B, dev)
                cur = {sid: local[src] for sid, src in nd["next"].items()}
            return tuple(cur[s] for s in sids)

        if n_nograd:
            with torch.no_grad():
                state = run(n_nograd, *state)
        rest = n_iter - n_nograd
        if ckpt and torch.is_grad_enabled():
            while rest > 0:
                s = min(ckpt, rest)
                state = checkpoint(run, s, *state, use_reentrant=False)
                rest -= s
        else:
            state = run(rest, *state)
        for sid, v in zip(sids, state):
            env[nd["id"] + "." + sid] = v


def fa_forward_step(net: LearnedFA, obs, mem, a_prev, avail, loops=None, ckpt=0):
    """forward_step for FA students: (logits masked, click (B,64,64), mem_next (B,M))."""
    env = net.inputs(obs, a_prev, avail)
    if net.M:
        env["mem"] = mem
    _fa_exec(net, net.graph, env, obs.shape[0], obs.device, loops, ckpt)
    logits = env["logits"].masked_fill(~avail, float("-inf"))
    return logits, env["click"][:, 0], (env["mem_next"] if net.M else mem[:, :0])


def fa_forward_window(net: LearnedFA, obs, mem0, prev, avail, reset=None, loops=None, ckpt=0):
    """forward_window for FA students (same shapes; mem (W,L,M)). The memory-independent nodes run
    once on all W*L frames; the memory-dependent nodes step by step over the window."""
    W, L = obs.shape[:2]
    dev = obs.device
    env = net.inputs(obs.reshape(W * L, 64, 64), prev.reshape(W * L, -1), avail.reshape(W * L, -1))
    _fa_exec(net, net.static, env, W * L, dev, loops, ckpt)
    per_t = {k: [] for k in ("logits", "click", "mem_next") if k in net.dyn_ids}
    mem = mem0
    if net.dynamic:
        front = {k: env[k].reshape(W, L, *env[k].shape[1:]) for k in net.frontier if k != "coords"}
        for t in range(L):
            if reset is not None:
                mem = torch.where(reset[:, t, None], torch.zeros_like(mem), mem)
            e = {k: v[:, t] for k, v in front.items()}
            e["coords"] = net.coords.expand(W, -1, -1, -1)
            e["mem"] = mem
            _fa_exec(net, net.dynamic, e, W, dev, loops, ckpt)
            for k in per_t:
                per_t[k].append(e[k])
            if net.M:
                mem = e["mem_next"] if "mem_next" in per_t else env["mem_next"].reshape(W, L, -1)[:, t]

    def out(k):
        return torch.stack(per_t[k], 1) if k in per_t else env[k].reshape(W, L, *env[k].shape[1:])

    logits = out("logits").masked_fill(~avail, float("-inf"))
    mems = out("mem_next") if net.M else mem0[:, None, :0].expand(W, L, 0)
    return logits, out("click")[:, :, 0], mems


def fa_progressive(rng: np.random.Generator, loop_T: dict, p: float = 0.5) -> dict:
    """Progressive loss for FA students: with probability p per minibatch, every loop gets its own
    n ~ U{0..T-1}, k ~ U{1..T-n} (n iterations without gradient, then k with). A graph with one loop
    draws exactly as progressive_depth; graphs without loops draw nothing."""
    if not loop_T or rng.random() >= p:
        return {}
    out = {}
    for lid, T in loop_T.items():
        n = int(rng.integers(0, T))
        k = int(rng.integers(1, T - n + 1))
        out[lid] = (n + k, n)
    return {"loops": out}


FA_INIT_PROBE = ("BC loss on the start frames of <= 16 evenly spaced training specs (train_specs[::len//16][:16]) "
                 "as W=n/2 windows of 2 frames, zero memory and previous action, one random available action "
                 "and one random click cell per frame (torch.Generator seed 1), before training; "
                 "init_dead_grad_frac = fraction of parameter entries whose gradient is exactly zero (1.0 if "
                 "the loss is non-finite); init_nan = non-finite available logits or loss")


def fa_init_probe(net: LearnedFA, specs, device: str, ckpt: int = 4, n: int = 16) -> dict:
    """Gradient flow of a freshly initialised FA student (FA_INIT_PROBE). Parameters unchanged;
    gradients are cleared afterwards; no training rng is used."""
    sel = list(specs)[::max(1, len(specs) // n)][:n]
    W, L = (len(sel) // 2, 2) if len(sel) >= 2 else (len(sel), 1)
    games = [build_start(s) for s in sel[:W * L]]
    obs = torch.as_tensor(np.stack([obs_of(g) for g in games])).long().reshape(W, L, 64, 64).to(device)
    av = torch.as_tensor(np.stack([avail_mask(g) for g in games])).reshape(W, L, 8)
    g = torch.Generator().manual_seed(1)
    y = torch.stack([torch.multinomial(a.float(), 1, generator=g)[0] for a in av.reshape(-1, 8)]).to(device)
    cell = torch.randint(0, 4096, (W * L,), generator=g).to(device)
    av = av.to(device)
    net.zero_grad(set_to_none=True)
    lg, ck, _ = fa_forward_window(net, obs, torch.zeros(W, net.M, device=device), torch.zeros(W, L, 8, device=device),
                                  av, ckpt=ckpt)
    lgf, avf = lg.reshape(-1, 8), av.reshape(-1, 8)
    loss = _bc_loss(lgf, ck.reshape(-1, 64, 64), y, cell, avf)
    nan = not bool(torch.isfinite(lgf[avf]).all()) or not bool(torch.isfinite(loss))
    if bool(torch.isfinite(loss)):
        loss.backward()
    total = sum(int(p.numel()) for p in net.parameters())
    dead = sum(int(p.numel()) for p in net.parameters() if p.grad is None or float(p.grad.abs().sum()) == 0)
    net.zero_grad(set_to_none=True)
    return {"init_dead_grad_frac": round(dead / total, 6), "init_nan": nan, "init_probe_frames": W * L,
            "init_probe": FA_INIT_PROBE}


def save_fa_student(net: LearnedFA, out_dir: Path, stem: str = "student") -> dict[str, np.ndarray]:
    """<stem>.graph.json (arch, as round<k>.graph.json) + <stem>_fa.npz (params, as round<k>.npz).
    No student.npz: tools that read student.npz assume the fixed actor (wsp.actor.load_npz)."""
    arch, p32 = FA.validate(net.arch, net.export())
    np.savez_compressed(out_dir / f"{stem}_fa.npz", **p32)
    (out_dir / f"{stem}.graph.json").write_text(json.dumps(arch))
    return p32


def load_fa_student(out_dir: Path, stem: str = "student") -> tuple[dict, dict]:
    arch = json.loads((Path(out_dir) / f"{stem}.graph.json").read_text())
    with np.load(Path(out_dir) / f"{stem}_fa.npz") as z:
        params = {n: z[n] for n in z.files}
    return FA.validate(arch, params)


def make_windows(ep: np.ndarray, act: np.ndarray, L: int, rng: np.random.Generator | None = None):
    """Index windows (W,L) over contiguous episode segments; padded entries repeat the last valid
    index and are masked out by `valid` (W,L). With rng, each episode gets a random offset in
    [0, L) for its first window. Windows without any labelled step are dropped."""
    n = len(ep)
    if n == 0:
        return np.zeros((0, L), np.int64), np.zeros((0, L), bool)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], n]
    rows, valid = [], []
    for s, e in zip(starts, ends):
        cuts = list(range(s, e, L))
        if rng is not None and e - s > 1:
            off = int(rng.integers(0, L))
            cuts = [s] + list(range(s + off, e, L)) if off else cuts
            cuts = sorted(set(cuts))
        for j, t0 in enumerate(cuts):
            t1 = cuts[j + 1] if j + 1 < len(cuts) else e
            t1 = min(t1, t0 + L)
            if not (act[t0:t1] >= 0).any():
                continue
            r = np.arange(t0, t0 + L)
            rows.append(np.minimum(r, t1 - 1))
            valid.append(r < t1)
    return np.array(rows, np.int64).reshape(-1, L), np.array(valid, bool).reshape(-1, L)


@torch.no_grad()
def transformed_episodes(actor: A.Actor, obs, avail, prev, act, cell, ep, kinds,
                         rng: np.random.Generator, device: str,
                         chunk_frames: int = RECOMPUTE_CHUNK_FRAMES,
                         enc_batch: int = RECOMPUTE_ENC_BATCH, max_episodes: int | None = None,
                         deadline: float | None = None):
    """aug_memory='recompute': transformed copy of a sequence buffer with recomputed memory.

    obs (N,64,64) int, avail (N,8) bool, prev (N,8), act (N,), cell (N,): flat CPU buffer tensors;
    ep (N,): episode ids, each episode contiguous and complete from its first step (as stored by
    `collect(sequences=True)`). Every episode is one leading index of `augment_batch`, so it gets
    one transformation shared by all its steps, with exactly the remapping used per window by the
    stored path (kinds None/empty: identity, no rng draws). The memory entering every step is then
    recomputed by rolling `actor` (deployed depth, no gradient) through the transformed episode from
    zero memory, batched across the episodes of a chunk, step by step over time. Episodes are
    processed in chunks of <= chunk_frames padded frames (at least one episode per chunk) and, if
    given, <= max_episodes episodes. FA students (LearnedFA): the whole graph is run per step on the
    episodes of the chunk still running (memory_size M; nothing to run when M = 0).
    Returns CPU tensors (obs int8, avail, prev, act, cell, mem (N,M)) in buffer row order, or None
    if the wall-clock `deadline` passes (checked between chunks)."""
    fa = isinstance(actor, LearnedFA)
    M = actor.M if fa else A.MEM
    ep = np.asarray(ep)
    n = len(ep)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]]) if n else np.zeros(0, np.int64)
    lens = np.diff(np.r_[starts, n])
    out = (torch.empty((n, 64, 64), dtype=torch.int8), torch.empty_like(avail),
           torch.empty_like(prev), torch.empty_like(act), torch.empty_like(cell),
           torch.empty((n, M), dtype=torch.float32))
    i = 0
    while i < len(starts):
        if _time_up(deadline):
            return None
        j, tmax = i + 1, int(lens[i])
        while j < len(starts) and (j + 1 - i) * max(tmax, int(lens[j])) <= chunk_frames \
                and (max_episodes is None or j - i < max_episodes):
            tmax = max(tmax, int(lens[j]))
            j += 1
        s, ln = starts[i:j], lens[i:j]
        E, t = j - i, np.arange(tmax)
        pidx = s[:, None] + np.minimum(t[None], ln[:, None] - 1)       # (E,T); padding = last step
        vt = t[None] < ln[:, None]
        ix = torch.as_tensor(pidx)
        o, av, p, y, c = (x[ix].to(device) for x in (obs, avail, prev, act, cell))
        o, av, p, y, c = augment_batch(o.long(), av, p, y, c, kinds, rng)
        vm = torch.as_tensor(vt, device=device)
        ov = o[vm]                                                      # valid frames, row order
        if fa:
            m = torch.zeros(E, M, device=ov.device)
            mems = torch.empty(E, tmax, M, device=ov.device)
            for k in range(tmax):
                mems[:, k] = m                                          # memory entering step k
                run = torch.nonzero(vm[:, k]).flatten()                 # episodes still running
                if M and len(run):
                    m = m.clone()
                    m[run] = actor(o[run, k], m[run], p[run, k], av[run, k])[2]
        else:
            z = torch.cat([heads(actor, *features(actor, ov[k:k + enc_batch]))[0]
                           for k in range(0, len(ov), enc_batch)])
            zp = torch.zeros(E, tmax, z.shape[1], device=z.device, dtype=z.dtype)
            zp[vm] = z
            m = torch.zeros(E, A.MEM, device=z.device, dtype=z.dtype)
            mems = torch.empty(E, tmax, A.MEM, device=z.device, dtype=z.dtype)
            for k in range(tmax):
                mems[:, k] = m                                          # memory entering step k
                _, m = mem_step(actor, zp[:, k], m, p[:, k], av[:, k])
        rows = torch.as_tensor(pidx[vt])
        for dst, src in zip(out, (ov.to(torch.int8), av[vm], p[vm], y[vm], c[vm], mems[vm])):
            dst[rows] = src.cpu()
        i = j
    return out


# --------------------------------------------------------------------------- training
def _bc_loss(logits, click, act, cell, avail):
    logits = logits.masked_fill(~avail, -1e9)
    loss = F.cross_entropy(logits, act)
    sel = cell >= 0
    if sel.any():
        loss = loss + F.cross_entropy(click[sel].flatten(1), cell[sel])
    return loss


def train(student: A.Actor, buf: Buffer, device: str, epochs: int, batch: int, lr: float,
          rng: np.random.Generator, augment: set | None = None, bptt: int | None = None,
          progressive: bool = False, grad_ckpt: int = 0,
          aux_rng: np.random.Generator | None = None, aug_memory: str = "stored",
          deadline: float | None = None, status: dict | None = None) -> float:
    """One DAgger training phase (fresh Adam). Returns the last minibatch loss.
    deadline: wall-clock cap (time.time() value), checked before every epoch and minibatch (and
    between recompute chunks); when passed, training stops and status["truncated"] is set.
    aug_memory: 'stored' (as run) or 'recompute' (see module docstring; needs buf.ep).
    student: wsp.actor.Actor or LearnedFA (learned free-architecture student; progressive and
    grad_ckpt then act on every loop of its graph, see fa_progressive / _fa_exec)."""
    if aug_memory not in AUG_MEMORY:
        raise ValueError(f"aug_memory must be one of {AUG_MEMORY}, got {aug_memory!r}")
    recompute = aug_memory == "recompute"
    if recompute and len(buf.ep) != len(buf):
        raise ValueError("aug_memory='recompute' needs sequence storage (collect(sequences=True))")
    aux = aux_rng if aux_rng is not None else rng
    if not any(a >= 0 for a in buf.act):          # nothing the teacher could label (yet)
        return float("nan")
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    obs = torch.as_tensor(np.stack(buf.obs))       # int8; recompute keeps it compact (CPU memory)
    obs = obs if recompute else obs.long()
    av = torch.as_tensor(np.stack(buf.avail))
    mem = torch.as_tensor(np.stack(buf.mem))
    prev = torch.as_tensor(np.stack(buf.prev))
    act_np = np.array(buf.act)
    act = torch.as_tensor(act_np)
    cell = torch.as_tensor(np.array(buf.cell))
    last = 0.0
    student.train()

    T = depth(student)
    fa = isinstance(student, LearnedFA)
    fwd_window, fwd_step = (fa_forward_window, fa_forward_step) if fa else (forward_window, forward_step)

    def enc():
        if fa:
            e = fa_progressive(aux, student.loop_T) if progressive else {}
        else:
            e = progressive_depth(aux, T) if progressive else {}
        if grad_ckpt:
            e["ckpt"] = grad_ckpt
        return e

    def step(loss):
        nonlocal last
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())

    def epoch_data():
        """(obs, avail, mem, prev, act, cell) for one epoch: the buffer as stored, or (recompute)
        its per-episode transformed copy with memory recomputed by the current student."""
        if not recompute:
            return obs, av, mem, prev, act, cell
        out = transformed_episodes(student, obs, av, prev, act, cell, np.array(buf.ep), augment, aux, device,
                                   deadline=deadline, **({"max_episodes": RECOMPUTE_FA_EPISODES} if fa else {}))
        if out is None:                           # wall-clock deadline passed
            return None
        o_, a_, p_, y_, c_, m_ = out
        return o_, a_, m_, p_, y_, c_

    def loops() -> bool:
        """All epochs; False if stopped by the wall-clock deadline (between minibatches)."""
        if bptt:
            L = int(bptt)
            ep = np.array(buf.ep)
            wb = max(1, batch // L)
            for _ in range(epochs):
                data = None if _time_up(deadline) else epoch_data()
                if data is None:
                    return False
                e_obs, e_av, e_mem, e_prev, e_act, e_cell = data
                idx, valid = make_windows(ep, act_np, L, aux)
                perm = rng.permutation(len(idx))
                for s in range(0, len(idx), wb):
                    if _time_up(deadline):
                        return False
                    ix = torch.as_tensor(idx[perm[s:s + wb]])
                    vm = torch.as_tensor(valid[perm[s:s + wb]])
                    o, a_, p = e_obs[ix], e_av[ix], e_prev[ix]
                    y = torch.where(vm, e_act[ix], -1)
                    c = torch.where(vm, e_cell[ix], -1)
                    o, a_, p, y, c = (t.to(device) for t in (o, a_, p, y, c))
                    if recompute:                     # already transformed, per episode
                        o = o.long()
                    else:
                        o, a_, p, y, c = augment_batch(o, a_, p, y, c, augment, aux)
                    logits, click, _ = fwd_window(student, o, e_mem[ix[:, 0]].to(device), p, a_, **enc())
                    lab = y >= 0
                    step(_bc_loss(logits[lab], click[lab], y[lab], c[lab], a_[lab]))
        else:
            n = len(buf)
            lab_idx = np.flatnonzero(act_np >= 0)
            for _ in range(epochs):
                data = None if _time_up(deadline) else epoch_data()
                if data is None:
                    return False
                e_obs, e_av, e_mem, e_prev, e_act, e_cell = data
                perm = rng.permutation(n) if len(lab_idx) == n else lab_idx[rng.permutation(len(lab_idx))]
                for s in range(0, len(perm), batch):
                    if _time_up(deadline):
                        return False
                    idx = torch.as_tensor(perm[s:s + batch])
                    o, a_, m, p = e_obs[idx], e_av[idx], e_mem[idx], e_prev[idx]
                    y, c = e_act[idx], e_cell[idx]
                    o, a_, m, p, y, c = (t.to(device) for t in (o, a_, m, p, y, c))
                    if recompute:                     # already transformed, per episode
                        o = o.long()
                    elif augment:
                        o, a_, p, y, c = augment_batch(o, a_, p, y, c, augment, aux)
                    logits, click, _ = fwd_step(student, o, m, p, a_, **enc())
                    step(_bc_loss(logits, click, y, c, a_))
        return True

    if not loops() and status is not None:
        status["truncated"] = True
    student.eval()
    return last


def dagger(train_specs: list[EpisodeSpec], teacher, out_dir: Path, seed: int = 0,
           iters: int = 6, epochs: int = 4, batch: int = 64, lr: float = 1e-3,
           device: str = "cpu", max_samples: int = 200_000, augment: set | None = None,
           bptt: int | None = None, progressive: bool = False, grad_ckpt: int = 0,
           aug_memory: str = "stored", student_arch: str | Path | None = None,
           student_bias_init: str = "zero", time_cap_h: float | None = None) -> dict:
    """DAgger from theta_0. Returns {'log', 'params', 'unlabelled_frac', 'teacher', 'config'}.

    time_cap_h: wall-clock cap in hours, measured from the start of this call and checked before
    every collection episode and every training epoch/minibatch. When it passes, collection and
    training stop, the current student is saved as at a normal finish, and dagger_summary.json gets
    truncated / iterations_completed / stopped_in ('collect' | 'train') / stopped_iter / elapsed_h
    (these keys are written whenever a cap is given; None = no cap, unchanged behaviour).

    student_arch: path of a writer FA graph (round<k>.graph.json). The student is then that graph
    with re-initialised parameters (LearnedFA, FA_INIT; biases: FA_BIAS_INIT[student_bias_init])
    instead of the fixed actor, saved as student.graph.json + student_fa.npz (init_fa.npz: the
    initial parameters)."""
    t_start = time.time()
    deadline = None if time_cap_h is None else t_start + float(time_cap_h) * 3600
    if aug_memory not in AUG_MEMORY:
        raise ValueError(f"aug_memory must be one of {AUG_MEMORY}, got {aug_memory!r}")
    sequences = bool(bptt) or aug_memory == "recompute"     # recompute needs whole episodes
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    aux = np.random.default_rng([seed, 1])          # augmentation / progressive / window offsets
    torch.manual_seed(seed)
    if student_arch is None:
        student = A.Actor(init_params(seed)).to(device)
    else:
        arch, written = load_fa_graph(student_arch)
        init = fa_init_params(arch, written, seed, bias=student_bias_init)
        student = LearnedFA(arch, init).to(device)
        np.savez_compressed(out_dir / "init_fa.npz", **init)
    mem_size = student.M if student_arch is not None else A.MEM
    buf = Buffer.new()
    config = {"seed": seed, "iters": iters, "epochs": epochs, "batch": batch, "lr": lr,
              "max_samples": max_samples, "augment": sorted(augment) if augment else None,
              "bptt": bptt, "progressive": progressive, "grad_ckpt": grad_ckpt,
              "aug_memory": aug_memory,
              "teacher": type(teacher).__name__,
              "teacher_budget_ms": getattr(teacher, "budget_ms", None),
              "teacher_reset_fallback": getattr(teacher, "reset_fallback", None)}
    if student_arch is not None:
        config.update({
            "student_arch": str(student_arch),
            "student_arch_sha256": hashlib.sha256(Path(student_arch).read_bytes()).hexdigest(),
            "student_init": FA_INIT + FA_BIAS_INIT[student_bias_init],
            "student_bias_init": student_bias_init, "student_precision": FA_PRECISION,
            "student_count": {k: v for k, v in student.info.items() if k != "shapes"},
            "student_loops": student.loop_T,
            "student_split": {"static_nodes": len(student.static), "dynamic_nodes": len(student.dynamic),
                              "dynamic_loops": [nd["id"] for nd in student.dynamic if nd["op"] == "loop"]},
            **fa_init_probe(student, train_specs, device, ckpt=grad_ckpt or 4)})
    if time_cap_h is not None:
        config["time_cap_h"] = time_cap_h
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=1))
    log = []
    tot_visited = tot_unlab = 0
    t0 = time.time()
    stopped = None                                    # (where, iteration) if the time cap hit
    for it in range(iters + 1):
        beta = 1.0 if it == 0 else 0.5 ** it
        st = collect(train_specs, None if it == 0 else student, teacher, beta, buf, device, rng,
                     sequences=sequences, mem_size=mem_size, deadline=deadline)
        tot_visited += st["visited"]
        tot_unlab += st["unlabelled"]
        if st["truncated"]:
            stopped = ("collect", it)
            break
        if len(buf) > max_samples:
            if sequences:                             # drop whole random episodes
                ep = np.array(buf.ep)
                eps, counts = np.unique(ep, return_counts=True)
                order = nrng.permutation(len(eps))
                keep_eps = set(eps[order[np.cumsum(counts[order]) <= max_samples]].tolist())
                buf.select(k for k in range(len(buf)) if buf.ep[k] in keep_eps)
            else:
                buf.select(nrng.choice(len(buf), max_samples, replace=False))
        t_train = time.time()
        status = {}
        loss = train(student, buf, device, epochs, batch, lr, nrng, augment=augment, bptt=bptt,
                     progressive=progressive, grad_ckpt=grad_ckpt, aux_rng=aux,
                     aug_memory=aug_memory, deadline=deadline, status=status)
        t_train = time.time() - t_train
        if status.get("truncated"):
            stopped = ("train", it)
            break
        log.append({"iter": it, "beta": beta, "samples": len(buf), "loss": loss,
                    "rollout_solved": st["solved"], "rollout_n": st["episodes"],
                    "visited": st["visited"], "unlabelled": st["unlabelled"],
                    "unlabelled_frac": st["unlabelled"] / max(1, st["visited"]),
                    "train_s": round(t_train, 1), "elapsed_s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
    if student_arch is None:
        params = {k: v.detach().cpu().numpy() for k, v in student.state_dict().items() if k in A.SCHEMA}
        A.save_npz(params, out_dir / "student.npz")
    else:
        params = save_fa_student(student, out_dir)
    (out_dir / "dagger_log.json").write_text(json.dumps(log, indent=1))
    tstats = teacher.stats() if hasattr(teacher, "stats") else {}
    summary = {"unlabelled_frac": tot_unlab / max(1, tot_visited), "visited": tot_visited,
               "unlabelled": tot_unlab, "teacher": tstats, "config": config}
    if time_cap_h is not None:
        summary.update({"time_cap_h": time_cap_h, "truncated": stopped is not None,
                        "iterations_completed": len(log),
                        "stopped_in": stopped[0] if stopped else None,
                        "stopped_iter": stopped[1] if stopped else None,
                        "elapsed_h": round((time.time() - t_start) / 3600, 4)})
        if stopped:
            print(json.dumps({"truncated": True, "stopped_in": stopped[0], "stopped_iter": stopped[1],
                              "iterations_completed": len(log), "elapsed_h": summary["elapsed_h"]}), flush=True)
    (out_dir / "dagger_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"unlabelled_frac": summary["unlabelled_frac"], "teacher": tstats}), flush=True)
    teacher.close()
    return {"log": log, "params": params, **summary}

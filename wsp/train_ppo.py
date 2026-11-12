"""PPO into the fixed actor from theta_0 or a written artifact. Sparse completion reward by default.

The critic is a separate training-only CNN (not part of the deployed actor). Episodes are drawn
from the support specs. Default behaviour (all options below off): memory is carried during the
rollout and stored as an input (no BPTT); one Adam over actor + critic.

Options (all off by default):

* ``bptt=L``: recurrent PPO. The rollout horizon of every env is cut into windows of L steps; in
  the update the memory is unrolled from the stored memory at the window start (zeroed at episode
  boundaries), so gradients flow through Wm/Wmz/Wma across steps. A minibatch holds
  ``minibatch // L`` windows (same number of frames as a per-step minibatch). At the first epoch
  the recomputed memory equals the rollout memory, so the PPO ratio starts at 1.
* ``grad_ckpt=s``: checkpoint the 128-iteration tied block in segments of s iterations.
* ``shaping='planner'``: r = r_sparse + scale * (gamma * Phi(s') - Phi(s)),
  Phi(s) = -(tightest certified planner distance to go within ``shaping_budget_ms``), cached by
  state hash (``train_bc.PlannerOracle``). Every episode-ending transition (win, GAME_OVER, cap)
  has Phi(s') = 0 (Ng et al. 1999 terminal convention; the cap is treated as terminal by the GAE
  below). If Phi(s) or Phi(s') cannot be certified the step gets no shaping term; the fraction of
  such steps is logged (``shaping_uncertified_frac``). The shaped return telescopes to the sparse
  return minus Phi(s0), leaving the optimal policy unchanged, only when every potential on the
  trajectory is certified (uncertified fraction 0).
* ``optimizer='sgd'`` (actor; momentum ``sgd_momentum``), ``critic_lr`` (separate Adam for the
  critic; default lr for Adam, 3e-4 for SGD), ``masked`` (only entries nonzero in the init are
  updated), ``kl_to_init`` (coefficient of KL(pi || pi_init) with a frozen copy of the initial
  actor), ``critic_warmup`` (env steps during which only the critic is trained).
* Diagnostics every log interval: entropy, action margin (top1 - top2 logit over available
  actions), KL to the initial policy, gradient norms, non-finite logits/activations (rollout and
  training; non-finite values are sanitised so sampling does not crash, non-finite losses skip the
  step). Greedy evaluation checkpoints (``eval_specs``, ``eval_at``, ``eval_every``) with
  envs.run_episodes + actor.NeuralPolicy; at step 0 the frozen initial actor is also evaluated
  under sampled decoding. Evaluation entries go to ``ppo_eval.json`` (and the returned 'evals'),
  so ``ppo_log.json`` keeps its original schema (every entry has steps/solve_rate_window).
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import actor as A
from .envs import ACTIONS, A_INDEX, CLICK, avail_mask, build_start, obs_of, run_episodes, summarize
from .train_bc import PlannerOracle, forward_step, forward_window, init_params


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 32, 4, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=4)
        self.fc = nn.Linear(64 * 16 + A.MEM + len(ACTIONS), 256)
        self.v = nn.Linear(256, 1)

    def forward(self, obs, mem, prev):
        x = F.one_hot(obs.long(), 16).permute(0, 3, 1, 2).float()
        x = F.relu(self.c2(F.relu(self.c1(x)))).flatten(1)
        return self.v(F.relu(self.fc(torch.cat([x, mem, prev], 1))))[:, 0]


class VecEnv:
    def __init__(self, specs, n, seed, shaped: PlannerOracle | None = None, gamma: float = 0.99,
                 shaping_scale: float = 1.0):
        self.specs, self.n, self.rng = specs, n, random.Random(seed)
        self.games, self.spec, self.steps = [None] * n, [None] * n, [0] * n
        self.mem = np.zeros((n, A.MEM), np.float32)
        self.prev = np.zeros((n, len(ACTIONS)), np.float32)
        self.ep_returns, self.done_eps, self.solved_eps = [], 0, 0
        self.shaped, self.gamma, self.scale = shaped, gamma, shaping_scale
        self.phi: list[float | None] = [None] * n
        self.shaping = Counter()                       # applied / uncertified steps
        for i in range(n):
            self._reset(i)

    def _phi(self, game) -> float | None:
        d = self.shaped.distance(game)
        return None if d is None else -float(d)

    def _reset(self, i):
        s = self.rng.choice(self.specs)
        self.games[i], self.spec[i], self.steps[i] = build_start(s), s, 0
        self.mem[i] = 0
        self.prev[i] = 0
        if self.shaped is not None:
            self.phi[i] = self._phi(self.games[i])

    def obs(self):
        return (np.stack([obs_of(g) for g in self.games]),
                np.stack([avail_mask(g) for g in self.games]))

    def step(self, moves, mem_next):
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, np.float32)
        for i, (a, r, c) in enumerate(moves):
            g, s = self.games[i], self.spec[i]
            _, state, level, _, solved = g.step(a, r, c)
            self.steps[i] += 1
            self.mem[i] = mem_next[i]
            self.prev[i] = 0
            self.prev[i, A_INDEX[a]] = 1
            win = solved or level > s.level
            if win:
                rew[i] = 1.0
            ended = win or state == "GAME_OVER" or self.steps[i] >= s.cap
            if self.shaped is not None:
                phi_next = 0.0 if ended else self._phi(g)
                if self.phi[i] is not None and phi_next is not None:
                    rew[i] += self.scale * (self.gamma * phi_next - self.phi[i])
                    self.shaping["applied"] += 1
                else:
                    self.shaping["uncertified"] += 1
                self.phi[i] = phi_next
            if ended:
                done[i] = 1
                self.done_eps += 1
                self.solved_eps += int(win)
                self._reset(i)
        return rew, done


# --------------------------------------------------------------------------- helpers
def sanitize(logits, click, avail):
    """Replace non-finite available-action / click logits. Returns (logits, click, n_bad_rows)."""
    bad = ~(torch.isfinite(logits) | ~avail).all(-1) | ~torch.isfinite(click).flatten(1).all(-1)
    n = int(bad.sum())
    if n:
        logits = torch.where(avail, torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4),
                             torch.full_like(logits, float("-inf")))
        click = torch.nan_to_num(click, nan=0.0, posinf=1e4, neginf=-1e4)
    return logits, click, n


def policy_kl(lg, ck, lg0, ck0, avail):
    """KL(pi || pi0) of the joint (action, click) distribution per row."""
    lp = F.log_softmax(lg.masked_fill(~avail, -1e9), -1)
    lp0 = F.log_softmax(lg0.masked_fill(~avail, -1e9), -1)
    p = lp.exp()
    kl_a = (p * (lp - lp0)).sum(-1)
    lc = F.log_softmax(ck.flatten(1), -1)
    lc0 = F.log_softmax(ck0.flatten(1), -1)
    kl_c = (lc.exp() * (lc - lc0)).sum(-1)
    return kl_a + p[:, CLICK] * kl_c


def action_margin(lg, avail):
    m = lg.masked_fill(~avail, float("-inf"))
    ok = avail.sum(-1) >= 2
    if not ok.any():
        return float("nan")
    top = m[ok].topk(2, dim=-1).values
    d = top[:, 0] - top[:, 1]
    d = d[torch.isfinite(d)]
    return float(d.mean()) if d.numel() else float("nan")


class SampledPolicy(A.NeuralPolicy):
    """NeuralPolicy with sampled (instead of argmax) action and click decoding."""

    def __init__(self, actor: A.Actor, device: str = "cpu", seed: int = 0):
        super().__init__(actor, device)
        self.gen = torch.Generator().manual_seed(seed)
        self.nonfinite = 0

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
        logits, click, n = sanitize(logits, click, av)
        self.nonfinite += n
        mem2 = torch.nan_to_num(mem2)
        a = torch.multinomial(torch.softmax(logits, 1).cpu(), 1, generator=self.gen)[:, 0].tolist()
        c = torch.multinomial(torch.softmax(click.flatten(1), 1).cpu(), 1, generator=self.gen)[:, 0].tolist()
        moves = [(ACTIONS[k], cc // 64, cc % 64) if k == CLICK else (ACTIONS[k], None, None)
                 for k, cc in zip(a, c)]
        for j, i in enumerate(slots):
            self.mem[i] = mem2[j]
            self.prev[i] = a[j]
        return moves


def eval_schedule(eval_at, eval_every):
    """next_after(x): smallest checkpoint > x (listed steps, then multiples of eval_every)."""
    listed = sorted(set(int(s) for s in (eval_at or ())))
    last = listed[-1] if listed else 0

    def next_after(x):
        c = [s for s in listed if s > x]
        if eval_every:
            k = max(x, last) // eval_every + 1
            c.append(k * eval_every)
        return min(c) if c else math.inf
    return next_after


def _windows(H: int, E: int, L: int):
    """Flat (t*E+e) indices of recurrent windows (E*ceil(H/L), L) and their validity mask."""
    nC = -(-H // L)
    t = np.arange(nC * L).reshape(nC, L)
    valid = t < H
    t = np.minimum(t, H - 1)
    flat = (t[None] * E + np.arange(E)[:, None, None]).reshape(E * nC, L)
    return torch.as_tensor(flat), torch.as_tensor(np.tile(valid, (E, 1)))


# --------------------------------------------------------------------------- PPO
def ppo(train_specs, out_dir: Path, seed: int = 0, total_steps: int = 2_000_000, n_envs: int = 64,
        horizon: int = 128, epochs: int = 4, minibatch: int = 1024, lr: float = 3e-4,
        gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2, ent: float = 0.01,
        device: str = "cuda", init: dict | None = None, log_every: int = 10, *,
        bptt: int | None = None, grad_ckpt: int = 0,
        shaping: str = "none", shaping_budget_ms: float = 200, shaping_scale: float = 1.0,
        optimizer: str = "adam", critic_lr: float | None = None, sgd_momentum: float = 0.0,
        masked: bool = False, kl_to_init: float = 0.0, critic_warmup: int = 0,
        eval_specs=None, eval_at=(), eval_every: int | None = None, eval_batch: int = 64,
        eval_final: bool = True, eval_sampled_init: bool = True, diag_states: int = 256) -> dict:
    if shaping not in ("none", "planner"):
        raise ValueError(f"shaping must be 'none' or 'planner', got {shaping!r}")
    if optimizer not in ("adam", "sgd"):
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = A.Actor(init if init is not None else init_params(seed)).to(device)
    critic = Critic().to(device)
    init_actor = copy.deepcopy(actor).eval()
    for p in init_actor.parameters():
        p.requires_grad_(False)
    masks = {k: (getattr(init_actor, k) != 0).float() for k in A.SCHEMA} if masked else None
    a_params, c_params = list(actor.parameters()), list(critic.parameters())
    if optimizer == "adam" and critic_lr is None:          # default: one Adam over both
        opts, actor_opt = [torch.optim.Adam(a_params + c_params, lr=lr)], None
    else:
        actor_opt = (torch.optim.Adam(a_params, lr=lr) if optimizer == "adam"
                     else torch.optim.SGD(a_params, lr=lr, momentum=sgd_momentum))
        clr = critic_lr if critic_lr is not None else (lr if optimizer == "adam" else 3e-4)
        opts = [actor_opt, torch.optim.Adam(c_params, lr=clr)]
    oracle = PlannerOracle(shaping_budget_ms) if shaping == "planner" else None
    env = VecEnv(train_specs, n_envs, seed, shaped=oracle, gamma=gamma, shaping_scale=shaping_scale)
    config = {k: v for k, v in dict(
        seed=seed, total_steps=total_steps, n_envs=n_envs, horizon=horizon, epochs=epochs,
        minibatch=minibatch, lr=lr, gamma=gamma, lam=lam, clip=clip, ent=ent, bptt=bptt,
        grad_ckpt=grad_ckpt, shaping=shaping, shaping_budget_ms=shaping_budget_ms,
        shaping_scale=shaping_scale, optimizer=optimizer, critic_lr=critic_lr,
        sgd_momentum=sgd_momentum, masked=masked, kl_to_init=kl_to_init,
        critic_warmup=critic_warmup, eval_at=list(eval_at or ()), eval_every=eval_every).items()}
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=1))
    enc = {"ckpt": grad_ckpt} if grad_ckpt else {}

    if eval_specs is not None and not isinstance(eval_specs, dict):
        eval_specs = {"eval": list(eval_specs)}
    log, evals, steps, t0, upd = [], [], 0, time.time(), 0
    diag = Counter()                     # accumulated over a log interval
    gn_max = 0.0

    def save_log():
        (out_dir / "ppo_log.json").write_text(json.dumps(log, indent=1))
        if evals:
            (out_dir / "ppo_eval.json").write_text(json.dumps(evals, indent=1))

    def evaluate(at_steps: int, initial: bool = False):
        for name, specs in (eval_specs or {}).items():
            res = run_episodes(specs, A.NeuralPolicy(actor, device), batch=eval_batch, keep_trace=False)
            entry = {"eval_steps": at_steps, "set": name, "decoding": "greedy",
                     "actor": "init" if initial else "current", **summarize(res)}
            evals.append(entry)
            print(json.dumps(entry), flush=True)
            if initial and eval_sampled_init:
                pol = SampledPolicy(init_actor, device, seed)
                res = run_episodes(specs, pol, batch=eval_batch, keep_trace=False)
                entry = {"eval_steps": at_steps, "set": name, "decoding": "sampled", "actor": "init",
                         "nonfinite_rows": pol.nonfinite, **summarize(res)}
                evals.append(entry)
                print(json.dumps(entry), flush=True)
        actor.train()
        save_log()

    next_after = eval_schedule(eval_at, eval_every)
    next_eval = next_after(-1)
    if eval_specs:
        evaluate(0, initial=True)
        next_eval = next_after(0)

    while steps < total_steps:
        steps_before = steps
        warm = steps_before < critic_warmup
        buf = {k: [] for k in ("obs", "av", "mem", "prev", "a", "cell", "logp", "v", "r", "d")}
        for _ in range(horizon):
            o, av = env.obs()
            ot, avt = torch.as_tensor(o, device=device), torch.as_tensor(av, device=device)
            mt = torch.as_tensor(env.mem, device=device)
            pt = torch.as_tensor(env.prev, device=device)
            with torch.no_grad():
                logits, click, mem2 = actor(ot, mt, pt, avt)
                logits, click, nbad = sanitize(logits, click, avt)
                if nbad or not bool(torch.isfinite(mem2).all()):
                    diag["nonfinite_rollout"] += max(nbad, 1)
                    mem2 = torch.nan_to_num(mem2)
                v = critic(ot, mt, pt)
                da = torch.distributions.Categorical(logits=logits)
                a = da.sample()
                dc = torch.distributions.Categorical(logits=click.flatten(1))
                cell = dc.sample()
                logp = da.log_prob(a) + torch.where(a == CLICK, dc.log_prob(cell), torch.zeros_like(v))
            a_np, c_np = a.cpu().numpy(), cell.cpu().numpy()
            moves = [(ACTIONS[k], int(c) // 64, int(c) % 64) if k == CLICK else (ACTIONS[k], None, None)
                     for k, c in zip(a_np, c_np)]
            for k, val in (("obs", o.astype(np.int8)), ("av", av), ("mem", env.mem.copy()),
                           ("prev", env.prev.copy()), ("a", a_np), ("cell", c_np),
                           ("logp", logp.cpu().numpy()), ("v", v.cpu().numpy())):
                buf[k].append(val)
            r, d = env.step(moves, mem2.cpu().numpy())
            buf["r"].append(r)
            buf["d"].append(d)
            steps += n_envs
        with torch.no_grad():
            o, av = env.obs()
            last_v = critic(torch.as_tensor(o, device=device), torch.as_tensor(env.mem, device=device),
                            torch.as_tensor(env.prev, device=device)).cpu().numpy()
        R, D, V = np.array(buf["r"]), np.array(buf["d"]), np.array(buf["v"])
        adv = np.zeros_like(R)
        gae = 0
        for t in reversed(range(horizon)):
            nv = last_v if t == horizon - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        data = {k: torch.as_tensor(np.concatenate([np.asarray(x) for x in buf[k]]))
                for k in ("obs", "av", "mem", "prev", "a", "cell", "logp")}
        adv_t = torch.as_tensor(adv.reshape(-1))
        ret_t = torch.as_tensor(ret.reshape(-1))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        N = adv_t.shape[0]
        if bptt:
            L = int(bptt)
            win_idx, win_valid = _windows(horizon, n_envs, L)
            reset = torch.zeros(N, dtype=torch.bool)
            reset[n_envs:] = torch.as_tensor(D[:-1].reshape(-1) > 0)
            n_units, per_mb = len(win_idx), max(1, minibatch // L)
        else:
            n_units, per_mb = N, minibatch

        def losses(ix, vm=None):
            """PPO terms on flat indices ix (B,) or windows ix (W,L) with validity vm."""
            dd = {k: data[k][ix].to(device) for k in data}
            ob, avb, mb, pb = dd["obs"].long(), dd["av"], dd["mem"], dd["prev"]
            if vm is not None:
                vm = vm.to(device)
                sel = lambda x: x[vm]
                v_pred = critic(sel(ob), sel(mb), sel(pb))
            else:
                sel = lambda x: x
                v_pred = critic(ob, mb, pb)
            vl = F.mse_loss(v_pred, sel(ret_t[ix].to(device)))
            if warm:
                return 0.5 * vl
            if vm is not None:
                rs = reset[ix].to(device)
                logits, click, _ = forward_window(actor, ob, mb[:, 0], pb, avb, rs, **enc)
                logits, click = sel(logits), sel(click)
            else:
                logits, click, _ = forward_step(actor, ob, mb, pb, avb, **enc)
            avs = sel(avb)
            logits, click, nbad = sanitize(logits, click, avs)
            diag["nonfinite_train"] += nbad
            da = torch.distributions.Categorical(logits=logits)
            dc = torch.distributions.Categorical(logits=click.flatten(1))
            ab, cb = sel(dd["a"]), sel(dd["cell"])
            newlp = da.log_prob(ab) + torch.where(ab == CLICK, dc.log_prob(cb), torch.zeros_like(ab, dtype=torch.float))
            ratio = torch.exp(newlp - sel(dd["logp"]))
            if first_mb:                  # params unchanged since the rollout -> ratio == 1
                diag["ratio0_maxdev"] = max(diag["ratio0_maxdev"], float((ratio - 1).detach().abs().max()))
            ad = sel(adv_t[ix].to(device))
            pl = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
            loss = pl + 0.5 * vl - ent * da.entropy().mean()
            if kl_to_init > 0:
                with torch.no_grad():
                    if vm is not None:
                        lg0, ck0, _ = forward_window(init_actor, ob, mb[:, 0], pb, avb, rs)
                        lg0, ck0 = sel(lg0), sel(ck0)
                    else:
                        lg0, ck0, _ = init_actor(ob, mb, pb, avb)
                    lg0, ck0, _ = sanitize(lg0, ck0, avs)
                kl = policy_kl(logits, click, lg0, ck0, avs).mean()
                loss = loss + kl_to_init * kl
                diag["kl_train_sum"] += float(kl.detach())
            return loss

        first_mb = True
        for _ in range(epochs):
            perm = torch.randperm(n_units)
            for s in range(0, n_units, per_mb):
                idx = perm[s:s + per_mb]
                loss = losses(win_idx[idx], win_valid[idx]) if bptt else losses(idx)
                for opt in opts:
                    opt.zero_grad()
                first_mb = False
                if not bool(torch.isfinite(loss)):
                    diag["skipped_nonfinite_loss"] += 1
                    continue
                loss.backward()
                if masks is not None:
                    for k, m in masks.items():
                        g = getattr(actor, k).grad
                        if g is not None:
                            g.mul_(m)
                ga = [p.grad for p in a_params if p.grad is not None]
                gn_actor = float(torch.norm(torch.stack([g.norm() for g in ga]))) if ga else 0.0
                gn = float(nn.utils.clip_grad_norm_(a_params + c_params, 0.5))
                diag["grad_norm_sum"] += gn
                diag["grad_norm_actor_sum"] += gn_actor
                diag["n_mb"] += 1
                gn_max = max(gn_max, gn)
                for opt in opts:
                    if warm and opt is actor_opt:
                        continue
                    opt.step()
                if masks is not None:
                    with torch.no_grad():
                        for k, m in masks.items():
                            getattr(actor, k).mul_(m)
        upd += 1
        if upd % log_every == 0 or steps >= total_steps:
            rate = env.solved_eps / max(1, env.done_eps)
            entry = {"steps": steps, "episodes": env.done_eps, "solve_rate_window": rate,
                     "elapsed_s": round(time.time() - t0)}
            # diagnostics on evenly spaced states of the last rollout (stored memory as input)
            with torch.no_grad():
                di = torch.linspace(0, N - 1, min(diag_states, N)).long()
                ob, avb = data["obs"][di].long().to(device), data["av"][di].to(device)
                mb, pb = data["mem"][di].to(device), data["prev"][di].to(device)
                lg, ck, _ = actor(ob, mb, pb, avb)
                lg0, ck0, _ = init_actor(ob, mb, pb, avb)
                lg, ck, nb = sanitize(lg, ck, avb)
                lg0, ck0, _ = sanitize(lg0, ck0, avb)
                entry.update({
                    "entropy": float(torch.distributions.Categorical(logits=lg).entropy().mean()),
                    "margin": action_margin(lg, avb),
                    "kl_init": float(policy_kl(lg, ck, lg0, ck0, avb).mean()),
                    "nonfinite_diag": nb})
            nmb = max(1, diag["n_mb"])
            entry.update({"grad_norm": diag["grad_norm_sum"] / nmb,
                          "grad_norm_actor": diag["grad_norm_actor_sum"] / nmb,
                          "grad_norm_max": gn_max,
                          "nonfinite_rollout": diag["nonfinite_rollout"],
                          "nonfinite_train": diag["nonfinite_train"],
                          "skipped_nonfinite_loss": diag["skipped_nonfinite_loss"],
                          "ratio0_maxdev": diag["ratio0_maxdev"],
                          "critic_warmup": warm})
            if kl_to_init > 0 and not warm:
                entry["kl_train"] = diag["kl_train_sum"] / nmb
            if oracle is not None:
                tot = env.shaping["applied"] + env.shaping["uncertified"]
                entry.update({"shaping_uncertified_frac": env.shaping["uncertified"] / max(1, tot),
                              "shaping_steps": tot, "planner_queries": oracle.stats["queries"],
                              "planner_cache_hits": oracle.stats["cache_hits"],
                              "planner_ms": oracle.stats["planner_ms"],
                              "planner_cutoffs": oracle.stats["cutoffs"]})
                env.shaping.clear()
            diag.clear()
            gn_max = 0.0
            log.append(entry)
            env.done_eps = env.solved_eps = 0
            print(json.dumps(entry), flush=True)
            params = {k: v.detach().cpu().numpy() for k, v in actor.state_dict().items() if k in A.SCHEMA}
            A.save_npz(params, out_dir / "student_latest.npz")
            save_log()
        if eval_specs and steps >= next_eval:
            evaluate(steps)
            next_eval = next_after(steps)
    if eval_specs and eval_final and not any(e["eval_steps"] == steps for e in evals):
        evaluate(steps)
    params = {k: v.detach().cpu().numpy() for k, v in actor.state_dict().items() if k in A.SCHEMA}
    A.save_npz(params, out_dir / "student.npz")
    save_log()
    return {"log": log, "evals": evals, "params": params, "config": config}

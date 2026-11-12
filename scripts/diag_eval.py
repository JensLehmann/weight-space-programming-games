"""Revision-2 diagnostics.

  floors <family>...        replay floor + scripted floors per family (rev-2 manifests)
  artifacts <root>          per-artifact: first-frame blindness, counterfactual-frame sensitivity,
                            RESET-disabled perturbed starts  -> diag.json next to eval.json
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of, rev_of  # noqa: E402

from wsp import actor as A
from wsp.envs import ACTIONS, A_INDEX, avail_mask, build_start, obs_of, run_episodes, summarize
from wsp.queries import load_sets, certify, _effective_moves
from arc3_games.api import evaluate_state


# ---------------------------------------------------------------- policies used as floors/diagnostics
class FirstFrame:
    def __init__(self, inner):
        self.inner, self.first = inner, {}

    def begin(self, slots):
        self.inner.begin(slots)
        for i in slots:
            self.first.pop(i, None)

    def act(self, slots, obs, avail):
        o = obs.copy()
        for j, i in enumerate(slots):
            if i not in self.first:
                self.first[i] = obs[j].copy()
            o[j] = self.first[i]
        return self.inner.act(slots, o, avail)


class NoReset:
    def __init__(self, inner):
        self.inner = inner

    def begin(self, slots):
        self.inner.begin(slots)

    def act(self, slots, obs, avail):
        av = avail.copy()
        av[:, A_INDEX["RESET"]] = False
        return self.inner.act(slots, obs, av)


class Replay:
    """Replays a fixed action list per slot (support-plan replay floor)."""
    def __init__(self, plans):
        self.plans, self.ptr = plans, {}

    def begin(self, slots):
        for i in slots:
            self.ptr[i] = 0

    def act(self, slots, obs, avail):
        out = []
        for i in slots:
            plan = self.plans[i]
            k = self.ptr[i]
            out.append(tuple(plan[k]) if k < len(plan) else ("ACTION1", None, None))
            self.ptr[i] = k + 1
        return out


class Schedule:
    """Open-loop cyclic schedule of actions (clicks at a fixed cell)."""
    def __init__(self, moves):
        self.moves, self.t = moves, {}

    def begin(self, slots):
        for i in slots:
            self.t[i] = 0

    def act(self, slots, obs, avail):
        out = []
        for i in slots:
            out.append(self.moves[self.t[i] % len(self.moves)])
            self.t[i] += 1
        return out


class RandomEffective:
    def __init__(self, specs, seed=0):
        self.rng, self.games, self.specs = random.Random(seed), {}, specs

    def begin(self, slots):
        pass

    def act(self, slots, obs, avail):  # resolved via env replay below
        raise NotImplementedError


def plan_of(spec):
    g = build_start(spec)
    r = evaluate_state(g, time_budget_ms=5000)
    return [(p.action, p.row, p.col) for p in (r.direct_plan or r.plan or ())]


def nearest_support(q_obs, sup):
    best, bd = None, None
    for s, o in sup:
        d = int((o != q_obs).sum())
        if bd is None or d < bd:
            best, bd = s, d
    return best


def floors(fam):
    qs = load_sets(fam, rev=2)
    sup = [(s, obs_of(build_start(s))) for s in qs["S-start"]]
    plans = {s.key: plan_of(s) for s, _ in sup}
    out = {}
    for qset in ("Q-lvl", "Q-off"):
        specs = qs[qset]
        replay = {}
        for i, s in enumerate(specs):
            near = nearest_support(obs_of(build_start(s)), sup)
            replay[i] = plans[near.key]
        res = run_episodes(specs, Replay(replay), batch=64, keep_trace=False)
        out[f"{qset}/replay"] = summarize(res)["solve_rate"]
        # best single-action / pair open-loop schedules (clicks at the click-cell of the support plan)
        g = build_start(specs[0])
        names = [a for a in g.available_actions if a not in ("RESET", "ACTION6")]
        scheds = [[(a, None, None)] for a in names] + [[(a, None, None), (b, None, None)] for a in names for b in names if a < b]
        best = 0.0
        for sch in scheds[:40]:
            r = summarize(run_episodes(specs, Schedule(sch), batch=64, keep_trace=False))["solve_rate"]
            best = max(best, r)
        out[f"{qset}/best_open_loop"] = best
    return out


def counterfactual_sensitivity(params, variant, specs, n=40, seed=0, net=None, mem_dim=A.MEM):
    """Fraction of decisions whose action changes when the frame is swapped for another episode's frame.
    `net` (with `mem_dim`) replaces the fixed Actor, e.g. a wsp.freearch.FreeArchNet."""
    import torch
    rng = random.Random(seed)
    net = A.Actor(params, variant=variant).eval() if net is None else net
    frames = [obs_of(build_start(s)) for s in specs[: min(len(specs), 200)]]
    changed, total = 0, 0
    for s in rng.sample(specs, min(n, len(specs))):
        g = build_start(s)
        mem = torch.zeros(1, mem_dim)
        prev = torch.zeros(1, len(ACTIONS))
        for _ in range(min(s.cap, 40)):
            o, av = obs_of(g), avail_mask(g)
            with torch.no_grad():
                lg, ck, m2 = net(torch.as_tensor(o[None]), mem, prev, torch.as_tensor(av[None]))
                cf = rng.choice(frames)
                lg2, ck2, _ = net(torch.as_tensor(cf[None]), mem, prev, torch.as_tensor(av[None]))
            a1, a2 = A.decode(lg, ck)[0], A.decode(lg2, ck2)[0]
            changed += int(a1 != a2)
            total += 1
            if a1[0] not in A_INDEX:
                break
            _, st, lvl, _, solved = g.step(*a1)
            mem = m2
            prev = torch.zeros(1, len(ACTIONS)); prev[0, A_INDEX[a1[0]]] = 1
            if solved or lvl > s.level or st != "NOT_FINISHED":
                break
    return changed / max(1, total)


def artifacts(root):
    import torch
    # exact FP32 semantics (TF32 convolutions flip argmax decisions of high-gain written actors)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from wsp.session import load_final, evaluate
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for sp in sorted(set(list(Path(root).rglob("session.json")) + list(Path(root).rglob("student.npz")))):
        d = sp.parent
        if (d / "diag.json").exists() or not (d / "eval.json").exists():
            continue
        rec = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else None
        fam = family_of(d)
        qs = load_sets(fam, rev=rev_of(d, rec))
        if rec is not None:
            art = load_final(d, rec["condition"])
        else:
            from wsp.session import Artifact
            art = Artifact("weights", "", params=A.load_npz(d / "student.npz"))
        if art is None:
            continue
        out = {}
        pert = [s for s in qs["Q-lvl"] + qs["Q-off"] if s.prefix]
        if art.condition in ("weights", "edit"):
            pol = lambda: A.NeuralPolicy(A.Actor(art.params, variant=art.variant), dev)
            out["Q-lvl/first_frame"] = summarize(run_episodes(qs["Q-lvl"], FirstFrame(pol()), batch=64, keep_trace=False))["solve_rate"]
            out["perturbed/no_reset"] = summarize(run_episodes(pert, NoReset(pol()), batch=64, keep_trace=False))["solve_rate"]
            out["cf_sensitivity"] = counterfactual_sensitivity(art.params, art.variant, qs["Q-lvl"])
        elif art.condition == "freearch":
            from wsp import freearch
            pol = lambda: freearch.policy(art, dev)
            out["Q-lvl/first_frame"] = summarize(run_episodes(qs["Q-lvl"], FirstFrame(pol()), batch=64, keep_trace=False))["solve_rate"]
            out["perturbed/no_reset"] = summarize(run_episodes(pert, NoReset(pol()), batch=64, keep_trace=False))["solve_rate"]
            net = freearch.FreeArchNet(art.graph, art.params).eval()
            out["cf_sensitivity"] = counterfactual_sensitivity(None, None, qs["Q-lvl"], net=net, mem_dim=net.M)
        else:
            from wsp.session import policy_for
            p1 = policy_for(art); out["perturbed/no_reset"] = summarize(run_episodes(pert, NoReset(p1), batch=16, keep_trace=False))["solve_rate"]; p1.close()
        (d / "diag.json").write_text(json.dumps(out))
        print(d, out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "floors":
        from wsp.manifest import RUNS
        outd = RUNS / "rev2_floors"; outd.mkdir(parents=True, exist_ok=True)
        for fam in sys.argv[2:]:
            r = floors(fam)
            (outd / f"{fam}.json").write_text(json.dumps(r))
            print(fam, r, flush=True)
    else:
        artifacts(sys.argv[2])

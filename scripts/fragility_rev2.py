"""Fragility controls for the perturbation result. No writer (model) calls.

Three questions about perturbation_rev2.py (isotropic noise on the 18 written W2 fixed actors):
  nz     Is the collapse caused by noise on the exact zeros (93% of entries)? Noise on nonzero entries only:
         theta_k' = theta_k + eps * rms_nz(theta_k) * (z_k * [theta_k != 0]), rms over the nonzero entries of
         tensor k, so ||theta' - theta|| / ||theta|| ~= eps per tensor and overall, as for the isotropic noise;
         same z (perturbation_rev2.noise_direction), eps grid, seeds, sets and subsample.
  fa     Are free-architecture (FA) networks equally fragile? The final FA W2 networks (run 0) of the same
         writers and families (18), isotropic noise (perturbation_rev2.perturb over all tensors) and
         nonzero-only noise; eps = 0 on the subsample and on the full sets (must equal eval.json).
  ties   Do decisions rest on exact ties or float32-scale margins? Unperturbed fixed and FA actors on the
         full S-start and Q-lvl sets: per decision the gap between the two largest available action logits
         (and, for clicks, between the two largest click logits), exact ties, gaps relative to the largest
         |logit|; then the same episodes with the opposite tie-break (highest action index, last cell),
         which changes an outcome only where a decision was an exact tie.
  analyze  writes runs/analysis_rev2/fragility.json and tables/fragility.tex (Q-lvl level-balanced on the
         subsample, family bootstrap as in perturbation_rev2).

Usage: fragility_rev2.py run [--workers 3] [--device mps] [--only ties,nz,fa], then analyze.
"""
import argparse
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import perturbation_rev2 as P  # noqa: E402

R, ROOT = P.R, P.ROOT
WORK = R / "rev2_fragility"
FAMS, WRITERS, EPS, SEEDS = P.FAMS, P.WRITERS, P.EPS, P.NOISE_SEEDS


def fa_dir(fam, w):
    return R / "rev2_fa" / fam / w / "freearch_feedback_M_r0"


def load(kind, fam, w):
    """(params, arch or None) of the final artifact."""
    if kind == "fixed":
        return P.load_actor(fam, w), None
    from wsp import freearch as FA
    d = fa_dir(fam, w)
    rec = json.loads((d / "session.json").read_text())
    arch, params = FA.load_net(d, rec["final_round"])
    return params, arch


def evaljson(kind, fam, w):
    d = P.actor_dir(fam, w) if kind == "fixed" else fa_dir(fam, w)
    return json.loads((d / "eval.json").read_text())


def policy(kind, params, arch, device):
    from wsp import actor as A
    if kind == "fixed":
        return A.NeuralPolicy(A.Actor(params), device)
    from wsp import freearch as FA
    return FA.FreeArchPolicy(FA.FreeArchNet(arch, params), device)


def run_eval(kind, params, arch, sets, device):
    from wsp.envs import run_episodes
    if not all(np.isfinite(v).all() for v in params.values()):
        return {s: [[sp.key, 0, 0, 0, 0] for sp in specs] for s, specs in sets.items()}, True
    out = {}
    for s, specs in sets.items():
        res = run_episodes(specs, policy(kind, params, arch, device), batch=64, keep_trace=False)
        out[s] = [[r.key, int(r.success), r.actions, r.resets, r.invalid] for r in res]
    return out, False


def perturb_nz(params, eps, z):
    out = {}
    for k, v in params.items():
        v64 = v.astype(np.float64)
        m = v64 != 0
        rms = float(np.sqrt(np.mean(v64[m] ** 2))) if m.any() else 0.0
        out[k] = (v64 + eps * rms * z[k] * m).astype(np.float32)
    return out


# --------------------------------------------------------------------------- noise jobs
def noise_job(args):
    kind, mode, fam, w, eps, seed, which, device = args
    name = f"eps0_{which}.json" if eps == 0 else f"{mode}_eps{eps:g}_s{seed}.json"
    out = WORK / kind / fam / w / name
    if out.exists():
        return str(out), "cached"
    try:
        P.setup_torch()
        t0 = time.time()
        p0, arch = load(kind, fam, w)
        sets = P.eval_sets(fam)
        size = None
        if eps == 0:
            p = p0
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl-full" if which == "full" else "Q-lvl"]}
        else:
            z = P.noise_direction(p0, seed, f"{fam}/{w}" if kind == "fixed" else f"fa/{fam}/{w}")
            p = P.perturb(p0, eps, z) if mode == "iso" else perturb_nz(p0, eps, z)
            size = {k: v for k, v in P.rel_change(p0, p).items() if k != "per_tensor"}
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl"]}
        res, nonfinite = run_eval(kind, p, arch, sets, device)
        rec = {"kind": kind, "mode": mode, "family": fam, "writer": w, "eps": eps, "seed": seed, "which": which,
               "device": device, "episodes": res, "nonfinite_params": nonfinite, "size": size,
               "wall_s": time.time() - t0}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return str(out), {"S-start": P.rate(res["S-start"]), "Q-lvl": P.qlvl_balanced(fam, res["Q-lvl"]),
                          "wall_s": round(rec["wall_s"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


# --------------------------------------------------------------------------- margins and tie-break
class Recorder:
    """Replacement for wsp.actor.decode: records decision margins; optional opposite tie-break."""

    def __init__(self, flip):
        self.flip = flip
        self.reset()

    def reset(self):
        self.n = self.single = self.tie = self.click_n = self.click_tie = 0
        self.gaps, self.rel, self.click_gaps, self.click_rel = [], [], [], []

    def __call__(self, logits, click):
        import torch
        from wsp import actor as A
        lg = logits.detach().cpu().double().numpy()     # MPS has no float64
        ck = click.detach().flatten(1).cpu().double().numpy()
        if self.flip:
            a_idx = [lg.shape[1] - 1 - int(np.argmax(r[::-1])) for r in lg]
            flat = [ck.shape[1] - 1 - int(np.argmax(r[::-1])) for r in ck]
        else:
            a_idx = torch.argmax(logits, 1).tolist()
            flat = torch.argmax(click.flatten(1), 1).tolist()
        moves = []
        for j, (a, cell) in enumerate(zip(a_idx, flat)):
            r = lg[j][np.isfinite(lg[j])]
            if len(r) >= 2:
                top = np.sort(r)[-2:]
                gap = float(top[1] - top[0])
                self.n += 1
                self.tie += gap == 0
                self.gaps.append(gap)
                self.rel.append(gap / max(abs(top[1]), abs(top[0]), 1e-30))
            else:
                self.single += 1
            if a == A.CLICK:
                c = ck[j][np.isfinite(ck[j])]
                if len(c) >= 2:
                    top = np.partition(c, -2)[-2:]
                    top.sort()
                    gap = float(top[1] - top[0])
                    self.click_n += 1
                    self.click_tie += gap == 0
                    self.click_gaps.append(gap)
                    self.click_rel.append(gap / max(abs(top[1]), abs(top[0]), 1e-30))
                moves.append((A.ACTIONS[a], cell // 64, cell % 64))
            else:
                moves.append((A.ACTIONS[a], None, None))
        return moves

    def summary(self):
        q = lambda xs: [float(np.quantile(xs, p)) for p in (0.01, 0.1, 0.5)] if xs else None
        frac = lambda xs, t: float(np.mean(np.array(xs) < t)) if xs else None
        return {"decisions": self.n, "single_available": self.single, "exact_ties": self.tie,
                "gap_q01_q10_q50": q(self.gaps), "rel_gap_q01_q10_q50": q(self.rel),
                "frac_rel_gap_lt_1e-6": frac(self.rel, 1e-6), "frac_rel_gap_lt_1e-4": frac(self.rel, 1e-4),
                "click_decisions": self.click_n, "click_exact_ties": self.click_tie,
                "click_gap_q01_q10_q50": q(self.click_gaps), "click_rel_gap_q01_q10_q50": q(self.click_rel),
                "click_frac_rel_gap_lt_1e-6": frac(self.click_rel, 1e-6)}


def ties_job(args):
    kind, fam, w, device = args
    out = WORK / kind / fam / w / "ties.json"
    if out.exists():
        return str(out), "cached"
    try:
        P.setup_torch()
        from wsp import actor as A
        t0 = time.time()
        p0, arch = load(kind, fam, w)
        s = P.eval_sets(fam)
        sets = {"S-start": s["S-start"], "Q-lvl": s["Q-lvl-full"]}
        rec = {"kind": kind, "family": fam, "writer": w, "device": device}
        for tag, flip in (("lowest", False), ("highest", True)):
            r = Recorder(flip)
            A.decode = r
            res, _ = run_eval(kind, p0, arch, sets, device)
            rec[tag] = {"episodes": res, "margins": r.summary()}
        ev = evaljson(kind, fam, w)
        ref = {st: {e[0]: e[1:] for e in ev["episodes"][st]} for st in sets}
        rec["lowest_equals_evaljson"] = {st: sum(e[1:] != ref[st][e[0]] for e in rec["lowest"]["episodes"][st])
                                         for st in sets}
        rec["outcome_changed_by_flip"] = {
            st: sum(a[1] != b[1] for a, b in zip(rec["lowest"]["episodes"][st], rec["highest"]["episodes"][st]))
            for st in sets}
        rec["wall_s"] = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        m = rec["lowest"]["margins"]
        return str(out), {"ties": m["exact_ties"], "decisions": m["decisions"], "click_ties": m["click_exact_ties"],
                          "flip_changed": rec["outcome_changed_by_flip"], "mismatch": rec["lowest_equals_evaljson"],
                          "wall_s": round(rec["wall_s"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


def run_job(args):
    # pool workers are reused: ties_job replaces wsp.actor.decode, so every job starts from the original
    from wsp import actor as A
    if not hasattr(A, "_orig_decode"):
        A._orig_decode = A.decode
    A.decode = A._orig_decode
    try:
        return ties_job(args[1:]) if args[0] == "ties" else noise_job(args[1:])
    finally:
        A.decode = A._orig_decode


def jobs(only, device, seeds=SEEDS):
    J = []
    if "ties" in only:
        J += [("ties", k, f, w, device) for k in ("fixed", "fa") for f in FAMS for w in WRITERS]
    if "fa" in only:
        J += [("noise", "fa", "none", f, w, 0.0, 0, which, device) for f in FAMS for w in WRITERS for which in ("sub", "full")]
    for s in seeds:                     # seed 0 of every condition first, then seed 1
        if "nz" in only:
            J += [("noise", "fixed", "nz", f, w, e, s, "sub", device) for e in EPS for f in FAMS for w in WRITERS]
        if "fa" in only:
            J += [("noise", "fa", m, f, w, e, s, "sub", device) for m in ("iso", "nz") for e in EPS
                  for f in FAMS for w in WRITERS]
    return J


# --------------------------------------------------------------------------- analysis
def analyze(seeds=SEEDS):
    res = {"design": __doc__.split("\n\n")[1].strip(), "noise_seeds": seeds, "noise": {}, "ties": {}, "sanity": {}}
    base_fixed = {(f, w): P.qlvl_balanced(f, json.loads((P.WORK / "noise" / f / w / "eps0_sub.json").read_text())["episodes"]["Q-lvl"])
                  for f in FAMS for w in WRITERS}
    base_fa, fa_mism = {}, []
    for f in FAMS:
        for w in WRITERS:
            d = WORK / "fa" / f / w
            base_fa[(f, w)] = P.qlvl_balanced(f, json.loads((d / "eps0_sub.json").read_text())["episodes"]["Q-lvl"])
            full = json.loads((d / "eps0_full.json").read_text())
            ev = evaljson("fa", f, w)
            ref = {s: {e[0]: e[1:] for e in ev["episodes"][s]} for s in ("S-start", "Q-lvl")}
            fa_mism.append(sum(e[1:] != ref[s][e[0]] for s in ("S-start", "Q-lvl") for e in full["episodes"][s]))
    res["sanity"]["fa_eps0_full_mismatching_episodes"] = int(sum(fa_mism))

    def cond(kind, mode, base):
        out = {"0": {"Q-lvl": P.boot_family(_fam({k: [v] for k, v in base.items()}))}}
        for e in EPS:
            vals, sizes = defaultdict(list), []
            for f in FAMS:
                for w in WRITERS:
                    if kind == "fixed" and mode == "iso":
                        fs = [P.WORK / "noise" / f / w / f"eps{e:g}_s{s}.json" for s in SEEDS]   # as in Table perturbation
                    else:
                        fs = [WORK / kind / f / w / f"{mode}_eps{e:g}_s{s}.json" for s in seeds]
                    rs = [json.loads(p.read_text()) for p in fs if p.exists()]
                    if rs:
                        vals[f].append(float(np.mean([P.qlvl_balanced(f, r["episodes"]["Q-lvl"]) for r in rs])))
                        sizes += [r["size"]["overall"] for r in rs if r.get("size")]
            out[f"{e:g}"] = {"Q-lvl": P.boot_family(vals), "realised_rel_overall_mean": float(np.mean(sizes)) if sizes else None}
        return out

    res["noise"]["fixed_iso"] = cond("fixed", "iso", base_fixed)
    res["noise"]["fixed_nz"] = cond("fixed", "nz", base_fixed)
    res["noise"]["fa_iso"] = cond("fa", "iso", base_fa)
    res["noise"]["fa_nz"] = cond("fa", "nz", base_fa)
    for kind in ("fixed", "fa"):
        rows = []
        for f in FAMS:
            for w in WRITERS:
                p = WORK / kind / f / w / "ties.json"
                if p.exists():
                    t = json.loads(p.read_text())
                    rows.append({"family": f, "writer": w, **t["lowest"]["margins"],
                                 "flip_changed": t["outcome_changed_by_flip"], "mismatch": t["lowest_equals_evaljson"],
                                 "episodes": sum(len(v) for v in t["lowest"]["episodes"].values())})
        if rows:
            res["ties"][kind] = {
                "actors": len(rows), "decisions": sum(r["decisions"] for r in rows),
                "exact_ties": sum(r["exact_ties"] for r in rows),
                "click_decisions": sum(r["click_decisions"] for r in rows),
                "click_exact_ties": sum(r["click_exact_ties"] for r in rows),
                "actors_with_any_tie": sum(r["exact_ties"] + r["click_exact_ties"] > 0 for r in rows),
                "episodes": sum(r["episodes"] for r in rows),
                "episodes_changed_by_flip": sum(sum(r["flip_changed"].values()) for r in rows),
                "evaljson_mismatching_episodes": sum(sum(r["mismatch"].values()) for r in rows),
                "median_rel_gap_q01": float(np.median([r["rel_gap_q01_q10_q50"][0] for r in rows if r["rel_gap_q01_q10_q50"]])),
                "median_rel_gap_q50": float(np.median([r["rel_gap_q01_q10_q50"][2] for r in rows if r["rel_gap_q01_q10_q50"]])),
                "rows": rows}
    res["table_tex"] = table(res)
    (R / "analysis_rev2" / "fragility.json").write_text(json.dumps(res, indent=1, default=float))
    (R / "analysis_rev2" / "tables" / "fragility.tex").write_text(res["table_tex"])
    print(res["table_tex"])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res["ties"].items()}, indent=1))
    print("sanity:", res["sanity"])


def _fam(d):
    out = defaultdict(list)
    for (f, w), v in d.items():
        out[f] += v
    return out


def table(res):
    cols = ["0"] + [f"{e:g}" for e in EPS]
    head = " & ".join("0" if c == "0" else f"$10^{{{int(round(math.log10(float(c))))}}}$" for c in cols)
    L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         r"Held-out Q-lvl at relative noise $\epsilon$ & " + head + r"\\", r"\midrule"]
    names = {"fixed_iso": "Fixed actor, all entries", "fixed_nz": "Fixed actor, nonzero entries only",
             "fa_iso": "FA network, all entries", "fa_nz": "FA network, nonzero entries only"}
    for k, lab in names.items():
        n = res["noise"][k]
        L.append(lab + " & " + " & ".join(f"{n[c]['Q-lvl']['mean']:.2f}" if n.get(c) and n[c]["Q-lvl"]["mean"] is not None
                                          else "--" for c in cols) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--only", default="ties,nz,fa")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if a.cmd == "run":
        J = jobs(set(a.only.split(",")), a.device, [int(x) for x in a.seeds.split(",")])
        print(f"# {len(J)} jobs", flush=True)
        P.run_pool(run_job, J, a.workers)
    else:
        analyze([int(x) for x in a.seeds.split(",")])

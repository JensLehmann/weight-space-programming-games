"""Robust-weights evaluation: noise curves of margin-robust W2 actors against standard W2 actors. No writer calls.

Robust actors: runs/rev2_robust/<fam>/<w>/weights_feedback_M_robust_r<k> (all finished runs).
Reference: the standard W2 actors runs/rev2_private/<fam>/<w>/weights_feedback_M_full_r<k>, same runs.
Noise, subsample and scoring are those of perturbation_rev2.py (isotropic per-tensor-RMS noise on every
entry, eps grid EPS, seeds 0 and 1, the fixed level-stratified Q-lvl subsample, level-balanced Q-lvl).
eps = 0 is scored on the full S-start and Q-lvl sets.
Exact held-out results per run (eval.json) come from eval_artifacts.py runs/rev2_robust.

Usage: eval_robust.py run [--workers 8] [--device cpu]   then   eval_robust.py analyze
Output: runs/rev2_robust/_noise/<kind>/<fam>/<w>/r<k>/eps*.json, runs/analysis_rev2/robust.json
"""
import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import perturbation_rev2 as P  # noqa: E402

R = P.R
ROBUST = R / "rev2_robust"
WORK = ROBUST / "_noise"
FAMS = ["ztl", "lys", "esm", "ruz", "crg", "mds"]
WRITERS = ["astra-6", "sol-6.1", "opus-5.5"]
EPS = [0.0] + P.EPS
SEEDS = P.NOISE_SEEDS
NAMES = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5"}
DIRS = {"robust": lambda f, w, k: ROBUST / f / w / f"weights_feedback_M_robust_r{k}",
        "standard": lambda f, w, k: R / "rev2_private" / f / w / f"weights_feedback_M_full_r{k}"}


def load(d):
    from wsp import actor as A
    rec = json.loads((d / "session.json").read_text())
    if rec.get("final_round") is None:
        return None
    return A.load_npz(d / f"round{rec['final_round']}.npz")


def job(args):
    kind, fam, w, k, eps, seed, device = args
    out = WORK / kind / fam / w / f"r{k}" / (f"eps0.json" if eps == 0 else f"eps{eps:g}_s{seed}.json")
    if out.exists():
        return str(out), "cached"
    try:
        P.setup_torch()
        t0 = time.time()
        p0 = load(DIRS[kind](fam, w, k))
        sets = P.eval_sets(fam)
        if eps == 0 or p0 is None:
            p = p0
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl-full"]}
        else:
            p = P.perturb(p0, eps, P.noise_direction(p0, seed, f"{fam}/{w}"))
            sets = {"S-start": sets["S-start"], "Q-lvl": sets["Q-lvl"]}
        res, nonfinite = P.run_eval(p, sets, device)
        rec = {"kind": kind, "family": fam, "writer": w, "run": k, "eps": eps, "seed": seed,
               "no_artifact": p0 is None, "nonfinite_params": nonfinite, "episodes": res,
               "wall_s": time.time() - t0}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return str(out), {"Q-lvl": P.qlvl_balanced(fam, res["Q-lvl"]), "wall_s": round(rec["wall_s"])}
    except Exception:
        return str(out), "ERROR " + traceback.format_exc()[-1500:]


def jobs(device):
    out = []
    for f in FAMS:
        for w in WRITERS:
            for k in range(3):
                if not (DIRS["robust"](f, w, k) / "session.json").exists():
                    continue
                for kind in ("robust", "standard"):
                    if not (DIRS[kind](f, w, k) / "session.json").exists():
                        continue
                    for eps in EPS:
                        for s in ([0] if eps == 0 else SEEDS):
                            out.append((kind, f, w, k, eps, s, device))
    return out


def analyze():
    by = defaultdict(list)          # (kind, w, eps) -> per-actor Q-lvl (mean over seeds)
    per_actor = defaultdict(list)
    sub = {}
    for p in WORK.glob("*/*/*/r*/eps*.json"):
        r = json.loads(p.read_text())
        eps_q = r["episodes"]["Q-lvl"]
        if r["eps"] == 0:          # eps0 ran on the full set; also score it on the noisy points' subsample
            f = r["family"]
            if f not in sub:
                sub[f] = {s.key for s in P.eval_sets(f)["Q-lvl"]}
            per_actor[(r["kind"], f, r["writer"], r["run"], "0sub")].append(
                P.qlvl_balanced(f, [e for e in eps_q if e[0] in sub[f]]) or 0.0)
        per_actor[(r["kind"], r["family"], r["writer"], r["run"], r["eps"])].append(
            P.qlvl_balanced(r["family"], eps_q) or 0.0)
    fam_means = defaultdict(list)   # family macro: mean over runs within family first
    tmp = defaultdict(list)
    for (kind, f, w, k, eps), v in per_actor.items():
        tmp[(kind, f, w, eps)].append(float(np.mean(v)))
    for (kind, f, w, eps), v in tmp.items():
        fam_means[(kind, w, eps)].append(float(np.mean(v)))
        fam_means[(kind, "pooled", eps)].append(float(np.mean(v)))
    lab = lambda e: e if isinstance(e, str) else f"{e:g}"
    table = {f"{kind}|{w}|{lab(eps)}": round(float(np.mean(v)), 3)
             for (kind, w, eps), v in sorted(fam_means.items(), key=lambda kv: str(kv[0]))}
    out = R / "analysis_rev2" / "robust.json"
    out.write_text(json.dumps({"eps": EPS, "qlvl_family_macro": table,
                               "n_actors": len({k[:4] for k in per_actor})}, indent=1))
    nfam = defaultdict(set)
    for (kind, f, w, eps) in tmp:
        nfam[(kind, w)].add(f)
    cols = [0.0] + P.EPS
    body = ["\\begin{tabular}{llrrrrrrrr}", "\\toprule",
            "Prompt & Writer (families) & 0 & " + " & ".join(f"$10^{{{int(np.log10(e))}}}$" for e in P.EPS) + "\\\\",
            "\\midrule"]
    for i, w in enumerate(WRITERS):
        if i:
            body.append("\\midrule")
        for kind, label in (("standard", "Standard"), ("robust", "Robust")):
            vals = " & ".join(f"{table[f'{kind}|{w}|{lab(e)}']:.2f}" for e in cols)
            body.append(f"{label} & {NAMES[w]} ({len(nfam[(kind, w)])}) & {vals}\\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    tdir = R / "analysis_rev2" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "robust.tex").write_text("\n".join(body) + "\n")
    for kind in ("standard", "robust"):
        for w in WRITERS + ["pooled"]:
            print(f"{kind:9s} {w:9s}", " ".join(f"{table.get(f'{kind}|{w}|{lab(e)}', float('nan')):5.2f}"
                                               for e in ["0sub"] + EPS))
    print("eps                ", " ".join(f"{lab(e):>5s}" for e in ["0sub"] + EPS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze()
        sys.exit(0)
    js = jobs(a.device)
    print(f"{len(js)} jobs", flush=True)
    P.run_pool(job, js, a.workers)      # prints one JSON line per job

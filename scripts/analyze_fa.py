"""Free-architecture arm analysis: FA vs fixed-actor written weights on the 6 ablation families.

Per writer and mode: family-macro Q-lvl (runs averaged within family) for FA and for the fixed
actor (runs/rev2_private weights_<mode>_M_full_r*), paired difference with a bootstrap over
families, delivery, budget use (params, MACs), architecture statistics from round*.graph.json,
and closed-loop diagnostics. The FA vs fixed table (fa.tex) covers the 6 ablation families; the rows,
architecture statistics and closed-loop summary ("closed_loop") cover every family in runs/rev2_fa.
Writes runs/analysis_rev2/fa.json and tables/fa.tex.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "runs"
S6 = ["ztl", "lys", "esm", "ruz", "crg", "mds"]
WRITERS = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]
LABEL = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5", "terra-6.1": "GPT-6.1 Terra"}


def q(d, metric="Q-lvl"):
    p = d / "eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["sets"].get(metric, {}).get("solve_rate")


def boot(x, B=10000):
    x = np.asarray(x, float)
    if not len(x):
        return math.nan, math.nan, math.nan
    b = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def arch_stats(d, rec):
    if rec.get("final_round") is None:
        return None
    g = d / f"round{rec['final_round']}.graph.json"
    if not g.exists():
        return None
    spec = json.loads(g.read_text())
    nodes = spec.get("graph", spec) if isinstance(spec, dict) else spec
    ops = Counter()
    loops = []

    def walk(ns):
        for n in ns:
            ops[n.get("op")] += 1
            if n.get("op") == "loop":
                loops.append(int(n.get("T", n.get("iters", 0)) or 0))
                walk(n.get("body", []))
    walk(nodes)
    cand = [c for c in rec["candidates"] if c["round"] == rec["final_round"]]
    meta = ((cand[0].get("audit") or {}).get("count") or {}) if cand else {}
    return {"ops": dict(ops), "n_nodes": sum(ops.values()), "loops": loops, "memory": meta.get("memory_size"),
            "params": meta.get("params"), "macs": meta.get("macs_per_step")}


if __name__ == "__main__":
    rows = []
    for sj in sorted((R / "rev2_fa").glob("*/*/freearch_*/session.json")):
        d = sj.parent
        fam, w = d.parent.parent.name, d.parent.name
        mode = d.name.split("_")[1]
        rec = json.loads(sj.read_text())
        delivered = rec.get("final_round") is not None
        blind = json.loads((d / "blind.json").read_text()) if (d / "blind.json").exists() else {}
        diag = json.loads((d / "diag.json").read_text()) if (d / "diag.json").exists() else {}
        rows.append({"family": fam, "writer": w, "mode": mode, "run": int(d.name.split("_r")[-1]), "delivered": delivered,
                     "qlvl": q(d) if delivered else 0.0, "sstart": q(d, "S-start") if delivered else 0.0,
                     "qtpl": q(d, "Q-tpl") if delivered else 0.0, "evaluated": (d / "eval.json").exists() or not delivered,
                     "blind_qlvl": blind.get("Q-lvl"), "first_frame": diag.get("Q-lvl/first_frame"),
                     "cf": diag.get("cf_sensitivity"), "arch": arch_stats(d, rec),
                     "outcome": rec["candidates"][-1].get("outcome") if rec.get("candidates") else None})
    out = {"rows": rows, "cells": {}}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Writer & Mode & Fixed actor & Free architecture & $\Delta$ [95\% CI] & FA $\ge$ fixed\\", r"\midrule"]
    for w in WRITERS:
        for mode, tag in (("oneshot", "W1"), ("feedback", "W2")):
            fa = defaultdict(list)
            for r in rows:
                if r["writer"] == w and r["mode"] == mode and r["evaluated"] and r["qlvl"] is not None:
                    fa[r["family"]].append(r["qlvl"])
            fx = defaultdict(list)
            for k in range(3):
                for f in S6:
                    d = R / "rev2_private" / f / w / f"weights_{mode}_M_full_r{k}"
                    if (d / "session.json").exists():
                        rec = json.loads((d / "session.json").read_text())
                        v = q(d) if rec.get("final_round") is not None else 0.0
                        if v is not None:
                            fx[f].append(v)
            fams = [f for f in S6 if fa.get(f) and fx.get(f)]
            dif = [np.mean(fa[f]) - np.mean(fx[f]) for f in fams]
            m, lo, hi = boot(dif)
            c = {"fixed": float(np.mean([np.mean(fx[f]) for f in fams])) if fams else math.nan,
                 "fa": float(np.mean([np.mean(fa[f]) for f in fams])) if fams else math.nan,
                 "diff": m, "ci": [lo, hi], "fams": len(fams), "fa_better_or_equal": int(sum(x >= 0 for x in dif))}
            out["cells"][f"{w}/{tag}"] = c
            L.append(f"{LABEL[w]} & {tag} & {c['fixed']:.2f} & {c['fa']:.2f} & {m:+.2f} [{lo:+.2f},{hi:+.2f}] & "
                     f"{c['fa_better_or_equal']}/{c['fams']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    # architecture summary
    arch = [r["arch"] for r in rows if r["arch"]]
    ops = Counter()
    for a in arch:
        ops.update(a["ops"])
    out["arch_summary"] = {"n": len(arch), "ops_total": dict(ops.most_common()),
                           "median_nodes": float(np.median([a["n_nodes"] for a in arch])) if arch else None,
                           "loop_T": [t for a in arch for t in a["loops"]],
                           "params": [a["params"] for a in arch if a["params"]], "macs": [a["macs"] for a in arch if a["macs"]]}
    out["delivery"] = {w: float(np.mean([r["delivered"] for r in rows if r["writer"] == w])) for w in WRITERS}
    out["blind"] = {w: [r["blind_qlvl"] for r in rows if r["writer"] == w and r["blind_qlvl"] is not None] for w in WRITERS}

    # closed-loop controls over every delivered artifact with diagnostics (one-shot and feedback, all
    # families): per writer, family mean (artifacts averaged within a family first) and artifact maximum
    def fmean(rs, key):
        g = defaultdict(list)
        for r in rs:
            if r[key] is not None:
                g[r["family"]].append(r[key])
        return float(np.mean([np.mean(v) for v in g.values()])) if g else math.nan
    cl = {}
    for w in WRITERS + ["*"]:
        for mode in ("oneshot", "feedback", "*"):
            rs = [r for r in rows if r["delivered"] and r["blind_qlvl"] is not None and r["first_frame"] is not None
                  and r["cf"] is not None and w in ("*", r["writer"]) and mode in ("*", r["mode"])]
            cl[f"{w}|{mode}"] = {"n_artifacts": len(rs), "n_families": len({r["family"] for r in rs}),
                                 "zero_frame_family_mean": fmean(rs, "blind_qlvl"),
                                 "zero_frame_max": max((r["blind_qlvl"] for r in rs), default=math.nan),
                                 "first_frame_family_mean": fmean(rs, "first_frame"),
                                 "cf_sensitivity_family_mean": fmean(rs, "cf")}
    out["closed_loop"] = cl
    out["n_delivered"] = sum(r["delivered"] for r in rows)
    (R / "analysis_rev2" / "fa.json").write_text(json.dumps(out, indent=1, default=str))
    (R / "analysis_rev2" / "tables").mkdir(parents=True, exist_ok=True)
    (R / "analysis_rev2" / "tables" / "fa.tex").write_text("\n".join(L) + "\n")
    print(f"{len(rows)} FA sessions, evaluated {sum(r['evaluated'] for r in rows)}")
    for k, c in out["cells"].items():
        print(f"{k:16s} fixed {c['fixed']:.2f}  FA {c['fa']:.2f}  diff {c['diff']:+.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] fams {c['fams']}")
    print("delivery", out["delivery"]); print("arch", {k: v for k, v in out["arch_summary"].items() if k in ("n", "median_nodes")},
                                                   "top ops", list(ops.most_common(12)))

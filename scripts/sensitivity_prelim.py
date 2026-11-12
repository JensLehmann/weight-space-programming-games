"""Sensitivity to the three study families used in the preliminary study (ztl, lys, esm).

From the released per-session record (data/sessions.csv), frontier writers, private tier: per-writer
fixed-actor W2, FA W2 and P2, the paired FA - fixed gain, stratum means of W2, FA W2 and P2 (frontier
pooled, family macro, bootstrap 95% CI over the families of a stratum), and the exploratory H3 trend
test (tables_rev2.h3_trend), all on the 17 remaining families and, for reference, on all 20; and the
one-shot ablation arms of the three ablation writers on the preliminary and the other ablation families.
Writes runs/analysis_rev2/sensitivity_prelim.json and the LaTeX table prelim.tex (--tables).

usage: sensitivity_prelim.py [--sessions <sessions.csv>] [--out <json>] [--tables <dir>]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tables_rev2 import FRONTIER, STRATA, STRATUM, boot, h3_trend  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRELIM = ("ztl", "lys", "esm")
ABLATION_FAMILIES = ("ztl", "lys", "esm", "ruz", "crg", "mds")
ARMS = {"W2": ("rev2_private", "weights"), "FA W2": ("rev2_fa", "freearch"), "P2": ("rev2_private", "python")}
ABL_WRITERS = ("astra-6", "sol-6.1", "opus-5.5")
ABL_ARMS = {"W1": ("rev2_private", "full", "high", "M"), "W1 M+S": ("rev2_c3", "full", "high", "MS"),
            "no axial": ("rev2_c3", "noaxial", "high", "M"), "8 iter.": ("rev2_c3", "t8", "high", "M"),
            "effort medium": ("rev2_c3", "full", "medium", "M")}
LABEL = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5", "terra-6.1": "GPT-6.1 Terra"}


def fam_means(rows, tag, cond, writers):
    """Mean over runs per family (undelivered sessions score 0), main protocol, with feedback."""
    v = defaultdict(list)
    for r in rows:
        if (r["tier"], r["tag"], r["condition"], r["mode"], r["info"], r["variant"], r["effort"]) == \
                ("private", tag, cond, "feedback", "M", "full", "high") and r["writer"] in writers:
            x = r["q_lvl_level_balanced"]
            v[r["family"]].append(float(x) if x not in ("", None) else 0.0)
    return {f: float(np.mean(x)) for f, x in v.items()}


def paired(a, b, fams, B=10000, seed=0):
    d = np.array([a[f] - b[f] for f in fams])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "better": int((d > 1e-6).sum()), "n": len(d)}


def analyse(rows, fams):
    per = {w: {arm: {f: x for f, x in fam_means(rows, *ARMS[arm], [w]).items() if f in fams} for arm in ARMS}
           for w in FRONTIER}
    pooled = {arm: {f: x for f, x in fam_means(rows, *ARMS[arm], FRONTIER).items() if f in fams} for arm in ARMS}
    out = {"families": sorted(fams), "n_families": len(fams),
           "per_writer": {w: {arm: float(np.mean(list(per[w][arm].values()))) for arm in ARMS}
                          | {"FA W2 - W2": paired(per[w]["FA W2"], per[w]["W2"], sorted(fams))} for w in FRONTIER},
           "strata": {}, "h3_trend": h3_trend(pooled["W2"], pooled["P2"])}
    for arm in ARMS:
        cells = {}
        for s in STRATA:
            v = [x for f, x in pooled[arm].items() if STRATUM[f] == s]
            m, lo, hi = boot(v)
            cells[s] = {"mean": m, "ci": [lo, hi] if len(v) > 1 else None, "n_families": len(v)}
        cells["macro"] = float(np.mean(list(pooled[arm].values())))
        out["strata"][arm] = cells
    return out


def ablations(rows, fams):
    """One-shot written weights per ablation arm (mean over runs per family, then over families)."""
    out = {}
    for w in ABL_WRITERS:
        out[w] = {}
        for arm, (tag, variant, effort, info) in ABL_ARMS.items():
            v = defaultdict(list)
            for r in rows:
                if (r["tier"], r["tag"], r["condition"], r["mode"], r["variant"], r["effort"], r["info"]) == \
                        ("private", tag, "weights", "oneshot", variant, effort, info) and r["writer"] == w \
                        and r["family"] in fams:
                    x = r["q_lvl_level_balanced"]
                    v[r["family"]].append(float(x) if x not in ("", None) else 0.0)
            out[w][arm] = float(np.mean([np.mean(x) for x in v.values()]))
    return out


def table(res):
    """Two stacked blocks: stratum means (frontier pooled; all-20 value in parentheses where it differs)
    and per-writer means on the 17 families."""
    ex, al = res["without_prelim"], res["all_families"]
    f2 = lambda x: f"{x:.2f}"
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         "Frontier writers pooled & " + " & ".join(f"{s} ({ex['strata']['W2'][s]['n_families']})" for s in STRATA)
         + r" & macro\\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for s in STRATA + ["macro"]:
            a = ex["strata"][arm][s] if s == "macro" else ex["strata"][arm][s]["mean"]
            b = al["strata"][arm][s] if s == "macro" else al["strata"][arm][s]["mean"]
            cells.append(f2(a) + ("" if f2(a) == f2(b) else f" ({f2(b)})"))
        L.append(f"{arm} & " + " & ".join(cells) + r"\\")
    L += [r"\midrule", r"Per writer & W2 & FA W2 & P2 & \multicolumn{2}{l}{FA W2 $-$ W2 [95\% CI]}\\", r"\midrule"]
    for w in FRONTIER:
        p = ex["per_writer"][w]
        d = p["FA W2 - W2"]
        L.append(f"{LABEL[w]} & {f2(p['W2'])} & {f2(p['FA W2'])} & {f2(p['P2'])} & "
                 rf"\multicolumn{{2}}{{l}}{{{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}] ({d['better']}/{d['n']})}}\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(Path(__file__).resolve().parents[1] / "data" / "sessions.csv"))
    ap.add_argument("--out", default=str(ROOT / "runs/analysis_rev2/sensitivity_prelim.json"))
    ap.add_argument("--tables", default=str(ROOT / "runs/analysis_rev2/tables"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.sessions)))
    mech = json.loads((ROOT / "runs/analysis_rev2/mechanism.json").read_text())["artifacts"]
    res = {"prelim_families": list(PRELIM),
           "share_of_ablation_families": [sum(f in PRELIM for f in ABLATION_FAMILIES), len(ABLATION_FAMILIES)],
           "share_of_mechanism_actors": [sum(f in PRELIM for f in mech), len(mech)],
           "without_prelim": analyse(rows, set(STRATUM) - set(PRELIM)),
           "all_families": analyse(rows, set(STRATUM)),
           "ablations_oneshot": {"prelim": ablations(rows, set(PRELIM)),
                                 "other": ablations(rows, set(ABLATION_FAMILIES) - set(PRELIM))}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.tables).mkdir(parents=True, exist_ok=True)
    (Path(a.tables) / "prelim.tex").write_text(table(res))
    print(json.dumps(res, indent=1))

"""FA across all 20 private families and the public tier.

Writes runs/analysis_rev2/fa_all.json and tables fa_all.tex (per writer: fixed W1/W2, FA W1/W2,
paired FA-fixed with bootstrap CI over families) and fa_strata.tex (frontier pooled, per stratum:
fixed W2, FA W2, P2, P-react), and fa_public.tex (FA W2 vs fixed W2 vs P2 on the public tier).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "runs"
STRATUM = {"ztl": "easy", "wih": "easy", "qcn": "easy", "lys": "medium", "esm": "medium", "aiv": "medium",
           "nkp": "medium", "obn": "medium", "qwv": "medium", "ruz": "medium", "vvf": "medium", "ubr": "medium",
           "yuz": "medium", "bvn": "medium", "trx": "hard", "yiy": "hard", "crg": "hard", "qpv": "hard",
           "ydv": "hard", "mds": "very hard"}
STRATA = ["easy", "medium", "hard", "very hard"]
W4 = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]
LABEL = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5", "terra-6.1": "GPT-6.1 Terra",
         "luna-6": "GPT-6 Luna", "sonnet-5.5": "Claude Sonnet 5.5", "haiku-5.5": "Claude Haiku 5.5",
         "sol-5.6": "GPT-5.6 Sol$^\\dagger$", "opus-5": "Claude Opus 5$^\\dagger$"}
W5 = ["luna-6", "sonnet-5.5", "haiku-5.5", "sol-5.6", "opus-5"]   # weaker and previous-generation writers (run tags rev2_*_weak)


def q(d, metric="Q-lvl"):
    p = d / "eval.json"
    return json.loads(p.read_text())["sets"].get(metric, {}).get("solve_rate") if p.exists() else None


_LEVELS = {}


def _levels(fam):
    if fam not in _LEVELS:
        from wsp.queries import load_sets
        _LEVELS[fam] = {s.key: s.level for s in load_sets(fam, rev=2)["Q-lvl"]}
    return _LEVELS[fam]


def score(d):
    """Level-balanced Q-lvl (mean over held-out levels of per-level success, as in analyze_rev2);
    0 if the session delivered no artifact; None if not yet evaluated."""
    sj = d / "session.json"
    if not sj.exists():
        return None
    rec = json.loads(sj.read_text())
    if rec.get("final_round") is None:
        return 0.0
    p = d / "eval.json"
    if not p.exists():
        return None
    fam = d.parts[-3]
    lv = _levels(fam)
    per = defaultdict(list)
    for e in json.loads(p.read_text())["episodes"].get("Q-lvl", []):
        per[lv[e[0]]].append(e[1])
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


def fam_scores(root, fams, w, dirpat):
    out = {}
    for f in fams:
        xs = [score(d) for d in sorted((root / f / w).glob(dirpat))]
        xs = [x for x in xs if x is not None]
        if xs:
            out[f] = float(np.mean(xs))
    return out


def boot(x, B=10000):
    x = np.asarray(x, float)
    if not len(x):
        return math.nan, math.nan, math.nan
    b = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def f2(x):
    return "--" if x is None or x != x else f"{x:.2f}"


if __name__ == "__main__":
    T, FA_T, PUB, FA_P = R / "rev2_private", R / "rev2_fa", R / "rev2_public", R / "rev2_fa_public"
    from wsp.public_scope import assert_public_cohort_complete
    assert_public_cohort_complete()
    fams = list(STRATUM)
    pubs = sorted(p.name for p in PUB.iterdir() if p.is_dir())
    res = {"private": {}, "public": {}, "strata": {}}
    L = [r"\begin{tabular}{lrrrrll}", r"\toprule",
         r"Writer & W1 fixed & W1 FA & W2 fixed & W2 FA & $\Delta$ W2 [95\% CI] & FA$>$fixed\\", r"\midrule"]
    for w in W4 + W5:
        c = {}
        if w in W5 and w == W5[0]:
            L.append(r"\midrule")
        for mode, tag in (("oneshot", "W1"), ("feedback", "W2")):
            fx_root = R / "rev2_private_weak" if (w in W5 and mode == "feedback") else T
            fa_root = R / "rev2_fa_weak" if w in W5 else FA_T
            fx = fam_scores(fx_root, fams, w, f"weights_{mode}_M_full_r*")
            fa = fam_scores(fa_root, fams, w, f"freearch_{mode}_M_r*")
            both = [f for f in fams if f in fx and f in fa]
            m, lo, hi = boot([fa[f] - fx[f] for f in both])
            c[tag] = {"fixed": float(np.mean([fx[f] for f in both])) if both else math.nan,
                      "fa": float(np.mean([fa[f] for f in both])) if both else math.nan,
                      "diff": m, "ci": [lo, hi], "n": len(both), "fa_better": int(sum(fa[f] > fx[f] for f in both)),
                      "fa_by_family": fa, "fixed_by_family": fx}
        res["private"][w] = c
        L.append(f"{LABEL[w]} & {f2(c['W1']['fixed'])} & {f2(c['W1']['fa'])} & {f2(c['W2']['fixed'])} & {f2(c['W2']['fa'])} & "
                 f"{c['W2']['diff']:+.2f} [{c['W2']['ci'][0]:+.2f},{c['W2']['ci'][1]:+.2f}] & {c['W2']['fa_better']}/{c['W2']['n']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    # strata (frontier pooled, family macro within stratum)
    c3 = json.loads((R / "analysis_rev2_c3" / "rows.json").read_text())
    pre = defaultdict(lambda: defaultdict(list))
    for w in W4:
        for f in fams:
            for name, dic in (("Written weights, fixed actor (W2)", res["private"][w]["W2"]["fixed_by_family"]),
                              ("Written weights, free architecture (FA W2)", res["private"][w]["W2"]["fa_by_family"])):
                if f in dic:
                    pre[name][f].append(dic[f])
            p2 = fam_scores(T, [f], w, "python_feedback_M_full_r*")
            if f in p2:
                pre["Python player (P2)"][f].append(p2[f])
    for r in c3:
        if r.get("evaluated") and r["arm"] == "PR1" and r["model"] in W4:
            pre["Reactive Python (P-react)"][r["family"]].append(r["Q-lvl"])
    S = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         "Policy source & " + " & ".join(f"{s} ({sum(1 for v in STRATUM.values() if v == s)})" for s in STRATA) + r" & macro\\",
         r"\midrule"]
    for name in ("Written weights, fixed actor (W2)", "Written weights, free architecture (FA W2)", "Reactive Python (P-react)",
                 "Python player (P2)"):
        fm = {f: float(np.mean(v)) for f, v in pre[name].items()}
        vals = [np.mean([x for f, x in fm.items() if STRATUM[f] == s]) if any(STRATUM[f] == s for f in fm) else math.nan for s in STRATA]
        res["strata"][name] = dict(zip(STRATA, vals)) | {"macro": float(np.mean(list(fm.values())))}
        S.append(f"{name} & " + " & ".join(f2(v) for v in vals) + f" & {f2(np.mean(list(fm.values())))}\\\\")
    S += [r"\bottomrule", r"\end{tabular}"]
    # public tier
    P = [r"\begin{tabular}{lrrrl}", r"\toprule", r"Writer & W2 fixed & W2 FA & P2 & $\Delta$ FA$-$fixed [95\% CI]\\", r"\midrule"]
    for w in W4:
        fx = fam_scores(PUB, pubs, w, "weights_feedback_M_full_r*")
        fa = fam_scores(FA_P, pubs, w, "freearch_feedback_M_r*")
        p2 = fam_scores(PUB, pubs, w, "python_feedback_M_full_r*")
        both = [g for g in pubs if g in fx and g in fa]
        m, lo, hi = boot([fa[g] - fx[g] for g in both])
        res["public"][w] = {"fixed": float(np.mean([fx[g] for g in both])) if both else math.nan,
                            "fa": float(np.mean([fa[g] for g in both])) if both else math.nan,
                            "p2": float(np.mean([p2[g] for g in both if g in p2])) if both else math.nan,
                            "diff": m, "ci": [lo, hi], "n": len(both)}
        c = res["public"][w]
        P.append(f"{LABEL[w]} & {f2(c['fixed'])} & {f2(c['fa'])} & {f2(c['p2'])} & {m:+.2f} [{lo:+.2f},{hi:+.2f}] ({c['n']} games)\\\\")
    P += [r"\bottomrule", r"\end{tabular}"]
    out = R / "analysis_rev2" / "tables"
    (out / "fa_all.tex").write_text("\n".join(L) + "\n")
    (out / "fa_strata.tex").write_text("\n".join(S) + "\n")
    (out / "fa_public.tex").write_text("\n".join(P) + "\n")
    (R / "analysis_rev2" / "fa_all.json").write_text(json.dumps(res, indent=1))
    for w in W4 + W5:
        c = res["private"][w]
        print(f"{w:10s} W1 fixed {c['W1']['fixed']:.2f} FA {c['W1']['fa']:.2f} | W2 fixed {c['W2']['fixed']:.2f} FA {c['W2']['fa']:.2f} "
              f"Δ {c['W2']['diff']:+.2f} [{c['W2']['ci'][0]:+.2f},{c['W2']['ci'][1]:+.2f}] n={c['W2']['n']}")
    print(json.dumps(res["strata"], indent=1)); print(json.dumps(res["public"], indent=1))

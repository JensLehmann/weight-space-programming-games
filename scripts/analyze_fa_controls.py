"""FA language controls: fixed graph (A) and fixed ops (B) vs fixed W2 and FA W2.

Per writer and pooled (frontier writers): family-macro level-balanced Q-lvl, paired differences with
bootstrap CIs over families (pooled: per-family mean over writers, then bootstrap over families).
  A - fixed W2   interface effect (same graph, FA prompt/language/executor)
  B - A          topology within the fixed op set
  FA - B         extra operations
Writes runs/analysis_rev2/fa_controls.json and tables/fa_controls.tex; also reports how often
candidates were rejected by the control check and how many B graphs equal the fixed graph.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fa_all import LABEL, STRATUM, W4, boot, f2, fam_scores, R  # noqa: E402

ARMS = [("fixed", R / "rev2_private", "weights_feedback_M_full_r*"),
        ("A", R / "rev2_fa_fixedgraph", "freearch_feedback_M_r*"),
        ("B", R / "rev2_fa_fixedops", "freearch_feedback_M_r*"),
        ("FA", R / "rev2_fa", "freearch_feedback_M_r*")]
NAME = {"fixed": "Fixed actor (W2)", "A": "FA language, fixed graph", "B": "FA language, fixed ops",
        "FA": "Free architecture (FA W2)"}
CONTRASTS = [("A", "fixed"), ("B", "A"), ("FA", "B"), ("FA", "fixed")]


def rejections(root):
    """Count candidates whose materialisation failed with a control error, and B graphs equal to the fixed graph."""
    from wsp import freearch as FA
    fixed = json.dumps(FA._normalise(FA.fixed_actor_arch())["graph"], sort_keys=True)
    n_cand = n_ctrl = n_final = n_same = 0
    for sj in root.glob("*/*/freearch_feedback_M_r*/session.json"):
        rec = json.loads(sj.read_text())
        for c in rec.get("candidates", []):
            n_cand += 1
            if "this run requires exactly the fixed actor graph" in str(c.get("artifact_error") or "") or "this run allows only the fixed actor" in str(c.get("artifact_error") or ""):
                n_ctrl += 1
        if rec.get("final_round") is not None:
            g = sj.parent / f"round{rec['final_round']}.graph.json"
            if g.exists():
                n_final += 1
                n_same += json.dumps(FA._normalise(json.loads(g.read_text()))["graph"], sort_keys=True) == fixed
    return {"candidates": n_cand, "control_rejected": n_ctrl, "final_artifacts": n_final, "final_equal_fixed_graph": n_same}


if __name__ == "__main__":
    fams = list(STRATUM)
    per = {}                      # per[w][arm] = {fam: score}
    for w in W4:
        per[w] = {a: fam_scores(root, fams, w, pat) for a, root, pat in ARMS}
    res = {"per_writer": {}, "pooled": {}, "rejections": {}}
    rows = []
    for w in W4 + ["pooled"]:
        if w == "pooled":
            sc = {a: {f: float(np.mean([per[x][a][f] for x in W4 if f in per[x][a]]))
                      for f in fams if all(f in per[x][a] for x in W4)} for a, _, _ in ARMS}
        else:
            sc = per[w]
        common = [f for f in fams if all(f in sc[a] for a, _, _ in ARMS)]
        r = {"n": len(common), "mean": {a: float(np.mean([sc[a][f] for f in common])) if common else math.nan
                                        for a, _, _ in ARMS}}
        for hi_, lo_ in CONTRASTS:
            m, lo, hi = boot([sc[hi_][f] - sc[lo_][f] for f in common])
            r[f"{hi_}-{lo_}"] = {"diff": m, "ci": [lo, hi], "better": int(sum(sc[hi_][f] > sc[lo_][f] for f in common))}
        r["by_family"] = {a: sc[a] for a, _, _ in ARMS}
        (res["pooled"] if w == "pooled" else res["per_writer"]).update({w: r} if w != "pooled" else r)
        rows.append((w, r))
    for tag in ("rev2_fa_fixedgraph", "rev2_fa_fixedops"):
        if (R / tag).exists():
            res["rejections"][tag] = rejections(R / tag)
    # Budget-matched (descriptive): FA / B artifacts within the fixed actor's parameters and MACs
    # (57,289 params; 1,331,914,880 MACs/step under the FA counting rule), paired with fixed W2.
    from analyze_fa_all import score
    P_MAX, M_MAX = 57_289, 1_331_914_880
    res["budget_matched"] = {}
    for arm, root in (("FA", R / "rev2_fa"), ("B", R / "rev2_fa_fixedops")):
        fam_in, n_all, n_in = {}, 0, 0
        for f in fams:
            xs = []
            for w in W4:
                for d in sorted((root / f / w).glob("freearch_feedback_M_r*")):
                    sj = d / "session.json"
                    if not sj.exists():
                        continue
                    rec = json.loads(sj.read_text())
                    if rec.get("final_round") is None:
                        continue
                    c = [c for c in rec["candidates"] if str(c["round"]) == str(rec["final_round"])]
                    cnt = ((c[0].get("audit") or {}).get("count") or {}) if c else {}
                    if not cnt or (s := score(d)) is None:
                        continue
                    n_all += 1
                    if cnt["params"] <= P_MAX and cnt["macs_per_step"] <= M_MAX:
                        n_in += 1
                        xs.append(s)
            if xs:
                fam_in[f] = float(np.mean(xs))
        fx = res["pooled"]["by_family"]["fixed"]
        both = [f for f in fam_in if f in fx]
        m, lo, hi = boot([fam_in[f] - fx[f] for f in both])
        res["budget_matched"][arm] = {"artifacts": n_all, "within_budget": n_in, "families": len(both),
                                      "mean_within": float(np.mean([fam_in[f] for f in both])) if both else math.nan,
                                      "fixed_same_families": float(np.mean([fx[f] for f in both])) if both else math.nan,
                                      "diff": m, "ci": [lo, hi], "by_family": fam_in}
        print(f"budget-matched {arm}: {n_in}/{n_all} artifacts within budget, {len(both)} families, "
              f"{res['budget_matched'][arm]['mean_within']:.2f} vs fixed {res['budget_matched'][arm]['fixed_same_families']:.2f}, "
              f"diff {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    L = [r"\begin{tabular}{lrrrrlll}", r"\toprule",
         r"Writer & Fixed & A & B & FA & A$-$fixed & B$-$A & FA$-$B\\", r"\midrule"]
    for w, r in rows:
        lab = "Pooled (4 writers)" if w == "pooled" else LABEL[w]
        cs = " & ".join(f"{r[k]['diff']:+.2f} [{r[k]['ci'][0]:+.2f},{r[k]['ci'][1]:+.2f}]" for k in ("A-fixed", "B-A", "FA-B"))
        if w == "pooled":
            L.append(r"\midrule")
        L.append(f"{lab} & " + " & ".join(f2(r['mean'][a]) for a, _, _ in ARMS) + f" & {cs}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    out = R / "analysis_rev2"
    (out / "tables" / "fa_controls.tex").write_text("\n".join(L) + "\n")
    (out / "fa_controls.json").write_text(json.dumps(res, indent=1))
    for w, r in rows:
        print(f"{w:10s} n={r['n']:2d} " + " ".join(f"{a} {r['mean'][a]:.2f}" for a, _, _ in ARMS) + " | " +
              " ".join(f"{k} {r[k]['diff']:+.2f} [{r[k]['ci'][0]:+.2f},{r[k]['ci'][1]:+.2f}] ({r[k]['better']})"
                       for k in ("A-fixed", "B-A", "FA-B", "FA-fixed")))
    print(json.dumps(res["rejections"], indent=1))

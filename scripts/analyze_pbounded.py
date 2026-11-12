"""Bounded program player (pbounded): NumPy players under the FA computation model, with feedback,
frontier writers, 20 private families. Paired per family against FA W2, fixed-actor W2 and P2.
Writes runs/analysis_rev2/pbounded.json and tables/pbounded.tex."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fa_all import LABEL, STRATUM, W4, R, boot, f2, fam_scores  # noqa: E402

FAMS = list(STRATUM)
ARMS = [("fixed", R / "rev2_private", "weights_feedback_M_full_r*"),
        ("fa", R / "rev2_fa", "freearch_feedback_M_r*"),
        ("pb", R / "rev2_pbounded", "pbounded_feedback_M_r*"),
        ("p2", R / "rev2_private", "python_feedback_M_full_r*")]

if __name__ == "__main__":
    per = {w: {a: fam_scores(root, FAMS, w, pat) for a, root, pat in ARMS} for w in W4}
    pooled = {a: {f: float(np.mean([per[w][a][f] for w in W4])) for f in FAMS if all(f in per[w][a] for w in W4)}
              for a, _, _ in ARMS}
    res = {"per_writer": {}, "pooled": {}}
    L = [r"\begin{tabular}{lrrrrll}", r"\toprule",
         r"Writer & Fixed W2 & FA W2 & Bounded & P2 & Bounded$-$FA & P2$-$Bounded\\", r"\midrule"]
    rows = [(w, per[w]) for w in W4] + [("pooled", pooled)]
    for w, sc in rows:
        common = [f for f in FAMS if all(f in sc[a] for a, _, _ in ARMS)]
        r = {"n": len(common), "mean": {a: float(np.mean([sc[a][f] for f in common])) for a, _, _ in ARMS}}
        for hi_, lo_ in (("pb", "fa"), ("p2", "pb"), ("pb", "fixed")):
            m, lo, hi = boot([sc[hi_][f] - sc[lo_][f] for f in common])
            r[f"{hi_}-{lo_}"] = {"diff": m, "ci": [lo, hi], "better": int(sum(sc[hi_][f] > sc[lo_][f] for f in common))}
        (res["pooled"].update(r) if w == "pooled" else res["per_writer"].__setitem__(w, r))
        lab = "Pooled (4 writers)" if w == "pooled" else LABEL[w]
        if w == "pooled":
            L.append(r"\midrule")
        d1, d2 = r["pb-fa"], r["p2-pb"]
        L.append(f"{lab} & " + " & ".join(f2(r['mean'][a]) for a in ("fixed", "fa", "pb", "p2")) +
                 f" & {d1['diff']:+.2f} [{d1['ci'][0]:+.2f},{d1['ci'][1]:+.2f}] & {d2['diff']:+.2f} [{d2['ci'][0]:+.2f},{d2['ci'][1]:+.2f}]\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    # delivery
    deliv = {}
    for w in W4:
        ss = list((R / "rev2_pbounded").glob(f"*/{w}/pbounded_feedback_M_r*/session.json"))
        deliv[w] = sum(json.loads(s.read_text()).get("final_round") is not None for s in ss) / max(1, len(ss))
    res["delivery"] = deliv
    # strata (pooled)
    res["strata"] = {s: float(np.mean([pooled["pb"][f] for f in FAMS if STRATUM[f] == s and f in pooled["pb"]]))
                     for s in ("easy", "medium", "hard", "very hard")}
    (R / "analysis_rev2" / "tables" / "pbounded.tex").write_text("\n".join(L) + "\n")
    (R / "analysis_rev2" / "pbounded.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L)); print(json.dumps({"delivery": deliv, "strata": res["strata"]}, indent=1))

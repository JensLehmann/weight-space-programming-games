"""Share of written parameters that are structural constants (exact 0, +-1, other small integers,
and signed powers of two such as 0.5 or 4) for FA networks and fixed-actor written weights (W2),
frontier writers, private tier. Also reports the share of exact zeros and, among nonzero entries, the
share of structural constants (per artifact: median and IQR; pooled), and FA W1 and W2 together.
Writes runs/analysis_rev2/weight_structure.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fa_all import W4, R  # noqa: E402

def classify(x):
    x = np.asarray(x, np.float64).ravel()
    a = np.abs(x)
    zero = a == 0
    one = a == 1
    small_int = (~zero) & (~one) & (a <= 16) & (np.round(a) == a)
    with np.errstate(divide="ignore"):
        lg = np.log2(np.where(a > 0, a, 1))
    pow2 = (~zero) & (~one) & (~small_int) & (np.round(lg) == lg) & (a >= 2 ** -8) & (a <= 2 ** 16)
    other = ~(zero | one | small_int | pow2)
    return {k: int(v.sum()) for k, v in (("zero", zero), ("one", one), ("small_int", small_int), ("pow2", pow2), ("other", other))}

def _med_iqr(x):
    return (float(np.median(x)), [float(np.percentile(x, 25)), float(np.percentile(x, 75))]) if len(x) else (None, None)


def summarise(counts):
    tot = sum(sum(c.values()) for c in counts)
    per = [{k: v / max(1, sum(c.values())) for k, v in c.items()} for c in counts]
    structural = [1 - p["other"] for p in per]
    zero = [p["zero"] for p in per]
    nz = [c for c in counts if sum(c.values()) > c["zero"]]
    nz_structural = [1 - c["other"] / (sum(c.values()) - c["zero"]) for c in nz]
    nz_tot = sum(sum(c.values()) - c["zero"] for c in counts)
    s_med, s_iqr = _med_iqr(structural)
    z_med, z_iqr = _med_iqr(zero)
    n_med, n_iqr = _med_iqr(nz_structural)
    return {"artifacts": len(counts), "params": tot,
            "pooled_share": {k: sum(c[k] for c in counts) / tot for k in counts[0]} if counts else {},
            "structural_share_median": s_med, "structural_share_iqr": s_iqr,
            "artifacts_ge_90pct_structural": int(sum(s >= 0.9 for s in structural)),
            "zero_share_median": z_med, "zero_share_iqr": z_iqr,
            "nonzero_structural_share_median": n_med, "nonzero_structural_share_iqr": n_iqr,
            "nonzero_structural_share_pooled": (1 - sum(c["other"] for c in counts) / nz_tot) if nz_tot else None}

if __name__ == "__main__":
    out, allc = {}, {}
    for name, root, pat, kind in (("FA W2", R / "rev2_fa", "freearch_feedback_M_r*", "fa"),
                                  ("FA W1", R / "rev2_fa", "freearch_oneshot_M_r*", "fa"),
                                  ("fixed W2", R / "rev2_private", "weights_feedback_M_full_r*", "w")):
        counts = []
        for w in W4:
            for d in sorted(root.glob(f"*/{w}/{pat}")):
                sj = d / "session.json"
                if not sj.exists():
                    continue
                fr = json.loads(sj.read_text()).get("final_round")
                p = d / f"round{fr}.npz"
                if fr is None or not p.exists():
                    continue
                z = np.load(p)
                c = None
                for k in z.files:
                    ck = classify(z[k])
                    c = ck if c is None else {kk: c[kk] + ck[kk] for kk in c}
                counts.append(c)
        out[name] = summarise(counts)
        allc[name] = counts
        print(name, json.dumps(out[name]))
    out["FA W1+W2"] = summarise(allc["FA W1"] + allc["FA W2"])
    print("FA W1+W2", json.dumps(out["FA W1+W2"]))
    (R / "analysis_rev2" / "weight_structure.json").write_text(json.dumps(out, indent=1))

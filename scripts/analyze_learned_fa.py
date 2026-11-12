"""Learned FA networks: the writer's own FA graph (Astra, FA W2 run 0, per family) with re-initialised
weights, trained by BC-support (planner teacher) and by distillation of Astra's run-0 P2 player, both
without augmentation and with recomputed memory, 6 h cap. Diverged runs (non-finite weights) score 0.
Compared per family with the same graph's written weights and with the same learners in the fixed actor.
Writes runs/analysis_rev2/learned_fa.json and tables/learned_fa.tex."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fa_all import STRATUM, R, boot, f2, score  # noqa: E402

FAMS = list(STRATUM)


def lvl_balanced(d):
    """Level-balanced Q-lvl from eval.json for a learned-baseline dir (no session.json)."""
    from collections import defaultdict
    from wsp.queries import load_sets
    p = d / "eval.json"
    if not p.exists():
        return None
    fam = d.parts[-2] if d.parts[-3] == "rev2_baselines" else d.parts[-4]
    lv = {s.key: s.level for s in load_sets(fam, rev=2)["Q-lvl"]}
    per = defaultdict(list)
    for e in json.loads(p.read_text())["episodes"].get("Q-lvl", []):
        per[lv[e[0]]].append(e[1])
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


def rate(d, k):
    p = d / "eval.json"
    return json.loads(p.read_text())["sets"].get(k, {}).get("solve_rate") if p.exists() else None


def info(d):
    cfg = json.loads((d / "train_config.json").read_text()) if (d / "train_config.json").exists() else {}
    summ = json.loads((d / "dagger_summary.json").read_text()) if (d / "dagger_summary.json").exists() else {}
    return {"init_dead_grad_frac": cfg.get("init_dead_grad_frac"), "init_nan": cfg.get("init_nan"),
            "truncated": summ.get("truncated"), "iterations_completed": summ.get("iterations_completed")}


if __name__ == "__main__":
    rows, res = [], {"per_family": {}}
    for f in FAMS:
        fa_dir = R / "rev2_fa" / f / "astra-6" / "freearch_feedback_M_r0"
        written = score(fa_dir)
        cells = {"written_fa": written, "written_fa_support": rate(fa_dir, "S-start")}
        for name, d, fixed in (("bc", R / "rev2_baselines" / f / "bc_support_learnedfa_s0", R / "rev2_baselines" / f / "bc_support_memfix_s0"),
                               ("distill", R / "rev2_private" / f / "astra-6" / "python_feedback_M_full_r0" / "distill_learnedfa_s0",
                                R / "rev2_private" / f / "astra-6" / "python_feedback_M_full_r0" / "distill_plain_memfix_s0")):
            done = (d / "eval.json").exists()
            cells[f"{name}_fa"] = lvl_balanced(d) if done else 0.0          # diverged: non-finite weights -> 0
            cells[f"{name}_fa_support"] = rate(d, "S-start") if done else 0.0
            cells[f"{name}_fa_diverged"] = not done
            cells[f"{name}_fa_info"] = info(d) if d.exists() else {}
            cells[f"{name}_fixed"] = lvl_balanced(fixed)
            cells[f"{name}_fixed_support"] = rate(fixed, "S-start")
        res["per_family"][f] = cells
    pf = res["per_family"]
    def mean(k):
        xs = [pf[f][k] for f in FAMS if pf[f][k] is not None]
        return float(np.mean(xs)) if xs else math.nan, len(xs)
    summ = {}
    for k in ("written_fa", "written_fa_support", "bc_fa", "bc_fa_support", "distill_fa", "distill_fa_support",
              "bc_fixed", "bc_fixed_support", "distill_fixed", "distill_fixed_support"):
        summ[k] = mean(k)[0]
    for name in ("bc", "distill"):
        d = [pf[f]["written_fa"] - pf[f][f"{name}_fa"] for f in FAMS if pf[f]["written_fa"] is not None]
        m, lo, hi = boot(d)
        summ[f"written_minus_{name}_fa"] = {"diff": m, "ci": [lo, hi], "written_better": int(sum(x > 0 for x in d))}
        summ[f"{name}_diverged"] = int(sum(pf[f][f"{name}_fa_diverged"] for f in FAMS))
        summ[f"{name}_truncated"] = int(sum(bool(pf[f][f"{name}_fa_info"].get("truncated")) for f in FAMS))
    summ["init_dead_ge_99"] = int(sum((pf[f]["bc_fa_info"].get("init_dead_grad_frac") or 0) >= 0.99 for f in FAMS))
    res["summary"] = summ
    L = [r"\begin{tabular}{lrrl}", r"\toprule",
         r"Weights in the writer's FA graph & S-start & Q-lvl & Written $-$ learned [95\% CI]\\", r"\midrule",
         f"Written (Astra, FA W2 run 0) & {f2(summ['written_fa_support'])} & {f2(summ['written_fa'])} & \\\\"]
    for name, lab in (("bc", "Learned: BC-support"), ("distill", "Learned: distill (no aug.)")):
        w = summ[f"written_minus_{name}_fa"]
        L.append(f"{lab} & {f2(summ[name + '_fa_support'])} & {f2(summ[name + '_fa'])} & "
                 f"{w['diff']:+.2f} [{w['ci'][0]:+.2f},{w['ci'][1]:+.2f}] ({w['written_better']}/20)\\\\")
    L += [r"\midrule"]
    for name, lab in (("bc", "Fixed actor: BC-support"), ("distill", "Fixed actor: distill (no aug.)")):
        L.append(f"{lab} & {f2(summ[name + '_fixed_support'])} & {f2(summ[name + '_fixed'])} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (R / "analysis_rev2" / "tables" / "learned_fa.tex").write_text("\n".join(L) + "\n")
    (R / "analysis_rev2" / "learned_fa.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1))
    for f in FAMS:
        c = pf[f]
        print(f"{f:4s} written {f2(c['written_fa'])} | bc {f2(c['bc_fa'])} sup {f2(c['bc_fa_support'])} div {c['bc_fa_diverged']} "
              f"trunc {c['bc_fa_info'].get('truncated')} dead {c['bc_fa_info'].get('init_dead_grad_frac')} | "
              f"distill {f2(c['distill_fa'])} sup {f2(c['distill_fa_support'])} div {c['distill_fa_diverged']} trunc {c['distill_fa_info'].get('truncated')}")

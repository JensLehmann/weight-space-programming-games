"""PPO from robust vs standard W2 actors (runs/rev2_stage2_robust). No GPU, no writer calls.

The runs are scripts/stage2.py with exact FP32 (WSP_NO_TF32=1), for every family in
{ztl, lys, esm, ruz, crg, mds}, writer in {astra-6, sol-6.1, opus-5.5} with an artifact in run 0 of both prompts,
seed in {0, 1, 2} and reward in {sparse, shaped}:

  stage2.py wsp_rl:runs/<src>/<fam>/<w>/<session> <fam> <seed> --split-rev 2 --steps 300000 --bptt 16 \
      --eval-sets S-start,Q-lvl --eval-at 0,8192,16384,40000,80000,160000 --eval-every 300000 \
      [--shaping planner] --out runs/rev2_stage2_robust/<fam>/<kind>-<w>_std_<reward>_s<seed>

with <src>/<session> = rev2_robust/.../weights_feedback_M_robust_r0 (kind robust) or
rev2_private/.../weights_feedback_M_full_r0 (kind standard). The calibrated-head check appends ':cal' to the
robust kind (seed 0, --steps 81920, --out runs/rev2_stage2_robust_cal/...); the update-size check runs
--steps 8192 (one update, sparse, seed 0, --out runs/rev2_stage2_robust_upd/...), read by matched_robust.py.


Held-out success (Q-lvl, greedy) per checkpoint. A run that diverged to non-finite weights scores 0 from the
first checkpoint it did not reach (the paper's rule for diverged runs). Aggregation: mean over seeds and
rewards within (kind, writer, family), then family macro per (kind, writer) over families present for both kinds.
Decision rule as in the paper: a run "beats" if its final success exceeds both its initialisation and RL(theta_0)
for that family and reward (from runs/rev2_stage2/<fam>/theta0_<reward>_s<seed>).
Output: runs/analysis_rev2/stage2_robust.json and a printed table.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "runs"
D = R / "rev2_stage2_robust"
NAMES = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5"}
STEPS = [0, 8192, 16384, 40960, 81920, 163840, 303104]


def curve(run):
    ev = json.loads((run / "ppo_eval.json").read_text()) if (run / "ppo_eval.json").exists() else []
    q = {e["eval_steps"]: e["solve_rate"] for e in ev
         if e["set"] == "Q-lvl" and e["decoding"] == "greedy"}
    samp = next((e["solve_rate"] for e in ev if e["set"] == "Q-lvl" and e["decoding"] == "sampled"
                 and e["actor"] == "init"), None)
    out, alive = [], True
    for s in STEPS:
        if alive and s in q:
            out.append(q[s])
        else:
            alive = False
            out.append(0.0)
    return out, samp, len(q) < len(STEPS)


def theta0(fam, reward, seed):
    p = R / "rev2_stage2" / fam / f"theta0_{reward}_s{seed}" / "ppo_eval.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text())
    fin = [e for e in ev if e["set"] == "Q-lvl" and e["decoding"] == "greedy"]
    return max(fin, key=lambda e: e["eval_steps"])["solve_rate"] if fin else None


def main():
    cells = defaultdict(list)
    rows, beats, diverged, samp_gap = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(list)
    for run in sorted(D.glob("*/*_std_*_s*")):
        fam = run.parent.name
        kind_w, _, reward, seed = run.name.rsplit("_", 3)[0], None, run.name.split("_")[-2], int(run.name[-1])
        kind, w = kind_w.split("-", 1)
        c, samp, div = curve(run)
        cells[(kind, w, fam)].append(c)
        diverged[(kind, w)][0] += div
        diverged[(kind, w)][1] += 1
        if samp is not None:
            samp_gap[(kind, w)].append((c[0], samp))
        t0 = theta0(fam, reward, seed)
        if t0 is not None:
            beats[(kind, w)][0] += int(c[-1] > c[0] and c[-1] > t0)
            beats[(kind, w)][1] += 1
        rows.append({"kind": kind, "writer": w, "family": fam, "reward": reward, "seed": seed,
                     "curve": c, "sampled_init": samp, "diverged": div, "theta0_final": t0})
    fams = {k: sorted({f for (kk, ww, f) in cells if (kk, ww) == k}) for k in {(a, b) for (a, b, _) in cells}}
    table = {}
    for w in ["astra-6", "sol-6.1", "opus-5.5"]:
        common = sorted(set(fams.get(("robust", w), [])) & set(fams.get(("standard", w), [])))
        for kind in ["standard", "robust"]:
            per_fam = [np.mean(cells[(kind, w, f)], axis=0) for f in common]
            table[f"{kind}|{w}"] = {"families": common, "curve": np.mean(per_fam, axis=0).round(3).tolist()}
    out = {"steps": STEPS, "table": table, "runs": rows,
           "beats_init_and_theta0": {f"{k[0]}|{k[1]}": v for k, v in beats.items()},
           "diverged": {f"{k[0]}|{k[1]}": v for k, v in diverged.items()},
           "greedy_vs_sampled_init": {f"{k[0]}|{k[1]}": [round(float(np.mean([a for a, _ in v])), 3),
                                                          round(float(np.mean([b for _, b in v])), 3)]
                                      for k, v in samp_gap.items()}}
    (R / "analysis_rev2").mkdir(exist_ok=True)
    (R / "analysis_rev2" / "stage2_robust.json").write_text(json.dumps(out, indent=1))
    body = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
            "Prompt & Writer (families) & " + " & ".join("0" if s == 0 else f"{round(s / 1000)}k" for s in STEPS)
            + "\\\\", "\\midrule"]
    for i, w in enumerate(["astra-6", "sol-6.1", "opus-5.5"]):
        if i:
            body.append("\\midrule")
        for kind, label in (("standard", "Standard"), ("robust", "Robust")):
            v = table[f"{kind}|{w}"]
            body.append(f"{label} & {NAMES[w]} ({len(v['families'])}) & " + " & ".join(f"{x:.2f}" for x in v["curve"])
                        + "\\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    (R / "analysis_rev2" / "tables").mkdir(parents=True, exist_ok=True)
    (R / "analysis_rev2" / "tables" / "robust_ppo.tex").write_text("\n".join(body) + "\n")
    print("steps(k)          ", " ".join(f"{s // 1000:>5d}" for s in STEPS))
    for k, v in table.items():
        print(f"{k:18s}", " ".join(f"{x:5.2f}" for x in v["curve"]), f"  ({len(v['families'])} fam)")
    print("diverged:", out["diverged"])
    print("beats init and RL(theta0):", out["beats_init_and_theta0"])
    print("init Q-lvl greedy vs sampled:", out["greedy_vs_sampled_init"])


if __name__ == "__main__":
    main()

"""Revision-2 Stage 2 analysis (plan §6), from runs/rev2_stage2 and runs/rev2_stage2_edit.

WSP->RL: per run (family, init, variant, reward, seed) the exact-FP32 Q-lvl/S-start of the final
student (eval.json); diverged runs (no student.npz) score 0 and are counted. The written init's
exact score is the W2 artifact's eval.json in runs/rev2_private (calibration rescales only the output
head, so greedy behaviour is unchanged). Training-time greedy checkpoints (ppo_eval.json) are TF32
and are used only for curves and for the sampled-vs-greedy diagnostic at step 0.
Decision rule (pre-registered): hybrid benefit for (family, writer, variant, reward) iff the
minimum over seeds of the final Q-lvl exceeds both the written init's Q-lvl and the maximum over
seeds of RL(theta0)'s final Q-lvl.

RL->WSP: edit sessions (summary / inspect) vs the RL base actor, fresh synthesis (W2 of the same
writer and family) and continued RL (rlcont_shaped_s*).
Writes runs/analysis_rev2/stage2.json and prints tables.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "runs"
S6 = ["ztl", "lys", "esm", "ruz", "crg", "mds"]


def q(path, metric="Q-lvl"):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())["sets"].get(metric, {}).get("solve_rate")


def step0(d):
    ev = d / "ppo_eval.json"
    if not ev.exists():
        return {}
    out = {}
    for e in json.loads(ev.read_text()):
        if e.get("eval_steps") == 0 and e.get("actor") == "init":
            out[(e["set"], e["decoding"])] = e["solve_rate"]
    return out


def stage2_rows():
    rows = []
    for meta in sorted((R / "rev2_stage2").glob("*/*/meta.json")):
        d = meta.parent
        fam, name = d.parent.name, d.name
        parts = name.split("_")
        seed = int(parts[-1][1:])
        reward = parts[-2]
        if parts[0] == "theta0":
            writer, variant = None, "theta0"
        elif parts[0] == "rlcont":
            writer, variant = None, "rlcont"
        else:
            writer, variant = "_".join(parts[:-3]), parts[-3]
        done = (d / "student.npz").exists()
        evaluated = (d / "eval.json").exists()
        fq = q(d / "eval.json") if evaluated else (0.0 if not done else None)
        fs = q(d / "eval.json", "S-start") if evaluated else (0.0 if not done else None)
        init_q = q(R / "rev2_private" / fam / writer / "weights_feedback_M_full_r0" / "eval.json") if writer else None
        s0 = step0(d)
        rows.append({"family": fam, "writer": writer, "variant": variant, "reward": reward, "seed": seed,
                     "diverged": not done, "evaluated": evaluated or not done, "final_qlvl": fq, "final_sstart": fs,
                     "init_qlvl": init_q, "init_greedy_sstart": s0.get(("S-start", "greedy")),
                     "init_sampled_sstart": s0.get(("S-start", "sampled")), "dir": str(d)})
    return rows


def summarize(rows):
    out = {"by_variant": {}, "decision": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["reward"])].append(r)
    for (v, rw), rs in sorted(groups.items()):
        ev = [r for r in rs if r["evaluated"] and r["final_qlvl"] is not None]
        fam_final = defaultdict(list)
        for r in ev:
            fam_final[r["family"]].append(r["final_qlvl"])
        init = [r["init_qlvl"] for r in rs if r["init_qlvl"] is not None]
        delta = [r["final_qlvl"] - r["init_qlvl"] for r in ev if r["init_qlvl"] is not None]
        out["by_variant"][f"{v}/{rw}"] = {
            "runs": len(rs), "evaluated": len(ev), "diverged": sum(r["diverged"] for r in rs),
            "final_qlvl_macro": float(np.mean([np.mean(x) for x in fam_final.values()])) if fam_final else math.nan,
            "init_qlvl_mean": float(np.mean(init)) if init else math.nan,
            "delta_mean": float(np.mean(delta)) if delta else math.nan,
            "frac_improved": float(np.mean([x > 0.02 for x in delta])) if delta else math.nan,
            "frac_degraded": float(np.mean([x < -0.02 for x in delta])) if delta else math.nan,
            "init_greedy_sstart": float(np.nanmean([r["init_greedy_sstart"] for r in rs if r["init_greedy_sstart"] is not None] or [math.nan])),
            "init_sampled_sstart": float(np.nanmean([r["init_sampled_sstart"] for r in rs if r["init_sampled_sstart"] is not None] or [math.nan])),
        }
    # decision rule per (family, writer, variant, reward)
    theta = defaultdict(list)
    for r in rows:
        if r["variant"] == "theta0" and r["final_qlvl"] is not None:
            theta[(r["family"], r["reward"])].append(r["final_qlvl"])
    cells = defaultdict(list)
    for r in rows:
        if r["writer"] and r["final_qlvl"] is not None:
            cells[(r["family"], r["writer"], r["variant"], r["reward"])].append(r)
    wins = defaultdict(lambda: [0, 0])
    for (fam, w, v, rw), rs in cells.items():
        if len(rs) < 2 or v == "matched":
            continue
        init = rs[0]["init_qlvl"] or 0.0
        th = max(theta.get((fam, rw), [0.0]))
        ok = min(r["final_qlvl"] for r in rs) > max(init, th)
        wins[f"{v}/{rw}"][0] += int(ok)
        wins[f"{v}/{rw}"][1] += 1
    out["decision"] = {k: {"hybrid_benefit_cells": a, "cells": b} for k, (a, b) in wins.items()}
    return out


def edit_rows():
    rows = []
    for sj in sorted((R / "rev2_stage2_edit").glob("*/*/*/session.json")):
        d = sj.parent
        fam, writer, arm = d.parent.parent.name, d.parent.name, d.name.split("_")[1]
        rec = json.loads(sj.read_text())
        rows.append({"family": fam, "writer": writer, "arm": arm, "run": int(d.name.split("_r")[-1]),
                     "qlvl": q(d / "eval.json") if rec.get("final_round") is not None else 0.0,
                     "sstart": q(d / "eval.json", "S-start") if rec.get("final_round") is not None else 0.0,
                     "delivered": rec.get("final_round") is not None})
    return rows


def edit_summary(rows):
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        base = q(R / "rev2_baselines" / fam / "ppo_shaped_s0" / "eval.json")
        cont = [q(p) for p in (R / "rev2_stage2" / fam).glob("rlcont_shaped_s*/eval.json")]
        cont = [c for c in cont if c is not None]
        out[fam] = {"rl_base_qlvl": base, "continued_rl_qlvl": float(np.mean(cont)) if cont else None, "writers": {}}
        for w in sorted({r["writer"] for r in rows if r["family"] == fam}):
            fresh = [q(R / "rev2_private" / fam / w / f"weights_feedback_M_full_r{k}" / "eval.json") for k in range(3)]
            fresh = [x for x in fresh if x is not None]
            e = {"fresh_W2_qlvl": float(np.mean(fresh)) if fresh else None}
            for arm in ("summary", "inspect"):
                xs = [r["qlvl"] for r in rows if r["family"] == fam and r["writer"] == w and r["arm"] == arm and r["qlvl"] is not None]
                e[arm] = float(np.mean(xs)) if xs else None
                e[arm + "_runs"] = len(xs)
            out[fam]["writers"][w] = e
    return out


if __name__ == "__main__":
    rows = stage2_rows()
    summ = summarize(rows)
    erows = edit_rows()
    esum = edit_summary(erows)
    (R / "analysis_rev2").mkdir(exist_ok=True)
    (R / "analysis_rev2" / "stage2.json").write_text(json.dumps({"rows": rows, "summary": summ, "edit_rows": erows,
                                                               "edit_summary": esum}, indent=1))
    print(f"{len(rows)} stage-2 runs; evaluated/diverged:",
          sum(r['evaluated'] for r in rows), sum(r['diverged'] for r in rows))
    print(f"{'variant/reward':22s} {'runs':>4s} {'div':>4s} {'init':>5s} {'final':>6s} {'Δ':>6s} {'impr':>5s} {'degr':>5s} {'S0 g/s':>11s}")
    for k, v in summ["by_variant"].items():
        print(f"{k:22s} {v['runs']:4d} {v['diverged']:4d} {v['init_qlvl_mean']:5.2f} {v['final_qlvl_macro']:6.2f} "
              f"{v['delta_mean']:6.2f} {v['frac_improved']:5.2f} {v['frac_degraded']:5.2f} "
              f"{v['init_greedy_sstart']:5.2f}/{v['init_sampled_sstart']:4.2f}")
    print("decision rule:", summ["decision"])
    print(json.dumps(esum, indent=1))

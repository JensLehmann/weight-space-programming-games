"""Revision-2 figures (PDF for LaTeX) from runs/analysis_rev2*/rows.json and baseline evals.

  fig_difficulty.pdf  Q-lvl by difficulty stratum for W2, P2, P-react and the best learned baseline
                      (Astra, Sol 6.1, Opus 5.5, Terra; family-macro within stratum, bootstrap CI)
  fig_support_vs_heldout.pdf  per-artifact S-start vs Q-lvl for written weights, Python players and
                      learned baselines (small multiples, one panel per policy source)
  fig_stage2.pdf      greedy Q-lvl over PPO steps from written init (std / structure-preserving /
                      SGD), RL(theta0) and matched-random, sparse vs shaped reward

Colours follow a validated categorical palette (blue = written weights, orange = Python,
aqua = learned); identity is never colour-alone (markers + legend + direct labels).
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8984"
YELLOW, MAGENTA = "#eda100", "#e87ba4"   # validated order: blue, orange, aqua, yellow, magenta
INK, INK2 = "#0b0b0b", "#52514e"
STRATA = ["easy", "medium", "hard", "very hard"]
STRATUM = {"ztl": "easy", "wih": "easy", "qcn": "easy", "lys": "medium", "esm": "medium", "aiv": "medium",
           "nkp": "medium", "obn": "medium", "qwv": "medium", "ruz": "medium", "vvf": "medium", "ubr": "medium",
           "yuz": "medium", "bvn": "medium", "trx": "hard", "yiy": "hard", "crg": "hard", "qpv": "hard",
           "ydv": "hard", "mds": "very hard"}
FRONTIER = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]


def style():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2,
                         "legend.frameon": False, "pdf.fonttype": 42})


def fam_means(rows, pred, metric="Q-lvl"):
    v = defaultdict(list)
    for r in rows:
        if r.get("evaluated") and pred(r) and not math.isnan(r.get(metric, math.nan)):
            v[r["family"]].append(r[metric])
    return {f: float(np.mean(x)) for f, x in v.items()}


def boot_ci(vals, B=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def baseline_fam(base_root, kinds):
    """Per family: mean over seeds of Q-lvl for each baseline kind; returns {kind: {fam: q}}."""
    out = defaultdict(lambda: defaultdict(list))
    for p in Path(base_root).glob("*/*/eval.json"):
        fam, name = p.parent.parent.name, p.parent.name
        kind = name.rsplit("_s", 1)[0]
        if kind.endswith("_memfix"):          # learned baselines use the recomputed-memory trainer
            kind = kind[: -len("_memfix")]
        elif kind in ("bc_support", "bc_aug", "bc_aug_dihedral", "bc_oracle"):
            continue
        if kind in kinds:
            q = json.loads(p.read_text())["sets"].get("Q-lvl", {}).get("solve_rate")
            if q is not None:
                out[kind][fam].append(q)
    return {k: {f: float(np.mean(v)) for f, v in d.items()} for k, d in out.items()}


def distill_fam(private_root, sub="distill_s0"):
    out = defaultdict(list)
    for p in Path(private_root).glob(f"*/*/python_feedback_M_full_r*/{sub}/eval.json"):
        q = json.loads(p.read_text())["sets"].get("Q-lvl", {}).get("solve_rate")
        if q is not None:
            out[p.parts[-5]].append(q)
    return {f: float(np.mean(v)) for f, v in out.items()}


def fig_difficulty(rows, c3rows, base, dist, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.9))     # as wide as the two-column legend above the axes
    fa_path = Path("../runs/analysis_rev2/fa_all.json")
    fa_fm = {}
    if fa_path.exists():
        fa = json.loads(fa_path.read_text())["private"]
        acc = defaultdict(list)
        for w in FRONTIER:
            for f, v in fa.get(w, {}).get("W2", {}).get("fa_by_family", {}).items():
                acc[f].append(v)
        fa_fm = {f: float(np.mean(v)) for f, v in acc.items()}
    series = [
        ("Written weights, fixed actor (W2)", BLUE, "o", fam_means(rows, lambda r: r["arm"] == "W2" and r["model"] in FRONTIER)),
        ("Written weights, free architecture (FA W2)", YELLOW, "v", fa_fm),
        ("Python player (P2)", ORANGE, "s", fam_means(rows, lambda r: r["arm"] == "P2" and r["model"] in FRONTIER)),
        ("Reactive Python (P-react)", MAGENTA, "^", fam_means(c3rows, lambda r: r["arm"] == "PR1" and r["model"] in FRONTIER)),
    ]
    learned = {}
    for f in STRATUM:
        cands = [d[f] for d in list(base.values()) + list(dist) if f in d]
        if cands:
            learned[f] = max(cands)
    series.append(("Best learned (non-privileged)", AQUA, "D", learned))
    x = np.arange(len(STRATA))
    for i, (name, col, mk, fm) in enumerate(series):
        ys, lo, hi = [], [], []
        for s in STRATA:
            m, l, h = boot_ci([v for f, v in fm.items() if STRATUM.get(f) == s])
            ys.append(m); lo.append(m - l if not math.isnan(m) else 0); hi.append(h - m if not math.isnan(m) else 0)
        off = (i - 2) * 0.05
        ax.errorbar(x + off, ys, yerr=[lo, hi], color=col, marker=mk, ms=4.5, capsize=0, lw=1.4, elinewidth=0.8,
                    label=name, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(x, [f"{s}\n(n={sum(1 for v in STRATUM.values() if v == s)})" for s in STRATA])
    ax.set_ylim(-0.03, 1.05); ax.set_ylabel("Q-lvl success (family macro)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=6.3, handlelength=1.6)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)



def fig_support_vs_heldout(rows, base_root, private_root, out):
    panels = [("Written weights (W2)", BLUE,
               [(r["S-start"], r["Q-lvl"]) for r in rows if r.get("evaluated") and r["arm"] == "W2"]),
              ("Python players (P2)", ORANGE,
               [(r["S-start"], r["Q-lvl"]) for r in rows if r.get("evaluated") and r["arm"] == "P2"])]
    for kind, name in (("bc_support", "BC-support"), ("bc_aug", "BC-aug"), ("ppo", "PPO"), ("ppo_shaped", "PPO-shaped")):
        pts = []
        sub = f"{kind}_memfix" if kind.startswith("bc_") else kind   # DAgger learners: recomputed memory
        for p in Path(base_root).glob(f"*/{sub}_s*/eval.json"):
            if p.parent.parent.name.startswith("official."):
                continue
            # exact kind only: the glob "ppo_s*" also matches ppo_shaped_s*; "_trunc" marks the
            # evaluated last checkpoint of a truncated run (it replaces the run's missing final eval)
            if not re.fullmatch(rf"{sub}_s\d+(_trunc)?", p.parent.name):
                continue
            s = json.loads(p.read_text())["sets"]
            pts.append((s.get("S-start", {}).get("solve_rate", math.nan), s.get("Q-lvl", {}).get("solve_rate", math.nan)))
        panels.append((name, AQUA, pts))
    pts = []
    for p in Path(private_root).glob("*/*/python_feedback_M_full_r*/distill_memfix_s0/eval.json"):
        s = json.loads(p.read_text())["sets"]
        pts.append((s.get("S-start", {}).get("solve_rate", math.nan), s.get("Q-lvl", {}).get("solve_rate", math.nan)))
    panels.append(("Distill-aug", AQUA, pts))
    pts = []
    for p in Path(private_root).glob("*/*/python_feedback_M_full_r*/distill_plain_memfix_s0/eval.json"):
        s = json.loads(p.read_text())["sets"]
        pts.append((s.get("S-start", {}).get("solve_rate", math.nan), s.get("Q-lvl", {}).get("solve_rate", math.nan)))
    if pts:
        panels.append(("Distill (no aug.)", AQUA, pts))
    fig, axs = plt.subplots(1, len(panels), figsize=(7.2, 1.75), sharex=True, sharey=True)
    for ax, (name, col, pts) in zip(axs, panels):
        pts = np.array([p for p in pts if not any(math.isnan(v) for v in p)]) if pts else np.zeros((0, 2))
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=7, color=col, alpha=0.55, edgecolors="none")
        ax.set_title(f"{name.replace(' (', chr(10) + '(')}\nn={len(pts)}", fontsize=6.0, color=INK)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("held-out levels", fontsize=7)
    fig.supxlabel("support starts", fontsize=7, y=0.02)
    fig.tight_layout(pad=0.4, w_pad=0.6); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_stage2(stage2_root, out):
    """Diverged runs (non-finite weights, hence no student.npz) score 0, as at the endpoints: their
    curve is extended at 0 over every checkpoint after the last one they recorded."""
    curves = defaultdict(lambda: defaultdict(list))   # (reward, init) -> step -> [q]
    runs = []                                         # (reward, init, {step: q}, diverged)
    for meta_p in Path(stage2_root).glob("*/*/meta.json"):
        d = meta_p.parent
        name = d.name                         # <writer>_<variant>_<reward>_s<k> or theta0_<reward>_s<k>
        parts = name.split("_")
        reward = parts[-2]
        if parts[0] == "rlcont":
            continue
        init = "RL(θ0)" if parts[0] == "theta0" else {"std": "written → PPO", "sp": "written → PPO (structure-preserving)",
                                                       "sgd": "written → SGD", "cal": "written (calibrated) → PPO",
                                                       "matched": "matched random → PPO"}[parts[-3]]
        ev = d / "ppo_eval.json"
        if not ev.exists():
            continue
        pts = {e["eval_steps"]: e["solve_rate"] for e in json.loads(ev.read_text())
               if e.get("set") == "Q-lvl" and e.get("decoding") == "greedy" and e.get("actor") in ("init", "current")}
        runs.append((reward, init, pts, not (d / "student.npz").exists()))
    grid = sorted({s for _, _, pts, div in runs if not div for s in pts})
    for reward, init, pts, div in runs:
        if div and pts:
            pts = {**pts, **{s: 0.0 for s in grid if s > max(pts)}}
        for s, v in pts.items():
            curves[(reward, init)][s].append(v)
    if not curves:
        return
    cols = {"written → PPO": BLUE, "written → PPO (structure-preserving)": VIOLET, "written → SGD": ORANGE,
            "written (calibrated) → PPO": "#e87ba4", "matched random → PPO": GREY, "RL(θ0)": AQUA}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, reward in zip(axs, ("sparse", "shaped")):
        for init, col in cols.items():
            d = curves.get((reward, init))
            if not d:
                continue
            steps = sorted(d)
            ax.plot([s / 1000 for s in steps], [np.mean(d[s]) for s in steps], color=col, marker="o", ms=3, label=init)
        ax.set_title(f"{reward} reward", fontsize=7.5); ax.set_xlabel("PPO steps (thousands)")
    axs[0].set_ylabel("greedy Q-lvl success")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=3, fontsize=6.5, handlelength=2.0, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.77), pad=0.6)
    fig.savefig(out); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="../runs/analysis_rev2")
    ap.add_argument("--c3", default="../runs/analysis_rev2_c3")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    style()
    rows = json.loads((Path(a.analysis) / "rows.json").read_text())
    c3rows = json.loads((Path(a.c3) / "rows.json").read_text()) if (Path(a.c3) / "rows.json").exists() else []
    base = baseline_fam("../runs/rev2_baselines", {"bc_support", "bc_aug", "bc_aug_dihedral", "ppo"})
    base = {k: {f: v for f, v in d.items() if not f.startswith("official.")} for k, d in base.items()}
    dist = [distill_fam("../runs/rev2_private", "distill_memfix_s0"), distill_fam("../runs/rev2_private", "distill_plain_memfix_s0")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_difficulty(rows, c3rows, base, dist, out / "fig_difficulty.pdf")
    fig_support_vs_heldout(rows, "../runs/rev2_baselines", "../runs/rev2_private", out / "fig_support_vs_heldout.pdf")
    fig_stage2("../runs/rev2_stage2", out / "fig_stage2.pdf")
    print("figures ->", out)

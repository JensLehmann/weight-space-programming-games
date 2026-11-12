"""Headline numbers of the paper from the exported data/ only (no runs directory, no games needed).

  * main table: family-macro level-balanced Q-lvl per writer and arm (W1, W2, P1, P2) on the
    private tier (protocol P2: information M, full actor, effort high), and W1 delivery rate; the
    W2 cells of the smaller and previous-generation writers come from their own tag (rev2_private_weak);
  * free architecture vs fixed actor (W1/W2 private, W2 public) with the paired family bootstrap of
    scripts/analyze_fa_all.py (same estimator and seed);
  * learned baselines in the fixed actor: family-macro of the raw Q-lvl solve rate per method, for the
    runs the paper reports (DAgger learners and PPO);
  * learned FA networks (scripts/analyze_learned_fa.py): the writer's own FA graph with its written
    weights vs re-initialised and trained by BC-support and by distillation, and the same learners in
    the fixed actor (level-balanced Q-lvl; runs without a saved student diverged and score 0);
  * Stage 2 (written init -> PPO): final Q-lvl macro per variant/reward;
  * bounded program player (pbounded, scripts/analyze_pbounded.py): NumPy players under the FA
    computation model, private tier, frontier writers, paired per family against FA W2, fixed-actor W2
    and P2 (per writer and pooled over the four writers), delivery rate and pooled means per stratum.

With --check the results are compared with the analysis outputs shipped in data/analysis/
(analysis_rev2/summary.json, analysis_rev2_weak/summary.json, fa_all.json, learned_fa.json and
pbounded.json, written by scripts/analyze_rev2.py, scripts/analyze_fa_all.py,
scripts/analyze_learned_fa.py and scripts/analyze_pbounded.py from the full runs directory); the
largest absolute deviation is printed. Sessions that finished
after those outputs were written can cause small deviations.

Usage: analyze_from_data.py [--data data] [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FRONTIER = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]
WRITERS = FRONTIER + ["luna-6", "sonnet-5.5", "haiku-5.5", "sol-5.6", "opus-5"]
ARMS = {("weights", "oneshot"): "W1", ("weights", "feedback"): "W2",
        ("python", "oneshot"): "P1", ("python", "feedback"): "P2"}
# private families in the order of scripts/analyze_fa_all.py::STRATUM (easy -> very hard); the paired
# bootstrap resamples families by index, so its CI depends on this order (public games: sorted)
FAMILY_ORDER = ["ztl", "wih", "qcn", "lys", "esm", "aiv", "nkp", "obn", "qwv", "ruz", "vvf", "ubr", "yuz", "bvn",
                "trx", "yiy", "crg", "qpv", "ydv", "mds"]
STRATUM = {f: "easy" if i < 3 else "medium" if i < 14 else "hard" if i < 19 else "very hard"
           for i, f in enumerate(FAMILY_ORDER)}                 # scripts/analyze_fa_all.py::STRATUM


def ordered(fams):
    """Families in FAMILY_ORDER, then any others sorted (public games)."""
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(fams, key=lambda f: (rank.get(f, len(rank)), f))


def num(x):
    return None if x in ("", None) else float(x)


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def fam_means(rows, **match) -> dict[str, float]:
    """Mean over runs of q_lvl_level_balanced per family, for rows matching all `match` fields."""
    d = defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in match.items()) and num(r["q_lvl_level_balanced"]) is not None:
            d[r["family"]].append(num(r["q_lvl_level_balanced"]))
    return {f: float(np.mean(v)) for f, v in d.items()}


def boot(x, B=10000):
    """Family bootstrap as in scripts/analyze_fa_all.py::boot (seed 0)."""
    x = np.asarray(x, float)
    if not len(x):
        return math.nan, math.nan, math.nan
    b = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main_table(rows):
    out = {}
    base = dict(tag="rev2_private", info="M", variant="full", effort="high")
    for w in WRITERS:
        for (cond, mode), arm in ARMS.items():
            fm = fam_means(rows, writer=w, condition=cond, mode=mode, **base)
            if not fm and w not in FRONTIER and arm == "W2":     # weaker writers' W2: tag rev2_private_weak
                fm = fam_means(rows, writer=w, condition=cond, mode=mode, **{**base, "tag": "rev2_private_weak"})
            if fm:
                out[(w, arm)] = float(np.mean(list(fm.values())))
        sess = [r for r in rows if r["writer"] == w and r["condition"] == "weights" and r["mode"] == "oneshot"
                and all(r[k] == v for k, v in base.items())]
        if sess:
            out[(w, "W1 delivery")] = float(np.mean([r["delivered"] == "True" for r in sess]))
    return out


def fa_vs_fixed(rows):
    res = {"private": {}, "public": {}}
    for w in WRITERS:
        weak = w not in FRONTIER          # smaller / previous-generation writers: tags rev2_*_weak
        for tag, mode in (("W1", "oneshot"), ("W2", "feedback")):
            fx_tag = "rev2_private_weak" if (weak and mode == "feedback") else "rev2_private"
            fx = fam_means(rows, tag=fx_tag, writer=w, condition="weights", mode=mode, info="M", variant="full")
            fa = fam_means(rows, tag="rev2_fa_weak" if weak else "rev2_fa", writer=w, condition="freearch", mode=mode, info="M")
            both = ordered(f for f in fx if f in fa)
            m, lo, hi = boot([fa[f] - fx[f] for f in both])
            res["private"].setdefault(w, {})[tag] = {
                "fixed": float(np.mean([fx[f] for f in both])) if both else math.nan,
                "fa": float(np.mean([fa[f] for f in both])) if both else math.nan,
                "diff": m, "ci": [lo, hi], "n": len(both), "fa_better": int(sum(fa[f] > fx[f] for f in both))}
        if weak:
            continue
        fx = fam_means(rows, tag="rev2_public", writer=w, condition="weights", mode="feedback", info="M", variant="full")
        fa = fam_means(rows, tag="rev2_fa_public", writer=w, condition="freearch", mode="feedback", info="M")
        p2 = fam_means(rows, tag="rev2_public", writer=w, condition="python", mode="feedback", info="M", variant="full")
        both = ordered(g for g in fx if g in fa)
        m, lo, hi = boot([fa[g] - fx[g] for g in both])
        res["public"][w] = {"fixed": float(np.mean([fx[g] for g in both])) if both else math.nan,
                            "fa": float(np.mean([fa[g] for g in both])) if both else math.nan,
                            "p2": float(np.mean([p2[g] for g in both if g in p2])) if both else math.nan,
                            "diff": m, "ci": [lo, hi], "n": len(both)}
    return res


def paper_baseline(r) -> bool:
    """Rows of the learned baselines in the fixed actor (DAgger learners and PPO)."""
    return r["student"] == "fixed_actor"


def baselines(brows, tier="private"):
    d = defaultdict(lambda: defaultdict(list))
    for r in brows:
        if r["tier"] == tier and paper_baseline(r) and num(r.get("rate_Q-lvl")) is not None:
            d[r["method"]][r["family"]].append(num(r["rate_Q-lvl"]))
    return {m: (float(np.mean([np.mean(v) for v in fams.values()])), len(fams)) for m, fams in sorted(d.items())}


def learned_fa(rows, brows):
    """As scripts/analyze_learned_fa.py: per private family, the FA graph of astra-6's FA W2 run 0 with its
    written weights, and re-initialised and trained (seed 0, recomputed memory, no augmentation) by
    BC-support and by distillation of astra-6's run-0 P2 player; a run without a saved student diverged
    (non-finite weights) and scores 0. The same learners in the fixed actor for comparison."""
    written = {r["family"]: r for r in rows if r["tag"] == "rev2_fa" and r["writer"] == "astra-6"
               and r["condition"] == "freearch" and r["mode"] == "feedback" and r["info"] == "M" and r["run"] == "0"}

    def pick(method, student, teacher):
        return {r["family"]: r for r in brows if r["tier"] == "private" and r["method"] == method
                and r["student"] == student and r["seed"] == "0"
                and r["variant"] == "" and r["teacher_writer"] == teacher and r["teacher_run"] == ("0" if teacher else "")}
    learned = {"bc": ("bc_support", ""), "distill": ("distill_plain", "astra-6")}
    fams = ordered(written)
    per = {}
    for f in fams:
        w = written[f]
        c = {"written_fa": num(w["q_lvl_level_balanced"]), "written_fa_support": num(w.get("rate_S-start"))}
        for name, (method, teacher) in learned.items():
            fa, fx = pick(method, "writer_fa_graph", teacher).get(f), pick(method, "fixed_actor", teacher).get(f)
            done = fa is not None and fa["evaluated"] == "True"
            c[f"{name}_fa"] = num(fa["q_lvl_level_balanced"]) if done else (0.0 if fa is not None else None)
            c[f"{name}_fa_support"] = num(fa["rate_S-start"]) if done else (0.0 if fa is not None else None)
            c[f"{name}_fa_diverged"] = fa is not None and fa["student_saved"] != "True"
            c[f"{name}_fixed"] = num(fx["q_lvl_level_balanced"]) if fx else None
            c[f"{name}_fixed_support"] = num(fx["rate_S-start"]) if fx else None
        per[f] = c
    summ = {}
    for k in ("written_fa", "written_fa_support", "bc_fa", "bc_fa_support", "distill_fa", "distill_fa_support",
              "bc_fixed", "bc_fixed_support", "distill_fixed", "distill_fixed_support"):
        xs = [per[f][k] for f in fams if per[f][k] is not None]
        summ[k] = float(np.mean(xs)) if xs else math.nan
    for name in learned:
        d = [per[f]["written_fa"] - per[f][f"{name}_fa"] for f in fams
             if per[f]["written_fa"] is not None and per[f][f"{name}_fa"] is not None]
        m, lo, hi = boot(d)
        summ[f"written_minus_{name}_fa"] = {"diff": m, "ci": [lo, hi], "written_better": int(sum(x > 0 for x in d)),
                                            "n": len(d)}
        summ[f"{name}_diverged"] = int(sum(per[f][f"{name}_fa_diverged"] for f in fams))
    return summ


PB_ARMS = {"fixed": dict(tag="rev2_private", condition="weights", mode="feedback", info="M", variant="full"),
           "fa": dict(tag="rev2_fa", condition="freearch", mode="feedback", info="M"),
           "pb": dict(tag="rev2_pbounded", condition="pbounded", mode="feedback", info="M"),
           "p2": dict(tag="rev2_private", condition="python", mode="feedback", info="M", variant="full")}
PB_DIFFS = (("pb", "fa"), ("p2", "pb"), ("pb", "fixed"))
TIE = 1e-6        # sessions.csv keeps 6 significant digits: smaller family-mean differences are ties


def pbounded(rows):
    """As scripts/analyze_pbounded.py: per frontier writer and pooled (per-family mean over the four writers),
    family-macro level-balanced Q-lvl of fixed W2, FA W2, the bounded player and P2 on the families all four
    arms have, paired differences with the family bootstrap; delivery rate; pooled bounded-player means per stratum."""
    per = {w: {a: fam_means(rows, writer=w, **m) for a, m in PB_ARMS.items()} for w in FRONTIER}
    pooled = {a: {f: float(np.mean([per[w][a][f] for w in FRONTIER])) for f in FAMILY_ORDER
                  if all(f in per[w][a] for w in FRONTIER)} for a in PB_ARMS}

    def compare(sc):
        common = [f for f in FAMILY_ORDER if all(f in sc[a] for a in PB_ARMS)]
        r = {"n": len(common), "mean": {a: float(np.mean([sc[a][f] for f in common])) for a in PB_ARMS}}
        for hi_, lo_ in PB_DIFFS:
            m, lo, hi = boot([sc[hi_][f] - sc[lo_][f] for f in common])
            r[f"{hi_}-{lo_}"] = {"diff": m, "ci": [lo, hi],
                                 "better": int(sum(sc[hi_][f] > sc[lo_][f] + TIE for f in common))}
        return r
    res = {"per_writer": {w: compare(per[w]) for w in FRONTIER}, "pooled": compare(pooled)}
    res["delivery"] = {w: float(np.mean([r["delivered"] == "True" for r in rows if r["writer"] == w
                                         and all(r[k] == v for k, v in PB_ARMS["pb"].items())])) for w in FRONTIER}
    res["strata"] = {s: float(np.mean([v for f, v in pooled["pb"].items() if STRATUM[f] == s]))
                     for s in ("easy", "medium", "hard", "very hard")}
    return res


def stage2(srows):
    d = defaultdict(lambda: defaultdict(list))
    for r in srows:
        if r["evaluated"] == "True" and num(r["final_qlvl"]) is not None:
            d[f"{r['variant']}/{r['reward']}"][r["family"]].append(num(r["final_qlvl"]))
    return {k: float(np.mean([np.mean(v) for v in fams.values()])) for k, fams in sorted(d.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    rows = load(data / "sessions.csv")

    mt = main_table(rows)
    print("Main table (private tier, family-macro level-balanced Q-lvl)")
    print(f"{'writer':12s} " + " ".join(f"{x:>6s}" for x in ("W1", "W2", "P1", "P2", "W1del")))
    for w in WRITERS:
        vals = [mt.get((w, k)) for k in ("W1", "W2", "P1", "P2", "W1 delivery")]
        print(f"{w:12s} " + " ".join("    --" if v is None else f"{v:6.3f}" for v in vals))

    fa = fa_vs_fixed(rows)
    print("\nFree architecture vs fixed actor (family-macro; paired family bootstrap)")
    for w, c in fa["private"].items():
        for t in ("W1", "W2"):
            x = c[t]
            print(f"  private {w:10s} {t}: fixed {x['fixed']:.3f} FA {x['fa']:.3f} diff {x['diff']:+.3f} "
                  f"[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}] FA>fixed {x['fa_better']}/{x['n']}")
    for w, x in fa["public"].items():
        print(f"  public  {w:10s} W2: fixed {x['fixed']:.3f} FA {x['fa']:.3f} P2 {x['p2']:.3f} diff {x['diff']:+.3f} "
              f"[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}] ({x['n']} games)")

    pb = pbounded(rows)
    print("\nBounded program player (pbounded), private tier (family-macro level-balanced Q-lvl; paired family "
          "bootstrap)")
    print(f"  {'writer':10s} {'fixed W2':>8s} {'FA W2':>6s} {'bounded':>7s} {'P2':>6s}  bounded-FA / P2-bounded / "
          "bounded-fixed")
    for w, x in list(pb["per_writer"].items()) + [("pooled", pb["pooled"])]:
        m = x["mean"]
        print(f"  {w:10s} {m['fixed']:8.3f} {m['fa']:6.3f} {m['pb']:7.3f} {m['p2']:6.3f}  " + " / ".join(
            f"{x[k]['diff']:+.3f} [{x[k]['ci'][0]:+.3f},{x[k]['ci'][1]:+.3f}] ({x[k]['better']}/{x['n']})"
            for k in ("pb-fa", "p2-pb", "pb-fixed")))
    print("  delivery " + ", ".join(f"{w} {v:.3f}" for w, v in pb["delivery"].items())
          + " | pooled bounded per stratum " + ", ".join(f"{s} {v:.3f}" for s, v in pb["strata"].items()))

    lfa = None
    if (data / "baselines.csv").exists():
        brows = load(data / "baselines.csv")
        print("\nLearned baselines, private tier, fixed actor (recomputed-memory DAgger learners and PPO;"
              " family-macro raw Q-lvl; families)")
        for m, (v, n) in baselines(brows).items():
            print(f"  {m:16s} {v:.3f} ({n})")
        lfa = learned_fa(rows, brows)
        print("\nLearned FA networks, private tier (family-macro level-balanced Q-lvl / S-start)")
        print(f"  written (astra-6 FA W2 r0)     {lfa['written_fa']:.3f} / {lfa['written_fa_support']:.3f}")
        for name, lab in (("bc", "BC-support"), ("distill", "distill (no aug.)")):
            w = lfa[f"written_minus_{name}_fa"]
            print(f"  learned FA, {lab:18s} {lfa[name + '_fa']:.3f} / {lfa[name + '_fa_support']:.3f}  written - learned "
                  f"{w['diff']:+.3f} [{w['ci'][0]:+.3f},{w['ci'][1]:+.3f}] written better {w['written_better']}/{w['n']}, "
                  f"diverged {lfa[name + '_diverged']}")
            print(f"  fixed actor, {lab:17s} {lfa[name + '_fixed']:.3f} / {lfa[name + '_fixed_support']:.3f}")
    if (data / "stage2_rl.csv").exists():
        print("\nStage 2, written init -> PPO (family-macro final Q-lvl)")
        for k, v in stage2(load(data / "stage2_rl.csv")).items():
            print(f"  {k:18s} {v:.3f}")

    if a.check:
        dev = []
        summ = json.loads((data / "analysis" / "analysis_rev2" / "summary.json").read_text())
        for c in summ["Q-lvl"]:
            k = (c["model"], c["arm"])
            if c["tag"] == "rev2_private" and c["info"] == "M" and c["variant"] == "full" and c["effort"] == "high" \
                    and k in mt and c["mean"] == c["mean"]:
                dev.append(abs(mt[k] - c["mean"]))
        weak = data / "analysis" / "analysis_rev2_weak" / "summary.json"
        if weak.exists():                    # weaker writers' W2 (tag rev2_private_weak)
            for c in json.loads(weak.read_text())["Q-lvl"]:
                k = (c["model"], c["arm"])
                if c["tag"] == "rev2_private_weak" and c["info"] == "M" and c["variant"] == "full" \
                        and c["effort"] == "high" and k in mt and c["mean"] == c["mean"]:
                    dev.append(abs(mt[k] - c["mean"]))
        ref = json.loads((data / "analysis" / "analysis_rev2" / "fa_all.json").read_text())
        n_count_mismatch = 0
        for tier in ("private", "public"):
            for w, c in ref[tier].items():
                ours = fa[tier][w]
                for t, x in (c.items() if tier == "private" else [(None, c)]):
                    o = ours[t] if t else ours
                    dev += [abs(o[k] - x[k]) for k in ("fixed", "fa", "diff") if x[k] == x[k]]
                    dev += [abs(a - b) for a, b in zip(o["ci"], x["ci"]) if b == b]
                    n_count_mismatch += o["n"] != x.get("n", o["n"])     # families / public games compared
        ref_lfa = data / "analysis" / "analysis_rev2" / "learned_fa.json"
        if lfa is not None and ref_lfa.exists():
            rs = json.loads(ref_lfa.read_text())["summary"]
            for k, x in rs.items():
                if isinstance(x, dict):              # written - learned: mean difference, families better
                    dev += [abs(lfa[k]["diff"] - x["diff"])] + [abs(a - b) for a, b in zip(lfa[k]["ci"], x["ci"])]
                    n_count_mismatch += lfa[k]["written_better"] != x["written_better"]
                elif k.endswith("_diverged"):
                    n_count_mismatch += lfa[k] != x
                elif k in lfa and x == x:
                    dev.append(abs(lfa[k] - x))
        ref_pb = data / "analysis" / "analysis_rev2" / "pbounded.json"
        if ref_pb.exists():                          # bounded program player (scripts/analyze_pbounded.py)
            rp = json.loads(ref_pb.read_text())
            for o, x in [(pb["per_writer"][w], c) for w, c in rp["per_writer"].items()] + [(pb["pooled"], rp["pooled"])]:
                n_count_mismatch += o["n"] != x["n"]
                dev += [abs(o["mean"][a] - x["mean"][a]) for a in x["mean"]]
                for k in ("pb-fa", "p2-pb", "pb-fixed"):
                    dev += [abs(o[k]["diff"] - x[k]["diff"])] + [abs(a - b) for a, b in zip(o[k]["ci"], x[k]["ci"])]
                    n_count_mismatch += o[k]["better"] != x[k]["better"]
            dev += [abs(pb["delivery"][w] - v) for w, v in rp["delivery"].items()]
            dev += [abs(pb["strata"][s] - v) for s, v in rp["strata"].items()]
        print(f"\ncheck: {len(dev)} values compared with data/analysis; max |deviation| = {max(dev):.4f}"
              + (f"; {n_count_mismatch} count mismatches" if n_count_mismatch else ""))


if __name__ == "__main__":
    main()

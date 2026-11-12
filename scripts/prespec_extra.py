"""Protocol-specified secondary analyses not shown elsewhere in the paper (private tier, frontier writers).

From the released per-session record (data/sessions.csv), main protocol (information M, full actor, effort high),
for fixed-actor W2, P2 and FA W2 per frontier writer:
  * delivery: fraction of sessions with a final artifact;
  * Q-lvl (primary): family macro, runs averaged within a family first, undelivered sessions score 0;
  * conditional on delivery: mean over delivered sessions (= primary / delivery, since every family has 3 runs);
  * excluding undelivered sessions: family macro with family means over delivered runs only (a family without a
    delivered run drops out);
  * Q-pal: palette-permuted held-out levels (seeds 10-19, reviewed starts), family macro over the families where
    the palette changes the frame, undelivered sessions score 0;
  * k/n runs with Q-lvl >= 0.5 and a Wilson 95% interval.
The same cells for the one-shot arms W1 and FA W1 (lower delivery) are stored in the JSON, not tabulated.
Q-lvl is the level-balanced session value (q_lvl_level_balanced), as in the main tables.
Also: best-of-k(W1) against W2 at matched tokens. k is the ratio of mean W2 to mean W1 tokens per session,
rounded and capped at 3 (the number of one-shot runs per family); best-of-k picks, per family, the run with the
most support-start successes (ties to the lower run index; undelivered runs rank last) and averages over all
k-subsets of the three runs.
Writes runs/analysis_rev2/prespec_extra.json and the LaTeX table prespec_extra.tex (--tables).

usage: prespec_extra.py [--sessions <sessions.csv>] [--out <json>] [--tables <dir>]
"""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tables_rev2 import FRONTIER, LABEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"W2": ("rev2_private", "weights", "feedback"), "P2": ("rev2_private", "python", "feedback"),
        "FA W2": ("rev2_fa", "freearch", "feedback")}
W1 = ("rev2_private", "weights", "oneshot")
ONESHOT = {"W1": W1, "FA W1": ("rev2_fa", "freearch", "oneshot")}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def num(x):
    return float(x) if x not in ("", None) else math.nan


def select(rows, arm, writer):
    tag, cond, mode = arm
    return [r for r in rows if (r["tier"], r["tag"], r["condition"], r["mode"], r["info"], r["variant"],
                                r["effort"], r["writer"]) == ("private", tag, cond, mode, "M", "full", "high", writer)]


def fam_macro(sess, key, delivered_only=False):
    v = defaultdict(list)
    for r in sess:
        if delivered_only and r["delivered"] != "True":
            continue
        x = num(r[key])
        if x == x:
            v[r["family"]].append(x)
    return float(np.mean([np.mean(x) for x in v.values()])), len(v)


def cell(sess):
    n = len(sess)
    deliv = [r for r in sess if r["delivered"] == "True"]
    q = [num(r["q_lvl_level_balanced"]) for r in sess]
    k = sum(x >= 0.5 for x in q)
    prim, nf = fam_macro(sess, "q_lvl_level_balanced")
    excl, nf_excl = fam_macro(sess, "q_lvl_level_balanced", delivered_only=True)
    pal, nf_pal = fam_macro(sess, "rate_Q-pal")
    return {"n_sessions": n, "n_delivered": len(deliv), "delivery": len(deliv) / n,
            "q_lvl": prim, "n_families": nf,
            "q_lvl_cond_delivery": float(np.mean([num(r["q_lvl_level_balanced"]) for r in deliv])),
            "q_lvl_excl_undelivered": excl, "n_families_excl": nf_excl,
            "q_pal": pal, "n_families_pal": nf_pal,
            "runs_ge_half": [int(k), n], "runs_ge_half_wilson": wilson(k, n)}


def tokens(sess, kind):
    if kind == "total":
        return float(np.mean([num(r["tokens_in"]) + num(r["tokens_out"]) for r in sess]))
    return float(np.mean([num(r["tokens_out"]) for r in sess]))


def best_of_k(w1, k):
    """Family macro of best-of-k one-shot runs, selected by support-start success, averaged over k-subsets."""
    by_fam = defaultdict(dict)
    for r in w1:
        by_fam[r["family"]][int(r["run"])] = r
    fam_vals = []
    for fam, runs in by_fam.items():
        idx = sorted(runs)
        vals = []
        for sub in itertools.combinations(idx, k):
            score = lambda i: num(runs[i]["support_success"]) if runs[i]["delivered"] == "True" else -1.0
            best = max(sub, key=lambda i: (score(i), -i))
            vals.append(num(runs[best]["q_lvl_level_balanced"]))
        fam_vals.append(np.mean(vals))
    return float(np.mean(fam_vals))


def matched(rows, writer):
    w1 = select(rows, W1, writer)
    w2 = select(rows, ARMS["W2"], writer)
    out = {"W1": fam_macro(w1, "q_lvl_level_balanced")[0], "W2": fam_macro(w2, "q_lvl_level_balanced")[0]}
    for kind in ("output", "total"):
        ratio = tokens(w2, kind) / tokens(w1, kind)
        k = int(min(3, max(1, round(ratio))))
        out[kind] = {"ratio": ratio, "k": k, "capped": round(ratio) > 3, "best_of_k": best_of_k(w1, k),
                     "tokens_W1": tokens(w1, kind), "tokens_W2": tokens(w2, kind)}
    out["best_of"] = {k: best_of_k(w1, k) for k in (1, 2, 3)}
    out["best_of_3_token_share_total"] = 3 * tokens(w1, "total") / tokens(w2, "total")
    return out


def table(res):
    f2 = lambda x: f"{x:.2f}"
    L = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Arm & Writer & Deliv. & Q-lvl & Cond. deliv. & Excl. undeliv. & Q-pal & runs $\ge0.5$ [Wilson 95\% CI]\\"]
    for arm in ARMS:
        L.append(r"\midrule")
        for i, w in enumerate(FRONTIER):
            c = res["cells"][arm][w]
            lo, hi = c["runs_ge_half_wilson"]
            excl = f2(c["q_lvl_excl_undelivered"]) + ("" if c["n_families_excl"] == c["n_families"]
                                                      else f" ({c['n_families_excl']})")
            L.append(f"{arm if i == 0 else ''} & {LABEL[w]} & {f2(c['delivery'])} & {f2(c['q_lvl'])} & "
                     f"{f2(c['q_lvl_cond_delivery'])} & {excl} & {f2(c['q_pal'])} & "
                     f"{c['runs_ge_half'][0]}/{c['runs_ge_half'][1]} [{lo:.2f}, {hi:.2f}]" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(Path(__file__).resolve().parents[1] / "data" / "sessions.csv"))
    ap.add_argument("--out", default=str(ROOT / "runs/analysis_rev2/prespec_extra.json"))
    ap.add_argument("--tables", default=str(ROOT / "runs/analysis_rev2/tables"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.sessions)))
    res = {"cells": {arm: {w: cell(select(rows, ARMS[arm], w)) for w in FRONTIER} for arm in ARMS},
           "cells_oneshot": {arm: {w: cell(select(rows, ONESHOT[arm], w)) for w in FRONTIER} for arm in ONESHOT},
           "matched_token_best_of_k": {w: matched(rows, w) for w in FRONTIER}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.tables).mkdir(parents=True, exist_ok=True)
    (Path(a.tables) / "prespec_extra.tex").write_text(table(res))
    print(json.dumps(res, indent=1))

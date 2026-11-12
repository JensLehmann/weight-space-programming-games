"""Revision-2 analysis (protocol P2 §Analysis).

Reads runs/<tag>/<fam>/<model>/<cond>_<mode>_<info>_<variant>_r<k>/{session,eval,diag,blind}.json for
every tag starting with 'rev2', computes per-session metrics (Q-lvl, Q-lvl-mid, Q-lvl-final,
seen/held-out templates, RESET-free, secondary sets) and aggregates per (model, arm):
equal-weight-per-family means (runs averaged within family first), hierarchical bootstrap CIs
(families -> held-out levels -> runs), per-stratum means, k/n runs with Q-lvl >= 0.5 (Wilson),
missingness, and paired W-vs-P differences.

Usage: analyze_rev2.py [--tags rev2_private,...] [--boot 10000] [--out runs/analysis_rev2]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsp import private
from wsp.manifest import RUNS
from wsp.queries import load_sets

STRATA = {
    "easy": ["ztl", "wih", "qcn"],
    "medium": ["lys", "esm", "aiv", "nkp", "obn", "qwv", "ruz", "vvf", "ubr", "yuz", "bvn"],
    "hard": ["trx", "yiy", "crg", "qpv", "ydv"],
    "very hard": ["mds"],
}
PRIVATE20 = [f for v in STRATA.values() for f in v]
STRATUM_OF = {f: s for s, fs in STRATA.items() for f in fs}
SUPPORT_TEMPLATES = set(range(5))
# families with fewer than 10 renderer templates (renderer = seed % k): from the private module, empty without it
N_TEMPLATES: dict[str, int] = private.get("N_TEMPLATES", {})


def template_of(fam, seed):
    return seed % N_TEMPLATES.get(fam, 10)
ARM = {("weights", "oneshot"): "W1", ("weights", "feedback"): "W2",
       ("python", "oneshot"): "P1", ("python", "feedback"): "P2",
       ("pyreact", "oneshot"): "PR1", ("pyreact", "feedback"): "PR2"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


_index_cache = {}


def index(fam):
    """key -> (qset, level, seed, perturbed) for the rev-2 manifest; plus the final level."""
    if fam not in _index_cache:
        sets = load_sets(fam, rev=2)
        idx = {}
        for qs, specs in sets.items():
            for s in specs:
                idx[(qs, s.key)] = (s.level, s.seed, bool(s.prefix))
        levels = sorted({s.level for s in sets["Q-lvl"]})
        _index_cache[fam] = (idx, levels[-1] if levels else None)
    return _index_cache[fam]


def outcome(rec):
    if rec.get("final_round") is not None:
        return "artifact"
    return rec["candidates"][-1].get("outcome", "unknown") if rec.get("candidates") else "unknown"


def session_metrics(fam, ev):
    """Per-level success arrays for Q-lvl (overall, mid, final) and scalar secondary metrics."""
    idx, final = index(fam)
    per_level = defaultdict(list)
    m = {}
    tpl = {"seen": [], "held": []}
    reset_free_pert = []
    for e in ev["episodes"].get("Q-lvl", []):
        key, succ = e[0], e[1]
        resets = e[3] if len(e) > 3 else 0
        lvl, seed, pert = idx[("Q-lvl", key)]
        per_level[lvl].append(succ)
        tpl["seen" if template_of(fam, seed) in SUPPORT_TEMPLATES else "held"].append(succ)
        if pert:
            reset_free_pert.append(int(succ and resets == 0))
    m["Q-lvl_by_level"] = {int(k): float(np.mean(v)) for k, v in per_level.items()}
    lv = m["Q-lvl_by_level"]
    m["Q-lvl"] = float(np.mean(list(lv.values()))) if lv else math.nan
    m["Q-lvl-final"] = lv.get(final, math.nan)
    mid = [v for k, v in lv.items() if k != final]
    m["Q-lvl-mid"] = float(np.mean(mid)) if mid else math.nan
    m["Q-lvl-tpl-seen"] = float(np.mean(tpl["seen"])) if tpl["seen"] else math.nan
    m["Q-lvl-tpl-held"] = float(np.mean(tpl["held"])) if tpl["held"] else math.nan
    m["Q-lvl-pert-noreset"] = float(np.mean(reset_free_pert)) if reset_free_pert else math.nan
    for s in ("S-start", "Q-tpl", "Q-off", "Q-pal"):
        eps = ev["episodes"].get(s, [])
        if s == "Q-tpl":   # only renderers never shown in support (with k < 10 templates, seeds >= k reuse them)
            eps = [e for e in eps if template_of(fam, idx[("Q-tpl", e[0])][1]) not in SUPPORT_TEMPLATES]
        m[s] = float(np.mean([e[1] for e in eps])) if eps else math.nan
    allq = [e for e in ev["episodes"].get("Q-lvl", [])]
    m["invalid_per_ep"] = float(np.mean([e[4] for e in allq])) if allq and len(allq[0]) > 4 else math.nan
    m["resets_per_ep"] = float(np.mean([e[3] for e in allq])) if allq and len(allq[0]) > 3 else math.nan
    return m


def collect(tags):
    rows = []
    for tag in tags:
        for sess in sorted((RUNS / tag).glob("*/*/*/session.json")):
            d = sess.parent
            fam, model = d.parent.parent.name, d.parent.name
            name = d.name
            cond, mode, info, variant, r = name.split("_", 4)
            rec = json.loads(sess.read_text())
            row = {"tag": tag, "family": fam, "model": model, "condition": cond, "mode": mode,
                   "info": info, "variant": variant, "run": int(r[1:]), "effort": rec.get("effort"),
                   "arm": ARM.get((cond, mode), f"{cond}:{mode}"), "outcome": outcome(rec),
                   "tokens_out": rec.get("tokens_out"), "rounds": rec.get("rounds_used"), "dir": str(d)}
            ev_p = d / "eval.json"
            if ev_p.exists():
                row.update(session_metrics(fam, json.loads(ev_p.read_text())))
                row["evaluated"] = True
            else:
                row["evaluated"] = False
            for extra in ("diag.json", "blind.json"):
                if (d / extra).exists():
                    row[extra[:-5]] = json.loads((d / extra).read_text())
            rows.append(row)
    return rows


def cell_key(r):
    return (r["model"], r["arm"], r["info"], r["variant"], r["effort"], r["tag"])


def hier_boot(fam_run_levels, B, rng):
    """fam_run_levels: list over families of arrays [runs, levels] (per-level success means).
    Resample families, then levels within family, then runs. Returns (mean, lo, hi)."""
    if not fam_run_levels:
        return (math.nan, math.nan, math.nan)
    point = float(np.mean([a.mean() for a in fam_run_levels]))
    F = len(fam_run_levels)
    stats = np.empty(B)
    for b in range(B):
        fi = rng.integers(0, F, F)
        vals = []
        for i in fi:
            a = fam_run_levels[i]
            li = rng.integers(0, a.shape[1], a.shape[1])
            ri = rng.integers(0, a.shape[0], a.shape[0])
            vals.append(a[np.ix_(ri, li)].mean())
        stats[b] = np.mean(vals)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (point, float(lo), float(hi))


def aggregate(rows, B=10000, seed=0, metric="Q-lvl"):
    rng = np.random.default_rng(seed)
    cells = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)
    out = []
    for ck, rs in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        by_fam = defaultdict(list)
        for r in rs:
            by_fam[r["family"]].append(r)
        fam_mean, mats = {}, []
        for fam, frs in by_fam.items():
            ev = [r for r in frs if r["evaluated"]]
            if not ev:
                continue
            levels = sorted({k for r in ev for k in r["Q-lvl_by_level"]})
            if metric == "Q-lvl" and levels:
                # failed sessions (no artifact) have 0 on every level: their eval rows are zeros already
                mat = np.array([[r["Q-lvl_by_level"].get(l, 0.0) for l in levels] for r in ev])
                mats.append(mat)
            fam_mean[fam] = float(np.nanmean([r[metric] for r in ev]))
        strata = {}
        for s, fs in STRATA.items():
            v = [fam_mean[f] for f in fs if f in fam_mean]
            strata[s] = float(np.mean(v)) if v else math.nan
        boot = hier_boot(mats, B, rng) if metric == "Q-lvl" else (
            float(np.mean(list(fam_mean.values()))) if fam_mean else math.nan, math.nan, math.nan)
        ev_rows = [r for r in rs if r["evaluated"]]
        k = sum(r["Q-lvl"] >= 0.5 for r in ev_rows)
        outc = defaultdict(int)
        for r in rs:
            outc[r["outcome"]] += 1
        out.append({"model": ck[0], "arm": ck[1], "info": ck[2], "variant": ck[3], "effort": ck[4],
                    "tag": ck[5], "metric": metric, "n_sessions": len(rs), "n_families": len(fam_mean),
                    "mean": boot[0], "ci": [boot[1], boot[2]], "strata": strata, "by_family": fam_mean,
                    "runs_ge_half": [k, len(ev_rows)], "runs_ge_half_wilson": wilson(k, len(ev_rows)),
                    "outcomes": dict(outc),
                    "delivery_rate": outc.get("artifact", 0) / len(rs) if rs else math.nan})
    return out


def paired(rows, a="W1", b="P1", metric="Q-lvl", B=10000, seed=1):
    """Per model: family-level difference (mean over runs) a - b, bootstrap over families."""
    rng = np.random.default_rng(seed)
    fm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["evaluated"] and r["info"] == "M" and r["variant"] == "full" and r["arm"] in (a, b):
            fm[(r["model"], r["effort"])][(r["family"], r["arm"])].append(r[metric])
    res = []
    for (model, effort), d in sorted(fm.items(), key=lambda kv: str(kv[0])):
        fams = sorted({f for f, _ in d if (f, a) in d and (f, b) in d})
        if not fams:
            continue
        diff = np.array([np.mean(d[(f, a)]) - np.mean(d[(f, b)]) for f in fams])
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
        res.append({"model": model, "effort": effort, "a": a, "b": b, "metric": metric, "n_families": len(fams),
                    "diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "families_a_better": int((diff > 0).sum()), "families_b_better": int((diff < 0).sum())})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=str(RUNS / "analysis_rev2"))
    a = ap.parse_args()
    tags = a.tags.split(",") if a.tags else sorted(p.name for p in RUNS.glob("rev2*") if p.is_dir()
                                                    and not p.name.startswith("rev2_floors"))
    if "rev2_public" in tags:
        from wsp.public_scope import assert_public_cohort_complete
        assert_public_cohort_complete()
    rows = collect(tags)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows))
    summary = {"tags": tags, "n_rows": len(rows)}
    for metric in ("Q-lvl", "Q-lvl-mid", "Q-lvl-final", "Q-lvl-tpl-seen", "Q-lvl-tpl-held",
                   "Q-lvl-pert-noreset", "S-start", "Q-tpl", "Q-off", "Q-pal"):
        summary[metric] = aggregate(rows, B=a.boot if metric == "Q-lvl" else 0, metric=metric)
    summary["paired"] = paired(rows, "W1", "P1", B=a.boot) + paired(rows, "W2", "P2", B=a.boot) + \
        paired(rows, "W2", "W1", B=a.boot)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(rows)} sessions over tags {tags}")
    print(f"{'model':12s} {'arm':4s} {'info':3s} {'var':8s} {'eff':6s} {'fam':>3s} {'Q-lvl':>6s} {'95% CI':>15s} "
          f"{'mid':>5s} {'final':>5s} {'deliv':>5s}")
    mid = {(c["model"], c["arm"], c["info"], c["variant"], c["effort"], c["tag"]): c for c in summary["Q-lvl-mid"]}
    fin = {(c["model"], c["arm"], c["info"], c["variant"], c["effort"], c["tag"]): c for c in summary["Q-lvl-final"]}
    for c in summary["Q-lvl"]:
        k = (c["model"], c["arm"], c["info"], c["variant"], c["effort"], c["tag"])
        print(f"{c['model']:12s} {c['arm']:4s} {c['info']:3s} {c['variant']:8s} {str(c['effort']):6s} "
              f"{c['n_families']:3d} {c['mean']:6.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}] "
              f"{mid[k]['mean']:5.2f} {fin[k]['mean']:5.2f} {c['delivery_rate']:5.2f}")

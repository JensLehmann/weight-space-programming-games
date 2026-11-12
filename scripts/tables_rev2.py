"""Revision-2 LaTeX tables (*.tex files for the paper).

Inputs: runs/analysis_rev2 (private tier), runs/analysis_rev2_weak (weaker writers' W2, tag rev2_private_weak),
runs/analysis_rev2_c3 (ablations), runs/analysis_rev2_public,
runs/rev2_baselines (BC/PPO students), runs/rev2_private/**/distill_s0, runs/rev2_floors, runs/rev2_stage2,
runs/rev2_stage2_edit.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATUM = {"ztl": "easy", "wih": "easy", "qcn": "easy", "lys": "medium", "esm": "medium", "aiv": "medium",
           "nkp": "medium", "obn": "medium", "qwv": "medium", "ruz": "medium", "vvf": "medium", "ubr": "medium",
           "yuz": "medium", "bvn": "medium", "trx": "hard", "yiy": "hard", "crg": "hard", "qpv": "hard",
           "ydv": "hard", "mds": "very hard"}
STRATA = ["easy", "medium", "hard", "very hard"]
S6 = ["ztl", "lys", "esm", "ruz", "crg", "mds"]
MODELS = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1", "luna-6", "sonnet-5.5", "haiku-5.5", "sol-5.6", "opus-5"]
FRONTIER = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]
LABEL = {"astra-6": "GPT-6 Astra", "sol-6.1": "GPT-6.1 Sol", "opus-5.5": "Claude Opus 5.5", "terra-6.1": "GPT-6.1 Terra",
         "luna-6": "GPT-6 Luna", "sonnet-5.5": "Claude Sonnet 5.5", "haiku-5.5": "Claude Haiku 5.5",
         "sol-5.6": "GPT-5.6 Sol$^\\dagger$", "opus-5": "Claude Opus 5$^\\dagger$"}
R = Path("../runs")


def f2(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def ci(c):
    if c is None:
        return "--"
    lo, hi = c["ci"]
    if lo != lo:
        return f2(c["mean"])
    return f"{c['mean']:.2f} {{\\scriptsize[{lo:.2f},{hi:.2f}]}}"


def fam_mean(rows, pred, metric="Q-lvl", fams=None):
    v = defaultdict(list)
    for r in rows:
        if r.get("evaluated") and pred(r) and (fams is None or r["family"] in fams):
            x = r.get(metric, math.nan)
            if x == x:
                v[r["family"]].append(x)
    return {f: float(np.mean(x)) for f, x in v.items()}


def macro(d, fams=None):
    vals = [v for f, v in d.items() if fams is None or f in fams]
    return float(np.mean(vals)) if vals else math.nan


def boot(vals, B=10000, seed=0):
    vals = np.asarray(vals, float)
    if not len(vals):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    b = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


DAGGER_KINDS = ("bc_support", "bc_aug", "bc_aug_dihedral", "bc_oracle")


def baseline_kind(dirname):
    """Learned baselines use the recurrent trainer with recomputed memory (dirs tagged _memfix);
    stored-memory DAgger runs are not used. PPO runs are unaffected. Returns None to skip."""
    kind = dirname.rsplit("_s", 1)[0]
    if kind.endswith("_memfix"):
        return kind[: -len("_memfix")]
    return None if kind in DAGGER_KINDS else kind


def baselines():
    out = defaultdict(lambda: defaultdict(list))
    for p in (R / "rev2_baselines").glob("*/*/eval.json"):
        fam, kind = p.parent.parent.name, baseline_kind(p.parent.name)
        if kind is None or "learnedfa" in kind:
            continue
        s = json.loads(p.read_text())["sets"]
        out[kind][fam].append({k: v.get("solve_rate") for k, v in s.items()})
    for sub, kind in (("distill_memfix_s0", "distill_aug"), ("distill_plain_memfix_s0", "distill_plain")):
        for p in (R / "rev2_private").glob(f"*/*/python_feedback_M_full_r*/{sub}/eval.json"):
            s = json.loads(p.read_text())["sets"]
            out[kind][p.parts[-5]].append({k: v.get("solve_rate") for k, v in s.items()})
    return out


def bmean(b, kind, fam, metric="Q-lvl"):
    xs = [e.get(metric) for e in b.get(kind, {}).get(fam, []) if e.get(metric) is not None]
    return float(np.mean(xs)) if xs else math.nan


def tab_main(S, out, SW=None):
    """SW: summary of the fixed-actor W2 runs of the weaker writers (tag rev2_private_weak,
    analysis_rev2_weak); it fills their W2 and final-level cells. P2 was not run for them."""
    def sel(summ, metric, tag):
        return {(c["model"], c["arm"]): c for c in summ[metric] if c["info"] == "M" and c["variant"] == "full"
                and c["effort"] == "high" and c["tag"] == tag}
    cells, fin = sel(S, "Q-lvl", "rev2_private"), sel(S, "Q-lvl-final", "rev2_private")
    if SW is not None:
        for d, metric in ((cells, "Q-lvl"), (fin, "Q-lvl-final")):
            for k, c in sel(SW, metric, "rev2_private_weak").items():
                assert k[1] == "W2" and k not in d, k
                d[k] = c
    L = [r"\begin{tabular}{lllllll}", r"\toprule",
         r"Writer & W1 & W2 & P1 & P2 & W1 deliv. & W2 final lvl\\", r"\midrule"]
    for m in MODELS:
        c = lambda a: ci(cells.get((m, a)))
        dl = cells.get((m, "W1"), {}).get("delivery_rate")
        L.append(f"{LABEL[m]} & {c('W1')} & {c('W2')} & {c('P1')} & {c('P2')} & {f2(dl)} & "
                 f"{f2(fin.get((m, 'W2'), {}).get('mean'))}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "main_qlvl.tex").write_text("\n".join(L) + "\n")


def _avg_rank(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def h3_trend(w2, p2, n_perm=100000, seed=0):
    """Exploratory H3 trend test, family as unit: Spearman correlation between stratum rank
    (easy=1 ... very hard=4) and the per-family P2 - W2 gap (frontier writers pooled), with a
    one-sided permutation p-value (gaps permuted over families)."""
    fams = [f for f in STRATUM if f in w2 and f in p2]
    rs = _avg_rank([STRATA.index(STRATUM[f]) for f in fams])
    rg = _avg_rank([p2[f] - w2[f] for f in fams])
    rho = float(np.corrcoef(rs, rg)[0, 1])
    perm = np.random.default_rng(seed).permuted(np.tile(rg, (n_perm, 1)), axis=1)
    rs_c, perm_c = rs - rs.mean(), perm - perm.mean(1, keepdims=True)
    null = (perm_c @ rs_c) / (np.linalg.norm(perm_c, axis=1) * np.linalg.norm(rs_c))
    return {"rho": rho, "p_one_sided": float((1 + (null >= rho - 1e-12).sum()) / (n_perm + 1)),
            "p_two_sided": float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (n_perm + 1)),
            "n_families": len(fams), "n_perm": n_perm}


def tab_strata(rows, c3rows, b, out):
    """Stratum means (frontier pooled, family macro). W2 and P2 also get 95% bootstrap CIs over the
    families within each stratum (none for a one-family stratum). Returns the exploratory H3 trend test."""
    best = {f: max([bmean(b, k, f) for k in ("bc_support", "bc_aug", "bc_aug_dihedral", "distill_aug", "distill_plain", "ppo")
                    if not math.isnan(bmean(b, k, f))] or [math.nan]) for f in STRATUM}
    w2 = fam_mean(rows, lambda r: r["arm"] == "W2" and r["model"] in FRONTIER)
    p2 = fam_mean(rows, lambda r: r["arm"] == "P2" and r["model"] in FRONTIER)
    w1 = fam_mean(rows, lambda r: r["arm"] == "W1" and r["model"] in FRONTIER)
    p1 = fam_mean(rows, lambda r: r["arm"] == "P1" and r["model"] in FRONTIER)
    series = [("Written weights, W2", w2, True),
              ("Written weights, W1", w1, False),
              ("Python player, P2", p2, True),
              ("Python player, P1", p1, False),
              ("Reactive Python, P-react", fam_mean(c3rows, lambda r: r["arm"] == "PR1" and r["model"] in FRONTIER), False),
              ("Best learned baseline (non-privileged)", best, False),
              ("BC-oracle (capacity)", {f: bmean(b, "bc_oracle", f) for f in STRATUM}, False)]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         "Policy source & " + " & ".join(f"{s} ({sum(1 for v in STRATUM.values() if v == s)})" for s in STRATA)
         + r" & macro\\", r"\midrule"]
    cis = {}
    for name, d, with_ci in series:
        cells = []
        for s in STRATA:
            v = [x for f, x in d.items() if STRATUM.get(f) == s and x == x]
            m, lo, hi = boot(v)
            if with_ci and len(v) > 1:
                cells.append(f"{m:.2f} {{\\scriptsize[{lo:.2f},{hi:.2f}]}}")
                cis.setdefault(name, {})[s] = [m, lo, hi]
            else:
                cells.append(f2(macro({f: x for f, x in d.items() if STRATUM.get(f) == s and x == x})))
        L.append(f"{name} & " + " & ".join(cells) + f" & {f2(macro({f: v for f, v in d.items() if v == v}))}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "strata.tex").write_text("\n".join(L) + "\n")
    return {"trend": h3_trend(w2, p2), "trend_one_shot": h3_trend(w1, p1), "stratum_ci": cis}


def tab_baselines(rows, b, out):
    w2 = fam_mean(rows, lambda r: r["arm"] == "W2" and r["model"] == "astra-6")
    kinds = [("bc_support", "BC-support"), ("bc_aug", "BC-aug"), ("bc_aug_dihedral", "BC-aug (dihedral only)"), ("distill_aug", "Distill-aug"), ("distill_plain", "Distill (no aug.)"), ("ppo", "PPO"),
             ("ppo_shaped", "PPO-shaped$^\\ast$"), ("bc_oracle", "BC-oracle$^\\ast$")]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & S-start & Q-lvl & Q-tpl & families\\", r"\midrule"]
    for k, name in kinds:
        fams = [f for f in STRATUM if b.get(k, {}).get(f)]
        if not fams:
            continue
        L.append(f"{name} & " + " & ".join(f2(np.nanmean([bmean(b, k, f, m) for f in fams])) for m in ("S-start", "Q-lvl", "Q-tpl"))
                 + f" & {len(fams)}\\\\")
    fams = list(w2)
    L.append(r"\midrule")
    L.append("GPT-6 Astra W2 & " + " & ".join(f2(macro(fam_mean(rows, lambda r: r["arm"] == "W2" and r["model"] == "astra-6", m)))
                                             for m in ("S-start", "Q-lvl", "Q-tpl")) + f" & {len(fams)}\\\\")
    # paired H2: Astra W2 minus best learned (non-privileged) per family
    diffs = []
    for f in fams:
        bl = [bmean(b, k, f) for k in ("bc_support", "bc_aug", "bc_aug_dihedral", "distill_aug", "distill_plain", "ppo") if not math.isnan(bmean(b, k, f))]
        if bl:
            diffs.append(w2[f] - max(bl))
    m, lo, hi = boot(diffs)
    L += [r"\midrule", f"Paired $\\Delta$ (W2 $-$ best non-privileged) & & {f2(m)} {{\\scriptsize[{f2(lo)},{f2(hi)}]}} & & {len(diffs)}\\\\",
          r"\bottomrule", r"\end{tabular}"]
    (out / "baselines.tex").write_text("\n".join(L) + "\n")
    return {"h2_diff": m, "ci": [lo, hi], "n": len(diffs), "w2_better": int(sum(d > 0 for d in diffs))}


def tab_ablations(rows, c3rows, out):
    def cell(rs, **k):
        return macro(fam_mean(rs, lambda r: all(r.get(a) == v for a, v in k.items()), fams=S6), S6)
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Writer & W1 & W1, M+S & no axial & 8 iter. & effort med. & P1 & P1, M+S\\", r"\midrule"]
    for m in ("astra-6", "sol-6.1", "opus-5.5"):
        L.append(f"{LABEL[m]} & " + " & ".join(f2(x) for x in (
            cell(rows, model=m, arm="W1", info="M", variant="full"),
            cell(c3rows, model=m, arm="W1", info="MS"),
            cell(c3rows, model=m, arm="W1", variant="noaxial"),
            cell(c3rows, model=m, arm="W1", variant="t8"),
            cell(c3rows, model=m, arm="W1", effort="medium"),
            cell(rows, model=m, arm="P1", info="M"),
            cell(c3rows, model=m, arm="P1", info="MS"))) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "ablations.tex").write_text("\n".join(L) + "\n")


def tab_controls(rows, out):
    keys = [("Q-lvl", "Q-lvl"), ("blind/Q-lvl", "zero frame"), ("diag/Q-lvl/first_frame", "first frame"),
            ("diag/perturbed/no_reset", "perturbed, no RESET"), ("diag/cf_sensitivity", "CF sensitivity")]
    def get(r, k):
        if k == "Q-lvl":
            return r.get("Q-lvl")
        src, *rest = k.split("/")
        d = r.get(src)
        return None if d is None else d.get("/".join(rest))
    L = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule", "Writer (W2) & " + " & ".join(n for _, n in keys) + r"\\", r"\midrule"]
    for m in FRONTIER:
        vals = []
        for k, _ in keys:
            d = defaultdict(list)
            for r in rows:
                if r.get("evaluated") and r["arm"] == "W2" and r["model"] == m:
                    x = get(r, k)
                    if x is not None and x == x:
                        d[r["family"]].append(x)
            vals.append(macro({f: float(np.mean(v)) for f, v in d.items()}))
        L.append(f"{LABEL[m]} & " + " & ".join(f2(v) for v in vals) + r"\\")
    fl = [json.loads(p.read_text()) for p in (R / "rev2_floors").glob("*.json") if not p.name.startswith("official.")]
    if fl:
        L.append(r"\midrule")
        L.append("Replay floor / best open loop & " + f2(np.mean([x["Q-lvl/replay"] for x in fl])) + " / "
                 + f2(np.mean([x["Q-lvl/best_open_loop"] for x in fl])) + " & & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "controls.tex").write_text("\n".join(L) + "\n")


def tab_public(P, out):
    from wsp.public_scope import assert_public_cohort_complete
    assert_public_cohort_complete()
    cells = {(c["model"], c["arm"]): c for c in P["Q-lvl"]}
    L = [r"\begin{tabular}{lllll}", r"\toprule", r"Writer & W1 & W2 & P1 & P2\\", r"\midrule"]
    for m in MODELS[:7]:
        L.append(f"{LABEL[m]} & " + " & ".join(ci(cells.get((m, a))) for a in ("W1", "W2", "P1", "P2")) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "public.tex").write_text("\n".join(L) + "\n")


def tab_families(out):
    from wsp.envs import new_game
    from wsp.queries import load_sets
    L = [r"\begin{tabular}{llllrrrr}", r"\toprule",
         r"Family & Stratum & Actions & Held-out levels & Levels & S-start & Q-lvl & Q-tpl\\", r"\midrule"]
    for f in STRATUM:
        g = new_game(f, 0)
        acts = ",".join(a.replace("ACTION", "A") for a in g.available_actions if a != "RESET")
        sets = load_sets(f, rev=2)
        held = sorted({s.level + 1 for s in sets["Q-lvl"]})
        L.append(f"\\texttt{{{f}}} & {STRATUM[f]} & {acts} & {','.join(map(str, held))} & {g.n_levels} & "
                 f"{len(sets['S-start'])} & {len(sets['Q-lvl'])} & {len(sets['Q-tpl'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "families.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(R / "analysis_rev2"))
    ap.add_argument("--c3", default=str(R / "analysis_rev2_c3"))
    ap.add_argument("--public", default=str(R / "analysis_rev2_public"))
    ap.add_argument("--weak", default=str(R / "analysis_rev2_weak"),
                    help="analyze_rev2.py --tags rev2_private_weak output (weaker writers' W2)")
    ap.add_argument("--out", default=str(R / "analysis_rev2" / "tables"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((Path(a.analysis) / "rows.json").read_text())
    S = json.loads((Path(a.analysis) / "summary.json").read_text())
    c3 = json.loads((Path(a.c3) / "rows.json").read_text()) if (Path(a.c3) / "rows.json").exists() else []
    b = baselines()
    SW = json.loads((Path(a.weak) / "summary.json").read_text()) if (Path(a.weak) / "summary.json").exists() else None
    tab_main(S, out, SW)
    tab_families(out)
    h3 = tab_strata(rows, c3, b, out)
    h2 = tab_baselines(rows, b, out)
    tab_ablations(rows, c3, out)
    tab_controls(rows, out)
    if (Path(a.public) / "summary.json").exists():
        tab_public(json.loads((Path(a.public) / "summary.json").read_text()), out)
    (out / "numbers.json").write_text(json.dumps({"h2": h2, "h3": h3}, indent=1))
    print("tables ->", out, "H2:", h2, "H3:", h3["trend"])

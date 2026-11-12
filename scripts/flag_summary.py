"""Sandbox flags on delivered network artifacts: counts by type and the effect of
excluding flagged artifacts on every cell (tag, condition, writer). Flags are assigned automatically by
the sandbox's static checks on the initialiser code (wsp.sandbox): while loops, linear-algebra
routines, more than 1,000 numeric literals, more than 30 s CPU. Writes runs/analysis_rev2/flags.json.

A delivered artifact is the final artifact of a fixed-actor, FA or RL-repair session. Cell score: level-
balanced Q-lvl, family macro (analyze_fa_all.score); excluding drops flagged sessions, and families
left without a session.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fa_all import score  # noqa: E402

R = Path(__file__).resolve().parents[2] / "runs"
PREFIXES = ("weights_", "freearch_", "edit_")


def kind(f):
    f = str(f)
    return ("while loop" if "while" in f else "linear algebra" if "linear" in f else
            "numeric literals" if "literal" in f else "CPU time" if "cpu" in f.lower() else f)


if __name__ == "__main__":
    n_art, n_flag, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for sj in sorted(R.glob("rev2*/*/*/*/session.json")):
        tag, fam, w, sess = sj.parts[-5:-1]
        if not sess.startswith(PREFIXES):
            continue
        rec = json.loads(sj.read_text())
        fl = []
        if rec.get("final_round") is not None:
            c = [c for c in rec["candidates"] if str(c["round"]) == str(rec["final_round"])][0]
            fl = (c.get("audit") or {}).get("flags") or []
            n_art[tag] += 1
            if fl:
                n_flag[tag] += 1
                for k in {kind(f) for f in fl}:
                    kinds[k] += 1
        if not sess.startswith("edit_") and (s := score(sj.parent)) is not None:
            cells[(tag, sess.rsplit("_r", 1)[0], w)][fam].append((s, bool(fl)))
    changes = []
    for (tag, cond, w), fams in cells.items():
        nfl = sum(f for v in fams.values() for _, f in v)
        if not nfl:
            continue
        a = float(np.mean([np.mean([s for s, _ in v]) for v in fams.values()]))
        kept = {f: [s for s, fl in v if not fl] for f, v in fams.items()}
        kept = {f: v for f, v in kept.items() if v}
        b = float(np.mean([np.mean(v) for v in kept.values()]))
        changes.append({"tag": tag, "condition": cond, "writer": w, "flagged": nfl, "families": len(fams),
                        "families_kept": len(kept), "all": a, "without_flagged": b, "diff": b - a})
    changes.sort(key=lambda r: -abs(r["diff"]))
    res = {"artifacts": sum(n_art.values()), "flagged": sum(n_flag.values()), "by_kind": dict(kinds),
           "by_tag": {t: {"artifacts": n_art[t], "flagged": n_flag[t]} for t in sorted(n_art)},
           "max_abs_cell_change": abs(changes[0]["diff"]) if changes else 0.0, "cells": changes}
    (R / "analysis_rev2" / "flags.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
    for r in changes[:5]:
        print(r)

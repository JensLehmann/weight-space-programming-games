"""Matched random update for robust weights: is PPO's first update on robust actors more destructive than a random
update of the same per-tensor size and support? (perturbation_rev2.matched_update, the paper's control.)

For every run in runs/rev2_stage2_robust_upd/<fam>/<kind>-<w>_std_sparse_s0 with init.npz and student.npz:
evaluate on the fixed Q-lvl subsample (perturbation_rev2.eval_sets) the initialisation, the PPO student after
one update, and matched random updates (seeds 0, 1). Exact FP32, CPU.
Usage: matched_robust.py [--workers N]   Output: runs/analysis_rev2/matched_robust.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import perturbation_rev2 as P  # noqa: E402

R = P.R
U = R / "rev2_stage2_robust_upd"


def job(args):
    d, which, seed = args
    P.setup_torch()
    from wsp import actor as A
    fam = Path(d).parent.name
    p0 = A.load_npz(Path(d) / "init.npz")
    p1 = {k: v for k, v in np.load(Path(d) / "student.npz").items()}
    if which == "init":
        p = p0
    elif which == "ppo":
        p = p1
    else:
        p = P.matched_update(p0, p1, seed, f"{fam}/{Path(d).name}")
    sets = P.eval_sets(fam)
    res, nonfinite = P.run_eval(p, {"Q-lvl": sets["Q-lvl"]}, "cpu")
    return str(Path(d).relative_to(R)), which, seed, P.qlvl_balanced(fam, res["Q-lvl"]), nonfinite


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dirs = [str(d) for d in sorted(U.glob("*/*")) if (d / "student.npz").exists()]
    jobs = [(d, w, s) for d in dirs for (w, s) in (("init", 0), ("ppo", 0), ("matched", 0), ("matched", 1))]
    import multiprocessing as mp
    out = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            print(json.dumps(r), flush=True)
    (R / "analysis_rev2").mkdir(exist_ok=True)
    (R / "analysis_rev2" / "matched_robust.json").write_text(json.dumps(out, indent=1))

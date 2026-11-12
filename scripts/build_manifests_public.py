"""Build revision-2 public-tier manifests (manifests/rev2/queries_official.<id>.json).

Usage: build_manifests_public.py <id|official.id|all>... [--jobs 8] [--wall-per-start 60]
One process per game (spawned); prints one JSON summary line per game.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# rough cost order from the public survey (most expensive first) so long games start early
COST_ORDER = ("sb26 lf52 bp35 sc25 sk48 wa30 s5i5 tn36 ar25 su15 dc22 ls20 ka59 vc33 r11l lp85 "
              "re86 g50t sp80 cd82 m0r0 cn04 tu93 ft09 tr87").split()


def build(gid: str, wall_per_start: float) -> dict:
    from wsp.queries_public import write_public_sets
    t = time.time()
    log_lines = []
    try:
        d = write_public_sets(gid, log=lambda m: (log_lines.append(m), print(m, file=sys.stderr, flush=True)),
                              wall_per_start_s=wall_per_start)
    except Exception:
        return {"family": f"official.{gid}", "error": traceback.format_exc()[-1500:]}
    return {"family": d["family"], "n_levels": d["n_levels"], "held": [h + 1 for h in d["held_out_levels"]],
            "sizes": {k: len(v) for k, v in d["sets"].items()},
            "q_lvl_per_level": d["q_lvl_per_level"], "excluded_from_macro": d["excluded_from_macro"],
            "exclusion_reason": d["exclusion_reason"], "problems": d["construction_problems"],
            "s": round(time.time() - t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--wall-per-start", type=float, default=60.0)
    a = ap.parse_args()
    from wsp.public import public_ids
    ids = list(public_ids()) if a.ids == ["all"] else [i.removeprefix("official.") for i in a.ids]
    unknown = sorted(set(ids) - set(public_ids()))
    if unknown:
        raise SystemExit(f"unknown public games: {unknown}")
    ids.sort(key=lambda g: COST_ORDER.index(g) if g in COST_ORDER else -1)
    if a.jobs <= 1:
        for g in ids:
            print(json.dumps(build(g, a.wall_per_start)), flush=True)
        return
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(build, g, a.wall_per_start): g for g in ids}
        for f in as_completed(futs):
            print(json.dumps(f.result()), flush=True)


if __name__ == "__main__":
    main()

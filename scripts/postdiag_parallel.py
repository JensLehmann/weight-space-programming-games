"""Parallel blind + diag diagnostics (same computations as blind_eval.py and diag_eval.py artifacts).

Shards session/student directories that have eval.json but lack blind.json or diag.json across
worker processes, one GPU per worker. blind.json: S-start and Q-lvl with an all-zero frame (weights
or freearch only). diag.json: diag_eval.artifacts() applied to the single directory.

Usage: postdiag_parallel.py <root> [<root> ...] [--workers N] [--gpus N]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def targets(roots):
    out = []
    for root in roots:
        for p in sorted(set(Path(root).rglob("session.json")) | set(Path(root).rglob("student.npz"))):
            d = p.parent
            if not (d / "eval.json").exists():
                continue
            if not (d / "blind.json").exists() or not (d / "diag.json").exists():
                out.append(str(d))
    return sorted(set(out))


def blind_one(d):
    import numpy as np
    from eval_artifacts import family_of, rev_of
    from blind_eval import Blind
    from wsp import actor as A
    from wsp.envs import run_episodes, summarize
    from wsp.queries import load_sets
    import torch
    d = Path(d)
    rec = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else None
    fa = rec is not None and rec["condition"] == "freearch"
    if rec is not None:
        if rec["final_round"] is None or rec["condition"] not in ("weights", "edit", "freearch"):
            return
        if fa:
            from wsp import freearch
            arch, params = freearch.load_net(d, rec["final_round"])
        else:
            params = A.load_npz(d / f"round{rec['final_round']}.npz")
        variant = rec.get("variant", "full")
    else:
        params, variant = A.load_npz(d / "student.npz"), "full"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qs = load_sets(family_of(d), rev=rev_of(d, rec))
    out = {}
    for s in ("S-start", "Q-lvl"):
        if fa:
            pol = Blind(freearch.FreeArchPolicy(freearch.FreeArchNet(arch, params), dev))
        else:
            pol = Blind(A.NeuralPolicy(A.Actor(params, variant=variant), dev))
        out[s] = summarize(run_episodes(qs[s], pol, batch=64, keep_trace=False))["solve_rate"]
    (d / "blind.json").write_text(json.dumps(out))


def work(args):
    i, n, gpu, dirs = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    # exact FP32 semantics (TF32 convolutions flip argmax decisions of high-gain written actors)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    import traceback
    import diag_eval
    done = 0
    for d in dirs[i::n]:
        try:
            if not (Path(d) / "blind.json").exists():
                blind_one(d)
            if not (Path(d) / "diag.json").exists():
                diag_eval.artifacts(d)
            done += 1
        except Exception:
            print("ERROR", d, traceback.format_exc()[-600:], flush=True)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args()
    dirs = targets(a.roots)
    print(f"{len(dirs)} directories need blind/diag", flush=True)
    if not dirs:
        sys.exit(0)
    n = min(a.workers, len(dirs))
    with mp.get_context("spawn").Pool(n) as pool:
        done = pool.map(work, [(i, n, i % a.gpus, dirs) for i in range(n)])
    print("done", sum(done), flush=True)

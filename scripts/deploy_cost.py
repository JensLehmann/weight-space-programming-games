"""Deployment cost per action: MACs and parameter memory of the fixed actor per variant,
measured torch latency (CPU batch 1; GPU batch 1 and 64 if available), and measured per-action wall
time of Python players (sandboxed, as evaluated) from a sample of artifacts.

Usage: deploy_cost.py [--py-root runs/rev2_private] [--n-py 20] [--out runs/rev2_deploy_cost.json]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of, rev_of  # noqa: E402

from wsp import actor as A
from wsp.envs import ACTIONS, run_episodes
from wsp.queries import load_sets
from wsp.sandbox import PythonPlayerPolicy
from wsp.session import load_final


def macs(variant):
    H = W = 64
    s = A.SCHEMA
    v = A.VARIANTS[variant]
    m = H * W * np.prod(s["W1"])                      # 5x5 conv
    m += H * W * np.prod(s["W2"])                     # 3x3 conv
    m += H * W * s["W3"][0] * (96 if v["axial"] else 32)
    m += H * W * np.prod(s["Uh"])
    m += v["T"] * H * W * np.prod(s["Wh"])            # tied recurrent block
    m += np.prod(s["Wz"]) + np.prod(s["Wm"]) + np.prod(s["Wmz"]) + np.prod(s["Wma"]) + np.prod(s["Wo"])
    m += H * W * np.prod(s["Wc"])
    return int(m)


def torch_latency(variant, device, batch, reps=20):
    actor = A.Actor(A.random_params(0), variant=variant).to(device).eval()
    obs = torch.randint(0, 16, (batch, 64, 64), device=device)
    mem = torch.zeros(batch, A.MEM, device=device)
    ap = torch.zeros(batch, len(ACTIONS), device=device)
    av = torch.ones(batch, len(ACTIONS), dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(3):
            actor(obs, mem, ap, av)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            actor(obs, mem, ap, av)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def python_cost(root, n):
    dirs = []
    for p in Path(root).rglob("session.json"):
        rec = json.loads(p.read_text())
        if rec.get("condition") in ("python", "pyreact") and rec.get("final_round") is not None:
            dirs.append((p.parent, rec))
    random.Random(0).shuffle(dirs)
    rows = []
    for d, rec in dirs[:n]:
        fam = family_of(d)
        art = load_final(d, rec["condition"])
        specs = load_sets(fam, rev=rev_of(d, rec))["S-start"]
        pol = PythonPlayerPolicy(art.code)
        try:
            t = time.perf_counter()
            res = run_episodes(specs, pol, batch=16, keep_trace=False)
            wall = time.perf_counter() - t
        finally:
            if hasattr(pol, "close"):
                pol.close()
        steps = sum(r.actions for r in res)
        rows.append({"dir": str(d), "family": fam, "code_chars": len(art.code), "steps": steps,
                     "s_per_action": wall / max(1, steps)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-root", default=None)
    ap.add_argument("--n-py", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = {"n_params": A.N_PARAMS, "param_bytes_f32": A.N_PARAMS * 4, "variants": {}}
    for v in A.VARIANTS:
        r = {"macs_per_action": macs(v), "cpu_b1_s": torch_latency(v, "cpu", 1, reps=5)}
        if torch.cuda.is_available():
            r["gpu_b1_s"] = torch_latency(v, "cuda", 1)
            r["gpu_b64_s_per_action"] = torch_latency(v, "cuda", 64) / 64
        out["variants"][v] = r
        print(v, r, flush=True)
    if a.py_root:
        rows = python_cost(a.py_root, a.n_py)
        spa = [r["s_per_action"] for r in rows]
        out["python"] = {"n": len(rows), "median_s_per_action": float(np.median(spa)) if spa else None,
                         "p90_s_per_action": float(np.percentile(spa, 90)) if spa else None,
                         "median_code_chars": float(np.median([r["code_chars"] for r in rows])) if rows else None,
                         "rows": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))

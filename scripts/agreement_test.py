"""Numerical agreement of the batched torch actor with the float32 NumPy reference on real artifacts.

For each sampled weight artifact (writer final round), plays S-start episodes with the torch policy
(CPU, float32) and, at every step, recomputes the reference step from the same observation, memory
and previous action. Reports decision disagreements and the max relative logit error.

Usage: agreement_test.py <root> [--n 30] [--steps 40] [--out runs/rev2_agreement.json]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of, rev_of  # noqa: E402

from wsp import actor as A
from wsp.envs import ACTIONS, avail_mask, build_start, obs_of
from wsp.queries import load_sets


def check(params, variant, specs, steps, device="cpu"):
    actor = A.Actor(params, variant=variant).to(device).eval()
    pol = A.NeuralPolicy(actor, device)
    n = dis = 0
    max_rel = 0.0
    for sp in specs:
        g = build_start(sp)
        pol.begin([0])
        mem_ref = np.zeros(A.MEM, np.float32)
        prev = None
        for _ in range(steps):
            o, av = obs_of(g), avail_mask(g)
            a_prev = np.zeros(len(ACTIONS), np.float32)
            if prev is not None:
                a_prev[prev] = 1.0
            ref_move, ref_logits, mem_ref = A.reference_step(params, o, mem_ref, a_prev, av, variant=variant)
            with torch.no_grad():
                lt, ck, _ = actor(torch.as_tensor(o[None], device=device), pol.mem[0][None], torch.as_tensor(a_prev[None], device=device),
                                  torch.as_tensor(av[None], device=device))
            move = pol.act([0], o[None], av[None])[0]
            n += 1
            fin = np.isfinite(ref_logits)
            if fin.any():
                t = lt[0].cpu().numpy()
                rel = np.abs(t[fin] - ref_logits[fin]).max() / max(1e-6, np.abs(ref_logits[fin]).max())
                max_rel = max(max_rel, float(rel))
            if tuple(move) != tuple(ref_move):
                dis += 1
            if move[0] not in ACTIONS:
                break
            prev = ACTIONS.index(move[0])
            g.step(*move)
            if g.state != "NOT_FINISHED" or g.level != sp.level:
                break
    return n, dis, max_rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-tf32", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    print("device", a.device, "cudnn.allow_tf32", torch.backends.cudnn.allow_tf32, flush=True)
    dirs = []
    for p in Path(a.root).rglob("session.json"):
        rec = json.loads(p.read_text())
        if rec.get("condition") == "weights" and rec.get("final_round") is not None:
            dirs.append((p.parent, rec))
    random.Random(0).shuffle(dirs)
    rows = []
    for d, rec in dirs[:a.n]:
        fam = family_of(d)
        params = A.load_npz(d / f"round{rec['final_round']}.npz")
        specs = load_sets(fam, rev=rev_of(d, rec))["S-start"][:3]
        n, dis, rel = check(params, rec.get("variant", "full"), specs, a.steps, a.device)
        rows.append({"dir": str(d), "steps": n, "disagree": dis, "max_rel_logit_err": rel})
        print(json.dumps(rows[-1]), flush=True)
    tot = sum(r["steps"] for r in rows)
    summ = {"artifacts": len(rows), "steps": tot, "disagreements": sum(r["disagree"] for r in rows),
            "max_rel_logit_err": max((r["max_rel_logit_err"] for r in rows), default=None), "rows": rows}
    print({k: v for k, v in summ.items() if k != "rows"})
    if a.out:
        Path(a.out).write_text(json.dumps(summ, indent=1))

"""Agreement of the batched FA executor (exact FP32) with the NumPy reference on real FA artifacts.
usage: agreement_fa.py <root> [--n 24] [--steps 30] [--device cuda] [--out file]"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of  # noqa: E402
from wsp import freearch as FA  # noqa: E402
from wsp.envs import ACTIONS, avail_mask, build_start, obs_of  # noqa: E402
from wsp.queries import load_sets  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    FA.exact_fp32()
    dirs = [p.parent for p in Path(a.root).rglob("session.json")
            if json.loads(p.read_text()).get("final_round") is not None]
    random.Random(0).shuffle(dirs)
    rows = []
    for d in dirs[:a.n]:
        rec = json.loads((d / "session.json").read_text())
        arch, params = FA.load_net(d, rec["final_round"])
        net = FA.FreeArchNet(arch, params).to(a.device).eval()
        pol = FA.FreeArchPolicy(net, a.device)
        n = dis = 0
        rel = 0.0
        for sp in load_sets(family_of(d), rev=2)["S-start"][:3]:
            g = build_start(sp)
            pol.begin([0])
            mem = np.zeros(net.M, np.float32)
            prev = None
            for _ in range(a.steps):
                o, av = obs_of(g), avail_mask(g)
                ap_ = np.zeros(len(ACTIONS), np.float32)
                if prev is not None:
                    ap_[prev] = 1.0
                ref_move, ref_logits, mem = FA.reference_step(arch, params, o, mem, ap_, av)
                with torch.no_grad():
                    lt, _, _ = net(torch.as_tensor(o[None], device=a.device), pol.mem[0][None],
                                   torch.as_tensor(ap_[None], device=a.device), torch.as_tensor(av[None], device=a.device))
                move = pol.act([0], o[None], av[None])[0]
                n += 1
                fin = np.isfinite(ref_logits)
                if fin.any():
                    t = lt[0].cpu().numpy()
                    rel = max(rel, float(np.abs(t[fin] - ref_logits[fin]).max() / max(1e-6, np.abs(ref_logits[fin]).max())))
                dis += tuple(move) != tuple(ref_move)
                if move[0] not in ACTIONS:
                    break
                prev = ACTIONS.index(move[0])
                g.step(*move)
                if g.state != "NOT_FINISHED" or g.level != sp.level:
                    break
        rows.append({"dir": str(d), "steps": n, "disagree": dis, "max_rel": rel})
        print(json.dumps(rows[-1]), flush=True)
    summ = {"artifacts": len(rows), "steps": sum(r["steps"] for r in rows), "disagreements": sum(r["disagree"] for r in rows),
            "max_rel": max(r["max_rel"] for r in rows)}
    print(summ)
    if a.out:
        Path(a.out).write_text(json.dumps({**summ, "rows": rows}, indent=1))

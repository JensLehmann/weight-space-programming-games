"""Stage 2: weight-space programming x RL (one process per run).

kinds
  wsp_rl:<session_dir>[:cal]   PPO from a written artifact (cal = logit-temperature calibration)
  matched_rl:<session_dir>     PPO from random weights matched to the artifact's per-tensor
                               nonzero count and L2 norm (control for scale/sparsity)
  distill_rl:<student_npz>     PPO from a distilled writer-player student
  rl_cont:<student_npz>        continue PPO from an RL-trained actor for extra steps
  theta0_rl                    PPO from theta_0 (RL(theta_0) control with the same PPO options)

PPO options (all off by default) are shared with scripts/baselines.py:
  --shaping planner (planner-potential reward), --bptt 16 (recurrent PPO), --grad-ckpt 16,
  --masked --kl-to-init C --critic-warmup 50000 (structure-preserving), --optimizer sgd --lr 1e-5,
  --eval-sets S-start,Q-lvl [--eval-at ... --eval-every ...] (greedy checkpoints, plus greedy and
  sampled evaluation of the frozen initialisation at step 0).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import add_ppo_args, ppo_kwargs  # noqa: E402  (stdlib-only module scope)

import numpy as np


def calibrate(params, specs, target=5.0):
    """Rescale only the output head so the max |logit| on support start states is `target`."""
    import torch
    from wsp import actor as A
    from wsp.envs import avail_mask, build_start, obs_of
    net = A.Actor(params).eval()
    mx = 0.0
    with torch.no_grad():
        for s in specs[:32]:
            g = build_start(s)
            lg, _, _ = net(torch.as_tensor(obs_of(g)[None]), torch.zeros(1, A.MEM),
                           torch.zeros(1, 8), torch.as_tensor(avail_mask(g)[None]))
            v = lg[torch.isfinite(lg)].abs()
            if v.numel():
                mx = max(mx, float(v.max()))
    c = max(1.0, mx / target)
    p = {k: v.copy() for k, v in params.items()}
    p["Wo"] = p["Wo"] / c
    p["bo"] = p["bo"] / c
    p["Wc"] = p["Wc"] / max(1.0, float(np.abs(p["Wc"]).max()) / target) if np.abs(p["Wc"]).max() > 0 else p["Wc"]
    return p, c


def matched_random(params, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in params.items():
        n = int(np.count_nonzero(v))
        w = np.zeros(v.size, np.float32)
        if n:
            idx = rng.choice(v.size, n, replace=False)
            r = rng.standard_normal(n).astype(np.float32)
            r *= np.linalg.norm(v) / (np.linalg.norm(r) + 1e-12)
            w[idx] = r
        out[k] = w.reshape(v.shape)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("family"); ap.add_argument("seed", type=int)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-rev", type=int, default=None, help="query manifest revision (default: current)")
    ap.add_argument("--device", default="cuda")
    add_ppo_args(ap)                                  # includes --lr (default 3e-4)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if os.environ.get("WSP_NO_TF32"):              # exact-FP32 training
        import torch
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    from wsp import actor as A
    from wsp.queries import load_sets
    from wsp.train_bc import init_params
    from wsp.train_ppo import ppo
    sets = load_sets(a.family) if a.split_rev is None else load_sets(a.family, rev=a.split_rev)
    support = sets["S-start"] + sets["S-dev"]
    parts = a.kind.split(":")
    kind = parts[0]
    src = Path(parts[1]) if len(parts) > 1 else None
    kw = ppo_kwargs(a, sets)
    meta = {"kind": a.kind, "family": a.family, "seed": a.seed, "steps": a.steps, "lr": a.lr,
            "ppo_options": {k: v for k, v in kw.items() if k != "eval_specs"},
            "eval_sets": [k for k in a.eval_sets.split(",") if k]}
    if kind == "theta0_rl":
        init = init_params(a.seed)
    elif kind in ("wsp_rl", "matched_rl"):
        rec = json.loads((src / "session.json").read_text())
        init = A.load_npz(src / f"round{rec['final_round']}.npz")
        if kind == "matched_rl":
            init = matched_random(init, a.seed)
        elif len(parts) > 2 and parts[2] == "cal":
            init, c = calibrate(init, support)
            meta["calibration_divisor"] = c
    else:
        init = A.load_npz(src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta))
    A.save_npz(init, out / "init.npz")
    ppo(support, out, seed=a.seed, total_steps=a.steps, device=a.device, init=init, **kw)
    print("DONE", out, flush=True)

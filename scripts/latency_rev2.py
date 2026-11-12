"""Deployment cost of written agents: size and warmed-up latency.

For each artifact (free-architecture FA actors, fixed-actor written weights, Python / P-react players):
  size        params, MACs/step, node evaluations/step (FA: from the artifact's own count)
  latency     exact FP32 on one GPU, after warm-up, on a real observation sequence:
                gpu_b1_eager_ms     batch 1, eager PyTorch
                gpu_b1_graph_ms     batch 1, CUDA-graph captured forward (removes interpreter overhead);
                                    falls back to eager if capture fails (recorded)
                gpu_b64_us_per_act  batch 64 throughput, microseconds per action
                cpu_b1_ms           one CPU thread, batch 1, eager
  python      act() wall time per step inside the sandbox as evaluated (includes IPC)
Usage (on an otherwise idle GPU): latency_rev2.py --fa runs/rev2_fa runs/rev2_fa_public --fixed runs/rev2_private
              --python runs/rev2_private runs/rev2_c3 --n-fixed 40 --n-python 40 --out runs/latency_rev2.json
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of, rev_of  # noqa: E402
from wsp import actor as A  # noqa: E402
from wsp import freearch as FA  # noqa: E402
from wsp.envs import ACTIONS, avail_mask, build_start, obs_of, run_episodes  # noqa: E402
from wsp.queries import load_sets  # noqa: E402

WARM, TIMED = 20, 200


def obs_sequence(fam, rec, n=TIMED):
    """Real frames: the first frames of the support starts, cycled to length n."""
    frames = []
    for sp in load_sets(fam, rev=rev_of(Path("."), rec) if rec else 2)["S-start"][:8]:
        g = build_start(sp)
        frames.append((obs_of(g), avail_mask(g)))
    return [frames[i % len(frames)] for i in range(n)]


def time_net(make_net, mem_size, seq, device, batch=1, graph=False):
    net = make_net().to(device).eval()
    obs = torch.as_tensor(np.stack([o for o, _ in seq[:batch]] * (batch // min(batch, len(seq)) + 1))[:batch], device=device)
    av = torch.as_tensor(np.stack([a for _, a in seq[:batch]] * (batch // min(batch, len(seq)) + 1))[:batch], device=device)
    mem = torch.zeros(batch, mem_size, device=device)
    ap = torch.zeros(batch, len(ACTIONS), device=device)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    note = "eager"
    with torch.no_grad():
        fwd = lambda: net(obs, mem, ap, av)
        if graph and device == "cuda":
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        fwd()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fwd()
                fwd = g.replay
                note = "cuda_graph"
            except Exception as exc:          # dynamic / host-side ops in the executor
                note = f"eager (graph capture failed: {type(exc).__name__})"
        for _ in range(WARM):
            fwd()
        sync()
        ts = []
        for i in range(TIMED):
            t = time.perf_counter()
            fwd()
            sync()
            ts.append(time.perf_counter() - t)
    return statistics.median(ts), note


def measure(make_net, mem_size, seq):
    out = {}
    if torch.cuda.is_available():
        t, _ = time_net(make_net, mem_size, seq, "cuda", 1, graph=False)
        out["gpu_b1_eager_ms"] = t * 1e3
        t, note = time_net(make_net, mem_size, seq, "cuda", 1, graph=True)
        out["gpu_b1_graph_ms"] = t * 1e3
        out["graph_note"] = note
        t, _ = time_net(make_net, mem_size, seq, "cuda", 64, graph=False)
        out["gpu_b64_us_per_act"] = t / 64 * 1e6
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    t, _ = time_net(make_net, mem_size, seq[:40], "cpu", 1, graph=False)
    out["cpu_b1_ms"] = t * 1e3
    torch.set_num_threads(threads)
    return out


def sessions(roots, cond):
    out = []
    for root in roots:
        for sj in Path(root).rglob("session.json"):
            rec = json.loads(sj.read_text())
            if rec.get("condition") in cond and rec.get("final_round") is not None:
                out.append((sj.parent, rec))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", nargs="*", default=[])
    ap.add_argument("--fixed", nargs="*", default=[])
    ap.add_argument("--python", nargs="*", default=[])
    ap.add_argument("--n-fa", type=int, default=10 ** 6)
    ap.add_argument("--n-fixed", type=int, default=40)
    ap.add_argument("--n-python", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FA.exact_fp32()
    rows = []
    rng = random.Random(0)
    fa_sessions = sessions(a.fa, ("freearch",))
    rng.shuffle(fa_sessions)                       # random sample of FA actors (stated in the paper)
    for d, rec in fa_sessions[:a.n_fa]:
        arch, params = FA.load_net(d, rec["final_round"])
        info = FA.count(arch, params)
        seq = obs_sequence(family_of(d), rec)
        r = {"kind": "freearch", "dir": str(d), "params": info.get("params"), "macs": info.get("macs_per_step"),
             "node_evals": info.get("node_evals_per_step"), "memory": arch.get("memory_size")}
        r.update(measure(lambda: FA.FreeArchNet(arch, params), arch["memory_size"], seq))
        rows.append(r)
        print(json.dumps(r), flush=True)
    fixed = sessions(a.fixed, ("weights",))
    rng.shuffle(fixed)
    # the fixed actor written in the FA language, counted with the FA rule (as for FA actors):
    # 1,331,914,880 MACs and 412 node evaluations per step (1,322,870,784 MACs counting only
    # convolutions and dense layers, as in deploy_cost.py)
    fixed_info = FA.count(FA.fixed_actor_arch(), FA.fixed_actor_params(A.zero_params()))
    for d, rec in fixed[:a.n_fixed]:
        params = A.load_npz(d / f"round{rec['final_round']}.npz")
        seq = obs_sequence(family_of(d), rec)
        r = {"kind": "fixed", "dir": str(d), "params": A.N_PARAMS, "macs": fixed_info["macs_per_step"],
             "node_evals": fixed_info["node_evals_per_step"], "memory": A.MEM}
        r.update(measure(lambda: A.Actor(params, variant=rec.get("variant", "full")), A.MEM, seq))
        rows.append(r)
        print(json.dumps(r), flush=True)
    py = sessions(a.python, ("python", "pyreact"))
    rng.shuffle(py)
    from wsp.sandbox import PythonPlayerPolicy
    from wsp.session import load_final
    for d, rec in py[:a.n_python]:
        art = load_final(d, rec["condition"])
        specs = load_sets(family_of(d), rev=rev_of(d, rec))["S-start"][:6]
        class Timed:
            """Times only act() calls (one sandboxed process per episode is started in begin())."""
            def __init__(self, inner):
                self.inner, self.ts = inner, []
            def begin(self, slots):
                self.inner.begin(slots)
            def act(self, slots, obs, avail):
                t = time.perf_counter(); m = self.inner.act(slots, obs, avail); self.ts.append(time.perf_counter() - t)
                return m
        pol = Timed(PythonPlayerPolicy(art.code))
        try:
            run_episodes(specs, pol, batch=1, keep_trace=False)
        finally:
            if hasattr(pol.inner, "close"):
                pol.inner.close()
        ts = pol.ts[3:] or pol.ts                      # drop the first calls (imports, warm-up)
        r = {"kind": rec["condition"], "dir": str(d), "code_chars": len(art.code),
             "sandbox_ms_per_act": float(np.median(ts)) * 1e3, "sandbox_p90_ms": float(np.percentile(ts, 90)) * 1e3}
        rows.append(r)
        print(json.dumps(r), flush=True)
    summ = {}
    for kind in ("freearch", "fixed", "python", "pyreact"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        keys = [k for k in ("params", "macs", "node_evals", "gpu_b1_eager_ms", "gpu_b1_graph_ms", "gpu_b64_us_per_act",
                            "cpu_b1_ms", "sandbox_ms_per_act") if any(k in r and r[k] is not None for r in rs)]
        summ[kind] = {"n": len(rs), **{k: {"median": float(np.median([r[k] for r in rs if r.get(k) is not None])),
                                           "p90": float(np.percentile([r[k] for r in rs if r.get(k) is not None], 90)),
                                           "max": float(np.max([r[k] for r in rs if r.get(k) is not None]))} for k in keys}}
        if kind in ("freearch", "fixed"):
            summ[kind]["graph_captured"] = sum(r.get("graph_note") == "cuda_graph" for r in rs)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))

"""Observation-blindness diagnostic for weight (and free-architecture) artifacts.

Re-runs Q-lvl and S-start with the actor's observation replaced by a constant all-zero frame
(the environment still advances normally). Success that survives is open-loop behaviour.
Writes blind.json next to eval.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from wsp import actor as A
from wsp.envs import run_episodes, summarize
from wsp.queries import load_sets


class Blind:
    def __init__(self, inner):
        self.inner = inner

    def begin(self, slots):
        self.inner.begin(slots)

    def act(self, slots, obs, avail):
        return self.inner.act(slots, np.zeros_like(obs), avail)


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from eval_artifacts import family_of, rev_of  # noqa: E402


if __name__ == "__main__":
    import torch
    root = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [p.parent for p in root.rglob("session.json")
               if "/weights_" in str(p) or "/edit_" in str(p) or "/freearch_" in str(p)]
    targets += [p.parent for p in root.rglob("student.npz")]
    for d in sorted(set(targets)):
        if (d / "blind.json").exists():
            continue
        rec = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else None
        fa = rec is not None and rec["condition"] == "freearch"
        if rec is not None:
            if rec["final_round"] is None:
                continue
            if fa:
                from wsp import freearch
                arch, params = freearch.load_net(d, rec["final_round"])
            else:
                params = A.load_npz(d / f"round{rec['final_round']}.npz")
            variant = rec.get("variant", "full")
        else:
            params = A.load_npz(d / "student.npz")
            variant = "full"
        fam = family_of(d)
        qs = load_sets(fam, rev=rev_of(d, rec))
        out = {}
        for s in ("S-start", "Q-lvl"):
            if fa:
                pol = Blind(freearch.FreeArchPolicy(freearch.FreeArchNet(arch, params), dev))
            else:
                pol = Blind(A.NeuralPolicy(A.Actor(params, variant=variant), dev))
            out[s] = summarize(run_episodes(qs[s], pol, batch=64, keep_trace=False))["solve_rate"]
        (d / "blind.json").write_text(json.dumps(out))
        print(d, out, flush=True)

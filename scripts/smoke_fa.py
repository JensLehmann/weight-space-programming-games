"""Offline smoke test of the free-architecture (FA) pipeline. No LLM/API call: wsp.writer.call is replaced by
canned responses containing hand-written build() programs.

A feedback session (3 rounds) on one family: round 0 = the fixed actor graph with zero weights, round 1 = an
over-budget network (invalid artifact), round 2 = a small looped network that always prefers ACTION1. Then
eval_artifacts.py, postdiag_parallel.py (blind + diag) and blind_eval.py run on the session directory.

Usage: smoke_fa.py [--family ztl] [--out <runs>/fa_smoke]
"""
import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from wsp import freearch as FA
from wsp import writer
from wsp.manifest import RUNS
from wsp.prompts_fa import build_prompt_fa
import wsp.session as S

HERE = Path(__file__).resolve().parent

FIXED_ZERO = "import numpy as np\n\n" + inspect.getsource(FA.fixed_actor_arch) + f'''

SHAPES = {FA.fixed_actor_shapes()!r}


def build():
    arch = fixed_actor_arch()
    return {{"graph": arch["graph"], "memory_size": arch["memory_size"],
            "params": {{k: np.zeros(s) for k, s in SHAPES.items()}}}}
'''

OVER_BUDGET = FIXED_ZERO.replace('def fixed_actor_arch(T=128):', 'def fixed_actor_arch(T=200):')

SMALL = '''
import numpy as np


def build():
    rng = np.random.default_rng(0)
    g = [
        {"id": "e", "op": "conv", "in": ["x0"], "w": "We", "b": "be"},          # 1x1 colour embedding -> 4
        {"id": "L", "op": "loop", "T": 8, "state": [{"id": "h", "init": "e"}, {"id": "c", "zeros": [4]}],
         "body": [{"id": "hc", "op": "conv", "in": ["h"], "w": "Wh"},           # 3x3 propagation
                  {"id": "hn", "op": "tanh", "in": ["hc"]},
                  {"id": "hg", "op": "gmean", "in": ["hn"]},
                  {"id": "cn", "op": "vadd", "in": ["c", "hg"]}],
         "next": {"h": "hn", "c": "cn"}},
        {"id": "s", "op": "sub", "in": ["L.h", "e"]},
        {"id": "r", "op": "rowmax", "in": ["s"]},
        {"id": "g", "op": "vconcat", "in": ["L.c", "mem", "aprev"]},
        {"id": "z", "op": "dense", "in": ["g"], "w": "Wz", "b": "bz"},
        {"id": "logits", "op": "vadd", "in": ["z", "avail"]},
        {"id": "mem_next", "op": "dense", "in": ["L.c"], "w": "Wm"},
        {"id": "click", "op": "conv", "in": ["r"], "w": "Wc", "b": "bc"},
    ]
    bz = np.zeros(8)
    bz[1] = 5.0                                                                  # prefer ACTION1
    p = {"We": rng.normal(0, 0.5, (4, 16, 1, 1)), "be": np.zeros(4), "Wh": rng.normal(0, 0.2, (4, 4, 3, 3)),
         "Wz": rng.normal(0, 0.1, (8, 4 + 2 + 8)), "bz": bz, "Wm": np.zeros((2, 4)),
         "Wc": rng.normal(0, 1, (1, 4, 1, 1)), "bc": np.zeros(1)}
    return {"graph": g, "memory_size": 2, "params": p}
'''


def fake_writer(programs):
    def call(model, messages, tag, out_dir, effort=None, task_budget=None):
        k = int(tag.replace("round", ""))
        text = f"Smoke-test program {k} (hand-written).\n\n```python\n{programs[k]}```\n"
        (Path(out_dir) / f"{tag}.call.json").write_text(json.dumps({"fake": True, "messages": len(messages)}))
        return writer.CallResult(f"fake-{k}", model, "completed", text=text, usage={"input": 0, "output": 0})
    return call


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0, cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="ztl")
    ap.add_argument("--out", default=str(RUNS / "fa_smoke"))
    a = ap.parse_args()
    root = Path(a.out)
    sdir = root / a.family / "smoke-fake" / "freearch_feedback_M_r0"
    if sdir.exists():
        shutil.rmtree(sdir)
    sys.path.insert(0, str(HERE))
    from run_freearch import prompt_for
    prompt, sets = prompt_for(a.family)
    sdir.mkdir(parents=True)
    (sdir / "prompt.md").write_text(prompt)
    writer.call = fake_writer([FIXED_ZERO, OVER_BUDGET, SMALL])
    rec = S.run_session("smoke-fake", "freearch", "feedback", prompt, sets["S-start"], sdir, rounds=2,
                        meta={"split_version": 2, "info": "M", "family": a.family, "protocol": "P2-FA-smoke"})
    for c in rec["candidates"]:
        print("round", c["round"], c["outcome"], c.get("support_solved"), "/", c.get("support_n"),
              (c.get("artifact_error") or "")[:160], (c.get("audit") or {}).get("count"), flush=True)
    assert [c["outcome"] for c in rec["candidates"]] == ["evaluated", "invalid_artifact", "evaluated"]
    assert "budget" in rec["candidates"][1]["artifact_error"]
    fb0 = (sdir / "feedback0.md").read_text()
    assert "channel_max_at" in fb0 and "Revise your build() network" in fb0
    assert "budget" in (sdir / "feedback1.md").read_text()
    assert (sdir / f"round{rec['final_round']}.graph.json").exists()
    env = dict(os.environ)
    py = sys.executable
    run([py, str(HERE / "eval_artifacts.py"), str(root / a.family), "--py-workers", "1"], env)
    ev = json.loads((sdir / "eval.json").read_text())
    print("eval", {k: v["solve_rate"] for k, v in ev["sets"].items()}, flush=True)
    run([py, str(HERE / "postdiag_parallel.py"), str(root / a.family), "--workers", "1", "--gpus", "1"], env)
    blind, diag = json.loads((sdir / "blind.json").read_text()), json.loads((sdir / "diag.json").read_text())
    print("blind", blind, "diag", diag, flush=True)
    (sdir / "blind.json").rename(sdir / "blind_postdiag.json")
    run([py, str(HERE / "blind_eval.py"), str(root / a.family)], env)
    assert json.loads((sdir / "blind.json").read_text()) == blind
    print("SMOKE OK", sdir, flush=True)

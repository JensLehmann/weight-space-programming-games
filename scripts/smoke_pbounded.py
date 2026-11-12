"""Offline smoke test of the bounded program player (pbounded) pipeline on the instrumentation-only toy game
`toy_reach` (wsp.toy; not a study result). No model call: wsp.writer.call is replaced by canned responses
containing hand-written players.

A feedback session (3 rounds) on the toy game's support set: round 0 = all-zero scores (always the lowest available
action), round 1 = a player over the operation budget (invalid artifact), round 2 = the prompt's example player.
Then eval_artifacts.py evaluates the final artifact on the support set (S-start, from a toy query manifest written
under --out) and its success must equal the session's own support evaluation of that round.

Usage: smoke_pbounded.py [--out <runs>/pb_smoke]
"""
import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wsp import toy, writer  # noqa: E402
from wsp.envs import build_start, obs_of  # noqa: E402
from wsp.manifest import RUNS, sha256_json  # noqa: E402
from wsp.prompts_pb import EXAMPLE, build_prompt_pb  # noqa: E402
from wsp.queries import PRIVATE_MANIFESTS_ENV, reviewed_start  # noqa: E402
import wsp.session as S  # noqa: E402

FAMILY = "toy_reach"
GAME = ("## The game\n\nThe complete simulator source code is below. Levels are generated from the seed; "
        "evaluation uses unseen seeds.\n\n```python\n" + inspect.getsource(toy) + "```\n")   # as scripts/smoke_toy.py

ZERO = '''import numpy as np

MEMORY_SIZE = 0


def act(obs, available, memory, prev_action):
    return np.zeros(8), np.zeros((64, 64)), memory
'''

OVER_BUDGET = '''import numpy as np

MEMORY_SIZE = 4
T = 256


def act(obs, available, memory, prev_action):
    x = obs * 1.0
    for _ in range(T):
''' + "".join(f"        x = np.maximum(x - {i}.5, 0.0)\n" for i in range(20)) + '''    return np.zeros(8), x, memory
'''


def fake_writer(programs):
    def call(model, messages, tag, out_dir, effort=None, task_budget=None):
        k = int(tag.replace("round", ""))
        text = f"Smoke-test program {k} (hand-written).\n\n```python\n{programs[k]}```\n"
        (Path(out_dir) / f"{tag}.call.json").write_text(json.dumps({"fake": True, "messages": len(messages)}))
        return writer.CallResult(f"fake-{k}", model, "completed", text=text, usage={"input": 0, "output": 0})
    return call


def write_toy_manifest(root: Path, specs) -> Path:
    """<root>/manifests/rev2/queries_toy_reach.json with the support set as S-start (layout of wsp.queries)."""
    sets = {"S-start": [s.to_json() for s in specs]}
    p = root / "manifests" / "rev2" / f"queries_{FAMILY}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"family": FAMILY, "split_version": 2, "sets": sets, "sha256": sha256_json(sets)}))
    return root / "manifests"


def run(cmd, env):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    assert r.returncode == 0, cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "pb_smoke"))
    a = ap.parse_args()
    root = Path(a.out)
    sdir = root / FAMILY / "smoke-fake" / "pbounded_feedback_M_r0"
    if sdir.exists():
        shutil.rmtree(sdir)
    support = [reviewed_start(FAMILY, s, l, "S-start") for s in range(5) for l in range(4)]
    prompt = build_prompt_pb(GAME, [(f"seed {s.seed}, level {s.level + 1}, start", obs_of(build_start(s)))
                                    for s in support[:4]])
    sdir.mkdir(parents=True)
    (sdir / "prompt.md").write_text(prompt)
    writer.call = fake_writer([ZERO, OVER_BUDGET, EXAMPLE])
    rec = S.run_session("smoke-fake", "pbounded", "feedback", prompt, support, sdir, rounds=2, effort="high",
                        meta={"split_version": 2, "info": "M", "family": FAMILY, "protocol": "P2-PB-smoke"})
    for c in rec["candidates"]:
        print("round", c["round"], c["outcome"], c.get("support_solved"), "/", c.get("support_n"),
              (c.get("artifact_error") or "")[:200], (c.get("audit") or {}).get("count"), flush=True)
    assert [c["outcome"] for c in rec["candidates"]] == ["evaluated", "invalid_artifact", "evaluated"]
    assert "budget" in rec["candidates"][1]["artifact_error"]
    fb0, fb1 = (sdir / "feedback0.md").read_text(), (sdir / "feedback1.md").read_text()
    assert "Player outputs at that frame" in fb0 and "Revise your act() program." in fb0, fb0[-2000:]
    assert "budget" in fb1
    k = rec["final_round"]
    assert k == 2 and (sdir / "round2.npz").exists() and (sdir / "round2.pb.json").exists()
    for key in ("model", "condition", "mode", "rounds_used", "effort", "task_budget", "variant", "split_version",
                "info", "family", "protocol", "final_round", "wall_s", "candidates", "tokens_out", "tokens_in"):
        assert key in rec, key
    env = dict(os.environ, **{PRIVATE_MANIFESTS_ENV: str(write_toy_manifest(root, support))})
    run([sys.executable, str(HERE / "eval_artifacts.py"), str(root / FAMILY), "--sets", "S-start",
         "--py-workers", "1"], env)
    ev = json.loads((sdir / "eval.json").read_text())
    s = ev["sets"]["S-start"]
    print("eval", ev["condition"], s, flush=True)
    assert ev["condition"] == "pbounded" and s["n"] == rec["candidates"][2]["support_n"]
    assert round(s["solve_rate"] * s["n"]) == rec["candidates"][2]["support_solved"]
    assert s["errors"] == 0
    print("SMOKE OK", sdir, flush=True)

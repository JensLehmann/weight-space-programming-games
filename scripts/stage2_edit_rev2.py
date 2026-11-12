"""Revision-2 RL->WSP (plan §6): writers edit an RL-trained actor via edit(params), with support feedback.

Arms (interface):
  summary  per-tensor shapes, norms, max|w| and nonzero counts only
  inspect  summary + full values of the small tensors (output, memory and readout layers) and
           per-output-channel statistics of the convolutional / encoder tensors
Mechanics-only source (condition M), revision-2 sets. The budget-matched fresh-synthesis control
is the W2 session of the same writer and family in runs/rev2_private (same section, support
observations, feedback rounds and request budget, no trained actor).

runs/<tag>/<family>/<model>/edit_<arm>_r<k>/
Usage: stage2_edit_rev2.py --family F --base runs/rev2_baselines/F/ppo_s0/student.npz
                           --models astra-6,sol-6.1,opus-5.5 --arms summary,inspect --runs 3
"""
import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_rev2 as R  # noqa: E402  (locked evaluator, section cache)
from wsp import actor as A  # noqa: E402
from wsp.manifest import RUNS, sha256_bytes  # noqa: E402
from wsp.prompts import EDIT_TASK, actor_spec, episode_rules, grid_hex, weight_summary  # noqa: E402
from wsp.queries import load_sets  # noqa: E402
from wsp.writer import MODELS  # noqa: E402
import wsp.session as S  # noqa: E402  (the module run_rev2 patches with the locked evaluator)


def diagnosis_text(params, specs):
    """Support-episode diagnosis of the trained actor shown to the writer (table + failed episodes)."""
    art = S.Artifact("weights", "", params=params)
    res = S.evaluate(art, specs)
    summary, failures = S.diagnose(art, specs, res)
    lines = [f"Solved {summary['solved']}/{summary['n']} support episodes.", "",
             "| episode | template | level | start | result | actions |", "|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        lines.append(f"| {r['i']} | {r['seed']} | {r['level'] + 1} | {r['start']} | {r['result']} | {r['actions']} |")
    for f in failures:
        lines += ["", f"### Failed episode {f['i']}: {f['why']}",
                  f"First actions: {' '.join(f['trace'][:40])}",
                  f"Frame at step {f['step']}:", "```\n" + grid_hex(f["frame"]) + "\n```"]
        if f.get("internals"):
            lines.append("Network internals at that frame: " + json.dumps(f["internals"]))
    return "\n".join(lines), summary


SMALL = ("Wo", "bo", "Wm", "Wmz", "Wma", "bm", "Wc", "bc")
CONV = ("W1", "b1", "W2", "b2", "W3", "b3", "Wh", "Uh", "bh", "Wz", "bz")


def _fmt(v):
    return np.array2string(np.asarray(v, np.float32), precision=3, separator=",", threshold=10 ** 7,
                           max_line_width=160, suppress_small=False)


def inspection_text(p):
    lines = ["Full values of the small tensors (float32, 3 significant decimals):"]
    for k in SMALL:
        lines += [f"### {k} {tuple(np.asarray(p[k]).shape)}", "```", _fmt(p[k]), "```"]
    lines.append("Per-output-channel statistics of the encoder tensors (l2, mean, min, max over each output row):")
    for k in CONV:
        v = np.asarray(p[k], np.float64)
        rows = v.reshape(v.shape[0], -1) if v.ndim > 1 else v[:, None]
        stats = [f"{i}:({np.linalg.norm(r):.3f},{r.mean():.3f},{r.min():.3f},{r.max():.3f})" for i, r in enumerate(rows)]
        lines += [f"### {k} {tuple(v.shape)}", " ".join(stats)]
    return "\n".join(lines)


def build_prompt(section, support_obs, base, diagnosis, arm):
    head = ("## The current trained weights (summary; the full arrays are passed to edit())\n\n"
            + weight_summary(base))
    if arm == "inspect":
        head += "\n\n" + inspection_text(base)
    parts = ["# Repairing a compact neural game-playing agent", section, episode_rules(),
             "## Example observations (support set)\n\nEach grid is 64 rows of 64 hexadecimal colour ids.\n"
             + "\n\n".join(f"### {name}\n```\n{grid_hex(g)}\n```" for name, g in support_obs),
             actor_spec(), head, "## Current behaviour on the support episodes\n\n" + diagnosis, EDIT_TASK]
    return "\n\n".join(parts)


def job(tag, fam, base_path, model, arm, run, rounds, effort):
    out = RUNS / tag / fam / model / f"edit_{arm}_r{run}"
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    sets = load_sets(fam, rev=2)
    base = A.load_npz(Path(base_path))
    diag, _ = diagnosis_text(base, sets["S-start"])
    sec, _ = R.section(fam, "M")
    prompt = build_prompt(sec, R.support_observations(sets), base, diag, arm)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = R.CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": "M", "family": fam, "protocol": "P2", "arm": arm,
            "base": str(base_path), "prompt_sha256": sha256_bytes(prompt.encode())}
    return R.S.run_session(model, "edit", "feedback", prompt, sets["S-start"], out, rounds=rounds, base=base,
                           effort=effort, task_budget=tb, meta=meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev2_stage2_edit")
    ap.add_argument("--family", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--arms", default="summary,inspect")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        sets = load_sets(a.family, rev=2); base = A.load_npz(Path(a.base))
        sec, _ = R.section(a.family, "M")
        for arm in a.arms.split(","):
            p = build_prompt(sec, R.support_observations(sets), base, "(diagnosis)", arm)
            print(arm, len(p) // 4, "tok~")
        sys.exit(0)
    jobs = [(a.tag, a.family, a.base, m, arm, r, a.rounds, a.effort)
            for r in range(a.runs) for arm in a.arms.split(",") for m in a.models.split(",")]
    print(f"{len(jobs)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                print(json.dumps({"job": [j[1], j[3], j[4], j[5]],
                                  "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"), c.get("support_n"),
                                              c["call"]["status"]) for c in rec["candidates"]]}), flush=True)
            except Exception:
                print("FAILED", j, traceback.format_exc()[-1200:], flush=True)

"""Public-tier writer sessions (protocol P2, public addendum). Same runner as run_rev2.py with three
public-specific differences:
  * feedback rounds evaluate on S-start + S-dev (S-start alone is only 3-5 episodes);
  * the task text does not mention renderer templates (public games have one canonical rendering);
  * support observations are labelled by level only.

runs/<tag>/official.<id>/<model>/<cond>_<mode>_M_<variant>_r<k>/
"""
import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_rev2 as R  # noqa: E402  (installs the locked evaluator)
from wsp.envs import build_start, obs_of  # noqa: E402
from wsp.manifest import RUNS, sha256_bytes  # noqa: E402
from wsp.prompts import build_prompt_v2  # noqa: E402
from wsp.queries import load_sets  # noqa: E402
from wsp.writer import MODELS  # noqa: E402

import threading  # noqa: E402
_PY_EVAL = threading.BoundedSemaphore(4)   # cap concurrent sandboxed python evaluations per process
_locked = R.S.evaluate


def _capped_eval(art, specs, device=None, batch=64):
    if art.condition in ("python", "pyreact"):
        with _PY_EVAL:
            return _locked(art, specs, device, batch)
    return _locked(art, specs, device, batch)


R.S.evaluate = _capped_eval

EXCLUDED = set()   # all 25 public games
_SUBS = [
    ("on levels, layouts and visual variants it has never seen** (the evaluation uses held-out levels, held-out "
     "renderer templates and perturbed start states)",
     "on levels and layouts it has never seen** (the evaluation uses held-out levels and perturbed start states)"),
    ("**by itself on\nlevels, layouts and visual variants it has never seen** (the evaluation uses held-out levels,\n"
     "held-out renderer templates and perturbed start states)",
     "**by itself on\nlevels and layouts it has never seen** (the evaluation uses held-out levels and perturbed\n"
     "start states)"),
    ("**on levels,\nlayouts and visual variants it has never seen** (the evaluation uses held-out levels, held-out\n"
     "renderer templates and perturbed start states)",
     "**on levels\nand layouts it has never seen** (the evaluation uses held-out levels and perturbed start states)"),
]


def public_prompt(sec, cond, obs, variant):
    p = build_prompt_v2(sec, cond, obs, variant=variant)
    n = 0
    for a, b in _SUBS:
        if a in p:
            p, n = p.replace(a, b), n + 1
    if n != 1 or "renderer templates and perturbed" in p:
        raise RuntimeError(f"public task-text substitution failed ({n} matches)")
    return p


def job(tag, fam, model, cond, mode, info, variant, run, rounds, effort):
    out = RUNS / tag / fam / model / f"{cond}_{mode}_{info}_{variant}_r{run}"
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    sets = load_sets(fam, rev=2)
    sec, report = R.section(fam, info)
    obs = [(f"level {s.level + 1}, start", obs_of(build_start(s))) for s in sets["S-start"]]
    prompt = public_prompt(sec, cond, obs, variant)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = R.CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": info, "family": fam, "protocol": "P2-public", "tier": "public",
            "prompt_sha256": sha256_bytes(prompt.encode()), "feedback_sets": ["S-start", "S-dev"]}
    return R.S.run_session(model, cond, mode, prompt, sets["S-start"] + sets["S-dev"], out, rounds=rounds,
                           effort=effort, task_budget=tb, variant=variant, meta=meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev2_public")
    ap.add_argument("--games", required=True, help="comma list of ids (ar25,...) or 'all'")
    ap.add_argument("--models", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="build prompts only (no calls)")
    a = ap.parse_args()
    from wsp.redact_public import GAMES
    ids = list(GAMES) if a.games == "all" else a.games.split(",")
    fams = [f"official.{g}" for g in ids if f"official.{g}" not in EXCLUDED]
    if a.dry:
        for f in fams:
            sets = load_sets(f, rev=2)
            sec, _ = R.section(f, "M")
            obs = [(f"level {s.level + 1}, start", obs_of(build_start(s))) for s in sets["S-start"]]
            for c in ("weights", "python"):
                p = public_prompt(sec, c, obs, "full")
                print(f, c, len(p) // 4, "tok~", len(sets["S-start"]), "+", len(sets["S-dev"]), "feedback eps")
        sys.exit(0)
    jobs = []
    for r in range(a.runs):
        for c in a.conds.split(","):
            cond, mode, info, variant = R.expand(c)
            for f in fams:
                for m in a.models.split(","):
                    jobs.append((a.tag, f, m, cond, mode, info, variant, r, a.rounds, a.effort))
    print(f"{len(jobs)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                print(json.dumps({"job": [j[1], j[2], j[3], j[4], j[5], j[6], j[7]],
                                  "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"), c.get("support_n"),
                                              c["call"]["status"]) for c in rec["candidates"]],
                                  "tok_out": rec["tokens_out"]}), flush=True)
            except Exception:
                print("FAILED", j, traceback.format_exc()[-1200:], flush=True)

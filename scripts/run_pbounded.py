"""Bounded program player (pbounded) writer sessions: the writer writes a NumPy player under the free-architecture
computation model (wsp.pbounded). Modelled on run_freearch.py (same game sections, support observations, feedback
episodes, task budget and meta fields). Resumable.

runs/<tag>/<family>/<model>/pbounded_<mode>_M_r<k>/
    prompt.md, round<k>.py, round<k>.npz (constants), round<k>.pb.json (constant structure + static analysis),
    feedback<k>.md, session.json, ...

Usage: run_pbounded.py [--tag rev2_pbounded] [--families ztl,...] [--models astra-6,sol-6.1,opus-5.5,terra-6.1]
                       [--modes feedback] [--runs 3] [--rounds 3] [--effort high] [--workers N] [--max-evals 3]
                       [--dry] [--no-freeze-check]
--dry builds every prompt and prints its size; no API call and nothing is written.
Before launching, every prompt is compared with manifests/rev2/freeze_pbounded.json (unless --no-freeze-check).
"""
import argparse
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_rev2 as R2  # noqa: E402  (section(), support_observations(), CLAUDE_TASK_BUDGET, eval lock)
from wsp import pbounded as PB  # noqa: E402
from wsp.manifest import RUNS, sha256_bytes  # noqa: E402
from wsp.prompts_pb import build_prompt_pb  # noqa: E402
from wsp.queries import load_sets  # noqa: E402
from wsp.writer import MODELS  # noqa: E402
import wsp.session as S  # noqa: E402

INFO = "M"                                # mechanics-only game section (primary condition)
COND = "pbounded"
PB_FAMILIES = ["ztl", "wih", "qcn", "lys", "esm", "aiv", "nkp", "obn", "qwv", "ruz", "vvf", "ubr", "yuz", "bvn", "trx",
               "yiy", "crg", "qpv", "ydv", "mds"]
PB_MODELS = ["astra-6", "sol-6.1", "opus-5.5", "terra-6.1"]
FREEZE = Path(__file__).resolve().parent.parent / "manifests" / "rev2" / "freeze_pbounded.json"

# Feedback evaluations run sandboxed player processes (16 per evaluation): cap how many run at once.
_PB_EVAL = threading.BoundedSemaphore(3)
_inner_eval = S.evaluate


def _capped_eval(art, specs, device=None, batch=64):
    if art.condition == COND:
        with _PB_EVAL:
            return _inner_eval(art, specs, device, batch)
    return _inner_eval(art, specs, device, batch)


S.evaluate = _capped_eval

PUBLIC_SUB = ("**by itself on levels, layouts and visual variants it has never seen**\n(the evaluation uses held-out "
              "levels, held-out renderer templates and perturbed start states).",
              "**by itself on levels and layouts it has never seen**\n(the evaluation uses held-out levels and "
              "perturbed start states).")


def prompt_for(fam):
    sets = load_sets(fam, rev=2)
    sec, _ = R2.section(fam, INFO)
    if fam.startswith("official."):          # public tier: one canonical rendering (as run_freearch.py)
        from wsp.envs import build_start, obs_of
        obs = [(f"level {s.level + 1}, start", obs_of(build_start(s))) for s in sets["S-start"]]
        p = build_prompt_pb(sec, obs)
        if PUBLIC_SUB[0] not in p:
            raise RuntimeError("public task-text substitution failed")
        return p.replace(PUBLIC_SUB[0], PUBLIC_SUB[1]), sets
    return build_prompt_pb(sec, R2.support_observations(sets)), sets


def feedback_specs(fam, sets):
    return sets["S-start"] + sets["S-dev"] if fam.startswith("official.") else sets["S-start"]


def job(tag, fam, model, mode, run, rounds, effort):
    out = RUNS / tag / fam / model / f"{COND}_{mode}_{INFO}_r{run}"
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    prompt, sets = prompt_for(fam)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = R2.CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": INFO, "family": fam, "protocol": "P2-PB",
            "prompt_sha256": sha256_bytes(prompt.encode()), "pb_budget": PB.budget()}
    return S.run_session(model, COND, mode, prompt, feedback_specs(fam, sets), out, rounds=rounds, effort=effort,
                         task_budget=tb, variant="full", meta=meta)


def dry(families):
    print(f"{'family':8s} {'chars':>8s} {'~tokens':>8s}  sha256", flush=True)
    for f in families:
        p, _ = prompt_for(f)
        print(f"{f:8s} {len(p):8d} {len(p) // 4:8d}  {sha256_bytes(p.encode())[:16]}", flush=True)


def check_freeze(families):
    if not FREEZE.exists():
        raise SystemExit(f"{FREEZE} is missing (run scripts/freeze_pbounded.py, or pass --no-freeze-check)")
    frozen = json.loads(FREEZE.read_text())["prompts"][COND]
    bad = [f for f in families if frozen.get(f) != sha256_bytes(prompt_for(f)[0].encode())]
    if bad:
        raise SystemExit(f"prompts differ from the freeze for {bad}")
    print(f"prompts match the freeze ({len(families)} families)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev2_pbounded")
    ap.add_argument("--families", default=",".join(PB_FAMILIES))
    ap.add_argument("--models", default=",".join(PB_MODELS))
    ap.add_argument("--modes", default="feedback", help="comma list of oneshot / feedback")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-evals", type=int, default=3, help="concurrent feedback evaluations of players")
    ap.add_argument("--dry", action="store_true", help="build prompts and print sizes; no API call")
    ap.add_argument("--no-freeze-check", action="store_true")
    a = ap.parse_args()
    fams = a.families.split(",")
    if a.dry:
        dry(fams)
        raise SystemExit(0)
    modes = a.modes.split(",")
    if any(m not in ("oneshot", "feedback") for m in modes):
        raise SystemExit(f"unknown mode in {modes} (use oneshot / feedback)")
    models = a.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    if not a.no_freeze_check:
        check_freeze(fams)
    _PB_EVAL = threading.BoundedSemaphore(a.max_evals)
    jobs = [(a.tag, f, m, mode, r, a.rounds, a.effort) for r in range(a.runs) for mode in modes
            for f in fams for m in models]
    print(f"{len(jobs)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                print(json.dumps({"job": [j[1], j[2], COND, j[3], INFO, j[4]],
                                  "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"), c.get("support_n"),
                                              c["call"]["status"]) for c in rec["candidates"]],
                                  "tok_out": rec["tokens_out"]}), flush=True)
            except Exception:
                print("FAILED", j, traceback.format_exc()[-1200:], flush=True)

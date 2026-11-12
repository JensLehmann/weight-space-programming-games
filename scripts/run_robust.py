"""Robust-weights W2 writer sessions (wsp/robust.py). Resumable; one directory per session.

runs/rev2_robust/<family>/<model>/weights_feedback_M_robust_r<k>/

Usage: run_robust.py [--families ztl,lys,esm,ruz,crg,mds] [--models astra-6,sol-6.1,opus-5.5]
                     [--runs 3] [--workers N] [--dry]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent))
import run_rev2 as R2  # noqa: E402  (installs the evaluation lock and the section cache)
from wsp.manifest import RUNS, sha256_bytes  # noqa: E402
from wsp.prompts import build_prompt_v2  # noqa: E402
from wsp.queries import load_sets  # noqa: E402
from wsp.robust import ROBUST_EPS, robust_prompt, run_robust_session  # noqa: E402
from wsp.writer import MODELS  # noqa: E402
import wsp.session as S  # noqa: E402

TAG = "rev2_robust"
FAMS = "ztl,lys,esm,ruz,crg,mds"              # the six RL (Stage 2) families
MODELS_DEFAULT = "astra-6,sol-6.1,opus-5.5"


def out_dir(fam, model, run):
    return RUNS / TAG / fam / model / f"weights_feedback_M_robust_r{run}"


def prompt_for(fam):
    sets = load_sets(fam, rev=2)
    sec, _ = R2.section(fam, "M")
    return sets, robust_prompt(build_prompt_v2(sec, "weights", R2.support_observations(sets), variant="full"))


def job(fam, model, run, rounds, effort):
    out = out_dir(fam, model, run)
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    sets, prompt = prompt_for(fam)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = R2.CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": "M", "family": fam, "protocol": "P2+robust",
            "prompt_sha256": sha256_bytes(prompt.encode())}
    rec = run_robust_session(model, prompt, sets["S-start"], out, rounds=rounds, effort=effort,
                             task_budget=tb, meta=meta)
    if any(c.get("outcome") == "writer_error" for c in rec["candidates"]):
        # a request ended without an outcome: keep the record for request accounting but make the
        # session restartable on the next launch
        n = len(list(out.glob("session_restarted_*.json")))
        (out / "session.json").rename(out / f"session_restarted_{n}.json")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=FAMS)
    ap.add_argument("--models", default=MODELS_DEFAULT)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="write prompt.md files only, no writer calls")
    a = ap.parse_args()
    S.evaluate = R2._locked_eval   # the robust session imports evaluate from wsp.session at call time
    import wsp.robust as RB
    RB.evaluate = R2._locked_eval
    jobs = [(f, m, r, a.rounds, a.effort) for r in range(a.runs) for f in a.families.split(",")
            for m in a.models.split(",")]
    print(f"{len(jobs)} sessions, eps {ROBUST_EPS:g}", flush=True)
    if a.dry:
        for f in a.families.split(","):
            _, p = prompt_for(f)
            d = RUNS / TAG / "_dry"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{f}.md").write_text(p)
        print("dry prompts in", RUNS / TAG / "_dry")
        sys.exit(0)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                print(json.dumps({"job": list(j[:3]),
                                  "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"),
                                              c.get("support_solved_noisy"), c.get("support_n"),
                                              c["call"]["status"]) for c in rec["candidates"]],
                                  "tok_out": rec["tokens_out"]}), flush=True)
            except Exception:
                print("FAILED", j, traceback.format_exc()[-1200:], flush=True)

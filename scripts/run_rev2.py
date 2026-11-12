"""Revision-2 writer sessions (protocol P2). Resumable; one directory per session.

runs/<tag>/<family>/<model>/<cond>_<mode>_<info>_<variant>_r<k>/
"""
import argparse
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from wsp.envs import build_start, new_game, obs_of
from wsp.manifest import RUNS, sha256_bytes
from wsp.prompts import build_prompt_v2
from wsp.queries import load_sets
from wsp.redact2 import game_section_v2
from wsp.writer import MODELS
import wsp.session as S

CLAUDE_TASK_BUDGET = 96_000
_eval_lock = threading.Lock()
_orig_eval = S.evaluate


def _locked_eval(art, specs, device=None, batch=64):
    if art.condition in ("weights", "edit"):
        with _eval_lock:
            return _orig_eval(art, specs, device, batch)
    return _orig_eval(art, specs, device, batch)


S.evaluate = _locked_eval
_section_cache: dict = {}
_cache_lock = threading.Lock()


def section(fam, info):
    with _cache_lock:
        if (fam, info) not in _section_cache:
            if fam.startswith("official."):          # public tier: per-game reviewed configs, condition M only
                from wsp.redact_public import public_section
                _section_cache[(fam, info)] = public_section(fam.split(".", 1)[1], info)
            else:
                n = new_game(fam, 0).n_levels
                _section_cache[(fam, info)] = game_section_v2(fam, n, info)
        return _section_cache[(fam, info)]


def support_observations(sets):
    start = sets["S-start"]
    t0 = [s for s in start if s.seed == start[0].seed]
    first = min(s.level for s in start)
    others = [s for s in start if s.seed != t0[0].seed and s.level == first]
    return [(f"template {s.seed}, level {s.level + 1}, start", obs_of(build_start(s))) for s in t0 + others]


def job(tag, fam, model, cond, mode, info, variant, run, rounds, effort):
    out = RUNS / tag / fam / model / f"{cond}_{mode}_{info}_{variant}_r{run}"
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    sets = load_sets(fam, rev=2)
    sec, report = section(fam, info)
    prompt = build_prompt_v2(sec, cond, support_observations(sets), variant=variant)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": info, "family": fam, "protocol": "P2",
            "prompt_sha256": sha256_bytes(prompt.encode())}
    rec = S.run_session(model, cond, mode, prompt, sets["S-start"], out, rounds=rounds, effort=effort,
                        task_budget=tb, variant=variant, meta=meta)
    return rec


def expand(spec: str):
    """cond:mode:info:variant, e.g. weights:oneshot:M:full"""
    parts = spec.split(":")
    while len(parts) < 4:
        parts.append({2: "M", 3: "full"}[len(parts)])
    return tuple(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--conds", required=True, help="comma list of cond:mode[:info[:variant]]")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8, help="parallel sessions (does not affect results)")
    a = ap.parse_args()
    jobs = []
    for r in range(a.runs):
        for c in a.conds.split(","):
            cond, mode, info, variant = expand(c)
            for f in a.families.split(","):
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

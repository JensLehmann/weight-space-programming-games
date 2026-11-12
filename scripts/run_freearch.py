"""Free-architecture (FA) writer sessions: the writer designs the network (wsp.freearch) and its weights.
Modelled on run_rev2.py (same game sections, support observations, task budget, meta). Resumable.

runs/<tag>/<family>/<model>/freearch_<mode>_M_r<k>/
    prompt.md, round<k>.py, round<k>.npz (params), round<k>.graph.json (network), session.json, ...

Usage: run_freearch.py --tag rev2_fa --families ztl,lys --models astra-6 [--conds freearch:oneshot,freearch:feedback]
                       [--runs 3] [--rounds 3] [--effort high] [--workers N] [--dry]
--dry builds every prompt and prints its size; no API call and nothing is written.
"""
import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_rev2 as R2                     # section(), support_observations(), CLAUDE_TASK_BUDGET, eval lock
from wsp import freearch as FA
from wsp.manifest import RUNS, sha256_bytes
from wsp.prompts_fa import build_prompt_fa
from wsp.queries import load_sets
from wsp.writer import MODELS
import wsp.session as S

INFO = "M"                                # mechanics-only game section (primary condition)
FA_FAMILIES = ["ztl", "lys", "esm", "ruz", "crg", "mds"]


def _locked_eval(art, specs, device=None, batch=64):
    """GPU policies (weights, edit, freearch) share one lock, as in run_rev2."""
    if art.condition in ("weights", "edit", "freearch"):
        with R2._eval_lock:
            return R2._orig_eval(art, specs, device, batch)
    return R2._orig_eval(art, specs, device, batch)


S.evaluate = _locked_eval


PUBLIC_SUB = ("**by itself on levels, layouts and visual variants it has never seen**\n(the evaluation uses held-out "
              "levels, held-out renderer templates and perturbed start states).",
              "**by itself on levels and layouts it has never seen**\n(the evaluation uses held-out levels and "
              "perturbed start states).")


CONTROL_TEXT = {
    "fixed_graph": ("**Constraint for this run:** use exactly the fixed actor graph shown above (the same nodes, "
                    "ids, operations, shapes, loop count and memory size). Only the parameter values are yours to "
                    "choose; any other graph is rejected.\n\n"),
    "fixed_ops": ("**Constraint for this run:** you may design any graph within the budget, but use only the "
                  "fixed actor's operations: add, colmax, concat, conv, dense, gmax, gmean, loop, relu, rowmax, "
                  "vadd, vconcat, vrelu, vtanh. A graph with any other operation is rejected.\n\n"),
}


def _with_control(p):
    if FA.CONTROL is None:
        return p
    anchor = "Rules:\n- Any architecture expressible in the language"
    if anchor not in p:
        raise RuntimeError("control anchor not found in FA prompt")
    return p.replace(anchor, CONTROL_TEXT[FA.CONTROL] + anchor, 1)


def prompt_for(fam):
    return _prompt_for(fam)


def _prompt_for(fam):
    sets = load_sets(fam, rev=2)
    sec, _ = R2.section(fam, INFO)
    if fam.startswith("official."):          # public tier: one canonical rendering (as scripts/run_public.py)
        from wsp.envs import build_start, obs_of
        obs = [(f"level {s.level + 1}, start", obs_of(build_start(s))) for s in sets["S-start"]]
        p = build_prompt_fa(sec, obs)
        if PUBLIC_SUB[0] not in p:
            raise RuntimeError("public FA task-text substitution failed")
        return _with_control(p.replace(PUBLIC_SUB[0], PUBLIC_SUB[1])), sets
    return _with_control(build_prompt_fa(sec, R2.support_observations(sets))), sets


def feedback_specs(fam, sets):
    return sets["S-start"] + sets["S-dev"] if fam.startswith("official.") else sets["S-start"]


def job(tag, fam, model, mode, run, rounds, effort):
    out = RUNS / tag / fam / model / f"freearch_{mode}_{INFO}_r{run}"
    if (out / "session.json").exists():
        return json.loads((out / "session.json").read_text())
    prompt, sets = prompt_for(fam)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt)
    tb = R2.CLAUDE_TASK_BUDGET if MODELS[model].transport == "anthropic" else None
    meta = {"split_version": 2, "info": INFO, "family": fam, "protocol": "P2-FA",
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "fa_budget": {"params": FA.MAX_PARAMS, "macs_per_step": FA.MAX_MACS}, "fa_control": FA.CONTROL}
    return S.run_session(model, "freearch", mode, prompt, feedback_specs(fam, sets), out, rounds=rounds, effort=effort,
                         task_budget=tb, variant="full", meta=meta)


def dry(families):
    print(f"{'family':8s} {'chars':>8s} {'~tokens':>8s}  sha256", flush=True)
    for f in families:
        p, _ = prompt_for(f)
        print(f"{f:8s} {len(p):8d} {len(p) // 4:8d}  {sha256_bytes(p.encode())[:16]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rev2_fa")
    ap.add_argument("--families", default=",".join(FA_FAMILIES))
    ap.add_argument("--models", default="astra-6,sol-6.1,opus-5.5")
    ap.add_argument("--conds", default="freearch:oneshot,freearch:feedback", help="comma list of freearch:<mode>")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="build prompts and print sizes; no API call")
    ap.add_argument("--control", choices=("fixed_graph", "fixed_ops"), default=None,
                    help="language controls: fixed actor graph only, or fixed actor operations only")
    a = ap.parse_args()
    FA.CONTROL = a.control
    fams = a.families.split(",")
    if a.dry:
        dry(fams)
        raise SystemExit(0)
    modes = []
    for c in a.conds.split(","):
        cond, mode = c.split(":")
        if cond != "freearch" or mode not in ("oneshot", "feedback"):
            raise SystemExit(f"unknown condition {c!r} (use freearch:oneshot / freearch:feedback)")
        modes.append(mode)
    models = a.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    jobs = [(a.tag, f, m, mode, r, a.rounds, a.effort) for r in range(a.runs) for mode in modes
            for f in fams for m in models]
    print(f"{len(jobs)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                rec = fu.result()
                print(json.dumps({"job": [j[1], j[2], "freearch", j[3], INFO, j[4]],
                                  "rounds": [(c["round"], c.get("outcome"), c.get("support_solved"), c.get("support_n"),
                                              c["call"]["status"]) for c in rec["candidates"]],
                                  "tok_out": rec["tokens_out"]}), flush=True)
            except Exception:
                print("FAILED", j, traceback.format_exc()[-1200:], flush=True)

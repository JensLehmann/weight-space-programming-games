"""Robust written weights: W2 sessions whose actor must survive weight noise. Fixed actor only.

Question: are written weights fragile by nature, or only because nobody asked for robustness?
The writer gets the standard W2 prompt plus one requirement: the actor must keep its behaviour under
isotropic weight noise of ROBUST_EPS times each tensor's RMS, added to every entry including the exact
zeros (the perturbation of scripts/perturbation_rev2.py, which collapses standard written actors
already at 1e-7). Feedback reports support success of the exact actor and of one noisy draw; each
round uses a fresh draw (seeds FEEDBACK_SEED0 + round, disjoint from the evaluation seeds 0 and 1), so
a writer cannot tune to one draw. Selection of the final round uses exact + noisy support successes.

Everything else (game section, support set, rounds, effort, task budget, sandbox) is the standard W2
session of scripts/run_rev2.py and wsp/session.py, which stay unchanged.
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import actor as A
from . import writer
from .prompts import feedback_message
from .session import diagnose, evaluate, materialize

ROBUST_EPS = 1e-3
FEEDBACK_SEED0 = 1000

ROBUST_RULE = f"""- **Robustness requirement for this experiment.** The actor must keep playing well when every
  tensor receives independent Gaussian noise with standard deviation {ROBUST_EPS:g} times that tensor's
  root-mean-square value, added to every entry, including entries that are zero. The evaluation
  scores the actor both exactly and under such noise, and feedback reports both. Weights whose
  behaviour depends on exact cancellations, exact zeros or very large gains usually fail under
  this noise."""

_ANCHOR = "- Any tensor you do not need can be zero.\n"


def robust_prompt(prompt: str) -> str:
    assert prompt.count(_ANCHOR) == 1, "weights-task rules not found in prompt"
    return prompt.replace(_ANCHOR, _ANCHOR + ROBUST_RULE + "\n")


def noise_direction(params: dict, seed: int, tag: str) -> dict:
    """Same construction as scripts/perturbation_rev2.noise_direction."""
    rng = np.random.default_rng([seed, zlib.crc32(tag.encode())])
    return {k: rng.standard_normal(params[k].shape) for k in sorted(params)}


def perturb(params: dict, eps: float, z: dict) -> dict:
    """Same construction as scripts/perturbation_rev2.perturb (isotropic, per-tensor RMS, all entries)."""
    out = {}
    for k, v in params.items():
        v64 = np.asarray(v).astype(np.float64)
        rms = float(np.sqrt(np.mean(v64 ** 2)))
        out[k] = (v64 + eps * rms * z[k]).astype(np.float32)
    return out


def _robust_feedback(clean: dict, noisy: dict, failures: list, k: int, left: int, noisy_shown: bool) -> str:
    base = feedback_message(clean if not noisy_shown else noisy, failures, k, left, "weights")
    head = (f"**Robustness check:** the exact actor solved {clean['solved']}/{clean['n']} support "
            f"episodes, the actor under one random noise draw (std {ROBUST_EPS:g} x tensor RMS on every "
            f"entry; a new draw is used each round and in the evaluation) solved "
            f"{noisy['solved']}/{noisy['n']}. The table and failure details below are for the "
            f"{'noisy' if noisy_shown else 'exact'} actor.\n\n")
    return head + base


def run_robust_session(model: str, prompt: str, support: list, out_dir: Path, rounds: int = 3,
                       effort: str | None = None, task_budget: int | None = None,
                       meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt}]
    candidates = []
    n_rounds = 1 + rounds
    t0 = time.time()
    tag = f"robust/{out_dir.parent.parent.name}/{out_dir.parent.name}/{out_dir.name}"
    for k in range(n_rounds):
        res = writer.call(model, messages, tag=f"round{k}", out_dir=out_dir, effort=effort,
                          task_budget=task_budget)
        cand = {"round": k, "call": res.to_json()}
        if res.status != "completed":
            cand["outcome"] = f"writer_{res.status}"
            candidates.append(cand)
            break
        code = writer.extract_code(res.text)
        art = materialize("weights", code)
        (out_dir / f"round{k}.py").write_text(code or "")
        if art.params is not None:
            A.save_npz(art.params, out_dir / f"round{k}.npz")
        clean_res = evaluate(art, support)
        clean, clean_fail = diagnose(art, support, clean_res)
        if art.params is not None:
            z = noise_direction(art.params, FEEDBACK_SEED0 + k, tag)
            nart = replace(art, params=perturb(art.params, ROBUST_EPS, z))
        else:
            nart = art
        noisy_res = evaluate(nart, support)
        noisy, noisy_fail = diagnose(nart, support, noisy_res)
        cand.update({"artifact_error": art.error, "audit": art.audit, "init_wall_s": art.init_wall_s,
                     "support_solved": clean["solved"], "support_n": clean["n"],
                     "support_solved_noisy": noisy["solved"], "noise_seed": FEEDBACK_SEED0 + k,
                     "outcome": "invalid_artifact" if art.error else "evaluated"})
        candidates.append(cand)
        if k + 1 < n_rounds:
            if clean["solved"] == clean["n"] and noisy["solved"] == noisy["n"]:
                break
            noisy_shown = clean["solved"] == clean["n"]
            fb = _robust_feedback(clean, noisy, noisy_fail if noisy_shown else clean_fail,
                                  k, n_rounds - k - 1, noisy_shown)
            messages += [{"role": "assistant", "content": res.text}, {"role": "user", "content": fb}]
            (out_dir / f"feedback{k}.md").write_text(fb)
    evaluated = [c for c in candidates if c.get("outcome") == "evaluated"]
    final = (max(evaluated, key=lambda c: (c["support_solved"] + c["support_solved_noisy"], c["round"]))
             if evaluated else None)
    record = {"model": model, "condition": "weights", "mode": "feedback", "variant": "full",
              "robust_eps": ROBUST_EPS, "rounds_used": len(candidates), "effort": effort,
              "task_budget": task_budget, **(meta or {}),
              "final_round": None if final is None else final["round"],
              "wall_s": time.time() - t0, "candidates": candidates,
              "tokens_out": sum((c["call"]["usage"] or {}).get("output") or 0 for c in candidates),
              "tokens_in": sum((c["call"]["usage"] or {}).get("input") or 0 for c in candidates)}
    (out_dir / "session.json").write_text(json.dumps(record, indent=1, default=str))
    return record

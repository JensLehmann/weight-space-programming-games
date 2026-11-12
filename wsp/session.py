"""One writer session (one-shot or feedback-guided) and artifact evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import actor as A
from . import writer
from .envs import ACTIONS, EpisodeResult, EpisodeSpec, build_start, obs_of, avail_mask, run_episodes
from .prompts import feedback_message
from .sandbox import PythonPlayerPolicy, run_initializer


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Artifact:
    condition: str          # weights | python | pyreact | edit | freearch | pbounded
    code: str
    params: dict | None = None
    error: str | None = None
    audit: dict | None = None
    init_wall_s: float | None = None
    variant: str = "full"
    graph: dict | None = None   # freearch: {"graph": [...], "memory_size": M} (wsp.freearch);
                                # pbounded: {"spec": ..., "consts": ...} (wsp.pbounded; params = constant arrays)


PYREACT_STEP_TIMEOUT = 0.1     # seconds (prompt asks for 50 ms; 100 ms hard limit incl. IPC)
PYREACT_FLAGS = ("deepcopy", "copy.copy(", "def step", "def simulate", "def _simulate", "heapq", "deque(")


def materialize(condition: str, code: str | None, base: dict | None = None) -> Artifact:
    if code is None:
        return Artifact(condition, "", error="no python code block in response")
    if condition == "freearch":
        from . import freearch
        return freearch.materialize(code)
    if condition == "pbounded":
        from . import pbounded
        return pbounded.materialize(code)
    if condition == "edit":
        res = run_initializer(code, timeout=120, base=base)
        if res.error:
            return Artifact("weights", code, error=res.error, audit=res.audit.to_json())
        try:
            params = A.validate(res.params)
        except ValueError as exc:
            return Artifact("weights", code, error=f"invalid tensors: {exc}", audit=res.audit.to_json())
        au = res.audit.to_json()
        au["changed_entries"] = int(sum((np.asarray(params[k]) != base[k]).sum() for k in base))
        return Artifact("weights", code, params=params, audit=au, init_wall_s=res.wall_s)
    if condition in ("python", "pyreact"):
        from .sandbox import audit
        au = audit(code, "act")
        aj = au.to_json()
        if condition == "pyreact":
            aj["react_flags"] = [f for f in PYREACT_FLAGS if f in code]
        return Artifact(condition, code, error=None if au.ok else "audit: " + "; ".join(au.violations[:10]),
                        audit=aj)
    res = run_initializer(code, timeout=120)
    if res.error:
        return Artifact(condition, code, error=res.error, audit=res.audit.to_json(), init_wall_s=res.wall_s)
    try:
        params = A.validate(res.params)
    except ValueError as exc:
        return Artifact(condition, code, error=f"invalid tensors: {exc}", audit=res.audit.to_json(),
                        init_wall_s=res.wall_s)
    au = res.audit.to_json()
    au["cast"] = A.cast_report(res.params)
    return Artifact(condition, code, params=params, audit=au, init_wall_s=res.wall_s)


def policy_for(art: Artifact, device: str | None = None):
    if art.condition in ("weights", "edit"):
        return A.NeuralPolicy(A.Actor(art.params, variant=art.variant), device or default_device())
    if art.condition == "freearch":
        from . import freearch
        return freearch.policy(art, device or default_device())
    if art.condition == "pbounded":
        from . import pbounded
        return pbounded.policy(art)
    if art.condition == "pyreact":
        return PythonPlayerPolicy(art.code, step_timeout=PYREACT_STEP_TIMEOUT)
    return PythonPlayerPolicy(art.code)


def evaluate(art: Artifact, specs: list[EpisodeSpec], device: str | None = None,
             batch: int = 64) -> list[EpisodeResult]:
    if art.error:
        return [EpisodeResult(s.key, s.qset, s.family, s.seed, s.level, error=art.error) for s in specs]
    pol = policy_for(art, device)
    try:
        return run_episodes(specs, pol, batch=batch if art.condition in ("weights", "edit", "freearch") else 16)
    finally:
        if hasattr(pol, "close"):
            pol.close()


# --------------------------------------------------------------------------- diagnostics
def _probe_step(art: Artifact, spec: EpisodeSpec, trace: list, step: int):
    """Frame at `step` and (for weights) logits + channel summaries at that decision."""
    game = build_start(spec)
    for a, r, c in trace[:step]:
        if a in ACTIONS:
            game.step(a, r, c)
    frame = obs_of(game)
    if art.condition == "freearch":
        from . import freearch
        return frame, freearch.probe_internals(art, spec, trace, step)
    if art.condition == "pbounded":
        from . import pbounded
        return frame, pbounded.probe_internals(art, spec, trace, step)
    if art.condition not in ("weights", "edit"):
        return frame, None
    net = A.Actor(art.params, variant=art.variant).eval()
    mem = torch.zeros(1, A.MEM)
    prev = torch.zeros(1, len(ACTIONS))
    g2 = build_start(spec)
    with torch.no_grad():
        for k in range(step + 1):
            o = torch.as_tensor(obs_of(g2)[None])
            av = torch.as_tensor(avail_mask(g2)[None])
            logits, click, mem2 = net(o, mem, prev, av)
            if k == step:
                f3, h = net.encode(o)
                def summ(t):
                    out = []
                    for ch in range(t.shape[1]):
                        v = t[0, ch]
                        idx = int(torch.argmax(v))
                        out.append([round(float(v.max()), 3), idx // 64, idx % 64])
                    return out
                internals = {"logits": [None if not np.isfinite(x) else round(float(x), 3)
                                        for x in logits[0].tolist()],
                             "F3_channel_max_at": summ(f3), "H_channel_max_at": summ(h),
                             "memory_l2": round(float(mem2.norm()), 3)}
                return frame, internals
            a, r, c = trace[k] if k < len(trace) else ("ACTION1", None, None)
            mem = mem2
            prev = torch.zeros(1, len(ACTIONS))
            if a in ACTIONS:
                prev[0, ACTIONS.index(a)] = 1
                g2.step(a, r, c)
    return frame, None


def _first_problem(spec: EpisodeSpec, res: EpisodeResult) -> tuple[int, str]:
    game = build_start(spec)
    seen = {obs_of(game).astype(np.int8).tobytes(): 0}
    for k, (a, r, c) in enumerate(res.trace):
        if a not in ACTIONS:
            return k, "invalid action"
        _, _, _, changed, _ = game.step(a, r, c)
        if not changed and a != "RESET":
            return k, f"{a} had no effect"
        key = obs_of(game).astype(np.int8).tobytes()
        if key in seen:
            return k + 1, f"returned to the frame of step {seen[key]} (loop)"
        seen[key] = k + 1
    return max(0, len(res.trace) - 1), "ran out of actions"


def diagnose(art: Artifact, specs: list[EpisodeSpec], results: list[EpisodeResult],
             max_failures: int = 3) -> tuple[dict, list[dict]]:
    rows, failures = [], []
    for i, (s, r) in enumerate(zip(specs, results)):
        res = "solved" if r.success else ("game over" if r.game_over else
                                          ("error" if r.error else "not solved"))
        rows.append({"i": i, "seed": s.seed, "level": s.level,
                     "start": "fresh" if not s.prefix else f"+{len(s.prefix)} moves",
                     "result": res, "actions": r.actions})
        if not r.success and len(failures) < max_failures:
            if r.error and not r.trace:
                failures.append({"i": i, "why": "artifact error", "trace": [], "step": 0,
                                 "frame": obs_of(build_start(s)), "error": r.error})
                continue
            step, why = _first_problem(s, r)
            frame, internals = _probe_step(art, s, r.trace, step)
            failures.append({"i": i, "why": why, "trace": [t[0] if t[0] != "ACTION6" else f"ACTION6@{t[1]},{t[2]}"
                                                            for t in r.trace],
                             "step": step, "frame": frame, "internals": internals, "error": r.error})
    return {"n": len(specs), "solved": sum(r.success for r in results), "rows": rows}, failures


# --------------------------------------------------------------------------- session
def run_session(model: str, condition: str, mode: str, prompt: str, support: list[EpisodeSpec],
                out_dir: Path, rounds: int = 3, effort: str | None = None,
                base: dict | None = None, task_budget: int | None = None,
                variant: str = "full", meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt}]
    candidates = []
    n_rounds = 1 + (rounds if mode == "feedback" else 0)
    t0 = time.time()
    for k in range(n_rounds):
        res = writer.call(model, messages, tag=f"round{k}", out_dir=out_dir, effort=effort,
                          task_budget=task_budget)
        cand = {"round": k, "call": res.to_json()}
        if res.status != "completed":
            cand["outcome"] = f"writer_{res.status}"
            candidates.append(cand)
            break
        code = writer.extract_code(res.text)
        art = materialize(condition, code, base=base)
        art.variant = variant
        (out_dir / f"round{k}.py").write_text(code or "")
        if art.params is not None and art.condition == "freearch":
            from . import freearch
            freearch.save_artifact(art, out_dir, k)     # round<k>.npz + round<k>.graph.json
        elif art.params is not None and art.condition == "pbounded":
            from . import pbounded
            pbounded.save_artifact(art, out_dir, k)     # round<k>.npz (constants) + round<k>.pb.json
        elif art.params is not None:
            A.save_npz(art.params, out_dir / f"round{k}.npz")
        results = evaluate(art, support)
        summary, failures = diagnose(art, support, results)
        cand.update({"artifact_error": art.error, "audit": art.audit, "init_wall_s": art.init_wall_s,
                     "support_solved": summary["solved"], "support_n": summary["n"],
                     "outcome": "invalid_artifact" if art.error else "evaluated"})
        candidates.append(cand)
        if k + 1 < n_rounds:
            if summary["solved"] == summary["n"]:
                break                                   # nothing left to fix on support
            if condition == "freearch":
                from .prompts_fa import feedback_message_fa as fb_fn
            elif condition == "pbounded":
                from .prompts_pb import feedback_message_pb as fb_fn
            else:
                fb_fn = feedback_message
            fb = fb_fn(summary, failures, k, n_rounds - k - 1, condition)
            messages += [{"role": "assistant", "content": res.text}, {"role": "user", "content": fb}]
            (out_dir / f"feedback{k}.md").write_text(fb)
    evaluated = [c for c in candidates if c.get("outcome") == "evaluated"]
    final = max(evaluated, key=lambda c: (c["support_solved"], c["round"])) if evaluated else None
    record = {"model": model, "condition": condition, "mode": mode, "rounds_used": len(candidates),
              "effort": effort, "task_budget": task_budget, "variant": variant, **(meta or {}),
              "final_round": None if final is None else final["round"],
              "wall_s": time.time() - t0, "candidates": candidates,
              "tokens_out": sum((c["call"]["usage"] or {}).get("output") or 0 for c in candidates),
              "tokens_in": sum((c["call"]["usage"] or {}).get("input") or 0 for c in candidates)}
    (out_dir / "session.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def load_final(out_dir: Path, condition: str) -> Artifact | None:
    rec = json.loads((out_dir / "session.json").read_text())
    if rec["final_round"] is None:
        return None
    k = rec["final_round"]
    code = (out_dir / f"round{k}.py").read_text()
    if condition == "freearch":
        from . import freearch
        return freearch.load_artifact(out_dir, k, code)
    if condition == "pbounded":
        from . import pbounded
        return pbounded.load_artifact(out_dir, k, code)
    if condition in ("weights", "edit"):
        return Artifact("weights", code, params=A.load_npz(out_dir / f"round{k}.npz"),
                        variant=rec.get("variant", "full"))
    return Artifact(condition, code)

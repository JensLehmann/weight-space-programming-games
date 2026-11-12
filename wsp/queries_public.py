"""Public-tier (official ARC3 games) support/query construction, revision 2.

Same mechanical rules as :mod:`wsp.queries`, adapted to games with only the canonical rendering:
  * held-out levels: ``held_out_levels(n, version=2)`` (the final level is always held out);
  * S-start: the canonical start of every support level;
  * S-dev: 4 certified perturbed starts per support level (feedback/BC only);
  * Q-off: 6 certified perturbed starts per support level, disjoint from S-dev and S-start;
  * Q-lvl: per held-out level 1 reviewed start + 20 certified perturbed starts;
  * no Q-tpl / Q-pal (no templates or palettes exist);
  * action caps ``cap_for(ref_len)``.
Perturbed starts are 1-6 random observation-changing moves from the continuous level entry; a
start is kept only if the prefix neither completed the level nor ended the attempt, and the
planner's direct plan (no RESET) from that exact state replays to completion on an independent
continuous reconstruction with no GAME_OVER, no level restart and no lost life. Attempt ends use
the public detector (:func:`wsp.public.attempt_ended`), not the private families' ``lives`` attribute.

Where a level cannot reach its quota, what certifies is used and the shortfall is recorded. A
game whose held-out levels do not each get at least one certified start is flagged
``excluded_from_macro``. Planner outputs are used only to certify; no plan is stored.
"""
from __future__ import annotations

import json
import random
import signal
import threading
import time
from dataclasses import replace

from . import public as P
from .envs import EpisodeSpec, build_start, cap_for, enter_level
from .manifest import GAMES_COMMIT, sha256_json
from .queries import _effective_moves, _grid_key, evaluate_state, held_out_levels, manifest_path

BASE_SEED = 20261104
REV = 2
SETS = ("S-start", "S-dev", "Q-lvl", "Q-tpl", "Q-off", "Q-pal")
QUOTA = {"S-dev": 4, "Q-off": 6, "Q-lvl": 20}
BUDGET_MS = 5000
HARD_S = 60.0
WALL_PER_START_S = 60.0        # per (set, level) sampling wall cap = this x requested starts


class _Timeout(Exception):
    pass


def _guarded(seconds, fn, *a, **kw):
    """Hard wall guard (SIGALRM; main thread only, otherwise unguarded)."""
    if threading.current_thread() is not threading.main_thread():
        return fn(*a, **kw)

    def _alarm(_sig, _frm):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def certify_public(spec: EpisodeSpec, budget_ms: float = BUDGET_MS,
                   hard_s: float = HARD_S) -> tuple[int | None, str]:
    """(certified direct-plan length or None, outcome) for spec's start."""
    game = build_start(spec)
    try:
        res = _guarded(hard_s, evaluate_state, game, time_budget_ms=budget_ms)
    except _Timeout:
        return None, "planner_hard_timeout"
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return None, f"planner_error:{type(exc).__name__}"
    plan = [(pa.action, pa.row, pa.col) for pa in (res.direct_plan or ())]
    if not plan:
        return None, f"no_direct_plan:{res.status.name}"
    if any(a == "RESET" for a, _, _ in plan):
        return None, "direct_plan_has_reset"
    replay = build_start(spec)
    for i, (a, r, c) in enumerate(plan):
        before = P.attempt_marker(replay)
        _, state, level, _, solved = replay.step(a, r, c)
        if P.attempt_ended(before, replay, a, solved):
            return None, "attempt_end"
        if solved or level > spec.level:
            return len(plan), "completed" if i == len(plan) - 1 else "completed_early"
        if state != "NOT_FINISHED":
            return None, f"state_{state}"
    return None, "plan_exhausted"


def _bump(d: dict, k: str, v: int = 1) -> None:
    d[k] = d.get(k, 0) + v


def perturbed_starts_public(family: str, level: int, n: int, rng: random.Random,
                            exclude: set[bytes], rejected: set[bytes], kmin: int = 1,
                            kmax: int = 6, max_tries: int = 40,
                            wall_s: float | None = None) -> tuple[list[EpisodeSpec], dict]:
    """`wsp.queries.perturbed_starts` with the public attempt detector and a wall cap.

    States whose certification failed are remembered in `rejected` and not re-planned."""
    out: list[EpisodeSpec] = []
    stats = {"tries": 0, "dup": 0, "left_level": 0, "attempt_end": 0, "no_moves": 0,
             "uncertified": 0, "dup_rejected": 0, "uncertified_reasons": {}}
    wall_s = WALL_PER_START_S * n if wall_s is None else wall_s
    t0 = time.perf_counter()
    while len(out) < n and stats["tries"] < max_tries * n:
        if time.perf_counter() - t0 > wall_s:
            stats["wall_cap_hit"] = True
            break
        stats["tries"] += 1
        game = enter_level(family, 0, level)
        k = rng.randint(kmin, kmax)
        prefix, why = [], None
        for _ in range(k):
            moves = _effective_moves(game, rng)
            if not moves:
                break
            move = rng.choice(moves)
            before = P.attempt_marker(game)
            _, state, lvl, _, solved = game.step(*move)
            prefix.append(move)
            if P.attempt_ended(before, game, move[0], solved):
                why = "attempt_end"
                break
            if solved or lvl != level or state != "NOT_FINISHED":
                why = "left_level"
                break
        if why or not prefix:
            _bump(stats, why or "no_moves")
            continue
        key = _grid_key(game)
        if key in exclude:
            stats["dup"] += 1
            continue
        if key in rejected:
            stats["dup_rejected"] += 1
            continue
        spec = EpisodeSpec(family, 0, level, tuple(prefix))
        ref, outcome = certify_public(spec)
        if ref is None:
            stats["uncertified"] += 1
            _bump(stats["uncertified_reasons"], outcome)
            rejected.add(key)
            continue
        exclude.add(key)
        out.append(replace(spec, ref_len=ref, cap=cap_for(ref)))
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    return out, stats


def build_public_sets(family: str, base_seed: int = BASE_SEED,
                      wall_per_start_s: float = WALL_PER_START_S, log=None) -> dict:
    family = P.family_name(family)
    t_all = time.perf_counter()
    n_levels = P.n_levels(family)
    held = held_out_levels(n_levels, version=REV)
    support_levels = tuple(l for l in range(n_levels) if l not in held)
    rng = random.Random(f"{base_seed}:{family}")
    sets: dict[str, list[EpisodeSpec]] = {k: [] for k in SETS}
    stats: dict[str, dict] = {}
    per_level: list[dict] = []
    seen: dict[int, set] = {}
    rejected: dict[int, set] = {}
    problems: list[str] = []

    def say(msg):
        if log:
            log(f"[{family}] {msg}")

    def add_reviewed(qset, level):
        spec = EpisodeSpec(family, 0, level, (), 0, 0, qset)
        t = time.perf_counter()
        ref, outcome = certify_public(spec)
        seen.setdefault(level, set()).add(_grid_key(build_start(spec)))
        row = {"qset": qset, "level": level, "kind": "reviewed", "requested": 1,
               "built": int(ref is not None), "outcome": outcome,
               "wall_s": round(time.perf_counter() - t, 1)}
        per_level.append(row)
        if ref is None:
            problems.append(f"{qset} L{level + 1} reviewed start not certified ({outcome})")
        else:
            sets[qset].append(replace(spec, ref_len=ref, cap=cap_for(ref)))
        say(f"{qset} L{level + 1} reviewed {outcome} ref={ref}")

    def add_perturbed(qset, level, n):
        specs, st = perturbed_starts_public(family, level, n, rng, seen.setdefault(level, set()),
                                            rejected.setdefault(level, set()),
                                            wall_s=wall_per_start_s * n)
        sets[qset].extend(replace(s, qset=qset) for s in specs)
        per_level.append({"qset": qset, "level": level, "kind": "perturbed", "requested": n,
                          "built": len(specs), **st})
        agg = stats.setdefault(qset, {"requested": 0, "built": 0})
        agg["requested"] += n
        agg["built"] += len(specs)
        for k, v in st.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    _bump(agg.setdefault(k, {}), kk, vv)
            elif isinstance(v, bool):
                agg[k] = agg.get(k, 0) + int(v)
            else:
                agg[k] = round(agg.get(k, 0) + v, 1)
        if len(specs) < n:
            problems.append(f"{qset} L{level + 1}: {len(specs)}/{n} perturbed starts certified")
        say(f"{qset} L{level + 1} perturbed {len(specs)}/{n} tries={st['tries']} wall={st['wall_s']}s")

    for l in support_levels:
        add_reviewed("S-start", l)
    for l in support_levels:            # support-side perturbations first (feedback/BC only)
        add_perturbed("S-dev", l, QUOTA["S-dev"])
    for l in support_levels:
        add_perturbed("Q-off", l, QUOTA["Q-off"])
    for l in held:
        add_reviewed("Q-lvl", l)
        add_perturbed("Q-lvl", l, QUOTA["Q-lvl"])

    q_by_level = {l: sum(1 for s in sets["Q-lvl"] if s.level == l) for l in held}
    empty = [l + 1 for l in held if q_by_level[l] == 0]
    excluded = bool(empty)
    return {
        "family": family,
        "tier": "public",
        "game_id": P.game_id(family),
        "split_version": REV,
        "master_seed": base_seed,
        "n_levels": n_levels,
        "held_out_levels": list(held),
        "support_levels": list(support_levels),
        "support_templates": [0],
        "query_templates": [],
        "palette_distinct": False,
        "sets_not_applicable": ["Q-tpl", "Q-pal"],
        "q_lvl_per_level": {str(l): q_by_level[l] for l in held},
        "excluded_from_macro": excluded,
        "exclusion_reason": (f"held-out level(s) {empty} have no certified start" if excluded else None),
        "construction_problems": problems,
        "construction_stats": stats,
        "construction_per_level": per_level,
        "construction_wall_s": round(time.perf_counter() - t_all, 1),
        "rules": {"quota": QUOTA, "perturb_k": [1, 6], "max_tries_per_start": 40,
                  "wall_cap_per_set_level_s_per_start": wall_per_start_s,
                  "planner_budget_ms": BUDGET_MS, "planner_hard_timeout_s": HARD_S,
                  "cap": "3*ref_len+10"},
        "env": {"games_commit": GAMES_COMMIT, "source_digest": P.source_root().name,
                "source_sha256": P.source_sha256(family), "route_sha256": P.route_sha256(family),
                "lives_exprs": list(P.lives_exprs(family)), "reset_semantics": P.RESET_SEMANTICS},
        "sets": {k: [s.to_json() for s in v] for k, v in sets.items()},
    }


def write_public_sets(family: str, log=None, **kw) -> dict:
    data = build_public_sets(family, log=log, **kw)
    data["sha256"] = sha256_json(data["sets"])
    path = manifest_path(data["family"], REV)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    return data

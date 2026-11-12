"""Support/query construction from reviewed content only (no level generators).

Mechanical rules, fixed before any writer call:
  * templates: seeds 0-4 support, 5-9 query (template = seed % 10)
  * held-out levels (split version 2): levels n, n-2, ... >= 2
  * perturbed starts: random effective-action prefixes from the continuous level entry,
    kept only if the family planner returns a direct plan that replays to completion
    without ending the attempt.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import replace

import numpy as np

from .envs import EpisodeSpec, build_start, cap_for, enter_level, new_game, obs_of
from .manifest import MANIFESTS, games_package, sha256_json

SUPPORT_TEMPLATES = (0, 1, 2, 3, 4)
QUERY_TEMPLATES = (5, 6, 7, 8, 9)
PALETTE_SEEDS = tuple(range(10, 20))


SPLIT_VERSION = 2
PRIVATE_MANIFESTS_ENV = "WSP_PRIVATE_MANIFESTS"   # directory with the private families' query manifests


def evaluate_state(game, **kw):
    """Planner evaluation of a game state (game-collection package, imported on first use)."""
    games_package()
    from arc3_games.api import evaluate_state as _evaluate_state
    return _evaluate_state(game, **kw)


def attempt_marker(game):
    games_package()
    from arc3_games.games.base import attempt_marker as _attempt_marker
    return _attempt_marker(game)


def attempt_ended(before, game) -> bool:
    games_package()
    from arc3_games.games.base import attempt_ended as _attempt_ended
    return _attempt_ended(before, game)


def _check_version(version: int | None) -> int:
    version = SPLIT_VERSION if version is None else version
    if version != SPLIT_VERSION:
        raise ValueError(f"unknown split version {version} (the study uses version {SPLIT_VERSION})")
    return version


def held_out_levels(n: int, version: int | None = None) -> tuple[int, ...]:
    """0-based held-out level indices: 1-based levels n, n-2, n-4, ... down to level 2 (level 1
    always seen, the final level always held out), i.e. ~50% of levels (split version 2)."""
    _check_version(version)
    return tuple(sorted(l - 1 for l in range(n, 1, -2)))


def certify(spec: EpisodeSpec, budget_ms: float = 5000) -> int | None:
    """Return the certified direct-plan length from spec's start, or None."""
    game = build_start(spec)
    if hasattr(game, "direct_plan"):                       # instrumentation toy game
        plan = [(a, None, None) for a in (game.direct_plan() or [])]
    else:
        res = evaluate_state(game, time_budget_ms=budget_ms)
        plan = [(pa.action, pa.row, pa.col) for pa in (res.direct_plan or ())]
    if not plan:
        return None
    replay = build_start(spec)
    for a, r, c in plan:
        before = attempt_marker(replay)
        _, state, level, _, solved = replay.step(a, r, c)
        if attempt_ended(before, replay):
            return None
        if solved or level > spec.level:
            return len(plan)
        if state != "NOT_FINISHED":
            return None
    return None


def _grid_key(game) -> bytes:
    return obs_of(game).astype(np.int8).tobytes()


def _effective_moves(game, rng: random.Random, max_click_probe: int = 48):
    """Moves that change the observation (sampling aid only; final specs are replay-verified)."""
    moves = []
    base = _grid_key(game)
    for a in game.available_actions:
        if a == "RESET":
            continue
        if a == "ACTION6":
            grid = obs_of(game)
            bg = np.bincount(grid.ravel()).argmax()
            cells = list(zip(*np.nonzero(grid != bg)))
            rng.shuffle(cells)
            for r, c in cells[:max_click_probe]:
                g2 = copy.deepcopy(game)
                g2.step("ACTION6", int(r), int(c))
                if _grid_key(g2) != base:
                    moves.append(("ACTION6", int(r), int(c)))
        else:
            g2 = copy.deepcopy(game)
            g2.step(a, None, None)
            if _grid_key(g2) != base:
                moves.append((a, None, None))
    return moves


def perturbed_starts(family: str, seed: int, level: int, n: int, rng: random.Random,
                     exclude: set[bytes], kmin: int = 1, kmax: int = 6,
                     max_tries: int = 40) -> tuple[list[EpisodeSpec], dict]:
    out: list[EpisodeSpec] = []
    stats = {"tries": 0, "dup": 0, "left_level": 0, "uncertified": 0}
    while len(out) < n and stats["tries"] < max_tries * n:
        stats["tries"] += 1
        game = enter_level(family, seed, level)
        k = rng.randint(kmin, kmax)
        prefix, ok = [], True
        for _ in range(k):
            moves = _effective_moves(game, rng)
            if not moves:
                break
            move = rng.choice(moves)
            before = attempt_marker(game)
            _, state, lvl, _, solved = game.step(*move)
            prefix.append(move)
            if solved or lvl != level or state != "NOT_FINISHED" or attempt_ended(before, game):
                ok = False
                break
        if not ok or not prefix:
            stats["left_level"] += 1
            continue
        key = _grid_key(game)
        if key in exclude:
            stats["dup"] += 1
            continue
        spec = EpisodeSpec(family, seed, level, tuple(prefix))
        ref = certify(spec)
        if ref is None:
            stats["uncertified"] += 1
            continue
        exclude.add(key)
        out.append(replace(spec, ref_len=ref, cap=cap_for(ref)))
    return out, stats


def reviewed_start(family: str, seed: int, level: int, qset: str) -> EpisodeSpec:
    spec = EpisodeSpec(family, seed, level, (), 0, 0, qset)
    ref = certify(spec)
    if ref is None:
        raise RuntimeError(f"reviewed start not certified: {spec}")
    return replace(spec, ref_len=ref, cap=cap_for(ref))


def manifest_path(family: str, rev: int | None = None):
    """Query manifests live in manifests/rev2/.

    This release ships only the public tier's manifests. The private families' manifests are
    looked up under ``$WSP_PRIVATE_MANIFESTS`` (same layout) when they are not in manifests/."""
    import os
    from pathlib import Path
    rev = _check_version(rev)
    sub = Path(f"rev{rev}")
    path = MANIFESTS / sub / f"queries_{family}.json"
    private = os.environ.get(PRIVATE_MANIFESTS_ENV)
    if not path.exists() and private and (Path(private) / sub / path.name).exists():
        return Path(private) / sub / path.name
    return path


def build_family_sets(family: str, rev: int | None = None) -> dict:
    rev = _check_version(rev)
    base_seed = 20261104
    n_levels = new_game(family, 0).n_levels
    held = held_out_levels(n_levels, version=rev)
    support_levels = tuple(l for l in range(n_levels) if l not in held)
    rng = random.Random(f"{base_seed}:{family}")
    palette_distinct = _grid_key(new_game(family, 10)) != _grid_key(new_game(family, 0))
    sets: dict[str, list[EpisodeSpec]] = {k: [] for k in
                                          ("S-start", "S-dev", "Q-lvl", "Q-tpl", "Q-off", "Q-pal")}
    stats: dict[str, dict] = {}
    seen: dict[tuple, set] = {}

    def excl(seed, level):
        return seen.setdefault((seed, level), set())

    def add_reviewed(qset, seed, level):
        s = reviewed_start(family, seed, level, qset)
        excl(seed, level).add(_grid_key(build_start(s)))
        sets[qset].append(s)

    def add_perturbed(qset, seed, level, n):
        specs, st = perturbed_starts(family, seed, level, n, rng, excl(seed, level))
        sets[qset].extend(replace(s, qset=qset) for s in specs)
        agg = stats.setdefault(qset, {"requested": 0, "built": 0})
        agg["requested"] += n
        agg["built"] += len(specs)
        for k, v in st.items():
            agg[k] = agg.get(k, 0) + v

    for t in SUPPORT_TEMPLATES:
        for l in support_levels:
            add_reviewed("S-start", t, l)
    for t in SUPPORT_TEMPLATES:          # support-side perturbations first (feedback/BC only)
        for l in support_levels:
            add_perturbed("S-dev", t, l, 4)
    for t in SUPPORT_TEMPLATES:
        for l in support_levels:
            add_perturbed("Q-off", t, l, 6)
    for t in SUPPORT_TEMPLATES + QUERY_TEMPLATES:
        for l in held:
            add_reviewed("Q-lvl", t, l)
            add_perturbed("Q-lvl", t, l, 5)
    for t in QUERY_TEMPLATES:
        for l in support_levels:
            add_reviewed("Q-tpl", t, l)
            add_perturbed("Q-tpl", t, l, 2)
    if palette_distinct:
        for s in PALETTE_SEEDS:
            for l in held:
                add_reviewed("Q-pal", s, l)
    return {
        "family": family,
        "split_version": rev,
        "n_levels": n_levels,
        "held_out_levels": list(held),
        "support_levels": list(support_levels),
        "support_templates": list(SUPPORT_TEMPLATES),
        "query_templates": list(QUERY_TEMPLATES),
        "palette_distinct": palette_distinct,
        "construction_stats": stats,
        "sets": {k: [s.to_json() for s in v] for k, v in sets.items()},
    }


def write_family_sets(family: str, rev: int | None = None) -> dict:
    data = build_family_sets(family, rev=rev)
    data["sha256"] = sha256_json(data["sets"])
    path = manifest_path(family, rev)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    return data


def manifest_meta(family: str, rev: int | None = None) -> dict:
    data = json.loads(manifest_path(family, rev).read_text())
    return {k: v for k, v in data.items() if k != "sets"}


def load_sets(family: str, rev: int | None = None) -> dict[str, list[EpisodeSpec]]:
    data = json.loads(manifest_path(family, rev).read_text())
    assert sha256_json(data["sets"]) == data["sha256"], f"query manifest for {family} modified"
    return {k: [EpisodeSpec.from_json(d) for d in v] for k, v in data["sets"].items()}

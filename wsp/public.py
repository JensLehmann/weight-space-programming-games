"""Public tier: the 25 official ARC3 games.

Families are named ``official.<id>`` (e.g. ``official.ls20``). Only the canonical rendering
exists (seed 0). The official game files (``<root>/<id>/<version>/<id>.py``) are read from
``$WSP_OFFICIAL_ROOT`` if set, else from the pinned, content-addressed cache of the game-collection
package (``runs/private/arc3_cache``; ``ARC3_GAMES_CACHE_DIR`` overrides it), and are always passed
to ``make_game`` as explicit roots. Without the package, games are built by the release adapter
:mod:`wsp.public_games` from the game file and ``arcengine`` (``$WSP_OFFICIAL_ROOT`` is then needed).

Level entry is continuous play of the frozen winning route recorded by the public-game survey
(``manifests/public/official_routes.json``; ``WSP_PUBLIC_SURVEY`` overrides the directory), never
``engine.set_level``.

Attempt ends, mirroring ``arc3_games.games.base.attempt_ended``:
  * GAME_OVER;
  * an internal level restart: an executed non-RESET action that neither changes level nor
    completes it, but does not advance ``engine._action_count`` by exactly 1;
  * a lost life: a decrease of a per-game lives expression from ``lives_map.json``
    (``e.<attr>`` or ``-e.<attr>`` over the engine ``e``; currently ls20 and r11l). sp80's
    failed-spill counter is legal by convention and is not a life.

RESET semantics: the harness runs official engines with ``ONLY_RESET_LEVELS=true`` (an
arcengine switch the public planners model). RESET then restarts the current level, as in the
private games; without it a RESET with no action taken since the level start restarts the whole
campaign at level 1, which would take a per-level episode out of its level. Observations along
all 25 frozen routes are identical in both modes (checked before adoption).
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from .manifest import MANIFESTS, RUNS

PREFIX = "official."
SURVEY = Path(os.environ.get("WSP_PUBLIC_SURVEY") or MANIFESTS / "public")
ROUTES_PATH = SURVEY / "official_routes.json"
LIVES_PATH = SURVEY / "lives_map.json"
DEFAULT_CACHE = RUNS / "private" / "arc3_cache"
RESET_ENV = ("ONLY_RESET_LEVELS", "true")
RESET_SEMANTICS = "level"          # RESET restarts the current level (ONLY_RESET_LEVELS=true)

os.environ[RESET_ENV[0]] = RESET_ENV[1]


def is_public(family: str) -> bool:
    return family.startswith(PREFIX)


def family_name(gid: str) -> str:
    return gid if gid.startswith(PREFIX) else PREFIX + gid


def game_id(family: str) -> str:
    gid = family[len(PREFIX):] if family.startswith(PREFIX) else family
    if gid not in _routes():
        raise KeyError(f"unknown public game: {family}")
    return gid


def cache_dir() -> Path:
    return Path(os.environ.get("ARC3_GAMES_CACHE_DIR") or DEFAULT_CACHE)


@lru_cache(maxsize=1)
def source_root() -> Path:
    if os.environ.get("WSP_OFFICIAL_ROOT"):
        return Path(os.environ["WSP_OFFICIAL_ROOT"]).expanduser().resolve()
    try:
        from arc3_games.sources import source_root as _root
    except ImportError as exc:
        raise FileNotFoundError("set WSP_OFFICIAL_ROOT to a directory with the official game files "
                                "<id>/<version>/<id>.py (see README)") from exc
    return Path(_root("official", cache_dir=cache_dir()))


@lru_cache(maxsize=1)
def _routes() -> dict:
    return json.loads(ROUTES_PATH.read_text())


@lru_cache(maxsize=1)
def _lives_map() -> dict:
    return json.loads(LIVES_PATH.read_text())


def public_ids() -> tuple[str, ...]:
    """The 25 official game ids (sorted)."""
    return tuple(sorted(_routes()))


def public_families() -> tuple[str, ...]:
    return tuple(PREFIX + g for g in public_ids())


def n_levels(family: str) -> int:
    return int(_routes()[game_id(family)]["n_levels"])


def levels_table() -> dict[str, int]:
    """{official.<id>: n_levels} for all 25 games."""
    return {PREFIX + g: int(r["n_levels"]) for g, r in sorted(_routes().items())}


def source_path(family: str) -> Path:
    gid = game_id(family)
    hits = sorted((source_root() / gid).glob(f"*/{gid}.py"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one pinned source for {gid}, found {hits}")
    return hits[0]


@lru_cache(maxsize=None)
def source_sha256(family: str) -> str:
    return hashlib.sha256(source_path(family).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def route(family: str) -> tuple[tuple[tuple, ...], ...]:
    """Frozen per-level winning route (level-start direct plans, concatenated by level)."""
    gid = game_id(family)
    rec = _routes()[gid]
    if rec["source_sha256"] != source_sha256(family):
        raise ValueError(f"frozen route for {gid} was recorded on a different source")
    return tuple(tuple((a, r, c) for a, r, c in lv) for lv in rec["levels"])


def route_sha256(family: str) -> str:
    rec = _routes()[game_id(family)]
    return hashlib.sha256(json.dumps(rec["levels"], separators=(",", ":")).encode()).hexdigest()


def new_game_public(family: str, seed: int = 0):
    if seed != 0:
        raise ValueError(f"{family}: official games have only the canonical seed 0")
    assert os.environ.get(RESET_ENV[0]) == RESET_ENV[1], "level-only RESET must stay enabled"
    gid = game_id(family)
    try:
        from arc3_games.api import make_game
    except ImportError:                                # release adapter (official game file + arcengine)
        from .public_games import OfficialGame
        return OfficialGame(gid, source_path(family), _routes()[gid]["source_sha256"])
    return make_game(family_name(gid), seed=0, roots={"official": source_root()})


def enter_level_public(family: str, level: int, seed: int = 0):
    """Reach `level` (0-based) by continuous play of the frozen route (no set_level, no planner)."""
    game = new_game_public(family, seed)
    levels = route(family)
    if not 0 <= level < len(levels):
        raise ValueError(f"{family} has no level {level}")
    for lv in levels[:level]:
        for a, r, c in lv:
            game.step(a, r, c)
    if game.level != level or game.state != "NOT_FINISHED":
        raise RuntimeError(f"could not enter level {level} of {family}: at {game.level}/{game.state}")
    return game


# ------------------------------------------------------------------------- attempt detection
def _parse_expr(expr: str) -> tuple[int, tuple[str, ...]]:
    sign, body = (-1, expr[1:]) if expr.startswith("-") else (1, expr)
    parts = body.split(".")
    if parts[0] != "e" or len(parts) < 2 or not all(p.isidentifier() for p in parts[1:]):
        raise ValueError(f"unsupported lives expression {expr!r}")
    return sign, tuple(parts[1:])


@lru_cache(maxsize=None)
def lives_exprs(family: str) -> tuple[str, ...]:
    return tuple(_lives_map().get(game_id(family), ()))


@lru_cache(maxsize=None)
def _lives_getters(family: str) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple(_parse_expr(x) for x in lives_exprs(family))


def lives(game) -> tuple[int | None, ...]:
    """Signed lives values (higher = more lives left) for the game's lives expressions."""
    out = []
    for sign, path in _lives_getters(game.family):
        v = game.engine
        try:
            for p in path:
                v = getattr(v, p)
            out.append(sign * int(v))
        except (AttributeError, TypeError, ValueError):
            out.append(None)
    return tuple(out)


def attempt_marker(game) -> tuple:
    """Everything :func:`attempt_ended` compares; take it before every action."""
    return (game.level, game.state, int(game.engine._action_count), lives(game),
            tuple(game.available_actions))


def attempt_ended(before: tuple, game, action: str, solved: bool) -> bool:
    """Did the step from `before` (an :func:`attempt_marker`) with `action` end the attempt?"""
    if game.state == "GAME_OVER":
        return True
    level_b, state_b, count_b, lives_b, avail_b = before
    if game.level != level_b or solved or action == "RESET":
        return False
    if action not in avail_b or state_b in ("WIN", "GAME_OVER"):
        return False                                   # not executed: inert no-op
    if int(game.engine._action_count) != count_b + 1:
        return True                                    # internal level restart
    return any(b is not None and a is not None and a < b for b, a in zip(lives_b, lives(game)))

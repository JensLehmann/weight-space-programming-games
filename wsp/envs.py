"""Observation-only episodes over pinned ARC3 games.

A policy only ever receives (observation grid, available-action mask). The live game object,
planner, and privileged snapshots stay inside this module and the evaluator.

Families named ``official.<id>`` (the public tier) are routed to :mod:`wsp.public` for game
construction, level entry and attempt detection; every other family takes the private-family path.
Both paths construct games through the game-collection package ``arc3_games`` (not part of this
release; see README), which is imported only when a game is built; without it, the public path
builds the official games with :mod:`wsp.public_games`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from .manifest import games_package

ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")
A_INDEX = {a: i for i, a in enumerate(ACTIONS)}
CLICK = A_INDEX["ACTION6"]
Move = tuple  # (action: str, row: int | None, col: int | None)
PUBLIC_PREFIX = "official."


@dataclass(frozen=True)
class EpisodeSpec:
    family: str
    seed: int
    level: int
    prefix: tuple = ()          # ((action, row, col), ...) applied after continuous level entry
    ref_len: int = 0            # certified direct-plan length from the start state
    cap: int = 0                # action cap for the episode
    qset: str = ""              # query/support set label

    def to_json(self) -> dict:
        d = asdict(self)
        d["prefix"] = [list(m) for m in self.prefix]
        return d

    @staticmethod
    def from_json(d: dict) -> "EpisodeSpec":
        return EpisodeSpec(d["family"], d["seed"], d["level"],
                           tuple(tuple(m) for m in d["prefix"]), d["ref_len"], d["cap"], d["qset"])

    @property
    def key(self) -> str:
        core = [self.family, self.seed, self.level, [list(m) for m in self.prefix]]
        return hashlib.sha256(json.dumps(core).encode()).hexdigest()[:16]


def cap_for(ref_len: int) -> int:
    return 3 * ref_len + 10


def new_game(family: str, seed: int):
    if family == "toy_reach":
        from .toy import ToyReach
        return ToyReach(seed)
    if family.startswith(PUBLIC_PREFIX):
        from .public import new_game_public
        return new_game_public(family, seed)
    games_package()
    from arc3_games.api import make_game
    return make_game(f"authored.{family}", seed=seed)


def enter_level(family: str, seed: int, level: int):
    """Reach `level` by continuous play of the family's own winning witness (no debug init)."""
    if family.startswith(PUBLIC_PREFIX):
        from .public import enter_level_public
        return enter_level_public(family, level, seed)
    game = new_game(family, seed)
    witness = game.solution_actions()
    i = 0
    while game.level < level:
        if i >= len(witness):
            raise RuntimeError(f"witness exhausted before level {level} ({family}, seed {seed})")
        a, r, c = witness[i]
        game.step(a, r, c)
        i += 1
    if game.level != level or game.state != "NOT_FINISHED":
        raise RuntimeError(f"could not enter level {level} of {family} seed {seed}")
    return game


def build_start(spec: EpisodeSpec):
    game = enter_level(spec.family, spec.seed, spec.level)
    for a, r, c in spec.prefix:
        game.step(a, r, c)
    if game.level != spec.level or game.state != "NOT_FINISHED":
        raise RuntimeError(f"prefix left level {spec.level}: {spec}")
    return game


def _private_attempt_ended(before, game, action, solved) -> bool:
    from arc3_games.games.base import attempt_ended
    return attempt_ended(before, game)


def attempt_detector(family: str):
    """(marker, ended) for counting ended attempts; ``ended(before, game, action, solved)``."""
    if family.startswith(PUBLIC_PREFIX):
        from .public import attempt_ended as public_ended, attempt_marker as public_marker
        return public_marker, public_ended
    games_package()
    from arc3_games.games.base import attempt_marker
    return attempt_marker, _private_attempt_ended


def obs_of(game) -> np.ndarray:
    return np.asarray(game.observe(), dtype=np.int64)


def avail_mask(game) -> np.ndarray:
    m = np.zeros(len(ACTIONS), dtype=bool)
    for a in game.available_actions:
        m[A_INDEX[a]] = True
    return m


class Policy(Protocol):
    """Batched policy. `begin` marks new episodes (memory reset); `act` returns one move per slot."""

    def begin(self, slots: Sequence[int]) -> None: ...
    def act(self, slots: Sequence[int], obs: np.ndarray, avail: np.ndarray) -> list[Move]: ...


@dataclass
class EpisodeResult:
    key: str
    qset: str
    family: str
    seed: int
    level: int
    success: bool = False
    actions: int = 0
    resets: int = 0
    invalid: int = 0
    noops: int = 0
    attempts_lost: int = 0
    game_over: bool = False
    error: str | None = None
    trace: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def _valid(move, avail: np.ndarray) -> bool:
    if not isinstance(move, (tuple, list)) or len(move) != 3:
        return False
    a, r, c = move
    if a not in A_INDEX or not avail[A_INDEX[a]]:
        return False
    if a == "ACTION6":
        return isinstance(r, (int, np.integer)) and isinstance(c, (int, np.integer)) \
            and 0 <= r < 64 and 0 <= c < 64
    return True


def run_episodes(specs: Sequence[EpisodeSpec], policy: Policy, batch: int = 64,
                 keep_trace: bool = True) -> list[EpisodeResult]:
    """Run episodes in lockstep batches. One decision = one charged step of the cap."""
    results: list[EpisodeResult] = []
    for start in range(0, len(specs), batch):
        chunk = list(specs[start:start + batch])
        games = [build_start(s) for s in chunk]
        res = [EpisodeResult(s.key, s.qset, s.family, s.seed, s.level) for s in chunk]
        detect = [attempt_detector(s.family) for s in chunk]
        active = list(range(len(chunk)))
        policy.begin(active)
        while active:
            obs = np.stack([obs_of(games[i]) for i in active])
            avail = np.stack([avail_mask(games[i]) for i in active])
            try:
                moves = policy.act(active, obs, avail)
            except Exception as exc:  # policy crash ends the affected batch as failures
                for i in active:
                    res[i].error = f"{type(exc).__name__}: {exc}"[:500]
                break
            still = []
            for j, i in enumerate(active):
                g, s, r = games[i], chunk[i], res[i]
                move = moves[j]
                r.actions += 1
                if not _valid(move, avail[j]):
                    r.invalid += 1
                    if keep_trace:
                        r.trace.append(["INVALID", None, None])
                else:
                    a, row, col = move[0], move[1], move[2]
                    row = None if row is None else int(row)
                    col = None if col is None else int(col)
                    marker, ended = detect[i]
                    before = marker(g)
                    _, state, level, changed, solved = g.step(a, row, col)
                    if keep_trace:
                        r.trace.append([a, row, col])
                    if a == "RESET":
                        r.resets += 1
                    elif not changed:
                        r.noops += 1
                    if ended(before, g, a, solved):
                        r.attempts_lost += 1
                    if solved or level > s.level:
                        r.success = True
                    if state == "GAME_OVER":
                        r.game_over = True
                if r.success or r.game_over or r.actions >= s.cap:
                    continue
                still.append(i)
            active = still
        results.extend(res)
    return results


def summarize(results: Iterable[EpisodeResult]) -> dict:
    rs = list(results)
    n = len(rs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "solve_rate": sum(r.success for r in rs) / n,
        "errors": sum(r.error is not None for r in rs),
        "mean_actions": float(np.mean([r.actions for r in rs])),
        "invalid_rate": sum(r.invalid for r in rs) / max(1, sum(r.actions for r in rs)),
        "noop_rate": sum(r.noops for r in rs) / max(1, sum(r.actions for r in rs)),
        "resets": sum(r.resets for r in rs),
        "game_over": sum(r.game_over for r in rs),
    }

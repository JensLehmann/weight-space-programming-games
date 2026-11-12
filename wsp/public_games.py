"""Official ARC-AGI-3 games for the public tier, without the game-collection package.

Evaluation needs only a small, policy-visible view of an official game, the one :mod:`wsp.envs`
and :mod:`wsp.public` use: ``step(action, row, col) -> (grid, state, level, changed, solved)``,
``observe()``, ``available_actions``, ``level``, ``state``, ``n_levels``, and ``engine`` (the
``arcengine`` game, read by the attempt detector). :class:`OfficialGame` provides exactly that
from an official game file (``<root>/<id>/<version>/<id>.py`` as published by the ARC Prize
Foundation) and ``arcengine``. It follows the public-game environment of the game-collection
package used in the study: the same action encoding (``ACTION6`` clicks as ``x = col``,
``y = row``), the frames of the last action as the observation, level completion as a level or
score increase, and unavailable actions or actions after the end of the game as inert no-ops.
No planner is involved; the query manifests already contain the certified starts.

The game file is checked against its SHA-256 (``source_sha256`` of the frozen routes) before it
is executed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=None)
def game_type(path: str, sha256: str) -> type:
    """The one ``ARCBaseGame`` subclass defined by the official game file at `path`."""
    from arcengine import ARCBaseGame
    p = Path(path)
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{p}: SHA-256 differs from the frozen routes (another version of the game?)")
    name = f"_wsp_official_{p.stem}_{sha256[:16]}"
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, p))
    sys.modules[name] = module                     # dataclasses and typing look the module up by name
    try:
        exec(compile(data, str(p), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    classes = [v for v in vars(module).values() if isinstance(v, type) and issubclass(v, ARCBaseGame)
               and v is not ARCBaseGame and v.__module__ == name]
    if len(classes) != 1:
        raise ImportError(f"expected one ARCBaseGame subclass in {p}, found {classes}")
    return classes[0]


def _check_move(action: str, row, col) -> None:
    from arcengine import GameAction
    if action not in GameAction.__members__:
        raise ValueError(f"unknown ARC action {action!r}")
    if action != "ACTION6":
        if row is not None or col is not None:
            raise ValueError(f"{action} does not accept click coordinates")
    elif any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 64 for v in (row, col)):
        raise ValueError("click coordinates must be integers in [0, 63]")


class OfficialGame:
    """An official game at its canonical (only) variant, seed 0."""

    def __init__(self, gid: str, path: Path, sha256: str):
        self.family = gid
        self.seed = 0
        self.engine = game_type(str(Path(path).resolve()), sha256)()
        self._frames = [self.engine.camera.render(self.engine.current_level.get_sprites()).tolist()]
        self.step("RESET")

    @property
    def level(self) -> int:
        return int(self.engine._current_level_index)

    @property
    def state(self) -> str:
        return self.engine._state.name

    @property
    def n_levels(self) -> int:
        return len(self.engine._levels)

    @property
    def available_actions(self) -> list[str]:
        from arcengine import GameAction
        return list(dict.fromkeys(["RESET", *[GameAction.from_id(int(a)).name
                                               for a in self.engine._available_actions]]))

    def observe(self) -> list:
        return copy.deepcopy(self._frames[-1])

    def step(self, action: str, row: int | None = None, col: int | None = None):
        from arcengine import ActionInput, GameAction
        _check_move(action, row, col)
        before = self.observe()
        self._frames = [before]
        if action not in self.available_actions or (self.state in ("WIN", "GAME_OVER") and action != "RESET"):
            return before, self.state, self.level, False, False
        level, score = self.level, int(getattr(self.engine, "_score", 0))
        data = {"x": col, "y": row} if action == "ACTION6" else {}
        result = self.engine.perform_action(ActionInput(id=GameAction[action], data=data), raw=True)
        frames = [np.asarray(f).astype(int).tolist() for f in result.frame]
        if frames:
            self._frames = frames
        completed = self.level > level or int(getattr(self.engine, "_score", 0)) > score
        obs = self.observe()
        return obs, self.state, self.level, before != obs, completed and action != "RESET"

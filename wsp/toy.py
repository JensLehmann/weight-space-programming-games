"""`toy_reach`: instrumentation-only game (not from any game collection).

A 2x2 player (colour 9) moves 2 px per action (ACTION1 up, ACTION2 down, ACTION3 left,
ACTION4 right) on a black field with a 1-px grey border (colour 5). The level is solved when the
player lies inside the hollow 6x6 target frame (colour 4), i.e. its top-left corner equals the
frame's top-left + (2, 2). Walls block. Seed selects layouts; level k has k+1 interior walls.
"""
from __future__ import annotations

import random

import numpy as np

SOURCE_DESCRIPTION = __doc__
BG, BORDER, PLAYER, TARGET, WALL = 0, 5, 9, 4, 8
STEP = 2
DIRS = {"ACTION1": (-STEP, 0), "ACTION2": (STEP, 0), "ACTION3": (0, -STEP), "ACTION4": (0, STEP)}


class ToyReach:
    family = "toy_reach"
    n_levels = 4
    available_actions = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    def __init__(self, seed: int):
        self.seed = seed
        self.level = 0
        self.state = "NOT_FINISHED"
        self._init_level(0)

    def _init_level(self, level: int) -> None:
        rng = random.Random(self.seed * 1000 + level)
        while True:
            walls = set()
            for _ in range(level + 1):
                r, c = rng.randrange(4, 56, 2), rng.randrange(4, 56, 2)
                horiz = rng.random() < 0.5
                for k in range(0, rng.randrange(8, 24, 2), 2):
                    walls.add((r, c + k) if horiz else (r + k, c))
            self.walls = {(r, c) for r, c in walls if 2 <= r < 60 and 2 <= c < 60}
            self.pos = (rng.randrange(2, 58, 2), rng.randrange(2, 58, 2))
            t = (rng.randrange(2, 54, 2), rng.randrange(2, 54, 2))
            self.goal = (t[0] + 2, t[1] + 2)
            if self._free(self.pos) and self._free(self.goal) and self.pos != self.goal \
                    and self._bfs(self.pos) is not None:
                break
        self._render()

    def _free(self, p) -> bool:
        r, c = p
        return 2 <= r <= 60 and 2 <= c <= 60 and (r, c) not in self.walls

    def _bfs(self, start):
        from collections import deque
        prev = {start: None}
        q = deque([start])
        while q:
            p = q.popleft()
            if p == self.goal:
                path = []
                while prev[p] is not None:
                    p, a = prev[p]
                    path.append(a)
                return path[::-1]
            for a, (dr, dc) in DIRS.items():
                n = (p[0] + dr, p[1] + dc)
                if n not in prev and self._free(n):
                    prev[n] = (p, a)
                    q.append(n)
        return None

    def _render(self) -> None:
        g = np.full((64, 64), BG, dtype=int)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = BORDER
        for r, c in self.walls:
            g[r:r + 2, c:c + 2] = WALL
        tr, tc = self.goal[0] - 2, self.goal[1] - 2
        g[tr, tc:tc + 6] = g[tr + 5, tc:tc + 6] = TARGET
        g[tr:tr + 6, tc] = g[tr:tr + 6, tc + 5] = TARGET
        r, c = self.pos
        g[r:r + 2, c:c + 2] = PLAYER
        self.grid = g

    def observe(self):
        return self.grid.tolist()

    def step(self, action, row=None, col=None):
        if action == "RESET":
            self._init_level(self.level)
            return self.observe(), self.state, self.level, True, False
        if self.state != "NOT_FINISHED":
            return self.observe(), self.state, self.level, False, False
        dr, dc = DIRS.get(action, (0, 0))
        n = (self.pos[0] + dr, self.pos[1] + dc)
        changed = False
        if (dr or dc) and self._free(n):
            self.pos, changed = n, True
        solved = self.pos == self.goal
        if solved:
            if self.level + 1 >= self.n_levels:
                self.state = "WIN"
            else:
                self.level += 1
                self._init_level(self.level)
        else:
            self._render()
        return self.observe(), self.state, self.level, changed, solved

    def direct_plan(self):
        return self._bfs(self.pos)

    def solution_actions(self):
        sim = ToyReach(self.seed)
        out = []
        while sim.state == "NOT_FINISHED":
            for a in sim.direct_plan():
                out.append((a, None, None))
                sim.step(a)
        return out

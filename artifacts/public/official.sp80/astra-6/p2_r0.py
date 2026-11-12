import numpy as np
import math
import time
import random
import heapq
from collections import deque


def _components(grid, colors):
    h, w = grid.shape
    remaining = {
        (x, y) for y in range(h) for x in range(w)
        if int(grid[y, x]) in colors
    }
    result = []
    while remaining:
        start = min(remaining, key=lambda p: (p[1], p[0]))
        remaining.remove(start)
        stack = [start]
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(comp)
    return result


def _read_image(obs):
    a = np.asarray(obs, dtype=np.int8)
    edges = (a[0, :], a[:, 0], a[-1, :], a[:, -1])
    k = 0
    for j, edge in enumerate(edges):
        if np.all((edge == 0) | (edge == 14)):
            k = j
            break
    a = np.rot90(a, -k).copy()

    # Find the largest exact pixel replication factor. Padding may be
    # retained as extra wall cells; this does not change click coordinates.
    for scale in range(8, 0, -1):
        lim = 64 // scale
        good_w = []
        good_h = []
        for n in range(1, lim + 1):
            off = (64 - n * scale) // 2
            centers = off + np.arange(n) * scale + scale // 2

            r = np.ones_like(a)
            r[:, off:off + n * scale] = np.repeat(
                a[:, centers], scale, axis=1
            )
            if np.array_equal(r[1:], a[1:]):
                good_w.append(n)

            r = np.ones_like(a)
            r[off:off + n * scale, :] = np.repeat(
                a[centers, :], scale, axis=0
            )
            if np.array_equal(r[1:], a[1:]):
                good_h.append(n)

        if good_w and good_h:
            w, h = max(good_w), max(good_h)
            ox = (64 - w * scale) // 2
            oy = (64 - h * scale) // 2
            xs = ox + np.arange(w) * scale + scale // 2
            ys = oy + np.arange(h) * scale + scale // 2
            grid = a[np.ix_(ys, xs)].copy()
            return grid, k, scale, ox, oy

    return a, k, 1, 0, 0


def _make_pieces(grid, remembered):
    visible = {
        (x, y) for y in range(grid.shape[0])
        for x in range(grid.shape[1])
        if int(grid[y, x]) in (8, 9, 15)
    }

    pieces = None
    if remembered:
        predicted = set()
        for p in remembered:
            predicted.update(
                (p["x"] + dx, p["y"] + dy) for dx, dy in p["cells"]
            )
        if predicted == visible:
            pieces = [
                {
                    "x": int(p["x"]), "y": int(p["y"]),
                    "w": int(p["w"]), "h": int(p["h"]),
                    "kind": p["kind"],
                    "cells": [tuple(c) for c in p["cells"]]
                }
                for p in remembered
            ]

    if pieces is None:
        pieces = []
        for color in (8, 15, 9):
            for comp in _components(grid, {color}):
                x0 = min(x for x, y in comp)
                y0 = min(y for x, y in comp)
                w = max(x for x, y in comp) - x0 + 1
                h = max(y for x, y in comp) - y0 + 1
                cells = sorted((x - x0, y - y0) for x, y in comp)
                kind = "bend" if color == 15 else "bar"
                if color == 9 and w > 1 and h > 1:
                    kind = "bend"
                pieces.append(dict(
                    x=x0, y=y0, w=w, h=h, cells=cells, kind=kind
                ))
        pieces.sort(key=lambda p: (
            p["kind"] == "bend", p["y"], p["x"]
        ))

    selected_cells = {
        (x, y) for y in range(grid.shape[0])
        for x in range(grid.shape[1]) if int(grid[y, x]) == 9
    }
    selected = -1
    if selected_cells:
        for i, p in enumerate(pieces):
            cells = {
                (p["x"] + dx, p["y"] + dy) for dx, dy in p["cells"]
            }
            if cells == selected_cells:
                selected = i
                break
        if selected < 0:
            for i, p in enumerate(pieces):
                cells = {
                    (p["x"] + dx, p["y"] + dy) for dx, dy in p["cells"]
                }
                if selected_cells <= cells:
                    selected = i
                    break
    return pieces, selected


class _World:
    # Grid codes: 0 empty, 1 water, 2 splitter, 3 harmless boundary,
    # 4 hazardous wall, 5 other solid, 16+i bend, 128+i cup.
    def __init__(self, grid, pieces, selected):
        self.grid = grid
        self.H, self.W = grid.shape
        self.S = self.W + 2
        self.A = self.S * (self.H + 2)
        self.pieces = pieces
        self.n = len(pieces)
        self.selected = selected
        self.dirs = (self.S, 1, -self.S, -1)
        self.orig = tuple(self.pos(p["x"], p["y"]) for p in pieces)
        self.offsets = [
            [dy * self.S + dx for dx, dy in p["cells"]]
            for p in pieces
        ]

        self.cups = []
        cup_at = {}
        for comp in _components(grid, {11, 13}):
            x = min(q[0] for q in comp)
            y = min(q[1] for q in comp)
            w = max(q[0] for q in comp) - x + 1
            h = max(q[1] for q in comp) - y + 1
            cells = set(comp)
            holes = [
                (xx, yy) for yy in range(y, y + h)
                for xx in range(x, x + w) if (xx, yy) not in cells
            ]
            mouth = holes[0] if holes else (x + w // 2, y)
            mx, my = mouth
            if w == 3 and h == 2:
                direction = 0 if my == y else 2
            elif w == 2 and h == 3:
                direction = 1 if mx == x else 3
            else:
                direction = 0
            ci = len(self.cups)
            self.cups.append(dict(
                x=x, y=y, w=w, h=h, mouth=mouth,
                direction=direction, cells=comp
            ))
            for q in comp:
                cup_at[q] = ci

        source_marks = [
            (x, y) for y in range(self.H) for x in range(self.W)
            if int(grid[y, x]) == 4
        ]
        droplets = [
            (x, y) for y in range(self.H) for x in range(self.W)
            if int(grid[y, x]) == 6
        ]
        if source_marks:
            self.min_y = min(y for x, y in source_marks) + 3
        elif droplets:
            self.min_y = min(y for x, y in droplets) + 2
        else:
            self.min_y = 3

        self.base = [3] * self.A
        for y in range(self.H):
            for x in range(self.W):
                color = int(grid[y, x])
                v = 0
                if (x, y) in cup_at:
                    v = 128 + cup_at[x, y]
                elif color == 1:
                    v = 4
                elif color == 6:
                    v = 1
                elif color == 4:
                    v = 5
                elif color not in (8, 9, 12, 15):
                    v = 5
                self.base[self.pos(x, y)] = v

        self.sources = [self.pos(x, y) for x, y in droplets]
        self.extra_sources = [
            self.pos(x, y + 1) for x, y in source_marks
            if y + 1 < self.H
        ]

        self.domains = []
        self.legal = []
        self.dist = []
        self.parent = []
        for i, p in enumerate(pieces):
            legal = set()
            for y in range(max(0, self.min_y), self.H - p["h"] + 1):
                for x in range(self.W - p["w"] + 1):
                    if any(
                        x < c["x"] + c["w"] + 1
                        and x + p["w"] > c["x"] - 1
                        and y < c["y"] + c["h"] + 1
                        and y + p["h"] > c["y"] - 1
                        for c in self.cups
                    ):
                        continue
                    q = self.pos(x, y)
                    if all(self.base[q + off] == 0
                           for off in self.offsets[i]):
                        legal.add(q)

            root = self.orig[i]
            legal.add(root)
            dist = {root: 0}
            parent = {}
            queue = deque([root])
            while queue:
                q = queue.popleft()
                for di, delta in enumerate(self.dirs):
                    nq = q + delta
                    if nq in legal and nq not in dist:
                        dist[nq] = dist[q] + 1
                        parent[nq] = (q, di)
                        queue.append(nq)
            domain = sorted(dist, key=lambda q: (dist[q], q))
            self.domains.append(domain)
            self.legal.append(set(domain))
            self.dist.append(dist)
            self.parent.append(parent)

        self.bits = [1 << i for i in range(4 * self.A)]
        self.goal_masks = []
        for c in self.cups:
            mx, my = c["mouth"]
            exact = [0] * (self.W + self.H + 10)
            for di in range(4):
                turn_penalty = 0 if di == c["direction"] else 6
                for y in range(self.H):
                    for x in range(self.W):
                        d = abs(x - mx) + abs(y - my) + turn_penalty
                        q = self.pos(x, y) + di * self.A
                        exact[d] |= self.bits[q]
            acc = 0
            masks = []
            for mask in exact:
                acc |= mask
                masks.append(acc)
            self.goal_masks.append(masks)

        self.cache = {}
        self.all_cups = (1 << len(self.cups)) - 1

    def pos(self, x, y):
        return (y + 1) * self.S + x + 1

    def xy(self, p):
        return p % self.S - 1, p // self.S - 1

    def move_cost(self, i, p):
        d = self.dist[i].get(p, 10000)
        return d + (int(i != self.selected) if d else 0)

    def cost(self, config):
        return sum(self.move_cost(i, p) for i, p in enumerate(config))

    def simulate(self, config):
        config = tuple(config)
        cached = self.cache.get(config)
        if cached is not None:
            return cached

        g = self.base.copy()
        # Prefer splitters over bends at overlapping cells, as in the
        # engine's movable-sprite ordering.
        order = sorted(
            range(self.n), key=lambda i: self.pieces[i]["kind"] == "bend"
        )
        for i in reversed(order):
            code = 2 if self.pieces[i]["kind"] == "bar" else 16 + i
            for off in self.offsets[i]:
                g[config[i] + off] = code

        queue = [(p, 0) for p in self.sources]
        for p in self.extra_sources:
            if g[p] == 0:
                g[p] = 1
                queue.append((p, 0))

        coverage = 0
        filled = 0
        bad = set()
        head = 0
        A, S = self.A, self.S
        dirs, bits = self.dirs, self.bits

        while head < len(queue):
            p, di = queue[head]
            head += 1
            bit = bits[p + di * A]
            if coverage & bit:
                continue
            coverage |= bit
            nxt = p + dirs[di]
            v = g[nxt]

            if v == 0:
                g[nxt] = 1
                queue.append((nxt, di))
                continue
            if v == 1:
                queue.append((nxt, di))
                continue
            if v == 4:
                bad.add(nxt)
                continue

            side = 1 if di % 2 == 0 else S
            a, b = p - side, p + side

            if v == 2:
                if g[a] == 0:
                    g[a] = 1
                    queue.append((a, di))
                if g[b] == 0:
                    g[b] = 1
                    queue.append((b, di))
            elif v >= 128:
                if g[a] == v and g[b] == v:
                    filled |= 1 << (v - 128)
                else:
                    if g[a] == 0:
                        g[a] = 1
                        queue.append((a, di))
                    if g[b] == 0:
                        g[b] = 1
                        queue.append((b, di))
            elif 16 <= v < 128:
                va, vb = g[a], g[b]
                if va == v and vb == 0:
                    nd = (di + 1) % 4
                    q = p + dirs[nd]
                    if g[q] == 0:
                        g[q] = 1
                    queue.append((q, nd))
                if vb == v and va == 0:
                    nd = (di - 1) % 4
                    q = p + dirs[nd]
                    if g[q] == 0:
                        g[q] = 1
                    queue.append((q, nd))
                else:
                    if g[a] == 0:
                        g[a] = 1
                        queue.append((a, di))
                    if g[b] == 0:
                        g[b] = 1
                        queue.append((b, di))

        win = filled == self.all_cups and not bad
        cost = self.cost(config)
        if win:
            value = 0.08 * cost
        else:
            distance = 0
            missing = 0
            for j, masks in enumerate(self.goal_masks):
                if filled & (1 << j):
                    continue
                missing += 1
                lo, hi = 0, len(masks) - 1
                if not coverage:
                    distance += hi
                    continue
                while lo < hi:
                    mid = (lo + hi) // 2
                    if coverage & masks[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                distance += lo
            value = 50 * missing + 4 * distance + 10 * len(bad)
            value += 0.08 * cost

        result = (value, win)
        if len(self.cache) < 60000:
            self.cache[config] = result
        return result

    def abstract_search(self, deadline):
        """Beam search over vertically ordered splitter/turning events."""
        if not self.n or self.n > 9:
            return []

        bottom = 0
        side_mask = 0
        bottom_x = []
        side_cups = []
        for j, c in enumerate(self.cups):
            if c["direction"] == 0:
                x, y = c["mouth"]
                bottom |= 1 << x
                bottom_x.append(x)
            elif c["direction"] in (1, 3):
                side_mask |= 1 << j
                side_cups.append((j, c))
            else:
                return []

        source_mask = 0
        for p in self.sources:
            x, y = self.xy(p)
            source_mask |= 1 << x
        for p in self.extra_sources:
            if self.base[p] == 0:
                x, y = self.xy(p)
                source_mask |= 1 << x

        bar_options = {}
        bend_options = {}
        bend_ids = []
        anchors = set()
        bottom_y = [
            c["mouth"][1] for c in self.cups if c["direction"] == 0
        ]
        hi_bar = min(bottom_y) - 2 if bottom_y else self.H - 2

        for i, piece in enumerate(self.pieces):
            if piece["kind"] == "bar":
                if piece["h"] != 1 or len(piece["cells"]) != piece["w"]:
                    return []
                by_y = {}
                width = piece["w"]
                for p in self.domains[i]:
                    x, y = self.xy(p)
                    if y > hi_bar:
                        continue
                    cover = ((1 << width) - 1) << x
                    exits = 0
                    if x > 0 and self.base[p - self.S - 1] == 0:
                        exits |= 1 << (x - 1)
                    if x + width < self.W:
                        q = p - self.S + width
                        if self.base[q] == 0:
                            exits |= 1 << (x + width)
                    by_y.setdefault(y, []).append(
                        (p, cover, exits, self.move_cost(i, p))
                    )
                bar_options[i] = by_y
                anchors.add(piece["y"] - 2)
            else:
                if piece["w"] != 2 or piece["h"] != 2:
                    return []
                cells = set(piece["cells"])
                holes = [
                    (x, y) for y in range(2) for x in range(2)
                    if (x, y) not in cells
                ]
                if len(holes) != 1 or holes[0][1] != 0:
                    return []
                hx, hy = holes[0]
                out = 1 if hx == 1 else -1
                di = 1 if out == 1 else 3
                options = []
                for j, cup in side_cups:
                    if cup["direction"] != di:
                        continue
                    mx, my = cup["mouth"]
                    for p in self.domains[i]:
                        x, y = self.xy(p)
                        hole_x = x + hx
                        if y != my or (mx - hole_x) * out <= 0:
                            continue
                        clear = True
                        for xx in range(hole_x + out, mx + out, out):
                            if self.base[self.pos(xx, my)] != 0:
                                clear = False
                                break
                        if clear:
                            options.append((
                                p, y, 1 << hole_x, 3 << x,
                                1 << j, self.move_cost(i, p)
                            ))
                            anchors.add(y - 2)
                            anchors.add(y + 2)
                # An unused bend can stay in place.
                p = self.orig[i]
                x, y = self.xy(p)
                options.append((p, y, 0, 3 << x, 0, 0))
                bend_options[i] = options
                bend_ids.append(i)

        if len(side_cups) > len(bend_ids):
            return []

        bend_bits = sum(1 << i for i in bend_ids)
        all_used = (1 << self.n) - 1
        rank_cache = {}

        def rank(flow, done, used, cost):
            key = (flow, done, used)
            extra = rank_cache.get(key)
            if extra is None:
                xs = [x for x in range(self.W) if flow & (1 << x)]
                d = sum(
                    min((abs(x - t) for x in xs), default=self.W)
                    for t in bottom_x
                )
                needed_side = (side_mask & ~done).bit_count()
                expected = bottom.bit_count() + needed_side
                extra = 2.0 * d + 3.0 * abs(flow.bit_count() - expected)
                remaining_bars = (
                    (all_used & ~used & ~bend_bits).bit_count()
                )
                if remaining_bars == 0:
                    extra += 12 * (flow ^ bottom).bit_count()
                rank_cache[key] = extra
            return cost + extra

        # cost, flow, side cups filled, used pieces, last event y, positions
        beam = [(0, source_mask, 0, 0, self.min_y - 2, self.orig)]
        near = []
        width = 550 if self.n <= 5 else 400

        for depth in range(self.n):
            candidates = {}
            for si, state in enumerate(beam):
                if si % 16 == 0 and time.perf_counter() > deadline:
                    near.extend(s[5] for s in beam[:24])
                    return near
                cost, flow, done, used, last_y, config = state
                for i, piece in enumerate(self.pieces):
                    bit = 1 << i
                    if used & bit:
                        continue
                    new_used = used | bit
                    transitions = []

                    if i in bar_options:
                        lo = max(self.min_y, last_y + 2)
                        if lo > hi_bar:
                            continue
                        oy = piece["y"]
                        ys = {
                            lo, max(lo, min(hi_bar, oy)),
                            max(lo, min(hi_bar, oy - 1))
                        }
                        ys.update(
                            y for y in anchors
                            if lo <= y <= hi_bar
                        )
                        ys = sorted(
                            ys, key=lambda y: (abs(y - oy), y)
                        )[:6]
                        if lo not in ys:
                            ys.append(lo)
                        for y in ys:
                            best_flow = {}
                            for p, cover, exits, dc in bar_options[i].get(y, ()):
                                nf = flow
                                if flow & cover:
                                    nf = (flow & ~cover) | exits
                                old = best_flow.get(nf)
                                if old is None or dc < old[0]:
                                    best_flow[nf] = (dc, p)
                            for nf, (dc, p) in best_flow.items():
                                transitions.append((nf, done, y, p, dc))
                    else:
                        for p, y, hole, body, goal, dc in bend_options[i]:
                            if y < last_y + 2:
                                continue
                            if goal:
                                if done & goal or not (flow & hole):
                                    continue
                                if flow & body & ~hole:
                                    continue
                                transitions.append(
                                    (flow & ~hole, done | goal, y, p, dc)
                                )
                            elif not flow & body:
                                transitions.append((flow, done, y, p, dc))

                    for nf, nd, ny, p, dc in transitions:
                        unfilled = (side_mask & ~nd).bit_count()
                        remaining_bends = (bend_bits & ~new_used).bit_count()
                        if unfilled > remaining_bends:
                            continue
                        nc = cost + dc
                        key = (nf, nd, new_used, ny)
                        old = candidates.get(key)
                        if old is not None and old[0] <= nc:
                            continue
                        cfg = list(config)
                        cfg[i] = p
                        candidates[key] = (
                            nc, nf, nd, new_used, ny, tuple(cfg)
                        )

            if not candidates:
                return near + [s[5] for s in beam[:24]]

            values = list(candidates.values())
            if depth == self.n - 1:
                exact = [
                    s for s in values
                    if s[1] == bottom and s[2] == side_mask
                ]
                exact.sort(key=lambda s: s[0])
                near.extend(s[5] for s in exact[:60])
                approximate = heapq.nsmallest(
                    32, values,
                    key=lambda s: rank(s[1], s[2], s[3], s[0])
                )
                near.extend(s[5] for s in approximate)
                return near

            beam = heapq.nsmallest(
                width, values,
                key=lambda s: rank(s[1], s[2], s[3], s[0])
            )
        return near


def _solve(world, seeds, round_number, deadline):
    rng = random.Random(
        198731 + round_number * 100003
        + sum((i + 1) * p for i, p in enumerate(world.orig))
    )
    original = world.orig
    best = original
    best_value, best_win = world.simulate(best)
    if best_win:
        return best, True, [best]

    candidates = [original]
    for seed in seeds or []:
        if len(seed) == world.n and all(
            p in world.legal[i] for i, p in enumerate(seed)
        ):
            candidates.append(tuple(seed))

    if world.n == 1:
        for p in world.domains[0]:
            config = (p,)
            value, win = world.simulate(config)
            if value < best_value:
                best, best_value, best_win = config, value, win
        return best, best_win, [best]

    if round_number == 0:
        abstract_deadline = min(deadline - 3.0, time.perf_counter() + 3.6)
        candidates.extend(world.abstract_search(abstract_deadline))

    ranked = []
    for config in dict.fromkeys(candidates):
        if time.perf_counter() > deadline - 0.15:
            break
        value, win = world.simulate(config)
        ranked.append((value, config))
        if value < best_value:
            best, best_value, best_win = config, value, win
    ranked.sort(key=lambda q: q[0])
    elites = [cfg for value, cfg in ranked[:16]] or [best]

    # Useful destinations for non-local mutations.
    columns = set()
    rows = {world.min_y}
    for p in world.sources:
        x, y = world.xy(p)
        columns.add(x)
    for cup in world.cups:
        x, y = cup["mouth"]
        columns.add(x)
        rows.update((y, y - 2, y - 3))
    for piece in world.pieces:
        columns.update((piece["x"] - 1, piece["x"] + piece["w"]))
        rows.update((piece["y"], piece["y"] - 2, piece["y"] + 2))

    hints = []
    for i, piece in enumerate(world.pieces):
        xs = set(columns)
        if piece["kind"] == "bar":
            xs.update(x + 1 for x in columns)
            xs.update(x - piece["w"] for x in columns)
        else:
            xs.update(x - 1 for x in columns)
        hp = [
            p for p in world.domains[i]
            if world.xy(p)[0] in xs or world.xy(p)[1] in rows
        ]
        hints.append(hp or world.domains[i])

    current = best
    current_value = best_value
    iteration = 0

    while not best_win and time.perf_counter() < deadline - 0.35:
        iteration += 1
        phase = iteration % 700
        if phase == 1 and iteration > 1:
            current = best if rng.random() < 0.7 else rng.choice(elites)
            trial = list(current)
            for _ in range(rng.randint(1, min(3, world.n))):
                i = rng.randrange(world.n)
                trial[i] = rng.choice(hints[i])
            current = tuple(trial)
            current_value = world.simulate(current)[0]

        i = rng.randrange(world.n)
        trial = list(current)
        r = rng.random()
        if r < 0.49:
            p = trial[i]
            delta = rng.choice(world.dirs)
            if rng.random() < 0.20:
                delta *= rng.choice((2, 3))
            q = p + delta
            if q not in world.legal[i]:
                continue
            trial[i] = q
        elif r < 0.86:
            trial[i] = rng.choice(hints[i])
        else:
            trial[i] = rng.choice(world.domains[i])

        if rng.random() < 0.065 and world.n > 1:
            j = rng.randrange(world.n)
            trial[j] = rng.choice(hints[j])
        trial = tuple(trial)
        if trial == current:
            continue

        value, win = world.simulate(trial)
        if value < best_value:
            best, best_value, best_win = trial, value, win
            elites.insert(0, best)
            del elites[20:]
        temperature = 3.0 + 34.0 * (1.0 - phase / 700.0) ** 2
        if value <= current_value or rng.random() < math.exp(
            max(-60.0, (current_value - value) / temperature)
        ):
            current, current_value = trial, value

        # Periodic coordinate minimization supplies precise alignments.
        if iteration % 180 == 0:
            i = rng.randrange(world.n)
            choices = world.domains[i]
            if len(choices) > 180:
                choices = rng.sample(choices, 180)
            local = current
            local_value = current_value
            for q in choices:
                if time.perf_counter() >= deadline - 0.35:
                    break
                trial = list(current)
                trial[i] = q
                trial = tuple(trial)
                value, win = world.simulate(trial)
                if value < local_value:
                    local, local_value = trial, value
                if value < best_value:
                    best, best_value, best_win = trial, value, win
                if best_win:
                    break
            current, current_value = local, local_value

    # Once a solution exists, return pieces toward their starting positions
    # whenever doing so preserves the solution.
    if best_win:
        changed = True
        while changed and time.perf_counter() < deadline - 0.12:
            changed = False
            order = sorted(
                range(world.n),
                key=lambda i: world.move_cost(i, best[i]),
                reverse=True
            )
            for i in order:
                old_cost = world.move_cost(i, best[i])
                if old_cost == 0:
                    continue
                for q in world.domains[i]:
                    if world.move_cost(i, q) >= old_cost:
                        continue
                    if time.perf_counter() >= deadline - 0.12:
                        break
                    trial = list(best)
                    trial[i] = q
                    trial = tuple(trial)
                    value, win = world.simulate(trial)
                    if win:
                        best, best_value = trial, value
                        changed = True
                        break

    saved = [best]
    saved.extend(cfg for cfg in elites if cfg != best)
    return best, best_win, saved[:16]


def _display_click(x, y, k, scale, ox, oy):
    px = ox + x * scale + scale // 2
    py = oy + y * scale + scale // 2
    if k == 1:
        px, py = py, 63 - px
    elif k == 2:
        px, py = 63 - px, 63 - py
    elif k == 3:
        px, py = 63 - py, px
    return ("ACTION6", int(max(0, min(63, py))),
            int(max(0, min(63, px))))


def _click_point(i, pieces, positions):
    p = pieces[i]
    x, y = positions[i]
    for dy in range(p["h"]):
        for dx in range(p["w"]):
            xx, yy = x + dx, y + dy
            blocked = False
            for j, other in enumerate(pieces):
                if i == j:
                    continue
                ox, oy = positions[j]
                if ox <= xx < ox + other["w"] and oy <= yy < oy + other["h"]:
                    blocked = True
                    break
            if not blocked:
                return xx, yy
    return None


def _make_plan(world, target, fire, transform):
    k, scale, ox, oy = transform
    positions = [world.xy(p) for p in world.orig]
    remaining = {i for i in range(world.n) if target[i] != world.orig[i]}
    selected = world.selected
    actions = []

    # Internal directions: down, right, up, left.
    world_actions = (2, 4, 1, 3)
    remap = (
        {1: 1, 2: 2, 3: 3, 4: 4},
        {1: 3, 2: 4, 3: 2, 4: 1},
        {1: 2, 2: 1, 3: 4, 4: 3},
        {1: 4, 2: 3, 3: 1, 4: 2},
    )[k]

    while remaining:
        options = []
        for i in remaining:
            click = _click_point(i, world.pieces, positions)
            if click is None and selected != i:
                continue
            tx, ty = world.xy(target[i])
            piece = world.pieces[i]
            covers = 0
            for j in remaining:
                if i == j:
                    continue
                x, y = positions[j]
                other = world.pieces[j]
                if (tx <= x and ty <= y
                        and tx + piece["w"] >= x + other["w"]
                        and ty + piece["h"] >= y + other["h"]):
                    covers += 1
            options.append((covers, i != selected, i, click))

        if options:
            _, _, i, click = min(options)
        else:
            i = min(remaining)
            click = positions[i]

        if selected != i:
            if click is None:
                click = positions[i]
            actions.append(_display_click(
                click[0], click[1], k, scale, ox, oy
            ))
            selected = i

        path = []
        q = target[i]
        while q != world.orig[i]:
            parent = world.parent[i].get(q)
            if parent is None:
                break
            q, di = parent
            path.append(di)
        for di in reversed(path):
            action = remap[world_actions[di]]
            actions.append(("ACTION%d" % action, None, None))

        positions[i] = world.xy(target[i])
        remaining.remove(i)

    if fire:
        actions.append(("ACTION5", None, None))

    model = []
    for i, piece in enumerate(world.pieces):
        p = dict(piece)
        p["cells"] = [list(c) for c in piece["cells"]]
        p["x"], p["y"] = positions[i]
        model.append(p)
    return actions, model


def act(obs, available, memory):
    started = time.perf_counter()
    queued = memory.get("_queue")
    if queued:
        action = queued.pop(0)
        return tuple(action)

    grid, k, scale, ox, oy = _read_image(obs)
    pieces, selected = _make_pieces(grid, memory.get("_model"))

    if not pieces:
        if "ACTION5" in available:
            return "ACTION5", None, None
        return available[0], None, None

    world = _World(grid, pieces, selected)
    rounds = int(memory.get("_search_round", 0))
    seeds = memory.get("_seeds", [])
    target, solved, saved = _solve(
        world, seeds, rounds, started + 8.65
    )
    memory["_seeds"] = [list(cfg) for cfg in saved]

    if solved:
        actions, model = _make_plan(
            world, target, True, (k, scale, ox, oy)
        )
        memory["_queue"] = actions
        memory["_model"] = model
        memory["_search_round"] = 0
        return tuple(memory["_queue"].pop(0))

    memory["_search_round"] = rounds + 1

    # A selection click preserves the board while allowing another bounded
    # search on the next call. After several rounds, make useful progress
    # toward the best arrangement instead.
    if rounds >= 4 and target != world.orig:
        actions, model = _make_plan(
            world, target, False, (k, scale, ox, oy)
        )
        if actions:
            memory["_queue"] = actions
            memory["_model"] = model
            memory["_search_round"] = 1
            return tuple(memory["_queue"].pop(0))

    memory["_model"] = [
        {
            "x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"],
            "kind": p["kind"], "cells": [list(c) for c in p["cells"]]
        }
        for p in pieces
    ]
    i = selected if selected >= 0 else 0
    positions = [(p["x"], p["y"]) for p in pieces]
    point = _click_point(i, pieces, positions)
    if point is None:
        point = positions[i]
    return _display_click(point[0], point[1], k, scale, ox, oy)

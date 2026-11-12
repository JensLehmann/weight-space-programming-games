import time
import math
import random
from collections import deque
import numpy as np


class _TimeLimit(Exception):
    pass


def _components(grid, allowed, seeds=None):
    h, w = grid.shape
    allowed = set(allowed)
    seen = set()
    result = []
    if seeds is None:
        seeds = allowed
    seeds = set(seeds)
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or int(grid[y, x]) not in seeds:
                continue
            todo = [(x, y)]
            seen.add((x, y))
            comp = []
            while todo:
                xx, yy = todo.pop()
                comp.append((xx, yy))
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and (nx, ny) not in seen
                            and int(grid[ny, nx]) in allowed):
                        seen.add((nx, ny))
                        todo.append((nx, ny))
            result.append(comp)
    return result


def _decode(obs):
    a = np.asarray(obs)
    edges = (a[0, :], a[:, 0], a[-1, :], a[:, -1])
    k = max(range(4), key=lambda j:
            int(np.count_nonzero((edges[j] == 14) | (edges[j] == 0))))
    a = np.rot90(a, -k)

    scale = 0
    interesting = {6, 8, 9, 11, 13, 15}
    for y in range(1, 64):
        x = 0
        while x < 64:
            c = int(a[y, x])
            j = x + 1
            while j < 64 and int(a[y, j]) == c:
                j += 1
            if c in interesting:
                scale = math.gcd(scale, j - x)
            x = j
    if not scale:
        scale = 4
    while scale > 8:
        scale //= 2

    mask = (a != 1) & (a != 0) & (a != 14)
    mask[0, :] = False
    ys, xs = np.where(mask)
    if len(xs):
        padding = min(int(xs.min()), 63 - int(xs.max()),
                      int(ys.min()), 63 - int(ys.max()))
        if int(ys.min()) <= 1:
            padding = 0
    else:
        padding = 0
    while padding > 0 and (64 - 2 * padding) % scale:
        padding -= 1

    size = (64 - 2 * padding) // scale
    size = max(4, min(64, size))
    centers = padding + np.arange(size) * scale + scale // 2
    centers = np.clip(centers, 0, 63)
    grid = a[np.ix_(centers, centers)].astype(np.int8)
    return grid, k, scale, padding


class _Player:
    def __init__(self, obs):
        grid, self.rotation, self.scale, self.padding = _decode(obs)
        self.h, self.w = grid.shape
        self.stride = self.w + 2
        self.n = (self.h + 2) * self.stride
        S = self.stride
        self.delta = (-S, S, -1, 1)
        self.lateral = (1, 1, S, S)
        self.turn_a = (2, 3, 1, 0)
        self.turn_b = (3, 2, 0, 1)
        self.rng = random.Random(81273)
        self.calls = 0
        self.cache = {}
        self.banned = set()
        self.elite = []
        self.solution = None
        self.fired = None
        self.failures = 0
        self.waits = 0

        def index(x, y):
            return (y + 1) * S + x + 1

        self.base = [-3] * self.n
        for y in range(self.h):
            for x in range(self.w):
                c = int(grid[y, x])
                p = index(x, y)
                if c == 12 or c in (8, 9, 11, 13, 15):
                    self.base[p] = 0
                elif c == 6:
                    self.base[p] = -1
                elif c in (1, 14):
                    self.base[p] = -2
                else:
                    self.base[p] = -3

        self.cups = []
        cup_comps = _components(grid, (11, 13))
        for comp in cup_comps:
            cells = {index(x, y) for x, y in comp}
            xs = [x for x, y in comp]
            ys = [y for x, y in comp]
            self.cups.append({
                "cells": cells,
                "box": (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            })
        for r, cup in enumerate(self.cups):
            for p in cup["cells"]:
                self.base[p] = 10 + r

        raw_parts = []
        used = set()
        for color in (9, 8, 15):
            for comp in _components(grid, (color, 4), (color,)):
                comp = [(x, y) for x, y in comp if (x, y) not in used]
                if not comp or not any(int(grid[y, x]) == color for x, y in comp):
                    continue
                used.update(comp)
                x0 = min(x for x, y in comp)
                y0 = min(y for x, y in comp)
                width = max(x for x, y in comp) - x0 + 1
                height = max(y for x, y in comp) - y0 + 1
                if color == 15:
                    corner = True
                elif color == 8:
                    corner = False
                else:
                    corner = (width > 1 and height > 1
                              and len(comp) < width * height)
                offsets = tuple(sorted((y - y0) * S + x - x0
                                       for x, y in comp))
                yellow = tuple(sorted((y - y0) * S + x - x0
                                      for x, y in comp
                                      if int(grid[y, x]) == 4))
                visible_offsets = tuple(o for o in offsets if o not in yellow)
                raw_parts.append({
                    "corner": corner, "width": width, "height": height,
                    "offsets": offsets, "yellow": yellow,
                    "visible": visible_offsets,
                    "start": index(x0, y0),
                    "selected": color == 9
                })
                for x, y in comp:
                    self.base[index(x, y)] = 0

        # Bars precede elbows in the simulator's collision queries.
        raw_parts.sort(key=lambda p: (p["corner"], p["width"], p["start"]))
        self.parts = raw_parts
        self.sources = [index(x, y) for y in range(self.h)
                        for x in range(self.w) if int(grid[y, x]) == 6]
        self.emitters = [index(x, y) for y in range(self.h)
                         for x in range(self.w)
                         if int(grid[y, x]) == 4 and (x, y) not in used]

        self.current = []
        self.selected = None
        for i, part in enumerate(self.parts):
            width, height = part["width"], part["height"]
            valid = set()
            for y in range(3, self.h - height + 1):
                for x in range(self.w - width + 1):
                    blocked = False
                    for cup in self.cups:
                        rx, ry, ex, ey = cup["box"]
                        if (x < ex + 1 and x + width > rx - 1
                                and y < ey + 1 and y + height > ry - 1):
                            blocked = True
                            break
                    if blocked:
                        continue
                    p = index(x, y)
                    if all(self.base[p + o] == 0 for o in part["offsets"]):
                        valid.add(p)
            start = part["start"]
            valid.add(start)
            reachable = {start}
            todo = deque([start])
            while todo:
                p = todo.popleft()
                for d in self.delta:
                    q = p + d
                    if q in valid and q not in reachable:
                        reachable.add(q)
                        todo.append(q)
            positions = sorted(reachable)
            lookup = {p: j for j, p in enumerate(positions)}
            part["positions"] = positions
            part["lookup"] = lookup
            part["cells"] = [tuple(p + o for o in part["offsets"])
                             for p in positions]
            part["emit"] = [tuple(p + o for o in part["yellow"])
                            for p in positions]
            part["adj"] = [
                tuple(lookup[p + d] for d in self.delta if p + d in lookup)
                for p in positions
            ]
            self.current.append(lookup[start])
            if part["selected"]:
                self.selected = i

        self.goal_options = []
        for r, cup in enumerate(self.cups):
            cells = cup["cells"]
            options = []
            rx, ry, ex, ey = cup["box"]
            for y in range(max(0, ry - 1), min(self.h, ey + 1)):
                for x in range(max(0, rx - 1), min(self.w, ex + 1)):
                    p = index(x, y)
                    if p in cells:
                        continue
                    for d in range(4):
                        side = self.lateral[d]
                        if (p + self.delta[d] in cells
                                and p - side in cells and p + side in cells):
                            options.append((p, d))
            self.goal_options.append(options)

        self.bits = [[1 << (d * self.n + p) for p in range(self.n)]
                     for d in range(4)]
        self.rings = []
        self.leak_distance = [[100.0] * self.n for _ in range(4)]

        for options in self.goal_options:
            buckets = {}
            if not options:
                self.rings.append(([0], [100]))
                continue
            for d in range(4):
                for y in range(self.h):
                    for x in range(self.w):
                        p = index(x, y)
                        best = 10000
                        for gp, gd in options:
                            gx = gp % S - 1
                            gy = gp // S - 1
                            if gd < 2:
                                cost = 4 * abs(x - gx) + abs(y - gy)
                            else:
                                cost = 4 * abs(y - gy) + abs(x - gx)
                            if d != gd:
                                cost += 32 if d // 2 == gd // 2 else 16
                            best = min(best, cost)
                        buckets[best] = buckets.get(best, 0) | self.bits[d][p]
                        self.leak_distance[d][p] = min(
                            self.leak_distance[d][p], best / 4.0)
            masks, values = [], []
            acc = 0
            for value in sorted(buckets):
                acc |= buckets[value]
                masks.append(acc)
                values.append(value)
            self.rings.append((masks, values))

        self.prepare_paths()

    def grid_from_obs(self, obs):
        a = np.rot90(np.asarray(obs), -self.rotation)
        centers = self.padding + np.arange(self.w) * self.scale + self.scale // 2
        centers = np.clip(centers, 0, 63)
        return a[np.ix_(centers, centers)]

    def synchronize(self, obs):
        grid = self.grid_from_obs(obs)
        ys, xs = np.where(grid == 9)
        if not len(xs):
            return
        observed = {(int(y) + 1) * self.stride + int(x) + 1
                    for y, x in zip(ys, xs)}
        smallest = min(observed)
        matches = []
        for i, part in enumerate(self.parts):
            offsets = part["visible"]
            if not offsets or len(offsets) != len(observed):
                continue
            p = smallest - min(offsets)
            if p in part["lookup"] and {p + o for o in offsets} == observed:
                oldp = part["positions"][self.current[i]]
                distance = abs(p % self.stride - oldp % self.stride)
                distance += abs(p // self.stride - oldp // self.stride)
                matches.append((distance, i != self.selected, i, p))
        if matches:
            _, _, i, p = min(matches)
            self.selected = i
            self.current[i] = self.parts[i]["lookup"][p]

    def prepare_paths(self):
        self.distances = []
        self.parents = []
        self.orders = []
        self.costs = []
        for i, part in enumerate(self.parts):
            start = self.current[i]
            count = len(part["positions"])
            distances = [10000] * count
            parents = [-1] * count
            distances[start] = 0
            todo = deque([start])
            while todo:
                p = todo.popleft()
                for q in part["adj"][p]:
                    if distances[q] == 10000:
                        distances[q] = distances[p] + 1
                        parents[q] = p
                        todo.append(q)
            self.distances.append(distances)
            self.parents.append(parents)
            self.orders.append(sorted(range(count), key=lambda j: distances[j]))
            self.costs.append([
                d + (1 if d and i != self.selected else 0)
                for d in distances
            ])

    def simulate(self, config):
        cached = self.cache.get(config)
        if cached is not None:
            return cached

        board = self.base[:]
        overlap = 0
        emitters = self.emitters[:]
        for i, (part, choice) in enumerate(zip(self.parts, config)):
            value = 100 + i if part["corner"] else 1
            for p in part["cells"][choice]:
                if board[p] == 0:
                    board[p] = value
                else:
                    overlap += 1
            emitters.extend(part["emit"][choice])

        active = [(p, 1) for p in self.sources]
        for p in emitters:
            q = p + self.stride
            if 0 <= q < self.n and board[q] == 0:
                board[q] = -1
                active.append((q, 1))

        filled = 0
        leaks = set()
        trace = 0
        work = 0
        loop = False
        max_ticks = 4 * (self.w + self.h)
        delta = self.delta
        lateral = self.lateral
        bits = self.bits

        for tick in range(max_ticks):
            if not active:
                break
            following = []
            seen_next = set()

            def append(q, d, create=False):
                if not 0 <= q < self.n:
                    return
                if create and board[q] == 0:
                    board[q] = -1
                key = (q, d)
                if key not in seen_next:
                    seen_next.add(key)
                    following.append(key)

            for p, d in active:
                work += 1
                if work > 7000:
                    loop = True
                    break
                trace |= bits[d][p]
                q = p + delta[d]
                if not 0 <= q < self.n:
                    continue
                value = board[q]
                if value == 0:
                    append(q, d, True)
                elif value == -1:
                    append(q, d)
                elif value == -2:
                    leaks.add((p, d))
                elif value == 1:
                    side = lateral[d]
                    for q2 in (p - side, p + side):
                        if 0 <= q2 < self.n and board[q2] == 0:
                            append(q2, d, True)
                elif 10 <= value < 100:
                    side = lateral[d]
                    a, b = p - side, p + side
                    va = board[a] if 0 <= a < self.n else -3
                    vb = board[b] if 0 <= b < self.n else -3
                    if va == value and vb == value:
                        filled |= 1 << (value - 10)
                    else:
                        for q2 in (a, b):
                            if 0 <= q2 < self.n and board[q2] == 0:
                                append(q2, d, True)
                elif value >= 100:
                    side = lateral[d]
                    a, b = p - side, p + side
                    va = board[a] if 0 <= a < self.n else -3
                    vb = board[b] if 0 <= b < self.n else -3
                    if va == value and vb == 0:
                        nd = self.turn_a[d]
                        append(p + delta[nd], nd, True)
                    if vb == value and va == 0:
                        nd = self.turn_b[d]
                        append(p + delta[nd], nd, True)
                    else:
                        for q2 in (a, b):
                            if 0 <= q2 < self.n and board[q2] == 0:
                                append(q2, d, True)
            if loop:
                break
            active = following
        else:
            loop = bool(active)

        missing = len(self.cups) - filled.bit_count()
        distance_sum = 0.0
        for r, (masks, values) in enumerate(self.rings):
            if filled & (1 << r):
                continue
            if not masks or not (masks[-1] & trace):
                distance_sum += 30.0
                continue
            lo, hi = 0, len(masks) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if masks[mid] & trace:
                    hi = mid
                else:
                    lo = mid + 1
            distance_sum += values[lo] / 4.0

        leak_distance = sum(self.leak_distance[d][p] for p, d in leaks)
        result = (missing, distance_sum, len(leaks), leak_distance,
                  overlap, int(loop))
        if len(self.cache) > 65000:
            self.cache.clear()
        self.cache[config] = result
        return result

    def winning(self, config, metrics=None):
        if config in self.banned:
            return False
        if metrics is None:
            metrics = self.simulate(config)
        return metrics[0] == 0 and metrics[2] == 0 and metrics[5] == 0

    def movement_cost(self, config):
        return sum(cost[c] for cost, c in zip(self.costs, config))

    def quality(self, config, weights=(4.5, 1.2, 2.8, 0.65)):
        metrics = self.simulate(config)
        cost = self.movement_cost(config)
        if self.winning(config, metrics):
            return -10000.0 + cost + metrics[4] * 0.03
        m, d, leaks, ld, overlap, loop = metrics
        penalty = 500.0 if config in self.banned else 0.0
        return (weights[0] * m + weights[1] * d
                + weights[2] * leaks + weights[3] * ld
                + 0.12 * overlap + 100.0 * loop
                + 0.055 * cost + penalty)

    def remember_elite(self, config):
        candidates = list(dict.fromkeys(self.elite + [config]))
        candidates.sort(key=self.quality)
        selected = []
        for candidate in candidates:
            if not selected or all(
                    sum(a != b for a, b in zip(candidate, old)) >= 2
                    for old in selected):
                selected.append(candidate)
            if len(selected) >= 10:
                break
        self.elite = selected

    def search(self, deadline):
        self.prepare_paths()
        self.deadline = deadline
        self.evaluations = 0
        current = tuple(self.current)
        candidates = [current] + self.elite
        self.best = min(candidates, key=self.quality)
        self.best_quality = self.quality(self.best)
        self.solution = None

        def evaluate(config, weights):
            self.evaluations += 1
            if self.evaluations % 32 == 0 and time.perf_counter() >= self.deadline:
                raise _TimeLimit
            value = self.quality(config, weights)
            ordinary = self.quality(config)
            if ordinary < self.best_quality:
                self.best_quality = ordinary
                self.best = config
            if self.winning(config):
                if (self.solution is None
                        or self.quality(config) < self.quality(self.solution)):
                    first = self.solution is None
                    self.solution = config
                    if first:
                        self.deadline = min(self.deadline,
                                            time.perf_counter() + 0.8)
            return value

        rng = self.rng
        count = len(self.parts)
        if not count:
            return current
        try:
            evaluate(current, (4.5, 1.2, 2.8, 0.65))
            restart = 0
            while time.perf_counter() < self.deadline:
                if restart == 0:
                    config = list(self.best)
                    weights = (4.5, 1.2, 2.8, 0.65)
                else:
                    pool = self.elite + [self.best]
                    config = list(rng.choice(pool))
                    weights = (rng.uniform(3.0, 7.0),
                               rng.uniform(0.75, 1.6),
                               rng.uniform(1.4, 4.0),
                               rng.uniform(0.3, 0.95))
                    if rng.random() < 0.13:
                        indices = list(range(count))
                    else:
                        number = 1 + (rng.random() < 0.55) + (rng.random() < 0.2)
                        indices = rng.sample(range(count), min(count, number))
                    for i in indices:
                        order = self.orders[i]
                        if rng.random() < 0.55:
                            config[i] = rng.choice(order[:min(len(order), 50)])
                        else:
                            config[i] = rng.randrange(len(order))

                for sweep in range(5):
                    changed = False
                    order_parts = list(range(count))
                    rng.shuffle(order_parts)
                    if restart == 0 and sweep == 0 and self.selected is not None:
                        order_parts.remove(self.selected)
                        order_parts.insert(0, self.selected)

                    for i in order_parts:
                        old = config[i]
                        best_choice = old
                        best_value = evaluate(tuple(config), weights)
                        for choice in self.orders[i]:
                            if choice == old:
                                continue
                            config[i] = choice
                            value = evaluate(tuple(config), weights)
                            if value < best_value - 1e-9:
                                best_value = value
                                best_choice = choice
                        config[i] = best_choice
                        changed |= best_choice != old
                    self.remember_elite(tuple(config))
                    if not changed:
                        break
                restart += 1
        except _TimeLimit:
            pass
        self.remember_elite(self.best)
        return self.solution if self.solution is not None else self.best

    def screen_click(self, x, y):
        x = self.padding + x * self.scale + self.scale // 2
        y = self.padding + y * self.scale + self.scale // 2
        k = self.rotation
        if k == 1:
            x, y = y, 63 - x
        elif k == 2:
            x, y = 63 - x, 63 - y
        elif k == 3:
            x, y = 63 - y, x
        return "ACTION6", int(y), int(x)

    def screen_move(self, direction):
        mappings = (
            (1, 2, 3, 4),
            (3, 4, 2, 1),
            (2, 1, 4, 3),
            (4, 3, 1, 2),
        )
        return "ACTION" + str(mappings[self.rotation][direction]), None, None

    def box(self, i, choice=None):
        part = self.parts[i]
        if choice is None:
            choice = self.current[i]
        p = part["positions"][choice]
        x, y = p % self.stride - 1, p // self.stride - 1
        return x, y, x + part["width"], y + part["height"]

    def click_location(self, i):
        x0, y0, x1, y1 = self.box(i)
        others = [self.box(j) for j in range(len(self.parts)) if j != i]
        for y in range(y0, y1):
            for x in range(x0, x1):
                if not any(a <= x < c and b <= y < d
                           for a, b, c, d in others):
                    return x, y
        return None

    def move_toward(self, i, target):
        start = self.current[i]
        if target == start:
            return None
        parent = self.parents[i]
        q = target
        while parent[q] != start:
            q = parent[q]
            if q < 0:
                return None
        part = self.parts[i]
        difference = part["positions"][q] - part["positions"][start]
        if difference not in self.delta:
            return None
        direction = self.delta.index(difference)
        self.current[i] = q
        self.selected = i
        return self.screen_move(direction)

    def execute(self, target):
        self.prepare_paths()
        pending = [i for i in range(len(self.parts))
                   if self.current[i] != target[i]]
        if not pending:
            config = tuple(self.current)
            if self.winning(config):
                self.fired = config
                return "ACTION5", None, None
            return None

        if self.selected in pending:
            return self.move_toward(self.selected, target[self.selected])

        choices = []
        for i in pending:
            click = self.click_location(i)
            if click is None:
                continue
            a, b, c, d = self.box(i, target[i])
            covering = 0
            for j in pending:
                if j == i:
                    continue
                e, f, g, h = self.box(j)
                if a <= e and b <= f and c >= g and d >= h:
                    covering += 1
            choices.append((100 * covering + self.distances[i][target[i]], i, click))
        if choices:
            _, i, (x, y) = min(choices)
            self.selected = i
            return self.screen_click(x, y)

        # An overlapping arrangement can hide a piece's entire click box.
        # Move the selected covering piece aside and replan.
        if self.selected is not None:
            i = self.selected
            neighbors = self.parts[i]["adj"][self.current[i]]
            if neighbors:
                self.solution = None
                q = self.rng.choice(neighbors)
                return self.move_toward(i, q)
        return None


def act(obs, available, memory):
    start_time = time.perf_counter()
    available = set(available)

    player = memory.get("_sp80_player")
    if player is None:
        try:
            player = _Player(obs)
            memory["_sp80_player"] = player
        except Exception:
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION5", None, None

    player.calls += 1
    player.synchronize(obs)

    # A returned spill frame on the same episode means the attempted
    # configuration did not complete the level.
    if player.fired is not None:
        player.banned.add(player.fired)
        player.fired = None
        player.solution = None
        player.failures += 1

    if player.solution is not None:
        action = player.execute(player.solution)
        if action is not None and action[0] in available:
            return action

    target = player.search(start_time + 8.1)
    action = player.execute(target)
    if action is not None and action[0] in available:
        return action

    # Continue searching on the next call without disturbing the geometry.
    player.waits += 1
    if player.waits > 5 or player.failures >= 3:
        if "RESET" in available:
            memory.pop("_sp80_player", None)
            return "RESET", None, None

    if "ACTION6" in available:
        if player.selected is not None:
            location = player.click_location(player.selected)
            if location is None:
                a, b, c, d = player.box(player.selected)
                location = (a, b)
            return player.screen_click(*location)
        if player.parts:
            a, b, c, d = player.box(0)
            return player.screen_click(a, b)

    if "ACTION5" in available:
        player.fired = tuple(player.current)
        return "ACTION5", None, None

    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
        if name in available:
            return name, None, None
    return "ACTION5", None, None

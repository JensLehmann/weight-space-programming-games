import math
import random
from collections import deque

import numpy as np


DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _runs(row, colors):
    result = []
    x = 0
    while x < len(row):
        if int(row[x]) not in colors:
            x += 1
            continue
        color = int(row[x])
        start = x
        while x < len(row) and int(row[x]) == color:
            x += 1
        result.append((start, x, color))
    return result


def _components(grid, color):
    h, w = grid.shape
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if int(grid[y, x]) != color or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for dx, dy in DIRS:
                    xx, yy = a + dx, b + dy
                    if (0 <= xx < w and 0 <= yy < h
                            and (xx, yy) not in seen
                            and int(grid[yy, xx]) == color):
                        seen.add((xx, yy))
                        stack.append((xx, yy))
            result.append(cells)
    return result


def _read_board(obs):
    a = np.asarray(obs, dtype=np.int8)
    fours = np.argwhere(a == 4)
    sixes = np.argwhere(a == 6)
    if len(fours) == 0 or len(sixes) == 0:
        return None

    delta = sixes.mean(axis=0) - fours.mean(axis=0)
    dy, dx = float(delta[0]), float(delta[1])
    if abs(dy) >= abs(dx):
        undo = 0 if dy > 0 else 2
    else:
        undo = 1 if dx < 0 else 3
    a = np.rot90(a, undo)

    lengths = []
    for row in a:
        for start, end, _ in _runs(row, {4, 6, 8, 9, 11, 15}):
            lengths.append(end - start)
    if not lengths:
        return None

    scale = 0
    for length in lengths:
        scale = math.gcd(scale, length)
    if scale < 2 or scale > 8:
        return None

    ys, _ = np.where(a == 4)
    if not len(ys):
        return None
    pad = int(ys.min())
    if pad == 1 and scale >= 3:
        pad = 0

    n = (64 - 2 * pad) // scale
    if n < 4 or n > 32 or pad + n * scale > 64:
        return None

    grid = np.empty((n, n), dtype=np.int8)
    for y in range(n):
        sy = pad + y * scale + scale // 2
        for x in range(n):
            sx = pad + x * scale + scale // 2
            grid[y, x] = a[sy, sx]
    return grid, n, scale, pad, undo


def _targets(grid):
    n = len(grid)
    base = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.int8)
    patterns = []
    for k in range(4):
        p = np.rot90(base, k)
        cells = [(x, y) for y in range(p.shape[0])
                 for x in range(p.shape[1]) if p[y, x]]
        holes = [(x, y) for y in range(p.shape[0])
                 for x in range(p.shape[1]) if not p[y, x]]
        patterns.append((p.shape[1], p.shape[0], cells, holes[0]))

    used = set()
    goals = []
    for y in range(n):
        for x in range(n):
            for w, h, cells, hole in patterns:
                if x + w > n or y + h > n:
                    continue
                absolute = [(x + dx, y + dy) for dx, dy in cells]
                if any(c in used or int(grid[c[1], c[0]]) != 11
                       for c in absolute):
                    continue
                hx, hy = x + hole[0], y + hole[1]
                if int(grid[hy, hx]) == 11:
                    continue
                goals.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "cells": absolute, "mouth": (hx, hy)
                })
                used.update(absolute)
                break
    return goals


def _objects(grid):
    n = len(grid)
    objects = []

    for color in (9, 15):
        for cells in _components(grid, color):
            x0 = min(x for x, _ in cells)
            y0 = min(y for _, y in cells)
            x1 = max(x for x, _ in cells)
            y1 = max(y for _, y in cells)
            objects.append({
                "x": x0,
                "y": y0,
                "w": x1 - x0 + 1,
                "h": y1 - y0 + 1,
                "mask": [(x - x0, y - y0) for x, y in cells],
                "reflector": color == 15 or y1 > y0,
                "selected": color == 9,
            })

    for y in range(n):
        for start, end, _ in _runs(grid[y], {8}):
            length = end - start
            objects.append({
                "x": start,
                "y": y,
                "w": length,
                "h": 1,
                "mask": [(i, 0) for i in range(length)],
                "reflector": False,
                "selected": False,
            })
    return objects


def _reachable(obj, grid, goals):
    n = len(grid)
    w, h = obj["w"], obj["h"]
    mask = obj["mask"]

    def valid(x, y):
        if x < 0 or y < 3 or x + w > n or y + h > n:
            return False
        for g in goals:
            if (x < g["x"] + g["w"] + 1
                    and x + w > g["x"] - 1
                    and y < g["y"] + g["h"] + 1
                    and y + h > g["y"] - 1):
                return False
        for dx, dy in mask:
            if int(grid[y + dy, x + dx]) in (1, 4, 6, 11):
                return False
        return True

    start = (obj["x"], obj["y"])
    positions = [start]
    index = {start: 0}
    parents = [None]
    distances = [0]
    queue = deque([0])

    while queue:
        i = queue.popleft()
        x, y = positions[i]
        for direction, (dx, dy) in enumerate(DIRS):
            p = (x + dx, y + dy)
            if p not in index and valid(*p):
                j = len(positions)
                index[p] = j
                positions.append(p)
                parents.append((i, direction))
                distances.append(distances[i] + 1)
                queue.append(j)
    return positions, parents, distances, index


def _simulator(grid, objects, goals, choices):
    n = len(grid)
    base = [0] * (n * n)
    sources = []

    for y in range(n):
        for x in range(n):
            v = int(grid[y, x])
            i = y * n + x
            if v == 1:
                base[i] = 1
            elif v == 4:
                base[i] = 2
            elif v == 6:
                base[i] = 3
                sources.append((x, y, 0, 1))
            elif v == 11:
                base[i] = 2

    for j, goal in enumerate(goals):
        for x, y in goal["cells"]:
            base[y * n + x] = 16 + j

    mouths = [g["mouth"] for g in goals]

    def evaluate(state):
        board = base.copy()
        occupied = set()
        for j, obj in enumerate(objects):
            x, y = choices[j][0][state[j]]
            code = 5 if obj["reflector"] else 4
            for dx, dy in obj["mask"]:
                i = (y + dy) * n + x + dx
                if i in occupied:
                    return -10000.0, False
                occupied.add(i)
                board[i] = code

        def at(x, y):
            if x < 0 or y < 0 or x >= n or y >= n:
                return 2
            return board[y * n + x]

        active = list(sources)
        water = [(x, y) for x, y, _, _ in sources]
        seen = set()
        hit = set()
        wall = False
        operations = 0

        def add(x, y, dx, dy, nxt):
            board[y * n + x] = 3
            water.append((x, y))
            nxt.append((x, y, dx, dy))

        while active and operations < 2500:
            nxt = []
            for x, y, dx, dy in active:
                key = (x, y, dx, dy)
                if key in seen:
                    continue
                seen.add(key)
                operations += 1
                nx, ny = x + dx, y + dy
                v = at(nx, ny)

                if v == 0:
                    add(nx, ny, dx, dy, nxt)
                    continue
                if v == 3:
                    nxt.append((nx, ny, dx, dy))

                sides = [(-1, 0), (1, 0)] if dy != 0 else [(0, -1), (0, 1)]

                if v == 4:
                    for sx, sy in sides:
                        if at(x + sx, y + sy) == 0:
                            add(x + sx, y + sy, dx, dy, nxt)

                elif v >= 16:
                    if (at(x + sides[0][0], y + sides[0][1]) == v
                            and at(x + sides[1][0], y + sides[1][1]) == v):
                        hit.add(v - 16)
                    else:
                        for sx, sy in sides:
                            if at(x + sx, y + sy) == 0:
                                add(x + sx, y + sy, dx, dy, nxt)

                elif v == 5:
                    a = at(x + sides[0][0], y + sides[0][1])
                    b = at(x + sides[1][0], y + sides[1][1])
                    if a == 5 and b == 0:
                        rdx, rdy = dy, -dx
                        add(x + rdx, y + rdy, rdx, rdy, nxt)
                    if b == 5 and a == 0:
                        rdx, rdy = -dy, dx
                        add(x + rdx, y + rdy, rdx, rdy, nxt)
                    else:
                        for sx, sy in sides:
                            if at(x + sx, y + sy) == 0:
                                add(x + sx, y + sy, dx, dy, nxt)

                elif v == 1:
                    wall = True

            active = nxt

        win = len(hit) == len(goals) and not wall and operations < 2500
        score = 12.0 * len(hit) - (4.0 if wall else 0.0)
        for j, (mx, my) in enumerate(mouths):
            if j in hit:
                continue
            distance = min(
                (abs(x - mx) + abs(y - my) for x, y in water),
                default=n * 2,
            )
            score += 2.0 / (1.0 + distance)
        return score, win

    return evaluate


def _display_point(x, y, scale, pad, undo):
    x = pad + x * scale + scale // 2
    y = pad + y * scale + scale // 2
    k = (-undo) % 4
    if k == 1:
        x, y = y, 63 - x
    elif k == 2:
        x, y = 63 - x, 63 - y
    elif k == 3:
        x, y = 63 - y, x
    return y, x


def _display_direction(direction, undo):
    dx, dy = DIRS[direction]
    k = (-undo) % 4
    if k == 1:
        dx, dy = dy, -dx
    elif k == 2:
        dx, dy = -dx, -dy
    elif k == 3:
        dx, dy = -dy, dx
    return "ACTION" + str(DIRS.index((dx, dy)) + 1)


def _make_plan(state, objects, choices, scale, pad, undo, pour):
    positions = [(o["x"], o["y"]) for o in objects]
    selected = next(
        (i for i, o in enumerate(objects) if o["selected"]), None
    )
    remaining = {i for i, s in enumerate(state) if s != 0}
    plan = []

    while remaining:
        candidates = list(remaining)
        if selected in remaining:
            candidates.remove(selected)
            candidates.insert(0, selected)

        chosen = None
        click_cell = None
        for i in candidates:
            if i == selected:
                chosen = i
                break

            x, y = positions[i]
            obj = objects[i]
            for dx, dy in obj["mask"]:
                xx, yy = x + dx, y + dy
                unique = True
                for j, other in enumerate(objects):
                    if i == j:
                        continue
                    ox, oy = positions[j]
                    if (ox <= xx < ox + other["w"]
                            and oy <= yy < oy + other["h"]):
                        unique = False
                        break
                if unique:
                    chosen = i
                    click_cell = (xx, yy)
                    break
            if chosen is not None:
                break

        if chosen is None:
            return None

        if chosen != selected:
            row, col = _display_point(
                click_cell[0], click_cell[1], scale, pad, undo
            )
            plan.append(("ACTION6", row, col))
            selected = chosen

        index = state[chosen]
        parents = choices[chosen][1]
        path = []
        while index != 0:
            parent, direction = parents[index]
            path.append(direction)
            index = parent
        path.reverse()

        for direction in path:
            plan.append((_display_direction(direction, undo), None, None))
        positions[chosen] = choices[chosen][0][state[chosen]]
        remaining.remove(chosen)

    if pour:
        plan.append(("ACTION5", None, None))
    return plan


def _selected_bounds(obs):
    ys, xs = np.where(np.asarray(obs) == 9)
    if not len(xs):
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _emit(action, obs, memory, verify=True):
    action = tuple(action)
    name, row, col = action
    memory["dirty"] = True

    if name == "ACTION5":
        memory["poured"] = True
    elif name == "ACTION6" and verify:
        memory["check_click"] = (int(row), int(col))
    elif name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        bounds = _selected_bounds(obs)
        scale = memory.get("scale")
        if bounds is not None and scale:
            dy, dx = {
                "ACTION1": (-scale, 0),
                "ACTION2": (scale, 0),
                "ACTION3": (0, -scale),
                "ACTION4": (0, scale),
            }[name]
            memory["check_move"] = (
                bounds[0] + dy, bounds[1] + dx,
                bounds[2] + dy, bounds[3] + dx,
            )
    return action


def _reset(obs, memory):
    memory["plan"] = []
    memory.pop("check_click", None)
    memory.pop("check_move", None)
    memory.pop("poured", None)

    # A reset with no preceding action can be a full-game reset.
    if not memory.get("dirty"):
        memory["pending_reset"] = True
        return _emit(("ACTION6", 0, 0), obs, memory, verify=False)

    memory["resets"] = int(memory.get("resets", 0)) + 1
    memory["dirty"] = False
    return ("RESET", None, None)


def act(obs, available, memory):
    if memory.pop("pending_reset", False):
        return _reset(obs, memory)

    if memory.pop("poured", False):
        if int(memory.get("resets", 0)) < 3:
            return _reset(obs, memory)
        memory["plan"] = []

    check_click = memory.pop("check_click", None)
    if check_click is not None:
        row, col = check_click
        if int(obs[row, col]) != 9:
            memory["plan"] = []
            if int(memory.get("resets", 0)) < 3:
                return _reset(obs, memory)

    check_move = memory.pop("check_move", None)
    if check_move is not None and _selected_bounds(obs) != tuple(check_move):
        memory["plan"] = []
        memory["move_failures"] = int(memory.get("move_failures", 0)) + 1
        if (memory["move_failures"] >= 2
                and int(memory.get("resets", 0)) < 3):
            memory["move_failures"] = 0
            return _reset(obs, memory)

    if memory.get("plan"):
        action = memory["plan"].pop(0)
        return _emit(action, obs, memory)

    parsed = _read_board(obs)
    if parsed is None:
        if int(memory.get("resets", 0)) < 2:
            return _reset(obs, memory)
        return _emit(("ACTION5", None, None), obs, memory)

    grid, n, scale, pad, undo = parsed
    memory["scale"] = scale
    goals = _targets(grid)
    objects = _objects(grid)

    if not goals or not objects:
        if int(memory.get("resets", 0)) < 2:
            return _reset(obs, memory)
        return _emit(("ACTION5", None, None), obs, memory)

    choices = [_reachable(o, grid, goals) for o in objects]
    simulate = _simulator(grid, objects, goals, choices)
    initial = tuple(0 for _ in objects)

    def cost(state):
        movement = sum(choices[i][2][s] for i, s in enumerate(state))
        moved = sum(s != 0 for s in state)
        selected_moved = any(
            objects[i]["selected"] and s != 0
            for i, s in enumerate(state)
        )
        return movement + moved - int(selected_moved) + 1

    initial_score, initial_win = simulate(initial)
    if initial_win:
        return _emit(("ACTION5", None, None), obs, memory)

    best_state = initial
    best_score = initial_score
    best_win = None
    best_win_cost = 10 ** 9

    def consider(state, raw_score, win):
        nonlocal best_state, best_score, best_win, best_win_cost
        adjusted = raw_score - 0.035 * (cost(state) - 1)
        if adjusted > best_score:
            best_score = adjusted
            best_state = state
        if win:
            c = cost(state)
            if c < best_win_cost:
                best_win = state
                best_win_cost = c
        return adjusted

    for i in range(len(objects)):
        for j in range(1, len(choices[i][0])):
            state = list(initial)
            state[i] = j
            state = tuple(state)
            score, win = simulate(state)
            consider(state, score, win)
        if best_win is not None and best_win_cost <= 14:
            break

    if best_win is None or best_win_cost > 14:
        attempt = int(memory.get("search_attempt", 0))
        seed = int(np.asarray(obs, dtype=np.int64).sum())
        seed += 1000003 * attempt + 7919 * len(objects)
        rng = random.Random(seed)

        current = initial
        current_score = initial_score
        movable = [i for i, c in enumerate(choices) if len(c[0]) > 1]
        iterations = 7000 if movable else 0

        for t in range(iterations):
            if t and t % 180 == 0:
                r = rng.random()
                if r < 0.55:
                    current = best_state
                elif r < 0.8:
                    current = initial
                else:
                    current = tuple(
                        rng.randrange(len(c[0])) for c in choices
                    )
                raw, win = simulate(current)
                current_score = consider(current, raw, win)

            i = rng.choice(movable)
            positions, _, _, index = choices[i]
            if rng.random() < 0.55:
                new_index = rng.randrange(len(positions))
            else:
                x, y = positions[current[i]]
                p = (x + rng.randint(-4, 4), y + rng.randint(-4, 4))
                new_index = index.get(p, rng.randrange(len(positions)))

            candidate = list(current)
            candidate[i] = new_index
            candidate = tuple(candidate)
            raw, win = simulate(candidate)
            value = consider(candidate, raw, win)

            phase = (t % 180) / 180.0
            temperature = 0.7 + 3.8 * (1.0 - phase)
            if (value >= current_score
                    or rng.random() < math.exp(
                        max(-40.0, (value - current_score) / temperature)
                    )):
                current = candidate
                current_score = value

            if best_win is not None and best_win_cost <= 14:
                break

    memory["search_attempt"] = int(memory.get("search_attempt", 0)) + 1

    if best_win is not None:
        plan = _make_plan(
            best_win, objects, choices, scale, pad, undo, True
        )
        if plan:
            memory["plan"] = plan[1:]
            return _emit(plan[0], obs, memory)

    # Restart once before committing to a partial solution. This is especially
    # useful when a perturbed state has overlapping or obscured objects.
    if int(memory.get("resets", 0)) < 1:
        return _reset(obs, memory)

    if best_state != initial:
        plan = _make_plan(
            best_state, objects, choices, scale, pad, undo, False
        )
        if plan:
            memory["plan"] = plan[1:]
            return _emit(plan[0], obs, memory)

    # Retry the stochastic search with a different seed before pouring.
    retries = int(memory.get("idle_retries", 0))
    if retries < 2:
        memory["idle_retries"] = retries + 1
        return _emit(("ACTION6", 0, 0), obs, memory, verify=False)

    if int(memory.get("resets", 0)) < 3:
        memory["idle_retries"] = 0
        return _reset(obs, memory)

    return _emit(("ACTION5", None, None), obs, memory)

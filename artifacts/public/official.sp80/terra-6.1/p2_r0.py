import math
import random
import time
from collections import deque
from itertools import permutations

import numpy as np


_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _components(a, value):
    h, w = a.shape
    seen = set()
    out = []

    for y in range(h):
        for x in range(w):
            if int(a[y, x]) != value or (x, y) in seen:
                continue

            q = [(x, y)]
            seen.add((x, y))
            cells = []

            for px, py in q:
                cells.append((px, py))
                for dx, dy in _DIRS:
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < w and 0 <= ny < h
                            and (nx, ny) not in seen
                            and int(a[ny, nx]) == value):
                        seen.add((nx, ny))
                        q.append((nx, ny))

            out.append(cells)

    return out


def _box(cells):
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _decode(obs):
    a = np.asarray(obs, dtype=np.int16)
    water_components = _components(a, 6)
    if not water_components:
        return None

    sizes = []
    for c in water_components:
        x0, y0, x1, y1 = _box(c)
        sizes.append(min(x1 - x0, y1 - y0))
    scale = max(1, min(sizes))

    fours = _components(a, 4)
    best = None

    for wc in water_components:
        wx0, wy0, wx1, wy1 = _box(wc)
        wx = (wx0 + wx1 - 1) / 2
        wy = (wy0 + wy1 - 1) / 2

        for fc in fours:
            fx0, fy0, fx1, fy1 = _box(fc)
            fx = (fx0 + fx1 - 1) / 2
            fy = (fy0 + fy1 - 1) / 2
            dx, dy = wx - fx, wy - fy

            if abs(dx) >= abs(dy):
                dist, sideways = abs(dx), abs(dy)
            else:
                dist, sideways = abs(dy), abs(dx)

            err = abs(dist - scale) + 2 * sideways
            if dist >= scale * 0.4 and (best is None or err < best[0]):
                best = (err, dx, dy)

    if best is not None and best[0] < max(3, scale * 1.6):
        dx, dy = best[1], best[2]
        if abs(dy) >= abs(dx):
            rotation = 0 if dy > 0 else 2
        else:
            rotation = 1 if dx > 0 else 3
    else:
        targets = _components(a, 11)
        if not targets:
            return None

        wn = sum(map(len, water_components))
        tn = sum(map(len, targets))
        wx = sum(x for c in water_components for x, y in c) / wn
        wy = sum(y for c in water_components for x, y in c) / wn
        tx = sum(x for c in targets for x, y in c) / tn
        ty = sum(y for c in targets for x, y in c) / tn
        dx, dy = tx - wx, ty - wy

        if abs(dy) >= abs(dx):
            rotation = 0 if dy > 0 else 2
        else:
            rotation = 1 if dx > 0 else 3

    a = np.rot90(a, -rotation)
    water_components = _components(a, 6)
    if not water_components:
        return None

    min_water_y = min(y for c in water_components for x, y in c)
    offset = min_water_y - scale
    if not (0 <= offset < 16):
        return None

    size = (64 - 2 * offset) // scale
    if size < 4 or size > 64:
        return None

    grid = np.empty((size, size), dtype=np.int16)
    for y in range(size):
        for x in range(size):
            py = min(63, offset + y * scale + scale // 2)
            px = min(63, offset + x * scale + scale // 2)
            grid[y, x] = a[py, px]

    sources = [
        (x, y)
        for y in range(size)
        for x in range(size)
        if int(grid[y, x]) == 6
    ]
    if not sources:
        return None

    return grid, sources, scale, offset, rotation


def _make_board(decoded):
    grid, sources, scale, offset, rotation = decoded
    n = grid.shape[0]

    target_cells = _components(grid, 11)
    if not target_cells:
        return None

    targets = []
    for cells in target_cells:
        targets.append({
            "cells": set(cells),
            "box": _box(cells),
        })

    pieces = []

    for color in (8, 9, 15):
        for cells in _components(grid, color):
            x0, y0, x1, y1 = _box(cells)

            if y1 - y0 == 1 and x1 - x0 > 6:
                chunks = []
                left = x0
                while left < x1:
                    right = min(x1, left + 6)
                    chunks.append([(x, y0) for x in range(left, right)])
                    left = right
            else:
                chunks = [cells]

            for chunk in chunks:
                bx0, by0, bx1, by1 = _box(chunk)
                mask = tuple((x - bx0, y - by0) for x, y in chunk)
                reflector = color == 15 or (
                    color == 9 and by1 - by0 > 1
                )

                pieces.append({
                    "x": bx0,
                    "y": by0,
                    "w": bx1 - bx0,
                    "h": by1 - by0,
                    "mask": mask,
                    "reflector": reflector,
                    "selected": color == 9,
                })

    if not pieces:
        return None

    # The game's collision and click order places bars before reflectors.
    pieces.sort(key=lambda p: (p["reflector"], p["y"], p["x"]))

    # 0 empty, 1 dangerous boundary, 5 inert obstruction, 10+i target.
    static = [0] * (n * n)
    for y in range(n):
        for x in range(n):
            c = int(grid[y, x])
            if c == 1:
                static[y * n + x] = 1
            elif c == 4:
                static[y * n + x] = 5

    for x, y in sources:
        if y > 0:
            static[(y - 1) * n + x] = 5

    for i, target in enumerate(targets):
        for x, y in target["cells"]:
            static[y * n + x] = 10 + i

    side_wall = any(
        static[y * n] == 1 or static[y * n + n - 1] == 1
        for y in range(2, max(2, n - 2))
    )

    entries = []
    for target in targets:
        cells = target["cells"]
        x0, y0, x1, y1 = target["box"]
        options = []

        for y in range(y0 - 1, y1 + 1):
            for x in range(x0 - 1, x1 + 1):
                if (x, y) in cells:
                    continue

                for di, (dx, dy) in enumerate(_DIRS):
                    sx, sy = (-1, 0) if dy else (0, -1)
                    if ((x + dx, y + dy) in cells
                            and (x + sx, y + sy) in cells
                            and (x - sx, y - sy) in cells):
                        options.append((x, y, di))

        entries.append(options)

    return {
        "n": n,
        "grid": grid,
        "sources": sources,
        "targets": targets,
        "pieces": pieces,
        "static": static,
        "side_wall": side_wall,
        "entries": entries,
        "scale": scale,
        "offset": offset,
        "rotation": rotation,
    }


def _reachable(board, piece):
    n = board["n"]
    static = board["static"]
    targets = board["targets"]
    w, h = piece["w"], piece["h"]

    def valid(x, y):
        if x < 0 or y < 3 or x + w > n or y + h > n:
            return False

        for target in targets:
            rx0, ry0, rx1, ry1 = target["box"]
            if (x < rx1 + 1 and x + w > rx0 - 1
                    and y < ry1 + 1 and y + h > ry0 - 1):
                return False

        for ox, oy in piece["mask"]:
            if static[(y + oy) * n + x + ox]:
                return False

        return True

    start = (piece["x"], piece["y"])
    parent = {start: None}
    distance = {start: 0}
    q = deque([start])

    while q:
        x, y = q.popleft()
        for di, (dx, dy) in enumerate(_DIRS):
            p = (x + dx, y + dy)
            if p not in parent and valid(*p):
                parent[p] = ((x, y), di)
                distance[p] = distance[(x, y)] + 1
                q.append(p)

    return parent, distance


def _simulate(board, positions, distances):
    n = board["n"]
    pieces = board["pieces"]
    targets = board["targets"]
    cells = board["static"].copy()
    nt = len(targets)

    # get_sprite_at returns the first matching sprite. In particular,
    # a bar takes priority over a reflector when their pixels overlap.
    for i, piece in enumerate(pieces):
        x, y = positions[i]
        code = 3 if piece["reflector"] else 2

        for ox, oy in piece["mask"]:
            px, py = x + ox, y + oy
            if 0 <= px < n and 0 <= py < n:
                index = py * n + px
                if cells[index] == 0:
                    cells[index] = code

    water = bytearray(n * n)
    front = []

    for x, y in board["sources"]:
        if 0 <= x < n and 0 <= y < n:
            water[y * n + x] = 1
            front.append((x, y, 0, 1))

    def at(x, y):
        if x < 0 or x >= n:
            return 1 if board["side_wall"] else 5
        if y < 0 or y >= n:
            return 5

        value = cells[y * n + x]
        if value:
            return value
        return 6 if water[y * n + x] else 0

    def put(x, y, dx, dy, new_front):
        if 0 <= x < n and 0 <= y < n:
            water[y * n + x] = 1
            new_front.append((x, y, dx, dy))

    captured = 0
    leak = False
    seen = set()
    visited_water = []
    loops = 0

    while front and loops < 500:
        loops += 1
        new_front = []

        for x, y, dx, dy in front:
            state = (x, y, dx, dy)
            if state in seen:
                continue
            seen.add(state)
            visited_water.append(state)

            sx, sy = (-1, 0) if dy else (0, -1)
            nx, ny = x + dx, y + dy
            value = at(nx, ny)

            if value == 0:
                put(nx, ny, dx, dy, new_front)
                continue

            if value == 6:
                new_front.append((nx, ny, dx, dy))
                continue

            if value == 2:
                for ax, ay in ((sx, sy), (-sx, -sy)):
                    if at(x + ax, y + ay) == 0:
                        put(x + ax, y + ay, dx, dy, new_front)
                continue

            if value >= 10:
                a = at(x + sx, y + sy)
                b = at(x - sx, y - sy)

                if a == value and b == value:
                    captured |= 1 << (value - 10)
                else:
                    for ax, ay in ((sx, sy), (-sx, -sy)):
                        if at(x + ax, y + ay) == 0:
                            put(x + ax, y + ay, dx, dy, new_front)
                continue

            if value == 3:
                a = at(x + sx, y + sy)
                b = at(x - sx, y - sy)

                if a == 3 and b == 0:
                    ndx, ndy = dy, -dx
                    put(x + ndx, y + ndy, ndx, ndy, new_front)

                if b == 3 and a == 0:
                    ndx, ndy = -dy, dx
                    put(x + ndx, y + ndy, ndx, ndy, new_front)
                else:
                    for ax, ay in ((sx, sy), (-sx, -sy)):
                        if at(x + ax, y + ay) == 0:
                            put(x + ax, y + ay, dx, dy, new_front)
                continue

            if value == 1:
                leak = True

        front = new_front

    won = captured == (1 << nt) - 1 and not leak and not front

    movement_cost = sum(
        distances[i].get(positions[i], 1000)
        for i in range(len(pieces))
    )

    proximity = 0.0
    for ti, options in enumerate(board["entries"]):
        if captured & (1 << ti):
            continue

        best = n * 3
        for x, y, dx, dy in visited_water:
            for ex, ey, di in options:
                ddx, ddy = _DIRS[di]
                d = abs(x - ex) + abs(y - ey)
                if (dx, dy) != (ddx, ddy):
                    d += 2
                if d < best:
                    best = d

        proximity += 16.0 / (1 + best)

    overlap = 0
    for i in range(len(pieces)):
        xi, yi = positions[i]
        pi = pieces[i]

        for j in range(i):
            xj, yj = positions[j]
            pj = pieces[j]

            if (xi < xj + pj["w"] and xi + pi["w"] > xj
                    and yi < yj + pj["h"] and yi + pi["h"] > yj):
                overlap += 1

    score = (
        100 * captured.bit_count()
        + proximity
        - 28 * int(leak)
        - 0.24 * movement_cost
        - 8 * overlap
    )

    # Prefer solutions that do not conceal one sprite beneath another.
    solution_cost = movement_cost + 15 * overlap
    return won, score, solution_cost


def _search(board, parents, distances, deadline, seed):
    pieces = board["pieces"]
    count = len(pieces)
    start = tuple((p["x"], p["y"]) for p in pieces)
    positions_list = [list(d.keys()) for d in distances]
    rng = random.Random(seed)
    cache = {}

    def evaluate(state):
        if state not in cache:
            cache[state] = _simulate(board, state, distances)
        return cache[state]

    start_result = evaluate(start)
    if start_result[0]:
        return start, True

    best = start
    best_score = start_result[1]
    best_solution = None
    best_solution_cost = float("inf")
    archive = [(best_score, start)]

    # Try every placement reachable by changing just one piece.
    for i in range(count):
        candidates = sorted(
            positions_list[i],
            key=lambda p: distances[i][p]
        )

        for p in candidates:
            if time.monotonic() >= deadline:
                break
            if p == start[i]:
                continue

            state = list(start)
            state[i] = p
            state = tuple(state)
            win, score, cost = evaluate(state)

            if score > best_score:
                best_score, best = score, state

            if win and cost < best_solution_cost:
                best_solution = state
                best_solution_cost = cost

        if time.monotonic() >= deadline:
            break

    if best_solution is not None:
        return best_solution, True

    archive.append((best_score, best))
    current = best
    current_score = best_score
    iterations = 0
    first_win_time = None

    while time.monotonic() < deadline:
        iterations += 1

        if iterations % 180 == 0:
            if rng.random() < 0.45:
                current = start
            elif archive:
                current = rng.choice(
                    archive[:min(12, len(archive))]
                )[1]
            current_score = evaluate(current)[1]

        i = rng.randrange(count)
        oldx, oldy = current[i]
        kind = rng.random()
        candidate = None

        if kind < 0.58:
            for _ in range(8):
                x = oldx + rng.randint(-4, 4)
                y = oldy + rng.randint(-3, 3)
                if (x, y) in distances[i]:
                    candidate = (x, y)
                    break

        elif kind < 0.80:
            anchors = [x for x, y in board["sources"]]
            for options in board["entries"]:
                anchors.extend(x for x, y, di in options)

            if anchors:
                ax = rng.choice(anchors)
                p = pieces[i]
                x = rng.choice((
                    ax,
                    ax - p["w"] + 1,
                    ax + 1,
                    ax - p["w"],
                    oldx,
                ))
                y = rng.randrange(3, board["n"])
                if (x, y) in distances[i]:
                    candidate = (x, y)

        if candidate is None:
            candidate = rng.choice(positions_list[i])

        if candidate == current[i]:
            continue

        state = list(current)
        state[i] = candidate
        state = tuple(state)
        win, score, cost = evaluate(state)

        if score > best_score:
            best, best_score = state, score
            archive.append((score, state))
            archive.sort(key=lambda z: z[0], reverse=True)
            if len(archive) > 25:
                archive.pop()

        if win and cost < best_solution_cost:
            best_solution = state
            best_solution_cost = cost
            if first_win_time is None:
                first_win_time = time.monotonic()

        temperature = 5.0 + 12.0 * (
            1.0 - (iterations % 180) / 180.0
        )
        delta = score - current_score
        if delta >= 0 or rng.random() < math.exp(
                max(-50.0, delta / temperature)):
            current, current_score = state, score

        if (first_win_time is not None
                and time.monotonic() - first_win_time > 0.4):
            break

    if best_solution is not None:
        return best_solution, True
    return best, False


def _path(parent, destination):
    result = []
    p = destination

    while parent[p] is not None:
        prev, direction = parent[p]
        result.append(direction)
        p = prev

    result.reverse()
    return result


def _display_move(direction, rotation):
    dx, dy = _DIRS[direction]

    for _ in range(rotation):
        dx, dy = dy, -dx

    if (dx, dy) == (0, -1):
        return "ACTION1"
    if (dx, dy) == (0, 1):
        return "ACTION2"
    if (dx, dy) == (-1, 0):
        return "ACTION3"
    return "ACTION4"


def _display_click(board, x, y):
    scale = board["scale"]
    offset = board["offset"]
    rotation = board["rotation"]

    x = offset + x * scale + scale // 2
    y = offset + y * scale + scale // 2

    if rotation == 1:
        x, y = y, 63 - x
    elif rotation == 2:
        x, y = 63 - x, 63 - y
    elif rotation == 3:
        x, y = 63 - y, x

    return int(y), int(x)


def _build_plan(board, destinations, parents, solved):
    pieces = board["pieces"]
    initial = [(p["x"], p["y"]) for p in pieces]
    changed = [
        i for i in range(len(pieces))
        if initial[i] != destinations[i]
    ]
    selected = next(
        (i for i, p in enumerate(pieces) if p["selected"]),
        None
    )

    if len(changed) <= 6:
        orders = permutations(changed)
    else:
        orders = [
            tuple(sorted(changed, key=lambda i: i != selected))
        ]

    best_plan = None

    for order in orders:
        positions = initial[:]
        current_selected = selected
        plan = []
        valid = True

        for i in order:
            piece = pieces[i]

            if current_selected != i:
                click = None
                x, y = positions[i]

                for ox, oy in piece["mask"]:
                    cx, cy = x + ox, y + oy
                    occupied = False

                    for j, other in enumerate(pieces):
                        if i == j:
                            continue
                        px, py = positions[j]

                        if (px <= cx < px + other["w"]
                                and py <= cy < py + other["h"]):
                            occupied = True
                            break

                    if not occupied:
                        click = (cx, cy)
                        break

                if click is None:
                    valid = False
                    break

                row, col = _display_click(board, *click)
                plan.append(("ACTION6", row, col))
                current_selected = i

            for direction in _path(parents[i], destinations[i]):
                plan.append((
                    _display_move(direction, board["rotation"]),
                    None,
                    None,
                ))

            positions[i] = destinations[i]

        if valid and (best_plan is None or len(plan) < len(best_plan)):
            best_plan = plan

    if best_plan is None:
        return None

    if solved:
        best_plan.append(("ACTION5", None, None))

    return best_plan


def _return_action(action, memory):
    if action[0] == "ACTION5":
        memory["awaiting_spill"] = True
    return action[0], action[1], action[2]


def act(obs, available, memory):
    # A successful spill ends the episode. If another observation arrives,
    # the predicted solution was wrong, often due to an obscured sprite.
    if memory.pop("awaiting_spill", False):
        memory.pop("plan", None)
        return "RESET", None, None

    pending = memory.get("plan")
    if pending:
        return _return_action(pending.pop(0), memory)

    started = time.monotonic()
    decoded = _decode(obs)

    if decoded is None:
        return _return_action(("ACTION5", None, None), memory)

    board = _make_board(decoded)
    if board is None:
        return _return_action(("ACTION5", None, None), memory)

    parents = []
    distances = []

    for piece in board["pieces"]:
        parent, distance = _reachable(board, piece)
        parents.append(parent)
        distances.append(distance)

    search_number = int(memory.get("searches", 0))
    seed = 13 + search_number * 100003

    for p in board["pieces"]:
        seed = (
            seed * 131 + p["x"] * 17 + p["y"] * 7 + p["w"]
        ) & 0xFFFFFFFF

    memory["searches"] = search_number + 1

    destination, solved = _search(
        board, parents, distances, started + 8.0, seed
    )
    plan = _build_plan(board, destination, parents, solved)

    if not plan:
        return _return_action(("ACTION5", None, None), memory)

    memory["plan"] = [list(a) for a in plan[1:]]
    return _return_action(plan[0], memory)

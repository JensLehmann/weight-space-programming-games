import math
import random
import time
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells = []
            while stack:
                px, py = stack.pop()
                cells.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py),
                               (px, py - 1), (px, py + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and mask[ny, nx] and (nx, ny) not in seen):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            result.append(cells)
    return result


def _box(cells):
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _read_board(obs):
    image = np.asarray(obs, dtype=np.int16)
    water_components = _components(image == 6)
    if not water_components:
        return None

    dimensions = []
    for cells in water_components:
        x0, y0, x1, y1 = _box(cells)
        dimensions.append(min(x1 - x0, y1 - y0))
    dimensions = [d for d in dimensions if 1 <= d <= 8]
    if not dimensions:
        return None
    scale = min(dimensions)

    # The yellow emitter is immediately above its blue source before rotation.
    votes = [0, 0, 0, 0]
    offsets = [(0, -scale), (-scale, 0),
               (0, scale), (scale, 0)]
    for cells in water_components:
        x0, y0, x1, y1 = _box(cells)
        if min(x1 - x0, y1 - y0) != scale:
            continue
        cx = (x0 + x1 - 1) // 2
        cy = (y0 + y1 - 1) // 2
        for k, (dx, dy) in enumerate(offsets):
            x, y = cx + dx, cy + dy
            if 0 <= x < 64 and 0 <= y < 64 and image[y, x] == 4:
                votes[k] += 1
    if not max(votes):
        return None
    rotation = max(range(4), key=lambda k: votes[k])
    raw = np.rot90(image, -rotation)

    yellow_y = np.where(raw == 4)[0]
    if len(yellow_y) == 0:
        return None
    top = int(yellow_y.min())
    # The progress indicator can overwrite the first display row.
    offset = 0 if top <= 1 else top

    sizes = [n for n in range(1, 65)
             if 64 // n == scale and (64 - n * scale) // 2 == offset]
    if not sizes:
        return None
    n = max(sizes)
    if offset + n * scale > 64:
        return None

    indices = offset + np.arange(n) * scale + scale // 2
    grid = raw[np.ix_(indices, indices)].copy()

    sources = [(x, y) for y in range(n) for x in range(n)
               if grid[y, x] == 6]
    if not sources:
        return None

    # Recover target shapes. Each target is a rotated five-pixel cup.
    template = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.int8)
    patterns = []
    for k in range(4):
        a = np.rot90(template, k)
        occupied = [(x, y) for y in range(a.shape[0])
                    for x in range(a.shape[1]) if a[y, x]]
        holes = [(x, y) for y in range(a.shape[0])
                 for x in range(a.shape[1]) if not a[y, x]]
        patterns.append((a.shape[1], a.shape[0], occupied, holes[0]))

    target_cells = {(x, y) for y in range(n) for x in range(n)
                    if grid[y, x] == 11}
    candidates = []
    by_cell = {}
    for w, h, shape, hole in patterns:
        for y in range(n - h + 1):
            for x in range(n - w + 1):
                pixels = frozenset((x + dx, y + dy) for dx, dy in shape)
                gap = (x + hole[0], y + hole[1])
                if pixels <= target_cells and gap not in target_cells:
                    entry = (pixels, (x, y, w, h), gap)
                    idx = len(candidates)
                    candidates.append(entry)
                    for p in pixels:
                        by_cell.setdefault(p, []).append(idx)

    def cover(remaining):
        if not remaining:
            return []
        p = min(remaining, key=lambda q: len(by_cell.get(q, ())))
        for idx in by_cell.get(p, ()):
            pixels = candidates[idx][0]
            if pixels <= remaining:
                rest = cover(remaining - pixels)
                if rest is not None:
                    return [idx] + rest
        return None

    chosen = cover(frozenset(target_cells))
    if chosen is None:
        return None
    targets = [candidates[i] for i in chosen]
    if not targets:
        return None

    # Movables are bars (8), elbows (15), and the selected piece (9).
    pieces = []

    def add_piece(cells, selected, kind):
        if not cells:
            return
        x0, y0, x1, y1 = _box(cells)
        shape = tuple(sorted((x - x0, y - y0) for x, y in cells))
        pieces.append({
            "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            "shape": shape, "selected": selected, "kind": kind
        })

    for color in (8, 9, 15):
        for cells in _components(grid == color):
            x0, y0, x1, y1 = _box(cells)
            w, h = x1 - x0, y1 - y0
            if color == 15 or (color == 9 and len(cells) == 3
                               and w == 2 and h == 2):
                add_piece(cells, color == 9, "elbow")
            elif w == 1 or h == 1:
                add_piece(cells, color == 9, "bar")
            else:
                # Adjacent bars of the same colour can form one component.
                rows = {}
                for x, y in cells:
                    rows.setdefault(y, []).append(x)
                for y, xs in sorted(rows.items()):
                    xs.sort()
                    run = [xs[0]]
                    for x in xs[1:]:
                        if x == run[-1] + 1:
                            run.append(x)
                        else:
                            add_piece([(xx, y) for xx in run],
                                      color == 9, "bar")
                            run = [x]
                    add_piece([(xx, y) for xx in run],
                              color == 9, "bar")

    # Click selection checks bars before elbows.
    pieces.sort(key=lambda p: (p["kind"] == "elbow", p["y"], p["x"]))
    if not pieces:
        return None

    selected = next((i for i, p in enumerate(pieces) if p["selected"]), None)
    if selected is None:
        selected = min(range(len(pieces)),
                       key=lambda i: pieces[i]["x"] ** 2 + pieces[i]["y"] ** 2)

    # Board values: 2 water, 3 forbidden wall, 4 other solid,
    # 10 bar, 11 elbow, 100+i target i.
    base = {}
    for y in range(n):
        for x in range(n):
            v = int(grid[y, x])
            if v == 1:
                base[(x, y)] = 3
            elif v == 4:
                base[(x, y)] = 4
            elif v == 6:
                base[(x, y)] = 2
    for i, (pixels, _, _) in enumerate(targets):
        for pos in pixels:
            base[pos] = 100 + i

    return {
        "grid": grid, "n": n, "scale": scale, "offset": offset,
        "rotation": rotation, "sources": sources, "targets": targets,
        "pieces": pieces, "selected": selected, "base": base
    }


def _reachable(board, piece):
    n = board["n"]
    base = board["base"]
    targets = board["targets"]
    w, h = piece["w"], piece["h"]
    start = (piece["x"], piece["y"])

    def valid(x, y):
        if x < 0 or y < 3 or x + w > n or y + h > n:
            return False
        for _, (rx, ry, rw, rh), _ in targets:
            if (x < rx + rw + 1 and x + w > rx - 1
                    and y < ry + rh + 1 and y + h > ry - 1):
                return False
        for dx, dy in piece["shape"]:
            if (x + dx, y + dy) in base:
                return False
        return True

    previous = {start: None}
    distance = {start: 0}
    queue = deque([start])
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]
    while queue:
        x, y = queue.popleft()
        for dx, dy, action in directions:
            p = (x + dx, y + dy)
            if p not in previous and valid(*p):
                previous[p] = ((x, y), action)
                distance[p] = distance[(x, y)] + 1
                queue.append(p)
    return previous, distance


def _simulate(board, positions):
    n = board["n"]
    pieces = board["pieces"]
    targets = board["targets"]
    occupied = board["base"].copy()

    for piece, (x, y) in zip(pieces, positions):
        kind = 10 if piece["kind"] == "bar" else 11
        for dx, dy in piece["shape"]:
            occupied.setdefault((x + dx, y + dy), kind)

    holes = [t[2] for t in targets]
    nearest = [100] * len(holes)
    frontier = [(x, y, 0, 1) for x, y in board["sources"]]
    reached = 0
    leak = False

    def note(x, y):
        for i, (gx, gy) in enumerate(holes):
            d = abs(x - gx) + abs(y - gy)
            if d < nearest[i]:
                nearest[i] = d

    for x, y in board["sources"]:
        note(x, y)

    def get(x, y):
        if x < 0 or y < 0 or x >= n or y >= n:
            return 4
        return occupied.get((x, y), 0)

    def spawn(x, y, dx, dy, dest):
        if get(x, y) == 0:
            occupied[(x, y)] = 2
            dest.append((x, y, dx, dy))
            note(x, y)

    total = 0
    for _ in range(250):
        if not frontier:
            break
        new = []
        for x, y, dx, dy in frontier:
            total += 1
            if total > 6000:
                return False, 0, True, nearest
            nx, ny = x + dx, y + dy
            value = get(nx, ny)
            if value == 0:
                spawn(nx, ny, dx, dy, new)
                continue
            if value == 2:
                new.append((nx, ny, dx, dy))
                continue

            if dy:
                side = ((-1, 0), (1, 0))
            else:
                side = ((0, -1), (0, 1))

            if value == 10:
                for sx, sy in side:
                    spawn(x + sx, y + sy, dx, dy, new)
                continue

            if value >= 100:
                a = get(x + side[0][0], y + side[0][1])
                b = get(x + side[1][0], y + side[1][1])
                if a == value and b == value:
                    reached |= 1 << (value - 100)
                else:
                    for sx, sy in side:
                        spawn(x + sx, y + sy, dx, dy, new)
                continue

            if value == 11:
                a = get(x + side[0][0], y + side[0][1])
                b = get(x + side[1][0], y + side[1][1])
                if a == 11 and b == 0:
                    spawn(x + dy, y - dx, dy, -dx, new)
                if b == 11 and a == 0:
                    spawn(x - dy, y + dx, -dy, dx, new)
                else:
                    for sx, sy in side:
                        spawn(x + sx, y + sy, dx, dy, new)
                continue

            if value == 3:
                leak = True
        frontier = new
    else:
        return False, reached, True, nearest

    success = not leak and reached == (1 << len(targets)) - 1
    return success, reached, leak, nearest


def _display_point(board, x, y):
    scale = board["scale"]
    offset = board["offset"]
    px = offset + x * scale + scale // 2
    py = offset + y * scale + scale // 2
    k = board["rotation"]
    if k == 1:
        px, py = py, 63 - px
    elif k == 2:
        px, py = 63 - px, 63 - py
    elif k == 3:
        px, py = 63 - py, px
    return py, px


def _make_plan(board, positions, previous, spill):
    pieces = board["pieces"]
    starts = [(p["x"], p["y"]) for p in pieces]
    moving = [i for i in range(len(pieces)) if positions[i] != starts[i]]
    selected = board["selected"]
    rotation = board["rotation"]

    # Internal action -> display action.
    action_maps = [
        {1: 1, 2: 2, 3: 3, 4: 4},
        {1: 3, 2: 4, 3: 2, 4: 1},
        {1: 2, 2: 1, 3: 4, 4: 3},
        {1: 4, 2: 3, 3: 1, 4: 2},
    ]

    def click_cell(index, current):
        p = pieces[index]
        x, y = current[index]
        for yy in range(y, y + p["h"]):
            for xx in range(x, x + p["w"]):
                blocked = False
                for j in range(index):
                    q = pieces[j]
                    qx, qy = current[j]
                    if qx <= xx < qx + q["w"] and qy <= yy < qy + q["h"]:
                        blocked = True
                        break
                if not blocked and 0 <= xx < board["n"] and 0 <= yy < board["n"]:
                    return xx, yy
        return None

    def path_for(i):
        p = positions[i]
        actions = []
        while previous[i][p] is not None:
            p, action = previous[i][p]
            actions.append(action)
        actions.reverse()
        return actions

    def search(current, remaining, current_selected):
        if not remaining:
            return []
        order = sorted(remaining, key=lambda i: (i != current_selected, i))
        for i in order:
            prefix = []
            if i != current_selected:
                point = click_cell(i, current)
                if point is None:
                    continue
                row, col = _display_point(board, *point)
                prefix.append(("ACTION6", row, col))
            for action in path_for(i):
                shown = action_maps[rotation][action]
                prefix.append(("ACTION%d" % shown, None, None))
            updated = list(current)
            updated[i] = positions[i]
            rest = search(updated, [j for j in remaining if j != i], i)
            if rest is not None:
                return prefix + rest
        return None

    plan = search(starts, moving, selected)
    if plan is not None and spill:
        plan.append(("ACTION5", None, None))
    return plan


def _search(board, memory):
    pieces = board["pieces"]
    previous = []
    distances = []
    choices = []
    for piece in pieces:
        prev, dist = _reachable(board, piece)
        previous.append(prev)
        distances.append(dist)
        choices.append(list(dist))

    initial = tuple((p["x"], p["y"]) for p in pieces)
    selected = board["selected"]
    target_count = len(board["targets"])

    def cost(state):
        total = 0
        for i, pos in enumerate(state):
            d = distances[i][pos]
            total += d
            if d and i != selected:
                total += 1
        return total

    def evaluate(state):
        success, reached, leak, near = _simulate(board, state)
        missing = target_count - reached.bit_count()
        movement = cost(state)
        score = (120.0 * missing + 90.0 * int(leak)
                 + 3.0 * sum(min(d, 25) for d in near)
                 + 0.55 * movement)
        return score, success, movement

    initial_score, initial_success, initial_cost = evaluate(initial)
    if initial_success:
        return [("ACTION5", None, None)]

    seed = hash((initial, tuple(board["sources"]), memory.get("attempt", 0)))
    rng = random.Random(seed)
    deadline = time.monotonic() + 8.0

    current = initial
    current_score = initial_score
    best = initial
    best_score = initial_score
    solution = None
    solution_cost = 10 ** 9
    solution_time = None
    iterations = 0

    while time.monotonic() < deadline:
        if solution_time is not None and time.monotonic() - solution_time > 1.0:
            break

        if iterations and iterations % 650 == 0:
            if rng.random() < 0.6:
                current, current_score = best, best_score
            else:
                current, current_score = initial, initial_score

        i = rng.randrange(len(pieces))
        if len(choices[i]) <= 1:
            iterations += 1
            continue

        old = current[i]
        candidate = None
        if rng.random() < 0.7:
            for _ in range(8):
                dx = rng.randint(-4, 4)
                dy = rng.randint(-4, 4)
                p = (old[0] + dx, old[1] + dy)
                if p in distances[i] and p != old:
                    candidate = p
                    break
        if candidate is None:
            candidate = rng.choice(choices[i])
        if candidate == old:
            iterations += 1
            continue

        trial_list = list(current)
        trial_list[i] = candidate
        trial = tuple(trial_list)
        score, success, movement = evaluate(trial)

        if success and movement < solution_cost:
            plan = _make_plan(board, trial, previous, True)
            if plan is not None:
                solution = plan
                solution_cost = len(plan)
                if solution_time is None:
                    solution_time = time.monotonic()

        if score < best_score:
            best, best_score = trial, score

        phase = (iterations % 650) / 650.0
        temperature = 25.0 * (1.0 - phase) + 1.5
        delta = score - current_score
        if delta <= 0 or rng.random() < math.exp(-min(delta / temperature, 50)):
            current, current_score = trial, score
        iterations += 1

    if solution is not None:
        return solution

    # Move toward the best partial arrangement, then search again.
    if best != initial:
        plan = _make_plan(board, best, previous, False)
        if plan:
            return plan

    # A harmless reselection allows another independent search on the next call.
    p = pieces[selected]
    row, col = _display_point(board, p["x"], p["y"])
    return [("ACTION6", row, col)]


def act(obs, available, memory):
    plan = memory.get("plan")
    if plan:
        action = plan.pop(0)
        return tuple(action)

    board = _read_board(obs)
    if board is None:
        # A reset is safer than starting an unmodelled spill.
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION5", None, None)

    memory["attempt"] = memory.get("attempt", 0) + 1
    plan = _search(board, memory)
    if not plan:
        return ("ACTION5", None, None)
    memory["plan"] = list(plan[1:])
    return tuple(plan[0])

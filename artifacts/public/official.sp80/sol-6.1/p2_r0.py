import math
import time
import random
from collections import deque
import numpy as np


_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        root = remaining.pop()
        comp = {root}
        stack = [root]
        while stack:
            x, y = stack.pop()
            for dx, dy in _DIRS:
                q = (x + dx, y + dy)
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _unrotate(obs):
    a = np.asarray(obs)
    edges = (a[0, :], a[:, 0], a[-1, :], a[:, -1])
    candidates = []
    for k, edge in enumerate(edges):
        if np.all((edge == 14) | (edge == 0)):
            candidates.append(k)
    k = candidates[0] if candidates else 0
    return np.rot90(a, -k).copy(), k


def _geometry(a):
    # Sprite boundaries have a common integer pixel scale and alignment.
    yy, xx = np.where(a[1:, 1:] != a[1:, :-1])
    xedges = xx + 1
    yy, xx = np.where(a[2:, :] != a[1:-1, :])
    yedges = yy + 2

    choices = []
    for s in range(2, 9):
        if len(xedges):
            xc = np.bincount(xedges % s, minlength=s)
            xr = int(np.argmax(xc))
            xf = float(xc[xr]) / len(xedges)
        else:
            xr, xf = 0, 1.0
        if len(yedges):
            yc = np.bincount(yedges % s, minlength=s)
            yr = int(np.argmax(yc))
            yf = float(yc[yr]) / len(yedges)
        else:
            yr, yf = 0, 1.0
        choices.append((min(xf, yf), s, xr, yr))

    good = [v for v in choices if v[0] >= 0.985]
    if good:
        _, s, xr, yr = max(good, key=lambda v: v[1])
    else:
        _, s, xr, yr = max(choices, key=lambda v: (v[0], v[1]))

    useful = a != 1
    useful[0, :] = False
    ys, xs = np.where(useful)
    xmin = int(xs.min()) if len(xs) else 0
    xmax = int(xs.max()) if len(xs) else 63
    ymin = int(ys.min()) if len(ys) else 1
    ymax = int(ys.max()) if len(ys) else 63

    sizes = []
    for w in range(6, 33):
        for h in range(6, 33):
            if min(64 // w, 64 // h) != s:
                continue
            ox = (64 - w * s) // 2
            oy = (64 - h * s) // 2
            if ox % s != xr or oy % s != yr:
                continue
            if not (ox <= xmin and xmax < ox + w * s):
                continue
            if not (oy <= ymin and ymax < oy + h * s):
                continue
            sizes.append((w * h, abs(w - h), w, h, ox, oy))

    if sizes:
        _, _, w, h, ox, oy = min(sizes)
    else:
        w = h = 64 // s
        ox = oy = (64 - w * s) // 2

    grid = np.empty((h, w), dtype=np.int8)
    for y in range(h):
        for x in range(w):
            py = min(63, oy + y * s + s // 2)
            px = min(63, ox + x * s + s // 2)
            grid[y, x] = a[py, px]
    return grid, s, ox, oy


def _cup_shapes(cup_points):
    shapes = []
    leftovers = set(cup_points)

    for comp in _components(cup_points):
        if len(comp) == 5:
            shapes.append(comp)
            leftovers.difference_update(comp)

    # Separate adjacent cups, if they formed one connected component.
    templates = []
    mask = {(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)}
    w, h = 3, 2
    for _ in range(4):
        templates.append((set(mask), w, h))
        mask = {(h - 1 - y, x) for x, y in mask}
        w, h = h, w

    while leftovers:
        found = None
        for x, y in sorted(leftovers, key=lambda q: (q[1], q[0])):
            for mask, w, h in templates:
                for mx, my in mask:
                    ox, oy = x - mx, y - my
                    candidate = {(ox + u, oy + v) for u, v in mask}
                    if not candidate <= leftovers:
                        continue
                    holes = {
                        (ox + u, oy + v)
                        for v in range(h) for u in range(w)
                    } - candidate
                    if holes & cup_points:
                        continue
                    found = candidate
                    break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            break
        shapes.append(found)
        leftovers.difference_update(found)
    return shapes, leftovers


def _read_scene(obs):
    a, rotation = _unrotate(obs)
    grid, scale, ox, oy = _geometry(a)
    h, w = grid.shape
    stride = w + 2
    size = (h + 2) * stride

    # 0 empty, 1 water, 2 splitter, 4 solid, 5 dangerous wall,
    # 6 outlet, >=10 cup identity, negative elbow identity.
    fixed = [4] * size
    for y in range(h):
        for x in range(w):
            fixed[(y + 1) * stride + x + 1] = 0

    cup_points = {
        (x, y) for y in range(h) for x in range(w)
        if int(grid[y, x]) in (11, 13)
    }
    cup_shapes, cup_leftovers = _cup_shapes(cup_points)
    cups = []
    for j, points in enumerate(cup_shapes):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        target = None
        for y in range(bbox[1], bbox[3]):
            for x in range(bbox[0], bbox[2]):
                if (x, y) in points:
                    continue
                for d, (dx, dy) in enumerate(_DIRS):
                    px, py = (-1, 0) if dy else (0, -1)
                    if ((x + dx, y + dy) in points
                            and (x + px, y + py) in points
                            and (x - px, y - py) in points):
                        target = (x, y, d)
                        break
                if target is not None:
                    break
            if target is not None:
                break
        if target is None:
            target = ((bbox[0] + bbox[2] - 1) // 2, bbox[1], 1)
        cups.append({"bbox": bbox, "target": target})
        for x, y in points:
            fixed[(y + 1) * stride + x + 1] = 10 + j

    sources = []
    outlets = []
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            p = (y + 1) * stride + x + 1
            if (x, y) in cup_points:
                if (x, y) in cup_leftovers:
                    fixed[p] = 4
                continue
            if color == 1:
                fixed[p] = 5
            elif color == 6:
                fixed[p] = 1
                sources.append(p)
            elif color == 4:
                fixed[p] = 6
                outlets.append(p)
            elif color not in (8, 9, 12, 15):
                fixed[p] = 4

    pieces = []
    selected = -1
    ambiguous = False
    for color in (9, 8, 15):
        points = {
            (x, y) for y in range(h) for x in range(w)
            if int(grid[y, x]) == color
        }
        for comp in _components(points):
            xs = [p[0] for p in comp]
            ys = [p[1] for p in comp]
            x0, y0 = min(xs), min(ys)
            pw, ph = max(xs) - x0 + 1, max(ys) - y0 + 1

            groups = [comp]
            if color == 8 and ph > 1 and pw > 2 and len(comp) == pw * ph:
                groups = [
                    {(x, y) for x, yy in comp if yy == y}
                    for y in range(y0, y0 + ph)
                ]
                ambiguous = True

            for group in groups:
                gx = min(p[0] for p in group)
                gy = min(p[1] for p in group)
                gw = max(p[0] for p in group) - gx + 1
                gh = max(p[1] for p in group) - gy + 1
                mask = sorted((x - gx, y - gy) for x, y in group)
                elbow = color == 15 or (
                    color == 9 and gw > 1 and gh > 1
                    and len(group) < gw * gh
                )
                piece = {
                    "pos": (gx, gy), "w": gw, "h": gh,
                    "mask": mask, "elbow": elbow
                }
                if color == 9:
                    selected = len(pieces)
                pieces.append(piece)

    # Partial overlap can hide the true length of an unselected bar.
    for y in range(h):
        for x in range(w - 1):
            if {int(grid[y, x]), int(grid[y, x + 1])} == {8, 9}:
                ambiguous = True

    return {
        "w": w, "h": h, "stride": stride, "fixed": fixed,
        "sources": sources, "outlets": outlets, "cups": cups,
        "pieces": pieces, "selected": selected,
        "rotation": rotation, "scale": scale, "ox": ox, "oy": oy,
        "ambiguous": ambiguous
    }


def _legal_positions(scene, piece):
    w, h, stride = scene["w"], scene["h"], scene["stride"]
    pw, ph = piece["w"], piece["h"]
    fixed = scene["fixed"]
    legal = set()

    for y in range(3, h - ph + 1):
        for x in range(w - pw + 1):
            blocked = False
            for cup in scene["cups"]:
                rx, ry, ex, ey = cup["bbox"]
                if (x < ex + 1 and x + pw > rx - 1
                        and y < ey + 1 and y + ph > ry - 1):
                    blocked = True
                    break
            if blocked:
                continue
            for mx, my in piece["mask"]:
                if fixed[(y + my + 1) * stride + x + mx + 1] != 0:
                    blocked = True
                    break
            if not blocked:
                legal.add((x, y))
    return legal


def _paths(legal, start):
    parent = {start: None}
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in _DIRS:
            p = (x + dx, y + dy)
            if p in legal and p not in parent:
                parent[p] = (x, y)
                dist[p] = dist[(x, y)] + 1
                q.append(p)
    return parent, dist


def _distance_tables(scene):
    stride = scene["stride"]
    size = len(scene["fixed"])
    tables = []
    terminal = [100] * (size * 4)

    for cup in scene["cups"]:
        tx, ty, td = cup["target"]
        table = [100] * (size * 4)
        tdx, tdy = _DIRS[td]
        for p in range(size):
            x, y = p % stride - 1, p // stride - 1
            for d, (dx, dy) in enumerate(_DIRS):
                if d == td:
                    cross = abs((x - tx) * tdy - (y - ty) * tdx)
                    past = max(0, (x - tx) * tdx + (y - ty) * tdy)
                    value = 2 * cross + 3 * past
                else:
                    opposite = dx == -tdx and dy == -tdy
                    value = abs(x - tx) + abs(y - ty) + (7 if opposite else 3)
                table[p * 4 + d] = value

                cross = abs((x - tx) * dy - (y - ty) * dx)
                tv = 2 * cross + (0 if d == td else 6)
                si = p * 4 + d
                if tv < terminal[si]:
                    terminal[si] = tv
        tables.append(table)
    return tables, terminal


def _flow(scene, layout, tables, terminal):
    board = scene["fixed"][:]
    stride = scene["stride"]
    pieces = scene["pieces"]

    # During a spill, ordinary splitters have priority over elbows.
    for elbow_pass in (True, False):
        for i, piece in enumerate(pieces):
            if piece["elbow"] != elbow_pass:
                continue
            x, y = layout[i]
            value = -1 - i if piece["elbow"] else 2
            for mx, my in piece["mask"]:
                board[(y + my + 1) * stride + x + mx + 1] = value

    q = [(p, 1) for p in scene["sources"]]
    for p in scene["outlets"]:
        below = p + stride
        if 0 <= below < len(board) and board[below] == 0:
            board[below] = 1
            q.append((below, 1))

    offsets = (-stride, stride, -1, 1)
    turn_a = (2, 3, 1, 0)
    turn_b = (3, 2, 0, 1)
    seen = bytearray(len(board) * 4)
    visited = []
    hazards = {}
    filled = 0
    head = 0

    while head < len(q):
        p, d = q[head]
        head += 1
        si = p * 4 + d
        if seen[si]:
            continue
        seen[si] = 1
        visited.append(si)

        nxt = p + offsets[d]
        if not 0 <= nxt < len(board):
            continue
        value = board[nxt]
        if value == 0:
            board[nxt] = 1
            q.append((nxt, d))
            continue
        if value == 1:
            q.append((nxt, d))
            continue

        side = 1 if d < 2 else stride
        pa, pb = p - side, p + side
        if not (0 <= pa < len(board) and 0 <= pb < len(board)):
            continue

        if value >= 10:
            if board[pa] == value and board[pb] == value:
                filled |= 1 << (value - 10)
                continue
            for np_ in (pa, pb):
                if board[np_] == 0:
                    board[np_] = 1
                    q.append((np_, d))
            continue

        if value == 2:
            for np_ in (pa, pb):
                if board[np_] == 0:
                    board[np_] = 1
                    q.append((np_, d))
            continue

        if value < 0:
            va, vb = board[pa], board[pb]
            if va == value and vb == 0:
                nd = turn_a[d]
                np_ = p + offsets[nd]
                board[np_] = 1
                q.append((np_, nd))
            if vb == value and va == 0:
                nd = turn_b[d]
                np_ = p + offsets[nd]
                board[np_] = 1
                q.append((np_, nd))
            else:
                for np_ in (pa, pb):
                    if board[np_] == 0:
                        board[np_] = 1
                        q.append((np_, d))
            continue

        if value == 5:
            hazards[nxt] = si

    count = len(scene["cups"])
    missing = count - filled.bit_count()
    success = missing == 0 and not hazards
    if success:
        return True, 0.0

    distance = 0
    for j, table in enumerate(tables):
        if not (filled >> j) & 1:
            distance += min((table[si] for si in visited), default=40)
    leak_distance = sum(terminal[si] for si in hazards.values())
    score = (
        120 * missing + 80 * len(hazards)
        + 12 * distance + 2 * leak_distance
    )
    return False, float(score)


def _search(scene, memory, deadline):
    pieces = scene["pieces"]
    initial = tuple(tuple(p["pos"]) for p in pieces)
    n = len(pieces)
    rng = random.Random(
        9173 + 1009 * memory.get("search_count", 0)
        + sum((i + 1) * (x + 37 * y) for i, (x, y) in enumerate(initial))
    )
    memory["search_count"] = memory.get("search_count", 0) + 1

    domains, distances = [], []
    for piece in pieces:
        if "legal" not in piece:
            piece["legal"] = _legal_positions(scene, piece)
        _, dist = _paths(piece["legal"], tuple(piece["pos"]))
        distances.append(dist)
        domains.append(list(dist))

    tables, terminal = _distance_tables(scene)
    cache = {}
    best_layout = initial
    best_value = float("inf")
    best_win = False
    selected = scene["selected"]

    def evaluate(layout):
        nonlocal best_layout, best_value, best_win
        key = tuple(layout)
        if key in cache:
            return cache[key]
        cost = 1
        for i, p in enumerate(key):
            steps = distances[i].get(p, 1000)
            cost += steps
            if steps and i != selected:
                cost += 1
        win, raw = _flow(scene, key, tables, terminal)
        value = -100000.0 + cost if win else raw + 0.12 * cost
        result = (value, win)
        if len(cache) < 70000:
            cache[key] = result
        if value < best_value:
            best_value = value
            best_layout = key
            best_win = win
        return result

    evaluate(initial)
    if best_win or n == 0:
        return best_layout, best_win

    seeds = []
    for seed in memory.get("seeds", []):
        seed = tuple(tuple(p) for p in seed)
        if len(seed) == n and all(seed[i] in distances[i] for i in range(n)):
            seeds.append(seed)

    local_minima = []
    trial = 0
    win_passes = 0

    while time.perf_counter() < deadline:
        if best_win:
            layout = list(best_layout)
            win_passes += 1
            if win_passes > 2:
                break
        elif trial == 0:
            layout = list(initial)
        elif seeds:
            layout = list(seeds.pop(0))
        else:
            if trial % 3:
                layout = list(best_layout)
                changed = rng.sample(range(n), rng.randint(1, min(n, 3)))
            else:
                layout = list(initial)
                changed = list(range(n))
            for i in changed:
                if domains[i]:
                    layout[i] = rng.choice(domains[i])
        trial += 1

        current_value, _ = evaluate(layout)
        for _ in range(3):
            improved = False
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                old = layout[i]
                chosen = old
                chosen_value = current_value
                candidates = domains[i][:]
                rng.shuffle(candidates)
                for ci, candidate in enumerate(candidates):
                    if ci % 16 == 0 and time.perf_counter() >= deadline:
                        break
                    if candidate == old:
                        continue
                    layout[i] = candidate
                    value, _ = evaluate(layout)
                    if value < chosen_value:
                        chosen_value = value
                        chosen = candidate
                layout[i] = chosen
                if chosen != old:
                    improved = True
                current_value = chosen_value
                if time.perf_counter() >= deadline:
                    break
            if not improved or time.perf_counter() >= deadline:
                break
        local_minima.append((current_value, tuple(layout)))
        if best_win and win_passes >= 1 and not improved:
            break

    unique = {}
    for value, layout in local_minima:
        unique[layout] = min(value, unique.get(layout, float("inf")))
    memory["seeds"] = [
        layout for layout, value in
        sorted(unique.items(), key=lambda kv: kv[1])[:8]
    ]

    if not best_win and best_layout == initial:
        alternatives = [
            (value, layout) for layout, value in unique.items()
            if layout != initial
        ]
        if alternatives:
            best_layout = min(alternatives)[1]
        else:
            # One legal exploratory step provides another planning turn.
            for i in range(n):
                near = [p for p, d in distances[i].items() if d == 1]
                if near:
                    layout = list(initial)
                    layout[i] = rng.choice(near)
                    best_layout = tuple(layout)
                    break

    return best_layout, best_win


def _display_point(scene, x, y):
    px = scene["ox"] + x * scene["scale"] + scene["scale"] // 2
    py = scene["oy"] + y * scene["scale"] + scene["scale"] // 2
    k = scene["rotation"]
    if k == 0:
        return py, px
    if k == 1:
        return 63 - px, py
    if k == 2:
        return 63 - py, 63 - px
    return px, 63 - py


def _click_cell(scene, index):
    pieces = scene["pieces"]
    piece = pieces[index]
    x, y = piece["pos"]
    candidates = [(x + mx, y + my) for mx, my in piece["mask"]]
    candidates += [
        (x + mx, y + my)
        for my in range(piece["h"]) for mx in range(piece["w"])
    ]
    for cx, cy in candidates:
        covered = False
        for j, other in enumerate(pieces):
            if j == index:
                continue
            bx, by = other["pos"]
            if (bx <= cx < bx + other["w"]
                    and by <= cy < by + other["h"]):
                covered = True
                break
        if not covered:
            return cx, cy
    return None


def _execute(scene, goals, available):
    pieces = scene["pieces"]
    moving = [
        i for i, piece in enumerate(pieces)
        if tuple(piece["pos"]) != tuple(goals[i])
    ]
    if not moving:
        return ("ACTION5", None, None)

    selected = scene["selected"]
    if selected in moving:
        index = selected
    else:
        selectable = [(i, _click_cell(scene, i)) for i in moving]
        selectable = [(i, p) for i, p in selectable if p is not None]
        if not selectable:
            return None
        index, cell = min(
            selectable,
            key=lambda ip: (
                abs(pieces[ip[0]]["pos"][0] - goals[ip[0]][0])
                + abs(pieces[ip[0]]["pos"][1] - goals[ip[0]][1])
            )
        )
        row, col = _display_point(scene, *cell)
        scene["selected"] = index
        return ("ACTION6", int(row), int(col))

    piece = pieces[index]
    start = tuple(piece["pos"])
    parent, _ = _paths(piece["legal"], start)
    goal = tuple(goals[index])
    if goal not in parent:
        return None
    nxt = goal
    while parent[nxt] != start:
        nxt = parent[nxt]
        if nxt is None:
            return None

    dx, dy = nxt[0] - start[0], nxt[1] - start[1]
    canonical = 1 if dy < 0 else 2 if dy > 0 else 3 if dx < 0 else 4
    inverse = (
        (0, 1, 2, 3, 4),
        (0, 3, 4, 2, 1),
        (0, 2, 1, 4, 3),
        (0, 4, 3, 1, 2),
    )
    action = "ACTION" + str(inverse[scene["rotation"]][canonical])
    piece["pos"] = nxt
    return action, None, None


def act(obs, available, memory):
    started = time.perf_counter()
    available = set(available)

    # A successful spill ends the episode. If called again, its model was
    # inaccurate; recover rather than repeatedly testing the same arrangement.
    if memory.pop("released", False):
        memory.pop("scene", None)
        memory.pop("goals", None)
        memory.pop("seeds", None)
        if "RESET" in available and not memory.get("recovered", False):
            memory["recovered"] = True
            return "RESET", None, None

    scene = memory.get("scene")
    if scene is None:
        scene = _read_scene(obs)
        if (scene["ambiguous"] and not memory.get("ambiguity_reset", False)
                and "RESET" in available):
            memory["ambiguity_reset"] = True
            return "RESET", None, None
        memory["scene"] = scene

    goals = memory.get("goals")
    if goals is None:
        goals, winning = _search(scene, memory, started + 8.35)
        if winning:
            memory["goals"] = goals
        else:
            # Execute only a movement toward a useful partial arrangement;
            # do not spend a spill attempt on a configuration known to fail.
            current = tuple(tuple(p["pos"]) for p in scene["pieces"])
            if tuple(goals) == current:
                selected = scene["selected"]
                if 0 <= selected < len(scene["pieces"]):
                    cell = _click_cell(scene, selected)
                    if cell is not None and "ACTION6" in available:
                        row, col = _display_point(scene, *cell)
                        return "ACTION6", int(row), int(col)
                if "RESET" in available:
                    memory.pop("scene", None)
                    return "RESET", None, None

    answer = _execute(scene, goals, available)
    if answer is None:
        memory.pop("scene", None)
        memory.pop("goals", None)
        memory.pop("seeds", None)
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION5", None, None

    if answer[0] not in available:
        memory.pop("scene", None)
        memory.pop("goals", None)
        if "RESET" in available:
            return "RESET", None, None
        for name in sorted(available):
            if name != "ACTION6":
                return name, None, None

    if answer[0] == "ACTION5":
        memory["released"] = True
    return answer

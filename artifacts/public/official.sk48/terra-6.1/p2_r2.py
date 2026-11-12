import heapq
import itertools
import time

import numpy as np


_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_ACTION_FOR_DIR = {
    (0, -1): "ACTION1",
    (0, 1): "ACTION2",
    (-1, 0): "ACTION3",
    (1, 0): "ACTION4",
}

# 20 = outline, 21 = inner ring, 22 = center, -1 = aperture.
_HEAD = np.array([
    [20, 20, 20, 20, 20, 20],
    [20, 21, 21, 21, 21, 20],
    [20, 21, 22, 22, 21, -1],
    [20, 21, 22, 22, 21, -1],
    [20, 21, 21, 21, 21, 20],
    [20, 20, 20, 20, 20, 20],
], dtype=int)

_LINE = np.array([
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
    [3, 2, 2, 3, 2, 2],
    [2, 2, 3, 2, 2, 3],
    [-1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
], dtype=int)

_HEADS = [np.rot90(_HEAD, -i) for i in range(4)]
_LINES = [np.rot90(_LINE, -i) for i in range(4)]


def _find_heads(a):
    heads = []
    for y in range(59):
        for x in range(59):
            q = int(a[y, x])
            if q in (0, 1, 2, 3, 4):
                continue
            if (int(a[y, x + 5]) != q or
                    int(a[y + 5, x]) != q or
                    int(a[y + 5, x + 5]) != q):
                continue
            center = int(a[y + 2, x + 2])
            if not np.all(a[y + 2:y + 4, x + 2:x + 4] == center):
                continue
            if center != q and not (q == 5 and center == 11):
                continue

            patch = a[y:y + 6, x:x + 6]
            found = None
            for rotation, template in enumerate(_HEADS):
                line = _LINES[rotation]
                apertures = np.argwhere(template == -1)
                for ring in ((5,) if q == 5 else (0, 4)):
                    if not np.all(patch[template == 20] == q):
                        continue
                    if not np.all(patch[template == 21] == ring):
                        continue
                    if not np.all(patch[template == 22] == center):
                        continue
                    selected = ring == 0
                    good = True
                    for yy, xx in apertures:
                        expected = int(line[yy, xx])
                        if selected:
                            expected -= 1
                        if int(patch[yy, xx]) != expected:
                            good = False
                            break
                    if good:
                        found = {
                            "x": x, "y": y, "rotation": rotation,
                            "color": center, "outline": q,
                            "selected": selected, "clickable": q != 5,
                        }
                        break
                if found is not None:
                    break
            if found is not None:
                heads.append(found)
    return heads


def _find_tiles(a):
    tiles = {}
    for y in range(59):
        for x in range(59):
            c = int(a[y + 1, x + 1])
            if c < 6:
                continue
            if x > 0 and int(a[y + 1, x]) == c:
                continue
            if y > 0 and int(a[y, x + 1]) == c:
                continue
            p = a[y + 1:y + 5, x + 1:x + 5]
            if (np.all(p[0, :] == c) and np.all(p[3, :] == c) and
                    np.all(p[:, 0] == c) and np.all(p[:, 3] == c)):
                tiles[(x, y)] = c
    return tiles


def _line_length(a, head):
    rotation = head["rotation"]
    dx, dy = _DIRS[rotation]
    line = _LINES[rotation]
    selected = head["selected"]

    # The two pixels at the end of a segment remain visible even when a
    # colored tile covers the middle of the segment.
    if rotation == 0:
        samples = ((2, 5), (3, 5))
    elif rotation == 1:
        samples = ((5, 2), (5, 3))
    elif rotation == 2:
        samples = ((2, 0), (3, 0))
    else:
        samples = ((0, 2), (0, 3))

    length = 0
    for i in range(16):
        x = head["x"] + 6 * dx * i
        y = head["y"] + 6 * dy * i
        if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
            break
        good = True
        for yy, xx in samples:
            expected = int(line[yy, xx])
            if selected:
                expected -= 1
            if int(a[y + yy, x + xx]) != expected:
                good = False
                break
        if not good:
            break
        length += 1
    return max(1, length)


def _board_bounds(a):
    counts = np.count_nonzero(a[:53, :] == 4, axis=1)
    mx = int(counts.max())
    if mx < 12:
        return None
    threshold = max(8, int(mx * 0.42))
    ys = [i for i, n in enumerate(counts) if int(n) >= threshold]
    if not ys:
        return None
    y0, y1 = min(ys), max(ys) + 1
    col_counts = np.count_nonzero(a[y0:y1, :] == 4, axis=0)
    mc = int(col_counts.max())
    xs = [i for i, n in enumerate(col_counts)
          if int(n) >= max(5, int(mc * 0.35))]
    if not xs:
        return None
    x0, x1 = min(xs), max(xs) + 1
    if (x1 - x0) % 6 or (y1 - y0) % 6:
        return None
    return x0, y0, (x1 - x0) // 6, (y1 - y0) // 6


def _parse(a):
    bounds = _board_bounds(a)
    if bounds is None:
        return None
    bx, by, width, height = bounds
    if not (1 <= width <= 10 and 1 <= height <= 10):
        return None

    heads = _find_heads(a)
    all_tiles = _find_tiles(a)
    upper = [h for h in heads if h["y"] < 53]
    lower = [h for h in heads if h["y"] >= 53]
    if not upper or not lower:
        return None

    for h in heads:
        h["length"] = _line_length(a, h)

    # A color identifies a pair of handles. The unpaired black handle, if
    # present, is an immovable arm that still participates in collisions.
    upper.sort(key=lambda h: (h["color"], h["rotation"], h["x"], h["y"]))
    goals = []
    for h in upper:
        partner = next((p for p in lower if p["color"] == h["color"]), None)
        if partner is None:
            goals.append(None)
            continue
        dx, dy = _DIRS[partner["rotation"]]
        seq = []
        for j in range(partner["length"]):
            p = (partner["x"] + 6 * dx * j,
                 partner["y"] + 6 * dy * j)
            if p in all_tiles:
                seq.append(all_tiles[p])
        goals.append(tuple(seq))

    if not any(g is not None for g in goals):
        return None

    machine_data = []
    active = -1
    flat = []
    for i, h in enumerate(upper):
        if (h["x"] - bx) % 6 or (h["y"] - by) % 6:
            return None
        x = (h["x"] - bx) // 6
        y = (h["y"] - by) // 6
        machine_data.append({
            "direction": _DIRS[h["rotation"]],
            "clickable": h["clickable"] and goals[i] is not None,
            "goal": goals[i],
        })
        flat.extend((x, y, h["length"]))
        if h["selected"]:
            active = i

    if active < 0:
        active = next((i for i, m in enumerate(machine_data)
                       if m["clickable"]), 0)

    board_tiles = []
    for (x, y), c in all_tiles.items():
        if (x - bx) % 6 or (y - by) % 6:
            continue
        gx, gy = (x - bx) // 6, (y - by) // 6
        if 0 <= gx < width and 0 <= gy < height:
            board_tiles.append((c, gy * width + gx))
    board_tiles.sort()
    colors = tuple(c for c, _ in board_tiles)
    flat.extend(p for _, p in board_tiles)
    flat.append(active)

    obstacles = set()
    for gy in range(height):
        for gx in range(width):
            x, y = bx + 6 * gx, by + 6 * gy
            if np.count_nonzero(a[y:y + 6, x:x + 6] == 5) >= 28:
                obstacles.add((gx, gy))

    model = {
        "bx": bx, "by": by, "w": width, "h": height,
        "machines": machine_data, "colors": colors,
        "obstacles": obstacles,
    }
    return model, tuple(flat)


def _unpack(model, state):
    n = len(model["machines"])
    machines = [tuple(state[3 * i:3 * i + 3]) for i in range(n)]
    positions = state[3 * n:3 * n + len(model["colors"])]
    return machines, positions, state[-1]


def _pack(model, machines, positions, active):
    pairs = sorted(zip(model["colors"], positions))
    result = []
    for m in machines:
        result.extend(m)
    result.extend(p for _, p in pairs)
    result.append(active)
    return tuple(result)


def _is_goal(model, state):
    machines, positions, _ = _unpack(model, state)
    w = model["w"]
    tiles = {(p % w, p // w): c
             for c, p in zip(model["colors"], positions)}
    for i, spec in enumerate(model["machines"]):
        goal = spec["goal"]
        if goal is None:
            continue
        x, y, length = machines[i]
        dx, dy = spec["direction"]
        seq = []
        for j in range(length):
            c = tiles.get((x + j * dx, y + j * dy))
            if c is not None:
                seq.append(c)
        if len(seq) < len(goal) or tuple(seq[:len(goal)]) != goal:
            return False
    return True


def _heuristic(model, state):
    machines, positions, active = _unpack(model, state)
    w = model["w"]
    tiles = [(c, p % w, p // w)
             for c, p in zip(model["colors"], positions)]
    at = {(x, y): c for c, x, y in tiles}
    score = 0.0

    for i, spec in enumerate(model["machines"]):
        goal = spec["goal"]
        if not goal:
            continue
        x, y, length = machines[i]
        dx, dy = spec["direction"]
        seq = []
        for j in range(length):
            c = at.get((x + dx * j, y + dy * j))
            if c is not None:
                seq.append(c)

        matched = 0
        while (matched < len(goal) and matched < len(seq) and
               goal[matched] == seq[matched]):
            matched += 1
        missing = len(goal) - matched
        if not missing:
            continue
        score += 2.8 * missing
        if matched < len(seq):
            score += 1.5

        wanted = goal[matched]
        best = 30
        for c, tx, ty in tiles:
            if c != wanted:
                continue
            along = (tx - x) * dx + (ty - y) * dy
            across = abs((tx - x) * dy - (ty - y) * dx)
            extension = max(0, along - length + 1)
            behind = max(0, -along)
            best = min(best, across + extension + behind)
        score += 0.8 * best
        if active != i:
            score += 0.3
    return score


def _move(model, state, direction):
    machines, positions, active = _unpack(model, state)
    specs = model["machines"]
    w, h = model["w"], model["h"]
    obstacles = model["obstacles"]
    dx, dy = direction
    ax, ay, length = machines[active]
    fx, fy = specs[active]["direction"]

    forward = (dx, dy) == (fx, fy)
    backward = (dx, dy) == (-fx, -fy)
    if backward and length <= 1:
        return None
    if not forward and not backward:
        nx, ny = ax + dx, ay + dy
        # Handles run along the side of the rectangular playing field.
        if fx:
            if not (0 <= ny < h):
                return None
        else:
            if not (0 <= nx < w):
                return None

    def blocked(x, y):
        return not (0 <= x < w and 0 <= y < h) or (x, y) in obstacles

    if forward:
        ex = ax + (length - 1) * fx
        ey = ay + (length - 1) * fy
        if blocked(ex + dx, ey + dy):
            return None

    segments = {}
    for mi, (mx, my, ml) in enumerate(machines):
        sx, sy = specs[mi]["direction"]
        for j in range(ml):
            segments[(mi, j)] = (mx + j * sx, my + j * sy, sx, sy)

    # Querying an arm at a location is orientation-sensitive.
    def segment_at(x, y, horizontal, ignore=None):
        for sid, (sx, sy, sdx, sdy) in segments.items():
            if sid == ignore:
                continue
            if sx == x and sy == y and ((sdx != 0) == horizontal):
                return sid
        return None

    tile_at = {(p % w, p // w): i for i, p in enumerate(positions)}
    moved_segments = set()
    moved_tiles = set()

    def move_tile(ti, origin=None):
        if ti in moved_tiles:
            return True
        p = positions[ti]
        x, y = p % w, p // w
        if blocked(x + dx, y + dy):
            return False

        # Perpendicular arms prevent a tile from passing through them.
        perpendicular_horizontal = (dx == 0)
        for xx, yy in ((x, y), (x + dx, y + dy)):
            if segment_at(xx, yy, perpendicular_horizontal, origin) is not None:
                return False

        other = tile_at.get((x + dx, y + dy))
        if other is not None and not move_tile(other):
            return False
        moved_tiles.add(ti)
        return True

    def move_segment(sid):
        if sid in moved_segments:
            return True
        x, y, sx, sy = segments[sid]
        if blocked(x + dx, y + dy):
            if not ((sx, sy) == (-dx, -dy) or blocked(x, y)):
                return False
        for xx, yy in ((x, y), (x + dx, y + dy)):
            ti = tile_at.get((xx, yy))
            if ti is not None:
                if not move_tile(ti, sid):
                    if (sx == 0) != (dx == 0):
                        return False
        moved_segments.add(sid)
        return True

    first = 1 if backward else 0
    for j in range(first, length):
        if not move_segment((active, j)):
            return None

    new_positions = list(positions)
    for ti in moved_tiles:
        p = positions[ti]
        new_positions[ti] = (p // w + dy) * w + (p % w + dx)

    new_machines = list(machines)
    if forward:
        new_machines[active] = (ax, ay, length + 1)
    elif backward:
        new_machines[active] = (ax, ay, length - 1)
    else:
        new_machines[active] = (ax + dx, ay + dy, length)

    result = _pack(model, new_machines, new_positions, active)
    return result if result != state else None


def _successors(model, state):
    machines, _, active = _unpack(model, state)
    for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nxt = _move(model, state, d)
        if nxt is not None:
            yield nxt, (_ACTION_FOR_DIR[d], None, None)

    for i, spec in enumerate(model["machines"]):
        if i == active or not spec["clickable"]:
            continue
        x, y, _ = machines[i]
        px = model["bx"] + 6 * x + 1
        py = model["by"] + 6 * y + 1
        if 0 <= px < 64 and 0 <= py < 64:
            nxt = list(state)
            nxt[-1] = i
            yield tuple(nxt), ("ACTION6", py, px)


def _search(model, start, recent):
    deadline = time.monotonic() + 7.5
    counter = itertools.count()
    h0 = _heuristic(model, start)
    heap = [(2.4 * h0, next(counter), 0, start)]
    costs = {start: 0}
    parents = {}
    best = None
    best_rank = (h0, float("inf"))
    recent_set = set(recent)
    expansions = 0

    while heap and expansions < 120000:
        if expansions % 128 == 0 and time.monotonic() >= deadline:
            break
        _, _, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        expansions += 1

        if state != start and _is_goal(model, state):
            best = state
            break

        if state != start:
            hs = _heuristic(model, state)
            rank = (hs + (1.5 if state in recent_set else 0), g)
            if rank < best_rank:
                best_rank = rank
                best = state

        for nxt, action in _successors(model, state):
            ng = g + 1
            if ng >= costs.get(nxt, 10 ** 9):
                continue
            costs[nxt] = ng
            parents[nxt] = (state, action)
            hn = _heuristic(model, nxt)
            heapq.heappush(heap, (ng + 2.4 * hn, next(counter), ng, nxt))

    if best is None:
        return []
    path = []
    cur = best
    while cur != start:
        previous, action = parents[cur]
        path.append([action[0], action[1], action[2], list(cur)])
        cur = previous
    path.reverse()
    return path


def act(obs, available, memory):
    try:
        parsed = _parse(np.asarray(obs))
        if parsed is None:
            if not memory.get("parse_reset") and "RESET" in available:
                memory["parse_reset"] = True
                return "RESET", None, None
            return "ACTION1", None, None

        model, state = parsed
        expected = memory.get("expected")
        plan = memory.get("plan", [])

        if expected is not None and tuple(expected) != state:
            plan = []
        if plan and plan[0][0] not in available:
            plan = []

        recent = [tuple(s) for s in memory.get("recent", [])]
        recent.append(state)
        recent = recent[-24:]
        memory["recent"] = [list(s) for s in recent]

        if not plan:
            plan = _search(model, state, recent[:-1])

        if plan:
            name, row, col, predicted = plan.pop(0)
            memory["plan"] = plan
            memory["expected"] = predicted
            return name, row, col

        # If the current arrangement already satisfies the goal, or search
        # found no useful partial plan, make a legal modeled movement.
        choices = list(_successors(model, state))
        choices = [(s, a) for s, a in choices if a[0] in available]
        if choices:
            choices.sort(key=lambda item: (
                item[0] in set(recent),
                _heuristic(model, item[0])
            ))
            nxt, action = choices[0]
            memory["plan"] = []
            memory["expected"] = list(nxt)
            return action

        if "RESET" in available and not memory.get("stuck_reset"):
            memory["stuck_reset"] = True
            memory["expected"] = None
            return "RESET", None, None
        return "ACTION1", None, None
    except Exception:
        # Keep the player operational if an unfamiliar rendering cannot be
        # interpreted by the model.
        memory["plan"] = []
        memory["expected"] = None
        if "RESET" in available and not memory.get("error_reset"):
            memory["error_reset"] = True
            return "RESET", None, None
        return "ACTION1", None, None

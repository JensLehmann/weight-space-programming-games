import numpy as np
import heapq
import itertools
import time
from collections import deque

_PLAYER = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=bool)

_GOAL = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

_BUTTON = np.array([
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
], dtype=bool)


def _matches(image, mask, color, exact=True):
    h, w = image.shape
    mh, mw = mask.shape
    if mh > h or mw > w:
        return []
    out = np.ones((h - mh + 1, w - mw + 1), dtype=bool)
    for yy, xx in np.argwhere(mask):
        out &= image[yy:yy + out.shape[0],
                     xx:xx + out.shape[1]] == color
        if not out.any():
            return []
    if exact:
        for yy, xx in np.argwhere(~mask):
            out &= image[yy:yy + out.shape[0],
                         xx:xx + out.shape[1]] != color
            if not out.any():
                return []
    return [(int(x), int(y)) for y, x in np.argwhere(out)]


def _objects(image, base, max_scale=4, loose=False):
    results = []
    colors = (11, 8, 9, 15, 14)
    for scale in range(1, max_scale + 1):
        for rotation in (0, 90, 180, 270):
            mask = np.rot90(base, -(rotation // 90))
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
            for color in colors:
                for x, y in _matches(image, mask, color, not loose):
                    patch = image[y:y + mask.shape[0],
                                  x:x + mask.shape[1]]
                    holes = int(np.sum(patch[~mask] != color))
                    results.append(
                        (holes, x, y, rotation, scale, color,
                         mask.shape[1], mask.shape[0])
                    )
    return results


def _program(obs):
    cells = {}
    allowed = np.zeros((6, 5), dtype=bool)
    allowed[1, 1:4] = True
    allowed[3:6, 2] = True

    for y in range(28, 48):
        for x in range(60):
            p = obs[y:y + 6, x:x + 5]
            if p.shape != (6, 5):
                continue
            hc, vc = int(p[1, 2]), int(p[4, 2])
            if hc not in (1, 5) or vc not in (1, 5):
                continue
            if not np.all(p[1, 1:4] == hc):
                continue
            if not np.all(p[3:6, 2] == vc):
                continue
            if not np.all(p[~allowed] == 0):
                continue
            cells[(x, y)] = (hc == 5, vc == 5)

    if not cells:
        return None

    # Find the largest regular, connected bank of instruction cells.
    remaining = set(cells)
    groups = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for q in ((x - 5, y), (x + 5, y),
                      (x, y - 6), (x, y + 6)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)

    group = max(groups, key=lambda g: (len(g), max(x for x, y in g)))
    xs = sorted({x for x, y in group})
    ys = sorted({y for x, y in group})
    xs = [x for x in xs if all((x, y) in group for y in ys)]
    if not xs or not ys:
        return None

    values, clicks = [], []
    for x in xs:
        value = 0
        column_clicks = []
        for j, y in enumerate(ys):
            h, v = cells[(x, y)]
            value |= int(h) << (2 * j)
            value |= int(v) << (2 * j + 1)
            column_clicks.extend([(y + 1, x + 2), (y + 4, x + 2)])
        values.append(value)
        clicks.append(column_clicks)
    return values, clicks, min(ys), min(xs), max(xs) + 5


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        points = []
        while stack:
            xx, yy = stack.pop()
            points.append((xx, yy))
            for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                           (xx, yy - 1), (xx, yy + 1)):
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(points)
    return result


def _read_scene(obs, panel, memory):
    values, clicks, top, left, right = panel
    image = obs[:top].copy()

    goals = _objects(image, _GOAL)
    if not goals:
        goals = _objects(image, _GOAL, loose=True)
    if not goals:
        return None

    # The editable bank belongs to the right-hand board.
    center = (left + right) / 2
    goal_obj = min(
        goals,
        key=lambda o: abs((o[1] + o[6] / 2) - center) - 0.01 * o[0]
    )
    _, gx, gy, gr, gs, gc, gw, gh = goal_obj
    goal = (gx + gs, gy + gs, gr, gs, gc)

    players = _objects(image, _PLAYER)
    if not players:
        players = _objects(image, _PLAYER, loose=True)
    if not players:
        return None

    player_obj = min(
        players,
        key=lambda o: (
            abs((o[1] + o[6] / 2) - center)
            + (25 if o[1] + o[6] < left - 10 else 0)
            - 0.05 * o[0]
        )
    )
    _, px, py, pr, ps, pc, pw, ph = player_obj
    start = (px, py, pr, ps, pc)

    checkpoints = set(tuple(q) for q in memory.get("checkpoints", []))
    for scale in range(1, 4):
        size = 4 * scale
        for phase in (0, 1):
            small = np.fromfunction(
                lambda y, x: ((x + y + phase) % 2) == 0, (4, 4)
            ).astype(bool)
            mask = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
            for x, y in _matches(image, mask, 11):
                if x + size < left - 10:
                    continue
                if x < gx + gw and x + size > gx and y < gy + gh and y + size > gy:
                    continue
                checkpoints.add((x, y, scale))
    memory["checkpoints"] = [list(q) for q in checkpoints]

    walls = np.zeros((64, 64), dtype=bool)
    walls[:top] = obs[:top] == 6

    # Restore nearly rectangular walls where a foreground object covers pixels.
    for points in _components(walls):
        xx = [p[0] for p in points]
        yy = [p[1] for p in points]
        x0, x1, y0, y1 = min(xx), max(xx) + 1, min(yy), max(yy) + 1
        area = (x1 - x0) * (y1 - y0)
        if len(points) >= 0.8 * area:
            walls[y0:y1, x0:x1] = True

    # Conservatively avoid unfamiliar visible hazard sprites.
    unusual = np.zeros((64, 64), dtype=bool)
    for color in (2, 7, 8, 9, 10, 12, 13, 15):
        unusual[3:top] |= obs[3:top] == color
    unusual[gy:gy + gh, gx:gx + gw] = False
    unusual[py:py + ph, px:px + pw] = False
    if left >= 30:
        unusual[:, :31] = False
    for points in _components(unusual):
        if len(points) < 3:
            continue
        xx = [p[0] for p in points]
        yy = [p[1] for p in points]
        walls[min(yy):max(yy) + 1, min(xx):max(xx) + 1] = True

    button_matches = _matches(obs[48:], _BUTTON, 9)
    if button_matches:
        bx, by = max(button_matches, key=lambda q: q[0])
        button = (by + 48 + 4, bx + 4)
    else:
        points = np.argwhere((obs == 9) & (np.indices(obs.shape)[0] >= 51))
        if len(points):
            button = (int(np.median(points[:, 0])),
                      int(np.median(points[:, 1])))
        else:
            button = (58, 57)

    return start, goal, sorted(checkpoints), walls, button


def _search(start, goal, checkpoints, walls, current, bits, deadline, salt=0):
    n = len(current)
    limit = 1 << bits

    kinds = {
        0: "nop", 1: "left", 2: "right", 3: "down",
        5: "cw", 6: "ccw", 7: "half", 8: "grow", 9: "shrink",
        10: "right2", 11: "right2", 12: "left2", 13: "left2",
        14: "c9", 15: "c8", 16: "ccw",
        33: "up", 34: "left", 63: "c15"
    }
    useful_colors = {start[4], goal[4]}
    max_scale = max([start[3], goal[3]] + [q[2] for q in checkpoints])

    options = []
    for value in current:
        best = {}
        for code in range(limit):
            kind = kinds.get(code, "nop")
            if kind.startswith("c") and kind not in ("cw", "ccw"):
                if int(kind[1:]) not in useful_colors:
                    continue
            if kind == "grow" and max_scale == 1:
                continue
            cost = (code ^ value).bit_count()
            rank = (cost, (code + salt) % limit)
            if kind not in best or rank < best[kind][0]:
                best[kind] = (rank, code)
        options.append([
            (kind, item[1], item[0][0])
            for kind, item in sorted(best.items(), key=lambda kv: kv[1][0])
        ])

    integral = np.pad(walls.astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1).tolist()

    def blocked(x, y, s):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(64, x + 4 * s), min(64, y + 4 * s)
        if x1 <= x0 or y1 <= y0:
            return False
        return (integral[y1][x1] - integral[y0][x1]
                - integral[y1][x0] + integral[y0][x0]) > 0

    positions = [(start[0], start[1]), (goal[0], goal[1])]
    positions += [(q[0], q[1]) for q in checkpoints]
    xmin = min(x for x, y in positions) - 16
    xmax = max(x for x, y in positions) + 16
    ymin = min(y for x, y in positions) - 16
    ymax = max(y for x, y in positions) + 16

    transitions = {}

    def advance(state, kind):
        key = (state, kind)
        if key in transitions:
            return transitions[key]
        x, y, r, s, c = state
        dx = dy = 0
        if kind == "left": dx = -4
        elif kind == "right": dx = 4
        elif kind == "left2": dx = -8
        elif kind == "right2": dx = 8
        elif kind == "up": dy = -4
        elif kind == "down": dy = 4
        elif kind == "cw": r = (r + 90) % 360
        elif kind == "ccw": r = (r - 90) % 360
        elif kind == "half": r = (r + 180) % 360
        elif kind == "grow":
            ns = s + 1
            if ns <= max_scale and not blocked(x, y, ns):
                s = ns
            elif ns > max_scale:
                transitions[key] = None
                return None
        elif kind == "shrink":
            ns = max(1, s - 1)
            if not blocked(x, y, ns):
                s = ns
        elif kind in ("c8", "c9", "c15"):
            c = int(kind[1:])

        if dx or dy:
            if not blocked(x + dx, y + dy, s):
                x, y = x + dx, y + dy
        result = (x, y, r, s, c)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            result = None
        transitions[key] = result
        return result

    def at_checkpoint(state):
        x, y, r, s, c = state
        for cx, cy, cs in checkpoints:
            if (s == cs and x < cx + 4 * cs and x + 4 * s > cx
                    and y < cy + 4 * cs and y + 4 * s > cy):
                return True
        return False

    def programs(origin):
        layer = {origin: (0, ())}
        for i in range(n):
            nxt = {}
            for state, (cost, path) in layer.items():
                for kind, code, extra in options[i]:
                    dest = advance(state, kind)
                    if dest is None:
                        continue
                    nc = cost + extra
                    old = nxt.get(dest)
                    if old is None or nc < old[0]:
                        nxt[dest] = (nc, path + (code,))
            layer = nxt
            if time.perf_counter() > deadline:
                return []
        result = []
        for state, (cost, path) in layer.items():
            if state == goal or at_checkpoint(state):
                result.append((state, cost + 1, path))
        return result

    serial = itertools.count()
    heap = [(0, next(serial), start, None)]
    distances = {start: 0}
    best_fallback = None
    expanded = 0

    while heap and expanded < 40 and time.perf_counter() < deadline:
        cost, _, origin, first_program = heapq.heappop(heap)
        if cost != distances.get(origin):
            continue
        if origin == goal and first_program is not None:
            return first_program
        expanded += 1
        for dest, edge_cost, program in programs(origin):
            first = program if first_program is None else first_program
            nc = cost + edge_cost
            if dest == goal:
                heapq.heappush(heap, (nc, next(serial), dest, first))
                if nc < distances.get(dest, float("inf")):
                    distances[dest] = nc
            elif dest != origin and nc < distances.get(dest, float("inf")):
                distances[dest] = nc
                heapq.heappush(heap, (nc, next(serial), dest, first))
            if first_program is None and dest != origin:
                score = (
                    abs(dest[0] - goal[0]) + abs(dest[1] - goal[1])
                    + 4 * (dest[3] != goal[3]) + edge_cost
                )
                if best_fallback is None or score < best_fallback[0]:
                    best_fallback = (score, program)

    if goal in distances:
        for cost, _, state, first in sorted(heap):
            if state == goal and first is not None:
                return first
    return best_fallback[1] if best_fallback else None


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=np.int8)

    queue = memory.get("queue")
    if queue:
        row, col = queue.pop(0)
        return "ACTION6", int(row), int(col)

    deadline = time.perf_counter() + 8.0
    panel = _program(obs)
    if panel is None:
        return "RESET", None, None

    scene = _read_scene(obs, panel, memory)
    if scene is None:
        return "RESET", None, None

    start, goal, checkpoints, walls, button = scene
    current, clicks, top, left, right = panel
    bits = len(clicks[0])

    signature = (tuple(start), tuple(current))
    previous = memory.get("last_signature")
    if signature == previous:
        memory["stalls"] = memory.get("stalls", 0) + 1
    else:
        memory["stalls"] = 0
    memory["last_signature"] = signature

    if memory["stalls"] >= 3:
        memory.clear()
        return "RESET", None, None

    if start == goal:
        return "ACTION6", int(button[0]), int(button[1])

    program = _search(
        start, goal, checkpoints, walls, current, bits, deadline,
        memory.get("stalls", 0)
    )
    if program is None:
        # If conservative hazard detection made the model disconnected,
        # retry with only the game's ordinary purple walls.
        plain = np.zeros((64, 64), dtype=bool)
        plain[:top] = obs[:top] == 6
        if time.perf_counter() < deadline - 0.3:
            program = _search(
                start, goal, checkpoints, plain, current, bits, deadline,
                memory.get("stalls", 0) + 1
            )

    if program is None:
        memory.clear()
        return "RESET", None, None

    actions = []
    for i, target in enumerate(program):
        difference = current[i] ^ target
        for bit in range(bits):
            if difference & (1 << bit):
                actions.append(clicks[i][bit])
    actions.append(button)

    memory["queue"] = actions
    row, col = memory["queue"].pop(0)
    return "ACTION6", int(row), int(col)

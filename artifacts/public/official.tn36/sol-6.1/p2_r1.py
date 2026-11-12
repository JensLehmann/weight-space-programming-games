import numpy as np
import heapq
import math
import time


def _program(a):
    """Locate the right-hand editable bit columns."""
    rows = []
    for y in range(28, 49):
        xs = []
        for x in range(1, 61):
            h = a[y, x:x + 3]
            if not ((h == 1).all() or (h == 5).all()):
                continue
            if not (a[y - 1, x] == 0 and a[y + 1, x] == 0
                    and a[y, x - 1] == 0 and a[y, x + 3] == 0):
                continue
            v = a[y + 2:y + 5, x + 1]
            if len(v) != 3 or not ((v == 1).all() or (v == 5).all()):
                continue
            if a[y + 2, x] != 0 or a[y + 2, x + 2] != 0:
                continue
            xs.append(x)
        if xs:
            rows.append((y, xs))
    if not rows:
        return None

    y, xs = rows[0]
    # Reject an accidental smaller pattern above the actual program.
    largest = max(len(z[1]) for z in rows)
    for yy, xx in rows:
        if len(xx) == largest:
            y, xs = yy, xx
            break

    bitrows = []
    for yy in range(y, 49, 6):
        if yy + 4 >= 64:
            break
        good = True
        for x in xs:
            h = a[yy, x:x + 3]
            v = a[yy + 2:yy + 5, x + 1]
            if not (((h == 1).all() or (h == 5).all())
                    and ((v == 1).all() or (v == 5).all())
                    and a[yy + 1, x] == 0
                    and a[yy + 2, x] == 0):
                good = False
                break
        if not good:
            break
        bitrows.extend([yy, yy + 3])

    codes, buttons = [], []
    for x in xs:
        code = 0
        bb = []
        for i, yy in enumerate(bitrows):
            if int(a[yy, x + 1]) == 5:
                code |= 1 << i
            bb.append((yy, x + 1))
        codes.append(code)
        buttons.append(bb)
    return codes, buttons, y - 1


def _board(a, paneltop):
    candidates = []
    for y in range(3, min(22, paneltop)):
        x = 0
        while x < 64:
            if a[y, x] != 0:
                x += 1
                continue
            e = x
            while e < 64 and a[y, e] == 0:
                e += 1
            if e - x >= 20 and np.any(a[y + 1, x:e] == 4):
                candidates.append((e - x, x, e, y))
            x = e
    if candidates:
        _, x, e, y = max(candidates)
        return x, e, y, paneltop
    return 0, 64, 3, paneltop


_PIECE = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=bool)

_TARGET = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

# The nominal target includes an additional empty row after row 0.
_TARGET = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)


def _matches(a, mask, color, box):
    l, r, t, b = box
    h, w = mask.shape
    if r - l < w or b - t < h:
        return []
    eq = a[t:b, l:r] == color
    oh, ow = eq.shape[0] - h + 1, eq.shape[1] - w + 1
    ok = np.ones((oh, ow), dtype=bool)
    for dy in range(h):
        for dx in range(w):
            z = eq[dy:dy + oh, dx:dx + ow]
            if mask[dy, dx]:
                ok &= z
            else:
                ok &= ~z
            if not ok.any():
                return []
    yy, xx = np.nonzero(ok)
    return [(int(x + l), int(y + t)) for x, y in zip(xx, yy)]


def _find_sprite(a, base, box, colors, maxscale=4):
    for color in colors:
        for scale in range(1, maxscale + 1):
            for rotation in range(4):
                mask = np.rot90(base, -rotation)
                mask = np.repeat(np.repeat(mask, scale, 0), scale, 1)
                matches = _matches(a, mask, color, box)
                if matches:
                    x, y = matches[0]
                    return x, y, rotation, scale, color
    return None


def _run_button(a):
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 2:7] = True
    mask[1, 1:8] = True
    mask[2:7, :] = True
    mask[7, 1:8] = True
    mask[8, 2:7] = True
    for color in (9, 10):
        hits = _matches(a, mask, color, (0, 64, 48, 64))
        if hits:
            x, y = max(hits)
            return y + 4, x + 4
    return 58, 57


def _integral(mask):
    z = np.zeros((65, 65), dtype=np.int32)
    z[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    return z.tolist()


def _blocked(table, x, y, scale):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + 4 * scale), min(64, y + 4 * scale)
    if x1 <= x0 or y1 <= y0:
        return False
    return (table[y1][x1] - table[y0][x1]
            - table[y1][x0] + table[y0][x0]) > 0


def _checkpoint(state, cps):
    x, y, _, s, _ = state
    size = 4 * s
    for cx, cy, cs in cps:
        if cs == s and x < cx + size and x + size > cx \
                and y < cy + size and y + size > cy:
            return True
    return False


def _operation(code):
    if code in (1, 34):
        return ("m", -4, 0)
    if code == 2:
        return ("m", 4, 0)
    if code == 3:
        return ("m", 0, 4)
    if code == 33:
        return ("m", 0, -4)
    if code in (10, 11):
        return ("m", 8, 0)
    if code in (12, 13):
        return ("m", -8, 0)
    if code == 5:
        return ("r", 1)
    if code in (6, 16):
        return ("r", 3)
    if code == 7:
        return ("r", 2)
    if code == 8:
        return ("s", 1)
    if code == 9:
        return ("s", -1)
    if code == 14:
        return ("c", 9)
    if code == 15:
        return ("c", 8)
    if code == 63:
        return ("c", 15)
    return ("n",)


def _choices(codes, bits):
    groups = {}
    for code in range(1 << bits):
        groups.setdefault(_operation(code), []).append(code)
    result = []
    for old in codes:
        choices = []
        for op, values in groups.items():
            code = min(values, key=lambda c: ((c ^ old).bit_count(), c != old, c))
            choices.append((op, code, (code ^ old).bit_count()))
        result.append(choices)
    return result


def _transition(state, op, walls):
    x, y, r, s, c = state
    kind = op[0]
    if kind == "m":
        nx, ny = x + op[1], y + op[2]
        if not _blocked(walls, nx, ny, s):
            x, y = nx, ny
    elif kind == "r":
        r = (r + op[1]) % 4
    elif kind == "s":
        ns = max(1, s + op[1])
        if _blocked(walls, x, y, ns):
            ns = max(1, ns - op[1])
        s = ns
    elif kind == "c":
        c = op[1]
    return x, y, r, s, c


def _need(state, goal, bits):
    x, y, r, s, c = state
    gx, gy, gr, gs, gc = goal
    dx, dy = gx - x, gy - y
    if dx % 4 or dy % 4:
        return 10000
    if bits <= 2:
        if dy < 0 or r != gr or s != gs or c != gc:
            return 10000
        return (abs(dx) + abs(dy)) // 4
    if c != gc and gc not in (8, 9, 15):
        return 10000
    return ((abs(dx) + 7) // 8 + (abs(dy) + 3) // 4
            + abs(s - gs) + (r != gr) + (c != gc))


def _direct(start, goal, choices, bits, walls, deadline):
    n = len(choices)
    states = {start: (0, ())}
    for i in range(n):
        remaining = n - i - 1
        nxt = {}
        for state, (cost, path) in states.items():
            for op, code, flips in choices[i]:
                ns = _transition(state, op, walls)
                if _need(ns, goal, bits) > remaining:
                    continue
                nc = cost + flips
                old = nxt.get(ns)
                if old is None or nc < old[0]:
                    nxt[ns] = (nc, path + (code,))
        states = nxt
        if not states or time.monotonic() > deadline:
            return None
    answer = states.get(goal)
    return answer[1] if answer else None


def _multi(start, goal, choices, bits, walls, cps, deadline, visited):
    """Search instruction space, requiring a checkpoint at run boundaries."""
    n = len(choices)
    maxruns = 5
    maxscale = max([start[3], goal[3]] + [p[2] for p in cps])
    points = [(start[0], start[1]), (goal[0], goal[1])]
    points += [(p[0], p[1]) for p in cps]
    xmin = max(-12, min(p[0] for p in points) - 12)
    xmax = min(68, max(p[0] for p in points) + 12)
    ymin = max(-12, min(p[1] for p in points) - 12)
    ymax = min(60, max(p[1] for p in points) + 12)

    root = (0, 0, start)
    dist = {root: 0}
    parent = {}
    heap = [(3 * _need(start, goal, bits), 0, 0, root)]
    serial = 0
    best_boundary = None
    best_score = float("inf")
    expanded = 0

    def recover(key):
        path = []
        while key != root:
            prev, code = parent[key]
            path.append(code)
            key = prev
        path.reverse()
        return tuple(path[:n])

    while heap and expanded < 160000:
        if expanded % 256 == 0 and time.monotonic() > deadline:
            break
        _, cost, _, key = heapq.heappop(heap)
        if dist.get(key) != cost:
            continue
        batch, phase, state = key
        expanded += 1
        if batch and phase == 0 and state == goal:
            return recover(key)
        if batch >= maxruns:
            continue

        for op, code, flips in choices[phase]:
            ns = _transition(state, op, walls)
            x, y, r, s, c = ns
            if not (xmin <= x <= xmax and ymin <= y <= ymax
                    and 1 <= s <= maxscale):
                continue
            if c != goal[4] and goal[4] not in (8, 9, 15):
                continue

            nb, np_ = batch, phase + 1
            boundary = np_ == n
            if boundary:
                nb, np_ = batch + 1, 0
                if ns != goal and not _checkpoint(ns, cps):
                    continue

            remaining = (maxruns - nb) * n - np_
            h = _need(ns, goal, bits)
            if h > remaining:
                continue
            nc = cost + 1 + 2 * flips
            if boundary:
                nc += 2
                if batch == 0 and ns in visited and ns != goal:
                    nc += 35

            nk = (nb, np_, ns)
            if nc >= dist.get(nk, float("inf")):
                continue
            dist[nk] = nc
            parent[nk] = (key, code)
            serial += 1
            heapq.heappush(heap, (nc + 3 * h, nc, serial, nk))

            if boundary and batch == 0 and ns != start and ns != goal:
                score = nc + 5 * h
                if ns in visited:
                    score += 80
                if score < best_score:
                    best_score = score
                    best_boundary = nk

    if best_boundary is not None:
        return recover(best_boundary)
    return None


def _make_model(a, box, goal):
    mask = a == 6
    walls = _integral(mask)

    checker = np.indices((4, 4)).sum(0) % 2 == 0
    cps = []
    for scale in range(1, 5):
        m = np.repeat(np.repeat(checker, scale, 0), scale, 1)
        for x, y in _matches(a, m, 11, box):
            cps.append((x, y, scale))

    unknown = np.zeros((64, 64), dtype=bool)
    l, r, t, b = box
    region = a[t:b, l:r]
    unknown[t:b, l:r] = ~np.isin(region, [0, 4, 5, 6, 11])
    # Do not classify the target as an obstacle.
    gx, gy, _, gs, _ = goal
    ox, oy = gx - gs, gy - gs
    unknown[max(0, oy):min(64, oy + 6 * gs),
            max(0, ox):min(64, ox + 6 * gs)] = False

    return {
        "goal": goal,
        "box": box,
        "cps": cps,
        "walls": walls,
        "careful_walls": _integral(mask | unknown),
    }


def _emit(memory):
    action = memory["queue"].pop(0)
    memory["previous_was_run"] = not memory["queue"]
    return tuple(action)


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int16)

    if memory.get("queue"):
        return _emit(memory)

    decoded = _program(a)
    if decoded is None:
        return ("RESET", None, None)

    codes, buttons, paneltop = decoded
    bits = len(buttons[0])
    n = len(codes)
    if not n or not bits:
        return ("RESET", None, None)

    signature = (tuple(p[0][1] for p in buttons), paneltop, bits)
    if memory.get("signature") != signature:
        memory.clear()
        memory["signature"] = signature

    box = _board(a, paneltop)
    model = memory.get("model")
    if model is None:
        target = _find_sprite(a, _TARGET, box, (11, 9, 8, 15, 14))
        if target is None:
            return ("RESET", None, None)
        x, y, r, s, c = target
        goal = (x + s, y + s, r, s, c)
        model = _make_model(a, box, goal)
        memory["model"] = model
    goal = tuple(model["goal"])

    # Include a little space outside the board for a moved piece.
    l, r, t, b = model["box"]
    piecebox = (max(0, l - 8), min(64, r + 8), max(3, t - 2), b)
    start = _find_sprite(a, _PIECE, piecebox, (11, 9, 8, 15, 14))
    if start is None:
        memory.clear()
        return ("RESET", None, None)
    start = tuple(start)

    if memory.pop("previous_was_run", False):
        expected = memory.get("expected")
        if expected is not None and tuple(expected) != start:
            memory["failures"] = memory.get("failures", 0) + 1
            memory["careful"] = True

    visited = {tuple(s) for s in memory.get("visited", [])}
    visited.add(start)
    memory["visited"] = list(visited)

    choices = _choices(codes, bits)
    deadline = time.monotonic() + 7.5
    walls = model["careful_walls"] if memory.get("careful") else model["walls"]

    plan = _direct(start, goal, choices, bits, walls, deadline)
    if plan is None and model["cps"] and time.monotonic() < deadline:
        plan = _multi(start, goal, choices, bits, walls,
                      model["cps"], deadline, visited)

    # Unknown colored objects may be traversable or time-dependent.
    if plan is None and memory.get("careful") and time.monotonic() < deadline:
        walls = model["walls"]
        plan = _direct(start, goal, choices, bits, walls, deadline)
        if plan is None and model["cps"]:
            plan = _multi(start, goal, choices, bits, walls,
                          model["cps"], deadline, visited)

    if plan is None:
        memory.clear()
        return ("RESET", None, None)

    end = start
    for code in plan:
        end = _transition(end, _operation(code), walls)
    memory["expected"] = end

    queue = []
    for column, (old, new) in enumerate(zip(codes, plan)):
        change = old ^ new
        for bit in range(bits):
            if change & (1 << bit):
                row, col = buttons[column][bit]
                queue.append(("ACTION6", int(row), int(col)))
    row, col = _run_button(a)
    queue.append(("ACTION6", int(row), int(col)))
    memory["queue"] = queue
    return _emit(memory)

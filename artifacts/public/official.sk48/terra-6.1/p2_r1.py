import itertools
import heapq
import time

import numpy as np


_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_ACTIONS = {(0, -1): "ACTION1", (0, 1): "ACTION2",
            (-1, 0): "ACTION3", (1, 0): "ACTION4"}

_WIRE = np.array([
    [-2, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -2],
    [3, 2, 2, 3, 2, 2],
    [2, 2, 3, 2, 2, 3],
    [-2, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -2],
])
_CONNECTOR = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, -1],
    [1, 0, 1, 1, 0, -1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])
_WIRES = [np.rot90(_WIRE, -r) for r in range(4)]
_CONNECTORS = [np.rot90(_CONNECTOR, -r) for r in range(4)]


def _wire_at(a, x, y, r):
    if x < 0 or y < 0 or x + 6 > 64 or y + 6 > 64:
        return False
    t = _WIRES[r]
    if r % 2 == 0:
        groups = [[(2, 0), (3, 0)], [(2, 5), (3, 5)]]
    else:
        groups = [[(0, 2), (0, 3)], [(5, 2), (5, 3)]]
    for group in groups:
        actual = tuple(int(a[y + yy, x + xx]) for yy, xx in group)
        normal = tuple(int(t[yy, xx]) for yy, xx in group)
        active = tuple(v - 1 for v in normal)
        if actual != normal and actual != active:
            return False
    return True


def _trace(a, x, y, r):
    dx, dy = _DIRS[r]
    n = 1
    while n < 12 and _wire_at(a, x + 6 * n * dx, y + 6 * n * dy, r):
        n += 1
    return n


def _read(obs):
    a = np.asarray(obs)
    if a.shape != (64, 64):
        return None

    # The playing field is the large rectangle of colour 4 above the HUD.
    row_counts = np.sum(a[:53] == 4, axis=1)
    rows = np.where(row_counts >= 8)[0]
    if not len(rows):
        return None
    y0, y1 = int(rows[0]), int(rows[-1])
    col_counts = np.sum(a[y0:y1 + 1] == 4, axis=0)
    cols = np.where(col_counts >= max(3, int((y1 - y0 + 1) * 0.25)))[0]
    if not len(cols):
        return None
    x0, x1 = int(cols[0]), int(cols[-1])
    bounds = (x0, y0, x1 + 1, y1 + 1)

    # Find the two-colour connector sprites, including their rotations.
    connectors = []
    for y in range(59):
        for x in range(59):
            patch = a[y:y + 6, x:x + 6]
            for color in (6, 15):
                if np.count_nonzero(patch == color) < 18:
                    continue
                for r, t in enumerate(_CONNECTORS):
                    if not np.all(patch[t == 1] == color):
                        continue
                    body = patch[t == 0]
                    if not (np.all(body == 4) or np.all(body == 0)):
                        continue
                    connectors.append((x, y, r, color, bool(np.all(body == 0))))
                    break
    # Remove duplicate detections if a template happens to match twice.
    unique = {}
    for c in connectors:
        unique[(c[0], c[1], c[3])] = c
    connectors = list(unique.values())

    # Find 4x4 coloured blocks. Their centres can contain black goal markers.
    candidates = []
    allowed_colors = {6, 7, 8, 9, 10, 12, 13, 14, 15}
    connector_rects = [(x, y, x + 6, y + 6) for x, y, _, _, _ in connectors]
    for y in range(61):
        for x in range(61):
            c = int(a[y, x])
            if c not in allowed_colors:
                continue
            if (int(a[y, x + 3]) != c or int(a[y + 3, x]) != c
                    or int(a[y + 3, x + 3]) != c):
                continue
            if any(x < rx1 and x + 4 > rx0 and y < ry1 and y + 4 > ry0
                   for rx0, ry0, rx1, ry1 in connector_rects):
                continue
            score = int(np.count_nonzero(a[y:y + 4, x:x + 4] == c))
            if score >= 12:
                candidates.append((-score, y, x, c))
    candidates.sort()
    blocks = []
    for _, y, x, c in candidates:
        if any(abs(x - bx) < 4 and abs(y - by) < 4
               for _, bx, by in blocks):
            continue
        blocks.append((c, x - 1, y - 1))

    main_connectors = [c for c in connectors if c[1] < 53]
    hud_connectors = [c for c in connectors if c[1] >= 53]
    main_connectors.sort(key=lambda c: (c[3], c[2], c[1], c[0]))

    sources = []
    selected = 0
    for x, y, r, color, active in main_connectors:
        hud = next((h for h in hud_connectors if h[3] == color), None)
        if hud is None:
            # An unmatched connector still contributes a stationary wire.
            sources.append((x, y, r, _trace(a, x, y, r), None, False))
            continue
        hx, hy, hr, _, _ = hud
        hn = _trace(a, hx, hy, hr)
        hdx, hdy = _DIRS[hr]
        target = []
        for i in range(hn):
            p = (hx + 6 * i * hdx, hy + 6 * i * hdy)
            for c, bx, by in blocks:
                if (bx, by) == p:
                    target.append(c)
                    break
        if active:
            selected = len(sources)
        sources.append((x, y, r, _trace(a, x, y, r), tuple(target), True))

    # Black, unclickable anchors have a distinctive 2x2 colour-11 centre.
    seen_black = set()
    for y in range(63):
        for x in range(63):
            if not np.all(a[y:y + 2, x:x + 2] == 11):
                continue
            ax, ay = x - 2, y - 2
            if (ax, ay) in seen_black or ax < 0 or ay < 0:
                continue
            seen_black.add((ax, ay))
            possible = []
            for r, (dx, dy) in enumerate(_DIRS):
                if _wire_at(a, ax + 6 * dx, ay + 6 * dy, r):
                    possible.append(r)
            if possible:
                r = possible[0]
            elif ax + 6 <= x0:
                r = 0
            elif ay + 6 <= y0:
                r = 1
            elif ax >= x1:
                r = 2
            else:
                r = 3
            sources.append((ax, ay, r, _trace(a, ax, ay, r), None, False))

    if not sources or not any(s[5] for s in sources):
        return None
    if not sources[selected][5]:
        selected = next(i for i, s in enumerate(sources) if s[5])

    board_blocks = tuple(sorted(
        (c, x, y) for c, x, y in blocks
        if y < 53 and x0 <= x < x1 + 1 and y0 <= y < y1 + 1
    ))

    # The visible black wall tiles are 6x6 squares on the field lattice.
    walls = []
    for y in range(y0, y1 - 4, 6):
        for x in range(x0, x1 - 4, 6):
            if np.all(a[y:y + 6, x:x + 6] == 5):
                walls.append((x, y))

    configs = tuple((s[0], s[1], s[3]) for s in sources)
    state = (configs, board_blocks, selected)
    return bounds, tuple(walls), tuple(sources), state


def _goal(state, sources):
    configs, blocks, _ = state
    at = {(x, y): c for c, x, y in blocks}
    for i, source in enumerate(sources):
        target = source[4]
        if target is None:
            continue
        x, y, n = configs[i]
        dx, dy = _DIRS[source[2]]
        found = []
        for j in range(n):
            c = at.get((x + 6 * j * dx, y + 6 * j * dy))
            if c is not None:
                found.append(c)
        if tuple(found[:len(target)]) != target:
            return False
    return True


def _move(state, which, move, sources, bounds, walls):
    configs, blocks, selected = state
    x0, y0, x1, y1 = bounds
    mx, my = move
    sx, sy = 6 * mx, 6 * my
    hx, hy, length = configs[which]
    r = sources[which][2]
    dx, dy = _DIRS[r]

    def blocked(x, y):
        if x < x0 or y < y0 or x + 6 > x1 or y + 6 > y1:
            return True
        return any(wx <= x < wx + 6 and wy <= y < wy + 6
                   for wx, wy in walls)

    # The handle may slide along the side of the rectangular field.
    forward = (mx, my) == (dx, dy)
    backward = (mx, my) == (-dx, -dy)
    if forward:
        if blocked(hx + 6 * (length - 1) * dx + sx,
                   hy + 6 * (length - 1) * dy + sy):
            return None
        indices = range(length)
    elif backward:
        if length <= 1:
            return None
        indices = range(1, length)
    else:
        nx, ny = hx + sx, hy + sy
        if dx:
            if ny < y0 or ny + 6 > y1:
                return None
        else:
            if nx < x0 or nx + 6 > x1:
                return None
        indices = range(length)

    segments = []
    horizontal = {}
    vertical = {}
    for i, (cx, cy, n) in enumerate(configs):
        rr = sources[i][2]
        ux, uy = _DIRS[rr]
        for j in range(n):
            px, py = cx + 6 * j * ux, cy + 6 * j * uy
            key = (i, j)
            segments.append((px, py, rr, key))
            table = horizontal if rr % 2 == 0 else vertical
            table.setdefault((px, py), []).append(key)

    block_at = {(x, y): i for i, (_, x, y) in enumerate(blocks)}
    moved_blocks = set()
    moved_segments = set()

    def move_block(bi, excluded=None):
        if bi in moved_blocks:
            return True
        _, bx, by = blocks[bi]
        nx, ny = bx + sx, by + sy
        if blocked(nx, ny):
            return False
        perpendicular = vertical if mx else horizontal
        for p in ((bx, by), (nx, ny)):
            for key in perpendicular.get(p, ()):
                if key != excluded:
                    return False
        other = block_at.get((nx, ny))
        if other is not None and not move_block(other):
            return False
        moved_blocks.add(bi)
        return True

    def move_segment(j):
        key = (which, j)
        if key in moved_segments:
            return True
        px, py = hx + 6 * j * dx, hy + 6 * j * dy
        if blocked(px + sx, py + sy):
            if (mx, my) != (-dx, -dy) and not blocked(px, py):
                return False
        for p in ((px, py), (px + sx, py + sy)):
            bi = block_at.get(p)
            if bi is not None and not move_block(bi, key):
                if (dx == 0) != (mx == 0):
                    return False
        moved_segments.add(key)
        return True

    for j in indices:
        if not move_segment(j):
            return None

    new_configs = list(configs)
    if forward:
        new_configs[which] = (hx, hy, length + 1)
    elif backward:
        new_configs[which] = (hx, hy, length - 1)
    else:
        new_configs[which] = (hx + sx, hy + sy, length)

    new_blocks = []
    for i, (c, bx, by) in enumerate(blocks):
        if i in moved_blocks:
            bx, by = bx + sx, by + sy
        new_blocks.append((c, bx, by))
    return (tuple(new_configs), tuple(sorted(new_blocks)), which)


def _assignments(blocks, target):
    choices = [[i for i, b in enumerate(blocks) if b[0] == c] for c in target]
    if any(not q for q in choices):
        return []
    out = []
    for indices in itertools.product(*choices):
        if len(set(indices)) == len(indices):
            out.append(indices)
            if len(out) >= 128:
                break
    return out


def _search(start, sources, bounds, walls, bad, seconds=7.5):
    deadline = time.monotonic() + seconds
    assignments = {}
    for i, s in enumerate(sources):
        if s[4] is not None:
            assignments[i] = _assignments(start[1], s[4])

    cache = {}

    def heuristic(state):
        if state in cache:
            return cache[state]
        configs, blocks, _ = state
        total = 0
        at = {(x, y): c for c, x, y in blocks}
        for i, source in enumerate(sources):
            target = source[4]
            if target is None or not target:
                continue
            hx, hy, n = configs[i]
            dx, dy = _DIRS[source[2]]
            pmax = 0
            for p in range(1, 13):
                xx, yy = hx + 6 * p * dx, hy + 6 * p * dy
                if (xx < bounds[0] or yy < bounds[1]
                        or xx + 6 > bounds[2] or yy + 6 > bounds[3]):
                    break
                pmax = p
            best = 1000
            for assignment in assignments[i]:
                ps = []
                cost = 0
                for bi in assignment:
                    _, bx, by = blocks[bi]
                    p = ((bx - hx) * dx + (by - hy) * dy) // 6
                    perpendicular = abs((bx - hx) * dy - (by - hy) * dx) // 6
                    ps.append(p)
                    cost += perpendicular + max(0, 1 - p) + max(0, p - pmax)
                for j in range(1, len(ps)):
                    cost += max(0, ps[j - 1] - ps[j] + 1)
                if ps:
                    cost += max(0, max(ps) + 1 - n)
                best = min(best, cost)
            found = []
            for j in range(n):
                c = at.get((hx + 6 * j * dx, hy + 6 * j * dy))
                if c is not None:
                    found.append(c)
            for j in range(min(len(found), len(target))):
                if found[j] != target[j]:
                    best += 2
            total += best
        cache[state] = total
        return total

    counter = itertools.count()
    heap = [(heuristic(start) * 1.8, 0, next(counter), start)]
    distance = {start: 0}
    parent = {}
    best_state = start
    best_h = heuristic(start)
    expansions = 0
    solved = None

    while heap and expansions < 100000:
        if expansions % 128 == 0 and time.monotonic() >= deadline:
            break
        _, g, _, state = heapq.heappop(heap)
        if g != distance.get(state):
            continue
        h = heuristic(state)
        if h < best_h:
            best_h, best_state = h, state
        if _goal(state, sources):
            solved = state
            break
        expansions += 1
        for i, source in enumerate(sources):
            if not source[5]:
                continue
            switching = i != state[2]
            if switching:
                hx, hy, _ = state[0][i]
                click = ("ACTION6", hy + 1, hx + 1)
                if (repr(state), repr(click)) in bad:
                    continue
                switched = (state[0], state[1], i)
            else:
                switched = state
                click = None
            for direction, name in _ACTIONS.items():
                action = (name, None, None)
                if (repr(switched), repr(action)) in bad:
                    continue
                nxt = _move(switched, i, direction, sources, bounds, walls)
                if nxt is None or nxt == switched:
                    continue
                ng = g + 1 + int(switching)
                if ng >= distance.get(nxt, 10**9):
                    continue
                distance[nxt] = ng
                parent[nxt] = (state, click, switched, action)
                nh = heuristic(nxt)
                heapq.heappush(heap, (ng + 1.8 * nh, ng, next(counter), nxt))

    end = solved if solved is not None else best_state
    if end == start:
        return []
    edges = []
    while end != start:
        previous, click, switched, action = parent[end]
        edges.append((previous, click, switched, action, end))
        end = previous
    edges.reverse()
    plan = []
    for previous, click, switched, action, nxt in edges:
        if click is not None:
            plan.append((click, repr(previous), repr(switched)))
        plan.append((action, repr(switched), repr(nxt)))
    return plan


def act(obs, available, memory):
    parsed = _read(obs)
    if parsed is None:
        for name in ("ACTION7", "RESET", "ACTION1"):
            if name in available:
                return name, None, None
        return available[0], None, None

    bounds, walls, sources, state = parsed
    signature = repr(state)
    bad = {tuple(item) for item in memory.get("bad", [])}

    expected = memory.get("expected")
    if expected is not None and signature != expected:
        previous = memory.get("previous")
        action = memory.get("last_action")
        if previous == signature and action is not None:
            bad.add((previous, action))
        memory["plan"] = []

    plan = memory.get("plan", [])
    if plan and plan[0][1] != signature:
        plan = []
    if not plan:
        plan = _search(state, sources, bounds, walls, bad)

    memory["bad"] = list(bad)[-100:]
    if plan:
        action, before, after = plan.pop(0)
        memory["plan"] = plan
        memory["expected"] = after
        memory["previous"] = before
        memory["last_action"] = repr(tuple(action))
        return tuple(action)

    # If search did not find progress, try a legal-looking move not already
    # observed to fail, then re-read the board on the next invocation.
    for direction, name in _ACTIONS.items():
        action = (name, None, None)
        if name not in available or (signature, repr(action)) in bad:
            continue
        nxt = _move(state, state[2], direction, sources, bounds, walls)
        if nxt is not None and nxt != state:
            memory["plan"] = []
            memory["expected"] = repr(nxt)
            memory["previous"] = signature
            memory["last_action"] = repr(action)
            return action
    if "ACTION7" in available:
        memory["expected"] = None
        return "ACTION7", None, None
    return "RESET", None, None

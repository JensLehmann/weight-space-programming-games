import heapq
import math
import numpy as np

EMPTY, WALL, BREAK, FLIP, SOLID, GHOST, GROW, SPIKE, GOAL, UNKNOWN, CAP, LAVA = range(12)
W = 11
PASSABLE = (EMPTY, GHOST, CAP)

_ART = [
    (EMPTY, ["       "] * 7, {}),
    (WALL, [
        "xxxxxxx", "xxoxxxx", "xxxxxxx", "xxxxxxx",
        "xoxxxxx", "xxxxoxx", "xxxxxxx"
    ], {"x": 5, "o": 3}),
    (BREAK, [
        ".......", ".oxxxo.", ".xxxxx.", ".xxxxx.",
        ".xxxxx.", ".oxxxo.", "......."
    ], {".": 5, "o": 3, "x": 14}),
    (FLIP, [
        ".......", ".oxxxo.", ".xxxxx.", ".xxxxx.",
        ".xxxxx.", ".oxxxo.", "......."
    ], {".": 5, "o": 3, "x": 8}),
    (SOLID, [
        ".......", ".oxxxo.", ".xxxxx.", ".xxxxx.",
        ".xxxxx.", ".oxxxo.", "......."
    ], {".": 5, "o": 3, "x": 12}),
    (GHOST, [
        ".......", "..xxx..", ".xoxox.", ".xxoxx.",
        ".xoxox.", "..xxx..", "......."
    ], {"o": 12}),
    (GROW, [
        ".......", ".oxxxo.", ".xxxxx.", ".xxxxx.",
        ".xxxxx.", ".oxxxo.", "......."
    ], {".": 5, "o": 3, "x": 15}),
    (SPIKE, [
        "       ", " o...o", " o...o", " o...o",
        " o...o", " oxrxo", " ooooo"
    ], {"o": 5, ".": 15, "x": 11, "r": 0}),
    (SPIKE, [
        " ooooo", " oxrxo", " o...o", " o...o",
        " o...o", " o...o", "       "
    ], {"o": 5, ".": 15, "x": 11, "r": 0}),
    (GOAL, [
        "  ... ", " ..x..", " .xxx.", " ..x..",
        "  ... ", "      ", "      "
    ], {".": 5, "x": 7}),
    (CAP, [
        "      ", "      ", "      ", " .... ",
        "......", "..oo.o", ".ooooo"
    ], {".": 5, "o": 15}),
    (LAVA, [
        "oooo.o", "oooo.o", "o.oooo", "o.o.oo",
        "o.o.oo", "o.oooo", "oooo.o"
    ], {".": 5, "o": 15}),
]

_TEMPLATES = []
for _kind, _rows, _palette in _ART:
    _a = []
    for _r in range(1, 6):
        _line = _rows[_r].ljust(7)
        for _c in range(1, 6):
            _a.append(_palette.get(_line[_c], 10))
    _TEMPLATES.append((_kind, tuple(_a)))
_EXACT = {a: k for k, a in _TEMPLATES}


def _decode(obs):
    ys, xs = np.where(obs[:63] == 9)
    if not len(ys):
        return None
    px = int(xs.min()) // 6
    sy = int(ys.min()) - 1
    gravity = 1 if abs(sy - 26) < abs(sy - 36) else -1
    rel = {}

    for dy in range(-12, 13):
        top = sy + 6 * dy
        rows = [r for r in range(1, 6) if 0 <= top + r < 63]
        if not rows:
            continue
        for x in range(W):
            cols = [c for c in range(1, 6) if x * 6 + c < 64]
            if len(rows) * len(cols) < 6:
                continue
            if x == px and dy == 0:
                rel[x, dy] = EMPTY
                continue
            inds, vals = [], []
            for r in rows:
                for c in cols:
                    inds.append((r - 1) * 5 + c - 1)
                    vals.append(int(obs[top + r, x * 6 + c]))
            if 9 in vals:
                kind = EMPTY
            elif 14 in vals:
                kind = BREAK
            elif 8 in vals:
                kind = FLIP
            elif 7 in vals:
                kind = GOAL
            elif len(vals) == 25 and tuple(vals) in _EXACT:
                kind = _EXACT[tuple(vals)]
            else:
                best, kind = 1000, EMPTY
                for t, pat in _TEMPLATES:
                    error = sum(v != pat[i] for i, v in zip(inds, vals))
                    if error < best:
                        best, kind = error, t
            rel[x, dy] = kind
    return px, sy, gravity, rel


def _indices(board, value):
    token = bytes((value,))
    i = board.find(token)
    while i >= 0:
        yield i
        i = board.find(token, i + 1)


def _fall(board, start, gravity):
    q = start
    step = gravity * W
    for _ in range(len(board) // W + 2):
        nq = q + step
        if not 0 <= nq < len(board):
            return q, True, False
        t = board[nq]
        if t in PASSABLE:
            q = nq
        elif t == UNKNOWN:
            return nq, True, False
        elif t == GOAL:
            return nq, False, True
        elif t == SPIKE:
            return None
        else:
            return q, False, False
    return None


def _simulate(board, p, gravity, action):
    kind, target = action
    if kind in (3, 4):
        dx = -1 if kind == 3 else 1
        if not 0 <= p % W + dx < W:
            return None
        q = p + dx
        t = board[q]
        if t == GOAL:
            return q, gravity, board, False, True
        if t == UNKNOWN:
            return q, gravity, board, True, False
        if t not in PASSABLE:
            return None
        result = _fall(board, q, gravity)
        if result is None:
            return None
        q, unknown, win = result
        return q, gravity, board, unknown, win

    if kind != 6 or not 0 <= target < len(board):
        return None
    t = board[target]
    if t not in (BREAK, FLIP, SOLID, GHOST, GROW):
        return None

    nb = bytearray(board)
    ng = gravity
    if t == BREAK:
        nb[target] = EMPTY
    elif t == SOLID:
        nb[target] = GHOST
    elif t == GHOST:
        nb[target] = SOLID
    elif t == GROW:
        nb[target] = EMPTY
        tx = target % W
        neighbours = [target - W, target + W]
        if tx:
            neighbours.append(target - 1)
        if tx + 1 < W:
            neighbours.append(target + 1)
        for q in neighbours:
            if 0 <= q < len(board) and q != p and board[q] == EMPTY:
                nb[q] = GROW
    else:
        ng = -gravity
        nb[target] = EMPTY

    q, unknown, win = p, False, False
    support = p + ng * W
    if t == FLIP:
        if not 0 <= support < len(board):
            return p, ng, bytes(nb), True, False
        if board[support] not in (WALL, BREAK, FLIP, SOLID, GROW):
            result = _fall(board, support, ng)
            if result is None:
                return None
            q, unknown, win = result
    elif target == support:
        result = _fall(nb, support, ng)
        if result is None:
            return None
        q, unknown, win = result
    return q, ng, bytes(nb), unknown, win


def _visible(p, gravity, q):
    sy = 36 if gravity == -1 else 26
    top = sy + 6 * (q // W - p // W)
    return top <= 62 and top + 5 >= 0


def _actions(board, p, gravity, clickable=True, goal=None):
    yield 3, -1
    yield 4, -1
    if not clickable:
        return

    used = set()
    candidates = [p + gravity * W]
    if p % W:
        candidates.append(p - 1)
    if p % W + 1 < W:
        candidates.append(p + 1)

    for q in candidates:
        if (0 <= q < len(board) and
                board[q] in (BREAK, SOLID, GROW) and
                _visible(p, gravity, q)):
            used.add(q)
            yield 6, q

    for q in _indices(board, FLIP):
        if _visible(p, gravity, q):
            used.add(q)
            yield 6, q

    for q in _indices(board, GHOST):
        if abs(q % W - p % W) <= 2 and _visible(p, gravity, q):
            used.add(q)
            yield 6, q

    growers = []
    px, py = p % W, p // W
    for q in _indices(board, GROW):
        if q in used or not _visible(p, gravity, q):
            continue
        forward = (q // W - py) * gravity
        # Old scaffolding behind the player is not a useful expansion
        # frontier. Adjacent obstacles were already included above.
        if forward < 0:
            continue
        dx = abs(q % W - px)
        score = dx + 0.35 * forward
        if goal is not None:
            score += 0.10 * abs(q % W - goal % W)
        growers.append((score, q))
    growers.sort()
    for _, q in growers[:16]:
        yield 6, q


def _bridge_heuristic(board, p, gravity):
    py = p // W
    yellows = [
        q for q in _indices(board, GROW)
        if _visible(p, gravity, q)
        and (q // W - py) * gravity >= 0
    ]
    if not yellows:
        return 0.0

    px = p % W
    yy = [(q % W, q // W) for q in yellows]
    options = [(px, 0)]
    for direction in (-1, 1):
        extra = 0
        x = px + direction
        while 0 <= x < W:
            t = board[py * W + x]
            if t in (WALL, SPIKE, LAVA, UNKNOWN):
                break
            if t in (BREAK, SOLID, FLIP, GROW):
                extra += 1
            options.append((x, extra))
            x += direction

    best = 10.0
    for x, extra in options:
        q = py * W + x
        distance = 0
        while True:
            q += gravity * W
            distance += 1
            if not 0 <= q < len(board):
                t = UNKNOWN
                break
            t = board[q]
            if t not in PASSABLE:
                break
        horizontal = abs(x - px) + extra
        if t == GOAL:
            best = min(best, horizontal + 1)
        elif t == UNKNOWN:
            best = min(best, horizontal + 5)
        elif t != SPIKE and distance >= 2:
            best = min(best, horizontal + 1)
        elif t == GROW and distance == 1:
            best = min(best, horizontal + 2)
        elif t == SPIKE and distance >= 3:
            for d in range(2, min(distance, 11)):
                y = py + gravity * d
                grow_cost = min(abs(x - xx) + abs(y - y0)
                                for xx, y0 in yy)
                best = min(best, horizontal + grow_cost)

    if any(_visible(p, gravity, q) for q in _indices(board, FLIP)):
        best = min(best, 2.0)
    return min(best, 9.0)


def _make_board(world, px, py):
    ys = [y for x, y in world]
    lo = min(min(ys), py) - 2
    hi = max(max(ys), py) + 2
    b = bytearray([UNKNOWN]) * ((hi - lo + 1) * W)
    for (x, y), t in world.items():
        if 0 <= x < W:
            b[(y - lo) * W + x] = t
    return bytes(b), (py - lo) * W + px, lo


def _potential(board, goal, gravity):
    """Relaxed, directed route to the goal through removable terrain."""
    inf = 1000000.0
    dist = [inf] * len(board)
    dist[goal] = 0.0
    heap = [(0.0, goal)]
    while heap:
        d, q = heapq.heappop(heap)
        if d != dist[q]:
            continue
        neighbours = []
        if q % W:
            neighbours.append((q - 1, 1.0))
        if q % W + 1 < W:
            neighbours.append((q + 1, 1.0))
        if gravity:
            neighbours.append((q - gravity * W, 0.18))
        else:
            neighbours.append((q - W, 0.18))
            neighbours.append((q + W, 0.18))
        for r, cost in neighbours:
            if not 0 <= r < len(board):
                continue
            if board[r] in (WALL, SPIKE, LAVA, UNKNOWN):
                continue
            nd = d + cost
            if nd < dist[r]:
                dist[r] = nd
                heapq.heappush(heap, (nd, r))
    return dist


def _search(board, start, gravity, lo, world, banned, clickable):
    goals = list(_indices(board, GOAL))
    goal = min(
        goals,
        key=lambda q: abs(q // W - start // W)
        + abs(q % W - start % W)
    ) if goals else None

    hasgrow = board.find(bytes((GROW,))) >= 0
    hasflip = board.find(bytes((FLIP,))) >= 0
    potentials = None
    if goal is not None and hasgrow:
        potentials = {
            -1: _potential(board, goal, -1),
            1: _potential(board, goal, 1),
            0: _potential(board, goal, 0),
        }
        if potentials[gravity][start] >= 100000 and not hasflip:
            goal = None
    elif (goal is not None and
          (goal // W - start // W) * gravity < 0 and not hasflip):
        goal = None

    novelty_cache = {}

    def novelty(p, g):
        key = p, g
        if key in novelty_cache:
            return novelty_cache[key]
        sy = 36 if g == -1 else 26
        y = p // W + lo
        first = math.ceil(-sy / 6)
        last = math.floor((57 - sy) / 6)
        n = sum((x, y + d) not in world
                for d in range(first, last + 1)
                for x in range(1, W - 1))
        novelty_cache[key] = n
        return n

    base_novelty = novelty(start, gravity)

    def heuristic(b, p, g):
        if goal is not None and potentials is not None:
            d = potentials[g][p]
            if d < 100000:
                return 1.2 * d
            if b.find(bytes((FLIP,))) >= 0:
                d = potentials[-g][p]
                if d < 100000:
                    return 1.2 * d + 2.0
                d = potentials[0][p]
                return min(30.0, 1.2 * d + 4.0)
            return 45.0

        h = 0.0
        if goal is not None:
            dx = abs(p % W - goal % W)
            dy = goal // W - p // W
            h = 0.75 * dx + 0.16 * abs(dy)
            if dy * g < 0:
                h += 3.0
        if hasgrow:
            h += _bridge_heuristic(b, p, g) * (
                0.4 if goal is not None else 1.25
            )
        return h

    # position, gravity, board, parent, action, cost, unknown, win
    nodes = [(start, gravity, board, -1, None, 0, False, False)]
    initial_h = heuristic(board, start, gravity)
    heap = [(initial_h, 0)]
    seen = {(start, gravity, board): 0}
    frontier = None
    frontier_score = float("inf")
    partial = None
    partial_score = initial_h
    limit = 28000 if goal is not None else 17000
    expanded = 0
    blind_cost = 12.0 if hasgrow else 5.0

    def path(index):
        result = []
        while nodes[index][3] >= 0:
            a = nodes[index][4]
            if a[0] == 6:
                result.append((6, a[1] % W, a[1] // W + lo))
            else:
                result.append((a[0], None, None))
            index = nodes[index][3]
        result.reverse()
        return result

    while heap and expanded < limit:
        priority, ni = heapq.heappop(heap)
        p, g, b, parent, action, cost, unknown, win = nodes[ni]
        if seen.get((p, g, b)) != cost:
            continue
        expanded += 1
        if win:
            return path(ni), True

        nv = novelty(p, g) - base_novelty
        if cost and (unknown or nv >= 4):
            score = cost + (blind_cost if unknown else 0.0)
            score -= min(30, nv) * 0.025
            if goal is not None:
                old_dist = (abs(start // W - goal // W)
                            + abs(start % W - goal % W))
                new_dist = (abs(p // W - goal // W)
                            + abs(p % W - goal % W))
                score += 0.10 * (new_dist - old_dist)
            if score < frontier_score:
                frontier, frontier_score = ni, score
            if goal is None and not unknown:
                return path(ni), False
        if unknown:
            continue
        if (goal is None and frontier is not None and
                priority > frontier_score + 2):
            return path(frontier), False
        if cost >= 48:
            continue

        h = heuristic(b, p, g)
        if cost and h + 0.07 * cost < partial_score - 0.7:
            partial = ni
            partial_score = h + 0.07 * cost

        for a in _actions(b, p, g, clickable, goal):
            if parent < 0:
                wa = ((6, a[1] % W, a[1] // W + lo)
                      if a[0] == 6 else (a[0], None, None))
                if wa in banned:
                    continue
            result = _simulate(b, p, g, a)
            if result is None:
                continue
            np_, ng, nb, un, won = result

            # Do not consume the last known gravity switch and land
            # on the wrong side of an impenetrable goal enclosure.
            if (goal is not None and potentials is not None
                    and not un and not won
                    and nb.find(bytes((FLIP,))) < 0
                    and potentials[ng][np_] >= 100000):
                continue

            nc = cost + 1
            key = np_, ng, nb
            if seen.get(key, 10**9) <= nc:
                continue
            if len(nodes) >= 105000 and not won:
                continue
            seen[key] = nc
            idx = len(nodes)
            nodes.append((np_, ng, nb, ni, a, nc, un, won))
            if won:
                return path(idx), True
            nh = 0.0 if un else heuristic(nb, np_, ng)
            heapq.heappush(heap, (nc + nh + (blind_cost if un else 0), idx))

    if frontier is not None:
        return path(frontier), False
    if partial is not None and expanded >= limit // 2:
        return path(partial), False
    return [], False


def _align_unknown(world, rel, px, expected):
    entry = expected["y"]
    direction = expected["g"]
    candidates = [entry + direction * k for k in range(65)]
    if px == expected.get("oldx"):
        candidates.append(expected["oldy"])
    best_y, best_score = entry, -1e30
    for y in candidates:
        score = -0.003 * abs(y - entry)
        for (x, dy), t in rel.items():
            if x == px and dy == 0:
                continue
            old = world.get((x, y + dy))
            if old is None:
                continue
            if old == t:
                score += 0.065 if t in (WALL, EMPTY) else 1.1
            elif old in PASSABLE and t in PASSABLE:
                score -= 0.25
            else:
                score -= 1.65
        if world.get((px, y)) in (WALL, BREAK, FLIP, GROW, LAVA):
            score -= 8
        if score > best_score:
            best_y, best_score = y, score
    return best_y


def _state_key(world, px, py, gravity):
    return px, py, gravity, tuple(sorted(world.items()))


def _reset(memory, available):
    count = memory.get("resets", 0) + 1
    memory.clear()
    memory["resets"] = count
    if "RESET" in available:
        return "RESET", None, None
    for a in ("ACTION7", "ACTION3", "ACTION4"):
        if a in available:
            return a, None, None
    return available[0], None, None


def _undo(memory, available):
    history = memory.get("history", [])
    if not history or "ACTION7" not in available:
        return _reset(memory, available)

    old_world, px, py, g, action = history.pop()
    world = dict(memory["world"])
    world.update(old_world)
    memory["world"] = world
    key = _state_key(world, px, py, g)
    memory.setdefault("bans", {}).setdefault(key, set()).add(action)
    memory["expected"] = {
        "x": px, "y": py, "g": g, "unknown": False,
        "oldx": px, "oldy": py
    }
    memory["plan"] = []
    return "ACTION7", None, None


def act(obs, available, memory):
    available = list(available)
    decoded = _decode(obs)
    if decoded is None:
        if memory.get("history"):
            return _undo(memory, available)
        return _reset(memory, available)

    px, sy, gravity, rel = decoded

    if "world" not in memory:
        world = {(x, dy): t for (x, dy), t in rel.items()}
        py = 0
        memory["world"] = world
        memory["history"] = []
        memory["bans"] = {}
        memory["plan"] = []
        memory["visits"] = {}
    else:
        world = memory["world"]
        expected = memory.get("expected")
        if expected is None:
            py = memory.get("py", 0)
        elif expected.get("unknown"):
            py = _align_unknown(world, rel, px, expected)
        else:
            py = expected["y"]

        changed = False
        for (x, dy), t in rel.items():
            key = x, py + dy
            if x == px and dy == 0 and world.get(key) == GHOST:
                t = GHOST
            if key in world and world[key] != t:
                changed = True
            world[key] = t

        if expected and (expected["x"] != px or expected["g"] != gravity):
            changed = True
        if changed:
            memory["plan"] = []

    memory["px"], memory["py"], memory["g"] = px, py, gravity
    board, p, lo = _make_board(world, px, py)
    statekey = _state_key(world, px, py, gravity)
    banned = memory["bans"].get(statekey, set())

    visits = memory.setdefault("visits", {})
    visits[statekey] = visits.get(statekey, 0) + 1
    if visits[statekey] > 3:
        memory["plan"] = []

    plan = memory.get("plan", [])
    solved = bool(plan)
    if plan and plan[0] in banned:
        plan = []
    if not plan:
        plan, solved = _search(
            board, p, gravity, lo, world, banned,
            "ACTION6" in available
        )
    if not plan:
        return _undo(memory, available)

    chosen = plan[0]
    kind, tx, ty = chosen
    action_name = "ACTION" + str(kind)
    if action_name not in available:
        return _undo(memory, available)

    if kind == 6:
        target = (ty - lo) * W + tx
        action = 6, target
        row = sy + 6 * (ty - py) + 3
        col = tx * 6 + 3
        if not (-2 <= row <= 65 and 0 <= col <= 63):
            memory["plan"] = []
            return _undo(memory, available)
        row = max(0, min(62, row))
    else:
        action = kind, -1
        row = col = None

    result = _simulate(board, p, gravity, action)
    if result is None:
        memory["plan"] = []
        memory["bans"].setdefault(statekey, set()).add(chosen)
        return _undo(memory, available)

    np_, ng, nb, unknown, won = result
    memory["history"].append((dict(world), px, py, gravity, chosen))
    if nb != board:
        for i, (old, new) in enumerate(zip(board, nb)):
            if old != new:
                world[i % W, i // W + lo] = new

    memory["expected"] = {
        "x": np_ % W,
        "y": np_ // W + lo,
        "g": ng,
        "unknown": unknown,
        "oldx": px,
        "oldy": py,
    }
    memory["plan"] = plan[1:] if solved and not unknown else []
    return action_name, row, col

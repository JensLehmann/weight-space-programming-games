import numpy as np
import time
from collections import defaultdict, Counter, deque


_TILE_COLORS = {0, 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 15}


def _parse(obs):
    a = np.asarray(obs)
    valid = np.ones((64, 64), dtype=bool)
    valid[:, 0] = False
    for i in range(8):
        valid[1, 10 + 5 * i:14 + 5 * i] = False

    seen = np.zeros((64, 64), dtype=bool)
    components = []
    for y in range(64):
        for x in range(1, 64):
            color = int(a[y, x])
            if seen[y, x] or not valid[y, x] or color in (3, 4):
                continue
            seen[y, x] = True
            stack = [(x, y)]
            xmin = xmax = x
            ymin = ymax = y
            count = 0
            while stack:
                xx, yy = stack.pop()
                count += 1
                xmin, xmax = min(xmin, xx), max(xmax, xx)
                ymin, ymax = min(ymin, yy), max(ymax, yy)
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if (0 <= nx < 64 and 0 <= ny < 64
                            and valid[ny, nx] and not seen[ny, nx]
                            and int(a[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            components.append(
                (color, xmin, ymin, xmax - xmin + 1,
                 ymax - ymin + 1, count)
            )

    scales = []
    for c, x, y, w, h, count in components:
        if c in (8, 14):
            if w * 4 == h * 3 and w % 3 == 0:
                scales.append(w // 3)
            elif w * 3 == h * 4 and w % 4 == 0:
                scales.append(w // 4)
    scales = [s for s in scales if s in (1, 2)]
    if scales:
        scale = Counter(scales).most_common(1)[0][0]
    else:
        sizes = Counter(
            w for c, x, y, w, h, count in components
            if c in _TILE_COLORS and w == h and count == w * h
            and w in (2, 4)
        )
        scale = 2 if sizes[4] > sizes[2] else 1

    board = {}
    buttons = []
    corners = {11: set(), 12: set()}
    for c, x, y, w, h, count in components:
        if (c in _TILE_COLORS and w == h == 2 * scale
                and count == w * h):
            board[(x, y)] = c
        if c in corners and w == h == scale and count == scale * scale:
            corners[c].add((x, y))
        if c in (8, 14):
            if (w, h) == (3 * scale, 4 * scale):
                buttons.append((y + h // 2, x + w // 2, c, 0))
            elif (w, h) == (4 * scale, 3 * scale):
                buttons.append((y + h // 2, x + w // 2, c, 1))

    targets = {11: set(), 12: set()}
    d = 3 * scale
    for c in (11, 12):
        cs = corners[c]
        for x, y in cs:
            if ((x + d, y) in cs and (x, y + d) in cs
                    and (x + d, y + d) in cs):
                targets[c].add((x + scale, y + scale))

    # Recover any lattice-aligned tile partly obscured by the interface.
    if board:
        modulus = 3 * scale
        rx = Counter(x % modulus for x, y in board).most_common(1)[0][0]
        ry = Counter(y % modulus for x, y in board).most_common(1)[0][0]
        size = 2 * scale
        for y in range(ry, 65 - size, modulus):
            for x in range(rx, 65 - size, modulus):
                v = a[y:y + size, x:x + size]
                mask = valid[y:y + size, x:x + size]
                pixels = v[mask]
                if len(pixels) >= scale * scale:
                    c = int(pixels[0])
                    if c in _TILE_COLORS and np.all(pixels == c):
                        board[(x, y)] = c

    buttons.sort()
    return board, targets, buttons, scale


def _pair_buttons(buttons):
    unused = set(range(len(buttons)))
    groups = []
    # Matching arrows normally share their perpendicular coordinate.
    for i, b in enumerate(buttons):
        if i not in unused or b[2] != 14:
            continue
        r, c, color, axis = b
        candidates = []
        for j in unused:
            q = buttons[j]
            if q[2] != 8 or q[3] != axis:
                continue
            aligned = q[0] == r if axis == 0 else q[1] == c
            if aligned:
                distance = abs(q[1] - c) if axis == 0 else abs(q[0] - r)
                candidates.append((distance, j))
        if candidates:
            j = min(candidates)[1]
            groups.append((i, j))
            unused.remove(i)
            unused.remove(j)
    for i in sorted(unused):
        groups.append((i, None))
    return groups


def _hungarian(cost):
    n = len(cost)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [10**18] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**18
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j] = cur
                        way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _infer_permutation(sites, logs, scale):
    n = len(sites)
    if not logs:
        return list(range(n))

    sources = defaultdict(list)
    destinations = defaultdict(list)
    for i, pos in enumerate(sites):
        sources[tuple(before.get(pos, -1) for before, after in logs)].append(i)
        destinations[tuple(after.get(pos, -1) for before, after in logs)].append(i)

    if set(sources) != set(destinations):
        return None

    result = [0] * n
    step = 3 * scale
    for signature, src in sources.items():
        dst = destinations[signature]
        if len(src) != len(dst):
            return None
        if len(src) == 1:
            result[src[0]] = dst[0]
            continue

        costs = []
        for i in src:
            x, y = sites[i]
            row = []
            for j in dst:
                xx, yy = sites[j]
                dx, dy = abs(x - xx), abs(y - yy)
                if i == j:
                    value = 0
                elif max(dx, dy) <= step:
                    value = 10 + (dx + dy) // scale
                else:
                    # Prefer one wraparound edge over several long jumps.
                    value = 1000 + (dx + dy) // scale
                row.append(value)
            costs.append(row)
        assignment = _hungarian(costs)
        for k, j in enumerate(assignment):
            result[src[k]] = dst[j]
    return result


def _models(memory):
    sites = sorted(memory["sites"])
    maps = {}
    repaired = []
    changed = False
    for primary, secondary in memory["groups"]:
        logs = list(memory["history"][primary])
        if secondary is not None:
            logs += [(after, before)
                     for before, after in memory["history"][secondary]]
        p = _infer_permutation(sites, logs, memory["scale"])
        if p is None and secondary is not None:
            repaired.extend([(primary, None), (secondary, None)])
            changed = True
            continue
        if p is None:
            return None
        maps[primary] = p
        if secondary is not None:
            inverse = [0] * len(p)
            for i, j in enumerate(p):
                inverse[j] = i
            maps[secondary] = inverse
        repaired.append((primary, secondary))

    if changed:
        memory["groups"] = repaired
        memory["phase"] = 1
        return None
    return sites, maps


def _search(board, targets, sites, maps, seconds):
    deadline = time.monotonic() + seconds
    index = {pos: i for i, pos in enumerate(sites)}
    kinds = [c for c in (11, 12) if targets[c]]
    if not kinds:
        return []

    counts = [sum(value == c for value in board.values()) for c in kinds]
    if any(counts[i] < len(targets[c]) for i, c in enumerate(kinds)):
        return None

    pack = bytes if len(sites) <= 256 else tuple
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)

    start_values = []
    for c in kinds:
        start_values.extend(sorted(index[pos] for pos, value in board.items()
                                   if value == c))
    start = pack(start_values)

    def move(state, permutation):
        values = []
        for i in range(len(kinds)):
            values.extend(sorted(permutation[j]
                                 for j in state[offsets[i]:offsets[i + 1]]))
        return pack(values)

    def won(state):
        for i, c in enumerate(kinds):
            present = set(state[offsets[i]:offsets[i + 1]])
            if not all(index[pos] in present for pos in targets[c]):
                return False
        return True

    if won(start):
        return []

    actions = sorted(maps)
    inverse_maps = {}
    for action in actions:
        inverse = [0] * len(sites)
        for i, j in enumerate(maps[action]):
            inverse[j] = i
        inverse_maps[action] = inverse

    equal_counts = all(counts[i] == len(targets[c])
                       for i, c in enumerate(kinds))

    if not equal_counts:
        parents = {start: None}
        queue = deque([start])
        while queue and len(parents) < 250000:
            if time.monotonic() > deadline:
                return None
            state = queue.popleft()
            for action in actions:
                child = move(state, maps[action])
                if child in parents:
                    continue
                parents[child] = (state, action)
                if won(child):
                    path = []
                    while parents[child] is not None:
                        child, action = parents[child]
                        path.append(action)
                    return path[::-1]
                queue.append(child)
        return None

    target_values = []
    for c in kinds:
        target_values.extend(sorted(index[pos] for pos in targets[c]))
    target = pack(target_values)
    forward = {start: None}
    backward = {target: None}
    front = [start]
    back = [target]

    while front and back and len(forward) + len(backward) < 350000:
        if time.monotonic() > deadline:
            return None
        is_forward = len(front) <= len(back)
        frontier = front if is_forward else back
        own = forward if is_forward else backward
        other = backward if is_forward else forward
        permutations = maps if is_forward else inverse_maps
        next_frontier = []
        meet = None
        for state in frontier:
            for action in actions:
                child = move(state, permutations[action])
                if child in own:
                    continue
                own[child] = (state, action)
                if child in other:
                    meet = child
                    break
                next_frontier.append(child)
            if meet is not None:
                break
            if len(own) % 2048 == 0 and time.monotonic() > deadline:
                return None
        if meet is not None:
            left = []
            state = meet
            while forward[state] is not None:
                state, action = forward[state]
                left.append(action)
            right = []
            state = meet
            while backward[state] is not None:
                state, action = backward[state]
                right.append(action)
            return left[::-1] + right
        if is_forward:
            front = next_frontier
        else:
            back = next_frontier
    return None


def _goal_positions(board, targets):
    return {
        c: frozenset(pos for pos, value in board.items() if value == c)
        for c in (11, 12) if targets[c]
    }


def _issue(memory, board, action, model=None):
    expected = None
    if model is not None:
        sites, maps = model
        p = maps[action]
        index = {pos: i for i, pos in enumerate(sites)}
        expected = {}
        for c, positions in _goal_positions(board, memory["targets"]).items():
            expected[c] = frozenset(sites[p[index[pos]]] for pos in positions)
    memory["last"] = (action, dict(board), expected)
    row, col, color, axis = memory["buttons"][action]
    return "ACTION6", int(row), int(col)


def act(obs, available, memory):
    board, targets, buttons, scale = _parse(obs)

    if not memory.get("initialized"):
        memory.clear()
        memory.update({
            "initialized": True,
            "buttons": buttons,
            "groups": _pair_buttons(buttons),
            "history": [[] for _ in buttons],
            "targets": targets,
            "scale": scale,
            "sites": set(board) | targets[11] | targets[12],
            "phase": 1,
            "plan": [],
            "model": None,
        })

    if not memory["buttons"]:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    memory["sites"].update(board)
    last = memory.pop("last", None)
    if last is not None:
        action, before, expected = last
        memory["sites"].update(before)
        memory["history"][action].append((before, dict(board)))
        if (expected is not None
                and expected != _goal_positions(board, memory["targets"])):
            memory["plan"] = []
            memory["model"] = None
            memory["phase"] = 3

    if memory["plan"]:
        action = memory["plan"].pop(0)
        return _issue(memory, board, action, memory["model"])

    # First obtain one transition for every independent arrow pair.
    for primary, secondary in memory["groups"]:
        if not memory["history"][primary]:
            return _issue(memory, board, primary)

    model = _models(memory)
    if model is None:
        for primary, secondary in memory["groups"]:
            if not memory["history"][primary]:
                return _issue(memory, board, primary)
        primary = min(
            (g[0] for g in memory["groups"]),
            key=lambda i: len(memory["history"][i])
        )
        return _issue(memory, board, primary)

    # A short solution can often be recognized with only one probe per pair.
    path = None
    if memory["phase"] == 1:
        sites, maps = model
        path = _search(board, memory["targets"], sites, maps, 1.5)
        if path is None or len(path) > 3:
            memory["phase"] = 2
            path = None
        else:
            memory["phase"] = 3

    if memory["phase"] == 2:
        for primary, secondary in memory["groups"]:
            if len(memory["history"][primary]) < 2:
                return _issue(memory, board, primary)
        memory["phase"] = 3

    if path is None:
        sites, maps = model
        path = _search(board, memory["targets"], sites, maps, 6.0)

    if path:
        memory["model"] = model
        memory["plan"] = path[1:]
        return _issue(memory, board, path[0], model)

    if path == []:
        # Winning is checked only after a button press.
        goals = _goal_positions(board, memory["targets"])
        sites, maps = model
        index = {pos: i for i, pos in enumerate(sites)}
        for action, p in maps.items():
            if all(frozenset(sites[p[index[pos]]] for pos in positions)
                   == positions for positions in goals.values()):
                return _issue(memory, board, action, model)
        for primary, secondary in memory["groups"]:
            if secondary is not None:
                memory["model"] = model
                memory["plan"] = [secondary]
                return _issue(memory, board, primary, model)

    # More observations resolve repeated-colour or empty-cell ambiguities.
    primary = min(
        (g[0] for g in memory["groups"]),
        key=lambda i: len(memory["history"][i])
    )
    return _issue(memory, board, primary)

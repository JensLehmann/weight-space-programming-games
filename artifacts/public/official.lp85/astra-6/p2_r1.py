import math
import heapq
from collections import defaultdict, deque, Counter

import numpy as np


def _hungarian(cost):
    """Minimum-cost assignment for a square matrix."""
    n = len(cost)
    if not n:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
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
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    answer = [0] * n
    for j in range(1, n + 1):
        answer[p[j] - 1] = j - 1
    return answer


def _parse(obs):
    a = np.asarray(obs)
    background = a == 4
    background[:, 0] = False
    background[1, :] = False
    yy, xx = np.where(background)
    if not len(yy):
        return {}, {}, [], 1

    r0, r1 = int(yy.min()), int(yy.max())
    c0, c1 = int(xx.min()), int(xx.max())
    mask = np.zeros((64, 64), dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    mask &= a != 4
    mask[:, 0] = False
    mask[1, :] = False

    seen = np.zeros((64, 64), dtype=bool)
    components = []
    for rr, cc in zip(*np.where(mask)):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        color = int(a[r, c])
        stack = [(r, c)]
        seen[r, c] = True
        lo_r = hi_r = r
        lo_c = hi_c = c
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            lo_r, hi_r = min(lo_r, y), max(hi_r, y)
            lo_c, hi_c = min(lo_c, x), max(hi_c, x)
            for ny, nx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if (0 <= ny < 64 and 0 <= nx < 64
                        and mask[ny, nx] and not seen[ny, nx]
                        and int(a[ny, nx]) == color):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append((color, lo_r, lo_c,
                           hi_r - lo_r + 1, hi_c - lo_c + 1, size))

    scales = []
    for color, r, c, h, w, size in components:
        if color in (8, 14):
            small, large = min(h, w), max(h, w)
            if small % 3 == 0 and large == 4 * (small // 3):
                scales.append(small // 3)
    if scales:
        scale = Counter(scales).most_common(1)[0][0]
    else:
        scales = [
            h // 2 for color, r, c, h, w, size in components
            if color not in (3, 4, 8, 11, 12, 14)
            and h == w and h % 2 == 0 and size == h * w
        ]
        scale = Counter(scales).most_common(1)[0][0] if scales else 1

    tiles, targets, buttons = {}, {}, []
    for color, r, c, h, w, size in components:
        if (h == w == 2 * scale and size == h * w
                and color not in (4, 8, 14)):
            tiles[(r, c)] = color

        if (color in (11, 12) and h == w == scale
                and size == scale * scale
                and r + 4 * scale <= 64 and c + 4 * scale <= 64):
            good = True
            for dy, dx in ((0, 0), (0, 3 * scale),
                           (3 * scale, 0), (3 * scale, 3 * scale)):
                if not np.all(a[r + dy:r + dy + scale,
                                c + dx:c + dx + scale] == color):
                    good = False
                    break
            if good:
                targets[(r + scale, c + scale)] = color

        if color in (8, 14):
            if (min(h, w) == 3 * scale
                    and max(h, w) == 4 * scale):
                buttons.append((r + h // 2, c + w // 2,
                                color, 0 if h > w else 1))

    buttons.sort()
    return tiles, targets, buttons, scale


def _pair_buttons(buttons, scale):
    groups = []
    taken = set()
    for axis in (0, 1):
        left = [i for i, b in enumerate(buttons)
                if b[2] == 8 and b[3] == axis]
        right = [i for i, b in enumerate(buttons)
                 if b[2] == 14 and b[3] == axis]
        n = max(len(left), len(right))
        if not n:
            continue
        costs = [[1e8] * n for _ in range(n)]
        for i, li in enumerate(left):
            for j, ri in enumerate(right):
                l, r = buttons[li], buttons[ri]
                across = abs(l[0] - r[0]) if axis == 0 else abs(l[1] - r[1])
                along = abs(l[1] - r[1]) if axis == 0 else abs(l[0] - r[0])
                if across <= scale:
                    costs[i][j] = 1000 * across * across + along * along
        matching = _hungarian(costs)
        for i, j in enumerate(matching):
            if i < len(left) and j < len(right) and costs[i][j] < 1e8:
                # The right-coloured button is the representative.
                groups.append([right[j], left[i]])
                taken.update((right[j], left[i]))

    for i in range(len(buttons)):
        if i not in taken:
            groups.append([i])
    groups.sort(key=lambda g: (buttons[g[0]][0], buttons[g[0]][1]))
    return groups


def _inverse(p):
    q = [0] * len(p)
    for i, j in enumerate(p):
        q[j] = i
    return tuple(q)


def _fit_permutation(records, positions, distance, soft=False):
    if not records:
        return None

    n = len(positions)
    before_sig = [
        tuple(b.get(pos, -1) for b, a in records) for pos in positions
    ]
    after_sig = [
        tuple(a.get(pos, -1) for b, a in records) for pos in positions
    ]
    sources, destinations = defaultdict(list), defaultdict(list)
    for i, sig in enumerate(before_sig):
        sources[sig].append(i)
    for i, sig in enumerate(after_sig):
        destinations[sig].append(i)

    consistent = (
        sources.keys() == destinations.keys()
        and all(len(v) == len(destinations[k]) for k, v in sources.items())
    )
    if not consistent:
        if not soft:
            return None
        costs = []
        for i in range(n):
            row = []
            for j in range(n):
                errors = sum(x != y for x, y in
                             zip(before_sig[i], after_sig[j]))
                row.append(10000 * errors + distance[i][j])
            costs.append(row)
        return tuple(_hungarian(costs))

    permutation = list(range(n))
    for sig, src in sources.items():
        dst = destinations[sig]
        if len(src) == 1:
            permutation[src[0]] = dst[0]
        elif set(src) == set(dst):
            # All identities are feasible and have zero spatial cost.
            continue
        else:
            costs = [[distance[i][j] for j in dst] for i in src]
            matching = _hungarian(costs)
            for k, j in enumerate(matching):
                permutation[src[k]] = dst[j]
    return tuple(permutation)


def _make_models(mem, positions, scale):
    n = len(positions)
    unit = max(1, 3 * scale)
    distance = []
    for y, x in positions:
        row = []
        for yy, xx in positions:
            d = math.hypot(y - yy, x - xx) / unit
            # Convex near-neighbour costs recover runs of identical tiles.
            # A cap permits the long closing edge of an open-looking cycle.
            row.append(min(d ** 2.5, 16 + 0.25 * d) + 0.001 * d)
        distance.append(row)

    while True:
        models = {}
        split = None
        for gi, group in enumerate(mem["groups"]):
            records = list(mem["records"][group[0]])
            if len(group) == 2:
                records += [(a, b) for b, a in mem["records"][group[1]]]
            if not records:
                continue
            p = _fit_permutation(records, positions, distance)
            if p is None and len(group) == 2:
                split = gi
                break
            if p is None:
                p = _fit_permutation(records, positions, distance, soft=True)
            models[group[0]] = p
            if len(group) == 2:
                models[group[1]] = _inverse(p)

        if split is None:
            return models
        group = mem["groups"].pop(split)
        mem["groups"][split:split] = [[group[0]], [group[1]]]


def _state_function(counts):
    k, l = counts
    if k <= 1 and l <= 1:
        return lambda st, p: tuple(p[i] for i in st)
    if k == 0 or l == 0:
        return lambda st, p: tuple(sorted(p[i] for i in st))
    return lambda st, p: (
        tuple(sorted(p[i] for i in st[:k]))
        + tuple(sorted(p[i] for i in st[k:]))
    )


def _goal_function(target, target_counts, counts):
    k = counts[0]
    tk = target_counts[0]
    t0, t1 = set(target[:tk]), set(target[tk:])
    if counts == target_counts:
        return lambda st: st == target
    return lambda st: t0.issubset(st[:k]) and t1.issubset(st[k:])


def _reconstruct(meeting, forward, backward=None):
    path = []
    s = meeting
    while forward[s][0] is not None:
        parent, action = forward[s]
        path.append(action)
        s = parent
    path.reverse()
    if backward is not None:
        s = meeting
        while backward[s][0] is not None:
            s, action = backward[s]
            path.append(action)
    return path


def _search(start, target, counts, target_counts, models, order):
    if not models:
        return None, False
    change = _state_function(counts)
    is_goal = _goal_function(target, target_counts, counts)
    if is_goal(start):
        return [], False

    limit = min(70000, 280000 // max(1, len(start)))
    moves = [(a, models[a], _inverse(models[a])) for a in order]

    if counts != target_counts:
        queue = deque([start])
        visited = {start: (None, None)}
        while queue:
            s = queue.popleft()
            for a, p, ip in moves:
                t = change(s, p)
                if t in visited:
                    continue
                visited[t] = (s, a)
                if is_goal(t):
                    return _reconstruct(t, visited), False
                queue.append(t)
                if len(visited) >= limit:
                    return None, True
        return None, False

    forward = {start: (None, None)}
    backward = {target: (None, None)}
    front, back = [start], [target]
    depth = 0
    while front and back and depth < 80:
        from_start = len(front) <= len(back)
        layer = front if from_start else back
        own = forward if from_start else backward
        other = backward if from_start else forward
        new_layer = []
        for s in layer:
            for a, p, ip in moves:
                t = change(s, p if from_start else ip)
                if t in own:
                    continue
                own[t] = (s, a)
                if t in other:
                    return _reconstruct(t, forward, backward), False
                new_layer.append(t)
                if len(forward) + len(backward) >= limit:
                    return None, True
        if from_start:
            front = new_layer
        else:
            back = new_layer
        depth += 1
    return None, bool(front and back)


def _heuristic_search(start, target, counts, target_counts, models, order):
    """Bounded weighted A*, also providing a useful partial route."""
    if not models or not start:
        return None, False
    n = len(next(iter(models.values())))
    inverse_moves = [_inverse(models[a]) for a in order]
    tables = []
    for goal in target:
        distances = [10000] * n
        distances[goal] = 0
        queue = deque([goal])
        while queue:
            p = queue.popleft()
            for inv in inverse_moves:
                q = inv[p]
                if distances[q] == 10000:
                    distances[q] = distances[p] + 1
                    queue.append(q)
        tables.append(distances)

    k, tk = counts[0], target_counts[0]
    cache = {}

    def heuristic(st):
        if st in cache:
            return cache[st]
        values = []
        for j, distances in enumerate(tables):
            group = st[:k] if j < tk else st[k:]
            values.append(min((distances[p] for p in group), default=10000))
        h = max(max(values, default=0),
                (sum(values) + len(st) - 1) // len(st))
        cache[st] = h
        return h

    change = _state_function(counts)
    is_goal = _goal_function(target, target_counts, counts)
    h0 = heuristic(start)
    if h0 >= 10000:
        return None, False

    queue = [(2.5 * h0, 0, start)]
    best_g = {start: 0}
    parent = {start: (None, None)}
    best_state, best_h = start, h0
    expansions = 0

    while queue and expansions < 6000 and len(best_g) < 30000:
        _, g, s = heapq.heappop(queue)
        if g != best_g.get(s):
            continue
        expansions += 1
        hs = heuristic(s)
        if hs < best_h:
            best_h, best_state = hs, s
        if is_goal(s):
            return _reconstruct(s, parent), True
        for a in order:
            t = change(s, models[a])
            ng = g + 1
            if ng >= best_g.get(t, 100000):
                continue
            ht = heuristic(t)
            if ht >= 10000:
                continue
            best_g[t] = ng
            parent[t] = (s, a)
            heapq.heappush(queue, (ng + 2.5 * ht, ng, t))

    if best_state != start:
        return _reconstruct(best_state, parent), False
    return None, False


def act(obs, available, memory):
    tiles, targets, buttons, scale = _parse(obs)

    if not buttons:
        if "ACTION6" in available:
            return "ACTION6", 32, 32
        action = next((a for a in available if a != "RESET"),
                      available[0] if available else "RESET")
        return action, None, None

    mem = memory.get("_lp85_player")
    if mem is None:
        mem = {
            "buttons": buttons,
            "groups": _pair_buttons(buttons, scale),
            "records": [[] for _ in buttons],
            "counts": [0] * len(buttons),
            "positions": set(),
            "previous": None,
            "last_action": None,
            "goal_edges": {},
        }
        memory["_lp85_player"] = mem

    # Incorporate the preceding experiment.
    if mem["previous"] is not None and mem["last_action"] is not None:
        a = mem["last_action"]
        sample = (mem["previous"], tiles.copy())
        records = mem["records"][a]
        if not any(b == sample[0] and c == sample[1] for b, c in records):
            records.append(sample)

    mem["positions"].update(tiles)
    mem["positions"].update(targets)
    positions = sorted(mem["positions"])
    index = {p: i for i, p in enumerate(positions)}
    models = _make_models(mem, positions, scale)

    source_groups = [
        sorted(index[p] for p, color in tiles.items() if color == c)
        for c in (11, 12)
    ]
    target_groups = [
        sorted(index[p] for p, color in targets.items() if color == c)
        for c in (11, 12)
    ]
    counts = tuple(map(len, source_groups))
    target_counts = tuple(map(len, target_groups))
    start = tuple(source_groups[0] + source_groups[1])
    target = tuple(target_groups[0] + target_groups[1])

    goal_key = tuple(
        tuple(sorted(p for p, color in tiles.items() if color == c))
        for c in (11, 12)
    )
    edge_counts = mem["goal_edges"].setdefault(
        goal_key, [0] * len(mem["buttons"])
    )
    order = sorted(models, key=lambda a: (
        edge_counts[a], mem["counts"][a], a
    ))

    fresh = [
        g for g in mem["groups"]
        if not any(mem["records"][a] for a in g)
    ]
    path, limited = None, False
    if target and all(x >= y for x, y in zip(counts, target_counts)):
        path, limited = _search(
            start, target, counts, target_counts, models, order
        )

    partial = False
    if path is None and limited and not fresh:
        path, complete = _heuristic_search(
            start, target, counts, target_counts, models, order
        )
        partial = bool(path) and not complete

    chosen = None
    if path:
        if not fresh or len(path) <= 3 + len(fresh):
            chosen = path[0]

    if chosen is not None and edge_counts[chosen] >= 2 and len(buttons) > 1:
        # Repeated goal-state loops indicate an unresolved model ambiguity.
        chosen = None

    if chosen is None:
        if fresh:
            group = min(
                fresh,
                key=lambda g: sum(mem["counts"][a] for a in g)
            )
            chosen = group[0]
        else:
            # Balanced, repeated probes reveal colour signatures along cycles.
            group = min(
                mem["groups"],
                key=lambda g: (
                    sum(mem["counts"][a] for a in g)
                    + 2 * min(edge_counts[a] for a in g),
                    min(edge_counts[a] for a in g),
                )
            )
            chosen = group[0]
            if edge_counts[chosen] >= 2 and len(group) > 1:
                chosen = min(group, key=lambda a: (
                    edge_counts[a], mem["counts"][a]
                ))

    mem["previous"] = tiles.copy()
    mem["last_action"] = chosen
    mem["counts"][chosen] += 1
    edge_counts[chosen] += 1

    row, col = mem["buttons"][chosen][:2]
    return "ACTION6", int(row), int(col)

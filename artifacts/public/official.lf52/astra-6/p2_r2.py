import numpy as np
import heapq
import time
from collections import deque

_W = 64
_MARGIN = 16
_DIRS = (-64, 64, -1, 1)
_STEPS = {1: -64, 2: 64, 3: -1, 4: 1}
_DISC = ((1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4),
         (3, 1), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))


def _pack(x, y):
    return (y + _MARGIN) * _W + x + _MARGIN


def _xy(p):
    return p % _W - _MARGIN, p // _W - _MARGIN


def _screen(p, translation):
    x, y = _xy(p)
    return translation[0] + 6 * x + 3, translation[1] + 6 * y + 3


def _clickable(p, translation):
    x, y = _screen(p, translation)
    return 0 <= x < 64 and 1 <= y < 64


def _parse(obs, old_translation=None):
    a = np.asarray(obs)
    h = 59

    def sub(r, c):
        return a[r:r + h, c:c + h]

    center = sub(2, 2)
    disc = np.isin(center, (14, 8, 9, 2))
    for r, c in _DISC:
        disc &= sub(r, c) == center

    purple = ((sub(0, 1) == 15) & (sub(0, 4) == 15) &
              (sub(1, 1) == 15) & (sub(1, 3) == 7) &
              (sub(2, 2) == 7) & (sub(3, 4) == 15))
    white = ((sub(0, 0) == 0) & (sub(0, 5) == 0) &
             (sub(5, 0) == 0) & (sub(5, 5) == 0) &
             ((sub(1, 1) == 1) | (sub(1, 4) == 1)))
    yellow = ((sub(0, 0) == 11) & (sub(0, 5) == 11) &
              (sub(5, 0) == 11) & (sub(5, 5) == 11))

    votes = np.zeros(36, dtype=float)
    for mask, weight in ((disc, 9), (purple, 7), (white, 2), (yellow, 8)):
        yy, xx = np.nonzero(mask)
        if len(xx):
            votes += np.bincount((yy % 6) * 6 + xx % 6,
                                 minlength=36) * weight
    if old_translation is not None:
        votes[(old_translation[1] % 6) * 6 +
              old_translation[0] % 6] += .1
    oy, ox = divmod(int(np.argmax(votes)), 6)

    def get(x, y, r, c):
        xx, yy = x + c, y + r
        if 0 <= xx < 64 and 1 <= yy < 64:
            return int(a[yy, xx])
        return -1

    cells = []
    for j in range(-1, 12):
        y = oy + j * 6
        if y >= 64 or y + 5 < 1:
            continue
        for i in range(-1, 12):
            x = ox + i * 6
            if x >= 64 or x + 5 < 0:
                continue
            vals = [get(x, y, r, c) for r, c in _DISC]
            present = [v for v in vals if v >= 0]
            peg = None
            if len(present) >= 5:
                for color in (14, 8, 9, 2):
                    if sum(v == color for v in present) >= .82 * len(present):
                        peg = color
                        break

            pattern = ((0, 1, 15), (0, 4, 15), (1, 1, 15),
                       (1, 3, 7), (2, 2, 7), (2, 3, 7), (3, 4, 15))
            pp = [(r, c, v) for r, c, v in pattern
                  if get(x, y, r, c) >= 0]
            pillar = (len(pp) >= 3 and
                      sum(get(x, y, r, c) == v for r, c, v in pp)
                      >= .85 * len(pp))

            corners = [get(x, y, r, c) for r, c in
                       ((0, 0), (0, 5), (5, 0), (5, 5))]
            interior = [get(x, y, r, c)
                        for r in range(1, 5) for c in range(1, 5)]
            platform = 11 in corners or sum(v == 12 for v in interior) >= 3
            floor = ((0 in corners and 1 in interior) or
                     (sum(v == 0 for v in corners) >= 2 and
                      (peg is not None or pillar or 2 in interior)))
            core = [get(x, y, r, c)
                    for r, c in ((2, 2), (2, 3), (3, 2), (3, 3))]
            rail = (not floor and not pillar and peg is None and
                    sum(v == 5 for v in core) >= 3)
            terrain = 2 if platform or rail else (1 if floor else 0)
            ring = peg is None and sum(v == 2 for v in vals) >= 4
            shape = 0
            if rail:
                for bit, points in enumerate((
                    ((0, 2), (1, 2)), ((4, 2), (5, 2)),
                    ((2, 0), (2, 1)), ((2, 4), (2, 5))
                )):
                    if all(get(x, y, r, c) == 5 for r, c in points):
                        shape |= 1 << bit
            safe = (x + 1 >= 0 and x + 4 < 64 and
                    y + 1 >= 1 and y + 4 < 64)
            cells.append((i, j, terrain, peg, platform, pillar,
                          ring, shape, safe))
    return ox, oy, cells


def _register(m, ox, oy, cells):
    if "translation" not in m:
        return ox, oy
    tx, ty = m["translation"]
    terrain, pegs = m["terrain"], m["pegs"]
    platforms, pillars = m["platforms"], m["pillars"]
    shapes = m.get("rail_shapes", {})
    focus = m.get("focus")
    old_kx = int(round((tx - ox) / 6))
    old_ky = int(round((ty - oy) / 6))
    best, best_score = (tx, ty), -1e100

    for ky in range(old_ky - 9, old_ky + 10):
        for kx in range(old_kx - 12, old_kx + 13):
            ntx, nty = ox + 6 * kx, oy + 6 * ky
            score = -.025 * abs(ntx - tx) - .04 * abs(nty - ty)
            for i, j, t, c, pl, pi, ring, shape, safe in cells:
                p = _pack(i - kx, j - ky)
                old = terrain.get(p)
                if old is not None:
                    if old == t:
                        score += 2.5 if t == 1 else (2.1 if t == 2 else .035)
                    elif safe:
                        score -= 3.6 if old or t else 0
                    if t == old == 2 and shape and p in shapes:
                        score += .9 if shapes[p] == shape else -.9

                oldc = pegs.get(p)
                if c is not None:
                    if oldc is not None:
                        score += 13 if c == oldc or c == 2 else -4
                    else:
                        score -= 2.5
                elif safe and oldc is not None:
                    score -= 10
                if pl:
                    score += 8 if p in platforms else -2
                elif safe and p in platforms:
                    score -= 5
                if pi:
                    score += 6 if p in pillars else -2
                elif safe and p in pillars:
                    score -= 4
                if focus is not None and p == focus and safe:
                    score += 32 if c is not None and (
                        c == pegs.get(p) or c == 2) else -24
            if score > best_score:
                best_score, best = score, (ntx, nty)
    return best


def _observe(obs, m):
    ox, oy, raw = _parse(obs, m.get("translation"))
    translation = _register(m, ox, oy, raw)
    kx, ky = (translation[0] - ox) // 6, (translation[1] - oy) // 6
    m["translation"] = translation
    terrain, pegs = m["terrain"], m["pegs"]
    platforms, pillars = m["platforms"], m["pillars"]
    shapes = m.setdefault("rail_shapes", {})
    old_pegs = pegs.copy()
    records, visible, rings = [], set(), set()
    gray = 0
    for i, j, t, color, pl, pi, ring, shape, safe in raw:
        p = _pack(i - kx, j - ky)
        records.append((p, t, color, pl, pi, ring, shape, safe))
        if safe:
            visible.add(p)
            pegs.pop(p, None)
            platforms.discard(p)
            pillars.discard(p)
        if t or safe:
            terrain[p] = t
        if shape:
            shapes[p] = shape
    for p, t, color, pl, pi, ring, shape, safe in records:
        if color is not None:
            if color == 2:
                gray += 1
                color = old_pegs.get(p, 14)
            pegs[p] = color
        if pl:
            platforms.add(p)
        if pi:
            pillars.add(p)
        if ring:
            rings.add(p)
    m["visible_cells"], m["rings"], m["gray"] = visible, rings, gray
    return records


def _state(m):
    pegs, platforms = m["pegs"], m["platforms"]
    return (tuple(sorted(p for p, c in pegs.items() if c == 14)),
            tuple(sorted(p for p, c in pegs.items() if c == 8)),
            tuple(sorted(p for p, c in pegs.items() if c == 9)),
            tuple(sorted(platforms)),
            tuple(sorted(m["pillars"] & platforms)))


def _snapshot(m):
    return (m["pegs"].copy(), m["platforms"].copy(),
            m["pillars"].copy(), m["translation"])


def _restore(m, snapshot):
    pegs, platforms, pillars, translation = snapshot
    m["pegs"], m["platforms"] = pegs.copy(), platforms.copy()
    m["pillars"], m["translation"] = pillars.copy(), translation
    m["focus"], m["plan"], m["pending"] = None, [], None


def _platform_transition(platforms, step, rails, cache):
    key = platforms, step
    if key in cache:
        return cache[key]
    occupied = set(platforms)
    if abs(step) == 1:
        order = sorted(platforms, key=lambda p: p % 64, reverse=step > 0)
    else:
        order = sorted(platforms, key=lambda p: p // 64, reverse=step > 0)
    mapping = {}
    for p in order:
        q = p + step
        if q in rails and q not in occupied:
            occupied.remove(p)
            occupied.add(q)
            mapping[p] = q
    result = tuple(sorted(occupied)), mapping
    cache[key] = result
    return result


def _arrow_state(state, step, rails, cache):
    platforms, mapping = _platform_transition(state[3], step, rails, cache)
    if not mapping:
        return None
    groups = [tuple(sorted(mapping.get(p, p) for p in state[i]))
              for i in (0, 1, 2)]
    pillars = tuple(sorted(mapping.get(p, p) for p in state[4]))
    return tuple(groups) + (platforms, pillars)


def _legal_jump(m, source, dest):
    pegs = m["pegs"]
    if source not in pegs or dest in pegs or dest in m["pillars"]:
        return False
    delta = dest - source
    if delta not in (-128, 128, -2, 2):
        return False
    middle = source + delta // 2
    return ((middle in pegs or middle in m["pillars"]) and
            (m["terrain"].get(dest) == 1 or dest in m["platforms"]))


def _jump_state(state, source, dest):
    for ci in range(3):
        if source in state[ci]:
            group = set(state[ci])
            group.remove(source)
            group.add(dest)
            if ci < 2:
                group.discard((source + dest) // 2)
            parts = list(state)
            parts[ci] = tuple(sorted(group))
            return tuple(parts)
    return state


def _apply_jump(m, source, dest):
    color = m["pegs"].pop(source)
    middle = (source + dest) // 2
    if color != 9 and m["pegs"].get(middle) == color:
        m["pegs"].pop(middle, None)
    m["pegs"][dest] = color
    m["focus"] = dest


def _apply_arrow(m, action):
    rails = {p for p, t in m["terrain"].items() if t == 2}
    platforms, mapping = _platform_transition(
        tuple(sorted(m["platforms"])), _STEPS[action], rails, {})
    m["platforms"] = set(platforms)
    old_pegs = m["pegs"]
    moved_green = [mapping[p] for p, c in old_pegs.items()
                   if c == 14 and p in mapping]
    m["pegs"] = {mapping.get(p, p): c for p, c in old_pegs.items()}
    m["pillars"] = {mapping.get(p, p) for p in m["pillars"]}
    m["focus"] = max(moved_green) if moved_green else None


def _frontier(obs, m, records):
    tx, ty = m["translation"]
    counts = [0, 0, 0, 0]
    for p, t, color, pl, pi, ring, shape, safe in records:
        if not t:
            continue
        x, y = _xy(p)
        xx, yy = tx + 6 * x, ty + 6 * y
        counts[0] += int(xx + 5 >= 63)
        counts[1] += int(xx < 0)
        counts[2] += int(yy + 5 >= 63)
        counts[3] += int(yy < 1)

    # A tile whose only visible column is its outer border cannot be
    # classified as a tile, but it still proves the board continues.
    edge_colors = (0, 1, 5, 7, 8, 11, 12, 14, 15)
    a = np.asarray(obs)
    counts[0] += int(np.isin(a[2:, 63], edge_colors).sum() >= 2) * 3
    counts[1] += int(np.isin(a[2:, 0], edge_colors).sum() >= 2) * 3
    counts[2] += int(np.isin(a[63, :], edge_colors).sum() >= 2) * 3
    counts[3] += int(np.isin(a[1, :], edge_colors).sum() >= 2) * 3

    initial = m.get("initial_translation", (tx, ty))
    direction = None
    if counts[0]:
        direction = 0
    elif counts[2] and ty >= initial[1] - 3:
        direction = 2
    elif counts[3] and ty <= initial[1] + 3 and not counts[2]:
        direction = 3
    elif counts[1] and tx >= initial[0] - 3:
        direction = 1

    if direction is None:
        off = [0, 0, 0, 0]
        for p, c in m["pegs"].items():
            if c == 9:
                continue
            x, y = _screen(p, (tx, ty))
            if x >= 64:
                off[0] += 1
            elif x < 0:
                off[1] += 1
            elif y >= 64:
                off[2] += 1
            elif y < 1:
                off[3] += 1
        if any(off):
            direction = max(range(4), key=lambda i: off[i])

    if direction is None:
        return None, set()
    m["travel_direction"] = direction
    targets = set()
    for p, t in m["terrain"].items():
        if not t:
            continue
        x, y = _screen(p, (tx, ty))
        if ((direction == 0 and x >= 52) or
            (direction == 1 and x <= 12) or
            (direction == 2 and y >= 52) or
            (direction == 3 and y <= 12)):
            targets.add(p)
    return direction, targets


def _optimistic_distances(nodes, rails, targets):
    distance = {p: 0 for p in targets if p in nodes}
    queue = deque(distance)
    while queue:
        p = queue.popleft()
        nd = distance[p] + 1
        for step in _DIRS:
            q = p + 2 * step
            if q in nodes and p + step in nodes and q not in distance:
                distance[q] = nd
                queue.append(q)
            q = p + step
            if p in rails and q in rails and q not in distance:
                distance[q] = nd
                queue.append(q)
    return distance


def _geometry(m, floor, rails, fixed):
    key = frozenset(floor), frozenset(rails), frozenset(fixed)
    if m.get("geometry_key") == key:
        return m["geometry"]
    nodes = sorted((floor | rails) - fixed)
    ix = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    d = np.full((n, n), 1000., dtype=np.float32)
    r = np.full((n, n), 1000., dtype=np.float32)
    np.fill_diagonal(d, 0.)
    np.fill_diagonal(r, 0.)
    adjacency = {}
    allnodes = floor | rails | fixed
    configurations = []
    for p in nodes:
        i = ix[p]
        edges = []
        for step in _DIRS:
            q, middle = p + 2 * step, p + step
            if q in ix and middle in allnodes:
                d[i, ix[q]] = 2.
                edges.append((q, middle))
                if middle in ix:
                    configurations.append((i, ix[middle]))
            q = p + step
            if p in rails and q in rails and q in ix:
                d[i, ix[q]] = 1.
                r[i, ix[q]] = 1.
                edges.append((q, None))
        adjacency[p] = edges
    for k in range(n):
        np.minimum(d, d[:, k:k+1] + d[k:k+1, :], out=d)
        np.minimum(r, r[:, k:k+1] + r[k:k+1, :], out=r)
    cap = np.full((n, n), 1000., dtype=np.float32)
    for i, j in configurations:
        np.minimum(cap, d[:, i:i+1] + d[:, j:j+1].T + 2., out=cap)
    cap = np.minimum(cap, cap.T)
    result = ix, d.tolist(), r.tolist(), cap.tolist(), adjacency
    m["geometry_key"], m["geometry"] = key, result
    return result


def _complex_metric_factory(m, floor, rails, fixed, minimum):
    ix, d, rd, cap, adjacency = _geometry(m, floor, rails, fixed)

    def distance(a, b):
        ia, ib = ix.get(a), ix.get(b)
        return d[ia][ib] if ia is not None and ib is not None else 1000.

    def rail_distance(a, b):
        ia, ib = ix.get(a), ix.get(b)
        return rd[ia][ib] if ia is not None and ib is not None else 1000.

    def capture(a, b):
        ia, ib = ix.get(a), ix.get(b)
        return cap[ia][ib] if ia is not None and ib is not None else 1000.

    def metric(state):
        nonblue = state[0] + state[1]
        n = len(nonblue)
        occupied = set(nonblue + state[2])
        platforms = set(state[3])
        mp = set(state[4])
        supports = occupied | fixed | mp
        allpegs = nonblue + state[2]
        empty_platforms = platforms - occupied - mp
        mst = 0.
        best_pair = None
        best_pair_cost = 1001.

        for group in state[:2]:
            if len(group) < 2:
                continue
            used = {group[0]}
            left = set(group[1:])
            while left:
                value, q = min((min(capture(p, q) for p in used), q)
                               for q in left)
                mst += min(90., value)
                used.add(q)
                left.remove(q)
            for ai in range(len(group)):
                for bi in range(ai + 1, len(group)):
                    a, b = group[ai], group[bi]
                    value = capture(a, b)
                    if value < best_pair_cost:
                        best_pair_cost, best_pair = value, (a, b)

        prep = 0.
        if best_pair is not None and best_pair_cost < 999:
            a, b = best_pair

            def support_cost(p, middle):
                if middle in supports:
                    return 0.
                value = min((distance(q, middle) for q in allpegs if q != p),
                            default=1000.)
                if middle in rails:
                    value = min(value, min(
                        (rail_distance(q, middle) for q in mp), default=1000.))
                return min(45., value)

            def landing_cost(p, q):
                if q not in rails:
                    return 3. if q in occupied else 0.
                if q in empty_platforms:
                    return 0.
                choices = [rail_distance(t, q) for t in empty_platforms
                           if t != p]
                if choices:
                    return min(40., min(choices))
                choices = [rail_distance(t, q) + 8 for t in platforms
                           if t != p and t not in mp]
                return min(40., min(choices, default=40.))

            options = []
            for p, other in ((a, b), (b, a)):
                delta = other - p
                if delta in _DIRS:
                    landing = other + delta
                    if landing in ix:
                        options.append(2. + .8 * landing_cost(p, landing))
                for q, middle in adjacency.get(p, ()):
                    remaining = capture(q, other)
                    stepcost = 1. if middle is None else 2.
                    # Demand geometric progress, so moving back and forth
                    # cannot hide a necessary support-placement task.
                    if remaining >= best_pair_cost - .2:
                        continue
                    if middle is None:
                        if p in platforms:
                            options.append(stepcost + remaining)
                    else:
                        extra = (.85 * support_cost(p, middle) +
                                 .8 * landing_cost(p, q))
                        options.append(stepcost + remaining + extra)
            if options:
                prep = max(0., min(50., min(options) - best_pair_cost))

        heuristic = max(0, n - minimum) * 7. + mst + prep
        value = n * 90. + mst * 4. + prep * 4.
        return heuristic, value

    return metric


def _single_can_escape(state, floor, rails, fixed, targets):
    pieces = state[0] + state[1]
    if len(pieces) != 1 or state[2]:
        return True
    source = pieces[0]
    supports = fixed | set(state[4])
    if state[4]:
        supports |= rails
    seen, queue = {source}, deque([source])
    while queue:
        p = queue.popleft()
        if p in targets:
            return True
        for step in _DIRS:
            q = p + 2 * step
            if (p + step in supports and q not in fixed and
                    (q in floor or q in rails) and q not in seen):
                seen.add(q)
                queue.append(q)
            q = p + step
            if state[3] and p in rails and q in rails and q not in seen:
                seen.add(q)
                queue.append(q)
    return False


def _pan_allowed(new, command, m):
    if command[0] != "j":
        return True
    tx, ty = m["translation"]
    pan = m.get("pan_rules", {}).get((command[2], tx, ty))
    if pan is None:
        return True
    ntx, nty = pan
    for p in new[0] + new[1]:
        x, y = _screen(p, pan)
        if ((ntx < tx and x < 1) or (ntx > tx and x >= 63) or
            (nty < ty and y < 2) or (nty > ty and y >= 63)):
            return False
    return True


def _search(m, targets, deadline, move_blue=True, limit=190000):
    floor = {p for p, t in m["terrain"].items() if t == 1}
    rails = {p for p, t in m["terrain"].items() if t == 2}
    nodes = floor | rails
    fixed = m["pillars"] - m["platforms"]
    start = _state(m)
    partial = bool(targets)
    minimum = max(1, int(bool(start[0])) + int(bool(start[1])))
    translation = m["translation"]
    clickable = {p for p in nodes if _clickable(p, translation)}
    visits = m.get("visits", {})
    bad_edges = m.get("bad_edges", set())

    jumps = {}
    for p in nodes:
        if p in clickable:
            jumps[p] = [(p + 2 * s, p + s) for s in _DIRS
                        if p + 2 * s in nodes and
                        p + 2 * s in clickable and p + s in nodes]

    distance = _optimistic_distances(nodes, rails, targets) if partial else {}
    advanced = not partial and bool(start[1] or start[2])
    if advanced:
        metric = _complex_metric_factory(m, floor, rails, fixed, minimum)
    else:
        def metric(state):
            n = len(state[0]) + len(state[1])
            spread = 0
            for group in state[:2]:
                if len(group) > 1:
                    xs, ys = [p % 64 for p in group], [p // 64 for p in group]
                    spread += max(xs) - min(xs) + max(ys) - min(ys)
            escape = 0.
            if partial and n:
                ds = [distance.get(p, 45) for p in state[0] + state[1]]
                escape = max(ds) * .65 + sum(ds) / (n ** .35) * .35
            return (max(0, n - minimum) * 17 + spread * .42 + escape * 1.4,
                    n * 90 + spread * .45 + escape * 3.)

    h0, v0 = metric(start)
    best_value = v0 + min(30, visits.get(start, 0) * 5)
    best_node = 0
    states, parents, moves, costs = [start], [-1], [None], [0]
    heap = [(h0 * 2.35, 0, 0)]
    best_cost = {start: 0}
    serial, expansions, goal_node = 0, 0, None
    arrow_cache = {}

    def add(state, parent, command, cost):
        nonlocal serial, best_value, best_node
        if cost >= best_cost.get(state, 10 ** 9) or cost > 215:
            return
        if parent == 0 and (start, command) in bad_edges:
            return
        if not _pan_allowed(state, command, m):
            return
        best_cost[state] = cost
        heuristic, value = metric(state)
        value += cost * .075 + min(45, visits.get(state, 0) * 8)
        if value < best_value - .01:
            if partial and not _single_can_escape(
                    state, floor, rails, fixed, targets):
                value += 350
            if value < best_value - .01:
                best_value, best_node = value, len(states)
        node = len(states)
        states.append(state)
        parents.append(parent)
        moves.append(command)
        costs.append(cost)
        serial += 1
        priority = cost + (2.7 if advanced else 2.15) * heuristic
        priority += min(25, visits.get(state, 0) * 3)
        heapq.heappush(heap, (priority, serial, node))

    while heap:
        if expansions % 64 == 0 and (
                time.perf_counter() >= deadline or len(best_cost) >= limit):
            break
        _, _, node = heapq.heappop(heap)
        state, cost = states[node], costs[node]
        if cost != best_cost.get(state):
            continue
        expansions += 1
        if node:
            nonblue = state[0] + state[1]
            if partial:
                if (nonblue and all(p in targets for p in nonblue) and
                        state[:2] != start[:2] and not visits.get(state, 0)):
                    goal_node = node
                    break
            elif len(nonblue) <= minimum and moves[node][0] == "j":
                goal_node = node
                break

        groups = [set(state[i]) for i in range(3)]
        occupied = groups[0] | groups[1] | groups[2]
        pillars = fixed | set(state[4])
        support = occupied | pillars
        landing = floor | set(state[3])
        for ci in range(3 if move_blue else 2):
            for source in state[ci]:
                for dest, middle in jumps.get(source, ()):
                    if (middle not in support or dest not in landing or
                            dest in occupied or dest in pillars):
                        continue
                    group = groups[ci].copy()
                    group.remove(source)
                    group.add(dest)
                    if ci < 2:
                        group.discard(middle)
                    parts = list(state)
                    parts[ci] = tuple(sorted(group))
                    add(tuple(parts), node, ("j", source, dest), cost + 2)

        for action, step in _STEPS.items():
            new = _arrow_state(state, step, rails, arrow_cache)
            if new is not None:
                add(new, node, ("a", action), cost + 1)

    chosen = goal_node if goal_node is not None else best_node
    if chosen == 0:
        return [], False, best_value
    path = []
    while parents[chosen] >= 0:
        path.append(moves[chosen])
        chosen = parents[chosen]
    path.reverse()
    return path, goal_node is not None, best_value


def _make_plan(m, targets, deadline):
    blue = any(c == 9 for c in m["pegs"].values())
    if blue and targets:
        first_deadline = min(deadline, time.perf_counter() + 1.8)
        p1, solved1, v1 = _search(
            m, targets, first_deadline, move_blue=False, limit=85000)
        if solved1:
            return p1
        if time.perf_counter() < deadline - .15:
            p2, solved2, v2 = _search(
                m, targets, deadline, move_blue=True, limit=190000)
            if solved2 or (p2 and (not p1 or v2 < v1)):
                return p2
        return p1
    return _search(m, targets, deadline,
                   move_blue=blue, limit=230000)[0]


def _emit_selection(m, source, dest):
    m["pending"], m["focus"] = (source, dest), source
    m["last"], m["before"] = ("s", source, dest), None
    x, y = _screen(source, m["translation"])
    return "ACTION6", int(y), int(x)


def _emit_arrow(m, action):
    m["before"] = _snapshot(m)
    m["last_edge"] = _state(m), ("a", action)
    _apply_arrow(m, action)
    m["last"] = ("a", action)
    return "ACTION" + str(action), None, None


def _recover(m, available):
    m["plan"], m["pending"], m["focus"], m["stuck"] = [], None, None, 0
    before = m.get("before")
    if before is not None and "ACTION7" in available:
        _restore(m, before)
        m["before"], m["last"] = None, ("undo",)
        return "ACTION7", None, None
    if "RESET" in available and m.get("resets", 0) < 2:
        resets = m.get("resets", 0) + 1
        m.clear()
        m["resets"] = resets
        return "RESET", None, None
    if "ACTION7" in available:
        m["last"] = ("undo",)
        return "ACTION7", None, None
    return None


def _camera_safety(m, previous, before, expected, available):
    # Platform-induced camera movement is reversible. Only constrain
    # large, jump-triggered transitions that hide an unfinished peg.
    if before is None or not previous or previous[0] != "j":
        return None
    old_translation, new_translation = before[3], m["translation"]
    if old_translation == new_translation:
        return None
    tx, ty = old_translation
    ntx, nty = new_translation
    m["pan_rules"][(previous[2], tx, ty)] = new_translation
    stranded = False
    for p, color in expected.items():
        if color == 9 or p not in m["pegs"]:
            continue
        ox, oy = _screen(p, old_translation)
        nx, ny = _screen(p, new_translation)
        if ((ntx < tx and ox >= 1 and nx < 1) or
            (ntx > tx and ox < 63 and nx >= 63) or
            (nty < ty and oy >= 2 and ny < 2) or
            (nty > ty and oy < 63 and ny >= 63)):
            stranded = True
            break
    if stranded and "ACTION7" in available:
        newly_seen = {p: c for p, c in m["pegs"].items()
                      if p not in expected and
                      not _clickable(p, old_translation) and
                      p not in m["platforms"]}
        _restore(m, before)
        m["pegs"].update(newly_seen)
        m["before"], m["last"], m["stuck"] = None, ("undo",), 0
        return "ACTION7", None, None
    return None


def _fallback(m, available, direction):
    state = _state(m)
    visits, translation = m["visits"], m["translation"]
    candidates = []
    trials = m.setdefault("jump_trials", {})
    for source, color in m["pegs"].items():
        if not _clickable(source, translation):
            continue
        for step in _DIRS:
            dest = source + 2 * step
            if (not _clickable(dest, translation) or
                    not _legal_jump(m, source, dest)):
                continue
            new = _jump_state(state, source, dest)
            command = ("j", source, dest)
            if not _pan_allowed(new, command, m):
                continue
            if (state, command) in m["bad_edges"]:
                continue
            removes = color != 9 and m["pegs"].get(source + step) == color
            dx, dy = _xy(dest)
            sx, sy = _xy(source)
            progress = (dx - sx if direction == 0 else
                        sx - dx if direction == 1 else
                        dy - sy if direction == 2 else
                        sy - dy if direction == 3 else 0)
            key = state, source, dest
            score = 20 * removes + 2 * progress + (color != 9) * 3
            score -= 12 * trials.get(key, 0) + 8 * visits.get(new, 0)
            candidates.append((score, source, dest, key))

    rails = {p for p, t in m["terrain"].items() if t == 2}
    arrows = []
    for action, step in _STEPS.items():
        if "ACTION" + str(action) not in available:
            continue
        command = ("a", action)
        if (state, command) in m["bad_edges"]:
            continue
        new = _arrow_state(state, step, rails, {})
        if new is None:
            continue
        score = -8 * visits.get(new, 0)
        desired = {0: 4, 1: 3, 2: 2, 3: 1}.get(direction, 4)
        if action == desired:
            score += 2
        if new[:2] != state[:2]:
            score += 3
        arrows.append((score, -action, action))

    if candidates and "ACTION6" in available:
        score, source, dest, key = max(candidates)
        if not arrows or score >= max(arrows)[0]:
            trials[key] = trials.get(key, 0) + 1
            m["plan"] = [("j", source, dest)]
            return _emit_selection(m, source, dest)
    if arrows:
        return _emit_arrow(m, max(arrows)[2])
    if candidates and "ACTION6" in available:
        _, source, dest, key = max(candidates)
        trials[key] = trials.get(key, 0) + 1
        m["plan"] = [("j", source, dest)]
        return _emit_selection(m, source, dest)
    return _recover(m, available)


def act(obs, available, memory):
    started = time.perf_counter()
    available = set(available)
    m = memory
    if not m.get("_initialized"):
        resets = m.get("resets", 0)
        m.clear()
        m.update(_initialized=True, terrain={}, pegs={}, platforms=set(),
                 pillars=set(), plan=[], pending=None, visits={}, stuck=0,
                 resets=resets, pan_rules={}, bad_edges=set())

    previous = m.get("last")
    before = m.get("before")
    expected = m["pegs"].copy()
    old_translation = m.get("translation")
    records = _observe(obs, m)
    m.setdefault("initial_translation", m["translation"])

    result = _camera_safety(m, previous, before, expected, available)
    if result is not None:
        return result

    current = _state(m)
    if previous and previous[0] == "j":
        _, source, dest, color = previous
        if m["pegs"].get(source) == color and dest not in m["pegs"]:
            m["plan"], m["pending"] = [], None
            m["stuck"] = m.get("stuck", 0) + 1
            edge = m.get("last_edge")
            if edge is not None:
                m["bad_edges"].add(edge)
        else:
            m["stuck"] = 0

    if previous and previous[0] == "a" and before is not None:
        oldpegs, oldplatforms, oldpillars, oldt = before
        if (m["pegs"] == oldpegs and m["platforms"] == oldplatforms and
                m["pillars"] == oldpillars and m["translation"] == oldt):
            edge = m.get("last_edge")
            if edge is not None:
                m["bad_edges"].add(edge)
            m["plan"] = []

    visits = m["visits"]
    if not previous or previous[0] != "s":
        visits[current] = visits.get(current, 0) + 1
        if visits[current] >= 4:
            m["plan"] = []
            edge = m.get("last_edge")
            if edge is not None and visits[current] >= 6:
                m["bad_edges"].add(edge)

    if m["gray"] and m.get("gray_recoveries", 0) < 3:
        m["gray_recoveries"] = m.get("gray_recoveries", 0) + 1
        result = _recover(m, available)
        if result is not None:
            return result

    pending = m.get("pending")
    if pending is not None and "ACTION6" in available:
        source, dest = pending
        if _legal_jump(m, source, dest) and _clickable(dest, m["translation"]):
            color = m["pegs"][source]
            m["before"] = _snapshot(m)
            m["last_edge"] = current, ("j", source, dest)
            m["pending"] = None
            plan = m.get("plan", [])
            if plan and plan[0] == ("j", source, dest):
                plan.pop(0)
            _apply_jump(m, source, dest)
            m["last"] = ("j", source, dest, color)
            x, y = _screen(dest, m["translation"])
            return "ACTION6", int(y), int(x)
        m["pending"], m["plan"] = None, []

    if m.get("stuck", 0) >= 3:
        result = _recover(m, available)
        if result is not None:
            return result

    direction, targets = _frontier(obs, m, records)
    old_mode = m.get("partial_mode")
    partial_mode = bool(targets)
    if old_mode is not None and old_mode != partial_mode:
        m["plan"] = []
    m["partial_mode"] = partial_mode
    plan = m.get("plan", [])

    for attempt in range(2):
        while plan:
            command = plan[0]
            old = _state(m)
            if (old, command) in m["bad_edges"]:
                plan.clear()
                break
            if command[0] == "j":
                _, source, dest = command
                if ("ACTION6" in available and
                    _legal_jump(m, source, dest) and
                    _clickable(source, m["translation"]) and
                    _clickable(dest, m["translation"]) and
                    _pan_allowed(_jump_state(old, source, dest), command, m)):
                    return _emit_selection(m, source, dest)
                plan.clear()
                break
            action = command[1]
            if "ACTION" + str(action) not in available:
                plan.clear()
                break
            rails = {p for p, t in m["terrain"].items() if t == 2}
            new = _arrow_state(old, _STEPS[action], rails, {})
            if new is None:
                plan.clear()
                break
            plan.pop(0)
            return _emit_arrow(m, action)

        if attempt == 0 and time.perf_counter() < started + 7.9:
            plan = _make_plan(m, targets, started + 8.25)
            m["plan"] = plan
        else:
            break

    result = _fallback(m, available, direction)
    if result is not None:
        return result
    if "ACTION6" in available:
        for p in m["pegs"]:
            if _clickable(p, m["translation"]):
                x, y = _screen(p, m["translation"])
                m["last"] = None
                return "ACTION6", int(y), int(x)
        return "ACTION6", 32, 32
    for name in ("ACTION7", "ACTION4", "ACTION3", "ACTION1", "ACTION2", "RESET"):
        if name in available:
            return name, None, None
    return "RESET", None, None

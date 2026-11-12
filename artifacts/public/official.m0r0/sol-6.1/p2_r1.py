import time
import heapq
import itertools
from collections import deque, Counter
import numpy as np


SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def geometry(obs):
    floor = obs == 5
    xs = np.sum(floor[2:62, 1:] != floor[2:62, :-1], axis=0)
    ys = np.sum(floor[1:, 2:62] != floor[:-1, 2:62], axis=1)

    def score_axis(values, scale):
        values = values.astype(float).copy()
        values[values < max(3, values.max() * .12)] = 0
        total = values.sum() + 1e-9
        scores = [
            sum(values[i] for i in range(63)
                if (i + 1 - phase) % scale == 0) / total
            for phase in range(scale)
        ]
        phase = max(range(scale), key=lambda p: scores[p])
        return scores[phase], phase

    best = None
    for scale in range(2, 9):
        a, ox = score_axis(xs, scale)
        b, oy = score_axis(ys, scale)
        item = (a + b + .004 * scale, scale, ox, oy)
        if best is None or item[0] > best[0]:
            best = item
    _, scale, ox, oy = best
    return scale, ox, oy, (64 - ox) // scale, (64 - oy) // scale


def read_board(obs, geo):
    scale, ox, oy, w, h = geo
    wall_colors = {
        int(obs[2, 2]), int(obs[2, 61]),
        int(obs[61, 2]), int(obs[61, 61]), 0
    } - {5}
    walls, hazards = set(), set()
    actors, blocks = {}, []
    selected = None
    colored = {12: set(), 14: set(), 15: set()}

    for y in range(h):
        for x in range(w):
            p = y * w + x
            y0, x0 = oy + y * scale, ox + x * scale
            tile = obs[max(1, y0):min(63, y0 + scale),
                       x0:x0 + scale]
            if not tile.size:
                walls.add(p)
                continue
            vals = set(map(int, np.unique(tile)))
            center = int(obs[min(62, y0 + scale // 2),
                             min(63, x0 + scale // 2)])
            other = vals - {5}
            if center in (9, 11) and 5 in vals and other == {center}:
                blocks.append(p)
                if center == 11:
                    selected = p
            elif vals in ({10}, {1}):
                actors[p] = next(iter(vals))
            elif len(vals) == 1 and center in colored and center not in wall_colors:
                colored[center].add(p)
            elif 5 in vals and other:
                alternating = (
                    np.count_nonzero(tile[:, 1:] != tile[:, :-1]) +
                    np.count_nonzero(tile[1:, :] != tile[:-1, :])
                )
                if not other <= wall_colors or alternating > tile.size:
                    hazards.add(p)
            elif 5 not in vals:
                walls.add(p)

    return {
        "walls": walls, "actors": actors,
        "blocks": tuple(sorted(blocks)), "selected": selected,
        "colored": colored, "hazards": hazards
    }


def components(cells, w):
    remaining = set(cells)
    result = []
    while remaining:
        root = remaining.pop()
        part, stack = {root}, [root]
        while stack:
            p = stack.pop()
            for q in (p - w, p + w, p - 1, p + 1):
                if q in remaining and (
                    abs(q - p) == w or q // w == p // w
                ):
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def update_model(memory, reading, geo):
    _, _, _, w, h = geo
    fresh = "model" not in memory
    if fresh:
        memory["model"] = {
            "w": w, "h": h, "walls": set(reading["walls"]),
            "plates": {12: set(), 14: set(), 15: set()},
            "gates": {12: set(), 14: set(), 15: set()},
            "hazards": set()
        }
    m = memory["model"]

    def signature():
        return (
            frozenset(m["walls"]), frozenset(m["hazards"]),
            tuple(frozenset(m["plates"][c]) for c in (12, 14, 15)),
            tuple(frozenset(m["gates"][c]) for c in (12, 14, 15))
        )

    old = signature()
    for c in (12, 14, 15):
        for part in components(reading["colored"][c], w):
            if len(part) >= 2 or part & m["gates"][c]:
                m["gates"][c].update(part)
                m["plates"][c].difference_update(part)
            else:
                m["plates"][c].update(part)
        m["walls"].difference_update(m["plates"][c] | m["gates"][c])
    m["hazards"].update(reading["hazards"])
    m["walls"].difference_update(
        set(reading["actors"]) | set(reading["blocks"]) | m["hazards"]
    )
    changed = fresh or old != signature()

    if changed:
        n = w * h
        nav = [list(range(n)) for _ in range(4)]
        for p in range(n):
            if p in m["walls"]:
                continue
            x, y = p % w, p // w
            for a, (dx, dy) in enumerate(DIRS):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    q = yy * w + xx
                    if q not in m["walls"]:
                        nav[a][p] = q
        m["nav"] = nav
        m["gate_at"] = [0] * n
        m["plate_at"] = [0] * n
        for bit, c in enumerate((12, 14, 15)):
            for p in m["gates"][c]:
                m["gate_at"][p] |= 1 << bit
            for p in m["plates"][c]:
                m["plate_at"][p] |= 1 << bit
        m["all_plates"] = set().union(*m["plates"].values())
        m["floor"] = [p for p in range(n) if p not in m["walls"]]
        m["index"] = {p: i for i, p in enumerate(m["floor"])}
        m["relax"] = {}
        m["graph_dist"] = {}
        m["hc"] = {}
        m["fixed_cache"] = {}
        m.pop("parking_scores", None)
    return m, changed


def observed_open(m, reading):
    result = 0
    for bit, c in enumerate((12, 14, 15)):
        if m["gates"][c] and not (m["gates"][c] & reading["colored"][c]):
            result |= 1 << bit
    return result


def plate_open(m, actors):
    result = 0
    for p in actors:
        if p >= 0:
            result |= m["plate_at"][p]
    return result


def transformed_action(a, identity):
    dx, dy = DIRS[a]
    sx, sy = SIGNS[identity]
    dx, dy = dx * sx, dy * sy
    return 0 if dy < 0 else 1 if dy > 0 else 2 if dx < 0 else 3


def transition(m, state, action):
    actors, blocks, mode, opened = state
    if action >= 10:
        b = action - 10
        if b >= len(blocks):
            return None
        return (actors, blocks, b, opened), ()
    if action == 9:
        return (actors, blocks, -1, opened), ()
    a = action - 1
    if not 0 <= a < 4:
        return None

    nav, gates = m["nav"], m["gate_at"]
    if mode >= 0:
        p = blocks[mode]
        q = nav[a][p]
        occupied = set(blocks)
        occupied.discard(p)
        if (q in occupied or q in actors or q in m["all_plates"]
                or q in m["hazards"] or gates[q] & ~opened):
            q = p
        if q == p:
            return state, ()
        bs = list(blocks)
        bs[mode] = q
        return (actors, tuple(bs), mode, plate_open(m, actors)), ()

    positions = list(actors)
    block_set = set(blocks)
    active = [i for i, p in enumerate(actors) if p >= 0]
    for i in active:
        p = actors[i]
        q = nav[transformed_action(a, i)][p]
        if q in block_set or gates[q] & ~opened:
            q = p
        positions[i] = q

    if any(p in m["hazards"] for p in positions if p >= 0):
        return None

    w = m["w"]
    for ii, i in enumerate(active):
        for j in active[ii + 1:]:
            if (actors[i] // w == actors[j] // w
                    and abs(actors[i] - actors[j]) == 1
                    and (positions[i] == actors[j] or positions[j] == actors[i])):
                x = (positions[i] % w + positions[j] % w) // 2
                y = (positions[i] // w + positions[j] // w) // 2
                positions[i] = positions[j] = y * w + x

    groups = {}
    for i in active:
        groups.setdefault(positions[i], []).append(i)
    retired = []
    for p, ids in groups.items():
        if len(ids) >= 2:
            retired.append(p)
            positions[ids[0]] = positions[ids[1]] = -1
            for i in ids[2:]:
                positions[i] = actors[i]
    positions = tuple(positions)
    return (positions, blocks, -1, plate_open(m, positions)), tuple(retired)


def initial_candidates(m, reading):
    visible = sorted(reading["actors"])
    blocks = reading["blocks"]
    mode = -1
    if reading["selected"] in blocks:
        mode = blocks.index(reading["selected"])
    opened = observed_open(m, reading)
    configurations = []

    if mode >= 0 and any(c == 1 for c in reading["actors"].values()):
        active = sorted(p for p, c in reading["actors"].items() if c == 1)
        retired = tuple(p for p, c in reading["actors"].items() if c == 10)
        if len(active) == 4:
            configurations.extend((tuple(p), retired)
                                  for p in itertools.permutations(active))
        elif len(active) == 2:
            labels = [(0, 1)] if not retired else list(
                itertools.combinations(range(4), 2)
            )
            for ids in labels:
                for perm in itertools.permutations(active):
                    aa = [-1] * 4
                    for i, p in zip(ids, perm):
                        aa[i] = p
                    configurations.append((tuple(aa), retired))
    elif len(visible) == 4:
        configurations.extend((tuple(p), ())
                              for p in itertools.permutations(visible))
    elif len(visible) == 3:
        for retired in visible:
            active = [p for p in visible if p != retired]
            for ids in itertools.combinations(range(4), 2):
                for perm in itertools.permutations(active):
                    aa = [-1] * 4
                    for i, p in zip(ids, perm):
                        aa[i] = p
                    configurations.append((tuple(aa), (retired,)))
    elif len(visible) == 2:
        configurations.extend((tuple(p) + (-1, -1), ())
                              for p in itertools.permutations(visible))

    return [
        (i, (actors, blocks, mode, opened), retired)
        for i, (actors, retired) in enumerate(configurations)
    ]


def matches(candidate, reading):
    _, state, retired = candidate
    actors, blocks, mode, _ = state
    if {p for p in actors if p >= 0} | set(retired) != set(reading["actors"]):
        return False
    if sorted(blocks) != list(reading["blocks"]):
        return False
    if (blocks[mode] if mode >= 0 else None) != reading["selected"]:
        return False
    if mode >= 0:
        ones = {p for p, c in reading["actors"].items() if c == 1}
        if ones and ones != {p for p in actors if p >= 0}:
            return False
    return True


def diagnostic(m, candidates, directions):
    if len(candidates) <= 1:
        return None
    best, best_score = None, 0
    for action in directions:
        outcomes = Counter()
        for _, state, retired in candidates:
            result = transition(m, state, action)
            if result is None:
                break
            nxt, added = result
            visible = tuple(sorted(
                {p for p in nxt[0] if p >= 0} | set(retired) | set(added)
            ))
            outcomes[visible] += 1
        else:
            total = len(candidates)
            score = total * total - sum(v * v for v in outcomes.values())
            if score > best_score:
                best_score, best = score, action
    return best


def fixed_pair_path(m, state, directions, deadline):
    """Exact shortest player-only solution with the blocks held fixed."""
    actors, blocks, mode, opened = state
    active = [i for i, p in enumerate(actors) if p >= 0]
    if len(active) != 2:
        return None
    i, j = active
    prefix = [9] if mode >= 0 else []
    key = (actors, blocks, opened, tuple(directions))
    cache = m["fixed_cache"]
    if key in cache:
        path = cache[key]
        return None if path is None else prefix + list(path)

    p, q = actors[i], actors[j]
    root = (p, q, opened)
    parents = {root: None}
    queue = deque([root])
    nav, gate = m["nav"], m["gate_at"]
    block_set = set(blocks)
    blocked_nav = []
    for a in range(4):
        blocked_nav.append([
            u if nav[a][u] in block_set else nav[a][u]
            for u in range(m["w"] * m["h"])
        ])
    action_maps = [
        (a, transformed_action(a - 1, i), transformed_action(a - 1, j))
        for a in directions
    ]
    w, hazards, plates = m["w"], m["hazards"], m["plate_at"]
    expanded = 0
    finish = None
    finish_action = None

    while queue:
        current = queue.popleft()
        p, q, op = current
        adjacent = p // w == q // w and abs(p - q) == 1
        for action, ai, aj in action_maps:
            pp, qq = blocked_nav[ai][p], blocked_nav[aj][q]
            if gate[pp] & ~op:
                pp = p
            if gate[qq] & ~op:
                qq = q
            if pp in hazards or qq in hazards:
                continue
            if pp == qq or (adjacent and (pp == q or qq == p)):
                finish, finish_action = current, action
                break
            nxt = (pp, qq, plates[pp] | plates[qq])
            if nxt not in parents:
                parents[nxt] = current, action
                queue.append(nxt)
        if finish is not None:
            break
        expanded += 1
        if expanded % 512 == 0 and time.perf_counter() >= deadline:
            return None

    if finish is None:
        cache[key] = None
        return None
    path = [finish_action]
    current = finish
    while parents[current] is not None:
        previous, action = parents[current]
        path.append(action)
        current = previous
    path.reverse()
    if len(cache) > 1500:
        cache.clear()
    cache[key] = tuple(path)
    return prefix + path


def make_relaxation(m, relative, deadline):
    if relative in m["relax"]:
        return
    floor, index = m["floor"], m["index"]
    n = len(floor)
    if not n:
        return
    nav = m["nav"]
    reverse_a, reverse_b = [], []
    for a, (dx, dy) in enumerate(DIRS):
        ra = [[i] for i in range(n)]
        rb = [[i] for i in range(n)]
        vx, vy = dx * relative[0], dy * relative[1]
        b = 0 if vy < 0 else 1 if vy > 0 else 2 if vx < 0 else 3
        for i, p in enumerate(floor):
            u = index[nav[a][p]]
            if u != i:
                ra[u].append(i)
            u = index[nav[b][p]]
            if u != i:
                rb[u].append(i)
        reverse_a.append(ra)
        reverse_b.append(rb)

    distance = [-1] * (n * n)
    queue = deque()
    for i in range(n):
        k = i * n + i
        distance[k] = 0
        queue.append(k)
    expanded = 0
    while queue:
        k = queue.popleft()
        u, v = divmod(k, n)
        d = distance[k] + 1
        for a in range(4):
            for x in reverse_a[a][u]:
                base = x * n
                for y in reverse_b[a][v]:
                    z = base + y
                    if distance[z] < 0:
                        distance[z] = d
                        queue.append(z)
        expanded += 1
        if expanded % 1024 == 0 and time.perf_counter() >= deadline:
            break
    m["relax"][relative] = distance


def pair_heuristic(m, i, p, j, q):
    index = m["index"]
    if p not in index or q not in index:
        return 0
    relative = (
        SIGNS[i][0] * SIGNS[j][0],
        SIGNS[i][1] * SIGNS[j][1]
    )
    table = m["relax"].get(relative)
    if table is not None:
        d = table[index[p] * len(index) + index[q]]
        if d >= 0:
            return d
    if p not in m["graph_dist"]:
        distances = {p: 0}
        queue = deque([p])
        while queue:
            u = queue.popleft()
            for a in range(4):
                v = m["nav"][a][u]
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)
        m["graph_dist"][p] = distances
    return m["graph_dist"][p].get(q, 2000) // 2


def heuristic(m, actors):
    cache = m["hc"]
    if actors in cache:
        return cache[actors]
    active = [(i, p) for i, p in enumerate(actors) if p >= 0]
    if len(active) == 2:
        (i, p), (j, q) = active
        value = pair_heuristic(m, i, p, j, q)
    elif len(active) == 4:
        ds = {}
        for a in range(4):
            for b in range(a + 1, 4):
                i, p = active[a]
                j, q = active[b]
                ds[a, b] = pair_heuristic(m, i, p, j, q)
        value = min(
            max(ds[0, 1], ds[2, 3]),
            max(ds[0, 2], ds[1, 3]),
            max(ds[0, 3], ds[1, 2])
        )
    else:
        value = 0
    cache[actors] = value
    return value


def search(m, start, directions, deadline, movable):
    def h(state):
        return heuristic(m, state[0]) + int(state[2] >= 0)

    initial_h = h(start)
    heap = [(initial_h, initial_h, 0, 0, start)]
    costs, parents = {start: 0}, {}
    serial, expanded = 0, 0
    best, best_score = start, (initial_h, 0)
    solved = False

    while heap:
        _, _, _, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        if all(p < 0 for p in state[0]):
            best, solved = state, True
            break
        if expanded % 128 == 0 and time.perf_counter() >= deadline:
            break
        expanded += 1
        if expanded > 240000:
            break
        actions = list(directions)
        if state[2] >= 0:
            actions.append(9)
        if movable:
            actions.extend(10 + b for b in range(len(state[1]))
                           if b != state[2])

        for action in actions:
            result = transition(m, state, action)
            if result is None:
                continue
            nxt, _ = result
            if nxt == state:
                continue
            ng = g + 1
            if ng >= costs.get(nxt, 10**9):
                continue
            hh = h(nxt)
            costs[nxt] = ng
            parents[nxt] = state, action
            serial += 1
            heapq.heappush(heap, (ng + hh, hh, serial, ng, nxt))
            score = (hh, ng)
            if score < best_score:
                best, best_score = nxt, score

    path = []
    while best != start and best in parents:
        previous, action = parents[best]
        path.append(action)
        best = previous
    path.reverse()
    return path, solved


def parking_scores(m):
    """Measure how much removing each floor square disconnects the board."""
    if "parking_scores" in m:
        return m["parking_scores"]
    floor = set(m["floor"]) - m["hazards"]
    neighbors = {
        p: {m["nav"][a][p] for a in range(4)} - {p}
        for p in floor
    }
    neighbors = {p: ns & floor for p, ns in neighbors.items()}
    scores = {}
    degrees = {}
    for removed in floor:
        degrees[removed] = len(neighbors[removed])
        remaining = floor - {removed}
        sizes = []
        while remaining:
            root = remaining.pop()
            stack, size = [root], 1
            while stack:
                p = stack.pop()
                for q in neighbors[p]:
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
                        size += 1
            sizes.append(size)
        scores[removed] = sum(
            sizes[i] * sizes[j]
            for i in range(len(sizes))
            for j in range(i + 1, len(sizes))
        )
    m["parking_scores"] = (scores, degrees)
    return scores, degrees


def block_routes(m, state, b, directions):
    actors, blocks, mode, opened = state
    start = blocks[b]
    forbidden = (
        set(blocks) - {start}
    ) | {p for p in actors if p >= 0} | m["all_plates"] | m["hazards"]
    parents = {start: None}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for action in directions:
            q = m["nav"][action - 1][p]
            if (q == p or q in forbidden
                    or m["gate_at"][q] & ~opened or q in parents):
                continue
            parents[q] = p, action
            distances[q] = distances[p] + 1
            queue.append(q)
    return parents, distances


def parking_plan(m, start, directions, deadline):
    """
    Search macro-actions that move individual obstacles to harmless
    leaves or bypassable squares, checking for a player-only finish.
    """
    actors, blocks, mode, opened = start
    if not blocks:
        return [], False
    scores, degrees = parking_scores(m)
    original_impact = sum(scores.get(p, 0) for p in blocks)

    heap = [(original_impact * .08, 0, 0, start, 0, ())]
    seen = {(blocks, mode, 0): 0}
    serial, expanded = 0, 0
    best_path = ()
    best_progress = (original_impact, 0)

    while heap and time.perf_counter() < deadline:
        _, cost, _, state, used, path = heapq.heappop(heap)
        key = (state[1], state[2], used)
        if cost != seen.get(key):
            continue
        if path:
            finish_deadline = min(deadline, time.perf_counter() + .30)
            finish = fixed_pair_path(m, state, directions, finish_deadline)
            if finish is not None:
                return list(path) + finish, True

        expanded += 1
        if expanded > 180:
            break
        for b, p in enumerate(state[1]):
            if used & (1 << b):
                continue
            parents, distances = block_routes(m, state, b, directions)
            choices = [q for q in parents if q != p and scores.get(q, 0) == 0]
            leaves = sorted(
                (q for q in choices if degrees.get(q, 0) <= 1),
                key=lambda q: distances[q]
            )
            others = sorted(
                (q for q in choices if degrees.get(q, 0) > 1),
                key=lambda q: (distances[q], -degrees.get(q, 0))
            )
            targets = leaves[:10] + others[:7]
            if not targets:
                targets = sorted(
                    (q for q in parents if q != p),
                    key=lambda q: (scores.get(q, 0), distances[q])
                )[:5]

            for q in targets:
                route = []
                current = q
                while current != p:
                    previous, action = parents[current]
                    route.append(action)
                    current = previous
                route.reverse()
                macro = ([] if state[2] == b else [10 + b]) + route + [9]
                bs = list(state[1])
                bs[b] = q
                nxt = (actors, tuple(bs), -1, plate_open(m, actors))
                nu = used | (1 << b)
                nc = cost + len(macro)
                nk = (nxt[1], -1, nu)
                if nc >= seen.get(nk, 10**9):
                    continue
                seen[nk] = nc
                npth = path + tuple(macro)
                impact = sum(scores.get(x, 0) for x in bs)
                progress = (impact, nc)
                if progress < best_progress:
                    best_progress, best_path = progress, npth
                serial += 1
                heapq.heappush(
                    heap, (nc + .08 * impact, nc, serial, nxt, nu, npth)
                )
            if time.perf_counter() >= deadline:
                break
    return list(best_path), False


def emit(memory, m, candidates, action, geo):
    predictions = []
    for ident, state, retired in candidates:
        result = transition(m, state, action)
        if result is not None:
            nxt, added = result
            predictions.append((
                ident, nxt, tuple(sorted(set(retired) | set(added)))
            ))
    memory["candidates"] = predictions
    memory["pending"] = True

    scale, ox, oy, w, h = geo
    if action >= 10:
        p = candidates[0][1][1][action - 10]
    elif action == 9:
        state = candidates[0][1]
        occupied = set(state[1])
        p = next((p for p in state[0] if p >= 0 and p not in occupied), None)
        if p is None:
            p = next((p for p in range(w * h) if p not in occupied), 0)
    else:
        return "ACTION" + str(action), None, None

    row = min(63, oy + (p // w) * scale + scale // 2)
    col = min(63, ox + (p % w) * scale + scale // 2)
    return "ACTION6", int(row), int(col)


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 8.5
    available = set(available)
    directions = [a for a in range(1, 5) if "ACTION" + str(a) in available]
    can_click = "ACTION6" in available

    if "geo" not in memory:
        memory["geo"] = geometry(obs)
    geo = memory["geo"]
    reading = read_board(obs, geo)
    m, changed = update_model(memory, reading, geo)

    candidates = memory.get("candidates", [])
    if memory.get("pending"):
        candidates = [c for c in candidates if matches(c, reading)]
        memory["pending"] = False
    if not candidates:
        candidates = initial_candidates(m, reading)
        memory["plan"] = []
        memory["primary"] = None

    if changed:
        memory["plan"] = []
        opened = observed_open(m, reading)
        candidates = [
            (ident, (st[0], st[1], st[2], opened), retired)
            for ident, st, retired in candidates
        ]

    if not candidates:
        for name in ("ACTION5", "ACTION1", "RESET"):
            if name in available:
                return name, None, None
        return next(iter(available), "RESET"), None, None

    primary = memory.get("primary")
    if primary is not None:
        matching = [c for c in candidates if c[0] == primary]
        if matching:
            candidates = matching + [c for c in candidates if c[0] != primary]
        else:
            memory["plan"] = []

    fingerprint = (
        tuple(sorted(reading["actors"].items())),
        reading["blocks"], reading["selected"]
    )
    counts = memory.setdefault("observed_counts", {})
    counts[fingerprint] = counts.get(fingerprint, 0) + 1
    if counts[fingerprint] >= 3:
        memory["plan"] = []

    memory["candidates"] = candidates
    if not memory.get("plan"):
        probe = diagnostic(m, candidates, directions)
        if probe is not None:
            return emit(memory, m, candidates, probe, geo)

    state = candidates[0][1]
    active = [i for i, p in enumerate(state[0]) if p >= 0]

    # Always check for a short exact finish, even during a longer plan.
    if len(active) == 2:
        exact = fixed_pair_path(
            m, state, directions, min(deadline, started + .65)
        )
        if exact and (can_click or exact[0] < 9):
            memory["primary"] = candidates[0][0]
            memory["plan"] = exact[1:]
            return emit(memory, m, candidates, exact[0], geo)

    if memory.get("plan"):
        action = memory["plan"].pop(0)
        if action < 9 or can_click:
            return emit(memory, m, candidates, action, geo)
        memory["plan"] = []

    # Plan obstacle relocations as whole routes rather than chasing
    # a character-distance heuristic one action at a time.
    if len(active) == 2 and state[1] and can_click:
        parking_deadline = min(deadline - 1.5, time.perf_counter() + 4.5)
        path, solved = parking_plan(m, state, directions, parking_deadline)
        if path:
            memory["primary"] = candidates[0][0]
            memory["plan"] = path[1:]
            return emit(memory, m, candidates, path[0], geo)

    needed = {
        (SIGNS[i][0] * SIGNS[j][0], SIGNS[i][1] * SIGNS[j][1])
        for i, j in itertools.combinations(active, 2)
    }
    relax_deadline = min(deadline - .8, time.perf_counter() + 2.0)
    for relative in needed:
        if time.perf_counter() >= relax_deadline:
            break
        make_relaxation(m, relative, relax_deadline)

    fixed_deadline = (
        min(deadline, time.perf_counter() + .65)
        if state[1] and can_click else deadline
    )
    path, solved = search(m, state, directions, fixed_deadline, False)
    if not solved and state[1] and can_click and time.perf_counter() < deadline:
        joint_path, joint_solved = search(
            m, state, directions, deadline, True
        )
        if joint_solved or joint_path:
            path = joint_path

    if path:
        memory["primary"] = candidates[0][0]
        memory["plan"] = path[1:]
        return emit(memory, m, candidates, path[0], geo)

    visits = memory.setdefault("visits", {})
    choices = list(directions)
    if can_click:
        if state[2] >= 0:
            choices.append(9)
        choices.extend(10 + b for b in range(len(state[1]))
                       if b != state[2])
    scored = []
    for action in choices:
        result = transition(m, state, action)
        if result is None or result[0] == state:
            continue
        nxt = result[0]
        key = (state, action)
        score = heuristic(m, nxt[0]) + 4 * visits.get(key, 0)
        scored.append((score, action, key))
    if scored:
        _, action, key = min(scored)
        visits[key] = visits.get(key, 0) + 1
        return emit(memory, m, candidates, action, geo)
    if directions:
        return emit(memory, m, candidates, directions[0], geo)
    return ("ACTION5" if "ACTION5" in available else "RESET"), None, None

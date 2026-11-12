import numpy as np
import math
import time
import heapq
from collections import deque, Counter


def _components(mask):
    h, w = mask.shape
    bits = mask.ravel().tolist()
    result = []
    for start in range(h * w):
        if not bits[start]:
            continue
        bits[start] = False
        stack = [start]
        r0 = r1 = start // w
        c0 = c1 = start % w
        count = 0
        while stack:
            z = stack.pop()
            r, c = divmod(z, w)
            r0, r1 = min(r0, r), max(r1, r)
            c0, c1 = min(c0, c), max(c1, c)
            count += 1
            if r and bits[z - w]:
                bits[z - w] = False
                stack.append(z - w)
            if r + 1 < h and bits[z + w]:
                bits[z + w] = False
                stack.append(z + w)
            if c and bits[z - 1]:
                bits[z - 1] = False
                stack.append(z - 1)
            if c + 1 < w and bits[z + 1]:
                bits[z + 1] = False
                stack.append(z + 1)
        result.append((r0, r1 + 1, c0, c1 + 1, count))
    return result


def _runs(row):
    out = []
    start = None
    for i, value in enumerate(row):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(row)))
    return out


def _direction(obs):
    a = np.asarray(obs).copy()
    a[0, :] = 3
    votes = Counter()
    for color in (11, 14):
        for r0, r1, c0, c1, _ in _components(a == color):
            if r0:
                votes["down"] += int(np.sum(a[r0 - 1, c0:c1] == 4))
            if r1 < 64:
                votes["up"] += int(np.sum(a[r1, c0:c1] == 4))
            if c0:
                votes["right"] += int(np.sum(a[r0:r1, c0 - 1] == 4))
            if c1 < 64:
                votes["left"] += int(np.sum(a[r0:r1, c1] == 4))
    return votes.most_common(1)[0][0] if votes and max(votes.values()) else None


def _normalize(obs, direction):
    a = np.asarray(obs).copy()
    # The only display overlay is the counter along the first row.
    a[0, :] = a[1, :]
    if direction == "down":
        return a
    if direction == "up":
        return a[::-1, :]
    if direction == "right":
        return a.T
    return a[:, ::-1].T


def _screen(p, g, direction):
    p, g = int(p), int(g)
    if direction == "down":
        return g, p
    if direction == "up":
        return 63 - g, p
    if direction == "right":
        return p, g
    return p, 63 - g


def _colored_objects(a):
    players, targets = [], []
    for color in (11, 14):
        for g0, g1, p0, p1, _ in _components(a == color):
            item = (color, p0, p1, g0, g1)
            if g0 > 0 and np.any(a[g0 - 1, p0:p1] == 4):
                players.append(item)
            else:
                targets.append(item)
    return players, targets


def _build_model(obs):
    direction = _direction(obs)
    if direction is None:
        return None
    a = _normalize(obs, direction)
    buttons = _components(a == 9)
    if not buttons:
        return None
    d = Counter(min(g1 - g0, p1 - p0)
                for g0, g1, p0, p1, _ in buttons).most_common(1)[0][0]
    if d <= 0:
        return None

    players, targets = _colored_objects(a)
    if not players:
        return None

    fixed = (a == 5) | (a == 1) | (a == 12) | (a == 11) | (a == 14)
    for _, p0, p1, g0, g1 in players:
        fixed[g0:g1, p0:p1] = False

    strips = {}
    for g in range(64):
        for p0, p1 in _runs(fixed[g]):
            if p1 - p0 == d:
                strips.setdefault((p0, p1), []).append(g)

    candidates = [(len(gs), p0, p1, min(gs), max(gs) + 1)
                  for (p0, p1), gs in strips.items()
                  if len(gs) >= max(2, d)]
    walls = []
    for _, p0, p1, g0, g1 in sorted(candidates, reverse=True):
        if not any(p0 < q1 and q0 < p1 for q0, q1, _, _ in walls):
            walls.append((p0, p1, g0, g1))
    walls.sort()
    if not walls:
        return None

    slots = []
    last = 0
    for p0, p1, _, _ in walls:
        if p0 - last > d:
            slots.append((last, p0))
        last = max(last, p1)
    if 64 - last > d:
        slots.append((last, 64))

    floor_mask = a == 5
    for p0, p1, _, _ in walls:
        floor_mask[:, p0:p1] = False
    floors = []
    for g0, g1, p0, p1, count in _components(floor_mask):
        if p1 - p0 > d and count >= (g1 - g0) * (p1 - p0) * 0.8:
            floors.append((p0, p1, g0, g1))

    # Each entry is (perpendicular start, end, fixed back edge, surface).
    platforms = []
    water = (a == 0) | (a == 9)
    for p0, p1 in slots:
        full = np.all(water[:, p0:p1], axis=1)
        full &= np.any(a[:, p0:p1] == 0, axis=1)
        for g0, g1 in _runs(full):
            platforms.append((p0, p1, g1, g0))

    # Empty columns leave no black pixels. Their controls and backing
    # boundaries still identify them.
    for p0, p1 in slots:
        possible_ends = {64}
        for f0, f1, fg0, fg1 in floors:
            if f0 == p0 and f1 >= p1:
                possible_ends.add(fg0)
        for end in possible_ends:
            marked = any(bp0 >= p0 and bp1 <= p1 and bg1 == end
                         for bg0, bg1, bp0, bp1, _ in buttons)
            marked |= any(pp0 >= p0 and pp1 <= p1 and pg1 == end
                          for _, pp0, pp1, pg0, pg1 in players)
            if marked and not any(x0 == p0 and x1 == p1 and e == end
                                  for x0, x1, e, s in platforms):
                platforms.append((p0, p1, end, end))

    platforms = sorted(set(platforms), key=lambda x: (x[0], x[2]))
    if not platforms:
        return None
    n = len(platforms)

    colors_and_places = []
    for color, p0, p1, g0, g1 in players:
        matches = [i for i, (x0, x1, e, s) in enumerate(platforms)
                   if x0 <= p0 < x1 and s == g1]
        if not matches:
            return None
        colors_and_places.append((color, matches[0]))
    colors_and_places.sort()
    colors = tuple(c for c, i in colors_and_places)
    places = tuple(i for c, i in colors_and_places)

    goals = {c: [set() for _ in range(n)] for c in set(colors)}
    for color, p0, p1, g0, g1 in targets:
        if color not in goals:
            continue
        target_walls = [w for w in walls
                        if w[0] <= p0 and p1 <= w[1]]
        for i, (x0, x1, e, s) in enumerate(platforms):
            if any(w[1] == x0 or w[0] == x1 for w in target_walls):
                goals[color][i].add(g1)

    positive = direction in ("down", "right")
    scale = 2 if d == 4 else 1
    clearance = 6 if direction == "left" and d == 3 else 4 * scale
    bounds0, bounds1 = [], []
    for p0, p1, e, s in platforms:
        relevant_floors = [f for f in floors if f[0] == p0 and f[3] < e]
        if relevant_floors:
            if positive:
                bound = max(f[3] for f in relevant_floors)
                occupied_bound = bound - clearance
            else:
                bound = min(f[3] for f in relevant_floors)
                occupied_bound = bound + clearance
        else:
            adjacent = [w[2] for w in walls if w[1] == p0 or w[0] == p1]
            bound = max(adjacent) if adjacent else 0
            occupied_bound = bound
        bounds0.append(bound)
        bounds1.append(occupied_bound)

    # Actions: (kind, shrinking/left column, growing/right column,
    #           gate surface, screen row, screen column).
    actions = []
    for bg0, bg1, bp0, bp1, _ in buttons:
        bw = bp1 - bp0
        at_start = [i for i, (p0, p1, e, s) in enumerate(platforms)
                    if p0 == bp0 and e >= bg0]
        if at_start:
            other = [i for i, (p0, p1, e, s) in enumerate(platforms)
                     if p1 == bp0 - bw and e >= bg0]
            if not other:
                continue
            j = min(at_start, key=lambda i: platforms[i][2])
            i = min(other, key=lambda i: platforms[i][2])
        else:
            inside = [i for i, (p0, p1, e, s) in enumerate(platforms)
                      if p0 < bp0 < p1 and e >= bg0]
            other = [i for i, (p0, p1, e, s) in enumerate(platforms)
                     if p0 == bp0 + 2 * bw and e >= bg0]
            if not inside or not other:
                continue
            j = min(inside, key=lambda i: platforms[i][2])
            i = min(other, key=lambda i: platforms[i][2])
        row, col = _screen((bp0 + bp1 - 1) // 2,
                           (bg0 + bg1 - 1) // 2, direction)
        actions.append((0, i, j, 0, row, col))

    gates = _components((a == 1) | (a == 12))
    for g0, g1, p0, p1, _ in gates:
        if p1 - p0 != d:
            continue
        left = [i for i, p in enumerate(platforms) if p[1] == p0]
        right = [i for i, p in enumerate(platforms) if p[0] == p1]
        row, col = _screen((p0 + p1 - 1) // 2,
                           (g0 + g1 - 1) // 2, direction)
        for i in left:
            for j in right:
                if platforms[i][2] >= g1 and platforms[j][2] >= g1:
                    actions.append((1, i, j, g1, row, col))

    if not actions:
        return None
    groups = []
    for color in sorted(set(colors)):
        groups.append(tuple(k for k, c in enumerate(colors) if c == color))
    state = tuple(p[3] for p in platforms) + places
    model = {
        "direction": direction, "d": d, "n": n,
        "platforms": platforms, "colors": colors, "groups": groups,
        "goals": goals, "b0": tuple(bounds0), "b1": tuple(bounds1),
        "actions": actions,
    }
    _prepare_heuristic(model, state)
    return model, state


def _observe(obs, model):
    a = _normalize(obs, model["direction"])
    surfaces = []
    water = (a == 0) | (a == 9)
    for p0, p1, end, initial in model["platforms"]:
        full = np.all(water[:, p0:p1], axis=1)
        full &= np.any(a[:, p0:p1] == 0, axis=1)
        s = end
        while s > 0 and full[s - 1]:
            s -= 1
        surfaces.append(s)

    players, _ = _colored_objects(a)
    cp = []
    for color, p0, p1, g0, g1 in players:
        matches = [i for i, p in enumerate(model["platforms"])
                   if p[0] <= p0 < p[1] and surfaces[i] == g1]
        if not matches:
            return None
        cp.append((color, matches[0]))
    cp.sort()
    if tuple(c for c, i in cp) != model["colors"]:
        return None
    return tuple(surfaces) + tuple(i for c, i in cp)


def _canonical_places(places, model):
    places = list(places)
    for group in model["groups"]:
        values = sorted(places[k] for k in group)
        for k, v in zip(group, values):
            places[k] = v
    return tuple(places)


def _transition(state, action, model):
    kind, i, j, height, row, col = action
    n, d = model["n"], model["d"]
    places = state[n:]
    if kind == 0:
        end_i = model["platforms"][i][2]
        bound_j = model["b1"][j] if j in places else model["b0"][j]
        if state[i] >= end_i or state[j] <= bound_j:
            return None
        new = list(state)
        new[i] += d
        new[j] -= d
        return tuple(new)
    if state[i] != height or state[j] != height:
        return None
    if i not in places and j not in places:
        return None
    moved = [j if p == i else i if p == j else p for p in places]
    new = state[:n] + _canonical_places(moved, model)
    return new if new != state else None


def _is_goal(state, model):
    n = model["n"]
    return all(state[state[n + k]] in model["goals"][color][state[n + k]]
               for k, color in enumerate(model["colors"]))


def _prepare_heuristic(model, state):
    n, d = model["n"], model["d"]
    actions = model["actions"]
    grow, shrink = [False] * n, [False] * n
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    direct_pairs = set()
    incident = [[] for _ in range(n)]
    for kind, i, j, height, _, _ in actions:
        if kind == 0:
            shrink[i] = True
            grow[j] = True
            parent[root(i)] = root(j)
            direct_pairs.add((min(i, j), max(i, j)))
        else:
            incident[i].append((j, height))
            incident[j].append((i, height))

    volume = Counter()
    for i, p in enumerate(model["platforms"]):
        volume[root(i)] += p[2] - state[i]

    nodes = []
    allowed = []
    for i, p in enumerate(model["platforms"]):
        raw = min(model["b0"][i], model["b1"][i], state[i])
        low = state[i] - int(math.ceil((state[i] - raw) / d)) * d
        volume_low = p[2] - volume[root(i)]
        if low < volume_low:
            low += int(math.ceil((volume_low - low) / d)) * d
        hs = set(range(low, p[2] + 1, d))
        hs.add(state[i])
        allowed.append(hs)
        nodes.extend((i, h) for h in hs)

    reverse = {node: [] for node in nodes}

    def edge(u, v):
        if u in reverse and v in reverse:
            reverse[v].append(u)

    for i, h in nodes:
        if shrink[i]:
            edge((i, h), (i, h + d))
        if grow[i] and h > model["b1"][i]:
            edge((i, h), (i, h - d))
        for j, height in incident[i]:
            if h == height:
                edge((i, h), (j, h))

    distances = {}
    for color in set(model["colors"]):
        dist = {}
        queue = deque()
        for i in range(n):
            for h in model["goals"][color][i]:
                if (i, h) in reverse:
                    dist[(i, h)] = 0
                    queue.append((i, h))
        while queue:
            v = queue.popleft()
            nd = dist[v] + 1
            for u in reverse[v]:
                if u not in dist:
                    dist[u] = nd
                    queue.append(u)
        distances[color] = dist

    model["grow"], model["shrink"] = grow, shrink
    model["allowed"] = allowed
    model["incident"] = incident
    model["direct_pairs"] = direct_pairs
    model["distances"] = distances


def _heuristic(state, model):
    n, d = model["n"], model["d"]
    vals = []
    infinity = 100000

    def can_adjust(i, start, end):
        if end not in model["allowed"][i]:
            return False
        if end < start:
            return model["grow"][i] and end + d > model["b1"][i]
        if end > start:
            return model["shrink"][i]
        return True

    for k, color in enumerate(model["colors"]):
        i = state[n + k]
        s = state[i]
        distances = model["distances"][color]
        base = distances.get((i, s), infinity)
        if base == infinity:
            return infinity
        best = infinity
        for target in model["goals"][color][i]:
            if can_adjust(i, s, target):
                best = min(best, abs(s - target) // d)
        for j, height in model["incident"][i]:
            remaining = distances.get((j, height), infinity)
            if remaining == infinity or not can_adjust(i, s, height):
                continue
            if height not in model["allowed"][j]:
                continue
            dc = (height - s) // d
            dj = (height - state[j]) // d
            if (min(i, j), max(i, j)) in model["direct_pairs"]:
                alignment = max(max(dc, 0) + max(dj, 0),
                                max(-dc, 0) + max(-dj, 0))
            else:
                alignment = abs(dc) + abs(dj)
            best = min(best, alignment + 1 + remaining)
        vals.append(max(base, best if best != infinity else base))
    if not vals:
        return 0
    return max(max(vals), sum(vals) / 2.0)


def _search(start, model, banned, visits):
    if _is_goal(start, model):
        return []
    began = time.perf_counter()
    limit = 7.2
    h0 = _heuristic(start, model)
    heap = [(1.45 * h0, h0, 0, 0, start)]
    costs = {start: 0}
    parents = {}
    serial = 0
    best_state = start
    best_quality = (h0, 0)
    finish = None
    expansions = 0
    actions = model["actions"]

    while heap:
        _, h, _, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        if _is_goal(state, model):
            finish = state
            break
        expansions += 1
        if expansions % 128 == 0:
            if time.perf_counter() - began > limit:
                break
        if g >= 49:
            continue
        for ai, action in enumerate(actions):
            if (state, ai) in banned:
                continue
            nxt = _transition(state, action, model)
            if nxt is None:
                continue
            ng = g + 1
            if costs.get(nxt, 10 ** 9) <= ng:
                continue
            nh = _heuristic(nxt, model)
            if nh >= 100000:
                continue
            costs[nxt] = ng
            parents[nxt] = (state, ai)
            quality = (nh, ng)
            if quality < best_quality:
                best_quality = quality
                best_state = nxt
            serial += 1
            penalty = 0.15 * visits.get(nxt, 0)
            heapq.heappush(heap, (ng + 1.45 * nh + penalty,
                                  nh, serial, ng, nxt))

    if finish is None:
        finish = best_state
    if finish == start:
        choices = []
        for ai, action in enumerate(actions):
            if (start, ai) in banned:
                continue
            nxt = _transition(start, action, model)
            if nxt is not None:
                choices.append((_heuristic(nxt, model) +
                                2 * visits.get(nxt, 0), ai, nxt))
        if not choices:
            return []
        _, ai, nxt = min(choices)
        return [(ai, nxt)]

    path = []
    while finish != start:
        previous, ai = parents[finish]
        path.append((ai, finish))
        finish = previous
    path.reverse()
    return path


def _fallback(obs, available, memory):
    count = memory.get("fallback_count", 0)
    memory["fallback_count"] = count + 1
    boxes = _components(np.asarray(obs) == 9)
    if boxes:
        r0, r1, c0, c1, _ = boxes[count % len(boxes)]
        return "ACTION6", (r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2
    if "RESET" in available:
        return "RESET", None, None
    return "ACTION6", 63, 63


def act(obs, available, memory):
    model = memory.get("model")
    state = _observe(obs, model) if model is not None else None

    if model is None or state is None:
        built = _build_model(obs)
        if built is None:
            return _fallback(obs, available, memory)
        model, state = built
        memory["model"] = model
        memory["plan"] = []
        memory["banned"] = set()
        memory["visits"] = {}
        memory.pop("last", None)

    banned = memory.setdefault("banned", set())
    visits = memory.setdefault("visits", {})
    visits[state] = visits.get(state, 0) + 1

    last = memory.get("last")
    plan = memory.get("plan", [])
    if last is not None:
        previous, ai, predicted = last
        if state != predicted:
            plan = []
            if state == previous:
                banned.add((previous, ai))

    if _is_goal(state, model):
        # A harmless click makes the simulator perform its goal check.
        free = np.argwhere(np.asarray(obs) == 3)
        if len(free):
            row, col = map(int, free[len(free) // 2])
        else:
            row, col = 63, 63
        memory["plan"] = []
        memory.pop("last", None)
        return "ACTION6", row, col

    if not plan:
        plan = _search(state, model, banned, visits)
    if not plan:
        memory["plan"] = []
        return _fallback(obs, available, memory)

    ai, predicted = plan.pop(0)
    action = model["actions"][ai]
    memory["plan"] = plan
    memory["last"] = (state, ai, predicted)
    return "ACTION6", int(action[4]), int(action[5])

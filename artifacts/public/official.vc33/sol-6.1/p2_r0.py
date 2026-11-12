import numpy as np
import heapq
import math
import time
from collections import Counter


def _runs(mask):
    a = np.asarray(mask, dtype=bool)
    d = np.diff(np.r_[False, a, False].astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(),
                    np.flatnonzero(d == -1).tolist()))


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    H, W = mask.shape
    out = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        r0 = r1 = r
        c0 = c1 = c
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            r0, r1 = min(r0, y), max(r1, y)
            c0, c1 = min(c0, x), max(c1, x)
            for yy, xx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if (0 <= yy < H and 0 <= xx < W
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        out.append((r0, r1 + 1, c0, c1 + 1, area))
    return out


def _unrotate(r, c, k):
    r, c = int(r), int(c)
    if k == 0:
        return r, c
    if k == 1:
        return c, 63 - r
    if k == 2:
        return 63 - r, 63 - c
    return 63 - c, r


def _limit(model, node, occupied):
    lane, height, bottom = model["nodes"][node]
    floors = [b for a, b in model["floors"][lane] if b < bottom]
    if floors:
        if model["positive"]:
            limit = max(floors)
            if occupied:
                limit -= 4 * model["scale"]
        else:
            limit = min(floors)
            if occupied:
                if model["rotation"] == 1 and model["unit"] == 3:
                    limit += 6
                else:
                    limit += 4 * model["scale"]
        return limit
    return max((model["walls"][w][2]
                for w in model["lane_walls"][lane]), default=0)


def _build_model(obs):
    raw = np.array(obs, dtype=np.int16, copy=True)
    raw[0] = raw[1]

    yellow = [
        b for b in _components(raw == 4)
        if 2 <= b[1] - b[0] <= 12
        and 2 <= b[3] - b[2] <= 12
        and b[4] >= 4
    ]
    if not yellow:
        return None

    r0, r1, c0, c1, _ = max(yellow, key=lambda b: b[4])
    glyph = raw[r0:r1, c0:c1] == 4
    hh, ww = glyph.shape
    if ww > hh:
        down = int(glyph[-1].sum()) >= int(glyph[0].sum())
        k = 0 if down else 2
    else:
        right = int(glyph[:, -1].sum()) >= int(glyph[:, 0].sum())
        k = 3 if right else 1

    # Rotate gravity to point down.
    image = np.rot90(raw, k)

    buttons = []
    sizes = []
    for a, b, x, z, area in _components(image == 9):
        if b - a >= 2 and z - x >= 2:
            buttons.append((a, b, x, z))
            sizes.append(min(b - a, z - x))
    unit = Counter(sizes).most_common(1)[0][0] if sizes else 3
    scale = 2 if unit == 4 else 1

    actors_raw = []
    for a, b, x, z, area in _components(image == 4):
        if not (2 <= b - a <= 12 and 2 <= z - x <= 12):
            continue
        if b >= 64:
            continue
        color = int(image[b, (x + z) // 2])
        if color in (0, 1, 3, 4, 5, 7, 9, 12):
            continue
        end = b
        while end < min(64, b + 6):
            if np.mean(image[end, x:z] == color) < 0.7:
                break
            end += 1
        if end > b:
            actors_raw.append((x, z, a, end, color))
    if not actors_raw:
        return None
    actor_colors = set(a[4] for a in actors_raw)

    wall_pixels = np.isin(image, [5, 1, 12])
    long_columns = np.zeros(64, dtype=bool)
    for c in range(64):
        long_columns[c] = any(b - a >= 8
                              for a, b in _runs(wall_pixels[:, c]))

    walls = []
    for x, z in _runs(long_columns):
        if z - x > 2 * unit:
            continue
        c = (x + z) // 2
        mask = wall_pixels[:, c].copy()
        for color in actor_colors:
            mask |= image[:, c] == color
        ys = np.flatnonzero(mask)
        if len(ys):
            walls.append((x, z, int(ys[0]), int(ys[-1]) + 1))
    walls.sort()
    if not walls:
        return None

    lanes = []
    lane_walls = []
    last = 0
    previous = None
    for wi, (x, z, a, b) in enumerate(walls):
        if x > last:
            lanes.append((last, x))
            lane_walls.append(
                ([previous] if previous is not None else []) + [wi]
            )
        last = z
        previous = wi
    if last < 64:
        lanes.append((last, 64))
        lane_walls.append([previous] if previous is not None else [])

    def lane_at(c):
        for i, (x, z) in enumerate(lanes):
            if x <= c < z:
                return i
        return None

    floors = []
    nodes = []
    for li, (x, z) in enumerate(lanes):
        part = image[:, x:z]
        support_rows = np.mean((part == 0) | (part == 9), axis=1) >= 0.65
        for a, b in _runs(support_rows):
            nodes.append((li, a, b))
        floor_rows = np.mean(part == 5, axis=1) >= 0.9
        floors.append(_runs(floor_rows))

    # Buttons need not sit at the bottom of their platform. Only infer an
    # invisible depleted platform when no existing platform reaches that
    # button. This avoids phantom platforms beside long supports.
    for a, b, x, z in buttons:
        li = lane_at((x + z) // 2)
        if li is not None and not any(
                n[0] == li and n[2] >= b for n in nodes):
            nodes.append((li, b, b))

    for x, z, a, end, color in actors_raw:
        li = lane_at((x + z) // 2)
        if li is None:
            continue
        if not any(n[0] == li and n[1] == end for n in nodes):
            possible = [
                b for aa, b, xx, zz in buttons
                if lane_at((xx + zz) // 2) == li and b >= end
            ]
            bottom = min(possible) if possible else 64
            nodes.append((li, end, bottom))

    nodes = sorted(set(nodes), key=lambda n: (n[0], n[2], n[1]))
    heights = tuple(n[1] for n in nodes)
    actors = []
    colors = []
    actor_heights = []
    for x, z, a, end, color in actors_raw:
        li = lane_at((x + z) // 2)
        candidates = [
            i for i, n in enumerate(nodes) if n[0] == li and n[1] == end
        ]
        if candidates:
            actors.append(candidates[0])
            colors.append(color)
            actor_heights.append(end - a)
    if not actors:
        return None

    positive = k in (0, 3)

    def select_node(li, button_top):
        if li is None:
            return None
        candidates = [
            i for i, n in enumerate(nodes)
            if n[0] == li and n[2] >= button_top
        ]
        if not candidates:
            return None
        if positive:
            return min(candidates, key=lambda i: nodes[i][1])
        return min(candidates, key=lambda i: nodes[i][2])

    actions = []
    transfers = []
    used_pairs = set()
    for a, b, x, z in buttons:
        for wi, (wx, wz, wa, wb) in enumerate(walls):
            if z == wx:
                receiver_lane = lane_at((x + z) // 2)
                donor_lane = lane_at(wz)
            elif x == wz:
                receiver_lane = lane_at((x + z) // 2)
                donor_lane = lane_at(wx - 1)
            else:
                continue
            donor = select_node(donor_lane, a)
            receiver = select_node(receiver_lane, a)
            if donor is None or receiver is None or donor == receiver:
                continue
            if (donor, receiver) in used_pairs:
                continue
            used_pairs.add((donor, receiver))
            aid = len(actions)
            actions.append(_unrotate((a + b - 1) // 2,
                                     (x + z - 1) // 2, k))
            transfers.append((donor, receiver, aid))

    bridges_raw = []
    goals_raw = []
    for wi, (x, z, a, b) in enumerate(walls):
        c = (x + z) // 2
        for aa, bb in _runs(np.isin(image[:, c], [1, 12])):
            if bb - aa >= 3:
                aid = len(actions)
                actions.append(_unrotate((aa + bb - 1) // 2, c, k))
                bridges_raw.append((wi, bb, aid))
        for color in actor_colors:
            for aa, bb in _runs(image[:, c] == color):
                goals_raw.append((wi, color, bb - 3 * (bb - aa)))

    model = {
        "rotation": k,
        "positive": positive,
        "unit": unit,
        "scale": scale,
        "walls": walls,
        "lanes": lanes,
        "lane_walls": lane_walls,
        "floors": floors,
        "nodes": nodes,
        "actions": actions,
        "transfers": transfers,
        "colors": colors,
        "actor_heights": actor_heights,
    }

    N = len(nodes)
    lower = []
    upper = []
    for i, n in enumerate(nodes):
        h, bottom = n[1], n[2]
        lim = _limit(model, i, False)
        lo = h - max(0, math.ceil((h - lim) / unit)) * unit
        hi = h + max(0, math.ceil((bottom - h) / unit)) * unit
        lower.append(min(h, lo))
        upper.append(max(h, hi))

    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for d, r, aid in transfers:
        u, v = find(d), find(r)
        if u != v:
            parent[u] = v
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    group_of = {}
    sums = {}
    for g, inds in groups.items():
        sums[g] = sum(heights[i] for i in inds)
        for i in inds:
            group_of[i] = g

    def feasible(fixed):
        for i, h in fixed.items():
            if not lower[i] <= h <= upper[i]:
                return False
            if (h - heights[i]) % unit:
                return False
        for g in set(group_of[i] for i in fixed):
            inds = groups[g]
            fixed_sum = sum(fixed[i] for i in inds if i in fixed)
            free = [i for i in inds if i not in fixed]
            rest = sums[g] - fixed_sum
            if not (sum(lower[i] for i in free) <= rest
                    <= sum(upper[i] for i in free)):
                return False
        return True

    goals = []
    for ai, color in enumerate(colors):
        options = []
        ah = actor_heights[ai]
        for wi, gc, goal_top in goals_raw:
            if gc != color:
                continue
            target = goal_top + ah
            for ni, n in enumerate(nodes):
                if wi in lane_walls[n[0]] and feasible({ni: target}):
                    options.append((ni, target))
        goals.append(sorted(set(options)))

    edges = []
    adjacency = [[] for _ in nodes]
    for wi, target, aid in bridges_raw:
        wx, wz, _, _ = walls[wi]
        left = lane_at(wx - 1)
        right = lane_at(wz)
        for i, n in enumerate(nodes):
            if n[0] != left:
                continue
            for j, m in enumerate(nodes):
                if m[0] == right and feasible({i: target, j: target}):
                    edges.append((i, j, target, aid))
                    adjacency[i].append((j, target))
                    adjacency[j].append((i, target))

    model["goals"] = goals
    model["edges"] = edges
    model["adjacency"] = adjacency
    model["N"] = N
    model["initial"] = heights + tuple(actors)
    return model


def _won(model, state):
    N = model["N"]
    return all(
        any(state[N + a] == node and state[node] == height
            for node, height in options)
        for a, options in enumerate(model["goals"])
    )


def _successors(model, state, bad):
    N = model["N"]
    unit = model["unit"]
    occupied = set(state[N:])
    emitted = set()

    for donor, receiver, aid in model["transfers"]:
        if (state, aid) in bad:
            continue
        if state[donor] >= model["nodes"][donor][2]:
            continue
        lim = _limit(model, receiver, receiver in occupied)
        if state[receiver] <= lim:
            continue
        ns = list(state)
        ns[donor] += unit
        ns[receiver] -= unit
        ns = tuple(ns)
        if ns not in emitted:
            emitted.add(ns)
            yield aid, ns

    for i, j, target, aid in model["edges"]:
        if (state, aid) in bad:
            continue
        if state[i] != target or state[j] != target:
            continue
        if i not in occupied and j not in occupied:
            continue
        ns = list(state)
        for a in range(N, len(ns)):
            if state[a] == i:
                ns[a] = j
            elif state[a] == j:
                ns[a] = i
        ns = tuple(ns)
        if ns != state and ns not in emitted:
            emitted.add(ns)
            yield aid, ns


def _heuristic(model, state):
    N = model["N"]
    unit = model["unit"]
    adjacency = model["adjacency"]
    costs = []
    for ai, options in enumerate(model["goals"]):
        if not options:
            return 1000.0
        start = state[N + ai]
        dist = [float("inf")] * N
        dist[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            d, i = heapq.heappop(heap)
            if d != dist[i]:
                continue
            for j, target in adjacency[i]:
                activation = max(abs(state[i] - target),
                                 abs(state[j] - target)) / unit
                nd = d + 1.0 + activation
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))
        cost = min(
            dist[i] + abs(state[i] - target) / unit
            for i, target in options
        )
        if not math.isfinite(cost):
            cost = 1000.0
        costs.append(cost)
    if not costs:
        return 1000.0
    largest = max(costs)
    return largest + 0.4 * (sum(costs) - largest)


def _search(model, start, bad, budget):
    if _won(model, start):
        return []
    deadline = time.monotonic() + 7.5
    h0 = _heuristic(model, start)
    heap = [(1.8 * h0, h0, 0, 0, start)]
    distance = {start: 0}
    parent = {}
    counter = 0
    expanded = 0
    best = start
    best_score = h0

    def path(end):
        answer = []
        while end != start:
            previous, aid = parent[end]
            answer.append((aid, end))
            end = previous
        answer.reverse()
        return answer

    while heap and expanded < 160000:
        _, h, g, _, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        expanded += 1
        if expanded % 256 == 0 and time.monotonic() >= deadline:
            break
        if _won(model, state):
            return path(state)
        score = h + 0.025 * g
        if score < best_score:
            best_score, best = score, state
        if g >= budget:
            continue
        for aid, ns in _successors(model, state, bad):
            ng = g + 1
            if ng >= distance.get(ns, 10**9):
                continue
            distance[ns] = ng
            parent[ns] = (state, aid)
            if _won(model, ns):
                return path(ns)
            nh = _heuristic(model, ns)
            counter += 1
            heapq.heappush(heap, (ng + 1.8 * nh, nh, ng, counter, ns))

    return path(best) if best != start else []


def _fallback(obs, available, memory):
    boxes = _components(np.asarray(obs) == 9)
    if boxes:
        index = memory.get("_fallback_index", 0)
        memory["_fallback_index"] = index + 1
        a, b, x, z, _ = boxes[index % len(boxes)]
        return "ACTION6", (a + b - 1) // 2, (x + z - 1) // 2
    if "RESET" in available:
        return "RESET", None, None
    return "ACTION6", 32, 32


def act(obs, available, memory):
    obs = np.asarray(obs)
    signature = hash(obs[1:].tobytes())
    model = memory.get("_vc_model")

    if model is not None and "_vc_signature" in memory:
        if signature == memory["_vc_signature"]:
            before = memory.get("_vc_before")
            aid = memory.get("_vc_last_action")
            if before is not None:
                memory["_vc_state"] = before
                memory.setdefault("_vc_bad", set()).add((before, aid))
            memory["_vc_plan"] = []

    if model is None:
        try:
            model = _build_model(obs)
        except (ValueError, IndexError, TypeError, ZeroDivisionError):
            model = None
        if model is None:
            return _fallback(obs, available, memory)
        memory["_vc_model"] = model
        memory["_vc_state"] = model["initial"]
        memory["_vc_plan"] = []
        memory["_vc_bad"] = set()
        memory["_vc_visits"] = {}

    state = memory["_vc_state"]
    visits = memory.setdefault("_vc_visits", {})
    visits[state] = visits.get(state, 0) + 1

    plan = memory.get("_vc_plan", [])
    if not plan:
        green = int(np.count_nonzero(obs[0] == 7))
        remaining = int(round(green * 50 / 64)) if green else 50
        remaining = max(1, min(50, remaining))
        plan = _search(model, state, memory["_vc_bad"], remaining)

    if not plan:
        choices = list(_successors(model, state, memory["_vc_bad"]))
        if not choices:
            return _fallback(obs, available, memory)
        aid, ns = min(
            choices,
            key=lambda p: (_heuristic(model, p[1])
                           + 2.0 * visits.get(p[1], 0))
        )
    else:
        aid, ns = plan.pop(0)

    memory["_vc_plan"] = plan
    memory["_vc_before"] = state
    memory["_vc_state"] = ns
    memory["_vc_last_action"] = aid
    memory["_vc_signature"] = signature
    row, col = model["actions"][aid]
    return "ACTION6", int(row), int(col)

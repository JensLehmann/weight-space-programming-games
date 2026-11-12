import math
import time
import heapq
from collections import deque
import numpy as np

# Action order: up, down, left, right.
_DIR = ((0, -1), (0, 1), (-1, 0), (1, 0))
_OPP = (1, 0, 3, 2)
_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_DELTA_DIR = {v: i for i, v in enumerate(_DIR)}
_INF = 10**6


def _reset(available, memory):
    memory.clear()
    if "RESET" in available:
        return "RESET", None, None
    for name in _NAMES:
        if name in available:
            return name, None, None
    return (available[0] if available else "ACTION1"), None, None


def _parse(obs):
    yy, xx = np.where(obs == 9)
    if not len(xx):
        return None

    px, py = int(xx.min()), int(yy.min())
    side = min(int(xx.max()) - px + 1, int(yy.max()) - py + 1)
    if side < 3 or side % 3:
        return None

    spacing = 2 * side
    ox, oy = px % spacing, py % spacing
    player = ((px - ox) // spacing, (py - oy) // spacing)
    nodes = set()
    entities = []
    goals = set()

    half = side // 2
    edge = side // 6
    samples = ((half, edge), (half, side - 1 - edge),
               (edge, half), (side - 1 - edge, half))

    for y in range(oy, 64 - side + 1, spacing):
        for x in range(ox, 64 - side + 1, spacing):
            color = int(obs[y + half, x + half])
            if color not in (0, 8, 9, 12, 13, 14):
                continue
            pos = ((x - ox) // spacing, (y - oy) // spacing)
            nodes.add(pos)
            if color == 14:
                goals.add(pos)
            elif color in (8, 12, 13):
                direction = 0
                active = 0
                for d, (sx, sy) in enumerate(samples):
                    mark = int(obs[y + sy, x + sx])
                    if mark in (11, 15):
                        direction = d
                        active = int(mark == 11)
                        break
                entities.append((color, pos[0], pos[1], direction, active))

    def has_edge(pos, d):
        dx, dy = _DIR[d]
        x = ox + pos[0] * spacing + half + dx * side
        y = oy + pos[1] * spacing + half + dy * side
        return 0 <= x < 64 and 0 <= y < 63 and int(obs[y, x]) == 2

    # Also infer endpoints directly from visible bridges.
    for pos in list(nodes):
        for d, (dx, dy) in enumerate(_DIR):
            if has_edge(pos, d):
                q = (pos[0] + dx, pos[1] + dy)
                x, y = ox + q[0] * spacing, oy + q[1] * spacing
                if 0 <= x <= 64 - side and 0 <= y <= 64 - side:
                    nodes.add(q)

    nodes.add(player)
    graph = {}
    for pos in nodes:
        neighbors = []
        for d, (dx, dy) in enumerate(_DIR):
            q = (pos[0] + dx, pos[1] + dy)
            neighbors.append(q if q in nodes and has_edge(pos, d) else None)
        graph[pos] = tuple(neighbors)

    signature = (
        side, ox, oy,
        tuple((p, graph[p]) for p in sorted(graph))
    )
    return graph, player, entities, goals, signature, (side, ox, oy)


def _merge_entities(observed, predicted, geometry):
    """Retain previously known sprites when another sprite occludes them."""
    side, ox, oy = geometry
    occupied = {(e[1], e[2]) for e in observed}
    result = []
    for e in predicted:
        pos = (e[1], e[2])
        x, y = ox + 2 * side * pos[0], oy + 2 * side * pos[1]
        outside = x + side <= 0 or y + side <= 0 or x >= 64 or y >= 63
        if pos in occupied or outside:
            result.append(tuple(e))

    for e in observed:
        if e in result:
            continue
        matches = [
            i for i, old in enumerate(result)
            if old[:3] == e[:3]
        ]
        if matches:
            result[matches[-1]] = e
        else:
            result.append(e)
    return result


def _predict(graph, player, entities, action):
    target = graph[player][action]
    result = []
    for color, x, y, d, active in entities:
        if (x, y) == target:
            continue

        dx, dy = _DIR[d]
        if color == 8:
            if (x + dx, y + dy) == target:
                x, y, active = x + dx, y + dy, 1

        elif color == 12:
            x, y = x + dx, y + dy
            neighbors = graph.get((x, y))
            if neighbors is None or neighbors[d] is None:
                d = _OPP[d]

        elif color == 13:
            if active:
                nx, ny = x + dx, y + dy
                # Its queued direction is recoverable from the player and
                # the follower's next position.
                queued = _DELTA_DIR.get(
                    (player[0] - nx, player[1] - ny), d
                )
                x, y, d = nx, ny, queued
            elif (x + 2 * dx, y + 2 * dy) == target:
                active = 1

        result.append((color, x, y, d, active))
    return result


def _counter_candidates(obs, old):
    bar = int(np.count_nonzero(obs[63] == 6))
    if old:
        current = [
            (total, max(0, remaining - 1))
            for total, remaining in old
            if round(64 * max(0, remaining - 1) / total) == bar
        ]
        if current:
            return current

    # The bar is rounded, so preserve every compatible interpretation.
    result = []
    for total in range(1, 257):
        lo = max(0, int((bar - 0.5) * total / 64) - 1)
        hi = min(total, int((bar + 0.5) * total / 64) + 1)
        for remaining in range(lo, hi + 1):
            if round(64 * remaining / total) == bar:
                result.append((total, remaining))
    return result


def _relaxed_heuristic(adj, goals, kill8, danger8, count8):
    """Exact distances with sentries, but without moving enemies."""
    n = len(adj)
    reverse = [[] for _ in range(n)]
    for p, row in enumerate(adj):
        for q in row:
            if q >= 0:
                reverse[q].append(p)

    if count8 > 10:
        dist = [_INF] * n
        queue = deque()
        for g in goals:
            dist[g] = 0
            queue.append(g)
        while queue:
            p = queue.popleft()
            for q in reverse[p]:
                if dist[q] == _INF:
                    dist[q] = dist[p] + 1
                    queue.append(q)
        return dist, False

    masks = 1 << count8
    dist = [_INF] * (masks * n)
    queue = deque()
    for mask in range(masks):
        for goal in goals:
            if not (mask & kill8[goal]):
                code = mask * n + goal
                dist[code] = 0
                queue.append(code)

    subsets = []
    for capture in kill8:
        choices = []
        sub = capture
        while True:
            choices.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & capture
        subsets.append(choices)

    while queue:
        code = queue.popleft()
        mask, p = divmod(code, n)
        if mask & danger8[p]:
            continue
        nd = dist[code] + 1
        for restored in subsets[p]:
            previous_mask = mask | restored
            base = previous_mask * n
            for q in reverse[p]:
                if previous_mask & kill8[q]:
                    continue
                index = base + q
                if dist[index] == _INF:
                    dist[index] = nd
                    queue.append(index)
    return dist, True


def _trajectory(graph, entity):
    _, x, y, d, _ = entity
    history = []
    seen = {}
    while (x, y, d) not in seen:
        seen[(x, y, d)] = len(history)
        history.append((x, y, d))
        dx, dy = _DIR[d]
        x, y = x + dx, y + dy
        row = graph.get((x, y))
        if row is None or row[d] is None:
            d = _OPP[d]
    prefix = seen[(x, y, d)]
    return history, prefix, len(history) - prefix


def _search(graph, player, entities, goals, allowed, depth_limit, deadline):
    positions = sorted(graph)
    index = {p: i for i, p in enumerate(positions)}
    n = len(positions)
    adj = [
        [index[q] if q is not None else -1 for q in graph[p]]
        for p in positions
    ]
    goal_ids = {index[g] for g in goals if g in index}
    if not goal_ids:
        return None

    sentries = [e for e in entities if e[0] == 8]
    patrols = [e for e in entities if e[0] == 12]
    sleepers = [e for e in entities if e[0] == 13 and not e[4]]
    followers = [e for e in entities if e[0] == 13 and e[4]]

    kill8 = [0] * n
    danger8 = [0] * n
    for i, e in enumerate(sentries):
        _, x, y, d, _ = e
        bit = 1 << i
        if (x, y) in index:
            kill8[index[(x, y)]] |= bit
        dx, dy = _DIR[d]
        ahead = (x + dx, y + dy)
        if ahead in index:
            danger8[index[ahead]] |= bit

    kill13 = [0] * n
    activation = [[0] * 4 for _ in range(n)]
    activation_all = [0] * n
    for i, e in enumerate(sleepers):
        _, x, y, d, _ = e
        bit = 1 << i
        if (x, y) in index:
            kill13[index[(x, y)]] |= bit
        dx, dy = _DIR[d]
        trigger = (x + 2 * dx, y + 2 * dy)
        if trigger in index:
            p = index[trigger]
            activation[p][d] |= bit
            activation_all[p] |= bit

    forbidden = 0
    for _, x, y, d, _ in followers:
        dx, dy = _DIR[d]
        delta = (x + dx - player[0], y + dy - player[1])
        if delta in _DELTA_DIR:
            forbidden |= 1 << _DELTA_DIR[delta]
    if followers and not forbidden:
        forbidden = 16  # Chasing flag, with no immediately adjacent hazard.

    htable, mask_heuristic = _relaxed_heuristic(
        adj, goal_ids, kill8, danger8, len(sentries)
    )

    def heuristic(p, mask):
        return htable[mask * n + p] if mask_heuristic else htable[p]

    start8 = (1 << len(sentries)) - 1
    start12 = (1 << len(patrols)) - 1
    start13 = (1 << len(sleepers)) - 1
    startp = index[player]
    h0 = heuristic(startp, start8)
    if h0 == _INF or h0 > depth_limit:
        return None

    trajectories = [_trajectory(graph, e) for e in patrols]
    occupancy = [[0] * n for _ in range(depth_limit + 2)]
    for i, (history, prefix, period) in enumerate(trajectories):
        bit = 1 << i
        for t in range(depth_limit + 2):
            k = t if t < prefix else prefix + (t - prefix) % period
            x, y, _ = history[k]
            p = index.get((x, y))
            if p is not None:
                occupancy[t][p] |= bit

    period_cache = {0: (0, 1)}

    def phase(t, alive):
        pair = period_cache.get(alive)
        if pair is None:
            prefix, period = 0, 1
            bits = alive
            while bits:
                bit = bits & -bits
                i = bit.bit_length() - 1
                _, mu, per = trajectories[i]
                prefix = max(prefix, mu)
                period = period * per // math.gcd(period, per)
                bits -= bit
            pair = (prefix, period)
            period_cache[alive] = pair
        mu, per = pair
        return t if t < mu else mu + (t - mu) % per

    start = (startp, start8, start12, start13, forbidden, phase(0, start12))
    costs = {start: 0}
    parents = {start: None}
    heap = [(1.08 * h0, h0, 0, start)]
    best_partial = None
    best_quality = (_INF, _INF)
    expansions = 0
    interrupted = False

    def recover(state):
        result = []
        while parents[state] is not None:
            state, action = parents[state]
            result.append(action)
        result.reverse()
        return result

    while heap:
        _, h, g, state = heapq.heappop(heap)
        expansions += 1
        if expansions % 128 == 0:
            if time.perf_counter() >= deadline or len(costs) >= 250000:
                interrupted = True
                break
        if costs.get(state) != g:
            continue

        p, alive8, alive12, asleep13, forbid, _ = state
        if p in goal_ids:
            return recover(state)

        if g and (h, g) < best_quality:
            best_quality = (h, g)
            best_partial = state
        if g >= depth_limit:
            continue

        for action in allowed:
            q = adj[p][action]
            if q < 0 or forbid & (1 << action):
                continue

            next8 = alive8 & ~kill8[q]
            if next8 & danger8[q]:
                continue

            next12 = alive12 & ~occupancy[g][q]
            if next12 & occupancy[g + 1][q]:
                continue

            next13 = asleep13 & ~kill13[q]
            next_forbid = (1 << _OPP[action]) if forbid else 0
            triggered = next13 & activation_all[q]
            if triggered:
                for d in range(4):
                    if triggered & activation[q][d]:
                        next_forbid |= 1 << _OPP[d]
                next13 &= ~triggered

            ng = g + 1
            nh = heuristic(q, next8)
            if nh == _INF or ng + nh > depth_limit:
                continue

            successor = (
                q, next8, next12, next13, next_forbid, phase(ng, next12)
            )
            if ng >= costs.get(successor, _INF):
                continue
            costs[successor] = ng
            parents[successor] = (state, action)
            heapq.heappush(heap, (ng + 1.08 * nh, nh, ng, successor))

    if interrupted and best_partial is not None:
        return recover(best_partial)
    return None


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.0
    available = list(available)
    parsed = _parse(obs)
    if parsed is None:
        return _reset(available, memory)

    graph, player, observed, visible_goals, signature, geometry = parsed
    same_board = memory.get("signature") == signature

    if not same_board:
        memory.clear()
        entities = observed
    else:
        entities = _merge_entities(
            observed, memory.get("entities", []), geometry
        )

    goals = set(visible_goals)
    if same_board:
        goals.update(memory.get("goals", ()))
    goals.intersection_update(graph)

    counts = _counter_candidates(
        obs, memory.get("counts") if same_board else None
    )
    memory["counts"] = counts
    remaining_upper = max((r for _, r in counts), default=200)
    if remaining_upper <= 0:
        return _reset(available, memory)

    allowed = [i for i, name in enumerate(_NAMES) if name in available]
    if not allowed:
        return _reset(available, memory)

    visits = memory.setdefault("visits", {})
    visits[player] = visits.get(player, 0) + 1
    goal_key = tuple(sorted(goals))

    cache_valid = (
        same_board
        and memory.get("expected_player") == player
        and tuple(sorted(memory.get("entities", []))) == tuple(sorted(entities))
        and memory.get("goal_key") == goal_key
        and memory.get("plan")
        and memory["plan"][0] in allowed
    )

    if cache_valid:
        plan = list(memory["plan"])
    else:
        search_goals = goals
        if not search_goals:
            # A moving sprite may temporarily hide the exit.
            candidates = [p for p in graph if p != player]
            if not candidates:
                return _reset(available, memory)
            fewest = min(visits.get(p, 0) for p in candidates)
            search_goals = {
                p for p in candidates if visits.get(p, 0) == fewest
            }

        plan = _search(
            graph, player, entities, search_goals, allowed,
            min(200, remaining_upper), deadline
        )

    if not plan or len(plan) > remaining_upper:
        return _reset(available, memory)

    action = plan[0]
    target = graph[player][action]
    if target is None or action not in allowed:
        return _reset(available, memory)

    memory["signature"] = signature
    memory["goals"] = tuple(sorted(goals))
    memory["goal_key"] = goal_key
    memory["expected_player"] = target
    memory["entities"] = _predict(graph, player, entities, action)
    memory["plan"] = plan[1:]

    return _NAMES[action], None, None

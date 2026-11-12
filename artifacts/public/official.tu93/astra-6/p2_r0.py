import math
import time
import heapq
from collections import deque

import numpy as np


# Direction order matches ACTION1 ... ACTION4.
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _add(p, d, distance=1):
    dx, dy = _DIRS[d]
    return (p[0] + distance * dx, p[1] + distance * dy)


def _read_frame(obs):
    """Read the graph, player, goals, and visible enemies."""
    yy, xx = np.where(obs[:63] == 9)
    if not len(xx):
        return None

    # Normally there is exactly one 3x3 player. Use its connected component
    # so an unrelated occurrence of the same colour cannot enlarge its box.
    start = (int(xx[0]), int(yy[0]))
    seen = {start}
    todo = [start]
    for x, y in todo:
        for dx, dy in _DIRS:
            q = (x + dx, y + dy)
            if (0 <= q[0] < 64 and 0 <= q[1] < 63
                    and q not in seen and obs[q[1], q[0]] == 9):
                seen.add(q)
                todo.append(q)

    x0 = min(p[0] for p in seen)
    y0 = min(p[1] for p in seen)
    width = max(p[0] for p in seen) - x0 + 1
    height = max(p[1] for p in seen) - y0 + 1
    scale = max(1, int(round(max(width, height) / 3.0)))
    stride = 6 * scale
    half = 3 * scale
    ax, ay = x0 % stride, y0 % stride
    player = ((x0 - ax) // stride, (y0 - ay) // stride)

    def pixel(x, y):
        if 0 <= x < 64 and 0 <= y < 63:
            return int(obs[y, x])
        return -1

    goals = set()
    enemies = []
    nodes = set()
    edges = set()
    limit = 64 // stride + 2
    centre_offset = scale + scale // 2

    for gy in range(-1, limit):
        for gx in range(-1, limit):
            x = ax + gx * stride
            y = ay + gy * stride
            cx, cy = x + centre_offset, y + centre_offset
            p = (gx, gy)
            c = pixel(cx, cy)

            if c in (0, 8, 9, 12, 13, 14):
                nodes.add(p)
            if c == 14:
                goals.add(p)

            if c in (8, 12, 13):
                marker_samples = (
                    (x + scale + scale // 2, y + scale // 2),
                    (x + scale + scale // 2, y + 2 * scale + scale // 2),
                    (x + scale // 2, y + scale + scale // 2),
                    (x + 2 * scale + scale // 2, y + scale + scale // 2),
                )
                direction, active = 0, 0
                for d, (mx, my) in enumerate(marker_samples):
                    marker = pixel(mx, my)
                    if marker in (11, 15):
                        direction = d
                        active = int(marker == 11)
                        break

                queued = -1
                if c == 13 and active:
                    # An activated purple enemy is two player moves behind:
                    # player = enemy + current_direction + queued_direction.
                    nxt = _add(p, direction)
                    delta = (player[0] - nxt[0], player[1] - nxt[1])
                    queued = _DIRS.index(delta) if delta in _DIRS else direction

                enemies.append((c, gx, gy, direction, active, queued))

            if pixel(cx + half, cy) == 2:
                edges.add((p, (gx + 1, gy)))
            if pixel(cx, cy + half) == 2:
                edges.add((p, (gx, gy + 1)))

    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    nodes.add(player)
    nodes.update(goals)

    graph = {p: [None] * 4 for p in nodes}
    for u, v in edges:
        d = 3 if v[0] > u[0] else 1
        graph[u][d] = v
        graph[v][d ^ 1] = u

    signature = (scale, ax, ay, tuple(sorted(edges)))
    return graph, player, goals, tuple(sorted(set(enemies))), signature


def _advance(graph, player, enemies, action):
    """Exact surviving end-of-action transition."""
    links = graph.get(player)
    if links is None or links[action] is None:
        return None
    dest = links[action]
    result = []

    for enemy in enemies:
        colour, x, y, direction, active, queued = enemy
        p = (x, y)

        # The player's attack finishes before any enemy moves.
        if p == dest:
            continue

        if colour == 8:
            if _add(p, direction) == dest:
                return None
            result.append(enemy)

        elif colour == 12:
            q = _add(p, direction)
            if q == dest:
                return None
            new_direction = direction
            if graph.get(q, (None,) * 4)[direction] is None:
                new_direction ^= 1
            result.append((12, q[0], q[1], new_direction, 0, -1))

        elif colour == 13:
            if active:
                q = _add(p, direction)
                if q == dest:
                    return None
                new_direction = queued if queued >= 0 else direction
                result.append((13, q[0], q[1], new_direction, 1, action))
            elif _add(p, direction, 2) == dest:
                result.append((13, x, y, direction, 1, direction))
            else:
                result.append(enemy)

    return dest, tuple(sorted(set(result)))


def _budget_candidates(width, old):
    """Infer possible remaining move budgets from the progress bar."""
    if old:
        updated = []
        for maximum, remaining in old:
            remaining -= 1
            if remaining >= 0 and round(64 * remaining / maximum) == width:
                updated.append((maximum, remaining))
        if updated:
            return updated

    candidates = []
    for maximum in range(1, 201):
        approximate = width * maximum / 64.0
        lo = max(0, int(approximate) - 2)
        hi = min(maximum, int(approximate) + 3)
        for remaining in range(lo, hi + 1):
            if round(64 * remaining / maximum) == width:
                candidates.append((maximum, remaining))
    return candidates or [(200, 200)]


def _patrol_cycle(graph, enemy):
    _, x, y, direction, _, _ = enemy
    p = (x, y)
    seen = {}
    positions = []

    while (p, direction) not in seen:
        seen[(p, direction)] = len(positions)
        positions.append(p)
        p = _add(p, direction)
        if graph.get(p, (None,) * 4)[direction] is None:
            direction ^= 1

    prefix = seen[(p, direction)]
    return positions, prefix, len(positions) - prefix


def _make_problem(graph, player, enemies, goals, allowed, max_depth):
    coordinates = sorted(graph)
    index = {p: i for i, p in enumerate(coordinates)}
    n = len(coordinates)

    neighbours = [[] for _ in range(n)]
    for p, i in index.items():
        for d in allowed:
            q = graph[p][d]
            if q is not None:
                neighbours[i].append((d, index[q]))

    # Relaxed graph distances are an admissible search heuristic.
    distance = [10 ** 6] * n
    queue = deque()
    for goal in goals:
        if goal in index:
            i = index[goal]
            distance[i] = 0
            queue.append(i)
    while queue:
        u = queue.popleft()
        p = coordinates[u]
        for q in graph[p]:
            if q is not None:
                v = index[q]
                if distance[v] > distance[u] + 1:
                    distance[v] = distance[u] + 1
                    queue.append(v)

    kill = [0] * n
    red_danger = [0] * n
    activate = [0] * n
    activation_directions = [[] for _ in range(n)]
    initial_ban = 0
    statics = []
    patrols = []

    for enemy in enemies:
        colour, x, y, direction, active, queued = enemy
        p = (x, y)

        if colour == 13 and active:
            hazard = _add(p, direction)
            delta = (hazard[0] - player[0], hazard[1] - player[1])
            if delta in _DIRS:
                initial_ban |= 1 << _DIRS.index(delta)
            else:
                # Preserve the fact that pursuit is active even if a
                # partially hidden initial observation is unusual.
                initial_ban |= 16

        elif colour == 12:
            # Opposite-parity patrols can be captured, but cannot land on
            # the player. Same-parity patrols are dangerous and cannot be
            # captured. Thus their phase needs no capture-mask dimension.
            if (x + y - player[0] - player[1]) % 2 == 0:
                patrols.append(_patrol_cycle(graph, enemy))

        else:
            statics.append(enemy)

    for k, enemy in enumerate(statics):
        colour, x, y, direction, active, queued = enemy
        bit = 1 << k
        p = (x, y)
        if p in index:
            kill[index[p]] |= bit
        if colour == 8:
            q = _add(p, direction)
            if q in index:
                red_danger[index[q]] |= bit
        elif colour == 13:
            q = _add(p, direction, 2)
            if q in index:
                i = index[q]
                activate[i] |= bit
                activation_directions[i].append(
                    (bit, 1 << (direction ^ 1))
                )

    prefix = 0
    period = 1
    for positions, pre, per in patrols:
        prefix = max(prefix, pre)
        period = period * per // math.gcd(period, per)

    # Avoid materialising a very large least-common-multiple cycle.
    cyclic = prefix + period <= max_depth + 1
    phase_count = prefix + period if cyclic else max_depth + 1
    phase_count = max(1, phase_count)
    patrol_danger = [0] * phase_count

    for t in range(phase_count):
        bits = 0
        for positions, pre, per in patrols:
            when = t + 1
            if when >= len(positions):
                when = pre + (when - pre) % per
            q = positions[when]
            if q in index:
                bits |= 1 << index[q]
        patrol_danger[t] = bits

    next_phase = list(range(1, phase_count + 1))
    next_phase[-1] = prefix if cyclic else phase_count - 1

    start = (
        index[player],
        (1 << len(statics)) - 1,
        0,
        initial_ban,
    )
    return {
        "start": start,
        "neighbours": neighbours,
        "distance": distance,
        "kill": kill,
        "red": red_danger,
        "activate": activate,
        "activation_directions": activation_directions,
        "patrol": patrol_danger,
        "next_phase": next_phase,
        "max_depth": max_depth,
    }


def _reconstruct(parent, state):
    route = []
    while state in parent:
        state, action = parent[state]
        route.append(action)
    route.reverse()
    return route


def _search(problem, weight, deadline, expansion_limit):
    """Return (complete route, exhausted, promising partial route)."""
    start = problem["start"]
    neighbours = problem["neighbours"]
    h = problem["distance"]
    kill = problem["kill"]
    red = problem["red"]
    activate = problem["activate"]
    activation_directions = problem["activation_directions"]
    patrol = problem["patrol"]
    next_phase = problem["next_phase"]
    max_depth = problem["max_depth"]

    if h[start[0]] > max_depth:
        return None, True, []

    best_g = {start: 0}
    parent = {}
    serial = 0
    heap = [(weight * h[start[0]], 0, serial, start)]
    partial = start
    partial_rank = (h[start[0]], 0)
    expanded = 0

    while heap:
        if expanded >= expansion_limit:
            return None, False, _reconstruct(parent, partial)
        if expanded % 256 == 0 and time.monotonic() >= deadline:
            return None, False, _reconstruct(parent, partial)

        _, neg_g, _, state = heapq.heappop(heap)
        g = -neg_g
        if best_g.get(state) != g:
            continue

        p, mask, phase, ban = state
        if h[p] == 0:
            return _reconstruct(parent, state), False, []

        expanded += 1
        rank = (h[p], g)
        if rank < partial_rank:
            partial_rank, partial = rank, state
        if g >= max_depth:
            continue

        new_phase = next_phase[phase]
        danger = patrol[phase]
        ng = g + 1

        for action, dest in neighbours[p]:
            if ban & (1 << action):
                continue
            if ng + h[dest] > max_depth:
                continue
            if danger & (1 << dest):
                continue

            remaining = mask & ~kill[dest]
            if remaining & red[dest]:
                continue

            # Every already-active purple enemy's next hazard becomes
            # the player's previous location: immediate reversal is fatal.
            new_ban = (1 << (action ^ 1)) if ban else 0
            triggered = remaining & activate[dest]
            if triggered:
                for bit, forbidden in activation_directions[dest]:
                    if triggered & bit:
                        new_ban |= forbidden
                remaining &= ~triggered

            child = (dest, remaining, new_phase, new_ban)
            if best_g.get(child, 10 ** 9) <= ng:
                continue
            best_g[child] = ng
            parent[child] = (state, action)
            serial += 1
            heapq.heappush(
                heap,
                (ng + weight * h[dest], -ng, serial, child),
            )

    return None, True, _reconstruct(parent, partial)


def _fallback(graph, player, enemies, goals, allowed, visits):
    """Safe exploration when a goal is temporarily covered or search times out."""
    distances = {}
    queue = deque()
    for goal in goals:
        if goal in graph:
            distances[goal] = 0
            queue.append(goal)
    while queue:
        p = queue.popleft()
        for q in graph[p]:
            if q is not None and q not in distances:
                distances[q] = distances[p] + 1
                queue.append(q)

    choices = []
    for action in allowed:
        result = _advance(graph, player, enemies, action)
        if result is None:
            continue
        p, es = result
        onward = sum(
            _advance(graph, p, es, a) is not None for a in allowed
        )
        score = (
            4 * distances.get(p, 0)
            + 3 * visits.get(p, 0)
            + (100 if not onward and p not in goals else 0)
            - 2 * (len(enemies) - len(es))
        )
        choices.append((score, action))
    return min(choices)[1] if choices else None


def act(obs, available, memory):
    started = time.monotonic()
    old = memory.get("_tu93", {})
    parsed = _read_frame(obs)

    def reset():
        memory.pop("_tu93", None)
        return ("RESET", None, None)

    if parsed is None:
        return reset()

    graph, player, visible_goals, visible_enemies, signature = parsed
    allowed = [i for i, name in enumerate(_NAMES) if name in available]
    if not allowed:
        allowed = list(range(4))

    width = int(np.count_nonzero(obs[63] == 6))
    budgets = _budget_candidates(width, old.get("budgets"))
    max_remaining = max(r for _, r in budgets)
    if max_remaining <= 0:
        return reset()

    same_board = old.get("board") == signature
    goals = set(visible_goals)
    if same_board:
        goals.update(old.get("goals", ()))

    expected = old.get("expected") if same_board else None
    reuse = expected is not None and expected[0] == player
    enemies = visible_enemies
    route = []

    if reuse:
        # Retain predicted enemies that are hidden underneath another
        # sprite. Add newly revealed enemies and invalidate the old plan.
        predicted = list(expected[1])
        known = {e[:5]: i for i, e in enumerate(predicted)}
        unexpected = False
        for enemy in visible_enemies:
            key = enemy[:5]
            if key in known:
                predicted[known[key]] = enemy
            else:
                known[key] = len(predicted)
                predicted.append(enemy)
                unexpected = True
        enemies = tuple(sorted(set(predicted)))
        if not unexpected and goals == set(old.get("goals", ())):
            route = list(old.get("route", ()))

    visits = dict(old.get("visits", {})) if same_board else {}
    visits[player] = visits.get(player, 0) + 1
    complete_plan = bool(route)

    if route:
        if len(route) > max_remaining:
            return reset()
        if route[0] not in allowed or _advance(
                graph, player, enemies, route[0]) is None:
            route = []
            complete_plan = False

    if not route and goals:
        max_depth = min(120, max_remaining)
        problem = _make_problem(
            graph, player, enemies, goals, allowed, max_depth
        )
        solution, exhausted, partial = _search(
            problem, 1.0, started + 4.3, 150000
        )

        if solution is None and not exhausted:
            solution2, exhausted2, partial2 = _search(
                problem, 1.8, started + 8.0, 180000
            )
            if solution2 is not None:
                solution = solution2
            elif exhausted2:
                exhausted = True
            elif partial2:
                partial = partial2

        if solution is not None:
            route = solution
            complete_plan = True
        elif exhausted:
            # This can occur in perturbed starts after a pursuer has
            # trapped the player, or when too little budget remains.
            return reset()
        elif partial:
            route = partial[:1]
            complete_plan = False

    if route:
        action = route[0]
        result = _advance(graph, player, enemies, action)
        if result is None or action not in allowed:
            route = []
        else:
            next_route = route[1:] if complete_plan else []
    if not route:
        action = _fallback(
            graph, player, enemies, goals, allowed, visits
        )
        if action is None:
            return reset()
        result = _advance(graph, player, enemies, action)
        next_route = []

    if max_remaining <= 1 and result[0] not in goals:
        return reset()

    memory["_tu93"] = {
        "board": signature,
        "goals": tuple(sorted(goals)),
        "expected": result,
        "route": next_route,
        "budgets": budgets,
        "visits": visits,
    }
    return (_NAMES[action], None, None)

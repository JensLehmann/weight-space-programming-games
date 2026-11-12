import math
import time
import heapq
from collections import deque

import numpy as np


# Directions correspond to ACTION1, ACTION2, ACTION3, ACTION4.
_DX = (0, 0, -1, 1)
_DY = (-1, 1, 0, 0)
_OPP = (1, 0, 3, 2)
_INF = 10**8


def _direction(patch, marker_colors, default=0):
    mask = np.zeros(patch.shape, dtype=bool)
    for color in marker_colors:
        mask |= patch == color
    yy, xx = np.where(mask)
    if not len(xx):
        return default
    dx = float(xx.mean()) - (patch.shape[1] - 1) / 2
    dy = float(yy.mean()) - (patch.shape[0] - 1) / 2
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2
    return 1 if dy > 0 else 0


def _player_box(obs):
    points = set(zip(*np.where(obs == 9)))
    components = []
    while points:
        seed = points.pop()
        stack = [seed]
        component = [seed]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x),
                      (y, x - 1), (y, x + 1)):
                if q in points:
                    points.remove(q)
                    stack.append(q)
                    component.append(q)
        components.append(component)

    components.sort(key=len, reverse=True)
    for component in components:
        ys = [p[0] for p in component]
        xs = [p[1] for p in component]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        if w == h and w >= 3 and w % 3 == 0:
            if len(component) >= 7 * (w // 3) ** 2:
                return x, y, w
    return None


def _decode(obs):
    box = _player_box(obs)
    if box is None:
        return None
    px, py, tile = box
    step = 2 * tile
    pdir = _direction(obs[py:py + tile, px:px + tile], (4,))
    rx, ry = px % step, py % step
    enemies = []
    goals = set()

    for y in range(ry, 64 - tile + 1, step):
        for x in range(rx, 64 - tile + 1, step):
            color = int(obs[y + tile // 2, x + tile // 2])
            if color == 14:
                goals.add((x, y))
            elif color in (8, 12, 13):
                patch = obs[y:y + tile, x:x + tile]
                direction = _direction(patch, (15, 11))
                active = bool(np.any(patch == 11))
                queue_dir = -1
                if color == 13 and active:
                    # An activated follower is exactly two queued moves
                    # behind the player. This recovers its hidden queue.
                    vx = px - x - step * _DX[direction]
                    vy = py - y - step * _DY[direction]
                    queue_dir = pdir
                    for a in range(4):
                        if (vx, vy) == (step * _DX[a], step * _DY[a]):
                            queue_dir = a
                            break
                enemies.append(
                    (color, x, y, direction, active, queue_dir)
                )
    return tile, (px, py), pdir, enemies, goals


def _make_graph(obs, tile, player, goals):
    step = 2 * tile
    rx, ry = player[0] % step, player[1] % step
    edges = []
    vertices = {player}
    vertices.update(goals)

    for y in range(ry, 64 - tile + 1, step):
        for x in range(rx, 64 - tile + 1, step):
            if x + step + tile <= 64:
                if obs[y + tile // 2, x + tile + tile // 2] == 2:
                    u, v = (x, y), (x + step, y)
                    edges.append((u, v))
                    vertices.update((u, v))
            if y + step + tile <= 64:
                if obs[y + tile + tile // 2, x + tile // 2] == 2:
                    u, v = (x, y), (x, y + step)
                    edges.append((u, v))
                    vertices.update((u, v))

    signature = (tile, rx, ry, tuple(edges))
    coords = sorted(vertices, key=lambda p: (p[1], p[0]))
    index = {p: i for i, p in enumerate(coords)}
    neighbors = [[-1] * 4 for _ in coords]
    for u, v in edges:
        i, j = index[u], index[v]
        if u[0] == v[0]:
            neighbors[i][1] = j
            neighbors[j][0] = i
        else:
            neighbors[i][3] = j
            neighbors[j][2] = i
    return signature, coords, index, neighbors


def _merge_enemies(obs, tile, visible, predicted):
    """Retain predicted enemies hidden underneath other enemies."""
    result = []
    for enemy in predicted:
        color, x, y, direction, active, queue_dir = enemy
        cx, cy = x + tile // 2, y + tile // 2
        if not (0 <= cx < 64 and 0 <= cy < 63):
            result.append(enemy)
        elif int(obs[cy, cx]) in (8, 12, 13):
            result.append(enemy)

    for enemy in visible:
        candidates = [
            i for i, old in enumerate(result)
            if old[:3] == enemy[:3]
        ]
        if any(result[i] == enemy for i in candidates):
            continue
        if len(candidates) == 1:
            result[candidates[0]] = enemy
        else:
            result.append(enemy)
    return result


def _advance(player, enemies, action, step, index, neighbors, coords):
    """Exact one-turn prediction, also used to validate planned actions."""
    pi = index.get(player)
    if pi is None:
        return None
    ni = neighbors[pi][action]
    if ni < 0:
        return None
    target = coords[ni]

    # The player captures occupants before enemies move.
    survivors = [
        e for e in enemies if (e[1], e[2]) != target
    ]
    result = []

    for color, x, y, direction, active, queue_dir in survivors:
        if color == 8:
            attack = (x + step * _DX[direction],
                      y + step * _DY[direction])
            if attack == target:
                return None
            result.append((color, x, y, direction, active, queue_dir))

        elif color == 12:
            x += step * _DX[direction]
            y += step * _DY[direction]
            if (x, y) == target:
                return None
            node = index.get((x, y))
            if node is None or neighbors[node][direction] < 0:
                direction = _OPP[direction]
            result.append((color, x, y, direction, active, queue_dir))

        elif color == 13:
            if active:
                x += step * _DX[direction]
                y += step * _DY[direction]
                if (x, y) == target:
                    return None
                direction = queue_dir if queue_dir >= 0 else direction
                queue_dir = action
            elif target == (x + 2 * step * _DX[direction],
                            y + 2 * step * _DY[direction]):
                active = True
                queue_dir = direction
            result.append((color, x, y, direction, active, queue_dir))

    return target, result


def _heuristics(neighbors, goals):
    n = len(neighbors)
    plain = [_INF] * n
    queue = deque()
    for g in goals:
        plain[g] = 0
        queue.append(g)
    while queue:
        v = queue.popleft()
        for u in neighbors[v]:
            if u >= 0 and plain[u] == _INF:
                plain[u] = plain[v] + 1
                queue.append(u)

    # Distance when an active follower prohibits reversing the last move.
    directed = [[_INF] * 4 for _ in range(n)]
    queue = deque()
    for g in goals:
        for a in range(4):
            directed[g][a] = 0
            queue.append((g, a))
    while queue:
        v, last = queue.popleft()
        u = neighbors[v][_OPP[last]]
        if u < 0:
            continue
        distance = directed[v][last] + 1
        for previous in range(4):
            if previous != _OPP[last] and directed[u][previous] == _INF:
                directed[u][previous] = distance
                queue.append((u, previous))

    table = [plain]
    for forbidden_queues in range(1, 16):
        row = [_INF] * n
        for v in range(n):
            if v in goals:
                row[v] = 0
                continue
            for a, u in enumerate(neighbors[v]):
                if u >= 0 and not (forbidden_queues & (1 << _OPP[a])):
                    row[v] = min(row[v], 1 + directed[u][a])
        table.append(row)
    return table


def _search(player, enemies, goals, step, coords, index, neighbors,
            allowed, max_depth, deadline):
    n = len(coords)
    start = index[player]
    goals = {index[g] for g in goals if g in index}
    if not goals:
        return None, False

    heuristic = _heuristics(neighbors, goals)
    static_kill = [0] * n
    threats = [0] * n
    triggers = [0] * n
    trigger_directions = [[0] * n for _ in range(4)]
    patrols = []
    alive = 0
    follower_queues = 0
    bit_number = 0

    for color, x, y, direction, active, queue_dir in enemies:
        if color == 13 and active:
            if queue_dir >= 0:
                follower_queues |= 1 << queue_dir
            continue

        bit = 1 << bit_number
        bit_number += 1
        alive |= bit
        node = index.get((x, y))

        if color == 12:
            patrols.append((bit, x, y, direction))
        else:
            if node is not None:
                static_kill[node] |= bit
            if color == 8:
                target = index.get((x + step * _DX[direction],
                                    y + step * _DY[direction]))
                if target is not None:
                    threats[target] |= bit
            elif color == 13:
                target = index.get((x + 2 * step * _DX[direction],
                                    y + 2 * step * _DY[direction]))
                if target is not None:
                    triggers[target] |= bit
                    trigger_directions[direction][target] |= bit

    # Patrol positions are deterministic functions of the valid-move count.
    patrol_at = [[0] * n for _ in range(max_depth + 1)]
    periods = []
    patrol_bits = 0

    for bit, x, y, direction in patrols:
        patrol_bits |= bit
        seen = {}
        history = []
        state = (x, y, direction)

        while state not in seen and len(history) < 512:
            seen[state] = len(history)
            history.append(state)
            x, y, direction = state
            x += step * _DX[direction]
            y += step * _DY[direction]
            node = index.get((x, y))
            if node is None or neighbors[node][direction] < 0:
                direction = _OPP[direction]
            state = (x, y, direction)

        if state in seen:
            prefix = seen[state]
            period = len(history) - prefix
        else:
            prefix, period = len(history), 0

        periods.append((bit, period if prefix == 0 else 0))
        for t in range(max_depth + 1):
            if t < len(history):
                x, y, _ = history[t]
            elif period:
                x, y, _ = history[prefix + (t - prefix) % period]
            else:
                break
            node = index.get((x, y))
            if node is not None:
                patrol_at[t][node] |= bit

    period_cache = {0: 1}

    def phase(mask, depth):
        mask &= patrol_bits
        modulus = period_cache.get(mask)
        if modulus is None:
            modulus = 1
            for bit, period in periods:
                if mask & bit:
                    if period == 0:
                        modulus = 0
                        break
                    modulus = modulus * period // math.gcd(modulus, period)
            period_cache[mask] = modulus
        return depth % modulus if modulus else depth

    initial = (start, alive, follower_queues, 0)
    h0 = heuristic[follower_queues][start]
    if h0 >= _INF:
        return None, False

    best_g = {initial: 0}
    parent = {initial: None}
    serial = 0
    heap = [(h0, h0, serial, 0, initial)]
    best_partial = None
    partial_score = (_INF, _INF)
    expansions = 0
    weight = 1.0
    timed_out = False

    def reconstruct(key):
        path = []
        while parent[key] is not None:
            previous, action = parent[key]
            path.append(action)
            key = previous
        path.reverse()
        return path

    while heap:
        _, h, _, depth, key = heapq.heappop(heap)
        if best_g.get(key) != depth:
            continue

        position, mask, queues, _ = key
        if position in goals and depth:
            return reconstruct(key), True
        if depth >= max_depth:
            continue

        expansions += 1
        if expansions % 256 == 0:
            if time.perf_counter() >= deadline or expansions >= 220000:
                timed_out = True
                break

        # Usually ordinary A* finishes quickly. Under a large search,
        # emphasize progress while retaining the explored state space.
        if expansions == 70000:
            weight = 1.7
            heap = [
                (g + weight * hh, hh, s, g, k)
                for _, hh, s, g, k in heap
                if best_g.get(k) == g
            ]
            heapq.heapify(heap)

        next_depth = depth + 1
        for action in allowed:
            if queues & (1 << _OPP[action]):
                continue
            target = neighbors[position][action]
            if target < 0:
                continue

            new_mask = mask & ~(
                static_kill[target] | patrol_at[depth][target]
            )
            if new_mask & threats[target]:
                continue
            if new_mask & patrol_at[next_depth][target]:
                continue

            new_queues = (1 << action) if queues else 0
            activated = new_mask & triggers[target]
            if activated:
                new_mask &= ~activated
                for d in range(4):
                    if activated & trigger_directions[d][target]:
                        new_queues |= 1 << d

            new_h = heuristic[new_queues][target]
            if new_h >= _INF or next_depth + new_h > max_depth:
                continue

            new_key = (
                target, new_mask, new_queues,
                phase(new_mask, next_depth)
            )
            if next_depth >= best_g.get(new_key, _INF):
                continue

            best_g[new_key] = next_depth
            parent[new_key] = (key, action)
            serial += 1
            heapq.heappush(
                heap,
                (next_depth + weight * new_h, new_h,
                 serial, next_depth, new_key)
            )
            score = (new_h, next_depth)
            if score < partial_score:
                partial_score = score
                best_partial = new_key

    if timed_out and best_partial is not None:
        return reconstruct(best_partial), False
    return None, False


def _budget_candidates(bar):
    candidates = []
    for total in range(1, 257):
        center = int(bar * total / 64)
        for remaining in range(max(0, center - 3),
                               min(total, center + 4) + 1):
            if round(64 * remaining / total) == bar:
                candidates.append((total, remaining))
    return candidates


def _reset(memory):
    memory["reset_pending"] = True
    memory.pop("prediction", None)
    memory["plan"] = []
    memory.pop("budgets", None)
    return "RESET", None, None


def act(obs, available, memory):
    began = time.perf_counter()
    obs = np.asarray(obs)
    decoded = _decode(obs)
    if decoded is None:
        return _reset(memory)

    tile, player, pdir, visible, observed_goals = decoded
    step = 2 * tile
    signature, coords, index, neighbors = _make_graph(
        obs, tile, player, observed_goals
    )

    if memory.get("signature") != signature:
        memory.clear()
        memory["signature"] = signature

    reset_pending = memory.pop("reset_pending", False)
    prediction = None if reset_pending else memory.get("prediction")
    prediction_valid = (
        prediction is not None and prediction[0] == player
    )

    if prediction_valid:
        enemies = _merge_enemies(obs, tile, visible, prediction[1])
        unchanged = sorted(enemies) == sorted(prediction[1])
    else:
        enemies = visible
        unchanged = False

    previous_goals = set(memory.get("goals", ()))
    goals = previous_goals | observed_goals
    memory["goals"] = goals
    goals_changed = goals != previous_goals

    visits = memory.setdefault("visits", {})
    visits[player] = visits.get(player, 0) + 1

    # The bottom bar gradually identifies the unknown level step limit.
    bar = int(np.count_nonzero(obs[63] == 6))
    candidates = None
    if prediction_valid and "budgets" in memory:
        candidates = [
            (total, remaining - 1)
            for total, remaining in memory["budgets"]
            if remaining > 0
            and round(64 * (remaining - 1) / total) == bar
        ]
    if not candidates:
        candidates = _budget_candidates(bar)
    memory["budgets"] = candidates

    remaining_upper = max((r for _, r in candidates), default=128)
    if remaining_upper <= 0:
        return _reset(memory)
    max_depth = min(128, remaining_upper)

    allowed = [
        a for a in range(4)
        if "ACTION" + str(a + 1) in available
    ]
    if not allowed:
        allowed = list(range(4))

    path = None
    complete_plan = False
    old_plan = memory.get("plan", [])
    if (prediction_valid and unchanged and not goals_changed
            and old_plan and len(old_plan) <= max_depth):
        if old_plan[0] in allowed:
            test = _advance(
                player, enemies, old_plan[0], step,
                index, neighbors, coords
            )
            if test is not None:
                path = list(old_plan)
                complete_plan = True

    if path is None:
        targets = {g for g in goals if g in index}
        known_exit = bool(targets)

        if not targets:
            # An enemy can temporarily cover the exit. First investigate
            # enemy squares; a moving patrol often reveals it immediately.
            targets = {
                (e[1], e[2]) for e in enemies
                if not (e[0] == 13 and e[4])
                and (e[1], e[2]) in index
                and (e[1], e[2]) != player
            }
            if not targets:
                possible = [p for p in coords if p != player]
                if possible:
                    fewest = min(visits.get(p, 0) for p in possible)
                    targets = {
                        p for p in possible
                        if visits.get(p, 0) == fewest
                    }

        path, complete_plan = _search(
            player, enemies, targets, step, coords, index, neighbors,
            allowed, max_depth, began + 6.8
        )
        if path is None and known_exit:
            return _reset(memory)

    predicted = None
    action = None
    if path:
        action = path[0]
        predicted = _advance(
            player, enemies, action, step, index, neighbors, coords
        )

    if predicted is None:
        # Conservative fallback for a partly occluded or unexpected state.
        choices = []
        for candidate in allowed:
            outcome = _advance(
                player, enemies, candidate, step,
                index, neighbors, coords
            )
            if outcome is None:
                continue
            destination = outcome[0]
            distance = min(
                (abs(destination[0] - g[0]) +
                 abs(destination[1] - g[1])) // step
                for g in goals
            ) if goals else 0
            score = distance + 3 * visits.get(destination, 0)
            choices.append((score, candidate, outcome))
        if not choices:
            return _reset(memory)
        _, action, predicted = min(choices, key=lambda x: (x[0], x[1]))
        path = [action]
        complete_plan = False

    memory["prediction"] = predicted
    memory["plan"] = path[1:] if complete_plan else []
    return "ACTION" + str(action + 1), None, None

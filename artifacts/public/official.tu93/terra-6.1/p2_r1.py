import heapq
import time
from collections import deque

import numpy as np


# Directions correspond to ACTION1, ACTION2, ACTION3, ACTION4.
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_OPPOSITE = (1, 0, 3, 2)


def _marker_direction(frame, x, y, scale, marker_colors, default=0):
    for direction, (dx, dy) in enumerate(_DIRS):
        xx, yy = x + dx * scale, y + dy * scale
        if 0 <= xx < 64 and 0 <= yy < 63:
            if int(frame[yy, xx]) in marker_colors:
                return direction, int(frame[yy, xx])
    return default, None


def _read_frame(obs):
    frame = np.asarray(obs)
    ys, xs = np.where(frame[:63] == 9)
    if len(xs) == 0:
        return None

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    width, height = xmax - xmin + 1, ymax - ymin + 1
    if width < 3 or height < 3:
        return None

    scale = max(1, min(width, height) // 3)
    x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
    player = y * 64 + x
    player_dir, _ = _marker_direction(frame, x, y, scale, {4})

    spacing = 6 * scale
    xs_grid = list(range(x % spacing, 64, spacing))
    ys_grid = list(range(y % spacing, 63, spacing))
    delta = tuple(dy * spacing * 64 + dx * spacing for dx, dy in _DIRS)

    graph = {}
    enemies = []
    goals = set()

    for yy in ys_grid:
        for xx in xs_grid:
            pos = yy * 64 + xx
            neighbors = [-1, -1, -1, -1]
            for direction, (dx, dy) in enumerate(_DIRS):
                nx, ny = xx + dx * spacing, yy + dy * spacing
                mx = xx + dx * (spacing // 2)
                my = yy + dy * (spacing // 2)
                if (0 <= nx < 64 and 0 <= ny < 63
                        and 0 <= mx < 64 and 0 <= my < 63
                        and int(frame[my, mx]) == 2):
                    neighbors[direction] = ny * 64 + nx
            graph[pos] = tuple(neighbors)

            color = int(frame[yy, xx])
            if color == 14:
                goals.add(pos)
            elif color in (8, 12, 13):
                direction, marker = _marker_direction(
                    frame, xx, yy, scale, {11, 15}
                )
                active = int(marker == 11)
                # An active purple piece normally has the player's previous
                # direction queued. Memory refines this when available.
                queued = player_dir if color == 13 and active else -1
                enemies.append((color, pos, direction, active, queued))

    return player, player_dir, tuple(sorted(enemies)), goals, graph, delta, scale


def _reconcile_enemies(observed, predicted, player_dir):
    if predicted is None:
        return observed

    predicted = [tuple(int(v) for v in e) for e in predicted]
    used = set()
    result = []
    visible_positions = {e[1] for e in observed}

    for enemy in observed:
        kind, pos, direction, active, queued = enemy
        candidates = [
            i for i, old in enumerate(predicted)
            if i not in used and old[0] == kind and old[1] == pos
        ]
        if candidates:
            # Prefer a matching orientation if several pieces overlap.
            index = min(
                candidates,
                key=lambda i: (
                    predicted[i][2] != direction,
                    predicted[i][3] != active,
                ),
            )
            used.add(index)
            old = predicted[index]
            if kind == 13 and active:
                queued = old[4] if old[3] and old[4] >= 0 else player_dir
        result.append((kind, pos, direction, active, queued))

    # A piece may be completely covered by another piece at the same node.
    for i, enemy in enumerate(predicted):
        if i in used:
            continue
        pos = enemy[1]
        x, y = pos % 64, pos // 64
        if pos in visible_positions or not (0 <= x < 64 and 0 <= y < 63):
            result.append(enemy)

    return tuple(sorted(result))


def _advance(state, action, graph, delta):
    player, enemies = state
    neighbors = graph.get(player)
    if neighbors is None:
        return None
    target = neighbors[action]
    if target < 0:
        return None

    updated = []
    for kind, pos, direction, active, queued in enemies:
        # The player captures pieces at the destination before they move.
        if pos == target:
            continue

        if kind == 8:
            # Blue pieces attack when the player ends one node ahead.
            if pos + delta[direction] == target:
                return None
            updated.append((kind, pos, direction, active, queued))

        elif kind == 12:
            new_pos = pos + delta[direction]
            if new_pos == target:
                return None
            next_neighbors = graph.get(new_pos)
            if next_neighbors is None or next_neighbors[direction] < 0:
                direction = _OPPOSITE[direction]
            updated.append((kind, new_pos, direction, active, -1))

        else:  # Purple follower
            if active:
                new_pos = pos + delta[direction]
                if new_pos == target:
                    return None
                new_direction = queued if queued >= 0 else direction
                updated.append((kind, new_pos, new_direction, 1, action))
            else:
                if pos + 2 * delta[direction] == target:
                    updated.append((kind, pos, direction, 1, direction))
                else:
                    updated.append((kind, pos, direction, 0, -1))

    return target, tuple(sorted(updated))


def _distances(graph, goals):
    distances = {}
    queue = deque()
    for goal in goals:
        if goal in graph:
            distances[goal] = 0
            queue.append(goal)

    while queue:
        pos = queue.popleft()
        d = distances[pos] + 1
        for neighbor in graph[pos]:
            if neighbor >= 0 and neighbor not in distances:
                distances[neighbor] = d
                queue.append(neighbor)
    return distances


def _path_to(state, parents):
    actions = []
    while parents[state] is not None:
        previous, action = parents[state]
        actions.append(action)
        state = previous
    actions.reverse()
    return actions


def _search(initial, graph, delta, goals, visits):
    distances = _distances(graph, goals)
    if initial[0] not in distances:
        return [], True

    started = time.monotonic()
    serial = 0
    heap = [(distances[initial[0]], 0, serial, initial)]
    best_cost = {initial: 0}
    parents = {initial: None}
    best_partial = initial
    best_partial_key = (
        distances[initial[0]],
        visits.get(str(initial[0]), 0),
        0,
    )
    expansions = 0
    exhausted = True

    while heap:
        if expansions % 512 == 0:
            if time.monotonic() - started > 5.5 or expansions >= 160000:
                exhausted = False
                break

        _, cost, _, state = heapq.heappop(heap)
        if cost != best_cost.get(state):
            continue

        player = state[0]
        if player in goals:
            return _path_to(state, parents), True

        expansions += 1
        for action in range(4):
            next_state = _advance(state, action, graph, delta)
            if next_state is None:
                continue
            next_pos = next_state[0]
            if next_pos not in distances:
                continue

            new_cost = cost + 1
            if new_cost >= best_cost.get(next_state, 10**9):
                continue

            best_cost[next_state] = new_cost
            parents[next_state] = (state, action)
            h = distances[next_pos]
            serial += 1
            heapq.heappush(
                heap, (new_cost + h, new_cost, serial, next_state)
            )

            key = (h, visits.get(str(next_pos), 0), new_cost)
            if key < best_partial_key:
                best_partial_key = key
                best_partial = next_state

    if best_partial != initial:
        return _path_to(best_partial, parents), exhausted
    return [], exhausted


def _exploration_goal(player, graph, visits):
    queue = deque([player])
    distance = {player: 0}
    while queue:
        pos = queue.popleft()
        for nxt in graph.get(pos, ()):
            if nxt >= 0 and nxt not in distance:
                distance[nxt] = distance[pos] + 1
                queue.append(nxt)

    candidates = [p for p in distance if p != player]
    if not candidates:
        return set()

    target = min(
        candidates,
        key=lambda p: (visits.get(str(p), 0), distance[p], p),
    )
    return {target}


def act(obs, available, memory):
    parsed = _read_frame(obs)
    if parsed is None:
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        return available[0], None, None

    player, player_dir, observed, visible_goals, graph, delta, scale = parsed
    signature = [scale, player % (6 * scale),
                 (player // 64) % (6 * scale)]

    if memory.get("signature") != signature:
        memory.clear()
        memory["signature"] = signature

    goals = set(int(p) for p in memory.get("goals", []))
    goals.update(visible_goals)
    memory["goals"] = sorted(goals)

    visits = memory.setdefault("visits", {})
    key = str(player)
    visits[key] = int(visits.get(key, 0)) + 1

    expected = memory.get("expected")
    predicted_enemies = None
    if expected is not None and int(expected[0]) == player:
        predicted_enemies = expected[1]
    enemies = _reconcile_enemies(observed, predicted_enemies, player_dir)
    state = (player, enemies)

    search_goals = goals if goals else _exploration_goal(player, graph, visits)
    goal_key = sorted(search_goals)

    plan = []
    if expected is not None and memory.get("plan_goals") == goal_key:
        expected_state = (
            int(expected[0]),
            tuple(sorted(tuple(int(v) for v in e) for e in expected[1])),
        )
        if state == expected_state:
            plan = [int(a) for a in memory.get("plan", [])]

    exhausted = False
    if not plan and search_goals:
        plan, exhausted = _search(state, graph, delta, search_goals, visits)

    if not plan:
        safe = []
        distances = _distances(graph, search_goals) if search_goals else {}
        for action in range(4):
            nxt = _advance(state, action, graph, delta)
            if nxt is None:
                continue
            safe.append((
                distances.get(nxt[0], 10**6),
                visits.get(str(nxt[0]), 0),
                action,
            ))

        if safe and not (goals and exhausted):
            plan = [min(safe)[2]]
        elif "RESET" in available:
            memory.clear()
            return "RESET", None, None
        elif safe:
            plan = [min(safe)[2]]
        else:
            for action in range(4):
                name = "ACTION" + str(action + 1)
                if name in available:
                    return name, None, None
            return available[0], None, None

    action = plan[0]
    name = "ACTION" + str(action + 1)
    if name not in available:
        for alternative in range(4):
            candidate = "ACTION" + str(alternative + 1)
            if candidate in available and _advance(
                state, alternative, graph, delta
            ) is not None:
                action, name, plan = alternative, candidate, [alternative]
                break

    next_state = _advance(state, action, graph, delta)
    if next_state is not None:
        memory["expected"] = [
            next_state[0],
            [list(enemy) for enemy in next_state[1]],
        ]
        memory["plan"] = plan[1:]
        memory["plan_goals"] = goal_key
    else:
        memory.pop("expected", None)
        memory["plan"] = []

    return name, None, None

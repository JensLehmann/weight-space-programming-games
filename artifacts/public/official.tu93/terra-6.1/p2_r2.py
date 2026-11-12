import heapq
import time
from collections import deque

import numpy as np


# Directions are ordered like ACTION1, ACTION2, ACTION3, ACTION4.
_DIRS = ((0, -6), (0, 6), (-6, 0), (6, 0))
_MARKERS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_OPPOSITE = (1, 0, 3, 2)


def _player(frame):
    ys, xs = np.where(frame == 4)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for d, (dx, dy) in enumerate(_MARKERS):
            cx, cy = x - dx, y - dy
            if not (1 <= cx < 63 and 1 <= cy < 62):
                continue
            block = frame[cy - 1:cy + 2, cx - 1:cx + 2]
            if np.count_nonzero(block == 9) == 8:
                return (cx, cy), d
    return None, None


def _read_frame(frame, player_pos):
    px, py = player_pos
    nodes = [
        (x, y)
        for y in range(py % 6, 63, 6)
        for x in range(px % 6, 64, 6)
    ]
    node_set = set(nodes)
    graph = {}
    for x, y in nodes:
        edges = []
        for d, (dx, dy) in enumerate(_DIRS):
            nxt = (x + dx, y + dy)
            mx, my = x + dx // 2, y + dy // 2
            if (
                nxt in node_set
                and 0 <= mx < 64
                and 0 <= my < 63
                and int(frame[my, mx]) == 2
            ):
                edges.append((d, nxt))
        graph[(x, y)] = tuple(edges)

    goals = []
    enemies = []
    for x, y in nodes:
        c = int(frame[y, x])
        if c == 14:
            goals.append((x, y))
        if c not in (8, 12, 13):
            continue

        direction = None
        active = False
        for d, (dx, dy) in enumerate(_MARKERS):
            xx, yy = x + dx, y + dy
            if 0 <= xx < 64 and 0 <= yy < 63:
                marker = int(frame[yy, xx])
                if marker in (11, 15):
                    direction = d
                    active = marker == 11
                    break
        if direction is not None:
            enemies.append((c, (x, y), direction, active))
    return graph, tuple(goals), tuple(enemies)


def _initial_state(player_pos, player_dir, enemies):
    blues = []
    patrols = []
    chasers = []

    for kind, pos, direction, active in enemies:
        if kind == 8:
            blues.append((pos, direction))
        elif kind == 12:
            patrols.append((pos, direction))
        else:
            if not active:
                queued = -1
            else:
                dx, dy = _DIRS[direction]
                # A newly activated chaser starts two nodes behind the player.
                if (pos[0] + 2 * dx, pos[1] + 2 * dy) == player_pos:
                    queued = direction
                else:
                    queued = player_dir
            chasers.append((pos, direction, queued))

    blues = tuple(blues)
    state = (
        player_pos,
        (1 << len(blues)) - 1,
        tuple(patrols),
        tuple(chasers),
    )
    return blues, state


def _matches(state, blues, player_pos, observed):
    if state[0] != player_pos:
        return False

    expected = {}
    _, mask, patrols, chasers = state

    for i, (pos, direction) in enumerate(blues):
        if mask & (1 << i):
            expected.setdefault(pos, []).append((8, direction, False))
    for item in patrols:
        if item is not None:
            pos, direction = item
            expected.setdefault(pos, []).append((12, direction, False))
    for item in chasers:
        if item is not None:
            pos, direction, queued = item
            expected.setdefault(pos, []).append(
                (13, direction, queued != -1)
            )

    seen = {}
    for kind, pos, direction, active in observed:
        seen[pos] = (kind, direction, active)

    # Several enemies can overlap. Only the topmost one is rendered.
    if set(seen) != set(expected):
        return False
    return all(seen[pos] in possibilities
               for pos, possibilities in expected.items())


def _advance(state, direction, graph, blues):
    player, mask, patrols, chasers = state
    destination = None
    for d, pos in graph.get(player, ()):
        if d == direction:
            destination = pos
            break
    if destination is None:
        return None

    # The player removes enemies at the destination before they move.
    new_mask = mask
    for i, (pos, _) in enumerate(blues):
        if pos == destination:
            new_mask &= ~(1 << i)

    new_patrols = []
    for item in patrols:
        if item is None or item[0] == destination:
            new_patrols.append(None)
        else:
            new_patrols.append(item)

    new_chasers = []
    for item in chasers:
        if item is None or item[0] == destination:
            new_chasers.append(None)
        else:
            new_chasers.append(item)

    # A blue enemy facing the player from an adjacent node attacks.
    for i, (pos, d) in enumerate(blues):
        if new_mask & (1 << i):
            dx, dy = _DIRS[d]
            if (pos[0] + dx, pos[1] + dy) == destination:
                return None

    # Patrols move one edge and reverse when their next edge is absent.
    for i, item in enumerate(new_patrols):
        if item is None:
            continue
        pos, d = item
        dx, dy = _DIRS[d]
        nxt = (pos[0] + dx, pos[1] + dy)
        if nxt == destination:
            return None
        if not any(edge_d == d for edge_d, _ in graph.get(nxt, ())):
            d = _OPPOSITE[d]
        new_patrols[i] = (nxt, d)

    # Active chasers move in their current direction, then consume the
    # queued direction. Dormant ones activate when the player is two
    # nodes directly ahead of them.
    for i, item in enumerate(new_chasers):
        if item is None:
            continue
        pos, d, queued = item
        if queued != -1:
            dx, dy = _DIRS[d]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt == destination:
                return None
            new_chasers[i] = (nxt, queued, direction)
        else:
            dx, dy = _DIRS[d]
            if (pos[0] + 2 * dx, pos[1] + 2 * dy) == destination:
                new_chasers[i] = (pos, d, d)

    return destination, new_mask, tuple(new_patrols), tuple(new_chasers)


def _distances(graph, goals):
    dist = {}
    queue = deque()
    for goal in goals:
        if goal in graph and goal not in dist:
            dist[goal] = 0
            queue.append(goal)
    while queue:
        pos = queue.popleft()
        for _, nxt in graph.get(pos, ()):
            if nxt not in dist:
                dist[nxt] = dist[pos] + 1
                queue.append(nxt)
    return dist


def _search(start, graph, blues, goals, allowed):
    dist = _distances(graph, goals)
    if start[0] not in dist:
        return None, True

    deadline = time.monotonic() + 4.5
    serial = 0
    heap = [(dist[start[0]], dist[start[0]], serial, 0, start)]
    best_cost = {start: 0}
    parent = {}
    best_partial = start
    best_partial_key = (dist[start[0]], 0)
    expanded = 0
    exhausted = True

    while heap:
        if expanded >= 180000 or time.monotonic() >= deadline:
            exhausted = False
            break

        _, h, _, cost, state = heapq.heappop(heap)
        if best_cost.get(state) != cost:
            continue

        if state[0] in goals:
            path = []
            while state != start:
                previous, direction = parent[state]
                path.append(direction)
                state = previous
            path.reverse()
            return path, False

        expanded += 1
        for direction, _ in graph.get(state[0], ()):
            if direction not in allowed:
                continue
            nxt = _advance(state, direction, graph, blues)
            if nxt is None or nxt[0] not in dist:
                continue
            new_cost = cost + 1
            if new_cost >= best_cost.get(nxt, 10**9):
                continue

            best_cost[nxt] = new_cost
            parent[nxt] = (state, direction)
            new_h = dist[nxt[0]]
            serial += 1
            heapq.heappush(
                heap, (new_cost + new_h, new_h, serial, new_cost, nxt)
            )
            key = (new_h, new_cost)
            if key < best_partial_key:
                best_partial_key = key
                best_partial = nxt

    if not exhausted and best_partial != start:
        path = []
        state = best_partial
        while state != start:
            previous, direction = parent[state]
            path.append(direction)
            state = previous
        path.reverse()
        return path, False

    return None, exhausted


def act(obs, available, memory):
    frame = np.asarray(obs)
    available = set(available)
    allowed = {
        d for d in range(4) if "ACTION" + str(d + 1) in available
    }
    if not allowed:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION1", None, None

    player_pos, player_dir = _player(frame)
    if player_pos is None:
        return "ACTION" + str(min(allowed) + 1), None, None

    graph, visible_goals, observed = _read_frame(frame, player_pos)
    goals = set(memory.get("goals", ()))
    goals.update(visible_goals)
    memory["goals"] = tuple(goals)

    visited = set(memory.get("visited", ()))
    visited.add(player_pos)
    memory["visited"] = tuple(visited)

    predicted = memory.get("predicted")
    blues = memory.get("blues")
    valid_prediction = (
        predicted is not None
        and blues is not None
        and _matches(predicted, blues, player_pos, observed)
    )

    if valid_prediction:
        state = predicted
    else:
        blues, state = _initial_state(player_pos, player_dir, observed)
        memory["blues"] = blues
        memory["plan"] = []
        memory["tentative"] = False

    plan = list(memory.get("plan", ()))
    if memory.get("tentative") and goals:
        plan = []
        memory["tentative"] = False

    # Validate the next cached move against the reconstructed graph.
    if plan:
        if plan[0] not in allowed or _advance(state, plan[0], graph, blues) is None:
            plan = []

    if not plan:
        targets = set(goals)
        tentative = False

        if not targets:
            # The exit may be hidden under an enemy. Move toward visible
            # enemies or unexplored nodes until it becomes visible.
            tentative = True
            targets = {pos for _, pos, _, _ in observed}
            targets.discard(player_pos)
            if not targets:
                targets = {
                    pos for pos, edges in graph.items()
                    if edges and pos not in visited
                }

        if targets:
            result, exhausted = _search(
                state, graph, blues, targets, allowed
            )
            if result:
                plan = result
                memory["tentative"] = tentative
            elif exhausted and goals and memory.get("resets", 0) < 1:
                if "RESET" in available:
                    memory["resets"] = memory.get("resets", 0) + 1
                    memory["plan"] = []
                    memory.pop("predicted", None)
                    return "RESET", None, None

    if not plan:
        # Safe local fallback if search was inconclusive.
        distance = _distances(graph, goals) if goals else {}
        choices = []
        for direction, pos in graph.get(player_pos, ()):
            if direction not in allowed:
                continue
            nxt = _advance(state, direction, graph, blues)
            if nxt is not None:
                choices.append((
                    distance.get(pos, 1000),
                    pos in visited,
                    direction,
                ))
        if choices:
            choices.sort()
            plan = [choices[0][2]]
        elif "RESET" in available and memory.get("resets", 0) < 1:
            memory["resets"] = memory.get("resets", 0) + 1
            memory["plan"] = []
            memory.pop("predicted", None)
            return "RESET", None, None
        else:
            return "ACTION" + str(min(allowed) + 1), None, None

    direction = plan.pop(0)
    next_state = _advance(state, direction, graph, blues)
    memory["plan"] = plan
    memory["predicted"] = next_state
    memory["blues"] = blues
    return "ACTION" + str(direction + 1), None, None

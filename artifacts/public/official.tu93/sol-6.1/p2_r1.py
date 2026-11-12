import numpy as np
import heapq
import math
import time
from collections import deque

# Directions correspond to ACTION1 through ACTION4.
_D = (-256, 256, -1, 1)
_OPPOSITE = (1, 0, 3, 2)
_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_DEAD = (-1000000000, 0, -2)
_EMPTY_EDGES = (None, None, None, None)


def _facing(tile, scale, marker_colors):
    # Marker positions in the unrotated and rotated 3x3 sprites.
    samples = ((0, 1), (2, 1), (1, 0), (1, 2))
    mid = scale // 2
    for direction, (row, col) in enumerate(samples):
        value = int(tile[row * scale + mid, col * scale + mid])
        if value in marker_colors:
            return direction
    return 0


def _read_frame(obs):
    image = np.asarray(obs)
    yy, xx = np.where(image[:63] == 9)
    if not len(xx):
        return None

    px, py = int(xx.min()), int(yy.min())
    extent = max(int(xx.max()) - px + 1, int(yy.max()) - py + 1)
    scale = max(1, extent // 3)
    tile_size = 3 * scale
    stride = 6 * scale

    if px + tile_size > 64 or py + tile_size > 63:
        return None

    x_origin = px % stride
    y_origin = py % stride
    xs = list(range(x_origin, 65 - tile_size, stride))
    ys = list(range(y_origin, 64 - tile_size, stride))

    player = ((px - x_origin) // stride
              + 256 * ((py - y_origin) // stride))
    player_tile = image[py:py + tile_size, px:px + tile_size]
    player_direction = _facing(player_tile, scale, (4,))

    graph = {}
    objects = {8: [], 12: [], 13: []}
    goals = set()
    offsets = ((0, -3 * scale), (0, 3 * scale),
               (-3 * scale, 0), (3 * scale, 0))
    threshold = 4 * scale * scale

    for iy, top in enumerate(ys):
        for ix, left in enumerate(xs):
            pos = ix + 256 * iy
            edges = [None] * 4
            sample_x = left + scale
            sample_y = top + scale

            for direction, (dx, dy) in enumerate(offsets):
                tx, ty = sample_x + dx, sample_y + dy
                nx = ix + (direction == 3) - (direction == 2)
                ny = iy + (direction == 1) - (direction == 0)
                if (0 <= nx < len(xs) and 0 <= ny < len(ys)
                        and 0 <= tx < 64 and 0 <= ty < 63
                        and int(image[ty, tx]) == 2):
                    edges[direction] = pos + _D[direction]
            graph[pos] = tuple(edges)

            tile = image[top:top + tile_size, left:left + tile_size]
            counts = np.bincount(tile.ravel().astype(np.int64), minlength=16)

            if counts[14] >= threshold:
                goals.add(pos)

            for color in (8, 12, 13):
                if counts[color] >= threshold:
                    direction = _facing(tile, scale, (15, 11))
                    active = bool(counts[11])
                    objects[color].append((pos, direction, active))

    for color in objects:
        objects[color].sort()

    signature = (
        scale, x_origin, y_origin,
        tuple((p, e) for p, e in sorted(graph.items())
              if any(v is not None for v in e))
    )
    return {
        "graph": graph,
        "signature": signature,
        "player": player,
        "direction": player_direction,
        "objects": objects,
        "goals": frozenset(goals),
    }


def _goal_guesses(parsed):
    graph = parsed["graph"]
    occupied = {
        pos for group in parsed["objects"].values()
        for pos, direction, active in group
    }
    endpoints = {
        p for p in occupied
        if sum(v is not None for v in graph.get(p, _EMPTY_EDGES)) == 1
    }
    if endpoints:
        return frozenset(endpoints)
    if occupied:
        return frozenset(occupied)
    return frozenset(
        p for p, edges in graph.items()
        if sum(v is not None for v in edges) == 1
        and p != parsed["player"]
    )


def _distances(graph, goals):
    distances = {p: 0 for p in goals}
    queue = deque(goals)
    while queue:
        p = queue.popleft()
        distance = distances[p] + 1
        for q in graph.get(p, _EMPTY_EDGES):
            if q is not None and q not in distances:
                distances[q] = distance
                queue.append(q)
    return distances


def _patrol_orbit(graph, pos, direction):
    sequence = []
    seen = {}
    current = (pos, direction)
    while current not in seen:
        seen[current] = len(sequence)
        sequence.append(current)
        p, d = current
        p += _D[d]
        if graph.get(p, _EMPTY_EDGES)[d] is None:
            d = _OPPOSITE[d]
        current = (p, d)
    prefix = seen[current]
    return tuple(sequence), prefix, len(sequence) - prefix


def _new_model(parsed):
    graph = parsed["graph"]
    blues = tuple((p, d) for p, d, active in parsed["objects"][8])
    kill_masks = {}
    danger_masks = {}
    for i, (p, d) in enumerate(blues):
        bit = 1 << i
        kill_masks[p] = kill_masks.get(p, 0) | bit
        front = p + _D[d]
        danger_masks[front] = danger_masks.get(front, 0) | bit

    patrols = tuple(
        _patrol_orbit(graph, p, d)
        for p, d, active in parsed["objects"][12]
    )
    prefix = max((orbit[1] for orbit in patrols), default=0)
    period = 1
    for sequence, start, length in patrols:
        period = period * length // math.gcd(period, length)

    purples = []
    p = parsed["player"]
    player_direction = parsed["direction"]
    for enemy_pos, direction, active in parsed["objects"][13]:
        if not active:
            queued = -1
        elif p == enemy_pos + 2 * _D[direction]:
            # Just activated, or two identical recent player moves.
            queued = direction
        else:
            # Once moving, its queue contains the latest player direction.
            queued = player_direction
        purples.append((enemy_pos, direction, queued))

    goals = parsed["goals"] or _goal_guesses(parsed)
    model = {
        "graph": graph,
        "signature": parsed["signature"],
        "blues": blues,
        "kill_masks": kill_masks,
        "danger_masks": danger_masks,
        "patrols": patrols,
        "prefix": prefix,
        "period": period,
        "turn_cache": {},
        "goals": goals,
        "guess": not bool(parsed["goals"]),
        "distances": _distances(graph, goals),
    }
    state = (
        p,
        (1 << len(blues)) - 1,
        (1 << len(patrols)) - 1,
        tuple(purples),
        0,
    )
    return model, state


def _patrol_turn(model, phase):
    cache = model["turn_cache"]
    if phase in cache:
        return cache[phase]

    result = []
    for sequence, prefix, period in model["patrols"]:
        index = phase if phase < prefix else prefix + (phase - prefix) % period
        result.append(sequence[index])
    result = tuple(result)
    cache[phase] = result
    return result


def _next_phase(model, phase):
    phase += 1
    if phase >= model["prefix"] + model["period"]:
        phase = model["prefix"]
    return phase


def _advance(model, state, action):
    player, blue_mask, patrol_mask, purples, phase = state
    target = model["graph"].get(player, _EMPTY_EDGES)[action]
    if target is None:
        return None

    # Entering an enemy's current cell destroys that enemy before it moves.
    new_blue_mask = blue_mask & ~model["kill_masks"].get(target, 0)

    # A surviving blue enemy attacks a player directly in front of it.
    if new_blue_mask & model["danger_masks"].get(target, 0):
        return None

    following_phase = _next_phase(model, phase)
    new_patrol_mask = patrol_mask
    if patrol_mask:
        current_patrols = _patrol_turn(model, phase)
        following_patrols = _patrol_turn(model, following_phase)
        for i, (pos, direction) in enumerate(current_patrols):
            bit = 1 << i
            if not (new_patrol_mask & bit):
                continue
            if target == pos:
                new_patrol_mask &= ~bit
            elif following_patrols[i][0] == target:
                return None

    new_purples = []
    changed = False
    for old in purples:
        pos, direction, queued = old
        if queued == -2:
            new = old
        elif target == pos:
            new = _DEAD
        elif queued == -1:
            if target == pos + 2 * _D[direction]:
                # Activation leaves one copy of the original direction queued.
                new = (pos, direction, direction)
            else:
                new = old
        else:
            pos += _D[direction]
            if pos == target:
                return None
            new = (pos, queued, action)
        new_purples.append(new)
        if new != old:
            changed = True

    if changed:
        new_purples = tuple(new_purples)
    else:
        new_purples = purples

    if not new_patrol_mask:
        following_phase = 0

    return (
        target, new_blue_mask, new_patrol_mask,
        new_purples, following_phase
    )


def _matches(model, state, parsed):
    if state[0] != parsed["player"]:
        return False

    player, blue_mask, patrol_mask, purples, phase = state
    expected = set()

    for i, (pos, direction) in enumerate(model["blues"]):
        if blue_mask & (1 << i):
            expected.add((8, pos, direction, False))

    patrol_turn = _patrol_turn(model, phase)
    for i, (pos, direction) in enumerate(patrol_turn):
        if patrol_mask & (1 << i):
            expected.add((12, pos, direction, False))

    for pos, direction, queued in purples:
        if queued != -2:
            expected.add((13, pos, direction, queued >= 0))

    observed = {
        (color, pos, direction, active)
        for color, group in parsed["objects"].items()
        for pos, direction, active in group
    }

    if not observed.issubset(expected):
        return False

    # Enemies can overlap and obscure one another in the rendering.
    occupied = {token[1] for token in observed}
    return all(token in observed or token[1] in occupied
               for token in expected)


def _trace(parents, state):
    route = []
    while parents[state] is not None:
        previous, action = parents[state]
        route.append(action)
        state = previous
    route.reverse()
    return route


def _search(model, initial, allowed):
    distances = model["distances"]
    if initial[0] not in distances:
        return None, False
    if initial[0] in model["goals"]:
        return [], True

    deadline = time.perf_counter() + 7.0
    scores = {initial: 0}
    parents = {initial: None}
    serial = 0
    h = distances[initial[0]]
    heap = [(1.25 * h, h, serial, 0, initial)]
    expanded = 0

    while heap:
        priority, heuristic, _, cost, state = heapq.heappop(heap)
        if scores.get(state) != cost:
            continue

        if state[0] in model["goals"]:
            return _trace(parents, state), True

        expanded += 1
        if expanded >= 180000 or (
                expanded % 128 == 0 and time.perf_counter() >= deadline):
            # A safe partial route also gives useful progress on large searches.
            route = _trace(parents, state)
            if route:
                return route[:8], False
            break

        if cost >= 100:
            continue

        for action in allowed:
            nxt = _advance(model, state, action)
            if nxt is None:
                continue
            h = distances.get(nxt[0])
            if h is None:
                continue
            new_cost = cost + 1
            if new_cost >= scores.get(nxt, 1000000):
                continue

            scores[nxt] = new_cost
            parents[nxt] = (state, action)
            serial += 1
            heapq.heappush(
                heap, (new_cost + 1.25 * h, h, serial, new_cost, nxt)
            )

    return None, False


def act(obs, available, memory):
    parsed = _read_frame(obs)
    if parsed is None:
        memory.pop("_tu93_player", None)
        return "RESET", None, None

    allowed = tuple(i for i, name in enumerate(_NAMES) if name in available)
    if not allowed:
        allowed = (0, 1, 2, 3)

    saved = memory.get("_tu93_player")
    route = []

    if (saved is not None
            and saved["model"]["signature"] == parsed["signature"]
            and _matches(saved["model"], saved["expected"], parsed)):
        model = saved["model"]
        state = saved["expected"]
        route = saved["route"]
    else:
        model, state = _new_model(parsed)

    # Retain a known exit when an enemy temporarily covers it.
    if parsed["goals"]:
        goals = parsed["goals"]
        guessing = False
    elif model["guess"]:
        goals = _goal_guesses(parsed)
        guessing = True
    else:
        goals = model["goals"]
        guessing = False

    if goals != model["goals"]:
        model["goals"] = goals
        model["distances"] = _distances(model["graph"], goals)
        route = []
    model["guess"] = guessing

    if route and route[0] not in allowed:
        route = []

    nxt = _advance(model, state, route[0]) if route else None
    if route and nxt is None:
        route = []

    if not route:
        route, complete = _search(model, state, allowed)
        if not route:
            # A perturbed start may already be irreversibly trapped.
            memory.pop("_tu93_player", None)
            return "RESET", None, None
        nxt = _advance(model, state, route[0])

    if nxt is None:
        memory.pop("_tu93_player", None)
        return "RESET", None, None

    action = route[0]
    memory["_tu93_player"] = {
        "model": model,
        "expected": nxt,
        "route": route[1:],
    }
    return _NAMES[action], None, None

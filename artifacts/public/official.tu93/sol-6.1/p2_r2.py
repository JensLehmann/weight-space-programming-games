import heapq
import time
from collections import deque
import numpy as np


# Directions use the game's action numbers.
_DELTA = {1: -128, 2: 128, 3: -1, 4: 1}
_XY = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def _parse(obs):
    yy, xx = np.nonzero(obs[:63] == 9)
    if len(xx) == 0:
        return None

    x0, x1 = int(xx.min()), int(xx.max())
    y0, y1 = int(yy.min()), int(yy.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1

    scale = max(1, int(round(max(width, height) / 3.0)))
    if width != 3 * scale or height != 3 * scale:
        return None

    cx = x0 + width // 2
    cy = y0 + height // 2
    step = 6 * scale
    rx, ry = cx % step, cy % step
    geom = (rx, ry, step, scale)

    def position(x, y):
        return ((y - ry) // step) * 128 + (x - rx) // step

    player = position(cx, cy)
    graph = {}
    goals = set()
    entities = []

    def color(x, y):
        if 0 <= x < 64 and 0 <= y < 63:
            return int(obs[y, x])
        return 5

    for y in range(ry, 63, step):
        for x in range(rx, 64, step):
            p = position(x, y)
            c = color(x, y)

            if c == 14:
                goals.add(p)
            elif c in (8, 12, 13):
                direction = 1
                active = False
                for a in (1, 2, 3, 4):
                    dx, dy = _XY[a]
                    mark = color(x + dx * scale, y + dy * scale)
                    if mark in (11, 15):
                        direction = a
                        active = mark == 11
                        break
                entities.append((c, p, direction, active))

            # A valid movement is determined by the intervening red block.
            for a in (1, 2, 3, 4):
                dx, dy = _XY[a]
                mx = x + dx * 3 * scale
                my = y + dy * 3 * scale
                nx = x + dx * step
                ny = y + dy * step
                if (0 <= nx < 64 and 0 <= ny < 63
                        and color(mx, my) == 2):
                    graph.setdefault(p, {})[a] = p + _DELTA[a]

    for adj in list(graph.values()):
        for p in adj.values():
            graph.setdefault(p, {})
    for p in goals:
        graph.setdefault(p, {})

    signature = (
        geom,
        tuple(sorted((p, tuple(sorted(adj.items())))
                     for p, adj in graph.items()))
    )
    return {
        "graph": graph,
        "goals": goals,
        "player": player,
        "entities": entities,
        "geom": geom,
        "signature": signature,
    }


def _pixel_color(obs, geom, p):
    rx, ry, step, _ = geom
    y, x = divmod(p, 128)
    px, py = rx + x * step, ry + y * step
    if 0 <= px < 64 and 0 <= py < 63:
        return int(obs[py, px])
    return None


def _turret_maps(world):
    kill = {}
    danger = {}
    for i, (p, direction) in enumerate(world["turrets"]):
        bit = 1 << i
        kill[p] = kill.get(p, 0) | bit
        attack = p + _DELTA[direction]
        danger[attack] = danger.get(attack, 0) | bit
    world["kill"] = kill
    world["danger"] = danger


def _initialize(parsed, previous_state=None):
    turrets = []
    patrols = []
    mimics = []

    for kind, p, direction, active in sorted(parsed["entities"]):
        if kind == 8:
            turrets.append((p, direction))
        elif kind == 12:
            patrols.append((p, direction))
        else:
            # -1 means dormant; 0 means an unknown pending direction.
            pending = 0 if active else -1
            if active and previous_state is not None:
                candidates = set()
                for group in previous_state[3]:
                    for ep, ed, eq in group:
                        if ep == p and ed == direction and eq >= 0:
                            candidates.add(eq)
                if len(candidates) == 1:
                    pending = next(iter(candidates))
            mimics.append(((p, direction, pending),))

    world = {
        "graph": parsed["graph"],
        "goals": set(parsed["goals"]),
        "geom": parsed["geom"],
        "signature": parsed["signature"],
        "turrets": tuple(turrets),
    }
    _turret_maps(world)
    state = (
        parsed["player"],
        (1 << len(turrets)) - 1,
        tuple(patrols),
        tuple(mimics),
    )
    return world, state


def _reconcile(obs, parsed, world, state):
    """Narrow hidden-state uncertainty and discover previously occluded enemies."""
    p, mask, patrols, mimics = state
    if p != parsed["player"]:
        return None

    seen = parsed["entities"]
    geom = world["geom"]
    occupied_colors = (8, 9, 12, 13)

    for i, (ep, ed) in enumerate(world["turrets"]):
        if mask & (1 << i):
            c = _pixel_color(obs, geom, ep)
            if c is not None and c not in occupied_colors:
                return None

    for item in patrols:
        if item is None:
            continue
        ep, ed = item
        c = _pixel_color(obs, geom, ep)
        if c is not None and c not in occupied_colors:
            return None
        if c == 12:
            # A different patrol can obscure this patrol, but its direction
            # must be accounted for by some predicted patrol at this position.
            actual = [(sd, sa) for sk, sp, sd, sa in seen
                      if sk == 12 and sp == ep]
            if actual and not any(
                    other is not None and other[0] == ep
                    and other[1] == actual[0][0]
                    for other in patrols):
                return None

    narrowed = []
    for index, group in enumerate(mimics):
        keep = []
        for ep, ed, eq in group:
            c = _pixel_color(obs, geom, ep)
            if c is None or c in (8, 9, 12):
                keep.append((ep, ed, eq))
                continue
            if c != 13:
                continue

            actual = [(sd, sa) for sk, sp, sd, sa in seen
                      if sk == 13 and sp == ep]
            matches = any(sd == ed and sa == (eq >= 0)
                          for sd, sa in actual)
            if matches:
                keep.append((ep, ed, eq))
                continue

            # Another mimic might be drawn over this one.
            obscured = False
            for j, other_group in enumerate(mimics):
                if j == index:
                    continue
                if any(op == ep and any(
                        sd == od and sa == (oq >= 0)
                        for sd, sa in actual)
                       for op, od, oq in other_group):
                    obscured = True
                    break
            if obscured:
                keep.append((ep, ed, eq))
        narrowed.append(tuple(sorted(set(keep))))

    mimics = tuple(narrowed)
    known = set()
    for i, (ep, ed) in enumerate(world["turrets"]):
        if mask & (1 << i):
            known.add((8, ep, ed, None))
    for item in patrols:
        if item is not None:
            known.add((12, item[0], item[1], None))
    for group in mimics:
        for ep, ed, eq in group:
            known.add((13, ep, ed, eq >= 0))

    added = False
    turret_list = list(world["turrets"])
    patrol_list = list(patrols)
    mimic_list = list(mimics)

    for kind, ep, ed, active in seen:
        key = (kind, ep, ed, active if kind == 13 else None)
        if key in known:
            continue
        added = True
        known.add(key)
        if kind == 8:
            mask |= 1 << len(turret_list)
            turret_list.append((ep, ed))
        elif kind == 12:
            patrol_list.append((ep, ed))
        else:
            mimic_list.append(((ep, ed, 0 if active else -1),))

    if added:
        world["turrets"] = tuple(turret_list)
        _turret_maps(world)

    new_state = (p, mask, tuple(patrol_list), tuple(mimic_list))
    return new_state, added


def _advance(world, state, action):
    p, mask, patrols, mimics = state
    target = world["graph"].get(p, {}).get(action)
    if target is None:
        return None

    # The player first destroys every enemy occupying its destination.
    mask &= ~world["kill"].get(target, 0)

    # Stationary red enemies attack the node immediately in front of them.
    if mask & world["danger"].get(target, 0):
        return None

    new_patrols = []
    graph = world["graph"]
    for item in patrols:
        if item is None:
            new_patrols.append(None)
            continue
        ep, ed = item
        if ep == target:
            new_patrols.append(None)
            continue

        ep += _DELTA[ed]
        if ep == target:
            return None
        if ed not in graph.get(ep, {}):
            ed = _OPPOSITE[ed]
        new_patrols.append((ep, ed))

    new_mimics = []
    for group in mimics:
        possibilities = set()
        for ep, ed, pending in group:
            if ep == target:
                continue

            if pending < 0:
                if target == ep + 2 * _DELTA[ed]:
                    # Activation seeds the two-step direction delay.
                    possibilities.add((ep, ed, ed))
                else:
                    possibilities.add((ep, ed, -1))
            else:
                ep += _DELTA[ed]
                if ep == target:
                    return None
                if pending == 0:
                    # At an arbitrary episode start the hidden queued
                    # direction may be any of the four directions.
                    for next_direction in (1, 2, 3, 4):
                        possibilities.add((ep, next_direction, action))
                else:
                    possibilities.add((ep, pending, action))
        new_mimics.append(tuple(sorted(possibilities)))

    return target, mask, tuple(new_patrols), tuple(new_mimics)


def _distances(world):
    graph = world["graph"]
    distances = {}
    queue = deque()
    for p in world["goals"]:
        distances[p] = 0
        queue.append(p)

    reverse = {}
    for p, adj in graph.items():
        for target in adj.values():
            reverse.setdefault(target, []).append(p)

    while queue:
        p = queue.popleft()
        for previous in reverse.get(p, ()):
            if previous not in distances:
                distances[previous] = distances[p] + 1
                queue.append(previous)
    return distances


def _recover(parent, state):
    result = []
    while parent[state] is not None:
        previous, action = parent[state]
        result.append(action)
        state = previous
    result.reverse()
    return result


def _search(world, start, allowed):
    heuristic = _distances(world)
    if start[0] not in heuristic:
        return [], False
    if start[0] in world["goals"]:
        return [], True

    deadline = time.monotonic() + 7.5
    max_depth = min(120, 3 * len(world["graph"]) + 10)

    distance = {start: 0}
    parent = {start: None}
    counter = 0
    h0 = heuristic[start[0]]
    heap = [(h0, h0, 0, counter, start)]
    best = start
    best_key = (h0, 0)
    expansions = 0
    limited = False

    while heap:
        _, h, g, _, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        if state[0] in world["goals"]:
            return _recover(parent, state), True

        expansions += 1
        if expansions % 128 == 0:
            if (time.monotonic() >= deadline or expansions >= 180000
                    or len(distance) >= 350000):
                limited = True
                break

        if g >= max_depth:
            continue

        options = world["graph"].get(state[0], {})
        actions = [a for a in options if a in allowed]
        actions.sort(key=lambda a: heuristic.get(options[a], 10000))

        for action in actions:
            next_state = _advance(world, state, action)
            if next_state is None:
                continue
            nh = heuristic.get(next_state[0])
            if nh is None:
                continue
            ng = g + 1
            if ng >= distance.get(next_state, 10 ** 9):
                continue

            distance[next_state] = ng
            parent[next_state] = (state, action)
            key = (nh, ng)
            if key < best_key:
                best_key = key
                best = next_state

            if next_state[0] in world["goals"]:
                return _recover(parent, next_state), True

            counter += 1
            heapq.heappush(
                heap, (ng + nh, nh, ng, counter, next_state)
            )

    if limited and best != start:
        # Execute only a short prefix, then search again.
        return _recover(parent, best)[:3], False
    return [], False


def _reset_or_fallback(available, memory, world=None, state=None):
    if "RESET" in available:
        memory.clear()
        return "RESET", None, None

    if world is not None and state is not None:
        candidates = []
        visited = memory.setdefault("visited", set())
        distances = _distances(world)
        for action in (1, 2, 3, 4):
            if "ACTION%d" % action not in available:
                continue
            next_state = _advance(world, state, action)
            if next_state is not None:
                target = next_state[0]
                candidates.append((
                    target in visited,
                    distances.get(target, 10000),
                    action,
                    next_state
                ))
        if candidates:
            _, _, action, next_state = min(candidates, key=lambda x: x[:3])
            memory["state"] = next_state
            memory["plan"] = []
            return "ACTION%d" % action, None, None

    for name in available:
        if name.startswith("ACTION") and name != "ACTION6":
            return name, None, None
    return "RESET", None, None


def act(obs, available, memory):
    available = set(available)
    allowed = {a for a in (1, 2, 3, 4)
               if "ACTION%d" % a in available}

    parsed = _parse(obs)
    if parsed is None:
        return _reset_or_fallback(available, memory)

    old_world = memory.get("world")
    old_state = memory.get("state")
    plan = list(memory.get("plan", ()))

    if (old_world is not None and old_state is not None
            and old_world["signature"] == parsed["signature"]):
        world = old_world
        previous_goals = set(world["goals"])
        world["goals"].update(parsed["goals"])
        reconciled = _reconcile(obs, parsed, world, old_state)
        if reconciled is None:
            world, state = _initialize(parsed, old_state)
            world["goals"].update(previous_goals)
            plan = []
        else:
            state, added = reconciled
            if added or world["goals"] != previous_goals:
                plan = []
    else:
        world, state = _initialize(parsed)
        plan = []

    memory["world"] = world
    memory["state"] = state
    memory.setdefault("visited", set()).add(state[0])

    if not world["goals"]:
        # A goal may be hidden under a sprite in a perturbed start.
        return _reset_or_fallback(available, memory, world, state)

    if plan:
        first = plan[0]
        if first not in allowed or _advance(world, state, first) is None:
            plan = []

    if not plan:
        plan, solved = _search(world, state, allowed)
        if not solved:
            plan = plan[:3]

    if not plan:
        return _reset_or_fallback(available, memory, world, state)

    action = plan.pop(0)
    next_state = _advance(world, state, action)
    if next_state is None:
        return _reset_or_fallback(available, memory, world, state)

    memory["state"] = next_state
    memory["plan"] = plan
    return "ACTION%d" % action, None, None

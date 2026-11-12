import numpy as np
from collections import deque
from itertools import permutations
import time


# Actions: up, down, left, right.
_DX = (0, 0, -1, 1)
_DY = (-1, 1, 0, 0)
_ENEMY_ACTION = (1, 3, 0, 2)  # down, right, up, left


def _square_actors(a, color):
    out = []
    for y, x in zip(*np.where(a == 5)):
        if y < 8 or y > 60 or x < 2 or x > 61:
            continue
        z = a[y-2:y+3, x-2:x+3]
        if z.shape == (5, 5) and np.count_nonzero(z == color) == 24:
            out.append((int(x), int(y)))
    return out


def _gate_templates():
    result = []
    for c in (8, 11, 15):
        for prongs in (True, False):
            b = np.full((7, 7), 5, dtype=np.int8)
            b[1:6, 1:6] = c
            b[6, 3] = c
            if prongs:
                b[1, 2] = 5
                b[1, 4] = 5
            for r in range(4):
                result.append((c, r, np.rot90(b, -r)))
    return result


_GATE_TEMPLATES = _gate_templates()


def _goal_templates():
    b = np.full((7, 7), 5, dtype=np.int8)
    b[0, :] = 9
    b[6, :] = 9
    b[:, 6] = 9
    b[3, 3] = 9
    return [np.rot90(b, -r) for r in range(4)]


_GOAL_TEMPLATES = _goal_templates()


def _patch(a, x, y, radius):
    if (x-radius < 0 or y-radius < 0 or
            x+radius >= 64 or y+radius >= 63):
        return None
    return a[y-radius:y+radius+1, x-radius:x+radius+1]


def _parse(a):
    players = _square_actors(a, 9)
    if not players:
        return None
    start_xy = players[0]
    ox, oy = start_xy[0] % 6, start_xy[1] % 6

    lattice = [(x, y)
               for y in range(oy, 62, 6) if y >= 7
               for x in range(ox, 62, 6) if 2 <= x <= 61]

    gates_xy = []
    buttons_xy = []
    portals_xy = []
    goal_xy = None

    for x, y in lattice:
        p = _patch(a, x, y, 3)
        if p is None:
            continue

        found = False
        for c, r, template in _GATE_TEMPLATES:
            if np.array_equal(p, template):
                gates_xy.append((x, y, c, r))
                found = True
                break

        if a[y, x] == 9:
            for template in _GOAL_TEMPLATES:
                if np.array_equal(p, template):
                    goal_xy = (x, y)
                    break

        if (np.all(p[0, :] == 15) and np.all(p[-1, :] == 15)
                and np.all(p[:, 0] == 15) and np.all(p[:, -1] == 15)):
            portals_xy.append((x, y))

        if not found:
            q = a[y-1:y+2, x-1:x+2]
            c = int(a[y, x])
            if c in (8, 11, 15) and np.all(q == c):
                buttons_xy.append((x, y, c))

    if goal_xy is None:
        # A conservative fallback for partially covered goal artwork.
        candidates = []
        for x, y in lattice:
            p = _patch(a, x, y, 3)
            if p is not None and a[y, x] == 9:
                n = int(np.count_nonzero(p == 9))
                if 14 <= n <= 23:
                    candidates.append((n, (x, y)))
        if candidates:
            goal_xy = max(candidates)[1]

    # Thin decorative wires are not walkable. Real floor covers almost
    # the whole central 5x5 region of a lattice cell.
    floor = set()
    for x, y in lattice:
        q = _patch(a, x, y, 2)
        if q is not None and np.count_nonzero(q != 0) >= 23:
            floor.add((x, y))
    floor.add(start_xy)
    if goal_xy is not None:
        floor.add(goal_xy)
    for x, y, c, r in gates_xy:
        floor.add((x, y))
    for x, y, c in buttons_xy:
        floor.add((x, y))
    floor.update(portals_xy)

    coords = sorted(floor, key=lambda p: (p[1], p[0]))
    index = {p: i for i, p in enumerate(coords)}
    n = len(coords)
    adj = []
    for x, y in coords:
        adj.append(tuple(index.get((x + 6*dx, y + 6*dy), -1)
                         for dx, dy in zip(_DX, _DY)))

    # Connected colored wiring. Small endpoint gaps are bridged only
    # along rows or columns.
    parent = list(range(4096))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    for y in range(7, 63):
        for x in range(64):
            c = int(a[y, x])
            if c not in (8, 11, 15):
                continue
            for dx, dy in ((1, 0), (0, 1)):
                for d in range(1, 5):
                    xx, yy = x + dx*d, y + dy*d
                    if xx >= 64 or yy >= 63:
                        break
                    v = int(a[yy, xx])
                    if v == c:
                        union(y*64+x, yy*64+xx)
                        break
                    if v not in (0, 5):
                        break

    def component(x, y):
        return find(y*64+x)

    gates = []
    gate_groups = {}
    closed_bits = [0] * n
    open_bits = [0] * n
    toggle_bits = 0
    for i, (x, y, c, r) in enumerate(gates_xy):
        closed = index.get((x, y), -1)
        dx, dy = ((0, 1), (-1, 0), (0, -1), (1, 0))[r]
        opened = index.get((x + 6*dx, y + 6*dy), -1)
        gates.append((closed, opened, c, r, x, y))
        bit = 1 << i
        if closed >= 0:
            closed_bits[closed] |= bit
        if opened >= 0:
            open_bits[opened] |= bit
        if c == 11:
            toggle_bits |= bit
        gate_groups.setdefault(component(x, y), []).append(i)

    portal_groups = {}
    for x, y in portals_xy:
        portal_groups.setdefault(component(x, y-3), []).append(index[(x, y)])

    pairs_by_group = {}
    for group, nodes in portal_groups.items():
        nodes = list(nodes)
        pairs = []
        while len(nodes) >= 2:
            best = None
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    p, q = coords[nodes[i]], coords[nodes[j]]
                    distance = abs(p[0]-q[0]) + abs(p[1]-q[1])
                    aligned = p[0] == q[0] or p[1] == q[1]
                    score = distance + (0 if aligned else 100)
                    if best is None or score < best[0]:
                        best = (score, i, j)
            _, i, j = best
            pairs.append((nodes[i], nodes[j]))
            nodes.pop(j)
            nodes.pop(i)
        pairs_by_group[group] = tuple(pairs)

    buttons = []
    button_at = [-1] * n
    for x, y, c in buttons_xy:
        node = index[(x, y)]
        group = component(x, y)
        bits = 0
        for g in gate_groups.get(group, ()):
            bits |= 1 << g
        normal = bits & ~toggle_bits
        toggle = bits & toggle_bits
        pairs = pairs_by_group.get(group, ())
        button_at[node] = len(buttons)
        buttons.append((node, normal, toggle, pairs, c))

    enemy_xy = _square_actors(a, 14)
    enemies = tuple(index[p] * 4 for p in enemy_xy if p in index)
    start = index[start_xy]
    return {
        "coords": coords,
        "index": index,
        "adj": adj,
        "start": start,
        "goal": index.get(goal_xy, -1),
        "gates": gates,
        "closed": closed_bits,
        "opened": open_bits,
        "buttons": buttons,
        "button_at": button_at,
        "enemies": enemies,
        "slots": max(1, sum(
            int(a[2, x]) in (0, 1) and int(a[1, x]) in (1, 2, 9)
            for x in range(2, 31, 4)
        )),
    }


def _initial_state(m, scripts, ending=None):
    # Enemy headings survive a rewind, although their positions rewind.
    ep = []
    for i, original in enumerate(m["enemies"]):
        d = original % 4
        if ending is not None and i < len(ending[4]):
            e = ending[4][i]
            e = e if e >= 0 else -e-1
            d = e % 4
        ep.append((original // 4)*4 + d)
    return (m["start"], 0, 0,
            tuple(m["start"] for _ in scripts), tuple(ep))


def _enemy_position(e):
    return (e if e >= 0 else -e-1) // 4


def _transition(m, scripts, s, action, absolute_time=False):
    p, mask, t, ghosts, enemies = s
    adj = m["adj"]
    closed = m["closed"]
    opened = m["opened"]

    def blocked(node):
        return node < 0 or bool((closed[node] & ~mask) |
                                (opened[node] & mask))

    np_ = adj[p][action]
    if blocked(np_):
        return None

    old = [p] + list(ghosts) + [_enemy_position(e) for e in enemies]
    new = [np_]
    moved = [True]

    for i, g in enumerate(ghosts):
        dest = g
        did = False
        if i < len(scripts) and t < len(scripts[i]):
            q = adj[g][scripts[i][t]]
            if not blocked(q):
                dest, did = q, True
        new.append(dest)
        moved.append(did)

    new_enemies = []
    ng = len(ghosts)
    for e in enemies:
        pos = _enemy_position(e)
        if e < 0:
            new_enemies.append(e)
            new.append(pos)
            moved.append(False)
            continue
        direction = e % 4
        dest = pos
        did = False
        for attempt in range(4):
            selected = None
            for d in (direction, (direction-1) % 4,
                      (direction+1) % 4, (direction+2) % 4):
                q = adj[pos][_ENEMY_ACTION[d]]
                if q >= 0:
                    selected = (q, d)
                    break
            if selected is None:
                # A stationary valid enemy move can still pulse a switch.
                did = True
                break
            q, direction = selected
            if not blocked(q):
                dest, did = q, True
                break
            direction = (direction + 2) % 4
        new_enemies.append(dest*4 + direction)
        new.append(dest)
        moved.append(did)

    ba = m["button_at"]
    buttons = m["buttons"]
    counts = [0] * len(buttons)
    for node in old:
        b = ba[node]
        if b >= 0:
            counts[b] += 1

    active_pairs = []

    def emit(b, value, current_mask):
        _, normal, toggle, pairs, _ = buttons[b]
        if value:
            current_mask = (current_mask | normal) ^ toggle
            active_pairs.extend(pairs)
        else:
            current_mask &= ~normal
        return current_mask

    # Departures happen before movement animation and new contacts.
    departure_mask = mask
    for i, did in enumerate(moved):
        if did:
            b = ba[old[i]]
            if b >= 0:
                counts[b] -= 1
                if counts[b] == 0:
                    departure_mask = emit(b, False, departure_mask)

    def close_casualties(before, after):
        falling = before & ~after
        if not falling:
            return False
        for gi, g in enumerate(m["gates"]):
            if falling & (1 << gi):
                at = g[0]
                if new[0] == at:
                    return True
                for ei in range(len(new_enemies)):
                    if new_enemies[ei] >= 0 and new[1+ng+ei] == at:
                        new_enemies[ei] = -new_enemies[ei]-1
        return False

    if close_casualties(mask, departure_mask):
        return None

    # Enemy contact occurs before any delayed teleport.
    for ei, e in enumerate(new_enemies):
        if e >= 0 and new[1+ng+ei] == new[0]:
            return None

    arrival_mask = departure_mask
    arrival_order = ([0] +
                     list(range(1+ng, len(new))) +
                     list(range(1, 1+ng)))
    for i in arrival_order:
        if not moved[i]:
            continue
        b = ba[new[i]]
        if b >= 0:
            was_empty = counts[b] == 0
            counts[b] += 1
            if was_empty:
                arrival_mask = emit(b, True, arrival_mask)

    if close_casualties(departure_mask, arrival_mask):
        return None

    for u, v in active_pairs:
        for i, node in enumerate(new):
            if node == u:
                new[i] = v
            elif node == v:
                new[i] = u

    for ei, e in enumerate(new_enemies):
        packed = e if e >= 0 else -e-1
        packed = new[1+ng+ei]*4 + packed % 4
        new_enemies[ei] = packed if e >= 0 else -packed-1

    nt = t + 1
    if not absolute_time:
        nt = min(nt, max((len(x) for x in scripts), default=0))
    return (new[0], arrival_mask, nt,
            tuple(new[1:1+ng]), tuple(new_enemies))


def _unroll(states, parents, actions, i):
    path = []
    while parents[i] >= 0:
        path.append(actions[i])
        i = parents[i]
    path.reverse()
    return path


def _bfs(m, scripts, initial, limit, deadline, forbidden_targets=()):
    """Find an exit route and useful shortest recording endpoints."""
    states = [initial]
    parents = [-1]
    actions = [-1]
    depths = [0]
    seen = {initial}
    q = deque([0])
    options = []
    option_masks = {}
    forbidden_targets = set(forbidden_targets)
    goal = m["goal"]
    fallback = None
    best_distance = 10**9
    gx, gy = m["coords"][goal] if goal >= 0 else (32, 32)

    while q:
        i = q.popleft()
        s = states[i]
        p = s[0]
        depth = depths[i]
        if i % 128 == 0 and time.perf_counter() > deadline:
            break

        x, y = m["coords"][p]
        dist = abs(x-gx) + abs(y-gy)
        if depth and dist < best_distance:
            best_distance = dist
            fallback = i

        if p == goal:
            return (_unroll(states, parents, actions, i), s), options, (
                _unroll(states, parents, actions, fallback)
                if fallback is not None else [])

        b = m["button_at"][p]
        if b >= 0 and p not in forbidden_targets:
            old_masks = option_masks.setdefault(p, set())
            # A second useful state is sometimes needed after a toggle.
            if s[1] not in old_masks and len(old_masks) < 2:
                old_masks.add(s[1])
                path = _unroll(states, parents, actions, i)
                options.append((p, path, s, m["buttons"][b][4]))

        if depth >= limit or len(states) >= 45000:
            continue

        for a in range(4):
            ns = _transition(m, scripts, s, a)
            if ns is not None and ns not in seen:
                seen.add(ns)
                states.append(ns)
                parents.append(i)
                actions.append(a)
                depths.append(depth+1)
                q.append(len(states)-1)

    options.sort(key=lambda z: (
        0 if z[3] == 8 else 1 if z[3] == 15 else 2,
        len(z[1])))
    return None, options, (
        _unroll(states, parents, actions, fallback)
        if fallback is not None else [])


def _delayed_options(m, scripts, initial, target, earliest, limit, deadline):
    """Routes whose first arrival at a teleport switch is delayed."""
    if initial[0] == target:
        return []
    maximum = min(limit, earliest + 18)
    states = [initial]
    parents = [-1]
    actions = [-1]
    depths = [0]
    seen = {initial}
    q = deque([0])
    found_depths = set()
    results = []
    color = m["buttons"][m["button_at"][target]][4]

    while q:
        i = q.popleft()
        if i % 128 == 0 and time.perf_counter() > deadline:
            break
        s, d = states[i], depths[i]
        if s[0] == target:
            if d > earliest and d not in found_depths:
                found_depths.add(d)
                results.append((target,
                                _unroll(states, parents, actions, i),
                                s, color))
                if len(results) >= 9:
                    break
            continue
        if d >= maximum or len(states) >= 25000:
            continue
        for a in range(4):
            ns = _transition(m, scripts, s, a, True)
            if ns is not None and ns not in seen:
                seen.add(ns)
                states.append(ns)
                parents.append(i)
                actions.append(a)
                depths.append(d+1)
                q.append(len(states)-1)
    return results


def _plan(m, scripts, state, history, used_targets, budget, deadline):
    cache = {}
    delay_cache = {}
    root_options = []
    root_fallback = []

    def inspect(ss, st, remaining, targets):
        key = (tuple(tuple(x) for x in ss), st, remaining, tuple(targets))
        if key not in cache:
            cache[key] = _bfs(m, ss, st, remaining, deadline, targets)
        return cache[key]

    def solve(ss, st, prefix, targets, remaining, extra, root=False):
        nonlocal root_options, root_fallback
        if time.perf_counter() > deadline or remaining < 0:
            return None
        goal, options, fallback = inspect(ss, st, remaining, targets)
        if root:
            root_options, root_fallback = options, fallback
        if goal is not None:
            return goal[0]
        if extra <= 0 or len(ss) + 1 >= m["slots"]:
            return None

        options = [o for o in options if prefix or o[1]]
        for option in options:
            if time.perf_counter() > deadline:
                return None
            target, route, end_state, color = option
            if len(route) + 1 >= remaining:
                continue

            candidates = [option]
            if color == 15:
                key = (tuple(tuple(x) for x in ss), st, target, remaining)
                if key not in delay_cache:
                    delay_cache[key] = _delayed_options(
                        m, ss, st, target, len(route),
                        min(remaining-2, 60), deadline)
                candidates += delay_cache[key]

            for target, route, end_state, color in candidates:
                if time.perf_counter() > deadline:
                    return None
                new_script = list(prefix) + route
                if not new_script:
                    continue
                new_scripts = list(ss) + [new_script]
                ns = _initial_state(m, new_scripts, end_state)
                rest = solve(new_scripts, ns, [],
                             list(targets) + [target],
                             remaining-len(route)-1, extra-1)
                if rest is not None:
                    return route + [4] + rest
        return None

    max_extra = max(0, m["slots"] - len(scripts) - 1)
    max_extra = min(max_extra, 3)
    for extra in range(max_extra + 1):
        result = solve(scripts, state, history, used_targets,
                       budget, extra, True)
        if result is not None:
            return result
        if time.perf_counter() > deadline:
            break

    # If exhaustive planning ran out of time, commit one useful recording.
    # The next call after its rewind gets a fresh planning budget.
    if len(scripts) + 1 < m["slots"]:
        for target, route, end_state, color in root_options:
            if (history or route) and target not in used_targets:
                return route + [4]
    return root_fallback[:12]


def _read_gate_mask(a, m, predicted):
    mask = predicted
    for i, (_, _, c, r, x, y) in enumerate(m["gates"]):
        dx, dy = ((0, 1), (-1, 0), (0, -1), (1, 0))[r]
        matches = []
        for is_open, xx, yy in (
                (False, x, y), (True, x+6*dx, y+6*dy)):
            p = _patch(a, xx, yy, 3)
            if p is None:
                continue
            for tc, tr, template in _GATE_TEMPLATES:
                if tc == c and tr == r and np.array_equal(p, template):
                    matches.append(is_open)
                    break
        if len(matches) == 1:
            if matches[0]:
                mask |= 1 << i
            else:
                mask &= ~(1 << i)
    return mask


def _assign_positions(predicted, observed, coords):
    if len(predicted) != len(observed) or len(observed) > 5:
        return tuple(predicted)
    if set(predicted) == set(observed):
        return tuple(predicted)
    best, best_cost = None, 10**9
    for order in permutations(observed):
        cost = 0
        for p, q in zip(predicted, order):
            x, y = coords[p]
            xx, yy = coords[q]
            cost += abs(x-xx) + abs(y-yy)
        if cost < best_cost:
            best, best_cost = order, cost
    return tuple(best) if best is not None else tuple(predicted)


def _reconcile(a, memory):
    m = memory["model"]
    s = memory["state"]
    p, mask, t, ghosts, enemies = s
    observed = _square_actors(a, 9)
    if not observed or observed[0] not in m["index"]:
        return False

    actual_p = m["index"][observed[0]]
    mask = _read_gate_mask(a, m, mask)
    previous = memory.get("previous")
    last = memory.get("last", -1)

    if (previous is not None and 0 <= last < 4
            and actual_p == previous[0] and actual_p != p):
        # The proposed step did not advance time. Usually this reveals
        # a decorative cell that was mistakenly classified as floor.
        if memory["history"]:
            memory["history"].pop()
        source = previous[0]
        destination = m["adj"][source][last]
        is_gate = destination >= 0 and (
            m["closed"][destination] or m["opened"][destination])
        if destination >= 0 and not is_gate:
            row = list(m["adj"][source])
            row[last] = -1
            m["adj"][source] = tuple(row)
        p, _, t, ghosts, enemies = previous

    p = actual_p

    visible_ghosts = [m["index"][q] for q in _square_actors(a, 2)
                      if q in m["index"]]
    ghosts = _assign_positions(ghosts, visible_ghosts, m["coords"])

    visible_enemies = [m["index"][q] for q in _square_actors(a, 14)
                       if q in m["index"]]
    enemy_positions = tuple(_enemy_position(e) for e in enemies)
    assigned = _assign_positions(enemy_positions, visible_enemies, m["coords"])
    new_enemies = []
    for i, (e, pos) in enumerate(zip(enemies, assigned)):
        packed = e if e >= 0 else -e-1
        direction = packed % 4
        if pos != _enemy_position(e) and previous is not None:
            old_pos = _enemy_position(previous[4][i])
            x, y = m["coords"][old_pos]
            xx, yy = m["coords"][pos]
            if xx == x and yy > y:
                direction = 0
            elif xx > x and yy == y:
                direction = 1
            elif xx == x and yy < y:
                direction = 2
            elif xx < x and yy == y:
                direction = 3
        value = pos*4 + direction
        new_enemies.append(value if e >= 0 else -value-1)

    ns = (p, mask, t, ghosts, tuple(new_enemies))
    changed = ns != memory["state"]
    memory["state"] = ns
    if changed:
        memory["plan"] = []
    return True


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int8)

    # Make RESET unambiguously a level reset, including when evaluation
    # starts before the first ordinary action of a non-first level.
    if not memory:
        memory["boot"] = 1
        return ("ACTION5", None, None)
    if memory.get("boot") == 1:
        memory["boot"] = 2
        return ("RESET", None, None)

    if memory.get("boot") == 2:
        model = _parse(a)
        if model is None:
            return ("ACTION5", None, None)

        # Read the actual number of timeline slots from the HUD.
        slots = 0
        for x in range(2, 63, 4):
            p = a[1:4, x-1:x+2]
            if p.shape != (3, 3):
                break
            if np.all(p == 1):
                slots += 1
            elif (int(a[2, x]) == 0 and
                  np.count_nonzero((p == 9) | (p == 2)) >= 8):
                slots += 1
            else:
                break
        model["slots"] = max(1, slots)

        memory["model"] = model
        memory["scripts"] = []
        memory["targets"] = []
        memory["history"] = []
        memory["state"] = _initial_state(model, [])
        memory["plan"] = []
        memory["spent"] = 0
        memory["boot"] = 3
    else:
        if not _reconcile(a, memory):
            # Recovery from a lost animation or an unexpected death.
            memory["boot"] = 2
            return ("RESET", None, None)

    m = memory["model"]
    scripts = memory["scripts"]
    state = memory["state"]

    if not memory["plan"]:
        deadline = time.perf_counter() + 7.7
        budget = max(1, 127 - memory["spent"])
        memory["plan"] = _plan(
            m, scripts, state, memory["history"],
            memory["targets"], budget, deadline)

    if not memory["plan"]:
        # Prefer a legal, reversible movement to an invalid-action loop.
        choices = []
        gx, gy = (m["coords"][m["goal"]]
                  if m["goal"] >= 0 else (32, 32))
        for action in range(4):
            ns = _transition(m, scripts, state, action)
            if ns is not None:
                x, y = m["coords"][ns[0]]
                choices.append((abs(x-gx)+abs(y-gy), action))
        if choices:
            choices.sort()
            memory["plan"] = [choices[0][1]]
        elif memory["history"]:
            memory["plan"] = [4]
        else:
            memory["boot"] = 2
            return ("RESET", None, None)

    action = memory["plan"].pop(0)
    memory["previous"] = state
    memory["last"] = action
    memory["spent"] += 1

    if action == 4:
        if memory["history"]:
            new_script = list(memory["history"])
            memory["scripts"].append(new_script)
            memory["targets"].append(state[0])
            if len(memory["scripts"]) >= m["slots"]:
                memory["scripts"] = []
                memory["targets"] = []
            memory["history"] = []
            memory["state"] = _initial_state(
                m, memory["scripts"], state)
        return ("ACTION5", None, None)

    ns = _transition(m, scripts, state, action)
    if ns is not None:
        memory["history"].append(action)
        memory["state"] = ns
    else:
        memory["plan"] = []
    return ("ACTION" + str(action + 1), None, None)

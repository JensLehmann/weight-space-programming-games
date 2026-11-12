from collections import deque
import numpy as np


CENTERS = [(4 + 6 * x, 10 + 6 * y) for y in range(9) for x in range(10)]
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def _neighbor(cell, action):
    x, y = cell % 10, cell // 10
    dx, dy = DIRS[action]
    x += dx
    y += dy
    if 0 <= x < 10 and 0 <= y < 9:
        return y * 10 + x
    return -1


def _find_player(obs):
    for cell, (x, y) in enumerate(CENTERS):
        if x < 2 or x >= 62 or y < 2 or y >= 62:
            continue
        patch = obs[y - 2:y + 3, x - 2:x + 3]
        if patch[2, 2] == 5 and np.count_nonzero(patch == 9) == 24:
            return cell
    return None


def _components(obs):
    labels = {}
    colors = {}
    next_id = 0
    for color in (8, 11, 15):
        seen = set()
        for y in range(7, 63):
            for x in range(64):
                if int(obs[y, x]) != color or (x, y) in seen:
                    continue
                next_id += 1
                colors[next_id] = color
                stack = [(x, y)]
                seen.add((x, y))
                while stack:
                    px, py = stack.pop()
                    labels[(px, py)] = next_id
                    for nx, ny in ((px + 1, py), (px - 1, py),
                                   (px, py + 1), (px, py - 1)):
                        if (0 <= nx < 64 and 7 <= ny < 63
                                and (nx, ny) not in seen
                                and int(obs[ny, nx]) == color):
                            seen.add((nx, ny))
                            stack.append((nx, ny))
    return labels, colors


def _door_patterns():
    blue = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 8, 5, 8, 5, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 5, 5, 8, 5, 5, 5],
    ], dtype=np.int8)
    purple = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 5, 5, 11, 5, 5, 5],
    ], dtype=np.int8)
    result = []
    for color, base in ((8, blue), (11, purple)):
        for rotation in range(4):
            result.append((color, rotation, np.rot90(base, -rotation)))
    return result


DOOR_PATTERNS = _door_patterns()

GOAL_BASE = np.array([
    [-2, -2, -2, -2, -2, -2, -2, -2, -2],
    [-2,  9,  9,  9,  9,  9,  9,  9, -2],
    [-2, -2,  5,  5,  5,  5,  5,  9, -2],
    [-2, -2,  5,  5,  5,  5,  5,  9, -2],
    [-2, -2,  5,  5,  9,  5,  5,  9, -2],
    [-2, -2,  5,  5,  5,  5,  5,  9, -2],
    [-2, -2,  5,  5,  5,  5,  5,  9, -2],
    [-2,  9,  9,  9,  9,  9,  9,  9, -2],
    [-2, -2, -2, -2, -2, -2, -2, -2, -2],
], dtype=np.int8)
GOAL_MASKS = [np.rot90(GOAL_BASE, -r) == 9 for r in range(4)]


def _parse(obs):
    player = _find_player(obs)
    if player is None:
        return None

    labels, component_colors = _components(obs)
    doors_raw = []
    portals = []
    switches_raw = []
    goal = None
    enemies = set()

    for cell, (x, y) in enumerate(CENTERS):
        if not (3 <= x < 61 and 3 <= y < 61):
            continue

        patch7 = obs[y - 3:y + 4, x - 3:x + 4]
        for color, rotation, pattern in DOOR_PATTERNS:
            if np.array_equal(patch7, pattern):
                action = (2, 3, 1, 4)[rotation]
                doors_raw.append((
                    cell, _neighbor(cell, action), color,
                    labels.get((x, y))
                ))
                break

        border = np.concatenate((
            patch7[0, :], patch7[-1, :],
            patch7[1:-1, 0], patch7[1:-1, -1]
        ))
        if np.count_nonzero(border == 15) >= 23:
            ids = [
                labels.get((x - 3, y - 3)),
                labels.get((x + 3, y - 3)),
                labels.get((x - 3, y + 3)),
                labels.get((x + 3, y + 3))
            ]
            ids = [v for v in ids if v is not None]
            portals.append((cell, ids[0] if ids else None))

        patch5 = obs[y - 2:y + 3, x - 2:x + 3]
        if patch5[2, 2] == 5 and np.count_nonzero(patch5 == 14) == 24:
            enemies.add(cell)

        if 4 <= x < 60 and 4 <= y < 60:
            patch9 = obs[y - 4:y + 5, x - 4:x + 5]
            for mask in GOAL_MASKS:
                if np.count_nonzero(patch9[mask] == 9) >= 19:
                    goal = cell
                    break

    door_cells = {d[0] for d in doors_raw}
    portal_cells = {p[0] for p in portals}

    for cell, (x, y) in enumerate(CENTERS):
        if cell in door_cells or cell in portal_cells:
            continue
        if not (2 <= x < 62 and 2 <= y < 62):
            continue
        patch3 = obs[y - 1:y + 2, x - 1:x + 2]
        for color in (8, 11, 15):
            if np.all(patch3 == color):
                switches_raw.append((cell, color, labels.get((x, y))))
                break

    switch_components = {
        comp for _, _, comp in switches_raw if comp is not None
    }

    def nearest_switch_component(cell, color, own):
        if own in switch_components:
            return own
        x, y = CENTERS[cell]
        best = None
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                distance = abs(dx) + abs(dy)
                if distance > 5:
                    continue
                comp = labels.get((x + dx, y + dy))
                if (comp in switch_components
                        and component_colors.get(comp) == color):
                    item = (distance, comp)
                    if best is None or item < best:
                        best = item
        return best[1] if best else own

    switches = {}
    for cell, color, comp in switches_raw:
        if comp is None:
            comp = -(cell + 1)
        switches[cell] = (color, comp)

    doors = []
    for cell, destination, color, comp in doors_raw:
        comp = nearest_switch_component(cell, color, comp)
        if comp is None:
            comp = -(1000 + cell)
        doors.append((cell, destination, color, comp))

    portal_groups = {}
    for cell, comp in portals:
        comp = nearest_switch_component(cell, 15, comp)
        if comp is not None:
            portal_groups.setdefault(comp, []).append(cell)

    floor = set()
    for cell, (x, y) in enumerate(CENTERS):
        if not (2 <= x < 62 and 2 <= y < 62):
            continue
        patch = obs[y - 2:y + 3, x - 2:x + 3]
        black = int(np.count_nonzero(patch == 5))
        center = int(obs[y, x])
        if black >= 15 or (center in (8, 11, 15) and black >= 8):
            floor.add(cell)

    floor.add(player)
    floor.update(door_cells)
    floor.update(portal_cells)
    floor.update(switches)
    floor.update(enemies)
    if goal is not None:
        floor.add(goal)

    toggle_groups = sorted(
        {group for _, _, color, group in doors if color == 11}
        | {group for color, group in switches.values() if color == 11}
    )
    toggle_index = {group: i for i, group in enumerate(toggle_groups)}

    lives = 0
    for i in range(8):
        x = 1 + 4 * i
        if x + 2 >= 64:
            break
        patch = obs[1:4, x:x + 3]
        if np.count_nonzero(np.isin(patch, (1, 2, 9))) >= 7:
            lives += 1
        else:
            break
    lives = max(1, min(lives, 4))

    return {
        "start": player,
        "goal": goal,
        "floor": floor,
        "switches": switches,
        "doors": doors,
        "portals": {
            group: tuple(cells[:2])
            for group, cells in portal_groups.items() if len(cells) >= 2
        },
        "toggle_index": toggle_index,
        "lives": lives,
        "enemies": enemies,
    }


def _tick(world, state, action, scripts, blocked, reverse=False,
          reverse_index=None, fast_reverse=False):
    player, ghost_positions, t, bits = state
    positions = [player] + list(ghost_positions)
    switches = world["switches"]
    doors = world["doors"]
    floor = world["floor"]
    toggle_index = world["toggle_index"]

    occupants = {}
    for actor, cell in enumerate(positions):
        if cell in switches:
            occupants.setdefault(cell, set()).add(actor)

    portal_at = {}
    for group, pair in world["portals"].items():
        for cell in pair:
            portal_at[cell] = group

    portal_occupants = {}
    for actor, cell in enumerate(positions):
        if cell in portal_at:
            portal_occupants.setdefault(cell, set()).add(actor)

    def switch_active(group):
        for cell, members in occupants.items():
            if members and switches[cell][1] == group:
                return True
        return False

    def door_open(color, group):
        if color == 11:
            index = toggle_index.get(group)
            return index is not None and bool(bits & (1 << index))
        return switch_active(group)

    def teleport(group):
        pair = world["portals"].get(group)
        if pair is None:
            return
        a, b = pair
        aa = set(portal_occupants.get(a, ()))
        bb = set(portal_occupants.get(b, ()))
        for actor in aa:
            positions[actor] = b
        for actor in bb:
            positions[actor] = a
        portal_occupants[a] = bb
        portal_occupants[b] = aa

    def switch_event(cell, value):
        nonlocal bits
        color, group = switches[cell]
        if color == 11 and value:
            index = toggle_index.get(group)
            if index is not None:
                bits ^= 1 << index
        elif color == 15:
            if (not reverse or not fast_reverse) and value:
                teleport(group)
            elif reverse and fast_reverse and not value:
                teleport(group)

    initially_open = [
        door_open(color, group) for _, _, color, group in doors
    ]
    blocked_cells = set()
    for is_open, (base, destination, color, group) in zip(
            initially_open, doors):
        blocked_cells.add(destination if is_open and destination >= 0 else base)

    if reverse:
        index = reverse_index
        moves = []
        for i, script in enumerate(scripts):
            if index < len(script):
                moves.append((i + 1, OPPOSITE[script[index]]))
        moves.append((0, OPPOSITE[action]))
    else:
        moves = [(0, action)]
        for i, script in enumerate(scripts):
            if t < len(script):
                moves.append((i + 1, script[t]))

    player_moved = False
    for actor, direction in moves:
        old = positions[actor]
        destination = _neighbor(old, direction)
        allowed = (
            destination in floor
            and destination not in blocked_cells
            and (actor != 0 or (old, direction) not in blocked)
        )
        if not allowed:
            if actor == 0 and not reverse:
                return None
            continue

        if actor == 0:
            player_moved = True

        if old in occupants and actor in occupants[old]:
            occupants[old].remove(actor)
            if not occupants[old]:
                switch_event(old, False)

        if old in portal_occupants:
            portal_occupants[old].discard(actor)

        positions[actor] = destination

    if not reverse and not player_moved:
        return None

    for was_open, (base, destination, color, group) in zip(
            initially_open, doors):
        if was_open and not door_open(color, group) and positions[0] == base:
            return None

    events = []
    for actor in range(len(positions)):
        cell = positions[actor]
        if cell in switches:
            members = occupants.setdefault(cell, set())
            if actor not in members:
                was_empty = not members
                members.add(actor)
                if was_empty:
                    events.append(cell)
        cell = positions[actor]
        if cell in portal_at:
            portal_occupants.setdefault(cell, set()).add(actor)

    for cell in events:
        switch_event(cell, True)

    if reverse:
        new_t = 0
    else:
        maximum = max((len(s) for s in scripts), default=0)
        new_t = min(t + 1, maximum)

    return (positions[0], tuple(positions[1:]), new_t, bits)


def _route(previous, state):
    actions = []
    states = []
    while previous[state] is not None:
        parent, action = previous[state]
        actions.append(action)
        states.append(state)
        state = parent
    actions.reverse()
    states.reverse()
    return actions, states


def _search_phase(world, initial, scripts, blocked, budget):
    queue = deque([initial])
    previous = {initial: None}
    depth = {initial: 0}
    targets = set(world["switches"])
    for pair in world["portals"].values():
        targets.update(pair)

    candidates = []
    candidate_counts = {}
    expanded = 0

    while queue and budget[0] > 0 and expanded < 10000:
        state = queue.popleft()
        d = depth[state]
        expanded += 1
        budget[0] -= 1

        if world["goal"] is not None and state[0] == world["goal"]:
            actions, states = _route(previous, state)
            return (actions, states), []

        if d and state[0] in targets:
            count = candidate_counts.get(state[0], 0)
            if count < 2:
                candidate_counts[state[0]] = count + 1
                candidates.append(state)

        if d >= 70:
            continue

        for action in (1, 2, 3, 4):
            nxt = _tick(world, state, action, scripts, blocked)
            if nxt is None or nxt in previous:
                continue
            previous[nxt] = (state, action)
            depth[nxt] = d + 1
            queue.append(nxt)

    recordings = []
    for state in candidates[:16]:
        actions, states = _route(previous, state)
        if actions:
            recordings.append((actions, states, state))
    return None, recordings


def _rewind(world, end_state, recording, scripts, blocked):
    state = end_state
    for index in range(len(recording) - 1, -1, -1):
        state = _tick(
            world, state, recording[index], scripts, blocked,
            reverse=True, reverse_index=index,
            fast_reverse=(index != len(recording) - 1)
        )
        if state is None:
            return None

    spawn = world["start"]
    ghosts = tuple(state[1]) + (spawn,)
    return (spawn, ghosts, 0, state[3])


def _simulate_recording(world, initial, scripts, blocked, actions):
    state = initial
    states = []
    for action in actions:
        state = _tick(world, state, action, scripts, blocked)
        if state is None:
            return None
        states.append(state)
    return states


def _make_plan(world, blocked):
    if world["goal"] is None:
        return None

    budget = [100000]
    initial = (world["start"], (), 0, 0)
    phases = [(initial, (), [])]
    max_ghosts = max(0, world["lives"] - 1)
    searches = 0

    for ghost_count in range(max_ghosts + 1):
        next_phases = []
        for state, scripts, prefix in phases:
            if budget[0] <= 0 or searches >= 100:
                break

            searches += 1
            solution, recordings = _search_phase(
                world, state, scripts, blocked, budget
            )

            if solution is not None:
                actions, states = solution
                return prefix + [
                    (a, s[0]) for a, s in zip(actions, states)
                ]

            if ghost_count >= max_ghosts:
                continue

            for actions, states, end_state in recordings:
                variants = [(actions, states, end_state)]

                # Two-step excursions delay activation without requiring
                # an invalid move, which would not advance the ghosts.
                for direction in (1, 2, 3, 4):
                    first = _tick(world, state, direction, scripts, blocked)
                    if first is None:
                        continue
                    back = _tick(
                        world, first, OPPOSITE[direction], scripts, blocked
                    )
                    if back is None or back[0] != state[0]:
                        continue

                    for repeats in (1, 2, 3):
                        padded = (
                            [direction, OPPOSITE[direction]] * repeats
                            + list(actions)
                        )
                        padded_states = _simulate_recording(
                            world, state, scripts, blocked, padded
                        )
                        if (padded_states is not None
                                and padded_states[-1][0] == end_state[0]):
                            variants.append((
                                padded, padded_states, padded_states[-1]
                            ))
                    break

                for recording, rec_states, final_state in variants:
                    rewound = _rewind(
                        world, final_state, recording, scripts, blocked
                    )
                    if rewound is None:
                        continue

                    new_prefix = prefix + [
                        (a, s[0]) for a, s in zip(recording, rec_states)
                    ] + [(5, world["start"])]

                    if len(new_prefix) > 130:
                        continue

                    new_scripts = scripts + (tuple(recording),)
                    next_phases.append((rewound, new_scripts, new_prefix))

        if not next_phases:
            break
        next_phases.sort(key=lambda item: len(item[2]))
        phases = next_phases[:70]

    return None


def _fallback_route(world, blocked):
    start, goal = world["start"], world["goal"]
    if goal is None:
        return []

    queue = deque([start])
    parent = {start: None}
    while queue and goal not in parent:
        cell = queue.popleft()
        for action in (1, 2, 3, 4):
            nxt = _neighbor(cell, action)
            if (nxt in world["floor"] and nxt not in parent
                    and (cell, action) not in blocked):
                parent[nxt] = (cell, action)
                queue.append(nxt)

    plan = []
    if goal in parent:
        cell = goal
        while parent[cell] is not None:
            previous, action = parent[cell]
            plan.append((action, cell))
            cell = previous
        plan.reverse()
    return plan


def act(obs, available, memory):
    names = set(available)

    def output(action):
        name = "ACTION" + str(action)
        if name not in names:
            name = next((n for n in available if n.startswith("ACTION")),
                        "RESET")
        return name, None, None

    phase = memory.get("phase")

    if phase is None:
        memory["phase"] = "reset"
        return output(5)

    if phase == "reset":
        memory["phase"] = "parse"
        if "RESET" in names:
            return "RESET", None, None
        return output(5)

    if phase == "play":
        expected = memory.pop("expected", None)
        actual = _find_player(obs)

        if expected is not None and actual is not None and actual != expected:
            failures = memory.get("failures", 0) + 1
            memory["failures"] = failures
            old = memory.get("last_cell")
            last_action = memory.get("last_action")

            if (actual == old and old is not None
                    and last_action in (1, 2, 3, 4)):
                blocked = memory.setdefault("blocked", [])
                edge = [old, last_action]
                if edge not in blocked:
                    blocked.append(edge)

            if failures <= 6 and "RESET" in names:
                memory["phase"] = "parse"
                memory.pop("plan", None)
                return "RESET", None, None

        plan = memory.get("plan", [])
        if plan:
            action, predicted = plan.pop(0)
            memory["last_cell"] = actual
            memory["last_action"] = action
            memory["expected"] = predicted
            return output(action)

        if memory.get("failures", 0) < 6 and "RESET" in names:
            memory["failures"] = memory.get("failures", 0) + 1
            memory["phase"] = "parse"
            return "RESET", None, None
        return output(1)

    world = _parse(obs)
    if world is None:
        return output(1)

    blocked = {tuple(edge) for edge in memory.get("blocked", [])}
    plan = _make_plan(world, blocked)

    if not plan:
        plan = _fallback_route(world, blocked)

    if not plan:
        memory["phase"] = "play"
        memory["plan"] = []
        return output(1)

    memory["phase"] = "play"
    memory["plan"] = [[int(a), int(p)] for a, p in plan]
    action, predicted = memory["plan"].pop(0)
    memory["last_cell"] = world["start"]
    memory["last_action"] = action
    memory["expected"] = predicted
    return output(action)

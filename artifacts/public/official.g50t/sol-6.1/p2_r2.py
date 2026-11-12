import numpy as np
import time
from collections import deque
from itertools import permutations

DELTA = {1: (0, -6), 2: (0, 6), 3: (-6, 0), 4: (6, 0)}
OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}
HEADING_ACTION = (2, 4, 1, 3)


def _matches(image, template):
    h, w = template.shape
    H, W = image.shape
    if h > H or w > W:
        return []
    mask = np.ones((H - h + 1, W - w + 1), dtype=bool)
    for y, x in zip(*np.where(template >= 0)):
        mask &= image[y:y + mask.shape[0], x:x + mask.shape[1]] == template[y, x]
        if not mask.any():
            return []
    yy, xx = np.where(mask)
    return list(zip(xx.tolist(), yy.tolist()))


def _actors(image, color):
    template = np.full((5, 5), color, dtype=int)
    template[2, 2] = 5
    return [(x + 2, y + 2) for x, y in _matches(image, template)
            if y >= 7]


def _gate_template(color, solid=False):
    template = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, color, 5, color, 5, color, 5],
        [5, color, color, color, color, color, 5],
        [5, color, color, color, color, color, 5],
        [5, color, color, color, color, color, 5],
        [5, color, color, color, color, color, 5],
        [5, 5, 5, color, 5, 5, 5],
    ], dtype=int)
    if solid:
        template[1, 1:6] = color
    return template


def _goal_template():
    return np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 9, 9, 9, 9, 9, 9, 9, -1],
        [-1, -1, 5, 5, 5, 5, 5, 9, -1],
        [-1, -1, 5, 5, 5, 5, 5, 9, -1],
        [-1, -1, 5, 5, 9, 5, 5, 9, -1],
        [-1, -1, 5, 5, 5, 5, 5, 9, -1],
        [-1, -1, 5, 5, 5, 5, 5, 9, -1],
        [-1, 9, 9, 9, 9, 9, 9, 9, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=int)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class _Board:
    def __init__(self, image):
        players = _actors(image, 9)
        self.start_xy = players[0] if players else (16, 10)

        goals = []
        for k in range(4):
            for x, y in _matches(image, np.rot90(_goal_template(), k)):
                if y >= 7:
                    goals.append((x + 4, y + 4))
        self.goal_xy = goals[0] if goals else self.start_xy

        self.gates = []
        self.gate_templates = []
        used = set()
        directions = {0: (0, 6), 90: (-6, 0),
                      180: (0, -6), 270: (6, 0)}
        for color in (8, 11):
            # Purple gates have a solid square face, unlike orange gates.
            styles = (True, False) if color == 11 else (False, True)
            for solid in styles:
                template = _gate_template(color, solid)
                for rotation in (0, 90, 180, 270):
                    rotated = np.rot90(template, (-rotation // 90) % 4)
                    for x, y in _matches(image, rotated):
                        if y < 7 or (x, y) in used:
                            continue
                        used.add((x, y))
                        center = (x + 3, y + 3)
                        dx, dy = directions[rotation]
                        opened = (center[0] + dx, center[1] + dy)
                        self.gates.append((center, opened, color, rotation))
                        self.gate_templates.append(rotated)

        portal_template = np.full((7, 7), -1, dtype=int)
        portal_template[0, :] = portal_template[-1, :] = 15
        portal_template[:, 0] = portal_template[:, -1] = 15
        self.portal_xy = []
        for x, y in _matches(image, portal_template):
            center = (x + 3, y + 3)
            if y >= 7 and center not in self.portal_xy:
                self.portal_xy.append(center)

        self.plates = []
        gate_centers = [g[0] for g in self.gates]
        for y in range(10, 61, 6):
            for x in range(4, 61, 6):
                patch = image[y - 1:y + 2, x - 1:x + 2]
                if patch.shape != (3, 3):
                    continue
                color = int(patch[1, 1])
                if color not in (8, 11, 15) or not np.all(patch == color):
                    continue
                if any(abs(x - gx) <= 3 and abs(y - gy) <= 3
                       for gx, gy in gate_centers):
                    continue
                self.plates.append(((x, y), color))

        enemy_xy = _actors(image, 14)

        nodes = set()
        for y in range(10, 61, 6):
            for x in range(4, 61, 6):
                patch = image[y - 2:y + 3, x - 2:x + 3]
                if patch.shape == (5, 5) and np.all(patch != 0):
                    nodes.add((x, y))
        nodes.update((self.start_xy, self.goal_xy))
        nodes.update(p[0] for p in self.plates)
        nodes.update(self.portal_xy)
        nodes.update(g[0] for g in self.gates)
        nodes.update(enemy_xy)

        self.xy = sorted(nodes, key=lambda p: (p[1], p[0]))
        self.index = {p: i for i, p in enumerate(self.xy)}
        self.start = self.index[self.start_xy]
        self.goal = self.index[self.goal_xy]
        self.neighbors = [{} for _ in self.xy]
        for i, (x, y) in enumerate(self.xy):
            for action, (dx, dy) in DELTA.items():
                target = self.index.get((x + dx, y + dy))
                if target is not None:
                    self.neighbors[i][action] = target

        self.gate_closed = [self.index.get(g[0], -1) for g in self.gates]
        self.gate_opened = [self.index.get(g[1], -1) for g in self.gates]
        self.toggle_mask = sum(1 << i for i, g in enumerate(self.gates)
                               if g[2] == 11)
        self.plate_nodes = [self.index[p[0]] for p in self.plates]
        self.portal_nodes = [self.index[p] for p in self.portal_xy]
        self.enemies = tuple(self.index[p] for p in enemy_xy)
        self.invalid = set()

        self._read_wiring(image)
        self._make_patrols()
        self.learned_enemy = [{} for _ in self.enemies]

        self.endpoints = {}
        for i, node in enumerate(self.plate_nodes):
            wire = self.plate_wire[i]
            if self.wire_gates[wire] or self.wire_portals[wire]:
                penalty = 10 if self.plates[i][1] == 11 else 0
                self.endpoints[node] = min(
                    self.endpoints.get(node, 100), penalty)
        for node in self.portal_nodes:
            self.endpoints.setdefault(node, 12)

        self.distance = [1000] * len(self.xy)
        self.distance[self.goal] = 0
        queue = deque([self.goal])
        while queue:
            node = queue.popleft()
            for target in self.neighbors[node].values():
                if self.distance[target] > self.distance[node] + 1:
                    self.distance[target] = self.distance[node] + 1
                    queue.append(target)

    def _read_wiring(self, image):
        colored = np.isin(image, [8, 11, 15])
        labels = np.full((64, 64), -1, dtype=int)
        count = 0
        for y in range(7, 63):
            for x in range(64):
                if not colored[y, x] or labels[y, x] >= 0:
                    continue
                labels[y, x] = count
                stack = [(x, y)]
                while stack:
                    xx, yy = stack.pop()
                    for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                                   (xx, yy - 1), (xx, yy + 1)):
                        if (0 <= nx < 64 and 7 <= ny < 63
                                and colored[ny, nx] and labels[ny, nx] < 0):
                            labels[ny, nx] = count
                            stack.append((nx, ny))
                count += 1

        def component(x, y, radius=0):
            for yy in range(max(7, y - radius), min(63, y + radius + 1)):
                for xx in range(max(0, x - radius), min(64, x + radius + 1)):
                    label = int(labels[yy, xx])
                    if label >= 0:
                        return label
            return None

        plate_components = []
        for (x, y), color in self.plates:
            label = component(x, y)
            if label is None:
                label = count
                count += 1
            plate_components.append(label)

        gate_components = [component(g[0][0], g[0][1], 2)
                           for g in self.gates]
        portal_components = [component(x - 3, y)
                             for x, y in self.portal_xy]
        components = sorted(set(plate_components))
        component_index = {c: i for i, c in enumerate(components)}

        self.plate_wire = [component_index[c] for c in plate_components]
        self.wire_plates = [[] for _ in components]
        self.wire_gates = [[] for _ in components]
        self.wire_portals = [[] for _ in components]
        for i, wire in enumerate(self.plate_wire):
            self.wire_plates[wire].append(self.plate_nodes[i])
        for i, label in enumerate(gate_components):
            if label in component_index:
                self.wire_gates[component_index[label]].append(i)
        for i, label in enumerate(portal_components):
            if label in component_index:
                self.wire_portals[component_index[label]].append(
                    self.portal_nodes[i])

        for wire, pads in enumerate(self.wire_portals):
            if len(pads) != 2:
                self.wire_portals[wire] = []

        if len(self.portal_nodes) == 2 and not any(self.wire_portals):
            for i, (_, color) in enumerate(self.plates):
                if color == 15:
                    self.wire_portals[self.plate_wire[i]] = list(
                        self.portal_nodes)

    def _make_patrols(self):
        self.patrols = []
        for enemy in self.enemies:
            ex, ey = self.xy[enemy]
            best = None
            for row in range(10, ey + 1, 6):
                leg = [(ex, y) for y in range(row, ey + 1, 6)]
                if any(p not in self.index for p in leg):
                    continue
                left = right = ex
                while (left - 6, row) in self.index:
                    left -= 6
                while (right + 6, row) in self.index:
                    right += 6
                width = (right - left) // 6 + 1
                score = 10 * width - (ey - row) // 6
                if best is None or score > best[0]:
                    best = (score, row, left, right, leg)
            route = {enemy}
            if best is not None:
                _, row, left, right, leg = best
                route.update(self.index[p] for p in leg)
                route.update(self.index[(x, row)]
                             for x in range(left, right + 1, 6))
            if len(route) < 2:
                route.update(self.neighbors[enemy].values())
            self.patrols.append(route)

    def blocked(self, node, bits):
        for i in range(len(self.gates)):
            square = (self.gate_opened[i] if bits & (1 << i)
                      else self.gate_closed[i])
            if node == square:
                return True
        return False

    def legal(self, node, action, bits):
        if node < 0 or (node, action, bits) in self.invalid:
            return None
        target = self.neighbors[node].get(action)
        if target is None or self.blocked(target, bits):
            return None
        return target

    def occupancy(self, positions):
        occupied = {p for p in positions if p >= 0}
        return [any(p in occupied for p in plates)
                for plates in self.wire_plates]

    def signal(self, before, after, bits, positions, rewinding):
        closing = []
        for wire, (old, new) in enumerate(zip(before, after)):
            if old == new:
                continue
            for gate in self.wire_gates[wire]:
                flag = 1 << gate
                was_open = bool(bits & flag)
                if self.gates[gate][2] == 11:
                    if new:
                        bits ^= flag
                elif new:
                    bits |= flag
                else:
                    bits &= ~flag
                if was_open and not bits & flag:
                    closing.append(gate)

            pads = self.wire_portals[wire]
            trigger = (not new) if rewinding else new
            if pads and trigger:
                p, q = pads
                positions = [q if x == p else p if x == q else x
                             for x in positions]
        return bits, positions, closing

    def enemy_move(self, number, position, heading, bits):
        if position < 0:
            return position, heading
        known = self.learned_enemy[number].get((position, heading, bits))
        if known is not None:
            return known
        route = self.patrols[number]
        direction = heading
        for _ in range(4):
            selected = None
            for h in (direction, (direction - 1) % 4,
                      (direction + 1) % 4, (direction + 2) % 4):
                action = HEADING_ACTION[h]
                target = self.neighbors[position].get(action)
                if target is not None and target in route:
                    selected = target, h
                    break
            if selected is None:
                return position, direction
            target, h = selected
            if not self.blocked(target, bits):
                return target, h
            direction = (h + 2) % 4
        return position, direction

    def advance(self, state, action, paths, reverse=0, index=None,
                enemy_previous=None, arrival_rewinding=False):
        player, ghosts, bits, enemies, headings, tick = state
        target = self.legal(player, action, bits)
        if target is None and not reverse:
            return None

        positions = [player] + list(ghosts) + list(enemies)
        targets = positions[:]
        moving = []
        if target is not None:
            targets[0] = target
            moving.append(0)

        step_index = index if reverse else tick
        for j, path in enumerate(paths):
            if step_index < len(path):
                a = path[step_index]
                if reverse:
                    a = OPPOSITE[a]
                target = self.legal(ghosts[j], a, bits)
                if target is not None:
                    targets[1 + j] = target
                    moving.append(1 + j)

        new_headings = list(headings)
        offset = 1 + len(ghosts)
        for j, position in enumerate(enemies):
            if reverse and enemy_previous is not None:
                old_position = enemy_previous[j]
                if old_position >= 0:
                    if (position < 0 or old_position == position
                            or not self.blocked(old_position, bits)):
                        targets[offset + j] = old_position
                        moving.append(offset + j)
            elif not reverse:
                target, heading = self.enemy_move(
                    j, position, headings[j], bits)
                new_headings[j] = heading
                if target >= 0:
                    targets[offset + j] = target
                    moving.append(offset + j)

        before = self.occupancy(positions)
        departed = positions[:]
        for j in moving:
            departed[j] = -1
        middle = self.occupancy(departed)
        new_bits, _, closing = self.signal(
            before, middle, bits, departed, reverse == 2)

        for gate in closing:
            square = self.gate_closed[gate]
            if targets[0] == square:
                return None
            for j in range(len(enemies)):
                if targets[offset + j] == square:
                    targets[offset + j] = -1

        if any(targets[0] == p for p in targets[offset:] if p >= 0):
            return None

        after = self.occupancy(targets)
        new_bits, targets, closing = self.signal(
            middle, after, new_bits, targets, arrival_rewinding)
        for gate in closing:
            square = self.gate_closed[gate]
            if targets[0] == square:
                return None
            for j in range(len(enemies)):
                if targets[offset + j] == square:
                    targets[offset + j] = -1

        new_tick = index if reverse else tick + 1
        return (targets[0], tuple(targets[1:offset]), new_bits,
                tuple(targets[offset:]), tuple(new_headings), new_tick)

    def rewind(self, actions, states, paths):
        if not actions:
            return None
        state = states[-1]
        for i in range(len(actions) - 1, -1, -1):
            mode = 1 if i == len(actions) - 1 else 2
            state = self.advance(
                state, OPPOSITE[actions[i]], paths,
                reverse=mode, index=i, enemy_previous=states[i][3],
                arrival_rewinding=(i > 0))
            if state is None:
                return None
        _, ghosts, bits, enemies, headings, _ = state
        return (self.start, ghosts + (self.start,), bits,
                enemies, headings, 0)

    def observed_bits(self, image, previous):
        bits = previous
        matches_cache = {}
        for i, (center, opened, color, rotation) in enumerate(self.gates):
            template = self.gate_templates[i]
            key = template.tobytes()
            if key not in matches_cache:
                matches_cache[key] = {
                    (x + 3, y + 3) for x, y in _matches(image, template)}
            found = matches_cache[key]
            if opened in found:
                bits |= 1 << i
            elif center in found:
                bits &= ~(1 << i)
        return bits


def _trace(end, parents):
    actions = []
    states = [end]
    state = end
    while parents[state] is not None:
        previous, action = parents[state]
        actions.append(action)
        states.append(previous)
        state = previous
    actions.reverse()
    states.reverse()
    return actions, states


def _stage_search(board, initial, paths, can_record, deadline):
    parents = {initial: None}
    queue = deque([initial])
    endpoints = []
    endpoint_times = {}
    initial_tick = initial[5]
    replay_end = max((len(path) for path in paths), default=0)

    # Once all replays finish, absolute time is irrelevant to the final
    # route. Enemy position and heading already encode patrol phase.
    time_cap = replay_end + (36 if can_record else 0)

    def visit_key(state):
        return state[:5] + (min(state[5], time_cap),)

    visited = {visit_key(initial)}
    max_depth = min(85, max(40, 2 * len(board.xy)))
    processed = 0

    while queue:
        state = queue.popleft()
        depth = state[5] - initial_tick
        if state[0] == board.goal:
            actions, _ = _trace(state, parents)
            return actions, endpoints, parents

        if can_record and 0 < depth <= 48:
            node = state[0]
            if node in board.endpoints and node not in state[1]:
                key = (node, state[2] & board.toggle_mask)
                times = endpoint_times.setdefault(key, [])
                tick = state[5]
                if not times:
                    times.append(tick)
                    endpoints.append(state)
                elif len(times) < 6:
                    delays = (4, 8, 12, 20, 28)
                    if tick >= times[0] + delays[len(times) - 1]:
                        times.append(tick)
                        endpoints.append(state)

        if depth >= max_depth:
            continue
        processed += 1
        if processed % 128 == 0:
            if time.monotonic() >= deadline or len(parents) > 60000:
                break

        actions = sorted(
            board.neighbors[state[0]],
            key=lambda a: board.distance[board.neighbors[state[0]][a]])
        for action in actions:
            next_state = board.advance(state, action, paths)
            if next_state is None:
                continue
            key = visit_key(next_state)
            if key not in visited:
                visited.add(key)
                parents[next_state] = (state, action)
                queue.append(next_state)

    return None, endpoints, parents


def _plan(board, initial, paths, history, history_states, slots, deadline):
    fallback = [None]
    searches = [0]

    def recurse(state, ghost_paths, prefix, prefix_states):
        if time.monotonic() >= deadline:
            return None
        searches[0] += 1
        if searches[0] > 45:
            return None

        can_record = len(ghost_paths) < slots - 1
        goal_path, endings, parents = _stage_search(
            board, state, ghost_paths, can_record, deadline)
        if goal_path is not None:
            return goal_path
        if not can_record:
            return None

        endings.sort(key=lambda s: (
            s[5] + board.endpoints.get(s[0], 0),
            board.distance[s[0]]))
        for ending in endings:
            if time.monotonic() >= deadline:
                break
            tail, tail_states = _trace(ending, parents)
            full_path = prefix + tail
            full_states = prefix_states + tail_states[1:]
            next_state = board.rewind(full_path, full_states, ghost_paths)
            if next_state is None:
                continue
            if fallback[0] is None:
                fallback[0] = tail + [5]
            result = recurse(
                next_state, ghost_paths + [tuple(full_path)],
                [], [next_state])
            if result is not None:
                return tail + [5] + result
        return None

    answer = recurse(initial, paths, history, history_states)
    return answer if answer is not None else fallback[0]


def _assign_observed(board, predicted, observed_xy):
    observed = [board.index[p] for p in observed_xy if p in board.index]
    if not observed or not predicted:
        return tuple(predicted)
    n = len(predicted)
    if len(observed) >= n:
        best = None
        for choice in permutations(observed, n):
            cost = sum(_distance(board.xy[p], board.xy[q])
                       if p >= 0 else 100
                       for p, q in zip(predicted, choice))
            if best is None or cost < best[0]:
                best = (cost, choice)
        return tuple(best[1])
    result = list(predicted)
    remaining = set(range(n))
    for node in observed:
        j = min(remaining, key=lambda i:
                _distance(board.xy[result[i]], board.xy[node])
                if result[i] >= 0 else 10000)
        result[j] = node
        remaining.remove(j)
    return tuple(result)


def _slots(image):
    count = 0
    for x in range(1, 33, 4):
        patch = image[1:4, x:x + 3]
        if patch.shape != (3, 3):
            continue
        if np.all(patch == 1) or (
                int(patch[1, 1]) == 0
                and np.count_nonzero(np.isin(patch, [2, 9])) >= 7):
            count += 1
    return max(2, min(4, count))


def act(obs, available, memory):
    started = time.monotonic()
    available = set(available)

    if not memory.get("_initialized") and not memory.get("_reset_sent"):
        if "RESET" in available:
            memory["_reset_sent"] = True
            return "RESET", None, None

    if not memory.get("_initialized"):
        board = _Board(obs)
        state = (board.start, (), 0, board.enemies,
                 tuple(0 for _ in board.enemies), 0)
        memory.update({
            "_board": board,
            "_slots": _slots(obs),
            "_state": state,
            "_paths": [],
            "_history": [],
            "_history_states": [state],
            "_queue": [],
            "_visits": {},
            "_initialized": True,
        })

    board = memory["_board"]
    state = memory["_state"]
    paths = memory["_paths"]
    history = memory["_history"]
    history_states = memory["_history_states"]
    queue = memory["_queue"]

    player, ghosts, bits, enemies, headings, tick = state
    actual_players = _actors(obs, 9)
    actual_player = (board.index.get(actual_players[0], player)
                     if actual_players else player)
    actual_ghosts = _assign_observed(board, ghosts, _actors(obs, 2))
    enemy_xy = _actors(obs, 14)
    actual_enemies = _assign_observed(board, enemies, enemy_xy)
    if enemies and not enemy_xy:
        actual_enemies = tuple(-1 for _ in enemies)
    actual_bits = board.observed_bits(obs, bits)
    actual_headings = list(headings)

    previous = memory.get("_previous")
    last_action = memory.get("_last_action")
    rejected = False
    if previous is not None and last_action in DELTA:
        rejected = (
            actual_player == previous[0]
            and actual_ghosts == previous[1]
            and actual_bits == previous[2]
            and actual_enemies == previous[3]
            and player != previous[0])
        if rejected:
            board.invalid.add((previous[0], last_action, previous[2]))
            tick = previous[5]
            actual_headings = list(previous[4])
            if history:
                history.pop()
            if len(history_states) > 1:
                history_states.pop()
            queue.clear()
        else:
            for j, (old_node, new_node) in enumerate(
                    zip(previous[3], actual_enemies)):
                if old_node < 0 or new_node < 0:
                    continue
                x0, y0 = board.xy[old_node]
                x1, y1 = board.xy[new_node]
                movement = (x1 - x0, y1 - y0)
                heading = {
                    (0, 6): 0, (6, 0): 1,
                    (0, -6): 2, (-6, 0): 3
                }.get(movement)
                if heading is not None:
                    actual_headings[j] = heading
                    board.patrols[j].update((old_node, new_node))
                    board.learned_enemy[j][
                        (old_node, previous[4][j], previous[2])
                    ] = (new_node, heading)

    corrected = (actual_player, actual_ghosts, actual_bits,
                 actual_enemies, tuple(actual_headings), tick)
    if corrected != state:
        state = corrected
        memory["_state"] = state
        if history_states:
            history_states[-1] = state
        queue.clear()

    visits = memory["_visits"]
    location_key = state[:5]
    visits[location_key] = visits.get(location_key, 0) + 1

    if not queue:
        result = _plan(
            board, state, paths, history, history_states,
            memory["_slots"], started + 8.3)
        if result:
            queue.extend(result)

    action = None
    while queue:
        candidate = queue.pop(0)
        if "ACTION%d" % candidate not in available:
            continue
        if candidate == 5:
            if history and len(paths) < memory["_slots"] - 1:
                action = candidate
                break
        elif board.advance(state, candidate, paths) is not None:
            action = candidate
            break
        else:
            queue.clear()
            break

    if action is None:
        options = []
        for candidate in board.neighbors[state[0]]:
            if "ACTION%d" % candidate not in available:
                continue
            next_state = board.advance(state, candidate, paths)
            if next_state is None:
                continue
            score = board.distance[next_state[0]]
            if (next_state[0] in board.endpoints
                    and next_state[0] not in next_state[1]):
                score -= 15
            score += 6 * visits.get(next_state[:5], 0)
            if history and candidate == OPPOSITE[history[-1]]:
                score += 3
            options.append((score, candidate))
        if options:
            action = min(options)[1]
        elif history and "ACTION5" in available:
            action = 5
        elif "RESET" in available:
            memory.clear()
            memory["_reset_sent"] = True
            return "RESET", None, None
        else:
            name = next((a for a in available if a != "RESET"), "ACTION1")
            return name, None, None

    memory["_previous"] = state
    memory["_last_action"] = action

    if action == 5:
        new_state = board.rewind(history, history_states, paths)
        if new_state is not None:
            paths.append(tuple(history))
            history.clear()
            history_states[:] = [new_state]
            memory["_state"] = new_state
            visits.clear()
        else:
            queue.clear()
    else:
        next_state = board.advance(state, action, paths)
        if next_state is not None:
            history.append(action)
            history_states.append(next_state)
            memory["_state"] = next_state

    return "ACTION%d" % action, None, None

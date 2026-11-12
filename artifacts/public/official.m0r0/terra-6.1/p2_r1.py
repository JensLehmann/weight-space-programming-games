from collections import Counter, deque
from itertools import permutations
import heapq
import math

import numpy as np


SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _components(obs, color):
    mask = obs == color
    seen = np.zeros((64, 64), dtype=np.bool_)
    result = []
    for y in range(64):
        for x in range(64):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs, ys = [], []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < 64 and 0 <= ny < 64
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            if len(xs) == w * h and len(xs) <= 256:
                result.append((x0, y0, w, h))
    return result


def _geometry(obs):
    comps = _components(obs, 10) + _components(obs, 1)
    votes = Counter()
    for x, y, w, h in comps:
        s = math.gcd(w, h)
        if 2 <= s <= 8 and w <= 4 * s and h <= 4 * s:
            votes[(s, x % s, y % s)] += 1
    if not votes:
        return None
    s, px, py = max(votes, key=lambda k: (votes[k], k[0]))
    return s, px, py


def _read_board(obs, geometry, previous):
    s, ox, oy = geometry
    width = (64 - ox) // s
    height = (64 - oy) // s
    wall_colors = (
        (int(obs[1, 1]), int(obs[1, 62])),
        (int(obs[62, 1]), int(obs[62, 62])),
    )

    walls = set()
    players = []
    boxes = []
    selected = -1
    colored = {12: set(), 14: set(), 15: set()}
    traps = set() if previous is None else set(previous["traps"])

    for gy in range(height):
        for gx in range(width):
            p = gy * width + gx
            x0, y0 = ox + gx * s, oy + gy * s
            cx, cy = x0 + s // 2, y0 + s // 2
            value = int(obs[cy, cx])
            wc = wall_colors[cy >= 32][cx >= 32]
            corners = (
                int(obs[y0, x0]),
                int(obs[y0, x0 + s - 1]),
                int(obs[y0 + s - 1, x0]),
                int(obs[y0 + s - 1, x0 + s - 1]),
            )

            if value in (9, 11) and all(v == 5 for v in corners):
                boxes.append(p)
                if value == 11:
                    selected = p
                continue

            if value in (1, 10) and value != wc:
                block = obs[y0:y0 + s, x0:x0 + s]
                if np.count_nonzero(block == value) >= s * s - s:
                    players.append(p)
                    continue

            if s >= 3:
                a, b = [], []
                for yy in range(y0, y0 + s):
                    if yy == 0 or yy == 63:
                        continue
                    for xx in range(x0, x0 + s):
                        (a if (xx + yy) % 2 else b).append(int(obs[yy, xx]))
                if a and b:
                    if ((all(v == 5 for v in a)
                         and len(set(b)) == 1 and b[0] != 5)
                            or (all(v == 5 for v in b)
                                and len(set(a)) == 1 and a[0] != 5)):
                        traps.add(p)
                        continue

            if value == wc:
                walls.add(p)
            elif value in colored:
                colored[value].add(p)
            elif value != 5:
                walls.add(p)

    gates = {} if previous is None else dict(previous["gates"])
    buttons = {} if previous is None else dict(previous["buttons"])

    for color, cells in colored.items():
        remaining = set(cells)
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                p = stack.pop()
                x, y = p % width, p // width
                for dx, dy in DIRS.values():
                    nx, ny = x + dx, y + dy
                    np_ = ny * width + nx
                    if 0 <= nx < width and 0 <= ny < height and np_ in remaining:
                        remaining.remove(np_)
                        component.add(np_)
                        stack.append(np_)
            if len(component) >= 3:
                for p in component:
                    if p not in buttons:
                        gates[p] = color
            else:
                for p in component:
                    if p not in gates:
                        buttons[p] = color

    walls.difference_update(gates)
    walls.difference_update(buttons)
    walls.difference_update(players)
    walls.difference_update(boxes)
    walls.difference_update(traps)

    world = {
        "w": width,
        "h": height,
        "walls": walls,
        "gates": gates,
        "buttons": buttons,
        "traps": traps,
    }
    return world, tuple(sorted(players)), tuple(sorted(boxes)), selected


def _signature(world):
    return (
        frozenset(world["walls"]),
        frozenset(world["gates"].items()),
        frozenset(world["buttons"].items()),
        frozenset(world["traps"]),
    )


def _initial_hypotheses(players):
    hypotheses = []
    if len(players) == 2:
        for signs in permutations(SIGNS, 2):
            hypotheses.append((tuple(signs), tuple(players), ()))
    elif len(players) == 4:
        for signs in permutations(SIGNS, 4):
            hypotheses.append((tuple(signs), tuple(players), ()))
    elif len(players) == 3:
        for locked in players:
            active = tuple(p for p in players if p != locked)
            for signs in permutations(SIGNS, 2):
                hypotheses.append((tuple(signs), active, (locked,)))
    return hypotheses


def _advance(state, command, world, signs):
    tokens, boxes, selected = state
    width, height = world["w"], world["h"]
    walls = world["walls"]
    gates = world["gates"]
    buttons = world["buttons"]
    traps = world["traps"]

    if isinstance(command, tuple):
        target = command[1]
        if target in boxes:
            return (tokens, boxes, target), ()
        return (tokens, boxes, -1), ()

    dx, dy = DIRS[command]
    active = set(p for p in tokens if p >= 0)
    open_colors = {buttons[p] for p in active if p in buttons}

    def blocked(p):
        return p in walls or (p in gates and gates[p] not in open_colors)

    if selected >= 0:
        x, y = selected % width, selected // width
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            return state, ()
        np_ = ny * width + nx
        if (blocked(np_) or np_ in boxes or np_ in active
                or np_ in buttons or np_ in traps):
            return state, ()
        new_boxes = list(boxes)
        new_boxes.remove(selected)
        new_boxes.append(np_)
        return (tokens, tuple(sorted(new_boxes)), np_), ()

    old = list(tokens)
    new = list(tokens)
    box_set = set(boxes)

    for i, p in enumerate(old):
        if p < 0:
            continue
        x, y = p % width, p // width
        sx, sy = signs[i]
        nx, ny = x + dx * sx, y + dy * sy
        if 0 <= nx < width and 0 <= ny < height:
            np_ = ny * width + nx
            if not blocked(np_) and np_ not in box_set:
                new[i] = np_

    if any(p in traps for p in new if p >= 0):
        return None, ()

    for i in range(len(old)):
        if old[i] < 0:
            continue
        for j in range(i + 1, len(old)):
            if old[j] < 0:
                continue
            if (old[i] // width == old[j] // width
                    and abs(old[i] % width - old[j] % width) == 1
                    and (new[i] == old[j] or new[j] == old[i])):
                x = ((new[i] % width) + (new[j] % width)) // 2
                y = ((new[i] // width) + (new[j] // width)) // 2
                new[i] = new[j] = y * width + x

    groups = {}
    for i, p in enumerate(new):
        if p >= 0:
            groups.setdefault(p, []).append(i)
    locks = []
    for p, indices in groups.items():
        if len(indices) >= 2:
            new[indices[0]] = -1
            new[indices[1]] = -1
            locks.append(p)
            for i in indices[2:]:
                new[i] = old[i]

    return (tuple(new), boxes, -1), tuple(locks)


def _visit_key(state):
    tokens, boxes, selected = state
    return tuple(sorted(p for p in tokens if p >= 0)), boxes, selected


def _search(world, tokens, boxes, selected, signs, visits, limit=220000):
    width, height = world["w"], world["h"]
    walls = world["walls"] | world["traps"]
    initial = (tokens, boxes, selected)
    distance_cache = {}

    def distances(source):
        cached = distance_cache.get(source)
        if cached is not None:
            return cached
        dist = {source: 0}
        q = deque([source])
        while q:
            p = q.popleft()
            x, y = p % width, p // width
            for dx, dy in DIRS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    np_ = ny * width + nx
                    if np_ not in walls and np_ not in dist:
                        dist[np_] = dist[p] + 1
                        q.append(np_)
        distance_cache[source] = dist
        return dist

    def pair_bound(a, b):
        return (distances(a).get(b, 1000) + 1) // 2

    def heuristic(state):
        active = [p for p in state[0] if p >= 0]
        if not active:
            return 0
        if len(active) == 2:
            return pair_bound(active[0], active[1])
        if len(active) == 4:
            a, b, c, d = active
            return min(
                max(pair_bound(a, b), pair_bound(c, d)),
                max(pair_bound(a, c), pair_bound(b, d)),
                max(pair_bound(a, d), pair_bound(b, c)),
            )
        return 500

    def partial_rank(state, h, g):
        count = visits.get(_visit_key(state), 0)
        # Prefer unselected states for partial plans.
        selection_penalty = 1 if state[2] >= 0 else 0
        return (h * 4 + min(count, 5) * 7 + selection_penalty, g)

    h0 = heuristic(initial)
    queue = [(2 * h0, h0, 0, 0, initial)]
    costs = {initial: 0}
    parents = {}
    serial = 0
    best_state = initial
    best_rank = partial_rank(initial, h0, 0)
    goal = None
    expanded = 0

    while queue and expanded < limit:
        _, h, g, _, state = heapq.heappop(queue)
        if costs.get(state) != g:
            continue
        expanded += 1

        rank = partial_rank(state, h, g)
        if rank < best_rank:
            best_rank, best_state = rank, state
        if all(p < 0 for p in state[0]):
            goal = state
            break

        commands = [1, 2, 3, 4]
        if state[2] < 0:
            # A block can be selected remotely, but postponing its movement
            # until a player approaches avoids a large irrelevant search tree.
            active = [p for p in state[0] if p >= 0]
            for box in state[1]:
                bx, by = box % width, box // width
                if any(abs(bx - p % width) + abs(by - p // width) <= 1
                       for p in active):
                    commands.append(("click", box))
        else:
            commands.append(("click", -1))
            # Switching directly to a second nearby block is also possible.
            active = [p for p in state[0] if p >= 0]
            for box in state[1]:
                if box == state[2]:
                    continue
                bx, by = box % width, box // width
                if any(abs(bx - p % width) + abs(by - p // width) <= 1
                       for p in active):
                    commands.append(("click", box))

        for command in commands:
            next_state, _ = _advance(state, command, world, signs)
            if next_state is None or next_state == state:
                continue
            ng = g + 1
            if ng >= costs.get(next_state, 10**9):
                continue
            costs[next_state] = ng
            parents[next_state] = (state, command)
            nh = heuristic(next_state)
            serial += 1
            heapq.heappush(
                queue, (ng + 2 * nh, nh, ng, serial, next_state)
            )

    end = goal if goal is not None else best_state
    if end == initial:
        return [], False
    path = []
    while end != initial:
        end, command = parents[end]
        path.append(command)
    path.reverse()
    return path, goal is not None


def _click_coordinates(target, geometry, width, boxes, active_tokens):
    s, ox, oy = geometry
    if target >= 0:
        return (
            oy + (target // width) * s + s // 2,
            ox + (target % width) * s + s // 2,
        )

    for p in active_tokens:
        if p >= 0 and p not in boxes:
            return (
                oy + (p // width) * s + s // 2,
                ox + (p % width) * s + s // 2,
            )

    for row, col in ((32, 32), (31, 31), (31, 32), (32, 31)):
        on_box = False
        for p in boxes:
            x0 = ox + (p % width) * s
            y0 = oy + (p // width) * s
            if x0 <= col < x0 + s and y0 <= row < y0 + s:
                on_box = True
                break
        if not on_box:
            return row, col
    return 32, 32


def act(obs, available, memory):
    obs = np.asarray(obs)
    geometry = memory.get("geometry")
    if geometry is None:
        geometry = _geometry(obs)
        if geometry is None:
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION1", None, None
        memory["geometry"] = geometry

    previous_world = memory.get("world")
    world, players, boxes, selected = _read_board(
        obs, geometry, previous_world
    )
    hypotheses = []

    if "hypotheses" in memory and "last" in memory:
        old_boxes = memory["boxes"]
        old_selected = memory["selected"]
        for signs, tokens, locked in memory["hypotheses"]:
            state = (tokens, old_boxes, old_selected)
            next_state, new_locks = _advance(
                state, memory["last"], previous_world, signs
            )
            if next_state is None:
                continue
            ntokens, nboxes, nselected = next_state
            nlocked = tuple(sorted(locked + new_locks))
            visible = set(p for p in ntokens if p >= 0) | set(nlocked)
            if (visible == set(players) and nboxes == boxes
                    and nselected == selected):
                hypotheses.append((signs, ntokens, nlocked))

    if not hypotheses:
        hypotheses = _initial_hypotheses(players)
        memory.pop("plan", None)
        memory.pop("predicted_choice", None)

    if not hypotheses:
        if selected >= 0 and players:
            row, col = _click_coordinates(
                -1, geometry, world["w"], boxes, players
            )
            return "ACTION6", int(row), int(col)
        return "ACTION1", None, None

    visits = memory.setdefault("visits", {})
    current_key = (tuple(sorted(players)), boxes, selected)
    visits[current_key] = visits.get(current_key, 0) + 1

    chosen = hypotheses[0]
    saved_choice = memory.get("predicted_choice")
    saved_plan = memory.get("plan", [])

    if (saved_choice in hypotheses and saved_plan
            and memory.get("world_signature") == _signature(world)):
        chosen = saved_choice
        plan = list(saved_plan)
    else:
        limit = 220000 if boxes else 60000
        plan, _ = _search(
            world, chosen[1], boxes, selected, chosen[0], visits, limit
        )

    if not plan:
        for candidate in hypotheses[1:3]:
            alternative, _ = _search(
                world, candidate[1], boxes, selected,
                candidate[0], visits, 50000
            )
            if alternative:
                chosen, plan = candidate, alternative
                break

    if not plan:
        # Take a valid least-visited step instead of immediately resetting.
        choices = []
        state = (chosen[1], boxes, selected)
        commands = [1, 2, 3, 4]
        if selected >= 0:
            commands.append(("click", -1))
        else:
            commands.extend(("click", p) for p in boxes)
        for command in commands:
            next_state, _ = _advance(state, command, world, chosen[0])
            if next_state is None or next_state == state:
                continue
            choices.append((visits.get(_visit_key(next_state), 0), command))
        if choices:
            choices.sort(key=lambda item: item[0])
            plan = [choices[0][1]]
        else:
            plan = [1]

    command = plan.pop(0)
    predicted_state, new_locks = _advance(
        (chosen[1], boxes, selected), command, world, chosen[0]
    )
    predicted_choice = None
    if predicted_state is not None:
        predicted_choice = (
            chosen[0],
            predicted_state[0],
            tuple(sorted(chosen[2] + new_locks)),
        )

    memory["world"] = world
    memory["world_signature"] = _signature(world)
    memory["hypotheses"] = hypotheses
    memory["boxes"] = boxes
    memory["selected"] = selected
    memory["last"] = command
    memory["plan"] = plan
    memory["predicted_choice"] = predicted_choice

    if isinstance(command, tuple):
        row, col = _click_coordinates(
            command[1], geometry, world["w"], boxes, chosen[1]
        )
        return "ACTION6", int(row), int(col)
    return "ACTION%d" % command, None, None

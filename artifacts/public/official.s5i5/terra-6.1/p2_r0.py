import time
import heapq
from collections import Counter, deque
from functools import lru_cache

import numpy as np


DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
BOARD = 21


def components(points):
    points = set(points)
    result = []
    while points:
        start = points.pop()
        stack = [start]
        comp = [start]
        while stack:
            x, y = stack.pop()
            for p in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if p in points:
                    points.remove(p)
                    stack.append(p)
                    comp.append(p)
        result.append(comp)
    return result


def find_controls(obs):
    points = {(x, y) for y in range(63) for x in range(64)
              if int(obs[y, x]) == 2}
    controls = []
    covered = set()

    for comp in components(points):
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 5 or h < 5 or w > 20 or h > 20:
            continue

        border = (
            [(x, y0) for x in range(x0, x1 + 1)] +
            [(x, y1) for x in range(x0, x1 + 1)] +
            [(x0, y) for y in range(y0 + 1, y1)] +
            [(x1, y) for y in range(y0 + 1, y1)]
        )
        if any(int(obs[y, x]) != 2 for x, y in border):
            continue

        counts = Counter()
        for y in range(y0 + 1, y1):
            for x in range(x0 + 1, x1):
                c = int(obs[y, x])
                if c not in (2, 3, 4, 5, 13, 15):
                    counts[c] += 1
        if not counts:
            continue

        color = counts.most_common(1)[0][0]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                covered.add((x, y))

        if w == 5 and h == 5:
            controls.append((color, "rotate", x0 + 2, y0 + 2))
        elif w > h:
            controls.append((color, "shorter",
                             x0 + w // 4, y0 + h // 2))
            controls.append((color, "longer",
                             x0 + 3 * w // 4, y0 + h // 2))
        elif h > w:
            controls.append((color, "shorter",
                             x0 + w // 2, y0 + h // 4))
            controls.append((color, "longer",
                             x0 + w // 2, y0 + 3 * h // 4))

    return controls, covered


def read_cells(obs, covered, colors):
    result = {}
    obscured = set()
    candidates = set(colors) | {1, 15}

    for gy in range(BOARD):
        for gx in range(BOARD):
            x0, y0 = 3 * gx, 3 * gy
            if any((x, y) in covered
                   for y in range(y0, y0 + 3)
                   for x in range(x0, x0 + 3)):
                obscured.add((gx, gy))
                continue

            counts = Counter(
                int(obs[y, x])
                for y in range(y0, y0 + 3)
                for x in range(x0, x0 + 3)
                if int(obs[y, x]) in candidates
            )
            if counts:
                color, n = counts.most_common(1)[0]
                if n >= 2:
                    result[(gx, gy)] = color

    return result, obscured


def read_markers(obs, covered):
    markers = []
    targets = []
    for gy in range(BOARD):
        for gx in range(BOARD):
            x, y = 3 * gx + 1, 3 * gy + 1
            points = ((x, y), (x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1))
            if any(p in covered for p in points):
                continue
            if int(obs[y, x]) == 13:
                markers.append((gx, gy))
            if all(int(obs[yy, xx]) == 13
                   for xx, yy in points[1:]):
                targets.append((gx, gy))
    return markers, targets


def read_arm(obs, comp):
    cells = set(comp)
    if len({x for x, y in cells}) != 1 and \
       len({y for x, y in cells}) != 1:
        return None

    candidates = []
    for gx, gy in cells:
        x, y = 3 * gx, 3 * gy
        edges = (
            [(x + k, y + 2) for k in range(3)],
            [(x, y + k) for k in range(3)],
            [(x + k, y) for k in range(3)],
            [(x + 2, y + k) for k in range(3)],
        )
        for d, edge in enumerate(edges):
            score = sum(int(obs[yy, xx]) == 3 for xx, yy in edge)
            if score < 2:
                continue
            dx, dy = DIRS[d]
            expected = {(gx + k * dx, gy + k * dy)
                        for k in range(len(cells))}
            if expected == cells:
                candidates.append((score, gx, gy, d, len(cells)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, x, y, d, length = candidates[0]
    return (x, y, d, length)


def arm_cells(arm):
    x, y, d, length = arm
    dx, dy = DIRS[d]
    return tuple((x + k * dx, y + k * dy)
                 for k in range(length))


def parse_frame(obs):
    controls, covered = find_controls(obs)
    colors = sorted({c for c, kind, x, y in controls})
    cells, obscured = read_cells(obs, covered, colors)
    markers, targets = read_markers(obs, covered)

    types = []
    piece_colors = []
    state = []

    for color in colors:
        points = [p for p, c in cells.items() if c == color]
        found = []
        for comp in components(points):
            arm = read_arm(obs, comp)
            if arm is not None:
                found.append(arm)
        found.sort()
        for arm in found:
            types.append("arm")
            piece_colors.append(color)
            state.append(arm)

    if 1 not in colors:
        points = [p for p, c in cells.items() if c == 1]
        rigid = [tuple(sorted(comp)) for comp in components(points)]
        rigid.sort()
        for comp in rigid:
            types.append("rigid")
            piece_colors.append(1)
            state.append(comp)

    walls = frozenset(p for p, c in cells.items() if c == 15)
    count = len(state)
    occupied = []
    for i in range(count):
        if types[i] == "arm":
            occupied.append(set(arm_cells(state[i])))
        else:
            occupied.append(set(state[i]))

    # An attached piece normally starts immediately beyond its parent's
    # free end. This remains true when the attached piece is rotated.
    candidates = []
    for i in range(count):
        if types[i] != "arm":
            continue
        x, y, d, length = state[i]
        dx, dy = DIRS[d]
        beyond = (x + length * dx, y + length * dy)
        for j in range(count):
            if i == j or beyond not in occupied[j]:
                continue
            score = 0
            if types[j] == "arm":
                if (state[j][0], state[j][1]) == beyond:
                    score += 5
                score += 1
            else:
                score += 2
            candidates.append((-score, i, j))

    parent = [None] * count

    def makes_cycle(i, j):
        p = i
        while p is not None:
            if p == j:
                return True
            p = parent[p]
        return False

    for _, i, j in sorted(candidates):
        if parent[j] is None and not makes_cycle(i, j):
            parent[j] = i

    marker_owner = []
    for p in markers:
        owners = [i for i in range(count) if p in occupied[i]]
        marker_owner.append(owners[0] if owners else None)

    children = [[] for _ in range(count + len(markers))]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)
    for k, owner in enumerate(marker_owner):
        if owner is not None:
            children[owner].append(count + k)

    state.extend(markers)
    model = {
        "controls": controls,
        "covered": covered,
        "obscured": obscured,
        "types": types,
        "colors": piece_colors,
        "walls": walls,
        "parent": parent,
        "children": children,
        "count": count,
        "markers": len(markers),
        "targets": tuple(targets),
    }
    return model, tuple(state)


def frame_matches(obs, model, state):
    count = model["count"]
    expected = {}
    for i in range(count):
        if model["types"][i] == "arm":
            points = arm_cells(state[i])
        else:
            points = state[i]
        for p in points:
            expected[p] = model["colors"][i]

    control_colors = {c for c, kind, x, y in model["controls"]}
    observed, obscured = read_cells(
        obs, model["covered"], control_colors
    )
    relevant = control_colors | {1}
    all_points = set(expected)
    all_points.update(p for p, c in observed.items() if c in relevant)

    for p in all_points:
        if p in obscured:
            continue
        if expected.get(p) != observed.get(p):
            return False

    actual_markers, _ = read_markers(obs, model["covered"])
    predicted_markers = {
        state[count + k]
        for k in range(model["markers"])
        if state[count + k] not in obscured
    }
    return set(actual_markers) == predicted_markers


def make_simulator(model):
    count = model["count"]
    types = model["types"]
    colors = model["colors"]
    children = model["children"]
    parent = model["parent"]
    walls = model["walls"]
    controls = model["controls"]

    @lru_cache(maxsize=50000)
    def cached_arm_cells(arm):
        return arm_cells(arm)

    def transform(s, i, mode, a, b):
        if i >= count:
            x, y = s[i]
            if mode == "move":
                s[i] = (x + a, y + b)
            else:
                s[i] = (a + y - b, b - x + a)
        elif types[i] == "arm":
            x, y, d, length = s[i]
            if mode == "move":
                s[i] = (x + a, y + b, d, length)
            else:
                s[i] = (a + y - b, b - x + a,
                        (d - 1) % 4, length)
        else:
            if mode == "move":
                s[i] = tuple(sorted(
                    (x + a, y + b) for x, y in s[i]
                ))
            else:
                s[i] = tuple(sorted(
                    (a + y - b, b - x + a) for x, y in s[i]
                ))
        for j in children[i]:
            transform(s, j, mode, a, b)

    def collides(s):
        occupied = set(walls)
        for i in range(count):
            if types[i] == "arm":
                points = cached_arm_cells(s[i])
            else:
                points = s[i]
            for x, y in points:
                if x < 0 or y < 0 or x >= BOARD or y >= BOARD:
                    return True
                if (x, y) in occupied:
                    return True
                occupied.add((x, y))
        return False

    def move(state, action):
        color, kind, _, _ = controls[action]
        s = list(state)
        changed = False

        for i in range(count):
            if types[i] != "arm" or colors[i] != color:
                continue

            x, y, d, length = s[i]
            if kind == "rotate":
                turns = 1
                p = parent[i]
                if p is not None and types[p] == "arm":
                    if d == (s[p][2] - 1) % 4:
                        turns = 2
                for _ in range(turns):
                    transform(s, i, "rotate", x, y)
                changed = True
            else:
                delta = 1 if kind == "longer" else -1
                if length + delta < 1 or length + delta > BOARD:
                    continue
                s[i] = (x, y, d, length + delta)
                dx, dy = DIRS[d]
                for j in children[i]:
                    transform(s, j, "move", dx * delta, dy * delta)
                changed = True

        if not changed:
            return None
        result = tuple(s)
        if result == state or collides(result):
            return None
        return result

    return move


def target_distances(model):
    walls = model["walls"]
    tables = []
    for target in model["targets"]:
        dist = {target: 0}
        queue = deque([target])
        while queue:
            x, y = queue.popleft()
            nd = dist[(x, y)] + 1
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if (0 <= q[0] < BOARD and 0 <= q[1] < BOARD
                        and q not in walls and q not in dist):
                    dist[q] = nd
                    queue.append(q)
        tables.append(dist)
    return tables


def search(model, start, blocked, visits):
    controls = model["controls"]
    count = model["count"]
    nmarkers = model["markers"]
    targets = model["targets"]

    if not controls or not targets or nmarkers < len(targets):
        return []

    move = make_simulator(model)
    tables = target_distances(model)

    @lru_cache(maxsize=60000)
    def heuristic(marker_positions):
        costs = {0: 0}
        for ti, (tx, ty) in enumerate(targets):
            next_costs = {}
            for mask, cost in costs.items():
                for i, (x, y) in enumerate(marker_positions):
                    if mask & (1 << i):
                        continue
                    dist = tables[ti].get(
                        (x, y), 30 + abs(x - tx) + abs(y - ty)
                    )
                    nm = mask | (1 << i)
                    nc = cost + dist
                    if nc < next_costs.get(nm, 10**9):
                        next_costs[nm] = nc
            costs = next_costs
        return min(costs.values()) if costs else 10**6

    def h(state):
        return heuristic(tuple(state[count:count + nmarkers]))

    if h(start) == 0:
        return []

    deadline = time.monotonic() + 4.4
    serial = 0
    heap = [(1.7 * h(start), 0, serial, start)]
    best_cost = {start: 0}
    previous = {}
    best_state = None
    best_rank = float("inf")
    expanded = 0

    while heap and expanded < 45000 and time.monotonic() < deadline:
        _, cost, _, state = heapq.heappop(heap)
        if best_cost.get(state) != cost:
            continue

        expanded += 1
        dh = h(state)
        if dh == 0:
            best_state = state
            break

        if state != start:
            rank = dh + 3.0 * min(5, visits.get(state, 0)) + 0.025 * cost
            if rank < best_rank:
                best_rank = rank
                best_state = state

        for action in range(len(controls)):
            if (state, action) in blocked:
                continue
            ns = move(state, action)
            if ns is None:
                continue

            revisit = min(3, visits.get(ns, 0))
            new_cost = cost + 1 + revisit
            if new_cost >= best_cost.get(ns, 10**9):
                continue
            best_cost[ns] = new_cost
            previous[ns] = (state, action)
            serial += 1
            priority = new_cost + 1.7 * h(ns)
            heapq.heappush(heap, (priority, new_cost, serial, ns))

    if best_state is None:
        return []

    path = []
    state = best_state
    while state != start:
        if state not in previous:
            return []
        state, action = previous[state]
        path.append(action)
    path.reverse()
    return path


def act(obs, available, memory):
    obs = np.asarray(obs)

    if "ACTION6" not in available:
        if "RESET" in available:
            return ("RESET", None, None)
        if available:
            return (available[0], None, None)
        return ("RESET", None, None)

    model = memory.get("model")
    state = memory.get("state")
    plan = memory.get("plan", [])
    blocked = memory.setdefault("blocked", set())
    visits = memory.setdefault("visits", {})

    if model is None or state is None:
        model, state = parse_frame(obs)
        plan = []
    elif not frame_matches(obs, model, state):
        old_state = memory.get("previous")
        last_action = memory.get("last_action")
        if old_state is not None and frame_matches(obs, model, old_state):
            if last_action is not None:
                blocked.add((old_state, last_action))
            state = old_state
        else:
            model, state = parse_frame(obs)
            blocked.clear()
            visits.clear()
        plan = []

    visits[state] = visits.get(state, 0) + 1
    if len(visits) > 3000:
        visits.clear()
        visits[state] = 1

    if not plan:
        plan = search(model, state, blocked, visits)

    if not plan:
        fresh_model, fresh_state = parse_frame(obs)
        if (fresh_state != state or
                fresh_model["controls"] != model["controls"]):
            model, state = fresh_model, fresh_state
            blocked.clear()
            visits.clear()
            plan = search(model, state, blocked, visits)

    if not plan:
        memory["model"] = model
        memory["state"] = state
        memory["plan"] = []
        if "RESET" in available and memory.get("resets", 0) < 1:
            memory.clear()
            memory["resets"] = 1
            return ("RESET", None, None)
        if model["controls"]:
            _, _, x, y = model["controls"][0]
            return ("ACTION6", int(y), int(x))
        return ("ACTION6", 0, 0)

    action = plan[0]
    next_state = make_simulator(model)(state, action)

    if next_state is None:
        blocked.add((state, action))
        memory["model"] = model
        memory["state"] = state
        memory["plan"] = []
        return act(obs, available, memory)

    _, _, x, y = model["controls"][action]
    memory["model"] = model
    memory["previous"] = state
    memory["last_action"] = action
    memory["state"] = next_state
    memory["plan"] = plan[1:]
    return ("ACTION6", int(y), int(x))

import math
import time
import heapq
from collections import deque

import numpy as np


def _components(a, mask):
    h, w = a.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = 1
        q = [(y, x)]
        pts = []
        while q:
            r, c = q.pop()
            pts.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                if (0 <= rr < h and 0 <= cc < w
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = 1
                    q.append((rr, cc))
        result.append(pts)
    return result


def _box(pts):
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return min(ys), max(ys) + 1, min(xs), max(xs) + 1


def _find_players(frame):
    # A player consists of a colour-4 arrow touching a coloured strip.
    excluded = {0, 1, 3, 4, 5, 7, 9, 12}
    colour_components = []
    label = np.full((64, 64), -1, dtype=np.int32)
    for colour in range(16):
        if colour in excluded:
            continue
        for pts in _components(frame, frame == colour):
            idx = len(colour_components)
            colour_components.append((colour, pts))
            for r, c in pts:
                label[r, c] = idx

    players = []
    for body in _components(frame, frame == 4):
        if len(body) < 2:
            continue
        adjacent = set()
        for r, c in body:
            for rr, cc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                if 0 <= rr < 64 and 0 <= cc < 64:
                    idx = int(label[rr, cc])
                    if idx >= 0:
                        adjacent.add(idx)
        if not adjacent:
            continue
        # Player strips are small; prefer the closest-sized adjacent component.
        idx = min(adjacent, key=lambda i: len(colour_components[i][1]))
        colour, strip = colour_components[idx]
        if len(strip) > 128:
            continue
        br = sum(p[0] for p in body) / len(body)
        bc = sum(p[1] for p in body) / len(body)
        sr = sum(p[0] for p in strip) / len(strip)
        sc = sum(p[1] for p in strip) / len(strip)
        dr, dc = sr - br, sc - bc
        if abs(dr) >= abs(dc):
            direction = "down" if dr > 0 else "up"
        else:
            direction = "right" if dc > 0 else "left"
        players.append((colour, body, strip, idx, direction))
    return players, colour_components


def _transform(frame, direction):
    if direction == "down":
        return frame.copy()
    if direction == "up":
        return frame[::-1, :].copy()
    if direction == "right":
        return frame.T.copy()
    return frame.T[::-1, :].copy()


def _to_normal(r, c, direction):
    if direction == "down":
        return r, c
    if direction == "up":
        return 63 - r, c
    if direction == "right":
        return c, r
    return 63 - c, r


def _from_normal(v, u, direction):
    if direction == "down":
        return v, u
    if direction == "up":
        return 63 - v, u
    if direction == "right":
        return u, v
    return u, 63 - v


def _runs(values):
    out = []
    start = None
    for i, value in enumerate(list(values) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def _build(frame):
    f = np.array(frame, dtype=np.int16, copy=True)
    # The first display row is the action counter, not the game world.
    f[0, :] = 3
    raw_players, coloured = _find_players(f)
    if not raw_players:
        return None

    directions = [p[4] for p in raw_players]
    direction = max(set(directions), key=directions.count)
    raw_players = [p for p in raw_players if p[4] == direction]
    a = _transform(f, direction)

    buttons = []
    for pts in _components(a, a == 9):
        v0, v1, u0, u1 = _box(pts)
        if (2 <= v1 - v0 <= 8 and 2 <= u1 - u0 <= 8
                and len(pts) >= 4):
            buttons.append((v0, v1, u0, u1))
    if not buttons:
        return None
    sizes = sorted(min(b[1] - b[0], b[3] - b[2]) for b in buttons)
    step = sizes[len(sizes) // 2]

    # Long strips parallel to gravity separate adjacent platform lanes.
    wall_mask = (a == 5) | (a == 1) | (a == 12)
    counts = wall_mask.sum(axis=0)
    wall_runs = _runs(counts >= 2 * step + 1)
    walls = []
    for u0, u1 in wall_runs:
        if u1 - u0 <= step + 1:
            walls.append((u0, u1))
        else:
            # A wide run can contain multiple adjacent wall strips.
            pos = u0
            while pos + step <= u1:
                walls.append((pos, pos + step))
                pos += step
    if not walls:
        return None
    walls.sort()

    lanes = []
    edge = 0
    for u0, u1 in walls:
        if u0 > edge:
            lanes.append((edge, u0))
        edge = max(edge, u1)
    if edge < 64:
        lanes.append((edge, 64))

    # Each visible rectangular zero-region is a platform. A lane may
    # contain several platforms separated by a fixed transverse barrier.
    floors = []
    barriers = {}
    for li, (u0, u1) in enumerate(lanes):
        width = u1 - u0
        if width <= 0:
            continue
        zeros = (a[:, u0:u1] == 0).sum(axis=1)
        threshold = max(1, (width - step) // 2)
        occupied = zeros >= threshold
        for v0, v1 in _runs(occupied):
            if v1 - v0 >= 1:
                floors.append((li, v0, v1))

        transverse = (a[:, u0:u1] == 5).sum(axis=1)
        barriers[li] = [
            (v0, v1) for v0, v1 in _runs(transverse >= max(1, width - 1))
            if v1 - v0 <= 2 * step
        ]

    if not floors:
        return None

    def adjacent_wall(lane, wall):
        l0, l1 = lanes[lane]
        return l1 == wall[0] or l0 == wall[1]

    wall_start = []
    for u0, u1 in walls:
        ys = np.nonzero(wall_mask[:, u0:u1])[0]
        wall_start.append(int(ys.min()) if len(ys) else 0)

    limits = []
    for li, start, end in floors:
        wl = [wall_start[j] for j, w in enumerate(walls)
              if adjacent_wall(li, w)]
        limit = max(wl) if wl else 0
        fixed = [b1 for b0, b1 in barriers.get(li, [])
                 if b1 <= start and b1 < end]
        barrier = max(fixed) if fixed else None
        if barrier is not None:
            limit = max(limit, barrier)
        limits.append((limit, barrier))

    def lane_at(u):
        for i, (u0, u1) in enumerate(lanes):
            if u0 <= u < u1:
                return i
        return None

    def floor_for(li, depth):
        candidates = [i for i, (lane, start, end) in enumerate(floors)
                      if lane == li and end >= depth]
        if not candidates:
            return None
        return min(candidates, key=lambda i: floors[i][1])

    player_floors = []
    player_colours = []
    player_depths = []
    player_component_ids = set()
    for colour, body, strip, idx, _ in raw_players:
        points = [_to_normal(r, c, direction) for r, c in body + strip]
        v0, v1, u0, u1 = _box(points)
        li = lane_at((u0 + u1 - 1) // 2)
        if li is None:
            continue
        candidates = [i for i, (lane, start, end) in enumerate(floors)
                      if lane == li]
        if not candidates:
            continue
        fi = min(candidates, key=lambda i: abs(floors[i][1] - v1))
        if abs(floors[fi][1] - v1) > step:
            continue
        player_floors.append(fi)
        player_colours.append(colour)
        player_depths.append(v1 - v0)
        player_component_ids.add(idx)

    if not player_floors:
        return None

    goals = []
    for idx, (colour, pts) in enumerate(coloured):
        if idx in player_component_ids or colour not in player_colours:
            continue
        points = [_to_normal(r, c, direction) for r, c in pts]
        v0, v1, u0, u1 = _box(points)
        for wi, wall in enumerate(walls):
            if u0 < wall[1] and u1 > wall[0]:
                targets = tuple(
                    i for i, (li, start, end) in enumerate(floors)
                    if adjacent_wall(li, wall) and end >= v1
                )
                if targets:
                    goals.append((colour, v1, targets))

    if any(not any(g[0] == colour for g in goals)
           for colour in player_colours):
        return None

    controls = []
    for v0, v1, u0, u1 in buttons:
        for wall in walls:
            if u1 == wall[0]:
                own_u, other_u = u0, wall[1]
            elif u0 == wall[1]:
                own_u, other_u = u0, wall[0] - 1
            else:
                continue
            own_lane, other_lane = lane_at(own_u), lane_at(other_u)
            if own_lane is None or other_lane is None:
                continue
            receiver = floor_for(own_lane, v0)
            donor = floor_for(other_lane, v0)
            if receiver is None or donor is None or receiver == donor:
                continue
            r, c = _from_normal((v0 + v1 - 1) // 2,
                                (u0 + u1 - 1) // 2, direction)
            controls.append((donor, receiver, int(r), int(c)))
            break

    gates = []
    for colour in (1, 12):
        for pts in _components(a, a == colour):
            v0, v1, u0, u1 = _box(pts)
            if len(pts) < step * step:
                continue
            for wall in walls:
                if u0 >= wall[0] and u1 <= wall[1]:
                    left = lane_at(wall[0] - 1)
                    right = lane_at(wall[1])
                    if left is None or right is None:
                        break
                    f1, f2 = floor_for(left, v1), floor_for(right, v1)
                    if f1 is not None and f2 is not None:
                        r, c = _from_normal((v0 + v1 - 1) // 2,
                                            (u0 + u1 - 1) // 2, direction)
                        gates.append((f1, f2, v1, int(r), int(c)))
                    break

    return (floors, limits, controls, gates, goals, step,
            tuple(player_floors), tuple(player_colours),
            tuple(player_depths))


def _plan(model):
    (floors, limits, controls, gates, goals, step,
     initial_players, colours, depths) = model
    n = len(floors)
    initial_starts = tuple(f[1] for f in floors)
    ends = tuple(f[2] for f in floors)

    # Minimum number of gate crossings between each pair of platforms.
    graph = [[] for _ in range(n)]
    for x, y, _, _, _ in gates:
        graph[x].append(y)
        graph[y].append(x)
    distance = []
    for source in range(n):
        d = [999] * n
        d[source] = 0
        q = deque([source])
        while q:
            x = q.popleft()
            for y in graph[x]:
                if d[y] == 999:
                    d[y] = d[x] + 1
                    q.append(y)
        distance.append(d)

    def heuristic(starts, players):
        result = 0
        for p, colour in zip(players, colours):
            best = 999
            for gc, depth, targets in goals:
                if gc != colour:
                    continue
                for t in targets:
                    crossings = distance[p][t]
                    if crossings < 999:
                        cost = ((abs(starts[p] - depth) + step - 1) // step
                                + crossings)
                        best = min(best, cost)
            if best == 999:
                return 999
            result = max(result, best)
        return result

    def solved(starts, players):
        for p, colour in zip(players, colours):
            if not any(gc == colour and starts[p] == depth and p in targets
                       for gc, depth, targets in goals):
                return False
        return True

    initial = (initial_starts, initial_players)
    h = heuristic(*initial)
    if h == 999:
        return None

    queue = [(h, 0, 0, initial)]
    best = {initial: 0}
    parent = {}
    serial = 0
    deadline = time.monotonic() + 7.5
    expanded = 0

    while queue and expanded < 180000:
        if expanded % 1024 == 0 and time.monotonic() > deadline:
            break
        _, cost, _, state = heapq.heappop(queue)
        if best.get(state) != cost:
            continue
        starts, players = state
        if solved(starts, players):
            path = []
            while state != initial:
                previous, click = parent[state]
                path.append(click)
                state = previous
            path.reverse()
            return path

        expanded += 1
        if cost >= 49:
            continue

        successors = []
        for donor, receiver, row, col in controls:
            if starts[donor] >= ends[donor]:
                continue
            limit, barrier = limits[receiver]
            if barrier is not None and limit == barrier:
                for p, depth in zip(players, depths):
                    if p == receiver:
                        limit = max(limit, barrier + depth)
            if starts[receiver] <= limit:
                continue
            ns = list(starts)
            ns[donor] += step
            ns[receiver] -= step
            if ns[donor] > ends[donor] + step - 1:
                continue
            successors.append(((tuple(ns), players), (row, col)))

        for x, y, gate_depth, row, col in gates:
            if starts[x] != gate_depth or starts[y] != gate_depth:
                continue
            if x not in players and y not in players:
                continue
            ps = tuple(y if p == x else x if p == y else p for p in players)
            successors.append(((starts, ps), (row, col)))

        for new_state, click in successors:
            new_cost = cost + 1
            if new_cost >= best.get(new_state, 10**9):
                continue
            nh = heuristic(*new_state)
            if nh == 999:
                continue
            best[new_state] = new_cost
            parent[new_state] = (state, click)
            serial += 1
            heapq.heappush(queue, (new_cost + nh, new_cost, serial, new_state))
    return None


def act(obs, available, memory):
    plan = memory.get("plan")
    if plan:
        row, col = plan.pop(0)
        return "ACTION6", int(row), int(col)

    model = _build(obs)
    if model is not None:
        plan = _plan(model)
        if plan:
            row, col = plan.pop(0)
            memory["plan"] = plan
            return "ACTION6", int(row), int(col)

    # If reconstruction or search failed, try visible buttons rather than
    # issuing an invalid action. Cycle through them across calls.
    a = np.asarray(obs)
    buttons = _components(a, a == 9)
    candidates = []
    for pts in buttons:
        r0, r1, c0, c1 = _box(pts)
        if 2 <= r1 - r0 <= 8 and 2 <= c1 - c0 <= 8:
            candidates.append(((r0 + r1 - 1) // 2,
                               (c0 + c1 - 1) // 2))
    if candidates:
        index = memory.get("fallback_index", 0)
        memory["fallback_index"] = index + 1
        row, col = candidates[index % len(candidates)]
        return "ACTION6", int(row), int(col)

    if "RESET" in available:
        return "RESET", None, None
    return "ACTION6", 32, 32

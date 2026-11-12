import heapq
import time
from collections import deque

import numpy as np


def _components(a, colour):
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or a[y, x] != colour:
                continue
            seen[y, x] = True
            stack = [(y, x)]
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < h and 0 <= cc < w
                            and not seen[rr, cc]
                            and a[rr, cc] == colour):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(component)
    return result


def _bounds(component):
    ys = [p[0] for p in component]
    xs = [p[1] for p in component]
    return min(ys), max(ys) + 1, min(xs), max(xs) + 1


def _original_point(rotation, row, col):
    if rotation == 0:
        return row, col
    if rotation == 1:
        return col, 63 - row
    if rotation == 2:
        return 63 - row, 63 - col
    return 63 - col, row


def _point_in_component(component, rotation):
    y0, y1, x0, x1 = _bounds(component)
    cy = (y0 + y1 - 1) / 2
    cx = (x0 + x1 - 1) / 2
    p = min(component, key=lambda q: abs(q[0] - cy) + abs(q[1] - cx))
    return _original_point(rotation, p[0], p[1])


def _rotation(a):
    # Identify which side of a character's yellow glyph carries its
    # coloured strip. That side is the direction of gravity.
    votes = [0, 0, 0, 0]
    directions = ((1, 0), (0, -1), (-1, 0), (0, 1))
    for component in _components(a, 4):
        if not 3 <= len(component) <= 150:
            continue
        for y, x in component:
            for i, (dy, dx) in enumerate(directions):
                yy, xx = y + dy, x + dx
                if 0 <= yy < 64 and 0 <= xx < 64:
                    if a[yy, xx] in (11, 14):
                        votes[i] += 1
    if not any(votes):
        return None
    return max(range(4), key=lambda i: votes[i])


def _runs(values):
    if not values:
        return []
    out = []
    start = last = values[0]
    for v in values[1:]:
        if v == last + 1:
            last = v
        else:
            out.append((start, last + 1))
            start = last = v
    out.append((start, last + 1))
    return out


def _parse(obs):
    original = np.asarray(obs, dtype=np.int16)
    if original.shape != (64, 64):
        return None

    a = original.copy()
    a[0, :] = 3
    k = _rotation(a)
    if k is None:
        return None
    a = np.rot90(a, k)

    # Walls run vertically after rotation. Their colour may be interrupted
    # by gates or goal markers.
    wall_mask = []
    colours = (5, 1, 12, 11, 14)
    for x in range(64):
        good = np.isin(a[:, x], colours)
        longest = current = 0
        for value in good:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        wall_mask.append(longest >= 7)

    walls = []
    x = 0
    while x < 64:
        if not wall_mask[x]:
            x += 1
            continue
        left = x
        while x < 64 and wall_mask[x]:
            x += 1
        right = x
        if right - left <= 8:
            ys = np.where(np.isin(a[:, left:right], colours))[0]
            if len(ys):
                walls.append((left, right, int(ys.min())))

    lanes = []
    previous = 0
    for left, right, _ in walls:
        if left > previous:
            lanes.append((previous, left))
        previous = right
    if previous < 64:
        lanes.append((previous, 64))
    if not lanes:
        return None

    def lane_at(x):
        for i, (left, right) in enumerate(lanes):
            if left <= x < right:
                return i
        return None

    def neighbours(wall_index):
        left, right, _ = walls[wall_index]
        li = ri = None
        for i, (lo, hi) in enumerate(lanes):
            if hi == left:
                li = i
            if lo == right:
                ri = i
        return li, ri

    # Locate characters and their supporting platform fronts.
    players = []
    for component in _components(a, 4):
        if not 3 <= len(component) <= 150:
            continue
        y0, y1, x0, x1 = _bounds(component)
        candidates = []
        for yy in range(y1, min(64, y1 + 6)):
            row = []
            for xx in range(x0, x1):
                if a[yy, xx] in (11, 14):
                    row.append((yy, xx, int(a[yy, xx])))
            if row:
                candidates.extend(row)
            elif candidates:
                break
        if not candidates:
            continue
        counts = {11: 0, 14: 0}
        for _, _, colour in candidates:
            counts[colour] += 1
        colour = max(counts, key=counts.get)
        strip = [(yy, xx) for yy, xx, cc in candidates if cc == colour]
        lane = lane_at((x0 + x1 - 1) // 2)
        if lane is None or not strip:
            continue
        front = max(yy for yy, _ in strip) + 1
        depth = front - y0
        players.append((colour, lane, front, depth, x0))
    players.sort(key=lambda p: (p[0], p[1], p[4], p[2]))
    if not players:
        return None

    # Fixed crosswise floors divide a lane into independently movable
    # platform sections.
    floors_by_lane = []
    for left, right in lanes:
        width = right - left
        threshold = max(1, int(np.ceil(width * 0.75)))
        counts = np.sum(a[:, left:right] == 5, axis=1)
        rows = [int(v) for v in np.flatnonzero(counts >= threshold)]
        runs = [(lo, hi) for lo, hi in _runs(rows)
                if 1 <= hi - lo <= 6]
        floors_by_lane.append(runs)

    # Each segment is (lane, interval start, fixed bottom, floor before).
    segments = []
    lane_segments = [[] for _ in lanes]
    for lane, floor_runs in enumerate(floors_by_lane):
        start = 0
        preceding_floor = None
        for lo, hi in floor_runs:
            if lo > start:
                idx = len(segments)
                segments.append((lane, start, lo, preceding_floor))
                lane_segments[lane].append(idx)
            start = hi
            preceding_floor = (lo, hi)
        if start < 64:
            idx = len(segments)
            segments.append((lane, start, 64, preceding_floor))
            lane_segments[lane].append(idx)

    if not segments:
        return None

    tops = []
    limits = []
    for lane, start, bottom, preceding_floor in segments:
        left, right = lanes[lane]
        centre = (left + right - 1) // 2

        # The centre column remains visible when buttons cover the edges
        # of a platform. This also avoids mistaking gate brackets for it.
        black_rows = [
            y for y in range(start, bottom) if a[y, centre] == 0
        ]
        if not black_rows and right - left <= 3:
            black_rows = [
                y for y in range(start, bottom)
                if np.any(a[y, left:right] == 0)
            ]
        top = black_rows[0] if black_rows else bottom
        tops.append(top)

        adjacent = [
            wall_top for wl, wr, wall_top in walls
            if wr == left or wl == right
        ]
        wall_limit = max(adjacent) if adjacent else 0
        if preceding_floor is not None:
            wall_limit = max(wall_limit, preceding_floor[1])
        limits.append(wall_limit)

    # Match each character to the segment directly beneath it.
    player_segments = []
    for colour, lane, front, depth, _ in players:
        candidates = [
            i for i in lane_segments[lane]
            if segments[i][1] <= front <= segments[i][2]
        ]
        if not candidates:
            return None
        seg = min(candidates, key=lambda i: abs(tops[i] - front))
        tops[seg] = front
        player_segments.append(seg)

    # Goals are colour strips over a wall. Character and target templates
    # have the same gravity-axis depth, so matching their front edges
    # matches the simulator's position test.
    goals = []
    for wi, (left, right, _) in enumerate(walls):
        li, ri = neighbours(wi)
        for colour in (11, 14):
            rows = [
                int(v) for v in np.flatnonzero(
                    np.any(a[:, left:right] == colour, axis=1)
                )
            ]
            for _, end in _runs(rows):
                for lane in (li, ri):
                    if lane is None:
                        continue
                    for seg in lane_segments[lane]:
                        _, start, bottom, _ = segments[seg]
                        if start <= end <= bottom:
                            goals.append((colour, seg, end))

    # Each gate connects the two platform sections whose fronts can meet
    # its gravity-facing edge.
    gates = []
    for colour in (1, 12):
        for component in _components(a, colour):
            y0, y1, x0, x1 = _bounds(component)
            if y1 - y0 < 5:
                continue
            cx = (x0 + x1 - 1) // 2
            wi = next(
                (j for j, (l, r, _) in enumerate(walls) if l <= cx < r),
                None
            )
            if wi is None:
                continue
            li, ri = neighbours(wi)
            if li is None or ri is None:
                continue
            left_candidates = [
                s for s in lane_segments[li]
                if segments[s][1] <= y1 <= segments[s][2]
            ]
            right_candidates = [
                s for s in lane_segments[ri]
                if segments[s][1] <= y1 <= segments[s][2]
            ]
            if not left_candidates or not right_candidates:
                continue
            for ls in left_candidates:
                for rs in right_candidates:
                    gates.append((ls, rs, y1,
                                  _point_in_component(component, k)))

    # Buttons transfer a step from the other side of the adjacent wall
    # into the platform on the button's side. When a lane contains several
    # sections, the first section extending beyond the button is selected.
    buttons = []
    sizes = []
    for component in _components(a, 9):
        y0, y1, x0, x1 = _bounds(component)
        cx = (x0 + x1 - 1) / 2
        lane = lane_at(int(cx))
        if lane is None:
            continue
        possible = []
        for wi, (wl, wr, _) in enumerate(walls):
            li, ri = neighbours(wi)
            if li == lane and ri is not None:
                possible.append((abs(wl - cx), ri))
            elif ri == lane and li is not None:
                possible.append((abs(cx - wr), li))
        if not possible:
            continue
        distance, other_lane = min(possible)
        if distance > max(7, 2 * (x1 - x0)):
            continue

        def section_for_button(which_lane):
            candidates = [
                s for s in lane_segments[which_lane]
                if segments[s][2] > y0
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda s: segments[s][1])

        receiver = section_for_button(lane)
        donor = section_for_button(other_lane)
        if receiver is None or donor is None:
            continue
        buttons.append((receiver, donor, _point_in_component(component, k)))
        sizes.append(min(y1 - y0, x1 - x0))

    if not buttons or not goals:
        return None

    step = max(set(sizes), key=sizes.count)
    if step <= 0:
        return None

    remaining = int(round(int(np.sum(original[0] == 7)) * 50 / 64))
    remaining = max(1, min(50, remaining))

    return {
        "n": len(segments),
        "state": tuple(tops) + tuple(player_segments),
        "colours": tuple(p[0] for p in players),
        "depths": tuple(p[3] for p in players),
        "segments": tuple(segments),
        "limits": tuple(limits),
        "step": step,
        "direction": k,
        "goals": tuple(goals),
        "gates": tuple(gates),
        "buttons": tuple(buttons),
        "remaining": remaining,
    }


def _distance_tables(model):
    n = model["n"]
    step = model["step"]
    tables = {}
    for colour in set(model["colours"]):
        dist = [[10000] * 65 for _ in range(n)]
        queue = deque()
        for gc, seg, front in model["goals"]:
            if gc == colour and 0 <= front <= 64:
                if dist[seg][front] != 0:
                    dist[seg][front] = 0
                    queue.append((seg, front))
        while queue:
            seg, front = queue.popleft()
            nd = dist[seg][front] + 1
            for nf in (front - step, front + step):
                if 0 <= nf <= 64 and nd < dist[seg][nf]:
                    dist[seg][nf] = nd
                    queue.append((seg, nf))
            for left, right, gate_front, _ in model["gates"]:
                if front != gate_front:
                    continue
                other = right if seg == left else left if seg == right else None
                if other is not None and nd < dist[other][front]:
                    dist[other][front] = nd
                    queue.append((other, front))
        tables[colour] = dist
    return tables


def _search(model, visited):
    n = model["n"]
    colours = model["colours"]
    depths = model["depths"]
    count = len(colours)
    step = model["step"]
    segments = model["segments"]
    limits = model["limits"]
    buttons = model["buttons"]
    gates = model["gates"]
    direction = model["direction"]
    goalset = set(model["goals"])
    initial = model["state"]
    remaining = model["remaining"]
    tables = _distance_tables(model)

    def solved(state):
        for i, colour in enumerate(colours):
            seg = state[n + i]
            if (colour, seg, state[seg]) not in goalset:
                return False
        return True

    def heuristic(state):
        distances = []
        for i, colour in enumerate(colours):
            seg = state[n + i]
            front = state[seg]
            if not 0 <= front <= 64:
                return 10000.0
            distances.append(tables[colour][seg][front])
        if not distances:
            return 0.0
        m = max(distances)
        return m + 0.35 * (sum(distances) - m)

    def successors(state):
        for receiver, donor, point in buttons:
            donor_bottom = segments[donor][2]
            if state[donor] >= donor_bottom:
                continue

            limit = limits[receiver]
            floor = segments[receiver][3]
            if floor is not None:
                occupants = [
                    depths[i] for i in range(count)
                    if state[n + i] == receiver
                ]
                if direction == 1:
                    if occupants:
                        limit = max(limit, floor[1] + max(occupants))
                elif occupants:
                    limit = max(limits[receiver], floor[1] - max(occupants))

            if state[receiver] <= limit:
                continue

            nxt = list(state)
            nxt[donor] += step
            nxt[receiver] -= step
            if nxt[donor] > segments[donor][2] or nxt[receiver] < 0:
                continue
            yield tuple(nxt), point

        for left, right, front, point in gates:
            if state[left] != front or state[right] != front:
                continue
            nxt = list(state)
            changed = False
            for i in range(count):
                if nxt[n + i] == left:
                    nxt[n + i] = right
                    changed = True
                elif nxt[n + i] == right:
                    nxt[n + i] = left
                    changed = True
            if changed:
                yield tuple(nxt), point

    if solved(initial):
        return []

    initial_h = heuristic(initial)
    heap = [(1.4 * initial_h, initial_h, 0, 0, initial)]
    best_g = {initial: 0}
    parent = {}
    serial = 0
    best_state = initial
    best_value = initial_h
    result = None
    deadline = time.monotonic() + 7.0
    expanded = 0

    while heap and expanded < 120000 and time.monotonic() < deadline:
        _, h, _, g, state = heapq.heappop(heap)
        if best_g.get(state) != g:
            continue
        if solved(state):
            result = state
            break

        value = h + 0.05 * g
        if value < best_value:
            best_value = value
            best_state = state

        if g >= remaining:
            continue
        expanded += 1
        for nxt, point in successors(state):
            ng = g + 1
            if ng >= best_g.get(nxt, 10**9):
                continue
            best_g[nxt] = ng
            parent[nxt] = (state, point)
            nh = heuristic(nxt)
            serial += 1
            heapq.heappush(heap, (ng + 1.4 * nh, nh, serial, ng, nxt))

    if result is None and best_state != initial:
        result = best_state

    if result is None:
        options = list(successors(initial))
        if not options:
            return None
        options.sort(key=lambda p: (
            visited.get(str(p[0]), 0),
            heuristic(p[0])
        ))
        nxt, point = options[0]
        return [(point, nxt)]

    path = []
    while result != initial:
        prev, point = parent[result]
        path.append((point, result))
        result = prev
    path.reverse()
    return path


def _fallback(obs, memory):
    a = np.asarray(obs)
    choices = []
    for colour in (12, 9, 1):
        for component in _components(a, colour):
            if component:
                y0, y1, x0, x1 = _bounds(component)
                cy = (y0 + y1 - 1) / 2
                cx = (x0 + x1 - 1) / 2
                point = min(
                    component,
                    key=lambda p: abs(p[0] - cy) + abs(p[1] - cx)
                )
                choices.append(point)
    if not choices:
        return ("ACTION6", 32, 32)
    visits = memory.setdefault("fallback_visits", {})
    point = min(choices, key=lambda p: visits.get(str(p), 0))
    key = str(point)
    visits[key] = visits.get(key, 0) + 1
    return ("ACTION6", int(point[0]), int(point[1]))


def act(obs, available, memory):
    model = _parse(obs)
    if model is None:
        memory.pop("plan", None)
        memory.pop("expected", None)
        return _fallback(obs, memory)

    state = model["state"]
    visited = memory.setdefault("visited", {})
    key = str(state)
    visited[key] = visited.get(key, 0) + 1

    plan = memory.get("plan")
    expected = memory.get("expected")
    if plan and expected is not None and tuple(expected) == state:
        point, next_state = plan.pop(0)
        memory["plan"] = plan
        memory["expected"] = list(next_state)
        return ("ACTION6", int(point[0]), int(point[1]))

    path = _search(model, visited)
    if path:
        point, next_state = path[0]
        memory["plan"] = path[1:]
        memory["expected"] = list(next_state)
        return ("ACTION6", int(point[0]), int(point[1]))

    memory.pop("plan", None)
    memory.pop("expected", None)
    return _fallback(obs, memory)

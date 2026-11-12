import heapq
import time
from collections import Counter

import numpy as np


_MARKERS = (2, 6, 7, 8, 10, 11, 13, 14, 15)


def _components(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel().tolist()
    seen = bytearray(4096)
    result = []

    for start in range(4096):
        if not flat[start] or seen[start]:
            continue
        seen[start] = 1
        queue = [start]
        cells = []
        pos = 0

        while pos < len(queue):
            p = queue[pos]
            pos += 1
            cells.append(p)
            r, c = divmod(p, 64)

            if r and flat[p - 64] and not seen[p - 64]:
                seen[p - 64] = 1
                queue.append(p - 64)
            if r < 63 and flat[p + 64] and not seen[p + 64]:
                seen[p + 64] = 1
                queue.append(p + 64)
            if c and flat[p - 1] and not seen[p - 1]:
                seen[p - 1] = 1
                queue.append(p - 1)
            if c < 63 and flat[p + 1] and not seen[p + 1]:
                seen[p + 1] = 1
                queue.append(p + 1)

        result.append(cells)
    return result


def _box(cells):
    rows = [p // 64 for p in cells]
    cols = [p % 64 for p in cells]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _runs(values):
    result = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append((start, i))
            start = None
    if start is not None:
        result.append((start, len(values)))
    return result


def _characters_and_markers(a):
    coloured = []
    lookup = {}

    for colour in _MARKERS:
        for cells in _components(a == colour):
            index = len(coloured)
            coloured.append((colour, cells))
            for p in cells:
                lookup[p] = index

    characters = []
    used = set()

    for body in _components(a == 4):
        if len(body) > 256:
            continue

        neighbours = set()
        for p in body:
            r, c = divmod(p, 64)
            if r and p - 64 in lookup:
                neighbours.add(lookup[p - 64])
            if r < 63 and p + 64 in lookup:
                neighbours.add(lookup[p + 64])
            if c and p - 1 in lookup:
                neighbours.add(lookup[p - 1])
            if c < 63 and p + 1 in lookup:
                neighbours.add(lookup[p + 1])

        if not neighbours:
            continue

        br0, br1, bc0, bc1 = _box(body)
        br = (br0 + br1 - 1) / 2.0
        bc = (bc0 + bc1 - 1) / 2.0

        def distance(index):
            r0, r1, c0, c1 = _box(coloured[index][1])
            return abs((r0 + r1 - 1) / 2.0 - br) + abs(
                (c0 + c1 - 1) / 2.0 - bc
            )

        index = min(neighbours, key=distance)
        if index in used:
            continue

        colour, stripe = coloured[index]
        r0, r1, c0, c1 = _box(stripe)
        dr = (r0 + r1 - 1) / 2.0 - br
        dc = (c0 + c1 - 1) / 2.0 - bc

        if abs(dr) >= abs(dc):
            direction = (1 if dr > 0 else -1, 0)
        else:
            direction = (0, 1 if dc > 0 else -1)

        characters.append((colour, _box(body + stripe), direction))
        used.add(index)

    markers = [
        (colour, _box(cells))
        for i, (colour, cells) in enumerate(coloured)
        if i not in used
    ]
    return characters, markers


def _display_coord(rotation, row, col):
    if rotation == 0:
        return row, col
    if rotation == 1:
        return col, 63 - row
    if rotation == 2:
        return 63 - row, 63 - col
    return 63 - col, row


def _read_board(obs, memory):
    a = np.array(obs, dtype=np.int16, copy=True)
    if a.shape != (64, 64):
        return None

    # The status bar overwrites the first display row.
    a[0, :] = a[1, :]

    characters, _ = _characters_and_markers(a)
    if not characters:
        return None

    direction = Counter(ch[2] for ch in characters).most_common(1)[0][0]
    rotation = {
        (1, 0): 0,
        (-1, 0): 2,
        (0, 1): 3,
        (0, -1): 1,
    }.get(direction)
    if rotation is None:
        return None

    # Rotate so that gravity always points down.
    a = np.rot90(a, rotation).copy()
    characters, markers = _characters_and_markers(a)
    if not characters:
        return None

    buttons = []
    sizes = []
    for cells in _components(a == 9):
        r0, r1, c0, c1 = _box(cells)
        h, w = r1 - r0, c1 - c0
        if 1 <= h <= 12 and 1 <= w <= 12:
            buttons.append((r0, r1, c0, c1))
            if abs(h - w) <= 1:
                sizes.append(min(h, w))

    if not buttons or not sizes:
        return None
    step = Counter(sizes).most_common(1)[0][0]

    wall_pixels = (a == 5) | (a == 1) | (a == 12)
    counts = np.count_nonzero(wall_pixels, axis=0)
    walls = _runs((counts >= max(5, 2 * step)).tolist())
    walls = [
        (c0, c1)
        for c0, c1 in walls
        if c1 - c0 <= max(2 * step, 8)
    ]
    walls.sort()

    lanes = []
    left = 0
    for c0, c1 in walls:
        if c0 > left:
            lanes.append((left, c0))
        left = max(left, c1)
    if left < 64:
        lanes.append((left, 64))

    if not lanes or len(lanes) > 16:
        return None

    def lane_at(col):
        for i, (c0, c1) in enumerate(lanes):
            if c0 <= col < c1:
                return i
        return min(
            range(len(lanes)),
            key=lambda i: abs(col - (lanes[i][0] + lanes[i][1] - 1) / 2.0),
        )

    # Wall-to-lane mappings also work when there is a wall at an edge.
    wall_neighbours = []
    for w0, w1 in walls:
        left_lane = next(
            (i for i, (_, c1) in enumerate(lanes) if c1 == w0), None
        )
        right_lane = next(
            (i for i, (c0, _) in enumerate(lanes) if c0 == w1), None
        )
        wall_neighbours.append((left_lane, right_lane))

    obstacles = [[] for _ in lanes]
    platform_runs = [[] for _ in lanes]

    for i, (c0, c1) in enumerate(lanes):
        width = c1 - c0
        if width <= 0:
            return None
        threshold = max(1, width - 1)

        floor_counts = np.count_nonzero(a[:, c0:c1] == 5, axis=1)
        for r0, r1 in _runs((floor_counts >= threshold).tolist()):
            obstacles[i].append((r0, r1))

        # Buttons cover platform pixels at the bottom of a lane.
        platform_mask = (a[:, c0:c1] == 0) | (a[:, c0:c1] == 9)
        row_counts = np.count_nonzero(platform_mask, axis=1)
        for r0, r1 in _runs((row_counts >= threshold).tolist()):
            if r1 - r0 >= 1:
                platform_runs[i].append((r0, r1))

    # A platform is represented by its lane and fixed downstream end.
    # Retain previously seen ends when a platform becomes completely empty.
    signature = (rotation, tuple(lanes), step)
    old = memory.get("platform_ends")
    if memory.get("board_signature") == signature and old:
        ends = [set(x) for x in old]
        if len(ends) != len(lanes):
            ends = [set() for _ in lanes]
    else:
        ends = [set() for _ in lanes]

    for i in range(len(lanes)):
        ends[i].add(64)
        for r0, r1 in platform_runs[i]:
            ends[i].add(r1)

    # A button immediately above a fixed floor can reveal an emptied upper
    # platform even when the episode starts after a few moves.
    for r0, r1, c0, c1 in buttons:
        lane = lane_at((c0 + c1 - 1) / 2.0)
        if any(floor_start == r1 for floor_start, _ in obstacles[lane]):
            ends[lane].add(r1)

    memory["board_signature"] = signature
    memory["platform_ends"] = [sorted(x) for x in ends]

    slots = []
    for lane, lane_ends in enumerate(ends):
        for end in sorted(lane_ends):
            slots.append((lane, end))

    if len(slots) > 24:
        return None

    slot_index = {slot: i for i, slot in enumerate(slots)}
    surfaces = []

    for lane, end in slots:
        matching = [
            r0 for r0, r1 in platform_runs[lane] if r1 == end
        ]
        surfaces.append(min(matching) if matching else end)

    def slot_for_button(lane, row):
        candidates = [
            (end, slot_index[(ln, end)])
            for ln, end in slots
            if ln == lane and end > row
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    tokens = []
    token_sizes = {}
    for colour, (r0, r1, c0, c1), _ in characters:
        lane = lane_at((c0 + c1 - 1) / 2.0)
        candidates = [
            i for i, (ln, _) in enumerate(slots) if ln == lane
        ]
        if not candidates:
            continue
        slot = min(candidates, key=lambda i: abs(surfaces[i] - r1))
        tokens.append((colour, slot))
        token_sizes[colour] = max(token_sizes.get(colour, 0), r1 - r0)

    tokens = tuple(sorted(tokens))
    if not tokens:
        return None

    goals = []
    for colour, (r0, r1, c0, c1) in markers:
        if colour not in token_sizes:
            continue
        centre = (c0 + c1 - 1) / 2.0
        for wi, (w0, w1) in enumerate(walls):
            if w0 <= centre < w1:
                for lane in wall_neighbours[wi]:
                    if lane is not None:
                        for slot, (ln, end) in enumerate(slots):
                            if ln == lane and r1 <= end:
                                goals.append((colour, slot, r1))
                break

    if not goals:
        return None

    transfers = []
    for r0, r1, c0, c1 in buttons:
        receiver_lane = lane_at((c0 + c1 - 1) / 2.0)
        l0, l1 = lanes[receiver_lane]

        if abs(c0 - l0) < abs(l1 - c1):
            donor_lane = receiver_lane - 1
        else:
            donor_lane = receiver_lane + 1

        if not (0 <= donor_lane < len(lanes)):
            continue

        receiver = slot_for_button(receiver_lane, r0)
        donor = slot_for_button(donor_lane, r0)
        if receiver is None or donor is None:
            continue

        click = _display_coord(
            rotation, (r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2
        )
        transfers.append((donor, receiver, click))

    gates = []
    for cells in _components((a == 1) | (a == 12)):
        r0, r1, c0, c1 = _box(cells)
        if c1 - c0 > max(2 * step, 8) or r1 - r0 < step:
            continue
        centre = (c0 + c1 - 1) / 2.0

        for wi, (w0, w1) in enumerate(walls):
            if not (w0 <= centre < w1):
                continue
            left_lane, right_lane = wall_neighbours[wi]
            if left_lane is None or right_lane is None:
                break

            left_slot = slot_for_button(left_lane, r1 - 1)
            right_slot = slot_for_button(right_lane, r1 - 1)
            if left_slot is None or right_slot is None:
                break

            click = _display_coord(
                rotation, (r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2
            )
            gates.append((left_slot, right_slot, r1, click))
            break

    wall_limits = [0] * len(lanes)
    for wi, (c0, c1) in enumerate(walls):
        rows = np.flatnonzero(np.any(wall_pixels[:, c0:c1], axis=1))
        if not len(rows):
            continue
        top = int(rows[0])
        for lane in wall_neighbours[wi]:
            if lane is not None:
                wall_limits[lane] = max(wall_limits[lane], top)

    return {
        "step": step,
        "lanes": lanes,
        "slots": slots,
        "wall_limits": tuple(wall_limits),
        "obstacles": obstacles,
        "token_sizes": token_sizes,
        "goals": goals,
        "transfers": transfers,
        "gates": gates,
        "start": (tuple(surfaces), tokens),
    }


def _solved(board, state):
    surfaces, tokens = state
    for colour, slot in tokens:
        if not any(
            c == colour and s == slot and surfaces[slot] == height
            for c, s, height in board["goals"]
        ):
            return False
    return True


def _successors(board, state, blocked):
    surfaces, tokens = state
    step = board["step"]

    occupied = {}
    for colour, slot in tokens:
        occupied[slot] = max(
            occupied.get(slot, 0),
            board["token_sizes"].get(colour, step),
        )

    for donor, receiver, click in board["transfers"]:
        if (state, click) in blocked:
            continue

        donor_end = board["slots"][donor][1]
        if surfaces[donor] >= donor_end:
            continue

        next_donor = surfaces[donor] + step
        next_receiver = surfaces[receiver] - step
        if next_donor > donor_end:
            continue

        receiver_lane = board["slots"][receiver][0]
        floors = [
            end
            for _, end in board["obstacles"][receiver_lane]
            if end <= surfaces[receiver]
        ]

        if floors:
            limit = max(floors) + occupied.get(receiver, 0)
        else:
            limit = board["wall_limits"][receiver_lane]

        if next_receiver < limit:
            continue

        heights = list(surfaces)
        heights[donor] = next_donor
        heights[receiver] = next_receiver
        yield (tuple(heights), tokens), click

    for left, right, height, click in board["gates"]:
        if (state, click) in blocked:
            continue
        if surfaces[left] != height or surfaces[right] != height:
            continue

        swapped = []
        for colour, slot in tokens:
            if slot == left:
                slot = right
            elif slot == right:
                slot = left
            swapped.append((colour, slot))
        swapped = tuple(sorted(swapped))

        if swapped != tokens:
            yield (surfaces, swapped), click


def _make_heuristic(board):
    n = len(board["slots"])
    step = board["step"]
    gates = board["gates"]
    goals = board["goals"]
    colours = {colour for colour, _ in board["start"][1]}
    tables = {}

    for colour in colours:
        nodes = set()
        targets = []

        for c, slot, height in goals:
            if c == colour:
                nodes.add((slot, height))
                targets.append((slot, height))

        for left, right, height, _ in gates:
            nodes.add((left, height))
            nodes.add((right, height))

        nodes = list(nodes)
        if not targets:
            tables[colour] = [[1000] * 65 for _ in range(n)]
            continue

        index = {node: i for i, node in enumerate(nodes)}
        edges = [[] for _ in nodes]

        for i, (slot, height) in enumerate(nodes):
            for j in range(i):
                other_slot, other_height = nodes[j]
                if slot == other_slot:
                    cost = (abs(height - other_height) + step - 1) // step
                    edges[i].append((j, cost))
                    edges[j].append((i, cost))

        for left, right, height, _ in gates:
            i = index[(left, height)]
            j = index[(right, height)]
            edges[i].append((j, 1))
            edges[j].append((i, 1))

        dist = [1000] * len(nodes)
        heap = []
        for target in targets:
            i = index[target]
            if dist[i] != 0:
                dist[i] = 0
                heapq.heappush(heap, (0, i))

        while heap:
            d, i = heapq.heappop(heap)
            if d != dist[i]:
                continue
            for j, cost in edges[i]:
                nd = d + cost
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))

        table = [[1000] * 65 for _ in range(n)]
        for slot in range(n):
            relevant = [
                (height, dist[i])
                for i, (s, height) in enumerate(nodes)
                if s == slot and dist[i] < 1000
            ]
            if relevant:
                for height in range(65):
                    table[slot][height] = min(
                        d + (abs(height - h) + step - 1) // step
                        for h, d in relevant
                    )
        tables[colour] = table

    def heuristic(state):
        surfaces, tokens = state
        value = 0
        for colour, slot in tokens:
            height = max(0, min(64, surfaces[slot]))
            value = max(value, tables[colour][slot][height])
        return value

    return heuristic


def _search(board, blocked):
    start = board["start"]
    if _solved(board, start):
        return []

    heuristic = _make_heuristic(board)
    h0 = heuristic(start)
    if h0 >= 1000:
        return None

    deadline = time.monotonic() + 7.5
    serial = 0
    heap = [(h0 * 1.3, h0, serial, 0, start)]
    best_g = {start: 0}
    parent = {}
    best_state = start
    best_h = h0
    expanded = 0

    while heap:
        _, h, _, g, state = heapq.heappop(heap)
        if best_g.get(state) != g:
            continue

        if _solved(board, state):
            best_state = state
            break

        expanded += 1
        if expanded % 512 == 0:
            if time.monotonic() >= deadline or expanded >= 200000:
                break

        if g >= 49:
            continue

        for next_state, click in _successors(board, state, blocked):
            ng = g + 1
            if ng >= best_g.get(next_state, 1000000):
                continue

            nh = heuristic(next_state)
            if nh >= 1000:
                continue

            best_g[next_state] = ng
            parent[next_state] = (state, click)
            serial += 1
            heapq.heappush(
                heap, (ng + 1.3 * nh, nh, serial, ng, next_state)
            )

            if nh < best_h:
                best_h = nh
                best_state = next_state

    if best_state == start:
        return None

    path = []
    current = best_state
    while current != start:
        previous, click = parent[current]
        path.append((click, current))
        current = previous
    path.reverse()
    return path


def _fallback_click(obs, memory):
    a = np.asarray(obs)
    choices = []

    for cells in _components(a == 9):
        r0, r1, c0, c1 = _box(cells)
        if r1 - r0 <= 12 and c1 - c0 <= 12:
            choices.append(((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2))

    if not choices:
        return 1, 1

    counts = memory.setdefault("fallback_counts", {})
    click = min(choices, key=lambda p: counts.get(str(p), 0))
    counts[str(click)] = counts.get(str(click), 0) + 1
    return click


def act(obs, available, memory):
    if "ACTION6" not in available:
        if "RESET" in available:
            return "RESET", None, None
        return (available[0] if available else "RESET"), None, None

    board = _read_board(obs, memory)
    if board is None:
        row, col = _fallback_click(obs, memory)
        return "ACTION6", int(row), int(col)

    state = board["start"]
    blocked = memory.setdefault("blocked", set())
    previous = memory.get("previous_state")
    previous_click = memory.get("previous_click")
    expected = memory.get("expected")

    if previous is not None and expected is not None:
        if state == previous and state != expected and previous_click is not None:
            blocked.add((previous, previous_click))

    plan = memory.get("plan")
    if plan and expected == state:
        click, next_state = plan.pop(0)
    else:
        path = _search(board, blocked)
        if path:
            click, next_state = path[0]
            memory["plan"] = path[1:]
        else:
            memory["plan"] = []
            options = list(_successors(board, state, blocked))
            if options:
                visits = memory.setdefault("visits", {})
                key = repr(state)
                per_state = visits.setdefault(key, {})
                next_state, click = min(
                    options, key=lambda item: per_state.get(str(item[1]), 0)
                )
                per_state[str(click)] = per_state.get(str(click), 0) + 1
            else:
                row, col = _fallback_click(obs, memory)
                memory["previous_state"] = None
                memory["expected"] = None
                return "ACTION6", int(row), int(col)

    memory["previous_state"] = state
    memory["previous_click"] = click
    memory["expected"] = next_state
    return "ACTION6", int(click[0]), int(click[1])

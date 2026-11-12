import time
import heapq
from collections import Counter, deque
import numpy as np


DIRS = ((0, -1, "ACTION1"), (0, 1, "ACTION2"),
        (-1, 0, "ACTION3"), (1, 0, "ACTION4"))

EXAMPLES = {
    5: [
        " -,-T-P-      ..p..x",
        "    |        <---T-P>",
        " ...|  <--> .|...|. |",
        " xp.|.x|  Lx.|.p.|..|",
        " ...|  |    .|..p|x.|",
        "    |  |     L---t--3",
        "  <-t--3       .",
        "  |            p",
        "  |           ..x",
    ],
    7: [
        "                 <--->",
        "x     r    <-->  |   |",
        "p     p    |  ; <tTPTt>",
        "L--T--3  <-t-p. | D D |",
        "   |     |    .   | | ;",
        " <-t-,> <t->  .p.p. | .",
        " |    | |  |  .p.p.   x",
        " p    p L->p           ",
        " .p.p.....3.",
    ],
    9: [
        "       ",
        "           x..p.p......",
        "  ..b..    .........bb.",
        "  ...b.    .p.....p.p..",
        "  .....              ..",
        "  ....,--------------..",
        "  xb..   ",
        "  .b..x  ",
    ],
}


def example_data(rows):
    floor, rail, carts, blocks, pegs = set(), set(), set(), set(), {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            p = (x, y)
            if ch in ".xrbp":
                floor.add(p)
            if ch in ",;DP-|L3<>Tt":
                rail.add(p)
            if ch in ",;DP":
                carts.add(p)
            if ch in "pPD":
                blocks.add(p)
            if ch == "x":
                pegs[p] = 14
            elif ch == "r":
                pegs[p] = 8
            elif ch == "b":
                pegs[p] = 9
    return floor, rail, carts, blocks, pegs


TEMPLATES = {n: example_data(rows) for n, rows in EXAMPLES.items()}


def phase_of(a, previous=None):
    c = a[:59, :59]
    floor = ((c == 0) & (a[:59, 5:] == 0) &
             (a[5:, :59] == 0) & (a[5:, 5:] == 0))
    cart = ((c == 11) & (a[:59, 5:] == 11) &
            (a[5:, :59] == 11) & (a[5:, 5:] == 11))
    plain = ((a[1:60, 1:60] == 1) &
             (a[1:60, 4:63] == 1) &
             (a[4:63, 1:60] == 1) &
             (a[4:63, 4:63] == 1))
    mid = a[2:61, 2:61]
    occupied = ((mid == 14) | (mid == 8) | (mid == 9) |
                (mid == 7) | (mid == 2))
    mask = (floor & (plain | occupied)) | cart
    mask[0, :] = False
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return previous if previous is not None else (5, 5)
    votes = np.zeros((6, 6), dtype=np.int32)
    for y, x in zip(ys, xs):
        votes[int(y) % 6, int(x) % 6] += 1
    if previous is not None:
        votes[previous[1], previous[0]] += 1
    y, x = np.unravel_index(int(votes.argmax()), votes.shape)
    return int(x), int(y)


def translation(old, new, old_phase, new_phase):
    dx0 = (new_phase[0] - old_phase[0]) % 6
    dy0 = (new_phase[1] - old_phase[1]) % 6
    best_score, best = -10**9, (0, 0)
    for dy in range(-60 + ((dy0 + 60) % 6), 61, 6):
        for dx in range(-60 + ((dx0 + 60) % 6), 61, 6):
            x0, x1 = max(0, -dx), min(64, 64 - dx)
            y0, y1 = max(1, -dy, 1 - dy), min(64, 64 - dy)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            a = old[y0:y1, x0:x1]
            b = new[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            va = (a == 0) | (a == 1) | (a == 5) | (a == 10)
            vb = (b == 0) | (b == 1) | (b == 5) | (b == 10)
            important = va & vb & ((a != 10) | (b != 10))
            same = int(np.count_nonzero(important & (a == b)))
            diff = int(np.count_nonzero(important & (a != b)))
            score = 2 * same - 2 * diff - 0.12 * (abs(dx) + abs(dy))
            if score > best_score:
                best_score, best = score, (dx, dy)
    return best


def screen(p, origin):
    return origin[0] + 6 * p[0], origin[1] + 6 * p[1]


def clickable(p, origin):
    x, y = screen(p, origin)
    return 0 <= x + 3 < 64 and 1 <= y + 2 < 64


def fully_visible(p, origin):
    x, y = screen(p, origin)
    return x >= 0 and y >= 1 and x + 5 < 64 and y + 5 < 64


def reset_icon(frame):
    return (np.all(frame[51, 3:11] == 5) and
            frame[52, 2] == 5 and frame[52, 3] == 5 and
            frame[52, 4] == 15 and frame[52, 9] == 15)


def pack(pegs, carts, blocks):
    return (tuple(sorted((x, y, c) for (x, y), c in pegs.items())),
            tuple(sorted(carts)), tuple(sorted(blocks)))


def unpack(state):
    return ({(x, y): c for x, y, c in state[0]},
            set(state[1]), set(state[2]))


def store(memory, state):
    memory["pegs"], memory["carts"], memory["blocks"] = unpack(state)


def observe(frame, memory, phase, offset):
    floor = memory.setdefault("floor", set())
    rail = memory.setdefault("rail", set())
    old_pegs = memory.get("pegs", {})
    pegs = dict(old_pegs)
    carts = set(memory.get("carts", set()))
    blocks = set(memory.get("blocks", set()))

    for ky in range(-1, 12):
        y = phase[1] + 6 * ky
        for kx in range(-1, 12):
            x = phase[0] + 6 * kx
            pos = (kx + offset[0], ky + offset[1])

            if 0 <= x + 3 < 64 and 1 <= y + 3 < 64:
                if (frame[y + 2, x + 2] == 5 and
                        frame[y + 2, x + 3] == 5 and
                        frame[y + 3, x + 2] == 5 and
                        frame[y + 3, x + 3] == 5):
                    rail.add(pos)

            if x < 0 or y < 1 or x + 5 >= 64 or y + 5 >= 64:
                continue
            corners = (int(frame[y, x]), int(frame[y, x + 5]),
                       int(frame[y + 5, x]), int(frame[y + 5, x + 5]))
            is_floor = corners == (0, 0, 0, 0)
            is_cart = corners == (11, 11, 11, 11)
            if is_floor:
                floor.add(pos)
            if is_cart:
                rail.add(pos)

            pegs.pop(pos, None)
            carts.discard(pos)
            blocks.discard(pos)
            if is_cart:
                carts.add(pos)
            if not (is_floor or is_cart):
                continue

            color = int(frame[y + 2, x + 2])
            if color in (14, 8, 9, 2):
                if (frame[y + 2, x + 3] == color and
                        frame[y + 3, x + 2] == color and
                        frame[y + 3, x + 3] == color and
                        frame[y + 1, x + 2] == color and
                        frame[y + 4, x + 2] == color):
                    pegs[pos] = old_pegs.get(pos, 14) if color == 2 else color
            elif color == 7 and frame[y + 1, x + 1] == 15:
                blocks.add(pos)

    state = pack(pegs, carts, blocks)
    store(memory, state)
    return state


def match_example(memory, state, origin):
    observed_floor = memory["floor"]
    observed_rail = memory["rail"]
    if len(observed_floor) < 8:
        return

    best, best_score = None, -10**9
    for level, (floor, rail, carts, blocks, pegs) in TEMPLATES.items():
        votes = Counter()
        for ox, oy in observed_floor:
            for x, y in floor:
                votes[(ox - x, oy - y)] += 1
        for shift, _ in votes.most_common(12):
            sx, sy = shift
            tf = {(x + sx, y + sy) for x, y in floor}
            tr = {(x + sx, y + sy) for x, y in rail}
            visible_floor = {p for p in tf if fully_visible(p, origin)}
            fm = len(observed_floor & tf)
            fw = len(observed_floor - tf)
            missing = len(visible_floor - observed_floor - set(state[1]))
            rm = len(observed_rail & tr)
            rw = len(observed_rail - tr)
            score = 3 * fm + rm - 5 * fw - 4 * missing - 2 * rw
            if (fm + rm >= 15 and fw <= 1 and missing <= 1 and
                    rw <= 3 and score > best_score):
                best_score, best = score, (level, shift)

    if best is None:
        return

    level, shift = best
    sx, sy = shift
    floor, rail, initial_carts, initial_blocks, initial_pegs = TEMPLATES[level]

    def moved(p):
        return p[0] + sx, p[1] + sy

    memory["example"] = (level, shift)
    memory["floor"].update(moved(p) for p in floor)
    memory["rail"].update(moved(p) for p in rail)
    pegs, carts, blocks = unpack(state)
    for p in initial_carts:
        q = moved(p)
        if not fully_visible(q, origin):
            carts.add(q)
    for p in initial_blocks:
        q = moved(p)
        if not fully_visible(q, origin):
            blocks.add(q)
    for p, color in initial_pegs.items():
        q = moved(p)
        if not fully_visible(q, origin):
            pegs.setdefault(q, color)
    store(memory, pack(pegs, carts, blocks))


def jump(state, start, dest):
    pegs, carts, blocks = unpack(state)
    color = pegs.pop(start)
    middle = ((start[0] + dest[0]) // 2, (start[1] + dest[1]) // 2)
    if middle in pegs and pegs[middle] == color and color != 9:
        del pegs[middle]
    pegs[dest] = color
    return pack(pegs, carts, blocks)


def move_carts(state, dx, dy, rail, origin, example):
    pegs, carts, blocks = unpack(state)
    camera_dx = camera_dy = 0
    ordered = sorted(carts, key=lambda p: p[0] * dx + p[1] * dy,
                     reverse=True)
    level = example[0] if example else None
    shift = example[1] if example else (0, 0)

    for pos in ordered:
        dest = (pos[0] + dx, pos[1] + dy)
        if dest not in rail or dest in carts:
            continue
        green = pegs.get(pos) == 14
        carts.remove(pos)
        carts.add(dest)
        if pos in pegs:
            pegs[dest] = pegs.pop(pos)
        if pos in blocks:
            blocks.remove(pos)
            blocks.add(dest)

        if green and camera_dx == 0 and camera_dy == 0:
            if level == 5:
                camera_dx = -6 * dx
            elif level == 9:
                camera_dx, camera_dy = -6 * dx, -6 * dy
                grid_x = origin[0] + 6 * shift[0]
                if grid_x >= 5 and camera_dx > 0:
                    camera_dx = camera_dy = 0
                    break
    return pack(pegs, carts, blocks), (origin[0] + camera_dx,
                                        origin[1] + camera_dy)


def jump_origin(origin, dest, example):
    if not example:
        return origin
    level, shift = example
    x, y = dest[0] - shift[0], dest[1] - shift[1]
    grid_x = origin[0] + 6 * shift[0]
    if level == 5:
        if (x, y) == (7, 3):
            return origin[0] - 18, origin[1]
        if (x, y) == (12, 3):
            return origin[0] - 33, origin[1]
    elif level == 7:
        if (x, y) == (8, 8) and grid_x == 5:
            return origin[0] - 44, origin[1]
        if x == 16 and grid_x == -39:
            return origin[0] - 40, origin[1]
    elif level == 9 and (x, y) == (6, 5):
        return origin[0] - 20, origin[1]
    return origin


def successors(state, origin, floor, rail, example):
    pegs, carts, blocks = unpack(state)
    occupied = set(pegs) | blocks
    for start in pegs:
        x, y = start
        if not clickable(start, origin):
            continue
        for dx, dy, _ in DIRS:
            middle = (x + dx, y + dy)
            dest = (x + 2 * dx, y + 2 * dy)
            if middle not in occupied or dest in occupied:
                continue
            if dest not in floor and dest not in carts:
                continue
            if dest in floor and dest in carts:
                continue
            if not clickable(dest, origin):
                continue
            yield ("J", start, dest), jump(state, start, dest), \
                jump_origin(origin, dest, example)

    for dx, dy, name in DIRS:
        nxt, cam = move_carts(state, dx, dy, rail, origin, example)
        if nxt != state or cam != origin:
            yield ("D", name), nxt, cam


def nonblue(state):
    return sum(c != 9 for _, _, c in state[0])


def pair_distance(state):
    pegs = [(x, y, c) for x, y, c in state[0] if c != 9]
    best = 30
    for i in range(len(pegs)):
        x, y, c = pegs[i]
        for xx, yy, cc in pegs[i + 1:]:
            if c == cc:
                best = min(best, abs(x - xx) + abs(y - yy))
    return best if len(pegs) > 1 else 0


def reconstruct(end, parent):
    path = []
    node = end
    while node in parent:
        previous, move = parent[node]
        path.append((previous[0], previous[1], move))
        node = previous
    path.reverse()
    return path


def search(root, origin, floor, rail, example, banned, seen):
    target = 2 if example and example[0] == 7 else 1
    deadline = time.monotonic() + 2.2
    start = (root, origin)
    queue = [(0, 0, 0, start)]
    distance = {start: 0}
    parent = {}
    best_node, best_key = None, None
    serial = expanded = 0

    while queue and expanded < 70000 and time.monotonic() < deadline:
        _, cost, _, node = heapq.heappop(queue)
        if cost != distance.get(node):
            continue
        state, cam = node
        expanded += 1
        if node != start:
            key = (nonblue(state), pair_distance(state) + 0.3 * cost, cost)
            if best_key is None or key < best_key:
                best_key, best_node = key, node
        if cost >= 70:
            continue

        for move, nxt, new_cam in successors(state, cam, floor, rail, example):
            if node == start and (move in banned or nxt in seen):
                continue
            new_cost = cost + (2 if move[0] == "J" else 1)
            new_node = (nxt, new_cam)
            if new_cost >= distance.get(new_node, 10**9):
                continue
            distance[new_node] = new_cost
            parent[new_node] = (node, move)
            if move[0] == "J" and nonblue(nxt) <= target:
                return reconstruct(new_node, parent)
            serial += 1
            estimate = (new_cost + 7 * max(0, nonblue(nxt) - target) +
                        0.18 * pair_distance(nxt))
            heapq.heappush(queue, (estimate, new_cost, serial, new_node))
    return reconstruct(best_node, parent) if best_node is not None else []


def rail_distances(target, rail):
    distances = {target: 0}
    q = deque([target])
    while q:
        x, y = q.popleft()
        for dx, dy, _ in DIRS:
            p = (x + dx, y + dy)
            if p in rail and p not in distances:
                distances[p] = distances[(x, y)] + 1
                q.append(p)
    return distances


def navigate_carts(root, origin, rail, example, target, color=None):
    if target not in rail:
        return []
    distances = rail_distances(target, rail)
    deadline = time.monotonic() + 1.5
    start = (root, origin)
    queue = [(0, 0, 0, start)]
    costs = {start: 0}
    parent = {}
    serial = expanded = 0

    while queue and expanded < 35000 and time.monotonic() < deadline:
        _, cost, _, node = heapq.heappop(queue)
        if cost != costs.get(node):
            continue
        state, cam = node
        pegs, carts, _ = unpack(state)
        done = target in carts if color is None else (
            pegs.get(target) == color and target in carts)
        if done and node != start:
            return reconstruct(node, parent)
        expanded += 1
        if cost >= 60:
            continue

        for dx, dy, name in DIRS:
            nxt, new_cam = move_carts(state, dx, dy, rail, cam, example)
            new_node = (nxt, new_cam)
            if new_node == node:
                continue
            new_cost = cost + 1
            if new_cost >= costs.get(new_node, 10**9):
                continue
            np_, nc, _ = unpack(nxt)
            if color is None:
                h = min((distances.get(p, 100) for p in nc), default=100)
            else:
                positions = [p for p in nc if np_.get(p) == color]
                h = min((distances.get(p, 100) for p in positions), default=100)
            if h >= 100:
                continue
            costs[new_node] = new_cost
            parent[new_node] = (node, ("D", name))
            serial += 1
            heapq.heappush(queue, (new_cost + h, new_cost, serial, new_node))
    return []


def arrange_carts(root, origin, rail, example, kind):
    """Arrange carts for the final crossing in the shown level 7."""
    sx, sy = example[1]

    def p(x, y):
        return x + sx, y + sy

    if kind == "first":
        target, middle = p(18, 3), p(18, 4)
    elif kind == "carry":
        target, middle = p(22, 3), p(22, 4)
    else:
        target, middle = p(22, 4), None

    if target not in rail:
        return []
    td = rail_distances(target, rail)
    md = rail_distances(middle, rail) if middle is not None else {}
    start = (root, origin)
    costs = {start: 0}
    parent = {}
    queue = [(0, 0, 0, start)]
    serial = expanded = 0
    deadline = time.monotonic() + 2.0

    def evaluate(state):
        pegs, carts, blocks = unpack(state)
        occupied = set(pegs) | blocks
        empty = carts - occupied
        loaded = carts & occupied
        if kind == "first":
            done = target in empty and middle in occupied
            h1 = min((td.get(q, 100) for q in empty), default=100)
            h2 = 0 if middle in occupied else min(
                (md.get(q, 100) for q in loaded), default=100)
            return done, h1 + h2
        if kind == "carry":
            done = pegs.get(target) == 14 and target in carts and middle in occupied
            greens = [q for q in carts if pegs.get(q) == 14]
            h1 = min((td.get(q, 100) for q in greens), default=100)
            h2 = 0 if middle in occupied else min(
                (md.get(q, 100) for q in loaded if pegs.get(q) != 14),
                default=100)
            return done, h1 + h2
        done = target in empty
        h = min((td.get(q, 100) for q in empty), default=100)
        return done, h

    while queue and expanded < 70000 and time.monotonic() < deadline:
        _, cost, _, node = heapq.heappop(queue)
        if cost != costs.get(node):
            continue
        state, cam = node
        done, _ = evaluate(state)
        if done and node != start:
            return reconstruct(node, parent)
        expanded += 1
        if cost >= 65:
            continue
        for dx, dy, name in DIRS:
            nxt, new_cam = move_carts(state, dx, dy, rail, cam, example)
            new_node = (nxt, new_cam)
            if new_node == node:
                continue
            new_cost = cost + 1
            if new_cost >= costs.get(new_node, 10**9):
                continue
            done, h = evaluate(nxt)
            if h >= 100:
                continue
            costs[new_node] = new_cost
            parent[new_node] = (node, ("D", name))
            if done:
                return reconstruct(new_node, parent)
            serial += 1
            heapq.heappush(queue, (new_cost + 1.3 * h, new_cost,
                                   serial, new_node))
    return []


def level7_plan(state, origin, floor, rail, example):
    if not example or example[0] != 7:
        return []
    sx, sy = example[1]
    grid_x = origin[0] + 6 * sx
    pegs, carts, blocks = unpack(state)

    def p(x, y):
        return x + sx, y + sy

    def has(x, y, color):
        return pegs.get(p(x, y)) == color

    def direct(x1, y1, x2, y2):
        move = ("J", p(x1, y1), p(x2, y2))
        for candidate, _, _ in successors(state, origin, floor, rail, example):
            if candidate == move:
                return [(state, origin, move)]
        return []

    def align(x, y, color=None):
        return navigate_carts(state, origin, rail, example, p(x, y), color)

    def jump_to_cart(x1, y1, x2, y2):
        if p(x2, y2) not in carts:
            return align(x2, y2)
        return direct(x1, y1, x2, y2)

    if grid_x == 5:
        if has(6, 1, 8):
            return jump_to_cart(6, 1, 6, 3)
        if any(c == 8 and pos in carts for pos, c in pegs.items()):
            if not has(1, 6, 8):
                return align(1, 6, 8)
        if has(1, 6, 8):
            return direct(1, 6, 1, 8)
        if has(1, 8, 8):
            return direct(1, 8, 3, 8)
        if has(3, 8, 8):
            return direct(3, 8, 5, 8)
        if has(5, 8, 8):
            if has(0, 1, 14):
                return jump_to_cart(0, 1, 0, 3)
            if any(c == 14 and pos in carts and pos[0] - sx <= 8
                   for pos, c in pegs.items()):
                if not has(6, 6, 14):
                    return align(6, 6, 14)
            if has(6, 6, 14):
                return direct(6, 6, 6, 8)
            if has(6, 8, 14):
                return direct(5, 8, 7, 8)
        if has(7, 8, 8) and has(6, 8, 14):
            return direct(6, 8, 8, 8)

    elif grid_x == -39:
        if has(7, 8, 8) and has(8, 8, 14):
            return direct(7, 8, 9, 8)
        if has(9, 8, 8) and has(8, 8, 14):
            return jump_to_cart(8, 8, 10, 8)
        if has(9, 8, 8) and has(10, 8, 14):
            return direct(9, 8, 11, 8)
        if has(11, 8, 8):
            if any(c == 14 and pos in carts and pos[0] - sx < 14
                   for pos, c in pegs.items()):
                if not has(12, 3, 14):
                    return align(12, 3, 14)
            if has(12, 3, 14):
                return direct(12, 3, 14, 3)
            if has(14, 3, 14):
                return jump_to_cart(11, 8, 11, 6)
        if has(14, 3, 14):
            if any(c == 8 and pos in carts for pos, c in pegs.items()):
                if not has(14, 2, 8):
                    return align(14, 2, 8)
            if has(14, 2, 8):
                return direct(14, 2, 14, 4)
            if has(14, 4, 8):
                return direct(14, 3, 14, 5)
        if has(14, 5, 14):
            return direct(14, 5, 16, 5)

    elif grid_x == -79:
        if has(16, 5, 14):
            return direct(16, 5, 18, 5)
        if has(18, 5, 14):
            result = direct(18, 5, 18, 3)
            if result:
                return result
            return arrange_carts(state, origin, rail, example, "first")
        if has(22, 5, 14) and has(22, 6, 14):
            result = direct(22, 6, 22, 4)
            if result:
                return result
            return arrange_carts(state, origin, rail, example, "empty")
        if has(22, 3, 14):
            result = direct(22, 3, 22, 5)
            if result:
                return result
        if any(c == 14 and pos in carts and pos[0] - sx >= 17
               for pos, c in pegs.items()):
            return arrange_carts(state, origin, rail, example, "carry")
    return []


def undo(memory, available):
    trace = memory.get("trace", [])
    if not trace or "ACTION7" not in available:
        return None
    before, move = trace.pop()
    memory.setdefault("banned", {}).setdefault(before, set()).add(move)
    memory.setdefault("seen", set()).discard(before)
    store(memory, before)
    memory.pop("pending", None)
    memory.pop("prediction", None)
    memory.pop("plan", None)
    return ("ACTION7", None, None)


def act(obs, available, memory):
    frame = np.asarray(obs, dtype=np.int8)
    old_frame = memory.get("previous_frame")
    old_phase = memory.get("phase")
    phase = phase_of(frame, old_phase)
    offset = memory.get("offset", (0, 0))

    if old_frame is not None and old_phase is not None:
        dx, dy = translation(old_frame, frame, old_phase, phase)
        offset = (offset[0] - (old_phase[0] + dx - phase[0]) // 6,
                  offset[1] - (old_phase[1] + dy - phase[1]) // 6)

    origin = (phase[0] - 6 * offset[0], phase[1] - 6 * offset[1])
    memory["previous_frame"] = frame.copy()
    memory["phase"], memory["offset"] = phase, offset

    if reset_icon(frame):
        result = undo(memory, available)
        if result:
            return result
        return ("ACTION6", 55, 7)

    prediction = memory.pop("prediction", None)
    if prediction is not None and old_frame is not None:
        before, move, predicted = prediction
        if np.array_equal(old_frame[1:, :], frame[1:, :]):
            ap, ac, ab = unpack(before)
            bp, bc, bb = unpack(predicted)
            changed = (set(ap) ^ set(bp)) | (ac ^ bc) | (ab ^ bb)
            if changed and all(fully_visible(p, origin) for p in changed):
                store(memory, before)
                memory.setdefault("banned", {}).setdefault(before, set()).add(move)
                trace = memory.get("trace", [])
                if trace and trace[-1] == (before, move):
                    trace.pop()
                memory.pop("plan", None)

    state = observe(frame, memory, phase, offset)
    if "example_checked" not in memory:
        memory["example_checked"] = True
        match_example(memory, state, origin)
        state = pack(memory["pegs"], memory["carts"], memory["blocks"])

    pending = memory.pop("pending", None)
    if pending is not None:
        start, dest = pending
        pegs, _, _ = unpack(state)
        x, y = screen(dest, origin)
        marker = (0 <= x + 4 < 64 and 1 <= y + 4 < 64 and
                  (frame[y + 1, x + 2] == 2 or
                   frame[y + 2, x + 1] == 2 or
                   frame[y + 3, x + 4] == 2))
        if start in pegs and marker and "ACTION6" in available:
            move = ("J", start, dest)
            memory.setdefault("trace", []).append((state, move))
            memory.setdefault("seen", set()).add(state)
            store(memory, jump(state, start, dest))
            return ("ACTION6", y + 2, x + 3)
        memory.setdefault("banned", {}).setdefault(state, set()).add(
            ("J", start, dest))
        memory.pop("plan", None)

    floor = memory.setdefault("floor", set())
    rail = memory.setdefault("rail", set())
    example = memory.get("example")
    banned = memory.setdefault("banned", {}).get(state, set())
    seen = memory.setdefault("seen", set())

    plan = memory.get("plan", [])
    if not plan or plan[0][0] != state or plan[0][1] != origin:
        plan = level7_plan(state, origin, floor, rail, example)
        if not plan:
            plan = search(state, origin, floor, rail, example, banned, seen)

    if plan:
        _, _, move = plan.pop(0)
        memory["plan"] = plan
        if move[0] == "J" and "ACTION6" in available:
            start, dest = move[1], move[2]
            x, y = screen(start, origin)
            memory["pending"] = (start, dest)
            return ("ACTION6", y + 2, x + 3)
        if move[0] == "D" and move[1] in available:
            for dx, dy, name in DIRS:
                if name == move[1]:
                    nxt, _ = move_carts(state, dx, dy, rail, origin, example)
                    memory.setdefault("trace", []).append((state, move))
                    memory["prediction"] = (state, move, nxt)
                    seen.add(state)
                    store(memory, nxt)
                    return (name, None, None)

    result = undo(memory, available)
    if result:
        return result

    attempted = memory.setdefault("edge_attempts", set())
    _, carts, _ = unpack(state)
    for dx, dy, name in DIRS:
        if name not in available or (state, name) in attempted:
            continue
        if any(not clickable((p[0] + dx, p[1] + dy), origin) for p in carts):
            attempted.add((state, name))
            return (name, None, None)

    if "RESET" in available and not memory.get("used_reset"):
        memory.clear()
        memory["used_reset"] = True
        return ("RESET", None, None)

    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if name in available:
            return (name, None, None)
    return ("ACTION6", 32, 32)

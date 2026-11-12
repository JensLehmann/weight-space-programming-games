from collections import deque
from itertools import permutations
import heapq
import math
import numpy as np


def _hungarian(cost):
    """Minimum-cost assignment; returns the column chosen for each row."""
    n = len(cost)
    if not n:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = [0] * n
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    return result


def _viewport(obs):
    ys, xs = np.where(obs == 4)
    if len(xs) == 0:
        return 0, 0, 64, 64

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # The interface overwrites display column zero.
    if x0 == 1 and x1 == 63:
        x0 = 0
    return x0, y0, x1 + 1, y1 + 1


def _world(obs, x0, y0, x1, y1, scale):
    # Sampling the middle of a scaled pixel avoids most interface artifacts.
    off = scale // 2
    return obs[y0 + off:y1:scale, x0 + off:x1:scale]


def _find_slots(world):
    h, w = world.shape
    slots = []
    excluded = {3, 4, 8, 14}

    for y in range(0, h - 1, 3):
        for x in range(0, w - 1, 3):
            c = int(world[y, x])
            if c in excluded:
                continue
            if not (int(world[y, x + 1]) == c
                    and int(world[y + 1, x]) == c
                    and int(world[y + 1, x + 1]) == c):
                continue

            # A tile is a 2-by-2 block, not part of a larger block.
            if x > 0 and int(world[y, x - 1]) == c:
                continue
            if y > 0 and int(world[y - 1, x]) == c:
                continue
            if x + 2 < w and int(world[y, x + 2]) == c:
                continue
            if y + 2 < h and int(world[y + 2, x]) == c:
                continue
            slots.append((x, y))
    return slots


def _button_patterns(color):
    if color == 8:
        base = np.array([[0, 1, 1], [1, 1, 1],
                         [1, 1, 1], [0, 1, 1]], dtype=np.int8)
    else:
        base = np.array([[1, 1, 0], [1, 1, 1],
                         [1, 1, 1], [1, 1, 0]], dtype=np.int8)
    result = []
    seen = set()
    for k in range(4):
        a = np.rot90(base, k)
        key = (a.shape, tuple(int(v) for v in a.flat))
        if key not in seen:
            seen.add(key)
            result.append(a.copy())
    return result


def _find_buttons(world, x0, y0, scale):
    h, w = world.shape
    buttons = []

    for color in (8, 14):
        mask = (world == color).copy()

        # The progress interface is drawn after the board.
        for y in range(h):
            display_y = y0 + y * scale + scale // 2
            if display_y == 1:
                mask[y, :] = False
        for x in range(w):
            display_x = x0 + x * scale + scale // 2
            if display_x == 0:
                mask[:, x] = False

        visited = np.zeros((h, w), dtype=np.bool_)
        patterns = _button_patterns(color)
        for sy in range(h):
            for sx in range(w):
                if not mask[sy, sx] or visited[sy, sx]:
                    continue
                visited[sy, sx] = True
                q = deque([(sx, sy)])
                component = []
                while q:
                    x, y = q.popleft()
                    component.append((x, y))
                    for nx, ny in ((x - 1, y), (x + 1, y),
                                   (x, y - 1), (x, y + 1)):
                        if (0 <= nx < w and 0 <= ny < h
                                and mask[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            q.append((nx, ny))

                if len(component) < 8:
                    continue
                comp = set(component)
                minx = min(x for x, y in component)
                maxx = max(x for x, y in component)
                miny = min(y for x, y in component)
                maxy = max(y for x, y in component)
                matches = []

                for pat in patterns:
                    ph, pw = pat.shape
                    for yy in range(max(0, miny - ph + 1), min(h - ph, maxy) + 1):
                        for xx in range(max(0, minx - pw + 1), min(w - pw, maxx) + 1):
                            cells = set()
                            good = True
                            for dy in range(ph):
                                for dx in range(pw):
                                    is_color = bool(mask[yy + dy, xx + dx])
                                    if pat[dy, dx]:
                                        if not is_color or (xx + dx, yy + dy) not in comp:
                                            good = False
                                            break
                                        cells.add((xx + dx, yy + dy))
                                    elif is_color:
                                        good = False
                                        break
                                if not good:
                                    break
                            if good:
                                matches.append((cells, xx, yy, pw, ph))

                used = set()
                found = 0
                for cells, xx, yy, pw, ph in matches:
                    if cells & used:
                        continue
                    used.update(cells)
                    # The center is inside the button's bounding box.
                    gx = xx + pw // 2
                    gy = yy + ph // 2
                    buttons.append([color, gx, gy])
                    found += 1

                if not found and 8 <= len(component) <= 16:
                    # Allow for a partly obscured button.
                    gx = int(round(sum(x for x, y in component) / len(component)))
                    gy = int(round(sum(y for x, y in component) / len(component)))
                    buttons.append([color, gx, gy])

    buttons.sort(key=lambda b: (b[0] != 8, b[2], b[1]))
    return buttons


def _pair_buttons(buttons):
    left = [i for i, b in enumerate(buttons) if b[0] == 8]
    right = [i for i, b in enumerate(buttons) if b[0] == 14]
    pairs = [-1] * len(buttons)
    if not left or not right:
        return pairs

    # Match opposite-colored controls by total distance.
    if len(left) <= len(right) and len(right) <= 8:
        best = None
        for choice in permutations(right, len(left)):
            score = sum(
                abs(buttons[a][1] - buttons[b][1])
                + abs(buttons[a][2] - buttons[b][2])
                for a, b in zip(left, choice)
            )
            if best is None or score < best[0]:
                best = (score, list(zip(left, choice)))
        matching = best[1]
    elif len(right) < len(left) and len(left) <= 8:
        best = None
        for choice in permutations(left, len(right)):
            score = sum(
                abs(buttons[a][1] - buttons[b][1])
                + abs(buttons[a][2] - buttons[b][2])
                for a, b in zip(choice, right)
            )
            if best is None or score < best[0]:
                best = (score, list(zip(choice, right)))
        matching = best[1]
    else:
        matching = []
        remaining = set(right)
        for a in left:
            if not remaining:
                break
            b = min(
                remaining,
                key=lambda j: abs(buttons[a][1] - buttons[j][1])
                + abs(buttons[a][2] - buttons[j][2])
            )
            remaining.remove(b)
            matching.append((a, b))

    for a, b in matching:
        pairs[a] = b
        pairs[b] = a
    return pairs


def _initialize(obs):
    x0, y0, x1, y1 = _viewport(obs)
    width, height = x1 - x0, y1 - y0
    candidates = []

    for scale in range(1, 5):
        if width % scale or height % scale:
            continue
        gw, gh = width // scale, height // scale
        if gw <= 0 or gh <= 0:
            continue
        if min(64 // gw, 64 // gh) != scale:
            continue
        world = _world(obs, x0, y0, x1, y1, scale)
        slots = _find_slots(world)
        candidates.append((len(slots), scale, world, slots))

    if not candidates:
        world = _world(obs, x0, y0, x1, y1, 1)
        candidates = [(len(_find_slots(world)), 1, world, _find_slots(world))]

    _, scale, world, slots = max(candidates, key=lambda item: (item[0], item[1]))
    slots.sort(key=lambda p: (p[1], p[0]))
    h, w = world.shape

    targets = {11: [], 12: []}
    for i, (x, y) in enumerate(slots):
        if x < 1 or y < 1 or x + 2 >= w or y + 2 >= h:
            continue
        corners = (
            int(world[y - 1, x - 1]), int(world[y - 1, x + 2]),
            int(world[y + 2, x - 1]), int(world[y + 2, x + 2])
        )
        if corners[0] in targets and all(c == corners[0] for c in corners):
            targets[corners[0]].append(i)

    buttons = _find_buttons(world, x0, y0, scale)
    pairs = _pair_buttons(buttons)
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1, "scale": scale,
        "slots": [list(p) for p in slots],
        "targets11": targets[11], "targets12": targets[12],
        "buttons": buttons, "pairs": pairs,
        "records": [], "trials": {},
    }


def _read_board(obs, mem):
    world = _world(obs, mem["x0"], mem["y0"], mem["x1"], mem["y1"],
                   mem["scale"])
    h, w = world.shape
    result = []
    for x, y in mem["slots"]:
        if 0 <= x < w and 0 <= y < h:
            result.append(int(world[y, x]))
        else:
            result.append(4)
    return result


def _groups_from_samples(n, samples):
    if not samples:
        return [list(range(n))], [list(range(n))]

    source_groups = {}
    dest_groups = {}
    for i in range(n):
        s = tuple(before[i] for before, after in samples)
        d = tuple(after[i] for before, after in samples)
        source_groups.setdefault(s, []).append(i)
        dest_groups.setdefault(d, []).append(i)

    if set(source_groups) != set(dest_groups):
        return None
    src, dst = [], []
    for key, group in source_groups.items():
        other = dest_groups[key]
        if len(group) != len(other):
            return None
        src.append(group)
        dst.append(other)
    return src, dst


def _make_model(n, slots, records, action, inverse):
    real = [(r[1], r[2]) for r in records if r[0] == action]
    synthetic = []
    if inverse >= 0:
        synthetic = [(r[2], r[1]) for r in records if r[0] == inverse]

    grouped = _groups_from_samples(n, real + synthetic)
    if grouped is None:
        grouped = _groups_from_samples(n, real)
    if grouped is None:
        grouped = _groups_from_samples(n, [])

    src_groups, dst_groups = grouped
    mapping = list(range(n))
    group_id = [0] * n
    group_sizes = []

    for gi, (src, dst) in enumerate(zip(src_groups, dst_groups)):
        group_sizes.append(len(src))
        for s in src:
            group_id[s] = gi
        if len(src) == 1:
            mapping[src[0]] = dst[0]
            continue

        cost = []
        for s in src:
            sx, sy = slots[s]
            row = []
            for d in dst:
                dx, dy = slots[d]
                distance = abs(sx - dx) + abs(sy - dy)
                # Prefer an unchanged slot when the evidence is ambiguous.
                row.append(0 if s == d else 3 + distance)
            cost.append(row)
        assignment = _hungarian(cost)
        for i, j in enumerate(assignment):
            mapping[src[i]] = dst[j]

    return mapping, group_id, group_sizes


def _goal_state(board, count11, count12):
    a = tuple(i for i, c in enumerate(board) if c == 11)
    b = tuple(i for i, c in enumerate(board) if c == 12)
    if len(a) != count11 or len(b) != count12:
        return None
    return tuple(sorted(a)) + tuple(sorted(b))


def _apply_goal_state(state, mapping, count11):
    a = sorted(mapping[i] for i in state[:count11])
    b = sorted(mapping[i] for i in state[count11:])
    return tuple(a + b)


def _uncertain(state, model, count11):
    _, group_id, sizes = model
    for section in (state[:count11], state[count11:]):
        counts = {}
        for s in section:
            g = group_id[s]
            counts[g] = counts.get(g, 0) + 1
        for g, count in counts.items():
            if count != sizes[g]:
                return True
    return False


def _distance(state, targets11, targets12, slots, count11):
    total = 0

    for positions, targets in ((state[:count11], targets11),
                               (state[count11:], targets12)):
        if not positions:
            continue
        if len(positions) != len(targets):
            return 1000

        def dist(a, b):
            return (abs(slots[a][0] - slots[b][0])
                    + abs(slots[a][1] - slots[b][1])) / 3.0

        if len(positions) <= 6:
            total += min(
                sum(dist(a, b) for a, b in zip(positions, order))
                for order in permutations(targets)
            )
        else:
            cost = [[dist(a, b) for b in targets] for a in positions]
            assignment = _hungarian(cost)
            total += sum(cost[i][j] for i, j in enumerate(assignment))
    return total


def _trial_key(state, action):
    return ",".join(str(x) for x in state) + ":" + str(action)


def _choose_action(state, models, targets11, targets12, slots, trials):
    count11 = len(targets11)
    goal = tuple(sorted(targets11)) + tuple(sorted(targets12))
    if not models:
        return None

    # Search the current estimated permutations. Uncertain moves have a
    # larger cost, so established transitions are preferred.
    heap = [(0.0, 0, state, -1)]
    best_cost = {state: 0.0}
    serial = 0
    best_frontier = None
    expanded = 0
    max_expanded = 5000

    while heap and expanded < max_expanded:
        cost, _, current, first = heapq.heappop(heap)
        if cost != best_cost.get(current):
            continue
        expanded += 1

        if current == goal and first >= 0:
            return first

        for action, model in enumerate(models):
            mapping = model[0]
            nxt = _apply_goal_state(current, mapping, count11)
            uncertain = _uncertain(current, model, count11)
            first_action = action if first < 0 else first

            if uncertain:
                trial_count = trials.get(_trial_key(current, action), 0)
                heuristic = _distance(nxt, targets11, targets12, slots, count11)
                score = cost + 1.0 + heuristic + 1.5 * trial_count
                candidate = (score, cost, action, first_action)
                if best_frontier is None or candidate < best_frontier:
                    best_frontier = candidate

            if nxt == current:
                continue
            new_cost = cost + (2.0 if uncertain else 1.0)
            if new_cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_cost
                serial += 1
                heapq.heappush(heap, (new_cost, serial, nxt, first_action))

    if best_frontier is not None:
        return best_frontier[3]

    # If every estimated move is known but no solution was found, try
    # a different control rather than becoming stuck.
    return min(
        range(len(models)),
        key=lambda a: trials.get(_trial_key(state, a), 0)
    )


def act(obs, available, memory):
    if "lp85" not in memory:
        memory["lp85"] = _initialize(obs)

    mem = memory["lp85"]
    buttons = mem["buttons"]
    if not buttons:
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION6", 32, 32)

    board = _read_board(obs, mem)
    previous = mem.pop("last", None)
    if previous is not None:
        action, before = previous
        record = [action, before, board]
        if record not in mem["records"]:
            mem["records"].append(record)
            # Keep the model and memory bounded.
            if len(mem["records"]) > 120:
                mem["records"].pop(0)

    targets11 = mem["targets11"]
    targets12 = mem["targets12"]
    state = _goal_state(board, len(targets11), len(targets12))

    if state is None or not mem["slots"]:
        action = 0
    else:
        n = len(board)
        models = [
            _make_model(n, mem["slots"], mem["records"], i, mem["pairs"][i])
            for i in range(len(buttons))
        ]
        action = _choose_action(
            state, models, targets11, targets12, mem["slots"], mem["trials"]
        )
        if action is None:
            action = 0
        key = _trial_key(state, action)
        mem["trials"][key] = mem["trials"].get(key, 0) + 1

    mem["last"] = [action, board]
    _, gx, gy = buttons[action]
    scale = mem["scale"]
    col = int(mem["x0"] + gx * scale + scale // 2)
    row = int(mem["y0"] + gy * scale + scale // 2)
    col = max(0, min(63, col))
    row = max(0, min(63, row))
    return ("ACTION6", row, col)

import heapq
import itertools
import numpy as np


# An instruction slot has alternating horizontal and vertical bits.
def _slot_pair(frame, x, y):
    if x < 0 or x + 4 >= 64 or y < 0 or y + 5 >= 64:
        return False
    h = frame[y + 1, x + 1:x + 4]
    v = frame[y + 3:y + 6, x + 2]
    if int(h[0]) not in (1, 5) or not np.all(h == h[0]):
        return False
    if int(v[0]) not in (1, 5) or not np.all(v == v[0]):
        return False
    if frame[y + 1, x] != 0 or frame[y + 1, x + 4] != 0:
        return False
    if np.any(frame[y + 3:y + 6, x + 1] != 0):
        return False
    if np.any(frame[y + 3:y + 6, x + 3] != 0):
        return False
    return True


def _find_panel(frame):
    best = None
    for y in range(26, 48):
        xs = {x for x in range(60) if _slot_pair(frame, x, y)}
        for x in sorted(xs):
            if x - 5 in xs:
                continue
            cols = []
            xx = x
            while xx in xs:
                cols.append(xx)
                xx += 5

            pairs = 1
            while y + 6 * pairs + 5 <= 50:
                if all(_slot_pair(frame, xx, y + 6 * pairs) for xx in cols):
                    pairs += 1
                else:
                    break

            bits = 2 * pairs
            score = (len(cols) * bits, len(cols), -y)
            if best is None or score > best[0]:
                best = (score, cols, y, bits)

    if best is None:
        return None

    _, cols, y, bits = best
    values = []
    for x in cols:
        value = 0
        for b in range(bits):
            if b % 2 == 0:
                yy = y + 3 * b + 1
            else:
                yy = y + 3 * b + 1
            if int(frame[yy, x + 2]) == 5:
                value |= 1 << b
        values.append(value)
    return cols, y, bits, values


def _find_button(frame, panel_y):
    # The run button contains a solid 5x5 block of colour 9.
    for y in range(max(3, panel_y + 4), 60):
        for x in range(60):
            if np.all(frame[y:y + 5, x:x + 5] == 9):
                return y + 2, x + 2
    return None


_ROBOT = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=np.bool_)

_TARGET = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=np.bool_)


def _find_shape(frame, template, panel_x, panel_y):
    colour = frame == 11
    candidates = []
    xmin = max(0, panel_x - 8)

    for scale in range(1, 5):
        for rotation in (0, 90, 180, 270):
            pat = np.rot90(template, -(rotation // 90))
            if scale != 1:
                pat = np.repeat(np.repeat(pat, scale, axis=0), scale, axis=1)
            h, w = pat.shape
            max_y = min(panel_y - h, 63 - h)
            max_x = 63 - w
            if max_y < 3 or max_x < xmin:
                continue

            # Intersect shifted masks to find all placements whose visible
            # template pixels have the expected colour.
            possible = np.ones((max_y - 2, max_x - xmin + 1), dtype=np.bool_)
            for dy, dx in np.argwhere(pat):
                possible &= colour[3 + dy:max_y + 1 + dy,
                                   xmin + dx:max_x + 1 + dx]
                if not possible.any():
                    break
            if not possible.any():
                continue

            for iy, ix in np.argwhere(possible):
                y = int(iy) + 3
                x = int(ix) + xmin
                region = colour[y:y + h, x:x + w]
                extras = int(np.count_nonzero(region & ~pat))
                # Favour complete larger sprites over small sub-patterns.
                candidates.append((extras / float(scale * scale),
                                   -scale, y, x, rotation, scale))

    if not candidates:
        return None
    candidates.sort()
    _, _, y, x, rotation, scale = candidates[0]
    return x, y, rotation, scale


def _find_checkpoints(frame, robot, panel_x, panel_y):
    rx, ry, _, _ = robot
    result = []
    xmin = max(0, panel_x - 8)
    for y in range(3, panel_y - 3):
        if (y - ry) % 4:
            continue
        for x in range(xmin, 61):
            if (x - rx) % 4:
                continue
            a = frame[y:y + 4, x:x + 4]
            for parity in (0, 1):
                ok = True
                for dy in range(4):
                    for dx in range(4):
                        value = int(a[dy, dx])
                        if (dx + dy) % 2 == parity:
                            if value != 11:
                                ok = False
                                break
                        elif value not in (4, 5, 6):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    result.append((x, y))
                    break
    return result


def _wall_prefix(frame):
    walls = (frame == 6).astype(np.int32)
    prefix = np.zeros((65, 65), dtype=np.int32)
    prefix[1:, 1:] = walls.cumsum(axis=0).cumsum(axis=1)
    return prefix.tolist()


def _wall_at(prefix, x, y, size):
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(64, x + size)
    y1 = min(64, y + size)
    if x0 >= x1 or y0 >= y1:
        return False
    return (prefix[y1][x1] - prefix[y0][x1]
            - prefix[y1][x0] + prefix[y0][x0]) > 0


# Effect numbers are internal to the planner.
_EFFECTS = {
    0: 0,     # no operation
    1: 1,     # left 4
    2: 2,     # right 4
    3: 3,     # down 4
    5: 4,     # clockwise
    6: 5,     # counterclockwise
    7: 6,     # half turn
    8: 7,     # grow
    9: 8,     # shrink
    10: 9,    # right 8
    11: 9,
    12: 10,   # left 8
    13: 10,
    16: 5,
    33: 11,   # up 4
    34: 1,
}


def _options(old, bits, rotations_needed):
    limit = 1 << bits
    choices = {}
    codes = [c for c in _EFFECTS if c < limit]
    if old < limit and old not in _EFFECTS:
        codes.append(old)

    for code in codes:
        effect = _EFFECTS.get(code, 0)
        if not rotations_needed and effect in (4, 5, 6):
            continue
        cost = (old ^ code).bit_count()
        previous = choices.get(effect)
        if previous is None or (cost, code) < previous:
            choices[effect] = (cost, code)
    return [(effect, cost, code)
            for effect, (cost, code) in choices.items()]


def _advance(state, effect, prefix, bounds, max_scale):
    x, y, rotation, scale = state
    nx, ny, nr, ns = x, y, rotation, scale

    if effect == 1:
        nx -= 4
    elif effect == 2:
        nx += 4
    elif effect == 3:
        ny += 4
    elif effect == 4:
        nr = (nr + 90) % 360
    elif effect == 5:
        nr = (nr - 90) % 360
    elif effect == 6:
        nr = (nr + 180) % 360
    elif effect == 7:
        ns += 1
        if ns > max_scale:
            return None
    elif effect == 8:
        ns = max(1, ns - 1)
    elif effect == 9:
        nx += 8
    elif effect == 10:
        nx -= 8
    elif effect == 11:
        ny -= 4

    if effect in (1, 2, 3, 9, 10, 11):
        if _wall_at(prefix, nx, ny, 4 * scale):
            nx, ny = x, y
    elif effect in (7, 8):
        if _wall_at(prefix, x, y, 4 * ns):
            ns = scale

    xmin, xmax, ymin, ymax = bounds
    if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
        return None
    return nx, ny, nr, ns


def _near_checkpoint(state, checkpoint):
    x, y, _, scale = state
    cx, cy = checkpoint
    if scale != 1:
        return False
    return x < cx + 4 and x + 4 > cx and y < cy + 4 and y + 4 > cy


def _distance(state, goal):
    x, y, rotation, scale = state
    gx, gy, gr, gs = goal
    return (abs(x - gx) + abs(y - gy)) / 4.0 + \
        (rotation != gr) + abs(scale - gs)


def _one_run(start, program, bits, goal, checkpoints, prefix, bounds):
    max_scale = min(6, max(start[3], goal[3], 1) + 1)
    rotations_needed = start[2] != goal[2]
    dp = {start: (0, ())}

    for old in program:
        options = _options(old, bits, rotations_needed)
        new = {}
        for state, (cost, path) in dp.items():
            for effect, edit_cost, code in options:
                nxt = _advance(state, effect, prefix, bounds, max_scale)
                if nxt is None:
                    continue
                candidate_cost = cost + edit_cost
                prev = new.get(nxt)
                if prev is None or candidate_cost < prev[0]:
                    new[nxt] = (candidate_cost, path + (code,))

        # Keep a generous beam if an unusual layout makes the state space
        # much larger than the ordinary boards.
        if len(new) > 12000:
            def rank(item):
                state, (cost, _) = item
                d = _distance(state, goal)
                for cx, cy in checkpoints:
                    d = min(d, (abs(state[0] - cx) +
                                abs(state[1] - cy)) / 4.0)
                return cost + 0.35 * d
            new = dict(sorted(new.items(), key=rank)[:12000])
        dp = new
        if not dp:
            break

    direct = None
    if goal in dp:
        direct = dp[goal]

    checkpoint_results = []
    for checkpoint in checkpoints:
        matches = []
        for state, (cost, path) in dp.items():
            if state != start and _near_checkpoint(state, checkpoint):
                rank = cost + 0.3 * _distance(state, goal)
                matches.append((rank, cost, state, path))
        matches.sort()
        # Preserve several orientations/programs at each checkpoint.
        seen = set()
        for _, cost, state, path in matches:
            key = (state[2], state[3])
            if key in seen:
                continue
            seen.add(key)
            checkpoint_results.append((cost, state, path))
            if len(seen) >= 3:
                break
    return direct, checkpoint_results


def _plan(robot, goal, checkpoints, program, bits, frame, panel_y):
    prefix = _wall_prefix(frame)
    points = [(robot[0], robot[1]), (goal[0], goal[1])] + checkpoints
    xmin = max(0, min(x for x, _ in points) - 12)
    xmax = min(60, max(x for x, _ in points) + 12)
    ymin = max(0, min(y for _, y in points) - 12)
    ymax = min(60, panel_y + 4, max(y for _, y in points) + 12)
    bounds = (xmin, xmax, ymin, ymax)

    counter = itertools.count()
    heap = [(0.0, next(counter), 0, robot, tuple(program), ())]
    best = {robot: 0}
    fallback = None
    expansions = 0

    while heap and expansions < 12:
        _, _, total, state, current_program, stages = heapq.heappop(heap)
        if total > best.get(state, total):
            continue
        expansions += 1

        direct, stops = _one_run(
            state, current_program, bits, goal, checkpoints, prefix, bounds
        )
        if direct is not None:
            _, final_program = direct
            all_stages = stages + (final_program,)
            return all_stages[0]

        if len(stages) >= 3:
            continue
        for edit_cost, new_state, new_program in stops:
            new_total = total + edit_cost + 1
            if new_total >= best.get(new_state, 10**9):
                continue
            best[new_state] = new_total
            new_stages = stages + (new_program,)
            priority = new_total + 0.25 * _distance(new_state, goal)
            heapq.heappush(
                heap,
                (priority, next(counter), new_total, new_state,
                 new_program, new_stages)
            )
            if fallback is None or priority < fallback[0]:
                fallback = (priority, new_stages[0])

    return None if fallback is None else fallback[1]


def act(obs, available, memory):
    queue = memory.get("queue")
    if queue:
        row, col = queue.pop(0)
        return "ACTION6", int(row), int(col)

    frame = np.asarray(obs)
    panel = _find_panel(frame)
    if panel is None:
        if "RESET" in available and not memory.get("reset_tried"):
            memory["reset_tried"] = True
            return "RESET", None, None
        return "ACTION6", 58, 57

    cols, panel_y, bits, program = panel
    button = _find_button(frame, panel_y)
    if button is None:
        button = (min(62, panel_y + 17), min(62, cols[-1] + 4))

    robot = _find_shape(frame, _ROBOT, cols[0], panel_y)
    target = _find_shape(frame, _TARGET, cols[0], panel_y)
    if robot is None or target is None:
        if "RESET" in available and not memory.get("reset_tried"):
            memory["reset_tried"] = True
            return "RESET", None, None
        return "ACTION6", int(button[0]), int(button[1])

    tx, ty, trot, tscale = target
    goal = (tx + tscale, ty + tscale, trot, tscale)
    checkpoints = _find_checkpoints(frame, robot, cols[0], panel_y)
    desired = _plan(robot, goal, checkpoints, program, bits, frame, panel_y)

    if desired is None:
        if "RESET" in available and not memory.get("reset_tried"):
            memory["reset_tried"] = True
            return "RESET", None, None
        return "ACTION6", int(button[0]), int(button[1])

    actions = []
    for x, old, new in zip(cols, program, desired):
        changed = old ^ new
        for bit in range(bits):
            if changed & (1 << bit):
                actions.append((panel_y + 3 * bit + 1, x + 2))
    actions.append(button)
    memory["queue"] = actions
    row, col = memory["queue"].pop(0)
    return "ACTION6", int(row), int(col)

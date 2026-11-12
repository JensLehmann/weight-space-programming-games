import numpy as np
from collections import defaultdict, deque

_AGENT = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=bool)

_GOAL = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

# Operation numbers:
# 0 wait, 1 left, 2 right, 3 down, 4 clockwise, 5 counterclockwise,
# 6 half turn, 7 grow, 8 shrink, 9 right twice, 10 left twice,
# 11 up, 12 color 9, 13 color 8, 14 color 15.
_COMMANDS = {
    0: 0, 1: 1, 2: 2, 3: 3,
    5: 4, 6: 5, 7: 6, 8: 7, 9: 8,
    10: 9, 11: 9, 12: 10, 13: 10,
    14: 12, 15: 13, 16: 5,
    33: 11, 34: 1, 63: 14,
}


def _program(obs):
    """Find glyph columns on the editable (black) program panel."""
    columns = defaultdict(list)
    for y in range(22, 53):
        for x in range(2, 62):
            c = int(obs[y, x])
            if c not in (1, 5):
                continue
            p = obs[y-1:y+2, x-1:x+2]
            horizontal = (
                np.all(p[1] == c)
                and np.all(p[0] == 0)
                and np.all(p[2] == 0)
                and obs[y, x-2] == 0 and obs[y, x+2] == 0
            )
            vertical = (
                np.all(p[:, 1] == c)
                and np.all(p[:, 0] == 0)
                and np.all(p[:, 2] == 0)
                and obs[y-2, x] == 0 and obs[y+2, x] == 0
            )
            if horizontal or vertical:
                columns[x].append((y, 0 if horizontal else 1))

    groups = defaultdict(list)
    for x, entries in columns.items():
        entries = sorted(set(entries))
        runs = []
        run = []
        for item in entries:
            if run and not (
                item[0] == run[-1][0] + 3 and item[1] != run[-1][1]
            ):
                runs.append(run)
                run = []
            run.append(item)
        if run:
            runs.append(run)
        for run in runs:
            if len(run) >= 2 and run[0][1] == 0:
                groups[tuple(run)].append(x)

    choices = []
    for signature, xs in groups.items():
        xs.sort()
        chunks = []
        chunk = []
        for x in xs:
            if chunk and x != chunk[-1] + 5:
                chunks.append(chunk)
                chunk = []
            chunk.append(x)
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            if len(chunk) >= 2:
                choices.append((chunk, [v[0] for v in signature]))

    if not choices:
        return None
    xs, ys = max(
        choices,
        key=lambda q: (max(q[0]) >= 32, max(q[0]), len(q[0]) * len(q[1]))
    )
    values = [
        sum((int(obs[y, x]) == 5) << i for i, y in enumerate(ys))
        for x in xs
    ]
    return xs, ys, values


def _button(obs):
    mask = ((obs == 9) | (obs == 10)).copy()
    mask[:48] = False
    points = set(zip(*np.nonzero(mask)))
    candidates = []
    while points:
        seed = points.pop()
        stack = [seed]
        component = [seed]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (y + dy, x + dx)
                if q in points:
                    points.remove(q)
                    stack.append(q)
                    component.append(q)
        if len(component) < 45:
            continue
        yy = [p[0] for p in component]
        xx = [p[1] for p in component]
        if max(yy) - min(yy) >= 6 and max(xx) - min(xx) >= 6:
            candidates.append((
                (min(xx) + max(xx)) // 2,
                (min(yy) + max(yy)) // 2,
            ))
    if candidates:
        x, y = max(candidates)
        return y, x
    return 58, 57


def _bounds(obs, xs, ys):
    bottom = min(ys) - 1
    candidates = []
    for y in range(2, max(3, bottom - 2)):
        x = 0
        while x < 64:
            if obs[y, x] != 0:
                x += 1
                continue
            start = x
            while x < 64 and obs[y, x] == 0:
                x += 1
            if x - start >= 16 and start < min(xs) and x - 1 > max(xs):
                candidates.append((y, start, x - 1))
    if candidates:
        y, left, right = min(candidates)
        return left + 1, y + 1, right, bottom
    if min(xs) < 30:
        return 14, 9, 50, bottom
    return 33, 4, 61, bottom


def _scaled(mask, scale):
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def _matches(obs, bounds, base, colors, maxscale=4, strict=False):
    """Template matches, scored by unexpected same-colored pixels."""
    left, top, right, bottom = bounds
    x0, y0 = max(0, left - 1), max(0, top - 1)
    x1, y1 = min(64, right + 1), min(64, bottom + 1)
    crop = obs[y0:y1, x0:x1]
    result = []
    for scale in range(1, maxscale + 1):
        for rotation in range(4):
            mask = _scaled(np.rot90(base, -rotation), scale)
            h, w = mask.shape
            if h > crop.shape[0] or w > crop.shape[1]:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(crop, (h, w))
            for color in colors:
                good = np.all(windows[..., mask] == color, axis=-1)
                for yy, xx in zip(*np.nonzero(good)):
                    y, x = int(yy) + y0, int(xx) + x0
                    patch = obs[y:y+h, x:x+w]
                    extra = int(np.count_nonzero(patch[~mask] == color))
                    if strict and extra:
                        continue
                    rim = 0
                    if y > 0:
                        rim += int(np.count_nonzero(obs[y-1, x:x+w] == color))
                    if y + h < 64:
                        rim += int(np.count_nonzero(obs[y+h, x:x+w] == color))
                    if x > 0:
                        rim += int(np.count_nonzero(obs[y:y+h, x-1] == color))
                    if x + w < 64:
                        rim += int(np.count_nonzero(obs[y:y+h, x+w] == color))
                    score = extra / (scale * scale) + 0.25 * rim / scale
                    result.append((score, x, y, scale, rotation, int(color)))
    result.sort(key=lambda a: (a[0], -a[3], a[2], a[1]))
    return result


def _actor(obs, bounds):
    matches = _matches(obs, bounds, _AGENT, (11, 9, 8, 15, 14))
    if not matches:
        return None
    _, x, y, s, r, c = matches[0]
    return x, y, s, r, c


def _target(obs, bounds):
    matches = _matches(obs, bounds, _GOAL, (11, 9, 8, 15, 14))
    if not matches:
        return None
    _, x, y, s, r, c = matches[0]
    return x + s, y + s, s, r, c


def _checkpoints(obs, bounds, actor):
    checker = (np.indices((4, 4)).sum(axis=0) % 2 == 0)
    matches = _matches(obs, bounds, checker, (11,), maxscale=3, strict=True)
    found = set()
    ax, ay = actor[:2]
    left, top, right, bottom = bounds
    for _, x, y, s, _, _ in matches:
        if (
            (x - ax) % 4 == 0 and (y - ay) % 4 == 0
            and left <= x < right and top <= y < bottom
        ):
            found.add((x, y, s))
    return sorted(found)


def _cell_rectangles(mask, bounds):
    left, top, right, bottom = bounds
    rectangles = []
    for y in range(top, bottom, 4):
        for x in range(left, right, 4):
            if np.any(mask[y:min(y+4, bottom), x:min(x+4, right)]):
                rectangles.append((x, y, x + 4, y + 4))
    return rectangles


def _scene(obs, bounds, actor, goal, checkpoints):
    walls = _cell_rectangles(obs == 6, bounds)
    unusual = ~np.isin(obs, (4, 5, 6))
    left, top, right, bottom = bounds
    roi = np.zeros((64, 64), dtype=bool)
    roi[top:bottom, left:right] = True
    unusual &= roi

    def erase(x, y, w, h):
        unusual[max(0, y):min(64, y+h), max(0, x):min(64, x+w)] = False

    x, y, s = actor[:3]
    erase(x, y, 4*s, 4*s)
    x, y, s = goal[:3]
    erase(x-s, y-s, 6*s, 6*s)
    for x, y, s in checkpoints:
        erase(x, y, 4*s, 4*s)
    hazards = _cell_rectangles(unusual, bounds)
    return walls, hazards


def _overlap(x, y, size, rectangle):
    a, b, c, d = rectangle
    return x < c and x + size > a and y < d and y + size > b


def _plan(start, goal, bounds, checkpoints, walls, hazards,
          old_program, nbits, margin=0, ignore_hazards=False, attempt=0):
    length = len(old_program)
    left, top, right, bottom = bounds
    ax, ay, initial_scale, initial_rot, initial_color = start
    maxscale = max([initial_scale, goal[2]] + [p[2] for p in checkpoints])
    blockers = walls + ([] if ignore_hazards else hazards)

    def blocked(x, y, scale):
        size = 4 * scale
        return any(_overlap(x, y, size, rect) for rect in blockers)

    spatial = []
    for s in range(1, maxscale + 1):
        lo_x, hi_x = left - margin, right + margin - 4*s
        lo_y, hi_y = top - margin, bottom + margin - 4*s
        first_x = ax + 4 * ((lo_x - ax + 3) // 4)
        first_y = ay + 4 * ((lo_y - ay + 3) // 4)
        for y in range(first_y, hi_y + 1, 4):
            for x in range(first_x, hi_x + 1, 4):
                if not blocked(x, y, s):
                    spatial.append((x, y, s))
    for p in (tuple(start[:3]), tuple(goal[:3])):
        if p not in spatial:
            spatial.append(p)
    spatial_index = {p: i for i, p in enumerate(spatial)}

    colors = sorted(set((initial_color, goal[4])))
    nc = len(colors)
    color_index = {c: i for i, c in enumerate(colors)}
    state_count = len(spatial) * 4 * nc

    def index(sp, rot, color):
        return (sp * 4 + rot) * nc + color_index[color]

    start_id = index(spatial_index[tuple(start[:3])], initial_rot, initial_color)
    goal_id = index(spatial_index[tuple(goal[:3])], goal[3], goal[4])

    codes = defaultdict(list)
    limit = 1 << min(nbits, 8)
    for code in range(limit):
        codes[_COMMANDS.get(code, 0)].append(code)

    transitions = {}
    deltas = {1: (-4, 0), 2: (4, 0), 3: (0, 4),
              9: (8, 0), 10: (-8, 0), 11: (0, -4)}
    paint = {12: 9, 13: 8, 14: 15}

    for op in sorted(codes):
        if op in paint and paint[op] not in color_index:
            continue
        out = [-1] * state_count
        for sp, (x, y, s) in enumerate(spatial):
            destination_sp = sp
            if op in deltas:
                dx, dy = deltas[op]
                nx, ny = x + dx, y + dy
                if not blocked(nx, ny, s):
                    destination_sp = spatial_index.get((nx, ny, s), -1)
            elif op in (7, 8):
                ns = s + 1 if op == 7 else max(1, s - 1)
                if not blocked(x, y, ns):
                    destination_sp = spatial_index.get((x, y, ns), -1)
            if destination_sp < 0:
                continue
            for r in range(4):
                nr = (r + {4: 1, 5: -1, 6: 2}.get(op, 0)) % 4
                for c in colors:
                    new_color = paint.get(op, c)
                    out[index(sp, r, c)] = index(destination_sp, nr, new_color)
        transitions[op] = out

    reverse = [[] for _ in range(state_count)]
    for source in range(state_count):
        destinations = {
            out[source] for out in transitions.values()
            if out[source] >= 0 and out[source] != source
        }
        for destination in destinations:
            reverse[destination].append(source)

    checkpoint_state = [False] * state_count
    for sp, (x, y, s) in enumerate(spatial):
        is_checkpoint = any(
            s == cs and _overlap(x, y, 4*s, (cx, cy, cx+4*cs, cy+4*cs))
            for cx, cy, cs in checkpoints
        )
        if is_checkpoint:
            begin = sp * 4 * nc
            checkpoint_state[begin:begin + 4*nc] = [True] * (4*nc)

    # Backward waves: each wave represents one more whole program run.
    infinity = 10**9
    runs = [infinity] * state_count
    runs[goal_id] = 0
    frontier = [goal_id]
    used_terminals = {goal_id}
    wave = 0
    while frontier and wave < 20:
        wave += 1
        seen = set(frontier)
        queue = deque((v, 0) for v in frontier)
        while queue:
            v, depth = queue.popleft()
            if runs[v] > wave:
                runs[v] = wave
            if depth == length:
                continue
            for u in reverse[v]:
                if u not in seen:
                    seen.add(u)
                    queue.append((u, depth + 1))
        frontier = [
            v for v in seen
            if checkpoint_state[v] and v not in used_terminals
        ]
        used_terminals.update(frontier)
        if runs[start_id] < infinity:
            break

    if runs[start_id] == infinity:
        return None

    distance = [infinity] * state_count
    distance[goal_id] = 0
    queue = deque([goal_id])
    while queue:
        v = queue.popleft()
        for u in reverse[v]:
            if distance[u] == infinity:
                distance[u] = distance[v] + 1
                queue.append(u)

    # Dynamic programming through the program slots, minimizing edit clicks.
    dp = {start_id: 0}
    predecessors = []
    for slot, old in enumerate(old_program):
        next_dp = {}
        previous = {}
        for op, out in transitions.items():
            code = min(codes[op], key=lambda c: ((c ^ old).bit_count(), c))
            weight = (code ^ old).bit_count() * 10000
            if attempt:
                weight += ((slot + 3) * (op + 7) * (attempt * 13 + 5)) % 37
            for state, cost in dp.items():
                destination = out[state]
                if destination < 0:
                    continue
                new_cost = cost + weight
                if new_cost < next_dp.get(destination, infinity):
                    next_dp[destination] = new_cost
                    previous[destination] = (state, code)
        if not next_dp:
            return None
        predecessors.append(previous)
        dp = next_dp

    candidates = []
    for state, cost in dp.items():
        if state == goal_id:
            candidates.append((0, cost / 10000.0, state))
        elif checkpoint_state[state] and runs[state] < runs[start_id]:
            estimate = cost / 10000.0 + 0.4 * distance[state]
            candidates.append((runs[state], estimate, state))
    if not candidates:
        return None
    _, _, endpoint = min(candidates)

    program = [0] * length
    state = endpoint
    for slot in range(length - 1, -1, -1):
        state, code = predecessors[slot][state]
        program[slot] = code

    ci = endpoint % nc
    remainder = endpoint // nc
    rotation = remainder % 4
    sp = remainder // 4
    x, y, s = spatial[sp]
    end_state = (x, y, s, rotation, colors[ci])
    return program, end_state


def _reset(memory):
    count = int(memory.get("resets", 0)) + 1
    memory.clear()
    memory["resets"] = count
    return "RESET", None, None


def act(obs, available, memory):
    obs = np.asarray(obs)
    parsed = _program(obs)
    if parsed is None:
        return _reset(memory)
    xs, ys, values = parsed
    signature = (tuple(xs), tuple(ys))

    if memory.get("signature") != signature:
        resets = memory.get("resets", 0)
        memory.clear()
        memory["resets"] = resets
        memory["signature"] = signature

    button_row, button_col = _button(obs)

    # A plan remains fixed while its individual switches are being edited.
    if "program" in memory:
        desired = memory["program"]
        for column, (current, target) in enumerate(zip(values, desired)):
            difference = current ^ target
            if difference:
                bit = (difference & -difference).bit_length() - 1
                return "ACTION6", int(ys[bit]), int(xs[column])
        memory["pending"] = {
            "expected": memory["endpoint"],
            "before": memory["start"],
        }
        del memory["program"]
        return "ACTION6", int(button_row), int(button_col)

    bounds = tuple(memory.get("bounds", _bounds(obs, xs, ys)))
    actor = _actor(obs, bounds)
    if actor is None:
        return _reset(memory)

    pending = memory.pop("pending", None)
    if pending is not None:
        expected = tuple(pending["expected"])
        if tuple(actor) != expected:
            memory["attempt"] = int(memory.get("attempt", 0)) + 1
        else:
            memory["attempt"] = 0

    if "goal" in memory:
        goal = tuple(memory["goal"])
    else:
        goal = _target(obs, bounds)
        if goal is None:
            return _reset(memory)

    checkpoints = set(tuple(p) for p in memory.get("checkpoints", []))
    checkpoints.update(_checkpoints(obs, bounds, actor))
    checkpoints = sorted(checkpoints)

    walls, hazards = _scene(obs, bounds, actor, goal, checkpoints)
    # Keep walls already seen, including portions later covered by sprites.
    walls = sorted(set(tuple(r) for r in memory.get("walls", [])) | set(walls))

    memory["bounds"] = bounds
    memory["goal"] = goal
    memory["checkpoints"] = checkpoints
    memory["walls"] = walls

    attempt = int(memory.get("attempt", 0))
    solution = None
    for margin, ignore_hazards in ((0, False), (8, False), (12, True)):
        solution = _plan(
            actor, goal, bounds, checkpoints, walls, hazards,
            values, len(ys), margin, ignore_hazards, attempt
        )
        if solution is not None:
            break

    if solution is None:
        return _reset(memory)

    desired, endpoint = solution
    memory["program"] = desired
    memory["endpoint"] = endpoint
    memory["start"] = actor

    for column, (current, target) in enumerate(zip(values, desired)):
        difference = current ^ target
        if difference:
            bit = (difference & -difference).bit_length() - 1
            return "ACTION6", int(ys[bit]), int(xs[column])

    memory["pending"] = {"expected": endpoint, "before": actor}
    del memory["program"]
    return "ACTION6", int(button_row), int(button_col)

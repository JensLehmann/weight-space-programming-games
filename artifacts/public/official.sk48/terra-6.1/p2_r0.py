import heapq
import itertools
import numpy as np


# Head template, including the transparent opening.
_HEAD = np.array([
    [6, 6, 6, 6, 6, 6],
    [6, 4, 4, 4, 4, 6],
    [6, 4, 6, 6, 4, -1],
    [6, 4, 6, 6, 4, -1],
    [6, 4, 4, 4, 4, 6],
    [6, 6, 6, 6, 6, 6],
], dtype=np.int8)

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))
_HEAD_MASKS = []
for _rotation in range(4):
    _t = np.rot90(_HEAD, -_rotation)
    _HEAD_MASKS.append((
        [(int(y), int(x)) for y, x in zip(*np.where(_t == 6))],
        [(int(y), int(x)) for y, x in zip(*np.where(_t == 4))],
    ))


def _signature(state):
    return repr(state)


def _read_frame(obs):
    a = np.asarray(obs)
    if a.shape != (64, 64):
        return None

    # The playfield is the large rectangle of colour 4 above the timer.
    row_counts = np.sum(a[:53, :] == 4, axis=1)
    peak = int(row_counts.max())
    if peak < 10:
        return None
    row_ids = np.where(row_counts >= max(8, int(peak * 0.42)))[0]
    if len(row_ids) == 0:
        return None
    by0, by1 = int(row_ids[0]), int(row_ids[-1]) + 1

    col_counts = np.sum(a[by0:by1, :] == 4, axis=0)
    col_peak = int(col_counts.max())
    col_ids = np.where(col_counts >= max(7, int(col_peak * 0.40)))[0]
    if len(col_ids) == 0:
        return None
    bx0, bx1 = int(col_ids[0]), int(col_ids[-1]) + 1
    if bx1 - bx0 < 12 or by1 - by0 < 12:
        return None
    bounds = (bx0, by0, bx1, by1)

    # Identify coloured 4x4 blocks. The target indicators can cover the
    # central 2x2 pixels in black, so only the corners must be intact.
    blocks = []
    for y in range(61):
        for x in range(61):
            c = int(a[y, x])
            if c <= 5:
                continue
            if (int(a[y, x + 3]) != c or
                    int(a[y + 3, x]) != c or
                    int(a[y + 3, x + 3]) != c):
                continue
            patch = a[y:y + 4, x:x + 4]
            if int(np.count_nonzero((patch == c) | (patch == 0))) == 16:
                blocks.append((c, x - 1, y - 1))

    # Find the square heads and their orientation.
    heads = []
    for y in range(59):
        for x in range(59):
            c = int(a[y, x])
            if c <= 5:
                continue
            if (int(a[y, x + 5]) != c or
                    int(a[y + 5, x]) != c or
                    int(a[y + 5, x + 5]) != c):
                continue
            if any(int(a[y + dy, x + dx]) != c
                   for dy, dx in ((2, 2), (2, 3), (3, 2), (3, 3))):
                continue
            for rotation, (ring, inside) in enumerate(_HEAD_MASKS):
                if not all(int(a[y + dy, x + dx]) == c for dy, dx in ring):
                    continue
                v = int(a[y + inside[0][0], x + inside[0][1]])
                if v not in (0, 4):
                    continue
                if not all(int(a[y + dy, x + dx]) == v for dy, dx in inside):
                    continue
                heads.append((c, x, y, rotation, v == 0, True))
                break

    # Some fixed, non-clickable rods have a black head with a small 11 mark.
    # Its opening reveals the direction of the rod.
    for y in range(61):
        for x in range(61):
            if not (int(a[y, x]) == 11 and int(a[y, x + 1]) == 11 and
                    int(a[y + 1, x]) == 11 and int(a[y + 1, x + 1]) == 11):
                continue
            if ((x > 0 and int(a[y, x - 1]) == 11) or
                    (y > 0 and int(a[y - 1, x]) == 11) or
                    (x + 2 < 64 and int(a[y, x + 2]) == 11) or
                    (y + 2 < 64 and int(a[y + 2, x]) == 11)):
                continue
            hx, hy = x - 2, y - 2
            if hx < 0 or hy < 0 or hx + 5 >= 64 or hy + 5 >= 64:
                continue
            if any(h[1] == hx and h[2] == hy for h in heads):
                continue
            openings = (
                ((hx + 5, hy + 2), (hx + 5, hy + 3)),
                ((hx + 2, hy + 5), (hx + 3, hy + 5)),
                ((hx, hy + 2), (hx, hy + 3)),
                ((hx + 2, hy), (hx + 3, hy)),
            )
            possible = []
            for r, points in enumerate(openings):
                if all(int(a[py, px]) in (1, 2, 3) for px, py in points):
                    possible.append(r)
            if len(possible) == 1:
                heads.append((11, hx, hy, possible[0], False, False))

    def rod_present(x, y, rotation):
        if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
            return False
        if rotation % 2 == 0:
            pts = ((x, y + 2), (x, y + 3),
                   (x + 5, y + 2), (x + 5, y + 3))
        else:
            pts = ((x + 2, y), (x + 3, y),
                   (x + 2, y + 5), (x + 3, y + 5))
        return sum(int(a[py, px]) in (1, 2, 3) for px, py in pts) >= 3

    upper = [h for h in heads if h[2] < 53]
    lower = [h for h in heads if h[2] >= 53]
    if not upper:
        return None
    upper.sort(key=lambda h: (h[0], h[2], h[1], h[3]))
    lower.sort(key=lambda h: (h[0], h[2], h[1], h[3]))

    block_at = {(x, y): c for c, x, y in blocks}
    main_blocks = sorted(
        [(c, x, y) for c, x, y in blocks if y < 53],
        key=lambda b: (b[0], b[2], b[1])
    )

    rods = []
    head_states = []
    active = None
    for i, h in enumerate(upper):
        c, x, y, rotation, is_active, clickable = h
        dx, dy = _DIRS[rotation]
        length = 1
        for k in range(1, 12):
            if rod_present(x + 6 * k * dx, y + 6 * k * dy, rotation):
                length += 1
            else:
                break

        target = None
        match = next((g for g in lower if g[0] == c), None)
        if match is not None:
            _, gx, gy, gr, _, _ = match
            gdx, gdy = _DIRS[gr]
            target = []
            for k in range(1, 12):
                xx, yy = gx + 6 * k * gdx, gy + 6 * k * gdy
                if not rod_present(xx, yy, gr):
                    break
                if (xx, yy) in block_at:
                    target.append(block_at[(xx, yy)])
            target = tuple(target)
        rods.append((dx, dy, target, clickable and match is not None))
        head_states.append((x, y, length))
        if is_active:
            active = i

    if active is None:
        active = next((i for i, r in enumerate(rods) if r[3]), 0)

    colors = tuple(b[0] for b in main_blocks)
    positions = tuple((b[1], b[2]) for b in main_blocks)

    # Full black 6x6 cells inside the playfield are obstacles.
    obstacles = []
    for y in range(by0, by1 - 5, 6):
        for x in range(bx0, bx1 - 5, 6):
            if np.all(a[y:y + 6, x:x + 6] == 5):
                obstacles.append((x, y))

    # Equal-colour blocks are interchangeable; canonicalize their positions.
    state = (tuple(head_states), positions, active)
    problem = (bounds, tuple(obstacles), tuple(rods), colors)
    return problem, state


def _canonical_blocks(positions, colors):
    result = list(positions)
    start = 0
    while start < len(colors):
        end = start + 1
        while end < len(colors) and colors[end] == colors[start]:
            end += 1
        result[start:end] = sorted(result[start:end], key=lambda p: (p[1], p[0]))
        start = end
    return tuple(result)


def _transition(problem, state, action):
    bounds, obstacles, rods, colors = problem
    hs, bs, active = state
    bx0, by0, bx1, by1 = bounds

    if action >= 4:
        new_active = action - 4
        if new_active == active or not rods[new_active][3]:
            return None
        return (hs, bs, new_active)

    mx, my = _MOVES[action]
    step_x, step_y = 6 * mx, 6 * my
    hx, hy, length = hs[active]
    dx, dy, _, _ = rods[active]

    def blocked(x, y, move_x, move_y):
        x += 6 * move_x
        y += 6 * move_y
        if x < bx0 or y < by0 or x + 6 > bx1 or y + 6 > by1:
            return True
        for ox, oy in obstacles:
            if ox <= x < ox + 6 and oy <= y < oy + 6:
                return True
        return False

    # Segment lookup includes all rods, including fixed rods.
    segment_positions = {}
    segment_map = {}
    for r, (rx, ry, ln) in enumerate(hs):
        rdx, rdy, _, _ = rods[r]
        axis = 0 if rdx else 1
        for k in range(ln):
            pos = (rx + 6 * k * rdx, ry + 6 * k * rdy)
            token = (0, r, k)
            segment_positions[token] = pos
            segment_map.setdefault((pos[0], pos[1], axis), []).append(token)
    block_map = {p: i for i, p in enumerate(bs)}
    moved = set()

    def move_object(token, parent=None):
        if token in moved:
            return True
        is_segment = token[0] == 0
        if is_segment:
            x, y = segment_positions[token]
            r = token[1]
            rdx, rdy, _, _ = rods[r]
        else:
            x, y = bs[token[1]]

        if blocked(x, y, mx, my):
            if not (is_segment and
                    ((rdx, rdy) == (-mx, -my) or blocked(x, y, 0, 0))):
                return False

        if is_segment:
            for px, py in ((x, y), (x + step_x, y + step_y)):
                j = block_map.get((px, py))
                if j is None:
                    continue
                bt = (1, j)
                if move_object(bt, token):
                    moved.add(bt)
                elif (rdx == 0) != (mx == 0):
                    return False
        else:
            # A block cannot pass through a perpendicular rod.
            perpendicular_axis = 1 if mx else 0
            for px, py in ((x, y), (x + step_x, y + step_y)):
                for other in segment_map.get((px, py, perpendicular_axis), ()):
                    if other != parent:
                        return False
            j = block_map.get((x + step_x, y + step_y))
            if j is not None and not move_object((1, j)):
                return False

        moved.add(token)
        return True

    forward = (mx, my) == (dx, dy)
    backward = (mx, my) == (-dx, -dy)

    if forward:
        last_x = hx + 6 * (length - 1) * dx
        last_y = hy + 6 * (length - 1) * dy
        if blocked(last_x, last_y, mx, my):
            return None
        indices = range(length)
        new_head = (hx, hy, length + 1)
    elif backward:
        if length <= 1:
            return None
        indices = range(1, length)
        new_head = (hx, hy, length - 1)
    else:
        # The controls move along the rail bordering the playfield.
        nx, ny = hx + step_x, hy + step_y
        if dx:
            if ny < by0 or ny + 6 > by1:
                return None
        else:
            if nx < bx0 or nx + 6 > bx1:
                return None
        indices = range(length)
        new_head = (nx, ny, length)

    required = [(0, active, k) for k in indices]
    for token in required:
        ok = move_object(token)
        if not ok and not (forward or backward):
            return None

    # Unusual partial rope movements can make the internal segment list
    # non-contiguous. Avoid approximating those as ordinary movements.
    if any(token not in moved for token in required):
        return None

    new_blocks = list(bs)
    for token in moved:
        if token[0] == 1:
            i = token[1]
            x, y = new_blocks[i]
            new_blocks[i] = (x + step_x, y + step_y)

    new_heads = list(hs)
    new_heads[active] = new_head
    new_blocks = _canonical_blocks(new_blocks, colors)
    result = (tuple(new_heads), new_blocks, active)
    return None if result == state else result


def _goal_and_heuristic(problem, state):
    _, _, rods, colors = problem
    hs, bs, _ = state
    total = 0.0
    solved = True

    for r, (dx, dy, target, _) in enumerate(rods):
        if target is None:
            continue
        hx, hy, length = hs[r]
        occupied = {}
        for i, (x, y) in enumerate(bs):
            occupied[(x, y)] = colors[i]
        actual = []
        for k in range(length):
            p = (hx + 6 * k * dx, hy + 6 * k * dy)
            if p in occupied:
                actual.append(occupied[p])

        prefix = 0
        while (prefix < len(target) and prefix < len(actual) and
               actual[prefix] == target[prefix]):
            prefix += 1
        if prefix < len(target):
            solved = False
        if not target:
            continue

        # Dynamic programming estimates how difficult it is to align blocks
        # of the requested colours in the requested order.
        projections = []
        costs = []
        for x, y in bs:
            along = ((x - hx) * dx + (y - hy) * dy) / 6.0
            across = abs((x - hx) * dy - (y - hy) * dx) / 6.0
            cost = (across + max(0.0, 1.0 - along) +
                    0.55 * max(0.0, along - (length - 1)))
            projections.append(along)
            costs.append(cost)

        previous = {}
        for wanted in target:
            current = {}
            for i, color in enumerate(colors):
                if color != wanted:
                    continue
                if not previous:
                    current[i] = costs[i]
                else:
                    best = float("inf")
                    for j, old_cost in previous.items():
                        if i == j:
                            continue
                        inversion = max(0.0, 1.0 + projections[j] - projections[i])
                        value = old_cost + costs[i] + 1.2 * inversion
                        if value < best:
                            best = value
                    if best < float("inf"):
                        current[i] = best
            previous = current
            if not previous:
                break
        alignment = min(previous.values()) if previous else 30.0
        total += alignment + 2.0 * (len(target) - prefix)

    return solved, total


def _search(problem, start, recent, limit=20000):
    rods = problem[2]
    switch_actions = [4 + i for i, r in enumerate(rods) if r[3]]
    actions = list(range(4)) + switch_actions

    solved, initial_h = _goal_and_heuristic(problem, start)
    if solved:
        # A move is needed to trigger the game's completion check.
        for action in actions:
            nxt = _transition(problem, start, action)
            if nxt is not None and _goal_and_heuristic(problem, nxt)[0]:
                return [(action, nxt)]
        return []

    counter = itertools.count()
    heap = [(2.0 * initial_h, 0, next(counter), start)]
    distance = {start: 0}
    parent = {}
    forbidden = set(recent[-30:])
    forbidden.discard(_signature(start))
    best = None
    best_value = float("inf")
    expanded = 0

    while heap and expanded < limit:
        _, g, _, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        if state != start and _signature(state) in forbidden:
            continue
        expanded += 1

        is_goal, h = _goal_and_heuristic(problem, state)
        if is_goal:
            best = state
            break
        if state != start:
            value = h + 0.025 * g
            if value < best_value:
                best_value = value
                best = state

        for action in actions:
            if action >= 4 and action - 4 == state[2]:
                continue
            nxt = _transition(problem, state, action)
            if nxt is None:
                continue
            ng = g + 1
            if ng >= distance.get(nxt, 10**9):
                continue
            distance[nxt] = ng
            parent[nxt] = (state, action)
            _, nh = _goal_and_heuristic(problem, nxt)
            heapq.heappush(heap, (ng + 2.0 * nh, ng, next(counter), nxt))

    if best is None:
        return []
    path = []
    while best != start:
        previous, action = parent[best]
        path.append((action, best))
        best = previous
    path.reverse()
    return path


def _make_output(action, state):
    if action < 4:
        return ("ACTION" + str(action + 1), None, None)
    selected = action - 4
    x, y, _ = state[0][selected]
    return ("ACTION6", int(y), int(x))


def act(obs, available, memory):
    parsed = _read_frame(obs)
    if parsed is None:
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION1", None, None)

    problem, state = parsed
    signature = _signature(state)
    recent = memory.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    if not recent or recent[-1] != signature:
        recent.append(signature)
    memory["recent"] = recent[-80:]

    plan = memory.get("plan", [])
    if memory.get("expected") != signature:
        plan = []

    if not plan:
        path = _search(problem, state, memory["recent"])
        plan = []
        previous = state
        for action, successor in path:
            output = _make_output(action, previous)
            plan.append([output[0], output[1], output[2], _signature(successor)])
            previous = successor

    if plan:
        step = plan.pop(0)
        memory["plan"] = plan
        memory["expected"] = step[3]
        if step[0] in available:
            return (step[0], step[1], step[2])

    memory["plan"] = []
    memory["expected"] = None
    # If the approximation cannot find a route, try a legal state-changing
    # move rather than repeatedly issuing a blocked action.
    for action in range(4):
        nxt = _transition(problem, state, action)
        name = "ACTION" + str(action + 1)
        if nxt is not None and name in available:
            if _signature(nxt) not in memory["recent"][-15:]:
                return (name, None, None)
    if "RESET" in available:
        return ("RESET", None, None)
    return ("ACTION1", None, None)

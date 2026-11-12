import time


def _read_targets(g):
    targets = []
    for y in range(1, 53):
        for x in range(1, 58):
            c = g[y][x]
            if c == 5:
                continue
            if any(g[y - 1][xx] != 5 or g[y + 6][xx] != 5
                   for xx in range(x - 1, x + 7)):
                continue
            if any(g[yy][x - 1] != 5 or g[yy][x + 6] != 5
                   for yy in range(y, y + 6)):
                continue
            if any(g[y][xx] != c or g[y + 5][xx] != c
                   for xx in range(x, x + 6)):
                continue
            if any(g[yy][x] != c or g[yy][x + 5] != c
                   for yy in range(y + 1, y + 5)):
                continue
            if any(g[yy][xx] != 5
                   for yy in range(y + 1, y + 5)
                   for xx in range(x + 1, x + 5)):
                continue
            targets.append((y, x, c))
    targets.sort()
    return [c for _, _, c in targets]


def _read_frames(g):
    found = []
    for y in range(8, 44):
        for x in range(64):
            c = g[y + 2][x]
            if c == 4:
                continue
            if any(g[yy][x] != c for yy in range(y + 2, y + 8)):
                continue
            for n in range(1, 10):
                w = 6 * n + 4
                right = x + w - 1
                if right >= 64:
                    break
                if any(g[yy][right] != c
                       for yy in range(y + 2, y + 8)):
                    continue
                # The root's black selection corners can cover two pixels
                # at either end of the top and bottom borders.
                if any(g[y][xx] != c or g[y + 9][xx] != c
                       for xx in range(x + 2, right - 1)):
                    continue
                found.append((y, x, n, c))
    found.sort()
    return found


def _tile_at(g, x, y):
    """Read a 6x6 tile at its sprite origin."""
    if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
        return None
    c = g[y + 1][x + 1]
    if c == 4:
        return None
    if any(g[y + 1][xx] != c or g[y + 4][xx] != c
           for xx in range(x + 1, x + 5)):
        return None
    if any(g[yy][x + 1] != c or g[yy][x + 4] != c
           for yy in range(y + 1, y + 5)):
        return None
    middle = [g[yy][xx]
              for yy in range(y + 2, y + 4)
              for xx in range(x + 2, x + 4)]
    if all(v == c for v in middle):
        return ("term", c)
    if all(v == 4 for v in middle):
        return ("ref", c)
    return None


def _read_inventory(g):
    items = []
    for y in range(54, 59):
        for x in range(0, 59):
            item = _tile_at(g, x, y)
            if item is not None:
                items.append((item, x, y))
    return items


def _selected_tile(g):
    # Selection is displayed as a black 6x6 outline around the tile.
    for y in range(8, 59):
        for x in range(0, 59):
            if any(g[y][xx] != 0 or g[y + 5][xx] != 0
                   for xx in range(x, x + 6)):
                continue
            if any(g[yy][x] != 0 or g[yy][x + 5] != 0
                   for yy in range(y + 1, y + 5)):
                continue
            if _tile_at(g, x, y) is not None:
                return x, y
    return None


def _solve(g):
    target = _read_targets(g)
    frames = _read_frames(g)
    inventory = _read_inventory(g)
    if not target or not frames:
        return None

    # A reference invokes the first frame of its color, in reading order.
    frame_by_color = {}
    for fi, (_, _, _, color) in enumerate(frames):
        if color not in frame_by_color:
            frame_by_color[color] = fi

    slots = []
    frame_slots = []
    fixed = []
    for y, x, n, _ in frames:
        ids = []
        for j in range(n):
            sx, sy = x + 2 + 6 * j, y + 2
            ids.append(len(slots))
            slots.append((sx, sy))
            fixed.append(_tile_at(g, sx, sy))
        frame_slots.append(ids)

    types = sorted(set(item for item, _, _ in inventory) |
                   set(item for item in fixed if item is not None))
    if not types:
        return None
    type_id = {t: i for i, t in enumerate(types)}
    assignment = [-1 if t is None else type_id[t] for t in fixed]
    counts = [0] * len(types)
    donors = [[] for _ in types]
    for item, x, y in inventory:
        k = type_id[item]
        counts[k] += 1
        donors[k].append((x, y))

    # A state contains the call stack, target position, and assignments
    # already made. The trail detects calls that recur without producing
    # a terminal.
    memo = set()
    nodes = [0]
    deadline = time.monotonic() + 4.0
    empty_trail = frozenset()

    def search(stack, out_pos, trail):
        nodes[0] += 1
        if nodes[0] % 512 == 0:
            if time.monotonic() > deadline or nodes[0] > 300000:
                raise TimeoutError

        # Returning from a function advances its caller past the call.
        while stack and stack[-1][1] >= len(frame_slots[stack[-1][0]]):
            stack = stack[:-1]
            if stack:
                f, p = stack[-1]
                stack = stack[:-1] + ((f, p + 1),)
        if not stack:
            return False
        if len(stack) > 100:
            return False

        f, p = stack[-1]
        marker = (f, p)
        if marker in trail:
            return False
        next_trail = trail | frozenset((marker,))

        key = (stack, out_pos, tuple(assignment), trail)
        if key in memo:
            return False
        memo.add(key)

        sid = frame_slots[f][p]
        already = assignment[sid]

        if already >= 0:
            choices = (already,)
        else:
            # A terminal must match the next target color. References are
            # also candidates because their expansion may match it.
            choices = []
            for k, (kind, color) in enumerate(types):
                if counts[k] and kind == "term" and color == target[out_pos]:
                    choices.append(k)
            for k, (kind, color) in enumerate(types):
                if counts[k] and kind == "ref" and color in frame_by_color:
                    choices.append(k)

        for k in choices:
            kind, color = types[k]
            new_assignment = already < 0
            if new_assignment:
                assignment[sid] = k
                counts[k] -= 1

            success = False
            if kind == "term":
                if color == target[out_pos]:
                    if out_pos + 1 == len(target):
                        success = True
                    else:
                        advanced = stack[:-1] + ((f, p + 1),)
                        success = search(advanced, out_pos + 1, empty_trail)
            elif color in frame_by_color:
                # This is the simulator's special recursion guard.
                guarded = (
                    p == 0
                    and len(stack) > 1
                    and (f, 0) in stack[:-1]
                    and stack[-2][1] == 0
                )
                if not guarded:
                    child = frame_by_color[color]
                    success = search(stack + ((child, 0),),
                                     out_pos, next_trail)

            if success:
                return True
            if new_assignment:
                counts[k] += 1
                assignment[sid] = -1
        return False

    try:
        solved = search(((0, 0),), 0, empty_trail)
    except (TimeoutError, RecursionError):
        return None
    if not solved:
        return None

    actions = []
    for sid, k in enumerate(assignment):
        if fixed[sid] is not None or k < 0:
            continue
        if not donors[k]:
            return None
        x, y = donors[k].pop()
        sx, sy = slots[sid]
        actions.append(("ACTION6", y + 2, x + 2))
        actions.append(("ACTION6", sy + 2, sx + 2))
    actions.append(("ACTION5", None, None))
    return actions


def act(obs, available, memory):
    # Finish a previously planned sequence.
    queue = memory.get("queue")
    if queue:
        action = queue.pop(0)
        if action[0] == "ACTION5":
            memory["tested"] = True
        return tuple(action)

    # Ensure the reset is a level reset even if no action has yet been
    # taken: the preceding ACTION7 increments the engine's action count.
    if memory.pop("reset_next", False):
        memory["resets"] = memory.get("resets", 0) + 1
        memory["tested"] = False
        return ("RESET", None, None)

    g = obs.tolist()

    if not memory.get("tested", False):
        selected = _selected_tile(g)
        if selected is not None:
            x, y = selected
            return ("ACTION6", y + 2, x + 2)

        plan = _solve(g)
        if plan is not None:
            energy = sum(v == 2 for v in g[53])
            moves_needed = (len(plan) - 1) // 2
            if energy > moves_needed or memory.get("resets", 0):
                memory["queue"] = plan[1:]
                first = plan[0]
                if first[0] == "ACTION5":
                    memory["tested"] = True
                return first

    if memory.get("resets", 0) == 0:
        memory["reset_next"] = True
        return ("ACTION7", None, None)

    # No model-consistent plan was found, even after a reset.
    return ("ACTION5", None, None)

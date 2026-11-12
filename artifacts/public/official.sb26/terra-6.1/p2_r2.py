from collections import deque


def act(obs, available, memory):
    if memory.get("actions"):
        action, row, col = memory["actions"].pop(0)
        if action == "ACTION5":
            memory["tested"] = True
        return action, row, col

    if memory.pop("tested", False):
        memory.clear()
        return "RESET", None, None

    a = obs

    # Deselect a tile if the episode began with a selection already active.
    for y in list(range(8, 48)) + list(range(54, 59)):
        for x in range(59):
            if int(a[y, x]) != 0 or int(a[y + 5, x]) != 0:
                continue
            if (all(int(a[y, x + k]) == 0 for k in range(6)) and
                all(int(a[y + 5, x + k]) == 0 for k in range(6)) and
                all(int(a[y + k, x]) == 0 for k in range(6)) and
                all(int(a[y + k, x + 5]) == 0 for k in range(6))):
                return "ACTION6", y + 1, x + 1

    # Read the target sequence from the 6-by-6 outlines.
    goals = []
    for y in range(48):
        for x in range(59):
            c = int(a[y, x])
            if c in (4, 5):
                continue
            if (all(int(a[y, x + k]) == c for k in range(6)) and
                all(int(a[y + 5, x + k]) == c for k in range(6)) and
                all(int(a[y + k, x]) == c for k in range(6)) and
                all(int(a[y + k, x + 5]) == c for k in range(6)) and
                all(int(a[y + dy, x + dx]) == 5
                    for dy in range(1, 5) for dx in range(1, 5))):
                goals.append((y, x, c))
    goals.sort()
    target = tuple(c for _, _, c in goals)

    # A program frame has height 10 and width 6*n+4.
    frames = []
    for y in range(8, 45):
        for x in range(64):
            c = int(a[y + 4, x])
            if c == 4:
                continue
            if not all(int(a[y + k, x]) == c for k in range(2, 8)):
                continue
            for n in range(1, 10):
                right = x + 6 * n + 3
                if right >= 64:
                    break
                if not all(int(a[y + k, right]) == c
                           for k in range(2, 8)):
                    continue
                if not all(int(a[y, xx]) == c
                           for xx in range(x + 2, right - 1)):
                    continue
                if y + 9 != 53 and not all(
                    int(a[y + 9, xx]) == c
                    for xx in range(x + 2, right - 1)
                ):
                    continue
                frames.append((y, x, n, c))
    frames.sort()

    # Solid tiles output a colour; hollow tiles call a frame.
    def tile_at(y, x):
        p = [[int(a[y + dy, x + dx]) for dx in range(4)]
             for dy in range(4)]
        c = p[0][0]
        if c == 4:
            return -1
        if all(p[dy][dx] == c for dy in range(4) for dx in range(4)):
            return c
        if (all(p[0][k] == c and p[3][k] == c for k in range(4)) and
            all(p[k][0] == c and p[k][3] == c for k in range(4)) and
            all(p[dy][dx] == 4 for dy in (1, 2) for dx in (1, 2))):
            return 16 + c
        return -1

    slot_positions = []
    initial = []
    frame_slots = []
    for y, x, n, c in frames:
        indices = []
        for i in range(n):
            sx, sy = x + 2 + 6 * i, y + 2
            indices.append(len(slot_positions))
            slot_positions.append((sy, sx))
            initial.append(tile_at(sy + 1, sx + 1))
        frame_slots.append(tuple(indices))

    # Find available tiles below the energy bar.
    inventory = {}
    seen = set()
    for y in range(54, 64):
        for x in range(64):
            c = int(a[y, x])
            if c == 4 or (y, x) in seen:
                continue
            q = deque([(y, x)])
            seen.add((y, x))
            cells = []
            while q:
                yy, xx = q.popleft()
                cells.append((yy, xx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (54 <= ny < 64 and 0 <= nx < 64 and
                        (ny, nx) not in seen and int(a[ny, nx]) == c):
                        seen.add((ny, nx))
                        q.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            if y1 - y0 != 3 or x1 - x0 != 3:
                continue
            kind = tile_at(y0, x0)
            if kind >= 0 and len(cells) in (12, 16):
                inventory.setdefault(kind, []).append((y0 - 1, x0 - 1))

    def fallback():
        if sum(int(a[53, x]) == 2 for x in range(64)) < 64:
            memory.clear()
            return "RESET", None, None
        memory["tested"] = True
        return "ACTION5", None, None

    if not target or not frames:
        return fallback()

    frame_by_color = {}
    for i, (_, _, _, color) in enumerate(frames):
        frame_by_color.setdefault(color, i)

    counts = [0] * 32
    for kind, positions in inventory.items():
        counts[kind] = len(positions)

    initial = tuple(initial)
    counts = tuple(counts)
    visited = set()
    nodes = [0]
    node_limit = 100000

    def solve(pos, stack, assignment, remaining):
        if pos == len(target):
            return assignment
        if not stack or nodes[0] >= node_limit:
            return None

        nodes[0] += 1
        key = (pos, stack, assignment, remaining)
        if key in visited:
            return None
        visited.add(key)

        fi, si, entered_at = stack[-1]
        slots = frame_slots[fi]
        if si >= len(slots):
            return solve(pos, stack[:-1], assignment, remaining)

        slot = slots[si]
        current = assignment[slot]
        next_stack = stack[:-1] + ((fi, si + 1, entered_at),)

        def execute(kind, new_assignment, new_remaining):
            if kind < 16:
                if kind != target[pos]:
                    return None
                return solve(pos + 1, next_stack,
                             new_assignment, new_remaining)

            callee = frame_by_color.get(kind - 16)
            if callee is None:
                return None
            # Re-entering a frame without output would loop forever.
            if any(f == callee and start == pos for f, _, start in stack):
                return None
            return solve(pos, next_stack + ((callee, 0, pos),),
                         new_assignment, new_remaining)

        if current >= 0:
            return execute(current, assignment, remaining)

        choices = []
        wanted = target[pos]
        if remaining[wanted]:
            choices.append(wanted)

        calls = []
        for color, callee in frame_by_color.items():
            kind = 16 + color
            if not remaining[kind]:
                continue
            if any(f == callee and start == pos for f, _, start in stack):
                continue
            first = assignment[frame_slots[callee][0]]
            if 0 <= first < 16 and first != wanted:
                continue
            priority = 0 if first == wanted else (1 if first >= 16 else 2)
            calls.append((priority, kind))
        calls.sort()
        choices.extend(kind for _, kind in calls)

        for kind in choices:
            new_assignment = list(assignment)
            new_assignment[slot] = kind
            new_remaining = list(remaining)
            new_remaining[kind] -= 1
            result = execute(kind, tuple(new_assignment), tuple(new_remaining))
            if result is not None:
                return result
            if nodes[0] >= node_limit:
                break
        return None

    solution = solve(0, ((0, 0, 0),), initial, counts)
    if solution is None:
        return fallback()

    actions = []
    for i, kind in enumerate(solution):
        if initial[i] >= 0 or kind < 0:
            continue
        if not inventory.get(kind):
            return fallback()
        iy, ix = inventory[kind].pop()
        sy, sx = slot_positions[i]
        actions.append(("ACTION6", iy + 2, ix + 2))
        actions.append(("ACTION6", sy + 2, sx + 2))
    actions.append(("ACTION5", None, None))

    memory["actions"] = actions
    action, row, col = memory["actions"].pop(0)
    if action == "ACTION5":
        memory["tested"] = True
    return action, row, col

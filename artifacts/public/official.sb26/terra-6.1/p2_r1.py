from collections import defaultdict


def act(obs, available, memory):
    grid = obs.tolist() if hasattr(obs, "tolist") else obs

    def reset_or_run():
        retries = memory.get("retries", 0)
        if retries < 1:
            memory.clear()
            memory["retries"] = retries + 1
            return ("RESET", None, None)
        return ("ACTION5", None, None)

    def border_is_black(x, y):
        if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
            return False
        return (
            all(grid[y][x + i] == 0 for i in range(6))
            and all(grid[y + 5][x + i] == 0 for i in range(6))
            and all(grid[y + i][x] == 0 for i in range(6))
            and all(grid[y + i][x + 5] == 0 for i in range(6))
        )

    def tile_at(x, y):
        if x < 0 or y < 0 or x + 4 >= 64 or y + 4 >= 64:
            return None
        c = grid[y + 1][x + 1]
        if c in (4, 5):
            return None
        for j in range(4):
            for i in range(4):
                if i in (0, 3) or j in (0, 3):
                    if grid[y + 1 + j][x + 1 + i] != c:
                        return None
        middle = [
            grid[y + 2][x + 2],
            grid[y + 2][x + 3],
            grid[y + 3][x + 2],
            grid[y + 3][x + 3],
        ]
        if all(v == c for v in middle):
            return ("s", c)
        if all(v == 4 for v in middle):
            return ("r", c)
        return None

    wait = memory.pop("wait", None)
    if wait is not None:
        kind, x, y, expected = wait
        if kind == "select":
            if not border_is_black(x, y):
                return reset_or_run()
        elif kind == "place":
            if tile_at(x, y) != tuple(expected):
                return reset_or_run()

    if memory.get("ran"):
        return reset_or_run()

    def issue_queued():
        queue = memory.get("queue", [])
        if not queue:
            memory.pop("queue", None)
            memory["ran"] = True
            return ("ACTION5", None, None)
        kind, x, y, token = queue.pop(0)
        memory["wait"] = (kind, x, y, token)
        if kind == "select":
            return ("ACTION6", y + 1, x + 1)
        return ("ACTION6", y + 2, x + 2)

    if "queue" in memory:
        return issue_queued()

    target_positions = []
    for y in range(0, 48):
        for x in range(0, 59):
            c = grid[y][x]
            if c in (4, 5):
                continue
            if any(grid[y][x + i] != c for i in range(6)):
                continue
            if any(grid[y + 5][x + i] != c for i in range(6)):
                continue
            if any(
                grid[y + j][x] != c or grid[y + j][x + 5] != c
                for j in range(1, 5)
            ):
                continue
            if not all(
                grid[y + j][x + i] == 5
                for j in range(1, 5)
                for i in range(1, 5)
            ):
                continue
            target_positions.append((y, x, c))
    target_positions.sort()
    targets = [c for _, _, c in target_positions]

    frames = []
    for y in range(8, 44):
        x = 0
        while x < 64:
            c = grid[y][x]
            if c in (4, 5):
                x += 1
                continue
            end = x + 1
            while end < 64 and grid[y][end] == c:
                end += 1
            width = end - x
            if (
                width >= 10
                and (width - 4) % 6 == 0
                and y + 9 < 53
                and all(grid[y + 9][i] == c for i in range(x, end))
                and all(
                    grid[y + j][x] == c and grid[y + j][end - 1] == c
                    for j in range(1, 9)
                )
            ):
                n = (width - 4) // 6
                slots = [(x + 2 + 6 * i, y + 2) for i in range(n)]
                frames.append({"x": x, "y": y, "color": c, "slots": slots})
            x = end
    frames.sort(key=lambda f: (f["y"], f["x"]))

    if not targets or not frames:
        return reset_or_run()

    inventory = []
    for y in range(54, 61):
        for x in range(0, 59):
            token = tile_at(x, y)
            if token is not None:
                inventory.append((x, y, token))

    # Deselect a tile selected before this episode began.
    possible_items = [(x, y) for x, y, _ in inventory]
    for frame in frames:
        for x, y in frame["slots"]:
            if tile_at(x, y) is not None:
                possible_items.append((x, y))
    for x, y in possible_items:
        if border_is_black(x, y) and tile_at(x, y) is not None:
            return ("ACTION6", y + 1, x + 1)

    callee = {}
    for i, frame in enumerate(frames):
        callee.setdefault(frame["color"], i)

    slot_coords = []
    frame_slots = []
    fixed = {}
    for frame in frames:
        ids = []
        for x, y in frame["slots"]:
            idx = len(slot_coords)
            slot_coords.append((x, y))
            ids.append(idx)
            token = tile_at(x, y)
            if token is not None:
                fixed[idx] = token
        frame_slots.append(ids)

    token_set = set(token for _, _, token in inventory)
    token_set.update(fixed.values())
    token_set.update(("s", c) for c in targets)
    tokens = sorted(token_set)
    token_id = {token: i for i, token in enumerate(tokens)}

    counts = [0] * len(tokens)
    for _, _, token in inventory:
        counts[token_id[token]] += 1

    assignment = [-1] * len(slot_coords)
    for idx, token in fixed.items():
        assignment[idx] = token_id[token]

    nodes = [0]
    failed = set()
    exhausted = [False]

    def search(stack, pos):
        nodes[0] += 1
        if nodes[0] > 200000:
            exhausted[0] = True
            return False

        stack = list(stack)
        while stack and stack[-1][1] >= len(frame_slots[stack[-1][0]]):
            stack.pop()
            if stack:
                f, i, entry = stack[-1]
                stack[-1] = (f, i + 1, entry)
        if not stack:
            return False

        stack = tuple(stack)
        key = (stack, pos, tuple(assignment))
        if key in failed:
            return False

        f, i, entry = stack[-1]
        slot = frame_slots[f][i]
        assigned = assignment[slot]

        if assigned >= 0:
            choices = [assigned]
        else:
            choices = []
            simple = token_id.get(("s", targets[pos]))
            if simple is not None and counts[simple] > 0:
                choices.append(simple)
            for color in callee:
                ref = token_id.get(("r", color))
                if ref is not None and counts[ref] > 0:
                    choices.append(ref)

        for tid in choices:
            new_assignment = assigned < 0
            if new_assignment:
                assignment[slot] = tid
                counts[tid] -= 1

            kind, color = tokens[tid]
            success = False

            if kind == "s":
                if color == targets[pos]:
                    if pos + 1 == len(targets):
                        success = True
                    else:
                        next_stack = stack[:-1] + ((f, i + 1, entry),)
                        success = search(next_stack, pos + 1)
            else:
                dest = callee.get(color)
                if dest is not None:
                    if not any(
                        sf == dest and start == pos
                        for sf, _, start in stack
                    ):
                        success = search(stack + ((dest, 0, pos),), pos)

            if success:
                return True
            if new_assignment:
                counts[tid] += 1
                assignment[slot] = -1
            if exhausted[0]:
                return False

        if len(failed) < 100000:
            failed.add(key)
        return False

    if not search(((0, 0, 0),), 0):
        return reset_or_run()

    sources = defaultdict(list)
    for x, y, token in inventory:
        sources[token].append((x, y))

    queue = []
    for idx, tid in enumerate(assignment):
        if tid < 0 or idx in fixed:
            continue
        token = tokens[tid]
        if not sources[token]:
            return reset_or_run()
        sx, sy = sources[token].pop()
        dx, dy = slot_coords[idx]
        queue.append(("select", sx, sy, token))
        queue.append(("place", dx, dy, token))

    memory["queue"] = queue
    return issue_queued()

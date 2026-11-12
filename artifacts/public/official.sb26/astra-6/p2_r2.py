import numpy as np


def _tile(a, x, y):
    """Read a six-by-six tile footprint."""
    if x < 0 or y < 0 or x + 6 > 64 or y + 6 > 64:
        return None
    b = a[y + 1:y + 5, x + 1:x + 5]
    c = int(b[0, 0])
    if c == 4:
        return None
    if not (np.all(b[0, :] == c) and np.all(b[-1, :] == c)
            and np.all(b[:, 0] == c) and np.all(b[:, -1] == c)):
        return None
    if np.all(b[1:3, 1:3] == c):
        return (0, c)                 # Literal
    if np.all(b[1:3, 1:3] == 4):
        return (1, c)                 # Function call
    return None


def _selected(a, x, y):
    b = a[y:y + 6, x:x + 6]
    return (b.shape == (6, 6)
            and np.all(b[0, :] == 0)
            and np.all(b[-1, :] == 0)
            and np.all(b[:, 0] == 0)
            and np.all(b[:, -1] == 0))


def _read(a):
    targets = []
    # Targets have a six-pixel colored outline over a color-5 backing.
    for y in range(53):
        for x in range(59):
            c = int(a[y, x])
            if c in (4, 5):
                continue
            if a[y, x + 5] != c or a[y + 5, x] != c:
                continue
            b = a[y:y + 6, x:x + 6]
            if (np.all(b[0, :] == c) and np.all(b[-1, :] == c)
                    and np.all(b[:, 0] == c) and np.all(b[:, -1] == c)
                    and np.all(b[1:5, 1:5] == 5)):
                targets.append((y, x, c))

    frames = []
    # Function boxes are ten pixels high and 6*n+4 pixels wide.
    for y in range(8, 44):
        x = 0
        while x < 64:
            start = x
            c = int(a[y, x])
            while x < 64 and a[y, x] == c:
                x += 1
            end = x
            if c == 4 or end - start < 10:
                continue
            for left in range(start, end - 9):
                if not np.all(a[y:y + 10, left] == c):
                    continue
                for width in range(10, min(64 - left, end - left) + 1, 6):
                    right = left + width - 1
                    if (np.all(a[y:y + 10, right] == c)
                            and np.all(a[y + 9, left:right + 1] == c)):
                        frames.append((y, left, c, (width - 4) // 6))

    frames = sorted(set(frames))
    targets.sort()
    slots = []
    for y, x, c, n in frames:
        for i in range(n):
            sx, sy = x + 2 + 6 * i, y + 2
            slots.append((sx, sy, _tile(a, sx, sy)))

    pool = []
    for y in range(54, 59):
        for x in range(59):
            t = _tile(a, x, y)
            if t is not None:
                pool.append((x, y, t))

    selected = None
    for x, y, t in slots + pool:
        if t is not None and _selected(a, x, y):
            selected = (x, y, t)
            break

    return targets, frames, slots, pool, selected


def _synthesize(targets, frames, slots, pool, variant=0):
    target = [v[2] for v in targets]
    if not target or not frames:
        return None

    types = sorted(set(
        [t for _, _, t in slots if t is not None]
        + [t for _, _, t in pool]
    ))
    if not types:
        return None
    type_id = {t: i for i, t in enumerate(types)}
    counts = [0] * len(types)
    for _, _, t in pool:
        counts[type_id[t]] += 1

    assignment = [
        -1 if t is None else type_id[t] for _, _, t in slots
    ]

    starts = []
    next_pc = []
    index = []
    color_start = {}
    offset = 0
    for _, _, color, n in frames:
        starts.append(offset)
        color_start.setdefault(color, offset)
        for j in range(n):
            index.append(j)
            next_pc.append(offset + j + 1 if j + 1 < n else -1)
        offset += n

    destinations = [
        color_start.get(c, -1) if kind else -1
        for kind, c in types
    ]
    literals = {c: i for i, (kind, c) in enumerate(types) if kind == 0}
    calls = [
        i for i, (kind, _) in enumerate(types)
        if kind == 1 and destinations[i] >= 0
    ]
    if variant and calls:
        shift = variant % len(calls)
        calls = calls[shift:] + calls[:shift]
        if variant % 2:
            calls.reverse()

    # The search recurses only when assigning a previously unknown slot.
    # All already-determined program execution is simulated iteratively.
    budget = [1600000]
    exhausted = [False]

    def search(pc, stack, pos, seen, cost):
        while True:
            budget[0] -= 1
            if budget[0] < 0:
                exhausted[0] = True
                return None
            if pc < 0 or cost > 970:
                return None

            tid = assignment[pc]
            if tid < 0:
                options = []
                lit = literals.get(target[pos])
                if lit is not None and counts[lit]:
                    options.append(lit)
                options.extend(t for t in calls if counts[t])

                for choice in options:
                    assignment[pc] = choice
                    counts[choice] -= 1
                    answer = search(pc, stack, pos, seen, cost)
                    if answer is not None:
                        return answer
                    counts[choice] += 1
                    assignment[pc] = -1
                    if exhausted[0]:
                        break
                return None

            kind, color = types[tid]
            if kind == 1:
                dest = destinations[tid]
                if dest < 0 or pc in seen:
                    return None

                # Reproduce the simulator's first-slot recursion guard.
                if (index[pc] == 0 and pc in stack and stack
                        and index[stack[-1]] == 0):
                    return None

                seen = seen | frozenset((pc,))
                stack = stack + (pc,)
                pc = dest
                cost += 16
                continue

            if color != target[pos]:
                return None
            pos += 1
            cost += 13
            if pos == len(target):
                if cost + 16 <= 1000:
                    return assignment[:]
                return None

            seen = frozenset()
            pc = next_pc[pc]
            while pc < 0 and stack:
                caller = stack[-1]
                stack = stack[:-1]
                pc = next_pc[caller]
                cost += 16

            if pc < 0:
                return None
            next_type = assignment[pc]
            cost += (11 if next_type >= 0 and types[next_type][0] == 0
                     else 16)

    answer = search(starts[0], (), 0, frozenset(), 16)
    if answer is None:
        return None

    return [
        (slots[i][0], slots[i][1], types[t][0], types[t][1])
        for i, t in enumerate(answer) if t >= 0
    ]


def _recover(memory, available, fingerprint):
    memory.pop("_sb_plan", None)
    memory.pop("_sb_tested", None)
    memory["_sb_variant"] = memory.get("_sb_variant", 0) + 1

    # Undo is always safe for the original fixed tiles.
    previous = memory.get("_sb_undo_image")
    memory["_sb_undo_image"] = fingerprint
    if "ACTION7" in available and previous != fingerprint:
        return ("ACTION7", None, None)
    if "RESET" in available:
        memory.pop("_sb_undo_image", None)
        return ("RESET", None, None)
    if "ACTION7" in available:
        return ("ACTION7", None, None)
    return ("ACTION5", None, None)


def act(obs, available, memory):
    a = np.asarray(obs)
    targets, frames, slots, pool, selected = _read(a)
    fingerprint = (
        tuple(targets), tuple(frames),
        tuple((x, y, t) for x, y, t in slots),
        tuple((x, y, t) for x, y, t in pool),
    )
    signature = (tuple(targets), tuple(frames))

    if memory.get("_sb_signature") != signature:
        for key in (
            "_sb_plan", "_sb_tested", "_sb_undo_image",
            "_sb_variant"
        ):
            memory.pop(key, None)
        memory["_sb_signature"] = signature

    if not targets or not frames:
        return _recover(memory, available, fingerprint)

    if memory.pop("_sb_tested", False):
        # Normally the episode has ended after a successful execution.
        return _recover(memory, available, fingerprint)

    plan = memory.get("_sb_plan")
    if plan is not None:
        # Unexpected placements invalidate the cached assignment.
        for x, y, kind, color in plan:
            present = _tile(a, x, y)
            if present is not None and present != (kind, color):
                plan = None
                memory.pop("_sb_plan", None)
                break

    if plan is None:
        plan = _synthesize(
            targets, frames, slots, pool,
            memory.get("_sb_variant", 0)
        )
        if plan is None:
            return _recover(memory, available, fingerprint)
        memory["_sb_plan"] = plan
        memory.pop("_sb_undo_image", None)

    missing = []
    for x, y, kind, color in plan:
        if _tile(a, x, y) != (kind, color):
            missing.append((x, y, (kind, color)))

    if not missing:
        memory["_sb_tested"] = True
        return ("ACTION5", None, None)

    # Exploit an already-selected pool tile when it belongs in the plan.
    if selected is not None and selected[1] > 53:
        sx, sy, st = selected
        for x, y, wanted in missing:
            if wanted == st:
                return ("ACTION6", y + 2, x + 2)

    # A selected board tile must be deselected before touching the pool,
    # otherwise the game would swap it with the newly clicked tile.
    if selected is not None and selected[1] <= 53:
        sx, sy, _ = selected
        return ("ACTION6", sy + 2, sx + 2)

    for x, y, wanted in missing:
        sources = [(sx, sy) for sx, sy, t in pool if t == wanted]
        if sources:
            sx, sy = min(
                sources, key=lambda p: abs(p[0] - x) + abs(p[1] - y)
            )
            return ("ACTION6", sy + 2, sx + 2)

    return _recover(memory, available, fingerprint)

def _sb_read(obs):
    a = obs.tolist()

    def tile(x, y):
        if not (0 <= x <= 58 and 0 <= y <= 58):
            return -1
        c = a[y + 1][x + 1]
        if c == 4:
            return -1
        for k in range(4):
            if (a[y + 1][x + 1 + k] != c or
                a[y + 4][x + 1 + k] != c or
                a[y + 1 + k][x + 1] != c or
                a[y + 1 + k][x + 4] != c):
                return -1
        center = [a[y + j][x + i] for j in (2, 3) for i in (2, 3)]
        if all(v == c for v in center):
            return c
        if all(v == 4 for v in center):
            return c + 16
        return -1

    goals = []
    for y in range(49):
        for x in range(59):
            c = a[y][x]
            if c in (4, 5):
                continue
            if not all(a[y][x + k] == c and
                       a[y + 5][x + k] == c and
                       a[y + k][x] == c and
                       a[y + k][x + 5] == c for k in range(6)):
                continue
            if all(a[y + j][x + i] == 5
                   for j in range(1, 5) for i in range(1, 5)):
                goals.append((y, x, c))
    goals.sort()
    target = [g[2] for g in goals]

    frames = []
    for y in range(44):
        x = 0
        while x < 64:
            c = a[y][x]
            end = x + 1
            while end < 64 and a[y][end] == c:
                end += 1
            width = end - x
            if (c not in (4, 5) and width >= 10 and
                    (width - 4) % 6 == 0 and (width - 4) // 6 <= 9):
                if (all(a[y + 9][xx] == c for xx in range(x, end)) and
                    all(a[yy][x] == c and a[yy][end - 1] == c
                        for yy in range(y, y + 10))):
                    frames.append((x, y, (width - 4) // 6, c))
            x = end
    frames.sort(key=lambda f: (f[1], f[0]))

    positions = []
    current = []
    for x, y, n, c in frames:
        for k in range(n):
            p = (x + 2 + 6 * k, y + 2)
            positions.append(p)
            current.append(tile(*p))

    palette = []
    for y in range(54, 59):
        for x in range(59):
            v = tile(x, y)
            if v >= 0:
                palette.append(((x, y), v))
            else:
                empty = all(
                    a[y + j][x + i] == (2 if i in (2, 3) and j in (2, 3) else 4)
                    for j in range(1, 5) for i in range(1, 5)
                )
                if empty:
                    palette.append(((x, y), -1))
    palette.sort(key=lambda t: (t[0][1], t[0][0]))

    selected = None
    all_cells = list(zip(positions, current)) + palette
    for (x, y), v in all_cells:
        if v >= 0 and all(
            a[y][x + k] == 0 and a[y + 5][x + k] == 0 and
            a[y + k][x] == 0 and a[y + k][x + 5] == 0
            for k in range(6)
        ):
            selected = (x, y)
            break

    return target, frames, positions, current, palette, selected


def _sb_solve(target, frames, current, palette, fixed, keep_all,
              budget, variant, banned):
    size = len(current)
    values = [-1] * size
    counts = [0] * 32
    for v in current:
        if v >= 0:
            counts[v] += 1
    for _, v in palette:
        if v >= 0:
            counts[v] += 1

    for i, v in enumerate(current):
        if v >= 0 and (keep_all or i in fixed):
            values[i] = v
            counts[v] -= 1

    starts = {}
    next_pc = [-1] * size
    offset = [0] * size
    at = 0
    for x, y, n, c in frames:
        if c not in starts:
            starts[c] = at
        for j in range(n):
            offset[at + j] = j
            if j + 1 < n:
                next_pc[at + j] = at + j + 1
        at += n

    calls = [16 + c for c in starts if counts[16 + c] > 0]
    calls.sort(key=lambda v: (starts[v - 16] == 0, starts[v - 16]))
    if calls and variant > 1:
        r = (variant - 1) % len(calls)
        calls = calls[r:] + calls[:r]

    failed = {}
    remaining = budget
    answer = None

    def walk(stack, p, seen, ticks):
        nonlocal remaining, answer

        while stack:
            remaining -= 1
            if remaining <= 0 or ticks > 940:
                return False

            pc = stack[-1]
            bit = 1 << pc
            if seen & bit:
                return False
            v = values[pc]

            if v < 0:
                key = (tuple(values), stack, p, seen)
                old = failed.get(key)
                if old is not None and ticks >= old:
                    return False

                literal = target[p]
                options = []
                if counts[literal] > 0:
                    options.append(literal)
                options.extend(vv for vv in calls if counts[vv] > 0)

                if variant % 3 == 1 and len(options) > 1:
                    options.sort(key=lambda vv: vv < 16)

                preferred = current[pc]
                if preferred in options:
                    options.remove(preferred)
                    options.insert(0, preferred)

                for choice in options:
                    values[pc] = choice
                    counts[choice] -= 1
                    good = walk(stack, p, seen, ticks)
                    counts[choice] += 1
                    values[pc] = -1
                    if good:
                        return True
                    if remaining <= 0:
                        return False

                if len(failed) < 45000:
                    if old is None or ticks < old:
                        failed[key] = ticks
                return False

            if v < 16:
                if v != target[p]:
                    return False
                ticks += 24
                p += 1

                if p == len(target):
                    if ticks > 940:
                        return False
                    for bad in banned:
                        if len(bad) == size and all(
                            values[i] < 0 or values[i] == bad[i]
                            for i in range(size)
                        ):
                            return False
                    answer = values[:]
                    return True

                seen = 0
                ss = list(stack)
                ss[-1] = next_pc[pc]
                while ss and ss[-1] < 0:
                    ss.pop()
                    if ss:
                        ticks += 16
                        ss[-1] = next_pc[ss[-1]]
                stack = tuple(ss)

            else:
                start = starts.get(v - 16)
                if start is None:
                    return False

                # Reproduce the game's additional recursion guard.
                if (offset[pc] == 0 and pc in stack[:-1] and
                        len(stack) > 1 and offset[stack[-2]] == 0):
                    return False

                seen |= bit
                stack = stack + (start,)
                ticks += 16

        return False

    if size and target:
        walk((0,), 0, 0, 20)
    return answer


def act(obs, available, memory):
    target, frames, positions, current, palette, selected = _sb_read(obs)

    def fallback():
        if "ACTION7" in available:
            return ("ACTION7", None, None)
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION5", None, None)

    if not frames or not target:
        return fallback()

    signature = repr((target, frames))
    m = memory.get("_sb26")
    if not isinstance(m, dict) or m.get("signature") != signature:
        m = {
            "signature": signature,
            "fixed": [],
            "free": [],
            "round": 0,
            "banned": [],
        }
        memory["_sb26"] = m

    index = {p: i for i, p in enumerate(positions)}
    cells = dict(zip(positions, current))
    cells.update(dict(palette))

    fixed = set(m.get("fixed", []))
    free = set(m.get("free", []))

    # An empty program position necessarily accepts movable tiles.
    for i, v in enumerate(current):
        if v < 0:
            free.add(i)
            fixed.discard(i)

    previous = m.get("previous")
    if previous is not None and len(previous) == len(current):
        for i, (old, new) in enumerate(zip(previous, current)):
            if old != new:
                free.add(i)
                fixed.discard(i)

    if selected in index:
        free.add(index[selected])
        fixed.discard(index[selected])

    last = m.pop("last", None)
    if last:
        kind = last.get("kind")
        if kind == "select":
            p = tuple(last["pos"])
            if selected != p and p in index and cells.get(p, -1) >= 0:
                i = index[p]
                fixed.add(i)
                free.discard(i)
                m.pop("plan", None)

        elif kind == "swap":
            src = tuple(last["src"])
            dst = tuple(last["dst"])
            success = (
                selected is None and
                cells.get(src, -1) == last["b"] and
                cells.get(dst, -1) == last["a"]
            )
            if success:
                for p in (src, dst):
                    if p in index:
                        free.add(index[p])
                        fixed.discard(index[p])
            else:
                if selected == src and dst in index and cells.get(dst, -1) >= 0:
                    fixed.add(index[dst])
                    free.discard(index[dst])
                m.pop("plan", None)

        elif kind == "run":
            bad = last.get("board")
            if bad is not None:
                m.setdefault("banned", []).append(bad)
                m["banned"] = m["banned"][-4:]
            m.pop("plan", None)
            m["round"] = m.get("round", 0) + 1

    # Every original palette position has one movable tile. Empty palette
    # positions remain visible, so this gives the number of fixed tiles.
    total_tiles = sum(v >= 0 for v in current)
    total_tiles += sum(v >= 0 for _, v in palette)
    fixed_number = total_tiles - len(palette)
    unknown = [
        i for i, v in enumerate(current)
        if v >= 0 and i not in fixed and i not in free
    ]
    if 0 <= fixed_number <= len(current):
        left = fixed_number - len(fixed)
        if left == 0:
            free.update(i for i in range(len(current)) if i not in fixed)
        elif left == len(unknown):
            fixed.update(unknown)

    def save():
        m["fixed"] = sorted(fixed)
        m["free"] = sorted(free)
        m["previous"] = current[:]

    def click(p, kind, **extra):
        save()
        record = {"kind": kind, "pos": list(p)}
        record.update(extra)
        m["last"] = record
        return ("ACTION6", int(p[1] + 2), int(p[0] + 2))

    plan = m.get("plan")
    if plan is not None:
        if len(plan) != len(current) or any(
            plan[i] >= 0 and plan[i] != current[i] for i in fixed
        ):
            plan = None

    if plan is None:
        variant = m.get("round", 0)
        movable_present = any(
            v >= 0 and i not in fixed for i, v in enumerate(current)
        )

        plan = _sb_solve(
            target, frames, current, palette, fixed, True,
            500000 if movable_present else 1900000,
            variant, m.get("banned", [])
        )
        if plan is None and movable_present:
            plan = _sb_solve(
                target, frames, current, palette, fixed, False,
                1500000, variant, m.get("banned", [])
            )

        if plan is None:
            save()
            m["round"] = variant + 1
            m.pop("plan", None)
            m["last"] = {"kind": "undo"}
            return fallback()

        m["plan"] = plan

    mismatches = [
        i for i, v in enumerate(plan)
        if v >= 0 and current[i] != v
    ]

    if not mismatches:
        save()
        m["last"] = {"kind": "run", "board": current[:]}
        return ("ACTION5", None, None)

    def protected(p):
        i = index.get(p)
        if i is None:
            return False
        return i in fixed or (plan[i] >= 0 and plan[i] == current[i])

    def uncertain(p):
        i = index.get(p)
        return (
            i is not None and current[i] >= 0 and
            i not in fixed and i not in free
        )

    def fixes_after_swap(src, dst):
        result = 0
        va, vb = cells[src], cells[dst]
        for p, new in ((src, vb), (dst, va)):
            i = index.get(p)
            if i is not None and plan[i] >= 0:
                if current[i] != plan[i] and new == plan[i]:
                    result += 1
        return result

    if selected is not None:
        options = []
        sv = cells.get(selected, -1)

        # Normal direction: move the selected tile to a required position.
        if sv >= 0 and not protected(selected):
            for i in mismatches:
                dst = positions[i]
                if dst != selected and plan[i] == sv and i not in fixed:
                    score = (
                        -fixes_after_swap(selected, dst),
                        int(uncertain(dst)),
                        i,
                    )
                    options.append((score, dst))

        # Reverse direction: select a misplaced tile, then exchange it with
        # the tile it needs. This also tests uncertain fixed tiles cheaply.
        si = index.get(selected)
        if si is not None and si in mismatches and si not in fixed:
            wanted = plan[si]
            for p, v in cells.items():
                if p != selected and v == wanted and not protected(p):
                    score = (
                        -fixes_after_swap(selected, p),
                        int(uncertain(p)),
                        index.get(p, len(current)),
                    )
                    options.append((score, p))

        if options:
            options.sort(key=lambda t: t[0])
            dst = options[0][1]
            return click(
                dst, "swap",
                src=list(selected), dst=list(dst),
                a=cells[selected], b=cells[dst]
            )

        return click(selected, "deselect")

    operations = []
    for i in mismatches:
        if i in fixed:
            continue
        dst = positions[i]
        wanted = plan[i]
        for src, v in cells.items():
            if src == dst or v != wanted or protected(src):
                continue

            ambiguous = uncertain(src) or uncertain(dst)
            score = (
                0 if ambiguous else 1,
                -fixes_after_swap(src, dst),
                int(src in index),
                i,
                src[1],
                src[0],
            )
            # If the destination might be fixed, select it first. A
            # successful selection can be followed by the reverse swap.
            first = dst if uncertain(dst) else src
            operations.append((score, first))

    if operations:
        operations.sort(key=lambda t: t[0])
        return click(operations[0][1], "select")

    # Normally unreachable; recover from an unexpected visual discrepancy.
    save()
    m.pop("plan", None)
    m["round"] = m.get("round", 0) + 1
    m["last"] = {"kind": "undo"}
    return fallback()

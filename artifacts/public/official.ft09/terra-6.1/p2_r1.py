import itertools
import numpy as np


def _read_board(obs):
    # The canonical 32-by-32 game view is displayed at scale 2.
    a = np.asarray(obs)[::2, ::2]
    h, w = a.shape

    # Detect the tutorial's corner frame, if present. Its other diagrams
    # are illustrations rather than clickable tiles.
    frame = None
    corner = [
        (0, 0), (1, 0), (2, 0), (0, 1), (0, 2),
        (14, 0), (13, 0), (12, 0), (14, 1), (14, 2),
        (0, 14), (1, 14), (2, 14), (0, 13), (0, 12),
        (14, 14), (13, 14), (12, 14), (14, 13), (14, 12),
    ]
    for y in range(h - 14):
        for x in range(w - 14):
            if all(int(a[y + dy, x + dx]) == 2 for dx, dy in corner):
                frame = (x, y, x + 14, y + 14)
                break
        if frame is not None:
            break

    # The colour key, when present, is a two-pixel-wide vertical strip.
    legend = []
    for x in range(w - 1):
        first = int(a[0, x])
        second = int(a[2, x])
        if first in (0, 2, 3, 4, 5, 6) or second in (0, 2, 3, 4, 5, 6):
            continue
        if first == second:
            continue
        if not all(int(a[dy, x + dx]) == first
                   for dy in (0, 1) for dx in (0, 1)):
            continue
        if not all(int(a[dy, x + dx]) == second
                   for dy in (2, 3) for dx in (0, 1)):
            continue
        if x > 0 and int(a[0, x - 1]) == first:
            continue
        if x + 2 < w and int(a[0, x + 2]) == first:
            continue
        colors = [first, second]
        yy = 4
        while yy + 1 < h:
            c = int(a[yy, x])
            if c in (0, 2, 3, 4, 5, 6) or c in colors:
                break
            if not all(int(a[yy + dy, x + dx]) == c
                       for dy in (0, 1) for dx in (0, 1)):
                break
            colors.append(c)
            yy += 2
        legend = colors
        break

    groups = {}
    for y in range(h - 2):
        for x in range(w - 2):
            if frame is not None:
                x0, y0, x1, y1 = frame
                if not (x0 < x and x + 2 < x1 and
                        y0 < y and y + 2 < y1):
                    continue

            p = a[y:y + 3, x:x + 3]
            c = int(p[1, 1])
            if c in (0, 2, 3, 4, 5, 6) and c not in legend:
                continue

            values = [int(v) for v in p.flat]
            kind = None
            if all(v == c for v in values):
                kind = "normal"
            elif any(v == 6 for v in values) and all(
                    v == c or v == 6 for v in values):
                kind = "special"
            else:
                outer = [int(p[dy, dx])
                         for dy in range(3) for dx in range(3)
                         if (dx, dy) != (1, 1)]
                if all(v in (0, 2, 3) for v in outer):
                    kind = "clue"

            if kind is not None:
                key = (x % 4, y % 4)
                groups.setdefault(key, []).append(
                    (x, y, kind, c, values)
                )

    if not groups:
        return [], [], legend

    def group_score(items):
        return sum(2 if item[2] == "clue" else 1 for item in items)

    items = max(groups.values(), key=group_score)
    tiles = []
    clues = []
    for x, y, kind, c, values in items:
        if kind == "clue":
            clues.append((x, y, c, values))
        else:
            offsets = [(0, 0)]
            if kind == "special":
                for dy in range(3):
                    for dx in range(3):
                        if values[dy * 3 + dx] == 6:
                            off = (dx - 1, dy - 1)
                            if off not in offsets:
                                offsets.append(off)
            tiles.append((x, y, kind, c, offsets))

    tiles.sort(key=lambda t: (t[1], t[0]))
    return tiles, clues, legend


def _effects(tiles, memory):
    positions = {(t[0], t[1]): i for i, t in enumerate(tiles)}
    learned = memory.get("normal_mask", {})
    normal_offsets = [(0, 0)]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dx, dy) == (0, 0):
                continue
            if learned.get("%d,%d" % (dx, dy), 0):
                normal_offsets.append((dx, dy))
    if learned.get("0,0", 1) == 0:
        normal_offsets = [p for p in normal_offsets if p != (0, 0)]

    result = []
    for x, y, kind, _, offsets in tiles:
        use = offsets if kind == "special" else normal_offsets
        affected = []
        for dx, dy in use:
            j = positions.get((x + 4 * dx, y + 4 * dy))
            if j is not None:
                affected.append(j)
        result.append(affected)
    return result


def _constraints(tiles, clues, palette):
    positions = {(t[0], t[1]): i for i, t in enumerate(tiles)}
    allowed = [set(palette) for _ in tiles]
    for x, y, color, pixels in clues:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = positions.get((x + 4 * dx, y + 4 * dy))
                if j is None:
                    continue
                equal = pixels[(dy + 1) * 3 + dx + 1] == 0
                if equal:
                    allowed[j].intersection_update((color,))
                else:
                    allowed[j].discard(color)
    return allowed


def _binary_plan(tiles, palette, allowed, effects):
    n = len(tiles)
    rows = []
    for i, tile in enumerate(tiles):
        if not allowed[i]:
            return None
        if len(allowed[i]) == 2:
            continue
        target = next(iter(allowed[i]))
        if tile[3] not in palette or target not in palette:
            return None
        rhs = int(tile[3] != target)
        coeff = 0
        for j, affected in enumerate(effects):
            if i in affected:
                coeff |= 1 << j
        rows.append(coeff | (rhs << n))

    pivots = []
    r = 0
    for col in range(n):
        pivot = next((k for k in range(r, len(rows))
                      if (rows[k] >> col) & 1), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for k in range(len(rows)):
            if k != r and ((rows[k] >> col) & 1):
                rows[k] ^= rows[r]
        pivots.append(col)
        r += 1

    mask = (1 << n) - 1
    for row in rows:
        if (row & mask) == 0 and ((row >> n) & 1):
            return None

    solution = 0
    for k, col in enumerate(pivots):
        if (rows[k] >> n) & 1:
            solution |= 1 << col

    pivot_set = set(pivots)
    basis = []
    for free in range(n):
        if free in pivot_set:
            continue
        b = 1 << free
        for k, col in enumerate(pivots):
            if (rows[k] >> free) & 1:
                b |= 1 << col
        # A free click affecting no constraint cannot improve a solution.
        if b != (1 << free):
            basis.append(b)

    best = solution
    best_count = best.bit_count()
    if len(basis) <= 16:
        current = solution
        previous_gray = 0
        for q in range(1, 1 << len(basis)):
            gray = q ^ (q >> 1)
            changed = gray ^ previous_gray
            current ^= basis[changed.bit_length() - 1]
            previous_gray = gray
            count = current.bit_count()
            if count < best_count:
                best, best_count = current, count
    else:
        improved = True
        while improved:
            improved = False
            for b in basis:
                candidate = best ^ b
                if candidate.bit_count() < best_count:
                    best = candidate
                    best_count = best.bit_count()
                    improved = True

    return [i for i in range(n) if (best >> i) & 1]


def _multi_plan(tiles, palette, allowed, effects):
    k = len(palette)
    n = len(tiles)
    if any(not s for s in allowed):
        return None

    specials = [i for i, t in enumerate(tiles) if t[2] == "special"]
    normals = [i for i, t in enumerate(tiles) if t[2] == "normal"]

    # This solver uses independent normal clicks. If feedback reveals a
    # different pattern, the caller falls back to incremental play.
    if any(effects[i] != [i] for i in normals):
        return None

    specials.sort(key=lambda j: -sum(
        1 for i in effects[j] if len(allowed[i]) < k
    ))
    initial = [palette.index(t[3]) for t in tiles]
    changes = [0] * n
    last = {}
    for depth, j in enumerate(specials):
        for i in effects[j]:
            if tiles[i][2] == "special":
                last[i] = depth

    best_cost = [10 ** 9]
    best_plan = [None]
    chosen = [0] * len(specials)
    nodes = [0]

    def search(depth, cost):
        nodes[0] += 1
        if nodes[0] > 150000 or cost >= best_cost[0]:
            return

        if depth == len(specials):
            for i in specials:
                c = palette[(initial[i] + changes[i]) % k]
                if c not in allowed[i]:
                    return

            plan = []
            for d, j in enumerate(specials):
                plan.extend([j] * chosen[d])
            total = cost
            for i in normals:
                start = (initial[i] + changes[i]) % k
                steps = next((d for d in range(k)
                              if palette[(start + d) % k] in allowed[i]), None)
                if steps is None:
                    return
                total += steps
                if total >= best_cost[0]:
                    return
                plan.extend([i] * steps)
            best_cost[0] = total
            best_plan[0] = plan
            return

        j = specials[depth]
        for amount in range(k):
            chosen[depth] = amount
            for i in effects[j]:
                changes[i] += amount

            valid = True
            for i in effects[j]:
                if (tiles[i][2] == "special" and last.get(i) == depth and
                        palette[(initial[i] + changes[i]) % k] not in allowed[i]):
                    valid = False
                    break
            if valid:
                search(depth + 1, cost + amount)

            for i in effects[j]:
                changes[i] -= amount

    search(0, 0)
    return best_plan[0]


def act(obs, available, memory):
    tiles, clues, legend = _read_board(obs)
    if not tiles:
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION6", 32, 32)

    positions = [[t[0], t[1]] for t in tiles]
    colors_now = [t[3] for t in tiles]
    pending = memory.pop("pending", None)
    reusable_plan = None

    if pending and pending.get("positions") == positions:
        old = pending["colors"]
        clicked = pending["clicked"]
        if len(old) == len(colors_now):
            transitions = memory.setdefault("transitions", {})
            for before, after in zip(old, colors_now):
                if before != after:
                    transitions[str(before)] = after

            if pending.get("kind") == "normal":
                mask = memory.setdefault("normal_mask", {})
                cx, cy = clicked
                for i, (x, y) in enumerate(positions):
                    dx, dy = x - cx, y - cy
                    if dx % 4 == 0 and dy % 4 == 0:
                        dx //= 4
                        dy //= 4
                        if -1 <= dx <= 1 and -1 <= dy <= 1:
                            mask["%d,%d" % (dx, dy)] = int(
                                old[i] != colors_now[i]
                            )
            if pending.get("expected") == colors_now:
                reusable_plan = pending.get("remaining")

    known = list(memory.get("colors", []))
    for c in colors_now:
        if c not in known:
            known.append(c)
    if legend:
        palette = list(legend)
        for c in known:
            if c not in palette:
                palette.append(c)
    else:
        palette = known[:]
        for _, _, c, _ in clues:
            if c not in palette:
                palette.append(c)
    memory["colors"] = palette[:]

    # Use observed transitions to improve the cycle order.
    if len(palette) > 2:
        transitions = memory.get("transitions", {})
        ordered = [palette[0]]
        while len(ordered) < len(palette):
            nxt = transitions.get(str(ordered[-1]))
            if nxt in palette and nxt not in ordered:
                ordered.append(nxt)
            else:
                ordered.append(next(c for c in palette if c not in ordered))
        palette = ordered

    effects = _effects(tiles, memory)
    allowed = _constraints(tiles, clues, palette)

    if reusable_plan:
        pos_to_index = {(t[0], t[1]): i for i, t in enumerate(tiles)}
        plan = []
        for p in reusable_plan:
            i = pos_to_index.get(tuple(p))
            if i is None:
                plan = None
                break
            plan.append(i)
    else:
        plan = None

    if plan is None:
        if len(palette) == 2:
            plan = _binary_plan(tiles, palette, allowed, effects)
        elif len(palette) > 2:
            plan = _multi_plan(tiles, palette, allowed, effects)

    if not plan:
        # Incremental fallback, also used to discover an unseen colour.
        unsatisfied = [i for i, t in enumerate(tiles)
                       if t[3] not in allowed[i]]
        if unsatisfied:
            plan = [unsatisfied[0]]
        else:
            # If the board is already solved, a harmless click may trigger
            # the game's completion check.
            constrained = {i for i, s in enumerate(allowed)
                           if len(s) < len(palette)}
            harmless = next((j for j, e in enumerate(effects)
                             if not constrained.intersection(e)), None)
            plan = [harmless if harmless is not None else 0]

    chosen = plan[0]
    x, y, kind, _, _ = tiles[chosen]
    expected = None
    if len(palette) >= 2 and all(c in palette for c in colors_now):
        expected = colors_now[:]
        for i in effects[chosen]:
            old_index = palette.index(expected[i])
            expected[i] = palette[(old_index + 1) % len(palette)]

    memory["pending"] = {
        "positions": positions,
        "colors": colors_now,
        "clicked": [x, y],
        "kind": kind,
        "expected": expected,
        "remaining": [[tiles[i][0], tiles[i][1]] for i in plan[1:]],
    }
    return ("ACTION6", 2 * y + 2, 2 * x + 2)

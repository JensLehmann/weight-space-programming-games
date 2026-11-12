import itertools
import random
import numpy as np


def _read_board(obs):
    """Find scaled 3x3 tiles, excluding the tutorial's grey-backed examples."""
    best = None
    best_score = -1

    for scale in (1, 2, 3, 4):
        for oy in range(scale):
            for ox in range(scale):
                h = (64 - oy) // scale
                w = (64 - ox) // scale
                if h < 3 or w < 3:
                    continue
                g = obs[oy:oy + h * scale:scale,
                        ox:ox + w * scale:scale].tolist()
                tiles = []

                for y in range(h - 2):
                    for x in range(w - 2):
                        color = g[y + 1][x + 1]
                        if color in (0, 2, 3, 4, 5, 6):
                            continue

                        border = [
                            g[y][x], g[y][x + 1], g[y][x + 2],
                            g[y + 1][x], g[y + 1][x + 2],
                            g[y + 2][x], g[y + 2][x + 1],
                            g[y + 2][x + 2]
                        ]
                        if all(v == color for v in border):
                            kind = 0       # Ordinary button
                        elif (6 in border and
                              all(v in (color, 6) for v in border)):
                            kind = 1       # Patterned button
                        elif all(v in (0, 2, 3) for v in border):
                            kind = 2       # Constraint clue
                        else:
                            continue

                        surrounding = []
                        for xx, yy in ((x - 1, y + 1), (x + 3, y + 1),
                                       (x + 1, y - 1), (x + 1, y + 3)):
                            surrounding.append(
                                g[yy][xx] if 0 <= xx < w and 0 <= yy < h
                                else 4
                            )
                        if (any(v not in (4, 5) for v in surrounding) or
                                surrounding.count(4) < 2):
                            continue

                        patch = [g[y + j][x:x + 3] for j in range(3)]
                        px, py = ox + x * scale, oy + y * scale
                        if scale > 1:
                            expected = np.repeat(
                                np.repeat(np.asarray(patch), scale, axis=0),
                                scale, axis=1
                            )
                            actual = obs[py:py + 3 * scale,
                                         px:px + 3 * scale]
                            if not np.array_equal(expected, actual):
                                continue
                        tiles.append((px, py, kind, color, patch))

                nn = sum(t[2] != 2 for t in tiles)
                nc = sum(t[2] == 2 for t in tiles)
                if not nn:
                    continue
                score = len(tiles) + min(nn, nc)
                if score > best_score:
                    best_score = score
                    best = (scale, ox, oy, g, tiles)
    return best


def _read_legend(g):
    """Palette legends are isolated, two-cell-wide vertical colour strips."""
    h, w = len(g), len(g[0])
    best = []
    excluded = (0, 2, 3, 4, 5, 6)

    for x in range(w - 1):
        for y in range(h - 3):
            colors = []
            yy = y
            while yy + 1 < h:
                c = g[yy][x]
                if c in excluded or c in colors:
                    break
                if not (g[yy][x + 1] == c and
                        g[yy + 1][x] == c and
                        g[yy + 1][x + 1] == c):
                    break
                good = True
                for ry in (yy, yy + 1):
                    if x > 0 and g[ry][x - 1] != 4:
                        good = False
                    if x + 2 < w and g[ry][x + 2] != 4:
                        good = False
                if not good:
                    break
                colors.append(c)
                yy += 2

            top_ok = y == 0 or (
                g[y - 1][x] == 4 and g[y - 1][x + 1] == 4
            )
            bottom_ok = yy == h or (
                g[yy][x] == 4 and g[yy][x + 1] == 4
            )
            if (top_ok and bottom_ok and
                    len(colors) >= 2 and len(colors) > len(best)):
                best = colors
    return best


def _binary_plan(effects, required, satisfied, n):
    """Solve GF(2) equations, then reduce the weight of the solution."""
    pivots = {}
    for target in required:
        row = 0
        for button, affected in enumerate(effects):
            if target in affected:
                row |= 1 << button
        rhs = int(not satisfied[target])

        while row:
            p = (row & -row).bit_length() - 1
            if p not in pivots:
                pivots[p] = (row, rhs)
                break
            old, old_rhs = pivots[p]
            row ^= old
            rhs ^= old_rhs
        if not row and rhs:
            return None

    ordered = sorted(pivots, reverse=True)
    solution = 0
    for p in ordered:
        row, rhs = pivots[p]
        if ((row & solution).bit_count() & 1) != rhs:
            solution |= 1 << p

    basis = []
    for j in range(n):
        if j in pivots:
            continue
        v = 1 << j
        for p in ordered:
            row, _ = pivots[p]
            if (row & v).bit_count() & 1:
                v |= 1 << p
        basis.append(v)

    best = solution
    best_weight = best.bit_count()

    if len(basis) <= 19:
        value = solution
        previous_gray = 0
        for k in range(1, 1 << len(basis)):
            gray = k ^ (k >> 1)
            changed = gray ^ previous_gray
            value ^= basis[changed.bit_length() - 1]
            previous_gray = gray
            weight = value.bit_count()
            if weight < best_weight:
                best, best_weight = value, weight
                if not weight:
                    break
    else:
        rng = random.Random(9109)
        for trial in range(16):
            value = solution
            if trial:
                for b in basis:
                    if rng.randrange(2):
                        value ^= b
            for _ in range(8):
                changed = False
                order = list(basis)
                rng.shuffle(order)
                for b in order:
                    alternative = value ^ b
                    if alternative.bit_count() < value.bit_count():
                        value = alternative
                        changed = True
                if not changed:
                    break
            if value.bit_count() < best_weight:
                best, best_weight = value, value.bit_count()

    if best == 0:
        # Victory is checked only after a button click.
        if basis:
            best = min(basis, key=int.bit_count)
        elif n:
            best = 1

    return [j for j in range(n) if (best >> j) & 1]


def _cycle_plan(nodes, effects, palette, allowed, normal_mask):
    """Exact small search over special buttons; ordinary buttons are local."""
    m = len(palette)
    index = {c: i for i, c in enumerate(palette)}
    if any(t[3] not in index for t in nodes):
        return None
    initial = [index[t[3]] for t in nodes]
    specials = [i for i, t in enumerate(nodes) if t[2] == 1]
    ordinary = [i for i, t in enumerate(nodes) if t[2] == 0]
    identity = normal_mask == [0, 0, 0, 0, 1, 0, 0, 0, 0]

    if identity and m ** len(specials) <= 150000:
        best = None
        best_cost = float("inf")
        for choices in itertools.product(range(m), repeat=len(specials)):
            cost = sum(choices)
            if cost >= best_cost:
                continue
            state = initial[:]
            counts = [0] * len(nodes)
            for button, amount in zip(specials, choices):
                counts[button] = amount
                if amount:
                    for j in effects[button]:
                        state[j] = (state[j] + amount) % m

            if any(state[j] not in allowed[j] for j in specials):
                continue

            valid = True
            for j in ordinary:
                increment = next(
                    (q for q in range(m)
                     if (state[j] + q) % m in allowed[j]), None
                )
                if increment is None:
                    valid = False
                    break
                counts[j] = increment
                cost += increment
                if cost >= best_cost:
                    valid = False
                    break
            if valid:
                best, best_cost = counts, cost
        if best is not None:
            return [i for i, q in enumerate(best) for _ in range(q)]

    # Backup for a multi-colour palette with a coupled ordinary-button mask.
    rng = random.Random(1909)
    best_counts = None
    best_cost = float("inf")
    n = len(nodes)

    for restart in range(10):
        counts = ([0] * n if restart == 0 else
                  [rng.randrange(m) for _ in range(n)])
        state = initial[:]
        for i, q in enumerate(counts):
            for j in effects[i]:
                state[j] = (state[j] + q) % m

        for step in range(100):
            bad = sum(state[j] not in allowed[j] for j in range(n))
            cost = sum(counts)
            if bad == 0 and cost < best_cost:
                best_counts, best_cost = counts[:], cost

            best_move = None
            best_delta = 0
            for i in range(n):
                for new_q in range(m):
                    if new_q == counts[i]:
                        continue
                    delta_color = (new_q - counts[i]) % m
                    delta = new_q - counts[i]
                    for j in effects[i]:
                        before = state[j] not in allowed[j]
                        after = ((state[j] + delta_color) % m
                                 not in allowed[j])
                        delta += 100 * (int(after) - int(before))
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, new_q)

            if best_move is None:
                i = rng.randrange(n)
                new_q = (counts[i] + rng.randrange(1, m)) % m
            else:
                i, new_q = best_move
            change = (new_q - counts[i]) % m
            counts[i] = new_q
            for j in effects[i]:
                state[j] = (state[j] + change) % m

    if best_counts is not None:
        return [i for i, q in enumerate(best_counts) for _ in range(q)]
    return None


def act(obs, available, memory):
    if "ACTION6" not in available:
        name = next((a for a in available if a != "RESET"), "RESET")
        return name, None, None

    board = _read_board(obs)
    if board is None:
        # Conservative fallback: click a non-background coloured area.
        for y in range(4, 60):
            for x in range(4, 60):
                if int(obs[y, x]) in (8, 9, 10, 12, 14, 15):
                    return "ACTION6", y, x
        return "ACTION6", 32, 32

    scale, ox, oy, grid, tiles = board
    nodes = sorted((t for t in tiles if t[2] != 2),
                   key=lambda t: (t[1], t[0]))
    clues = [t for t in tiles if t[2] == 2]
    positions = {(t[0], t[1]): i for i, t in enumerate(nodes)}
    current = {(t[0], t[1]): t[3] for t in nodes}

    signature = (
        scale,
        tuple((t[0], t[1], t[2],
               tuple(v for row in t[4] for v in row) if t[2] == 2
               else tuple(v == 6 for row in t[4] for v in row)
               if t[2] == 1 else ())
              for t in sorted(tiles, key=lambda t: (t[1], t[0])))
    )

    if memory.get("signature") != signature:
        memory.clear()
        memory["signature"] = signature
        memory["mask"] = [-1] * 9
        memory["edges"] = {}
        memory["seen_colors"] = []
        memory["clicks"] = {}

    previous = memory.get("previous")
    last = memory.get("last")
    if previous is not None and last is not None:
        for pos, color in current.items():
            old = previous.get(pos)
            if old is not None and old != color:
                memory["edges"][old] = color
                for c in (old, color):
                    if c not in memory["seen_colors"]:
                        memory["seen_colors"].append(c)

        if last[2] == 0:
            lx, ly = last[:2]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    pos = (lx + dx * 4 * scale, ly + dy * 4 * scale)
                    if pos in current and pos in previous:
                        memory["mask"][(dy + 1) * 3 + dx + 1] = int(
                            current[pos] != previous[pos]
                        )

    normal_mask = [
        value if value >= 0 else int(i == 4)
        for i, value in enumerate(memory["mask"])
    ]

    requirements = [[] for _ in nodes]
    for x, y, kind, color, patch in clues:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                pos = (x + dx * 4 * scale, y + dy * 4 * scale)
                j = positions.get(pos)
                if j is not None:
                    requirements[j].append(
                        (color, patch[dy + 1][dx + 1] == 0)
                    )

    def satisfies(j, color):
        return all((color == c) == equal
                   for c, equal in requirements[j])

    satisfied = [satisfies(j, t[3]) for j, t in enumerate(nodes)]
    required = [j for j, req in enumerate(requirements) if req]

    effects = []
    for x, y, kind, color, patch in nodes:
        if kind == 1:
            mask = [int(i == 4 or patch[i // 3][i % 3] == 6)
                    for i in range(9)]
        else:
            mask = normal_mask
        affected = []
        for k, enabled in enumerate(mask):
            if enabled:
                dx, dy = k % 3 - 1, k // 3 - 1
                j = positions.get(
                    (x + dx * 4 * scale, y + dy * 4 * scale)
                )
                if j is not None:
                    affected.append(j)
        effects.append(affected)

    legend = _read_legend(grid)
    if legend:
        palette = legend
    else:
        colors = set(memory["seen_colors"])
        colors.update(t[3] for t in nodes)
        colors.update(t[3] for t in clues)
        edges = memory["edges"]
        # Construct a cycle order consistent with transitions seen so far.
        starts = [c for c in colors if c not in edges.values()]
        start = min(starts or colors)
        palette = []
        c = start
        while c in colors and c not in palette:
            palette.append(c)
            c = edges.get(c)
        palette.extend(sorted(colors - set(palette)))

    if len(palette) <= 2:
        plan = _binary_plan(effects, required, satisfied, len(nodes))
    else:
        allowed = [
            {k for k, color in enumerate(palette) if satisfies(j, color)}
            for j in range(len(nodes))
        ]
        plan = _cycle_plan(nodes, effects, palette, allowed, normal_mask)

    clicks = memory["clicks"]

    def information(i):
        x, y, kind, _, _ = nodes[i]
        if kind != 0:
            return 0
        count = 0
        for k, value in enumerate(memory["mask"]):
            if value < 0:
                dx, dy = k % 3 - 1, k // 3 - 1
                if (x + dx * 4 * scale,
                        y + dy * 4 * scale) in positions:
                    count += 1
        return count

    if plan:
        chosen = max(set(plan), key=lambda i: (
            information(i),
            int(not satisfied[i]),
            -clicks.get((nodes[i][0], nodes[i][1]), 0),
            -i
        ))
    elif plan == []:
        # A harmless click can immediately confirm an already solved board.
        harmless = [
            i for i, affected in enumerate(effects)
            if not any(requirements[j] for j in affected)
        ]
        chosen = harmless[0] if harmless else 0
    else:
        # An unexpected transition usually means an unlearned toggle mask.
        chosen = max(range(len(nodes)), key=lambda i: (
            information(i),
            int(not satisfied[i]),
            -clicks.get((nodes[i][0], nodes[i][1]), 0),
            -i
        ))

    x, y, kind, _, _ = nodes[chosen]
    memory["previous"] = current
    memory["last"] = (x, y, kind)
    clicks[(x, y)] = clicks.get((x, y), 0) + 1
    return "ACTION6", int(y + scale), int(x + scale)

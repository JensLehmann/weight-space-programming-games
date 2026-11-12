import numpy as np


def act(obs, available, memory):
    if "ACTION6" not in available:
        if "RESET" in available:
            return ("RESET", None, None)
        return (available[0], None, None)

    # The canonical camera renders the 32x32 game grid at scale 2.
    grid = np.asarray(obs)[::2, ::2]
    h, w = grid.shape

    # The tutorial has an outlined active board alongside illustrative boards.
    # Find the outline rather than treating the illustrations as playable tiles.
    box = None
    for y in range(h - 14):
        if box is not None:
            break
        for x in range(w - 14):
            if (
                np.all(grid[y, x:x + 3] == 2)
                and np.all(grid[y:y + 3, x] == 2)
                and np.all(grid[y, x + 12:x + 15] == 2)
                and np.all(grid[y:y + 3, x + 14] == 2)
                and np.all(grid[y + 14, x:x + 3] == 2)
                and np.all(grid[y + 12:y + 15, x] == 2)
                and np.all(grid[y + 14, x + 12:x + 15] == 2)
                and np.all(grid[y + 12:y + 15, x + 14] == 2)
            ):
                box = (x, y, x + 14, y + 14)
                break

    def in_board(x, y):
        if box is None:
            return True
        return (
            x > box[0] and y > box[1]
            and x + 2 < box[2] and y + 2 < box[3]
        )

    # Estimate the two tile colours. Most levels show them as a small
    # two-colour swatch at the top; the tutorial instead supplies examples.
    hist = [0] * 16
    clue_hist = [0] * 16
    symbols = {0, 2, 3}
    excluded = {0, 2, 3, 4, 5, 6}

    for y in range(h - 2):
        for x in range(w - 2):
            if not in_board(x, y):
                continue
            p = grid[y:y + 3, x:x + 3]
            c = int(p[1, 1])
            if c not in excluded and np.all(p == c):
                hist[c] += 1
            if c not in excluded:
                outer = [
                    int(p[j, i])
                    for j in range(3) for i in range(3)
                    if (i, j) != (1, 1)
                ]
                if all(v in symbols for v in outer):
                    clue_hist[c] += 1

    swatches = []
    for y in range(min(2, h - 3)):
        for x in range(w - 1):
            a = int(grid[y, x])
            b = int(grid[y + 2, x])
            if a == b or a in excluded or b in excluded:
                continue
            if (
                np.all(grid[y:y + 2, x:x + 2] == a)
                and np.all(grid[y + 2:y + 4, x:x + 2] == b)
                and hist[a] + hist[b] > 0
            ):
                score = hist[a] + hist[b] + 3 * (clue_hist[a] + clue_hist[b])
                swatches.append((score, a, b))

    if swatches:
        _, c0, c1 = max(swatches)
    else:
        candidates = [
            c for c in range(16)
            if c not in excluded and hist[c] + clue_hist[c] > 0
        ]
        candidates.sort(
            key=lambda c: hist[c] + 3 * clue_hist[c], reverse=True
        )
        if len(candidates) >= 2:
            c0, c1 = candidates[:2]
        elif len(candidates) == 1:
            c0 = candidates[0]
            # Fallback for a board without a visible swatch or second colour.
            c1 = {9: 8, 8: 12, 12: 8, 14: 15, 15: 14}.get(c0, 8)
            if c1 == c0:
                c1 = 9
        else:
            return ("ACTION6", 32, 32)

    palette = (c0, c1)
    possible = {}

    # Classify 3x3 sprites. A special tile has colour 6 in its click mask.
    for y in range(h - 2):
        for x in range(w - 2):
            if not in_board(x, y):
                continue
            p = grid[y:y + 3, x:x + 3]
            c = int(p[1, 1])
            if c not in palette:
                continue
            if np.all(p == c):
                possible[(x, y)] = ("tile", c, None)
                continue
            vals = [int(v) for v in p.flat]
            if 6 in vals and all(v == c or v == 6 for v in vals):
                mask = tuple(int(p[j, i]) == 6 for j in range(3) for i in range(3))
                possible[(x, y)] = ("special", c, mask)
                continue
            outer = [
                int(p[j, i])
                for j in range(3) for i in range(3)
                if (i, j) != (1, 1)
            ]
            if all(v in symbols for v in outer):
                possible[(x, y)] = ("clue", c, tuple(vals))

    # All board sprites lie on a four-pixel lattice. Prefer the lattice with
    # many sprites and many clue-to-tile adjacencies.
    best_offset = None
    best_score = -1
    for oy in range(4):
        for ox in range(4):
            group = {
                pos: data for pos, data in possible.items()
                if pos[0] % 4 == ox and pos[1] % 4 == oy
            }
            tiles_here = {
                pos for pos, data in group.items() if data[0] != "clue"
            }
            links = 0
            for (x, y), data in group.items():
                if data[0] == "clue":
                    for dy in (-4, 0, 4):
                        for dx in (-4, 0, 4):
                            if (dx or dy) and (x + dx, y + dy) in tiles_here:
                                links += 1
            score = 3 * len(group) + 2 * links
            if score > best_score:
                best_score = score
                best_offset = (ox, oy)

    if best_offset is None:
        return ("ACTION6", 32, 32)

    ox, oy = best_offset
    group = {
        pos: data for pos, data in possible.items()
        if pos[0] % 4 == ox and pos[1] % 4 == oy
    }
    tiles = sorted(
        (pos for pos, data in group.items() if data[0] != "clue"),
        key=lambda p: (p[1], p[0]),
    )
    clues = [
        (pos, data) for pos, data in group.items() if data[0] == "clue"
    ]
    if not tiles:
        return ("ACTION6", 32, 32)

    tile_index = {pos: i for i, pos in enumerate(tiles)}

    # Learn the normal-tile mask from the last click. Unknown entries default
    # to the visible levels' centre-only mask until evidence says otherwise.
    mask = list(memory.get("mask", [0, 0, 0, 0, 1, 0, 0, 0, 0]))
    known = list(memory.get("known", [False] * 9))
    last = memory.pop("last", None)
    if last and last.get("plain"):
        old = {(x, y): c for x, y, c in last["tiles"]}
        lx, ly = last["pos"]
        for j in range(3):
            for i in range(3):
                pos = (lx + 4 * (i - 1), ly + 4 * (j - 1))
                if pos in old and pos in tile_index:
                    new_colour = group[pos][1]
                    k = 3 * j + i
                    mask[k] = int(new_colour != old[pos])
                    known[k] = True
    memory["mask"] = mask
    memory["known"] = known

    # Each clue specifies equality for a 0 symbol and inequality otherwise.
    targets = {}
    conflict = False
    for (x, y), data in clues:
        clue_colour, pixels = data[1], data[2]
        for j in range(3):
            for i in range(3):
                if i == 1 and j == 1:
                    continue
                pos = (x + 4 * (i - 1), y + 4 * (j - 1))
                if pos not in tile_index:
                    continue
                want = clue_colour
                if pixels[3 * j + i] != 0:
                    want = c1 if clue_colour == c0 else c0
                if pos in targets and targets[pos] != want:
                    conflict = True
                targets[pos] = want

    n = len(tiles)
    constrained = sorted(targets, key=lambda p: (p[1], p[0]))
    rows = []

    for target_pos in constrained:
        coeff = 0
        for col, pos in enumerate(tiles):
            dx = (target_pos[0] - pos[0]) // 4
            dy = (target_pos[1] - pos[1]) // 4
            if not (-1 <= dx <= 1 and -1 <= dy <= 1):
                continue
            k = (dy + 1) * 3 + dx + 1
            data = group[pos]
            if data[0] == "special":
                affects = (k == 4) or data[2][k]
            else:
                affects = bool(mask[k])
            if affects:
                coeff |= 1 << col
        rhs = int(group[target_pos][1] != targets[target_pos])
        rows.append(coeff | (rhs << n))

    # Reduced row echelon form over GF(2).
    rank = 0
    pivots = []
    for col in range(n):
        pivot = next(
            (r for r in range(rank, len(rows)) if (rows[r] >> col) & 1),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        pivots.append(col)
        rank += 1

    variable_bits = (1 << n) - 1
    inconsistent = conflict or any(
        (r & variable_bits) == 0 and ((r >> n) & 1) for r in rows
    )

    solution = None
    if not inconsistent:
        particular = 0
        for r, col in enumerate(pivots):
            if (rows[r] >> n) & 1:
                particular |= 1 << col

        pivot_set = set(pivots)
        basis = []
        for free in range(n):
            if free in pivot_set:
                continue
            v = 1 << free
            for r, col in enumerate(pivots):
                if (rows[r] >> free) & 1:
                    v |= 1 << col
            # A free move affecting no constrained tile cannot improve a plan.
            if v != (1 << free):
                basis.append(v)

        solution = particular
        best_weight = solution.bit_count()
        if len(basis) <= 17:
            current = particular
            previous_gray = 0
            for k in range(1, 1 << len(basis)):
                gray = k ^ (k >> 1)
                changed = gray ^ previous_gray
                index = changed.bit_length() - 1
                current ^= basis[index]
                weight = current.bit_count()
                if weight < best_weight:
                    solution, best_weight = current, weight
                previous_gray = gray
        else:
            # Bounded local improvement for unusually underconstrained boards.
            changed = True
            while changed:
                changed = False
                for v in basis:
                    candidate = solution ^ v
                    if candidate.bit_count() < solution.bit_count():
                        solution = candidate
                        changed = True

    if solution:
        chosen = (solution & -solution).bit_length() - 1
    elif solution == 0:
        # Already satisfied: a harmless click may trigger the completion test.
        # If no single click is harmless, the next observation lets us undo it.
        chosen = 0
        for col in range(n):
            if all(((r >> col) & 1) == 0 for r in rows):
                chosen = col
                break
    else:
        # The current assumed mask cannot solve the equations. Probe a plain
        # tile that reveals as many still-unknown offsets as possible.
        chosen = 0
        best_probe = -1
        for col, (x, y) in enumerate(tiles):
            if group[(x, y)][0] != "tile":
                continue
            score = 0
            for j in range(3):
                for i in range(3):
                    pos = (x + 4 * (i - 1), y + 4 * (j - 1))
                    if pos in tile_index:
                        score += 2 if not known[3 * j + i] else 1
            if score > best_probe:
                best_probe = score
                chosen = col

    x, y = tiles[chosen]
    memory["last"] = {
        "pos": (x, y),
        "plain": group[(x, y)][0] == "tile",
        "tiles": [(tx, ty, group[(tx, ty)][1]) for tx, ty in tiles],
    }
    # Click the centre of the 3x3 sprite in display coordinates.
    return ("ACTION6", 2 * y + 3, 2 * x + 3)

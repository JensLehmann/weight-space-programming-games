import math
import numpy as np


def _read_board(obs):
    """Recover pixel scale, tile locations, clue patterns, and special masks."""
    a = np.asarray(obs, dtype=np.int16)
    horizontal = []
    vertical = []
    spacing = 0

    # Ignore the bottom progress-bar row.
    for row in a[:62]:
        p = np.flatnonzero(row[1:] != row[:-1]) + 1
        horizontal.extend(p.tolist())
        for d in np.diff(p):
            spacing = math.gcd(spacing, int(d))

    for x in range(a.shape[1]):
        col = a[:62, x]
        p = np.flatnonzero(col[1:] != col[:-1]) + 1
        vertical.extend(p.tolist())
        for d in np.diff(p):
            spacing = math.gcd(spacing, int(d))

    scale = spacing if 1 <= spacing <= 8 else 1

    def residue(values):
        if not values:
            return 0
        counts = [0] * scale
        for v in values:
            counts[v % scale] += 1
        return max(range(scale), key=lambda i: counts[i])

    ox = residue(horizontal)
    oy = residue(vertical)
    grid = a[oy::scale, ox::scale]
    h, w = grid.shape
    candidates = []

    for y in range(h - 2):
        for x in range(w - 2):
            p = grid[y:y + 3, x:x + 3]
            center = int(p[1, 1])
            outer = [
                int(p[0, 0]), int(p[0, 1]), int(p[0, 2]),
                int(p[1, 0]), int(p[1, 2]),
                int(p[2, 0]), int(p[2, 1]), int(p[2, 2]),
            ]

            kind = None
            mask = 0

            if center not in (4, 5, 6) and np.all(p == center):
                kind = "R"
            elif center not in (0, 2, 3, 4, 5, 6):
                if all(v in (0, 2, 3) for v in outer):
                    kind = "C"
                elif (
                    6 in outer
                    and all(v == center or v == 6 for v in outer)
                ):
                    kind = "S"
                    mask = 1 << 4
                    for j in range(3):
                        for i in range(3):
                            if int(p[j, i]) == 6:
                                mask |= 1 << (3 * j + i)

            if kind is None:
                continue

            # Teaching examples in level 1 have a different background.
            boundary = []
            for xx, yy in (
                (x - 1, y + 1), (x + 3, y + 1),
                (x + 1, y - 1), (x + 1, y + 3),
            ):
                if 0 <= xx < w and 0 <= yy < h:
                    boundary.append(int(grid[yy, xx]))
            on_board = 4 in boundary

            candidates.append({
                "x": x,
                "y": y,
                "kind": kind,
                "color": center,
                "mask": mask,
                "pixels": tuple(int(v) for v in p.ravel()),
                "on_board": on_board,
            })

    if any(t["on_board"] for t in candidates):
        candidates = [t for t in candidates if t["on_board"]]

    controls = sorted(
        (t for t in candidates if t["kind"] != "C"),
        key=lambda t: (t["y"], t["x"]),
    )
    clues = sorted(
        (t for t in candidates if t["kind"] == "C"),
        key=lambda t: (t["y"], t["x"]),
    )
    return controls, clues, scale, ox, oy


def _offset_index(dx, dy):
    if dx % 4 or dy % 4:
        return None
    dx //= 4
    dy //= 4
    if -1 <= dx <= 1 and -1 <= dy <= 1:
        return (dy + 1) * 3 + dx + 1
    return None


def _minimum_toggle_solution(columns, target):
    """Return original column indices, or None if the system is inconsistent."""
    if target == 0:
        return []

    # Zero and duplicate columns cannot help a minimum-weight binary solution.
    unique = {}
    for i, c in enumerate(columns):
        if c and c not in unique:
            unique[c] = i

    values = list(unique)
    originals = [unique[c] for c in values]
    n = len(values)
    lookup = {c: i for i, c in enumerate(values)}

    if target in lookup:
        return [originals[lookup[target]]]

    for i, c in enumerate(values):
        j = lookup.get(target ^ c)
        if j is not None and j > i:
            return [originals[i], originals[j]]

    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairs.setdefault(values[i] ^ values[j], (i, j))

    for i, c in enumerate(values):
        pair = pairs.get(target ^ c)
        if pair is not None and i not in pair:
            return [originals[i], originals[pair[0]], originals[pair[1]]]

    for syndrome, pair in pairs.items():
        other = pairs.get(target ^ syndrome)
        if other is not None and not set(pair).intersection(other):
            return [originals[i] for i in pair + other]

    # Column-space elimination also supplies a null-space basis.
    basis = {}
    null = []
    for i, c in enumerate(values):
        assignment = 1 << i
        while c:
            pivot = c.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (c, assignment)
                break
            bc, ba = basis[pivot]
            c ^= bc
            assignment ^= ba
        if not c:
            null.append(assignment)

    c = target
    solution = 0
    while c:
        pivot = c.bit_length() - 1
        if pivot not in basis:
            return None
        bc, ba = basis[pivot]
        c ^= bc
        solution ^= ba

    best = solution
    best_weight = best.bit_count()
    k = len(null)

    if k <= 21:
        # Gray-code enumeration changes only one basis vector each iteration.
        current = solution
        for i in range(1, 1 << k):
            bit = (i & -i).bit_length() - 1
            current ^= null[bit]
            weight = current.bit_count()
            if weight < best_weight:
                best, best_weight = current, weight
                if weight <= 5:
                    break
    else:
        # Large null spaces: deterministic multi-start local optimization.
        moves = list(null)
        if k <= 70:
            moves.extend(
                null[i] ^ null[j]
                for i in range(k)
                for j in range(i + 1, k)
            )

        seed = 123456789
        for restart in range(20):
            current = solution if restart == 0 else best
            if restart:
                for v in null:
                    seed = (1664525 * seed + 1013904223) & 0xffffffff
                    if seed & 0x80000000:
                        current ^= v

            weight = current.bit_count()
            while True:
                improved = current
                improved_weight = weight
                for v in moves:
                    candidate = current ^ v
                    cw = candidate.bit_count()
                    if cw < improved_weight:
                        improved, improved_weight = candidate, cw
                if improved_weight == weight:
                    break
                current, weight = improved, improved_weight

            if weight < best_weight:
                best, best_weight = current, weight
                if weight <= 5:
                    break

    return [originals[i] for i in range(n) if (best >> i) & 1]


def act(obs, available, memory):
    if "ACTION6" not in available:
        name = available[0] if available else "RESET"
        return name, None, None

    controls, clues, scale, ox, oy = _read_board(obs)
    if not controls:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    positions = {(t["x"], t["y"]): i for i, t in enumerate(controls)}

    fingerprint = repr((
        scale, ox, oy,
        tuple(
            (t["x"], t["y"], t["kind"], t["mask"])
            for t in controls
        ),
        tuple((t["x"], t["y"], t["pixels"]) for t in clues),
    ))

    state = memory.get("ft09_player")
    if not isinstance(state, dict) or state.get("key") != fingerprint:
        state = {
            "key": fingerprint,
            "known": {},
            "last": None,
            "next_color": {},
            "colors": [],
            "steps": 0,
        }
        memory["ft09_player"] = state

    known = state["known"]
    last = state.get("last")

    if last:
        old_colors = {
            (x, y): color for x, y, color in last["colors"]
        }
        for t in controls:
            pos = (t["x"], t["y"])
            if pos not in old_colors:
                continue
            before = old_colors[pos]
            after = t["color"]
            if before != after:
                state["next_color"][str(before)] = after

            if last["kind"] == "R":
                offset = _offset_index(
                    t["x"] - last["x"], t["y"] - last["y"]
                )
                if offset is not None:
                    known[str(offset)] = int(before != after)

    observed_colors = set(state["colors"])
    observed_colors.update(t["color"] for t in controls)
    observed_colors.update(t["color"] for t in clues)
    state["colors"] = sorted(observed_colors)

    # Each clue constrains whichever clickable tiles exist around it.
    requirements = {}
    for clue in clues:
        for j in range(3):
            for i in range(3):
                if i == 1 and j == 1:
                    continue
                pos = (
                    clue["x"] + 4 * (i - 1),
                    clue["y"] + 4 * (j - 1),
                )
                index = positions.get(pos)
                if index is not None:
                    equal = clue["pixels"][3 * j + i] == 0
                    requirements.setdefault(index, []).append(
                        (clue["color"], equal)
                    )

    constrained = sorted(requirements)
    row_of = {index: row for row, index in enumerate(constrained)}

    def satisfies(index, color):
        return all(
            (color == wanted) if equal else (color != wanted)
            for wanted, equal in requirements[index]
        )

    target = 0
    for index in constrained:
        if not satisfies(index, controls[index]["color"]):
            target |= 1 << row_of[index]

    # Unobserved regular offsets initially use the simplest plausible mask.
    regular_mask = 0
    for offset in range(9):
        active = known.get(str(offset), int(offset == 4))
        if active:
            regular_mask |= 1 << offset

    effects = []
    columns = []
    unknown_scores = []

    for control in controls:
        mask = control["mask"] if control["kind"] == "S" else regular_mask
        affected = []
        column = 0
        unknown_score = 0

        for index, other in enumerate(controls):
            offset = _offset_index(
                other["x"] - control["x"], other["y"] - control["y"]
            )
            if offset is None:
                continue

            if (
                control["kind"] == "R"
                and str(offset) not in known
            ):
                unknown_score += 3 if index in requirements else 1

            if (mask >> offset) & 1:
                affected.append(index)
                if index in row_of:
                    column |= 1 << row_of[index]

        effects.append(affected)
        columns.append(column)
        unknown_scores.append(unknown_score)

    chosen = None

    if len(observed_colors) <= 2:
        solution = _minimum_toggle_solution(columns, target)

        if solution:
            def priority(index):
                c = columns[index]
                benefit = (
                    (c & target).bit_count()
                    - (c & ~target).bit_count()
                )
                return unknown_scores[index], benefit, -index

            chosen = max(solution, key=priority)
        elif solution == []:
            # A click is still needed to invoke the game's goal check.
            zero_effect = [i for i, c in enumerate(columns) if c == 0]
            chosen = zero_effect[0] if zero_effect else 0
        else:
            # The partial mask model is inconsistent: test an unknown offset.
            candidates = [
                i for i, t in enumerate(controls)
                if t["kind"] == "R" and unknown_scores[i] > 0
            ]
            if candidates:
                chosen = max(
                    candidates,
                    key=lambda i: (unknown_scores[i], -i),
                )

    if chosen is None:
        # Defensive fallback for longer colour cycles or an incomplete model.
        palette = sorted(observed_colors)
        next_color = state["next_color"]

        def advance(color):
            successor = next_color.get(str(color))
            if successor is not None:
                return successor
            if color in palette and len(palette) > 1:
                return palette[(palette.index(color) + 1) % len(palette)]
            return color

        current_bad = sum(
            not satisfies(i, controls[i]["color"])
            for i in constrained
        )
        best_key = None
        for index, affected in enumerate(effects):
            colors = {i: controls[i]["color"] for i in affected}
            for repetitions in range(1, max(2, len(palette))):
                for i in affected:
                    colors[i] = advance(colors[i])
                bad = current_bad
                for i in affected:
                    if i in requirements:
                        bad -= int(not satisfies(i, controls[i]["color"]))
                        bad += int(not satisfies(i, colors[i]))
                key = (
                    current_bad - bad,
                    -repetitions,
                    unknown_scores[index],
                    -index,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    chosen = index

        if chosen is None:
            chosen = state["steps"] % len(controls)

    tile = controls[chosen]
    state["last"] = {
        "kind": tile["kind"],
        "x": tile["x"],
        "y": tile["y"],
        "colors": [
            [t["x"], t["y"], t["color"]] for t in controls
        ],
    }
    state["steps"] += 1

    row = oy + (tile["y"] + 1) * scale + scale // 2
    col = ox + (tile["x"] + 1) * scale + scale // 2
    return "ACTION6", int(row), int(col)

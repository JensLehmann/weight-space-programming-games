import itertools
import random
import numpy as np


def _scale_and_sample(obs):
    """Recover world pixels and their display-coordinate mapping."""
    a = np.asarray(obs)
    for s in (4, 3, 2):
        x_offsets = []
        y_offsets = []
        for off in range(s):
            ok = True
            for x in range(63):
                if (x + 1 - off) % s and np.any(a[:63, x] != a[:63, x + 1]):
                    ok = False
                    break
            if ok:
                x_offsets.append(off)
            ok = True
            for y in range(62):
                if (y + 1 - off) % s and np.any(a[y, :] != a[y + 1, :]):
                    ok = False
                    break
            if ok:
                y_offsets.append(off)
        if x_offsets and y_offsets:
            ox, oy = x_offsets[0], y_offsets[0]
            ys = list(range(oy, 63, s))
            xs = list(range(ox, 64, s))
            return a[np.ix_(ys, xs)].tolist(), s, ox, oy
    return a[:63, :].tolist(), 1, 0, 0


def _read_board(raw):
    h, w = len(raw), len(raw[0])
    candidates = []

    for y in range(h - 2):
        for x in range(w - 2):
            p = [raw[y + j][x + i] for j in range(3) for i in range(3)]
            c = p[4]
            ring = p[:4] + p[5:]

            if c not in (0, 2, 3, 4, 5, 6) and all(v == c for v in p):
                candidates.append((x, y, "normal", c, p))
            elif (c not in (0, 2, 3, 4, 5, 6)
                  and p[1] == p[3] == p[5] == p[7] == 6
                  and p[0] == p[2] == p[6] == p[8] == c):
                candidates.append((x, y, "special", c, p))
            elif (c not in (0, 2, 3, 4, 5, 6)
                  and all(v in (0, 2, 3) for v in ring)):
                candidates.append((x, y, "clue", c, p))

    # The introductory board is surrounded by four 3-pixel corner brackets.
    # Its other pictures are examples, not interactive tiles.
    bracket = None
    for y in range(max(0, h - 14)):
        if bracket:
            break
        for x in range(max(0, w - 14)):
            points = set()
            for dx, dy, sx, sy in (
                (0, 0, 1, 1), (14, 0, -1, 1),
                (0, 14, 1, -1), (14, 14, -1, -1)
            ):
                for k in range(3):
                    points.add((x + dx + sx * k, y + dy))
                    points.add((x + dx, y + dy + sy * k))
            if all(raw[py][px] == 2 for px, py in points):
                inside = [
                    q for q in candidates
                    if x + 2 <= q[0] <= x + 10
                    and y + 2 <= q[1] <= y + 10
                ]
                if len(inside) >= 5 and any(q[2] == "clue" for q in inside):
                    bracket = (x, y)
                    break

    if bracket:
        bx, by = bracket
        candidates = [
            q for q in candidates
            if bx + 2 <= q[0] <= bx + 10
            and by + 2 <= q[1] <= by + 10
        ]

    groups = {}
    for q in candidates:
        groups.setdefault((q[0] % 4, q[1] % 4), []).append(q)
    if not groups:
        return {}, [], candidates

    def group_score(group):
        cells = sum(q[2] != "clue" for q in group)
        clues = sum(q[2] == "clue" for q in group)
        return (cells + clues, cells, clues)

    group = max(groups.values(), key=group_score)
    cells = {}
    clues = []
    for x, y, kind, color, pixels in group:
        if kind == "clue":
            clues.append((x, y, color, pixels))
        else:
            cells[(x, y)] = (kind, color)
    return cells, clues, candidates


def _legend(raw):
    """Read the vertical color swatch, when one is present."""
    h, w = len(raw), len(raw[0])
    if h < 4:
        return []
    choices = []
    for x in range(w - 1):
        colors = []
        y = 0
        while y + 1 < h:
            c = raw[y][x]
            if (c in (0, 2, 3, 4, 5, 6)
                    or raw[y][x + 1] != c
                    or raw[y + 1][x] != c
                    or raw[y + 1][x + 1] != c):
                break
            if c in colors:
                break
            colors.append(c)
            y += 2
        if len(colors) >= 2:
            choices.append(colors)
    return max(choices, key=len) if choices else []


def _binary_solution(rows, n):
    """Solve rows of (variable mask, right-hand bit), minimizing clicks."""
    rows = [[m, b] for m, b in rows]
    pivots = []
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, len(rows))
                      if (rows[i][0] >> col) & 1), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        pm, pb = rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i][0] >> col) & 1):
                rows[i][0] ^= pm
                rows[i][1] ^= pb
        pivots.append((col, r))
        r += 1
        if r == len(rows):
            break

    if any(m == 0 and b for m, b in rows):
        return None

    pivot_cols = {col for col, _ in pivots}
    particular = 0
    for col, ri in pivots:
        if rows[ri][1]:
            particular |= 1 << col

    basis = []
    for free in range(n):
        if free in pivot_cols:
            continue
        v = 1 << free
        for col, ri in pivots:
            if (rows[ri][0] >> free) & 1:
                v |= 1 << col
        # A unit free variable is optimally left at zero.
        if v.bit_count() > 1:
            basis.append(v)

    best = particular
    best_cost = best.bit_count()
    k = len(basis)
    if k <= 18:
        cur = particular
        for i in range(1, 1 << k):
            bit = (i & -i).bit_length() - 1
            cur ^= basis[bit]
            cost = cur.bit_count()
            if cost < best_cost:
                best, best_cost = cur, cost
    else:
        rng = random.Random(1409 + n * 31 + len(rows))
        for restart in range(16):
            cur = particular
            if restart:
                for v in basis:
                    if rng.randrange(4) == 0:
                        cur ^= v
            while True:
                old_cost = cur.bit_count()
                next_cur = cur
                next_cost = old_cost
                for v in basis:
                    t = cur ^ v
                    c = t.bit_count()
                    if c < next_cost:
                        next_cur, next_cost = t, c
                if next_cost == old_cost and k <= 50:
                    for i in range(k):
                        for j in range(i):
                            t = cur ^ basis[i] ^ basis[j]
                            c = t.bit_count()
                            if c < next_cost:
                                next_cur, next_cost = t, c
                if next_cost == old_cost:
                    break
                cur = next_cur
            if cur.bit_count() < best_cost:
                best, best_cost = cur, cur.bit_count()
    return best


def _effects(pos, kind, normal_pattern):
    if kind == "special":
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = normal_pattern
    x, y = pos
    return [(x + 4 * dx, y + 4 * dy) for dx, dy in offsets]


def _multicolor_plan(cells, allowed, colors, normal_pattern):
    """
    For a multi-color board with self-only ordinary tiles, enumerate
    special-tile clicks and independently repair ordinary tiles.
    """
    if set(normal_pattern) != {(0, 0)}:
        return None
    m = len(colors)
    index = {c: i for i, c in enumerate(colors)}
    if any(c not in index for _, c in cells.values()):
        return None

    specials = [p for p, (k, _) in cells.items() if k == "special"]
    normals = [p for p, (k, _) in cells.items() if k == "normal"]
    if m ** len(specials) > 200000:
        return None

    influenced = {}
    for p in cells:
        influenced[p] = [
            i for i, s in enumerate(specials)
            if p in _effects(s, "special", normal_pattern)
        ]

    best_plan = None
    best_cost = 10**9
    for counts in itertools.product(range(m), repeat=len(specials)):
        cost = sum(counts)
        if cost >= best_cost:
            continue
        plan = []
        valid = True
        for i, count in enumerate(counts):
            plan.extend([specials[i]] * count)

        for p in specials:
            if p not in allowed:
                continue
            cur = index[cells[p][1]]
            cur = (cur + sum(counts[i] for i in influenced[p])) % m
            if colors[cur] not in allowed[p]:
                valid = False
                break
        if not valid:
            continue

        for p in normals:
            if p not in allowed:
                continue
            cur = index[cells[p][1]]
            cur = (cur + sum(counts[i] for i in influenced[p])) % m
            steps = next((d for d in range(m)
                          if colors[(cur + d) % m] in allowed[p]), None)
            if steps is None:
                valid = False
                break
            cost += steps
            if cost >= best_cost:
                valid = False
                break
            plan.extend([p] * steps)

        if valid and cost < best_cost:
            best_cost = cost
            best_plan = plan
    return best_plan


def act(obs, available, memory):
    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    raw, scale, ox, oy = _scale_and_sample(obs)
    cells, clues, _ = _read_board(raw)
    if not cells:
        return "ACTION6", 32, 32

    # Learn an ordinary click's stencil from the preceding transition.
    previous = memory.pop("_ft09_previous", None)
    transitions = memory.setdefault("_ft09_transitions", {})
    known_pattern = memory.setdefault("_ft09_pattern", {})
    if previous:
        old_cells = previous["cells"]
        if set(old_cells) == set(cells):
            for p, old_color in old_cells.items():
                new_color = cells[p][1]
                if old_color != new_color:
                    transitions[old_color] = new_color
            if previous["kind"] == "normal":
                x, y = previous["pos"]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        p = (x + 4 * dx, y + 4 * dy)
                        if p in old_cells and p in cells:
                            known_pattern[(dx, dy)] = int(
                                old_cells[p] != cells[p][1]
                            )

    normal_pattern = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            default = int(dx == 0 and dy == 0)
            if known_pattern.get((dx, dy), default):
                normal_pattern.append((dx, dy))

    seen_colors = {color for _, color in cells.values()}
    seen_colors.update(c for _, _, c, _ in clues)
    legend = _legend(raw)
    if legend and seen_colors.issubset(set(legend)):
        colors = list(legend)
    else:
        colors = sorted(seen_colors)
        if len(colors) > 2 and transitions:
            # Use observed successor relationships when no swatch is visible.
            start = colors[0]
            ordered = [start]
            while ordered[-1] in transitions:
                nxt = transitions[ordered[-1]]
                if nxt not in colors or nxt in ordered:
                    break
                ordered.append(nxt)
            ordered.extend(c for c in colors if c not in ordered)
            colors = ordered

    if len(colors) == 1:
        colors.append(-1)  # The unseen other color in a binary palette.
    if not colors:
        colors = [-1, -2]

    # Each clue constrains the neighboring tiles at the corresponding
    # ring positions: zero means equal; any other ring color means unequal.
    allowed = {}
    palette = set(colors)
    for x, y, clue_color, pixels in clues:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                p = (x + 4 * dx, y + 4 * dy)
                if p not in cells:
                    continue
                a = allowed.setdefault(p, set(palette))
                marker = pixels[(dy + 1) * 3 + dx + 1]
                if marker == 0:
                    a.intersection_update({clue_color})
                else:
                    a.discard(clue_color)

    positions = sorted(
        cells,
        key=lambda p: (cells[p][0] == "special", p[1], p[0])
    )
    chosen = None

    if len(colors) == 2 and all(allowed.values()):
        variable = {p: i for i, p in enumerate(positions)}
        masks = {p: 0 for p in allowed}
        for p in positions:
            for target in _effects(p, cells[p][0], normal_pattern):
                if target in masks:
                    masks[target] |= 1 << variable[p]

        rows = []
        for p, targets in allowed.items():
            if len(targets) == 1:
                target = next(iter(targets))
                rows.append((masks[p], int(cells[p][1] != target)))

        solution = _binary_solution(rows, len(positions))
        if solution is not None and solution:
            selected = [p for i, p in enumerate(positions)
                        if (solution >> i) & 1]
            # Specials first: the subsequent frame gives a fresh solution.
            chosen = next((p for p in selected
                           if cells[p][0] == "special"), selected[0])
        elif solution == 0:
            # A solved initial board still needs a click to invoke the
            # game's completion check. Prefer a harmless click.
            chosen = min(
                positions,
                key=lambda p: sum(
                    q in allowed and len(allowed[q]) == 1
                    for q in _effects(p, cells[p][0], normal_pattern)
                )
            )
    elif len(colors) > 2 and all(allowed.values()):
        plan = _multicolor_plan(cells, allowed, colors, normal_pattern)
        if plan:
            chosen = plan[0]
        elif plan == []:
            chosen = min(
                positions,
                key=lambda p: sum(
                    q in allowed
                    for q in _effects(p, cells[p][0], normal_pattern)
                )
            )

    if chosen is None:
        # An inconsistent model can result from an as-yet-unobserved
        # stencil. Probe an ordinary tile with as many neighbors as possible.
        normals = [p for p in positions if cells[p][0] == "normal"]
        pool = normals or positions
        chosen = max(
            pool,
            key=lambda p: sum(
                (p[0] + 4 * dx, p[1] + 4 * dy) in cells
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            )
        )

    memory["_ft09_previous"] = {
        "pos": chosen,
        "kind": cells[chosen][0],
        "cells": {p: color for p, (_, color) in cells.items()},
    }
    x, y = chosen
    col = ox + scale * (x + 1) + scale // 2
    row = oy + scale * (y + 1) + scale // 2
    return "ACTION6", int(row), int(col)

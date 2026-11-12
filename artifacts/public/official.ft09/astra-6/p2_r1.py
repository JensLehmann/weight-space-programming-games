import itertools
import random
import numpy as np


_SYMBOLS = {0, 2, 3, 4, 5, 6}


def _read_board(obs):
    a = np.asarray(obs, dtype=np.int16).copy()
    a[63, :] = 4  # Exclude the move-budget display.
    h, w = a.shape
    unseen = (a != 4) & (a != 5)
    candidates = []
    rectangles = []

    for sy in range(h):
        for sx in range(w):
            if not unseen[sy, sx]:
                continue
            unseen[sy, sx] = False
            stack = [(sx, sy)]
            xmin = xmax = sx
            ymin = ymax = sy
            area = 0

            while stack:
                x, y = stack.pop()
                area += 1
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
                for xx, yy in ((x - 1, y), (x + 1, y),
                               (x, y - 1), (x, y + 1)):
                    if 0 <= xx < w and 0 <= yy < h and unseen[yy, xx]:
                        unseen[yy, xx] = False
                        stack.append((xx, yy))

            bw, bh = xmax - xmin + 1, ymax - ymin + 1
            if area != bw * bh:
                continue

            surround = a[max(0, ymin - 1):min(h, ymax + 2),
                         max(0, xmin - 1):min(w, xmax + 2)]
            # Black-backed examples are instructional, not interactive.
            if np.count_nonzero(surround == 5) > np.count_nonzero(surround == 4):
                continue

            rectangles.append((xmin, ymin, bw, bh))
            if bw != bh or bw % 3 or not 1 <= bw // 3 <= 8:
                continue
            scale = bw // 3
            patch = a[ymin:ymax + 1, xmin:xmax + 1]
            small = patch[::scale, ::scale]
            if not np.array_equal(
                patch, np.repeat(np.repeat(small, scale, 0), scale, 1)
            ):
                continue

            p = tuple(int(v) for v in small.flat)
            c = p[4]
            if c in _SYMBOLS:
                continue
            if all(v == c for v in p):
                kind = 0  # Ordinary mutable square.
            elif all(v == c or v == 6 for v in p):
                kind = 1  # Mutable square with a displayed click stencil.
            elif all(p[k] in (0, 2, 3) for k in range(9) if k != 4):
                kind = 2  # Constraint clue.
            else:
                continue
            candidates.append((xmin, ymin, scale, kind, p))

    if not candidates:
        return [], [], [], 2

    scales = {}
    for t in candidates:
        scales[t[2]] = scales.get(t[2], 0) + 1
    scale = max(scales, key=scales.get)
    candidates = [t for t in candidates if t[2] == scale]
    tiles = sorted((t for t in candidates if t[3] != 2),
                   key=lambda t: (t[3], t[1], t[0]))
    clues = sorted((t for t in candidates if t[3] == 2),
                   key=lambda t: (t[1], t[0]))

    legend = []
    unit = 2 * scale
    for x, y, bw, bh in rectangles:
        strips = None
        if bw == unit and bh % unit == 0 and 2 <= bh // unit <= 8:
            strips = [a[y + k:y + k + unit, x:x + unit]
                      for k in range(0, bh, unit)]
        elif bh == unit and bw % unit == 0 and 2 <= bw // unit <= 8:
            strips = [a[y:y + unit, x + k:x + k + unit]
                      for k in range(0, bw, unit)]
        if strips is None:
            continue
        values = []
        for strip in strips:
            c = int(strip[0, 0])
            if c in _SYMBOLS or not np.all(strip == c):
                values = []
                break
            values.append(c)
        if len(set(values)) >= 2 and len(values) > len(legend):
            legend = list(dict.fromkeys(values))

    return tiles, clues, legend, scale


def _minimum_binary_solution(rows, n, seed):
    """Solve augmented GF(2) rows and minimize the solution's Hamming weight."""
    rows = rows[:]
    pivots = []
    r = 0
    for col in range(n):
        bit = 1 << col
        pivot = next((k for k in range(r, len(rows)) if rows[k] & bit), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for k in range(len(rows)):
            if k != r and rows[k] & bit:
                rows[k] ^= rows[r]
        pivots.append(col)
        r += 1

    coefficients = (1 << n) - 1
    if any((v & coefficients) == 0 and ((v >> n) & 1) for v in rows):
        return None

    particular = 0
    for k, col in enumerate(pivots):
        if (rows[k] >> n) & 1:
            particular |= 1 << col

    pivot_set = set(pivots)
    basis = []
    for free in range(n):
        if free in pivot_set:
            continue
        v = 1 << free
        for k, col in enumerate(pivots):
            if (rows[k] >> free) & 1:
                v |= 1 << col
        basis.append(v)

    best = particular
    best_weight = best.bit_count()
    if best_weight == 0 or not basis:
        return best

    if len(basis) <= 20:
        current = particular
        for k in range(1, 1 << len(basis)):
            current ^= basis[(k & -k).bit_length() - 1]
            weight = current.bit_count()
            if weight < best_weight:
                best, best_weight = current, weight
                if weight == 0:
                    break
    else:
        rng = random.Random(seed)
        for restart in range(64):
            current = particular if restart == 0 else best
            if restart:
                for v in basis:
                    if rng.randrange(3) == 0:
                        current ^= v
            weight = current.bit_count()
            for _ in range(8):
                changed = False
                order = list(range(len(basis)))
                rng.shuffle(order)
                for k in order:
                    trial = current ^ basis[k]
                    tw = trial.bit_count()
                    if tw < weight:
                        current, weight = trial, tw
                        changed = True
                if not changed:
                    break
            if weight < best_weight:
                best, best_weight = current, weight

    return best


def _colour_order(colours, legend, previous, successors):
    base = []
    for c in legend + previous + sorted(colours):
        if c in colours and c not in base:
            base.append(c)

    def valid(order):
        indexes = {c: i for i, c in enumerate(order)}
        return all(
            a not in indexes or b not in indexes
            or order[(indexes[a] + 1) % len(order)] == b
            for a, b in successors.items()
        )

    if len(base) < 3 or valid(base):
        return base
    if len(base) <= 7:
        for tail in itertools.permutations(base[1:]):
            trial = [base[0]] + list(tail)
            if valid(trial):
                return trial
    return base


def _multicolour_plan(current, domains, affects, tiles, q, ordinary_mask, seed):
    """General fallback, including exact search over a small number of stencils."""
    n = len(tiles)
    special = [i for i, t in enumerate(tiles) if t[3] == 1]
    constrained = {i for i in range(n) if len(domains[i]) < q}
    useful_special = [
        j for j in special if any(i in constrained for i in affects[j])
    ]

    limit = min(65536, max(1, 1500000 // max(1, n * q)))
    combinations = q ** len(useful_special)

    if ordinary_mask == (1 << 4) and combinations <= limit:
        best = None
        best_cost = float("inf")
        for counts in itertools.product(range(q), repeat=len(useful_special)):
            cost = sum(counts)
            if cost >= best_cost:
                continue
            values = current[:]
            plan = [0] * n
            for j, count in zip(useful_special, counts):
                plan[j] = count
                if count:
                    for i in affects[j]:
                        values[i] = (values[i] + count) % q

            if any(values[i] not in domains[i] for i in special):
                continue

            possible = True
            for j, t in enumerate(tiles):
                if t[3] != 0:
                    continue
                step = next(
                    (d for d in range(q) if (values[j] + d) % q in domains[j]),
                    None
                )
                if step is None:
                    possible = False
                    break
                plan[j] = step
                cost += step
                if cost >= best_cost:
                    possible = False
                    break
            if possible and cost < best_cost:
                best, best_cost = plan, cost
        return best

    # Bounded discrete local search for less usual colour cycles or kernels.
    rng = random.Random(seed)
    counts = [0] * n
    values = current[:]
    bad = sum(values[i] not in domains[i] for i in range(n))
    cost = 0
    best = counts[:] if bad == 0 else None
    best_cost = cost if bad == 0 else float("inf")
    iterations = min(2000, max(60, 200000 // max(1, n * q)))

    for iteration in range(iterations):
        chosen = None
        best_delta = 0
        for j in range(n):
            if not affects[j]:
                continue
            for delta in range(1, q):
                new_count = (counts[j] + delta) % q
                dc = new_count - counts[j]
                db = 0
                for i in affects[j]:
                    db += int((values[i] + delta) % q not in domains[i])
                    db -= int(values[i] not in domains[i])
                score_delta = 1000 * db + dc
                if score_delta < best_delta:
                    best_delta = score_delta
                    chosen = (j, delta, db, dc)

        if chosen is None:
            useful = [j for j in range(n) if affects[j]]
            if not useful:
                break
            j = rng.choice(useful)
            delta = rng.randrange(1, q)
            db = sum(
                int((values[i] + delta) % q not in domains[i])
                - int(values[i] not in domains[i])
                for i in affects[j]
            )
            dc = (counts[j] + delta) % q - counts[j]
            chosen = (j, delta, db, dc)

        j, delta, db, dc = chosen
        counts[j] = (counts[j] + delta) % q
        for i in affects[j]:
            values[i] = (values[i] + delta) % q
        bad += db
        cost += dc
        if bad == 0 and cost < best_cost:
            best, best_cost = counts[:], cost

        if iteration % 200 == 199 and best is not None:
            counts = best[:]
            values = current[:]
            for j, count in enumerate(counts):
                for i in affects[j]:
                    values[i] = (values[i] + count) % q
            bad = 0
            cost = sum(counts)

    return best


def act(obs, available, memory):
    if "ACTION6" not in available:
        action = available[0] if available else "RESET"
        return action, None, None

    tiles, clues, legend, scale = _read_board(obs)
    if not tiles:
        return "ACTION6", 32, 32

    signature = (
        tuple((t[0], t[1], t[2], t[3],
               tuple(k for k, v in enumerate(t[4]) if v == 6))
              for t in tiles),
        tuple((t[0], t[1], t[4]) for t in clues)
    )
    if memory.get("signature") != signature:
        memory.clear()
        memory.update({
            "signature": signature,
            "mask": 1 << 4,
            "known": 0,
            "successors": {},
            "palette": [],
            "turn": 0,
            "uses": [0] * len(tiles),
        })

    n = len(tiles)
    colours_now = [t[4][4] for t in tiles]
    positions = {(t[0], t[1]): i for i, t in enumerate(tiles)}

    def at(x, y):
        direct = positions.get((x, y))
        if direct is not None:
            return direct
        for i, t in enumerate(tiles):
            if t[0] <= x < t[0] + 3 * scale and \
               t[1] <= y < t[1] + 3 * scale:
                return i
        return None

    offsets = [(k % 3 - 1, k // 3 - 1) for k in range(9)]
    neighbours = []
    for t in tiles:
        neighbours.append([
            at(t[0] + dx * 4 * scale, t[1] + dy * 4 * scale)
            for dx, dy in offsets
        ])

    last = memory.get("last")
    if last is not None:
        old = last["colours"]
        for before, after in zip(old, colours_now):
            if before != after:
                memory["successors"][before] = after

        j = last["tile"]
        if tiles[j][3] == 0:
            for k, i in enumerate(neighbours[j]):
                if i is None:
                    continue
                bit = 1 << k
                memory["known"] |= bit
                if old[i] != colours_now[i]:
                    memory["mask"] |= bit
                else:
                    memory["mask"] &= ~bit

    palette = set(colours_now)
    palette.update(t[4][4] for t in clues)
    palette.update(legend)
    palette.update(memory["palette"])
    order = _colour_order(
        palette, legend, memory["palette"], memory["successors"]
    )
    memory["palette"] = order
    q = len(order)
    index = {c: i for i, c in enumerate(order)}
    current = [index[c] for c in colours_now]

    allowed = [set(order) for _ in tiles]
    for clue in clues:
        cx, cy, _, _, pixels = clue
        colour = pixels[4]
        for k, (dx, dy) in enumerate(offsets):
            if k == 4:
                continue
            i = at(cx + dx * 4 * scale, cy + dy * 4 * scale)
            if i is None:
                continue
            if pixels[k] == 0:
                allowed[i].intersection_update((colour,))
            else:
                allowed[i].discard(colour)
    domains = [{index[c] for c in domain} for domain in allowed]

    affects = []
    for j, t in enumerate(tiles):
        if t[3] == 0:
            mask = memory["mask"]
        else:
            mask = 1 << 4
            for k, value in enumerate(t[4]):
                if value == 6:
                    mask |= 1 << k
        affects.append(sorted({
            i for k, i in enumerate(neighbours[j])
            if i is not None and mask & (1 << k)
        }))

    constrained = [i for i in range(n) if len(domains[i]) < q]
    constrained_set = set(constrained)
    plan = None

    if q == 2 and all(domains):
        active = [
            j for j in range(n)
            if any(i in constrained_set for i in affects[j])
        ]
        row_index = {i: k for k, i in enumerate(constrained)}
        row_masks = [0] * len(constrained)
        for local_j, j in enumerate(active):
            for i in affects[j]:
                if i in row_index:
                    row_masks[row_index[i]] |= 1 << local_j

        rows = []
        for k, i in enumerate(constrained):
            rhs = int(current[i] not in domains[i])
            rows.append(row_masks[k] | (rhs << len(active)))

        solution = _minimum_binary_solution(
            rows, len(active), 1907 + memory["turn"]
        )
        if solution is not None:
            plan = [0] * n
            for local_j, j in enumerate(active):
                if solution & (1 << local_j):
                    plan[j] = 1
    elif q > 2 and all(domains):
        plan = _multicolour_plan(
            current, domains, affects, tiles, q,
            memory["mask"], 1907 + memory["turn"]
        )
    elif q == 1 and all(domains):
        plan = [0] * n

    def information(j):
        if tiles[j][3] != 0:
            return 0
        return sum(
            i is not None and not (memory["known"] & (1 << k))
            for k, i in enumerate(neighbours[j])
        )

    def improvement(j):
        if q < 2:
            return 0
        return sum(
            int(current[i] not in domains[i])
            - int((current[i] + 1) % q not in domains[i])
            for i in affects[j]
        )

    choices = [j for j in range(n) if plan is not None and plan[j] > 0]
    if choices:
        # Useful planned moves also serve as inexpensive model probes.
        chosen = max(
            choices,
            key=lambda j: (information(j), improvement(j),
                           -memory["uses"][j], -j)
        )
    else:
        probes = [j for j in range(n) if information(j) > 0]
        if plan is None and probes:
            chosen = max(
                probes,
                key=lambda j: (information(j), improvement(j),
                               -memory["uses"][j])
            )
        else:
            chosen = max(
                range(n),
                key=lambda j: (improvement(j), -memory["uses"][j], -j)
            )

    memory["last"] = {"tile": chosen, "colours": colours_now}
    memory["uses"][chosen] += 1
    memory["turn"] += 1

    x, y = tiles[chosen][0], tiles[chosen][1]
    return "ACTION6", int(y + 3 * scale // 2), int(x + 3 * scale // 2)

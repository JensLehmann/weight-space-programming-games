import itertools
import math
import random
import numpy as np


def _read_board(obs):
    a = np.asarray(obs, dtype=int)
    mask = (a != 4) & (a != 5)
    mask = mask.copy()
    mask[63, :] = False
    seen = np.zeros((64, 64), dtype=bool)
    tiles, clues, legends = [], [], []

    for y0 in range(63):
        for x0 in range(64):
            if seen[y0, x0] or not mask[y0, x0]:
                continue

            stack = [(y0, x0)]
            seen[y0, x0] = True
            xmin = xmax = x0
            ymin = ymax = y0

            while stack:
                y, x = stack.pop()
                xmin, xmax = min(xmin, x), max(xmax, x)
                ymin, ymax = min(ymin, y), max(ymax, y)
                for yy, xx in ((y - 1, x), (y + 1, x),
                               (y, x - 1), (y, x + 1)):
                    if (0 <= yy < 63 and 0 <= xx < 64
                            and mask[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = True
                        stack.append((yy, xx))

            h, w = ymax - ymin + 1, xmax - xmin + 1
            r = a[ymin:ymax + 1, xmin:xmax + 1]

            # The black-backed diagrams in the tutorial are not interactive.
            ring = []
            if ymin:
                ring.extend(a[ymin - 1, xmin:xmax + 1].tolist())
            if ymax + 1 < 63:
                ring.extend(a[ymax + 1, xmin:xmax + 1].tolist())
            if xmin:
                ring.extend(a[ymin:ymax + 1, xmin - 1].tolist())
            if xmax + 1 < 64:
                ring.extend(a[ymin:ymax + 1, xmax + 1].tolist())
            if ring.count(4) == 0 or ring.count(5) > ring.count(4):
                continue

            # Palette swatches consist of adjoining, solid-colored squares.
            for z in (r, r.T):
                zh, zw = z.shape
                if zh > zw and np.all(z == z[:, :1]):
                    runs = []
                    for c in z[:, 0].tolist():
                        if runs and runs[-1][0] == c:
                            runs[-1][1] += 1
                        else:
                            runs.append([int(c), 1])
                    cs = [v[0] for v in runs]
                    if (len(cs) >= 2 and len(set(cs)) == len(cs)
                            and all(length == zw for _, length in runs)
                            and not set(cs).intersection({0, 2, 3, 4, 5, 6})):
                        legends.append(cs)

            if h != w or w % 3 or not 3 <= w <= 18:
                continue
            s = w // 3
            q = r[s // 2::s, s // 2::s]
            if q.shape != (3, 3):
                continue
            if not np.array_equal(r, np.repeat(np.repeat(q, s, 0), s, 1)):
                continue

            flat = tuple(int(v) for v in q.ravel())
            c = flat[4]
            if c in {0, 2, 3, 4, 5, 6}:
                continue
            outer = flat[:4] + flat[5:]

            if all(v == c for v in flat):
                tiles.append((xmin, ymin, s, "n", 0, c))
            elif all(v in {0, 2, 3} for v in outer):
                clues.append((xmin, ymin, s, flat))
            elif 6 in outer and all(v in {c, 6} for v in flat):
                bits = sum(1 << i for i, v in enumerate(flat) if v == 6)
                tiles.append((xmin, ymin, s, "s", bits | (1 << 4), c))

    if not tiles:
        return [], [], legends, 2

    scales = {}
    for t in tiles:
        scales[t[2]] = scales.get(t[2], 0) + 1
    for c in clues:
        scales[c[2]] = scales.get(c[2], 0) + 2
    s = max(scales, key=scales.get)
    tiles = sorted((t for t in tiles if t[2] == s),
                   key=lambda t: (t[1], t[0]))
    clues = sorted((c for c in clues if c[2] == s),
                   key=lambda c: (c[1], c[0]))
    return tiles, clues, legends, s


def _palette(actual, clue_colors, legends, next_colors):
    actual = set(actual)
    good = [p for p in legends if actual.issubset(p)]
    if good:
        base = list(min(good, key=len))
    else:
        base = sorted(actual | set(clue_colors))

    if not base:
        base = [9, 8]
    elif len(base) == 1:
        other = {8: 9, 9: 8, 12: 8, 14: 15, 15: 14}.get(base[0], 8)
        if other == base[0]:
            other = 9
        base.append(other)

    def agrees(p):
        where = {c: i for i, c in enumerate(p)}
        return all(
            old not in where or new not in where
            or p[(where[old] + 1) % len(p)] == new
            for old, new in next_colors.items()
        )

    if agrees(base) or len(base) > 7:
        return base
    for tail in itertools.permutations(base[1:]):
        p = [base[0]] + list(tail)
        if agrees(p):
            return p
    return base


def _binary_plan(effects, current, allowed):
    n = len(current)
    rows = []
    for j, goals in enumerate(allowed):
        if not goals:
            return None
        if len(goals) == 2:
            continue
        mask = 0
        for i, targets in enumerate(effects):
            if j in targets:
                mask |= 1 << i
        rhs = current[j] ^ next(iter(goals))
        rows.append(mask | (rhs << n))

    rank = 0
    pivots = []
    for col in range(n):
        pivot = next((r for r in range(rank, len(rows))
                      if (rows[r] >> col) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        pivots.append(col)
        rank += 1

    low_mask = (1 << n) - 1
    if any((r & low_mask) == 0 and ((r >> n) & 1) for r in rows):
        return None

    solution = 0
    for r, p in enumerate(pivots):
        if (rows[r] >> n) & 1:
            solution |= 1 << p

    pivot_set = set(pivots)
    basis = []
    for f in range(n):
        if f in pivot_set:
            continue
        v = 1 << f
        for r, p in enumerate(pivots):
            if (rows[r] >> f) & 1:
                v |= 1 << p
        # An entirely irrelevant action cannot improve a solution.
        if v.bit_count() > 1:
            basis.append(v)

    best = solution
    best_cost = best.bit_count()
    d = len(basis)

    if d <= 20:
        z = solution
        for g in range(1, 1 << d):
            z ^= basis[(g & -g).bit_length() - 1]
            cost = z.bit_count()
            if cost < best_cost:
                best, best_cost = z, cost
                if cost == 0:
                    break
    else:
        rng = random.Random(193 + n)

        def descend(z):
            for _ in range(10):
                changed = False
                cost = z.bit_count()
                for v in basis:
                    q = z ^ v
                    qc = q.bit_count()
                    if qc < cost:
                        z, cost, changed = q, qc, True
                if not changed:
                    break
            return z

        for trial in range(48):
            z = best if trial else solution
            if trial:
                for _ in range(1 + trial % 9):
                    z ^= basis[rng.randrange(d)]
            z = descend(z)
            if z.bit_count() < best_cost:
                best, best_cost = z, z.bit_count()

        for _ in range(4):
            improved = False
            for i in range(d):
                for j in range(i):
                    q = best ^ basis[i] ^ basis[j]
                    if q.bit_count() < best_cost:
                        best, best_cost = q, q.bit_count()
                        improved = True
            if not improved:
                break

    return [(best >> i) & 1 for i in range(n)]


def _independent_plan(effects, current, allowed, k):
    """Exact bounded search when most controls affect one constrained cell."""
    n = len(current)
    relevant = {j for j in range(n) if len(allowed[j]) < k}
    if any(not allowed[j] for j in relevant):
        return None

    independent = {}
    coupled = []
    used_columns = set()
    for i, targets in enumerate(effects):
        col = tuple(j for j in targets if j in relevant)
        if not col or col in used_columns:
            continue
        used_columns.add(col)
        if len(col) == 1:
            independent[col[0]] = i
        else:
            coupled.append((i, col))

    # Larger nonlocal systems are better handled by elimination.
    if len(coupled) > 15:
        return None

    fixed = relevant - set(independent)
    last = {j: -1 for j in fixed}
    for d, (_, col) in enumerate(coupled):
        for j in col:
            if j in last:
                last[j] = d
    if any(last[j] == -1 and current[j] not in allowed[j] for j in fixed):
        return None

    finalized = [[] for _ in coupled]
    for j, d in last.items():
        if d >= 0:
            finalized[d].append(j)

    repair = {
        j: [min((g - c) % k for g in allowed[j]) for c in range(k)]
        for j in independent
    }
    state = list(current)
    values = [0] * len(coupled)
    best = None
    best_cost = n * k + 1
    visited = 0

    def search(d, cost):
        nonlocal best, best_cost, visited
        visited += 1
        if visited > 150000 or cost >= best_cost:
            return
        if d == len(coupled):
            result = [0] * n
            total = cost
            for j, action in independent.items():
                count = repair[j][state[j]]
                result[action] = count
                total += count
                if total >= best_cost:
                    return
            for e, (action, _) in enumerate(coupled):
                result[action] = values[e]
            best, best_cost = result, total
            return

        action, col = coupled[d]
        for value in range(k):
            if (visited <= 150000 and cost + value < best_cost
                    and all(state[j] in allowed[j] for j in finalized[d])):
                values[d] = value
                search(d + 1, cost + value)
            for j in col:
                state[j] = (state[j] + 1) % k

    search(0, 0)
    return best


def _modular_plan(effects, current, allowed, k):
    """General modular elimination, with sampled inequality targets."""
    n = len(current)
    ids = [j for j in range(n) if len(allowed[j]) < k]
    if any(not allowed[j] for j in ids):
        return None
    if not ids:
        return [0] * n

    m = len(ids)
    A = np.array([[int(j in effects[i]) for i in range(n)]
                  for j in ids], dtype=np.int64)
    R = np.concatenate((A.copy(), np.eye(m, dtype=np.int64)), axis=1)
    pivots = []
    rank = 0

    for col in range(n):
        pivot = next((r for r in range(rank, m)
                      if math.gcd(int(R[r, col]), k) == 1), None)
        if pivot is None:
            continue
        R[[rank, pivot]] = R[[pivot, rank]]
        R[rank] = R[rank] * pow(int(R[rank, col]), -1, k) % k
        factors = R[:, col].copy()
        factors[rank] = 0
        R = (R - factors[:, None] * R[rank]) % k
        pivots.append(col)
        rank += 1
        if rank == m:
            break

    basis = []
    pivot_set = set(pivots)
    for f in range(n):
        if f in pivot_set:
            continue
        v = np.zeros(n, dtype=np.int64)
        v[f] = 1
        for r, p in enumerate(pivots):
            v[p] = -R[r, f] % k
        if np.count_nonzero(v) > 1 and np.all((A @ v) % k == 0):
            basis.append(v)

    rng = random.Random(419 + sum(current))
    best, best_cost = None, n * k + 1

    for trial in range(32):
        goals = []
        for j in ids:
            choices = sorted(allowed[j])
            if trial == 0:
                g = min(choices, key=lambda c: (c - current[j]) % k)
            else:
                g = rng.choice(choices)
            goals.append((g - current[j]) % k)
        rhs = np.array(goals, dtype=np.int64)
        transformed = R[:, n:] @ rhs % k
        x = np.zeros(n, dtype=np.int64)
        for r, p in enumerate(pivots):
            x[p] = transformed[r]
        if not np.array_equal(A @ x % k, rhs):
            continue

        if trial % 3 == 2 and basis:
            for _ in range(3):
                x = (x + rng.randrange(k) * rng.choice(basis)) % k

        cost = int(x.sum())
        for _ in range(5):
            changed = False
            for v in basis:
                for multiple in range(1, k):
                    q = (x + multiple * v) % k
                    qc = int(q.sum())
                    if qc < cost:
                        x, cost, changed = q, qc, True
            if not changed:
                break
        if cost < best_cost:
            best, best_cost = x.tolist(), cost

    return best


def act(obs, available, memory):
    if "ACTION6" not in available:
        name = next((a for a in available if a != "RESET"), "RESET")
        return name, None, None

    tiles, clues, legends, scale = _read_board(obs)
    if not tiles:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    n = len(tiles)
    positions = {(t[0], t[1]): i for i, t in enumerate(tiles)}
    colors = [t[5] for t in tiles]
    key = (
        scale,
        tuple((t[0], t[1], t[3], t[4]) for t in tiles),
        tuple((c[0], c[1], c[3]) for c in clues),
    )

    m = memory.get("_ft09")
    if not isinstance(m, dict) or m.get("board") != key:
        m = {
            "board": key, "normal": {}, "special": {},
            "next": {}, "seen": [], "tried": {}
        }
        memory["_ft09"] = m

    pitch = 4 * scale
    neighbors = []
    for t in tiles:
        ns = {}
        for d in range(9):
            dx, dy = d % 3 - 1, d // 3 - 1
            j = positions.get((t[0] + pitch * dx, t[1] + pitch * dy))
            if j is not None:
                ns[d] = j
        neighbors.append(ns)

    previous = m.pop("last", None)
    followed = False
    if previous is not None:
        old = previous["colors"]
        action = previous["action"]
        followed = previous["prediction"] == colors
        for before, after in zip(old, colors):
            if before != after:
                m["next"][before] = after

        observed = {d: int(old[j] != colors[j])
                    for d, j in neighbors[action].items()}
        if tiles[action][3] == "n":
            m["normal"].update(observed)
        else:
            m["special"].setdefault(action, {}).update(observed)

    m["seen"] = sorted(set(m["seen"]) | set(colors))
    palette = _palette(
        m["seen"], [c[3][4] for c in clues], legends, m["next"]
    )
    k = len(palette)
    index = {c: i for i, c in enumerate(palette)}
    current = [index[c] for c in colors]

    effects = []
    unknown = []
    for i, t in enumerate(tiles):
        targets = []
        if t[3] == "n":
            pattern = {d: m["normal"].get(d, int(d == 4))
                       for d in range(9)}
            unknown.append(sum(d not in m["normal"] for d in neighbors[i]))
        else:
            pattern = {d: int(bool(t[4] & (1 << d))) for d in range(9)}
            pattern.update(m["special"].get(i, {}))
            unknown.append(0)
        for d, j in neighbors[i].items():
            if pattern[d]:
                targets.append(j)
        effects.append(tuple(sorted(targets)))

    allowed = [set(range(k)) for _ in tiles]
    for x, y, _, pixels in clues:
        central = pixels[4]
        for d in range(9):
            if d == 4:
                continue
            dx, dy = d % 3 - 1, d // 3 - 1
            j = positions.get((x + pitch * dx, y + pitch * dy))
            if j is None:
                continue
            if pixels[d] == 0:
                allowed[j].intersection_update(
                    {index[central]} if central in index else set()
                )
            elif central in index:
                allowed[j].discard(index[central])

    signature = (
        tuple(palette), tuple(effects),
        tuple(tuple(sorted(g)) for g in allowed)
    )
    plan = None
    saved = m.get("plan")
    if (followed and m.get("signature") == signature
            and isinstance(saved, list) and len(saved) == n and any(saved)):
        plan = list(saved)

    if plan is None:
        if k == 2:
            plan = _binary_plan(effects, current, allowed)
        else:
            plan = _independent_plan(effects, current, allowed, k)
            if plan is None:
                plan = _modular_plan(effects, current, allowed, k)

    state_key = tuple(colors)
    tried = m["tried"].setdefault(state_key, [0] * n)
    candidates = [] if plan is None else [i for i, v in enumerate(plan) if v]

    if candidates:
        # Favor useful moves that also reveal unknown ordinary-click effects.
        action = max(
            candidates,
            key=lambda i: (
                unknown[i],
                int(current[i] not in allowed[i]),
                len(effects[i]),
                -tried[i],
                -i,
            )
        )
        plan[action] -= 1
        m["plan"] = plan
        m["signature"] = signature
    else:
        m["plan"] = None

        def fallback_score(i):
            predicted = list(current)
            for j in effects[i]:
                predicted[j] = (predicted[j] + 1) % k
            bad = sum(predicted[j] not in allowed[j] for j in range(n))
            if plan is None and unknown[i]:
                return (0, tried[i], -unknown[i], bad, i)
            return (1, bad + 2 * tried[i], -unknown[i], tried[i], i)

        action = min(range(n), key=fallback_score)

    tried[action] += 1
    if len(m["tried"]) > 200:
        oldest = next(iter(m["tried"]))
        if oldest != state_key:
            del m["tried"][oldest]

    prediction = list(colors)
    for j in effects[action]:
        prediction[j] = palette[(current[j] + 1) % k]
    m["last"] = {
        "colors": list(colors),
        "action": action,
        "prediction": prediction,
    }

    t = tiles[action]
    col = int(t[0] + (3 * scale) // 2)
    row = int(t[1] + (3 * scale) // 2)
    return "ACTION6", row, col

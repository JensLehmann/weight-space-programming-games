import itertools
import math
import random
import numpy as np


_BG = {4, 5}
_OFFSETS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _read_board(obs):
    a = np.asarray(obs, dtype=int)

    # Infer the integer display scale. Ignore the bottom status bar.
    scale = 1
    for s in (4, 2):
        h = (63 // s) * s
        if 64 % s:
            continue
        b = a[:h].reshape(h // s, s, 64 // s, s)
        if np.all(b == b[:, :1, :, :1]):
            scale = s
            break

    g = a[::scale, ::scale]
    H, W = g.shape
    tiles, clues = [], []

    def perimeter(x, y, w, h):
        p = []
        if y:
            p.extend(g[y - 1, x:x + w].tolist())
        if y + h < H:
            p.extend(g[y + h, x:x + w].tolist())
        if x:
            p.extend(g[y:y + h, x - 1].tolist())
        if x + w < W:
            p.extend(g[y:y + h, x + w].tolist())
        return p

    for y in range(H - 2):
        for x in range(W - 2):
            p = g[y:y + 3, x:x + 3]
            if np.any((p == 4) | (p == 5)):
                continue
            edge = perimeter(x, y, 3, 3)
            if not edge or any(v not in _BG for v in edge):
                continue
            # The tutorial's non-interactive examples have background 5.
            if 4 not in edge:
                continue

            c = int(p[1, 1])
            flat = tuple(int(v) for v in p.flat)
            border = flat[:4] + flat[5:]

            if all(v == c for v in flat):
                tiles.append((x, y, c, (4,)))
            elif c != 6 and all(v in (c, 6) for v in flat):
                mask = tuple(sorted({4} | {
                    k for k, v in enumerate(flat) if v == 6
                }))
                tiles.append((x, y, c, mask))
            elif all(v in (0, 2, 3) for v in border):
                clues.append((x, y, c, flat))

    # Read small rectangular palette legends, when present.
    seen = set()
    legends = []
    for y in range(H):
        for x in range(W):
            if (x, y) in seen or int(g[y, x]) in _BG:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            comp = []
            while stack:
                xx, yy = stack.pop()
                comp.append((xx, yy))
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if (0 <= nx < W and 0 <= ny < H
                            and (nx, ny) not in seen
                            and int(g[ny, nx]) not in _BG):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs, ys = zip(*comp)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            if len(comp) != w * h:
                continue
            colors = []
            if w == 2 and 4 <= h <= 16 and h % 2 == 0:
                for yy in range(y0, y1 + 1, 2):
                    b = g[yy:yy + 2, x0:x1 + 1]
                    if not np.all(b == b[0, 0]):
                        colors = []
                        break
                    colors.append(int(b[0, 0]))
            elif h == 2 and 4 <= w <= 16 and w % 2 == 0:
                for xx in range(x0, x1 + 1, 2):
                    b = g[y0:y1 + 1, xx:xx + 2]
                    if not np.all(b == b[0, 0]):
                        colors = []
                        break
                    colors.append(int(b[0, 0]))
            if len(colors) >= 2 and len(set(colors)) == len(colors):
                legends.append(colors)

    tiles.sort(key=lambda t: (t[1], t[0]))
    clues.sort(key=lambda t: (t[1], t[0]))
    return scale, tiles, clues, legends


def _min_binary_solution(rows, n):
    """Solve a binary linear system and minimize the solution's Hamming weight."""
    piv = {}
    for mask, rhs in rows:
        rhs = int(rhs)
        while mask:
            p = mask.bit_length() - 1
            if p not in piv:
                piv[p] = (mask, rhs)
                break
            pm, pr = piv[p]
            mask ^= pm
            rhs ^= pr
        if not mask and rhs:
            return None

    order = sorted(piv)

    def complete(v, homogeneous):
        for p in order:
            mask, rhs = piv[p]
            bit = ((mask & v).bit_count() & 1)
            if not homogeneous:
                bit ^= rhs
            if bit:
                v |= 1 << p
        return v

    particular = complete(0, False)
    basis = [
        complete(1 << j, True)
        for j in range(n) if j not in piv
    ]
    best = particular
    best_cost = best.bit_count()
    if not basis or best_cost == 0:
        return best

    d = len(basis)
    if d <= 21:
        v = particular
        for k in range(1, 1 << d):
            j = (k & -k).bit_length() - 1
            v ^= basis[j]
            cost = v.bit_count()
            if cost < best_cost:
                best, best_cost = v, cost
                if cost == 0:
                    break
        return best

    # Large nullspaces: descent with single/pair moves and randomized restarts.
    rng = random.Random(particular + 7919 * n)
    for attempt in range(60):
        v = best if attempt else particular
        if attempt:
            for b in basis:
                if rng.random() < 0.14:
                    v ^= b
        for _ in range(8):
            old = v.bit_count()
            seq = basis[:]
            rng.shuffle(seq)
            for b in seq:
                w = v ^ b
                if w.bit_count() < v.bit_count():
                    v = w
            if v.bit_count() == old:
                pair_best = v
                pair_cost = old
                for i in range(d):
                    for j in range(i):
                        w = v ^ basis[i] ^ basis[j]
                        cost = w.bit_count()
                        if cost < pair_cost:
                            pair_best, pair_cost = w, cost
                v = pair_best
                if pair_cost == old:
                    break
        cost = v.bit_count()
        if cost < best_cost:
            best, best_cost = v, cost
    return best


def _direct_multicolor(current, allowed, effects, special, q):
    """Exact search when ordinary tiles affect only themselves."""
    n = len(current)
    k = len(special)
    combinations = q ** k
    rng = random.Random(173 + sum(current))

    if combinations <= 150000:
        choices = itertools.product(range(q), repeat=k)
    else:
        def samples():
            yield (0,) * k
            for _ in range(15000):
                yield tuple(rng.randrange(q) for _ in range(k))
        choices = samples()

    best, best_cost = None, float("inf")
    special_set = set(special)
    for values in choices:
        cost = sum(values)
        if cost >= best_cost:
            continue
        delta = [0] * n
        counts = [0] * n
        for j, count in zip(special, values):
            counts[j] = count
            if count:
                for i in effects[j]:
                    delta[i] = (delta[i] + count) % q

        if any((current[i] + delta[i]) % q not in allowed[i]
               for i in special):
            continue

        good = True
        for i in range(n):
            if i in special_set:
                continue
            start = (current[i] + delta[i]) % q
            increments = [(target - start) % q for target in allowed[i]]
            if not increments:
                good = False
                break
            count = min(increments)
            counts[i] = count
            cost += count
            if cost >= best_cost:
                good = False
                break
        if good and cost < best_cost:
            best, best_cost = counts, cost
    return best


def _modular_solution(matrix, rhs, q):
    """Modular Gaussian elimination, including any invertible pivots."""
    if not matrix:
        return [], []
    n = len(matrix[0])
    a = [[v % q for v in row] + [r % q]
         for row, r in zip(matrix, rhs)]
    rank, pivots = 0, []
    for col in range(n):
        found = next((r for r in range(rank, len(a))
                      if math.gcd(a[r][col], q) == 1), None)
        if found is None:
            continue
        a[rank], a[found] = a[found], a[rank]
        inv = pow(a[rank][col], -1, q)
        a[rank] = [(v * inv) % q for v in a[rank]]
        for r in range(len(a)):
            if r != rank and a[r][col]:
                f = a[r][col]
                a[r] = [(u - f * v) % q
                        for u, v in zip(a[r], a[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(a):
            break

    for row in a[rank:]:
        if any(row[:-1]) or row[-1]:
            return None

    part = [0] * n
    for r, col in enumerate(pivots):
        part[col] = a[r][-1]
    free = [j for j in range(n) if j not in pivots]
    basis = []
    for j in free:
        b = [0] * n
        b[j] = 1
        for r, col in enumerate(pivots):
            b[col] = (-a[r][j]) % q
        basis.append(b)
    return part, basis


def _general_multicolor(current, allowed, effects, q):
    n = len(current)
    constrained = [i for i in range(n) if len(allowed[i]) < q]
    active = [j for j in range(n)
              if any(i in effects[j] for i in constrained)]
    matrix = [[int(i in effects[j]) for j in active]
              for i in constrained]
    rng = random.Random(65537 + sum(current))
    best, best_cost = None, float("inf")

    for trial in range(40):
        targets = []
        for i in constrained:
            options = sorted(allowed[i])
            if not options:
                return None
            if trial == 0:
                target = min(options, key=lambda v: (v - current[i]) % q)
            else:
                target = rng.choice(options)
            targets.append((target - current[i]) % q)
        result = _modular_solution(matrix, targets, q)
        if result is None:
            continue
        part, basis = result
        for restart in range(4):
            v = part[:]
            if restart:
                for b in basis:
                    f = rng.randrange(q)
                    v = [(x + f * y) % q for x, y in zip(v, b)]
            for _ in range(5):
                improved = False
                for b in basis:
                    old_cost = sum(v)
                    chosen = v
                    for f in range(1, q):
                        w = [(x + f * y) % q for x, y in zip(v, b)]
                        if sum(w) < sum(chosen):
                            chosen = w
                    if sum(chosen) < old_cost:
                        v = chosen
                        improved = True
                if not improved:
                    break
            cost = sum(v)
            if cost < best_cost:
                best_cost = cost
                best = [0] * n
                for j, count in zip(active, v):
                    best[j] = count
    return best


def act(obs, available, memory):
    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    scale, tiles, clues, legends = _read_board(obs)
    if not tiles:
        # A conservative recovery for an unrecognizable terminal/transition frame.
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    signature = (
        scale,
        tuple((x, y, mask) for x, y, c, mask in tiles),
        tuple(clues),
    )
    state = memory.get("_ft09")
    if not isinstance(state, dict) or state.get("signature") != signature:
        state = {
            "signature": signature,
            "known": [None, None, None, None, 1, None, None, None, None],
        }
        memory["_ft09"] = state

    positions = {(t[0], t[1]): i for i, t in enumerate(tiles)}
    raw_colors = [t[2] for t in tiles]
    known = state["known"]

    # Learn the ordinary-click mask from the previous action's changes.
    previous = state.get("previous")
    if previous and previous["normal"]:
        x, y = previous["position"]
        before = previous["colors"]
        if len(before) == len(tiles):
            for k, (dx, dy) in enumerate(_OFFSETS):
                i = positions.get((x + 4 * dx, y + 4 * dy))
                if i is not None:
                    known[k] = int(before[i] != raw_colors[i])

    detected = set(raw_colors) | {c[2] for c in clues}
    palette = list(state.get("palette", []))
    for legend in legends:
        if detected <= set(legend):
            palette = legend[:]
            break
    else:
        if not palette:
            palette = sorted(detected)
        else:
            palette.extend(sorted(detected - set(palette)))

    if len(palette) == 1:
        # Normally the legend or a clue exposes the other colour.
        c = palette[0]
        other = {8: 9, 9: 8, 12: 8, 14: 15, 15: 14}.get(c, 8)
        if other == c:
            other = 9
        palette.append(other)
    state["palette"] = palette

    q = len(palette)
    color_index = {c: i for i, c in enumerate(palette)}
    current = [color_index[c] for c in raw_colors]
    allowed = [set(range(q)) for _ in tiles]

    # A zero on a clue means equality; every other border colour means inequality.
    for x, y, center, pixels in clues:
        reference = color_index[center]
        for k, (dx, dy) in enumerate(_OFFSETS):
            if k == 4:
                continue
            i = positions.get((x + 4 * dx, y + 4 * dy))
            if i is None:
                continue
            if pixels[k] == 0:
                allowed[i].intersection_update({reference})
            else:
                allowed[i].discard(reference)

    normal_mask = tuple(k for k, value in enumerate(known) if value == 1)
    effects = []
    special = []
    for j, (x, y, c, mask) in enumerate(tiles):
        is_special = mask != (4,)
        if is_special:
            special.append(j)
            use = mask
        else:
            use = normal_mask
        affected = set()
        for k in use:
            dx, dy = _OFFSETS[k]
            i = positions.get((x + 4 * dx, y + 4 * dy))
            if i is not None:
                affected.add(i)
        effects.append(affected)

    constrained = {i for i in range(len(tiles)) if len(allowed[i]) < q}
    solved = all(current[i] in allowed[i] for i in range(len(tiles)))
    counts = None

    if not solved and all(allowed):
        if q == 2:
            active = [j for j, e in enumerate(effects) if e & constrained]
            rows = []
            for i in sorted(constrained):
                if len(allowed[i]) != 1:
                    continue
                mask = 0
                for column, j in enumerate(active):
                    if i in effects[j]:
                        mask |= 1 << column
                target = next(iter(allowed[i]))
                rows.append((mask, int(current[i] != target)))
            solution = _min_binary_solution(rows, len(active))
            if solution is not None:
                counts = [0] * len(tiles)
                for column, j in enumerate(active):
                    counts[j] = (solution >> column) & 1
        elif normal_mask == (4,):
            counts = _direct_multicolor(
                current, allowed, effects, special, q
            )
        else:
            counts = _general_multicolor(current, allowed, effects, q)

    special_set = set(special)

    def information(j):
        if j in special_set:
            return 0
        x, y = tiles[j][:2]
        return sum(
            known[k] is None
            and (x + 4 * dx, y + 4 * dy) in positions
            for k, (dx, dy) in enumerate(_OFFSETS)
        )

    def priority(j):
        useful = sum(current[i] not in allowed[i] for i in effects[j])
        return information(j), useful, -len(effects[j]), -j

    if counts is not None and any(counts):
        candidates = [j for j, count in enumerate(counts) if count]
        chosen = max(candidates, key=priority)
    elif solved:
        # The engine checks victory only after a valid tile click.
        chosen = min(
            range(len(tiles)),
            key=lambda j: (len(effects[j] & constrained),
                           len(effects[j]), j),
        )
    else:
        # Missing mask information can make the provisional system inconsistent.
        # Probe the ordinary tile that exposes the most unknown offsets.
        normals = [j for j in range(len(tiles)) if j not in special_set]
        candidates = normals if normals else list(range(len(tiles)))
        chosen = max(candidates, key=priority)

    x, y = tiles[chosen][:2]
    state["previous"] = {
        "normal": chosen not in special_set,
        "position": (x, y),
        "colors": raw_colors[:],
    }

    row = (y + 1) * scale + scale // 2
    col = (x + 1) * scale + scale // 2
    return "ACTION6", int(row), int(col)

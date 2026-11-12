import numpy as np
import heapq
import itertools
import time
from collections import Counter

D = ((0, -1), (1, 0), (0, 1), (-1, 0))
V, W, O, MAXN = 21, 69, 24, 21


def components(points):
    remaining = set(points)
    result = []
    while remaining:
        q = remaining.pop()
        stack, part = [q], [q]
        while stack:
            x, y = stack.pop()
            for p in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.append(p)
        result.append(part)
    return result


def bit(x, y):
    return 1 << ((y + O) * W + x + O)


def rectangle_bits(x, y, d, n):
    dx, dy = D[d]
    ex, ey = x + dx * (n - 1), y + dy * (n - 1)
    x0, y0 = min(x, ex), min(y, ey)
    if (n < 1 or n > MAXN or x0 < -O or y0 < -O
            or max(x, ex) >= W - O or max(y, ey) >= W - O):
        return None
    if dy == 0:
        value = (1 << n) - 1
    else:
        value = VB[n]
    return value << ((y0 + O) * W + x0 + O)


VB = [0]
for _n in range(1, MAXN + 1):
    VB.append(VB[-1] | (1 << ((_n - 1) * W)))


def read_image(obs):
    a = np.asarray(obs)
    ui = np.zeros((64, 64), dtype=bool)
    buttons = []
    for part in components((int(x), int(y))
                           for y, x in np.argwhere(a == 2)):
        xs, ys = zip(*part)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if min(w, h) < 5:
            continue
        box = a[y0:y1 + 1, x0:x1 + 1]
        if not (np.all(box[0] == 2) and np.all(box[-1] == 2)
                and np.all(box[:, 0] == 2)
                and np.all(box[:, -1] == 2)):
            continue
        vals = box[(box != 2) & (box != 3) & (box != 4)]
        if not vals.size:
            continue
        c = Counter(map(int, vals)).most_common(1)[0][0]
        ui[y0:y1 + 1, x0:x1 + 1] = True
        if w == h:
            buttons.append((c, 0, y0 + h // 2, x0 + w // 2))
        elif w > h:
            buttons.extend(((c, -1, y0 + h // 2, x0 + 1),
                            (c, 1, y0 + h // 2, x1 - 1)))
        else:
            buttons.extend(((c, -1, y0 + 1, x0 + w // 2),
                            (c, 1, y1 - 1, x0 + w // 2)))

    diamonds = ((a[:-2, 1:-1] == 13)
                & (a[2:, 1:-1] == 13)
                & (a[1:-1, :-2] == 13)
                & (a[1:-1, 2:] == 13))
    gp = [(int(x), int(y)) for y, x in np.argwhere(diamonds)]
    if gp:
        ox, oy = Counter((x % 3, y % 3)
                         for x, y in gp).most_common(1)[0][0]
    else:
        ox = oy = 0
    goals = {((x - ox) // 3, (y - oy) // 3)
             for x, y in gp
             if (x - ox) % 3 == 0 and (y - oy) % 3 == 0}
    marks = set()
    for y, x in np.argwhere(a == 13):
        x, y = int(x), int(y)
        if ((x - ox - 1) % 3 == 0 and (y - oy - 1) % 3 == 0
                and not ui[y, x]):
            marks.add(((x - ox - 1) // 3, (y - oy - 1) // 3))

    colors = {b[0] for b in buttons}
    colors.update((0, 6, 7, 8, 9, 10, 11, 12, 14))
    colors.difference_update((1, 2, 3, 4, 5, 13, 15))
    cells, stripes, skipped, white = {}, {}, set(), set()
    for y in range(V):
        for x in range(V):
            px, py = ox + x * 3, oy + y * 3
            if px < 0 or py < 0 or px + 3 > 64 or py + 3 > 63:
                skipped.add((x, y))
                continue
            if np.any(ui[py:py + 3, px:px + 3]):
                skipped.add((x, y))
                continue
            b = a[py:py + 3, px:px + 3]
            if np.any(b == 15):
                white.add((x, y))
            if np.any(b == 1) and np.all((b == 1) | (b == 13)):
                cells[(x, y)] = 1
                continue
            cs = [int(c) for c in np.unique(b) if int(c) in colors]
            if len(cs) != 1:
                continue
            c = cs[0]
            if not np.all((b == c) | (b == 3) | (b == 13)):
                continue
            cells[(x, y)] = c
            for d, side in enumerate((b[-1, :], b[:, 0],
                                      b[0, :], b[:, -1])):
                if np.any(side == 3) and np.all((side == 3) | (side == 13)):
                    stripes[(x, y)] = d
                    break

    walls = 0
    for x, y in white:
        walls |= bit(x, y)

    # Continue an identifiable enclosing frame just beyond the image.
    for x in range(V):
        if not all((x, y) in white for y in range(3)):
            continue
        for side in (-1, 1):
            span = range(x, V) if side == 1 else range(0, x + 1)
            if len(span) < 4:
                continue
            bottom = next((y for y in range(3, V)
                           if all((xx, y) in white for xx in span)), None)
            if bottom is None:
                continue
            for xx in span:
                walls |= bit(xx, -1)
            edge = V if side == 1 else -1
            for yy in range(-1, bottom + 1):
                walls |= bit(edge, yy)
            break

    return {
        "cells": cells, "stripes": stripes, "buttons": buttons,
        "goals": goals, "marks": marks, "walls": walls,
        "skipped": skipped, "offset": (ox, oy),
        "signature": (tuple(sorted((x, y, c)
                                  for (x, y), c in cells.items())),
                      tuple(sorted(marks)))
    }


def parse(info):
    cells, stripes = info["cells"], info["stripes"]
    nodes, used = [], set()
    for anchor, d in sorted(stripes.items()):
        if anchor in used:
            continue
        c = cells[anchor]
        dx, dy = D[d]
        x, y = anchor
        run = []
        while cells.get((x, y)) == c:
            if (run and (x, y) in stripes) or (x, y) in used:
                break
            run.append((x, y))
            x, y = x + dx, y + dy
        if run:
            used.update(run)
            nodes.append({"c": c, "a": anchor, "n": len(run),
                          "d": d, "p": -1, "blue": False})

    fixed = []
    for part in components(q for q, c in cells.items() if c == 1):
        xs, ys = zip(*part)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if (len(part) == (x1 - x0 + 1) * (y1 - y0 + 1)
                and (x0 == x1 or y0 == y1)):
            d = 2 if x0 == x1 and y0 != y1 else 1
            nodes.append({"c": 1, "a": (x0, y0),
                          "n": max(x1 - x0 + 1, y1 - y0 + 1),
                          "d": d, "p": -1, "blue": True})
        else:
            fixed.extend(part)
    if not nodes:
        return None

    tips = {}
    for i, node in enumerate(nodes):
        if not node["blue"]:
            x, y = node["a"]
            dx, dy = D[node["d"]]
            tips.setdefault((x + dx * node["n"],
                             y + dy * node["n"]), []).append(i)
    for i, node in enumerate(nodes):
        if not node["blue"]:
            ps = [p for p in tips.get(node["a"], ()) if p != i]
            if ps:
                node["p"] = ps[0]
        else:
            x, y = node["a"]
            dx, dy = D[node["d"]]
            end = (x + dx * (node["n"] - 1),
                   y + dy * (node["n"] - 1))
            if node["a"] in tips:
                node["p"] = tips[node["a"]][0]
            elif end in tips:
                node["p"] = tips[end][0]
                node["a"], node["d"] = end, (node["d"] + 2) % 4

    for i in range(len(nodes)):
        seen, j = set(), i
        while j >= 0:
            if j in seen:
                nodes[i]["p"] = -1
                break
            seen.add(j)
            j = nodes[j]["p"]

    topo, done = [], set()
    while len(topo) < len(nodes):
        before = len(done)
        for i, node in enumerate(nodes):
            if i not in done and (node["p"] < 0 or node["p"] in done):
                topo.append(i)
                done.add(i)
        if len(done) == before:
            return None

    state = []
    for node in nodes:
        p = node["p"]
        r = node["d"] if p < 0 else (node["d"] - nodes[p]["d"]) % 4
        state.extend((node["n"], r))

    attachments = []
    for mx, my in sorted(info["marks"]):
        found = None
        for i, node in enumerate(nodes):
            ax, ay = node["a"]
            dx, dy = D[node["d"]]
            along = (mx - ax) * dx + (my - ay) * dy
            lateral = -(mx - ax) * dy + (my - ay) * dx
            if lateral == 0 and 0 <= along < node["n"]:
                found = (i, along - node["n"] + 1, lateral)
                break
        attachments.append(found if found is not None else (-1, mx, my))

    actions, lg, rg = [], {}, {}
    for c, kind, row, col in info["buttons"]:
        ids = tuple(i for i, node in enumerate(nodes) if node["c"] == c)
        if ids:
            actions.append((kind, ids, row, col))
            (lg if kind else rg)[c] = ids
    walls = info["walls"]
    for x, y in fixed:
        walls |= bit(x, y)
    return {
        "nodes": nodes, "topo": topo, "attachments": attachments,
        "goals": info["goals"], "actions": actions,
        "lg": list(lg.values()), "rg": list(rg.values()),
        "walls": walls, "fixed": fixed,
        "skipped": info["skipped"], "offset": info["offset"],
        "state": tuple(state)
    }


def geometry(m, s, collision=True):
    pos = [None] * len(m["nodes"])
    dirs = [0] * len(pos)
    occupied = m["walls"]
    for i in m["topo"]:
        node = m["nodes"][i]
        n, r = s[2 * i:2 * i + 2]
        p = node["p"]
        if p < 0:
            x, y = node["a"]
            d = r
        else:
            px, py = pos[p]
            pd = dirs[p]
            dx, dy = D[pd]
            x, y = px + dx * s[2 * p], py + dy * s[2 * p]
            d = (pd + r) % 4
        pos[i], dirs[i] = (x, y), d
        if collision:
            bits = rectangle_bits(x, y, d, n)
            if bits is None or bits & occupied:
                return None
            occupied |= bits

    marks = []
    for i, a, b in m["attachments"]:
        if i < 0:
            marks.append((a, b))
        else:
            x, y = pos[i]
            dx, dy = D[dirs[i]]
            n = s[2 * i]
            marks.append((x + dx * (n - 1 + a) - dy * b,
                          y + dy * (n - 1 + a) + dx * b))
    return pos, dirs, marks


def signature(m, s):
    pos, dirs, marks = geometry(m, s, False)
    cells = []
    skip = m["skipped"]
    for i, node in enumerate(m["nodes"]):
        x, y = pos[i]
        dx, dy = D[dirs[i]]
        for k in range(s[2 * i]):
            q = (x + dx * k, y + dy * k)
            if 0 <= q[0] < V and 0 <= q[1] < V and q not in skip:
                cells.append((q[0], q[1], node["c"]))
    for q in m["fixed"]:
        if q not in skip:
            cells.append((q[0], q[1], 1))
    ox, oy = m["offset"]
    visible = [(x, y) for x, y in marks
               if 0 <= ox + 3 * x + 1 < 64
               and 0 <= oy + 3 * y + 1 < 63
               and (x, y) not in skip]
    return tuple(sorted(cells)), tuple(sorted(visible))


def rotate(r, parent):
    nr = (r - 1) % 4
    return (nr - 1) % 4 if parent and nr == 2 else nr


def change(m, s, ai):
    kind, ids, _, _ = m["actions"][ai]
    t = list(s)
    if kind:
        for i in ids:
            t[2 * i] = max(1, t[2 * i] + kind)
            if t[2 * i] > MAXN:
                return None
    else:
        for i in ids:
            t[2 * i + 1] = rotate(t[2 * i + 1], m["nodes"][i]["p"] >= 0)
    t = tuple(t)
    return t if t != s else None


def predecessors(m, s, ai):
    kind, ids, _, _ = m["actions"][ai]
    options = []
    for i in ids:
        if kind == 1:
            n = s[2 * i]
            options.append([n - 1] if n > 1 else [])
        elif kind == -1:
            n = s[2 * i]
            options.append([1, 2] if n == 1 else
                           ([n + 1] if n < MAXN else []))
        else:
            r = s[2 * i + 1]
            parent = m["nodes"][i]["p"] >= 0
            options.append([q for q in range(4) if rotate(q, parent) == r])
    if any(not q for q in options):
        return
    for values in itertools.product(*options):
        t = list(s)
        for i, v in zip(ids, values):
            t[2 * i + (0 if kind else 1)] = v
        t = tuple(t)
        if t != s:
            yield t


def rot_variants(m, s, ids):
    q = tuple(s[2 * i + 1] for i in ids)
    seen, result = set(), []
    count = 0
    while q not in seen and count < 13:
        seen.add(q)
        result.append((q, count))
        q = tuple(rotate(r, m["nodes"][i]["p"] >= 0)
                  for i, r in zip(ids, q))
        count += 1
    return result


def reduction(A, b, lower, upper, current):
    n, count = len(lower), len(A)
    rows = [list(map(float, row)) + [float(v)] for row, v in zip(A, b)]
    pivots, rr = [], 0
    for c in range(n):
        p = next((j for j in range(rr, count)
                  if abs(rows[j][c]) > 1e-8), None)
        if p is None:
            continue
        rows[p], rows[rr] = rows[rr], rows[p]
        z = rows[rr][c]
        rows[rr] = [v / z for v in rows[rr]]
        for j in range(count):
            if j != rr and abs(rows[j][c]) > 1e-8:
                z = rows[j][c]
                rows[j] = [v - z * w for v, w in zip(rows[j], rows[rr])]
        pivots.append(c)
        rr += 1
        if rr == count:
            break
    for row in rows[rr:]:
        if all(abs(v) < 1e-7 for v in row[:n]) and abs(row[-1]) > 1e-7:
            return None
    free = [j for j in range(n) if j not in pivots]
    expr, lb = [], 0.0
    for k, p in enumerate(pivots):
        co = [-rows[k][j] for j in free]
        const = rows[k][-1]
        lo = hi = const
        for c, j in zip(co, free):
            lo += c * (lower[j] if c >= 0 else upper[j])
            hi += c * (upper[j] if c >= 0 else lower[j])
        lo, hi = max(lo, lower[p]), min(hi, upper[p])
        if lo > hi + 1e-7:
            return None
        lb += max(lo - current[p], current[p] - hi, 0)
        expr.append((p, const, co))
    return free, expr, lb


def goal_states(m, s, deadline, broad=False):
    groups = m["lg"]
    k = len(groups)
    current = [s[2 * ids[0]] for ids in groups]
    lower, upper, var, offset = [], [], {}, {}
    for j, ids in enumerate(groups):
        ds = [s[2 * i] - current[j] for i in ids]
        lower.append(1 - min(ds))
        upper.append(MAXN - max(ds))
        for i, v in zip(ids, ds):
            var[i], offset[i] = j, v

    variants = [rot_variants(m, s, ids) for ids in m["rg"]]
    combos = itertools.product(*variants) if variants else [()]
    goals = sorted(m["goals"])
    count = len(m["attachments"])
    if not goals or count < len(goals):
        return []
    assignments = list(itertools.islice(
        itertools.permutations(range(count), len(goals)), 24))
    branches, heap, serial = [], [], 0

    for combo in combos:
        if time.perf_counter() >= deadline:
            break
        rs = [s[2 * i + 1] for i in range(len(m["nodes"]))]
        rc = 0
        for ids, (values, clicks) in zip(m["rg"], combo):
            rc += clicks
            for i, r in zip(ids, values):
                rs[i] = r
        dirs = [0] * len(rs)
        ex, ey = [None] * len(rs), [None] * len(rs)
        for i in m["topo"]:
            node, p = m["nodes"][i], m["nodes"][i]["p"]
            if p < 0:
                dirs[i] = rs[i]
                ex[i] = [float(node["a"][0])] + [0.0] * k
                ey[i] = [float(node["a"][1])] + [0.0] * k
            else:
                dirs[i] = (dirs[p] + rs[i]) % 4
                ex[i], ey[i] = ex[p][:], ey[p][:]
                dx, dy = D[dirs[p]]
                if p in var:
                    j = var[p]
                    ex[i][j + 1] += dx
                    ey[i][j + 1] += dy
                    ex[i][0] += dx * offset[p]
                    ey[i][0] += dy * offset[p]
                else:
                    ex[i][0] += dx * s[2 * p]
                    ey[i][0] += dy * s[2 * p]

        expressions = []
        for i, a, b in m["attachments"]:
            if i < 0:
                expressions.append(([float(a)] + [0.0] * k,
                                    [float(b)] + [0.0] * k))
                continue
            xx, yy = ex[i][:], ey[i][:]
            dx, dy = D[dirs[i]]
            const = a - 1
            if i in var:
                j = var[i]
                xx[j + 1] += dx
                yy[j + 1] += dy
                const += offset[i]
            else:
                const += s[2 * i]
            xx[0] += dx * const - dy * b
            yy[0] += dy * const + dx * b
            expressions.append((xx, yy))

        for assignment in assignments:
            A, bvec = [], []
            for g, mi in zip(goals, assignment):
                xx, yy = expressions[mi]
                A.extend((xx[1:], yy[1:]))
                bvec.extend((g[0] - xx[0], g[1] - yy[0]))
            red = reduction(A, bvec, lower, upper, current)
            if red is None:
                continue
            free, expr, plb = red
            values = [sorted(range(lower[j], upper[j] + 1),
                             key=lambda v, j=j: (abs(v - current[j]), v))
                      for j in free]
            if any(not vs for vs in values):
                continue
            residual = sum(abs(bb - sum(c * v for c, v in zip(row, current)))
                           for row, bb in zip(A, bvec))
            capacity = max([sum(abs(A[r][j]) for r in range(len(A)))
                            for j in range(k)] + [1])
            elb = residual / capacity
            indices = (0,) * len(free)
            branches.append((rs[:], rc, free, expr, values, plb, elb,
                             {indices}))
            heapq.heappush(heap, (rc + max(plb, elb), serial,
                                  len(branches) - 1, indices, 0))
            serial += 1

    candidates = {}
    best = float("inf")
    iterations = 0
    while heap and time.perf_counter() < deadline:
        lb, _, bi, indices, fc = heapq.heappop(heap)
        if not broad and best < float("inf") and lb > best + 9:
            break
        rs, rc, free, expr, values, plb, elb, seen = branches[bi]
        vals = [values[j][indices[j]] for j in range(len(free))]
        lengths = current[:]
        for j, v in zip(free, vals):
            lengths[j] = v
        valid = True
        for p, const, co in expr:
            v = const + sum(c * z for c, z in zip(co, vals))
            iv = int(round(v))
            if abs(v - iv) > 1e-6 or not lower[p] <= iv <= upper[p]:
                valid = False
                break
            lengths[p] = iv
        if valid:
            t = list(s)
            for i in range(len(m["nodes"])):
                t[2 * i + 1] = rs[i]
                if i in var:
                    t[2 * i] = lengths[var[i]] + offset[i]
            t = tuple(t)
            geom = geometry(m, t)
            if geom is not None and m["goals"].issubset(set(geom[2])):
                cost = rc + sum(abs(a - b) for a, b in zip(lengths, current))
                candidates[t] = cost
                best = min(best, cost)
                if not broad and len(candidates) > 64:
                    keys = sorted(candidates, key=candidates.get)[:40]
                    candidates = {q: candidates[q] for q in keys}
        for j in range(len(free)):
            if indices[j] + 1 >= len(values[j]):
                continue
            ni = list(indices)
            ni[j] += 1
            ni = tuple(ni)
            if ni in seen:
                continue
            seen.add(ni)
            old, new = values[j][indices[j]], values[j][ni[j]]
            nc = fc - abs(old - current[free[j]]) + abs(new - current[free[j]])
            heapq.heappush(heap, (rc + max(nc + plb, elb),
                                  serial, bi, ni, nc))
            serial += 1
        iterations += 1
        if iterations >= (240000 if broad else 90000):
            break

    ordered = sorted(candidates, key=candidates.get)
    if broad:
        return ordered
    result, families = [], set()
    for t in ordered:
        family = tuple(t[2 * i + 1] for i in range(len(m["nodes"])))
        if family not in families:
            families.add(family)
            result.append(t)
    for t in ordered:
        if t not in result:
            result.append(t)
        if len(result) >= 24:
            break
    return result[:24]


def rdist(a, b, parent):
    for n in range(9):
        if a == b:
            return n
        a = rotate(a, parent)
    return 9


RD = {(p, a, b): rdist(a, b, p)
      for p in (False, True) for a in range(4) for b in range(4)}


def distance(m, s, t):
    value = 0
    for ids in m["lg"]:
        value += max(abs(s[2 * i] - t[2 * i]) for i in ids)
    for ids in m["rg"]:
        value += max(RD[(m["nodes"][i]["p"] >= 0,
                        s[2 * i + 1], t[2 * i + 1])] for i in ids)
    return value


def raw_distance(m, marks):
    goals = list(m["goals"])
    if not marks:
        return 100
    if len(goals) <= len(marks) <= 4:
        return min(sum(abs(marks[i][0] - g[0]) + abs(marks[i][1] - g[1])
                       for i, g in zip(order, goals))
                   for order in itertools.permutations(range(len(marks)),
                                                       len(goals)))
    return sum(min(abs(x - gx) + abs(y - gy) for x, y in marks)
               for gx, gy in goals)


def chain_heuristic(m, targets):
    if len(m["attachments"]) != 1 or len(m["goals"]) != 1 or not targets:
        return None
    families = {}
    for t in targets:
        key = tuple(t[2 * i + 1] for i in range(len(m["nodes"])))
        families.setdefault(key, t)
    reps = list(families.values())
    arrays, constants, caps = [], [], []
    goal = np.asarray(next(iter(m["goals"])), dtype=float)
    for t in reps:
        base = list(t)
        for ids in m["lg"]:
            for i in ids:
                base[2 * i] -= t[2 * ids[0]] - 1
        p0 = np.asarray(geometry(m, tuple(base), False)[2][0], dtype=float)
        cols = []
        for ids in m["lg"]:
            q = base[:]
            for i in ids:
                q[2 * i] += 1
            p1 = np.asarray(geometry(m, tuple(q), False)[2][0], dtype=float)
            cols.append(p1 - p0)
        A = np.asarray(cols, dtype=float).T
        if not cols:
            A = np.zeros((2, 0))
        arrays.append(A)
        constants.append(p0 - A.dot(np.ones(len(cols))) - goal)
        caps.append(max(1.0, max(
            (float(np.abs(A[:, j]).sum()) for j in range(len(cols))),
            default=1.0)))
    AA = np.asarray(arrays)
    CC = np.asarray(constants)
    CAP = np.asarray(caps)
    rot_cache = {}

    def heuristic(s):
        key = tuple(s[2 * i + 1] for i in range(len(m["nodes"])))
        costs = rot_cache.get(key)
        if costs is None:
            costs = np.asarray([
                sum(max(RD[(m["nodes"][i]["p"] >= 0,
                            s[2 * i + 1], t[2 * i + 1])]
                        for i in ids) for ids in m["rg"])
                for t in reps], dtype=float)
            rot_cache[key] = costs
        lengths = np.asarray([s[2 * ids[0]] for ids in m["lg"]], dtype=float)
        residual = np.abs(np.matmul(AA, lengths) + CC).sum(axis=1) / CAP
        return float(np.min(costs + residual))
    return heuristic


def search(m, start, targets, deadline, blocked, visits, broad=False):
    initial = geometry(m, start)
    if initial is None:
        return [], False
    if m["goals"].issubset(set(initial[2])):
        return [], True
    targets = [t for t in targets if geometry(m, t) is not None]
    special_h = chain_heuristic(m, targets) if broad else None

    forward_cost = {start: 0}
    backward_cost = {t: 0 for t in targets}
    fp, bp, hf, hb = {}, {}, {}, {}
    serial = 0

    def fh(s, geom=None):
        if s not in hf:
            if special_h is not None:
                hf[s] = special_h(s)
            elif targets:
                hf[s] = min(distance(m, s, t) for t in targets)
            else:
                if geom is None:
                    geom = geometry(m, s, False)
                hf[s] = raw_distance(m, geom[2]) * (
                    0.4 if m["rg"] else 1.0)
        return hf[s]

    def bh(s):
        if s not in hb:
            hb[s] = distance(m, start, s)
        return hb[s]

    weight = 1.8 if broad else 2.4
    h0 = fh(start, initial)
    forward = [(weight * h0, h0, serial, start, 0)]
    backward = []
    for t in targets:
        serial += 1
        h = bh(t)
        backward.append((weight * h, h, serial, t, 0))
    heapq.heapify(backward)

    best, best_rank = start, (h0, 0)
    meet = None
    expansions, turn = 0, 0
    while forward and time.perf_counter() < deadline:
        use_back = bool(backward) and turn % 2 == 1
        turn += 1
        heap = backward if use_back else forward
        costs = backward_cost if use_back else forward_cost
        other = forward_cost if use_back else backward_cost
        parents = bp if use_back else fp
        _, hs, _, state, g = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        if state in other:
            meet = state
            break
        geom = geometry(m, state)
        if geom is None:
            continue
        if not use_back:
            if m["goals"].issubset(set(geom[2])):
                backward_cost[state] = 0
                meet = state
                break
            rank = (hs + min(10, visits.get(state, 0)) * 0.8, g)
            if state != start and rank < best_rank:
                best, best_rank = state, rank

        for ai in range(len(m["actions"])):
            if use_back:
                choices = predecessors(m, state, ai)
            else:
                if (state, ai) in blocked:
                    continue
                q = change(m, state, ai)
                choices = () if q is None else (q,)
            for nxt in choices:
                if use_back and (nxt, ai) in blocked:
                    continue
                ng = g + 1
                if ng >= costs.get(nxt, 10**9):
                    continue
                newgeom = geometry(m, nxt)
                if newgeom is None:
                    continue
                costs[nxt] = ng
                parents[nxt] = (state, ai)
                if nxt in other:
                    meet = nxt
                    break
                if not use_back and m["goals"].issubset(set(newgeom[2])):
                    backward_cost[nxt] = 0
                    meet = nxt
                    break
                nh = bh(nxt) if use_back else fh(nxt, newgeom)
                penalty = 0 if use_back else min(10, visits.get(nxt, 0)) * 0.2
                serial += 1
                heapq.heappush(heap, (ng + weight * nh + penalty,
                                      nh, serial, nxt, ng))
            if meet is not None:
                break
        if meet is not None:
            break
        expansions += 1
        if expansions >= 220000:
            break

    solved = meet is not None
    finish = meet if solved else best
    plan = []
    q = finish
    while q != start and q in fp:
        q, ai = fp[q]
        plan.append(ai)
    plan.reverse()
    if solved:
        q = finish
        while q in bp:
            q, ai = bp[q]
            plan.append(ai)
    return plan, solved


def control_components(m):
    parent = list(range(len(m["nodes"])))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a

    for i, n in enumerate(m["nodes"]):
        if n["p"] >= 0:
            join(i, n["p"])
    for _, ids, _, _ in m["actions"]:
        for i in ids[1:]:
            join(ids[0], i)
    groups = {}
    for i in range(len(parent)):
        groups.setdefault(root(i), []).append(i)
    return [tuple(ids) for ids in groups.values()]


def submodel(m, state, ids, goals):
    ids = tuple(sorted(ids))
    index = {old: new for new, old in enumerate(ids)}
    nodes = []
    for old in ids:
        n = dict(m["nodes"][old])
        n["p"] = index.get(n["p"], -1)
        nodes.append(n)
    actions, action_map = [], []
    for ai, (kind, group, row, col) in enumerate(m["actions"]):
        if group and group[0] in index:
            actions.append((kind, tuple(index[i] for i in group), row, col))
            action_map.append(ai)
    walls = m["walls"]
    pos, dirs, _ = geometry(m, state, False)
    for i in range(len(m["nodes"])):
        if i not in index:
            value = rectangle_bits(*pos[i], dirs[i], state[2 * i])
            if value is not None:
                walls |= value
    sm = {
        "nodes": nodes,
        "topo": [index[i] for i in m["topo"] if i in index],
        "attachments": [(index[i], a, b) for i, a, b in m["attachments"]
                        if i in index],
        "goals": set(goals), "actions": actions,
        "lg": [tuple(index[i] for i in group)
               for group in m["lg"] if group[0] in index],
        "rg": [tuple(index[i] for i in group)
               for group in m["rg"] if group[0] in index],
        "walls": walls
    }
    ss = tuple(v for i in ids for v in state[2 * i:2 * i + 2])
    return sm, ss, action_map


def independent_plan(m, state, memory, deadline):
    groups = control_components(m)
    large = [ids for ids in groups
             if len(ids) >= 3
             and sum(i in ids for i, _, _ in m["attachments"]) == 1]
    if not large:
        return None

    _, _, marks = geometry(m, state, False)
    goalset = m["goals"]

    # First restore simple independent marked bars if necessary.
    for ids in groups:
        attached = [j for j, (i, _, _) in enumerate(m["attachments"]) if i in ids]
        if len(ids) != 1 or len(attached) != 1:
            continue
        if marks[attached[0]] in goalset:
            continue
        occupied_goals = {p for j, p in enumerate(marks)
                          if j != attached[0] and p in goalset}
        for goal in sorted(goalset - occupied_goals):
            sm, ss, amap = submodel(m, state, ids, {goal})
            targets = goal_states(sm, ss, min(deadline, time.perf_counter() + 0.15))
            if not targets:
                continue
            plan, solved = search(sm, ss, targets,
                                  min(deadline, time.perf_counter() + 0.25),
                                  set(), {}, False)
            if solved and plan:
                return [amap[a] for a in plan], True

    for ids in large:
        attached = [j for j, (i, _, _) in enumerate(m["attachments"]) if i in ids]
        outside = [p for j, p in enumerate(marks) if j not in attached]
        if any(p not in goalset for p in outside):
            continue
        remaining = goalset - set(outside)
        if len(remaining) != 1:
            continue
        if marks[attached[0]] in remaining:
            continue

        sm, ss, amap = submodel(m, state, ids, remaining)
        outside_state = tuple(state[2 * i:2 * i + 2]
                              for i in range(len(m["nodes"])) if i not in ids)
        key = (ids, outside_state, tuple(sorted(remaining)), m["walls"])
        cache = memory.get("chain_cache")
        if cache is None or cache["key"] != key:
            targets = goal_states(
                sm, ss, min(deadline, time.perf_counter() + 2.7), broad=True)
            cache = {"key": key, "targets": targets, "visits": {},
                     "blocked": set()}
            memory["chain_cache"] = cache
        targets = cache["targets"]
        if not targets:
            continue

        cache["visits"][ss] = cache["visits"].get(ss, 0) + 1
        for full_state, ai in memory.get("blocked", set()):
            if ai in amap:
                sub_state = tuple(v for i in sorted(ids)
                                  for v in full_state[2 * i:2 * i + 2])
                other = tuple(full_state[2 * i:2 * i + 2]
                              for i in range(len(m["nodes"])) if i not in ids)
                if other == outside_state:
                    cache["blocked"].add((sub_state, amap.index(ai)))

        plan, solved = search(sm, ss, targets, deadline,
                              cache["blocked"], cache["visits"], broad=True)
        if plan:
            return [amap[a] for a in (plan if solved else plan[:6])], solved
    return None


def learn_wall(m, old, new):
    def extents(s):
        pos, dirs, _ = geometry(m, s, False)
        out = []
        for i, (x, y) in enumerate(pos):
            dx, dy = D[dirs[i]]
            ex, ey = x + dx * (s[2 * i] - 1), y + dy * (s[2 * i] - 1)
            out.append((min(x, ex), max(x, ex), min(y, ey), max(y, ey)))
        return out

    a, b = extents(old), extents(new)
    crossings = []
    if min(q[0] for q in a) >= 0 and min(q[0] for q in b) < 0:
        crossings.append(0)
    if max(q[1] for q in a) < V and max(q[1] for q in b) >= V:
        crossings.append(1)
    if min(q[2] for q in a) >= 0 and min(q[2] for q in b) < 0:
        crossings.append(2)
    if max(q[3] for q in a) < V and max(q[3] for q in b) >= V:
        crossings.append(3)
    if len(crossings) != 1:
        return False
    side, added = crossings[0], 0
    for x0, x1, y0, y1 in b:
        if side == 0 and x0 < 0:
            for y in range(y0, y1 + 1):
                added |= bit(-1, y)
        elif side == 1 and x1 >= V:
            for y in range(y0, y1 + 1):
                added |= bit(V, y)
        elif side == 2 and y0 < 0:
            for x in range(x0, x1 + 1):
                added |= bit(x, -1)
        elif side == 3 and y1 >= V:
            for x in range(x0, x1 + 1):
                added |= bit(x, V)
    m["walls"] |= added
    return bool(added)


def act(obs, available, memory):
    started = time.perf_counter()
    available = set(available)
    info = read_image(obs)
    m, state = memory.get("model"), memory.get("state")

    if m is not None and state is not None:
        if signature(m, state) != info["signature"]:
            previous = memory.get("previous")
            if previous is not None and signature(m, previous) == info["signature"]:
                ai = memory.get("last_action", -1)
                memory.setdefault("blocked", set()).add((previous, ai))
                if learn_wall(m, previous, state):
                    memory["targets"] = []
                    memory.pop("chain_cache", None)
                state = previous
                memory["state"] = state
                memory["plan"] = []
            else:
                memory.clear()
                m = None

    if m is None:
        m = parse(info)
        if m is None:
            if "ACTION6" in available:
                if info["buttons"]:
                    _, _, row, col = info["buttons"][0]
                    return "ACTION6", int(row), int(col)
                return "ACTION6", 0, 0
            name = "RESET" if "RESET" in available else next(iter(available))
            return name, None, None
        state = m["state"]
        memory.update({"model": m, "state": state, "plan": [],
                       "targets": [], "blocked": set(), "visits": {}})

    visits = memory.setdefault("visits", {})
    visits[state] = visits.get(state, 0) + 1
    blocked = memory.setdefault("blocked", set())
    plan = memory.get("plan", [])
    if plan and (state, plan[0]) in blocked:
        plan = []
        memory["plan"] = []

    if not plan:
        independent = independent_plan(m, state, memory, started + 9.05)
        if independent is not None:
            memory["plan"] = independent[0]
        elif time.perf_counter() < started + 8.8:
            targets = memory.get("targets", [])
            if not targets or visits[state] >= 3:
                targets = goal_states(
                    m, state, min(started + 2.8, time.perf_counter() + 2.5))
                memory["targets"] = targets
            plan, solved = search(
                m, state, targets, started + 9.15, blocked, visits)
            memory["plan"] = plan if solved else plan[:5]

    if not memory.get("plan"):
        choices = []
        targets = memory.get("targets", [])
        for ai in range(len(m["actions"])):
            if (state, ai) in blocked:
                continue
            nxt = change(m, state, ai)
            if nxt is None:
                continue
            geom = geometry(m, nxt)
            if geom is None:
                continue
            score = (min(distance(m, nxt, t) for t in targets)
                     if targets else raw_distance(m, geom[2]))
            score += 4 * visits.get(nxt, 0)
            choices.append((score, ai))
        if choices:
            memory["plan"] = [min(choices)[1]]
        elif "RESET" in available:
            memory.clear()
            return "RESET", None, None
        else:
            return "ACTION6", 0, 0

    ai = memory["plan"].pop(0)
    nxt = change(m, state, ai)
    memory["previous"] = state
    memory["state"] = nxt if nxt is not None else state
    memory["last_action"] = ai
    _, _, row, col = m["actions"][ai]
    return "ACTION6", int(row), int(col)

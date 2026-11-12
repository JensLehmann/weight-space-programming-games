import numpy as np
import math
import time
import heapq
import itertools
from collections import Counter
from functools import lru_cache

# Coordinates throughout the program are (x, y).
DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
           if abs(x) + abs(y) <= 2]
DBODY = [p for p in DIAMOND if p != (0, 0)]
BALL = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
        if not (abs(x) == 2 and abs(y) == 2)]
RING = [(-1, -3), (1, -3), (-2, -2), (2, -2),
        (-3, -1), (3, -1), (-3, 1), (3, 1),
        (-2, 2), (2, 2), (-1, 3), (1, 3)]
FILLED_RING = [(x, y) for y in range(-3, 4) for x in range(-3, 4)
               if abs(x) <= (1 if abs(y) == 3 else
                             2 if abs(y) == 2 else 3)]
BALL_DIFF = set((a - c, b - d) for a, b in BALL for c, d in BALL)
GOAL_DIFF = set((a - c, b - d)
                for a, b in FILLED_RING for c, d in BALL)
BAD_OBJECT = {1, 2, 3, 5, 10}
YY, XX = np.indices((64, 64))


def _at(im, x, y, offsets):
    return [int(im[y + dy, x + dx]) for dx, dy in offsets
            if 0 <= x + dx < 64 and 0 <= y + dy < 64]


def _center(A, ids):
    n = len(ids)
    return (sum(A[i][0] for i in ids) // n,
            sum(A[i][1] for i in ids) // n)


def _paint_set(pix):
    return frozenset(int(c) for c in pix if c > 0)


def _zone(x, y, differences):
    z = np.zeros((64, 64), dtype=bool)
    for dx, dy in differences:
        a, b = x + dx, y + dy
        if 0 <= a < 64 and 0 <= b < 64:
            z[b, a] = True
    return z


def _collision_map(terrain, offsets):
    out = np.zeros((64, 64), dtype=bool)
    for dx, dy in offsets:
        x0, x1 = max(0, -dx), min(64, 64 - dx)
        y0, y1 = max(0, -dy), min(64, 64 - dy)
        out[y0:y1, x0:x1] |= terrain[y0 + dy:y1 + dy,
                                      x0 + dx:x1 + dx]
    return out


def _line_support(im, a, b):
    x, y = a
    tx, ty = b
    dx, dy = abs(tx - x), abs(ty - y)
    sx, sy = (1 if x < tx else -1), (1 if y < ty else -1)
    err = dx - dy
    yes = total = 0
    while True:
        if 0 <= x < 64 and 0 <= y < 64:
            v = int(im[y, x])
            if v in (1, 5, 10):
                total += 1
                yes += (v == 1)
        if x == tx and y == ty:
            break
        e = 2 * err
        if e > -dy:
            err -= dy
            x += sx
        if e < dx:
            err += dx
            y += sy
    return (yes + 0.5) / (total + 0.5) if total else 0.75


def _vision(im):
    anchors = []
    possible = np.argwhere(~np.isin(im, [0, 1, 2, 3, 5, 6, 10]))
    for y, x in possible:
        x, y = int(x), int(y)
        vals = _at(im, x, y, DBODY)
        if len(vals) < 5:
            continue
        good = sum(v in (0, 3) for v in vals)
        if good < max(5, len(vals) - 2):
            continue
        outside = [(dx, dy) for dy in range(-2, 3)
                   for dx in range(-2, 3)
                   if abs(dx) + abs(dy) == 3]
        ext = _at(im, x, y, outside)
        if sum(v in (0, 3) for v in ext) > 3:
            continue
        score = good - 0.3 * sum(v in (0, 3) for v in ext)
        selected = vals.count(0) > vals.count(3)
        anchors.append((score, x, y, int(im[y, x]), selected))

    anchors.sort(reverse=True)
    accepted = []
    for a in anchors:
        if not any(max(abs(a[1] - b[1]), abs(a[2] - b[2])) < 3
                   for b in accepted):
            accepted.append(a)
    accepted.sort(key=lambda a: a[1] ** 2 + a[2] ** 2)
    A = [[a[1], a[2], a[3]] for a in accepted]
    selected = next((i for i, a in enumerate(accepted) if a[4]), 0)

    circles = []
    for y in range(2, 62):
        for x in range(2, 62):
            if int(im[y, x]) in BAD_OBJECT:
                continue
            pix = _at(im, x, y, BALL)
            good = sum(v not in BAD_OBJECT for v in pix)
            if good != 21:
                continue
            corners = _at(im, x, y, [(-2, -2), (2, -2),
                                     (-2, 2), (2, 2)])
            score = 42 + 0.3 * sum(v in BAD_OBJECT for v in corners)
            if int(im[y, x]) == 6:
                score += 3
            circles.append((score, x, y, pix))
    circles.sort(reverse=True)
    C = []
    for c in circles:
        if not any(max(abs(c[1] - b["x"]), abs(c[2] - b["y"])) < 3
                   for b in C):
            C.append({"x": c[1], "y": c[2], "pix": c[3]})

    groups = []
    used_circles = set()
    colors = sorted(set(a[2] for a in A))
    for color in colors:
        ids = [i for i, a in enumerate(A) if a[2] == color]
        candidates = []
        for j, c in enumerate(C):
            if color == 4:
                ok = c["pix"][BALL.index((0, 0))] != 6
            else:
                ok = c["pix"].count(color) >= 12
            if ok:
                candidates.append(j)

        by_pos = {}
        for j in candidates:
            by_pos.setdefault((C[j]["x"], C[j]["y"]), []).append(j)

        options = []
        k = len(ids)
        if k <= 13:
            for mask in range(1, 1 << k):
                count = mask.bit_count()
                if count < 2 and k > 1:
                    continue
                sub = [ids[t] for t in range(k) if mask >> t & 1]
                bx, by = _center(A, sub)
                matches = by_pos.get((bx, by), [])
                if not matches:
                    continue
                j = matches[0]
                support = sum(_line_support(im, A[i][:2], (bx, by))
                              for i in sub)
                score = 5 * support - 2 * count - 0.05
                options.append((mask, score, sub, j))

        lookup = [[] for _ in range(k)]
        for q, o in enumerate(options):
            for t in range(k):
                if o[0] >> t & 1:
                    lookup[t].append(q)

        @lru_cache(None)
        def cover(mask):
            if mask == 0:
                return (0.0, ())
            bit = (mask & -mask).bit_length() - 1
            best = (-1e20, ())
            for q in lookup[bit]:
                o = options[q]
                if o[0] & mask != o[0]:
                    continue
                value, rest = cover(mask ^ o[0])
                value += o[1]
                if value > best[0]:
                    best = (value, (q,) + rest)
            return best

        result = cover((1 << k) - 1) if k and options else (-1e20, ())
        partitions = []
        if result[0] > -1e10:
            partitions = [(options[q][2], options[q][3]) for q in result[1]]
        else:
            # The common case also provides a reliable fallback when an
            # anchor or ball partially obscures another sprite.
            bx, by = _center(A, ids)
            near = sorted(candidates,
                          key=lambda j: abs(C[j]["x"] - bx) + abs(C[j]["y"] - by))
            if near and abs(C[near[0]]["x"] - bx) + abs(C[near[0]]["y"] - by) <= 3:
                partitions = [(ids, near[0])]
            elif candidates:
                buckets = {}
                for i in ids:
                    j = max(candidates,
                            key=lambda j: _line_support(
                                im, A[i][:2], (C[j]["x"], C[j]["y"])))
                    buckets.setdefault(j, []).append(i)
                partitions = [(v, j) for j, v in buckets.items()]
            else:
                partitions = [(ids, None)]

        for sub, j in partitions:
            if not sub:
                continue
            bx, by = _center(A, sub)
            if j is not None:
                pix = list(C[j]["pix"])
                used_circles.add(j)
            else:
                pix = [color if color != 4 else 0 for _ in BALL]
                if color != 4:
                    pix[BALL.index((0, 0))] = 6
            groups.append({"ids": sub, "base": 0 if color == 4 else color,
                           "pix": pix})

    pickups = []
    for j, c in enumerate(C):
        if j in used_circles:
            continue
        colors_here = _paint_set(c["pix"])
        if not colors_here or 6 in colors_here:
            continue
        if 0 not in c["pix"]:
            continue
        pickups.append({"x": c["x"], "y": c["y"],
                        "pix": list(c["pix"]), "alive": True,
                        "colors": colors_here})

    raw_goals = []
    for y in range(3, 61):
        for x in range(3, 61):
            vals = _at(im, x, y, RING)
            good = sum(v not in BAD_OBJECT for v in vals)
            if good < 10:
                continue
            counts = Counter(v for v in vals if v > 0 and v not in BAD_OBJECT)
            colors_here = frozenset(v for v, count in counts.items() if count >= 2)
            if vals.count(0) >= 6 or not colors_here:
                nearest = None
                for g in groups:
                    bx, by = _center(A, g["ids"])
                    if (bx - x, by - y) in GOAL_DIFF:
                        dist = (bx - x) ** 2 + (by - y) ** 2
                        if nearest is None or dist < nearest[0]:
                            cs = (frozenset([g["base"]]) if g["base"]
                                  else _paint_set(g["pix"]))
                            nearest = (dist, cs)
                if nearest is not None:
                    colors_here = nearest[1]
            if colors_here:
                raw_goals.append((good, x, y, colors_here))

    raw_goals.sort(reverse=True)
    goals = []
    for score, x, y, cs in raw_goals:
        if any(max(abs(x - g["x"]), abs(y - g["y"])) < 4 for g in goals):
            continue
        goals.append({"x": x, "y": y, "colors": cs,
                      "zone": _zone(x, y, GOAL_DIFF)})

    colored = set(g["base"] for g in groups if g["base"])
    possible_paint = set()
    for p in pickups:
        possible_paint.update(p["colors"])
    for g in groups:
        if not g["base"]:
            possible_paint.update(_paint_set(g["pix"]))
    split_exists = any(len(g["colors"]) > 1 and
                       set(g["colors"]) <= possible_paint for g in goals)
    goals = [g for g in goals
             if (len(g["colors"]) == 1 and next(iter(g["colors"])) in colored)
             or (set(g["colors"]) <= possible_paint
                 and (len(g["colors"]) > 1 or not split_exists))]

    return A, selected, groups, pickups, goals


def _select_point(A, i):
    x, y = A[i][:2]
    points = [(x, y)]
    points += [(x + dx, y + dy) for dy in range(-2, 3)
               for dx in range(-2, 3) if dx or dy]
    for a, b in points:
        if not (0 <= a < 64 and 0 <= b < 64):
            continue
        if all(j == i or abs(a - p[0]) > 2 or abs(b - p[1]) > 2
               for j, p in enumerate(A)):
            return a, b
    return None


def _terrain(im, M, initial=False):
    if initial:
        M["wall"] = (im == 2)
        M["hazard"] = (im == 10)
        unknown = ~np.isin(im, [2, 5, 10])
        a = np.zeros((64, 64), dtype=np.int16)
        b = np.zeros_like(a)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                y0, y1 = max(0, -dy), min(64, 64 - dy)
                x0, x1 = max(0, -dx), min(64, 64 - dx)
                sl = im[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                a[y0:y1, x0:x1] += (sl == 10)
                b[y0:y1, x0:x1] += (sl == 5)
        M["hazard"] |= unknown & (a > b) & (a > 0)
    else:
        M["wall"][im == 2] = True
        M["wall"][(im == 5) | (im == 10)] = False
        M["hazard"][im == 10] = True
        M["hazard"][(im == 5) | (im == 2)] = False

    M["wall"][:, 0] = M["wall"][:, 1]
    # A displayed stable ball is evidence that its current footprint is safe.
    for g in M["groups"]:
        x, y = _center(M["A"], g["ids"])
        for dx, dy in BALL:
            if 0 <= x + dx < 64 and 0 <= y + dy < 64:
                M["hazard"][y + dy, x + dx] = False
    allowed = ~_collision_map(M["wall"], DIAMOND)
    allowed[:2, :] = allowed[62:, :] = False
    allowed[:, :2] = allowed[:, 62:] = False
    M["allowed"] = allowed
    M["safe"] = ~_collision_map(M["hazard"], BALL)


def _resolve(im, M):
    pending = M.pop("pending", None)
    if pending is not None:
        aid, nx, ny, oldx, oldy = pending
        color = M["A"][aid][2]

        def evidence(x, y):
            vals = _at(im, x, y, DBODY)
            s = sum(v == 0 for v in vals)
            if 0 <= x < 64 and 0 <= y < 64 and int(im[y, x]) == color:
                s += 4
            return s

        success = evidence(nx, ny) >= 9 or evidence(oldx, oldy) < 9
        if not success:
            group = next(g for g in M["groups"] if aid in g["ids"])
            old_ball = _center(M["A"], group["ids"])
            n = len(group["ids"])
            nb = (old_ball[0], old_ball[1])
            sx = sum(M["A"][i][0] for i in group["ids"]) - oldx + nx
            sy = sum(M["A"][i][1] for i in group["ids"]) - oldy + ny
            nb = sx // n, sy // n
            if nb != old_ball and group["base"]:
                success = int(im[nb[1], nb[0]]) == 6

        if success:
            M["A"][aid][0:2] = [nx, ny]
            M["failures"] = 0
            for g in M["groups"]:
                if g["base"]:
                    continue
                bx, by = _center(M["A"], g["ids"])
                for p in M["pickups"]:
                    if p["alive"] and (bx - p["x"], by - p["y"]) in BALL_DIFF:
                        g["pix"] = [v if v > 0 else old
                                    for old, v in zip(g["pix"], p["pix"])]
                        p["alive"] = False
        else:
            M["plan"] = []
            M["failures"] = M.get("failures", 0) + 1
            key = (tuple((a[0], a[1]) for a in M["A"]), aid, nx, ny)
            M.setdefault("blocked", set()).add(key)

    for g in M["groups"]:
        if g["base"]:
            continue
        x, y = _center(M["A"], g["ids"])
        pix = _at(im, x, y, BALL)
        if len(pix) == 21 and all(v not in BAD_OBJECT for v in pix):
            g["pix"] = pix


def _recipe(M, group, goal):
    desired = goal["colors"]
    pix = group["pix"]
    bx, by = _center(M["A"], group["ids"])
    if _paint_set(pix) == desired:
        return (), math.hypot(bx - goal["x"], by - goal["y"])
    choices = [i for i, p in enumerate(M["pickups"])
               if p["alive"] and p["colors"] <= desired]
    if len(choices) > 7:
        choices = choices[:7]
    best = None
    for length in range(1, len(choices) + 1):
        for order in itertools.permutations(choices, length):
            pp = list(pix)
            x, y = bx, by
            cost = 16 * length
            for i in order:
                p = M["pickups"][i]
                pp = [v if v > 0 else old for old, v in zip(pp, p["pix"])]
                cost += math.hypot(x - p["x"], y - p["y"])
                x, y = p["x"], p["y"]
            if _paint_set(pp) != desired:
                continue
            cost += math.hypot(x - goal["x"], y - goal["y"])
            if best is None or cost < best[1]:
                best = (order, cost)
        if best is not None:
            break
    return best


def _assign(M):
    groups, goals = M["groups"], M["goals"]
    assigned = {}
    black = []
    black_goals = []
    colored_values = set(g["base"] for g in groups if g["base"])
    for i, g in enumerate(groups):
        if g["base"]:
            options = [j for j, t in enumerate(goals)
                       if t["colors"] == frozenset([g["base"]])]
            if options:
                bx, by = _center(M["A"], g["ids"])
                assigned[i] = min(options, key=lambda j:
                                  (goals[j]["x"] - bx) ** 2 +
                                  (goals[j]["y"] - by) ** 2)
        else:
            black.append(i)
    for j, t in enumerate(goals):
        if not (len(t["colors"]) == 1 and
                next(iter(t["colors"])) in colored_values):
            black_goals.append(j)

    recipes = {}
    for i in black:
        for j in black_goals:
            r = _recipe(M, groups[i], goals[j])
            if r is not None:
                recipes[i, j] = r

    if black and len(black_goals) >= len(black):
        best = None
        for js in itertools.permutations(black_goals, len(black)):
            if not all((i, j) in recipes for i, j in zip(black, js)):
                continue
            cost = sum(recipes[i, j][1] * len(groups[i]["ids"])
                       for i, j in zip(black, js))
            if best is None or cost < best[0]:
                best = cost, js
        if best is not None:
            assigned.update(zip(black, best[1]))
    return assigned, recipes


def _square_dilate(a, r):
    p = np.pad(a.astype(np.int32), ((r, r), (r, r)))
    s = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    w = 2 * r + 1
    return (s[w:, w:] - s[:-w, w:] -
            s[w:, :-w] + s[:-w, :-w]) > 0


def _plan(M, gi, target, zone, forbidden, deadline):
    ids = M["groups"][gi]["ids"]
    A = M["A"]
    n = len(ids)
    safe = M["safe"] & ~forbidden
    zone = zone & safe
    if not np.any(zone):
        return []
    qy, qx = np.nonzero(M["allowed"])
    if not len(qx):
        return []
    qx = qx.astype(np.int16)
    qy = qy.astype(np.int16)
    radius = max(1, (63 + n - 1) // n)
    field = np.full((64, 64), 99, dtype=np.int16)
    reached = zone.copy()
    field[zone] = 0
    for d in range(1, 15):
        new = _square_dilate(reached, radius) & safe & ~reached
        if not np.any(new):
            break
        field[new] = d
        reached |= new
    tx, ty = target
    distance = np.hypot(XX - tx, YY - ty) / radius
    zy, zx = np.nonzero(zone)
    lo_x, hi_x = int(zx.min()), int(zx.max())
    lo_y, hi_y = int(zy.min()), int(zy.max())
    initial = tuple((A[i][0], A[i][1]) for i in ids)
    initial_sel = ids.index(M["sel"]) if M["sel"] in ids else -1
    outside = [p[:2] for i, p in enumerate(A) if i not in ids]
    blocked = M.get("blocked", set())
    past = M.setdefault("past", {})
    past_key = (gi, initial)
    past[past_key] = past.get(past_key, 0) + 1

    def heuristic(pos):
        sx = sum(p[0] for p in pos)
        sy = sum(p[1] for p in pos)
        bx, by = sx // n, sy // n
        moves = int(field[by, bx])
        for axis, s, low, high in ((0, sx, lo_x, hi_x),
                                   (1, sy, lo_y, hi_y)):
            if s < n * low:
                need = n * low - s
                gains = sorted((61 - p[axis] for p in pos), reverse=True)
            elif s > n * (high + 1) - 1:
                need = s - (n * (high + 1) - 1)
                gains = sorted((p[axis] - 2 for p in pos), reverse=True)
            else:
                continue
            total = 0
            count = 0
            for gain in gains:
                count += 1
                total += gain
                if total >= need:
                    break
            moves = max(moves, count)
        return 1.5 * moves + 0.22 * float(distance[by, bx])

    serial = itertools.count()
    h0 = heuristic(initial)
    heap = [(1.8 * h0, next(serial), 0, initial, initial_sel, ())]
    seen = {(initial, initial_sel): 0}
    best_path = ()
    best_measure = h0
    expansions = 0
    while heap and expansions < 1400 and time.monotonic() < deadline:
        _, _, cost, pos, sel, path = heapq.heappop(heap)
        if seen.get((pos, sel), 1e9) < cost:
            continue
        expansions += 1
        sx = sum(p[0] for p in pos)
        sy = sum(p[1] for p in pos)
        bx, by = sx // n, sy // n
        if len(path) >= 11:
            continue
        order = list(range(n))
        if sel >= 0:
            order.remove(sel)
            order.insert(0, sel)

        for ai in order:
            aid = ids[ai]
            if ai != sel:
                temp_A = [list(a) for a in A]
                for j, p in zip(ids, pos):
                    temp_A[j][:2] = p
                if _select_point(temp_A, aid) is None:
                    continue
            ox, oy = pos[ai]
            nbx = (qx + sx - ox) // n
            nby = (qy + sy - oy) // n
            valid = safe[nby, nbx].copy()
            valid &= (nbx != bx) | (nby != by)
            for x, y in outside + list(pos):
                valid &= (np.abs(qx - x) > 2) | (np.abs(qy - y) > 2)
            if not np.any(valid):
                continue
            if not path and blocked:
                state = tuple((a[0], a[1]) for a in A)
                for st, bad_id, x, y in blocked:
                    if st == state and bad_id == aid:
                        valid &= (qx != x) | (qy != y)

            quality = (2.0 * field[nby, nbx] + distance[nby, nbx] +
                       0.0005 * np.hypot(qx - tx, qy - ty))
            quality = np.where(valid, quality, 1e9)
            finish = valid & zone[nby, nbx]
            if np.any(finish):
                q = int(np.argmin(np.where(finish, quality, 1e9)))
                return list(path + ((aid, int(qx[q]), int(qy[q])),))

            inds = np.flatnonzero(valid)
            take = min(70, len(inds))
            if take < len(inds):
                inds = inds[np.argpartition(quality[inds], take - 1)[:take]]
            inds = inds[np.argsort(quality[inds])]
            chosen = []
            centers = []
            for q in inds:
                c = (int(nbx[q]), int(nby[q]))
                if all(max(abs(c[0] - a), abs(c[1] - b)) >= 2
                       for a, b in centers):
                    chosen.append(int(q))
                    centers.append(c)
                if len(chosen) >= 7:
                    break
            # Include alternatives around both sides of obstacles, not just
            # endpoints with the best immediate distance.
            dxbin = np.sign(nbx - bx).astype(np.int8)
            dybin = np.sign(nby - by).astype(np.int8)
            for dx, dy in itertools.product((-1, 0, 1), repeat=2):
                if dx == dy == 0:
                    continue
                mask = valid & (dxbin == dx) & (dybin == dy)
                if np.any(mask):
                    chosen.append(int(np.argmin(np.where(mask, quality, 1e9))))
            for ux, uy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                rank = quality - 0.025 * (ux * qx + uy * qy)
                chosen.append(int(np.argmin(rank)))

            for q in set(chosen):
                if not valid[q]:
                    continue
                p = (int(qx[q]), int(qy[q]))
                new_pos = pos[:ai] + (p,) + pos[ai + 1:]
                new_cost = cost + (1 if ai == sel else 2)
                key = new_pos, ai
                if seen.get(key, 1e9) <= new_cost:
                    continue
                seen[key] = new_cost
                new_path = path + ((aid, p[0], p[1]),)
                h = heuristic(new_pos)
                repeated = past.get((gi, new_pos), 0)
                measure = h + 0.07 * new_cost + 2 * repeated
                if measure < best_measure:
                    best_measure = measure
                    best_path = new_path
                priority = new_cost + 1.8 * h + 3 * repeated
                heapq.heappush(heap, (priority, next(serial), new_cost,
                                      new_pos, ai, new_path))
        if len(heap) > 65000:
            heap = heapq.nsmallest(30000, heap)
            heapq.heapify(heap)
    return list(best_path)


def _fallback(available, memory):
    resets = memory.get("_resets", 0)
    if "RESET" in available and resets < 2:
        memory.clear()
        memory["_resets"] = resets + 1
        return "RESET", None, None
    return "ACTION6", 32, 32


def act(obs, available, memory):
    start = time.monotonic()
    im = np.asarray(obs)
    if "A" not in memory:
        A, sel, groups, pickups, goals = _vision(im)
        if not A or not groups or not goals:
            return _fallback(available, memory)
        memory.update(A=A, sel=sel, groups=groups, pickups=pickups,
                      goals=goals, plan=[], failures=0, ticks=0,
                      active=None, blocked=set(), past={})
        _terrain(im, memory, True)
    else:
        _resolve(im, memory)
        _terrain(im, memory, False)

    M = memory
    M["ticks"] += 1
    if M.get("failures", 0) >= 4:
        return _fallback(available, memory)

    assignment, recipes = _assign(M)
    unsolved = []
    for gi, gj in assignment.items():
        group, goal = M["groups"][gi], M["goals"][gj]
        bx, by = _center(M["A"], group["ids"])
        right_color = bool(group["base"]) or _paint_set(group["pix"]) == goal["colors"]
        if not (right_color and goal["zone"][by, bx]):
            unsolved.append(gi)

    if not unsolved:
        # Normally the engine has already ended the episode. If the supplied
        # state merely needs a move to trigger its win test, preserve the goals.
        if assignment:
            unsolved = list(assignment)
        else:
            return _fallback(available, memory)

    active = M.get("active")
    if active not in unsolved:
        def task_cost(gi):
            g = M["groups"][gi]
            goal = M["goals"][assignment[gi]]
            bx, by = _center(M["A"], g["ids"])
            if g["base"]:
                c = math.hypot(bx - goal["x"], by - goal["y"])
            else:
                c = recipes.get((gi, assignment[gi]), ((), 1e5))[1]
            return c * len(g["ids"]) / 55 - (2 if M["sel"] in g["ids"] else 0)
        active = min(unsolved, key=task_cost)
        M["active"] = active
        M["plan"] = []

    gi = active
    group = M["groups"][gi]
    gj = assignment[gi]
    goal = M["goals"][gj]
    pickup_id = -1
    if not group["base"]:
        recipe = recipes.get((gi, gj))
        if recipe is None:
            return _fallback(available, memory)
        if recipe[0]:
            pickup_id = recipe[0][0]

    if pickup_id >= 0:
        p = M["pickups"][pickup_id]
        target = (p["x"], p["y"])
        zone = _zone(*target, BALL_DIFF)
    else:
        target = (goal["x"], goal["y"])
        zone = goal["zone"]

    mission = (gi, gj, pickup_id)
    if M.get("mission") != mission:
        M["plan"] = []
        M["mission"] = mission

    if not M["plan"]:
        forbidden = np.zeros((64, 64), dtype=bool)
        if not group["base"]:
            for p in M["pickups"]:
                if p["alive"] and not p["colors"] <= goal["colors"]:
                    forbidden |= _zone(p["x"], p["y"], BALL_DIFF)
        M["plan"] = _plan(M, gi, target, zone, forbidden, start + 8.2)

    if not M["plan"]:
        M["active"] = None
        M["failures"] = M.get("failures", 0) + 1
        if M["failures"] >= 2:
            return _fallback(available, memory)
        # Selecting a useful control gives a harmless opportunity to replan.
        aid = group["ids"][0]
        point = _select_point(M["A"], aid)
        if point is not None:
            M["sel"] = aid
            return "ACTION6", int(point[1]), int(point[0])
        return _fallback(available, memory)

    aid, x, y = M["plan"][0]
    if M["sel"] != aid:
        point = _select_point(M["A"], aid)
        if point is None:
            M["plan"] = []
            return _fallback(available, memory)
        M["sel"] = aid
        return "ACTION6", int(point[1]), int(point[0])

    M["plan"].pop(0)
    oldx, oldy = M["A"][aid][:2]
    M["pending"] = (aid, x, y, oldx, oldy)
    return "ACTION6", int(y), int(x)

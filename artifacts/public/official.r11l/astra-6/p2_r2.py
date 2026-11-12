import numpy as np
import math
import heapq
import itertools

_BODY = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
         if not (abs(x) == 2 and abs(y) == 2)]
_DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
            if abs(x) + abs(y) <= 2]
_DBORDER = [p for p in _DIAMOND if p != (0, 0)]
_DEXTRA = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
           if abs(x) + abs(y) == 3]
_RING = [(-1, -3), (1, -3), (-2, -2), (2, -2),
         (-3, -1), (3, -1), (-3, 1), (3, 1),
         (-2, 2), (2, 2), (-1, 3), (1, 3)]
_GBODY = [(x, y) for y in range(-3, 4)
          for x in range(-{0: 3, 1: 3, 2: 2, 3: 1}[abs(y)],
                         {0: 3, 1: 3, 2: 2, 3: 1}[abs(y)] + 1)]
_PICK_TOUCH = frozenset((x - a, y - b)
                        for x, y in _BODY for a, b in _BODY)
_GOAL_TOUCH = frozenset((x - a, y - b)
                        for x, y in _GBODY for a, b in _BODY)
_INK = frozenset((0, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15))


def _pixel(o, x, y, default=5):
    return int(o[y, x]) if 0 <= x < 64 and 0 <= y < 64 else default


def _body_at(o, x, y):
    p = np.full((5, 5), -1, dtype=np.int16)
    for dx, dy in _BODY:
        p[dy + 2, dx + 2] = _pixel(o, x + dx, y + dy, 0)
    return p


def _colors(p):
    return frozenset(int(c) for c in np.unique(p) if c > 0)


def _touch(a, b, shape):
    return (a[0] - b[0], a[1] - b[1]) in shape


def _mean(points):
    return (sum(p[0] for p in points) // len(points),
            sum(p[1] for p in points) // len(points))


def _handle_strength(o, p, label):
    x, y = p
    score = sum(_pixel(o, x + dx, y + dy) in (0, 3)
                for dx, dy in _DBORDER)
    return score + 3 * (_pixel(o, x, y) == label)


def _detect_handles(o):
    handles = []
    yy, xx = np.nonzero(~np.isin(o, (0, 1, 2, 3, 5)))
    for y, x in zip(yy.tolist(), xx.tolist()):
        label = int(o[y, x])
        visible = [(dx, dy) for dx, dy in _DBORDER
                   if 0 <= x + dx < 64 and 0 <= y + dy < 64]
        if len(visible) < 5:
            continue
        vals = [int(o[y + dy, x + dx]) for dx, dy in visible]
        n0, n3 = vals.count(0), vals.count(3)
        if max(n0, n3) < max(5, len(visible) - 2):
            continue
        extra = sum(_pixel(o, x + dx, y + dy) in (0, 3)
                    for dx, dy in _DEXTRA)
        if extra <= 2:
            handles.append({"p": (x, y), "label": label,
                            "selected": n0 > n3})
    return handles


def _line_score(o, a, b):
    x, y = a
    xx, yy = b
    dx, dy = abs(xx - x), abs(yy - y)
    sx, sy = (1 if x < xx else -1), (1 if y < yy else -1)
    err = dx - dy
    hit = miss = 0
    while True:
        c = _pixel(o, x, y)
        if c == 1:
            hit += 1
        elif c in (5, 10):
            miss += 1
        if x == xx and y == yy:
            break
        e = 2 * err
        if e > -dy:
            err -= dy
            x += sx
        if e < dx:
            err += dx
            y += sy
    return hit - 2.5 * miss


def _dilate(mask, offsets):
    out = np.zeros((64, 64), dtype=bool)
    for dx, dy in offsets:
        x0, x1 = max(0, -dx), min(64, 64 - dx)
        y0, y1 = max(0, -dy), min(64, 64 - dy)
        out[y0:y1, x0:x1] |= mask[y0 + dy:y1 + dy,
                                  x0 + dx:x1 + dx]
    return out


def _stamp(mask, center, offsets, value=True):
    x, y = center
    for dx, dy in offsets:
        xx, yy = x + dx, y + dy
        if 0 <= xx < 64 and 0 <= yy < 64:
            mask[yy, xx] = value


def _observe(o):
    handles = _detect_handles(o)
    circles = []
    for y in range(2, 62):
        for x in range(2, 62):
            if int(o[y, x]) not in _INK:
                continue
            good = sum(int(o[y + dy, x + dx]) in _INK
                       for dx, dy in _BODY)
            if good < 20:
                continue
            corners = sum(int(o[y + dy, x + dx]) not in _INK
                          for dx, dy in ((-2, -2), (2, -2),
                                         (-2, 2), (2, 2)))
            if corners < 2:
                continue
            if any(abs(x - c["p"][0]) <= 1 and
                   abs(y - c["p"][1]) <= 1 for c in circles):
                continue
            p = _body_at(o, x, y)
            edge = [int(p[dy + 2, dx + 2])
                    for dx, dy in _BODY if (dx, dy) != (0, 0)]
            colored = (int(p[2, 2]) == 6 and len(set(edge)) == 1
                       and edge[0] > 0)
            circles.append({"p": (x, y), "body": p,
                            "colored": colored,
                            "label": edge[0] if colored else 4})

    groups = []
    used_circles = set()
    used_handles = set()

    labels = sorted(set(h["label"] for h in handles if h["label"] != 4))
    for label in labels:
        inds = [i for i, h in enumerate(handles) if h["label"] == label]
        center = _mean([handles[i]["p"] for i in inds])
        choices = [(abs(c["p"][0] - center[0]) +
                    abs(c["p"][1] - center[1]), j)
                   for j, c in enumerate(circles)
                   if c["colored"] and c["label"] == label]
        j = min(choices)[1] if choices else None
        if j is not None:
            body = circles[j]["body"].copy()
            used_circles.add(j)
        else:
            body = np.full((5, 5), -1, dtype=np.int16)
            for dx, dy in _BODY:
                body[dy + 2, dx + 2] = label
            body[2, 2] = 6
        groups.append({"ids": inds, "p": center, "body": body,
                       "black": False, "label": label,
                       "goal": None, "reserved": []})
        used_handles.update(inds)

    black_ids = [i for i in range(len(handles)) if i not in used_handles]
    n = len(black_ids)
    candidates = []
    if 2 <= n <= 15:
        sums = [(0, 0, 0)] * (1 << n)
        by_pos = {}
        for j, c in enumerate(circles):
            if j not in used_circles and not c["colored"]:
                by_pos.setdefault(c["p"], []).append(j)
        for bits in range(1, 1 << n):
            bit = bits & -bits
            k = bit.bit_length() - 1
            sx, sy, count = sums[bits ^ bit]
            hx, hy = handles[black_ids[k]]["p"]
            sx, sy, count = sx + hx, sy + hy, count + 1
            sums[bits] = (sx, sy, count)
            if count < 2:
                continue
            center = (sx // count, sy // count)
            for j in by_pos.get(center, []):
                inds = [black_ids[k] for k in range(n) if bits >> k & 1]
                score = sum(_line_score(o, handles[i]["p"], center)
                            for i in inds)
                candidates.append((bits, j, inds, score))

        by_bit = [[] for _ in range(n)]
        for c in candidates:
            for k in range(n):
                if c[0] >> k & 1:
                    by_bit[k].append(c)
        best = [None, -1e30]
        nodes = [0]

        def cover(left, taken, answer, score):
            nodes[0] += 1
            if nodes[0] > 30000:
                return
            if not left:
                if score > best[1]:
                    best[:] = [answer[:], score]
                return
            possible_bits = [k for k in range(n) if left >> k & 1]
            k = min(possible_bits, key=lambda z: len(by_bit[z]))
            for c in by_bit[k]:
                bits, j, inds, value = c
                if bits & left == bits and j not in taken:
                    cover(left ^ bits, taken | {j}, answer + [c],
                          score + value + 2)

        cover((1 << n) - 1, set(), [], 0)
        if best[0] is not None:
            for bits, j, inds, score in best[0]:
                c = circles[j]
                groups.append({"ids": inds, "p": c["p"],
                               "body": c["body"].copy(), "black": True,
                               "label": 4, "goal": None, "reserved": []})
                used_circles.add(j)
                used_handles.update(inds)

    remaining = [i for i in range(len(handles)) if i not in used_handles]
    if remaining:
        bins = {}
        for i in remaining:
            options = []
            for j, c in enumerate(circles):
                if j in used_circles or c["colored"]:
                    continue
                options.append((_line_score(o, handles[i]["p"], c["p"]), j))
            if options:
                j = max(options)[1]
                bins.setdefault(j, []).append(i)
        if not bins:
            bins = {-1: remaining}
        for j, inds in bins.items():
            center = _mean([handles[i]["p"] for i in inds])
            body = (circles[j]["body"].copy() if j >= 0
                    else _body_at(o, *center))
            groups.append({"ids": inds, "p": center, "body": body,
                           "black": True, "label": 4,
                           "goal": None, "reserved": []})
            if j >= 0:
                used_circles.add(j)

    goals_raw = []
    for y in range(3, 61):
        for x in range(3, 61):
            vals = [int(o[y + dy, x + dx]) for dx, dy in _RING]
            good = sum(v in _INK for v in vals)
            if good < 10:
                continue
            cols = frozenset(v for v in vals if v in _INK and v > 0)
            near = [g for g in groups
                    if _touch(g["p"], (x, y), _GOAL_TOUCH)]
            if vals.count(0) >= 4 and near:
                nearest = min(near, key=lambda g:
                              abs(g["p"][0] - x) + abs(g["p"][1] - y))
                cols = (_colors(nearest["body"]) if nearest["black"]
                        else frozenset((nearest["label"],)))
            if cols:
                goals_raw.append((good, (x, y), cols))

    goals = []
    for good, p, cols in sorted(goals_raw, reverse=True):
        if any(max(abs(p[0] - q["p"][0]),
                   abs(p[1] - q["p"][1])) <= 3 for q in goals):
            continue
        goals.append({"p": p, "colors": cols})

    picks = {j: {"p": c["p"], "body": c["body"].copy()}
             for j, c in enumerate(circles)
             if j not in used_circles and not c["colored"]
             and _colors(c["body"])}

    wall = (o == 2).copy()
    wall[:, 0] = wall[:, 1]
    hazard = (o == 10).copy()
    for _ in range(2):
        near = np.zeros((64, 64), dtype=np.int16)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1)):
            shifted = np.zeros_like(hazard)
            x0, x1 = max(0, -dx), min(64, 64 - dx)
            y0, y1 = max(0, -dy), min(64, 64 - dy)
            shifted[y0:y1, x0:x1] = hazard[y0 + dy:y1 + dy,
                                           x0 + dx:x1 + dx]
            near += shifted
        hazard |= (o == 1) & (near >= 4)

    free_anchor = ~_dilate(wall, _DIAMOND)
    free_anchor[:2] = False
    free_anchor[-2:] = False
    free_anchor[:, :2] = False
    free_anchor[:, -2:] = False
    safe = ~_dilate(hazard | wall, _BODY)
    safe_loose = ~_dilate(hazard, _BODY)
    selected = next((i for i, h in enumerate(handles) if h["selected"]), 0)

    universe = set()
    for p in picks.values():
        universe.update(_colors(p["body"]))
    for g in groups:
        if g["black"]:
            universe.update(_colors(g["body"]))
    prefer_pairs = any(len(t["colors"]) == 2 and
                       t["colors"].issubset(universe) for t in goals)

    return {"handles": handles, "groups": groups, "goals": goals,
            "picks": picks, "selected": selected,
            "free_anchor": free_anchor, "safe": safe,
            "safe_loose": safe_loose, "prefer_pairs": prefer_pairs,
            "plan": [], "pending": None, "active": None,
            "failures": 0, "failed_move": False,
            "uncertain": False, "confirm": 0}


def _paint(body, patch):
    result = body.copy()
    mask = patch > 0
    result[mask] = patch[mask]
    return result


def _mission_choices(w, gi, fixed=False, excluded=frozenset()):
    g = w["groups"][gi]
    targets = ([g["goal"]] if fixed and g["goal"] is not None
               else range(len(w["goals"])))
    choices = []
    n = len(g["ids"])
    for ti in targets:
        target = w["goals"][ti]
        desired = target["colors"]
        if not g["black"]:
            if desired != frozenset((g["label"],)):
                continue
            d = max(abs(g["p"][0] - target["p"][0]),
                    abs(g["p"][1] - target["p"][1]))
            choices.append((max(0, d - 4) * n / 55.0, ti, ()))
            continue
        eligible = [j for j, p in w["picks"].items()
                    if j not in excluded and
                    _colors(p["body"]).issubset(desired)]
        for length in range(min(3, len(eligible)) + 1):
            for seq in itertools.permutations(eligible, length):
                body = g["body"]
                for j in seq:
                    body = _paint(body, w["picks"][j]["body"])
                if _colors(body) != desired:
                    continue
                pos = g["p"]
                cost = 1.2 * length
                for j in seq:
                    q = w["picks"][j]["p"]
                    cost += max(abs(pos[0] - q[0]),
                                abs(pos[1] - q[1])) * n / 55.0
                    pos = q
                q = target["p"]
                cost += max(0, max(abs(pos[0] - q[0]),
                                   abs(pos[1] - q[1])) - 4) * n / 55.0
                if w["prefer_pairs"] and len(desired) < 2:
                    cost += 30
                choices.append((cost, ti, seq))
    choices.sort()
    return choices[:48]


def _assign(w):
    groups = w["groups"]
    opts = [_mission_choices(w, i) for i in range(len(groups))]
    if any(not a for a in opts):
        return False
    order = sorted(range(len(groups)), key=lambda i: len(opts[i]))
    best = [float("inf"), None]
    nodes = [0]

    def rec(k, used_goals, used_picks, score, chosen):
        nodes[0] += 1
        if nodes[0] > 50000 or score >= best[0]:
            return
        if k == len(order):
            best[:] = [score, chosen.copy()]
            return
        i = order[k]
        for cost, ti, seq in opts[i]:
            if ti in used_goals or any(j in used_picks for j in seq):
                continue
            chosen[i] = (cost, ti, seq)
            rec(k + 1, used_goals | {ti}, used_picks | set(seq),
                score + cost, chosen)
            chosen.pop(i, None)

    rec(0, set(), set(), 0.0, {})
    if best[1] is None:
        return False
    for i, choice in best[1].items():
        groups[i]["goal"] = choice[1]
        groups[i]["reserved"] = list(choice[2])
    return True


def _done(w, gi):
    g = w["groups"][gi]
    if g["goal"] is None:
        return False
    t = w["goals"][g["goal"]]
    cols = _colors(g["body"]) if g["black"] else frozenset((g["label"],))
    return cols == t["colors"] and _touch(g["p"], t["p"], _GOAL_TOUCH)


def _distance_field(safe, terminal, radius):
    dist = np.full((64, 64), 1e5, dtype=float)
    heap = []
    ys, xs = np.nonzero(terminal & safe)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0
        heap.append((0.0, x, y))
    heapq.heapify(heap)
    half = max(1, radius // 2)
    jumps = set()
    for r in (1, half, radius):
        for dx, dy in ((-1, -1), (0, -1), (1, -1),
                       (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            jumps.add((dx * r, dy * r))
    for a, b in ((half, radius), (radius, half)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                jumps.add((sx * a, sy * b))
    jumps = [(dx, dy, 1.0 + 0.12 * math.hypot(dx, dy) / radius)
             for dx, dy in jumps]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x] + 1e-8:
            continue
        for dx, dy, weight in jumps:
            xx, yy = x + dx, y + dy
            if 0 <= xx < 64 and 0 <= yy < 64 and safe[yy, xx]:
                nd = d + weight
                if nd < dist[yy, xx] - 1e-8:
                    dist[yy, xx] = nd
                    heapq.heappush(heap, (nd, xx, yy))
    return dist


def _search(w, gi, center, is_pickup, pickup_id=None,
            width=32, depth_limit=9, branch=7, loose=False):
    g = w["groups"][gi]
    ids = g["ids"]
    n = len(ids)
    safe = (w["safe_loose"] if loose else w["safe"]).copy()
    terminal = np.zeros((64, 64), dtype=bool)
    _stamp(terminal, center, _PICK_TOUCH if is_pickup else _GOAL_TOUCH)

    if g["black"]:
        for j, p in w["picks"].items():
            if j != pickup_id:
                _stamp(safe, p["p"], _PICK_TOUCH, False)
    for other, gg in enumerate(w["groups"]):
        if other != gi and not _touch(gg["p"], center, _GOAL_TOUCH):
            _stamp(safe, gg["p"], _PICK_TOUCH, False)

    ys, xs = np.nonzero(w["free_anchor"])
    ax, ay = xs.astype(np.int16), ys.astype(np.int16)
    if not len(ax) or not np.any(terminal & safe):
        return None
    field = _distance_field(safe, terminal, max(1, (63 + n - 1) // n))
    initial = tuple(w["handles"][i]["p"] for i in ids)
    fixed_handles = [h["p"] for i, h in enumerate(w["handles"]) if i not in ids]

    bias = w["goals"][g["goal"]]["p"] if g["goal"] is not None else center
    anchor_tie = 0.000015 * ((ax.astype(float) - bias[0]) ** 2 +
                             (ay.astype(float) - bias[1]) ** 2)

    beam = [(initial, w["selected"], 0, [])]
    seen = {(initial, w["selected"]): 0}
    partial = None
    gx, gy = g["p"]
    partial_h = float(field[gy, gx]) if 0 <= gx < 64 and 0 <= gy < 64 else 1e5
    for depth in range(depth_limit):
        future = []
        solutions = []
        for points, last, cost, path in beam:
            sx, sy = sum(p[0] for p in points), sum(p[1] for p in points)
            for local, global_id in enumerate(ids):
                if path and global_id == last:
                    continue
                old = points[local]
                bx = (sx - old[0] + ax) // n
                by = (sy - old[1] + ay) // n
                valid = safe[by, bx].copy()
                for p in points + tuple(fixed_handles):
                    valid &= ~((np.abs(ax - p[0]) <= 3) &
                               (np.abs(ay - p[1]) <= 3))
                indices = np.flatnonzero(valid)
                if not len(indices):
                    continue
                nc = cost + 1 + (global_id != last)
                hit = indices[terminal[by[indices], bx[indices]]]
                if len(hit):
                    endscore = ((bx[hit] - bias[0]) ** 2 +
                                (by[hit] - bias[1]) ** 2) + anchor_tie[hit]
                    k = int(hit[int(np.argmin(endscore))])
                    move = (global_id, (int(ax[k]), int(ay[k])))
                    solutions.append((nc, float(endscore.min()), path + [move]))
                    continue

                bid = by[indices].astype(np.int32) * 64 + bx[indices]
                tie = anchor_tie[indices].copy()
                tie += 0.12 * ((np.abs(ax[indices] - bx[indices]) <= 4) &
                               (np.abs(ay[indices] - by[indices]) <= 4))
                order = np.lexsort((tie, bid))
                sorted_ids = bid[order]
                first = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
                choices = indices[order[first]]
                score = field[by[choices], bx[choices]] + anchor_tie[choices]
                choices = choices[np.argsort(score)]
                retained = []
                spacing = 2 if n >= 4 else 3
                for kk in choices:
                    k = int(kk)
                    cp = (int(bx[k]), int(by[k]))
                    if any(max(abs(cp[0] - q[0]), abs(cp[1] - q[1])) < spacing
                           for q in retained):
                        continue
                    retained.append(cp)
                    newp = list(points)
                    dest = (int(ax[k]), int(ay[k]))
                    newp[local] = dest
                    newp = tuple(newp)
                    key = (newp, global_id)
                    if seen.get(key, 1e9) <= nc:
                        continue
                    seen[key] = nc
                    newpath = path + [(global_id, dest)]
                    h = float(field[cp[1], cp[0]])
                    shape = sum(max(abs(p[0] - center[0]),
                                    abs(p[1] - center[1])) for p in newp) / 64
                    priority = nc + 1.65 * h + 0.04 * shape
                    future.append((priority, newp, global_id, nc, newpath, cp))
                    if h < partial_h - 0.05:
                        partial_h = h
                        partial = newpath
                    if len(retained) >= branch:
                        break
        if solutions:
            solutions.sort(key=lambda z: (z[0], z[1]))
            return solutions[0][2]
        if not future:
            break
        future.sort(key=lambda z: z[0])
        beam = []
        quota = {}
        for priority, points, last, cost, path, cp in future:
            cell = (cp[0] // 3, cp[1] // 3, last)
            if quota.get(cell, 0) >= 3:
                continue
            quota[cell] = quota.get(cell, 0) + 1
            beam.append((points, last, cost, path))
            if len(beam) >= width:
                break
    if loose and partial:
        return partial[:2]
    return None


def _reconcile(w, o):
    changed = False
    detected = _detect_handles(o)
    for h in detected:
        if h["label"] == 4:
            continue
        if any(h["p"] == old["p"] and h["label"] == old["label"]
               for old in w["handles"]):
            continue
        group = next((g for g in w["groups"]
                      if not g["black"] and g["label"] == h["label"]), None)
        if group is None:
            continue
        i = len(w["handles"])
        w["handles"].append(h)
        group["ids"].append(i)
        changed = True
        if h["selected"]:
            w["selected"] = i

    for g in w["groups"]:
        g["p"] = _mean([w["handles"][j]["p"] for j in g["ids"]])
    if changed:
        w["plan"] = []

    # A visible colored-ball center supplies an independent model check.
    centers = {}
    yy, xx = np.nonzero(o == 6)
    labels = {g["label"] for g in w["groups"] if not g["black"]}
    for y, x in zip(yy.tolist(), xx.tolist()):
        for label in labels:
            count = sum(_pixel(o, x + dx, y + dy) == label
                        for dx, dy in _BODY if (dx, dy) != (0, 0))
            if count >= 17:
                centers.setdefault(label, []).append((x, y))

    w["uncertain"] = False
    for g in w["groups"]:
        if g["black"] or g["label"] not in centers:
            continue
        p = min(centers[g["label"]],
                key=lambda q: abs(q[0] - g["p"][0]) + abs(q[1] - g["p"][1]))
        if p != g["p"]:
            w["uncertain"] = True
            w["plan"] = []


def _update(w, o):
    w["failed_move"] = False
    pending = w.get("pending")
    if pending is not None:
        i, dest, old, gi = pending
        h = w["handles"][i]
        new_strength = _handle_strength(o, dest, h["label"])
        old_strength = _handle_strength(o, old, h["label"])
        success = new_strength >= 12 or old_strength < 10
        if success:
            h["p"] = dest
            g = w["groups"][gi]
            g["p"] = _mean([w["handles"][j]["p"] for j in g["ids"]])
            if g["black"]:
                eaten = [j for j, p in w["picks"].items()
                         if _touch(g["p"], p["p"], _PICK_TOUCH)]
                for j in eaten:
                    g["body"] = _paint(g["body"], w["picks"][j]["body"])
                    del w["picks"][j]
            w["failures"] = max(0, w["failures"] - 1)
        else:
            g = w["groups"][gi]
            points = [dest if j == i else w["handles"][j]["p"]
                      for j in g["ids"]]
            bad = _mean(points)
            if 0 <= bad[0] < 64 and 0 <= bad[1] < 64:
                w["safe"][bad[1], bad[0]] = False
                w["safe_loose"][bad[1], bad[0]] = False
            w["free_anchor"][dest[1], dest[0]] = False
            w["plan"] = []
            w["failures"] += 1
            w["failed_move"] = True
        w["pending"] = None

    _reconcile(w, o)
    selected = []
    for i, h in enumerate(w["handles"]):
        x, y = h["p"]
        zeros = sum(_pixel(o, x + dx, y + dy) == 0
                    for dx, dy in _DBORDER)
        if _pixel(o, x, y) == h["label"] and zeros >= 10:
            selected.append(i)
    if len(selected) == 1:
        w["selected"] = selected[0]


def _execute(w):
    i, dest = w["plan"][0]
    if w["selected"] != i:
        w["selected"] = i
        x, y = w["handles"][i]["p"]
        return "ACTION6", int(y), int(x)
    w["plan"].pop(0)
    old = w["handles"][i]["p"]
    gi = next(j for j, g in enumerate(w["groups"]) if i in g["ids"])
    w["pending"] = (i, dest, old, gi)
    return "ACTION6", int(dest[1]), int(dest[0])


def _confirmation_move(w):
    for i, h in sorted(enumerate(w["handles"]),
                       key=lambda z: z[0] != w["selected"]):
        gi = next((j for j, g in enumerate(w["groups"])
                   if i in g["ids"]), None)
        if gi is None:
            continue
        g = w["groups"][gi]
        for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3),
                       (4, 0), (-4, 0), (0, 4), (0, -4)):
            p = (h["p"][0] + dx, h["p"][1] + dy)
            if not (2 <= p[0] < 62 and 2 <= p[1] < 62):
                continue
            if not w["free_anchor"][p[1], p[0]]:
                continue
            if any(j != i and max(abs(p[0] - hh["p"][0]),
                                  abs(p[1] - hh["p"][1])) <= 2
                   for j, hh in enumerate(w["handles"])):
                continue
            c = _mean([p if j == i else w["handles"][j]["p"]
                       for j in g["ids"]])
            if not w["safe_loose"][c[1], c[0]]:
                continue
            if g["goal"] is not None and _touch(
                    c, w["goals"][g["goal"]]["p"], _GOAL_TOUCH):
                return [(i, p)]
    return []


def _reset(memory):
    memory["resets"] = memory.get("resets", 0) + 1
    memory.pop("world", None)
    return "RESET", None, None


def act(obs, available, memory):
    o = np.asarray(obs, dtype=np.int16)

    # RESET is a global game action. The left-hand action bar distinguishes
    # a warm state from a fresh level, where resetting could be unnecessary.
    if not memory.get("booted", False):
        memory["booted"] = True
        if int(o[0, 0]) == 5 and np.any(o[:, 0] == 0):
            return _reset(memory)

    if "world" not in memory:
        memory["world"] = _observe(o)
    w = memory["world"]
    _update(w, o)

    # Do not spend the remaining hazard strikes repeating a bad model.
    if (w["failed_move"] or w["uncertain"]) and memory.get("resets", 0) < 2:
        if int(o[0, 0]) == 5:
            return _reset(memory)

    if not w["handles"] or not w["groups"] or not w["goals"]:
        if memory.get("resets", 0) < 2 and int(o[0, 0]) == 5:
            return _reset(memory)
        return "ACTION6", 32, 32

    if w["plan"]:
        return _execute(w)

    if any(g["goal"] is None for g in w["groups"]):
        if not _assign(w):
            used = set()
            for i, g in enumerate(w["groups"]):
                choices = [c for c in _mission_choices(w, i) if c[1] not in used]
                if choices:
                    _, g["goal"], seq = choices[0]
                    g["reserved"] = list(seq)
                    used.add(g["goal"])

    unfinished = [i for i in range(len(w["groups"])) if not _done(w, i)]
    if not unfinished:
        w["confirm"] += 1
        w["plan"] = _confirmation_move(w)
        if w["plan"]:
            return _execute(w)
        x, y = w["handles"][w["selected"]]["p"]
        return "ACTION6", int(y), int(x)

    active = w.get("active")
    if active not in unfinished:
        active = next((i for i in unfinished
                       if w["selected"] in w["groups"][i]["ids"]), None)
    if active is None:
        estimates = []
        for i in unfinished:
            choices = _mission_choices(w, i, fixed=True)
            if choices:
                estimates.append((choices[0][0], i))
        active = min(estimates)[1] if estimates else unfinished[0]
    w["active"] = active

    excluded = set()
    for i, gg in enumerate(w["groups"]):
        if i != active and not _done(w, i):
            excluded.update(gg.get("reserved", []))
    choices = _mission_choices(w, active, fixed=True, excluded=excluded)
    if not choices:
        choices = _mission_choices(w, active, fixed=True)
    if not choices and _assign(w):
        choices = _mission_choices(w, active, fixed=True)

    if choices:
        _, ti, seq = choices[0]
        g = w["groups"][active]
        g["goal"] = ti
        g["reserved"] = list(seq)
        if seq:
            pickup_id = seq[0]
            target = w["picks"][pickup_id]["p"]
            is_pickup = True
        else:
            pickup_id = None
            target = w["goals"][ti]["p"]
            is_pickup = False

        plan = _search(w, active, target, is_pickup, pickup_id)
        if not plan:
            plan = _search(w, active, target, is_pickup, pickup_id,
                           width=64, depth_limit=12, branch=12, loose=True)
        if plan:
            w["plan"] = plan
            return _execute(w)

    if memory.get("resets", 0) < 2 and int(o[0, 0]) == 5:
        return _reset(memory)

    g = w["groups"][active]
    target = (w["goals"][g["goal"]]["p"]
              if g["goal"] is not None else (32, 32))
    best = None
    ys, xs = np.nonzero(w["free_anchor"])
    for i in g["ids"]:
        others = [w["handles"][j]["p"] for j in g["ids"] if j != i]
        n = len(g["ids"])
        sx, sy = sum(p[0] for p in others), sum(p[1] for p in others)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if any(abs(x - h["p"][0]) <= 3 and abs(y - h["p"][1]) <= 3
                   for h in w["handles"]):
                continue
            c = ((sx + x) // n, (sy + y) // n)
            if not w["safe_loose"][c[1], c[0]]:
                continue
            if g["black"] and any(_touch(c, p["p"], _PICK_TOUCH)
                                  for p in w["picks"].values()):
                continue
            score = (abs(c[0] - target[0]) + abs(c[1] - target[1])
                     + 2 * (i != w["selected"]))
            if best is None or score < best[0]:
                best = (score, i, (x, y))
    if best is not None:
        w["plan"] = [(best[1], best[2])]
        return _execute(w)
    x, y = w["handles"][w["selected"]]["p"]
    return "ACTION6", int(y), int(x)

import math
import time
import heapq
import itertools
import numpy as np

# Types 0..8 are square pieces; 9..11 are the three outlined pieces.
_COL = (10, 6, 15, 11, 12, 8, 9, 7, 14, 7, 14, 13)
_W = (1, 2, 3, 4, 5, 7, 8, 9, 10, 5, 5, 5)
_H = (1, 2, 3, 4, 5, 7, 8, 9, 10, 4, 4, 4)
_GMASK = (
    ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
    ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2), (4, 2),
     (1, 3), (3, 3)),
    ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
)
_GSETS = tuple(set(m) for m in _GMASK)
_DIR8 = ((1, 0), (1, 1), (0, 1), (-1, 1),
         (-1, 0), (-1, -1), (0, -1), (1, -1))
_DIR16 = _DIR8 + ((2, 1), (1, 2), (-1, 2), (-2, 1),
                 (-2, -1), (-1, -2), (1, -2), (2, -1))


def _center(b):
    t, x, y = b
    return x + _W[t] // 2, y + _H[t] // 2


def _affected(b, x, y):
    t, bx, by = b
    dx = max(bx - x, 0, x - (bx + _W[t] - 1))
    dy = max(by - y, 0, y - (by + _H[t] - 1))
    return dx * dx + dy * dy <= 64


def _overlap(a, b):
    ta, xa, ya = a
    tb, xb, yb = b
    return (xa < xb + _W[tb] and xb < xa + _W[ta] and
            ya < yb + _H[tb] and yb < ya + _H[ta])


def _components(mask, diagonal=True):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    dirs = _DIR8 if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    for yy, xx in zip(*np.nonzero(mask)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        todo = [(xx, yy)]
        cells = []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    todo.append((nx, ny))
        result.append(cells)
    return result


def _read_goals(obs):
    mask = np.zeros((10, 64), dtype=bool)
    mask[:, 24:] = (obs[:10, 24:] != 4) & (obs[:10, 24:] != 5)
    goals = {}
    side_to_type = {s: i for i, s in enumerate(_W[:9])}
    for cells in _components(mask):
        x0 = min(p[0] for p in cells)
        x1 = max(p[0] for p in cells)
        y0 = min(p[1] for p in cells)
        y1 = max(p[1] for p in cells)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        shape = {(x - x0, y - y0) for x, y in cells}
        typ = None
        if w == h and w in side_to_type and len(cells) == w * h:
            typ = side_to_type[w]
        elif w == 5 and h == 4:
            errors = [len(shape.symmetric_difference(s)) for s in _GSETS]
            k = min(range(3), key=lambda z: errors[z])
            if errors[k] <= 2:
                typ = 9 + k
        if typ is not None:
            goals[typ] = goals.get(typ, 0) + 1
    return tuple(sorted(goals.items()))


def _read_board(obs):
    a = np.asarray(obs)
    occupied = np.zeros((64, 64), dtype=bool)
    pieces = []

    # Fit squares, allowing another square to cover a small part of one.
    fruit_colors = set(_COL[:9])
    for t in range(8, -1, -1):
        size, color = _W[t], _COL[t]
        if size == 1:
            for y, x in zip(*np.nonzero(a[10:63] == color)):
                pieces.append((t, int(x), int(y) + 10))
                occupied[int(y) + 10, int(x)] = True
            continue

        mask = (a == color).astype(np.int32)
        integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        counts = (integral[size:, size:] - integral[:-size, size:] -
                  integral[size:, :-size] + integral[:-size, :-size])
        threshold = size * size if t == 6 else math.ceil(size * size * .76)
        candidates = []
        ys, xs = np.nonzero(counts[10:] >= threshold - size)
        for y, x in zip(ys, xs):
            y, x = int(y) + 10, int(x)
            if y + size > 64:
                continue
            block = a[y:y + size, x:x + size]
            known = block if y + size < 64 else block[:-1]
            nmatch = int(np.count_nonzero(known == color))
            if t == 6:
                valid = bool(np.all(known == color))
            else:
                valid = nmatch >= math.ceil(known.size * .76)
                if valid:
                    vals = set(int(v) for v in np.unique(known))
                    valid = vals.issubset(fruit_colors)
            if valid:
                candidates.append((-nmatch, y, x))
        candidates.sort()
        used = np.zeros((64, 64), dtype=bool)
        for negcount, y, x in candidates:
            matching = (a[y:y + size, x:x + size] == color)
            if np.any(used[y:y + size, x:x + size] & matching):
                continue
            pieces.append((t, x, y))
            used[y:y + size, x:x + size] |= matching
            occupied[y:y + size, x:x + size] = True

    # Outlined pieces can be partly hidden behind squares.
    for k, points in enumerate(_GMASK):
        t, color = k + 9, _COL[k + 9]
        hits = []
        for y in range(10, 61):
            for x in range(60):
                matched = 0
                bad = False
                for dx, dy in points:
                    xx, yy = x + dx, y + dy
                    if occupied[yy, xx] or yy == 63:
                        continue
                    if a[yy, xx] != color:
                        bad = True
                        break
                    matched += 1
                if bad or matched < 3:
                    continue
                for dy in range(4):
                    for dx in range(5):
                        if (dx, dy) in _GSETS[k]:
                            continue
                        xx, yy = x + dx, y + dy
                        if not occupied[yy, xx] and yy != 63 and a[yy, xx] == color:
                            bad = True
                            break
                    if bad:
                        break
                if not bad:
                    hits.append((-matched, x, y))
        hits.sort()
        chosen = []
        for _, x, y in hits:
            b = (t, x, y)
            if any(_overlap(b, old) for old in chosen):
                continue
            chosen.append(b)
            pieces.append(b)
            for dx, dy in points:
                occupied[y + dy, x + dx] = True

    # Static target areas: rounded 9-by-9 shapes, possibly occluded or red.
    holes = []
    proposals = []
    circle = {(x, y) for y in range(9) for x in range(9)
              if (2 <= x <= 6 if y in (0, 8)
                  else 1 <= x <= 7 if y in (1, 7) else True)}
    for y in range(10, 56):
        for x in range(56):
            score = 0
            bad = 0
            for dy in range(9):
                for dx in range(9):
                    yy, xx = y + dy, x + dx
                    if occupied[yy, xx] or yy == 63:
                        continue
                    v = int(a[yy, xx])
                    inside = (dx, dy) in circle
                    if v in (9, 2):
                        if inside:
                            score += 1
                        else:
                            bad += 1
                    elif inside and v not in (0, 3):
                        bad += 1
                    if bad > 0:
                        break
                if bad:
                    break
            if bad == 0 and score >= 8:
                proposals.append((-score, x, y))
    proposals.sort()
    for _, x, y in proposals:
        if any(abs(x - hx) < 9 and abs(y - hy) < 9 for hx, hy in holes):
            continue
        holes.append((x, y))

    if not holes:
        mask = ((a == 9) | (a == 2)) & ~occupied
        mask[:10] = False
        mask[63] = False
        for cells in _components(mask):
            if len(cells) < 10:
                continue
            x0 = min(x for x, y in cells)
            x1 = max(x for x, y in cells)
            y0 = min(y for x, y in cells)
            y1 = max(y for x, y in cells)
            if x1 - x0 <= 9 and y1 - y0 <= 9:
                holes.append((max(0, min(55, (x0 + x1) // 2 - 4)),
                              max(10, min(55, (y0 + y1) // 2 - 4))))
    return tuple(sorted(set(pieces))), tuple(sorted(set(holes)))


def _prediction_matches(state, obs):
    canvas = np.full((64, 64), -1, dtype=np.int16)
    ghosts = [b for b in state if b[0] >= 9]
    fruits = [b for b in state if b[0] < 9]
    for t, x, y in ghosts:
        for dx, dy in _GMASK[t - 9]:
            if 0 <= y + dy < 63 and 0 <= x + dx < 64:
                canvas[y + dy, x + dx] = _COL[t]
    # These are the layers of ordinary, newly constructed squares.
    fruits.sort(key=lambda b: 10 if b[0] == 0 else
                9 if b[0] == 1 else 2 if b[0] == 8 else 0)
    for t, x, y in fruits:
        canvas[y:min(63, y + _H[t]), x:x + _W[t]] = _COL[t]
    visible = canvas >= 0
    if np.any(obs[visible] != canvas[visible]):
        return False
    special = np.isin(obs, (10, 6, 15, 11, 12, 8, 7, 14, 13))
    special[:10] = False
    special[63] = False
    return not np.any(special & (canvas != obs))


def _make_simulator(goals, holes):
    goal_dict = dict(goals)
    required_types = set(goal_dict)

    def won(bs):
        counts = {}
        for t, x, y in bs:
            if t not in required_types:
                continue
            cx, cy = x + _W[t] // 2, y + _H[t] // 2
            if any(hx <= cx < hx + 9 and hy <= cy < hy + 9
                   for hx, hy in holes):
                counts[t] = counts.get(t, 0) + 1
        return bool(goals and holes) and all(
            counts.get(t, 0) == n for t, n in goals)

    def simulate(state, click):
        px, py = click
        bs = [list(b) for b in state]
        selected = [i for i, b in enumerate(bs) if _affected(b, px, py)]
        selset = set(selected)
        velocity = {}
        for i in selected:
            t, x, y = bs[i]
            if t == 9:
                cx, cy = x + 2, y + 2
                dx, dy = px - cx, py - cy
                d = math.hypot(dx, dy)
                velocity[i] = (3.4 * dx / d, 3.4 * dy / d) if d else (0., 0.)

        # bite: age, original x/y, unit direction x/y, float x/y, old type
        bites = {}
        cooldown = {}

        def clamp(b):
            t = b[0]
            b[1] = max(0, min(64 - _W[t], b[1]))
            b[2] = max(10, min(64 - _H[t], b[2]))

        def merge(ids, ghost):
            ids = [i for i in ids if bs[i][0] >= 0]
            n = len(ids)
            if n < 2:
                return True
            parent = list(range(n))

            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for j in range(n):
                a = bs[ids[j]]
                for k in range(j):
                    b = bs[ids[k]]
                    if not _overlap(a, b):
                        continue
                    if a[0] != b[0]:
                        return False
                    rj, rk = root(j), root(k)
                    if rj != rk:
                        parent[rj] = rk
            groups = {}
            for j, i in enumerate(ids):
                groups.setdefault(root(j), []).append(i)
            for group in groups.values():
                if len(group) < 2:
                    continue
                typ = bs[group[0]][0]
                cx = sum(bs[i][1] + _W[typ] // 2 for i in group) // len(group)
                cy = sum(bs[i][2] + _H[typ] // 2 for i in group) // len(group)
                for i in group:
                    bs[i][0] = -1
                    bites.pop(i, None)
                    cooldown.pop(i, None)
                if typ == (11 if ghost else 8):
                    continue
                nt = typ + 1
                b = [nt, cx - _W[nt] // 2, cy - _H[nt] // 2]
                clamp(b)
                ni = len(bs)
                bs.append(b)
                if ghost:
                    cooldown[ni] = 4
            return True

        did_merge = False
        for tick in range(28):
            frozen = any(4 <= v[0] < 8 and v[7] > 0 for v in bites.values())

            for i in selected:
                b = bs[i]
                t = b[0]
                if t < 0 or i in bites or (t >= 9 and frozen):
                    continue
                if i in velocity:
                    vx, vy = velocity[i]
                    b[1] = int(round(b[1] + vx))
                    b[2] = int(round(b[2] + vy))
                else:
                    cx = b[1] + _W[t] // 2
                    cy = b[2] + _H[t] // 2
                    b[1] += max(-4, min(4, px - cx))
                    b[2] += max(-4, min(4, py - cy))
                clamp(b)

            for i in list(bites):
                v = bites[i]
                age, ox, oy, vx, vy, fx, fy, old = v
                if age < 4:
                    yy = oy - 1 if age == 1 else oy + 1 if age == 3 else oy
                    bs[i][1], bs[i][2] = ox, yy
                else:
                    if age == 4:
                        bs[i][0] = old - 1
                    fx += vx * 2.5
                    fy += vy * 2.5
                    v[5], v[6] = fx, fy
                    bs[i][1], bs[i][2] = int(round(fx)), int(round(fy))
                clamp(bs[i])
                v[0] += 1
                if v[0] >= (4 if old == 0 else 8):
                    if old == 0:
                        bs[i][0] = -1
                    del bites[i]

            for i in list(cooldown):
                cooldown[i] -= 1
                if cooldown[i] <= 0:
                    del cooldown[i]

            fruits = [i for i, b in enumerate(bs) if 0 <= b[0] < 9]
            ghosts = [i for i, b in enumerate(bs) if b[0] >= 9]

            if tick < 4 and fruits and ghosts and not frozen:
                for i in ghosts:
                    if i in selset or i in cooldown:
                        continue
                    t, x, y = bs[i]
                    cx, cy = x + 2, y + 2
                    target = None
                    nearest = 10 ** 9
                    for j in fruits:
                        if j in bites:
                            continue
                        ft, fx, fy = bs[j]
                        tx, ty = fx + _W[ft] // 2, fy + _H[ft] // 2
                        d = (cx - tx) ** 2 + (cy - ty) ** 2
                        if d < nearest:
                            nearest, target = d, (tx, ty)
                    if target is not None:
                        tx, ty = target
                        speed = 2 if t == 11 else 1
                        bs[i][1] += speed * ((tx > cx) - (tx < cx))
                        bs[i][2] += speed * ((ty > cy) - (ty < cy))
                        clamp(bs[i])

                for i in ghosts:
                    if i in selset or i in cooldown:
                        continue
                    gt, gx, gy = bs[i]
                    for j in fruits:
                        if j in bites or bs[j][0] < 0:
                            continue
                        if not _overlap(bs[i], bs[j]):
                            continue
                        ft, fx, fy = bs[j]
                        dx = fx + _W[ft] // 2 - (gx + 2)
                        dy = fy + _H[ft] // 2 - (gy + 2)
                        d = math.hypot(dx, dy)
                        vx, vy = (dx / d, dy / d) if d else (0., -1.)
                        bites[j] = [0, fx, fy, vx, vy, float(fx), float(fy), ft]
                        cooldown[i] = 9
                if not merge(ghosts, True):
                    return None, False

            if tick >= 3 and not did_merge:
                ghosts = [i for i, b in enumerate(bs) if b[0] >= 9]
                if not merge(ghosts, True):
                    return None, False
                ids = [i for i in selected if 0 <= bs[i][0] < 9]
                if not merge(ids, False):
                    return None, False
                did_merge = True

            if won(bs):
                return tuple(sorted(tuple(b) for b in bs if b[0] >= 0)), True
            if tick >= 3 and not bites:
                break

        result = tuple(sorted(tuple(b) for b in bs if b[0] >= 0))
        if not any(b[0] < 9 for b in result):
            return None, False
        return result, False

    return simulate, won


def _make_heuristic(goals, holes):
    fruit_goals = [(t, n) for t, n in goals if t < 9]
    ghost_goals = [(t, n) for t, n in goals if t >= 9]
    needed_mass = sum((1 << t) * n for t, n in fruit_goals)
    max_target = max((t for t, n in fruit_goals), default=0)

    def distance_hole(x, y):
        return min((math.hypot(max(hx - x, 0, x - hx - 8),
                               max(hy - y, 0, y - hy - 8))
                    for hx, hy in holes), default=50.)

    def gap(a, b):
        ta, xa, ya = a
        tb, xb, yb = b
        return math.hypot(max(xa - xb - _W[tb], xb - xa - _W[ta], 0),
                          max(ya - yb - _H[tb], yb - ya - _H[ta], 0))

    def creation_cost(state, target, quantity, ghost):
        groups = [[] for _ in range(12)]
        ghosts = [b for b in state if b[0] >= 9]
        low = 9 if ghost else 0
        for b in state:
            t, x, y = b
            if (t >= 9) != ghost:
                continue
            cx, cy = _center(b)
            cost = 0.
            if t > target:
                if ghost or not ghosts:
                    continue
                dg = min(gap(b, g) for g in ghosts)
                cost = 2.25 * (t - target) + max(0., dg - 4.) / 12.
                t = target
            groups[t].append((float(cx), float(cy), cost))

        # A geometric merge-tree relaxation.
        for t in range(low, target):
            group = groups[t]
            while len(group) >= 2:
                best = None
                best_pair = None
                for i in range(len(group)):
                    ax, ay, ac = group[i]
                    for j in range(i):
                        bx, by, bc = group[j]
                        d = math.hypot(ax - bx, ay - by)
                        rank = d + 1.5 * (ac + bc)
                        if best is None or rank < best:
                            best = rank
                            best_pair = (j, i, d)
                j, i, d = best_pair
                ax, ay, ac = group[i]
                bx, by, bc = group[j]
                del group[i]
                del group[j]
                mobility = 11.5 if ghost else 8. + _W[t] / 2.
                capture = 18. if ghost else 16. + _W[t] * .7
                mc = 1. + max(0., d - capture) / mobility
                groups[t + 1].append(((ax + bx) / 2., (ay + by) / 2.,
                                      ac + bc + mc))
        costs = []
        speed = 12. if target == 9 else 9. + _W[target] / 2.
        for x, y, c in groups[target]:
            costs.append(c + distance_hole(x, y) / speed)
        if len(costs) < quantity:
            return None
        costs.sort()
        return sum(costs[:quantity])

    def heuristic(state):
        mass = sum(1 << t for t, x, y in state if t < 9)
        if mass < needed_mass:
            return float("inf")
        total = 0.
        for t, n in fruit_goals:
            c = creation_cost(state, t, n, False)
            if c is None:
                return float("inf")
            total += c
        for t, n in ghost_goals:
            c = creation_cost(state, t, n, True)
            if c is None:
                return float("inf")
            total += .90 * c

        # Exact counts matter: an additional required-type piece in a hole
        # sometimes needs to be moved back out.
        for t, n in goals:
            inside = 0
            for b in state:
                if b[0] == t:
                    cx, cy = _center(b)
                    inside += any(hx <= cx < hx + 9 and hy <= cy < hy + 9
                                  for hx, hy in holes)
            total += 1.8 * max(0, inside - n)

        ghosts = [b for b in state if b[0] >= 9]
        if ghosts:
            spare = mass - needed_mass
            for b in state:
                t = b[0]
                if t >= 9 or t > max_target:
                    continue
                damage = 1 if t == 0 else 1 << (t - 1)
                danger = min(gap(b, g) for g in ghosts)
                weight = .12 if damage > spare else .025
                total += weight * max(0., 9. - danger)
        return total

    return heuristic


def _clicks(state, holes):
    points = set()

    def add(x, y):
        points.add((max(0, min(63, int(round(x)))),
                    max(10, min(62, int(round(y))))))

    def reach(b, tx, ty):
        tx, ty = max(0, min(63, tx)), max(10, min(62, ty))
        if _affected(b, tx, ty):
            add(tx, ty)
            return
        cx, cy = _center(b)
        dx, dy = tx - cx, ty - cy
        d = math.hypot(dx, dy)
        if not d:
            add(cx, cy)
            return
        for r in range(11 + max(_W[b[0]], _H[b[0]]) // 2, 0, -1):
            x = max(0, min(63, int(round(cx + dx * r / d))))
            y = max(10, min(62, int(round(cy + dy * r / d))))
            if _affected(b, x, y):
                add(x, y)
                return

    directions = _DIR16 if len(state) <= 4 else _DIR8
    for b in state:
        t, x, y = b
        cx, cy = _center(b)
        add(cx, cy)
        for dx, dy in directions:
            reach(b, cx + dx * 40, cy + dy * 40)
        if t >= 9 or len(state) <= 3:
            for dx, dy in _DIR8:
                add(cx + dx * 3, cy + dy * 3)
        for hx, hy in holes:
            tx, ty = max(hx, min(hx + 8, cx)), max(hy, min(hy + 8, cy))
            reach(b, tx, ty)
            reach(b, hx + 4, hy + 4)
            if _affected(b, hx + 4, hy + 4):
                add(hx + 4, hy + 4)

    for i, a in enumerate(state):
        ax, ay = _center(a)
        for b in state[:i]:
            bx, by = _center(b)
            same = a[0] == b[0]
            cross = (a[0] >= 9) != (b[0] >= 9)
            if not same and not cross:
                continue
            reach(a, bx, by)
            reach(b, ax, ay)
            if same and abs(ax - bx) <= 30 and abs(ay - by) <= 30:
                mx, my = (ax + bx) / 2., (ay + by) / 2.
                add(mx, my)
                add(math.floor(mx), math.floor(my))
                for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                    xx, yy = round(mx + dx), round(my + dy)
                    if _affected(a, xx, yy) and _affected(b, xx, yy):
                        add(xx, yy)
                for hx, hy in holes:
                    dx, dy = hx + 4 - mx, hy + 4 - my
                    d = math.hypot(dx, dy)
                    if d:
                        xx, yy = round(mx + dx * 4 / d), round(my + dy * 4 / d)
                        if _affected(a, xx, yy) and _affected(b, xx, yy):
                            add(xx, yy)

    for p in ((0, 10), (63, 10), (0, 62), (63, 62), (32, 10), (32, 62)):
        add(*p)
    return sorted(points)


def _search(state, goals, holes, history, deadline):
    simulate, won = _make_simulator(goals, holes)
    heuristic = _make_heuristic(goals, holes)
    hcache = {}

    def h(s):
        if s not in hcache:
            hcache[s] = heuristic(s)
        return hcache[s]

    initial_h = h(state)
    if not math.isfinite(initial_h):
        return None, None, None

    serial = itertools.count()
    heap = [(1.85 * initial_h, next(serial), 0, state, ())]
    best_g = {state: 0}
    best_partial = None
    partial_score = float("inf")
    expanded = 0
    simulations = 0

    while heap and time.perf_counter() < deadline:
        _, _, g, s, path = heapq.heappop(heap)
        if best_g.get(s) != g:
            continue
        if g >= 26:
            continue
        expanded += 1
        children = {}
        for click in _clicks(s, holes):
            simulations += 1
            if simulations % 64 == 0 and time.perf_counter() >= deadline:
                break
            ns, success = simulate(s, click)
            if ns is None:
                continue
            npth = path + (click,)
            if success:
                return npth, True, simulate
            if ns == s:
                continue
            ng = g + 1
            if best_g.get(ns, 10 ** 9) <= ng:
                continue
            nh = h(ns)
            if not math.isfinite(nh):
                continue
            repeated = history.get(ns, 0)
            score = nh + .20 * ng + .9 * repeated
            if score < partial_score:
                partial_score = score
                best_partial = npth
            rank = ng + 1.85 * nh + .9 * repeated
            old = children.get(ns)
            if old is None or rank < old[0]:
                children[ns] = (rank, ng, npth)

        # Keep several alternatives, rather than only the locally best click.
        ordered = sorted(children.items(), key=lambda item: item[1][0])
        limit = 24 if len(state) <= 4 else 18
        for ns, (rank, ng, npth) in ordered[:limit]:
            if best_g.get(ns, 10 ** 9) <= ng:
                continue
            best_g[ns] = ng
            heapq.heappush(heap, (rank, next(serial), ng, ns, npth))

        if expanded >= 1100 or simulations >= 65000:
            break

    if best_partial:
        return best_partial, False, simulate

    # This also covers a state already satisfying the target after an undo.
    if won(state):
        return ((0, 10),), False, simulate
    return None, False, simulate


def act(obs, available, memory):
    started = time.perf_counter()
    obs = np.asarray(obs)
    names = set(available)

    def reset_or_undo():
        old_resets = memory.get("resets", 0)
        memory.clear()
        memory["resets"] = old_resets + 1
        if "RESET" in names:
            return "RESET", None, None
        if "ACTION7" in names:
            return "ACTION7", None, None
        return "ACTION6", 10, 0

    if "ACTION6" not in names:
        if "ACTION7" in names:
            return "ACTION7", None, None
        if "RESET" in names:
            return "RESET", None, None
        return (available[0], None, None)

    predicted = memory.get("predicted")
    matched = predicted is not None and _prediction_matches(predicted, obs)
    if matched:
        state = predicted
        holes = memory.get("holes", ())
        goals = memory.get("goals", ())
    else:
        state, detected_holes = _read_board(obs)
        goals = memory.get("goals") or _read_goals(obs)
        holes = memory.get("holes") or detected_holes
        memory.pop("plan", None)

    if not goals:
        goals = _read_goals(obs)
    if not holes:
        _, holes = _read_board(obs)
    if not goals or not holes or not state:
        return reset_or_undo()

    memory["goals"] = goals
    memory["holes"] = holes
    history = memory.setdefault("history", {})
    history[state] = history.get(state, 0) + 1

    simulate, _ = _make_simulator(goals, holes)
    saved = memory.get("plan")
    if matched and saved:
        click = saved[0]
        ns, success = simulate(state, click)
        if ns is not None:
            memory["plan"] = saved[1:]
            memory["predicted"] = ns
            return "ACTION6", int(click[1]), int(click[0])
        memory.pop("plan", None)

    # Leave time for parsing, synchronization and returning the result.
    deadline = started + 8.35
    plan, complete, sim = _search(state, goals, holes, history, deadline)
    if not plan:
        return reset_or_undo()

    click = plan[0]
    sim = sim or simulate
    ns, success = sim(state, click)
    if ns is None:
        return reset_or_undo()

    memory["predicted"] = ns
    memory["plan"] = plan[1:] if complete else ()
    memory["turn"] = memory.get("turn", 0) + 1

    # Avoid retaining a needlessly large transposition history.
    if len(history) > 250:
        memory["history"] = {state: history[state]}
    return "ACTION6", int(click[1]), int(click[0])

import math
import time
import heapq
from collections import Counter
import numpy as np


COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
RANKS = {c: i for i, c in enumerate(COLORS)}
GHOST_COLORS = (7, 14, 13)
GHOST_MASKS = (
    ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
    ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2),
     (4, 2), (1, 3), (3, 3)),
    ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
)
DISK = np.array([
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
], dtype=bool)


def components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    h, w = mask.shape
    for yy, xx in zip(*np.nonzero(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < w and 0 <= ny < h
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
        out.append(comp)
    return out


def read_requirements(a):
    req = Counter()
    for color in range(16):
        if color in (4, 5, 3):
            continue
        mask = a == color
        mask[10:, :] = False
        mask[:, :24] = False
        for comp in components(mask):
            xs, ys = zip(*comp)
            x0, y0 = min(xs), min(ys)
            w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
            shape = {(x - x0, y - y0) for x, y in comp}
            if w == h and len(comp) >= .85 * w * h:
                if color in RANKS:
                    rank = RANKS[color]
                elif w in SIZES:
                    rank = SIZES.index(w)
                else:
                    continue
                req[rank] += 1
            elif w <= 5 and h <= 4 and len(comp) >= 5:
                if color in GHOST_COLORS:
                    rank = 9 + GHOST_COLORS.index(color)
                else:
                    rank = 9 + min(
                        range(3),
                        key=lambda k: len(
                            shape.symmetric_difference(set(GHOST_MASKS[k])))
                    )
                req[rank] += 1
    return tuple(sorted(req.items()))


def read_targets(a):
    targets = []
    known_fruit = []
    mask = (a == 9) | (a == 2)
    mask[:10] = False
    mask[63:] = False

    for comp in components(mask):
        xs, ys = zip(*comp)
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if (w == 8 and h == 8 and len(comp) == 64
                and a[y0, x0] == 9):
            known_fruit.append((6, x0, y0))
            for x, y in comp:
                mask[y, x] = False

    for unused in range(10):
        comps = components(mask)
        candidates = set()
        for comp in comps:
            if len(comp) < 15:
                continue
            xs, ys = zip(*comp)
            for y in range(max(10, min(ys) - 8), min(55, max(ys)) + 1):
                for x in range(max(0, min(xs) - 8), min(55, max(xs)) + 1):
                    candidates.add((x, y))
        best = None
        best_score = 20
        for x, y in candidates:
            patch = mask[y:y + 9, x:x + 9]
            inside = int(np.count_nonzero(patch & DISK))
            outside = int(np.count_nonzero(patch & ~DISK))
            score = inside - 3 * outside
            if score > best_score:
                best_score, best = score, (x, y)
        if best is None:
            break
        x, y = best
        targets.append((x, y, 9, 9))
        mask[y:y + 9, x:x + 9] = False
    return tuple(sorted(targets)), known_fruit


def read_state(a, targets, known_fruit=()):
    work = a.copy()
    ghosts = []
    for j, color in enumerate(GHOST_COLORS):
        pattern = GHOST_MASKS[j]
        negative = [(x, y) for y in range(4) for x in range(5)
                    if (x, y) not in pattern]
        candidates = []
        ys, xs = np.nonzero(work[10:63] == color)
        origins = set()
        for yy, xx in zip(ys, xs):
            px, py = int(xx), int(yy) + 10
            for dx, dy in pattern:
                x, y = px - dx, py - dy
                if 0 <= x <= 59 and 10 <= y <= 60:
                    origins.add((x, y))
        for x, y in origins:
            pos = sum(work[y + dy, x + dx] == color for dx, dy in pattern)
            neg = sum(work[y + dy, x + dx] == color for dx, dy in negative)
            if pos >= len(pattern) - 2 and neg <= 1:
                candidates.append((int(pos) * 4 - int(neg) * 5, x, y))
        candidates.sort(reverse=True)
        used = []
        for score, x, y in candidates:
            if any(abs(x - u) < 4 and abs(y - v) < 3 for u, v in used):
                continue
            used.append((x, y))
            ghosts.append((9 + j, x, y))
            for dx, dy in pattern:
                if work[y + dy, x + dx] == color:
                    work[y + dy, x + dx] = 5

    fruits = list(known_fruit)
    for t, color in enumerate(COLORS):
        mask = work == color
        mask[:10] = False
        mask[63:] = False
        if color == 9:
            for x, y, w, h in targets:
                mask[y:y + h, x:x + w] = False
            for unused, x, y in known_fruit:
                mask[y:y + 8, x:x + 8] = False

        if t == 0:
            for y, x in zip(*np.nonzero(mask)):
                fruits.append((0, int(x), int(y)))
            continue

        s = SIZES[t]
        for comp in components(mask):
            xs, ys = zip(*comp)
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            if x1 - x0 + 1 == s and y1 - y0 + 1 == s:
                if len(comp) >= .65 * s * s:
                    fruits.append((t, x0, y0))
                    continue
            points = set(comp)
            candidates = []
            for y in range(max(10, y0 - s + 1), min(64 - s, y1) + 1):
                for x in range(max(0, x0 - s + 1), min(64 - s, x1) + 1):
                    hits = {(xx, yy) for xx, yy in points
                            if x <= xx < x + s and y <= yy < y + s}
                    if len(hits) < max(2, .45 * s * s):
                        continue
                    patch = work[y:y + s, x:x + s]
                    bad = (patch == 5) | (patch == 4) | (patch == 3)
                    if y + s == 64:
                        bad = bad.copy()
                        bad[-1] = False
                    if np.any(bad):
                        continue
                    candidates.append((x, y, hits))
            uncovered = set(points)
            while uncovered and candidates:
                x, y, hits = max(
                    candidates, key=lambda q: len(q[2] & uncovered))
                if len(hits & uncovered) < max(1, s):
                    break
                fruits.append((t, x, y))
                uncovered.difference_update(hits)
                candidates = [q for q in candidates
                              if q[0] != x or q[1] != y]

    fruits = sorted(set(fruits), key=lambda q: (q[1], q[2], q[0]))
    ghosts = sorted(set(ghosts))
    return tuple(fruits + ghosts)


def compatible(a, state, targets):
    image = np.full((64, 64), -1, dtype=np.int16)
    for t, x, y in state:
        if t >= 9:
            for dx, dy in GHOST_MASKS[t - 9]:
                if 0 <= x + dx < 64 and 0 <= y + dy < 63:
                    image[y + dy, x + dx] = GHOST_COLORS[t - 9]

    def layer(q):
        t = q[0]
        return 10 if t == 0 else 9 if t == 1 else 2 if t == 8 else 0

    for t, x, y in sorted((q for q in state if q[0] < 9), key=layer):
        s = SIZES[t]
        image[max(10, y):min(63, y + s),
              max(0, x):min(64, x + s)] = COLORS[t]
    painted = image >= 0
    if np.any(a[painted] != image[painted]):
        return False
    actual = np.isin(a, list(COLORS) + [13])
    actual[:10] = False
    actual[63:] = False
    for x, y, w, h in targets:
        patch = actual[y:y + h, x:x + w]
        patch[a[y:y + h, x:x + w] == 9] = False
    return not np.any(actual & ~painted)


# Ordinary attributes only: no special constructor or object-layout names.
class Body:
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def place(self, x, y):
        self.x = max(0, min(64 - self.w, int(x)))
        self.y = max(10, min(64 - self.h, int(y)))


def make_body(t, x, y):
    b = Body()
    b.t, b.x, b.y = t, x, y
    b.w = SIZES[t] if t < 9 else 5
    b.h = b.w if t < 9 else 4
    b.sel = False
    b.vx = b.vy = 0.0
    b.hit = -1
    b.ox = b.oy = b.ux = b.uy = 0.0
    b.fx = b.fy = 0.0
    b.cool = 0
    return b


def near(x, y, b):
    dx = max(b.x - x, 0, x - (b.x + b.w - 1))
    dy = max(b.y - y, 0, y - (b.y + b.h - 1))
    return dx * dx + dy * dy <= 64


def overlap(a, b):
    return (a.x < b.x + b.w and b.x < a.x + a.w
            and a.y < b.y + b.h and b.y < a.y + a.h)


def inside(x, y, targets):
    return any(tx <= x < tx + w and ty <= y < ty + h
               for tx, ty, w, h in targets)


def won_bodies(fruits, ghosts, targets, req):
    count = {}
    for b in fruits:
        if inside(b.x + b.w // 2, b.y + b.h // 2, targets):
            count[b.t] = count.get(b.t, 0) + 1
    for b in ghosts:
        if inside(b.x + 2, b.y + 2, targets):
            count[b.t] = count.get(b.t, 0) + 1
    return all(count.get(t, 0) == n for t, n in req)


def merge_bodies(bodies, selected_only=False, creature=False):
    active = [b for b in bodies if not selected_only or b.sel]
    n = len(active)
    if n < 2:
        return bodies, True
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = active[i], active[j]
            if overlap(a, b):
                if a.t != b.t:
                    return bodies, False
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i, b in enumerate(active):
        groups.setdefault(root(i), []).append(b)

    removed = set()
    new = []
    for group in groups.values():
        if len(group) < 2:
            continue
        removed.update(group)
        t = group[0].t
        if t == (11 if creature else 8):
            continue
        cx = sum(b.x + b.w // 2 for b in group) // len(group)
        cy = sum(b.y + b.h // 2 for b in group) // len(group)
        b = make_body(t + 1, 0, 10)
        b.place(cx - b.w // 2, cy - b.h // 2)
        if creature:
            b.cool = 4
        new.append(b)
    if removed:
        bodies = [b for b in bodies if b not in removed] + new
    return bodies, True


def pack(fruits, ghosts):
    return tuple((b.t, b.x, b.y) for b in fruits + ghosts)


def simulate(state, action, targets, req):
    x, y = action
    fruits, ghosts = [], []
    for t, bx, by in state:
        b = make_body(t, bx, by)
        b.sel = near(x, y, b)
        if t < 9:
            fruits.append(b)
        else:
            ghosts.append(b)
            if b.sel and t == 9:
                cx, cy = b.center()
                dx, dy = x - cx, y - cy
                d = math.hypot(dx, dy)
                if d:
                    b.vx, b.vy = 3.4 * dx / d, 3.4 * dy / d

    if not ghosts:
        for tick in range(4):
            for b in fruits:
                if b.sel:
                    cx, cy = b.center()
                    b.place(b.x + max(-4, min(4, x - cx)),
                            b.y + max(-4, min(4, y - cy)))
            if tick == 3:
                fruits, ok = merge_bodies(fruits, True, False)
                if not ok:
                    return None, False
            if won_bodies(fruits, ghosts, targets, req):
                return pack(fruits, ghosts), True
        return (pack(fruits, ghosts), False) if fruits else (None, False)

    merged = False
    for tick in range(35):
        pause = any(b.t > 0 and 4 <= b.hit < 8 for b in fruits)

        for b in fruits:
            if b.sel and b.hit < 0:
                cx, cy = b.center()
                b.place(b.x + max(-4, min(4, x - cx)),
                        b.y + max(-4, min(4, y - cy)))
        if not pause:
            for b in ghosts:
                if not b.sel:
                    continue
                if b.t == 9:
                    b.place(round(b.x + b.vx), round(b.y + b.vy))
                else:
                    cx, cy = b.center()
                    b.place(b.x + max(-4, min(4, x - cx)),
                            b.y + max(-4, min(4, y - cy)))

        gone = []
        for b in fruits:
            if b.hit < 0:
                continue
            phase = b.hit
            if phase < 4:
                sy = -1 if phase == 1 else 1 if phase == 3 else 0
                b.place(round(b.ox), round(b.oy + sy))
            else:
                if phase == 4:
                    b.t -= 1
                    b.w = b.h = SIZES[b.t]
                b.fx += b.ux * 2.5
                b.fy += b.uy * 2.5
                b.place(round(b.fx), round(b.fy))
            b.hit += 1
            if phase == 3 and b.t == 0:
                gone.append(b)
            elif b.hit >= 8:
                b.hit = -1
        if gone:
            fruits = [b for b in fruits if b not in gone]

        for b in ghosts:
            if b.cool > 0:
                b.cool -= 1

        if tick < 4 and fruits and ghosts and not pause:
            eligible = [b for b in ghosts if not b.sel and b.cool <= 0]
            for b in eligible:
                prey = [f for f in fruits if f.hit < 0]
                if not prey:
                    break
                cx, cy = b.center()
                f = min(prey, key=lambda q:
                        (q.x + q.w // 2 - cx) ** 2
                        + (q.y + q.h // 2 - cy) ** 2)
                tx, ty = f.center()
                speed = 2 if b.t == 11 else 1
                dx = speed if tx > cx else -speed if tx < cx else 0
                dy = speed if ty > cy else -speed if ty < cy else 0
                b.place(b.x + dx, b.y + dy)

            for b in eligible:
                for f in fruits:
                    if f.hit >= 0 or not overlap(b, f):
                        continue
                    fx, fy = f.center()
                    ex, ey = b.center()
                    dx, dy = fx - ex, fy - ey
                    d = math.hypot(dx, dy)
                    if d:
                        ux, uy = dx / d, dy / d
                    else:
                        ux, uy = 0.0, -1.0
                    f.hit = 0
                    f.ox, f.oy = float(f.x), float(f.y)
                    f.fx, f.fy = float(f.x), float(f.y)
                    f.ux, f.uy = ux, uy
                    b.cool = 9

            ghosts, ok = merge_bodies(ghosts, False, True)
            if not ok:
                return None, False

        if tick + 1 >= 4 and not merged:
            ghosts, ok = merge_bodies(ghosts, False, True)
            if not ok:
                return None, False
            fruits, ok = merge_bodies(fruits, True, False)
            if not ok:
                return None, False
            merged = True

        if won_bodies(fruits, ghosts, targets, req):
            return pack(fruits, ghosts), True
        if tick + 1 >= 4 and not any(b.hit >= 0 for b in fruits):
            return (pack(fruits, ghosts), False) if fruits else (None, False)
    return None, False


def distance_norm(dx, dy):
    dx, dy = abs(dx), abs(dy)
    return max(dx, dy) + .32 * min(dx, dy)


def target_distance(x, y, targets):
    best = 1000.0
    for tx, ty, w, h in targets:
        dx = max(tx - x, 0, x - (tx + w - 1))
        dy = max(ty - y, 0, y - (ty + h - 1))
        best = min(best, distance_norm(dx, dy))
    return best


def piece_center(q):
    t, x, y = q
    half = SIZES[t] // 2 if t < 9 else 2
    return x + half, y + half


def piece_gap(a, b):
    ta, ax, ay = a
    tb, bx, by = b
    wa, ha = (SIZES[ta], SIZES[ta]) if ta < 9 else (5, 4)
    wb, hb = (SIZES[tb], SIZES[tb]) if tb < 9 else (5, 4)
    dx = max(ax - (bx + wb), bx - (ax + wa), 0)
    dy = max(ay - (by + hb), by - (ay + ha), 0)
    return distance_norm(dx, dy)


def assembly_options(pieces, available_mask, target, creatures):
    levels = {}
    options = []
    for i, q in enumerate(pieces):
        if not (available_mask >> i) & 1:
            continue
        t, x, y = q
        cx, cy = piece_center(q)
        if t == target:
            options.append((0.0, cx, cy, 1 << i))
        elif t < target:
            levels.setdefault(t, []).append((0.0, cx, cy, 1 << i))

    low = 9 if creatures else 0
    for rank in range(low, target):
        nodes = levels.get(rank, [])
        while len(nodes) >= 2:
            best = None
            value = float("inf")
            reach = 12.0 if creatures else 8.0 + SIZES[rank] / 2
            together = 17.0 if creatures else 14.0 + SIZES[rank]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    d = distance_norm(a[1] - b[1], a[2] - b[2])
                    v = a[0] + b[0] + 1.0 + max(0.0, d - together) / reach
                    if v < value:
                        value, best = v, (i, j)
            i, j = best
            a, b = nodes[i], nodes[j]
            node = (value, (a[1] + b[1]) * .5,
                    (a[2] + b[2]) * .5, a[3] | b[3])
            nodes.pop(j)
            nodes.pop(i)
            if rank + 1 == target:
                options.append(node)
            else:
                levels.setdefault(rank + 1, []).append(node)
    return options


def heuristic(state, targets, req):
    fruits = [q for q in state if q[0] < 9]
    ghosts = [q for q in state if q[0] >= 9]
    needed_f = sorted(
        [t for t, n in req if t < 9 for unused in range(n)], reverse=True)
    needed_g = sorted(
        [t for t, n in req if t >= 9 for unused in range(n)], reverse=True)

    total = 0.0
    available_mask = (1 << len(fruits)) - 1
    for rank in needed_f:
        options = assembly_options(fruits, available_mask, rank, False)
        if ghosts:
            for i, q in enumerate(fruits):
                if not (available_mask >> i) & 1 or q[0] <= rank:
                    continue
                cx, cy = piece_center(q)
                approach = min(piece_gap(q, g) for g in ghosts) / 9.0
                options.append((1.85 * (q[0] - rank) + approach,
                                cx, cy, 1 << i))
        if not options:
            return 100000.0
        reach = 8.0 + SIZES[rank] / 2
        best = min(options, key=lambda z:
                   z[0] + target_distance(z[1], z[2], targets) / reach)
        total += best[0] + target_distance(best[1], best[2], targets) / reach
        available_mask &= ~best[3]

    available_mask = (1 << len(ghosts)) - 1
    for rank in needed_g:
        options = assembly_options(ghosts, available_mask, rank, True)
        if not options:
            return 100000.0
        reach = 12.0 if rank == 9 else 10.0
        best = min(options, key=lambda z:
                   z[0] + target_distance(z[1], z[2], targets) / reach)
        total += .9 * (
            best[0] + target_distance(best[1], best[2], targets) / reach)
        available_mask &= ~best[3]

    count = Counter()
    for q in state:
        cx, cy = piece_center(q)
        if inside(cx, cy, targets):
            count[q[0]] += 1
    for t, n in req:
        if count[t] > n:
            total += 3.0 * (count[t] - n)

    target_ranks = set(needed_f)
    for f in fruits:
        if f[0] in target_ranks and ghosts:
            gap = min(piece_gap(f, g) for g in ghosts)
            total += .12 * max(0.0, 7.0 - gap) / 7.0
    return total


def state_key(state, coarse=1):
    return tuple(sorted((t, x // coarse, y // coarse) for t, x, y in state))


def candidate_actions(state, targets):
    bodies = [make_body(*q) for q in state]
    result = []
    seen = set()

    def add(x, y, force=False):
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < 64 and 10 <= y < 63):
            return
        p = x, y
        if p in seen:
            return
        if not force and not any(near(x, y, b) for b in bodies):
            return
        seen.add(p)
        result.append(p)

    goalpoints = []
    for x, y, w, h in targets:
        goalpoints += [
            (x + w // 2, y + h // 2),
            (x, y), (x + w - 1, y),
            (x, y + h - 1), (x + w - 1, y + h - 1),
        ]
    for p in goalpoints:
        add(*p)

    centers = [b.center() for b in bodies]
    for i, b in enumerate(bodies):
        cx, cy = centers[i]
        add(cx, cy)
        intentions = list(goalpoints)
        intentions += [c for j, c in enumerate(centers) if i != j]
        for tx, ty in intentions:
            dx, dy = tx - cx, ty - cy
            distance = math.hypot(dx, dy)
            if not distance:
                continue
            if near(int(round(tx)), int(round(ty)), b):
                add(tx, ty)
            length = min(distance, 15.0)
            while length > 0:
                px = int(round(cx + dx * length / distance))
                py = int(round(cy + dy * length / distance))
                if (0 <= px < 64 and 10 <= py < 63 and near(px, py, b)):
                    add(px, py)
                    break
                length -= .65

        for k in range(16):
            angle = k * math.pi / 8
            dx, dy = math.cos(angle), math.sin(angle)
            for length in (12, 10, 8, 6):
                px, py = round(cx + dx * length), round(cy + dy * length)
                if near(px, py, b):
                    add(px, py)
                    break

    for i, a in enumerate(bodies):
        for j in range(i + 1, len(bodies)):
            b = bodies[j]
            if a.t != b.t:
                continue
            ax, ay = centers[i]
            bx, by = centers[j]
            mx, my = (ax + bx) / 2, (ay + by) / 2
            for dx, dy in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                add(mx + dx, my + dy)

    for b in bodies:
        x0, x1 = max(0, b.x - 8), min(63, b.x + b.w + 7)
        y0, y1 = max(10, b.y - 8), min(62, b.y + b.h + 7)
        for y in range(10 + ((y0 - 10 + 3) // 4) * 4, y1 + 1, 4):
            for x in range(((x0 + 3) // 4) * 4, x1 + 1, 4):
                if near(x, y, b):
                    add(x, y)

    if any(b.t >= 9 for b in bodies):
        for p in ((0, 10), (63, 10), (0, 62), (63, 62)):
            add(*p, force=True)
    return result


def search(state, targets, req, visits, banned, deadline):
    h0 = heuristic(state, targets, req)
    serial = 0
    heap = [(1.85 * h0, serial, 0, h0, state, ())]
    closed = {state_key(state, 2): 0}
    rootkey = state_key(state)
    best_path = ()
    best_score = float("inf")
    trials = 0

    while heap and time.perf_counter() < deadline:
        priority, unused, depth, h, current, path = heapq.heappop(heap)
        if depth >= 28:
            continue
        if depth > closed.get(state_key(current, 2), depth):
            continue
        local = set()

        for action in candidate_actions(current, targets):
            trials += 1
            if trials % 32 == 0 and time.perf_counter() >= deadline:
                return best_path, False
            if depth == 0 and action in banned:
                continue

            nxt, won = simulate(current, action, targets, req)
            if nxt is None:
                continue
            newpath = path + (action,)
            if won:
                return newpath, True

            exact = state_key(nxt)
            if exact == rootkey or exact in local:
                continue
            local.add(exact)
            key = state_key(nxt, 2)
            nd = depth + 1
            if closed.get(key, 100000) <= nd:
                continue

            nh = heuristic(nxt, targets, req)
            if nh >= 10000:
                continue
            closed[key] = nd
            repeat = min(6, visits.get(exact, 0))
            score = nh + .19 * nd + .85 * repeat
            if score < best_score:
                best_score, best_path = score, newpath

            serial += 1
            priority = nd + 1.85 * nh + 1.1 * repeat
            heapq.heappush(heap, (priority, serial, nd, nh, nxt, newpath))

        if len(heap) > 14000:
            heap = heapq.nsmallest(6500, heap)
            heapq.heapify(heap)
    return best_path, False


def act(obs, available, memory):
    start = time.perf_counter()
    a = np.asarray(obs, dtype=np.int16)
    available_set = set(available)
    can_undo = "ACTION7" in available_set
    can_reset = "RESET" in available_set

    if "targets" not in memory:
        targets, known_fruit = read_targets(a)
        req = read_requirements(a)
        memory["targets"] = targets
        memory["requirements"] = req
        memory["visits"] = {}
        memory["bad"] = {}
        memory["undos"] = 0
    else:
        known_fruit = ()

    targets = tuple(tuple(q) for q in memory["targets"])
    req = tuple(tuple(q) for q in memory["requirements"])

    if not targets or not req:
        targets, known_fruit = read_targets(a)
        req = read_requirements(a)
        if targets:
            memory["targets"] = targets
        if req:
            memory["requirements"] = req
        if not targets or not req:
            if can_undo and memory.get("undos", 0) < 2:
                memory["undos"] = memory.get("undos", 0) + 1
                return "ACTION7", None, None
            if can_reset:
                return "RESET", None, None
            return "ACTION6", 10, 0

    predicted = memory.get("predicted")
    if predicted is not None:
        predicted = tuple(tuple(q) for q in predicted)
    prediction_ok = predicted is not None and compatible(a, predicted, targets)
    if prediction_ok:
        state = predicted
    else:
        state = read_state(a, targets, known_fruit)
        memory.pop("plan", None)

    exact = state_key(state)
    visits = memory.setdefault("visits", {})
    visits[exact] = visits.get(exact, 0) + 1

    last_state = memory.get("last_state")
    last_action = memory.get("last_action")
    if (not prediction_ok and last_state == exact
            and last_action is not None):
        memory.setdefault("bad", {}).setdefault(exact, set()).add(
            tuple(last_action))

    h = heuristic(state, targets, req)
    if h >= 10000 or not any(q[0] < 9 for q in state):
        memory.pop("predicted", None)
        memory.pop("plan", None)
        memory["undos"] = memory.get("undos", 0) + 1
        if can_undo and memory["undos"] <= 3:
            return "ACTION7", None, None
        if can_reset:
            memory["visits"] = {}
            memory["bad"] = {}
            memory["undos"] = 0
            return "RESET", None, None

    plan = memory.get("plan", ())
    if prediction_ok and plan:
        action = tuple(plan[0])
        nxt, won = simulate(state, action, targets, req)
        if nxt is not None:
            memory["plan"] = tuple(plan[1:])
            memory["predicted"] = nxt
            memory["last_state"] = exact
            memory["last_action"] = action
            return "ACTION6", int(action[1]), int(action[0])
        memory.pop("plan", None)

    banned = memory.setdefault("bad", {}).get(exact, set())
    deadline = min(start + 8.5, time.perf_counter() + 8.0)
    path, solved = search(state, targets, req, visits, banned, deadline)

    if path:
        action = path[0]
        nxt, won = simulate(state, action, targets, req)
        memory["predicted"] = nxt
        memory["last_state"] = exact
        memory["last_action"] = action
        if solved:
            memory["plan"] = tuple(path[1:])
        else:
            memory.pop("plan", None)
        memory["undos"] = 0
        return "ACTION6", int(action[1]), int(action[0])

    best = None
    best_value = float("inf")
    for action in candidate_actions(state, targets):
        if action in banned:
            continue
        nxt, won = simulate(state, action, targets, req)
        if nxt is None:
            continue
        value = -10000 if won else heuristic(nxt, targets, req)
        value += visits.get(state_key(nxt), 0)
        if value < best_value:
            best_value = value
            best = action, nxt
        if time.perf_counter() > start + 9.2:
            break

    if best is not None:
        action, nxt = best
        memory["predicted"] = nxt
        memory["last_state"] = exact
        memory["last_action"] = action
        return "ACTION6", int(action[1]), int(action[0])

    memory.pop("predicted", None)
    if can_undo:
        return "ACTION7", None, None
    if can_reset:
        return "RESET", None, None
    return "ACTION6", 10, 0

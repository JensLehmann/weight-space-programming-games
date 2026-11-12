import math
import time
from collections import Counter

import numpy as np


_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
_MONSTER_COLORS = (7, 14, 13)
_PATTERNS = (
    ((0, 2), (1, 1), (1, 3), (2, 0), (2, 4),
     (3, 1), (3, 2), (3, 3)),
    ((0, 2), (1, 1), (1, 3), (2, 0), (2, 2), (2, 4),
     (3, 1), (3, 3)),
    ((0, 0), (0, 2), (0, 4), (1, 1), (1, 3),
     (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
     (3, 1), (3, 2), (3, 3)),
)


def _dims(t):
    return (_SIZES[t], _SIZES[t]) if t < 9 else (5, 4)


def _center(o):
    t, x, y = o
    w, h = _dims(t)
    return x + w // 2, y + h // 2


def _clamp(t, x, y):
    w, h = _dims(t)
    return max(0, min(64 - w, x)), max(10, min(64 - h, y))


def _overlap(a, b):
    wa, ha = _dims(a[0])
    wb, hb = _dims(b[0])
    return (a[1] < b[1] + wb and b[1] < a[1] + wa
            and a[2] < b[2] + hb and b[2] < a[2] + ha)


def _selected(o, x, y):
    w, h = _dims(o[0])
    dx = max(o[1] - x, 0, x - (o[1] + w - 1))
    dy = max(o[2] - y, 0, y - (o[2] + h - 1))
    return dx * dx + dy * dy <= 64


def _box_sums(a, h, w):
    z = np.pad(a.astype(np.float32), ((1, 0), (1, 0)))
    z = z.cumsum(0).cumsum(1)
    return z[h:, w:] - z[:-h, w:] - z[h:, :-w] + z[:-h, :-w]


def _read_goals(obs):
    seen = np.zeros((10, 64), dtype=bool)
    counts = Counter()
    for y in range(10):
        for x in range(24, 64):
            c = int(obs[y, x])
            if seen[y, x] or c in (4, 5):
                continue
            stack = [(x, y)]
            seen[y, x] = True
            pts = []
            while stack:
                px, py = stack.pop()
                pts.append((px, py))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx, ny = px + dx, py + dy
                        if (24 <= nx < 64 and 0 <= ny < 10
                                and not seen[ny, nx]
                                and int(obs[ny, nx]) == c):
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            if w == 5 and h == 4 and len(pts) < 20:
                if c in _MONSTER_COLORS:
                    t = 9 + _MONSTER_COLORS.index(c)
                else:
                    # Recoloured warning markers retain their geometry.
                    rel = {(py - min(ys), px - min(xs)) for px, py in pts}
                    t = 9 + min(
                        range(3),
                        key=lambda k: len(rel.symmetric_difference(_PATTERNS[k]))
                    )
                counts[t] += 1
            elif w == h and w in _SIZES:
                counts[_SIZES.index(w)] += 1
            elif c in _COLORS and len(pts) >= 1:
                # A clipped marker is uncommon, but its colour is sufficient.
                counts[_COLORS.index(c)] += 1
    return tuple(sorted(counts.items()))


def _read_bowls(obs):
    mask = ((obs == 9) | (obs == 2)).astype(np.float32)
    inside = np.zeros((56, 56), dtype=np.float32)
    corners = np.zeros_like(inside)
    for dy in range(9):
        for dx in range(9):
            allowed = not (
                (dy in (0, 8) and dx in (0, 1, 7, 8))
                or (dy in (1, 7) and dx in (0, 8))
            )
            target = inside if allowed else corners
            target += mask[dy:dy + 56, dx:dx + 56]
    score = inside - 2.0 * corners
    score[corners > 1.0] = -1000
    score[:10] = -1000
    bowls = []
    for _ in range(12):
        flat = int(np.argmax(score))
        y, x = divmod(flat, 56)
        if score[y, x] < 27:
            break
        bowls.append((x, y))
        score[max(0, y - 8):min(56, y + 9),
              max(0, x - 8):min(56, x + 9)] = -1000
    return tuple(bowls)


def _read_pieces(obs):
    fruits = []
    fruit_mask = np.zeros((64, 64), dtype=bool)

    ys, xs = np.where(obs[10:63] == 10)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        fruits.append((0, xx, yy + 10))
        fruit_mask[yy + 10, xx] = True

    for t in range(1, 9):
        s, c = _SIZES[t], _COLORS[t]
        mask = (obs == c).astype(np.float32)
        mask[:10] = 0
        mask[63] = 0

        obscured = ((obs == 10) | (obs == 6)).astype(np.float32)
        if t == 1:
            obscured = (obs == 10).astype(np.float32)
        obscured[:10] = 0
        obscured[63] = 1

        if t == 6:
            threshold = 62
        elif t >= 7:
            threshold = int(math.ceil(s * s * 0.72))
        else:
            threshold = int(math.ceil(s * s * 0.72))

        for _ in range(24):
            counts = _box_sums(mask, s, s)
            score = counts + 0.55 * _box_sums(obscured, s, s)
            score[counts < threshold] = -1000
            score[:10] = -1000
            flat = int(np.argmax(score))
            y, x = divmod(flat, score.shape[1])
            if score[y, x] < 0:
                break
            fruits.append((t, x, y))
            fruit_mask[y:y + s, x:x + s] = True
            mask[y:y + s, x:x + s] = 0

    monsters = []
    for rank in range(3):
        c = _MONSTER_COLORS[rank]
        own = (obs == c).astype(np.float32)
        own[:10] = 0
        own[fruit_mask] = 0
        pattern = set(_PATTERNS[rank])
        n = len(pattern)
        for _ in range(16):
            hits = np.zeros((61, 60), dtype=np.float32)
            bad = np.zeros_like(hits)
            extra = np.zeros_like(hits)
            for dy in range(4):
                for dx in range(5):
                    sl = own[dy:dy + 61, dx:dx + 60]
                    if (dy, dx) in pattern:
                        hits += sl
                        covered = fruit_mask[dy:dy + 61, dx:dx + 60]
                        original = obs[dy:dy + 61, dx:dx + 60]
                        bad += ((sl == 0) & ~covered & (original != 0))
                    else:
                        extra += sl
            score = hits - 1.8 * bad - 0.65 * extra
            score[(hits < max(3, n // 2))] = -1000
            score[:10] = -1000
            flat = int(np.argmax(score))
            y, x = divmod(flat, 60)
            if score[y, x] < max(2, n * 0.35):
                break
            monsters.append((9 + rank, x, y))
            for dy, dx in pattern:
                own[y + dy, x + dx] = 0

    fruits.sort(key=lambda o: (o[2], o[1], o[0]))
    monsters.sort(key=lambda o: (o[0], o[2], o[1]))
    return tuple(fruits), tuple(monsters)


def _prediction_matches(obs, state, bowls):
    if state is None:
        return False
    image = np.full((64, 64), 5, dtype=np.int16)
    for x, y in bowls:
        for dy in range(9):
            for dx in range(9):
                if not (
                    (dy in (0, 8) and dx in (0, 1, 7, 8))
                    or (dy in (1, 7) and dx in (0, 8))
                ):
                    image[y + dy, x + dx] = 9
    for t, x, y in state[1]:
        for dy, dx in _PATTERNS[t - 9]:
            image[y + dy, x + dx] = _MONSTER_COLORS[t - 9]
    # The small pieces and the largest square have elevated layers.
    order = sorted(
        state[0],
        key=lambda o: 10 if o[0] == 0 else
                      9 if o[0] == 1 else
                      2 if o[0] == 8 else 0
    )
    for t, x, y in order:
        s = _SIZES[t]
        image[y:y + s, x:x + s] = _COLORS[t]
    actual = obs.astype(np.int16).copy()
    actual[actual == 2] = 9
    valid = np.ones((64, 64), dtype=bool)
    valid[:10] = False
    valid[63] = False
    valid[(obs == 0) | (obs == 3)] = False
    return int(np.count_nonzero((image != actual) & valid)) <= 3


def _goal(state, goals, bowls):
    counts = Counter()
    for o in state[0] + state[1]:
        cx, cy = _center(o)
        if any(x <= cx < x + 9 and y <= cy < y + 9 for x, y in bowls):
            counts[o[0]] += 1
    return all(counts[t] == n for t, n in goals)


def _simulate(state, click, goals, bowls):
    """One complete click, including all animation-frame transitions."""
    px, py = click
    objects = {}
    fl, ml = [], []
    next_id = 0
    for group, ids in ((state[0], fl), (state[1], ml)):
        for o in group:
            objects[next_id] = list(o)
            ids.append(next_id)
            next_id += 1

    sf = {i for i in fl if _selected(objects[i], px, py)}
    sm = {i for i in ml if _selected(objects[i], px, py)}
    directions = {}
    for i in sm:
        if objects[i][0] == 9:
            cx, cy = _center(objects[i])
            dx, dy = px - cx, py - cy
            d = math.hypot(dx, dy)
            directions[i] = (dx / d, dy / d) if d else (0.0, 0.0)

    # age, original x/y, direction x/y, floating x/y, delete-after-shake
    knocked = {}
    cooldown = {}
    merged = False

    def snapshot():
        return (tuple(tuple(objects[i]) for i in fl),
                tuple(tuple(objects[i]) for i in ml))

    def remove(i, ids):
        if i in ids:
            ids.remove(i)
        objects.pop(i, None)
        sf.discard(i)
        sm.discard(i)
        knocked.pop(i, None)
        cooldown.pop(i, None)
        directions.pop(i, None)

    def merge_group(ids, monster=False):
        nonlocal next_id
        if len(ids) < 2:
            return True
        parent = list(range(len(ids)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                oa, ob = objects[ids[a]], objects[ids[b]]
                if _overlap(oa, ob):
                    if oa[0] != ob[0]:
                        return False
                    ra, rb = root(a), root(b)
                    parent[rb] = ra
        groups = {}
        for j, i in enumerate(ids):
            groups.setdefault(root(j), []).append(i)
        for group in groups.values():
            if len(group) < 2:
                continue
            old_t = objects[group[0]][0]
            cx = sum(_center(objects[i])[0] for i in group) // len(group)
            cy = sum(_center(objects[i])[1] for i in group) // len(group)
            limit = 11 if monster else 8
            destination = ml if monster else fl
            if old_t < limit:
                nt = old_t + 1
                w, h = _dims(nt)
                x, y = _clamp(nt, cx - w // 2, cy - h // 2)
                ni = next_id
                next_id += 1
                objects[ni] = [nt, x, y]
                destination.append(ni)
                if monster:
                    cooldown[ni] = 4
            for i in group:
                remove(i, destination)
        return True

    for frame in range(24):
        frozen = any(4 <= k[0] < 8 and not k[7]
                     for k in knocked.values())

        for i in list(sf):
            if i not in objects or i in knocked:
                continue
            o = objects[i]
            cx, cy = _center(o)
            dx = max(-4, min(4, px - cx))
            dy = max(-4, min(4, py - cy))
            o[1], o[2] = _clamp(o[0], o[1] + dx, o[2] + dy)

        for i in list(sm):
            if i not in objects or frozen:
                continue
            o = objects[i]
            if i in directions:
                dx, dy = directions[i]
                x = int(round(o[1] + 3.4 * dx))
                y = int(round(o[2] + 3.4 * dy))
            else:
                cx, cy = _center(o)
                x = o[1] + max(-4, min(4, px - cx))
                y = o[2] + max(-4, min(4, py - cy))
            o[1], o[2] = _clamp(o[0], x, y)

        for i in list(knocked):
            if i not in objects:
                knocked.pop(i, None)
                continue
            k = knocked[i]
            o = objects[i]
            age, ox, oy, dx, dy, fx, fy, delete = k
            if age < 4:
                wiggle = -1 if age % 4 == 1 else 1 if age % 4 == 3 else 0
                x, y = ox, oy + wiggle
            else:
                if age == 4:
                    o[0] -= 1
                fx += 2.5 * dx
                fy += 2.5 * dy
                k[5], k[6] = fx, fy
                x, y = int(round(fx)), int(round(fy))
            o[1], o[2] = _clamp(o[0], x, y)
            k[0] += 1
            if k[0] >= (4 if delete else 8):
                if delete:
                    remove(i, fl)
                else:
                    knocked.pop(i, None)

        for i in list(cooldown):
            cooldown[i] -= 1
            if cooldown[i] <= 0:
                cooldown.pop(i, None)
        cooling = set(cooldown)

        if frame < 4 and not frozen and fl and ml:
            for i in list(ml):
                if i in sm or i in cooling:
                    continue
                eligible = [j for j in fl if j not in knocked]
                if not eligible:
                    break
                o = objects[i]
                cx, cy = _center(o)
                j = min(
                    eligible,
                    key=lambda q: (
                        (_center(objects[q])[0] - cx) ** 2
                        + (_center(objects[q])[1] - cy) ** 2
                    )
                )
                tx, ty = _center(objects[j])
                step = 2 if o[0] == 11 else 1
                dx = step if tx > cx else -step if tx < cx else 0
                dy = step if ty > cy else -step if ty < cy else 0
                o[1], o[2] = _clamp(o[0], o[1] + dx, o[2] + dy)

            for i in list(ml):
                if i in sm or i in cooling:
                    continue
                for j in list(fl):
                    if j in knocked or not _overlap(objects[i], objects[j]):
                        continue
                    fcx, fcy = _center(objects[j])
                    mcx, mcy = _center(objects[i])
                    dx, dy = fcx - mcx, fcy - mcy
                    d = math.hypot(dx, dy)
                    if d:
                        dx, dy = dx / d, dy / d
                    else:
                        dx, dy = 0.0, -1.0
                    o = objects[j]
                    knocked[j] = [
                        0, o[1], o[2], dx, dy,
                        float(o[1]), float(o[2]), o[0] == 0
                    ]
                    cooldown[i] = 9

            if not merge_group(list(ml), True):
                return None, False

        if frame + 1 >= 4 and not merged:
            if not merge_group(list(ml), True):
                return None, False
            if not merge_group([i for i in fl if i in sf], False):
                return None, False
            merged = True

        result = snapshot()
        if _goal(result, goals, bowls):
            return result, True

        if frame + 1 >= 4 and not knocked:
            return (result, False) if fl else (None, False)

    return None, False


def _distance_to_bowl(cx, cy, bowls):
    if not bowls:
        return 50.0
    return min(
        max(max(x - cx, 0, cx - (x + 8)),
            max(y - cy, 0, cy - (y + 8)))
        for x, y in bowls
    )


def _heuristic(state, goals, bowls):
    fruits, monsters = state
    fruit_goals = [(t, n) for t, n in goals if t < 9]
    monster_goals = [(t, n) for t, n in goals if t >= 9]

    needed_mass = sum((1 << t) * n for t, n in fruit_goals)
    mass = sum(1 << o[0] for o in fruits)
    if mass < needed_mass:
        return 10000.0 + needed_mass - mass

    for t, n in fruit_goals:
        capacity = sum(min(1.0, 2.0 ** (o[0] - t)) for o in fruits)
        if capacity + 1e-6 < n:
            return 9000.0
    for t, n in monster_goals:
        capacity = sum(
            2.0 ** (o[0] - t) for o in monsters if o[0] <= t
        )
        if capacity + 1e-6 < n:
            return 9000.0

    total = 0.0
    used_f = 0
    used_m = 0

    def travel(node, speed):
        return _distance_to_bowl(node[0], node[1], bowls) / speed

    def construct(group, target, used, monster):
        # Nodes contain center x/y, production cost, and a source-piece bitmask.
        levels = {}
        direct = []
        for i, o in enumerate(group):
            bit = 1 << i
            if used & bit:
                continue
            r = o[0]
            cx, cy = _center(o)
            if r <= target:
                levels.setdefault(r, []).append((cx, cy, 0.0, bit))
            elif not monster and monsters:
                choices = []
                for m in monsters:
                    mx, my = _center(m)
                    d = math.hypot(cx - mx, cy - my)
                    ux, uy = ((cx - mx) / d, (cy - my) / d) if d else (0, -1)
                    ex, ey = mx + 13 * ux, my + 13 * uy
                    approach = max(0.0, max(abs(cx - mx), abs(cy - my)) - 6) / 10.0
                    cost = approach + 1.8 * (r - target)
                    choices.append((
                        cost + _distance_to_bowl(ex, ey, bowls) / 8,
                        ex, ey, cost
                    ))
                _, ex, ey, cost = min(choices)
                direct.append((ex, ey, cost, bit))

        first = 9 if monster else 0
        speed = 15.0 if monster else 8.0
        for r in range(first, target):
            nodes = levels.get(r, [])
            while len(nodes) >= 2:
                best = None
                for a in range(len(nodes)):
                    for b in range(a + 1, len(nodes)):
                        na, nb = nodes[a], nodes[b]
                        separation = max(abs(na[0] - nb[0]), abs(na[1] - nb[1]))
                        merge_cost = 0.65 + max(0.0, separation - 4) / speed
                        value = merge_cost + 0.2 * (na[2] + nb[2])
                        if best is None or value < best[0]:
                            best = (value, a, b, merge_cost)
                _, a, b, mc = best
                nb = nodes.pop(b)
                na = nodes.pop(a)
                node = (
                    (na[0] + nb[0]) // 2,
                    (na[1] + nb[1]) // 2,
                    na[2] + nb[2] + mc,
                    na[3] | nb[3]
                )
                levels.setdefault(r + 1, []).append(node)
        return levels.get(target, []) + direct

    for target, count in sorted(fruit_goals, reverse=True):
        nodes = construct(fruits, target, used_f, False)
        nodes.sort(key=lambda n: n[2] + travel(n, 8.0))
        chosen = 0
        for node in nodes:
            if node[3] & used_f:
                continue
            total += node[2] + travel(node, 8.0)
            used_f |= node[3]
            chosen += 1
            if chosen == count:
                break
        if chosen < count:
            total += 200.0 * (count - chosen)

    for target, count in sorted(monster_goals, reverse=True):
        nodes = construct(monsters, target, used_m, True)
        nodes.sort(key=lambda n: n[2] + travel(n, 10.0))
        chosen = 0
        for node in nodes:
            if node[3] & used_m:
                continue
            total += node[2] + travel(node, 10.0)
            used_m |= node[3]
            chosen += 1
            if chosen == count:
                break
        if chosen < count:
            total += 200.0 * (count - chosen)

    inside = Counter()
    for o in fruits + monsters:
        cx, cy = _center(o)
        if _distance_to_bowl(cx, cy, bowls) == 0:
            inside[o[0]] += 1
    for t, n in goals:
        total += 3.0 * max(0, inside[t] - n)
        total -= 0.15 * min(n, inside[t])
    return total


def _candidates(state, goals, bowls):
    fruits, monsters = state
    all_objects = fruits + monsters
    points = set()

    def add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < 64 and 10 <= y < 63:
            points.add((x, y))

    def toward(o, tx, ty):
        cx, cy = _center(o)
        dx, dy = tx - cx, ty - cy
        d = math.hypot(dx, dy)
        if not d:
            add(cx, cy)
            return
        ux, uy = dx / d, dy / d
        lo, hi = 0.0, 17.0
        for _ in range(7):
            mid = (lo + hi) / 2
            x, y = int(round(cx + ux * mid)), int(round(cy + uy * mid))
            if _selected(o, x, y):
                lo = mid
            else:
                hi = mid
        extent = min(d, max(0, lo - 0.15))
        add(cx + ux * extent, cy + uy * extent)
        add(cx + ux * min(extent, 4), cy + uy * min(extent, 4))
        if _selected(o, int(round(tx)), int(round(ty))):
            add(tx, ty)

    compass = ((1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1))

    for o in all_objects:
        cx, cy = _center(o)
        add(cx, cy)
        for dx, dy in compass:
            toward(o, cx + 30 * dx, cy + 30 * dy)

        bowl_order = sorted(
            bowls,
            key=lambda b: max(abs(b[0] + 4 - cx), abs(b[1] + 4 - cy))
        )
        for bx, by in bowl_order[:3]:
            toward(o, bx + 4, by + 4)
            ex = max(bx, min(bx + 8, cx))
            ey = max(by, min(by + 8, cy))
            toward(o, ex, ey)

        same = [q for q in all_objects if q != o and q[0] == o[0]]
        same.sort(key=lambda q: max(abs(_center(q)[0] - cx),
                                   abs(_center(q)[1] - cy)))
        for q in same[:3]:
            tx, ty = _center(q)
            toward(o, tx, ty)

        if o[0] < 9:
            for q in sorted(
                monsters,
                key=lambda q: max(abs(_center(q)[0] - cx),
                                  abs(_center(q)[1] - cy))
            )[:2]:
                toward(o, *_center(q))

    for group in (fruits, monsters):
        for i, a in enumerate(group):
            ax, ay = _center(a)
            for b in group[i + 1:]:
                if a[0] != b[0]:
                    continue
                bx, by = _center(b)
                if max(abs(ax - bx), abs(ay - by)) <= 27:
                    mx, my = (ax + bx) / 2, (ay + by) / 2
                    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                        add(mx + dx, my + dy)

    for x, y in bowls:
        for dx, dy in ((4, 4), (1, 1), (7, 1), (1, 7), (7, 7)):
            add(x + dx, y + dy)

    # All empty clicks have the same effect: four frames of chasing.
    wait = None
    for y in (10, 62, 34, 18, 50):
        for x in (0, 63, 32, 16, 48):
            if not any(_selected(o, x, y) for o in all_objects):
                wait = (x, y)
                break
        if wait is not None:
            break

    filtered = [
        p for p in points
        if any(_selected(o, p[0], p[1]) for o in all_objects)
    ]
    if wait is not None:
        filtered.append(wait)
    elif not filtered:
        filtered.append((0, 10))

    # Stable order makes repeated planning reproducible.
    return sorted(set(filtered), key=lambda p: (p[1], p[0]))


def _key(state):
    return tuple(sorted(state[0])), tuple(sorted(state[1]))


def _coarse_key(state):
    return (
        tuple(sorted((t, x // 5, y // 5) for t, x, y in state[0])),
        tuple(sorted((t, x // 5, y // 5) for t, x, y in state[1]))
    )


def _search(state, goals, bowls, visits, seconds=7.6):
    deadline = time.perf_counter() + seconds
    beam = [(state, (), _heuristic(state, goals, bowls))]
    seen = {_key(state): 0}
    hcache = {}
    best_path = ()
    best_value = float("inf")
    simulations = 0
    maximum = 6500
    width = 16

    for depth in range(1, 9):
        successors = []
        exhausted = False
        for current, path, _ in beam:
            for click in _candidates(current, goals, bowls):
                simulations += 1
                if simulations > maximum or (
                    simulations % 32 == 0 and time.perf_counter() >= deadline
                ):
                    exhausted = True
                    break
                nxt, won = _simulate(current, click, goals, bowls)
                if nxt is None:
                    continue
                npth = path + (click,)
                if won:
                    return npth, True
                k = _key(nxt)
                if k == _key(current):
                    continue
                old_depth = seen.get(k)
                if old_depth is not None and old_depth <= depth:
                    continue
                seen[k] = depth
                h = hcache.get(k)
                if h is None:
                    h = _heuristic(nxt, goals, bowls)
                    hcache[k] = h
                if h >= 1000:
                    continue
                repetition = visits.get(k, 0)
                value = h + 0.65 * depth + 1.4 * repetition
                if value < best_value:
                    best_value = value
                    best_path = npth
                successors.append((value, h, nxt, npth))
            if exhausted:
                break

        if not successors:
            break
        successors.sort(key=lambda z: (z[0], z[1]))
        beam = []
        geometry = set()
        roots = Counter()
        for value, h, nxt, path in successors:
            ck = _coarse_key(nxt)
            if ck in geometry or roots[path[0]] >= 4:
                continue
            geometry.add(ck)
            roots[path[0]] += 1
            beam.append((nxt, path, h))
            if len(beam) >= width:
                break
        if exhausted or not beam:
            break
    return best_path, False


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=np.int16)

    if "goals" not in memory:
        memory["goals"] = _read_goals(obs)
        memory["bowls"] = _read_bowls(obs)
        memory["visits"] = {}
        memory["plan"] = []
        memory["pred"] = None
        memory["stalls"] = 0

    goals = tuple(memory["goals"])
    bowls = tuple(memory["bowls"])

    # If recognition was temporarily obstructed, retry on the next frame.
    if not goals:
        goals = _read_goals(obs)
        memory["goals"] = goals
    if not bowls:
        bowls = _read_bowls(obs)
        memory["bowls"] = bowls

    predicted = memory.get("pred")
    matches = _prediction_matches(obs, predicted, bowls)
    if matches:
        state = predicted
    else:
        state = _read_pieces(obs)
        memory["plan"] = []

    k = _key(state)
    visits = memory.setdefault("visits", {})
    visits[k] = visits.get(k, 0) + 1

    if not goals or not bowls:
        # A safe exploratory click also lets any obscured pieces separate.
        memory["pred"] = None
        return "ACTION6", 10, 0

    h = _heuristic(state, goals, bowls)
    if h >= 1000:
        # Undo does not spend the in-game move budget.
        undo_count = memory.get("emergency_undo", 0)
        memory["plan"] = []
        memory["pred"] = None
        if undo_count < 2 and "ACTION7" in available:
            memory["emergency_undo"] = undo_count + 1
            return "ACTION7", None, None
        memory.clear()
        return "RESET", None, None
    memory["emergency_undo"] = 0

    plan = list(memory.get("plan", []))
    if plan and matches:
        click = tuple(plan.pop(0))
        nxt, won = _simulate(state, click, goals, bowls)
        if nxt is not None:
            memory["plan"] = plan
            memory["pred"] = nxt
            return "ACTION6", int(click[1]), int(click[0])
        plan = []

    path, solved = _search(state, goals, bowls, visits)
    if path:
        click = path[0]
        nxt, won = _simulate(state, click, goals, bowls)
        memory["pred"] = nxt
        # Complete solutions are replayed. Partial plans are kept short
        # so that fresh visual information can correct the model.
        memory["plan"] = list(path[1:] if solved else path[1:3])
        memory["stalls"] = 0
        return "ACTION6", int(click[1]), int(click[0])

    # A shorter one-step fallback is useful if the broad search was trapped.
    choices = []
    for click in _candidates(state, goals, bowls):
        nxt, won = _simulate(state, click, goals, bowls)
        if nxt is None:
            continue
        if won:
            memory["pred"] = nxt
            memory["plan"] = []
            return "ACTION6", int(click[1]), int(click[0])
        nk = _key(nxt)
        if nk != k:
            value = _heuristic(nxt, goals, bowls) + 2 * visits.get(nk, 0)
            choices.append((value, click, nxt))
    if choices:
        _, click, nxt = min(choices, key=lambda z: z[0])
        memory["pred"] = nxt
        memory["plan"] = []
        return "ACTION6", int(click[1]), int(click[0])

    memory["stalls"] = memory.get("stalls", 0) + 1
    memory["pred"] = None
    if memory["stalls"] <= 2 and "ACTION7" in available:
        return "ACTION7", None, None
    memory.clear()
    return "RESET", None, None

import math
import time
import heapq
import numpy as np

_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
_ENEMY_COLORS = (7, 14, 13)
_PATTERNS = (
    ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
    ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2), (4, 2),
     (1, 3), (3, 3)),
    ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
     (1, 3), (2, 3), (3, 3)),
)
_CIRCLE = tuple(
    (x, y) for y in range(9) for x in range(9)
    if (2 <= x <= 6 if y in (0, 8)
        else 1 <= x <= 7 if y in (1, 7)
        else True)
)


def _rectsum(a, h, w):
    z = np.pad(a.astype(np.int16), ((1, 0), (1, 0)))
    z = z.cumsum(0).cumsum(1)
    return z[h:, w:] - z[:-h, w:] - z[h:, :-w] + z[:-h, :-w]


def _fruit_image(obs):
    fruits = []
    cover = np.zeros((64, 64), dtype=bool)
    cover[63, :] = True

    for r in (0, 1, 8, 2, 3, 4, 5, 6, 7):
        s = _SIZES[r]
        own = obs == _COLORS[r]
        good = own | cover
        total = _rectsum(good, s, s)
        visible = _rectsum(own & ~cover, s, s)
        ys, xs = np.where((total == s * s) & (visible > 0))
        candidates = [
            (-int(visible[y, x]), int(y), int(x))
            for y, x in zip(ys, xs)
            if y >= 10
        ]
        candidates.sort()
        used = np.zeros((64, 64), dtype=bool)
        for _, y, x in candidates:
            if used[y:y+s, x:x+s].any():
                continue
            fruits.append((r, x, y))
            used[y:y+s, x:x+s] = True
            cover[y:y+s, x:x+s] = True

    return tuple(sorted(fruits)), cover


def _enemy_image(obs, cover, lo, hi, header=False):
    found = []
    for k, positions in enumerate(_PATTERNS, 1):
        color = _ENEMY_COLORS[k - 1]
        matches = np.zeros((61, 60), dtype=np.int16)
        misses = np.zeros_like(matches)
        conflicts = np.zeros_like(matches)
        positive = set(positions)

        for dx, dy in positions:
            a = obs[dy:dy+61, dx:dx+60]
            c = cover[dy:dy+61, dx:dx+60]
            matches += (a == color) & ~c
            misses += (a != color) & ~c

        for dy in range(4):
            for dx in range(5):
                if (dx, dy) in positive:
                    continue
                a = obs[dy:dy+61, dx:dx+60]
                c = cover[dy:dy+61, dx:dx+60]
                conflicts += (a == color) & ~c

        threshold = len(positions) - 1 if header else 3
        ys, xs = np.where(
            (matches >= threshold) & (misses <= 1) & (conflicts == 0)
        )
        choices = [
            (-int(matches[y, x]), int(y), int(x))
            for y, x in zip(ys, xs)
            if lo <= y and y + 4 <= hi and (not header or x >= 24)
        ]
        choices.sort()
        occupied = []
        for _, y, x in choices:
            if any(
                x < xx + 5 and xx < x + 5 and
                y < yy + 4 and yy < y + 4
                for xx, yy in occupied
            ):
                continue
            occupied.append((x, y))
            found.append((k, x, y))
    return tuple(sorted(found))


def _read_goals(obs):
    mask = np.zeros((64, 64), dtype=bool)
    enemies = _enemy_image(obs, mask, 0, 10, True)
    goals = {}
    for k, x, y in enemies:
        goals[(1, k)] = goals.get((1, k), 0) + 1
        mask[y:y+4, x:x+5] = True

    for r, color in enumerate(_COLORS):
        a = (obs[:10, 24:] == color) & ~mask[:10, 24:]
        seen = np.zeros_like(a)
        areas = []
        for y, x in zip(*np.where(a)):
            y, x = int(y), int(x)
            if seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            area = 0
            while stack:
                xx, yy = stack.pop()
                area += 1
                for nx, ny in (
                    (xx-1, yy), (xx+1, yy),
                    (xx, yy-1), (xx, yy+1)
                ):
                    if (0 <= nx < 40 and 0 <= ny < 10 and
                            a[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            areas.append(area)
        if areas:
            s = _SIZES[r]
            count = sum(max(1, int(round(v / float(s * s))))
                        for v in areas)
            goals[(0, r)] = count
    return goals


def _read_circles(obs, fruits, enemies):
    cover = np.zeros((64, 64), dtype=bool)
    cover[63, :] = True

    # Static guide dots and the instructional pointer can obscure goals.
    # Neither color belongs to a movable fruit or enemy.
    cover |= (obs == 3) | (obs == 0)

    for r, x, y in fruits:
        s = _SIZES[r]
        cover[y:y+s, x:x+s] = True
    for k, x, y in enemies:
        for dx, dy in _PATTERNS[k-1]:
            cover[y+dy, x+dx] = True

    candidates = []
    positive = set(_CIRCLE)
    for color in (9, 2):
        matches = np.zeros((56, 56), dtype=np.int16)
        misses = np.zeros_like(matches)
        conflicts = np.zeros_like(matches)
        for dx, dy in _CIRCLE:
            a = obs[dy:dy+56, dx:dx+56]
            c = cover[dy:dy+56, dx:dx+56]
            matches += (a == color) & ~c
            misses += (a != color) & ~c
        for dy in range(9):
            for dx in range(9):
                if (dx, dy) in positive:
                    continue
                a = obs[dy:dy+56, dx:dx+56]
                c = cover[dy:dy+56, dx:dx+56]
                conflicts += (a == color) & ~c
        ys, xs = np.where(
            (matches >= 8) & (misses <= 1) & (conflicts <= 1)
        )
        for y, x in zip(ys, xs):
            if y >= 10:
                score = int(matches[y, x] - 3 * misses[y, x]
                            - 3 * conflicts[y, x])
                candidates.append((-score, int(x), int(y)))

    candidates.sort()
    result = []
    for _, x, y in candidates:
        if any(max(abs(x-xx), abs(y-yy)) < 9 for xx, yy in result):
            continue
        result.append((x, y))
    return tuple(sorted(result))


def _pack(fruits, enemies):
    return (
        tuple(sorted(tuple(v) for v in fruits.values())),
        tuple(sorted(tuple(v) for v in enemies.values())),
    )


def _center(v, enemy=False):
    r, x, y = v
    if enemy:
        return x + 2, y + 2
    s = _SIZES[r]
    return x + s // 2, y + s // 2


def _touch(a, b, enemy_a=False, enemy_b=False):
    wa, ha = (5, 4) if enemy_a else (_SIZES[a[0]],) * 2
    wb, hb = (5, 4) if enemy_b else (_SIZES[b[0]],) * 2
    return (a[1] < b[1] + wb and b[1] < a[1] + wa and
            a[2] < b[2] + hb and b[2] < a[2] + ha)


def _selected(v, x, y, enemy=False):
    w, h = (5, 4) if enemy else (_SIZES[v[0]],) * 2
    dx = max(v[1] - x, 0, x - (v[1] + w - 1))
    dy = max(v[2] - y, 0, y - (v[2] + h - 1))
    return dx * dx + dy * dy <= 64


def _clamp_position(v, x, y, enemy=False):
    w, h = (5, 4) if enemy else (_SIZES[v[0]],) * 2
    return max(0, min(64-w, x)), max(10, min(64-h, y))


def _inside(cx, cy, circles):
    return any(x <= cx < x+9 and y <= cy < y+9 for x, y in circles)


def _won(fruits, enemies, goals, circles):
    counts = {}
    for enemy, objects in ((0, fruits), (1, enemies)):
        for v in objects.values():
            cx, cy = _center(v, bool(enemy))
            if _inside(cx, cy, circles):
                key = (enemy, v[0])
                counts[key] = counts.get(key, 0) + 1
    return bool(goals) and all(counts.get(k, 0) == n
                               for k, n in goals.items())


def _merge(objects, selected, enemy, next_id, immune):
    ids = [i for i in objects if enemy or i in selected]
    if len(ids) < 2:
        return next_id
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for n, i in enumerate(ids):
        for j in ids[n+1:]:
            if _touch(objects[i], objects[j], enemy, enemy):
                if objects[i][0] != objects[j][0]:
                    return -1
                a, b = root(i), root(j)
                parent[b] = a

    groups = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    for group in groups.values():
        if len(group) < 2:
            continue
        rank = objects[group[0]][0]
        centers = [_center(objects[i], enemy) for i in group]
        cx = sum(p[0] for p in centers) // len(group)
        cy = sum(p[1] for p in centers) // len(group)
        maximum = 3 if enemy else 8
        if rank < maximum:
            nr = rank + 1
            w, h = (5, 4) if enemy else (_SIZES[nr],) * 2
            v = [nr, 0, 10]
            x, y = _clamp_position(v, cx-w//2, cy-h//2, enemy)
            objects[next_id] = [nr, x, y]
            if enemy:
                immune[next_id] = 4
            next_id += 1
        for i in group:
            del objects[i]
            selected.discard(i)
            immune.pop(i, None)
    return next_id


def _simulate(state, x, y, goals, circles):
    f = {i: list(v) for i, v in enumerate(state[0])}
    start = len(f)
    e = {start+i: list(v) for i, v in enumerate(state[1])}
    next_id = len(f) + len(e)
    sf = {i for i, v in f.items() if _selected(v, x, y)}
    se = {i for i, v in e.items() if _selected(v, x, y, True)}
    direction = {}
    for i in se:
        v = e[i]
        if v[0] == 1:
            cx, cy = _center(v, True)
            dx, dy = x-cx, y-cy
            length = math.hypot(dx, dy)
            direction[i] = (dx/length, dy/length) if length else (0., 0.)

    # time, original x/y, direction x/y, floating x/y, resulting rank
    knock = {}
    immune = {}
    fruit_merged = False
    frames = 0

    def frozen():
        return any(v[7] >= 0 and 4 <= v[0] < 8
                   for i, v in knock.items() if i in f)

    while frames < 24:
        freeze_before_move = frozen()

        for i in tuple(sf):
            if i not in f or i in knock:
                continue
            v = f[i]
            cx, cy = _center(v)
            nx = v[1] + max(-4, min(4, x-cx))
            ny = v[2] + max(-4, min(4, y-cy))
            v[1], v[2] = _clamp_position(v, nx, ny)

        if not freeze_before_move:
            for i in tuple(se):
                if i not in e:
                    continue
                v = e[i]
                if i in direction:
                    ux, uy = direction[i]
                    nx = int(round(v[1] + ux * 3.4))
                    ny = int(round(v[2] + uy * 3.4))
                else:
                    cx, cy = _center(v, True)
                    nx = v[1] + max(-4, min(4, x-cx))
                    ny = v[2] + max(-4, min(4, y-cy))
                v[1], v[2] = _clamp_position(v, nx, ny, True)

        freeze = frozen()

        for i in tuple(knock):
            if i not in f:
                del knock[i]
                continue
            q = knock[i]
            v = f[i]
            t = q[0]
            if t < 4:
                jiggle = -1 if t % 4 == 1 else 1 if t % 4 == 3 else 0
                nx, ny = q[1], q[2] + jiggle
            else:
                if v[0] != q[7]:
                    v[0] = q[7]
                q[5] += q[3] * 2.5
                q[6] += q[4] * 2.5
                nx, ny = q[5], q[6]
            v[1], v[2] = _clamp_position(
                v, int(round(nx)), int(round(ny))
            )
            q[0] += 1
            lifetime = 4 if q[7] < 0 else 8
            if q[0] >= lifetime:
                if q[7] < 0:
                    del f[i]
                    sf.discard(i)
                del knock[i]

        for i in tuple(immune):
            immune[i] -= 1
            if immune[i] <= 0:
                del immune[i]

        if frames < 4 and f and e and not freeze:
            immune_now = set(immune)
            for i, v in list(e.items()):
                if i in se or i in immune_now:
                    continue
                targets = [(j, u) for j, u in f.items() if j not in knock]
                if not targets:
                    break
                cx, cy = _center(v, True)
                j, u = min(
                    targets,
                    key=lambda p: (
                        (_center(p[1])[0]-cx)**2 +
                        (_center(p[1])[1]-cy)**2
                    )
                )
                tx, ty = _center(u)
                speed = 2 if v[0] == 3 else 1
                nx = v[1] + (speed if tx > cx else -speed if tx < cx else 0)
                ny = v[2] + (speed if ty > cy else -speed if ty < cy else 0)
                v[1], v[2] = _clamp_position(v, nx, ny, True)

            for i, v in list(e.items()):
                if i in se or i in immune_now:
                    continue
                for j, u in list(f.items()):
                    if j in knock or not _touch(v, u, True, False):
                        continue
                    fx, fy = _center(u)
                    ex, ey = _center(v, True)
                    dx, dy = fx-ex, fy-ey
                    length = math.hypot(dx, dy)
                    ux, uy = (dx/length, dy/length) if length else (0., -1.)
                    knock[j] = [
                        0, float(u[1]), float(u[2]), ux, uy,
                        float(u[1]), float(u[2]), u[0]-1
                    ]
                    immune[i] = 9

            next_id = _merge(e, se, True, next_id, immune)
            if next_id < 0:
                return None, False

        frames += 1
        if frames >= 4 and not fruit_merged:
            next_id = _merge(e, se, True, next_id, immune)
            if next_id < 0:
                return None, False
            next_id = _merge(f, sf, False, next_id, {})
            if next_id < 0:
                return None, False
            knock = {i: q for i, q in knock.items() if i in f}
            fruit_merged = True

        if _won(f, e, goals, circles):
            return _pack(f, e), True
        if frames >= 4 and not knock:
            break

    if not f:
        return None, False
    return _pack(f, e), False


def _goal_distance(cx, cy, circles):
    if not circles:
        return 40.
    return float(min(
        max(max(x-cx, 0, cx-(x+8)),
            max(y-cy, 0, cy-(y+8)))
        for x, y in circles
    ))


def _heuristic(state, goals, circles):
    f, e = state
    if not f:
        return 500.
    need_mass = sum((1 << r) * n
                    for (kind, r), n in goals.items() if kind == 0)
    if sum(1 << v[0] for v in f) < need_mass:
        return 500.

    score = 0.
    for kind, originals in ((0, f), (1, e)):
        items = [
            [v[0], *_center(v, bool(kind))]
            for v in originals
        ]
        requests = sorted(
            (r for (k, r), n in goals.items() if k == kind
             for _ in range(n)), reverse=True
        )
        for target in requests:
            for _ in range(40):
                exact = [i for i, v in enumerate(items) if v[0] == target]
                if exact:
                    i = min(exact, key=lambda j:
                            _goal_distance(items[j][1], items[j][2], circles))
                    v = items.pop(i)
                    score += _goal_distance(v[1], v[2], circles) / (
                        11.5 if kind and target == 1 else 9.
                    )
                    break

                higher = [i for i, v in enumerate(items) if v[0] > target]
                lower_mass = sum(
                    1 << v[0] for v in items if v[0] < target
                )
                if kind:
                    higher = []

                if higher and lower_mass < (1 << target):
                    if not e:
                        return score + 200.

                    def conversion_cost(i):
                        r, cx, cy = items[i]
                        s = _SIZES[r]
                        contact = min(
                            max(0., max(abs(cx-_center(u, True)[0]),
                                        abs(cy-_center(u, True)[1]))
                                - s/2. - 2.)
                            for u in e
                        )
                        return ((r-target) * 1.6 + contact / 6.5 +
                                _goal_distance(cx, cy, circles) / 9.)

                    i = min(higher, key=conversion_cost)
                    score += conversion_cost(i)
                    items.pop(i)
                    break

                pairs = []
                for i, a in enumerate(items):
                    if a[0] >= target:
                        continue
                    for j in range(i+1, len(items)):
                        b = items[j]
                        if a[0] != b[0]:
                            continue
                        half = 4. if kind else float(_SIZES[a[0]])
                        gap = max(0., max(abs(a[1]-b[1]),
                                           abs(a[2]-b[2])) - half)
                        cost = 1. + gap / (17. if kind else 13.)
                        pairs.append((cost + .04*(target-a[0]), cost, i, j))
                if not pairs:
                    if higher and e:
                        i = min(higher, key=lambda j: items[j][0])
                        v = items.pop(i)
                        cx, cy = v[1:]
                        contact = min(
                            max(abs(cx-_center(u, True)[0]),
                                abs(cy-_center(u, True)[1]))
                            for u in e
                        )
                        score += ((v[0]-target)*1.6 +
                                  max(0., contact-5.)/6.5 +
                                  _goal_distance(cx, cy, circles)/9.)
                        break
                    return score + 200.

                _, cost, i, j = min(pairs)
                a, b = items[i], items[j]
                new = [a[0]+1, (a[1]+b[1])//2, (a[2]+b[2])//2]
                items.pop(j)
                items.pop(i)
                items.append(new)
                score += cost
            else:
                return score + 200.
    return score


def _actions(state, circles, goals):
    f, e = state
    result = []
    present = set()

    def add(x, y):
        p = (max(0, min(63, int(round(x)))),
             max(10, min(62, int(round(y)))))
        if p not in present:
            present.add(p)
            result.append(p)

    def toward(v, tx, ty, enemy=False):
        cx, cy = _center(v, enemy)
        dx, dy = tx-cx, ty-cy
        length = math.hypot(dx, dy)
        radius = 10. if enemy else 8. + _SIZES[v[0]]//2
        if length <= radius:
            add(tx, ty)
        elif length:
            add(cx + dx*radius/length, cy + dy*radius/length)

    for enemy, objects in ((False, f), (True, e)):
        for v in objects:
            cx, cy = _center(v, enemy)
            for gx, gy in circles:
                tx = max(gx, min(gx+8, cx))
                ty = max(gy, min(gy+8, cy))
                toward(v, tx, ty, enemy)
                toward(v, gx+4, gy+4, enemy)
                if _selected(v, gx+4, gy+4, enemy):
                    add(gx+4, gy+4)

    for enemy, objects in ((False, f), (True, e)):
        for i, a in enumerate(objects):
            ax, ay = _center(a, enemy)
            for b in objects[i+1:]:
                if a[0] != b[0]:
                    continue
                bx, by = _center(b, enemy)
                if max(abs(ax-bx), abs(ay-by)) <= 28:
                    mx, my = (ax+bx)/2., (ay+by)/2.
                    add(mx, my)
                    add(math.floor(mx), math.floor(my))
                    add(math.ceil(mx), math.ceil(my))
                else:
                    toward(a, bx, by, enemy)
                    toward(b, ax, ay, enemy)

    desired = [r for (k, r) in goals if k == 0]
    for v in f:
        if desired and v[0] > min(desired):
            fx, fy = _center(v)
            for u in e:
                ex, ey = _center(u, True)
                toward(v, ex, ey)
                toward(u, fx, fy, True)

    for enemy, objects in ((False, f), (True, e)):
        for v in objects:
            cx, cy = _center(v, enemy)
            add(cx, cy)
            radius = 10. if enemy else 8. + _SIZES[v[0]]//2
            for dx, dy in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ):
                scale = radius / math.hypot(dx, dy)
                add(cx+dx*scale, cy+dy*scale)

    for p in ((0, 10), (63, 10), (0, 62), (63, 62),
              (0, 36), (63, 36), (32, 10), (32, 62)):
        add(*p)
    return result


def _coarse(state):
    return tuple(
        tuple((r, x//4, y//4) for r, x, y in objects)
        for objects in state
    )


def _consistent(state, obs, circles):
    picture = np.full((64, 64), 5, dtype=np.int16)
    circle_color = 2 if np.count_nonzero(obs[10:63] == 2) > 12 else 9
    for x, y in circles:
        for dx, dy in _CIRCLE:
            if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                picture[y+dy, x+dx] = circle_color
    for k, x, y in state[1]:
        for dx, dy in _PATTERNS[k-1]:
            if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                picture[y+dy, x+dx] = _ENEMY_COLORS[k-1]

    def layer(v):
        return 10 if v[0] == 0 else 9 if v[0] == 1 else 2 if v[0] == 8 else 0

    for r, x, y in sorted(state[0], key=layer):
        s = _SIZES[r]
        picture[y:y+s, x:x+s] = _COLORS[r]
    a, b = picture[10:63], obs[10:63]
    relevant = ((a >= 6) | (b >= 6)) & (b != 3) & (b != 0)
    return bool(np.all(a[relevant] == b[relevant]))


def _search(state, goals, circles, past, deadline):
    hcache = {}

    def h(s):
        if s not in hcache:
            hcache[s] = _heuristic(s, goals, circles)
        return hcache[s]

    need_mass = sum((1 << r)*n
                    for (kind, r), n in goals.items() if kind == 0)
    width = 18 if not state[1] else 14
    if len(state[0]) + len(state[1]) <= 4:
        width = 26

    beam = [(h(state), state, ())]
    seen = {state: 0}
    best = None
    serial = 0
    checks = 0

    for depth in range(1, 25):
        generated = []
        expired = False
        for _, current, path in beam:
            for x, y in _actions(current, circles, goals):
                checks += 1
                if checks % 32 == 0 and time.monotonic() >= deadline:
                    expired = True
                    break
                new, win = _simulate(current, x, y, goals, circles)
                if new is None:
                    continue
                newpath = path + ((x, y),)
                if win:
                    return newpath, True
                if sum(1 << v[0] for v in new[0]) < need_mass:
                    continue
                if seen.get(new, 100000) <= depth:
                    continue
                seen[new] = depth
                value = h(new)
                if value >= 190:
                    continue
                penalty = .65 if new in past else 0.
                overall = value + .16*depth + penalty
                if best is None or overall < best[0]:
                    best = (overall, newpath)
                serial += 1
                generated.append((value+penalty, serial, new, newpath))
            if expired:
                break

        if expired or not generated:
            break
        choices = heapq.nsmallest(width * 6, generated)
        beam = []
        coarse_seen = set()
        for value, _, new, path in choices:
            signature = _coarse(new)
            if signature in coarse_seen:
                continue
            coarse_seen.add(signature)
            beam.append((value, new, path))
            if len(beam) >= width:
                break

    if best is not None:
        return best[1], False

    choices = []
    for x, y in _actions(state, circles, goals):
        if time.monotonic() > deadline + .3:
            break
        new, win = _simulate(state, x, y, goals, circles)
        if new is not None:
            value = -1000. if win else h(new) + (.7 if new in past else 0.)
            choices.append((value, x, y))
    if choices:
        _, x, y = min(choices)
        return ((x, y),), False
    return (), False


def act(obs, available, memory):
    obs = np.asarray(obs)
    names = {str(a).upper().split(".")[-1] for a in available}
    deadline = time.monotonic() + 8.2

    fruits, cover = _fruit_image(obs)
    enemies = _enemy_image(obs, cover, 10, 64)
    parsed = (fruits, enemies)

    if "goals" not in memory:
        memory["goals"] = _read_goals(obs)
    goals = memory["goals"]

    if "circles" not in memory:
        memory["circles"] = _read_circles(obs, fruits, enemies)
    circles = memory["circles"]

    expected = memory.get("expected")
    if expected is not None and _consistent(expected, obs, circles):
        state = expected
    else:
        state = parsed
        memory.pop("route", None)
        memory.pop("expected", None)

    need_mass = sum((1 << r)*n
                    for (kind, r), n in goals.items() if kind == 0)
    bad = (not goals or not circles or not state[0] or
           sum(1 << v[0] for v in state[0]) < need_mass)

    if bad:
        attempts = memory.get("resets", 0)
        if attempts < 2:
            memory.clear()
            memory["resets"] = attempts + 1
            if "RESET" in names or not names:
                return "RESET", None, None
            if "ACTION7" in names:
                return "ACTION7", None, None

    if "ACTION6" not in names and names:
        if "ACTION7" in names:
            return "ACTION7", None, None
        return sorted(names)[0], None, None

    past = set(memory.get("past", ()))
    past.add(state)
    memory["past"] = list(past)[-100:]

    route = memory.get("route", ())
    if not route:
        route, solved = _search(state, goals, circles, past, deadline)
        if not solved:
            route = route[:2]

    if route:
        x, y = route[0]
        predicted, win = _simulate(state, x, y, goals, circles)
        if predicted is not None:
            memory["expected"] = predicted
            memory["route"] = tuple(route[1:])
            return "ACTION6", int(y), int(x)

    memory.pop("route", None)
    memory.pop("expected", None)
    if "ACTION7" in names:
        return "ACTION7", None, None
    if "RESET" in names:
        memory.clear()
        return "RESET", None, None
    return "ACTION6", 10, 0

import math
import time
import heapq
import numpy as np
from functools import lru_cache

# Kinds 0..8 are fruits; 9..11 are the three enemy types.
_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
_ENEMY = (
    np.array([[-1, -1, 7, -1, -1],
              [-1, 7, -1, 7, -1],
              [7, -1, -1, -1, 7],
              [-1, 7, 7, 7, -1]], dtype=int),
    np.array([[-1, -1, 14, -1, -1],
              [-1, 14, -1, 14, -1],
              [14, -1, 14, -1, 14],
              [-1, 14, -1, 14, -1]], dtype=int),
    np.array([[13, -1, 13, -1, 13],
              [-1, 13, -1, 13, -1],
              [13, 13, 13, 13, 13],
              [-1, 13, 13, 13, -1]], dtype=int),
)
_PAD = np.ones((9, 9), dtype=bool)
_PAD[0, :2] = _PAD[0, 7:] = False
_PAD[8, :2] = _PAD[8, 7:] = False
_PAD[1, 0] = _PAD[1, 8] = False
_PAD[7, 0] = _PAD[7, 8] = False
_BIG = 100000.0


def _dims(k):
    return (_SIZES[k], _SIZES[k]) if k < 9 else (5, 4)


def _center(o):
    k, x, y = o[:3]
    w, h = _dims(k)
    return x + w // 2, y + h // 2


def _clamp(k, x, y):
    w, h = _dims(k)
    return max(0, min(64 - w, int(x))), max(10, min(64 - h, int(y)))


def _overlap(a, b):
    wa, ha = _dims(a[0])
    wb, hb = _dims(b[0])
    return (a[1] < b[1] + wb and b[1] < a[1] + wa
            and a[2] < b[2] + hb and b[2] < a[2] + ha)


def _hit(o, x, y):
    w, h = _dims(o[0])
    dx = max(o[1] - x, 0, x - (o[1] + w - 1))
    dy = max(o[2] - y, 0, y - (o[2] + h - 1))
    return dx * dx + dy * dy <= 64


def _state(objects):
    return tuple(sorted(tuple(o[:3]) for o in objects))


def _inside(cx, cy, pads):
    return any(x <= cx < x + 9 and y <= cy < y + 9 for x, y in pads)


def _won(objects, goals, pads):
    counts = {}
    for o in objects:
        cx, cy = _center(o)
        if _inside(cx, cy, pads):
            counts[o[0]] = counts.get(o[0], 0) + 1
    return bool(goals) and all(counts.get(k, 0) == n for k, n in goals.items())


def _paint_objects(objects):
    canvas = np.full((64, 64), 5, dtype=int)
    coverage = np.zeros((64, 64), dtype=bool)

    def layer(o):
        k = o[0]
        if k >= 9:
            return -1
        return 10 if k == 0 else 9 if k == 1 else 2 if k == 8 else 0

    for k, x, y in sorted(objects, key=layer):
        w, h = _dims(k)
        if x < 0 or y < 0 or x + w > 64 or y + h > 64:
            continue
        if k < 9:
            canvas[y:y+h, x:x+w] = _COLORS[k]
            coverage[y:y+h, x:x+w] = True
        else:
            template = _ENEMY[k - 9]
            mask = template >= 0
            view = canvas[y:y+h, x:x+w]
            view[mask] = template[mask]
            coverage[y:y+h, x:x+w] |= mask
    return canvas, coverage


def _extract_objects(obs, y0=10, y1=63, x0=0):
    fruits = []
    used = np.zeros((64, 64), dtype=bool)

    # Solid squares can be found without relying on connected-component separation.
    for k, (color, size) in enumerate(zip(_COLORS, _SIZES)):
        region = obs[y0:y1, x0:64] == color
        if region.shape[0] < size or region.shape[1] < size:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            region, (size, size)
        )
        candidates = np.argwhere(windows.sum(axis=(-1, -2)) == size * size)
        selected = []
        for yy, xx in candidates:
            x, y = int(xx) + x0, int(yy) + y0
            obj = (k, x, y)
            if any(_overlap(obj, old) for old in selected):
                continue
            selected.append(obj)
            fruits.append(obj)
            used[y:y+size, x:x+size] = True

    # Recover partly covered squares, where enough of their own colour remains.
    for k, (color, size) in enumerate(zip(_COLORS, _SIZES)):
        if size == 1:
            continue
        remaining = (obs == color) & ~used
        remaining[:y0] = False
        remaining[y1:] = False
        remaining[:, :x0] = False
        seen = np.zeros((64, 64), dtype=bool)
        for sy, sx in np.argwhere(remaining):
            sy, sx = int(sy), int(sx)
            if seen[sy, sx]:
                continue
            stack = [(sx, sy)]
            seen[sy, sx] = True
            component = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if (x0 <= nx < 64 and y0 <= ny < y1
                            and remaining[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) < max(2, int(size * size * .50)):
                continue
            xs, ys = zip(*component)
            lo_x, hi_x = min(xs), max(xs)
            lo_y, hi_y = min(ys), max(ys)
            if hi_x - lo_x >= size or hi_y - lo_y >= size:
                continue
            best = None
            for y in range(max(y0, hi_y-size+1), min(lo_y, y1-size)+1):
                for x in range(max(x0, hi_x-size+1), min(lo_x, 64-size)+1):
                    box = obs[y:y+size, x:x+size]
                    own = box == color
                    covered = used[y:y+size, x:x+size]
                    if np.all(own | covered) and own.sum() >= size * size * .50:
                        score = int(own.sum())
                        if best is None or score > best[0]:
                            best = (score, x, y)
            if best is not None:
                _, x, y = best
                obj = (k, x, y)
                if not any(o[0] == k and _overlap(o, obj) for o in fruits):
                    fruits.append(obj)
                    used[y:y+size, x:x+size] = True

    enemies = []
    for e, template in enumerate(_ENEMY):
        k = e + 9
        color = (7, 14, 13)[e]
        positive = template >= 0
        negative = ~positive
        for y in range(y0, y1 - 3):
            for x in range(x0, 60):
                box = obs[y:y+4, x:x+5]
                cover = used[y:y+4, x:x+5]
                visible = positive & ~cover
                if np.count_nonzero(visible & (box == color)) < 3:
                    continue
                if not np.all((box == color)[positive] | cover[positive]):
                    continue
                if np.any((box == color) & negative & ~cover):
                    continue
                obj = (k, x, y)
                if not any(_overlap(obj, old) for old in enemies):
                    enemies.append(obj)
    return _state(fruits + enemies)


def _find_pads(obs, objects):
    _, cover = _paint_objects(objects)
    region = obs[10:63]
    cov = cover[10:63]
    ow = np.lib.stride_tricks.sliding_window_view(region, (9, 9))
    cw = np.lib.stride_tricks.sliding_window_view(cov, (9, 9))
    candidates = []
    for color in (9, 2):
        decoration = (ow == 3) | (ow == 0)
        valid_pos = (ow == color) | cw | decoration
        valid_neg = (ow == 5) | cw | decoration
        errors = ((~valid_pos) & _PAD).sum(axis=(-1, -2))
        errors += ((~valid_neg) & ~_PAD).sum(axis=(-1, -2))
        evidence = ((ow == color) & _PAD & ~cw).sum(axis=(-1, -2))
        for yy, xx in np.argwhere((errors == 0) & (evidence >= 8)):
            yy, xx = int(yy), int(xx)
            score = int(evidence[yy, xx])
            candidates.append((score, xx, yy + 10))
    pads = []
    for score, x, y in sorted(candidates, reverse=True):
        if not any(abs(x-px) < 8 and abs(y-py) < 8 for px, py in pads):
            pads.append((x, y))
    return tuple(sorted(pads))


def _read_goals(obs):
    icons = _extract_objects(obs, 0, 10, 24)
    goals = {}
    for k, x, y in icons:
        goals[k] = goals.get(k, 0) + 1
    return goals


def _prediction_matches(obs, state, pads):
    drawing, cover = _paint_objects(state)
    base = np.full((64, 64), 5, dtype=int)
    for x, y in pads:
        v = base[y:y+9, x:x+9]
        if v.shape == (9, 9):
            v[_PAD] = 9
    base[cover] = drawing[cover]
    actual = obs[10:63]
    expected = base[10:63]
    relevant = (actual != 0) & (actual != 3)
    errors = np.count_nonzero((actual != expected) & relevant)
    return errors <= 2


def _simulate(state, click, goals, pads):
    """Return (new_state, won), or None for an automatically rolled-back move."""
    xclick, yclick = click
    objects = [[k, x, y, i] for i, (k, x, y) in enumerate(state)]
    next_id = len(objects)
    selected = {o[3] for o in objects if _hit(o, xclick, yclick)}
    selected_enemies = {o[3] for o in objects if o[0] >= 9 and o[3] in selected}
    vectors = {}
    for o in objects:
        if o[0] == 9 and o[3] in selected_enemies:
            cx, cy = _center(o)
            dx, dy = xclick-cx, yclick-cy
            d = math.hypot(dx, dy)
            vectors[o[3]] = (dx/d, dy/d) if d else (0.0, 0.0)

    # id -> [age, old_x, old_y, direction_x, direction_y,
    #        float_x, float_y, pending_kind, kill]
    bitten = {}
    immunity = {}

    def remove_ids(ids):
        nonlocal objects
        objects = [o for o in objects if o[3] not in ids]
        for oid in ids:
            selected.discard(oid)
            selected_enemies.discard(oid)
            vectors.pop(oid, None)
            bitten.pop(oid, None)
            immunity.pop(oid, None)

    def merge_group(enemy):
        nonlocal next_id, objects
        members = [o for o in objects
                   if (o[0] >= 9 if enemy else o[0] < 9 and o[3] in selected)]
        if len(members) < 2:
            return True
        parent = list(range(len(members)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if not _overlap(members[i], members[j]):
                    continue
                if members[i][0] != members[j][0]:
                    return False
                a, b = root(i), root(j)
                parent[b] = a
        groups = {}
        for i, o in enumerate(members):
            groups.setdefault(root(i), []).append(o)
        for group in groups.values():
            if len(group) < 2:
                continue
            k = group[0][0]
            ids = {o[3] for o in group}
            cx = sum(_center(o)[0] for o in group) // len(group)
            cy = sum(_center(o)[1] for o in group) // len(group)
            remove_ids(ids)
            if k == (11 if enemy else 8):
                continue
            nk = k + 1
            w, h = _dims(nk)
            nx, ny = _clamp(nk, cx-w//2, cy-h//2)
            new = [nk, nx, ny, next_id]
            next_id += 1
            objects.append(new)
            if enemy:
                immunity[new[3]] = 4
        return True

    merged_fruits = False
    for frame in range(24):
        freeze = any(4 <= b[0] < 8 and not b[8] for b in bitten.values())

        # The pull phase.
        for o in objects:
            oid = o[3]
            if oid not in selected or oid in bitten:
                continue
            if o[0] >= 9 and freeze:
                continue
            if oid in vectors:
                dx, dy = vectors[oid]
                nx = round(o[1] + 3.4 * dx)
                ny = round(o[2] + 3.4 * dy)
            else:
                cx, cy = _center(o)
                nx = o[1] + max(-4, min(4, xclick-cx))
                ny = o[2] + max(-4, min(4, yclick-cy))
            o[1], o[2] = _clamp(o[0], nx, ny)

        # Bite animation, downgrade, and knockback.
        finished = set()
        killed = set()
        lookup = {o[3]: o for o in objects}
        for oid, b in list(bitten.items()):
            o = lookup.get(oid)
            if o is None:
                finished.add(oid)
                continue
            age = b[0]
            if age < 4:
                wobble = -1 if age % 4 == 1 else 1 if age % 4 == 3 else 0
                nx, ny = b[1], b[2] + wobble
            else:
                if b[7] is not None:
                    o[0] = b[7]
                    b[7] = None
                b[5] += b[3] * 2.5
                b[6] += b[4] * 2.5
                nx, ny = round(b[5]), round(b[6])
            o[1], o[2] = _clamp(o[0], nx, ny)
            b[0] += 1
            if b[0] >= (4 if b[8] else 8):
                finished.add(oid)
                if b[8]:
                    killed.add(oid)
        for oid in finished:
            bitten.pop(oid, None)
        if killed:
            remove_ids(killed)

        for oid in list(immunity):
            immunity[oid] -= 1
            if immunity[oid] <= 0:
                del immunity[oid]

        fruits = [o for o in objects if o[0] < 9]
        enemies = [o for o in objects if o[0] >= 9]

        # Enemies chase only during the first four frames.
        if frame < 4 and fruits and enemies and not freeze:
            blocked = set(immunity)
            for e in enemies:
                if e[3] in selected_enemies or e[3] in blocked:
                    continue
                targets = [f for f in fruits if f[3] not in bitten]
                if not targets:
                    break
                ex, ey = _center(e)
                target = min(targets, key=lambda f:
                             (_center(f)[0]-ex)**2 + (_center(f)[1]-ey)**2)
                tx, ty = _center(target)
                speed = 2 if e[0] == 11 else 1
                nx = e[1] + (speed if tx > ex else -speed if tx < ex else 0)
                ny = e[2] + (speed if ty > ey else -speed if ty < ey else 0)
                e[1], e[2] = _clamp(e[0], nx, ny)

            for e in enemies:
                if e[3] in selected_enemies or e[3] in blocked:
                    continue
                for f in fruits:
                    if f[3] in bitten or not _overlap(e, f):
                        continue
                    fx, fy = _center(f)
                    ex, ey = _center(e)
                    dx, dy = fx-ex, fy-ey
                    d = math.hypot(dx, dy)
                    ux, uy = (dx/d, dy/d) if d else (0.0, -1.0)
                    kill = f[0] == 0
                    bitten[f[3]] = [
                        0, float(f[1]), float(f[2]), ux, uy,
                        float(f[1]), float(f[2]),
                        None if kill else f[0]-1, kill
                    ]
                    immunity[e[3]] = 9
            if not merge_group(True):
                return None

        if frame >= 3 and not merged_fruits:
            if not merge_group(True):
                return None
            if not merge_group(False):
                return None
            merged_fruits = True

        if _won(objects, goals, pads):
            return _state(objects), True
        if frame >= 3 and not bitten:
            break

    if not any(o[0] < 9 for o in objects):
        return None
    return _state(objects), False


def _pad_distance(cx, cy, pads):
    best = float("inf")
    for x, y in pads:
        dx = max(x-cx, 0, cx-(x+8))
        dy = max(y-cy, 0, cy-(y+8))
        best = min(best, math.hypot(dx, dy))
    return best


def _inventory_estimate(pieces, needs, pads, enemies, is_enemy=False):
    if not needs:
        return 0.0
    if not pieces:
        return _BIG

    low = 9 if is_enemy else 0
    high = max(needs)
    nodes = []
    pools = {}
    for i, o in enumerate(pieces):
        cx, cy = _center(o)
        node = (o[0], float(cx), float(cy), 0.0, 1 << i)
        nodes.append(node)
        pools.setdefault(o[0], []).append(node)

    # Build a relaxed spatial merge forest. Retain its constituent nodes too;
    # bit masks prevent allocating the same piece to two requested outputs.
    for rank in range(low, high):
        pool = list(pools.get(rank, ()))
        while len(pool) >= 2:
            best = None
            for i in range(len(pool)):
                for j in range(i+1, len(pool)):
                    a, b = pool[i], pool[j]
                    distance = math.hypot(a[1]-b[1], a[2]-b[2])
                    w, _ = _dims(rank)
                    merge = (1.0 + max(0.0, distance-(16+w))/11.0
                             + .01 * distance)
                    cost = a[3] + b[3] + merge
                    if best is None or cost < best[0]:
                        best = (cost, i, j)
            cost, i, j = best
            a, b = pool[i], pool[j]
            new = (rank+1, (a[1]+b[1])/2, (a[2]+b[2])/2,
                   cost, a[4] | b[4])
            pool.pop(j)
            pool.pop(i)
            pools.setdefault(rank+1, []).append(new)
            nodes.append(new)

    requested = []
    for k in sorted(needs, reverse=True):
        requested.extend([k] * needs[k])

    options = {}
    for target in needs:
        choices = []
        for rank, cx, cy, creation, mask in nodes:
            if rank != target:
                continue
            step = 12.0 if is_enemy else 8.0 + _SIZES[target] / 2.0
            route = _pad_distance(cx, cy, pads) / step
            choices.append((creation + route, mask))
        if not is_enemy and enemies:
            for i, o in enumerate(pieces):
                if o[0] <= target:
                    continue
                cx, cy = _center(o)
                proximity = min(math.hypot(cx-_center(e)[0], cy-_center(e)[1])
                                for e in enemies)
                bite = 1.45 * (o[0]-target) + max(0, proximity-6) / 12.0
                route = _pad_distance(cx, cy, pads) / (8 + _SIZES[target]/2)
                choices.append((bite + route, 1 << i))
        # Same resource mask may appear through more than one relaxed route.
        by_mask = {}
        for cost, mask in choices:
            by_mask[mask] = min(cost, by_mask.get(mask, _BIG))
        options[target] = sorted((cost, mask) for mask, cost in by_mask.items())
        if not options[target]:
            return _BIG

    @lru_cache(None)
    def allocate(index, used):
        if index == len(requested):
            return 0.0
        best = _BIG
        for cost, mask in options[requested[index]]:
            if not mask & used:
                rest = allocate(index+1, used | mask)
                best = min(best, cost + rest)
        return best

    return allocate(0, 0)


def _heuristic(state, goals, pads):
    if not pads:
        return _BIG
    fruits = [o for o in state if o[0] < 9]
    enemies = [o for o in state if o[0] >= 9]
    fruit_goals = {k: n for k, n in goals.items() if k < 9}
    enemy_goals = {k: n for k, n in goals.items() if k >= 9}
    mass = sum(1 << o[0] for o in fruits)
    required = sum((1 << k) * n for k, n in fruit_goals.items())
    if mass < required:
        return _BIG
    fh = _inventory_estimate(fruits, fruit_goals, pads, enemies)
    eh = _inventory_estimate(enemies, enemy_goals, pads, (), True)
    if fh >= _BIG or eh >= _BIG:
        return _BIG

    # Before downgrades are finished, avoid insisting that the biting enemy
    # immediately leave the fruit and travel to its own landing pad.
    need_bites = (fruit_goals and enemies
                  and any(o[0] > max(fruit_goals) for o in fruits))
    enemy_weight = .65 if need_bites else 1.0
    h = fh + enemy_weight * eh

    counts = {}
    for o in state:
        if _inside(*_center(o), pads):
            counts[o[0]] = counts.get(o[0], 0) + 1
    for k, n in goals.items():
        h += 1.5 * max(0, counts.get(k, 0)-n)
    return h


def _candidate_clicks(state, pads):
    points = set()

    def add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < 64 and 10 <= y < 63:
            points.add((x, y))

    centers = [_center(o) for o in state]
    destinations = centers + [(x+4, y+4) for x, y in pads]

    for o in state:
        cx, cy = _center(o)
        add(cx, cy)
        # Nearby lattice points give alternative selection sets.
        w, h = _dims(o[0])
        xmin, xmax = max(0, o[1]-8), min(63, o[1]+w+7)
        ymin, ymax = max(10, o[2]-8), min(62, o[2]+h+7)
        for x in range((xmin+3)//4*4, xmax+1, 4):
            for y in range(10 + max(0, (ymin-10+3)//4)*4, ymax+1, 4):
                if _hit(o, x, y):
                    add(x, y)

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dirs += [(tx-cx, ty-cy) for tx, ty in destinations
                 if tx != cx or ty != cy]
        for dx, dy in dirs:
            length = math.hypot(dx, dy)
            if not length:
                continue
            dx, dy = dx/length, dy/length
            # Find the farthest integer pull in this useful direction.
            last = None
            for distance in (4, 6, 8, 10, 12, 14, 16):
                x, y = round(cx+dx*distance), round(cy+dy*distance)
                if 0 <= x < 64 and 10 <= y < 63 and _hit(o, x, y):
                    last = (x, y)
                else:
                    break
            if last:
                add(*last)
            add(cx+dx*6, cy+dy*6)

    for i, a in enumerate(state):
        for b in state[i+1:]:
            if a[0] != b[0]:
                continue
            ax, ay = _center(a)
            bx, by = _center(b)
            mx, my = (ax+bx)/2, (ay+by)/2
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                add(mx+dx, my+dy)

    for x, y in pads:
        for dx in (0, 4, 8):
            for dy in (0, 4, 8):
                add(x+dx, y+dy)
        for cx, cy in centers:
            add(max(x, min(x+8, cx)), max(y, min(y+8, cy)))

    # All clicks selecting nothing produce the same chase-only transition.
    active = []
    empty = None
    for p in sorted(points):
        if any(_hit(o, *p) for o in state):
            active.append(p)
        elif empty is None:
            empty = p
    if empty is None:
        for y in (10, 30, 62):
            for x in (0, 31, 63):
                if not any(_hit(o, x, y) for o in state):
                    empty = (x, y)
                    break
            if empty is not None:
                break
    if empty is not None:
        active.append(empty)
    return active


def _search(root, goals, pads, visits):
    deadline = time.perf_counter() + 7.6
    hcache = {}

    def h(state):
        if state not in hcache:
            hcache[state] = _heuristic(state, goals, pads)
        return hcache[state]

    initial_h = h(root)
    if initial_h >= _BIG:
        return None, False

    counter = 0
    queue = [(2.2 * initial_h, counter, 0, root, ())]
    best_g = {root: 0}
    best_path = None
    best_score = float("inf")
    expansions = 0

    while queue and expansions < 220 and time.perf_counter() < deadline:
        _, _, depth, state, path = heapq.heappop(queue)
        if best_g.get(state) != depth:
            continue
        if depth >= 18:
            continue
        expansions += 1
        local_seen = set()

        for index, click in enumerate(_candidate_clicks(state, pads)):
            if index % 16 == 0 and time.perf_counter() >= deadline:
                break
            result = _simulate(state, click, goals, pads)
            if result is None:
                continue
            new, won = result
            if new in local_seen:
                continue
            local_seen.add(new)
            if new == state:
                continue
            new_path = path + ((click, new),)
            if won:
                return list(new_path), True
            nd = depth + 1
            if best_g.get(new, 999) <= nd:
                continue
            nh = h(new)
            if nh >= _BIG:
                continue
            repetition = min(4, visits.get(new, 0))
            score = nh + .025 * nd + 1.7 * repetition
            if score < best_score:
                best_score, best_path = score, new_path
            best_g[new] = nd
            counter += 1
            priority = nd + 2.2 * nh + 2.0 * repetition
            heapq.heappush(queue, (priority, counter, nd, new, new_path))

    return (list(best_path) if best_path else None), False


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=int)

    if "goals" not in memory:
        goals = _read_goals(obs)
        root = _extract_objects(obs)
        pads = _find_pads(obs, root)
        if not goals or not pads:
            memory.clear()
            return "RESET", None, None
        memory["goals"] = goals
        memory["pads"] = pads
        memory["visits"] = {}
        memory["plan"] = []
    else:
        goals = memory["goals"]
        pads = memory["pads"]
        predicted = memory.get("predicted")
        if predicted is not None and _prediction_matches(obs, predicted, pads):
            root = predicted
        else:
            root = _extract_objects(obs)
            memory["plan"] = []

    visits = memory.setdefault("visits", {})
    visits[root] = visits.get(root, 0) + 1
    if len(visits) > 350:
        for key in list(visits)[:100]:
            visits.pop(key, None)

    # An infeasible perturbed state, or an unexpected lost resource, can be
    # recovered with undo when it was our own move; otherwise restart the level.
    if _heuristic(root, goals, pads) >= _BIG:
        previous = memory.get("previous")
        if previous is not None and not memory.get("just_undid", False):
            memory["predicted"] = previous
            memory["plan"] = []
            memory["just_undid"] = True
            memory.pop("previous", None)
            return "ACTION7", None, None
        memory.clear()
        return "RESET", None, None

    memory["just_undid"] = False
    plan = memory.get("plan", [])
    if not plan:
        plan, winning = _search(root, goals, pads, visits)
        if plan:
            # Replan regularly unless the search found a complete solution.
            if not winning:
                plan = plan[:3]
            memory["plan"] = plan

    if not plan:
        if visits.get(root, 0) >= 3:
            memory.clear()
            return "RESET", None, None
        # A safe fallback still evaluates all locally meaningful pulls.
        best = None
        for click in _candidate_clicks(root, pads):
            result = _simulate(root, click, goals, pads)
            if result is None:
                continue
            new, won = result
            score = -1000 if won else (
                _heuristic(new, goals, pads) + 2 * visits.get(new, 0)
            )
            if new != root and (best is None or score < best[0]):
                best = (score, click, new)
        if best is None or best[0] >= _BIG:
            memory.clear()
            return "RESET", None, None
        _, click, predicted = best
        memory["plan"] = []
    else:
        click, predicted = plan.pop(0)
        memory["plan"] = plan

    memory["previous"] = root
    memory["predicted"] = predicted
    x, y = click
    return "ACTION6", int(y), int(x)

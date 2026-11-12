import math
import numpy as np

SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)

# Creature pixels, relative to their 5-by-4 bounding boxes.
PATTERNS = {
    1: ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)),
    2: ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2), (4, 2),
        (1, 3), (3, 3)),
    3: ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)),
}
ECOLORS = {1: 7, 2: 14, 3: 13}


def _creatures(frame, top, bottom, left, right, header=False):
    candidates = []
    for level, pattern in PATTERNS.items():
        color = ECOLORS[level]
        marks = set(pattern)
        holes = [(x, y) for y in range(4) for x in range(5)
                 if (x, y) not in marks]
        possible_colors = (color, 2) if header else (color,)
        for y in range(top, bottom - 3):
            for x in range(left, right - 4):
                for c in possible_colors:
                    good = sum(int(frame[y + dy, x + dx] == c)
                               for dx, dy in pattern)
                    if header:
                        if good != len(pattern):
                            continue
                        if any(frame[y + dy, x + dx] == c
                               for dx, dy in holes):
                            continue
                        # Header icons are drawn on the color-4 background.
                        if sum(int(frame[y + dy, x + dx] == 4)
                               for dx, dy in holes) < len(holes) - 1:
                            continue
                    else:
                        if good < len(pattern) - 1:
                            continue
                        if sum(int(frame[y + dy, x + dx] == c)
                               for dx, dy in holes) > 1:
                            continue
                    candidates.append((good, level, x, y, c))

    candidates.sort(reverse=True)
    result = []
    for good, level, x, y, c in candidates:
        if any(abs(x - ox) < 5 and abs(y - oy) < 4
               for _, ox, oy, _ in result):
            continue
        result.append((level, x, y, c))
    return result


def _read_goals(frame):
    goals = []
    used = set()
    for level, x, y, color in _creatures(frame, 0, 10, 24, 64, True):
        goals.append((1, level))
        for dx, dy in PATTERNS[level]:
            used.add((x + dx, y + dy))

    seen = set()
    for y in range(10):
        for x in range(24, 64):
            color = int(frame[y, x])
            if color in (4, 5) or (x, y) in used or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            component = []
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py),
                               (px, py - 1), (px, py + 1)):
                    if (24 <= nx < 64 and 0 <= ny < 10
                            and (nx, ny) not in seen
                            and (nx, ny) not in used
                            and int(frame[ny, nx]) == color):
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs = [p[0] for p in component]
            ys = [p[1] for p in component]
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            if w != h or len(component) != w * h:
                continue
            if w not in SIZES:
                continue
            level = SIZES.index(w)
            if color == COLORS[level] or color == 2:
                goals.append((0, level))

    counts = {}
    for goal in goals:
        counts[goal] = counts.get(goal, 0) + 1
    return tuple(sorted((kind, level, n)
                        for (kind, level), n in counts.items()))


def _read_pools(frame):
    # Rounded 9-by-9 pool template.
    shape = []
    for dy in range(9):
        if dy in (0, 8):
            xx = range(2, 7)
        elif dy in (1, 7):
            xx = range(1, 8)
        else:
            xx = range(9)
        shape.extend((dx, dy) for dx in xx)

    # Candidate top-lefts: x=0..55, y=10..54.
    score = np.zeros((45, 56), dtype=np.int16)
    for dx, dy in shape:
        score += (frame[10 + dy:55 + dy, dx:56 + dx] == 9)
    corners_ok = np.ones((45, 56), dtype=bool)
    for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
        corners_ok &= frame[10 + dy:55 + dy, dx:56 + dx] != 9

    ys, xs = np.where((score >= 46) & corners_ok)
    choices = sorted(((int(score[y, x]), int(x), int(y) + 10)
                      for y, x in zip(ys, xs)), reverse=True)
    pools = []
    for value, x, y in choices:
        if any(abs(x - px) <= 6 and abs(y - py) <= 6
               for px, py in pools):
            continue
        pools.append((x, y))
    return tuple(sorted(pools))


def _read_objects(frame):
    enemies = _creatures(frame, 10, 63, 0, 64)
    objects = [(1, level, x, y) for level, x, y, _ in enemies]

    # Integral images give the number of matching pixels in each square.
    for level in range(8, -1, -1):
        n = SIZES[level]
        color = COLORS[level]
        mask = (frame[10:63, :] == color).astype(np.int16)
        integ = np.pad(mask.cumsum(axis=0).cumsum(axis=1),
                       ((1, 0), (1, 0)))
        sums = (integ[n:, n:] - integ[:-n, n:]
                - integ[n:, :-n] + integ[:-n, :-n])
        area = n * n
        if level == 6:
            threshold = area  # Do not mistake the rounded pools for fruit.
        else:
            threshold = area if area <= 9 else area - max(1, area // 12)
        ys, xs = np.where(sums >= threshold)
        candidates = sorted(((int(sums[y, x]), int(x), int(y) + 10)
                             for y, x in zip(ys, xs)), reverse=True)
        accepted = []
        for matches, x, y in candidates:
            if any(x < ax + n and ax < x + n
                   and y < ay + n and ay < y + n
                   for ax, ay in accepted):
                continue
            if matches != area:
                patch = frame[y:y + n, x:x + n]
                missing = patch[patch != color]
                # Background and pool pixels cannot cover a fruit.
                if any(int(v) in (4, 5) for v in missing):
                    continue
            accepted.append((x, y))
            objects.append((0, level, x, y))
    return tuple(sorted(objects))


def _size(obj):
    return (SIZES[obj[1]], SIZES[obj[1]]) if obj[0] == 0 else (5, 4)


def _center(obj):
    w, h = _size(obj)
    return obj[2] + w // 2, obj[3] + h // 2


def _overlap(a, b):
    aw, ah = _size(a)
    bw, bh = _size(b)
    return (a[2] < b[2] + bw and b[2] < a[2] + aw
            and a[3] < b[3] + bh and b[3] < a[3] + ah)


def _selected(obj, x, y):
    w, h = _size(obj)
    dx = max(obj[2] - x, 0, x - (obj[2] + w - 1))
    dy = max(obj[3] - y, 0, y - (obj[3] + h - 1))
    return dx * dx + dy * dy <= 64


def _clamp(x, y, kind, level):
    w, h = (SIZES[level], SIZES[level]) if kind == 0 else (5, 4)
    return max(0, min(64 - w, x)), max(10, min(64 - h, y))


def _inside_pool(obj, pools):
    x, y = _center(obj)
    return any(px <= x < px + 9 and py <= y < py + 9
               for px, py in pools)


def _won(objects, goals, pools):
    if not pools or not goals:
        return False
    for kind, level, need in goals:
        count = sum(1 for o in objects
                    if o[0] == kind and o[1] == level
                    and _inside_pool(o, pools))
        if count != need:
            return False
    return True


def _components(indices, objects):
    parent = {i: i for i in indices}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos, i in enumerate(indices):
        for j in indices[pos + 1:]:
            if _overlap(objects[i], objects[j]):
                if objects[i][1] != objects[j][1]:
                    return None
                a, b = root(i), root(j)
                if a != b:
                    parent[b] = a
    groups = {}
    for i in indices:
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _simulate(state, click, goals, pools):
    # Mutable records are [kind, level, x, y, selected, alive, cooldown].
    xclick, yclick = click
    records = []
    velocities = {}
    for obj in state:
        chosen = _selected(obj, xclick, yclick)
        r = [obj[0], obj[1], obj[2], obj[3], chosen, True, 0]
        idx = len(records)
        records.append(r)
        if chosen and obj[0] == 1 and obj[1] == 1:
            cx, cy = _center(obj)
            dx, dy = xclick - cx, yclick - cy
            d = math.hypot(dx, dy)
            velocities[idx] = (dx / d, dy / d) if d else (0.0, 0.0)

    if not any(r[4] for r in records):
        return state, False

    bites = {}
    merged_fruit = False

    for tick in range(18):
        pause = any(t >= 4 and records[i][1] > 0
                    for i, (t, _, _, _, _) in bites.items()
                    if records[i][5])

        # Attraction movement.
        for i, r in enumerate(records):
            if not r[5] or not r[4] or i in bites:
                continue
            if r[0] == 1 and pause:
                continue
            if i in velocities:
                vx, vy = velocities[i]
                nx = int(round(r[2] + 3.4 * vx))
                ny = int(round(r[3] + 3.4 * vy))
            else:
                cx, cy = _center(r)
                dx, dy = xclick - cx, yclick - cy
                nx = r[2] + max(-4, min(4, dx))
                ny = r[3] + max(-4, min(4, dy))
            r[2], r[3] = _clamp(nx, ny, r[0], r[1])

        # Advance bite animations.
        finished = []
        for i, (timer, ox, oy, vx, vy) in list(bites.items()):
            r = records[i]
            if not r[5]:
                finished.append(i)
                continue
            if timer < 4:
                offset = -1 if timer % 4 == 1 else (1 if timer % 4 == 3 else 0)
                nx, ny = ox, oy + offset
            else:
                if timer == 4 and r[1] > 0:
                    r[1] -= 1
                k = timer - 3
                nx = int(round(ox + vx * 2.5 * k))
                ny = int(round(oy + vy * 2.5 * k))
            r[2], r[3] = _clamp(nx, ny, 0, r[1])
            timer += 1
            if timer >= (4 if r[1] == 0 and timer <= 4 else 8):
                # A fruit that started at level zero disappears; a fruit
                # downgraded to zero survives.
                if timer == 4 and r[1] == 0:
                    r[5] = False
                finished.append(i)
            else:
                bites[i] = (timer, ox, oy, vx, vy)
        for i in finished:
            bites.pop(i, None)

        for r in records:
            if r[6] > 0:
                r[6] -= 1

        if tick < 4 and not pause:
            fruits = [i for i, r in enumerate(records)
                      if r[5] and r[0] == 0 and i not in bites]
            for i, r in enumerate(records):
                if not r[5] or r[0] != 1 or r[4] or r[6] > 0:
                    continue
                if not fruits:
                    break
                cx, cy = _center(r)
                target = min(fruits,
                             key=lambda j: (_center(records[j])[0] - cx) ** 2
                             + (_center(records[j])[1] - cy) ** 2)
                tx, ty = _center(records[target])
                speed = 2 if r[1] == 3 else 1
                dx = speed if tx > cx else (-speed if tx < cx else 0)
                dy = speed if ty > cy else (-speed if ty < cy else 0)
                r[2], r[3] = _clamp(r[2] + dx, r[3] + dy, 1, r[1])

            for i, r in enumerate(records):
                if not r[5] or r[0] != 1 or r[4] or r[6] > 0:
                    continue
                for j, f in enumerate(records):
                    if not f[5] or f[0] != 0 or j in bites:
                        continue
                    if not _overlap(r, f):
                        continue
                    ex, ey = _center(r)
                    fx, fy = _center(f)
                    dx, dy = fx - ex, fy - ey
                    d = math.hypot(dx, dy)
                    vx, vy = (dx / d, dy / d) if d else (0.0, -1.0)
                    bites[j] = (0, float(f[2]), float(f[3]), vx, vy)
                    r[6] = 9
                    break

            # Creature mergers are checked every active animation frame.
            enemy_indices = [i for i, r in enumerate(records)
                             if r[5] and r[0] == 1]
            groups = _components(enemy_indices, records)
            if groups is None:
                return None, False
            for group in groups:
                level = records[group[0]][1]
                if level < 3:
                    cx = sum(_center(records[i])[0] for i in group) // len(group)
                    cy = sum(_center(records[i])[1] for i in group) // len(group)
                    nx, ny = _clamp(cx - 2, cy - 2, 1, level + 1)
                    records.append([1, level + 1, nx, ny, False, True, 4])
                for i in group:
                    records[i][5] = False

        current = tuple((r[0], r[1], r[2], r[3])
                        for r in records if r[5])
        if _won(current, goals, pools):
            return tuple(sorted(current)), True

        # Fruit mergers occur once, after four attraction frames.
        if tick >= 3 and not merged_fruit:
            merged_fruit = True
            selected_fruits = [i for i, r in enumerate(records)
                               if r[5] and r[0] == 0 and r[4]]
            groups = _components(selected_fruits, records)
            if groups is None:
                return None, False
            for group in groups:
                level = records[group[0]][1]
                if level < 8:
                    cx = sum(_center(records[i])[0] for i in group) // len(group)
                    cy = sum(_center(records[i])[1] for i in group) // len(group)
                    size = SIZES[level + 1]
                    nx, ny = _clamp(cx - size // 2, cy - size // 2,
                                    0, level + 1)
                    records.append([0, level + 1, nx, ny, False, True, 0])
                for i in group:
                    records[i][5] = False
                    bites.pop(i, None)

        current = tuple((r[0], r[1], r[2], r[3])
                        for r in records if r[5])
        if _won(current, goals, pools):
            return tuple(sorted(current)), True
        if tick >= 3 and not bites:
            break

    result = tuple(sorted((r[0], r[1], r[2], r[3])
                          for r in records if r[5]))
    if not any(o[0] == 0 for o in result):
        return None, False
    return result, False


def _pool_distance(obj, pools):
    if not pools:
        return 0
    x, y = _center(obj)
    return min(max(max(px - x, 0, x - (px + 8)),
                   max(py - y, 0, y - (py + 8)))
               for px, py in pools)


def _distance(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _score(state, goals, pools):
    fruits = [o for o in state if o[0] == 0]
    enemies = [o for o in state if o[0] == 1]
    fg = [(level, n) for kind, level, n in goals if kind == 0]
    eg = [(level, n) for kind, level, n in goals if kind == 1]
    value = 0.0

    if fg:
        required_mass = sum((1 << level) * n for level, n in fg)
        mass = sum(1 << o[1] for o in fruits)
        max_level = max(level for level, _ in fg)
        if mass < required_mass:
            value += 160.0 * (required_mass - mass) / max(1, required_mass)
        elif mass > required_mass:
            value += 9.0 * math.log2(mass / required_mass)
        value += 10.0 * sum((1 << o[1]) * abs(max_level - o[1])
                            for o in fruits) / max(1, required_mass)
        value += 0.10 * sum((1 << o[1]) * _pool_distance(o, pools)
                            for o in fruits) / max(1, mass)

        for level, need in fg:
            matches = [o for o in fruits if o[1] == level]
            value += 18.0 * max(0, need - len(matches))
            distances = sorted(_pool_distance(o, pools) for o in matches)
            value += 0.60 * sum(distances[:need])
            inside = sum(_inside_pool(o, pools) for o in matches)
            value += 25.0 * max(0, inside - need)

        # Encourage equal-level material to approach without making a
        # distant next-level pair outweigh the benefit of a merger.
        for level in range(max_level):
            group = [o for o in fruits if o[1] == level]
            if len(group) >= 2:
                nearest = min(_distance(a, b)
                              for i, a in enumerate(group)
                              for b in group[i + 1:])
                value += 0.12 * min(25.0, nearest)

    for level, need in eg:
        matches = [o for o in enemies if o[1] == level]
        value += 20.0 * max(0, need - len(matches))
        value += 0.55 * sum(sorted(_pool_distance(o, pools)
                                   for o in matches)[:need])
        inside = sum(_inside_pool(o, pools) for o in matches)
        value += 25.0 * max(0, inside - need)
        if len(matches) < need and level > 1:
            for lower in range(1, level):
                group = [o for o in enemies if o[1] == lower]
                if len(group) >= 2:
                    nearest = min(_distance(a, b)
                                  for i, a in enumerate(group)
                                  for b in group[i + 1:])
                    value += 0.20 * min(40.0, nearest)

    return value


def _candidates(state, goals, pools):
    points = set()
    pool_centers = [(x + 4, y + 4) for x, y in pools]
    max_fruit_goal = max((level for kind, level, _ in goals if kind == 0),
                         default=8)

    def add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < 64 and 10 <= y < 63:
            points.add((x, y))

    def toward(obj, tx, ty):
        cx, cy = _center(obj)
        dx, dy = tx - cx, ty - cy
        d = math.hypot(dx, dy)
        if d < 0.01:
            add(cx, cy)
            return
        ux, uy = dx / d, dy / d
        # Find a far point whose radius-8 selection circle still reaches
        # the object's bounding box.
        last = None
        for step in range(1, 19):
            x = int(round(cx + ux * step))
            y = int(round(cy + uy * step))
            if 0 <= x < 64 and 10 <= y < 63 and _selected(obj, x, y):
                last = (x, y)
        if last:
            add(*last)
        if d <= 18:
            add(tx, ty)
        add(cx + ux * min(6.0, d), cy + uy * min(6.0, d))

    for i, obj in enumerate(state):
        cx, cy = _center(obj)
        for tx, ty in pool_centers:
            toward(obj, tx, ty)
        for other in state:
            if other is obj:
                continue
            if obj[0] == other[0] and obj[1] == other[1]:
                tx, ty = _center(other)
                toward(obj, tx, ty)
                add((cx + tx) / 2.0, (cy + ty) / 2.0)
            elif (obj[0] == 0 and other[0] == 1
                  and obj[1] > max_fruit_goal):
                tx, ty = _center(other)
                toward(obj, tx, ty)

        # General escape and approach directions.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            toward(obj, cx + 30 * dx, cy + 30 * dy)

    return sorted(points)


def act(obs, available, memory):
    frame = np.asarray(obs)
    if "goals" not in memory:
        memory["goals"] = _read_goals(frame)
    if "pools" not in memory:
        memory["pools"] = _read_pools(frame)

    goals = tuple(tuple(g) for g in memory.get("goals", ()))
    pools = tuple(tuple(p) for p in memory.get("pools", ()))
    if not goals:
        goals = _read_goals(frame)
        if goals:
            memory["goals"] = goals
    if not pools:
        pools = _read_pools(frame)
        if pools:
            memory["pools"] = pools

    state = _read_objects(frame)
    if not state:
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION6", 32, 32)

    # Red goal icons indicate that insufficient fruit value remains.
    required = sum((1 << level) * n for kind, level, n in goals
                   if kind == 0)
    mass = sum(1 << o[1] for o in state if o[0] == 0)
    if (required and mass < required and np.any(frame[:10, 24:] == 2)
            and "RESET" in available):
        return ("RESET", None, None)

    if "ACTION6" not in available:
        if "ACTION7" in available:
            return ("ACTION7", None, None)
        return (available[0], None, None)

    initial_score = _score(state, goals, pools)
    best = None
    best_value = float("inf")
    beam = [(state, None)]
    seen = {state: 0}

    # Replan each turn; a small beam keeps this bounded on crowded layouts.
    for depth in range(1, 4):
        next_beam = []
        for current, first in beam:
            for click in _candidates(current, goals, pools):
                new_state, won = _simulate(current, click, goals, pools)
                if new_state is None:
                    continue
                first_click = click if first is None else first
                if won:
                    return ("ACTION6", int(first_click[1]),
                            int(first_click[0]))
                if new_state == current:
                    continue
                if new_state in seen and seen[new_state] <= depth:
                    continue
                seen[new_state] = depth
                cost = _score(new_state, goals, pools) + 0.35 * depth
                if cost < best_value:
                    best_value = cost
                    best = first_click
                next_beam.append((cost, new_state, first_click))
        if not next_beam:
            break
        next_beam.sort(key=lambda item: item[0])
        beam = [(s, first) for _, s, first in next_beam[:7]]

    if best is None:
        # Move the first fruit toward the nearest known pool.
        fruit = next((o for o in state if o[0] == 0), state[0])
        cx, cy = _center(fruit)
        if pools:
            px, py = min(((x + 4, y + 4) for x, y in pools),
                         key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            dx, dy = px - cx, py - cy
            d = max(1.0, math.hypot(dx, dy))
            best = (max(0, min(63, int(round(cx + 8 * dx / d)))),
                    max(10, min(62, int(round(cy + 8 * dy / d)))))
        else:
            best = (cx, cy)

    return ("ACTION6", int(best[1]), int(best[0]))

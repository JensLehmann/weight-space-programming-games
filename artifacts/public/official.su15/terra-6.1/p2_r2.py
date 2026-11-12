import math
import numpy as np

_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)

_ENEMY_PIXELS = {
    1: (7, ((2, 0), (1, 1), (3, 1), (0, 2), (4, 2),
            (1, 3), (2, 3), (3, 3))),
    2: (14, ((2, 0), (1, 1), (3, 1), (0, 2), (2, 2), (4, 2),
             (1, 3), (3, 3))),
    3: (13, ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
             (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
             (1, 3), (2, 3), (3, 3))),
}

_GOAL_MASK = tuple(
    (x, y)
    for y, row in enumerate((
        "..#####..",
        ".#######.",
        "#########",
        "#########",
        "#########",
        "#########",
        "#########",
        ".#######.",
        "..#####..",
    ))
    for x, c in enumerate(row) if c == "#"
)
_GOAL_SET = set(_GOAL_MASK)


def _rect_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _near_click(x, y, ox, oy, w, h):
    dx = max(ox - x, 0, x - (ox + w - 1))
    dy = max(oy - y, 0, y - (oy + h - 1))
    return dx * dx + dy * dy <= 64


def _clamp_pos(x, y, w, h):
    return max(0, min(64 - w, x)), max(10, min(64 - h, y))


def _find_goals(a):
    candidates = []
    for color in (9, 2):
        for y in range(10, 56):
            for x in range(56):
                if sum(
                    int(a[y + dy, x + dx] == color)
                    for dx, dy in ((2, 0), (4, 0), (6, 0),
                                   (0, 4), (8, 4))
                ) < 3:
                    continue
                missing = 0
                extra = 0
                for dy in range(9):
                    yy = y + dy
                    if yy >= 63:
                        continue
                    for dx in range(9):
                        is_color = int(a[yy, x + dx]) == color
                        should = (dx, dy) in _GOAL_SET
                        if should and not is_color:
                            missing += 1
                        elif is_color and not should:
                            extra += 1
                error = missing + extra
                # Decorations can cover ten pixels of a goal.
                if error <= 12:
                    candidates.append((error, x, y))

    candidates.sort()
    found = []
    for error, x, y in candidates:
        if all(abs(x - xx) > 6 or abs(y - yy) > 6 for xx, yy in found):
            found.append((x, y))
    return tuple(sorted(found))


def _enemy_match(a, x, y, typ, y_limit, partial=False):
    color, points = _ENEMY_PIXELS[typ]
    if x < 0 or x + 5 > 64 or y < 0 or y + 4 > 64:
        return False
    point_set = set(points)
    matches = 0
    possible = 0
    extras = 0
    for dy in range(4):
        yy = y + dy
        if yy >= y_limit:
            continue
        for dx in range(5):
            v = int(a[yy, x + dx])
            if (dx, dy) in point_set:
                possible += 1
                matches += (v == color)
            elif v == color:
                extras += 1
    if partial:
        return (possible > 0
                and matches >= max(3, (possible * 2 + 2) // 3)
                and extras <= 2)
    return possible > 0 and matches == possible and extras == 0


def _read_header(a):
    work = a.copy()
    fruit_targets = []
    enemy_targets = []

    for typ in (3, 2, 1):
        color, points = _ENEMY_PIXELS[typ]
        for y in range(7):
            for x in range(24, 60):
                if _enemy_match(work, x, y, typ, 10):
                    enemy_targets.append(typ)
                    for dx, dy in points:
                        work[y + dy, x + dx] = 4

    for typ in range(8, -1, -1):
        color, size = _COLORS[typ], _SIZES[typ]
        for y in range(0, 11 - size):
            for x in range(24, 65 - size):
                if np.all(work[y:y + size, x:x + size] == color):
                    fruit_targets.append(typ)
                    work[y:y + size, x:x + size] = 4

    return tuple(sorted(fruit_targets)), tuple(sorted(enemy_targets))


def _read_objects(a, prediction=None):
    fruits = []
    enemies = []
    occupied = np.zeros((64, 64), dtype=bool)

    for typ in range(8, -1, -1):
        color, size = _COLORS[typ], _SIZES[typ]
        for y in range(10, 65 - size):
            visible_h = min(size, 63 - y)
            if visible_h <= 0:
                continue
            for x in range(65 - size):
                if int(a[y, x]) != color or occupied[y, x]:
                    continue
                region = a[y:y + visible_h, x:x + size]
                if (np.all(region == color)
                        and not np.any(occupied[y:y + visible_h, x:x + size])):
                    fruits.append((typ, x, y))
                    occupied[y:y + visible_h, x:x + size] = True

    for typ in (3, 2, 1):
        for y in range(10, 61):
            for x in range(60):
                if _enemy_match(a, x, y, typ, 63):
                    if all(abs(x - ex) > 2 or abs(y - ey) > 2
                           for _, ex, ey in enemies):
                        enemies.append((typ, x, y))

    if prediction:
        pf, pe = prediction
        for typ, x, y in pf:
            size, color = _SIZES[typ], _COLORS[typ]
            if not (0 <= x <= 64 - size and 10 <= y <= 64 - size):
                continue
            if any(
                _rect_overlap((x, y, size, size),
                              (fx, fy, _SIZES[ft], _SIZES[ft]))
                for ft, fx, fy in fruits
            ):
                continue
            h = min(size, 63 - y)
            if h > 0:
                region = a[y:y + h, x:x + size]
                if np.count_nonzero(region == color) >= max(
                    1, (region.size * 3 + 3) // 4
                ):
                    fruits.append((typ, x, y))

        for typ, x, y in pe:
            if any(abs(x - ex) <= 2 and abs(y - ey) <= 2
                   for _, ex, ey in enemies):
                continue
            if _enemy_match(a, x, y, typ, 63, partial=True):
                enemies.append((typ, x, y))

    return tuple(sorted(fruits)), tuple(sorted(enemies))


def _inside_goal(x, y, goals):
    return any(gx <= x < gx + 9 and gy <= y < gy + 9
               for gx, gy in goals)


def _won(fruits, enemies, goals, ft, et):
    for typ in set(ft):
        count = sum(
            1 for t, x, y in fruits
            if t == typ and _inside_goal(
                x + _SIZES[t] // 2, y + _SIZES[t] // 2, goals
            )
        )
        if count != ft.count(typ):
            return False
    for typ in set(et):
        count = sum(
            1 for t, x, y in enemies
            if t == typ and _inside_goal(x + 2, y + 2, goals)
        )
        if count != et.count(typ):
            return False
    return bool(ft or et)


def _merge_enemies(enemies):
    n = len(enemies)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = enemies[i]
        for j in range(i + 1, n):
            b = enemies[j]
            if _rect_overlap((a["x"], a["y"], 5, 4),
                             (b["x"], b["y"], 5, 4)):
                if a["t"] != b["t"]:
                    return None
                parent[root(j)] = root(i)

    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(enemies[i])

    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
            continue
        typ = group[0]["t"]
        if typ >= 3:
            continue
        cx = sum(e["x"] + 2 for e in group) // len(group)
        cy = sum(e["y"] + 2 for e in group) // len(group)
        x, y = _clamp_pos(cx - 2, cy - 2, 5, 4)
        result.append({
            "t": typ + 1, "x": x, "y": y,
            "sel": False, "cool": 4, "vx": 0.0, "vy": 0.0
        })
    return result


def _merge_fruits(fruits):
    selected = [f for f in fruits if f["sel"]]
    n = len(selected)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = selected[i]
        sa = _SIZES[a["t"]]
        for j in range(i + 1, n):
            b = selected[j]
            sb = _SIZES[b["t"]]
            if _rect_overlap((a["x"], a["y"], sa, sa),
                             (b["x"], b["y"], sb, sb)):
                if a["t"] != b["t"]:
                    return None
                parent[root(j)] = root(i)

    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(selected[i])

    removed = set()
    added = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for f in group:
            removed.add(id(f))
        typ = group[0]["t"]
        if typ >= 8:
            continue
        old_size = _SIZES[typ]
        cx = sum(f["x"] + old_size // 2 for f in group) // len(group)
        cy = sum(f["y"] + old_size // 2 for f in group) // len(group)
        size = _SIZES[typ + 1]
        x, y = _clamp_pos(cx - size // 2, cy - size // 2, size, size)
        added.append({
            "t": typ + 1, "x": x, "y": y,
            "sel": False, "hit": None
        })

    return [f for f in fruits if id(f) not in removed] + added


def _simulate(state, click, goals, ft, et):
    px, py = click
    fs, es = state
    fruits = []
    enemies = []

    for typ, x, y in fs:
        size = _SIZES[typ]
        fruits.append({
            "t": typ, "x": x, "y": y,
            "sel": _near_click(px, py, x, y, size, size),
            "hit": None
        })

    for typ, x, y in es:
        selected = _near_click(px, py, x, y, 5, 4)
        dx, dy = px - (x + 2), py - (y + 2)
        length = math.hypot(dx, dy)
        vx = dx / length if length else 0.0
        vy = dy / length if length else 0.0
        enemies.append({
            "t": typ, "x": x, "y": y, "sel": selected,
            "cool": 0, "vx": vx, "vy": vy
        })

    for frame in range(15):
        paused = any(
            f["hit"] is not None and not f["hit"]["dead"]
            and 4 <= f["hit"]["age"] < 8
            for f in fruits
        )

        for f in fruits:
            if not f["sel"] or f["hit"] is not None:
                continue
            size = _SIZES[f["t"]]
            cx = f["x"] + size // 2
            cy = f["y"] + size // 2
            dx = max(-4, min(4, px - cx))
            dy = max(-4, min(4, py - cy))
            f["x"], f["y"] = _clamp_pos(
                f["x"] + dx, f["y"] + dy, size, size
            )

        if not paused:
            for e in enemies:
                if not e["sel"]:
                    continue
                if e["t"] == 1:
                    nx = int(round(e["x"] + 3.4 * e["vx"]))
                    ny = int(round(e["y"] + 3.4 * e["vy"]))
                else:
                    nx = e["x"] + max(-4, min(4, px - (e["x"] + 2)))
                    ny = e["y"] + max(-4, min(4, py - (e["y"] + 2)))
                e["x"], e["y"] = _clamp_pos(nx, ny, 5, 4)

        to_remove = []
        for f in fruits:
            hit = f["hit"]
            if hit is None:
                continue
            age = hit["age"]
            if age < 4:
                offset = -1 if age % 4 == 1 else (1 if age % 4 == 3 else 0)
                nx, ny = hit["sx"], hit["sy"] + offset
            else:
                if not hit["changed"]:
                    f["t"] -= 1
                    hit["changed"] = True
                hit["fx"] += hit["vx"] * 2.5
                hit["fy"] += hit["vy"] * 2.5
                nx, ny = round(hit["fx"]), round(hit["fy"])
            size = _SIZES[max(0, f["t"])]
            f["x"], f["y"] = _clamp_pos(int(nx), int(ny), size, size)
            hit["age"] += 1
            if hit["age"] >= (4 if hit["dead"] else 8):
                if hit["dead"]:
                    to_remove.append(f)
                else:
                    f["hit"] = None

        for f in to_remove:
            fruits.remove(f)

        for e in enemies:
            if e["cool"] > 0:
                e["cool"] -= 1

        if frame < 4 and not paused and fruits and enemies:
            for e in enemies:
                if e["sel"] or e["cool"] > 0:
                    continue
                candidates = [f for f in fruits if f["hit"] is None]
                if not candidates:
                    break
                ex, ey = e["x"] + 2, e["y"] + 2
                target = min(
                    candidates,
                    key=lambda f: (
                        (f["x"] + _SIZES[f["t"]] // 2 - ex) ** 2
                        + (f["y"] + _SIZES[f["t"]] // 2 - ey) ** 2
                    )
                )
                size = _SIZES[target["t"]]
                tx = target["x"] + size // 2
                ty = target["y"] + size // 2
                speed = 2 if e["t"] == 3 else 1
                dx = speed if tx > ex else (-speed if tx < ex else 0)
                dy = speed if ty > ey else (-speed if ty < ey else 0)
                e["x"], e["y"] = _clamp_pos(
                    e["x"] + dx, e["y"] + dy, 5, 4
                )

            for e in enemies:
                if e["sel"] or e["cool"] > 0:
                    continue
                for f in fruits:
                    if f["hit"] is not None:
                        continue
                    size = _SIZES[f["t"]]
                    if not _rect_overlap(
                        (e["x"], e["y"], 5, 4),
                        (f["x"], f["y"], size, size)
                    ):
                        continue
                    dx = f["x"] + size // 2 - (e["x"] + 2)
                    dy = f["y"] + size // 2 - (e["y"] + 2)
                    length = math.hypot(dx, dy)
                    vx, vy = ((dx / length, dy / length)
                              if length else (0.0, -1.0))
                    f["hit"] = {
                        "age": 0, "dead": f["t"] == 0,
                        "sx": f["x"], "sy": f["y"],
                        "fx": float(f["x"]), "fy": float(f["y"]),
                        "vx": vx, "vy": vy, "changed": False
                    }
                    e["cool"] = 9

            enemies = _merge_enemies(enemies)
            if enemies is None:
                return None, False

        if frame == 3:
            enemies = _merge_enemies(enemies)
            if enemies is None:
                return None, False
            fruits = _merge_fruits(fruits)
            if fruits is None:
                return None, False

        fstate = tuple(sorted((f["t"], f["x"], f["y"]) for f in fruits))
        estate = tuple(sorted((e["t"], e["x"], e["y"]) for e in enemies))
        if _won(fstate, estate, goals, ft, et):
            return (fstate, estate), True
        if frame >= 3 and not any(f["hit"] is not None for f in fruits):
            break

    if not fruits:
        return None, False
    return (fstate, estate), False


def _goal_distance(x, y, goals):
    if not goals:
        return 60.0
    return min(
        max(max(gx - x, 0, x - (gx + 8)),
            max(gy - y, 0, y - (gy + 8)))
        + 0.35 * min(max(gx - x, 0, x - (gx + 8)),
                     max(gy - y, 0, y - (gy + 8)))
        for gx, gy in goals
    )


def _score(state, goals, ft, et):
    fruits, enemies = state
    mass = sum(1 << t for t, _, _ in fruits)
    required = sum(1 << t for t in ft)
    value = -5000.0 * max(0, required - mass)
    value -= 13.0 * len(fruits)

    if ft:
        max_target = max(ft)
        for typ in set(ft):
            ds = sorted(
                _goal_distance(x + _SIZES[t] // 2,
                               y + _SIZES[t] // 2, goals)
                for t, x, y in fruits if t == typ
            )
            for d in ds[:ft.count(typ)]:
                value += 150.0 - 1.5 * d
                if d == 0:
                    value += 30.0

        for typ in range(max_target):
            positions = [
                (x + _SIZES[t] // 2, y + _SIZES[t] // 2)
                for t, x, y in fruits if t == typ
            ]
            while len(positions) >= 2:
                best = None
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = math.hypot(
                            positions[i][0] - positions[j][0],
                            positions[i][1] - positions[j][1]
                        )
                        if best is None or d < best[0]:
                            best = (d, i, j)
                d, i, j = best
                value -= 0.65 * d
                positions.pop(j)
                positions.pop(i)

        for typ, x, y in fruits:
            if typ > max_target:
                value -= 25.0 * (typ - max_target)
                if enemies:
                    size = _SIZES[typ]
                    cx, cy = x + size // 2, y + size // 2
                    d = min(
                        math.hypot(cx - (ex + 2), cy - (ey + 2))
                        for _, ex, ey in enemies
                    )
                    value -= 0.7 * d
                else:
                    value -= 1000.0

    for typ in set(et):
        ds = sorted(
            _goal_distance(x + 2, y + 2, goals)
            for t, x, y in enemies if t == typ
        )
        for d in ds[:et.count(typ)]:
            value += 100.0 - 1.4 * d
            if d == 0:
                value += 25.0

    if et:
        max_enemy = max(et)
        for typ in range(1, max_enemy):
            positions = [(x + 2, y + 2) for t, x, y in enemies if t == typ]
            while len(positions) >= 2:
                best = None
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = math.hypot(
                            positions[i][0] - positions[j][0],
                            positions[i][1] - positions[j][1]
                        )
                        if best is None or d < best[0]:
                            best = (d, i, j)
                d, i, j = best
                value -= 0.7 * d
                positions.pop(j)
                positions.pop(i)

    return value


def _candidate_clicks(state, goals, ft, et):
    fruits, enemies = state
    result = []
    seen = set()

    def add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < 64 and 10 <= y <= 62 and (x, y) not in seen:
            seen.add((x, y))
            result.append((x, y))

    def toward(cx, cy, w, h, tx, ty):
        dx, dy = tx - cx, ty - cy
        distance = math.hypot(dx, dy)
        if distance == 0:
            add(cx, cy)
            return
        ux, uy = dx / distance, dy / distance
        ox, oy = cx - w // 2, cy - h // 2
        lo, hi = 0.0, min(distance, 25.0)
        for _ in range(9):
            mid = (lo + hi) / 2
            xx, yy = round(cx + ux * mid), round(cy + uy * mid)
            if _near_click(xx, yy, ox, oy, w, h):
                lo = mid
            else:
                hi = mid
        add(cx + ux * lo, cy + uy * lo)
        if distance <= 16:
            add(cx + ux * min(distance, lo * 0.6),
                cy + uy * min(distance, lo * 0.6))

    for objects, is_enemy in ((fruits, False), (enemies, True)):
        for i, (t1, x1, y1) in enumerate(objects):
            s1 = 5 if is_enemy else _SIZES[t1]
            h1 = 4 if is_enemy else s1
            c1 = (x1 + s1 // 2, y1 + h1 // 2)
            for t2, x2, y2 in objects[i + 1:]:
                if t1 != t2:
                    continue
                s2 = 5 if is_enemy else _SIZES[t2]
                h2 = 4 if is_enemy else s2
                c2 = (x2 + s2 // 2, y2 + h2 // 2)
                add((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
                toward(c1[0], c1[1], s1, h1, c2[0], c2[1])
                toward(c2[0], c2[1], s2, h2, c1[0], c1[1])

    goal_centers = [(x + 4, y + 4) for x, y in goals]

    for typ, x, y in fruits:
        size = _SIZES[typ]
        cx, cy = x + size // 2, y + size // 2
        for gx, gy in goal_centers:
            toward(cx, cy, size, size, gx, gy)
        if ft and typ > max(ft):
            for _, ex, ey in enemies:
                toward(cx, cy, size, size, ex + 2, ey + 2)

    for typ, x, y in enemies:
        cx, cy = x + 2, y + 2
        for gx, gy in goal_centers:
            toward(cx, cy, 5, 4, gx, gy)
        if ft:
            for t, fx, fy in fruits:
                if t > max(ft):
                    size = _SIZES[t]
                    toward(cx, cy, 5, 4, fx + size // 2, fy + size // 2)

    directions = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    )
    for typ, x, y in fruits:
        size = _SIZES[typ]
        cx, cy = x + size // 2, y + size // 2
        for dx, dy in directions:
            toward(cx, cy, size, size, cx + dx * 30, cy + dy * 30)

    for typ, x, y in enemies:
        cx, cy = x + 2, y + 2
        for dx, dy in directions:
            toward(cx, cy, 5, 4, cx + dx * 30, cy + dy * 30)

    return result[:160]


def act(obs, available, memory):
    a = np.asarray(obs)
    goals = tuple(tuple(p) for p in memory.get("goals", ()))
    ft = tuple(memory.get("fruit_targets", ()))
    et = tuple(memory.get("enemy_targets", ()))

    if not goals:
        goals = _find_goals(a)
        if goals:
            memory["goals"] = [list(p) for p in goals]

    if not ft and not et:
        ft, et = _read_header(a)
        if ft or et:
            memory["fruit_targets"] = list(ft)
            memory["enemy_targets"] = list(et)

    if not goals or not (ft or et):
        memory.pop("prediction", None)
        return ("RESET", None, None)

    prediction = memory.pop("prediction", None)
    if prediction:
        prediction = (
            tuple(tuple(p) for p in prediction[0]),
            tuple(tuple(p) for p in prediction[1])
        )
    state = _read_objects(a, prediction)
    fruits, enemies = state

    required_mass = sum(1 << t for t in ft)
    if not fruits or sum(1 << t for t, _, _ in fruits) < required_mass:
        memory.pop("prediction", None)
        memory["history"] = []
        return ("RESET", None, None)

    history = [
        (tuple(tuple(v) for v in h[0]), tuple(tuple(v) for v in h[1]))
        for h in memory.get("history", [])
    ]
    history.append(state)
    history = history[-12:]
    memory["history"] = [
        [list(map(list, f)), list(map(list, e))]
        for f, e in history
    ]
    visited = set(history)

    beam = [(state, None, None)]
    best = None
    best_value = -float("inf")

    for depth in range(3):
        successors = {}
        for current, first_click, first_state in beam:
            for click in _candidate_clicks(current, goals, ft, et):
                nxt, won = _simulate(current, click, goals, ft, et)
                if nxt is None or nxt == current:
                    continue
                initial_click = click if first_click is None else first_click
                initial_state = nxt if first_state is None else first_state

                if won:
                    memory["prediction"] = [
                        [list(v) for v in initial_state[0]],
                        [list(v) for v in initial_state[1]]
                    ]
                    return ("ACTION6", initial_click[1], initial_click[0])

                score = _score(nxt, goals, ft, et)
                if nxt in visited:
                    score -= 22.0
                score -= depth * 0.15
                old = successors.get(nxt)
                if old is None or score > old[0]:
                    successors[nxt] = (
                        score, nxt, initial_click, initial_state
                    )
                if score > best_value:
                    best_value = score
                    best = (initial_click, initial_state)

        if not successors:
            break
        ranked = sorted(
            successors.values(), key=lambda item: item[0], reverse=True
        )[:12]
        beam = [
            (s, click, first_state)
            for _, s, click, first_state in ranked
        ]

    # If no short winning continuation was found and almost no budget
    # remains, make one attempt from the level's initial state.
    if (np.count_nonzero(a[63] == 0) <= 4
            and not memory.get("late_reset")
            and "RESET" in available):
        memory["late_reset"] = True
        memory["history"] = []
        memory.pop("prediction", None)
        return ("RESET", None, None)

    if best is None:
        x, y = fruits[0][1], fruits[0][2]
        return ("ACTION6", max(10, min(62, y)), max(0, min(63, x)))

    click, predicted = best
    memory["prediction"] = [
        [list(v) for v in predicted[0]],
        [list(v) for v in predicted[1]]
    ]
    return ("ACTION6", click[1], click[0])

from collections import deque
from itertools import permutations
import heapq
import numpy as np


MOVES = [
    ("ACTION1", 0, -3),
    ("ACTION2", 0, 3),
    ("ACTION3", -3, 0),
    ("ACTION4", 3, 0),
]
DELTA = {a: (dx, dy) for a, dx, dy in MOVES}


def segments(obs, color, dx, dy):
    result = []
    for y in range(63):
        for x in range(64):
            if 0 <= x - dx < 64 and 0 <= y - dy < 63:
                continue
            line = []
            xx, yy = x, y
            while 0 <= xx < 64 and 0 <= yy < 63:
                line.append((xx, yy, int(obs[yy, xx])))
                xx += dx
                yy += dy
            run = []
            for item in line + [(0, 0, 5)]:
                if item[2] not in (1, 2, 4, 5, 15):
                    run.append(item)
                else:
                    while run and run[0][2] not in (color, 0):
                        run.pop(0)
                    while run and run[-1][2] not in (color, 0):
                        run.pop()
                    if len(run) >= 5:
                        own = sum(t[2] == color for t in run)
                        if own >= 4 and own * 10 >= len(run) * 6:
                            result.append((
                                run[0][0], run[0][1],
                                run[-1][0], run[-1][1]
                            ))
                    run = []
    return result


def detect_objects(obs):
    counts = {}
    for v in obs[:63].ravel():
        v = int(v)
        if v not in (0, 1, 2, 4, 5, 15):
            counts[v] = counts.get(v, 0) + 1
    colors = [c for c, n in counts.items() if n >= 6]
    candidates = []
    seen = set()

    def add(color, kind, pixels, cx, cy):
        pixels = frozenset(pixels)
        if len(pixels) < 5 or not (0 <= cx < 64 and 0 <= cy < 64):
            return
        key = (color, pixels, cx, cy)
        if key in seen:
            return
        seen.add(key)
        own = set()
        missing = other = visible = 0
        for x, y in pixels:
            if not (0 <= x < 64 and 0 <= y < 63):
                continue
            visible += 1
            v = int(obs[y, x])
            if v == color or (v == 0 and x == cx and y == cy):
                own.add((x, y))
            elif v in (1, 2, 4, 5, 15):
                missing += 1
            else:
                other += 1
        if visible < 5 or len(own) < 5:
            return
        if missing > max(1, visible // 18):
            return
        if len(own) * 100 < visible * 72:
            return
        candidates.append({
            "color": color,
            "kind": kind,
            "pixels": pixels,
            "cx": cx,
            "cy": cy,
            "own": own,
            "quality": len(own) - 3 * missing - other,
        })

    for c in colors:
        hs = segments(obs, c, 1, 0)
        vs = segments(obs, c, 0, 1)
        ds = segments(obs, c, 1, 1)
        es = segments(obs, c, -1, 1)

        for x1, y, x2, _ in hs:
            for x, y1, _, y2 in vs:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    left, right, top, bottom = x1, x2, y1, y2
                    if left == 0:
                        left = min(left, 2 * x - right)
                    if right == 63:
                        right = max(right, 2 * x - left)
                    if top == 0:
                        top = min(top, 2 * y - bottom)
                    if bottom == 62:
                        bottom = max(bottom, 2 * y - top)
                    if right - left > 80 or bottom - top > 80:
                        continue
                    p = {(xx, y) for xx in range(left, right + 1)}
                    p.update((x, yy) for yy in range(top, bottom + 1))
                    add(c, "cross", p,
                        left + (right - left + 1) // 2,
                        top + (bottom - top + 1) // 2)

        for x1, y, x2, _ in hs:
            add(c, "line", {(x, y) for x in range(x1, x2 + 1)},
                (x1 + x2) // 2, y)
        for x, y1, _, y2 in vs:
            add(c, "line", {(x, y) for y in range(y1, y2 + 1)},
                x, (y1 + y2) // 2)

        for x1, y1, x2, y2 in ds:
            length = x2 - x1 + 1
            if length >= 7 and length % 2 == 1 and length == y2 - y1 + 1:
                p = set()
                for k in range(length):
                    p.add((x1 + k, y1 + k))
                    p.add((x2 - k, y1 + k))
                add(c, "x", p, x1 + length // 2, y1 + length // 2)

        for x1, y1, x2, y2 in es:
            length = x1 - x2 + 1
            if length >= 7 and length % 2 == 1 and length == y2 - y1 + 1:
                p = set()
                for k in range(length):
                    p.add((x2 + k, y1 + k))
                    p.add((x1 - k, y1 + k))
                add(c, "x", p, x2 + length // 2, y1 + length // 2)

        for x1, y1, x2, y2 in ds:
            r = x2 - x1
            if r >= 3 and r == y2 - y1:
                for cx, cy in ((x1, y1 + r), (x2, y1)):
                    p = set()
                    for d in range(-r, r + 1):
                        q = r - abs(d)
                        p.add((cx + d, cy + q))
                        p.add((cx + d, cy - q))
                    add(c, "outline", p, cx, cy)

        for x1, y1, x2, y2 in es:
            r = x1 - x2
            if r >= 3 and r == y2 - y1:
                for cx, cy in ((x1, y1 + r), (x2, y1)):
                    p = set()
                    for d in range(-r, r + 1):
                        q = r - abs(d)
                        p.add((cx + d, cy + q))
                        p.add((cx + d, cy - q))
                    add(c, "outline", p, cx, cy)

        for i, (x1, y1, x2, _) in enumerate(hs):
            for u1, y2, u2, _ in hs[i + 1:]:
                if y2 <= y1 or y2 - y1 < 4:
                    continue
                if abs(x1 - u1) > 1 or abs(x2 - u2) > 1:
                    continue
                left, right = min(x1, u1), max(x2, u2)
                if right - left < 4:
                    continue
                p = {(x, y1) for x in range(left, right + 1)}
                p.update((x, y2) for x in range(left, right + 1))
                p.update((left, y) for y in range(y1, y2 + 1))
                p.update((right, y) for y in range(y1, y2 + 1))
                add(c, "outline", p,
                    left + (right - left + 1) // 2,
                    y1 + (y2 - y1 + 1) // 2)

    remaining = {
        (x, y, c)
        for c in colors
        for y in range(63)
        for x in range(64)
        if int(obs[y, x]) == c
    }
    chosen = []
    while candidates and len(chosen) < 12:
        best = None
        best_score = 0
        for cand in candidates:
            new = sum((x, y, cand["color"]) in remaining
                      for x, y in cand["own"])
            score = new * 10 + cand["quality"] * 0.02
            if new >= 5 and score > best_score:
                best, best_score = cand, score
        if best is None:
            break
        candidates.remove(best)
        cx, cy = best["cx"], best["cy"]
        offsets = sorted((x - cx, y - cy) for x, y in best["pixels"])
        if best["kind"] == "outline":
            offsets = [p for p in offsets if p != (0, 0)]
        chosen.append({
            "x": cx,
            "y": cy,
            "color": best["color"],
            "kind": best["kind"],
            "mask": offsets,
        })
        for x, y in best["own"]:
            remaining.discard((x, y, best["color"]))
    return chosen


def detect_chargers(obs):
    found = []
    for n in range(4, 9):
        for y in range(64 - n):
            for x in range(65 - n):
                if int(obs[y, x]) != 2 or int(obs[y + n - 1, x + n - 1]) != 2:
                    continue
                border = []
                for k in range(n):
                    border.extend((
                        int(obs[y, x + k]),
                        int(obs[y + n - 1, x + k]),
                        int(obs[y + k, x]),
                        int(obs[y + k, x + n - 1]),
                    ))
                if sum(v == 2 for v in border) < len(border) - 2:
                    continue
                freq = {}
                for yy in range(y + 1, y + n - 1):
                    for xx in range(x + 1, x + n - 1):
                        v = int(obs[yy, xx])
                        if v not in (0, 1, 2, 4, 5, 15):
                            freq[v] = freq.get(v, 0) + 1
                if not freq:
                    continue
                color = max(freq, key=freq.get)
                if freq[color] < max(3, (n - 2) ** 2 * 2 // 3):
                    continue
                if any(abs(x - a[0]) <= 1 and abs(y - a[1]) <= 1
                       for a in found):
                    continue
                found.append((x, y, n, color))
    return found


def detect_obstacle(obs):
    points = [(x, y) for y in range(63) for x in range(64)
              if int(obs[y, x]) == 1]
    if not points:
        return set()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if max(xs) - min(xs) <= 7 and max(ys) - min(ys) <= 7:
        x, y = min(xs), min(ys)
        rows = (
            "11111111", "11100111", "11000011", "10000001",
            "10000001", "11000011", "11100111", "11111111",
        )
        return {(x + dx, y + dy) for dy in range(8) for dx in range(8)
                if rows[dy][dx] == "1"}
    return set(points)


def selected_object(obs, objects):
    possible = [
        i for i, o in enumerate(objects)
        if 0 <= o["x"] < 64 and 0 <= o["y"] < 63
        and int(obs[o["y"], o["x"]]) == 0
    ]
    return possible[0] if len(possible) == 1 else None


def covers(obj, x, y, selected=False):
    d = (x - obj["x"], y - obj["y"])
    return d in set(map(tuple, obj["mask"])) or (selected and d == (0, 0))


def read_targets(obs, objects, selected, old):
    targets = [dict(t) for t in old]
    changed = False
    candidates = []
    for y in range(1, 62):
        for x in range(1, 63):
            v = int(obs[y, x])
            if v in (1, 2, 4, 5, 15):
                continue
            border = sum(
                int(obs[y + dy, x + dx]) == 4
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dx or dy
            )
            if border >= 4:
                candidates.append((border, x, y))
    candidates.sort(reverse=True)
    for _, x, y in candidates:
        if not any(abs(t["x"] - x) <= 2 and abs(t["y"] - y) <= 2
                   for t in targets):
            targets.append({"x": x, "y": y, "color": None})
            changed = True
    for t in targets:
        if t["color"] is not None:
            continue
        x, y = t["x"], t["y"]
        obscured = any(covers(o, x, y, i == selected)
                       for i, o in enumerate(objects))
        v = int(obs[y, x])
        if not obscured and v not in (0, 1, 2, 4, 5, 15):
            t["color"] = v
            changed = True
    return targets, changed


def position_info(obj, chargers, obstacle):
    offsets = [tuple(p) for p in obj["mask"]]
    sx, sy = obj["x"], obj["y"]
    xs = [x for x in range(64) if (x - sx) % 3 == 0]
    ys = [y for y in range(64) if (y - sy) % 3 == 0]
    table = {}
    for y in ys:
        for x in xs:
            blocked = any((x + dx, y + dy) in obstacle
                          for dx, dy in offsets)
            paint = None
            for a, b, n, color in chargers:
                if any(a <= x + dx < a + n and b <= y + dy < b + n
                       for dx, dy in offsets):
                    paint = color
                    break
            table[(x, y)] = (blocked, paint)
    return table


def simple_bfs(obj, chargers, obstacle):
    table = position_info(obj, chargers, obstacle)
    start = (obj["x"], obj["y"], obj["color"])
    dist = {start: 0}
    pred = {}
    queue = deque([start])
    while queue:
        x, y, color = queue.popleft()
        for action, dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            info = table.get((nx, ny))
            if info is None or info[0]:
                continue
            nc = info[1] if info[1] is not None else color
            state = (nx, ny, nc)
            if state not in dist:
                dist[state] = dist[(x, y, color)] + 1
                pred[state] = ((x, y, color), action)
                queue.append(state)
    return dist, pred


def reconstruct_path(pred, state):
    result = []
    while state in pred:
        state, action = pred[state]
        result.append(action)
    result.reverse()
    return result


def explore_job(objects, selected, targets, chargers, obstacle):
    best = None
    for t in targets:
        if t["color"] is not None:
            continue
        for i, obj in enumerate(objects):
            if not covers(obj, t["x"], t["y"], i == selected):
                continue
            dist, pred = simple_bfs(obj, chargers, obstacle)
            mask = set(map(tuple, obj["mask"]))
            options = []
            for state, d in dist.items():
                x, y, _ = state
                if ((t["x"] - x, t["y"] - y) not in mask
                        and (x != t["x"] or y != t["y"])):
                    options.append((d, state))
            if options:
                d, state = min(options)
                score = d + (0 if i == selected else 2)
                if best is None or score < best[0]:
                    best = (score, i, reconstruct_path(pred, state))
    if best:
        return {"i": best[1], "path": best[2], "mode": "explore"}
    return None


def simple_solve(objects, selected, targets, chargers, obstacle, allow_bad=False):
    n = len(targets)
    if not objects or not n or n > 18:
        return None
    all_bits = (1 << n) - 1
    alternatives = []
    for obj in objects:
        dist, pred = simple_bfs(obj, chargers, obstacle)
        mask = set(map(tuple, obj["mask"]))
        best = {}
        for state, d in dist.items():
            x, y, color = state
            cover = bad = 0
            safe = True
            for k, t in enumerate(targets):
                dx, dy = t["x"] - x, t["y"] - y
                if dx == 0 and dy == 0:
                    safe = False
                if (dx, dy) in mask:
                    if color == t["color"]:
                        cover |= 1 << k
                    else:
                        bad |= 1 << k
            if bad and not allow_bad:
                continue
            key = (cover, bad if allow_bad else 0, safe)
            if key not in best or d < best[key][0]:
                best[key] = (d, state)
        alternatives.append([
            (cov, bad, safe, d, state, pred)
            for (cov, bad, safe), (d, state) in best.items()
        ])

    dp = {(0, False): (0, [])}
    for i, choices in enumerate(alternatives):
        nxt = {}
        for (covered, has_safe), (cost, assignment) in dp.items():
            for cov, bad, safe, d, state, pred in choices:
                key = ((covered & ~bad) | cov, has_safe or safe)
                new_cost = cost + d
                if key not in nxt or new_cost < nxt[key][0]:
                    nxt[key] = (
                        new_cost,
                        assignment + [(i, state, pred, safe)]
                    )
        dp = nxt
        if not dp:
            return None
    answer = dp.get((all_bits, True))
    if answer is None:
        return None
    jobs = []
    for i, state, pred, safe in answer[1]:
        path = reconstruct_path(pred, state)
        if path:
            jobs.append({"i": i, "path": path, "mode": "solve"})
    jobs.sort(key=lambda job: job["i"] != selected)
    return {"jobs": jobs}


# Dynamic state:
# left, top, width, height, vertical-line offset,
# horizontal-line offset, color.


def dynamic_object(obj):
    mask = set(map(tuple, obj["mask"]))
    if not mask:
        return None
    minx = min(x for x, y in mask)
    maxx = max(x for x, y in mask)
    miny = min(y for x, y in mask)
    maxy = max(y for x, y in mask)
    w, h = maxx - minx + 1, maxy - miny + 1
    left, top = obj["x"] + minx, obj["y"] + miny

    if obj["kind"] in ("cross", "line"):
        columns = [
            x for x in range(minx, maxx + 1)
            if all((x, y) in mask for y in range(miny, maxy + 1))
        ]
        rows = [
            y for y in range(miny, maxy + 1)
            if all((x, y) in mask for x in range(minx, maxx + 1))
        ]
        if not columns or not rows:
            return None
        a, b = columns[0] - minx, rows[0] - miny
        return {
            "kind": "cross",
            "state": (left, top, w, h, a, b, obj["color"])
        }

    if obj["kind"] == "outline":
        rect = {(x, miny) for x in range(minx, maxx + 1)}
        rect.update((x, maxy) for x in range(minx, maxx + 1))
        rect.update((minx, y) for y in range(miny, maxy + 1))
        rect.update((maxx, y) for y in range(miny, maxy + 1))
        if len(mask & rect) >= len(rect) * 9 // 10:
            return {
                "kind": "rect",
                "state": (left, top, w, h, 0, 0, obj["color"])
            }
    return None


def object_order(obs, objects):
    n = len(objects)
    if n > 6:
        return list(range(n))
    pixels = [
        {(o["x"] + dx, o["y"] + dy) for dx, dy in o["mask"]}
        for o in objects
    ]
    evidence = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if objects[i]["color"] == objects[j]["color"]:
                continue
            for x, y in pixels[i] & pixels[j]:
                if 0 <= x < 64 and 0 <= y < 63:
                    v = int(obs[y, x])
                    if v == objects[i]["color"]:
                        evidence[j][i] += 1
                    elif v == objects[j]["color"]:
                        evidence[i][j] += 1
    best, score = None, -1
    for order in permutations(range(n)):
        s = sum(evidence[order[i]][order[j]]
                for i in range(n) for j in range(i + 1, n))
        if s > score:
            best, score = list(order), s
    return best


def shape_touches_box(s, kind, bx, by, n):
    x, y, w, h, a, b, color = s
    right, bottom = x + w - 1, y + h - 1
    bx2, by2 = bx + n - 1, by + n - 1
    if right < bx or x > bx2 or bottom < by or y > by2:
        return False
    if kind == "cross":
        return (
            (bx <= x + a <= bx2 and y <= by2 and bottom >= by)
            or (by <= y + b <= by2 and x <= bx2 and right >= bx)
        )
    return (
        ((bx <= x <= bx2 or bx <= right <= bx2)
         and y <= by2 and bottom >= by)
        or ((by <= y <= by2 or by <= bottom <= by2)
            and x <= bx2 and right >= bx)
    )


def shape_hits_obstacle(s, kind, obstacle):
    x, y, w, h, a, b, color = s
    cx, cy = x + w // 2, y + h // 2
    if (cx, cy) in obstacle:
        return True
    right, bottom = x + w - 1, y + h - 1
    if kind == "cross":
        return any(
            (px == x + a and y <= py <= bottom)
            or (py == y + b and x <= px <= right)
            for px, py in obstacle
        )
    return any(
        ((px == x or px == right) and y <= py <= bottom)
        or ((py == y or py == bottom) and x <= px <= right)
        for px, py in obstacle
    )


def dynamic_step(s, kind, dx, dy, obstacle, chargers, bbox):
    x, y, w, h, a, b, color = s
    nx, ny = x + dx, y + dy
    if not (0 <= nx + w // 2 < 64 and 0 <= ny + h // 2 < 64):
        return s

    tentative = (nx, ny, w, h, a, b, color)
    if shape_hits_obstacle(tentative, kind, obstacle):
        if kind == "rect":
            if dx:
                if w > 6:
                    ny += h // 2 - (h + 3) // 2
                    w, h = w - 3, h + 3
                    ny = round(ny / 3) * 3
                    if dx < 0:
                        nx -= dx
                else:
                    nx, ny = x, y
            else:
                if h > 6:
                    nx += w // 2 - (w + 3) // 2
                    h, w = h - 3, w + 3
                    nx = round(nx / 3) * 3
                    if dy < 0:
                        ny -= dy
                else:
                    nx, ny = x, y
            # Rectangle-obstacle collisions return before charger checks.
            return (nx, ny, w, h, a, b, color)
        else:
            ox1, oy1, ox2, oy2 = bbox
            nb = ox1 <= nx + a < ox2
            vk = oy1 <= ny + b < oy2
            if dx:
                if nb and vk:
                    nx, ny = x, y
                elif nb:
                    can = a > 0 if dx > 0 else a < w - 2
                    if can:
                        a += -3 if dx > 0 else 3
                    else:
                        nx, ny = x, y
                elif vk:
                    nx, ny = x, y
                    can = a < w - 2 if dx > 0 else a > 0
                    if can:
                        a += 3 if dx > 0 else -3
            else:
                if nb and vk:
                    nx, ny = x, y
                elif vk:
                    can = b > 0 if dy > 0 else b < h - 2
                    if can:
                        b += -3 if dy > 0 else 3
                    else:
                        nx, ny = x, y
                elif nb:
                    nx, ny = x, y
                    can = b < h - 2 if dy > 0 else b > 0
                    if can:
                        b += 3 if dy > 0 else -3

    state = (nx, ny, w, h, a, b, color)
    for bx, by, size, new_color in chargers:
        if new_color != color and shape_touches_box(
                state, kind, bx, by, size):
            color = new_color
            break
    return (nx, ny, w, h, a, b, color)


def shape_at(s, kind, px, py):
    x, y, w, h, a, b, color = s
    if not (x <= px < x + w and y <= py < y + h):
        return False
    if kind == "cross":
        return px == x + a or py == y + b
    return px == x or px == x + w - 1 or py == y or py == y + h - 1


def dynamic_goals(dobj, group, all_targets):
    x0, y0, w0, h0, a0, b0, color0 = dobj["state"]
    kind = dobj["kind"]
    color = group[0]["color"]
    points = [(t["x"], t["y"]) for t in group]
    all_points = {(t["x"], t["y"]) for t in all_targets}
    wrong = [(t["x"], t["y"]) for t in all_targets
             if t["color"] != color]
    goals = set()

    if kind == "cross":
        for ix in {p[0] for p in points}:
            for iy in {p[1] for p in points}:
                if not all(tx == ix or ty == iy for tx, ty in points):
                    continue
                for x in range(-w0 // 2, 64 - w0 // 2):
                    if (x - x0) % 3:
                        continue
                    a = ix - x
                    if not (0 <= a < w0) or (a - a0) % 3:
                        continue
                    if not all(x <= tx < x + w0 for tx, ty in points):
                        continue
                    for y in range(-h0 // 2, 64 - h0 // 2):
                        if (y - y0) % 3:
                            continue
                        b = iy - y
                        if not (0 <= b < h0) or (b - b0) % 3:
                            continue
                        if not all(y <= ty < y + h0 for tx, ty in points):
                            continue
                        if (x + w0 // 2, y + h0 // 2) in all_points:
                            continue
                        state = (x, y, w0, h0, a, b, color)
                        if any(shape_at(state, kind, tx, ty)
                               for tx, ty in wrong):
                            continue
                        goals.add(state)
    else:
        for k in range(-10, 11):
            w, h = w0 + 3 * k, h0 - 3 * k
            if not (4 <= w <= 64 and 4 <= h <= 64):
                continue
            for x in range(-w // 2, 64 - w // 2):
                if x % 3 not in (0, x0 % 3):
                    continue
                for y in range(-h // 2, 64 - h // 2):
                    if y % 3 not in (0, y0 % 3):
                        continue
                    if (x + w // 2, y + h // 2) in all_points:
                        continue
                    state = (x, y, w, h, 0, 0, color)
                    if not all(shape_at(state, kind, tx, ty)
                               for tx, ty in points):
                        continue
                    if any(shape_at(state, kind, tx, ty)
                           for tx, ty in wrong):
                        continue
                    goals.add(state)
    return goals


def dynamic_assignment(objects, targets, obs):
    if not targets or any(t["color"] is None for t in targets):
        return None
    dyn = [dynamic_object(o) for o in objects]
    if any(d is None for d in dyn):
        return None
    colors = sorted({t["color"] for t in targets})
    if len(colors) > len(objects) or len(objects) > 6:
        return None
    groups = [[t for t in targets if t["color"] == c] for c in colors]
    goals = {}
    for i, d in enumerate(dyn):
        for j, group in enumerate(groups):
            goals[(i, j)] = dynamic_goals(d, group, targets)

    best = None
    for assignment in permutations(range(len(objects)), len(groups)):
        score = 0
        possible = True
        for j, i in enumerate(assignment):
            options = goals[(i, j)]
            if not options:
                possible = False
                break
            s = dyn[i]["state"]
            score += min(
                max(abs(s[0] - g[0]), abs(s[1] - g[1]),
                    abs(s[2] - g[2]), abs(s[4] - g[4]),
                    abs(s[5] - g[5]))
                for g in options
            )
        if possible and (best is None or score < best[0]):
            best = (score, assignment)
    if best is None:
        return None
    jobs = []
    for j, i in enumerate(best[1]):
        jobs.append({
            "i": i,
            "goals": list(goals[(i, j)]),
            "color": colors[j],
        })
    return {
        "objects": dyn,
        "jobs": [dict(j) for j in jobs],
        "alljobs": [dict(j) for j in jobs],
        "order": object_order(obs, objects),
        "repairs": 0,
    }


def dynamic_path(dobj, goals, obstacle, chargers):
    start = tuple(dobj["state"])
    kind = dobj["kind"]
    goalset = set(map(tuple, goals))
    if start in goalset:
        return []
    if not goalset:
        return None
    xs = [p[0] for p in obstacle]
    ys = [p[1] for p in obstacle]
    bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

    dimensions = (0, 1, 2, 4, 5)
    tables = {}
    for dim in dimensions:
        values = sorted({g[dim] for g in goalset})
        tables[dim] = [
            min(abs(v - z) for z in values)
            for v in range(-128, 129)
        ]

    def heuristic(s):
        d = 0
        for dim in dimensions:
            v = s[dim]
            z = tables[dim][v + 128] if -128 <= v <= 128 else 0
            d = max(d, z)
        return (d + 2) // 3

    dist = {start: 0}
    pred = {}
    heap = [(heuristic(start), 0, start)]
    expanded = 0
    while heap and expanded < 100000:
        _, cost, state = heapq.heappop(heap)
        if dist.get(state) != cost:
            continue
        if state in goalset:
            return reconstruct_path(pred, state)
        expanded += 1
        for action, dx, dy in MOVES:
            nxt = dynamic_step(state, kind, dx, dy,
                               obstacle, chargers, bbox)
            if nxt == state:
                continue
            newcost = cost + 1
            if newcost < dist.get(nxt, 10 ** 9):
                dist[nxt] = newcost
                pred[nxt] = (state, action)
                heapq.heappush(
                    heap, (newcost + heuristic(nxt), newcost, nxt)
                )
    return None


def update_object_from_dynamic(obj, dobj):
    x, y, w, h, a, b, color = dobj["state"]
    cx, cy = x + w // 2, y + h // 2
    if dobj["kind"] == "cross":
        pixels = {(x + a, y + j) for j in range(h)}
        pixels.update((x + i, y + b) for i in range(w))
    else:
        pixels = {(x + i, y) for i in range(w)}
        pixels.update((x + i, y + h - 1) for i in range(w))
        pixels.update((x, y + j) for j in range(h))
        pixels.update((x + w - 1, y + j) for j in range(h))
    obj["x"], obj["y"], obj["color"] = cx, cy, color
    obj["mask"] = [(px - cx, py - cy) for px, py in pixels]


def observed_object_color(obs, objects, i):
    obj = objects[i]
    occupied = set()
    for j, other in enumerate(objects):
        if j == i:
            continue
        occupied.update((other["x"] + dx, other["y"] + dy)
                        for dx, dy in other["mask"])
    freq = {}
    for dx, dy in obj["mask"]:
        x, y = obj["x"] + dx, obj["y"] + dy
        if not (0 <= x < 64 and 0 <= y < 63):
            continue
        if (x, y) in occupied:
            continue
        v = int(obs[y, x])
        if v not in (0, 1, 2, 4, 5, 15):
            freq[v] = freq.get(v, 0) + 1
    if not freq:
        return None
    color = max(freq, key=freq.get)
    if freq[color] >= 4 and freq[color] * 3 >= sum(freq.values()) * 2:
        return color
    return None


def return_action(memory, action, available, obj_index=None, dynamic=False):
    if action == "ACTION5":
        memory["switches"] = memory.get("switches", 0) + 1
        limit = 2 * max(1, len(memory.get("objects", []))) + 3
        if (memory["switches"] > limit
                and memory.get("resets", 0) < 1
                and "RESET" in available):
            resets = memory.get("resets", 0) + 1
            memory.clear()
            memory["resets"] = resets
            return ("RESET", None, None)
    else:
        memory["switches"] = 0
    memory["pending"] = {
        "action": action, "i": obj_index, "dynamic": dynamic
    }
    return (action, None, None)


def act(obs, available, memory):
    obs = np.asarray(obs)

    if "objects" not in memory:
        objects = detect_objects(obs)
        memory["objects"] = objects
        memory["chargers"] = detect_chargers(obs)
        memory["obstacle"] = list(detect_obstacle(obs))
        memory["targets"] = []
        memory["selected"] = selected_object(obs, objects)
        memory["failures"] = 0

    objects = memory["objects"]
    if not objects:
        return return_action(memory, "ACTION5", available)

    obstacle = set(map(tuple, memory.get("obstacle", [])))
    chargers = memory.get("chargers", [])
    pending = memory.pop("pending", None)
    dyn = memory.get("dynamic")

    if pending and pending["action"] in DELTA:
        i = pending.get("i")
        if i is not None and 0 <= i < len(objects):
            dx, dy = DELTA[pending["action"]]
            if pending.get("dynamic") and dyn:
                xs = [p[0] for p in obstacle]
                ys = [p[1] for p in obstacle]
                bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
                dobj = dyn["objects"][i]
                dobj["state"] = dynamic_step(
                    tuple(dobj["state"]), dobj["kind"], dx, dy,
                    obstacle, chargers, bbox
                )
                update_object_from_dynamic(objects[i], dobj)
                actual = observed_object_color(obs, objects, i)
                if actual is not None and actual != dobj["state"][-1]:
                    state = list(dobj["state"])
                    state[-1] = actual
                    dobj["state"] = tuple(state)
                    update_object_from_dynamic(objects[i], dobj)
                    active = dyn.get("active")
                    if active and active["i"] == i:
                        active["replan"] = True
            else:
                obj = objects[i]
                nx, ny = obj["x"] + dx, obj["y"] + dy
                if 0 <= nx < 64 and 0 <= ny < 64:
                    obj["x"], obj["y"] = nx, ny
                    for a, b, size, color in chargers:
                        if any(a <= nx + px < a + size
                               and b <= ny + py < b + size
                               for px, py in obj["mask"]):
                            obj["color"] = color
                            break

    seen_selected = selected_object(obs, objects)
    if seen_selected is not None:
        memory["selected"] = seen_selected
    elif pending and pending["action"] == "ACTION5":
        if dyn and memory.get("selected") in dyn["order"]:
            order = dyn["order"]
            p = order.index(memory["selected"])
            memory["selected"] = order[(p + 1) % len(order)]
        else:
            memory["selected"] = None
    selected = memory.get("selected")

    targets, changed = read_targets(
        obs, objects, selected, memory.get("targets", [])
    )
    memory["targets"] = targets
    if changed:
        memory.pop("plan", None)
        job = memory.get("job")
        if job and job.get("mode") == "solve":
            memory.pop("job", None)

    if obstacle and not memory.get("dynamic_failed"):
        if dyn is None:
            dyn = dynamic_assignment(objects, targets, obs)
            if dyn is not None:
                memory["dynamic"] = dyn
        if dyn is not None:
            active = dyn.get("active")
            if active:
                if active.get("replan"):
                    newpath = dynamic_path(
                        dyn["objects"][active["i"]], active["goals"],
                        obstacle, chargers
                    )
                    if newpath is None:
                        memory["dynamic_failed"] = True
                        memory.pop("dynamic", None)
                    else:
                        active["path"] = newpath
                        active["replan"] = False
                if not memory.get("dynamic_failed"):
                    if active["path"]:
                        i = active["i"]
                        if selected != i:
                            return return_action(memory, "ACTION5", available)
                        action = active["path"].pop(0)
                        return return_action(memory, action, available, i, True)
                    dyn.pop("active", None)

            if not memory.get("dynamic_failed"):
                if dyn["jobs"]:
                    job = dyn["jobs"].pop(0)
                    i = job["i"]
                    path = dynamic_path(
                        dyn["objects"][i], job["goals"], obstacle, chargers
                    )
                    if path is None:
                        memory["dynamic_failed"] = True
                        memory.pop("dynamic", None)
                    elif path:
                        dyn["active"] = {
                            "i": i, "path": path, "goals": job["goals"]
                        }
                        if selected != i:
                            return return_action(memory, "ACTION5", available)
                        action = path.pop(0)
                        return return_action(memory, action, available, i, True)
                else:
                    # Repair any assigned piece that did not reach its goal.
                    if dyn["repairs"] < 4:
                        bad = None
                        for job in dyn["alljobs"]:
                            i = job["i"]
                            if tuple(dyn["objects"][i]["state"]) not in set(
                                    map(tuple, job["goals"])):
                                bad = job
                                break
                        if bad is not None:
                            dyn["repairs"] += 1
                            dyn["jobs"].append(dict(bad))
                    return return_action(memory, "ACTION5", available)

    job = memory.get("job")
    if job:
        if not job["path"]:
            memory.pop("job", None)
        else:
            i = job["i"]
            if selected != i:
                return return_action(memory, "ACTION5", available)
            action = job["path"].pop(0)
            return return_action(memory, action, available, i)

    if any(t["color"] is None for t in targets):
        job = explore_job(objects, selected, targets, chargers, obstacle)
        if job and job["path"]:
            memory["job"] = job
            if selected != job["i"]:
                return return_action(memory, "ACTION5", available)
            action = job["path"].pop(0)
            return return_action(memory, action, available, job["i"])

    plan = memory.get("plan")
    if plan is None:
        known = [t for t in targets if t["color"] is not None]
        plan = simple_solve(objects, selected, known, chargers, obstacle, False)
        if plan is None:
            plan = simple_solve(objects, selected, known, chargers, obstacle, True)
        if plan is not None:
            memory["plan"] = plan

    if plan is not None:
        if plan["jobs"]:
            job = plan["jobs"].pop(0)
            memory["job"] = job
            if selected != job["i"]:
                return return_action(memory, "ACTION5", available)
            if job["path"]:
                action = job["path"].pop(0)
                return return_action(memory, action, available, job["i"])
        else:
            memory["failures"] = memory.get("failures", 0) + 1
            if memory["failures"] > len(objects) + 1:
                memory.pop("plan", None)
                memory["failures"] = 0
            return return_action(memory, "ACTION5", available)

    return return_action(memory, "ACTION5", available)

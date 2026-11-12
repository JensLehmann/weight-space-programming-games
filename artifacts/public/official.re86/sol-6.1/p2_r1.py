import numpy as np
import time
from collections import deque, Counter

DIRECTIONS = ((0, -3), (0, 3), (-3, 0), (3, 0))


def find_targets(im):
    candidates = []
    pairs = (((-1, -1), (1, 1)), ((1, -1), (-1, 1)),
             ((0, -1), (0, 1)), ((-1, 0), (1, 0)))
    for y in range(1, 62):
        for x in range(1, 63):
            count = sum(
                int(im[y + dy, x + dx]) == 4
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1) if dx or dy)
            if count < 4:
                continue
            symmetry = sum(
                int(im[y + a[1], x + a[0]]) == 4 and
                int(im[y + b[1], x + b[0]]) == 4
                for a, b in pairs)
            if symmetry >= 2:
                candidates.append((count + 2 * symmetry, x, y))
    candidates.sort(reverse=True)
    targets = []
    for score, x, y in candidates:
        if not any(abs(x - a) <= 1 and abs(y - b) <= 1
                   for a, b, _ in targets):
            targets.append((x, y, None))
    targets.sort(key=lambda t: (t[1], t[0]))
    return targets


def find_gates(im):
    candidates = []
    for n in (5, 6, 7, 8):
        for y in range(64 - n):
            for x in range(65 - n):
                corners = sum(
                    int(im[yy, xx]) == 2
                    for yy, xx in ((y, x), (y, x + n - 1),
                                   (y + n - 1, x), (y + n - 1, x + n - 1)))
                if corners < 2:
                    continue
                edge = (
                    np.count_nonzero(im[y, x:x + n] == 2)
                    + np.count_nonzero(im[y + n - 1, x:x + n] == 2)
                    + np.count_nonzero(im[y + 1:y + n - 1, x] == 2)
                    + np.count_nonzero(im[y + 1:y + n - 1, x + n - 1] == 2)
                )
                if edge < 0.60 * (4 * n - 4):
                    continue
                counts = Counter(
                    int(v) for v in
                    im[y + 1:y + n - 1, x + 1:x + n - 1].ravel()
                    if int(v) not in (0, 1, 2, 3, 4, 5))
                if counts:
                    c, k = counts.most_common(1)[0]
                    candidates.append((int(edge) + corners + k, x, y, n, c))
    candidates.sort(reverse=True)
    result = []
    for score, x, y, n, c in candidates:
        if any(x < a + w and a < x + n and y < b + h and b < y + n
               for a, b, w, h, color in result):
            continue
        result.append((x, y, n, n, c))
    result.sort(key=lambda g: (g[1], g[0]))
    return result


def find_obstacle(im):
    points = {(int(x), int(y)) for y, x in zip(*np.where(im[:63] == 1))}
    remaining = set(points)
    while remaining:
        p = remaining.pop()
        component = {p}
        stack = [p]
        while stack:
            x, y = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        x0 = min(x for x, y in component)
        x1 = max(x for x, y in component)
        y0 = min(y for x, y in component)
        y1 = max(y for x, y in component)
        if x1 - x0 == 7 and y1 - y0 == 7:
            pattern = ("11111111", "11100111", "11000011", "10000001",
                       "10000001", "11000011", "11100111", "11111111")
            for yy, row in enumerate(pattern):
                for xx, v in enumerate(row):
                    if v == "1":
                        points.add((x0 + xx, y0 + yy))
    rows = {}
    for x, y in points:
        rows[y] = rows.get(y, 0) | (1 << x)
    if points:
        box = (min(x for x, y in points), min(y for x, y in points),
               max(x for x, y in points) + 1, max(y for x, y in points) + 1)
    else:
        box = (0, 0, 0, 0)
    return rows, box


def template_rows(kind, w, h, col=0, row=0):
    rows = [0] * h
    if kind == "cross":
        rows = [1 << col] * h
        rows[row] = (1 << w) - 1
    elif kind == "horizontal":
        rows[0] = (1 << w) - 1
    elif kind == "vertical":
        rows = [1] * h
    elif kind == "rectangle":
        rows[0] = rows[-1] = (1 << w) - 1
        for y in range(1, h - 1):
            rows[y] = 1 | (1 << (w - 1))
    elif kind == "x":
        for y in range(h):
            rows[y] = (1 << y) | (1 << (w - 1 - y))
    elif kind == "diamond":
        r = w // 2
        for y in range(h):
            d = r - abs(y - r)
            rows[y] = (1 << (r - d)) | (1 << (r + d))
    return tuple(rows)


def axes(w, h, rows):
    full = (1 << w) - 1
    col, row = w - 1, h - 1
    for x in range(w):
        if all(r & (1 << x) for r in rows):
            col = x
            break
    for y in range(h):
        if rows[y] == full:
            row = y
            break
    return col, row


def normalize(w, h, rows, closed, force):
    result = list(rows)
    cx, cy = w // 2, h // 2
    bit = 1 << cx
    if closed:
        result[cy] &= ~bit
    elif force:
        result[cy] |= bit
    else:
        active = list(result)
        active[cy] |= bit
        col, row = axes(w, h, active)
        if cx == col or cy == row:
            result[cy] |= bit
        else:
            result[cy] &= ~bit
    return tuple(result)


def extract_lines(im, unknown):
    eligible = ((im != 5) & (im != 4) & (im != 2) & (im != 1)) | unknown
    lines = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dy == 0:
            starts = [(0, y) for y in range(63)]
        elif dx == 0:
            starts = [(x, 0) for x in range(64)]
        elif dy == 1:
            starts = [(x, 0) for x in range(64)]
            starts += [(0, y) for y in range(1, 64)]
        else:
            starts = [(x, 63) for x in range(64)]
            starts += [(0, y) for y in range(63)]

        def finish(run):
            if len(run) < 7:
                return
            counts = Counter(
                int(im[y, x]) for x, y in run
                if not unknown[y, x]
                and int(im[y, x]) not in (0, 1, 2, 3, 4, 5))
            for c, k in counts.items():
                if k < 4 or k < 0.22 * len(run):
                    continue
                indices = [i for i, (x, y) in enumerate(run)
                           if not unknown[y, x] and int(im[y, x]) == c]
                lo, hi = indices[0], indices[-1]
                while lo and unknown[run[lo - 1][1], run[lo - 1][0]]:
                    lo -= 1
                while hi + 1 < len(run) and unknown[run[hi + 1][1], run[hi + 1][0]]:
                    hi += 1
                if hi - lo + 1 >= 7:
                    a, b = run[lo], run[hi]
                    lines.append((dx, dy, a[0], a[1], b[0], b[1], c))
                a, b = run[0], run[-1]
                lines.append((dx, dy, a[0], a[1], b[0], b[1], c))

        for sx, sy in starts:
            run = []
            x, y = sx, sy
            while 0 <= x < 64 and 0 <= y < 64:
                if eligible[y, x]:
                    run.append((x, y))
                else:
                    finish(run)
                    run = []
                x += dx
                y += dy
            finish(run)
    return list(dict.fromkeys(lines))


def reconstruct_objects(im, gates):
    unknown = np.zeros((64, 64), dtype=bool)
    unknown[63] = True
    for x, y, w, h, c in gates:
        unknown[y:y + h, x:x + w] = True
    lines = extract_lines(im, unknown)
    candidates = {}

    def add(kind, x, y, w, h, color, col=0, row=0):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if not (1 <= w <= 64 and 1 <= h <= 64):
            return
        if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
            return
        if not (0 <= col < w and 0 <= row < h):
            return
        if kind in ("x", "diamond") and (w != h or not w % 2):
            return
        rows = template_rows(kind, w, h, col, row)
        key = (x, y, w, h, color, rows)
        if key in candidates:
            return
        cx, cy = w // 2, h // 2
        matched = set()
        other = 0
        missing = 0
        expected = 0
        for yy, bits0 in enumerate(rows):
            bits = bits0
            while bits:
                bit = bits & -bits
                xx = bit.bit_length() - 1
                bits -= bit
                gx, gy = x + xx, y + yy
                if xx == cx and yy == cy:
                    continue
                if not (0 <= gx < 64 and 0 <= gy < 63) or unknown[gy, gx]:
                    continue
                expected += 1
                v = int(im[gy, gx])
                if v == color or v == 0:
                    matched.add((gx, gy))
                elif v in (1, 2, 3, 4, 5):
                    missing += 1
                else:
                    other += 1
        if missing or len(matched) < 6:
            return
        if len(matched) < 0.48 * max(1, expected):
            return

        quality = 0.0
        if kind == "cross":
            quality += 0.8
            quality -= 2.5 * ((w % 2 == 0) + (h % 2 == 0))
            quality += 0.5 * (col == cx) + 0.5 * (row == cy)
        elif kind in ("rectangle", "diamond", "x"):
            quality += 0.5
        else:
            quality -= 0.5
            quality -= 2.0 * ((w > 1 and w % 2 == 0) or
                              (h > 1 and h % 2 == 0))
        if int(im[y + cy, x + cx]) == 0:
            quality += 8
        closed = kind in ("rectangle", "diamond")
        force = kind == "x" and bool(gates)
        candidates[key] = {
            "state": (0, x, y, color),
            "variant": (w, h, normalize(w, h, rows, closed, force)),
            "closed": closed, "force": force, "kind": kind,
            "matched": matched, "quality": quality, "other": other
        }

    horizontal = [l for l in lines if l[:2] == (1, 0)]
    vertical = [l for l in lines if l[:2] == (0, 1)]
    plus = [l for l in lines if l[:2] == (1, 1)]
    minus = [l for l in lines if l[:2] == (1, -1)]

    for _, _, x0, y, x1, _, c in horizontal:
        add("horizontal", x0, y, x1 - x0 + 1, 1, c)
    for _, _, x, y0, _, y1, c in vertical:
        add("vertical", x, y0, 1, y1 - y0 + 1, c)

    zeros = [(int(x), int(y)) for y, x in zip(*np.where(im[:63] == 0))]
    for a in horizontal:
        _, _, x0, yy, x1, _, c = a
        for b in vertical:
            _, _, xx, y0, _, y1, cc = b
            if c != cc or not (x0 <= xx <= x1 and y0 <= yy <= y1):
                continue
            w, h = x1 - x0 + 1, y1 - y0 + 1
            add("cross", x0, y0, w, h, c, xx - x0, yy - y0)

            if x0 == 0:
                nw = 2 * (x1 - xx) + 1
                add("cross", x1 - nw + 1, y0, nw, h, c,
                    xx - (x1 - nw + 1), yy - y0)
            if x1 == 63:
                nw = 2 * (xx - x0) + 1
                add("cross", x0, y0, nw, h, c, xx - x0, yy - y0)
            if y0 == 0:
                nh = 2 * (y1 - yy) + 1
                add("cross", x0, y1 - nh + 1, w, nh, c,
                    xx - x0, yy - (y1 - nh + 1))
            if y1 == 63:
                nh = 2 * (yy - y0) + 1
                add("cross", x0, y0, w, nh, c, xx - x0, yy - y0)
            if x0 == 0 or x1 == 63:
                add("cross", xx - h // 2, y0, h, h, c,
                    h // 2, yy - y0)
            if y0 == 0 or y1 == 63:
                add("cross", x0, yy - w // 2, w, w, c,
                    xx - x0, w // 2)
            for zx, zy in zeros:
                if x0 <= zx <= x1 and y0 <= zy <= y1:
                    if x0 == 0 or x1 == 63:
                        nw = 2 * max(zx - x0, x1 - zx) + 1
                        add("cross", zx - nw // 2, y0, nw, h, c,
                            xx - (zx - nw // 2), yy - y0)
                    if y0 == 0 or y1 == 63:
                        nh = 2 * max(zy - y0, y1 - zy) + 1
                        add("cross", x0, zy - nh // 2, w, nh, c,
                            xx - x0, yy - (zy - nh // 2))

    for i, a in enumerate(horizontal):
        _, _, x0, y0, x1, _, c = a
        for b in horizontal[i + 1:]:
            _, _, bx0, y1, bx1, _, cc = b
            if c != cc or abs(y1 - y0) < 3:
                continue
            if min(x1, bx1) - max(x0, bx0) < 3:
                continue
            left, right = min(x0, bx0), max(x1, bx1)
            top, bottom = min(y0, y1), max(y0, y1)
            for dl in range(4):
                for dr in range(4):
                    add("rectangle", left - dl, top,
                        right - left + 1 + dl + dr,
                        bottom - top + 1, c)

    for i, a in enumerate(vertical):
        _, _, x0, y0, _, y1, c = a
        for b in vertical[i + 1:]:
            _, _, x1, by0, _, by1, cc = b
            if c != cc or abs(x1 - x0) < 3:
                continue
            if min(y1, by1) - max(y0, by0) < 3:
                continue
            left, right = min(x0, x1), max(x0, x1)
            top, bottom = min(y0, by0), max(y1, by1)
            for dt in range(4):
                for db in range(4):
                    add("rectangle", left, top - dt,
                        right - left + 1,
                        bottom - top + 1 + dt + db, c)

    for a in plus:
        _, _, ax0, ay0, ax1, ay1, c = a
        aa = ay0 - ax0
        for b in minus:
            _, _, bx0, by0, bx1, by1, cc = b
            if c != cc:
                continue
            bb = by0 + bx0
            if (bb - aa) % 2:
                continue
            cx, cy = (bb - aa) // 2, (bb + aa) // 2
            if ax0 - 1 <= cx <= ax1 + 1 and bx0 - 1 <= cx <= bx1 + 1:
                r = max(abs(ax0 - cx), abs(ax1 - cx),
                        abs(bx0 - cx), abs(bx1 - cx))
                if 3 <= r <= 31:
                    add("x", cx - r, cy - r, 2 * r + 1, 2 * r + 1, c)

    for l in plus + minus:
        _, _, x0, y0, x1, y1, c = l
        r = x1 - x0
        if 4 <= r <= 25:
            for cx, cy in ((x0, y1), (x1, y0)):
                add("diamond", cx - r, cy - r, 2 * r + 1, 2 * r + 1, c)

    for _, _, x0, y, x1, _, c in horizontal:
        if x0 == 0 or x1 == 63:
            for zx, zy in zeros:
                if zy == y and x0 <= zx <= x1:
                    n = 2 * max(zx - x0, x1 - zx) + 1
                    add("horizontal", zx - n // 2, y, n, 1, c)
    for _, _, x, y0, _, y1, c in vertical:
        if y0 == 0 or y1 == 63:
            for zx, zy in zeros:
                if zx == x and y0 <= zy <= y1:
                    n = 2 * max(zy - y0, y1 - zy) + 1
                    add("vertical", x, zy - n // 2, 1, n, c)

    pool = list(candidates.values())
    explained = set()
    chosen = []
    while pool:
        best = max(pool, key=lambda a:
                   len(a["matched"] - explained)
                   + a["quality"] - 0.15 * a["other"])
        if len(best["matched"] - explained) < 6:
            break
        chosen.append(best)
        explained.update(best["matched"])
        pool.remove(best)

    objects = []
    for a in chosen:
        variant = a["variant"]
        objects.append({
            "state": a["state"], "closed": a["closed"],
            "force": a["force"], "kind": a["kind"],
            "variants": [variant], "variant_ids": {variant: 0}
        })
    return objects


def add_variant(obj, w, h, rows):
    key = (w, h, normalize(w, h, rows, obj["closed"], obj["force"]))
    if key not in obj["variant_ids"]:
        obj["variant_ids"][key] = len(obj["variants"])
        obj["variants"].append(key)
    return obj["variant_ids"][key]


def covers(obj, state, px, py, include_center=False):
    vid, x, y, color = state
    w, h, rows = obj["variants"][vid]
    xx, yy = px - x, py - y
    if include_center and xx == w // 2 and yy == h // 2:
        return True
    return (0 <= xx < w and 0 <= yy < h
            and bool(rows[yy] & (1 << xx)))


def gate_collision(rows, x, y, w, h, gate):
    gx, gy, gw, gh, gc = gate
    x0, x1 = max(x, gx), min(x + w, gx + gw)
    y0, y1 = max(y, gy), min(y + h, gy + gh)
    if x0 >= x1 or y0 >= y1:
        return False
    mask = ((1 << (x1 - x0)) - 1) << (x0 - x)
    return any(rows[yy - y] & mask for yy in range(y0, y1))


def geometry(game, obj, vid, x, y, cache):
    key = (vid, x, y)
    if key in cache:
        return cache[key]
    w, h, rows = obj["variants"][vid]
    cx, cy = w // 2, h // 2
    blocked = False
    bx0, by0, bx1, by1 = game["obstacle_box"]
    if x < bx1 and x + w > bx0 and y < by1 and y + h > by0:
        for gy in range(max(y, by0), min(y + h, by1)):
            bits = rows[gy - y]
            if gy - y == cy:
                bits |= 1 << cx
            global_bits = bits << x if x >= 0 else bits >> (-x)
            if global_bits & game["obstacle_rows"].get(gy, 0):
                blocked = True
                break
    gate_ids = tuple(
        i for i, gate in enumerate(game["gates"])
        if gate_collision(rows, x, y, w, h, gate))
    hit = 0
    goods = {}
    center_target = False
    for i, (tx, ty, color) in enumerate(game["targets"]):
        xx, yy = tx - x, ty - y
        if xx == cx and yy == cy:
            center_target = True
        if 0 <= xx < w and 0 <= yy < h and rows[yy] & (1 << xx):
            bit = 1 << i
            hit |= bit
            goods[color] = goods.get(color, 0) | bit
    result = (blocked, gate_ids, hit, goods, center_target)
    cache[key] = result
    return result


def move_state(game, obj, state, action, geo_cache, move_cache):
    vid, x, y, color = state
    key = (vid, x, y, action)
    if key in move_cache:
        nv, nx, ny, gate_ids = move_cache[key]
    else:
        dx, dy = DIRECTIONS[action - 1]
        w, h, rows = obj["variants"][vid]
        nx, ny, nv = x + dx, y + dy, vid
        if not (0 <= nx + w // 2 < 64 and 0 <= ny + h // 2 < 64):
            return state

        blocked = geometry(game, obj, vid, nx, ny, geo_cache)[0]
        early = False
        if blocked:
            if obj["closed"]:
                if dx:
                    if w > 6:
                        nw, nh = w - 3, h + 3
                        ny += h // 2 - nh // 2
                        ny = round(ny / 3) * 3
                        if dx < 0:
                            nx -= dx
                        nv = add_variant(obj, nw, nh,
                                         template_rows("rectangle", nw, nh))
                    else:
                        nx, ny = x, y
                else:
                    if h > 6:
                        nw, nh = w + 3, h - 3
                        nx += w // 2 - nw // 2
                        nx = round(nx / 3) * 3
                        if dy < 0:
                            ny -= dy
                        nv = add_variant(obj, nw, nh,
                                         template_rows("rectangle", nw, nh))
                    else:
                        nx, ny = x, y
                early = True
            else:
                active = list(rows)
                cx, cy = w // 2, h // 2
                active[cy] |= 1 << cx
                col, row = axes(w, h, active)
                bx0, by0, bx1, by1 = game["obstacle_box"]
                in_x = bx0 <= nx + col < bx1
                in_y = by0 <= ny + row < by1
                changed = False
                if in_x and in_y:
                    nx, ny = x, y
                elif dx:
                    back = -3 if dx > 0 else 3
                    forward = 3 if dx > 0 else -3
                    can_back = col > 0 if dx > 0 else col < w - 2
                    can_forward = col < w - 2 if dx > 0 else col > 0
                    if in_x:
                        if can_back:
                            nc = (col + back) % w
                            for yy in range(h):
                                active[yy] &= ~(1 << col)
                                active[yy] |= 1 << nc
                            active[row] |= 1 << col
                            changed = True
                        else:
                            nx, ny = x, y
                    elif in_y:
                        nx, ny = x, y
                        if can_forward:
                            nc = (col + forward) % w
                            for yy in range(h):
                                active[yy] &= ~(1 << col)
                                active[yy] |= 1 << nc
                            active[row] |= 1 << col
                            changed = True
                else:
                    back = -3 if dy > 0 else 3
                    forward = 3 if dy > 0 else -3
                    can_back = row > 0 if dy > 0 else row < h - 2
                    can_forward = row < h - 2 if dy > 0 else row > 0
                    if in_y:
                        if can_back:
                            nr = (row + back) % h
                            active[row] = 0
                            active[nr] = (1 << w) - 1
                            active[row] |= 1 << col
                            changed = True
                        else:
                            nx, ny = x, y
                    elif in_x:
                        nx, ny = x, y
                        if can_forward:
                            nr = (row + forward) % h
                            active[row] = 0
                            active[nr] = (1 << w) - 1
                            active[row] |= 1 << col
                            changed = True
                if changed:
                    active[cy] |= 1 << cx
                    nv = add_variant(obj, w, h, active)

        gate_ids = () if early else geometry(
            game, obj, nv, nx, ny, geo_cache)[1]
        move_cache[key] = (nv, nx, ny, gate_ids)

    new_color = color
    for i in gate_ids:
        gc = game["gates"][i][4]
        if gc != color:
            new_color = gc
            break
    return nv, nx, ny, new_color


def infer_draw_order(im, objects, preferred=None):
    n = len(objects)
    edges = [set() for _ in objects]
    for i in range(n):
        vi, xi, yi, ci = objects[i]["state"]
        wi, hi, ri = objects[i]["variants"][vi]
        for j in range(i + 1, n):
            vj, xj, yj, cj = objects[j]["state"]
            if ci == cj:
                continue
            wj, hj, rj = objects[j]["variants"][vj]
            ai = aj = 0
            for gy in range(max(0, yi, yj), min(63, yi + hi, yj + hj)):
                for gx in range(max(0, xi, xj), min(64, xi + wi, xj + wj)):
                    if ((ri[gy - yi] >> (gx - xi)) & 1 and
                            (rj[gy - yj] >> (gx - xj)) & 1):
                        v = int(im[gy, gx])
                        ai += v == ci
                        aj += v == cj
            if ai > aj:
                edges[j].add(i)
            elif aj > ai:
                edges[i].add(j)
    if preferred is None:
        preferred = list(range(n))
    ranks = {i: k for k, i in enumerate(preferred)}
    order = []
    remaining = set(range(n))
    while remaining:
        ready = [i for i in remaining
                 if not any(i in edges[j] for j in remaining)]
        choices = ready if ready else list(remaining)
        i = min(choices, key=lambda j: ranks.get(j, j))
        remaining.remove(i)
        order.append(i)
    return order


def identify_selected(im, game):
    zeros = {(int(x), int(y)) for y, x in zip(*np.where(im[:63] == 0))}
    candidates = []
    for i, obj in enumerate(game["objects"]):
        vid, x, y, color = obj["state"]
        w, h, rows = obj["variants"][vid]
        if (x + w // 2, y + h // 2) in zeros:
            candidates.append(i)
    old = game.get("selected", -1)
    predicted = game["next"].get(old, -1)
    if len(candidates) == 1:
        return candidates[0]
    if game.get("last_action") == 5:
        if predicted in candidates or not candidates:
            return predicted
    if old in candidates:
        return old
    if candidates:
        return candidates[0]
    return old


def observed_color(im, obj):
    vid, x, y, old = obj["state"]
    w, h, rows = obj["variants"][vid]
    counts = Counter()
    for yy, bits0 in enumerate(rows):
        gy = y + yy
        if not 0 <= gy < 63:
            continue
        bits = bits0
        while bits:
            bit = bits & -bits
            xx = bit.bit_length() - 1
            bits -= bit
            gx = x + xx
            if 0 <= gx < 64:
                c = int(im[gy, gx])
                if c not in (0, 1, 2, 3, 4, 5):
                    counts[c] += 1
    if counts:
        c, k = counts.most_common(1)[0]
        if k >= 4 and k >= 0.65 * sum(counts.values()):
            return c
    return old


def prediction_error(im, game):
    expected = {}
    selected = game["selected"]
    for i in game["draw_order"]:
        obj = game["objects"][i]
        vid, x, y, color = obj["state"]
        w, h, rows = obj["variants"][vid]
        cx, cy = w // 2, h // 2
        for yy, bits0 in enumerate(rows):
            gy = y + yy
            if not 0 <= gy < 63:
                continue
            bits = bits0
            if i == selected and yy == cy:
                bits |= 1 << cx
            while bits:
                bit = bits & -bits
                xx = bit.bit_length() - 1
                bits -= bit
                gx = x + xx
                if 0 <= gx < 64:
                    c = 0 if i == selected and xx == cx and yy == cy else color
                    expected[(gx, gy)] = c
    return sum(int(im[y, x]) != c for (x, y), c in expected.items())


def read_targets(im, game):
    targets = []
    for x, y, color in game["targets"]:
        if color is None:
            occupied = any(
                covers(obj, obj["state"], x, y, True)
                for obj in game["objects"])
            v = int(im[y, x])
            if not occupied and v not in (0, 1, 2, 3, 4, 5):
                color = v
        targets.append((x, y, color))
    game["targets"] = targets


def survey_action(game, selected):
    unknown = [(x, y) for x, y, c in game["targets"] if c is None]
    if not unknown:
        return None
    if selected < 0:
        return 5
    obj = game["objects"][selected]
    if not any(covers(obj, obj["state"], x, y, True) for x, y in unknown):
        return 5

    initial_count = sum(
        covers(obj, obj["state"], x, y, True) for x, y in unknown)
    queue = deque([(obj["state"], [])])
    seen = {obj["state"]}
    best = None
    geo_cache, move_cache = {}, {}
    while queue:
        state, path = queue.popleft()
        if path:
            count = sum(covers(obj, state, x, y, True) for x, y in unknown)
            reduction = initial_count - count
            if reduction > 0:
                vid, x, y, c = state
                penalty = 0.15 * (vid != obj["state"][0])
                penalty += 0.10 * (c != obj["state"][3])
                score = 10 * reduction - len(path) - penalty
                if best is None or score > best[0]:
                    best = (score, path[0])
        if len(path) == 3:
            continue
        for action in range(1, 5):
            ns = move_state(game, obj, state, action, geo_cache, move_cache)
            if ns not in seen:
                seen.add(ns)
                queue.append((ns, path + [action]))
    return best[1] if best else 5


def new_search(obj):
    s = obj["state"]
    return {
        "queue": deque([s]), "parents": {s: (None, 0, 0)},
        "options": {}, "geo": {}, "moves": {}, "done": False
    }


def start_planning(game):
    game["phase"] = "plan"
    game["searches"] = [new_search(o) for o in game["objects"]]
    game["search_index"] = 0
    game["depth_limit"] = 18
    game["paths"] = None
    game["finish_count"] = 0


def search_to_depth(game, obj, search, limit, deadline):
    if search["done"]:
        return True
    queue = search["queue"]
    parents = search["parents"]
    count = 0
    while queue:
        if count % 192 == 0 and time.perf_counter() >= deadline:
            return False
        count += 1
        state = queue.popleft()
        depth = parents[state][2]
        vid, x, y, color = state
        blocked, gates, hit, goods, center_target = geometry(
            game, obj, vid, x, y, search["geo"])
        good = goods.get(color, 0)
        bad = hit & ~good
        if not center_target:
            key = (good, bad)
            if key not in search["options"]:
                search["options"][key] = (depth, state)
        if depth >= limit:
            queue.appendleft(state)
            return True
        for action in range(1, 5):
            ns = move_state(
                game, obj, state, action, search["geo"], search["moves"])
            if ns not in parents:
                parents[ns] = (state, action, depth + 1)
                queue.append(ns)
    search["done"] = True
    return True


def strict_options(search):
    options = [(good, bad, cost, state)
               for (good, bad), (cost, state) in search["options"].items()
               if bad == 0]
    options.sort(key=lambda o: (o[2], -o[0].bit_count()))
    kept = []
    for o in options:
        if any((o[0] | p[0]) == p[0] and p[2] <= o[2] for p in kept):
            continue
        kept = [p for p in kept
                if not ((o[0] | p[0]) == o[0] and o[2] <= p[2])]
        kept.append(o)
    return kept


def trim_dp(dp, size=50000):
    if len(dp) <= size:
        return dp
    keys = sorted(dp, key=lambda k: dp[k][0] - 8 * k.bit_count())[:size]
    return {k: dp[k] for k in keys}


def combine_placements(game, allow_overlay=False):
    searches = game["searches"]
    n = len(searches)
    full = (1 << len(game["targets"])) - 1
    dp = {0: (0, [])}
    for search in searches:
        options = strict_options(search)
        ndp = {}
        for covered, (cost, choices) in dp.items():
            for o in options:
                nc, val = covered | o[0], cost + o[2]
                if nc not in ndp or val < ndp[nc][0]:
                    ndp[nc] = (val, choices + [o])
        dp = trim_dp(ndp)
        if not dp:
            break
    if full in dp:
        return dp[full][1]
    if not allow_overlay:
        return None

    dp = {0: (0, [None] * n)}
    for i in reversed(game["draw_order"]):
        options = [(g, b, c, s) for (g, b), (c, s)
                   in searches[i]["options"].items()]
        ndp = {}
        for covered, (cost, choices) in dp.items():
            for o in options:
                if o[1] & ~covered:
                    continue
                nc, val = covered | o[0], cost + o[2]
                if nc not in ndp or val < ndp[nc][0]:
                    ch = list(choices)
                    ch[i] = o
                    ndp[nc] = (val, ch)
        dp = trim_dp(ndp)
        if not dp:
            break
    return dp[full][1] if full in dp else None


def install_paths(game, choices):
    paths = {}
    for i, option in enumerate(choices):
        s = option[3]
        parents = game["searches"][i]["parents"]
        path = []
        while parents[s][0] is not None:
            previous, action, depth = parents[s]
            path.append(action)
            s = previous
        paths[i] = deque(reversed(path))
    game["paths"] = paths
    game["searches"] = []


def initialize(im):
    gates = find_gates(im)
    objects = reconstruct_objects(im, gates)
    obstacle_rows, obstacle_box = find_obstacle(im)
    order = infer_draw_order(im, objects)
    cycle = {}
    for k, i in enumerate(order):
        cycle[i] = order[(k + 1) % len(order)]
    game = {
        "objects": objects, "gates": gates,
        "targets": find_targets(im),
        "obstacle_rows": obstacle_rows, "obstacle_box": obstacle_box,
        "selected": -1, "next": cycle, "last_action": None,
        "last_object": None, "phase": "survey", "survey_steps": 0,
        "draw_order": order, "paths": None, "calls": 0, "reparses": 0,
        "finish_count": 0
    }
    game["selected"] = identify_selected(im, game)
    read_targets(im, game)
    return game


def act(obs, available, memory):
    im = np.asarray(obs)
    deadline = time.perf_counter() + 7.5
    allowed = set(available)

    def output(number):
        name = "ACTION" + str(number)
        if name not in allowed:
            name = next((a for a in available if a != "RESET"), "RESET")
        return name, None, None

    if "_re86" not in memory:
        memory["_re86"] = initialize(im)
    game = memory["_re86"]
    game["calls"] += 1
    objects = game["objects"]

    def recover(force_reset=False):
        if (force_reset and "RESET" in allowed
                and memory.get("_re86_resets", 0) < 1):
            memory["_re86_resets"] = memory.get("_re86_resets", 0) + 1
            memory.pop("_re86", None)
            return "RESET", None, None
        colors = {(x, y): c for x, y, c in game["targets"] if c is not None}
        new_game = initialize(im)
        new_game["targets"] = [
            (x, y, colors.get((x, y), c))
            for x, y, c in new_game["targets"]]
        new_game["reparses"] = game.get("reparses", 0) + 1
        new_game["last_action"] = 5
        memory["_re86"] = new_game
        return output(5)

    if not objects:
        if game["calls"] > 2:
            return recover(game["reparses"] >= 1)
        game["last_action"] = 5
        return output(5)

    old_selected = game["selected"]
    selected = identify_selected(im, game)
    if game["last_action"] == 5 and old_selected >= 0 and selected >= 0:
        game["next"][old_selected] = selected
    game["selected"] = selected

    moved = game.get("last_object")
    if game["last_action"] in (1, 2, 3, 4) and moved is not None:
        obj = objects[moved]
        vid, x, y, color = obj["state"]
        actual = observed_color(im, obj)
        if actual != color:
            obj["state"] = (vid, x, y, actual)
            if game["phase"] == "plan":
                start_planning(game)

        # Previously non-overlapping shapes may now reveal their true
        # layer order. Resolve this before comparing the predicted image.
        game["draw_order"] = infer_draw_order(
            im, objects, game["draw_order"])
        if prediction_error(im, game) >= 5:
            return recover(game["reparses"] >= 1)

    game["last_object"] = None
    read_targets(im, game)

    def issue(action):
        game["last_action"] = action
        if action in (1, 2, 3, 4) and selected >= 0:
            game["finish_count"] = 0
            obj = objects[selected]
            obj["state"] = move_state(game, obj, obj["state"], action, {}, {})
            game["last_object"] = selected
        return output(action)

    if game["phase"] == "survey":
        action = survey_action(game, selected)
        if action is not None:
            game["survey_steps"] += 1
            if game["survey_steps"] <= 24:
                return issue(action)
            if game["reparses"] < 2:
                return recover(game["reparses"] >= 1)

            known = [(x, y, c) for x, y, c in game["targets"] if c is not None]
            if known:
                corrected = []
                for x, y, c in game["targets"]:
                    if c is None:
                        votes = {}
                        for a, b, color in known:
                            d = (a - x) ** 2 + (b - y) ** 2
                            weight = (5 if a == x or b == y else 1) / (d + 16)
                            votes[color] = votes.get(color, 0) + weight
                        c = max(votes, key=votes.get)
                    corrected.append((x, y, c))
                game["targets"] = corrected
            else:
                return recover(True)
        start_planning(game)

    if game["paths"] is None:
        while time.perf_counter() < deadline - 0.35:
            i = game["search_index"]
            if i < len(objects):
                finished = search_to_depth(
                    game, objects[i], game["searches"][i],
                    game["depth_limit"], deadline - 0.35)
                if not finished:
                    return issue(5)
                game["search_index"] += 1
                continue

            choices = combine_placements(
                game, allow_overlay=game["depth_limit"] >= 36)
            if choices is not None:
                install_paths(game, choices)
                break

            if (all(s["done"] for s in game["searches"])
                    or game["depth_limit"] >= 100):
                return recover(game["reparses"] >= 1)

            old_limit = game["depth_limit"]
            game["depth_limit"] = (
                26 if old_limit < 26 else
                36 if old_limit < 36 else
                48 if old_limit < 48 else
                64 if old_limit < 64 else
                82 if old_limit < 82 else 100)
            game["search_index"] = 0

        if game["paths"] is None:
            return issue(5)

    paths = game["paths"]
    if selected >= 0 and paths[selected]:
        return issue(paths[selected].popleft())

    if any(path for path in paths.values()):
        game["finish_count"] += 1
        if game["finish_count"] > 2 * len(objects) + 2:
            return recover(game["reparses"] >= 1)
        return issue(5)

    game["finish_count"] += 1
    if game["finish_count"] > len(objects) + 1:
        return recover(game["reparses"] >= 1)
    return issue(5)

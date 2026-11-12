import numpy as np


# State:
# (x, y, width, height, vertical_arm, horizontal_arm, colour, kind)
# kind: 0 cross/line, 1 X, 2 diamond, 3 rectangle, 4 solid-centre X

def _geometry(s, cache):
    _, _, w, h, a, b, _, k = s
    key = (w, h, a, b, k)
    if key in cache:
        return cache[key]

    rows = [0] * h
    if k == 0:
        for y in range(h):
            rows[y] = (1 << a) if 0 <= a < w else 0
        if 0 <= b < h:
            rows[b] = (1 << w) - 1
    elif k in (1, 4):
        for y in range(h):
            if 0 <= y < w:
                rows[y] |= 1 << y
            z = w - 1 - y
            if 0 <= z < w:
                rows[y] |= 1 << z
    elif k == 2:
        cx, cy = w // 2, h // 2
        r = min(cx, cy)
        for y in range(h):
            d = r - abs(y - cy)
            if d >= 0:
                if 0 <= cx - d < w:
                    rows[y] |= 1 << (cx - d)
                if 0 <= cx + d < w:
                    rows[y] |= 1 << (cx + d)
    else:
        for y in range(h):
            if y in (0, h - 1):
                rows[y] = (1 << w) - 1
            else:
                rows[y] = 1 | (1 << (w - 1))

    cx, cy = w // 2, h // 2
    if k == 1:
        rows[cy] &= ~(1 << cx)

    normal = tuple(rows)
    active = list(rows)
    active[cy] |= 1 << cx
    cover = list(rows)
    cover[cy] &= ~(1 << cx)
    answer = normal, tuple(active), tuple(cover)
    cache[key] = answer
    return answer


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        x, y = int(xx), int(yy)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        comp = []
        while stack:
            px, py = stack.pop()
            comp.append((px, py))
            for ny in range(max(0, py - 1), min(h, py + 2)):
                for nx in range(max(0, px - 1), min(w, px + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
        result.append(comp)
    return result


def _objects(obs):
    wells = []
    mask = (obs == 2).copy()
    mask[63] = False
    for comp in _components(mask):
        xs, ys = zip(*comp)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 3 or h < 3:
            continue
        inner = obs[y0 + 1:y1, x0 + 1:x1]
        vals = inner[(inner >= 6) & (inner <= 14)]
        if vals.size:
            colour = int(np.bincount(vals, minlength=16).argmax())
            bits = ((1 << w) - 1) << x0
            wells.append((x0, y0, w, h, tuple([bits] * h), colour))

    blockers = []
    mask = (obs == 1).copy()
    mask[63] = False
    for comp in _components(mask):
        if len(comp) < 4:
            continue
        xs, ys = zip(*comp)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        rows = [0] * (y1 - y0 + 1)
        for x, y in comp:
            rows[y - y0] |= 1 << x
        blockers.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1,
                         tuple(rows), 1))
    return wells, blockers


def _markers(obs):
    mask = (obs == 4).copy()
    mask[63] = False
    candidates = {}
    for yy, xx in zip(*np.nonzero(mask)):
        x, y = int(xx), int(yy)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = x - dx, y - dy
                if not (1 <= cx < 63 and 1 <= cy < 62):
                    continue
                if int(obs[cy, cx]) in (4, 5, 2, 15):
                    continue
                n4 = bad = 0
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        if ox == 0 and oy == 0:
                            continue
                        v = int(obs[cy + oy, cx + ox])
                        n4 += v == 4
                        bad += v == 5
                if n4 >= 3 and bad == 0:
                    candidates[(cx, cy)] = n4

    chosen = []
    for p, score in sorted(candidates.items(), key=lambda z: -z[1]):
        if not any(abs(p[0] - q[0]) <= 2 and abs(p[1] - q[1]) <= 2
                   for q in chosen):
            chosen.append(p)
    return sorted(chosen, key=lambda p: (p[1], p[0]))


def _runs(obs):
    eligible = (((obs >= 6) & (obs <= 14)) | (obs == 0)).copy()
    eligible[63] = False
    result = {c: [[], [], [], []] for c in range(6, 15)}
    directions = ((1, 0), (0, 1), (1, 1), (-1, 1))

    for d, (dx, dy) in enumerate(directions):
        minimum = 7 if d < 2 else 5
        for y in range(63):
            for x in range(64):
                if not eligible[y, x]:
                    continue
                px, py = x - dx, y - dy
                if 0 <= px < 64 and 0 <= py < 63 and eligible[py, px]:
                    continue
                xx, yy = x, y
                counts = [0] * 16
                length = 0
                ex, ey = x, y
                while 0 <= xx < 64 and 0 <= yy < 63 and eligible[yy, xx]:
                    counts[int(obs[yy, xx])] += 1
                    length += 1
                    ex, ey = xx, yy
                    xx, yy = xx + dx, yy + dy
                if length < minimum:
                    continue
                for c in range(6, 15):
                    if counts[c] >= 5 and counts[c] >= 0.55 * length:
                        result[c][d].append((x, y, ex, ey, length))
    return result


def _parse_shapes(obs, cache):
    runs = _runs(obs)
    candidates = {}
    zeros = {(int(x), int(y)) for y, x in zip(*np.nonzero(obs[:63] == 0))}

    def add(x, y, w, h, a, b, colour, kind):
        x, y, w, h, a, b = map(int, (x, y, w, h, a, b))
        if w < 1 or h < 1 or w > 63 or h > 63:
            return
        if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
            return
        if kind in (1, 2, 4) and (w != h or w % 2 != 1):
            return
        if kind in (1, 4):
            kind = 4 if (x + w // 2, y + h // 2) in zeros else 1

        s = (x, y, w, h, a, b, colour, kind)
        rows = _geometry(s, cache)[0]
        own = set()
        other = bad = total = 0
        for iy, row in enumerate(rows):
            yy = y + iy
            if not 0 <= yy < 63:
                continue
            bits = row
            while bits:
                bit = bits & -bits
                ix = bit.bit_length() - 1
                bits -= bit
                xx = x + ix
                if not 0 <= xx < 64:
                    continue
                total += 1
                v = int(obs[yy, xx])
                if v == colour:
                    own.add(yy * 64 + xx)
                elif v == 0:
                    pass
                elif 6 <= v <= 14:
                    other += 1
                else:
                    bad += 1

        anchor = (x + w // 2, y + h // 2) in zeros
        if bad or total < 9 or len(own) < 7:
            return
        if len(own) + int(anchor) < 0.72 * total:
            return
        symmetry = 0
        if kind == 0:
            symmetry = int(a == w // 2) + int(b == h // 2)
        candidates[s] = (own, other, anchor, symmetry)

    for colour, lists in runs.items():
        hs, vs, ds1, ds2 = lists

        for hr in hs:
            x0, yy, x1, _, length = hr
            if length >= 9:
                add(x0, yy, x1 - x0 + 1, 1, 0, 0, colour, 0)
                for zx, zy in zeros:
                    if zy == yy and x0 <= zx <= x1:
                        r = min(zx - x0, x1 - zx)
                        if x0 == 0:
                            r = x1 - zx
                        elif x1 == 63:
                            r = zx - x0
                        if r >= 4:
                            add(zx - r, yy, 2 * r + 1, 1,
                                0, 0, colour, 0)

        for vr in vs:
            xx, y0, _, y1, length = vr
            if length >= 9:
                add(xx, y0, 1, y1 - y0 + 1, 0, 0, colour, 0)
                for zx, zy in zeros:
                    if zx == xx and y0 <= zy <= y1:
                        r = min(zy - y0, y1 - zy)
                        if y0 == 0:
                            r = y1 - zy
                        elif y1 == 62:
                            r = zy - y0
                        if r >= 4:
                            add(xx, zy - r, 1, 2 * r + 1,
                                0, 0, colour, 0)

        for hr in hs:
            hx0, hy, hx1, _, hl = hr
            for vr in vs:
                vx, vy0, _, vy1, vl = vr
                if not (hx0 <= vx <= hx1 and vy0 <= hy <= vy1):
                    continue
                if hl < 9 or vl < 9:
                    continue
                x0, x1, y0, y1 = hx0, hx1, vy0, vy1
                if x0 == 0:
                    x0 = 2 * vx - x1
                if x1 == 63:
                    x1 = 2 * vx - x0
                if y0 == 0:
                    y0 = 2 * hy - y1
                if y1 == 62:
                    y1 = 2 * hy - y0

                add(x0, y0, x1 - x0 + 1, y1 - y0 + 1,
                    vx - x0, hy - y0, colour, 0)
                if abs((vx - x0) - (x1 - vx)) in (1, 2):
                    r = min(vx - x0, x1 - vx)
                    add(vx - r, y0, 2 * r + 1, y1 - y0 + 1,
                        r, hy - y0, colour, 0)
                if abs((hy - y0) - (y1 - hy)) in (1, 2):
                    r = min(hy - y0, y1 - hy)
                    add(x0, hy - r, x1 - x0 + 1, 2 * r + 1,
                        vx - x0, r, colour, 0)

        for u in ds1:
            for v in ds2:
                diff = u[0] - u[1]
                summ = v[0] + v[1]
                if (summ + diff) % 2:
                    continue
                cx = (summ + diff) // 2
                cy = (summ - diff) // 2
                if not (min(u[0], u[2]) <= cx <= max(u[0], u[2])
                        and min(v[0], v[2]) <= cx <= max(v[0], v[2])):
                    continue
                r = max(abs(u[0] - cx), abs(u[2] - cx),
                        abs(v[0] - cx), abs(v[2] - cx))
                if r >= 4:
                    add(cx - r, cy - r, 2 * r + 1, 2 * r + 1,
                        r, r, colour, 1)

        for run in ds1 + ds2:
            x0, y0, x1, y1, length = run
            r = length - 1
            if r < 4:
                continue
            for cx, cy in ((x0, y1), (x1, y0)):
                add(cx - r, cy - r, 2 * r + 1, 2 * r + 1,
                    r, r, colour, 2)

        for i, u in enumerate(hs):
            for v in hs[i + 1:]:
                if u[0] != v[0] or u[2] != v[2]:
                    continue
                y0, y1 = sorted((u[1], v[1]))
                if y1 - y0 >= 4:
                    w, h = u[2] - u[0] + 1, y1 - y0 + 1
                    add(u[0], y0, w, h, w // 2, h // 2, colour, 3)

        for i, u in enumerate(vs):
            for v in vs[i + 1:]:
                if u[1] != v[1] or u[3] != v[3]:
                    continue
                x0, x1 = sorted((u[0], v[0]))
                if x1 - x0 >= 4:
                    w, h = x1 - x0 + 1, u[3] - u[1] + 1
                    add(x0, u[1], w, h, w // 2, h // 2, colour, 3)

    selected = []
    explained = set()
    remaining = dict(candidates)
    while remaining and len(selected) < 8:
        best = None
        best_score = -1e9
        for s, (own, other, anchor, symmetry) in remaining.items():
            new = len(own - explained)
            if new < 7 or new < 0.35 * len(own):
                continue
            score = (2.0 * new - 0.55 * (len(own) - new)
                     - 1.5 * other + 4 * anchor + 0.3 * symmetry)
            if score > best_score:
                best_score, best = score, s
        if best is None:
            break
        selected.append(best)
        explained.update(remaining[best][0])
        del remaining[best]
    return selected


def _hit(s, rows, obj):
    x, y, w, h = s[:4]
    ox, oy, ow, oh, orows = obj[:5]
    if x >= ox + ow or x + w <= ox or y >= oy + oh or y + h <= oy:
        return False
    for yy in range(max(y, oy), min(y + h, oy + oh)):
        bits = rows[yy - y]
        bits = bits << x if x >= 0 else bits >> (-x)
        if bits & orows[yy - oy]:
            return True
    return False


def _move(s, action, ctx):
    dx, dy = ((0, -3), (0, 3), (-3, 0), (3, 0))[action - 1]
    x, y, w, h, a, b, c, k = s
    x += dx
    y += dy
    if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
        return s

    for obj in ctx["blockers"]:
        t = (x, y, w, h, a, b, c, k)
        active = _geometry(t, ctx["geom"])[1]
        if not _hit(t, active, obj):
            continue

        if k in (2, 3):
            if dx:
                if w <= 6:
                    return s
                old_h = h
                w, h = w - 3, h + 3
                y += old_h // 2 - h // 2
                y = int(round(y / 3.0) * 3)
                if dx < 0:
                    x -= dx
            else:
                if h <= 6:
                    return s
                old_w = w
                h, w = h - 3, w + 3
                x += old_w // 2 - w // 2
                x = int(round(x / 3.0) * 3)
                if dy < 0:
                    y -= dy
            return (x, y, w, h, w // 2, h // 2, c, 3)

        if k != 0:
            return s

        ox, oy, ow, oh = obj[:4]
        vi = ox <= x + a < ox + ow
        hi = oy <= y + b < oy + oh
        if dx:
            backwards = -3 if dx > 0 else 3
            forwards = -backwards
            can_back = a > 0 if dx > 0 else a < w - 2
            can_forward = a < w - 2 if dx > 0 else a > 0
            if vi and hi:
                x -= dx
            elif vi:
                if can_back and 0 <= a + backwards < w:
                    a += backwards
                else:
                    x -= dx
            elif hi:
                x -= dx
                if can_forward and 0 <= a + forwards < w:
                    a += forwards
        else:
            backwards = -3 if dy > 0 else 3
            forwards = -backwards
            can_back = b > 0 if dy > 0 else b < h - 2
            can_forward = b < h - 2 if dy > 0 else b > 0
            if hi and vi:
                y -= dy
            elif hi:
                if can_back and 0 <= b + backwards < h:
                    b += backwards
                else:
                    y -= dy
            elif vi:
                y -= dy
                if can_forward and 0 <= b + forwards < h:
                    b += forwards

    t = (x, y, w, h, a, b, c, k)
    normal = _geometry(t, ctx["geom"])[0]
    for well in ctx["wells"]:
        nc = well[5]
        if nc != c and _hit(t, normal, well):
            c = nc
            break
    return (x, y, w, h, a, b, c, k)


def _coverage(s, targets, cache):
    x, y, w, h, _, _, colour, _ = s
    rows = _geometry(s, cache)[2]
    good = bad = 0
    cx, cy = x + w // 2, y + h // 2
    for i, (tx, ty, wanted) in enumerate(targets):
        bit = 1 << i
        if tx == cx and ty == cy:
            bad |= bit
            continue
        lx, ly = tx - x, ty - y
        if 0 <= lx < w and 0 <= ly < h and (rows[ly] >> lx) & 1:
            if wanted is None or wanted == colour:
                good |= bit
            else:
                bad |= bit
    return good, bad


def _search_shape(start, targets, ctx, limit=65000):
    states = [start]
    parent = [-1]
    incoming = [0]
    depth = [0]
    seen = {start: 0}
    strict = {}
    relaxed = {}
    head = 0

    while head < len(states):
        s = states[head]
        d = depth[head]
        good, bad = _coverage(s, targets, ctx["geom"])
        if not bad and (good not in strict or d < strict[good][0]):
            strict[good] = (d, head)
        score = d + 7 * bad.bit_count()
        if good not in relaxed or score < relaxed[good][0]:
            relaxed[good] = (score, head)

        if d < 75 and len(states) < limit:
            for action in (1, 2, 3, 4):
                t = _move(s, action, ctx)
                if t not in seen:
                    seen[t] = len(states)
                    states.append(t)
                    parent.append(head)
                    incoming.append(action)
                    depth.append(d + 1)
                    if len(states) >= limit:
                        break
        head += 1

    def materialize(table):
        result = []
        for mask, (cost, node) in table.items():
            path = []
            z = node
            while parent[z] >= 0:
                path.append(incoming[z])
                z = parent[z]
            path.reverse()
            result.append((mask, cost, states[node], path))
        return result

    return materialize(strict), materialize(relaxed)


def _combine(options, active, full):
    dp = {0: (0, [])}
    for index, choices in enumerate(options):
        ndp = {}
        for old_mask, (old_cost, selection) in dp.items():
            for mask, cost, state, path in choices:
                extra = 1 if path and index != active else 0
                nm = old_mask | mask
                nc = old_cost + cost + extra
                previous = ndp.get(nm)
                if previous is None or nc < previous[0]:
                    ndp[nm] = (nc, selection + [(state, path)])
        if not ndp:
            return None, 0
        if len(ndp) > 24000:
            ranked = sorted(
                ndp.items(),
                key=lambda z: (z[1][0] - 3 * z[0].bit_count(), z[1][0])
            )[:24000]
            ndp = dict(ranked)
        dp = ndp

    if full in dp:
        return dp[full][1], full
    best = min(dp, key=lambda m: ((full ^ m).bit_count(), dp[m][0]))
    return dp[best][1], best


def _plan(shapes, targets, active, ctx):
    strict, relaxed = [], []
    for shape in shapes:
        a, b = _search_shape(shape, targets, ctx)
        strict.append(a)
        relaxed.append(b)

    full = (1 << len(targets)) - 1
    selection, covered = _combine(strict, active, full)
    if selection is None or covered != full:
        alternate, alt_covered = _combine(relaxed, active, full)
        if alternate is not None and (
                selection is None or
                alt_covered.bit_count() > covered.bit_count()):
            selection, covered = alternate, alt_covered
    if selection is None:
        return None
    return {
        "goals": [s for s, p in selection],
        "paths": [list(p) for s, p in selection],
        "covered": covered,
        "full": full,
    }


def _active_from_image(obs, shapes):
    zeros = {(int(x), int(y)) for y, x in zip(*np.nonzero(obs[:63] == 0))}
    for i, s in enumerate(shapes):
        if (s[0] + s[2] // 2, s[1] + s[3] // 2) in zeros:
            return i
    return None


def _occupied(x, y, shapes, cache):
    for s in shapes:
        sx, sy, w, h = s[:4]
        lx, ly = x - sx, y - sy
        if not (0 <= lx < w and 0 <= ly < h):
            continue
        if lx == w // 2 and ly == h // 2:
            return True
        if (_geometry(s, cache)[0][ly] >> lx) & 1:
            return True
    return False


def _prediction_bad(obs, shapes, cache):
    # Only check pixels owned by exactly one predicted shape.
    count = np.zeros((63, 64), dtype=np.uint8)
    colours = np.zeros((63, 64), dtype=np.int8)
    for s in shapes:
        x, y, w, h, _, _, colour, _ = s
        rows = _geometry(s, cache)[0]
        for iy, row in enumerate(rows):
            yy = y + iy
            if not 0 <= yy < 63:
                continue
            bits = row
            while bits:
                bit = bits & -bits
                ix = bit.bit_length() - 1
                bits -= bit
                xx = x + ix
                if 0 <= xx < 64:
                    count[yy, xx] += 1
                    colours[yy, xx] = colour

    unique = count == 1
    total = int(unique.sum())
    if total < 15:
        return False
    bad = unique & (obs[:63] != colours) & (obs[:63] != 0)
    return int(bad.sum()) >= max(5, int(total * 0.15))


def _initialize(obs, memory, keep_labels=True):
    labels = dict(memory.get("labels", {})) if keep_labels else {}
    geom = {}
    wells, blockers = _objects(obs)
    shapes = _parse_shapes(obs, geom)
    for p in _markers(obs):
        labels.setdefault(p, None)

    memory["shapes"] = shapes
    memory["labels"] = labels
    memory["geom"] = geom
    memory["wells"] = wells
    memory["blockers"] = blockers
    memory["active"] = _active_from_image(obs, shapes)
    memory["plan"] = None
    memory["last"] = None
    memory["idle"] = 0
    memory["successors"] = {}
    memory["switch_from"] = None


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=int)
    allowed = set(available)

    def output(action):
        # RESET is a universally defined engine action.
        if action == 0:
            return "RESET", None, None
        name = "ACTION" + str(action)
        if name not in allowed:
            name = next((n for n in ("ACTION5", "ACTION1", "ACTION2",
                                     "ACTION3", "ACTION4", "RESET")
                         if n in allowed), available[0])
        return name, None, None

    memory["tick"] = memory.get("tick", 0) + 1
    board_key = obs[:63].astype(np.uint8).tobytes()
    visits = memory.setdefault("visits", {})
    visits[board_key] = visits.get(board_key, 0) + 1

    # Recover before a modelling error exhausts the step counter.
    if visits[board_key] >= 4 and memory["tick"] > 8:
        if not memory.get("reset_used"):
            memory.clear()
            memory["reset_used"] = True
            return output(0)
        if visits[board_key] >= 7:
            _initialize(obs, memory, keep_labels=False)
            memory["visits"] = {board_key: 1}

    if "shapes" not in memory:
        _initialize(obs, memory)

    shapes = memory["shapes"]
    last = memory.get("last")
    actual_active = _active_from_image(obs, shapes)

    if last == 5:
        previous = memory.get("switch_from")
        if actual_active is not None:
            if previous is not None and previous != actual_active:
                memory["successors"][previous] = actual_active
            memory["active"] = actual_active
        else:
            # Do not confuse an invisible new selection with the old one.
            memory["active"] = memory["successors"].get(previous)
    elif actual_active is not None:
        if last in (1, 2, 3, 4) and actual_active != memory.get("active"):
            _initialize(obs, memory)
            shapes = memory["shapes"]
            actual_active = _active_from_image(obs, shapes)
        memory["active"] = actual_active

    if last in (1, 2, 3, 4) and shapes:
        zeros = {(int(x), int(y))
                 for y, x in zip(*np.nonzero(obs[:63] == 0))}
        ai = memory.get("active")
        wrong_center = False
        if zeros and actual_active is None and ai is not None:
            expected = (shapes[ai][0] + shapes[ai][2] // 2,
                        shapes[ai][1] + shapes[ai][3] // 2)
            wrong_center = expected not in zeros
        if wrong_center or _prediction_bad(obs, shapes, memory["geom"]):
            _initialize(obs, memory)
            shapes = memory["shapes"]

    def switch():
        memory["switch_from"] = memory.get("active")
        memory["last"] = 5
        return output(5)

    if not shapes:
        return switch()

    geom = memory["geom"]
    labels = memory["labels"]
    changed = False
    for p in _markers(obs):
        if p not in labels:
            labels[p] = None
            changed = True

    # Target colours are immutable. Only learn an unknown colour when uncovered.
    for p, value in list(labels.items()):
        if value is not None:
            continue
        x, y = p
        visible = int(obs[y, x])
        if not _occupied(x, y, shapes, geom) and 6 <= visible <= 14:
            labels[p] = visible
            changed = True
    if changed:
        memory["plan"] = None

    active = memory.get("active")
    if active is None:
        return switch()

    targets = [(x, y, labels[(x, y)])
               for x, y in sorted(labels, key=lambda p: (p[1], p[0]))]
    ctx = {
        "geom": geom,
        "wells": memory["wells"],
        "blockers": memory["blockers"],
    }
    if not targets:
        return switch()

    if memory.get("plan") is None:
        memory["plan"] = _plan(shapes, targets, active, ctx)

    plan = memory.get("plan")
    if plan is not None:
        path = plan["paths"][active]
        if path:
            action = path.pop(0)
            predicted = _move(shapes[active], action, ctx)
            if predicted == shapes[active]:
                memory["plan"] = None
            else:
                shapes[active] = predicted
                memory["idle"] = 0
                memory["last"] = action
                return output(action)

        if memory.get("plan") is not None:
            if any(plan["paths"][i] for i in range(len(shapes))):
                return switch()

    # Reveal target centres hidden by provisional placements.
    unknown = [(x, y) for x, y, c in targets if c is None]
    if unknown:
        covering = []
        for i, s in enumerate(shapes):
            for x, y in unknown:
                if _occupied(x, y, [s], geom):
                    covering.append((i, x, y))
        current = [item for item in covering if item[0] == active]
        if current:
            _, tx, ty = current[0]
            best = None
            for action in (1, 2, 3, 4):
                t = _move(shapes[active], action, ctx)
                if t == shapes[active]:
                    continue
                uncovered = not _occupied(tx, ty, [t], geom)
                distance = abs(tx - (t[0] + t[2] // 2)) + abs(
                    ty - (t[1] + t[3] // 2))
                score = 100 * uncovered + distance
                if best is None or score > best[0]:
                    best = (score, action, t)
            if best is not None:
                _, action, t = best
                shapes[active] = t
                memory["plan"] = None
                memory["last"] = action
                return output(action)
        if covering:
            return switch()

    memory["idle"] = memory.get("idle", 0) + 1
    memory["plan"] = None

    if memory["idle"] == 2:
        _initialize(obs, memory)
        memory["idle"] = 2
        return switch()

    if memory["idle"] >= 5 and not memory.get("reset_used"):
        memory.clear()
        memory["reset_used"] = True
        return output(0)

    return switch()

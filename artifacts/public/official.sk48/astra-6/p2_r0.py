import math
import time
import heapq
from collections import deque
import numpy as np


_DIRS = (1, 16, -1, -16)
_ACTION = {1: "ACTION4", -1: "ACTION3", 16: "ACTION2", -16: "ACTION1"}


def _bases(o):
    labels = np.zeros((6, 6), dtype=np.int8)
    labels[1:5, 1:5] = 1
    labels[2:4, 2:4] = 2
    labels[2:4, 5] = 3
    corner = o[:59, :59]
    center = o[2:61, 2:61]
    found = {}

    for fixed in (False, True):
        for rotation in range(4):
            t = np.rot90(labels, -rotation)
            ring = None
            if fixed:
                good = center >= 6
            else:
                ry, rx = np.argwhere(t == 1)[0]
                ring = o[ry:ry + 59, rx:rx + 59]
                good = (center >= 6) & (corner >= 5)
                good &= (ring == 0) | (ring == 4)

            for y in range(6):
                for x in range(6):
                    label = int(t[y, x])
                    if label == 3:
                        continue
                    view = o[y:y + 59, x:x + 59]
                    if label == 2:
                        good &= view == center
                    elif fixed:
                        good &= view == 5
                    elif label == 0:
                        good &= view == corner
                    else:
                        good &= view == ring

            ys, xs = np.nonzero(good)
            for y, x in zip(ys.tolist(), xs.tolist()):
                if (x, y) not in found:
                    found[x, y] = {
                        "x": x, "y": y,
                        "id": int(o[y + 2, x + 2]),
                        "d": _DIRS[rotation],
                        "fixed": fixed,
                        "active": False if fixed else int(ring[y, x]) == 0
                    }
    return list(found.values())


def _beads(o):
    c = o[:61, :61]
    good = c >= 6
    for y in range(4):
        for x in range(4):
            v = o[y:y + 61, x:x + 61]
            if x in (0, 3) or y in (0, 3):
                good &= v == c
            else:
                good &= (v == c) | (v == 0)
    ys, xs = np.nonzero(good)
    return {(x - 1, y - 1): int(o[y, x])
            for y, x in zip(ys.tolist(), xs.tolist())}


def _stripe(a, b):
    return ((a == 1 and b == 2) or
            (a == 2 and b in (1, 3)) or
            (a == 3 and b == 2))


def _segment(o, x, y, horizontal):
    if not (0 <= x <= 58 and 0 <= y <= 58):
        return False
    if horizontal:
        return (_stripe(int(o[y + 2, x]), int(o[y + 3, x])) and
                _stripe(int(o[y + 2, x + 5]), int(o[y + 3, x + 5])))
    return (_stripe(int(o[y, x + 2]), int(o[y, x + 3])) and
            _stripe(int(o[y + 5, x + 2]), int(o[y + 5, x + 3])))


def _length(o, b):
    d = b["d"]
    dx = 6 if d == 1 else -6 if d == -1 else 0
    dy = 6 if d == 16 else -6 if d == -16 else 0
    n = 1
    while n < 12 and _segment(o, b["x"] + n * dx, b["y"] + n * dy,
                             abs(d) == 1):
        n += 1
    return n


def _encode(m, x, y):
    cx = int(round((x - m["x0"]) / 6))
    cy = int(round((y - m["y0"]) / 6))
    return (cy + 2) * 16 + cx + 2


def _inside(m, p):
    if not 0 <= p < 256:
        return False
    return (0 <= (p & 15) - 2 < m["w"] and
            0 <= (p >> 4) - 2 < m["h"])


def _rail_edges(o, m, b, p):
    horizontal = abs(b["d"]) == 1
    if horizontal:
        x = b["x"] + 2
        if not 0 <= x < 63:
            return set()
        a, z = o[:, x], o[:, x + 1]
        start = b["y"]
    else:
        y = b["y"] + 2
        if not 0 <= y < 63:
            return set()
        a, z = o[y, :], o[y + 1, :]
        start = b["x"]

    mask = ((a == z) & ((a == 2) | (a == 3))).copy()
    before = start > 0 and bool(mask[start - 1])
    after = start + 6 < 64 and bool(mask[start + 6])
    if before and after:
        mask[start:start + 6] = True
    elif before:
        mask[start:start + 4] = True
    elif after:
        mask[start + 2:start + 6] = True

    edges = set()
    extent = m["h"] if horizontal else m["w"]
    origin = m["y0"] if horizontal else m["x0"]
    for k in range(-2, extent + 2):
        if horizontal:
            q = (k + 2) * 16 + (p & 15)
            step = 16
        else:
            q = (p & ~15) + k + 2
            step = 1
        if not 0 <= q < 256:
            continue
        for sign in (-1, 1):
            sample = origin + 6 * k + 2 + 3 * sign
            if 0 <= sample < 64 and mask[sample]:
                edges.add((q, sign * step))
    return edges


def _read_state(o, m, bases=None, beads=None):
    bases = _bases(o) if bases is None else bases
    beads = _beads(o) if beads is None else beads
    top = [b for b in bases if b["y"] < 53]
    rods = []
    active = -1

    for i, info in enumerate(m["rods"]):
        candidates = [b for b in top
                      if b["id"] == info["id"] and b["d"] == info["d"]
                      and b["fixed"] == info["fixed"]]
        if not candidates:
            raise ValueError("Missing handle")
        b = candidates[0]
        rods.extend((_encode(m, b["x"], b["y"]), _length(o, b)))
        if b["active"]:
            active = i

    groups = {}
    for (x, y), color in beads.items():
        if y < 53:
            p = _encode(m, x, y)
            if _inside(m, p):
                groups.setdefault(color, []).append(p)

    positions = []
    for color, start, end in m["groups"]:
        ps = sorted(groups.get(color, []))
        if len(ps) != end - start:
            raise ValueError("Changed bead count")
        positions.extend(ps)

    if active < 0:
        active = 0
    return tuple(rods), tuple(positions), active


def _assignments(goal, bycolor):
    result = []

    def visit(j, used, path):
        if len(result) >= 3000:
            return
        if j == len(goal):
            result.append(tuple(path))
            return
        a, b = bycolor.get(goal[j], (0, 0))
        for index in range(a, b):
            if not used & (1 << index):
                path.append(index)
                visit(j + 1, used | (1 << index), path)
                path.pop()

    visit(0, 0, [])
    return result


def _make_model(o):
    bases, beads = _bases(o), _beads(o)
    field = o[:53] == 4
    rows, cols = field.sum(axis=1), field.sum(axis=0)
    yy = np.flatnonzero(rows >= max(6, int(rows.max() * .55)))
    xx = np.flatnonzero(cols >= max(6, int(cols.max() * .55)))
    if not len(xx) or not len(yy):
        raise ValueError("Missing board")

    m = {
        "x0": int(xx[0]), "y0": int(yy[0]),
        "w": int(round((int(xx[-1]) - int(xx[0]) + 1) / 6)),
        "h": int(round((int(yy[-1]) - int(yy[0]) + 1) / 6)),
        "rods": [], "fh": {}, "fv": {},
        "cost_cache": {}
    }
    if not (1 <= m["w"] <= 10 and 1 <= m["h"] <= 9):
        raise ValueError("Board size")

    valid = [False] * 256
    holes = 0
    for y in range(m["h"]):
        for x in range(m["w"]):
            px, py = m["x0"] + x * 6, m["y0"] + y * 6
            patch = o[py:py + 6, px:px + 6]
            blocked = patch.shape != (6, 6) or np.count_nonzero(patch == 5) >= 24
            valid[(y + 2) * 16 + x + 2] = not blocked
            holes += int(blocked)
    m["valid"] = valid

    bottom = [b for b in bases if b["y"] >= 53]
    top = [b for b in bases if b["y"] < 53]
    controls, fixed = [], []
    for b in top:
        partners = [z for z in bottom if z["id"] == b["id"]]
        if partners:
            controls.append((b, partners[0]))
        else:
            fixed.append(b)
    controls.sort(key=lambda pair: (pair[0]["id"], pair[0]["d"]))
    if not controls:
        raise ValueError("Missing controller")

    for b, target in controls:
        td = target["d"]
        dx = 6 if td == 1 else -6 if td == -1 else 0
        dy = 6 if td == 16 else -6 if td == -16 else 0
        goal = []
        for k in range(1, _length(o, target)):
            color = beads.get((target["x"] + k * dx, target["y"] + k * dy))
            if color is not None:
                goal.append(color)
        if not goal:
            raise ValueError("Missing target")
        p = _encode(m, b["x"], b["y"])
        m["rods"].append({
            "id": b["id"], "fixed": b["fixed"], "d": b["d"],
            "axis": 0 if abs(b["d"]) == 1 else 1,
            "sign": 1 if b["d"] > 0 else -1,
            "goal": tuple(goal),
            "click": (target["y"] + 1, target["x"] + 1),
            "side": _rail_edges(o, m, b, p),
            "lines": {}
        })

    for b in fixed:
        p, d = _encode(m, b["x"], b["y"]), b["d"]
        counts = m["fh"] if abs(d) == 1 else m["fv"]
        for k in range(_length(o, b)):
            q = p + k * d
            counts[q] = counts.get(q, 0) + 1

    colored = []
    for (x, y), color in beads.items():
        if y < 53:
            p = _encode(m, x, y)
            if _inside(m, p):
                colored.append((color, p))
    colored.sort()
    m["colors"] = tuple(c for c, p in colored)
    groups = []
    i = 0
    while i < len(colored):
        j = i + 1
        while j < len(colored) and colored[j][0] == colored[i][0]:
            j += 1
        groups.append((colored[i][0], i, j))
        i = j
    m["groups"] = groups
    m["bycolor"] = {c: (a, b) for c, a, b in groups}

    for info in m["rods"]:
        info["assignments"] = _assignments(info["goal"], m["bycolor"])
        repeated = []
        for j in range(len(info["goal"])):
            for k in range(j):
                if info["goal"][j] == info["goal"][k]:
                    repeated.append((k, j))
        info["repeated"] = repeated

    distances = [[99] * 256 for _ in range(256)]
    cells = [p for p in range(256) if valid[p]]
    for source in cells:
        dist = distances[source]
        dist[source] = 0
        queue = deque([source])
        while queue:
            p = queue.popleft()
            for d in _DIRS:
                q = p + d
                if not 0 <= q < 256 or not valid[q] or dist[q] != 99:
                    continue
                perpendicular = m["fv"] if abs(d) == 1 else m["fh"]
                if perpendicular.get(p, 0) or perpendicular.get(q, 0):
                    continue
                dist[q] = dist[p] + 1
                queue.append(q)
    m["dist"] = distances

    # Detect the general "two three-bead strings sharing their middle bead"
    # subproblem. Its geometric heuristic enforces both targets jointly.
    m["cross"] = None
    if len(m["rods"]) == 2 and not fixed and not holes:
        ih = next((i for i, z in enumerate(m["rods"]) if z["axis"] == 0), None)
        iv = next((i for i, z in enumerate(m["rods"]) if z["axis"] == 1), None)
        if ih is not None and iv is not None:
            gh, gv = m["rods"][ih]["goal"], m["rods"][iv]["goal"]
            if (len(gh) == len(gv) == 3 and
                    gh[0] == gh[2] and gv[0] == gv[2] and
                    gh[1] == gv[1] and len({gh[0], gv[0], gh[1]}) == 3):
                ah, bh = m["bycolor"].get(gh[0], (0, 0))
                av, bv = m["bycolor"].get(gv[0], (0, 0))
                ae, be = m["bycolor"].get(gh[1], (0, 0))
                if bh - ah == 2 and bv - av == 2 and be - ae == 1 and len(colored) == 5:
                    m["cross"] = (ih, iv, ah, bh - 1, av, bv - 1, ae)
    return m, _read_state(o, m, bases, beads)


def _context(m, s):
    r, bp, active = s
    bh, bv = m["fh"].copy(), m["fv"].copy()
    segments = []
    for i, info in enumerate(m["rods"]):
        p, length = r[2 * i:2 * i + 2]
        cells = tuple(p + k * info["d"] for k in range(length))
        segments.append(cells)
        counts = bh if info["axis"] == 0 else bv
        for q in cells:
            counts[q] = counts.get(q, 0) + 1
    return {p: i for i, p in enumerate(bp)}, bh, bv, segments


def _move(m, s, index, delta, context=None):
    r, bp, active = s
    info = m["rods"][index]
    p, length = r[2 * index:2 * index + 2]
    forward = info["d"]
    parallel = abs(delta) == abs(forward)
    valid = m["valid"]

    def free(q):
        return 0 <= q < 256 and valid[q]

    if delta == forward:
        if not free(p + length * forward):
            return None
        new_p, new_length = p, length + 1
    elif delta == -forward:
        if length <= 1:
            return None
        new_p, new_length = p, length - 1
    else:
        if (p, delta) not in info["side"]:
            return None
        new_p, new_length = p + delta, length

    bmap, bh, bv, segments = _context(m, s) if context is None else context
    perpendicular_axis = 1 if abs(delta) == 1 else 0
    perpendicular = bv if perpendicular_axis else bh
    moved = 0

    def push(pos, ignore_pos=-1000, ignore_axis=-1):
        nonlocal moved
        bi = bmap.get(pos)
        if bi is None:
            return True
        bit = 1 << bi
        if moved & bit:
            return True
        dest = pos + delta
        if not free(dest):
            return False
        for q in (pos, dest):
            count = perpendicular.get(q, 0)
            if ignore_axis == perpendicular_axis and q == ignore_pos:
                count -= 1
            if count > 0:
                return False
        if dest in bmap and not push(dest):
            return False
        moved |= bit
        return True

    cells = segments[index]
    if delta == -forward:
        cells = cells[1:]
    for q in cells:
        if not free(q + delta) and not (forward == -delta or not free(q)):
            if not parallel:
                return None
            continue
        for bead_position in (q, q + delta):
            if bead_position in bmap:
                ok = push(bead_position, q, info["axis"])
                if not ok and not parallel:
                    return None

    nr = list(r)
    nr[2 * index], nr[2 * index + 1] = new_p, new_length
    if moved:
        nb = [q + delta if moved & (1 << i) else q for i, q in enumerate(bp)]
        for color, a, b in m["groups"]:
            nb[a:b] = sorted(nb[a:b])
        nb = tuple(nb)
    else:
        nb = bp
    return tuple(nr), nb, index


def _line(m, info, base):
    result = info["lines"].get(base)
    if result is not None:
        return result
    cells = []
    entered = False
    for k in range(max(m["w"], m["h"]) + 3):
        p = base + k * info["d"]
        if _inside(m, p):
            entered = True
            if not m["valid"][p]:
                break
            cells.append((p, k))
        elif entered:
            break
    info["lines"][base] = cells
    return cells


def _held(m, s):
    r, bp, active = s
    bead_colors = dict(zip(bp, m["colors"]))
    strings = []
    won = True
    for i, info in enumerate(m["rods"]):
        base, length = r[2 * i:2 * i + 2]
        string = tuple(bead_colors[q] for q in
                       (base + k * info["d"] for k in range(length))
                       if q in bead_colors)
        strings.append(string)
        if string[:len(info["goal"])] != info["goal"]:
            won = False
    return strings, won


def _cross_heuristic(m, s):
    ih, iv, ha, hb, va, vb, ei = m["cross"]
    r, bp, active = s
    hp, hl = r[2 * ih:2 * ih + 2]
    vp, vl = r[2 * iv:2 * iv + 2]
    hx0, hy0 = (hp & 15) - 2, (hp >> 4) - 2
    vx0, vy0 = (vp & 15) - 2, (vp >> 4) - 2
    ex, ey = (bp[ei] & 15) - 2, (bp[ei] >> 4) - 2

    hax, hay = (bp[ha] & 15) - 2, (bp[ha] >> 4) - 2
    hbx, hby = (bp[hb] & 15) - 2, (bp[hb] >> 4) - 2
    vax, vay = (bp[va] & 15) - 2, (bp[va] >> 4) - 2
    vbx, vby = (bp[vb] & 15) - 2, (bp[vb] >> 4) - 2
    hlow, hhigh = min(hax, hbx), max(hax, hbx)
    vlow, vhigh = min(vay, vby), max(vay, vby)
    hs, vs = m["rods"][ih]["sign"], m["rods"][iv]["sign"]

    best = 1000.0
    for cy in range(1, m["h"] - 1):
        ht = abs(hay - cy) + abs(hby - cy)
        upper, lower = min(cy - 1, vlow), max(cy + 1, vhigh)
        vt = max(0, vlow - cy + 1) + max(0, cy + 1 - vhigh)
        vneed = max((upper - vy0) * vs, (lower - vy0) * vs) + 1
        ecost_y = 1.25 * abs(ey - cy)
        rail_y = .55 * abs(hy0 - cy)
        for cx in range(1, m["w"] - 1):
            left, right = min(cx - 1, hlow), max(cx + 1, hhigh)
            ht_x = max(0, hlow - cx + 1) + max(0, cx + 1 - hhigh)
            hneed = max((left - hx0) * hs, (right - hx0) * hs) + 1
            value = (ht + ht_x + vt + abs(vax - cx) + abs(vbx - cx) +
                     ecost_y + 1.25 * abs(ex - cx) +
                     rail_y + .55 * abs(vx0 - cx) +
                     .75 * (max(0, hneed - hl) + max(0, vneed - vl)))
            if value < best:
                best = value
    return 1.65 * best


def _bead_costs(m, info, base, length, pos, slots):
    key = (info["d"], base, length, pos)
    cached = m["cost_cache"].get(key)
    if cached is not None:
        return cached
    if info["axis"] == 0:
        bead_k = ((pos & 15) - (base & 15)) * info["sign"]
    else:
        bead_k = ((pos >> 4) - (base >> 4)) * info["sign"]
    dist = m["dist"][pos]
    result = []
    for target, k in slots:
        reach = max(0, min(bead_k, k) - length + 1)
        result.append(2 * min(30, dist[target]) + reach +
                      max(0, k - length + 1) // 2)
    result = tuple(result)
    m["cost_cache"][key] = result
    return result


def _heuristic(m, s):
    strings, won = _held(m, s)
    if won:
        return 0.0, True
    if m["cross"] is not None:
        return _cross_heuristic(m, s), False

    r, bp, active = s
    total = 0.0
    for i, info in enumerate(m["rods"]):
        base, length = r[2 * i:2 * i + 2]
        goal, held = info["goal"], strings[i]
        n = len(goal)
        if held[:n] == goal:
            continue

        dp = [2 * j for j in range(n + 1)]
        edit = dp[n]
        for color in held:
            nd = [dp[0] + 1]
            for j in range(1, n + 1):
                nd.append(min(dp[j] + 1, nd[j - 1] + 2,
                              dp[j - 1] + (0 if color == goal[j - 1] else 3)))
            dp = nd
            edit = min(edit, dp[n])
        if len(bp) == n:
            edit = dp[n]

        slots = _line(m, info, base)
        values = [_bead_costs(m, info, base, length, p, slots) for p in bp]
        if info["axis"] == 0:
            axis_keys = [((p & 15) * info["sign"], p) for p in bp]
        else:
            axis_keys = [((p >> 4) * info["sign"], p) for p in bp]

        transport = 10000
        assignments = info["assignments"]
        for assignment in assignments:
            if any(axis_keys[assignment[a]] > axis_keys[assignment[b]]
                   for a, b in info["repeated"]):
                continue
            dp = [0] + [10000] * n
            for slot_index in range(len(slots)):
                for j in range(n - 1, -1, -1):
                    v = dp[j] + values[assignment[j]][slot_index]
                    if v < dp[j + 1]:
                        dp[j + 1] = v
            val = dp[n]
            if val >= 10000:
                val = min(dp[j] + 18 * (n - j) for j in range(n + 1))
            if val < transport:
                transport = val

        if transport == 10000:
            transport = 20 * n

        acquisition = 0
        if len(m["rods"]) == 1:
            # A rod cannot collect a bead merely by approaching it sideways:
            # that pushes the bead ahead. It must first uncover its position.
            for color in set(goal):
                a, b = m["bycolor"].get(color, (0, 0))
                required = goal.count(color)
                candidate_costs = []
                for pos in bp[a:b]:
                    if info["axis"] == 0:
                        k = ((pos & 15) - (base & 15)) * info["sign"]
                        aligned = pos >> 4 == base >> 4
                    else:
                        k = ((pos >> 4) - (base >> 4)) * info["sign"]
                        aligned = pos & 15 == base & 15
                    cost = 0 if aligned else max(0, length - max(1, k))
                    candidate_costs.append(cost)
                candidate_costs.sort()
                if len(candidate_costs) >= required:
                    acquisition = max(acquisition, candidate_costs[required - 1])

        total += 2.1 * edit + .42 * transport + .95 * acquisition
    return total, False


def _new_search(m, start, attempt):
    h, won = _heuristic(m, start)
    weight = 2.35 if m["cross"] is not None else 2.1
    weight += min(1.2, .25 * attempt)
    cap = 65 if m["cross"] is not None else 125
    return {
        "root": start,
        "nodes": [(start, -1, -1, 0, 0, h)],
        "heap": [(weight * h, h, 0)],
        "seen": {start: 0},
        "hc": {(start[0], start[1]): (h, won)},
        "weight": weight, "maxg": cap,
        "best": 0, "bestvalue": h,
        "expanded": 0, "calls": 0,
    }


def _compact(search):
    heap = search["heap"]
    if len(heap) > 65000:
        heap = heapq.nsmallest(65000, heap)
    nodes = search["nodes"]
    keep = {0, search["best"]}
    for _, _, index in heap:
        while index >= 0 and index not in keep:
            keep.add(index)
            index = nodes[index][1]
    index = search["best"]
    while index >= 0:
        keep.add(index)
        index = nodes[index][1]

    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    compact = []
    for old in order:
        state, parent, rod, delta, g, h = nodes[old]
        compact.append((state, remap.get(parent, -1), rod, delta, g, h))
    search["nodes"] = compact
    search["heap"] = [(f, h, remap[i]) for f, h, i in heap if i in remap]
    heapq.heapify(search["heap"])
    search["best"] = remap[search["best"]]
    if len(search["hc"]) > 180000:
        search["hc"] = {
            (state[0], state[1]): (h, False)
            for state, parent, rod, delta, g, h in compact
        }


def _advance(m, search, deadline):
    if len(search["nodes"]) > 280000:
        _compact(search)
    nodes = search["nodes"]
    heap = search["heap"]
    seen = search["seen"]
    hcache = search["hc"]
    weight = search["weight"]
    expanded = 0
    search["calls"] += 1

    while heap:
        if expanded % 24 == 0:
            if time.monotonic() >= deadline or len(nodes) >= 360000:
                break
        _, _, ni = heapq.heappop(heap)
        state, parent, old_rod, old_delta, g, h = nodes[ni]
        if seen.get(state) != g:
            continue
        expanded += 1
        context = _context(m, state)

        # Try continuing with the currently selected rod first.
        rod_order = [state[2]] + [i for i in range(len(m["rods"])) if i != state[2]]
        for rod in rod_order:
            ng = g + (1 if state[2] == rod else 2)
            if ng > search["maxg"]:
                continue
            forward = m["rods"][rod]["d"]
            directions = (forward, -forward, 16, -16) if abs(forward) == 1 else (
                forward, -forward, 1, -1)
            for delta in directions:
                ns = _move(m, state, rod, delta, context)
                if ns is None or ng >= seen.get(ns, 1000000):
                    continue
                geo = (ns[0], ns[1])
                hv = hcache.get(geo)
                if hv is None:
                    hv = _heuristic(m, ns)
                    hcache[geo] = hv
                nh, won = hv
                seen[ns] = ng
                index = len(nodes)
                nodes.append((ns, ni, rod, delta, ng, nh))
                if won:
                    search["expanded"] += expanded
                    return index
                value = nh + .035 * ng
                if value < search["bestvalue"]:
                    search["bestvalue"] = value
                    search["best"] = index
                heapq.heappush(heap, (ng + weight * nh, nh, index))

    search["expanded"] += expanded
    return None


def _unroll(m, search, end):
    nodes = search["nodes"]
    chain = []
    while nodes[end][1] >= 0:
        chain.append(end)
        end = nodes[end][1]
    chain.reverse()
    plan = []
    for ni in chain:
        state, parent, rod, delta, g, h = nodes[ni]
        previous = nodes[parent][0]
        if previous[2] != rod:
            row, col = m["rods"][rod]["click"]
            selected = (previous[0], previous[1], rod)
            plan.append(("ACTION6", row, col, selected))
        plan.append((_ACTION[delta], None, None, state))
    return plan


def _wait(m, state, available, memory):
    if "ACTION6" in available:
        row, col = m["rods"][state[2]]["click"]
        memory["expected"] = state
        memory["waiting"] = memory.get("waiting", 0) + 1
        return "ACTION6", int(row), int(col)
    return None


def _fallback(available, memory):
    count = memory.get("errors", 0) + 1
    memory["errors"] = count
    memory.pop("search", None)
    memory.pop("plan", None)
    memory.pop("expected", None)
    if "ACTION7" in available and count <= 2:
        return "ACTION7", None, None
    if "RESET" in available and memory.get("resets", 0) < 1:
        resets = memory.get("resets", 0) + 1
        memory.clear()
        memory["resets"] = resets
        return "RESET", None, None
    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if name in available:
            return name, None, None
    return available[0], None, None


def act(obs, available, memory):
    started = time.monotonic()
    o = np.asarray(obs, dtype=np.int8)
    memory["steps"] = memory.get("steps", 0) + 1

    try:
        if "model" not in memory:
            model, state = _make_model(o)
            memory["model"] = model
            memory["attempt"] = 0
            memory["waiting"] = 0
        else:
            model = memory["model"]
            state = _read_state(o, model)

        expected = memory.get("expected")
        if expected is not None and expected != state:
            memory.pop("plan", None)
            memory.pop("search", None)
            memory["mismatches"] = memory.get("mismatches", 0) + 1

        plan = memory.get("plan", [])
        if not plan:
            search = memory.get("search")
            if search is None or search["root"] != state:
                search = _new_search(model, state, memory.get("attempt", 0))
                memory["search"] = search

            solution = _advance(model, search, started + 8.25)
            if solution is not None:
                plan = _unroll(model, search, solution)
                memory["plan"] = plan
                memory.pop("search", None)
            else:
                fixed = bool(model["fh"] or model["fv"])
                wait_cap = 28 if fixed else 17 if model["cross"] is not None else 10
                calls_cap = 20 if fixed else 14
                can_wait = (
                    search["heap"] and
                    search["calls"] < calls_cap and
                    memory.get("waiting", 0) < wait_cap
                )
                if can_wait:
                    result = _wait(model, state, available, memory)
                    if result is not None:
                        return result

                # If a complete plan is still unavailable, use a path that was
                # evaluated by the persistent search, rather than resetting.
                best = search["best"]
                if best == 0 and search["heap"]:
                    best = search["heap"][0][2]
                plan = _unroll(model, search, best)
                memory.pop("search", None)
                memory["attempt"] = memory.get("attempt", 0) + 1
                if not plan:
                    # Continue a broader search when the current beam exhausted.
                    search = _new_search(model, state, memory["attempt"])
                    search["weight"] += .8
                    memory["search"] = search
                    result = _wait(model, state, available, memory)
                    if result is not None:
                        return result
                    return _fallback(available, memory)
                memory["plan"] = plan

        name, row, col, predicted = memory["plan"].pop(0)
        if name not in available:
            return _fallback(available, memory)
        memory["expected"] = predicted
        memory["errors"] = 0
        return name, row, col

    except Exception:
        return _fallback(available, memory)

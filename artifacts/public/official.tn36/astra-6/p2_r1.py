import numpy as np
from collections import deque
import heapq
import math


_ROBOT = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=bool)

_TARGET = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

_CHECK = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
], dtype=bool)

# g: geometric transition; r: clockwise quarter-turns; c: colour.
_OPS = {
    1: ("g", 0), 2: ("g", 1), 3: ("g", 2),
    5: ("r", 1), 6: ("r", 3), 7: ("r", 2),
    8: ("g", 6), 9: ("g", 7),
    10: ("g", 4), 11: ("g", 4),
    12: ("g", 5), 13: ("g", 5),
    14: ("c", 9), 15: ("c", 8),
    16: ("r", 3), 33: ("g", 3),
    34: ("g", 0), 63: ("c", 15),
}


def _runs(values, spacing):
    values = sorted(set(values))
    out = []
    for v in values:
        if not out or v != out[-1][-1] + spacing:
            out.append([v])
        else:
            out[-1].append(v)
    return out


def _read_ui(a):
    pairs = []
    for y in range(20, 49):
        for x in range(3, 61):
            c = int(a[y, x])
            if c not in (1, 5):
                continue
            if not (a[y, x-1] == c and a[y, x+1] == c):
                continue
            if a[y, x-2] != 0 or a[y, x+2] != 0:
                continue
            if np.any(a[y-1, x-1:x+2] != 0):
                continue
            if np.any(a[y+1, x-1:x+2] != 0):
                continue
            v = int(a[y+3, x])
            if v not in (1, 5):
                continue
            if not np.all(a[y+2:y+5, x] == v):
                continue
            if np.any(a[y+2:y+5, x-1] != 0):
                continue
            if np.any(a[y+2:y+5, x+1] != 0):
                continue
            pairs.append((x, y))

    groups = {}
    for x, y in pairs:
        groups.setdefault((x % 5, y % 6), set()).add((x, y))

    best = None
    for points in groups.values():
        for xs in _runs([p[0] for p in points], 5):
            if len(xs) < 2:
                continue
            ys = [y for y in sorted({p[1] for p in points})
                  if all((x, y) in points for x in xs)]
            for yr in _runs(ys, 6):
                score = len(xs) * len(yr)
                candidate = (score, sum(xs) / len(xs), xs, yr)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

    if best is None:
        return None
    xs, ys = best[2], best[3]
    return (tuple(xs), int(ys[0]), 2 * len(ys))


def _program(a, ui):
    xs, y0, bits = ui
    return tuple(sum((1 << b) for b in range(bits)
                     if int(a[y0 + 3*b, x]) == 5) for x in xs)


def _board(a, ui):
    xs, y0, bits = ui
    candidates = []
    for y in range(2, y0 - 3):
        row = a[y]
        x = 0
        while x < 64:
            if row[x] != 0:
                x += 1
                continue
            left = x
            while x < 64 and row[x] == 0:
                x += 1
            right = x - 1
            if right-left+1 >= 20 and left < min(xs) and right > max(xs):
                candidates.append((y, left, right))
    if candidates:
        top, left, right = min(candidates)
        return (left + 1, top + 1, right - 1, y0 - 2)
    return (max(1, min(xs)-2), 4, min(62, max(xs)+2), y0-2)


def _run_button(a):
    mask = np.ones((9, 9), dtype=bool)
    mask[0, :2] = mask[0, -2:] = False
    mask[-1, :2] = mask[-1, -2:] = False
    mask[1, 0] = mask[1, -1] = False
    mask[-2, 0] = mask[-2, -1] = False
    best = None
    for y in range(49, 56):
        for x in range(0, 56):
            window = a[y:y+9, x:x+9]
            n = int(np.count_nonzero(
                ((window == 9) | (window == 10)) & mask))
            if n >= 63:
                candidate = (n, x, y)
                if best is None or candidate > best:
                    best = candidate
    if best:
        return (best[2]+4, best[1]+4)
    return (58, 57)


def _shapes(a, base, rect, kind):
    left, top, right, bottom = rect
    roi = np.zeros((64, 64), dtype=bool)
    roi[max(0, top):min(64, bottom+1),
        max(0, left):min(64, right+1)] = True

    colours = [c for c in (11, 9, 8, 15, 7, 14)
               if np.any((a == c) & roi)]
    max_scale = min(6, max(1, (min(right-left+1, bottom-top+1)+3)
                           // base.shape[0]))
    rotations = range(2) if kind == "check" else range(4)
    found = []

    for scale in range(1, max_scale+1):
        for rot in rotations:
            mask = np.rot90(base, -rot)
            if scale > 1:
                mask = np.repeat(np.repeat(mask, scale, 0), scale, 1)
            h, w = mask.shape
            on_y, on_x = np.nonzero(mask)
            off_y, off_x = np.nonzero(~mask)
            ay, ax = int(on_y[0]), int(on_x[0])

            for colour in colours:
                yy, xx = np.nonzero((a == colour) & roi)
                xx, yy = xx-ax, yy-ay
                valid = ((xx >= 0) & (yy >= 0) &
                         (xx+w <= 64) & (yy+h <= 64) &
                         (xx >= left-w) & (xx <= right) &
                         (yy >= top-h) & (yy <= bottom))
                xx, yy = xx[valid], yy[valid]
                if not len(xx):
                    continue

                # Filter candidates incrementally rather than constructing
                # a large image-window tensor.
                for dy, dx in zip(on_y, on_x):
                    good = a[yy+dy, xx+dx] == colour
                    xx, yy = xx[good], yy[good]
                    if not len(xx):
                        break
                if not len(xx):
                    continue

                extra = np.zeros(len(xx), dtype=np.int16)
                for dy, dx in zip(off_y, off_x):
                    extra += (a[yy+dy, xx+dx] == colour)

                tolerance = 4*scale*scale if kind == "target" else 0
                for x, y, e in zip(xx, yy, extra):
                    if int(e) <= tolerance:
                        found.append((int(x), int(y), scale, rot,
                                      colour, int(e)))

    found.sort(key=lambda z: (z[5] / (z[2]*z[2]), -z[2], z[1], z[0]))
    return found


def _pose(shape, target=False):
    x, y, s, r, c = shape[:5]
    if target:
        x, y = x+s, y+s
    return (x, y, s, r, c)


def _erase(mask, x, y, w, h):
    mask[max(0, y):min(64, y+h), max(0, x):min(64, x+w)] = False


def _component_rectangles(mask):
    result = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = [(x, y)]
        lo_x = hi_x = x
        lo_y = hi_y = y
        for px, py in queue:
            lo_x, hi_x = min(lo_x, px), max(hi_x, px)
            lo_y, hi_y = min(lo_y, py), max(hi_y, py)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = px+dx, py+dy
                    if (0 <= nx < 64 and 0 <= ny < 64 and
                            mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((nx, ny))
        result[lo_y:hi_y+1, lo_x:hi_x+1] = True
    return result


def _observe(a, model=None):
    if model is None:
        ui = _read_ui(a)
        if ui is None:
            return None, None
        rect = _board(a, ui)
        targets = _shapes(a, _TARGET, rect, "target")
        if not targets:
            return None, None
        target_shape = targets[0]
        model = {
            "ui": ui, "rect": rect, "button": _run_button(a),
            "target_shape": target_shape,
            "goal": _pose(target_shape, True),
            "checks": set(),
            "walls": np.zeros((64, 64), dtype=bool),
            "bad": np.zeros((64, 64), dtype=bool),
        }

    rect = model["rect"]
    robots = _shapes(a, _ROBOT, rect, "robot")
    if not robots:
        return model, None
    robot = robots[0]
    pose = _pose(robot)

    checks = _shapes(a, _CHECK, rect, "check")
    for x, y, s, r, c, e in checks:
        model["checks"].add((x, y, s))

    left, top, right, bottom = rect
    roi = np.zeros((64, 64), dtype=bool)
    roi[top:bottom+1, left:right+1] = True

    # Walls occupy grid-aligned tiles. Tile completion restores portions
    # hidden by the target without filling in the inside of L-shaped walls.
    raw_wall = (a == 6) & roi
    walls = np.zeros_like(raw_wall)
    rx, ry = pose[0] % 4, pose[1] % 4
    for y in range(ry-4, 64, 4):
        for x in range(rx-4, 64, 4):
            ys, ye = max(0, y), min(64, y+4)
            xs, xe = max(0, x), min(64, x+4)
            if ys < ye and xs < xe and np.any(raw_wall[ys:ye, xs:xe]):
                walls[ys:ye, xs:xe] = True
    model["walls"] |= walls

    unknown = roi & (a != 4) & (a != 5) & (a != 6)
    x, y, s = robot[:3]
    _erase(unknown, x, y, 4*s, 4*s)
    x, y, s = model["target_shape"][:3]
    _erase(unknown, x, y, 6*s, 6*s)
    for x, y, s in model["checks"]:
        _erase(unknown, x, y, 4*s, 4*s)
    model["bad"] |= _component_rectangles(unknown)
    return model, pose


def _prefix(mask):
    p = np.zeros((65, 65), dtype=np.int32)
    p[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), 0), 1)
    return p


def _hit(p, x, y, size):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x+size), min(64, y+size)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(p[y1, x1] - p[y0, x1] - p[y1, x0] + p[y0, x0])


def _reverse_dist(reverse, sources, limit=None):
    inf = 1000000
    dist = [inf] * len(reverse)
    queue = deque()
    for source in sources:
        if dist[source]:
            dist[source] = 0
            queue.append(source)
    while queue:
        v = queue.popleft()
        nd = dist[v]+1
        if limit is not None and nd > limit:
            continue
        for u in reverse[v]:
            if dist[u] > nd:
                dist[u] = nd
                queue.append(u)
    return dist


def _solve(model, start, current, failures, visits, seed, mode):
    ui = model["ui"]
    n, bits = len(current), ui[2]
    max_code = (1 << bits)-1
    goal = model["goal"]
    sx, sy, ss, sr, sc = start
    gx, gy, gs, gr, gc = goal

    if (sx-gx) % 4 or (sy-gy) % 4:
        return None

    effects = {_OPS.get(op, ("n", 0)) for op in range(max_code+1)}
    if sc != gc and ("c", gc) not in effects:
        return None
    rotations = {arg for kind, arg in effects if kind == "r"}
    if sr != gr and not rotations:
        return None

    left, top, right, bottom = model["rect"]
    max_s = max([ss, gs, 1] + [p[2] for p in model["checks"]])
    max_s = min(6, max_s)
    margin = 8 + 4*max_s
    min_x = min(left, sx, gx)-margin
    max_x = max(right, sx, gx)+margin
    min_y = min(top, sy, gy)-margin
    max_y = max(bottom, sy, gy)+margin
    rx, ry = sx % 4, sy % 4
    x0 = rx + 4*math.floor((min_x-rx)/4)
    y0 = ry + 4*math.floor((min_y-ry)/4)

    wp = _prefix(model["walls"])
    bp = _prefix(model["bad"])
    nodes = []
    index = {}
    for s in range(1, max_s+1):
        for y in range(y0, max_y+1, 4):
            for x in range(x0, max_x+1, 4):
                if (x, y, s) == start[:3] or not _hit(wp, x, y, 4*s):
                    index[(x, y, s)] = len(nodes)
                    nodes.append((x, y, s))
    start_id = index.get(start[:3])
    goal_id = index.get(goal[:3])
    if start_id is None or goal_id is None:
        return None

    bad = [_hit(bp, x, y, 4*s) for x, y, s in nodes]
    motion = [(-4, 0), (4, 0), (0, 4), (0, -4), (8, 0), (-8, 0)]
    transitions = []
    for i, (x, y, s) in enumerate(nodes):
        row = []
        for dx, dy in motion:
            nx, ny = x+dx, y+dy
            if _hit(wp, nx, ny, 4*s):
                row.append(i)
            else:
                row.append(index.get((nx, ny, s), -1))
        for delta in (1, -1):
            ns = max(1, s+delta)
            if _hit(wp, x, y, 4*ns):
                ns = max(1, ns-delta)
            row.append(index.get((x, y, ns), -1))
        transitions.append(row)

    geom_effects = sorted(arg for kind, arg in effects if kind == "g")
    reverse = [[] for _ in nodes]
    for i, row in enumerate(transitions):
        for j in {row[e] for e in geom_effects}:
            if j >= 0 and j != i and not (mode == 0 and bad[j]):
                reverse[j].append(i)

    distance = _reverse_dist(reverse, [goal_id])
    if distance[start_id] >= 1000000:
        return None

    checkpoint_nodes = set()
    for i, (x, y, s) in enumerate(nodes):
        # Saved endpoints should remain visible for reliable observation.
        if x < left or y < top or x+4*s > right+1 or y+4*s > bottom+1:
            continue
        if mode == 0 and bad[i]:
            continue
        for cx, cy, cs in model["checks"]:
            if (s == cs and x < cx+4*cs and x+4*s > cx and
                    y < cy+4*cs and y+4*s > cy):
                checkpoint_nodes.add(i)
                break

    # Number of future program executions, ignoring orientation and colour.
    remaining_runs = [1 if d <= n else 1000000 for d in distance]
    frontier = {i for i in checkpoint_nodes if remaining_runs[i] == 1}
    covered = set(frontier)
    round_number = 2
    while frontier:
        d = _reverse_dist(reverse, frontier, n)
        for i, di in enumerate(d):
            if di <= n and remaining_runs[i] > round_number:
                remaining_runs[i] = round_number
        frontier = {i for i in checkpoint_nodes
                    if i not in covered and d[i] <= n}
        covered |= frontier
        round_number += 1

    start_key = (start_id << 6) | (sr << 4) | sc
    goal_key = (goal_id << 6) | (gr << 4) | gc
    failed_set = set(failures)

    options = []
    for k, old in enumerate(current):
        best = {}
        for op in range(max_code+1):
            effect = _OPS.get(op, ("n", 0))
            cost = float((op ^ old).bit_count())
            if failures:
                cost += 0.42 * sum(p[k] == op for p in failures[-8:])
            if seed:
                cost += (((op*73 + k*151 + seed*97) % 101) / 20000.0)
            entry = (cost, op, effect[0], effect[1])
            if effect not in best or entry[:2] < best[effect][:2]:
                best[effect] = entry
        options.append(list(best.values()))

    def lower_bound(gid, rot, colour):
        d = distance[gid]
        if rot != gr:
            delta = (gr-rot) % 4
            d += 1 if delta in rotations else 2
        if colour != gc:
            d += 1
        return d

    def unpack(path):
        return tuple((path >> (6*k)) & 63 for k in range(n))

    def search(direct):
        if direct and lower_bound(start_id, sr, sc) > n:
            return None
        states = {start_key: (0.0, 0)}
        for k in range(n):
            new = {}
            active = (mode == 0 or
                      (mode == 1 and (k//3) % 2 == 0) or
                      (mode == 2 and (k//3) % 2 == 1))
            next_active = (mode == 1 and ((k+1)//3) % 2 == 0) or (
                           mode == 2 and ((k+1)//3) % 2 == 1)
            toggles_on = (mode in (1, 2) and k % 3 == 2 and
                          k < n-1 and next_active)
            left_steps = n-k-1

            for key, (cost, path) in states.items():
                gid, rot, colour = key >> 6, (key >> 4) & 3, key & 15
                for extra, op, kind, arg in options[k]:
                    ng, nr, nc = gid, rot, colour
                    if kind == "g":
                        ng = transitions[gid][arg]
                        if ng < 0 or (active and bad[ng]):
                            continue
                    elif kind == "r":
                        nr = (rot+arg) % 4
                    elif kind == "c":
                        nc = arg
                    if toggles_on and bad[ng]:
                        continue
                    if direct and lower_bound(ng, nr, nc) > left_steps:
                        continue

                    nk = (ng << 6) | (nr << 4) | nc
                    value = cost+extra
                    if seed:
                        value += ((ng*29 + nr*13 + op*7 + k + seed)
                                  % 31) / 50000.0
                    previous = new.get(nk)
                    if previous is None or value < previous[0]:
                        new[nk] = (value, path | (op << (6*k)))

            if not new:
                return None
            if len(new) > 18000:
                selected = heapq.nsmallest(
                    18000, new.items(),
                    key=lambda item: item[1][0] +
                    0.25 * min(80, distance[item[0] >> 6]))
                new = dict(selected)
            states = new

        if direct:
            item = states.get(goal_key)
            if item is None:
                return None
            program = unpack(item[1])
            if program in failed_set:
                return None
            return program, goal, item[0]

        best_result = None
        best_score = None
        for key, (cost, path) in states.items():
            gid, rot, colour = key >> 6, (key >> 4) & 3, key & 15
            if gid not in checkpoint_nodes or key == start_key:
                continue
            if remaining_runs[gid] >= 1000000:
                continue
            program = unpack(path)
            if program in failed_set:
                continue
            x, y, s = nodes[gid]
            end = (x, y, s, rot, colour)
            score = (
                remaining_runs[gid],
                distance[gid] + 2*(rot != gr) + 2*(colour != gc)
                + 0.12*cost + 8*visits.get(end, 0),
                cost,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_result = (program, end, cost)
        return best_result

    result = search(True)
    if result is not None:
        return result
    if checkpoint_nodes:
        return search(False)
    return None


def _emit(a, memory):
    model = memory["model"]
    plan = memory["plan"]
    current = _program(a, model["ui"])
    xs, y0, bits = model["ui"]
    for i, (old, wanted) in enumerate(zip(current, plan["program"])):
        difference = old ^ wanted
        if difference:
            bit = (difference & -difference).bit_length()-1
            return ("ACTION6", int(y0+3*bit), int(xs[i]))

    memory["last_run"] = {
        "start": plan["start"],
        "expected": plan["end"],
        "program": tuple(plan["program"]),
    }
    del memory["plan"]
    row, col = model["button"]
    return ("ACTION6", int(row), int(col))


def act(obs, available, memory):
    a = np.asarray(obs)

    if "plan" in memory and "model" in memory:
        return _emit(a, memory)

    model, pose = _observe(a, memory.get("model"))
    if model is None or pose is None:
        memory.clear()
        if "RESET" in available:
            return ("RESET", None, None)
        row, col = _run_button(a)
        return ("ACTION6", int(row), int(col))

    memory["model"] = model
    failures = memory.setdefault("failures", {})
    visits = memory.setdefault("visits", {})
    seed = memory.get("seed", 0)

    previous = memory.pop("last_run", None)
    if previous is not None:
        visits[pose] = visits.get(pose, 0)+1
        if pose != previous["expected"]:
            key = previous["start"]
            history = failures.setdefault(key, [])
            history.append(previous["program"])
            failures[key] = history[-12:]
            seed += 1
            memory["seed"] = seed

    current = _program(a, model["ui"])
    history = failures.get(pose, [])
    modes = [0, 1, 2, 3]
    if len(history) >= 2:
        modes = [1, 2, 0, 3]

    result = None
    for mode in modes:
        result = _solve(model, pose, current, history, visits,
                        seed + (3*mode if history else 0), mode)
        if result is not None:
            break

    if result is None:
        # A fresh observation after resetting is preferable to repeatedly
        # executing a program whose predicted outcome has been contradicted.
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        memory["seed"] = seed+1
        row, col = model["button"]
        return ("ACTION6", int(row), int(col))

    program, end, cost = result
    memory["plan"] = {
        "program": tuple(program),
        "start": pose,
        "end": end,
    }
    return _emit(a, memory)

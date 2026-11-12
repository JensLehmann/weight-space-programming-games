import heapq
import itertools
import numpy as np


# Instruction semantics. Multiple numbers can encode the same operation.
_TN_OPS = {
    "wait": [0],
    "left": [1, 34],
    "right": [2],
    "down": [3],
    "up": [33],
    "cw": [5],
    "ccw": [6, 16],
    "half": [7],
    "grow": [8],
    "shrink": [9],
    "right2": [10, 11],
    "left2": [12, 13],
    "color9": [14],
    "color8": [15],
    "color15": [63],
}
_TN_KNOWN = {v for values in _TN_OPS.values() for v in values}

_TN_ROBOT = np.ones((4, 4), dtype=bool)
_TN_ROBOT[3, 1:3] = False

_TN_GOAL = np.zeros((6, 6), dtype=bool)
_TN_GOAL[1:5, 0] = True
_TN_GOAL[1:5, 5] = True
_TN_GOAL[4, 2:4] = True
_TN_GOAL[5, :] = True

_TN_CHECK = np.fromfunction(
    lambda y, x: (x + y) % 2 == 0, (4, 4), dtype=int
).astype(bool)


def _tn_components(mask):
    """Eight-connected components, returned as (size, x0, y0, x1, y1)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        stack = [(xx, yy)]
        size = 0
        x0 = x1 = xx
        y0 = y1 = yy
        while stack:
            x, y = stack.pop()
            size += 1
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
        result.append((size, x0, y0, x1 + 1, y1 + 1))
    return result


def _tn_controls(a):
    columns = {}
    for y in range(17, 53):
        for x in range(1, 61):
            c = int(a[y, x])
            if c not in (1, 5):
                continue
            # Horizontal stroke.
            if (a[y, x + 1] == c and a[y, x + 2] == c
                    and a[y, x - 1] == 0 and a[y, x + 3] == 0
                    and np.all(a[y - 1, x:x + 3] == 0)
                    and np.all(a[y + 1, x:x + 3] == 0)):
                columns.setdefault(x + 1, []).append((y, x + 1, 0))
            # Vertical stroke.
            if (y < 51 and a[y + 1, x] == c and a[y + 2, x] == c
                    and a[y - 1, x] == 0 and a[y + 3, x] == 0
                    and np.all(a[y:y + 3, x - 1] == 0)
                    and np.all(a[y:y + 3, x + 1] == 0)):
                columns.setdefault(x, []).append((y + 1, x, 1))

    good = {}
    for x, points in columns.items():
        points = sorted(set(points))
        # Find the longest alternating, regularly spaced sequence.
        best = []
        for start in range(len(points)):
            if points[start][2] != 0:
                continue
            seq = [points[start]]
            for point in points[start + 1:]:
                if (point[0] == seq[-1][0] + 3
                        and point[2] != seq[-1][2]):
                    seq.append(point)
            if len(seq) > len(best):
                best = seq
        if len(best) >= 2:
            good[x] = best

    groups = []
    for x in sorted(good):
        if (groups and x == groups[-1][-1] + 5
                and [p[0] for p in good[x]]
                == [p[0] for p in good[groups[-1][-1]]]):
            groups[-1].append(x)
        else:
            groups.append([x])
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return None

    # The editable execution panel is the rightmost panel.
    group = max(groups, key=lambda g: (g[-1], len(g)))
    return [[(p[0], p[1]) for p in good[x]] for x in group]


def _tn_read_program(a, controls):
    return [
        sum((1 << b) for b, (y, x) in enumerate(column)
            if int(a[y, x]) == 5)
        for column in controls
    ]


def _tn_board(a, controls):
    first_y = min(y for column in controls for y, x in column)
    min_x = min(column[0][1] for column in controls)
    max_x = max(column[0][1] for column in controls)
    candidates = []
    for y in range(2, max(3, first_y - 2)):
        x = 0
        while x < 64:
            if a[y, x] != 0:
                x += 1
                continue
            start = x
            while x < 64 and a[y, x] == 0:
                x += 1
            if (x - start >= 15 and start <= min_x
                    and x > max_x):
                candidates.append((x - start, -y, start, y, x - 1))
    if candidates:
        _, _, left, top, right = max(candidates)
    else:
        left = max(0, min_x - 3)
        right = min(63, max_x + 3)
        top = 3
    return (left + 1, top + 1, right, first_y - 1)


def _tn_matches(a, base, rect, colors, scales, hole_fraction=0.0):
    """Match opaque template pixels; transparent pixels help reject impostors."""
    left, top, right, bottom = rect
    left, top = max(0, left), max(0, top)
    right, bottom = min(64, right), min(64, bottom)
    found = []
    for scale in scales:
        for rotation in range(4):
            mask = np.rot90(base, -rotation)
            if scale != 1:
                mask = np.repeat(np.repeat(mask, scale, axis=0),
                                 scale, axis=1)
            h, w = mask.shape
            oh, ow = bottom - top - h + 1, right - left - w + 1
            if oh <= 0 or ow <= 0:
                continue
            opaque = list(zip(*np.nonzero(mask)))
            # Check separated pixels first for fast rejection.
            opaque = opaque[::2] + opaque[1::2]
            holes = ~mask
            allowance = int(holes.sum() * hole_fraction + 0.01)
            for color in colors:
                eq = a == color
                valid = np.ones((oh, ow), dtype=bool)
                for dy, dx in opaque:
                    valid &= eq[top + dy:top + dy + oh,
                                left + dx:left + dx + ow]
                    if not valid.any():
                        break
                for ry, rx in zip(*np.nonzero(valid)):
                    x, y = left + int(rx), top + int(ry)
                    errors = int(np.count_nonzero(
                        eq[y:y + h, x:x + w] & holes))
                    if errors <= allowance:
                        found.append((errors, x, y, scale, rotation,
                                      int(color), w, h))
    return found


def _tn_overlap(a, b):
    return (a[0] < b[2] and a[2] > b[0]
            and a[1] < b[3] and a[3] > b[1])


def _tn_button(a, controls):
    mask = a == 9
    mask[:50, :] = False
    components = _tn_components(mask)
    candidates = [
        c for c in components
        if c[0] >= 45 and 7 <= c[3] - c[1] <= 13
        and 7 <= c[4] - c[2] <= 13
    ]
    if candidates:
        _, x0, y0, x1, y1 = max(candidates)
        return ((y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2)
    return (57, min(60, controls[-1][0][1] - 2))


def _tn_perceive(a, old=None):
    controls = _tn_controls(a)
    if controls is None:
        return old
    rect = _tn_board(a, controls)
    left, top, right, bottom = rect
    # Include a one-pixel margin for transparent template borders.
    match_rect = (max(0, left - 1), max(0, top - 1),
                  min(64, right + 1), min(64, bottom + 1))
    region = a[top:bottom, left:right]
    colors = [c for c in (11, 8, 9, 15, 7, 14)
              if np.any(region == c)]
    goals = _tn_matches(a, _TN_GOAL, match_rect, colors,
                        range(4, 0, -1), 0.05)
    if goals:
        goal_match = min(goals, key=lambda z: (-z[3], z[0], z[2], z[1]))
        _, x, y, scale, rot, color, w, h = goal_match
        goal = (x + scale, y + scale, scale, rot, color)
        goal_box = (x, y, x + w, y + h)
    elif old is not None:
        goal, goal_box = old["goal"], old["goal_box"]
    else:
        return None

    robots = _tn_matches(a, _TN_ROBOT, match_rect, colors,
                         range(5, 0, -1), 0.5)
    if robots:
        robot_match = min(robots, key=lambda z: (-z[3], z[0], z[2], z[1]))
        _, x, y, scale, rot, color, w, h = robot_match
        robot = (x, y, scale, rot, color)
        robot_box = (x, y, x + w, y + h)
    elif old is not None:
        robot = old["robot"]
        x, y, scale, _, _ = robot
        robot_box = (x, y, x + 4 * scale, y + 4 * scale)
    else:
        return None

    same = old is not None and old["goal"] == goal and old["rect"] == rect
    checkpoints = list(old["checkpoints"]) if same else []
    matches = _tn_matches(a, _TN_CHECK, match_rect, [11],
                          range(4, 0, -1), 0.0)
    for z in sorted(matches, key=lambda z: (-z[3], z[2], z[1])):
        _, x, y, scale, _, _, w, h = z
        box = (x, y, x + w, y + h)
        if _tn_overlap(box, robot_box) or _tn_overlap(box, goal_box):
            continue
        cp = (x, y, scale)
        if cp in checkpoints:
            continue
        if any(_tn_overlap(box, (cx, cy, cx + 4 * cs, cy + 4 * cs))
               for cx, cy, cs in checkpoints):
            continue
        checkpoints.append(cp)

    walls = np.zeros((64, 64), dtype=bool)
    walls[top:bottom, left:right] = region == 6

    # Remaining unusual board pixels are potential periodic hazards.
    unusual = np.zeros((64, 64), dtype=bool)
    ordinary = np.isin(region, [0, 4, 5, 6])
    unusual[top:bottom, left:right] = ~ordinary
    boxes = [robot_box, goal_box]
    boxes += [(x, y, x + 4 * s, y + 4 * s)
              for x, y, s in checkpoints]
    for x0, y0, x1, y1 in boxes:
        unusual[max(0, y0):min(64, y1),
                max(0, x0):min(64, x1)] = False
    hazards = np.zeros((64, 64), dtype=bool)
    for count, x0, y0, x1, y1 in _tn_components(unusual):
        if count >= 2:
            hazards[y0:y1, x0:x1] = True
    if same:
        hazards |= old["hazards"]

    return {
        "controls": controls,
        "rect": rect,
        "button": _tn_button(a, controls),
        "robot": robot,
        "goal": goal,
        "goal_box": goal_box,
        "checkpoints": checkpoints,
        "walls": walls,
        "hazards": hazards,
    }


def _tn_integral(mask):
    return np.pad(np.asarray(mask, dtype=np.int32),
                  ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _tn_hit(integral, x, y, size):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + size), min(64, y + size)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(integral[y1, x1] - integral[y0, x1]
                - integral[y1, x0] + integral[y0, x0])


def _tn_plan(model, current, banned, ignore_hazards=False):
    start = model["robot"]
    goal = model["goal"]
    sx, sy, ss, sr, sc = start
    tx, ty, ts, tr, tc = goal
    n = len(current)
    bits = len(model["controls"][0])
    limit = 1 << bits
    variants = {k: [v for v in vals if v < limit]
                for k, vals in _TN_OPS.items()}
    variants = {k: vals for k, vals in variants.items() if vals}
    variants["wait"] = [v for v in range(limit)
                        if v == 0 or v not in _TN_KNOWN]

    color_op = "color" + str(tc)
    need_color = sc != tc
    if need_color and color_op not in variants:
        return None
    if (sx - tx) % 4 or (sy - ty) % 4:
        return None

    max_scale = max([ss, ts] + [p[2] for p in model["checkpoints"]])
    left, top, right, bottom = model["rect"]
    xlo = max(-16, min(left - 12, sx - 8, tx - 8))
    xhi = min(80, max(right + 12, sx + 8, tx + 8))
    ylo = max(-16, min(top - 12, sy - 8, ty - 8))
    yhi = min(80, max(bottom + 12, sy + 8, ty + 8))
    xs = [x for x in range(xlo, xhi + 1) if (x - sx) % 4 == 0]
    ys = [y for y in range(ylo, yhi + 1) if (y - sy) % 4 == 0]
    geometries = [(x, y, s) for s in range(1, max_scale + 1)
                  for y in ys for x in xs]
    index = {g: i for i, g in enumerate(geometries)}
    if (tx, ty, ts) not in index:
        return None
    source = index[(sx, sy, ss)]
    target = index[(tx, ty, ts)]

    wall_integral = _tn_integral(model["walls"])
    hazard_integral = _tn_integral(
        np.zeros((64, 64), dtype=bool)
        if ignore_hazards else model["hazards"])

    directions = {
        "left": (-4, 0), "right": (4, 0),
        "up": (0, -4), "down": (0, 4),
        "left2": (-8, 0), "right2": (8, 0),
    }
    adjacency = [[] for _ in geometries]
    reverse = [[] for _ in geometries]
    checkpoints = [False] * len(geometries)
    min_cost = {op: min(v.bit_count() for v in vals)
                for op, vals in variants.items()}

    for gid, (x, y, scale) in enumerate(geometries):
        size = 4 * scale
        box = (x, y, x + size, y + size)
        checkpoints[gid] = any(
            scale == cs and _tn_overlap(
                box, (cx, cy, cx + 4 * cs, cy + 4 * cs))
            for cx, cy, cs in model["checkpoints"])
        for op in list(directions) + ["grow", "shrink"]:
            if op not in variants:
                continue
            if op in directions:
                dx, dy = directions[op]
                nx, ny, ns = x + dx, y + dy, scale
                if _tn_hit(wall_integral, nx, ny, size):
                    nx, ny = x, y
            else:
                delta = 1 if op == "grow" else -1
                ns = max(1, scale + delta)
                nx, ny = x, y
                if ns > max_scale:
                    continue
                if _tn_hit(wall_integral, nx, ny, 4 * ns):
                    ns = max(1, ns - delta)
                if ns > max_scale:
                    continue
            dest = index.get((nx, ny, ns))
            if dest is None or _tn_hit(hazard_integral, nx, ny, 4 * ns):
                continue
            adjacency[gid].append((op, dest))
            if dest != gid:
                reverse[dest].append((gid, min_cost[op]))

    inf = 10 ** 9
    distance = [inf] * len(geometries)
    distance[target] = 0
    heap = [(0, target)]
    while heap:
        d, gid = heapq.heappop(heap)
        if d != distance[gid]:
            continue
        for prev, cost in reverse[gid]:
            nd = d + cost
            if nd < distance[prev]:
                distance[prev] = nd
                heapq.heappush(heap, (nd, prev))
    if distance[source] == inf:
        return None

    rotations = [(op, delta) for op, delta in
                 (("cw", 1), ("ccw", 3), ("half", 2))
                 if op in variants]
    rot_distance = [inf] * 4
    rot_distance[tr] = 0
    for _ in range(4):
        for r in range(4):
            for op, delta in rotations:
                rot_distance[r] = min(
                    rot_distance[r],
                    min_cost[op] + rot_distance[(r + delta) % 4])
    if rot_distance[sr] == inf:
        return None
    color_cost = min_cost.get(color_op, inf) if need_color else 0

    # Each instruction is chosen by its Hamming distance from the current
    # column. Future runs use a neutral baseline estimate and are replanned.
    choices = []
    for value in current + [0]:
        choices.append({
            op: min(((value ^ v).bit_count(), v) for v in vals)
            for op, vals in variants.items()
        })
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + current[i].bit_count()

    def heuristic(gid, rotation, colored, slot, first):
        d = distance[gid] + rot_distance[rotation]
        if not colored:
            d += color_cost
        if first:
            d -= suffix[slot]
        return max(0, d) + 1

    def actual_state(gid, rotation, colored):
        x, y, scale = geometries[gid]
        return (x, y, scale, rotation, tc if colored else sc)

    # More than one path per state is retained after an observed failure.
    width = 4 if banned else 1
    records = {}
    counter = itertools.count()
    queue = []
    initial = (source, sr, int(not need_color), 0, 1)

    def push(key, cost, prefix, expected):
        gid, rotation, colored, slot, first = key
        if distance[gid] == inf or rot_distance[rotation] == inf:
            return
        old = records.get(key, [])
        if width == 1:
            if old and old[0][0] <= cost:
                return
            records[key] = [(cost, prefix)]
        else:
            for oc, op in old:
                if op == prefix and oc <= cost:
                    return
            old = [(oc, op) for oc, op in old if op != prefix]
            if len(old) >= width and cost >= max(v[0] for v in old):
                return
            old.append((cost, prefix))
            old.sort(key=lambda z: z[0])
            records[key] = old[:width]
        h = heuristic(gid, rotation, colored, slot, first)
        heapq.heappush(queue, (
            cost + 1.12 * h, cost, next(counter), key, prefix, expected))

    push(initial, 0, (), None)
    fallback = None
    fallback_score = inf
    expansions = 0

    while queue and expansions < 65000:
        _, cost, _, key, prefix, expected = heapq.heappop(queue)
        if (cost, prefix) not in records.get(key, ()):
            continue
        expansions += 1
        gid, rotation, colored, slot, first = key

        if slot == n:
            if first and prefix in banned:
                continue
            here = actual_state(gid, rotation, colored)
            if gid == target and rotation == tr and colored:
                return list(prefix), here if first else expected
            if not checkpoints[gid]:
                continue
            if first and here != start:
                score = cost + heuristic(gid, rotation, colored, 0, 0)
                if score < fallback_score:
                    fallback_score = score
                    fallback = (list(prefix), here)
            push((gid, rotation, colored, 0, 0),
                 cost + 1, prefix, here if first else expected)
            continue

        table = choices[slot] if first else choices[-1]

        def take(op, dest, nr, nc):
            edit, code = table[op]
            new_prefix = prefix + (code,) if first else prefix
            push((dest, nr, nc, slot + 1, first),
                 cost + edit, new_prefix, expected)

        take("wait", gid, rotation, colored)
        for op, dest in adjacency[gid]:
            take(op, dest, rotation, colored)
        for op, delta in rotations:
            take(op, gid, (rotation + delta) % 4, colored)
        if color_op in variants:
            take(color_op, gid, rotation, 1)

    return fallback


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int16)
    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    old = memory.get("_tn_model")
    awaiting = memory.pop("_tn_awaiting", None)
    if old is None or awaiting is not None:
        model = _tn_perceive(a, old)
        if model is None:
            # A safe recovery if the frame cannot yet be interpreted.
            if "RESET" in available and memory.get("_tn_resets", 0) < 2:
                resets = memory.get("_tn_resets", 0) + 1
                memory.clear()
                memory["_tn_resets"] = resets
                return "RESET", None, None
            controls = _tn_controls(a)
            if controls:
                y, x = _tn_button(a, controls)
                return "ACTION6", int(y), int(x)
            return "ACTION6", 58, 57
        memory["_tn_model"] = model

        if awaiting is not None:
            previous_start, program, expected = awaiting
            if tuple(model["robot"]) != tuple(expected):
                failures = memory.setdefault("_tn_failures", {})
                failed = failures.setdefault(tuple(previous_start), [])
                if tuple(program) not in failed:
                    failed.append(tuple(program))
            memory.pop("_tn_plan", None)
    else:
        model = old

    controls = model["controls"]
    current = _tn_read_program(a, controls)
    plan = memory.get("_tn_plan")

    if plan is None or len(plan[0]) != len(current):
        failed = memory.get("_tn_failures", {}).get(
            tuple(model["robot"]), [])
        banned = set(failed)
        result = _tn_plan(model, current, banned, False)
        if result is None and np.any(model["hazards"]):
            # If a permanently closed interpretation has no route, try
            # timing/path alternatives against the periodic objects.
            result = _tn_plan(model, current, banned, True)
        if result is None:
            if "RESET" in available and memory.get("_tn_resets", 0) < 2:
                resets = memory.get("_tn_resets", 0) + 1
                memory.clear()
                memory["_tn_resets"] = resets
                return "RESET", None, None
            # Re-execution can reveal another phase of an uncertain object.
            y, x = model["button"]
            memory["_tn_awaiting"] = (
                tuple(model["robot"]), tuple(current), tuple(model["robot"]))
            return "ACTION6", int(y), int(x)
        plan = result
        memory["_tn_plan"] = plan

    desired, expected = plan
    for column, (have, want) in enumerate(zip(current, desired)):
        difference = have ^ want
        if difference:
            bit = (difference & -difference).bit_length() - 1
            y, x = controls[column][bit]
            return "ACTION6", int(y), int(x)

    memory["_tn_awaiting"] = (
        tuple(model["robot"]), tuple(desired), tuple(expected))
    memory.pop("_tn_plan", None)
    y, x = model["button"]
    return "ACTION6", int(y), int(x)

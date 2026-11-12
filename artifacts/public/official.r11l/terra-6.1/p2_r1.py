import math
from collections import deque
from itertools import permutations

ANCHOR = [
    (0, -2), (-1, -1), (0, -1), (1, -1),
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    (-1, 1), (0, 1), (1, 1), (0, 2),
]
OUTLINE = [p for p in ANCHOR if p != (0, 0)]
BALL = [
    (x, y)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if not (abs(x) == 2 and abs(y) == 2)
]
RING = [
    (-1, -3), (1, -3),
    (-2, -2), (2, -2),
    (-3, -1), (3, -1),
    (-3, 1), (3, 1),
    (-2, 2), (2, 2),
    (-1, 3), (1, 3),
]
GOAL_MASK = [
    (x, y)
    for y in range(-3, 4)
    for x in range(-3, 4)
    if (
        (abs(y) == 3 and abs(x) <= 1)
        or (abs(y) == 2 and abs(x) <= 2)
        or abs(y) <= 1
    )
]
BALL_HITS = {
    (bx - px, by - py)
    for bx, by in BALL
    for px, py in BALL
}
GOAL_HITS = {
    (gx - bx, gy - by)
    for bx, by in BALL
    for gx, gy in GOAL_MASK
}
PALETTE = {4, 7, 8, 9, 11, 12, 13, 14, 15}


def _hit_ball(a, b):
    return (a[0] - b[0], a[1] - b[1]) in BALL_HITS


def _hit_goal(a, b):
    return (a[0] - b[0], a[1] - b[1]) in GOAL_HITS


def _anchors(f):
    found = []
    for y in range(2, 62):
        for x in range(2, 62):
            core = int(f[y, x])
            if core not in PALETTE:
                continue
            n0 = sum(int(f[y + dy, x + dx]) == 0 for dx, dy in OUTLINE)
            n3 = sum(int(f[y + dy, x + dx]) == 3 for dx, dy in OUTLINE)
            if max(n0, n3) >= 8:
                found.append({
                    "x": x,
                    "y": y,
                    "color": core,
                    "active": n0 > n3,
                })
    return found


def _mode(values):
    counts = {}
    for v in values:
        if v in PALETTE:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None, 0
    return max(counts.items(), key=lambda item: item[1])


def _rings(f):
    found = []
    for y in range(3, 61):
        for x in range(3, 61):
            best = None
            for axis in (0, 1):
                left, right = [], []
                for dx, dy in RING:
                    v = int(f[y + dy, x + dx])
                    if (dx if axis == 0 else dy) < 0:
                        left.append(v)
                    else:
                        right.append(v)
                a, na = _mode(left)
                b, nb = _mode(right)
                if a is not None and b is not None and na >= 4 and nb >= 4:
                    candidate = (na + nb, tuple(sorted({a, b})))
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                found.append({
                    "x": x,
                    "y": y,
                    "colors": list(best[1]),
                    "quality": best[0],
                })

    found.sort(key=lambda r: -r["quality"])
    result = []
    for r in found:
        if not any(
            abs(r["x"] - q["x"]) <= 2
            and abs(r["y"] - q["y"]) <= 2
            for q in result
        ):
            result.append(r)
    return result


def _disk_score(f, x, y):
    if not (2 <= x <= 61 and 2 <= y <= 61):
        return 0
    return sum(
        int(f[y + dy, x + dx]) in ({0} | PALETTE)
        for dx, dy in BALL
    )


def _white_groups(f, anchors, indices):
    n = min(len(indices), 12)
    indices = indices[:n]
    options = []

    for mask in range(1, 1 << n):
        count = mask.bit_count()
        if count < 2:
            continue
        xs = ys = 0
        for j in range(n):
            if mask & (1 << j):
                xs += anchors[indices[j]]["x"]
                ys += anchors[indices[j]]["y"]
        x, y = xs // count, ys // count
        quality = _disk_score(f, x, y)
        if quality >= 18:
            options.append((mask, quality))

    options.sort(key=lambda z: (-z[1], -z[0].bit_count()))
    options = options[:150]
    by_index = [[] for _ in range(n)]
    for option in options:
        for j in range(n):
            if option[0] & (1 << j):
                by_index[j].append(option)

    memo = {}

    def solve(remaining):
        if not remaining:
            return 0, []
        if remaining in memo:
            return memo[remaining]
        bit = remaining & -remaining
        j = bit.bit_length() - 1
        best_score, best_groups = solve(remaining ^ bit)
        for mask, quality in by_index[j]:
            if mask & remaining != mask:
                continue
            score, groups = solve(remaining ^ mask)
            score += 100 * mask.bit_count() + quality + 1
            if score > best_score:
                best_score = score
                best_groups = [mask] + groups
        memo[remaining] = (best_score, best_groups)
        return memo[remaining]

    _, masks = solve((1 << n) - 1)
    return [
        [indices[j] for j in range(n) if mask & (1 << j)]
        for mask in masks
    ]


def _center(group, anchors):
    ids = group["anchors"]
    if not ids:
        return 0, 0
    return (
        sum(anchors[i]["x"] for i in ids) // len(ids),
        sum(anchors[i]["y"] for i in ids) // len(ids),
    )


def _pickup_patterns():
    left = {
        (1, 0), (0, 1), (1, 1), (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3), (1, 4), (2, 4),
    }
    right = {
        (2, 0), (3, 0), (2, 1), (3, 1), (4, 1),
        (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (3, 4),
    }
    return left, right


def _pickups(f, ball_centers):
    left, right = _pickup_patterns()
    cells = [(dx + 2, dy + 2) for dx, dy in BALL]
    result = []

    for y in range(2, 62):
        for x in range(2, 62):
            if any(
                abs(x - bx) <= 1 and abs(y - by) <= 1
                for bx, by in ball_centers
            ):
                continue
            vals = [
                int(f[y + yy - 2, x + xx - 2])
                for xx, yy in cells
            ]
            colors = set(vals) & PALETTE
            if not colors or vals.count(0) < 7:
                continue
            best = None
            for color in colors:
                for side, pattern in enumerate((left, right)):
                    score = sum(
                        int(v == (color if (xx, yy) in pattern else 0))
                        for (xx, yy), v in zip(cells, vals)
                    )
                    if best is None or score > best[0]:
                        best = (score, color, side)
            if best and best[0] >= 20:
                result.append({
                    "x": x,
                    "y": y,
                    "color": best[1],
                    "side": best[2],
                    "quality": best[0],
                    "gone": False,
                })

    result.sort(key=lambda p: -p["quality"])
    kept = []
    for p in result:
        if not any(
            abs(p["x"] - q["x"]) <= 2
            and abs(p["y"] - q["y"]) <= 2
            for q in kept
        ):
            kept.append(p)
    return kept


def _initialize(f, memory):
    anchors = _anchors(f)
    groups = []

    colors = sorted({a["color"] for a in anchors if a["color"] != 4})
    for color in colors:
        ids = [i for i, a in enumerate(anchors) if a["color"] == color]
        if ids:
            groups.append({"anchors": ids, "color": color, "paint": []})

    whites = [i for i, a in enumerate(anchors) if a["color"] == 4]
    for ids in _white_groups(f, anchors, whites):
        groups.append({"anchors": ids, "color": 4, "paint": []})

    centers = [_center(g, anchors) for g in groups]
    active = next((i for i, a in enumerate(anchors) if a["active"]), None)
    if active is None and anchors:
        active = 0

    memory["anchors"] = anchors
    memory["groups"] = groups
    memory["rings"] = _rings(f)
    memory["pickups"] = _pickups(f, centers)
    memory["selected"] = active
    memory["danger"] = []
    memory["recent"] = []
    memory["pending"] = None
    memory["initialized"] = True


def _update_rings(f, memory):
    rings = memory["rings"]
    for r in _rings(f):
        nearby = next(
            (
                q for q in rings
                if abs(q["x"] - r["x"]) <= 2
                and abs(q["y"] - r["y"]) <= 2
            ),
            None,
        )
        if nearby is None:
            rings.append(r)
        elif r["quality"] > nearby.get("quality", 0):
            nearby.update(r)


def _update_after_move(f, memory):
    pending = memory.get("pending")
    if not pending:
        return False

    anchors = memory["anchors"]
    i = pending["anchor"]
    old = tuple(pending["old"])
    new = tuple(pending["new"])
    seen = _anchors(f)
    old_seen = any(
        a["color"] == anchors[i]["color"]
        and (a["x"], a["y"]) == old
        for a in seen
    )
    new_seen = any(
        a["color"] == anchors[i]["color"]
        and (a["x"], a["y"]) == new
        for a in seen
    )

    reverted = old != new and old_seen and not new_seen
    if reverted:
        anchors[i]["x"], anchors[i]["y"] = old
        memory.setdefault("bad_centers", []).append(pending["ball"])
    memory["pending"] = None
    return reverted


def _update_paint(f, memory):
    anchors = memory["anchors"]
    groups = memory["groups"]
    pickups = memory["pickups"]
    possible = {p["color"] for p in pickups}
    for r in memory["rings"]:
        possible.update(r["colors"])
    possible.discard(4)

    for group in groups:
        if group["color"] != 4:
            continue
        x, y = _center(group, anchors)
        if not (2 <= x <= 61 and 2 <= y <= 61):
            continue
        counts = {}
        visible = 0
        for dx, dy in BALL:
            c = int(f[y + dy, x + dx])
            if c == 0 or c in possible:
                visible += 1
            if c in possible:
                counts[c] = counts.get(c, 0) + 1

        # The ball is above pieces and rings in the rendering order.
        # Ignore isolated ring pixels and replace, rather than accumulate,
        # colors because a later piece can overwrite an earlier one.
        if visible >= 14:
            group["paint"] = sorted(
                color for color, count in counts.items() if count >= 4
            )

    for p in pickups:
        if p["gone"]:
            continue
        for group in groups:
            if (
                group["color"] == 4
                and p["color"] in group["paint"]
                and _hit_ball(_center(group, anchors), (p["x"], p["y"]))
            ):
                p["gone"] = True
                break


def _update_danger(f, memory):
    danger = {tuple(p) for p in memory.get("danger", [])}
    for y in range(64):
        for x in range(1, 64):
            if int(f[y, x]) == 10:
                danger.add((x, y))

    # A connection line can hide hazard pixels.
    inferred = set()
    for y in range(1, 63):
        for x in range(1, 63):
            if int(f[y, x]) != 1:
                continue
            neighbors = sum(
                (x + dx, y + dy) in danger
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if dx or dy
            )
            if neighbors >= 3:
                inferred.add((x, y))
    danger.update(inferred)
    memory["danger"] = [list(p) for p in danger]
    return danger


def _route_cost(start, colors, goal, pickups):
    missing = set(goal["colors"]) - set(colors)
    points = []
    for color in missing:
        choices = [
            (p["x"], p["y"])
            for p in pickups
            if not p["gone"] and p["color"] == color
        ]
        if not choices:
            return 10000.0
        points.append(min(choices, key=lambda q: math.dist(start, q)))

    end = (goal["x"], goal["y"])
    if not points:
        return math.dist(start, end)
    if len(points) <= 5:
        best = float("inf")
        for order in permutations(points):
            cost = math.dist(start, order[0])
            cost += sum(
                math.dist(order[i], order[i + 1])
                for i in range(len(order) - 1)
            )
            cost += math.dist(order[-1], end)
            best = min(best, cost)
        return best
    return sum(math.dist(start, p) for p in points)


def _assign_white_goals(memory):
    groups = memory["groups"]
    anchors = memory["anchors"]
    pickups = memory["pickups"]
    white = [i for i, g in enumerate(groups) if g["color"] == 4]
    if not white:
        return

    rings = memory["rings"]
    mixed = [r for r in rings if len(r["colors"]) >= 2]
    pickup_colors = {p["color"] for p in pickups}

    if len(mixed) >= len(white):
        candidates = mixed
    else:
        candidates = [
            r for r in rings
            if len(r["colors"]) >= 2 or set(r["colors"]) & pickup_colors
        ]
    if len(candidates) < len(white):
        return

    # Restrict the combinatorial search if many decorative rings exist.
    if len(candidates) > 10:
        candidates = sorted(
            candidates,
            key=lambda r: (
                -len(r["colors"]),
                min(
                    math.dist(_center(groups[i], anchors), (r["x"], r["y"]))
                    for i in white
                ),
            ),
        )[:10]

    previous = memory.get("white_goals", {})
    best_cost = float("inf")
    best = None
    for choice in permutations(range(len(candidates)), len(white)):
        cost = 0.0
        for gi, ri in zip(white, choice):
            group = groups[gi]
            goal = candidates[ri]
            paint = set(group["paint"])
            wanted = set(goal["colors"])
            cost += 150 * len(paint - wanted)
            cost += _route_cost(
                _center(group, anchors), paint, goal, pickups
            )
            old = previous.get(str(gi))
            if old and (old["x"], old["y"]) != (goal["x"], goal["y"]):
                cost += 3.0
        if cost < best_cost:
            best_cost = cost
            best = choice

    if best is not None:
        memory["white_goals"] = {
            str(gi): {
                "x": candidates[ri]["x"],
                "y": candidates[ri]["y"],
                "colors": candidates[ri]["colors"][:],
            }
            for gi, ri in zip(white, best)
        }


def _tasks(memory):
    groups = memory["groups"]
    anchors = memory["anchors"]
    rings = memory["rings"]
    pickups = memory["pickups"]
    selected = memory.get("selected")
    tasks = []

    for gi, group in enumerate(groups):
        pos = _center(group, anchors)
        color = group["color"]

        if color != 4:
            candidates = [r for r in rings if r["colors"] == [color]]
            if not candidates:
                continue
            goal = min(
                candidates,
                key=lambda r: math.dist(pos, (r["x"], r["y"])),
            )
            if not _hit_goal(pos, (goal["x"], goal["y"])):
                tasks.append((gi, "goal", goal, {color}))
        else:
            goal = memory.get("white_goals", {}).get(str(gi))
            if goal is None:
                continue
            wanted = set(goal["colors"])
            paint = set(group["paint"])
            missing = wanted - paint
            if missing:
                choices = [
                    p for p in pickups
                    if not p["gone"] and p["color"] in missing
                ]
                if choices:
                    piece = min(
                        choices,
                        key=lambda p: math.dist(pos, (p["x"], p["y"])),
                    )
                    tasks.append((gi, "pickup", piece, wanted))
            elif paint == wanted and not _hit_goal(
                pos, (goal["x"], goal["y"])
            ):
                tasks.append((gi, "goal", goal, wanted))

    tasks.sort(key=lambda t: (
        selected not in groups[t[0]]["anchors"],
        math.dist(
            _center(groups[t[0]], anchors),
            (t[2]["x"], t[2]["y"]),
        ),
    ))
    return tasks


def _maps(f, danger, bad_centers):
    valid = [[False] * 64 for _ in range(64)]
    safe = [[False] * 64 for _ in range(64)]
    bad = [tuple(p) for p in bad_centers]

    for y in range(2, 62):
        for x in range(2, 62):
            valid[y][x] = all(
                int(f[y + dy, x + dx]) != 2
                for dx, dy in ANCHOR
            )
            safe[y][x] = (
                all((x + dx, y + dy) not in danger for dx, dy in BALL)
                and all(abs(x - bx) > 2 or abs(y - by) > 2 for bx, by in bad)
            )
    return valid, safe


def _distance_map(safe, target, kind):
    tx, ty = target
    hit = _hit_goal if kind == "goal" else _hit_ball
    dist = [[-1] * 64 for _ in range(64)]
    queue = deque()

    for y in range(2, 62):
        for x in range(2, 62):
            if safe[y][x] and hit((x, y), (tx, ty)):
                dist[y][x] = 0
                queue.append((x, y))

    directions = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    )
    while queue:
        x, y = queue.popleft()
        nd = dist[y][x] + 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                2 <= nx <= 61 and 2 <= ny <= 61
                and safe[ny][nx] and dist[ny][nx] < 0
            ):
                dist[ny][nx] = nd
                queue.append((nx, ny))
    return dist


def _select_point(index, anchors):
    a = anchors[index]
    for dx, dy in (
        (0, 0), (-2, -2), (2, -2), (-2, 2), (2, 2),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ):
        x, y = a["x"] + dx, a["y"] + dy
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        if all(
            j == index or abs(x - b["x"]) > 2 or abs(y - b["y"]) > 2
            for j, b in enumerate(anchors)
        ):
            return x, y
    return None


def _plan(memory, task, valid, safe):
    gi, kind, target, wanted = task
    anchors = memory["anchors"]
    group = memory["groups"][gi]
    ids = group["anchors"]
    if not ids:
        return None

    selected = memory.get("selected")
    tx, ty = target["x"], target["y"]
    current = _center(group, anchors)
    dist = _distance_map(safe, (tx, ty), kind)
    use_map = (
        0 <= current[0] < 64 and 0 <= current[1] < 64
        and dist[current[1]][current[0]] >= 0
    )
    total_x = sum(anchors[i]["x"] for i in ids)
    total_y = sum(anchors[i]["y"] for i in ids)
    n = len(ids)
    recent = set(memory.get("recent", []))
    best = None

    for i in ids:
        if i != selected and _select_point(i, anchors) is None:
            continue
        old = anchors[i]
        sx = total_x - old["x"]
        sy = total_y - old["y"]

        for y in range(2, 62):
            for x in range(2, 62):
                if not valid[y][x]:
                    continue
                if any(
                    abs(x - a["x"]) <= 2 and abs(y - a["y"]) <= 2
                    for a in anchors
                ):
                    continue

                bx, by = (sx + x) // n, (sy + y) // n
                if not (
                    2 <= bx <= 61 and 2 <= by <= 61 and safe[by][bx]
                ):
                    continue

                if group["color"] == 4:
                    if any(
                        not p["gone"]
                        and p["color"] not in wanted
                        and _hit_ball((bx, by), (p["x"], p["y"]))
                        for p in memory["pickups"]
                    ):
                        continue

                signature = ",".join(
                    str(v)
                    for j in ids
                    for v in (
                        (x, y) if j == i
                        else (anchors[j]["x"], anchors[j]["y"])
                    )
                )
                repeat_penalty = 15.0 if signature in recent else 0.0
                d = dist[by][bx]
                if use_map and d >= 0:
                    score = float(d)
                elif use_map:
                    score = 1000.0 + math.hypot(bx - tx, by - ty)
                else:
                    score = math.hypot(bx - tx, by - ty)

                score += 0.08 * math.hypot(bx - tx, by - ty)
                score += 0.8 if i != selected else 0.0
                score += repeat_penalty
                reached = (
                    _hit_goal((bx, by), (tx, ty))
                    if kind == "goal"
                    else _hit_ball((bx, by), (tx, ty))
                )
                if reached:
                    score -= 100.0

                candidate = (score, i, x, y, signature, bx, by)
                if best is None or candidate < best:
                    best = candidate
    return best


def _reset(memory):
    count = memory.get("reset_count", 0) + 1
    bad = memory.get("bad_centers", [])[-20:]
    memory.clear()
    memory["reset_count"] = count
    memory["bad_centers"] = bad
    return ("RESET", None, None)


def act(obs, available, memory):
    if not memory.get("initialized"):
        _initialize(obs, memory)
    else:
        reverted = _update_after_move(obs, memory)
        if reverted and memory.get("reset_count", 0) < 1:
            return _reset(memory)

    _update_rings(obs, memory)
    _update_paint(obs, memory)
    danger = _update_danger(obs, memory)
    _assign_white_goals(memory)
    tasks = _tasks(memory)

    if tasks:
        valid, safe = _maps(obs, danger, memory.get("bad_centers", []))
        for task in tasks:
            plan = _plan(memory, task, valid, safe)
            if plan is None:
                continue
            _, index, x, y, signature, bx, by = plan
            anchors = memory["anchors"]

            if memory.get("selected") != index:
                point = _select_point(index, anchors)
                if point is None:
                    continue
                memory["selected"] = index
                return ("ACTION6", int(point[1]), int(point[0]))

            old = (anchors[index]["x"], anchors[index]["y"])
            anchors[index]["x"] = x
            anchors[index]["y"] = y
            memory["pending"] = {
                "anchor": index,
                "old": list(old),
                "new": [x, y],
                "ball": [bx, by],
            }
            memory.setdefault("recent", []).append(signature)
            memory["recent"] = memory["recent"][-12:]
            return ("ACTION6", int(y), int(x))

    # A reset recovers from an unparseable perturbed state.
    memory["stuck"] = memory.get("stuck", 0) + 1
    if memory["stuck"] >= 2 and memory.get("reset_count", 0) < 1:
        return _reset(memory)

    if "ACTION6" in available:
        return ("ACTION6", 32, 32)
    if "RESET" in available:
        return ("RESET", None, None)
    return (available[0] if available else "ACTION6", None, None)

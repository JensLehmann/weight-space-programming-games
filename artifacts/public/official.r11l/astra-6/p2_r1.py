import math
import heapq
import itertools
import numpy as np

# All internal coordinates are (x, y).
_DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
            if abs(x) + abs(y) <= 2]
_DIAMOND_RING = [p for p in _DIAMOND if p != (0, 0)]
_BALL = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
         if not (abs(x) == 2 and abs(y) == 2)]
_RING = [(-1, -3), (-2, -2), (-3, -1),
         (-3, 1), (-2, 2), (-1, 3),
         (1, -3), (2, -2), (3, -1),
         (3, 1), (2, 2), (1, 3)]
_YY, _XX = np.indices((64, 64), dtype=np.int32)
_OBJECT_COLORS = {0, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15}


def _colors(pattern):
    return frozenset(int(c) for c in pattern if c > 0)


def _allowed(block, offsets):
    result = np.ones((64, 64), dtype=bool)
    for dx, dy in offsets:
        result &= ~np.roll(block, (-dy, -dx), axis=(0, 1))
    result[:2, :] = False
    result[62:, :] = False
    result[:, :2] = False
    result[:, 62:] = False
    return result


def _touch(x, y, obj, kind):
    dx = abs(x - obj["x"])
    dy = abs(y - obj["y"])
    widths = (4, 4, 4, 3, 2) if kind == "p" else (5, 5, 5, 4, 3, 2)
    return dy < len(widths) and dx <= widths[dy]


def _zone(obj, kind):
    dx = np.abs(_XX - obj["x"])
    dy = np.abs(_YY - obj["y"])
    widths = (4, 4, 4, 3, 2) if kind == "p" else (5, 5, 5, 4, 3, 2)
    result = np.zeros((64, 64), dtype=bool)
    for d, width in enumerate(widths):
        result |= (dy == d) & (dx <= width)
    return result


def _line_score(obs, x0, y0, x1, y1):
    ax, ay, bx, by = x0, y0, x1, y1
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    good = bad = 0
    while True:
        if (max(abs(x0 - ax), abs(y0 - ay)) > 2 and
                max(abs(x0 - bx), abs(y0 - by)) > 2):
            c = int(obs[y0, x0])
            if c == 1:
                good += 1
            elif c in (5, 10):
                bad += 1
        if x0 == x1 and y0 == y1:
            break
        e = 2 * err
        if e > -dy:
            err -= dy
            x0 += sx
        if e < dx:
            err += dx
            y0 += sy
    if good + bad == 0:
        return 0.55
    return (good + 0.25) / (good + bad + 0.25)


def _read_scene(obs):
    object_mask = np.isin(obs, list(_OBJECT_COLORS))
    hits = np.ones((64, 64), dtype=bool)
    for dx, dy in _BALL:
        hits &= np.roll(object_mask, (-dy, -dx), axis=(0, 1))
    hits[:2] = False
    hits[62:] = False
    hits[:, :2] = False
    hits[:, 62:] = False

    circles = []
    for y, x in zip(*np.where(hits)):
        x, y = int(x), int(y)
        pat = [int(obs[y + dy, x + dx]) for dx, dy in _BALL]
        if len(set(pat)) > 3:
            continue
        corners = sum(int(object_mask[y + dy, x + dx])
                      for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2)))
        if corners > 2:
            continue
        if any(max(abs(x - c["x"]), abs(y - c["y"])) <= 1
               for c in circles):
            continue
        circles.append({"x": x, "y": y, "pat": pat})

    nodes = []
    extras = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
              if abs(x) + abs(y) == 3]
    for border in (0, 3):
        nh = np.ones((64, 64), dtype=bool)
        for dx, dy in _DIAMOND_RING:
            nh &= np.roll(obs == border, (-dy, -dx), axis=(0, 1))
        nh[:2] = False
        nh[62:] = False
        nh[:, :2] = False
        nh[:, 62:] = False
        for y, x in zip(*np.where(nh)):
            x, y = int(x), int(y)
            label = int(obs[y, x])
            if label not in _OBJECT_COLORS or label == 0:
                continue
            if sum(int(obs[y + dy, x + dx] == border)
                   for dx, dy in extras) > 2:
                continue
            nodes.append({"x": x, "y": y, "label": label,
                          "border": border})

    if not nodes or not circles:
        return None
    nodes.sort(key=lambda p: (p["y"], p["x"]))
    selected = next((i for i, n in enumerate(nodes)
                     if n["border"] == 0), -1)
    if selected < 0:
        return None

    links = np.empty((len(nodes), len(circles)), dtype=float)
    for i, node in enumerate(nodes):
        for j, circle in enumerate(circles):
            links[i, j] = _line_score(
                obs, node["x"], node["y"], circle["x"], circle["y"])

    balls = []
    used_circles = set()
    labels = sorted(set(n["label"] for n in nodes if n["label"] != 4))

    for label in labels:
        ids = [i for i, node in enumerate(nodes)
               if node["label"] == label]
        mx = sum(nodes[i]["x"] for i in ids) // len(ids)
        my = sum(nodes[i]["y"] for i in ids) // len(ids)
        options = []
        for j, circle in enumerate(circles):
            if j in used_circles or label not in circle["pat"]:
                continue
            error = abs(circle["x"] - mx) + abs(circle["y"] - my)
            score = 5 * error + sum(1 - links[i, j] for i in ids)
            options.append((score, j, error))
        if not options:
            return None
        _, j, error = min(options)
        if error > 2 and len(ids) > 1:
            return None
        circle = circles[j]
        balls.append({
            "x": circle["x"], "y": circle["y"],
            "pat": circle["pat"][:], "nodes": ids,
            "mix": False, "label": label
        })
        used_circles.add(j)

    black_nodes = [i for i, node in enumerate(nodes)
                   if node["label"] == 4]
    if black_nodes:
        candidates = [j for j in range(len(circles))
                      if j not in used_circles]
        if not candidates:
            return None

        choices = []
        for i in black_nodes:
            ordered = sorted(candidates, key=lambda j: -links[i, j])
            best = links[i, ordered[0]]
            opts = [j for j in ordered[:3]
                    if links[i, j] >= best - 0.42]
            choices.append(opts or ordered[:1])

        count = math.prod(len(c) for c in choices)
        if count <= 70000:
            mappings = itertools.product(*choices)
        else:
            partial = [(0.0, ())]
            for i, opts in zip(black_nodes, choices):
                nxt = [(cost + 1 - links[i, j], seq + (j,))
                       for cost, seq in partial for j in opts]
                nxt.sort(key=lambda z: z[0])
                partial = nxt[:5000]
            mappings = (seq for _, seq in partial)

        best_cost = float("inf")
        best_groups = None
        for mapping in mappings:
            groups = {}
            cost = 0.0
            for i, j in zip(black_nodes, mapping):
                groups.setdefault(j, []).append(i)
                cost += 3.0 * (1.0 - links[i, j])
            for j, ids in groups.items():
                circle = circles[j]
                mx = sum(nodes[i]["x"] for i in ids) // len(ids)
                my = sum(nodes[i]["y"] for i in ids) // len(ids)
                error = abs(circle["x"] - mx) + abs(circle["y"] - my)
                if len(ids) == 1:
                    cost += 5.0 + min(25.0, 2.0 * error)
                else:
                    cost += 5.0 * error
            if cost < best_cost:
                best_cost, best_groups = cost, groups

        if not best_groups:
            return None
        for j, ids in best_groups.items():
            circle = circles[j]
            mx = sum(nodes[i]["x"] for i in ids) // len(ids)
            my = sum(nodes[i]["y"] for i in ids) // len(ids)
            if (len(ids) > 1 and
                    abs(circle["x"] - mx) + abs(circle["y"] - my) > 2):
                return None
            balls.append({
                "x": circle["x"], "y": circle["y"],
                "pat": circle["pat"][:], "nodes": ids,
                "mix": True, "label": 4
            })
            used_circles.add(j)

    if not balls:
        return None
    for bi, ball in enumerate(balls):
        for i in ball["nodes"]:
            nodes[i]["ball"] = bi
    if any("ball" not in node for node in nodes):
        return None

    pickups = []
    for j, circle in enumerate(circles):
        if j not in used_circles:
            pickups.append({
                "x": circle["x"], "y": circle["y"],
                "pat": circle["pat"][:], "alive": True
            })

    occluded = np.zeros((64, 64), dtype=bool)
    for circle in circles:
        for dx, dy in _BALL:
            occluded[circle["y"] + dy, circle["x"] + dx] = True
    for node in nodes:
        for dx, dy in _DIAMOND:
            occluded[node["y"] + dy, node["x"] + dx] = True

    goal_candidates = []
    for y in range(3, 61):
        for x in range(3, 61):
            visible = [k for k, (dx, dy) in enumerate(_RING)
                       if not occluded[y + dy, x + dx]]
            if len(visible) < 7:
                continue
            vals = {k: int(obs[y + _RING[k][1], x + _RING[k][0]])
                    for k in visible}
            if any(v not in _OBJECT_COLORS for v in vals.values()):
                continue
            found = None
            for axis in (0, 1):
                left = [vals[k] for k in visible if _RING[k][axis] < 0]
                right = [vals[k] for k in visible if _RING[k][axis] > 0]
                if len(left) < 2 or len(right) < 2:
                    continue
                if len(set(left)) == 1 and len(set(right)) == 1:
                    found = frozenset(left + right) - {0}
                    break
            if found is not None:
                goal_candidates.append((len(visible), x, y, found))

    goal_candidates.sort(reverse=True)
    goals = []
    for score, x, y, colors in goal_candidates:
        if any(max(abs(x - g["x"]), abs(y - g["y"])) <= 4
               for g in goals):
            continue
        if not colors:
            near = [b for b in balls
                    if _touch(b["x"], b["y"], {"x": x, "y": y}, "g")]
            if near:
                b = min(near, key=lambda b:
                        (b["x"] - x) ** 2 + (b["y"] - y) ** 2)
                colors = (_colors(b["pat"]) if b["mix"]
                          else frozenset([b["label"]]))
        goals.append({"x": x, "y": y, "colors": colors})

    if not goals:
        return None

    walls = (obs == 2).copy()
    walls[:, 0] = walls[:, 1]
    hazard = obs == 10
    around = np.zeros((64, 64), dtype=np.int32)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        around += np.roll(hazard, (dy, dx), axis=(0, 1))
    hazard = hazard | ((obs == 1) & (around >= 2))

    free = _allowed(walls, _DIAMOND)
    safe = _allowed(hazard, _BALL)
    fy, fx = np.where(free)
    return {
        "nodes": nodes, "balls": balls, "pickups": pickups,
        "goals": goals, "sel": selected, "free": free, "safe": safe,
        # Use wide arithmetic for coordinates, projections, and sentinels.
        "fx": fx.astype(np.int32), "fy": fy.astype(np.int32),
        "tasks": None, "plan": [], "bad": set(),
        "pending": None, "stalls": 0
    }


def _paint(pattern, pickup):
    return [q if q > 0 else p
            for p, q in zip(pattern, pickup["pat"])]


def _route_estimate(scene, bi, seq, gi):
    ball = scene["balls"][bi]
    x, y = ball["x"], ball["y"]
    n = len(ball["nodes"])
    cost = 0.0
    for pi in seq:
        pickup = scene["pickups"][pi]
        distance = max(
            0.0, math.hypot(pickup["x"] - x, pickup["y"] - y) - 4.0)
        cost += 0.8 + distance * n / 34.0
        x, y = pickup["x"], pickup["y"]
    goal = scene["goals"][gi]
    distance = max(
        0.0, math.hypot(goal["x"] - x, goal["y"] - y) - 5.0)
    cost += 0.6 + distance * n / 34.0
    if scene["sel"] in ball["nodes"]:
        cost -= 0.25
    return cost


def _paint_routes(scene, bi, gi):
    ball = scene["balls"][bi]
    goal = scene["goals"][gi]["colors"]
    if not goal:
        return []

    eligible = []
    for pi, pickup in enumerate(scene["pickups"]):
        cs = _colors(pickup["pat"])
        if pickup["alive"] and cs and cs <= goal:
            eligible.append(pi)
    found = []

    def visit(pattern, seq):
        if _colors(pattern) == goal:
            found.append((_route_estimate(scene, bi, seq, gi), tuple(seq)))
            return
        if len(seq) >= min(4, len(eligible)):
            return
        for pi in eligible:
            if pi in seq:
                continue
            changed = _paint(pattern, scene["pickups"][pi])
            if changed != pattern:
                visit(changed, seq + [pi])

    visit(ball["pat"], [])
    found.sort(key=lambda z: z[0])
    return found[:10]


def _make_tasks(scene):
    tasks_by_ball = []
    used_goals = set()
    mixed = []

    for bi, ball in enumerate(scene["balls"]):
        if ball["mix"]:
            mixed.append(bi)
            continue
        options = [gi for gi, goal in enumerate(scene["goals"])
                   if goal["colors"] == frozenset([ball["label"]])]
        if not options:
            return None
        gi = min(options, key=lambda j:
                 math.hypot(scene["goals"][j]["x"] - ball["x"],
                            scene["goals"][j]["y"] - ball["y"]))
        used_goals.add(gi)
        tasks_by_ball.append((bi, [("g", gi, bi)]))

    if mixed:
        all_options = {}
        possible_goals = []
        for gi, goal in enumerate(scene["goals"]):
            if gi in used_goals or not goal["colors"]:
                continue
            options = []
            for bi in mixed:
                for cost, seq in _paint_routes(scene, bi, gi):
                    options.append((cost, bi, seq))
            if options:
                options.sort(key=lambda z: z[0])
                all_options[gi] = options
                possible_goals.append(gi)

        possible_goals.sort(key=lambda j: (
            -len(scene["goals"][j]["colors"]),
            min(z[0] for z in all_options[j])))
        if not possible_goals:
            return None

        solution = None
        minimum = sum(len(scene["goals"][j]["colors"]) > 1
                      for j in possible_goals)
        minimum = max(1, min(minimum, len(mixed)))
        maximum = min(len(mixed), len(possible_goals))

        for number in range(maximum, minimum - 1, -1):
            chosen = possible_goals[:number]
            order = sorted(chosen, key=lambda j: len(all_options[j]))
            best = [float("inf"), None]

            def assign(k, used_balls, used_paint, cost, assignments):
                if cost >= best[0]:
                    return
                if k == len(order):
                    best[0] = cost
                    best[1] = assignments[:]
                    return
                gi = order[k]
                for c, bi, seq in all_options[gi]:
                    if bi in used_balls or any(p in used_paint for p in seq):
                        continue
                    assign(
                        k + 1, used_balls | {bi}, used_paint | set(seq),
                        cost + c, assignments + [(bi, gi, seq)])

            assign(0, set(), set(), 0.0, [])
            if best[1] is not None:
                solution = best[1]
                break

        if solution is None:
            return None
        for bi, gi, seq in solution:
            route = [("p", pi, bi) for pi in seq] + [("g", gi, bi)]
            tasks_by_ball.append((bi, route))

    tasks_by_ball.sort(key=lambda z: (
        0 if scene["sel"] in scene["balls"][z[0]]["nodes"] else 1,
        len(z[1])))
    return [task for _, route in tasks_by_ball for task in route]


def _dilate_square(a, r):
    k = 2 * r + 1
    p = np.pad(a.astype(np.int32), ((r, r), (r, r)))
    c = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] -
            c[k:, :-k] + c[:-k, :-k]) > 0


def _plan_to(scene, task):
    kind, ti, bi = task
    ball = scene["balls"][bi]
    ids = ball["nodes"]
    n = len(ids)
    target = scene["pickups"][ti] if kind == "p" else scene["goals"][ti]

    valid_ball = scene["safe"].copy()
    for pi, pickup in enumerate(scene["pickups"]):
        if pickup["alive"] and not (kind == "p" and pi == ti):
            valid_ball &= ~_zone(pickup, "p")
    target_mask = valid_ball & _zone(target, kind)
    if not target_mask.any():
        return None

    radius = max(1, int(math.ceil(63.0 / n)))
    jump = np.full((64, 64), 40, dtype=np.int32)
    reach = target_mask.copy()
    jump[reach] = 0
    for d in range(1, 13):
        newer = _dilate_square(reach, radius) & valid_ball
        new = newer & ~reach
        if not new.any():
            break
        jump[new] = d
        reach |= newer

    distance = np.hypot(_XX - target["x"], _YY - target["y"])
    hmap = 1.8 * jump + distance / (2.2 * radius)
    hmap[target_mask] = 0.0

    fx, fy = scene["fx"], scene["fy"]
    root_pos = tuple(
        scene["nodes"][i]["x"] + 64 * scene["nodes"][i]["y"]
        for i in ids)
    selected = ids.index(scene["sel"]) if scene["sel"] in ids else -1
    foreign = [(q["x"], q["y"])
               for i, q in enumerate(scene["nodes"]) if i not in ids]

    base_free = np.ones(len(fx), dtype=bool)
    for x, y in foreign:
        base_free &= (np.abs(fx - x) > 4) | (np.abs(fy - y) > 4)

    def center(pos):
        return (sum(p % 64 for p in pos) // n,
                sum(p // 64 for p in pos) // n)

    rx, ry = center(root_pos)
    root_h = float(hmap[ry, rx])
    heap = [(root_h, 0, 0, root_pos, selected, ())]
    visited = {(root_pos, selected): 0}
    serial = 0
    best_partial = (root_h, ())
    expansions = 0
    limit = 310 if n <= 4 else 200

    while heap and expansions < limit:
        _, cost, _, pos, sel, path = heapq.heappop(heap)
        if visited.get((pos, sel), 10**9) != cost:
            continue
        expansions += 1
        xs = [p % 64 for p in pos]
        ys = [p // 64 for p in pos]
        sx, sy = sum(xs), sum(ys)
        cx, cy = sx // n, sy // n

        if path and target_mask[cy, cx]:
            return list(path)
        if len(path) >= 11:
            continue

        order = list(range(n))
        if sel >= 0:
            order.remove(sel)
            order.insert(0, sel)

        for local in order:
            nid = ids[local]
            ok = base_free.copy()
            for j in range(n):
                margin = 2 if j == local else 4
                ok &= ((np.abs(fx - xs[j]) > margin) |
                       (np.abs(fy - ys[j]) > margin))
            for bad_id, x, y in scene["bad"]:
                if bad_id == nid:
                    ok &= (fx != x) | (fy != y)

            bx = (sx - xs[local] + fx) // n
            by = (sy - ys[local] + fy) // n
            ok &= valid_ball[by, bx]
            ok &= (bx != cx) | (by != cy)
            if not ok.any():
                continue

            step_cost = 1 if local == sel else 2
            next_cost = cost + step_cost
            if next_cost > 30:
                continue

            wins = np.flatnonzero(ok & target_mask[by, bx])
            if len(wins):
                # Explicitly use floating point for ranking arithmetic.
                dx = fx[wins].astype(float) - xs[local]
                dy = fy[wins].astype(float) - ys[local]
                merit = distance[by[wins], bx[wins]] + 0.002 * (dx*dx + dy*dy)
                q = int(wins[int(np.argmin(merit))])
                return list(path) + [(nid, int(fx[q]), int(fy[q]))]

            scores = np.where(ok, hmap[by, bx], np.inf)
            working = scores.copy()
            chosen = []
            for _ in range(9):
                q = int(np.argmin(working))
                if not np.isfinite(working[q]):
                    break
                chosen.append(q)
                close = ((np.abs(bx - bx[q]) <= 2) &
                         (np.abs(by - by[q]) <= 2))
                working[close] = np.inf

            # Floating-point infinity avoids narrow-integer sentinel overflow.
            for projection in (fx, fy, fx + fy, fx - fy):
                projection = projection.astype(float)
                for sign in (-1.0, 1.0):
                    metric = np.where(ok, sign * projection, np.inf)
                    q = int(np.argmin(metric))
                    if q not in chosen:
                        chosen.append(q)

            for q in chosen:
                endpoint = int(fx[q]) + 64 * int(fy[q])
                new_pos = pos[:local] + (endpoint,) + pos[local + 1:]
                key = (new_pos, local)
                if visited.get(key, 10**9) <= next_cost:
                    continue
                visited[key] = next_cost
                move = (nid, int(fx[q]), int(fy[q]))
                new_path = path + (move,)
                h = float(hmap[int(by[q]), int(bx[q])])
                if h < best_partial[0] - 0.05:
                    best_partial = (h, new_path)
                serial += 1
                priority = next_cost + h + 0.0001 * len(new_path)
                heapq.heappush(
                    heap, (priority, next_cost, serial,
                           new_pos, local, new_path))

    if best_partial[1] and best_partial[0] < root_h - 0.1:
        return list(best_partial[1])
    return None


def _satisfied(scene, task):
    kind, ti, bi = task
    ball = scene["balls"][bi]
    if kind == "p":
        return not scene["pickups"][ti]["alive"]
    goal = scene["goals"][ti]
    return (_touch(ball["x"], ball["y"], goal, "g") and
            (not ball["mix"] or _colors(ball["pat"]) == goal["colors"]))


def _selection_point(scene, nid):
    node = scene["nodes"][nid]
    positions = [(node["x"], node["y"])]
    positions += [(node["x"] + dx, node["y"] + dy)
                  for dy in range(-2, 3) for dx in range(-2, 3)]
    for x, y in positions:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        if all(i == nid or abs(x - q["x"]) > 2 or abs(y - q["y"]) > 2
               for i, q in enumerate(scene["nodes"])):
            return x, y
    return node["x"], node["y"]


def _pattern_score(obs, x, y, pattern):
    good = total = 0
    for (dx, dy), color in zip(_BALL, pattern):
        xx, yy = x + dx, y + dy
        if 0 <= xx < 64 and 0 <= yy < 64:
            seen = int(obs[yy, xx])
            if seen == 2:
                continue
            total += 1
            good += int(seen == color)
    return good / max(1, total)


def _commit_move(scene, nid, x, y):
    old_nodes = [(p["x"], p["y"]) for p in scene["nodes"]]
    old_balls = [(b["x"], b["y"], b["pat"][:]) for b in scene["balls"]]
    old_alive = [p["alive"] for p in scene["pickups"]]
    bi = scene["nodes"][nid]["ball"]

    scene["nodes"][nid]["x"] = x
    scene["nodes"][nid]["y"] = y
    ball = scene["balls"][bi]
    ids = ball["nodes"]
    ball["x"] = sum(scene["nodes"][i]["x"] for i in ids) // len(ids)
    ball["y"] = sum(scene["nodes"][i]["y"] for i in ids) // len(ids)

    for other in scene["balls"]:
        if not other["mix"]:
            continue
        for pickup in scene["pickups"]:
            if pickup["alive"] and _touch(other["x"], other["y"], pickup, "p"):
                other["pat"] = _paint(other["pat"], pickup)
                pickup["alive"] = False

    scene["pending"] = {
        "bi": bi, "nid": nid, "dest": (x, y),
        "nodes": old_nodes, "balls": old_balls, "alive": old_alive
    }


def _verify(scene, obs):
    pending = scene.get("pending")
    if pending is None:
        return
    scene["pending"] = None
    bi = pending["bi"]
    ball = scene["balls"][bi]
    ox, oy, old_pattern = pending["balls"][bi]
    new_score = _pattern_score(obs, ball["x"], ball["y"], ball["pat"])
    old_score = _pattern_score(obs, ox, oy, old_pattern)

    if new_score < 0.65 and old_score > new_score + 0.20:
        for node, (x, y) in zip(scene["nodes"], pending["nodes"]):
            node["x"], node["y"] = x, y
        for b, (x, y, pat) in zip(scene["balls"], pending["balls"]):
            b["x"], b["y"], b["pat"] = x, y, pat
        for pickup, alive in zip(scene["pickups"], pending["alive"]):
            pickup["alive"] = alive
        x, y = pending["dest"]
        scene["bad"].add((pending["nid"], x, y))
        scene["plan"] = []
        scene["tasks"] = None
        scene["stalls"] += 1


def _reset(memory):
    retries = memory.get("_retries", 0) + 1
    memory.clear()
    memory["_retries"] = retries
    return "RESET", None, None


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=np.int32)

    if "scene" not in memory:
        scene = _read_scene(obs)
        if scene is None:
            return _reset(memory)
        memory["scene"] = scene
    else:
        scene = memory["scene"]
        _verify(scene, obs)

    if scene["stalls"] >= 4:
        return _reset(memory)

    if scene["tasks"] is None:
        scene["tasks"] = _make_tasks(scene)
        scene["plan"] = []
        if scene["tasks"] is None:
            return _reset(memory)

    while scene["tasks"] and _satisfied(scene, scene["tasks"][0]):
        scene["tasks"].pop(0)
        scene["plan"] = []

    if not scene["tasks"]:
        # A pre-satisfied starting state may need a move to run the win check.
        scene["stalls"] += 1
        nid = scene["sel"]
        node = scene["nodes"][nid]
        bi = node["ball"]
        ball = scene["balls"][bi]
        for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):
            x, y = node["x"] + dx, node["y"] + dy
            if not (2 <= x < 62 and 2 <= y < 62 and scene["free"][y, x]):
                continue
            if any(abs(x - q["x"]) <= 2 and abs(y - q["y"]) <= 2
                   for q in scene["nodes"]):
                continue
            ids = ball["nodes"]
            bx = (sum(scene["nodes"][i]["x"] for i in ids) + dx) // len(ids)
            by = (sum(scene["nodes"][i]["y"] for i in ids) + dy) // len(ids)
            if not scene["safe"][by, bx]:
                continue
            matching = [
                g for g in scene["goals"]
                if ((not ball["mix"] and g["colors"] == {ball["label"]}) or
                    (ball["mix"] and g["colors"] == _colors(ball["pat"])))
            ]
            if matching and not any(_touch(bx, by, g, "g") for g in matching):
                continue
            _commit_move(scene, nid, x, y)
            return "ACTION6", int(y), int(x)
        return _reset(memory)

    task = scene["tasks"][0]
    kind, ti, bi = task
    ball = scene["balls"][bi]

    if (kind == "g" and ball["mix"] and
            _colors(ball["pat"]) != scene["goals"][ti]["colors"]):
        scene["tasks"] = _make_tasks(scene)
        scene["plan"] = []
        if not scene["tasks"]:
            return _reset(memory)
        while scene["tasks"] and _satisfied(scene, scene["tasks"][0]):
            scene["tasks"].pop(0)
        if not scene["tasks"]:
            return _reset(memory)
        task = scene["tasks"][0]

    if not scene["plan"]:
        plan = _plan_to(scene, task)
        if not plan:
            scene["stalls"] += 1
            fresh = _read_scene(obs)
            if fresh is not None and not memory.get("_refreshed", False):
                memory["scene"] = fresh
                memory["_refreshed"] = True
                return act(obs, available, memory)
            return _reset(memory)
        scene["plan"] = plan

    nid, x, y = scene["plan"][0]
    if scene["sel"] != nid:
        sx, sy = _selection_point(scene, nid)
        scene["sel"] = nid
        return "ACTION6", int(sy), int(sx)

    scene["plan"].pop(0)
    _commit_move(scene, nid, x, y)
    return "ACTION6", int(y), int(x)

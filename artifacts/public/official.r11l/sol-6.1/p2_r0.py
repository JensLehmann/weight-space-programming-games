import math
import itertools
import numpy as np

# Coordinates in the program are (x, y).
_DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
            if abs(x) + abs(y) <= 2]
_NODE_EDGE = [p for p in _DIAMOND if p != (0, 0)]
_BALL = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
         if not (abs(x) == 2 and abs(y) == 2)]
_RING_BODY = [(x, y) for y in range(-3, 4) for x in range(-3, 4)
              if abs(x) <= (1 if abs(y) == 3 else
                            2 if abs(y) == 2 else 3)]
_RING_LEFT = [(-1, -3), (-2, -2), (-3, -1),
              (-3, 1), (-2, 2), (-1, 3)]
_RING_RIGHT = [(1, -3), (2, -2), (3, -1),
               (3, 1), (2, 2), (1, 3)]
_YY, _XX = np.indices((64, 64))


def _line(x0, y0, x1, y1):
    out = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return out
        e = 2 * err
        if e > -dy:
            err -= dy
            x0 += sx
        if e < dx:
            err += dx
            y0 += sy


def _line_score(obs, a, b):
    good = total = 0
    for x, y in _line(a[0], a[1], b[0], b[1]):
        if max(abs(x - a[0]), abs(y - a[1])) <= 2:
            continue
        if max(abs(x - b[0]), abs(y - b[1])) <= 2:
            continue
        if 0 <= x < 64 and 0 <= y < 64:
            v = int(obs[y, x])
            if v in (1, 5, 10):
                total += 1
                good += v == 1
    return good / max(1, total), good


def _collision_offsets(other_mask):
    return set((ox - bx, oy - by)
               for bx, by in _BALL for ox, oy in other_mask)


_PICK_REL = _collision_offsets(_BALL)
_GOAL_REL = _collision_offsets(_RING_BODY)


def _touch(a, b, ring=False):
    return (a[0] - b[0], a[1] - b[1]) in (
        _GOAL_REL if ring else _PICK_REL
    )


def _stamp(center, ring=False):
    z = np.zeros((64, 64), dtype=bool)
    rel = _GOAL_REL if ring else _PICK_REL
    for dx, dy in rel:
        x, y = center[0] + dx, center[1] + dy
        if 0 <= x < 64 and 0 <= y < 64:
            z[y, x] = True
    return z


def _blocked_centers(mask, footprint):
    z = np.zeros((64, 64), dtype=bool)
    for dx, dy in footprint:
        x0, x1 = max(0, -dx), min(64, 64 - dx)
        y0, y1 = max(0, -dy), min(64, 64 - dy)
        z[y0:y1, x0:x1] |= mask[y0 + dy:y1 + dy,
                                     x0 + dx:x1 + dx]
    return z


def _colors(pixels):
    return frozenset(int(v) for v in pixels if v > 0)


def _paint(pixels, pickup):
    q = list(pixels)
    for i, v in enumerate(pickup["pix"]):
        if v > 0:
            q[i] = int(v)
    return q


def _mean_position(world, ball_id, positions=None):
    ids = world["balls"][ball_id]["nodes"]
    if positions is None:
        positions = [n["p"] for n in world["nodes"]]
    n = len(ids)
    return (sum(positions[i][0] for i in ids) // n,
            sum(positions[i][1] for i in ids) // n)


def _parse(obs):
    nodes = []
    excluded_centers = {0, 1, 2, 3, 5, 10}
    for y in range(64):
        for x in range(1, 64):
            c = int(obs[y, x])
            if c in excluded_centers:
                continue
            vals = [int(obs[y + dy, x + dx])
                    for dx, dy in _NODE_EDGE
                    if 0 < x + dx < 64 and 0 <= y + dy < 64]
            if len(vals) >= 8 and (
                all(v == 0 for v in vals) or
                all(v == 3 for v in vals)
            ):
                nodes.append({"p": (x, y), "color": c,
                              "selected": vals[0] == 0})
    nodes.sort(key=lambda n: ((n["p"][0] - 2) ** 2 +
                             (n["p"][1] - 2) ** 2))

    rounded = []
    forbidden = {1, 2, 3, 5, 10}
    for y in range(2, 62):
        for x in range(2, 62):
            vals = [int(obs[y + dy, x + dx]) for dx, dy in _BALL]
            if any(v in forbidden for v in vals):
                continue
            pix = [-1] * 25
            for (dx, dy), v in zip(_BALL, vals):
                pix[(dy + 2) * 5 + dx + 2] = v
            rounded.append({"p": (x, y), "pix": pix,
                            "normal": int(obs[y, x]) == 6,
                            "nodes": []})

    for ni, node in enumerate(nodes):
        candidates = []
        for ri, obj in enumerate(rounded):
            if node["color"] == 4:
                if obj["normal"]:
                    continue
            else:
                outer = [v for j, v in enumerate(obj["pix"])
                         if v >= 0 and j != 12]
                if not obj["normal"] or node["color"] not in outer:
                    continue
            ratio, count = _line_score(obs, node["p"], obj["p"])
            distance = math.dist(node["p"], obj["p"])
            primary = obj["normal"] or all(
                v in (-1, 0) for v in obj["pix"])
            score = ratio * 100 + min(count, 15) * .1 - distance * .005
            if primary:
                score += .15
            candidates.append((score, ri))
        if candidates:
            ri = max(candidates)[1]
            rounded[ri]["nodes"].append(ni)

    balls, pickups = [], []
    for obj in rounded:
        if obj["nodes"]:
            obj["black"] = all(nodes[i]["color"] == 4
                               for i in obj["nodes"])
            obj["locked"] = False
            if not obj["black"]:
                obj["color"] = nodes[obj["nodes"][0]]["color"]
            balls.append(obj)
        elif not obj["normal"] and _colors(obj["pix"]):
            pickups.append({"p": obj["p"], "pix": obj["pix"]})

    rings = []
    for y in range(3, 61):
        for x in range(3, 61):
            left = [int(obs[y + dy, x + dx]) for dx, dy in _RING_LEFT]
            right = [int(obs[y + dy, x + dx]) for dx, dy in _RING_RIGHT]
            if len(set(left)) != 1 or len(set(right)) != 1:
                continue
            if left[0] in (1, 2, 3, 5, 6, 10, 12):
                continue
            if right[0] in (1, 2, 3, 5, 6, 10, 12):
                continue
            cs = frozenset(v for v in (left[0], right[0]) if v > 0)
            rings.append({"p": (x, y), "colors": cs})

    # A completed target is covered by a black outline.
    goals = []
    for ring in rings:
        if not ring["colors"]:
            for ball in balls:
                if _touch(ball["p"], ring["p"], True):
                    ball["locked"] = True
        else:
            goals.append(ring)

    normal_colors = {b.get("color") for b in balls if not b["black"]}
    paint_colors = set()
    for pickup in pickups:
        paint_colors.update(_colors(pickup["pix"]))
    for b in balls:
        if b["black"]:
            paint_colors.update(_colors(b["pix"]))

    eligible = []
    for g in goals:
        if len(g["colors"]) == 1 and next(iter(g["colors"])) in normal_colors:
            g["kind"] = "normal"
            eligible.append(g)
        elif g["colors"] <= paint_colors and any(b["black"] for b in balls):
            g["kind"] = "black"
            eligible.append(g)

    # Single-colour decorative rings are subsets of the real mixed targets.
    goals = [g for g in eligible
             if g["kind"] != "black" or not any(
                 g["colors"] < h["colors"] for h in eligible
                 if h["kind"] == "black")]

    wall = obs == 2
    wall[:, :2] = True
    bad_node = _blocked_centers(wall, _DIAMOND)
    valid = ~bad_node
    valid[:2, :] = valid[62:, :] = False
    valid[:, :2] = valid[:, 62:] = False
    dy, dx = np.where(valid)
    destinations = np.column_stack((dx, dy)).astype(np.int16)

    hazard = obs == 10
    # Interface lines hide some orange pixels.
    near = np.zeros_like(hazard)
    near[1:] |= hazard[:-1]
    near[:-1] |= hazard[1:]
    near[:, 1:] |= hazard[:, :-1]
    near[:, :-1] |= hazard[:, 1:]
    hazard |= (obs == 1) & near

    selected = next((i for i, n in enumerate(nodes) if n["selected"]), 0)
    return {"nodes": nodes, "balls": balls, "pickups": pickups,
            "goals": goals, "dest": destinations,
            "unsafe": _blocked_centers(hazard, _BALL),
            "selected": selected}


def _route(world, bi, goal):
    b = world["balls"][bi]
    wanted = goal["colors"]
    if _colors(b["pix"]) == wanted:
        return [], max(0, math.dist(b["p"], goal["p"]) - 5)

    pickups = world["pickups"]
    best = None
    sequences = [(i,) for i in range(len(pickups))]
    sequences += list(itertools.permutations(range(len(pickups)), 2))
    for seq in sequences:
        pix = b["pix"]
        for i in seq:
            pix = _paint(pix, pickups[i])
        if _colors(pix) != wanted:
            continue
        points = [b["p"]] + [pickups[i]["p"] for i in seq] + [goal["p"]]
        cost = sum(max(0, math.dist(a, c) - 4)
                   for a, c in zip(points, points[1:]))
        cost += 20 * len(seq)
        if best is None or cost < best[1]:
            best = (list(seq), cost)
    return best


def _tasks(world):
    tasks = []
    balls = world["balls"]
    for bi, b in enumerate(balls):
        if b["black"] or b["locked"]:
            continue
        choices = [g for g in world["goals"]
                   if g["kind"] == "normal" and
                   g["colors"] == frozenset([b["color"]])]
        if choices:
            goal = min(choices, key=lambda g: math.dist(b["p"], g["p"]))
            if not _touch(b["p"], goal["p"], True):
                tasks.append({"ball": bi, "p": goal["p"], "ring": True,
                              "pickup": None,
                              "cost": math.dist(b["p"], goal["p"]) *
                                      len(b["nodes"])})

    black_ids = [i for i, b in enumerate(balls)
                 if b["black"] and not b["locked"]]
    black_goals = [g for g in world["goals"] if g["kind"] == "black"]
    if black_ids and black_goals:
        count = min(len(black_ids), len(black_goals))
        best_assignment = None
        # The game's small object count makes explicit matching practical.
        for ids in itertools.permutations(black_ids, count):
            routes = []
            total = 0
            for bi, goal in zip(ids, black_goals[:count]):
                route = _route(world, bi, goal)
                if route is None:
                    total = float("inf")
                    break
                routes.append((bi, goal, route[0]))
                total += route[1] * max(1, len(balls[bi]["nodes"]) - .5)
            if best_assignment is None or total < best_assignment[0]:
                best_assignment = (total, routes)
        if best_assignment and math.isfinite(best_assignment[0]):
            for bi, goal, seq in best_assignment[1]:
                b = balls[bi]
                if seq:
                    pi = seq[0]
                    p = world["pickups"][pi]["p"]
                    tasks.append({"ball": bi, "p": p, "ring": False,
                                  "pickup": pi,
                                  "cost": math.dist(b["p"], p) *
                                          len(b["nodes"]) + 8})
                elif not _touch(b["p"], goal["p"], True):
                    tasks.append({"ball": bi, "p": goal["p"], "ring": True,
                                  "pickup": None,
                                  "cost": math.dist(b["p"], goal["p"]) *
                                          len(b["nodes"])})
    return sorted(tasks, key=lambda t: t["cost"])


def _plan(world, task):
    bi = task["ball"]
    b = world["balls"][bi]
    ids = b["nodes"]
    n = len(ids)
    if not n or len(world["dest"]) == 0:
        return None

    goal_mask = _stamp(task["p"], task["ring"])
    gy, gx = np.where(goal_mask)
    dist = np.full((64, 64), 10000.0)
    for x, y in zip(gx, gy):
        dist = np.minimum(dist, (_XX - x) ** 2 + (_YY - y) ** 2)

    unsafe = world["unsafe"].copy()
    if b["black"]:
        for pi, pickup in enumerate(world["pickups"]):
            if pi != task["pickup"]:
                unsafe |= _stamp(pickup["p"])

    initial = tuple(tuple(world["nodes"][i]["p"]) for i in ids)
    others = [tuple(v["p"]) for i, v in enumerate(world["nodes"])
              if i not in ids]
    dest = world["dest"]
    beam = [(initial, [], 0, world["selected"])]
    seen = {initial: 0}
    best_found = None

    for depth in range(1, 6):
        expanded = []
        for positions, path, cost, selected in beam:
            sx = sum(p[0] for p in positions)
            sy = sum(p[1] for p in positions)
            for local, node_id in enumerate(ids):
                old = positions[local]
                bx = (sx - old[0] + dest[:, 0]) // n
                by = (sy - old[1] + dest[:, 1]) // n
                allowed = ~unsafe[by, bx]
                for p in list(positions) + others:
                    allowed &= ((np.abs(dest[:, 0] - p[0]) > 2) |
                                (np.abs(dest[:, 1] - p[1]) > 2))
                indexes = np.flatnonzero(allowed)
                if not len(indexes):
                    continue

                move_cost = 1 + (selected != node_id)
                new_cost = cost + move_cost
                values = dist[by[indexes], bx[indexes]]
                hits = indexes[values == 0]
                if len(hits):
                    # Prefer a destination near the target, but retain clickability.
                    h = min(hits, key=lambda k:
                            (int(dest[k, 0]) - task["p"][0]) ** 2 +
                            (int(dest[k, 1]) - task["p"][1]) ** 2)
                    answer = path + [(node_id, int(dest[h, 0]), int(dest[h, 1]))]
                    if best_found is None or new_cost < best_found[0]:
                        best_found = (new_cost, answer)

                take = min(len(indexes), max(50, n * n * 12))
                if take < len(indexes):
                    subset = np.argpartition(values, take - 1)[:take]
                else:
                    subset = np.arange(len(indexes))
                subset = subset[np.argsort(values[subset], kind="stable")]

                used_centers = set()
                kept = 0
                for si in subset:
                    k = indexes[si]
                    center = (int(bx[k]), int(by[k]))
                    if center in used_centers:
                        continue
                    used_centers.add(center)
                    pp = list(positions)
                    pp[local] = (int(dest[k, 0]), int(dest[k, 1]))
                    pp = tuple(pp)
                    if seen.get(pp, 1000) <= new_cost:
                        continue
                    seen[pp] = new_cost
                    new_path = path + [(node_id, pp[local][0], pp[local][1])]
                    heuristic = math.sqrt(float(dist[center[1], center[0]]))
                    rank = heuristic + 3.0 * new_cost
                    expanded.append((rank, pp, new_path, new_cost, node_id))
                    kept += 1
                    if kept >= 9:
                        break

        if best_found is not None:
            return best_found[1]
        if not expanded:
            break
        expanded.sort(key=lambda q: q[0])
        # Diversify the beam by both ball position and last handle.
        beam = []
        bins = {}
        for _, pp, path, cost, selected in expanded:
            center = (sum(p[0] for p in pp) // n,
                      sum(p[1] for p in pp) // n)
            key = (center, selected)
            if bins.get(key, 0) >= 2:
                continue
            bins[key] = bins.get(key, 0) + 1
            beam.append((pp, path, cost, selected))
            if len(beam) >= 18:
                break
    return None


def _node_visible(obs, p):
    vals = [int(obs[p[1] + dy, p[0] + dx])
            for dx, dy in _NODE_EDGE
            if 0 <= p[0] + dx < 64 and 0 <= p[1] + dy < 64]
    return len(vals) >= 8 and all(v in (0, 3) for v in vals)


def _select_point(world, node_id):
    p = world["nodes"][node_id]["p"]
    candidates = [p] + [(p[0] + dx, p[1] + dy)
                        for dy in range(-2, 3) for dx in range(-2, 3)]
    for x, y in candidates:
        if not (0 <= x < 64 and 0 <= y < 64):
            continue
        if not any(abs(x - n["p"][0]) <= 2 and
                   abs(y - n["p"][1]) <= 2
                   for n in world["nodes"][:node_id]):
            return x, y
    return p


def act(obs, available, memory):
    obs = np.asarray(obs)
    if "world" not in memory:
        memory["world"] = _parse(obs)
        memory["queue"] = []
        memory["failures"] = 0

    world = memory["world"]

    pending = memory.pop("pending", None)
    if pending is not None:
        node_id, old, new, bi, new_ball = pending
        if not _node_visible(obs, new):
            world["nodes"][node_id]["p"] = old
            world["balls"][bi]["p"] = _mean_position(world, bi)
            world["unsafe"][new_ball[1], new_ball[0]] = True
            memory["queue"] = []
            memory["failures"] += 1

    if not world["nodes"] or not world["balls"]:
        memory["world"] = world = _parse(obs)
        memory["queue"] = []
        if not world["nodes"]:
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION6", 32, 32

    if not memory.get("queue"):
        for task in _tasks(world)[:3]:
            path = _plan(world, task)
            if path:
                memory["queue"] = path
                break

    if not memory.get("queue"):
        # Re-observe after an unexpected layout or a failed reconstruction.
        memory["failures"] = memory.get("failures", 0) + 1
        if memory["failures"] in (2, 4):
            memory["world"] = _parse(obs)
        if memory["failures"] >= 5 and "RESET" in available:
            memory.clear()
            return "RESET", None, None
        p = world["nodes"][world["selected"]]["p"]
        return "ACTION6", int(p[1]), int(p[0])

    node_id, x, y = memory["queue"][0]
    if world["selected"] != node_id:
        cx, cy = _select_point(world, node_id)
        world["selected"] = node_id
        return "ACTION6", int(cy), int(cx)

    memory["queue"].pop(0)
    old = world["nodes"][node_id]["p"]
    world["nodes"][node_id]["p"] = (x, y)
    bi = next(i for i, ball in enumerate(world["balls"])
              if node_id in ball["nodes"])
    ball = world["balls"][bi]
    ball["p"] = _mean_position(world, bi)

    # Paint collection is simultaneous for all black balls.
    remaining = []
    for pickup in world["pickups"]:
        collector = next((b for b in world["balls"]
                          if b["black"] and _touch(b["p"], pickup["p"])), None)
        if collector is None:
            remaining.append(pickup)
        else:
            collector["pix"] = _paint(collector["pix"], pickup)
    world["pickups"] = remaining

    memory["pending"] = (node_id, old, (x, y), bi, ball["p"])
    memory["failures"] = 0
    return "ACTION6", int(y), int(x)

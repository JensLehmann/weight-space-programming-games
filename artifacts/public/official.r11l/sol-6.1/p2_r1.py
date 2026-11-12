import math
import time
import itertools
import numpy as np

# Internal coordinates are (x, y).
_BODY = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
         if not (abs(x) == 2 and abs(y) == 2)]
_DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
            if abs(x) + abs(y) <= 2]
_RIM = [p for p in _DIAMOND if p != (0, 0)]
_OUTSIDE = [p for p in _BODY if p not in _DIAMOND]
_LEFT = [(-1, -2), (-2, -1), (-1, -1), (-2, 0), (-1, 0),
         (-2, 1), (-1, 1), (0, 1), (-1, 2), (0, 2)]
_RIGHT = [p for p in _BODY if p not in _LEFT]
_RING_L = [(-1, -3), (-2, -2), (-3, -1),
           (-3, 1), (-2, 2), (-1, 3)]
_RING_R = [(-x, y) for x, y in _RING_L]
_GOAL_BODY = [(x, y) for y in range(-3, 4) for x in range(-3, 4)
              if abs(x) + abs(y) <= 4]
_PICK_CONTACT = set((a[0] - b[0], a[1] - b[1])
                    for a in _BODY for b in _BODY)
_GOAL_CONTACT = set((a[0] - b[0], a[1] - b[1])
                    for a in _GOAL_BODY for b in _BODY)


def _values(obs, x, y, offsets):
    return [int(obs[y + dy, x + dx]) for dx, dy in offsets]


def _controls(obs):
    result = []
    for y in range(2, 62):
        for x in range(2, 62):
            c = int(obs[y - 2, x])
            if c not in (0, 3):
                continue
            if all(obs[y + dy, x + dx] == c for dx, dy in _RIM):
                outside = _values(obs, x, y, _OUTSIDE)
                if sum(v == c for v in outside) <= 3:
                    center = int(obs[y, x])
                    if center not in (1, 2, 5, 10):
                        result.append({"p": (x, y), "color": center,
                                       "selected": c == 0})
    return result


def _rounds(obs):
    result = []
    usable = set(range(16)) - {1, 2, 3, 4, 5, 10}
    noncenter = [p for p in _BODY if p != (0, 0)]
    for y in range(2, 62):
        for x in range(2, 62):
            center = int(obs[y, x])
            if center == 6:
                vals = _values(obs, x, y, noncenter)
                if vals[0] in usable - {0, 6} and len(set(vals)) == 1:
                    result.append({"p": (x, y), "native": vals[0],
                                   "lr": None})
                    continue
            l = _values(obs, x, y, _LEFT)
            r = _values(obs, x, y, _RIGHT)
            if (len(set(l)) == 1 and len(set(r)) == 1 and
                    l[0] in usable and r[0] in usable):
                result.append({"p": (x, y), "native": None,
                               "lr": (l[0], r[0])})
    return result


def _rings(obs):
    result = []
    usable = set(range(16)) - {1, 2, 3, 4, 5, 10}
    for y in range(3, 61):
        for x in range(3, 61):
            l = _values(obs, x, y, _RING_L)
            r = _values(obs, x, y, _RING_R)
            if (len(set(l)) == 1 and len(set(r)) == 1 and
                    l[0] in usable and r[0] in usable):
                colors = frozenset(v for v in (l[0], r[0]) if v > 0)
                result.append({"p": (x, y), "colors": colors})
    return result


def _line_score(obs, a, b):
    x, y = a
    tx, ty = b
    dx, dy = abs(tx - x), abs(ty - y)
    sx = 1 if x < tx else -1
    sy = 1 if y < ty else -1
    err = dx - dy
    hits = eligible = 0
    while True:
        if (max(abs(x - a[0]), abs(y - a[1])) > 2 and
                max(abs(x - tx), abs(y - ty)) > 2):
            v = int(obs[y, x])
            if v in (1, 5, 10):
                eligible += 1
                hits += v == 1
        if (x, y) == (tx, ty):
            break
        e = 2 * err
        if e > -dy:
            err -= dy
            x += sx
        if e < dx:
            err += dx
            y += sy
    if eligible:
        return hits / eligible
    return 0.65 if max(dx, dy) <= 7 else 0.05


def _contact_mask(p, offsets):
    mask = np.zeros((64, 64), dtype=bool)
    x, y = p
    for dx, dy in offsets:
        xx, yy = x + dx, y + dy
        if 0 <= xx < 64 and 0 <= yy < 64:
            mask[yy, xx] = True
    return mask


def _overlap(a, b, offsets):
    return (a[0] - b[0], a[1] - b[1]) in offsets


def _clearance(blocked, offsets):
    valid = np.ones((64, 64), dtype=bool)
    for dx, dy in offsets:
        shifted = np.ones((64, 64), dtype=bool)
        x0, x1 = max(0, -dx), min(64, 64 - dx)
        y0, y1 = max(0, -dy), min(64, 64 - dy)
        shifted[y0:y1, x0:x1] = blocked[y0 + dy:y1 + dy,
                                               x0 + dx:x1 + dx]
        valid &= ~shifted
    return valid


def _dilate(mask, radius):
    p = np.pad(mask.astype(np.int32), radius)
    s = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * radius + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] +
            s[:-k, :-k]) > 0


def _set_mass(group, controls):
    """Account for controls temporarily hidden underneath the ball."""
    count = len(group["ids"])
    sx = sum(controls[i]["p"][0] for i in group["ids"])
    sy = sum(controls[i]["p"][1] for i in group["ids"])
    mass = max(count, group.get("mass", count))
    if count >= 2 and mass == count:
        if (sx // count, sy // count) != group["p"]:
            mass += 1
    group["mass"] = mass
    if mass > count:
        group["hidden"] = (mass * group["p"][0] - sx,
                           mass * group["p"][1] - sy)
    else:
        group["hidden"] = (0, 0)


def _initialize(obs):
    controls = _controls(obs)
    controls.sort(key=lambda c: c["p"][0] ** 2 + c["p"][1] ** 2)
    rounds = _rounds(obs)
    goals = _rings(obs)
    groups = []
    used_rounds = set()
    used_controls = set()

    for ri, ball in enumerate(rounds):
        if ball["native"] is None:
            continue
        ids = [i for i, c in enumerate(controls)
               if c["color"] == ball["native"] and i not in used_controls]
        if ids:
            groups.append({"ids": ids, "p": ball["p"],
                           "native": ball["native"], "lr": None})
            used_rounds.add(ri)
            used_controls.update(ids)

    black_ids = [i for i, c in enumerate(controls)
                 if i not in used_controls and c["color"] == 4]
    candidates = [i for i, b in enumerate(rounds)
                  if i not in used_rounds and b["native"] is None]

    if black_ids and candidates:
        scores = [[_line_score(obs, controls[ci]["p"], rounds[ri]["p"])
                   for ri in candidates] for ci in black_ids]
        options = []
        for row in scores:
            order = sorted(range(len(candidates)),
                           key=lambda j: row[j], reverse=True)
            options.append(order[:min(3, len(order))])
        while math.prod(len(o) for o in options) > 50000:
            choices = [i for i, o in enumerate(options) if len(o) > 1]
            if not choices:
                break
            i = max(choices, key=lambda k:
                    scores[k][options[k][0]] - scores[k][options[k][-1]])
            options[i].pop()

        best_score = float("inf")
        best_assignment = None
        for assignment in itertools.product(*options):
            buckets = {}
            penalty = 0.0
            for k, j in enumerate(assignment):
                buckets.setdefault(j, []).append(black_ids[k])
                penalty += 2.0 * (1.0 - scores[k][j])
            penalty += 4.0 * len(buckets)
            for j, ids in buckets.items():
                bx, by = rounds[candidates[j]]["p"]
                mx = sum(controls[i]["p"][0] for i in ids) // len(ids)
                my = sum(controls[i]["p"][1] for i in ids) // len(ids)
                error = abs(mx - bx) + abs(my - by)
                if len(black_ids) == 1:
                    error = 0
                penalty += error * (8.0 if len(ids) >= 2 else 1.5)
            if penalty < best_score:
                best_score = penalty
                best_assignment = buckets

        if best_assignment:
            for j, ids in best_assignment.items():
                ri = candidates[j]
                b = rounds[ri]
                groups.append({"ids": ids, "p": b["p"],
                               "native": None, "lr": b["lr"]})
                used_rounds.add(ri)
                used_controls.update(ids)

    pickups = []
    for ri, b in enumerate(rounds):
        if ri in used_rounds or b["native"] is not None:
            continue
        l, r = b["lr"]
        if bool(l) != bool(r):
            pickups.append({"p": b["p"], "side": 0 if l else 1,
                            "color": l or r})

    for goal in goals:
        if not goal["colors"]:
            touching = [g for g in groups
                        if _overlap(g["p"], goal["p"], _GOAL_CONTACT)]
            if touching:
                g = min(touching, key=lambda g:
                        math.dist(g["p"], goal["p"]))
                goal["colors"] = (frozenset([g["native"]])
                                  if g["native"] is not None else
                                  frozenset(v for v in g["lr"] if v))

    wall = obs == 2
    hazard = obs == 10
    has_hazard = bool(np.any(hazard))
    if has_hazard:
        for _ in range(3):
            add = np.zeros_like(hazard)
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                a = np.roll(hazard, (dy, dx), (0, 1))
                b = np.roll(hazard, (-dy, -dx), (0, 1))
                add |= a & b
            hazard |= add & (obs == 1)
        hazard |= wall

    safe_base = _clearance(hazard, _BODY)
    safe = (_clearance(_dilate(hazard, 1), _BODY)
            if has_hazard else safe_base.copy())
    selected = next((i for i, c in enumerate(controls) if c["selected"]), 0)

    for p in pickups:
        p["mask"] = _contact_mask(p["p"], _PICK_CONTACT)
    for g in goals:
        g["mask"] = _contact_mask(g["p"], _GOAL_CONTACT)
    for g in groups:
        _set_mass(g, controls)

    return {"controls": controls, "groups": groups, "goals": goals,
            "pickups": pickups, "remaining": set(range(len(pickups))),
            "valid": _clearance(wall, _DIAMOND),
            "safe": safe, "safe_base": safe_base,
            "selected": selected, "routes": None, "active": None,
            "path": [], "pending": None, "failures": 0,
            "alternate": False}


def _paint_set(group):
    return frozenset(v for v in group["lr"] if v)


def _goal_done(model, group, goal_index):
    goal = model["goals"][goal_index]
    colors_ok = (group["native"] is not None or
                 _paint_set(group) == goal["colors"])
    return colors_ok and _overlap(group["p"], goal["p"], _GOAL_CONTACT)


def _route_options(model, group, goal_index):
    goal = model["goals"][goal_index]
    required = goal["colors"]
    if not required:
        return []
    if _goal_done(model, group, goal_index):
        return [(0.0, [], frozenset())]

    pickups = model["pickups"]
    lr = group["lr"]
    side_options = []
    for side in (0, 1):
        choices = [(lr[side], None)]
        choices.extend((pickups[i]["color"], i) for i in model["remaining"]
                       if pickups[i]["side"] == side and
                       pickups[i]["color"] in required)
        side_options.append(choices)

    result = []
    n = group.get("mass", len(group["ids"]))
    for left, right in itertools.product(*side_options):
        if frozenset(v for v in (left[0], right[0]) if v) != required:
            continue
        ids = [v[1] for v in (left, right) if v[1] is not None]
        if len(set(ids)) != len(ids):
            continue
        orders = list(itertools.permutations(ids)) if ids else [()]
        for order in orders:
            position = group["p"]
            cost = 0.0
            route = []
            for pi in order:
                dest = pickups[pi]["p"]
                cost += 1.0 + max(0.0, math.dist(position, dest) - 4) * n / 48
                position = dest
                route.append(("p", pi))
            cost += 1.0 + max(0.0, math.dist(position, goal["p"]) - 5) * n / 48
            route.append(("g", goal_index))
            result.append((cost, route, frozenset(ids)))
    result.sort(key=lambda a: a[0])
    return result[:10]


def _allocate(model):
    groups, goals = model["groups"], model["goals"]
    routes = [[] for _ in groups]
    claimed = set()
    for gi, group in enumerate(groups):
        if group["native"] is None:
            continue
        candidates = [j for j, goal in enumerate(goals)
                      if goal["colors"] == frozenset([group["native"]])
                      and j not in claimed]
        if not candidates:
            return False
        j = min(candidates, key=lambda j:
                (not _goal_done(model, group, j),
                 math.dist(group["p"], goals[j]["p"])))
        claimed.add(j)
        routes[gi] = [] if _goal_done(model, group, j) else [("g", j)]

    black = [i for i, g in enumerate(groups) if g["native"] is None]
    free = [j for j in range(len(goals)) if j not in claimed
            and goals[j]["colors"]]
    split = [j for j in free if len(goals[j]["colors"]) >= 2]
    if not model["alternate"] and len(split) >= len(black):
        free = split

    pair_options = {}
    for gi in black:
        options = []
        for j in free:
            for cost, route, resource in _route_options(model, groups[gi], j):
                if split and len(goals[j]["colors"]) == 1:
                    cost += 6
                options.append((cost, j, route, resource))
        options.sort(key=lambda a: a[0])
        pair_options[gi] = options

    best_cost = float("inf")
    best_routes = None

    def search(k, cost, used_goals, used_pickups, chosen):
        nonlocal best_cost, best_routes
        if cost >= best_cost:
            return
        if k == len(black):
            best_cost = cost
            best_routes = dict(chosen)
            return
        gi = black[k]
        for c, j, route, resource in pair_options[gi]:
            if j in used_goals or resource & used_pickups:
                continue
            chosen[gi] = route
            search(k + 1, cost + c, used_goals | {j},
                   used_pickups | resource, chosen)
            chosen.pop(gi, None)

    search(0, 0.0, set(), frozenset(), {})
    if black and best_routes is None:
        return False
    if best_routes:
        for gi, route in best_routes.items():
            routes[gi] = route
    model["routes"] = routes
    model["active"] = None
    model["path"] = []
    return True


def _simulate_move(model, control_index, destination):
    controls = model["controls"]
    controls[control_index]["p"] = destination
    model["selected"] = control_index
    for group in model["groups"]:
        if control_index in group["ids"]:
            ids = group["ids"]
            n = group["mass"]
            hx, hy = group["hidden"]
            group["p"] = (
                (hx + sum(controls[i]["p"][0] for i in ids)) // n,
                (hy + sum(controls[i]["p"][1] for i in ids)) // n)
            break

    for group in model["groups"]:
        if group["native"] is not None:
            continue
        lr = list(group["lr"])
        for pi in sorted(list(model["remaining"])):
            pickup = model["pickups"][pi]
            if _overlap(group["p"], pickup["p"], _PICK_CONTACT):
                lr[pickup["side"]] = pickup["color"]
                model["remaining"].remove(pi)
        group["lr"] = tuple(lr)


def _reconcile(model, obs):
    found = _controls(obs)
    controls = model["controls"]
    pending = model.get("pending")
    if pending and pending["kind"] == "move":
        new, old = pending["new"], pending["old"]
        at_new = any(c["p"] == new for c in found)
        at_old = any(c["p"] == old for c in found)
        if not at_new and at_old and old != new:
            failed_center = pending["ball"]
            controls[pending["index"]]["p"] = old
            for g, state in zip(model["groups"], pending["groups"]):
                g["p"], g["lr"] = state
            model["remaining"] = set(pending["remaining"])
            model["valid"][new[1], new[0]] = False
            fx, fy = failed_center
            if 0 <= fx < 64 and 0 <= fy < 64:
                model["safe"][fy, fx] = False
                model["safe_base"][fy, fx] = False
            model["path"] = []
            model["failures"] += 1
    model["pending"] = None

    rounds = _rounds(obs)
    changed = False

    # Observe native balls directly rather than trusting a previously
    # reconstructed centroid.
    for group in model["groups"]:
        if group["native"] is not None:
            candidates = [b for b in rounds
                          if b["native"] == group["native"]]
            if candidates:
                ball = min(candidates, key=lambda b:
                           math.dist(b["p"], group["p"]))
                if ball["p"] != group["p"]:
                    changed = True
                group["p"] = ball["p"]

    # Moving a ball can reveal a control that was completely covered at
    # episode start. Add it immediately and discard the old motion plan.
    for c in found:
        known = next((i for i, q in enumerate(controls)
                      if q["p"] == c["p"]), None)
        if known is None:
            candidates = [gi for gi, g in enumerate(model["groups"])
                          if ((g["native"] is not None and
                               c["color"] == g["native"]) or
                              (g["native"] is None and c["color"] == 4))]
            if candidates:
                gi = max(candidates, key=lambda j:
                         _line_score(obs, c["p"], model["groups"][j]["p"]))
                known = len(controls)
                controls.append(c.copy())
                model["groups"][gi]["ids"].append(known)
                changed = True
        if known is not None and c["selected"]:
            model["selected"] = known

    pickup_positions = {model["pickups"][i]["p"]
                        for i in model["remaining"]}
    black_rounds = [b for b in rounds if b["native"] is None
                    and b["p"] not in pickup_positions]
    assigned = set()
    for group in model["groups"]:
        if group["native"] is not None:
            continue
        candidates = [i for i in range(len(black_rounds))
                      if i not in assigned]
        if not candidates:
            continue
        exact = [i for i in candidates
                 if black_rounds[i]["p"] == group["p"]]
        if exact:
            ri = exact[0]
        else:
            def score(i):
                p = black_rounds[i]["p"]
                line = sum(_line_score(obs, controls[j]["p"], p)
                           for j in group["ids"]) / len(group["ids"])
                return line - 0.025 * math.dist(p, group["p"])
            ri = max(candidates, key=score)
            if score(ri) < -0.1:
                continue
        ball = black_rounds[ri]
        assigned.add(ri)
        if ball["p"] != group["p"]:
            changed = True
        group["p"] = ball["p"]
        group["lr"] = ball["lr"]

    for group in model["groups"]:
        previous = (group.get("mass"), group.get("hidden"))
        _set_mass(group, controls)
        if previous != (group["mass"], group["hidden"]):
            changed = True
    if changed:
        model["path"] = []


def _plan_stage(model, gi, stage, deadline, relaxed=False):
    group = model["groups"][gi]
    ids = group["ids"]
    n = group["mass"]
    hx, hy = group["hidden"]
    controls = model["controls"]
    positions = tuple(controls[i]["p"] for i in ids)
    selected = ids.index(model["selected"]) if model["selected"] in ids else -1

    kind, target_index = stage
    target_obj = (model["pickups"][target_index] if kind == "p"
                  else model["goals"][target_index])
    safety = model["safe_base"] if relaxed else model["safe"]
    allowed = safety.copy()
    if group["native"] is None:
        own_pickups = {j for k, j in model["routes"][gi] if k == "p"}
        for pi in model["remaining"] - own_pickups:
            allowed &= ~model["pickups"][pi]["mask"]
    target = target_obj["mask"] & allowed
    if not np.any(target):
        if not relaxed:
            return _plan_stage(model, gi, stage, deadline, True)
        return []

    radius = max(1, math.ceil(63 / n))
    hops = np.full((64, 64), 100.0)
    reached = target.copy()
    hops[target] = 0
    for d in range(1, 24):
        new = _dilate(reached, radius) & allowed & ~reached
        if not np.any(new):
            break
        hops[new] = d
        reached |= new

    yy, xx = np.indices((64, 64))
    tx, ty = target_obj["p"]
    geometric = np.maximum(0.0, np.hypot(xx - tx, yy - ty) - 4.5)
    heuristic = hops * 15.0 + geometric / 5.0

    cy, cx = np.nonzero(model["valid"])
    if not len(cx):
        return []
    static_controls = [c["p"] for i, c in enumerate(controls) if i not in ids]
    static_valid = np.ones(len(cx), dtype=bool)
    for x, y in static_controls:
        static_valid &= ((np.abs(cx - x) > 4) | (np.abs(cy - y) > 4))

    start_h = float(heuristic[group["p"][1], group["p"][0]])
    beam = [(positions, selected, [], 0.0)]
    visited = {}
    best_path = []
    best_value = start_h
    last_path = []

    for depth in range(22):
        if time.monotonic() > deadline:
            break
        children = []
        winning = []
        for pos, sel, path, cost in beam:
            sx = hx + sum(p[0] for p in pos)
            sy = hy + sum(p[1] for p in pos)
            for local_index, old in enumerate(pos):
                valid = static_valid.copy()
                for j, (x, y) in enumerate(pos):
                    clearance = 2 if j == local_index else 4
                    valid &= ((np.abs(cx - x) > clearance) |
                              (np.abs(cy - y) > clearance))
                bx = (sx - old[0] + cx) // n
                by = (sy - old[1] + cy) // n
                in_bounds = (bx >= 0) & (bx < 64) & (by >= 0) & (by < 64)
                valid &= in_bounds
                bxc, byc = np.clip(bx, 0, 63), np.clip(by, 0, 63)
                valid &= allowed[byc, bxc]
                indices = np.flatnonzero(valid)
                if not len(indices):
                    continue
                new_cost = cost + 1.0 + (sel != local_index)

                winners = indices[target[by[indices], bx[indices]]]
                if len(winners):
                    rank = np.hypot(cx[winners] - tx, cy[winners] - ty)
                    q = int(winners[int(np.argmin(rank))])
                    winning.append((new_cost, path +
                                    [(ids[local_index], (int(cx[q]), int(cy[q])))]))
                    continue

                rank = (heuristic[by[indices], bx[indices]] +
                        0.015 * np.hypot(cx[indices] - tx, cy[indices] - ty))
                count = min(240, len(indices))
                if count < len(indices):
                    subset = np.argpartition(rank, count - 1)[:count]
                    order = subset[np.argsort(rank[subset])]
                else:
                    order = np.argsort(rank)

                seen_centers = set()
                kept = 0
                for qi in order:
                    q = int(indices[int(qi)])
                    center = (int(bx[q]), int(by[q]))
                    if center in seen_centers:
                        continue
                    seen_centers.add(center)
                    new_pos = list(pos)
                    dest = (int(cx[q]), int(cy[q]))
                    new_pos[local_index] = dest
                    new_pos = tuple(new_pos)
                    key = (new_pos, local_index)
                    if visited.get(key, float("inf")) <= new_cost:
                        continue
                    visited[key] = new_cost
                    new_path = path + [(ids[local_index], dest)]
                    h = float(heuristic[center[1], center[0]])
                    spread = sum(math.dist(p, (tx, ty)) for p in new_pos) / len(ids)
                    value = h + 0.15 * new_cost + 0.025 * spread
                    children.append((value, new_pos, local_index,
                                     new_path, new_cost))
                    if h + 0.04 * new_cost < best_value:
                        best_value = h + 0.04 * new_cost
                        best_path = new_path
                    kept += 1
                    if kept >= 10:
                        break
        if winning:
            return min(winning, key=lambda a: a[0])[1]
        if not children:
            break
        children.sort(key=lambda a: a[0])
        beam = []
        coarse_seen = set()
        for value, pos, sel, path, cost in children:
            coarse = (tuple((x // 2, y // 2) for x, y in pos), sel)
            if coarse in coarse_seen:
                continue
            coarse_seen.add(coarse)
            beam.append((pos, sel, path, cost))
            if len(beam) >= 28:
                break
        if beam:
            last_path = beam[0][2]

    partial = best_path or last_path
    if not partial and not relaxed and time.monotonic() < deadline - 0.3:
        return _plan_stage(model, gi, stage, deadline, True)
    return partial[:3]


def _selection_point(model, index):
    x, y = model["controls"][index]["p"]
    others = [c["p"] for i, c in enumerate(model["controls"]) if i != index]
    points = [(x, y)]
    points.extend((x + dx, y + dy) for dy in range(-2, 3)
                  for dx in range(-2, 3))
    for px, py in points:
        if not (0 <= px < 64 and 0 <= py < 64):
            continue
        if all(abs(px - ox) > 2 or abs(py - oy) > 2 for ox, oy in others):
            return px, py
    return x, y


def act(obs, available, memory):
    started = time.monotonic()
    obs = np.asarray(obs)
    if "model" not in memory:
        memory["model"] = _initialize(obs)
    model = memory["model"]
    _reconcile(model, obs)

    def reset():
        memory.clear()
        return ("RESET", None, None)

    if not model["groups"] or not model["controls"]:
        return reset() if "RESET" in available else ("ACTION6", 32, 32)

    if model["routes"] is None:
        if not _allocate(model):
            model["alternate"] = True
            if not _allocate(model):
                return reset() if "RESET" in available else ("ACTION6", 32, 32)

    for gi, route in enumerate(model["routes"]):
        group = model["groups"][gi]
        while route:
            kind, index = route[0]
            if kind == "p" and index not in model["remaining"]:
                route.pop(0)
                if model["active"] == gi:
                    model["path"] = []
            elif kind == "g" and _goal_done(model, group, index):
                route.pop(0)
                if model["active"] == gi:
                    model["path"] = []
            else:
                break

    unfinished = [i for i, r in enumerate(model["routes"]) if r]
    if not unfinished:
        model["alternate"] = True
        model["routes"] = None
        if _allocate(model):
            unfinished = [i for i, r in enumerate(model["routes"]) if r]
        if not unfinished:
            return reset() if "RESET" in available else ("ACTION6", 32, 32)

    if model["active"] not in unfinished:
        def task_cost(gi):
            group = model["groups"][gi]
            kind, index = model["routes"][gi][0]
            obj = model["pickups"][index] if kind == "p" else model["goals"][index]
            switch = 0 if model["selected"] in group["ids"] else 1
            return switch + math.dist(group["p"], obj["p"]) * group["mass"] / 48
        model["active"] = min(unfinished, key=task_cost)
        model["path"] = []

    gi = model["active"]
    stage = model["routes"][gi][0]
    if not model["path"]:
        model["path"] = _plan_stage(model, gi, stage, started + 7.4)

    if not model["path"]:
        model["failures"] += 1
        if model["failures"] >= 2 and "RESET" in available:
            return reset()
        model["routes"] = None
        return ("ACTION6", 32, 32)

    index, destination = model["path"][0]
    if model["selected"] != index:
        px, py = _selection_point(model, index)
        model["selected"] = index
        model["pending"] = {"kind": "select"}
        return ("ACTION6", int(py), int(px))

    model["path"].pop(0)
    old = model["controls"][index]["p"]
    snapshot = [(g["p"], g["lr"]) for g in model["groups"]]
    remaining = set(model["remaining"])
    _simulate_move(model, index, destination)
    moved_group = next(g for g in model["groups"] if index in g["ids"])
    model["pending"] = {"kind": "move", "index": index,
                        "old": old, "new": destination,
                        "groups": snapshot, "remaining": remaining,
                        "ball": moved_group["p"]}
    return ("ACTION6", int(destination[1]), int(destination[0]))

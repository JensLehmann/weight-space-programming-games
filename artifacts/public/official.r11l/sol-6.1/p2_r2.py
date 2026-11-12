import numpy as np
import itertools
import math
import time

BALL = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
        if not (abs(x) == 2 and abs(y) == 2)]
DIAMOND = [(x, y) for y in range(-2, 3) for x in range(-2, 3)
           if abs(x) + abs(y) <= 2]
DEDGE = [p for p in DIAMOND if p != (0, 0)]
LEFT = [(-1, -3), (-2, -2), (-3, -1),
        (-3, 1), (-2, 2), (-1, 3)]
RIGHT = [(1, -3), (2, -2), (3, -1),
         (3, 1), (2, 2), (1, 3)]
RING = [(x, y) for y in range(-3, 4) for x in range(-3, 4)
        if abs(y) <= 1 or
        (abs(y) == 2 and abs(x) <= 2) or
        (abs(y) == 3 and abs(x) <= 1)]
GOAL_DELTA = {(a-c, b-d) for a, b in BALL for c, d in RING}
PICK_DELTA = {(a-c, b-d) for a, b in BALL for c, d in BALL}
SPECIAL = {1, 2, 3, 5, 10}


def values(obs, x, y, footprint):
    return [int(obs[y+dy, x+dx]) for dx, dy in footprint]


def hit(a, b, delta):
    return (b[0]-a[0], b[1]-a[1]) in delta


def detect_controls(obs):
    out = []
    for y, x in np.argwhere((obs != 0) & (obs != 1) & (obs != 2) &
                            (obs != 3) & (obs != 5) & (obs != 10)):
        x, y = int(x), int(y)
        if 2 <= x <= 61 and 2 <= y <= 61:
            edge = values(obs, x, y, DEDGE)
            if all(v == 0 for v in edge):
                out.append([x, y, int(obs[y, x]), True])
            elif all(v == 3 for v in edge):
                out.append([x, y, int(obs[y, x]), False])
    return out


def detect_bodies(obs):
    out = []
    for y in range(2, 62):
        for x in range(2, 62):
            if int(obs[y, x-1]) in SPECIAL:
                continue
            if int(obs[y, x+1]) in SPECIAL:
                continue
            if int(obs[y-2, x-1]) in SPECIAL:
                continue
            if int(obs[y+2, x+1]) in SPECIAL:
                continue
            vals = values(obs, x, y, BALL)
            if any(v in SPECIAL for v in vals):
                continue
            center = int(obs[y, x])
            outer = [v for p, v in zip(BALL, vals) if p != (0, 0)]
            regular = center == 6 and len(set(outer)) == 1
            if regular or center != 6:
                out.append({
                    "p": [x, y], "regular": regular,
                    "color": outer[0] if regular else None,
                    "left": int(obs[y, x-1]),
                    "right": int(obs[y, x+1]),
                    "vals": vals
                })
    return out


def infer_hidden_control(obs, controls, group, body):
    """Recover one partially occluded control using floor-average constraints."""
    ids = group["ids"]
    n = len(ids)
    if n < 2:
        return None
    sx = sum(controls[i][0] for i in ids)
    sy = sum(controls[i][1] for i in ids)
    tx, ty = body["p"]
    if [sx//n, sy//n] == [tx, ty]:
        return None

    hidden = {(tx+dx, ty+dy) for dx, dy in BALL}
    label = group["color"] if group["regular"] else 4
    candidates = []
    for x in range((n+1)*tx-sx, (n+1)*tx-sx+n+1):
        for y in range((n+1)*ty-sy, (n+1)*ty-sy+n+1):
            if not (2 <= x <= 61 and 2 <= y <= 61):
                continue
            if any(c[:2] == [x, y] for c in controls):
                continue
            if (x, y) not in hidden:
                continue
            if (x, y) not in hidden and int(obs[y, x]) != label:
                continue
            visible = [int(obs[y+dy, x+dx]) for dx, dy in DEDGE
                       if (x+dx, y+dy) not in hidden]
            if len(visible) < 3:
                continue
            if all(v == 3 for v in visible):
                selected = False
            elif all(v == 0 for v in visible):
                selected = True
            else:
                continue
            score = len(visible) - 0.01*((x-tx)**2+(y-ty)**2)
            candidates.append((score, x, y, selected))
    if not candidates:
        return None
    _, x, y, selected = max(candidates)
    index = len(controls)
    controls.append([x, y, label, selected])
    group["ids"].append(index)
    return index


def detect_goals(obs, bodies, controls):
    occluded = np.zeros((64, 64), dtype=bool)
    for b in bodies:
        x, y = b["p"]
        for dx, dy in BALL:
            occluded[y+dy, x+dx] = True
    for c in controls:
        x, y = c[:2]
        for dx, dy in DIAMOND:
            occluded[y+dy, x+dx] = True

    goals = []
    for y in range(3, 61):
        for x in range(3, 61):
            halves = []
            count = 0
            valid = True
            for offsets in (LEFT, RIGHT):
                vals = [int(obs[y+dy, x+dx]) for dx, dy in offsets
                        if not occluded[y+dy, x+dx]]
                count += len(vals)
                if len(vals) < 3 or len(set(vals)) != 1:
                    valid = False
                    break
                if vals[0] in SPECIAL:
                    valid = False
                    break
                halves.append(vals[0])
            if not valid or count < 8:
                continue
            lc, rc = halves
            if lc == rc == 0:
                near = [b for b in bodies
                        if hit(b["p"], [x, y], GOAL_DELTA)
                        and b["left"] > 0 and b["right"] > 0]
                if not near:
                    continue
                b = min(near, key=lambda q:
                        (q["p"][0]-x)**2 + (q["p"][1]-y)**2)
                lc, rc = b["left"], b["right"]
            if lc == 0 or rc == 0:
                continue
            if not any((q[0]-x)**2 + (q[1]-y)**2 <= 2 for q in goals):
                goals.append([x, y, lc, rc])

    fixed_colors = {b["color"] for b in bodies if b["regular"]}
    if any(q[2] != q[3] for q in goals):
        goals = [q for q in goals
                 if q[2] != q[3] or q[2] in fixed_colors]
    return goals


def line_score(obs, a, b):
    x, y = a
    tx, ty = b
    dx, dy = abs(tx-x), abs(ty-y)
    sx, sy = (1 if x < tx else -1), (1 if y < ty else -1)
    err = dx-dy
    good = total = 0
    while True:
        if 0 <= x < 64 and 0 <= y < 64:
            v = int(obs[y, x])
            if v in (1, 5, 10):
                total += 1
                good += v == 1
        if x == tx and y == ty:
            break
        e = 2*err
        if e > -dy:
            err -= dy
            x += sx
        if e < dx:
            err += dx
            y += sy
    return good / max(1, total)


def legal_map(blocked, footprint):
    out = np.zeros((64, 64), dtype=bool)
    out[2:62, 2:62] = True
    for dx, dy in footprint:
        shifted = np.ones((64, 64), dtype=bool)
        x0, x1 = max(0, -dx), min(64, 64-dx)
        y0, y1 = max(0, -dy), min(64, 64-dy)
        shifted[y0:y1, x0:x1] = blocked[
            y0+dy:y1+dy, x0+dx:x1+dx]
        out &= ~shifted
    return out


def reconstruct_hazard(obs):
    hazard = obs == 10
    unknown = obs == 1
    for _ in range(3):
        adjacent = np.zeros_like(hazard)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, -1), (1, -1), (-1, 1)):
            shifted = np.zeros_like(hazard)
            x0, x1 = max(0, dx), min(64, 64+dx)
            y0, y1 = max(0, dy), min(64, 64+dy)
            shifted[y0:y1, x0:x1] = hazard[
                y0-dy:y1-dy, x0-dx:x1-dx]
            adjacent |= shifted
        hazard |= unknown & adjacent
    return hazard


def initialize(obs, memory):
    controls = detect_controls(obs)
    bodies = detect_bodies(obs)
    if not controls or not bodies:
        return False
    groups = []
    used_controls = set()
    used_bodies = set()
    inferred = set()

    for bi, b in enumerate(bodies):
        if not b["regular"]:
            continue
        ids = [i for i, c in enumerate(controls)
               if i not in used_controls and c[2] == b["color"]]
        if ids:
            g = {
                "ids": ids, "regular": True, "color": b["color"],
                "p": b["p"][:], "left": b["left"],
                "right": b["right"], "moved": False
            }
            recovered = infer_hidden_control(obs, controls, g, b)
            if recovered is not None:
                inferred.add(recovered)
            groups.append(g)
            used_controls.update(g["ids"])
            used_bodies.add(bi)

    remaining = [i for i in range(len(controls)) if i not in used_controls]
    if len(remaining) > 14:
        return False
    candidates = []
    for size in range(1, min(7, len(remaining))+1):
        for ids in itertools.combinations(remaining, size):
            mx = sum(controls[i][0] for i in ids)//size
            my = sum(controls[i][1] for i in ids)//size
            for bi, b in enumerate(bodies):
                if bi in used_bodies or b["regular"]:
                    continue
                if b["p"] == [mx, my]:
                    score = sum(line_score(obs, controls[i][:2], b["p"])
                                for i in ids) + 0.15*size
                    candidates.append((score, ids, bi))

    best = [None, -1.0]

    def partition(left, chosen, bis, score):
        if not left:
            if score > best[1]:
                best[:] = [chosen[:], score]
            return
        first = min(left)
        opts = [q for q in candidates if first in q[1]
                and set(q[1]) <= left and q[2] not in bis]
        opts.sort(key=lambda q: -q[0])
        for s, ids, bi in opts[:20]:
            partition(left-set(ids), chosen+[(ids, bi)],
                      bis | {bi}, score+s)

    if remaining:
        partition(set(remaining), [], set(), 0.0)
        if best[0] is None:
            return False
        for ids, bi in best[0]:
            b = bodies[bi]
            groups.append({
                "ids": list(ids), "regular": False,
                "p": b["p"][:], "left": b["left"],
                "right": b["right"], "moved": False
            })
            used_bodies.add(bi)

    goals = detect_goals(obs, bodies, controls)
    if not goals:
        return False
    pickups = []
    for bi, b in enumerate(bodies):
        if bi not in used_bodies and not b["regular"]:
            if (b["left"] == 0) != (b["right"] == 0):
                pickups.append({
                    "p": b["p"][:], "left": b["left"],
                    "right": b["right"], "vals": b["vals"],
                    "alive": True
                })

    memory["controls"] = controls
    memory["groups"] = groups
    memory["goals"] = goals
    memory["pickups"] = pickups
    memory["inferred"] = inferred
    memory["anchor_ok"] = legal_map(obs == 2, DIAMOND)
    memory["ball_ok"] = legal_map(reconstruct_hazard(obs), BALL)
    for g in groups:
        x, y = g["p"]
        memory["ball_ok"][y, x] = True
    memory["queue"] = []
    memory["pending"] = None
    memory["selected"] = next((i for i, c in enumerate(controls) if c[3]), None)
    memory["steps"] = 0
    memory["failures"] = 0
    memory["history"] = {}
    memory["focus"] = None
    return True


def group_position(memory, g):
    ids = g["ids"]
    if len(ids) >= 2 or g["moved"]:
        controls = memory["controls"]
        return [sum(controls[i][0] for i in ids)//len(ids),
                sum(controls[i][1] for i in ids)//len(ids)]
    return g["p"][:]


def incorporate_controls(obs, memory, seen, bodies):
    controls = memory["controls"]
    groups = memory["groups"]
    changed = False
    inferred = memory["inferred"]

    for s in seen:
        exact = next((i for i, c in enumerate(controls)
                      if c[:2] == s[:2] and c[2] == s[2]), None)
        if exact is not None:
            inferred.discard(exact)
            continue

        near_inferred = [i for i in inferred
                         if controls[i][2] == s[2] and
                         abs(controls[i][0]-s[0]) <= 4 and
                         abs(controls[i][1]-s[1]) <= 4]
        if near_inferred:
            i = min(near_inferred, key=lambda i:
                    (controls[i][0]-s[0])**2+(controls[i][1]-s[1])**2)
            controls[i] = s[:]
            inferred.discard(i)
            changed = True
            continue

        matching = [g for g in groups
                    if g["regular"] and g["color"] == s[2]]
        if len(matching) == 1:
            i = len(controls)
            controls.append(s[:])
            matching[0]["ids"].append(i)
            changed = True
            continue

        if s[2] == 4:
            possible = []
            for gi, g in enumerate(groups):
                if g["regular"]:
                    continue
                n = len(g["ids"])+1
                sx = sum(controls[i][0] for i in g["ids"])+s[0]
                sy = sum(controls[i][1] for i in g["ids"])+s[1]
                predicted = [sx//n, sy//n]
                for b in bodies:
                    if not b["regular"] and b["p"] == predicted:
                        score = line_score(obs, s[:2], b["p"])
                        possible.append((score, gi))
            if possible:
                _, gi = max(possible)
                i = len(controls)
                controls.append(s[:])
                groups[gi]["ids"].append(i)
                changed = True

    for g in groups:
        if not g["regular"]:
            continue
        bs = [b for b in bodies
              if b["regular"] and b["color"] == g["color"]]
        if len(bs) == 1:
            recovered = infer_hidden_control(obs, controls, g, bs[0])
            if recovered is not None:
                inferred.add(recovered)
                changed = True

    if changed:
        memory["queue"] = []
        memory["history"] = {}
        memory["failures"] = 0
    return changed


def refresh(obs, memory):
    controls = memory["controls"]
    seen = detect_controls(obs)
    bodies = detect_bodies(obs)
    pending = memory.get("pending")
    moved = False
    rejected = None

    if pending is not None:
        i, gi, old, new, oldball, newball = pending
        c = controls[i]
        at_new = any(s[:2] == new and s[2] == c[2] for s in seen)
        at_old = any(s[:2] == old and s[2] == c[2] for s in seen)
        g = memory["groups"][gi]
        compatible = [b for b in bodies
                      if b["regular"] == g["regular"]
                      and (not g["regular"] or b["color"] == g["color"])]
        body_new = any(b["p"] == newball for b in compatible)
        body_old = any(b["p"] == oldball for b in compatible)
        failed = not at_new and (
            at_old or (newball != oldball and body_old and not body_new))
        if failed:
            c[0], c[1] = old
            memory["queue"] = []
            rejected = (gi, i, old, new)
            memory["failures"] += 1
        else:
            g["moved"] = True
            moved = True
        memory["pending"] = None

    changed = incorporate_controls(obs, memory, seen, bodies)

    if rejected is not None and not changed:
        gi, i, old, new = rejected
        g = memory["groups"][gi]
        n = len(g["ids"])
        sx = sum(controls[j][0] for j in g["ids"]) + new[0]-old[0]
        sy = sum(controls[j][1] for j in g["ids"]) + new[1]-old[1]
        x, y = sx//n, sy//n
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx*dx+dy*dy <= 4 and 0 <= x+dx < 64 and 0 <= y+dy < 64:
                    memory["ball_ok"][y+dy, x+dx] = False

    for i, c in enumerate(controls):
        s = next((s for s in seen if s[:2] == c[:2] and s[2] == c[2]), None)
        if s is not None:
            c[3] = s[3]
            if s[3]:
                memory["selected"] = i

    for g in memory["groups"]:
        g["p"] = group_position(memory, g)
        x, y = g["p"]
        if not g["regular"] and 2 <= x <= 61 and 2 <= y <= 61:
            b = next((b for b in bodies
                      if not b["regular"] and b["p"] == [x, y]), None)
            if b is not None:
                g["left"], g["right"] = b["left"], b["right"]

    for p in memory["pickups"]:
        if not p["alive"]:
            continue
        x, y = p["p"]
        if values(obs, x, y, BALL) != p["vals"]:
            p["alive"] = False
        elif moved and any(not g["regular"] and
                           hit(g["p"], p["p"], PICK_DELTA)
                           for g in memory["groups"]):
            p["alive"] = False


def color_set(g):
    return {v for v in (g["left"], g["right"]) if v > 0}


def covers(g, goal):
    correct = (goal[2] == goal[3] == g["color"]) if g["regular"] else (
        color_set(g) == set(goal[2:4]))
    return correct and hit(g["p"], goal[:2], GOAL_DELTA)


def route_options(g, goal, pickups):
    if g["regular"]:
        if goal[2] == goal[3] == g["color"]:
            d = math.hypot(g["p"][0]-goal[0], g["p"][1]-goal[1])
            return [(d, [])]
        return []
    alive = [i for i, p in enumerate(pickups) if p["alive"]]
    sequences = [()]
    sequences.extend((i,) for i in alive)
    sequences.extend((i, j) for i in alive for j in alive if i != j)
    out = []
    for seq in sequences:
        l, r = g["left"], g["right"]
        pos = g["p"]
        cost = 0.0
        for i in seq:
            p = pickups[i]
            cost += math.hypot(pos[0]-p["p"][0], pos[1]-p["p"][1])
            pos = p["p"]
            if p["left"]:
                l = p["left"]
            if p["right"]:
                r = p["right"]
        if {v for v in (l, r) if v > 0} == set(goal[2:4]):
            cost += math.hypot(pos[0]-goal[0], pos[1]-goal[1])
            cost += 12*len(seq)
            out.append((cost, list(seq)))
    out.sort(key=lambda q: q[0])
    return out


def choose_tasks(memory):
    groups, goals, pickups = (memory["groups"], memory["goals"],
                             memory["pickups"])
    covered = {qi for qi, q in enumerate(goals)
               if any(covers(g, q) for g in groups)}
    frozen = {gi for gi, g in enumerate(groups)
              if any(covers(g, goals[qi]) for qi in covered)}
    remaining = [qi for qi in range(len(goals)) if qi not in covered]
    options = {}
    fallback = []
    for qi in remaining:
        options[qi] = []
        for gi, g in enumerate(groups):
            if gi in frozen:
                continue
            for cost, seq in route_options(g, goals[qi], pickups)[:8]:
                task = (cost, gi, qi, seq)
                options[qi].append(task)
                fallback.append(task)
        options[qi].sort(key=lambda q: q[0])

    ordered = sorted(remaining, key=lambda qi: len(options[qi]))
    best = [float("inf"), None]

    def assign(k, used_groups, used_picks, total, tasks):
        if total >= best[0]:
            return
        if k == len(ordered):
            best[:] = [total, tasks[:]]
            return
        qi = ordered[k]
        for task in options[qi]:
            cost, gi, _, seq = task
            if gi in used_groups or set(seq) & used_picks:
                continue
            assign(k+1, used_groups | {gi}, used_picks | set(seq),
                   total+cost, tasks+[task])

    if len(ordered) <= 6:
        assign(0, set(), set(), 0.0, [])
    tasks = best[1] if best[1] is not None else fallback
    focus = memory.get("focus")
    tasks.sort(key=lambda q: (0 if focus == (q[1], q[2]) else 1, q[0]))
    return tasks, len(covered) == len(goals)


def event_maps(memory, g, target, pickup_index):
    safe = memory["ball_ok"].copy()
    if not g["regular"]:
        for i, p in enumerate(memory["pickups"]):
            if p["alive"] and i != pickup_index:
                px, py = p["p"]
                for dx, dy in PICK_DELTA:
                    x, y = px-dx, py-dy
                    if 0 <= x < 64 and 0 <= y < 64:
                        safe[y, x] = False
    event = np.zeros((64, 64), dtype=bool)
    delta = PICK_DELTA if pickup_index is not None else GOAL_DELTA
    tx, ty = target
    for dx, dy in delta:
        x, y = tx-dx, ty-dy
        if 0 <= x < 64 and 0 <= y < 64:
            event[y, x] = True
    event &= safe
    return safe, event


def search_moves(memory, gi, target, pickup_index, deadline):
    g = memory["groups"][gi]
    ids = g["ids"]
    n = len(ids)
    controls = memory["controls"]
    safe, event = event_maps(memory, g, target, pickup_index)
    if not np.any(event):
        return []

    root = tuple(v for i in ids for v in controls[i][:2])
    grid_y, grid_x = np.indices((64, 64))
    samples = safe & (((grid_x % 2 == 0) & (grid_y % 2 == 0)) | event)
    ys, xs = np.where(samples)
    distances = np.hypot(xs-target[0], ys-target[1])
    outside = [c[:2] for i, c in enumerate(controls) if i not in ids]
    selected = memory.get("selected")
    local_selected = ids.index(selected) if selected in ids else -1
    beam = [(root, [], local_selected, 0.0)]
    visited = {root: 0.0}
    best_path = []
    best_dist = math.hypot(g["p"][0]-target[0], g["p"][1]-target[1])
    local_deadline = min(deadline, time.perf_counter()+3.8)
    history = memory["history"]

    def rank(state, cost):
        totals = (sum(state[::2]), sum(state[1::2]))
        bound = 0
        for axis, total in enumerate(totals):
            need = max(0, abs(n*target[axis]-total)-3*n)
            if not need:
                continue
            caps = [(61-state[2*j+axis]) if n*target[axis] >= total
                    else state[2*j+axis]-2 for j in range(n)]
            accumulated = 0
            count = 0
            for cap in sorted(caps, reverse=True):
                accumulated += max(0, cap)
                count += 1
                if accumulated >= need:
                    break
            bound = max(bound, count)
        cx, cy = totals[0]//n, totals[1]//n
        return (cost + 1.5*bound +
                0.025*math.hypot(cx-target[0], cy-target[1]) +
                2.0*history.get((gi, state), 0))

    for depth in range(10):
        children = []
        for state, path, sel, cost in beam:
            if time.perf_counter() >= local_deadline:
                return best_path
            sx, sy = sum(state[::2]), sum(state[1::2])
            click_ok = memory["anchor_ok"].copy()
            positions = outside + [
                [state[2*j], state[2*j+1]] for j in range(n)]
            for px, py in positions:
                click_ok[max(0, py-2):min(64, py+3),
                         max(0, px-2):min(64, px+3)] = False

            for j in range(n):
                ox, oy = sx-state[2*j], sy-state[2*j+1]
                bx, by = n*xs-ox, n*ys-oy
                ex = np.full(len(xs), -1, dtype=int)
                ey = np.full(len(xs), -1, dtype=int)
                for rx in range(n):
                    for ry in range(n):
                        ax, ay = bx+rx, by+ry
                        ii = np.flatnonzero(
                            (ax >= 2) & (ax <= 61) &
                            (ay >= 2) & (ay <= 61) & (ex < 0))
                        if len(ii):
                            ii = ii[click_ok[ay[ii], ax[ii]]]
                            ex[ii], ey[ii] = ax[ii], ay[ii]
                possible = np.flatnonzero(ex >= 0)
                if not len(possible):
                    continue
                ordered = possible[np.argsort(distances[possible])]
                chosen = list(ordered[:3])
                for k in ordered[:300]:
                    if all((xs[k]-xs[q])**2+(ys[k]-ys[q])**2 >= 16
                           for q in chosen):
                        chosen.append(int(k))
                        if len(chosen) >= 10:
                            break
                for k in chosen:
                    new = list(state)
                    new[2*j], new[2*j+1] = int(ex[k]), int(ey[k])
                    new = tuple(new)
                    newcost = cost+1.0+(0.6 if sel != j else 0.0)
                    if visited.get(new, 1e9) <= newcost:
                        continue
                    visited[new] = newcost
                    newpath = path+[(ids[j], int(ex[k]), int(ey[k]))]
                    if event[ys[k], xs[k]]:
                        return newpath
                    if distances[k] < best_dist:
                        best_dist = float(distances[k])
                        best_path = newpath
                    children.append((rank(new, newcost), new,
                                     newpath, j, newcost))
        if not children:
            break
        children.sort(key=lambda q: q[0])
        beam = [(s, p, j, c) for _, s, p, j, c in children[:45]]
    return best_path


def emit(memory, move):
    i, x, y = move
    controls = memory["controls"]
    c = controls[i]
    if memory.get("selected") != i:
        memory["selected"] = i
        memory["steps"] += 1
        return "ACTION6", int(c[1]), int(c[0])

    gi = next(j for j, g in enumerate(memory["groups"]) if i in g["ids"])
    g = memory["groups"][gi]
    old = c[:2]
    oldball = g["p"][:]
    c[0], c[1] = x, y
    ids = g["ids"]
    newball = [sum(controls[j][0] for j in ids)//len(ids),
               sum(controls[j][1] for j in ids)//len(ids)]
    memory["pending"] = (i, gi, old, [x, y], oldball, newball)
    state = tuple(v for j in ids for v in controls[j][:2])
    key = (gi, state)
    memory["history"][key] = memory["history"].get(key, 0)+1
    if memory["queue"] and tuple(memory["queue"][0]) == tuple(move):
        memory["queue"].pop(0)
    memory["steps"] += 1
    return "ACTION6", int(y), int(x)


def finish_move(memory):
    controls, groups, goals = (memory["controls"], memory["groups"],
                              memory["goals"])
    selected = memory.get("selected")
    order = sorted(range(len(controls)), key=lambda i: i != selected)
    for i in order:
        gi = next(j for j, g in enumerate(groups) if i in g["ids"])
        g = groups[gi]
        ids, n = g["ids"], len(g["ids"])
        sx = sum(controls[j][0] for j in ids)
        sy = sum(controls[j][1] for j in ids)
        for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3),
                       (4, 1), (-4, -1), (1, 4), (-1, -4)):
            x, y = controls[i][0]+dx, controls[i][1]+dy
            if not (2 <= x <= 61 and 2 <= y <= 61):
                continue
            if not memory["anchor_ok"][y, x]:
                continue
            if any(abs(x-c[0]) <= 2 and abs(y-c[1]) <= 2 for c in controls):
                continue
            p = [(sx+dx)//n, (sy+dy)//n]
            if not memory["ball_ok"][p[1], p[0]]:
                continue
            old = g["p"]
            previous = [q for q in goals if covers(g, q)]
            g["p"] = p
            preserves = all(covers(g, q) for q in previous)
            g["p"] = old
            if preserves:
                return i, x, y
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    deadline = time.perf_counter()+8.7
    if "controls" not in memory:
        if not initialize(obs, memory):
            memory.clear()
            return "RESET", None, None

    refresh(obs, memory)

    if memory["failures"] >= 2:
        memory.clear()
        return "RESET", None, None

    if memory["queue"]:
        return emit(memory, tuple(memory["queue"][0]))

    tasks, complete = choose_tasks(memory)
    if complete:
        move = finish_move(memory)
        if move is not None:
            memory["queue"] = [move]
            return emit(memory, move)

    for _, gi, qi, seq in tasks[:8]:
        if time.perf_counter() >= deadline-0.1:
            break
        if seq:
            pi = seq[0]
            target = memory["pickups"][pi]["p"]
        else:
            pi = None
            target = memory["goals"][qi][:2]
        path = search_moves(memory, gi, target, pi, deadline)
        if path:
            memory["focus"] = (gi, qi)
            memory["queue"] = path
            return emit(memory, tuple(path[0]))

    memory.clear()
    return "RESET", None, None

import numpy as np
import time
import heapq
import itertools
from collections import Counter

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _segment_visible(o, x, y, horizontal):
    if x < 0 or y < 0 or x + 6 > 64 or y + 6 > 64:
        return False
    if horizontal:
        v = (int(o[y + 2, x]), int(o[y + 3, x]),
             int(o[y + 2, x + 5]), int(o[y + 3, x + 5]))
        return v in ((3, 2, 2, 3), (2, 1, 1, 2))
    v = (int(o[y, x + 2]), int(o[y, x + 3]),
         int(o[y + 5, x + 2]), int(o[y + 5, x + 3]))
    return v in ((2, 3, 3, 2), (1, 2, 2, 1))


def _read_objects(o):
    handles = []
    beads = []
    for y in range(59):
        for x in range(59):
            c = int(o[y, x])
            if c in (0, 1, 4):
                continue
            if c == 5:
                if (o[y + 2, x + 2] == 11 and
                    o[y + 2, x + 3] == 11 and
                    o[y + 3, x + 2] == 11 and
                    o[y + 3, x + 3] == 11 and
                    o[y + 1, x + 1] == 5 and
                    o[y + 4, x + 4] == 5 and
                    o[y, x + 5] == 5 and
                    o[y + 5, x] == 5):
                    possibilities = []
                    for d, (dx, dy) in enumerate(DIRS):
                        if _segment_visible(
                                o, x + 6 * dx, y + 6 * dy, dx != 0):
                            possibilities.append(d)
                    if _segment_visible(o, x, y, False):
                        d = 1
                    elif _segment_visible(o, x, y, True):
                        d = 0
                    else:
                        d = possibilities[0] if possibilities else 1
                    handles.append((x, y, d, 11, False))
            elif (o[y, x + 5] == c and o[y + 5, x] == c and
                  o[y + 5, x + 5] == c and
                  o[y + 2, x + 2] == c and
                  o[y + 2, x + 3] == c and
                  o[y + 3, x + 2] == c and
                  o[y + 3, x + 3] == c):
                ring = [int(o[y + yy, x + xx])
                        for yy, xx in ((1, 1), (1, 4), (4, 1), (4, 4))]
                if all(v == ring[0] for v in ring) and ring[0] in (0, 4):
                    edges = (
                        (int(o[y + 2, x + 5]), int(o[y + 3, x + 5])),
                        (int(o[y + 5, x + 2]), int(o[y + 5, x + 3])),
                        (int(o[y + 2, x]), int(o[y + 3, x])),
                        (int(o[y, x + 2]), int(o[y, x + 3])),
                    )
                    gaps = [i for i, e in enumerate(edges)
                            if e[0] != c and e[1] != c]
                    if gaps:
                        handles.append((x, y, gaps[0], c, ring[0] == 0))

    for y in range(61):
        for x in range(61):
            c = int(o[y, x])
            if c in (0, 1, 2, 3, 4, 5):
                continue
            if x and o[y, x - 1] == c:
                continue
            if y and o[y - 1, x] == c:
                continue
            if not (np.all(o[y, x:x + 4] == c) and
                    np.all(o[y + 3, x:x + 4] == c) and
                    np.all(o[y:y + 4, x] == c) and
                    np.all(o[y:y + 4, x + 3] == c)):
                continue
            middle = o[y + 1:y + 3, x + 1:x + 3]
            if np.all((middle == c) | (middle == 0)):
                beads.append((x - 1, y - 1, c))
    return handles, beads


def _board(o):
    runs = []
    longest = 0
    for y in range(53):
        x = 0
        while x < 64:
            if o[y, x] != 4:
                x += 1
                continue
            a = x
            while x < 64 and o[y, x] == 4:
                x += 1
            n = x - a
            if n > longest:
                longest = n
                runs = [(a, x - 1)]
            elif n == longest:
                runs.append((a, x - 1))
    if longest < 12:
        return None
    left, right = Counter(runs).most_common(1)[0][0]
    ys = [y for y in range(53)
          if o[y, left] == 4 or o[y, right] == 4]
    if not ys:
        return None
    return left, min(ys), right + 1, max(ys) + 1


def _rail_length(o, handle):
    x, y, d = handle[:3]
    dx, dy = DIRS[d]
    n = 1
    while n < 12 and _segment_visible(
            o, x + 6 * dx * n, y + 6 * dy * n, dx != 0):
        n += 1
    return n


def _canonical_positions(beads, colors):
    by_color = {}
    for x, y, c in beads:
        if y < 53:
            by_color.setdefault(c, []).append((x, y))
    result = []
    for c in sorted(set(colors)):
        points = sorted(by_color.get(c, []))
        if len(points) != colors.count(c):
            return None
        result.extend(points)
    return tuple(result)


def _joint_goals(model, state):
    movable = model["movable"]
    if len(movable) != 2:
        return None
    a, b = movable
    ra, rb = model["rails"][a], model["rails"][b]
    if (ra["dx"] == 0) == (rb["dx"] == 0):
        return None
    colors = model["colors"]
    if any(colors.count(c) > 4 for c in set(colors)):
        return None

    left, top, right, bottom = model["bounds"]
    configurations = []
    for ri in movable:
        r = model["rails"][ri]
        x0, y0, _ = state[0][ri]
        horizontal = r["dx"] != 0
        p0 = y0 if horizontal else x0
        low, high = ((top, bottom - 6) if horizontal
                     else (left, right - 6))
        perpendiculars = [p for p in range(low, high + 1)
                          if (p - p0) % 6 == 0]
        configs = []
        for p in perpendiculars:
            x, y = (r["axis"], p) if horizontal else (p, r["axis"])
            for slots in r["goal_slots"]:
                if not slots:
                    continue
                n = slots[-1] + 1
                good = True
                for j in range(1, n):
                    q = (x + 6 * j * r["dx"],
                         y + 6 * j * r["dy"])
                    if q in model["obstacles"]:
                        good = False
                        break
                if not good:
                    continue
                occupied = {
                    (x + 6 * j * r["dx"], y + 6 * j * r["dy"]): c
                    for j, c in zip(slots, r["target"])
                }
                configs.append((occupied, (x, y, n)))
        configurations.append(configs)

    ca, cb = configurations
    if len(ca) * len(cb) > 350000:
        return None
    wanted = Counter(colors)
    positions_list = []
    controller_list = []
    unique = set()
    for cells_a, controller_a in ca:
        for cells_b, controller_b in cb:
            if any(p in cells_a and cells_a[p] != c
                   for p, c in cells_b.items()):
                continue
            cells = dict(cells_a)
            cells.update(cells_b)
            if Counter(cells.values()) != wanted:
                continue
            by_color = {}
            for p, c in cells.items():
                by_color.setdefault(c, []).append(p)
            points = tuple(p for c in sorted(wanted)
                           for p in sorted(by_color[c]))
            key = (points, controller_a, controller_b)
            if key in unique:
                continue
            unique.add(key)
            positions_list.append(points)
            controller_list.append((controller_a, controller_b))
            if len(positions_list) > 6500:
                return None
    if not positions_list:
        return None

    goal_positions = np.asarray(positions_list, dtype=np.float32) / 6.0
    goal_controllers = np.asarray(controller_list, dtype=np.float32)
    goal_controllers[:, :, :2] /= 6.0
    group_permutations = [
        (indices, list(itertools.permutations(indices)))
        for c, indices in model["groups"]
    ]
    return {
        "positions": goal_positions,
        "controllers": goal_controllers,
        "groups": group_permutations,
        "rails": movable,
        "size": len(positions_list),
    }


def _make_model(o, handles, beads):
    bounds = _board(o)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    upper = sorted((h for h in handles if h[1] < 53),
                   key=lambda h: (h[3], h[2], h[1], h[0]))
    lower = [h for h in handles if h[1] >= 53]
    if not upper:
        return None

    rails = []
    coords = []
    covered = set()
    for h in upper:
        x, y, d, key, active = h
        dx, dy = DIRS[d]
        n = _rail_length(o, h)
        ref = next((t for t in lower if t[3] == key), None)
        target = None
        if ref is not None:
            tx, ty, td = ref[:3]
            tdx, tdy = DIRS[td]
            reference_length = _rail_length(o, ref)
            reference_cells = {
                (tx + 6 * j * tdx, ty + 6 * j * tdy): j
                for j in range(reference_length)
            }
            seq = [(reference_cells[(bx, by)], c)
                   for bx, by, c in beads
                   if (bx, by) in reference_cells]
            target = tuple(c for j, c in sorted(seq))

        guide_values = []
        if dx:
            if 0 <= x + 2 < 63:
                for p in range(53):
                    va, vb = int(o[p, x + 2]), int(o[p, x + 3])
                    if va == vb and va in (2, 3):
                        guide_values.append(p)
            fallback = (top + 2, bottom - 3)
        else:
            if 0 <= y + 2 < 63:
                for p in range(64):
                    va, vb = int(o[y + 2, p]), int(o[y + 3, p])
                    if va == vb and va in (2, 3):
                        guide_values.append(p)
            fallback = (left + 2, right - 3)
        guide = ((min(guide_values), max(guide_values))
                 if guide_values else fallback)

        rails.append({
            "key": key, "d": d, "dx": dx, "dy": dy,
            "axis": x if dx else y,
            "guide": guide,
            "target": target,
            "movable": target is not None and len(target) > 0,
            "cache": {},
        })
        coords.append((x, y, n))
        for j in range(n):
            covered.add((x + j * 6 * dx, y + j * 6 * dy, dx != 0))

    colors = sorted(c for x, y, c in beads if y < 53)
    positions = _canonical_positions(beads, colors)
    if positions is None:
        return None

    obstacles = set()
    extras = []
    for y in range(top, bottom - 5, 6):
        for x in range(left, right - 5, 6):
            if np.all(o[y:y + 6, x:x + 6] == 5):
                obstacles.add((x, y))
            for horizontal in (True, False):
                if ((x, y, horizontal) not in covered and
                        _segment_visible(o, x, y, horizontal)):
                    extras.append((x, y, 1 if horizontal else 0,
                                   0 if horizontal else 1))

    for r in rails:
        if not r["movable"]:
            continue
        choices = [[i for i, c in enumerate(colors) if c == wanted]
                   for wanted in r["target"]]
        assignments = []
        if all(choices):
            for ids in itertools.product(*choices):
                if len(set(ids)) == len(ids):
                    assignments.append(ids)
        r["assignments"] = assignments
        dx, dy = r["dx"], r["dy"]
        if dx:
            js = [(x - r["axis"]) * dx // 6
                  for x in range(left, right - 5, 6)]
        else:
            js = [(y - r["axis"]) * dy // 6
                  for y in range(top, bottom - 5, 6)]
        js = sorted(j for j in js if j >= 0)
        r["goal_slots"] = list(
            itertools.combinations(js, len(r["target"])))

    movable = [i for i, r in enumerate(rails) if r["movable"]]
    if not movable:
        return None
    selected = next((i for i, h in enumerate(upper)
                     if h[4] and i in movable), movable[0])
    model = {
        "bounds": bounds, "rails": rails, "colors": colors,
        "obstacles": obstacles, "extras": extras,
        "movable": movable,
        "groups": [(c, [i for i, v in enumerate(colors) if v == c])
                   for c in sorted(set(colors))],
        "heuristic_cache": {},
    }
    state = (tuple(coords), positions)
    model["joint"] = _joint_goals(model, state)
    return model, state, selected


def _observe_state(o, handles, beads, model, old_selected):
    coords = []
    selected = None
    used = set()
    for i, r in enumerate(model["rails"]):
        candidates = [(j, h) for j, h in enumerate(handles)
                      if j not in used and h[1] < 53 and
                      h[3] == r["key"] and h[2] == r["d"]]
        if not candidates:
            return None
        j, h = candidates[0]
        used.add(j)
        coords.append((h[0], h[1], _rail_length(o, h)))
        if h[4] and i in model["movable"]:
            selected = i
    positions = _canonical_positions(beads, model["colors"])
    if positions is None:
        return None
    if selected is None:
        selected = old_selected
    return (tuple(coords), positions), selected


def _layout(model, state):
    coords, positions = state
    segments = []
    own = []
    lookup = {}
    for i, ((x, y, n), r) in enumerate(zip(coords, model["rails"])):
        ids = []
        dx, dy = r["dx"], r["dy"]
        for j in range(n):
            k = len(segments)
            sx, sy = x + j * 6 * dx, y + j * 6 * dy
            segments.append((sx, sy, dx, dy))
            lookup[(sx, sy, dx != 0)] = k
            ids.append(k)
        own.append(ids)
    for x, y, dx, dy in model["extras"]:
        k = len(segments)
        segments.append((x, y, dx, dy))
        lookup.setdefault((x, y, dx != 0), k)
    return segments, own, lookup, {p: i for i, p in enumerate(positions)}


def _transition(model, state, ri, action, layout=None):
    coords, positions = state
    r = model["rails"][ri]
    mx, my = MOVES[action]
    dx, dy = r["dx"], r["dy"]
    x, y, length = coords[ri]
    forward = (mx, my) == (dx, dy)
    backward = (mx, my) == (-dx, -dy)
    if backward and length == 1:
        return None
    if not forward and not backward:
        midpoint = y + 2 + 3 * my if dx else x + 2 + 3 * mx
        lo, hi = r["guide"]
        if not lo <= midpoint <= hi:
            return None

    segments, own, lookup, bead_at = layout or _layout(model, state)
    left, top, right, bottom = model["bounds"]
    obstacles = model["obstacles"]
    step_x, step_y = 6 * mx, 6 * my

    def blocked(px, py):
        return (px < left or py < top or px + 6 > right or
                py + 6 > bottom or (px, py) in obstacles)

    if forward:
        sx, sy = segments[own[ri][-1]][:2]
        if blocked(sx + step_x, sy + step_y):
            return None

    moved_beads = set()
    moved_segments = set()

    def bead_move(bi, driver=None):
        if bi in moved_beads:
            return True
        bx, by = positions[bi]
        nx, ny = bx + step_x, by + step_y
        if blocked(nx, ny):
            return False
        perpendicular_orientation = mx == 0
        for px, py in ((bx, by), (nx, ny)):
            s = lookup.get((px, py, perpendicular_orientation))
            if s is not None and s != driver:
                return False
        other = bead_at.get((nx, ny))
        if other is not None and other != bi:
            if not bead_move(other):
                return False
        moved_beads.add(bi)
        return True

    def segment_move(si):
        if si in moved_segments:
            return True
        sx, sy, sdx, sdy = segments[si]
        nx, ny = sx + step_x, sy + step_y
        if blocked(nx, ny):
            if not ((sdx, sdy) == (-mx, -my) or blocked(sx, sy)):
                return False
        for px, py in ((sx, sy), (nx, ny)):
            bi = bead_at.get((px, py))
            if bi is not None and not bead_move(bi, si):
                if (sdx == 0) != (mx == 0):
                    return False
        moved_segments.add(si)
        return True

    ids = own[ri][1:] if backward else own[ri]
    for si in ids:
        success = segment_move(si)
        if not success and not forward and not backward:
            return None

    new_coords = list(coords)
    if forward:
        new_coords[ri] = (x, y, length + 1)
    elif backward:
        new_coords[ri] = (x, y, length - 1)
    else:
        new_coords[ri] = (x + step_x, y + step_y, length)

    new_positions = list(positions)
    for bi in moved_beads:
        bx, by = positions[bi]
        new_positions[bi] = (bx + step_x, by + step_y)
    for c, indices in model["groups"]:
        ordered = sorted(new_positions[i] for i in indices)
        for i, p in zip(indices, ordered):
            new_positions[i] = p
    result = (tuple(new_coords), tuple(new_positions))
    return None if result == state else result


def _sequences(model, state):
    coords, positions = state
    bead_at = {p: model["colors"][i] for i, p in enumerate(positions)}
    result = {}
    for ri in model["movable"]:
        x, y, n = coords[ri]
        r = model["rails"][ri]
        dx, dy = r["dx"], r["dy"]
        result[ri] = tuple(
            bead_at[(x + 6 * j * dx, y + 6 * j * dy)]
            for j in range(n)
            if (x + 6 * j * dx, y + 6 * j * dy) in bead_at)
    return result


def _goal(model, state):
    sequences = _sequences(model, state)
    return all(sequences[i][:len(model["rails"][i]["target"])] ==
               model["rails"][i]["target"]
               for i in model["movable"])


def _prefix_edit(sequence, target):
    old = list(range(len(target) + 1))
    best = old[-1]
    for a in sequence:
        new = [old[0] + 1]
        for j, b in enumerate(target, 1):
            new.append(min(old[j] + 1, new[j - 1] + 1,
                           old[j - 1] + (a != b)))
        best = min(best, new[-1])
        old = new
    return best


def _joint_heuristic(model, state):
    joint = model["joint"]
    goals = joint["positions"]
    current = np.asarray(state[1], dtype=np.float32) / 6.0
    total = np.zeros(joint["size"], dtype=np.float32)
    worst = np.zeros(joint["size"], dtype=np.float32)

    for indices, permutations in joint["groups"]:
        best_sum = None
        best_max = None
        destinations = goals[:, indices, :]
        for permutation in permutations:
            distance = np.abs(destinations - current[list(permutation)])
            distance = distance[:, :, 0] + distance[:, :, 1]
            ds = np.sum(distance, axis=1)
            dm = np.max(distance, axis=1)
            if best_sum is None:
                best_sum = ds
                best_max = dm
            else:
                take = ds < best_sum
                best_max = np.where(take, dm, best_max)
                best_sum = np.minimum(best_sum, ds)
        total += best_sum
        worst = np.maximum(worst, best_max)

    controllers = joint["controllers"]
    travel = np.zeros(joint["size"], dtype=np.float32)
    extension = np.zeros(joint["size"], dtype=np.float32)
    for j, ri in enumerate(joint["rails"]):
        x, y, n = state[0][ri]
        r = model["rails"][ri]
        axis = 1 if r["dx"] else 0
        p = (y if axis == 1 else x) / 6.0
        travel += np.abs(controllers[:, j, axis] - p)
        extension += np.maximum(0, controllers[:, j, 2] - n)

    costs = (0.52 * total + 0.30 * worst +
             0.72 * travel + 0.70 * extension)
    return float(np.min(costs))


def _single_heuristic(model, state):
    coords, positions = state
    sequences = _sequences(model, state)
    estimates = []
    for ri in model["movable"]:
        r = model["rails"][ri]
        target = r["target"]
        seq = sequences[ri]
        if seq[:len(target)] == target:
            estimates.append(0.0)
            continue
        x, y, length = coords[ri]
        horizontal = r["dx"] != 0
        cur_perp = (y if horizontal else x) / 6.0
        axis_values = [
            ((p[0] - r["axis"]) * r["dx"] if horizontal else
             (p[1] - r["axis"]) * r["dy"]) // 6
            for p in positions]
        perp_values = [(p[1] if horizontal else p[0]) / 6.0
                       for p in positions]
        best = 100.0
        cache = r["cache"]
        for ids in r["assignments"]:
            pp = [perp_values[i] for i in ids]
            lo, hi = min(pp), max(pp)
            gather = hi - lo + min(abs(cur_perp - lo),
                                   abs(cur_perp - hi))
            if gather >= best:
                continue
            axes = tuple(axis_values[i] for i in ids)
            key = (axes, length)
            axial = cache.get(key)
            if axial is None:
                axial = 100.0
                for slots in r["goal_slots"]:
                    distances = [abs(a - b) for a, b in zip(axes, slots)]
                    cost = (max(distances, default=0) +
                            0.25 * sum(distances) +
                            0.65 * max(0, slots[-1] + 1 - length))
                    axial = min(axial, cost)
                if len(cache) > 80000:
                    cache.clear()
                cache[key] = axial
            best = min(best, gather + axial)
        best = max(best, 1.25 * _prefix_edit(seq, target))
        estimates.append(best)
    if not estimates:
        return 100.0
    largest = max(estimates)
    return largest + 0.45 * (sum(estimates) - largest)


def _heuristic(model, state):
    cache = model["heuristic_cache"]
    value = cache.get(state)
    if value is not None:
        return value
    if model["joint"] is not None:
        value = _joint_heuristic(model, state)
    else:
        value = _single_heuristic(model, state)
    if len(cache) > 150000:
        cache.clear()
    cache[state] = value
    return value


def _plan(model, root, selected, deadline, visits):
    root_key = (root, selected)
    distances = {root_key: 0}
    parents = {}
    counter = itertools.count()
    h0 = _heuristic(model, root)
    weight = 1.85 if model["joint"] is not None else 2.2
    heap = [(weight * h0, h0, 0, next(counter), root, selected)]
    best_key = None
    best_rank = float("inf")
    found = None
    expanded = 0

    while heap:
        if expanded % 16 == 0 and time.perf_counter() >= deadline:
            break
        f, h, g, _, state, sel = heapq.heappop(heap)
        key = (state, sel)
        if distances.get(key) != g:
            continue
        expanded += 1
        if key != root_key and _goal(model, state):
            found = key
            break
        if expanded > 110000 or len(distances) > 220000:
            break
        layout = _layout(model, state)
        rail_order = [sel] + [ri for ri in model["movable"] if ri != sel]
        for ri in rail_order:
            switch_cost = int(ri != sel)
            for action in (4, 3, 1, 2):
                nxt = _transition(model, state, ri, action, layout)
                if nxt is None:
                    continue
                nk = (nxt, ri)
                ng = g + 1 + switch_cost
                if ng >= distances.get(nk, 10 ** 9):
                    continue
                distances[nk] = ng
                parents[nk] = (key, ri, action)
                if _goal(model, nxt):
                    found = nk
                    break
                nh = _heuristic(model, nxt)
                revisit = min(12.0, 3.0 * visits.get(nxt, 0))
                rank = nh + 0.025 * ng + revisit
                if rank < best_rank:
                    best_rank = rank
                    best_key = nk
                heapq.heappush(
                    heap, (ng + weight * nh + revisit,
                           nh, ng, next(counter), nxt, ri))
            if found is not None:
                break
        if found is not None:
            break

    endpoint = found or best_key
    if endpoint is None:
        return []
    edges = []
    while endpoint != root_key:
        previous, ri, action = parents[endpoint]
        edges.append((previous, endpoint, ri, action))
        endpoint = previous
    edges.reverse()
    if found is None:
        edges = edges[:5]

    result = []
    for previous, child, ri, action in edges:
        world, old_sel = previous
        next_world, next_sel = child
        if ri != old_sel:
            x, y, n = world[0][ri]
            result.append(("ACTION6", y + 1, x + 1, world, ri))
        result.append(("ACTION" + str(action), None, None,
                       next_world, next_sel))
    return result


def act(obs, available, memory):
    start = time.perf_counter()
    o = np.asarray(obs)
    allowed = set(available)
    handles, beads = _read_objects(o)

    model = memory.get("model")
    observed = None
    if model is not None:
        observed = _observe_state(
            o, handles, beads, model,
            memory.get("selected", model["movable"][0]))
    if observed is None:
        made = _make_model(o, handles, beads)
        if made is None:
            for action in ("ACTION1", "ACTION4", "ACTION2", "ACTION3"):
                if action in allowed:
                    return action, None, None
            return available[0], None, None
        model, state, selected = made
        memory.clear()
        memory["model"] = model
        memory["visits"] = {}
        memory["plan"] = []
    else:
        state, selected = observed

    memory["selected"] = selected
    expected = memory.get("expected")
    if expected is not None and expected != (state, selected):
        memory["plan"] = []

    visits = memory.setdefault("visits", {})
    if memory.get("last_state") != state:
        visits[state] = visits.get(state, 0) + 1
    memory["last_state"] = state
    if len(visits) > 600:
        visits.clear()
        visits[state] = 1

    plan = memory.get("plan", [])
    if not plan:
        plan = _plan(model, state, selected, start + 8.0, visits)
        memory["plan"] = plan

    while plan:
        action, row, col, next_state, next_selected = plan.pop(0)
        if action not in allowed:
            memory["plan"] = []
            break
        memory["expected"] = (next_state, next_selected)
        memory["selected"] = next_selected
        return action, row, col

    layout = _layout(model, state)
    candidates = []
    for action in (1, 2, 3, 4):
        name = "ACTION" + str(action)
        if name not in allowed:
            continue
        nxt = _transition(model, state, selected, action, layout)
        if nxt is not None:
            score = _heuristic(model, nxt) + 4 * visits.get(nxt, 0)
            candidates.append((score, action, nxt))
    if candidates:
        _, action, nxt = min(candidates, key=lambda z: z[0])
        memory["expected"] = (nxt, selected)
        return "ACTION" + str(action), None, None

    if "ACTION6" in allowed:
        for ri in model["movable"]:
            if ri != selected:
                x, y, n = state[0][ri]
                memory["expected"] = (state, ri)
                memory["selected"] = ri
                return "ACTION6", y + 1, x + 1
    memory.pop("expected", None)
    if "ACTION7" in allowed:
        return "ACTION7", None, None
    for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
        if action in allowed:
            return action, None, None
    return available[0], None, None

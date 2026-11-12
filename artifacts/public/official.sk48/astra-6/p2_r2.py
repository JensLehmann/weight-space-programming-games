import math
import time
import heapq
import itertools
import numpy as np


_DIRS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}
_ROT_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
_RING = (
    (1, 1), (2, 1), (3, 1), (4, 1),
    (1, 2), (4, 2), (1, 3), (4, 3),
    (1, 4), (2, 4), (3, 4), (4, 4),
)
_CENTER = ((2, 2), (3, 2), (2, 3), (3, 3))
_GAP = ((5, 2), (5, 3))


def _rotate_point(x, y, r):
    for _ in range(r):
        x, y = 5 - y, x
    return x, y


def _read_handles(im):
    a = im[:59, :59]
    center = im[2:61, 2:61]
    q = im[1:60, 1:60]

    corners = (
        (a == im[:59, 5:64]) &
        (a == im[5:64, :59]) &
        (a == im[5:64, 5:64])
    )
    centers = (
        (center == im[2:61, 3:62]) &
        (center == im[3:62, 2:61]) &
        (center == im[3:62, 3:62])
    )
    ordinary = corners & centers & (a >= 6) & (center == a)
    ordinary &= ((q == 0) | (q == 4))
    passive = corners & centers & (a == 5) & (center >= 6) & (q == 5)

    answer = []
    for is_passive, mask in ((False, ordinary), (True, passive)):
        ys, xs = np.where(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            outer = int(im[y, x])
            middle = int(im[y + 2, x + 2])
            ring = 5 if is_passive else int(im[y + 1, x + 1])
            if any(int(im[y + yy, x + xx]) != ring for xx, yy in _RING):
                continue

            for r in range(4):
                gap = {_rotate_point(xx, yy, r) for xx, yy in _GAP}
                good = True
                for yy in range(6):
                    for xx in range(6):
                        if (xx, yy) in gap or (xx, yy) in _RING:
                            continue
                        expected = middle if (xx, yy) in _CENTER else outer
                        if int(im[y + yy, x + xx]) != expected:
                            good = False
                            break
                    if not good:
                        break
                if not good:
                    continue

                gv = [int(im[y + yy, x + xx]) for xx, yy in gap]
                if is_passive:
                    if not all(v in (1, 2, 3) for v in gv):
                        continue
                elif all(v == outer for v in gv):
                    continue

                answer.append({
                    "x": x, "y": y, "r": r,
                    "key": ("p" if is_passive else "a", middle),
                    "active": not is_passive and ring == 0,
                    "passive": is_passive,
                })
                break
    return answer


def _read_tiles(im):
    a = im[:61, :61]
    mask = a >= 6
    for yy in range(4):
        for xx in range(4):
            b = im[yy:yy + 61, xx:xx + 61]
            if xx in (0, 3) or yy in (0, 3):
                mask &= b == a
            else:
                mask &= (b == a) | (b == 0)
    ys, xs = np.where(mask)
    return [(x - 1, y - 1, int(im[y, x]))
            for y, x in zip(ys.tolist(), xs.tolist())]


def _strip(im, x, y, horizontal):
    if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
        return False
    if horizontal:
        p = (int(im[y + 2, x]), int(im[y + 3, x]),
             int(im[y + 2, x + 5]), int(im[y + 3, x + 5]))
        return p in ((3, 2, 2, 3), (2, 1, 1, 2))
    p = (int(im[y, x + 2]), int(im[y, x + 3]),
         int(im[y + 5, x + 2]), int(im[y + 5, x + 3]))
    return p in ((2, 3, 3, 2), (1, 2, 2, 1))


def _rod_length(im, h):
    dx, dy = _ROT_DIRS[h["r"]]
    length = 1
    while length < 13:
        x = h["x"] + 6 * length * dx
        y = h["y"] + 6 * length * dy
        if not _strip(im, x, y, dx != 0):
            break
        length += 1
    return length


def _covered(x, y, handles):
    return any(h["x"] <= x < h["x"] + 6 and
               h["y"] <= y < h["y"] + 6 for h in handles)


def _rail_edges(im, h, bx, by, width, height, handles):
    horizontal_rod = h["r"] in (0, 2)
    n = height if horizontal_rod else width
    origin = by if horizontal_rod else bx
    cross = h["x"] + 2 if horizontal_rod else h["y"] + 2
    coverage = set()

    def coordinates(p, offset):
        if horizontal_rod:
            return cross + offset, p
        return p, cross + offset

    # Visible portions of the rail.
    for p in range(53 if horizontal_rod else 64):
        vals = []
        visible = True
        for offset in (0, 1):
            x, y = coordinates(p, offset)
            if not (0 <= x < 64 and 0 <= y < 53):
                visible = False
                break
            if _covered(x, y, handles):
                visible = False
                break
            vals.append(int(im[y, x]))
        if visible and len(vals) == 2 and all(v in (2, 3) for v in vals):
            coverage.add(p)

    # Fit the eight-pixel rail template through parts hidden by handles.
    pattern = (2, 2, 3, 3, 3, 3, 2, 2)
    for k in range(-2, n + 2):
        start = origin + 2 + 6 * k
        good, seen = True, 0
        for j, expected in enumerate(pattern):
            for offset in (0, 1):
                x, y = coordinates(start + j, offset)
                if not (0 <= x < 64 and 0 <= y < 53):
                    good = False
                    break
                if _covered(x, y, handles):
                    continue
                seen += 1
                if int(im[y, x]) != expected:
                    good = False
                    break
            if not good:
                break
        if good and seen >= 2:
            coverage.update(range(start, start + 8))

    edges = {}
    for lane in range(-1, n + 1):
        for sign in (-1, 1):
            dest = lane + sign
            if not (-1 <= dest <= n):
                continue
            query = origin + 6 * lane + 2 + 3 * sign
            if query in coverage:
                action = (1 if sign < 0 else 2) if horizontal_rod else (
                    3 if sign < 0 else 4)
                edges[lane, action] = dest
    return edges


def _geometry(model, arm, value):
    cache = model["arms"][arm]["geometry"]
    if value in cache:
        return cache[value]
    meta = model["arms"][arm]
    lane, length = value // 16 - 2, value % 16
    dx, dy = meta["direction"]
    x, y = (meta["fixed"], lane) if dx else (lane, meta["fixed"])
    cells = tuple((y + dy * i + 2) * model["stride"] + x + dx * i + 2
                  for i in range(length))
    mask = 0
    for p in cells:
        mask |= 1 << p
    result = (cells, mask)
    cache[value] = result
    return result


def _make_distance_tables(base, count, allowed_lanes=None):
    if count > base and allowed_lanes is None:
        return None
    size = base ** count
    if size > 22000:
        return None

    sources = np.indices((base,) * count, dtype=np.int16)
    sources = sources.reshape(count, -1).T

    if allowed_lanes is not None:
        targets = np.asarray(allowed_lanes, dtype=np.int16)
        if len(targets) == 0:
            return None
        costs = np.abs(sources[:, None, :] - targets[None, :, None])
        costs = costs.sum(axis=2).astype(np.float32)
        table = []
        for lane in range(base):
            z = costs + 0.7 * np.abs(targets - lane)[None, :]
            table.append(z.min(axis=1).tolist())
        return table

    slots = np.asarray(list(itertools.combinations(range(base), count)),
                       dtype=np.int16)
    if len(slots) == 0:
        return None
    costs = np.abs(sources[:, None, :] - slots[None, :, :])
    costs = costs.sum(axis=2).astype(np.float32)
    table = [None]
    for length in range(1, base + 2):
        extension = 0.7 * np.maximum(0, slots[:, -1] + 2 - length)
        table.append((costs + extension[None, :]).min(axis=1).tolist())
    return table


def _extract_state(model, im, handles, tiles):
    av = []
    active = None
    used = set()
    for i, meta in enumerate(model["arms"]):
        candidates = [(j, h) for j, h in enumerate(handles)
                      if j not in used and h["y"] < 53 and
                      h["key"] == meta["key"] and h["r"] == meta["r"]]
        if not candidates:
            raise ValueError("Missing arm")
        j, h = min(candidates, key=lambda z:
                   abs(z[1]["x"] - meta["initial_x"]) +
                   abs(z[1]["y"] - meta["initial_y"]))
        used.add(j)
        xx = (h["x"] - model["bx"]) // 6
        yy = (h["y"] - model["by"]) // 6
        lane = yy if meta["direction"][0] else xx
        av.append((lane + 2) * 16 + _rod_length(im, h))
        if h["active"] and i in model["controlled"]:
            active = i

    by_color = {}
    for x, y, color in tiles:
        if not (model["bx"] <= x < model["bx"] + 6 * model["w"] and
                model["by"] <= y < model["by"] + 6 * model["h"]):
            continue
        if (x - model["bx"]) % 6 or (y - model["by"]) % 6:
            continue
        xx, yy = (x - model["bx"]) // 6, (y - model["by"]) // 6
        p = (yy + 2) * model["stride"] + xx + 2
        by_color.setdefault(color, []).append(p)

    positions = []
    for color, count in model["color_counts"]:
        pp = sorted(by_color.get(color, []))
        if len(pp) != count:
            raise ValueError("Piece count changed")
        positions.extend(pp)
    if active is None:
        active = model.get("last_active", model["controlled"][0])
    model["last_active"] = active
    return tuple(av), tuple(positions), active


def _build_model(im, handles, tiles):
    board_handles = [h for h in handles if h["y"] < 53]
    ui_handles = [h for h in handles if h["y"] >= 53]

    mask = im[:53] == 4
    mask = mask.copy()
    for h in board_handles:
        mask[max(0, h["y"]):min(53, h["y"] + 6),
             max(0, h["x"]):min(64, h["x"] + 6)] = False
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No board")
    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max()) + 1 - bx, int(ys.max()) + 1 - by
    if bw % 6 or bh % 6:
        raise ValueError("Unrecognized board geometry")
    w, h = bw // 6, bh // 6
    stride = w + 4

    model = {
        "bx": bx, "by": by, "w": w, "h": h, "stride": stride,
        "arms": [], "controlled": [], "goals": [],
        "hcache": {}, "static_h": 0, "static_v": 0,
    }
    valid = set()
    for yy in range(h):
        for xx in range(w):
            x, y = bx + 6 * xx, by + 6 * yy
            black = all(int(im[y + dy, x + dx]) == 5
                        for dx, dy in ((0, 0), (5, 0), (0, 5), (5, 5)))
            if not black:
                valid.add((yy + 2) * stride + xx + 2)
    model["valid"] = valid

    board_tiles = {}
    for x, y, color in tiles:
        if (bx <= x < bx + bw and by <= y < by + bh and
                (x - bx) % 6 == 0 and (y - by) % 6 == 0):
            board_tiles.setdefault(color, []).append(
                ((y - by) // 6 + 2) * stride + (x - bx) // 6 + 2)
    model["color_counts"] = [(c, len(pp)) for c, pp in sorted(board_tiles.items())]
    colors, groups, color_indices = [], [], {}
    for color, count in model["color_counts"]:
        start = len(colors)
        colors.extend([color] * count)
        groups.append((start, start + count))
        color_indices[color] = tuple(range(start, start + count))
    model["colors"], model["groups"] = tuple(colors), groups

    ui_tiles = {(x, y): color for x, y, color in tiles if y >= 53}
    for source in board_handles:
        dx, dy = _ROT_DIRS[source["r"]]
        xx, yy = (source["x"] - bx) // 6, (source["y"] - by) // 6
        lane = yy if dx else xx
        fixed = xx if dx else yy
        edges = _rail_edges(im, source, bx, by, w, h, board_handles)
        reference = next((u for u in ui_handles
                          if u["key"] == source["key"] and not u["passive"]), None)
        meta = {
            "key": source["key"], "r": source["r"],
            "direction": (dx, dy), "fixed": fixed,
            "initial_x": source["x"], "initial_y": source["y"],
            "edges": edges, "geometry": {},
        }
        index = len(model["arms"])
        model["arms"].append(meta)
        if reference is None:
            continue

        rdx, rdy = _ROT_DIRS[reference["r"]]
        goal = []
        for k in range(1, 12):
            p = (reference["x"] + 6 * rdx * k,
                 reference["y"] + 6 * rdy * k)
            if p not in ui_tiles:
                break
            goal.append(ui_tiles[p])
        if not goal:
            continue

        model["controlled"].append(index)
        meta["click"] = (reference["y"] + 1, reference["x"] + 1)
        reachable = {lane}
        todo = [lane]
        while todo:
            u = todo.pop()
            for action in range(1, 5):
                v = edges.get((u, action))
                if v is not None and v not in reachable:
                    reachable.add(v)
                    todo.append(v)

        axis_base, lane_base = (w, h) if dx else (h, w)
        allowed_lanes = sorted(v for v in reachable if 0 <= v < lane_base)
        if not allowed_lanes:
            allowed_lanes = [max(0, min(lane_base - 1, lane))]

        axis_lookup, lane_lookup = {}, {}
        for p in valid:
            px, py = p % stride - 2, p // stride - 2
            if dx:
                axis_lookup[p] = px if dx > 0 else w - 1 - px
                lane_lookup[p] = py
            else:
                axis_lookup[p] = py if dy > 0 else h - 1 - py
                lane_lookup[p] = px

        repeated_slots = []
        choice_groups = []
        for color in sorted(set(goal)):
            slots = tuple(j for j, c in enumerate(goal) if c == color)
            ids = color_indices.get(color, ())
            if len(ids) < len(slots):
                raise ValueError("Impossible target")
            choice_groups.append((slots, list(itertools.combinations(ids, len(slots)))))
            if len(slots) > 1:
                repeated_slots.append(slots)

        assignments = []
        products = itertools.product(*(options for _, options in choice_groups))
        for choices in products:
            assignment = [0] * len(goal)
            for (slots, _), selected in zip(choice_groups, choices):
                for slot, piece in zip(slots, selected):
                    assignment[slot] = piece
            assignments.append(tuple(assignment))
            if len(assignments) >= 512:
                break

        g = {
            "arm": index, "word": tuple(goal),
            "assignments": assignments, "repeated": repeated_slots,
            "axis": axis_lookup, "lane": lane_lookup,
            "ab": axis_base, "lb": lane_base,
            "allowed": allowed_lanes,
            "axis_table": _make_distance_tables(axis_base, len(goal)),
            "lane_table": _make_distance_tables(lane_base, len(goal), allowed_lanes),
            "axis_cache": {}, "lane_cache": {},
            "slots": list(itertools.combinations(range(axis_base), len(goal))),
        }
        model["goals"].append(g)

    if not model["controlled"]:
        raise ValueError("No controllable arm")

    state = _extract_state(model, im, handles, tiles)

    # Include any visible, unowned stationary bar segments.
    owned_h = owned_v = 0
    for i, value in enumerate(state[0]):
        _, mm = _geometry(model, i, value)
        if model["arms"][i]["direction"][0]:
            owned_h |= mm
        else:
            owned_v |= mm
    for yy in range(h):
        for xx in range(w):
            p = (yy + 2) * stride + xx + 2
            bit = 1 << p
            x, y = bx + 6 * xx, by + 6 * yy
            if _strip(im, x, y, True) and not (owned_h & bit):
                model["static_h"] |= bit
            if _strip(im, x, y, False) and not (owned_v & bit):
                model["static_v"] |= bit
    return model, state


def _context(model, state):
    av, positions, _ = state
    geometries = [_geometry(model, i, v) for i, v in enumerate(av)]
    occupancy = {p: i for i, p in enumerate(positions)}
    return geometries, occupancy


def _move(model, state, arm, action, context=None):
    av, positions, _ = state
    meta = model["arms"][arm]
    mx, my = _DIRS[action]
    dx, dy = meta["direction"]
    lane, length = av[arm] // 16 - 2, av[arm] % 16
    delta = mx + my * model["stride"]
    if context is None:
        context = _context(model, state)
    geometries, occupancy = context
    cells, own_full_mask = geometries[arm]
    parallel = mx * dy == my * dx
    retract = parallel and (mx, my) == (-dx, -dy)

    if parallel:
        if retract:
            if length <= 1:
                return None
            segment_cells = cells[1:]
            new_value = av[arm] - 1
        else:
            if cells[-1] + delta not in model["valid"]:
                return None
            segment_cells = cells
            new_value = av[arm] + 1
    else:
        new_lane = meta["edges"].get((lane, action))
        if new_lane is None:
            return None
        segment_cells = cells
        new_value = (new_lane + 2) * 16 + length

    move_is_horizontal = mx != 0
    foreign = model["static_v"] if move_is_horizontal else model["static_h"]
    own_perpendicular = 0
    for i, (_, mask) in enumerate(geometries):
        rod_horizontal = model["arms"][i]["direction"][0] != 0
        if rod_horizontal == move_is_horizontal:
            continue
        if i == arm:
            own_perpendicular = mask
        else:
            foreign |= mask
    if retract and own_perpendicular:
        own_perpendicular &= ~(1 << cells[0])

    moving = set()
    valid = model["valid"]

    def move_piece(i, ignore=-1):
        if i in moving:
            return True
        p = positions[i]
        dest = p + delta
        if dest not in valid:
            return False
        bits = (1 << p) | (1 << dest)
        if foreign & bits:
            return False
        own_collision = own_perpendicular & bits
        if ignore >= 0:
            own_collision &= ~(1 << ignore)
        if own_collision:
            return False
        other = occupancy.get(dest)
        if other is not None and not move_piece(other):
            return False
        moving.add(i)
        return True

    for p in segment_cells:
        dest = p + delta
        if dest not in valid and not (retract or p not in valid):
            if not parallel:
                return None
            continue
        for q in (p, dest):
            i = occupancy.get(q)
            if i is not None:
                success = move_piece(i, p)
                if not success and not parallel:
                    return None

    new_positions = list(positions)
    for i in moving:
        new_positions[i] += delta
    for start, end in model["groups"]:
        if end - start > 1:
            new_positions[start:end] = sorted(new_positions[start:end])
    new_av = list(av)
    new_av[arm] = new_value
    result = (tuple(new_av), tuple(new_positions), arm)
    if result[:2] == state[:2]:
        return None
    return result


def _prefix_edit(word, sequence):
    previous = list(range(len(sequence) + 1))
    for color in word:
        current = [previous[0] + 1]
        for j, actual in enumerate(sequence, 1):
            current.append(min(previous[j] + 1,
                               current[-1] + 1,
                               previous[j - 1] + (0 if color == actual else 2)))
        previous = current
    return min(previous)


def _axis_cost(goal, values, length):
    base = goal["ab"]
    length = max(1, min(base + 1, length))
    table = goal["axis_table"]
    if table is not None:
        code = 0
        for x in values:
            code = code * base + x
        return table[length][code]
    key = tuple(values), length
    cache = goal["axis_cache"]
    if key not in cache:
        cache[key] = min(
            sum(abs(x - y) for x, y in zip(values, slots)) +
            0.7 * max(0, slots[-1] + 2 - length)
            for slots in goal["slots"]
        )
    return cache[key]


def _lane_cost(goal, values, lane):
    base = goal["lb"]
    table = goal["lane_table"]
    if table is not None and 0 <= lane < base:
        code = 0
        for x in values:
            code = code * base + x
        return table[lane][code]
    key = tuple(sorted(values)), lane
    cache = goal["lane_cache"]
    if key not in cache:
        cache[key] = min(
            sum(abs(x - target) for x in values) + 0.7 * abs(lane - target)
            for target in goal["allowed"])
    return cache[key]


def _heuristic(model, state):
    key = state[:2]
    cache = model["hcache"]
    known = cache.get(key)
    if known is not None:
        return known

    av, positions, _ = state
    occupancy = {p: color for p, color in zip(positions, model["colors"])}
    total = 0.0
    solved = True
    for goal in model["goals"]:
        arm = goal["arm"]
        lane, length = av[arm] // 16 - 2, av[arm] % 16
        cells, _ = _geometry(model, arm, av[arm])
        sequence = tuple(occupancy[p] for p in cells if p in occupancy)
        word = goal["word"]
        if sequence[:len(word)] == word:
            continue
        solved = False

        prefix = 0
        for x, y in zip(word, sequence):
            if x != y:
                break
            prefix += 1
        order_cost = 2.2 * _prefix_edit(word, sequence)
        order_cost += 1.8 * (len(word) - prefix)

        axes = [goal["axis"][p] for p in positions]
        lanes = [goal["lane"][p] for p in positions]
        best_shape = float("inf")
        for assignment in goal["assignments"]:
            aa = [axes[i] for i in assignment]
            pp = [lanes[i] for i in assignment]
            for slots in goal["repeated"]:
                ordered = sorted(aa[j] for j in slots)
                for j, value in zip(slots, ordered):
                    aa[j] = value
            value = _axis_cost(goal, aa, length)
            value += _lane_cost(goal, pp, lane)
            if value < best_shape:
                best_shape = value
        total += order_cost + best_shape

    result = (0.0 if solved else total, solved)
    cache[key] = result
    return result


def _reconstruct(parents, node):
    path = []
    while parents[node] is not None:
        previous, arm, action = parents[node]
        path.append((arm, action))
        node = previous
    path.reverse()
    return path


def _search(model, root, deadline, visits, actions, can_switch):
    if len(model["hcache"]) > 160000:
        model["hcache"].clear()

    h0, already_solved = _heuristic(model, root)
    arms = model["controlled"] if can_switch else [root[2]]
    if already_solved:
        context = _context(model, root)
        for arm in arms:
            for action in actions:
                nxt = _move(model, root, arm, action, context)
                if nxt is not None and _heuristic(model, nxt)[1]:
                    return [(arm, action)], True

    counter = 0
    heap = [(2.5 * h0, h0, counter, 0, root)]
    costs = {root: 0}
    parents = {root: None}
    best_node = None
    best_value = float("inf")
    expansions = 0

    while heap and len(costs) < 180000:
        if expansions % 16 == 0 and time.perf_counter() >= deadline:
            break
        _, _, _, cost, state = heapq.heappop(heap)
        if costs.get(state) != cost:
            continue
        expansions += 1
        context = _context(model, state)
        for arm in arms:
            step_cost = 1 + (arm != state[2])
            for action in actions:
                nxt = _move(model, state, arm, action, context)
                if nxt is None:
                    continue
                ng = cost + step_cost
                if ng >= costs.get(nxt, 10 ** 9):
                    continue
                costs[nxt] = ng
                parents[nxt] = (state, arm, action)
                hh, done = _heuristic(model, nxt)
                if done:
                    return _reconstruct(parents, nxt), True

                repetitions = visits.get(nxt[:2], 0)
                partial_value = hh + 0.035 * ng + 3.0 * repetitions
                if nxt[:2] != root[:2] and partial_value < best_value:
                    best_value = partial_value
                    best_node = nxt
                counter += 1
                priority = ng + 2.5 * hh + 4.0 * repetitions
                heapq.heappush(heap, (priority, hh, counter, ng, nxt))

    if best_node is not None:
        return _reconstruct(parents, best_node), False
    return [], False


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 8.4
    im = np.asarray(obs)
    available_set = set(available)
    move_actions = [a for a in range(1, 5)
                    if "ACTION" + str(a) in available_set]

    try:
        handles = _read_handles(im)
        tiles = _read_tiles(im)
        model = memory.get("_model")
        if model is None:
            model, state = _build_model(im, handles, tiles)
            memory["_model"] = model
            memory["_plan"] = []
            memory["_visits"] = {}
        else:
            try:
                state = _extract_state(model, im, handles, tiles)
            except (ValueError, KeyError, IndexError):
                model, state = _build_model(im, handles, tiles)
                memory["_model"] = model
                memory["_plan"] = []
                memory["_visits"] = {}
                memory.pop("_expected", None)

        expected = memory.pop("_expected", None)
        if expected is not None and expected != state:
            memory["_plan"] = []

        visits = memory.setdefault("_visits", {})
        physical = state[:2]
        if memory.get("_last_physical") != physical:
            visits[physical] = visits.get(physical, 0) + 1
        memory["_last_physical"] = physical

        plan = memory.get("_plan", [])
        if not plan:
            plan, complete = _search(
                model, state, deadline, visits, move_actions,
                "ACTION6" in available_set)
            memory["_plan"] = plan
            memory["_complete_plan"] = complete

        if plan:
            arm, action = plan[0]
            if arm != state[2] and "ACTION6" in available_set:
                row, col = model["arms"][arm]["click"]
                memory["_expected"] = (state[0], state[1], arm)
                model["last_active"] = arm
                return "ACTION6", int(row), int(col)

            nxt = _move(model, state, arm, action)
            if nxt is not None:
                memory["_plan"] = plan[1:]
                memory["_expected"] = nxt
                return "ACTION" + str(action), None, None
            memory["_plan"] = []

        # A bounded, observation-based fallback if the main search had no path.
        candidates = []
        context = _context(model, state)
        arms = model["controlled"] if "ACTION6" in available_set else [state[2]]
        for arm in arms:
            for action in move_actions:
                nxt = _move(model, state, arm, action, context)
                if nxt is not None:
                    score = _heuristic(model, nxt)[0]
                    score += 5 * visits.get(nxt[:2], 0)
                    candidates.append((score, arm, action, nxt))
        if candidates:
            _, arm, action, nxt = min(candidates, key=lambda z: z[:3])
            if arm != state[2] and "ACTION6" in available_set:
                memory["_plan"] = [(arm, action)]
                memory["_expected"] = (state[0], state[1], arm)
                row, col = model["arms"][arm]["click"]
                return "ACTION6", int(row), int(col)
            memory["_expected"] = nxt
            return "ACTION" + str(action), None, None

    except (ValueError, KeyError, IndexError, OverflowError):
        memory.pop("_model", None)
        memory.pop("_expected", None)
        memory["_plan"] = []

    if "ACTION7" in available_set:
        return "ACTION7", None, None
    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
        if name in available_set:
            return name, None, None
    if "ACTION6" in available_set:
        return "ACTION6", 0, 0
    return available[0], None, None

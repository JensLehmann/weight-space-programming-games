import numpy as np
import itertools
import collections
import time


_DIRS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


def _components(cells):
    cells = set(cells)
    result = []
    while cells:
        p = cells.pop()
        component = {p}
        todo = [p]
        while todo:
            x, y = todo.pop()
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if q in cells:
                    cells.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _decode(obs, size=None):
    if size is None:
        best = None
        for n in range(10, 33):
            s = 64 // n
            pad = (64 - n * s) // 2
            image = obs[pad:pad + n*s, pad:pad + n*s]
            blocks = image.reshape(n, s, n, s).transpose(0, 2, 1, 3)
            blocks = blocks.reshape(n, n, s*s)
            counts = np.stack([(blocks == c).sum(axis=2)
                               for c in range(16)], axis=2)
            base = counts.argmax(axis=2)
            expected = np.repeat(np.repeat(base, s, axis=0), s, axis=1)
            difference = image != expected
            dot = (64 - n*s + 1) // 2 - pad
            if s == 3:
                difference[dot::s, dot::s] = False
            elif s == 4:
                for dy in (dot + 1, dot + 2):
                    for dx in (dot + 1, dot + 2):
                        difference[dy::s, dx::s] = False
            elif s == 5:
                difference[dot + 2::s, dot + 2::s] = False

            outside = np.ones((64, 64), dtype=bool)
            outside[pad:pad+n*s, pad:pad+n*s] = False
            outside[:, 63] = False
            score = (int(difference.sum()) +
                     2 * int(((obs != 5) & outside).sum()) +
                     0.001 * abs(n - 21))
            if best is None or score < best[0]:
                best = (score, n)
        size = best[1]

    n = size
    s = 64 // n
    pad = (64 - n*s) // 2
    image = obs[pad:pad+n*s, pad:pad+n*s]
    blocks = image.reshape(n, s, n, s).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(n, n, s*s)
    counts = np.stack([(blocks == c).sum(axis=2)
                       for c in range(16)], axis=2)
    base = counts.argmax(axis=2)

    ceilpad = (64 - n*s + 1) // 2
    if s == 3:
        dot = ceilpad
    elif s in (4, 5):
        dot = ceilpad + 2
    else:
        dot = pad + s // 2
    coords = np.minimum(63, dot + np.arange(n)*s)
    centre = obs[np.ix_(coords, coords)]

    axes = {}
    threshold = max(4, n // 3)
    vc = (base == 10).sum(axis=0)
    hc = (base == 10).sum(axis=1)
    if int(vc.max()) >= threshold:
        axes["v"] = {"p": int(vc.argmax())}
    if int(hc.max()) >= threshold:
        axes["h"] = {"p": int(hc.argmax())}

    # The intersection inherits the highest-priority marker of BOTH axes.
    # It therefore cannot be used to identify selection or fixed status.
    selection_scores = []
    for kind, axis in axes.items():
        p = axis["p"]
        values = centre[:, p] if kind == "v" else centre[p, :]
        raw = base[:, p] if kind == "v" else base[p, :]
        exposed_mask = (raw == 10).copy()
        other = "h" if kind == "v" else "v"
        if other in axes:
            crossing = axes[other]["p"]
            if 0 <= crossing < n:
                exposed_mask[crossing] = False
        exposed = values[exposed_mask]
        axis["fixed"] = not bool(np.any(
            (exposed == 0) | (exposed == 9)))
        zeros = int((exposed == 0).sum())
        if zeros:
            selection_scores.append(
                (zeros / max(1, len(exposed)), zeros, kind))
    selected_axis = max(selection_scores)[2] if selection_scores else None

    original = set()
    occupied = set()
    targets = set()
    zeros = set()
    fixed = set()
    for y in range(n):
        for x in range(n):
            b, c = int(base[y, x]), int(centre[y, x])
            on_axis = (("v" in axes and x == axes["v"]["p"]) or
                       ("h" in axes and y == axes["h"]["p"]))
            is_original = b not in (0, 4, 9, 10, 11)
            if b == 4 and c in (0, 5, 9):
                if not on_axis or (c == 0 and selected_axis is None):
                    is_original = True
            if is_original:
                original.add((x, y))
                if c == 0:
                    zeros.add((x, y))
                elif c == 5:
                    fixed.add((x, y))
            if is_original or b == 4:
                occupied.add((x, y))
            if b == 11 or c == 11:
                targets.add((x, y))

    return {
        "n": n, "scale": s, "pad": pad,
        "base": base, "centre": centre,
        "axes": axes, "selected_axis": selected_axis,
        "orig": original, "occ": occupied, "targets": targets,
        "zeros": zeros, "fixed": fixed,
    }


def _grow_targets(seed, frame):
    result = set(seed)
    candidates = {
        p for p in frame["orig"]
        if int(frame["centre"][p[1], p[0]]) == 11
    }
    todo = list(result)
    while todo:
        x, y = todo.pop()
        for q in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if q in candidates and q not in result:
                result.add(q)
                todo.append(q)
    return result


def _make_object(cells, fixed=False):
    minx = min(x for x, y in cells)
    miny = min(y for x, y in cells)
    w = max(x for x, y in cells) - minx + 1
    h = max(y for x, y in cells) - miny + 1
    local = {(x-minx, y-miny) for x, y in cells}
    orientations = []
    dimensions = []
    for _ in range(4):
        dimensions.append((w, h))
        orientations.append(tuple(sorted(
            (x-w//2, y-h//2) for x, y in local)))
        local = {(y, w-1-x) for x, y in local}
        w, h = h, w
    return {
        "cx": minx + dimensions[0][0]//2,
        "cy": miny + dimensions[0][1]//2,
        "o": 0,
        "points": orientations,
        "dims": dimensions,
        "fixed": fixed,
        "rot": 0,
        "tested": 0,
        "possible": [3, 1, 2],
        "flag": 3,
    }


def _world(obj, pose=None):
    if pose is None:
        cx, cy, orientation = obj["cx"], obj["cy"], obj["o"]
    else:
        cx, cy, orientation = pose
    return {(cx+x, cy+y) for x, y in obj["points"][orientation]}


def _coverage(obj, axes, n, flag=None, pose=None):
    if flag is None:
        flag = obj["flag"]
    original = _world(obj, pose)
    seeds = {(x, y) for x, y in original
             if 0 <= x < n and 0 <= y < n}
    result = set(seeds)
    if flag & 1 and "v" in axes:
        a = axes["v"]["p"]
        result |= {(2*a-x, y) for x, y in seeds}
    if flag & 2 and "h" in axes:
        a = axes["h"]["p"]
        result |= {(x, 2*a-y) for x, y in list(result)}
    return {(x, y) for x, y in result
            if 0 <= x < n and 0 <= y < n}


def _infer_selected(frame, memory):
    if frame["selected_axis"] is not None:
        return frame["selected_axis"]
    dots = frame["zeros"]
    if dots:
        choices = []
        for i, obj in enumerate(memory["objects"]):
            if obj["fixed"]:
                continue
            cells = _world(obj)
            common = len(cells & dots)
            choices.append((common * 100 - len(dots - cells), i))
        if choices and max(choices)[0] > 0:
            return max(choices)[1]
    last = memory.get("last", {})
    wanted = last.get("want")
    if last.get("a") == 6 and wanted is not None:
        return wanted
    return memory.get("selected")


def _emit(memory, number, row=None, col=None, wanted=None):
    memory["last"] = {
        "a": number,
        "sel": memory.get("selected"),
        "want": wanted,
    }
    if number == 0:
        return "RESET", None, None
    return "ACTION" + str(number), row, col


def _inventory(frame, memory):
    inventory = memory.setdefault("inventory", {
        "seen": set(), "pieces": [], "count": 0
    })
    if frame["selected_axis"] is not None:
        token = ("axis", frame["selected_axis"])
        cells = set()
    elif frame["zeros"]:
        cells = _grow_targets(frame["zeros"], frame)
        token = ("shape", tuple(sorted(cells)))
    else:
        cells = set()
        token = ("unknown",)

    finished = token in inventory["seen"] or inventory["count"] >= 12
    if not finished:
        inventory["seen"].add(token)
        inventory["count"] += 1
        if cells:
            inventory["pieces"].append((cells, False))
        return _emit(memory, 5)

    pieces = list(inventory["pieces"])
    assigned = set().union(*(p for p, f in pieces)) if pieces else set()

    fixed_cells = frame["fixed"] - assigned
    for component in _components(fixed_cells):
        component = _grow_targets(component, frame) - assigned
        if component:
            pieces.append((component, True))
            assigned |= component

    remaining = frame["orig"] - assigned
    for component in _components(remaining):
        only_targets = all(
            int(frame["centre"][y, x]) == 11 for x, y in component)
        nearest = None
        if only_targets and pieces:
            for i, (p, fixed) in enumerate(pieces):
                if any((x+dx, y+dy) in p
                       for x, y in component
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    nearest = i
                    break
        if nearest is not None:
            pieces[nearest][0].update(component)
        else:
            pieces.append((component, False))

    if not pieces:
        return _emit(memory, 5)

    memory["objects"] = [_make_object(p, fixed)
                         for p, fixed in pieces if p]
    memory["axes"] = {k: dict(v) for k, v in frame["axes"].items()}
    memory["targets"] = sorted(frame["targets"])
    memory["selected"] = _infer_selected(frame, memory)
    memory.pop("inventory", None)
    memory.pop("last", None)
    return None


def _update(frame, memory):
    objects = memory["objects"]
    old_axes = memory["axes"]
    axes = frame["axes"]
    last = memory.get("last", {})
    action = last.get("a", 0)
    selected = last.get("sel")
    dx, dy = _DIRS.get(action, (0, 0))
    moved_x = ("v" in axes and "v" in old_axes and
               axes["v"]["p"] != old_axes["v"]["p"])
    moved_y = ("h" in axes and "h" in old_axes and
               axes["h"]["p"] != old_axes["h"]["p"])

    proposed = []
    options = []
    effects = []
    for i, obj in enumerate(objects):
        cx, cy, orientation = obj["cx"], obj["cy"], obj["o"]
        sx = sy = 0
        if isinstance(selected, int) and selected == i and not obj["fixed"]:
            w, h = obj["dims"][orientation]
            nx, ny = cx+dx, cy+dy
            if (0 <= nx-w//2 and nx-w//2+w <= frame["n"] and
                    0 <= ny-h//2 and ny-h//2+h <= frame["n"]):
                sx, sy = dx, dy
                cx, cy = nx, ny
        effect = (1 if moved_x or sx else 0) | (2 if moved_y or sy else 0)
        effects.append(effect)
        proposed.append((cx, cy))
        if obj["rot"]:
            options.append([1 if effect & obj["rot"] else 0])
        elif effect:
            options.append([0, 1])
        else:
            options.append([0])

    best = None
    for count, rotations in enumerate(itertools.product(*options)):
        if count >= 512:
            break
        union = set()
        selected_cells = None
        for i, (obj, turn) in enumerate(zip(objects, rotations)):
            pose = (*proposed[i], (obj["o"]+turn) % 4)
            cells = _world(obj, pose)
            union |= {(x, y) for x, y in cells
                      if 0 <= x < frame["n"] and 0 <= y < frame["n"]}
            if isinstance(selected, int) and selected == i:
                selected_cells = cells
        error = len(union ^ frame["orig"])
        marker_error = 0
        if action in _DIRS and selected_cells is not None:
            marker_error = len(frame["zeros"] - selected_cells)
        score = error * 20 + marker_error * 10 + sum(rotations) * 0.01
        if best is None or score < best[0]:
            best = (score, rotations, error)

    changed = False
    if best is not None:
        for i, (obj, turn) in enumerate(zip(objects, best[1])):
            obj["cx"], obj["cy"] = proposed[i]
            obj["o"] = (obj["o"] + turn) % 4
            if turn and not obj["rot"] and effects[i]:
                obj["rot"] = 1 if effects[i] & 1 else 2
                changed = True
            elif not turn:
                obj["tested"] |= effects[i]
        memory["bad_frames"] = (
            memory.get("bad_frames", 0) + 1 if best[2] > 3 else 0)

    memory["axes"] = {k: dict(v) for k, v in axes.items()}
    memory["selected"] = _infer_selected(frame, memory)
    if changed:
        memory.pop("goal", None)


def _infer_flags(frame, memory):
    changed = False
    for obj in memory["objects"]:
        valid = []
        for flag in obj["possible"]:
            predicted = _coverage(obj, memory["axes"], frame["n"], flag)
            if predicted <= frame["occ"]:
                valid.append(flag)
        if valid:
            obj["possible"] = valid
            flag = max(valid, key=lambda f: (
                len(_coverage(obj, memory["axes"], frame["n"], f)),
                f == 3, f))
            if flag != obj["flag"]:
                obj["flag"] = flag
                changed = True
    if changed:
        memory.pop("goal", None)


def _reachable(obj, n, initial_orientation):
    start = (obj["cx"], obj["cy"], initial_orientation)
    if obj["fixed"]:
        return {start: 0}
    if not obj["rot"]:
        w, h = obj["dims"][initial_orientation]
        return {
            (x, y, initial_orientation):
                abs(x-obj["cx"]) + abs(y-obj["cy"])
            for y in range(h//2, n-h+h//2+1)
            for x in range(w//2, n-w+w//2+1)
        }

    distances = {start: 0}
    queue = collections.deque([start])
    while queue:
        cx, cy, orientation = queue.popleft()
        w, h = obj["dims"][orientation]
        for action, (dx, dy) in _DIRS.items():
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx-w//2 and nx-w//2+w <= n and
                    0 <= ny-h//2 and ny-h//2+h <= n):
                continue
            turn = bool(dx) if obj["rot"] == 1 else bool(dy)
            successor = (nx, ny, (orientation+int(turn)) % 4)
            if successor not in distances:
                distances[successor] = distances[(cx, cy, orientation)] + 1
                queue.append(successor)
    return distances


def _or_all(values):
    result = 0
    for value in values:
        result |= value
    return result


def _plan(frame, memory):
    deadline = time.perf_counter() + 6.8
    n = frame["n"]
    objects = memory["objects"]
    targets = memory["targets"]
    axes = memory["axes"]
    full = (1 << len(targets)) - 1
    if not full:
        return None

    def axis_values(kind):
        if kind not in axes:
            return [None]
        if axes[kind]["fixed"]:
            return [axes[kind]["p"]]
        return list(range(n))

    current_v = axes.get("v", {}).get("p")
    current_h = axes.get("h", {}).get("p")
    target_set = set(targets)

    def rank(pair):
        v, h = pair
        distance = ((abs(v-current_v) if v is not None else 0) +
                    (abs(h-current_h) if h is not None else 0))
        symmetry = 0
        if v is not None:
            symmetry += sum((2*v-x, y) in target_set for x, y in targets)
        if h is not None:
            symmetry += sum((x, 2*h-y) in target_set for x, y in targets)
        return distance - 0.65 * symmetry, distance

    configurations = sorted(
        itertools.product(axis_values("v"), axis_values("h")), key=rank)
    reachable_cache = {}
    best_cost = float("inf")
    best_goal = None
    total_nodes = 0

    for v, h in configurations:
        if time.perf_counter() >= deadline:
            break
        desired_axes = {}
        axis_cost = 0
        for kind, value in (("v", v), ("h", h)):
            if value is not None:
                desired_axes[kind] = value
                distance = abs(value-axes[kind]["p"])
                axis_cost += distance
                if distance and memory.get("selected") != kind:
                    axis_cost += 1
        if axis_cost >= best_cost:
            continue

        reverse_maps = {}
        defaults = []
        tables = []
        fixed_coverage = 0

        for i, obj in enumerate(objects):
            shift = 0
            if obj["rot"] == 1 and v is not None:
                shift = abs(v-current_v) % 4
            elif obj["rot"] == 2 and h is not None:
                shift = abs(h-current_h) % 4
            initial_orientation = (obj["o"]+shift) % 4
            default = (obj["cx"], obj["cy"], initial_orientation)
            defaults.append(default)
            cache_key = (i, initial_orientation)
            if cache_key not in reachable_cache:
                reachable_cache[cache_key] = _reachable(
                    obj, n, initial_orientation)
            distances = reachable_cache[cache_key]

            flag = obj["flag"]
            if flag not in reverse_maps:
                reverse = {}
                for bit_index, (tx, ty) in enumerate(targets):
                    xs = [tx]
                    ys = [ty]
                    if flag & 1 and v is not None:
                        xs.append(2*v-tx)
                    if flag & 2 and h is not None:
                        ys.append(2*h-ty)
                    for px in set(xs):
                        for py in set(ys):
                            if 0 <= px < n and 0 <= py < n:
                                reverse[(px, py)] = (
                                    reverse.get((px, py), 0) |
                                    (1 << bit_index))
                reverse_maps[flag] = list(reverse.items())
            reverse = reverse_maps[flag]

            masks = {}
            possible_orientations = {p[2] for p in distances}
            for orientation in possible_orientations:
                points = obj["points"][orientation]
                for (px, py), target_bits in reverse:
                    for ox, oy in points:
                        pose = (px-ox, py-oy, orientation)
                        if pose in distances:
                            masks[pose] = masks.get(pose, 0) | target_bits

            if obj["fixed"]:
                fixed_coverage |= masks.get(default, 0)
                tables.append(None)
                continue

            by_mask = {}
            for pose, mask in masks.items():
                distance = distances[pose]
                cost = distance
                if distance and (axis_cost or memory.get("selected") != i):
                    cost += 1
                if axis_cost + cost >= best_cost:
                    continue
                old = by_mask.get(mask)
                if old is None or cost < old[0]:
                    by_mask[mask] = (cost, pose)
            tables.append(by_mask)

        need = full & ~fixed_coverage
        if not need:
            if axis_cost < best_cost:
                best_cost = axis_cost
                best_goal = {"axes": desired_axes, "poses": defaults[:]}
            continue

        movable = [i for i, obj in enumerate(objects) if not obj["fixed"]]
        candidates = []
        for i in movable:
            reduced = {}
            for mask, (cost, pose) in tables[i].items():
                mask &= need
                if not mask:
                    continue
                old = reduced.get(mask)
                if old is None or cost < old[0]:
                    reduced[mask] = (cost, pose)
            candidates.append([(mask, cost, pose)
                               for mask, (cost, pose) in reduced.items()])

        max_counts = [
            max((mask.bit_count() for mask, cost, pose in group), default=0)
            for group in candidates
        ]
        count_needed = need.bit_count()
        if sum(max_counts) < count_needed:
            continue

        total_max = sum(max_counts)
        for j in range(len(candidates)):
            minimum = max(1, count_needed - (total_max-max_counts[j]))
            candidates[j] = [
                candidate for candidate in candidates[j]
                if candidate[0].bit_count() >= minimum
            ]

        union_by_shape = []
        for group in candidates:
            union = 0
            for mask, cost, pose in group:
                union |= mask
            union_by_shape.append(union)
        if (_or_all(union_by_shape) & need) != need:
            continue

        k = len(movable)
        if k > 12:
            continue
        unions = [0] * (1 << k)
        sums = [0] * (1 << k)
        for mask in range(1, 1 << k):
            bit = mask & -mask
            j = bit.bit_length()-1
            unions[mask] = unions[mask ^ bit] | union_by_shape[j]
            sums[mask] = sums[mask ^ bit] + max_counts[j]

        adjacency = [[] for _ in targets]
        for j, group in enumerate(candidates):
            for mask, cost, pose in group:
                bits = mask
                while bits:
                    bit = bits & -bits
                    adjacency[bit.bit_length()-1].append(
                        (j, mask, cost, pose))
                    bits ^= bit

        memo = {}
        local_nodes = 0

        def search(uncovered, remaining, cost, chosen):
            nonlocal best_cost, best_goal, total_nodes, local_nodes
            total_nodes += 1
            local_nodes += 1
            if local_nodes > 25000 or total_nodes > 220000:
                return
            if total_nodes % 128 == 0 and time.perf_counter() >= deadline:
                return
            if axis_cost + cost >= best_cost:
                return
            if not uncovered:
                best_cost = axis_cost + cost
                poses = defaults[:]
                for j, pose in chosen:
                    poses[movable[j]] = pose
                best_goal = {"axes": desired_axes.copy(), "poses": poses}
                return
            if not remaining:
                return
            if uncovered & ~unions[remaining]:
                return
            if sums[remaining] < uncovered.bit_count():
                return
            key = (uncovered, remaining)
            if memo.get(key, float("inf")) <= cost:
                return
            memo[key] = cost

            options = None
            bits = uncovered
            while bits:
                bit = bits & -bits
                possibilities = [
                    item for item in adjacency[bit.bit_length()-1]
                    if remaining & (1 << item[0]) and
                    axis_cost + cost + item[2] < best_cost
                ]
                if not possibilities:
                    return
                if options is None or len(possibilities) < len(options):
                    options = possibilities
                    if len(options) == 1:
                        break
                bits ^= bit

            options.sort(key=lambda item: (
                item[2] / max(1, (item[1] & uncovered).bit_count()),
                -(item[1] & uncovered).bit_count(), item[2]))
            for j, cover, price, pose in options:
                search(uncovered & ~cover, remaining ^ (1 << j),
                       cost+price, chosen+[(j, pose)])
                if local_nodes > 25000:
                    return
                if total_nodes % 128 == 0 and time.perf_counter() >= deadline:
                    return

        search(need, (1 << k)-1, 0, [])
        if total_nodes > 220000:
            break

    return best_goal


def _select(entity, frame, memory):
    if memory.get("selected") == entity:
        return None
    n = frame["n"]
    cells = []
    if isinstance(entity, str):
        axis = memory["axes"].get(entity)
        if axis is None or axis["fixed"]:
            return None
        p = axis["p"]
        cells = ([(p, y) for y in range(n)] if entity == "v"
                 else [(x, p) for x in range(n)])
        other = "h" if entity == "v" else "v"
        cells = [
            (x, y) for x, y in cells
            if (x, y) not in frame["orig"] and
            not (other in memory["axes"] and
                 (y if other == "h" else x) == memory["axes"][other]["p"])
        ]
    else:
        obj = memory["objects"][entity]
        own = _world(obj)
        others = set()
        for j, other in enumerate(memory["objects"]):
            if j != entity and not other["fixed"]:
                others |= _world(other)
        cells = [
            (x, y) for x, y in own-others
            if 0 <= x < n and 0 <= y < n
        ]
        cells.sort(key=lambda p: (
            p not in frame["orig"], p in frame["targets"], p[1], p[0]))

    last = memory.get("last", {})
    repeated_failure = (last.get("a") == 6 and last.get("want") == entity)
    if not cells or repeated_failure:
        return _emit(memory, 5)

    x, y = cells[0]
    row = frame["pad"] + y*frame["scale"] + frame["scale"]//2
    col = frame["pad"] + x*frame["scale"] + frame["scale"]//2
    return _emit(memory, 6, int(row), int(col), entity)


def _path_step(obj, target, n):
    start = (obj["cx"], obj["cy"], obj["o"])
    target = tuple(target)
    if start == target:
        return None
    if not obj["rot"]:
        if start[2] != target[2]:
            return None
        if start[0] != target[0]:
            return 4 if target[0] > start[0] else 3
        return 2 if target[1] > start[1] else 1

    queue = collections.deque([start])
    first = {start: None}
    while queue:
        state = queue.popleft()
        cx, cy, orientation = state
        w, h = obj["dims"][orientation]
        directions = sorted(_DIRS, key=lambda a: (
            abs(cx+_DIRS[a][0]-target[0]) +
            abs(cy+_DIRS[a][1]-target[1])))
        for action in directions:
            dx, dy = _DIRS[action]
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx-w//2 and nx-w//2+w <= n and
                    0 <= ny-h//2 and ny-h//2+h <= n):
                continue
            turn = bool(dx) if obj["rot"] == 1 else bool(dy)
            successor = (nx, ny, (orientation+int(turn)) % 4)
            if successor in first:
                continue
            first[successor] = action if state == start else first[state]
            if successor == target:
                return first[successor]
            queue.append(successor)
    return None


def _execute_goal(frame, memory):
    goal = memory["goal"]
    pending_axes = [
        kind for kind, position in goal["axes"].items()
        if kind in memory["axes"] and memory["axes"][kind]["p"] != position
    ]
    if pending_axes:
        selected = memory.get("selected")
        kind = selected if selected in pending_axes else min(
            pending_axes,
            key=lambda k: abs(goal["axes"][k]-memory["axes"][k]["p"]))
        selection = _select(kind, frame, memory)
        if selection is not None:
            return selection
        increase = goal["axes"][kind] > memory["axes"][kind]["p"]
        action = ((4 if increase else 3) if kind == "v"
                  else (2 if increase else 1))
        return _emit(memory, action)

    pending = [
        i for i, obj in enumerate(memory["objects"])
        if not obj["fixed"] and
        (obj["cx"], obj["cy"], obj["o"]) != tuple(goal["poses"][i])
    ]
    if pending:
        selected = memory.get("selected")
        i = selected if selected in pending else min(
            pending, key=lambda j:
            abs(memory["objects"][j]["cx"]-goal["poses"][j][0]) +
            abs(memory["objects"][j]["cy"]-goal["poses"][j][1]))
        selection = _select(i, frame, memory)
        if selection is not None:
            return selection
        action = _path_step(memory["objects"][i],
                            goal["poses"][i], frame["n"])
        if action is not None:
            return _emit(memory, action)
        memory.pop("goal", None)
        return None

    for kind, axis in memory["axes"].items():
        if not axis["fixed"]:
            selection = _select(kind, frame, memory)
            if selection is not None:
                return selection
            return _emit(memory, 1 if kind == "v" else 3)

    for i, obj in enumerate(memory["objects"]):
        if obj["fixed"]:
            continue
        selection = _select(i, frame, memory)
        if selection is not None:
            return selection
        w, h = obj["dims"][obj["o"]]
        for action, (dx, dy) in _DIRS.items():
            x, y = obj["cx"]+dx-w//2, obj["cy"]+dy-h//2
            if (0 <= x and x+w <= frame["n"] and
                    0 <= y and y+h <= frame["n"]):
                return _emit(memory, action)
    return None


def _probe(frame, memory):
    for i, obj in enumerate(memory["objects"]):
        if obj["rot"]:
            continue
        if all(set(p) == set(obj["points"][0]) for p in obj["points"]):
            obj["tested"] = 3
            continue
        for bit, actions in ((1, (4, 3)), (2, (2, 1))):
            if obj["tested"] & bit:
                continue
            if obj["fixed"]:
                kind = "v" if bit == 1 else "h"
                axis = memory["axes"].get(kind)
                if axis is None or axis["fixed"]:
                    obj["tested"] |= bit
                    continue
                selection = _select(kind, frame, memory)
                if selection is not None:
                    return selection
                positive = axis["p"] < frame["n"]-1
                action = actions[0] if positive else actions[1]
                return _emit(memory, action)

            selection = _select(i, frame, memory)
            if selection is not None:
                return selection
            w, h = obj["dims"][obj["o"]]
            for action in actions:
                dx, dy = _DIRS[action]
                x, y = obj["cx"]+dx-w//2, obj["cy"]+dy-h//2
                if (0 <= x and x+w <= frame["n"] and
                        0 <= y and y+h <= frame["n"]):
                    return _emit(memory, action)
            obj["tested"] |= bit
    return None


def _greedy(frame, memory):
    targets = set(memory["targets"])
    objects = memory["objects"]
    best = None
    for i, obj in enumerate(objects):
        if obj["fixed"]:
            continue
        w, h = obj["dims"][obj["o"]]
        for action, (dx, dy) in _DIRS.items():
            cx, cy = obj["cx"]+dx, obj["cy"]+dy
            if not (0 <= cx-w//2 and cx-w//2+w <= frame["n"] and
                    0 <= cy-h//2 and cy-h//2+h <= frame["n"]):
                continue
            turn = ((obj["rot"] == 1 and dx != 0) or
                    (obj["rot"] == 2 and dy != 0))
            pose = (cx, cy, (obj["o"]+int(turn)) % 4)
            covered = set()
            for j, other in enumerate(objects):
                covered |= _coverage(
                    other, memory["axes"], frame["n"],
                    pose=pose if i == j else None)
            missing = targets-covered
            distance = sum(
                min((abs(x-u)+abs(y-v) for u, v in covered),
                    default=2*frame["n"])
                for x, y in missing)
            score = (len(missing)*100 + distance +
                     (0 if memory.get("selected") == i else 1))
            if best is None or score < best[0]:
                best = (score, i, action)
    if best is not None:
        selection = _select(best[1], frame, memory)
        if selection is not None:
            return selection
        return _emit(memory, best[2])
    return _emit(memory, 5)


def act(obs, available, memory):
    obs = np.asarray(obs)
    frame = _decode(obs, memory.get("size"))
    memory["size"] = frame["n"]

    if "objects" not in memory:
        result = _inventory(frame, memory)
        if result is not None:
            return result
    else:
        _update(frame, memory)

    _infer_flags(frame, memory)

    if (memory.get("bad_frames", 0) >= 2 and
            memory.get("rescans", 0) < 1):
        rescans = memory.get("rescans", 0) + 1
        resets = memory.get("resets", 0)
        size = memory["size"]
        memory.clear()
        memory.update(size=size, rescans=rescans, resets=resets)
        return _inventory(frame, memory)

    if "goal" in memory:
        result = _execute_goal(frame, memory)
        if result is not None:
            return result

    goal = _plan(frame, memory)
    if goal is not None:
        memory["goal"] = goal
        result = _execute_goal(frame, memory)
        if result is not None:
            return result

    result = _probe(frame, memory)
    if result is not None:
        return result

    if "RESET" in available and memory.get("resets", 0) < 1:
        resets = memory.get("resets", 0) + 1
        memory.clear()
        memory["resets"] = resets
        return _emit(memory, 0)

    return _greedy(frame, memory)

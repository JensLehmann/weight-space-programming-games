import time
from collections import deque
from itertools import product


W = H = 21


def _cells(obs):
    base = [[int(obs[3 * y, 3 * x]) for x in range(W)]
            for y in range(H)]
    mid = [[int(obs[3 * y + 1, 3 * x + 1]) for x in range(W)]
           for y in range(H)]
    return base, mid


def _line_positions(base):
    threshold = 7
    vs = [x for x in range(W)
          if sum(base[y][x] == 10 for y in range(H)) >= threshold]
    hs = [y for y in range(H)
          if sum(base[y][x] == 10 for x in range(W)) >= threshold]
    return vs, hs


def _components(points):
    points = set(points)
    out = []
    while points:
        start = points.pop()
        q = [start]
        comp = [start]
        while q:
            x, y = q.pop()
            for p in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if p in points:
                    points.remove(p)
                    q.append(p)
                    comp.append(p)
        out.append(comp)
    return out


def _read_board(obs):
    base, mid = _cells(obs)

    goals = []
    for y in range(H):
        for x in range(W):
            if base[y][x] == 11 or mid[y][x] == 11:
                goals.append((x, y))

    vs, hs = _line_positions(base)
    axes = []
    for kind, coords in (("v", vs), ("h", hs)):
        for coord in coords:
            # At a crossing, the selected mirror's marker can also appear
            # on the other mirror. Do not use crossings to classify either.
            if kind == "v":
                vals = [mid[y][coord] for y in range(H)
                        if y not in hs and base[y][coord] == 10]
            else:
                vals = [mid[coord][x] for x in range(W)
                        if x not in vs and base[coord][x] == 10]

            fixed = vals.count(10) > vals.count(9) + vals.count(0)
            selected = vals.count(0) > 0
            axes.append({
                "kind": kind,
                "pos": coord,
                "fixed": fixed,
                "selected": selected,
            })

    raw = {(x, y) for y in range(H) for x in range(W)
           if base[y][x] == 5}
    parts = []

    for comp in _components(raw):
        labels = {}
        for x, y in comp:
            c = mid[y][x]
            if c == 0:
                labels[(x, y)] = 0
            elif c == 9:
                labels[(x, y)] = 9
            elif c == 5:
                labels[(x, y)] = 5

        kinds = set(labels.values())
        if len(kinds) <= 1:
            parts.append((comp, next(iter(kinds), 9)))
            continue

        allowed = set(comp)
        assigned = dict(labels)
        q = deque(labels)
        while q:
            x, y = q.popleft()
            for p in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if p in allowed and p not in assigned:
                    assigned[p] = assigned[(x, y)]
                    q.append(p)

        for label in (0, 9, 5):
            subset = [p for p in comp if assigned.get(p, 9) == label]
            for sub in _components(subset):
                parts.append((sub, label))

    shapes = []
    for comp, label in parts:
        if not comp:
            continue
        x0 = min(x for x, y in comp)
        y0 = min(y for x, y in comp)
        offsets = sorted((x - x0, y - y0) for x, y in comp)
        shapes.append({
            "x": x0,
            "y": y0,
            "w": max(x for x, y in comp) - x0 + 1,
            "h": max(y for x, y in comp) - y0 + 1,
            "pixels": offsets,
            "fixed": label == 5,
            "selected": label == 0,
            "mode": 0,
        })
    shapes.sort(key=lambda s: (s["y"], s["x"]))

    selected = None
    for i, shape in enumerate(shapes):
        if shape["selected"]:
            selected = ("s", i)
            break
    if selected is None:
        for i, axis in enumerate(axes):
            if axis["selected"] and not axis["fixed"]:
                selected = ("a", i)
                break

    board = {
        "base": base,
        "mid": mid,
        "goals": goals,
        "axes": axes,
        "shapes": shapes,
        "selected": selected,
    }
    _infer_modes(board)
    return board


def _image_points(shape, vx, hy, mode, include_original=True):
    result = set()
    for dx, dy in shape["pixels"]:
        x, y = shape["x"] + dx, shape["y"] + dy
        xs = [x]
        ys = [y]
        if vx is not None and mode != 2:
            xs.append(2 * vx - x)
        if hy is not None and mode != 1:
            ys.append(2 * hy - y)
        for xx in xs:
            for yy in ys:
                if 0 <= xx < W and 0 <= yy < H:
                    if include_original or (xx, yy) != (x, y):
                        result.add((xx, yy))
    return result


def _infer_modes(board):
    shapes = board["shapes"]
    axes = board["axes"]
    vx = next((a["pos"] for a in axes if a["kind"] == "v"), None)
    hy = next((a["pos"] for a in axes if a["kind"] == "h"), None)

    if not shapes or (vx is None and hy is None) or len(shapes) > 6:
        return

    observed = {(x, y) for y in range(H) for x in range(W)
                if board["base"][y][x] == 4}
    actual = set()
    for s in shapes:
        actual.update((s["x"] + dx, s["y"] + dy)
                      for dx, dy in s["pixels"])

    options = []
    for s in shapes:
        per_mode = []
        for mode in (0, 1, 2):
            pts = _image_points(s, vx, hy, mode, False) - actual
            per_mode.append(pts)
        options.append(per_mode)

    best_score = None
    best_modes = None
    for modes in product((0, 1, 2), repeat=len(shapes)):
        predicted = set()
        for i, mode in enumerate(modes):
            predicted.update(options[i][mode])
        score = len(predicted ^ observed)
        key = (score, sum(m != 0 for m in modes))
        if best_score is None or key < best_score:
            best_score = key
            best_modes = modes

    for s, mode in zip(shapes, best_modes):
        s["mode"] = mode


def _move_cost(old, new, selected):
    d = abs(old - new)
    return d + (1 if d and not selected else 0)


def _shape_cost(shape, x, y, selected):
    d = abs(shape["x"] - x) + abs(shape["y"] - y)
    return d + (1 if d and not selected else 0)


def _coverage_at(shape, x, y, vx, hy, goal_index):
    mask = 0
    mode = shape["mode"]
    for dx, dy in shape["pixels"]:
        xx, yy = x + dx, y + dy
        xs = [xx]
        ys = [yy]
        if vx is not None and mode != 2:
            xs.append(2 * vx - xx)
        if hy is not None and mode != 1:
            ys.append(2 * hy - yy)
        for px in xs:
            for py in ys:
                bit = goal_index.get((px, py))
                if bit is not None:
                    mask |= 1 << bit
    return mask


def _candidates(shape, vx, hy, goals, needed, selected):
    placements = {}
    mode = shape["mode"]
    maxx = W - shape["w"]
    maxy = H - shape["h"]

    for gi, (gx, gy) in enumerate(goals):
        bit = 1 << gi
        if not (needed & bit):
            continue

        xs = [gx]
        ys = [gy]
        if vx is not None and mode != 2:
            rx = 2 * vx - gx
            if rx != gx:
                xs.append(rx)
        if hy is not None and mode != 1:
            ry = 2 * hy - gy
            if ry != gy:
                ys.append(ry)

        for px in xs:
            for py in ys:
                for dx, dy in shape["pixels"]:
                    x, y = px - dx, py - dy
                    if 0 <= x <= maxx and 0 <= y <= maxy:
                        key = (x, y)
                        placements[key] = placements.get(key, 0) | bit

    by_mask = {}
    for (x, y), mask in placements.items():
        cost = _shape_cost(shape, x, y, selected)
        old = by_mask.get(mask)
        if old is None or cost < old[0]:
            by_mask[mask] = (cost, x, y)

    result = [(mask, cost, x, y)
              for mask, (cost, x, y) in by_mask.items()]
    result.sort(key=lambda c: (c[1], -c[0].bit_count()))
    return result


def _search(board, seconds=8.0):
    deadline = time.monotonic() + seconds
    goals = board["goals"]
    shapes = board["shapes"]
    axes = board["axes"]
    selected = board["selected"]

    if not goals:
        return None

    vi = next((i for i, a in enumerate(axes) if a["kind"] == "v"), None)
    hi = next((i for i, a in enumerate(axes) if a["kind"] == "h"), None)
    if sum(a["kind"] == "v" for a in axes) > 1:
        return None
    if sum(a["kind"] == "h" for a in axes) > 1:
        return None

    def axis_options(i):
        if i is None:
            return [(None, 0)]
        a = axes[i]
        if a["fixed"]:
            coords = [a["pos"]]
        else:
            coords = range(W if a["kind"] == "v" else H)
        return [(p, _move_cost(a["pos"], p, selected == ("a", i)))
                for p in coords]

    pairs = [(cv + ch, v, h)
             for v, cv in axis_options(vi)
             for h, ch in axis_options(hi)]
    pairs.sort(key=lambda p: p[0])

    full = (1 << len(goals)) - 1
    goal_index = {p: i for i, p in enumerate(goals)}
    movable = [i for i, s in enumerate(shapes) if not s["fixed"]]
    fixed = [i for i, s in enumerate(shapes) if s["fixed"]]

    best_cost = float("inf")
    best_plan = None
    best_partial = None
    best_partial_key = (-1, float("-inf"))

    for axis_cost, vx, hy in pairs:
        if time.monotonic() >= deadline:
            break
        if axis_cost >= best_cost:
            continue

        base_mask = 0
        for i in fixed:
            s = shapes[i]
            base_mask |= _coverage_at(
                s, s["x"], s["y"], vx, hy, goal_index
            )
        needed = full & ~base_mask

        candidates = {}
        union = base_mask
        max_cover = 0

        for i in movable:
            s = shapes[i]
            cs = _candidates(
                s, vx, hy, goals, needed, selected == ("s", i)
            )
            candidates[i] = cs
            u = 0
            m = 0
            for mask, cost, x, y in cs:
                u |= mask
                m = max(m, mask.bit_count())
            union |= u
            max_cover += m

        partial_mask = base_mask
        partial_cost = axis_cost
        partial_places = {}
        remaining_pieces = set(movable)

        while remaining_pieces:
            choice = None
            for i in remaining_pieces:
                for mask, cost, x, y in candidates[i]:
                    gain = (mask & ~partial_mask).bit_count()
                    if gain:
                        key = (gain, -cost)
                        if choice is None or key > choice[0]:
                            choice = (key, i, mask, cost, x, y)
            if choice is None:
                break
            _, i, mask, cost, x, y = choice
            remaining_pieces.remove(i)
            partial_mask |= mask
            partial_cost += cost
            partial_places[i] = (x, y)

        partial_key = (partial_mask.bit_count(), -partial_cost)
        if partial_key > best_partial_key:
            best_partial_key = partial_key
            best_partial = (vx, hy, partial_places)

        if union != full or max_cover < needed.bit_count():
            continue
        if needed == 0:
            best_cost = axis_cost
            best_plan = (vx, hy, {})
            continue

        indices = {}
        for i in movable:
            idx = [[] for _ in goals]
            for candidate in candidates[i]:
                bits = candidate[0]
                while bits:
                    b = bits & -bits
                    idx[b.bit_length() - 1].append(candidate)
                    bits -= b
            indices[i] = idx

        nodes = [0]
        memo = {}

        def dfs(need, remaining, cost, chosen):
            nonlocal best_cost, best_plan
            nodes[0] += 1
            if nodes[0] > 1800 or time.monotonic() >= deadline:
                return
            if cost >= best_cost:
                return
            if not need:
                best_cost = cost
                best_plan = (vx, hy, dict(chosen))
                return
            if not remaining:
                return

            key = (need, remaining)
            old = memo.get(key)
            if old is not None and old <= cost:
                return
            memo[key] = cost

            possible = 0
            cover_bound = 0
            for i in remaining:
                u = 0
                largest = 0
                for mask, _, _, _ in candidates[i]:
                    relevant = mask & need
                    u |= relevant
                    largest = max(largest, relevant.bit_count())
                possible |= u
                cover_bound += largest

            if (possible & need) != need or cover_bound < need.bit_count():
                return

            best_bit = None
            best_count = None
            bits = need
            while bits:
                bit = bits & -bits
                gi = bit.bit_length() - 1
                count = sum(len(indices[i][gi]) for i in remaining)
                if best_count is None or count < best_count:
                    best_count, best_bit = count, gi
                    if count <= 1:
                        break
                bits -= bit

            if not best_count:
                return

            choices = []
            for i in remaining:
                for candidate in indices[i][best_bit]:
                    mask, extra, x, y = candidate
                    if cost + extra < best_cost:
                        choices.append(
                            (extra, -(mask & need).bit_count(),
                             i, mask, x, y)
                        )
            choices.sort()

            for extra, _, i, mask, x, y in choices:
                chosen[i] = (x, y)
                dfs(
                    need & ~mask,
                    tuple(j for j in remaining if j != i),
                    cost + extra,
                    chosen,
                )
                del chosen[i]

        dfs(needed, tuple(movable), axis_cost, {})

    if best_plan is not None:
        return best_plan, True
    if best_partial is not None:
        return best_partial, False
    return None


def _click_cell(x, y):
    return 3 * y + 1, 3 * x + 1


def _make_queue(board, plan):
    vx, hy, places = plan
    shapes = board["shapes"]
    axes = board["axes"]
    selected = board["selected"]

    targets = {}
    for i, a in enumerate(axes):
        p = vx if a["kind"] == "v" else hy
        if p is not None and p != a["pos"] and not a["fixed"]:
            targets[("a", i)] = p

    for i, pos in places.items():
        s = shapes[i]
        if not s["fixed"] and pos != (s["x"], s["y"]):
            targets[("s", i)] = pos

    order = list(targets)
    if selected in order:
        order.remove(selected)
        order.insert(0, selected)

    shape_pos = [(s["x"], s["y"]) for s in shapes]
    axis_pos = [a["pos"] for a in axes]
    queue = []

    def occupied(except_shape=None):
        result = set()
        for j, s in enumerate(shapes):
            if j == except_shape:
                continue
            x, y = shape_pos[j]
            result.update((x + dx, y + dy) for dx, dy in s["pixels"])
        return result

    for obj in order:
        kind, i = obj
        if selected != obj:
            cell = None

            if kind == "s":
                s = shapes[i]
                x, y = shape_pos[i]
                other = occupied(i)
                for dx, dy in s["pixels"]:
                    p = (x + dx, y + dy)
                    if p not in other and 0 <= p[0] < W and 0 <= p[1] < H:
                        cell = p
                        break
                if cell is None and s["pixels"]:
                    dx, dy = s["pixels"][0]
                    cell = (x + dx, y + dy)
            else:
                a = axes[i]
                other = occupied()
                if a["kind"] == "v":
                    x = axis_pos[i]
                    for y in range(H):
                        if (x, y) not in other and not any(
                            j != i and b["kind"] == "h" and axis_pos[j] == y
                            for j, b in enumerate(axes)
                        ):
                            cell = (x, y)
                            break
                else:
                    y = axis_pos[i]
                    for x in range(W):
                        if (x, y) not in other and not any(
                            j != i and b["kind"] == "v" and axis_pos[j] == x
                            for j, b in enumerate(axes)
                        ):
                            cell = (x, y)
                            break

            if cell is None:
                return []
            row, col = _click_cell(*cell)
            queue.append(("ACTION6", row, col, None))
            selected = obj

        if kind == "s":
            x, y = shape_pos[i]
            tx, ty = targets[obj]

            while x != tx:
                d = 1 if tx > x else -1
                queue.append((
                    "ACTION4" if d > 0 else "ACTION3",
                    None, None, ("s", i, d, 0)
                ))
                x += d

            while y != ty:
                d = 1 if ty > y else -1
                queue.append((
                    "ACTION2" if d > 0 else "ACTION1",
                    None, None, ("s", i, 0, d)
                ))
                y += d

            shape_pos[i] = (x, y)
        else:
            p = axis_pos[i]
            target = targets[obj]
            while p != target:
                d = 1 if target > p else -1
                if axes[i]["kind"] == "v":
                    action = "ACTION4" if d > 0 else "ACTION3"
                else:
                    action = "ACTION2" if d > 0 else "ACTION1"
                queue.append((action, None, None, ("a", i, d, 0)))
                p += d
            axis_pos[i] = p

    return queue


def _prediction(board):
    return {
        "shapes": [
            {"x": s["x"], "y": s["y"], "pixels": list(s["pixels"])}
            for s in board["shapes"]
        ],
        "axes": [
            {"kind": a["kind"], "pos": a["pos"]}
            for a in board["axes"]
        ],
    }


def _prediction_matches(obs, pred):
    base, _ = _cells(obs)
    expected = set()
    for s in pred["shapes"]:
        expected.update(
            (s["x"] + dx, s["y"] + dy)
            for dx, dy in s["pixels"]
            if 0 <= s["x"] + dx < W and 0 <= s["y"] + dy < H
        )

    actual = {(x, y) for y in range(H) for x in range(W)
              if base[y][x] == 5}
    if len(expected ^ actual) >= 3:
        return False

    vs, hs = _line_positions(base)
    for a in pred["axes"]:
        found = vs if a["kind"] == "v" else hs
        if found and a["pos"] not in found:
            return False
    return True


def _emit_queued(memory):
    action, row, col, update = memory["queue"].pop(0)
    if update is not None:
        kind, i, dx, dy = update
        if kind == "s":
            s = memory["pred"]["shapes"][i]
            s["x"] += dx
            s["y"] += dy
        else:
            memory["pred"]["axes"][i]["pos"] += dx
    return action, row, col


def act(obs, available, memory):
    if memory.get("queue"):
        if _prediction_matches(obs, memory["pred"]):
            return _emit_queued(memory)
        memory.pop("queue", None)
        memory.pop("pred", None)

    board = _read_board(obs)
    result = _search(board)

    if result is not None:
        plan, complete = result
        queue = _make_queue(board, plan)
        if queue:
            memory["queue"] = queue
            memory["pred"] = _prediction(board)
            return _emit_queued(memory)

        if complete:
            selected = board["selected"]
            if selected and selected[0] == "a":
                axis = board["axes"][selected[1]]
                action = "ACTION1" if axis["kind"] == "v" else "ACTION3"
                return action, None, None

    selected = board["selected"]
    if selected and selected[0] == "s" and board["goals"]:
        s = board["shapes"][selected[1]]
        gx, gy = min(
            board["goals"],
            key=lambda p: abs(p[0] - s["x"]) + abs(p[1] - s["y"]),
        )
        if gx != s["x"]:
            return ("ACTION4" if gx > s["x"] else "ACTION3"), None, None
        if gy != s["y"]:
            return ("ACTION2" if gy > s["y"] else "ACTION1"), None, None

    for name in ("ACTION5", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
        if name in available:
            return name, None, None
    return "ACTION1", None, None

import numpy as np
import itertools
import time
from collections import deque


def _axis_cells(axes):
    cells = set()
    for a in axes:
        if a["kind"] == "v":
            cells.update((a["pos"], y) for y in range(21))
        else:
            cells.update((x, a["pos"]) for x in range(21))
    return cells


def _read(obs):
    raw = np.asarray(obs)[0:63:3, 0:63:3].astype(int)
    mid = np.asarray(obs)[1:63:3, 1:63:3].astype(int)
    original = set(zip(*np.where(raw.T == 5)))
    reflected = set(zip(*np.where(raw.T == 4)))
    targets = set(zip(*np.where((mid.T == 11) | (raw.T == 11))))
    axes = []

    for x in range(21):
        ys = np.where(raw[:, x] == 10)[0]
        if len(ys) >= 6:
            values = [int(mid[y, x]) for y in ys]
            axes.append({
                "kind": "v", "pos": x,
                "fixed": not any(c in (0, 9) for c in values)
            })
    for y in range(21):
        xs = np.where(raw[y, :] == 10)[0]
        if len(xs) >= 6:
            values = [int(mid[y, x]) for x in xs]
            axes.append({
                "kind": "h", "pos": y,
                "fixed": not any(c in (0, 9) for c in values)
            })

    # Ignore crossings when determining whether a mirror is fixed.
    for a in axes:
        values = []
        if a["kind"] == "v":
            crossings = {b["pos"] for b in axes if b["kind"] == "h"}
            for y in range(21):
                if y not in crossings and raw[y, a["pos"]] == 10:
                    values.append(int(mid[y, a["pos"]]))
        else:
            crossings = {b["pos"] for b in axes if b["kind"] == "v"}
            for x in range(21):
                if x not in crossings and raw[a["pos"], x] == 10:
                    values.append(int(mid[a["pos"], x]))
        if values:
            a["fixed"] = not any(v in (0, 9) for v in values)
    return raw, mid, original, reflected, targets, axes


def _components(cells):
    left = set(cells)
    result = []
    while left:
        seed = min(left)
        left.remove(seed)
        component = {seed}
        queue = [seed]
        for x, y in queue:
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if q in left:
                    left.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _make_piece(cells, mid, axes):
    x0 = min(x for x, y in cells)
    y0 = min(y for x, y in cells)
    w = max(x for x, y in cells) - x0 + 1
    h = max(y for x, y in cells) - y0 + 1
    points = tuple(sorted((x - x0, y - y0) for x, y in cells))
    rotations = []
    for _ in range(4):
        rotations.append((w, h, points))
        points = tuple(sorted((y, w - 1 - x) for x, y in points))
        w, h = h, w
    axis_cells = _axis_cells(axes)
    values = [int(mid[y, x]) for x, y in cells
              if (x, y) not in axis_cells and mid[y, x] != 11]
    return {
        "rots": rotations,
        "cx": x0 + rotations[0][0] // 2,
        "cy": y0 + rotations[0][1] // 2,
        "r": 0,
        "fixed": bool(values) and all(v == 5 for v in values),
        "mode": 3,
        "link": None,
        "verified": False,
        "blocked": set()
    }


def _initial_pieces(original, mid, axes):
    axis_cells = _axis_cells(axes)
    selected = {q for q in original
                if q not in axis_cells and mid[q[1], q[0]] == 0}
    groups = _components(original)

    # Selection identifies disconnected parts of the same sprite. Crucially,
    # zeros on an axis do not identify a selected piece.
    selected_groups = [c for c in groups if c & selected]
    if selected_groups:
        joined = set().union(*selected_groups)
        groups = [c for c in groups if not c & selected]
        groups.append(joined)
    return [_make_piece(c, mid, axes) for c in groups if c]


def _cells(piece, cx=None, cy=None, r=None, clipped=True):
    cx = piece["cx"] if cx is None else cx
    cy = piece["cy"] if cy is None else cy
    r = piece["r"] if r is None else r
    w, h, points = piece["rots"][r % 4]
    x0, y0 = cx - w // 2, cy - h // 2
    result = {(x0 + x, y0 + y) for x, y in points}
    if clipped:
        result = {(x, y) for x, y in result
                  if 0 <= x < 21 and 0 <= y < 21}
    return result


def _orbit(point, axes, mode):
    useful = [
        a for a in axes
        if (a["kind"] == "v" and mode & 1)
        or (a["kind"] == "h" and mode & 2)
    ]
    if not useful:
        return (point,)
    if len(useful) <= 2 and len({a["kind"] for a in useful}) == len(useful):
        result = {point}
        for a in useful:
            if a["kind"] == "v":
                result |= {(2 * a["pos"] - x, y) for x, y in list(result)}
            else:
                result |= {(x, 2 * a["pos"] - y) for x, y in list(result)}
        return tuple(result)

    seen = {point}
    queue = deque([(point, 0)])
    while queue:
        (x, y), depth = queue.popleft()
        if depth > 12:
            continue
        for a in useful:
            q = ((2 * a["pos"] - x, y) if a["kind"] == "v"
                 else (x, 2 * a["pos"] - y))
            if q not in seen:
                seen.add(q)
                queue.append((q, depth + 1))
    return tuple(seen)


def _coverage(piece, axes, cx=None, cy=None, r=None):
    original = _cells(piece, cx, cy, r)
    result = set(original)
    for q in original:
        result.update(t for t in _orbit(q, axes, piece["mode"])
                      if 0 <= t[0] < 21 and 0 <= t[1] < 21)
    return result


def _learn_modes(pieces, axes, original, reflected):
    if not pieces or not axes:
        return False
    choices = []
    for p in pieces:
        variants = []
        seen = set()
        for mode in (p["mode"], 3, 1, 2):
            cells = set()
            for q in _cells(p):
                cells.update(t for t in _orbit(q, axes, mode)
                             if 0 <= t[0] < 21 and 0 <= t[1] < 21)
            cells -= original
            key = frozenset(cells)
            if key not in seen:
                seen.add(key)
                variants.append((mode, cells))
        choices.append(variants)

    count = 1
    for c in choices:
        count *= len(c)
    if count > 1000:
        return False

    best_score, best = None, None
    for combination in itertools.product(*choices):
        predicted = set()
        changes = 0
        for p, (mode, cells) in zip(pieces, combination):
            predicted |= cells
            changes += mode != p["mode"]
        score = (len(predicted ^ reflected), changes)
        if best_score is None or score < best_score:
            best_score, best = score, combination

    changed = False
    if best is not None:
        for p, (mode, _) in zip(pieces, best):
            changed |= p["mode"] != mode
            p["mode"] = mode
    return changed


def _selected(pieces, axes, raw, mid, fallback=None):
    axis_cells = _axis_cells(axes)
    zeros = {(x, y) for x in range(21) for y in range(21)
             if (x, y) not in axis_cells
             and mid[y, x] == 0 and raw[y, x] == 5}
    if zeros:
        scores = [len(_cells(p) & zeros) for p in pieces]
        if scores and max(scores):
            return ("p", max(range(len(scores)), key=lambda i: scores[i]))

    # At a crossing, the selected horizontal axis also paints a zero on the
    # vertical axis (and vice versa). Such pixels must not decide selection.
    for i, a in enumerate(axes):
        if a["kind"] == "v":
            crossings = {b["pos"] for b in axes if b["kind"] == "h"}
            if any(y not in crossings
                   and raw[y, a["pos"]] == 10
                   and mid[y, a["pos"]] == 0 for y in range(21)):
                return ("a", i)
        else:
            crossings = {b["pos"] for b in axes if b["kind"] == "v"}
            if any(x not in crossings
                   and raw[a["pos"], x] == 10
                   and mid[a["pos"], x] == 0 for x in range(21)):
                return ("a", i)
    return fallback


def _track(memory, original, detected_axes):
    pieces, axes = memory["pieces"], memory["axes"]
    last = memory.get("last")
    old_axes = [dict(a) for a in axes]
    if len(axes) != len(detected_axes):
        return False, True

    unused = set(range(len(detected_axes)))
    for a in axes:
        possible = [i for i in unused
                    if detected_axes[i]["kind"] == a["kind"]]
        if not possible:
            return False, True
        j = min(possible,
                key=lambda i: abs(detected_axes[i]["pos"] - a["pos"]))
        unused.remove(j)
        a["pos"] = detected_axes[j]["pos"]

    axis_steps = {"v": 0, "h": 0}
    for old, new in zip(old_axes, axes):
        axis_steps[new["kind"]] += abs(new["pos"] - old["pos"])

    old_states = [(p["cx"], p["cy"], p["r"]) for p in pieces]
    hypotheses = []
    for i, p in enumerate(pieces):
        cx, cy, r = old_states[i]
        ex, ey = cx, cy
        moved = last and last.get("kind") == "p" and last.get("i") == i
        if moved:
            ex += last.get("dx", 0)
            ey += last.get("dy", 0)
        expected_r = r
        if p["link"]:
            expected_r = (expected_r + axis_steps[p["link"]]) % 4
            if moved:
                parallel = (last.get("dx", 0) if p["link"] == "v"
                            else last.get("dy", 0))
                expected_r = (expected_r + abs(parallel)) % 4
        centers = [(ex, ey)]
        if moved and (ex, ey) != (cx, cy):
            centers.append((cx, cy))
        options, seen = [], set()
        for center_number, (xx, yy) in enumerate(centers):
            for dr in range(4):
                rr = (expected_r + dr) % 4
                cells = _cells(p, xx, yy, rr)
                key = frozenset(cells)
                if key not in seen:
                    seen.add(key)
                    options.append((xx, yy, rr, cells,
                                    8 * center_number + dr))
        hypotheses.append(options)

    count = 1
    for options in hypotheses:
        count *= len(options)

    best_score, best = None, None
    if count <= 4096:
        for combination in itertools.product(*hypotheses):
            union = set()
            penalty = 0
            for option in combination:
                union |= option[3]
                penalty += option[4]
            score = (len(union ^ original), penalty)
            if best_score is None or score < best_score:
                best_score, best = score, combination
    else:
        best = [min(opts, key=lambda o: (len(o[3] - original), o[4]))
                for opts in hypotheses]
        union = set()
        for option in best:
            union |= option[3]
        best_score = (len(union ^ original), 0)

    learned = False
    for i, (p, option) in enumerate(zip(pieces, best or [])):
        cx, cy, rr = option[:3]
        ox, oy, old_r = old_states[i]
        if rr != old_r and p["link"] is None:
            causes = [k for k in ("v", "h") if axis_steps[k]]
            if not causes and last and last.get("kind") == "p" \
                    and last.get("i") == i:
                if cx != ox and any(a["kind"] == "v" for a in axes):
                    causes = ["v"]
                elif cy != oy and any(a["kind"] == "h" for a in axes):
                    causes = ["h"]
            if len(causes) == 1:
                p["link"] = causes[0]
                learned = True
        p["cx"], p["cy"], p["r"] = cx, cy, rr

    memory["old_axes"] = old_axes
    memory["old_states"] = old_states
    return best_score is not None and best_score[0] == 0, learned


def _after_axis_rotation(piece, axes, new_axes):
    r = piece["r"]
    if piece["link"]:
        r += sum(abs(a["pos"] - b["pos"])
                 for a, b in zip(axes, new_axes)
                 if a["kind"] == piece["link"])
    return r % 4


def _solve(pieces, axes, targets, selected):
    deadline = time.perf_counter() + 7.0
    target_list = sorted(targets)
    full = (1 << len(target_list)) - 1
    best_cost = float("inf")
    best_plan = None
    partial_score, partial_plan = None, None

    domains = [[a["pos"]] if a["fixed"] else list(range(21))
               for a in axes]
    configurations = []
    for coordinates in itertools.product(*domains):
        cost = sum(
            abs(coordinate - a["pos"]) + int(selected != ("a", i))
            for i, (a, coordinate) in enumerate(zip(axes, coordinates))
            if coordinate != a["pos"]
        )
        configurations.append((cost, coordinates))
    configurations.sort()

    for axis_cost, coordinates in configurations:
        if time.perf_counter() > deadline:
            break
        if axis_cost >= best_cost:
            continue
        new_axes = [dict(a, pos=x) for a, x in zip(axes, coordinates)]
        orbits = {
            mode: [_orbit(t, new_axes, mode) for t in target_list]
            for mode in {p["mode"] for p in pieces}
        }
        lists, defaults = [], []
        fixed_mask = 0
        aborted = False

        for i, p in enumerate(pieces):
            after_r = _after_axis_rotation(p, axes, new_axes)
            default = (p["cx"], p["cy"], after_r)
            defaults.append(default)
            if p["fixed"]:
                covered = _coverage(p, new_axes, *default)
                fixed_mask |= sum(1 << j for j, t in enumerate(target_list)
                                  if t in covered)
                lists.append([])
                continue

            placement_masks = {}
            rotations = range(4) if p["link"] else (after_r,)
            for r in rotations:
                w, h, cells = p["rots"][r]
                for bit, orbit in enumerate(orbits[p["mode"]]):
                    b = 1 << bit
                    for tx, ty in orbit:
                        for px, py in cells:
                            x, y = tx - px, ty - py
                            if 0 <= x <= 21 - w and 0 <= y <= 21 - h:
                                key = (x, y, r)
                                placement_masks[key] = \
                                    placement_masks.get(key, 0) | b

            by_mask = {}
            for (x, y, r), mask in placement_masks.items():
                w, h, _ = p["rots"][r]
                cx, cy = x + w // 2, y + h // 2
                dx, dy = cx - p["cx"], cy - p["cy"]
                distance = abs(dx) + abs(dy)
                if p["link"]:
                    parallel = abs(dx) if p["link"] == "v" else abs(dy)
                    natural_r = (after_r + parallel) % 4
                    difference = (r - natural_r) % 4
                    if difference & 1:
                        continue
                    distance += difference
                elif r != after_r:
                    continue
                changed = (cx, cy, r) != default
                click_cost = int(changed and (
                    selected != ("p", i) or any(
                        a["pos"] != b["pos"]
                        for a, b in zip(axes, new_axes))))
                cost = distance + click_cost
                if axis_cost + cost >= best_cost:
                    continue
                candidate = (mask, cost, cx, cy, r)
                previous = by_mask.get(mask)
                if previous is None or cost < previous[1]:
                    by_mask[mask] = candidate

            covered = _coverage(p, new_axes, *default)
            mask = sum(1 << j for j, t in enumerate(target_list)
                       if t in covered)
            if mask:
                candidate = (mask, 0, *default)
                if mask not in by_mask or by_mask[mask][1] > 0:
                    by_mask[mask] = candidate
            lists.append(sorted(
                by_mask.values(), key=lambda c: (-c[0].bit_count(), c[1])))
            if time.perf_counter() > deadline:
                aborted = True
                break
        if aborted:
            break

        remaining = full & ~fixed_mask
        active = [i for i, candidates in enumerate(lists) if candidates]
        possible, capacity = fixed_mask, 0
        for i in active:
            union, maximum = 0, 0
            for c in lists[i]:
                union |= c[0]
                maximum = max(maximum, (c[0] & remaining).bit_count())
            possible |= union
            capacity += maximum

        greedy_remaining = remaining
        greedy_goals = list(defaults)
        greedy_cost = axis_cost
        for i in sorted(active, key=lambda j: -max(
                (c[0] & remaining).bit_count() for c in lists[j])):
            candidate = min(lists[i], key=lambda c: (
                -(c[0] & greedy_remaining).bit_count(), c[1]))
            if candidate[0] & greedy_remaining:
                greedy_goals[i] = candidate[2:]
                greedy_remaining &= ~candidate[0]
                greedy_cost += candidate[1]
        score = (greedy_remaining.bit_count(), greedy_cost)
        if partial_score is None or score < partial_score:
            partial_score = score
            partial_plan = {
                "axes": list(coordinates),
                "pieces": greedy_goals,
                "partial": True
            }

        if not remaining:
            if axis_cost < best_cost:
                best_cost = axis_cost
                best_plan = {
                    "axes": list(coordinates),
                    "pieces": list(defaults),
                    "partial": False
                }
            continue
        if possible != full or capacity < remaining.bit_count():
            continue

        bit_options = []
        for bit in range(len(target_list)):
            b = 1 << bit
            bit_options.append([
                (i, c) for i in active for c in lists[i] if c[0] & b
            ])
        memo = {}
        nodes = [0]

        def search(rem, unused, cost, assigned):
            nonlocal best_cost, best_plan
            nodes[0] += 1
            if nodes[0] % 128 == 0 and time.perf_counter() > deadline:
                return
            if nodes[0] > 14000 or cost + axis_cost >= best_cost:
                return
            if not rem:
                best_cost = axis_cost + cost
                goals = list(defaults)
                for i, c in assigned:
                    goals[i] = c[2:]
                best_plan = {
                    "axes": list(coordinates),
                    "pieces": goals,
                    "partial": False
                }
                return
            key = (rem, unused)
            if memo.get(key, float("inf")) <= cost:
                return
            memo[key] = cost

            union, maximum_sum = 0, 0
            for i in active:
                if unused & (1 << i):
                    maximum = 0
                    for c in lists[i]:
                        union |= c[0]
                        maximum = max(maximum, (c[0] & rem).bit_count())
                    maximum_sum += maximum
            if rem & ~union or maximum_sum < rem.bit_count():
                return

            chosen = None
            bits = rem
            while bits:
                low = bits & -bits
                bit = low.bit_length() - 1
                options = [
                    (i, c) for i, c in bit_options[bit]
                    if unused & (1 << i)
                    and cost + c[1] + axis_cost < best_cost
                ]
                if not options:
                    return
                if chosen is None or len(options) < len(chosen):
                    chosen = options
                bits -= low
            chosen.sort(key=lambda ic: (
                -(ic[1][0] & rem).bit_count(), ic[1][1]))
            for i, c in chosen:
                search(rem & ~c[0], unused & ~(1 << i),
                       cost + c[1], assigned + [(i, c)])

        search(remaining, sum(1 << i for i in active), 0, [])

    return best_plan if best_plan is not None else partial_plan


def _route(piece, goal):
    start = (piece["cx"], piece["cy"], piece["r"])
    goal = tuple(goal)
    if start == goal:
        return []
    if not piece["link"] and not piece["blocked"]:
        if start[2] != goal[2]:
            return None
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        return ([(1 if dx > 0 else -1, 0)] * abs(dx)
                + [(0, 1 if dy > 0 else -1)] * abs(dy))

    queue = deque([start])
    previous = {start: None}
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    while queue:
        cx, cy, r = queue.popleft()
        w, h, _ = piece["rots"][r]
        for dx, dy in directions:
            if (cx, cy, r, dx, dy) in piece["blocked"]:
                continue
            xx, yy = cx + dx, cy + dy
            if not (0 <= xx - w // 2 <= 21 - w
                    and 0 <= yy - h // 2 <= 21 - h):
                continue
            parallel = (dx if piece["link"] == "v"
                        else dy if piece["link"] == "h" else 0)
            rr = (r + bool(parallel)) % 4
            state = (xx, yy, rr)
            if state in previous:
                continue
            previous[state] = ((cx, cy, r), (dx, dy))
            if state == goal:
                path = []
                current = state
                while previous[current] is not None:
                    current, move = previous[current]
                    path.append(move)
                return path[::-1]
            queue.append(state)
    return None


def _plan_covers(plan, pieces, axes, targets):
    if plan is None or len(plan["pieces"]) != len(pieces):
        return False
    final_axes = [dict(a, pos=x) for a, x in zip(axes, plan["axes"])]
    covered = set()
    for p, goal in zip(pieces, plan["pieces"]):
        covered |= _coverage(p, final_axes, *goal)
    return targets <= covered


def _click_point(which, pieces, axes, raw, mid):
    kind, index = which
    if kind == "a":
        a = axes[index]
        if a["kind"] == "v":
            options = [(a["pos"], y) for y in range(21)
                       if raw[y, a["pos"]] == 10]
            crossings = {b["pos"] for b in axes if b["kind"] == "h"}
            options.sort(key=lambda q: (q[1] in crossings, q[1]))
        else:
            options = [(x, a["pos"]) for x in range(21)
                       if raw[a["pos"], x] == 10]
            crossings = {b["pos"] for b in axes if b["kind"] == "v"}
            options.sort(key=lambda q: (q[0] in crossings, q[0]))
        if options:
            return options[0]
        return (a["pos"], 0) if a["kind"] == "v" else (0, a["pos"])

    cells = _cells(pieces[index])
    options = [q for q in cells if raw[q[1], q[0]] == 5]
    if not options:
        options = list(cells)
    if not options:
        return (0, 0)
    all_cells = [_cells(p) for p in pieces]
    axis_cells = _axis_cells(axes)
    return min(options, key=lambda q: (
        sum(q in c for c in all_cells),
        q in axis_cells,
        mid[q[1], q[0]] == 11,
        q[1], q[0]
    ))


def act(obs, available, memory):
    raw, mid, original, reflected, targets, detected_axes = _read(obs)
    available = set(available)

    if "pieces" not in memory:
        memory["pieces"] = _initial_pieces(original, mid, detected_axes)
        memory["axes"] = detected_axes
        memory["plan"] = None
        memory["selected"] = None
        memory["last"] = None
        memory["click_fail"] = 0
    else:
        fit, learned = _track(memory, original, detected_axes)
        if not fit:
            old = memory["pieces"]
            new = _initial_pieces(original, mid, detected_axes)
            for p in new:
                for q in old:
                    if _cells(p) == _cells(q):
                        p["mode"] = q["mode"]
                        p["link"] = q["link"]
                        p["fixed"] = q["fixed"]
                        p["verified"] = q["verified"]
                        break
            memory["pieces"] = new
            memory["axes"] = detected_axes
            memory["plan"] = None
            memory["selected"] = None
        elif learned:
            memory["plan"] = None

    pieces, axes = memory["pieces"], memory["axes"]
    last = memory.get("last")

    # Reconsider grouping when selecting a previously unverified component
    # reveals disconnected selected portions or an adjacent second sprite.
    if last and last.get("kind") == "click" \
            and last.get("which", (None,))[0] == "p":
        i = last["which"][1]
        if i < len(pieces):
            axis_cells = _axis_cells(axes)
            zeros = {q for q in original if q not in axis_cells
                     and mid[q[1], q[0]] == 0}
            expected = _cells(pieces[i])
            wrong = bool(zeros - expected)
            if not pieces[i]["verified"]:
                wrong |= any(
                    q not in axis_cells and mid[q[1], q[0]] == 9
                    for q in expected)
            if zeros and wrong:
                old = pieces
                pieces = _initial_pieces(original, mid, axes)
                for p in pieces:
                    for q in old:
                        if _cells(p) == _cells(q):
                            p["mode"], p["link"] = q["mode"], q["link"]
                            p["verified"] = q["verified"]
                            break
                memory["pieces"] = pieces
                memory["plan"] = None
                memory["selected"] = None
            else:
                pieces[i]["verified"] = True

    if _learn_modes(pieces, axes, original, reflected):
        memory["plan"] = None

    selected = _selected(pieces, axes, raw, mid, memory.get("selected"))
    memory["selected"] = selected

    if last and last.get("kind") == "click":
        if selected != tuple(last["which"]):
            memory["click_fail"] = memory.get("click_fail", 0) + 1
        else:
            memory["click_fail"] = 0

    # React to failed movement rather than repeating it indefinitely.
    if last and last.get("kind") == "a" and last.get("delta"):
        i = last["i"]
        old_axes = memory.get("old_axes", [])
        if i < len(axes) and i < len(old_axes):
            if axes[i]["pos"] == old_axes[i]["pos"]:
                memory["plan"] = None
                if selected == ("a", i):
                    axes[i]["fixed"] = True
    if last and last.get("kind") == "p":
        i = last["i"]
        old_states = memory.get("old_states", [])
        if i < len(pieces) and i < len(old_states):
            p = pieces[i]
            old = old_states[i]
            if (p["cx"], p["cy"], p["r"]) == old:
                memory["plan"] = None
                if selected == ("p", i):
                    p["blocked"].add((*old, last["dx"], last["dy"]))

    def output(name, info=None, row=None, col=None):
        if name not in available:
            name = next((a for a in (
                "ACTION6", "ACTION5", "ACTION1", "ACTION2",
                "ACTION3", "ACTION4", "ACTION7", "RESET"
            ) if a in available), "RESET")
            info = None
            row = col = 1 if name == "ACTION6" else None
        memory["last"] = info
        return name, row, col

    def select(which):
        if selected == which:
            return None
        if memory.get("click_fail", 0) >= 2 and "ACTION5" in available:
            memory["click_fail"] = 0
            memory["selected"] = None
            return output("ACTION5", {"kind": "cycle"})
        x, y = _click_point(which, pieces, axes, raw, mid)
        memory["selected"] = which
        return output("ACTION6", {"kind": "click", "which": which},
                      3 * y + 1, 3 * x + 1)

    def move_piece(i, dx, dy):
        action = ("ACTION4" if dx > 0 else "ACTION3" if dx < 0
                  else "ACTION2" if dy > 0 else "ACTION1")
        return output(action, {"kind": "p", "i": i, "dx": dx, "dy": dy})

    covered = set()
    for p in pieces:
        covered |= _coverage(p, axes)

    if targets <= covered:
        movable_axes = [i for i, a in enumerate(axes) if not a["fixed"]]
        if movable_axes:
            i = next((i for i in movable_axes if selected == ("a", i)),
                     movable_axes[0])
            result = select(("a", i))
            if result is not None:
                return result
            return output(
                "ACTION1" if axes[i]["kind"] == "v" else "ACTION3",
                {"kind": "a", "i": i, "delta": 0})
        # If only fixed axes exist, a small reversible piece move will cause
        # the simulator to check the goal on the return move.
        for i, p in enumerate(pieces):
            if p["fixed"]:
                continue
            result = select(("p", i))
            if result is not None:
                return result
            w, h, _ = p["rots"][p["r"]]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                xx, yy = p["cx"] + dx, p["cy"] + dy
                if (0 <= xx - w // 2 <= 21 - w
                        and 0 <= yy - h // 2 <= 21 - h):
                    return move_piece(i, dx, dy)

    plan = memory.get("plan")
    if plan is not None and not plan.get("partial") \
            and not _plan_covers(plan, pieces, axes, targets):
        plan = None
    if plan is None:
        plan = _solve(pieces, axes, targets, selected)
        memory["plan"] = plan

    if plan is not None:
        for i, a in enumerate(axes):
            destination = plan["axes"][i]
            if a["pos"] == destination:
                continue
            result = select(("a", i))
            if result is not None:
                return result
            delta = 1 if destination > a["pos"] else -1
            if a["kind"] == "v":
                action = "ACTION4" if delta > 0 else "ACTION3"
            else:
                action = "ACTION2" if delta > 0 else "ACTION1"
            return output(action, {"kind": "a", "i": i, "delta": delta})

        order = list(range(len(pieces)))
        if selected and selected[0] == "p" and selected[1] in order:
            order.remove(selected[1])
            order.insert(0, selected[1])
        for i in order:
            p = pieces[i]
            goal = tuple(plan["pieces"][i])
            if (p["cx"], p["cy"], p["r"]) == goal or p["fixed"]:
                continue
            route = _route(p, goal)
            if not route:
                memory["plan"] = None
                continue
            result = select(("p", i))
            if result is not None:
                return result
            return move_piece(i, *route[0])
        memory["plan"] = None

    for i, p in enumerate(pieces):
        if not p["fixed"] and selected != ("p", i):
            return select(("p", i))
    if selected and selected[0] == "p":
        i = selected[1]
        if i < len(pieces):
            p = pieces[i]
            w, h, _ = p["rots"][p["r"]]
            for dx, dy in ((0, -1), (-1, 0), (0, 1), (1, 0)):
                xx, yy = p["cx"] + dx, p["cy"] + dy
                if (p["cx"], p["cy"], p["r"], dx, dy) in p["blocked"]:
                    continue
                if (0 <= xx - w // 2 <= 21 - w
                        and 0 <= yy - h // 2 <= 21 - h):
                    return move_piece(i, dx, dy)
    return output("ACTION5", {"kind": "cycle"})

import math
import time
import itertools
from collections import deque
import numpy as np


_DIRS = [(0, -1, "ACTION1"), (0, 1, "ACTION2"),
         (-1, 0, "ACTION3"), (1, 0, "ACTION4")]


def _bit_positions(mask):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def _board_mask(points, n):
    out = 0
    for x, y in points:
        if 0 <= x < n and 0 <= y < n:
            out |= 1 << (y * n + x)
    return out


def _world(s):
    return {(s["x"] + x, s["y"] + y) for x, y in s["p"]}


def _templates(s):
    p, w, h = tuple(s["p"]), s["w"], s["h"]
    result = []
    for _ in range(4):
        result.append((p, w, h))
        p = tuple(sorted((y, w - 1 - x) for x, y in p))
        w, h = h, w
    return result


def _turn(s):
    cx = s["x"] + s["w"] // 2
    cy = s["y"] + s["h"] // 2
    p, w, h = _templates(s)[1]
    s["p"], s["w"], s["h"] = p, w, h
    s["x"], s["y"] = cx - w // 2, cy - h // 2


def _tile_data(obs, n):
    scale = 64 // n
    off = (64 - n * scale) // 2
    view = obs[off:off + n * scale, off:off + n * scale].copy()
    if off + n * scale == 64:
        view[:, -1] = -1
    tiles = view.reshape(n, scale, n, scale).transpose(0, 2, 1, 3)
    counts = np.stack([(tiles == c).sum(axis=(2, 3))
                       for c in range(16)], axis=2)
    base = counts.argmax(axis=2).astype(int)

    delta = math.ceil((64 - n * scale) / 2) - off
    if scale == 3:
        dot = delta
        ignored = [(dot, dot)]
    elif scale == 4:
        dot = delta + 1
        ignored = [(a, b) for a in (dot, dot + 1)
                   for b in (dot, dot + 1)]
    elif scale == 5:
        dot = delta + 2
        ignored = [(dot, dot)]
    else:
        dot = scale // 2
        ignored = []

    diff = (tiles != base[:, :, None, None]) & (tiles >= 0)
    for yy, xx in ignored:
        if 0 <= yy < scale and 0 <= xx < scale:
            diff[:, :, yy, xx] = False
    score = int(diff.sum())

    outside = np.ones((64, 64), dtype=bool)
    outside[off:off + n * scale, off:off + n * scale] = False
    outside[:, 63] = False
    score += int(((obs != 5) & outside).sum())

    dot = min(scale - 1, max(0, dot))
    center = tiles[:, :, dot, dot].copy()
    target = (tiles == 11).any(axis=(2, 3))
    return score, scale, off, base, center, target


def _read(obs, m=None):
    if m is None:
        candidates = [21] + [n for n in range(9, 33) if n != 21]
        best = None
        for n in candidates:
            data = _tile_data(obs, n)
            if best is None or data[0] < best[0]:
                best = (data[0], n, data)
            if data[0] == 0:
                break
        n, data = best[1], best[2]
    else:
        n = m["n"]
        data = _tile_data(obs, n)

    _, scale, off, base, center, target = data
    original, occupied, targets = set(), set(), set()
    for y in range(n):
        for x in range(n):
            c = int(base[y, x])
            if c not in (0, 4, 9, 10, 11):
                original.add((x, y))
                occupied.add((x, y))
            elif c == 4:
                occupied.add((x, y))
            if target[y, x]:
                targets.add((x, y))

    axes = []
    threshold = max(3, int(n * .40))
    for d in (1, 2):
        for pos in range(n):
            line = base[:, pos] if d == 1 else base[pos, :]
            inds = np.flatnonzero(line == 10)
            if len(inds) < threshold:
                continue
            cs = [int(center[k, pos] if d == 1 else center[pos, k])
                  for k in inds]
            movable = any(c in (0, 9) for c in cs)
            if scale not in (3, 4, 5):
                movable = True
            axes.append({"d": d, "v": pos, "mov": movable})
    return n, scale, off, base, center, targets, original, occupied, axes


def _on_axis(p, a):
    return p[0] == a["v"] if a["d"] == 1 else p[1] == a["v"]


def _make_shapes(original, center, axes, n, scale):
    left = set(original)
    groups = []
    while left:
        start = min(left, key=lambda p: (p[1], p[0]))
        left.remove(start)
        group = {start}
        q = [start]
        while q:
            x, y = q.pop()
            for dx, dy, _ in _DIRS:
                p = (x + dx, y + dy)
                if p in left:
                    left.remove(p)
                    group.add(p)
                    q.append(p)
        groups.append(group)

    selected_groups = []
    for i, group in enumerate(groups):
        cs = [int(center[y, x]) for x, y in group
              if not any(_on_axis((x, y), a) for a in axes)]
        useful = [c for c in cs if c != 11]
        if useful and all(c == 0 for c in useful):
            selected_groups.append(i)
    if len(selected_groups) > 1:
        joined = set()
        for i in selected_groups:
            joined.update(groups[i])
        groups = [g for i, g in enumerate(groups)
                  if i not in selected_groups] + [joined]

    shapes = []
    for group in groups:
        x0 = min(x for x, y in group)
        y0 = min(y for x, y in group)
        w = max(x for x, y in group) - x0 + 1
        h = max(y for x, y in group) - y0 + 1
        cs = [int(center[y, x]) for x, y in group
              if not any(_on_axis((x, y), a) for a in axes)]
        fixed = any(c == 5 for c in cs) and not any(c in (0, 9) for c in cs)
        if scale not in (3, 4, 5):
            fixed = False
        shapes.append({
            "x": x0, "y": y0, "w": w, "h": h,
            "p": tuple(sorted((x - x0, y - y0) for x, y in group)),
            "mov": not fixed, "mode": 0, "flag": 3,
            "opts": [3, 1, 2]
        })
    return shapes


def _orbit(x, y, axes, flag, n):
    vs = [a["v"] for a in axes if a["d"] == 1 and flag & 1]
    hs = [a["v"] for a in axes if a["d"] == 2 and flag & 2]
    if len(vs) <= 1 and len(hs) <= 1:
        xs = (x, 2 * vs[0] - x) if vs else (x,)
        ys = (y, 2 * hs[0] - y) if hs else (y,)
        return {(xx, yy) for xx in xs for yy in ys
                if 0 <= xx < n and 0 <= yy < n}

    seen = {(x, y)}
    q = deque([(x, y, 0)])
    while q:
        xx, yy, depth = q.popleft()
        if depth == 13:
            continue
        for v in vs:
            p = (2 * v - xx, yy)
            if p not in seen:
                seen.add(p)
                q.append((p[0], p[1], depth + 1))
        for h in hs:
            p = (xx, 2 * h - yy)
            if p not in seen:
                seen.add(p)
                q.append((p[0], p[1], depth + 1))
    return {(xx, yy) for xx, yy in seen if 0 <= xx < n and 0 <= yy < n}


def _shape_coverage(s, axes, flag, n):
    result = set()
    for x, y in _world(s):
        if 0 <= x < n and 0 <= y < n:
            result.update(_orbit(x, y, axes, flag, n))
    return _board_mask(result, n)


def _infer_flags(m, occupied):
    shapes, axes, n = m["shapes"], m["axes"], m["n"]
    if not shapes:
        return False
    observed = _board_mask(occupied, n)
    options = []
    for s in shapes:
        choices = []
        for flag in s["opts"]:
            mask = _shape_coverage(s, axes, flag, n)
            if mask & ~observed == 0:
                choices.append((flag, mask))
        if not choices:
            return False
        options.append(choices)

    solutions = []
    suffix = [0] * (len(shapes) + 1)
    for i in range(len(shapes) - 1, -1, -1):
        suffix[i] = suffix[i + 1]
        for _, mask in options[i]:
            suffix[i] |= mask
    visits = [0]

    def rec(i, covered, flags):
        visits[0] += 1
        if visits[0] > 15000:
            return
        if (covered | suffix[i]) != observed:
            return
        if i == len(shapes):
            if covered == observed:
                solutions.append(tuple(flags))
            return
        for flag, mask in options[i]:
            rec(i + 1, covered | mask, flags + [flag])

    rec(0, 0, [])
    if not solutions:
        return False
    old = tuple(s["flag"] for s in shapes)
    chosen = min(solutions, key=lambda f:
                 (sum(a != b for a, b in zip(f, old)),
                  sum(a != 3 for a in f)))
    changed = chosen != old
    for i, s in enumerate(shapes):
        s["opts"] = [f for f in (3, 1, 2)
                     if any(sol[i] == f for sol in solutions)]
        s["flag"] = chosen[i]
    return changed


def _selected(m, base, center, fallback=None):
    """Read selection from uncontaminated indicator pixels."""
    n = m["n"]
    axes = m["axes"]
    shape_sets = [_world(s) for s in m["shapes"]]
    all_original = set().union(*shape_sets) if shape_sets else set()
    candidates = []

    for i, a in enumerate(axes):
        values = []
        for k in range(n):
            p = (a["v"], k) if a["d"] == 1 else (k, a["v"])
            x, y = p
            if not (0 <= x < n and 0 <= y < n):
                continue
            # At a crossing, the other mirror's indicator may win.
            if any(j != i and _on_axis(p, b)
                   for j, b in enumerate(axes)):
                continue
            if p in all_original or base[y, x] != 10:
                continue
            c = int(center[y, x])
            if c != 11:
                values.append(c)
        if values:
            zeros = values.count(0)
            fraction = zeros / len(values)
            if zeros and fraction >= .60:
                candidates.append((fraction, zeros, ("a", i)))

    for i, s in enumerate(m["shapes"]):
        values = []
        for p in shape_sets[i]:
            x, y = p
            if not (0 <= x < n and 0 <= y < n):
                continue
            if any(_on_axis(p, a) for a in axes):
                continue
            if any(j != i and p in ss for j, ss in enumerate(shape_sets)):
                continue
            c = int(center[y, x])
            if c != 11:
                values.append(c)
        if values:
            zeros = values.count(0)
            fraction = zeros / len(values)
            if zeros and fraction >= .60:
                candidates.append((fraction, zeros, ("s", i)))

    if candidates:
        for fraction, zeros, key in candidates:
            if key == fallback and fraction >= .90:
                return key
        return max(candidates)[2]
    return fallback


def _reconcile(m, obs, parsed):
    n, scale, off, base, center, targets, original, occupied, axes = parsed
    changed = False
    pending = m.pop("pending", None)

    if pending and pending[0] == "move":
        _, kind, idx, direction = pending
        dx, dy, _ = _DIRS[direction]
        unchanged = np.array_equal(obs, m.get("last_obs"))
        if unchanged:
            changed = True
            m["reselect"] = (kind, idx)
            m["click_tries"] = 0
        else:
            affected = []
            if kind == "a" and idx < len(m["axes"]):
                a = m["axes"][idx]
                event = a["d"]
                a["v"] += dx if event == 1 else dy
                affected = [(i, event) for i, s in enumerate(m["shapes"])
                            if s["mode"] in (0, event)]
            elif kind == "s" and idx < len(m["shapes"]):
                s = m["shapes"][idx]
                s["x"] += dx
                s["y"] += dy
                event = 1 if dx else 2
                if s["mode"] in (0, event):
                    affected = [(idx, event)]

            om = _board_mask(original, n)
            cm = _board_mask(occupied, n)
            alternatives = []
            affected_ids = {i for i, event in affected}
            fixed_union = 0
            for i, s in enumerate(m["shapes"]):
                if i not in affected_ids:
                    fixed_union |= _board_mask(_world(s), n)
            for i, event in affected:
                s = m["shapes"][i]
                alt = dict(s)
                _turn(alt)
                alternatives.append((
                    i, event, alt,
                    _board_mask(_world(s), n),
                    _board_mask(_world(alt), n)
                ))

            best = None
            # Rotation choices are usually one or two binary variables.
            if len(alternatives) <= 10:
                patterns = itertools.product((0, 1), repeat=len(alternatives))
            else:
                patterns = [tuple(int(m["shapes"][i]["mode"] == event)
                                  for i, event, alt, mask0, mask1 in alternatives)]
            for bits in patterns:
                union = fixed_union
                penalty = 0
                for bit, (i, event, alt, mask0, mask1) in zip(bits, alternatives):
                    union |= mask1 if bit else mask0
                    expected = int(m["shapes"][i]["mode"] == event)
                    penalty += int(bit != expected)
                score = (5 * (union & ~cm).bit_count()
                         + 5 * (om & ~union).bit_count()
                         + (union & ~om).bit_count())
                value = (score, penalty)
                if best is None or value < best[0]:
                    best = (value, bits)
            if best is not None:
                for bit, (i, event, alt, mask0, mask1) in zip(best[1], alternatives):
                    s = m["shapes"][i]
                    expected = int(s["mode"] == event)
                    if bit:
                        if s["mode"] != event:
                            changed = True
                        alt["mode"] = event
                        m["shapes"][i] = alt
                    if bit != expected:
                        changed = True

    old_axes = m["axes"]
    if len(old_axes) == len(axes):
        unused = set(range(len(axes)))
        for a in old_axes:
            choices = [j for j in unused if axes[j]["d"] == a["d"]]
            if not choices:
                changed = True
                break
            j = min(choices, key=lambda j: abs(axes[j]["v"] - a["v"]))
            unused.remove(j)
            if a["v"] != axes[j]["v"]:
                changed = True
            a.update(axes[j])
    else:
        m["axes"] = axes
        changed = True

    union = set()
    for s in m["shapes"]:
        union.update(p for p in _world(s)
                     if 0 <= p[0] < n and 0 <= p[1] < n)
    bad = len(original - union) + len(union - occupied)
    if bad > max(3, len(original) * .40):
        old_shapes = m["shapes"]
        fresh = _make_shapes(original, center, m["axes"], n, scale)
        for s in fresh:
            matches = [o for o in old_shapes if len(o["p"]) == len(s["p"])]
            if matches:
                o = min(matches, key=lambda o:
                        abs(o["x"] - s["x"]) + abs(o["y"] - s["y"]))
                s["mode"] = o["mode"]
                s["opts"] = list(o["opts"])
                s["flag"] = o["flag"]
        m["shapes"] = fresh
        changed = True

    if not targets.issubset(m["targets"]):
        changed = True
    m["targets"].update(targets)

    fallback = m.get("sel")
    if pending and pending[0] == "click":
        fallback = (pending[1], pending[2])
        m.pop("reselect", None)
    m["sel"] = _selected(m, base, center, fallback)

    if _infer_flags(m, occupied):
        changed = True
    if changed:
        m["plan"] = []
    return changed


def _axis_configurations(m, targets):
    n, axes = m["n"], m["axes"]
    symmetry = {}
    ts = set(targets)
    ranges = []
    for i, a in enumerate(axes):
        scores = []
        for v in range(n):
            score = sum(((2 * v - x, y) if a["d"] == 1 else
                         (x, 2 * v - y)) in ts for x, y in targets)
            symmetry[i, v] = score
            scores.append(v)
        if not a["mov"]:
            ranges.append([a["v"]])
        elif len(axes) <= 2:
            ranges.append(scores)
        else:
            scores.sort(key=lambda v:
                        -symmetry[i, v] + .6 * abs(v - a["v"]))
            ranges.append(list(dict.fromkeys([a["v"]] + scores[:7])))

    configs = list(itertools.product(*ranges)) if axes else [()]

    def key(vals):
        dist = sum(abs(v - a["v"]) for v, a in zip(vals, axes))
        sym = sum(symmetry[i, v] for i, v in enumerate(vals))
        return dist * .65 - sym, dist

    configs.sort(key=key)
    current = tuple(a["v"] for a in axes)
    if current in configs:
        configs.remove(current)
        configs.insert(0, current)
    return configs[:1800]


def _placement_options(s, axes, axis_steps, targets, preimages, n):
    total = len(targets)
    all_bits = [(1 << i) for i in range(total)]
    cx0 = s["x"] + s["w"] // 2
    cy0 = s["y"] + s["h"] // 2
    mode = s["mode"]
    turns_from_axes = axis_steps.get(mode, 0) if mode else 0
    templates = _templates(s)

    if not s["mov"]:
        r = turns_from_axes % 4 if mode else 0
        p, w, h = templates[r]
        x, y = cx0 - w // 2, cy0 - h // 2
        ss = dict(s, x=x, y=y, w=w, h=h, p=p)
        cov = _shape_coverage(ss, axes, s["flag"], n)
        mask = 0
        for i, (tx, ty) in enumerate(targets):
            if cov & (1 << (ty * n + tx)):
                mask |= 1 << i
        return [(mask, 0, x, y, r, mask.bit_count())]

    dedup = {}
    rotations = range(4) if mode else (0,)
    for r in rotations:
        p, w, h = templates[r]
        if w > n or h > n:
            continue
        covers = [0] * (n * n)
        xmax, ymax = n - w, n - h
        for i, points in enumerate(preimages[s["flag"]]):
            bit = all_bits[i]
            for qx, qy in points:
                for px, py in p:
                    x, y = qx - px, qy - py
                    if 0 <= x <= xmax and 0 <= y <= ymax:
                        covers[y * n + x] |= bit

        for y in range(ymax + 1):
            cy = y + h // 2
            for x in range(xmax + 1):
                mask = covers[y * n + x]
                cx = x + w // 2
                cost = abs(cx - cx0) + abs(cy - cy0)
                if mode:
                    relevant = abs(cx - cx0) if mode == 1 else abs(cy - cy0)
                    difference = (r - relevant - turns_from_axes) % 4
                    if difference & 1:
                        continue
                    if difference == 2:
                        cost += 2
                if mask == 0 and cost:
                    continue
                item = (mask, cost, x, y, r, mask.bit_count())
                old = dedup.get(mask)
                if old is None or cost < old[1]:
                    dedup[mask] = item
    return sorted(dedup.values(), key=lambda o: (o[1], -o[5]))


def _cover_search(options, movable, full, limit, deadline):
    count = full.bit_count()
    inverse = []
    maxcounts = []
    for opts in options:
        inv = [[] for _ in range(count)]
        for o in opts:
            for t in _bit_positions(o[0]):
                inv[t].append(o)
        inverse.append(inv)
        maxcounts.append(max((o[5] for o in opts), default=0))

    missing = full
    chosen = {}
    remaining = []
    for i, mov in enumerate(movable):
        if not mov:
            o = options[i][0]
            missing &= ~o[0]
            chosen[i] = o
        else:
            remaining.append(i)

    best_cost = limit
    best_choices = None
    memo = {}
    nodes = 0

    def dfs(miss, rem, cost, choice):
        nonlocal best_cost, best_choices, nodes
        nodes += 1
        if nodes > 35000:
            return
        if nodes % 128 == 0 and time.monotonic() > deadline:
            return
        if not miss:
            if cost < best_cost:
                best_cost = cost
                best_choices = dict(choice)
            return
        if not rem or cost >= best_cost:
            return
        key = (rem, miss)
        if memo.get(key, 10**9) <= cost:
            return
        memo[key] = cost
        needed = miss.bit_count()
        if sum(maxcounts[i] for i in rem) < needed:
            return

        t = min(_bit_positions(miss),
                key=lambda t: sum(len(inverse[i][t]) for i in rem))
        if len(rem) == 1:
            i = rem[0]
            for o in inverse[i][t]:
                if cost + o[1] >= best_cost:
                    break
                if o[0] & miss == miss:
                    choice[i] = o
                    dfs(0, (), cost + o[1], choice)
                    choice.pop(i, None)
            return

        branches = []
        maxsum = sum(maxcounts[i] for i in rem)
        for i in rem:
            for o in inverse[i][t]:
                newcost = cost + o[1]
                if newcost >= best_cost:
                    break
                nm = miss & ~o[0]
                if nm.bit_count() > maxsum - maxcounts[i]:
                    continue
                priority = newcost + .30 * nm.bit_count()
                branches.append((priority, i, o, nm))
        branches.sort(key=lambda item: item[0])
        for _, i, o, nm in branches:
            if cost + o[1] >= best_cost:
                continue
            choice[i] = o
            dfs(nm, tuple(j for j in rem if j != i), cost + o[1], choice)
            choice.pop(i, None)

    dfs(missing, tuple(remaining), 0, chosen)
    return best_cost, best_choices


def _solve(m, deadline):
    n, shapes = m["n"], m["shapes"]
    targets = sorted(m["targets"], key=lambda p: (p[1], p[0]))
    if not shapes or not targets:
        return None
    full = (1 << len(targets)) - 1
    configs = _axis_configurations(m, targets)
    best_cost = 10**6
    best = None
    first_solution = None

    for ci, vals in enumerate(configs):
        if time.monotonic() > deadline:
            break
        if first_solution is not None and ci > first_solution + 70:
            break
        axes = [dict(a, v=v) for a, v in zip(m["axes"], vals)]
        axis_cost = sum(abs(a["v"] - b["v"])
                        for a, b in zip(axes, m["axes"]))
        if axis_cost >= best_cost:
            continue
        steps = {1: 0, 2: 0}
        for a, b in zip(axes, m["axes"]):
            steps[a["d"]] += abs(a["v"] - b["v"])
        flags = {s["flag"] for s in shapes}
        preimages = {
            flag: [_orbit(x, y, axes, flag, n) for x, y in targets]
            for flag in flags
        }
        options = []
        possible = True
        for s in shapes:
            opts = _placement_options(s, axes, steps, targets, preimages, n)
            if not opts:
                possible = False
                break
            options.append(opts)
        if not possible:
            continue
        if sum(max(o[5] for o in opts) for opts in options) < len(targets):
            continue

        cost, choices = _cover_search(
            options, [s["mov"] for s in shapes], full,
            best_cost - axis_cost, deadline)
        if choices is not None and cost + axis_cost < best_cost:
            best_cost = cost + axis_cost
            goals = [None] * len(shapes)
            for i, o in choices.items():
                goals[i] = o
            best = (vals, goals)
            if first_solution is None:
                first_solution = ci
            if best_cost == 0:
                break
    return best


def _piece_path(templates, mode, start, goal, n):
    if start == goal:
        return []
    q = deque([start])
    parent = {start: None}
    while q:
        cx, cy, r = q.popleft()
        _, w, h = templates[r]
        for direction, (dx, dy, _) in enumerate(_DIRS):
            nx, ny = cx + dx, cy + dy
            x, y = nx - w // 2, ny - h // 2
            if x < 0 or y < 0 or x + w > n or y + h > n:
                continue
            turn = (mode == 1 and dx != 0) or (mode == 2 and dy != 0)
            nr = (r + int(turn)) % 4
            state = (nx, ny, nr)
            if state in parent:
                continue
            parent[state] = ((cx, cy, r), direction)
            if state == goal:
                path = []
                cur = state
                while parent[cur] is not None:
                    prev, d = parent[cur]
                    path.append(d)
                    cur = prev
                return path[::-1]
            q.append(state)
    return None


def _build_plan(m, solution, deadline):
    vals, goals = solution
    n, shapes, axes = m["n"], m["shapes"], m["axes"]
    templates = [_templates(s) for s in shapes]
    tasks = []
    for i, (a, v) in enumerate(zip(axes, vals)):
        if a["v"] != v:
            tasks.append(("a", i))
    for i, goal in enumerate(goals):
        if goal is not None and goal[1] > 0 and shapes[i]["mov"]:
            tasks.append(("s", i))
    if not tasks:
        return []

    orders = [tuple(tasks)]
    if len(tasks) <= 6:
        perms = list(itertools.permutations(tasks))
        perms.sort(key=lambda order:
                   (order[0] != m.get("sel"),
                    sum(1 for j in range(len(order) - 1)
                        if order[j][0] == "s" and order[j + 1][0] == "a")))
        orders += perms
    else:
        orders += [tuple(reversed(tasks))]
    cache = {}
    seen_orders = set()

    for order in orders[:150]:
        if order in seen_orders:
            continue
        seen_orders.add(order)
        if time.monotonic() > deadline:
            break
        positions = [(s["x"] + s["w"] // 2,
                      s["y"] + s["h"] // 2, 0) for s in shapes]
        axis_pos = [a["v"] for a in axes]
        plan = []
        good = True
        for kind, idx in order:
            if kind == "a":
                a = axes[idx]
                delta = vals[idx] - axis_pos[idx]
                direction = (3 if delta > 0 else 2) if a["d"] == 1 else (
                    1 if delta > 0 else 0)
                steps = abs(delta)
                plan.extend(("a", idx, direction) for _ in range(steps))
                axis_pos[idx] = vals[idx]
                for j, s in enumerate(shapes):
                    if s["mode"] == a["d"]:
                        cx, cy, r = positions[j]
                        positions[j] = (cx, cy, (r + steps) % 4)
            else:
                s = shapes[idx]
                o = goals[idx]
                _, _, gx, gy, gr, _ = o
                _, gw, gh = templates[idx][gr]
                gcx, gcy = gx + gw // 2, gy + gh // 2
                remaining_turns = 0
                if s["mode"]:
                    for j, a in enumerate(axes):
                        if a["d"] == s["mode"]:
                            remaining_turns += abs(vals[j] - axis_pos[j])
                target_r = (gr - remaining_turns) % 4 if s["mode"] else 0
                goal = (gcx, gcy, target_r)
                key = (idx, positions[idx], goal)
                if key not in cache:
                    cache[key] = _piece_path(
                        templates[idx], s["mode"], positions[idx], goal, n)
                path = cache[key]
                if path is None:
                    good = False
                    break
                plan.extend(("s", idx, d) for d in path)
                positions[idx] = goal
        if good:
            return plan
    return None


def _click_point(m, key):
    kind, idx = key
    n = m["n"]
    shape_sets = [_world(s) for s in m["shapes"]]
    if kind == "s":
        points = shape_sets[idx]
        if not points:
            return None

        def score(p):
            x, y = p
            overlap = sum(p in ss for j, ss in enumerate(shape_sets) if j != idx)
            on_axis = any(_on_axis(p, a) for a in m["axes"])
            return overlap * 10 + int(on_axis), y, x
    else:
        a = m["axes"][idx]
        points = {(a["v"], k) if a["d"] == 1 else (k, a["v"])
                  for k in range(n)}

        def score(p):
            x, y = p
            overlap = sum(p in ss for ss in shape_sets)
            other_axis = sum(_on_axis(p, b)
                             for j, b in enumerate(m["axes"]) if j != idx)
            return overlap * 10 + other_axis, y, x

    points = [p for p in points if 0 <= p[0] < n and 0 <= p[1] < n]
    if not points:
        return None
    x, y = min(points, key=score)
    row = min(63, m["off"] + y * m["scale"] + m["scale"] // 2)
    col = min(62, m["off"] + x * m["scale"] + m["scale"] // 2)
    return row, col


def _probe(m):
    n = m["n"]
    probes = []
    for i, a in enumerate(m["axes"]):
        if a["mov"]:
            direction = (3 if a["v"] < n // 2 else 2) if a["d"] == 1 else (
                1 if a["v"] < n // 2 else 0)
            probes.append(("a", i, direction))
    for i, s in enumerate(m["shapes"]):
        if not s["mov"]:
            continue
        for direction, (dx, dy, _) in enumerate(_DIRS):
            x, y = s["x"] + dx, s["y"] + dy
            if 0 <= x and 0 <= y and x + s["w"] <= n and y + s["h"] <= n:
                probes.append(("s", i, direction))
    if not probes:
        return None
    counter = m.get("probe_count", 0)
    m["probe_count"] = counter + 1
    return probes[counter % len(probes)]


def act(obs, available, memory):
    start = time.monotonic()
    obs = np.asarray(obs, dtype=int)
    m = memory.get("_ar25")
    parsed = _read(obs, m)
    n, scale, off, base, center, targets, original, occupied, axes = parsed

    if m is None:
        m = {
            "n": n, "scale": scale, "off": off,
            "axes": axes,
            "shapes": _make_shapes(original, center, axes, n, scale),
            "targets": set(targets), "plan": [], "sel": None,
            "probe_count": 0
        }
        m["sel"] = _selected(m, base, center)
        _infer_flags(m, occupied)
        memory["_ar25"] = m
    else:
        _reconcile(m, obs, parsed)

    def finish(action, row=None, col=None, pending=None):
        if pending is not None:
            m["pending"] = pending
        m["last_obs"] = obs.copy()
        if action not in available:
            action = next((a for a in available if a != "RESET"), available[0])
            row = col = None
            m.pop("pending", None)
            if action == "ACTION6":
                row = col = 1
        return action, row, col

    if not m["plan"]:
        solution = _solve(m, start + 7.3)
        if solution is not None:
            plan = _build_plan(m, solution, start + 8.7)
            if plan:
                m["plan"] = plan
            elif plan == []:
                for i, a in enumerate(m["axes"]):
                    if a["mov"]:
                        direction = 0 if a["d"] == 1 else 2
                        m["plan"] = [("a", i, direction)]
                        break
        if not m["plan"]:
            probe = _probe(m)
            if probe is not None:
                m["plan"] = [probe]

    if not m["plan"]:
        return finish("ACTION5")

    kind, idx, direction = m["plan"][0]
    wanted = (kind, idx)
    if (kind == "s" and idx >= len(m["shapes"])) or (
            kind == "a" and idx >= len(m["axes"])):
        m["plan"] = []
        return finish("ACTION5")

    if m.get("sel") != wanted or m.get("reselect") == wanted:
        point = _click_point(m, wanted)
        previous_wanted = m.get("click_wanted")
        tries = m.get("click_tries", 0) if previous_wanted == wanted else 0
        m["click_wanted"] = wanted
        m["click_tries"] = tries + 1
        if point is not None and (tries == 0 or tries % 4 == 0):
            return finish("ACTION6", point[0], point[1],
                          ("click", kind, idx))
        return finish("ACTION5", pending=("tab", kind, idx))

    m["click_tries"] = 0
    m["click_wanted"] = None
    m["plan"].pop(0)
    return finish(_DIRS[direction][2],
                  pending=("move", kind, idx, direction))

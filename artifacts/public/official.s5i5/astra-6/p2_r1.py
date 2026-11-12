import numpy as np
import heapq
import itertools
import time
from collections import deque, Counter

# Counterclockwise directions: right, up, left, down.
_DIR = ((1, 0), (0, -1), (-1, 0), (0, 1))
_INF = 1000000.0


def _rot(x, y, d):
    d &= 3
    if d == 0:
        return x, y
    if d == 1:
        return y, -x
    if d == 2:
        return -x, -y
    return -y, x


def _components(points):
    points = set(points)
    result = []
    while points:
        p = points.pop()
        comp = {p}
        stack = [p]
        while stack:
            x, y = stack.pop()
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if q in points:
                    points.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _read_controls(obs):
    ys, xs = np.where(obs == 2)
    groups = _components(zip(xs.tolist(), ys.tolist()))
    controls = []
    ui = np.zeros((64, 64), dtype=bool)

    for comp in groups:
        x0 = min(p[0] for p in comp)
        x1 = max(p[0] for p in comp)
        y0 = min(p[1] for p in comp)
        y1 = max(p[1] for p in comp)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if min(w, h) < 5:
            continue
        patch = obs[y0:y1 + 1, x0:x1 + 1]
        counts = Counter(int(v) for v in patch.flat
                         if int(v) not in (2, 3, 4, 5))
        if not counts:
            continue
        color = counts.most_common(1)[0][0]
        ui[y0:y1 + 1, x0:x1 + 1] = True

        if w == h:
            controls.append((color, 0, y0 + h // 2, x0 + w // 2))
        elif w > h:
            controls.append((color, -1, y0 + h // 2, x0 + 1))
            controls.append((color, 1, y0 + h // 2, x1 - 1))
        else:
            controls.append((color, -1, y0 + 1, x0 + w // 2))
            controls.append((color, 1, y1 - 1, x0 + w // 2))
    return controls, ui


def _read_purple(obs, ui):
    purple = {(int(x), int(y)) for y, x in np.argwhere(obs == 13)
              if not ui[y, x]}
    goals, arms = [], set()
    for y in range(1, 63):
        for x in range(1, 63):
            a = ((x - 1, y), (x + 1, y),
                 (x, y - 1), (x, y + 1))
            if all(p in purple for p in a):
                goals.append((x - 1, y - 1))
                arms.update(a)
    dots = [(x - 1, y - 1) for x, y in purple if (x, y) not in arms]
    return sorted(set(dots)), sorted(set(goals))


def _rotation_distance(a, b, child):
    if not child:
        return (b - a) & 3
    for cost in range(5):
        if a == b:
            return cost
        a = 3 if a == 1 else ((a + 1) & 3)
    return 100


def _configuration(m, s):
    ps, ds, lengths = [], [], []
    for i, v in enumerate(s):
        parent = m["parents"][i]
        length = v // 4 + 1
        if parent < 0:
            p = m["roots"][i]
            d = v & 3
        else:
            d = ((v & 3) + ds[parent]) & 3
            dx, dy = _DIR[ds[parent]]
            p = (ps[parent][0] + dx * lengths[parent],
                 ps[parent][1] + dy * lengths[parent])
        ps.append(p)
        ds.append(d)
        lengths.append(length)
    return ps, ds, lengths


def _geometry(m, s):
    W, H = m["W"], m["H"]
    ps, ds, lengths = [], [], []
    occupied = 0
    tables = m["rod_masks"]
    aux = m["aux_masks"]

    for i, v in enumerate(s):
        parent = m["parents"][i]
        length = v // 4 + 1
        if length > m["maxlen"]:
            return None
        if parent < 0:
            x, y = m["roots"][i]
            d = v & 3
        else:
            d = ((v & 3) + ds[parent]) & 3
            dx, dy = _DIR[ds[parent]]
            x = ps[parent][0] + dx * lengths[parent]
            y = ps[parent][1] + dy * lengths[parent]

        if not (0 <= x < W and 0 <= y < H):
            return None
        if i in aux:
            mask = aux[i][d][length - 1]
        else:
            mask = tables[d][y * W + x][length - 1]
        if not mask or mask & occupied:
            return None
        occupied |= mask
        ps.append((x, y))
        ds.append(d)
        lengths.append(length)

    for owner, table in m["rigid_tables"]:
        dx, dy = _DIR[ds[owner]]
        x = ps[owner][0] + dx * (lengths[owner] - 1)
        y = ps[owner][1] + dy * (lengths[owner] - 1)
        if not (0 <= x < W and 0 <= y < H):
            return None
        mask = table[ds[owner]][y * W + x]
        if not mask or mask & occupied:
            return None
        occupied |= mask

    markers = []
    for owner, offset in m["markers"]:
        if owner < 0:
            markers.append(offset)
        else:
            dx, dy = _DIR[ds[owner]]
            xx, yy = _rot(offset[0], offset[1], ds[owner])
            markers.append((
                ps[owner][0] + dx * (lengths[owner] - 1) + xx,
                ps[owner][1] + dy * (lengths[owner] - 1) + yy
            ))
    return tuple(markers)


def _transition(m, s, action):
    indices, op, row, col = m["actions"][action]
    a = bytearray(s)
    for i in indices:
        v = a[i]
        if op == 1:
            if v // 4 + 1 >= m["maxlen"]:
                return None
            a[i] += 4
        elif op == -1:
            if v >= 4:
                a[i] -= 4
        else:
            d = v & 3
            if m["parents"][i] >= 0 and d == 1:
                nd = 3
            else:
                nd = (d + 1) & 3
            a[i] = (v & ~3) | nd
    result = bytes(a)
    return None if result == s else result


def _make_model(obs):
    controls, ui = _read_controls(obs)
    dots_px, goals_px = _read_purple(obs, ui)
    if not controls or not goals_px or not dots_px:
        return None

    positions = dots_px + goals_px
    ox = Counter(x % 3 for x, y in positions).most_common(1)[0][0]
    oy = Counter(y % 3 for x, y in positions).most_common(1)[0][0]
    W, H = (64 - ox) // 3, (64 - oy) // 3
    N, maxlen = W * H, max(W, H)

    image = obs.copy()
    image[ui] = 5
    image[63, :] = 5
    palette = set(c[0] for c in controls)
    palette.update((0, 6, 7, 8, 9, 10, 11, 12, 14))
    palette.difference_update((1, 2, 3, 4, 5, 13, 15))

    cells, patches = {}, {}
    walls, blue = set(), set()
    for y in range(H):
        for x in range(W):
            p = image[oy + 3*y:oy + 3*y + 3,
                      ox + 3*x:ox + 3*x + 3]
            patches[x, y] = p
            if np.any(p == 15):
                walls.add((x, y))
            if np.any(p == 1):
                blue.add((x, y))
            counts = Counter(int(v) for v in p.flat if int(v) in palette)
            if counts:
                cells[x, y] = counts.most_common(1)[0][0]

    rods = []
    for color in sorted(set(cells.values())):
        points = {p for p, c in cells.items() if c == color}
        caps = {}
        for p in points:
            a = patches[p]
            scores = (
                int(np.count_nonzero(a[:, 0] == 3)),
                int(np.count_nonzero(a[-1, :] == 3)),
                int(np.count_nonzero(a[:, -1] == 3)),
                int(np.count_nonzero(a[0, :] == 3)),
            )
            score = max(scores)
            if score >= 2:
                caps[p] = (scores.index(score), score)

        used = set()
        for pivot in sorted(caps,
                            key=lambda p: (-caps[p][1], p[1], p[0])):
            if pivot in used:
                continue
            d = caps[pivot][0]
            dx, dy = _DIR[d]
            line, q = [], pivot
            while q in points and q not in used:
                if q != pivot and q in caps:
                    break
                line.append(q)
                q = q[0] + dx, q[1] + dy
            if line:
                used.update(line)
                rods.append({
                    "color": color, "pivot": pivot, "d": d,
                    "length": len(line), "cells": set(line)
                })
    if not rods:
        return None

    parents = [-1] * len(rods)
    for j, child in enumerate(rods):
        candidates = []
        for i, parent in enumerate(rods):
            if i == j:
                continue
            dx, dy = _DIR[parent["d"]]
            end = (parent["pivot"][0] + parent["length"] * dx,
                   parent["pivot"][1] + parent["length"] * dy)
            if end == child["pivot"]:
                backward = child["d"] == ((parent["d"] + 2) & 3)
                candidates.append((backward, i))
        if candidates:
            candidates.sort()
            if not candidates[0][0]:
                parents[j] = candidates[0][1]

    for i in range(len(rods)):
        seen, j = set(), i
        while j >= 0:
            if j in seen:
                parents[j] = -1
                break
            seen.add(j)
            j = parents[j]

    order, added = [], set()
    while len(order) < len(rods):
        progress = False
        for i in range(len(rods)):
            if i not in added and (parents[i] < 0 or parents[i] in added):
                order.append(i)
                added.add(i)
                progress = True
        if not progress:
            return None

    remap = {old: new for new, old in enumerate(order)}
    old_rods = rods
    rods = [old_rods[i] for i in order]
    parents = [-1 if parents[i] < 0 else remap[parents[i]] for i in order]
    colors = [r["color"] for r in rods]
    roots = [r["pivot"] for r in rods]

    state = []
    for i, r in enumerate(rods):
        relative = r["d"]
        if parents[i] >= 0:
            relative = (relative - rods[parents[i]]["d"]) & 3
        state.append(4 * (r["length"] - 1) + relative)
    state = bytes(state)

    rigid, static_blue = [], set()
    for comp in _components(blue):
        choices = []
        for i, r in enumerate(rods):
            dx, dy = _DIR[r["d"]]
            p = (r["pivot"][0] + dx * r["length"],
                 r["pivot"][1] + dy * r["length"])
            if p in comp:
                choices.append(i)
        if choices:
            i = choices[0]
            r = rods[i]
            dx, dy = _DIR[r["d"]]
            tip = (r["pivot"][0] + dx * (r["length"] - 1),
                   r["pivot"][1] + dy * (r["length"] - 1))
            offsets = tuple(_rot(x - tip[0], y - tip[1], -r["d"])
                            for x, y in comp)
            rigid.append((i, offsets))
        else:
            static_blue.update(comp)

    walls.update(static_blue)
    wallmask = 0
    for x, y in walls:
        wallmask |= 1 << (y * W + x)

    goals = tuple(((x - ox) // 3, (y - oy) // 3) for x, y in goals_px)
    markers = []
    for x, y in dots_px:
        p = ((x - ox) // 3, (y - oy) // 3)
        owner = next((i for i, r in enumerate(rods) if p in r["cells"]), -1)
        if owner < 0:
            markers.append((-1, p))
        else:
            r = rods[owner]
            dx, dy = _DIR[r["d"]]
            tx = r["pivot"][0] + dx * (r["length"] - 1)
            ty = r["pivot"][1] + dy * (r["length"] - 1)
            markers.append((owner, _rot(p[0] - tx, p[1] - ty, -r["d"])))
    if len(markers) < len(goals):
        return None

    actions = []
    for color, op, row, col in controls:
        indices = tuple(i for i, c in enumerate(colors) if c == color)
        if indices:
            actions.append((indices, op, row, col))
    if not actions:
        return None

    resize_colors = {c for c, op, row, col in controls if op != 0}
    rotate_colors = {c for c, op, row, col in controls if op == 0}

    rod_masks = [[None] * N for _ in range(4)]
    for d, (dx, dy) in enumerate(_DIR):
        for pos in range(N):
            x, y = pos % W, pos // W
            masks, mask, valid = [], 0, True
            for length in range(1, maxlen + 1):
                xx, yy = x + dx * (length - 1), y + dy * (length - 1)
                if not (0 <= xx < W and 0 <= yy < H):
                    valid = False
                if valid:
                    bit = 1 << (yy * W + xx)
                    if wallmask & bit:
                        valid = False
                    else:
                        mask |= bit
                masks.append(mask if valid else 0)
            rod_masks[d][pos] = tuple(masks)

    rigid_tables = []
    for owner, offsets in rigid:
        table = [[0] * N for _ in range(4)]
        for d in range(4):
            points = [_rot(x, y, d) for x, y in offsets]
            for pos in range(N):
                x, y = pos % W, pos // W
                mask, valid = 0, True
                for xx, yy in points:
                    xx, yy = xx + x, yy + y
                    if not (0 <= xx < W and 0 <= yy < H):
                        valid = False
                        break
                    bit = 1 << (yy * W + xx)
                    if bit & wallmask:
                        valid = False
                        break
                    mask |= bit
                if valid:
                    table[d][pos] = mask
        rigid_tables.append((owner, table))

    # The camera boundary is not intrinsically a wall. In particular,
    # independent rotating obstacles can temporarily extend off screen.
    marked_ancestors = set()
    for owner, _ in markers:
        while owner >= 0:
            marked_ancestors.add(owner)
            owner = parents[owner]
    has_children = set(p for p in parents if p >= 0)
    rigid_owners = set(i for i, _ in rigid)
    aux_masks = {}
    for i, r in enumerate(rods):
        if (parents[i] < 0 and i not in marked_ancestors
                and i not in has_children and i not in rigid_owners
                and colors[i] in rotate_colors):
            x, y = roots[i]
            table = []
            for dx, dy in _DIR:
                masks, mask, valid = [], 0, True
                for length in range(1, maxlen + 1):
                    xx, yy = x + dx * (length - 1), y + dy * (length - 1)
                    if 0 <= xx < W and 0 <= yy < H:
                        bit = 1 << (yy * W + xx)
                        if bit & wallmask:
                            valid = False
                        mask |= bit
                    masks.append(mask if valid else 0)
                table.append(tuple(masks))
            aux_masks[i] = tuple(table)

    descriptions = []
    for owner, offset in markers:
        if owner < 0:
            descriptions.append({"fixed": offset})
            continue
        chain, i = [], owner
        while i >= 0:
            chain.append(i)
            i = parents[i]
        chain.reverse()
        changing_lengths = tuple(i for i in chain if colors[i] in resize_colors)
        rotating = tuple(i for i in chain if colors[i] in rotate_colors)

        choices = []
        for i in chain:
            if colors[i] in rotate_colors:
                if parents[i] < 0 or (state[i] & 3) == 2:
                    choices.append((0, 1, 2, 3))
                else:
                    choices.append((0, 1, 3))
            else:
                choices.append((state[i] & 3,))

        patterns = []
        for relative in itertools.product(*choices):
            ds, d = [], 0
            for k, i in enumerate(chain):
                d = relative[k] if k == 0 else ((d + relative[k]) & 3)
                ds.append(d)
            last_dx, last_dy = _DIR[ds[-1]]
            offx, offy = _rot(offset[0], offset[1], ds[-1])
            bx = roots[chain[0]][0] - last_dx + offx
            by = roots[chain[0]][1] - last_dy + offy
            coefficients = []
            for i, d in zip(chain, ds):
                dx, dy = _DIR[d]
                if colors[i] in resize_colors:
                    coefficients.append((dx, dy))
                else:
                    length = state[i] // 4 + 1
                    bx += length * dx
                    by += length * dy
            minx = maxx = bx
            miny = maxy = by
            for dx, dy in coefficients:
                minx += dx if dx >= 0 else dx * maxlen
                maxx += dx * maxlen if dx >= 0 else dx
                miny += dy if dy >= 0 else dy * maxlen
                maxy += dy * maxlen if dy >= 0 else dy
            reachable = tuple(j for j, (gx, gy) in enumerate(goals)
                              if minx <= gx <= maxx and miny <= gy <= maxy)
            if reachable:
                rot_values = tuple(relative[chain.index(i)] for i in rotating)
                patterns.append((rot_values, tuple(coefficients),
                                 bx, by, reachable))
            if len(patterns) >= 4096:
                break
        descriptions.append({
            "chain": tuple(chain), "lengths": changing_lengths,
            "rotating": rotating,
            "children": tuple(parents[i] >= 0 for i in rotating),
            "choices": tuple(choices), "offset": offset,
            "patterns": patterns, "cache": {}, "rotcache": {}, "db": {},
        })

    goal_maps = []
    for gx, gy in goals:
        distances = [60] * N
        if 0 <= gx < W and 0 <= gy < H:
            start = gy * W + gx
            distances[start] = 0
            queue = deque([start])
            while queue:
                p = queue.popleft()
                x, y = p % W, p // W
                for dx, dy in _DIR:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < W and 0 <= yy < H:
                        q = yy * W + xx
                        if not (wallmask >> q) & 1 and distances[q] == 60:
                            distances[q] = distances[p] + 1
                            queue.append(q)
        goal_maps.append(distances)

    assignments = list(itertools.islice(
        itertools.permutations(range(len(markers)), len(goals)), 720))
    rough = []
    for desc in descriptions:
        if "fixed" in desc:
            rough.append({j for j, g in enumerate(goals) if g == desc["fixed"]})
        else:
            rough.append(set(j for p in desc["patterns"] for j in p[4]))
    feasible = [a for a in assignments
                if all(g in rough[i] for g, i in enumerate(a))]
    if feasible:
        assignments = feasible

    model = {
        "W": W, "H": H, "N": N, "ox": ox, "oy": oy,
        "maxlen": maxlen, "parents": parents, "roots": roots,
        "colors": colors, "palette": tuple(palette),
        "actions": actions, "rod_masks": rod_masks,
        "aux_masks": aux_masks,
        "rigid": rigid, "rigid_tables": rigid_tables,
        "static_blue": static_blue, "markers": markers,
        "goals": goals, "goalset": set(goals),
        "descriptions": descriptions, "goal_maps": goal_maps,
        "assignments": assignments, "ui": ui,
        "banned": set(), "history": {}, "initial": state,
        "tasks": [], "active_build": None, "allocated": 0,
    }
    if _geometry(model, state) is None:
        return None

    # Allocate strong goal-distance tables only where rotation makes the
    # simpler endpoint heuristic inadequate.
    candidates = []
    for di, desc in enumerate(descriptions):
        if "fixed" in desc or not desc["rotating"] or len(desc["chain"]) < 2:
            continue
        spec, shape = [], []
        for k, i in enumerate(desc["chain"]):
            if i in desc["lengths"]:
                spec.append((i, 0, None))
                shape.append(maxlen)
            if i in desc["rotating"]:
                values = desc["choices"][k]
                spec.append((i, 1, values))
                shape.append(len(values))
        volume = 1
        for n in shape:
            volume *= n
        if not shape or volume > 10000000:
            continue
        used_goals = {g for a in assignments for g, mi in enumerate(a)
                      if mi == di}
        for g in used_goals:
            candidates.append((-len(desc["chain"]), volume, di, g,
                               tuple(spec), tuple(shape)))
    candidates.sort()
    reserved = 0
    for _, volume, di, g, spec, shape in candidates:
        if reserved + volume <= 18000000:
            reserved += volume
            model["tasks"].append((di, g, spec, shape, volume))
    return model, state


class _IKStop(Exception):
    pass


def _goal_configurations(m, desc, goal_index, deadline):
    """Enumerate valid final geometries for one marker's ancestor chain."""
    chain = desc["chain"]
    count = len(chain)
    initial = m["initial"]
    W, H = m["W"], m["H"]
    gx, gy = m["goals"][goal_index]
    rootx, rooty = m["roots"][chain[0]]
    lengths_set = set(desc["lengths"])
    lo, hi = [], []
    for i in chain:
        if i in lengths_set:
            lo.append(1)
            hi.append(m["maxlen"])
        else:
            length = initial[i] // 4 + 1
            lo.append(length)
            hi.append(length)
    rigid_by_owner = {}
    for owner, table in m["rigid_tables"]:
        rigid_by_owner.setdefault(owner, []).append(table)

    results, codes = [], [0] * count
    calls = [0]
    complete = True

    try:
        for relative in itertools.product(*desc["choices"]):
            directions, d = [], 0
            for k, rel in enumerate(relative):
                d = rel if k == 0 else ((d + rel) & 3)
                directions.append(d)
            offx, offy = _rot(desc["offset"][0], desc["offset"][1],
                             directions[-1])
            dx, dy = _DIR[directions[-1]]
            bounds = [None] * (count + 1)
            bounds[count] = (offx - dx, offx - dx,
                             offy - dy, offy - dy)
            for k in range(count - 1, -1, -1):
                dx, dy = _DIR[directions[k]]
                a, b, c, e = bounds[k + 1]
                bounds[k] = (
                    a + min(dx * lo[k], dx * hi[k]),
                    b + max(dx * lo[k], dx * hi[k]),
                    c + min(dy * lo[k], dy * hi[k]),
                    e + max(dy * lo[k], dy * hi[k])
                )

            def visit(k, x, y, occupied):
                calls[0] += 1
                if calls[0] % 1024 == 0:
                    if time.perf_counter() >= deadline or len(results) >= 50000:
                        raise _IKStop()
                a, b, c, e = bounds[k]
                if not (a <= gx - x <= b and c <= gy - y <= e):
                    return
                if not (0 <= x < W and 0 <= y < H):
                    return
                d = directions[k]
                dx, dy = _DIR[d]
                masks = m["rod_masks"][d][y * W + x]
                if k == count - 1:
                    vx, vy = gx - x - offx, gy - y - offy
                    if vx * dy != vy * dx:
                        return
                    length = vx * dx + vy * dy + 1
                    if not (lo[k] <= length <= hi[k]):
                        return
                    options = (length,)
                else:
                    options = range(lo[k], hi[k] + 1)

                for length in options:
                    mask = masks[length - 1]
                    if not mask:
                        break
                    if mask & occupied:
                        continue
                    whole = occupied | mask
                    tx = x + dx * (length - 1)
                    ty = y + dy * (length - 1)
                    valid = True
                    for table in rigid_by_owner.get(chain[k], ()):
                        extra = table[d][ty * W + tx]
                        if not extra or extra & whole:
                            valid = False
                            break
                        whole |= extra
                    if not valid:
                        continue
                    codes[k] = 4 * (length - 1) + relative[k]
                    if k == count - 1:
                        results.append(tuple(codes))
                    else:
                        visit(k + 1, x + dx * length, y + dy * length, whole)

            visit(0, rootx, rooty, 0)
            if time.perf_counter() >= deadline:
                raise _IKStop()
    except _IKStop:
        complete = False
    return results, complete


def _prepare_heuristics(m, deadline):
    """Incrementally construct separable distance transforms of goal sets."""
    while time.perf_counter() < deadline:
        active = m["active_build"]
        if active is None:
            if not m["tasks"]:
                return
            di, goal, spec, shape, volume = m["tasks"].pop(0)
            desc = m["descriptions"][di]
            until = min(deadline - 0.15, time.perf_counter() + 2.0)
            if until <= time.perf_counter():
                m["tasks"].insert(0, (di, goal, spec, shape, volume))
                return
            configurations, complete = _goal_configurations(m, desc, goal, until)
            if not configurations:
                continue

            array = np.full(shape, 3000, dtype=np.int16)
            columns = []
            lookup_spec = []
            for i, kind, values in spec:
                k = desc["chain"].index(i)
                if kind == 0:
                    columns.append(np.fromiter(
                        (v[k] // 4 for v in configurations),
                        dtype=np.intp, count=len(configurations)))
                    lookup_spec.append((i, 0, None))
                else:
                    mapping = {v: j for j, v in enumerate(values)}
                    columns.append(np.fromiter(
                        (mapping[v[k] & 3] for v in configurations),
                        dtype=np.intp, count=len(configurations)))
                    lookup_spec.append((i, 1, mapping))
            array[tuple(columns)] = 0

            operations = []
            for axis, (i, kind, values) in enumerate(spec):
                if kind == 0:
                    for j in range(1, shape[axis]):
                        operations.append((axis, j, j - 1))
                    for j in range(shape[axis] - 2, -1, -1):
                        operations.append((axis, j, j + 1))
                else:
                    child = m["parents"][i] >= 0
                    cycle = (0, 1, 3) if child else (0, 1, 2, 3)
                    mapping = {v: j for j, v in enumerate(values)}
                    for repeat in range(2):
                        for v in reversed(cycle):
                            nxt = 3 if child and v == 1 else ((v + 1) & 3)
                            operations.append((axis, mapping[v], mapping[nxt]))
                    if child and 2 in mapping:
                        operations.append((axis, mapping[2], mapping[3]))

            active = {
                "di": di, "goal": goal, "array": array,
                "spec": tuple(lookup_spec), "operations": operations, "pos": 0,
            }
            m["active_build"] = active
            m["allocated"] += volume

        array = active["array"]
        operations = active["operations"]
        while active["pos"] < len(operations):
            if time.perf_counter() >= deadline:
                return
            axis, a, b = operations[active["pos"]]
            dst = [slice(None)] * array.ndim
            src = [slice(None)] * array.ndim
            dst[axis] = slice(a, a + 1)
            src[axis] = slice(b, b + 1)
            destination = array[tuple(dst)]
            np.minimum(destination, array[tuple(src)] + 1, out=destination)
            active["pos"] += 1

        desc = m["descriptions"][active["di"]]
        desc["db"][active["goal"]] = (array, active["spec"])
        desc["cache"].clear()
        m["active_build"] = None


def _marker_costs(m, s, desc):
    if "fixed" in desc:
        return tuple(0.0 if p == desc["fixed"] else _INF for p in m["goals"])

    key = bytes(s[i] for i in desc["chain"])
    cached = desc["cache"].get(key)
    if cached is not None:
        return cached

    rotkey = tuple(s[i] & 3 for i in desc["rotating"])
    rotation_options = desc["rotcache"].get(rotkey)
    patterns = desc["patterns"]
    if rotation_options is None:
        rotation_options = []
        for k, pattern in enumerate(patterns):
            cost = sum(_rotation_distance(a, b, child)
                       for a, b, child in
                       zip(rotkey, pattern[0], desc["children"]))
            rotation_options.append((cost, k))
        if len(rotation_options) > 80:
            rotation_options.sort()
            rotation_options = rotation_options[:80]
        desc["rotcache"][rotkey] = rotation_options

    lengths = tuple(s[i] // 4 + 1 for i in desc["lengths"])
    costs = [_INF] * len(m["goals"])
    for rcost, k in rotation_options:
        _, coefficients, x, y, reachable = patterns[k]
        for length, (dx, dy) in zip(lengths, coefficients):
            x += length * dx
            y += length * dy
        for j in reachable:
            gx, gy = m["goals"][j]
            value = rcost + abs(x - gx) + abs(y - gy)
            if value < costs[j]:
                costs[j] = value

    for goal, (array, spec) in desc["db"].items():
        index = tuple(s[i] // 4 if kind == 0 else mapping[s[i] & 3]
                      for i, kind, mapping in spec)
        costs[goal] = int(array[index])

    result = tuple(costs)
    desc["cache"][key] = result
    return result


def _heuristic(m, s, markers):
    matrix = [_marker_costs(m, s, d) for d in m["descriptions"]]
    best = _INF
    W, H = m["W"], m["H"]
    for assignment in m["assignments"]:
        value = 0.0
        for goal_index, marker_index in enumerate(assignment):
            c = matrix[marker_index][goal_index]
            x, y = markers[marker_index]
            if c >= _INF:
                gx, gy = m["goals"][goal_index]
                c = 30 + abs(gx - x) + abs(gy - y)
            if 0 <= x < W and 0 <= y < H:
                spatial = m["goal_maps"][goal_index][y * W + x]
            else:
                spatial = 60
            value += c + 0.08 * spatial
        if value < best:
            best = value
    return best


def _paint_rectangle(image, x0, y0, x1, y1, color):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(64, x1), min(64, y1)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1] = color


def _fits(m, s, obs):
    try:
        if _geometry(m, s) is None:
            return False
        ps, ds, lengths = _configuration(m, s)
        expected = np.full((64, 64), 5, dtype=np.int8)
        ox, oy = m["ox"], m["oy"]

        for i, ((x, y), d, length) in enumerate(zip(ps, ds, lengths)):
            dx, dy = _DIR[d]
            ex, ey = x + dx * (length - 1), y + dy * (length - 1)
            x0, x1 = ox + 3 * min(x, ex), ox + 3 * max(x, ex) + 3
            y0, y1 = oy + 3 * min(y, ey), oy + 3 * max(y, ey) + 3
            _paint_rectangle(expected, x0, y0, x1, y1, m["colors"][i])
            px, py = ox + 3*x, oy + 3*y
            if d == 0:
                _paint_rectangle(expected, px, py, px + 1, py + 3, 3)
            elif d == 1:
                _paint_rectangle(expected, px, py + 2, px + 3, py + 3, 3)
            elif d == 2:
                _paint_rectangle(expected, px + 2, py, px + 3, py + 3, 3)
            else:
                _paint_rectangle(expected, px, py, px + 3, py + 1, 3)

        blue_points = set(m["static_blue"])
        for owner, offsets in m["rigid"]:
            dx, dy = _DIR[ds[owner]]
            tx = ps[owner][0] + dx * (lengths[owner] - 1)
            ty = ps[owner][1] + dy * (lengths[owner] - 1)
            for x, y in offsets:
                x, y = _rot(x, y, ds[owner])
                blue_points.add((tx + x, ty + y))
        for x, y in blue_points:
            px, py = ox + 3*x, oy + 3*y
            _paint_rectangle(expected, px, py, px + 3, py + 3, 1)

        relevant = np.isin(obs, m["palette"] + (1, 3)) | (expected != 5)
        relevant &= ~m["ui"]
        relevant &= obs != 13
        relevant[63, :] = False
        return not np.any((expected != obs) & relevant)
    except Exception:
        return False


def _path(nodes, index):
    result = []
    while index > 0:
        _, parent, action, _, _ = nodes[index]
        result.append(action)
        index = parent
    result.reverse()
    return result


def _search(m, start, deadline):
    markers = _geometry(m, start)
    if markers is None:
        return []
    if m["goalset"].issubset(set(markers)):
        return [0]

    for desc in m["descriptions"]:
        if len(desc.get("cache", ())) > 180000:
            desc["cache"].clear()

    h0 = _heuristic(m, start, markers)
    nodes = [(start, -1, -1, 0, h0)]
    heap = [(1.85 * h0, h0, 0, 0)]
    distances = {start: 0}
    best_index, best_progress = 0, h0
    expansions = 0
    n_actions = len(m["actions"])
    history, banned = m["history"], m["banned"]

    while heap:
        if expansions % 64 == 0:
            if time.perf_counter() >= deadline or len(nodes) >= 350000:
                break
        _, _, g, index = heapq.heappop(heap)
        s, _, _, stored_g, h = nodes[index]
        if g != distances.get(s) or stored_g != g:
            continue
        expansions += 1

        for action in range(n_actions):
            if (s, action) in banned:
                continue
            ns = _transition(m, s, action)
            if ns is None:
                continue
            ng = g + 1 + 6 * min(3, history.get(ns, 0))
            if ng >= distances.get(ns, 10**9):
                continue
            new_markers = _geometry(m, ns)
            if new_markers is None:
                continue
            distances[ns] = ng

            if m["goalset"].issubset(set(new_markers)):
                nodes.append((ns, index, action, ng, 0.0))
                return _path(nodes, len(nodes) - 1)

            nh = _heuristic(m, ns, new_markers)
            ni = len(nodes)
            nodes.append((ns, index, action, ng, nh))
            heapq.heappush(heap, (ng + 1.85 * nh, nh, ng, ni))
            progress = nh + 0.015 * ng
            if progress < best_progress:
                best_progress, best_index = progress, ni

    if best_index == 0:
        while heap:
            _, _, g, index = heapq.heappop(heap)
            if g == distances.get(nodes[index][0]):
                best_index = index
                break
    if best_index:
        return _path(nodes, best_index)[:4]

    candidates = []
    for action in range(n_actions):
        if (start, action) in banned:
            continue
        ns = _transition(m, start, action)
        if ns is None:
            continue
        p = _geometry(m, ns)
        if p is not None:
            candidates.append((_heuristic(m, ns, p)
                               + 10 * history.get(ns, 0), action))
    return [min(candidates)[1]] if candidates else []


def act(obs, available, memory):
    began = time.perf_counter()
    deadline = began + 8.5
    obs = np.asarray(obs, dtype=np.int8)

    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    data = memory.get("_s5_player_v2")
    if data is not None and data.get("pending") is not None:
        predicted, action = data.pop("pending")
        model = data["model"]
        if _fits(model, predicted, obs):
            data["state"] = predicted
        elif np.array_equal(obs[:63], data["last"][:63]):
            model["banned"].add((data["state"], action))
            data["plan"] = []
        else:
            # Re-observation also recovers lengths that were initially
            # partially hidden beyond the camera boundary.
            data = None

    if data is None:
        built = _make_model(obs)
        if built is None:
            controls, _ = _read_controls(obs)
            memory.pop("_s5_player_v2", None)
            if controls:
                _, _, row, col = controls[0]
                return "ACTION6", int(row), int(col)
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION6", 0, 0
        model, state = built
        data = {
            "model": model, "state": state,
            "plan": [], "last": obs.copy(),
        }
        memory["_s5_player_v2"] = data

    model, state = data["model"], data["state"]
    if model["tasks"] or model["active_build"] is not None:
        _prepare_heuristics(model, min(began + 5.5, deadline - 1.5))

    model["history"][state] = model["history"].get(state, 0) + 1
    action, predicted = None, None

    while data["plan"]:
        candidate = data["plan"].pop(0)
        ns = _transition(model, state, candidate)
        if (ns is not None and (state, candidate) not in model["banned"]
                and _geometry(model, ns) is not None):
            action, predicted = candidate, ns
            break
        data["plan"] = []

    if action is None:
        data["plan"] = _search(model, state, deadline)
        if data["plan"]:
            action = data["plan"].pop(0)
            predicted = _transition(model, state, action)

    if action is None or predicted is None:
        memory.pop("_s5_player_v2", None)
        if "RESET" in available:
            return "RESET", None, None
        _, _, row, col = model["actions"][0]
        return "ACTION6", int(row), int(col)

    data["pending"] = (predicted, action)
    data["last"] = obs.copy()
    memory["_s5_player_v2"] = data
    _, _, row, col = model["actions"][action]
    return "ACTION6", int(row), int(col)

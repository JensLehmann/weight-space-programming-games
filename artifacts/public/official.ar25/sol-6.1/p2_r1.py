import numpy as np
from collections import deque
from itertools import product
from copy import deepcopy


_DIRS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


def _geometry(a):
    yy, xx = np.where(a[:, :63] != 5)
    ex = int(xx.max() - xx.min() + 1) if len(xx) else 63
    ey = int(yy.max() - yy.min() + 1) if len(yy) else 63
    best = None
    for scale in range(2, 7):
        width = max(1, min(64, int(round(ex / scale))))
        height = max(1, min(64, int(round(ey / scale))))
        if min(64 // width, 64 // height) != scale:
            continue
        px = (64 - width * scale) // 2
        py = (64 - height * scale) // 2
        bad = total = 0
        for y in range(height):
            for x in range(width):
                b = a[py+y*scale:py+(y+1)*scale,
                      px+x*scale:min(63, px+(x+1)*scale)]
                if not b.size:
                    continue
                counts = np.bincount(b.ravel(), minlength=16)
                bad += b.size - int(counts.max())
                total += b.size
        score = bad / max(1, total)
        score += 0.000001 * abs(width - 21)
        candidate = (score, width, height, scale, px, py)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return 21, 21, 3, 0, 0
    return best[1:]


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = left.pop()
        q = [start]
        group = {start}
        for x, y in q:
            for p in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if p in left:
                    left.remove(p)
                    group.add(p)
                    q.append(p)
        result.append(group)
    return result


def _decode(obs, geom):
    width, height, scale, px, py = geom
    base = np.full((height, width), 9, dtype=int)
    mark = np.full_like(base, 9)
    mx = int(np.ceil((64-width*scale)/2))
    my = int(np.ceil((64-height*scale)/2))
    if scale == 3:
        off = 0
    elif scale in (4, 5):
        off = 2
    else:
        off = None

    for y in range(height):
        for x in range(width):
            block = obs[py+y*scale:py+(y+1)*scale,
                        px+x*scale:min(63, px+(x+1)*scale)]
            if block.size:
                base[y, x] = int(np.bincount(
                    block.ravel(), minlength=16).argmax())
            if off is not None:
                xx, yy = mx+x*scale+off, my+y*scale+off
                if 0 <= xx < 63 and 0 <= yy < 64:
                    mark[y, x] = int(obs[yy, xx])
            else:
                mark[y, x] = base[y, x]

    actual = {(int(x), int(y)) for y, x in zip(*np.where(base == 5))}
    reflected = {(int(x), int(y)) for y, x in zip(*np.where(base == 4))}
    goals = {(int(x), int(y)) for y, x in zip(
        *np.where((base == 11) | (mark == 11)))}

    axes = []
    for x in range(width):
        if int(np.sum(base[:, x] == 10)) >= max(3, height // 2):
            vals = [int(mark[y, x]) for y in range(height)
                    if base[y, x] == 10 and mark[y, x] != 11]
            # Locked axes retain colour 10 at their inset. Movable axes
            # instead have colour 9 (unselected) or 0 (selected).
            fixed = bool(vals) and sum(v in (5, 10) for v in vals) > len(vals)//2
            axes.append(("v", x, fixed))
    for y in range(height):
        if int(np.sum(base[y, :] == 10)) >= max(3, width // 2):
            vals = [int(mark[y, x]) for x in range(width)
                    if base[y, x] == 10 and mark[y, x] != 11]
            fixed = bool(vals) and sum(v in (5, 10) for v in vals) > len(vals)//2
            axes.append(("h", y, fixed))
    return base, mark, actual, reflected, goals, axes


def _variants(group):
    x0 = min(x for x, y in group)
    y0 = min(y for x, y in group)
    width = max(x for x, y in group) - x0 + 1
    height = max(y for x, y in group) - y0 + 1
    a = np.zeros((height, width), dtype=np.int8)
    for x, y in group:
        a[y-y0, x-x0] = 1
    result = []
    for r in range(4):
        b = np.rot90(a, r)
        h, w = b.shape
        offsets = tuple((int(x)-w//2, int(y)-h//2)
                        for y, x in zip(*np.where(b)))
        result.append((offsets, w, h))
    return result, x0+width//2, y0+height//2


def _initial_world(decoded, geom):
    base, mark, actual, reflected, goals, axes = decoded
    width, height = geom[:2]
    objects = []
    for kind, pos, fixed in axes:
        objects.append(dict(kind=kind, pos=pos, fixed=fixed))

    groups = _components(actual)
    split_groups = []
    for group in groups:
        labels = {}
        for p in group:
            x, y = p
            m = int(mark[y, x])
            if m in (0, 9, 5):
                labels.setdefault(m, set()).add(p)
        if 0 in labels and 9 in labels:
            pieces = []
            for label in (0, 9, 5):
                pieces.extend(_components(labels.get(label, set())))
            assigned = set().union(*pieces) if pieces else set()
            for p in group - assigned:
                i = min(range(len(pieces)), key=lambda k: min(
                    abs(p[0]-q[0])+abs(p[1]-q[1]) for q in pieces[k]))
                pieces[i].add(p)
            split_groups.extend(pieces)
        else:
            split_groups.append(group)

    for group in split_groups:
        variants, cx, cy = _variants(group)
        vals = [int(mark[y, x]) for x, y in group if mark[y, x] != 11]
        fixed = bool(vals) and 5 in vals and 0 not in vals and 9 not in vals
        objects.append(dict(kind="s", cx=cx, cy=cy, r=0,
                            variants=variants, fixed=fixed,
                            mode=3, rot=None))
    return dict(width=width, height=height, objects=objects,
                goals=set(goals))


def _source_cells(o, width, height, state=None):
    if state is None:
        cx, cy, r = o["cx"], o["cy"], o["r"]
    else:
        cx, cy, r = state
    offsets = o["variants"][r][0]
    return {(cx+x, cy+y) for x, y in offsets
            if 0 <= cx+x < width and 0 <= cy+y < height}


def _axis_values(world):
    vs, hs = [], []
    for o in world["objects"]:
        if o["kind"] == "v":
            vs.append(o["pos"])
        elif o["kind"] == "h":
            hs.append(o["pos"])
    return vs, hs


def _orbit(p, vs, hs, mode):
    if len(vs) <= 1 and len(hs) <= 1:
        xs = {p[0]}
        ys = {p[1]}
        if mode & 1 and vs:
            xs.add(2*vs[0]-p[0])
        if mode & 2 and hs:
            ys.add(2*hs[0]-p[1])
        return [(x, y) for x in xs for y in ys]

    seen = {p}
    q = deque([(p, 0)])
    while q:
        (x, y), depth = q.popleft()
        if depth >= 13:
            continue
        nxt = []
        if mode & 1:
            nxt.extend((2*v-x, y) for v in vs)
        if mode & 2:
            nxt.extend((x, 2*h-y) for h in hs)
        for p2 in nxt:
            if p2 not in seen:
                seen.add(p2)
                q.append((p2, depth+1))
    return list(seen)


def _footprint(o, world, mode=None, state=None):
    width, height = world["width"], world["height"]
    vs, hs = _axis_values(world)
    result = set()
    for p in _source_cells(o, width, height, state):
        for x, y in _orbit(p, vs, hs, o["mode"] if mode is None else mode):
            if 0 <= x < width and 0 <= y < height:
                result.add((x, y))
    return result


def _synchronize(world, decoded, last):
    base, mark, actual, reflected, goals, axes = decoded
    width, height = world["width"], world["height"]
    changed = False
    world["goals"].update(goals)

    unused = list(axes)
    for o in world["objects"]:
        if o["kind"] == "s":
            continue
        candidates = [(abs(pos-o["pos"]), j, pos)
                      for j, (kind, pos, fixed) in enumerate(unused)
                      if kind == o["kind"]]
        if candidates:
            _, j, pos = min(candidates)
            if pos != o["pos"]:
                changed = True
                o["pos"] = pos
            unused.pop(j)

    shapes = [(i, o) for i, o in enumerate(world["objects"])
              if o["kind"] == "s"]
    predicted = set()
    for i, o in shapes:
        predicted.update(_source_cells(o, width, height))

    if predicted != actual:
        changed = True
        explained = set()
        pending = []
        for i, o in sorted(shapes, key=lambda z: -len(z[1]["variants"][0][0])):
            cells = _source_cells(o, width, height)
            if cells and cells <= actual:
                explained.update(cells)
            else:
                pending.append((i, o))

        for i, o in pending:
            best = None
            for r in range(4):
                for cy in range(o["cy"]-4, o["cy"]+5):
                    for cx in range(o["cx"]-4, o["cx"]+5):
                        cells = _source_cells(o, width, height, (cx, cy, r))
                        if not cells or not cells <= actual:
                            continue
                        visible = len(cells)
                        full = len(o["variants"][r][0])
                        if visible < max(1, full*3//4):
                            continue
                        fresh = len(cells - explained)
                        distance = abs(cx-o["cx"]) + abs(cy-o["cy"])
                        score = (fresh, visible, -distance,
                                 int(r == o["r"]))
                        if best is None or score > best[0]:
                            best = (score, cx, cy, r, cells)
            if best is not None:
                _, cx, cy, r, cells = best
                if r != o["r"] and last is not None:
                    moving = world["objects"][last[0]]
                    dx, dy = last[1], last[2]
                    if moving["kind"] == "v":
                        o["rot"] = "v"
                    elif moving["kind"] == "h":
                        o["rot"] = "h"
                    elif last[0] == i:
                        o["rot"] = "v" if dx else "h"
                o["cx"], o["cy"], o["r"] = cx, cy, r
                explained.update(cells)

    occupied = actual | reflected
    for i, o in shapes:
        scored = []
        for mode in (3, 1, 2, 0):
            fp = _footprint(o, world, mode)
            extras = len(fp - occupied)
            explains = len(fp & reflected)
            scored.append((-extras, explains, int(mode == o["mode"]),
                           mode.bit_count(), mode))
        mode = max(scored)[-1]
        if mode != o["mode"]:
            o["mode"] = mode
            changed = True
    return changed


def _selected(world, base, mark, fallback):
    width, height = world["width"], world["height"]
    scores = []
    for i, o in enumerate(world["objects"]):
        if o["kind"] == "s":
            cells = _source_cells(o, width, height)
            score = sum(mark[y, x] == 0 and base[y, x] == 5
                        for x, y in cells)
        elif o["kind"] == "v":
            x = o["pos"]
            score = sum(mark[y, x] == 0 and base[y, x] == 10
                        for y in range(height))
        else:
            y = o["pos"]
            score = sum(mark[y, x] == 0 and base[y, x] == 10
                        for x in range(width))
        if score:
            scores.append((score, i))
    return max(scores)[1] if scores else fallback


def _legal_shape_step(o, state, dx, dy, world):
    cx, cy, r = state
    offsets, width, height = o["variants"][r]
    nx, ny = cx+dx, cy+dy
    left, top = nx-width//2, ny-height//2
    if left < 0 or top < 0:
        return None
    if left+width > world["width"] or top+height > world["height"]:
        return None
    if (o["rot"] == "v" and dx) or (o["rot"] == "h" and dy):
        r = (r+1) % 4
    return nx, ny, r


def _apply(world, oid, action):
    dx, dy = _DIRS[action]
    o = world["objects"][oid]
    if o["kind"] == "s":
        nxt = _legal_shape_step(o, (o["cx"], o["cy"], o["r"]),
                                dx, dy, world)
        if nxt is None:
            return False
        o["cx"], o["cy"], o["r"] = nxt
    else:
        delta = dx if o["kind"] == "v" else dy
        limit = world["width"] if o["kind"] == "v" else world["height"]
        if not 0 <= o["pos"]+delta < limit:
            return False
        o["pos"] += delta
        if delta:
            for s in world["objects"]:
                if s["kind"] == "s" and s["rot"] == o["kind"]:
                    s["r"] = (s["r"]+1) % 4
    return True


def _solve(world, selected):
    objects = world["objects"]
    width, height = world["width"], world["height"]
    goals = sorted(world["goals"])
    if not goals:
        return None
    goal_bits = [(p, 1 << j) for j, p in enumerate(goals)]
    all_mask = (1 << len(goals))-1
    axis_ids = [i for i, o in enumerate(objects) if o["kind"] != "s"]
    shape_ids = [i for i, o in enumerate(objects) if o["kind"] == "s"]
    if not shape_ids:
        return None

    ranges = []
    for i in axis_ids:
        o = objects[i]
        limit = width if o["kind"] == "v" else height
        ranges.append([o["pos"]] if o["fixed"] else list(range(limit)))

    axis_combos = []
    for values in product(*ranges):
        cost = sum(abs(pos-objects[i]["pos"]) +
                   int(pos != objects[i]["pos"] and i != selected)
                   for i, pos in zip(axis_ids, values))
        axis_combos.append((cost, values))
    axis_combos.sort()
    if len(axis_combos) > 2000:
        axis_combos = axis_combos[:2000]

    best_cost = 10**9
    best = None
    operations = 0
    search_iterations = 0

    for axis_cost, values in axis_combos:
        if axis_cost >= best_cost:
            break
        if operations > 7000000 or search_iterations > 350000:
            break
        vs = [p for i, p in zip(axis_ids, values)
              if objects[i]["kind"] == "v"]
        hs = [p for i, p in zip(axis_ids, values)
              if objects[i]["kind"] == "h"]
        axis_steps = {
            kind: sum(abs(p-objects[i]["pos"])
                      for i, p in zip(axis_ids, values)
                      if objects[i]["kind"] == kind)
            for kind in ("v", "h")
        }

        fixed_mask = 0
        lists = []
        fixed_targets = {}
        orbit_cache = {}
        for oid in shape_ids:
            o = objects[oid]
            rot = o["rot"]
            astep = axis_steps.get(rot, 0)
            if o["fixed"]:
                r = (o["r"]+astep) % 4
                fp = set()
                for dx, dy in o["variants"][r][0]:
                    p = (o["cx"]+dx, o["cy"]+dy)
                    if 0 <= p[0] < width and 0 <= p[1] < height:
                        fp.update(_orbit(p, vs, hs, o["mode"]))
                mask = 0
                for p, bit in goal_bits:
                    if p in fp:
                        mask |= bit
                fixed_mask |= mask
                fixed_targets[oid] = (o["cx"], o["cy"], r)
                continue

            mode = o["mode"]
            if mode not in orbit_cache:
                orbit_cache[mode] = [
                    (_orbit(p, vs, hs, mode), bit) for p, bit in goal_bits]
            orbits = orbit_cache[mode]
            candidates = {}
            orientations = range(4) if rot else (o["r"],)
            for r in orientations:
                offsets, sw, sh = o["variants"][r]
                xmin, xmax = sw//2, width-sw+sw//2
                ymin, ymax = sh//2, height-sh+sh//2
                masks = {}
                for orbit, bit in orbits:
                    operations += len(orbit)*len(offsets)
                    for gx, gy in orbit:
                        for ox, oy in offsets:
                            cx, cy = gx-ox, gy-oy
                            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                                key = (cx, cy)
                                masks[key] = masks.get(key, 0) | bit
                masks.setdefault((o["cx"], o["cy"]), 0)
                for (cx, cy), mask in masks.items():
                    if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                        continue
                    dx = abs(cx-o["cx"])
                    dy = abs(cy-o["cy"])
                    extra = 0
                    if rot:
                        steps = astep + (dx if rot == "v" else dy)
                        discrepancy = (r-o["r"]-steps) % 4
                        if discrepancy & 1:
                            continue
                        extra = discrepancy
                    movement = dx+dy+extra
                    cost = movement + int(movement > 0 and oid != selected)
                    if axis_cost+cost >= best_cost:
                        continue
                    target = (cx, cy, r)
                    old = candidates.get(mask)
                    if old is None or cost < old[0]:
                        candidates[mask] = (cost, target)
            lists.append((oid, candidates))

        need = all_mask & ~fixed_mask
        if not need:
            if axis_cost < best_cost:
                best_cost = axis_cost
                best = (dict(zip(axis_ids, values)), fixed_targets.copy())
            continue

        reduced = []
        for oid, candidates in lists:
            dedup = {}
            for mask, (cost, target) in candidates.items():
                mask &= need
                if mask not in dedup or cost < dedup[mask][0]:
                    dedup[mask] = (cost, target)
            options = [(cost, mask, target)
                       for mask, (cost, target) in dedup.items()]
            if not options:
                reduced = []
                break
            reduced.append((oid, options))
        if not reduced:
            continue
        maxima = [max(mask.bit_count() for cost, mask, target in opts)
                  for oid, opts in reduced]
        if sum(maxima) < need.bit_count():
            continue

        filtered = []
        for k, (oid, opts) in enumerate(reduced):
            necessary = max(0, need.bit_count() - sum(maxima) + maxima[k])
            opts = [z for z in opts if z[1].bit_count() >= necessary]
            opts.sort(key=lambda z: (z[0], -z[1].bit_count()))
            if not opts:
                filtered = []
                break
            filtered.append((oid, opts))
        if not filtered:
            continue
        filtered.sort(key=lambda z: len(z[1]))
        n = len(filtered)
        unions = [0]*(n+1)
        max_counts = [0]*(n+1)
        minimums = [0]*(n+1)
        for k in range(n-1, -1, -1):
            opts = filtered[k][1]
            union = 0
            for cost, mask, target in opts:
                union |= mask
            unions[k] = unions[k+1] | union
            max_counts[k] = max_counts[k+1] + max(z[1].bit_count() for z in opts)
            minimums[k] = minimums[k+1] + opts[0][0]

        chosen = {}

        def search(k, missing, cost):
            nonlocal best, best_cost, search_iterations
            if search_iterations > 350000:
                return
            if cost+minimums[k] >= best_cost:
                return
            if missing & ~unions[k]:
                return
            if missing.bit_count() > max_counts[k]:
                return
            if k == n:
                if not missing:
                    best_cost = cost
                    targets = fixed_targets.copy()
                    targets.update(chosen)
                    best = (dict(zip(axis_ids, values)), targets)
                return
            oid, opts = filtered[k]
            for c, mask, target in opts:
                search_iterations += 1
                if cost+c+minimums[k+1] >= best_cost:
                    break
                chosen[oid] = target
                search(k+1, missing & ~mask, cost+c)
            chosen.pop(oid, None)

        search(0, need, axis_cost)
    return best


def _shape_path(o, target, world):
    start = (o["cx"], o["cy"], o["r"])
    if start == target:
        return []
    parents = {start: None}
    q = deque([start])
    found = None
    while q:
        state = q.popleft()
        for action, (dx, dy) in _DIRS.items():
            nxt = _legal_shape_step(o, state, dx, dy, world)
            if nxt is None or nxt in parents:
                continue
            parents[nxt] = (state, action)
            if nxt == target:
                found = nxt
                q.clear()
                break
            q.append(nxt)
        if found is not None:
            break
    if found is None:
        return None
    path = []
    while parents[found] is not None:
        found, action = parents[found]
        path.append(action)
    path.reverse()
    return path


def _make_tasks(world, solution, selected):
    if solution is None:
        return None
    axis_targets, shape_targets = solution
    simulated = deepcopy(world)
    tasks = []
    ids = sorted(axis_targets, key=lambda i: i != selected)
    for oid in ids:
        o = simulated["objects"][oid]
        target = axis_targets[oid]
        while o["pos"] != target:
            if o["kind"] == "v":
                action = "ACTION4" if target > o["pos"] else "ACTION3"
            else:
                action = "ACTION2" if target > o["pos"] else "ACTION1"
            if not _apply(simulated, oid, action):
                return None
            tasks.append((oid, action))
    ids = sorted(shape_targets, key=lambda i: i != selected)
    for oid in ids:
        o = simulated["objects"][oid]
        if o["fixed"]:
            continue
        path = _shape_path(o, shape_targets[oid], simulated)
        if path is None:
            return None
        for action in path:
            tasks.append((oid, action))
            _apply(simulated, oid, action)
    return tasks


def _click_point(world, oid, geom):
    o = world["objects"][oid]
    width, height = world["width"], world["height"]
    if o["kind"] == "s":
        cells = list(_source_cells(o, width, height))
    elif o["kind"] == "v":
        cells = [(o["pos"], y) for y in range(height)]
    else:
        cells = [(x, o["pos"]) for x in range(width)]

    other_sources = set()
    other_axes = set()
    for i, other in enumerate(world["objects"]):
        if i == oid:
            continue
        if other["kind"] == "s":
            other_sources.update(_source_cells(other, width, height))
        elif other["kind"] == "v":
            other_axes.update((other["pos"], y) for y in range(height))
        else:
            other_axes.update((x, other["pos"]) for x in range(width))
    cells.sort(key=lambda p: (p in other_sources, p in other_axes,
                              p[1], p[0]))
    if not cells:
        return None
    x, y = cells[0]
    _, _, scale, px, py = geom
    return py+y*scale+scale//2, px+x*scale+scale//2


def _explore(world, memory):
    probes = memory.setdefault("probes", set())
    objects = world["objects"]

    for i, o in enumerate(objects):
        if o["kind"] == "s" or o["fixed"] or ("axis", i) in probes:
            continue
        probes.add(("axis", i))
        limit = world["width"] if o["kind"] == "v" else world["height"]
        positive = o["pos"] < limit//2
        if o["kind"] == "v":
            action = "ACTION4" if positive else "ACTION3"
        else:
            action = "ACTION2" if positive else "ACTION1"
        return [(i, action)]

    for i, o in enumerate(objects):
        if o["kind"] != "s" or o["fixed"]:
            continue
        for dim, choices in (("v", ("ACTION4", "ACTION3")),
                             ("h", ("ACTION2", "ACTION1"))):
            if ("shape", i, dim) in probes:
                continue
            probes.add(("shape", i, dim))
            for action in choices:
                dx, dy = _DIRS[action]
                if _legal_shape_step(o, (o["cx"], o["cy"], o["r"]),
                                     dx, dy, world) is not None:
                    return [(i, action)]

    visits = memory.setdefault("visits", {})
    best = None
    for i, o in enumerate(objects):
        if o["fixed"]:
            continue
        for action in _DIRS:
            trial = deepcopy(world)
            if not _apply(trial, i, action):
                continue
            fp = set()
            for s in trial["objects"]:
                if s["kind"] == "s":
                    fp.update(_footprint(s, trial))
            if not fp:
                continue
            missing = trial["goals"] - fp
            distance = sum(min(abs(gx-x)+abs(gy-y) for x, y in fp)
                           for gx, gy in missing)
            signature = tuple(
                (s["pos"],) if s["kind"] != "s"
                else (s["cx"], s["cy"], s["r"])
                for s in trial["objects"])
            score = (len(missing)*30 + distance +
                     visits.get(signature, 0)*15)
            if best is None or score < best[0]:
                best = (score, i, action, signature)
    if best is not None:
        _, i, action, signature = best
        visits[signature] = visits.get(signature, 0)+1
        return [(i, action)]
    return []


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=int)
    available = set(available)
    if "geom" not in memory:
        memory["geom"] = _geometry(obs)
    geom = memory["geom"]
    decoded = _decode(obs, geom)
    base, mark = decoded[:2]

    if "world" not in memory:
        memory["world"] = _initial_world(decoded, geom)
        memory["tasks"] = []
        memory["selected"] = None

    world = memory["world"]
    changed = _synchronize(world, decoded, memory.get("last"))
    memory["last"] = None
    selected = _selected(world, base, mark, memory.get("selected"))
    memory["selected"] = selected
    if changed:
        memory["tasks"] = []

    if not memory.get("tasks"):
        solution = _solve(world, selected)
        tasks = _make_tasks(world, solution, selected)
        if tasks is None:
            tasks = _explore(world, memory)
        if not tasks:
            # A covered state still needs a movement to trigger the goal check.
            for i, o in enumerate(world["objects"]):
                if o["kind"] != "s" and not o["fixed"]:
                    action = "ACTION1" if o["kind"] == "v" else "ACTION3"
                    tasks = [(i, action)]
                    break
        if not tasks:
            tasks = _explore(world, memory)
        memory["tasks"] = tasks

    tasks = memory.get("tasks", [])
    if not tasks:
        action = "ACTION5" if "ACTION5" in available else next(iter(available))
        return action, None, None

    oid, action = tasks[0]
    if selected != oid:
        point = _click_point(world, oid, geom)
        repeated = memory.get("click_for") == (selected, oid)
        if "ACTION6" in available and point is not None and not repeated:
            memory["click_for"] = (selected, oid)
            memory["selected"] = oid
            return "ACTION6", int(point[0]), int(point[1])
        if "ACTION5" in available:
            memory["click_for"] = None
            return "ACTION5", None, None

    memory["click_for"] = None
    tasks.pop(0)
    if action not in available:
        memory["tasks"] = []
        fallback = "ACTION5" if "ACTION5" in available else next(iter(available))
        return fallback, None, None

    dx, dy = _DIRS[action]
    if _apply(world, oid, action):
        memory["last"] = (oid, dx, dy)
    else:
        memory["tasks"] = []
    memory["selected"] = oid
    return action, None, None

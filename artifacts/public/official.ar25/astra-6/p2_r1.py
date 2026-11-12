import math
import time
import itertools
from collections import deque
import numpy as np


_DIRS = [
    ("ACTION1", 0, -1),
    ("ACTION2", 0, 1),
    ("ACTION3", -1, 0),
    ("ACTION4", 1, 0),
]
_DELTA = {a: (dx, dy) for a, dx, dy in _DIRS}


def _read_frame(obs):
    a = np.asarray(obs, dtype=np.int16)
    useful = a != 5
    useful[:, 63] = False
    yy, xx = np.nonzero(useful)

    if len(xx):
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, 63, 63

    bw, bh = x1 - x0, y1 - y0
    choices = []
    for s in range(1, 9):
        if bw % s or bh % s:
            continue
        w, h = bw // s, bh // s
        if not w or not h or min(64 // w, 64 // h) != s:
            continue
        choices.append((s, w, h))

    if not choices:
        choices = [(3, 21, 21)]
        x0 = y0 = 0

    best = None
    for s, w, h in choices:
        base = np.empty((h, w), dtype=np.int16)
        dot = np.empty((h, w), dtype=np.int16)
        px = math.ceil((64 - w * s) / 2) - x0
        py = math.ceil((64 - h * s) / 2) - y0
        if s == 3:
            mx, my = px, py
            holes = [(mx, my)]
        elif s == 4:
            mx, my = px + 1, py + 1
            holes = [(mx, my), (mx + 1, my),
                     (mx, my + 1), (mx + 1, my + 1)]
        elif s == 5:
            mx, my = px + 2, py + 2
            holes = [(mx, my)]
        else:
            mx = my = s // 2
            holes = []

        error = 0
        for y in range(h):
            for x in range(w):
                tile = a[y0 + y*s:y0 + (y+1)*s,
                         x0 + x*s:x0 + (x+1)*s]
                if tile.shape != (s, s):
                    base[y, x] = dot[y, x] = 9
                    error += s*s
                    continue
                color = int(np.bincount(tile.ravel(), minlength=16).argmax())
                base[y, x] = color
                dot[y, x] = int(tile[min(s-1, max(0, my)),
                                           min(s-1, max(0, mx))])
                wrong = tile != color
                for hx, hy in holes:
                    if 0 <= hx < s and 0 <= hy < s:
                        wrong[hy, hx] = False
                error += int(wrong.sum())
        score = error / max(1, w*h*s*s) + 0.002 / s
        if best is None or score < best[0]:
            best = (score, s, w, h, base, dot)

    _, s, w, h, base, dot = best
    axes = {}
    fixed_axes = {}
    selected_axis = None

    cc = (base == 10).sum(axis=0)
    rr = (base == 10).sum(axis=1)
    if len(cc) and int(cc.max()) >= max(3, h // 3):
        x = int(cc.argmax())
        axes["v"] = x
        ds = dot[:, x][base[:, x] == 10]
        fixed_axes["v"] = not bool(np.any((ds == 0) | (ds == 9)))
        if int((ds == 0).sum()) >= 2:
            selected_axis = "v"
    if len(rr) and int(rr.max()) >= max(3, w // 3):
        y = int(rr.argmax())
        axes["h"] = y
        ds = dot[y, :][base[y, :] == 10]
        fixed_axes["h"] = not bool(np.any((ds == 0) | (ds == 9)))
        if int((ds == 0).sum()) >= 2:
            selected_axis = "h"

    targets = set()
    real = set()
    occupied = set()
    excluded = {0, 4, 9, 10, 11}
    for y in range(h):
        for x in range(w):
            b, d = int(base[y, x]), int(dot[y, x])
            if b == 11 or d == 11:
                targets.add((x, y))
            if b not in excluded:
                real.add((x, y))
                occupied.add((x, y))
            elif b == 4:
                occupied.add((x, y))

    return dict(w=w, h=h, s=s, x0=x0, y0=y0, base=base, dot=dot,
                axes=axes, fixed_axes=fixed_axes,
                selected_axis=selected_axis, targets=targets,
                real=real, occupied=occupied)


def _make_rotations(points, w, h):
    out = []
    p = tuple(sorted(points))
    for _ in range(4):
        out.append((p, w, h))
        p = tuple(sorted((y, w - 1 - x) for x, y in p))
        w, h = h, w
    return tuple(out)


def _cells(obj, pose=None):
    cx, cy, r = pose if pose is not None else (obj["cx"], obj["cy"], obj["r"])
    p, w, h = obj["rots"][r]
    ox, oy = cx - w // 2, cy - h // 2
    return {(ox + x, oy + y) for x, y in p}


def _kind(obj):
    return 0 if 0 in obj["rk"] else min(obj["rk"])


def _mode(obj):
    return 3 if 3 in obj["mk"] else min(obj["mk"])


def _orbit(cells, axes, mode, w, h):
    result = set()
    vx, hy = axes.get("v"), axes.get("h")
    for x, y in cells:
        if not (0 <= x < w and 0 <= y < h):
            continue
        xs = (x, 2*vx-x) if vx is not None and mode & 1 else (x,)
        ys = (y, 2*hy-y) if hy is not None and mode & 2 else (y,)
        for xx in xs:
            for yy in ys:
                if 0 <= xx < w and 0 <= yy < h:
                    result.add((xx, yy))
    return result


def _new_scene(rd):
    remaining = set(rd["real"])
    comps = []
    while remaining:
        p = remaining.pop()
        comp = {p}
        q = [p]
        while q:
            x, y = q.pop()
            for z in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if z in remaining:
                    remaining.remove(z)
                    comp.add(z)
                    q.append(z)
        comps.append(comp)

    separated = []
    for comp in comps:
        categories = {}
        for x, y in comp:
            d = int(rd["dot"][y, x])
            if d == 0:
                ax = rd["selected_axis"]
                if ((ax == "v" and x == rd["axes"].get("v")) or
                        (ax == "h" and y == rd["axes"].get("h"))):
                    d = 9
            if d in (0, 9, 5):
                categories[(x, y)] = d

        kinds = set(categories.values())
        if len(kinds) <= 1:
            separated.append((comp, next(iter(kinds), 9)))
        else:
            labels = dict(categories)
            q = deque(categories)
            while q:
                x, y = q.popleft()
                for z in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if z in comp and z not in labels:
                        labels[z] = labels[(x, y)]
                        q.append(z)
            for k in sorted(kinds):
                part = {p for p in comp if labels.get(p) == k}
                if part:
                    separated.append((part, k))

    objects = []
    selected = rd["selected_axis"]
    for comp, cat in separated:
        minx = min(x for x, y in comp)
        miny = min(y for x, y in comp)
        w = max(x for x, y in comp) - minx + 1
        h = max(y for x, y in comp) - miny + 1
        p = [(x-minx, y-miny) for x, y in comp]
        obj = dict(rots=_make_rotations(p, w, h),
                   cx=minx+w//2, cy=miny+h//2, r=0,
                   rk={0, 1, 2}, mk={1, 2, 3}, fixed=(cat == 5))
        if cat == 0 and selected is None:
            selected = len(objects)
        objects.append(obj)

    return dict(objects=objects, axes=dict(rd["axes"]),
                fixed_axes=dict(rd["fixed_axes"]),
                targets=set(rd["targets"]), w=rd["w"], h=rd["h"],
                sel=selected)


def _detect_selection(st, rd):
    if rd["selected_axis"] is not None:
        return rd["selected_axis"]
    for i, obj in enumerate(st["objects"]):
        for x, y in _cells(obj):
            if (0 <= x < rd["w"] and 0 <= y < rd["h"]
                    and int(rd["dot"][y, x]) == 0):
                return i
    return st.get("sel")


def _learn_modes(st, rd):
    objects = st["objects"]
    if not objects:
        return
    options = []
    for obj in objects:
        possibilities = []
        for m in obj["mk"]:
            cover = _orbit(_cells(obj), st["axes"], m, st["w"], st["h"])
            if not (cover - rd["occupied"]):
                possibilities.append((m, cover))
        if not possibilities:
            return
        options.append(possibilities)

    count = math.prod(len(o) for o in options)
    if count <= 729:
        accepted = [set() for _ in objects]
        found = False
        for combo in itertools.product(*options):
            union = set()
            for m, c in combo:
                union.update(c)
            if union == rd["occupied"]:
                found = True
                for i, (m, c) in enumerate(combo):
                    accepted[i].add(m)
        if found:
            for obj, modes in zip(objects, accepted):
                obj["mk"] = modes
            return
    for obj, opts in zip(objects, options):
        obj["mk"] = {m for m, c in opts}


def _update_scene(st, rd, last):
    if (st["w"] != rd["w"] or st["h"] != rd["h"]
            or st["targets"] != rd["targets"]
            or set(st["axes"]) != set(rd["axes"])):
        return _new_scene(rd), True

    old_axes = st["axes"]
    new_axes = rd["axes"]
    action, selected = last if last else (None, None)
    dx, dy = _DELTA.get(action, (0, 0))

    for i, obj in enumerate(st["objects"]):
        oldcx, oldcy, oldr = obj["cx"], obj["cy"], obj["r"]
        cx, cy = oldcx, oldcy
        _, ow, oh = obj["rots"][oldr]
        if selected == i and not obj["fixed"] and (dx or dy):
            nx, ny = cx + dx, cy + dy
            bx, by = nx - ow//2, ny - oh//2
            if 0 <= bx and bx + ow <= st["w"] and 0 <= by and by + oh <= st["h"]:
                cx, cy = nx, ny

        trials = []
        for k in obj["rk"]:
            r = oldr
            axis = "v" if k == 1 else "h" if k == 2 else None
            if axis is not None and axis in old_axes and axis in new_axes:
                before = oldcx if k == 1 else oldcy
                after = cx if k == 1 else cy
                if abs(before-old_axes[axis]) != abs(after-new_axes[axis]):
                    r = (r + 1) % 4
            cells = _cells(obj, (cx, cy, r))
            cells = {(x, y) for x, y in cells
                     if 0 <= x < st["w"] and 0 <= y < st["h"]}
            score = len(cells - rd["real"])
            trials.append((score, r, k))

        best_score = min(t[0] for t in trials)
        best_trials = [t for t in trials if t[0] == best_score]
        rotations = {t[1] for t in best_trials}
        r = oldr if oldr in rotations else min(rotations)

        if best_score:
            old_cells = {(x, y) for x, y in _cells(obj)
                         if 0 <= x < st["w"] and 0 <= y < st["h"]}
            if not (old_cells - rd["real"]):
                cx, cy, r = oldcx, oldcy, oldr
            else:
                obj["rk"] = {k for score, rr, k in best_trials if rr == r}
        else:
            obj["rk"] = {k for score, rr, k in best_trials if rr == r}
        obj["cx"], obj["cy"], obj["r"] = cx, cy, r

    st["axes"] = dict(new_axes)
    st["fixed_axes"] = dict(rd["fixed_axes"])
    predicted = set()
    for obj in st["objects"]:
        predicted.update((x, y) for x, y in _cells(obj)
                         if 0 <= x < st["w"] and 0 <= y < st["h"])
    if len(predicted ^ rd["real"]) > max(3, len(rd["real"]) // 3):
        return _new_scene(rd), True

    st["sel"] = _detect_selection(st, rd)
    return st, False


def _signature(st):
    return (
        tuple((_kind(o), _mode(o), o["r"] if _kind(o) == 0 else -1,
               o["fixed"]) for o in st["objects"]),
        tuple(sorted(st["fixed_axes"].items())),
    )


def _relative_shape(obj, r):
    p, w, h = obj["rots"][r]
    return frozenset((x-w//2, y-h//2) for x, y in p)


def _route(obj, start, goal, w, h, kind):
    sx, sy, sr = start
    gx, gy, gr = goal
    acceptable = {r for r in range(4)
                  if _relative_shape(obj, r) == _relative_shape(obj, gr)}
    if sx == gx and sy == gy and sr in acceptable:
        return []
    if kind == 0:
        if sr not in acceptable:
            return None
        _, sw, sh = obj["rots"][sr]
        for x, y in ((sx, sy), (gx, gy)):
            if not (0 <= x-sw//2 and x-sw//2+sw <= w
                    and 0 <= y-sh//2 and y-sh//2+sh <= h):
                return None
        result = []
        result.extend(["ACTION4" if gx > sx else "ACTION3"] * abs(gx-sx))
        result.extend(["ACTION2" if gy > sy else "ACTION1"] * abs(gy-sy))
        return result

    start = (sx, sy, sr)
    q = deque([start])
    previous = {start: None}
    reached = None
    while q:
        cx, cy, r = node = q.popleft()
        _, ow, oh = obj["rots"][r]
        for action, dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            bx, by = nx - ow//2, ny - oh//2
            if not (0 <= bx and bx+ow <= w and 0 <= by and by+oh <= h):
                continue
            nr = (r + int((kind == 1 and dx != 0)
                          or (kind == 2 and dy != 0))) % 4
            nxt = (nx, ny, nr)
            if nxt in previous:
                continue
            previous[nxt] = (node, action)
            if nx == gx and ny == gy and nr in acceptable:
                reached = nxt
                q.clear()
                break
            q.append(nxt)
        if reached is not None:
            break

    if reached is None:
        return None
    result = []
    while previous[reached] is not None:
        reached, action = previous[reached]
        result.append(action)
    result.reverse()
    return result


class _Expired(Exception):
    pass


def _solve(st, deadline, resume=0):
    objects = st["objects"]
    targets = sorted(st["targets"])
    n = len(targets)
    if not n or not objects:
        return None, True, 0

    w, h = st["w"], st["h"]
    index = {p: i for i, p in enumerate(targets)}
    allmask = (1 << n) - 1
    old_axes = st["axes"]
    selected = st.get("sel")
    best_plan = None
    best_cost = 10**9

    def domain(axis, size):
        if axis not in old_axes:
            return [None]
        if st["fixed_axes"].get(axis, False):
            return [old_axes[axis]]
        return list(range(size))

    vx = domain("v", w)
    hy = domain("h", h)
    target_set = set(targets)

    def symmetry(axis, value):
        if value is None:
            return 0
        if axis == "v":
            return sum((2*value-x, y) in target_set for x, y in targets)
        return sum((x, 2*value-y) in target_set for x, y in targets)

    vscore = {v: symmetry("v", v) for v in vx}
    hscore = {v: symmetry("h", v) for v in hy}

    def axis_distance(v, y):
        return ((abs(v-old_axes["v"]) if v is not None else 0)
                + (abs(y-old_axes["h"]) if y is not None else 0))

    configs = list(itertools.product(vx, hy))
    configs.sort(key=lambda c: (
        -vscore[c[0]]-hscore[c[1]] + 0.12*axis_distance(*c),
        axis_distance(*c)))

    def mask_at(obj, pose, axes, mode):
        mask = 0
        for p in _orbit(_cells(obj, pose), axes, mode, w, h):
            j = index.get(p)
            if j is not None:
                mask |= 1 << j
        return mask

    ci = resume
    try:
        for ci in range(resume, len(configs)):
            if time.perf_counter() > deadline:
                raise _Expired()
            v, yaxis = configs[ci]
            axes = {}
            if v is not None:
                axes["v"] = v
            if yaxis is not None:
                axes["h"] = yaxis

            axis_cost = 0
            for key, value in axes.items():
                d = abs(value-old_axes[key])
                if d:
                    axis_cost += d + int(selected != key)
            if axis_cost >= best_cost:
                continue

            inverse = {}
            for mode in { _mode(o) for o in objects }:
                rows = []
                for j, (tx, ty) in enumerate(targets):
                    xs = (tx, 2*v-tx) if v is not None and mode & 1 else (tx,)
                    ys = (ty, 2*yaxis-ty) if yaxis is not None and mode & 2 else (ty,)
                    rows.append((1 << j, set(itertools.product(xs, ys))))
                inverse[mode] = rows

            candidates = []
            defaults = []
            starts = []
            effective_kinds = []
            failed = False

            for oi, obj in enumerate(objects):
                kind, mode = _kind(obj), _mode(obj)
                axis = "v" if kind == 1 else "h" if kind == 2 else None
                effective = kind if axis in axes else 0
                effective_kinds.append(effective)
                ad = abs(axes[axis]-old_axes[axis]) if effective else 0
                ar = (obj["r"] + ad) % 4
                start = (obj["cx"], obj["cy"], ar)
                starts.append(start)
                default = (mask_at(obj, start, axes, mode), 0, *start)
                defaults.append(default)

                if obj["fixed"]:
                    candidates.append([default])
                    continue

                bymask = {default[0]: default}
                orientations = range(4) if effective else [obj["r"]]
                for r in orientations:
                    if time.perf_counter() > deadline:
                        raise _Expired()
                    points, sw, sh = obj["rots"][r]
                    xmax, ymax = w-sw, h-sh
                    if xmax < 0 or ymax < 0:
                        continue
                    placements = {}
                    for bit, images in inverse[mode]:
                        for tx, ty in images:
                            for px, py in points:
                                bx, by = tx-px, ty-py
                                if 0 <= bx <= xmax and 0 <= by <= ymax:
                                    code = by*w+bx
                                    placements[code] = placements.get(code, 0) | bit

                    for code, mask in placements.items():
                        by, bx = divmod(code, w)
                        cx, cy = bx+sw//2, by+sh//2
                        dist = abs(cx-obj["cx"]) + abs(cy-obj["cy"])
                        if effective:
                            d = (abs(cx-obj["cx"]) if effective == 1
                                 else abs(cy-obj["cy"]))
                            difference = (r-obj["r"]-ad-d) % 4
                            if difference & 1:
                                continue
                            dist += difference
                        cost = dist + int(dist > 0 and selected != oi)
                        if cost + axis_cost >= best_cost:
                            continue
                        cand = (mask, cost, cx, cy, r)
                        old = bymask.get(mask)
                        if old is None or (cost, r != ar) < (old[1], old[4] != ar):
                            bymask[mask] = cand
                candidates.append(list(bymask.values()))

            possible = []
            for cs in candidates:
                union = 0
                for c in cs:
                    union |= c[0]
                possible.append(union)
            union = 0
            for p in possible:
                union |= p
            if union != allmask:
                continue
            if sum(max(c[0].bit_count() for c in cs) for cs in candidates) < n:
                continue

            # Targets that only one object can reach are mandatory for it.
            for _ in range(2):
                for i in range(len(objects)):
                    others = 0
                    for j, p in enumerate(possible):
                        if i != j:
                            others |= p
                    mandatory = allmask & ~others
                    if mandatory:
                        candidates[i] = [c for c in candidates[i]
                                         if c[0] & mandatory == mandatory]
                        if not candidates[i]:
                            failed = True
                            break
                        u = 0
                        for c in candidates[i]:
                            u |= c[0]
                        possible[i] = u
                if failed:
                    break
            if failed:
                continue

            posts = []
            maxima = []
            for cs in candidates:
                cs.sort(key=lambda c: (c[1], -c[0].bit_count()))
                maxima.append(max(c[0].bit_count() for c in cs))
                table = [[] for _ in targets]
                for c in cs:
                    bits = c[0]
                    while bits:
                        bit = bits & -bits
                        table[bit.bit_length()-1].append(c)
                        bits -= bit
                posts.append(table)

            memo = {}
            node_count = [0]
            assignment = {}

            def visit(left, missing, cost):
                nonlocal best_cost, best_plan
                node_count[0] += 1
                if node_count[0] % 128 == 0 and time.perf_counter() > deadline:
                    raise _Expired()
                if cost + axis_cost >= best_cost:
                    return
                if not missing:
                    poses = [tuple(c[2:5]) for c in defaults]
                    actual_cost = axis_cost
                    for i, c in assignment.items():
                        pose = tuple(c[2:5])
                        poses[i] = pose
                        route = _route(objects[i], starts[i], pose, w, h,
                                       effective_kinds[i])
                        if route is None:
                            return
                        actual_cost += len(route) + int(bool(route) and selected != i)
                    if actual_cost < best_cost:
                        best_cost = actual_cost
                        best_plan = dict(axes=dict(axes), poses=poses,
                                         cost=actual_cost)
                    return
                if not left:
                    return
                if missing.bit_count() > sum(maxima[i] for i in left):
                    return
                u = 0
                for i in left:
                    u |= possible[i]
                if missing & ~u:
                    return
                key = (left, missing)
                if memo.get(key, 10**9) <= cost:
                    return
                memo[key] = cost

                bits = missing
                chosen_bit = None
                smallest = 10**9
                while bits:
                    bit = bits & -bits
                    j = bit.bit_length()-1
                    count = sum(len(posts[i][j]) for i in left)
                    if count < smallest:
                        smallest, chosen_bit = count, j
                    bits -= bit
                if not smallest:
                    return

                for i in left:
                    rest = tuple(j for j in left if j != i)
                    for c in posts[i][chosen_bit]:
                        if cost+c[1]+axis_cost >= best_cost:
                            break
                        if not rest and missing & ~c[0]:
                            continue
                        assignment[i] = c
                        visit(rest, missing & ~c[0], cost+c[1])
                        del assignment[i]

            visit(tuple(range(len(objects))), allmask, 0)

        return best_plan, True, len(configs)
    except _Expired:
        return best_plan, False, ci


def _click_location(st, rd, key):
    w, h = st["w"], st["h"]
    source_sets = [_cells(o) for o in st["objects"]]
    if isinstance(key, int):
        cells = source_sets[key]
        others = set()
        for i, c in enumerate(source_sets):
            if i != key:
                others.update(c)
        choices = sorted(cells, key=lambda p: (
            p in others,
            int(rd["base"][p[1], p[0]]) == 4
            if 0 <= p[0] < w and 0 <= p[1] < h else True))
    elif key == "v":
        x = st["axes"]["v"]
        all_sources = set().union(*source_sets) if source_sets else set()
        choices = sorted(((x, y) for y in range(h)),
                         key=lambda p: (p in all_sources,
                                        p[1] == st["axes"].get("h")))
    elif key == "h":
        y = st["axes"]["h"]
        all_sources = set().union(*source_sets) if source_sets else set()
        choices = sorted(((x, y) for x in range(w)),
                         key=lambda p: (p in all_sources,
                                        p[0] == st["axes"].get("v")))
    else:
        choices = [(w-1, h-1)]

    for x, y in choices:
        if 0 <= x < w and 0 <= y < h:
            row = rd["y0"] + y*rd["s"] + rd["s"]//2
            col = rd["x0"] + x*rd["s"] + rd["s"]//2
            return min(63, row), min(63, col)
    return 62, 62


def _execute_plan(st, plan):
    axes_to_move = [k for k, v in plan["axes"].items()
                    if st["axes"].get(k) != v]
    if axes_to_move:
        axes_to_move.sort(key=lambda k: (
            st.get("sel") != k,
            -abs(plan["axes"][k]-st["axes"][k])))
        key = axes_to_move[0]
        delta = plan["axes"][key]-st["axes"][key]
        action = (("ACTION4" if delta > 0 else "ACTION3") if key == "v"
                  else ("ACTION2" if delta > 0 else "ACTION1"))
        return key, action

    choices = []
    for i, (obj, goal) in enumerate(zip(st["objects"], plan["poses"])):
        start = (obj["cx"], obj["cy"], obj["r"])
        k = _kind(obj)
        axis = "v" if k == 1 else "h" if k == 2 else None
        if axis not in st["axes"]:
            k = 0
        route = _route(obj, start, goal, st["w"], st["h"], k)
        if route is None:
            return None
        if route:
            if obj["fixed"]:
                return None
            choices.append((st.get("sel") != i, len(route), i, route[0]))
    if choices:
        _, _, key, action = min(choices)
        return key, action
    return ("done", None)


def _probe(st):
    targets = st["targets"]
    mx = (min(x for x, y in targets) + max(x for x, y in targets)) / 2 if targets else st["w"]/2
    my = (min(y for x, y in targets) + max(y for x, y in targets)) / 2 if targets else st["h"]/2
    options = []
    for key, kind, mid, size in (("v", 1, mx, st["w"]), ("h", 2, my, st["h"])):
        if key not in st["axes"] or st["fixed_axes"].get(key):
            continue
        uncertain = sum(kind in o["rk"] and len(o["rk"]) > 1 for o in st["objects"])
        if uncertain:
            p = st["axes"][key]
            sign = 1 if mid > p else -1
            if not 0 <= p+sign < size:
                sign = -sign
            action = (("ACTION4" if sign > 0 else "ACTION3") if key == "v"
                      else ("ACTION2" if sign > 0 else "ACTION1"))
            options.append((st.get("sel") != key, -uncertain, key, action))
    if options:
        _, _, key, action = min(options)
        return key, action

    for i, obj in enumerate(st["objects"]):
        if obj["fixed"]:
            continue
        for kind, axis in ((1, "v"), (2, "h")):
            if kind not in obj["rk"] or len(obj["rk"]) <= 1 or axis not in st["axes"]:
                continue
            _, sw, sh = obj["rots"][obj["r"]]
            mid = mx if kind == 1 else my
            pos = obj["cx"] if kind == 1 else obj["cy"]
            sign = 1 if mid > pos else -1
            for d in (sign, -sign):
                dx, dy = (d, 0) if kind == 1 else (0, d)
                bx = obj["cx"]+dx-sw//2
                by = obj["cy"]+dy-sh//2
                if 0 <= bx and bx+sw <= st["w"] and 0 <= by and by+sh <= st["h"]:
                    action = (("ACTION4" if d > 0 else "ACTION3") if kind == 1
                              else ("ACTION2" if d > 0 else "ACTION1"))
                    return i, action

    # Conservative fallback: bring an object toward the target region.
    for i, obj in enumerate(st["objects"]):
        if obj["fixed"]:
            continue
        _, sw, sh = obj["rots"][obj["r"]]
        moves = sorted(_DIRS, key=lambda z:
                       abs(obj["cx"]+z[1]-mx) + abs(obj["cy"]+z[2]-my))
        for action, dx, dy in moves:
            bx, by = obj["cx"]+dx-sw//2, obj["cy"]+dy-sh//2
            if 0 <= bx and bx+sw <= st["w"] and 0 <= by and by+sh <= st["h"]:
                return i, action
    return st.get("sel"), "ACTION5"


def act(obs, available, memory):
    started = time.perf_counter()
    rd = _read_frame(obs)

    st = memory.get("scene")
    if st is None:
        st = _new_scene(rd)
        reset_model = True
    else:
        st, reset_model = _update_scene(st, rd, memory.get("last"))
    memory["scene"] = st

    if reset_model:
        memory.pop("plan", None)
        memory.pop("search", None)
        memory.pop("forced", None)

    _learn_modes(st, rd)
    st["sel"] = _detect_selection(st, rd)
    sig = _signature(st)

    def emit(action, key=None, click=False):
        if click:
            row, col = _click_location(st, rd, key)
            action = "ACTION6"
            if key is not None:
                st["sel"] = key
        else:
            row = col = None
        if action not in available:
            if "ACTION6" in available:
                action, row, col = "ACTION6", 62, 62
            else:
                action = next((a for a in available if a != "RESET"),
                              available[0] if available else "ACTION5")
                row = col = None
        memory["last"] = (action, st.get("sel"))
        return action, row, col

    def perform(key, action, force=False):
        if key is not None and st.get("sel") != key:
            if force:
                memory["forced"] = (key, action)
            return emit("ACTION6", key, True)
        return emit(action)

    forced = memory.pop("forced", None)
    if forced is not None:
        return perform(forced[0], forced[1], True)

    plan = memory.get("plan")
    if plan is not None and plan.get("signature") != sig:
        memory.pop("plan", None)
        plan = None

    if plan is not None:
        step = _execute_plan(st, plan)
        if step is not None and step[0] != "done":
            return perform(*step)
        if step is None:
            memory.pop("plan", None)
            plan = None
        else:
            # A perpendicular axis action checks the goal without translating it.
            if st["axes"]:
                key = next(iter(st["axes"]))
                action = "ACTION1" if key == "v" else "ACTION3"
                return perform(key, action)
            return emit("ACTION5")

    if not st["objects"]:
        return emit("ACTION5")

    search_key = (
        sig, tuple(sorted(st["axes"].items())),
        tuple((o["cx"], o["cy"], o["r"]) for o in st["objects"]),
        st.get("sel")
    )
    search = memory.get("search")
    resume = search[1] if search and search[0] == search_key else 0
    plan, completed, next_index = _solve(st, started + 7.7, resume)

    if plan is not None:
        plan["signature"] = sig
        memory["plan"] = plan
        memory.pop("search", None)
        step = _execute_plan(st, plan)
        if step is not None and step[0] != "done":
            return perform(*step)
        if step is not None and st["axes"]:
            key = next(iter(st["axes"]))
            return perform(key, "ACTION1" if key == "v" else "ACTION3")
        memory.pop("plan", None)

    if not completed:
        memory["search"] = (search_key, next_index)
        # Selection is free in the game's energy budget and leaves geometry intact.
        return emit("ACTION6", st.get("sel"), True)

    memory.pop("search", None)
    key, action = _probe(st)
    return perform(key, action, True)

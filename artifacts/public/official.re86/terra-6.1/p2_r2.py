import heapq
import itertools
from collections import deque

import numpy as np


DIRS = {
    "ACTION1": (0, -3),
    "ACTION2": (0, 3),
    "ACTION3": (-3, 0),
    "ACTION4": (3, 0),
}


def offsets(kind, a, b):
    p = set()
    if kind == "plus":
        p.update((x, 0) for x in range(-a, a + 1))
        p.update((0, y) for y in range(-b, b + 1))
    elif kind == "hline":
        p.update((x, 0) for x in range(-a, a + 1))
    elif kind == "vline":
        p.update((0, y) for y in range(-b, b + 1))
    elif kind == "x":
        for t in range(-a, a + 1):
            p.add((t, t))
            p.add((t, -t))
    elif kind == "diamond":
        for y in range(-a, a + 1):
            x = a - abs(y)
            p.add((x, y))
            p.add((-x, y))
    elif kind == "rect":
        for x in range(-a, a + 1):
            p.add((x, -b))
            p.add((x, b))
        for y in range(-b, b + 1):
            p.add((-a, y))
            p.add((a, y))
    return p


def stations_and_obstacle(obs):
    stations = []
    used = set()
    for size in (6, 5):
        for y in range(64 - size):
            for x in range(65 - size):
                border = []
                for i in range(size):
                    border.append((x + i, y))
                    border.append((x + i, y + size - 1))
                for i in range(1, size - 1):
                    border.append((x, y + i))
                    border.append((x + size - 1, y + i))
                if sum(obs[yy, xx] == 2 for xx, yy in border) < len(border) - 3:
                    continue
                inner = obs[y + 1:y + size - 1, x + 1:x + size - 1]
                counts = {}
                for value in inner.flat:
                    value = int(value)
                    if value not in (0, 2, 4, 5):
                        counts[value] = counts.get(value, 0) + 1
                if not counts:
                    continue
                color = max(counts, key=counts.get)
                if counts[color] < max(3, inner.size // 2):
                    continue
                if any(abs(x - xx) <= 2 and abs(y - yy) <= 2
                       for xx, yy in used):
                    continue
                stations.append([x, y, size, size, color])
                used.add((x, y))
    obstacle = {
        (x, y) for y in range(63) for x in range(64)
        if obs[y, x] == 1
    }
    return stations, obstacle


def targets(obs):
    result = []
    for y in range(1, 62):
        for x in range(1, 63):
            ring = [
                int(obs[y + dy, x + dx])
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            ]
            if (ring.count(4) >= 3 and 5 not in ring and
                    int(obs[y, x]) != 4):
                result.append([x, y, int(obs[y, x])])
    return result


def peaks(arr, count=5):
    flat = arr.ravel()
    k = min(flat.size, count * 12)
    if not k:
        return []
    inds = np.argpartition(flat, -k)[-k:]
    inds = sorted(inds, key=lambda i: float(flat[i]), reverse=True)
    out = []
    for ind in inds:
        value = float(flat[ind])
        if value < 7:
            break
        y, x = divmod(int(ind), 64)
        if any(abs(x - xx) <= 2 and abs(y - yy) <= 2
               for xx, yy, _ in out):
            continue
        out.append((x, y, value))
        if len(out) == count:
            break
    return out


def shape_candidates(obs, stations):
    station_mask = np.zeros((64, 64), dtype=bool)
    for x, y, w, h, _ in stations:
        station_mask[y:y + h, x:x + w] = True

    colors = []
    for color in range(3, 16):
        if color in (4, 5):
            continue
        n = int(np.count_nonzero(
            (obs[:63] == color) & (~station_mask[:63])
        ))
        if n >= 9:
            colors.append(color)

    candidates = []
    pad = 64

    for color in colors:
        base = np.full((64, 64), -1.25, dtype=np.float32)
        base[(obs != 5) & (obs != 4)] = -0.12
        base[obs == color] = 1.0
        base[obs == 0] = 0.8
        base[station_mask] = 0.0
        base[63, :] = 0.0

        padded = np.pad(base, pad, mode="constant")
        cache = {}

        def sh(dy, dx):
            key = (dy, dx)
            if key not in cache:
                cache[key] = padded[
                    pad + dy:pad + dy + 64,
                    pad + dx:pad + dx + 64
                ]
            return cache[key]

        def add(arr, kind, a, b, count=5):
            for x, y, score in peaks(arr, count):
                aa = int(a(y, x)) if callable(a) else int(a)
                bb = int(b(y, x)) if callable(b) else int(b)
                candidates.append((score, color, kind, aa, bb, x, y))

        h = np.zeros((64, 64), dtype=np.float32)
        v = np.zeros((64, 64), dtype=np.float32)
        best_h = np.full((64, 64), -1e6, dtype=np.float32)
        best_v = np.full((64, 64), -1e6, dtype=np.float32)
        rh = np.zeros((64, 64), dtype=np.int16)
        rv = np.zeros((64, 64), dtype=np.int16)
        hs = [base.copy()]
        vs = [base.copy()]

        for r in range(1, 31):
            h = h + sh(0, r) + sh(0, -r)
            v = v + sh(r, 0) + sh(-r, 0)
            hs.append(base + h)
            vs.append(base + v)
            if r >= 2:
                better = h > best_h + 0.001
                best_h = np.where(better, h, best_h)
                rh = np.where(better, r, rh)
                better = v > best_v + 0.001
                best_v = np.where(better, v, best_v)
                rv = np.where(better, r, rv)

        add(base + best_h + best_v, "plus",
            lambda y, x: rh[y, x],
            lambda y, x: rv[y, x], 14)
        add(base + best_h, "hline", lambda y, x: rh[y, x], 0, 8)
        add(base + best_v, "vline", 0, lambda y, x: rv[y, x], 8)

        for r in range(3, 26):
            add(hs[r] + vs[r] - base + 0.1, "plus", r, r, 3)

        diag = base.copy()
        for r in range(1, 26):
            diag = (diag + sh(r, r) + sh(r, -r) +
                    sh(-r, r) + sh(-r, -r))
            if r >= 3:
                add(diag, "x", r, r, 5)

        for r in range(3, 24):
            arr = np.zeros((64, 64), dtype=np.float32)
            for dy in range(-r, r + 1):
                dx = r - abs(dy)
                arr = arr + sh(dy, dx)
                if dx:
                    arr = arr + sh(dy, -dx)
            arr = arr + (obs == 0).astype(np.float32) * 0.8
            add(arr, "diamond", r, r, 4)

        def shifted(arr, dy, dx):
            p = np.pad(arr, pad, mode="constant")
            return p[
                pad + dy:pad + dy + 64,
                pad + dx:pad + dx + 64
            ]

        for a in range(3, 13):
            for b in range(3, 13):
                arr = (
                    shifted(hs[a], -b, 0) +
                    shifted(hs[a], b, 0) +
                    shifted(vs[b], 0, -a) +
                    shifted(vs[b], 0, a) -
                    sh(-b, -a) - sh(-b, a) -
                    sh(b, -a) - sh(b, a)
                )
                arr = arr + (obs == 0).astype(np.float32) * 0.8
                add(arr, "rect", a, b, 2)

    unique = {}
    for c in candidates:
        score, color, kind, a, b, x, y = c
        key = (color, kind, a, b, x, y)
        if key not in unique or score > unique[key][0]:
            unique[key] = c
    return list(unique.values()), station_mask


def extract_shapes(obs, station_mask, candidates):
    remaining = {
        (x, y)
        for y in range(63)
        for x in range(64)
        if int(obs[y, x]) not in (0, 1, 2, 4, 5)
        and not station_mask[y, x]
    }

    prepared = []
    for _, color, kind, a, b, x, y in candidates:
        if a > 30 or b > 30 or a < 0 or b < 0:
            continue
        off = offsets(kind, a, b)
        visible = [
            (x + dx, y + dy)
            for dx, dy in off
            if 0 <= x + dx < 64 and 0 <= y + dy < 63
        ]
        if len(visible) >= 8:
            prepared.append((color, kind, a, b, x, y, off, visible))

    shapes = []
    for _ in range(8):
        best = None
        best_score = 8.0
        for item in prepared:
            color, kind, a, b, x, y, off, visible = item
            fresh = old = missing = 0
            for px, py in visible:
                if station_mask[py, px]:
                    continue
                value = int(obs[py, px])
                if value == color:
                    if (px, py) in remaining:
                        fresh += 1
                    else:
                        old += 1
                elif value == 0 and px == x and py == y:
                    fresh += 1
                elif value in (4, 5):
                    missing += 1
            if fresh < 9:
                continue
            score = fresh + 0.12 * old - 1.5 * missing
            score -= 0.005 * len(off)
            if kind in ("diamond", "rect") and int(obs[y, x]) == 0:
                score += 1.0
            if score > best_score:
                best_score = score
                best = item

        if best is None:
            break
        color, kind, a, b, x, y, off, visible = best
        shapes.append({
            "color": int(color),
            "kind": kind,
            "a": int(a),
            "b": int(b),
            "x": int(x),
            "y": int(y),
        })
        for px, py in visible:
            if int(obs[py, px]) == color:
                remaining.discard((px, py))

    return shapes


def resolve_targets(obs, goals, shapes):
    known = []
    unknown = []
    offs = [offsets(s["kind"], s["a"], s["b"]) for s in shapes]
    for g in goals:
        x, y, color = g
        covered = False
        for s, off in zip(shapes, offs):
            if (x - s["x"], y - s["y"]) in off:
                covered = True
                break
            if (x, y) == (s["x"], s["y"]) and int(obs[y, x]) == 0:
                covered = True
                break
        if color in (0, 4, 5) or covered:
            unknown.append(g)
        else:
            known.append(g)

    for g in unknown:
        x, y, color = g
        if known:
            nearest = min(known, key=lambda q: abs(q[0] - x) + abs(q[1] - y))
            g[2] = nearest[2]
        elif color not in (0, 4, 5):
            g[2] = color
        elif shapes:
            nearest = min(
                shapes,
                key=lambda s: abs(s["x"] - x) + abs(s["y"] - y)
            )
            g[2] = nearest["color"]

    return [g for g in goals if g[2] not in (0, 4, 5)]


def selected_from_frame(obs, shapes, fallback=None):
    found = []
    for i, s in enumerate(shapes):
        x, y = s["x"], s["y"]
        if 0 <= x < 64 and 0 <= y < 63 and int(obs[y, x]) == 0:
            found.append(i)
    if len(found) == 1:
        return found[0]
    if fallback in found:
        return fallback
    return found[0] if found else fallback


def choose_rotation(obs, shapes, cyclic_order):
    if len(cyclic_order) <= 1:
        return cyclic_order
    offs = {
        i: offsets(shapes[i]["kind"], shapes[i]["a"], shapes[i]["b"])
        for i in cyclic_order
    }
    best = cyclic_order
    best_score = -1
    for k in range(len(cyclic_order)):
        order = cyclic_order[k:] + cyclic_order[:k]
        score = 0
        for a in range(len(order)):
            i = order[a]
            si = shapes[i]
            for b in range(a + 1, len(order)):
                j = order[b]
                sj = shapes[j]
                for dx, dy in offs[i]:
                    x, y = si["x"] + dx, si["y"] + dy
                    if not (0 <= x < 64 and 0 <= y < 63):
                        continue
                    if (x - sj["x"], y - sj["y"]) not in offs[j]:
                        continue
                    value = int(obs[y, x])
                    if value == sj["color"]:
                        score += 1
                    elif value == si["color"]:
                        score -= 1
        if score > best_score:
            best_score = score
            best = order
    return best


def reachable(shape, stations, obstacle):
    off = offsets(shape["kind"], shape["a"], shape["b"])
    collision = set(off)
    collision.add((0, 0))
    x0, y0, color0 = shape["x"], shape["y"], shape["color"]

    positions = {}
    for y in range(y0 % 3, 64, 3):
        for x in range(x0 % 3, 64, 3):
            blocked = any((x + dx, y + dy) in obstacle for dx, dy in collision)
            paints = []
            for sx, sy, w, h, c in stations:
                if any(sx <= x + dx < sx + w and sy <= y + dy < sy + h
                       for dx, dy in collision):
                    paints.append(c)
            positions[(x, y)] = (blocked, paints)

    start = (x0, y0, color0)
    dist = {start: 0}
    prev = {}
    queue = deque([start])
    while queue:
        x, y, color = queue.popleft()
        for action, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            info = positions.get((nx, ny))
            if info is None or info[0]:
                continue
            new_color = color
            for c in info[1]:
                if c != new_color:
                    new_color = c
                    break
            state = (nx, ny, new_color)
            if state not in dist:
                dist[state] = dist[(x, y, color)] + 1
                prev[state] = ((x, y, color), action)
                queue.append(state)
    return off, dist, prev, start


def path_from_prev(prev, start, end):
    actions = []
    while end != start:
        if end not in prev:
            return []
        old, action = prev[end]
        actions.append((action, end))
        end = old
    actions.reverse()
    return actions


def make_plan(shapes, goals, stations, obstacle, order, selected):
    n = len(shapes)
    if not n or not goals or len(goals) > 18:
        return []

    reach = [reachable(s, stations, obstacle) for s in shapes]
    full = (1 << len(goals)) - 1
    best_solution = None

    for final_selected in range(n):
        choices = {}
        for i in order:
            off, dist, prev, start = reach[i]
            by_effect = {}
            for state, cost in dist.items():
                x, y, color = state
                occupied = matching = 0
                for j, (gx, gy, gc) in enumerate(goals):
                    is_center = (gx == x and gy == y)
                    on_shape = (gx - x, gy - y) in off
                    if i == final_selected and is_center:
                        on_shape = True
                    if not on_shape:
                        continue
                    bit = 1 << j
                    occupied |= bit
                    if color == gc and not (i == final_selected and is_center):
                        matching |= bit
                key = (occupied, matching)
                old = by_effect.get(key)
                if old is None or cost < old[0]:
                    by_effect[key] = (cost, state)
            choices[i] = [
                (occ, match, cost, state)
                for (occ, match), (cost, state) in by_effect.items()
            ]

        dp = {0: (0, None)}
        layers = []
        for i in order:
            new = {}
            for mask, (cost0, _) in dp.items():
                for occ, match, cost, state in choices[i]:
                    nm = (mask & ~occ) | match
                    nc = cost0 + cost
                    old = new.get(nm)
                    if old is None or nc < old[0]:
                        new[nm] = (nc, (mask, state))
            if not new:
                break
            if len(new) > 25000:
                keep = sorted(
                    new, key=lambda m: (-m.bit_count(), new[m][0])
                )[:25000]
                new = {m: new[m] for m in keep}
            layers.append(new)
            dp = new

        if len(layers) != len(order) or full not in dp:
            continue

        states = {}
        mask = full
        for pos in range(len(order) - 1, -1, -1):
            i = order[pos]
            old_mask, state = layers[pos][mask][1]
            states[i] = state
            mask = old_mask

        paths = {}
        for i in order:
            _, _, prev, start = reach[i]
            paths[i] = path_from_prev(prev, start, states[i])

        moving = [i for i in order if paths[i]]
        permutations = (
            itertools.permutations(moving)
            if len(moving) <= 6 else [tuple(moving)]
        )
        order_index = {v: k for k, v in enumerate(order)}
        for execution in permutations:
            cur = selected
            switches = 0
            for i in execution:
                switches += (order_index[i] - order_index[cur]) % n
                cur = i
            switches += (order_index[final_selected] - order_index[cur]) % n
            total = dp[full][0] + switches
            if best_solution is None or total < best_solution[0]:
                best_solution = (total, paths, execution, final_selected)

    if best_solution is None:
        return []

    _, paths, execution, final_selected = best_solution
    plan = []
    current = selected
    order_index = {v: k for k, v in enumerate(order)}

    def select(target):
        nonlocal current
        while current != target:
            current = order[(order_index[current] + 1) % n]
            plan.append(["ACTION5", current, None])

    for i in execution:
        select(i)
        for action, state in paths[i]:
            plan.append([action, i, list(state)])
    select(final_selected)
    return plan


def initialize(obs):
    stations, obstacle = stations_and_obstacle(obs)
    candidates, station_mask = shape_candidates(obs, stations)
    shapes = extract_shapes(obs, station_mask, candidates)
    goals = resolve_targets(obs, targets(obs), shapes)
    selected = selected_from_frame(obs, shapes, 0 if shapes else None)
    return {
        "shapes": shapes,
        "goals": goals,
        "stations": stations,
        "obstacle": [list(p) for p in obstacle],
        "selected": selected,
        "cycle": [selected] if selected is not None else [],
        "phase": "discover",
        "discovery_steps": 0,
        "plan": [],
    }


# Obstacle-specific state:
# cross: (left, top, internal_column, internal_row)
# rectangle: (left, top, width, height)


def special_start(shape):
    x = shape["x"] - shape["a"]
    y = shape["y"] - shape["b"]
    if shape["kind"] == "plus":
        return (x, y, shape["a"], shape["b"])
    return (x, y, 2 * shape["a"] + 1, 2 * shape["b"] + 1)


def special_dims(shape, state):
    if shape["kind"] == "plus":
        return 2 * shape["a"] + 1, 2 * shape["b"] + 1
    return state[2], state[3]


def special_contains(shape, state, px, py, selected=True):
    x, y, p, q = state
    w, h = special_dims(shape, state)
    dx, dy = px - x, py - y
    if selected and dx == w // 2 and dy == h // 2:
        return True
    if not (0 <= dx < w and 0 <= dy < h):
        return False
    if shape["kind"] == "plus":
        return dx == p or dy == q
    return dx == 0 or dx == w - 1 or dy == 0 or dy == h - 1


def lattice_positions(size, residue):
    # Positions are sprite top-left coordinates, with centers on-screen.
    lower = -(size // 2)
    upper = 63 - size // 2
    first = lower + ((residue - lower) % 3)
    return range(first, upper + 1, 3)


def special_candidates(shape, points, limit=100):
    start = special_start(shape)
    sx, sy, sp, sq = start
    candidates = []
    rx, ry = sx % 3, sy % 3

    if shape["kind"] == "plus":
        w = 2 * shape["a"] + 1
        h = 2 * shape["b"] + 1
        for ax in range(shape["a"] % 3, w, 3):
            for ay in range(shape["b"] % 3, h, 3):
                for x in lattice_positions(w, rx):
                    for y in lattice_positions(h, ry):
                        good = True
                        for gx, gy in points:
                            dx, dy = gx - x, gy - y
                            if not (
                                0 <= dx < w and 0 <= dy < h and
                                (dx == ax or dy == ay)
                            ):
                                good = False
                                break
                        if good:
                            score = (
                                (abs(x - sx) + abs(y - sy)) / 3.0 +
                                4.0 * (abs(ax - sp) + abs(ay - sq)) / 3.0
                            )
                            candidates.append((score, (x, y, ax, ay)))

    elif shape["kind"] == "rect":
        total = sp + sq
        for k in range(-10, 11):
            w = sp + 3 * k
            h = total - w
            if w < 4 or h < 4 or w > 60 or h > 60:
                continue
            for xm in {rx, 0}:
                for ym in {ry, 0}:
                    for x in lattice_positions(w, xm):
                        for y in lattice_positions(h, ym):
                            good = True
                            for gx, gy in points:
                                dx, dy = gx - x, gy - y
                                if not (
                                    0 <= dx < w and 0 <= dy < h and
                                    (dx == 0 or dx == w - 1 or
                                     dy == 0 or dy == h - 1)
                                ):
                                    good = False
                                    break
                            if good:
                                score = (
                                    (abs(x - sx) + abs(y - sy)) / 3.0 +
                                    4.0 * abs(w - sp) / 3.0
                                )
                                candidates.append((score, (x, y, w, h)))

    candidates.sort()
    out = []
    seen = set()
    for score, state in candidates:
        if state not in seen:
            seen.add(state)
            out.append((score, state))
            if len(out) >= limit:
                break
    return out


def special_assignment(shapes, goals):
    groups = {}
    for x, y, color in goals:
        groups.setdefault(color, []).append((x, y))
    colors = list(groups)
    if not colors or len(colors) > len(shapes):
        return None

    options = {}
    for i, shape in enumerate(shapes):
        if shape["kind"] not in ("plus", "rect"):
            continue
        for color in colors:
            options[(i, color)] = special_candidates(
                shape, groups[color], 100
            )

    best = None
    for assigned in itertools.permutations(range(len(shapes)), len(colors)):
        cost = 0.0
        valid = True
        for color, i in zip(colors, assigned):
            opts = options.get((i, color), [])
            if not opts:
                valid = False
                break
            cost += opts[0][0]
        if valid and (best is None or cost < best[0]):
            best = (cost, assigned)

    if best is None:
        return None
    return [
        {
            "index": i,
            "color": color,
            "targets": [list(state) for _, state in options[(i, color)]],
        }
        for color, i in zip(colors, best[1])
    ]


def special_move(shape, state, action, obstacle, box):
    x, y, p, q = state
    dx, dy = DIRS[action]
    nx, ny = x + dx, y + dy
    w, h = special_dims(shape, state)
    if not (0 <= nx + w // 2 < 64 and 0 <= ny + h // 2 < 64):
        return state

    moved = (nx, ny, p, q)
    collision = any(
        special_contains(shape, moved, ox, oy, True)
        for ox, oy in obstacle
    )
    if not collision:
        return moved

    xmin, ymin, xmax, ymax = box
    if shape["kind"] == "rect":
        if dx:
            if w <= 6:
                return state
            new_w, new_h = w - 3, h + 3
            new_y = ny + h // 2 - new_h // 2
            new_y = round(new_y / 3) * 3
            if dx < 0:
                nx -= dx
            return (nx, new_y, new_w, new_h)
        if h <= 6:
            return state
        new_h, new_w = h - 3, w + 3
        new_x = nx + w // 2 - new_w // 2
        new_x = round(new_x / 3) * 3
        if dy < 0:
            ny -= dy
        return (new_x, ny, new_w, new_h)

    in_x = xmin <= nx + p <= xmax
    in_y = ymin <= ny + q <= ymax
    if dx:
        if in_x and in_y:
            return state
        if in_x:
            can_shift = p > 0 if dx > 0 else p < w - 2
            if can_shift:
                return (nx, ny, p - (3 if dx > 0 else -3), q)
            return state
        if in_y:
            can_shift = p < w - 2 if dx > 0 else p > 0
            if can_shift:
                return (x, y, p + (3 if dx > 0 else -3), q)
            return state
    if dy:
        if in_x and in_y:
            return state
        if in_y:
            can_shift = q > 0 if dy > 0 else q < h - 2
            if can_shift:
                return (nx, ny, p, q - (3 if dy > 0 else -3))
            return state
        if in_x:
            can_shift = q < h - 2 if dy > 0 else q > 0
            if can_shift:
                return (x, y, p, q + (3 if dy > 0 else -3))
            return state
    return moved


def touches_wrong_station(shape, state, color, stations):
    x, y, _, _ = state
    w, h = special_dims(shape, state)
    for sx, sy, sw, sh, station_color in stations:
        if station_color == color:
            continue
        if x >= sx + sw or x + w <= sx or y >= sy + sh or y + h <= sy:
            continue
        for py in range(sy, sy + sh):
            for px in range(sx, sx + sw):
                if special_contains(shape, state, px, py, True):
                    return True
    return False


def special_search(shape, start, target_states, color,
                   stations, obstacle, max_nodes=40000):
    if not target_states or not obstacle:
        return []

    box = (
        min(x for x, _ in obstacle),
        min(y for _, y in obstacle),
        max(x for x, _ in obstacle),
        max(y for _, y in obstacle),
    )
    target_list = [tuple(s) for s in target_states]
    target_list.sort(key=lambda t: (
        abs(t[0] - start[0]) + abs(t[1] - start[1]) +
        4 * (abs(t[2] - start[2]) + abs(t[3] - start[3]))
    ))
    target_list = target_list[:60]
    target_set = set(target_list)

    def heuristic(s):
        x, y, p, q = s
        return min(
            (abs(x - tx) + abs(y - ty)) / 3.0 +
            2.0 * (abs(p - tp) + abs(q - tq)) / 3.0
            for tx, ty, tp, tq in target_list
        )

    heap = [(heuristic(start), 0, start)]
    dist = {start: 0}
    prev = {}
    count = 0

    while heap and count < max_nodes:
        _, g, state = heapq.heappop(heap)
        if g != dist.get(state):
            continue
        count += 1
        if state in target_set:
            result = []
            while state != start:
                old, action = prev[state]
                result.append((action, state))
                state = old
            result.reverse()
            return result

        for action in DIRS:
            new_state = special_move(shape, state, action, obstacle, box)
            if new_state == state:
                continue
            if touches_wrong_station(shape, new_state, color, stations):
                continue
            ng = g + 1
            if ng < dist.get(new_state, 10 ** 9):
                dist[new_state] = ng
                prev[new_state] = (state, action)
                heapq.heappush(
                    heap, (ng + heuristic(new_state), ng, new_state)
                )
    return []


def safe_paint_path(shape, wanted, stations, obstacle):
    off = offsets(shape["kind"], shape["a"], shape["b"])
    off.add((0, 0))
    x0, y0 = shape["x"], shape["y"]
    initial_color = shape["color"]
    positions = {}

    for y in range(y0 % 3, 64, 3):
        for x in range(x0 % 3, 64, 3):
            if any((x + dx, y + dy) in obstacle for dx, dy in off):
                positions[(x, y)] = None
                continue
            touched = set()
            for sx, sy, w, h, color in stations:
                if any(sx <= x + dx < sx + w and
                       sy <= y + dy < sy + h for dx, dy in off):
                    touched.add(color)
            positions[(x, y)] = touched

    start = (x0, y0)
    dist = {start: 0}
    prev = {}
    queue = deque([start])
    ends = []
    while queue:
        x, y = queue.popleft()
        for action, (dx, dy) in DIRS.items():
            nxt = (x + dx, y + dy)
            touched = positions.get(nxt)
            if touched is None:
                continue
            different = touched - {initial_color}
            if different and different != {wanted}:
                continue
            if nxt in dist:
                continue
            dist[nxt] = dist[(x, y)] + 1
            prev[nxt] = ((x, y), action)
            if wanted in different:
                ends.append(nxt)
            else:
                queue.append(nxt)

    if not ends:
        return []
    if obstacle:
        ox = sum(x for x, _ in obstacle) / len(obstacle)
        oy = sum(y for _, y in obstacle) / len(obstacle)
    else:
        ox, oy = x0, y0
    end = min(
        ends,
        key=lambda p: dist[p] + 0.4 * (abs(p[0] - ox) + abs(p[1] - oy)) / 3.0
    )
    result = []
    pos = end
    while pos != start:
        old, action = prev[pos]
        color = wanted if pos == end else initial_color
        result.append((action, (pos[0], pos[1], color)))
        pos = old
    result.reverse()
    return result


def next_selection(m, target):
    order = m["order"]
    selected = m["selected"]
    if selected == target:
        return None
    i = order.index(selected)
    m["selected"] = order[(i + 1) % len(order)]
    return ("ACTION5", None, None)


def special_action(obs, m):
    if "special" not in m:
        assignment = special_assignment(m["shapes"], m["goals"])
        if assignment is None:
            m["special"] = {"failed": True}
        else:
            m["special"] = {
                "assignment": assignment,
                "index": 0,
                "stage": "select",
                "actions": [],
            }

    sp = m["special"]
    if sp.get("failed"):
        return None

    shapes = m["shapes"]
    obstacle = {tuple(p) for p in m["obstacle"]}

    for _ in range(10):
        if sp["actions"]:
            action, i, state = sp["actions"].pop(0)
            if len(state) == 3:
                x, y, color = state
                shapes[i]["x"] = int(x)
                shapes[i]["y"] = int(y)
                shapes[i]["color"] = int(color)
            else:
                x, y, p, q = state
                shape = shapes[i]
                w, h = special_dims(shape, state)
                shape["x"] = int(x + w // 2)
                shape["y"] = int(y + h // 2)
                sp["current_state"] = list(state)
            return (action, None, None)

        if sp["index"] >= len(sp["assignment"]):
            target_points = {(x, y) for x, y, _ in m["goals"]}
            safe = [
                i for i, s in enumerate(shapes)
                if (s["x"], s["y"]) not in target_points
            ]
            if safe:
                action = next_selection(m, safe[0])
                if action is not None:
                    return action
            return None

        job = sp["assignment"][sp["index"]]
        i = job["index"]
        shape = shapes[i]
        color = job["color"]

        if sp["stage"] == "select":
            action = next_selection(m, i)
            if action is not None:
                return action
            sp["stage"] = "paint"

        if sp["stage"] == "paint":
            if shape["color"] != color:
                path = safe_paint_path(shape, color, m["stations"], obstacle)
                if not path:
                    sp["failed"] = True
                    return None
                sp["actions"] = [
                    [action, i, list(state)] for action, state in path
                ]
                sp["stage"] = "reshape"
                continue
            sp["stage"] = "reshape"

        if sp["stage"] == "reshape":
            start = special_start(shape)
            path = special_search(
                shape, start, job["targets"], color,
                m["stations"], obstacle
            )
            if not path and start not in {tuple(s) for s in job["targets"]}:
                sp["failed"] = True
                return None
            sp["actions"] = [
                [action, i, list(state)] for action, state in path
            ]
            sp["stage"] = "done"
            if sp["actions"]:
                continue

        if sp["stage"] == "done":
            sp["index"] += 1
            sp["stage"] = "select"

    return None


def retry(memory, available):
    if not memory.get("re86_retried") and "RESET" in available:
        memory["re86_retried"] = True
        memory.pop("re86", None)
        return ("RESET", None, None)
    return ("ACTION5", None, None)


def act(obs, available, memory):
    obs = np.asarray(obs)
    if "re86" not in memory:
        memory["re86"] = initialize(obs)

    m = memory["re86"]
    shapes = m["shapes"]
    if not shapes:
        return retry(memory, available)

    if m["phase"] == "discover":
        observed = selected_from_frame(obs, shapes, m["selected"])
        if m["discovery_steps"] > 0 and observed is not None:
            m["selected"] = observed
            if observed not in m["cycle"]:
                m["cycle"].append(observed)
            else:
                m["phase"] = "planning"
        if (m["phase"] == "discover" and
                m["discovery_steps"] < len(shapes) + 1):
            m["discovery_steps"] += 1
            return ("ACTION5", None, None)
        m["phase"] = "planning"

    if "order" not in m:
        cycle = list(m["cycle"])
        for i in range(len(shapes)):
            if i not in cycle:
                cycle.append(i)
        m["order"] = choose_rotation(obs, shapes, cycle)
        if m["selected"] is None:
            m["selected"] = m["order"][0]

    if m.get("use_special"):
        action = special_action(obs, m)
        return action if action is not None else retry(memory, available)

    if not m["plan"]:
        if m.get("plan_executed"):
            return retry(memory, available)
        obstacle = {tuple(p) for p in m["obstacle"]}
        m["plan"] = make_plan(
            shapes, m["goals"], m["stations"], obstacle,
            m["order"], m["selected"]
        )
        if not m["plan"]:
            if obstacle:
                m["use_special"] = True
                action = special_action(obs, m)
                return action if action is not None else retry(memory, available)
            return retry(memory, available)
        m["plan_executed"] = True

    action, selected, state = m["plan"].pop(0)
    m["selected"] = selected
    if state is not None:
        x, y, color = state
        shapes[selected]["x"] = int(x)
        shapes[selected]["y"] = int(y)
        shapes[selected]["color"] = int(color)

    if action not in available:
        return ("ACTION5", None, None)
    return (action, None, None)

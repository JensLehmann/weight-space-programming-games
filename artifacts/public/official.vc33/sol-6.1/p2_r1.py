import numpy as np
import heapq
import time
from collections import defaultdict, Counter

_BG = {0, 1, 3, 4, 5, 7, 9, 12}
_INF = 10**8


def _runs(v):
    result = []
    start = None
    for i, yes in enumerate(v):
        if yes and start is None:
            start = i
        elif not yes and start is not None:
            result.append((start, i))
            start = None
    if start is not None:
        result.append((start, len(v)))
    return result


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                if (0 <= nr < h and 0 <= nc < w and
                        mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        ys = [p[0] for p in cells]
        xs = [p[1] for p in cells]
        result.append((cells, (min(xs), min(ys),
                               max(xs) + 1, max(ys) + 1)))
    return result


def _pieces(a):
    h, w = a.shape
    marks = []
    labels = np.full((h, w), -1, dtype=np.int16)
    for color in range(16):
        if color in _BG:
            continue
        for cells, box in _components(a == color):
            idx = len(marks)
            marks.append((color, cells, box))
            for y, x in cells:
                labels[y, x] = idx

    players = []
    used = set()
    directions = []
    for cells, box in _components(a == 4):
        if len(cells) > 256:
            continue
        contacts = Counter()
        for y, x in cells:
            for ny, nx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    lab = int(labels[ny, nx])
                    if lab >= 0:
                        contacts[lab] += 1
        if not contacts:
            continue
        idx = contacts.most_common(1)[0][0]
        if idx in used:
            continue
        color, mcells, mb = marks[idx]
        used.add(idx)
        x0 = min(box[0], mb[0])
        y0 = min(box[1], mb[1])
        x1 = max(box[2], mb[2])
        y1 = max(box[3], mb[3])
        players.append((x0, y0, x1 - x0, y1 - y0, color))
        ym = sum(y for y, x in mcells) / len(mcells)
        xm = sum(x for y, x in mcells) / len(mcells)
        yf = sum(y for y, x in cells) / len(cells)
        xf = sum(x for y, x in cells) / len(cells)
        directions.append((xm - xf, ym - yf, len(cells)))

    goals = [(color, box) for i, (color, cells, box) in enumerate(marks)
             if i not in used]
    return players, goals, directions


def _clean(obs):
    a = np.asarray(obs, dtype=np.int16).copy()
    a[0] = a[1]
    return a


def _normalize(obs, scale, rotation):
    a = _clean(obs)
    if scale == 2:
        a = a[::2, ::2]
    return np.rot90(a, rotation).copy()


def _display_position(model, pos):
    r, c = pos
    n = model["n"]
    k = model["rotation"]
    if k == 1:
        r, c = c, n - 1 - r
    elif k == 2:
        r, c = n - 1 - r, n - 1 - c
    elif k == 3:
        r, c = n - 1 - c, r
    s = model["scale"]
    return int(r * s + s // 2), int(c * s + s // 2)


def _canonical_places(model, places):
    places = list(places)
    for indices in model["groups"]:
        values = sorted(places[i] for i in indices)
        for i, value in zip(indices, values):
            places[i] = value
    return tuple(places)


def _material_rows(block):
    """Buttons occlude material; they are not material themselves."""
    zeros = np.sum(block == 0, axis=1)
    known = block.shape[1] - np.sum(block == 9, axis=1)
    occupied = (zeros > 0) & (zeros >= 0.62 * np.maximum(known, 1))
    unknown = known == 0
    return occupied, unknown


def _observe(model, a, expected=None):
    tops = []
    for i, (x, right, bottom) in enumerate(model["geometry"]):
        if right <= x or not 0 < bottom <= len(a):
            return None
        occupied, unknown = _material_rows(a[:, x:right])
        if expected is not None:
            predicted_top = expected[0][i]
            occupied = occupied | (
                unknown & (np.arange(len(a)) >= predicted_top) &
                (np.arange(len(a)) < bottom)
            )
        t = bottom
        while t > 0 and occupied[t - 1]:
            t -= 1
        tops.append(t)

    players, _, _ = _pieces(a)
    buckets = defaultdict(list)
    for x, y, width, height, color in players:
        matches = [
            i for i, (fx, fr, fb) in enumerate(model["geometry"])
            if fx <= x < fr and y + height == tops[i]
        ]
        if not matches:
            return None
        buckets[(color, height, width)].append(matches[0])

    places = []
    for definition in model["players"]:
        if not buckets[definition]:
            return None
        places.append(buckets[definition].pop())
    if any(buckets.values()):
        return None
    return tuple(tops), _canonical_places(model, places)


def _parse(obs):
    full = _clean(obs)
    _, _, directions = _pieces(full)
    if not directions:
        return None
    dx, dy, _ = max(directions, key=lambda p: p[2])
    if abs(dx) > abs(dy):
        rotation = 3 if dx > 0 else 1
    else:
        rotation = 0 if dy > 0 else 2

    doubled = (
        np.array_equal(full[:, ::2], full[:, 1::2]) and
        np.array_equal(full[2::2], full[3::2])
    )
    scale = 2 if doubled else 1
    a = _normalize(obs, scale, rotation)
    n = len(a)
    players, goals, _ = _pieces(a)
    if not players:
        return None

    button_boxes = [box for cells, box in _components(a == 9)]
    button_sizes = [min(b[2] - b[0], b[3] - b[1])
                    for b in button_boxes]
    q = Counter(button_sizes).most_common(1)[0][0] if button_sizes else (
        2 if n <= 32 else 3
    )
    q = max(1, min(4, q))

    wall_mask = (a == 5) | (a == 1) | (a == 12)
    columns = np.sum(wall_mask, axis=0) >= max(5, q + 2)
    walls = []
    for x, right in _runs(columns):
        if right - x > max(4, q + 1):
            continue
        rows = np.any(wall_mask[:, x:right], axis=1)
        inds = np.flatnonzero(rows)
        if len(inds):
            walls.append((x, right, int(inds[0]), int(inds[-1]) + 1))
    if not walls:
        return None

    lanes = []
    left = 0
    for x, right, top, bottom in sorted(walls):
        if x > left:
            lanes.append((left, x))
        left = max(left, right)
    if left < n:
        lanes.append((left, n))

    terrain = (a == 0) | (a == 9)
    floors = []
    supports = []
    for left, right in lanes:
        if right <= left:
            continue
        rows = np.mean(terrain[:, left:right], axis=1) > 0.62
        for top, bottom in _runs(rows):
            block = terrain[top:bottom, left:right]
            cols = np.mean(block, axis=0) > 0.62
            spans = _runs(cols)
            if not spans:
                continue
            cx, cr = max(spans, key=lambda p: p[1] - p[0])
            x, r = left + cx, left + cr
            if r - x < 2 or not np.any(a[top:bottom, x:r] == 0):
                continue
            # Remove button-only rows that falsely extend the platform.
            material, unknown = _material_rows(a[:, x:r])
            actual_top = bottom
            while actual_top > top and (
                    material[actual_top - 1] or unknown[actual_top - 1]):
                actual_top -= 1
            if actual_top < bottom:
                floors.append((x, r, bottom, actual_top))

        rows5 = np.mean(a[:, left:right] == 5, axis=1) > 0.70
        for top, bottom in _runs(rows5):
            if bottom - top > 2 * q + 1:
                continue
            cols = np.mean(a[top:bottom, left:right] == 5, axis=0) > 0.70
            spans = _runs(cols)
            if spans:
                cx, cr = max(spans, key=lambda p: p[1] - p[0])
                if cr - cx >= 4:
                    supports.append((left + cx, left + cr, top, bottom))

    # Recover platforms that currently have zero length.
    for bx, by, br, bb in button_boxes:
        lane = next(((l, r) for l, r in lanes if l <= bx < r), None)
        if lane is None:
            continue
        l, r = lane
        candidates = [s[2] for s in supports
                      if s[0] <= bx < s[1] and s[2] >= bb]
        bottom = min(candidates) if candidates else n
        exists = any(x <= bx < fr and fb == bottom
                     for x, fr, fb, ft in floors)
        if not exists:
            same_lane = [f for f in floors if l <= f[0] < r]
            if same_lane:
                ref = max(same_lane, key=lambda f: f[1] - f[0])
                x, fr = ref[0], ref[1]
            else:
                x, fr = l, r
            floors.append((x, fr, bottom, bottom))

    floors = sorted(set(floors), key=lambda f: (f[0], f[2], f[1]))
    if not floors:
        return None
    geometry = [(f[0], f[1], f[2]) for f in floors]
    tops = tuple(f[3] for f in floors)
    adjacent = []
    for x, r, b, t in floors:
        adjacent.append([
            wi for wi, wall in enumerate(walls)
            if wall[1] == x or wall[0] == r
        ])

    positive = rotation in (0, 3)
    clearance = 6 if rotation == 1 and q == 3 else 4
    bounds = []
    player_bounds = []
    for i, (x, r, bottom, top) in enumerate(floors):
        ss = []
        for sx, sr, st, sb in supports:
            lateral_match = sr == r if rotation in (2, 3) else sx == x
            if lateral_match and sb < bottom:
                ss.append(sb)
        if ss:
            base = max(ss) if positive else min(ss)
            pb = base - clearance if positive else base + clearance
        else:
            base = max([walls[w][2] for w in adjacent[i]] or [0])
            pb = base
        bounds.append(base)
        player_bounds.append(pb)

    actions = []
    for bx, by, br, bb in button_boxes:
        for wi, (wx, wr, wt, wb) in enumerate(walls):
            if br == wx:
                own_side = "left"
            elif bx == wr:
                own_side = "right"
            else:
                continue
            lf = [i for i, (x, r, b, t) in enumerate(floors)
                  if r == wx and b >= by]
            rf = [i for i, (x, r, b, t) in enumerate(floors)
                  if x == wr and b >= by]
            if not lf or not rf:
                continue
            key = (lambda i: floors[i][3]) if positive else (
                lambda i: floors[i][2]
            )
            li, ri = min(lf, key=key), min(rf, key=key)
            shrink, grow = (ri, li) if own_side == "left" else (li, ri)
            actions.append(("button", shrink, grow,
                            ((by + bb) // 2, (bx + br) // 2)))
            break

    parent = list(range(len(floors)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for action in actions:
        i, j = action[1], action[2]
        parent[find(i)] = find(j)
    comps = defaultdict(list)
    for i in range(len(floors)):
        comps[find(i)].append(i)
    component = {}
    for indices in comps.values():
        for i in indices:
            component[i] = indices

    lo = []
    hi = []
    for i, (x, r, b, t) in enumerate(floors):
        if len(component[i]) == 1:
            lo.append(t)
            hi.append(t)
        else:
            lattice_bound = bounds[i] - ((bounds[i] - t) % q)
            lo.append(min(t, lattice_bound))
            hi.append(b - ((b - t) % q))

    def fixed_feasible(fixed):
        for indices in comps.values():
            known = {i: fixed[i] for i in indices if i in fixed}
            if not known:
                continue
            total = sum(tops[i] for i in indices) - sum(known.values())
            other = [i for i in indices if i not in known]
            if not (sum(lo[i] for i in other) <= total <=
                    sum(hi[i] for i in other)):
                return False
        return True

    gate_connections = [[] for _ in floors]
    gate_mask = (a == 1) | (a == 12)
    for cells, box in _components(gate_mask):
        gx, gy, gr, gb = box
        if gr - gx > max(4, q + 1) or gb - gy < 3:
            continue
        lf = [i for i, (x, r, b, t) in enumerate(floors) if r == gx]
        rf = [i for i, (x, r, b, t) in enumerate(floors) if x == gr]
        for li in lf:
            for ri in rf:
                if not (lo[li] <= gb <= hi[li] and
                        lo[ri] <= gb <= hi[ri]):
                    continue
                if (gb - tops[li]) % q or (gb - tops[ri]) % q:
                    continue
                if not fixed_feasible({li: gb, ri: gb}):
                    continue
                idx = len(actions)
                actions.append(("gate", li, ri,
                                ((gy + gb) // 2, (gx + gr) // 2), gb))
                gate_connections[li].append((ri, gb, idx))
                gate_connections[ri].append((li, gb, idx))

    player_defs = []
    places = []
    for x, y, width, height, color in sorted(
            players, key=lambda p: (p[4], p[3], p[2], p[0], p[1])):
        fs = [i for i, (fx, fr, b, t) in enumerate(floors)
              if fx <= x < fr and y + height == t]
        if not fs:
            return None
        player_defs.append((color, height, width))
        places.append(fs[0])

    groups_by_def = defaultdict(list)
    for i, definition in enumerate(player_defs):
        groups_by_def[definition].append(i)
    groups = list(groups_by_def.values())

    goal_specs = []
    for color, (gx, gy, gr, gb) in goals:
        wall_ids = [wi for wi, (wx, wr, wt, wb) in enumerate(walls)
                    if wx <= (gx + gr - 1) / 2 < wr]
        if not wall_ids:
            continue
        stripe_height = gb - gy
        target_top = gb - 3 * stripe_height
        acceptable = [i for i in range(len(floors))
                      if any(w in adjacent[i] for w in wall_ids)]
        goal_specs.append((color, target_top, acceptable))

    player_goals = []
    for color, height, width in player_defs:
        entries = []
        for gc, target_top, acceptable in goal_specs:
            if gc != color:
                continue
            surface = target_top + height
            for f in acceptable:
                if lo[f] <= surface <= hi[f] and (
                        surface - tops[f]) % q == 0:
                    entries.append((f, surface))
        player_goals.append(list(set(entries)))

    model = {
        "n": n, "scale": scale, "rotation": rotation, "q": q,
        "geometry": geometry, "bounds": bounds,
        "player_bounds": player_bounds, "players": player_defs,
        "groups": groups, "actions": actions, "connections": gate_connections,
        "component": component, "lo": lo, "hi": hi,
        "goals": player_goals,
    }
    state = (tops, _canonical_places(model, places))
    return model, state


def _distance_tables(model):
    q = model["q"]
    tables = []
    memo = {}
    for pi, definition in enumerate(model["players"]):
        key = (definition, tuple(sorted(model["goals"][pi])))
        if key in memo:
            tables.append(memo[key])
            continue
        dist = {}
        heap = []
        for f, t in model["goals"][pi]:
            dist[(f, t)] = 0
            heapq.heappush(heap, (0, f, t))
        while heap:
            d, f, t = heapq.heappop(heap)
            if dist.get((f, t)) != d:
                continue
            neighbors = []
            if len(model["component"][f]) > 1:
                if t - q >= model["lo"][f]:
                    neighbors.append((f, t - q))
                if (t + q <= model["hi"][f] and
                        t + q > model["player_bounds"][f]):
                    neighbors.append((f, t + q))
            for other, level, ai in model["connections"][f]:
                if t == level:
                    neighbors.append((other, t))
            for nf, nt in neighbors:
                nd = d + 1
                if nd < dist.get((nf, nt), _INF):
                    dist[(nf, nt)] = nd
                    heapq.heappush(heap, (nd, nf, nt))
        memo[key] = dist
        tables.append(dist)
    return tables


def _is_goal(model, state):
    tops, places = state
    return all((f, tops[f]) in model["goals"][i]
               for i, f in enumerate(places))


def _heuristic(model, tables, state):
    tops, places = state
    q = model["q"]
    result = 0
    for pi, f in enumerate(places):
        t = tops[f]
        base = tables[pi].get((f, t), _INF)
        if base >= _INF:
            return _INF
        if base == 0:
            continue

        bound = model["player_bounds"][f]
        lattice_bound = bound - ((bound - t) % q)
        reachable_low = min(t, lattice_bound)
        adjustable = len(model["component"][f]) > 1
        first = _INF
        for gf, gt in model["goals"][pi]:
            if gf == f and reachable_low <= gt <= model["hi"][f]:
                if adjustable or gt == t:
                    first = min(first, abs(t - gt) // q)
        for other, level, ai in model["connections"][f]:
            if level < reachable_low or (not adjustable and level != t):
                continue
            remaining = tables[pi].get((other, level), _INF)
            if remaining >= _INF:
                continue
            alignment = max(abs(t - level), abs(tops[other] - level)) // q
            first = min(first, alignment + 1 + remaining)
        value = max(base, first if first < _INF else base)
        result = max(result, value)
    return result


def _neighbors(model, state, blocked=None):
    tops, places = state
    occupied = set(places)
    q = model["q"]
    forbidden = blocked.get(state, ()) if blocked else ()
    for ai, action in enumerate(model["actions"]):
        if ai in forbidden:
            continue
        kind, i, j = action[:3]
        if kind == "button":
            bound = (model["player_bounds"][j] if j in occupied
                     else model["bounds"][j])
            if tops[i] >= model["geometry"][i][2] or tops[j] <= bound:
                continue
            tt = list(tops)
            tt[i] += q
            tt[j] -= q
            yield ai, (tuple(tt), places)
        else:
            level = action[4]
            if tops[i] != level or tops[j] != level:
                continue
            if i not in occupied and j not in occupied:
                continue
            pp = [j if f == i else i if f == j else f for f in places]
            pp = _canonical_places(model, pp)
            if pp != places:
                yield ai, (tops, pp)


def _search(model, start, visits, blocked):
    if _is_goal(model, start):
        return []
    tables = model.get("tables")
    if tables is None:
        tables = _distance_tables(model)
        model["tables"] = tables

    h0 = _heuristic(model, tables, start)
    scores = {start: 0}
    parents = {}
    serial = 0
    heap = [(1.55 * h0, h0, 0, serial, start)]
    best = start
    best_rank = (h0, 0)
    deadline = time.monotonic() + 6.5
    expanded = 0
    goal = None

    while heap and expanded < 110000:
        if expanded % 256 == 0 and time.monotonic() >= deadline:
            break
        _, h, g, _, state = heapq.heappop(heap)
        if scores.get(state) != g:
            continue
        expanded += 1
        if _is_goal(model, state):
            goal = state
            break
        rank = (h, g + 3 * visits.get(state, 0))
        if state != start and rank < best_rank:
            best, best_rank = state, rank

        for ai, nxt in _neighbors(model, state, blocked):
            ng = g + 1
            if ng >= scores.get(nxt, _INF):
                continue
            nh = _heuristic(model, tables, nxt)
            if nh >= _INF:
                continue
            scores[nxt] = ng
            parents[nxt] = (state, ai)
            serial += 1
            heapq.heappush(heap, (ng + 1.55 * nh, nh, ng, serial, nxt))

    target = goal if goal is not None else best
    if target == start:
        choices = []
        for ai, nxt in _neighbors(model, start, blocked):
            h = _heuristic(model, tables, nxt)
            choices.append((h + 4 * visits.get(nxt, 0), ai, nxt))
        if not choices:
            return []
        _, ai, nxt = min(choices, key=lambda p: (p[0], p[1]))
        return [(ai, nxt)]

    route = []
    while target != start:
        prev, ai = parents[target]
        route.append((ai, target))
        target = prev
    route.reverse()
    return route


def _fallback(obs, memory):
    targets = []
    for color in (9, 12, 1):
        for cells, (x, y, r, b) in _components(np.asarray(obs) == color):
            if len(cells) >= 2:
                targets.append(((y + b) // 2, (x + r) // 2))
    if targets:
        count = memory.get("_fallback_count", 0)
        memory["_fallback_count"] = count + 1
        row, col = targets[count % len(targets)]
        return "ACTION6", int(row), int(col)
    return "ACTION6", 0, 0


def act(obs, available, memory):
    if "ACTION6" not in available:
        name = "RESET" if "RESET" in available else available[0]
        return name, None, None

    cache = memory.get("_vc33")
    model = None
    state = None
    route = None
    blocked = memory.setdefault("_vc33_blocked", {})
    visits = memory.setdefault("_vc33_visits", {})

    if cache is not None:
        old_model = cache["model"]
        normalized = _normalize(
            obs, old_model["scale"], old_model["rotation"]
        )
        observed = _observe(old_model, normalized, cache.get("expected"))
        if observed is not None:
            model, state = old_model, observed
            if observed == cache.get("expected"):
                route = cache.get("route", [])
            elif observed == cache.get("previous"):
                ai = cache.get("action")
                if ai is not None:
                    blocked.setdefault(observed, set()).add(ai)

    if model is None:
        parsed = _parse(obs)
        if parsed is None:
            memory.pop("_vc33", None)
            return _fallback(obs, memory)
        model, state = parsed
        blocked = {}
        visits = {}
        memory["_vc33_blocked"] = blocked
        memory["_vc33_visits"] = visits

    if _is_goal(model, state):
        return "ACTION6", 0, 0

    visits[state] = visits.get(state, 0) + 1

    if route and route[0][0] in blocked.get(state, ()):
        route = None
    if not route:
        route = _search(model, state, visits, blocked)
    if not route:
        memory["_vc33"] = {
            "model": model, "expected": state, "previous": state,
            "action": None, "route": []
        }
        return _fallback(obs, memory)

    ai, expected = route[0]
    action = model["actions"][ai]
    row, col = _display_position(model, action[3])
    memory["_vc33"] = {
        "model": model,
        "previous": state,
        "action": ai,
        "expected": expected,
        "route": route[1:],
    }
    return "ACTION6", row, col

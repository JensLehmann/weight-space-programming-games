import heapq
import time
from collections import deque

import numpy as np


def act(obs, available, memory):
    deadline = time.monotonic() + 8.5
    a = np.asarray(obs)
    available = set(available)

    def fallback():
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if name in available:
                return name, None, None
        return next(iter(available)), None, None

    directions = [
        ("ACTION1", 0, -3),
        ("ACTION2", 0, 3),
        ("ACTION3", -3, 0),
        ("ACTION4", 3, 0),
    ]
    directions = [d for d in directions if d[0] in available]

    # Identify rectangular controllable sprites.
    candidates = []
    checked = set()
    for cy, cx in np.argwhere(np.isin(a[:63], (0, 4, 5))):
        cx, cy = int(cx), int(cy)
        for w, h in ((3, 3), (3, 6), (6, 3), (6, 6)):
            ox = 1 if w == 3 else 2
            oy = 1 if h == 3 else 2
            x, y = cx - ox, cy - oy
            key = (x, y, w, h)
            if key in checked:
                continue
            checked.add(key)
            if x < 0 or y < 0 or x + w > 64 or y + h > 63:
                continue
            patch = a[y:y + h, x:x + w]
            cw = 1 if w == 3 else 2
            ch = 1 if h == 3 else 2
            center = patch[oy:oy + ch, ox:ox + cw]
            color = int(center[0, 0])
            if color not in (0, 4, 5) or not np.all(center == color):
                continue
            good = True
            n14 = 0
            for yy in range(h):
                for xx in range(w):
                    if oy <= yy < oy + ch and ox <= xx < ox + cw:
                        continue
                    v = int(patch[yy, xx])
                    if v == 14:
                        n14 += 1
                    elif not (
                        v == 0 and
                        (xx == 0 or yy == 0 or xx == w - 1 or yy == h - 1)
                    ):
                        good = False
                        break
                if not good:
                    break
            if good and n14 >= 3:
                candidates.append(
                    ("r%dx%d" % (w, h), x, y, w, h, color == 0, n14)
                )

    candidates.sort(key=lambda p: (-p[3] * p[4], -p[6], p[2], p[1]))
    pieces = []
    used = set()
    for kind, x, y, w, h, selected, _ in candidates:
        cells = {(x + dx, y + dy) for dy in range(h) for dx in range(w)}
        if cells & used:
            continue
        used.update(cells)
        pieces.append((kind, x, y, w, h, selected, 0, 0))

    # Identify plus sprites.
    seen = set()
    for yy in range(63):
        for xx in range(64):
            if int(a[yy, xx]) != 11 or (xx, yy) in seen:
                continue
            todo = [(xx, yy)]
            seen.add((xx, yy))
            component = []
            while todo:
                x, y = todo.pop()
                component.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y),
                               (x, y - 1), (x, y + 1)):
                    if (0 <= nx < 64 and 0 <= ny < 63 and
                            (nx, ny) not in seen and int(a[ny, nx]) == 11):
                        seen.add((nx, ny))
                        todo.append((nx, ny))
            x0 = min(x for x, y in component)
            x1 = max(x for x, y in component)
            y0 = min(y for x, y in component)
            y1 = max(y for x, y in component)
            if x1 - x0 == 8 and y1 - y0 == 8 and len(component) >= 40:
                pieces.append(("plus", x0, y0, 9, 9, False, 0, 0))

    # Identify bombs and infer their blast direction from the fuse.
    previous_bombs = memory.get("bomb_info", [])
    bomb_info = []
    bomb_used = set()
    for y in range(58):
        for x in range(59):
            if int(a[y, x]) not in (12, 13):
                continue
            patch = a[y:y + 6, x:x + 6]
            if not np.all((patch == 12) | (patch == 13)):
                continue
            cells = {(x + dx, y + dy) for dy in range(6) for dx in range(6)}
            if cells & bomb_used:
                continue
            bomb_used.update(cells)
            red = patch == 12
            count = int(np.sum(red))
            tick = min(5, count // 6)
            rotation = None
            if count:
                rows = [j for j in range(6) if np.all(red[j, :])]
                cols = [j for j in range(6) if np.all(red[:, j])]
                if rows:
                    rotation = 0 if sum(rows) / len(rows) < 2.5 else 180
                elif cols:
                    rotation = 270 if sum(cols) / len(cols) < 2.5 else 90
            if rotation is None and previous_bombs:
                nearest = min(
                    previous_bombs,
                    key=lambda b: abs(b[0] - x) + abs(b[1] - y)
                )
                if abs(nearest[0] - x) + abs(nearest[1] - y) <= 18:
                    rotation = nearest[2]
            if rotation is None:
                rotation = 180 if y >= 33 else 0
            bomb_info.append((x, y, rotation))
            pieces.append(("bomb", x, y, 6, 6, False, tick, rotation))
    memory["bomb_info"] = bomb_info

    pieces.sort(key=lambda p: (p[0], p[2], p[1]))
    kinds = [p[0] for p in pieces]
    widths = [p[3] for p in pieces]
    heights = [p[4] for p in pieces]
    players = [i for i, p in enumerate(pieces) if p[0].startswith("r")]
    bombs = [i for i, p in enumerate(pieces) if p[0] == "bomb"]
    pluses = [i for i, p in enumerate(pieces) if p[0] == "plus"]
    if not players:
        memory.pop("plan", None)
        return fallback()

    selected_list = [i for i in players if pieces[i][5]]
    selected = selected_list[0] if selected_list else players[0]
    pos0 = tuple(p[2] * 64 + p[1] for p in pieces)
    ticks0 = tuple(pieces[i][6] for i in bombs)
    rotations = [pieces[i][7] for i in bombs]

    bases = []
    for kind, w, h in zip(kinds, widths, heights):
        mask = 0
        for dy in range(h):
            for dx in range(w):
                if kind != "plus" or 3 <= dx < 6 or 3 <= dy < 6:
                    mask |= 1 << (dy * 64 + dx)
        bases.append(mask)

    mask_cache = {}

    def mask_at(i, p):
        key = (i, p)
        result = mask_cache.get(key)
        if result is None:
            result = bases[i] << p
            mask_cache[key] = result
        return result

    def signature(pos, sel, ticks):
        bomb_tick = {i: ticks[j] for j, i in enumerate(bombs)}
        objects = tuple(sorted(
            (kinds[i], pos[i], bomb_tick.get(i, -1))
            for i in range(len(pos))
        ))
        return objects, kinds[sel], pos[sel]

    sig = signature(pos0, selected, ticks0)
    expected = memory.pop("expected", None)
    if expected is not None and expected != sig:
        memory.pop("plan", None)

    visits = memory.get("visits", [])
    if not visits or visits[-1] != sig:
        visits.append(sig)
        visits = visits[-120:]
    memory["visits"] = visits
    visited = set(visits)

    wall2 = int(memory.get("wall2", 0))
    wall15 = int(memory.get("wall15", 0))
    for y in range(63):
        for x in range(64):
            v = int(a[y, x])
            if v == 2:
                wall2 |= 1 << (y * 64 + x)
            elif v == 15:
                wall15 |= 1 << (y * 64 + x)
    wall2 |= ((1 << 64) - 1) << (63 * 64)
    memory["wall2"], memory["wall15"] = wall2, wall15

    plan = memory.get("plan")
    if plan:
        action, next_sig = plan.pop(0)
        memory["plan"] = plan
        memory["expected"] = next_sig
        return action

    dynamic = 0
    for i, p in enumerate(pos0):
        dynamic |= mask_at(i, p)

    # Record targets before moving pieces over their outlines.
    if "goals" not in memory:
        def covered(x, y):
            return bool(dynamic & (1 << (y * 64 + x)))

        def is_outline(x, y, outline):
            visible = 0
            for dx, dy in outline:
                xx, yy = x + dx, y + dy
                if int(a[yy, xx]) == 4 and not covered(xx, yy):
                    visible += 1
                elif not covered(xx, yy):
                    return False
            return visible >= max(5, int(len(outline) * 0.6))

        found = set()
        for w, h in ((5, 5), (5, 8), (8, 5), (8, 8)):
            wanted = "r%dx%d" % (w - 2, h - 2)
            compatible = [
                (pos0[i] % 64 % 3, pos0[i] // 64 % 3)
                for i in players if kinds[i] == wanted
            ]
            if not compatible:
                continue
            outline = [
                (x, y) for y in range(h) for x in range(w)
                if x == 0 or y == 0 or x == w - 1 or y == h - 1
            ]
            for y in range(64 - h):
                for x in range(65 - w):
                    if ((x + 1) % 3, (y + 1) % 3) not in compatible:
                        continue
                    if is_outline(x, y, outline):
                        found.add((x + 1, y + 1, w - 2, h - 2, "rect"))

        inside = {
            (x + 1, y + 1)
            for y in range(9) for x in range(9)
            if 3 <= x < 6 or 3 <= y < 6
        }
        outline = set()
        for x, y in inside:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q not in inside and 0 <= q[0] < 11 and 0 <= q[1] < 11:
                        outline.add(q)
        if pluses:
            for y in range(53):
                for x in range(54):
                    if not any(
                        (x + 1 - pos0[i] % 64) % 3 == 0 and
                        (y + 1 - pos0[i] // 64) % 3 == 0
                        for i in pluses
                    ):
                        continue
                    if is_outline(x, y, outline):
                        found.add((x + 1, y + 1, 9, 9, "plus"))
        memory["goals"] = tuple(sorted(found))

    goals = memory.get("goals", ())
    if not goals:
        return fallback()

    def wanted_kind(goal):
        if goal[4] == "plus":
            return "plus"
        return "r%dx%d" % (goal[2], goal[3])

    def goal_done(goal, pos):
        target = goal[1] * 64 + goal[0]
        kind = wanted_kind(goal)
        return any(kinds[i] == kind and pos[i] == target for i in range(len(pos)))

    solved_initial = [g for g in goals if goal_done(g, pos0)]
    unfinished = [g for g in goals if not goal_done(g, pos0)]
    if not unfinished:
        return fallback()

    def shifted(p, i, dx, dy):
        x, y = p % 64, p // 64
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx + widths[i] > 64 or ny + heights[i] > 63:
            return None
        return ny * 64 + nx

    def push_one(pos, i, dx, dy, stack):
        if i in stack:
            return False
        newp = shifted(pos[i], i, dx, dy)
        if newp is None:
            return False
        newmask = mask_at(i, newp)
        if newmask & wall2:
            return False
        oldp = pos[i]
        pos[i] = newp
        stack.add(i)
        for j in range(len(pos)):
            if j != i and newmask & mask_at(j, pos[j]):
                if not push_one(pos, j, dx, dy, stack):
                    pos[i] = oldp
                    stack.remove(i)
                    return False
        stack.remove(i)
        return True

    def rectangle_mask(x, y, w, h):
        result = 0
        left, right = max(0, x), min(64, x + w)
        if right <= left:
            return 0
        row = (1 << (right - left)) - 1
        for yy in range(max(0, y), min(63, y + h)):
            result |= row << (yy * 64 + left)
        return result

    def advance_bombs(pos, ticks):
        if not bombs:
            return tuple(pos), ticks
        ticks = list(ticks)
        exploding = []
        for j, i in enumerate(bombs):
            ticks[j] += 1
            if ticks[j] >= 6:
                ticks[j] = 0
                exploding.append((i, rotations[j]))
        if not exploding:
            return tuple(pos), tuple(ticks)

        pushed_directions = {}
        order = pluses + bombs + players
        for stage in (1, 2, 3):
            for bi, rot in exploding:
                bx, by = pos[bi] % 64, pos[bi] // 64
                length = 3 * stage
                if rot == 0:
                    blast = rectangle_mask(bx, by + 6, 6, length)
                    dx, dy = 0, 3
                elif rot == 90:
                    blast = rectangle_mask(bx - length, by, length, 6)
                    dx, dy = -3, 0
                elif rot == 180:
                    blast = rectangle_mask(bx, by - length, 6, length)
                    dx, dy = 0, -3
                else:
                    blast = rectangle_mask(bx + 6, by, length, 6)
                    dx, dy = 3, 0
                for i in order:
                    if blast & mask_at(i, pos[i]):
                        push_one(pos, i, dx, dy, set())
                        pushed_directions[i] = (dx, dy)

        for _ in range(30):
            changed = False
            for i, (dx, dy) in pushed_directions.items():
                if mask_at(i, pos[i]) & wall15:
                    before = tuple(pos)
                    push_one(pos, i, dx, dy, set())
                    if tuple(pos) != before:
                        changed = True
            if not changed:
                break
        return tuple(pos), tuple(ticks)

    def move_state(pos_tuple, sel, ticks, dx, dy):
        pos = list(pos_tuple)
        p = shifted(pos[sel], sel, dx, dy)
        if p is not None:
            m = mask_at(sel, p)
            if not (m & (wall2 | wall15)):
                hits = [
                    i for i in range(len(pos))
                    if i != sel and m & mask_at(i, pos[i])
                ]
                if not hits:
                    pos[sel] = p
                else:
                    for i in hits:
                        count = 0
                        while count < 5 or (mask_at(i, pos[i]) & wall15):
                            before = tuple(pos)
                            if not push_one(pos, i, dx, dy, set()):
                                break
                            if tuple(pos) == before:
                                break
                            count += 1
                            if count >= 32:
                                break
        return advance_bombs(pos, ticks)

    graph_cache = {}

    def graph_for(i):
        key = (kinds[i], pos0[i] % 64 % 3, pos0[i] // 64 % 3)
        if key in graph_cache:
            return graph_cache[key]
        xr, yr = key[1], key[2]
        nodes = []
        for y in range(yr, 63 - heights[i] + 1, 3):
            for x in range(xr, 64 - widths[i] + 1, 3):
                p = y * 64 + x
                if not (mask_at(i, p) & wall2):
                    nodes.append(p)
        node_set = set(nodes)
        forward = {p: [] for p in nodes}
        reverse = {p: [] for p in nodes}
        walk_reverse = {p: [] for p in nodes}
        for p in nodes:
            for _, dx, dy in directions:
                q = shifted(p, i, dx, dy)
                if q is None or q not in node_set:
                    continue
                if kinds[i].startswith("r") and not (mask_at(i, q) & wall15):
                    forward[p].append((q, 1, "walk", dx, dy))
                    reverse[q].append((p, 1))
                    walk_reverse[q].append(p)
                cur = p
                count = 0
                while count < 5 or (mask_at(i, cur) & wall15):
                    nxt = shifted(cur, i, dx, dy)
                    if nxt is None or nxt not in node_set:
                        break
                    cur = nxt
                    count += 1
                    if count >= 32:
                        break
                if cur != p:
                    forward[p].append((cur, 2, "push", dx, dy))
                    reverse[cur].append((p, 2))
        graph_cache[key] = forward, reverse, walk_reverse
        return graph_cache[key]

    def reverse_dist(reverse, destinations):
        dist = {}
        heap = []
        for p in destinations:
            if p in reverse and p not in dist:
                dist[p] = 0
                heapq.heappush(heap, (0, p))
        while heap:
            d, p = heapq.heappop(heap)
            if dist[p] != d:
                continue
            for q, cost in reverse[p]:
                nd = d + cost
                if nd < dist.get(q, 10 ** 9):
                    dist[q] = nd
                    heapq.heappush(heap, (nd, q))
        return dist

    stance_cache = {}
    approach_cache = {}

    def stances(target, tp, helper, dx, dy):
        key = (target, tp, helper, dx, dy)
        if key in stance_cache:
            return stance_cache[key]
        tx, ty = tp % 64, tp // 64
        tw, th = widths[target], heights[target]
        hw, hh = widths[helper], heights[helper]
        xr, yr = pos0[helper] % 64 % 3, pos0[helper] // 64 % 3
        if dx > 0:
            xs = range(tx - hw - 2, tx - hw + 1)
            ys = range(ty - hh + 1, ty + th)
        elif dx < 0:
            xs = range(tx + tw, tx + tw + 3)
            ys = range(ty - hh + 1, ty + th)
        elif dy > 0:
            xs = range(tx - hw + 1, tx + tw)
            ys = range(ty - hh - 2, ty - hh + 1)
        else:
            xs = range(tx - hw + 1, tx + tw)
            ys = range(ty + th, ty + th + 3)
        result = []
        tm = mask_at(target, tp)
        for y in ys:
            if y < 0 or y + hh > 63 or y % 3 != yr:
                continue
            for x in xs:
                if x < 0 or x + hw > 64 or x % 3 != xr:
                    continue
                p = y * 64 + x
                if mask_at(helper, p) & (wall2 | wall15 | tm):
                    continue
                q = shifted(p, helper, dx, dy)
                if q is not None and mask_at(helper, q) & tm:
                    result.append(p)
        stance_cache[key] = tuple(result)
        return stance_cache[key]

    def helper_distance(target, tp, helper, hp, dx, dy):
        key = (target, tp, helper, dx, dy)
        if key not in approach_cache:
            destinations = stances(target, tp, helper, dx, dy)
            _, reverse, _ = graph_for(helper)
            approach_cache[key] = reverse_dist(reverse, destinations)
        return approach_cache[key].get(hp, 60)

    initial = (pos0, selected, ticks0)
    best_partial = None

    def ordinary_search(target, destination, end_time, goal=None):
        nonlocal best_partial
        forward, reverse, _ = graph_for(target)
        distances = reverse_dist(reverse, [destination])
        if pos0[target] not in distances:
            return None

        good_pushes = {}
        can_walk = {}
        for p, edges in forward.items():
            d = distances.get(p, 10 ** 9)
            pushes = []
            walk = False
            for q, cost, typ, dx, dy in edges:
                if distances.get(q, 10 ** 9) + cost <= d + 1:
                    if typ == "walk":
                        walk = True
                    else:
                        pushes.append((dx, dy))
            good_pushes[p] = pushes
            can_walk[p] = walk

        hcache = {}

        def heuristic(state):
            pos, sel, ticks = state
            key = (pos, sel)
            if key in hcache:
                return hcache[key]
            tp = pos[target]
            d = distances.get(tp, 80)
            extra = 0.0
            if not kinds[target].startswith("r") or not can_walk.get(tp, False):
                best = 60
                for dx, dy in good_pushes.get(tp, ()):
                    for helper in players:
                        if helper == target:
                            continue
                        hd = helper_distance(target, tp, helper, pos[helper], dx, dy)
                        if helper != sel:
                            hd += 1
                        best = min(best, hd)
                extra = 0.8 * best
            elif sel != target:
                extra = 1.0

            penalty = 0.0
            for g in solved_initial:
                if not goal_done(g, pos):
                    penalty += 6.0
            for g in unfinished:
                gk = wanted_kind(g)
                gx, gy, gw, gh, _ = g
                gm = rectangle_mask(gx, gy, gw, gh)
                for i in players:
                    if kinds[i] != gk and mask_at(i, pos[i]) & gm:
                        penalty += 1.5
            value = d + extra + penalty
            hcache[key] = value
            return value

        h0 = heuristic(initial)
        heap = [(2.0 * h0, 0, 0, initial)]
        costs = {initial: 0}
        parent = {}
        best_h = h0
        best_state = None
        serial = 0
        expanded = 0

        while heap and expanded < 35000:
            if expanded % 64 == 0 and time.monotonic() >= end_time:
                break
            _, g, _, state = heapq.heappop(heap)
            if costs.get(state) != g:
                continue
            pos, sel, ticks = state
            if pos[target] == destination:
                return state, parent
            expanded += 1
            successors = []
            for name, dx, dy in directions:
                np_, nt = move_state(pos, sel, ticks, dx, dy)
                if np_ != pos or nt != ticks:
                    successors.append(((np_, sel, nt), (name, None, None)))
            if "ACTION6" in available:
                for i in players:
                    if i != sel:
                        x, y = pos[i] % 64, pos[i] // 64
                        successors.append(((pos, i, ticks), ("ACTION6", y + 1, x + 1)))

            for nxt, action in successors:
                ng = g + 1
                if ng >= costs.get(nxt, 10 ** 9):
                    continue
                costs[nxt] = ng
                parent[nxt] = (state, action)
                h = heuristic(nxt)
                if h < best_h - 0.05:
                    nsig = signature(nxt[0], nxt[1], nxt[2])
                    if nsig not in visited:
                        best_h, best_state = h, nxt
                serial += 1
                repeat_penalty = 1.0 if signature(nxt[0], nxt[1], nxt[2]) in visited else 0.0
                heapq.heappush(heap, (ng + 2.0 * h + repeat_penalty, ng, serial, nxt))

        if best_state is not None:
            improvement = h0 - best_h
            if best_partial is None or improvement > best_partial[0]:
                best_partial = (improvement, best_state, parent, goal)
        return None

    def walk_region_to_plus(helper, target):
        destinations = []
        for _, dx, dy in directions:
            destinations.extend(stances(target, pos0[target], helper, dx, dy))
        _, _, walk_reverse = graph_for(helper)
        region = set(p for p in destinations if p in walk_reverse)
        queue = deque(region)
        while queue:
            p = queue.popleft()
            for q in walk_reverse[p]:
                if q not in region:
                    region.add(q)
                    queue.append(q)
        return region

    def launch_search(helper, bi, region, end_time):
        bomb_j = bombs.index(bi)

        def heuristic(state):
            pos, sel, ticks = state
            hp = pos[helper]
            if hp in region:
                return 0.0
            hx, hy = hp % 64, hp // 64
            bx, by = pos[bi] % 64, pos[bi] // 64
            best = 100.0
            for x in (bx, bx + 3):
                for y in (by - 9, by - 6, by - 3):
                    d = (abs(hx - x) + abs(hy - y)) / 3.0
                    best = min(best, d)
            if best < 0.5:
                best += (6 - ticks[bomb_j]) * 0.3
            return best + (1 if sel != helper else 0)

        h0 = heuristic(initial)
        heap = [(2 * h0, 0, 0, initial)]
        costs = {initial: 0}
        parent = {}
        serial = 0
        best_h = h0
        best_state = None
        expanded = 0

        while heap and expanded < 25000:
            if expanded % 64 == 0 and time.monotonic() >= end_time:
                break
            _, g, _, state = heapq.heappop(heap)
            if costs.get(state) != g:
                continue
            pos, sel, ticks = state
            if pos[helper] in region:
                return state, parent
            expanded += 1
            for name, dx, dy in directions:
                np_, nt = move_state(pos, sel, ticks, dx, dy)
                nxt = (np_, sel, nt)
                if nxt == state:
                    continue
                ng = g + 1
                if ng >= costs.get(nxt, 10 ** 9):
                    continue
                costs[nxt] = ng
                parent[nxt] = (state, (name, None, None))
                h = heuristic(nxt)
                if h < best_h - 0.05 and signature(np_, sel, nt) not in visited:
                    best_h, best_state = h, nxt
                serial += 1
                heapq.heappush(heap, (ng + 2 * h, ng, serial, nxt))

        if best_state is not None:
            return best_state, parent
        return None

    result = None
    chosen_goal = None

    # If a lone controller is separated from the plus by ice, prepare
    # an upward-facing bomb underneath the plus's side of the barrier.
    if len(players) == 1 and pluses and bombs:
        helper, target = players[0], pluses[0]
        region = walk_region_to_plus(helper, target)
        if region and pos0[helper] not in region:
            tx, ty = pos0[target] % 64, pos0[target] // 64
            usable = []
            for j, bi in enumerate(bombs):
                bx, by = pos0[bi] % 64, pos0[bi] // 64
                if rotations[j] == 180 and by > ty + 9:
                    usable.append((by - ty, bi))
            if usable:
                _, bi = min(usable)
                bx, by = pos0[bi] % 64, pos0[bi] // 64
                desired_x = tx + 6
                desired_x += (bx - desired_x) % 3
                desired_x = max(bx % 3, min(58, desired_x))
                destination = by * 64 + desired_x
                if bx != desired_x:
                    result = ordinary_search(
                        bi, destination, min(deadline, time.monotonic() + 5.5)
                    )
                    if result is None and best_partial is not None:
                        result = best_partial[1], best_partial[2]
                else:
                    result = launch_search(
                        helper, bi, region, min(deadline, time.monotonic() + 5.5)
                    )

    if result is None:
        best_partial = None
        tasks = []
        for goal in unfinished:
            wanted = wanted_kind(goal)
            destination = goal[1] * 64 + goal[0]
            for i, kind in enumerate(kinds):
                if kind != wanted:
                    continue
                if ((pos0[i] % 64 - goal[0]) % 3 or
                        (pos0[i] // 64 - goal[1]) % 3):
                    continue
                _, reverse, _ = graph_for(i)
                distance = reverse_dist(reverse, [destination]).get(pos0[i])
                if distance is not None:
                    priority = 0 if goal[4] == "plus" else 1
                    tasks.append((priority, distance, goal, i))

        tasks.sort()
        locked = memory.get("task")
        if locked is not None:
            locked = tuple(locked)
            if locked in unfinished:
                tasks.sort(key=lambda t: (t[2] != locked, t[0], t[1]))
            else:
                memory.pop("task", None)

        for n, (_, _, goal, target) in enumerate(tasks[:4]):
            now = time.monotonic()
            if now >= deadline:
                break
            allowance = 5.0 if n == 0 else 1.0
            destination = goal[1] * 64 + goal[0]
            result = ordinary_search(
                target, destination, min(deadline, now + allowance), goal
            )
            if result is not None:
                chosen_goal = goal
                break

        if result is None and best_partial is not None:
            _, state, parent, chosen_goal = best_partial
            result = state, parent

    if result is None:
        memory.pop("task", None)
        return fallback()

    if chosen_goal is not None:
        memory["task"] = chosen_goal
    state, parent = result
    path = []
    while state != initial:
        entry = parent.get(state)
        if entry is None:
            break
        previous, action = entry
        path.append((action, signature(state[0], state[1], state[2])))
        state = previous
    path.reverse()

    if not path:
        return fallback()
    action, next_sig = path.pop(0)
    memory["plan"] = path
    memory["expected"] = next_sig
    return action

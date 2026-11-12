import time
import math
import heapq
from collections import Counter
import numpy as np


def _runs(bits):
    out = []
    start = None
    for i, value in enumerate(list(bits) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for yy, xx in zip(*np.nonzero(mask)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        stack = [(yy, xx)]
        t = b = yy
        l = r = xx
        while stack:
            y, x = stack.pop()
            t, b = min(t, y), max(b, y)
            l, r = min(l, x), max(r, x)
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= ny < h and 0 <= nx < w
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((t, l, b+1, r+1))
    return out


def _players(im):
    out = []
    for t, l, b, r in _components(im == 4):
        width = r-l
        if not (2 <= width <= 12 and 1 <= b-t <= 8 and b < 64):
            continue
        if np.count_nonzero(im[t, l:r] == 4) >= np.count_nonzero(im[b-1, l:r] == 4):
            continue
        for color in (11, 14):
            if np.mean(im[b, l:r] == color) < 0.9:
                continue
            end = b
            while end < 64 and end-b < 6:
                if np.mean(im[end, l:r] == color) < 0.9:
                    break
                end += 1
            out.append((color, end-t, l, r, t, b, end))
            break
    return out


def _canonical(obs, rotation=None):
    im = np.array(obs, dtype=np.int16, copy=True)
    im[0] = im[1]
    if rotation is None:
        scores = []
        for k in range(4):
            ps = _players(np.rot90(im, k))
            scores.append(sum(h*(r-l) for color, h, l, r, t, b, end in ps))
        rotation = max(range(4), key=lambda k: scores[k])
    return np.rot90(im, rotation).copy(), rotation


def _unrotate(k, row, col):
    row, col = int(row), int(col)
    if k == 0:
        return row, col
    if k == 1:
        return col, 63-row
    if k == 2:
        return 63-row, 63-col
    return 63-col, row


def _black_intervals(im, lane):
    l, r = lane
    black = np.count_nonzero(im[:, l:r] == 0, axis=1)
    filled = np.count_nonzero(np.isin(im[:, l:r], [0, 9]), axis=1)
    # Buttons can conceal part of a platform, but cannot create one alone.
    solid = (black > 0) & (filled >= 0.60*(r-l))
    return _runs(solid)


def _normalize(model, player_positions):
    positions = list(player_positions)
    for indices in model["groups"]:
        values = sorted(positions[i] for i in indices)
        for i, value in zip(indices, values):
            positions[i] = value
    return tuple(positions)


def _limit(model, platform_id, occupied):
    p = model["platforms"][platform_id]
    floors = p["floors"]
    if not floors:
        return p["wall_limit"]
    if model["negative"]:
        return min(floors) + (model["offset"] if occupied else 0)
    return max(floors) - (model["offset"] if occupied else 0)


def _parse(im, rotation):
    players = _players(im)
    if not players:
        return None

    buttons = [v for v in _components(im == 9)
               if v[2]-v[0] == v[3]-v[1] and 1 <= v[3]-v[1] <= 8]
    widths = [r-l for t, l, b, r in buttons]
    step = Counter(widths).most_common(1)[0][0] if widths else 3

    structural = np.isin(im, [5, 1, 12, 11, 14])
    for color, h, l, r, t, base, end in players:
        structural[base:end, l:r] = False

    spans = _runs(structural.sum(axis=0) >= max(6, 2*step))
    walls = []
    for l, r in spans:
        if not (1 <= r-l <= max(8, 2*step)):
            continue
        rows = np.flatnonzero(np.any(structural[:, l:r], axis=1))
        if len(rows):
            walls.append((l, r, int(rows[0]), int(rows[-1])+1))
    walls.sort()
    if not walls:
        return None

    lanes = []
    edge = 0
    for l, r, t, b in walls:
        if edge < l:
            lanes.append((edge, l))
        edge = r
    if edge < 64:
        lanes.append((edge, 64))

    lane_floors = []
    intervals = []
    for lane in lanes:
        l, r = lane
        gray = np.mean(im[:, l:r] == 5, axis=1) >= 0.90
        lane_floors.append(_runs(gray))
        intervals.append(_black_intervals(im, lane))

    platforms = []
    tops = []
    for lane_id, lane in enumerate(lanes):
        anchors = {b for a, b in intervals[lane_id]}
        # Recover platforms of zero height at the board edge or against a floor.
        for t, l, b, r in buttons:
            if lane[0] <= l and r <= lane[1]:
                if b == 64 or any(a == b for a, end in lane_floors[lane_id]):
                    anchors.add(b)

        adjacent = [t for l, r, t, b in walls
                    if r == lane[0] or l == lane[1]]
        wall_limit = max(adjacent) if adjacent else 0
        for anchor in sorted(anchors):
            top = next((a for a, b in intervals[lane_id] if b == anchor), anchor)
            platforms.append({
                "lane": lane_id, "l": lane[0], "r": lane[1],
                "anchor": anchor, "wall_limit": wall_limit,
                "floors": [b for a, b in lane_floors[lane_id] if b < anchor]
            })
            tops.append(top)

    if not platforms:
        return None

    negative = rotation in (1, 2)
    scale = 2 if step == 4 else 1
    model = {
        "rotation": rotation, "step": step,
        "negative": negative,
        "offset": 6 if rotation == 1 and step == 3 else 4*scale,
        "platforms": platforms, "lanes": lanes,
        "n": len(platforms), "overrides": {},
    }

    def support_for(player):
        color, h, l, r, t, base, end = player
        x = (l+r)/2
        candidates = [i for i, p in enumerate(platforms)
                      if p["l"] <= x < p["r"] and tops[i] == end]
        if not candidates:
            return None
        return candidates[0]

    records = []
    for player in players:
        p = support_for(player)
        if p is None:
            return None
        records.append((player[0], player[1], p))
    records.sort()
    model["types"] = [(color, height) for color, height, p in records]
    groups = {}
    for i, typ in enumerate(model["types"]):
        groups.setdefault(typ, []).append(i)
    model["groups"] = list(groups.values())

    colored = np.isin(im, [11, 14])
    for color, h, l, r, t, base, end in players:
        colored[base:end, l:r] = False

    targets = []
    for color in (11, 14):
        for t, l, b, r in _components(colored & (im == color)):
            if b-t > 6 or r-l > 12:
                continue
            matching = [wall for wall in walls
                        if min(r, wall[1]) > max(l, wall[0])]
            if not matching:
                continue
            wall = max(matching,
                       key=lambda w: min(r, w[1])-max(l, w[0]))
            adjacent = [i for i, p in enumerate(platforms)
                        if p["r"] == wall[0] or p["l"] == wall[1]]
            targets.append((color, b, 3*(b-t), adjacent))
    model["targets"] = targets

    actions = []
    for t, l, b, r in buttons:
        wall = next((w for w in walls if r == w[0] or l == w[1]), None)
        if wall is None:
            continue
        left = [i for i, p in enumerate(platforms)
                if p["r"] == wall[0] and p["anchor"] >= t]
        right = [i for i, p in enumerate(platforms)
                 if p["l"] == wall[1] and p["anchor"] >= t]
        if not left or not right:
            continue
        key = ((lambda i: platforms[i]["anchor"]) if negative
               else (lambda i: tops[i]))
        a, z = min(left, key=key), min(right, key=key)
        receiver, donor = (a, z) if r == wall[0] else (z, a)
        actions.append(("transfer", receiver, donor,
                        ((t+b-1)//2, (l+r-1)//2)))

    bridges = []
    for t, l, b, r in _components(np.isin(im, [1, 12])):
        if b-t < 2 or r-l > 8:
            continue
        if not any(w[0] == l and w[1] == r for w in walls):
            continue
        left = [i for i, p in enumerate(platforms) if p["r"] == l]
        right = [i for i, p in enumerate(platforms) if p["l"] == r]
        if left and right:
            bridges.append((left, right, b))
            actions.append(("bridge", left, right, b,
                            ((t+b-1)//2, (l+r-1)//2)))

    model["actions"] = actions
    model["bridges"] = bridges
    if not actions or not targets:
        return None
    state = tuple(tops) + _normalize(model, [p for c, h, p in records])
    _prepare(model)
    return model, state


def _read(model, im):
    by_lane = [_black_intervals(im, lane) for lane in model["lanes"]]
    tops = []
    for p in model["platforms"]:
        tops.append(next((a for a, b in by_lane[p["lane"]]
                          if b == p["anchor"]), p["anchor"]))

    records = []
    for color, h, l, r, t, base, end in _players(im):
        x = (l+r)/2
        choices = [i for i, p in enumerate(model["platforms"])
                   if p["l"] <= x < p["r"] and tops[i] == end]
        if not choices:
            return None
        records.append((color, h, choices[0]))
    records.sort()
    if [(c, h) for c, h, p in records] != model["types"]:
        return None
    return tuple(tops) + _normalize(model, [p for c, h, p in records])


def _prepare(model):
    n, step = model["n"], model["step"]
    minimum = [min(_limit(model, i, False), _limit(model, i, True))-step+1
               for i in range(n)]
    maximum = [p["anchor"] for p in model["platforms"]]

    edges = []
    for left, right, level in model["bridges"]:
        for a in left:
            for b in right:
                if (minimum[a] <= level <= maximum[a]
                        and minimum[b] <= level <= maximum[b]):
                    edges.append((a, b, level))

    goals_by_type = {}
    choices_by_type = {}
    for typ in set(model["types"]):
        color, height = typ
        goals = [set() for _ in range(n)]
        levels = [set() for _ in range(n)]
        for c, bottom, full_height, adjacent in model["targets"]:
            if c != color:
                continue
            support = bottom if model["negative"] else bottom-full_height+height
            for p in adjacent:
                if minimum[p] <= support <= maximum[p]:
                    goals[p].add(support)
                    levels[p].add(support)
        for a, b, level in edges:
            levels[a].add(level)
            levels[b].add(level)

        nodes = [(p, level) for p in range(n) for level in sorted(levels[p])]
        index = {v: i for i, v in enumerate(nodes)}
        graph = [[] for _ in nodes]
        for p in range(n):
            for x in levels[p]:
                for y in levels[p]:
                    if x != y:
                        graph[index[(p, x)]].append(
                            (index[(p, y)], math.ceil(abs(x-y)/step)))
        for a, b, level in edges:
            i, j = index[(a, level)], index[(b, level)]
            graph[i].append((j, 1))
            graph[j].append((i, 1))

        distance = [100000] * len(nodes)
        queue = []
        for p in range(n):
            for level in goals[p]:
                i = index[(p, level)]
                distance[i] = 0
                heapq.heappush(queue, (0, i))
        while queue:
            d, i = heapq.heappop(queue)
            if d != distance[i]:
                continue
            for j, cost in graph[i]:
                if d+cost < distance[j]:
                    distance[j] = d+cost
                    heapq.heappush(queue, (d+cost, j))

        choices = [[] for _ in range(n)]
        for a, b, level in edges:
            choices[a].append((b, level, distance[index[(b, level)]]))
            choices[b].append((a, level, distance[index[(a, level)]]))
        goals_by_type[typ] = goals
        choices_by_type[typ] = choices

    model["goals"] = goals_by_type
    model["choices"] = choices_by_type


def _goal(model, state):
    n = model["n"]
    return all(state[state[n+i]] in model["goals"][typ][state[n+i]]
               for i, typ in enumerate(model["types"]))


def _heuristic(model, state):
    n, step = model["n"], model["step"]
    answer = 0
    for i, typ in enumerate(model["types"]):
        p = state[n+i]
        top = state[p]
        best = min((math.ceil(abs(top-g)/step)
                    for g in model["goals"][typ][p]), default=100000)
        for other, level, after in model["choices"][typ][p]:
            alignment = max(math.ceil(abs(top-level)/step),
                            math.ceil(abs(state[other]-level)/step))
            best = min(best, alignment+1+after)
        answer = max(answer, best)
    return answer


def _transition(model, state, action_id):
    override_key = (state, action_id)
    if override_key in model["overrides"]:
        return model["overrides"][override_key]
    n, step = model["n"], model["step"]
    tops = list(state[:n])
    positions = list(state[n:])
    action = model["actions"][action_id]

    if action[0] == "transfer":
        receiver, donor = action[1:3]
        anchor = model["platforms"][donor]["anchor"]
        if tops[donor] >= anchor:
            return None
        if tops[receiver] <= _limit(model, receiver, receiver in positions):
            return None
        tops[receiver] -= step
        tops[donor] = min(anchor, tops[donor]+step)
    else:
        left, right, level = action[1:4]
        a = next((p for p in left if tops[p] == level), None)
        b = next((p for p in right if tops[p] == level), None)
        if a is None or b is None:
            return None
        changed = False
        for i, p in enumerate(positions):
            if p == a:
                positions[i] = b
                changed = True
            elif p == b:
                positions[i] = a
                changed = True
        if not changed:
            return None

    result = tuple(tops) + _normalize(model, positions)
    return result if result != state else None


def _path(parents, state):
    result = []
    while state in parents:
        previous, action = parents[state]
        result.append(action)
        state = previous
    result.reverse()
    return result


def _solve(model, start, remaining):
    if _goal(model, start):
        return []
    deadline = time.perf_counter()+7.4
    h = _heuristic(model, start)
    queue = [(1.6*h, h, 0, 0, start)]
    costs = {start: 0}
    parents = {}
    serial = expanded = 0
    best, best_key = start, (h, 0)
    first = None

    while queue:
        priority, h, g, serial_old, state = heapq.heappop(queue)
        if costs.get(state) != g:
            continue
        if _goal(model, state):
            return _path(parents, state)
        if state != start and first is None:
            first = state
        if (h, g) < best_key:
            best, best_key = state, (h, g)
        expanded += 1
        if expanded % 128 == 0:
            if time.perf_counter() >= deadline or expanded >= 300000:
                break
        if g >= remaining:
            continue

        for action in range(len(model["actions"])):
            nxt = _transition(model, state, action)
            if nxt is None:
                continue
            ng = g+1
            if ng >= costs.get(nxt, 10**9):
                continue
            nh = _heuristic(model, nxt)
            if nh < 100000 and ng+nh > remaining:
                continue
            costs[nxt] = ng
            parents[nxt] = (state, action)
            serial += 1
            heapq.heappush(queue, (ng+1.6*nh, nh, ng, serial, nxt))

    if best != start:
        return _path(parents, best)
    return _path(parents, first) if first is not None else []


def _fallback(obs, memory):
    im, k = _canonical(obs)
    choices = []
    for color in (12, 9, 1):
        for t, l, b, r in _components(im == color):
            choices.append(_unrotate(k, (t+b-1)//2, (l+r-1)//2))
    if not choices:
        return "ACTION6", 32, 32
    count = memory.get("_fallback_count", 0)
    memory["_fallback_count"] = count+1
    row, col = choices[count % len(choices)]
    return "ACTION6", row, col


def act(obs, available, memory):
    if "ACTION6" not in available:
        return (available[0] if available else "RESET"), None, None

    model = memory.get("_model")
    state = None
    if model is not None:
        im, rotation = _canonical(obs, model["rotation"])
        state = _read(model, im)

    if state is None:
        im, rotation = _canonical(obs)
        parsed = _parse(im, rotation)
        if parsed is None:
            return _fallback(obs, memory)
        model, state = parsed
        memory["_model"] = model
        memory["_plan"] = []
        memory.pop("_last", None)
        memory.pop("_expected", None)

    plan = memory.get("_plan", [])
    expected = memory.get("_expected")
    last = memory.get("_last")
    if last is not None and state != expected:
        previous, action = last
        model["overrides"][(previous, action)] = (
            None if state == previous else state)
        plan = []

    visits = memory.setdefault("_visits", {})
    visits[state] = visits.get(state, 0)+1

    if not plan:
        remaining = max(1, min(50, round(
            np.count_nonzero(np.asarray(obs)[0] == 7)*50/64)))
        plan = _solve(model, state, remaining)

    if not plan:
        candidates = []
        for action in range(len(model["actions"])):
            nxt = _transition(model, state, action)
            if nxt is not None:
                candidates.append((visits.get(nxt, 0),
                                   _heuristic(model, nxt), action))
        if candidates:
            plan = [min(candidates)[2]]
        else:
            return _fallback(obs, memory)

    action = plan[0]
    memory["_plan"] = plan[1:]
    memory["_last"] = (state, action)
    memory["_expected"] = _transition(model, state, action)
    row, col = model["actions"][action][-1]
    row, col = _unrotate(model["rotation"], row, col)
    return "ACTION6", int(row), int(col)

import numpy as np
import heapq
import math
import time
from collections import Counter, defaultdict


def _components(mask):
    """Bounding boxes of four-connected components."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        x0 = x1 = x
        y0 = y1 = y
        count = 0
        while stack:
            r, c = stack.pop()
            count += 1
            x0, x1 = min(x0, c), max(x1, c)
            y0, y1 = min(y0, r), max(y1, r)
            for rr, cc in ((r-1, c), (r+1, c),
                           (r, c-1), (r, c+1)):
                if (0 <= rr < h and 0 <= cc < w
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append((x0, y0, x1 + 1, y1 + 1, count))
    return result


def _runs(bits):
    result = []
    start = None
    for i, b in enumerate(list(bits) + [False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            result.append((start, i))
            start = None
    return result


def _players(a):
    """Find upright arrows, after gravity has been rotated downwards."""
    result = []
    for x, y, x1, y1, count in _components(a == 4):
        w, h = x1-x, y1-y
        if w % 3 or h != 2*(w//3):
            continue
        q = w//3
        if q < 1 or q > 4 or y + 3*q > 64:
            continue
        yellow = a[y:y+2*q, x:x+3*q] == 4
        expected = np.zeros((2*q, 3*q), dtype=bool)
        expected[:q, q:2*q] = True
        expected[q:, :] = True
        if not np.array_equal(yellow, expected):
            continue
        base = a[y+2*q:y+3*q, x:x+3*q]
        color = int(base[0, 0])
        if color in (0, 3, 4, 5, 7, 9) or not np.all(base == color):
            continue
        result.append((color, x, y, 3*q, 3*q))
    return sorted(result)


def _image(obs, k):
    a = np.asarray(obs, dtype=np.int16).copy()
    # The step counter hides one display row.
    a[0, :] = a[1, :]
    return np.rot90(a, k).copy()


def _display(k, x, y):
    x, y = int(x), int(y)
    if k == 0:
        return y, x
    if k == 1:
        return x, 63-y
    if k == 2:
        return 63-y, 63-x
    return 63-x, y


def _canonical(owners, colors):
    owners = list(owners)
    i = 0
    while i < len(colors):
        j = i+1
        while j < len(colors) and colors[j] == colors[i]:
            j += 1
        owners[i:j] = sorted(owners[i:j])
        i = j
    return tuple(owners)


def _limit(m, i, occupied):
    b = m["basins"][i]
    floors = b["floors"]
    if not floors:
        return b["limit"]
    if m["negative"]:
        return min(floors) + (m["clearance"] if occupied else 0)
    return max(floors) - (m["clearance"] if occupied else 0)


def _parse(obs):
    oriented = []
    for k in range(4):
        a = _image(obs, k)
        ps = _players(a)
        oriented.append((len(ps), k, a, ps))
    _, k, a, ps = max(oriented, key=lambda z: z[0])
    if not ps:
        return None, None

    colors = tuple(sorted(p[0] for p in ps))
    palette = set(colors)
    buttons = []
    for box in _components(a == 9):
        x, y, x1, y1, n = box
        if x1-x == y1-y and n == (x1-x)*(y1-y):
            buttons.append(box)
    if not buttons:
        return None, None
    step = Counter(b[2]-b[0] for b in buttons).most_common(1)[0][0]

    nonpad = np.argwhere(a != 4)
    vx0 = int(nonpad[:, 1].min())
    vx1 = int(nonpad[:, 1].max()) + 1
    vy1 = int(nonpad[:, 0].max()) + 1

    ink = np.isin(a, list({5, 1, 12} | palette))
    columns = []
    threshold = max(6, 2*step)
    for x in range(64):
        columns.append(any(e-s >= threshold for s, e in _runs(ink[:, x])))

    walls = []
    for x0, x1 in _runs(columns):
        if x1-x0 > 2*step:
            continue
        yy = np.nonzero(np.any(ink[:, x0:x1], axis=1))[0]
        if len(yy):
            walls.append((x0, x1, int(yy.min()), int(yy.max())+1))
    if not walls:
        return None, None
    walls.sort()

    wall_columns = np.zeros(64, dtype=bool)
    for x0, x1, _, _ in walls:
        wall_columns[x0:x1] = True

    floor_mask = (a == 5) & (~wall_columns[None, :])
    floors = []
    for x, y, x1, y1, n in _components(floor_mask):
        if x1-x >= 2*step and y1-y < x1-x:
            if n == (x1-x)*(y1-y):
                floors.append((x, y, x1, y1))

    spans = []
    last = vx0
    for x0, x1, _, _ in walls:
        if x0 > last:
            spans.append((last, x0))
        last = max(last, x1)
    if last < vx1:
        spans.append((last, vx1))

    basins = []
    water = np.isin(a, [0, 9])
    for x0, x1 in spans:
        solid = np.all(water[:, x0:x1], axis=1)
        solid[vy1:] = False
        bottoms = {e for s, e in _runs(solid) if e > s}
        local_buttons = [
            b for b in buttons if x0 <= b[0] and b[2] <= x1
        ]
        physical = {vy1}
        for fx, fy, fx1, fy1 in floors:
            if fx == x0 and fx1 == x1:
                physical.add(fy)

        # Buttons may float above a reservoir. Only infer invisible,
        # empty reservoirs at actual supporting surfaces.
        for bottom in physical:
            if any(b[3] == bottom for b in local_buttons):
                bottoms.add(bottom)
            if any(x0 <= p[1] < x1 and p[2]+p[4] == bottom for p in ps):
                bottoms.add(bottom)
        if not bottoms and local_buttons:
            needed = max(b[3] for b in local_buttons)
            choices = [v for v in physical if v >= needed]
            if choices:
                bottoms.add(min(choices))

        for bottom in sorted(bottoms):
            top = bottom
            while top > 0 and solid[top-1]:
                top -= 1
            # Thin marks from gate outlines are not reservoirs.
            if bottom-top == 1 and x1-x0 <= 2:
                continue
            adjacent = [
                w for w in walls if w[1] == x0 or w[0] == x1
            ]
            if not adjacent:
                continue
            applicable_floors = [
                f[3] for f in floors if f[0] == x0 and f[3] < bottom
            ]
            basins.append({
                "x": x0, "end": x1, "bottom": bottom,
                "limit": max(w[2] for w in adjacent),
                "floors": applicable_floors,
                "initial": top,
            })

    basins.sort(key=lambda b: (b["x"], b["bottom"]))
    if not basins:
        return None, None

    actions = []
    for x, y, x1, y1, _ in buttons:
        wall = None
        receiving_left = False
        for w in walls:
            if x1 == w[0]:
                wall, receiving_left = w, True
                break
            if x == w[1]:
                wall, receiving_left = w, False
                break
        if wall is None:
            continue
        left = [i for i, b in enumerate(basins)
                if b["end"] == wall[0] and b["bottom"] >= y1]
        right = [i for i, b in enumerate(basins)
                 if b["x"] == wall[1] and b["bottom"] >= y1]
        if not left or not right:
            continue
        li = min(left, key=lambda i: basins[i]["bottom"])
        ri = min(right, key=lambda i: basins[i]["bottom"])
        donor, recipient = (ri, li) if receiving_left else (li, ri)
        row, col = _display(k, (x+x1-1)//2, (y+y1-1)//2)
        actions.append(("t", donor, recipient, row, col))

    for x, y, x1, y1, _ in _components(np.isin(a, [1, 12])):
        if y1-y <= x1-x:
            continue
        wall = next((w for w in walls
                     if w[0] <= x and x1 <= w[1]), None)
        if wall is None:
            continue
        left = tuple(i for i, b in enumerate(basins)
                     if b["end"] == wall[0])
        right = tuple(i for i, b in enumerate(basins)
                      if b["x"] == wall[1])
        if left and right:
            row, col = _display(k, (x+x1-1)//2, (y+y1-1)//2)
            actions.append(("g", left, right, y1, row, col))

    goals = defaultdict(set)
    for color in palette:
        for x, y, x1, y1, _ in _components(a == color):
            wall = next((w for w in walls
                         if w[0] <= x and x1 <= w[1]), None)
            if wall is None:
                continue
            for i, b in enumerate(basins):
                if b["end"] == wall[0] or b["x"] == wall[1]:
                    goals[color].add((i, y1))

    scale = 2 if step == 4 else 1
    m = {
        "k": k, "step": step, "basins": basins,
        "colors": colors, "actions": actions, "goals": dict(goals),
        "negative": k in (1, 2),
        "clearance": 6 if k == 1 and step == 3 else 4*scale,
    }
    state = _read(m, obs)
    return m, state


def _read(m, obs):
    a = _image(obs, m["k"])
    water = np.isin(a, [0, 9])
    tops = []
    for b in m["basins"]:
        solid = np.all(water[:, b["x"]:b["end"]], axis=1)
        top = b["bottom"]
        while top > 0 and solid[top-1]:
            top -= 1
        tops.append(top)

    pieces = _players(a)
    if tuple(sorted(p[0] for p in pieces)) != m["colors"]:
        return None
    found = []
    for color, x, y, w, h in pieces:
        choices = [
            i for i, b in enumerate(m["basins"])
            if b["x"] <= x < b["end"] and tops[i] == y+h
        ]
        if not choices:
            return None
        owner = min(choices, key=lambda i: m["basins"][i]["bottom"])
        found.append((color, owner))
    found.sort()
    owners = _canonical([z[1] for z in found], m["colors"])
    return tuple(tops) + owners


def _successors(m, state):
    n = len(m["basins"])
    tops, owners = state[:n], state[n:]
    occupied = set(owners)
    step = m["step"]
    for ai, action in enumerate(m["actions"]):
        if action[0] == "t":
            donor, recipient = action[1:3]
            if tops[donor] >= m["basins"][donor]["bottom"]:
                continue
            if tops[recipient] <= _limit(m, recipient, recipient in occupied):
                continue
            newtops = list(tops)
            newtops[donor] += step
            newtops[recipient] -= step
            yield ai, tuple(newtops) + owners
        else:
            left, right, height = action[1:4]
            ls = [i for i in left if tops[i] == height]
            rs = [i for i in right if tops[i] == height]
            if not ls or not rs:
                continue
            li, ri = ls[0], rs[0]
            if li not in occupied and ri not in occupied:
                continue
            newowners = tuple(
                ri if i == li else li if i == ri else i for i in owners
            )
            newowners = _canonical(newowners, m["colors"])
            newstate = tops + newowners
            if newstate != state:
                yield ai, newstate


def _heuristic(m, start):
    """Relaxed shortest routes through gate-height waypoints."""
    n = len(m["basins"])
    step = m["step"]
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for action in m["actions"]:
        if action[0] == "t":
            i, j = action[1:3]
            parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)

    lower, upper = [0]*n, [0]*n
    for group in groups.values():
        volume = sum(m["basins"][i]["bottom"]-start[i] for i in group)
        for i in group:
            lim = min(_limit(m, i, False), _limit(m, i, True))
            residue = start[i] % step
            lattice_limit = residue + ((lim-residue)//step)*step
            lower[i] = max(
                min(start[i], lattice_limit),
                m["basins"][i]["bottom"] - volume
            )
        capacity = sum(m["basins"][i]["bottom"]-lower[i] for i in group)
        for i in group:
            b = m["basins"][i]["bottom"]
            other_capacity = capacity - (b-lower[i])
            upper[i] = min(b, b-volume+other_capacity)

    def feasible(i, h):
        return (lower[i] <= h <= upper[i]
                and (h-start[i]) % step == 0)

    points = {}
    by_basin = [[] for _ in range(n)]

    def point(i, h):
        key = (i, h)
        if key not in points:
            points[key] = len(points)
            by_basin[i].append((h, points[key]))
        return points[key]

    goalnodes = defaultdict(list)
    for color, goals in m["goals"].items():
        for i, h in goals:
            if feasible(i, h):
                goalnodes[color].append(point(i, h))

    bridges = []
    for action in m["actions"]:
        if action[0] != "g":
            continue
        left, right, height = action[1:4]
        for i in left:
            for j in right:
                if feasible(i, height) and feasible(j, height):
                    bridges.append((point(i, height), point(j, height)))

    graph = [[] for _ in points]
    for nodes in by_basin:
        nodes.sort()
        for (h1, p1), (h2, p2) in zip(nodes, nodes[1:]):
            cost = (abs(h1-h2)+step-1)//step
            graph[p1].append((p2, cost))
            graph[p2].append((p1, cost))
    for p1, p2 in bridges:
        graph[p1].append((p2, 1))
        graph[p2].append((p1, 1))

    distances = {}
    inf = 10000
    for color in set(m["colors"]):
        d = [inf]*len(points)
        heap = []
        for p in goalnodes[color]:
            d[p] = 0
            heapq.heappush(heap, (0, p))
        while heap:
            cost, p = heapq.heappop(heap)
            if d[p] != cost:
                continue
            for q, w in graph[p]:
                nd = cost+w
                if nd < d[q]:
                    d[q] = nd
                    heapq.heappush(heap, (nd, q))
        distances[color] = d

    cache = {}

    def evaluate(state):
        maximum = total = 0
        for color, owner in zip(m["colors"], state[n:]):
            height = state[owner]
            key = (color, owner, height)
            value = cache.get(key)
            if value is None:
                d = distances[color]
                value = min(
                    (d[p] + (abs(height-h)+step-1)//step
                     for h, p in by_basin[owner]),
                    default=inf
                )
                cache[key] = value
            maximum = max(maximum, value)
            total += value
        return maximum, total

    return evaluate


def _is_goal(m, state):
    n = len(m["basins"])
    return all(
        (owner, state[owner]) in m["goals"].get(color, ())
        for color, owner in zip(m["colors"], state[n:])
    )


def _search(m, start, deadline, remaining):
    evaluate = _heuristic(m, start)
    h0, hs0 = evaluate(start)
    if _is_goal(m, start):
        return [], evaluate
    if h0 >= 10000:
        return [], evaluate

    serial = 0
    heap = [(1.65*h0, hs0, serial, 0, start)]
    costs = {start: 0}
    parents = {}
    best = start
    best_key = (h0, hs0, 0)
    reached = None
    expanded = 0

    while heap and expanded < 220000:
        _, _, _, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        expanded += 1
        if expanded % 128 == 0 and time.perf_counter() > deadline:
            break
        if _is_goal(m, state):
            reached = state
            break
        if g >= remaining:
            continue

        for ai, newstate in _successors(m, state):
            ng = g+1
            if ng >= costs.get(newstate, 10**9):
                continue
            h, hs = evaluate(newstate)
            if h >= 10000 or ng+h > remaining:
                continue
            costs[newstate] = ng
            parents[newstate] = (state, ai)
            key = (h, hs, ng)
            if key < best_key:
                best_key, best = key, newstate
            serial += 1
            heapq.heappush(
                heap, (ng+1.65*h, hs, serial, ng, newstate)
            )

    end = reached if reached is not None else best
    path = []
    while end != start and end in parents:
        prev, ai = parents[end]
        path.append((ai, end))
        end = prev
    path.reverse()
    return path, evaluate


def _emit(m, item, memory, rest):
    ai, nextstate = item
    action = m["actions"][ai]
    memory["expected"] = nextstate
    memory["plan"] = rest
    return "ACTION6", int(action[-2]), int(action[-1])


def act(obs, available, memory):
    started = time.perf_counter()
    try:
        m = memory.get("model")
        state = _read(m, obs) if m is not None else None

        if m is not None and state is not None:
            plan = memory.get("plan", [])
            if state == memory.get("expected") and plan:
                return _emit(m, plan[0], memory, plan[1:])

        if m is None or state is None:
            m, state = _parse(obs)
            memory["plan"] = []
            memory.pop("expected", None)

        if m is not None and state is not None:
            memory["model"] = m
            visits = memory.setdefault("visits", {})
            visits[state] = visits.get(state, 0)+1

            if _is_goal(m, state):
                return "ACTION6", 1, 1

            bar = int(np.count_nonzero(np.asarray(obs)[0] == 7))
            remaining = max(1, int(round(50*bar/64))) if bar else 50
            path, evaluate = _search(
                m, state, started+8.3, remaining
            )
            if path:
                return _emit(m, path[0], memory, path[1:])

            # A bounded-search fallback still uses legal model transitions.
            choices = []
            for ai, ns in _successors(m, state):
                h, hs = evaluate(ns)
                penalty = 4*visits.get(ns, 0)
                choices.append((h+penalty, hs, ai, ns))
            if choices:
                _, _, ai, ns = min(choices)
                return _emit(m, (ai, ns), memory, [])

        # Safe image-based fallback if an unusual rendering was not parsed.
        cells = np.argwhere(np.asarray(obs) == 12)
        if len(cells):
            r, c = cells[len(cells)//2]
            return "ACTION6", int(r), int(c)

        boxes = _components(np.asarray(obs) == 9)
        if boxes:
            turn = memory.get("fallback_turn", 0)
            memory["fallback_turn"] = turn+1
            x, y, x1, y1, _ = boxes[turn % len(boxes)]
            return "ACTION6", (y+y1-1)//2, (x+x1-1)//2

    except Exception:
        # Keep the interface total even for an unanticipated partial image.
        memory.pop("model", None)
        memory.pop("plan", None)
        cells = np.argwhere(np.asarray(obs) == 9)
        if len(cells):
            r, c = cells[len(cells)//2]
            return "ACTION6", int(r), int(c)

    return "ACTION6", 32, 32

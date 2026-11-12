import time
import heapq
from collections import deque
import numpy as np

_INF = 1000000


def _parse(image):
    a = np.asarray(image)
    handles = []
    covered = np.zeros((64, 64), dtype=bool)

    for y in range(59):
        for x in range(59):
            c = int(a[y, x])
            if c < 5:
                continue
            if not (a[y, x + 5] == c and a[y + 5, x] == c
                    and a[y + 5, x + 5] == c):
                continue
            z = int(a[y + 2, x + 2])
            if z < 6 or not np.all(a[y + 2:y + 4, x + 2:x + 4] == z):
                continue
            if covered[y + 2, x + 2]:
                continue
            w = a[y:y + 6, x:x + 6]
            found = None
            for k in range(4):
                u = np.rot90(w, k)
                r = int(u[1, 1])
                if r not in (0, 4, 5):
                    continue
                if r == 5 and c != 5:
                    continue
                if not (np.all(u[0] == c) and np.all(u[5] == c)
                        and np.all(u[:, 0] == c)):
                    continue
                if not (u[1, 5] == c and u[4, 5] == c):
                    continue
                if not (np.all(u[1, 1:5] == r)
                        and np.all(u[4, 1:5] == r)
                        and np.all(u[2:4, 1] == r)
                        and np.all(u[2:4, 4] == r)):
                    continue
                found = {
                    "x": x, "y": y, "rot": k,
                    "color": z, "selected": r == 0,
                }
                break
            if found is not None:
                handles.append(found)
                covered[y:y + 6, x:x + 6] = True

    boardmask = (a == 4) & ~covered
    boardmask[53:] = False
    yy, xx = np.where(boardmask)
    if len(xx) == 0:
        raise ValueError("Board not found")
    bx, by = int(xx.min()), int(yy.min())
    bw, bh = int(xx.max()) - bx + 1, int(yy.max()) - by + 1
    W, H = int(round(bw / 6)), int(round(bh / 6))
    if not (1 <= W <= 10 and 1 <= H <= 9):
        raise ValueError("Bad board dimensions")
    stride = W + 4

    def enc(x, y):
        return (y + 2) * stride + x + 2

    blocks = {}
    for y in range(61):
        for x in range(61):
            if covered[y, x]:
                continue
            c = int(a[y, x])
            if c < 6:
                continue
            if not (a[y, x + 3] == c and a[y + 3, x] == c
                    and a[y + 3, x + 3] == c):
                continue
            if not (np.all(a[y, x:x + 4] == c)
                    and np.all(a[y + 3, x:x + 4] == c)
                    and np.all(a[y:y + 4, x] == c)
                    and np.all(a[y:y + 4, x + 3] == c)):
                continue
            middle = a[y + 1:y + 3, x + 1:x + 3]
            if np.all((middle == c) | (middle == 0)):
                blocks[(x - 1, y - 1)] = c

    def track(x, y, horizontal):
        if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
            return False
        if horizontal:
            v = (int(a[y + 2, x]), int(a[y + 2, x + 5]),
                 int(a[y + 3, x]), int(a[y + 3, x + 5]))
            return v in ((3, 2, 2, 3), (2, 1, 1, 2))
        v = (int(a[y, x + 2]), int(a[y, x + 3]),
             int(a[y + 5, x + 2]), int(a[y + 5, x + 3]))
        return v in ((2, 3, 3, 2), (1, 2, 2, 1))

    vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for h in handles:
        dx, dy = vectors[h["rot"]]
        n = 1
        while n < 12 and track(h["x"] + 6 * dx * n,
                               h["y"] + 6 * dy * n, dx != 0):
            n += 1
        h["n"] = n

    footer = [h for h in handles if h["y"] >= 53]
    main = [h for h in handles if h["y"] < 53]
    main.sort(key=lambda h: (h["color"], h["rot"],
                             h["x"] if h["rot"] % 2 == 0 else h["y"],
                             h["y"], h["x"]))

    def pixel(x, y):
        if 0 <= x < 64 and 0 <= y < 64:
            return int(a[y, x])
        return -1

    def occluded(x, y):
        return 0 <= x < 64 and 0 <= y < 64 and covered[y, x]

    arms = []
    armstate = []
    active = None
    for h in main:
        dx, dy = vectors[h["rot"]]
        horizontal = dx != 0
        gx = int(round((h["x"] - bx) / 6))
        gy = int(round((h["y"] - by) / 6))
        axis = gx if horizontal else gy
        v = gy if horizontal else gx
        partner = next((f for f in footer if f["color"] == h["color"]), None)
        goal = []
        if partner is not None:
            fx, fy = vectors[partner["rot"]]
            for j in range(1, partner["n"]):
                p = (partner["x"] + 6 * j * fx,
                     partner["y"] + 6 * j * fy)
                if p in blocks:
                    goal.append(blocks[p])
            if not goal:
                for j in range(1, 11):
                    p = (partner["x"] + 6 * j * fx,
                         partner["y"] + 6 * j * fy)
                    if p not in blocks:
                        break
                    goal.append(blocks[p])

        extent = H if horizontal else W
        edges = set()
        if goal:
            for t in range(extent - 1):
                if horizontal:
                    qx, qy = bx + 6 * axis + 2, by + 6 * t + 5
                    centers = ((qx, by + 6 * t + 2),
                               (qx, by + 6 * (t + 1) + 2))
                else:
                    qx, qy = bx + 6 * t + 5, by + 6 * axis + 2
                    centers = ((bx + 6 * t + 2, qy),
                               (bx + 6 * (t + 1) + 2, qy))
                present = pixel(qx, qy) in (2, 3)
                if not present and occluded(qx, qy):
                    present = any(pixel(cx, cy) in (2, 3)
                                  for cx, cy in centers if not occluded(cx, cy))
                if present:
                    edges.add(t)

        anchor = enc(axis, 0) if horizontal else enc(0, axis)
        arm = {
            "color": h["color"], "horizontal": horizontal,
            "axis": axis, "anchor": anchor,
            "pd": stride if horizontal else 1,
            "d": dx + dy * stride,
            "goal": tuple(goal), "edges": frozenset(edges),
            "v": v, "n": h["n"], "extent": extent,
        }
        arms.append(arm)
        armstate.extend((v, h["n"]))
        if goal and h["selected"]:
            active = len(arms) - 1

    controls = [i for i, arm in enumerate(arms) if arm["goal"]]
    if not controls:
        raise ValueError("No controllable arm")
    if active is None:
        active = controls[0]

    obstacles = set()
    base_positions = {(h["x"], h["y"]) for h in main}
    for y in range(H):
        for x in range(W):
            px, py = bx + 6 * x, by + 6 * y
            if (px, py) in base_positions:
                continue
            tile = a[py:py + 6, px:px + 6]
            if tile.shape != (6, 6):
                continue
            if (tile[0, 0] == 5 and tile[0, 5] == 5
                    and tile[5, 0] == 5 and tile[5, 5] == 5
                    and np.count_nonzero(tile == 5) >= 24):
                obstacles.add(enc(x, y))

    colored = []
    for (px, py), c in blocks.items():
        if py >= 53:
            continue
        x, y = (px - bx) // 6, (py - by) // 6
        if (0 <= x < W and 0 <= y < H
                and (px - bx) % 6 == 0 and (py - by) % 6 == 0):
            colored.append((c, enc(x, y)))
    colored.sort()
    colors = tuple(c for c, p in colored)
    positions = tuple(p for c, p in colored)

    signature = (
        bx, by, W, H, colors, tuple(sorted(obstacles)),
        tuple((arm["color"], arm["horizontal"], arm["axis"], arm["d"],
               arm["goal"], tuple(sorted(arm["edges"])),
               None if arm["goal"] else (arm["v"], arm["n"]))
              for arm in arms),
    )
    spec = dict(bx=bx, by=by, W=W, H=H, stride=stride,
                arms=arms, controls=controls, obstacles=obstacles,
                colors=colors, signature=signature)
    return spec, (tuple(armstate), positions, active)


class _World:
    def __init__(self, spec):
        self.bx = spec["bx"]
        self.by = spec["by"]
        self.W = spec["W"]
        self.H = spec["H"]
        self.stride = spec["stride"]
        self.arms = spec["arms"]
        self.controls = spec["controls"]
        self.obstacles = spec["obstacles"]
        self.colors = spec["colors"]
        self.signature = spec["signature"]

        self.valid = {
            (y + 2) * self.stride + x + 2
            for y in range(self.H) for x in range(self.W)
        } - self.obstacles
        self.deltas = {1: -self.stride, 2: self.stride, 3: -1, 4: 1}

        groups = []
        start = 0
        while start < len(self.colors):
            end = start + 1
            while end < len(self.colors) and self.colors[end] == self.colors[start]:
                end += 1
            groups.append((start, end))
            start = end
        self.groups = groups

        fixed_h, fixed_v = set(), set()
        for arm in self.arms:
            if arm["goal"]:
                continue
            base = arm["anchor"] + arm["v"] * arm["pd"]
            target = fixed_h if arm["horizontal"] else fixed_v
            target.update(base + j * arm["d"] for j in range(arm["n"]))

        graph = {}
        for p in self.valid:
            ns = []
            for delta in (-1, 1, -self.stride, self.stride):
                q = p + delta
                if q not in self.valid:
                    continue
                pins = fixed_v if abs(delta) == 1 else fixed_h
                if p in pins or q in pins:
                    continue
                ns.append(q)
            graph[p] = ns

        self.dist = {}
        for p in self.valid:
            distances = {p: 0}
            queue = deque([p])
            while queue:
                u = queue.popleft()
                nd = distances[u] + 1
                for q in graph[u]:
                    if q not in distances:
                        distances[q] = nd
                        queue.append(q)
            self.dist[p] = distances

        self.tables = {}
        self.assignments = {}
        self.lines = {}
        for ai in self.controls:
            arm = self.arms[ai]
            goal = arm["goal"]
            masks = {0}
            transitions = {}
            frontier = {0}
            for k, color in enumerate(goal):
                following = set()
                for mask in frontier:
                    opts = []
                    for b, c in enumerate(self.colors):
                        if c == color and not (mask & (1 << b)):
                            nm = mask | (1 << b)
                            opts.append((b, nm))
                            following.add(nm)
                    transitions[mask] = opts
                if k + 1 < len(goal):
                    masks.update(following)
                frontier = following
            ordered = sorted(masks, key=lambda m: (bin(m).count("1"), m))
            index = {m: j for j, m in enumerate(ordered)}
            compiled = []
            for mask in ordered:
                opts = []
                for b, nm in transitions.get(mask, ()):
                    opts.append((b, index.get(nm, -1)))
                compiled.append(opts)
            self.assignments[ai] = compiled

            reachable = {arm["v"]}
            queue = [arm["v"]]
            while queue:
                v = queue.pop()
                for nv in (v - 1, v + 1):
                    if (0 <= nv < arm["extent"]
                            and min(v, nv) in arm["edges"]
                            and nv not in reachable):
                        reachable.add(nv)
                        queue.append(nv)
            lines = []
            for v in sorted(reachable):
                base = arm["anchor"] + v * arm["pd"]
                cells = []
                for k in range(1, 13):
                    p = base + k * arm["d"]
                    if p in self.valid:
                        cells.append((p, k + 1))
                if len(cells) >= len(goal):
                    lines.append((v, cells))
            self.lines[ai] = lines

    def canonical(self, positions):
        result = list(positions)
        for lo, hi in self.groups:
            if hi - lo > 1:
                result[lo:hi] = sorted(result[lo:hi])
        return tuple(result)

    def goal(self, state):
        av, positions, selected = state
        occupied = dict(zip(positions, self.colors))
        for ai in self.controls:
            arm = self.arms[ai]
            v, n = av[2 * ai:2 * ai + 2]
            base = arm["anchor"] + v * arm["pd"]
            sequence = []
            for j in range(n):
                c = occupied.get(base + j * arm["d"])
                if c is not None:
                    sequence.append(c)
            target = arm["goal"]
            if len(sequence) < len(target) or tuple(sequence[:len(target)]) != target:
                return False
        return True

    def move(self, state, ai, action):
        av, positions, selected = state
        arm = self.arms[ai]
        delta = self.deltas[action]
        direction = arm["d"]
        v, n = av[2 * ai:2 * ai + 2]
        base = arm["anchor"] + v * arm["pd"]
        along = delta == direction
        back = delta == -direction

        if along:
            if base + n * direction not in self.valid:
                return None
        elif back:
            if n <= 1:
                return None
        else:
            change = -1 if action in (1, 3) else 1
            if min(v, v + change) not in arm["edges"]:
                return None

        hmap, vmap = {}, {}
        for j, other in enumerate(self.arms):
            ov, on = av[2 * j:2 * j + 2]
            ob = other["anchor"] + ov * other["pd"]
            target = hmap if other["horizontal"] else vmap
            begin = 1 if j == ai and back else 0
            for k in range(begin, on):
                p = ob + k * other["d"]
                target[p] = target.get(p, 0) + 1

        occupied = {p: b for b, p in enumerate(positions)}
        moved = 0
        move_horizontal = abs(delta) == 1
        pins = vmap if move_horizontal else hmap
        segment_perpendicular = arm["horizontal"] != move_horizontal

        def push(b, asking=None):
            nonlocal moved
            bit = 1 << b
            if moved & bit:
                return True
            p = positions[b]
            q = p + delta
            if q not in self.valid:
                return False
            for cell in (p, q):
                count = pins.get(cell, 0)
                if (asking is not None and segment_perpendicular
                        and cell == asking):
                    count -= 1
                if count > 0:
                    return False
            nxt = occupied.get(q)
            if nxt is not None and not push(nxt):
                return False
            moved |= bit
            return True

        first = 1 if back else 0
        for k in range(first, n):
            p = base + k * direction
            q = p + delta
            if q not in self.valid and not (back or p not in self.valid):
                return None
            for cell in (p, q):
                b = occupied.get(cell)
                if b is not None and not push(b, p):
                    if segment_perpendicular:
                        return None

        new_av = list(av)
        if along:
            new_av[2 * ai + 1] += 1
        elif back:
            new_av[2 * ai + 1] -= 1
        else:
            new_av[2 * ai] += -1 if action in (1, 3) else 1
        new_positions = [
            p + delta if moved & (1 << b) else p
            for b, p in enumerate(positions)
        ]
        return tuple(new_av), self.canonical(new_positions), ai

    def _goal_tables(self, positions):
        cached = self.tables.get(positions)
        if cached is not None:
            return cached

        occupied = {p: b for b, p in enumerate(positions)}
        distances = [self.dist.get(p, {}) for p in positions]
        answer = {}
        for ai in self.controls:
            compiled = self.assignments[ai]
            candidates = []
            for v, cells in self.lines[ai]:
                dp = [_INF] * len(compiled)
                dp[0] = 0
                for p, needed_length in cells:
                    occupant = occupied.get(p, -1)
                    skip = 2 if occupant >= 0 else 0
                    nxt_dp = [value + skip for value in dp]
                    finish = _INF
                    costs = [2 * d.get(p, _INF) for d in distances]
                    for j, value in enumerate(dp):
                        if value >= _INF:
                            continue
                        for b, nj in compiled[j]:
                            nd = value + costs[b]
                            if occupant >= 0 and occupant != b:
                                nd += 1
                            if nj < 0:
                                if nd < finish:
                                    finish = nd
                            elif nd < nxt_dp[nj]:
                                nxt_dp[nj] = nd
                    if finish < _INF:
                        candidates.append((v, needed_length, finish))
                    dp = nxt_dp
            answer[ai] = candidates

        if len(self.tables) >= 24000:
            self.tables.clear()
        self.tables[positions] = answer
        return answer

    def heuristic(self, state):
        av, positions, selected = state
        tables = self._goal_tables(positions)
        total = 0.0
        for ai in self.controls:
            v, n = av[2 * ai:2 * ai + 2]
            best = float(_INF)
            for gv, length, cost in tables[ai]:
                value = (0.5 * cost + 0.8 * abs(gv - v)
                         + 0.8 * max(0, length - n))
                if value < best:
                    best = value
            total += best
        return total

    def click(self, state, ai):
        av = state[0]
        arm = self.arms[ai]
        v = av[2 * ai]
        if arm["horizontal"]:
            x, y = arm["axis"], v
        else:
            x, y = v, arm["axis"]
        return ("ACTION6", int(self.by + y * 6 + 2),
                int(self.bx + x * 6 + 2))


def _recover(records, state):
    path = []
    while records[state][1] is not None:
        g, parent, move = records[state]
        path.append(move)
        state = parent
    path.reverse()
    return path


def _search(world, root, allowed, walk, deadline):
    actions = [k for k in (1, 2, 3, 4) if "ACTION%d" % k in allowed]
    can_switch = "ACTION6" in allowed
    root_h = world.heuristic(root)
    records = {root: (0, None, None)}
    heap = [(2.2 * root_h, root_h, 0, 0, root)]
    serial = 1
    best_state = None
    best_score = float("inf")
    expansions = 0

    while heap and len(records) < 130000:
        if expansions % 8 == 0 and time.monotonic() >= deadline:
            break
        f, h, g, _, state = heapq.heappop(heap)
        if records[state][0] != g:
            continue
        expansions += 1
        if g > 110:
            continue

        controls = [state[2]] + [i for i in world.controls if i != state[2]]
        for ai in controls:
            if ai not in world.controls:
                continue
            if ai != state[2] and not can_switch:
                continue
            step_cost = 1 + (ai != state[2])
            for action in actions:
                nxt = world.move(state, ai, action)
                if nxt is None or nxt == state:
                    continue
                ng = g + step_cost
                previous = records.get(nxt)
                if previous is not None and previous[0] <= ng:
                    continue
                records[nxt] = (ng, state, (ai, action))
                if world.goal(nxt):
                    return _recover(records, nxt), True

                nh = world.heuristic(nxt)
                revisit = walk.get(nxt, 0)
                progress = nh + 0.015 * ng + 4.0 * revisit
                if progress < best_score:
                    best_score = progress
                    best_state = nxt
                priority = ng + 2.2 * nh + 5.0 * revisit
                heapq.heappush(heap, (priority, nh, ng, serial, nxt))
                serial += 1

    if best_state is not None:
        return _recover(records, best_state)[:4], False
    return [], False


def act(obs, available, memory):
    started = time.monotonic()
    allowed = set(available)

    def fallback():
        for name in ("ACTION7", "RESET", "ACTION1", "ACTION2",
                     "ACTION3", "ACTION4"):
            if name in allowed:
                return name, None, None
        if "ACTION6" in allowed:
            return "ACTION6", 0, 0
        return "RESET", None, None

    try:
        spec, state = _parse(obs)
        world = memory.get("_world")
        if world is None or world.signature != spec["signature"]:
            world = _World(spec)
            memory["_world"] = world
            memory["_plan"] = []
            memory["_walk"] = {}
            memory.pop("_expected", None)

        expected = memory.get("_expected")
        if expected is not None and expected != state:
            memory["_plan"] = []

        walk = memory.setdefault("_walk", {})
        walk[state] = walk.get(state, 0) + 1
        plan = memory.get("_plan", [])

        if plan:
            ai, action = plan[0]
            if world.move(state, ai, action) is None:
                plan = []

        if not plan:
            plan, solved = _search(
                world, state, allowed, walk, started + 8.0
            )
            memory["_plan"] = plan

        if not plan:
            memory.pop("_expected", None)
            return fallback()

        ai, action = plan[0]
        if ai != state[2]:
            if "ACTION6" not in allowed:
                memory["_plan"] = []
                return fallback()
            memory["_expected"] = (state[0], state[1], ai)
            return world.click(state, ai)

        nxt = world.move(state, ai, action)
        if nxt is None:
            memory["_plan"] = []
            memory.pop("_expected", None)
            return fallback()

        memory["_plan"] = plan[1:]
        memory["_expected"] = nxt
        return "ACTION%d" % action, None, None

    except Exception:
        memory.pop("_expected", None)
        memory["_plan"] = []
        return fallback()

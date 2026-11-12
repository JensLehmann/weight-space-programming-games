import numpy as np
import heapq
import time
from collections import deque
from itertools import permutations

_DXY = ((0, 0), (0, -3), (0, 3), (-3, 0), (3, 0))
_DP = (0, -192, 192, -3, 3)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for yy, xx in zip(*np.where(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        cells = []
        while stack:
            u, v = stack.pop()
            cells.append((u, v))
            for nx, ny in ((u-1, v), (u+1, v), (u, v-1), (u, v+1)):
                if 0 <= nx < 64 and 0 <= ny < 63:
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
        result.append(cells)
    return result


def _pieces(a):
    pieces = []
    occupied = np.zeros((64, 64), dtype=bool)

    # Largest-first is important: black selection borders can make a small
    # subrectangle of a selected large block resemble another selected block.
    for h, w in ((6, 6), (6, 3), (3, 6), (3, 3)):
        cy = [1] if h == 3 else [2, 3]
        cx = [1] if w == 3 else [2, 3]
        hole = np.zeros((h, w), dtype=bool)
        hole[np.ix_(cy, cx)] = True
        edge = np.zeros((h, w), dtype=bool)
        edge[0, :] = True
        edge[-1, :] = True
        edge[:, 0] = True
        edge[:, -1] = True
        interior = ~edge & ~hole

        candidates = set()
        for yy, xx in zip(*np.where(np.isin(a[:63], (0, 4, 5)))):
            x, y = int(xx)-cx[0], int(yy)-cy[0]
            if x < 0 or y < 0 or x+w > 64 or y+h > 63:
                continue
            candidates.add((x, y))

        for x, y in sorted(candidates, key=lambda xy: (xy[1], xy[0])):
            if occupied[y:y+h, x:x+w].any():
                continue
            b = a[y:y+h, x:x+w]
            centre = b[hole]
            color = int(centre[0])
            if color not in (0, 4, 5) or not np.all(centre == color):
                continue
            if not np.all(b[interior] == 14):
                continue
            if color == 0:
                if not np.all(np.isin(b[edge], (0, 14))):
                    continue
            elif not np.all(b[edge] == 14):
                continue
            if not np.any(b == 14) and color != 0:
                continue
            pieces.append(dict(k="N", x=x, y=y, w=w, h=h,
                               selected=(color == 0)))
            occupied[y:y+h, x:x+w] = True

    for cells in _components(a == 11):
        xs, ys = zip(*cells)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        pieces.append(dict(
            k="P", x=x, y=y, w=w, h=h,
            cells=[(u-x, v-y) for u, v in cells]
        ))
        for u, v in cells:
            occupied[v, u] = True

    for cells in _components(np.isin(a, (12, 13))):
        xs, ys = zip(*cells)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        if len(cells) != w*h:
            continue
        b = a[y:y+h, x:x+w]
        charged = b == 12
        count = int(charged.sum())
        rot, timer = None, 0
        if count:
            tests = [
                (0, np.all(charged ==
                           (np.arange(h)[:, None] < count//w))),
                (180, np.all(charged ==
                             (np.arange(h)[:, None] >= h-count//w))),
                (90, np.all(charged ==
                            (np.arange(w)[None, :] >= w-count//h))),
                (270, np.all(charged ==
                             (np.arange(w)[None, :] < count//h))),
            ]
            for r, ok in tests:
                if ok:
                    rot = r
                    timer = count // (w if r in (0, 180) else h)
                    break
        pieces.append(dict(k="B", x=x, y=y, w=w, h=h,
                           rot=rot, timer=timer))
        occupied[y:y+h, x:x+w] = True

    unusual = np.isin(a, (3, 6, 7, 8, 9, 10))
    unusual[63] = False
    for cells in _components(unusual):
        xs, ys = zip(*cells)
        x, y = min(xs), min(ys)
        pieces.append(dict(
            k="E", x=x, y=y, w=max(xs)-x+1, h=max(ys)-y+1,
            cells=[(u-x, v-y) for u, v in cells]
        ))
        for u, v in cells:
            occupied[v, u] = True

    return pieces, occupied


def _goal_templates():
    out = []
    for h, w in ((5, 5), (8, 5), (5, 8), (8, 8)):
        s = np.zeros((h, w), dtype=bool)
        s[0, :] = True
        s[-1, :] = True
        s[:, 0] = True
        s[:, -1] = True
        out.append(("N", w-2, h-2, s))
    s = np.array([
        [0,0,0,1,1,1,1,1,0,0,0],
        [0,0,0,1,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,1,0,0,0],
        [1,1,1,1,0,0,0,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,0,0,0,1,1,1,1],
        [0,0,0,1,0,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,1,0,0,0],
        [0,0,0,1,1,1,1,1,0,0,0],
    ], dtype=bool)
    out.append(("P", 9, 9, s))
    return out


def _goals(a, occ, old):
    candidates = []
    fours = list(zip(*np.where(a[:63] == 4)))
    for kind, iw, ih, template in _goal_templates():
        h, w = template.shape
        ry, rx = np.where(template)
        origins = set()
        for yy, xx in fours:
            for dy, dx in zip(ry, rx):
                x, y = int(xx-dx), int(yy-dy)
                if 0 <= x <= 64-w and 0 <= y <= 63-h:
                    origins.add((x, y))
        for x, y in origins:
            vals = a[y+ry, x+rx]
            hidden = occ[y+ry, x+rx]
            visible = vals == 4
            n = int(visible.sum())
            if n < max(8, int(len(rx)*0.58)):
                continue
            if not np.all(visible | hidden):
                continue
            evidence = frozenset(
                (int(x+dx), int(y+dy))
                for dx, dy, v in zip(rx, ry, visible) if v
            )
            candidates.append(
                (n, len(rx), (kind, iw, ih, x+1, y+1), evidence)
            )
    candidates.sort(reverse=True, key=lambda v: (v[0], v[1]))
    result = list(old)
    used = set()
    for n, total, goal, evidence in candidates:
        if len(evidence-used) < max(6, n*0.65):
            continue
        used.update(evidence)
        if goal not in result:
            result.append(goal)
    return result


def _terrain(a, occ, previous):
    if previous is None:
        t = np.zeros((64, 64), dtype=np.int8)
        unknown = occ.copy()
        unknown |= a == 4
        t[(a == 2) & ~unknown] = 2
        t[(a == 15) & ~unknown] = 15
        for _ in range(3):
            changed = False
            for yy, xx in zip(*np.where(unknown[:63])):
                x, y = int(xx), int(yy)
                vals = []
                for dx, dy in ((-1,0), (1,0), (0,-1), (0,1)):
                    u, v = x+dx, y+dy
                    while 0 <= u < 64 and 0 <= v < 63 and unknown[v, u]:
                        u, v = u+dx, v+dy
                    vals.append(
                        int(t[v, u])
                        if 0 <= u < 64 and 0 <= v < 63 else 2
                    )
                if (vals[0] == vals[1] == 15 or
                        vals[2] == vals[3] == 15):
                    t[y, x] = 15
                    unknown[y, x] = False
                    changed = True
            if not changed:
                break
    else:
        t = np.array(previous, dtype=np.int8).reshape(64, 64).copy()
    visible = ~occ
    visible[63] = False
    for color in (1, 2, 15):
        t[visible & (a == color)] = 0 if color == 1 else color
    t[63, :] = 2
    return t


def _bitboard(mask):
    result = 0
    for y in range(64):
        row = 0
        for x in np.flatnonzero(mask[y]):
            row |= 1 << int(x)
        result |= row << (64*y)
    return result


class _Model:
    def __init__(self, pieces, terrain, goals, rots):
        self.pieces = pieces
        self.n = len(pieces)
        self.goals = goals
        self.normals = [i for i,p in enumerate(pieces) if p["k"] == "N"]
        self.bombs = [i for i,p in enumerate(pieces) if p["k"] == "B"]
        self.enemies = [i for i,p in enumerate(pieces) if p["k"] == "E"]
        self.movers = [i for i,p in enumerate(pieces) if p["k"] != "E"]
        self.plus = next(
            (i for i,p in enumerate(pieces) if p["k"] == "P"), None
        )
        self.rots = rots
        self.wall = _bitboard(terrain == 2)
        self.gulch = _bitboard(terrain == 15)
        self.masks = []
        self.valid = []
        self.dry = []
        self.distances = {}

        for p in pieces:
            w, h = p["w"], p["h"]
            cells = p.get("cells")
            if p["k"] in ("N", "B") or cells is None:
                cells = [(x,y) for y in range(h) for x in range(w)]
            base = 0
            for x,y in cells:
                base |= 1 << (x+64*y)
            masks, valid, dry = {}, set(), set()
            for y in range(p["y"] % 3, 64-h, 3):
                for x in range(p["x"] % 3, 65-w, 3):
                    z = x+64*y
                    b = base << z
                    masks[z] = b
                    if not b & self.wall:
                        valid.add(z)
                        if not b & self.gulch:
                            dry.add(z)
            self.masks.append(masks)
            self.valid.append(valid)
            self.dry.append(dry)

        for i, p in enumerate(pieces):
            if p["k"] not in ("N", "P"):
                continue
            for g in goals:
                if g[:3] != (p["k"], p["w"], p["h"]):
                    continue
                target = g[3]+64*g[4]
                d = {target: 0}
                q = deque([target])
                while q:
                    z = q.popleft()
                    for dp in _DP[1:]:
                        v = z+dp
                        if v in self.valid[i] and v not in d:
                            d[v] = d[z]+1
                            q.append(v)
                self.distances[i, g] = d
        self.hcache = {}

    def solved(self, s):
        pos = s[0]
        for g in self.goals:
            if not any(
                self.pieces[i]["k"] == g[0]
                and self.pieces[i]["w"] == g[1]
                and self.pieces[i]["h"] == g[2]
                and pos[i] == g[3]+64*g[4]
                for i in self.movers
            ):
                return False
        return bool(self.goals)

    def heuristic(self, s):
        key = s[0]
        if key in self.hcache:
            return self.hcache[key]
        pos = s[0]
        value = 0.0
        groups = {}
        for g in self.goals:
            groups.setdefault(g[:3], []).append(g)
        for typ, gs in groups.items():
            ids = [
                i for i,p in enumerate(self.pieces)
                if (p["k"],p["w"],p["h"]) == typ
            ]
            if len(ids) < len(gs):
                value += 100
                continue
            costs = []
            for g in gs:
                row = []
                for i in ids:
                    z = pos[i]
                    d = self.distances.get((i,g), {}).get(z, 100)
                    if z == g[3]+64*g[4]:
                        row.append(0.0)
                    elif typ[0] == "P":
                        row.append(0.65*d + 2.0)
                    else:
                        row.append(0.85*d + 0.5)
                costs.append(row)
            if len(ids) <= 7:
                value += min(
                    sum(costs[j][k] for j,k in enumerate(order))
                    for order in permutations(range(len(ids)), len(gs))
                )
            else:
                value += sum(min(row) for row in costs)

        if self.plus is not None:
            i = self.plus
            pg = [g for g in self.goals if g[0] == "P"]
            if pg and all(pos[i] != g[3]+64*g[4] for g in pg):
                px, py = pos[i] % 64, pos[i] // 64
                if self.normals:
                    approach = min(
                        max(0, abs(pos[j] % 64-px) +
                            abs(pos[j] // 64-py)-9)/3
                        for j in self.normals
                    )
                    value += 0.45*approach
        self.hcache[key] = value
        return value

    def step(self, s, action):
        oldpos, selected, oldtimers, oldvel = s
        if action >= 5:
            i = action-5
            if i not in self.normals:
                return None
            return (oldpos, i, oldtimers, oldvel)
        if action not in (1,2,3,4):
            return None

        pos = list(oldpos)
        timers = list(oldtimers)
        vel = list(oldvel)
        dp = _DP[action]

        def push(i, delta, depth=0):
            if depth > 35:
                return True
            before = pos[i]
            new = before+delta
            if new not in self.valid[i]:
                return True
            pos[i] = new
            mask = self.masks[i][new]
            hits = [
                j for j in self.movers
                if j != i and mask & self.masks[j].get(pos[j], 0)
            ]
            for j in hits:
                if push(j, delta, depth+1):
                    pos[i] = before
                    return True
            return False

        proposed = pos[selected]+dp
        if proposed in self.dry[selected]:
            mask = self.masks[selected][proposed]
            hits = [
                j for j in self.movers
                if j != selected and mask & self.masks[j].get(pos[j], 0)
            ]
            if not hits:
                pos[selected] = proposed
            else:
                for j in hits:
                    vel[j] = action
                for tick in range(90):
                    stopped = []
                    for j in hits:
                        in_gulch = bool(
                            self.masks[j].get(pos[j],0) & self.gulch
                        )
                        if tick >= 5 and not in_gulch:
                            stopped.append(True)
                        else:
                            stopped.append(push(j, dp))
                    if all(stopped):
                        break

        if self.plus is not None:
            target = pos[self.plus]
            tx, ty = target % 64, target // 64
            for i in self.enemies:
                x, y = pos[i] % 64, pos[i] // 64
                for _ in range(6):
                    dx, dy = tx-x, ty-y
                    if abs(dx) > abs(dy):
                        x += 1 if dx > 0 else -1
                    elif dy > 0:
                        y += 1
                    else:
                        y -= 1
                    if (x,y) == (tx,ty):
                        return None
                pos[i] = x+64*y

        exploding = []
        for j in self.bombs:
            limit = self.pieces[j]["h"]
            if timers[j] < limit:
                timers[j] += 1
                if timers[j] == limit:
                    exploding.append(j)

        if exploding:
            for radius in (3,6,9):
                for j in exploding:
                    p = self.pieces[j]
                    x, y = pos[j] % 64, pos[j] // 64
                    w = p["w"]
                    rot = self.rots[j]
                    if rot == 0:
                        ex, ey, ew, eh, direction = x, y+w, w, radius, 2
                    elif rot == 180:
                        ex, ey, ew, eh, direction = x, y-radius, w, radius, 1
                    elif rot == 90:
                        ex, ey, ew, eh, direction = x-radius, y, radius, w, 3
                    else:
                        ex, ey, ew, eh, direction = x+w, y, radius, w, 4
                    blast = 0
                    for by in range(max(0,ey), min(63,ey+eh)):
                        lo, hi = max(0,ex), min(64,ex+ew)
                        if hi > lo:
                            blast |= ((1 << (hi-lo))-1) << (lo+64*by)
                    for i in self.movers:
                        if blast & self.masks[i].get(pos[i],0):
                            push(i, _DP[direction])
                            vel[i] = direction
            for j in exploding:
                timers[j] = 0
            for _ in range(100):
                moved = False
                for i in self.movers:
                    if vel[i] and self.masks[i].get(pos[i],0) & self.gulch:
                        if not push(i, _DP[vel[i]]):
                            moved = True
                if not moved:
                    break

        return tuple(pos), selected, tuple(timers), tuple(vel)

    def walking(self, s, i):
        pos = s[0]
        occupied = 0
        for j in self.movers:
            if j != i:
                occupied |= self.masks[j].get(pos[j],0)
        start = pos[i]
        parent = {start: None}
        distance = {start: 0}
        q = deque([start])
        fires = []
        while q:
            z = q.popleft()
            for action in (1,2,3,4):
                new = z+_DP[action]
                if new not in self.dry[i]:
                    continue
                if self.masks[i][new] & occupied:
                    fires.append((z, action, distance[z]))
                elif new not in parent:
                    parent[new] = (z, action)
                    distance[new] = distance[z]+1
                    q.append(new)
        return parent, distance, fires

    @staticmethod
    def walk_path(parent, z):
        path = []
        while parent[z] is not None:
            z, action = parent[z]
            path.append(action)
        path.reverse()
        return path

    def neighbors(self, s, macro):
        results = {}
        if not macro:
            for action in (1,2,3,4):
                v = self.step(s, action)
                if v is not None and v != s:
                    results[v] = [action]
            for i in self.normals:
                if i != s[1]:
                    v = self.step(s, 5+i)
                    results[v] = [5+i]

        for i in self.normals:
            parent, distance, fires = self.walking(s, i)
            prefix = [] if i == s[1] else [5+i]
            options = []
            p = self.pieces[i]
            for g in self.goals:
                if g[:3] == ("N", p["w"], p["h"]):
                    z = g[3]+64*g[4]
                    if z in parent and z != s[0][i]:
                        options.append((z, None))
            if macro:
                options.extend((z, action) for z,action,d in fires)
            else:
                fires.sort(key=lambda item: item[2])
                options.extend(
                    (z, action) for z,action,d in fires[:14] if d <= 9
                )

            for z, action in options:
                path = prefix + self.walk_path(parent, z)
                if action is not None:
                    path.append(action)
                if not path:
                    continue
                if macro:
                    positions = list(s[0])
                    positions[i] = z
                    v = (tuple(positions), i, s[2], s[3])
                    if action is not None:
                        v = self.step(v, action)
                else:
                    v = s
                    for aa in path:
                        v = self.step(v, aa)
                        if v is None:
                            break
                if v is None or v == s:
                    continue
                if v not in results or len(path) < len(results[v]):
                    results[v] = path

        items = list(results.items())
        if macro and len(items) > 90:
            items.sort(
                key=lambda vp: len(vp[1]) + 2.7*self.heuristic(vp[0])
            )
            items = items[:90]
        return items


def _search(model, start, seconds, history):
    deadline = time.perf_counter()+seconds
    macro = not model.bombs and not model.enemies
    h0 = model.heuristic(start)
    heap = [(2.8*h0, 0, 0, start)]
    costs = {start: 0}
    parents = {start: None}
    serial = 0
    best = start
    best_metric = h0
    solved = False
    expansions = 0

    while heap and time.perf_counter() < deadline:
        _, g, _, s = heapq.heappop(heap)
        if g != costs.get(s):
            continue
        if model.solved(s):
            best, solved = s, True
            break
        hs = model.heuristic(s)
        metric = hs + 0.035*g + 1.5*history.get(s,0)
        if s != start and metric < best_metric:
            best, best_metric = s, metric
        for v, path in model.neighbors(s, macro):
            ng = g+len(path)
            if ng >= costs.get(v, 10**9):
                continue
            costs[v] = ng
            parents[v] = (s, path)
            hv = model.heuristic(v)
            serial += 1
            penalty = min(8, history.get(v,0)*2)
            heapq.heappush(
                heap, (ng+2.8*hv+penalty, ng, serial, v)
            )
        expansions += 1
        if expansions >= (18000 if macro else 65000):
            break

    segments = []
    s = best
    while parents.get(s) is not None:
        prev, segment = parents[s]
        segments.append(segment)
        s = prev
    path = [action for segment in reversed(segments) for action in segment]

    if not path:
        alternatives = []
        for action in (1,2,3,4):
            v = model.step(start, action)
            if v is not None:
                alternatives.append((
                    model.heuristic(v) + 3*history.get(v,0), action
                ))
        if alternatives:
            path = [min(alternatives)[1]]
    return path, solved


def _order_pieces(found, previous):
    if not previous or len(previous) != len(found):
        return sorted(
            found,
            key=lambda p: (
                {"P":0, "B":1, "N":2, "E":3}[p["k"]], p["y"], p["x"]
            )
        )
    remaining = list(found)
    ordered = []
    for old in previous:
        candidates = [
            p for p in remaining
            if (p["k"],p["w"],p["h"]) ==
               (old["k"],old["w"],old["h"])
        ]
        if not candidates:
            return _order_pieces(found, None)
        p = min(
            candidates,
            key=lambda p: abs(p["x"]-old["x"])+abs(p["y"]-old["y"])
        )
        remaining.remove(p)
        ordered.append(p)
    return ordered


def act(obs, available, memory):
    started = time.perf_counter()
    a = np.asarray(obs, dtype=np.int8)
    allowed = set(available)

    if np.count_nonzero(a[63] == 4) == 0 and "RESET" in allowed:
        memory.clear()
        return "RESET", None, None

    found, occ = _pieces(a)
    pieces = _order_pieces(found, memory.get("predicted_pieces"))
    normals = [i for i,p in enumerate(pieces) if p["k"] == "N"]
    if not normals:
        memory.clear()
        if "RESET" in allowed:
            return "RESET", None, None
        name = next((x for x in available if x != "RESET"), "ACTION1")
        return (name, 32, 32) if name == "ACTION6" else (name, None, None)

    goals = _goals(a, occ, memory.get("goals", []))
    terrain = _terrain(a, occ, memory.get("terrain"))
    pos = tuple(p["x"]+64*p["y"] for p in pieces)
    selected = next(
        (i for i in normals if pieces[i].get("selected")), None
    )
    prediction = memory.get("prediction")
    if selected is None:
        selected = (
            prediction[1]
            if prediction and prediction[1] in normals else normals[0]
        )

    old_rots = memory.get("rots", [])
    rots = []
    unknown_rotation = False
    for i,p in enumerate(pieces):
        if p["k"] != "B":
            rots.append(0)
        elif p.get("rot") is not None:
            rots.append(p["rot"])
        elif i < len(old_rots):
            rots.append(old_rots[i])
        else:
            rots.append(0)
            unknown_rotation = True

    timers = tuple(p.get("timer",0) for p in pieces)
    matches = bool(
        prediction and prediction[0] == pos
        and prediction[1] == selected and prediction[2] == timers
    )
    vel = prediction[3] if matches else (0,)*len(pieces)
    state = (pos, selected, timers, vel)

    model = _Model(pieces, terrain, goals, rots)
    history = memory.setdefault("history", {})
    history[state] = history.get(state,0)+1
    if len(history) > 1000:
        history.clear()
        history[state] = 1

    plan = memory.get("plan", [])
    if not matches or old_rots != rots:
        plan = []

    # Discard a redundant selection rather than spending an action on it.
    while plan and plan[0] == 5+selected:
        plan.pop(0)

    if not plan:
        desired = 0.65 if unknown_rotation else 7.3
        remaining = max(0.05, 8.7-(time.perf_counter()-started))
        plan, complete = _search(
            model, state, min(desired, remaining), history
        )
        if unknown_rotation:
            plan = plan[:1]
        elif not complete and model.bombs:
            plan = plan[:3]

    if not plan:
        plan = [1]
    action = plan.pop(0)

    if action >= 5:
        i = action-5
        name = "ACTION6"
        row = pieces[i]["y"] + pieces[i]["h"]//2
        col = pieces[i]["x"] + pieces[i]["w"]//2
    else:
        name = "ACTION"+str(action)
        row = col = None

    if name not in allowed:
        name = next(
            (n for n in ("ACTION1","ACTION2","ACTION3","ACTION4")
             if n in allowed),
            available[0]
        )
        row = col = None
        action = int(name[-1]) if name.startswith("ACTION") else 1
        plan = []
        if name == "ACTION6":
            row = pieces[selected]["y"] + pieces[selected]["h"]//2
            col = pieces[selected]["x"] + pieces[selected]["w"]//2
            action = 5+selected

    predicted = model.step(state, action)
    predicted_pieces = []
    if predicted is not None:
        for i,p in enumerate(pieces):
            q = dict(p)
            q["x"] = predicted[0][i] % 64
            q["y"] = predicted[0][i] // 64
            predicted_pieces.append(q)

    memory["terrain"] = terrain.ravel().tolist()
    memory["goals"] = goals
    memory["rots"] = rots
    memory["plan"] = plan
    memory["prediction"] = predicted
    memory["predicted_pieces"] = predicted_pieces
    return name, row, col

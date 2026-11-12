import numpy as np
import heapq
import time
from collections import deque

N = []
for p in range(256):
    y, x = divmod(p, 16)
    N.append((
        p - 16 if y else -1,
        p + 16 if y < 15 else -1,
        p - 1 if x else -1,
        p + 1 if x < 15 else -1,
    ))

ORDER = (2, 3, 0, 1)
OPP = (1, 0, 3, 2)
INF = 10000

PAT = []
for flags in range(16):
    bits = 0
    for y in range(4):
        for x in range(4):
            if ((flags & 1 and x == 0)
                    or (flags & 2 and x == 3)
                    or (flags & 4 and y == 0)
                    or (flags & 8 and y == 3)):
                bits |= 1 << (4 * y + x)
    PAT.append(bits)


def md(a, b):
    if a < 0 or b < 0:
        return 100
    ay, ax = divmod(a, 16)
    by, bx = divmod(b, 16)
    return abs(ax - bx) + abs(ay - by)


def direction(a, b):
    if a >= 0:
        for d in range(4):
            if N[a][d] == b:
                return d
    return -1


def assign_positions(old, observed):
    result = [-1] * len(old)
    remaining = set(observed)
    for i, p in enumerate(old):
        if p >= 0 and p in remaining:
            result[i] = p
            remaining.remove(p)

    unfilled = [i for i in range(len(old)) if result[i] < 0]
    pairs = sorted(
        (md(old[i], p), i, p)
        for i in unfilled for p in remaining
    )
    used = set()
    for _, i, p in pairs:
        if i not in used and p in remaining:
            result[i] = p
            used.add(i)
            remaining.remove(p)
    return result


def infer_goals(known):
    goals = set()
    seeds = {
        p for p, v in enumerate(known)
        if v is not None and v[0] > 0
    }

    while seeds:
        anchor = min(seeds)
        ay, ax = divmod(anchor, 16)
        av, am = known[anchor]
        best = None
        best_score = -1e30

        for y0 in range(ay + 1):
            for y1 in range(ay + 1, 17):
                yf = ((4 if ay == y0 else 0)
                      | (8 if ay == y1 - 1 else 0))
                for x0 in range(ax + 1):
                    for x1 in range(ax + 1, 17):
                        flags = (yf
                                 | (1 if ax == x0 else 0)
                                 | (2 if ax == x1 - 1 else 0))
                        if (PAT[flags] & am) != av:
                            continue

                        value = 0.0
                        valid = True
                        cells = []
                        for y in range(y0, y1):
                            if not valid:
                                break
                            fy = ((4 if y == y0 else 0)
                                  | (8 if y == y1 - 1 else 0))
                            for x in range(x0, x1):
                                p = y * 16 + x
                                cells.append(p)
                                v = known[p]
                                if v is None:
                                    value -= 1.0
                                    continue
                                if v[0] < 0:
                                    valid = False
                                    break
                                f = (fy
                                     | (1 if x == x0 else 0)
                                     | (2 if x == x1 - 1 else 0))
                                if (PAT[f] & v[1]) != v[0]:
                                    valid = False
                                    break
                                value += 1000.0 if v[0] else 1.0

                        if valid:
                            value -= 0.01 * len(cells)
                            if value > best_score:
                                best_score = value
                                best = cells

        if best is None:
            goals.add(anchor)
            seeds.remove(anchor)
        else:
            goals.update(best)
            seeds.difference_update(best)
    return goals


def read_board(obs, memory):
    old_known = memory.get("known", [None] * 256)
    known = list(old_known)
    walls = set(memory.get("walls", ()))
    traps = set(memory.get("traps", ()))

    boxes = {}
    friendly = []
    hostile = []
    player = None
    rot = None

    for p in range(256):
        y, x = divmod(p, 16)
        h = 3 if y == 15 else 4
        a = obs[4*y:4*y+h, 4*x:4*x+4]
        center = a[1:3, 1:3]

        if np.any(a == 14):
            player = p
            if np.all(a[0, :] == 0):
                rot = 0
            elif h == 4 and np.all(a[3, :] == 0):
                rot = 1
            elif np.all(a[:, 0] == 0):
                rot = 2
            elif np.all(a[:, 3] == 0):
                rot = 3
            continue

        if np.all(center == 12):
            friendly.append(p)
            continue

        if np.all(center == 15):
            hostile.append(p)
            continue

        if np.all(center == 9):
            edge = int(a[0, 0])
            if edge in (0, 3, 4, 5):
                boxes[p] = edge
                continue

        if np.all(a == 5):
            walls.add(p)
            known[p] = (-1, 0)
            continue

        if np.all((a == 2) | (a == 9)):
            bits = 0
            for yy in range(h):
                for xx in range(4):
                    if a[yy, xx] == 9:
                        bits |= 1 << (4*yy + xx)
            known[p] = (bits, (1 << (4*h)) - 1)
        elif np.any(a == 2) and np.all((a == 1) | (a == 2)):
            traps.add(p)
            known[p] = (-1, 0)
        elif np.all(a == 1):
            known[p] = (-1, 0)

    if player is None:
        return None

    old = memory.get("expected", memory.get("state"))
    kinds = memory.get("kinds")

    if old is None or kinds is None:
        pos = [player] + sorted(friendly) + sorted(hostile)
        kinds = [0] + [1] * len(friendly) + [2] * len(hostile)
        bp = sorted(boxes)
    else:
        pos = [-1] * len(kinds)
        pos[0] = player
        for kind, observations in ((1, friendly), (2, hostile)):
            indices = [i for i, k in enumerate(kinds) if k == kind]
            assigned = assign_positions(
                [old[0][i] for i in indices], observations
            )
            for i, p in zip(indices, assigned):
                pos[i] = p
        bp = assign_positions(old[3], boxes)
        if any(p < 0 for p in bp):
            bp = sorted(boxes)

    att = [-1] * len(pos)
    if rot is None:
        rot = old[1] if old is not None else 0

    claimed = set()
    for j, b in enumerate(bp):
        if boxes.get(b, 4) == 0 and direction(player, b) >= 0:
            att[0] = j
            claimed.add(0)

    for j, b in enumerate(bp):
        edge = boxes.get(b, 4)
        if edge in (0, 4):
            continue
        candidates = [
            i for i in range(1, len(pos))
            if pos[i] >= 0 and i not in claimed
            and direction(pos[i], b) >= 0
        ]

        preferred = None
        if old is not None:
            for i in candidates:
                if old[2][i] == j:
                    preferred = i
                    break

        if preferred is None and candidates:
            if edge == 5 or (old is None and edge == 3):
                preferred = min(
                    candidates, key=lambda i: (kinds[i] != 1, i)
                )

        if preferred is not None:
            att[preferred] = j
            claimed.add(preferred)

    if known != old_known or "goals" not in memory:
        goals = infer_goals(known)
    else:
        goals = set(memory["goals"])

    storage = {
        p for p, v in enumerate(known)
        if v is not None and v[0] == 0 and p not in goals
    }

    memory["known"] = known
    memory["walls"] = tuple(sorted(walls))
    memory["traps"] = tuple(sorted(traps))
    memory["goals"] = tuple(sorted(goals))
    memory["kinds"] = tuple(kinds)

    state = (tuple(pos), rot, tuple(att), tuple(bp))
    return state, kinds, walls, traps, goals, storage


class Model:
    def __init__(self, kinds, walls, traps, goals, storage,
                 deadline, actions):
        self.kinds = kinds
        self.walls = walls
        self.traps = traps
        self.goals = goals
        self.storage = storage
        self.deadline = deadline
        self.actions = actions
        self.cache = {}
        self.tables = None

    def expired(self):
        return time.monotonic() >= self.deadline

    def occupied(self, s):
        result = [False] * 256
        for p in self.walls:
            result[p] = True
        for p in s[0]:
            if p >= 0:
                result[p] = True
        for p in s[3]:
            if p >= 0:
                result[p] = True
        return result

    def valid(self, q, occupied, actor, box=-1, d=-1):
        if q < 0 or q in self.traps:
            return False
        if box < 0:
            return not occupied[q]
        if d < 0:
            return False
        b = N[q][d]
        return (
            b >= 0
            and (not occupied[q] or q == box)
            and (not occupied[b] or b == actor)
        )

    def win(self, s):
        return (
            all(b in self.goals for b in s[3])
            and all(j < 0 for j in s[2])
        )

    def first_step(self, s, actor_index, targets):
        pos, _, att, bp = s
        start = pos[actor_index]
        j = att[actor_index]
        box = bp[j] if j >= 0 else -1
        d = direction(start, box) if j >= 0 else -1

        if start < 0 or (j >= 0 and d < 0):
            return None
        if start in targets:
            return None

        occupied = self.occupied(s)
        seen = [False] * 256
        seen[start] = True
        first = [-1] * 256
        queue = deque([start])

        while queue:
            p = queue.popleft()
            for move in ORDER:
                q = N[p][move]
                if q < 0 or seen[q]:
                    continue
                if not self.valid(q, occupied, start, box, d):
                    continue
                seen[q] = True
                first[q] = q if p == start else first[p]
                if q in targets:
                    return first[q]
                queue.append(q)
        return None

    def carrying_can_deliver(self, s, i):
        j = s[2][i]
        if j < 0:
            return False
        if s[3][j] in self.goals:
            return True
        d = direction(s[0][i], s[3][j])
        if d < 0:
            return False
        targets = {
            N[b][OPP[d]]
            for b in self.goals
            if N[b][OPP[d]] >= 0
        }
        return self.first_step(s, i, targets) is not None

    def step(self, s, action):
        key = (s, action)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        pos = list(s[0])
        rot = s[1]
        att = list(s[2])
        bp = list(s[3])

        def state():
            return (tuple(pos), rot, tuple(att), tuple(bp))

        def owner(j):
            for i, a in enumerate(att):
                if a == j:
                    return i
            return -1

        def grab(i, j):
            previous = owner(j)
            if previous >= 0:
                att[previous] = -1
            att[i] = j

        def move(i, q):
            if q is None or q < 0:
                return
            occupied = self.occupied(state())
            j = att[i]
            box = bp[j] if j >= 0 else -1
            d = direction(pos[i], box) if j >= 0 else -1
            if j >= 0 and d < 0:
                return
            if self.valid(q, occupied, pos[i], box, d):
                pos[i] = q
                if j >= 0:
                    bp[j] = N[q][d]

        if action <= 4:
            d = action - 1
            if att[0] < 0:
                rot = d
            move(0, N[pos[0]][d])
        else:
            if att[0] >= 0:
                att[0] = -1
            else:
                front = N[pos[0]][rot]
                for j, b in enumerate(bp):
                    if b == front:
                        grab(0, j)
                        break
                for i in range(1, len(pos)):
                    if self.kinds[i] == 2 and pos[i] == front:
                        att[i] = -1
                        pos[i] = -1
                        break

        for kind in (1, 2):
            region = self.goals if kind == 1 else self.storage
            for i in range(1, len(pos)):
                if self.kinds[i] != kind or pos[i] < 0:
                    continue

                j = att[i]
                if j >= 0:
                    if bp[j] in region:
                        att[i] = -1
                        continue
                    d = direction(pos[i], bp[j])
                    if d < 0:
                        continue
                    targets = {
                        N[b][OPP[d]]
                        for b in region if N[b][OPP[d]] >= 0
                    }
                    q = self.first_step(state(), i, targets)
                    move(i, q)
                else:
                    eligible = []
                    for j, b in enumerate(bp):
                        o = owner(j)
                        if kind == 1:
                            ok = o < 0 and b not in self.goals
                        else:
                            ok = (
                                (o < 0 or self.kinds[o] != 2)
                                and b not in self.storage
                            )
                        if ok:
                            eligible.append(j)

                    picked = False
                    for j in eligible:
                        if direction(pos[i], bp[j]) >= 0:
                            grab(i, j)
                            picked = True
                            break
                    if picked:
                        break

                    targets = set()
                    for j in eligible:
                        targets.update(q for q in N[bp[j]] if q >= 0)
                    if targets:
                        q = self.first_step(state(), i, targets)
                        move(i, q)

        result = state()
        if len(self.cache) < 30000:
            self.cache[key] = result
        return result

    def paths(self, s, carrying=False):
        start = s[0][0]
        j = s[2][0] if carrying else -1
        box = s[3][j] if j >= 0 else -1
        d = direction(start, box) if j >= 0 else -1
        occupied = self.occupied(s)

        distance = [-1] * 256
        parent = [None] * 256
        distance[start] = 0
        queue = deque([start])

        if j >= 0 and d < 0:
            return distance, parent

        while queue:
            p = queue.popleft()
            for move in ORDER:
                q = N[p][move]
                if q < 0 or distance[q] >= 0:
                    continue
                if self.valid(q, occupied, start, box, d):
                    distance[q] = distance[p] + 1
                    parent[q] = (p, move + 1)
                    queue.append(q)
        return distance, parent

    def path(self, parents, start, end):
        result = []
        p = end
        while p != start:
            entry = parents[p]
            if entry is None:
                return None
            p, a = entry
            result.append(a)
        result.reverse()
        return result

    def quiet(self, s):
        occupied = self.occupied(s)
        j = s[2][0]
        box = s[3][j] if j >= 0 else -1
        d = direction(s[0][0], box) if j >= 0 else -1

        for a in self.actions:
            if a <= 4 and not self.valid(
                N[s[0][0]][a-1], occupied, s[0][0], box, d
            ):
                return a

        if j < 0 and 5 in self.actions:
            front = N[s[0][0]][s[1]]
            if front not in s[3] and front not in s[0][1:]:
                return 5
        return self.actions[0]

    def roll(self, s, plan):
        executed = []
        for a in plan:
            if self.expired():
                break
            s = self.step(s, a)
            executed.append(a)
            if self.win(s):
                break
        return s, executed

    def wait(self, s, n):
        plan = []
        for _ in range(n):
            if self.expired() or self.win(s):
                break
            a = self.quiet(s)
            s = self.step(s, a)
            plan.append(a)
        return s, plan


def static_tables(model, s, memory):
    free = [
        p not in model.walls and p not in model.traps
        for p in range(256)
    ]
    component = [-1] * 256
    number = 0

    for start in range(256):
        if not free[start] or component[start] >= 0:
            continue
        component[start] = number
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q in N[p]:
                if q >= 0 and free[q] and component[q] < 0:
                    component[q] = number
                    queue.append(q)
        number += 1

    pc = component[s[0][0]]
    fc = {
        component[s[0][i]]
        for i, k in enumerate(model.kinds)
        if k == 1 and s[0][i] >= 0
    }

    signature = (
        tuple(sorted(model.walls)),
        tuple(sorted(model.traps)),
        tuple(sorted(model.goals)),
        pc,
        tuple(sorted(fc)),
    )
    stored = memory.get("tables")
    if stored is not None and stored[0] == signature:
        model.tables = stored[1]
        return

    distances = []
    for d in range(4):
        valid = [
            free[q] and N[q][d] >= 0 and N[q][d] not in model.walls
            for q in range(256)
        ]
        dist = [INF] * 256
        queue = deque()
        for q in range(256):
            if valid[q] and N[q][d] in model.goals:
                dist[q] = 0
                queue.append(q)

        while queue:
            p = queue.popleft()
            for q in N[p]:
                if q >= 0 and valid[q] and dist[q] == INF:
                    dist[q] = dist[p] + 1
                    queue.append(q)
        distances.append(dist)

    friend = [INF] * 256
    for b in range(256):
        for d in range(4):
            q = N[b][OPP[d]]
            if q >= 0 and component[q] in fc:
                friend[b] = min(friend[b], distances[d][q] + 2)

    relay = []
    for d in range(4):
        valid = [
            component[q] == pc and free[q]
            and N[q][d] >= 0 and N[q][d] not in model.walls
            for q in range(256)
        ]
        dist = [INF] * 256
        heap = []
        for q in range(256):
            if not valid[q]:
                continue
            b = N[q][d]
            value = 0 if b in model.goals else friend[b] + 2
            if value < INF:
                dist[q] = value
                heapq.heappush(heap, (value, q))

        while heap:
            cost, p = heapq.heappop(heap)
            if cost != dist[p]:
                continue
            for q in N[p]:
                if q >= 0 and valid[q] and cost + 1 < dist[q]:
                    dist[q] = cost + 1
                    heapq.heappush(heap, (cost + 1, q))
        relay.append(dist)

    potential = list(friend)
    for b in range(256):
        if b in model.goals:
            potential[b] = 0
        for d in range(4):
            q = N[b][OPP[d]]
            if q >= 0 and component[q] == pc:
                potential[b] = min(potential[b], relay[d][q] + 2)

    model.tables = (component, distances, friend, relay, potential)
    memory["tables"] = (signature, model.tables)


def score(model, s, check_blocked=True):
    _, distances, _, _, potential = model.tables
    owners = [-1] * len(s[3])
    for i, j in enumerate(s[2]):
        if j >= 0:
            owners[j] = i

    total = 0.0
    for j, b in enumerate(s[3]):
        i = owners[j]
        if b in model.goals and i < 0:
            continue

        total += 12.0 + 0.55 * min(potential[b], 90)
        if i < 0:
            continue

        d = direction(s[0][i], b)
        kind = model.kinds[i]
        if kind == 1:
            if d < 0 or distances[d][s[0][i]] >= INF:
                total += 24.0
            elif (check_blocked and b not in model.goals
                  and not model.carrying_can_deliver(s, i)):
                total += 14.0
        elif kind == 2:
            total += 12.0
        elif b not in model.goals:
            total += 1.0
    return total


def remove_enemy(model, start):
    component = model.tables[0]
    candidates = [
        i for i, kind in enumerate(model.kinds)
        if kind == 2 and start[0][i] >= 0
        and component[start[0][i]] == component[start[0][0]]
    ]
    if not candidates or start[2][0] >= 0:
        return None

    target = min(candidates, key=lambda i: md(start[0][0], start[0][i]))
    distance_cache = {}

    def heuristic(s):
        e = s[0][target]
        if e < 0:
            return 0
        if e not in distance_cache:
            dist = [INF] * 256
            queue = deque()
            for q in N[e]:
                if q >= 0 and q not in model.walls and q not in model.traps:
                    dist[q] = 0
                    queue.append(q)
            while queue:
                p = queue.popleft()
                for q in N[p]:
                    if (q >= 0 and q not in model.walls
                            and q not in model.traps and dist[q] == INF):
                        dist[q] = dist[p] + 1
                        queue.append(q)
            distance_cache[e] = dist

        h = distance_cache[e][s[0][0]] + 1
        if h == 1 and N[s[0][0]][s[1]] != e:
            h += 1
        if s[2][0] >= 0:
            h += 3
        return h

    h0 = heuristic(start)
    heap = [(3.0*h0, 0, 0, start)]
    costs = {start: 0}
    parent = {}
    counter = 0
    best = start
    best_rank = (h0, 0)
    expansions = 0

    while heap and expansions < 2600 and not model.expired():
        _, g, _, s = heapq.heappop(heap)
        if costs.get(s) != g:
            continue
        if s[0][target] < 0 or model.win(s):
            best = s
            break
        if g >= 44:
            continue

        expansions += 1
        for a in model.actions:
            t = model.step(s, a)
            ng = g + 1
            if ng >= costs.get(t, INF):
                continue
            costs[t] = ng
            parent[t] = (s, a)
            h = heuristic(t)
            rank = (h, ng)
            if rank < best_rank:
                best_rank = rank
                best = t
            counter += 1
            heapq.heappush(heap, (ng + 3.0*h, ng, counter, t))

    if best == start:
        return None

    plan = []
    s = best
    while s != start:
        s, a = parent[s]
        plan.append(a)
    plan.reverse()

    if best[0][target] >= 0 and not model.win(best):
        plan = plan[:3]
    return plan


def delivery_candidates(model, start, prefix, exclude=None):
    j = start[2][0]
    if j < 0:
        return []

    d = direction(start[0][0], start[3][j])
    if d < 0:
        return []

    if start[3][j] in model.goals and 5 in model.actions:
        s = model.step(start, 5)
        return [(s, list(prefix) + [5])]

    distance, parents = model.paths(start, True)
    _, _, friend, _, _ = model.tables
    friendly_positions = [
        start[0][i] for i, k in enumerate(model.kinds)
        if k == 1 and start[0][i] >= 0
    ]

    goals = []
    stages = []
    for q in range(256):
        if distance[q] < 0:
            continue
        b = N[q][d]
        if b < 0 or b == exclude:
            continue

        if b in model.goals:
            degree = sum(n in model.goals for n in N[b] if n >= 0)
            goals.append((distance[q] + 0.3*degree, q))
        elif friend[b] < INF and friendly_positions:
            proximity = min(md(c, b) for c in friendly_positions)
            stages.append((
                distance[q] + 0.65*friend[b] + 0.3*proximity + 3,
                q,
            ))

    goals.sort()
    stages.sort()
    endpoints = [q for _, q in goals[:3]]
    endpoints += [q for _, q in stages[:3]]

    result = []
    seen_endpoints = set()
    for endpoint in endpoints:
        if endpoint in seen_endpoints:
            continue
        seen_endpoints.add(endpoint)
        if model.expired():
            break

        path = model.path(parents, start[0][0], endpoint)
        if path is None:
            continue
        s, executed = model.roll(start, path)
        plan = list(prefix) + executed

        if model.win(s):
            result.append((s, plan))
            continue

        retries = 0
        while (s[2][0] == j and s[0][0] != endpoint
               and retries < 4 and not model.expired()):
            ds, ps = model.paths(s, True)
            if ds[endpoint] < 0:
                break
            route = model.path(ps, s[0][0], endpoint)
            if not route:
                break
            t = model.step(s, route[0])
            if t == s:
                break
            s = t
            plan.append(route[0])
            retries += 1

        if s[2][0] >= 0 and 5 in model.actions:
            s = model.step(s, 5)
            plan.append(5)
        result.append((s, plan))

    return result


def plan_boxes(model, start, stalled):
    initial_score = score(model, start)
    ws, wp = model.wait(start, 8)
    if model.win(ws):
        return wp

    best_plan = wp[:3]
    best_value = score(model, ws) + 0.20 * len(wp)
    moving_plan = None
    moving_value = INF

    def consider(s, plan, forced=False):
        nonlocal best_plan, best_value, moving_plan, moving_value
        if not plan:
            return

        projected, virtual = model.wait(s, 6)
        value = score(model, projected) + 0.20 * len(plan)
        if model.win(s):
            value = -1000.0 + 0.01 * len(plan)
        elif model.win(projected):
            value = 0.01 * len(plan) + 0.015 * len(virtual)

        if value < best_value:
            best_value = value
            best_plan = plan
        if forced or s[3] != start[3]:
            if value < moving_value:
                moving_value = value
                moving_plan = plan

    if start[2][0] >= 0:
        j = start[2][0]
        if start[3][j] in model.goals and 5 in model.actions:
            return [5]

        deliveries = delivery_candidates(model, start, [])
        for s, plan in deliveries:
            consider(s, plan, True)

        if moving_plan is not None:
            return moving_plan
        return [5] if 5 in model.actions else best_plan

    for a in model.actions:
        if a > 4 or model.expired():
            continue
        s, plan = model.roll(start, [a, a])
        consider(s, plan)

    distance, parents = model.paths(start, False)
    _, distances, friend, relay, potential = model.tables
    owners = [-1] * len(start[3])
    for i, j in enumerate(start[2]):
        if j >= 0:
            owners[j] = i

    tasks = []
    for j, b in enumerate(start[3]):
        if b in model.goals and owners[j] < 0 and stalled < 8:
            continue

        i = owners[j]
        autonomous = (
            i >= 0 and model.kinds[i] == 1
            and model.carrying_can_deliver(start, i)
        )

        for d in range(4):
            q = N[b][OPP[d]]
            if q < 0 or distance[q] < 0:
                continue
            if relay[d][q] >= INF and distances[d][q] >= INF:
                continue

            priority = distance[q] + 0.3 * min(potential[b], 80)
            if autonomous:
                priority += 18
            if friend[b] >= INF:
                priority -= 8
            if b in model.goals:
                priority += 25
            tasks.append((priority, j, d, q))

    tasks.sort()
    for _, j, d, q in tasks[:28]:
        if model.expired():
            break

        route = model.path(parents, start[0][0], q)
        if route is None:
            continue

        arrived, executed = model.roll(start, route)
        if model.win(arrived):
            consider(arrived, executed)
            continue

        tail = []
        front = N[arrived[0][0]][arrived[1]]
        desired = N[arrived[0][0]][d]
        if front != desired:
            tail.append(d + 1)
        tail.append(5)
        if any(a not in model.actions for a in tail):
            continue

        s, extra = model.roll(arrived, tail)
        prefix = executed + extra
        if model.win(s):
            consider(s, prefix)
            continue
        if s[2][0] < 0:
            continue

        exclude = start[3][j] if start[3][j] in model.goals else None
        for t, plan in delivery_candidates(model, s, prefix, exclude):
            consider(t, plan)

    if stalled >= 7 and best_value >= initial_score - 0.5:
        if moving_plan is not None:
            return moving_plan
    return best_plan


def physical_state(s):
    # Rotation-only changes do not represent useful deadlock progress.
    return s[0], s[2], s[3]


def restart(memory):
    count = memory.get("resets", 0) + 1
    memory.clear()
    memory["resets"] = count
    return "RESET", None, None


def act(obs, available, memory):
    begun = time.monotonic()
    actions = [
        i for i in range(1, 6)
        if "ACTION" + str(i) in available
    ]
    if not actions:
        name = "RESET" if "RESET" in available else available[0]
        return name, None, None

    parsed = read_board(obs, memory)
    if parsed is None:
        return "ACTION" + str(actions[0]), None, None

    state, kinds, walls, traps, goals, storage = parsed
    model = Model(
        kinds, walls, traps, goals, storage,
        begun + 7.6, actions,
    )
    static_tables(model, state, memory)

    progress = (
        state[3], state[2],
        tuple(state[0][i] for i, k in enumerate(kinds) if k == 1),
    )
    if progress == memory.get("progress"):
        memory["stalled"] = memory.get("stalled", 0) + 1
    else:
        memory["stalled"] = 0
    memory["progress"] = progress

    physical = physical_state(state)
    if physical == memory.get("last_physical"):
        memory["immobile"] = memory.get("immobile", 0) + 1
    else:
        memory["immobile"] = 0
    memory["last_physical"] = physical

    signature = (
        tuple(sorted(walls)),
        tuple(sorted(traps)),
        tuple(sorted(goals)),
        tuple(sorted(storage)),
    )
    plan = memory.get("plan", [])
    if (memory.get("expected") != state
            or memory.get("plan_map") != signature):
        plan = []

    if not plan:
        plan = remove_enemy(model, state)
        if not plan and not model.expired():
            plan = plan_boxes(model, state, memory.get("stalled", 0))
        if not plan:
            plan = [model.quiet(state)]

    # Restart a genuine stationary deadlock early, leaving enough time
    # for a fresh attempt. Productive autonomous waiting does not trigger
    # this because all actor positions participate in the test.
    if (memory.get("immobile", 0) >= 6
            and memory.get("resets", 0) == 0
            and "RESET" in available
            and not model.expired()):
        trial, executed = model.roll(state, plan[:8])
        future, waiting = model.wait(trial, 8)
        if (not model.win(trial)
                and not model.win(future)
                and physical_state(trial) == physical
                and physical_state(future) == physical):
            return restart(memory)

    # Also recover once from a nearly exhausted attempt that cannot
    # finish with its current plan.
    bar = int(np.count_nonzero(obs[63] == 7))
    if (bar <= 1 and "RESET" in available
            and memory.get("resets", 0) == 0
            and not model.expired()):
        trial, executed = model.roll(state, plan[:12])
        if not model.win(trial):
            trial, waiting = model.wait(trial, 3)
        if not model.win(trial):
            return restart(memory)

    action = plan[0]
    if action not in actions:
        action = model.quiet(state)
        plan = [action]

    memory["state"] = state
    memory["expected"] = model.step(state, action)
    memory["plan"] = list(plan[1:])
    memory["plan_map"] = signature
    return "ACTION" + str(action), None, None

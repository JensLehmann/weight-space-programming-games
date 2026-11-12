import numpy as np
import heapq
import time
from collections import deque

# Directions correspond to ACTION1 ... ACTION4.
_DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ORDER = (2, 3, 0, 1)
_NB = []
for _p in range(256):
    _x, _y = _p % 16, _p // 16
    _NB.append(tuple(
        (_y + dy) * 16 + _x + dx
        if 0 <= _x + dx < 16 and 0 <= _y + dy < 16 else -1
        for dx, dy in _DXY
    ))


def _md(a, b):
    return abs(a % 16 - b % 16) + abs(a // 16 - b // 16)


def _direction(a, b):
    if a < 0 or b < 0:
        return -1
    for d in range(4):
        if _NB[a][d] == b:
            return d
    return -1


def _match(predicted, observed, previous=None):
    """Preserve object identities, favoring exact predicted matches."""
    out = [-1] * len(predicted)
    unused = set(observed)
    for i, p in enumerate(predicted):
        if p >= 0 and p in unused:
            out[i] = p
            unused.remove(p)
    if previous is not None:
        for i, p in enumerate(previous):
            if out[i] < 0 and p >= 0 and p in unused:
                out[i] = p
                unused.remove(p)
    pairs = []
    for i, p in enumerate(predicted):
        if out[i] >= 0:
            continue
        ref = p
        if ref < 0 and previous is not None:
            ref = previous[i]
        for q in unused:
            pairs.append((_md(ref, q) if ref >= 0 else 100, i, q))
    for _, i, q in sorted(pairs):
        if out[i] < 0 and q in unused:
            out[i] = q
            unused.remove(q)
    return out


def _terrain(tiles):
    walls = {p for p, v in enumerate(tiles) if v == 1}
    fences = {p for p, v in enumerate(tiles) if v == 2}
    goals = set()

    # Fit the visible portions of rectangular, outlined goal regions.
    seeds = [p for p, v in enumerate(tiles) if v >= 17]
    for p in seeds:
        if p in goals:
            continue
        x, y = p % 16, p // 16
        bits = tiles[p] - 16
        ls = [x] if bits & 1 else range(x - 1, -1, -1)
        rs = [x] if bits & 2 else range(x + 1, 16)
        ts = [y] if bits & 4 else range(y - 1, -1, -1)
        bs = [y] if bits & 8 or y == 15 else range(y + 1, 16)
        candidates = []
        for l in ls:
            for r in rs:
                for t in ts:
                    for b in bs:
                        candidates.append(((r-l+1)*(b-t+1), l, r, t, b))
        candidates.sort()
        for _, l, r, t, b in candidates:
            good = True
            rect = []
            for yy in range(t, b + 1):
                for xx in range(l, r + 1):
                    q = yy * 16 + xx
                    v = tiles[q]
                    if v == -1:
                        rect.append(q)
                        continue
                    if v < 16:
                        good = False
                        break
                    expected = ((1 if xx == l else 0) |
                                (2 if xx == r else 0) |
                                (4 if yy == t else 0) |
                                (8 if yy == b else 0))
                    mask = 7 if yy == 15 else 15
                    if (expected & mask) != ((v - 16) & mask):
                        good = False
                        break
                    rect.append(q)
                if not good:
                    break
            if good:
                goals.update(rect)
                break

    # Unoutlined red regions are the hostile agents' destinations.
    red = {p for p, v in enumerate(tiles) if v == 16 and p not in goals}
    bad = set(red)
    unseen = set(red)
    while unseen:
        start = unseen.pop()
        comp = {start}
        queue = [start]
        for p in queue:
            for q in _NB[p]:
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        xs = [p % 16 for p in comp]
        ys = [p // 16 for p in comp]
        for y in range(min(ys), max(ys) + 1):
            for x in range(min(xs), max(xs) + 1):
                p = y * 16 + x
                if p not in goals and tiles[p] in (-1, 16):
                    bad.add(p)
    return walls, fences, goals, bad


def _read(obs, memory):
    tiles = list(memory.get("tiles", [-1] * 256))
    boxes, good, evil = [], [], []
    border = {}
    player, rotation = None, 0

    for y in range(16):
        for x in range(16):
            p = 16*y + x
            tile = obs[4*y:min(4*y+4, 63), 4*x:4*x+4]
            a = tile.tolist()
            flat = [v for row in a for v in row]
            center = int(obs[4*y+1, 4*x+1])
            if 14 in flat:
                player = p
                if all(v == 0 for v in a[0]):
                    rotation = 0
                elif all(row[0] == 0 for row in a):
                    rotation = 2
                elif all(row[-1] == 0 for row in a):
                    rotation = 3
                else:
                    rotation = 1
                continue
            if center == 12:
                good.append(p)
                continue
            if center == 15:
                evil.append(p)
                continue
            if center == 9 and int(obs[4*y+2, 4*x+2]) == 9:
                boxes.append(p)
                border[p] = int(a[0][0])
                continue
            if all(v == 5 for v in flat):
                tiles[p] = 1
            elif all(v in (2, 9) for v in flat):
                bits = 0
                if all(row[0] == 9 for row in a):
                    bits |= 1
                if all(row[-1] == 9 for row in a):
                    bits |= 2
                if all(v == 9 for v in a[0]):
                    bits |= 4
                if y != 15 and all(v == 9 for v in a[-1]):
                    bits |= 8
                tiles[p] = 16 + bits
            elif 2 in flat:
                tiles[p] = 2
            else:
                tiles[p] = 0

    if player is None:
        return None

    old = memory.get("state")
    pred = memory.get("pred", old)
    types = memory.get("types")
    fresh = (pred is None or types is None or
             len(pred[2]) != len(boxes) or
             types.count("g") != len(good) or
             types.count("e") < len(evil))

    if fresh:
        types = ["p"] + ["g"] * len(good) + ["e"] * len(evil)
        positions = [player] + good + evil
        bp = boxes
        predicted_holds = [-1] * len(positions)
        memory["order"] = list(range(len(boxes)))
        memory.pop("todo", None)
    else:
        pp, ph, pb, _ = pred
        op = old[0] if old is not None else pp
        ob = old[2] if old is not None else pb
        positions = [player] + [-1] * (len(types) - 1)
        for kind, seen in (("g", good), ("e", evil)):
            ids = [i for i, t in enumerate(types) if t == kind]
            matched = _match([pp[i] for i in ids], seen, [op[i] for i in ids])
            for i, q in zip(ids, matched):
                positions[i] = q
        bp = _match(pb, boxes, ob)
        predicted_holds = list(ph)

    walls, fences, goals, bad = _terrain(tiles)
    holds = [-1] * len(positions)
    uncertain = []

    for b, p in enumerate(bp):
        c = border.get(p, 4)
        if c == 0:
            holds[0] = b
        elif c in (3, 5):
            uncertain.append((c != 5, b, c))
    uncertain.sort()

    for _, b, c in uncertain:
        p = bp[b]
        candidates = [
            i for i in range(1, len(positions))
            if positions[i] >= 0 and holds[i] < 0
            and _direction(positions[i], p) >= 0
        ]
        predicted = [i for i in candidates if predicted_holds[i] == b]
        if predicted:
            holds[predicted[0]] = b
        elif candidates and c == 5:
            # Hostile agents act after friendly ones and can steal cargo.
            candidates.sort(key=lambda i: types[i] != "e")
            holds[candidates[0]] = b
        elif fresh and candidates and c == 3:
            eligible = [
                i for i in candidates
                if (types[i] == "g" and p not in goals) or
                   (types[i] == "e" and p not in bad)
            ]
            if eligible:
                holds[eligible[-1]] = b

    state = (tuple(positions), tuple(holds), tuple(bp), rotation)
    signature = (tuple(sorted(walls)), tuple(sorted(fences)),
                 tuple(sorted(goals)), tuple(sorted(bad)))

    # Learn otherwise invisible tie-breaking preferences when possible.
    order = memory.get("order", list(range(len(bp))))
    if not fresh:
        for i in range(1, len(holds)):
            actual, expected = holds[i], predicted_holds[i]
            if actual >= 0 and expected >= 0 and actual != expected:
                if actual in order and expected in order:
                    order.remove(actual)
                    order.insert(order.index(expected), actual)

    consistent = (not fresh and state == pred and
                  signature == memory.get("signature"))
    memory["tiles"] = tiles
    memory["types"] = types
    memory["order"] = order
    memory["signature"] = signature
    memory["state"] = state
    return state, types, walls, fences, goals, bad, consistent


class _World:
    def __init__(self, types, walls, fences, goals, bad, order, deadline):
        self.types = types
        self.walls = walls
        self.fences = fences
        self.goals = goals
        self.bad = bad
        self.order = order
        self.deadline = deadline
        self.cache = {}
        self.attack_distances = {}

    def solved(self, s):
        return all(p in self.goals for p in s[2]) and all(b < 0 for b in s[1])

    def _first(self, start, occupied, targets, offset=-1, cargo=-1):
        """The agents' left/right/up/down BFS, returning its first move."""
        queue = deque([(start, -1)])
        seen = {start}
        while queue:
            p, first = queue.popleft()
            target = p if offset < 0 else _NB[p][offset]
            if target in targets:
                return first
            for d in _ORDER:
                q = _NB[p][d]
                if q < 0 or q in seen or q in self.fences:
                    continue
                if q in occupied and q != cargo:
                    continue
                if offset >= 0:
                    c = _NB[q][offset]
                    if c < 0 or (c in occupied and c != start):
                        continue
                seen.add(q)
                queue.append((q, d if first < 0 else first))
        return -1

    def step(self, s, action):
        key = (s, action)
        if key in self.cache:
            return self.cache[key]
        pp, hh, bb, rot = s
        pos, holds, boxes = list(pp), list(hh), list(bb)

        def occupied():
            return self.walls | {p for p in pos if p >= 0} | set(boxes)

        def move(i, d):
            old = pos[i]
            q = _NB[old][d]
            if q < 0 or q in self.fences:
                return
            occ = occupied()
            b = holds[i]
            if b < 0:
                if q not in occ:
                    pos[i] = q
            else:
                oldb = boxes[b]
                nb = _NB[oldb][d]
                if nb < 0:
                    return
                if (q not in occ or q == oldb) and (nb not in occ or nb == old):
                    pos[i], boxes[b] = q, nb

        def attach(i, b):
            for j in range(len(holds)):
                if holds[j] == b:
                    holds[j] = -1
            holds[i] = b

        if 1 <= action <= 4:
            d = action - 1
            if holds[0] < 0:
                rot = d
            move(0, d)
        elif action == 5:
            if holds[0] >= 0:
                holds[0] = -1
            else:
                front = _NB[pos[0]][rot]
                for b in self.order:
                    if boxes[b] == front:
                        attach(0, b)
                        break
                for i, t in enumerate(self.types):
                    if t == "e" and pos[i] == front:
                        holds[i] = -1
                        pos[i] = -1
                        break

        for kind, destinations in (("g", self.goals), ("e", self.bad)):
            for i, typ in enumerate(self.types):
                if typ != kind or pos[i] < 0:
                    continue
                b = holds[i]
                if b >= 0:
                    if boxes[b] in destinations:
                        holds[i] = -1
                    else:
                        off = _direction(pos[i], boxes[b])
                        if off >= 0:
                            d = self._first(pos[i], occupied(), destinations,
                                            off, boxes[b])
                            if d >= 0:
                                move(i, d)
                    continue

                owners = [-1] * len(boxes)
                for j, cargo in enumerate(holds):
                    if cargo >= 0:
                        owners[cargo] = j
                eligible = []
                for cargo in self.order:
                    own = owners[cargo]
                    allowed = (own < 0 if kind == "g"
                               else own < 0 or self.types[own] != "e")
                    if allowed and boxes[cargo] not in destinations:
                        eligible.append(cargo)

                grabbed = False
                for cargo in eligible:
                    if _direction(pos[i], boxes[cargo]) >= 0:
                        attach(i, cargo)
                        grabbed = True
                        break
                if grabbed:
                    # A grab ends this whole group of agents' update.
                    break

                targets = set()
                for cargo in eligible:
                    targets.update(q for q in _NB[boxes[cargo]] if q >= 0)
                if targets:
                    d = self._first(pos[i], occupied(), targets)
                    if d >= 0:
                        move(i, d)

        result = (tuple(pos), tuple(holds), tuple(boxes), rot)
        if len(self.cache) < 30000:
            self.cache[key] = result
        return result

    def idle_action(self, s):
        p, h, b, r = s
        if h[0] >= 0:
            return 5
        front = _NB[p[0]][r]
        # Walking into a box is a harmless wait; toggling would grab it.
        return r + 1 if front in b else 5

    def automatic(self, s, limit=115):
        path = []
        seen = set()
        for _ in range(limit):
            if self.solved(s) or s in seen:
                break
            if time.perf_counter() > self.deadline:
                break
            seen.add(s)
            a = self.idle_action(s)
            ns = self.step(s, a)
            path.append(a)
            if ns == s:
                s = ns
                break
            s = ns
        return s, path

    def tree(self, start, blocked, offset=-1):
        dist = [-1] * 256
        parent = [None] * 256
        dist[start] = 0
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for d in _ORDER:
                q = _NB[p][d]
                if q < 0 or dist[q] >= 0 or q in blocked or q in self.fences:
                    continue
                if offset >= 0:
                    c = _NB[q][offset]
                    if c < 0 or c in blocked:
                        continue
                dist[q] = dist[p] + 1
                parent[q] = (p, d + 1)
                queue.append(q)
        return dist, parent

    @staticmethod
    def path(parent, start, end):
        actions = []
        while end != start:
            link = parent[end]
            if link is None:
                return None
            end, a = link
            actions.append(a)
        actions.reverse()
        return actions

    def helper_costs(self, s, b):
        pos, holds, boxes, _ = s
        friends = [i for i, t in enumerate(self.types)
                   if t == "g" and pos[i] >= 0]
        if not friends or not self.goals:
            return [100000] * 256
        blocked = self.walls | {q for j, q in enumerate(boxes) if j != b}
        reach = []
        for i in friends:
            ds, _ = self.tree(pos[i], blocked)
            delay = 10 if holds[i] >= 0 and holds[i] != b else 0
            reach.append((ds, delay))
        costs = [100000] * 256
        for off in range(4):
            valid = [False] * 256
            distance = [-1] * 256
            queue = deque()
            for p in range(256):
                c = _NB[p][off]
                valid[p] = (p not in blocked and p not in self.fences
                            and c >= 0 and c not in blocked)
                if valid[p] and c in self.goals:
                    distance[p] = 0
                    queue.append(p)
            while queue:
                p = queue.popleft()
                for q in _NB[p]:
                    if q >= 0 and valid[q] and distance[q] < 0:
                        distance[q] = distance[p] + 1
                        queue.append(q)
            for p in range(256):
                if distance[p] < 0:
                    continue
                arrival = min((ds[p] + delay for ds, delay in reach
                               if ds[p] >= 0), default=100000)
                c = _NB[p][off]
                costs[c] = min(costs[c], arrival + distance[p] + 2)
        return costs

    def macros(self, s, include_settled=False):
        pos, holds, boxes, rot = s
        occupied = self.walls | set(boxes) | {p for p in pos if p >= 0}
        pd, pp = self.tree(pos[0], occupied)
        owners = {b: i for i, b in enumerate(holds) if b >= 0}
        candidates = []
        if holds[0] >= 0:
            candidates.append((1, holds[0], -1, [5]))

        for b, box in enumerate(boxes):
            if time.perf_counter() > self.deadline - 0.6:
                break
            if holds[0] >= 0 and holds[0] != b:
                continue
            if (box in self.goals and holds[0] != b and
                    not include_settled):
                continue
            owner = owners.get(b, -1)
            if owner > 0 and self.types[owner] == "g":
                off = _direction(pos[owner], box)
                if box in self.goals:
                    continue
                if off >= 0 and self._first(
                        pos[owner], occupied, self.goals, off, box) >= 0:
                    continue

            hc = self.helper_costs(s, b)
            blocked = self.walls | {q for j, q in enumerate(boxes) if j != b}
            for off in range(4):
                if holds[0] == b:
                    if _direction(pos[0], box) != off:
                        continue
                    start = pos[0]
                    pickup = []
                else:
                    opposite = (1, 0, 3, 2)[off]
                    start = _NB[box][opposite]
                    if start < 0 or pd[start] < 0:
                        continue
                    pickup = self.path(pp, pos[0], start)
                    if pickup is None:
                        continue
                    facing = pickup[-1] - 1 if pickup else rot
                    if facing != off:
                        pickup = pickup + [off + 1]
                    pickup = pickup + [5]

                ds, parent = self.tree(start, blocked, off)
                direct, transfer = [], []
                for p, d in enumerate(ds):
                    if d < 0:
                        continue
                    c = _NB[p][off]
                    if c < 0:
                        continue
                    if c in self.goals:
                        if include_settled and c == box and box in self.goals:
                            continue
                        packing = sum(q in self.goals for q in _NB[c])
                        direct.append((d - .12 * packing, p))
                    elif hc[c] < 100000:
                        transfer.append((d + .65 * hc[c], p))
                direct.sort()
                transfer.sort()
                choices = [(v, p, False) for v, p in direct[:3]]
                selected = []
                for value, p in transfer:
                    c = _NB[p][off]
                    if all(_md(c, _NB[q][off]) >= 2 for q in selected):
                        selected.append(p)
                        choices.append((value, p, True))
                        if len(selected) >= 4:
                            break

                for value, end, handoff in choices:
                    carry = self.path(parent, start, end)
                    if carry is None:
                        continue
                    seq = pickup + carry + [5]
                    if not seq or len(seq) > 110:
                        continue
                    estimate = len(pickup) + value + 1
                    candidates.append((estimate, b, off, seq))

        # Moving aside can unblock an otherwise self-sufficient helper.
        spaces = []
        for p, d in enumerate(pd):
            if 1 <= d <= 5 and p not in self.goals:
                clearance = min((_md(p, q) for q in boxes), default=10)
                spaces.append((d - .3 * clearance, p))
        for _, p in sorted(spaces)[:5]:
            seq = self.path(pp, pos[0], p)
            if seq:
                candidates.append((len(seq) + 15, -1, -1, seq))

        candidates.sort(key=lambda c: c[0])
        # Retain orientation and cargo diversity, not just similar routes.
        result, used, groups = [], set(), set()
        for entry in candidates:
            _, b, off, seq = entry
            group = (b, off)
            key = tuple(seq)
            if group not in groups and key not in used:
                result.append(entry)
                groups.add(group)
                used.add(key)
                if len(result) >= 24:
                    break
        for entry in candidates:
            key = tuple(entry[3])
            if key not in used:
                result.append(entry)
                used.add(key)
                if len(result) >= 40:
                    break
        return result

    def score(self, s, elapsed, automatic_steps):
        _, holds, boxes, _ = s
        carried = {b for b in holds if b >= 0}
        remaining = [i for i, p in enumerate(boxes)
                     if p not in self.goals or i in carried]
        if not remaining:
            return elapsed + automatic_steps
        free_goals = self.goals - {
            p for i, p in enumerate(boxes) if i not in remaining
        }
        targets = free_goals or self.goals
        distance = sum(
            min((_md(boxes[i], q) for q in targets), default=30)
            for i in remaining
        )
        return (1000 * len(remaining) + 2 * distance + elapsed
                + .35 * automatic_steps + (40 if holds[0] >= 0 else 0))

    def enemy_distance(self, player, enemy):
        if enemy not in self.attack_distances:
            distance = [-1] * 256
            queue = deque()
            for p in _NB[enemy]:
                if p >= 0 and p not in self.walls and p not in self.fences:
                    distance[p] = 0
                    queue.append(p)
            while queue:
                p = queue.popleft()
                for q in _NB[p]:
                    if (q >= 0 and distance[q] < 0 and
                            q not in self.walls and q not in self.fences):
                        distance[q] = distance[p] + 1
                        queue.append(q)
            self.attack_distances[enemy] = distance
        d = self.attack_distances[enemy][player]
        return 10000 if d < 0 else d

    def attack(self, s):
        ids = [i for i, t in enumerate(self.types)
               if t == "e" and s[0][i] >= 0]
        if not ids:
            return None

        def heuristic(q):
            p, h, _, r = q
            best = 10000
            for i in ids:
                if p[i] < 0:
                    return -100
                d = self.enemy_distance(p[0], p[i])
                if d == 0 and _NB[p[0]][r] != p[i]:
                    d += 1
                best = min(best, d)
            return best + (5 if h[0] >= 0 else 0)

        initial = heuristic(s)
        if initial >= 10000:
            return None
        beam = [(initial, s, [])]
        visited = {s}
        best_partial = None
        best_h = initial
        stop = min(self.deadline - 0.3, time.perf_counter() + 4.0)
        for depth in range(42):
            if time.perf_counter() > stop:
                break
            new = []
            for _, q, route in beam:
                p, h, b, r = q
                actions = [1, 2, 3, 4]
                if h[0] >= 0 or any(p[i] == _NB[p[0]][r] for i in ids):
                    actions.append(5)
                for a in actions:
                    ns = self.step(q, a)
                    if ns in visited:
                        continue
                    visited.add(ns)
                    path = route + [a]
                    if any(ns[0][i] < 0 for i in ids):
                        return path
                    value = heuristic(ns)
                    if value < best_h:
                        best_h, best_partial = value, path
                    new.append((value, ns, path))
            if not new:
                break
            new.sort(key=lambda item: item[0])
            beam = new[:24]
        return best_partial[:8] if best_partial else None

    def single_box_search(self, s):
        """Fallback allowing repeated releases and re-grasps in tight passages."""
        pos, holds, boxes, rotation = s
        if holds[0] >= 0:
            choices = [holds[0]]
        else:
            choices = sorted(
                (i for i, p in enumerate(boxes) if p not in self.goals),
                key=lambda i: _md(pos[0], boxes[i])
            )[:2]
        stop = min(self.deadline - .15, time.perf_counter() + 1.5)

        for target in choices:
            if time.perf_counter() >= stop:
                break
            blocked = self.walls | {
                p for i, p in enumerate(boxes) if i != target
            }
            start = (pos[0], boxes[target], holds[0] == target, rotation)
            parent = {start: None}
            costs = {start: 0}
            heap = []
            serial = 0

            def estimate(q):
                p, b, held, _ = q
                d = min((_md(b, g) for g in self.goals), default=40)
                return d + (1 if held else max(0, _md(p, b)-1) + 2)

            heapq.heappush(heap, (estimate(start), 0, serial, start))
            expanded = 0
            while heap and expanded < 18000:
                if expanded % 128 == 0 and time.perf_counter() >= stop:
                    break
                _, cost, _, q = heapq.heappop(heap)
                if costs.get(q) != cost:
                    continue
                expanded += 1
                p, b, held, r = q
                if b in self.goals and not held:
                    route = []
                    while parent[q] is not None:
                        q, a = parent[q]
                        route.append(a)
                    return route[::-1]

                successors = []
                for d in range(4):
                    np_ = _NB[p][d]
                    if held:
                        nb = _NB[b][d]
                        if (np_ >= 0 and nb >= 0 and
                                np_ not in blocked and np_ not in self.fences
                                and nb not in blocked):
                            successors.append(((np_, nb, True, r), d + 1))
                    else:
                        if (np_ < 0 or np_ in blocked or
                                np_ in self.fences or np_ == b):
                            np_ = p
                        successors.append(((np_, b, False, d), d + 1))
                if held:
                    successors.append(((p, b, False, r), 5))
                elif _NB[p][r] == b:
                    successors.append(((p, b, True, r), 5))

                for nq, a in successors:
                    nc = cost + 1
                    if nc >= costs.get(nq, 100000):
                        continue
                    costs[nq] = nc
                    parent[nq] = (q, a)
                    serial += 1
                    heapq.heappush(heap, (
                        nc + 1.15 * estimate(nq), nc, serial, nq
                    ))
        return None

    def plan(self, s):
        attack = self.attack(s)
        if attack:
            return attack
        if time.perf_counter() > self.deadline - .4:
            return [self.idle_action(s)]

        final, waiting = self.automatic(s)
        baseline = self.score(final, 0, len(waiting))
        best_score = baseline
        best = waiting[:6] if waiting else [self.idle_action(s)]
        macros = self.macros(s)

        for _, _, _, route in macros:
            if time.perf_counter() > self.deadline - .35:
                break
            q = s
            executed = []
            for a in route:
                q = self.step(q, a)
                executed.append(a)
                if self.solved(q):
                    break
            end, auto = self.automatic(q)
            value = self.score(end, len(executed), len(auto))
            if value < best_score - .01:
                best_score = value
                best = executed
                if self.solved(q):
                    best = executed

        # If nothing useful is happening, try rearranging a settled box.
        if (best_score >= baseline - .01 and len(waiting) <= 2
                and not self.solved(final)
                and time.perf_counter() < self.deadline - 1.0):
            if s[1][0] < 0:
                route = self.single_box_search(s)
                if route:
                    return route
            if time.perf_counter() < self.deadline - .6:
                for _, _, _, route in self.macros(s, include_settled=True)[:12]:
                    if time.perf_counter() > self.deadline - .2:
                        break
                    q = s
                    for a in route:
                        q = self.step(q, a)
                    end, auto = self.automatic(q)
                    value = self.score(end, len(route), len(auto))
                    if value < best_score - .01:
                        best_score, best = value, route
        return best or [self.idle_action(s)]


def act(obs, available, memory):
    started = time.perf_counter()
    parsed = _read(np.asarray(obs), memory)
    names = set(available)

    if parsed is None:
        action = "ACTION5" if "ACTION5" in names else next(
            (a for a in available if a != "RESET"), available[0]
        )
        return action, None, None

    state, types, walls, fences, goals, bad, consistent = parsed
    world = _World(
        types, walls, fences, goals, bad,
        memory.get("order", list(range(len(state[2])))),
        started + 8.0
    )

    todo = list(memory.get("todo", [])) if consistent else []
    if not todo:
        todo = world.plan(state)

    action_id = todo.pop(0) if todo else world.idle_action(state)
    action = "ACTION" + str(action_id)
    if action not in names:
        action = "ACTION5" if "ACTION5" in names else next(
            (a for a in available if a != "RESET"), available[0]
        )
        todo = []
        if action.startswith("ACTION"):
            action_id = int(action[6:])
        else:
            action_id = 5

    memory["todo"] = todo
    memory["pred"] = world.step(state, action_id)
    memory["state"] = state
    return action, None, None

import time
import heapq
from collections import deque

_D = (-18, 18, -1, 1)
_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
_CELLS = tuple((r + 1) * 18 + c + 1
               for r in range(16) for c in range(16))
_CELLSET = set(_CELLS)
_BORDER = set(range(324)) - _CELLSET
_NB = [tuple(p + d for d in _D) if p in _CELLSET else ()
       for p in range(324)]
_INF = 1000000


def _dist(a, b):
    if a < 0 or b < 0:
        return 1000
    return abs(a // 18 - b // 18) + abs(a % 18 - b % 18)


def _assignment(observed, predicted, previous):
    m, n = len(observed), len(predicted)
    if not m:
        return []
    if n < m:
        return list(range(m))
    cost = []
    for p in observed:
        cost.append([
            5 * _dist(p, q) +
            _dist(p, previous[j] if j < len(previous) else q)
            for j, q in enumerate(predicted)
        ])

    u = [0] * (m + 1)
    v = [0] * (n + 1)
    match = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, m + 1):
        match[0] = i
        j0 = 0
        mins = [_INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = match[j0]
            delta, j1 = _INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < mins[j]:
                        mins[j], way[j] = cur, j0
                    if mins[j] < delta:
                        delta, j1 = mins[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    mins[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while True:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
            if not j0:
                break

    answer = [-1] * m
    for j in range(1, n + 1):
        if match[j]:
            answer[match[j] - 1] = j - 1
    return answer


def _read(obs, memory):
    raw = [-2] * 256
    entities = {}
    found_boxes, found_helpers, found_enemies = [], [], []
    player, face = None, 0

    for r in range(16):
        for c in range(16):
            p = (r + 1) * 18 + c + 1
            k = r * 16 + c
            h = 3 if r == 15 else 4
            tile = obs[r * 4:r * 4 + h, c * 4:c * 4 + 4]
            center = [int(obs[r * 4 + y, c * 4 + x])
                      for y in (1, 2) for x in (1, 2)]
            vals = set(int(x) for x in tile.flat)

            if 14 in center:
                player = p
                if all(int(tile[0, x]) == 0 for x in range(4)):
                    face = 0
                elif all(int(tile[y, 3]) == 0 for y in range(h)):
                    face = 3
                elif all(int(tile[y, 0]) == 0 for y in range(h)):
                    face = 2
                else:
                    face = 1
                entities[p] = ("p", 0)
                raw[k] = None
            elif all(x == 9 for x in center):
                rim = int(tile[0, 0])
                found_boxes.append(p)
                entities[p] = ("b", rim)
                raw[k] = None
            elif all(x == 12 for x in center):
                found_helpers.append(p)
                entities[p] = ("h", 12)
                raw[k] = None
            elif all(x == 15 for x in center):
                found_enemies.append(p)
                entities[p] = ("e", 15)
                raw[k] = None
            elif vals == {5}:
                raw[k] = -3
            elif 2 in vals and 1 in vals:
                raw[k] = -4
            elif vals and vals <= {2, 9}:
                bits = 0
                if all(int(tile[0, x]) == 9 for x in range(4)):
                    bits |= 1
                if h == 4 and all(int(tile[3, x]) == 9 for x in range(4)):
                    bits |= 2
                if all(int(tile[y, 0]) == 9 for y in range(h)):
                    bits |= 4
                if all(int(tile[y, 3]) == 9 for y in range(h)):
                    bits |= 8
                raw[k] = bits
            else:
                raw[k] = -2

    candidates = []
    outlined = {k for k, code in enumerate(raw)
                if code is not None and code > 0}
    if outlined:
        for top in range(16):
            for bottom in range(top, 16):
                for left in range(16):
                    for right in range(left, 16):
                        evidence = 0
                        valid = True
                        for r in range(top, bottom + 1):
                            base = r * 16
                            vertical = ((1 if r == top else 0) |
                                        (2 if r == bottom else 0))
                            for c in range(left, right + 1):
                                code = raw[base + c]
                                if code is None:
                                    continue
                                expected = (vertical |
                                            (4 if c == left else 0) |
                                            (8 if c == right else 0))
                                mask = 13 if r == 15 else 15
                                if code < 0 or (code & mask) != (expected & mask):
                                    valid = False
                                    break
                                if code:
                                    evidence += 1
                            if not valid:
                                break
                        if valid and evidence:
                            area = (bottom - top + 1) * (right - left + 1)
                            candidates.append(
                                (-evidence, area, top, bottom, left, right))
        candidates.sort()

    goal_tiles = set()
    covered = set()
    for _, _, top, bottom, left, right in candidates:
        cells = {r * 16 + c
                 for r in range(top, bottom + 1)
                 for c in range(left, right + 1)}
        if (cells & outlined) - covered:
            goal_tiles.update(cells)
            covered.update(cells & outlined)

    terrain = list(memory.get("terrain", [0] * 256))
    known = set(memory.get("known", []))
    for k, code in enumerate(raw):
        if code is None:
            continue
        known.add(k)
        if code == -3:
            terrain[k] = 1
        elif code == -4:
            terrain[k] = 2
        elif k in goal_tiles:
            terrain[k] = 3
        elif code == 0:
            terrain[k] = 4
        elif code == -2:
            terrain[k] = 0
    for k in goal_tiles:
        terrain[k] = 3
        known.add(k)

    red = {k for k, code in enumerate(raw)
           if code == 0 and k not in goal_tiles}
    visited = set()
    for start in list(red):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        visible = []
        while queue:
            k = queue.popleft()
            if k in red:
                visible.append(k)
            r, c = divmod(k, 16)
            for rr, cc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                if 0 <= rr < 16 and 0 <= cc < 16:
                    q = rr * 16 + cc
                    if q not in visited and (q in red or raw[q] is None):
                        visited.add(q)
                        queue.append(q)
        if not visible:
            continue
        rs = [k // 16 for k in visible]
        cs = [k % 16 for k in visible]
        rectangle = [r * 16 + c
                     for r in range(min(rs), max(rs) + 1)
                     for c in range(min(cs), max(cs) + 1)]
        if all(k in red or raw[k] is None for k in rectangle):
            for k in rectangle:
                if terrain[k] != 3:
                    terrain[k] = 4
                    known.add(k)

    previous = memory.get("state")
    predicted = memory.get("expected", previous)
    if previous is None or predicted is None:
        nh, ne = len(found_helpers), len(found_enemies)
        apos = [player if player is not None else _CELLS[0]]
        apos += found_helpers + found_enemies
        boxes = list(found_boxes)
        memory["nh"], memory["ne"] = nh, ne
        memory["ho"] = list(range(1, nh + 1))
        memory["eo"] = list(range(nh + 1, nh + ne + 1))
        memory["bo"] = list(range(len(boxes)))
    else:
        nh, ne = memory["nh"], memory["ne"]
        pa, pb, _, _ = predicted
        oa, ob, _, _ = previous
        apos = [-1] * (1 + nh + ne)
        apos[0] = player if player is not None else pa[0]
        ids = _assignment(found_helpers, list(pa[1:1 + nh]),
                          list(oa[1:1 + nh]))
        for p, j in zip(found_helpers, ids):
            apos[1 + j] = p
        ids = _assignment(found_enemies, list(pa[1 + nh:]),
                          list(oa[1 + nh:]))
        for p, j in zip(found_enemies, ids):
            apos[1 + nh + j] = p
        ids = _assignment(found_boxes, list(pb), list(ob))
        boxes = list(pb)
        for p, j in zip(found_boxes, ids):
            if j < len(boxes):
                boxes[j] = p

    att = [-1] * len(apos)
    claimed = set()
    pred_att = predicted[2] if predicted is not None else ()
    old_att = previous[2] if previous is not None else ()
    for b, p in enumerate(boxes):
        rim = entities.get(p, ("b", 4))[1]
        if rim == 0 and _dist(apos[0], p) == 1:
            att[0] = b
            claimed.add(b)

    ownership_choices = []
    for b, p in enumerate(boxes):
        if b in claimed:
            continue
        rim = entities.get(p, ("b", 4))[1]
        if rim not in (5, 3):
            continue
        for i in range(1, len(apos)):
            if apos[i] < 0 or _dist(apos[i], p) != 1:
                continue
            score = 100 if rim == 5 else 0
            if i < len(pred_att) and pred_att[i] == b:
                score += 40
            if i < len(old_att) and old_att[i] == b:
                score += 15
            if i > nh:
                score += 2
            if rim == 3 and score < 15:
                continue
            ownership_choices.append((-score, i, b))
    ownership_choices.sort()
    for _, i, b in ownership_choices:
        if att[i] < 0 and b not in claimed:
            att[i] = b
            claimed.add(b)

    state = (tuple(apos), tuple(boxes), tuple(att), face)
    if previous is not None:
        oa, ob, oat, _ = previous
        for key in ("ho", "eo"):
            order = memory.get(key, [])
            newly = [i for i in order
                     if i < len(oat) and oat[i] < 0 <= att[i] and apos[i] >= 0]
            if len(newly) == 1:
                grabbed = newly[0]
                moved = [i for i in order
                         if i != grabbed and apos[i] >= 0 and oa[i] != apos[i]]
                rest = [i for i in order
                        if i not in moved and i != grabbed]
                memory[key] = moved + [grabbed] + rest
                b = att[grabbed]
                bo = memory.get("bo", list(range(len(boxes))))
                memory["bo"] = [b] + [j for j in bo if j != b]

    memory["terrain"] = terrain
    memory["known"] = list(known)
    return state, terrain


class _World:
    def __init__(self, terrain, memory):
        self.walls = set(_BORDER)
        self.pattern = set()
        self.goals = set()
        self.red = set()
        self.unknown = set()
        known = set(memory.get("known", []))
        for k, value in enumerate(terrain):
            p = (k // 16 + 1) * 18 + k % 16 + 1
            if value == 1:
                self.walls.add(p)
            elif value == 2:
                self.pattern.add(p)
            elif value == 3:
                self.goals.add(p)
            elif value == 4:
                self.red.add(p)
            if k not in known:
                self.unknown.add(p)
        self.ho = list(memory.get("ho", []))
        self.eo = list(memory.get("eo", []))
        self.bo = list(memory.get("bo", []))
        self.memo = {}

    def won(self, s):
        return (all(p in self.goals for p in s[1]) and
                all(b < 0 for b in s[2]))

    def _route(self, start, targets, occupied, box=None, destination=None):
        if box is None:
            if not targets or start in targets:
                return None
        else:
            offset = box - start
            if box in destination:
                return None

        seen = {start}
        queue = deque([(start, None)])
        while queue:
            p, first = queue.popleft()
            for d in (-1, 1, -18, 18):
                q = p + d
                if q in seen or q not in _CELLSET or q in self.pattern:
                    continue
                if box is None:
                    if q in occupied:
                        continue
                else:
                    z = q + offset
                    if ((q in occupied and q != box) or
                        z not in _CELLSET or
                            (z in occupied and z != start)):
                        continue
                seen.add(q)
                f = q if first is None else first
                if ((box is None and q in targets) or
                    (box is not None and q + offset in destination)):
                    return f
                queue.append((q, f))
        return None

    def step(self, s, action):
        key = (s, action)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        aa, bb, tt, face = s
        a, b, att = list(aa), list(bb), list(tt)
        occupied = self.walls | {p for p in a if p >= 0} | set(b)

        def owner_of(j):
            for i, x in enumerate(att):
                if x == j:
                    return i
            return -1

        def release(i):
            att[i] = -1

        def attach(i, j):
            old = owner_of(j)
            if old >= 0:
                release(old)
            att[i] = j

        def move(i, q):
            old = a[i]
            j = att[i]
            if j < 0:
                if q in occupied or q in self.pattern or q not in _CELLSET:
                    return
                occupied.discard(old)
                a[i] = q
                occupied.add(q)
            else:
                z = q + b[j] - old
                if (q not in _CELLSET or z not in _CELLSET or
                    q in self.pattern or
                    (q in occupied and q != b[j]) or
                        (z in occupied and z != old)):
                    return
                occupied.discard(old)
                occupied.discard(b[j])
                a[i], b[j] = q, z
                occupied.add(q)
                occupied.add(z)

        if action < 4:
            if att[0] < 0:
                face = action
            move(0, a[0] + _D[action])
        elif att[0] >= 0:
            release(0)
        else:
            front = a[0] + _D[face]
            for j in self.bo:
                if b[j] == front:
                    attach(0, j)
                    break
            for i in self.eo:
                if a[i] == front:
                    release(i)
                    occupied.discard(a[i])
                    a[i] = -1
                    break

        for order, hostile in ((self.ho, False), (self.eo, True)):
            dest = self.red if hostile else self.goals
            for i in order:
                if a[i] < 0:
                    continue
                if att[i] >= 0:
                    j = att[i]
                    if b[j] in dest:
                        release(i)
                    else:
                        q = self._route(a[i], None, occupied, b[j], dest)
                        if q is not None:
                            move(i, q)
                else:
                    eligible = []
                    for j in self.bo:
                        owner = owner_of(j)
                        if hostile:
                            good = owner not in self.eo and b[j] not in self.red
                        else:
                            good = owner < 0 and b[j] not in self.goals
                        if good:
                            eligible.append(j)
                    grabbed = False
                    for j in eligible:
                        if _dist(a[i], b[j]) == 1:
                            attach(i, j)
                            grabbed = True
                            break
                    if grabbed:
                        break
                    targets = set()
                    for j in eligible:
                        targets.update(_NB[b[j]])
                    q = self._route(a[i], targets, occupied)
                    if q is not None:
                        move(i, q)

        result = (tuple(a), tuple(b), tuple(att), face)
        if len(self.memo) < 24000:
            self.memo[key] = result
        return result

    def idle(self, s):
        a, boxes, att, face = s
        if att[0] >= 0:
            return 4
        occupied = self.walls | set(boxes) | {p for p in a[1:] if p >= 0}
        front = a[0] + _D[face]
        if front not in boxes:
            return 4
        for d in (face, 0, 1, 2, 3):
            if a[0] + _D[d] in occupied or a[0] + _D[d] in self.pattern:
                return d
        return 4


def _flood(start, blocked, limit=None):
    distance = {start: 0}
    parent = {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if limit is not None and distance[p] >= limit:
            continue
        for action, q in enumerate(_NB[p]):
            if q not in distance and q not in blocked and q in _CELLSET:
                distance[q] = distance[p] + 1
                parent[q] = (p, action)
                queue.append(q)
    return distance, parent


def _path(parent, start, end):
    result = []
    while end != start:
        if end not in parent:
            return None
        end, action = parent[end]
        result.append(action)
    result.reverse()
    return result


def _carry_flood(start, offset, occupied, pattern):
    distance = {start: 0}
    parent = {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for action, q in enumerate(_NB[p]):
            z = q + offset
            if (q in distance or q not in _CELLSET or z not in _CELLSET or
                    q in occupied or z in occupied or q in pattern):
                continue
            distance[q] = distance[p] + 1
            parent[q] = (p, action)
            queue.append(q)
    return distance, parent


def _reverse_carry(world, occupied, offset):
    distance = {}
    queue = deque()
    for box in world.goals:
        p = box - offset
        if (p in _CELLSET and p not in occupied and box not in occupied
                and p not in world.pattern):
            distance[p] = 0
            queue.append(p)
    while queue:
        p = queue.popleft()
        for q in _NB[p]:
            z = q + offset
            if (q not in distance and q in _CELLSET and z in _CELLSET and
                    q not in occupied and z not in occupied and
                    q not in world.pattern):
                distance[q] = distance[p] + 1
                queue.append(q)
    return distance


def _helper_distances(world, s):
    result = {}
    blocked = world.walls | world.pattern | set(s[1])
    for i in world.ho:
        if s[0][i] >= 0:
            ds, _ = _flood(s[0][i], blocked - {s[0][i]})
            for p, d in ds.items():
                if d < result.get(p, _INF):
                    result[p] = d
    return result


def _assist_cost(world, s, j, helper_distance):
    result = {}
    if not world.ho:
        return result
    static_occ = world.walls | (set(s[1]) - {s[1][j]})
    for offset in _D:
        ds = _reverse_carry(world, static_occ, offset)
        for q, carry_cost in ds.items():
            walk_cost = helper_distance.get(q, _INF)
            if walk_cost < _INF:
                z = q + offset
                cost = walk_cost + carry_cost + 2
                if cost < result.get(z, _INF):
                    result[z] = cost
    return result


def _tasks(world, s, stalled, prefix=(), deadline=None):
    a, boxes, att, face = s
    player = a[0]
    occupied = world.walls | set(boxes) | {p for p in a if p >= 0}
    walk_dist, walk_parent = _flood(
        player, (occupied - {player}) | world.pattern)
    helper_distance = _helper_distances(world, s)
    tasks = []
    for j, box in enumerate(boxes):
        if deadline is not None and time.perf_counter() > deadline:
            break
        if att[0] >= 0 and att[0] != j:
            continue
        if box in world.goals and att[0] != j:
            continue
        owner = next((i for i, x in enumerate(att) if x == j), -1)
        if owner > 0 and j not in stalled:
            continue
        assist = (_assist_cost(world, s, j, helper_distance)
                  if j in stalled else {})

        for direction, offset in enumerate(_D):
            q = box - offset
            if att[0] == j:
                if player != q:
                    continue
                pickup = []
            else:
                if q not in walk_dist:
                    continue
                pickup = _path(walk_parent, player, q)
                last_face = pickup[-1] if pickup else face
                if last_face != direction:
                    pickup = pickup + [direction]
                pickup = pickup + [4]

            carry_occ = occupied - {player, box}
            ds, parent = _carry_flood(q, offset, carry_occ, world.pattern)
            goal_ends = [p for p in ds if p + offset in world.goals]
            goal_ends.sort(key=lambda p: ds[p])
            selected = [(p, 0) for p in goal_ends[:3]]

            if assist:
                old_assist = assist.get(box, _INF)
                handoffs = []
                for p, cost in ds.items():
                    z = p + offset
                    ac = assist.get(z, _INF)
                    if (z != box and z not in world.goals and ac < _INF and
                        (old_assist >= _INF or ac + 2 < old_assist or
                         (j in stalled and ac <= old_assist + 2))):
                        handoffs.append((cost + 0.65 * ac, p, ac))
                handoffs.sort()
                selected += [(p, ac) for _, p, ac in handoffs[:3]]

            for end, assist_cost in selected:
                carry = _path(parent, q, end)
                seq = list(prefix) + pickup + carry + [4]
                goal_distance = min(
                    (_dist(box, g) for g in world.goals), default=30)
                rank = len(seq) + 0.55 * assist_cost - 0.25 * goal_distance
                tasks.append((rank, seq, j, end + offset))
    return tasks


def _regrasp_tasks(world, s, stalled, deadline):
    """Shortest carry routes with explicit release/walk/regrasp edges."""
    a, boxes, att, face = s
    player = a[0]
    occupied = world.walls | set(boxes) | {p for p in a if p >= 0}
    walk_dist, walk_parent = _flood(
        player, (occupied - {player}) | world.pattern)
    helper_distance = _helper_distances(world, s)
    tasks = []

    order = sorted(range(len(boxes)),
                   key=lambda j: (j not in stalled, _dist(player, boxes[j])))
    for j in order:
        if time.perf_counter() >= deadline:
            break
        box = boxes[j]
        if att[0] >= 0 and att[0] != j:
            continue
        if box in world.goals and att[0] != j:
            continue
        owner = next((i for i, x in enumerate(att) if x == j), -1)
        if owner > 0 and j not in stalled:
            continue

        assist = _assist_cost(world, s, j, helper_distance)
        fixed = occupied - {player, box}
        distances = {}
        parents = {}
        changed = {}
        heap = []

        for direction, offset in enumerate(_D):
            q = box - offset
            if att[0] == j:
                if q != player:
                    continue
                seq = []
            else:
                if q not in walk_dist:
                    continue
                seq = _path(walk_parent, player, q)
                last_face = seq[-1] if seq else face
                if last_face != direction:
                    seq += [direction]
                seq += [4]
            node = (box, direction)
            distances[node] = len(seq)
            parents[node] = (None, seq)
            changed[node] = False
            heapq.heappush(heap, (len(seq), node))

        possible = []
        expanded = 0
        while heap:
            cost, node = heapq.heappop(heap)
            if cost != distances.get(node):
                continue
            if cost > 85:
                break
            expanded += 1
            if expanded % 24 == 0 and time.perf_counter() >= deadline:
                break

            bp, direction = node
            offset = _D[direction]
            pp = bp - offset

            if changed[node]:
                if bp in world.goals:
                    possible.append((cost + 1, node, 0))
                elif bp != box and bp in assist:
                    possible.append(
                        (cost + 1 + 0.55 * assist[bp], node, assist[bp]))

            for action, d in enumerate(_D):
                nq, nb = pp + d, bp + d
                if (nq not in _CELLSET or nb not in _CELLSET or
                    nq in fixed or nb in fixed or nq in world.pattern):
                    continue
                nxt = (nb, direction)
                nc = cost + 1
                if nc < distances.get(nxt, _INF):
                    distances[nxt] = nc
                    parents[nxt] = (node, [action])
                    changed[nxt] = changed[node]
                    heapq.heappush(heap, (nc, nxt))

            # After dropping the crate, find paths to its other three sides.
            targets = []
            for nd, noff in enumerate(_D):
                nq = bp - noff
                if (nd != direction and nq in _CELLSET and
                        nq not in fixed and nq not in world.pattern):
                    targets.append((nd, nq))
            if not targets:
                continue
            blocked = fixed | world.pattern | {bp}
            ds, parent = _flood(pp, blocked, 16)
            for nd, nq in targets:
                if nq not in ds:
                    continue
                route = _path(parent, pp, nq)
                edge = [4] + route
                last_face = route[-1] if route else direction
                if last_face != nd:
                    edge.append(nd)
                edge.append(4)
                nxt = (bp, nd)
                nc = cost + len(edge)
                if nc < distances.get(nxt, _INF):
                    distances[nxt] = nc
                    parents[nxt] = (node, edge)
                    changed[nxt] = True
                    heapq.heappush(heap, (nc, nxt))

        possible.sort(key=lambda x: x[0])
        selected = []
        used = set()
        # Keep several spatially distinct handoffs rather than only the
        # closest three locations.
        for rank, node, ac in possible:
            bp, direction = node
            key = (bp, direction)
            if key not in used:
                used.add(key)
                selected.append((rank, node, ac))
            if len(selected) >= 48:
                break

        for rank, node, ac in selected:
            edges = []
            cur = node
            while cur is not None:
                prev, edge = parents[cur]
                edges.append(edge)
                cur = prev
            sequence = []
            for edge in reversed(edges):
                sequence.extend(edge)
            sequence.append(4)
            goal_distance = min(
                (_dist(box, g) for g in world.goals), default=30)
            tasks.append((rank - 0.25 * goal_distance,
                          sequence, j, node[0]))
    return tasks


def _hunt(world, initial, deadline):
    count = sum(initial[0][i] >= 0 for i in world.eo)

    def heuristic(s):
        p = s[0][0]
        return min((max(0, _dist(p, s[0][i]) - 1)
                    for i in world.eo if s[0][i] >= 0), default=0)

    serial = 0
    heap = [(1.8 * heuristic(initial), serial, 0, initial, ())]
    seen = {initial: 0}
    best_path = ()
    best_value = heuristic(initial)
    expanded = 0
    while heap and expanded < 2600 and time.perf_counter() < deadline:
        _, _, g, s, path = heapq.heappop(heap)
        if g != seen.get(s):
            continue
        expanded += 1
        if g >= 35:
            continue
        for action in (0, 1, 2, 3, 4):
            ns = world.step(s, action)
            if ns == s:
                continue
            ng = g + 1
            if seen.get(ns, _INF) <= ng:
                continue
            np = path + (action,)
            if (world.won(ns) or
                    sum(ns[0][i] >= 0 for i in world.eo) < count):
                return list(np)
            seen[ns] = ng
            h = heuristic(ns)
            value = h + 0.15 * ng
            if value < best_value:
                best_value, best_path = value, np
            serial += 1
            heapq.heappush(heap, (ng + 1.8 * h, serial, ng, ns, np))
    if best_path:
        return list(best_path[:3])
    return [world.idle(initial)]


def _residual(world, s):
    boxes, att = s[1], s[2]
    owned = {j for j in att if j >= 0}
    remaining = [j for j, p in enumerate(boxes)
                 if p not in world.goals or j in owned]
    distance = sum(min((_dist(boxes[j], g) for g in world.goals),
                       default=30) for j in remaining)
    return 500 * len(remaining) + 4 * distance + 5 * len(owned)


def _rollout(world, state, horizon, deadline, snapshots=False):
    s = state
    actions = []
    saved = []
    if world.won(s):
        return s, 0, actions, saved
    for t in range(horizon):
        if (t & 7) == 0 and time.perf_counter() > deadline:
            break
        action = world.idle(s)
        ns = world.step(s, action)
        actions.append(action)
        s = ns
        if snapshots and t + 1 in (6, 15, 30):
            saved.append(s)
        if world.won(s):
            return s, t + 1, actions, saved
        if ns == state and t > 2:
            break
        state = ns
    return s, None, actions, saved


def _staging_tasks(world, s, memory, deadline):
    """Last-resort rearrangements, also revealing initially covered terrain."""
    a, boxes, att, face = s
    player = a[0]
    occupied = world.walls | set(boxes) | {p for p in a if p >= 0}
    ds, parent = _flood(player, (occupied - {player}) | world.pattern)
    recent = memory.get("staging_recent", [])
    tasks = []
    for j, box in enumerate(boxes):
        if time.perf_counter() > deadline:
            break
        if att[0] >= 0 and att[0] != j:
            continue
        owner = next((i for i, x in enumerate(att) if x == j), -1)
        if owner > 0:
            continue
        for direction, offset in enumerate(_D):
            q = box - offset
            if att[0] == j:
                if q != player:
                    continue
                pickup = []
            else:
                if q not in ds:
                    continue
                pickup = _path(parent, player, q)
                last = pickup[-1] if pickup else face
                if last != direction:
                    pickup += [direction]
                pickup += [4]
            fixed = occupied - {player, box}
            cd, cp = _carry_flood(q, offset, fixed, world.pattern)
            for end, length in cd.items():
                target = end + offset
                if not 1 <= length <= 3 or target in world.pattern:
                    continue
                free_sides = sum(
                    p in _CELLSET and p not in fixed and p not in world.pattern
                    for p in _NB[target])
                old_sides = sum(
                    p in _CELLSET and p not in fixed and p not in world.pattern
                    for p in _NB[box])
                rank = (len(pickup) + length + 1 +
                        (25 if box in world.goals else 0) -
                        3 * (free_sides - old_sides) -
                        (7 if box in world.unknown else 0))
                if [j, target] in recent:
                    rank += 30
                sequence = pickup + _path(cp, q, end) + [4]
                tasks.append((rank, sequence, j, target))
    return sorted(tasks, key=lambda t: t[0])


def _choose(world, s, memory, deadline):
    if any(s[0][i] >= 0 for i in world.eo):
        return _hunt(world, s, min(deadline, time.perf_counter() + 5.0))

    horizon = 105
    remaining = memory.get("remaining")
    if remaining is not None:
        horizon = max(12, min(110, remaining))

    final, win_time, idle_actions, forecasts = _rollout(
        world, s, horizon, deadline, True)
    stalled = {j for j, p in enumerate(final[1])
               if p not in world.goals or j in final[2]}

    if win_time is not None:
        best_score = -10000 + win_time
    else:
        best_score = _residual(world, final)
    baseline = best_score
    best_plan = idle_actions[:1] or [world.idle(s)]
    evaluated_sequences = set()

    def evaluate(tasks, cap):
        nonlocal best_score, best_plan
        unique = {}
        for task in tasks:
            key = tuple(task[1])
            if key in evaluated_sequences:
                continue
            if key not in unique or task[0] < unique[key][0]:
                unique[key] = task
        ordered = sorted(unique.values(), key=lambda x: x[0])
        evaluated = 0

        for rank, sequence, box_id, target in ordered:
            if evaluated >= cap or time.perf_counter() > deadline - 0.12:
                break
            if len(sequence) > horizon:
                continue
            evaluated_sequences.add(tuple(sequence))
            cur = s
            good = True
            early_win = None
            for k, action in enumerate(sequence):
                if (k & 15) == 0 and time.perf_counter() > deadline - 0.1:
                    good = False
                    break
                if k == len(sequence) - 1:
                    if cur[2][0] != box_id or cur[1][box_id] != target:
                        good = False
                        break
                cur = world.step(cur, action)
                if world.won(cur):
                    early_win = k + 1
                    break
            if early_win is not None:
                score = -10000 + early_win
                if score < best_score:
                    best_score = score
                    best_plan = sequence[:early_win]
                evaluated += 1
                continue
            if not good:
                continue

            evaluated += 1
            end, wt, _, _ = _rollout(
                world, cur, max(0, horizon - len(sequence)), deadline)
            if wt is not None:
                score = -10000 + len(sequence) + wt
            else:
                score = _residual(world, end) + 0.8 * len(sequence)
            if score < best_score - 0.01:
                best_score, best_plan = score, sequence

    tasks = _tasks(world, s, stalled, deadline=deadline)
    if s[2][0] < 0:
        for f in forecasts[:2]:
            if time.perf_counter() > deadline - 2.0:
                break
            aa = list(f[0])
            aa[0] = s[0][0]
            tt = list(f[2])
            tt[0] = -1
            forecast = (tuple(aa), f[1], tuple(tt), s[3])
            tasks.extend(_tasks(world, forecast, stalled, deadline=deadline))
    else:
        released = world.step(s, 4)
        tasks.extend(_tasks(world, released, stalled, prefix=(4,),
                            deadline=deadline))
    evaluate(tasks, 42)

    # If simple straight-orientation deliveries make no progress, search
    # the larger configuration graph with changes of carrying orientation.
    if (win_time is None and best_score >= baseline - 0.01 and stalled and
            time.perf_counter() < deadline - 0.8):
        search_deadline = min(deadline - 0.65, time.perf_counter() + 2.8)
        advanced = _regrasp_tasks(world, s, stalled, search_deadline)
        evaluate(advanced, 70)

    # Move out of doorways that obstruct the automatic carriers.
    if time.perf_counter() < deadline - 0.2 and s[2][0] < 0:
        occupied = (world.walls | world.pattern | set(s[1]) |
                    {p for p in s[0][1:] if p >= 0})
        ds, parent = _flood(s[0][0], occupied)
        endpoints = []
        for p, d in ds.items():
            if 1 <= d <= 6:
                clearance = sum(q not in occupied for q in _NB[p])
                near_goal = min(
                    (_dist(p, g) for g in world.goals), default=8)
                endpoints.append((d - 0.4 * clearance -
                                  0.15 * min(near_goal, 6), p))
        endpoints.sort()
        sampled = []
        used_sectors = set()
        for _, p in endpoints:
            sector = ((p // 18 - s[0][0] // 18 > 0),
                      (p % 18 - s[0][0] % 18 > 0),
                      ds[p] // 2)
            if sector not in used_sectors:
                used_sectors.add(sector)
                sampled.append(p)
            if len(sampled) >= 10:
                break

        for p in sampled:
            if time.perf_counter() > deadline - 0.1:
                break
            sequence = _path(parent, s[0][0], p)
            cur = s
            for action in sequence:
                cur = world.step(cur, action)
            end, wt, _, _ = _rollout(
                world, cur, max(0, horizon - len(sequence)), deadline)
            if wt is not None:
                score = -10000 + len(sequence) + wt
            else:
                score = _residual(world, end) + len(sequence)
            if score < best_score - 0.01:
                best_score, best_plan = score, sequence

    # A static board should not cause unlimited no-op actions. If neither
    # delivery nor regrasp search works, try a small, non-repeating staging
    # move after confirming that the automatic carriers are not progressing.
    if (win_time is None and best_score >= baseline - 0.01 and
            memory.get("stationary", 0) >= 3 and
            time.perf_counter() < deadline - 0.15):
        staging = _staging_tasks(world, s, memory, deadline - 0.1)
        for rank, sequence, j, target in staging[:28]:
            if time.perf_counter() > deadline - 0.1:
                break
            cur = s
            valid = True
            for k, action in enumerate(sequence):
                if k == len(sequence) - 1:
                    if cur[2][0] != j or cur[1][j] != target:
                        valid = False
                        break
                cur = world.step(cur, action)
            if valid:
                recent = memory.get("staging_recent", [])
                recent.extend([[j, s[1][j]], [j, target]])
                memory["staging_recent"] = recent[-24:]
                best_plan = sequence
                break

    return best_plan or [world.idle(s)]


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 8.6
    memory["turn"] = memory.get("turn", 0) + 1

    bar = sum(int(x) == 7 for x in obs[63])
    if "initial_bar" not in memory:
        memory["initial_bar"] = bar
        memory["initial_turn"] = memory["turn"]
    elapsed = memory["turn"] - memory["initial_turn"]
    consumed = memory["initial_bar"] - bar
    if elapsed >= 8 and consumed >= 3:
        total = 64.0 * elapsed / consumed
        memory["remaining"] = max(1, int(total * bar / 64.0))

    state, terrain = _read(obs, memory)
    world = _World(terrain, memory)

    previous = memory.get("state")
    if previous is not None and state[:3] == previous[:3]:
        memory["stationary"] = memory.get("stationary", 0) + 1
    else:
        memory["stationary"] = 0

    plan = memory.get("plan", [])
    if state != memory.get("expected"):
        plan = []
    if not plan:
        plan = _choose(world, state, memory, deadline)

    action = plan[0] if plan else world.idle(state)
    name = _NAMES[action]
    if name not in available:
        allowed = [i for i, n in enumerate(_NAMES) if n in available]
        if allowed:
            action = allowed[0]
            name = _NAMES[action]
            plan = [action]
        elif "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        else:
            return (available[0], None, None)

    memory["state"] = state
    memory["expected"] = world.step(state, action)
    memory["plan"] = list(plan[1:])
    return (name, None, None)

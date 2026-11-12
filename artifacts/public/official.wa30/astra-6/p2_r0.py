import numpy as np
import heapq
import time

_INF = 10000
_ALL = (1 << 256) - 1
_BIT = [1 << i for i in range(256)]
_LEFT = sum(_BIT[y * 16] for y in range(16))
_RIGHT = _LEFT << 15
_OPP = (1, 0, 3, 2)
_ORDER = (2, 3, 0, 1)  # Simulator BFS order: left, right, up, down.
_N = []
for _p in range(256):
    _x, _y = _p % 16, _p // 16
    _N.append((
        _p - 16 if _y else -1,
        _p + 16 if _y < 15 else -1,
        _p - 1 if _x else -1,
        _p + 1 if _x < 15 else -1,
    ))


def _bits(mask):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def _expand(mask):
    return (((mask & ~_LEFT) >> 1) |
            ((mask & ~_RIGHT) << 1) |
            (mask >> 16) |
            ((mask << 16) & _ALL))


def _pre(mask, d):
    """Carrier positions whose carried object, in direction d, is in mask."""
    if d == 0:
        return (mask << 16) & _ALL
    if d == 1:
        return mask >> 16
    if d == 2:
        return ((mask & ~_RIGHT) << 1) & _ALL
    return (mask & ~_LEFT) >> 1


def _relative(p, c):
    for d in range(4):
        if _N[p][d] == c:
            return d
    return 0


def _first_step(start, allowed, targets):
    if targets & _BIT[start]:
        return start
    allowed |= _BIT[start]
    front = _BIT[start]
    seen = front
    layers = [front]
    while front:
        front = _expand(front) & allowed & ~seen
        if not front:
            return -1
        layers.append(front)
        seen |= front
        hit = front & targets
        if hit:
            back = hit
            for i in range(len(layers) - 2, 0, -1):
                back = _expand(back) & layers[i]
            for d in _ORDER:
                q = _N[start][d]
                if q >= 0 and back & _BIT[q]:
                    return q
            return -1
    return -1


def _distances(start, allowed):
    result = [_INF] * 256
    if start < 0 or not (allowed & _BIT[start]):
        return result
    front = _BIT[start]
    seen = front
    depth = 0
    while front:
        for p in _bits(front):
            result[p] = depth
        front = _expand(front) & allowed & ~seen
        seen |= front
        depth += 1
    return result


def _rect_mask(x0, y0, x1, y1):
    row = ((1 << (x1 - x0)) - 1) << x0
    result = 0
    for y in range(y0, y1):
        result |= row << (16 * y)
    return result


def _prefix(a):
    a = np.asarray(a, dtype=np.int32).reshape(16, 16)
    out = np.zeros((17, 17), dtype=np.int32)
    out[1:, 1:] = a.cumsum(0).cumsum(1)
    return out.tolist()


def _sumrect(p, x0, y0, x1, y1):
    return p[y1][x1] - p[y0][x1] - p[y1][x0] + p[y0][x0]


def _extract(obs, memory):
    codes = list(memory.get("_terrain", [-1] * 256))
    walls = memory.get("_walls", 0)
    fences = memory.get("_fences", 0)
    player = -1
    direction = 0
    crates, helpers, enemies = [], [], []
    borders = {}

    for p in range(256):
        x, y = (p % 16) * 4, (p // 16) * 4
        tile = obs[y:min(y + 4, 63), x:x + 4]
        center = int(obs[y + 1, x + 1])

        if center == 14:
            player = p
            if np.all(tile[0] == 0):
                direction = 0
            elif np.all(tile[:, 0] == 0):
                direction = 2
            elif np.all(tile[:, -1] == 0):
                direction = 3
            else:
                direction = 1
            continue

        if center == 12:
            helpers.append(p)
            continue

        if center == 15:
            enemies.append(p)
            continue

        if center == 9 and int(tile[0, 0]) in (0, 3, 4, 5):
            crates.append(p)
            borders[p] = int(tile[0, 0])
            continue

        if np.all(tile == 5):
            walls |= _BIT[p]
            codes[p] = -2
            continue

        if np.any(tile == 2) and np.any(tile == 1):
            fences |= _BIT[p]
            codes[p] = -2
            continue

        if np.all((tile == 2) | (tile == 9)):
            code = 0
            if np.all(tile[0] == 9):
                code |= 1
            if y < 60 and np.all(tile[-1] == 9):
                code |= 2
            if np.all(tile[:, 0] == 9):
                code |= 4
            if np.all(tile[:, -1] == 9):
                code |= 8

            expected = np.full(tile.shape, 2, dtype=np.int8)
            if code & 1:
                expected[0] = 9
            if code & 2:
                expected[-1] = 9
            if code & 4:
                expected[:, 0] = 9
            if code & 8:
                expected[:, -1] = 9
            if np.array_equal(expected, tile):
                codes[p] = code
            continue

        if np.all(tile == 1):
            codes[p] = -2

    memory["_terrain"] = codes
    memory["_walls"] = walls
    memory["_fences"] = fences

    # Recover rectangular goal areas, including portions hidden by objects.
    known = [int(c >= 0) for c in codes]
    border = [int(c > 0) for c in codes]
    forbidden = [int(c == -2) for c in codes]
    pk = _prefix(known)
    pb = _prefix(border)
    pf = _prefix(forbidden)
    ones, bit_known = [], []
    for bit in (1, 2, 4, 8):
        ones.append(_prefix([
            int(c >= 0 and bool(c & bit)) for c in codes
        ]))
        bit_known.append(_prefix([
            int(c >= 0 and not (bit == 2 and p // 16 == 15))
            for p, c in enumerate(codes)
        ]))

    candidates = []
    if any(border):
        for y0 in range(16):
            for y1 in range(y0 + 1, 17):
                for x0 in range(16):
                    for x1 in range(x0 + 1, 17):
                        if _sumrect(pf, x0, y0, x1, y1):
                            continue
                        if not _sumrect(pb, x0, y0, x1, y1):
                            continue
                        edges = (
                            (x0, y0, x1, y0 + 1),
                            (x0, y1 - 1, x1, y1),
                            (x0, y0, x0 + 1, y1),
                            (x1 - 1, y0, x1, y1),
                        )
                        valid = True
                        for j, edge in enumerate(edges):
                            edge_ones = _sumrect(ones[j], *edge)
                            if _sumrect(ones[j], x0, y0, x1, y1) != edge_ones:
                                valid = False
                                break
                            if _sumrect(bit_known[j], *edge) != edge_ones:
                                valid = False
                                break
                        if valid:
                            area = (x1 - x0) * (y1 - y0)
                            nk = _sumrect(pk, x0, y0, x1, y1)
                            candidates.append((
                                -(9 * nk - area), area, x0, y0, x1, y1
                            ))

    goals = 0
    visible_border = sum(_BIT[p] for p in range(256) if border[p])
    for _, _, x0, y0, x1, y1 in sorted(candidates):
        mask = _rect_mask(x0, y0, x1, y1)
        if not (mask & goals) and (mask & visible_border & ~goals):
            goals |= mask

    bad = sum(_BIT[p] for p, c in enumerate(codes)
              if c == 0 and not (goals & _BIT[p]))
    todo = bad
    while todo:
        start = todo & -todo
        component = start
        front = start
        while front:
            front = _expand(front) & bad & ~component
            component |= front
        todo &= ~component
        cells = list(_bits(component))
        xs = [p % 16 for p in cells]
        ys = [p // 16 for p in cells]
        rect = _rect_mask(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        if not (rect & (goals | walls | fences)):
            if all(codes[p] in (-1, 0) for p in _bits(rect)):
                bad |= rect

    # Fill small occluded gaps in solid-red areas.
    for _ in range(3):
        add = 0
        for p, c in enumerate(codes):
            if c != -1 or (goals | bad | walls | fences) & _BIT[p]:
                continue
            u, d, l, r = _N[p]
            if ((u >= 0 and d >= 0 and bad & _BIT[u] and bad & _BIT[d]) or
                    (l >= 0 and r >= 0 and bad & _BIT[l] and bad & _BIT[r])):
                add |= _BIT[p]
        bad |= add

    return player, direction, crates, helpers, enemies, borders, (
        walls, fences, goals, bad
    )


def _match(positions, expected, previous):
    n = max(len(expected), len(previous), len(positions))
    if not expected and not previous:
        return tuple(sorted(positions))
    choices = []
    for i in range(n):
        ep = expected[i] if i < len(expected) else -1
        pp = previous[i] if i < len(previous) else -1
        for j, p in enumerate(positions):
            def md(q):
                if q < 0:
                    return 100
                return abs(p % 16 - q % 16) + abs(p // 16 - q // 16)
            cost = min(2 * md(ep), 2 * md(pp) + 1)
            choices.append((cost, i, j))
    result = [-1] * n
    used_i, used_j = set(), set()
    for _, i, j in sorted(choices):
        if i not in used_i and j not in used_j:
            result[i] = positions[j]
            used_i.add(i)
            used_j.add(j)
    return tuple(result)


def _read_state(parsed, memory):
    p, d, cp, hp, ep, borders, _ = parsed
    old = memory.get("_state")
    pred = memory.get("_expected", old)
    if old is None:
        cp, hp, ep = tuple(cp), tuple(hp), tuple(ep)
    else:
        cp = _match(cp, pred[2], old[2])
        hp = _match(hp, pred[3], old[3])
        ep = _match(ep, pred[4], old[4])

    h = len(hp)
    bots = (p,) + hp + ep
    owners = [-1] * len(cp)
    used = set()

    for i, c in enumerate(cp):
        if borders.get(c) == 0:
            owners[i] = 0
            used.add(0)

    order = sorted(range(len(cp)),
                   key=lambda i: 0 if borders.get(cp[i]) == 5 else 1)
    for i in order:
        if owners[i] == 0:
            continue
        color = borders.get(cp[i], 4)
        if color not in (3, 5):
            continue

        candidates = [
            j for j in range(1, len(bots))
            if j not in used and bots[j] >= 0 and cp[i] in _N[bots[j]]
        ]
        preferred = []
        for prior in (pred, old):
            if prior is not None and i < len(prior[5]):
                owner = prior[5][i]
                if owner in candidates:
                    preferred.append(owner)

        if preferred:
            owners[i] = preferred[0]
        elif color == 5 and candidates:
            owners[i] = candidates[0]
        if owners[i] >= 0:
            used.add(owners[i])

    if p < 0 and old is not None:
        p, d = old[:2]
    return (p, d, tuple(cp), tuple(hp), tuple(ep), tuple(owners))


def _build_world(signature, old_world=None):
    walls, fences, goals, bad = signature
    floor = _ALL & ~(walls | fences)
    walk = [_distances(p, floor) for p in range(256)]

    valid = [False] * 1024
    carriers = [-1] * 1024
    for c in range(256):
        if walls & _BIT[c]:
            continue
        for d in range(4):
            p = _N[c][_OPP[d]]
            if p >= 0 and floor & _BIT[p]:
                n = c * 4 + d
                valid[n] = True
                carriers[n] = p

    graph = [[] for _ in range(1024)]
    for c in range(256):
        for d in range(4):
            n = c * 4 + d
            if not valid[n]:
                continue
            for q in _N[c]:
                if q >= 0 and valid[q * 4 + d]:
                    graph[n].append((q * 4 + d, 1))
            dist = _distances(carriers[n], floor & ~_BIT[c])
            for e in range(4):
                m = c * 4 + e
                if e != d and valid[m] and dist[carriers[m]] < _INF:
                    # Optimistic release, walk around, and reattach cost.
                    graph[n].append((m, dist[carriers[m]] + 2))

    return {
        "signature": signature,
        "walls": walls,
        "fences": fences,
        "floor": floor,
        "goals": goals,
        "bad": bad,
        "walk": walk,
        "valid": valid,
        "carriers": carriers,
        "graph": graph,
        "cache": {},
        "crate_order": list(old_world.get("crate_order", [])) if old_world else [],
        "helper_order": list(old_world.get("helper_order", [])) if old_world else [],
        "enemy_order": list(old_world.get("enemy_order", [])) if old_world else [],
    }


def _simulate(s, action, w):
    p, facing, c0, h0, e0, o0 = s
    crates, helpers, enemies, owners = list(c0), list(h0), list(e0), list(o0)
    h = len(helpers)
    floor = w["floor"]
    walls = w["walls"]
    goals, bad = w["goals"], w["bad"]

    occ = walls | _BIT[p]
    for q in crates + helpers + enemies:
        if q >= 0:
            occ |= _BIT[q]

    held = owners.index(0) if 0 in owners else -1
    if 1 <= action <= 4:
        direction = action - 1
        q = _N[p][direction]
        if held < 0:
            facing = direction
            if q >= 0 and floor & _BIT[q] and not (occ & _BIT[q]):
                occ = (occ & ~_BIT[p]) | _BIT[q]
                p = q
        else:
            c = crates[held]
            cq = _N[c][direction]
            other = occ & ~(_BIT[p] | _BIT[c])
            if (q >= 0 and cq >= 0 and floor & _BIT[q] and
                    not (other & (_BIT[q] | _BIT[cq]))):
                occ = other | _BIT[q] | _BIT[cq]
                p, crates[held] = q, cq

    elif action == 5:
        if held >= 0:
            owners[held] = -1
        else:
            front = _N[p][facing]
            if front >= 0:
                for k in w["crate_order"]:
                    if crates[k] == front:
                        owners[k] = 0
                        break
                for i, q in enumerate(enemies):
                    if q == front:
                        owner_id = 1 + h + i
                        if owner_id in owners:
                            owners[owners.index(owner_id)] = -1
                        enemies[i] = -1
                        occ &= ~_BIT[q]
                        break

    def run_bots(is_enemy):
        nonlocal occ
        positions = enemies if is_enemy else helpers
        ordering = w["enemy_order"] if is_enemy else w["helper_order"]
        destination = bad if is_enemy else goals
        for i in ordering:
            bp = positions[i]
            if bp < 0:
                continue
            owner_id = 1 + (h if is_enemy else 0) + i
            held_index = owners.index(owner_id) if owner_id in owners else -1

            if held_index >= 0:
                c = crates[held_index]
                if destination & _BIT[c]:
                    owners[held_index] = -1
                    continue
                d = _relative(bp, c)
                other = occ & ~(_BIT[bp] | _BIT[c])
                allowed = (floor & ~other &
                           _pre(_ALL & ~other & ~walls, d))
                q = _first_step(bp, allowed, _pre(destination, d))
                if q >= 0 and q != bp:
                    cq = _N[q][d]
                    occ = other | _BIT[q] | _BIT[cq]
                    positions[i], crates[held_index] = q, cq
                continue

            eligible = 0
            attached = False
            for k in w["crate_order"]:
                c = crates[k]
                if is_enemy:
                    okay = owners[k] < 1 + h and not (bad & _BIT[c])
                else:
                    okay = owners[k] == -1 and not (goals & _BIT[c])
                if not okay:
                    continue
                eligible |= _BIT[c]
                if c in _N[bp]:
                    owners[k] = owner_id
                    attached = True
                    break

            if attached:
                # The game's attachment operation ends this bot group's turn.
                break

            if eligible:
                q = _first_step(bp, floor & ~occ, _expand(eligible))
                if q >= 0 and q != bp:
                    occ = (occ & ~_BIT[bp]) | _BIT[q]
                    positions[i] = q

    run_bots(False)
    run_bots(True)
    return (p, facing, tuple(crates), tuple(helpers),
            tuple(enemies), tuple(owners))


def _dijkstra(w, permitted, seeds):
    dist = [_INF] * 1024
    heap = []
    for n, value in seeds:
        if permitted[n] and value < dist[n]:
            dist[n] = value
            heapq.heappush(heap, (value, n))
    graph = w["graph"]
    while heap:
        value, n = heapq.heappop(heap)
        if value != dist[n]:
            continue
        for q, cost in graph[n]:
            nv = value + cost
            if permitted[q] and nv < dist[q]:
                dist[q] = nv
                heapq.heappush(heap, (nv, q))
    return dist


def _transport_tables(s, w):
    settled = 0
    for c, owner in zip(s[2], s[5]):
        if owner == -1 and w["goals"] & _BIT[c]:
            settled |= _BIT[c]
    if settled in w["cache"]:
        return w["cache"][settled]

    permitted = [
        w["valid"][n] and
        not (settled & (_BIT[n // 4] | _BIT[w["carriers"][n]]))
        for n in range(1024)
    ]
    free_goals = w["goals"] & ~settled
    seeds = [
        (n, 1) for n in range(1024)
        if permitted[n] and free_goals & _BIT[n // 4]
    ]
    direct = _dijkstra(w, permitted, seeds)

    fixed = [_INF] * 1024
    for d in range(4):
        allowed = (w["floor"] & ~settled &
                   _pre(_ALL & ~w["walls"] & ~settled, d))
        front = _pre(free_goals, d) & allowed
        seen = front
        depth = 1
        while front:
            for p in _bits(front):
                c = _N[p][d]
                if c >= 0:
                    fixed[c * 4 + d] = depth
            front = _expand(front) & allowed & ~seen
            seen |= front
            depth += 1

    result = (direct, fixed, permitted, free_goals)
    if len(w["cache"]) > 12:
        w["cache"].clear()
    w["cache"][settled] = result
    return result


def _handoff_info(s, w, base):
    _, fixed, permitted, free_goals = base
    prices = [_INF] * 256
    stage = 0
    for c in range(256):
        for d in range(4):
            n = c * 4 + d
            if not permitted[n] or fixed[n] >= _INF:
                continue
            p = w["carriers"][n]
            for hp in s[3]:
                if hp < 0:
                    continue
                distance = w["walk"][hp][p]
                if distance < _INF:
                    price = 3 + 0.30 * (distance + fixed[n])
                    if price < prices[c]:
                        prices[c] = price
        if prices[c] < _INF:
            stage |= _BIT[c]
    return stage, prices


def _forecast(s, w, limit=120):
    done = [_INF] * len(s[2])
    seen = set()
    for t in range(limit + 1):
        for i, (c, owner) in enumerate(zip(s[2], s[5])):
            if owner == -1 and w["goals"] & _BIT[c] and done[i] == _INF:
                done[i] = t
        if all(v < _INF for v in done):
            return done, t
        if s in seen:
            break
        seen.add(s)
        s = _simulate(s, 0, w)
    return done, _INF


def _heuristic(s, task, w):
    p, facing, crates, helpers, enemies, owners = s
    if task["kind"] == "kill":
        ep = enemies[task["index"]]
        if ep < 0:
            return 0
        result = _INF
        for d in range(4):
            q = _N[ep][_OPP[d]]
            if q < 0:
                continue
            distance = w["walk"][p][q]
            extra = 1 if p == q and facing == d else 2
            result = min(result, distance + extra)
        return result + (1 if 0 in owners else 0)

    i = task["index"]
    c = crates[i]
    table = task["table"]
    if owners[i] == 0:
        return table[c * 4 + _relative(p, c)]

    result = _INF
    for d in range(4):
        q = _N[c][_OPP[d]]
        if q < 0:
            continue
        value = table[c * 4 + d]
        if value >= _INF:
            continue
        distance = w["walk"][p][q]
        extra = 1 if p == q and facing == d else 2
        result = min(result, distance + extra + value)
    if 0 in owners:
        result += 1
    return result


def _task_done(s, task, w):
    if task["kind"] == "kill":
        return s[4][task["index"]] < 0
    i = task["index"]
    c, owner = s[2][i], s[5][i]
    if owner == -1 and w["goals"] & _BIT[c]:
        return True
    if task["kind"] == "stage":
        return (c != task["avoid"] and
                bool(task["stage"] & _BIT[c]) and
                owner != 0 and owner < 1 + len(s[3]))
    return False


def _make_tasks(s, w):
    base = _transport_tables(s, w)
    direct, _, permitted, free_goals = base
    tasks = []

    for i, ep in enumerate(s[4]):
        if ep < 0:
            continue
        task = {"kind": "kill", "index": i}
        h = _heuristic(s, task, w)
        if h < _INF:
            task["score"] = h - 200
            task["h0"] = h
            tasks.append(task)

    held = s[5].index(0) if 0 in s[5] else -1
    if tasks:
        if held >= 0:
            task = {"kind": "deliver", "index": held, "table": direct}
            h = _heuristic(s, task, w)
            if h <= 5:
                task["score"], task["h0"] = -250 + h, h
                tasks.append(task)
        return sorted(tasks, key=lambda t: t["score"])

    if s[3]:
        automatic, finish_time = _forecast(s, w)
        stage, prices = _handoff_info(s, w, base)
    else:
        automatic = [_INF] * len(s[2])
        finish_time = _INF
        stage, prices = 0, None

    for i, (c, owner) in enumerate(zip(s[2], s[5])):
        if owner == -1 and w["goals"] & _BIT[c]:
            continue
        if held >= 0 and i != held:
            continue

        task = {"kind": "deliver", "index": i, "table": direct}
        h = _heuristic(s, task, w)
        auto = automatic[i]
        if h < _INF:
            useful = held == i or auto == _INF or auto > h + 4
            if useful:
                bonus = 65 if auto == _INF else 0.60 * min(auto, 110)
                penalty = 10 if 1 <= owner <= len(s[3]) and auto < _INF else 0
                task["score"] = h - bonus + penalty
                task["h0"] = h
                tasks.append(task)

        if prices is None or auto < _INF:
            continue

        already_staged = bool(stage & _BIT[c]) and owner != 0
        avoid = c if already_staged else -1
        seeds = []
        for n in range(1024):
            if not permitted[n]:
                continue
            q = n // 4
            if free_goals & _BIT[q]:
                seeds.append((n, 1))
            elif q != avoid and prices[q] < _INF:
                seeds.append((n, prices[q]))
        table = _dijkstra(w, permitted, seeds)
        task = {
            "kind": "stage", "index": i, "table": table,
            "stage": stage, "avoid": avoid,
        }
        sh = _heuristic(s, task, w)
        if sh < _INF:
            task["h0"] = sh
            task["score"] = sh - 60 + (15 if already_staged else 0)
            tasks.append(task)

    return sorted(tasks, key=lambda t: t["score"])


def _reconstruct(parent, state, initial):
    path = []
    while state != initial:
        if state not in parent:
            return []
        state, action = parent[state]
        path.append(action)
    path.reverse()
    return path


def _search(initial, task, w, deadline, max_nodes=7500):
    h0 = _heuristic(initial, task, w)
    if h0 >= _INF or _task_done(initial, task, w):
        return [], False, 0

    weight = 2.0 if task["kind"] == "kill" else 2.3
    heap = [(weight * h0, h0, 0, 0, initial)]
    distance = {initial: 0}
    parent = {}
    counter = 0
    expanded = 0
    best_state = initial
    best_value = h0
    fallback_state = None
    fallback_value = _INF
    depth_limit = min(105, int(3 * h0 + 25))

    while heap and expanded < max_nodes:
        _, h, _, g, s = heapq.heappop(heap)
        if distance.get(s) != g:
            continue
        if expanded % 32 == 0 and time.perf_counter() >= deadline:
            break
        expanded += 1
        if g >= depth_limit:
            continue

        p, facing, crates, helpers, enemies, owners = s
        front = _N[p][facing]
        actions = [1, 2, 3, 4]
        can_toggle = True
        if 0 not in owners and front in crates:
            ci = crates.index(front)
            if task["kind"] == "kill" or ci != task["index"]:
                can_toggle = False
        if can_toggle:
            actions.append(5)

        for action in actions:
            ns = _simulate(s, action, w)
            ng = g + 1
            if ng >= distance.get(ns, _INF):
                continue
            distance[ns] = ng
            parent[ns] = (s, action)
            if _task_done(ns, task, w):
                return _reconstruct(parent, ns, initial), True, h0

            nh = _heuristic(ns, task, w)
            if nh >= _INF:
                continue
            value = nh + 0.025 * ng
            if value < best_value:
                best_value, best_state = value, ns
            fv = nh + 0.25 * ng
            if fv < fallback_value:
                fallback_value, fallback_state = fv, ns

            counter += 1
            heapq.heappush(heap, (
                ng + weight * nh, nh, counter, ng, ns
            ))

    if best_state != initial:
        path = _reconstruct(parent, best_state, initial)
        return path[:6], False, h0 - best_value
    if fallback_state is not None:
        return _reconstruct(parent, fallback_state, initial)[:2], False, 0
    return [], False, 0


def _safe_idle(s, w):
    p, d, crates, helpers, enemies, owners = s
    if 0 in owners:
        return 5
    front = _N[p][d]
    if front not in crates and front not in enemies:
        return 5

    occ = w["walls"] | w["fences"]
    for q in crates + helpers + enemies:
        if q >= 0:
            occ |= _BIT[q]
    for direction, q in enumerate(_N[p]):
        if q < 0 or occ & _BIT[q]:
            return direction + 1
    return 5


def _fallback(s, w):
    idle = _safe_idle(s, w)
    if s[3] and not any(p >= 0 for p in s[4]):
        done, finish = _forecast(s, w, 100)
        if finish < _INF:
            return idle

        best_action, best_score = idle, -float("inf")
        for action in (idle, 1, 2, 3, 4):
            ns = _simulate(s, action, w)
            times, _ = _forecast(ns, w, 70)
            score = -sum(min(t, 150) for t in times)
            p = ns[0]
            clearance = sum(
                1 for q in _N[p]
                if q >= 0 and w["floor"] & _BIT[q]
            )
            score += 0.01 * clearance
            if action == idle:
                score += 0.005
            if score > best_score:
                best_score, best_action = score, action
        return best_action
    return idle


def _learn_order(s, previous, w):
    if previous is None or len(previous[2]) != len(s[2]):
        return
    h = len(s[3])
    old_bots = (previous[0],) + previous[3] + previous[4]
    order = w["crate_order"]
    for i, owner in enumerate(s[5]):
        if owner <= 0 or owner == previous[5][i] or owner >= len(old_bots):
            continue
        bp = old_bots[owner]
        if bp < 0 or previous[2][i] not in _N[bp]:
            continue
        competitors = []
        for j, c in enumerate(previous[2]):
            if c not in _N[bp]:
                continue
            if owner <= h:
                eligible = (previous[5][j] == -1 and
                            not (w["goals"] & _BIT[c]))
            else:
                eligible = (previous[5][j] < h + 1 and
                            not (w["bad"] & _BIT[c]))
            if eligible:
                competitors.append(j)
        if i in competitors and len(competitors) > 1:
            first = min(order.index(j) for j in competitors)
            order.remove(i)
            order.insert(first, i)


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 8.4

    parsed = _extract(np.asarray(obs), memory)
    previous = memory.get("_state")
    state = _read_state(parsed, memory)
    signature = parsed[-1]

    world = memory.get("_world")
    terrain_changed = world is None or world["signature"] != signature
    if terrain_changed:
        world = _build_world(signature, world)
        memory["_world"] = world

    if len(world["crate_order"]) != len(state[2]):
        world["crate_order"] = list(range(len(state[2])))
    if len(world["helper_order"]) != len(state[3]):
        world["helper_order"] = list(range(len(state[3])))
    if len(world["enemy_order"]) != len(state[4]):
        world["enemy_order"] = list(range(len(state[4])))

    _learn_order(state, previous, world)

    cached = memory.get("_plan", [])
    expected = memory.get("_expected")
    if cached and not terrain_changed and expected == state:
        action = cached[0]
        memory["_plan"] = cached[1:]
    else:
        memory["_plan"] = []
        tasks = _make_tasks(state, world)
        chosen = []
        best_partial = []
        best_gain = -float("inf")

        for number, task in enumerate(tasks[:3]):
            now = time.perf_counter()
            if now >= deadline - 0.10:
                break
            allowance = 5.0 if number == 0 else 2.0
            search_deadline = min(deadline, now + allowance)
            path, success, gain = _search(
                state, task, world, search_deadline
            )
            if success and path:
                chosen = path
                break
            normalized_gain = gain / max(1.0, task.get("h0", 1))
            if path and normalized_gain > best_gain:
                best_gain = normalized_gain
                best_partial = path
            if path and gain > 1.0:
                chosen = path
                break

        if not chosen:
            chosen = best_partial
        if chosen:
            action = chosen[0]
            memory["_plan"] = chosen[1:]
        else:
            action = _fallback(state, world)

    name = "ACTION%d" % action
    if name not in available:
        choices = [a for a in available if a.startswith("ACTION") and a != "ACTION6"]
        name = choices[0] if choices else available[0]
        try:
            action = int(name[6:])
        except (ValueError, IndexError):
            action = 0
        memory["_plan"] = []

    memory["_state"] = state
    memory["_expected"] = _simulate(state, action, world)
    return name, None, None

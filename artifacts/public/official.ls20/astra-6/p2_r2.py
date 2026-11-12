import math
import time
import heapq
from array import array
from collections import deque
import numpy as np


# Symbol pixels, before rotation.
_SHAPES = (
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 1)),
)
_COLORS = (12, 9, 14, 8)
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_N = 144


def _xy(p):
    return p % 12, p // 12


def _neighbor(p, action):
    x, y = _xy(p)
    dx, dy = _DIRS[action - 1]
    x, y = x + dx, y + dy
    return y * 12 + x if 0 <= x < 12 and 0 <= y < 12 else -1


def _symbol(a):
    a = np.asarray(a)
    for c, color in enumerate(_COLORS):
        if not np.any(a == color) or not np.all((a == color) | (a == 5)):
            continue
        mask = a == color
        for s, shape in enumerate(_SHAPES):
            shape = np.asarray(shape, dtype=bool)
            for r in range(4):
                if np.array_equal(mask, np.rot90(shape, -r)):
                    return s * 16 + c * 4 + r
    return None


def _inspect(obs):
    player = None
    for y in range(0, 60, 5):
        for x in range(4, 60, 5):
            if (np.all(obs[y:y + 2, x:x + 5] == 12)
                    and np.all(obs[y + 2:y + 5, x:x + 5] == 9)):
                player = (y // 5) * 12 + (x - 4) // 5
                break
        if player is not None:
            break

    attr = _symbol(obs[55:61:2, 3:9:2])
    energy = int(np.count_nonzero(obs[61, 13:55] == 11))
    capacity = 0
    for x in range(13, 56):
        if int(obs[61, x]) in (3, 11):
            capacity += 1
        else:
            break
    capacity = capacity or 42
    lives = sum(int(obs[61, 56 + 3 * i]) == 8 for i in range(3))
    fog = int(np.count_nonzero(obs[:60] == 5)) > 850

    visible = np.ones((64, 64), dtype=bool)
    if fog and player is not None:
        cx, cy = _xy(player)
        yy, xx = np.ogrid[:64, :64]
        visible = ((xx - (4 + 5 * cx + 1.5)) ** 2
                   + (yy - (5 * cy + 1.5)) ** 2 <= 400.0001)

    patterns = (
        np.array(((0, 3, 3), (3, 0, 0), (3, 0, 3))),
        np.array(((9, 14, 14), (9, 0, 8), (12, 12, 8))),
        np.array(((3, 0, 3), (1, 0, 0), (3, 1, 3))),
    )
    goals, pickups, switches, barriers = {}, set(), [], set()
    cells, fully_seen = {}, set()

    for p in range(_N):
        cx, cy = _xy(p)
        x, y = 4 + 5 * cx, 5 * cy
        tile = obs[y:y + 5, x:x + 5]
        vis = visible[y:y + 5, x:x + 5].copy()
        # The lower-left symbol display is not part of the board.
        if x < 12 and y >= 50:
            for j in range(5):
                for i in range(5):
                    if x + i < 12 and y + j >= 52:
                        vis[j, i] = False
        if p == player:
            cells[p] = 0
        elif np.count_nonzero((tile == 4) & vis) >= 3:
            cells[p] = 1
        elif np.count_nonzero(vis) >= 20:
            cells[p] = 0

        if np.all(visible[y + 1:y + 4, x + 1:x + 4]):
            fully_seen.add(p)
            center = obs[y + 1:y + 4, x + 1:x + 4]
            a = _symbol(center)
            if a is not None and p != player:
                # Goal symbols have a black backing.
                if (int(tile[0, 0]) == 5 and int(tile[4, 4]) == 5):
                    goals[p] = a
                    barriers.add(p)
            if np.array_equal(center, np.array(
                    ((11, 11, 11), (11, 3, 11), (11, 11, 11)))):
                pickups.add(p)
            for kind, pattern in enumerate(patterns):
                if any(np.array_equal(center, np.rot90(pattern, r))
                       for r in range(4)):
                    switches.append((kind, p))
                    break

        if x >= 1 and y >= 1 and y + 5 < 64 and x + 5 < 64:
            if (np.all(obs[y - 1, x - 1:x + 6] == 5)
                    and np.all(obs[y + 5, x - 1:x + 6] == 5)
                    and np.all(obs[y - 1:y + 6, x - 1] == 5)
                    and np.all(obs[y - 1:y + 6, x + 5] == 5)
                    and np.any(tile == 3)):
                barriers.add(p)

    pushers = {}
    for x in range(4, 60, 5):
        for y in range(60):
            if np.all(obs[y, x:x + 5] == 1):
                cx = (x - 4) // 5
                if y % 5 == 4:
                    cy, direction = y // 5 + 1, (0, 1)
                elif y % 5 == 0:
                    cy, direction = y // 5 - 1, (0, -1)
                else:
                    continue
                if 0 <= cy < 12:
                    pushers[cy * 12 + cx] = direction
    for y in range(0, 60, 5):
        for x in range(64):
            if np.all(obs[y:y + 5, x] == 1):
                if x % 5 == 3:
                    cx, direction = (x - 3) // 5, (1, 0)
                elif x % 5 == 4:
                    cx, direction = (x - 4) // 5 - 1, (-1, 0)
                else:
                    continue
                if 0 <= cx < 12:
                    pushers[(y // 5) * 12 + cx] = direction

    return (player, attr, energy, capacity, lives, fog, visible,
            cells, fully_seen, goals, pickups, switches, barriers, pushers)


def _fit_cycle(history, old, board):
    history = history[-40:]
    if not history:
        return old
    if len({p for _, p in history}) == 1:
        return ((history[-1][1],), 0)

    # Prefer a directly observed complete period.
    lo, hi = history[0][0], history[-1][0]
    for period in range(2, min(33, hi - lo - 1)):
        values = {}
        good = True
        for t, p in history:
            r = t % period
            if r in values and values[r] != p:
                good = False
                break
            values[r] = p
        if good and len(values) == period:
            return (tuple(values[i] for i in range(period)), 0)

    if old:
        cycle, offset = old
        if len(cycle) > 1 and all(
                cycle[(t + offset) % len(cycle)] == p for t, p in history):
            return old

    first_t, first_p = history[0]
    xs = [p % 12 for _, p in history]
    ys = [p // 12 for _, p in history]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    best = None
    best_score = float("inf")

    def consider(cycle, rectangle=False):
        nonlocal best, best_score
        cycle = tuple(cycle)
        length = len(cycle)
        if not length:
            return
        positions = [i for i, p in enumerate(cycle) if p == first_p]
        for i in positions:
            offset = (i - first_t) % length
            if not all(cycle[(t + offset) % length] == p
                       for t, p in history):
                continue
            score = 0.25 * abs(length - 8) + (0.15 if rectangle else 0)
            score += 5 * sum(board[p] == 1 for p in set(cycle))
            score += 0.02 * sum(board[p] < 0 for p in set(cycle))
            score += 0.001 * length
            if len(history) > 1:
                t, p = history[-1]
                t0, p0 = history[-2]
                if t == t0 + 1:
                    nxt = cycle[(t + 1 + offset) % length]
                    dx, dy = p % 12 - p0 % 12, p // 12 - p0 // 12
                    if (nxt % 12 - p % 12, nxt // 12 - p // 12) != (dx, dy):
                        score += 0.04
            if score < best_score:
                best_score, best = score, (cycle, offset)

    if miny == maxy:
        for length in range(2, 10):
            for left in range(max(0, maxx - length + 1),
                              min(minx, 12 - length) + 1):
                line = [miny * 12 + x for x in range(left, left + length)]
                consider(line + line[-2:0:-1])
    if minx == maxx:
        for length in range(2, 10):
            for top in range(max(0, maxy - length + 1),
                             min(miny, 12 - length) + 1):
                line = [y * 12 + minx for y in range(top, top + length)]
                consider(line + line[-2:0:-1])

    for width in range(2, 7):
        for height in range(2, 7):
            for left in range(max(0, maxx - width + 1),
                              min(minx, 12 - width) + 1):
                right = left + width - 1
                for top in range(max(0, maxy - height + 1),
                                 min(miny, 12 - height) + 1):
                    bottom = top + height - 1
                    if any(not (x in (left, right) or y in (top, bottom))
                           for x, y in zip(xs, ys)):
                        continue
                    cycle = [top * 12 + x for x in range(left, right + 1)]
                    cycle += [y * 12 + right for y in range(top + 1, bottom + 1)]
                    cycle += [bottom * 12 + x for x in range(right - 1, left - 1, -1)]
                    cycle += [y * 12 + left for y in range(bottom - 1, top, -1)]
                    consider(cycle, True)
                    consider(cycle[::-1], True)

    return best if best is not None else ((history[-1][1],), 0)


def _update_tracks(m, detections, player, seen, obs):
    tick = m["tick"]
    tracks = m["tracks"]
    matched = set()
    pending = list(detections)

    # Exact predicted matches first, then one-cell moves.
    for exact in (True, False):
        remaining = []
        for kind, p in pending:
            choices = []
            for i, tr in enumerate(tracks):
                if i in matched or tr["kind"] != kind:
                    continue
                cyc, off = tr["cycle"]
                predicted = cyc[(tick + off) % len(cyc)]
                last = tr["history"][-1][1]
                distance = abs(p % 12 - last % 12) + abs(p // 12 - last // 12)
                if (exact and p == predicted) or (not exact and distance <= 1):
                    choices.append((0 if p == predicted else 1, distance, i))
            if choices:
                _, _, i = min(choices)
                matched.add(i)
                tracks[i]["history"].append((tick, p))
            else:
                remaining.append((kind, p))
        pending = remaining

    for kind, p in pending:
        tracks.append({"kind": kind, "history": [(tick, p)],
                       "cycle": ((p,), 0)})
        matched.add(len(tracks) - 1)

    for i, tr in enumerate(tracks):
        if i not in matched:
            last_t, last_p = tr["history"][-1]
            cyc, off = tr["cycle"]
            predicted = cyc[(tick + off) % len(cyc)]
            if predicted == player:
                tr["history"].append((tick, player))
            elif last_p in seen and last_p != player and last_t == tick - 1:
                d = abs(player % 12 - last_p % 12) + abs(player // 12 - last_p // 12)
                if d <= 1:
                    x, y = 4 + 5 * (last_p % 12), 5 * (last_p // 12)
                    if np.all(obs[y + 1:y + 4, x + 1:x + 4] == 3):
                        tr["history"].append((tick, player))
        # Keep only the most recent observation for each tick.
        hist = dict(tr["history"][-48:])
        tr["history"] = sorted(hist.items())
        tr["cycle"] = _fit_cycle(tr["history"], tr.get("cycle"), m["board"])


def _shift_table(inc, inverse=False):
    ds, dc, dr = inc // 16, (inc // 4) % 4, inc % 4
    sign = -1 if inverse else 1
    return tuple(
        ((a // 16 + sign * ds) % 6) * 16
        + (((a // 4) % 4 + sign * dc) % 4) * 4
        + ((a % 4 + sign * dr) % 4)
        for a in range(96)
    )


def _build_model(m, player, attr, energy):
    board = m["board"]
    goals = sorted((p, a) for p, a in m["goals"].items() if p not in m["done"])
    pickups = sorted(m["pickups"] - m["used"])
    period = 1
    for tr in m["tracks"]:
        length = len(tr["cycle"][0])
        period = period * length // math.gcd(period, length)
    period = min(period, 32)

    walkable = {p for p in range(_N) if board[p] == 0}
    walkable.add(player)
    walkable.update(p for p, _ in goals)
    walkable.update(pickups)
    walkable.update(m["pushers"])

    endpoints = {}
    for p, (dx, dy) in m["pushers"].items():
        if p in m["pushend"]:
            endpoints[p] = m["pushend"][p]
            continue
        x, y = _xy(p)
        endpoint = p
        for distance in range(1, 12):
            xx, yy = x + dx * distance, y + dy * distance
            if not (0 <= xx < 12 and 0 <= yy < 12):
                break
            q = yy * 12 + xx
            if board[q] == 1 or q in m["barriers"]:
                break
            endpoint = q
        endpoints[p] = endpoint
    walkable.update(endpoints.values())

    edges = [[] for _ in range(period * _N)]
    reverse = [[] for _ in range(period * _N)]
    plus, minus = {}, {}
    tick = m["tick"]

    for phase in range(period):
        next_phase = (phase + 1) % period
        modifications = {}
        for tr in m["tracks"]:
            cyc, off = tr["cycle"]
            p = cyc[(tick + phase + 1 + off) % len(cyc)]
            counts = modifications.setdefault(p, [0, 0, 0])
            counts[tr["kind"]] += 1
        for p in walkable:
            for action in range(1, 5):
                dest = _neighbor(p, action)
                if dest not in walkable or (dest >= 0 and board[dest] == 1):
                    continue
                land = endpoints.get(dest, dest)
                pushed = land != dest
                counts = list(modifications.get(dest, (0, 0, 0)))
                if pushed:
                    other = modifications.get(land, (0, 0, 0))
                    counts = [counts[i] + other[i] for i in range(3)]
                inc = (counts[0] % 6) * 16 + (counts[1] % 4) * 4 + counts[2] % 4
                if inc not in plus:
                    plus[inc] = _shift_table(inc)
                    minus[inc] = _shift_table(inc, True)
                oldv = phase * _N + p
                newv = next_phase * _N + land
                edges[oldv].append((action, dest, land, inc, pushed, next_phase))
                reverse[newv].append((oldv * 96, minus[inc]))

    return {
        "period": period, "edges": edges, "reverse": reverse, "plus": plus,
        "goals": goals, "gmap": {p: (i, a) for i, (p, a) in enumerate(goals)},
        "pickups": pickups, "pmap": {p: 1 << i for i, p in enumerate(pickups)},
        "capacity": m["capacity"], "cost": m["cost"],
        "start": player * 96 + attr, "energy": energy,
        "board": board,
    }


def _transitions(model, node, remaining, used, energy):
    a = node % 96
    v = node // 96
    p = v % _N
    phase = v // _N
    gmap = model["gmap"]
    pmap = model["pmap"]

    for action, dest, land, inc, pushed, next_phase in model["edges"][v]:
        goal = gmap.get(dest)
        if goal is not None:
            i, required = goal
            if remaining & (1 << i) and a != required:
                continue

        na = model["plus"][inc][a]
        newly_used = pmap.get(dest, 0)
        if pushed:
            newly_used |= pmap.get(land, 0)
        newly_used &= ~used
        nu = used | newly_used
        ne = model["capacity"] if newly_used else energy - model["cost"]
        nr = remaining
        goal = gmap.get(land)
        if not pushed and goal is not None and na == goal[1]:
            nr &= ~(1 << goal[0])
        if ne < 0 and nr:
            continue
        nn = (next_phase * _N + land) * 96 + na
        yield action, nn, nr, nu, max(0, ne), True

    # A launcher may deliver the player onto a goal without checking victory.
    goal = gmap.get(p)
    if goal is not None and remaining & (1 << goal[0]) and a == goal[1]:
        legal_actions = {edge[0] for edge in model["edges"][v]}
        for action in range(1, 5):
            q = _neighbor(p, action)
            other = gmap.get(q)
            blocked = action not in legal_actions
            if other is not None and remaining & (1 << other[0]) and a != other[1]:
                blocked = True
            if blocked:
                nr = remaining & ~(1 << goal[0])
                ne = energy - model["cost"]
                if ne >= 0 or not nr:
                    yield action, node, nr, used, max(0, ne), False
                break


def _goal_distances(model, deadline):
    size = model["period"] * _N * 96
    result = []
    reverse = model["reverse"]
    for p, a in model["goals"]:
        dist = array("i", [-1]) * size
        queue = deque()
        for phase in range(model["period"]):
            n = (phase * _N + p) * 96 + a
            dist[n] = 0
            queue.append(n)
        count = 0
        while queue:
            n = queue.popleft()
            a0 = n % 96
            nd = dist[n] + 1
            for base, inverse in reverse[n // 96]:
                prev = base + inverse[a0]
                if dist[prev] < 0:
                    dist[prev] = nd
                    queue.append(prev)
            count += 1
            if count % 4096 == 0 and time.monotonic() > deadline:
                return result
        result.append(dist)
    return result


def _navigation_distances(model, target):
    dist = [-1] * len(model["edges"])
    queue = deque()
    for phase in range(model["period"]):
        v = phase * _N + target
        dist[v] = 0
        queue.append(v)
    while queue:
        v = queue.popleft()
        nd = dist[v] + 1
        for base, _ in model["reverse"][v]:
            prev = base // 96
            if dist[prev] < 0:
                dist[prev] = nd
                queue.append(prev)
    return dist


def _solve(model, distances, deadline, target_mask=None, nav=None):
    count_goals = len(model["goals"])
    initial_remaining = (1 << count_goals) - 1
    if target_mask is None:
        target_mask = initial_remaining
    start = model["start"]
    infinity = 10 ** 7

    if nav is not None:
        nd = _navigation_distances(model, nav)
        if nd[start // 96] < 0:
            return None

        def heuristic(node, remaining):
            d = nd[node // 96]
            return d if d >= 0 else infinity
    else:
        if not target_mask:
            return None
        between = [[infinity] * count_goals for _ in range(count_goals)]
        for i, (p, a) in enumerate(model["goals"]):
            for j in range(len(distances)):
                candidates = [
                    distances[j][(phase * _N + p) * 96 + a]
                    for phase in range(model["period"])
                ]
                candidates = [d for d in candidates if d >= 0]
                if candidates:
                    between[i][j] = min(candidates)
        tails = {}

        def tail(i, mask):
            key = i, mask
            if key in tails:
                return tails[key]
            if not mask:
                return 0
            best = infinity
            for j in range(count_goals):
                if mask & (1 << j):
                    best = min(best, between[i][j] + tail(j, mask ^ (1 << j)))
            tails[key] = best
            return best

        options = {}
        for mask in range(1 << count_goals):
            mask &= target_mask
            if mask not in options:
                options[mask] = [
                    (distances[i], tail(i, mask ^ (1 << i)))
                    for i in range(len(distances)) if mask & (1 << i)
                ]

        def heuristic(node, remaining):
            mask = remaining & target_mask
            if not mask:
                return 0
            best = infinity
            for dist, rest in options.get(mask, ()):
                d = dist[node]
                if d >= 0:
                    best = min(best, d + rest)
            return best

    h0 = heuristic(start, initial_remaining)
    if h0 >= infinity:
        return None

    # Record: node, remaining, used refills, energy, parent, action, g, moved.
    records = [(start, initial_remaining, 0, model["energy"], -1, 0, 0, False)]
    heap = [(h0, 0, -model["energy"], 0)]
    labels = {(start, initial_remaining, 0): [(0, model["energy"])]}
    best_partial = None
    best_partial_score = (target_mask.bit_count(), infinity)

    def reconstruct(index):
        path = []
        while records[index][4] >= 0:
            path.append(records[index])
            index = records[index][4]
        path.reverse()
        return path

    while heap and len(records) < 240000:
        _, _, _, index = heapq.heappop(heap)
        node, rem, used, energy, parent, action, g, moved = records[index]
        if nav is not None:
            if node // 96 % _N == nav and g:
                return reconstruct(index)
        elif not (rem & target_mask):
            return reconstruct(index)

        key = node, rem, used
        if any(gg <= g and ee >= energy and (gg < g or ee > energy)
               for gg, ee in labels.get(key, ())):
            continue

        if nav is None:
            progress = (rem & target_mask).bit_count()
            score = progress, heuristic(node, rem)
            if progress < target_mask.bit_count() and score < best_partial_score:
                best_partial_score, best_partial = score, index

        for action, nn, nr, nu, ne, success in _transitions(model, node, rem, used, energy):
            ng = g + 1
            key = nn, nr, nu
            old = labels.get(key, ())
            if any(gg <= ng and ee >= ne for gg, ee in old):
                continue
            h = heuristic(nn, nr)
            if h >= infinity:
                continue
            labels[key] = [(gg, ee) for gg, ee in old
                           if not (ng <= gg and ne >= ee)] + [(ng, ne)]
            records.append((nn, nr, nu, ne, index, action, ng, success))
            ni = len(records) - 1
            heapq.heappush(heap, (ng + h, -ng, -ne, ni))

        if len(records) % 256 < 5 and time.monotonic() > deadline:
            break

    if best_partial is not None:
        return reconstruct(best_partial)
    return None


def _physical_distances(model):
    p = model["start"] // 96 % _N
    dist = {p: 0}
    queue = deque([p])
    while queue:
        p = queue.popleft()
        for edge in model["edges"][p]:
            q = edge[2]
            if q not in dist:
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def _encode_plan(path, model, m):
    result = []
    for node, rem, used, energy, parent, action, g, success in path:
        done = set(m["done"])
        consumed = set(m["used"])
        done.update(p for i, (p, _) in enumerate(model["goals"])
                    if not rem & (1 << i))
        consumed.update(p for i, p in enumerate(model["pickups"])
                        if used & (1 << i))
        result.append({
            "action": action, "p": node // 96 % _N, "attr": node % 96,
            "energy": energy, "done": done, "used": consumed,
            "success": success,
        })
    return result


def act(obs, available, memory):
    started = time.monotonic()
    obs = np.asarray(obs)
    names = {str(a).upper().split(".")[-1] for a in available}
    m = memory.setdefault("ls20_player", {
        "board": [-1] * _N, "goals": {}, "done": set(),
        "pickups": set(), "used": set(), "barriers": set(),
        "pushers": {}, "pushend": {}, "tracks": [], "tick": 0,
        "cost": 2, "capacity": 42, "plan": [], "visits": {},
        "blocked": set(), "resets": 0,
    })

    (player, attr, energy, capacity, lives, fog, visible, cells, seen,
     goals, pickups, switches, barriers, pushers) = _inspect(obs)

    if player is None:
        action = "RESET" if "RESET" in names else next(
            (f"ACTION{i}" for i in range(1, 5) if f"ACTION{i}" in names),
            "ACTION1")
        return action, None, None
    if attr is None:
        attr = m.get("last", {}).get("attr", 0)

    last = m.get("last")
    expected = m.pop("expected", None)
    matched = False
    restarted = bool(last and (lives < last["lives"] or last["action"] == 0))

    if restarted:
        m["done"].clear()
        m["used"].clear()
        m["tracks"] = []
        m["tick"] = 0
        m["plan"] = []
    elif last:
        matched = bool(expected and expected["p"] == player
                       and expected["attr"] == attr)
        moved = player != last["p"]
        if moved or (matched and expected["success"]):
            m["tick"] += 1
        if matched:
            m["done"].update(expected["done"])
            m["used"].update(expected["used"])
            if energy < expected["energy"]:
                m["plan"] = []
        else:
            m["plan"] = []

        delta = last["energy"] - energy
        if delta in (1, 2) and not restarted:
            m["cost"] = delta

        if last["action"] in (1, 2, 3, 4):
            target = _neighbor(last["p"], last["action"])
            if (player == last["p"] and expected and expected["p"] != player
                    and target >= 0 and target not in m["goals"]):
                m["blocked"].add(target)
            if target in m["pushers"] and moved and player != target:
                m["pushend"][target] = player

    m["capacity"] = capacity
    for p, value in cells.items():
        m["board"][p] = value
    for p in m["blocked"]:
        m["board"][p] = 1
    m["board"][player] = 0
    m["blocked"].discard(player)

    m["goals"].update(goals)
    m["barriers"].update(barriers)
    m["barriers"].update(goals)
    m["pickups"].update(pickups)
    m["pushers"].update(pushers)

    # Visible, uncovered objects provide direct evidence of collection.
    for p in list(m["goals"]):
        if p in goals:
            m["done"].discard(p)
        elif p in seen and p != player:
            x, y = 4 + 5 * (p % 12), 5 * (p // 12)
            center = obs[y + 1:y + 4, x + 1:x + 4]
            if not any(np.any(center == c) for c in _COLORS):
                m["done"].add(p)
    for p in m["pickups"]:
        if p in pickups:
            m["used"].discard(p)
        elif p in seen and p != player:
            m["used"].add(p)

    for p in goals:
        m["board"][p] = 0
    for p in pickups:
        m["board"][p] = 0
    for p in pushers:
        m["board"][p] = 0

    _update_tracks(m, switches, player, seen, obs)
    m["visits"][player] = m["visits"].get(player, 0) + 1

    signature = (
        tuple(m["board"]), tuple(sorted(m["goals"].items())),
        tuple(sorted(m["pickups"])), tuple(sorted(m["barriers"])),
        tuple(sorted(m["pushers"].items())), tuple(sorted(m["pushend"].items())),
        tuple((tr["kind"], tr["cycle"]) for tr in m["tracks"]),
        m["cost"], m["capacity"],
    )

    plan = m["plan"] if matched and m.get("signature") == signature else []
    model = None

    if not plan:
        model = _build_model(m, player, attr, energy)
        path = None
        if model["goals"]:
            distances = _goal_distances(model, started + 5.0)
            reachable = 0
            for i, dist in enumerate(distances):
                if dist[model["start"]] >= 0:
                    reachable |= 1 << i
            if reachable:
                path = _solve(model, distances, min(started + 6.8,
                                                    time.monotonic() + 4.0),
                              target_mask=reachable)

        if not path and time.monotonic() < started + 8.1:
            pdist = _physical_distances(model)
            candidates = []
            unknown = [p for p in range(_N) if m["board"][p] < 0]

            # Refills become urgent when little energy remains.
            for p in model["pickups"]:
                d = pdist.get(p)
                if d is not None and d > 0:
                    score = (18.0 if energy <= 12 else 3.0) / (d + 1)
                    candidates.append((score, p))

            if unknown:
                for p, d in pdist.items():
                    if d <= 0:
                        continue
                    x, y = _xy(p)
                    gain = sum((q % 12 - x) ** 2 + (q // 12 - y) ** 2 <= 12
                               for q in unknown)
                    if gain:
                        score = gain / (d + 3.0)
                        score /= 1 + 0.15 * m["visits"].get(p, 0)
                        if d * m["cost"] > energy:
                            score *= 0.05
                        candidates.append((score, p))

            # A short observation move also uncovers a switch hidden by the player.
            if not candidates:
                for p, d in pdist.items():
                    if 0 < d <= 3:
                        score = 0.2 / (d + m["visits"].get(p, 0) + 1)
                        candidates.append((score, p))

            candidates.sort(reverse=True)
            tried = set()
            for _, target in candidates[:8]:
                if target in tried or time.monotonic() > started + 8.5:
                    continue
                tried.add(target)
                path = _solve(model, [], min(started + 8.7,
                                            time.monotonic() + 0.65),
                              nav=target)
                if path:
                    break

        if path:
            plan = _encode_plan(path, model, m)
        else:
            plan = []

        m["signature"] = signature
        m["plan"] = plan

    if plan:
        step = plan.pop(0)
        action_id = step["action"]
        action_name = f"ACTION{action_id}"
        if action_name in names or not names:
            m["expected"] = step
        else:
            action_id = 0 if "RESET" in names else 1
            action_name = "RESET" if action_id == 0 else "ACTION1"
            m["plan"] = []
    else:
        action_id = None
        if model is None:
            model = _build_model(m, player, attr, energy)
        rem = (1 << len(model["goals"])) - 1
        choices = []
        for tr in _transitions(model, model["start"], rem, 0, energy):
            action, nn, nr, nu, ne, success = tr
            q = nn // 96 % _N
            score = (100 * (rem.bit_count() - nr.bit_count())
                     + 5 * bool(nu) - m["visits"].get(q, 0))
            choices.append((score, tr))
        if choices and energy >= m["cost"]:
            _, tr = max(choices, key=lambda item: item[0])
            action_id, nn, nr, nu, ne, success = tr
            record = (nn, nr, nu, ne, 0, action_id, 1, success)
            m["expected"] = _encode_plan([record], model, m)[0]
        elif "RESET" in names:
            action_id = 0
            m["resets"] += 1
        else:
            action_id = next((i for i in range(1, 5)
                              if f"ACTION{i}" in names), 1)
        action_name = "RESET" if action_id == 0 else f"ACTION{action_id}"

    m["last"] = {
        "p": player, "attr": attr, "energy": energy,
        "lives": lives, "action": action_id,
    }
    return action_name, None, None

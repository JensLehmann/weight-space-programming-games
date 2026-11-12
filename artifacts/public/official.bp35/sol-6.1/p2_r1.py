import numpy as np
import heapq
import time
from collections import deque

EMPTY, WALL, BREAK, GRAVITY, SOLID, GHOST, GROW, SPIKE, GOAL = range(9)
UNKNOWN = -1
W = 11


def _decode(obs, partial=False):
    ys, xs = np.where(obs[:63] == 9)
    if len(ys) == 0:
        return None

    top = int(ys.min()) - 1
    px = int(xs.min()) // 6
    tiles = {}

    for r in range(-12, 13):
        y = top + 6 * r
        if partial:
            if y + 5 < 0 or y + 1 >= 63:
                continue
        elif y < 0 or y + 5 >= 63:
            continue

        for x in range(W):
            left = x * 6
            if not partial and left + 5 >= 64:
                continue
            ya, yb = max(0, y + 1), min(63, y + 6)
            xa, xb = left + 1, min(64, left + 6)
            if ya >= yb or xa >= xb:
                continue

            a = obs[ya:yb, xa:xb]
            colors = set(map(int, a.ravel()))
            if 9 in colors:
                t = EMPTY
            elif 14 in colors:
                t = BREAK
            elif 8 in colors:
                t = GRAVITY
            elif 7 in colors:
                t = GOAL
            elif 11 in colors:
                t = SPIKE
            elif 12 in colors:
                threshold = max(4, int(a.size * .45))
                t = SOLID if np.count_nonzero(a == 12) >= threshold else GHOST
            elif 15 in colors:
                t = GROW if 3 in colors else WALL
            else:
                cy = min(62, max(0, y + 3))
                cx = min(63, left + 3)
                t = EMPTY if int(obs[cy, cx]) == 10 else WALL
            tiles[x, r] = t

    tiles[px, 0] = EMPTY
    return px, top, tiles


def _locate(memory, px, top, view):
    board = memory.get("board", {})
    pending = memory.get("expected")

    if not board or pending is None:
        d = 1 if top == 26 else -1
        return 0, d, False

    ex, ey, d, uncertain, old_y = pending
    if not uncertain:
        return ey if px == ex else old_y, d, False

    best = None
    for shift in range(0, 85):
        candidate = old_y + d * shift
        good = bad = count = 0
        for (x, r), t in view.items():
            old = board.get((x, candidate + r))
            if old is None:
                continue
            count += 1
            weight = 1 if t in (EMPTY, WALL) and old in (EMPTY, WALL) else 4
            if old == t:
                good += weight
            elif old in (SOLID, GHOST) and t in (SOLID, GHOST):
                bad += .3
            elif old == GROW or t == GROW:
                bad += .5
            else:
                bad += weight
        if count < 12:
            continue
        score = good - 4 * bad + .08 * count
        score -= .02 * abs(candidate - ey)
        if best is None or score > best[0]:
            best = score, candidate, count, bad

    if best is not None and best[3] <= max(5, best[2] * .14):
        return best[1], d, False
    return ey, d, True


class Model:
    def __init__(self, board, position, direction):
        self.lo = min(y for x, y in board) - 1
        self.hi = max(y for x, y in board) + 1
        self.height = self.hi - self.lo + 1
        self.n = self.height * W
        self.base = [UNKNOWN] * self.n
        self.mutable = []
        self.gravities = []
        self.goals = []
        growth = 0

        for y in range(self.lo, self.hi + 1):
            self.base[self.idx(0, y)] = WALL
            self.base[self.idx(10, y)] = WALL

        for (x, y), t in board.items():
            if not 0 <= x < W:
                continue
            q = self.idx(x, y)
            if t == GROW:
                growth |= 1 << q
                self.base[q] = EMPTY
            else:
                self.base[q] = t
            if t in (BREAK, SOLID, GHOST):
                self.mutable.append(q)
            if t == GRAVITY:
                self.gravities.append(q)
            if t == GOAL:
                self.goals.append(q)

        p = self.idx(*position)
        self.base[p] = EMPTY
        self.start = p, direction, 0, 0, growth
        self.has_growth = bool(growth)
        self.known_lo = self.lo + 1
        self.known_hi = self.hi - 1

    def idx(self, x, y):
        return (y - self.lo) * W + x

    def xy(self, q):
        return q % W, q // W + self.lo

    def tile(self, q, s):
        if not 0 <= q < self.n:
            return UNKNOWN
        p, d, removed, toggled, growth = s
        bit = 1 << q
        if growth & bit:
            return GROW
        t = self.base[q]
        if t in (BREAK, GRAVITY) and removed & bit:
            return EMPTY
        if t in (SOLID, GHOST) and toggled & bit:
            return GHOST if t == SOLID else SOLID
        return t

    def visible(self, q, s):
        if not 0 <= q < self.n:
            return False
        x, y = self.xy(q)
        _, py = self.xy(s[0])
        anchor = 36 if s[1] == -1 else 26
        row = anchor + 6 * (y - py) + 3
        return 0 <= row < 63 and 0 <= x < W

    def fall(self, q, s):
        p, d, removed, toggled, growth = s
        cur = q
        if self.tile(cur, s) == UNKNOWN:
            return (cur, d, removed, toggled, growth), "unknown"

        for _ in range(self.height + 2):
            nxt = cur + d * W
            t = self.tile(nxt, s)
            if t == UNKNOWN:
                return (nxt, d, removed, toggled, growth), "unknown"
            if t == GOAL:
                return (nxt, d, removed, toggled, growth), "win"
            if t == SPIKE:
                return None, "dead"
            if t in (EMPTY, GHOST):
                cur = nxt
            else:
                return (cur, d, removed, toggled, growth), "ok"
        return None, "dead"

    def move(self, s, dx):
        p, d, removed, toggled, growth = s
        x, _ = self.xy(p)
        if not 0 <= x + dx < W:
            return None, "blocked"
        q = p + dx
        t = self.tile(q, s)
        if t == GOAL:
            return (q, d, removed, toggled, growth), "win"
        if t == UNKNOWN:
            return (q, d, removed, toggled, growth), "unknown"
        if t not in (EMPTY, GHOST):
            return None, "blocked"
        return self.fall(q, s)

    def click(self, s, q):
        p, d, removed, toggled, growth = s
        t = self.tile(q, s)
        if t not in (BREAK, GRAVITY, SOLID, GHOST, GROW):
            return None, "blocked"

        if t == GRAVITY:
            nd = -d
            support = p + nd * W
            before = self.tile(support, s)
            ns = p, nd, removed | (1 << q), toggled, growth
            if before in (WALL, BREAK, GRAVITY, SOLID, GROW):
                return ns, "ok"
            return self.fall(support, ns)

        if t == BREAK:
            removed |= 1 << q
        elif t in (SOLID, GHOST):
            toggled ^= 1 << q
        else:
            x, y = self.xy(q)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and self.lo <= ny <= self.hi):
                    continue
                k = self.idx(nx, ny)
                if k != p and self.tile(k, s) == EMPTY:
                    growth |= 1 << k
            growth &= ~(1 << q)

        ns = p, d, removed, toggled, growth
        if q == p + d * W:
            return self.fall(q, ns)
        return ns, "ok"

    def successors(self, s, basic=False):
        p, d = s[:2]
        px, py = self.xy(p)

        for dx, a in ((-1, 3), (1, 4)):
            ns, out = self.move(s, dx)
            if ns is not None:
                yield [((a, None), ns, out)], ns, out

        candidates = set()
        for q in (p + d * W, p - 1, p + 1):
            if 0 <= q < self.n:
                candidates.add(q)

        for q in self.gravities:
            if self.tile(q, s) == GRAVITY and self.visible(q, s):
                candidates.add(q)

        for q in self.mutable:
            if not self.visible(q, s):
                continue
            x, y = self.xy(q)
            t = self.tile(q, s)
            if basic:
                if t == GHOST and abs(x - px) <= 5:
                    if (y - py) * d >= 0 or abs(y - py) <= 1:
                        candidates.add(q)
            else:
                if t == GHOST:
                    candidates.add(q)
                elif t in (BREAK, SOLID) and abs(x - px) <= 1:
                    if abs(y - py) <= 2:
                        candidates.add(q)

        bits = s[4]
        while bits:
            b = bits & -bits
            q = b.bit_length() - 1
            bits ^= b
            x, y = self.xy(q)
            if self.visible(q, s) and (
                basic
                or abs(x - px) + abs(y - py) <= 3
                or q == p + d * W
            ):
                candidates.add(q)

        iterator = candidates if basic else sorted(candidates)
        for q in iterator:
            if not self.visible(q, s):
                continue
            ns, out = self.click(s, q)
            if ns is not None and ns != s:
                yield [((6, q), ns, out)], ns, out

    def growth_macros(self, s):
        if not s[4]:
            return

        px, py = self.xy(s[0])
        queue = deque()
        parent = {}
        distance = {}
        bits = s[4]

        while bits:
            b = bits & -bits
            q = b.bit_length() - 1
            bits ^= b
            if self.visible(q, s):
                parent[q] = None
                distance[q] = 0
                queue.append(q)

        while queue:
            q = queue.popleft()
            if distance[q] >= 8:
                continue
            x, y = self.xy(q)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not (0 < nx < 10 and self.lo <= ny <= self.hi):
                    continue
                k = self.idx(nx, ny)
                if k in parent or k == s[0] or not self.visible(k, s):
                    continue
                if self.tile(k, s) != EMPTY:
                    continue
                parent[k] = q
                distance[k] = distance[q] + 1
                queue.append(k)

        targets = []
        for dx in (-1, 0, 1):
            x = px + dx
            if not 1 <= x <= 9:
                continue
            for amount in (2, 3, 4, 5, 6):
                y = py + s[1] * amount
                if self.lo <= y <= self.hi:
                    q = self.idx(x, y)
                    if q in parent and distance[q] > 0:
                        targets.append((q, dx, False))

        for goal in self.goals:
            gx, gy = self.xy(goal)
            for dx in (-3, -2, -1, 0, 1):
                for dy in (-2, -1, 1, 2):
                    x, y = gx + dx, gy + dy
                    if not (1 <= x <= 9 and self.lo <= y <= self.hi):
                        continue
                    q = self.idx(x, y)
                    if q in parent and 0 < distance[q] <= 5:
                        targets.append((q, 0, True))

        targets.sort(key=lambda item: (
            distance[item[0]], -abs(self.xy(item[0])[1] - py)
        ))
        emitted = set()

        for target, dx, bridge in targets[:22]:
            chain = []
            q = target
            while parent[q] is not None:
                chain.append(parent[q])
                q = parent[q]
            chain.reverse()

            ns = s
            sequence = []
            valid = True
            for source in chain:
                if self.tile(source, ns) != GROW or not self.visible(source, ns):
                    valid = False
                    break
                nxt, out = self.click(ns, source)
                if nxt is None or out != "ok":
                    valid = False
                    break
                sequence.append(((6, source), nxt, out))
                ns = nxt

            if not valid or not sequence:
                continue

            if bridge:
                if ns not in emitted:
                    emitted.add(ns)
                    yield sequence, ns, "ok"
                continue

            if ns[0] != s[0]:
                if ns not in emitted:
                    emitted.add(ns)
                    yield sequence, ns, "ok"
                continue

            if dx:
                landed, out = self.move(ns, dx)
                a = (3 if dx < 0 else 4, None)
            else:
                support = ns[0] + ns[1] * W
                if self.tile(support, ns) not in (BREAK, SOLID, GROW):
                    continue
                landed, out = self.click(ns, support)
                a = (6, support)

            if landed is None or out not in ("ok", "win"):
                continue
            if landed[0] == s[0] or landed in emitted:
                continue
            emitted.add(landed)
            yield sequence + [(a, landed, out)], landed, out

    def distances(self, direction, goal_mode):
        dist = [1e6] * self.n
        heap = []
        targets = self.goals if goal_mode else [
            q for q, t in enumerate(self.base)
            if t == UNKNOWN and 0 < q % W < 10
        ]
        for q in targets:
            dist[q] = 0.
            heapq.heappush(heap, (0., q))

        while heap:
            cost, q = heapq.heappop(heap)
            if cost != dist[q]:
                continue
            x = q % W
            predecessors = [(q - direction * W, .22)]
            if x > 0:
                predecessors.append((q - 1, 1.))
            if x < W - 1:
                predecessors.append((q + 1, 1.))

            for k, step in predecessors:
                if not 0 <= k < self.n:
                    continue
                if self.base[k] in (WALL, SPIKE):
                    continue
                if self.base[q] in (WALL, SPIKE):
                    continue
                extra = .7 if self.base[q] in (BREAK, SOLID, GRAVITY) else 0.
                nc = cost + step + extra
                if nc < dist[k]:
                    dist[k] = nc
                    heapq.heappush(heap, (nc, k))
        return dist


def _path(parent, state, tail=None):
    segments = []
    if tail is not None:
        segments.append(tail)
    while parent[state] is not None:
        previous, sequence = parent[state]
        segments.append(sequence)
        state = previous

    result = []
    for segment in reversed(segments):
        result.extend(segment)
    return result


def _basic_search(model, visits, deadline):
    start = model.start
    goal_mode = bool(model.goals)
    tables = {
        -1: model.distances(-1, goal_mode),
        1: model.distances(1, goal_mode),
    }
    can_flip = bool(model.gravities)

    def heuristic(s):
        p, d = s[:2]
        if not 0 <= p < model.n:
            return 0.
        value = tables[d][p]
        if can_flip:
            value = min(value, 1.5 + tables[-d][p])
        if value >= 1e5:
            value = 18.
            if model.goals:
                x, y = model.xy(p)
                value = min(
                    abs(x - model.xy(q)[0])
                    + abs(y - model.xy(q)[1]) / 5.
                    for q in model.goals
                ) + 8.
        return value

    parent = {start: None}
    costs = {start: 0}
    serial = 0
    heap = [(1.45 * heuristic(start), 0, 0, start)]
    best_frontier = None
    expanded = 0

    while heap and expanded < 26000:
        if expanded % 64 == 0 and time.perf_counter() >= deadline:
            break
        _, _, g, s = heapq.heappop(heap)
        if costs.get(s) != g:
            continue
        expanded += 1
        if g >= 32:
            continue

        for sequence, ns, outcome in model.successors(s, basic=True):
            ng = g + 1
            if outcome == "win":
                return _path(parent, s, sequence), True

            if outcome == "unknown":
                x, y = model.xy(ns[0])
                penalty = visits.get((x, y, ns[1]), 0) * 2
                score = ng + penalty
                if not goal_mode:
                    return _path(parent, s, sequence), False
                if best_frontier is None or score < best_frontier[0]:
                    best_frontier = score, s, sequence
                continue

            if costs.get(ns, 10**9) <= ng:
                continue
            costs[ns] = ng
            parent[ns] = s, sequence
            x, y = model.xy(ns[0])
            repetition = min(3, visits.get((x, y, ns[1]), 0)) * .18
            priority = ng + 1.45 * heuristic(ns) + repetition
            serial += 1
            heapq.heappush(heap, (priority, serial, ng, ns))

    if best_frontier is not None:
        _, s, sequence = best_frontier
        return _path(parent, s, sequence), False
    return None, False


def _advanced_search(model, visits, deadline):
    start = model.start
    goal_mode = bool(model.goals)
    tables = {
        -1: model.distances(-1, goal_mode),
        1: model.distances(1, goal_mode),
    }
    can_flip = bool(model.gravities)
    start_x, start_y = model.xy(start[0])

    def heuristic(s):
        p, d = s[:2]
        if not 0 <= p < model.n:
            return 0.
        value = tables[d][p]
        if can_flip:
            value = min(value, 1.5 + tables[-d][p])
        if value >= 1e5:
            if model.goals:
                x, y = model.xy(p)
                value = min(
                    abs(x - model.xy(q)[0])
                    + abs(y - model.xy(q)[1]) / 4.
                    for q in model.goals
                ) + 10.
            else:
                _, y = model.xy(p)
                value = .5 * min(
                    abs(y - model.known_lo), abs(y - model.known_hi)
                ) + 3.
        return value

    def new_view(s):
        _, y = model.xy(s[0])
        low = y - (6 if s[1] == -1 else 4)
        high = y + (3 if s[1] == -1 else 5)
        return max(0, model.known_lo - low) + max(0, high - model.known_hi)

    parent = {start: None}
    costs = {start: 0}
    serial = 0
    heap = [(1.4 * heuristic(start), 0, 0, start)]
    best_unknown = None
    best_safe = None
    best_other = None
    expanded = 0
    limit = 7000 if model.has_growth else 23000

    while heap and expanded < limit:
        if expanded % 32 == 0 and time.perf_counter() >= deadline:
            break
        _, _, g, s = heapq.heappop(heap)
        if costs.get(s) != g:
            continue
        expanded += 1
        if g >= 42:
            continue

        successors = list(model.successors(s))
        if model.has_growth and expanded < 2300 and g < 24:
            successors.extend(model.growth_macros(s))

        for sequence, ns, outcome in successors:
            ng = g + len(sequence)
            if outcome == "win":
                return _path(parent, s, sequence), True

            x, y = model.xy(ns[0])
            repetition = visits.get((x, y, ns[1]), 0)

            if outcome == "unknown":
                score = ng + 5 * repetition + (
                    14 if model.has_growth else 4
                )
                if y < model.known_lo or y > model.known_hi:
                    score -= 1.
                if best_unknown is None or score < best_unknown[0]:
                    best_unknown = score, s, sequence
                continue

            if costs.get(ns, 10**9) <= ng:
                continue
            costs[ns] = ng
            parent[ns] = s, sequence

            novelty = new_view(ns)
            if ns[0] != start[0] and novelty > 0:
                score = ng + 4 * repetition - 1.5 * min(6, novelty)
                if best_safe is None or score < best_safe[0]:
                    best_safe = score, ns

            if ns[0] != start[0]:
                score = ng + 4 * repetition + .25 * heuristic(ns)
                if ns[1] == start[1]:
                    score -= .12 * abs(y - start_y)
                if best_other is None or score < best_other[0]:
                    best_other = score, ns

            priority = ng + 1.4 * heuristic(ns)
            priority += min(5, repetition) * .35
            if not goal_mode:
                priority -= min(6, novelty) * .6
            serial += 1
            heapq.heappush(heap, (priority, serial, ng, ns))

        if not goal_mode and expanded >= 350 and best_safe is not None:
            break

    if best_safe is not None:
        if best_unknown is None or best_safe[0] <= best_unknown[0]:
            return _path(parent, best_safe[1]), False
    if best_unknown is not None:
        _, s, sequence = best_unknown
        return _path(parent, s, sequence), False
    if best_safe is not None:
        return _path(parent, best_safe[1]), False
    if best_other is not None:
        return _path(parent, best_other[1]), False
    return None, False


def _export(model, path):
    result = []
    for (a, q), ns, out in path:
        target = model.xy(q) if q is not None else None
        result.append((a, target, model.xy(ns[0]), ns[1], out))
    return result


def act(obs, available, memory):
    begun = time.perf_counter()
    allowed = set(available)
    parsed = _decode(obs)

    if parsed is None:
        name = "RESET" if "RESET" in allowed else (
            "ACTION7" if "ACTION7" in allowed else sorted(allowed)[0]
        )
        return name, None, None

    # Select by observed mechanics, not by level identity.
    special = {GRAVITY, SOLID, GHOST, GROW}
    if any(t in special for t in parsed[2].values()):
        memory["advanced"] = True
    advanced = memory.get("advanced", False)
    if not advanced:
        parsed = _decode(obs, partial=True)

    px, top, view = parsed
    py, direction, disconnected = _locate(memory, px, top, view)
    board = {} if disconnected else dict(memory.get("board", {}))
    for (x, r), t in view.items():
        board[x, py + r] = t
    board[px, py] = EMPTY
    memory["board"] = board

    visits = memory.setdefault("visits", {})
    key = px, py, direction
    visits[key] = visits.get(key, 0) + 1

    plan = memory.get("plan", [])
    if disconnected or memory.get("plan_position") != key:
        plan = []

    if plan:
        a, target, _, _, _ = plan[0]
        if target is not None:
            x, y = target
            row = top + 6 * (y - py) + 3
            if not 0 <= row < 63:
                plan = []
            elif board.get(target) not in (
                BREAK, GRAVITY, SOLID, GHOST, GROW
            ):
                plan = []

    if not plan:
        model = Model(board, (px, py), direction)
        if advanced:
            path, winning = _advanced_search(
                model, visits, begun + 8.0
            )
        else:
            path, winning = _basic_search(
                model, visits, begun + 8.0
            )

        if path is None:
            choices = []
            for sequence, ns, out in model.successors(
                model.start, basic=not advanced
            ):
                if out not in ("ok", "win"):
                    continue
                x, y = model.xy(ns[0])
                score = 4 * visits.get((x, y, ns[1]), 0)
                if ns[0] == model.start[0]:
                    score += 2
                if sequence[0][0][0] == 6:
                    q = sequence[0][0][1]
                    if q == model.start[0] + direction * W:
                        score -= 1
                choices.append((score, sequence))
            if choices:
                path = min(choices, key=lambda z: z[0])[1]

        if path is None:
            memory["plan"] = []
            memory.pop("expected", None)
            if "ACTION7" in allowed:
                memory["board"] = {}
                return "ACTION7", None, None
            name = "RESET" if "RESET" in allowed else sorted(allowed)[0]
            memory.clear()
            return name, None, None

        exported = _export(model, path)
        if winning or not advanced:
            plan = exported
        else:
            plan = []
            for item in exported:
                plan.append(item)
                if item[2] != (px, py) or item[4] == "unknown":
                    break

    a, target, next_pos, next_direction, outcome = plan.pop(0)
    name = "ACTION" + str(a)
    if name not in allowed:
        memory["plan"] = []
        return sorted(allowed)[0], None, None

    memory["plan"] = plan
    memory["plan_position"] = (
        next_pos[0], next_pos[1], next_direction
    )
    memory["expected"] = (
        next_pos[0], next_pos[1], next_direction,
        outcome == "unknown", py
    )
    memory["direction"] = next_direction

    if a == 6:
        x, y = target
        row = top + 6 * (y - py) + 3
        col = 6 * x + 3
        return name, int(max(0, min(62, row))), int(max(0, min(63, col)))
    return name, None, None

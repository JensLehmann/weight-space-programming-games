import time
import heapq
from collections import deque
import numpy as np


def _read(obs):
    yy, xx = np.where(obs[:63] == 9)
    if not len(yy):
        return None
    top = int(yy.min()) - 1
    px = int(xx.min()) // 6
    direction = -1 if abs(top - 36) <= abs(top - 26) else 1
    tiles = {}

    for ry in range(-12, 13):
        y0 = top + ry * 6
        offsets = [j for j in range(1, 6) if 0 <= y0 + j < 63]
        if len(offsets) < 2:
            continue
        rows = [y0 + j for j in offsets]
        for x in range(11):
            cols = [x * 6 + j for j in range(1, 6)
                    if x * 6 + j < 64]
            if not cols:
                continue
            a = obs[np.ix_(rows, cols)]
            c = np.bincount(a.ravel(), minlength=16)
            n = a.size

            if c[9]:
                typ = "P"
            elif c[7]:
                typ = "+"
            elif c[14]:
                typ = "x"
            elif c[8]:
                typ = "g"
            elif c[12]:
                typ = "C" if c[12] > n * .43 else "c"
            elif c[15]:
                samples = [
                    int(obs[y0 + j, x * 6 + 1])
                    for j in (2, 3, 4)
                    if 0 <= y0 + j < 63 and x * 6 + 1 < 64
                ]
                if c[11] or c[0]:
                    typ = "s"
                elif samples and all(v == 5 for v in samples):
                    typ = "s"
                elif c[3] or (
                    samples and all(v == 15 for v in samples)
                    and c[15] >= n * .65
                ):
                    typ = "Y"
                else:
                    typ = "#"
            elif c[10] > n * .40:
                typ = "."
            else:
                typ = "#"
            tiles[x, ry] = typ

    return px, top, direction, tiles


def _signature(tiles, x, y, d):
    return (
        x, y, d,
        tuple(tiles.get((xx, yy), "?")
              for yy in range(y - 2, y + 3)
              for xx in range(1, 10))
    )


def _register(memory, px, direction, local):
    old = dict(memory.get("map", {}))
    expected = memory.get("expected")
    previous = memory.get("player", (px, 0, direction))
    clear = False
    new_chart = False

    if not old:
        py = 0
    else:
        hint = expected["y"] if expected else previous[1]
        certain = bool(expected and expected.get("certain"))
        py = hint
        if expected and (px != expected["x"] or direction != expected["d"]):
            clear = True

        def agreement(candidate):
            same = wrong = special = overlap = 0
            for (x, ry), typ in local.items():
                if typ == "P" or x < 1 or x > 9:
                    continue
                was = old.get((x, candidate + ry))
                if was is None:
                    continue
                overlap += 1
                if was == typ:
                    same += 1
                    if typ not in (".", "#"):
                        special += 1
                else:
                    wrong += 1
            return same, wrong, special, overlap

        base = agreement(hint)
        if not certain or base[1] > max(4, base[3] * .13):
            if certain:
                candidates = range(hint - 10, hint + 11)
            else:
                d = expected.get("d", direction) if expected else direction
                candidates = [hint + d * k for k in range(70)]

            best = None
            for candidate in candidates:
                same, wrong, special, overlap = agreement(candidate)
                if overlap < 18 or wrong > max(2, overlap * .09):
                    continue
                score = (
                    .07 * same + 1.8 * special - 3.5 * wrong
                    - .045 * abs(candidate - hint)
                )
                if best is None or score > best[0]:
                    best = score, candidate

            if best is not None:
                py = best[1]
                if py != hint:
                    clear = True
            elif not certain:
                old = {}
                py = 0
                clear = True
                new_chart = True

    disagreements = added = 0
    for (x, ry), typ in local.items():
        key = x, py + ry
        if typ == "P":
            old.setdefault(key, ".")
            continue
        was = old.get(key)
        if was is None:
            added += 1
        elif was != typ:
            disagreements += 1
        old[key] = typ

    if disagreements > 3 or added > 5:
        clear = True
    if clear:
        memory["plan"] = []
    if new_chart:
        memory["history"] = []
        memory["bans"] = {}

    return py, old


class Board:
    W = 11

    def __init__(self, tiles, x, y, d, construction=False):
        self.tiles = tiles
        self.construction = construction
        ys = [p[1] for p in tiles]
        self.lo = min(ys + [y]) - 1
        self.hi = max(ys + [y]) + 1
        self.n = (self.hi - self.lo + 1) * self.W
        self.bits = [1 << i for i in range(self.n)]

        self.known = 0
        self.wall = 0
        self.spikes = 0
        self.gems = 0
        self.csites = 0
        xb = gb = cb = yb = 0

        for (tx, ty), typ in tiles.items():
            if not 0 <= tx < self.W:
                continue
            i = self.index(tx, ty)
            bit = self.bits[i]
            self.known |= bit
            if typ == "#":
                self.wall |= bit
            elif typ == "s":
                self.spikes |= bit
            elif typ == "+":
                self.gems |= bit
            elif typ == "x":
                xb |= bit
            elif typ == "g":
                gb |= bit
            elif typ in ("C", "c"):
                self.csites |= bit
                if typ == "C":
                    cb |= bit
            elif typ == "Y":
                yb |= bit

        self.initial = self.index(x, y), d, xb, gb, cb, yb
        self.mutable = xb | gb | cb | yb | self.csites
        self.info_cache = {}
        self.dead_cache = {}
        self.heuristic_cache = {}
        self.base_info = self.information(self.initial)
        self.goals = list(self.iterbits(self.gems))

    def index(self, x, y):
        return (y - self.lo) * self.W + x

    def xy(self, i):
        return i % self.W, i // self.W + self.lo

    @staticmethod
    def iterbits(mask):
        while mask:
            b = mask & -mask
            yield b.bit_length() - 1
            mask ^= b

    def solid(self, s):
        return self.wall | s[2] | s[3] | s[4] | s[5]

    def kind(self, s, i):
        if not 0 <= i < self.n:
            return "?"
        b = self.bits[i]
        if not self.known & b:
            return "?"
        if self.wall & b:
            return "#"
        if self.spikes & b:
            return "s"
        if self.gems & b:
            return "+"
        if s[2] & b:
            return "x"
        if s[3] & b:
            return "g"
        if s[4] & b:
            return "C"
        if s[5] & b:
            return "Y"
        if self.csites & b:
            return "c"
        return "."

    def visible(self, s, i):
        if not 0 <= i < self.n:
            return False
        top = 36 if s[1] == -1 else 26
        row = top + 6 * (i // self.W - s[0] // self.W)
        return row <= 62 and row + 5 >= 0

    def information(self, s):
        key = s[0], s[1]
        if key in self.info_cache:
            return self.info_cache[key]

        top = 36 if s[1] == -1 else 26
        base = s[0] // self.W
        count = 0
        for dy in range(-7, 8):
            if not 1 <= top + 6 * dy + 2 <= 61:
                continue
            for x in range(1, 10):
                i = (base + dy) * self.W + x
                if not 0 <= i < self.n or not self.known & self.bits[i]:
                    count += 1

        self.info_cache[key] = count
        return count

    def frontier(self, s):
        return self.information(s) > self.base_info + 4

    def fall(self, s, start):
        pos = start
        step = s[1] * self.W
        solid = self.solid(s)

        for _ in range(self.n + 1):
            nxt = pos + step
            if not 0 <= nxt < self.n:
                return pos, 2
            bit = self.bits[nxt]
            if not self.known & bit:
                return pos, 2
            if self.gems & bit:
                return nxt, 1
            if self.spikes & bit:
                return pos, -1
            if solid & bit:
                return pos, 0
            pos = nxt

        return pos, 2

    def simulate(self, s, action):
        p, d, xb, gb, cb, yb = s
        aid = action[0]

        if aid in (3, 4):
            dx = -1 if aid == 3 else 1
            if not 0 <= p % self.W + dx < self.W:
                return None
            q = p + dx
            typ = self.kind(s, q)
            if typ == "+":
                return (q, d, xb, gb, cb, yb), 1
            if typ not in (".", "c"):
                return None
            land, status = self.fall(s, q)
            if status < 0:
                return None
            return (land, d, xb, gb, cb, yb), status

        if aid != 6:
            return None
        i = action[1]
        if i == p or not self.visible(s, i):
            return None
        typ = self.kind(s, i)
        if typ not in ("x", "g", "C", "c", "Y"):
            return None
        bit = self.bits[i]

        if typ == "g":
            nd = -d
            q = p + nd * self.W
            support = self.kind(s, q)
            ns = p, nd, xb, gb & ~bit, cb, yb
            if support in ("#", "x", "g", "C", "Y"):
                return ns, 0
            if support == "?":
                return ns, 2
            land, status = self.fall(ns, q)
            if status < 0:
                return None
            return (land, nd, xb, gb & ~bit, cb, yb), status

        if typ == "x":
            xb &= ~bit
        elif typ in ("C", "c"):
            cb ^= bit
        else:
            ix = i % self.W
            neighbours = [i - self.W, i + self.W]
            if ix:
                neighbours.append(i - 1)
            if ix + 1 < self.W:
                neighbours.append(i + 1)
            for j in neighbours:
                if j != p and self.kind(s, j) == ".":
                    yb |= self.bits[j]
            yb &= ~bit

        ns = p, d, xb, gb, cb, yb
        if i == p + d * self.W:
            land, status = self.fall(ns, i)
            if status < 0:
                return None
            return (land, d, xb, gb, cb, yb), status
        return ns, 0

    def actions(self, s, available):
        p, d = s[:2]
        if "ACTION3" in available:
            yield (3,)
        if "ACTION4" in available:
            yield (4,)
        if "ACTION6" not in available:
            return

        clicks = set()
        near = [p + d * self.W]
        if p % self.W:
            near.append(p - 1)
        if p % self.W + 1 < self.W:
            near.append(p + 1)
        for i in near:
            if self.kind(s, i) in ("x", "C", "c", "Y"):
                clicks.add(i)

        visible_g = []
        for i in self.iterbits(s[3]):
            if self.visible(s, i):
                clicks.add(i)
                visible_g.append(i)

        for i in self.iterbits(self.csites & ~s[4]):
            if (i != p and abs(i % self.W - p % self.W) <= 1
                    and self.visible(s, i)):
                dy = i // self.W - p // self.W
                if dy * d > 0 or visible_g:
                    clicks.add(i)

        for i in sorted(clicks):
            if self.visible(s, i):
                yield 6, i

    def dead_end(self, s, available):
        if s in self.dead_cache:
            return self.dead_cache[s]
        if s[5]:
            self.dead_cache[s] = False
            return False

        for a in self.actions(s, available):
            r = self.simulate(s, a)
            if r is not None and r[0] != s:
                self.dead_cache[s] = False
                return False

        self.dead_cache[s] = True
        return True

    def heuristic_table(self, gravity_blocks):
        if gravity_blocks in self.heuristic_cache:
            return self.heuristic_cache[gravity_blocks]

        dist = [float("inf")] * (self.n * 2)
        blocked = self.wall | self.spikes
        passable = [
            not bool(blocked & self.bits[i]) and 0 < i % self.W < 10
            for i in range(self.n)
        ]
        switch_rows = [i // self.W for i in self.iterbits(gravity_blocks)]
        flip_ok = [[False] * self.n for _ in range(2)]

        if switch_rows:
            for i in range(self.n):
                if not passable[i]:
                    continue
                row = i // self.W
                for k, d in enumerate((-1, 1)):
                    top = 36 if d == -1 else 26
                    flip_ok[k][i] = any(
                        -5 <= top + 6 * (r - row) <= 62
                        for r in switch_rows
                    )

        heap = []
        for goal in self.goals:
            for k in (0, 1):
                node = goal * 2 + k
                dist[node] = 0.0
                heapq.heappush(heap, (0.0, node))

        while heap:
            cost, node = heapq.heappop(heap)
            if cost != dist[node]:
                continue
            i, k = divmod(node, 2)
            d = -1 if k == 0 else 1
            x = i % self.W
            predecessors = []

            if x > 1:
                predecessors.append((i - 1, k, 1.0))
            if x < 9:
                predecessors.append((i + 1, k, 1.0))

            j = i - d * self.W
            if 0 <= j < self.n:
                predecessors.append((j, k, .55))
            if flip_ok[1 - k][i]:
                predecessors.append((i, 1 - k, 2.0))

            for j, jk, weight in predecessors:
                if not passable[j]:
                    continue
                nxt = j * 2 + jk
                nc = cost + weight
                if nc < dist[nxt]:
                    dist[nxt] = nc
                    heapq.heappush(heap, (nc, nxt))

        self.heuristic_cache[gravity_blocks] = dist
        return dist

    def heuristic(self, s):
        if not self.goals:
            return float("inf")
        table = self.heuristic_table(s[3])
        return table[s[0] * 2 + (1 if s[1] == 1 else 0)]

    def growth_macros(self, s, deadline):
        if not s[5]:
            return []
        p, d = s[:2]
        roots = [i for i in self.iterbits(s[5])
                 if i != p and self.visible(s, i)]
        if not roots:
            return []

        starts = []
        px = p % self.W
        for dx in (-1, 1):
            if not 0 <= px + dx < self.W:
                continue
            q = p + dx
            if self.kind(s, q) not in (".", "c"):
                continue
            land, status = self.fall(s, q)
            if status != 0 or abs(land - q) >= 2 * self.W:
                starts.append((q, d))

        if any(self.visible(s, i) for i in self.iterbits(s[3])):
            q = p - d * self.W
            if self.kind(s, q) in (".", "c", "s", "+", "?"):
                starts.append((p, -d))

        targets = set()
        for q, direction in starts:
            j = q + direction * self.W
            for _ in range(11):
                if not 0 <= j < self.n or not self.visible(s, j):
                    break
                typ = self.kind(s, j)
                if typ not in (".", "c"):
                    break
                if typ == "." and j != p:
                    targets.add(j)
                j += direction * self.W

        if s[3]:
            for dx in (-1, 0, 1):
                if not 1 <= px + dx <= 9:
                    continue
                for dy in range(2, 6):
                    j = p + dx - d * dy * self.W
                    if self.visible(s, j) and self.kind(s, j) == ".":
                        targets.add(j)

        if not targets:
            return []

        parent = {i: None for i in roots}
        depth = {i: 0 for i in roots}
        queue = deque(roots)

        while queue:
            i = queue.popleft()
            if depth[i] >= 11:
                continue
            x = i % self.W
            neighbours = [i - self.W, i + self.W]
            if x:
                neighbours.append(i - 1)
            if x + 1 < self.W:
                neighbours.append(i + 1)

            for j in neighbours:
                if j in parent or j == p or not self.visible(s, j):
                    continue
                if self.kind(s, j) not in (".", "Y"):
                    continue
                parent[j] = i
                depth[j] = depth[i] + 1
                queue.append(j)

        candidates = [i for i in targets if i in parent and depth[i] > 0]
        candidates.sort(key=lambda i: (
            depth[i], abs(i // self.W - p // self.W), i))
        results = []
        sequences = set()

        for target in candidates[:16]:
            if time.perf_counter() >= deadline:
                break
            path = []
            cur = target
            while parent[cur] is not None:
                cur = parent[cur]
                path.append(cur)
            path.reverse()

            key = tuple(path)
            if key in sequences:
                continue
            sequences.add(key)

            state = s
            seq = []
            status = 0
            valid = True
            for i in path:
                if self.kind(state, i) != "Y":
                    valid = False
                    break
                r = self.simulate(state, (6, i))
                if r is None:
                    valid = False
                    break
                state, status = r
                seq.append((6, i))
                if status:
                    break

            if valid and seq:
                results.append((state, status, tuple(seq)))

        return results

    def changed_map(self, tiles, s):
        out = dict(tiles)
        mask = self.mutable | s[2] | s[3] | s[4] | s[5]
        for i in self.iterbits(mask):
            out[self.xy(i)] = self.kind(s, i)
        return out

    def logical(self, action):
        if action[0] == 6:
            x, y = self.xy(action[1])
            return 6, x, y
        return (action[0],)

    def indexed(self, action):
        if action[0] == 6:
            return 6, self.index(action[1], action[2])
        return (action[0],)


def _search(board, available, banned, deadline):
    start = board.initial
    h0 = board.heuristic(start) if board.construction else float("inf")
    directed = h0 < float("inf")

    states = [start]
    parents = [None]
    best_cost = {start: 0}
    heap = [(2.2 * h0 if directed else 0, 0, 0)]
    goal = frontier = partial = None
    popped = 0
    node_limit = 30000 if board.construction else 24000

    def recover(index):
        chunks = []
        while parents[index] is not None:
            prev, seq = parents[index]
            chunks.append(seq)
            index = prev
        result = []
        for chunk in reversed(chunks):
            result.extend(chunk)
        return result

    def geometric(s):
        if not board.goals:
            return float("inf")
        x, y = board.xy(s[0])
        return min(
            abs(x - board.xy(g)[0]) + .3 * abs(y - board.xy(g)[1])
            for g in board.goals
        )

    initial_distance = h0 if directed else geometric(start)

    while heap and popped < node_limit and time.perf_counter() < deadline:
        priority, cost, index = heapq.heappop(heap)
        s = states[index]
        if best_cost.get(s) != cost:
            continue

        if not directed:
            if goal is not None and cost >= goal[0]:
                break
            if frontier is not None and cost > frontier[0] + 7:
                break

        if cost > (58 if directed else 44):
            continue
        popped += 1

        distance = board.heuristic(s) if directed else geometric(s)
        moved = s[0] != start[0]
        if board.construction:
            moved = moved or s[1] != start[1]

        if (index and moved and distance + .2 < initial_distance
                and not board.dead_end(s, available)):
            score = distance + (.13 if board.construction else .2) * cost
            if partial is None or score < partial[0]:
                partial = score, index

        transitions = []
        for action in board.actions(s, available):
            if index == 0 and board.logical(action) in banned:
                continue
            r = board.simulate(s, action)
            if r is not None:
                ns, status = r
                transitions.append((ns, status, (action,)))

        if "ACTION6" in available and s[5] and cost <= 40:
            for ns, status, seq in board.growth_macros(s, deadline):
                if index == 0 and board.logical(seq[0]) in banned:
                    continue
                transitions.append((ns, status, seq))

        for ns, status, seq in transitions:
            new_cost = cost + len(seq)
            if status == 0 and board.dead_end(ns, available):
                continue

            if status == 1:
                ni = len(states)
                states.append(ns)
                parents.append((index, seq))
                if directed:
                    return recover(ni), True
                if goal is None or new_cost < goal[0]:
                    goal = new_cost, ni
                continue

            is_frontier = status == 2 or board.frontier(ns)
            if is_frontier:
                ni = len(states)
                states.append(ns)
                parents.append((index, seq))
                gain = max(0, board.information(ns) - board.base_info)
                score = (
                    new_cost + (23 if status == 2 else 0)
                    - .035 * min(45, gain)
                )

                if directed:
                    nh = board.heuristic(ns)
                    score += min(60, nh if nh < float("inf") else 60)

                if frontier is None or score < frontier[0]:
                    frontier = score, ni
                if status == 2:
                    continue

            if new_cost >= best_cost.get(ns, 10 ** 9):
                continue

            nh = board.heuristic(ns) if directed else 0
            if directed and nh == float("inf"):
                if not ns[3] and not is_frontier:
                    continue
                nh = 80

            best_cost[ns] = new_cost
            ni = len(states)
            states.append(ns)
            parents.append((index, seq))
            heapq.heappush(heap, (
                new_cost + (2.2 * nh if directed else 0),
                new_cost, ni
            ))

    if goal is not None:
        if directed or frontier is None or goal[0] <= frontier[0] + 8:
            return recover(goal[1]), True
    if directed and partial is not None:
        return recover(partial[1]), False
    if frontier is not None:
        return recover(frontier[1]), True
    if goal is not None:
        return recover(goal[1]), True
    if partial is not None:
        return recover(partial[1]), False
    return [], False


def _undo(memory, tiles, available):
    history = memory.get("history", [])
    if "ACTION7" not in available or not history:
        return None

    snapshot = history.pop()
    bans = memory.setdefault("bans", {})
    bans.setdefault(snapshot["signature"], set()).add(snapshot["action"])

    restored = dict(tiles)
    restored.update(snapshot["map"])
    memory["map"] = restored
    memory["player"] = snapshot["player"]
    x, y, d = snapshot["player"]
    memory["expected"] = {"x": x, "y": y, "d": d, "certain": True}
    memory["plan"] = []
    return "ACTION7", None, None


def act(obs, available, memory):
    began = time.perf_counter()
    available = set(available)
    parsed = _read(obs)

    if parsed is None:
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        if "ACTION7" in available:
            memory["plan"] = []
            return "ACTION7", None, None
        return next(iter(sorted(available)), "ACTION3"), None, None

    px, top, d, local = parsed
    if any(typ == "Y" for typ in local.values()):
        memory["construction"] = True

    py, tiles = _register(memory, px, d, local)
    construction = bool(memory.get("construction", False))
    board = Board(tiles, px, py, d, construction)
    state = board.initial

    signature = _signature(tiles, px, py, d)
    banned = memory.setdefault("bans", {}).get(signature, set())
    visits = memory.setdefault("visits", {})
    visits[signature] = visits.get(signature, 0) + 1

    queued = list(memory.get("plan", []))
    chosen = None
    if queued:
        candidate = tuple(queued[0])
        indexed = board.indexed(candidate)
        r = board.simulate(state, indexed)
        if ("ACTION%d" % candidate[0] in available
                and candidate not in banned and r is not None
                and (r[1] != 0 or not board.dead_end(r[0], available))):
            chosen = candidate
            queued = queued[1:]
        else:
            queued = []

    if chosen is None:
        allowance = 7.2 if construction else 7.3
        plan, conclusive = _search(
            board, available, banned, began + allowance
        )
        logical = [board.logical(a) for a in plan]
        if logical and (conclusive or visits[signature] <= 2):
            chosen = logical[0]
            queued = logical[1:]

    if chosen is None:
        options = []
        if visits[signature] <= 2:
            directed = construction and board.heuristic(state) < float("inf")

            for action in board.actions(state, available):
                logical = board.logical(action)
                if logical in banned:
                    continue
                r = board.simulate(state, action)
                if r is None:
                    continue
                ns, status = r
                if status == 0 and board.dead_end(ns, available):
                    continue

                nx, ny = board.xy(ns[0])
                changed = board.changed_map(tiles, ns)
                key = _signature(changed, nx, ny, ns[1])
                seen = visits.get(key, 0)
                score = (
                    12 * seen + (30 if status == 2 else 0)
                    - (200 if status == 1 else 0)
                    - .06 * board.information(ns)
                )
                if directed:
                    h = board.heuristic(ns)
                    score += min(100, h if h < float("inf") else 100)
                options.append((score, action))

        if options:
            options.sort(key=lambda a: (a[0], a[1]))
            chosen = board.logical(options[0][1])
            queued = []
        else:
            undo = _undo(memory, tiles, available)
            if undo is not None:
                return undo
            if "RESET" in available:
                memory.clear()
                return "RESET", None, None
            for name in ("ACTION3", "ACTION4", "ACTION7"):
                if name in available:
                    return name, None, None
            return next(iter(sorted(available)), "ACTION3"), None, None

    indexed = board.indexed(chosen)
    r = board.simulate(state, indexed)
    if r is None:
        memory["plan"] = []
        undo = _undo(memory, tiles, available)
        if undo is not None:
            return undo
        return ("ACTION3" if "ACTION3" in available else "ACTION4"), None, None

    ns, status = r
    nx, ny = board.xy(ns[0])
    history = memory.setdefault("history", [])
    history.append({
        "map": dict(tiles),
        "player": (px, py, d),
        "signature": signature,
        "action": tuple(chosen)
    })
    if len(history) > 100:
        del history[0]

    memory["map"] = board.changed_map(tiles, ns)
    memory["player"] = (px, py, d)
    memory["expected"] = {
        "x": nx, "y": ny, "d": ns[1], "certain": status != 2
    }

    moved = ns[0] != state[0] or ns[1] != state[1]
    memory["plan"] = queued if status == 0 and not moved else []

    if chosen[0] != 6:
        return "ACTION%d" % chosen[0], None, None

    x, y = chosen[1], chosen[2]
    row = max(0, min(62, top + 6 * (y - py) + 3))
    col = max(0, min(63, x * 6 + 3))
    return "ACTION6", int(row), int(col)

import numpy as np
import math
import time
import heapq
import itertools
from collections import deque, Counter

# Directions: up, down, left, right.
_DIRS = ((0, 1, 2, 3), (0, 1, 3, 2),
         (1, 0, 2, 3), (1, 0, 3, 2))
_DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_INF = 10**8


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(x, y)]
        out = []
        for px, py in q:
            out.append((px, py))
            for dx, dy in _DXY:
                nx, ny = px + dx, py + dy
                if (0 <= nx < w and 0 <= ny < h
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((nx, ny))
        result.append(out)
    return result


def _palette(obs):
    ans = []
    for y0, y1 in ((1, 32), (32, 63)):
        for x0, x1 in ((0, 32), (32, 64)):
            a = obs[y0:y1, x0:x1].ravel()
            counts = Counter(int(v) for v in a
                             if int(v) not in (5, 10, 1, 9))
            if counts:
                ans.append(counts.most_common(1)[0][0])
            else:
                ans.append(int(obs[y0, x0]))
    return ans


def _find_view(obs, pal):
    rects = []
    for color in (10, 1):
        if color in pal:
            continue
        for component in _components(obs == color):
            xs = [p[0] for p in component]
            ys = [p[1] for p in component]
            x0, y0 = min(xs), min(ys)
            w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
            if w * h == len(component) and min(w, h) >= 2:
                rects.append((x0, y0, w, h))
    if rects:
        sizes = []
        for x, y, w, h in rects:
            g = math.gcd(w, h)
            if 2 <= g <= 8:
                sizes.append(g)
        s = Counter(sizes).most_common(1)[0][0] if sizes else 4
        good = [r for r in rects if r[2] % s == 0 and r[3] % s == 0]
        if not good:
            good = rects
        ox = Counter(r[0] % s for r in good).most_common(1)[0][0]
        oy = Counter(r[1] % s for r in good).most_common(1)[0][0]
    else:
        # Fallback: find the cell lattice which best explains uniform tiles.
        best = None
        for s0 in range(3, 7):
            for ox0 in range(s0):
                for oy0 in range(s0):
                    error, useful = 0.0, 0
                    for y in range(oy0, 64 - s0 + 1, s0):
                        for x in range(ox0, 64 - s0 + 1, s0):
                            a = obs[max(y, 1):min(y + s0, 63), x:x + s0]
                            if not a.size:
                                continue
                            n5 = int(np.count_nonzero(a == 5))
                            if 0 < n5 < a.size:
                                error += min(n5, a.size - n5) / a.size
                            if n5:
                                useful += 1
                    score = error / max(1, useful) + 0.0001 * useful
                    if best is None or score < best[0]:
                        best = (score, s0, ox0, oy0)
        _, s, ox, oy = best
    # Extra complete padding cells are harmless: they decode as walls.
    return (s, ox, oy, (64 - ox) // s, (64 - oy) // s)


def _read(obs, memory):
    pal = _palette(obs)
    if "view" not in memory:
        memory["view"] = _find_view(obs, pal)
    s, ox, oy, w, h = memory["view"]

    expected = np.empty((64, 64), dtype=np.int16)
    expected[:32, :32] = pal[0]
    expected[:32, 32:] = pal[1]
    expected[32:, :32] = pal[2]
    expected[32:, 32:] = pal[3]

    walls, hazards = set(), set()
    agents, boxes, colored = {}, set(), {}
    selected = None

    for gy in range(h):
        for gx in range(w):
            p = gy * w + gx
            x, y = ox + gx * s, oy + gy * s
            ya, yb = max(1, y), min(63, y + s)
            a = obs[ya:yb, x:x + s]
            if not a.size:
                walls.add(p)
                continue
            e = expected[ya:yb, x:x + s]
            if np.all(a == e):
                walls.add(p)
                continue

            full = obs[y:y + s, x:x + s]
            inner = full[1:-1, 1:-1]
            if s >= 3 and inner.size:
                border = np.concatenate((full[0], full[-1],
                                         full[1:-1, 0], full[1:-1, -1]))
                if (np.all(border == 5)
                        and (np.all(inner == 9) or np.all(inner == 11))):
                    boxes.add(p)
                    if int(inner.flat[0]) == 11:
                        selected = p
                    continue

            vals, counts = np.unique(a, return_counts=True)
            dominant = int(vals[int(np.argmax(counts))])
            if len(vals) == 1:
                if dominant in (10, 1):
                    agents[p] = dominant
                elif dominant == 5:
                    pass
                elif dominant in (12, 14, 15):
                    colored[p] = dominant
                else:
                    walls.add(p)
                continue

            n5 = int(np.count_nonzero(a == 5))
            if n5 == a.size:
                continue
            if 0.28 * a.size <= n5 <= 0.72 * a.size:
                # Hazard cells are checkerboard stippled by the renderer.
                yy, xx = np.indices(a.shape)
                parity = (yy + ya + xx + x) & 1
                if (np.all(a[parity == 0] == 5)
                        or np.all(a[parity == 1] == 5)):
                    hazards.add(p)
                    continue
            if np.count_nonzero(a == e) >= 0.8 * a.size:
                walls.add(p)

    oldg = memory.get("gates", {})
    olds = memory.get("switches", {})
    gates, switches = dict(oldg), dict(olds)
    seen = set()
    for start, color in colored.items():
        if start in seen:
            continue
        group = [start]
        seen.add(start)
        for p in group:
            x, y = p % w, p // w
            for dx, dy in _DXY:
                nx, ny = x + dx, y + dy
                q = ny * w + nx
                if (0 <= nx < w and 0 <= ny < h and q not in seen
                        and colored.get(q) == color):
                    seen.add(q)
                    group.append(q)
        bit = {14: 1, 12: 2, 15: 4}[color]
        if len(group) >= 2:
            for p in group:
                if p not in olds:
                    gates[p] = bit
                    switches.pop(p, None)
        else:
            p = group[0]
            if p not in gates:
                switches[p] = bit

    # Remember features hidden by pieces or by an open gate.
    hazards |= set(memory.get("hazards", ()))
    walls -= set(gates)
    walls -= set(switches)
    walls -= set(agents)
    walls -= boxes
    memory["gates"] = gates
    memory["switches"] = switches
    memory["hazards"] = hazards

    fp = (w, h, tuple(sorted(walls)), tuple(sorted(hazards)),
          tuple(sorted(gates.items())), tuple(sorted(switches.items())))
    return {
        "w": w, "h": h, "walls": walls, "hazards": hazards,
        "gates": gates, "switches": switches, "agents": agents,
        "boxes": tuple(sorted(boxes)), "selected": selected, "fp": fp
    }


class _World:
    def __init__(self, data, boxes=None):
        self.w, self.h = data["w"], data["h"]
        self.n = self.w * self.h
        self.walls = set(data["walls"])
        self.hazards = set(data["hazards"])
        self.boxes = tuple(data["boxes"] if boxes is None else sorted(boxes))
        self.boxset = set(self.boxes)
        self.gates = dict(data["gates"])
        self.switches = dict(data["switches"])
        self.gb = [0] * self.n
        self.sb = [0] * self.n
        for p, b in self.gates.items():
            self.gb[p] = b
        for p, b in self.switches.items():
            self.sb[p] = b

        blocked = self.walls | self.boxset
        self.base = [[0] * self.n for _ in range(4)]
        for p in range(self.n):
            x, y = p % self.w, p // self.w
            for a, (dx, dy) in enumerate(_DXY):
                nx, ny = x + dx, y + dy
                q = ny * self.w + nx
                if not (0 <= nx < self.w and 0 <= ny < self.h) or q in blocked:
                    q = p
                self.base[a][p] = q

        self.cells = [p for p in range(self.n)
                      if p not in blocked and p not in self.hazards]
        self.index = [-1] * self.n
        for i, p in enumerate(self.cells):
            self.index[p] = i
        self.f = len(self.cells)
        self.tables = {}
        self.rev = None
        self.opts = None

    def mask(self, st):
        mask = 0
        for p in st:
            if p >= 0:
                mask |= self.sb[p]
        return mask

    def step(self, st, a):
        mask = self.mask(st)
        active = [i for i in range(4) if st[i] >= 0]
        ns = list(st)
        for i in active:
            p = st[i]
            q = self.base[_DIRS[i][a]][p]
            if self.gb[q] & ~mask:
                q = p
            if q in self.hazards:
                return None
            ns[i] = q

        if len(active) == 2:
            i, j = active
            p, q = st[i], st[j]
            adjacent = abs(p - q) == 1 and p // self.w == q // self.w
            if ns[i] == ns[j] or (
                    adjacent and (ns[i] == q or ns[j] == p)):
                meet = (ns[i] + ns[j]) // 2
                ns[i] = ns[j] = -1
                return tuple(ns), (meet,)
            return tuple(ns), ()

        for ii, i in enumerate(active):
            for j in active[ii + 1:]:
                p, q = st[i], st[j]
                if (abs(p - q) == 1 and p // self.w == q // self.w
                        and (ns[i] == q or ns[j] == p)):
                    ns[i] = ns[j] = (ns[i] + ns[j]) // 2

        groups = {}
        for i in active:
            groups.setdefault(ns[i], []).append(i)
        merged = []
        for p, group in groups.items():
            if len(group) >= 2:
                merged.append(p)
                for i in group[:2]:
                    ns[i] = -1
                for i in group[2:]:
                    ns[i] = st[i]
        return tuple(ns), tuple(merged)

    def prepare_reverse(self):
        if self.rev is not None:
            return
        f = self.f
        rev = [[[] for _ in range(f)] for _ in range(4)]
        opts = [[None] * f for _ in range(4)]
        for a in range(4):
            for i, p in enumerate(self.cells):
                q = self.base[a][p]
                possibilities = []
                if q not in self.hazards and self.index[q] >= 0:
                    possibilities.append(self.index[q])
                # Other pieces may hold a gate open or let it close.
                if self.gb[q] and i not in possibilities:
                    possibilities.append(i)
                opts[a][i] = possibilities
                for j in possibilities:
                    rev[a][j].append(i)
        self.rev, self.opts = rev, opts

    def table(self, relative, deadline):
        if relative in self.tables:
            return self.tables[relative]
        self.prepare_reverse()
        f, rev, opts = self.f, self.rev, self.opts
        dist = [-1] * (f * f)
        q = deque()
        for i in range(f):
            k = i * f + i
            dist[k] = 0
            q.append(k)

        # The simulator also joins horizontally adjacent pieces when one
        # enters the other's previous cell.
        for i, p in enumerate(self.cells):
            other = p + 1
            if p % self.w == self.w - 1 or other >= self.n:
                continue
            j = self.index[other]
            if j < 0:
                continue
            for u, v in ((i, j), (j, i)):
                merge = False
                for a in (2, 3):
                    aa = _DIRS[relative][a]
                    if opts[a][u] and opts[aa][v]:
                        if v in opts[a][u] or u in opts[aa][v]:
                            merge = True
                            break
                k = u * f + v
                if merge and dist[k] < 0:
                    dist[k] = 1
                    q.append(k)

        count = 0
        while q:
            k = q.popleft()
            u, v = divmod(k, f)
            nd = dist[k] + 1
            for a in range(4):
                ru = rev[a][u]
                rv = rev[_DIRS[relative][a]][v]
                for pu in ru:
                    base = pu * f
                    for pv in rv:
                        z = base + pv
                        if dist[z] < 0:
                            dist[z] = nd
                            q.append(z)
            count += 1
            if count % 8192 == 0 and time.monotonic() > deadline:
                return None
        self.tables[relative] = dist
        return dist

    def box_target(self, selected, a, st):
        dx, dy = _DXY[a]
        x, y = selected % self.w + dx, selected // self.w + dy
        if not (0 <= x < self.w and 0 <= y < self.h):
            return selected
        p = y * self.w + x
        if (p in self.walls or p in self.boxset or p in self.hazards
                or p in self.switches or p in st
                or self.gb[p] & ~self.mask(st)):
            return selected
        return p


def _signature(hyp):
    st, ghosts = hyp
    return frozenset(p for p in st if p >= 0) | frozenset(ghosts)


def _hypotheses(agents, selected):
    positions = sorted(agents)
    n = len(positions)
    result = []
    if n == 4:
        for perm in itertools.permutations(positions):
            result.append((tuple(perm), ()))
    elif n == 3:
        active_colored = {p for p, c in agents.items() if c == 1}
        for ghost in positions:
            active = [p for p in positions if p != ghost]
            if selected is not None and active_colored:
                if set(active) != active_colored:
                    continue
            for ids in itertools.combinations(range(4), 2):
                for pair in (active, active[::-1]):
                    st = [-1] * 4
                    st[ids[0]], st[ids[1]] = pair
                    result.append((tuple(st), (ghost,)))
    elif n == 2:
        # Include all possible pairs of mirrored control types.
        for ids in itertools.combinations(range(4), 2):
            for pair in (positions, positions[::-1]):
                st = [-1] * 4
                st[ids[0]], st[ids[1]] = pair
                result.append((tuple(st), ()))
    elif n == 1:
        result.append(((positions[0], -1, -1, -1), ()))
    return result


def _path(parent, st):
    ans = []
    while st in parent:
        previous, action = parent[st]
        ans.append(action)
        st = previous
    ans.reverse()
    return ans


def _solve(world, start, deadline, maxnodes=150000):
    active = [i for i, p in enumerate(start) if p >= 0]
    if not active:
        return [], True, True
    if len(active) not in (2, 4):
        return None, False, True

    relatives = {i ^ j for i, j in itertools.combinations(active, 2)}
    tables = {}
    for rel in relatives:
        table = world.table(rel, deadline)
        if table is None:
            return None, False, False
        tables[rel] = table
    f, index = world.f, world.index

    def pair(st, i, j):
        x, y = index[st[i]], index[st[j]]
        if x < 0 or y < 0:
            return _INF
        d = tables[i ^ j][x * f + y]
        return d if d >= 0 else _INF

    def heuristic(st):
        ids = [i for i, p in enumerate(st) if p >= 0]
        if not ids:
            return 0
        if len(ids) == 2:
            return pair(st, ids[0], ids[1])
        return min(
            max(pair(st, 0, 1), pair(st, 2, 3)),
            max(pair(st, 0, 2), pair(st, 1, 3)),
            max(pair(st, 0, 3), pair(st, 1, 2))
        )

    h0 = heuristic(start)
    if h0 >= _INF:
        return None, False, True
    heap = [(1.3 * h0, h0, 0, 0, start)]
    costs = {start: 0}
    parent = {}
    serial = 0
    best, besth, bestg = start, h0, 0
    expanded = 0

    while heap:
        _, hh, _, g, st = heapq.heappop(heap)
        if costs.get(st) != g:
            continue
        if all(p < 0 for p in st):
            return _path(parent, st), True, True
        if hh < besth or (hh == besth and g < bestg):
            best, besth, bestg = st, hh, g
        if g < 140:
            for a in range(4):
                transition = world.step(st, a)
                if transition is None:
                    continue
                ns = transition[0]
                if ns == st:
                    continue
                ng = g + 1
                if ng >= costs.get(ns, _INF):
                    continue
                h = heuristic(ns)
                if h >= _INF:
                    continue
                costs[ns] = ng
                parent[ns] = (st, a)
                serial += 1
                heapq.heappush(heap, (ng + 1.3 * h, h, serial, ng, ns))
        expanded += 1
        if expanded % 256 == 0:
            if expanded >= maxnodes or time.monotonic() >= deadline:
                if best != start:
                    return _path(parent, best)[:16], False, False
                if heap:
                    return _path(parent, heap[0][-1])[:8], False, False
                break

    return None, False, True


def _topology_score(world, boxes, st):
    blocked = world.walls | world.hazards | set(boxes)
    component = [-1] * world.n
    sizes = []
    for root in range(world.n):
        if root in blocked or component[root] >= 0:
            continue
        cid = len(sizes)
        component[root] = cid
        q = [root]
        for p in q:
            x, y = p % world.w, p // world.w
            for dx, dy in _DXY:
                nx, ny = x + dx, y + dy
                v = ny * world.w + nx
                if (0 <= nx < world.w and 0 <= ny < world.h
                        and v not in blocked and component[v] < 0):
                    component[v] = cid
                    q.append(v)
        sizes.append(len(q))
    counts = Counter(component[p] for p in st
                     if p >= 0 and component[p] >= 0)
    matched = sum(v // 2 for v in counts.values())
    reach = sum(sizes[c] * k for c, k in counts.items())
    n = world.n
    return 4 * n * n * matched + sum(v * v for v in sizes) + n * reach


def _box_candidates(world, st, history, deadline):
    current = _topology_score(world, world.boxes, st)
    mask = world.mask(st)
    candidates = []
    for box in world.boxes:
        blocked = (world.walls | world.hazards | set(world.switches)
                   | (world.boxset - {box})
                   | {p for p in st if p >= 0}
                   | {p for p, bit in world.gates.items() if bit & ~mask})
        parent = {}
        distance = {box: 0}
        q = deque([box])
        while q:
            p = q.popleft()
            x, y = p % world.w, p // world.w
            for a, (dx, dy) in enumerate(_DXY):
                nx, ny = x + dx, y + dy
                v = ny * world.w + nx
                if (0 <= nx < world.w and 0 <= ny < world.h
                        and v not in blocked and v not in distance):
                    distance[v] = distance[p] + 1
                    parent[v] = (p, a)
                    q.append(v)

        local = []
        for target, length in distance.items():
            if target == box:
                continue
            nb = tuple(sorted((world.boxset - {box}) | {target}))
            if history.get(nb, 0) >= 2:
                continue
            score = _topology_score(world, nb, st)
            gain = score - current
            efficiency = gain / (length + 2)
            efficiency -= history.get(nb, 0) * world.n * world.n
            path = _path(parent, target)
            local.append((efficiency, gain, -length, box, target, path, nb))
            if time.monotonic() >= deadline:
                break
        local.sort(key=lambda z: z[:3], reverse=True)
        candidates.extend(local[:5])
        if time.monotonic() >= deadline:
            break
    candidates.sort(key=lambda z: z[:3], reverse=True)
    return candidates[:12]


def _click(p, view):
    s, ox, oy, w, h = view
    return ("ACTION6", int(oy + (p // w) * s + s // 2),
            int(ox + (p % w) * s + s // 2))


def _arrow(a):
    return ("ACTION" + str(a + 1), None, None)


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 8.6
    available = set(available)
    data = _read(np.asarray(obs), memory)
    observed = frozenset(data["agents"])

    oldfp = memory.get("fp")
    if oldfp is not None and oldfp != data["fp"]:
        memory.pop("queue", None)
    memory["fp"] = data["fp"]

    if ("expected_boxes" in memory
            and frozenset(data["boxes"]) != memory["expected_boxes"]):
        memory.pop("queue", None)

    hypotheses = []
    for hyp in memory.get("predictions", ()):
        if _signature(hyp) == observed:
            hypotheses.append(hyp)
    hypotheses = list(dict.fromkeys(hypotheses))
    if not hypotheses:
        hypotheses = _hypotheses(data["agents"], data["selected"])
        memory.pop("queue", None)
        memory.pop("chosen", None)
    if not hypotheses:
        return ("ACTION5", None, None)

    chosen = memory.get("chosen")
    if chosen not in hypotheses:
        if chosen is not None:
            memory.pop("queue", None)
        chosen = hypotheses[0]

    key = (data["fp"], data["boxes"])
    if memory.get("world_key") == key:
        world = memory["world"]
    else:
        world = _World(data)
        memory["world_key"], memory["world"] = key, world

    history = memory.setdefault("box_history", {})
    if memory.get("last_boxes") != data["boxes"]:
        history[data["boxes"]] = history.get(data["boxes"], 0) + 1
        memory["last_boxes"] = data["boxes"]

    def emit(action, selected_hyp=chosen):
        name, row, col = action
        if name not in available:
            name, row, col = ("ACTION5", None, None)
            if name not in available:
                alternatives = sorted(a for a in available if a != "RESET")
                name = alternatives[0] if alternatives else "RESET"

        predictions = []
        next_chosen = None
        expected_boxes = set(data["boxes"])
        a = int(name[6:]) - 1 if name in (
            "ACTION1", "ACTION2", "ACTION3", "ACTION4") else None

        if a is not None and data["selected"] is None:
            for hyp in hypotheses:
                result = world.step(hyp[0], a)
                if result is None:
                    continue
                nh = (result[0], tuple(sorted(set(hyp[1]) | set(result[1]))))
                predictions.append(nh)
                if hyp == selected_hyp:
                    next_chosen = nh
        else:
            predictions = list(hypotheses)
            next_chosen = selected_hyp
            if a is not None and data["selected"] is not None:
                p = data["selected"]
                target = world.box_target(p, a, selected_hyp[0])
                expected_boxes.discard(p)
                expected_boxes.add(target)

        if name == "RESET":
            view = memory.get("view")
            memory.clear()
            if view is not None:
                memory["view"] = view
        else:
            memory["predictions"] = predictions
            memory["chosen"] = next_chosen
            memory["expected_boxes"] = frozenset(expected_boxes)
        return name, row, col

    neutral = next((p for p in chosen[0] if p >= 0),
                   next(iter(observed), 0))

    queue = memory.get("queue", [])
    if queue:
        action = queue.pop(0)
        memory["queue"] = queue
        return emit(action)

    # A perturbed initial state may have a movable obstacle selected.
    if data["selected"] is not None:
        return emit(_click(neutral, memory["view"]))

    # Learn control identities with safe, informative movements.
    if len(hypotheses) > 1 and memory.get("probes", 0) < 5:
        best = None
        for a in range(4):
            groups = Counter()
            moved = 0
            safe = True
            for hyp in hypotheses:
                result = world.step(hyp[0], a)
                if result is None:
                    safe = False
                    break
                nh = (result[0], tuple(set(hyp[1]) | set(result[1])))
                groups[_signature(nh)] += 1
                moved += result[0] != hyp[0]
            if not safe:
                continue
            information = len(hypotheses)**2 - sum(v * v for v in groups.values())
            score = (information, moved)
            if information and (best is None or score > best[0]):
                best = (score, a)
        if best is not None:
            memory["probes"] = memory.get("probes", 0) + 1
            return emit(_arrow(best[1]))

    # Search the synchronized movement puzzle with obstacles fixed.
    search_deadline = min(deadline, time.monotonic() + (
        4.5 if data["boxes"] else 7.6))
    path, solved, exhausted = _solve(world, chosen[0], search_deadline)
    if path:
        queue = [_arrow(a) for a in path]
        memory["queue"] = queue[1:]
        memory["stalls"] = 0
        return emit(queue[0])

    # Relocate independently controllable obstacles to useful parking cells.
    if data["boxes"] and time.monotonic() < deadline - 0.3:
        candidates = _box_candidates(
            world, chosen[0], history, min(deadline - 0.2,
                                          time.monotonic() + 1.2))
        best_plan = None
        best_length = _INF
        for candidate in candidates[:5]:
            if time.monotonic() >= deadline - 0.25:
                break
            _, gain, _, box, target, movement, newboxes = candidate
            trial = _World(data, newboxes)
            tail, success, _ = _solve(
                trial, chosen[0], min(deadline - 0.1,
                                     time.monotonic() + 0.8), maxnodes=30000)
            if success and tail is not None:
                plan = ([_click(box, memory["view"])]
                        + [_arrow(a) for a in movement]
                        + [_click(neutral, memory["view"])]
                        + [_arrow(a) for a in tail])
                if len(plan) < best_length:
                    best_plan, best_length = plan, len(plan)

        if best_plan is None and candidates:
            # Several bottlenecks may have to be cleared before a full
            # movement solution exists.
            candidate = candidates[0]
            _, gain, _, box, target, movement, newboxes = candidate
            if gain > 0 or history.get(newboxes, 0) == 0:
                best_plan = ([_click(box, memory["view"])]
                             + [_arrow(a) for a in movement]
                             + [_click(neutral, memory["view"])])

        if best_plan:
            memory["queue"] = best_plan[1:]
            memory["stalls"] = 0
            return emit(best_plan[0])

    # Conservative exploration is also useful when a switch was initially
    # concealed under a piece, or an identity remained ambiguous.
    visits = memory.setdefault("visits", {})
    current_key = (tuple(sorted(observed)), data["boxes"])
    visits[current_key] = visits.get(current_key, 0) + 1
    options = []
    for a in range(4):
        transition = world.step(chosen[0], a)
        if transition is None or transition[0] == chosen[0]:
            continue
        nh = (transition[0], tuple(set(chosen[1]) | set(transition[1])))
        sig = tuple(sorted(_signature(nh)))
        count = visits.get((sig, data["boxes"]), 0)
        removed = sum(p < 0 for p in transition[0])
        options.append((count, -removed, a))
    memory["stalls"] = memory.get("stalls", 0) + 1

    if (memory["stalls"] > 12 and "RESET" in available
            and not memory.get("reset_used")):
        result = emit(("RESET", None, None))
        memory["reset_used"] = True
        return result
    if options:
        options.sort()
        return emit(_arrow(options[0][2]))
    return emit(("ACTION5", None, None))

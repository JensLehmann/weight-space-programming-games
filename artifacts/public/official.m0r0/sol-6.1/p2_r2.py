import math
import time
import heapq
import itertools
from collections import deque
import numpy as np


_SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_GATE_BITS = {14: 1, 12: 2, 15: 4}


def _geometry(obs):
    candidates = {}
    for color in (10, 1):
        flat = (obs == color).ravel()
        seen = set()
        for start in np.flatnonzero(flat):
            start = int(start)
            if start in seen:
                continue
            seen.add(start)
            stack = [start]
            xs, ys = [], []
            while stack:
                p = stack.pop()
                y, x = divmod(p, 64)
                xs.append(x)
                ys.append(y)
                for q in (p - 64, p + 64,
                          p - 1 if x else -1,
                          p + 1 if x < 63 else -1):
                    if 0 <= q < 4096 and flat[q] and q not in seen:
                        seen.add(q)
                        stack.append(q)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if len(xs) > 300 or bw > 24 or bh > 24:
                continue
            if len(xs) < 4 or len(xs) < 0.9 * bw * bh:
                continue
            s = math.gcd(bw, bh)
            if s == 1 and (y0 == 1 or y1 == 62):
                s = bw
            if 2 <= s <= 8:
                key = (s, x0 % s, y0 % s)
                candidates[key] = candidates.get(key, 0) + 1
    if candidates:
        s, ox, oy = max(candidates, key=lambda k: (candidates[k], k[0]))
    else:
        floor = obs == 5
        ex = np.sum(floor[1:63, 1:] != floor[1:63, :-1], axis=0)
        ey = np.sum(floor[1:, :] != floor[:-1, :], axis=1)
        ex[31] = 0
        ey[0] = ey[31] = ey[62] = 0
        best = None
        for s0 in range(2, 9):
            sx = [sum(ex[x - 1] for x in range(1, 64)
                      if x % s0 == r) for r in range(s0)]
            sy = [sum(ey[y - 1] for y in range(1, 64)
                      if y % s0 == r) for r in range(s0)]
            rx = max(range(s0), key=lambda r: sx[r])
            ry = max(range(s0), key=lambda r: sy[r])
            score = (sx[rx] / max(1, float(sum(ex))) +
                     sy[ry] / max(1, float(sum(ey))) + 0.003 * s0)
            item = (score, s0, rx, ry)
            if best is None or item > best:
                best = item
        _, s, ox, oy = best
    width = max(1, (64 - 2 * ox) // s)
    height = max(1, (64 - 2 * oy) // s)
    return s, ox, oy, width, height


def _scene(obs, memory):
    if "geometry" not in memory:
        memory["geometry"] = _geometry(obs)
    s, ox, oy, width, height = memory["geometry"]
    count = width * height

    corners = [
        int(obs[1, 0]), int(obs[1, 63]),
        int(obs[62, 0]), int(obs[62, 63])
    ]
    for k, color in enumerate(corners):
        if color == 5:
            y0, y1 = (1, 32) if k < 2 else (32, 63)
            x0, x1 = (0, 32) if k % 2 == 0 else (32, 64)
            vals = obs[y0:y1, x0:x1].ravel()
            vals = vals[vals != 5]
            if len(vals):
                corners[k] = int(np.bincount(vals, minlength=16).argmax())

    gates = {c: set(v) for c, v in memory.get("gates", {}).items()}
    switches = dict(memory.get("switches", {}))
    hazards = set(memory.get("hazards", ()))
    walls = [False] * count
    agents = set()
    agent_colors = {}
    blockers = []
    selected = -1
    colored = {}

    for gy in range(height):
        for gx in range(width):
            p = gy * width + gx
            x, y = ox + gx * s, oy + gy * s
            ya, yb = max(1, y), min(63, y + s)
            tile = obs[ya:yb, x:x + s]
            if tile.size == 0:
                walls[p] = True
                continue
            yy = np.arange(ya, yb)[:, None]
            xx = np.arange(x, x + s)[None, :]
            expect = np.where(
                yy < 32,
                np.where(xx < 32, corners[0], corners[1]),
                np.where(xx < 32, corners[2], corners[3])
            )
            wall_fraction = float(np.mean(tile == expect))
            vals = np.bincount(tile.ravel(), minlength=16)
            dominant = int(vals.argmax())
            fraction = vals[dominant] / tile.size

            full = obs[y:y + s, x:x + s]
            center = int(obs[min(63, y + s // 2),
                             min(63, x + s // 2)])
            is_block = False
            if s >= 3 and center in (9, 11) and full.shape == (s, s):
                border = np.concatenate(
                    (full[0], full[-1], full[:, 0], full[:, -1]))
                interior = full[1:-1, 1:-1]
                is_block = (
                    np.mean(border == 5) >= 0.9 and
                    interior.size > 0 and np.all(interior == center)
                )
            if is_block:
                blockers.append(p)
                if center == 11:
                    selected = p
                continue

            if wall_fraction >= 0.97:
                walls[p] = True
                continue
            if dominant in (10, 1) and fraction >= 0.7:
                agents.add(p)
                agent_colors[p] = dominant
                continue
            if dominant == 5 and fraction >= 0.97:
                continue

            parity = (xx + yy) % 2
            odd = tile[parity == 1]
            even = tile[parity == 0]
            checker = (
                len(odd) > 0 and len(even) > 0 and
                np.all(odd == 5) and np.all(even == even[0]) and
                int(even[0]) != 5 and
                np.count_nonzero(tile != 5) >= tile.size / 3
            )
            if checker:
                hazards.add(p)
                continue
            if dominant in _GATE_BITS and fraction >= 0.7:
                colored[p] = dominant
                continue
            if fraction > 0.65 and dominant != 5:
                walls[p] = True

    for color in _GATE_BITS:
        remaining = {p for p, c in colored.items() if c == color}
        while remaining:
            start = remaining.pop()
            component = {start}
            todo = [start]
            while todo:
                p = todo.pop()
                y, x = divmod(p, width)
                ns = []
                if x:
                    ns.append(p - 1)
                if x + 1 < width:
                    ns.append(p + 1)
                if y:
                    ns.append(p - width)
                if y + 1 < height:
                    ns.append(p + width)
                for q in ns:
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        todo.append(q)
            if len(component) >= 2 or component & gates.get(color, set()):
                gates.setdefault(color, set()).update(component)
                for p in component:
                    switches.pop(p, None)
            else:
                for p in component:
                    switches[p] = color

    special = set(hazards) | set(switches) | agents | set(blockers)
    for cells in gates.values():
        special.update(cells)
    for p in special:
        if 0 <= p < count:
            walls[p] = False

    gate_at = [0] * count
    switch_at = [0] * count
    for color, cells in gates.items():
        for p in cells:
            if 0 <= p < count:
                gate_at[p] |= _GATE_BITS[color]
    for p, color in switches.items():
        if 0 <= p < count:
            switch_at[p] |= _GATE_BITS[color]

    nav = [[0] * count for _ in range(4)]
    for p in range(count):
        y, x = divmod(p, width)
        candidates = (
            p - width if y else p,
            p + width if y + 1 < height else p,
            p - 1 if x else p,
            p + 1 if x + 1 < width else p
        )
        for a, q in enumerate(candidates):
            nav[a][p] = p if walls[q] else q

    memory["gates"] = gates
    memory["switches"] = switches
    memory["hazards"] = hazards
    world = {
        "width": width, "height": height, "count": count,
        "walls": tuple(walls), "hazards": frozenset(hazards),
        "gate": tuple(gate_at), "switch": tuple(switch_at),
        "nav": nav
    }
    world["signature"] = (
        width, height, world["walls"], world["hazards"],
        world["gate"], world["switch"]
    )
    return (world, tuple(sorted(agents)), agent_colors,
            tuple(sorted(blockers)), selected)


def _open_mask(tokens, active, world):
    result = 0
    for i, p in enumerate(tokens):
        if active & (1 << i) and p >= 0:
            result |= world["switch"][p]
    return result


def _move_tokens(tokens, active, action, blockers, world, keep=False):
    old = tuple(tokens)
    new = list(tokens)
    opened = _open_mask(old, active, world)
    blocked = set(blockers)

    for i, p in enumerate(old):
        if not active & (1 << i):
            continue
        sx, sy = _SIGNS[i]
        if action == 1:
            a = 0 if sy > 0 else 1
        elif action == 2:
            a = 1 if sy > 0 else 0
        elif action == 3:
            a = 2 if sx > 0 else 3
        else:
            a = 3 if sx > 0 else 2
        q = world["nav"][a][p]
        if q in blocked or world["gate"][q] & ~opened:
            q = p
        if q in world["hazards"]:
            return None
        new[i] = q

    width = world["width"]
    indices = [i for i in range(len(old)) if active & (1 << i)]
    for z, i in enumerate(indices):
        if not active & (1 << i):
            continue
        for j in indices[z + 1:]:
            if not active & (1 << j):
                continue
            if abs(old[i] - old[j]) == 1 and old[i] // width == old[j] // width:
                if new[i] == old[j] or new[j] == old[i]:
                    xi, yi = new[i] % width, new[i] // width
                    xj, yj = new[j] % width, new[j] // width
                    q = ((yi + yj) // 2) * width + (xi + xj) // 2
                    new[i] = new[j] = q

    groups = {}
    for i in indices:
        groups.setdefault(new[i], []).append(i)
    for group in groups.values():
        if len(group) >= 2:
            for i in group[:2]:
                active &= ~(1 << i)
            for i in group[2:]:
                new[i] = old[i]

    if not keep:
        for i in range(len(new)):
            if not active & (1 << i):
                new[i] = -1
    return tuple(new), active


def _mask(tokens):
    return sum(1 << i for i, p in enumerate(tokens) if p >= 0)


def _canonical(hypothesis):
    tokens, active = hypothesis
    return tuple(p if active & (1 << i) else -1
                 for i, p in enumerate(tokens))


def _initial_hypotheses(agents, colors, world):
    if len(agents) == 2:
        ordered = tuple(sorted(agents, key=lambda p: (p % world["width"], p)))
        return [(p, 3) for p in itertools.permutations(ordered)]
    if len(agents) == 4:
        ordered = tuple(sorted(
            agents, key=lambda p: (-(p // world["width"]), p % world["width"])))
        return [(p, 15) for p in itertools.permutations(ordered)]
    if len(agents) == 3:
        result = []
        center = (world["height"] // 2) * world["width"] + world["width"] // 2
        frozen_candidates = sorted(agents, key=lambda p: abs(p - center))
        if any(colors.get(p) == 1 for p in agents):
            frozen_candidates = [
                p for p in frozen_candidates if colors.get(p) == 10]
        for frozen in frozen_candidates:
            moving = [p for p in agents if p != frozen]
            for live_ids in itertools.combinations(range(4), 2):
                active = sum(1 << i for i in live_ids)
                for pair in itertools.permutations(moving):
                    pos = [frozen] * 4
                    for i, p in zip(live_ids, pair):
                        pos[i] = p
                    result.append((tuple(pos), active))
        return result
    if len(agents) == 1:
        return [((agents[0], agents[0]), 0)]
    return []


def _track(memory, agents, colors, world, selected):
    old_hyps = memory.get("hypotheses", [])
    old_world = memory.get("old_world")
    last = memory.get("last")
    matched = []
    if old_hyps and old_world is not None and last is not None:
        action, target = last
        for environment in (old_world, world):
            for tokens, active in old_hyps:
                predicted = (tokens, active)
                if action in (1, 2, 3, 4) and memory.get("old_selected", -1) < 0:
                    predicted = _move_tokens(
                        tokens, active, action,
                        memory.get("old_blockers", ()), environment, True)
                if predicted is None:
                    continue
                pp, aa = predicted
                if tuple(sorted(set(pp))) != agents:
                    continue
                if selected >= 0:
                    live_positions = {
                        p for i, p in enumerate(pp) if aa & (1 << i)}
                    if any(colors.get(p) != 1 for p in live_positions):
                        continue
                hypothesis = (pp, aa)
                if hypothesis not in matched:
                    matched.append(hypothesis)
            if matched:
                break
    if not matched:
        matched = _initial_hypotheses(agents, colors, world)
    memory["hypotheses"] = matched
    return matched


def _step_state(state, action, target, world):
    tokens, blockers, selected = state
    if action == 6:
        return tokens, blockers, target if target in blockers else -1
    if action not in (1, 2, 3, 4):
        return state

    active = _mask(tokens)
    if selected >= 0:
        q = world["nav"][action - 1][selected]
        opened = _open_mask(tokens, active, world)
        occupied = set(blockers)
        occupied.discard(selected)
        occupied.update(p for p in tokens if p >= 0)
        if (q in occupied or q in world["hazards"] or
                world["switch"][q] or world["gate"][q] & ~opened):
            q = selected
        if q == selected:
            return state
        new_blocks = tuple(sorted(
            q if p == selected else p for p in blockers))
        return tokens, new_blocks, q

    moved = _move_tokens(tokens, active, action, blockers, world)
    if moved is None:
        return None
    return moved[0], blockers, -1


class _Heuristic:
    def __init__(self, world):
        self.world = world
        self.key = (world["width"], world["height"],
                    world["walls"], world["hazards"])
        self.cells = [
            p for p in range(world["count"])
            if not world["walls"][p] and p not in world["hazards"]
        ]
        self.local = {p: i for i, p in enumerate(self.cells)}
        self.size = len(self.cells)
        self.tables = {}
        self.cache = {}
        self.graph_distances = {}
        self.moves = []
        for a in range(4):
            row = []
            for p in self.cells:
                q = world["nav"][a][p]
                row.append(self.local.get(q, self.local[p]))
            self.moves.append(row)
        self.safe, self.degree = self._parking_cells()

    def _parking_cells(self):
        """Find articulation points: never permanently park in a bottleneck."""
        n = self.size
        graph = [[] for _ in range(n)]
        for i in range(n):
            graph[i] = list({self.moves[a][i] for a in range(4)
                             if self.moves[a][i] != i})
        discovery = [-1] * n
        low = [0] * n
        parent = [-1] * n
        cut = set()
        clock = [0]

        def dfs(v):
            discovery[v] = low[v] = clock[0]
            clock[0] += 1
            children = 0
            for u in graph[v]:
                if discovery[u] < 0:
                    parent[u] = v
                    children += 1
                    dfs(u)
                    low[v] = min(low[v], low[u])
                    if parent[v] >= 0 and low[u] >= discovery[v]:
                        cut.add(v)
                elif u != parent[v]:
                    low[v] = min(low[v], discovery[u])
            if parent[v] < 0 and children > 1:
                cut.add(v)

        for i in range(n):
            if discovery[i] < 0:
                dfs(i)
        safe = {self.cells[i] for i in range(n) if i not in cut}
        degree = {self.cells[i]: len(graph[i]) for i in range(n)}
        return safe, degree

    def _table(self, rx, ry):
        key = (rx, ry)
        if key in self.tables:
            return self.tables[key]
        n = self.size
        distance = [10000] * (n * n)
        queue = deque()
        for i in range(n):
            distance[i * n + i] = 0
            queue.append(i * n + i)

        if rx < 0:
            width = self.world["width"]
            for i, p in enumerate(self.cells):
                q = p + 1
                if p // width == q // width and q in self.local:
                    j = self.local[q]
                    for z in (i * n + j, j * n + i):
                        if distance[z] > 1:
                            distance[z] = 1
                            queue.append(z)

        inverses = []
        for a in range(4):
            if a < 2:
                b = a if ry > 0 else 1 - a
            else:
                b = a if rx > 0 else 5 - a
            first = [[] for _ in range(n)]
            second = [[] for _ in range(n)]
            for i in range(n):
                first[self.moves[a][i]].append(i)
                second[self.moves[b][i]].append(i)
            inverses.append((first, second))

        while queue:
            z = queue.popleft()
            i, j = divmod(z, n)
            nd = distance[z] + 1
            for first, second in inverses:
                for pi in first[i]:
                    base = pi * n
                    for pj in second[j]:
                        nz = base + pj
                        if nd < distance[nz]:
                            distance[nz] = nd
                            queue.append(nz)
        self.tables[key] = distance
        return distance

    def _graph_distance(self, p, q):
        if p not in self.graph_distances:
            distances = {p: 0}
            todo = deque([p])
            while todo:
                v = todo.popleft()
                for a in range(4):
                    u = self.world["nav"][a][v]
                    if u not in self.local or u in distances:
                        continue
                    distances[u] = distances[v] + 1
                    todo.append(u)
            self.graph_distances[p] = distances
        return self.graph_distances[p].get(q, 20000)

    def pair(self, i, p, j, q):
        if p not in self.local or q not in self.local:
            return 10000
        rx = _SIGNS[i][0] * _SIGNS[j][0]
        ry = _SIGNS[i][1] * _SIGNS[j][1]
        table = self._table(rx, ry)
        value = table[self.local[p] * self.size + self.local[q]]
        if value >= 10000:
            # Movable blocks can create synchronization unavailable in the
            # empty-maze relaxation; do not incorrectly reject such states.
            distance = self._graph_distance(p, q)
            if distance < 20000:
                return (distance + 1) // 2 + 5
        return value

    def __call__(self, tokens):
        if tokens in self.cache:
            return self.cache[tokens]
        live = [i for i, p in enumerate(tokens) if p >= 0]
        if not live:
            value = 0
        elif len(live) == 2:
            i, j = live
            value = self.pair(i, tokens[i], j, tokens[j])
        elif len(live) == 4:
            a, b, c, d = live
            values = []
            for i, j, k, l in ((a, b, c, d), (a, c, b, d), (a, d, b, c)):
                x = self.pair(i, tokens[i], j, tokens[j])
                y = self.pair(k, tokens[k], l, tokens[l])
                values.append(max(x, y) + 0.25 * min(x, y))
            value = min(values)
        else:
            value = 10000
        self.cache[tokens] = value
        return value


def _reconstruct(parent, finish):
    answer = []
    node = finish
    while parent[node] is not None:
        previous, action = parent[node]
        answer.append(action)
        node = previous
    answer.reverse()
    return answer


def _reconstruct_segments(parent, finish):
    segments = []
    node = finish
    while parent[node] is not None:
        previous, segment = parent[node]
        segments.append(segment)
        node = previous
    answer = []
    for segment in reversed(segments):
        answer.extend(segment)
    return answer


def _search_tokens(tokens, blockers, world, heuristic, until, cap=90000):
    h = heuristic(tokens)
    if h >= 10000:
        return None
    heap = [(1.8 * h, h, 0, tokens)]
    costs = {tokens: 0}
    parent = {tokens: None}
    expanded = 0
    while heap:
        _, _, g, current = heapq.heappop(heap)
        if costs.get(current) != g:
            continue
        if all(p < 0 for p in current):
            return _reconstruct(parent, current)
        expanded += 1
        if expanded > cap:
            return None
        if expanded % 64 == 0 and time.monotonic() >= until:
            return None
        if g >= 140:
            continue
        active = _mask(current)
        for action in (1, 2, 3, 4):
            moved = _move_tokens(current, active, action, blockers, world)
            if moved is None:
                continue
            nxt = moved[0]
            if nxt == current:
                continue
            ng = g + 1
            if ng >= costs.get(nxt, 10 ** 9):
                continue
            nh = heuristic(nxt)
            if nh >= 10000:
                continue
            costs[nxt] = ng
            parent[nxt] = (current, (action, -1))
            heapq.heappush(heap, (ng + 1.8 * nh, nh, ng, nxt))
    return None


def _route_cells(tokens, route, world):
    cells = {p for p in tokens if p >= 0}
    current = tokens
    if route:
        for action, _ in route:
            moved = _move_tokens(current, _mask(current), action, (), world)
            if moved is None:
                break
            current = moved[0]
            cells.update(p for p in current if p >= 0)
    return cells


def _relocations(tokens, blockers, world, heuristic, future,
                 temporary=False, per_block=5):
    """Return entire click/move/deselect macros for useful block locations."""
    active = _mask(tokens)
    opened = _open_mask(tokens, active, world)
    live_cells = {p for p in tokens if p >= 0}
    results = []

    for block in blockers:
        forbidden = set(blockers) - {block}
        forbidden.update(live_cells)
        parent = {block: None}
        depth = {block: 0}
        queue = deque([block])
        while queue:
            p = queue.popleft()
            for action in (1, 2, 3, 4):
                q = world["nav"][action - 1][p]
                if q == p or q in parent:
                    continue
                if (q in forbidden or q in world["hazards"] or
                        world["switch"][q] or world["gate"][q] & ~opened):
                    continue
                parent[q] = (p, action)
                depth[q] = depth[p] + 1
                queue.append(q)

        ranked = []
        for target, distance in depth.items():
            if target == block or target not in heuristic.safe:
                continue
            if world["gate"][target]:
                continue
            degree = heuristic.degree.get(target, 4)
            score = (
                distance +
                (40 if target in future else 0) +
                (0.25 * degree) -
                (1.0 if degree <= 1 else 0)
            )
            ranked.append((score, distance, target))
        ranked.sort()
        targets = [item[2] for item in ranked[:per_block]]

        # Include a reachable dead end even when nearer chamber cells dominate.
        leaves = [item for item in ranked
                  if heuristic.degree.get(item[2], 4) <= 1]
        if leaves and leaves[0][2] not in targets:
            targets.append(leaves[0][2])

        if temporary:
            for action in (1, 2, 3, 4):
                q = world["nav"][action - 1][block]
                if q != block and q in parent and q not in targets:
                    targets.append(q)

        for target in targets:
            path = []
            p = target
            while parent[p] is not None:
                previous, action = parent[p]
                path.append((action, -1))
                p = previous
            path.reverse()
            macro = [(6, block)] + path + [(6, -1)]
            new_blocks = tuple(sorted(
                target if p == block else p for p in blockers))
            safety = 0 if target in heuristic.safe else 8
            priority = (len(macro) + safety +
                        (35 if target in future else 0))
            results.append((priority, new_blocks, macro))
    results.sort(key=lambda item: item[0])
    return results


def _clear_blocks(tokens, blockers, world, heuristic, relaxed, until):
    """
    Search block configurations while holding the mirrored pieces stationary.
    This is much smaller than unrestricted joint search and avoids greedy
    parking in narrow corridors.
    """
    future = _route_cells(tokens, relaxed, world)

    def obstruction(bs):
        return sum(p in future for p in bs)

    heap = [(7 * obstruction(blockers), 0, blockers)]
    costs = {blockers: 0}
    parent = {blockers: None}
    expanded = 0
    while heap and time.monotonic() < until:
        _, g, current = heapq.heappop(heap)
        if costs.get(current) != g:
            continue
        expanded += 1
        if expanded > 90:
            break

        if current != blockers:
            budget = min(until, time.monotonic() + 0.20)
            cap = 14000 if len(tokens) <= 2 else 18000
            solution = _search_tokens(
                tokens, current, world, heuristic, budget, cap)
            if solution is not None:
                return _reconstruct_segments(parent, current) + solution

        if g > 65:
            continue
        moves = _relocations(
            tokens, current, world, heuristic, future,
            temporary=False, per_block=4)
        for _, nxt, macro in moves:
            ng = g + len(macro)
            if ng >= costs.get(nxt, 10 ** 9):
                continue
            costs[nxt] = ng
            parent[nxt] = (current, macro)
            heapq.heappush(
                heap, (ng + 7 * obstruction(nxt), ng, nxt))
    return None


def _search_joint(state, world, heuristic, until, cap=35000):
    """Joint search with multi-step block relocations as macro actions."""
    tokens, blockers, selected = state
    start = (tokens, blockers)
    prefix = [(6, -1)] if selected >= 0 else []
    h = heuristic(tokens)
    if h >= 10000:
        return None
    serial = 0
    heap = [(2.4 * h, h, 0, serial, start)]
    costs = {start: 0}
    parent = {start: None}
    expanded = 0

    while heap:
        _, _, g, _, current = heapq.heappop(heap)
        if costs.get(current) != g:
            continue
        ts, bs = current
        if all(p < 0 for p in ts):
            return prefix + _reconstruct_segments(parent, current)
        expanded += 1
        if expanded > cap:
            return None
        if expanded % 16 == 0 and time.monotonic() >= until:
            return None
        if g >= 135:
            continue

        successors = []
        active = _mask(ts)
        future = {p for p in ts if p >= 0}
        for action in (1, 2, 3, 4):
            moved = _move_tokens(ts, active, action, bs, world)
            relaxed = _move_tokens(ts, active, action, (), world)
            if relaxed is not None:
                future.update(p for p in relaxed[0] if p >= 0)
            if moved is not None and moved[0] != ts:
                successors.append(
                    ((moved[0], bs), [(action, -1)]))

        for _, new_blocks, macro in _relocations(
                ts, bs, world, heuristic, future,
                temporary=True, per_block=3):
            successors.append(((ts, new_blocks), macro))

        for nxt, segment in successors:
            ng = g + len(segment)
            if ng >= costs.get(nxt, 10 ** 9):
                continue
            nh = heuristic(nxt[0])
            if nh >= 10000:
                continue
            costs[nxt] = ng
            parent[nxt] = (current, segment)
            serial += 1
            heapq.heappush(
                heap, (ng + 2.4 * nh, nh, ng, serial, nxt))
    return None


def _plan(state, world, heuristic, deadline, visits):
    tokens, blockers, selected = state
    prefix = [(6, -1)] if selected >= 0 else []
    now = time.monotonic()
    fixed = _search_tokens(
        tokens, blockers, world, heuristic,
        min(deadline, now + 1.5))
    if fixed is not None:
        return prefix + fixed

    if not blockers:
        return None

    now = time.monotonic()
    relaxed = _search_tokens(
        tokens, (), world, heuristic, min(deadline, now + 0.8),
        cap=60000)

    # Try coherent, complete plans for parking several blocks before moving
    # the pieces. This prevents a block being left between the two pieces.
    now = time.monotonic()
    clearance_budget = 3.1 if len(tokens) <= 2 else 2.4
    clear = _clear_blocks(
        tokens, blockers, world, heuristic, relaxed,
        min(deadline - 0.4, now + clearance_budget))
    if clear is not None:
        return prefix + clear

    # Repeated configurations require actual joint reasoning, not another
    # locally attractive relocation.
    if visits.get(state, 0) >= 2:
        return _search_joint(state, world, heuristic, deadline)

    if relaxed:
        real = tokens
        imagined = tokens
        safe_prefix = []
        for action, target in relaxed:
            rn = _move_tokens(real, _mask(real), action, blockers, world)
            inn = _move_tokens(
                imagined, _mask(imagined), action, (), world)
            if rn is None or inn is None or rn[0] != inn[0]:
                break
            nxt_state = (rn[0], blockers, -1)
            if visits.get(nxt_state, 0) >= 2:
                break
            safe_prefix.append((action, target))
            real, imagined = rn[0], inn[0]
        if safe_prefix:
            return prefix + safe_prefix

        future = _route_cells(tokens, relaxed, world)
        moves = _relocations(
            tokens, blockers, world, heuristic, future,
            temporary=False, per_block=5)
        for _, new_blocks, macro in moves:
            end = (tokens, new_blocks, -1)
            if visits.get(end, 0) == 0:
                return prefix + macro

    return _search_joint(state, world, heuristic, deadline)


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 9.0
    obs = np.asarray(obs, dtype=np.int16)
    available_set = set(available)

    world, agents, colors, blockers, selected = _scene(obs, memory)
    hypotheses = _track(memory, agents, colors, world, selected)

    if not hypotheses:
        memory["plan"] = []
        action = next((a for a in ("ACTION1", "ACTION4", "ACTION2", "ACTION3")
                       if a in available_set), available[0])
        memory["last"] = None
        return action, None, None

    state = (_canonical(hypotheses[0]), blockers, selected)
    visits = memory.setdefault("visits", {})
    visits[state] = visits.get(state, 0) + 1

    queue = memory.get("plan", [])
    if (memory.get("expected") != state or
            memory.get("plan_signature") != world["signature"]):
        queue = []

    key = (world["width"], world["height"],
           world["walls"], world["hazards"])
    heuristic = memory.get("heuristic")
    if heuristic is None or heuristic.key != key:
        heuristic = _Heuristic(world)
        memory["heuristic"] = heuristic

    if not queue:
        queue = _plan(
            state, world, heuristic, deadline, visits) or []

    if queue:
        action, target = queue.pop(0)
        memory["failures"] = 0
    else:
        failures = memory.get("failures", 0) + 1
        memory["failures"] = failures
        if selected >= 0:
            action, target = 6, -1
        else:
            choices = []
            for a in (1, 2, 3, 4):
                nxt = _step_state(state, a, -1, world)
                if nxt is not None and nxt != state:
                    score = (
                        10 * visits.get(nxt, 0) +
                        heuristic(nxt[0]) +
                        0.01 * ((a + failures) % 4)
                    )
                    choices.append((score, a))
            if choices:
                action = min(choices)[1]
            else:
                action = (1, 4, 2, 3)[(failures - 1) % 4]
            target = -1

    name = "ACTION" + str(action)
    if name not in available_set:
        memory["plan"] = []
        fallback = next((a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                         if a in available_set), available[0])
        memory["last"] = None
        return fallback, None, None

    memory["expected"] = _step_state(state, action, target, world)
    memory["plan"] = queue
    memory["plan_signature"] = world["signature"]
    memory["last"] = (action, target)
    memory["old_world"] = world
    memory["old_blockers"] = blockers
    memory["old_selected"] = selected

    if action != 6:
        return name, None, None

    if target < 0:
        live = [p for p in state[0] if p >= 0]
        if live:
            target = live[0]
        elif agents:
            target = agents[0]
        else:
            target = next(
                (p for p in range(world["count"])
                 if p not in blockers and not world["walls"][p]), 0)

    s, ox, oy, width, height = memory["geometry"]
    gy, gx = divmod(target, width)
    row = min(63, max(0, oy + gy * s + s // 2))
    col = min(63, max(0, ox + gx * s + s // 2))
    return name, int(row), int(col)

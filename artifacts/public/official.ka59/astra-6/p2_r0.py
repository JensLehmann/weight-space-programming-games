import math
import time
import heapq
import itertools
from collections import deque, defaultdict
import numpy as np


# Directions correspond to ACTION1 through ACTION4.
_DX = (0, 0, -1, 1)
_DY = (-1, 1, 0, 0)
_OPP = (1, 0, 3, 2)
_INF = 100000


def _components(a, colors):
    h, w = a.shape
    allowed = np.isin(a, list(colors))
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(allowed)):
        x, y = int(xx), int(yy)
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(x, y)]
        pts = []
        lo_x = hi_x = x
        lo_y = hi_y = y
        while q:
            x, y = q.pop()
            pts.append((x, y))
            lo_x, hi_x = min(lo_x, x), max(hi_x, x)
            lo_y, hi_y = min(lo_y, y), max(hi_y, y)
            for dx, dy in zip(_DX, _DY):
                u, v = x + dx, y + dy
                if (0 <= u < w and 0 <= v < h and
                        allowed[v, u] and not seen[v, u]):
                    seen[v, u] = True
                    q.append((u, v))
        result.append((lo_x, lo_y, hi_x + 1, hi_y + 1, pts))
    return result


def _get_goals(a):
    goals = []
    for x, y, X, Y, pts in _components(a[:63], {4}):
        w, h = X - x, Y - y
        if w < 5 or h < 5:
            continue
        if (w - 2) % 3 or (h - 2) % 3:
            continue
        kind = "x" if w == 11 and h == 11 else "p"
        goals.append((kind, (w - 2) // 3, (h - 2) // 3, x + 1, y + 1))
    return goals


def _parse_pieces(a, rx, ry):
    items = []

    # Exact player templates, including the selection/contact indicators.
    occupied = set()
    candidates = []
    for w, h in ((6, 6), (3, 6), (6, 3), (3, 3)):
        if w == 3:
            cx0, cx1 = 1, 2
        else:
            cx0, cx1 = 2, 4
        if h == 3:
            cy0, cy1 = 1, 2
        else:
            cy0, cy1 = 2, 4
        center = np.zeros((h, w), dtype=bool)
        center[cy0:cy1, cx0:cx1] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        inner = ~(center | border)
        for y in range(ry, 64 - h, 3):
            for x in range(rx, 65 - w, 3):
                p = a[y:y + h, x:x + w]
                marker = int(p[cy0, cx0])
                if marker not in (0, 4, 5):
                    continue
                if not np.all(p[center] == marker):
                    continue
                if np.any(p[inner] != 14):
                    continue
                if marker == 0:
                    if np.any((p[border] != 14) & (p[border] != 0)):
                        continue
                elif np.any(p[border] != 14):
                    continue
                candidates.append((w * h, x, y, w, h, marker))

    candidates.sort(reverse=True)
    for area, x, y, w, h, marker in candidates:
        cells = {(u, v) for v in range(y, y + h)
                 for u in range(x, x + w)}
        if occupied & cells:
            continue
        occupied |= cells
        items.append((("p", w // 3, h // 3), x, y, marker, -1, 0))

    for x, y, X, Y, pts in _components(a[:63], {11}):
        w, h = X - x, Y - y
        if w >= 3 and h >= 3 and w % 3 == 0 and h % 3 == 0:
            items.append((("x", w // 3, h // 3), x, y, -1, -1, 0))

    # Bombs are solid rectangles; adjacent bombs can form one component.
    for x, y, X, Y, pts in _components(a[:63], {12, 13}):
        w, h = X - x, Y - y
        if w % 3 or h % 3:
            continue
        size = 6 if w >= 6 and h >= 6 else 3
        if w == h and w in (3, 6, 9):
            size = w
        for by in range(y, Y, size):
            for bx in range(x, X, size):
                if by + size > Y or bx + size > X:
                    continue
                p = a[by:by + size, bx:bx + size]
                if not np.all((p == 12) | (p == 13)):
                    continue
                red = int(np.count_nonzero(p == 12))
                charge = red // size
                direction = -1
                if 0 < charge < size:
                    if np.all(p[0, :] == 12):
                        direction = 1
                    elif np.all(p[-1, :] == 12):
                        direction = 0
                    elif np.all(p[:, -1] == 12):
                        direction = 2
                    elif np.all(p[:, 0] == 12):
                        direction = 3
                items.append((("b", size // 3, size // 3),
                              bx, by, -1, direction, charge))

    # Evaluation-only enemies use otherwise unused colors.
    enemy = []
    for x, y, X, Y, pts in _components(a[:63], {3, 6, 7, 8, 9, 10}):
        if len(pts) >= 3:
            enemy.append((x, y))
    return items, enemy


def _assignment(old, new):
    """Match visually identical objects to their predicted identities."""
    n = len(old)
    if n != len(new):
        return None
    if n <= 7:
        best = None
        best_cost = _INF
        for perm in itertools.permutations(range(n)):
            cost = sum(abs(old[i][0] - new[j][0]) +
                       abs(old[i][1] - new[j][1])
                       for i, j in enumerate(perm))
            if cost < best_cost:
                best_cost, best = cost, perm
                if cost == 0:
                    break
        return best
    unused = set(range(n))
    out = []
    for x, y in old:
        j = min(unused, key=lambda k: abs(x - new[k][0]) + abs(y - new[k][1]))
        unused.remove(j)
        out.append(j)
    return out


class _Board:
    def __init__(self, a):
        raw_goals = _get_goals(a)
        if raw_goals:
            self.rx = max(range(3), key=lambda r:
                          sum(g[3] % 3 == r for g in raw_goals))
            self.ry = max(range(3), key=lambda r:
                          sum(g[4] % 3 == r for g in raw_goals))
        else:
            ys, xs = np.nonzero(a == 14)
            self.rx = int(xs.min()) % 3 if len(xs) else 0
            self.ry = int(ys.min()) % 3 if len(ys) else 0

        self.W = (64 - self.rx) // 3
        self.H = (63 - self.ry) // 3
        self.N = self.W * self.H
        self.xy = [(i % self.W, i // self.W) for i in range(self.N)]
        self.next = []
        for x, y in self.xy:
            self.next.append(tuple(
                (y + dy) * self.W + x + dx
                if 0 <= x + dx < self.W and 0 <= y + dy < self.H else -1
                for dx, dy in zip(_DX, _DY)
            ))

        self.wall = 0
        self.ice = 0
        for p, (x, y) in enumerate(self.xy):
            xx, yy = self.rx + 3 * x, self.ry + 3 * y
            patch = a[yy:yy + 3, xx:xx + 3]
            if np.any(patch == 2):
                self.wall |= 1 << p
            if np.any(patch == 15):
                self.ice |= 1 << p

        items, enemies = _parse_pieces(a, self.rx, self.ry)
        items.sort(key=lambda v: ({"p": 0, "x": 1, "b": 2}[v[0][0]],
                                  v[2], v[1]))
        self.keys = [v[0] for v in items]
        self.players = [i for i, k in enumerate(self.keys) if k[0] == "p"]
        self.crosses = [i for i, k in enumerate(self.keys) if k[0] == "x"]
        self.bombs = [i for i, k in enumerate(self.keys) if k[0] == "b"]
        self.order = self.crosses + self.bombs + self.players
        self.M = len(items)

        self.goals = []
        for kind, w, h, x, y in raw_goals:
            p = self.encode(x, y)
            if p >= 0:
                self.goals.append(((kind, w, h), p))

        self.tables = {}
        for key in set(self.keys):
            self._make_table(key)
        self.tab = [self.tables[k] for k in self.keys]

        self.rots = {}
        for i in self.bombs:
            _, x, y, marker, rot, charge = items[i]
            if rot < 0:
                size = self.keys[i][1] * 3
                scores = []
                for d in range(4):
                    # A launcher usually points away from its backing wall.
                    behind = _OPP[d]
                    sx = x + (size if _DX[behind] > 0 else -1 if _DX[behind] < 0 else 0)
                    sy = y + (size if _DY[behind] > 0 else -1 if _DY[behind] < 0 else 0)
                    score = 0
                    if _DX[behind]:
                        if 0 <= sx < 64:
                            score = int(np.count_nonzero(
                                a[max(0, y):min(63, y + size), sx] == 2))
                    else:
                        if 0 <= sy < 63:
                            score = int(np.count_nonzero(
                                a[sy, max(0, x):min(64, x + size)] == 2))
                    scores.append(score)
                rot = max(range(4), key=lambda d: scores[d])
            self.rots[i] = rot

        self.period = 1
        for i in self.bombs:
            self.period = math.lcm(self.period, self.keys[i][2] * 3)
        self.period = min(self.period, 180)

        positions = tuple(self.encode(v[1], v[2]) for v in items)
        selected = next((i for i in self.players if items[i][3] == 0),
                        self.players[0] if self.players else 0)
        phase = self._phase(items)
        self.initial = (positions, selected, phase, 0,
                        tuple(y * 64 + x for x, y in enemies))
        self.graphs = {}
        self.dist_cache = {}
        self.reverse_dist = {}
        self.regions = {}
        self.contacts_cache = {}
        self.transport_cache = {}
        self.dynamic_graph_cache = {}
        self.dynamic_dist_cache = {}
        self.h_cache = {}
        self.goal_groups = defaultdict(list)
        for key, p in self.goals:
            self.goal_groups[key].append(p)
        self.ids_by_key = defaultdict(list)
        for i, key in enumerate(self.keys):
            self.ids_by_key[key].append(i)

    def encode(self, x, y):
        if (x - self.rx) % 3 or (y - self.ry) % 3:
            return -1
        u, v = (x - self.rx) // 3, (y - self.ry) // 3
        if 0 <= u < self.W and 0 <= v < self.H:
            return v * self.W + u
        return -1

    def pixel(self, p):
        x, y = self.xy[p]
        return self.rx + 3 * x, self.ry + 3 * y

    def _phase(self, items):
        if not self.bombs:
            return 0
        for t in range(self.period):
            if all(t % (self.keys[i][2] * 3) == items[i][5] %
                   (self.keys[i][2] * 3) for i in self.bombs):
                return t
        return items[self.bombs[0]][5] % self.period

    def _make_table(self, key):
        kind, w, h = key
        cells = []
        for y in range(h):
            for x in range(w):
                if kind != "x" or x == w // 2 or y == h // 2:
                    cells.append((x, y))
        masks, normal, push, icy = [], [], [], []
        for x, y in self.xy:
            if x + w > self.W or y + h > self.H:
                masks.append(0)
                normal.append(False)
                push.append(False)
                icy.append(False)
                continue
            m = 0
            for dx, dy in cells:
                m |= 1 << ((y + dy) * self.W + x + dx)
            masks.append(m)
            push.append(not bool(m & self.wall))
            normal.append(not bool(m & (self.wall | self.ice)))
            icy.append(bool(m & self.ice))
        self.tables[key] = (masks, normal, push, icy)

    def read(self, a, expected):
        items, enemies = _parse_pieces(a, self.rx, self.ry)
        grouped = defaultdict(list)
        for item in items:
            grouped[item[0]].append(item)

        result = [None] * self.M
        changed = False
        for key, ids in self.ids_by_key.items():
            news = grouped.get(key, [])
            olds = [self.pixel(expected[0][i]) for i in ids]
            perm = _assignment(olds, [(v[1], v[2]) for v in news])
            if perm is None:
                return expected, False
            for j, i in enumerate(ids):
                result[i] = news[perm[j]]

        pos = tuple(self.encode(v[1], v[2]) for v in result)
        if any(p < 0 for p in pos):
            return expected, False
        selected = next((i for i in self.players if result[i][3] == 0),
                        expected[1])
        for i in self.bombs:
            d = result[i][4]
            if d >= 0 and d != self.rots[i]:
                self.rots[i] = d
                changed = True
        if changed:
            self.transport_cache.clear()
            self.dynamic_graph_cache.clear()
            self.dynamic_dist_cache.clear()
            self.h_cache.clear()

        phase = self._phase(result)
        ep = expected[4]
        if len(enemies) == len(ep):
            old = [(p % 64, p // 64) for p in ep]
            perm = _assignment(old, enemies)
            ep = tuple(enemies[j][1] * 64 + enemies[j][0] for j in perm)
        directions = expected[3] if pos == expected[0] else 0
        return (pos, selected, phase, directions, ep), changed

    def rect(self, x, y, w, h):
        x0, x1 = max(0, x), min(self.W, x + w)
        y0, y1 = max(0, y), min(self.H, y + h)
        if x0 >= x1 or y0 >= y1:
            return 0
        row = ((1 << (x1 - x0)) - 1) << x0
        m = 0
        for yy in range(y0, y1):
            m |= row << (yy * self.W)
        return m

    def beam(self, key, p, d, radius):
        x, y = self.xy[p]
        _, w, h = key
        if d == 0:
            return self.rect(x, y - radius, w, radius)
        if d == 1:
            return self.rect(x, y + h, w, radius)
        if d == 2:
            return self.rect(x - radius, y, radius, h)
        return self.rect(x + w, y, radius, h)

    def free_kick(self, key, p, d):
        masks, normal, push, icy = self.tables[key]
        q = p
        for t in range(80):
            if t >= 5 and not icy[q]:
                break
            v = self.next[q][d]
            if v < 0 or not push[v]:
                break
            q = v
        return q

    def simulate(self, state, actor, d):
        oldpos, selected, phase, directions, enemies = state
        pos = list(oldpos)
        masks = [self.tab[i][0][p] for i, p in enumerate(pos)]
        n = self.M

        def move(i, direction, stack=0):
            if stack & (1 << i):
                return False
            p = pos[i]
            q = self.next[p][direction]
            if q < 0 or not self.tab[i][2][q]:
                return False
            oldmask = masks[i]
            pos[i], masks[i] = q, self.tab[i][0][q]
            colliding = [j for j in self.order
                         if j != i and masks[j] & masks[i]]
            for j in colliding:
                if not move(j, direction, stack | (1 << i)):
                    pos[i], masks[i] = p, oldmask
                    return False
            return True

        q = self.next[pos[actor]][d]
        launched = []
        if q >= 0 and self.tab[actor][1][q]:
            m = self.tab[actor][0][q]
            launched = [i for i in self.order
                        if i != actor and masks[i] & m]
            if not launched:
                pos[actor], masks[actor] = q, m

        if launched:
            for i in launched:
                directions = (directions & ~(7 << (3 * i))) | ((d + 1) << (3 * i))
            for tick in range(90):
                finished = True
                for i in launched:
                    if tick >= 5 and not self.tab[i][3][pos[i]]:
                        continue
                    if move(i, d):
                        finished = False
                if finished:
                    break
            else:
                return None

        # Enemies advance before a new explosion is resolved.
        if enemies and self.crosses:
            tx, ty = self.pixel(pos[self.crosses[0]])
            new_enemies = []
            for code in enemies:
                ex, ey = code % 64, code // 64
                for _ in range(6):
                    dx, dy = tx - ex, ty - ey
                    if abs(dx) > abs(dy):
                        ex += 1 if dx > 0 else -1
                    else:
                        ey += 1 if dy > 0 else -1
                    if ex == tx and ey == ty:
                        return None
                new_enemies.append(ey * 64 + ex)
            enemies = tuple(new_enemies)

        newphase = (phase + 1) % self.period
        exploding = [i for i in self.bombs
                     if (phase + 1) % (self.keys[i][2] * 3) == 0]
        if exploding:
            for radius in (1, 2, 3):
                for b in exploding:
                    bd = self.rots[b]
                    blast = self.beam(self.keys[b], pos[b], bd, radius)
                    for i in self.order:
                        if masks[i] & blast:
                            move(i, bd)
                            directions = ((directions & ~(7 << (3 * i))) |
                                          ((bd + 1) << (3 * i)))
            for _ in range(90):
                moved = False
                for i in self.order:
                    di = ((directions >> (3 * i)) & 7) - 1
                    if di >= 0 and self.tab[i][3][pos[i]]:
                        if move(i, di):
                            moved = True
                if not moved:
                    break
            else:
                return None

        return (tuple(pos), actor, newphase, directions, enemies)

    def graph(self, key):
        if key in self.graphs:
            return self.graphs[key]
        masks, normal, push, icy = self.tables[key]
        graph = [[] for _ in range(self.N)]
        player = key[0] == "p"
        for p in range(self.N):
            if not push[p]:
                continue
            edges = {}
            if player and normal[p]:
                for q in self.next[p]:
                    if q >= 0 and normal[q]:
                        edges[q] = 1
            for d in range(4):
                q = self.free_kick(key, p, d)
                if q != p:
                    cost = 4 if player else 5
                    edges[q] = min(edges.get(q, _INF), cost)
                # Partial impulses occur through chains and explosions.
                if not player and self.bombs:
                    q = p
                    for k in range(5):
                        q = self.next[q][d]
                        if q < 0 or not push[q]:
                            break
                        if not icy[q]:
                            edges[q] = min(edges.get(q, _INF), 8)
            graph[p] = list(edges.items())
        self.graphs[key] = graph
        return graph

    @staticmethod
    def dijkstra(graph, start):
        dist = [_INF] * len(graph)
        if start < 0:
            return dist
        dist[start] = 0
        heap = [(0, start)]
        while heap:
            cost, p = heapq.heappop(heap)
            if cost != dist[p]:
                continue
            for q, w in graph[p]:
                nc = cost + w
                if nc < dist[q]:
                    dist[q] = nc
                    heapq.heappush(heap, (nc, q))
        return dist

    def distances(self, key, p):
        ck = (key, p)
        if ck not in self.dist_cache:
            self.dist_cache[ck] = self.dijkstra(self.graph(key), p)
        return self.dist_cache[ck]

    def to_goal(self, key, goal):
        ck = (key, goal)
        if ck not in self.reverse_dist:
            graph = self.graph(key)
            rev = [[] for _ in range(self.N)]
            for p, edges in enumerate(graph):
                for q, cost in edges:
                    rev[q].append((p, cost))
            self.reverse_dist[ck] = self.dijkstra(rev, goal)
        return self.reverse_dist[ck]

    def region(self, key):
        if key in self.regions:
            return self.regions[key]
        normal = self.tables[key][1]
        labels = [-1] * self.N
        label = 0
        for p in range(self.N):
            if not normal[p] or labels[p] >= 0:
                continue
            labels[p] = label
            q = [p]
            for u in q:
                for v in self.next[u]:
                    if v >= 0 and normal[v] and labels[v] < 0:
                        labels[v] = label
                        q.append(v)
            label += 1
        self.regions[key] = labels
        return labels

    def contacts(self, akey, tkey, tp, d):
        ck = (akey, tkey, tp, d)
        if ck in self.contacts_cache:
            return self.contacts_cache[ck]
        am, normal, _, _ = self.tables[akey]
        tm = self.tables[tkey][0][tp]
        x, y = self.xy[tp]
        _, aw, ah = akey
        _, tw, th = tkey
        result = []
        for yy in range(max(0, y - ah - 1), min(self.H, y + th + 1)):
            for xx in range(max(0, x - aw - 1), min(self.W, x + tw + 1)):
                p = yy * self.W + xx
                q = self.next[p][d]
                if (normal[p] and q >= 0 and normal[q] and
                        not (am[p] & tm) and (am[q] & tm)):
                    result.append(p)
        self.contacts_cache[ck] = result
        return result

    def transport_routes(self, akey, bkey, bd):
        ck = (akey, bkey, bd)
        if ck in self.transport_cache:
            return self.transport_cache[ck]
        am, normal, push, icy = self.tables[akey]
        bm, _, bpush, _ = self.tables[bkey]
        regions = self.region(akey)
        routes = []
        aw, ah = akey[1:]
        bw, bh = bkey[1:]
        for bp in range(self.N):
            if not bpush[bp]:
                continue
            bx, by = self.xy[bp]
            finalbeam = self.beam(bkey, bp, bd, 3)
            if not finalbeam:
                continue
            for yy in range(max(0, by - 3 - ah), min(self.H, by + bh + 4)):
                for xx in range(max(0, bx - 3 - aw), min(self.W, bx + bw + 4)):
                    p = yy * self.W + xx
                    if (not normal[p] or am[p] & bm[bp] or
                            not am[p] & finalbeam):
                        continue
                    q = p
                    for radius in (1, 2, 3):
                        if am[q] & self.beam(bkey, bp, bd, radius):
                            v = self.next[q][bd]
                            if v >= 0 and push[v]:
                                q = v
                    for _ in range(40):
                        if not icy[q]:
                            break
                        v = self.next[q][bd]
                        if v < 0 or not push[v]:
                            break
                        q = v
                    if (normal[q] and q != p and
                            regions[p] != regions[q]):
                        routes.append((p, q, bp))
        self.transport_cache[ck] = routes
        return routes

    def navigation(self, state, actor):
        key = self.keys[actor]
        p = state[0][actor]
        if len(self.players) != 1 or not self.bombs:
            return self.distances(key, p)

        config = tuple(state[0][b] for b in self.bombs)
        gkey = (key, config)
        dkey = (gkey, p)
        if dkey in self.dynamic_dist_cache:
            return self.dynamic_dist_cache[dkey]

        if gkey not in self.dynamic_graph_cache:
            normal = self.tables[key][1]
            edges = [dict() for _ in range(self.N)]
            for u in range(self.N):
                if normal[u]:
                    for v in self.next[u]:
                        if v >= 0 and normal[v]:
                            edges[u][v] = 1
            for b in self.bombs:
                bkey = self.keys[b]
                distances = self.distances(bkey, state[0][b])
                for u, v, bp in self.transport_routes(key, bkey, self.rots[b]):
                    cost = distances[bp] + 3
                    if cost < edges[u].get(v, _INF):
                        edges[u][v] = cost
            graph = [list(e.items()) for e in edges]
            if len(self.dynamic_graph_cache) > 250:
                self.dynamic_graph_cache.clear()
                self.dynamic_dist_cache.clear()
            self.dynamic_graph_cache[gkey] = graph

        dist = self.dijkstra(self.dynamic_graph_cache[gkey], p)
        self.dynamic_dist_cache[dkey] = dist
        return dist

    def solved(self, state):
        pos = state[0]
        for key, targets in self.goal_groups.items():
            have = {pos[i] for i in self.ids_by_key.get(key, [])}
            if any(p not in have for p in targets):
                return False
        return bool(self.goals)

    def heuristic(self, state):
        # Direction histories do not affect the heuristic.
        ck = (state[0], state[1], state[2] if state[4] else 0)
        if ck in self.h_cache:
            return self.h_cache[ck]
        pos = state[0]
        score = 0.0

        for key, targets in self.goal_groups.items():
            ids = self.ids_by_key.get(key, [])
            if not ids:
                score += 100
                continue
            costs = []
            for i in ids:
                row = []
                for target in targets:
                    if pos[i] == target:
                        row.append(0.0)
                        continue
                    base = self.to_goal(key, target)[pos[i]]
                    if base >= _INF:
                        x, y = self.xy[pos[i]]
                        gx, gy = self.xy[target]
                        base = 30 + abs(x - gx) + abs(y - gy)

                    if key[0] == "p":
                        if len(self.players) == 1 and self.bombs:
                            nav = self.navigation(state, i)[target]
                            if nav < _INF:
                                base = nav
                            else:
                                base += 25
                        row.append(float(base))
                    else:
                        best = _INF
                        for d in range(4):
                            end = self.free_kick(key, pos[i], d)
                            if end == pos[i]:
                                continue
                            after = self.to_goal(key, target)[end]
                            if after >= _INF or after > base + 6:
                                continue
                            setup = _INF
                            for a in self.players:
                                nav = self.navigation(state, a)
                                contacts = self.contacts(
                                    self.keys[a], key, pos[i], d)
                                for c in contacts:
                                    value = nav[c]
                                    if value >= _INF:
                                        value = self.distances(
                                            self.keys[a], pos[a])[c] + 25
                                    if a != state[1]:
                                        value += 1
                                    setup = min(setup, value)
                            best = min(best, setup + 5 + after)
                        row.append(float(min(best, base + 45)))
                costs.append(row)

            if len(targets) <= len(ids) and len(ids) <= 7:
                best = _INF
                for perm in itertools.permutations(range(len(ids)), len(targets)):
                    best = min(best, sum(costs[perm[j]][j]
                                         for j in range(len(targets))))
                score += best
            else:
                score += sum(min(row[j] for row in costs)
                             for j in range(len(targets)))

        if state[4] and self.crosses:
            x, y = self.pixel(pos[self.crosses[0]])
            distance = min(abs(x - e % 64) + abs(y - e // 64)
                           for e in state[4])
            score += max(0, 18 - distance) * 0.5
        if len(self.h_cache) > 70000:
            self.h_cache.clear()
        self.h_cache[ck] = score
        return score

    def macros(self, state):
        pos, selected, phase, directions, enemies = state
        allmask = 0
        for i, p in enumerate(pos):
            allmask |= self.tab[i][0][p]
        output = []
        seen = set()

        def emit(actor, path):
            if not path:
                return
            path = tuple(path)
            k = (actor, path)
            if k not in seen:
                seen.add(k)
                output.append(k)

        cross_targets = [pos[i] for i in self.crosses
                         if not any(key == self.keys[i] and gp == pos[i]
                                    for key, gp in self.goals)]

        for actor in self.players:
            key = self.keys[actor]
            am, normal, _, _ = self.tab[actor]
            obstacles = allmask ^ am[pos[actor]]
            dist = [-1] * self.N
            prev = [-1] * self.N
            how = [-1] * self.N
            root = pos[actor]
            dist[root] = 0
            queue = [root]
            pushes = []
            for u in queue:
                for d, v in enumerate(self.next[u]):
                    if v < 0 or not normal[v]:
                        continue
                    if am[v] & obstacles:
                        pushes.append((u, d))
                    elif dist[v] < 0:
                        dist[v] = dist[u] + 1
                        prev[v], how[v] = u, d
                        queue.append(v)

            paths = {root: ()}

            def path_to(p):
                if p in paths:
                    return paths[p]
                out = []
                q = p
                while q != root and q >= 0:
                    out.append(how[q])
                    q = prev[q]
                out.reverse()
                paths[p] = tuple(out)
                return paths[p]

            for u, d in pushes:
                emit(actor, path_to(u) + (d,))

            for gkey, target in self.goals:
                if gkey == key and dist[target] >= 0:
                    emit(actor, path_to(target))

            # Staging squares from which a second player can kick this one
            # across an otherwise impassable strip.
            if len(self.players) > 1:
                labels = self.region(key)
                bins = defaultdict(list)
                target_fields = [self.to_goal(key, gp)
                                 for gkey, gp in self.goals if gkey == key]
                for u in queue:
                    if u == root:
                        continue
                    for d in range(4):
                        v = self.free_kick(key, u, d)
                        if (v == u or labels[v] < 0 or
                                labels[v] == labels[u]):
                            continue
                        if not any(self.contacts(self.keys[a], key, u, d)
                                   for a in self.players if a != actor):
                            continue
                        onward = min((field[v] for field in target_fields),
                                     default=20)
                        if cross_targets:
                            vx, vy = self.xy[v]
                            near = min(abs(vx - self.xy[c][0]) +
                                       abs(vy - self.xy[c][1])
                                       for c in cross_targets)
                            onward = min(onward, near + 2)
                        value = dist[u] + 0.9 * min(onward, 100)
                        bins[(labels[v], d)].append((value, u))
                for values in bins.values():
                    values.sort()
                    for _, u in values[:3]:
                        emit(actor, path_to(u))

            # Approach a launcher and remain in its beam until it fires.
            for b in self.bombs:
                bd = self.rots[b]
                beam = self.beam(self.keys[b], pos[b], bd, 3)
                candidates = [u for u in queue if am[u] & beam]
                candidates.sort(key=lambda u: dist[u])
                # Include all beam squares on small boards, but limit duplicates.
                for u in candidates[:18]:
                    path = path_to(u)
                    emit(actor, path)
                    period = self.keys[b][2] * 3
                    wait = period - ((phase + len(path)) % period)
                    stationary = None
                    swing = None
                    for d, v in enumerate(self.next[u]):
                        if v < 0 or not normal[v]:
                            stationary = d
                            break
                    if stationary is None:
                        preferences = ((2, 3, 0, 1) if bd in (0, 1)
                                       else (0, 1, 2, 3))
                        for d in preferences:
                            v = self.next[u][d]
                            if (v >= 0 and normal[v] and
                                    not am[v] & obstacles and am[v] & beam):
                                swing = d
                                break
                    if stationary is not None:
                        tail = (stationary,) * wait
                    elif swing is not None:
                        tail = tuple(swing if k % 2 == 0 else _OPP[swing]
                                     for k in range(wait))
                    else:
                        continue
                    emit(actor, path + tail)

            # Short moves also permit parking, clearing a route, and adjusting
            # the firing phase without committing to a long macro.
            if self.bombs or len(output) < 12:
                for d, v in enumerate(self.next[root]):
                    if v >= 0 and normal[v] and not am[v] & obstacles:
                        emit(actor, (d,))
            if self.bombs:
                for d, v in enumerate(self.next[root]):
                    if v < 0 or not normal[v]:
                        emit(actor, (d,))
                        break

        return output

    def apply_macro(self, state, actor, path):
        s = state
        for d in path:
            s = self.simulate(s, actor, d)
            if s is None:
                return None
        return s


def _unroll(nodes, index, board):
    chunks = []
    while nodes[index][1] >= 0:
        state, parent, macro, g = nodes[index]
        source = nodes[parent][0]
        actor, path = macro
        chunk = []
        if actor != source[1]:
            x, y = board.pixel(source[0][actor])
            chunk.append(("ACTION6", y + 1, x + 1, actor))
        chunk.extend(("ACTION" + str(d + 1), None, None, actor) for d in path)
        chunks.append(chunk)
        index = parent
    out = []
    for chunk in reversed(chunks):
        out.extend(chunk)
    return out


def _plan(board, start, deadline, visits):
    h0 = board.heuristic(start)
    nodes = [(start, -1, None, 0)]
    heap = [(1.65 * h0, 0, 0)]
    best_g = {start: 0}
    best_index = 0
    best_value = h0
    serial = 0
    expanded = 0

    while heap and time.perf_counter() < deadline:
        priority, _, index = heapq.heappop(heap)
        state, parent, macro, g = nodes[index]
        if g != best_g.get(state):
            continue
        if board.solved(state):
            return _unroll(nodes, index, board), True

        h = board.heuristic(state)
        value = h + 0.025 * g + 1.5 * visits.get((state[0], state[2]), 0)
        if index and (value < best_value or best_index == 0):
            best_value, best_index = value, index

        expanded += 1
        if expanded > 10000:
            break
        if g >= 135:
            continue

        macros = board.macros(state)
        # Small, goal-directed macros are considered first.
        macros.sort(key=lambda m: len(m[1]) + (m[0] != state[1]))
        for actor, path in macros:
            if time.perf_counter() >= deadline:
                break
            ng = g + len(path) + (actor != state[1])
            if ng > 140:
                continue
            ns = board.apply_macro(state, actor, path)
            if ns is None or ns == state:
                continue
            if ng >= best_g.get(ns, _INF):
                continue
            best_g[ns] = ng
            ni = len(nodes)
            nodes.append((ns, index, (actor, path), ng))
            nh = board.heuristic(ns)
            if board.solved(ns):
                return _unroll(nodes, ni, board), True
            nv = nh + 0.025 * ng + 1.5 * visits.get((ns[0], ns[2]), 0)
            if nv < best_value:
                best_value, best_index = nv, ni
            serial += 1
            penalty = 2.0 * visits.get((ns[0], ns[2]), 0)
            heapq.heappush(heap, (ng + 1.65 * nh + penalty, serial, ni))

    if best_index:
        return _unroll(nodes, best_index, board), False
    return [], False


def act(obs, available, memory):
    started = time.perf_counter()
    a = np.asarray(obs)
    allowed = set(available)

    if "_ka_board" not in memory:
        board = _Board(a)
        memory["_ka_board"] = board
        state = board.initial
        memory["_ka_expected"] = state
        memory["_ka_plan"] = []
        memory["_ka_visits"] = {}
        memory["_ka_steps"] = 0
    else:
        board = memory["_ka_board"]
        expected = memory["_ka_expected"]
        state, changed = board.read(a, expected)
        if changed or state != expected:
            memory["_ka_plan"] = []
        memory["_ka_expected"] = state

    if not board.players or any(p < 0 for p in state[0]):
        if "RESET" in allowed:
            return "RESET", None, None
        name = next((v for v in available if v != "ACTION6"), available[0])
        return name, None, None

    visits = memory["_ka_visits"]
    visit_key = (state[0], state[2])
    visits[visit_key] = visits.get(visit_key, 0) + 1
    memory["_ka_steps"] += 1

    plan = memory["_ka_plan"]
    if not plan:
        deadline = started + 8.7
        plan, complete = _plan(board, state, deadline, visits)
        if not complete:
            # Reconsider partial plans after a small amount of real progress.
            plan = plan[:14]
        memory["_ka_plan"] = plan

    while memory["_ka_plan"]:
        name, row, col, actor = memory["_ka_plan"].pop(0)
        if name not in allowed:
            memory["_ka_plan"] = []
            break
        if name == "ACTION6":
            ns = (state[0], actor, state[2], state[3], state[4])
            memory["_ka_expected"] = ns
            return name, int(row), int(col)
        d = int(name[6:]) - 1
        ns = board.simulate(state, actor, d)
        if ns is not None:
            memory["_ka_expected"] = ns
            return name, None, None
        memory["_ka_plan"] = []
        break

    # Safe one-step fallback, also useful if a search stopped at its time limit.
    choices = []
    for actor in board.players:
        for d in range(4):
            name = "ACTION" + str(d + 1)
            if name not in allowed:
                continue
            ns = board.simulate(state, actor, d)
            if ns is None:
                continue
            cost = board.heuristic(ns)
            cost += 3 * visits.get((ns[0], ns[2]), 0)
            cost += 1 if actor != state[1] else 0
            if ns[0] == state[0] and not board.bombs:
                cost += 10
            choices.append((cost, actor, d, ns))

    if choices:
        _, actor, d, ns = min(choices, key=lambda z: z[0])
        if actor != state[1] and "ACTION6" in allowed:
            x, y = board.pixel(state[0][actor])
            memory["_ka_expected"] = (state[0], actor, state[2],
                                      state[3], state[4])
            memory["_ka_plan"] = [
                ("ACTION" + str(d + 1), None, None, actor)
            ]
            return "ACTION6", int(y + 1), int(x + 1)
        memory["_ka_expected"] = ns
        return "ACTION" + str(d + 1), None, None

    if "RESET" in allowed:
        memory.clear()
        return "RESET", None, None
    name = next((v for v in available if v != "ACTION6"), available[0])
    if name == "ACTION6":
        x, y = board.pixel(state[0][state[1]])
        return name, int(y + 1), int(x + 1)
    return name, None, None

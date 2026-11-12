import numpy as np
import math
import time
import heapq
from collections import deque, defaultdict

_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_INF = 1000000


def _components(mask):
    h, w = mask.shape
    a = bytearray(np.asarray(mask, dtype=np.uint8).ravel())
    result = []
    for start in range(h * w):
        if not a[start]:
            continue
        a[start] = 0
        stack = [start]
        points = []
        xmin = xmax = start % w
        ymin = ymax = start // w
        while stack:
            z = stack.pop()
            y, x = divmod(z, w)
            points.append((x, y))
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
            for zz in (z - w, z + w):
                if 0 <= zz < h * w and a[zz]:
                    a[zz] = 0
                    stack.append(zz)
            if x and a[z - 1]:
                a[z - 1] = 0
                stack.append(z - 1)
            if x + 1 < w and a[z + 1]:
                a[z + 1] = 0
                stack.append(z + 1)
        result.append((xmin, ymin, xmax - xmin + 1,
                       ymax - ymin + 1, points))
    return result


def _scale_of(a, previous=None):
    if previous is not None:
        return previous
    scores = [0, 0, 0, 0]
    mask = a == 4
    mask[63] = False
    for x, y, w, h, pts in _components(mask):
        for s in (1, 2, 3):
            if ((w, h) in ((5*s, 5*s), (5*s, 8*s),
                          (8*s, 5*s), (8*s, 8*s), (11*s, 11*s))):
                scores[s] += len(pts)
    return max((1, 2, 3), key=lambda s: (scores[s], -s))


def _read_objects(a, s):
    """Objects are represented in display-pixel coordinates."""
    objects = []
    occupied = np.zeros((64, 64), dtype=bool)
    seeds = np.argwhere(np.isin(a[:63], (0, 4, 5)))
    candidates = set()
    for w0, h0 in ((3, 3), (3, 6), (6, 3), (6, 6)):
        w, h = w0*s, h0*s
        cx = (1 if w0 == 3 else 2)*s
        cy = (1 if h0 == 3 else 2)*s
        cw = (1 if w0 == 3 else 2)*s
        ch = (1 if h0 == 3 else 2)*s
        for yy, xx in seeds:
            # One seed at the upper-left pixel of a center is sufficient.
            x, y = int(xx)-cx, int(yy)-cy
            if x >= 0 and y >= 0 and x+w <= 64 and y+h <= 63:
                candidates.add((x, y, w0, h0))

    valid = []
    for x, y, w0, h0 in candidates:
        w, h = w0*s, h0*s
        r = a[y:y+h:s, x:x+w:s]
        if r.shape != (h0, w0):
            continue
        cx = 1 if w0 == 3 else 2
        cy = 1 if h0 == 3 else 2
        cw = 1 if w0 == 3 else 2
        ch = 1 if h0 == 3 else 2
        color = int(r[cy, cx])
        if color not in (0, 4, 5):
            continue
        if not np.all(r[cy:cy+ch, cx:cx+cw] == color):
            continue
        expected = np.full((h0, w0), 14, dtype=np.int8)
        expected[cy:cy+ch, cx:cx+cw] = color
        if color == 0:
            if np.all(r[0, 1:-1] == 0):
                expected[0, :] = 0
            if np.all(r[-1, 1:-1] == 0):
                expected[-1, :] = 0
            if np.all(r[1:-1, 0] == 0):
                expected[:, 0] = 0
            if np.all(r[1:-1, -1] == 0):
                expected[:, -1] = 0
        if not np.array_equal(r, expected):
            continue
        big = np.repeat(np.repeat(expected, s, axis=0), s, axis=1)
        if np.array_equal(a[y:y+h, x:x+w], big):
            valid.append((w*h, x, y, w, h, color == 0))

    for area, x, y, w, h, selected in sorted(valid, reverse=True):
        if occupied[y:y+h, x:x+w].any():
            continue
        occupied[y:y+h, x:x+w] = True
        objects.append(dict(k="P", x=x, y=y, w=w, h=h,
                            selected=selected, charge=0, direction=0))

    # Crosses: template matching also separates touching crosses.
    bmask = (a == 11)
    bmask[63] = False
    used = np.zeros_like(bmask)
    unit = 3*s
    cross_offsets = [(dx, dy) for dy in range(3) for dx in range(3)
                     if dx == 1 or dy == 1]
    cross_candidates = set()
    for yy, xx in np.argwhere(bmask):
        for dx, dy in ((1, 0), (0, 1)):
            x, y = int(xx)-dx*unit, int(yy)-dy*unit
            if 0 <= x <= 64-3*unit and 0 <= y <= 63-3*unit:
                cross_candidates.add((x, y))
    for x, y in sorted(cross_candidates, key=lambda z: (z[1], z[0])):
        good = True
        for dx, dy in cross_offsets:
            q = bmask[y+dy*unit:y+(dy+1)*unit,
                      x+dx*unit:x+(dx+1)*unit]
            u = used[y+dy*unit:y+(dy+1)*unit,
                     x+dx*unit:x+(dx+1)*unit]
            if not q.all() or u.any():
                good = False
                break
        if good:
            for dx, dy in cross_offsets:
                used[y+dy*unit:y+(dy+1)*unit,
                     x+dx*unit:x+(dx+1)*unit] = True
            objects.append(dict(k="C", x=x, y=y, w=3*unit, h=3*unit,
                                selected=False, charge=0, direction=0))

    # Bombs are solid 3x3 or 6x6 squares, with a charged strip.
    bombmask = np.isin(a, (12, 13))
    bombmask[63] = False
    remaining = bombmask.copy()
    for yy, xx in np.argwhere(bombmask):
        x, y = int(xx), int(yy)
        if not remaining[y, x]:
            continue
        size = 0
        for n in (6*s, 3*s):
            if x+n <= 64 and y+n <= 63 and remaining[y:y+n, x:x+n].all():
                size = n
                break
        if not size:
            continue
        r = a[y:y+size, x:x+size]
        remaining[y:y+size, x:x+size] = False
        count = int(np.count_nonzero(r == 12))
        charge = count // (size*s)
        direction = 0
        if count:
            if np.all(r[:s] == 12):
                direction = 2
            elif np.all(r[-s:] == 12):
                direction = 1
            elif np.all(r[:, :s] == 12):
                direction = 4
            elif np.all(r[:, -s:] == 12):
                direction = 3
        objects.append(dict(k="B", x=x, y=y, w=size, h=size,
                            selected=False, charge=charge,
                            direction=direction))

    # The enemy-only sprites use colors not used by the ordinary board.
    unknown = ~np.isin(a, (0, 1, 2, 4, 5, 11, 12, 13, 14, 15))
    unknown[63] = False
    enemies = []
    for x, y, w, h, pts in _components(unknown):
        if len(pts) >= 2:
            enemies.append((x, y, w, h))
    enemies.sort(key=lambda e: (e[1], e[0]))
    return objects, enemies


def _read_goals(a, objects, s, old):
    dynamic = np.zeros((64, 64), dtype=bool)
    yellow = (a == 4)
    yellow[63] = False
    for o in objects:
        x, y, w, h = o["x"], o["y"], o["w"], o["h"]
        if o["k"] == "C":
            u = 3*s
            dynamic[y:y+h, x+u:x+2*u] = True
            dynamic[y+u:y+2*u, x:x+w] = True
        else:
            dynamic[y:y+h, x:x+w] = True
        if o["k"] == "P":
            yellow[y:y+h, x:x+w] = False

    templates = []
    for w, h in ((5, 5), (5, 8), (8, 5), (8, 8)):
        pts = [(x, y) for y in range(h) for x in range(w)
               if x in (0, w-1) or y in (0, h-1)]
        templates.append(("P", w, h, pts,
                          ((0, 0), (w-1, 0), (0, h-1), (w-1, h-1))))
    rows = (
        "...44444...", "...4...4...", "...4...4...",
        "4444...4444", "4.........4", "4.........4",
        "4.........4", "4444...4444", "...4...4...",
        "...4...4...", "...44444..."
    )
    pts = [(x, y) for y, row in enumerate(rows)
           for x, c in enumerate(row) if c == "4"]
    anchors = ((3, 0), (7, 0), (0, 3), (10, 3),
               (0, 7), (10, 7), (3, 10), (7, 10))
    templates.append(("C", 11, 11, pts, anchors))

    yellow_points = [(int(x), int(y)) for y, x in np.argwhere(yellow)]
    choices = []
    for kind, w0, h0, pts0, anchors in templates:
        w, h = w0*s, h0*s
        # Sample all pixels for occlusion-aware matching.
        pts = [(x*s+dx, y*s+dy) for x, y in pts0
               for dy in range(s) for dx in range(s)]
        positions = set()
        for xx, yy in yellow_points:
            for ax, ay in anchors:
                x, y = xx-ax*s, yy-ay*s
                if x >= 0 and y >= 0 and x+w <= 64 and y+h <= 63:
                    positions.add((x, y))
        for x, y in positions:
            visible = []
            good = True
            for dx, dy in pts:
                xx, yy = x+dx, y+dy
                if yellow[yy, xx]:
                    visible.append((xx, yy))
                elif not dynamic[yy, xx]:
                    good = False
                    break
            if good and len(visible) >= max(6*s, len(pts)*0.40):
                goal = (kind, x+s, y+s, w-2*s, h-2*s)
                choices.append((len(visible), len(pts), goal, visible))
    found = list(old)
    covered = set()
    for n, total, goal, visible in sorted(choices, reverse=True):
        if goal in found:
            covered.update(visible)
            continue
        if sum(p in covered for p in visible) > n*0.35:
            continue
        found.append(goal)
        covered.update(visible)
    return found


class _Model:
    def __init__(self, a, objects, enemies, goals, terrain, scale, old=None):
        self.s = scale
        self.u = 3*scale
        players = [o for o in objects if o["k"] == "P"]
        self.rx = players[0]["x"] % self.u
        self.ry = players[0]["y"] % self.u
        self.W = (64-self.rx+self.u-1)//self.u
        self.H = (63-self.ry+self.u-1)//self.u
        self.N = self.W*self.H
        self.neighbor = []
        for p in range(self.N):
            x, y = p % self.W, p // self.W
            self.neighbor.append(tuple(
                (y+dy)*self.W+x+dx
                if 0 <= x+dx < self.W and 0 <= y+dy < self.H else -1
                for dx, dy in _DIRS
            ))

        order = {"C": 0, "B": 1, "P": 2}
        objects = sorted(objects, key=lambda o: (order[o["k"]], o["y"], o["x"]))
        self.objects = objects
        self.n = len(objects)
        self.players = [i for i, o in enumerate(objects) if o["k"] == "P"]
        self.crosses = [i for i, o in enumerate(objects) if o["k"] == "C"]
        self.bombs = [i for i, o in enumerate(objects) if o["k"] == "B"]
        self.kind = [o["k"] for o in objects]
        self.width = [o["w"]//self.u for o in objects]
        self.height = [o["h"]//self.u for o in objects]
        self.types = []
        self.type_defs = []
        for i, o in enumerate(objects):
            definition = (o["k"] == "C", self.width[i], self.height[i])
            if definition not in self.type_defs:
                self.type_defs.append(definition)
            self.types.append(self.type_defs.index(definition))

        walls = 0
        gulches = 0
        for p in range(self.N):
            x, y = self.pixel(p)
            if x+self.u > 64 or y+self.u > 63:
                walls |= 1 << p
                continue
            r = terrain[y:y+self.u, x:x+self.u]
            if np.any(r == 2):
                walls |= 1 << p
            if np.any(r == 15):
                gulches |= 1 << p
        self.wall_cells = walls
        self.gulch_cells = gulches
        self.occ, self.bad, self.gap, self.free = [], [], [], []
        self.offsets = []
        for cross, w, h in self.type_defs:
            offsets = [(dx, dy) for dy in range(h) for dx in range(w)
                       if not cross or dx == w//2 or dy == h//2]
            self.offsets.append(offsets)
            occ, bad, gap = [], [], []
            for p in range(self.N):
                x, y = p % self.W, p // self.W
                if x+w > self.W or y+h > self.H:
                    occ.append(0)
                    bad.append(True)
                    gap.append(False)
                    continue
                bits = 0
                for dx, dy in offsets:
                    bits |= 1 << (p+dy*self.W+dx)
                occ.append(bits)
                bad.append(bool(bits & walls))
                gap.append(bool(bits & gulches))
            self.occ.append(occ)
            self.bad.append(bad)
            self.gap.append(gap)
            self.free.append([not (bad[p] or gap[p]) for p in range(self.N)])

        self.bsize = {}
        self.bbase = {}
        self.bdir = {}
        self.unknown_direction = False
        for i in self.bombs:
            o = objects[i]
            size = o["h"]//scale
            direction = o["direction"]
            if not direction and old is not None:
                om, ost = old
                for j in om.bombs:
                    ox, oy = om.pixel(ost[0][j])
                    if ox == o["x"] and oy == o["y"] and om.objects[j]["w"] == o["w"]:
                        direction = om.bdir[j]
                        break
            if not direction:
                self.unknown_direction = True
                direction = 2
            self.bsize[i] = size
            self.bbase[i] = o["charge"] % size
            self.bdir[i] = direction
        self.period = 1
        for size in self.bsize.values():
            self.period = math.lcm(self.period, size)
        self.firing = []
        for phase in range(self.period):
            self.firing.append(tuple(i for i in self.bombs
                                     if (self.bbase[i]+phase+1) % self.bsize[i] == 0))

        positions = tuple(self.position(o["x"], o["y"]) for o in objects)
        active = next((i for i in self.players if objects[i]["selected"]),
                      self.players[0])
        velocities = [0]*self.n
        if old is not None:
            om, ost = old
            for i, o in enumerate(objects):
                for j, oo in enumerate(om.objects):
                    if (o["k"], o["w"], o["h"]) == (oo["k"], oo["w"], oo["h"]):
                        if (o["x"], o["y"]) == om.pixel(ost[0][j]):
                            velocities[i] = ost[3][j]
                            break
        self.enemy_sizes = [(e[2], e[3]) for e in enemies]
        self.initial = (positions, active, 0, tuple(velocities),
                        tuple((e[0], e[1]) for e in enemies))
        self.goals = []
        for kind, x, y, w, h in goals:
            if ((x-self.rx) % self.u or (y-self.ry) % self.u):
                continue
            p = self.position(x, y)
            eligible = tuple(i for i, o in enumerate(objects)
                             if o["k"] == kind and o["w"] == w and o["h"] == h)
            if eligible and 0 <= p < self.N:
                self.goals.append((p, eligible))
        self.player_goal_positions = defaultdict(set)
        for p, ids in self.goals:
            for i in ids:
                if self.kind[i] == "P":
                    self.player_goal_positions[i].add(p)
        self.contacts_cache = {}
        self.solo_cache = {}
        self.distances = []
        self.setup_cache = {}
        self.hcache = {}
        self.prepared = False

    def pixel(self, p):
        return self.rx+(p % self.W)*self.u, self.ry+(p//self.W)*self.u

    def position(self, x, y):
        return ((y-self.ry)//self.u)*self.W+(x-self.rx)//self.u

    def key(self, st):
        p, active, phase, vel, enemy = st
        # Old velocities only immediately matter for pieces inside gulches.
        v = tuple(vel[i] if self.gap[self.types[i]][p[i]] else 0
                  for i in range(self.n))
        return p, active, phase, v, enemy

    def won(self, st):
        p = st[0]
        return bool(self.goals) and all(any(p[i] == g for i in ids)
                                        for g, ids in self.goals)

    def _push(self, p, i, d, depth=0):
        if depth > self.n+2:
            return False
        t = self.types[i]
        q = self.neighbor[p[i]][d]
        if q < 0 or self.bad[t][q]:
            return False
        old = p[i]
        p[i] = q
        bits = self.occ[t][q]
        collisions = [j for j in range(self.n) if j != i and
                      bits & self.occ[self.types[j]][p[j]]]
        for j in collisions:
            if not self._push(p, j, d, depth+1):
                p[i] = old
                return False
        return True

    def motion(self, positions, active, d, velocities):
        p = list(positions)
        v = list(velocities)
        t = self.types[active]
        q = self.neighbor[p[active]][d]
        if q < 0 or not self.free[t][q]:
            return p, v
        bits = self.occ[t][q]
        roots = [j for j in range(self.n) if j != active and
                 bits & self.occ[self.types[j]][p[j]]]
        if not roots:
            p[active] = q
            return p, v
        for j in roots:
            v[j] = d+1
        for tick in range(80):
            finished = True
            for j in roots:
                tj = self.types[j]
                if tick >= 5 and not self.gap[tj][p[j]]:
                    continue
                if self._push(p, j, d):
                    finished = False
            if finished:
                break
        return p, v

    def _enemy_tick(self, enemy, p):
        if not enemy or not self.crosses:
            return enemy, True
        tx, ty = self.pixel(p[self.crosses[0]])
        out = []
        for ex, ey in enemy:
            for _ in range(6):
                dx, dy = tx-ex, ty-ey
                if abs(dx) > abs(dy):
                    ex += self.s if dx > 0 else -self.s
                else:
                    ey += self.s if dy > 0 else -self.s
                if ex == tx and ey == ty:
                    return (), False
            out.append((ex, ey))
        return tuple(out), True

    def _rect(self, x, y, w, h):
        bits = 0
        for yy in range(max(0, y), min(self.H, y+h)):
            xa, xb = max(0, x), min(self.W, x+w)
            if xb > xa:
                bits |= ((1 << (xb-xa))-1) << (yy*self.W+xa)
        return bits

    def tick(self, p, phase, v, enemy):
        enemy, alive = self._enemy_tick(enemy, p)
        if not alive:
            return None
        firing = self.firing[phase]
        if firing:
            for length in (1, 2, 3):
                for b in firing:
                    x, y = p[b] % self.W, p[b]//self.W
                    w, h = self.width[b], self.height[b]
                    d = self.bdir[b]-1
                    if d == 0:
                        beam = self._rect(x, y-length, w, length)
                    elif d == 1:
                        beam = self._rect(x, y+h, w, length)
                    elif d == 2:
                        beam = self._rect(x-length, y, length, h)
                    else:
                        beam = self._rect(x+w, y, length, h)
                    for j in range(self.n):
                        if beam & self.occ[self.types[j]][p[j]]:
                            self._push(p, j, d)
                            v[j] = d+1
            for _ in range(80):
                moved = False
                for j in range(self.n):
                    if v[j] and self.gap[self.types[j]][p[j]]:
                        if self._push(p, j, v[j]-1):
                            moved = True
                if not moved:
                    break
        return tuple(p), (phase+1) % self.period, tuple(v), enemy

    def step(self, st, action):
        p, active, phase, v, enemy = st
        if action > 4:
            return p, action-5, phase, v, enemy
        pp, vv = self.motion(p, active, action-1, v)
        z = self.tick(pp, phase, vv, enemy)
        if z is None:
            return None
        return z[0], active, z[1], z[2], z[3]

    def contacts(self, target_type, player_type, d):
        key = target_type, player_type, d
        if key in self.contacts_cache:
            return self.contacts_cache[key]
        target = set(self.offsets[target_type])
        own = self.offsets[player_type]
        _, tw, th = self.type_defs[target_type]
        _, pw, ph = self.type_defs[player_type]
        dx, dy = _DIRS[d]
        out = []
        for oy in range(-ph, th+1):
            for ox in range(-pw, tw+1):
                before = {(ox+x, oy+y) for x, y in own}
                after = {(ox+x+dx, oy+y+dy) for x, y in own}
                if not before & target and after & target:
                    out.append((ox, oy))
        self.contacts_cache[key] = out
        return out

    def contact_positions(self, p, target_type, player_type, d):
        x, y = p % self.W, p//self.W
        out = []
        for ox, oy in self.contacts(target_type, player_type, d):
            xx, yy = x+ox, y+oy
            if not (0 <= xx < self.W and 0 <= yy < self.H):
                continue
            q = yy*self.W+xx
            r = self.neighbor[q][d]
            if r >= 0 and self.free[player_type][q] and self.free[player_type][r]:
                out.append(q)
        return out

    def solo(self, t, p, d, count=5):
        key = t, p, d, count
        if key in self.solo_cache:
            return self.solo_cache[key]
        q = p
        for tick in range(80):
            if tick >= count and not self.gap[t][q]:
                break
            r = self.neighbor[q][d]
            if r < 0 or self.bad[t][r]:
                break
            q = r
        self.solo_cache[key] = q
        return q

    def prepare(self):
        if self.prepared:
            return
        self.prepared = True
        graphs = {}
        player_types = set(self.types[i] for i in self.players)
        for goal, ids in self.goals:
            i = ids[0]
            t = self.types[i]
            is_player = self.kind[i] == "P"
            can_kick = not is_player or len(self.players) > 1
            gkey = t, is_player, can_kick
            if gkey not in graphs:
                reverse = [[] for _ in range(self.N)]
                for p in range(self.N):
                    if self.bad[t][p]:
                        continue
                    for d in range(4):
                        r = self.neighbor[p][d]
                        if is_player and r >= 0 and self.free[t][r]:
                            reverse[r].append((p, 1.0))
                        if can_kick:
                            possible = any(self.contact_positions(p, t, pt, d)
                                           for pt in player_types)
                            if possible:
                                r = self.solo(t, p, d, 5)
                                if r != p:
                                    reverse[r].append((p, 2.0))
                        if self.bombs:
                            r = self.solo(t, p, d, 1)
                            if r != p:
                                reverse[r].append((p, 2.5))
                graphs[gkey] = reverse
            reverse = graphs[gkey]
            dist = [float(_INF)]*self.N
            dist[goal] = 0.0
            heap = [(0.0, goal)]
            while heap:
                value, p = heapq.heappop(heap)
                if value != dist[p]:
                    continue
                for q, cost in reverse[p]:
                    nv = value+cost
                    if nv < dist[q]:
                        dist[q] = nv
                        heapq.heappush(heap, (nv, q))
            self.distances.append(dist)

    def setup_distance(self, cross, p, goal_index, player_type):
        key = cross, p, goal_index, player_type
        if key in self.setup_cache:
            return self.setup_cache[key]
        ct = self.types[cross]
        field = self.distances[goal_index]
        directions = [d for d in range(4)
                      if field[self.solo(ct, p, d, 5)] < field[p]]
        if not directions:
            directions = range(4)
        seeds = []
        for d in directions:
            seeds.extend(self.contact_positions(p, ct, player_type, d))
        distance = [_INF]*self.N
        queue = deque()
        crossbits = self.occ[ct][p]
        for q in seeds:
            if distance[q]:
                distance[q] = 0
                queue.append(q)
        while queue:
            q = queue.popleft()
            nd = distance[q]+1
            for r in self.neighbor[q]:
                if r < 0 or distance[r] <= nd:
                    continue
                if not self.free[player_type][r]:
                    continue
                if self.occ[player_type][r] & crossbits:
                    continue
                distance[r] = nd
                queue.append(r)
        self.setup_cache[key] = distance
        return distance

    def heuristic(self, st):
        p, active, phase, vel, enemy = st
        if p in self.hcache:
            value = self.hcache[p]
        else:
            groups = defaultdict(list)
            for gi, (g, ids) in enumerate(self.goals):
                groups[ids].append(gi)
            value = 0.0
            for ids, gis in groups.items():
                # Small assignment problem: match equal-sized pieces to goals.
                dp = {0: 0.0}
                for gi in gis:
                    nxt = {}
                    field = self.distances[gi]
                    for mask, cost in dp.items():
                        for bit, i in enumerate(ids):
                            if mask & (1 << bit):
                                continue
                            nc = cost+min(80.0, field[p[i]])
                            nm = mask | (1 << bit)
                            if nc < nxt.get(nm, float("inf")):
                                nxt[nm] = nc
                    dp = nxt
                value += min(dp.values()) if dp else 100.0
            for c in self.crosses:
                matches = [gi for gi, (g, ids) in enumerate(self.goals)
                           if c in ids]
                if not matches or any(p[c] == self.goals[gi][0] for gi in matches):
                    continue
                gi = min(matches, key=lambda g: self.distances[g][p[c]])
                near = _INF
                for i in self.players:
                    dist = self.setup_distance(c, p[c], gi, self.types[i])
                    near = min(near, dist[p[i]])
                if near >= _INF:
                    x, y = p[c] % self.W, p[c]//self.W
                    near = 12+min(abs(p[i] % self.W-x)+abs(p[i]//self.W-y)
                                  for i in self.players)
                value += 0.75*near
            self.hcache[p] = value
        if enemy and self.crosses:
            x, y = self.pixel(p[self.crosses[0]])
            distance = min(abs(ex-x)+abs(ey-y) for ex, ey in enemy)/self.s
            value += max(0.0, 15.0-distance)*0.45
        return value

    def macros(self, st, deadline):
        """Shortest walks ending at a goal, a kick, or a moving explosion."""
        base, active, initial_phase, initial_v, initial_enemy = st
        outcomes = {}
        for actor in self.players:
            if time.monotonic() > deadline:
                break
            t = self.types[actor]
            blockers = 0
            for j in range(self.n):
                if j != actor:
                    blockers |= self.occ[self.types[j]][base[j]]
            selection = [] if actor == active else [5+actor]
            # record: position, phase, parent, action, depth, velocity tuple
            records = [(base[actor], initial_phase, -1, 0, 0, initial_v)]
            seen = {(base[actor], initial_phase)}
            head = 0
            enemy_at = [initial_enemy]
            enemy_limit = [False]

            def enemies_after(depth):
                while len(enemy_at) <= depth and not enemy_limit[0]:
                    e, alive = self._enemy_tick(enemy_at[-1], base)
                    if not alive:
                        enemy_limit[0] = True
                        break
                    enemy_at.append(e)
                return enemy_at[depth] if depth < len(enemy_at) else None

            def path_to(index, last=0):
                path = [last] if last else []
                while records[index][2] >= 0:
                    path.append(records[index][3])
                    index = records[index][2]
                path.reverse()
                return selection+path

            def emit(state, index, last=0):
                if state is None:
                    return
                key = self.key(state)
                cost = records[index][4]+bool(last)+len(selection)
                if cost == 0:
                    return
                previous = outcomes.get(key)
                if previous is None or cost < previous[0]:
                    outcomes[key] = (cost, state, path_to(index, last))

            while head < len(records):
                if head % 128 == 0 and time.monotonic() > deadline:
                    break
                index = head
                p, phase, parent, action, depth, velocity = records[head]
                head += 1
                enemy = enemies_after(depth)
                if enemy is None:
                    continue
                if p in self.player_goal_positions[actor] and (
                        p != base[actor] or phase != initial_phase):
                    pp = list(base)
                    pp[actor] = p
                    emit((tuple(pp), actor, phase, velocity, enemy), index)
                if depth >= 55:
                    continue
                next_phase = (phase+1) % self.period
                used_wait = False
                for d in range(4):
                    q = self.neighbor[p][d]
                    if q < 0 or not self.free[t][q]:
                        q = p
                        if self.period == 1 or used_wait:
                            continue
                        used_wait = True
                    hit = (q != p and bool(self.occ[t][q] & blockers))
                    if hit:
                        pp = list(base)
                        pp[actor] = p
                        moved, vv = self.motion(tuple(pp), actor, d, velocity)
                        if moved == pp and not self.firing[phase]:
                            continue
                        z = self.tick(moved, phase, vv, enemy)
                        if z is not None:
                            ns = (z[0], actor, z[1], z[2], z[3])
                            if ns[0] != base or ns[2] != initial_phase:
                                emit(ns, index, d+1)
                        continue

                    visit_key = q, next_phase
                    if visit_key in seen:
                        continue
                    seen.add(visit_key)
                    pp = list(base)
                    pp[actor] = q
                    if self.firing[phase]:
                        z = self.tick(pp[:], phase, list(velocity), enemy)
                        if z is None:
                            continue
                        ns = (z[0], actor, z[1], z[2], z[3])
                        if z[0] != tuple(pp):
                            emit(ns, index, d+1)
                            continue
                        vv = z[2]
                    else:
                        if enemies_after(depth+1) is None:
                            continue
                        vv = velocity
                    records.append((q, next_phase, index, d+1, depth+1, vv))

        return [(state, path) for cost, state, path in outcomes.values()]


def _observed_signature(objects, enemies):
    out = []
    for o in objects:
        out.append((o["k"], o["x"], o["y"], o["w"], o["h"],
                    bool(o["selected"]), o["charge"],
                    o["direction"] if o["charge"] else 0))
    return tuple(sorted(out)), tuple(sorted(enemies))


def _predicted_signature(m, st):
    p, active, phase, vel, enemy = st
    out = []
    for i, o in enumerate(m.objects):
        x, y = m.pixel(p[i])
        charge = 0
        direction = 0
        if i in m.bsize:
            charge = (m.bbase[i]+phase) % m.bsize[i]
            if charge:
                direction = m.bdir[i]
        out.append((o["k"], x, y, o["w"], o["h"],
                    i == active, charge, direction))
    en = tuple(sorted((xy[0], xy[1], wh[0], wh[1])
                      for xy, wh in zip(enemy, m.enemy_sizes)))
    return tuple(sorted(out)), en


def _search(m, start, deadline, history):
    m.prepare()
    if m.won(start):
        return [5+start[1]], True
    h0 = m.heuristic(start)
    nodes = [(start, 0, h0, -1, [])]
    heap = [(3.2*h0, h0, 0)]
    best_g = {m.key(start): 0}
    best_index = 0
    best_value = h0
    fallback_index = None
    fallback_value = float("inf")
    solution = None
    expansions = 0
    while heap and time.monotonic() < deadline:
        priority, _, ni = heapq.heappop(heap)
        st, g, h, parent, macro = nodes[ni]
        if best_g.get(m.key(st)) != g:
            continue
        if m.won(st):
            solution = ni
            break
        expansions += 1
        options = m.macros(st, deadline)
        ranked = []
        for ns, path in options:
            ng = g+len(path)
            if ng > 160:
                continue
            key = m.key(ns)
            if ng >= best_g.get(key, _INF):
                continue
            nh = m.heuristic(ns)
            repeat = history.get((ns[0], ns[1], ns[2]), 0)
            score = ng+3.2*nh+5.0*repeat
            ranked.append((score, nh, ng, key, ns, path))
        ranked.sort(key=lambda z: (z[0], z[1]))
        limit = 110 if m.bombs else 75
        for score, nh, ng, key, ns, path in ranked[:limit]:
            if ng >= best_g.get(key, _INF):
                continue
            best_g[key] = ng
            index = len(nodes)
            nodes.append((ns, ng, nh, ni, path))
            heapq.heappush(heap, (score, nh, index))
            repeat = history.get((ns[0], ns[1], ns[2]), 0)
            progress = nh+0.075*ng+3.0*repeat
            if progress < best_value:
                best_value, best_index = progress, index
            if progress < fallback_value:
                fallback_value, fallback_index = progress, index
            if m.won(ns):
                solution = index
                break
        if solution is not None or len(nodes) > 35000:
            break
    chosen = solution if solution is not None else best_index
    if chosen == 0 and fallback_index is not None:
        chosen = fallback_index
    path = []
    pieces = []
    while chosen > 0:
        node = nodes[chosen]
        pieces.append(node[4])
        chosen = node[3]
    for piece in reversed(pieces):
        path.extend(piece)
    return path, solution is not None


def _fallback(m, st, history):
    m.prepare()
    choices = []
    for action in range(1, 5):
        ns = m.step(st, action)
        if ns is None:
            continue
        moved = ns[0] != st[0]
        repeat = history.get((ns[0], ns[1], ns[2]), 0)
        score = m.heuristic(ns)+3*repeat+(0 if moved else 2)
        choices.append((score, action))
    if choices:
        return min(choices)[1]
    return 5+st[1]


def _act_impl(obs, available, memory, deadline):
    a = np.asarray(obs, dtype=np.int8)
    old_model = memory.get("_model")
    old_state = memory.get("_state")
    scale = _scale_of(a, old_model.s if old_model is not None else None)
    objects, enemies = _read_objects(a, scale)
    if not any(o["k"] == "P" for o in objects):
        name = "RESET" if "RESET" in available else available[0]
        return name, None, None

    valid_prediction = (
        old_model is not None and old_state is not None and
        _observed_signature(objects, enemies) ==
        _predicted_signature(old_model, old_state)
    )
    if valid_prediction:
        m, st = old_model, old_state
    else:
        terrain = memory.get("_terrain")
        if terrain is None:
            terrain = np.ones((64, 64), dtype=np.int8)
        else:
            terrain = terrain.copy()
        terrain[a == 2] = 2
        terrain[a == 15] = 15
        # Newly visible empty ground corrects old occlusion assumptions.
        terrain[a == 1] = 1
        terrain[63] = 2
        goals = _read_goals(a, objects, scale, memory.get("_goals", []))
        old = (old_model, old_state) if old_model is not None and old_state is not None else None
        m = _Model(a, objects, enemies, goals, terrain, scale, old)
        st = m.initial
        memory["_model"] = m
        memory["_terrain"] = terrain
        memory["_goals"] = goals
        memory["_plan"] = []

    history = memory.setdefault("_history", {})
    hk = (st[0], st[1], st[2])
    history[hk] = history.get(hk, 0)+1
    plan = memory.get("_plan", [])

    if not plan:
        if m.unknown_direction:
            # One directional action exposes each bomb's orientation.
            m.prepare()
            action = _fallback(m, st, history)
            plan = [action]
        else:
            plan, solved = _search(m, st, deadline, history)
            if not plan:
                plan = [_fallback(m, st, history)]
        memory["_plan"] = plan

    action = plan.pop(0)
    if action > 4:
        actor = action-5
        x, y = m.pixel(st[0][actor])
        name = "ACTION6"
        row = int(y + min(m.objects[actor]["h"]-1, m.s))
        col = int(x + min(m.objects[actor]["w"]-1, m.s))
    else:
        name, row, col = "ACTION"+str(action), None, None

    if name not in available:
        memory["_plan"] = []
        legal = [x for x in available if x.startswith("ACTION") and x != "ACTION6"]
        if legal:
            name, row, col = legal[0], None, None
            action = int(name[6:])
        elif "ACTION6" in available:
            action = 5+st[1]
            name = "ACTION6"
            x, y = m.pixel(st[0][st[1]])
            row, col = y+m.s, x+m.s
        else:
            return available[0], None, None

    predicted = m.step(st, action)
    memory["_state"] = predicted
    memory["_model"] = m
    return name, row, col


def act(obs, available, memory):
    deadline = time.monotonic()+8.6
    try:
        return _act_impl(obs, available, memory, deadline)
    except Exception:
        # Keep the interface valid even for an unusually occluded frame.
        memory["_plan"] = []
        memory["_state"] = None
        a = np.asarray(obs)
        zeros = np.argwhere(a[:63] == 0)
        if len(zeros):
            y, x = map(int, zeros[len(zeros)//2])
            for name, dx, dy in (
                ("ACTION1", 0, -3), ("ACTION4", 3, 0),
                ("ACTION2", 0, 3), ("ACTION3", -3, 0)
            ):
                yy, xx = y+dy, x+dx
                if name in available and 0 <= yy < 63 and 0 <= xx < 64:
                    if int(a[yy, xx]) not in (2, 15):
                        return name, None, None
        for name in ("ACTION1", "ACTION4", "ACTION2", "ACTION3", "RESET"):
            if name in available:
                return name, None, None
        if "ACTION6" in available:
            return "ACTION6", 32, 32
        return available[0], None, None

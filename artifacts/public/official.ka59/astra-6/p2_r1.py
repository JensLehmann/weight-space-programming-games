import math
import time
import heapq
import itertools
from collections import deque

import numpy as np


INF = 3000
DX = (0, 0, -1, 1)
DY = (-1, 1, 0, 0)
OPP = (1, 0, 3, 2)


def components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, bool)
    result = []
    H, W = mask.shape
    for y, x in zip(*np.where(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(int(x), int(y))]
        cells = []
        while q:
            xx, yy = q.pop()
            cells.append((xx, yy))
            for nx, ny in ((xx-1, yy), (xx+1, yy),
                           (xx, yy-1), (xx, yy+1)):
                if (0 <= nx < W and 0 <= ny < H and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((nx, ny))
        result.append(cells)
    return result


def bbox(cells):
    xs, ys = zip(*cells)
    x, y = min(xs), min(ys)
    return x, y, max(xs)-x+1, max(ys)-y+1


def cross_pixels():
    a = np.zeros((9, 9), bool)
    a[:, 3:6] = True
    a[3:6, :] = True
    return a


CROSS = cross_pixels()
CROSS_GOAL_INNER = np.pad(CROSS, 1)
CROSS_GOAL = np.zeros((11, 11), bool)
for _dy in range(3):
    for _dx in range(3):
        CROSS_GOAL[_dy:_dy+9, _dx:_dx+9] |= CROSS
CROSS_GOAL &= ~CROSS_GOAL_INNER


def perceive(obs):
    raw = np.asarray(obs, dtype=np.int16).copy()
    raw[63, :] = 2
    comps = components(raw == 4)
    large = [bbox(c) for c in comps if len(c) >= 8]
    scale, sx, sy = 1, 0, 0
    best = -1
    for s in range(1, 5):
        good = [b for b in large
                if b[2] % s == 0 and b[3] % s == 0
                and (b[2]//s, b[3]//s) in
                ((5, 5), (5, 8), (8, 5), (8, 8), (11, 11))]
        score = len(good)
        if score > best and good:
            best = score
            scale = s
            sx, sy = good[0][0] % s, good[0][1] % s
    small = raw[sy::scale, sx::scale]
    a = np.full((64, 64), 2, np.int16)
    a[:small.shape[0], :small.shape[1]] = small

    objects = []
    used = np.zeros((64, 64), bool)
    candidates = []
    cy, cx = np.where((a == 0) | (a == 4) | (a == 5))
    for h, w in ((6, 6), (6, 3), (3, 6), (3, 3)):
        iy = 1 if h == 3 else 2
        ix = 1 if w == 3 else 2
        ch = 1 if h == 3 else 2
        cw = 1 if w == 3 else 2
        for yy, xx in zip(cy, cx):
            x, y = int(xx)-ix, int(yy)-iy
            if x < 0 or y < 0 or x+w > 64 or y+h > 63:
                continue
            b = a[y:y+h, x:x+w]
            color = int(b[iy, ix])
            if not np.all(b[iy:iy+ch, ix:ix+cw] == color):
                continue
            e = np.full((h, w), 14, np.int16)
            e[iy:iy+ch, ix:ix+cw] = color
            if color == 0:
                if np.all(b[0, 1:-1] == 0):
                    e[0, :] = 0
                if np.all(b[-1, 1:-1] == 0):
                    e[-1, :] = 0
                if np.all(b[1:-1, 0] == 0):
                    e[:, 0] = 0
                if np.all(b[1:-1, -1] == 0):
                    e[:, -1] = 0
            if np.array_equal(b, e):
                candidates.append((w*h, x, y, w, h, color == 0))
    candidates.sort(reverse=True)
    actors = []
    for _, x, y, w, h, selected in candidates:
        if used[y:y+h, x:x+w].any():
            continue
        used[y:y+h, x:x+w] = True
        actors.append(dict(k="a", x=x, y=y, w=w, h=h,
                           rot=0, charge=0, selected=selected, known=True))

    crosses = []
    for y in range(55):
        for x in range(55):
            if a[y, x+3] != 11:
                continue
            b = a[y:y+9, x:x+9]
            if np.all(b[CROSS] == 11) and not used[y:y+9, x:x+9][CROSS].any():
                used[y:y+9, x:x+9] |= CROSS
                crosses.append(dict(k="c", x=x, y=y, w=9, h=9,
                                    rot=0, charge=0, selected=False, known=True))

    bm = (a == 13) | (a == 12)
    bombs = []
    for size in (6, 3):
        for y in range(64-size):
            for x in range(65-size):
                if not bm[y, x] or not bm[y:y+size, x:x+size].all():
                    continue
                b = a[y:y+size, x:x+size]
                n12 = int(np.count_nonzero(b == 12))
                charge = min(size-1, n12 // size)
                known = 0 < n12 < size*size
                if known:
                    if np.all(b[0, :] == 12):
                        rot = 0
                    elif np.all(b[-1, :] == 12):
                        rot = 180
                    elif np.all(b[:, -1] == 12):
                        rot = 90
                    else:
                        rot = 270
                else:
                    def wall_count(xx, yy, ww, hh):
                        if xx < 0 or yy < 0 or xx+ww > 64 or yy+hh > 64:
                            return ww*hh
                        return int(np.count_nonzero(a[yy:yy+hh, xx:xx+ww] == 2))
                    backs = [
                        wall_count(x, y-1, size, 1),
                        wall_count(x, y+size, size, 1),
                        wall_count(x-1, y, 1, size),
                        wall_count(x+size, y, 1, size),
                    ]
                    rot = (0, 180, 270, 90)[int(np.argmax(backs))]
                bombs.append(dict(k="b", x=x, y=y, w=size, h=size,
                                  rot=rot, charge=charge, selected=False,
                                  known=known))
                bm[y:y+size, x:x+size] = False
                used[y:y+size, x:x+size] = True

    actors.sort(key=lambda o: (o["w"]*o["h"], o["w"], o["y"], o["x"]))
    crosses.sort(key=lambda o: (o["y"], o["x"]))
    bombs.sort(key=lambda o: (o["w"], o["y"], o["x"]))
    objects = crosses + bombs + actors

    goals = []
    if actors:
        rx, ry = actors[0]["x"] % 3, actors[0]["y"] % 3
        gc = []
        templates = []
        for h, w in ((5, 5), (8, 5), (5, 8), (8, 8)):
            t = np.zeros((h, w), bool)
            t[0, :] = t[-1, :] = True
            t[:, 0] = t[:, -1] = True
            templates.append((t, False))
        templates.append((CROSS_GOAL, True))
        for t, isc in templates:
            h, w = t.shape
            for y in range((ry-1) % 3, 65-h, 3):
                for x in range((rx-1) % 3, 65-w, 3):
                    b = a[y:y+h, x:x+w]
                    u = used[y:y+h, x:x+w]
                    visible = (b == 4) & t & ~u
                    count = int(visible.sum())
                    if count < 7:
                        continue
                    if not np.all((b == 4)[t] | u[t]):
                        continue
                    if np.any((b == 4) & ~t & ~u):
                        continue
                    gc.append((count, x, y, w, h, isc, visible.copy()))
        gc.sort(reverse=True, key=lambda z: z[0])
        gu = np.zeros((64, 64), bool)
        for count, x, y, w, h, isc, vis in gc:
            if np.any(gu[y:y+h, x:x+w] & vis):
                continue
            gu[y:y+h, x:x+w] |= vis
            goals.append(((w-2, h-2, isc), x+1, y+1))
        goals.sort(key=lambda g: (not g[0][2], g[2], g[1], g[0]))

    unknown = ~np.isin(a, (0, 1, 2, 4, 5, 11, 12, 13, 14, 15))
    enemies = [bbox(c)[:2] for c in components(unknown) if len(c) >= 3]

    terrain = np.ones((64, 64), np.int8)
    terrain[a == 2] = 2
    terrain[a == 15] = 15
    # Recover a gulch hidden behind a piece when both ends are visible.
    for _ in range(2):
        for transpose in (False, True):
            tt = terrain.T if transpose else terrain
            uu = used.T if transpose else used
            for y in range(64):
                x = 0
                while x < 64:
                    if not uu[y, x]:
                        x += 1
                        continue
                    z = x
                    while z < 64 and uu[y, z]:
                        z += 1
                    if x > 0 and z < 64 and tt[y, x-1] == 15 and tt[y, z] == 15:
                        tt[y, x:z] = 15
                    x = z
    return dict(a=a, objects=objects, goals=goals, enemies=enemies,
                terrain=terrain, scale=scale, sx=sx, sy=sy)


def closure(m):
    for k in range(len(m)):
        np.minimum(m, m[:, k:k+1] + m[k:k+1, :], out=m)
    return m


class World:
    def __init__(self, data):
        self.meta = [dict(o) for o in data["objects"]]
        self.scale, self.sx, self.sy = data["scale"], data["sx"], data["sy"]
        self.terrain = data["terrain"].copy()
        self.n = len(self.meta)
        self.actors = [i for i, o in enumerate(self.meta) if o["k"] == "a"]
        self.crosses = [i for i, o in enumerate(self.meta) if o["k"] == "c"]
        self.bombs = [i for i, o in enumerate(self.meta) if o["k"] == "b"]
        self.ne = len(data["enemies"])
        first = self.meta[self.actors[0]]
        self.rx, self.ry = first["x"] % 3, first["y"] % 3
        self.nx = (63-self.rx)//3 + 1
        self.ny = (63-self.ry)//3 + 1
        self.N = self.nx*self.ny
        self.types = []
        self.ot = []
        for o in self.meta:
            t = (o["w"]//3, o["h"]//3, o["k"] == "c")
            if t not in self.types:
                self.types.append(t)
            self.ot.append(self.types.index(t))
        self.atypes = sorted(set(self.ot[i] for i in self.actors))
        self.goals = []
        for (w, h, isc), x, y in data["goals"]:
            t = (w//3, h//3, isc)
            if t in self.types:
                p = self.pos(x, y)
                if p is not None:
                    self.goals.append((self.types.index(t), p, isc))
        self.cycle = 1
        for i in self.bombs:
            self.cycle = math.lcm(self.cycle, self.meta[i]["h"])
        self.build()
        self.initial = self.state_from(data, None)

    def pos(self, x, y):
        xx, yy = x-self.rx, y-self.ry
        if xx % 3 or yy % 3:
            return None
        xx, yy = xx//3, yy//3
        if 0 <= xx < self.nx and 0 <= yy < self.ny:
            return yy*self.nx+xx
        return None

    def xy(self, p):
        return self.rx+3*(p % self.nx), self.ry+3*(p//self.nx)

    def click(self, state, i):
        x, y = self.xy(state[i])
        return ("ACTION6", int((y+1)*self.scale+self.sy),
                int((x+1)*self.scale+self.sx))

    def rectmask(self, x, y, w, h):
        x0, x1 = max(0, x), min(self.nx, x+w)
        y0, y1 = max(0, y), min(self.ny, y+h)
        if x1 <= x0 or y1 <= y0:
            return 0
        row = ((1 << (x1-x0))-1) << x0
        m = 0
        for yy in range(y0, y1):
            m |= row << (yy*self.nx)
        return m

    def build(self):
        N = self.N
        self.nb = [[-1]*4 for _ in range(N)]
        for p in range(N):
            x, y = p % self.nx, p//self.nx
            for d in range(4):
                xx, yy = x+DX[d], y+DY[d]
                if 0 <= xx < self.nx and 0 <= yy < self.ny:
                    self.nb[p][d] = yy*self.nx+xx
        wall = pit = 0
        for p in range(N):
            x, y = self.xy(p)
            if x+3 > 64 or y+3 > 63:
                wall |= 1 << p
                continue
            b = self.terrain[y:y+3, x:x+3]
            if np.any(b == 2):
                wall |= 1 << p
            if np.any(b == 15):
                pit |= 1 << p

        self.cells, self.masks, self.hard, self.pit = [], [], [], []
        for w, h, isc in self.types:
            cells = [(x, y) for y in range(h) for x in range(w)
                     if not isc or x == 1 or y == 1]
            self.cells.append(cells)
            masks, hs, ps = [], [], []
            for p in range(N):
                x, y = p % self.nx, p//self.nx
                if x+w > self.nx or y+h > self.ny:
                    masks.append(0)
                    hs.append(True)
                    ps.append(False)
                    continue
                m = sum(1 << ((y+dy)*self.nx+x+dx) for dx, dy in cells)
                masks.append(m)
                hs.append(bool(m & wall))
                ps.append(bool(m & pit))
            self.masks.append(masks)
            self.hard.append(hs)
            self.pit.append(ps)

        self.ccache = {}
        self.ends, self.prefix = [], []
        for t in range(len(self.types)):
            ee, pp = [], []
            for p in range(N):
                er, pr = [], []
                for d in range(4):
                    q, path = p, []
                    if not self.hard[t][p]:
                        for j in range(50):
                            if j >= 5 and not self.pit[t][q]:
                                break
                            r = self.nb[q][d]
                            if r < 0 or self.hard[t][r]:
                                break
                            q = r
                            path.append(q)
                    er.append(q)
                    pr.append(tuple(path[:4]))
                ee.append(er)
                pp.append(pr)
            self.ends.append(ee)
            self.prefix.append(pp)

        self.W, self.Nav, self.reg = {}, {}, {}
        for t in self.atypes:
            w = np.full((N, N), INF, np.int16)
            np.fill_diagonal(w, 0)
            for p in range(N):
                if self.hard[t][p]:
                    continue
                for q in self.nb[p]:
                    if q >= 0 and not self.hard[t][q] and not self.pit[t][q]:
                        w[p, q] = 1
            self.W[t] = closure(w)
            regions = [-1]*N
            rno = 0
            for p in range(N):
                if self.hard[t][p] or self.pit[t][p] or regions[p] >= 0:
                    continue
                regions[p] = rno
                q = [p]
                while q:
                    v = q.pop()
                    for z in self.nb[v]:
                        if (z >= 0 and regions[z] < 0 and
                                not self.hard[t][z] and not self.pit[t][z]):
                            regions[z] = rno
                            q.append(z)
                rno += 1
            self.reg[t] = regions

        self.launch = {}
        for t in self.atypes:
            launches = []
            for p in range(N):
                if self.hard[t][p] or self.pit[t][p]:
                    continue
                for d in range(4):
                    e = self.ends[t][p][d]
                    if (e != p and not self.pit[t][e] and
                            self.reg[t][e] != self.reg[t][p] and
                            any(self.contacts(t, u, d)[p] for u in self.atypes)):
                        launches.append((p, d, e))
            self.launch[t] = launches

        for t in self.atypes:
            m = self.W[t].copy()
            if len(self.actors) > 1:
                for p, d, e in self.launch[t]:
                    m[p, e] = min(int(m[p, e]), 4)
            if self.bombs:
                dirs = set(self.bomb_dir(i) for i in self.bombs)
                for p in range(N):
                    if self.hard[t][p]:
                        continue
                    for d in dirs:
                        q = p
                        for j in range(5):
                            z = self.nb[q][d]
                            if z < 0 or self.hard[t][z]:
                                break
                            q = z
                            e = q
                            for _ in range(30):
                                if not self.pit[t][e]:
                                    break
                                z = self.nb[e][d]
                                if z < 0 or self.hard[t][z]:
                                    break
                                e = z
                            if not self.pit[t][e]:
                                m[p, e] = min(int(m[p, e]), 6)
            self.Nav[t] = closure(m)

        self.P = {}
        for t in set(self.ot[i] for i in self.crosses+self.bombs):
            m = np.full((N, N), INF, np.int16)
            np.fill_diagonal(m, 0)
            for p in range(N):
                if self.hard[t][p]:
                    continue
                for d in range(4):
                    contact = any(self.contacts(t, u, d)[p] for u in self.atypes)
                    e = self.ends[t][p][d]
                    if contact and e != p:
                        m[p, e] = 1
                    if contact or self.bombs:
                        for e in self.prefix[t][p][d]:
                            if not self.pit[t][e]:
                                m[p, e] = min(int(m[p, e]), 4)
            self.P[t] = closure(m)

        self.parking = {t: set() for t in self.atypes}
        for gt, gp, isc in self.goals:
            for t in self.atypes:
                if t == gt and not isc:
                    self.parking[t].add(gp)
                for d in range(4):
                    self.parking[t].update(self.contacts(gt, t, d)[gp])
        self.blastcache = {}
        self.routecache = {}
        self.helpercache = {}
        self.bmovecache = {}

    def contacts(self, victim, actor, d):
        key = (victim, actor, d)
        if key in self.ccache:
            return self.ccache[key]
        vc = set(self.cells[victim])
        aw, ah, _ = self.types[actor]
        vw, vh, _ = self.types[victim]
        offsets = []
        for yy in range(-ah, vh+1):
            for xx in range(-aw, vw+1):
                ac = {(x+xx, y+yy) for x, y in self.cells[actor]}
                if ac & vc:
                    continue
                moved = {(x+DX[d], y+DY[d]) for x, y in ac}
                if moved & vc:
                    offsets.append((xx, yy))
        result = []
        for p in range(self.N):
            px, py = p % self.nx, p//self.nx
            a = []
            for dx, dy in offsets:
                x, y = px+dx, py+dy
                if not (0 <= x < self.nx and 0 <= y < self.ny):
                    continue
                q = y*self.nx+x
                z = self.nb[q][d]
                if (z >= 0 and not self.hard[actor][q] and not self.pit[actor][q]
                        and not self.hard[actor][z] and not self.pit[actor][z]):
                    a.append(q)
            result.append(tuple(a))
        self.ccache[key] = result
        return result

    def bomb_dir(self, i):
        return {0: 1, 90: 2, 180: 0, 270: 3}[self.meta[i]["rot"]]

    def stripe(self, t, p, d, stage):
        x, y = p % self.nx, p//self.nx
        w, h, _ = self.types[t]
        if d == 0:
            return self.rectmask(x, y-stage, w, stage)
        if d == 1:
            return self.rectmask(x, y+h, w, stage)
        if d == 2:
            return self.rectmask(x-stage, y, stage, h)
        return self.rectmask(x+w, y, stage, h)

    def simulate(self, s, actor, d):
        n = self.n
        pos = list(s[:n])
        masks = [self.masks[self.ot[i]][pos[i]] for i in range(n)]
        packed = s[n+2]
        dirs = [((packed >> (3*i)) & 7)-1 for i in range(n)]

        def force(i, direction, depth=0):
            if depth > n+2:
                return False
            t = self.ot[i]
            old = pos[i]
            q = self.nb[old][direction]
            if q < 0 or self.hard[t][q]:
                return False
            oldmask = masks[i]
            pos[i] = q
            masks[i] = self.masks[t][q]
            hits = [j for j in range(n) if j != i and masks[i] & masks[j]]
            for j in hits:
                if not force(j, direction, depth+1):
                    pos[i], masks[i] = old, oldmask
                    return False
            return True

        t = self.ot[actor]
        q = self.nb[pos[actor]][d]
        if q >= 0 and not self.hard[t][q] and not self.pit[t][q]:
            mm = self.masks[t][q]
            hits = [j for j in range(n) if j != actor and mm & masks[j]]
            if not hits:
                pos[actor], masks[actor] = q, mm
            else:
                for j in hits:
                    dirs[j] = d
                for counter in range(60):
                    done = True
                    for j in hits:
                        if counter >= 5 and not self.pit[self.ot[j]][pos[j]]:
                            continue
                        if force(j, d):
                            done = False
                    if done:
                        break

        enemy = list(s[n+3:])
        if self.ne and self.crosses:
            tx, ty = self.xy(pos[self.crosses[0]])
            for j in range(self.ne):
                x, y = enemy[2*j:2*j+2]
                for _ in range(6):
                    dx, dy = tx-x, ty-y
                    if abs(dx) > abs(dy):
                        x += 1 if dx > 0 else -1
                    else:
                        y += 1 if dy > 0 else -1
                    if x == tx and y == ty:
                        return None
                enemy[2*j:2*j+2] = [x, y]

        phase = (s[n+1]+1) % self.cycle
        firing = [i for i in self.bombs if phase % self.meta[i]["h"] == 0]
        if firing:
            for stage in (1, 2, 3):
                for b in firing:
                    direction = self.bomb_dir(b)
                    blast = self.stripe(self.ot[b], pos[b], direction, stage)
                    for j in range(n):
                        if masks[j] & blast:
                            force(j, direction)
                            dirs[j] = direction
            for _ in range(60):
                moved = False
                for j in range(n):
                    if dirs[j] >= 0 and self.pit[self.ot[j]][pos[j]]:
                        if force(j, dirs[j]):
                            moved = True
                if not moved:
                    break
        packed = 0
        if self.bombs:
            for j in range(n):
                if dirs[j] >= 0 and self.pit[self.ot[j]][pos[j]]:
                    packed |= (dirs[j]+1) << (3*j)
        return tuple(pos)+(actor, phase, packed)+tuple(enemy)

    def matching(self, g):
        t, p, isc = self.goals[g]
        return [i for i in (self.crosses if isc else self.actors) if self.ot[i] == t]

    def satisfied(self, s, g):
        return any(s[i] == self.goals[g][1] for i in self.matching(g))

    def state_from(self, data, predicted):
        obs_objects = data["objects"]
        groups = {}
        for j, o in enumerate(obs_objects):
            groups.setdefault((o["k"], o["w"], o["h"]), []).append(j)
        assigned = {}
        for key, js in groups.items():
            ids = [i for i, o in enumerate(self.meta)
                   if (o["k"], o["w"], o["h"]) == key]
            if len(ids) != len(js):
                raise ValueError("piece count changed")
            def cost(i, j):
                o = obs_objects[j]
                xy = self.xy(predicted[i]) if predicted is not None else (
                    self.meta[i]["x"], self.meta[i]["y"])
                return abs(xy[0]-o["x"])+abs(xy[1]-o["y"])
            if len(ids) <= 6:
                perm = min(itertools.permutations(js),
                           key=lambda z: sum(cost(i, j) for i, j in zip(ids, z)))
            else:
                left = set(js)
                perm = []
                for i in ids:
                    j = min(left, key=lambda j: cost(i, j))
                    left.remove(j)
                    perm.append(j)
            assigned.update(zip(ids, perm))
        if len(assigned) != self.n:
            raise ValueError("unrecognized piece")

        pos, selected = [], self.actors[0]
        rebuild = False
        phase = 0
        maxheight = 0
        for i in range(self.n):
            o = obs_objects[assigned[i]]
            p = self.pos(o["x"], o["y"])
            if p is None:
                raise ValueError("off lattice")
            pos.append(p)
            if o["selected"]:
                selected = i
            if o["k"] == "b":
                if o["known"] and o["rot"] != self.meta[i]["rot"]:
                    self.meta[i]["rot"] = o["rot"]
                    rebuild = True
                if o["h"] > maxheight:
                    phase = o["charge"]
                    maxheight = o["h"]
        enemy = []
        remaining = list(data["enemies"])
        for j in range(self.ne):
            if not remaining:
                break
            ref = (predicted[self.n+3+2*j:self.n+5+2*j]
                   if predicted is not None else remaining[0])
            k = min(range(len(remaining)),
                    key=lambda k: abs(remaining[k][0]-ref[0])+
                                  abs(remaining[k][1]-ref[1]))
            enemy.extend(remaining.pop(k))
        if len(enemy) != 2*self.ne:
            raise ValueError("enemy count changed")
        packed = predicted[self.n+2] if predicted is not None else 0
        state = tuple(pos)+(selected, phase % self.cycle, packed)+tuple(enemy)
        if predicted is not None:
            visible = np.isin(data["a"], (1, 2, 15))
            terrain = self.terrain.copy()
            terrain[visible] = data["a"][visible]
            if not np.array_equal(terrain, self.terrain):
                self.terrain = terrain
                rebuild = True
        if rebuild:
            self.build()
        return state

    def blast_options(self, actor_type, bomb_type, d):
        key = (actor_type, bomb_type, d)
        if key in self.blastcache:
            return self.blastcache[key]
        out = []
        aw, ah, _ = self.types[actor_type]
        bw, bh, _ = self.types[bomb_type]
        for bp in range(self.N):
            if self.hard[bomb_type][bp]:
                continue
            x, y = bp % self.nx, bp//self.nx
            for yy in range(max(0, y-3-ah+1), min(self.ny, y+bh+3)):
                for xx in range(max(0, x-3-aw+1), min(self.nx, x+bw+3)):
                    sp = yy*self.nx+xx
                    if self.hard[actor_type][sp] or self.pit[actor_type][sp]:
                        continue
                    if self.masks[actor_type][sp] & self.masks[bomb_type][bp]:
                        continue
                    if not (self.masks[actor_type][sp] &
                            self.stripe(bomb_type, bp, d, 3)):
                        continue
                    ep, moved = sp, False
                    for stage in (1, 2, 3):
                        if self.masks[actor_type][ep] & self.stripe(bomb_type, bp, d, stage):
                            q = self.nb[ep][d]
                            if q >= 0 and not self.hard[actor_type][q]:
                                ep, moved = q, True
                    if moved:
                        for _ in range(40):
                            if not self.pit[actor_type][ep]:
                                break
                            q = self.nb[ep][d]
                            if q < 0 or self.hard[actor_type][q]:
                                break
                            ep = q
                    if (not self.pit[actor_type][ep] and
                            self.reg[actor_type][sp] != self.reg[actor_type][ep]):
                        out.append((bp, sp, ep))
        arrays = (tuple(np.asarray(z, dtype=np.int32) for z in zip(*out))
                  if out else (np.array([], np.int32),)*3)
        self.blastcache[key] = arrays
        return arrays

    def helper_matrix(self, victim_type, helper_type):
        key = (victim_type, helper_type)
        if key in self.helpercache:
            return self.helpercache[key]
        ls = self.launch[victim_type]
        m = np.full((self.N, len(ls)), INF, np.int16)
        for k, (p, d, e) in enumerate(ls):
            contacts = self.contacts(victim_type, helper_type, d)[p]
            if contacts:
                m[:, k] = self.Nav[helper_type][:, list(contacts)].min(axis=1)
        self.helpercache[key] = m
        return m

    def route_info(self, t, targets):
        targets = tuple(sorted(set(targets)))
        key = (t, targets)
        if key in self.routecache:
            return self.routecache[key]
        walk = self.W[t][:, list(targets)].min(axis=1)
        nav = self.Nav[t][:, list(targets)].min(axis=1)
        ls = self.launch[t]
        if ls:
            starts = np.asarray([z[0] for z in ls], np.int32)
            rem = np.asarray([int(nav[z[2]]) for z in ls], np.float32)
        else:
            starts = np.array([], np.int32)
            rem = np.array([], np.float32)
        blasts = []
        for b in self.bombs:
            bp, sp, ep = self.blast_options(t, self.ot[b], self.bomb_dir(b))
            if len(bp):
                good = nav[ep] < INF
                blasts.append((b, bp[good], sp[good], nav[ep[good]].astype(np.float32)))
        result = (walk, nav, starts, rem, blasts)
        self.routecache[key] = result
        return result

    def bomb_move_cost(self, s, actor, b):
        ta, tb = self.ot[actor], self.ot[b]
        ap, bp = s[actor], s[b]
        key = (ta, ap, tb, bp)
        if key in self.bmovecache:
            return self.bmovecache[key]
        pmat = self.P[tb]
        costs = 2.5*pmat[bp].astype(np.float32)+12
        for d in range(4):
            e = self.ends[tb][bp][d]
            c = self.contacts(tb, ta, d)[bp]
            if e == bp or not c:
                continue
            approach = min(int(self.Nav[ta][ap, q]) for q in c)
            if approach >= INF:
                continue
            v = 2.5+2.5*pmat[e].astype(np.float32)+0.45*approach
            np.minimum(costs, v, out=costs)
        costs[bp] = 0
        if len(self.bmovecache) > 20000:
            self.bmovecache.clear()
        self.bmovecache[key] = costs
        return costs

    def route(self, s, actor, targets):
        if not targets:
            return 150.0
        t, p = self.ot[actor], s[actor]
        walk, nav, starts, rem, blasts = self.route_info(t, targets)
        best = float(walk[p])
        if len(self.actors) > 1 and len(starts):
            helpdist = np.full(len(starts), INF, np.float32)
            for j in self.actors:
                if j != actor:
                    v = self.helper_matrix(t, self.ot[j])[s[j]]
                    np.minimum(helpdist, v, out=helpdist)
            scores = self.W[t][p, starts]+3.0+rem+0.8*helpdist
            best = min(best, float(scores.min()))
        for b, bp, sp, remaining in blasts:
            if not len(bp):
                continue
            costs = self.bomb_move_cost(s, actor, b)
            scores = costs[bp]+self.W[t][p, sp]+remaining+3.0
            best = min(best, float(scores.min()))
        if best >= INF:
            best = min(180.0, float(nav[p])+25.0)
        return best

    def heuristic(self, s, goal):
        gt, gp, isc = self.goals[goal]
        ids = self.matching(goal)
        if not ids:
            return 200.0
        if not isc:
            result = min(self.route(s, i, (gp,)) for i in ids)
        else:
            result = 1000.0
            pm = self.P[gt]
            for i in ids:
                p = s[i]
                if p == gp:
                    result = 0.0
                    break
                fallback = 2.0*float(pm[p, gp])+20.0
                if fallback >= INF:
                    fallback = (abs(p % self.nx-gp % self.nx)+
                                abs(p//self.nx-gp//self.nx))*2+30.0
                cur = fallback
                for d in range(4):
                    e = self.ends[gt][p][d]
                    rest = 2.0+2.0*float(pm[e, gp]) if e != p else INF
                    for z in self.prefix[gt][p][d]:
                        if not self.pit[gt][z]:
                            rest = min(rest, 8.0+2.0*float(pm[z, gp]))
                    if rest >= INF:
                        continue
                    helper = 200.0
                    for a in self.actors:
                        c = self.contacts(gt, self.ot[a], d)[p]
                        if c:
                            helper = min(helper, self.route(s, a, c))
                    cur = min(cur, rest+helper)
                result = min(result, cur)
        # A small preference for retaining progress on the other goals.
        for g, (t, p, cross) in enumerate(self.goals):
            if g == goal or self.satisfied(s, g):
                continue
            mat = self.P.get(t) if cross else self.Nav.get(t)
            if mat is not None:
                ds = [int(mat[s[i], p]) for i in self.matching(g)]
                if ds:
                    result += 0.035*min(80, min(ds))
        return float(result)

    def choose_goal(self, s):
        unsolved = [g for g in range(len(self.goals)) if not self.satisfied(s, g)]
        if not unsolved:
            return None
        crosses = [g for g in unsolved if self.goals[g][2]]
        if crosses:
            return crosses[0]
        def score(g):
            t, p, _ = self.goals[g]
            ids = self.matching(g)
            if not ids:
                return -1000
            wd = min(int(self.W[t][s[i], p]) for i in ids)
            nd = min(int(self.Nav[t][s[i], p]) for i in ids)
            return (100 if wd >= INF else 0)+min(nd, 100)
        return max(unsolved, key=score)

    def macros(self, s, heuristic, deadline):
        n = self.n
        for actor in self.actors:
            if time.perf_counter() > deadline:
                return
            t, start = self.ot[actor], s[actor]
            others = 0
            for j in range(n):
                if j != actor:
                    others |= self.masks[self.ot[j]][s[j]]
            parent = {start: (-1, -1)}
            dist = {start: 0}
            q = deque([start])
            kicks = []
            while q:
                p = q.popleft()
                for d, z in enumerate(self.nb[p]):
                    if z < 0 or self.hard[t][z] or self.pit[t][z]:
                        continue
                    if self.masks[t][z] & others:
                        kicks.append((p, d))
                    elif z not in parent:
                        parent[z] = (p, d)
                        dist[z] = dist[p]+1
                        q.append(z)

            def path(p):
                r = []
                while p != start:
                    p, d = parent[p]
                    r.append(d)
                r.reverse()
                return tuple(r)

            switch = int(actor != s[n])
            emitted = set()
            for p, d in kicks:
                ss = list(s)
                ss[actor], ss[n] = p, actor
                ss = tuple(ss)
                ns = self.simulate(ss, actor, d)
                if ns is None or ns == ss or ns in emitted:
                    continue
                emitted.add(ns)
                yield ns, switch+dist[p]+1, actor, path(p)+(d,)

            parks = set(p for p in self.parking[t] if p in parent and p != start)
            groups = {}
            for p, d, e in self.launch[t]:
                if p not in parent or p == start:
                    continue
                ss = list(s)
                ss[actor], ss[n] = p, actor
                ss = tuple(ss)
                value = heuristic(ss)+0.4*dist[p]
                key = (self.reg[t][e], d)
                groups.setdefault(key, []).append((value, p))
            for vals in groups.values():
                vals.sort()
                parks.update(p for _, p in vals[:3])
            for p in parks:
                ss = list(s)
                ss[actor], ss[n] = p, actor
                ss = tuple(ss)
                if ss not in emitted:
                    yield ss, switch+dist[p], actor, path(p)


def search(world, start, goal, deadline, weight=2.25):
    cache = {}
    n = world.n

    def h(s):
        key = s[:n]
        if key not in cache:
            cache[key] = world.heuristic(s, goal)
        return cache[key]

    h0 = h(start)
    heap = [(weight*h0, h0, 0, start)]
    costs = {start: 0}
    parent = {}
    serial = 0
    best = start
    bestval = h0
    first = None
    found = False
    dynamic = bool(world.bombs or world.ne)
    expansions = 0

    while heap and time.perf_counter() < deadline:
        _, _, _, s = heapq.heappop(heap)
        g = costs[s]
        if world.satisfied(s, goal):
            best, found = s, True
            break
        expansions += 1
        if dynamic:
            edges = []
            for actor in world.actors:
                for d in range(4):
                    ns = world.simulate(s, actor, d)
                    if ns is not None and ns != s:
                        edges.append((ns, 1+int(actor != s[n]), actor, (d,)))
        else:
            edges = world.macros(s, h, deadline)
        for ns, cost, actor, directions in edges:
            ng = g+cost
            if ng >= costs.get(ns, 10**9):
                continue
            if ng > 180:
                continue
            hv = h(ns)
            costs[ns] = ng
            parent[ns] = (s, actor, directions)
            serial += 1
            value = hv+0.018*ng
            if first is None:
                first = ns
            if value < bestval:
                bestval, best = value, ns
            heapq.heappush(heap, (ng+weight*hv, hv, serial, ns))
            if world.satisfied(ns, goal):
                best, found = ns, True
                break
            if len(costs) > 180000 or time.perf_counter() >= deadline:
                break
        if found or len(costs) > 180000:
            break

    if best == start:
        if heap:
            choices = [entry for entry in heap if entry[3] in parent]
            if choices:
                best = min(choices)[3]
        if best == start and first is not None:
            best = first
    edges = []
    s = best
    while s in parent:
        previous, actor, directions = parent[s]
        edges.append((previous, actor, directions))
        s = previous
    edges.reverse()
    plan = []
    for previous, actor, directions in edges:
        if actor != previous[n]:
            plan.append(world.click(previous, actor))
        for d in directions:
            plan.append(("ACTION"+str(d+1), None, None))
    if not found:
        plan = plan[:8]
    return plan


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started+8.7
    data = perceive(obs)
    if not any(o["k"] == "a" for o in data["objects"]):
        arrows = [a for a in available if a in
                  ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]
        return (arrows[0] if arrows else available[0], None, None)

    world = memory.get("_world")
    expected = memory.get("_expected")
    rebuilt = False
    try:
        if world is None or len(data["objects"]) != world.n:
            raise ValueError("new board")
        old_rotations = tuple(o["rot"] for o in world.meta)
        state = world.state_from(data, expected)
        if old_rotations != tuple(o["rot"] for o in world.meta):
            memory["_plan"] = []
    except (ValueError, IndexError, KeyError):
        world = World(data)
        state = world.initial
        memory["_world"] = world
        memory["_plan"] = []
        memory["_goal"] = None
        expected = None
        rebuilt = True

    if expected is not None and state != expected:
        memory["_plan"] = []

    goal = memory.get("_goal")
    if goal is None or goal >= len(world.goals) or world.satisfied(state, goal):
        goal = world.choose_goal(state)
        memory["_goal"] = goal
        memory["_plan"] = []

    plan = memory.get("_plan", [])
    if not plan and goal is not None and time.perf_counter() < deadline:
        tries = memory.get("_searches", 0)
        weight = 2.25 if tries % 4 != 3 else 3.25
        plan = search(world, state, goal, deadline, weight)
        memory["_searches"] = tries+1

    if plan:
        action = plan.pop(0)
        memory["_plan"] = plan
    else:
        # A model-based one-step fallback also provides a harmless timing action.
        actor = state[world.n]
        choices = []
        for d in range(4):
            name = "ACTION"+str(d+1)
            if name not in available:
                continue
            ns = world.simulate(state, actor, d)
            if ns is None:
                continue
            if goal is not None and time.perf_counter() < deadline:
                val = world.heuristic(ns, goal)
            else:
                val = 0.0
            val += 1.0 if ns[:world.n] == state[:world.n] else 0.0
            choices.append((val, d))
        d = min(choices)[1] if choices else 0
        action = ("ACTION"+str(d+1), None, None)
        memory["_plan"] = []

    name, row, col = action
    if name not in available:
        name = next((a for a in available if a.startswith("ACTION") and a != "ACTION6"),
                    available[0])
        action = (name, None, None)
        memory["_plan"] = []

    if action[0] == "ACTION6":
        selected = state[world.n]
        for i in world.actors:
            if world.click(state, i)[1:] == action[1:]:
                selected = i
                break
        ss = list(state)
        ss[world.n] = selected
        memory["_expected"] = tuple(ss)
    elif action[0] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        d = int(action[0][-1])-1
        memory["_expected"] = world.simulate(state, state[world.n], d)
    else:
        memory["_expected"] = None
    memory["_world"] = world
    return action

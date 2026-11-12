import time
import math
import heapq
import itertools
from collections import Counter

import numpy as np


# Up, right, down, left.
_DIR = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _rot(x, y, k):
    k &= 3
    if k == 0:
        return x, y
    if k == 1:
        return -y, x
    if k == 2:
        return -x, -y
    return y, -x


def _vrot(x, y, k):
    return (
        np.where(k == 0, x, np.where(k == 1, -y,
                 np.where(k == 2, -x, y))),
        np.where(k == 0, y, np.where(k == 1, x,
                 np.where(k == 2, -y, -x)))
    )


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo = [(x, y)]
        comp = []
        while todo:
            x, y = todo.pop()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    todo.append((nx, ny))
        result.append(comp)
    return result


def _read_picture(obs):
    a = np.asarray(obs)
    ui = np.zeros((64, 64), dtype=bool)
    controls = []

    for comp in _components(a == 2):
        xs, ys = zip(*comp)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        if w < 5 or h < 5:
            continue
        box = a[y:y+h, x:x+w]
        counts = Counter(int(v) for v in box.ravel()
                         if int(v) not in (2, 3, 4, 5, 13, 15))
        if not counts:
            continue
        color = counts.most_common(1)[0][0]
        ui[y:y+h, x:x+w] = True
        if w == h:
            controls.append((color, 0, y+h//2, x+w//2))
        elif w > h:
            controls.append((color, -1, y+h//2, x+1))
            controls.append((color, 1, y+h//2, x+w-2))
        else:
            controls.append((color, -1, y+1, x+w//2))
            controls.append((color, 1, y+h-2, x+w//2))

    magenta = (a == 13) & ~ui
    targets = []
    cross_pixels = set()
    for y in range(1, 62):
        for x in range(1, 63):
            if (magenta[y-1, x] and magenta[y+1, x] and
                    magenta[y, x-1] and magenta[y, x+1]):
                targets.append((x-1, y-1))
                cross_pixels.update(((x, y-1), (x, y+1),
                                     (x-1, y), (x+1, y)))
    targets = list(dict.fromkeys(targets))
    dots = []
    for yy, xx in zip(*np.nonzero(magenta)):
        p = (int(xx), int(yy))
        if p not in cross_pixels:
            dots.append((p[0]-1, p[1]-1))

    seeds = set()
    for yy, xx in zip(*np.nonzero((a == 3) & ~ui)):
        x, y = int(xx), int(yy)
        if y == 63:
            continue
        for k in range(3):
            seeds.add((x-k, y-2, 0))
            seeds.add((x, y-k, 1))
            seeds.add((x-k, y, 2))
            seeds.add((x-2, y-k, 3))

    candidates = []
    forbidden = {1, 2, 3, 4, 5, 13, 15}
    for x, y, d in seeds:
        if not (0 <= x <= 61 and 0 <= y <= 60):
            continue
        if ui[y:y+3, x:x+3].any():
            continue
        box = a[y:y+3, x:x+3]
        edge = np.zeros((3, 3), dtype=bool)
        if d == 0:
            edge[2, :] = True
        elif d == 1:
            edge[:, 0] = True
        elif d == 2:
            edge[0, :] = True
        else:
            edge[:, 2] = True
        ev = box[edge]
        if not np.all((ev == 3) | (ev == 13)) or not np.any(ev == 3):
            continue
        vals = [int(v) for v in box[~edge] if int(v) != 13]
        if not vals or len(set(vals)) != 1 or vals[0] in forbidden:
            continue
        c = vals[0]
        vx, vy = _DIR[d]
        length = 1
        while length < 24:
            xx, yy = x+3*vx*length, y+3*vy*length
            if not (0 <= xx <= 61 and 0 <= yy <= 61):
                break
            if ui[yy:yy+3, xx:xx+3].any():
                break
            nxt = a[yy:yy+3, xx:xx+3]
            if not np.all((nxt == c) | (nxt == 13)):
                break
            length += 1
        candidates.append((c, x, y, length, d))

    lattice = Counter((x % 3, y % 3) for x, y in targets+dots)
    if not lattice:
        lattice = Counter((x % 3, y % 3)
                          for c, x, y, length, d in candidates)
    rx, ry = lattice.most_common(1)[0][0] if lattice else (0, 0)
    bars = sorted(
        set(v for v in candidates if v[1] % 3 == rx and v[2] % 3 == ry),
        key=lambda v: (v[0], v[2], v[1], v[4]))

    gm = (a == 1) & ~ui
    for y in range(ry, 62, 3):
        for x in range(rx, 62, 3):
            box = a[y:y+3, x:x+3]
            if (np.any(box == 1) and
                    np.all((box == 1) | (box == 13)) and
                    not ui[y:y+3, x:x+3].any()):
                gm[y:y+3, x:x+3] = True
    grey = []
    for comp in _components(gm):
        xs, ys = zip(*comp)
        x, y = min(xs), min(ys)
        grey.append((x, y, max(xs)-x+1, max(ys)-y+1))

    walls = (a == 15) & ~ui
    walls[63, :] = False
    return bars, grey, dots, targets, controls, ui, walls


def _axis_solutions(signs, wanted, limit, cache):
    key = (tuple(signs), int(wanted), limit)
    if key in cache:
        return cache[key]
    k = len(signs)
    wanted = int(wanted)
    if k == 0:
        out = np.empty((1 if wanted == 0 else 0, 0), dtype=np.int16)
    elif (limit+1) ** max(0, k-1) > 300000:
        return None
    else:
        if k == 1:
            first = np.empty((1, 0), dtype=np.int16)
            sums = np.zeros(1, dtype=np.int32)
        else:
            first = np.indices((limit+1,) * (k-1),
                               dtype=np.int16).reshape(k-1, -1).T
            sums = first.astype(np.int32).dot(
                np.asarray(signs[:-1], dtype=np.int32))
        last = (wanted-sums)*signs[-1]
        good = (last >= 0) & (last <= limit)
        out = np.column_stack((first[good], last[good])).astype(np.int16)
    if len(cache) < 3000:
        cache[key] = out
    return out


class _Model:
    def __init__(self, obs, deadline):
        (self.bars, self.grey, dots, self.targets, controls,
         self.ui, self.walls) = _read_picture(obs)
        self.n = n = len(self.bars)
        self.parents = [-1]*n
        self.gparents = [-1]*len(self.grey)
        self.offset = [(0, 0)]*n
        self.goffset = [(0, 0)]*len(self.grey)
        self.extra_walls = []

        def tip(i):
            c, x, y, length, d = self.bars[i]
            vx, vy = _DIR[d]
            return x+3*vx*(length-1), y+3*vy*(length-1)

        for p, bar in enumerate(self.bars):
            c, x, y, length, d = bar
            vx, vy = _DIR[d]
            ex, ey = x+3*vx*length, y+3*vy*length
            for j, child in enumerate(self.bars):
                if p == j or self.parents[j] != -1:
                    continue
                if (child[1], child[2]) != (ex, ey):
                    continue
                z, cyclic = p, False
                while z != -1:
                    if z == j:
                        cyclic = True
                        break
                    z = self.parents[z]
                if not cyclic:
                    self.parents[j] = p
            for j, (gx, gy, gw, gh) in enumerate(self.grey):
                if self.gparents[j] == -1:
                    if gx <= ex and gy <= ey and ex+3 <= gx+gw and ey+3 <= gy+gh:
                        self.gparents[j] = p

        for j, p in enumerate(self.parents):
            if p != -1:
                tx, ty = tip(p)
                self.offset[j] = self.bars[j][1]-tx, self.bars[j][2]-ty
        for j, p in enumerate(self.gparents):
            if p != -1:
                tx, ty = tip(p)
                self.goffset[j] = self.grey[j][0]-tx, self.grey[j][1]-ty

        self.order = []
        seen = set()

        def visit(i):
            if i in seen:
                return
            if self.parents[i] != -1:
                visit(self.parents[i])
            seen.add(i)
            self.order.append(i)

        for i in range(n):
            visit(i)

        self.actions = []
        self.resize = set()
        self.rotate = set()
        for color, mode, row, col in controls:
            ids = tuple(i for i, b in enumerate(self.bars) if b[0] == color)
            if ids:
                self.actions.append((mode, ids, row, col, color))
                (self.resize if mode else self.rotate).update(ids)

        lengths = [b[3] for b in self.bars]
        rel = []
        for i, b in enumerate(self.bars):
            p = self.parents[i]
            rel.append(b[4] if p == -1 else (b[4]-self.bars[p][4]) % 4)
        self.start = tuple(lengths+rel)
        self.maxlen = max([22]+lengths)

        self.markers = []
        for mx, my in dots:
            owner = -1
            for i, b in enumerate(self.bars):
                bx, by, bw, bh = self._bar_rect(b[1], b[2], b[3], b[4])
                if bx <= mx and by <= my and mx+3 <= bx+bw and my+3 <= by+bh:
                    owner = i
                    break
            if owner >= 0:
                tx, ty = tip(owner)
                self.markers.append((owner, mx-tx, my-ty))
                continue
            for j, (gx, gy, gw, gh) in enumerate(self.grey):
                if gx <= mx and gy <= my and mx+3 <= gx+gw and my+3 <= gy+gh:
                    owner = n+j
                    self.markers.append((owner, mx-gx, my-gy))
                    break
            if owner == -1:
                self.markers.append((-1, mx, my))

        self.pref = np.zeros((65, 65), dtype=np.int32)
        self.pref[1:, 1:] = self.walls.astype(np.int32).cumsum(0).cumsum(1)
        self.banned = set()
        self.visits = Counter()
        self.patterns = []
        self._make_patterns(min(deadline-0.6, time.perf_counter()+3.6))
        self.ignore = self.ui.copy()
        self.ignore[63, :] = True
        self.ignore |= self.render(self.start) != obs

    @staticmethod
    def _bar_rect(x, y, length, d):
        z = 3*(length-1)
        if d == 0:
            return x, y-z, 3, 3*length
        if d == 1:
            return x, y, 3*length, 3
        if d == 2:
            return x, y, 3, 3*length
        return x-z, y, 3*length, 3

    def geometry(self, state, check=False):
        n = self.n
        bases = [None]*n
        dirs = [0]*n
        tips = [None]*n
        rects = [None]*n
        for i in self.order:
            b = self.bars[i]
            p = self.parents[i]
            if p == -1:
                x, y = b[1], b[2]
                d = state[n+i]
            else:
                k = (dirs[p]-self.bars[p][4]) & 3
                ox, oy = _rot(*self.offset[i], k)
                x, y = tips[p][0]+ox, tips[p][1]+oy
                d = (dirs[p]+state[n+i]) & 3
            length = state[i]
            vx, vy = _DIR[d]
            bases[i] = x, y
            dirs[i] = d
            tips[i] = x+3*vx*(length-1), y+3*vy*(length-1)
            rects[i] = self._bar_rect(x, y, length, d)

        grefs, grots = [], []
        for j, (gx, gy, gw, gh) in enumerate(self.grey):
            p = self.gparents[j]
            if p == -1:
                x, y, k = gx, gy, 0
            else:
                k = (dirs[p]-self.bars[p][4]) & 3
                ox, oy = _rot(*self.goffset[j], k)
                x, y = tips[p][0]+ox, tips[p][1]+oy
            grefs.append((x, y))
            grots.append(k)
            if k == 0:
                r = x, y, gw, gh
            elif k == 1:
                r = x-gh+3, y, gh, gw
            elif k == 2:
                r = x-gw+3, y-gh+3, gw, gh
            else:
                r = x, y-gw+3, gh, gw
            rects.append(r)

        if check:
            pref = self.pref
            for i, (x, y, w, h) in enumerate(rects):
                if x < -96 or y < -96 or x+w > 160 or y+h > 160:
                    return None
                xa, xb = max(0, min(64, x)), max(0, min(64, x+w))
                ya, yb = max(0, min(64, y)), max(0, min(64, y+h))
                if pref[yb, xb]-pref[ya, xb]-pref[yb, xa]+pref[ya, xa]:
                    return None
                for j in range(i):
                    a, b, c, d = rects[j]
                    if x < a+c and a < x+w and y < b+d and b < y+h:
                        return None
                for a, b, c, d in self.extra_walls:
                    if x < a+c and a < x+w and y < b+d and b < y+h:
                        return None

        points = []
        for owner, ox, oy in self.markers:
            if owner == -1:
                points.append((ox, oy))
            elif owner < n:
                k = (dirs[owner]-self.bars[owner][4]) & 3
                dx, dy = _rot(ox, oy, k)
                points.append((tips[owner][0]+dx, tips[owner][1]+dy))
            else:
                j = owner-n
                dx, dy = _rot(ox, oy, grots[j])
                points.append((grefs[j][0]+dx, grefs[j][1]+dy))
        return bases, dirs, rects, points

    def _partial_valid(self, states, keep):
        """Vectorized wall/self collision test for inverse-kinematic goals."""
        count = len(states)
        n = self.n
        zeros = np.zeros(count, dtype=np.int32)
        dirs, tips, rects = [None]*n, [None]*n, [None]*n
        for i in self.order:
            b, p = self.bars[i], self.parents[i]
            if p == -1:
                x, y = zeros+b[1], zeros+b[2]
                d = states[:, n+i]
            else:
                k = (dirs[p]-self.bars[p][4]) & 3
                ox, oy = _vrot(self.offset[i][0], self.offset[i][1], k)
                x, y = tips[p][0]+ox, tips[p][1]+oy
                d = (dirs[p]+states[:, n+i]) & 3
            length = states[:, i].astype(np.int32)
            vx = np.where(d == 1, 1, np.where(d == 3, -1, 0))
            vy = np.where(d == 2, 1, np.where(d == 0, -1, 0))
            dirs[i] = d
            tips[i] = x+3*vx*(length-1), y+3*vy*(length-1)
            horizontal = (d & 1) != 0
            rects[i] = (
                x-np.where(d == 3, 3*(length-1), 0),
                y-np.where(d == 0, 3*(length-1), 0),
                np.where(horizontal, 3*length, 3),
                np.where(horizontal, 3, 3*length)
            )
        for j, (gx, gy, gw, gh) in enumerate(self.grey):
            p = self.gparents[j]
            if p == -1:
                x, y, k = zeros+gx, zeros+gy, zeros
            else:
                k = (dirs[p]-self.bars[p][4]) & 3
                ox, oy = _vrot(self.goffset[j][0], self.goffset[j][1], k)
                x, y = tips[p][0]+ox, tips[p][1]+oy
            rects.append((
                x-np.where(k == 1, gh-3, np.where(k == 2, gw-3, 0)),
                y-np.where(k == 2, gh-3, np.where(k == 3, gw-3, 0)),
                np.where((k & 1) != 0, gh, gw),
                np.where((k & 1) != 0, gw, gh)
            ))

        valid = np.ones(count, dtype=bool)
        chosen = [rects[i] for i in keep]
        pref = self.pref
        for i, (x, y, w, h) in enumerate(chosen):
            valid &= (x >= -96) & (y >= -96) & (x+w <= 160) & (y+h <= 160)
            xa, xb = np.clip(x, 0, 64), np.clip(x+w, 0, 64)
            ya, yb = np.clip(y, 0, 64), np.clip(y+h, 0, 64)
            valid &= (pref[yb, xb]-pref[ya, xb]-
                      pref[yb, xa]+pref[ya, xa]) == 0
            for a, b, c, d in chosen[:i]:
                valid &= ~((x < a+c) & (a < x+w) &
                           (y < b+d) & (b < y+h))
            for a, b, c, d in self.extra_walls:
                valid &= ~((x < a+c) & (a < x+w) &
                           (y < b+d) & (b < y+h))
        return valid

    def change(self, state, ai):
        mode, ids, row, col, color = self.actions[ai]
        s = list(state)
        n = self.n
        if mode:
            for i in ids:
                s[i] = max(1, s[i]+mode)
                if s[i] > self.maxlen:
                    return None
        else:
            for i in ids:
                q = s[n+i]
                s[n+i] = 1 if self.parents[i] != -1 and q == 3 else (q-1) & 3
        s = tuple(s)
        return None if s == state else s

    def won(self, geometry):
        points = set(geometry[3])
        return bool(self.targets) and all(t in points for t in self.targets)

    def render(self, state):
        frame = np.full((64, 64), 5, dtype=np.int8)
        frame[self.walls] = 15
        bases, dirs, rects, points = self.geometry(state)

        def fill(x, y, w, h, color):
            xa, ya = max(0, x), max(0, y)
            xb, yb = min(64, x+w), min(64, y+h)
            if xa < xb and ya < yb:
                frame[ya:yb, xa:xb] = color

        for i, r in enumerate(rects):
            fill(*r, self.bars[i][0] if i < self.n else 1)
        for i, (x, y) in enumerate(bases):
            d = dirs[i]
            if d == 0:
                fill(x, y+2, 3, 1, 3)
            elif d == 1:
                fill(x, y, 1, 3, 3)
            elif d == 2:
                fill(x, y, 3, 1, 3)
            else:
                fill(x+2, y, 1, 3, 3)
        for x, y in points:
            fill(x+1, y+1, 1, 1, 13)
        for x, y in self.targets:
            for dx, dy in ((1, 0), (0, 1), (2, 1), (1, 2)):
                fill(x+dx, y+dy, 1, 1, 13)
        return frame

    def agrees(self, obs, state):
        return not np.any((self.render(state) != obs) & ~self.ignore)

    def _chain(self, owner):
        if owner == -1:
            return []
        if owner >= self.n:
            owner = self.gparents[owner-self.n]
        chain = []
        while owner != -1:
            chain.append(owner)
            owner = self.parents[owner]
        return chain[::-1]

    def _make_patterns(self, deadline):
        self.patterns = []
        n = self.n
        child_dist = np.full((4, 4), 1000, dtype=np.int16)
        for start in range(4):
            q = start
            for steps in range(5):
                child_dist[start, q] = min(child_dist[start, q], steps)
                q = 1 if q == 3 else (q-1) & 3
        root_dist = np.array([[(s-t) % 4 for t in range(4)]
                              for s in range(4)], dtype=np.int16)
        axis_cache = {}

        fixed = set()
        for i in range(n+len(self.grey)):
            chain = self._chain(i)
            if not any(j in self.resize or j in self.rotate for j in chain):
                fixed.add(i)

        for mi, marker in enumerate(self.markers):
            chain = self._chain(marker[0])
            grow = [i for i in chain if i in self.resize]
            turn = [i for i in chain if i in self.rotate]
            options = []
            for i in turn:
                if self.parents[i] == -1 or self.start[n+i] == 2:
                    options.append((0, 1, 2, 3))
                else:
                    options.append((0, 1, 3))
            total = math.prod(map(len, options)) if options else 1
            stride = max(1, (total+1023)//1024)
            current = tuple(self.start[n+i] for i in turn)
            combos = [
                combo for k, combo in enumerate(itertools.product(*options))
                if k % stride == 0 or combo == current
            ] or [current]

            keep = set(chain) | fixed
            for j, p in enumerate(self.gparents):
                if p in chain:
                    keep.add(n+j)
            keep = sorted(keep)
            exact = [[] for _ in self.targets]
            complete = [stride == 1 for _ in self.targets]
            baselines, coefficients = [], []

            for combo in combos:
                st = list(self.start)
                for i in grow:
                    st[i] = 1
                for i, q in zip(turn, combo):
                    st[n+i] = q
                geom = self.geometry(tuple(st))
                base = geom[3][mi]
                coeff = [_DIR[geom[1][i]] for i in grow]
                baselines.append(base)
                coefficients.append(coeff)

                for ti, target in enumerate(self.targets):
                    dx, dy = target[0]-base[0], target[1]-base[1]
                    if dx % 3 or dy % 3:
                        continue
                    if time.perf_counter() >= deadline:
                        complete[ti] = False
                        continue
                    xi = [j for j, v in enumerate(coeff) if v[0]]
                    yi = [j for j, v in enumerate(coeff) if v[1]]
                    xs = _axis_solutions([coeff[j][0] for j in xi],
                                         dx//3, self.maxlen-1, axis_cache)
                    ys = _axis_solutions([coeff[j][1] for j in yi],
                                         dy//3, self.maxlen-1, axis_cache)
                    if xs is None or ys is None:
                        complete[ti] = False
                        continue
                    count = len(xs)*len(ys)
                    if not count:
                        continue
                    if count > 180000:
                        complete[ti] = False
                        continue
                    for first in range(0, count, 4096):
                        if time.perf_counter() >= deadline:
                            complete[ti] = False
                            break
                        indices = np.arange(first, min(count, first+4096))
                        states = np.tile(np.asarray(st, dtype=np.int16),
                                         (len(indices), 1))
                        xx = xs[indices//len(ys)]
                        yy = ys[indices % len(ys)]
                        for j, v in enumerate(xi):
                            states[:, grow[v]] = xx[:, j]+1
                        for j, v in enumerate(yi):
                            states[:, grow[v]] = yy[:, j]+1
                        good = self._partial_valid(states, keep)
                        if np.any(good):
                            columns = grow+[n+i for i in turn]
                            exact[ti].append(states[good][:, columns])

            pcount = len(combos)
            base = np.asarray(baselines, dtype=np.float64)
            coef = np.asarray(coefficients, dtype=np.float64).reshape(
                pcount, len(grow), 2)
            targets = np.asarray(self.targets, dtype=np.float64).reshape(-1, 2)
            want = (targets[None, :, :]-base[:, None, :])/3.0
            low = np.minimum(coef, 0).sum(axis=1)*(self.maxlen-1)
            high = np.maximum(coef, 0).sum(axis=1)*(self.maxlen-1)
            valid = np.all(
                (want >= low[:, None, :]-1e-6) &
                (want <= high[:, None, :]+1e-6) &
                (np.abs(want-np.rint(want)) < 1e-6), axis=2)

            rgroups, lgroups = {}, {}
            for j, i in enumerate(turn):
                rgroups.setdefault(self.bars[i][0], []).append(j)
            for j, i in enumerate(grow):
                lgroups.setdefault(self.bars[i][0], []).append(j)
            tables = []
            for j, i in enumerate(turn):
                table = root_dist if self.parents[i] == -1 else child_dist
                desired = np.array([c[j] for c in combos], dtype=int)
                tables.append(table[:, desired])

            goal_data = []
            for ti, chunks in enumerate(exact):
                if not chunks:
                    goal_data.append(None if not complete[ti] else False)
                    continue
                goals = np.concatenate(chunks, axis=0)
                if goals.shape[1]:
                    goals = np.unique(goals, axis=0)
                else:
                    goals = goals[:1]
                if len(goals) > 20000:
                    initial = np.array(
                        [self.start[i] for i in grow] +
                        [self.start[n+i] for i in turn])
                    rank = np.abs(goals-initial).sum(axis=1)
                    goals = goals[np.argpartition(rank, 19999)[:20000]]
                gtables = []
                for j, i in enumerate(turn):
                    table = root_dist if self.parents[i] == -1 else child_dist
                    gtables.append(table[:, goals[:, len(grow)+j]])
                goal_data.append((goals[:, :len(grow)].astype(np.int16),
                                  gtables, len(goals)))

            self.patterns.append({
                "grow": grow, "turn": turn, "coef": coef, "want": want,
                "valid": valid, "tables": tables,
                "rgroups": list(rgroups.values()),
                "lgroups": list(lgroups.values()),
                "mult": max([1]+[len(g) for g in lgroups.values()]),
                "goals": goal_data, "cache": {}, "p": pcount,
            })

    def heuristic(self, state):
        rows = []
        n = self.n
        for p in self.patterns:
            grow, turn = p["grow"], p["turn"]
            key = tuple(state[i] for i in grow)+tuple(state[n+i] for i in turn)
            if key in p["cache"]:
                rows.append(p["cache"][key])
                continue

            relaxed = None
            costs = []
            for ti, goal in enumerate(p["goals"]):
                if goal is False:
                    costs.append(10000.0)
                elif goal is not None:
                    lengths, tables, count = goal
                    values = np.zeros(count, dtype=np.float64)
                    for group in p["lgroups"]:
                        if len(group) == 1:
                            j = group[0]
                            values += np.abs(lengths[:, j]-state[grow[j]])
                        else:
                            values += np.maximum.reduce([
                                np.abs(lengths[:, j]-state[grow[j]])
                                for j in group])
                    for group in p["rgroups"]:
                        values += np.maximum.reduce([
                            tables[j][state[n+turn[j]]] for j in group])
                    costs.append(float(values.min()))
                else:
                    if relaxed is None:
                        lens = np.array([state[i]-1 for i in grow],
                                        dtype=np.float64)
                        position = (np.einsum("pvc,v->pc", p["coef"], lens)
                                    if grow else np.zeros((p["p"], 2)))
                        distance = np.abs(
                            p["want"]-position[:, None, :]).sum(axis=2)
                        distance /= p["mult"]
                        rotation = np.zeros(p["p"], dtype=np.float64)
                        for group in p["rgroups"]:
                            rotation += np.maximum.reduce([
                                p["tables"][j][state[n+turn[j]]]
                                for j in group])
                        relaxed = np.where(
                            p["valid"], distance+rotation[:, None], 10000.0)
                    costs.append(float(relaxed[:, ti].min()))
            costs = tuple(costs)
            if len(p["cache"]) > 60000:
                p["cache"].clear()
            p["cache"][key] = costs
            rows.append(costs)

        target_count = len(self.targets)
        if not target_count:
            return 10000.0
        dp = {0: 0.0}
        for row in rows:
            nd = dict(dp)
            for mask, value in dp.items():
                for j in range(target_count):
                    if not ((mask >> j) & 1):
                        nm = mask | (1 << j)
                        nv = value+row[j]
                        if nv < nd.get(nm, float("inf")):
                            nd[nm] = nv
            dp = nd
        return dp.get((1 << target_count)-1, 10000.0)

    def learn_boundary(self, previous, predicted):
        """An invisible border can be inferred from an otherwise legal rejection."""
        old = self.geometry(previous)[2]
        new = self.geometry(predicted)[2]
        additions = []
        for before, after in zip(old, new):
            if before == after:
                continue
            x, y, w, h = after
            ox, oy, ow, oh = before
            if y < 0 and oy >= 0:
                additions.append((x, -3, w, 3))
            if x < 0 and ox >= 0:
                additions.append((-3, y, 3, h))
            if x+w > 64 and ox+ow <= 64:
                additions.append((64, y, 3, h))
            if y+h > 64 and oy+oh <= 64:
                additions.append((x, 64, w, 3))
        for rect in additions:
            if rect not in self.extra_walls:
                self.extra_walls.append(rect)
        return bool(additions)


def _path(parents, state):
    result = []
    while parents[state] is not None:
        prev, ai = parents[state]
        result.append(ai)
        state = prev
    result.reverse()
    return result


def _search(model, start, deadline):
    if model.won(model.geometry(start)):
        return [], True

    h0 = model.heuristic(start)
    serial = 0
    weight = 2.1
    heap = [(weight*h0, h0, serial, 0, start)]
    costs = {start: 0}
    parents = {start: None}
    best_state = start
    best_value = h0+3*model.visits[start]
    expansions = 0

    while heap and len(costs) < 180000:
        if expansions % 24 == 0 and time.perf_counter() >= deadline:
            break
        _, h, _, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        expansions += 1

        for ai in range(len(model.actions)):
            if (state, ai) in model.banned:
                continue
            nxt = model.change(state, ai)
            if nxt is None:
                continue
            ng = g+1
            if costs.get(nxt, 10**9) <= ng:
                continue
            geom = model.geometry(nxt, check=True)
            if geom is None:
                continue
            costs[nxt] = ng
            parents[nxt] = state, ai
            if model.won(geom):
                return _path(parents, nxt), True
            nh = model.heuristic(nxt)
            novelty = 3*model.visits[nxt]
            value = nh+0.025*ng+novelty
            if value < best_value:
                best_value, best_state = value, nxt
            serial += 1
            heapq.heappush(heap, (
                ng+weight*nh+novelty, nh, serial, ng, nxt))

    if best_state != start:
        return _path(parents, best_state)[:5], False
    while heap:
        _, _, _, g, state = heapq.heappop(heap)
        if costs.get(state) == g and state != start:
            result = _path(parents, state)
            if result:
                return result[:4], False
    return [], False


def act(obs, available, memory):
    deadline = time.perf_counter()+8.65
    obs = np.asarray(obs, dtype=np.int8)

    if "ACTION6" not in available:
        return next((a for a in available if a != "RESET"), "RESET"), None, None

    model = memory.get("model")
    state = memory.get("state")
    plan = memory.get("plan", [])

    if model is not None and state is not None:
        if not model.agrees(obs, state):
            previous = memory.get("previous")
            ai = memory.get("last_action")
            if previous is not None and model.agrees(obs, previous):
                if ai is not None:
                    model.banned.add((previous, ai))
                if model.learn_boundary(previous, state):
                    model._make_patterns(min(deadline-0.6,
                                             time.perf_counter()+2.7))
                state, plan = previous, []
            else:
                model, plan = None, []

    if model is None:
        model = _Model(obs, deadline)
        state, plan = model.start, []
        memory.clear()
        memory["model"] = model

    model.visits[state] += 1

    if not model.actions:
        memory["state"] = state
        return "ACTION6", 0, 0

    if not plan and time.perf_counter() < deadline:
        plan, solved = _search(model, state, deadline)
        memory["complete_plan"] = solved

    if not plan:
        if model.won(model.geometry(state)):
            memory["state"] = state
            return "ACTION6", 0, 0

        candidates = []
        for ai in range(len(model.actions)):
            if (state, ai) in model.banned:
                continue
            nxt = model.change(state, ai)
            if nxt is None or model.geometry(nxt, check=True) is None:
                continue
            score = model.heuristic(nxt)+4*model.visits[nxt]
            candidates.append((score, ai))
        if candidates:
            plan = [min(candidates)[1]]
        else:
            memory.clear()
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION6", 0, 0

    ai = plan[0]
    nxt = model.change(state, ai)
    mode, ids, row, col, color = model.actions[ai]
    memory["previous"] = state
    memory["last_action"] = ai
    memory["state"] = nxt if nxt is not None else state
    memory["plan"] = plan[1:]
    return "ACTION6", int(row), int(col)

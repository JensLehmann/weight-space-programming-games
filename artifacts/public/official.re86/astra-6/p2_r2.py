import time
from collections import Counter, deque
from array import array
import numpy as np

_COLOURS = tuple(range(6, 15))
_DIRS = ((0, -3), (0, 3), (-3, 0), (3, 0))


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        directions += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        part = []
        while stack:
            xx, yy = stack.pop()
            part.append((xx, yy))
            for dx, dy in directions:
                nx, ny = xx + dx, yy + dy
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(part)
    return result


def _find_markers(obs, border=None):
    a = (obs == 4) if border is None else border
    result = []
    for y in range(1, 62):
        for x in range(1, 63):
            if a[y, x]:
                continue
            n = int(a[y-1:y+2, x-1:x+2].sum())
            if n >= 4:
                result.append((n, x, y))
    result.sort(reverse=True)
    chosen = []
    for n, x, y in result:
        if not any(abs(x-u) <= 2 and abs(y-v) <= 2 for u, v in chosen):
            chosen.append((x, y))
    return chosen


def _find_paints(obs):
    a = (obs[:63] == 2).astype(np.int16)
    integ = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def sums(size):
        return (integ[size:, size:] - integ[:-size, size:] -
                integ[size:, :-size] + integ[:-size, :-size])

    candidates = []
    for s in range(4, 10):
        outer = sums(s)
        inner_all = sums(s - 2)
        inner = inner_all[1:1+outer.shape[0], 1:1+outer.shape[1]]
        edge = outer - inner
        ys, xs = np.nonzero(edge >= max(2*s + 1, int(.62*(4*s-4))))
        for y, x in zip(ys.tolist(), xs.tolist()):
            inside = obs[y+1:y+s-1, x+1:x+s-1]
            vals = [int(v) for v in inside.ravel() if int(v) in _COLOURS]
            if not vals:
                continue
            colour, count = Counter(vals).most_common(1)[0]
            if count < max(2, (s-2)*(s-2)//3):
                continue
            score = float(edge[y, x]) / (4*s-4) + count / (s*s) * .1
            candidates.append((score, x, y, s, colour))
    candidates.sort(reverse=True)
    paints = []
    for score, x, y, s, c in candidates:
        if any(x < p[0]+p[2] and x+s > p[0] and
               y < p[1]+p[3] and y+s > p[1] for p in paints):
            continue
        paints.append((x, y, s, s, c))
    order6 = {14: 0, 6: 1, 10: 2, 7: 3, 9: 4, 8: 5, 11: 6}
    order5 = {10: 0, 7: 1, 8: 2, 12: 3, 6: 4, 11: 5, 14: 6, 9: 7}
    paints.sort(key=lambda p: (
        0 if p[2] == 6 else 1,
        (order6 if p[2] == 6 else order5).get(p[4], 20),
        p[1], p[0]))
    return paints


def _find_obstacles(obs):
    result = []
    for part in _components(obs[:63] == 1, True):
        if len(part) < 5:
            continue
        x0, x1 = min(p[0] for p in part), max(p[0] for p in part)
        y0, y1 = min(p[1] for p in part), max(p[1] for p in part)
        w, h = x1-x0+1, y1-y0+1
        rows = [0] * h
        for x, y in part:
            rows[y-y0] |= 1 << (x-x0)
        if w == h == 8:
            rows = [255, 231, 195, 129, 129, 195, 231, 255]
        result.append((x0, y0, w, h, tuple(rows)))
    return result


def _rank(kind, w, h, tag):
    if kind == "P":
        known = {(23, 23): 30, (29, 29): 34,
                 (37, 19): 40, (19, 19): 41, (27, 27): 42}
        return known.get((w, h), 33)
    if kind == "R":
        return 38
    if kind == "X":
        return 50 if tag == 2 else 45
    if kind == "D":
        return 51 if w == 19 else 46
    if kind == "V":
        return 48
    return 47


class _Form:
    def __init__(self, w, h, rows, tag, kind):
        self.w, self.h = w, h
        self.tag, self.kind = tag, kind
        rows = [int(r) for r in rows]
        cx, cy = w // 2, h // 2
        rows[cy] |= 1 << cx
        full = (1 << w) - 1
        both = full
        for r in rows:
            both &= r
        col = (both & -both).bit_length()-1 if both else w-1
        row = next((i for i, r in enumerate(rows) if r == full), h-1)
        if tag == 1:
            rows[cy] &= ~(1 << cx)
        elif tag == 2 or row == cy or col == cx:
            rows[cy] |= 1 << cx
        else:
            rows[cy] &= ~(1 << cx)
        self.rows = tuple(rows)
        selected = rows[:]
        selected[cy] |= 1 << cx
        self.selected = tuple(selected)
        both = full
        for r in selected:
            both &= r
        self.col = (both & -both).bit_length()-1 if both else w-1
        self.row = next((i for i, r in enumerate(selected) if r == full), h-1)
        points = []
        for y, r in enumerate(rows):
            while r:
                b = r & -r
                points.append((b.bit_length()-1, y))
                r -= b
        self.points = tuple(points)


def _shape_rows(kind, w, h, col=None, row=None):
    rows = [0] * h
    full = (1 << w) - 1
    if kind == "P":
        col = w // 2 if col is None else col
        row = h // 2 if row is None else row
        rows = [1 << col] * h
        rows[row] = full
    elif kind == "R":
        rows[0] = rows[-1] = full
        for y in range(1, h-1):
            rows[y] = 1 | (1 << (w-1))
    elif kind == "D":
        for y in range(h):
            d = abs(y - h//2)
            left, right = d, w-1-d
            if 0 <= left < w:
                rows[y] |= 1 << left
            if 0 <= right < w:
                rows[y] |= 1 << right
    elif kind == "X":
        for y in range(h):
            if y < w:
                rows[y] |= 1 << y
            if 0 <= w-1-y < w:
                rows[y] |= 1 << (w-1-y)
    elif kind == "H":
        rows[0] = full
    elif kind == "V":
        rows = [1] * h
    return rows


def _segments(scoremask, valid, dx, dy):
    if (dx, dy) == (1, 0):
        starts = [(0, y) for y in range(63)]
    elif (dx, dy) == (0, 1):
        starts = [(x, 0) for x in range(64)]
    elif dy == 1:
        starts = [(x, 0) for x in range(64)]
        starts += [(0, y) for y in range(1, 63)]
    else:
        starts = [(x, 62) for x in range(64)]
        starts += [(0, y) for y in range(62)]
    result = []

    def finish(run):
        if not run:
            return
        good = [i for i, (xx, yy) in enumerate(run) if scoremask[yy, xx]]
        if len(good) < 4:
            return
        lo, hi = good[0], good[-1]
        if hi-lo+1 < 4 or len(good) < .45*(hi-lo+1):
            return
        x0, y0 = run[lo]
        x1, y1 = run[hi]
        result.append((x0, y0, x1, y1))

    for sx, sy in starts:
        run = []
        x, y = sx, sy
        while 0 <= x < 64 and 0 <= y < 63:
            if valid[y, x]:
                run.append((x, y))
            else:
                finish(run)
                run = []
            x += dx
            y += dy
        finish(run)
    return result


def _recognize_shapes(obs, markers, paints):
    static_centres = np.zeros((64, 64), bool)
    for x, y in markers:
        static_centres[y, x] = True
    paintmask = np.zeros((64, 64), bool)
    for x, y, w, h, c in paints:
        paintmask[y:y+h, x:x+w] = True
    valid = np.isin(obs, (0,) + _COLOURS)
    zeroes = [(int(x), int(y)) for y, x in zip(*np.nonzero(obs[:63] == 0))]
    candidates, dedup = [], set()

    def add(kind, x, y, w, h, c, col=None, row=None):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if not (1 <= w <= 81 and 1 <= h <= 81):
            return
        cx, cy = x+w//2, y+h//2
        if not (0 <= cx < 64 and 0 <= cy < 64):
            return
        key = (kind, x, y, w, h, c, col, row)
        if key in dedup:
            return
        dedup.add(key)
        rows = _shape_rows(kind, w, h, col, row)
        bits, good = 0, 0
        for yy, r in enumerate(rows):
            gy = y+yy
            if not (0 <= gy < 63):
                continue
            while r:
                b = r & -r
                xx = b.bit_length()-1
                r -= b
                gx = x+xx
                if not (0 <= gx < 64) or (gx == cx and gy == cy):
                    continue
                v = int(obs[gy, gx])
                if v == c:
                    if not static_centres[gy, gx] and not paintmask[gy, gx]:
                        bits |= 1 << (gy*64+gx)
                        good += 1
                elif v == 0 or v in _COLOURS:
                    pass
                else:
                    return
        if good < 7:
            return
        tag = 1 if kind in ("R", "D") else 0
        if kind == "X" and (c == 11 or ((cx, cy) in zeroes and c != 8)):
            tag = 2
        bonus = 2.5 if (cx, cy) in zeroes else 0.0
        candidates.append((bits, good, bonus, kind, x, y, w, h, c,
                           col, row, tag, rows))

    def extend(lo, hi, centre, limit):
        if lo == 0 and hi < limit:
            lo = min(lo, 2*centre-hi)
        if hi == limit and lo > 0:
            hi = max(hi, 2*centre-lo)
        return lo, hi

    for c in _COLOURS:
        scoremask = (obs == c) & ~static_centres & ~paintmask
        if int(scoremask[:63].sum()) < 7:
            continue
        hs = _segments(scoremask, valid, 1, 0)
        vs = _segments(scoremask, valid, 0, 1)
        ds = _segments(scoremask, valid, 1, 1)
        es = _segments(scoremask, valid, 1, -1)

        for x0, yy, x1, _ in hs:
            add("H", x0, yy, x1-x0+1, 1, c)
            for zx, zy in zeroes:
                if zy == yy and x0 <= zx <= x1:
                    for r in {zx-x0, x1-zx}:
                        if r >= 4:
                            add("H", zx-r, yy, 2*r+1, 1, c)
        for xx, y0, _, y1 in vs:
            add("V", xx, y0, 1, y1-y0+1, c)
            for zx, zy in zeroes:
                if zx == xx and y0 <= zy <= y1:
                    for r in {zy-y0, y1-zy}:
                        if r >= 4:
                            add("V", xx, zy-r, 1, 2*r+1, c)

        for hx0, hy, hx1, _ in hs:
            for vx, vy0, _, vy1 in vs:
                if hx0 <= vx <= hx1 and vy0 <= hy <= vy1:
                    xc, yc = vx, hy
                    for zx, zy in zeroes:
                        if hx0 <= zx <= hx1 and vy0 <= zy <= vy1:
                            xc, yc = zx, zy
                    x0, x1 = extend(hx0, hx1, xc, 63)
                    y0, y1 = extend(vy0, vy1, yc, 62)
                    add("P", x0, y0, x1-x0+1, y1-y0+1, c,
                        vx-x0, hy-y0)
                    if (xc, yc) in zeroes:
                        for rx in {abs(xc-hx0), abs(hx1-xc)}:
                            for ry in {abs(yc-vy0), abs(vy1-yc)}:
                                if rx >= 3 and ry >= 3:
                                    ax, ay = xc-rx, yc-ry
                                    if ax <= vx <= xc+rx and ay <= hy <= yc+ry:
                                        add("P", ax, ay, 2*rx+1, 2*ry+1, c,
                                            vx-ax, hy-ay)
                if (abs(vx-hx0) <= 1 or abs(vx-hx1) <= 1) and (
                        abs(hy-vy0) <= 1 or abs(hy-vy1) <= 1):
                    add("R", hx0, vy0, hx1-hx0+1, vy1-vy0+1, c)

        for i, h1 in enumerate(hs):
            for h2 in hs[i+1:]:
                if h1[1] == h2[1]:
                    continue
                if abs(h1[0]-h2[0]) <= 1 and abs(h1[2]-h2[2]) <= 1:
                    x0, x1 = min(h1[0], h2[0]), max(h1[2], h2[2])
                    y0, y1 = sorted((h1[1], h2[1]))
                    add("R", x0, y0, x1-x0+1, y1-y0+1, c)
                    side = y1-y0+1
                    if x0 == 0:
                        add("R", x1-side+1, y0, side, side, c)
                    if x1 == 63:
                        add("R", x0, y0, side, side, c)
        for i, v1 in enumerate(vs):
            for v2 in vs[i+1:]:
                if v1[0] == v2[0]:
                    continue
                if abs(v1[1]-v2[1]) <= 1 and abs(v1[3]-v2[3]) <= 1:
                    x0, x1 = sorted((v1[0], v2[0]))
                    y0, y1 = min(v1[1], v2[1]), max(v1[3], v2[3])
                    add("R", x0, y0, x1-x0+1, y1-y0+1, c)

        for d in ds:
            a = d[1]-d[0]
            for e in es:
                b = e[1]+e[0]
                if (b-a) % 2:
                    continue
                cx, cy = (b-a)//2, (b+a)//2
                if not (d[0] <= cx <= d[2] and e[0] <= cx <= e[2]):
                    continue
                radii = {abs(d[0]-cx), abs(d[2]-cx),
                         abs(e[0]-cx), abs(e[2]-cx)}
                for r in radii:
                    if r >= 3:
                        add("X", cx-r, cy-r, 2*r+1, 2*r+1, c)
        for x0, y0, x1, y1 in ds + es:
            r = x1-x0
            if r >= 3:
                for cx, cy in ((x0, y1), (x1, y0)):
                    add("D", cx-r, cy-r, 2*r+1, 2*r+1, c)

    chosen, covered = [], 0
    while candidates and len(chosen) < 8:
        best, bestscore = None, 5.5
        for i, item in enumerate(candidates):
            new = (item[0] & ~covered).bit_count()
            if new < max(6, int(item[1]*.20)):
                continue
            score = new + item[2] - .002*(item[6]+item[7])
            if score > bestscore:
                best, bestscore = i, score
        if best is None:
            break
        item = candidates.pop(best)
        covered |= item[0]
        chosen.append(item)

    result = []
    for item in chosen:
        _, _, _, kind, x, y, w, h, c, col, row, tag, rows = item
        result.append(dict(x=x, y=y, w=w, h=h, c=c, tag=tag,
                           kind=kind, rows=rows))

    for c in _COLOURS:
        residual = (obs == c) & ~static_centres & ~paintmask
        for y, x in zip(*np.nonzero(residual[:63])):
            if (covered >> (int(y)*64+int(x))) & 1:
                residual[y, x] = False
        for part in _components(residual[:63], True):
            if len(part) < 12:
                continue
            x0, x1 = min(x for x, y in part), max(x for x, y in part)
            y0, y1 = min(y for x, y in part), max(y for x, y in part)
            if any(x0 <= o["x"]+o["w"]//2 <= x1 and
                   y0 <= o["y"]+o["h"]//2 <= y1 for o in result):
                continue
            w, h = x1-x0+1, y1-y0+1
            rows = [0]*h
            for x, y in part:
                rows[y-y0] |= 1 << (x-x0)
            result.append(dict(x=x0, y=y0, w=w, h=h, c=c,
                               tag=0, kind="G", rows=rows))
    result.sort(key=lambda o: _rank(o["kind"], o["w"], o["h"], o["tag"]))
    return result


class _Search:
    def __init__(self, player, start):
        self.p = player
        self.start = start
        self.queue = [start]
        self.depths = array('H', [0])
        self.head = 0
        self.parents = {start: -1}
        self.good = {}
        self.weak = {}
        self.finished = False

    def run(self, count, deadline):
        p = self.p
        stop = self.head+count
        while self.head < len(self.queue) and self.head < stop:
            if (self.head & 127) == 0 and time.perf_counter() >= deadline:
                return
            u = self.queue[self.head]
            d = self.depths[self.head]
            self.head += 1
            geo, c = u >> 4, u & 15
            hit = p.target_hits(geo)
            good = hit & p.colour_targets[c]
            key = (hit, good)
            if key not in self.weak:
                self.weak[key] = (d, u)
            if hit == good and good not in self.good:
                self.good[good] = (d, u)
            if d >= 145 or len(self.parents) >= 450000:
                continue
            for a, packed in enumerate(p.transitions(geo), 1):
                if packed < 0:
                    continue
                ng = packed >> p.paint_bits
                pm = packed & p.paint_mask
                nc = p.paint_colour(pm, c)
                v = (ng << 4) | nc
                if v not in self.parents:
                    self.parents[v] = (u << 3) | a
                    self.queue.append(v)
                    self.depths.append(d+1)
        if self.head >= len(self.queue):
            self.finished = True

    def path(self, end):
        result = []
        while end != self.start:
            value = self.parents.get(end, -1)
            if value < 0:
                return []
            result.append(value & 7)
            end = value >> 3
        result.reverse()
        return result


class _Player:
    def __init__(self, obs, old_targets=None, old_border=None):
        self.forms = []
        self.form_ids = {}
        self.geocache = {}
        self.hitcache = {}
        self.paint_lut = {}
        self.border = obs == 4
        if old_border is not None:
            self.border |= old_border
        self.markers = _find_markers(obs, self.border)
        old_targets = old_targets or {}
        self.targets = {xy: old_targets.get(xy) for xy in self.markers}
        self.paints = _find_paints(obs)
        self.paint_bits = max(1, len(self.paints))
        self.paint_mask = (1 << self.paint_bits)-1
        self.paint_order = list(range(len(self.paints)))
        self.obstacles = _find_obstacles(obs)
        descriptions = _recognize_shapes(obs, self.markers, self.paints)
        self.objects = []
        for o in descriptions:
            fid = self.form(o["w"], o["h"], o["rows"], o["tag"], o["kind"])
            cx, cy = o["x"]+o["w"]//2, o["y"]+o["h"]//2
            if 0 <= cx < 64 and 0 <= cy < 64:
                self.objects.append([fid*4096+cy*64+cx, o["c"]])
        self.selected = 0
        self.next_selected = {}
        self.lastkind = None
        self.last_obj = None
        self.last_paintmask = 0
        self.last_oldcolour = None
        self.last_reveal_axis = None
        self.reveal_visits = Counter()
        self.searches = None
        self.paths = None
        self.bestplan = None
        self.target_list = []
        self.colour_targets = [0]*16
        self.waits = 0
        self.empty_switches = 0
        self.reveal_stalls = 0
        self.observe(obs, initial=True)

    def form(self, w, h, rows, tag, kind):
        f = _Form(w, h, rows, tag, kind)
        key = (w, h, f.rows, tag)
        old = self.form_ids.get(key)
        if old is not None:
            return old
        fid = len(self.forms)
        self.forms.append(f)
        self.form_ids[key] = fid
        return fid

    def invalidate(self):
        self.searches = None
        self.paths = None
        self.bestplan = None
        self.hitcache.clear()
        self.empty_switches = 0

    def covers(self, geo, x, y, selected=False):
        f = self.forms[geo >> 12]
        cx, cy = geo & 63, (geo >> 6) & 63
        xx, yy = x-cx+f.w//2, y-cy+f.h//2
        if not (0 <= xx < f.w and 0 <= yy < f.h):
            return False
        rows = f.selected if selected else f.rows
        return bool((rows[yy] >> xx) & 1)

    def refresh_targets(self, obs):
        self.border |= obs == 4
        markers = _find_markers(obs, self.border)
        previous = self.targets
        updated = {xy: previous.get(xy) for xy in markers}
        changed = set(updated) != set(previous)
        for (x, y), c in list(updated.items()):
            if c is not None:
                continue
            hidden = any(self.covers(o[0], x, y, i == self.selected)
                         for i, o in enumerate(self.objects))
            v = int(obs[y, x])
            if not hidden and v in _COLOURS:
                updated[(x, y)] = v
                changed = True
        self.markers = markers
        self.targets = updated
        if changed:
            self.invalidate()

    def observe(self, obs, initial=False):
        zeroes = [(int(x), int(y)) for y, x in zip(*np.nonzero(obs[:63] == 0))]
        matches = []
        for i, (geo, c) in enumerate(self.objects):
            if (geo & 63, (geo >> 6) & 63) in zeroes:
                matches.append(i)
        oldsel = self.selected
        if matches:
            if self.lastkind == "move" and oldsel in matches:
                self.selected = oldsel
            else:
                self.selected = matches[0]
        elif self.lastkind == "switch":
            self.selected = self.next_selected.get(
                oldsel, (oldsel+1) % max(1, len(self.objects)))
        elif initial and zeroes and self.objects:
            zx, zy = zeroes[0]
            self.selected = min(
                range(len(self.objects)),
                key=lambda i: abs((self.objects[i][0] & 63)-zx) +
                              abs(((self.objects[i][0] >> 6) & 63)-zy))
        if self.lastkind == "switch" and matches:
            self.next_selected[oldsel] = self.selected

        if self.lastkind == "move" and self.objects:
            i = self.last_obj
            if i is not None and i < len(self.objects):
                geo, predicted = self.objects[i]
                f = self.forms[geo >> 12]
                x0 = (geo & 63)-f.w//2
                y0 = ((geo >> 6) & 63)-f.h//2
                counts = Counter()
                bad, visible = 0, 0
                for x, y in f.points:
                    xx, yy = x+x0, y+y0
                    if 0 <= xx < 64 and 0 <= yy < 63:
                        v = int(obs[yy, xx])
                        visible += 1
                        if v in _COLOURS:
                            counts[v] += 1
                        elif v != 0:
                            bad += 1
                if counts:
                    actual, n = counts.most_common(1)[0]
                    if actual != predicted and n >= max(3, visible//2):
                        self.objects[i][1] = actual
                        pm = self.last_paintmask
                        eligible = [j for j in self.paint_order
                                    if (pm >> j) & 1 and
                                    self.paints[j][4] != self.last_oldcolour]
                        actuals = [j for j in eligible if self.paints[j][4] == actual]
                        if actuals:
                            j = actuals[0]
                            first = min(self.paint_order.index(k) for k in eligible)
                            self.paint_order.remove(j)
                            self.paint_order.insert(first, j)
                            self.paint_lut.clear()
                        self.invalidate()
                if visible >= 12 and bad > max(4, visible*.18):
                    return False
                if zeroes and not matches:
                    return False
        self.refresh_targets(obs)
        self.lastkind = None
        return True

    @staticmethod
    def rect_overlap(f, x, y, rect, selected):
        rx, ry, rw, rh = rect[:4]
        if x >= rx+rw or x+f.w <= rx or y >= ry+rh or y+f.h <= ry:
            return False
        lo, hi = max(0, rx-x), min(f.w, rx+rw-x)
        bits = ((1 << (hi-lo))-1) << lo
        rows = f.selected if selected else f.rows
        return any(rows[yy] & bits
                   for yy in range(max(0, ry-y), min(f.h, ry+rh-y)))

    @staticmethod
    def obstacle_overlap(f, x, y, obstacle):
        ox, oy, ow, oh, orows = obstacle
        if x >= ox+ow or x+f.w <= ox or y >= oy+oh or y+f.h <= oy:
            return False
        shift = x-ox
        for gy in range(max(y, oy), min(y+f.h, oy+oh)):
            r = f.selected[gy-y]
            r = r << shift if shift >= 0 else r >> -shift
            if r & orows[gy-oy]:
                return True
        return False

    def make_geo(self, fid, x, y):
        f = self.forms[fid]
        cx, cy = x+f.w//2, y+f.h//2
        if not (0 <= cx < 64 and 0 <= cy < 64):
            return -1
        return fid*4096+cy*64+cx

    def one_move(self, geo, action):
        dx, dy = _DIRS[action-1]
        fid = geo >> 12
        f = self.forms[fid]
        x = (geo & 63)-f.w//2
        y = ((geo >> 6) & 63)-f.h//2
        nx, ny = x+dx, y+dy
        if not (0 <= nx+f.w//2 < 64 and 0 <= ny+f.h//2 < 64):
            return geo << self.paint_bits

        for obstacle in self.obstacles:
            if not self.obstacle_overlap(f, nx, ny, obstacle):
                continue
            if f.tag == 1:
                w, h = f.w, f.h
                if dx:
                    if w <= 6:
                        return geo << self.paint_bits
                    nw, nh = w-3, h+3
                    ny += h//2-nh//2
                    ny = round(ny/3)*3
                    if dx < 0:
                        nx -= dx
                else:
                    if h <= 6:
                        return geo << self.paint_bits
                    nw, nh = w+3, h-3
                    nx += w//2-nw//2
                    nx = round(nx/3)*3
                    if dy < 0:
                        ny -= dy
                nf = self.form(nw, nh, _shape_rows("R", nw, nh), 1, "R")
                ng = self.make_geo(nf, nx, ny)
                return -1 if ng < 0 else ng << self.paint_bits

            ox, oy, ow, oh = obstacle[:4]
            col, row = f.col, f.row
            vertical_inside = ox <= nx+col < ox+ow
            horizontal_inside = oy <= ny+row < oy+oh
            rows = list(f.selected)
            change = False
            if dx:
                backshift = -3 if dx > 0 else 3
                forwardshift = -backshift
                can_back = col > 0 if dx > 0 else col < f.w-2
                can_forward = col < f.w-2 if dx > 0 else col > 0
                dest = None
                if vertical_inside and horizontal_inside:
                    nx -= dx
                    ny -= dy
                elif vertical_inside:
                    if can_back:
                        dest = col+backshift
                    else:
                        nx -= dx
                        ny -= dy
                elif horizontal_inside:
                    nx -= dx
                    ny -= dy
                    if can_forward:
                        dest = col+forwardshift
                if dest is not None:
                    if not (-f.w <= dest < f.w):
                        return -1
                    dest %= f.w
                    for yy in range(f.h):
                        rows[yy] = (rows[yy] & ~(1 << col)) | (1 << dest)
                    rows[row] |= 1 << col
                    change = True
            else:
                backshift = -3 if dy > 0 else 3
                forwardshift = -backshift
                can_back = row > 0 if dy > 0 else row < f.h-2
                can_forward = row < f.h-2 if dy > 0 else row > 0
                dest = None
                if horizontal_inside and vertical_inside:
                    nx -= dx
                    ny -= dy
                elif horizontal_inside:
                    if can_back:
                        dest = row+backshift
                    else:
                        nx -= dx
                        ny -= dy
                elif vertical_inside:
                    nx -= dx
                    ny -= dy
                    if can_forward:
                        dest = row+forwardshift
                if dest is not None:
                    if not (-f.h <= dest < f.h):
                        return -1
                    dest %= f.h
                    rows[row] = 0
                    rows[dest] = (1 << f.w)-1
                    rows[row] |= 1 << col
                    change = True
            if change:
                rows[f.h//2] |= 1 << (f.w//2)
                fid = self.form(f.w, f.h, rows, f.tag, f.kind)
                f = self.forms[fid]

        ng = self.make_geo(fid, nx, ny)
        if ng < 0:
            return -1
        pm = 0
        for j, paint in enumerate(self.paints):
            if self.rect_overlap(f, nx, ny, paint, False):
                pm |= 1 << j
        return (ng << self.paint_bits) | pm

    def transitions(self, geo):
        old = self.geocache.get(geo)
        if old is not None:
            return old
        result = tuple(self.one_move(geo, a) for a in range(1, 5))
        self.geocache[geo] = result
        return result

    def paint_colour(self, mask, colour):
        if not mask:
            return colour
        lut = self.paint_lut.get(mask)
        if lut is None:
            lut = list(range(16))
            for c in range(16):
                for j in self.paint_order:
                    pc = self.paints[j][4]
                    if (mask >> j) & 1 and pc != c:
                        lut[c] = pc
                        break
            self.paint_lut[mask] = lut
        return lut[colour]

    def target_hits(self, geo):
        old = self.hitcache.get(geo)
        if old is not None:
            return old
        f = self.forms[geo >> 12]
        x0 = (geo & 63)-f.w//2
        y0 = ((geo >> 6) & 63)-f.h//2
        mask = 0
        for k, (x, y, c) in enumerate(self.target_list):
            xx, yy = x-x0, y-y0
            if 0 <= xx < f.w and 0 <= yy < f.h and (f.rows[yy] >> xx) & 1:
                mask |= 1 << k
        self.hitcache[geo] = mask
        return mask

    def begin_search(self):
        self.target_list = [(x, y, c) for (x, y), c in sorted(self.targets.items())
                            if c is not None]
        self.colour_targets = [0]*16
        for k, (x, y, c) in enumerate(self.target_list):
            self.colour_targets[c] |= 1 << k
        self.hitcache.clear()
        self.searches = [_Search(self, (geo << 4) | c)
                         for geo, c in self.objects]
        self.bestplan = None

    def combine(self, relaxed=False, partial=False):
        if not self.searches:
            return None
        full = (1 << len(self.target_list))-1
        choices = []
        for s in self.searches:
            if relaxed:
                opts = [(hit, good, cost, state)
                        for (hit, good), (cost, state) in s.weak.items()]
            else:
                items = sorted(s.good.items(),
                               key=lambda z: (z[1][0], -z[0].bit_count()))
                kept = []
                for mask, (cost, state) in items:
                    if any((mask | m) == m and d <= cost for m, d, st in kept):
                        continue
                    kept.append((mask, cost, state))
                opts = [(mask, mask, cost, state) for mask, cost, state in kept]
            if not opts:
                return None
            choices.append(opts)

        order = list(range(len(self.objects)))
        if relaxed:
            def rank(i):
                f = self.forms[self.objects[i][0] >> 12]
                return _rank(f.kind, f.w, f.h, f.tag)
            order.sort(key=rank)

        dp = {0: (0, ())}
        for i in order:
            nd = {}
            for covered, (cost, path) in dp.items():
                for hit, good, extra, state in choices[i]:
                    m = ((covered & ~hit) | good) if relaxed else covered | good
                    nc = cost + extra
                    old = nd.get(m)
                    if old is None or nc < old[0]:
                        nd[m] = (nc, path+((i, state),))
            dp = nd
            if len(dp) > 20000:
                keys = sorted(dp, key=lambda m: (-m.bit_count(), dp[m][0]))[:15000]
                dp = {m: dp[m] for m in keys}
        if full in dp:
            return dp[full]
        if partial and dp:
            bestmask = max(dp, key=lambda m: (m.bit_count(), -dp[m][0]))
            if dp[bestmask][0] > 0:
                return dp[bestmask]
        return None

    def plan(self, deadline):
        if self.searches is None:
            self.begin_search()
        rounds = 0
        while time.perf_counter() < deadline-.08:
            active = False
            for search in self.searches:
                if not search.finished:
                    search.run(512, deadline-.05)
                    active = True
                if time.perf_counter() >= deadline-.08:
                    break
            rounds += 1
            if rounds % 3 == 0 or not active:
                candidate = self.combine(False)
                if candidate and (self.bestplan is None or candidate[0] < self.bestplan[0]):
                    self.bestplan = candidate
                if self.bestplan is not None:
                    if all(s.finished for s in self.searches):
                        break
                    depths = [s.depths[s.head] if s.head < len(s.depths) else 1000
                              for s in self.searches]
                    if self.bestplan[0] <= 2*min(depths)+4:
                        break
            if all(s.finished for s in self.searches):
                break

        candidate = self.combine(False)
        if candidate and (self.bestplan is None or candidate[0] < self.bestplan[0]):
            self.bestplan = candidate
        if self.bestplan is None and all(s.finished for s in self.searches):
            self.bestplan = self.combine(True)
            if self.bestplan is None:
                self.bestplan = self.combine(False, partial=True)
        if self.bestplan is not None:
            self.paths = [deque() for _ in self.objects]
            for i, end in self.bestplan[1]:
                self.paths[i] = deque(self.searches[i].path(end))
            return True
        return False

    def reveal_action(self):
        unknown = [xy for xy, c in self.targets.items() if c is None]
        if not unknown:
            self.reveal_stalls = 0
            return None
        i = self.selected
        if not self.objects or not any(self.covers(self.objects[i][0], x, y, True)
                                       for x, y in unknown):
            self.reveal_stalls += 1
            return 5
        self.reveal_stalls = 0
        oldgeo, oldc = self.objects[i]
        best = None
        for a, packed in enumerate(self.transitions(oldgeo), 1):
            if packed < 0:
                continue
            ng = packed >> self.paint_bits
            if ng == oldgeo:
                continue
            uncovered = 0
            for x, y in unknown:
                hidden = self.covers(ng, x, y, True)
                if not hidden:
                    hidden = any(j != i and self.covers(o[0], x, y, False)
                                 for j, o in enumerate(self.objects))
                if not hidden:
                    uncovered += 1
            axis = 0 if a <= 2 else 1
            score = 20*uncovered - self.reveal_visits[(i, ng)]*3
            score -= .4 if axis == self.last_reveal_axis else 0
            score -= .05 if packed & self.paint_mask else 0
            if best is None or score > best[0]:
                best = (score, a, ng, axis)
        if best is None:
            return 5
        self.reveal_visits[(i, best[2])] += 1
        self.last_reveal_axis = best[3]
        return best[1]

    def emit(self, action):
        if action == 5:
            self.lastkind = "switch"
        elif 1 <= action <= 4 and self.objects:
            i = self.selected
            geo, c = self.objects[i]
            packed = self.transitions(geo)[action-1]
            self.lastkind = "move"
            self.last_obj = i
            self.last_oldcolour = c
            if packed >= 0:
                ng = packed >> self.paint_bits
                pm = packed & self.paint_mask
                self.last_paintmask = pm
                self.objects[i] = [ng, self.paint_colour(pm, c)]
            else:
                self.last_paintmask = 0
        return "ACTION" + str(action), None, None


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.2
    obs = np.asarray(obs, dtype=np.int16)
    p = memory.get("_re86_player")
    if p is None:
        p = _Player(obs)
        memory["_re86_player"] = p
    elif not p.observe(obs):
        p = _Player(obs, dict(p.targets), p.border)
        memory["_re86_player"] = p

    if not p.objects:
        action = "ACTION5" if "ACTION5" in available else next(
            (a for a in available if a != "RESET"), "RESET")
        return action, None, None

    reveal = p.reveal_action()
    if reveal is not None:
        if p.reveal_stalls > 2*len(p.objects)+2:
            p = _Player(obs, dict(p.targets), p.border)
            memory["_re86_player"] = p
            reveal = p.reveal_action()
            if reveal is None:
                reveal = 5
        return p.emit(reveal)
    if not p.targets:
        return p.emit(5)

    if p.paths is None:
        if not p.plan(deadline):
            p.waits += 1
            if p.waits > 10 and all(s.finished for s in p.searches):
                p = _Player(obs, dict(p.targets), p.border)
                memory["_re86_player"] = p
            return p.emit(5)
        p.waits = 0

    i = p.selected
    if i < len(p.paths) and p.paths[i]:
        a = p.paths[i].popleft()
        p.empty_switches = 0
        if "ACTION" + str(a) in available:
            return p.emit(a)

    if any(p.paths):
        return p.emit(5)

    p.empty_switches += 1
    if p.empty_switches > len(p.objects)+1:
        p.invalidate()
    return p.emit(5)

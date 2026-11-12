import time
import math
from collections import deque
import numpy as np

DIRS = ((0, -3), (0, 3), (-3, 0), (3, 0))


def pack(x, y):
    return int(x) * 256 + int(y) + 96


def unpack(p):
    return p // 256, p % 256 - 96


def shape_pixels(g, conservative=False):
    kind, w, h, vx, hy = g
    if kind == "C":
        p = {(x, hy) for x in range(w)}
        p.update((vx, y) for y in range(h))
    elif kind == "R":
        p = {(x, 0) for x in range(w)}
        p.update((x, h - 1) for x in range(w))
        p.update((0, y) for y in range(h))
        p.update((w - 1, y) for y in range(h))
    elif kind == "O":
        r = w // 2
        p = set()
        for y in range(h):
            d = r - abs(y - r)
            p.add((r - d, y))
            p.add((r + d, y))
    elif kind == "X":
        p = {(i, i) for i in range(w)}
        p.update((w - 1 - i, i) for i in range(w))
    elif kind == "D":
        p = {(i, i) for i in range(w)}
    else:
        p = {(w - 1 - i, i) for i in range(w)}
    if conservative and kind in ("X", "D", "E"):
        p.discard((w // 2, h // 2))
    return p


def detect_targets(obs):
    yellow = obs == 4
    ring = [(dx, dy) for dy in (-1, 0, 1)
            for dx in (-1, 0, 1) if dx or dy]
    counts = np.zeros((64, 64), dtype=np.int16)
    for dx, dy in ring:
        counts[1:63, 1:63] += yellow[1 + dy:63 + dy, 1 + dx:63 + dx]

    candidates = []
    ys, xs = np.where(counts >= 4)
    for y, x in zip(ys, xs):
        if not (1 <= x <= 62 and 1 <= y <= 61):
            continue
        if obs[y, x] == 4:
            continue
        support = {(x + dx, y + dy) for dx, dy in ring
                   if yellow[y + dy, x + dx]}
        corners = sum(int(yellow[y + dy, x + dx])
                      for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
        if corners < 2:
            continue
        candidates.append((len(support), corners, int(x), int(y), support))

    candidates.sort(reverse=True, key=lambda z: (z[0], z[1]))
    used = set()
    targets = []
    for count, corners, x, y, support in candidates:
        if support & used:
            continue
        if any(abs(x - xx) <= 2 and abs(y - yy) <= 2 for xx, yy in targets):
            continue
        targets.append((x, y))
        used.update(support)
    return sorted(targets, key=lambda z: (z[1], z[0]))


def components(mask):
    remaining = set(zip(*np.where(mask)))
    out = []
    while remaining:
        p = remaining.pop()
        part = {p}
        stack = [p]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = y + dy, x + dx
                    if q in remaining:
                        remaining.remove(q)
                        part.add(q)
                        stack.append(q)
        out.append(part)
    return out


def detect_stations(obs):
    a = (obs == 2).astype(np.int16)
    integ = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def squares(n):
        return (integ[n:, n:] - integ[:-n, n:]
                - integ[n:, :-n] + integ[:-n, :-n])

    candidates = []
    for n in range(4, 10):
        outer = squares(n)
        inner = squares(n - 2)[1:-1, 1:-1]
        rings = outer - inner
        yy, xx = np.where(
            (rings >= math.ceil(0.64 * (4 * n - 4)))
            & (inner <= max(1, n - 3))
        )
        for y, x in zip(yy, xx):
            patch = obs[y + 1:y + n - 1, x + 1:x + n - 1]
            vals = patch[(patch >= 6) & (patch <= 14)]
            if len(vals) < (n - 2) ** 2 // 2:
                continue
            hist = np.bincount(vals, minlength=16)
            c = int(hist.argmax())
            score = 3 * int(rings[y, x]) - 2 * (4 * n - 4)
            candidates.append((score, int(x), int(y), n, c))

    candidates.sort(reverse=True)
    used = np.zeros((64, 64), dtype=bool)
    stations = []
    for _, x, y, n, c in candidates:
        if used[y:y + n, x:x + n].any():
            continue
        stations.append((x, y, n, n, c))
        used[y:y + n, x:x + n] = True
    stations.sort(key=lambda z: (z[1], z[0]))
    return stations, used


def detect_obstacles(obs):
    mask = (obs == 1).copy()
    mask[63, :] = False
    groups = [c for c in components(mask) if len(c) >= 2]

    again = True
    while again:
        again = False
        for i in range(len(groups)):
            if again:
                break
            for j in range(i):
                both = groups[i] | groups[j]
                xs = [x for y, x in both]
                ys = [y for y, x in both]
                if max(xs) - min(xs) < 9 and max(ys) - min(ys) < 9:
                    groups[j] = both
                    groups.pop(i)
                    again = True
                    break

    obstacles = []
    for part in groups:
        if len(part) < 6:
            continue
        xs = [x for y, x in part]
        ys = [y for y, x in part]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        pixels = {(xx, yy) for yy, xx in part}
        if w == 8 and h == 8:
            holes = ((0, -1), (3, 4), (2, 5), (1, 6),
                     (1, 6), (2, 5), (3, 4), (0, -1))
            pixels = {(x + xx, y + yy)
                      for yy, (lo, hi) in enumerate(holes)
                      for xx in range(8) if not lo <= xx <= hi}
        obstacles.append((x, y, w, h, pixels))
    return obstacles


def image_lines():
    lines = []
    for y in range(63):
        lines.append(("H", [(x, y) for x in range(64)]))
    for x in range(64):
        lines.append(("V", [(x, y) for y in range(63)]))
    for d in range(-63, 63):
        p = [(x, x + d) for x in range(64) if 0 <= x + d < 63]
        if len(p) >= 5:
            lines.append(("D", p))
    for s in range(126):
        p = [(x, s - x) for x in range(64) if 0 <= s - x < 63]
        if len(p) >= 5:
            lines.append(("E", p))
    return lines


def detect_shapes(obs, targets, station_mask):
    isolated = np.zeros((64, 64), dtype=bool)
    for x, y in targets:
        c = int(obs[y, x])
        patch = obs[y - 1:y + 2, x - 1:x + 2]
        if np.count_nonzero(patch == c) <= 1:
            isolated[y, x] = True

    foreign = (obs >= 6) & (obs <= 14) & ~station_mask & ~isolated
    zeros = [(int(x), int(y)) for y, x in zip(*np.where(obs == 0))]
    lines = image_lines()
    all_candidates = []

    for color in range(6, 15):
        own = (obs == color) & ~station_mask & ~isolated
        own[63, :] = False
        if own.sum() < 7:
            continue
        allowed = foreign | (obs == color) | (obs == 0)
        segs = {k: [] for k in ("H", "V", "D", "E")}

        for orient, coords in lines:
            ids = [i for i, (x, y) in enumerate(coords) if own[y, x]]
            if len(ids) < 5:
                continue
            start = 0
            for k in range(1, len(ids) + 1):
                split = k == len(ids)
                if not split:
                    lo, hi = ids[k - 1], ids[k]
                    if hi - lo > 2:
                        split = not all(
                            allowed[coords[t][1], coords[t][0]]
                            for t in range(lo + 1, hi)
                        )
                if split:
                    group = ids[start:k]
                    if len(group) >= 5:
                        segs[orient].append(
                            (coords[group[0]], coords[group[-1]])
                        )
                    start = k

        candidates = set()

        def add(kind, x, y, w, h, vx=0, hy=0):
            x, y, w, h, vx, hy = map(int, (x, y, w, h, vx, hy))
            if not (1 <= w <= 85 and 1 <= h <= 85):
                return
            if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
                return
            if kind == "C" and not (0 <= vx < w and 0 <= hy < h):
                return
            candidates.add(((kind, w, h, vx, hy), x, y))

        def extend(lo, hi, center, edge):
            if lo == 0 and hi < edge:
                lo = min(lo, 2 * center - hi)
            if hi >= edge and lo > 0:
                hi = max(hi, 2 * center - lo)
            return lo, hi

        for a, b in segs["H"]:
            x0, yy = a
            x1 = b[0]
            for u, v in segs["V"]:
                xx, y0 = u
                y1 = v[1]
                if x0 <= xx <= x1 and y0 <= yy <= y1:
                    cx, cy = xx, yy
                    for zx, zy in zeros:
                        if x0 <= zx <= x1 and y0 <= zy <= y1:
                            if abs(zx - xx) <= 1 or abs(zy - yy) <= 1:
                                cx, cy = zx, zy
                    lx, rx = extend(x0, x1, cx, 63)
                    ty, by = extend(y0, y1, cy, 62)
                    if rx - lx >= 3 and by - ty >= 3:
                        add("C", lx, ty, rx - lx + 1, by - ty + 1,
                            xx - lx, yy - ty)

                # A perpendicular overlaid shape can hide an entire edge.
                # Extend the visible sides to test the missing edge instead
                # of interpreting the remaining three sides as a cross.
                if (min(abs(xx - x0), abs(xx - x1)) <= 2
                        and min(abs(yy - y0), abs(yy - y1)) <= 2
                        and x1 - x0 >= 3 and y1 - y0 >= 3):
                    for left in range(3):
                        for right in range(3):
                            lx, rx = x0 - left, x1 + right
                            if xx not in (lx, rx):
                                continue
                            for top in range(3):
                                for bottom in range(3):
                                    ty, by = y0 - top, y1 + bottom
                                    if yy not in (ty, by):
                                        continue
                                    add("R", lx, ty, rx - lx + 1, by - ty + 1)

        # Two parallel edges also determine a rectangle, including when both
        # perpendicular edges are obscured at their endpoints.
        for i, (a, b) in enumerate(segs["H"]):
            for u, v in segs["H"][:i]:
                if abs(a[0] - u[0]) > 2 or abs(b[0] - v[0]) > 2:
                    continue
                y0, y1 = sorted((a[1], u[1]))
                if y1 - y0 < 3:
                    continue
                lx, rx = min(a[0], u[0]), max(b[0], v[0])
                for dl in range(2):
                    for dr in range(2):
                        add("R", lx - dl, y0,
                            rx - lx + dl + dr + 1, y1 - y0 + 1)

        for i, (a, b) in enumerate(segs["V"]):
            for u, v in segs["V"][:i]:
                if abs(a[1] - u[1]) > 2 or abs(b[1] - v[1]) > 2:
                    continue
                x0, x1 = sorted((a[0], u[0]))
                if x1 - x0 < 3:
                    continue
                ty, by = min(a[1], u[1]), max(b[1], v[1])
                for dt in range(2):
                    for db in range(2):
                        add("R", x0, ty - dt,
                            x1 - x0 + 1, by - ty + dt + db + 1)

        for a, b in segs["D"]:
            d = a[1] - a[0]
            for u, v in segs["E"]:
                s = u[0] + u[1]
                if (s - d) % 2:
                    continue
                cx, cy = (s - d) // 2, (s + d) // 2
                if not (a[0] <= cx <= b[0] and u[0] <= cx <= v[0]):
                    continue
                r = max(abs(p[0] - cx) for p in (a, b, u, v))
                if r >= 3:
                    add("X", cx - r, cy - r, 2 * r + 1, 2 * r + 1)

        for orient in ("D", "E"):
            for a, b in segs[orient]:
                r = b[0] - a[0]
                if r >= 3:
                    for cx, cy in ((a[0], b[1]), (b[0], a[1])):
                        add("O", cx - r, cy - r, 2 * r + 1, 2 * r + 1)
                x0, x1 = a[0], b[0]
                y0, y1 = min(a[1], b[1]), max(a[1], b[1])
                if x1 - x0 == y1 - y0:
                    add(orient, x0, y0, x1 - x0 + 1, y1 - y0 + 1)

        for a, b in segs["H"]:
            x0, y = a
            x1 = b[0]
            for zx, zy in zeros:
                if zy == y and x0 <= zx <= x1:
                    x0, x1 = extend(x0, x1, zx, 63)
            add("C", x0, y, x1 - x0 + 1, 1, 0, 0)

        for a, b in segs["V"]:
            x, y0 = a
            y1 = b[1]
            for zx, zy in zeros:
                if zx == x and y0 <= zy <= y1:
                    y0, y1 = extend(y0, y1, zy, 62)
            add("C", x, y0, 1, y1 - y0 + 1, 0, 0)

        for g, x, y in candidates:
            evidence = set()
            missing = visible = 0
            for dx, dy in shape_pixels(g, True):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < 64 and 0 <= yy < 63):
                    continue
                visible += 1
                c = int(obs[yy, xx])
                if c == color:
                    if own[yy, xx]:
                        evidence.add(yy * 64 + xx)
                elif c == 0 or foreign[yy, xx]:
                    pass
                else:
                    missing += 1
            if missing > max(1, visible // 55):
                continue
            if len(evidence) < 7 or len(evidence) < visible * 0.40:
                continue
            cx, cy = x + g[1] // 2, y + g[2] // 2
            bonus = int(g[0] != "C" or min(g[1:3]) > 1)
            if 0 <= cy < 63 and obs[cy, cx] == 0:
                evidence.add(cy * 64 + cx)
                bonus += 3
            all_candidates.append(
                (g, x, y, color, evidence, missing, bonus)
            )

    chosen = []
    covered = set()
    while all_candidates and len(chosen) < 8:
        best_index = None
        best_score = 6
        for i, (g, x, y, c, ev, miss, bonus) in enumerate(all_candidates):
            gain = len(ev - covered)
            if gain < 0.55 * len(ev):
                continue
            score = gain - 4 * miss + bonus
            if score > best_score:
                best_score, best_index = score, i
        if best_index is None:
            break
        g, x, y, c, ev, _, _ = all_candidates.pop(best_index)
        chosen.append((g, pack(x, y), c))
        covered.update(ev)
    return chosen


class World:
    def __init__(self, stations, obstacles):
        self.stations = stations
        self.obstacles = obstacles
        self.geometries = []
        self.ids = {}
        self.cache = []
        self.targets = []
        self.color_masks = [0] * 16

    def gid(self, g):
        if g not in self.ids:
            self.ids[g] = len(self.geometries)
            self.geometries.append(g)
            self.cache.append(None)
        return self.ids[g]

    def set_targets(self, targets):
        self.targets = targets
        self.color_masks = [0] * 16
        for i, (_, _, c) in enumerate(targets):
            self.color_masks[c] |= 1 << i
        for d in self.cache:
            if d is not None:
                self.goal_data(d)

    def goal_data(self, d):
        g = d["g"]
        pixels = shape_pixels(g, True)
        covers = {}
        centers = set()
        for i, (tx, ty, _) in enumerate(self.targets):
            bit = 1 << i
            for dx, dy in pixels:
                p = pack(tx - dx, ty - dy)
                covers[p] = covers.get(p, 0) | bit
            centers.add(pack(tx - g[1] // 2, ty - g[2] // 2))
        d["covers"] = covers
        d["centers"] = centers

    @staticmethod
    def configuration_space(global_pixels, offsets):
        if not global_pixels or not offsets:
            return set()
        a = np.array(list(global_pixels), dtype=np.int64)
        b = np.array(list(offsets), dtype=np.int64)
        diff = a[:, None, :] - b[None, :, :]
        codes = diff[:, :, 0] * 256 + diff[:, :, 1] + 96
        return set(codes.ravel().tolist())

    def data(self, gid):
        d = self.cache[gid]
        if d is not None:
            return d
        g = self.geometries[gid]
        normal = shape_pixels(g)
        selected = normal | {(g[1] // 2, g[2] // 2)}
        obstacle_maps = [
            self.configuration_space(o[4], selected) for o in self.obstacles
        ]
        dye_maps = []
        for x, y, w, h, _ in self.stations:
            region = {(xx, yy) for xx in range(x, x + w)
                      for yy in range(y, y + h)}
            dye_maps.append(self.configuration_space(region, normal))
        d = {
            "g": g, "normal": normal,
            "obstacles": obstacle_maps, "dyes": dye_maps, "moves": {}
        }
        self.cache[gid] = d
        self.goal_data(d)
        return d

    def moves(self, gid, position):
        d = self.data(gid)
        if position in d["moves"]:
            return d["moves"][position]
        x0, y0 = unpack(position)
        original = self.geometries[gid]
        result = []

        for dx, dy in DIRS:
            x, y = x0 + dx, y0 + dy
            g, ng = original, gid
            kind, w, h, vx, hy = g
            if not (0 <= x + w // 2 < 64 and 0 <= y + h // 2 < 64):
                result.append((gid, position, ()))
                continue
            early = False

            for oi, obstacle in enumerate(self.obstacles):
                if pack(x, y) not in self.data(ng)["obstacles"][oi]:
                    continue
                ox, oy, ow, oh, _ = obstacle
                kind, w, h, vx, hy = g

                if kind in ("R", "O"):
                    if dx:
                        if w > 6:
                            nw, nh = w - 3, h + 3
                            y += h // 2 - nh // 2
                            y = int(round(y / 3.0) * 3)
                            if dx < 0:
                                x -= dx
                            g = ("R", nw, nh, 0, 0)
                            ng = self.gid(g)
                        else:
                            x, y = x - dx, y - dy
                    else:
                        if h > 6:
                            nw, nh = w + 3, h - 3
                            x += w // 2 - nw // 2
                            x = int(round(x / 3.0) * 3)
                            if dy < 0:
                                y -= dy
                            g = ("R", nw, nh, 0, 0)
                            ng = self.gid(g)
                        else:
                            x, y = x - dx, y - dy
                    early = True
                    break

                if kind != "C":
                    continue

                in_v = ox <= x + vx < ox + ow
                in_h = oy <= y + hy < oy + oh
                if dx:
                    toward = -3 if dx > 0 else 3
                    away = -toward
                    can_toward = vx > 0 if dx > 0 else vx < w - 2
                    can_away = vx < w - 2 if dx > 0 else vx > 0
                    if in_v and in_h:
                        x -= dx
                    elif in_v:
                        if can_toward:
                            vx = (vx + toward) % w
                        else:
                            x -= dx
                    elif in_h:
                        x -= dx
                        if can_away:
                            vx = (vx + away) % w
                else:
                    toward = -3 if dy > 0 else 3
                    away = -toward
                    can_toward = hy > 0 if dy > 0 else hy < h - 2
                    can_away = hy < h - 2 if dy > 0 else hy > 0
                    if in_v and in_h:
                        y -= dy
                    elif in_h:
                        if can_toward:
                            hy = (hy + toward) % h
                        else:
                            y -= dy
                    elif in_v:
                        y -= dy
                        if can_away:
                            hy = (hy + away) % h
                if h == 1:
                    vx = 0
                if w == 1:
                    hy = 0
                g = ("C", w, h, vx, hy)
                ng = self.gid(g)

            p = pack(x, y)
            hits = ()
            if not early:
                hits = tuple(i for i, m in enumerate(self.data(ng)["dyes"])
                             if p in m)
            result.append((ng, p, hits))

        d["moves"][position] = tuple(result)
        return d["moves"][position]

    def successor(self, state, action, cautious=False):
        gid, pos, color = state
        ng, p, hits = self.moves(gid, pos)[action - 1]
        colors = [self.stations[i][4] for i in hits
                  if self.stations[i][4] != color]
        if cautious and len(set(colors)) > 1:
            return None
        if colors:
            color = colors[0]
        return ng, p, color

    def center(self, state):
        g = self.geometries[state[0]]
        x, y = unpack(state[1])
        return x + g[1] // 2, y + g[2] // 2

    def occupies(self, state, tx, ty, selected=False):
        x, y = unpack(state[1])
        if (tx - x, ty - y) in self.data(state[0])["normal"]:
            return True
        return selected and self.center(state) == (tx, ty)


class Search:
    def __init__(self, world, state):
        self.world = world
        self.nodes = [(state[0], state[1], state[2], -1, 0, 0)]
        self.seen = {state: 0}
        self.head = 0
        self.best = {}
        self.reachable = 0
        self.stopped = False
        self.consider(0)

    def consider(self, index):
        g, p, c, _, _, _ = self.nodes[index]
        d = self.world.data(g)
        if p in d["centers"]:
            return
        mask = d["covers"].get(p, 0)
        if mask & ~self.world.color_masks[c]:
            return
        if mask not in self.best:
            self.best[mask] = index
            self.reachable |= mask

    def advance(self, depth, deadline):
        if self.stopped:
            return True
        steps = 0
        while self.head < len(self.nodes):
            node = self.nodes[self.head]
            if node[5] >= depth:
                return True
            if steps % 64 == 0 and time.monotonic() >= deadline:
                return False
            parent = self.head
            self.head += 1
            steps += 1
            g, p, c, _, _, dist = node
            for action, (ng, np_, hits) in enumerate(self.world.moves(g, p), 1):
                colors = {self.world.stations[i][4] for i in hits
                          if self.world.stations[i][4] != c}
                if len(colors) > 1:
                    continue
                nc = next(iter(colors)) if colors else c
                state = ng, np_, nc
                if state in self.seen:
                    continue
                index = len(self.nodes)
                self.seen[state] = index
                self.nodes.append((ng, np_, nc, parent, action, dist + 1))
                self.consider(index)
            if len(self.nodes) >= 210000:
                self.stopped = True
                return True
        self.stopped = True
        return True

    def path(self, index):
        actions = []
        while self.nodes[index][3] >= 0:
            n = self.nodes[index]
            actions.append(n[4])
            index = n[3]
        actions.reverse()
        return actions


def combine(searches, full):
    reachable = 0
    for s in searches:
        if not s.best:
            return None
        reachable |= s.reachable
    if reachable != full:
        return None

    options = []
    for s in searches:
        opts = [(mask, s.nodes[index][5], index)
                for mask, index in s.best.items()]
        opts.sort(key=lambda z: (z[1], -z[0].bit_count()))
        kept = []
        for mask, dist, index in opts:
            if any((mask | m) == m and d <= dist for m, d, _ in kept):
                continue
            kept.append((mask, dist, index))
        options.append(kept)

    dp = {0: (0, ())}
    for opts in options:
        nxt = {}
        for covered, (cost, choices) in dp.items():
            for mask, dist, index in opts:
                nm, nc = covered | mask, cost + dist
                if nm not in nxt or nc < nxt[nm][0]:
                    nxt[nm] = nc, choices + (index,)
        dp = nxt
    if full not in dp:
        return None
    return [s.path(i) for s, i in zip(searches, dp[full][1])]


def discard_plan(memory):
    memory["searches"] = None
    memory["plans"] = None
    memory["depth"] = 8
    memory["verification"] = 0
    memory["idle_switches"] = 0


def initialize(obs, memory, preserve=False):
    old_targets = dict(memory.get("targets", {})) if preserve else {}
    xy = detect_targets(obs)
    stations, station_mask = detect_stations(obs)
    obstacles = detect_obstacles(obs)
    parsed = detect_shapes(obs, xy, station_mask)
    world = World(stations, obstacles)
    shapes = [(world.gid(g), p, c) for g, p, c in parsed]

    targets = old_targets if old_targets else {p: None for p in xy}
    memory.update({
        "world": world,
        "shapes": shapes,
        "targets": targets,
        "station_mask": station_mask,
        "active": None,
        "next_shape": {},
        "switch_from": None,
        "last_moved": None,
        "reveal_steps": 0,
        "rebuild": False,
    })
    discard_plan(memory)


def observe(obs, memory):
    world = memory["world"]
    shapes = memory["shapes"]
    zeros = [(int(x), int(y)) for y, x in zip(*np.where(obs == 0))]
    detected = None
    for z in zeros:
        matches = [i for i, s in enumerate(shapes) if world.center(s) == z]
        if matches:
            detected = matches[-1]
            break

    if detected is not None:
        old = memory.get("switch_from")
        if old is not None and old != detected:
            memory["next_shape"][old] = detected
        memory["active"] = detected
    elif zeros:
        # A selection marker without a matching model is evidence of a
        # missing or incorrectly bounded shape, even if no movement occurred.
        memory["active"] = None
        memory["rebuild"] = True
    memory["switch_from"] = None

    changed = False
    moved = memory.get("last_moved")
    if moved is not None and moved < len(shapes):
        gid, pos, color = shapes[moved]
        x, y = unpack(pos)
        hist = [0] * 16
        missing = tested = 0
        center = world.center(shapes[moved])
        for dx, dy in world.data(gid)["normal"]:
            xx, yy = x + dx, y + dy
            if not (0 <= xx < 64 and 0 <= yy < 63):
                continue
            if (xx, yy) == center:
                continue
            c = int(obs[yy, xx])
            tested += 1
            if c in (1, 2, 4, 5):
                missing += 1
            if not memory["station_mask"][yy, xx] and 6 <= c <= 14:
                hist[c] += 1
        nc = max(range(6, 15), key=lambda c: hist[c])
        if nc != color and hist[nc] >= 5 and hist[nc] > 2 * hist[color]:
            shapes[moved] = gid, pos, nc
            changed = True
        if tested >= 10 and missing > max(4, tested // 3):
            memory["rebuild"] = True
    memory["last_moved"] = None

    active = memory.get("active")
    for (tx, ty), color in list(memory["targets"].items()):
        if color is not None:
            continue
        covering = any(world.occupies(s, tx, ty, i == active)
                       for i, s in enumerate(shapes))
        if not covering:
            value = int(obs[ty, tx])
            if 6 <= value <= 14:
                memory["targets"][(tx, ty)] = value
                changed = True
            elif value in (4, 5):
                del memory["targets"][(tx, ty)]
                changed = True

    if changed:
        discard_plan(memory)


def emit(memory, action, available):
    name = "ACTION" + str(action)
    if name not in available:
        candidates = [a for a in available if a != "RESET"]
        return (candidates[0] if candidates else available[0]), None, None
    active = memory.get("active")
    if action == 5:
        memory["switch_from"] = active
        memory["active"] = memory["next_shape"].get(active)
        plans = memory.get("plans")
        if plans is not None and any(plans):
            memory["idle_switches"] = memory.get("idle_switches", 0) + 1
            if memory["idle_switches"] > 2 * len(memory["shapes"]) + 2:
                memory["rebuild"] = True
    elif active is not None and active < len(memory["shapes"]):
        state = memory["shapes"][active]
        ns = memory["world"].successor(state, action)
        if ns is not None:
            memory["shapes"][active] = ns
        memory["last_moved"] = active
        memory["idle_switches"] = 0
    return name, None, None


def reveal(memory):
    world = memory["world"]
    shapes = memory["shapes"]
    active = memory.get("active")
    unknown = [p for p, c in memory["targets"].items() if c is None]
    covering = set()
    for tx, ty in unknown:
        for i, s in enumerate(shapes):
            if world.occupies(s, tx, ty, i == active):
                covering.add(i)

    if active is None or active not in covering:
        return 5

    state = shapes[active]
    best = None
    for action in range(1, 5):
        ns = world.successor(state, action)
        if ns is None or ns == state:
            continue
        cleared = sum(
            world.occupies(state, tx, ty, True)
            and not world.occupies(ns, tx, ty, True)
            for tx, ty in unknown
        )
        cx, cy = world.center(ns)
        distance = sum(abs(cx - tx) + abs(cy - ty) for tx, ty in unknown)
        score = (cleared * 1000 + distance * .01
                 - 2 * (ns[0] != state[0])
                 - .2 * (ns[2] != state[2]))
        if best is None or score > best[0]:
            best = score, action
    return best[1] if best else 5


def execute_plan(memory, available):
    plans = memory["plans"]
    active = memory.get("active")
    if active is not None and active < len(plans) and plans[active]:
        return emit(memory, plans[active].popleft(), available)
    return emit(memory, 5, available)


def act(obs, available, memory):
    obs = np.asarray(obs)
    deadline = time.monotonic() + 8.3

    if "world" not in memory:
        initialize(obs, memory)
        memory["rebuilds"] = 0
        memory["calls"] = 0

    memory["calls"] += 1
    if memory.get("rebuild"):
        initialize(obs, memory, preserve=True)
        memory["rebuilds"] = memory.get("rebuilds", 0) + 1

    if not memory["shapes"]:
        initialize(obs, memory, preserve=True)
        if not memory["shapes"]:
            return emit(memory, 5, available)

    observe(obs, memory)
    if memory.get("rebuild") and memory.get("rebuilds", 0) < 7:
        initialize(obs, memory, preserve=True)
        memory["rebuilds"] = memory.get("rebuilds", 0) + 1
        observe(obs, memory)
        memory["rebuild"] = False

    unknown = [p for p, c in memory["targets"].items() if c is None]
    if unknown:
        memory["reveal_steps"] += 1
        if memory["reveal_steps"] > 14:
            for x, y in unknown:
                value = int(obs[y, x])
                if 6 <= value <= 14:
                    memory["targets"][(x, y)] = value
                else:
                    memory["targets"].pop((x, y), None)
            discard_plan(memory)
        else:
            return emit(memory, reveal(memory), available)

    if memory["plans"] is not None:
        if any(memory["plans"]):
            return execute_plan(memory, available)
        memory["verification"] += 1
        if memory["verification"] <= len(memory["shapes"]):
            return emit(memory, 5, available)
        for p in memory["targets"]:
            memory["targets"][p] = None
        memory["reveal_steps"] = 0
        discard_plan(memory)
        return emit(memory, reveal(memory), available)

    world = memory["world"]
    if memory["searches"] is None:
        targets = [(x, y, c) for (x, y), c in memory["targets"].items()
                   if c is not None]
        world.set_targets(targets)
        memory["searches"] = [Search(world, state)
                              for state in memory["shapes"]]
        memory["depth"] = 8

    searches = memory["searches"]
    full = (1 << len(world.targets)) - 1

    while time.monotonic() < deadline:
        paths = combine(searches, full)
        if paths is not None:
            memory["plans"] = [deque(p) for p in paths]
            memory["searches"] = None
            memory["verification"] = 0
            return execute_plan(memory, available)

        complete = True
        for search in searches:
            if not search.advance(memory["depth"], deadline):
                complete = False
                break
        if not complete:
            break

        paths = combine(searches, full)
        if paths is not None:
            memory["plans"] = [deque(p) for p in paths]
            memory["searches"] = None
            memory["verification"] = 0
            return execute_plan(memory, available)

        memory["depth"] += 4
        if all(s.stopped for s in searches) or memory["depth"] > 108:
            active = memory.get("active")
            if active is not None:
                state = memory["shapes"][active]
                for action in (1, 4, 2, 3):
                    ns = world.successor(state, action)
                    if ns is not None and ns != state:
                        memory["rebuild"] = True
                        discard_plan(memory)
                        return emit(memory, action, available)
            discard_plan(memory)
            memory["rebuild"] = True
            return emit(memory, 5, available)

    return emit(memory, 5, available)

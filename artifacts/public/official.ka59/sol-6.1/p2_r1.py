import numpy as np
import collections
import heapq
import itertools
import math
import time

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
INF = 10000


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(int(x), int(y))]
        component = []
        while stack:
            xx, yy = stack.pop()
            component.append((xx, yy))
            for dx, dy in DIRS:
                nx, ny = xx + dx, yy + dy
                if (0 <= nx < w and 0 <= ny < h
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(component)
    return result


def _yellow_boxes(a):
    h, w = a.shape
    candidates = []
    for y, x in zip(*np.nonzero((a == 14) | (a == 0))):
        y, x = int(y), int(x)
        if y >= h - 1:
            continue
        for ww, hh in ((3, 3), (3, 6), (6, 3), (6, 6)):
            if x + ww > w or y + hh >= h:
                continue
            q = a[y:y + hh, x:x + ww]
            cy = slice(1, 2) if hh == 3 else slice(2, 4)
            cx = slice(1, 2) if ww == 3 else slice(2, 4)
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = border[-1, :] = True
            border[:, 0] = border[:, -1] = True
            center = np.zeros((hh, ww), dtype=bool)
            center[cy, cx] = True
            middle = ~(border | center)
            if not np.all((q[border] == 14) | (q[border] == 0)):
                continue
            if not np.all(q[middle] == 14):
                continue
            if not np.all(np.isin(q[center], (0, 4, 5))):
                continue
            candidates.append((ww * hh, x, y, ww, hh,
                               bool(np.all(q[center] == 0))))
    candidates.sort(reverse=True)
    occupied = np.zeros_like(a, dtype=bool)
    boxes = []
    for _, x, y, ww, hh, selected in candidates:
        if occupied[y:y + hh, x:x + ww].any():
            continue
        occupied[y:y + hh, x:x + ww] = True
        boxes.append(dict(kind="yellow", x=x, y=y, w=ww, h=hh,
                          selected=selected, rot=0, clock=0))
    boxes.sort(key=lambda b: (b["y"], b["x"], b["w"], b["h"]))
    return boxes


def _parse(a):
    boxes = _yellow_boxes(a)
    if not boxes:
        return None
    entities = []
    crosses = _components(a == 11)
    for c in crosses:
        if len(c) < 9:
            continue
        xs, ys = zip(*c)
        x, y = min(xs), min(ys)
        ww, hh = max(xs) - x + 1, max(ys) - y + 1
        if ww == 9 and hh == 9:
            entities.append(dict(kind="cross", x=x, y=y, w=9, h=9,
                                 selected=False, rot=0, clock=0))

    for c in _components((a == 13) | (a == 12)):
        xs, ys = zip(*c)
        x0, y0 = min(xs), min(ys)
        ww, hh = max(xs) - x0 + 1, max(ys) - y0 + 1
        if ww % 3 or hh % 3 or ww < 3 or hh < 3:
            continue
        tiles = [(x0, y0, ww, hh)]
        if ww > 9 and hh in (3, 6) and ww % hh == 0:
            tiles = [(x0 + k, y0, hh, hh) for k in range(0, ww, hh)]
        for x, y, bw, bh in tiles:
            q = a[y:y + bh, x:x + bw]
            if not np.all((q == 13) | (q == 12)):
                continue
            red = q == 12
            rot = 0
            clock = 0
            if red.any() and not red.all():
                top = bottom = left = right = 0
                while top < bh and red[top, :].all():
                    top += 1
                while bottom < bh and red[bh - 1 - bottom, :].all():
                    bottom += 1
                while left < bw and red[:, left].all():
                    left += 1
                while right < bw and red[:, bw - 1 - right].all():
                    right += 1
                clock, rot = max((top, 0), (bottom, 180),
                                 (left, 90), (right, 270))
            else:
                scores = []
                for r, xx, yy, sw, sh in (
                    (0, x, y - 3, bw, 3),
                    (180, x, y + bh, bw, 3),
                    (90, x + bw, y, 3, bh),
                    (270, x - 3, y, 3, bh),
                ):
                    xa, xb = max(0, xx), min(a.shape[1], xx + sw)
                    ya, yb = max(0, yy), min(a.shape[0] - 1, yy + sh)
                    score = 0
                    if xa < xb and ya < yb:
                        score = int(np.sum(a[ya:yb, xa:xb] == 2))
                    scores.append((score, r))
                rot = max(scores, key=lambda z: z[0])[1]
            entities.append(dict(kind="bomb", x=x, y=y, w=bw, h=bh,
                                 selected=False, rot=rot, clock=clock))

    entities.extend(boxes)
    bx, by = boxes[0]["x"] % 3, boxes[0]["y"] % 3
    cover = np.zeros_like(a, dtype=bool)
    for e in entities:
        x, y, w, h = e["x"], e["y"], e["w"], e["h"]
        if e["kind"] == "cross":
            cover[y:y + 9, x + 3:x + 6] = True
            cover[y + 3:y + 6, x:x + 9] = True
        else:
            cover[y:y + h, x:x + w] = True

    cross_border = []
    for yy in range(11):
        for xx in range(11):
            if not ((3 <= xx <= 7) or (3 <= yy <= 7)):
                continue
            if any(not (0 <= xx + dx < 11 and 0 <= yy + dy < 11
                        and ((3 <= xx + dx <= 7)
                             or (3 <= yy + dy <= 7)))
                   for dx, dy in DIRS):
                cross_border.append((xx, yy))

    templates = []
    for w, h in ((5, 5), (5, 8), (8, 5), (8, 8)):
        points = [(xx, yy) for yy in range(h) for xx in range(w)
                  if xx in (0, w - 1) or yy in (0, h - 1)]
        templates.append(("yellow", w, h, points))
    if crosses:
        templates.append(("cross", 11, 11, cross_border))

    candidates = []
    for y in range(a.shape[0] - 4):
        if (y + 1 - by) % 3:
            continue
        for x in range(a.shape[1] - 4):
            if (x + 1 - bx) % 3:
                continue
            for kind, w, h, points in templates:
                if x + w > a.shape[1] or y + h >= a.shape[0]:
                    continue
                visible = []
                valid = True
                for xx, yy in points:
                    px, py = x + xx, y + yy
                    if a[py, px] == 4:
                        visible.append((px, py))
                    elif not cover[py, px]:
                        valid = False
                        break
                if valid and len(visible) >= max(4, len(points) // 2):
                    candidates.append((len(visible), kind, x, y, w, h,
                                       visible))
    candidates.sort(key=lambda z: z[0], reverse=True)
    used = set()
    goals = []
    for count, kind, x, y, w, h, visible in candidates:
        fresh = sum(p not in used for p in visible)
        if fresh < max(4, count // 2):
            continue
        used.update(visible)
        goals.append(dict(kind=kind, x=x + 1, y=y + 1,
                          w=w - 2, h=h - 2))
    return entities, goals, bx, by


class _Model:
    def __init__(self, a, parsed):
        entities, goals, self.bx, self.by = parsed
        self.entities = entities
        self.n = len(entities)
        self.W = (a.shape[1] - self.bx + 2) // 3
        self.H = (a.shape[0] - self.by + 2) // 3
        self.P = self.W * self.H
        self.yellow = [i for i, e in enumerate(entities)
                       if e["kind"] == "yellow"]
        self.bombs = [i for i, e in enumerate(entities)
                      if e["kind"] == "bomb"]
        self.nb = len(self.bombs)
        self.wall_signature = (a[:-1] == 2).tobytes()
        self.shapes = {}
        self.types = []
        for e in entities:
            key = (e["kind"], e["w"] // 3, e["h"] // 3)
            if key not in self.shapes:
                if key[0] == "cross":
                    cells = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
                else:
                    cells = tuple((x, y) for y in range(key[2])
                                  for x in range(key[1]))
                self.shapes[key] = cells
            self.types.append(key)

        wall = gulch = 0
        for gy in range(self.H):
            for gx in range(self.W):
                x, y = self.bx + 3 * gx, self.by + 3 * gy
                bit = 1 << (gy * self.W + gx)
                if x + 3 > a.shape[1] or y + 3 > a.shape[0] - 1:
                    wall |= bit
                    continue
                q = a[y:y + 3, x:x + 3]
                if np.any(q == 2):
                    wall |= bit
                if np.any(q == 15):
                    gulch |= bit
        self.wall = wall
        self.gulch = gulch
        self.masks = {}
        self.bad = {}
        self.bad_normal = {}
        self.normal = {}
        self.nextpos = [[-1] * 4 for _ in range(self.P)]
        for p in range(self.P):
            x, y = p % self.W, p // self.W
            for d, (dx, dy) in enumerate(DIRS):
                xx, yy = x + dx, y + dy
                if 0 <= xx < self.W and 0 <= yy < self.H:
                    self.nextpos[p][d] = yy * self.W + xx

        small = ("yellow", 1, 1)
        if small not in self.shapes:
            self.shapes[small] = ((0, 0),)
        for key, cells in self.shapes.items():
            masks = [0] * self.P
            bad = [True] * self.P
            normal_bad = [True] * self.P
            for p in range(self.P):
                x, y = p % self.W, p // self.W
                if any(x + xx >= self.W or y + yy >= self.H
                       for xx, yy in cells):
                    continue
                m = 0
                for xx, yy in cells:
                    m |= 1 << ((y + yy) * self.W + x + xx)
                masks[p] = m
                bad[p] = bool(m & wall)
                normal_bad[p] = bool(m & (wall | gulch))
            self.masks[key] = masks
            self.bad[key] = bad
            self.bad_normal[key] = normal_bad
            self.normal[key] = [
                tuple(q if q >= 0 and not normal_bad[q] else -1
                      for q in self.nextpos[p])
                for p in range(self.P)
            ]

        self.start_cache = {}
        self.blast_cache = {}
        self.hcache = {}
        self.goal_groups = {}
        for g in goals:
            key = (g["kind"], g["w"] // 3, g["h"] // 3)
            if key not in self.masks:
                continue
            p = self.encode(g["x"], g["y"])
            if 0 <= p < self.P:
                self.goal_groups.setdefault(key, []).append(p)
        self.goals = goals
        self.distances = {}
        self._make_distances()

        positions = [self.encode(e["x"], e["y"]) for e in entities]
        selected = next((i for i in self.yellow
                         if entities[i]["selected"]), self.yellow[0])
        clocks = [min(entities[i]["clock"], entities[i]["h"] - 1)
                  for i in self.bombs]
        self.initial = tuple(positions + [selected] + clocks + [0] * self.n)

        self.walk = None
        self.region = None
        self.routes = []
        self.route_hops = None
        self.transport_cache = {}
        if len(self.yellow) == 1 and self.bombs:
            self._prepare_transport()

    def encode(self, x, y):
        return ((y - self.by) // 3) * self.W + (x - self.bx) // 3

    def xy(self, p):
        return self.bx + 3 * (p % self.W), self.by + 3 * (p // self.W)

    def manhattan(self, p, q):
        return abs(p % self.W - q % self.W) + abs(p // self.W - q // self.W)

    def starts(self, target_type, target_pos, pusher_type, d):
        cache_key = (target_type, target_pos, pusher_type, d)
        if cache_key in self.start_cache:
            return self.start_cache[cache_key]
        tx, ty = target_pos % self.W, target_pos // self.W
        dx, dy = DIRS[d]
        candidates = set()
        for ex, ey in self.shapes[target_type]:
            for px, py in self.shapes[pusher_type]:
                x, y = tx + ex - px - dx, ty + ey - py - dy
                if 0 <= x < self.W and 0 <= y < self.H:
                    candidates.add(y * self.W + x)
        target_mask = self.masks[target_type][target_pos]
        pm = self.masks[pusher_type]
        bad = self.bad_normal[pusher_type]
        result = []
        for p in candidates:
            q = self.nextpos[p][d]
            if (q >= 0 and not bad[p] and not bad[q]
                    and not (pm[p] & target_mask)
                    and (pm[q] & target_mask)):
                result.append(p)
        result.sort()
        self.start_cache[cache_key] = tuple(result)
        return self.start_cache[cache_key]

    def _make_distances(self):
        small = ("yellow", 1, 1)
        for key, goals in self.goal_groups.items():
            reverse = [[] for _ in range(self.P)]
            masks = self.masks[key]
            bad = self.bad[key]
            normal_bad = self.bad_normal[key]
            for p in range(self.P):
                if bad[p]:
                    continue
                for d in range(4):
                    q = self.nextpos[p][d]
                    if q >= 0 and not bad[q]:
                        if self.bombs or (key[0] == "yellow"
                                          and not normal_bad[q]):
                            reverse[q].append(p)
                    can_launch = (key[0] != "yellow"
                                  or len(self.yellow) > 1)
                    if not can_launch or not self.starts(key, p, small, d):
                        continue
                    q = self.launch_destination(key, p, d)
                    if q != p:
                        reverse[q].append(p)
            for goal in goals:
                dist = [100] * self.P
                dist[goal] = 0
                queue = collections.deque([goal])
                while queue:
                    q = queue.popleft()
                    nd = dist[q] + 1
                    for p in reverse[q]:
                        if nd < dist[p]:
                            dist[p] = nd
                            queue.append(p)
                self.distances[(key, goal)] = dist

    def launch_destination(self, key, p, d):
        q = p
        for step in range(self.W + self.H + 5):
            if step >= 5 and not (self.masks[key][q] & self.gulch):
                break
            nq = self.nextpos[q][d]
            if nq < 0 or self.bad[key][nq]:
                break
            q = nq
        return q

    def blast(self, p, bomb_index, stage):
        key = (p, bomb_index, stage)
        if key in self.blast_cache:
            return self.blast_cache[key]
        e = self.entities[bomb_index]
        x, y = p % self.W, p // self.W
        w, h = e["w"] // 3, e["h"] // 3
        rot = e["rot"]
        if rot == 0:
            cells = ((x + xx, y + h + yy)
                     for yy in range(stage) for xx in range(w))
            d = 1
        elif rot == 180:
            cells = ((x + xx, y - stage + yy)
                     for yy in range(stage) for xx in range(w))
            d = 0
        elif rot == 90:
            cells = ((x - stage + xx, y + yy)
                     for yy in range(h) for xx in range(stage))
            d = 2
        else:
            cells = ((x + w + xx, y + yy)
                     for yy in range(h) for xx in range(stage))
            d = 3
        mask = 0
        for xx, yy in cells:
            if 0 <= xx < self.W and 0 <= yy < self.H:
                mask |= 1 << (yy * self.W + xx)
        self.blast_cache[key] = (mask, d)
        return mask, d

    def _prepare_transport(self):
        yi = self.yellow[0]
        key = self.types[yi]
        neighbors = self.normal[key]
        bad = self.bad_normal[key]
        self.walk = [[INF] * self.P for _ in range(self.P)]
        self.region = [-1] * self.P
        nr = 0
        for start in range(self.P):
            if bad[start]:
                continue
            dist = self.walk[start]
            dist[start] = 0
            queue = collections.deque([start])
            while queue:
                p = queue.popleft()
                nd = dist[p] + 1
                for q in neighbors[p]:
                    if q >= 0 and dist[q] == INF:
                        dist[q] = nd
                        queue.append(q)
            if self.region[start] < 0:
                for p, value in enumerate(dist):
                    if value < INF:
                        self.region[p] = nr
                nr += 1
        hops = [[INF] * nr for _ in range(nr)]
        for r in range(nr):
            hops[r][r] = 0
        sw, sh = key[1], key[2]
        pm = self.masks[key]

        for bomb in self.bombs:
            bk = self.types[bomb]
            bw, bh = bk[1], bk[2]
            rot = self.entities[bomb]["rot"]
            for bp in range(self.P):
                if self.bad[bk][bp] or (self.masks[bk][bp] & self.gulch):
                    continue
                x, y = bp % self.W, bp // self.W
                if rot == 0:
                    xr = range(x - sw + 1, x + bw)
                    yr = range(y + bh - sh + 1, y + bh + 3)
                elif rot == 180:
                    xr = range(x - sw + 1, x + bw)
                    yr = range(y - 3 - sh + 1, y)
                elif rot == 90:
                    xr = range(x - 3 - sw + 1, x)
                    yr = range(y - sh + 1, y + bh)
                else:
                    xr = range(x + bw - sw + 1, x + bw + 3)
                    yr = range(y - sh + 1, y + bh)
                for yy in yr:
                    for xx in xr:
                        if not (0 <= xx < self.W and 0 <= yy < self.H):
                            continue
                        pad = yy * self.W + xx
                        if bad[pad] or pm[pad] & self.masks[bk][bp]:
                            continue
                        q = pad
                        direction = 0
                        for stage in (1, 2, 3):
                            bm, direction = self.blast(bp, bomb, stage)
                            if bm & pm[q]:
                                nq = self.nextpos[q][direction]
                                if nq >= 0 and not self.bad[key][nq]:
                                    q = nq
                        for _ in range(self.W + self.H):
                            if not (pm[q] & self.gulch):
                                break
                            nq = self.nextpos[q][direction]
                            if nq < 0 or self.bad[key][nq]:
                                break
                            q = nq
                        r1, r2 = self.region[pad], self.region[q]
                        if r1 >= 0 and r2 >= 0 and r1 != r2:
                            self.routes.append((bomb, bp, pad, q, r1, r2))
                            hops[r1][r2] = 1
        for k in range(nr):
            for i in range(nr):
                for j in range(nr):
                    hops[i][j] = min(hops[i][j], hops[i][k] + hops[k][j])
        self.route_hops = hops

    def _line_valid(self, key, p, q):
        x, y = p % self.W, p // self.W
        xx, yy = q % self.W, q // self.W
        if x != xx and y != yy:
            return False
        if x == xx:
            for row in range(min(y, yy), max(y, yy) + 1):
                if self.bad[key][row * self.W + x]:
                    return False
        else:
            for col in range(min(x, xx), max(x, xx) + 1):
                if self.bad[key][y * self.W + col]:
                    return False
        return True

    def transport_distance(self, state, start, target):
        if self.walk is None:
            return 6 * self.manhattan(start, target)
        direct = self.walk[start][target]
        if direct < INF:
            return direct
        r1, rt = self.region[start], self.region[target]
        if r1 < 0 or rt < 0:
            return 60 + self.manhattan(start, target)
        cache_key = (state[:self.n], start, target)
        if cache_key in self.transport_cache:
            return self.transport_cache[cache_key]
        best = 90 + self.manhattan(start, target)
        yi = self.yellow[0]
        yk = self.types[yi]
        for bomb, bp, pad, landing, source, destination in self.routes:
            if source != r1:
                continue
            hops = self.route_hops[destination][rt]
            if hops >= INF:
                continue
            current = state[bomb]
            if current == bp:
                cost = self.walk[start][pad]
                if cost >= INF:
                    continue
            else:
                bk = self.types[bomb]
                if not self._line_valid(bk, current, bp):
                    continue
                dx = bp % self.W - current % self.W
                dy = bp // self.W - current // self.W
                d = (3 if dx > 0 else 2) if dx else (1 if dy > 0 else 0)
                possible = self.starts(bk, current, yk, d)
                possible = [p for p in possible
                            if self.walk[start][p] < INF]
                if not possible:
                    continue
                distance = abs(dx) + abs(dy)
                shots = (distance + 4) // 5
                follow = max(0, distance - 5)
                ddx, ddy = DIRS[d]
                cost = INF
                for stance in possible:
                    sx, sy = stance % self.W, stance // self.W
                    lx, ly = sx + ddx * follow, sy + ddy * follow
                    after_walk = (abs(lx - pad % self.W)
                                  + abs(ly - pad // self.W))
                    value = (self.walk[start][stance] + shots
                             + follow + after_walk)
                    cost = min(cost, value)
            if destination == rt:
                finish = self.walk[landing][target]
            else:
                finish = 8 * hops + self.manhattan(landing, target)
            best = min(best, cost + 2 + finish)
        self.transport_cache[cache_key] = best
        return best

    def _single_piece_cost(self, state, i, goal):
        p = state[i]
        key = self.types[i]
        if p == goal:
            return 0
        yi = self.yellow[0]
        start = state[yi]
        if key[0] == "yellow":
            return self.transport_distance(state, start, goal)
        dist = self.distances[(key, goal)]
        best = 6 * dist[p] + 40
        for d in range(4):
            q = self.launch_destination(key, p, d)
            if q == p or dist[q] >= 100:
                continue
            stances = self.starts(key, p, self.types[yi], d)
            for stance in stances:
                value = (self.transport_distance(state, start, stance)
                         + 1 + 5 * dist[q])
                best = min(best, value)
        return best

    def heuristic(self, state):
        positions = state[:self.n]
        if positions in self.hcache:
            return self.hcache[positions]
        total = 0
        for key, goals in self.goal_groups.items():
            ids = [i for i, t in enumerate(self.types) if t == key]
            if len(ids) < len(goals):
                total += 600
                continue
            best = INF
            for assignment in itertools.permutations(ids, len(goals)):
                if self.walk is not None:
                    value = sum(self._single_piece_cost(state, i, g)
                                for i, g in zip(assignment, goals))
                else:
                    value = 6 * sum(
                        self.distances[(key, g)][positions[i]]
                        for i, g in zip(assignment, goals))
                best = min(best, value)
            total += best
        self.hcache[positions] = total
        return total

    def solved_count(self, state):
        result = 0
        for key, goals in self.goal_groups.items():
            places = [state[i] for i, t in enumerate(self.types) if t == key]
            result += sum(g in places for g in goals)
        return result

    def won(self, state):
        if not self.goal_groups:
            return False
        for key, goals in self.goal_groups.items():
            places = [state[i] for i, t in enumerate(self.types) if t == key]
            if any(g not in places for g in goals):
                return False
        return True

    def search_key(self, state):
        offset = self.n + 1 + self.nb
        directions = tuple(
            state[offset + i]
            if self.masks[self.types[i]][state[i]] & self.gulch else 0
            for i in range(self.n))
        return state[:offset] + directions

    def step(self, state, action):
        n = self.n
        if action >= 10:
            s = list(state)
            s[n] = action - 10
            return tuple(s)
        d = action - 1
        positions = list(state[:n])
        selected = state[n]
        clocks = list(state[n + 1:n + 1 + self.nb])
        last = list(state[n + 1 + self.nb:])
        masks = self.masks
        types = self.types

        def push(i, direction, depth=0):
            if depth > 2 * n + 4:
                return False
            old = positions[i]
            q = self.nextpos[old][direction]
            if q < 0 or self.bad[types[i]][q]:
                return False
            positions[i] = q
            m = masks[types[i]][q]
            hits = [j for j in range(n)
                    if j != i and (m & masks[types[j]][positions[j]])]
            for j in hits:
                if not push(j, direction, depth + 1):
                    positions[i] = old
                    return False
            return True

        p = positions[selected]
        q = self.nextpos[p][d]
        hits = []
        if q >= 0 and not self.bad_normal[types[selected]][q]:
            m = masks[types[selected]][q]
            hits = [i for i in range(n) if i != selected
                    and (m & masks[types[i]][positions[i]])]
            if not hits:
                positions[selected] = q
        if hits:
            for i in hits:
                last[i] = d + 1
            for frame in range(self.W + self.H + 10):
                done = []
                for i in hits:
                    if frame >= 5 and not (
                            masks[types[i]][positions[i]] & self.gulch):
                        done.append(True)
                    else:
                        done.append(not push(i, d))
                if all(done):
                    break

        exploding = []
        for k, i in enumerate(self.bombs):
            clocks[k] += 1
            if clocks[k] >= self.entities[i]["h"]:
                clocks[k] = 0
                exploding.append(i)
        if exploding:
            for stage in (1, 2, 3):
                for bomb in exploding:
                    bm, bd = self.blast(positions[bomb], bomb, stage)
                    for i in range(n):
                        if bm & masks[types[i]][positions[i]]:
                            push(i, bd)
                            last[i] = bd + 1
            for _ in range(self.W + self.H + 10):
                moved = False
                for i in range(n):
                    if (last[i] and
                            masks[types[i]][positions[i]] & self.gulch):
                        if push(i, last[i] - 1):
                            moved = True
                if not moved:
                    break
        return tuple(positions + [selected] + clocks + last)

    def bfs(self, state, i):
        occupied = 0
        for j in range(self.n):
            if j != i:
                occupied |= self.masks[self.types[j]][state[j]]
        start = state[i]
        parents = {start: (-1, -1)}
        distances = {start: 0}
        queue = collections.deque([start])
        neighbors = self.normal[self.types[i]]
        pm = self.masks[self.types[i]]
        while queue:
            p = queue.popleft()
            for d, q in enumerate(neighbors[p]):
                if q < 0 or q in parents or pm[q] & occupied:
                    continue
                parents[q] = (p, d + 1)
                distances[q] = distances[p] + 1
                queue.append(q)
        return parents, distances

    @staticmethod
    def path(parents, end):
        result = []
        p = end
        while parents[p][0] >= 0:
            p, action = parents[p]
            result.append(action)
        result.reverse()
        return result

    def run(self, state, actions):
        for action in actions:
            state = self.step(state, action)
            if self.won(state):
                break
        return state

    def wait(self, state):
        if not self.bombs:
            return state, []
        clocks = state[self.n + 1:self.n + 1 + self.nb]
        count = min(self.entities[i]["h"] - c
                    for i, c in zip(self.bombs, clocks))
        base = state[:self.n]
        selected = state[self.n]
        actions = []
        for _ in range(max(1, count)):
            options = []
            for a in range(1, 5):
                ns = self.step(state, a)
                same = ns[:self.n] == state[:self.n]
                displacement = self.manhattan(ns[selected], base[selected])
                score = (0 if same else 4) + 5 * displacement
                score += self.heuristic(ns)
                options.append((score, a, ns))
            _, a, state = min(options, key=lambda z: z[0])
            actions.append(a)
            if self.won(state):
                break
        return state, actions

    def macros(self, state, deadline):
        yielded = set()
        for i in self.yellow:
            if time.perf_counter() >= deadline:
                return
            parents, distances = self.bfs(state, i)
            click = [] if state[self.n] == i else [10 + i]
            prefixes = {}

            def prefix(p):
                if p not in prefixes:
                    actions = click + self.path(parents, p)
                    prefixes[p] = (self.run(state, actions), actions)
                return prefixes[p]

            for goal in self.goal_groups.get(self.types[i], ()):
                if goal in parents and state[i] != goal:
                    ns, actions = prefix(goal)
                    nk = self.search_key(ns)
                    if actions and ns != state and nk not in yielded:
                        yielded.add(nk)
                        yield ns, actions

            for j in range(self.n):
                if j == i:
                    continue
                for d in range(4):
                    starts = [p for p in self.starts(
                        self.types[j], state[j], self.types[i], d)
                              if p in parents]
                    starts.sort(key=lambda p: (distances[p], p))
                    for p in starts[:2]:
                        if time.perf_counter() >= deadline:
                            return
                        ps, actions = prefix(p)
                        ns = self.step(ps, d + 1)
                        nk = self.search_key(ns)
                        actions = actions + [d + 1]
                        if ns != state and nk not in yielded:
                            yielded.add(nk)
                            yield ns, actions

            if self.bombs:
                pads = set()
                for bomb in self.bombs:
                    blast_mask, _ = self.blast(state[bomb], bomb, 3)
                    possible = [
                        p for p in parents
                        if self.masks[self.types[i]][p] & blast_mask
                    ]
                    possible.sort(key=lambda p: (distances[p], p))
                    pads.update(possible[:10])
                pads.add(state[i])
                for p in sorted(pads, key=lambda p: (distances[p], p)):
                    if time.perf_counter() >= deadline:
                        return
                    ps, actions = prefix(p)
                    ns, waits = self.wait(ps)
                    nk = self.search_key(ns)
                    actions = actions + waits
                    if actions and ns != state and nk not in yielded:
                        yielded.add(nk)
                        yield ns, actions

    def search(self, initial, deadline, history):
        serial = itertools.count()
        ik = self.search_key(initial)
        h0 = self.heuristic(initial)
        heap = [(1.35 * h0, 0, next(serial), ik, initial)]
        records = {ik: (0, None, None)}
        best = ik
        best_rank = (-self.solved_count(initial), h0, 0)
        alternative = None
        alternative_rank = None
        final = None
        expanded = 0

        while heap and time.perf_counter() < deadline:
            _, g, _, sk, state = heapq.heappop(heap)
            if records[sk][0] != g:
                continue
            if self.won(state):
                final = sk
                break
            expanded += 1
            if expanded > 5000:
                break
            for ns, actions in self.macros(state, deadline):
                ng = g + len(actions)
                nk = self.search_key(ns)
                if ng >= records.get(nk, (INF,))[0]:
                    continue
                records[nk] = (ng, sk, actions)
                h = self.heuristic(ns)
                seen_penalty = 8 * history.get(ns[:self.n], 0)
                rank = (-self.solved_count(ns), h, ng + seen_penalty)
                if rank < best_rank:
                    best_rank, best = rank, nk
                arank = (ng + 1.35 * h + seen_penalty, h, ng)
                if alternative_rank is None or arank < alternative_rank:
                    alternative_rank, alternative = arank, nk
                if self.won(ns):
                    final = nk
                    break
                heapq.heappush(heap, (ng + 1.35 * h + seen_penalty,
                                     ng, next(serial), nk, ns))
            if final is not None:
                break

        target = final or (best if best != ik else alternative)
        if target is None:
            return []
        segments = []
        while target != ik:
            _, parent, actions = records[target]
            segments.append(actions)
            target = parent
        segments.reverse()
        return [a for segment in segments for a in segment]

    def matches(self, a, state, boxes):
        if (a[:-1] == 2).tobytes() != self.wall_signature:
            return False
        expected = sorted(
            (self.xy(state[i])[0], self.xy(state[i])[1],
             self.entities[i]["w"], self.entities[i]["h"],
             i == state[self.n])
            for i in self.yellow
        )
        actual = sorted((b["x"], b["y"], b["w"], b["h"], b["selected"])
                        for b in boxes)
        if expected != actual:
            return False
        for i, e in enumerate(self.entities):
            if e["kind"] == "yellow":
                continue
            x, y = self.xy(state[i])
            if (x < 0 or y < 0 or x + e["w"] > a.shape[1]
                    or y + e["h"] >= a.shape[0]):
                return False
            q = a[y:y + e["h"], x:x + e["w"]]
            if e["kind"] == "cross":
                if (np.sum(q[0:3, 3:6] == 11) < 6
                        or np.sum(q[3:6, :] == 11) < 18):
                    return False
            elif not np.all((q == 13) | (q == 12)):
                return False
        return True


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.5
    a = np.asarray(obs, dtype=np.int16)
    scale = memory.get("_scale", 1)
    model = memory.get("_model")

    if model is None:
        parsed = _parse(a)
        if parsed is None or not parsed[1]:
            reduced = a[1::2, 1::2]
            doubled = _parse(reduced)
            if doubled is not None and doubled[1]:
                scale = 2
                a = reduced
                parsed = doubled
            elif parsed is None:
                name = "RESET" if "RESET" in available else available[0]
                return name, None, None
        memory["_scale"] = scale
    elif scale == 2:
        a = a[1::2, 1::2]

    boxes = _yellow_boxes(a)
    state = memory.get("_state")
    valid = (model is not None and state is not None
             and model.matches(a, state, boxes))

    if not valid:
        parsed = _parse(a)
        if parsed is None:
            name = "RESET" if "RESET" in available else available[0]
            return name, None, None
        if model is not None and (
                (a[:-1] == 2).tobytes() == model.wall_signature):
            parsed = (parsed[0], model.goals, parsed[2], parsed[3])
        model = _Model(a, parsed)
        state = model.initial
        memory["_model"] = model
        memory["_state"] = state
        memory["_plan"] = []
        memory.setdefault("_history", {})

    plan = memory.get("_plan", [])
    if not plan:
        history = memory.setdefault("_history", {})
        positions = state[:model.n]
        history[positions] = history.get(positions, 0) + 1
        plan = model.search(state, deadline, history)

    if not plan:
        options = []
        for action in range(1, 5):
            if "ACTION" + str(action) not in available:
                continue
            ns = model.step(state, action)
            penalty = 6 * memory["_history"].get(ns[:model.n], 0)
            options.append((model.heuristic(ns) + penalty, action, ns))
        if not options:
            name = "RESET" if "RESET" in available else available[0]
            return name, None, None
        _, action, ns = min(options, key=lambda z: z[0])
        memory["_state"] = ns
        memory["_plan"] = []
        return "ACTION" + str(action), None, None

    action = plan.pop(0)
    memory["_plan"] = plan
    if action >= 10:
        i = action - 10
        x, y = model.xy(state[i])
        e = model.entities[i]
        col = x + e["w"] // 2
        row = y + e["h"] // 2
        if scale == 2:
            col = 2 * col + 1
            row = 2 * row + 1
        if "ACTION6" not in available:
            memory["_plan"] = []
            return available[0], None, None
        memory["_state"] = model.step(state, action)
        return "ACTION6", int(row), int(col)

    name = "ACTION" + str(action)
    if name not in available:
        memory["_plan"] = []
        return available[0], None, None
    memory["_state"] = model.step(state, action)
    return name, None, None

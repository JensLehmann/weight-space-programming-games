import numpy as np
import heapq
import time
from collections import deque


# Spell masks in the 3-by-3 input panel.
_SPELLS = {"T": 19, "R": 170, "F": 146}
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(y, x)]
        cells = []
        while q:
            a, b = q.pop()
            cells.append((a, b))
            for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                c, d = a + da, b + db
                if (0 <= c < h and 0 <= d < w and
                        mask[c, d] and not seen[c, d]):
                    seen[c, d] = True
                    q.append((c, d))
        result.append(cells)
    return result


def _player_and_goals(obs):
    template = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int8)
    goal_mask = np.zeros((64, 64), dtype=bool)
    goals = []
    nines = np.argwhere(obs == 9)
    for k in range(4):
        t = np.rot90(template, k)
        h, w = t.shape
        for yy, xx in nines:
            y, x = int(yy), int(xx)
            if y + h <= 64 and x + w <= 64:
                if np.array_equal(obs[y:y+h, x:x+w], t):
                    box = (x, y, w, h)
                    if box not in goals:
                        goals.append(box)
                        goal_mask[y:y+h, x:x+w] = True

    mask = ((obs == 9) | (obs == 10)) & ~goal_mask
    candidates = []
    for cells in _components(mask):
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        if w == h and w in (2, 4) and len(cells) == w * h:
            a = obs[y:y+h, x:x+w]
            n = w // 2
            direction = 4
            if np.all(a[:n] == 9) and np.all(a[n:] == 10):
                direction = 0
            elif np.all(a[:n] == 10) and np.all(a[n:] == 9):
                direction = 1
            elif np.all(a[:, :n] == 9) and np.all(a[:, n:] == 10):
                direction = 2
            elif np.all(a[:, :n] == 10) and np.all(a[:, n:] == 9):
                direction = 3
            candidates.append((y * 64 + x, int(w == 4), direction))

    if candidates:
        return candidates[0], goals, goal_mask

    # A fallback for a player touching another 9/10-colored object.
    for size in (4, 2):
        base = np.array([[10, 10], [9, 9]], dtype=np.int8)
        base = np.repeat(np.repeat(base, size // 2, axis=0),
                         size // 2, axis=1)
        for k, direction in ((0, 1), (1, 3), (2, 0), (3, 2)):
            t = np.rot90(base, k)
            for yy, xx in np.argwhere(mask):
                y, x = int(yy), int(xx)
                if y + size <= 64 and x + size <= 64:
                    if (not np.any(goal_mask[y:y+size, x:x+size]) and
                            np.array_equal(obs[y:y+size, x:x+size], t)):
                        return (y * 64 + x, int(size == 4), direction), goals, goal_mask
    return None, goals, goal_mask


def _panel_bits(obs):
    bits = 0
    for i in range(9):
        y = 50 + 5 * (i // 3)
        x = 25 + 5 * (i % 3)
        if int(obs[y, x]) in (6, 11, 14, 15):
            bits |= 1 << i
    return bits


def _ui_information(obs):
    rects = [(22, 47, 17, 17), (62, 0, 2, 64)]
    allowed = set()
    for cells in _components(obs == 3):
        ys = [a for a, b in cells]
        xs = [b for a, b in cells]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        if w == 10 and h == 10:
            a = obs[y:y+h, x:x+w]
            if (np.all(a[0] == 3) and np.all(a[-1] == 3) and
                    np.all(a[:, 0] == 3) and np.all(a[:, -1] == 3)):
                rects.append((x, y, w, h))
                if np.any(a == 6):
                    allowed.add("F")
                if np.any(a == 11):
                    allowed.add("T")
                if np.any(a == 15):
                    allowed.add("R")
    ui = np.zeros((64, 64), dtype=bool)
    for x, y, w, h in rects:
        ui[y:y+h, x:x+w] = True
    return ui, rects, allowed


def _counts(obs, rects):
    good = np.ones((64, 64), dtype=bool)
    for x, y, w, h in rects:
        good[y:y+h, x:x+w] = False
    return tuple(int(np.count_nonzero((obs == c) & good))
                 for c in (6, 13, 12))


def _click_path(start, target, allowed_masks):
    # Matching any other valid spell would cast it immediately.
    q = deque([start])
    prev = {start: None}
    last = None
    while q:
        b = q.popleft()
        for i in range(9):
            nb = b ^ (1 << i)
            if nb == target:
                prev_key = (b, i)
                path = [i]
                while prev[b] is not None:
                    b, j = prev[b]
                    path.append(j)
                path.reverse()
                return path
            if nb in allowed_masks or nb in prev:
                continue
            prev[nb] = (b, i)
            q.append(nb)
    return None


def _square_cover(mask, size, outside=True):
    a = mask.astype(np.int16)
    p = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (p[size:, size:] - p[:-size, size:]
            - p[size:, :-size] + p[:-size, :-size])
    out = np.full((64, 64), outside, dtype=bool)
    out[:65-size, :65-size] = sums > 0
    return out


class _World:
    def __init__(self, obs, player, goals, goal_mask, memory):
        self.obs = obs
        self.player = player
        self.goals = goals
        self.goal_mask = goal_mask
        self.ui, self.rects, self.allowed = _ui_information(obs)
        self.initial_bits = _panel_bits(obs)
        self.initial_counts = _counts(obs, self.rects)
        self.targets = []
        self.bonuses = []
        self.cache = {}
        self.memory = memory
        self.move_facts = memory.get("move_facts", {})
        self.resize_facts = memory.get("resize_facts", {})
        self.fire_facts = memory.get("fire_facts", {})

        target_mask = np.zeros((64, 64), dtype=bool)
        for yy, xx in np.argwhere((obs == 6) & ~self.ui):
            y, x = int(yy), int(xx)
            if x + 4 > 64 or y + 4 > 64 or target_mask[y, x]:
                continue
            a = obs[y:y+4, x:x+4]
            border = (np.all(a[0] == 6) and np.all(a[-1] == 6) and
                      np.all(a[:, 0] == 6) and np.all(a[:, -1] == 6))
            if border:
                center = a[1:3, 1:3]
                if np.all(center == 13) or np.all(center == 12):
                    group = 0 if int(center[0, 0]) == 13 else 1
                    cnt = tuple(int(np.count_nonzero(a == c))
                                for c in (6, 13, 12))
                    self.targets.append((x, y, group, cnt))
                    target_mask[y:y+4, x:x+4] = True

        self.doors = [
            (obs == 13) & ~self.ui & ~target_mask,
            (obs == 12) & ~self.ui & ~target_mask,
        ]
        self.door_counts = [int(np.count_nonzero(a)) for a in self.doors]

        # Detect the small teleport centers and movement-budget pickups.
        small = set(tuple(p) for p in memory.get("known_small", []))
        big = set(tuple(p) for p in memory.get("known_big", []))
        bonus_mask = np.zeros((64, 64), dtype=bool)
        small_center_mask = np.zeros((64, 64), dtype=bool)

        for color in (15, 14):
            m = (obs == color) & ~self.ui
            for cells in _components(m):
                ys = [a for a, b in cells]
                xs = [b for a, b in cells]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x + 1, max(ys) - y + 1
                if w == 2 and h == 2 and len(cells) == 4:
                    if color == 15:
                        small.add((x, y))
                    else:
                        self.bonuses.append((x, y, 2, 2))
                        bonus_mask[y:y+2, x:x+2] = True

        def corners(color, distance):
            m = (obs == color) & ~self.ui
            found = []
            for yy, xx in np.argwhere(m):
                y, x = int(yy), int(xx)
                if x + distance < 64 and y + distance < 64:
                    if (m[y, x+distance] and m[y+distance, x] and
                            m[y+distance, x+distance]):
                        found.append((x, y))
            return found

        active_big = []
        active_small = []
        for x, y in corners(11, 5):
            big.add((x + 1, y + 1))
            active_big.append((x + 1, y + 1))
        for x, y in corners(15, 5):
            small.add((x + 2, y + 2))
            active_small.append((x + 2, y + 2))

        ppos, ps, _ = player
        px, py = ppos % 64, ppos // 64
        rings = []
        for x, y in corners(11, 3):
            is_small = ((x + 1, y + 1) in small or
                        (ps == 0 and (px, py) == (x + 1, y + 1)))
            if is_small:
                small.add((x + 1, y + 1))
                rings.append((x, y))
            else:
                big.add((x, y))

        # A small destination's surrounding 4x4 ring is solid.
        for x, y in small:
            if (x - 1, y - 1) not in rings:
                if (0 <= x - 1 and x + 2 < 64 and
                        0 <= y - 1 and y + 2 < 64):
                    if all(obs[yy, xx] == 11 for xx, yy in
                           ((x-1, y-1), (x+2, y-1),
                            (x-1, y+2), (x+2, y+2))):
                        rings.append((x - 1, y - 1))
            if 0 <= x <= 62 and 0 <= y <= 62:
                small_center_mask[y:y+2, x:x+2] = True

        big = {p for p in big if p not in {(x-1, y-1) for x, y in small}}
        big = {p for p in big if 0 <= p[0] <= 60 and 0 <= p[1] <= 60}
        small = {p for p in small if 0 <= p[0] <= 62 and 0 <= p[1] <= 62}

        order_key = lambda p: (p[1], p[0])
        self.big = sorted(big, key=order_key)
        self.small = sorted(small, key=order_key)
        if active_big:
            a = next((p for p in active_big if p in big), None)
            if a is not None:
                self.big.remove(a)
                self.big.insert(0, a)

        if self.small:
            last = memory.get("last_small")
            if last is not None and tuple(last) in self.small:
                j = self.small.index(tuple(last))
                self.small = self.small[j+1:] + self.small[:j+1]
            elif active_small:
                a = next((p for p in active_small if p in small), None)
                if a is not None:
                    self.small.remove(a)
                    self.small.insert(0, a)

        memory["known_big"] = list(big)
        memory["known_small"] = list(small)
        memory["ui_rects"] = self.rects

        # Recover the underlying maze beneath non-solid paint and grilles.
        paint = np.array(obs, copy=True)
        paint[self.ui] = 5
        paint[target_mask | self.doors[0] | self.doors[1] | bonus_mask] = 2
        for x, y in big:
            paint[y:y+4, x:x+4] = 2
        for x, y in small:
            paint[y:y+2, x:x+2] = 2
        size = 4 if ps else 2
        paint[py:py+size, px:px+size] = 2
        paint[goal_mask] = 5

        unknown = (((obs == 4) | (obs == 11) | (obs == 15)) &
                   ~self.ui)
        for x, y in big:
            unknown[y:y+4, x:x+4] = False
        unknown[small_center_mask] = False
        unknown[py:py+size, px:px+size] = False
        unknown[goal_mask] = False

        # Opposite-side agreement reconstructs narrow corridors through gates.
        for yy, xx in np.argwhere(unknown):
            y, x = int(yy), int(xx)
            vals, distances = [], []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                v, dist = 5, 1
                a, b = x + dx, y + dy
                while 0 <= a < 64 and 0 <= b < 64:
                    if not unknown[b, a]:
                        v = 2 if paint[b, a] == 2 else 5
                        break
                    a += dx
                    b += dy
                    dist += 1
                vals.append(v)
                distances.append(dist)
            agreements = []
            for a, b in ((0, 1), (2, 3)):
                if vals[a] == vals[b]:
                    agreements.append((distances[a] + distances[b], vals[a]))
            if agreements:
                agreements.sort()
                value = agreements[0][1]
            else:
                value = 2 if vals.count(2) >= 3 else 5
            paint[y, x] = value

        ordinary = (paint == 2)
        self.base_ray = ~ordinary
        self.base_grow = ~ordinary
        self.base_move = ~ordinary

        # Gates are solid to the player, but not to fire or size-changing.
        self.base_move[(obs == 4) & ~self.ui] = True
        for x, y in rings:
            for xx, yy in ((x, y), (x+3, y), (x, y+3), (x+3, y+3)):
                if 0 <= xx < 64 and 0 <= yy < 64:
                    self.base_move[yy, xx] = True

        # These will be added dynamically rather than treated as fixed walls.
        dyn = target_mask | self.doors[0] | self.doors[1]
        self.base_move[dyn] = False
        self.base_ray[dyn] = False
        self.base_grow[dyn] = False

        self.goal_cover = [_square_cover(goal_mask, z, False).ravel()
                           for z in (2, 4)]
        self.clicks = {}
        masks = {_SPELLS[s] for s in self.allowed}
        for p, bits in ((0, 0), (1, self.initial_bits)):
            for spell in self.allowed:
                self.clicks[p, spell] = _click_path(bits, _SPELLS[spell], masks)

        ys, xs = np.indices((64, 64))
        h = np.full((64, 64), 100.0)
        for x, y, w, ht in goals:
            dx = np.maximum(np.maximum(x - (xs + 3), xs - (x + w - 1)), 0)
            dy = np.maximum(np.maximum(y - (ys + 3), ys - (y + ht - 1)), 0)
            h = np.minimum(h, (dx + 3) // 4 + (dy + 3) // 4)
        if "T" in self.allowed and (self.big or self.small):
            th = min(float(h[y, x]) for x, y in self.big + self.small)
            h = np.minimum(h, 1.0 + th)
        self.heuristic = h.ravel()

    def configuration(self, removed):
        if removed in self.cache:
            return self.cache[removed]

        move = self.base_move.copy()
        grow = self.base_grow.copy()
        ray = self.base_ray.copy()
        grid = np.zeros((64, 64), dtype=np.int16)
        counts = list(self.initial_counts)
        opened = [False, False]
        for i, (x, y, group, cnt) in enumerate(self.targets):
            if removed & (1 << i):
                opened[group] = True
                for j in range(3):
                    counts[j] -= cnt[j]

        for group in range(2):
            if opened[group]:
                counts[1 + group] -= self.door_counts[group]
            else:
                move |= self.doors[group]
                grow |= self.doors[group]
                ray |= self.doors[group]

        grid[ray] = -1
        for i, (x, y, group, cnt) in enumerate(self.targets):
            if not (removed & (1 << i)):
                move[y:y+4, x:x+4] = True
                grid[y:y+4, x:x+4] = i + 1

        counts = tuple(counts)
        covers = [_square_cover(move, z).ravel() for z in (2, 4)]
        grow_cover = _square_cover(grow, 4).ravel()
        p = np.arange(4096)
        xs, ys = p % 64, p // 64
        moves = np.empty((2, 4, 4096), dtype=np.int16)

        for s, size in enumerate((2, 4)):
            for d, (dx, dy) in enumerate(_DIRS):
                nx, ny = xs + size * dx, ys + size * dy
                valid = ((nx >= 0) & (ny >= 0) &
                         (nx + size <= 64) & (ny + size <= 64))
                ni = np.clip(ny, 0, 63) * 64 + np.clip(nx, 0, 63)
                bad = ~valid | covers[s][ni]
                dest = np.where(bad, p, ni)
                if s:
                    hx, hy = xs + 2 * dx, ys + 2 * dy
                    hv = ((hx >= 0) & (hy >= 0) &
                          (hx + size <= 64) & (hy + size <= 64))
                    hi = np.clip(hy, 0, 63) * 64 + np.clip(hx, 0, 63)
                    hb = ~hv | covers[s][hi]
                    dest = np.where(bad & ~hb, hi, dest)
                    win = bad & hb & hv & self.goal_cover[s][hi]
                else:
                    win = bad & valid & self.goal_cover[s][ni]
                dest[win] = -2
                moves[s, d] = dest

        for key, dest in self.move_facts.items():
            pos, s, d, cnt = key
            if cnt == counts and 0 <= pos < 4096:
                moves[s, d, pos] = dest

        rays = np.full((4, 64, 64), -1, dtype=np.int16)
        for y in range(64):
            last = -1
            for x in range(64):
                rays[2, y, x] = last
                if grid[y, x] != 0:
                    last = int(grid[y, x])
            last = -1
            for x in range(63, -1, -1):
                rays[3, y, x] = last
                if grid[y, x] != 0:
                    last = int(grid[y, x])
        for x in range(64):
            last = -1
            for y in range(64):
                rays[0, y, x] = last
                if grid[y, x] != 0:
                    last = int(grid[y, x])
            last = -1
            for y in range(63, -1, -1):
                rays[1, y, x] = last
                if grid[y, x] != 0:
                    last = int(grid[y, x])

        result = (moves, grow_cover, rays, counts)
        self.cache[removed] = result
        return result

    def fire_hit(self, pos, s, d, configuration):
        if d >= 4:
            return -1
        moves, grow, rays, cnt = configuration
        if (pos, s, d, cnt) in self.fire_facts:
            return -1
        x, y = pos % 64, pos // 64
        size = 4 if s else 2
        if d == 1:
            y += size - 1
        elif d == 3:
            x += size - 1
        if not (0 <= x < 64 and 0 <= y < 64):
            return -1
        return int(rays[d, y, x]) - 1

    def resize(self, pos, s, configuration):
        _, grow, _, cnt = configuration
        fact = self.resize_facts.get((pos, s, cnt))
        if fact is not None:
            return fact
        if s:
            return pos, 0
        x, y = pos % 64, pos // 64
        for dx in (-2, -1, 0):
            for dy in (-2, -1, 0):
                xx, yy = x + dx, y + dy
                if 0 <= xx <= 60 and 0 <= yy <= 60:
                    pp = yy * 64 + xx
                    if not grow[pp]:
                        return pp, 1
        return pos, s

    def collect(self, pos, s, taken, resizing=False):
        x, y = pos % 64, pos // 64
        size = (4 if s else 2) if resizing else 2
        result = taken
        # The simulator collects at most one pickup per action.
        for i, (bx, by, bw, bh) in enumerate(self.bonuses):
            if not (taken & (1 << i)):
                if x < bx+bw and x+size > bx and y < by+bh and y+size > by:
                    result |= 1 << i
                    break
        return result


def _search(world, direction):
    start_time = time.monotonic()
    pos, s, _ = world.player
    useful_fire = "F" in world.allowed and bool(world.targets)
    if not useful_fire:
        direction = 4
    pattern_flag = int(world.initial_bits != 0)
    start = (pos, s, direction, 0, 0, 0, 0, pattern_flag)
    distance = {start: 0}
    parent = {}
    heap = [(1.15 * float(world.heuristic[pos]), 0, 0, start)]
    counter = 0
    best = start
    best_metric = -float(world.heuristic[pos])
    final = None
    win_edge = None
    expansions = 0
    full_targets = (1 << len(world.targets)) - 1

    def relax(old, new, cost, edge):
        nonlocal counter
        ng = distance[old] + cost
        if ng < distance.get(new, 10**9):
            distance[new] = ng
            parent[new] = (old, edge)
            counter += 1
            priority = ng + 1.15 * float(world.heuristic[new[0]])
            heapq.heappush(heap, (priority, ng, counter, new))

    while heap:
        _, g, _, state = heapq.heappop(heap)
        if g != distance.get(state):
            continue
        expansions += 1
        if expansions % 512 == 0:
            if time.monotonic() - start_time > 7.6 or expansions > 180000:
                break

        pos, s, d, removed, bi, si, taken, pf = state
        conf = world.configuration(removed)
        moves, grow, rays, cnt = conf
        can_fire = useful_fire and removed != full_targets

        metric = (7 * removed.bit_count() + 4 * taken.bit_count()
                  - float(world.heuristic[pos]) - 0.01 * g)
        if metric > best_metric:
            best_metric, best = metric, state

        for nd in range(4):
            dest = int(moves[s, nd, pos])
            if dest == -2:
                final = state
                win_edge = ("M", nd)
                break
            ndir = nd if can_fire else 4
            ntaken = world.collect(dest, s, taken)
            if dest == pos and ntaken == taken:
                if ndir == d:
                    continue
                if not can_fire or world.fire_hit(pos, s, nd, conf) < 0:
                    continue
            new = (dest, s, ndir, removed, bi, si, ntaken, pf)
            reward = 10 if ntaken != taken else 0
            relax(state, new, 1 - reward, ("M", nd))
        if final is not None:
            break

        if "R" in world.allowed:
            clicks = world.clicks.get((pf, "R"))
            if clicks is not None:
                dest, ns = world.resize(pos, s, conf)
                if (dest, ns) != (pos, s):
                    ntaken = world.collect(dest, ns, taken, True)
                    new = (dest, ns, d, removed, bi, si, ntaken, 0)
                    reward = 10 if ntaken != taken else 0
                    relax(state, new, len(clicks) + 1 - reward, ("R", None))

        if "T" in world.allowed:
            clicks = world.clicks.get((pf, "T"))
            points = world.big if s else world.small
            if clicks is not None and points:
                index = (bi if s else si) % len(points)
                x, y = points[index]
                nb, ni = bi, si
                if s:
                    nb = (index + 1) % len(points)
                else:
                    ni = (index + 1) % len(points)
                dest = y * 64 + x
                new = (dest, s, d, removed, nb, ni, taken, 0)
                if new != state:
                    relax(state, new, len(clicks) + 1, ("T", None))

        if can_fire and d < 4:
            clicks = world.clicks.get((pf, "F"))
            hit = world.fire_hit(pos, s, d, conf)
            if clicks is not None and hit >= 0:
                nr = removed | (1 << hit)
                nd = d if nr != full_targets else 4
                new = (pos, s, nd, nr, bi, si, taken, 0)
                relax(state, new, len(clicks) + 1, ("F", hit))

    if final is None:
        if best == start:
            return []
        final = best

    transitions = []
    cursor = final
    while cursor != start:
        old, edge = parent[cursor]
        transitions.append((old, cursor, edge))
        cursor = old
    transitions.reverse()
    if win_edge is not None:
        transitions.append((final, None, win_edge))

    queue = []
    for old, new, edge in transitions:
        kind, arg = edge
        if kind == "M":
            bits = world.initial_bits if old[7] else 0
            expected = None if new is None else (new[0], new[1], bits)
            queue.append({
                "a": "ACTION" + str(arg + 1),
                "r": None, "c": None, "kind": "M",
                "expect": expected, "fd": arg
            })
            continue

        clicks = world.clicks[old[7], kind]
        bits = world.initial_bits if old[7] else 0
        for j, cell in enumerate(clicks):
            bits ^= 1 << cell
            last = j == len(clicks) - 1
            expected = ((new[0], new[1], 0) if last
                        else (old[0], old[1], bits))
            entry = {
                "a": "ACTION6",
                "r": 50 + 5 * (cell // 3),
                "c": 25 + 5 * (cell % 3),
                "kind": kind if last else "C",
                "expect": expected,
                "fd": old[2],
            }
            if last and kind == "F":
                x, y, _, _ = world.targets[arg]
                entry["target"] = (x, y)
            queue.append(entry)
    return queue


def act(obs, available, memory):
    obs = np.asarray(obs)
    memory["calls"] = memory.get("calls", 0) + 1
    calls = memory["calls"]
    player, goals, goal_mask = _player_and_goals(obs)
    bits = _panel_bits(obs)

    def emergency():
        if "RESET" in available and not memory.get("reset_used"):
            memory["reset_used"] = True
            for key in ("queue", "pending", "dir", "last_small",
                        "move_facts", "resize_facts", "fire_facts"):
                memory.pop(key, None)
            return "RESET", None, None
        for a in ("ACTION1", "ACTION3", "ACTION4", "ACTION2"):
            if a in available:
                return a, None, None
        return "ACTION6", 50, 30

    if player is None:
        return emergency()

    pos, s, rendered_direction = player
    pending = memory.pop("pending", None)
    if pending is not None:
        expected = pending.get("expect")
        valid = expected is not None and expected == (pos, s, bits)
        if pending["kind"] == "F" and "target" in pending:
            x, y = pending["target"]
            if np.any(obs[y:y+4, x:x+4] == 6):
                valid = False

        if pending["kind"] == "M":
            fd = pending["fd"]
            memory["dir"] = fd if rendered_direction == fd else 4

        if pending["kind"] == "T" and pending["before"][1] == 0:
            memory["last_small"] = (pos % 64, pos // 64)

        if not valid:
            memory["queue"] = []
            # Do not mistake the opening tutorial, which consumes an input,
            # for an invisible obstacle.
            if calls > 2:
                oldpos, olds = pending["before"]
                cnt = pending["cnt"]
                kind = pending["kind"]
                if kind == "M" and olds == s:
                    key = (oldpos, olds, pending["fd"], cnt)
                    memory.setdefault("move_facts", {})[key] = pos
                elif kind == "R":
                    key = (oldpos, olds, cnt)
                    memory.setdefault("resize_facts", {})[key] = (pos, s)
                elif kind == "F":
                    key = (oldpos, olds, pending["fd"], cnt)
                    memory.setdefault("fire_facts", {})[key] = True

    queue = memory.get("queue", [])
    if not queue:
        if not goals:
            return emergency()
        world = _World(obs, player, goals, goal_mask, memory)
        direction = memory.get("dir", 4)
        if direction != rendered_direction:
            direction = 4
        queue = _search(world, direction)
        memory["queue"] = queue
        if not queue:
            return emergency()

    entry = queue.pop(0)
    if entry["a"] not in available:
        memory["queue"] = []
        return emergency()

    rects = memory.get("ui_rects", [(22, 47, 17, 17), (62, 0, 2, 64)])
    entry["before"] = (pos, s)
    entry["cnt"] = _counts(obs, rects)
    memory["pending"] = entry
    return entry["a"], entry["r"], entry["c"]

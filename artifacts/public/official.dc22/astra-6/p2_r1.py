import numpy as np
import math
import time
import heapq
from collections import deque, Counter


def _components(mask, diagonal=True):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for yy, xx in zip(*np.nonzero(mask)):
        x, y = int(xx), int(yy)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        pts = []
        while stack:
            a, b = stack.pop()
            pts.append((a, b))
            for dx, dy in dirs:
                c, d = a + dx, b + dy
                if (0 <= c < w and 0 <= d < h and
                        mask[d, c] and not seen[d, c]):
                    seen[d, c] = True
                    stack.append((c, d))
        xs, ys = zip(*pts)
        out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, pts))
    return out


def _rect(x, y, w, h, color):
    return {(a, b): color
            for b in range(max(0, y), min(63, y + h))
            for a in range(max(0, x), min(64, x + w))}


def _find_player(obs):
    parts = _components(obs == 14, False)
    if not parts:
        return None
    p = max(parts, key=lambda z: len(z[4]))
    size = max(2, min(p[2] - p[0], p[3] - p[1]))
    return p[0], p[1], 2 if size < 3 else 4


def _find_panel(obs):
    for x in range(24, 59):
        if np.count_nonzero(obs[:63, x] == 5) >= 27:
            if np.count_nonzero(obs[:63, x:min(64, x + 4)] == 4) < 8:
                return x
    return 40


def _buttons(obs, panel):
    mask = np.zeros((64, 64), dtype=bool)
    mask[:63, panel:] = True
    result = {}
    yellow, orange = [], []
    for color in (8, 9, 6, 15, 2, 11, 10, 7, 12, 1):
        for x, y, X, Y, pts in _components((obs == color) & mask):
            w, h = X - x, Y - y
            point = min(
                pts,
                key=lambda p: abs(2 * p[0] - x - X + 1)
                + abs(2 * p[1] - y - Y + 1),
            )
            rc = (point[1], point[0])
            if color == 10 and 3 <= w <= 5 and 3 <= h <= 5:
                yellow.append((x, y, rc))
            elif color == 9 and 3 <= w <= 5 and 3 <= h <= 5:
                orange.append((x, y, rc))
            elif color == 8 and 7 <= w <= 11 and h <= 3:
                result[205] = rc
            elif w >= 8 and 4 <= h <= 9:
                result[100 + color] = rc
            elif color in (2, 11, 7, 12, 1) and w >= 4 and h >= 4:
                result[100 + color] = rc
    yellow.sort()
    orange.sort()
    if len(yellow) >= 2:
        result[203], result[204] = yellow[0][2], yellow[-1][2]
    if len(orange) >= 2:
        result[201], result[202] = orange[0][2], orange[-1][2]
    return result


def _crane_shapes():
    patterns = [
        [
            "   88888", "   8   8", " 8888   ", "8888    ",
            "8888    ", " 8888   ", "   8   8", "   88888",
        ],
        [
            "        ", "   88888", " 8888  8", "8888    ",
            "8888    ", " 8888  8", "   88888", "        ",
        ],
    ]
    return [
        [(x, y) for y, row in enumerate(p)
         for x, c in enumerate(row) if c == "8"]
        for p in patterns
    ]


_CRANES = _crane_shapes()


class _World:
    def __init__(self, obs, player, panel, buttons, memory):
        self.obs = obs
        self.panel = panel
        self.px, self.py, self.step = player
        self.ox, self.oy = self.px % self.step, self.py % self.step
        self.n = 64 // self.step
        self.size = self.n * self.n
        self.start = self.cell(self.px, self.py)
        self.goal = self.start
        self.buttons = buttons
        self.memory = memory
        self.entities = []
        self.teleports = []
        self.pickups = []
        self.static = set()
        self.covered = np.zeros((64, 64), dtype=bool)
        self.board = np.zeros((64, 64), dtype=bool)
        self.board[:63, :panel] = True

        self.crane_positions = []
        self.crane_offset = (0, 0)
        self.crane_body = []
        self.crane_index = 0
        self.crane_known = False
        self.crane_version = 1
        self.bridge_crane = False
        self.initial_attachment = 0

        self._perceive()
        self._compile()

    def cell(self, x, y):
        return (y // self.step) * self.n + x // self.step

    def xy(self, p):
        return ((p % self.n) * self.step + self.ox,
                (p // self.n) * self.step + self.oy)

    def bits(self, points):
        ans = 0
        for x, y in points:
            if (0 <= x < 64 and 0 <= y < 63 and
                    x % self.step == self.ox and
                    y % self.step == self.oy):
                p = self.cell(x, y)
                if p < self.size:
                    ans |= 1 << p
        return ans

    def collision_bits(self, points):
        ans = 0
        for x, y in points:
            a = (x - self.ox) // self.step
            b = (y - self.oy) // self.step
            if 0 <= a < self.n and 0 <= b < self.n:
                ans |= 1 << (b * self.n + a)
        return ans

    def mark(self, points):
        for x, y in points:
            if 0 <= x < 64 and 0 <= y < 64:
                self.covered[y, x] = True

    def add(self, group, variants, family, anchor=None,
            eligible=False, offset=0, walks=None, blocks=None):
        if walks is None:
            walks = [set(v) for v in variants]
        if blocks is None:
            blocks = [set() for _ in variants]
        e = {
            "group": group, "variants": variants, "family": family,
            "anchor": anchor, "eligible": eligible, "offset": offset,
            "walks": walks, "blocks": blocks,
        }
        self.entities.append(e)
        self.mark(variants[0])
        return e

    def _perceive(self):
        obs, board = self.obs, self.board
        scale = self.step // 2

        goals = [
            p for p in _components((obs == 11) & board, False)
            if p[2] - p[0] <= 4 * scale and p[3] - p[1] <= 4 * scale
        ]
        if goals:
            g = min(goals, key=lambda p: abs(len(p[4]) - self.step**2))
            self.goal = self.cell(g[0], g[1])

        for x, y, X, Y, pts in _components(
                ((obs == 6) | (obs == 7)) & board):
            colors = {int(obs[b, a]) for a, b in pts}
            if (6 in colors and 7 in colors and
                    X - x <= 4 * scale and Y - y <= 4 * scale):
                self.teleports.append((x, y))
                self.static.update(pts)

        pickup_pixels = set()
        for x, y, X, Y, pts in _components((obs == 15) & board):
            if (X - x <= 4 * scale and Y - y <= 2 * scale
                    and len(pts) >= 4 * scale * scale):
                self.pickups.append((x, y, X, Y))
                pickup_pixels.update(pts)
                self.static.update(pts)
                self.mark(pts)

        # Preserve exact floor geometry. In particular, do not fill the
        # bounding box of two tiles which meet only at a corner.
        floor = ((obs == 2) | (obs == 11) |
                 (obs == 6) | (obs == 7)) & board
        for x, y in pickup_pixels:
            floor[y, x] = True
        near_red = np.any(
            obs[max(0, self.py - 1):min(63, self.py + self.step + 1),
                max(0, self.px - 1):min(self.panel, self.px + self.step + 1)] == 2
        )
        if near_red:
            floor[self.py:self.py + self.step,
                  self.px:self.px + self.step] = True
        ys, xs = np.nonzero(floor)
        self.static.update((int(x), int(y)) for y, x in zip(ys, xs))
        self.mark(self.static)

        # Checkerboard platforms alternate between a closed barrier and
        # a solid, walkable platform.
        for color in (9, 15):
            mask = (obs == color) & board & ~self.covered
            for x, y, X, Y, pts in _components(mask):
                w, h = X - x, Y - y
                if w < 2 * scale or h < 3 * scale:
                    self.static.update(pts)
                    continue
                hidden = np.count_nonzero(obs[y:Y, x:X] == 14)
                solid = len(pts) + hidden > .78 * w * h
                full = _rect(x, y, w, h, color)
                checker = {
                    p: c for p, c in full.items()
                    if ((p[0] - x) // scale +
                        (p[1] - y) // scale) % 2 == 1
                }
                if solid:
                    variants = [full, checker]
                    walks = [set(full), set()]
                    blocks = [set(), set(full)]
                else:
                    variants = [checker, full]
                    walks = [set(), set(full)]
                    blocks = [set(full), set()]
                e = self.add(color, variants, "gate",
                             walks=walks, blocks=blocks)
                e["rect"] = (x, y, X, Y)

        # Purple centres identify hinged blue bridges.
        blue_parts = _components(((obs == 8) | (obs == 13)) & board)
        for ax, ay, AX, AY, anchor_pts in _components((obs == 13) & board):
            aset = set(anchor_pts)
            part = next((p for p in blue_parts if aset.intersection(p[4])), None)
            if part is None:
                continue
            x, y, X, Y, pts = part
            aw, ah = AX - ax, AY - ay
            horizontal = X - x >= Y - y
            if horizontal:
                strip = [p for p in pts if ay <= p[1] < AY]
                if not strip:
                    continue
                lo = min(p[0] for p in strip)
                hi = max(p[0] for p in strip) + 1
                left, right = ax - lo, hi - AX
                v0 = _rect(lo, ay, hi - lo, ah, 8)
                v1 = _rect(ax, ay - left, aw, left + ah + right, 8)
            else:
                strip = [p for p in pts if ax <= p[0] < AX]
                if not strip:
                    continue
                lo = min(p[1] for p in strip)
                hi = max(p[1] for p in strip) + 1
                left, right = ay - lo, hi - AY
                v0 = _rect(ax, lo, aw, hi - lo, 8)
                v1 = _rect(ax - left, ay, left + aw + right, ah, 8)
            pivot = _rect(ax, ay, aw, ah, 13)
            v0.update(pivot)
            v1.update(pivot)
            self.add(
                8, [v0, v1], "bridge",
                (ax + aw // 2, ay + ah // 2),
                eligible=abs(left - right) <= scale and left > 0,
            )
            self.mark(pts)

        self._find_crane()

        # Cream-coloured crane cargo.
        for x, y, X, Y, pts in _components(
                (obs == 12) & board & ~self.covered):
            if Y - y >= 6 * scale and X - x >= 4 * scale:
                v = {p: 12 for p in pts}
                if X - x == 6 * scale and Y - y == 12 * scale:
                    v = _rect(x + 2 * scale, y, 4 * scale, 12 * scale, 12)
                    v.update(_rect(x, y + 5 * scale, 2 * scale, 2 * scale, 12))
                    v.update(_rect(x, y + 4 * scale, scale, 4 * scale, 12))
                self.add(
                    None, [v], "object",
                    (x + (X - x) // 2, y + (Y - y) // 2), True,
                )
                self.mark(pts)

        # Bridges growing inward from both ends.
        strip_mask = ((obs == 5) | (obs == 1) | (obs == 12))
        strip_mask &= board & ~self.covered
        axes = []
        for x, y, X, Y, pts in _components(strip_mask, False):
            w, h = X - x, Y - y
            horizontal = h == 2 * scale and w >= 12 * scale
            vertical = w == 2 * scale and h >= 12 * scale
            if not (horizontal or vertical):
                continue
            length = w if horizontal else h
            if length > 24 * scale:
                continue
            cross = h if horizontal else w
            nonblank = sum(obs[b, a] in (1, 12) for a, b in pts)
            colored_length = nonblank // max(1, cross)
            increment = max(scale, length // 8)
            pattern = [
                0, increment, 2 * increment, 3 * increment,
                length // 2, 3 * increment, 2 * increment, increment,
            ]
            candidates = [
                k for k, end in enumerate(pattern)
                if abs((length if k == 4 else 2 * end)
                       - colored_length) <= scale
            ]
            key = (x, y, X, Y)
            remembered = self.memory.get("e_phases", {}).get(key)
            if remembered in candidates:
                phase = remembered
            elif candidates:
                phase = candidates[0]
            else:
                phase = 0
            variants = []
            for j in range(8):
                k = (phase + j) % 8
                end = pattern[k]
                v = {}
                for t in range(length):
                    if k == 4 or t < end or t >= length - end:
                        color = 12 if k == 4 else 1
                        for q in range(cross):
                            p = (x + t, y + q) if horizontal else (x + q, y + t)
                            v[p] = color
                variants.append(v)
            e = self.add(2, variants, "extend", offset=phase)
            e["rect"] = key
            axes.append((x, y, X, Y, horizontal))
            self.mark(pts)

        # Short platforms sliding through six positions.
        short_mask = ((obs == 1) | (obs == 12)) & board & ~self.covered
        for x, y, X, Y, pts in _components(short_mask, False):
            w, h = X - x, Y - y
            horizontal = h == 2 * scale and 4 * scale <= w <= 8 * scale
            vertical = w == 2 * scale and 4 * scale <= h <= 8 * scale
            if not (horizontal or vertical):
                continue
            length = w if horizontal else h
            cur = x if horizontal else y
            fixed = y if horizontal else x
            key = (horizontal, fixed, length)
            base = self.memory.get("f_bases", {}).get(key, cur)
            for ex, ey, EX, EY, eh in axes:
                if horizontal == eh:
                    eb = ex if horizontal else ey
                    if eb <= cur <= eb + 10 * scale:
                        base = eb
            if any(obs[b, a] == 12 for a, b in pts):
                base = cur - 10 * scale
            base = min(base, cur)
            phase = max(0, min(5, (cur - base) // (2 * scale)))
            variants = []
            for j in range(6):
                k = (phase + j) % 6
                pos = base + 2 * scale * k
                color = 12 if k == 5 else 1
                if horizontal:
                    variants.append(_rect(pos, y, length, h, color))
                else:
                    variants.append(_rect(x, pos, w, length, color))
            e = self.add(11, variants, "slide", offset=phase)
            e["base_key"], e["base"] = key, base
            self.mark(pts)

        remaining = board & ~self.covered
        for color in (1, 2, 6, 7, 8, 10, 11, 12, 13):
            ys, xs = np.nonzero((obs == color) & remaining)
            self.static.update((int(x), int(y)) for y, x in zip(ys, xs))

        for p in self.memory.get("teleports", []):
            if p not in self.teleports:
                x, y = p
                if abs(x - self.px) < self.step and abs(y - self.py) < self.step:
                    self.teleports.append(p)
                    self.static.update(_rect(x, y, self.step, self.step, 6))
        if len(self.teleports) == 1 and 106 in self.buttons:
            if (self.px, self.py) not in self.teleports:
                self.teleports.append((self.px, self.py))
                self.static.update(_rect(self.px, self.py,
                                         self.step, self.step, 6))
        self.teleports = list(dict.fromkeys(self.teleports))

        learned = self.memory.get("control_pixels", {})
        for e in self.entities:
            if e["group"] is None:
                continue
            union = set()
            for v in e["variants"]:
                union.update(v)
            tally = Counter(learned[p] for p in union if p in learned)
            if tally:
                color, count = tally.most_common(1)[0]
                if count >= max(3, len(union) // 12):
                    e["group"] = color

        if self.crane_positions:
            x, y = self.crane_positions[self.crane_index]
            anchor = (x + self.crane_offset[0], y + self.crane_offset[1])
            for i, e in enumerate(self.entities):
                if not e["eligible"]:
                    continue
                if not self.bridge_crane and e["family"] != "object":
                    e["eligible"] = False
                if (e["anchor"] == anchor and self.crane_known
                        and self.crane_version == 2 and e["family"] == "object"):
                    self.initial_attachment = i + 1

        # An unrecognized surface can be inferred under the actor, but never
        # extrapolated into adjacent empty space.
        explained = (self.px, self.py) in self.static
        explained |= any((self.px, self.py) in e["walks"][0]
                         for e in self.entities)
        if not explained:
            self.static.update(_rect(self.px, self.py,
                                     self.step, self.step, 2))

    def _find_crane(self):
        if not all(a in self.buttons for a in (201, 202, 203, 204, 205)):
            return
        obs = self.obs
        best = None
        for version, shape in enumerate(_CRANES, 1):
            for y in range(56):
                for x in range(max(0, self.panel - 7)):
                    if obs[y + 3, x] != 8 or obs[y + 4, x] != 8:
                        continue
                    correct = sum(obs[y + dy, x + dx] in (8, 14)
                                  for dx, dy in shape)
                    if correct >= len(shape) - 1:
                        score = correct / len(shape)
                        if best is None or score > best[0]:
                            best = (score, x, y, version, shape)
        if best is not None:
            _, x, y, version, shape = best
            self.crane_known = True
            self.crane_version = version
            self.crane_offset = (9, 4)
            self.crane_body = shape
            zeros = [
                (int(a), int(b))
                for b, a in np.argwhere((obs == 0) & self.board)
                if a < self.panel - 2 and b < 62
            ]
            if zeros:
                bx = min(p[0] for p in zeros) - 1
                by = min(p[1] for p in zeros) + 10
            else:
                bx, by = x, y
            positions = [(bx + 4 * i, by) for i in range(4)]
            positions += [(bx + 4 * i, by - 12) for i in range(4)]
            positions += [(bx, by - 4), (bx, by - 8)]
            if (x, y) not in positions:
                positions.append((x, y))
            self.crane_positions = positions
            self.crane_index = positions.index((x, y))
            self.mark((x + dx, y + dy) for dx, dy in shape)
            return

        parts = [
            p for p in _components((obs == 8) & self.board & ~self.covered)
            if len(p[4]) >= 10
        ]
        if not parts:
            return
        x, y, X, Y, pts = max(parts, key=lambda p: len(p[4]))
        w, h = X - x, Y - y
        if w > 24 or h > 24:
            return
        self.bridge_crane = True
        self.crane_body = [(a - x, b - y) for a, b in pts]
        self.crane_offset = (w // 2, h // 2)
        cx, cy = x + w // 2, y + h // 2
        positions = [(x, y)]
        for yy in range(cy % 4, 63, 4):
            for xx in range(cx % 4, self.panel, 4):
                if obs[yy, xx] == 0:
                    positions.append((xx - w // 2, yy - h // 2))
        for e in self.entities:
            if e["eligible"] and e["anchor"]:
                ax, ay = e["anchor"]
                if (ax - cx) % 4 == 0 and (ay - cy) % 4 == 0:
                    positions.append((ax - w // 2, ay - h // 2))
        self.crane_positions = list(dict.fromkeys(positions))
        self.mark(pts)

    def _compile(self):
        self.groups = {}
        for e in self.entities:
            c = e["group"]
            if c is not None:
                self.groups[c] = math.lcm(self.groups.get(c, 1),
                                          len(e["variants"]))
        self.group_list = sorted(self.groups)
        self.mult = {}
        total = 1
        for c in self.group_list:
            self.mult[c] = total
            total *= self.groups[c]

        self.static_bits = self.bits(self.static)
        self.pickup_bits = 0
        for x, y, X, Y in self.pickups:
            self.pickup_bits |= self.collision_bits(
                _rect(x, y, X - x, Y - y, 15))

        tele = list(dict.fromkeys(self.cell(x, y) for x, y in self.teleports))
        self.tele_target = {}
        if len(tele) > 1:
            for i, p in enumerate(tele):
                self.tele_target[p] = tele[(i + 1) % len(tele)]

        known = self.memory.get("known_buttons", {})
        conditional = self.memory.get("conditional", set())
        seen = self.memory.get("guard_seen", {})
        self.enabled, self.guards = set(), {}
        for c in self.group_list + [6]:
            token = 100 + c
            if token in self.buttons or token in known:
                self.enabled.add(c)
            if c == 15 and self.pickups:
                self.enabled.add(c)
            if token in conditional:
                self.guards[c] = self.bits(seen.get(token, set()))

        self.crane_block = []
        for x, y in self.crane_positions:
            self.crane_block.append(self.collision_bits(
                (x + dx, y + dy) for dx, dy in self.crane_body))
        if not self.crane_positions:
            self.crane_positions = [(0, 0)]
            self.crane_block = [0]

        self.union = self.static_bits
        self.group_near = {c: 0 for c in self.group_list}
        for e in self.entities:
            e["base_walk"] = [self.bits(v) for v in e["walks"]]
            e["base_block"] = [self.collision_bits(v) for v in e["blocks"]]
            e["carried_walk"], e["carried_block"] = [], []
            influence = set()
            for v in e["variants"]:
                influence.update(v)
            e["influence"] = self.collision_bits(influence)
            union = 0
            for z in e["base_walk"]:
                union |= z
            if e["eligible"] and e["anchor"] is not None:
                for x, y in self.crane_positions:
                    dx = x + self.crane_offset[0] - e["anchor"][0]
                    dy = y + self.crane_offset[1] - e["anchor"][1]
                    ws = [self.bits((a + dx, b + dy) for a, b in v)
                          for v in e["walks"]]
                    bs = [self.collision_bits((a + dx, b + dy) for a, b in v)
                          for v in e["blocks"]]
                    e["carried_walk"].append(ws)
                    e["carried_block"].append(bs)
                    for z in ws:
                        union |= z
            self.union |= union
            if e["group"] is not None:
                self.group_near[e["group"]] |= union

        self.neighbors = [[] for _ in range(self.size)]
        for p in range(self.size):
            x, y = p % self.n, p // self.n
            if y:
                self.neighbors[p].append((p - self.n, 1))
            if y + 1 < self.n:
                self.neighbors[p].append((p + self.n, 2))
            if x:
                self.neighbors[p].append((p - 1, 3))
            if x + 1 < self.n:
                self.neighbors[p].append((p + 1, 4))
        for c, bits in list(self.group_near.items()):
            expanded, v = bits, bits
            while v:
                low = v & -v
                p = low.bit_length() - 1
                v -= low
                if p < self.size:
                    for q, _ in self.neighbors[p]:
                        expanded |= 1 << q
            self.group_near[c] = expanded

        lookup = {p: i for i, p in enumerate(self.crane_positions)}
        self.crane_next = [[] for _ in self.crane_positions]
        for i, (x, y) in enumerate(self.crane_positions):
            for token, dx, dy in (
                    (201, 0, -4), (202, 0, 4),
                    (203, -4, 0), (204, 4, 0)):
                if (x + dx, y + dy) in lookup:
                    self.crane_next[i].append((lookup[(x + dx, y + dy)], token))

        self.learned_block = 0
        for (x, y), patterns in self.memory.get("failed_moves", {}).items():
            if 0 <= x <= 64 - self.step and 0 <= y <= 63 - self.step:
                patch = self.obs[y:y + self.step, x:x + self.step].tobytes()
                if patch in patterns:
                    self.learned_block |= 1 << self.cell(x, y)

        self.configs, self.maps, self.transitions = [], [], []
        self.config_ids = {}
        unlocked = int(115 in self.buttons or not self.pickups)
        self.initial = self.intern(
            (0, self.crane_index, self.initial_attachment, unlocked))

        # Relaxed distance ignores gates and permits every platform position.
        self.h = [10000] * self.size
        self.h[self.goal] = 0
        queue = deque([self.goal])
        reverse = {}
        for a, b in self.tele_target.items():
            reverse.setdefault(b, []).append(a)
        while queue:
            p = queue.popleft()
            candidates = [q for q, _ in self.neighbors[p]]
            candidates += reverse.get(p, [])
            for q in candidates:
                if (self.union >> q) & 1 and self.h[q] > self.h[p] + 1:
                    self.h[q] = self.h[p] + 1
                    queue.append(q)
        gx, gy = self.xy(self.goal)
        for p in range(self.size):
            if self.h[p] == 10000:
                x, y = self.xy(p)
                self.h[p] = 200 + (abs(x - gx) + abs(y - gy)) / self.step

    def intern(self, cfg):
        old = self.config_ids.get(cfg)
        if old is not None:
            return old
        i = len(self.configs)
        self.config_ids[cfg] = i
        self.configs.append(cfg)
        self.maps.append(None)
        self.transitions.append(None)
        return i

    def phase(self, code, e):
        c = e["group"]
        if c is None:
            return 0
        return (code // self.mult[c]) % len(e["variants"])

    def get_map(self, cid):
        cached = self.maps[cid]
        if cached is not None:
            return cached
        code, robot, attached, unlocked = self.configs[cid]
        support = self.static_bits
        blocked = self.crane_block[robot]
        changed = 0
        if robot != self.crane_index:
            changed |= self.crane_block[robot] | self.crane_block[self.crane_index]
        for i, e in enumerate(self.entities):
            k = self.phase(code, e)
            if attached == i + 1 and e["carried_walk"]:
                w = e["carried_walk"][robot][k]
                b = e["carried_block"][robot][k]
                support |= w
                blocked |= b
                if robot != self.crane_index or k:
                    changed |= e["influence"] | w | b
            else:
                support |= e["base_walk"][k]
                blocked |= e["base_block"][k]
                if k:
                    changed |= e["influence"]
        blocked |= self.learned_block & ~changed
        result = (support, support & ~blocked)
        self.maps[cid] = result
        return result

    def get_transitions(self, cid):
        old = self.transitions[cid]
        if old is not None:
            return old
        code, robot, attached, unlocked = self.configs[cid]
        result = []
        for c in self.group_list:
            if c not in self.enabled or (c == 15 and not unlocked):
                continue
            mul = self.mult[c]
            old_phase = (code // mul) % self.groups[c]
            new_phase = (old_phase + 1) % self.groups[c]
            nc = code + (new_phase - old_phase) * mul
            ni = self.intern((nc, robot, attached, unlocked))
            near = self.guards.get(c, self.group_near[c])
            result.append((ni, 100 + c, near))
        if 205 in self.buttons:
            for ri, token in self.crane_next[robot]:
                ni = self.intern((code, ri, attached, unlocked))
                result.append((ni, token, -1))
            if attached == 0:
                x, y = self.crane_positions[robot]
                anchor = (x + self.crane_offset[0], y + self.crane_offset[1])
                for i, e in enumerate(self.entities):
                    if e["eligible"] and e["anchor"] == anchor:
                        ni = self.intern((code, robot, i + 1, unlocked))
                        result.append((ni, 205, -1))
        self.transitions[cid] = result
        return result

    def after_move(self, cid, p):
        cfg = self.configs[cid]
        if not cfg[3] and ((self.pickup_bits >> p) & 1):
            return self.intern((cfg[0], cfg[1], cfg[2], 1))
        return cid

    def trace(self, end, parent):
        path = []
        key = end
        while key in parent:
            previous, token = parent[key]
            cid, p = divmod(key, self.size)
            path.append((token, cid, p))
            key = previous
        path.reverse()
        return path

    def search(self, seconds=5.7):
        deadline = time.perf_counter() + seconds
        size = self.size
        start = self.initial * size + self.start
        distance = {start: 0}
        parent = {}
        heap = [(1.6 * self.h[self.start], 0, start)]
        best, best_score = start, self.h[self.start]
        expanded = 0

        while heap:
            _, ng, key = heapq.heappop(heap)
            g = -ng
            if distance.get(key) != g:
                continue
            cid, p = divmod(key, size)
            if p == self.goal:
                path = self.trace(key, parent)
                # Victory is checked after a directional action.
                if path and path[-1][0] >= 100:
                    _, valid = self.get_map(cid)
                    for q, token in self.neighbors[p]:
                        if not ((valid >> q) & 1):
                            path.append((token, cid, p))
                            break
                return path
            score = self.h[p] + .025 * g
            if score < best_score:
                best_score, best = score, key
            expanded += 1
            if expanded % 256 == 0:
                if time.perf_counter() > deadline or expanded > 240000:
                    break

            support, valid = self.get_map(cid)
            successors = []
            for q, token in self.neighbors[p]:
                if (valid >> q) & 1:
                    successors.append((self.after_move(cid, q), q, token))
            bit = 1 << p
            for ni, token, near in self.get_transitions(cid):
                if near != -1 and not (near & bit):
                    continue
                if self.get_map(ni)[0] & bit:
                    successors.append((ni, p, token))
            if p in self.tele_target and 6 in self.enabled:
                allowed = self.guards.get(6, -1)
                if allowed == -1 or allowed & bit:
                    q = self.tele_target[p]
                    if (support >> q) & 1:
                        successors.append((cid, q, 106))

            for ni, q, token in successors:
                nk = ni * size + q
                cost = g + 1
                if cost >= distance.get(nk, 10**9):
                    continue
                distance[nk] = cost
                parent[nk] = (key, token)
                heapq.heappush(
                    heap, (cost + 1.6 * self.h[q], -cost, nk))

        if best != start:
            path = self.trace(best, parent)
            if path:
                return path
        return self.explore()

    def explore(self):
        _, valid = self.get_map(self.initial)
        queue = deque([self.start])
        parent = {}
        distance = {self.start: 0}
        while queue:
            p = queue.popleft()
            for q, token in self.neighbors[p]:
                if q not in distance and ((valid >> q) & 1):
                    parent[q] = (p, token)
                    distance[q] = distance[p] + 1
                    queue.append(q)
        visits = self.memory.get("visits", {})
        best, best_score = None, -1e30
        for p, d in distance.items():
            if p == self.start:
                continue
            x, y = self.xy(p)
            count = visits.get((x, y), 0)
            patch = self.obs[y:min(63, y + self.step),
                             x:min(64, x + self.step)]
            novelty = 16 if count == 0 else -6 * count
            marker = 3 if np.any(np.isin(patch, [1, 6, 7, 10, 11, 12, 15])) else 0
            score = novelty + marker - .65 * d - .025 * self.h[p]
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score > -15:
            rev = []
            p = best
            while p != self.start:
                previous, token = parent[p]
                rev.append((token, p))
                p = previous
            cid = self.initial
            path = []
            for token, p in reversed(rev):
                cid = self.after_move(cid, p)
                path.append((token, cid, p))
            return path

        if self.start in self.tele_target and 106 in self.buttons:
            return [(106, self.initial, self.tele_target[self.start])]
        options = []
        uses = self.memory.get("uses", {})
        for ni, token, near in self.get_transitions(self.initial):
            if token in self.buttons and (
                    self.get_map(ni)[0] >> self.start) & 1:
                options.append((uses.get(token, 0), token, ni))
        if options:
            _, token, ni = min(options)
            return [(token, ni, self.start)]
        return []

    def verify(self, obs, cid):
        if not 0 <= cid < len(self.configs):
            return False
        code, robot, attached, unlocked = self.configs[cid]
        rx, ry = self.crane_positions[robot]
        crane_pixels = {(rx + dx, ry + dy) for dx, dy in self.crane_body}
        for i, e in enumerate(self.entities):
            k = self.phase(code, e)
            dx = dy = 0
            if attached == i + 1 and e["anchor"] is not None:
                dx = rx + self.crane_offset[0] - e["anchor"][0]
                dy = ry + self.crane_offset[1] - e["anchor"][1]
            expected = e["variants"][k]
            errors = checked = 0
            for (x, y), color in expected.items():
                x, y = x + dx, y + dy
                if not (0 <= x < self.panel and 0 <= y < 63):
                    continue
                actual = int(obs[y, x])
                if actual == 14 or (x, y) in crane_pixels:
                    continue
                checked += 1
                errors += actual != color
            if errors > max(3, .20 * checked):
                return False
            if e["family"] == "gate":
                x, y, X, Y = e["rect"]
                color = next(iter(expected.values()), e["group"])
                actual = int(np.count_nonzero(obs[y:Y, x:X] == color))
                actor = int(np.count_nonzero(obs[y:Y, x:X] == 14))
                if abs(actual - len(expected)) > max(4, actor + 2):
                    return False
            if e["family"] == "extend" and not expected:
                x, y, X, Y = e["rect"]
                a = obs[y:Y, x:X]
                if np.count_nonzero((a == 1) | (a == 12)) > 3:
                    return False
        return True

    def remember(self, cid):
        code = self.configs[cid][0]
        phases = self.memory.setdefault("e_phases", {})
        bases = self.memory.setdefault("f_bases", {})
        for e in self.entities:
            if e["family"] == "extend":
                phases[e["rect"]] = (e["offset"] + self.phase(code, e)) % 8
            elif e["family"] == "slide":
                bases[e["base_key"]] = e["base"]


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=np.int8)
    player = _find_player(obs)
    if player is None:
        for action in ("ACTION1", "ACTION4", "ACTION2", "ACTION3", "RESET"):
            if action in available:
                return action, None, None
        return available[0], None, None

    px, py, step = player
    panel = _find_panel(obs)
    buttons = _buttons(obs, panel)
    known = memory.setdefault("known_buttons", {})
    guards = memory.setdefault("guard_seen", {})
    conditional = memory.setdefault("conditional", set())
    visits = memory.setdefault("visits", {})
    visits[(px, py)] = visits.get((px, py), 0) + 1

    for token, rc in buttons.items():
        known[token] = rc
        if 100 <= token < 200:
            guards.setdefault(token, set()).add((px, py))
    for token in list(known):
        if 100 <= token < 200 and token not in buttons:
            conditional.add(token)

    previous = memory.get("previous_obs")
    last_token = memory.get("last_token")
    if previous is not None and last_token is not None and 100 <= last_token < 200:
        changed = previous != obs
        changed[:, panel:] = False
        changed[63, :] = False
        changed[(previous == 14) | (obs == 14)] = False
        changed[(previous == 3) | (obs == 3)] = False
        learned = memory.setdefault("control_pixels", {})
        for yy, xx in zip(*np.nonzero(changed)):
            if previous[yy, xx] not in (0, 2) or obs[yy, xx] not in (0, 2):
                learned[(int(xx), int(yy))] = last_token - 100

    world = memory.get("world")
    plan = memory.get("plan", [])
    expected = memory.get("expected")
    good = world is not None and expected is not None
    if good:
        cid, p = expected
        ex, ey = world.xy(p)
        if (ex, ey) != (px, py):
            if last_token in (1, 2, 3, 4):
                old_player = memory.get("last_player")
                if old_player == (px, py):
                    if 0 <= ex <= 64 - step and 0 <= ey <= 63 - step:
                        patch = obs[ey:ey + step, ex:ex + step].tobytes()
                        failed = memory.setdefault("failed_moves", {})
                        failed.setdefault((ex, ey), set()).add(patch)
            good = False
        elif not world.verify(obs, cid):
            good = False
    if good and plan:
        token = plan[0][0]
        if token >= 100 and token not in buttons:
            if token < 200:
                conditional.add(token)
            good = False

    if not good or not plan:
        world = _World(obs, player, panel, buttons, memory)
        memory["teleports"] = list(world.teleports)
        plan = world.search()
        memory["world"] = world

    if plan and plan[0][0] >= 100 and plan[0][0] not in buttons:
        plan = []

    if not plan:
        cid = world.initial
        _, valid = world.get_map(cid)
        choices = []
        for q, token in world.neighbors[world.start]:
            if "ACTION" + str(token) in available and ((valid >> q) & 1):
                x, y = world.xy(q)
                choices.append((visits.get((x, y), 0),
                                world.h[q], token, q))
        if choices:
            _, _, token, p = min(choices)
            cid = world.after_move(cid, p)
            plan = [(token, cid, p)]
        else:
            options = []
            uses = memory.get("uses", {})
            for ni, token, near in world.get_transitions(cid):
                if token in buttons and (
                        world.get_map(ni)[0] >> world.start) & 1:
                    options.append((uses.get(token, 0), token, ni))
            if options:
                _, token, cid = min(options)
                plan = [(token, cid, world.start)]
            else:
                memory["plan"] = []
                memory["expected"] = None
                memory["previous_obs"] = obs.copy()
                memory["last_player"] = (px, py)
                memory["last_token"] = 1
                for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                    if action in available:
                        return action, None, None
                return available[0], None, None

    token, cid, p = plan.pop(0)
    memory["plan"] = plan
    memory["expected"] = (cid, p)
    memory["last_token"] = token
    memory["last_player"] = (px, py)
    memory["previous_obs"] = obs.copy()
    uses = memory.setdefault("uses", {})
    uses[token] = uses.get(token, 0) + 1
    world.remember(cid)

    if token in (1, 2, 3, 4):
        action = "ACTION" + str(token)
        if action in available:
            return action, None, None
    elif token in buttons and "ACTION6" in available:
        row, col = buttons[token]
        return "ACTION6", int(row), int(col)

    memory["plan"] = []
    memory["expected"] = None
    for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if action in available:
            return action, None, None
    return available[0], None, None

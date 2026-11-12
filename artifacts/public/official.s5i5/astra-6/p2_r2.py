import numpy as np
import heapq
import itertools
import time
from collections import Counter

_DIR = ((0, -1), (1, 0), (0, 1), (-1, 0))
_INF = 10000.0


def _rot(x, y, d):
    d &= 3
    if d == 0:
        return x, y
    if d == 1:
        return -y, x
    if d == 2:
        return -x, -y
    return y, -x


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        stack = [(xx, yy)]
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for nx, ny in ((x - 1, y), (x + 1, y),
                           (x, y - 1), (x, y + 1)):
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(comp)
    return result


def _rectangles(mask):
    active = {}
    out = []
    h, w = mask.shape
    for y in range(h + 1):
        runs = []
        if y < h:
            x = 0
            while x < w:
                if not mask[y, x]:
                    x += 1
                    continue
                a = x
                while x < w and mask[y, x]:
                    x += 1
                runs.append((a, x))
        new = {}
        for run in runs:
            new[run] = active.get(run, y)
        for run, sy in active.items():
            if run not in new:
                out.append((run[0], sy, run[1], y))
        active = new
    return out


def _overlap(a, b):
    return (a[0] < b[2] and b[0] < a[2] and
            a[1] < b[3] and b[1] < a[3])


def _initial_box(n):
    dx, dy = _DIR[n["d"]]
    tx = n["bx"] + 3 * (n["L"] - 1) * dx
    ty = n["by"] + 3 * (n["L"] - 1) * dy
    return (min(n["bx"], tx), min(n["by"], ty),
            max(n["bx"], tx) + 3, max(n["by"], ty) + 3)


def _select_arms(nodes, work):
    """Resolve ambiguous ownership of adjoining coloured/base stripes."""
    if len(nodes) < 2:
        return nodes

    required = 0
    for y in range(63):
        for x in range(64):
            if int(work[y, x]) not in (1, 2, 4, 5, 13, 15):
                required |= 1 << (64 * y + x)

    occupied_masks = []
    supports = []
    areas = []
    union = 0
    for n in nodes:
        x0, y0, x1, y1 = _initial_box(n)
        mask = 0
        row = ((1 << (x1 - x0)) - 1) << x0
        for y in range(y0, y1):
            mask |= row << (64 * y)
        occupied_masks.append(mask)
        supports.append(mask & required)
        union |= mask
        areas.append((x1 - x0) * (y1 - y0))
    required &= union

    # The usual scene has no ambiguity at all.
    disjoint = True
    seen = 0
    for mask in occupied_masks:
        if seen & mask:
            disjoint = False
            break
        seen |= mask
    if disjoint:
        return nodes

    offers = {}
    for i, support in enumerate(supports):
        bits = support
        while bits:
            bit = bits & -bits
            bits -= bit
            offers.setdefault(bit, []).append(i)

    ranked_pixels = sorted(offers, key=lambda b: len(offers[b]))
    budget = [12000]

    def solve(remaining, occupied, chosen):
        budget[0] -= 1
        if not remaining:
            return chosen
        if budget[0] < 0:
            return None

        available = [
            i for i in range(len(nodes))
            if supports[i] & remaining and not occupied_masks[i] & occupied
        ]
        available_set = set(available)
        cover = 0
        for i in available:
            cover |= supports[i]
        if remaining & ~cover:
            return None

        choices = None
        for bit in ranked_pixels:
            if not remaining & bit:
                continue
            options = [i for i in offers[bit] if i in available_set]
            if choices is None or len(options) < len(choices):
                choices = options
                if len(choices) <= 1:
                    break
        if not choices:
            return None
        choices.sort(key=lambda i: -areas[i])
        for i in choices:
            result = solve(remaining & ~supports[i],
                           occupied | occupied_masks[i], chosen + [i])
            if result is not None:
                return result
        return None

    selected = solve(required, 0, [])
    if selected is not None:
        selected = set(selected)
        return [n for i, n in enumerate(nodes) if i in selected]
    return nodes


class _Model:
    def __init__(self, obs):
        image = np.asarray(obs, dtype=np.int8)
        self.original = image.copy()
        self.ui_mask = np.zeros((64, 64), dtype=bool)
        self.controls = []
        self.bad = set()
        self.visits = {}
        self.plan = []
        self.heuristics = {}
        self.assignments = []

        boxes = []
        for comp in _components(image == 2):
            xs = [p[0] for p in comp]
            ys = [p[1] for p in comp]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x + 1, max(ys) - y + 1
            if w < 4 or h < 4:
                continue
            patch = image[y:y+h, x:x+w]
            counts = Counter(int(v) for v in patch.flat
                             if int(v) not in (2, 3, 4, 5, 13, 15))
            if not counts:
                continue
            color = counts.most_common(1)[0][0]
            self.ui_mask[y:y+h, x:x+w] = True
            boxes.append((x, y, w, h, color, bool(np.any(patch == 3))))

        work = image.copy()
        work[self.ui_mask] = 5
        work[63, :] = 5
        a = work.tolist()

        goals = []
        ring_pixels = set()
        for y in range(62):
            for x in range(62):
                cross = ((x+1, y), (x, y+1), (x+2, y+1), (x+1, y+2))
                if all(a[py][px] == 13 for px, py in cross):
                    goals.append((x, y))
                    ring_pixels.update(cross)
        points = []
        for y, x in zip(*np.nonzero(work == 13)):
            x, y = int(x), int(y)
            if (x, y) not in ring_pixels:
                points.append((x-1, y-1))
        self.goals = list(dict.fromkeys(goals))

        candidates = set()
        for y, x in zip(*np.nonzero(work == 3)):
            x, y = int(x), int(y)
            for k in range(3):
                candidates.add((x-k, y-2, 0))
                candidates.add((x, y-k, 1))
                candidates.add((x-k, y, 2))
                candidates.add((x-2, y-k, 3))

        nodes = []
        used = set()
        for bx, by, d in sorted(candidates):
            if not (0 <= bx <= 61 and 0 <= by <= 60):
                continue
            edge, body = [], []
            for yy in range(3):
                for xx in range(3):
                    on_edge = ((d == 0 and yy == 2) or
                               (d == 1 and xx == 0) or
                               (d == 2 and yy == 0) or
                               (d == 3 and xx == 2))
                    if on_edge:
                        edge.append(a[by+yy][bx+xx])
                    else:
                        body.append(a[by+yy][bx+xx])
            if not all(v in (3, 13) for v in edge) or 3 not in edge:
                continue
            colors = [v for v in body if v != 13]
            if not colors:
                continue
            c = colors[0]
            if c in (1, 2, 3, 4, 5, 13, 15):
                continue
            if any(v not in (c, 13) for v in body):
                continue
            if (bx, by, c) in used:
                continue

            dx, dy = _DIR[d]
            length = 1
            while length < 24:
                x, y = bx + 3*length*dx, by + 3*length*dy
                if not (0 <= x <= 61 and 0 <= y <= 60):
                    break
                if not all(a[y+yy][x+xx] in (c, 13)
                           for yy in range(3) for xx in range(3)):
                    break
                length += 1
            used.add((bx, by, c))
            nodes.append(dict(c=c, bx=bx, by=by, L=length,
                              d=d, par=-1, gray=False))

        nodes = _select_arms(nodes, work)

        # Restore holes made by target diamonds in unmarked bars.
        gray = work == 1
        for _ in range(3):
            changed = False
            for yy, xx in zip(*np.nonzero((work == 13) & ~gray)):
                x, y = int(xx), int(yy)
                fill = False
                for dx, dy in ((1, 0), (0, 1)):
                    left = False
                    right = False
                    for k in range(1, 4):
                        nx, ny = x-k*dx, y-k*dy
                        if not (0 <= nx < 64 and 0 <= ny < 63):
                            break
                        if gray[ny, nx]:
                            left = True
                            break
                        if work[ny, nx] != 13:
                            break
                    for k in range(1, 4):
                        nx, ny = x+k*dx, y+k*dy
                        if not (0 <= nx < 64 and 0 <= ny < 63):
                            break
                        if gray[ny, nx]:
                            right = True
                            break
                        if work[ny, nx] != 13:
                            break
                    if left and right:
                        fill = True
                if fill:
                    gray[y, x] = True
                    changed = True
            if not changed:
                break

        for comp in _components(gray):
            xs = [p[0] for p in comp]
            ys = [p[1] for p in comp]
            x, y = min(xs), min(ys)
            w, h = max(xs)-x+1, max(ys)-y+1
            if min(w, h) > 3 or max(w, h) < 3:
                continue
            d = 1 if w >= h else 2
            length = max(1, (max(w, h)+2)//3)
            nodes.append(dict(c=1, bx=x, by=y, L=length,
                              d=d, par=-1, gray=True))

        for j, child in enumerate(nodes):
            possible = []
            for i, parent in enumerate(nodes):
                if i == j or parent["gray"]:
                    continue
                dx, dy = _DIR[parent["d"]]
                ex = parent["bx"] + 3*parent["L"]*dx
                ey = parent["by"] + 3*parent["L"]*dy
                if child["gray"]:
                    b = _initial_box(child)
                    match = (b[0] <= ex <= b[2]-3 and
                             b[1] <= ey <= b[3]-3)
                    penalty = abs(ex-child["bx"]) + abs(ey-child["by"])
                else:
                    match = (child["bx"] == ex and child["by"] == ey and
                             (child["d"]-parent["d"]) % 4 != 2)
                    penalty = 0
                if match:
                    possible.append((penalty, parent["L"], i))
            if possible:
                child["par"] = min(possible)[2]

        for i in range(len(nodes)):
            seen = {i}
            p = nodes[i]["par"]
            while p >= 0:
                if p in seen:
                    nodes[i]["par"] = -1
                    break
                seen.add(p)
                p = nodes[p]["par"]

        order, done = [], set()

        def add_node(i):
            if i in done:
                return
            p = nodes[i]["par"]
            if p >= 0:
                add_node(p)
            done.add(i)
            order.append(i)

        for i in range(len(nodes)):
            add_node(i)
        remap = {old: new for new, old in enumerate(order)}
        self.nodes = [nodes[i] for i in order]
        for n in self.nodes:
            if n["par"] >= 0:
                n["par"] = remap[n["par"]]

        state = []
        for n in self.nodes:
            p = n["par"]
            if p < 0:
                n["off"] = (0, 0)
                rel = n["d"]
            else:
                pn = self.nodes[p]
                dx, dy = _DIR[pn["d"]]
                tx = pn["bx"] + 3*(pn["L"]-1)*dx
                ty = pn["by"] + 3*(pn["L"]-1)*dy
                n["off"] = _rot(n["bx"]-tx, n["by"]-ty, -pn["d"])
                rel = (n["d"]-pn["d"]) & 3
            n["offs"] = tuple(_rot(*n["off"], d) for d in range(4))
            state.append(4*n["L"] + rel)
        self.state = tuple(state)

        self.markers = []
        for px, py in points:
            owner = -1
            for i, n in enumerate(self.nodes):
                b = _initial_box(n)
                if b[0] <= px+1 < b[2] and b[1] <= py+1 < b[3]:
                    owner = i
                    break
            if owner >= 0:
                n = self.nodes[owner]
                dx, dy = _DIR[n["d"]]
                tx = n["bx"] + 3*(n["L"]-1)*dx
                ty = n["by"] + 3*(n["L"]-1)*dy
                off = _rot(px-tx, py-ty, -n["d"])
                self.markers.append(
                    (owner, tuple(_rot(*off, d) for d in range(4))))
            else:
                self.markers.append((-1, (px, py)))

        self.walls = _rectangles(work == 15)
        self.static_image = np.full((64, 64), 5, dtype=np.int8)
        self.static_image[work == 15] = 15

        self.resizable, self.rotatable = set(), set()
        for x, y, w, h, c, resize in boxes:
            ids = tuple(i for i, n in enumerate(self.nodes) if n["c"] == c)
            if not ids:
                continue
            if resize:
                self.resizable.update(ids)
                if w > h:
                    minus = (y+h//2, x+1)
                    plus = (y+h//2, x+w-2)
                else:
                    minus = (y+1, x+w//2)
                    plus = (y+h-2, x+w//2)
                self.controls.append(("len", -1, ids, minus))
                self.controls.append(("len", 1, ids, plus))
            else:
                self.rotatable.update(ids)
                self.controls.append(("rot", 0, ids, (y+h//2, x+w//2)))

        self.dependencies = []
        for owner, _ in self.markers:
            deps = set()
            while owner >= 0:
                if owner in self.resizable or owner in self.rotatable:
                    deps.add(self.nodes[owner]["c"])
                owner = self.nodes[owner]["par"]
            self.dependencies.append(deps)

    def poses(self, state):
        poses = []
        for i, n in enumerate(self.nodes):
            length, rel = state[i]//4, state[i] & 3
            p = n["par"]
            if p < 0:
                bx, by, d = n["bx"], n["by"], rel
            else:
                pbx, pby, plen, pd, _ = poses[p]
                dx, dy = _DIR[pd]
                ox, oy = n["offs"][pd]
                bx = pbx + 3*(plen-1)*dx + ox
                by = pby + 3*(plen-1)*dy + oy
                d = (pd+rel) & 3
            dx, dy = _DIR[d]
            tx, ty = bx + 3*(length-1)*dx, by + 3*(length-1)*dy
            box = (min(bx, tx), min(by, ty),
                   max(bx, tx)+3, max(by, ty)+3)
            poses.append((bx, by, length, d, box))
        return poses

    def collision(self, poses):
        boxes = [p[4] for p in poses]
        for i, b in enumerate(boxes):
            for wall in self.walls:
                if _overlap(b, wall):
                    return True
            for j in range(i):
                if _overlap(b, boxes[j]):
                    return True
        return False

    def move(self, state, action):
        kind, amount, ids, _ = self.controls[action]
        new = list(state)
        changed = False
        for i in ids:
            length, rel = new[i]//4, new[i] & 3
            if kind == "len":
                nl = max(1, length+amount)
                if nl > max(21, self.nodes[i]["L"]):
                    return None
                changed |= nl != length
                new[i] = 4*nl + rel
            else:
                nr = (rel-1) & 3
                if self.nodes[i]["par"] >= 0 and nr == 2:
                    nr = (nr-1) & 3
                new[i] = 4*length + nr
                changed = True
        return tuple(new) if changed else None

    def point_positions(self, poses):
        result = []
        for owner, offs in self.markers:
            if owner < 0:
                result.append(offs)
            else:
                bx, by, length, d, _ = poses[owner]
                dx, dy = _DIR[d]
                ox, oy = offs[d]
                result.append((bx + 3*(length-1)*dx + ox,
                               by + 3*(length-1)*dy + oy))
        return result

    def solved(self, poses):
        points = set(self.point_positions(poses))
        return bool(self.goals) and all(g in points for g in self.goals)

    def render(self, state):
        frame = self.static_image.copy()
        poses = self.poses(state)
        for i, pose in enumerate(poses):
            bx, by, length, d, b = pose
            x0, y0, x1, y1 = b
            xx0, yy0 = max(0, x0), max(0, y0)
            xx1, yy1 = min(64, x1), min(64, y1)
            if xx0 < xx1 and yy0 < yy1:
                frame[yy0:yy1, xx0:xx1] = self.nodes[i]["c"]
            if not self.nodes[i]["gray"]:
                if d == 0:
                    coords = [(bx+k, by+2) for k in range(3)]
                elif d == 1:
                    coords = [(bx, by+k) for k in range(3)]
                elif d == 2:
                    coords = [(bx+k, by) for k in range(3)]
                else:
                    coords = [(bx+2, by+k) for k in range(3)]
                for x, y in coords:
                    if 0 <= x < 64 and 0 <= y < 64:
                        frame[y, x] = 3
        for x, y in self.point_positions(poses):
            if 0 <= x+1 < 64 and 0 <= y+1 < 64:
                frame[y+1, x+1] = 13
        frame[self.ui_mask] = self.original[self.ui_mask]
        for x, y in self.goals:
            for xx, yy in ((x+1, y), (x, y+1), (x+2, y+1), (x+1, y+2)):
                if 0 <= xx < 64 and 0 <= yy < 64:
                    frame[yy, xx] = 13
        return frame

    def learn_boundary(self, previous, predicted):
        before = self.poses(previous)
        after = self.poses(predicted)
        additions = []
        for old, new in zip(before, after):
            if old[4] == new[4]:
                continue
            x0, y0, x1, y1 = new[4]
            ox0, oy0, ox1, oy1 = old[4]
            if x0 < 0 and ox0 >= 0:
                additions.append((-3, y0, 0, y1))
            if x1 > 64 and ox1 <= 64:
                additions.append((64, y0, 67, y1))
            if y0 < 0 and oy0 >= 0:
                additions.append((x0, -3, x1, 0))
            if y1 > 64 and oy1 <= 64:
                additions.append((x0, 64, x1, 67))
        for wall in additions:
            if wall not in self.walls:
                self.walls.append(wall)


class _GoalHeuristic:
    def __init__(self, model, marker, target):
        self.model = model
        self.target = target
        self.pdb = None
        self.cache = {}
        self.lvars, self.rvars = [], []
        self.choices = [()]
        self.chain = []

        owner, marker_offs = model.markers[marker]
        self.owner = owner
        self.marker_offs = marker_offs
        if owner < 0:
            self.constant = 0.0 if marker_offs == target else _INF
            self.feasible = self.constant == 0
            return

        self.constant = None
        p = owner
        while p >= 0:
            self.chain.append(p)
            p = model.nodes[p]["par"]
        self.chain.reverse()

        self.lvars = [i for i in self.chain if i in model.resizable]
        self.rvars = [i for i in self.chain if i in model.rotatable]
        self.lindex = {i: j for j, i in enumerate(self.lvars)}
        self.rindex = {i: j for j, i in enumerate(self.rvars)}
        self.rchoices = [
            (0, 1, 2, 3) if model.nodes[i]["par"] < 0 else (0, 1, 3)
            for i in self.rvars
        ]
        self.choices = (list(itertools.product(*self.rchoices))
                        if self.rchoices else [()])
        q, nl = len(self.choices), len(self.lvars)
        self.ax = np.zeros((q, nl), dtype=np.int16)
        self.ay = np.zeros((q, nl), dtype=np.int16)
        self.cx = np.zeros(q, dtype=np.int16)
        self.cy = np.zeros(q, dtype=np.int16)
        self.directions = []

        for k, choice in enumerate(self.choices):
            dirs = []
            cx = model.nodes[self.chain[0]]["bx"]
            cy = model.nodes[self.chain[0]]["by"]
            prev_d = None
            for i in self.chain:
                n = model.nodes[i]
                rel = (choice[self.rindex[i]] if i in self.rindex
                       else model.state[i] & 3)
                d = rel if prev_d is None else (prev_d+rel) & 3
                if prev_d is not None:
                    ox, oy = n["offs"][prev_d]
                    cx += ox
                    cy += oy
                dx, dy = _DIR[d]
                if i in self.lindex:
                    j = self.lindex[i]
                    self.ax[k, j], self.ay[k, j] = 3*dx, 3*dy
                    cx -= 3*dx
                    cy -= 3*dy
                else:
                    length = model.state[i]//4
                    cx += 3*(length-1)*dx
                    cy += 3*(length-1)*dy
                dirs.append(d)
                prev_d = d
            ox, oy = marker_offs[dirs[-1]]
            self.cx[k], self.cy[k] = cx+ox, cy+oy
            self.directions.append(dirs)

        lo_x = self.cx + np.minimum(self.ax, self.ax*21).sum(axis=1)
        hi_x = self.cx + np.maximum(self.ax, self.ax*21).sum(axis=1)
        lo_y = self.cy + np.minimum(self.ay, self.ay*21).sum(axis=1)
        hi_y = self.cy + np.maximum(self.ay, self.ay*21).sum(axis=1)
        gx, gy = target
        self.valid = ((lo_x <= gx) & (gx <= hi_x) &
                      (lo_y <= gy) & (gy <= hi_y) &
                      ((gx-self.cx) % 3 == 0) &
                      ((gy-self.cy) % 3 == 0))
        self.feasible = bool(np.any(self.valid))
        self.rotation_costs = []
        for j, node in enumerate(self.rvars):
            table = np.full((4, q), 20, dtype=np.int16)
            for start in range(4):
                cur = start
                for distance in range(5):
                    for k, ch in enumerate(self.choices):
                        if ch[j] == cur:
                            table[start, k] = min(table[start, k], distance)
                    cur = (cur-1) & 3
                    if model.nodes[node]["par"] >= 0 and cur == 2:
                        cur = (cur-1) & 3
            self.rotation_costs.append(table)

    def value(self, state):
        if self.constant is not None:
            return self.constant
        if not self.feasible:
            return _INF
        if self.pdb is not None:
            try:
                idx = tuple(state[i]//4-1 for i in self.lvars)
                idx += tuple(self.rchoices[j].index(state[i] & 3)
                             for j, i in enumerate(self.rvars))
                return float(self.pdb[idx])
            except (ValueError, IndexError):
                pass

        key = tuple(state[i] for i in self.chain)
        if key in self.cache:
            return self.cache[key]
        lengths = np.fromiter((state[i]//4 for i in self.lvars),
                              dtype=np.int16, count=len(self.lvars))
        px = self.cx + self.ax.dot(lengths)
        py = self.cy + self.ay.dot(lengths)
        costs = (np.abs(px-self.target[0]) +
                 np.abs(py-self.target[1])).astype(np.float32)/3.0
        for j, i in enumerate(self.rvars):
            costs += self.rotation_costs[j][state[i] & 3]
        costs[~self.valid] = _INF
        value = float(costs.min())
        if len(self.cache) < 150000:
            self.cache[key] = value
        return value

    def build_pdb(self, deadline):
        if self.constant is not None or not self.feasible:
            return
        nl, q = len(self.lvars), len(self.choices)
        count = 21**nl
        if nl > 4 or count*q > 8000000 or time.perf_counter() >= deadline:
            return

        if nl:
            grid = np.indices((21,)*nl, dtype=np.int16)
            lengths = grid.reshape(nl, -1).T + 1
        else:
            lengths = np.zeros((1, 0), dtype=np.int16)

        flat = np.full((count, q), 110, dtype=np.uint8)
        model = self.model
        found = False
        gx, gy = self.target

        for k, dirs in enumerate(self.directions):
            if time.perf_counter() >= deadline:
                return
            if not self.valid[k]:
                continue
            px = self.cx[k] + lengths.dot(self.ax[k])
            py = self.cy[k] + lengths.dot(self.ay[k])
            ids = np.flatnonzero((px == gx) & (py == gy))
            if not len(ids):
                continue

            lens = lengths[ids]
            valid = np.ones(len(ids), dtype=bool)
            boxes = []
            bx = np.full(len(ids), model.nodes[self.chain[0]]["bx"],
                         dtype=np.int16)
            by = np.full(len(ids), model.nodes[self.chain[0]]["by"],
                         dtype=np.int16)
            previous_length = 1

            for j, i in enumerate(self.chain):
                n, d = model.nodes[i], dirs[j]
                if j:
                    pd = dirs[j-1]
                    dx, dy = _DIR[pd]
                    ox, oy = n["offs"][pd]
                    bx = bx + 3*(previous_length-1)*dx + ox
                    by = by + 3*(previous_length-1)*dy + oy
                length = (lens[:, self.lindex[i]] if i in self.lindex
                          else model.state[i]//4)
                dx, dy = _DIR[d]
                tx = bx + 3*(length-1)*dx
                ty = by + 3*(length-1)*dy
                b = (np.minimum(bx, tx), np.minimum(by, ty),
                     np.maximum(bx, tx)+3, np.maximum(by, ty)+3)
                for w in model.walls:
                    valid &= ~((b[0] < w[2]) & (w[0] < b[2]) &
                               (b[1] < w[3]) & (w[1] < b[3]))
                for old in boxes:
                    valid &= ~((b[0] < old[2]) & (old[0] < b[2]) &
                               (b[1] < old[3]) & (old[1] < b[3]))
                boxes.append(b)
                previous_length = length

            good = ids[valid]
            if len(good):
                flat[good, k] = 0
                found = True

        if not found:
            self.feasible = False
            return

        shape = (21,)*nl + tuple(len(c) for c in self.rchoices)
        if not shape:
            self.pdb = flat.reshape(())
            return
        table = flat.reshape(shape)

        for axis in range(nl):
            if time.perf_counter() >= deadline:
                return
            for j in range(1, 21):
                dst = [slice(None)]*table.ndim
                src = dst.copy()
                dst[axis], src[axis] = j, j-1
                table[tuple(dst)] = np.minimum(
                    table[tuple(dst)], table[tuple(src)]+1)
            for j in range(19, -1, -1):
                dst = [slice(None)]*table.ndim
                src = dst.copy()
                dst[axis], src[axis] = j, j+1
                table[tuple(dst)] = np.minimum(
                    table[tuple(dst)], table[tuple(src)]+1)

        for r, choices in enumerate(self.rchoices):
            if time.perf_counter() >= deadline:
                return
            axis, size = nl+r, len(choices)
            for _ in range(size-1):
                for j in range(size):
                    dst = [slice(None)]*table.ndim
                    src = dst.copy()
                    dst[axis], src[axis] = j, (j-1) % size
                    table[tuple(dst)] = np.minimum(
                        table[tuple(dst)], table[tuple(src)]+1)
        self.pdb = table


def _prepare_heuristics(model, deadline):
    nm, ng = len(model.markers), len(model.goals)
    if not nm or not ng or nm < ng:
        model.heuristics, model.assignments = {}, []
        return

    hs = {}
    for m in range(nm):
        for g in range(ng):
            hs[m, g] = _GoalHeuristic(model, m, model.goals[g])

    assignments = []
    if nm <= 7:
        for assignment in itertools.permutations(range(nm), ng):
            if all(hs[assignment[g], g].feasible for g in range(ng)):
                assignments.append(assignment)
    if not assignments:
        assignments = [tuple(range(ng))]

    used = {(assignment[g], g)
            for assignment in assignments for g in range(ng)}
    ordered = sorted(used, key=lambda p:
                     (21**len(hs[p].lvars))*len(hs[p].choices))
    for pair in ordered:
        if time.perf_counter() >= deadline:
            break
        hs[pair].build_pdb(deadline)

    model.heuristics = hs
    model.assignments = assignments


def _heuristic(model, state):
    if not model.assignments:
        points = model.point_positions(model.poses(state))
        if not points or not model.goals:
            return 100.0
        return sum(min(abs(x-gx)+abs(y-gy) for x, y in points)/3
                   for gx, gy in model.goals)

    best, values = _INF, {}
    for assignment in model.assignments:
        costs = []
        for g, marker in enumerate(assignment):
            key = marker, g
            if key not in values:
                values[key] = model.heuristics[key].value(state)
            costs.append(values[key])
        if any(c >= _INF for c in costs):
            continue
        shared = False
        for a in range(len(assignment)):
            for b in range(a):
                if (model.dependencies[assignment[a]] &
                        model.dependencies[assignment[b]]):
                    shared = True
        value = (max(costs)+0.30*(sum(costs)-max(costs))
                 if shared else sum(costs))
        best = min(best, value)
    return best


def _search(model, deadline):
    start = model.state
    if not model.controls:
        return []
    start_h = _heuristic(model, start)
    weight = 2.0
    heap = [(weight*start_h, start_h, 0, 0, start)]
    distance = {start: 0}
    parent = {}
    serial = 0
    best_state = start
    best_score = start_h + 5.0*model.visits.get(start, 0)
    fallback_state = None
    fallback_score = float("inf")
    expansions = 0
    solved = None

    while heap:
        if expansions % 64 == 0 and time.perf_counter() >= deadline:
            break
        _, h, g, _, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        expansions += 1
        if h < 0.001 and model.solved(model.poses(state)):
            solved = state
            break

        for action in range(len(model.controls)):
            if (state, action) in model.bad:
                continue
            new = model.move(state, action)
            if new is None:
                continue
            ng = g+1
            if ng >= distance.get(new, 10**9):
                continue
            poses = model.poses(new)
            if model.collision(poses):
                continue
            nh = _heuristic(model, new)
            if nh >= _INF:
                continue
            distance[new] = ng
            parent[new] = state, action
            if model.solved(poses):
                solved = new
                break
            penalty = 4.0*model.visits.get(new, 0)
            score = nh + 0.035*ng + penalty
            if score < best_score:
                best_score, best_state = score, new
            if score < fallback_score:
                fallback_score, fallback_state = score, new
            serial += 1
            heapq.heappush(heap, (ng+weight*nh+penalty,
                                  nh, ng, serial, new))
        if solved is not None:
            break
        if len(distance) > 270000:
            break

    destination = solved
    if destination is None:
        destination = best_state if best_state != start else fallback_state
    if destination is None:
        return []
    path = []
    while destination != start:
        previous, action = parent[destination]
        path.append(action)
        destination = previous
    path.reverse()
    return path if solved is not None else path[:6]


def _reset(memory):
    resets = memory.get("_resets", 0)
    memory.clear()
    memory["_resets"] = resets+1
    return "RESET", None, None


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started+8.5
    obs = np.asarray(obs, dtype=np.int8)
    memory["_ticks"] = memory.get("_ticks", 0)+1
    record = memory.get("_s5_player")
    model = None

    if record is not None:
        model = record["model"]
        pending = record.get("pending")
        if pending is not None:
            if np.array_equal(model.render(pending)[:63], obs[:63]):
                model.state = pending
                record["stalls"] = 0
            elif np.array_equal(record["last_frame"], obs[:63]):
                previous = record["previous"]
                model.state = previous
                model.bad.add((previous, record["action"]))
                model.learn_boundary(previous, pending)
                model.plan = []
                record["stalls"] = record.get("stalls", 0)+1
            else:
                record["mismatches"] = record.get("mismatches", 0)+1
                model = None
        elif record.get("fallback", False):
            if memory.get("_resets", 0) < 2:
                return _reset(memory)
            model = None

        if (record.get("stalls", 0) >= 4 or
                record.get("mismatches", 0) >= 3):
            if memory.get("_resets", 0) < 2:
                return _reset(memory)

    if model is None:
        old_mismatches = 0 if record is None else record.get("mismatches", 0)
        model = _Model(obs)
        record = {"model": model, "stalls": 0,
                  "mismatches": old_mismatches}
        memory["_s5_player"] = record
        _prepare_heuristics(model, min(deadline-1.0, started+4.2))

    model.visits[model.state] = model.visits.get(model.state, 0)+1
    if not model.plan:
        model.plan = _search(model, deadline)

    chosen = None
    while model.plan:
        action = model.plan.pop(0)
        new = model.move(model.state, action)
        if (new is not None and (model.state, action) not in model.bad and
                not model.collision(model.poses(new))):
            chosen = action
            break

    if chosen is None:
        best = float("inf")
        for action in range(len(model.controls)):
            if (model.state, action) in model.bad:
                continue
            new = model.move(model.state, action)
            if new is None or model.collision(model.poses(new)):
                continue
            value = _heuristic(model, new)+5*model.visits.get(new, 0)
            if value < best:
                best, chosen = value, action

    if chosen is None:
        if memory["_ticks"] > 1 and memory.get("_resets", 0) < 2:
            return _reset(memory)
        record["pending"] = None
        record["fallback"] = True
        record["last_frame"] = obs[:63].copy()
        if model.controls:
            index = memory["_ticks"] % len(model.controls)
            row, col = model.controls[index][3]
        else:
            row, col = 0, 0
        return "ACTION6", int(row), int(col)

    record["fallback"] = False
    previous = model.state
    record["previous"] = previous
    record["pending"] = model.move(previous, chosen)
    record["action"] = chosen
    record["last_frame"] = obs[:63].copy()
    row, col = model.controls[chosen][3]
    return "ACTION6", int(row), int(col)

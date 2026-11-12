import numpy as np
import math
import time
import heapq
from collections import deque

ALL = (1 << 1024) - 1
LEFT = sum(1 << (32 * y) for y in range(32))
RIGHT = LEFT << 31
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ROLES = ("a", "b", "d", "e", "f")


def bits(n):
    n = int(n)
    while n:
        b = n & -n
        yield b.bit_length() - 1
        n ^= b


def neighbors(n):
    n = int(n)
    return (((n & ~RIGHT) << 1) | ((n & ~LEFT) >> 1) |
            (n << 32) | (n >> 32)) & ALL


def flood(p, valid):
    r = 1 << int(p)
    while True:
        q = r | (neighbors(r) & valid)
        if q == r:
            return r
        r = q


def components(mask, diagonal=False):
    h, w = mask.shape
    remaining = set(int(z) for z in np.flatnonzero(mask))
    result = []
    ds = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        ds += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    while remaining:
        stack = [remaining.pop()]
        pts = []
        while stack:
            z = stack.pop()
            y, x = divmod(z, w)
            pts.append((x, y))
            for dx, dy in ds:
                xx, yy = x + dx, y + dy
                q = yy * w + xx
                if 0 <= xx < w and 0 <= yy < h and q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(pts)
    return result


def bbox(pts):
    xs, ys = zip(*pts)
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


def rect(x0, y0, x1, y1):
    return [(x, y) for y in range(int(y0), int(y1))
            for x in range(int(x0), int(x1))]


def pointmask(m, pts, collision=False):
    out = 0
    ox, oy = m["offset"]
    for x, y in pts:
        x, y = int(x), int(y)
        if not collision and ((x - ox) % 2 or (y - oy) % 2):
            continue
        xx, yy = (x - ox) // 2, (y - oy) // 2
        if 0 <= xx < 32 and 0 <= yy < 32 and ox + 2 * xx < m["boundary"]:
            out |= 1 << (32 * yy + xx)
    return out


def pos(m, x, y):
    x, y = int(x), int(y)
    ox, oy = m["offset"]
    return 32 * ((y - oy) // 2) + (x - ox) // 2


def xy(m, p):
    p = int(p)
    ox, oy = m["offset"]
    return ox + 2 * (p % 32), oy + 2 * (p // 32)


def translate(mask, dx, dy, boundary):
    dx, dy = int(dx) // 2, int(dy) // 2
    out = 0
    for p in bits(mask):
        y, x = divmod(p, 32)
        x, y = x + dx, y + dy
        if 0 <= x < 32 and 0 <= y < 32 and 2 * x < boundary:
            out |= 1 << (32 * y + x)
    return out


def find_boundary(a):
    rows = np.where(np.any(a[:, :24] == 4, axis=1))[0]
    if not len(rows):
        return 38
    for x in range(22, 60):
        c = a[rows, x]
        d = a[rows, min(63, x + 2)]
        if np.mean((c == 0) | (c == 5)) > .86 and np.mean(d == 5) > .80:
            return x
    return 38


def find_player(a):
    ys, xs = np.where(a == 14)
    if len(xs):
        return int(xs.min()), int(ys.min())
    return None


def controls(a, boundary):
    result = {}
    small9, small10 = [], []
    names = {8: "a", 9: "b", 6: "c", 15: "d", 2: "e", 11: "f"}
    for color in (8, 9, 6, 15, 2, 11, 10):
        mask = a == color
        mask[:, :boundary] = False
        mask[63, :] = False
        for pts in components(mask):
            x0, y0, x1, y1 = bbox(pts)
            w, h = x1 - x0, y1 - y0
            x, y = pts[len(pts) // 2]
            click = (int(y), int(x))
            if w >= 11 and 3 <= h <= 8 and color in names:
                result[names[color]] = click
            elif color == 2 and w >= 5 and h >= 5:
                result["e"] = click
            elif color == 11 and w >= 5 and h >= 5:
                result["f"] = click
            elif color == 8 and 7 <= w <= 10 and h <= 4:
                result["grab"] = click
            elif color == 10 and w <= 5 and h <= 5 and len(pts) >= 8:
                small10.append((x0, click))
            elif color == 9 and w <= 5 and h <= 5 and len(pts) >= 8:
                small9.append((x0, click))
    if len(small10) >= 2:
        small10.sort()
        result["left"], result["right"] = small10[0][1], small10[-1][1]
    if len(small9) >= 2:
        small9.sort()
        result["up"], result["down"] = small9[0][1], small9[-1][1]
    return result


OPEN_CLAW = (
    (3, 4, 5, 6, 7), (3, 7), (1, 2, 3, 4), (0, 1, 2, 3),
    (0, 1, 2, 3), (1, 2, 3, 4), (3, 7), (3, 4, 5, 6, 7)
)
CLOSED_CLAW = (
    (), (3, 4, 5, 6, 7), (1, 2, 3, 4, 7), (0, 1, 2, 3),
    (0, 1, 2, 3), (1, 2, 3, 4, 7), (3, 4, 5, 6, 7), ()
)


def claw_pixels(x, y, closed=False):
    pat = CLOSED_CLAW if closed else OPEN_CLAW
    return [(int(x) + dx, int(y) + dy)
            for dy, row in enumerate(pat) for dx in row]


def find_claw(a, boundary, used=None):
    mask = a == 8
    mask[:, boundary:] = False
    if used:
        for x, y in used:
            if 0 <= x < 64 and 0 <= y < 64:
                mask[y, x] = False
    candidates = []
    for pts in components(mask):
        x0, y0, x1, y1 = bbox(pts)
        w, h = x1 - x0, y1 - y0
        if w == 8 and h in (6, 8):
            closed = h == 6
            candidates.append((0, (x0, y0 - int(closed)), closed, 8, 8, pts))
        elif w >= 5 and h >= 5 and len(pts) < .8 * w * h:
            candidates.append((1, (x0, y0), False, w, h, pts))
    if candidates:
        candidates.sort(key=lambda v: v[0])
        return candidates[0]
    return None


def observed_map(a, m, claw=None):
    safe, block = 0, 0
    ox, oy = m["offset"]
    keyrects = m.get("keyrects", [])
    for cy in range(32):
        y = oy + 2 * cy
        if y < 0 or y + 1 >= 63:
            continue
        for cx in range(32):
            x = ox + 2 * cx
            if x < 0 or x + 1 >= m["boundary"]:
                continue
            bit = 1 << (32 * cy + cx)
            c = int(a[y, x])
            q = a[y:y + 2, x:x + 2]
            key = any(x0 <= x + 1 and x < x1 and y0 <= y + 1 and y < y1
                      for x0, y0, x1, y1 in keyrects)
            checker = False
            for color in (9, 15):
                if np.any(q == color) and not (key and color == 15):
                    if np.any((q == 4) | (q == 5)):
                        checker = True
            if checker:
                block |= bit
            if c in (1, 2, 6, 7, 8, 11, 12, 13, 14):
                safe |= bit
            elif c in (9, 15):
                if key or np.all((q == c) | (q == 14)):
                    safe |= bit
    if claw is not None:
        _, (x, y), closed, w, h, pts = claw
        if w == 8 and h == 8:
            pts = claw_pixels(x, y, closed)
        cb = pointmask(m, pts, True)
        block |= cb
        for p in bits(cb):
            x, y = xy(m, p)
            if 0 <= x < 64 and 0 <= y < 64 and int(a[y, x]) == 8:
                safe &= ~(1 << p)
    return safe, block


def add_entity(m, role, variants, center=None, bridge=False):
    e = {"role": role, "variants": variants,
         "center": center, "bridge": bool(bridge)}
    m["entities"].append(e)
    return len(m["entities"]) - 1


def clear_cache(m):
    m["maps"] = {}
    m["regions"] = {}
    m["moved"] = {}


def initialize(a, player):
    boundary = find_boundary(a)
    ctr = controls(a, boundary)
    m = {
        "boundary": boundary,
        "offset": (int(player[0]) % 2, int(player[1]) % 2),
        "entities": [], "groups": [], "gindex": {},
        "controls": ctr, "visible": set(ctr),
        "keyrects": [], "keymask": 0, "portals": {},
        "extra": {}, "banned": set(), "plan": [], "pending": None,
        "probes": {}, "gated": set(), "gate_positions": {},
        "visits": {}, "crane": None, "calls": 0,
    }
    board = a.copy()
    board[:, boundary:] = 4
    board[63, :] = 4

    # Unlock tokens.
    for pts in components(board == 15, True):
        x0, y0, x1, y1 = bbox(pts)
        if y1 - y0 <= 2 and x1 - x0 <= 4:
            m["keyrects"].append((x0, y0, x1, y1))
            m["keymask"] |= pointmask(m, rect(x0, y0, x1, y1), True)
            for x, y in pts:
                board[y, x] = 2

    # Paired teleporters.
    portals = []
    for pts in components((board == 6) | (board == 7)):
        x0, y0, x1, y1 = bbox(pts)
        if x1 - x0 == 2 and y1 - y0 == 2:
            portals.append((pos(m, x0, y0), int(board[y0, x0])))
    for p, color in portals:
        others = [q for q, c in portals if q != p and c != color]
        if not others:
            others = [q for q, c in portals if q != p]
        if others:
            m["portals"][p] = others[0]

    # Pivoted blue bridges.
    used_blue = set()
    for pts in components((board == 8) | (board == 13) | (board == 14)):
        piv = [(x, y) for x, y in pts if int(board[y, x]) == 13]
        if not piv:
            continue
        x0, y0, x1, y1 = bbox(piv)
        cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
        body = [(x, y) for x, y in pts if int(board[y, x]) in (8, 13, 14)]
        if len(body) < 8:
            continue
        used_blue.update(body)
        bx0, by0, bx1, by1 = bbox(body)
        horizontal = bx1 - bx0 >= by1 - by0

        def rotate(ps, sign):
            return [(int(round(cx - sign * (y - cy))),
                     int(round(cy + sign * (x - cx)))) for x, y in ps]

        variants = [(pointmask(m, body), 0)]
        if bx1 - bx0 <= 4 or by1 - by0 <= 4:
            variants.append((pointmask(m, rotate(body, 1 if horizontal else -1)), 0))
        else:
            current = body
            for _ in range(3):
                current = rotate(current, 1)
                variants.append((pointmask(m, current), 0))
        add_entity(m, "a", variants,
                   (int(round(cx + .5)), int(round(cy + .5))), True)

    # Solid/checkered platforms.
    for color, role in ((9, "b"), (15, "d")):
        for pts in components(board == color, True):
            x0, y0, x1, y1 = bbox(pts)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            area = (x1 - x0) * (y1 - y0)
            hidden = int(np.count_nonzero(board[y0:y1, x0:x1] == 14))
            solid = len(pts) + hidden > .72 * area
            support = pointmask(m, rect(x0, y0, x1, y1))
            collision = pointmask(m, rect(x0, y0, x1, y1), True)
            variants = [(support, 0), (0, collision)]
            if not solid:
                variants.reverse()
            add_entity(m, role, variants)

    # Thin expanding and translating mechanisms.
    thin = []
    for pts in components((board == 1) | (board == 12) | (board == 5)):
        x0, y0, x1, y1 = bbox(pts)
        if y1 - y0 <= 2 and x1 - x0 >= 2:
            thin.append((x0, y0, x1, y1, pts))
    rows = {}
    for item in thin:
        rows.setdefault((item[1], item[3]), []).append(item)

    expanding_spans, movers = [], []
    for (y0, y1), items in rows.items():
        items.sort()
        consumed = set()
        for i, item in enumerate(items):
            if i in consumed:
                continue
            x0, _, x1, _, pts = item
            joined = list(pts)
            indices = [i]
            for j in range(i + 1, len(items)):
                z = items[j]
                if z[2] - x0 <= 16 and z[0] - x1 <= 12:
                    joined += z[4]
                    x1 = z[2]
                    indices.append(j)
            width = x1 - x0
            has5 = any(int(board[y, x]) == 5 for x, y in joined)
            if width >= 12 or (has5 and width >= 8):
                consumed.update(indices)
                expanding_spans.append((x0, y0, x1, y1))
                count = sum(int(board[y0, x]) in (1, 12) for x in range(x0, x1))
                if any(int(board[y, x]) == 12 for x, y in joined):
                    phase = 4
                else:
                    phase = min(3, int(round(count / 4)))
                variants = []
                for k in range(8):
                    r = (phase + k) % 8
                    n = min(width // 2, 2 * min(r, 8 - r))
                    if r == 4:
                        ps = rect(x0, y0, x1, y1)
                    else:
                        ps = rect(x0, y0, x0 + n, y1) + rect(x1 - n, y0, x1, y1)
                    variants.append((pointmask(m, ps), 0))
                add_entity(m, "e", variants)
            else:
                consumed.add(i)
                if item[2] - item[0] == 6:
                    movers.append(item)

    for x0, y0, x1, y1, pts in movers:
        starts = [x for x, y, xx, yy in expanding_spans if abs(x - x0) <= 10]
        if starts:
            start = min(starts, key=lambda x: abs(x - x0))
        elif np.any(board[y0:y1, x0:x1] == 12):
            start = x0 - 10
        else:
            start = x0
            for x in range(max(0, x0 - 10), x0 + 1, 2):
                yy0, yy1 = max(0, y0 - 4), min(63, y1 + 2)
                if x >= 2 and np.any(board[yy0:yy1, x - 2:x] == 2):
                    start = x
                    break
        phase = max(0, min(5, (x0 - start) // 2))
        variants = []
        for k in range(6):
            xx = start + 2 * ((phase + k) % 6)
            variants.append((pointmask(m, rect(xx, y0, xx + 6, y1)), 0))
        add_entity(m, "f", variants)

    has_crane = any(r in ctr for r in ("grab", "left", "up"))
    claw = find_claw(a, boundary, used_blue) if has_crane else None
    objects = []
    if claw is not None:
        for pts in components(board == 12):
            x0, y0, x1, y1 = bbox(pts)
            if y1 - y0 > 3 and x1 - x0 >= 3:
                objects.append(add_entity(
                    m, None, [(pointmask(m, pts), 0)],
                    ((x0 + x1) // 2, (y0 + y1) // 2)
                ))

    # All objects controlled by one button share a cyclic phase.
    for role in ROLES:
        ids = [i for i, e in enumerate(m["entities"]) if e["role"] == role]
        if not ids:
            continue
        period = 1
        for i in ids:
            n = len(m["entities"][i]["variants"])
            period = period * n // math.gcd(period, n)
        period = min(24, period)
        supports, blocks = [], []
        for k in range(period):
            s, b = 0, 0
            for i in ids:
                v = m["entities"][i]["variants"]
                ss, bb = v[k % len(v)]
                s |= ss
                b |= bb
            supports.append(s)
            blocks.append(b)
        region = 0
        for value in supports + blocks:
            region |= value
        m["gindex"][role] = len(m["groups"])
        m["groups"].append({
            "role": role, "ids": ids, "period": period,
            "sup": supports, "block": blocks, "region": region,
            "override": {}
        })

    if claw is not None:
        _, origin, closed, w, h, pts = claw
        regular = w == 8 and h == 8
        anchor = (9, 4) if regular and objects else (w // 2, h // 2)
        ys, xs = np.where(a[:, :boundary] == 0)
        rail = {(int(x), int(y)) for y, x in zip(ys, xs)}
        rail.update(pts)
        center = (w // 2, h // 2)
        positions = [origin]
        index = {origin: 0}
        queue = deque([origin])
        while queue and len(positions) < 90:
            x, y = queue.popleft()
            for dx, dy in DIRS:
                q = (x + 4 * dx, y + 4 * dy)
                cc = (q[0] + center[0], q[1] + center[1])
                if (q not in index and cc in rail and
                        0 <= cc[0] < boundary and 0 <= cc[1] < 63):
                    index[q] = len(positions)
                    positions.append(q)
                    queue.append(q)
        links = [[index.get((x + 4 * dx, y + 4 * dy), -1)
                  for dx, dy in DIRS] for x, y in positions]
        if regular:
            opened = pointmask(m, claw_pixels(*origin, False), True)
            closedmask = pointmask(m, claw_pixels(*origin, True), True)
        else:
            opened = closedmask = pointmask(m, pts, True)
        candidates = objects or [
            i for i, e in enumerate(m["entities"]) if e["bridge"]
        ]
        m["crane"] = {
            "positions": positions, "index": index, "links": links,
            "anchor": anchor, "origin": origin,
            "open": opened, "closed": closedmask,
            "candidates": candidates, "used_blue": used_blue
        }

    support, block = observed_map(a, m, claw)
    dynamic = 0
    for e in m["entities"]:
        dynamic |= e["variants"][0][0] | e["variants"][0][1]
    m["base"] = support & ~dynamic
    ys, xs = np.where(np.isin(board, (2, 6, 7, 11)))
    m["base"] |= pointmask(m, [(int(x), int(y)) for y, x in zip(ys, xs)])

    goals = [z for z in components(board == 11) if len(z) <= 16]
    if goals:
        pts = min(goals, key=lambda z: abs(len(z) - 4))
        gx, gy, _, _ = bbox(pts)
        goal = pos(m, gx, gy)
    else:
        goal = pos(m, *player)
    m["goal"] = goal
    m["base"] |= 1 << goal

    p = pos(m, *player)
    if not support & (1 << p):
        m["base"] |= 1 << p

    cp, attached = -1, -1
    if m["crane"]:
        cp = 0
        if claw[2]:
            cr = m["crane"]
            anchor = (origin[0] + cr["anchor"][0], origin[1] + cr["anchor"][1])
            for i in cr["candidates"]:
                if m["entities"][i]["center"] == anchor:
                    attached = i
                    break
    unlocked = "d" in ctr or not m["keyrects"]
    m["state"] = (p, tuple(0 for _ in m["groups"]), cp, attached, bool(unlocked))
    clear_cache(m)
    return m


def entity_at(m, eid, phase, cp, attached):
    e = m["entities"][eid]
    phase %= len(e["variants"])
    s, b = e["variants"][phase]
    if eid != attached or cp < 0:
        return s, b
    key = (eid, phase, cp)
    if key in m["moved"]:
        return m["moved"][key]
    cr = m["crane"]
    x, y = cr["positions"][cp]
    ax, ay = cr["anchor"]
    cx, cy = e["center"]
    dx, dy = x + ax - cx, y + ay - cy
    result = (translate(s, dx, dy, m["boundary"]),
              translate(b, dx, dy, m["boundary"]))
    m["moved"][key] = result
    return result


def world_map(m, phases, cp, attached):
    key = (phases, cp, attached)
    if key in m["maps"]:
        return m["maps"][key]
    safe, block = m["base"], 0
    for gi, g in enumerate(m["groups"]):
        k = phases[gi] % g["period"]
        if attached in g["ids"]:
            s, b = 0, 0
            for eid in g["ids"]:
                ss, bb = entity_at(m, eid, k, cp, attached)
                s |= ss
                b |= bb
        elif k in g["override"]:
            s, b = g["override"][k]
        else:
            s, b = g["sup"][k], g["block"][k]
        safe |= s
        block |= b
    for eid, e in enumerate(m["entities"]):
        if e["role"] is None:
            s, b = entity_at(m, eid, 0, cp, attached)
            safe |= s
            block |= b
    if cp >= 0:
        cr = m["crane"]
        x, y = cr["positions"][cp]
        ox, oy = cr["origin"]
        claw = cr["closed"] if attached >= 0 else cr["open"]
        block |= translate(claw, x - ox, y - oy, m["boundary"])
    block |= m["extra"].get(key, 0)
    result = (safe, block, safe & ~block)
    m["maps"][key] = result
    return result


def can_use(m, role, p, unlocked):
    if role == "d" and unlocked and m["keyrects"]:
        return True
    if role in m["gated"]:
        return p in m["gate_positions"].get(role, set())
    return role in m["visible"]


def optimistic_distances(m):
    union = m["base"]
    for g in m["groups"]:
        for s in g["sup"]:
            union |= s
        for s, b in g["override"].values():
            union |= s
    cr = m["crane"]
    for eid, e in enumerate(m["entities"]):
        if e["role"] is None:
            union |= e["variants"][0][0]
        if cr and eid in cr["candidates"]:
            for k in range(len(e["variants"])):
                for cp in range(len(cr["positions"])):
                    union |= entity_at(m, eid, k, cp, eid)[0]
    goal = m["goal"]
    dist = {goal: 0}
    reverse = {}
    for p, q in m["portals"].items():
        reverse.setdefault(q, []).append(p)
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        x, y = p % 32, p // 32
        ns = []
        if x:
            ns.append(p - 1)
        if x < 31:
            ns.append(p + 1)
        if y:
            ns.append(p - 32)
        if y < 31:
            ns.append(p + 32)
        ns.extend(reverse.get(p, ()))
        for q in ns:
            if q not in dist and union & (1 << q):
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def bridge_collision(m, phases, cp, attached):
    if attached < 0 or not m["entities"][attached]["bridge"]:
        return False
    e = m["entities"][attached]
    gi = m["gindex"].get(e["role"])
    k = phases[gi] if gi is not None else 0
    moving = entity_at(m, attached, k, cp, attached)[0]
    for eid, other in enumerate(m["entities"]):
        if eid == attached or not other["bridge"]:
            continue
        gj = m["gindex"].get(other["role"])
        phase = phases[gj] if gj is not None else 0
        if moving & entity_at(m, eid, phase, cp, -1)[0]:
            return True
    return False


def search(m, seconds=5.8):
    deadline = time.perf_counter() + seconds
    start, goal = m["state"], m["goal"]
    dist = optimistic_distances(m)
    gx, gy = goal % 32, goal // 32

    def heuristic(p):
        return dist.get(p, 35 + abs(p % 32 - gx) + abs(p // 32 - gy))

    heap = [(1.6 * heuristic(start[0]), 0, 0, start)]
    costs = {start: 0}
    parent = {}
    serial = 0
    best = start
    bestscore = heuristic(start[0])
    expanded = 0
    found = None

    def push(old, new, action, cost):
        nonlocal serial
        if (old, action) in m["banned"]:
            return
        if cost >= costs.get(new, 10**9):
            return
        costs[new] = cost
        parent[new] = (old, action)
        serial += 1
        heapq.heappush(heap, (
            cost + 1.6 * heuristic(new[0]), cost, serial, new
        ))

    while heap:
        _, cost, _, state = heapq.heappop(heap)
        if costs.get(state) != cost:
            continue
        p, phases, cp, attached, unlocked = state
        if p == goal:
            found = state
            break
        score = heuristic(p) + .025 * cost
        if score < bestscore:
            bestscore, best = score, state
        expanded += 1
        if expanded % 256 == 0:
            if time.perf_counter() >= deadline or expanded >= 200000:
                break

        safe, block, valid = world_map(m, phases, cp, attached)
        x, y = p % 32, p // 32
        for ai, (dx, dy) in enumerate(DIRS, 1):
            xx, yy = x + dx, y + dy
            if not (0 <= xx < 32 and 0 <= yy < 32):
                continue
            q = 32 * yy + xx
            if valid & (1 << q):
                key = unlocked or bool(m["keymask"] & (1 << q))
                push(state, (q, phases, cp, attached, key), ai, cost + 1)

        cache_key = (phases, cp, attached)
        regions = m["regions"].setdefault(cache_key, [])
        reachable = None
        for region in regions:
            if region & (1 << p):
                reachable = region
                break
        if reachable is None:
            reachable = flood(p, valid | (1 << p))
            regions.append(reachable)
        nearby = reachable | neighbors(reachable)

        for gi, g in enumerate(m["groups"]):
            role = g["role"]
            if not can_use(m, role, p, unlocked):
                continue
            if role not in m["gated"] and not (g["region"] & nearby):
                if attached not in g["ids"]:
                    continue
            ph = list(phases)
            ph[gi] = (ph[gi] + 1) % g["period"]
            ph = tuple(ph)
            nextsafe = world_map(m, ph, cp, attached)[0]
            if nextsafe & (1 << p):
                push(state, (p, ph, cp, attached, unlocked), role, cost + 1)

        if p in m["portals"] and can_use(m, "c", p, unlocked):
            q = m["portals"][p]
            if safe & (1 << q):
                push(state, (q, phases, cp, attached, unlocked), "c", cost + 1)

        if cp >= 0:
            cr = m["crane"]
            for ai, role in enumerate(("up", "down", "left", "right")):
                if role not in m["controls"]:
                    continue
                nc = cr["links"][cp][ai]
                if nc < 0 or bridge_collision(m, phases, nc, attached):
                    continue
                nextsafe = world_map(m, phases, nc, attached)[0]
                if nextsafe & (1 << p):
                    push(state, (p, phases, nc, attached, unlocked), role, cost + 1)
            if attached < 0 and "grab" in m["controls"]:
                x, y = cr["positions"][cp]
                ax, ay = cr["anchor"]
                for eid in cr["candidates"]:
                    if m["entities"][eid]["center"] == (x + ax, y + ay):
                        if world_map(m, phases, cp, eid)[0] & (1 << p):
                            push(state, (p, phases, cp, eid, unlocked), "grab", cost + 1)

    target = found if found is not None else best
    path = []
    while target != start and target in parent:
        previous, action = parent[target]
        path.append((action, target))
        target = previous
    path.reverse()
    if found is None:
        path = path[:12]
    m["last_search_solved"] = found is not None
    return path


def learn_switch(m, old, new, role, before, after, afterblock):
    gi = m["gindex"].get(role)
    if gi is None:
        return
    g = m["groups"][gi]
    if old[3] in g["ids"]:
        return
    k0, k1 = old[1][gi], new[1][gi]
    ignore = (1 << old[0]) | (1 << new[0])
    delta = (before ^ after) & ~ignore
    if not delta:
        return
    region = g["region"] | delta
    expected = world_map(m, new[1], new[2], new[3])[2]
    mismatch = (expected ^ after) & region & ~ignore
    if not mismatch:
        return

    if role in ("e", "f"):
        candidates = []
        for k, shape in enumerate(g["sup"]):
            score = ((shape ^ after) & region & ~ignore).bit_count()
            candidates.append((score, k))
        candidates.sort()
        if candidates and candidates[0][0] <= 1:
            actual = candidates[0][1]
            offset = (actual - k1) % g["period"]
            if offset:
                oldsup, oldblock = g["sup"], g["block"]
                g["sup"] = [oldsup[(k + offset) % g["period"]]
                            for k in range(g["period"])]
                g["block"] = [oldblock[(k + offset) % g["period"]]
                              for k in range(g["period"])]
                for eid in g["ids"]:
                    variants = m["entities"][eid]["variants"]
                    m["entities"][eid]["variants"] = [
                        variants[(k + offset) % len(variants)]
                        for k in range(len(variants))
                    ]
                g["override"] = {}

    g["region"] = region
    m["base"] &= ~delta
    g["override"][k0] = (before & region, g["block"][k0] & region)
    g["override"][k1] = (after & region, afterblock & region)
    clear_cache(m)
    m["plan"] = []


def probe(m):
    p, phases, cp, attached, unlocked = m["state"]
    options = []
    for gi, g in enumerate(m["groups"]):
        role = g["role"]
        if role not in m["visible"]:
            continue
        ph = list(phases)
        ph[gi] = (ph[gi] + 1) % g["period"]
        ph = tuple(ph)
        new = (p, ph, cp, attached, unlocked)
        if (world_map(m, ph, cp, attached)[0] & (1 << p)
                and (m["state"], role) not in m["banned"]):
            signature = (role, phases, p)
            options.append((m["probes"].get(signature, 0),
                            m["probes"].get(role, 0), role, new))
    if options:
        options.sort(key=lambda z: z[:3])
        _, _, role, new = options[0]
        signature = (role, phases, p)
        m["probes"][signature] = m["probes"].get(signature, 0) + 1
        m["probes"][role] = m["probes"].get(role, 0) + 1
        return role, new

    valid = world_map(m, phases, cp, attached)[2]
    gx, gy = m["goal"] % 32, m["goal"] // 32
    candidates = []
    for ai, (dx, dy) in enumerate(DIRS, 1):
        x, y = p % 32 + dx, p // 32 + dy
        if not (0 <= x < 32 and 0 <= y < 32):
            continue
        q = 32 * y + x
        if valid & (1 << q) and (m["state"], ai) not in m["banned"]:
            key = unlocked or bool(m["keymask"] & (1 << q))
            score = 12 * m["visits"].get(q, 0) + abs(x - gx) + abs(y - gy)
            candidates.append((score, ai, (q, phases, cp, attached, key)))
    if candidates:
        candidates.sort(key=lambda z: z[0])
        return candidates[0][1], candidates[0][2]
    return 1, m["state"]


def main_act(a, available, memory):
    player = find_player(a)
    if player is None:
        return "ACTION1", None, None

    m = memory.get("model")
    if m is None:
        m = initialize(a, player)
        memory["model"] = m
    m["calls"] += 1
    p = pos(m, *player)
    current_controls = controls(a, m["boundary"])
    old_visible = m["visible"]
    previous = m["pending"]
    crane = m["crane"]
    claw = find_claw(a, m["boundary"], crane["used_blue"]) if crane else None
    observed, observed_block = observed_map(a, m, claw)
    observed_valid = (observed & ~observed_block) | (1 << p)

    if previous is not None:
        old = previous["old"]
        predicted = previous["new"]
        action = previous["action"]
        new = predicted
        bad = False

        if p != predicted[0]:
            bad = True
            if isinstance(action, int):
                config = (old[1], old[2], old[3])
                m["extra"][config] = m["extra"].get(config, 0) | (1 << predicted[0])
                new = (p, old[1], old[2], old[3], old[4])
                m["banned"].add((old, action))
            elif action == "c":
                if p != old[0]:
                    m["portals"][old[0]] = p
                else:
                    m["banned"].add((old, action))
                new = (p, old[1], old[2], old[3], old[4])
            else:
                new = (p, old[1], old[2], old[3], old[4])

        if crane and claw is not None:
            actual_cp = crane["index"].get(claw[1])
            if actual_cp is not None and actual_cp != new[2]:
                new = (p, old[1], actual_cp, old[3], old[4])
                m["banned"].add((old, action))
                bad = True

        if isinstance(action, str) and action in m["gindex"]:
            oldmap = world_map(m, old[1], old[2], old[3])[2]
            newmap = world_map(m, predicted[1], predicted[2], predicted[3])[2]
            ignored = (1 << old[0]) | (1 << p)
            changed = (previous["before"] ^ observed_valid) & ~ignored
            expected_change = (oldmap ^ newmap) & ~ignored
            if not changed and expected_change:
                new = (p, old[1], old[2], old[3], old[4])
                m["banned"].add((old, action))
                bad = True
            elif new[1] == predicted[1]:
                learn_switch(m, old, new, action, previous["before"],
                             observed_valid, observed_block)

        if p != new[0]:
            new = (p, new[1], new[2], new[3], new[4])
        m["state"] = new
        if bad:
            m["plan"] = []
            clear_cache(m)
        m["pending"] = None
    else:
        s = m["state"]
        m["state"] = (p, s[1], s[2], s[3], s[4])

    for role in current_controls:
        m["gate_positions"].setdefault(role, set()).add(p)
    if previous and isinstance(previous["action"], int):
        for role in old_visible - set(current_controls):
            if role in ROLES or role == "c":
                m["gated"].add(role)
    m["controls"].update(current_controls)
    m["visible"] = set(current_controls)

    s = m["state"]
    if "d" in current_controls and not s[4]:
        m["state"] = (s[0], s[1], s[2], s[3], True)
    elif s[4] and m["keyrects"] and "d" not in current_controls:
        m["state"] = (s[0], s[1], s[2], s[3], False)
        if previous and previous["new"][4] and not previous["old"][4]:
            m["plan"] = []

    static = (a == 2) | (a == 6) | (a == 7)
    static[:, m["boundary"]:] = False
    static[63, :] = False
    ys, xs = np.where(static)
    base = m["base"] | pointmask(m, [(int(x), int(y)) for y, x in zip(ys, xs)])
    if base != m["base"]:
        m["base"] = base
        clear_cache(m)

    m["visits"][p] = m["visits"].get(p, 0) + 1

    if p == m["goal"]:
        return "ACTION1", None, None

    if m["plan"]:
        action, new = m["plan"][0]
        if isinstance(action, int):
            if not observed_valid & (1 << new[0]):
                m["plan"] = []
        elif action not in current_controls:
            m["plan"] = []

    if not m["plan"]:
        m["plan"] = search(m)

    if m["plan"]:
        action, new = m["plan"].pop(0)
    else:
        action, new = probe(m)

    if isinstance(action, str) and action not in current_controls:
        m["plan"] = []
        m["banned"].add((m["state"], action))
        action, new = probe(m)

    if isinstance(action, int):
        result = ("ACTION" + str(action), None, None)
    else:
        rc = current_controls.get(action, m["controls"].get(action))
        if rc is None or "ACTION6" not in available:
            action, new = 1, m["state"]
            result = ("ACTION1", None, None)
        else:
            result = ("ACTION6", int(rc[0]), int(rc[1]))

    if result[0] not in available:
        name = next((x for x in available if x in
                     ("ACTION1", "ACTION2", "ACTION3", "ACTION4")), "ACTION1")
        result = (name, None, None)
        action, new = int(name[-1]), m["state"]

    m["pending"] = {
        "old": m["state"], "new": new,
        "action": action, "before": observed_valid
    }
    return result


def fallback(a, available, memory):
    player = find_player(a)
    if player is None:
        return "ACTION1", None, None
    boundary = find_boundary(a)
    m = {"offset": (player[0] % 2, player[1] % 2),
         "boundary": boundary, "keyrects": []}
    p = pos(m, *player)
    safe, block = observed_map(a, m)
    valid = safe & ~block

    targets = []
    for pts in components(a[:, :boundary] == 11):
        if len(pts) <= 16:
            x, y, _, _ = bbox(pts)
            targets.append(pos(m, x, y))
    if targets:
        goal = targets[0]
        queue = deque([p])
        parent = {p: None}
        while queue:
            z = queue.popleft()
            if z == goal:
                break
            x, y = z % 32, z // 32
            for ai, (dx, dy) in enumerate(DIRS, 1):
                xx, yy = x + dx, y + dy
                q = 32 * yy + xx
                if (0 <= xx < 32 and 0 <= yy < 32 and q not in parent
                        and valid & (1 << q)):
                    parent[q] = (z, ai)
                    queue.append(q)
        if goal in parent and goal != p:
            q = goal
            while parent[q][0] != p:
                q = parent[q][0]
            return "ACTION" + str(parent[q][1]), None, None

    ctr = controls(a, boundary)
    count = memory.get("fallback_count", 0)
    memory["fallback_count"] = count + 1
    seen = memory.setdefault("fallback_visits", {})
    seen[p] = seen.get(p, 0) + 1
    options = []
    for ai, (dx, dy) in enumerate(DIRS, 1):
        x, y = p % 32 + dx, p // 32 + dy
        if 0 <= x < 32 and 0 <= y < 32:
            q = 32 * y + x
            if valid & (1 << q):
                options.append((seen.get(q, 0), ai))
    options.sort()
    if options and options[0][0] == 0:
        return "ACTION" + str(options[0][1]), None, None
    roles = [r for r in ("a", "b", "d", "e", "f", "c") if r in ctr]
    if roles and "ACTION6" in available and count % 3 == 0:
        role = roles[(count // 3) % len(roles)]
        row, col = ctr[role]
        return "ACTION6", int(row), int(col)
    if options:
        return "ACTION" + str(options[0][1]), None, None
    return "ACTION1", None, None


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int16)
    try:
        return main_act(a, available, memory)
    except Exception as exc:
        memory["last_error"] = str(exc)
        memory.pop("model", None)
        return fallback(a, available, memory)

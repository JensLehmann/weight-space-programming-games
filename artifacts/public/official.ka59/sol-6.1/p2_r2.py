import numpy as np
import heapq
import itertools
import math
import time
from collections import deque, defaultdict, Counter

W = 64
DIRS = ((0, -3), (0, 3), (-3, 0), (3, 0))
INF = 1000000.0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(int(x), int(y))]
        points = []
        while stack:
            xx, yy = stack.pop()
            points.append((xx, yy))
            for nx, ny in ((xx-1, yy), (xx+1, yy),
                           (xx, yy-1), (xx, yy+1)):
                if (0 <= nx < w and 0 <= ny < h
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(points)
    return result


def _bit_array(a):
    return int.from_bytes(
        np.packbits(a.reshape(-1).astype(np.uint8),
                    bitorder="little").tobytes(), "little")


def _footprint(kind, w, h):
    if kind == "C":
        # The movable cross has arms one third of its width.
        t = max(1, w // 3)
        u = max(1, h // 3)
        return [(x, y) for y in range(h) for x in range(w)
                if t <= x < w-t or u <= y < h-u]
    return [(x, y) for y in range(h) for x in range(w)]


def _canonical(obs):
    """Undo integer rendering scale, retaining display-relative coordinates."""
    a = np.asarray(obs)
    phase_x = phase_y = None
    for phase in (0, 1):
        xs = list(range(phase, 63, 2))
        if xs and np.array_equal(a[:62, xs], a[:62, [x+1 for x in xs]]):
            phase_x = phase
            break
    for phase in (0, 1):
        ys = list(range(phase, 61, 2))
        if ys and np.array_equal(a[ys, :], a[[y+1 for y in ys], :]):
            phase_y = phase
            break
    if phase_x is not None and phase_y is not None:
        b = a[phase_y:63:2, phase_x:64:2].copy()
        f = np.full((64, 64), 2, dtype=np.int8)
        f[:b.shape[0], :b.shape[1]] = b
        return f, (2, phase_y, phase_x)
    f = a.copy()
    f[63, :] = 2
    return f, (1, 0, 0)


def _read_pieces(f):
    pieces = []
    used = np.zeros((64, 64), bool)
    candidates = []

    # Clickable rectangles. Black edges indicate contact, not transparency.
    for h, w in ((3, 3), (6, 3), (3, 6), (6, 6)):
        hole = np.zeros((h, w), bool)
        if (h, w) == (3, 3):
            hole[1, 1] = True
        elif (h, w) == (6, 3):
            hole[2:4, 1] = True
        elif (h, w) == (3, 6):
            hole[1, 2:4] = True
        else:
            hole[2:4, 2:4] = True
        hy, hx = np.nonzero(hole)
        oy, ox = int(hy[0]), int(hx[0])
        for cy, cx in zip(*np.nonzero(np.isin(f, (0, 4, 5)))):
            y, x = int(cy)-oy, int(cx)-ox
            if x < 0 or y < 0 or x+w > 64 or y+h > 63:
                continue
            a = f[y:y+h, x:x+w]
            if not np.all(np.isin(a[hole], (0, 4, 5))):
                continue
            if not np.all(np.isin(a[~hole], (0, 14))):
                continue
            if not np.all(a[hole] == a[hole][0]):
                continue
            ecount = int(np.count_nonzero(a == 14))
            if ecount == 0 and int(a[hole][0]) != 0:
                continue
            active = int(a[hole][0]) == 0
            candidates.append((ecount, w*h, x, y, w, h, active))

    candidates.sort(reverse=True)
    active_piece = None
    for _, _, x, y, w, h, active in candidates:
        if np.any(used[y:y+h, x:x+w]):
            continue
        used[y:y+h, x:x+w] = True
        p = {"kind": "E", "w": w, "h": h, "p": y*64+x,
             "active": active, "ori": 0, "timer": 0}
        pieces.append(p)
        if active:
            active_piece = p

    # Crosses.
    cross_used = np.zeros((64, 64), bool)
    for component in _components(f == 11):
        xs = [q[0] for q in component]
        ys = [q[1] for q in component]
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        if w < 3 or h < 3:
            continue
        # Usually 9x9; allow other proportional cross sizes.
        if w != h or w % 3:
            w = h = 9
        pieces.append({"kind": "C", "w": w, "h": h,
                       "p": y*64+x, "active": False,
                       "ori": 0, "timer": 0})
        for xx, yy in _footprint("C", w, h):
            if 0 <= x+xx < 64 and 0 <= y+yy < 64:
                cross_used[y+yy, x+xx] = True

    # Solid bombs, including their partially filled fuse indicators.
    bombmask = np.isin(f, (12, 13))
    bomb_used = np.zeros_like(bombmask)
    for y, x in zip(*np.nonzero(bombmask)):
        x, y = int(x), int(y)
        if bomb_used[y, x]:
            continue
        size = None
        for d in (9, 6, 3):
            if (x+d <= 64 and y+d <= 63
                    and np.all(bombmask[y:y+d, x:x+d])
                    and not np.any(bomb_used[y:y+d, x:x+d])):
                size = d
                break
        if size is None:
            continue
        d = size
        bomb_used[y:y+d, x:x+d] = True
        a = f[y:y+d, x:x+d]
        filled = int(np.count_nonzero(a == 12))
        timer = filled // d
        ori = None
        if 0 < timer < d:
            for r in (0, 90, 180, 270):
                expected = np.zeros((d, d), bool)
                if r == 0:
                    expected[:timer, :] = True
                elif r == 90:
                    expected[:, d-timer:] = True
                elif r == 180:
                    expected[d-timer:, :] = True
                else:
                    expected[:, :timer] = True
                if np.array_equal(a == 12, expected):
                    ori = r
                    break
        pieces.append({"kind": "B", "w": d, "h": d,
                       "p": y*64+x, "active": False,
                       "ori": ori, "timer": timer})

    pieces.sort(key=lambda p: ({"C": 0, "B": 1, "E": 2}[p["kind"]],
                               p["p"], p["w"], p["h"]))
    selected = next((i for i, p in enumerate(pieces)
                     if p is active_piece), None)
    if selected is None:
        selected = next((i for i, p in enumerate(pieces)
                         if p["kind"] == "E"), None)

    unknown = used | cross_used | bomb_used
    enemies = []
    rare = np.isin(f, (3, 6, 7, 8, 9, 10))
    for comp in _components(rare):
        if len(comp) < 2:
            continue
        xs, ys = [q[0] for q in comp], [q[1] for q in comp]
        if max(xs)-min(xs) <= 12 and max(ys)-min(ys) <= 12:
            enemies.append(min(ys)*64+min(xs))
    enemies.sort()
    return pieces, selected, unknown, enemies


def _outlines():
    result = []
    for w, h in ((5, 5), (5, 8), (8, 5), (8, 8)):
        outline = [(x, y) for y in range(h) for x in range(w)
                   if x == 0 or y == 0 or x == w-1 or y == h-1]
        result.append(("E", w-2, h-2, w, h, outline))
    a = [
        "...#####...",
        "...#...#...",
        "...#...#...",
        "####...####",
        "#.........#",
        "#.........#",
        "#.........#",
        "####...####",
        "...#...#...",
        "...#...#...",
        "...#####...",
    ]
    result.append(("C", 9, 9, 11, 11,
                   [(x, y) for y, row in enumerate(a)
                    for x, v in enumerate(row) if v == "#"]))
    return result


def _read_goals(f, unknown):
    candidates = []
    for kind, iw, ih, w, h, outline in _outlines():
        ax, ay = outline[0]
        for y in range(64-h):
            for x in range(65-w):
                if f[y+ay, x+ax] != 4 and not unknown[y+ay, x+ax]:
                    continue
                matched = []
                good = True
                for xx, yy in outline:
                    if f[y+yy, x+xx] == 4 and not unknown[y+yy, x+xx]:
                        matched.append((x+xx, y+yy))
                    elif not unknown[y+yy, x+xx]:
                        good = False
                        break
                if good and len(matched) >= max(5, len(outline)//3):
                    candidates.append(
                        (len(matched), len(outline), kind, iw, ih,
                         (y+1)*64+x+1, matched))
    candidates.sort(key=lambda q: (q[0], q[1]), reverse=True)
    consumed = set()
    goals = []
    for count, total, kind, w, h, p, matched in candidates:
        if any(q in consumed for q in matched):
            continue
        consumed.update(matched)
        goals.append((kind, w, h, p))
    return sorted(set(goals))


def _static_maps(f, unknown, memory):
    walls = f == 2
    gulch = f == 15
    previous = memory.get("_gulch")
    if previous is not None and previous.shape == gulch.shape:
        gulch |= previous

    # Recover straight hazard strips hidden by movable pieces.
    for _ in range(3):
        changed = False
        for transpose in (False, True):
            a = gulch.T if transpose else gulch
            u = unknown.T if transpose else unknown
            for y in range(63):
                x = 0
                while x < 64:
                    if not (a[y, x] or u[y, x]):
                        x += 1
                        continue
                    start = x
                    while x < 64 and (a[y, x] or u[y, x]):
                        x += 1
                    end = x
                    indices = np.flatnonzero(a[y, start:end])
                    if len(indices) >= 2:
                        lo, hi = start+int(indices[0]), start+int(indices[-1])
                        if hi > lo and np.any(~a[y, lo:hi+1]):
                            a[y, lo:hi+1] = True
                            changed = True
        if not changed:
            break
    gulch[walls] = False
    walls[63, :] = True
    memory["_gulch"] = gulch.copy()
    return walls, gulch


def _make_model(pieces, selected, goals, walls, gulch, enemies):
    model = {"pieces": pieces, "goals": goals, "enemies": len(enemies)}
    n = len(pieces)
    model["n"] = n
    model["es"] = [i for i, q in enumerate(pieces) if q["kind"] == "E"]
    model["bs"] = [i for i, q in enumerate(pieces) if q["kind"] == "B"]
    model["cs"] = [i for i, q in enumerate(pieces) if q["kind"] == "C"]
    wallbits, gulchbits = _bit_array(walls), _bit_array(gulch)
    model["gulchbits"] = gulchbits

    specs = []
    for p in pieces:
        spec = (p["kind"], p["w"], p["h"])
        if spec not in specs:
            specs.append(spec)
    model["specs"] = specs
    model["types"] = [specs.index((p["kind"], p["w"], p["h"]))
                      for p in pieces]

    nextpos = [None] * 4096
    for p in range(4096):
        x, y = p % 64, p // 64
        nextpos[p] = tuple(
            (y+dy)*64+x+dx
            if 0 <= x+dx < 64 and 0 <= y+dy < 63 else -1
            for dx, dy in DIRS)
    model["next"] = nextpos

    tables = []
    for spec in specs:
        kind, w, h = spec
        residue = set()
        for p in pieces:
            if (p["kind"], p["w"], p["h"]) == spec:
                residue.add((p["p"] % 64 % 3, p["p"] // 64 % 3))
        for g in goals:
            if g[:3] == spec:
                residue.add((g[3] % 64 % 3, g[3] // 64 % 3))
        base = sum(1 << (yy*64+xx)
                   for xx, yy in _footprint(kind, w, h))
        masks = [0] * 4096
        gf = [False] * 4096
        valid = []
        for y in range(64-h):
            for x in range(65-w):
                if (x % 3, y % 3) not in residue:
                    continue
                p = y*64+x
                bits = base << p
                if bits & wallbits:
                    continue
                masks[p] = bits
                gf[p] = bool(bits & gulchbits)
                valid.append(p)
        slides = {}
        for p in valid:
            ends = []
            for di in range(4):
                q, t = p, 0
                while t < 50:
                    if t >= 5 and not gf[q]:
                        break
                    r = nextpos[q][di]
                    if r < 0 or not masks[r]:
                        break
                    q = r
                    t += 1
                ends.append(q)
            slides[p] = tuple(ends)
        tables.append({"mask": masks, "f": gf, "valid": valid,
                       "slide": slides, "w": w, "h": h, "kind": kind})
    model["tables"] = tables

    # Relaxed single-piece graphs provide maze-aware estimates.
    dtables = {}
    normal_tables = {}
    has_bombs = bool(model["bs"])
    for ti, tab in enumerate(tables):
        if tab["kind"] == "B":
            continue
        reverse = defaultdict(list)
        normal_reverse = defaultdict(list)
        masks, gf = tab["mask"], tab["f"]
        for p in tab["valid"]:
            for di, q in enumerate(nextpos[p]):
                if q >= 0 and masks[q]:
                    if tab["kind"] == "E" and not gf[q]:
                        reverse[q].append((p, 1.0))
                        normal_reverse[q].append((p, 1.0))
                    elif tab["kind"] == "C" and has_bombs:
                        reverse[q].append((p, 1.3))
                q = tab["slide"][p][di]
                if q != p:
                    reverse[q].append(
                        (p, 1.7 if tab["kind"] == "E" else 1.0))

        def distances(goal, graph):
            d = [INF] * 4096
            if not masks[goal]:
                return d
            d[goal] = 0.0
            heap = [(0.0, goal)]
            while heap:
                cost, p = heapq.heappop(heap)
                if cost != d[p]:
                    continue
                for q, weight in graph.get(p, ()):
                    nc = cost + weight
                    if nc < d[q]:
                        d[q] = nc
                        heapq.heappush(heap, (nc, q))
            return d

        for g in goals:
            if g[:3] == specs[ti]:
                dtables[g] = distances(g[3], reverse)
                normal_tables[g] = distances(g[3], normal_reverse)
    model["dist"] = dtables
    model["normal_dist"] = normal_tables

    groups = []
    for spec in specs:
        gs = [g for g in goals if g[:3] == spec]
        ids = [i for i, p in enumerate(pieces)
               if (p["kind"], p["w"], p["h"]) == spec]
        if gs:
            assignments = list(itertools.permutations(ids, len(gs)))
            groups.append((gs, ids, assignments))
    model["groups"] = groups
    model["contacts"] = {}

    positions = tuple(p["p"] for p in pieces)
    timers = tuple(pieces[i]["timer"] for i in model["bs"])
    if model["bs"]:
        state = positions + (selected,) + timers + (0,) * n + tuple(enemies)
    else:
        state = positions + (selected,) + tuple(enemies)
    model["initial"] = state
    return model


def _simulator(model):
    pieces = model["pieces"]
    n = model["n"]
    bs, cs, es = model["bs"], model["cs"], model["es"]
    nb = len(bs)
    tabs = [model["tables"][t] for t in model["types"]]
    masks = [t["mask"] for t in tabs]
    gf = [t["f"] for t in tabs]
    nxt = model["next"]
    enemy_count = model["enemies"]

    def simulate(state, action):
        pos = list(state[:n])
        sel = state[n]
        if action >= 4:
            j = action-4
            if j not in es:
                return None
            return tuple(pos) + (j,) + state[n+1:]

        di = action
        if nb:
            timers = list(state[n+1:n+1+nb])
            mom = list(state[n+1+nb:n+1+nb+n])
            enemy = list(state[n+1+nb+n:])
        else:
            timers, mom = [], [0]*n
            enemy = list(state[n+1:])

        def bits(i):
            p = pos[i]
            return masks[i][p] if 0 <= p < 4096 else 0

        def push(i, direction, depth=0):
            if depth > n+3:
                return True
            old = pos[i]
            q = nxt[old][direction] if 0 <= old < 4096 else -1
            if q < 0 or not masks[i][q]:
                return True
            pos[i] = q
            b = masks[i][q]
            collisions = [j for j in range(n) if j != i and b & bits(j)]
            for j in collisions:
                if push(j, direction, depth+1):
                    pos[i] = old
                    return True
            return False

        old = pos[sel]
        q = nxt[old][di] if 0 <= old < 4096 else -1
        hit = []
        if q >= 0 and masks[sel][q] and not gf[sel][q]:
            b = masks[sel][q]
            hit = [j for j in range(n)
                   if j != sel and b & bits(j)]
            if not hit:
                pos[sel] = q

        if hit:
            for j in hit:
                mom[j] = di+1
            t = 0
            while t < 80:
                stopped = []
                for j in hit:
                    if t >= 5 and not gf[j][pos[j]]:
                        stopped.append(True)
                    else:
                        stopped.append(push(j, di))
                if all(stopped):
                    break
                t += 1
            if t >= 80:
                return None

        # Enemies move only after a directional action.
        if enemy_count and cs:
            target = pos[cs[0]]
            tx, ty = target % 64, target // 64
            for k, ep in enumerate(enemy):
                ex, ey = ep % 64, ep // 64
                for _ in range(6):
                    dx, dy = tx-ex, ty-ey
                    if abs(dx) > abs(dy):
                        ex += 1 if dx > 0 else -1
                    else:
                        ey += 1 if dy > 0 else -1
                    if ex == tx and ey == ty:
                        return None
                enemy[k] = ey*64+ex

        triggered = []
        for k, j in enumerate(bs):
            timers[k] += 1
            if timers[k] >= pieces[j]["h"]:
                triggered.append(j)

        def rectangle_bits(x, y, w, h):
            x0, x1 = max(0, x), min(64, x+w)
            y0, y1 = max(0, y), min(64, y+h)
            if x1 <= x0 or y1 <= y0:
                return 0
            row = ((1 << (x1-x0))-1) << x0
            b = 0
            for yy in range(y0, y1):
                b |= row << (64*yy)
            return b

        if triggered:
            order = cs + bs + es
            for stage in (1, 2, 3):
                reach = 3*stage
                for j in triggered:
                    p = pos[j]
                    x, y = p % 64, p // 64
                    d = pieces[j]["w"]
                    ori = pieces[j]["ori"]
                    if ori is None:
                        ori = 0
                    if ori == 0:
                        b = rectangle_bits(x, y+d, d, reach)
                        bd = 1
                    elif ori == 90:
                        b = rectangle_bits(x-reach, y, reach, d)
                        bd = 2
                    elif ori == 180:
                        b = rectangle_bits(x, y-reach, d, reach)
                        bd = 0
                    else:
                        b = rectangle_bits(x+d, y, reach, d)
                        bd = 3
                    for k in order:
                        if b & bits(k):
                            push(k, bd)
                            mom[k] = bd+1
            for k, j in enumerate(bs):
                if j in triggered:
                    timers[k] = 0

            for _ in range(80):
                progressed = False
                for j in order:
                    if mom[j] and gf[j][pos[j]]:
                        if not push(j, mom[j]-1):
                            progressed = True
                if not progressed:
                    break
            else:
                return None

        if nb:
            return (tuple(pos) + (sel,) + tuple(timers)
                    + tuple(mom) + tuple(enemy))
        return tuple(pos) + (sel,) + tuple(enemy)

    return simulate


def _search(model, start, allowed, history, deadline):
    n = model["n"]
    pieces = model["pieces"]
    es, bs, cs = model["es"], model["bs"], model["cs"]
    tables = model["tables"]
    types = model["types"]
    dist = model["dist"]
    normal_dist = model["normal_dist"]
    groups = model["groups"]
    nxt = model["next"]
    simulate = _simulator(model)
    hcache = {}
    contacts = model["contacts"]

    def won(state):
        occupied = {(pieces[i]["kind"], pieces[i]["w"],
                     pieces[i]["h"], state[i]) for i in range(n)}
        return all(g in occupied for g in model["goals"])

    def contact_positions(target_i, target_p, di, helper_i):
        key = (types[target_i], target_p, di, types[helper_i])
        if key in contacts:
            return contacts[key]
        target_tab = tables[types[target_i]]
        helper_tab = tables[types[helper_i]]
        tm = target_tab["mask"][target_p]
        tx, ty = target_p % 64, target_p // 64
        tw, th = target_tab["w"], target_tab["h"]
        hw, hh = helper_tab["w"], helper_tab["h"]
        hm, hf = helper_tab["mask"], helper_tab["f"]
        rx = pieces[helper_i]["p"] % 64 % 3
        ry = pieces[helper_i]["p"] // 64 % 3
        result = []
        for y in range(max(0, ty-hh-3), min(63-hh, ty+th+3)+1):
            if y % 3 != ry:
                continue
            for x in range(max(0, tx-hw-3), min(64-hw, tx+tw+3)+1):
                if x % 3 != rx:
                    continue
                p = y*64+x
                if not hm[p] or hf[p] or hm[p] & tm:
                    continue
                q = nxt[p][di]
                if q >= 0 and hm[q] and not hf[q] and hm[q] & tm:
                    result.append(p)
        contacts[key] = result
        return result

    def heuristic(state):
        key = state[:n+1]
        if key in hcache:
            return hcache[key]
        pos, sel = state[:n], state[n]
        total = 0.0
        assignments = []
        for gs, ids, permutations in groups:
            if not permutations:
                return INF
            best, bestperm = INF, None
            for perm in permutations:
                cost = sum(dist.get(g, [INF]*4096)[pos[i]]
                           for g, i in zip(gs, perm))
                if cost < best:
                    best, bestperm = cost, perm
            if best >= INF:
                hcache[key] = INF
                return INF
            total += best
            for g, i in zip(gs, bestperm):
                if pos[i] != g[3]:
                    total += 0.8
                    assignments.append((g, i))

        assistance = 0.0
        for g, i in assignments:
            need_push = pieces[i]["kind"] == "C"
            if pieces[i]["kind"] == "E":
                nd = normal_dist.get(g)
                need_push = nd is not None and nd[pos[i]] >= INF
            if not need_push:
                continue
            tab = tables[types[i]]
            dd = dist[g]
            options = []
            for di in range(4):
                end = tab["slide"].get(pos[i], (pos[i],)*4)[di]
                if end != pos[i] and dd[end] < dd[pos[i]]:
                    options.append(di)
            if not options:
                continue
            besthelp = INF
            for j in es:
                if j == i:
                    continue
                hx, hy = pos[j] % 64, pos[j] // 64
                switch = 0.0 if j == sel else 1.0
                for di in options:
                    for p in contact_positions(i, pos[i], di, j):
                        px, py = p % 64, p // 64
                        cost = (abs(px-hx)+abs(py-hy))/3.0 + switch
                        if cost < besthelp:
                            besthelp = cost
            if besthelp < INF:
                weight = 0.45 if pieces[i]["kind"] == "C" else 0.35
                if bs:
                    weight *= 0.55
                assistance += weight*besthelp
        total += assistance
        hcache[key] = total
        return total

    if won(start):
        return []
    h0 = heuristic(start)
    nodes = [(start, 0, -1, None)]
    heap = [(2.15*h0, h0, 0)]
    bestg = {start: 0}
    best_index, best_value = 0, h0
    best_nonroot = None
    solution = None
    expansions = 0
    max_depth = 145 if bs else 125

    while heap and time.perf_counter() < deadline:
        _, _, idx = heapq.heappop(heap)
        state, g, parent, action = nodes[idx]
        if bestg.get(state) != g:
            continue
        if won(state):
            solution = idx
            break
        h = heuristic(state)
        progress = h + 0.025*g
        if idx and (best_nonroot is None
                    or progress < best_nonroot[0]):
            best_nonroot = (progress, idx)
        if progress < best_value-1e-8:
            best_value, best_index = progress, idx
        if g >= max_depth:
            continue
        expansions += 1
        if expansions > 65000 or len(nodes) > 240000:
            break

        actions = []
        for di in range(4):
            if di+1 in allowed:
                actions.append(di)
        if 6 in allowed:
            actions.extend(4+j for j in es if j != state[n])

        for a in actions:
            new = simulate(state, a)
            if new is None or new == state:
                continue
            ng = g+1
            if ng >= bestg.get(new, INF):
                continue
            nh = heuristic(new)
            if nh >= INF:
                continue
            bestg[new] = ng
            if a < 4:
                output = ("ACTION"+str(a+1), None, None)
            else:
                j = a-4
                p = state[j]
                output = ("ACTION6",
                          p//64 + pieces[j]["h"]//2,
                          p%64 + pieces[j]["w"]//2)
            ni = len(nodes)
            nodes.append((new, ng, idx, output))
            repetition = min(5, history.get(new, 0))
            timer_bonus = 0.0
            if bs:
                timer_bonus = 0.045*sum(new[n+1:n+1+len(bs)])
            priority = ng + 2.15*nh + 0.7*repetition - timer_bonus
            heapq.heappush(heap, (priority, nh, ni))

        if len(heap) > 120000:
            heap = heapq.nsmallest(65000, heap)
            heapq.heapify(heap)

    if solution is None:
        solution = best_index
        if solution == 0 and best_nonroot is not None:
            solution = best_nonroot[1]
    if solution == 0:
        return []

    path = []
    idx = solution
    while nodes[idx][2] >= 0:
        state, g, parent, action = nodes[idx]
        path.append((action, state))
        idx = parent
    path.reverse()
    if not won(nodes[solution][0]):
        path = path[:6]
    return path


def _matches(model, expected, pieces, selected, enemies):
    if len(pieces) != model["n"] or selected is None:
        return False
    n = model["n"]
    old = model["pieces"]
    expected_multiset = Counter(
        (old[i]["kind"], old[i]["w"], old[i]["h"], expected[i])
        for i in range(n))
    observed_multiset = Counter(
        (p["kind"], p["w"], p["h"], p["p"]) for p in pieces)
    if expected_multiset != observed_multiset:
        return False
    if expected[expected[n]] != pieces[selected]["p"]:
        return False
    nb = len(model["bs"])
    if nb:
        for k, i in enumerate(model["bs"]):
            matches = [p for p in pieces
                       if p["kind"] == "B" and p["p"] == expected[i]
                       and p["w"] == old[i]["w"]]
            if not matches or matches[0]["timer"] != expected[n+1+k]:
                return False
            observed_ori = matches[0]["ori"]
            if observed_ori is not None and old[i]["ori"] != observed_ori:
                return False
        expected_enemy = expected[n+1+nb+n:]
    else:
        expected_enemy = expected[n+1:]
    return sorted(expected_enemy) == sorted(enemies)


def _output(action, transform):
    name, row, col = action
    if name != "ACTION6":
        return name, None, None
    scale, py, px = transform
    return (name, int(max(0, min(63, row*scale+py))),
            int(max(0, min(63, col*scale+px))))


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 9.15
    allowed = set()
    for name in available:
        s = str(name).upper().split(".")[-1]
        if s.startswith("ACTION"):
            try:
                allowed.add(int(s[6:]))
            except ValueError:
                pass
    if not allowed:
        return "RESET", None, None

    f, transform = _canonical(obs)
    pieces, selected, unknown, enemies = _read_pieces(f)
    if selected is None:
        return "RESET", None, None

    model = memory.get("_model")
    expected = memory.get("_expected")
    if (model is not None and expected is not None
            and memory.get("_transform") == transform
            and _matches(model, expected, pieces, selected, enemies)):
        state = expected
        plan = memory.get("_plan", [])
        if plan:
            action, newstate = plan[0]
            memory["_plan"] = plan[1:]
            memory["_expected"] = newstate
            return _output(action, transform)
    else:
        model = None
        state = None
        memory["_plan"] = []

    if model is None:
        goals = _read_goals(f, unknown)
        oldgoals = memory.get("_goals", [])
        goals = sorted(set(goals) | set(oldgoals))
        memory["_goals"] = goals
        walls, gulch = _static_maps(f, unknown, memory)

        # At a fresh bomb cycle the solid image cannot reveal orientation.
        # One harmless directional move reveals its fuse stripe.
        unknown_bombs = [p for p in pieces
                         if p["kind"] == "B" and p["ori"] is None]
        if unknown_bombs:
            remembered = memory.get("_bomb_orientations", {})
            for p in unknown_bombs:
                key = (p["w"], p["p"])
                if key in remembered:
                    p["ori"] = remembered[key]
            unknown_bombs = [p for p in pieces
                             if p["kind"] == "B" and p["ori"] is None]
        if unknown_bombs:
            active = pieces[selected]
            x, y = active["p"] % 64, active["p"] // 64
            targets = [g for g in goals
                       if g[:3] == ("E", active["w"], active["h"])]
            choices = []
            occupied = np.zeros((64, 64), bool)
            for i, p in enumerate(pieces):
                if i == selected:
                    continue
                xx, yy = p["p"] % 64, p["p"] // 64
                for ox, oy in _footprint(p["kind"], p["w"], p["h"]):
                    if 0 <= xx+ox < 64 and 0 <= yy+oy < 64:
                        occupied[yy+oy, xx+ox] = True
            for di, (dx, dy) in enumerate(DIRS):
                if di+1 not in allowed:
                    continue
                nx, ny = x+dx, y+dy
                w, h = active["w"], active["h"]
                if nx < 0 or ny < 0 or nx+w > 64 or ny+h > 63:
                    continue
                region = np.s_[ny:ny+h, nx:nx+w]
                if np.any(walls[region] | gulch[region] | occupied[region]):
                    continue
                score = min(
                    (abs(nx-g[3]%64)+abs(ny-g[3]//64)
                     for g in targets), default=0)
                choices.append((score, di))
            if choices:
                di = min(choices)[1]
            else:
                di = next((i for i in range(4) if i+1 in allowed), 0)
            memory.pop("_model", None)
            memory.pop("_expected", None)
            memory["_transform"] = transform
            return "ACTION"+str(di+1), None, None

        orientations = memory.setdefault("_bomb_orientations", {})
        for p in pieces:
            if p["kind"] == "B" and p["ori"] is not None:
                orientations[(p["w"], p["p"])] = p["ori"]

        model = _make_model(
            pieces, selected, goals, walls, gulch, enemies)
        state = model["initial"]
        memory["_model"] = model
        memory["_transform"] = transform

    history = memory.setdefault("_history", {})
    history[state] = history.get(state, 0)+1
    if len(history) > 500:
        for key in list(history)[:150]:
            del history[key]

    plan = _search(model, state, allowed, history, deadline)
    if plan:
        action, newstate = plan[0]
        memory["_plan"] = plan[1:]
        memory["_expected"] = newstate
        return _output(action, transform)

    # Search fallback: choose a legal successor with useful displacement.
    simulate = _simulator(model)
    n = model["n"]
    candidates = []
    for di in range(4):
        if di+1 not in allowed:
            continue
        new = simulate(state, di)
        if new is None:
            continue
        moved = sum(new[i] != state[i] for i in range(n))
        cost = 0.0
        for gs, ids, permutations in model["groups"]:
            if permutations:
                cost += min(
                    sum(model["dist"].get(g, [INF]*4096)[new[i]]
                        for g, i in zip(gs, perm))
                    for perm in permutations)
        cost += 2.0*history.get(new, 0) - 0.15*moved
        candidates.append((cost, di, new))
    if candidates:
        _, di, new = min(candidates, key=lambda q: (q[0], q[1]))
        memory["_expected"] = new
        memory["_plan"] = []
        return "ACTION"+str(di+1), None, None

    if 6 in allowed:
        es = model["es"]
        j = next((i for i in es if i != state[n]), state[n])
        p = state[j]
        action = ("ACTION6", p//64+model["pieces"][j]["h"]//2,
                  p%64+model["pieces"][j]["w"]//2)
        memory.pop("_expected", None)
        return _output(action, transform)
    return "RESET", None, None

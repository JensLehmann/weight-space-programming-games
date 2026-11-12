import numpy as np
import heapq
import time
from collections import deque

STRIDE = 256
DIRS = (-256, 256, -1, 1)
NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
COLORS = (14, 8, 9)

PMASK = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=bool)
BORDER = np.ones((6, 6), dtype=bool)
BORDER[1:5, 1:5] = False


def enc(x, y):
    return x + 128 + STRIDE * (y + 128)


def xy(p):
    return p % STRIDE - 128, p // STRIDE - 128


def distance(a, b):
    return abs(a % STRIDE - b % STRIDE) + abs(a // STRIDE - b // STRIDE)


def point(p, tx, ty):
    x, y = xy(p)
    return 6 * x + tx + 3, 6 * y + ty + 3


def visible(p, tx, ty):
    x, y = point(p, tx, ty)
    return 1 <= x <= 62 and 1 <= y <= 62


def state(groups, pads, boxes):
    return tuple(tuple(sorted(g)) for g in groups) + (
        tuple(sorted(pads)), tuple(sorted(boxes))
    )


def count(s):
    return len(s[0]) + len(s[1])


def bits(mask):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


def lattice(obs, old=None):
    best = (-1e30, 0, 0)
    for oy in range(6):
        for ox in range(6):
            score = 0.0
            for y in range(oy, 59, 6):
                if y == 0:
                    continue
                for x in range(ox, 59, 6):
                    a = obs[y:y + 6, x:x + 6]
                    corners = (
                        int(a[0, 0]), int(a[0, 5]),
                        int(a[5, 0]), int(a[5, 5])
                    )
                    if corners == (0, 0, 0, 0):
                        inside = a[1:5, 1:5]
                        active = np.count_nonzero(
                            (inside != 0) & (inside != 5) & (inside != 10)
                        )
                        if active >= 4:
                            score += np.count_nonzero(a[BORDER] == 0)
                            score += np.count_nonzero(inside == 1)
                    elif corners == (11, 11, 11, 11):
                        score += np.count_nonzero(a[BORDER] == 11)
                        score += np.count_nonzero(a[1:5, 1:5] == 12)
                    for color in COLORS:
                        n = np.count_nonzero(a[PMASK] == color)
                        if n >= 10:
                            score += 30 + n
            if old is not None:
                score += 0.3 * (ox == old[0] % 6)
                score += 0.3 * (oy == old[1] % 6)
            if score > best[0]:
                best = (score, ox, oy)
    return best[1], best[2]


def read_image(obs, ox, oy):
    cells, pegs = {}, {}
    pads, boxes, markers, gray, clear = set(), set(), set(), set(), set()

    for ly in range(-1, 12):
        y = oy + 6 * ly
        for lx in range(-1, 12):
            x = ox + 6 * lx
            q = (lx, ly)
            if not (0 <= x + 2 and x + 3 < 64
                    and 1 <= y + 2 and y + 3 < 64):
                continue
            full = 0 <= x <= 58 and 1 <= y <= 58
            core = obs[y + 2:y + 4, x + 2:x + 4]
            kind = None

            if full:
                a = obs[y:y + 6, x:x + 6]
                corners = (
                    int(a[0, 0]), int(a[0, 5]),
                    int(a[5, 0]), int(a[5, 5])
                )
                inside = a[1:5, 1:5]
                active = np.count_nonzero(
                    (inside != 0) & (inside != 5) & (inside != 10)
                )
                if corners == (11, 11, 11, 11):
                    kind = "D"
                    pads.add(q)
                elif corners == (0, 0, 0, 0) and active >= 4:
                    kind = "N"
                elif np.all(core == 5):
                    kind = "R"
                elif np.all(core == 10):
                    kind = "E"

                if kind in ("N", "D"):
                    for color in COLORS:
                        if np.count_nonzero(a[PMASK] == color) >= 10:
                            pegs[q] = color
                            break
                    ng = np.count_nonzero(a[PMASK] == 2)
                    if ng >= 10:
                        gray.add(q)
                    elif ng >= 6 and np.count_nonzero(core == 2) <= 1:
                        markers.add(q)
                    if (np.count_nonzero(a == 15) >= 6
                            and np.count_nonzero(a == 7) >= 2):
                        boxes.add(q)
            else:
                if np.all(core == 5):
                    kind = "R"
                elif np.all(core == 10):
                    kind = "E"
                elif np.all(core == 1):
                    kind = "N"
                elif np.all(core == 12):
                    kind = "D"
                    pads.add(q)
                else:
                    for color in COLORS:
                        if np.all(core == color):
                            pegs[q] = color
                            break
                    patch = obs[
                        max(1, y):min(64, y + 6),
                        max(0, x):min(64, x + 6)
                    ]
                    if np.count_nonzero(patch == 11) >= 5:
                        kind = "D"
                        pads.add(q)
                    elif np.count_nonzero(core == 7) >= 2:
                        kind = "N"
                        boxes.add(q)

            if kind is not None:
                cells[q] = kind
            if full or kind in ("N", "D", "E"):
                clear.add(q)

    return {
        "cells": cells, "pegs": pegs, "pads": pads, "boxes": boxes,
        "markers": markers, "gray": gray, "clear": clear
    }


def convert(v, kx, ky):
    def cv(q):
        return enc(q[0] + kx, q[1] + ky)

    return {
        "cells": {cv(q): t for q, t in v["cells"].items()},
        "pegs": {cv(q): c for q, c in v["pegs"].items()},
        "pads": {cv(q) for q in v["pads"]},
        "boxes": {cv(q) for q in v["boxes"]},
        "markers": {cv(q) for q in v["markers"]},
        "gray": {cv(q) for q in v["gray"]},
        "clear": {cv(q) for q in v["clear"]}
    }


def locate(v, ox, oy, m):
    oldtx, oldty = m["tx"], m["ty"]
    if (m.get("target") is not None
            and ox == oldtx % 6 and oy == oldty % 6):
        return (
            int(round((ox - oldtx) / 6)),
            int(round((oy - oldty) / 6)), oldtx, oldty
        )

    expected = (oldtx, oldty)
    if m.get("pending") is not None:
        expected = m["pending"].get("expected_camera", expected)
    kx0 = int(round((ox - expected[0]) / 6))
    ky0 = int(round((oy - expected[1]) / 6))

    predicted = {}
    for group, color in zip(m["s"][:3], COLORS):
        for p in group:
            predicted[p] = color

    pd, pb = set(m["s"][3]), set(m["s"][4])
    normal, rails, empty = m["N"], m["R"], m["E"]
    best = (-1e30, kx0, ky0)

    for dy in range(-10, 11):
        for dx in range(-16, 17):
            kx, ky = kx0 + dx, ky0 + dy
            score = -0.65 * abs(dx) - 0.8 * abs(dy)
            for (lx, ly), kind in v["cells"].items():
                p = enc(lx + kx, ly + ky)
                if kind == "N":
                    if p in normal:
                        score += 3.0
                    elif p in rails:
                        score -= 4.0
                elif kind in ("R", "D"):
                    if p in rails:
                        score += 2.0
                    elif p in normal:
                        score -= 4.0
                elif p in empty:
                    score += 0.4
                elif p in normal or p in rails:
                    score -= 3.0

            for (lx, ly), color in v["pegs"].items():
                p = enc(lx + kx, ly + ky)
                if predicted.get(p) == color:
                    score += 22.0
                elif p in predicted:
                    score -= 13.0
                elif p in empty:
                    score -= 5.0

            for lx, ly in v["pads"]:
                score += 15.0 * (enc(lx + kx, ly + ky) in pd)
            for lx, ly in v["boxes"]:
                score += 13.0 * (enc(lx + kx, ly + ky) in pb)
            for p in predicted:
                px, py = xy(p)
                q = (px - kx, py - ky)
                if (q in v["clear"] and q not in v["pegs"]
                        and q not in v["gray"]):
                    score -= 11.0

            if score > best[0]:
                best = (score, kx, ky)

    _, kx, ky = best
    return kx, ky, ox - 6 * kx, oy - 6 * ky


def merge(v, m):
    for p, kind in v["cells"].items():
        if kind == "N":
            if p not in m["R"]:
                m["N"].add(p)
            m["E"].discard(p)
        elif kind in ("R", "D"):
            m["R"].add(p)
            m["N"].discard(p)
            m["E"].discard(p)
        elif p not in m["N"] and p not in m["R"]:
            m["E"].add(p)

    region = v["clear"]
    groups = [set(g) - region for g in m["s"][:3]]
    pads = set(m["s"][3]) - region
    boxes = set(m["s"][4]) - region
    for p, color in v["pegs"].items():
        for g in groups:
            g.discard(p)
        groups[COLORS.index(color)].add(p)
    pads.update(v["pads"])
    boxes.update(v["boxes"])
    m["s"] = state(groups, pads, boxes)
    m["blue"] |= bool(groups[2])
    m["red"] |= bool(groups[1])


def slide(s, d, rails):
    pads = set(s[3])
    moves = {}
    for p in sorted(s[3], reverse=d > 0):
        q = p + d
        if q in rails and q not in pads:
            pads.remove(p)
            pads.add(q)
            moves[p] = q
    if not moves:
        return None, {}
    groups = [
        tuple(sorted(moves.get(p, p) for p in group))
        for group in s[:3]
    ]
    boxes = tuple(sorted(moves.get(p, p) for p in s[4]))
    return tuple(groups) + (tuple(sorted(pads)), boxes), moves


def jump(s, gi, src, mid, dst):
    groups = [set(g) for g in s[:3]]
    groups[gi].remove(src)
    groups[gi].add(dst)
    if gi != 2 and mid in groups[gi]:
        groups[gi].remove(mid)
    return state(groups, s[3], s[4])


def camera_after(s, moves, d, tx, ty, m):
    if not any(p in moves for p in s[0]):
        return tx, ty
    if abs(d) == 1:
        rate = m["followx"]
        if rate is not None:
            tx = min(tx + int(round(rate * d)), m["maxtx"])
    else:
        rate = m["followy"]
        if rate is not None:
            ty = min(
                ty + int(round(rate * (d // STRIDE))), m["maxty"]
            )
    return tx, ty


def frontier_info(m):
    known = m["N"] | m["R"] | m["E"]
    tx, ty = m["tx"], m["ty"]
    fringe, targets = set(), set()

    for p in m["R"]:
        if visible(p, tx, ty):
            for d in DIRS:
                q = p + d
                if q not in known and not visible(q, tx, ty):
                    fringe.add(q)

    for p in m["N"]:
        if not visible(p, tx, ty) or p in m["landings"]:
            continue
        for d in DIRS:
            for step in range(1, 4):
                q = p + step * d
                if q in m["E"]:
                    break
                if q not in known:
                    if not visible(q, tx, ty):
                        targets.add(p)
                    break
                if q not in m["N"] and q not in m["R"]:
                    break

    connected, queue = set(), deque()
    for p in m["R"]:
        if any(p + d in fringe for d in DIRS):
            connected.add(p)
            queue.append(p)
    while queue:
        p = queue.popleft()
        for d in DIRS:
            q = p + d
            if q in m["R"] and q not in connected:
                connected.add(q)
                queue.append(q)

    padtargets = {
        p for p in m["s"][3]
        if p in connected and p not in m["landings"]
        and visible(p, tx, ty)
    }
    return fringe, targets, padtargets


def relaxed_distances(normal, rails, targets):
    floor = normal | rails
    dist, heap = {}, []
    for p in targets:
        if p in floor:
            dist[p] = 0.0
            heapq.heappush(heap, (0.0, p))
    while heap:
        g, p = heapq.heappop(heap)
        if g != dist.get(p):
            continue
        for d in DIRS:
            q = p + 2 * d
            if q in floor and p + d in floor:
                ng = g + 2.0
                if ng < dist.get(q, 1e30):
                    dist[q] = ng
                    heapq.heappush(heap, (ng, q))
            q = p + d
            if p in rails and q in rails:
                ng = g + 1.0
                if ng < dist.get(q, 1e30):
                    dist[q] = ng
                    heapq.heappush(heap, (ng, q))
    return dist


def can_progress(s, normal, rails):
    occupied = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
    pads = set(s[3])
    if occupied.intersection(pads):
        return True
    for group in s[:3]:
        for src in group:
            for d in DIRS:
                mid, dst = src + d, src + 2 * d
                if (mid in occupied and dst not in occupied
                        and (dst in normal or (dst in rails and pads))):
                    return True
    return False


def helper_work(src, dst, pads):
    work = 0.55 * distance(src, dst)
    sx, sy = xy(src)
    dx, dy = xy(dst)
    if (sx - dx) % 2 or (sy - dy) % 2:
        work += 7.0
        if src in pads:
            work -= 5.0
    return work


def blue_potential(s, normal, rails, goal_dist, exploring):
    pads = set(s[3])
    occupied = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
    movable = s[0] + s[1] + s[2]
    floor = normal | rails
    value = 0.0

    for group in s[:2]:
        if len(group) > 1:
            value += 0.25 * sum(
                min(distance(p, q) for q in group if p != q)
                for p in group
            )

    for group in s[:2]:
        for src in group:
            if src in pads:
                continue
            best = 17.0
            for d in DIRS:
                mid, dst = src + d, src + 2 * d
                if dst not in floor or dst in occupied:
                    continue
                padwork = 0.0
                if dst in rails and dst not in pads:
                    if not pads:
                        continue
                    padwork = 0.5 * min(distance(p, dst) for p in pads)
                if mid in occupied:
                    work = padwork
                else:
                    helpers = [p for p in movable if p != src]
                    if not helpers:
                        continue
                    work = 2.5 + min(
                        helper_work(p, mid, pads) for p in helpers
                    ) + padwork
                best = min(best, work)
            value += best

    if goal_dist:
        pieces = s[0] + s[1]
        nearest = min(
            (goal_dist.get(p, 45.0) for p in pieces), default=45.0
        )
        guided = 60.0
        for src in pieces:
            for d in DIRS:
                mid, dst = src + d, src + 2 * d
                if dst not in floor or dst in occupied:
                    continue
                future = goal_dist.get(dst, 45.0)
                work = 0.0
                if mid not in occupied:
                    helpers = [p for p in movable if p != src]
                    if not helpers:
                        continue
                    work += 2.0 + min(
                        helper_work(p, mid, pads) for p in helpers
                    )
                if dst in rails and dst not in pads:
                    if not pads:
                        continue
                    work += 0.6 * min(distance(p, dst) for p in pads)
                guided = min(guided, 2.0 + future + work)
            if src in pads:
                for d in DIRS:
                    if src + d in rails:
                        guided = min(
                            guided, 1.0 + goal_dist.get(src + d, 45.0)
                        )
        value += (1.1 if exploring else 0.45) * nearest
        value += (1.0 if exploring else 0.35) * guided
    return value


def explore_distance(s, fringe, m):
    if not fringe:
        return 0.0
    pads = set(s[3])
    carriers = [p for p in s[0] if p in pads]
    if carriers:
        return min(distance(p, q) for p in carriers for q in fringe)
    if not pads or not s[0]:
        return 40.0
    return 7.0 + min(distance(p, q) for p in s[0] for q in pads)


def reconstruct(node, parents):
    result = []
    while parents[node] is not None:
        previous, action = parents[node]
        result.append((action, node))
        node = previous
    result.reverse()
    return result


def ordinary_search(m):
    simple = not m["blue"]
    s0 = m["s"]
    root = (s0, m["tx"], m["ty"])
    normal = m["N"]
    fringe, targets, padtargets = frontier_info(m)
    rails = m["R"] | fringe
    unknown = bool(fringe or targets)
    startn = count(s0)
    goal = max(1, int(bool(s0[0])) + int(bool(s0[1])))
    exploring = startn <= goal
    goal_dist = relaxed_distances(normal, rails, targets | padtargets)
    parents, costs = {root: None}, {root: 0.0}
    pcache = {}

    def extra(s):
        if s not in pcache:
            if simple:
                h = 0.0
                for group in s[:2]:
                    if len(group) > 1:
                        xs = [p % STRIDE for p in group]
                        ys = [p // STRIDE for p in group]
                        h += 0.16 * (
                            max(xs) - min(xs) + max(ys) - min(ys)
                        )
                h += (0.9 if exploring else 0.20) * explore_distance(
                    s, fringe, m
                )
                pcache[s] = h
            else:
                pcache[s] = blue_potential(
                    s, normal, rails, goal_dist, unknown and exploring
                )
        return pcache[s]

    weight = 8.0 if simple else 11.0
    rootextra = extra(s0)
    started = explore_distance(s0, fringe, m)
    heap = [(weight * max(0, startn - goal) + rootextra, 0.0, 0, root)]
    best, bestrank, serial = root, -1e30, 0
    deadline = time.perf_counter() + 4.8
    expansions = 0

    while heap and expansions < 55000 and len(costs) < 160000:
        if expansions % 64 == 0 and time.perf_counter() >= deadline:
            break
        _, g, _, node = heapq.heappop(heap)
        if g != costs.get(node):
            continue
        expansions += 1
        s, tx, ty = node
        n = count(s)

        if node != root:
            previous, last = parents[node]
            if simple:
                exporting = bool(set(s[0]).intersection(s[3]).intersection(fringe))
                if (not exploring and n <= goal) or exporting:
                    return reconstruct(node, parents)
            else:
                if last[0] == "j":
                    gi, src, dst = last[1:]
                    if not unknown and n <= goal:
                        return reconstruct(node, parents)
                    if dst in targets and gi != 2:
                        return reconstruct(node, parents)
                    if dst in padtargets:
                        return reconstruct(node, parents)
                old = previous[0]
                carried = set(s[0]).intersection(s[3]).intersection(fringe)
                previous_carried = set(old[0]).intersection(old[3])
                if carried - previous_carried:
                    if m["followx"] != 0 or m["followy"] != 0:
                        return reconstruct(node, parents)

        if not simple and not can_progress(s, normal, rails):
            continue

        occupied = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
        destinations = normal | set(s[3])
        successors, mobility, removals = [], 0, 0

        for gi, group in enumerate(s[:3]):
            own = set(group)
            for src in group:
                if not visible(src, tx, ty):
                    continue
                for d in DIRS:
                    mid, dst = src + d, src + 2 * d
                    if (mid not in occupied or dst in occupied
                            or dst not in destinations
                            or not visible(dst, tx, ty)
                            or (s, src, dst) in m["banned"]):
                        continue
                    removing = gi != 2 and mid in own
                    mobility += 1
                    removals += int(removing)
                    ns = jump(s, gi, src, mid, dst)
                    if (not simple and unknown
                            and not can_progress(ns, normal, rails)
                            and dst not in targets | padtargets):
                        continue
                    successors.append((
                        (ns, tx, ty), ("j", gi, src, dst),
                        2.0 if gi != 2 else 2.7
                    ))

        for ai, d in enumerate(DIRS):
            ns, moves = slide(s, d, rails)
            if ns is None:
                continue
            outside = set(ns[3]).intersection(fringe)
            if outside:
                if not outside.intersection(ns[0]):
                    continue
                if not simple:
                    rate = m["followx"] if abs(d) == 1 else m["followy"]
                    if rate == 0:
                        continue
            ntx, nty = camera_after(s, moves, d, tx, ty, m)
            successors.append(((ns, ntx, nty), ("a", ai), 1.0))

        if node != root:
            if simple:
                rank = (
                    100.0 * (startn - n)
                    + 2.2 * (started - explore_distance(s, fringe, m))
                    + 3.0 * removals + 0.25 * min(mobility, 12)
                    - 0.35 * g
                    - 5.0 * min(4, m["visits"].get(node, 0))
                )
            else:
                rank = (
                    100.0 * (startn - n)
                    + 3.3 * (rootextra - extra(s))
                    + 2.0 * removals + 0.3 * min(mobility, 8)
                    - 0.45 * g
                    - 6.0 * min(6, m["visits"].get(node, 0))
                )
            if rank > bestrank:
                bestrank, best = rank, node

        if g > 130:
            continue
        for nsnode, action, cost in successors:
            ng = g + cost + min(15.0, 3.0 * m["visits"].get(nsnode, 0))
            if ng >= costs.get(nsnode, float("inf")):
                continue
            costs[nsnode] = ng
            parents[nsnode] = (node, action)
            serial += 1
            ns = nsnode[0]
            h = weight * max(0, count(ns) - goal) + extra(ns)
            heapq.heappush(heap, (ng + h, ng, serial, nsnode))

    return reconstruct(best, parents) if best != root else []


def multicolour_search(m):
    """Compact exact transitions, with movable-box support estimates."""
    started_time = time.perf_counter()
    deadline = started_time + 4.8
    fringe, targets, padtargets = frontier_info(m)
    rails = m["R"] | fringe
    normal = m["N"]
    positions = sorted(normal | rails | set().union(*map(set, m["s"])))
    if len(positions) > 380:
        return ordinary_search(m)

    ix = {p: i for i, p in enumerate(positions)}
    size = len(positions)
    one = [1 << i for i in range(size)]
    normalmask = sum(one[ix[p]] for p in normal if p in ix)
    railmask = sum(one[ix[p]] for p in rails if p in ix)
    fringemask = sum(one[ix[p]] for p in fringe if p in ix)
    exportmask = sum(one[ix[p]] for p in targets | padtargets if p in ix)
    visiblemask = sum(
        one[i] for i, p in enumerate(positions)
        if visible(p, m["tx"], m["ty"])
    )

    jumps = [[] for _ in positions]
    neighbours = [[-1] * size for _ in DIRS]
    graph = [[] for _ in positions]
    rgraph = [[] for _ in positions]

    for i, p in enumerate(positions):
        for ai, d in enumerate(DIRS):
            q, mid = p + 2 * d, p + d
            if q in ix and mid in ix:
                j, k = ix[q], ix[mid]
                jumps[i].append((k, j))
                graph[i].append((j, 2.0))
            if p in rails and mid in rails:
                j = ix[mid]
                neighbours[ai][i] = j
                graph[i].append((j, 1.0))
                rgraph[i].append((j, 1.0))

    def alldist(edges):
        result = []
        for start in range(size):
            dist = [1000.0] * size
            dist[start] = 0.0
            heap = [(0.0, start)]
            while heap:
                g, p = heapq.heappop(heap)
                if g != dist[p]:
                    continue
                for q, cost in edges[p]:
                    ng = g + cost
                    if ng < dist[q]:
                        dist[q] = ng
                        heapq.heappush(heap, (ng, q))
            result.append(dist)
        return result

    distances = alldist(graph)
    raildist = alldist(rgraph)
    railindices = list(bits(railmask))
    exportindices = list(bits(exportmask))
    edist = [
        min((distances[i][j] for j in exportindices), default=45.0)
        for i in range(size)
    ]

    root = tuple(
        sum(one[ix[p]] for p in group if p in ix) for group in m["s"]
    )
    startn = root[0].bit_count() + root[1].bit_count()
    goal = max(1, int(bool(root[0])) + int(bool(root[1])))
    unknown = bool(fringe or targets)
    exploring = startn <= goal and unknown
    option_cache = {}
    heuristic_cache = {}

    def options(src, other):
        key = (src, other)
        if key in option_cache:
            return option_cache[key]
        otherp = positions[other]
        finalsites = []
        for d in DIRS:
            q = otherp + d
            if q in ix and (otherp - d in ix or otherp + 2 * d in ix):
                finalsites.append(ix[q])
        future = [
            min((distances[i][t] for t in finalsites), default=35.0)
            for i in range(size)
        ]
        departures = railindices if one[src] & railmask else [src]
        result = []
        for dep in departures:
            travel = raildist[src][dep] if dep != src else 0.0
            if travel >= 1000:
                continue
            for mid, dst in jumps[dep]:
                if dst == other:
                    continue
                remainder = 0.0 if mid == other else future[dst]
                base = travel + 2.0 + remainder
                if base < 80:
                    result.append((base, mid, dst))
        result.sort()
        result = result[:24]
        option_cache[key] = result
        return result

    def estimate(st):
        if st in heuristic_cache:
            return heuristic_cache[st]
        occupied = st[0] | st[1] | st[2] | st[4]
        pads = list(bits(st[3]))
        peglist = list(bits(st[0] | st[1] | st[2]))
        mobileboxes = list(bits(st[4] & st[3]))
        helper_penalty = {}
        for p in peglist:
            legal = any(
                occupied & one[mid]
                and not occupied & one[dst]
                and (one[dst] & normalmask or one[dst] & st[3])
                for mid, dst in jumps[p]
            )
            helper_penalty[p] = 0.0 if legal else 3.5

        value = 0.0
        for group in st[:2]:
            members = list(bits(group))
            if len(members) < 2:
                continue
            best = 45.0
            for src in members:
                for other in members:
                    if src == other:
                        continue
                    for base, mid, dst in options(src, other):
                        if occupied & one[dst]:
                            continue
                        cost = base
                        if not occupied & one[mid]:
                            work = 1000.0
                            for helper in peglist:
                                if helper != src:
                                    work = min(
                                        work,
                                        0.70 * distances[helper][mid]
                                        + helper_penalty[helper]
                                    )
                            if one[mid] & railmask:
                                for helper in mobileboxes:
                                    work = min(
                                        work, 0.65 * raildist[helper][mid]
                                    )
                            cost += 2.0 + min(work, 35.0)
                        if one[dst] & railmask and not one[dst] & st[3]:
                            cost += 0.40 * min(
                                (raildist[p][dst] for p in pads),
                                default=70.0
                            )
                        best = min(best, cost)
            value += best

        if exportindices:
            pieces = list(bits(st[0] | st[1]))
            near = min((edist[p] for p in pieces), default=45.0)
            guided = 65.0
            for src in pieces:
                for mid, dst in jumps[src]:
                    if occupied & one[dst]:
                        continue
                    work = 0.0
                    if not occupied & one[mid]:
                        effort = 1000.0
                        for helper in peglist:
                            if helper != src:
                                effort = min(
                                    effort,
                                    0.65 * distances[helper][mid]
                                    + helper_penalty[helper]
                                )
                        if one[mid] & railmask:
                            for helper in mobileboxes:
                                effort = min(
                                    effort, 0.65 * raildist[helper][mid]
                                )
                        work += min(35.0, effort) + 2.0
                    if one[dst] & railmask and not one[dst] & st[3]:
                        work += 0.4 * min(
                            (raildist[p][dst] for p in pads), default=70.0
                        )
                    guided = min(guided, 2.0 + edist[dst] + work)
                if one[src] & st[3]:
                    for ai in range(4):
                        dst = neighbours[ai][src]
                        if dst >= 0:
                            guided = min(guided, 1.0 + edist[dst])
            value += (1.2 if exploring else 0.12) * near
            value += (0.8 if exploring else 0.08) * guided

        heuristic_cache[st] = value
        return value

    def decode(st):
        return tuple(
            tuple(positions[i] for i in bits(mask)) for mask in st
        )

    def make_plan(end, parents):
        raw = reconstruct(end, parents)
        result = []
        tx, ty = m["tx"], m["ty"]
        previous = m["s"]
        for action, st in raw:
            decoded = decode(st)
            if action[0] == "a":
                ai = action[1]
                _, moves = slide(previous, DIRS[ai], rails)
                tx, ty = camera_after(previous, moves, DIRS[ai], tx, ty, m)
            result.append((action, (decoded, tx, ty)))
            previous = decoded
        return result

    parents, costs = {root: None}, {root: 0.0}
    h0 = estimate(root)
    heap = [(11.0 * max(0, startn - goal) + h0, 0.0, 0, root)]
    best, bestrank, serial = root, -1e30, 0
    expansions = 0
    decoded_visits = {}

    def visits(st):
        if st not in decoded_visits:
            decoded_visits[st] = m["visits"].get(
                (decode(st), m["tx"], m["ty"]), 0
            )
        return decoded_visits[st]

    while heap and expansions < 80000 and len(costs) < 170000:
        if expansions % 32 == 0 and time.perf_counter() >= deadline:
            break
        _, g, _, st = heapq.heappop(heap)
        if g != costs.get(st):
            continue
        expansions += 1
        n = st[0].bit_count() + st[1].bit_count()

        if st != root:
            previous, last = parents[st]
            if last[0] == "j":
                gi, src, dst = last[1:]
                if n <= goal and (n < startn or not unknown):
                    return make_plan(st, parents)
                if one[ix[dst]] & exportmask:
                    return make_plan(st, parents)
            if st[0] & st[3] & fringemask:
                if m["followx"] != 0 or m["followy"] != 0:
                    return make_plan(st, parents)

        occupied = st[0] | st[1] | st[2] | st[4]
        destmask = normalmask | st[3]
        successors = []
        mobility = 0

        for gi in range(3):
            for src in bits(st[gi] & visiblemask):
                for mid, dst in jumps[src]:
                    if (not occupied & one[mid] or occupied & one[dst]
                            or not destmask & one[dst]
                            or not visiblemask & one[dst]):
                        continue
                    action = ("j", gi, positions[src], positions[dst])
                    decoded = None
                    if m["banned"]:
                        decoded = decode(st)
                        if (decoded, positions[src], positions[dst]) in m["banned"]:
                            continue
                    ns = list(st)
                    ns[gi] = (ns[gi] ^ one[src]) | one[dst]
                    if gi != 2 and ns[gi] & one[mid]:
                        ns[gi] ^= one[mid]
                    ns = tuple(ns)
                    mobility += int(gi != 2)
                    successors.append((ns, action, 2.0 if gi != 2 else 2.5))

        for ai, d in enumerate(DIRS):
            ns = list(st)
            padlist = list(bits(st[3]))
            if d > 0:
                padlist.reverse()
            changed = False
            for src in padlist:
                dst = neighbours[ai][src]
                if dst < 0 or ns[3] & one[dst]:
                    continue
                oldbit, newbit = one[src], one[dst]
                ns[3] = (ns[3] ^ oldbit) | newbit
                for gi in (0, 1, 2, 4):
                    if ns[gi] & oldbit:
                        ns[gi] = (ns[gi] ^ oldbit) | newbit
                changed = True
            if not changed:
                continue
            ns = tuple(ns)
            outside = ns[3] & fringemask
            if outside and not outside & ns[0]:
                continue
            successors.append((ns, ("a", ai), 1.0))

        ex = estimate(st)
        if st != root:
            rank = (
                110.0 * (startn - n) + 2.7 * (h0 - ex)
                + 0.3 * min(mobility, 8)
                - 0.40 * g - 6.0 * min(visits(st), 5)
            )
            if rank > bestrank:
                bestrank, best = rank, st

        if g > 130:
            continue
        for ns, action, cost in successors:
            ng = g + cost + min(15.0, 3.0 * visits(ns))
            if ng >= costs.get(ns, float("inf")):
                continue
            costs[ns] = ng
            parents[ns] = (st, action)
            serial += 1
            nn = ns[0].bit_count() + ns[1].bit_count()
            h = 11.0 * max(0, nn - goal) + estimate(ns)
            heapq.heappush(heap, (ng + h, ng, serial, ns))

    return make_plan(best, parents) if best != root else []


def remember(m):
    m["history"].append((m["s"], m["tx"], m["ty"]))
    if len(m["history"]) > 1000:
        del m["history"][:200]


def undo(available, m):
    m["target"] = None
    m["plan"] = []
    m["plan_start"] = None
    m["lastjump"] = None
    if "ACTION7" in available:
        if m["history"]:
            s, tx, ty = m["history"].pop()
            m["s"], m["tx"], m["ty"] = s, tx, ty
        m["pending"] = {
            "kind": "u", "expected_camera": (m["tx"], m["ty"])
        }
        return "ACTION7", None, None
    m.clear()
    return "RESET", None, None


def act(obs, available, memory):
    obs = np.asarray(obs)
    m = memory

    if "s" not in m:
        ox, oy = lattice(obs)
        local = read_image(obs, ox, oy)
        m.update({
            "N": set(), "R": set(), "E": set(),
            "s": ((), (), (), (), ()),
            "tx": ox, "ty": oy, "maxtx": ox, "maxty": oy,
            "followx": None, "followy": None,
            "plan": [], "plan_start": None,
            "target": None, "pending": None,
            "banned": set(), "visits": {}, "landings": set(),
            "history": [], "last_seen": None, "lastjump": None,
            "blue": False, "red": False
        })
        view = convert(local, 0, 0)
        merge(view, m)
    else:
        ox, oy = lattice(obs, (m["tx"], m["ty"]))
        local = read_image(obs, ox, oy)
        kx, ky, tx, ty = locate(local, ox, oy, m)
        view = convert(local, kx, ky)
        pending = m.get("pending")

        if view["gray"] and pending and pending["kind"] == "j":
            before = pending["before"]
            m["banned"].add((before, pending["src"], pending["dst"]))
            m["s"] = before
            return undo(available, m)

        if pending and pending["kind"] == "a":
            before, expected = pending["before"], m["s"]
            contradicted = view["pads"].intersection(
                set(before[3]) - set(expected[3])
            )
            if contradicted:
                m["s"] = before
                m["E"].update(pending.get("speculative", ()))
                m["plan"] = []
                m["plan_start"] = None
            else:
                moves, d = pending["moves"], pending["direction"]
                if any(p in moves for p in before[0]):
                    oldtx, oldty = pending["old_camera"]
                    if abs(d) == 1:
                        rate = (tx - oldtx) / d
                        if abs(rate) <= 12:
                            if rate != 0 or m["followx"] is None:
                                m["followx"] = rate
                    else:
                        rate = (ty - oldty) / (d // STRIDE)
                        if abs(rate) <= 12:
                            if rate != 0 or m["followy"] is None:
                                m["followy"] = rate

        if pending and pending["kind"] == "j":
            m["landings"].add(pending["dst"])
            m["lastjump"] = (
                pending["before"], pending["src"], pending["dst"]
            )

        m["tx"], m["ty"] = tx, ty
        m["maxtx"] = max(m["maxtx"], tx)
        m["maxty"] = max(m["maxty"], ty)
        merge(view, m)
        m["pending"] = None

    s = m["s"]
    root = (s, m["tx"], m["ty"])
    if root != m.get("last_seen"):
        m["visits"][root] = m["visits"].get(root, 0) + 1
        m["last_seen"] = root
        if len(m["visits"]) > 2500:
            for key in list(m["visits"])[:700]:
                if key != root:
                    del m["visits"][key]

    target = m.get("target")
    if target is not None:
        gi, src, dst, after, before = target
        m["target"] = None
        if (s == before and dst in view["markers"]
                and visible(dst, m["tx"], m["ty"])):
            x, y = point(dst, m["tx"], m["ty"])
            remember(m)
            oldcamera = (m["tx"], m["ty"])
            m["s"] = after
            m["plan_start"] = (after, m["tx"], m["ty"])
            m["pending"] = {
                "kind": "j", "before": before,
                "src": src, "dst": dst,
                "old_camera": oldcamera,
                "expected_camera": oldcamera
            }
            return "ACTION6", int(y), int(x)

        m["banned"].add((before, src, dst))
        m["plan"] = []
        m["plan_start"] = None

    if (m["blue"] or m["red"]) and not can_progress(s, m["N"], m["R"]):
        if m.get("lastjump") is not None:
            m["banned"].add(m["lastjump"])
        if "ACTION7" in available or "RESET" in available:
            return undo(available, m)

    root = (m["s"], m["tx"], m["ty"])
    s = m["s"]
    if m.get("plan_start") != root:
        m["plan"] = []
        m["plan_start"] = root

    if not m["plan"]:
        if m["red"]:
            m["plan"] = multicolour_search(m)
        else:
            m["plan"] = ordinary_search(m)
        m["plan_start"] = root

    while m["plan"]:
        action, after_node = m["plan"].pop(0)
        after, ntx, nty = after_node

        if action[0] == "j":
            _, gi, src, dst = action
            if (not visible(src, m["tx"], m["ty"])
                    or not visible(dst, m["tx"], m["ty"])
                    or view["pegs"].get(src) != COLORS[gi]):
                m["plan"] = []
                break
            if "ACTION6" not in available:
                break
            x, y = point(src, m["tx"], m["ty"])
            remember(m)
            m["target"] = (gi, src, dst, after, s)
            return "ACTION6", int(y), int(x)

        _, ai = action
        name = NAMES[ai]
        if name not in available:
            continue
        fringe, _, _ = frontier_info(m)
        predicted, moves = slide(s, DIRS[ai], m["R"] | fringe)
        if predicted is None:
            m["plan"] = []
            break
        # Use the same transition for execution and prediction.
        after = predicted
        ntx, nty = camera_after(
            s, moves, DIRS[ai], m["tx"], m["ty"], m
        )
        remember(m)
        oldcamera = (m["tx"], m["ty"])
        m["pending"] = {
            "kind": "a", "before": s,
            "speculative": tuple(set(after[3]).intersection(fringe)),
            "moves": moves, "direction": DIRS[ai],
            "old_camera": oldcamera,
            "expected_camera": (ntx, nty)
        }
        m["s"] = after
        m["plan_start"] = (after, ntx, nty)
        return name, None, None

    s = m["s"]
    occupied = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
    destinations = m["N"] | set(s[3])
    fringe, targets, padtargets = frontier_info(m)
    choices = []

    for gi, group in enumerate(s[:3]):
        for src in group:
            if (not visible(src, m["tx"], m["ty"])
                    or view["pegs"].get(src) != COLORS[gi]):
                continue
            for d in DIRS:
                mid, dst = src + d, src + 2 * d
                if (mid not in occupied or dst not in destinations
                        or dst in occupied
                        or not visible(dst, m["tx"], m["ty"])
                        or (s, src, dst) in m["banned"]):
                    continue
                after = jump(s, gi, src, mid, dst)
                if (m["blue"] and (fringe or targets)
                        and not can_progress(after, m["N"], m["R"])):
                    continue
                node = (after, m["tx"], m["ty"])
                score = (
                    20 * (count(s) - count(after))
                    + 12 * (dst in targets or dst in padtargets)
                    + 2 * (dst not in m["landings"])
                    - 7 * m["visits"].get(node, 0)
                    - 2 * (gi == 2)
                )
                choices.append((score, gi, src, dst, after))

    if choices and "ACTION6" in available:
        _, gi, src, dst, after = max(choices, key=lambda item: item[0])
        x, y = point(src, m["tx"], m["ty"])
        remember(m)
        m["target"] = (gi, src, dst, after, s)
        m["plan"] = []
        return "ACTION6", int(y), int(x)

    if "ACTION7" in available or "RESET" in available:
        return undo(available, m)
    for name in NAMES:
        if name in available:
            return name, None, None
    return "ACTION6", 32, 32

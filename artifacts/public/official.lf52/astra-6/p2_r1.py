import numpy as np
import heapq
import time
from collections import deque, Counter

_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_BALL = (
    (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4),
)
_CORNERS = ((1, 1), (4, 1), (1, 4), (4, 4))
_INF = 100000


def _read_frame(obs, old_origin=None):
    a = np.asarray(obs)
    scores = np.zeros((6, 6), dtype=float)
    for y in range(1, 59):
        for x in range(59):
            c = int(a[y, x])
            if c not in (0, 11):
                continue
            if not (a[y, x + 5] == c and a[y + 5, x] == c
                    and a[y + 5, x + 5] == c):
                continue
            left, right = a[y:y + 6, x], a[y:y + 6, x + 5]
            if c == 0:
                good = np.count_nonzero((left == 0) | (left == 3))
                good += np.count_nonzero((right == 0) | (right == 3))
            else:
                good = np.count_nonzero(left == 11)
                good += np.count_nonzero(right == 11)
            if good >= 9:
                scores[y % 6, x % 6] += 1
    if old_origin is not None:
        scores[old_origin[1] % 6, old_origin[0] % 6] += 0.1
    ym, xm = np.unravel_index(int(np.argmax(scores)), scores.shape)
    result = {k: set() for k in
              ("floor", "cart", "rail", "pillar", "selected", "markers", "seen")}
    result["peg"] = {}
    result["mod"] = int(xm), int(ym)

    for qy in range(-1, (63 - ym) // 6 + 1):
        y = int(ym + 6 * qy)
        for qx in range(-1, (63 - xm) // 6 + 1):
            x = int(xm + 6 * qx)
            p = qx, qy
            values = {}
            for ry in range(6):
                sy = y + ry
                if sy < 1 or sy >= 64:
                    continue
                for rx in range(6):
                    sx = x + rx
                    if 0 <= sx < 64:
                        values[rx, ry] = int(a[sy, sx])
            if not values:
                continue
            if 0 <= x + 2 and x + 3 < 64 and 1 <= y + 2 and y + 3 < 64:
                result["seen"].add(p)
            cnt = Counter(values.values())
            border = [v for (rx, ry), v in values.items()
                      if rx in (0, 5) or ry in (0, 5)]
            nz, ng, ny = border.count(0), border.count(3), border.count(11)
            if ny >= 2 or (cnt[12] >= 4 and cnt[11] >= 1):
                result["cart"].add(p)
                result["rail"].add(p)
            elif ((nz >= min(6, len(border)) and cnt[0] >= 4)
                  or (nz >= 2 and ng >= 8)):
                result["floor"].add(p)
            center = [values[k] for k in
                      ((2, 2), (3, 2), (2, 3), (3, 3)) if k in values]
            if len(center) >= 2 and all(v == 5 for v in center):
                result["rail"].add(p)
            bm = [values[k] for k in _BALL if k in values]
            corners = [values[k] for k in _CORNERS if k in values]
            if len(bm) >= 3:
                c = bm[0]
                if (c in (14, 8, 9, 2, 4) and all(v == c for v in bm)
                        and all(v != c for v in corners)):
                    result["peg"][p] = c
            if cnt[15] >= 3 and cnt[7] >= 1:
                result["pillar"].add(p)
            if ng >= 8:
                result["selected"].add(p)
            ring = [values[k] for k in
                    ((2, 1), (3, 1), (1, 2), (4, 2),
                     (1, 3), (4, 3), (2, 4), (3, 4)) if k in values]
            if (len(ring) >= 3 and all(v == 2 for v in ring)
                    and center and any(v != 2 for v in center)):
                result["markers"].add(p)
    return result


def _new_model():
    return {"floor": set(), "rail": set(), "seen": set(),
            "cart": set(), "pillar": set(), "peg": {}, "origin": (0, 0)}


def _align(raw, model, initialized):
    xm, ym = raw["mod"]
    if not initialized:
        return xm, ym
    ox, oy = model["origin"]
    base = int(round((ox - xm) / 6.0)), int(round((oy - ym) / 6.0))
    votes = Counter({base: 2})
    for q, c in raw["peg"].items():
        for p, pc in model["peg"].items():
            if c == pc or c == 2 or (c == 4 and pc == 14):
                votes[q[0] - p[0], q[1] - p[1]] += 12
    for q in raw["cart"]:
        for p in model["cart"]:
            votes[q[0] - p[0], q[1] - p[1]] += 7
    for q in raw["floor"]:
        for p in model["floor"]:
            k = q[0] - p[0], q[1] - p[1]
            if abs(k[0] - base[0]) <= 24 and abs(k[1] - base[1]) <= 16:
                votes[k] += 1
    candidates = [k for k, _ in votes.most_common(90)]
    if base not in candidates:
        candidates.append(base)
    best, best_score = base, -1e30
    for kx, ky in candidates:
        score = -0.045 * (abs(kx - base[0]) + abs(ky - base[1]))
        for q in raw["floor"]:
            p = q[0] - kx, q[1] - ky
            if p in model["floor"]:
                score += 2.2
            elif p in model["seen"]:
                score -= 3
        for q in raw["rail"]:
            if (q[0] - kx, q[1] - ky) in model["rail"]:
                score += 0.65
        for q, c in raw["peg"].items():
            p = q[0] - kx, q[1] - ky
            pc = model["peg"].get(p)
            if pc is not None and (pc == c or c in (2, 4)):
                score += 24
            elif p in model["seen"]:
                score -= 10
        for q in raw["cart"]:
            p = q[0] - kx, q[1] - ky
            if p in model["cart"]:
                score += 13
            elif p in model["seen"]:
                score -= 5
        for q in raw["pillar"]:
            if (q[0] - kx, q[1] - ky) in model["pillar"]:
                score += 8
        if score > best_score:
            best_score, best = score, (kx, ky)
    return xm + 6 * best[0], ym + 6 * best[1]


def _incorporate(raw, model, origin):
    xm, ym = raw["mod"]
    kx, ky = (origin[0] - xm) // 6, (origin[1] - ym) // 6

    def tr(p):
        return p[0] - kx, p[1] - ky

    mapped = {}
    for name in ("floor", "rail", "cart", "pillar",
                 "seen", "selected", "markers"):
        mapped[name] = {tr(p) for p in raw[name]}
    mapped["peg"] = {tr(p): c for p, c in raw["peg"].items()}
    old_size = len(model["floor"]), len(model["rail"]), len(model["seen"])
    for name in ("floor", "rail", "seen"):
        model[name].update(mapped[name])
    model["origin"] = origin

    def visible(p):
        x, y = origin[0] + 6 * p[0], origin[1] + 6 * p[1]
        return 0 <= x + 1 and x + 4 < 64 and 1 <= y + 1 and y + 4 < 64

    for name in ("cart", "pillar"):
        model[name] = {p for p in model[name]
                       if not visible(p) or p in mapped[name]} | mapped[name]
    if not any(c == 2 for c in mapped["peg"].values()):
        model["peg"] = {p: c for p, c in model["peg"].items()
                        if not visible(p) or p in mapped["peg"]}
        for p, c in mapped["peg"].items():
            model["peg"][p] = 14 if c == 4 else c
    new_size = len(model["floor"]), len(model["rail"]), len(model["seen"])
    return mapped, old_size != new_size


def _key(model):
    pegs = model["peg"]
    groups = [
        tuple(sorted(p for p, c in pegs.items() if c not in (8, 9))),
        tuple(sorted(p for p, c in pegs.items() if c == 8)),
        tuple(sorted(p for p, c in pegs.items() if c == 9)),
    ]
    pc = model["cart"] & model["pillar"]
    return tuple(groups + [tuple(sorted(model["cart"] - pc)), tuple(sorted(pc))])


def _snapshot(model):
    return dict(model["peg"]), set(model["cart"]), set(model["pillar"]), model["origin"]


def _restore(model, snap):
    model["peg"], model["cart"] = dict(snap[0]), set(snap[1])
    model["pillar"], model["origin"] = set(snap[2]), snap[3]


def _apply(model, action):
    if action[0] == "j":
        src, dst = action[1], action[2]
        if src not in model["peg"]:
            return
        c = model["peg"].pop(src)
        mid = (src[0] + dst[0]) // 2, (src[1] + dst[1]) // 2
        if c != 9 and model["peg"].get(mid) == c:
            del model["peg"][mid]
        model["peg"][dst] = c
    else:
        dx, dy = _DIRS[action[1]]
        order = sorted(model["cart"],
                       key=lambda p: p[0] * dx + p[1] * dy, reverse=True)
        occupied = set(model["cart"])
        for p in order:
            q = p[0] + dx, p[1] + dy
            if q not in model["rail"] or q in occupied:
                continue
            occupied.remove(p)
            occupied.add(q)
            if p in model["peg"]:
                model["peg"][q] = model["peg"].pop(p)
            if p in model["pillar"]:
                model["pillar"].remove(p)
                model["pillar"].add(q)
        model["cart"] = occupied


def _click_point(model, p):
    x, y = model["origin"][0] + 6 * p[0], model["origin"][1] + 6 * p[1]
    if x > 63 or y > 63 or x + 5 < 0 or y + 5 < 1:
        return None
    return (int((max(1, y) + min(63, y + 5)) // 2),
            int((max(0, x) + min(63, x + 5)) // 2))


def _geometry(model, memory):
    floor, rail = model["floor"], model["rail"]
    points = floor | rail | model["pillar"] | set(model["peg"]) | model["cart"]
    signature = frozenset(floor), frozenset(rail), frozenset(points)
    cached = memory.get("geometry")
    if cached is not None and cached["signature"] == signature:
        return cached
    cells = sorted(points)
    index = {p: i for i, p in enumerate(cells)}
    n = len(cells)
    fset, rset = {index[p] for p in floor}, {index[p] for p in rail}
    playable = fset | rset
    neighbors = [[-1] * 4 for _ in cells]
    jumps, relaxed = [[] for _ in cells], [[] for _ in cells]
    for i, (x, y) in enumerate(cells):
        for d, (dx, dy) in enumerate(_DIRS):
            mid = index.get((x + dx, y + dy), -1)
            dst = index.get((x + 2 * dx, y + 2 * dy), -1)
            if mid in rset:
                neighbors[i][d] = mid
            if mid >= 0 and dst in playable:
                jumps[i].append((mid, dst))
                relaxed[i].append(dst)
            if i in rset and mid in rset:
                relaxed[i].append(mid)

    distances = []
    for start in range(n):
        dist = [999] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            i = queue.popleft()
            nd = dist[i] + 1
            for j in relaxed[i]:
                if dist[j] == 999:
                    dist[j] = nd
                    queue.append(j)
        distances.append(dist)

    approaches = [[] for _ in cells]
    for j, (x, y) in enumerate(cells):
        for dx, dy in _DIRS:
            a = index.get((x - dx, y - dy), -1)
            b = index.get((x + dx, y + dy), -1)
            if a in playable and b in playable:
                approaches[j].append(a)
    pair = [[40] * n for _ in cells]
    for i in range(n):
        for j in range(n):
            if approaches[j]:
                pair[i][j] = min(
                    40, 1 + min(distances[i][a] for a in approaches[j]))
    for i in range(n):
        for j in range(i):
            pair[i][j] = pair[j][i] = min(pair[i][j], pair[j][i])

    components, component_of = [], {}
    for i in rset:
        if i in component_of:
            continue
        ci = len(components)
        component_of[i] = ci
        group, queue = [], [i]
        while queue:
            p = queue.pop()
            group.append(p)
            for q in neighbors[p]:
                if q >= 0 and q not in component_of:
                    component_of[q] = ci
                    queue.append(q)
        components.append(tuple(group))

    rail_dist = [[999] * n for _ in cells]
    for i in rset:
        row = rail_dist[i]
        row[i] = 0
        queue = deque([i])
        while queue:
            p = queue.popleft()
            for q in neighbors[p]:
                if q >= 0 and row[q] == 999:
                    row[q] = row[p] + 1
                    queue.append(q)

    cached = {
        "signature": signature, "cells": cells, "index": index,
        "floor": fset, "rail": rset, "neighbors": neighbors,
        "jumps": jumps, "distances": distances, "pair": pair,
        "components": components, "component_of": component_of,
        "rail_dist": rail_dist,
    }
    memory["geometry"] = cached
    return cached


def _build_pattern(geo, root, pillars, fixed, targets, memory, deadline):
    """Joint two-peg distance table, relaxing only platform arrangements."""
    n = len(geo["cells"])
    compof = geo["component_of"]
    empty_components = frozenset(compof[p] for p in root[3] if p in compof)
    pillar_components = frozenset(compof[p] for p in root[4] if p in compof)
    ck = (geo["signature"], frozenset(pillars), fixed, frozenset(targets),
          empty_components, pillar_components)
    cache = memory.setdefault("patterns", {})
    if ck in cache:
        return cache[ck]

    floor, rail = geo["floor"], geo["rail"]
    valid_set = ((floor | {p for p in rail
                          if compof.get(p) in empty_components}) - pillars)
    if fixed >= 0:
        valid_set.discard(fixed)
    valid = sorted(valid_set)
    if not valid:
        return None
    support = set(pillars)
    support.update(p for p in rail if compof.get(p) in pillar_components)
    vmask = [i in valid_set for i in range(n)]
    smask = [i in support for i in range(n)]
    edges = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    rail_edges = [[] for _ in range(n)]
    for p in valid:
        for mid, dst in geo["jumps"][p]:
            if dst in valid_set:
                edges[p].append((mid, dst))
                incoming[dst].append((mid, p))
        if p in rail:
            rail_edges[p] = [q for q in geo["neighbors"][p]
                             if q >= 0 and q in valid_set]

    winning = [[] for _ in range(n)]
    if fixed >= 0:
        for mid, dst in geo["jumps"][fixed]:
            if mid in valid_set and dst in valid_set:
                winning[mid].append((fixed, dst, mid))
        for p in valid:
            for mid, dst in geo["jumps"][p]:
                if mid == fixed and dst in valid_set:
                    winning[p].append((p, dst, fixed))

    dist = [_INF] * (n * n)
    heap = []
    if fixed < 0:
        for g in valid:
            for r in valid:
                if g != r and (g in targets or r in targets):
                    k = g * n + r
                    dist[k] = 0
                    heap.append((0, k))
    else:
        for g in valid:
            if not winning[g]:
                continue
            for r in valid:
                if r != g and any(dst != r for _, dst, _ in winning[g]):
                    k = g * n + r
                    dist[k] = 2
                    heap.append((2, k))
    heapq.heapify(heap)
    pops = 0
    while heap:
        d, k = heapq.heappop(heap)
        if d != dist[k]:
            continue
        pops += 1
        if pops % 2048 == 0 and time.monotonic() > deadline:
            return None
        g, r = divmod(k, n)
        d1, d2 = d + 1, d + 2
        for prev in rail_edges[g]:
            if prev != r:
                z = prev * n + r
                if d1 < dist[z]:
                    dist[z] = d1
                    heapq.heappush(heap, (d1, z))
        for prev in rail_edges[r]:
            if prev != g:
                z = g * n + prev
                if d1 < dist[z]:
                    dist[z] = d1
                    heapq.heappush(heap, (d1, z))
        for mid, prev in incoming[g]:
            if prev != r and mid != fixed and (mid == r or smask[mid]):
                z = prev * n + r
                if d2 < dist[z]:
                    dist[z] = d2
                    heapq.heappush(heap, (d2, z))
        for mid, prev in incoming[r]:
            if prev != g and (mid == g or mid == fixed or smask[mid]):
                z = g * n + prev
                if d2 < dist[z]:
                    dist[z] = d2
                    heapq.heappush(heap, (d2, z))

    pattern = {
        "n": n, "fixed": fixed, "targets": set(targets),
        "dist": dist, "edges": edges, "rail_edges": rail_edges,
        "winning": winning, "support": smask,
        "valid": vmask, "options": {},
    }
    if len(cache) >= 8:
        del cache[next(iter(cache))]
    cache[ck] = pattern
    return pattern


def _pattern_options(pattern, g, r):
    """Relaxed transitions that strictly approach the joint goal."""
    n = pattern["n"]
    k = g * n + r
    cached = pattern["options"].get(k)
    if cached is not None:
        return cached
    dist, fixed = pattern["dist"], pattern["fixed"]
    current = dist[k]
    support = pattern["support"]
    out = []
    if fixed >= 0:
        for src, dst, mid in pattern["winning"][g]:
            if dst != r:
                out.append((0, 2, src, dst, mid, -1, -1))
    for who, src, other in ((0, g, r), (1, r, g)):
        for mid, dst in pattern["edges"][src]:
            if dst == other:
                continue
            if who == 0 and mid == fixed:
                continue
            if not (mid == other or support[mid] or (who == 1 and mid == fixed)):
                continue
            ng, nr = (dst, r) if who == 0 else (g, dst)
            value = dist[ng * n + nr]
            if value < current:
                out.append((value, 2, src, dst, mid, ng, nr))
        for dst in pattern["rail_edges"][src]:
            if dst == other:
                continue
            ng, nr = (dst, r) if who == 0 else (g, dst)
            value = dist[ng * n + nr]
            if value < current:
                out.append((value, 1, src, dst, -1, ng, nr))
    pattern["options"][k] = out
    return out


def _pattern_positions(pattern, state):
    if len(state[1]) != 1:
        return None
    fixed = pattern["fixed"]
    if fixed < 0:
        if len(state[0]) != 1:
            return None
        return state[0][0], state[1][0]
    if len(state[0]) != 2 or fixed not in state[0]:
        return None
    return next(p for p in state[0] if p != fixed), state[1][0]


def _search(model, memory, deadline):
    geo = _geometry(model, memory)
    cells, index = geo["cells"], geo["index"]
    if not cells:
        return []
    floor, rail = geo["floor"], geo["rail"]
    jumps, neighbors = geo["jumps"], geo["neighbors"]
    pair, distances = geo["pair"], geo["distances"]
    compof = geo["component_of"]
    world_root = _key(model)
    full_root = tuple(tuple(sorted(index[p] for p in group)) for group in world_root)
    pillars = {index[p] for p in model["pillar"] - model["cart"]}
    frontier = set()
    for p in model["floor"] | model["rail"]:
        if any((p[0] + dx, p[1] + dy) not in model["seen"] for dx, dy in _DIRS):
            frontier.add(index[p])

    target_count = int(bool(full_root[0])) + int(bool(full_root[1]))
    start_count = len(full_root[0]) + len(full_root[1])
    if not start_count:
        return []
    exploring = start_count == target_count and bool(frontier)
    start_heroes = set(full_root[0]) | set(full_root[1])
    targets = (frontier & floor) - pillars
    if not targets:
        targets = frontier - pillars

    if exploring:
        extra = set()
        ox, oy = model["origin"]
        for p in floor - pillars:
            x, y = cells[p]
            sx, sy = ox + 6 * x + 2, oy + 6 * y + 2
            if not (sx >= 52 or sx <= 10 or sy >= 55 or sy <= 8):
                continue
            if any(abs(x - cells[q][0]) + abs(y - cells[q][1]) <= 1
                   for q in targets):
                extra.add(p)
        targets |= extra
        already = {index[p] for p in memory.get("peg_seen", set()) if p in index}
        fresh = targets - already - start_heroes
        targets = fresh if fresh else targets - start_heroes
        if not targets:
            exploring = False

    pattern = None
    if (len(full_root[1]) == 1 and not full_root[2]
            and len(full_root[0]) in (1, 2)
            and time.monotonic() < deadline - 1):
        choices = []
        if exploring and len(full_root[0]) == 1:
            choices = [-1]
        elif len(full_root[0]) == 2:
            rx, ry = cells[full_root[1][0]]
            choices = sorted(
                full_root[0],
                key=lambda p: abs(cells[p][0] - rx) + abs(cells[p][1] - ry),
                reverse=True)
        for fixed in choices:
            if fixed >= 0 and fixed in full_root[3] + full_root[4]:
                continue
            p = _build_pattern(geo, full_root, pillars, fixed,
                               targets if fixed < 0 else set(),
                               memory, deadline - 0.5)
            if p is None:
                continue
            positions = _pattern_positions(p, full_root)
            if positions is not None:
                g, r = positions
                if p["dist"][g * p["n"] + r] < _INF:
                    pattern = p
                    break

    keep_components = None
    if pattern is not None:
        # Follow an optimal relaxed cooperative route to identify which rail
        # components can influence this particular subproblem.
        keep_components = set()
        for p in full_root[0] + full_root[1]:
            if p in compof and p in full_root[3] + full_root[4]:
                keep_components.add(compof[p])
        g, r = _pattern_positions(pattern, full_root)
        for _ in range(400):
            value = pattern["dist"][g * pattern["n"] + r]
            if value == 0:
                break
            options = _pattern_options(pattern, g, r)
            if not options:
                break
            options = sorted(options, key=lambda o: (o[0] + o[1], o[1]))
            chosen = options[0]
            _, cost, src, dst, mid, ng, nr = chosen
            if cost == 1:
                if src in compof:
                    keep_components.add(compof[src])
            else:
                if dst not in floor and dst in compof:
                    keep_components.add(compof[dst])
                if mid not in pillars and mid not in (g, r, pattern["fixed"]):
                    if mid in compof:
                        keep_components.add(compof[mid])
            if ng < 0:
                break
            g, r = ng, nr

    elif full_root[1] and full_root[3] + full_root[4]:
        comps = geo["components"]
        occupied_components = {compof[p] for p in full_root[3] + full_root[4]
                               if p in compof}
        if len(occupied_components) > 1:
            keep_components = set()
            heroes = full_root[0] + full_root[1]
            for p in heroes:
                if p in compof:
                    keep_components.add(compof[p])
                px, py = cells[p]
                choices = []
                for ci in occupied_components:
                    d = min(abs(px - cells[q][0]) + abs(py - cells[q][1])
                            for q in comps[ci])
                    choices.append((d, ci))
                if choices:
                    best = min(d for d, _ in choices)
                    keep_components.update(ci for d, ci in choices if d == best)
            if not exploring:
                minx = min(cells[p][0] for p in heroes)
                for ci in occupied_components:
                    if max(cells[q][0] for q in comps[ci]) > minx - 2:
                        keep_components.add(ci)

    def encode(key):
        try:
            groups = [tuple(sorted(index[p] for p in group)) for group in key]
        except KeyError:
            return None
        if keep_components is not None:
            for k in (3, 4):
                groups[k] = tuple(p for p in groups[k]
                                  if compof.get(p) in keep_components)
        return tuple(groups)

    root = encode(world_root)
    tabu = {}
    for wk, count in memory.get("visits", {}).items():
        ek = encode(wk)
        if ek is not None:
            tabu[ek] = max(tabu.get(ek, 0), count)
    bans = {}
    for wk, action in memory.get("bans", set()):
        ek = encode(wk)
        if ek is None:
            continue
        if action[0] == "j":
            if action[1] not in index or action[2] not in index:
                continue
            ea = "j", index[action[1]], index[action[2]]
        else:
            ea = action
        bans.setdefault(ek, set()).add(ea)

    frontier_dist = [0] * len(cells)
    dt = targets if targets else frontier
    if dt:
        for i in range(len(cells)):
            frontier_dist[i] = min(80, min(distances[i][j] for j in dt))
    hcache, help_cache = {}, {}

    def peg_h(state):
        k = state[0], state[1]
        if k not in hcache:
            total = 0.0
            for group in k:
                if len(group) > 1:
                    total += 0.5 * sum(
                        min(pair[p][q] for q in group if p != q) for p in group)
            hcache[k] = total
        return hcache[k]

    def blue_help(state):
        if not state[2]:
            return 0
        gk = state[0], state[1]
        wanted = help_cache.get(gk)
        if wanted is None:
            wanted = set()
            before = peg_h(state)
            occupied = set(state[0]) | set(state[1])
            for color in (0, 1):
                group = state[color]
                for p in group:
                    for mid, dst in jumps[p]:
                        if dst not in floor or dst in occupied:
                            continue
                        ns = list(state)
                        ns[color] = tuple(sorted([q for q in group if q != p] + [dst]))
                        if peg_h(tuple(ns)) + 0.01 < before:
                            wanted.add(mid)
            help_cache[gk] = wanted
        if not wanted:
            return 0
        return min(12, min(distances[b][w] for b in state[2] for w in wanted))

    rail_dist = geo["rail_dist"]
    pattern_root_value = 0
    if pattern is not None:
        pg, pr = _pattern_positions(pattern, root)
        pattern_root_value = pattern["dist"][pg * pattern["n"] + pr]

    def heuristic(state):
        count = len(state[0]) + len(state[1])
        if count < start_count and count <= target_count:
            return 0
        if pattern is not None:
            positions = _pattern_positions(pattern, state)
            if positions is not None:
                g, r = positions
                pd = pattern["dist"][g * pattern["n"] + r]
                if pd < _INF:
                    need = 80
                    actual_support = pillars | set(state[4]) | set(state[0]) | set(state[1])
                    for nv, cost, src, dst, mid, _, _ in _pattern_options(pattern, g, r):
                        if nv + cost > pd + 2:
                            continue
                        if cost == 1:
                            extra = 1 if dst in state[3] + state[4] else 0
                        else:
                            a = b = 0
                            if dst not in floor:
                                a = min((rail_dist[p][dst] for p in state[3]), default=80)
                            if mid not in actual_support:
                                b = min((rail_dist[p][mid] for p in state[4]), default=80)
                            extra = min(80, max(a, b))
                        need = min(need, extra)
                    if pd == 0:
                        need = 0
                    return 3.1 * pd + 1.8 * need
            return 3.1 * pattern_root_value + 10 + 3 * peg_h(state)
        if exploring:
            ds = [frontier_dist[p] for p in state[0] + state[1]]
            return 4.2 * min(ds) + 0.12 * sum(ds)
        return (31 * (count - target_count) +
                2.6 * peg_h(state) + 1.2 * blue_help(state))

    def successors(state):
        occupied = set(state[0]) | set(state[1]) | set(state[2])
        carts = set(state[3]) | set(state[4])
        objects = occupied | pillars | set(state[4])
        landing = floor | carts
        forbidden = bans.get(state, ())
        boarding_possible = False
        for color, group in enumerate(state[:3]):
            for src in group:
                for mid, dst in jumps[src]:
                    if mid not in objects:
                        continue
                    if dst in rail and dst not in objects:
                        boarding_possible = True
                    if dst not in landing or dst in objects:
                        continue
                    action = "j", src, dst
                    if action in forbidden:
                        continue
                    ns = list(state)
                    moving = [p for p in group if p != src]
                    if color != 2 and mid in group:
                        moving.remove(mid)
                    moving.append(dst)
                    ns[color] = tuple(sorted(moving))
                    yield tuple(ns), action, 2.0
        if not carts:
            return
        if not state[4] and not (occupied & carts) and not boarding_possible:
            return
        for d, (dx, dy) in enumerate(_DIRS):
            action = "a", d
            if action in forbidden:
                continue
            order = sorted(carts,
                           key=lambda p: cells[p][0] * dx + cells[p][1] * dy,
                           reverse=True)
            taken, moved = set(carts), {}
            for p in order:
                q = neighbors[p][d]
                if q < 0 or q in taken:
                    continue
                taken.remove(p)
                taken.add(q)
                moved[p] = q
            if moved:
                ns = tuple(tuple(sorted(moved.get(p, p) for p in group)) for group in state)
                if ns != state:
                    yield ns, action, 1.0

    def path_to(state, records):
        result = []
        while records[state][1] is not None:
            _, parent, action = records[state]
            if action[0] == "j":
                result.append(("j", cells[action[1]], cells[action[2]]))
            else:
                result.append(action)
            state = parent
        result.reverse()
        return result

    records = {root: (0.0, None, None)}
    heap = [(heuristic(root), 0, 0.0, root)]
    serial = expanded = 0
    best_front = best_partial = best_other = None
    front_score = partial_score = other_score = float("inf")
    limit = 350000 if pattern is not None or exploring else 245000

    while heap:
        if expanded % 128 == 0:
            if time.monotonic() >= deadline or len(records) > limit:
                break
        _, _, cost, state = heapq.heappop(heap)
        rec = records.get(state)
        if rec is None or abs(rec[0] - cost) > 1e-8:
            continue
        expanded += 1
        count = len(state[0]) + len(state[1])
        if state != root:
            heroes = set(state[0]) | set(state[1])
            if (count <= target_count and
                    (start_count > target_count or
                     (not frontier and rec[2][0] == "j"))):
                return path_to(state, records)
            if exploring and heroes & targets:
                return path_to(state, records)
            ph = peg_h(state)
            visits = tabu.get(state, 0)
            changed = state[:2] != root[:2]
            moved_to_edge = (heroes - start_heroes) & frontier
            if moved_to_edge:
                score = count * 95 + 2 * ph + cost + visits * 30
                if score < front_score:
                    front_score, best_front = score, state
            fd = min((frontier_dist[p] for p in heroes), default=0)
            if pattern is not None:
                score = heuristic(state) + 0.15 * cost + visits * 25
            elif exploring:
                score = 5 * fd + 0.2 * cost + visits * 25
            else:
                score = count * 100 + 2.3 * ph + 0.55 * fd + 0.04 * cost + visits * 24
            if changed and count <= start_count and score < partial_score:
                partial_score, best_partial = score, state
            score = heuristic(state) + 0.3 * cost + visits * 30 + (0 if changed else 1)
            if score < other_score:
                other_score, best_other = score, state

        for ns, action, edge_cost in successors(state):
            nc = cost + edge_cost
            old = records.get(ns)
            if old is not None and old[0] <= nc:
                continue
            records[ns] = nc, state, action
            serial += 1
            priority = nc + heuristic(ns)
            if ns != root:
                priority += min(20, 5 * tabu.get(ns, 0))
            heapq.heappush(heap, (priority, serial, nc, ns))

    if best_front is not None and (pattern is None or exploring):
        return path_to(best_front, records)
    if best_partial is not None:
        return path_to(best_partial, records)
    if best_other is not None:
        return path_to(best_other, records)
    return []


def _record(memory, name, abstract=None):
    model = memory["model"]
    memory.setdefault("history", []).append(
        (_snapshot(model), _key(model), abstract, name))
    memory["last"] = name, abstract
    if abstract is not None:
        _apply(model, abstract)
    memory["plan_key"] = _key(model)


def _undo(memory, available, reject=False):
    if "ACTION7" not in available or not memory.get("history"):
        return None
    snap, before, abstract, name = memory["history"].pop()
    if reject and abstract is not None:
        memory["bans"].add((before, abstract))
    _restore(memory["model"], snap)
    memory.pop("pending", None)
    memory["plan"] = []
    memory["plan_key"] = _key(memory["model"])
    memory["last"] = "ACTION7", None
    return "ACTION7", None, None


def _reset(memory, available):
    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    if "ACTION6" in available:
        memory.clear()
        return "ACTION6", 57, 7
    return next((a for a in available if a != "RESET"), "ACTION1"), None, None


def act(obs, available, memory):
    start_time = time.monotonic()
    available = set(available)
    initialized = "model" in memory
    if not initialized:
        memory.update({
            "model": _new_model(), "history": [], "visits": {},
            "bans": set(), "plan": [], "peg_seen": set(),
        })
    model = memory["model"]
    raw = _read_frame(obs, model["origin"] if initialized else None)
    origin = _align(raw, model, initialized)
    mapped, knowledge_changed = _incorporate(raw, model, origin)

    if any(c == 2 for c in raw["peg"].values()):
        result = _undo(memory, available, reject=True)
        if result is not None:
            return result
        return _reset(memory, available)

    key = _key(model)
    memory["peg_seen"].update(p for p, c in model["peg"].items() if c != 9)
    if key != memory.get("last_observed_key"):
        memory["visits"][key] = memory["visits"].get(key, 0) + 1
        memory["last_observed_key"] = key
    if key != memory.get("plan_key", key) or knowledge_changed:
        memory["plan"] = []

    pending = memory.pop("pending", None)
    if pending is not None:
        src, dst, before = pending
        if key == before and dst in mapped["markers"]:
            click = _click_point(model, dst)
            if click is not None and "ACTION6" in available:
                _record(memory, "ACTION6", ("j", src, dst))
                return "ACTION6", click[0], click[1]
        else:
            memory["bans"].add((before, ("j", src, dst)))
            memory["plan"] = []

    if not model["peg"]:
        return _reset(memory, available)
    if not memory.get("plan"):
        memory["plan"] = _search(model, memory, start_time + 8.0)
        memory["plan_key"] = key

    for _ in range(2):
        plan = memory.get("plan", [])
        if not plan:
            break
        action = plan.pop(0)
        if action[0] == "a":
            name = _ACTIONS[action[1]]
            if name in available:
                _record(memory, name, action)
                return name, None, None
        else:
            src, dst = action[1], action[2]
            sc, dc = _click_point(model, src), _click_point(model, dst)
            if (src in model["peg"] and sc is not None and dc is not None
                    and "ACTION6" in available):
                if src in mapped["selected"] and dst in mapped["markers"]:
                    _record(memory, "ACTION6", action)
                    return "ACTION6", dc[0], dc[1]
                memory["pending"] = src, dst, key
                _record(memory, "ACTION6")
                return "ACTION6", sc[0], sc[1]
        memory["bans"].add((key, action))
        memory["plan"] = []
        if time.monotonic() < start_time + 7.0:
            memory["plan"] = _search(model, memory, start_time + 8.0)

    result = _undo(memory, available, reject=True)
    if result is not None:
        return result

    occupied = set(model["peg"]) | model["pillar"]
    landing = model["floor"] | model["cart"]
    local = []
    for src, color in model["peg"].items():
        for dx, dy in _DIRS:
            mid = src[0] + dx, src[1] + dy
            dst = src[0] + 2 * dx, src[1] + 2 * dy
            action = "j", src, dst
            if (mid in occupied and dst in landing and dst not in occupied
                    and (key, action) not in memory["bans"]
                    and _click_point(model, src) is not None
                    and _click_point(model, dst) is not None):
                removal = color != 9 and model["peg"].get(mid) == color
                local.append((not removal, action))
    if local and "ACTION6" in available:
        local.sort(key=lambda v: v[0])
        action = local[0][1]
        src, dst = action[1], action[2]
        click = _click_point(model, src)
        memory["pending"] = src, dst, key
        _record(memory, "ACTION6")
        return "ACTION6", click[0], click[1]

    for d, name in enumerate(_ACTIONS):
        if name not in available:
            continue
        snap = _snapshot(model)
        action = "a", d
        _apply(model, action)
        nk = _key(model)
        _restore(model, snap)
        if (nk != key and memory["visits"].get(nk, 0) == 0
                and (key, action) not in memory["bans"]):
            _record(memory, name, action)
            return name, None, None
    return _reset(memory, available)

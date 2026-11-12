import numpy as np
import random
import time
from collections import deque


def _components(a, colors):
    h, w = a.shape
    colors = set(colors)
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or int(a[y, x]) not in colors:
                continue
            todo = [(x, y)]
            seen.add((x, y))
            comp = []
            while todo:
                xx, yy = todo.pop()
                comp.append((xx, yy))
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and (nx, ny) not in seen
                            and int(a[ny, nx]) in colors):
                        seen.add((nx, ny))
                        todo.append((nx, ny))
            result.append(comp)
    return result


def _rotation(obs):
    edges = (obs[0, :], obs[:, 0], obs[-1, :], obs[:, -1])
    scores = [int(np.count_nonzero((e == 14) | (e == 0))) for e in edges]
    return max(range(4), key=lambda k: scores[k])


def _decode(obs, model=None):
    if model is None:
        k = _rotation(obs)
        a = np.rot90(obs, -k)
        balls = _components(a, (6,))
        if balls:
            scale = min(
                min(max(x for x, y in c) - min(x for x, y in c) + 1,
                    max(y for x, y in c) - min(y for x, y in c) + 1)
                for c in balls
            )
            scale = max(1, min(8, scale))
        else:
            scale = 4

        mask = a != 1
        mask[0, :] = False
        yy, xx = np.where(mask)
        candidates = []
        for n in range(4, 65):
            if 64 // n != scale:
                continue
            off = (64 - n * scale) // 2
            if len(xx):
                if (xx.min() < off or xx.max() >= off + n * scale
                        or yy.min() < off or yy.max() >= off + n * scale):
                    continue
            aligned = True
            for c in balls:
                bx = min(x for x, y in c)
                by = min(y for x, y in c)
                if (bx - off) % scale or (by - off) % scale:
                    aligned = False
                    break
            if aligned:
                candidates.append(n)
        n = min(candidates) if candidates else 64 // scale
        off = (64 - n * scale) // 2
    else:
        k, scale, n, off = (
            model["k"], model["scale"], model["n"], model["off"]
        )
        a = np.rot90(obs, -k)

    indices = off + np.arange(n) * scale + scale // 2
    indices = np.clip(indices, 0, 63)
    grid = a[np.ix_(indices, indices)].astype(np.int8)
    return grid, k, scale, n, off


def _make_model(obs):
    g, k, scale, n, off = _decode(obs)
    w = n + 2
    size = w * w

    def enc(x, y):
        return (y + 1) * w + x + 1

    objects = []
    movable_cells = set()
    for color in (8, 9, 15):
        for comp in _components(g, (color,)):
            x0 = min(x for x, y in comp)
            y0 = min(y for x, y in comp)
            x1 = max(x for x, y in comp)
            y1 = max(y for x, y in comp)
            cells = set(comp)
            # Preserve any special-colored pixels inside a movable's shape.
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if int(g[yy, xx]) == 4:
                        cells.add((xx, yy))
            rel = tuple(sorted((y - y0) * w + x - x0 for x, y in cells))
            typ = 1 if color == 15 else 0
            if color == 9 and x1 > x0 and y1 > y0:
                typ = 1
            obj = {
                "x": x0, "y": y0,
                "width": x1 - x0 + 1, "height": y1 - y0 + 1,
                "rel": rel, "type": typ, "selected": color == 9,
            }
            objects.append(obj)
            movable_cells.update(cells)
    objects.sort(key=lambda o: (o["y"], o["x"], o["type"]))

    cups = _components(g, (11, 13))
    boxes = []
    base = [2] * size
    for y in range(n):
        for x in range(n):
            base[enc(x, y)] = 0

    cup_cells = {}
    for i, comp in enumerate(cups):
        x0 = min(x for x, y in comp)
        x1 = max(x for x, y in comp)
        y0 = min(y for x, y in comp)
        y1 = max(y for x, y in comp)
        boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        for x, y in comp:
            cup_cells[(x, y)] = i
            base[enc(x, y)] = 100 + i

    sources = []
    for y in range(n):
        for x in range(n):
            if (x, y) in movable_cells or (x, y) in cup_cells:
                continue
            color = int(g[y, x])
            p = enc(x, y)
            if color == 6:
                base[p] = 3
                sources.append(p)
            elif color == 1:
                base[p] = 1
            elif color == 4:
                base[p] = 2
            elif color not in (12, 8, 9, 15, 14):
                base[p] = 2

    # A source can generate a new initial droplet when spill begins.
    for y in range(n - 1):
        for x in range(n):
            if int(g[y, x]) == 4 and (x, y) not in movable_cells:
                p = enc(x, y + 1)
                if base[p] == 0:
                    base[p] = 3
                    sources.append(p)
    sources = tuple(dict.fromkeys(sources))

    dirs = (w, 1, -w, -1)
    targets = []
    for i, comp in enumerate(cups):
        cells = {enc(x, y) for x, y in comp}
        found = []
        x0, y0, cw, ch = boxes[i]
        for y in range(max(0, y0 - 1), min(n, y0 + ch + 1)):
            for x in range(max(0, x0 - 1), min(n, x0 + cw + 1)):
                p = enc(x, y)
                if p in cells:
                    continue
                for d in range(4):
                    side = 1 if d % 2 == 0 else w
                    if (p + dirs[d] in cells and p - side in cells
                            and p + side in cells):
                        found.append((x, y, d))
        if not found:
            found = [(x0 + cw // 2, y0, 0)]
        targets.append(found)

    cfg = []
    selected = None
    for i, obj in enumerate(objects):
        p0 = enc(obj["x"], obj["y"])
        cfg.append(p0)
        if obj["selected"]:
            selected = i
        legal = []
        placements = {}
        ow, oh = obj["width"], obj["height"]
        for y in range(3, n - oh + 1):
            for x in range(n - ow + 1):
                okay = True
                for rx, ry, rw, rh in boxes:
                    if (x < rx + rw + 1 and x + ow > rx - 1
                            and y < ry + rh + 1 and y + oh > ry - 1):
                        okay = False
                        break
                if not okay:
                    continue
                p = enc(x, y)
                cells = tuple(p + r for r in obj["rel"])
                if any(base[q] != 0 for q in cells):
                    continue
                legal.append(p)
                placements[p] = cells
        placements[p0] = tuple(p0 + r for r in obj["rel"])
        obj["legal"] = tuple(legal)
        obj["valid"] = set(legal)
        obj["placements"] = placements

    if selected is None and objects:
        selected = min(
            range(len(objects)),
            key=lambda i: objects[i]["x"] ** 2 + objects[i]["y"] ** 2
        )

    # Bitset distance fields give a cheap, smooth measure of how closely
    # a simulated flow approaches each cup in the appropriate direction.
    maxdist = 2 * n + 12
    bits = [1 << j for j in range(size * 4)]
    masks = []
    for choices in targets:
        rings = [0] * (maxdist + 1)
        for p in range(size):
            x, y = p % w - 1, p // w - 1
            for d in range(4):
                dd = min(
                    abs(x - tx) + abs(y - ty) +
                    (0 if d == td else 10 if (d - td) % 4 == 2 else 6)
                    for tx, ty, td in choices
                )
                rings[min(maxdist, dd)] |= bits[p * 4 + d]
        accum = 0
        for j in range(maxdist + 1):
            accum |= rings[j]
            rings[j] = accum
        masks.append(rings)

    flat_targets = [t for choices in targets for t in choices]
    endcost = [0] * (size * 4)
    if flat_targets:
        for p in range(size):
            x, y = p % w - 1, p // w - 1
            for d in range(4):
                endcost[p * 4 + d] = min(
                    abs(x - tx) + abs(y - ty) + (3 if d != td else 0)
                    for tx, ty, td in flat_targets
                )

    seed = int(np.sum(obs.astype(np.int64) *
                      (np.arange(4096).reshape(64, 64) + 1)))
    return {
        "n": n, "w": w, "size": size, "k": k,
        "scale": scale, "off": off, "objects": objects,
        "base": tuple(base), "sources": sources,
        "targets": targets, "masks": masks, "bits": bits,
        "endcost": endcost, "dirs": dirs, "maxdist": maxdist,
        "cfg": tuple(cfg), "selected": selected,
        "cache": {}, "pool": [], "bad": set(), "goal": None,
        "rng": random.Random(seed), "pending": None,
        "origin": tuple(cfg),
    }


def _simulate(m, cfg):
    cache = m["cache"]
    if cfg in cache:
        return cache[cfg]

    board = list(m["base"])
    objects = m["objects"]
    w = m["w"]
    # Stable sprite priority: strips precede elbows. Overlap is allowed.
    order = sorted(range(len(objects)), key=lambda i: (objects[i]["type"], i))
    for i in reversed(order):
        obj = objects[i]
        cells = obj["placements"].get(cfg[i])
        if cells is None:
            cells = tuple(cfg[i] + r for r in obj["rel"])
        for p in cells:
            if 0 <= p < len(board) and board[p] in (0,) or (
                    0 <= p < len(board) and 10 <= board[p] < 100):
                board[p] = 10 + i

    queue = [(p, 0) for p in m["sources"]]
    seen = [0] * m["size"]
    visited = 0
    filled = 0
    leaks = 0
    leakdist = 0
    bits = m["bits"]
    dirs = m["dirs"]
    head = 0

    while head < len(queue):
        p, d = queue[head]
        head += 1
        if p < 0 or p >= len(board):
            continue
        flag = 1 << d
        if seen[p] & flag:
            continue
        seen[p] |= flag
        visited |= bits[p * 4 + d]
        q = p + dirs[d]
        if q < 0 or q >= len(board):
            continue
        v = board[q]
        if v == 0:
            board[q] = 3
            queue.append((q, d))
            continue
        if v == 3:
            queue.append((q, d))
            continue
        if v == 1:
            leaks += 1
            leakdist += m["endcost"][p * 4 + d]
            continue
        if v == 2:
            continue

        side = 1 if d % 2 == 0 else w
        a, b = p - side, p + side
        va = board[a] if 0 <= a < len(board) else 2
        vb = board[b] if 0 <= b < len(board) else 2

        if v >= 100:
            if va == v and vb == v:
                filled |= 1 << (v - 100)
                continue
            if va == 0:
                board[a] = 3
                queue.append((a, d))
            if vb == 0:
                board[b] = 3
                queue.append((b, d))
            continue

        if 10 <= v < 100:
            obj = objects[v - 10]
            if obj["type"] == 0:
                if va == 0:
                    board[a] = 3
                    queue.append((a, d))
                if vb == 0:
                    board[b] = 3
                    queue.append((b, d))
                continue

            # Keep the simulator's two separate conditionals, including
            # its fallback after the first turning case.
            if va == v and vb == 0:
                nd = (d + 1) % 4
                np_ = p + dirs[nd]
                if 0 <= np_ < len(board):
                    if board[np_] == 0:
                        board[np_] = 3
                    queue.append((np_, nd))
            if vb == v and va == 0:
                nd = (d - 1) % 4
                np_ = p + dirs[nd]
                if 0 <= np_ < len(board):
                    if board[np_] == 0:
                        board[np_] = 3
                    queue.append((np_, nd))
            else:
                if 0 <= a < len(board) and board[a] == 0:
                    board[a] = 3
                    queue.append((a, d))
                if 0 <= b < len(board) and board[b] == 0:
                    board[b] = 3
                    queue.append((b, d))

    count = len(m["targets"])
    missing = count - filled.bit_count()
    distance = 0
    for i, rings in enumerate(m["masks"]):
        if filled & (1 << i):
            continue
        lo, hi = 0, m["maxdist"]
        while lo < hi:
            mid = (lo + hi) // 2
            if rings[mid] & visited:
                hi = mid
            else:
                lo = mid + 1
        distance += lo
    result = (missing, distance, leaks, leakdist)
    cache[cfg] = result
    return result


def _distances(m):
    result = []
    w = m["w"]
    for obj, start in zip(m["objects"], m["cfg"]):
        valid = obj["valid"]
        dist = {start: 0}
        q = deque([start])
        while q:
            p = q.popleft()
            nd = dist[p] + 1
            for t in (p - w, p + w, p - 1, p + 1):
                if t in valid and t not in dist:
                    dist[t] = nd
                    q.append(t)
        result.append(dist)
    return result


def _travel(m, cfg, distances):
    total = 0
    for i, p in enumerate(cfg):
        d = distances[i].get(p, 10000)
        total += d
        if d and i != m["selected"]:
            total += 1
    return total


def _won(m, cfg, raw=None):
    if cfg in m["bad"]:
        return False
    if raw is None:
        raw = _simulate(m, cfg)
    return raw[0] == 0 and raw[2] == 0


def _base_score(raw):
    a, b, c, d = raw
    return 32.0 * a + 2.4 * b + 8.0 * c + 0.6 * d


def _candidate_positions(m, cfg, i, reachable, full=False):
    rng = m["rng"]
    obj = m["objects"][i]
    legal = reachable[i]
    if full or len(m["objects"]) <= 2:
        return legal

    n, w = m["n"], m["w"]
    p = cfg[i]
    x, y = p % w - 1, p // w - 1
    positions = {p, m["cfg"][i], m["origin"][i]}
    for xx in range(n - obj["width"] + 1):
        positions.add((y + 1) * w + xx + 1)
    for yy in range(3, n - obj["height"] + 1):
        positions.add((yy + 1) * w + x + 1)

    if legal:
        positions.update(rng.sample(legal, min(24, len(legal))))

    columns = {p % w - 1 for p in m["sources"]}
    rows = {3, y}
    for choices in m["targets"]:
        for tx, ty, td in choices:
            columns.add(tx)
            rows.update((ty - 1, ty, ty + 1))
    for j, other in enumerate(m["objects"]):
        xx, yy = cfg[j] % w - 1, cfg[j] // w - 1
        columns.update((xx - 1, xx, xx + other["width"],
                        xx + other["width"] - 1))
        rows.update((yy - 2, yy - 1, yy + 1, yy + 2))
    goodx = set()
    for xx in columns:
        if obj["type"] == 0:
            goodx.update((xx + 1, xx - obj["width"],
                          xx - obj["width"] // 2))
        else:
            for dx in range(obj["width"]):
                goodx.add(xx - dx)
    goodx = tuple(xx for xx in goodx if 0 <= xx <= n - obj["width"])
    goody = tuple(yy for yy in rows if 3 <= yy <= n - obj["height"])
    if goodx:
        for _ in range(30):
            xx = rng.choice(goodx)
            if goody and rng.random() < 0.65:
                yy = rng.choice(goody)
            else:
                yy = rng.randrange(3, max(4, n - obj["height"] + 1))
            positions.add((yy + 1) * w + xx + 1)
    reachable_set = set(legal)
    return [p for p in positions if p in reachable_set]


def _polish(m, goal, distances, deadline):
    rng = m["rng"]
    best = tuple(goal)
    bestcost = _travel(m, best, distances)
    if not _won(m, best):
        return best
    nobj = len(best)

    for _ in range(4):
        changed = False
        order = sorted(range(nobj),
                       key=lambda i: distances[i].get(best[i], 0),
                       reverse=True)
        for i in order:
            if time.perf_counter() >= deadline:
                return best
            positions = sorted(distances[i], key=lambda p: distances[i][p])
            for p in positions:
                if p == best[i]:
                    continue
                c = list(best)
                c[i] = p
                c = tuple(c)
                cost = _travel(m, c, distances)
                if cost >= bestcost:
                    continue
                if time.perf_counter() >= deadline:
                    return best
                if _won(m, c):
                    best, bestcost = c, cost
                    changed = True
                    break
        if not changed:
            break

    # Occasionally undo one entire object's movement while compensating
    # with a different object.
    pairs = [(i, j) for i in range(nobj) for j in range(nobj) if i != j]
    rng.shuffle(pairs)
    for i, j in pairs:
        if time.perf_counter() >= deadline:
            break
        if best[i] == m["cfg"][i]:
            continue
        c0 = list(best)
        c0[i] = m["cfg"][i]
        for p in sorted(distances[j], key=lambda p: distances[j][p]):
            c0[j] = p
            c = tuple(c0)
            cost = _travel(m, c, distances)
            if cost >= bestcost:
                continue
            if time.perf_counter() >= deadline:
                return best
            if _won(m, c):
                best, bestcost = c, cost
                break
    return best


def _search(m, distances, deadline):
    rng = m["rng"]
    current = m["cfg"]
    nobj = len(current)
    reachable = [tuple(d) for d in distances]
    pool = list(dict.fromkeys([current] + m["pool"]))
    pool = [c for c in pool
            if len(c) == nobj and all(c[i] in distances[i] for i in range(nobj))]

    def value(c):
        raw = _simulate(m, c)
        penalty = 40 if c in m["bad"] else 0
        return _base_score(raw) + 0.08 * _travel(m, c, distances) + penalty

    pool.sort(key=value)
    pool = pool[:24]
    best = pool[0]
    bestvalue = value(best)
    for c in pool:
        if _won(m, c):
            return c, best

    iteration = 0
    while time.perf_counter() < deadline:
        if iteration == 0:
            cfg = best
        else:
            r = rng.random()
            if r < 0.35:
                cfg = best
            elif r < 0.75:
                cfg = rng.choice(pool)
            else:
                cfg = current
            c = list(cfg)
            amount = rng.choice((1, 1, 2, 2, 3))
            for i in rng.sample(range(nobj), min(amount, nobj)):
                if reachable[i]:
                    c[i] = rng.choice(reachable[i])
            cfg = tuple(c)

        weights = (
            rng.uniform(24, 43),
            rng.uniform(1.5, 3.5),
            rng.uniform(5, 14),
            rng.uniform(0.3, 0.9),
        )

        def score(raw, travel, c):
            return (sum(a * b for a, b in zip(raw, weights))
                    + 0.08 * travel + (40 if c in m["bad"] else 0))

        for sweep in range(2):
            changed = False
            order = list(range(nobj))
            rng.shuffle(order)
            for i in order:
                if time.perf_counter() >= deadline:
                    m["pool"] = list(dict.fromkeys([best] + pool))[:24]
                    return None, best
                raw = _simulate(m, cfg)
                local_score = score(raw, _travel(m, cfg, distances), cfg)
                local_cfg = cfg
                full = (rng.random() < 0.16 or
                        (m["objects"][i]["type"] and rng.random() < 0.55))
                candidates = _candidate_positions(
                    m, cfg, i, reachable, full=full
                )
                for counter, p in enumerate(candidates):
                    if counter % 24 == 0 and time.perf_counter() >= deadline:
                        m["pool"] = list(dict.fromkeys([best] + pool))[:24]
                        return None, best
                    if p == cfg[i]:
                        continue
                    c = list(cfg)
                    c[i] = p
                    c = tuple(c)
                    travel = _travel(m, c, distances)
                    raw = _simulate(m, c)
                    if _won(m, c, raw):
                        m["pool"] = list(dict.fromkeys([c, best] + pool))[:24]
                        return c, best
                    bv = _base_score(raw) + 0.08 * travel
                    if c in m["bad"]:
                        bv += 40
                    if bv < bestvalue:
                        best, bestvalue = c, bv
                    sc = score(raw, travel, c)
                    if sc < local_score - 1e-9:
                        local_cfg, local_score = c, sc
                if local_cfg != cfg:
                    cfg = local_cfg
                    changed = True
                if cfg not in pool:
                    pool.append(cfg)
                    pool.sort(key=value)
                    pool = pool[:24]
            if not changed:
                break
        iteration += 1

    m["pool"] = list(dict.fromkeys([best] + pool))[:24]
    return None, best


def _screen_point(m, x, y):
    sx = m["off"] + x * m["scale"] + m["scale"] // 2
    sy = m["off"] + y * m["scale"] + m["scale"] // 2
    for _ in range(m["k"]):
        sx, sy = sy, 63 - sx
    return int(sy), int(sx)


def _click_object(m, i):
    obj = m["objects"][i]
    w = m["w"]
    x, y = m["cfg"][i] % w - 1, m["cfg"][i] // w - 1
    choices = []
    for dy in range(obj["height"]):
        for dx in range(obj["width"]):
            xx, yy = x + dx, y + dy
            cover = 0
            for j, other in enumerate(m["objects"]):
                if i == j:
                    continue
                ox = m["cfg"][j] % w - 1
                oy = m["cfg"][j] // w - 1
                if (ox <= xx < ox + other["width"]
                        and oy <= yy < oy + other["height"]):
                    cover += 1
            transparent = (dy * w + dx) not in obj["rel"]
            choices.append((cover, transparent, yy, xx))
    _, _, yy, xx = min(choices)
    return _screen_point(m, xx, yy)


def _execute_toward(m, target, distances):
    cfg = m["cfg"]
    selected = m["selected"]
    moving = [i for i in range(len(cfg)) if cfg[i] != target[i]]
    if not moving:
        return None

    if selected in moving:
        i = selected
    else:
        # Prefer an object with an unambiguous clickable location.
        def choice_key(j):
            row, col = _click_object(m, j)
            return distances[j].get(target[j], 10000), row, col
        i = min(moving, key=choice_key)

    if i != selected:
        row, col = _click_object(m, i)
        m["pending"] = ("click", cfg, selected)
        m["selected"] = i
        return "ACTION6", row, col

    dist = distances[i]
    p = target[i]
    if p not in dist:
        return None
    w = m["w"]
    while dist[p] > 1:
        p = next(q for q in (p - w, p + w, p - 1, p + 1)
                 if dist.get(q) == dist[p] - 1)
    delta = p - cfg[i]
    if delta == -w:
        dx, dy = 0, -1
    elif delta == w:
        dx, dy = 0, 1
    elif delta == -1:
        dx, dy = -1, 0
    elif delta == 1:
        dx, dy = 1, 0
    else:
        return None

    updated = list(cfg)
    updated[i] = p
    m["pending"] = ("move", cfg, selected)
    m["cfg"] = tuple(updated)
    for _ in range(m["k"]):
        dx, dy = dy, -dx
    action = {(0, -1): "ACTION1", (0, 1): "ACTION2",
              (-1, 0): "ACTION3", (1, 0): "ACTION4"}[(dx, dy)]
    return action, None, None


def act(obs, available, memory):
    started = time.perf_counter()
    obs = np.asarray(obs)
    m = memory.get("model")
    if m is None:
        m = _make_model(obs)
        memory["model"] = m
    else:
        grid, _, _, _, _ = _decode(obs, m)
        yy, xx = np.where(grid == 9)
        pending = m.get("pending")

        if pending and pending[0] == "spill":
            # Reaching another call means the attempt did not finish the
            # episode. Do not repeat that exact arrangement.
            m["bad"].add(m["cfg"])
            m["goal"] = None
            m["pool"] = [c for c in m["pool"] if c not in m["bad"]]

        if len(xx):
            actual_p = (int(yy.min()) + 1) * m["w"] + int(xx.min()) + 1
            actual_w = int(xx.max() - xx.min() + 1)
            actual_h = int(yy.max() - yy.min() + 1)
            sel = m["selected"]
            expected = m["cfg"][sel] if sel is not None else None
            if actual_p != expected:
                matches = [
                    i for i, obj in enumerate(m["objects"])
                    if m["cfg"][i] == actual_p
                    and obj["width"] == actual_w
                    and obj["height"] == actual_h
                ]
                if matches:
                    m["selected"] = matches[0]
                elif pending and pending[0] == "move":
                    old_cfg, old_sel = pending[1], pending[2]
                    if old_sel is not None and old_cfg[old_sel] == actual_p:
                        m["cfg"] = old_cfg
                        m["selected"] = old_sel
                        m["goal"] = None
                    else:
                        m = _make_model(obs)
                        memory["model"] = m
                else:
                    m = _make_model(obs)
                    memory["model"] = m
        m["pending"] = None

    if not m["objects"]:
        return "ACTION5", None, None

    if len(m["cache"]) > 150000:
        m["cache"].clear()

    distances = _distances(m)
    goal = m.get("goal")
    if goal is not None:
        if not all(goal[i] in distances[i] for i in range(len(goal))):
            goal = None
            m["goal"] = None
        elif _won(m, goal):
            goal = _polish(
                m, goal, distances, min(started + 0.75, started + 8.6)
            )
            m["goal"] = goal
        else:
            goal = None
            m["goal"] = None

    partial = m["cfg"]
    if goal is None:
        goal, partial = _search(m, distances, started + 7.15)
        if goal is not None:
            goal = _polish(m, goal, distances, started + 8.65)
            m["goal"] = goal

    if goal is not None and goal == m["cfg"]:
        m["pending"] = ("spill", m["cfg"], m["selected"])
        return "ACTION5", None, None

    target = goal if goal is not None else partial
    action = _execute_toward(m, target, distances)
    if action is not None:
        return action

    # Retain the search population and buy another planning turn without
    # disturbing a promising arrangement.
    selected = m["selected"]
    if selected is not None and "ACTION6" in available:
        row, col = _click_object(m, selected)
        return "ACTION6", row, col
    if "ACTION5" in available:
        m["pending"] = ("spill", m["cfg"], m["selected"])
        return "ACTION5", None, None
    for name in available:
        if name != "RESET":
            return name, None, None
    return "RESET", None, None

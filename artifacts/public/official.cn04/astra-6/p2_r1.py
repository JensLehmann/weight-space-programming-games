import time
from collections import deque
import numpy as np


def _components(points):
    points = set(points)
    result = []
    while points:
        start = points.pop()
        comp = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for q in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if q in points:
                    points.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _snapshot(points, grid):
    if not points:
        return None
    x0 = min(x for x, y in points)
    y0 = min(y for x, y in points)
    pts = tuple(sorted((x - x0, y - y0) for x, y in points))
    pins = tuple(sorted((x - x0, y - y0) for x, y in points
                        if int(grid[y, x]) in (8, 3)))
    return {
        "pts": pts,
        "pins": pins,
        "w": max(x for x, y in pts) + 1,
        "h": max(y for x, y in pts) + 1,
        "pos": [x0, y0],
    }


def _option(snap):
    return {
        "pts": tuple(snap["pts"]),
        "pins": tuple((x, y, k) for k, (x, y) in
                      enumerate(snap["pins"])),
        "w": snap["w"],
        "h": snap["h"],
    }


def _rotate(opt):
    h = opt["h"]
    return {
        "pts": tuple(sorted((h - 1 - y, x) for x, y in opt["pts"])),
        "pins": tuple((h - 1 - y, x, k) for x, y, k in opt["pins"]),
        "w": opt["h"],
        "h": opt["w"],
    }


def _same(opt, snap):
    return (opt["pts"] == snap["pts"] and
            sorted((x, y) for x, y, k in opt["pins"]) ==
            list(snap["pins"]))


def _opt(piece):
    if piece.get("opts"):
        return piece["opts"][piece["cur"]]
    return {
        "pts": piece["pts"], "pins": (),
        "w": piece["w"], "h": piece["h"]
    }


def _world(piece):
    x0, y0 = piece["pos"]
    return {(x0 + x, y0 + y) for x, y in _opt(piece)["pts"]}


def _active_points(grid, m, near=None):
    n = m["n"]
    bg = m["bg"]
    if m["grey"]:
        return {(x, y) for y in range(n) for x in range(n)
                if int(grid[y, x]) not in (bg, 4)}

    allowed = {(x, y) for y in range(n) for x in range(n)
               if int(grid[y, x]) in (0, 8, 3)
               and int(grid[y, x]) != bg}
    comps = _components(allowed)
    if not comps:
        return set()

    def value(c):
        zeros = sum(int(grid[y, x]) == 0 for x, y in c)
        distance = 0
        if near is not None:
            distance = min(abs(x - near[0]) + abs(y - near[1])
                           for x, y in c)
        return zeros * 1000 + len(c) - distance

    return max(comps, key=value)


def _read_active(grid, m, piece):
    points = _active_points(grid, m, piece["pos"])
    if not points:
        points = _world(piece)
    return _snapshot(points, grid)


def _distances(piece):
    count = len(piece["opts"])
    adj = [set() for _ in range(count)]
    for a, b in piece.get("edges", set()):
        adj[a].add(b)
        adj[b].add(a)
    distances = []
    for a in range(count):
        d = [count * 2] * count
        d[a] = 0
        queue = deque([a])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if d[v] > d[u] + 1:
                    d[v] = d[u] + 1
                    queue.append(v)
        distances.append(d)
    piece["dist"] = distances


def _initialize(obs):
    bg = int(np.bincount(np.asarray(obs[1:], dtype=np.int16).ravel(),
                         minlength=16).argmax())

    # The canonical boards have a 20-cell viewport, enlarged threefold.
    scale, offset, n = 3, 2, 20
    m = {
        "bg": bg, "scale": scale, "offset": offset, "n": n,
        "pieces": [], "selected": None, "phase": "explore",
        "pending": None, "exploring": None, "pool": {},
        "tried": set(), "target": None, "target_key": None,
        "did_move": False, "click_fail": {}, "search_round": 0,
        "all_variants": False,
    }
    indices = offset + scale * np.arange(n) + scale // 2
    grid = np.asarray(obs)[np.ix_(indices, indices)]
    foreground = {(x, y) for y in range(n) for x in range(n)
                  if int(grid[y, x]) != bg}
    m["grey"] = (bg != 4 and
                 sum(int(grid[y, x]) == 4 for x, y in foreground) >= 2)

    groups = []
    selected_group = None
    if m["grey"]:
        active = {q for q in foreground if int(grid[q[1], q[0]]) != 4}
        if active:
            groups.append(active)
            selected_group = 0
        groups.extend(sorted(_components(foreground - active),
                             key=lambda c: (min(y for x, y in c),
                                            min(x for x, y in c))))
    else:
        # Separate touching differently coloured pieces where possible.
        for comp in _components(foreground):
            bodies = {int(grid[y, x]) for x, y in comp
                      if int(grid[y, x]) not in (8, 3)}
            if len(bodies) <= 1:
                groups.append(comp)
                continue
            body_parts = []
            for colour in bodies:
                body_parts.extend(_components(
                    {q for q in comp if int(grid[q[1], q[0]]) == colour}))
            tiny_zero = [c for c in body_parts
                         if len(c) == 1 and
                         int(grid[next(iter(c))[1], next(iter(c))[0]]) == 0]
            if len(body_parts) == 2 and tiny_zero:
                groups.append(comp)
                continue
            markers = {q for q in comp if int(grid[q[1], q[0]]) in (8, 3)}
            used = set()
            for body in body_parts:
                part = set(body)
                changed = True
                while changed:
                    changed = False
                    for x, y in markers - part:
                        if any(q in part for q in
                               ((x - 1, y), (x + 1, y),
                                (x, y - 1), (x, y + 1))):
                            part.add((x, y))
                            changed = True
                used |= part
                groups.append(part)
            groups.extend(_components(comp - used))

        if groups:
            counts = [sum(int(grid[y, x]) == 0 for x, y in c)
                      for c in groups]
            if max(counts) > 0:
                selected_group = max(range(len(groups)),
                                     key=lambda i: counts[i])

    for comp in groups:
        snap = _snapshot(comp, grid)
        if snap is None:
            continue
        m["pieces"].append({
            "pts": snap["pts"], "w": snap["w"], "h": snap["h"],
            "pos": snap["pos"], "opts": [], "cur": 0,
            "known": False, "kind": "unknown",
        })
    m["selected"] = selected_group
    return m, grid


def _decode(obs, m):
    a = m["offset"] + m["scale"] * np.arange(m["n"]) + m["scale"] // 2
    return np.asarray(obs)[np.ix_(a, a)]


def _move(m, action):
    i = m["selected"]
    if i is not None:
        p = m["pieces"][i]
        o = _opt(p)
        dx, dy = {1: (0, -1), 2: (0, 1),
                  3: (-1, 0), 4: (1, 0)}[action]
        x, y = p["pos"]
        if (0 <= x + dx <= m["n"] - o["w"] and
                0 <= y + dy <= m["n"] - o["h"]):
            p["pos"] = [x + dx, y + dy]
    return "ACTION" + str(action), None, None


def _jitter(m):
    i = m["selected"]
    if i is None:
        return "ACTION1", None, None
    p = m["pieces"][i]
    o = _opt(p)
    x, y = p["pos"]
    if y > 0:
        return _move(m, 1)
    if y + o["h"] < m["n"]:
        return _move(m, 2)
    if x > 0:
        return _move(m, 3)
    return _move(m, 4)


def _select(m, i, grid):
    pieces = m["pieces"]
    target = _world(pieces[i])
    others = set()
    for j, p in enumerate(pieces):
        if j != i:
            others |= _world(p)
    failed = m["click_fail"].setdefault(i, set())
    candidates = [q for q in target
                  if 0 <= q[0] < m["n"] and 0 <= q[1] < m["n"]
                  and q not in failed]
    if not candidates:
        m["click_fail"][i] = set()
        if m["selected"] is not None and m["selected"] != i:
            return _jitter(m)
        candidates = list(target)
    if not candidates:
        return "ACTION1", None, None

    cx = sum(x for x, y in target) / max(1, len(target))
    cy = sum(y for x, y in target) / max(1, len(target))
    x, y = min(candidates, key=lambda q: (
        q in others,
        int(grid[q[1], q[0]]) == m["bg"],
        abs(q[0] - cx) + abs(q[1] - cy)))
    m["pending"] = {"type": "select", "i": i, "point": (x, y)}
    row = m["offset"] + y * m["scale"] + m["scale"] // 2
    col = m["offset"] + x * m["scale"] + m["scale"] // 2
    return "ACTION6", int(row), int(col)


def _process_pending(grid, m):
    pending = m.get("pending")
    m["pending"] = None
    if not pending:
        return
    kind = pending["type"]
    i = pending["i"]
    p = m["pieces"][i]

    if kind == "select":
        active = _active_points(grid, m, p["pos"])
        if not active:
            m["selected"] = None
            m["click_fail"].setdefault(i, set()).add(pending["point"])
            return
        scores = []
        for j, piece in enumerate(m["pieces"]):
            world = _world(piece)
            intersection = len(world & active)
            score = intersection / max(1, len(world | active))
            scores.append((score, j == i, j))
        chosen = max(scores)[2]
        m["selected"] = chosen
        if chosen != i:
            m["click_fail"].setdefault(i, set()).add(pending["point"])
        return

    snap = _read_active(grid, m, p)
    if snap is None:
        return
    p["pos"] = snap["pos"]

    if kind == "test":
        rotated = _rotate(p["opts"][0])
        if _same(rotated, snap):
            opts = [p["opts"][0]]
            for unused in range(3):
                opts.append(_rotate(opts[-1]))
            p["opts"] = opts
            p["cur"] = 1
            p["kind"] = "normal"
            p["known"] = True
            return
        p["kind"] = "variable"
        found = next((q for q, o in enumerate(p["opts"])
                      if _same(o, snap)), None)
        if found is None:
            found = len(p["opts"])
            p["opts"].append(_option(snap))
        p["edges"] = {(0, found)}
        p["sequence"] = [0, found]
        p["cur"] = found
        if found == 0:
            p["known"] = True
            _distances(p)
        return

    if kind in ("cycle", "var_run"):
        old = p["cur"]
        found = next((q for q, o in enumerate(p["opts"])
                      if _same(o, snap)), None)
        if found is None:
            found = len(p["opts"])
            p["opts"].append(_option(snap))
        edge = (old, found)
        repeated = edge in p.setdefault("edges", set())
        p["edges"].add(edge)
        p["cur"] = found
        if kind == "cycle":
            p.setdefault("sequence", []).append(found)
            if repeated or len(p["sequence"]) >= 32:
                p["known"] = True
                _distances(p)
        else:
            _distances(p)


def _explore(grid, m):
    pieces = m["pieces"]
    unfinished = [i for i, p in enumerate(pieces) if not p["known"]]
    if not unfinished:
        m["phase"] = "plan"
        return None

    i = m.get("exploring")
    if i not in unfinished:
        i = m["selected"] if m["selected"] in unfinished else unfinished[0]
        m["exploring"] = i
    p = pieces[i]
    if m["selected"] != i:
        return _select(m, i, grid)

    if not p["opts"]:
        # Selection changes colours but not geometry.
        snap = _snapshot(_world(p), grid)
        if not snap["pins"]:
            snap = _read_active(grid, m, p)
        p["opts"] = [_option(snap)]
        p["pos"] = snap["pos"]
        p["cur"] = 0
        m["pending"] = {"type": "test", "i": i}
        return "ACTION5", None, None

    m["pending"] = {"type": "cycle", "i": i}
    return "ACTION5", None, None


def _turn_cost(piece, q):
    if piece["kind"] == "normal":
        return (q - piece["cur"]) % 4
    if q == piece["cur"]:
        return 0
    if "dist" in piece and q < len(piece["dist"]):
        return piece["dist"][piece["cur"]][q]
    return abs(q - piece["cur"])


def _score(configuration, m):
    pieces = m["pieces"]
    n = m["n"]
    minx = min(x for q, x, y in configuration)
    miny = min(y for q, x, y in configuration)
    maxx = max(x + pieces[i]["opts"][q]["w"]
               for i, (q, x, y) in enumerate(configuration))
    maxy = max(y + pieces[i]["opts"][q]["h"]
               for i, (q, x, y) in enumerate(configuration))
    lx, hx = -minx, n - maxx
    ly, hy = -miny, n - maxy
    if lx > hx or ly > hy:
        return float("inf"), None

    dxs = sorted(pieces[i]["pos"][0] - x
                 for i, (q, x, y) in enumerate(configuration))
    dys = sorted(pieces[i]["pos"][1] - y
                 for i, (q, x, y) in enumerate(configuration))
    k = len(pieces)
    xlo = max(lx, min(hx, dxs[(k - 1) // 2]))
    xhi = max(lx, min(hx, dxs[k // 2]))
    ylo = max(ly, min(hy, dys[(k - 1) // 2]))
    yhi = max(ly, min(hy, dys[k // 2]))
    xs = {xlo, xhi}
    ys = {ylo, yhi}
    xs.update(d for d in dxs if xlo <= d <= xhi)
    ys.update(d for d in dys if ylo <= d <= yhi)

    turns = sum(_turn_cost(pieces[i], q)
                for i, (q, x, y) in enumerate(configuration))
    penalty = 0
    for i, (q, x, y) in enumerate(configuration):
        p = pieces[i]
        if p["kind"] == "variable":
            maximum = max(len(o["pins"]) for o in p["opts"])
            penalty += 4 * (maximum - len(p["opts"][q]["pins"]))

    best = (float("inf"), None)
    for dx in xs:
        for dy in ys:
            movement = 0
            changed = []
            target = []
            for i, (q, x, y) in enumerate(configuration):
                tx, ty = x + dx, y + dy
                p = pieces[i]
                distance = abs(tx - p["pos"][0]) + abs(ty - p["pos"][1])
                movement += distance
                if distance or q != p["cur"]:
                    changed.append(i)
                target.append((q, tx, ty))
            clicks = len(changed) - int(m["selected"] in changed)
            cost = movement + turns + clicks + penalty
            if cost < best[0]:
                best = (cost, target)
    return best


def _search(m, deadline, all_variants=False):
    pieces = m["pieces"]
    count = len(pieces)
    if not count:
        return
    n = m["n"]
    allowed = []
    for p in pieces:
        choices = list(range(len(p["opts"])))
        if p["kind"] == "variable" and not all_variants:
            maximum = max(len(o["pins"]) for o in p["opts"])
            choices = [q for q in choices
                       if len(p["opts"][q]["pins"]) == maximum]
        choices.sort(key=lambda q: (_turn_cost(p, q),
                                    -len(p["opts"][q]["pins"])))
        allowed.append(choices)

    root = max(range(count), key=lambda i: (
        max(len(pieces[i]["opts"][q]["pins"]) for q in allowed[i]),
        pieces[i]["kind"] == "variable",
        i == m["selected"]))
    root_position = pieces[root]["pos"]
    fullmask = (1 << count) - 1
    sum_cache = {0: {0}}
    max_cache = {0: 0}

    def sums(mask):
        if mask not in sum_cache:
            bit = mask & -mask
            i = bit.bit_length() - 1
            tail = sums(mask ^ bit)
            counts = {len(pieces[i]["opts"][q]["pins"]) for q in allowed[i]}
            sum_cache[mask] = {a + b for a in tail for b in counts}
            max_cache[mask] = max(sum_cache[mask])
        return sum_cache[mask]

    sums(fullmask)
    pool = m["pool"]
    tried = m["tried"]
    seen = set()
    nodes = [0]
    stop = [False]
    root_deadline = [deadline]

    def endpoint(i, q, pin):
        return (i, q if pieces[i]["kind"] == "variable" else 0, pin)

    def record(placed, pairs):
        key = tuple(sorted(pairs))
        if key in tried:
            return
        config = tuple(placed[i] for i in range(count))
        cost, target = _score(config, m)
        if target is None:
            return
        entries = pool.setdefault(key, [])
        if any(c == config for old_cost, c in entries):
            return
        entries.append((cost, config))
        entries.sort(key=lambda e: e[0])
        del entries[6:]
        if len(pool) >= 2500:
            stop[0] = True

    def candidates_at(point, mask, opens, closed, bounds):
        px, py = point
        result = []
        open_count = len(opens)
        minx, miny, maxx, maxy = bounds
        for i in range(count):
            bit = 1 << i
            if not mask & bit:
                continue
            rest = mask ^ bit
            rest_sums = sums(rest)
            rest_max = max_cache[rest]
            p = pieces[i]
            for q in allowed[i]:
                o = p["opts"][q]
                pin_count = len(o["pins"])
                for ax, ay, pin in o["pins"]:
                    tx, ty = px - ax, py - ay
                    bx0, by0 = min(minx, tx), min(miny, ty)
                    bx1 = max(maxx, tx + o["w"])
                    by1 = max(maxy, ty + o["h"])
                    if bx1 - bx0 > n or by1 - by0 > n:
                        continue
                    worldpins = [(tx + x, ty + y, k)
                                 for x, y, k in o["pins"]]
                    if any((x, y) in closed for x, y, k in worldpins):
                        continue
                    closes = sum((x, y) in opens for x, y, k in worldpins)
                    new_count = open_count + pin_count - 2 * closes
                    if new_count > rest_max:
                        continue
                    if not any(s >= new_count and (s - new_count) % 2 == 0
                               for s in rest_sums):
                        continue
                    distance = (
                        abs(tx - (p["pos"][0] - root_position[0])) +
                        abs(ty - (p["pos"][1] - root_position[1])) +
                        _turn_cost(p, q))
                    result.append((
                        -closes, distance, i, q, tx, ty, worldpins,
                        (bx0, by0, bx1, by1)))
        result.sort(key=lambda c: (c[0], c[1]))
        return result

    def dfs(placed, mask, opens, closed, pairs, bounds):
        nodes[0] += 1
        if stop[0]:
            return
        if nodes[0] % 64 == 0 and time.monotonic() >= root_deadline[0]:
            return
        if time.monotonic() >= deadline:
            stop[0] = True
            return

        signature = tuple(sorted(placed.items()))
        if signature in seen:
            return
        seen.add(signature)
        if len(seen) > 220000:
            stop[0] = True
            return

        if not mask:
            if not opens:
                record(placed, pairs)
            return
        possible = sums(mask)
        if not any(s >= len(opens) and (s - len(opens)) % 2 == 0
                   for s in possible):
            return

        if not opens:
            # A solution may contain several independent connected groups.
            i = (mask & -mask).bit_length() - 1
            p = pieces[i]
            rx = p["pos"][0] - root_position[0]
            ry = p["pos"][1] - root_position[1]
            for q in allowed[i]:
                o = p["opts"][q]
                for dx, dy in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                    tx, ty = rx + dx, ry + dy
                    b = (min(bounds[0], tx), min(bounds[1], ty),
                         max(bounds[2], tx + o["w"]),
                         max(bounds[3], ty + o["h"]))
                    if b[2] - b[0] > n or b[3] - b[1] > n:
                        continue
                    new_open = {(tx + x, ty + y): endpoint(i, q, k)
                                for x, y, k in o["pins"]}
                    if any(point in closed for point in new_open):
                        continue
                    placed[i] = (q, tx, ty)
                    dfs(placed, mask ^ (1 << i), new_open, closed,
                        pairs, b)
                    del placed[i]
                    if stop[0] or time.monotonic() >= root_deadline[0]:
                        return
            return

        best_candidates = None
        # Choose the most constrained currently exposed endpoint.
        points = sorted(opens)
        if len(points) > 9:
            points = points[:4] + points[-4:]
        for point in points:
            candidates = candidates_at(point, mask, opens, closed, bounds)
            if not candidates:
                return
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_candidates = candidates

        for candidate in best_candidates:
            _, _, i, q, tx, ty, worldpins, new_bounds = candidate
            new_open = dict(opens)
            new_closed = set(closed)
            new_pairs = list(pairs)
            for x, y, pin in worldpins:
                point = (x, y)
                eid = endpoint(i, q, pin)
                if point in new_open:
                    other = new_open.pop(point)
                    new_closed.add(point)
                    new_pairs.append(tuple(sorted((eid, other))))
                else:
                    new_open[point] = eid
            placed[i] = (q, tx, ty)
            dfs(placed, mask ^ (1 << i), new_open, new_closed,
                new_pairs, new_bounds)
            del placed[i]
            if stop[0] or time.monotonic() >= root_deadline[0]:
                return

    roots = list(allowed[root])
    if roots:
        shift = m["search_round"] % len(roots)
        roots = roots[shift:] + roots[:shift]
    for number, q in enumerate(roots):
        now = time.monotonic()
        if now >= deadline or stop[0]:
            break
        root_deadline[0] = min(
            deadline, now + (deadline - now) / max(1, len(roots) - number))
        o = pieces[root]["opts"][q]
        opens = {(x, y): endpoint(root, q, k) for x, y, k in o["pins"]}
        dfs({root: (q, 0, 0)}, fullmask ^ (1 << root), opens,
            set(), [], (0, 0, o["w"], o["h"]))
    m["search_round"] += 1


def _choose_plan(m):
    best_cost = float("inf")
    best_key = None
    best_target = None
    for key, entries in m["pool"].items():
        if key in m["tried"]:
            continue
        for old_cost, config in entries:
            cost, target = _score(config, m)
            if cost < best_cost:
                best_cost = cost
                best_key = key
                best_target = target
    if best_target is None:
        return False
    m["target"] = best_target
    m["target_key"] = best_key
    m["did_move"] = False
    return True


def _execute(grid, m):
    target = m["target"]
    pieces = m["pieces"]
    changed = [
        i for i, (q, x, y) in enumerate(target)
        if pieces[i]["cur"] != q or pieces[i]["pos"] != [x, y]
    ]
    if not changed:
        if m["did_move"]:
            m["tried"].add(m["target_key"])
            m["target"] = None
            m["target_key"] = None
            return None
        # A click alone does not invoke the game's completion test.
        if m["selected"] is None:
            return _select(m, 0, grid)
        m["did_move"] = True
        return _jitter(m)

    if m["selected"] in changed:
        i = m["selected"]
    else:
        i = min(changed, key=lambda j: (
            _turn_cost(pieces[j], target[j][0]) +
            abs(pieces[j]["pos"][0] - target[j][1]) +
            abs(pieces[j]["pos"][1] - target[j][2])))
        return _select(m, i, grid)

    p = pieces[i]
    q, tx, ty = target[i]
    m["did_move"] = True
    if p["cur"] != q:
        if p["kind"] == "normal":
            p["cur"] = (p["cur"] + 1) % 4
            o = _opt(p)
            p["pos"][0] = min(p["pos"][0], m["n"] - o["w"])
            p["pos"][1] = min(p["pos"][1], m["n"] - o["h"])
        else:
            m["pending"] = {"type": "var_run", "i": i}
        return "ACTION5", None, None

    x, y = p["pos"]
    if x < tx:
        return _move(m, 4)
    if x > tx:
        return _move(m, 3)
    if y < ty:
        return _move(m, 2)
    return _move(m, 1)


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 7.6

    m = memory.get("_cn04")
    if m is None:
        m, grid = _initialize(obs)
        memory["_cn04"] = m
    else:
        grid = _decode(obs, m)

    if not m["pieces"]:
        return "ACTION6", 32, 32

    _process_pending(grid, m)

    if m["phase"] == "explore":
        action = _explore(grid, m)
        if action is not None:
            return action

    for unused in range(3):
        if m["target"] is not None:
            action = _execute(grid, m)
            if action is not None:
                return action

        if _choose_plan(m):
            continue

        _search(m, deadline, m["all_variants"])
        if _choose_plan(m):
            continue

        if not m["all_variants"]:
            m["all_variants"] = True
            if time.monotonic() + 0.15 < deadline:
                _search(m, deadline, True)
                if _choose_plan(m):
                    continue
        break

    # Keeps the state legal if a search needs another invocation.
    if m["selected"] is None:
        return _select(m, 0, grid)
    return _jitter(m)

import numpy as np
from collections import deque
import heapq
import time


_PATTERNS = {
    "T": (1 << 0) | (1 << 1) | (1 << 4),
    "S": (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7),
    "F": (1 << 1) | (1 << 4) | (1 << 7),
}


def _matches(a, p):
    """Template matching; negative template entries are wildcards."""
    h, w = p.shape
    if h > 64 or w > 64:
        return []
    m = np.ones((65 - h, 65 - w), dtype=bool)
    for y, x in zip(*np.where(p >= 0)):
        m &= a[y:y + 65 - h, x:x + 65 - w] == p[y, x]
    yy, xx = np.where(m)
    return [(int(x), int(y)) for y, x in zip(yy, xx)]


def _components(mask):
    seen = np.zeros((64, 64), dtype=bool)
    result = []
    for yy, xx in zip(*np.where(mask)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        q = [(xx, yy)]
        cells = []
        while q:
            x, y = q.pop()
            cells.append((x, y))
            for nx, ny in ((x - 1, y), (x + 1, y),
                           (x, y - 1), (x, y + 1)):
                if (0 <= nx < 64 and 0 <= ny < 64 and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((nx, ny))
        result.append(cells)
    return result


def _rectangle_sums(a, w):
    p = np.pad(a.astype(np.int32), ((1, 0), (1, 0)))
    p = p.cumsum(0).cumsum(1)
    return p[w:, w:] - p[:-w, w:] - p[w:, :-w] + p[:-w, :-w]


def _fits(a, w):
    result = np.zeros((64, 64), dtype=bool)
    result[:65 - w, :65 - w] = _rectangle_sums(a, w) == 0
    return result


def _touches(a, w):
    result = np.zeros((64, 64), dtype=bool)
    result[:65 - w, :65 - w] = _rectangle_sums(a, w) > 0
    return result


def _rune_path(mask, spell, allowed):
    """Avoid accidentally completing a different spell en route."""
    target = _PATTERNS[spell]
    forbidden = {_PATTERNS[s] for s in allowed if s != spell}
    q = deque([mask])
    prev = {mask: None}
    end = None
    while q:
        v = q.popleft()
        for bit in range(9):
            z = v ^ (1 << bit)
            if z == target:
                prev_end = (v, bit)
                end = z
                q.clear()
                break
            if z in forbidden or z in prev:
                continue
            prev[z] = (v, bit)
            q.append(z)
        if end is not None:
            break
    if end is None:
        return []
    path = [prev_end[1]]
    v = prev_end[0]
    while prev[v] is not None:
        v, bit = prev[v]
        path.append(bit)
    return path[::-1]


def _extract(obs, memory):
    a = np.asarray(obs, dtype=np.int16)

    # Spell buttons have a ten-by-ten border.
    border = np.full((10, 10), -1, dtype=np.int16)
    border[0, :] = border[-1, :] = 3
    border[:, 0] = border[:, -1] = 3
    buttons = _matches(a, border)
    ui = np.zeros((64, 64), dtype=bool)
    allowed = set()
    for x, y in buttons:
        ui[y:y + 10, x:x + 10] = True
        interior = a[y + 1:y + 9, x + 1:x + 9]
        for color, spell in ((6, "F"), (11, "T"), (15, "S")):
            if np.count_nonzero(interior == color) >= 2:
                allowed.add(spell)
    ui[47:64, 22:39] = True
    ui[:, 62:64] = True
    if allowed:
        memory["spells"] = sorted(allowed)
    else:
        allowed = set(memory.get("spells", []))

    # Door templates, including rotations and reflections.
    door_base = np.full((6, 5), 10, dtype=np.int16)
    door_base[0, :] = door_base[-1, :] = 9
    door_base[:, 0] = 9
    doors = []
    for k in range(4):
        p = np.rot90(door_base, k)
        for pp in (p, np.fliplr(p)):
            for x, y in _matches(a, pp):
                if not ui[y:y + pp.shape[0], x:x + pp.shape[1]].any():
                    d = (x, y, pp.shape[1], pp.shape[0])
                    if d not in doors:
                        doors.append(d)
    if doors:
        memory["doors"] = doors
    else:
        doors = memory.get("doors", [])

    door_mask = np.zeros((64, 64), dtype=bool)
    for x, y, w, h in doors:
        door_mask[y:y + h, x:x + w] = True

    # Player: the two-color square, with its orientation encoding facing.
    player = None
    candidates = []
    base = np.array([[10, 10], [9, 9]], dtype=np.int16)
    facing = (1, 3, 0, 2)  # down, right, up, left
    for scale in (2, 1):
        p0 = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)
        w = 2 * scale
        for k in range(4):
            p = np.rot90(p0, k)
            for x, y in _matches(a, p):
                if (not ui[y:y + w, x:x + w].any() and
                        not door_mask[y:y + w, x:x + w].any()):
                    candidates.append((x, y, w, facing[k]))
        if candidates:
            break
    if candidates:
        expected = memory.get("pending", {}).get("post")
        if expected is None:
            expected = memory.get("last_player")
        if expected:
            player = min(
                candidates,
                key=lambda p: abs(p[0] - expected[0]) +
                abs(p[1] - expected[1]) + 5 * abs(p[2] - expected[2]))
        else:
            player = candidates[0]
    else:
        player = memory.get("pending", {}).get("post")
        if player is None:
            player = memory.get("last_player")
    if player is not None:
        player = tuple(player)
        memory["last_player"] = player

    # Rune selections, not the black preview hints.
    rune_mask = 0
    for r in range(3):
        for c in range(3):
            color = int(a[50 + 5 * r, 25 + 5 * c])
            if color in (6, 11, 14, 15):
                rune_mask |= 1 << (3 * r + c)

    # Switches are green-bordered squares with a colored center.
    switches = []
    switch_pixels = np.zeros((64, 64), dtype=bool)
    for color in (13, 12):
        p = np.full((4, 4), 6, dtype=np.int16)
        p[1:3, 1:3] = color
        for scale in (1, 2):
            pp = np.repeat(np.repeat(p, scale, axis=0), scale, axis=1)
            w = 4 * scale
            for x, y in _matches(a, pp):
                if ui[y:y + w, x:x + w].any():
                    continue
                item = (x, y, w, w, color)
                if item not in switches:
                    switches.append(item)
                    switch_pixels[y:y + w, x:x + w] = True
    switches.sort(key=lambda z: (z[1], z[0], z[4]))

    # Gates use the same center colors as their corresponding switches.
    gates = []
    for color in (13, 12):
        m = (a == color) & ~ui & ~switch_pixels
        for cells in _components(m):
            xs = np.array([p[0] for p in cells], dtype=np.int16)
            ys = np.array([p[1] for p in cells], dtype=np.int16)
            gates.append((color, xs, ys))

    # Recover teleport targets even when the actor obscures their centers.
    small = set(tuple(p) for p in memory.get("small_portals", []))
    psmall = np.full((2, 2), 15, dtype=np.int16)
    for x, y in _matches(a, psmall):
        if not ui[y:y + 2, x:x + 2].any():
            small.add((x, y))

    corner6f = np.full((6, 6), -1, dtype=np.int16)
    corner6f[0, 0] = corner6f[0, 5] = 15
    corner6f[5, 0] = corner6f[5, 5] = 15
    for x, y in _matches(a, corner6f):
        if not ui[y:y + 6, x:x + 6].any():
            small.add((x + 2, y + 2))

    large = set(tuple(p) for p in memory.get("large_portals", []))
    corners4 = np.full((4, 4), -1, dtype=np.int16)
    corners4[0, 0] = corners4[0, 3] = 11
    corners4[3, 0] = corners4[3, 3] = 11
    corner_places = []
    for x, y in _matches(a, corners4):
        if ui[y:y + 4, x:x + 4].any():
            continue
        corner_places.append((x, y))
        if (x + 1, y + 1) not in small:
            large.add((x, y))

    corner6b = np.full((6, 6), -1, dtype=np.int16)
    corner6b[0, 0] = corner6b[0, 5] = 11
    corner6b[5, 0] = corner6b[5, 5] = 11
    highlighted = []
    for x, y in _matches(a, corner6b):
        if not ui[y:y + 6, x:x + 6].any():
            large.add((x + 1, y + 1))
            highlighted.append((x + 1, y + 1))
    large = {p for p in large if (p[0] + 1, p[1] + 1) not in small}
    large = sorted(large, key=lambda p: (p[1], p[0]))
    small = sorted(small, key=lambda p: (p[1], p[0]))
    memory["large_portals"] = large
    memory["small_portals"] = small

    li = int(memory.get("li", 0)) % max(1, len(large))
    si = int(memory.get("si", 0)) % max(1, len(small))
    for p in highlighted:
        if p in large:
            li = large.index(p)
            memory["li"] = li
            break

    # Wall and ordinary collision maps are separate: fire and growth ignore
    # several objects which still block walking.
    known = np.isin(a, (2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15))
    unknown_solid = ~known & ~ui
    walls = (a == 5) | unknown_solid | ui
    normal = walls | (a == 4) | door_mask

    # Four-corner brackets around a small portal are physical obstacles.
    for x, y in corner_places:
        if (x + 1, y + 1) in small:
            normal[y, x] = normal[y, x + 3] = True
            normal[y + 3, x] = normal[y + 3, x + 3] = True

    # Unrecognized green objects are conservatively physical obstacles.
    normal |= (a == 6) & ~switch_pixels & ~ui
    # Player pixels must not be interpreted as obstacles.
    if player:
        x, y, w, f = player
        # Known floor behind the actor; do not erase genuine door cells.
        normal[y:y + w, x:x + w] &= door_mask[y:y + w, x:x + w]
        walls[y:y + w, x:x + w] = False

    scene = tuple(switches) + tuple(
        (int(g[0]), tuple(map(int, g[1])), tuple(map(int, g[2])))
        for g in gates)

    return {
        "a": a, "player": player, "allowed": allowed,
        "mask": rune_mask, "doors": doors, "goal": door_mask,
        "normal": normal, "walls": walls,
        "switches": switches, "gates": gates,
        "large": large, "small": small, "li": li, "si": si,
        "scene": scene,
    }


def _ray_tables(ray):
    left = np.zeros((64, 64), dtype=np.int16)
    right = left.copy()
    up = left.copy()
    down = left.copy()
    track = np.zeros(64, dtype=np.int16)
    for x in range(64):
        left[:, x] = track
        track = np.where(ray[:, x] != 0, ray[:, x], track)
    track[:] = 0
    for x in range(63, -1, -1):
        right[:, x] = track
        track = np.where(ray[:, x] != 0, ray[:, x], track)
    track[:] = 0
    for y in range(64):
        up[y, :] = track
        track = np.where(ray[y, :] != 0, ray[y, :], track)
    track[:] = 0
    for y in range(63, -1, -1):
        down[y, :] = track
        track = np.where(ray[y, :] != 0, ray[y, :], track)
    return up, down, left, right


def _plan(model, memory, movement_actions):
    player = model["player"]
    if player is None or not model["doors"]:
        return []

    deadline = time.perf_counter() + 7.0
    switches = model["switches"]
    gates = model["gates"]
    allowed = model["allowed"]
    large, small = model["large"], model["small"]
    scene = model["scene"]
    blocked = set(memory.get("blocked", []))
    cache = {}

    goal_touch = {w: _touches(model["goal"], w) for w in (2, 4)}

    def maps(rm):
        if rm in cache:
            return cache[rm]
        normal = model["normal"].copy()
        growth = model["walls"].copy()
        ray = np.where(model["walls"], -1, 0).astype(np.int16)
        opened = {
            switches[i][4] for i in range(len(switches))
            if rm & (1 << i)
        }
        for color, xs, ys in gates:
            if color not in opened:
                normal[ys, xs] = True
                growth[ys, xs] = True
                ray[ys, xs] = -1
        for i, (x, y, w, h, color) in enumerate(switches):
            if not rm & (1 << i):
                normal[y:y + h, x:x + w] = True
                ray[y:y + h, x:x + w] = i + 1
        result = (
            {w: _fits(normal, w) for w in (2, 4)},
            _fits(growth, 4),
            _ray_tables(ray),
        )
        cache[rm] = result
        return result

    def safe(tab, x, y):
        return 0 <= x < 64 and 0 <= y < 64 and bool(tab[y, x])

    def direct_h(x, y):
        best = 1000
        for dx, dy, dw, dh in model["doors"]:
            ex = max(dx - 3 - x, 0, x - (dx + dw - 1))
            ey = max(dy - 3 - y, 0, y - (dy + dh - 1))
            best = min(best, (ex + ey + 3) // 4)
        return best

    rune_paths = {
        s: _rune_path(model["mask"], s, allowed) for s in allowed
    }
    root_cost = {
        s: max(1, len(rune_paths[s])) for s in allowed
    }
    tele_h = 1000
    if "T" in allowed and (large or small):
        tele_h = root_cost["T"] + min(
            direct_h(x, y) for x, y in large + small)

    def heuristic(st):
        return min(direct_h(st[0], st[1]), tele_h)

    x, y, w, f = player
    root = (x, y, w, f, 0, model["li"], model["si"],
            int(model["mask"] != 0))
    dist = {root: 0}
    prev = {}
    serial = 0
    heap = [(heuristic(root), 0, serial, root)]
    best_partial = root
    best_rank = (heuristic(root), 0)
    final_parent = None
    final_action = None
    expanded = 0
    deltas = {1: (0, -1, 0), 2: (0, 1, 1),
              3: (-1, 0, 2), 4: (1, 0, 3)}

    def forbidden(kind, st):
        return (st[4] == 0 and
                (kind, st[0], st[1], st[2], st[3], scene) in blocked)

    while heap:
        _, g, _, st = heapq.heappop(heap)
        if dist.get(st) != g:
            continue
        expanded += 1
        if expanded % 256 == 0:
            if time.perf_counter() > deadline or expanded > 180000:
                break
        if g > 130:
            continue
        x, y, w, f, rm, li, si, flag = st
        rank = (heuristic(st), g)
        if rank < best_rank:
            best_rank, best_partial = rank, st
        fit, grow_fit, rays = maps(rm)
        edges = []

        for action in movement_actions:
            if forbidden(action, st):
                continue
            dx, dy, nf = deltas[action]
            step = w
            nx, ny = x + dx * step, y + dy * step
            if not safe(fit[w], nx, ny) and w == 4:
                nx, ny = x + dx * 2, y + dy * 2
            if not safe(fit[w], nx, ny):
                if safe(goal_touch[w], nx, ny):
                    final_parent, final_action = st, action
                    break
                nx, ny = x, y
            ns = (nx, ny, w, nf, rm, li, si, flag)
            if ns != st:
                edges.append((action, ns, 1))

        if final_parent is not None:
            break

        if "S" in allowed and not forbidden("S", st):
            nx, ny, nw = x, y, 2 if w == 4 else 4
            possible = True
            if w == 2:
                possible = False
                for ox in range(-2, 1):
                    for oy in range(-2, 1):
                        if safe(grow_fit, x + ox, y + oy):
                            nx, ny = x + ox, y + oy
                            possible = True
                            break
                    if possible:
                        break
            if possible:
                cost = root_cost["S"] if flag else 4
                edges.append(("S", (nx, ny, nw, f, rm, li, si, 0), cost))

        if "T" in allowed and not forbidden("T", st):
            portals = large if w == 4 else small
            index = li if w == 4 else si
            if portals:
                index %= len(portals)
                nx, ny = portals[index]
                nli, nsi = li, si
                if w == 4:
                    nli = (index + 1) % len(portals)
                else:
                    nsi = (index + 1) % len(portals)
                cost = root_cost["T"] if flag else 3
                ns = (nx, ny, w, f, rm, nli, nsi, 0)
                if ns != st:
                    edges.append(("T", ns, cost))

        if "F" in allowed and not forbidden("F", st):
            hit = 0
            if f == 0 and 0 <= x < 64 and 0 <= y < 64:
                hit = int(rays[0][y, x])
            elif f == 1 and 0 <= x < 64 and 0 <= y + w - 1 < 64:
                hit = int(rays[1][y + w - 1, x])
            elif f == 2 and 0 <= x < 64 and 0 <= y < 64:
                hit = int(rays[2][y, x])
            elif f == 3 and 0 <= x + w - 1 < 64 and 0 <= y < 64:
                hit = int(rays[3][y, x + w - 1])
            if hit > 0:
                bit = 1 << (hit - 1)
                if not rm & bit:
                    cost = root_cost["F"] if flag else 3
                    edges.append(
                        ("F", (x, y, w, f, rm | bit, li, si, 0), cost))

        for kind, ns, cost in edges:
            ng = g + cost
            if ng < dist.get(ns, 10 ** 9):
                dist[ns] = ng
                prev[ns] = (st, kind)
                serial += 1
                heapq.heappush(
                    heap, (ng + heuristic(ns), ng, serial, ns))

    endpoint = final_parent if final_parent is not None else best_partial
    if endpoint == root and final_parent is None:
        return []

    transitions = []
    v = endpoint
    while v != root:
        p, kind = prev[v]
        transitions.append((p, kind, v))
        v = p
    transitions.reverse()
    route = []
    for p, kind, ns in transitions:
        entry = {
            "kind": kind,
            "post": ns[:4],
            "li": ns[5],
            "si": ns[6],
        }
        if kind == "F":
            changed = p[4] ^ ns[4]
            i = changed.bit_length() - 1
            entry["target"] = switches[i]
        route.append(entry)
    if final_parent is not None:
        route.append({"kind": final_action, "post": None})
    return route


def _fallback(model, memory, movement_actions):
    player = model["player"]
    if player is None:
        return movement_actions[0] if movement_actions else 1
    x, y, w, f = player
    mask = model["normal"].copy()
    for sx, sy, sw, sh, color in model["switches"]:
        mask[sy:sy + sh, sx:sx + sw] = True
    for color, xs, ys in model["gates"]:
        mask[ys, xs] = True
    fit = _fits(mask, w)
    visits = memory.setdefault("visits", {})
    key = (x, y, w)
    visits[key] = visits.get(key, 0) + 1
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    scored = []
    for k in movement_actions:
        dx, dy = moves[k]
        nx, ny = x + dx * w, y + dy * w
        valid = 0 <= nx < 64 and 0 <= ny < 64 and fit[ny, nx]
        if not valid and w == 4:
            nx, ny = x + dx * 2, y + dy * 2
            valid = 0 <= nx < 64 and 0 <= ny < 64 and fit[ny, nx]
        if not valid:
            continue
        d = min(
            (abs(nx - gx) + abs(ny - gy)
             for gx, gy, gw, gh in model["doors"]),
            default=0)
        score = d + 8 * visits.get((nx, ny, w), 0)
        scored.append((score, k))
    if scored:
        return min(scored)[1]
    if "S" in model["allowed"]:
        return "S"
    return movement_actions[memory.get("calls", 0) %
                            len(movement_actions)] if movement_actions else 1


def act(obs, available, memory):
    names = set(available)
    moves = [i for i in (1, 2, 3, 4) if "ACTION" + str(i) in names]
    memory["calls"] = memory.get("calls", 0) + 1
    model = _extract(obs, memory)
    player = model["player"]

    pending = memory.pop("pending", None)
    if pending is not None and player is not None:
        okay = pending["post"] is None or tuple(player) == tuple(pending["post"])
        if pending["kind"] == "F":
            target = tuple(pending.get("target", ()))
            if target in model["switches"]:
                okay = False
        if pending["kind"] == "T":
            portals = model["large"] if player[2] == 4 else model["small"]
            position = tuple(player[:2])
            if position in portals:
                nxt = (portals.index(position) + 1) % len(portals)
                memory["li" if player[2] == 4 else "si"] = nxt
                model["li" if player[2] == 4 else "si"] = nxt
        if not okay:
            memory["route"] = []
            # Require repeated failures before ruling out a modeled edge.
            key = pending.get("edge")
            if key is not None:
                failures = memory.setdefault("failures", {})
                failures[key] = failures.get(key, 0) + 1
                if failures[key] >= 2:
                    blocked = memory.setdefault("blocked", [])
                    if key not in blocked:
                        blocked.append(key)
        else:
            memory["route_start"] = tuple(player)

    route = memory.get("route", [])
    route_start = memory.get("route_start")
    if route and route_start is not None and player is not None:
        if tuple(route_start) != tuple(player):
            route = []
            memory["route"] = []

    if not route:
        route = _plan(model, memory, moves)
        memory["route"] = route
        memory["route_start"] = player

    if route:
        entry = route[0]
    else:
        kind = _fallback(model, memory, moves)
        entry = {"kind": kind, "post": None}

    kind = entry["kind"]
    if isinstance(kind, int):
        if route:
            route.pop(0)
        post = entry.get("post")
        if post is not None and player is not None:
            memory["pending"] = dict(
                entry,
                edge=(kind, *player, model["scene"]),
            )
            memory["route_start"] = tuple(post)
        action = "ACTION" + str(kind)
        if action not in names:
            action = next((n for n in available if n != "RESET"), "RESET")
        return action, None, None

    if "ACTION6" not in names or kind not in model["allowed"]:
        memory["route"] = []
        action = "ACTION" + str(moves[0]) if moves else (
            next((n for n in available if n != "RESET"), "RESET"))
        return action, None, None

    path = _rune_path(model["mask"], kind, model["allowed"])
    if not path:
        memory["route"] = []
        return "ACTION6", 50, 25

    bit = path[0]
    after = model["mask"] ^ (1 << bit)
    if after == _PATTERNS[kind]:
        if route:
            route.pop(0)
        if player is not None:
            post = entry.get("post")
            if post is not None:
                memory["pending"] = dict(
                    entry,
                    edge=(kind, *player, model["scene"]),
                )
                memory["route_start"] = tuple(post)
                if kind == "T":
                    memory["li"] = entry.get("li", model["li"])
                    memory["si"] = entry.get("si", model["si"])
            else:
                memory["route"] = []

    row = 50 + 5 * (bit // 3)
    col = 25 + 5 * (bit % 3)
    return "ACTION6", int(row), int(col)

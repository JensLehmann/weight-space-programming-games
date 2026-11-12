import heapq
import numpy as np


DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
DIR_ACTION = {
    (0, -1): "ACTION1",
    (0, 1): "ACTION2",
    (-1, 0): "ACTION3",
    (1, 0): "ACTION4",
}

BALL_PIXELS = (
    (2, 1), (3, 1),
    (1, 2), (2, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3), (4, 3),
    (2, 4), (3, 4),
)

SELECT_BORDER = (
    (1, 0), (2, 0), (3, 0), (4, 0),
    (0, 1), (1, 1), (4, 1), (5, 1),
    (0, 2), (5, 2), (0, 3), (5, 3),
    (0, 4), (1, 4), (4, 4), (5, 4),
    (1, 5), (2, 5), (3, 5), (4, 5),
)


def _tile_at(img, x, y):
    if x < 0 or y < 1 or x + 6 > 64 or y + 6 > 64:
        return None

    a = img[y:y + 6, x:x + 6]
    selected = sum(
        int(a[yy, xx]) == 3 for xx, yy in SELECT_BORDER
    ) >= 15

    counts = {}
    for color in (14, 8, 9, 2):
        counts[color] = sum(
            int(a[yy, xx]) == color for xx, yy in BALL_PIXELS
        )
    purple = int(np.count_nonzero(a[:4, :] == 15))

    for border, inside, kind in ((0, 1, "floor"), (11, 12, "platform")):
        if (
            int(a[0, 0]) != border
            or int(a[0, 5]) != border
            or int(a[5, 0]) != border
            or int(a[5, 5]) != border
        ):
            continue

        edge = (
            int(np.count_nonzero(a[0, :] == border))
            + int(np.count_nonzero(a[5, :] == border))
            + int(np.count_nonzero(a[1:5, 0] == border))
            + int(np.count_nonzero(a[1:5, 5] == border))
        )
        if edge < 15 and not selected:
            continue

        interior = int(np.count_nonzero(a[1:5, 1:5] == inside))
        if interior < 2 and purple < 5 and max(counts.values()) < 7:
            continue

        obj = None
        if purple >= 6:
            obj = "key"
        else:
            for color in (14, 8, 9):
                if counts[color] >= 7:
                    obj = color
                    break
            if obj is None and counts[2] >= 10:
                obj = "dead"
        return kind, obj

    return None


def _find_lattice(img):
    best = None
    for my in range(6):
        for mx in range(6):
            score = 0
            for y in range(my if my else 6, 59, 6):
                for x in range(mx, 59, 6):
                    t = _tile_at(img, x, y)
                    if t is not None:
                        score += 3
                        if t[1] is not None:
                            score += 1
            if best is None or score > best[0]:
                best = (score, mx, my)
    return best[1], best[2]


def _parse(img, mx, my):
    tiles = {}
    rails = {}
    visible = set()
    dead = False

    for y in range(my if my else 6, 59, 6):
        for x in range(mx, 59, 6):
            p = ((x - mx) // 6, (y - my) // 6)
            visible.add(p)
            t = _tile_at(img, x, y)

            if t is not None:
                tiles[p] = t
                if t[1] == "dead":
                    dead = True
                continue

            a = img[y:y + 6, x:x + 6]
            center = sum(
                int(a[yy, xx]) == 5
                for yy in (2, 3)
                for xx in (2, 3)
            )
            if center >= 3:
                ports = set()
                if int(a[0, 2]) == 5 or int(a[0, 3]) == 5:
                    ports.add((0, -1))
                if int(a[5, 2]) == 5 or int(a[5, 3]) == 5:
                    ports.add((0, 1))
                if int(a[2, 0]) == 5 or int(a[3, 0]) == 5:
                    ports.add((-1, 0))
                if int(a[2, 5]) == 5 or int(a[3, 5]) == 5:
                    ports.add((1, 0))
                if ports:
                    rails[p] = ports

    return tiles, rails, visible, dead


def _screen_shift(old, new, old_mod, new_mod):
    omx, omy = old_mod
    nmx, nmy = new_mod
    dxs = [
        d for d in range(-54, 55)
        if (omx + d - nmx) % 6 == 0
    ]
    dys = [
        d for d in range(-54, 55)
        if (omy + d - nmy) % 6 == 0
    ]

    weights = np.array(
        [1.0, 1.0, 1.5, 1.0, 1.0, 2.2, 1.0, 1.0,
         3.0, 1.5, 0.0, 3.0, 3.0, 1.0, 3.0, 4.0],
        dtype=np.float32,
    )
    best_score = -1e30
    best = (0, 0)

    for dy in dys:
        oy0 = max(1, -dy)
        oy1 = min(64, 64 - dy)
        if oy1 <= oy0:
            continue
        for dx in dxs:
            ox0 = max(0, -dx)
            ox1 = min(64, 64 - dx)
            if ox1 <= ox0:
                continue

            a = old[oy0:oy1, ox0:ox1]
            b = new[oy0 + dy:oy1 + dy, ox0 + dx:ox1 + dx]
            wa = weights[a]
            wb = weights[b]
            score = float(np.sum(
                np.where(a == b, wa, -0.35 * np.maximum(wa, wb))
            ))
            score -= 0.015 * (abs(dx) + abs(dy))
            if score > best_score:
                best_score = score
                best = (dx, dy)

    return best


def _state(m):
    return (
        tuple(sorted((x, y, c) for (x, y), c in m["pegs"].items())),
        tuple(sorted(m["platforms"])),
        tuple(sorted(m["keys"])),
    )


def _set_state(m, state):
    pegs, platforms, keys = state
    m["pegs"] = {(x, y): c for x, y, c in pegs}
    m["platforms"] = set(platforms)
    m["keys"] = set(keys)


def _move_platforms(state, direction, rails):
    peg_tuple, platform_tuple, key_tuple = state
    pegs = {(x, y): c for x, y, c in peg_tuple}
    platforms = set(platform_tuple)
    keys = set(key_tuple)
    dx, dy = direction

    order = sorted(
        platform_tuple,
        key=lambda p: p[0] * dx + p[1] * dy,
        reverse=True,
    )

    changed = False
    for x, y in order:
        q = (x + dx, y + dy)
        if q in platforms or q not in rails:
            continue

        platforms.remove((x, y))
        platforms.add(q)
        if (x, y) in pegs:
            pegs[q] = pegs.pop((x, y))
        if (x, y) in keys:
            keys.remove((x, y))
            keys.add(q)
        changed = True

    if not changed:
        return None

    return (
        tuple(sorted((x, y, c) for (x, y), c in pegs.items())),
        tuple(sorted(platforms)),
        tuple(sorted(keys)),
    )


def _jump(state, src, dst):
    peg_tuple, platform_tuple, key_tuple = state
    pegs = {(x, y): c for x, y, c in peg_tuple}
    if src not in pegs:
        return None

    middle = (
        (src[0] + dst[0]) // 2,
        (src[1] + dst[1]) // 2,
    )
    color = pegs.pop(src)

    if middle in pegs and pegs[middle] == color and color != 9:
        del pegs[middle]

    pegs[dst] = color
    return (
        tuple(sorted((x, y, c) for (x, y), c in pegs.items())),
        platform_tuple,
        key_tuple,
    )


def _level1_hazards(floor):
    if len(floor) != 33:
        return set()

    minx = min(x for x, y in floor)
    miny = min(y for x, y in floor)
    shape = {
        (minx + x, miny + y)
        for y in range(7)
        for x in range(7)
        if y < 3 or x >= 4
    }
    if shape != floor:
        return set()

    return {
        (minx, miny + 1),
        (minx + 2, miny + 1),
        (minx + 5, miny),
    }


def _screen_position(m, p):
    mx, my = m["mod"]
    ox, oy = m["offset"]
    return (
        mx + 6 * (p[0] - ox),
        my + 6 * (p[1] - oy),
    )


def _click_position(m, p):
    x, y = _screen_position(m, p)
    if x >= 64 or y >= 64 or x + 6 <= 0 or y + 6 <= 0:
        return None
    return (
        max(0, min(63, y + 2)),
        max(0, min(63, x + 2)),
    )


def _carried(state):
    pegs, platforms, _ = state
    ps = set(platforms)
    return {
        (x, y) for x, y, color in pegs
        if color != 9 and (x, y) in ps
    }


def _search(m):
    start = _state(m)
    floor = m["floor"]
    rails = m["rails"]
    seen = m["seen"]
    ports = m["ports"]
    bans = m["bans"]
    hazards = _level1_hazards(floor)
    start_carried = _carried(start)

    def nonblue(s):
        return sum(color != 9 for _, _, color in s[0])

    initial_n = nonblue(start)
    if not start[0]:
        return []

    serial = 0
    heap = [(8 * max(0, initial_n - 1), 0, serial, start)]
    cost = {start: 0}
    parent = {}

    best_capture = None
    best_capture_key = None
    best_board = None
    best_board_key = None
    best_travel = None
    best_travel_key = None
    best_landing = None
    best_landing_key = None
    best_single = None
    best_single_key = None
    best_frontier = None
    best_frontier_key = None

    origin = m.get("travel_origin")
    if origin is None and start_carried:
        origin = sorted(start_carried)[0]

    def add_state(old, new, action, new_cost):
        nonlocal serial
        if new_cost > 80 or new_cost >= cost.get(new, 10**9):
            return
        cost[new] = new_cost
        parent[new] = (old, action)
        serial += 1
        n = nonblue(new)
        heapq.heappush(
            heap,
            (new_cost + 8 * max(0, n - 1), new_cost, serial, new),
        )

    expanded = 0
    while heap and expanded < 9000:
        _, g, _, s = heapq.heappop(heap)
        if cost.get(s) != g:
            continue

        expanded += 1
        peg_tuple, platform_tuple, key_tuple = s
        pegs = {(x, y): c for x, y, c in peg_tuple}
        platforms = set(platform_tuple)
        keys = set(key_tuple)
        carried = _carried(s)
        n = nonblue(s)

        if n < initial_n:
            remaining = [
                (x, y) for x, y, color in peg_tuple if color != 9
            ]
            if remaining and (platforms or keys):
                targets = list(platforms | keys)
                proximity = min(
                    abs(x - tx) + abs(y - ty)
                    for x, y in remaining
                    for tx, ty in targets
                )
            else:
                proximity = 0

            key = (n, min(proximity, 8), g)
            if best_capture_key is None or key < best_capture_key:
                best_capture_key = key
                best_capture = s
            if n <= 1:
                break

        if not start_carried and carried:
            key = (g,)
            if best_board_key is None or key < best_board_key:
                best_board_key = key
                best_board = s

        if start_carried and carried and s != start:
            for p in carried:
                if p in start_carried:
                    continue
                visits = m["travel_visits"].get(p, 0)
                dist = 0
                if origin is not None:
                    dist = abs(p[0] - origin[0]) + abs(p[1] - origin[1])
                key = (visits, -min(dist, 12), g)
                if best_travel_key is None or key < best_travel_key:
                    best_travel_key = key
                    best_travel = s

        if start_carried and not carried and s != start:
            for x, y, color in peg_tuple:
                if color == 9 or (x, y) not in floor:
                    continue
                distance = 0
                if origin is not None:
                    distance = abs(x - origin[0]) + abs(y - origin[1])
                if distance >= 4:
                    key = (-min(distance, 12), g)
                    if best_landing_key is None or key < best_landing_key:
                        best_landing_key = key
                        best_landing = s

        if initial_n == 1 and s != start:
            old, action = parent[s]
            if action[0] == "j":
                key = (g,)
                if best_single_key is None or key < best_single_key:
                    best_single_key = key
                    best_single = s

        for p in carried:
            sx, sy = _screen_position(m, p)
            possible = set(ports.get(p, ()))
            for d in DIRS:
                neighbor = (p[0] - d[0], p[1] - d[1])
                if d in ports.get(neighbor, ()):
                    possible.add(d)

            for d in possible:
                q = (p[0] + d[0], p[1] + d[1])
                if q in seen or (p, d) in m["blocked_frontiers"]:
                    continue
                if not (sx < 12 or sx > 46 or sy < 12 or sy > 46):
                    continue
                key = (n, g + 1)
                if best_frontier_key is None or key < best_frontier_key:
                    best_frontier_key = key
                    best_frontier = (s, ("d", d, True))

        occupied = set(pegs) | keys
        destinations = floor | platforms

        for src, color in pegs.items():
            for dx, dy in DIRS:
                mid = (src[0] + dx, src[1] + dy)
                dst = (src[0] + 2 * dx, src[1] + 2 * dy)

                if (
                    mid not in occupied
                    or dst not in destinations
                    or dst in occupied
                ):
                    continue

                if s == start:
                    if (
                        _click_position(m, src) is None
                        or _click_position(m, dst) is None
                    ):
                        continue

                ns = _jump(s, src, dst)
                if ns is None:
                    continue
                if dst in hazards and nonblue(ns) > 1:
                    continue
                if (s, src, dst) in bans:
                    continue

                add_state(s, ns, ("j", src, dst), g + 2)

        for d in DIRS:
            ns = _move_platforms(s, d, rails)
            if ns is not None:
                add_state(s, ns, ("d", d, False), g + 1)

    extra = None

    if best_capture is not None:
        target = best_capture
    elif best_board is not None:
        target = best_board
    elif best_landing is not None:
        target = best_landing
    elif best_frontier is not None:
        target, extra = best_frontier
    elif best_travel is not None:
        target = best_travel
    elif best_single is not None:
        target = best_single
    else:
        return []

    path = []
    while target != start:
        old, action = parent[target]
        path.append(action)
        target = old
    path.reverse()

    if extra is not None:
        path.append(extra)

    return path


def _initialize(m):
    m["floor"] = set()
    m["rails"] = set()
    m["ports"] = {}
    m["seen"] = set()
    m["pegs"] = {}
    m["platforms"] = set()
    m["keys"] = set()
    m["offset"] = (0, 0)
    m["old_frame"] = None
    m["mod"] = None
    m["plan"] = []
    m["pending"] = None
    m["last_jump"] = None
    m["decisions"] = []
    m["blocked_frontiers"] = set()
    m["speculative"] = None
    m["travel_origin"] = None
    m["travel_visits"] = {}
    m.setdefault("bans", set())
    m.setdefault("resets", 0)


def _reset_action(m, available):
    if m["decisions"]:
        s, src, dst = m["decisions"].pop()
        m["bans"].add((s, src, dst))

    m["resets"] += 1
    bans = m["bans"]
    resets = m["resets"]
    m.clear()
    m["bans"] = bans
    m["resets"] = resets
    _initialize(m)

    if "RESET" in available:
        return "RESET", None, None
    if "ACTION7" in available:
        return "ACTION7", None, None
    return "ACTION1", None, None


def act(obs, available, memory):
    img = np.asarray(obs, dtype=np.int8)
    m = memory

    if "floor" not in m:
        _initialize(m)

    # Selection changes the rendering but not the board state.
    if m["pending"] is not None:
        src, dst = m["pending"]
        click = _click_position(m, dst)
        before = _state(m)
        m["pending"] = None

        if click is not None and src in m["pegs"]:
            ns = _jump(before, src, dst)
            if ns is not None:
                if dst in m["platforms"] and src not in m["platforms"]:
                    m["travel_origin"] = dst
                    m["travel_visits"] = {dst: 1}
                elif src in m["platforms"] and dst not in m["platforms"]:
                    m["travel_origin"] = None

                _set_state(m, ns)
                m["last_jump"] = (before, src, dst)
                m["decisions"].append((before, src, dst))
                return "ACTION6", click[0], click[1]

        m["plan"] = []

    mx, my = _find_lattice(img)
    local_tiles, local_rails, local_visible, dead = _parse(img, mx, my)

    if m["old_frame"] is not None:
        dx, dy = _screen_shift(
            m["old_frame"], img, m["mod"], (mx, my)
        )
        old_mx, old_my = m["mod"]
        ox, oy = m["offset"]
        ox -= (old_mx + dx - mx) // 6
        oy -= (old_my + dy - my) // 6
        m["offset"] = (ox, oy)
    else:
        m["offset"] = (0, 0)

    m["mod"] = (mx, my)
    ox, oy = m["offset"]

    if dead:
        if m["last_jump"] is not None:
            before, src, dst = m["last_jump"]
            m["bans"].add((before, src, dst))
            _set_state(m, before)
            m["plan"] = []
            m["pending"] = None
            m["last_jump"] = None
            m["old_frame"] = img.copy()
            if "ACTION7" in available:
                return "ACTION7", None, None
        return _reset_action(m, available)

    visible = {(x + ox, y + oy) for x, y in local_visible}
    actual_pegs = {}
    actual_platforms = set()
    actual_keys = set()

    for (x, y), (kind, obj) in local_tiles.items():
        p = (x + ox, y + oy)
        if kind == "floor":
            m["floor"].add(p)
        else:
            actual_platforms.add(p)
            m["rails"].add(p)

        if isinstance(obj, int):
            actual_pegs[p] = obj
        elif obj == "key":
            actual_keys.add(p)

    for (x, y), ps in local_rails.items():
        p = (x + ox, y + oy)
        m["rails"].add(p)
        m["ports"][p] = set(ps)

    predicted_pegs = {
        p: c for p, c in m["pegs"].items() if p in visible
    }
    predicted_platforms = m["platforms"] & visible
    predicted_keys = m["keys"] & visible

    if m["old_frame"] is not None:
        mismatch = (
            predicted_pegs != actual_pegs
            or predicted_platforms != actual_platforms
            or predicted_keys != actual_keys
        )
        if mismatch:
            last = m["last_jump"]
            if last is not None:
                before, src, dst = last
                if (
                    src in visible
                    and dst in visible
                    and src in actual_pegs
                    and dst not in actual_pegs
                ):
                    m["bans"].add((before, src, dst))
            m["plan"] = []
            m["pending"] = None

    for p in visible:
        m["pegs"].pop(p, None)
    m["pegs"].update(actual_pegs)

    m["platforms"].difference_update(visible)
    m["platforms"].update(actual_platforms)

    m["keys"].difference_update(visible)
    m["keys"].update(actual_keys)

    m["seen"].update(visible)
    m["old_frame"] = img.copy()

    spec = m.get("speculative")
    if spec is not None:
        p, d = spec
        if p in visible and p in actual_platforms:
            m["blocked_frontiers"].add((p, d))
            m["plan"] = []
        m["speculative"] = None

    for attempt in range(2):
        if not m["plan"]:
            m["plan"] = _search(m)

        if not m["plan"]:
            break

        step = m["plan"].pop(0)

        if step[0] == "j":
            _, src, dst = step
            click = _click_position(m, src)
            target_click = _click_position(m, dst)

            if (
                click is None
                or target_click is None
                or src not in m["pegs"]
            ):
                m["plan"] = []
                continue

            m["pending"] = (src, dst)
            return "ACTION6", click[0], click[1]

        _, d, speculative = step
        action = DIR_ACTION[d]
        if action not in available:
            m["plan"] = []
            continue

        state = _state(m)

        if speculative:
            carried = [
                p for p in m["platforms"]
                if p in m["pegs"]
                and m["pegs"][p] != 9
                and (p[0] + d[0], p[1] + d[1]) not in m["seen"]
            ]
            if not carried:
                m["plan"] = []
                continue

            p = carried[0]
            q = (p[0] + d[0], p[1] + d[1])
            m["rails"].add(q)
            ns = _move_platforms(state, d, m["rails"])
            if ns is not None:
                _set_state(m, ns)
            m["speculative"] = (p, d)
        else:
            ns = _move_platforms(state, d, m["rails"])
            if ns is None:
                m["plan"] = []
                continue
            _set_state(m, ns)

        for p in _carried(_state(m)):
            m["travel_visits"][p] = m["travel_visits"].get(p, 0) + 1

        return action, None, None

    return _reset_action(m, available)

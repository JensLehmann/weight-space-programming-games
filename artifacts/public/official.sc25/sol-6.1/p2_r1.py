import numpy as np
from collections import deque
from heapq import heappush, heappop

# Spell-grid bit masks, in row-major order.
_PATTERNS = {"tele": 19, "size": 170, "fire": 146}
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ROT_DIR = (1, 2, 0, 3)


def _matches(a, template):
    h, w = template.shape
    if h > 64 or w > 64:
        return []
    v = np.lib.stride_tricks.sliding_window_view(a, (h, w))
    yy, xx = np.where(np.all(v == template, axis=(-2, -1)))
    return [(int(x), int(y)) for y, x in zip(yy, xx)]


def _corners(a, colour, size, ui):
    n = 65 - size
    z = (
        (a[:n, :n] == colour)
        & (a[:n, size - 1:size - 1 + n] == colour)
        & (a[size - 1:size - 1 + n, :n] == colour)
        & (a[size - 1:size - 1 + n,
             size - 1:size - 1 + n] == colour)
    )
    yy, xx = np.where(z)
    return [(int(x), int(y)) for y, x in zip(yy, xx)
            if not ui[y, x]]


def _buttons(a):
    v = np.lib.stride_tricks.sliding_window_view(a, (10, 10))
    ok = (
        np.all(v[:, :, 0, :] == 3, axis=-1)
        & np.all(v[:, :, -1, :] == 3, axis=-1)
        & np.all(v[:, :, :, 0] == 3, axis=-1)
        & np.all(v[:, :, :, -1] == 3, axis=-1)
    )
    yy, xx = np.where(ok)
    boxes, spells = [], set()
    for y, x in zip(yy, xx):
        x, y = int(x), int(y)
        inner = a[y + 1:y + 9, x + 1:x + 9]
        spell = None
        if np.any(inner == 15):
            spell = "size"
        elif np.any(inner == 11):
            spell = "tele"
        elif np.any(inner == 6):
            spell = "fire"
        if spell is not None:
            boxes.append((x, y))
            spells.add(spell)
    return boxes, spells


def _find_goals(a):
    t = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int8)
    result = []
    for k in range(4):
        q = np.rot90(t, -k)
        h, w = q.shape
        for x, y in _matches(a, q):
            r = (x, y, w, h)
            if r not in result:
                result.append(r)
    return result


def _find_player(a, goal_mask, previous=None):
    b = a.copy()
    b[goal_mask] = -1
    t = np.array([[10, 10], [9, 9]], dtype=np.int8)
    for size in (2, 1):
        for k in range(4):
            q = np.rot90(t, -k)
            q = np.repeat(np.repeat(q, size, axis=0), size, axis=1)
            hits = _matches(b, q)
            if hits:
                if previous is not None:
                    hits.sort(key=lambda p: abs(p[0] - previous[0])
                              + abs(p[1] - previous[1]))
                x, y = hits[0]
                return x, y, size, _ROT_DIR[k]

    # Growth can occasionally place the player partly behind a door.
    if previous is not None:
        x, y, size, direction = previous
        if 0 <= x <= 64 - 2 * size and 0 <= y <= 64 - 2 * size:
            k = _ROT_DIR.index(direction)
            q = np.rot90(t, -k)
            q = np.repeat(np.repeat(q, size, axis=0), size, axis=1)
            visible = ~goal_mask[y:y + 2 * size, x:x + 2 * size]
            patch = a[y:y + 2 * size, x:x + 2 * size]
            if np.any(visible) and np.all(patch[visible] == q[visible]):
                return previous
    return None


def _ui_mask(a):
    mask = 0
    for r in range(3):
        for c in range(3):
            if int(a[50 + 5 * r, 25 + 5 * c]) in (6, 11, 14, 15):
                mask |= 1 << (3 * r + c)
    return mask


def _rect(mask, x, y, w, h, value=True):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + w), min(64, y + h)
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = value


def _scene(a, memory):
    model = memory.get("model")
    last = memory.get("last")
    previous = last["after"][:4] if last else None

    if model is None:
        goals = _find_goals(a)
        gm = np.zeros((64, 64), dtype=bool)
        for x, y, w, h in goals:
            _rect(gm, x, y, w, h)

        boxes, spells = _buttons(a)
        ui = np.zeros((64, 64), dtype=bool)
        _rect(ui, 22, 47, 17, 17)
        _rect(ui, 62, 0, 2, 64)
        for x, y in boxes:
            _rect(ui, x, y, 10, 10)

        model = {
            "goals": goals, "gm": gm, "ui": ui, "spells": spells,
            "big": [], "small": [], "succb": {},
            "moves": {}, "badgrow": set(), "badfire": set(),
            "removed": 0,
        }
        memory["model"] = model

    ui, gm = model["ui"], model["gm"]
    player = _find_player(a, gm, previous)
    if player is None:
        return None, model

    big_highlights = _corners(a, 11, 6, ui)
    small_highlights = _corners(a, 15, 6, ui)
    selected_big = ((big_highlights[0][0] + 1,
                     big_highlights[0][1] + 1)
                    if big_highlights else None)
    selected_small = ((small_highlights[0][0] + 2,
                       small_highlights[0][1] + 2)
                      if small_highlights else None)

    small = _matches(a, np.full((2, 2), 15, dtype=np.int8))
    small = [p for p in small if not ui[p[1], p[0]]]
    if selected_small is not None:
        small.append(selected_small)
    for p in sorted(set(small), key=lambda p: (p[1], p[0])):
        if p not in model["small"]:
            model["small"].append(p)

    big = _corners(a, 11, 4, ui)
    # A four-corner obstacle surrounding a small teleport pad is not
    # itself a large teleport destination.
    big = [p for p in big
           if (p[0] + 1, p[1] + 1) not in model["small"]
           or p == selected_big]
    if selected_big is not None:
        big.append(selected_big)
    for p in sorted(set(big), key=lambda p: (p[1], p[0])):
        if p not in model["big"]:
            model["big"].append(p)

    if "base" not in model:
        switches = []
        switch_area = np.zeros((64, 64), dtype=bool)
        for group, colour in enumerate((13, 12)):
            q = np.full((4, 4), 6, dtype=np.int8)
            q[1:3, 1:3] = colour
            for x, y in _matches(a, q):
                if not ui[y, x]:
                    switches.append((x, y, group))
                    _rect(switch_area, x, y, 4, 4)
        switches.sort(key=lambda p: (p[2], p[1], p[0]))

        decoration = np.zeros((64, 64), dtype=bool)
        for x, y in model["big"]:
            _rect(decoration, x, y, 4, 4)
        for x, y in big_highlights:
            _rect(decoration, x, y, 6, 6)

        walls = (a == 5) & ~ui
        unknown = ~np.isin(a, [2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15])
        unknown &= ~ui & ~gm
        other_six = (a == 6) & ~ui & ~switch_area
        solid_eleven = (a == 11) & ~ui & ~decoration
        base = (unknown | other_six | (a == 4)
                | solid_eleven | ui)
        model.update({
            "walls": walls,
            "base": base,
            "raybase": unknown | other_six,
            "growbase": unknown,
            "blocks": [
                (a == 13) & ~ui & ~switch_area,
                (a == 12) & ~ui & ~switch_area,
            ],
            "switches": switches,
            "groupbits": [
                sum(1 << i for i, s in enumerate(switches) if s[2] == g)
                for g in range(2)
            ],
        })
    else:
        model["walls"] |= (a == 5) & ~ui

    removed = model["removed"]
    for i, (x, y, group) in enumerate(model["switches"]):
        patch = a[y:y + 4, x:x + 4]
        if (not np.any(patch == 6)
                and not np.any((patch == 9) | (patch == 10))):
            removed |= 1 << i
    model["removed"] = removed

    if last:
        memory["bi"] = last["after"][5]
        memory["si"] = last["after"][6]

    bi = int(memory.get("bi", 0))
    si = int(memory.get("si", 0))
    if selected_big in model["big"]:
        bi = model["big"].index(selected_big)
    if "si" not in memory and selected_small in model["small"]:
        si = model["small"].index(selected_small)
    bi %= max(1, len(model["big"]))
    si %= max(1, len(model["small"]))
    memory["bi"], memory["si"] = bi, si

    if last and last.get("spell") == "tele" and last["source"][2] == 2:
        if selected_big is not None:
            landed = (player[0], player[1])
            if landed in model["big"] and selected_big in model["big"]:
                model["succb"][landed] = selected_big

    state = tuple(player) + (removed, bi, si, _ui_mask(a))
    return state, model


def _click_route(start, spell, spells, cache):
    key = (start, spell)
    if key in cache:
        return cache[key]
    target = _PATTERNS[spell]
    forbidden = {_PATTERNS[s] for s in spells if s != spell}
    if start == target:
        bit = next(i for i in range(9) if target & (1 << i))
        result = [bit, bit]
        cache[key] = result
        return result

    parents = {start: None}
    queue = deque([start])
    end = None
    while queue:
        mask = queue.popleft()
        for bit in range(9):
            nxt = mask ^ (1 << bit)
            if nxt in forbidden or nxt in parents:
                continue
            parents[nxt] = (mask, bit)
            if nxt == target:
                end = nxt
                queue.clear()
                break
            queue.append(nxt)
        if end is not None:
            break

    result = []
    if end is not None:
        while parents[end] is not None:
            end, bit = parents[end]
            result.append(bit)
        result.reverse()
    cache[key] = result
    return result


def _square_hits(mask, width):
    p = np.pad(mask.astype(np.int16), ((1, 0), (1, 0)))
    p = p.cumsum(axis=0).cumsum(axis=1)
    sums = (p[width:, width:] - p[:-width, width:]
            - p[width:, :-width] + p[:-width, :-width])
    result = np.ones((64, 64), dtype=bool)
    n = 65 - width
    result[:n, :n] = sums > 0
    return result


def _ray_table(grid):
    result = np.zeros((4, 64, 64), dtype=np.int16)
    for y in range(64):
        hit = 0
        for x in range(64):
            result[2, y, x] = hit
            if grid[y, x]:
                hit = grid[y, x]
        hit = 0
        for x in range(63, -1, -1):
            result[3, y, x] = hit
            if grid[y, x]:
                hit = grid[y, x]
    for x in range(64):
        hit = 0
        for y in range(64):
            result[0, y, x] = hit
            if grid[y, x]:
                hit = grid[y, x]
        hit = 0
        for y in range(63, -1, -1):
            result[1, y, x] = hit
            if grid[y, x]:
                hit = grid[y, x]
    return result


def _plan(start, model):
    spells = model["spells"]
    routes = {}
    configurations = {}
    grow_cache = {}
    heuristic_cache = {}

    goal_hits = {
        1: _square_hits(model["gm"], 2),
        2: _square_hits(model["gm"], 4),
    }
    # Outside the board is not a goal.
    goal_hits[1][63:, :] = False
    goal_hits[1][:, 63:] = False
    goal_hits[2][61:, :] = False
    goal_hits[2][:, 61:] = False

    def configuration(removed):
        if removed in configurations:
            return configurations[removed]
        collision = model["base"] | model["walls"] | model["gm"]
        grow = model["growbase"] | model["walls"]
        ray = model["raybase"] | model["walls"]
        collision, grow, ray = collision.copy(), grow.copy(), ray.copy()

        for group in range(2):
            if not (removed & model["groupbits"][group]):
                block = model["blocks"][group]
                collision |= block
                grow |= block
                ray |= block
        codes = np.zeros((64, 64), dtype=np.int16)
        codes[ray] = -1
        for i, (x, y, group) in enumerate(model["switches"]):
            if not (removed & (1 << i)):
                _rect(collision, x, y, 4, 4)
                codes[y:y + 4, x:x + 4] = i + 1

        value = (
            {1: _square_hits(collision, 2),
             2: _square_hits(collision, 4)},
            _square_hits(grow, 4),
            _ray_table(codes) if "fire" in spells else None,
        )
        configurations[removed] = value
        return value

    def at(mask, x, y):
        return x < 0 or y < 0 or x >= 64 or y >= 64 or bool(mask[y, x])

    def distance(x, y):
        best = 1000
        for gx, gy, gw, gh in model["goals"]:
            dx = max(gx - 3 - x, 0, x - (gx + gw - 1))
            dy = max(gy - 3 - y, 0, y - (gy + gh - 1))
            best = min(best, (dx + 3) // 4 + (dy + 3) // 4)
        return best

    tele_distance = min(
        [distance(x, y) for x, y in model["big"] + model["small"]]
        or [1000]
    )

    def heuristic(state):
        x, y, size, direction, removed, bi, si, ui = state
        key = (x, y, ui)
        if key not in heuristic_cache:
            h = distance(x, y)
            if "tele" in spells:
                route = _click_route(ui, "tele", spells, routes)
                if route:
                    h = min(h, len(route) + tele_distance)
            heuristic_cache[key] = h
        return heuristic_cache[key]

    def reconstruct(state, final=None):
        path = []
        while state != start:
            prev, kind = parents[state]
            path.append((prev, kind, state))
            state = prev
        path.reverse()
        if final is not None:
            path.append(final)
        return path

    best_cost = {start: 0}
    parents = {}
    heap = []
    serial = 0
    heappush(heap, (heuristic(start), 0, serial, start))
    expansions = 0

    while heap and expansions < 85000:
        _, cost, _, state = heappop(heap)
        if best_cost.get(state) != cost:
            continue
        expansions += 1
        x, y, size, direction, removed, bi, si, ui = state
        fits, growth, rays = configuration(removed)
        edges = []

        for d, (dx, dy) in enumerate(_DIRS):
            override = model["moves"].get((x, y, size, d, removed))
            if override is not None:
                nx, ny = override
                nxt = (nx, ny, size, d, removed, bi, si, ui)
                if nxt != state:
                    edges.append((nxt, 1, ("move", d)))
                continue

            step = 2 * size
            tx, ty = x + step * dx, y + step * dy
            blocked = at(fits[size], tx, ty)
            if size == 2 and blocked:
                tx, ty = x + 2 * dx, y + 2 * dy
                blocked = at(fits[size], tx, ty)

            if blocked:
                hit_goal = (
                    0 <= tx < 64 and 0 <= ty < 64
                    and bool(goal_hits[size][ty, tx])
                )
                if hit_goal:
                    after = (x, y, size, d, removed, bi, si, ui)
                    return reconstruct(state, (state, ("move", d), after))
                tx, ty = x, y
            nxt = (tx, ty, size, d, removed, bi, si, ui)
            if nxt != state:
                edges.append((nxt, 1, ("move", d)))

        if "size" in spells:
            route = _click_route(ui, "size", spells, routes)
            if route:
                destination = None
                if size == 2:
                    destination = (x, y, 1)
                elif (x, y, removed) not in model["badgrow"]:
                    key = (x, y, removed)
                    if key not in grow_cache:
                        p = None
                        for ox in (-2, -1, 0):
                            for oy in (-2, -1, 0):
                                if not at(growth, x + ox, y + oy):
                                    p = (x + ox, y + oy, 2)
                                    break
                            if p is not None:
                                break
                        grow_cache[key] = p
                    destination = grow_cache[key]
                if destination is not None:
                    nx, ny, ns = destination
                    nxt = (nx, ny, ns, direction, removed, bi, si, 0)
                    edges.append((nxt, len(route), ("spell", "size")))

        if "tele" in spells:
            route = _click_route(ui, "tele", spells, routes)
            targets = model["big"] if size == 2 else model["small"]
            if route and targets:
                index = (bi if size == 2 else si) % len(targets)
                nx, ny = targets[index]
                nb, ns = bi, si
                if size == 2:
                    successor = model["succb"].get((nx, ny))
                    nb = (targets.index(successor) if successor in targets
                          else (index + 1) % len(targets))
                else:
                    ns = (index + 1) % len(targets)
                nxt = (nx, ny, size, direction, removed, nb, ns, 0)
                if nxt != state:
                    edges.append((nxt, len(route), ("spell", "tele")))

        if "fire" in spells:
            key = (x, y, size, direction, removed)
            if key not in model["badfire"]:
                ax = x + (2 * size - 1 if direction == 3 else 0)
                ay = y + (2 * size - 1 if direction == 1 else 0)
                if 0 <= ax < 64 and 0 <= ay < 64:
                    hit = int(rays[direction, ay, ax])
                    if hit > 0:
                        nr = removed | (1 << (hit - 1))
                        route = _click_route(ui, "fire", spells, routes)
                        if nr != removed and route:
                            nxt = (x, y, size, direction, nr, bi, si, 0)
                            edges.append((nxt, len(route), ("spell", "fire")))

        for nxt, extra, kind in edges:
            nc = cost + extra
            if nc >= best_cost.get(nxt, 10 ** 9):
                continue
            best_cost[nxt] = nc
            parents[nxt] = (state, kind)
            serial += 1
            heappush(heap, (nc + heuristic(nxt), nc, serial, nxt))

    return None


def _expand_plan(path, model):
    queue = []
    cache = {}
    for source, kind, destination in path:
        if kind[0] == "move":
            queue.append(("ACTION" + str(kind[1] + 1),
                          None, None, source, destination, None))
        else:
            spell = kind[1]
            route = _click_route(source[7], spell, model["spells"], cache)
            mask = source[7]
            current = source
            for i, bit in enumerate(route):
                mask ^= 1 << bit
                final = i == len(route) - 1
                after = destination if final else source[:7] + (mask,)
                queue.append((
                    "ACTION6", 50 + 5 * (bit // 3), 25 + 5 * (bit % 3),
                    current, after, spell if final else None
                ))
                current = after
    return queue


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int8)
    allowed = {str(x).upper().split(".")[-1] for x in available}
    state, model = _scene(a, memory)

    def reset_or_wait():
        memory.clear()
        if "RESET" in allowed:
            return "RESET", None, None
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if name in allowed:
                return name, None, None
        return "ACTION6", 50, 25

    if state is None or not model["goals"]:
        return reset_or_wait()

    last = memory.get("last")
    mismatch = False
    if last:
        expected = last["after"]
        mismatch = (
            state[:4] != expected[:4]
            or state[4] != expected[4]
            or state[7] != expected[7]
        )
        # The large teleport highlight shows the next destination.
        marker = last.get("next_big")
        highlights = _corners(a, 11, 6, model["ui"])
        if state[2] == 2 and marker is not None and highlights:
            actual_marker = (highlights[0][0] + 1, highlights[0][1] + 1)
            if marker != actual_marker:
                mismatch = True

        if mismatch:
            source = last["source"]
            name = last["name"]
            # The first action in the introductory level may be consumed
            # by its demonstration, so do not learn a collision from it.
            if memory.get("turn", 0) > 1:
                if name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                    d = int(name[-1]) - 1
                    if state[2] == source[2] and state[4] == source[4]:
                        model["moves"][
                            (source[0], source[1], source[2], d, source[4])
                        ] = (state[0], state[1])
                elif last.get("spell") == "size":
                    if source[2] == 1 and state[2] == 1:
                        model["badgrow"].add(
                            (source[0], source[1], source[4]))
                elif last.get("spell") == "fire":
                    if state[4] == source[4]:
                        model["badfire"].add(
                            (source[0], source[1], source[2],
                             source[3], source[4]))
            memory.pop("queue", None)
            memory["qi"] = 0

    queue = memory.get("queue")
    qi = int(memory.get("qi", 0))
    if queue is None or qi >= len(queue):
        path = _plan(state, model)
        if not path:
            return reset_or_wait()
        queue = _expand_plan(path, model)
        memory["queue"] = queue
        memory["qi"] = 0
        qi = 0

    name, row, col, source, after, spell = queue[qi]
    if name not in allowed:
        return reset_or_wait()

    next_big = None
    if after[2] == 2 and model["big"]:
        next_big = model["big"][after[5] % len(model["big"])]

    memory["last"] = {
        "name": name, "source": source, "after": after,
        "spell": spell, "next_big": next_big,
    }
    memory["qi"] = qi + 1
    memory["turn"] = int(memory.get("turn", 0)) + 1
    return name, row, col

import numpy as np
import heapq
import time
from collections import deque

PATTERNS = {"T": 19, "S": 170, "F": 146}
DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _components(mask):
    todo = set(map(int, np.flatnonzero(mask)))
    result = []
    while todo:
        p = todo.pop()
        stack = [p]
        comp = [p]
        while stack:
            q = stack.pop()
            x, y = q % 64, q // 64
            neighbors = []
            if x:
                neighbors.append(q - 1)
            if x < 63:
                neighbors.append(q + 1)
            if y:
                neighbors.append(q - 64)
            if y < 63:
                neighbors.append(q + 64)
            for n in neighbors:
                if n in todo:
                    todo.remove(n)
                    stack.append(n)
                    comp.append(n)
        result.append(comp)
    return result


def _box(comp):
    xs = [p % 64 for p in comp]
    ys = [p // 64 for p in comp]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _actors(obs):
    # Find exits by their actual sprite pattern, not by connectivity.
    # A player standing immediately beside an exit shares its colors.
    base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int8)
    exit_mask = np.zeros((64, 64), dtype=bool)
    exits = []
    candidates = np.argwhere(obs == 9)
    for scale in (2, 1):
        for rotation in range(4):
            pattern = np.rot90(base, rotation)
            if scale != 1:
                pattern = np.repeat(
                    np.repeat(pattern, scale, axis=0), scale, axis=1)
            h, w = pattern.shape
            for yy, xx in candidates:
                y, x = int(yy), int(xx)
                if y + h > 64 or x + w > 64 or exit_mask[y, x]:
                    continue
                if np.array_equal(obs[y:y + h, x:x + w], pattern):
                    exits.append((x, y, x + w, y + h))
                    exit_mask[y:y + h, x:x + w] = True

    player = None
    mask = ((obs == 9) | (obs == 10)) & ~exit_mask
    for comp in _components(mask):
        x, y, X, Y = _box(comp)
        w, h = X - x, Y - y
        if w == h and w in (2, 4) and len(comp) == w * h:
            a = obs[y:Y, x:X]
            if np.all(a[0] == 10):
                direction = 1
            elif np.all(a[-1] == 10):
                direction = 0
            elif np.all(a[:, 0] == 10):
                direction = 3
            elif np.all(a[:, -1] == 10):
                direction = 2
            else:
                continue
            player = (y * 64 + x, int(w == 4), direction)
            break

    return player, exits


def _ui(obs):
    mask = np.zeros((64, 64), dtype=bool)
    mask[47:64, 22:39] = True
    mask[:, 62:64] = True
    spells = set()
    for y in range(55):
        for x in range(55):
            if obs[y, x] != 3 or obs[y, x + 9] != 3:
                continue
            if not (np.all(obs[y, x:x + 10] == 3)
                    and np.all(obs[y + 9, x:x + 10] == 3)
                    and np.all(obs[y:y + 10, x] == 3)
                    and np.all(obs[y:y + 10, x + 9] == 3)):
                continue
            inside = obs[y + 1:y + 9, x + 1:x + 9]
            found = False
            for color, spell in ((15, "S"), (11, "T"), (6, "F")):
                if np.count_nonzero(inside == color) >= 4:
                    spells.add(spell)
                    found = True
            if found:
                mask[y:y + 10, x:x + 10] = True
    return mask, spells


def _markers(obs, ui, player=None):
    big, small, active, decoration = set(), set(), [], set()
    valid_b = (obs == 11) & ~ui
    valid_f = (obs == 15) & ~ui

    for y in range(63):
        for x in range(63):
            if np.all(valid_f[y:y + 2, x:x + 2]):
                small.add((x, y))

    for color_mask, color in ((valid_b, 11), (valid_f, 15)):
        for y in range(59):
            for x in range(59):
                if (color_mask[y, x] and color_mask[y, x + 5]
                        and color_mask[y + 5, x]
                        and color_mask[y + 5, x + 5]):
                    if color == 11:
                        big.add((x + 1, y + 1))
                        active.append((x + 1, y + 1))
                    else:
                        small.add((x + 2, y + 2))

    for y in range(61):
        for x in range(61):
            if not (valid_b[y, x] and valid_b[y, x + 3]
                    and valid_b[y + 3, x] and valid_b[y + 3, x + 3]):
                continue
            is_small = (x + 1, y + 1) in small
            if player is not None:
                p, z, _ = player
                if not z and p == (y + 1) * 64 + x + 1:
                    is_small = True
                    small.add((x + 1, y + 1))
            if is_small:
                decoration.update((
                    y * 64 + x, y * 64 + x + 3,
                    (y + 3) * 64 + x, (y + 3) * 64 + x + 3))
            else:
                big.add((x, y))

    return (sorted(big, key=lambda p: (p[1], p[0])),
            sorted(small, key=lambda p: (p[1], p[0])),
            active, decoration)


def _rect_sum(a, size):
    integral = np.pad(a.astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    return (integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size])


def _tables(move, grow, goal, dynamic, grow_dynamic):
    out = {k: [] for k in ("mb", "gb", "goal", "need", "gn")}
    for size in (2, 4):
        n = 65 - size
        for key, a in (("mb", move), ("gb", grow), ("goal", goal)):
            if key == "goal":
                table = np.zeros((64, 64), dtype=bool)
            else:
                table = np.ones((64, 64), dtype=bool)
            table[:n, :n] = _rect_sum(a, size) != 0
            out[key].extend(table.ravel().tolist())
        for key, a in (("need", dynamic), ("gn", grow_dynamic)):
            table = np.zeros((64, 64), dtype=np.int64)
            for dy in range(size):
                for dx in range(size):
                    table[:n, :n] |= a[dy:dy + n, dx:dx + n]
            out[key].extend(table.ravel().tolist())
    return out


def _make_world(obs, player, exits):
    ui, spells = _ui(obs)
    big, small, active, decoration = _markers(obs, ui, player)
    targets = []
    occupied = np.zeros((64, 64), dtype=bool)

    for size in (4, 8):
        for yy, xx in np.argwhere((obs == 6) & ~ui):
            y, x = int(yy), int(xx)
            if y + size > 64 or x + size > 64 or occupied[y, x]:
                continue
            a = obs[y:y + size, x:x + size]
            if not (np.all(a[0] == 6) and np.all(a[-1] == 6)
                    and np.all(a[:, 0] == 6) and np.all(a[:, -1] == 6)):
                continue
            center = a[1:-1, 1:-1]
            color = int(center[0, 0])
            if color in (12, 13) and np.all(center == color):
                targets.append((x, y, size, color))
                occupied[y:y + size, x:x + size] = True

    n_targets = len(targets)
    dyn = np.zeros((64, 64), dtype=np.int64)
    gdyn = np.zeros_like(dyn)
    doors = {}
    for i, (x, y, size, color) in enumerate(targets):
        dyn[y:y + size, x:x + size] |= 1 << i
    for color in (12, 13):
        bit = 1 << (n_targets + color - 12)
        locations = (obs == color) & ~occupied & ~ui
        dyn[locations] |= bit
        gdyn[locations] |= bit
        doors[color] = np.flatnonzero(locations).tolist()

    board = obs.copy()
    board[dyn != 0] = 2
    p, z, _ = player
    x, y = p % 64, p // 64
    size = 4 if z else 2
    board[y:y + size, x:x + size] = 2

    goal = np.zeros((64, 64), dtype=bool)
    for x, y, X, Y in exits:
        goal[y:Y, x:X] = True
        board[y:Y, x:X] = 2

    inside_big = np.zeros((64, 64), dtype=bool)
    for x, y in big:
        inside_big[y:y + 4, x:x + 4] = True
    for yy, xx in np.argwhere(((obs == 11) | (obs == 15)) & ~ui):
        y, x = int(yy), int(xx)
        if inside_big[y, x] or obs[y, x] == 15:
            board[y, x] = 2
        else:
            a = obs[max(0, y - 1):min(64, y + 2),
                    max(0, x - 1):min(64, x + 2)]
            board[y, x] = 5 if np.count_nonzero(a == 5) >= 4 else 2
    board[(obs == 14) & ~ui] = 2

    move = board != 2
    ray = (board != 2) & (board != 4)
    grow = ray.copy()
    move[ui] = True
    ray[ui] = True
    grow[ui] = True
    for q in decoration:
        move[q // 64, q % 64] = True

    world = {
        "ui": ui, "spells": spells, "big": big, "small": small,
        "targets": targets, "doors": doors, "n": n_targets,
        "ray": ray.ravel().tolist(), "dyn": dyn.ravel().tolist(),
        "rcache": {}, "moves": {}, "grows": {}, "badfire": set(),
        "bnext": {}, "exits": exits,
    }
    world.update(_tables(move, grow, goal, dyn, gdyn))
    world["h"] = _heuristic(world)
    return world, active


def _coordinate(p, dx, dy, step):
    x, y = p % 64 + dx * step, p // 64 + dy * step
    return y * 64 + x if 0 <= x < 64 and 0 <= y < 64 else -1


def _move(w, p, z, removed, a):
    key = (p, z, removed, a)
    if key in w["moves"]:
        return w["moves"][key], False
    dx, dy = DXY[a]
    last_goal = False
    for step in ((4, 2) if z else (2,)):
        q = _coordinate(p, dx, dy, step)
        if q < 0:
            last_goal = False
            continue
        k = z * 4096 + q
        last_goal = w["goal"][k]
        blocked = w["mb"][k] or last_goal or (w["need"][k] & ~removed)
        if not blocked:
            return q, False
    return p, bool(last_goal)


def _resize(w, p, z, removed):
    if z:
        return p, 0
    key = (p, removed)
    if key in w["grows"]:
        return w["grows"][key]
    x, y = p % 64, p // 64
    for dx in range(-2, 1):
        for dy in range(-2, 1):
            X, Y = x + dx, y + dy
            if not (0 <= X < 64 and 0 <= Y < 64):
                continue
            q = Y * 64 + X
            k = 4096 + q
            if not w["gb"][k] and not (w["gn"][k] & ~removed):
                return q, 1
    return p, 0


def _fire(w, p, z, direction, removed):
    key = (p, z, direction, removed)
    if key in w["badfire"]:
        return None
    ray_key = (p, z, direction)
    if ray_key not in w["rcache"]:
        x, y = p % 64, p // 64
        dx, dy = DXY[direction]
        size = 4 if z else 2
        if dx > 0:
            x += size - 1
        if dy > 0:
            y += size - 1
        bits = []
        for _ in range(64):
            x, y = x + dx, y + dy
            if not (0 <= x < 64 and 0 <= y < 64):
                break
            q = y * 64 + x
            if w["ray"][q]:
                break
            b = w["dyn"][q]
            if b and (not bits or b != bits[-1]):
                bits.append(b)
        w["rcache"][ray_key] = bits
    for b in w["rcache"][ray_key]:
        if b & ~removed:
            i = b.bit_length() - 1
            if i >= w["n"]:
                return None
            color = w["targets"][i][3]
            return removed | b | (1 << (w["n"] + color - 12))
    return None


def _heuristic(w):
    # Reverse distances in a relaxed graph with removed doors and targets.
    count = 8194
    rev = [[] for _ in range(count)]
    dist = [100000] * count
    heap = []

    for z in (0, 1):
        for p in range(4096):
            k = z * 4096 + p
            for dx, dy in DXY:
                for step in ((4, 2) if z else (2,)):
                    q = _coordinate(p, dx, dy, step)
                    if q < 0:
                        continue
                    j = z * 4096 + q
                    if w["goal"][j]:
                        if dist[k] > 1:
                            dist[k] = 1
                            heapq.heappush(heap, (1, k))
                    elif not w["mb"][j]:
                        rev[j].append((k, 1))
            if "S" in w["spells"]:
                if z:
                    rev[p].append((k, 5))
                else:
                    x, y = p % 64, p // 64
                    for dx in range(-2, 1):
                        for dy in range(-2, 1):
                            X, Y = x + dx, y + dy
                            if 0 <= X < 64 and 0 <= Y < 64:
                                j = 4096 + Y * 64 + X
                                if not w["gb"][j]:
                                    rev[j].append((k, 5))

    if "T" in w["spells"]:
        for z, targets in ((0, w["small"]), (1, w["big"])):
            if targets:
                hub = 8192 + z
                for p in range(4096):
                    rev[hub].append((z * 4096 + p, 4))
                for x, y in targets:
                    rev[z * 4096 + y * 64 + x].append((hub, 0))

    while heap:
        d, k = heapq.heappop(heap)
        if d != dist[k]:
            continue
        for j, cost in rev[k]:
            nd = d + cost
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(heap, (nd, j))
    return [d if d < 100000 else 0 for d in dist[:8192]]


def _slot_mask(obs):
    bits = 0
    for i in range(9):
        color = int(obs[50 + 5 * (i // 3), 25 + 5 * (i % 3)])
        if color in (14, 6, 11, 15):
            bits |= 1 << i
    return bits


def _spell_path(start, spell, allowed):
    target = PATTERNS[spell]
    if start == target:
        return []
    forbidden = {PATTERNS[s] for s in allowed if s != spell}
    queue = deque([start])
    parent = {start: None}
    while queue:
        state = queue.popleft()
        order = sorted(
            range(9),
            key=lambda i: (
                not bool((state & ~target) & (1 << i)),
                not bool(target & (1 << i)), i))
        for i in order:
            nxt = state ^ (1 << i)
            if nxt in parent or nxt in forbidden:
                continue
            parent[nxt] = (state, i)
            if nxt == target:
                result = []
                while nxt != start:
                    nxt, bit = parent[nxt]
                    result.append(bit)
                return result[::-1]
            queue.append(nxt)
    return None


def _search(w, start, bits, deadline):
    paths = {s: _spell_path(bits, s, w["spells"]) for s in w["spells"]}
    initial_costs = {
        s: len(path) + 1 for s, path in paths.items()
        if path is not None and path}
    default_cost = {"S": 5, "T": 4, "F": 4}
    htab = w["h"]

    def heuristic(state):
        p, z, _, _, _, _, dirty = state
        return max(0, htab[z * 4096 + p] - 8 * dirty)

    heap = [(heuristic(start), 0, 0, start)]
    costs = {start: 0}
    first = {}
    serial = 0
    expansions = 0
    best_fallback = None
    best_rank = (100000, 100000)

    while heap:
        _, g, _, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        expansions += 1
        if expansions % 256 == 0 and time.monotonic() > deadline:
            break

        p, z, direction, removed, bi, si, dirty = state
        rank = (heuristic(state), g)
        if state != start and rank < best_rank:
            best_rank = rank
            best_fallback = first[state]

        edges = []
        for a in range(4):
            q, win = _move(w, p, z, removed, a)
            ns = (q, z, a, removed, bi, si, dirty)
            if win:
                return (a, ns) if state == start else first[state]
            if ns != state:
                edges.append((a, ns, 1))

        for spell in sorted(w["spells"]):
            cost = initial_costs.get(spell) if dirty else default_cost[spell]
            if cost is None:
                continue
            if spell == "S":
                q, nz = _resize(w, p, z, removed)
                if (q, nz) == (p, z):
                    continue
                ns = (q, nz, direction, removed, bi, si, 0)
            elif spell == "T":
                targets = w["big"] if z else w["small"]
                if not targets:
                    continue
                idx = (bi if z else si) % len(targets)
                x, y = targets[idx]
                nbi, nsi = bi, si
                if z:
                    nbi = w["bnext"].get(idx, (idx + 1) % len(targets))
                else:
                    nsi = (idx + 1) % len(targets)
                ns = (y * 64 + x, z, direction, removed, nbi, nsi, 0)
                if ns == state:
                    continue
            else:
                nr = _fire(w, p, z, direction, removed)
                if nr is None:
                    continue
                ns = (p, z, direction, nr, bi, si, 0)
            edges.append((spell, ns, cost))

        for command, ns, cost in edges:
            ng = g + cost
            if ng >= costs.get(ns, 10**9):
                continue
            costs[ns] = ng
            first[ns] = (command, ns) if state == start else first[state]
            serial += 1
            heapq.heappush(heap, (ng + heuristic(ns), ng, serial, ns))

        if expansions >= 130000:
            break

    return best_fallback


def _removed(w, obs, previous):
    removed = previous
    for i, (x, y, size, _) in enumerate(w["targets"]):
        a = obs[y:y + size, x:x + size]
        border_count = (
            np.count_nonzero(a[0] == 6)
            + np.count_nonzero(a[-1] == 6)
            + np.count_nonzero(a[:, 0] == 6)
            + np.count_nonzero(a[:, -1] == 6))
        if border_count == 0:
            removed |= 1 << i
    flat = obs.ravel()
    for color, locations in w["doors"].items():
        if locations and not any(int(flat[p]) == color for p in locations):
            removed |= 1 << (w["n"] + color - 12)
    return removed


def _learn(memory, w, attempt, player, removed, active, obs):
    command, before, expected = attempt
    p, z, direction = player
    bp, bz, bd, br, bi, si, _ = before
    ep, ez = expected[:2]

    if isinstance(command, int):
        if (p, z) != (ep, ez) and direction == command:
            # The introductory demonstration consumes the first action.
            if not (memory.get("calls", 0) <= 2
                    and np.count_nonzero(obs[:, 62] == 0) == 0):
                w["moves"][(bp, bz, br, command)] = p
    elif command == "S":
        if (p, z) != (ep, ez):
            w["grows"][(bp, br)] = (p, z)
    elif command == "F":
        if expected[3] & ~removed:
            w["badfire"].add((bp, bz, bd, br))
    elif command == "T":
        pos = (p % 64, p // 64)
        targets = w["big"] if bz else w["small"]
        if pos in targets:
            used = targets.index(pos)
            if bz:
                next_idx = (used + 1) % len(targets)
                if active and active[0] in targets:
                    next_idx = targets.index(active[0])
                w["bnext"][used] = next_idx
                memory["bi"] = next_idx
            else:
                slot = si % len(targets)
                if slot != used:
                    targets[slot], targets[used] = targets[used], targets[slot]
                memory["si"] = (slot + 1) % len(targets)


def act(obs, available, memory):
    obs = np.asarray(obs)
    started = time.monotonic()
    memory["calls"] = memory.get("calls", 0) + 1

    player, exits = _actors(obs)
    if player is None:
        # Avoid throwing away progress on an unexpected animation frame.
        old = memory.get("last_player")
        if old is not None:
            p, z, d = old
            known_exits = memory.get("world", {}).get("exits", [])
            if known_exits:
                x, y = p % 64, p // 64
                ex = min(
                    known_exits,
                    key=lambda b: abs(x - (b[0] + b[2]) / 2)
                    + abs(y - (b[1] + b[3]) / 2))
                dx = (ex[0] + ex[2]) / 2 - x
                dy = (ex[1] + ex[3]) / 2 - y
                a = (3 if dx > 0 else 2) if abs(dx) > abs(dy) \
                    else (1 if dy > 0 else 0)
                name = "ACTION" + str(a + 1)
                if name in available:
                    return name, None, None
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        return "ACTION1", None, None

    memory["last_player"] = player

    if "world" not in memory:
        w, active = _make_world(obs, player, exits)
        memory["world"] = w
        memory["removed"] = 0
        memory["bi"] = 0
        memory["si"] = 0
        memory["visits"] = {}
    else:
        w = memory["world"]
        _, _, active, _ = _markers(obs, w["ui"], player)

    bits = _slot_mask(obs)
    removed = _removed(w, obs, memory.get("removed", 0))
    memory["removed"] = removed

    attempt = memory.pop("attempt", None)
    pending = memory.get("pending")
    if attempt is not None:
        if isinstance(attempt[0], int):
            _learn(memory, w, attempt, player, removed, active, obs)
        elif bits == 0:
            _learn(memory, w, attempt, player, removed, active, obs)
            memory.pop("pending", None)
            pending = None

    if active and active[0] in w["big"]:
        memory["bi"] = w["big"].index(active[0])

    p, z, direction = player
    root = (
        p, z, direction, removed,
        memory.get("bi", 0), memory.get("si", 0), int(bits != 0))

    def issue_spell(pending_record):
        spell, before, expected = pending_record
        path = _spell_path(bits, spell, w["spells"])
        if not path:
            memory.pop("pending", None)
            return None
        i = path[0]
        memory["pending"] = pending_record
        if len(path) == 1:
            memory["attempt"] = pending_record
        return "ACTION6", 50 + 5 * (i // 3), 25 + 5 * (i % 3)

    if pending is not None:
        result = issue_spell(pending)
        if result is not None:
            return result

    visit_key = (p, z, direction, removed)
    visits = memory["visits"]
    visits[visit_key] = visits.get(visit_key, 0) + 1

    choice = _search(w, root, bits, started + 7.5)
    if choice is not None:
        command, expected = choice
        if isinstance(command, int):
            name = "ACTION" + str(command + 1)
            if name in available:
                memory["attempt"] = (command, root, expected)
                return name, None, None
        elif "ACTION6" in available:
            result = issue_spell((command, root, expected))
            if result is not None:
                return result

    options = []
    for a in range(4):
        if "ACTION" + str(a + 1) not in available:
            continue
        q, win = _move(w, p, z, removed, a)
        v = visits.get((q, z, a, removed), 0)
        score = v * 20 + (10 if q == p else 0)
        if win:
            score = -1000
        options.append((score, a, q))
    if options:
        _, a, q = min(options)
        expected = (q, z, a, removed, root[4], root[5], root[6])
        memory["attempt"] = (a, root, expected)
        return "ACTION" + str(a + 1), None, None

    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    return "ACTION6", 50, 30

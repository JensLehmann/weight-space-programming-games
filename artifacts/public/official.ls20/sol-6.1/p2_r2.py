import numpy as np
import collections
import heapq
import itertools
import time

# Shape order used by the simulator.
_SHAPES = [
    [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
    [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
]
_COLORS = (12, 9, 14, 8)
_MASKS = [
    [np.rot90(np.array(s, dtype=bool), -r) for r in range(4)]
    for s in _SHAPES
]
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_DELTAS = (-12, 12, -1, 1)
_DIR_DELTAS = (12, 1, -12, -1)
_INF = 10**7


def _xy(p):
    return 4 + 5 * (p % 12), 5 * (p // 12)


def _cell(x, y):
    if (x - 4) % 5 or y % 5:
        return None
    c, r = (x - 4) // 5, y // 5
    return r * 12 + c if 0 <= c < 12 and 0 <= r < 12 else None


def _neighbor(p, a):
    r, c = divmod(p, 12)
    if a == 0:
        return p - 12 if r else None
    if a == 1:
        return p + 12 if r < 11 else None
    if a == 2:
        return p - 1 if c else None
    return p + 1 if c < 11 else None


def _distance(a, b):
    return abs(a // 12 - b // 12) + abs(a % 12 - b % 12)


def _decode_symbol(a):
    for ci, color in enumerate(_COLORS):
        mask = a == color
        if np.any(mask) and np.all((a == color) | (a == 5)):
            for sh in range(6):
                for rot in range(4):
                    if np.array_equal(mask, _MASKS[sh][rot]):
                        return sh * 16 + ci * 4 + rot
    return None


def _read(obs, previous):
    player = None
    for p in range(144):
        x, y = _xy(p)
        a = obs[y:y + 5, x:x + 5]
        if a.shape == (5, 5) and np.all(a[:2] == 12) and np.all(a[2:] == 9):
            player = p
            break
    if player is None:
        player = previous.get("p", 66)

    cfg = _decode_symbol(obs[55:61:2, 3:9:2])
    if cfg is None:
        cfg = previous.get("c", 0)

    steps = int(np.count_nonzero(obs[61, 13:55] == 11))
    fog = int(np.count_nonzero(obs[:55, 4:] == 5)) > 300
    px, py = _xy(player)

    def visible(p, margin=0):
        x, y = _xy(p)
        if not fog:
            return True
        return (x + 2 - px - 1.5)**2 + (y + 2 - py - 1.5)**2 <= (20 - margin)**2

    goals, pads, batteries, dark, terrain = [], [], [], [], {}
    for p in range(144):
        x, y = _xy(p)
        a = obs[y:y + 5, x:x + 5]
        if a.shape != (5, 5):
            continue

        if visible(p):
            center = int(a[2, 2])
            if center != 5 or visible(p, 1):
                terrain[p] = int(center == 4)
        if p == 132:  # The persistent symbol display covers this wall.
            terrain[p] = 1

        if not visible(p, 2):
            continue
        inner = a[1:4, 1:4]
        symbol = _decode_symbol(inner)
        if symbol is not None and np.count_nonzero(a == 5) >= 12:
            goals.append((p, symbol))
            dark.append(p)
            terrain[p] = 0
        elif np.count_nonzero(a == 5) >= 15 and p != 132:
            dark.append(p)
            terrain[p] = 0

        if np.array_equal(inner, np.array([
                [11, 11, 11], [11, 3, 11], [11, 11, 11]])):
            batteries.append(p)

        if np.array_equal(inner, np.array([
                [0, 3, 3], [3, 0, 0], [3, 0, 3]])):
            pads.append((0, p))
        elif np.array_equal(inner, np.array([
                [9, 14, 14], [9, 0, 8], [12, 12, 8]])):
            pads.append((1, p))
        elif np.array_equal(inner, np.array([
                [3, 0, 3], [1, 0, 0], [3, 1, 3]])):
            pads.append((2, p))

    launchers = {}
    # Full five-pixel lines are launchers, not the small rotation glyphs.
    for y in range(0, 60, 5):
        for x in range(3, 64):
            if np.all(obs[y:y + 5, x] == 1):
                if x % 5 == 4:
                    p = _cell(x - 5, y)
                    direction = 3
                elif x % 5 == 3:
                    p = _cell(x + 1, y)
                    direction = 1
                else:
                    continue
                if p is not None:
                    launchers[p] = direction
    for x in range(4, 60, 5):
        for y in range(60):
            if np.all(obs[y, x:x + 5] == 1):
                if y % 5 == 0:
                    p = _cell(x, y - 5)
                    direction = 2
                elif y % 5 == 4:
                    p = _cell(x, y + 1)
                    direction = 0
                else:
                    continue
                if p is not None:
                    launchers[p] = direction

    terrain[player] = 0
    for _, p in pads:
        terrain[p] = 0
    for p in batteries:
        terrain[p] = 0
    return player, cfg, steps, fog, visible, terrain, goals, pads, batteries, dark, launchers


def _update_tracks(tracks, pads, player, cfg, last, advanced):
    remaining = list(pads)
    expected = last.get("pads", []) if last else []
    old_cfg = last.get("before_c", cfg) if last else cfg
    changed = [
        cfg // 16 != old_cfg // 16,
        (cfg // 4) % 4 != (old_cfg // 4) % 4,
        cfg % 4 != old_cfg % 4,
    ]

    for i, tr in enumerate(tracks):
        old = tr["p"]
        prediction = expected[i][1] if i < len(expected) else old
        candidates = [(j, p) for j, (kind, p) in enumerate(remaining)
                      if kind == tr["kind"]]
        known = False
        if candidates:
            j, new = min(candidates,
                         key=lambda z: (_distance(z[1], prediction),
                                        _distance(z[1], old)))
            remaining.pop(j)
            known = True
        elif player == prediction or (
                changed[tr["kind"]] and _distance(old, player) <= 1):
            new = player
            known = True
        else:
            new = prediction if advanced else old

        if advanced and new != old:
            tr["moving"] = True
            delta = new - old
            if delta in _DIR_DELTAS and _distance(new, old) == 1:
                tr["dir"] = _DIR_DELTAS.index(delta)
            if known:
                # A gap in visibility can hide several positions of a straight rail.
                if old // 12 == new // 12:
                    lo, hi = sorted((old, new))
                    tr["seen"].update(range(lo, hi + 1))
                elif old % 12 == new % 12:
                    lo, hi = sorted((old, new))
                    tr["seen"].update(range(lo, hi + 1, 12))
        if known:
            tr["seen"].add(new)
        tr["p"] = new

    for kind, p in remaining:
        tracks.append({"kind": kind, "p": p, "dir": 0,
                       "moving": False, "seen": {p}})


def _station_phases(tracks):
    state = tuple((t["p"], t["dir"]) for t in tracks)
    phases, seen = [], {}
    next_phase = []
    while state not in seen and len(phases) < 16:
        seen[state] = len(phases)
        phases.append(state)
        new = []
        for t, (p, direction) in zip(tracks, state):
            if not t["moving"]:
                new.append((p, direction))
                continue
            chosen = (p, direction)
            for d in (direction, (direction - 1) % 4,
                      (direction + 1) % 4, (direction + 2) % 4):
                q = p + _DIR_DELTAS[d]
                if q in t["seen"] and _distance(p, q) == 1:
                    chosen = (q, d)
                    break
            new.append(chosen)
        state = tuple(new)

    if not phases:
        phases = [()]
    loop = seen.get(state, len(phases) - 1)
    next_phase = list(range(1, len(phases))) + [loop]
    return phases, next_phase


def _transform(cfg, kind):
    if kind == 0:
        return (cfg + 16) % 96
    if kind == 1:
        return (cfg // 16) * 16 + (((cfg // 4) + 1) % 4) * 4 + cfg % 4
    return (cfg // 4) * 4 + (cfg + 1) % 4


def _build(world, goals, batteries, deadline):
    terrain = world["terrain"]
    positions = []
    for p in range(144):
        blocked = terrain[p] == 1
        if terrain[p] == -1:
            r, c = divmod(p, 12)
            blocked = r in (0, 11) or c in (0, 11)
        if not blocked:
            positions.append(p)
    index = {p: i for i, p in enumerate(positions)}
    n = len(positions)
    if not n:
        return None

    phases, next_phase = _station_phases(world["tracks"])
    stop = {p for p, v in enumerate(terrain) if v == 1}
    stop.update(world["dark"])
    stop.update(g["p"] for g in world["goals"])
    launch = {}
    for p, d in world["launch"].items():
        q, count = p, 0
        for k in range(1, 12):
            q += _DIR_DELTAS[d]
            if not 0 <= q < 144 or _distance(p, q) != k:
                break
            if q in stop:
                count = k - 1
                break
        end = p + count * _DIR_DELTAS[d]
        if count and end in index:
            launch[p] = end

    goal_at = {g["p"]: i for i, g in enumerate(goals)}
    battery_at = {p: 1 << i for i, p in enumerate(batteries)}
    transforms = [[_transform(c, k) for c in range(96)] for k in range(3)]

    size = n * len(phases) * 96
    transitions = [-1] * (size * 4)
    reverse = [[] for _ in range(size)]
    # Geometry: destination, goal gate, collected goal bit, battery bits.
    geometry = [None] * (n * len(phases) * 4)

    for ph in range(len(phases)):
        nph = next_phase[ph]
        pad_at = {}
        for tr, (p, _) in zip(world["tracks"], phases[nph]):
            pad_at.setdefault(p, []).append(tr["kind"])

        for pi, p in enumerate(positions):
            base = (ph * n + pi) * 96
            for a in range(4):
                dest = _neighbor(p, a)
                if dest not in index:
                    continue
                end = launch.get(dest, dest)
                gate = goal_at.get(dest, -1)
                collected = 0 if dest in launch else (
                    1 << gate if gate >= 0 else 0)
                bbits = battery_at.get(dest, 0) | battery_at.get(end, 0)
                kinds = list(pad_at.get(dest, ()))
                if dest in launch:
                    kinds.extend(pad_at.get(end, ()))
                geometry[(ph * n + pi) * 4 + a] = (
                    dest, gate, collected, bbits)

                target_base = (nph * n + index[end]) * 96
                for cfg in range(96):
                    nc = cfg
                    for kind in kinds:
                        nc = transforms[kind][nc]
                    s, t = base + cfg, target_base + nc
                    transitions[s * 4 + a] = t
                    reverse[t].append(s)
            if pi % 32 == 0 and time.perf_counter() > deadline:
                return None

    return {
        "positions": positions, "index": index, "n": n,
        "phases": phases, "next": next_phase,
        "T": transitions, "rev": reverse, "geo": geometry,
        "tracks": world["tracks"], "goals": goals, "batteries": batteries,
    }


def _make_plan(nodes, finish, graph):
    path = []
    n = graph["n"]
    while nodes[finish][5] >= 0:
        s, gm, bm, rem, depth, parent, action = nodes[finish]
        ps, pgm, pbm, _, _, _, _ = nodes[parent]
        phpi = s // 96
        ph, pi = divmod(phpi, n)
        geo = graph["geo"][(ps // 96) * 4 + action]
        path.append({
            "a": action,
            "p": graph["positions"][pi],
            "c": s % 96,
            "rem": rem,
            "dest": geo[0],
            "pads": [(tr["kind"], state[0])
                     for tr, state in zip(graph["tracks"], graph["phases"][ph])],
            "done": [g["p"] for i, g in enumerate(graph["goals"])
                     if (gm & ~pgm) & (1 << i)],
            "take": [p for i, p in enumerate(graph["batteries"])
                     if (bm & ~pbm) & (1 << i)],
        })
        finish = parent
    path.reverse()
    return path


def _search(graph, player, cfg, steps, unit, full, deadline):
    goals = graph["goals"]
    if not goals or player not in graph["index"]:
        return None
    n, phases = graph["n"], graph["phases"]
    size = len(graph["rev"])
    distances = []
    for goal in goals:
        if goal["p"] not in graph["index"]:
            return None
        dist = [_INF] * size
        queue = collections.deque()
        pi = graph["index"][goal["p"]]
        for ph in range(len(phases)):
            s = (ph * n + pi) * 96 + goal["c"]
            dist[s] = 0
            queue.append(s)
        count = 0
        while queue:
            t = queue.popleft()
            nd = dist[t] + 1
            for s in graph["rev"][t]:
                if dist[s] == _INF:
                    dist[s] = nd
                    queue.append(s)
            count += 1
            if count % 8192 == 0 and time.perf_counter() > deadline:
                return None
        distances.append(dist)

    ng = len(goals)
    all_goals = (1 << ng) - 1
    between = [[0] * ng for _ in range(ng)]
    for j, goal in enumerate(goals):
        pi = graph["index"][goal["p"]]
        for k in range(ng):
            between[j][k] = min(
                distances[k][(ph * n + pi) * 96 + goal["c"]]
                for ph in range(len(phases)))

    tour_cache = {}

    def tour(j, mask):
        key = j, mask
        if key in tour_cache:
            return tour_cache[key]
        if not mask:
            return 0
        best = _INF
        bits = mask
        while bits:
            b = bits & -bits
            k = b.bit_length() - 1
            best = min(best, between[j][k] + tour(k, mask ^ b))
            bits ^= b
        tour_cache[key] = best
        return best

    def heuristic(s, gm):
        mask = all_goals ^ gm
        if not mask:
            return 0
        best = _INF
        bits = mask
        while bits:
            b = bits & -bits
            j = b.bit_length() - 1
            best = min(best, distances[j][s] + tour(j, mask ^ b))
            bits ^= b
        return best

    initial = graph["index"][player] * 96 + cfg
    h = heuristic(initial, 0)
    if h >= _INF:
        return None

    nodes = [(initial, 0, 0, steps, 0, -1, -1)]
    heap = [(h, h, -steps, 0)]
    battery_states = 1 << len(graph["batteries"])
    goal_states = 1 << ng

    def key(s, gm, bm):
        return (s * goal_states + gm) * battery_states + bm

    best = {key(initial, 0, 0): [(0, steps)]}
    expansions = 0
    while heap and expansions < 110000:
        _, _, _, ni = heapq.heappop(heap)
        s, gm, bm, rem, depth, _, _ = nodes[ni]
        if gm == all_goals:
            return _make_plan(nodes, ni, graph)
        expansions += 1
        if expansions % 2048 == 0 and time.perf_counter() > deadline:
            break

        for a in range(4):
            t = graph["T"][s * 4 + a]
            if t < 0:
                continue
            geo = graph["geo"][(s // 96) * 4 + a]
            _, gate, gbit, bbits = geo
            if gate >= 0 and not (gm & (1 << gate)):
                if s % 96 != goals[gate]["c"]:
                    continue
            ngm = gm | gbit
            nbm = bm | bbits
            nr = full if bbits & ~bm else rem - unit
            if nr < 0 and ngm != all_goals:
                continue

            nd = depth + 1
            k = key(t, ngm, nbm)
            old = best.get(k, ())
            if any(g <= nd and r >= nr for g, r in old):
                continue
            hh = heuristic(t, ngm)
            if hh >= _INF:
                continue
            best[k] = [(g, r) for g, r in old
                       if not (nd <= g and nr >= r)] + [(nd, nr)]
            nodes.append((t, ngm, nbm, nr, nd, ni, a))
            idx = len(nodes) - 1
            heapq.heappush(heap, (nd + hh, hh, -nr, idx))
    return None


def _navigate(graph, world, player, cfg, steps, unit, full, deadline):
    """Seek an energy pickup, otherwise reveal unseen terrain."""
    if player not in graph["index"]:
        return None
    initial = graph["index"][player] * 96 + cfg
    nodes = [(initial, 0, 0, steps, 0, -1, -1)]
    queue = collections.deque([0])
    best = {(initial, 0, 0): steps}
    chosen, chosen_score = None, -_INF
    first_depth = None
    battery_mode = bool(graph["batteries"])
    goals = graph["goals"]

    while queue and len(nodes) < 18000:
        ni = queue.popleft()
        s, gm, bm, rem, depth, _, _ = nodes[ni]
        p = graph["positions"][(s // 96) % graph["n"]]
        if depth:
            if battery_mode and bm:
                return _make_plan(nodes, ni, graph)
            if not battery_mode and world["terrain"][p] == -1:
                if first_depth is None:
                    first_depth = depth
                reward = sum(world["terrain"][q] == -1
                             for q in range(144) if _distance(p, q) <= 3)
                score = reward * 2 - depth
                if score > chosen_score:
                    chosen, chosen_score = ni, score
        if first_depth is not None and depth > first_depth + 2:
            break
        if depth > 30 or rem < unit:
            continue
        if len(nodes) % 1024 == 0 and time.perf_counter() > deadline:
            break

        for a in range(4):
            t = graph["T"][s * 4 + a]
            if t < 0:
                continue
            _, gate, gbit, bbits = graph["geo"][(s // 96) * 4 + a]
            if gate >= 0 and not (gm & (1 << gate)) and s % 96 != goals[gate]["c"]:
                continue
            ngm, nbm = gm | gbit, bm | bbits
            nr = full if bbits & ~bm else rem - unit
            if nr < 0:
                continue
            k = t, ngm, nbm
            if best.get(k, -1) >= nr:
                continue
            best[k] = nr
            nodes.append((t, ngm, nbm, nr, depth + 1, ni, a))
            queue.append(len(nodes) - 1)
    return _make_plan(nodes, chosen, graph) if chosen is not None else None


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.4
    if "ls20_player" not in memory:
        memory["ls20_player"] = {
            "terrain": [-1] * 144,
            "goals": [], "battery": {}, "dark": set(),
            "launch": {}, "tracks": [], "plan": [],
            "unit": 1, "full": 42, "last": {},
        }
    w = memory["ls20_player"]
    last = w.get("last", {})

    (player, cfg, steps, fog, visible, terrain, observed_goals,
     pads, batteries, dark, launchers) = _read(obs, last)

    was_reset = last.get("reset", False)
    if was_reset:
        for g in w["goals"]:
            g["done"] = False
        for p in w["battery"]:
            w["battery"][p] = False
        w["tracks"] = []
        w["plan"] = []

    good = bool(last) and not was_reset
    good = good and last.get("p") == player and last.get("c") == cfg
    delta = last.get("before_steps", steps) - steps
    if not was_reset and delta in (1, 2) and not last.get("take"):
        if w["unit"] != delta:
            w["unit"] = delta
            good = False
    if good and steps < last.get("rem", steps):
        good = False

    # Check the station forecast against every station currently visible.
    expected = last.get("pads", [])
    if expected and not was_reset:
        counts = collections.Counter(expected)
        for item in pads:
            if not counts[item]:
                good = False
            else:
                counts[item] -= 1
        for (kind, p), number in counts.items():
            if number and p != player and visible(p, 2):
                good = False
    elif pads and last and not was_reset:
        good = False

    old_player = last.get("before_p", player)
    pad_moved = any(
        kind == t["kind"] and p != t["p"] and _distance(p, t["p"]) == 1
        for kind, p in pads for t in w["tracks"])
    advanced = not was_reset and (
        player != old_player or pad_moved or
        (good and last.get("dest") != old_player))
    _update_tracks(w["tracks"], pads, player, cfg,
                   {} if was_reset else last, advanced)

    for p, v in terrain.items():
        w["terrain"][p] = v
    w["dark"].update(dark)

    new_launcher = any(w["launch"].get(p) != d for p, d in launchers.items())
    w["launch"].update(launchers)
    if new_launcher:
        good = False

    goal_by_pos = {g["p"]: g for g in w["goals"]}
    observed_goal_positions = {p for p, _ in observed_goals}
    for p, c in observed_goals:
        if p not in goal_by_pos:
            g = {"p": p, "c": c, "done": False}
            w["goals"].append(g)
            goal_by_pos[p] = g
            good = False
        elif goal_by_pos[p]["c"] != c:
            goal_by_pos[p]["c"] = c
            good = False
        w["terrain"][p] = 0

    if good:
        for p in last.get("done", ()):
            if p in goal_by_pos:
                goal_by_pos[p]["done"] = True
        for p in last.get("take", ()):
            w["battery"][p] = True

    for g in w["goals"]:
        if (visible(g["p"], 2) and g["p"] != player and
                g["p"] not in observed_goal_positions):
            g["done"] = True
        if g["p"] == player and cfg == g["c"] and not was_reset:
            g["done"] = True

    for p in batteries:
        w["battery"].setdefault(p, False)
    for p in list(w["battery"]):
        if p == player:
            w["battery"][p] = True
        elif visible(p, 2) and p not in batteries:
            w["battery"][p] = True

    if good:
        for entry in w["plan"]:
            if (w["terrain"][entry["p"]] == 1 or
                    w["terrain"][entry["dest"]] == 1):
                good = False
                break
    if not good:
        w["plan"] = []

    def emit(entry):
        w["last"] = dict(entry)
        w["last"].update({
            "before_p": player, "before_c": cfg,
            "before_steps": steps,
        })
        name = _ACTIONS[entry["a"]]
        if name not in available:
            name = next((a for a in _ACTIONS if a in available), available[0])
            w["plan"] = []
            w["last"] = {}
        return name, None, None

    if w["plan"]:
        return emit(w["plan"].pop(0))

    goals = [g for g in w["goals"] if not g["done"]]
    remaining_batteries = [p for p, taken in w["battery"].items() if not taken]
    graph = _build(w, goals, remaining_batteries, deadline - 0.3)
    plan = None
    if graph is not None:
        plan = _search(graph, player, cfg, steps, w["unit"],
                       w["full"], deadline - 0.4)
        if not plan:
            plan = _navigate(graph, w, player, cfg, steps, w["unit"],
                             w["full"], deadline - 0.1)

    if plan:
        w["plan"] = plan
        return emit(w["plan"].pop(0))

    # Reset before the final energy failure, retaining the learned map.
    if "RESET" in available and steps <= 2 * w["unit"]:
        w["last"] = {"reset": True, "p": player, "c": cfg}
        w["plan"] = []
        return "RESET", None, None

    # A short legal move also provides information about moving rails.
    choices = []
    for a, name in enumerate(_ACTIONS):
        if name not in available:
            continue
        q = _neighbor(player, a)
        if q is None or w["terrain"][q] == 1:
            continue
        goal = goal_by_pos.get(q)
        if goal and not goal["done"] and goal["c"] != cfg:
            continue
        unseen = sum(w["terrain"][p] == -1
                     for p in range(144) if _distance(q, p) <= 3)
        score = unseen * 3
        if q == last.get("before_p"):
            score -= 2
        if w["terrain"][q] == -1:
            score += 3
        choices.append((score, a, q))

    if choices:
        _, a, q = max(choices)
        w["last"] = {
            "p": q, "c": cfg, "rem": max(0, steps - w["unit"]),
            "before_p": player, "before_c": cfg,
            "before_steps": steps, "dest": q,
        }
        return _ACTIONS[a], None, None

    if "RESET" in available:
        w["last"] = {"reset": True, "p": player, "c": cfg}
        return "RESET", None, None

    name = next((a for a in _ACTIONS if a in available), available[0])
    w["last"] = {}
    return name, None, None

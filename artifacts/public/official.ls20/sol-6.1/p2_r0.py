import numpy as np
import heapq
import time
from array import array
from collections import deque

_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], bool),
    np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1]], bool),
]
_MASKS = [[np.rot90(a, -r) for r in range(4)] for a in _SHAPES]
_COLORS = [12, 9, 14, 8]
_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
_ADXY = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def _decode_shape(a):
    if a.shape != (3, 3):
        return None
    for c, color in enumerate(_COLORS):
        mask = a == color
        if np.any(mask) and np.all((a == color) | (a == 5)):
            for s in range(6):
                for r in range(4):
                    if np.array_equal(mask, _MASKS[s][r]):
                        return s, c, r
    return None


def _find_player(obs):
    candidates = np.argwhere((obs[:-4, :-4] == 12) &
                            (obs[2:-2, :-4] == 9))
    for y, x in candidates:
        a = obs[y:y + 5, x:x + 5]
        if np.all(a[:2] == 12) and np.all(a[2:] == 9):
            return int(x), int(y)
    return None


def _xy(p):
    return p % 12, p // 12


def _neighbor(p, dx, dy):
    x, y = _xy(p)
    x += dx
    y += dy
    return y * 12 + x if 0 <= x < 12 and 0 <= y < 12 else -1


def _route_step(p, direction, path):
    for d in (direction, (direction - 1) % 4,
              (direction + 1) % 4, (direction + 2) % 4):
        dx, dy = _DIRS[d]
        q = _neighbor(p, dx, dy)
        if q in path:
            return q, d
    return p, direction


def _infer_route(entity, board, tick):
    hist = entity.get("hist", [])
    entity["moving"] = entity.get("moving", False) or (
        len({p for _, p in hist}) > 1
    )
    if not entity["moving"] or not hist:
        return

    records = hist[-40:]
    points = {p for _, p in records}
    coords = [_xy(p) for p in points]
    xmin = min(x for x, y in coords)
    xmax = max(x for x, y in coords)
    ymin = min(y for x, y in coords)
    ymax = max(y for x, y in coords)
    candidates = []

    old = entity.get("path")
    if old and points.issubset(old):
        candidates.append((set(old), -0.25))

    if ymin == ymax:
        y = ymin
        for left in range(xmin + 1):
            for right in range(max(xmax, left + 1), 12):
                path = {y * 12 + x for x in range(left, right + 1)}
                penalty = .04 * len(path)
                for x, d in ((left, -1), (right, 1)):
                    q = _neighbor(y * 12 + x, d, 0)
                    if q >= 0 and board[q] == 0:
                        penalty += 1.3
                candidates.append((path, penalty))

    if xmin == xmax:
        x = xmin
        for top in range(ymin + 1):
            for bottom in range(max(ymax, top + 1), 12):
                path = {y * 12 + x for y in range(top, bottom + 1)}
                penalty = .04 * len(path)
                for y, d in ((top, -1), (bottom, 1)):
                    q = _neighbor(y * 12 + x, 0, d)
                    if q >= 0 and board[q] == 0:
                        penalty += 1.3
                candidates.append((path, penalty))

    for left in range(xmin + 1):
        for right in range(max(xmax, left + 1), 12):
            for top in range(ymin + 1):
                for bottom in range(max(ymax, top + 1), 12):
                    if not all(x in (left, right) or y in (top, bottom)
                               for x, y in coords):
                        continue
                    path = (
                        {top * 12 + x for x in range(left, right + 1)} |
                        {bottom * 12 + x for x in range(left, right + 1)} |
                        {y * 12 + left for y in range(top, bottom + 1)} |
                        {y * 12 + right for y in range(top, bottom + 1)}
                    )
                    candidates.append((path, .2 + .09 * len(path)))

    fallback = set(points)
    for p in points:
        for dx, dy in _DIRS:
            q = _neighbor(p, dx, dy)
            if q >= 0 and board[q] == 0:
                fallback.add(q)
    candidates.append((fallback, 8.0))

    wanted = dict(records)
    first_t, first_p = records[0]
    last_t = records[-1][0]
    best_score = float("inf")
    best = None

    for path, prior in candidates:
        if not points.issubset(path) or any(board[p] == 1 for p in path):
            continue
        score = prior + .35 * sum(board[p] < 0 for p in path)
        if score >= best_score:
            continue
        for initial_dir in range(4):
            p, d = first_p, initial_dir
            good = True
            for t in range(first_t + 1, last_t + 1):
                p, d = _route_step(p, d, path)
                if t in wanted and wanted[t] != p:
                    good = False
                    break
            if not good:
                continue
            for _ in range(last_t, tick):
                p, d = _route_step(p, d, path)
            if p != entity["pos"]:
                continue
            score2 = score + .001 * initial_dir
            if score2 < best_score:
                best_score = score2
                best = tuple(sorted(path)), d

    if best:
        entity["path"], entity["dir"] = best


def _make_model(memory, board, goals, batteries, pushers, entities):
    moving = []
    static = {}
    specifications = []
    for e in entities:
        path = e.get("path") if e.get("moving") else None
        if path and e["pos"] in path:
            moving.append((e["kind"], e["pos"], e.get("dir", 0),
                           set(path), e["id"]))
            specifications.append((e["id"], e["kind"], tuple(path)))
        else:
            static.setdefault(e["pos"], []).append(e["kind"])
            specifications.append((e["id"], e["kind"], e["pos"]))

    phases = []
    seen = {}
    cur = tuple((p, d) for _, p, d, _, _ in moving)
    repeat = None
    for _ in range(384):
        if cur in seen:
            repeat = seen[cur]
            break
        seen[cur] = len(phases)
        phases.append(cur)
        cur = tuple(
            _route_step(cur[i][0], cur[i][1], moving[i][3])
            for i in range(len(moving))
        )

    current_phase = 0
    if repeat == 0:
        shift = min(range(len(phases)), key=lambda i: phases[i])
        phases = phases[shift:] + phases[:shift]
        current_phase = (-shift) % len(phases)

    next_phase = list(range(1, len(phases))) + [
        repeat if repeat is not None else len(phases) - 1
    ]
    effects = []
    phase_positions = []
    deltas = []
    for phase in phases:
        ef = {p: list(k) for p, k in static.items()}
        positions = {}
        for i, (kind, _, _, _, eid) in enumerate(moving):
            ef.setdefault(phase[i][0], []).append(kind)
            positions[eid] = phase[i][0]
        effects.append(ef)
        phase_positions.append(positions)
        dd = {}
        for p, kinds in ef.items():
            dd[p] = (kinds.count(0) % 6,
                     kinds.count(1) % 4,
                     kinds.count(2) % 4)
        deltas.append(dd)

    goal_at = {g["pos"]: i for i, g in enumerate(goals)}
    battery_at = {b["pos"]: i for i, b in enumerate(batteries)}
    full = (1 << len(goals)) - 1
    gm0 = sum(1 << i for i, g in enumerate(goals) if g.get("done"))
    bm0 = sum(1 << i for i, b in enumerate(batteries) if b.get("used"))
    decrement = memory.get("decrement", 2)
    capacity = memory.get("capacity", 42)
    free_match = memory.get("free_match", False)

    def modify(p, s, c, r, phase):
        ds, dc, dr = deltas[phase].get(p, (0, 0, 0))
        return (s + ds) % 6, (c + dc) % 4, (r + dr) % 4, (
            p in effects[phase]
        )

    def transition(state, action):
        p, s, c, r, gm, bm, phase, energy = state
        dx, dy = _ADXY[action]
        q = _neighbor(p, dx, dy)
        if q < 0 or board[q] != 0:
            return None
        phase2 = next_phase[phase]
        s, c, r, changed = modify(q, s, c, r, phase2)

        gi = goal_at.get(q)
        if gi is not None and not (gm & (1 << gi)):
            if (s, c, r) != tuple(goals[gi]["target"]):
                return None

        bi = battery_at.get(q)
        if bi is not None and not (bm & (1 << bi)):
            bm |= 1 << bi
            energy = capacity
        elif not (free_match and changed and any(
            (s, c, r) == tuple(g["target"]) and not (gm & (1 << i))
            for i, g in enumerate(goals)
        )):
            energy -= decrement

        pushed = False
        if energy >= 0 and q in pushers and pushers[q] != q:
            q = pushers[q]
            pushed = True
            s, c, r, _ = modify(q, s, c, r, phase2)
            bi = battery_at.get(q)
            if bi is not None and not (bm & (1 << bi)):
                bm |= 1 << bi
                energy = capacity

        if not pushed:
            gi = goal_at.get(q)
            if gi is not None and (s, c, r) == tuple(goals[gi]["target"]):
                gm |= 1 << gi
        if energy < 0 and gm != full:
            return None
        return q, s, c, r, gm, bm, phase2, energy

    signature = (
        tuple(int(v) for v in board),
        tuple((g["pos"], tuple(g["target"])) for g in goals),
        tuple(b["pos"] for b in batteries),
        tuple(sorted(pushers.items())),
        tuple(sorted(specifications)),
        decrement, free_match,
    )
    return {
        "step": transition, "full": full, "gm": gm0, "bm": bm0,
        "signature": signature, "positions": phase_positions,
        "phases": tuple(phases), "phase": current_phase,
        "next_phase": next_phase, "deltas": deltas,
        "T": len(phases),
    }


def _build_heuristics(model, board, goals, pushers, deadline):
    """Reverse searches in the relaxed position/appearance/phase graph."""
    T = model["T"]
    phase_relaxed = T > 48
    count_pp = 144 if phase_relaxed else 144 * T
    reverse = [[] for _ in range(count_pp)]
    inverse_tables = {}

    def inverse(delta):
        if delta not in inverse_tables:
            ds, dc, dr = delta
            inverse_tables[delta] = tuple(
                ((v // 16 - ds) % 6) * 16 +
                (((v // 4) % 4 - dc) % 4) * 4 +
                ((v % 4 - dr) % 4)
                for v in range(96)
            )
        return inverse_tables[delta]

    added = set()
    for phase in range(T):
        phase2 = model["next_phase"][phase]
        changes = model["deltas"][phase2]
        for p in range(144):
            if board[p] != 0:
                continue
            pre = p if phase_relaxed else phase * 144 + p
            for dx, dy in _ADXY:
                q = _neighbor(p, dx, dy)
                if q < 0 or board[q] != 0:
                    continue
                ds, dc, dr = changes.get(q, (0, 0, 0))
                dest = pushers.get(q, q)
                if dest != q:
                    es, ec, er = changes.get(dest, (0, 0, 0))
                    ds, dc, dr = (ds + es) % 6, (dc + ec) % 4, (dr + er) % 4
                post = dest if phase_relaxed else phase2 * 144 + dest
                delta = ds, dc, dr
                edge = post, pre, delta
                if edge not in added:
                    added.add(edge)
                    reverse[post].append((pre * 96, inverse(delta)))

    distances = []
    N = count_pp * 96
    for goal in goals:
        ds = array("h", [-1]) * N
        s, c, r = goal["target"]
        value = s * 16 + c * 4 + r
        queue = deque()
        for phase in range(1 if phase_relaxed else T):
            node = (phase * 144 + goal["pos"]) * 96 + value
            ds[node] = 0
            queue.append(node)
        iterations = 0
        while queue:
            node = queue.popleft()
            pp, v = divmod(node, 96)
            nd = ds[node] + 1
            for base, inv in reverse[pp]:
                parent = base + inv[v]
                if ds[parent] < 0:
                    ds[parent] = nd
                    queue.append(parent)
            iterations += 1
            if iterations % 12000 == 0 and time.monotonic() > deadline:
                return None
        distances.append(ds)

    # Position-only distances supply a lower bound between successive goals.
    rev_pos = [[] for _ in range(144)]
    for p in range(144):
        if board[p] != 0:
            continue
        for dx, dy in _ADXY:
            q = _neighbor(p, dx, dy)
            if q >= 0 and board[q] == 0:
                rev_pos[pushers.get(q, q)].append(p)
    pos_distances = []
    for goal in goals:
        ds = [999] * 144
        ds[goal["pos"]] = 0
        queue = deque([goal["pos"]])
        while queue:
            p = queue.popleft()
            for q in rev_pos[p]:
                if ds[q] == 999:
                    ds[q] = ds[p] + 1
                    queue.append(q)
        pos_distances.append(ds)

    n = len(goals)
    mst = {}
    if n <= 10:
        for mask in range(1 << n):
            remaining = [i for i in range(n) if mask & (1 << i)]
            if len(remaining) < 2:
                mst[mask] = 0
                continue
            reached = {remaining[0]}
            total = 0
            while len(reached) < len(remaining):
                best = (999, None)
                for i in reached:
                    for j in remaining:
                        if j not in reached:
                            d = min(pos_distances[i][goals[j]["pos"]],
                                    pos_distances[j][goals[i]["pos"]])
                            if d < best[0]:
                                best = d, j
                if best[1] is None:
                    total = 999
                    break
                total += best[0]
                reached.add(best[1])
            mst[mask] = total
    return distances, phase_relaxed, mst, pos_distances


def _plan(model, root, board, goals, pushers, memory, deadline):
    if not goals or root[4] == model["full"]:
        return []

    cache_key = model["signature"], model["phases"]
    cache = memory.get("heuristic_cache")
    if not cache or cache["key"] != cache_key:
        data = _build_heuristics(model, board, goals, pushers, deadline - 2.0)
        if data is None:
            return None
        cache = {"key": cache_key, "data": data}
        memory["heuristic_cache"] = cache

    distances, relaxed, mst, pos_distances = cache["data"]
    full = model["full"]

    def heuristic(state):
        p, s, c, r, gm, bm, phase, energy = state
        remaining = full & ~gm
        if not remaining:
            return 0
        index = ((0 if relaxed else phase) * 144 + p) * 96
        index += s * 16 + c * 4 + r
        nearest = 999
        for i in range(len(goals)):
            if remaining & (1 << i):
                d = distances[i][index]
                if d < 0:
                    return 999
                if d < nearest:
                    nearest = d
        return nearest + mst.get(remaining, 0)

    h0 = heuristic(root)
    if h0 >= 999:
        return None

    def key(st):
        p, s, c, r, gm, bm, phase, energy = st
        return p, s * 16 + c * 4 + r, gm, bm, phase

    states = [root]
    parents = [-1]
    actions = [-1]
    heap = [(h0, -root[7], 0, 0)]
    labels = {key(root): [(0, root[7])]}
    step = model["step"]
    best_node = 0
    best_progress = (root[4].bit_count(), -h0, root[7])
    complete = False
    limited = False

    while heap:
        if len(states) > 450000 or time.monotonic() > deadline:
            limited = True
            break
        f, neg_energy, cost, index = heapq.heappop(heap)
        st = states[index]
        if (cost, st[7]) not in labels.get(key(st), ()):
            continue
        if st[4] == full:
            best_node = index
            complete = True
            break

        h = heuristic(st)
        progress = st[4].bit_count(), -h, st[7]
        if progress > best_progress:
            best_progress, best_node = progress, index

        for action in range(4):
            ns = step(st, action)
            if ns is None:
                continue
            ng = cost + 1
            k = key(ns)
            old = labels.get(k, [])
            if any(g <= ng and e >= ns[7] for g, e in old):
                continue
            nh = heuristic(ns)
            if nh >= 999:
                continue
            labels[k] = [(g, e) for g, e in old
                         if not (ng <= g and ns[7] >= e)] + [(ng, ns[7])]
            ni = len(states)
            states.append(ns)
            parents.append(index)
            actions.append(action)
            heapq.heappush(heap, (ng + nh, -ns[7], ng, ni))

    if not complete and (not limited or best_node == 0):
        return None
    result = []
    while parents[best_node] >= 0:
        result.append(actions[best_node])
        best_node = parents[best_node]
    result.reverse()
    if not complete:
        result = result[:8]
    return result


def _explore(root, model, board, seen_pixels, xoff, yoff, batteries, memory):
    queue = deque([root])
    visited = {root[0]}
    first = {root[0]: None}
    costs = {root[0]: 0}
    best_action = None
    best_score = -1e9
    visits = memory.setdefault("explore_visits", {})
    visits[root[0]] = visits.get(root[0], 0) + 1
    step = model["step"]

    while queue:
        st = queue.popleft()
        p = st[0]
        x, y = _xy(p)
        px, py = xoff + 5 * x + 2, yoff + 5 * y + 2
        unknown = 0
        for yy in range(max(0, py - 18), min(60, py + 19), 3):
            for xx in range(max(4, px - 18), min(64, px + 19), 3):
                if (xx - px) ** 2 + (yy - py) ** 2 <= 360:
                    unknown += not seen_pixels[yy, xx]
        score = unknown - 1.5 * costs[p] - .4 * visits.get(p, 0)
        if root[7] < 18:
            for b in batteries:
                if b["pos"] == p and not b.get("used"):
                    score += 30
        if first[p] is not None and score > best_score:
            best_score, best_action = score, first[p]
        if costs[p] >= 20:
            continue
        for a in range(4):
            ns = step(st, a)
            if ns is None or ns[0] in visited:
                continue
            q = ns[0]
            visited.add(q)
            costs[q] = costs[p] + 1
            first[q] = a if first[p] is None else first[p]
            queue.append(ns)
    return best_action, best_score


def act(obs, available, memory):
    started = time.monotonic()
    obs = np.asarray(obs)
    allowed = set(available)

    def emit(name):
        if name in allowed:
            return name, None, None
        for a in _ACTIONS:
            if a in allowed:
                return a, None, None
        return available[0], None, None

    player = _find_player(obs)
    if player is None:
        if "RESET" in allowed:
            memory["reset_pending"] = True
            return emit("RESET")
        return emit("ACTION1")

    px, py = player
    xoff = memory.setdefault("xoff", px % 5)
    yoff = memory.setdefault("yoff", py % 5)
    cx, cy = (px - xoff) // 5, (py - yoff) // 5
    if not (0 <= cx < 12 and 0 <= cy < 12):
        memory["reset_pending"] = True
        return emit("RESET" if "RESET" in allowed else "ACTION1")
    p = cy * 12 + cx

    shape = _decode_shape(obs[55:61:2, 3:9:2])
    if shape is None:
        shape = tuple(memory.get("shape", (0, 2, 0)))
    memory["shape"] = shape
    energy = int(np.count_nonzero(obs[61, 13:55] == 11))
    lives = sum(int(obs[61, x] == 8) for x in (56, 59, 62))
    memory.setdefault("capacity", 42)

    previous = memory.get("last")
    reset = memory.pop("reset_pending", False)
    death = previous is not None and lives < previous["lives"]
    if reset or death:
        memory["tick"] = 0
        memory["entities"] = []
        memory["plan"] = []
        memory["explore_visits"] = {}
        memory.pop("expected", None)
        memory.pop("heuristic_cache", None)
        for g in memory.get("goals", []):
            g["done"] = False
        for b in memory.get("batteries", []):
            b["used"] = False
        previous = None

    moved = previous is not None and p != previous["p"]
    tick = memory.get("tick", 0) + int(moved)
    memory["tick"] = tick
    if previous:
        diff = previous["energy"] - energy
        if moved and diff in (1, 2, 3, 4) and energy < 42:
            memory["decrement"] = diff

    yy, xx = np.indices((64, 64))
    radius = (xx - (px + 1.5)) ** 2 + (yy - (py + 1.5)) ** 2
    outside = (radius > 400) & (yy < 60) & (xx >= 4)
    if np.count_nonzero(outside & (obs == 5)) > 950:
        memory["fog"] = True
    visible = radius <= 400 if memory.get("fog") else np.ones((64, 64), bool)
    visible[60:] = False
    visible[53:, :12] = False

    if "image" not in memory:
        memory["image"] = np.full((64, 64), -1, dtype=np.int16)
        memory["seen"] = np.zeros((64, 64), dtype=bool)
    image = memory["image"]
    seen_pixels = memory["seen"]
    image[visible] = obs[visible]
    seen_pixels[visible] = True

    board = np.full(144, -1, dtype=np.int8)
    observed_entities = []
    found_goals = []
    found_batteries = []
    color_pattern = np.array([[9, 14, 14], [9, 0, 8], [12, 12, 8]])
    shape_zero = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], bool)
    rot_zero = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], bool)
    rot_one = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], bool)
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)

    for ty in range(12):
        for tx in range(12):
            q = ty * 12 + tx
            x, y = xoff + 5 * tx, yoff + 5 * ty
            if x + 5 > 64 or y + 5 > 60:
                continue
            tile = image[y:y + 5, x:x + 5]
            known = seen_pixels[y:y + 5, x:x + 5]
            if np.any(tile == 4):
                board[q] = 1
            elif np.all(known):
                board[q] = 0

            if np.all(seen_pixels[y + 1:y + 4, x + 1:x + 4]):
                target = _decode_shape(image[y + 1:y + 4, x + 1:x + 4])
                if target is not None and np.count_nonzero(tile == 5) >= 8:
                    found_goals.append((q, target))
                    board[q] = 0

            if not np.all(visible[y + 1:y + 4, x + 1:x + 4]):
                continue
            mid = obs[y + 1:y + 4, x + 1:x + 4]
            if np.array_equal(mid, color_pattern):
                observed_entities.append((1, q))
            elif np.array_equal(mid == 0, shape_zero):
                observed_entities.append((0, q))
            elif (np.array_equal(mid == 0, rot_zero) and
                  np.array_equal(mid == 1, rot_one)):
                observed_entities.append((2, q))
            if np.array_equal(mid == 11, ring):
                found_batteries.append(q)
                board[q] = 0

    board[p] = 0
    memory.setdefault("visited", set()).add(p)
    for q in memory["visited"]:
        board[q] = 0

    goals = memory.setdefault("goals", [])
    goal_lookup = {g["pos"]: g for g in goals}
    for q, target in found_goals:
        if q not in goal_lookup:
            g = {"pos": q, "target": target, "done": False}
            goals.append(g)
            goal_lookup[q] = g
        else:
            goal_lookup[q]["target"] = target
    for g in goals:
        board[g["pos"]] = 0
        if p == g["pos"] and shape == tuple(g["target"]):
            g["done"] = True
        gx, gy = _xy(g["pos"])
        x, y = xoff + 5 * gx, yoff + 5 * gy
        if p != g["pos"] and np.all(visible[y + 1:y + 4, x + 1:x + 4]):
            target = _decode_shape(obs[y + 1:y + 4, x + 1:x + 4])
            if target == tuple(g["target"]):
                g["done"] = False

    batteries = memory.setdefault("batteries", [])
    battery_lookup = {b["pos"]: b for b in batteries}
    for q in found_batteries:
        if q not in battery_lookup:
            b = {"pos": q, "used": False}
            batteries.append(b)
            battery_lookup[q] = b
        battery_lookup[q]["used"] = False
    for b in batteries:
        board[b["pos"]] = 0
        if moved and p == b["pos"]:
            b["used"] = True

    entities = memory.setdefault("entities", [])
    expected = memory.get("expected")
    old_positions = {e["id"]: e["pos"] for e in entities}
    if moved and expected:
        for e in entities:
            if e["id"] in expected.get("entities", {}):
                e["pos"] = expected["entities"][e["id"]]

    unmatched = set(range(len(entities)))
    assignments = []
    for kind, q in observed_entities:
        candidates = [
            (abs(_xy(e["pos"])[0] - _xy(q)[0]) +
             abs(_xy(e["pos"])[1] - _xy(q)[1]), i)
            for i, e in enumerate(entities)
            if i in unmatched and e["kind"] == kind
        ]
        if candidates and min(candidates)[0] <= 3:
            _, i = min(candidates)
            unmatched.remove(i)
            assignments.append((i, q))
        else:
            i = len(entities)
            eid = memory.get("next_entity_id", 0)
            memory["next_entity_id"] = eid + 1
            entities.append({
                "id": eid, "kind": kind, "pos": q,
                "hist": [], "moving": False, "dir": 0,
            })
            assignments.append((i, q))

    if moved and not memory.get("fog"):
        for i in unmatched:
            e = entities[i]
            old_p = old_positions.get(e["id"], e["pos"])
            ox, oy = _xy(old_p)
            if abs(ox - cx) + abs(oy - cy) <= 1:
                assignments.append((i, p))

    for i, q in assignments:
        e = entities[i]
        e["pos"] = q
        hist = e.setdefault("hist", [])
        if hist and hist[-1][0] == tick:
            hist[-1] = tick, q
        else:
            hist.append((tick, q))
        if len(hist) > 52:
            del hist[:-52]
        board[q] = 0

    for e in entities:
        _infer_route(e, board, tick)

    triggers = {}
    for ty in range(12):
        y = yoff + 5 * ty
        if y > 55:
            continue
        for x in range(64):
            if np.all(image[y:y + 5, x] == 1):
                if (x - xoff) % 5 == 4:
                    fx, dx = x + 1, 1
                elif (x - xoff) % 5 == 0:
                    fx, dx = x - 5, -1
                else:
                    continue
                tx = (fx - xoff) // 5
                if 0 <= tx < 12:
                    triggers[ty * 12 + tx] = dx, 0
    for tx in range(12):
        x = xoff + 5 * tx
        if x > 59:
            continue
        for y in range(60):
            if np.all(image[y, x:x + 5] == 1):
                if (y - yoff) % 5 == 4:
                    fy, dy = y + 1, 1
                elif (y - yoff) % 5 == 0:
                    fy, dy = y - 5, -1
                else:
                    continue
                ty = (fy - yoff) // 5
                if 0 <= ty < 12:
                    triggers[ty * 12 + tx] = 0, dy

    blockers = {q for q in range(144) if board[q] == 1}
    blockers.update(g["pos"] for g in goals)
    pushers = {}
    for q, (dx, dy) in triggers.items():
        if board[q] != 0:
            continue
        last = q
        for _ in range(11):
            dest = _neighbor(last, dx, dy)
            if dest < 0 or dest in blockers:
                pushers[q] = last
                break
            last = dest

    if (previous and moved and energy == previous["energy"] and
            shape != previous["shape"] and p not in battery_lookup and
            any(shape == tuple(g["target"]) and not g.get("done")
                for g in goals)):
        memory["free_match"] = True

    model = _make_model(memory, board, goals, batteries, pushers, entities)
    root = (p, shape[0], shape[1], shape[2],
            model["gm"], model["bm"], model["phase"], energy)

    valid_prediction = (
        expected is not None and expected.get("p") == p and
        tuple(expected.get("shape", ())) == shape and
        expected.get("energy") == energy and
        memory.get("plan_signature") == model["signature"]
    )
    if valid_prediction:
        observed_positions = {entities[i]["id"]: q for i, q in assignments}
        for eid, q in expected.get("entities", {}).items():
            if eid in observed_positions and observed_positions[eid] != q:
                valid_prediction = False
                break

    plan = memory.get("plan", []) if valid_prediction else []
    if plan and model["step"](root, plan[0]) is None:
        plan = []

    if not plan and goals and root[4] != model["full"]:
        plan = _plan(model, root, board, goals, pushers, memory,
                     started + 8.0) or []

    action = plan.pop(0) if plan else None
    if action is None:
        exploratory, score = _explore(
            root, model, board, seen_pixels, xoff, yoff, batteries, memory
        )
        unknown_exists = bool(np.any(board < 0))
        if exploratory is not None and (
                unknown_exists or not goals or score > 3):
            action = exploratory
        elif (previous is not None and memory.get("resets", 0) < 2 and
              "RESET" in allowed and (energy < 30 or tick > 8)):
            memory["resets"] = memory.get("resets", 0) + 1
            memory["reset_pending"] = True
            memory["plan"] = []
            memory["last"] = {
                "p": p, "shape": shape, "energy": energy, "lives": lives
            }
            return emit("RESET")
        elif exploratory is not None:
            action = exploratory
        else:
            action = next((a for a in range(4)
                           if model["step"](root, a) is not None), 0)

    memory["plan"] = plan
    memory["plan_signature"] = model["signature"]
    ns = model["step"](root, action)
    if ns is not None:
        predicted = {e["id"]: e["pos"] for e in entities}
        predicted.update(model["positions"][ns[6]])
        memory["expected"] = {
            "p": ns[0], "shape": tuple(ns[1:4]),
            "energy": ns[7], "entities": predicted,
        }
    else:
        memory.pop("expected", None)
        memory["plan"] = []
    memory["last"] = {
        "p": p, "shape": shape, "energy": energy, "lives": lives
    }
    return emit(_ACTIONS[action])

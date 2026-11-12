from collections import deque
from heapq import heappush, heappop


def _hungarian(cost):
    """Minimum-cost assignment for a square matrix."""
    n = len(cost)
    if not n:
        return []
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [10**12] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**12
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = [0] * n
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    return result


def _components(obs):
    """Find four-connected monochromatic components, excluding background/UI."""
    seen = [[False] * 64 for _ in range(64)]
    out = []
    excluded = {3, 4, 5}

    for r in range(64):
        for c in range(1, 64):
            color = int(obs[r, c])
            if r == 1 or color in excluded or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            n = 0
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                y, x = stack.pop()
                n += 1
                r0 = min(r0, y)
                r1 = max(r1, y)
                c0 = min(c0, x)
                c1 = max(c1, x)
                for yy, xx in ((y - 1, x), (y + 1, x),
                               (y, x - 1), (y, x + 1)):
                    if (0 <= yy < 64 and 1 <= xx < 64 and yy != 1
                            and not seen[yy][xx]
                            and int(obs[yy, xx]) == color):
                        seen[yy][xx] = True
                        stack.append((yy, xx))
            out.append((color, r0, c0, r1, c1, n))
    return out


def _read_board(obs):
    comps = _components(obs)

    # Tiles are solid 2-by-2 game-pixel squares. Camera scaling is integral.
    scores = {}
    for scale in (1, 2, 3):
        side = 2 * scale
        scores[scale] = sum(
            1 for color, r0, c0, r1, c1, n in comps
            if color not in (8, 14)
            and r1 - r0 + 1 == side
            and c1 - c0 + 1 == side
            and n == side * side
        )
    scale = max((1, 2, 3), key=lambda s: (scores[s], -s))
    side = 2 * scale

    tiles = {}
    for color, r0, c0, r1, c1, n in comps:
        if (color not in (8, 14)
                and r1 - r0 + 1 == side
                and c1 - c0 + 1 == side
                and n == side * side):
            tiles[str(r0 * 64 + c0)] = color

    markers = {11: [], 12: []}
    for key in tiles:
        pos = int(key)
        r, c = divmod(pos, 64)
        corners = [
            (r - scale, c - scale),
            (r - scale, c + 2 * scale),
            (r + 2 * scale, c - scale),
            (r + 2 * scale, c + 2 * scale),
        ]
        if all(0 <= y < 64 and 0 <= x < 64 for y, x in corners):
            color = int(obs[corners[0][0], corners[0][1]])
            if color in markers and all(
                    int(obs[y, x]) == color for y, x in corners):
                markers[color].append(pos)

    buttons = []
    for color, r0, c0, r1, c1, n in comps:
        if color not in (8, 14):
            continue
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (n >= 5 * scale * scale
                and h <= 6 * scale and w <= 6 * scale):
            buttons.append({
                "color": color,
                "r": (r0 + r1) // 2,
                "c": (c0 + c1) // 2,
            })
    buttons.sort(key=lambda b: (b["r"], b["c"], b["color"]))
    return scale, tiles, markers, buttons


def _pair_buttons(buttons):
    """Pair oppositely colored controls, allowing unpaired controls."""
    blue = [i for i, b in enumerate(buttons) if b["color"] == 8]
    orange = [i for i, b in enumerate(buttons) if b["color"] == 14]
    size = max(len(blue), len(orange))
    groups = []
    if size:
        costs = []
        for i in range(size):
            row = []
            for j in range(size):
                if i < len(blue) and j < len(orange):
                    a, b = buttons[blue[i]], buttons[orange[j]]
                    row.append(abs(a["r"] - b["r"]) +
                               abs(a["c"] - b["c"]))
                else:
                    row.append(1000)
            costs.append(row)
        assignment = _hungarian(costs)
        used_orange = set()
        for i, j in enumerate(assignment):
            if i < len(blue):
                if j < len(orange):
                    groups.append([blue[i], orange[j]])
                    used_orange.add(j)
                else:
                    groups.append([blue[i]])
        for j in range(len(orange)):
            if j not in used_orange:
                groups.append([orange[j]])
    return groups


def _compatible(samples, sites):
    """Whether the observations can describe one fixed permutation."""
    if not samples:
        return True
    left = {}
    right = {}
    for pos in sites:
        key = str(pos)
        a = tuple(before.get(key, 4) for before, after in samples)
        b = tuple(after.get(key, 4) for before, after in samples)
        left[a] = left.get(a, 0) + 1
        right[b] = right.get(b, 0) + 1
    return left == right


def _infer_permutation(samples, sites):
    n = len(sites)
    if not samples or not _compatible(samples, sites):
        return list(range(n))

    sources = {}
    destinations = {}
    for i, pos in enumerate(sites):
        key = str(pos)
        a = tuple(before.get(key, 4) for before, after in samples)
        b = tuple(after.get(key, 4) for before, after in samples)
        sources.setdefault(a, []).append(i)
        destinations.setdefault(b, []).append(i)

    perm = list(range(n))
    for signature, src in sources.items():
        dst = destinations[signature]
        if len(src) == 1:
            perm[src[0]] = dst[0]
            continue

        # Among visually indistinguishable assignments, prefer stationary
        # tiles and short moves. Subsequent observations refine the model.
        cost = []
        for i in src:
            r1, c1 = divmod(sites[i], 64)
            row = []
            for j in dst:
                r2, c2 = divmod(sites[j], 64)
                d = abs(r1 - r2) + abs(c1 - c2)
                row.append(0 if i == j else 20 + 4 * d)
            cost.append(row)
        assignment = _hungarian(cost)
        for k, j in enumerate(assignment):
            perm[src[k]] = dst[j]
    return perm


def _models(groups, records, sites, nbuttons):
    models = [list(range(len(sites))) for _ in range(nbuttons)]
    for group in groups:
        a = group[0]
        sa = records.get(str(a), [])
        if len(group) == 1:
            models[a] = _infer_permutation(sa, sites)
            continue

        b = group[1]
        sb = records.get(str(b), [])
        combined = list(sa) + [[after, before] for before, after in sb]
        if _compatible(combined, sites):
            p = _infer_permutation(combined, sites)
            inverse = [0] * len(p)
            for i, j in enumerate(p):
                inverse[j] = i
            models[a] = p
            models[b] = inverse
        else:
            # Overlapping controls need not always be exact inverses.
            models[a] = _infer_permutation(sa, sites)
            models[b] = _infer_permutation(sb, sites)
    return models


def _search(tiles, markers, sites, models):
    """A* over the positions of goal tiles; ordinary tiles need not be tracked."""
    index = {pos: i for i, pos in enumerate(sites)}
    target_groups = []
    start_groups = []
    for color in (11, 12):
        targets = [index[p] for p in markers.get(color, []) if p in index]
        goals = sorted(index[int(k)] for k, v in tiles.items()
                       if v == color and int(k) in index)
        if len(goals) < len(targets):
            return None
        target_groups.append(tuple(targets))
        start_groups.append(tuple(goals))

    if not any(target_groups):
        return None

    start = tuple(start_groups)
    if all(all(t in start[k] for t in target_groups[k]) for k in (0, 1)):
        # The game checks completion after a button press.
        for action, p in enumerate(models):
            nxt = tuple(tuple(sorted(p[x] for x in group)) for group in start)
            if all(all(t in nxt[k] for t in target_groups[k])
                   for k in (0, 1)):
                return action, 1
        return None

    n = len(sites)
    reverse = [[] for _ in range(n)]
    for p in models:
        for i, j in enumerate(p):
            if i != j:
                reverse[j].append(i)

    distances = {}
    for targets in target_groups:
        for target in targets:
            dist = [10**6] * n
            dist[target] = 0
            q = deque([target])
            while q:
                x = q.popleft()
                for y in reverse[x]:
                    if dist[y] > dist[x] + 1:
                        dist[y] = dist[x] + 1
                        q.append(y)
            distances[target] = dist

    def heuristic(state):
        h = 0
        for k in (0, 1):
            for target in target_groups[k]:
                d = min((distances[target][x] for x in state[k]),
                        default=10**6)
                if d > h:
                    h = d
        return h

    h0 = heuristic(start)
    if h0 >= 10**6:
        return None

    queue = []
    serial = 0
    heappush(queue, (h0, 0, serial, -1, start))
    best = {start: 0}
    expansions = 0

    while queue and expansions < 25000:
        _, depth, _, first, state = heappop(queue)
        if best.get(state) != depth:
            continue
        if all(all(t in state[k] for t in target_groups[k])
               for k in (0, 1)):
            return first, depth
        expansions += 1

        for action, p in enumerate(models):
            nxt = tuple(tuple(sorted(p[x] for x in group)) for group in state)
            if nxt == state:
                continue
            nd = depth + 1
            if nd >= best.get(nxt, 10**9):
                continue
            h = heuristic(nxt)
            if h >= 10**6:
                continue
            best[nxt] = nd
            serial += 1
            heappush(queue, (nd + h, nd, serial,
                             action if first < 0 else first, nxt))
    return None


def act(obs, available, memory):
    scale, tiles, observed_markers, found_buttons = _read_board(obs)
    bar = sum(1 for r in range(64) if int(obs[r, 0]) == 5)

    if "buttons" not in memory:
        memory["buttons"] = found_buttons
        memory["groups"] = _pair_buttons(found_buttons)
        memory["records"] = {}
        memory["markers"] = {
            str(color): list(observed_markers[color]) for color in (11, 12)
        }
        memory["sites"] = sorted(int(k) for k in tiles)
        memory["history"] = []
        memory["bar0"] = bar
        memory["moves"] = 0
        memory["resets"] = 0

    buttons = memory["buttons"]
    groups = memory["groups"]
    records = memory["records"]

    if memory.pop("after_reset", False):
        memory["bar0"] = bar
        memory["moves"] = 0
        memory["history"] = []

    pending = memory.pop("pending", None)
    if pending is not None:
        action, before = pending
        samples = records.setdefault(str(action), [])
        samples.append([before, dict(tiles)])
        # Recent samples are useful when different loops intersect.
        if len(samples) > 10:
            del samples[1]

    sites_set = set(memory["sites"])
    sites_set.update(int(k) for k in tiles)
    for color in (11, 12):
        stored = memory["markers"].setdefault(str(color), [])
        for pos in observed_markers[color]:
            if pos not in stored:
                stored.append(pos)
            sites_set.add(pos)
    sites = sorted(sites_set)
    memory["sites"] = sites
    markers = {
        color: memory["markers"].get(str(color), []) for color in (11, 12)
    }

    if not buttons:
        # A safe fallback for an unrecognized frame.
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    models = _models(groups, records, sites, len(buttons))
    plan = _search(tiles, markers, sites, models)

    goal_state = tuple(
        tuple(sorted(int(k) for k, v in tiles.items() if v == color))
        for color in (11, 12)
    )
    history = memory["history"]
    repetitions = sum(1 for old in history if old == goal_state)
    history.append(goal_state)
    if len(history) > 12:
        del history[0]

    # If the model is incomplete or repeatedly predicts an unproductive
    # route, gather observations in short batches, one control pair at a time.
    if plan is None or repetitions >= 4:
        if groups:
            def group_score(group):
                count = sum(len(records.get(str(i), [])) for i in group)
                return (count // 4, count, group[0])

            group = min(groups, key=group_score)
            chosen = group[0]
        else:
            chosen = 0
        plan_length = 10**6
    else:
        chosen, plan_length = plan

    # Recover once if the visible step meter indicates an imminent loss.
    moves = memory.get("moves", 0)
    delta = bar - memory.get("bar0", bar)
    if (moves >= 2 and delta > 0 and memory.get("resets", 0) == 0
            and "RESET" in available):
        rate = delta / moves
        if rate > 0 and 64 - bar <= rate + 0.5 and plan_length > 1:
            memory["resets"] = 1
            memory["after_reset"] = True
            return "RESET", None, None

    memory["pending"] = [chosen, dict(tiles)]
    memory["moves"] = moves + 1
    button = buttons[chosen]
    return "ACTION6", int(button["r"]), int(button["c"])

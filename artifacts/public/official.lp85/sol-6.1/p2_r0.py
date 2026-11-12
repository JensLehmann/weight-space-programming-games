import math
import time
import heapq
from collections import Counter, deque


def _components(obs):
    """Non-background, four-connected colour components; ignore the HUD."""
    seen = set()
    result = []
    for y in range(2, 64):
        for x in range(1, 64):
            c = int(obs[y, x])
            if c in (3, 4) or (y, x) in seen:
                continue
            stack = [(y, x)]
            seen.add((y, x))
            pixels = []
            while stack:
                yy, xx = stack.pop()
                pixels.append((yy, xx))
                for ny, nx in ((yy - 1, xx), (yy + 1, xx),
                               (yy, xx - 1), (yy, xx + 1)):
                    if (2 <= ny < 64 and 1 <= nx < 64
                            and (ny, nx) not in seen
                            and int(obs[ny, nx]) == c):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            result.append((c, min(ys), min(xs),
                           max(ys) - min(ys) + 1,
                           max(xs) - min(xs) + 1, len(pixels)))
    return result


def _parse(obs):
    comps = _components(obs)
    scales = []
    for c, y, x, h, w, area in comps:
        if c not in (8, 14):
            continue
        a, b = sorted((h, w))
        if a % 3 == 0 and b % 4 == 0 and a // 3 == b // 4:
            s = a // 3
            if s and area == 10 * s * s:
                scales.append(s)
    if scales:
        scale = Counter(scales).most_common(1)[0][0]
    else:
        squares = [h // 2 for c, y, x, h, w, area in comps
                   if c not in (8, 14) and h == w and h % 2 == 0
                   and area == h * w and 2 <= h <= 8]
        scale = Counter(squares).most_common(1)[0][0] if squares else 1

    buttons = []
    points = set()
    targets = {11: set(), 12: set()}
    for c, y, x, h, w, area in comps:
        if (c in (8, 14) and sorted((h, w)) == [3 * scale, 4 * scale]
                and area == 10 * scale * scale):
            buttons.append((c, y, x, h, w))
        elif (c not in (8, 14) and h == w == 2 * scale
              and area == 4 * scale * scale):
            points.add((y, x))

    # A target is a four-corner bracket surrounding a tile slot.
    s = scale
    for c, y, x, h, w, area in comps:
        if c not in (11, 12) or h != s or w != s or area != s * s:
            continue
        if y + 4 * s > 64 or x + 4 * s > 64:
            continue
        good = True
        for dy, dx in ((0, 0), (0, 3 * s), (3 * s, 0), (3 * s, 3 * s)):
            if not bool((obs[y + dy:y + dy + s,
                              x + dx:x + dx + s] == c).all()):
                good = False
                break
        if good:
            p = (y + s, x + s)
            targets[c].add(p)
            points.add(p)

    buttons.sort(key=lambda b: (b[1], b[2], b[0]))
    return scale, buttons, sorted(points), targets


def _encode(obs):
    digits = "0123456789abcdef"
    return "".join(digits[int(v)] for v in obs.ravel())


def _read(frame, points):
    return [int(frame[y * 64 + x], 16) for y, x in points]


def _pair_buttons(buttons):
    """Pair aligned opposite-colour controls, shortest pairs first."""
    n = len(buttons)
    pairs = []
    for i, a in enumerate(buttons):
        for j, b in enumerate(buttons):
            if a[0] != 8 or b[0] != 14 or a[3:5] != b[3:5]:
                continue
            vertical = a[4] > a[3]
            aligned = a[2] == b[2] if vertical else a[1] == b[1]
            if aligned:
                d = abs(a[1] - b[1]) + abs(a[2] - b[2])
                pairs.append((d, i, j))
    pairs.sort()
    owner = list(range(n))
    reverse = [False] * n
    used = set()
    for _, i, j in pairs:
        if i not in used and j not in used:
            owner[j] = i
            reverse[j] = True
            used.update((i, j))
    return owner, reverse


def _adjacency(points, scale):
    index = {tuple(p): i for i, p in enumerate(points)}
    step = 3 * scale
    adj = [[] for _ in points]
    for i, (y, x) in enumerate(points):
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if not (dy or dx):
                    continue
                j = index.get((y + dy, x + dx))
                if j is None:
                    continue
                # Do not shortcut an ordinary right-angle bend.
                if dy and dx:
                    if ((y + dy, x) in index or (y, x + dx) in index):
                        continue
                adj[i].append(j)
    return adj


def _hungarian(cost):
    """Minimum-cost square assignment, returning a column for each row."""
    n = len(cost)
    if n == 1:
        return [0]
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        mins = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < mins[j]:
                        mins[j] = cur
                        way[j] = j0
                    if mins[j] < delta:
                        delta, j1 = mins[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    mins[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    answer = [0] * n
    for j in range(1, n + 1):
        answer[p[j] - 1] = j - 1
    return answer


def _cycles(perm):
    labels = [-1] * len(perm)
    lengths = []
    for i in range(len(perm)):
        if labels[i] >= 0:
            continue
        label = len(lengths)
        cur = i
        length = 0
        while labels[cur] < 0:
            labels[cur] = label
            length += 1
            cur = perm[cur]
        lengths.append(length)
    return labels, lengths


def _learn(samples, points, adj, scale, single_group):
    n = len(points)
    before = [[] for _ in range(n)]
    after = [[] for _ in range(n)]
    active = set()
    for left, right in samples:
        a, b = _read(left, points), _read(right, points)
        for i in range(n):
            before[i].append(a[i])
            after[i].append(b[i])
            if a[i] != b[i]:
                active.add(i)

    # Extend visibly moving portions through geometrically simple runs.
    domain = set(active)
    if single_group:
        queue = list(active)
        for i in queue:
            for j in adj[i]:
                if j not in domain:
                    domain.add(j)
                    queue.append(j)
    else:
        for _ in range(2):
            add = set()
            for i in range(n):
                if i in domain:
                    continue
                neighbours = adj[i]
                if (len(neighbours) == 2
                        and all(j in domain for j in neighbours)):
                    add.add(i)
                elif len(neighbours) == 1 and neighbours[0] in domain:
                    add.add(i)
            domain.update(add)

    src_groups, dst_groups = {}, {}
    for i in range(n):
        src_groups.setdefault(tuple(before[i]), []).append(i)
        dst_groups.setdefault(tuple(after[i]), []).append(i)
    if set(src_groups) != set(dst_groups):
        return None
    if any(len(src_groups[k]) != len(dst_groups[k]) for k in src_groups):
        return None

    penalty = 10000 * (3 * scale) ** 2

    def cost(a, b):
        dy = points[a][0] - points[b][0]
        dx = points[a][1] - points[b][1]
        value = dy * dy + dx * dx
        if a == b and a in domain:
            value += penalty
        if (a in domain) != (b in domain):
            value += 2 * penalty
        return value

    perm = list(range(n))
    for key, sources in src_groups.items():
        destinations = dst_groups[key]
        matrix = [[cost(a, b) for a in sources] for b in destinations]
        assignment = _hungarian(matrix)
        for row, col in enumerate(assignment):
            perm[sources[col]] = destinations[row]

    # Repeated colours can split one rotation into several short cycles.
    # Merge compatible, spatially touching cycles without violating any sample.
    for _ in range(n):
        labels, lengths = _cycles(perm)
        touching = set()
        for i in range(n):
            if lengths[labels[i]] <= 1:
                continue
            for j in adj[i]:
                if (labels[i] != labels[j] and lengths[labels[j]] > 1):
                    touching.add(tuple(sorted((labels[i], labels[j]))))
        best = None
        for sources in src_groups.values():
            moving = [a for a in sources if perm[a] != a]
            for ia, a in enumerate(moving):
                for b in moving[ia + 1:]:
                    if labels[a] == labels[b]:
                        continue
                    if tuple(sorted((labels[a], labels[b]))) not in touching:
                        continue
                    if (a in domain) != (b in domain):
                        continue
                    delta = (cost(a, perm[b]) + cost(b, perm[a])
                             - cost(a, perm[a]) - cost(b, perm[b]))
                    if best is None or delta < best[0]:
                        best = (delta, a, b)
        if best is None:
            break
        _, a, b = best
        perm[a], perm[b] = perm[b], perm[a]

    inverse = [0] * n
    for i, j in enumerate(perm):
        inverse[j] = i
    return perm, inverse


def _move(state, perm, white_count):
    return (tuple(sorted(perm[i] for i in state[:white_count]))
            + tuple(sorted(perm[i] for i in state[white_count:])))


def _join_path(meet, forward, backward):
    left = []
    cur = meet
    while forward[cur][0] is not None:
        prev, action = forward[cur]
        left.append(action)
        cur = prev
    left.reverse()
    cur = meet
    while backward[cur][0] is not None:
        nxt, action = backward[cur]
        left.append(action)
        cur = nxt
    return left


def _bidirectional(start, target, actions, white_count, deadline,
                   node_limit=350000):
    if start == target:
        return []
    forward = {start: (None, None)}
    backward = {target: (None, None)}
    front = [start]
    back = [target]
    checks = 0
    while front and back:
        if len(forward) + len(backward) > node_limit:
            return None
        direction = len(front) <= len(back)
        layer = front if direction else back
        own = forward if direction else backward
        other = backward if direction else forward
        next_layer = []
        for state in layer:
            for action, perm, inverse in actions:
                ns = _move(state, perm if direction else inverse, white_count)
                if ns in own:
                    continue
                own[ns] = (state, action)
                if ns in other:
                    return _join_path(ns, forward, backward)
                next_layer.append(ns)
            checks += 1
            if checks % 512 == 0 and time.monotonic() >= deadline:
                return None
        if direction:
            front = next_layer
        else:
            back = next_layer
    return None


def _weighted_search(start, target, actions, white_count, n, deadline):
    """A heuristic fallback for unusually large goal configurations."""
    if start == target:
        return []
    targets = [target[:white_count], target[white_count:]]
    distances = []
    for group in targets:
        tables = []
        for destination in group:
            dist = [n + 10] * n
            dist[destination] = 0
            q = deque([destination])
            while q:
                v = q.popleft()
                for _, _, inv in actions:
                    u = inv[v]
                    if dist[u] > dist[v] + 1:
                        dist[u] = dist[v] + 1
                        q.append(u)
            tables.append(dist)
        distances.append(tables)

    def matching_cost(nodes, tables):
        m = len(nodes)
        if not m:
            return 0
        dp = {0: 0}
        for node in nodes:
            nd = {}
            for mask, value in dp.items():
                for j in range(m):
                    if not (mask >> j) & 1:
                        nm = mask | (1 << j)
                        nv = value + tables[j][node]
                        if nv < nd.get(nm, float("inf")):
                            nd[nm] = nv
            dp = nd
        return dp[(1 << m) - 1]

    cache = {}

    def heuristic(state):
        if state not in cache:
            v = (matching_cost(state[:white_count], distances[0])
                 + matching_cost(state[white_count:], distances[1]))
            cache[state] = v / max(1.0, math.sqrt(len(state)))
        return cache[state]

    parent = {start: (None, None)}
    gs = {start: 0}
    serial = 0
    heap = [(2.2 * heuristic(start), 0, serial, start)]
    expanded = 0
    while heap and expanded < 35000:
        _, g, _, state = heapq.heappop(heap)
        if g != gs.get(state):
            continue
        if state == target:
            path = []
            while parent[state][0] is not None:
                state, action = parent[state]
                path.append(action)
            return path[::-1]
        for action, perm, _ in actions:
            ns = _move(state, perm, white_count)
            ng = g + 1
            if ng < gs.get(ns, float("inf")):
                gs[ns] = ng
                parent[ns] = (state, action)
                serial += 1
                heapq.heappush(heap, (ng + 2.2 * heuristic(ns),
                                      ng, serial, ns))
        expanded += 1
        if expanded % 128 == 0 and time.monotonic() >= deadline:
            break
    return None


def act(obs, available, memory):
    started = time.monotonic()
    scale, buttons, observed_points, targets = _parse(obs)
    frame = _encode(obs)
    signature = repr((scale, buttons,
                      sorted(targets[11]), sorted(targets[12])))

    if not buttons:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    if memory.get("signature") != signature:
        memory.clear()
        owner, reverse = _pair_buttons(buttons)
        memory.update({
            "signature": signature,
            "owner": owner,
            "reverse": reverse,
            "points": observed_points,
            "samples": [],
            "pending": None,
            "previous": frame,
            "plan": [],
            "turn": 0,
        })

    owner = memory["owner"]
    reverse = memory["reverse"]
    points = [tuple(p) for p in memory["points"]]
    point_set = set(points)
    for p in observed_points:
        if p not in point_set:
            points.append(p)
            point_set.add(p)
    memory["points"] = points

    pending = memory.get("pending")
    mismatch = False
    if pending is not None:
        memory["samples"].append([pending, memory["previous"], frame])
        expected = memory.get("expected")
        if expected is not None:
            actual = _read(frame, points)
            mismatch = actual[:len(expected)] != expected
    memory["pending"] = None

    adj = _adjacency(points, scale)
    models = {}
    counts = {}
    # If a guessed inverse pair is contradicted, learn its controls separately.
    for attempt in range(2):
        grouped = {}
        for button, before, after in memory["samples"]:
            master = owner[button]
            if reverse[button]:
                before, after = after, before
            grouped.setdefault(master, []).append((before, after))
        masters = set(owner)
        models = {}
        bad = []
        counts = {m: len(grouped.get(m, [])) for m in masters}
        for master, samples in grouped.items():
            model = _learn(samples, points, adj, scale, len(masters) == 1)
            if model is None:
                bad.append(master)
            else:
                models[master] = model
        if not bad:
            break
        changed = False
        for master in bad:
            members = [i for i in range(len(buttons)) if owner[i] == master]
            if len(members) > 1:
                for i in members:
                    owner[i] = i
                    reverse[i] = False
                changed = True
        if not changed:
            break

    memory["owner"] = owner
    memory["reverse"] = reverse
    colours = _read(frame, points)
    index = {p: i for i, p in enumerate(points)}
    white = tuple(i for i, c in enumerate(colours) if c == 11)
    orange = tuple(i for i, c in enumerate(colours) if c == 12)
    start = tuple(sorted(white)) + tuple(sorted(orange))
    target = (tuple(sorted(index[p] for p in targets[11]))
              + tuple(sorted(index[p] for p in targets[12])))
    white_count = len(white)
    valid_goals = (len(white) == len(targets[11])
                   and len(orange) == len(targets[12])
                   and len(start) > 0)

    actions = []
    button_models = {}
    for i in range(len(buttons)):
        model = models.get(owner[i])
        if model is None:
            continue
        perm, inv = model
        if reverse[i]:
            perm, inv = inv, perm
        button_models[i] = (perm, inv)
        actions.append((i, perm, inv))

    # Preserve a queued solution only when it still solves the updated model.
    plan = list(memory.get("plan", []))
    if plan and valid_goals:
        state = start
        for button in plan:
            if button not in button_models:
                plan = []
                break
            state = _move(state, button_models[button][0], white_count)
        if plan and state != target:
            plan = []
    else:
        plan = []

    masters = sorted(set(owner))
    unknown = [m for m in masters if counts.get(m, 0) == 0]
    chosen = None

    if valid_goals and start == target:
        for button, perm, inv in actions:
            if _move(start, perm, white_count) == start:
                chosen = button
                plan = []
                break

    if chosen is None and not plan and actions and valid_goals:
        all_seen = not unknown
        search_time = 4.3 if all_seen else 0.8
        deadline = min(started + 7.5, time.monotonic() + search_time)
        candidate = _bidirectional(
            start, target, actions, white_count, deadline,
            350000 if all_seen else 45000
        )
        if candidate is None and all_seen:
            deadline = min(started + 8.3, time.monotonic() + 1.6)
            if deadline > time.monotonic() + 0.05:
                candidate = _weighted_search(
                    start, target, actions, white_count, len(points), deadline
                )
        if candidate:
            # A short solution can be tried before probing every control.
            if all_seen or len(candidate) <= 4:
                plan = candidate

    if chosen is None and plan:
        chosen = plan.pop(0)

    if chosen is None:
        # Probe an unlearned control first, otherwise refine the least sampled.
        pool = unknown if unknown else masters

        wrong_slots = [points[i] for i in start if i not in set(target)]

        def proximity(master):
            members = [i for i in range(len(buttons)) if owner[i] == master]
            if not wrong_slots:
                return master
            value = float("inf")
            for i in members:
                _, y, x, h, w = buttons[i]
                cy, cx = y + h / 2, x + w / 2
                for py, px in wrong_slots:
                    value = min(value, (py - cy) ** 2 + (px - cx) ** 2)
            return value

        if mismatch and pending is not None and not unknown:
            last_master = owner[pending]
            minimum = min(counts.get(m, 0) for m in pool)
            if counts.get(last_master, 0) <= minimum + 1:
                pool = [last_master]
        master = min(pool, key=lambda m: (counts.get(m, 0), proximity(m)))
        members = [i for i in range(len(buttons)) if owner[i] == master]
        chosen = next((i for i in members if not reverse[i]), members[0])
        plan = []

    memory["plan"] = plan
    memory["pending"] = chosen
    memory["previous"] = frame
    memory["turn"] += 1
    if chosen in button_models:
        perm = button_models[chosen][0]
        expected = [0] * len(points)
        for i, j in enumerate(perm):
            expected[j] = colours[i]
        memory["expected"] = expected
    else:
        memory["expected"] = None

    _, y, x, h, w = buttons[chosen]
    return "ACTION6", int(y + h // 2), int(x + w // 2)

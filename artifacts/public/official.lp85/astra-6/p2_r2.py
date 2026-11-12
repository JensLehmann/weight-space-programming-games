import math
import time
import heapq
from collections import defaultdict, Counter, deque


def _components(obs):
    a = obs.tolist()
    for r in range(64):
        a[r][0] = 4
    a[1] = [4] * 64  # Remove the fixed interface.

    seen = bytearray(4096)
    result = []
    for r in range(64):
        for c in range(1, 64):
            z = r * 64 + c
            colour = int(a[r][c])
            if colour == 4 or seen[z]:
                continue
            seen[z] = 1
            stack = [(r, c)]
            lo_r = hi_r = r
            lo_c = hi_c = c
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                lo_r = min(lo_r, y)
                hi_r = max(hi_r, y)
                lo_c = min(lo_c, x)
                hi_c = max(hi_c, x)
                for yy, xx in ((y-1, x), (y+1, x),
                               (y, x-1), (y, x+1)):
                    if 0 <= yy < 64 and 1 <= xx < 64:
                        q = yy * 64 + xx
                        if not seen[q] and a[yy][xx] == colour:
                            seen[q] = 1
                            stack.append((yy, xx))
            result.append((colour, lo_r, lo_c,
                           hi_r-lo_r+1, hi_c-lo_c+1, area))
    return result


def _read(obs):
    comps = _components(obs)
    sizes = Counter()
    for colour, r, c, h, w, area in comps:
        if colour not in (8, 14) and h == w and area == h*h:
            if h in (2, 4, 6, 8):
                sizes[h] += 1
    substantial = [h for h, n in sizes.items() if n >= 2]
    side = max(substantial or list(sizes) or [2])
    scale = side // 2

    tiles = {}
    corners = {11: set(), 12: set()}
    buttons = []
    for colour, r, c, h, w, area in comps:
        if h == w == 2*scale and area == 4*scale*scale:
            if colour not in (8, 14):
                tiles[(r, c)] = colour
        if colour in corners and h == w == scale and area == scale*scale:
            corners[colour].add((r, c))
        if colour in (8, 14):
            if (h, w) in ((4*scale, 3*scale),
                          (3*scale, 4*scale)):
                if 8*scale*scale <= area <= 12*scale*scale:
                    buttons.append({
                        "colour": colour, "r": r, "c": c,
                        "h": h, "w": w,
                        "click": (r+h//2, c+w//2),
                    })

    targets = {11: [], 12: []}
    d = 3*scale
    for colour, points in corners.items():
        for r, c in points:
            if ((r+d, c) in points and (r, c+d) in points
                    and (r+d, c+d) in points):
                targets[colour].append((r+scale, c+scale))
        targets[colour].sort()

    buttons.sort(key=lambda b: (b["r"], b["c"], b["colour"]))
    return scale, tiles, targets, buttons


def _hungarian(cost):
    """Minimum-cost square assignment, row -> column."""
    n = len(cost)
    if not n:
        return []
    u = [0.0] * (n+1)
    v = [0.0] * (n+1)
    p = [0] * (n+1)
    way = [0] * (n+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        best = [float("inf")] * (n+1)
        used = [False] * (n+1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = cost[i0-1]
            for j in range(1, n+1):
                if not used[j]:
                    cur = row[j-1] - u[i0] - v[j]
                    if cur < best[j]:
                        best[j] = cur
                        way[j] = j0
                    if best[j] < delta:
                        delta = best[j]
                        j1 = j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta
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
    for j in range(1, n+1):
        answer[p[j]-1] = j-1
    return answer


def _pair_buttons(buttons):
    left = [i for i, b in enumerate(buttons) if b["colour"] == 8]
    right = [i for i, b in enumerate(buttons) if b["colour"] == 14]
    if not left or not right:
        return [[i] for i in range(len(buttons))]

    n = max(len(left), len(right))
    cost = [[10000.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i >= len(left) or j >= len(right):
                cost[i][j] = 5000.0
                continue
            a, b = buttons[left[i]], buttons[right[j]]
            ah, bh = a["h"] > a["w"], b["h"] > b["w"]
            if ah != bh:
                cost[i][j] = 100000.0
                continue
            ar, ac = a["click"]
            br, bc = b["click"]
            if ah:
                cost[i][j] = 100*abs(ar-br) + abs(ac-bc)
            else:
                cost[i][j] = 100*abs(ac-bc) + abs(ar-br)

    assignment = _hungarian(cost)
    used = set()
    groups = []
    for i, j in enumerate(assignment):
        if i < len(left) and j < len(right) and cost[i][j] < 100000:
            groups.append([left[i], right[j]])
            used.update((left[i], right[j]))
    groups.extend([i] for i in range(len(buttons)) if i not in used)
    return groups


def _partitions(histories, nodes):
    sources = defaultdict(list)
    destinations = defaultdict(list)
    for i, point in enumerate(nodes):
        sources[tuple(before.get(point, 4)
                      for before, after in histories)].append(i)
        destinations[tuple(after.get(point, 4)
                           for before, after in histories)].append(i)
    if sources.keys() != destinations.keys():
        return None
    parts = []
    for signature, src in sources.items():
        dst = destinations[signature]
        if len(src) != len(dst):
            return None
        parts.append((src, dst))
    return parts


def _constraints(groups, histories, nodes):
    records = []
    bad_pairs = []
    for group in groups:
        samples = []
        for direction, button in enumerate(group):
            for before, after in histories[button]:
                samples.append((before, after) if direction == 0
                               else (after, before))
        if not samples:
            records.append((group, [], 0))
            continue
        parts = _partitions(samples, nodes)
        if parts is None and len(group) == 2:
            bad_pairs.append(tuple(group))
            continue
        if parts is None:
            # Defensive recovery from a partially obscured observation.
            for length in (3, 1):
                recent = samples[-length:]
                parts = _partitions(recent, nodes)
                if parts is not None:
                    samples = recent
                    break
        records.append((group, parts or [], len(samples) if parts else 0))
    return records, bad_pairs


def _inverse(permutation):
    result = [0] * len(permutation)
    for i, j in enumerate(permutation):
        result[j] = i
    return result


def _models(records, nodes, tiles, scale, variant):
    n = len(nodes)
    colours = [tiles.get(p, 4) for p in nodes]
    perms = {}
    information = {}
    samples = {}
    unit = float(3*scale)

    for group, parts, count in records:
        if not count:
            continue
        permutation = list(range(n))
        info_f = info_b = 0.0
        for src, dst in parts:
            m = len(src)
            if m == 1:
                permutation[src[0]] = dst[0]
                continue

            cf = Counter(colours[i] for i in src)
            cb = Counter(colours[i] for i in dst)
            info_f += m - max(cf.values())
            info_b += m - max(cb.values())
            if len(cf) > 1:
                info_f += 2 * sum(cf.get(c, 0) for c in (11, 12))
            if len(cb) > 1:
                info_b += 2 * sum(cb.get(c, 0) for c in (11, 12))

            if src == dst:
                continue

            matrix = []
            for i in src:
                r, c = nodes[i]
                row = []
                for j in dst:
                    rr, cc = nodes[j]
                    d2 = ((r-rr)/unit)**2 + ((c-cc)/unit)**2
                    distance = math.sqrt(d2)
                    if variant == 0:
                        value = min(d2, 5.0 + .08*distance)
                    elif variant == 1:
                        value = min(d2, 3.0 + .025*distance)
                    elif variant == 2:
                        value = d2
                    else:
                        value = min(d2, 5.0 + .08*distance)
                        if i != j:
                            value += .45 * (
                                ((i+1)*73 + (j+1)*131 + i*j*17) % 101
                            ) / 101.0
                    value += .00001 * ((i*31 + j*47) % 29)
                    row.append(value)
                matrix.append(row)
            assignment = _hungarian(matrix)
            for k, j in enumerate(assignment):
                permutation[src[k]] = dst[j]

        perms[group[0]] = permutation
        information[group[0]] = info_f
        samples[group[0]] = count
        if len(group) == 2:
            perms[group[1]] = _inverse(permutation)
            information[group[1]] = info_b
            samples[group[1]] = count
    return perms, information, samples


def _advance(state, permutation, n_first):
    n = len(state)
    if n == 1:
        return (permutation[state[0]],)
    if n == 2 and n_first == 1:
        return (permutation[state[0]], permutation[state[1]])
    if n_first == 0 or n_first == n:
        return tuple(sorted(permutation[p] for p in state))
    return (tuple(sorted(permutation[p] for p in state[:n_first])) +
            tuple(sorted(permutation[p] for p in state[n_first:])))


def _is_goal(state, target, n_first, t_first):
    return (set(target[:t_first]).issubset(state[:n_first]) and
            set(target[t_first:]).issubset(state[n_first:]))


def _reconstruct_forward(parents, state):
    path = []
    while parents[state][0] is not None:
        state, action = parents[state]
        path.append(action)
    path.reverse()
    return path


def _search(start, target, n_first, t_first, perms, deadline,
            max_states=100000):
    if _is_goal(start, target, n_first, t_first):
        return [], False

    moves = []
    seen_perms = set()
    for action, permutation in perms.items():
        key = tuple(permutation)
        if key in seen_perms or all(i == j for i, j in enumerate(key)):
            continue
        seen_perms.add(key)
        moves.append((action, permutation))
    if not moves:
        return None, False

    # Cheap reachability test before configuration-space search.
    n = len(moves[0][1])
    roots = list(range(n))

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    for action, permutation in moves:
        for i, j in enumerate(permutation):
            a, b = root(i), root(j)
            if a != b:
                roots[a] = b
    for current, wanted in (
        (start[:n_first], target[:t_first]),
        (start[n_first:], target[t_first:]),
    ):
        have = Counter(root(i) for i in current)
        need = Counter(root(i) for i in wanted)
        if any(have[k] < v for k, v in need.items()):
            return None, False

    exact = len(start) == len(target) and n_first == t_first
    if not exact:
        parents = {start: (None, None)}
        queue = deque([start])
        steps = 0
        while queue:
            state = queue.popleft()
            steps += 1
            if steps % 128 == 0:
                if len(parents) > max_states or time.monotonic() > deadline:
                    return None, True
            for action, permutation in moves:
                nxt = _advance(state, permutation, n_first)
                if nxt in parents:
                    continue
                parents[nxt] = (state, action)
                if _is_goal(nxt, target, n_first, t_first):
                    return _reconstruct_forward(parents, nxt), False
                queue.append(nxt)
        return None, False

    forward = {start: (None, None)}
    backward = {target: (None, None)}
    front_f, front_b = [start], [target]
    reverse_moves = [(a, _inverse(p)) for a, p in moves]
    iterations = 0

    while front_f and front_b:
        expanding_forward = len(front_f) <= len(front_b)
        if expanding_forward:
            frontier, own, other, active = front_f, forward, backward, moves
        else:
            frontier, own, other, active = (
                front_b, backward, forward, reverse_moves
            )
        next_frontier = []
        for state in frontier:
            iterations += 1
            if iterations % 128 == 0:
                if (len(forward) + len(backward) > max_states or
                        time.monotonic() > deadline):
                    return None, True
            for action, permutation in active:
                nxt = _advance(state, permutation, n_first)
                if nxt in own:
                    continue
                own[nxt] = (state, action)
                if nxt in other:
                    path = _reconstruct_forward(forward, nxt)
                    p = nxt
                    while backward[p][0] is not None:
                        p, a = backward[p]
                        path.append(a)
                    return path, False
                next_frontier.append(nxt)
        if expanding_forward:
            front_f = next_frontier
        else:
            front_b = next_frontier
    return None, False


def _astar(start, target, n_first, t_first, perms, deadline):
    if not perms:
        return None
    moves = list(perms.items())
    n = len(moves[0][1])
    reverse_edges = [set() for _ in range(n)]
    for action, permutation in moves:
        for i, j in enumerate(permutation):
            reverse_edges[j].add(i)

    distances = {}
    for goal in target:
        if goal in distances:
            continue
        d = [100000] * n
        d[goal] = 0
        q = deque([goal])
        while q:
            p = q.popleft()
            for previous in reverse_edges[p]:
                if d[previous] == 100000:
                    d[previous] = d[p] + 1
                    q.append(previous)
        distances[goal] = d

    def heuristic(state):
        result = 0
        for current, wanted in (
            (state[:n_first], target[:t_first]),
            (state[n_first:], target[t_first:]),
        ):
            if not wanted:
                continue
            if len(current) < len(wanted):
                return 100000
            matrix = [[distances[g][p] for p in current] for g in wanted]
            h = max(min(row) for row in matrix)
            if len(current) == len(wanted):
                h = max(h, max(min(row[i] for row in matrix)
                               for i in range(len(current))))
                if len(current) == 2:
                    h = min(max(matrix[0][0], matrix[1][1]),
                            max(matrix[0][1], matrix[1][0]))
            result = max(result, h)
        return result

    parents = {start: (None, None)}
    best = {start: 0}
    serial = 0
    heap = [(1.7*heuristic(start), 0, serial, start)]
    iterations = 0
    while heap:
        f, g, _, state = heapq.heappop(heap)
        if best.get(state) != g:
            continue
        if _is_goal(state, target, n_first, t_first):
            return _reconstruct_forward(parents, state)
        iterations += 1
        if iterations % 64 == 0:
            if time.monotonic() > deadline or len(best) > 100000:
                return None
        for action, permutation in moves:
            nxt = _advance(state, permutation, n_first)
            ng = g + 1
            if ng >= best.get(nxt, 100000):
                continue
            h = heuristic(nxt)
            if h >= 100000:
                continue
            best[nxt] = ng
            parents[nxt] = (state, action)
            serial += 1
            heapq.heappush(heap, (ng + 1.7*h, ng, serial, nxt))
    return None


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 8.3

    if "ACTION6" not in available:
        action = "RESET" if "RESET" in available else available[0]
        return action, None, None

    scale, tiles, targets, buttons = _read(obs)
    if not buttons:
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 32, 32

    fingerprint = (
        scale,
        tuple((b["colour"], b["r"], b["c"], b["h"], b["w"])
              for b in buttons),
        tuple(targets[11]), tuple(targets[12]),
    )
    saved = memory.get("_lp85")
    if saved is None or saved.get("fingerprint") != fingerprint:
        saved = {
            "fingerprint": fingerprint,
            "groups": _pair_buttons(buttons),
            "histories": [[] for _ in buttons],
            "nodes": set(),
            "previous": None,
            "last": None,
            "uses": {},
            "goal_visits": {},
            "variant": 0,
        }
        memory["_lp85"] = saved

    previous = saved["previous"]
    if previous is not None:
        button, before = previous
        saved["histories"][button].append((before, tiles.copy()))
    saved["previous"] = None

    saved["nodes"].update(tiles)
    saved["nodes"].update(targets[11])
    saved["nodes"].update(targets[12])
    nodes = sorted(saved["nodes"])
    index = {p: i for i, p in enumerate(nodes)}

    first = sorted(index[p] for p, c in tiles.items() if c == 11)
    second = sorted(index[p] for p, c in tiles.items() if c == 12)
    n_first = len(first)
    state = tuple(first + second)
    target = tuple(sorted(index[p] for p in targets[11]) +
                   sorted(index[p] for p in targets[12]))
    t_first = len(targets[11])

    board_key = tuple(sorted(tiles.items()))

    def goal_key(st):
        return (n_first, tuple(nodes[i] for i in st))

    current_key = goal_key(state)
    saved["goal_visits"][current_key] = (
        saved["goal_visits"].get(current_key, 0) + 1
    )

    records, bad = _constraints(
        saved["groups"], saved["histories"], nodes
    )
    if bad:
        bad = set(bad)
        groups = []
        for group in saved["groups"]:
            if tuple(group) in bad:
                groups.extend([[b] for b in group])
            else:
                groups.append(group)
        saved["groups"] = groups
        records, _ = _constraints(groups, saved["histories"], nodes)

    variant = saved["variant"]
    perms, information, sample_counts = _models(
        records, nodes, tiles, scale, variant
    )
    unknown = [group for group, parts, count in records if count == 0]

    def emit(button):
        saved["previous"] = (button, tiles.copy())
        saved["last"] = button
        key = (board_key, button)
        saved["uses"][key] = saved["uses"].get(key, 0) + 1
        r, c = buttons[button]["click"]
        return "ACTION6", int(r), int(c)

    path = None
    if target and state and perms:
        path, limited = _search(
            state, target, n_first, t_first, perms,
            min(deadline, time.monotonic() + 1.9)
        )
        if limited and not unknown and time.monotonic() < deadline - .3:
            path = _astar(
                state, target, n_first, t_first, perms,
                min(deadline, time.monotonic() + 1.5)
            )

    if path is not None and (not unknown or len(path) <= 4):
        if path:
            return emit(path[0])
        for button, permutation in perms.items():
            nxt = _advance(state, permutation, n_first)
            if _is_goal(nxt, target, n_first, t_first):
                return emit(button)

    # Learn previously unseen controls before committing to a long route.
    if unknown:
        goals = [p for p, c in tiles.items() if c in (11, 12)]
        wanted = [p for c in (11, 12) for p in targets[c]
                  if tiles.get(p) != c]
        if not wanted:
            wanted = targets[11] + targets[12]

        def relevance(group):
            centres = [buttons[b]["click"] for b in group]

            def nearest(points):
                if not points:
                    return 0
                return min(abs(r-y) + abs(c-x)
                           for r, c in centres for y, x in points)

            return nearest(goals) + nearest(wanted)

        group = min(unknown, key=relevance)
        return emit(group[0])

    # Different spatial tie-breaks provide alternative hypotheses for
    # edges that the observations do not yet uniquely identify.
    if path is None and state and target:
        for other_variant in (0, 1, 2, 3):
            if other_variant == variant:
                continue
            if time.monotonic() > deadline - .35:
                break
            alternate, _, _ = _models(
                records, nodes, tiles, scale, other_variant
            )
            candidate, limited = _search(
                state, target, n_first, t_first, alternate,
                min(deadline - .1, time.monotonic() + .9),
                max_states=60000,
            )
            if candidate is not None:
                saved["variant"] = other_variant
                perms = alternate
                path = candidate
                break

    if path:
        return emit(path[0])

    # Active exploration: prefer presses that distinguish unresolved
    # colour signatures, then presses reaching less-visited goal states.
    last = saved["last"]
    inverse_of_last = None
    for group in saved["groups"]:
        if len(group) == 2 and last in group:
            inverse_of_last = group[1] if last == group[0] else group[0]

    best_button = 0
    best_score = -float("inf")
    for button in range(len(buttons)):
        permutation = perms.get(button)
        score = 4.0 * information.get(button, 0.0)
        score += 1.0 / (1 + sample_counts.get(button, 0))
        score -= 3.0 * saved["uses"].get((board_key, button), 0)

        if permutation is not None:
            nxt = _advance(state, permutation, n_first)
            visits = saved["goal_visits"].get(goal_key(nxt), 0)
            score += 4.0 / (1 + visits)
            score += .15 * sum(a != b for a, b in zip(state, nxt))
            if nxt == state:
                score -= .6
        else:
            score += 2.0

        if button == inverse_of_last:
            score -= 1.0
        if button == last:
            score += .15

        if score > best_score:
            best_score = score
            best_button = button

    return emit(best_button)

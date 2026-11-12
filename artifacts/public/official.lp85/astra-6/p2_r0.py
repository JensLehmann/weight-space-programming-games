import math
import time
from collections import defaultdict, deque, Counter
import heapq
import numpy as np


def _components(obs):
    """Monochromatic, four-connected components, excluding the UI."""
    seen = np.zeros((64, 64), dtype=np.bool_)
    result = []

    for y in range(64):
        if y == 1:
            continue
        for x in range(1, 64):
            c = int(obs[y, x])
            if seen[y, x] or c in (3, 4):
                continue

            seen[y, x] = True
            stack = [(y, x)]
            pixels = []
            ymin = ymax = y
            xmin = xmax = x

            while stack:
                yy, xx = stack.pop()
                pixels.append((yy, xx))
                ymin = min(ymin, yy)
                ymax = max(ymax, yy)
                xmin = min(xmin, xx)
                xmax = max(xmax, xx)

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < 64 and ny != 1 and 1 <= nx < 64
                            and not seen[ny, nx]
                            and int(obs[ny, nx]) == c):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            result.append((c, ymin, xmin, ymax-ymin+1,
                           xmax-xmin+1, len(pixels)))
    return result


def _parse(obs):
    comps = _components(obs)

    sizes = Counter()
    for c, y, x, h, w, area in comps:
        if c in (1, 2, 9, 10, 15) and h == w and area == h*w and h % 2 == 0:
            sizes[h // 2] += 1

    if sizes:
        scale = sizes.most_common(1)[0][0]
    else:
        candidates = []
        for c, y, x, h, w, area in comps:
            if c in (8, 14):
                s = min(h, w) // 3
                if s and sorted((h, w)) == [3*s, 4*s]:
                    candidates.append(s)
        scale = Counter(candidates).most_common(1)[0][0] if candidates else 1

    tiles = {}
    corner_blocks = set()
    buttons = []

    for c, y, x, h, w, area in comps:
        if (h == w == 2*scale and area == h*w
                and c not in (3, 4, 8, 14)):
            tiles[(y, x)] = c

        if c in (11, 12) and h == w == scale and area == scale*scale:
            corner_blocks.add((c, y, x))

        if c in (8, 14):
            normal = (sorted((h, w)) == [3*scale, 4*scale])
            if normal or (
                area >= 7*scale*scale
                and 2*scale <= min(h, w)
                and max(h, w) <= 5*scale
            ):
                # row, column, colour, orientation
                buttons.append((y + h//2, x + w//2, c, int(w > h)))

    targets = set()
    d = 3*scale
    for c, y, x in corner_blocks:
        if ((c, y+d, x) in corner_blocks
                and (c, y, x+d) in corner_blocks
                and (c, y+d, x+d) in corner_blocks):
            targets.add((y+scale, x+scale, c))

    return scale, tiles, sorted(targets), sorted(set(buttons))


def _hungarian(cost):
    """Minimum-cost assignment for a square matrix."""
    n = len(cost)
    if n == 0:
        return []
    if n == 1:
        return [0]

    u = [0.0]*(n+1)
    v = [0.0]*(n+1)
    p = [0]*(n+1)
    way = [0]*(n+1)

    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minv = [float("inf")]*(n+1)
        used = [False]*(n+1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row = cost[i0-1]

            for j in range(1, n+1):
                if not used[j]:
                    cur = row[j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n+1):
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

    answer = [0]*n
    for j in range(1, n+1):
        answer[p[j]-1] = j-1
    return answer


def _pair_buttons(buttons, scale):
    left = [i for i, b in enumerate(buttons) if b[2] == 8]
    right = [i for i, b in enumerate(buttons) if b[2] == 14]
    if not left or not right:
        return [[i] for i in range(len(buttons))]

    n = max(len(left), len(right))
    costs = [[500.0]*n for _ in range(n)]

    for i, a in enumerate(left):
        ay, ax, _, orientation = buttons[a]
        for j, b in enumerate(right):
            by, bx, _, other_orientation = buttons[b]
            if orientation != other_orientation:
                costs[i][j] = 10000.0
            else:
                perpendicular = abs(ax-bx) if orientation else abs(ay-by)
                longitudinal = abs(ay-by) if orientation else abs(ax-bx)
                costs[i][j] = 100*perpendicular + longitudinal

    assignment = _hungarian(costs)
    groups = []
    used = set()

    for i, a in enumerate(left):
        j = assignment[i]
        if j >= len(right):
            continue
        b = right[j]
        ay, ax, _, orientation = buttons[a]
        by, bx, _, other_orientation = buttons[b]
        perpendicular = abs(ax-bx) if orientation else abs(ay-by)
        if orientation == other_orientation and perpendicular <= 2*scale:
            groups.append([a, b])
            used.add(a)
            used.add(b)

    groups.extend([i] for i in range(len(buttons)) if i not in used)
    return groups


def _geometry_cost(positions, scale):
    present = set(positions)
    step = 3*scale
    matrix = []

    for y, x in positions:
        row = []
        for yy, xx in positions:
            dy, dx = abs(y-yy), abs(x-xx)
            if dx == dy == 0:
                value = 0.0
            elif dx+dy == step:
                value = 1.0
            elif dx == dy == step and (y, xx) not in present and (yy, x) not in present:
                value = 1.4
            else:
                value = 100.0 + 0.2*math.hypot(dx, dy)/step
            row.append(value)
        matrix.append(row)
    return matrix


def _fit_group(group, records, positions, geometry):
    """
    Equal temporal colour signatures identify possible source/destination
    matches. Geometry breaks remaining ties.
    """
    examples = []
    for action, before, after in records:
        if action == group[0]:
            examples.append((before, after))
        elif len(group) == 2 and action == group[1]:
            examples.append((after, before))

    if not examples:
        return None, 0

    source_classes = defaultdict(list)
    destination_classes = defaultdict(list)

    for i, position in enumerate(positions):
        source_classes[tuple(a.get(position, -1) for a, b in examples)].append(i)
        destination_classes[tuple(b.get(position, -1) for a, b in examples)].append(i)

    if set(source_classes) != set(destination_classes):
        return False, len(examples)

    permutation = list(range(len(positions)))
    for signature, sources in source_classes.items():
        destinations = destination_classes[signature]
        if len(sources) != len(destinations):
            return False, len(examples)

        if len(sources) == 1:
            permutation[sources[0]] = destinations[0]
        else:
            costs = [[geometry[i][j] for j in destinations] for i in sources]
            matching = _hungarian(costs)
            for k, i in enumerate(sources):
                permutation[i] = destinations[matching[k]]

    return tuple(permutation), len(examples)


def _inverse(permutation):
    inv = [0]*len(permutation)
    for i, j in enumerate(permutation):
        inv[j] = i
    return tuple(inv)


def _move(state, permutation, split):
    if split <= 1 and len(state)-split <= 1:
        return tuple(permutation[i] for i in state)
    return (tuple(sorted(permutation[i] for i in state[:split]))
            + tuple(sorted(permutation[i] for i in state[split:])))


def _reconstruct_forward(parents, state):
    path = []
    while parents[state][0] is not None:
        previous, action = parents[state]
        path.append(action)
        state = previous
    path.reverse()
    return path


def _search(start, target, split, controls, deadline, limit=180000):
    if start == target:
        return [], True
    if not controls:
        return None, True

    # Collapse visually distinct buttons with the same predicted effect.
    unique = {}
    for action, permutation in controls:
        if any(i != j for i, j in enumerate(permutation)):
            unique.setdefault(permutation, action)
    controls = [(action, p) for p, action in unique.items()]
    if not controls:
        return None, True

    forward = {start: (None, None)}
    backward = {target: (None, None)}
    front = [start]
    back = [target]
    reverse_controls = [(a, _inverse(p)) for a, p in controls]
    expanded = 0

    while front and back:
        use_forward = len(front) <= len(back)
        layer = front if use_forward else back
        own = forward if use_forward else backward
        other = backward if use_forward else forward
        moves = controls if use_forward else reverse_controls
        next_layer = []

        for state in layer:
            expanded += 1
            if expanded % 256 == 0:
                if time.perf_counter() > deadline or len(forward)+len(backward) > limit:
                    return None, False

            for action, permutation in moves:
                nxt = _move(state, permutation, split)
                if nxt in own:
                    continue
                own[nxt] = (state, action)

                if nxt in other:
                    path = _reconstruct_forward(forward, nxt)
                    cursor = nxt
                    while backward[cursor][0] is not None:
                        cursor, a = backward[cursor]
                        path.append(a)
                    return path, True
                next_layer.append(nxt)

        if use_forward:
            front = next_layer
        else:
            back = next_layer

    return None, True


def _guided_search(start, target, split, controls, deadline):
    """A bounded weighted search for unusually large goal-state spaces."""
    if not controls or not start or len(start) != len(target):
        return None

    n = len(controls[0][1])
    reverse = [_inverse(p) for a, p in controls]
    distances = {}

    for destination in set(target):
        dist = [10000]*n
        dist[destination] = 0
        q = deque([destination])
        while q:
            node = q.popleft()
            nd = dist[node]+1
            for permutation in reverse:
                nxt = permutation[node]
                if dist[nxt] > nd:
                    dist[nxt] = nd
                    q.append(nxt)
        distances[destination] = dist

    group_cache = {}

    def group_bound(tokens, goals):
        key = (tokens, goals)
        if key in group_cache:
            return group_cache[key]
        k = len(tokens)
        if not k:
            return 0
        if k <= 6:
            dp = {0: 0}
            for token in tokens:
                ndp = {}
                for mask, old in dp.items():
                    for j, goal in enumerate(goals):
                        if not (mask >> j) & 1:
                            nm = mask | (1 << j)
                            value = max(old, distances[goal][token])
                            if value < ndp.get(nm, 100000):
                                ndp[nm] = value
                dp = ndp
            answer = dp.get((1 << k)-1, 10000)
        else:
            answer = max(
                max(min(distances[g][t] for g in goals) for t in tokens),
                max(min(distances[g][t] for t in tokens) for g in goals),
            )
        group_cache[key] = answer
        return answer

    def heuristic(state):
        return max(group_bound(state[:split], target[:split]),
                   group_bound(state[split:], target[split:]))

    h0 = heuristic(start)
    if h0 >= 10000:
        return None

    parents = {start: (None, None)}
    costs = {start: 0}
    heap = [(3*h0, h0, 0, start)]
    best = start
    best_key = (h0, 0)
    iterations = 0

    while heap:
        _, h, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        if state == target:
            return _reconstruct_forward(parents, state)

        iterations += 1
        if iterations % 128 == 0:
            if time.perf_counter() > deadline or len(costs) > 140000:
                break

        for action, permutation in controls:
            nxt = _move(state, permutation, split)
            ng = g+1
            if nxt == state or ng >= costs.get(nxt, 1000000):
                continue
            nh = heuristic(nxt)
            if nh >= 10000:
                continue
            costs[nxt] = ng
            parents[nxt] = (state, action)
            heapq.heappush(heap, (ng+3*nh, nh, ng, nxt))
            if (nh, ng) < best_key:
                best_key = (nh, ng)
                best = nxt

    if best != start:
        return _reconstruct_forward(parents, best)
    return None


def act(obs, available, memory):
    started = time.perf_counter()
    scale, tiles, targets, buttons = _parse(obs)

    if "ACTION6" not in available:
        action = available[0] if available else "RESET"
        return action, None, None

    if not buttons:
        return "ACTION6", 32, 32

    signature = (scale, tuple(buttons))
    m = memory.get("_lp85")
    if m is None or m.get("signature") != signature:
        m = {
            "signature": signature,
            "groups": _pair_buttons(buttons, scale),
            "records": [],
            "positions": set(),
            "previous": None,
            "bar_changes": [],
            "resets": 0,
            "calls": 0,
        }
        memory["_lp85"] = m

    used_bar = int(np.count_nonzero(obs[:, 0] == 5))

    previous = m.get("previous")
    if previous is not None:
        before, action, old_bar = previous
        m["records"].append((action, before, dict(tiles)))
        change = used_bar-old_bar
        if 0 <= change <= 16:
            m["bar_changes"].append(change)
            m["bar_changes"] = m["bar_changes"][-12:]
        m["previous"] = None

    m["positions"].update(tiles)
    m["positions"].update((y, x) for y, x, c in targets)
    positions = sorted(m["positions"])
    index = {p: i for i, p in enumerate(positions)}
    geometry = _geometry_cost(positions, scale)

    maps = {}
    group_info = []
    updated_groups = []

    for group in m["groups"]:
        permutation, count = _fit_group(
            group, m["records"], positions, geometry
        )

        if permutation is False and len(group) == 2:
            # Overlapping compound controls need not be exact inverses.
            for action in group:
                singleton = [action]
                p, k = _fit_group(
                    singleton, m["records"], positions, geometry
                )
                updated_groups.append(singleton)
                group_info.append((singleton, k))
                if p is not None and p is not False:
                    maps[action] = p
        else:
            updated_groups.append(group)
            group_info.append((group, count))
            if permutation is not None and permutation is not False:
                maps[group[0]] = permutation
                if len(group) == 2:
                    maps[group[1]] = _inverse(permutation)

    m["groups"] = updated_groups

    yellow = tuple(sorted(index[p] for p, c in tiles.items() if c == 11))
    orange = tuple(sorted(index[p] for p, c in tiles.items() if c == 12))
    target_yellow = tuple(sorted(index[(y, x)] for y, x, c in targets if c == 11))
    target_orange = tuple(sorted(index[(y, x)] for y, x, c in targets if c == 12))
    state = yellow + orange
    target = target_yellow + target_orange
    split = len(yellow)

    controls = sorted(maps.items())
    plan = None
    exhaustive = True

    if (len(yellow) == len(target_yellow)
            and len(orange) == len(target_orange)
            and targets):
        plan, exhaustive = _search(
            state, target, split, controls, started+3.7
        )

    unseen = [(group, count) for group, count in group_info if count == 0]

    misplaced = [
        (y, x) for y, x, c in targets
        if tiles.get((y, x)) != c
    ]
    misplaced += [
        p for p, c in tiles.items()
        if c in (11, 12) and (p[0], p[1], c) not in targets
    ]
    if not misplaced:
        misplaced = [(y, x) for y, x, c in targets] or list(tiles)

    def proximity(group):
        if not misplaced:
            return 0
        return min(
            (buttons[a][0]-y)**2 + (buttons[a][1]-x)**2
            for a in group for y, x in misplaced
        )

    chosen = None

    # Try each independent control before committing to a long, possibly
    # unnecessarily restricted route.
    if unseen and (plan is None or len(plan) > 2):
        group, _ = min(unseen, key=lambda item: proximity(item[0]))
        chosen = group[0]
    elif plan:
        chosen = plan[0]
    elif plan == []:
        # Winning is checked following a valid button click.
        for action, permutation in controls:
            if _move(state, permutation, split) == state:
                chosen = action
                break
        if chosen is None:
            chosen = 0
    elif not exhaustive and time.perf_counter() < started+6.7:
        guided = _guided_search(
            state, target, split, controls, started+7.5
        )
        if guided:
            plan = guided
            chosen = guided[0]

    if chosen is None:
        # Repeated observations separate equal-coloured tiles. Avoid
        # immediately undoing a probe, which would reveal no new information.
        group, count = min(
            group_info,
            key=lambda item: (item[1], proximity(item[0]))
        )
        chosen = group[0]
        for old_action, before, after in reversed(m["records"]):
            if old_action in group:
                chosen = old_action
                break

    # Preserve learned controls if a level reset is needed to replenish the
    # simulator's step counter.
    changes = m["bar_changes"]
    mean_change = sum(changes)/len(changes) if changes else 0.8
    if mean_change <= 0.01:
        mean_change = 0.8
    estimated_remaining = (64-used_bar)/mean_change
    needed = len(plan) if plan else 2

    if ("RESET" in available and m["resets"] == 0 and used_bar >= 54
            and needed > estimated_remaining+0.6):
        m["resets"] += 1
        m["previous"] = None
        m["bar_changes"] = []
        m["calls"] += 1
        return "RESET", None, None

    m["previous"] = (dict(tiles), chosen, used_bar)
    m["calls"] += 1
    row, col = buttons[chosen][:2]
    return "ACTION6", int(row), int(col)

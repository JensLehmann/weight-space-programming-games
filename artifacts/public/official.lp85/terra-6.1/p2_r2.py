import heapq
import itertools
from collections import deque


def _components(a, colors, ignore_ui=False):
    h, w = a.shape
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if (y, x) in seen or int(a[y, x]) not in colors:
                continue
            if ignore_ui and (x == 0 or y == 1):
                continue
            color = int(a[y, x])
            q = [(y, x)]
            seen.add((y, x))
            pts = []
            while q:
                yy, xx = q.pop()
                pts.append((yy, xx))
                for ny, nx in ((yy - 1, xx), (yy + 1, xx),
                               (yy, xx - 1), (yy, xx + 1)):
                    if (0 <= ny < h and 0 <= nx < w
                            and (ny, nx) not in seen
                            and not (ignore_ui and (nx == 0 or ny == 1))
                            and int(a[ny, nx]) == color):
                        seen.add((ny, nx))
                        q.append((ny, nx))
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            result.append((color, min(ys), min(xs),
                           max(ys), max(xs), len(pts)))
    return result


def _hungarian(cost):
    """Minimum-cost assignment for a square matrix."""
    n = len(cost)
    if not n:
        return []
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    inf = 10**12
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
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
    ans = [0] * n
    for j in range(1, n + 1):
        ans[p[j] - 1] = j - 1
    return ans


def _calibrate(obs):
    ys, xs = (obs == 4).nonzero()
    if len(xs):
        xp = min(int(xs.min()), 63 - int(xs.max()))
        yp = min(int(ys.min()), 63 - int(ys.max()))
    else:
        xp = yp = 0

    small = large = 0
    for _, y0, x0, y1, x1, count in _components(
            obs, {1, 2, 9, 10, 15}, True):
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if h == w == 2 and count == 4:
            small += 1
        elif h == w == 4 and count == 16:
            large += 1
    scale = 2 if large > small else 1
    return xp, yp, scale


def _buttons(obs, scale):
    buttons = []
    for color, y0, x0, y1, x1, count in _components(
            obs, {8, 14}, True):
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if ((h, w) not in ((4 * scale, 3 * scale),
                           (3 * scale, 4 * scale))
                or count != 10 * scale * scale):
            continue
        buttons.append([color, (y0 + y1) // 2, (x0 + x1) // 2])

    buttons.sort(key=lambda b: (b[1], b[2], b[0]))
    left = [i for i, b in enumerate(buttons) if b[0] == 8]
    right = [i for i, b in enumerate(buttons) if b[0] == 14]
    pairs = [-1] * len(buttons)
    if left and len(left) == len(right):
        cost = []
        for i in left:
            row = []
            for j in right:
                dy = buttons[i][1] - buttons[j][1]
                dx = buttons[i][2] - buttons[j][2]
                row.append(dy * dy + dx * dx)
            cost.append(row)
        assignment = _hungarian(cost)
        for k, t in enumerate(assignment):
            i, j = left[k], right[t]
            pairs[i] = j
            pairs[j] = i
    return buttons, pairs


def _read_board(obs, xp, yp, scale):
    """Return visible lattice tiles and the fixed target-frame positions."""
    tiles = {}
    markers = {11: [], 12: []}
    step = 3 * scale
    for gy in range(23):
        y = yp + gy * step
        if y + 2 * scale > 64 - yp:
            break
        for gx in range(23):
            x = xp + gx * step
            if x + 2 * scale > 64 - xp:
                break
            c = int(obs[y, x])
            if c not in (4, 8, 14):
                good = True
                for yy in range(y, y + 2 * scale):
                    for xx in range(x, x + 2 * scale):
                        if int(obs[yy, xx]) != c:
                            good = False
                            break
                    if not good:
                        break
                if good:
                    tiles[(gx, gy)] = c

            # Frames have four isolated pixels around a 2x2 tile.
            x0, y0 = x - scale, y - scale
            x1, y1 = x + 2 * scale, y + 2 * scale
            if x0 >= 0 and y0 >= 0 and x1 < 64 and y1 < 64:
                c0 = int(obs[y0, x0])
                if (c0 in (11, 12)
                        and int(obs[y0, x1]) == c0
                        and int(obs[y1, x0]) == c0
                        and int(obs[y1, x1]) == c0):
                    markers[c0].append((gx, gy))
    return tiles, markers


def _make_model(n, positions, samples):
    if not samples:
        return list(range(n)), [n] * n

    src = [a for a, _ in samples]
    dst = [b for _, b in samples]
    sig_src = [tuple(s[i] for s in src) for i in range(n)]
    sig_dst = [tuple(s[i] for s in dst) for i in range(n)]
    groups = {}
    for j, sig in enumerate(sig_dst):
        groups.setdefault(sig, []).append(j)

    huge = 1000000
    cost = []
    counts = []
    for i in range(n):
        candidates = groups.get(sig_src[i], [])
        counts.append(len(candidates))
        allowed = set(candidates)
        xi, yi = positions[i]
        row = []
        for j in range(n):
            if j not in allowed:
                row.append(huge)
            elif i == j:
                row.append(0)
            else:
                xj, yj = positions[j]
                row.append(2 + abs(xi - xj) + abs(yi - yj))
        cost.append(row)

    perm = _hungarian(cost)
    if any(cost[i][perm[i]] >= huge for i in range(n)):
        return None, None
    return perm, counts


def _goal_distance(state, target, positions):
    total = 0
    for current, desired in zip(state, target):
        if not current or not desired:
            if len(current) != len(desired):
                total += 20 * abs(len(current) - len(desired))
            continue
        if len(current) <= 5 and len(desired) <= 5:
            if len(current) <= len(desired):
                best = 10**9
                for chosen in itertools.permutations(desired, len(current)):
                    d = 0
                    for i, j in zip(current, chosen):
                        x, y = positions[i]
                        xx, yy = positions[j]
                        d += abs(x - xx) + abs(y - yy)
                    best = min(best, d)
            else:
                best = 10**9
                for chosen in itertools.permutations(current, len(desired)):
                    d = 0
                    for i, j in zip(chosen, desired):
                        x, y = positions[i]
                        xx, yy = positions[j]
                        d += abs(x - xx) + abs(y - yy)
                    best = min(best, d)
                best += 20 * (len(current) - len(desired))
        else:
            best = 0
            for i in current:
                x, y = positions[i]
                best += min(abs(x - positions[j][0]) +
                            abs(y - positions[j][1]) for j in desired)
        total += best
    return total


def _choose_action(state, target, positions, models, ambiguity, tried):
    n_actions = len(models)
    if not n_actions:
        return 0

    cache = {}

    def heuristic(s):
        if s not in cache:
            cache[s] = _goal_distance(s, target, positions)
        return cache[s]

    def moved(s, perm):
        return tuple(tuple(sorted(perm[i] for i in group)) for group in s)

    def uncertain(s, a):
        return any(ambiguity[a][i] > 1 for group in s for i in group)

    # Each first action has its own search, allowing penalties for moves
    # already tried at this exact observed board.
    heap = []
    serial = 0
    seen = {}
    best_score = float("inf")
    best_action = 0

    for a in range(n_actions):
        ns = moved(state, models[a])
        repeat = tried[a] if a < len(tried) else 0
        penalty = 2.5 * repeat
        score = heuristic(ns) + 0.35 + penalty
        if uncertain(state, a):
            score -= 0.6
        if score < best_score:
            best_score, best_action = score, a
        if ns != state:
            key = (ns, a)
            seen[key] = 1
            serial += 1
            heapq.heappush(heap, (heuristic(ns) + 0.35 + penalty,
                                  serial, 1, a, ns))

    expanded = 0
    while heap and expanded < 6000:
        _, _, depth, first, cur = heapq.heappop(heap)
        if seen.get((cur, first)) != depth:
            continue
        expanded += 1
        repeat = tried[first] if first < len(tried) else 0
        penalty = 2.5 * repeat
        h = heuristic(cur)
        if h == 0:
            return first
        if depth >= 14:
            continue
        for a in range(n_actions):
            ns = moved(cur, models[a])
            nd = depth + 1
            score = heuristic(ns) + 0.35 * nd + penalty
            if uncertain(cur, a):
                score -= 0.6
            if score < best_score:
                best_score, best_action = score, first
            if ns == cur:
                continue
            key = (ns, first)
            if nd >= seen.get(key, 10**9):
                continue
            seen[key] = nd
            serial += 1
            heapq.heappush(heap, (heuristic(ns) + 0.35 * nd + penalty,
                                  serial, nd, first, ns))
    return best_action


def act(obs, available, memory):
    if "ACTION6" not in available:
        if "RESET" in available:
            return ("RESET", None, None)
        return (available[0], None, None)

    if "cal" not in memory:
        xp, yp, scale = _calibrate(obs)
        buttons, pairs = _buttons(obs, scale)
        tiles, markers = _read_board(obs, xp, yp, scale)
        positions = sorted(set(tiles) | set(markers[11]) | set(markers[12]))
        memory["cal"] = [xp, yp, scale]
        memory["buttons"] = buttons
        memory["pairs"] = pairs
        memory["positions"] = [list(p) for p in positions]
        memory["markers"] = {
            "11": [list(p) for p in markers[11]],
            "12": [list(p) for p in markers[12]],
        }
        memory["samples"] = []
        memory["visits"] = {}

    xp, yp, scale = memory["cal"]
    buttons = memory["buttons"]
    pairs = memory["pairs"]
    tiles, _ = _read_board(obs, xp, yp, scale)

    if not buttons:
        buttons, pairs = _buttons(obs, scale)
        memory["buttons"], memory["pairs"] = buttons, pairs
    if not buttons:
        return ("ACTION6", 32, 32)

    positions = [tuple(p) for p in memory["positions"]]
    known = set(positions)
    new_positions = sorted(set(tiles) - known)
    if new_positions:
        positions.extend(new_positions)
        memory["positions"] = [list(p) for p in positions]
        suffix = "4" * len(new_positions)
        if "previous" in memory:
            memory["previous"] += suffix
        for sample in memory["samples"]:
            sample[1] += suffix
            sample[2] += suffix

    board = "".join(format(tiles.get(p, 4), "x") for p in positions)
    if "previous" in memory and "last_action" in memory:
        memory["samples"].append(
            [memory["last_action"], memory["previous"], board])
        memory["samples"] = memory["samples"][-48:]

    n = len(positions)
    index = {p: i for i, p in enumerate(positions)}
    target = []
    for color in (11, 12):
        target.append(tuple(sorted(
            index[tuple(p)] for p in memory["markers"][str(color)]
            if tuple(p) in index)))
    target = tuple(target)
    state = tuple(tuple(i for i, c in enumerate(board)
                        if c == format(color, "x"))
                  for color in (11, 12))

    models = []
    ambiguity = []
    for a in range(len(buttons)):
        direct = [(before, after)
                  for action, before, after in memory["samples"]
                  if action == a]
        inverse = []
        if a < len(pairs) and pairs[a] >= 0:
            inverse = [(after, before)
                       for action, before, after in memory["samples"]
                       if action == pairs[a]]
        perm, counts = _make_model(n, positions, direct + inverse)
        if perm is None:
            perm, counts = _make_model(n, positions, direct)
        if perm is None:
            perm, counts = list(range(n)), [n] * n
        models.append(perm)
        ambiguity.append(counts)

    visits = memory["visits"]
    if board not in visits:
        if len(visits) >= 100:
            visits.clear()
        visits[board] = [0] * len(buttons)
    tried = visits[board]
    if len(tried) != len(buttons):
        tried = [0] * len(buttons)
        visits[board] = tried

    chosen = _choose_action(state, target, positions, models, ambiguity, tried)
    tried[chosen] += 1
    memory["previous"] = board
    memory["last_action"] = chosen
    _, row, col = buttons[chosen]
    return ("ACTION6", int(row), int(col))

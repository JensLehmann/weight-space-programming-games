import math
import time
from collections import Counter, deque


def _components(obs):
    """Components of non-background colours, excluding the fixed UI."""
    h, w = obs.shape
    seen = set()
    result = []
    excluded = {3, 4}

    for y in range(h):
        for x in range(1, w):
            if y == 1 or (y, x) in seen:
                continue
            colour = int(obs[y, x])
            if colour in excluded:
                continue

            seen.add((y, x))
            stack = [(y, x)]
            points = []
            while stack:
                yy, xx = stack.pop()
                points.append((yy, xx))
                for ny, nx in ((yy - 1, xx), (yy + 1, xx),
                               (yy, xx - 1), (yy, xx + 1)):
                    if (0 <= ny < h and 1 <= nx < w and ny != 1
                            and (ny, nx) not in seen
                            and int(obs[ny, nx]) == colour):
                        seen.add((ny, nx))
                        stack.append((ny, nx))

            ys = [p[0] for p in points]
            xs = [p[1] for p in points]
            result.append((colour, min(xs), min(ys),
                           max(xs) + 1, max(ys) + 1, len(points)))
    return result


def _read_image(obs):
    comps = _components(obs)

    scales = Counter()
    for c, x0, y0, x1, y1, area in comps:
        side = x1 - x0
        if (c not in (8, 14) and side == y1 - y0
                and side % 2 == 0 and area == side * side):
            scales[side // 2] += 1
    scale = scales.most_common(1)[0][0] if scales else 1
    scale = max(1, min(4, scale))

    ys, xs = (obs == 4).nonzero()
    if len(xs):
        ox, oy = int(xs.min()), int(ys.min())
        ex, ey = int(xs.max()) + 1, int(ys.max()) + 1
        if ox == 1:
            ox = 0
    else:
        ox = oy = 0
        ex = ey = 64

    width = (ex - ox) // scale
    height = (ey - oy) // scale
    grid = obs[oy:oy + height * scale:scale,
               ox:ox + width * scale:scale]

    tiles = {}
    targets = {11: set(), 12: set()}
    for y in range(0, height - 1, 3):
        for x in range(0, width - 1, 3):
            c = int(grid[y, x])
            if (c not in (3, 4, 8, 14)
                    and int(grid[y, x + 1]) == c
                    and int(grid[y + 1, x]) == c
                    and int(grid[y + 1, x + 1]) == c):
                tiles[(x, y)] = c

            if x >= 1 and y >= 1 and x + 2 < width and y + 2 < height:
                c = int(grid[y - 1, x - 1])
                if (c in targets
                        and int(grid[y - 1, x + 2]) == c
                        and int(grid[y + 2, x - 1]) == c
                        and int(grid[y + 2, x + 2]) == c):
                    targets[c].add((x, y))

    buttons = []
    for c, x0, y0, x1, y1, area in comps:
        if c not in (8, 14):
            continue
        bw, bh = x1 - x0, y1 - y0
        if sorted((bw, bh)) != [3 * scale, 4 * scale]:
            continue
        buttons.append({
            "colour": c,
            "row": (y0 + y1 - 1) // 2,
            "col": (x0 + x1 - 1) // 2,
            "x": x0,
            "y": y0,
            "vertical": bw > bh,
        })
    buttons.sort(key=lambda b: (b["row"], b["col"], b["colour"]))
    return tiles, targets, buttons, scale


def _make_groups(buttons, scale):
    """Pair visually aligned opposite-direction buttons."""
    used = set()
    groups = []
    for i, b in enumerate(buttons):
        if i in used or b["colour"] != 14:
            continue
        choices = []
        for j, a in enumerate(buttons):
            if j in used or a["colour"] != 8:
                continue
            if a["vertical"] != b["vertical"]:
                continue
            alignment = (abs(a["x"] - b["x"]) if b["vertical"]
                         else abs(a["y"] - b["y"]))
            if alignment <= scale:
                distance = abs(a["row"] - b["row"]) + abs(a["col"] - b["col"])
                choices.append((alignment, distance, j))
        if choices:
            j = min(choices)[2]
            groups.append([(i, 1), (j, -1)])
            used.update((i, j))

    for i in range(len(buttons)):
        if i not in used:
            groups.append([(i, 1)])
    return groups


def _hungarian(cost):
    n = len(cost)
    if not n:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j] = cur
                        way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
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


def _inverse(p):
    q = [0] * len(p)
    for i, j in enumerate(p):
        q[j] = i
    return q


def _cycle_count(p):
    seen = [False] * len(p)
    count = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j = i
        length = 0
        while not seen[j]:
            seen[j] = True
            length += 1
            j = p[j]
        if length > 1:
            count += 1
    return count


def _infer(samples, positions, penalty, deadline):
    """Colour-history matching, followed by geometric cycle refinement."""
    n = len(positions)
    sources, destinations = {}, {}
    for i, pos in enumerate(positions):
        a = tuple(before.get(pos, 4) for before, after in samples)
        b = tuple(after.get(pos, 4) for before, after in samples)
        sources.setdefault(a, []).append(i)
        destinations.setdefault(b, []).append(i)

    if set(sources) != set(destinations):
        return None
    if any(len(sources[k]) != len(destinations[k]) for k in sources):
        return None

    costs = []
    for x, y in positions:
        row = []
        for xx, yy in positions:
            d2 = ((x - xx) ** 2 + (y - yy) ** 2) / 9.0
            row.append(d2 if d2 <= 2.01 else 3.0 + 0.25 * math.sqrt(d2))
        costs.append(row)

    p = list(range(n))
    domains = [None] * n
    swap_groups = []
    for signature, src in sources.items():
        dst = destinations[signature]
        match = _hungarian([[costs[i][j] for j in dst] for i in src])
        for k, i in enumerate(src):
            p[i] = dst[match[k]]
            domains[i] = tuple(dst)
        if len(src) > 1:
            swap_groups.append(src)

    # A real conveyor normally forms one long cycle rather than several
    # accidental short cycles caused by repeated tile colours.
    if penalty:
        for _ in range(min(n, 24)):
            if time.monotonic() >= deadline:
                break
            old_cycles = _cycle_count(p)
            best_delta = -1e-8
            best = None
            for src in swap_groups:
                for a in range(len(src)):
                    i = src[a]
                    for b in range(a + 1, len(src)):
                        j = src[b]
                        pi, pj = p[i], p[j]
                        edge_delta = (costs[i][pj] + costs[j][pi]
                                      - costs[i][pi] - costs[j][pj])
                        if edge_delta >= penalty * max(1, old_cycles):
                            continue
                        p[i], p[j] = pj, pi
                        delta = edge_delta + penalty * (_cycle_count(p) - old_cycles)
                        p[i], p[j] = pi, pj
                        if delta < best_delta:
                            best_delta = delta
                            best = (i, j)
            if best is None:
                break
            i, j = best
            p[i], p[j] = p[j], p[i]

    inverse_domains = [None] * n
    for signature, src in sources.items():
        for j in destinations[signature]:
            inverse_domains[j] = tuple(src)
    return p, domains, inverse_domains


def _models(memory, positions, penalty, deadline):
    raw = memory["samples"]
    groups = memory["groups"]
    result = {}
    bad = []

    for gi, group in enumerate(groups):
        signs = dict(group)
        samples = []
        for button, before, after in raw:
            if button in signs:
                samples.append((before, after) if signs[button] == 1
                               else (after, before))
        if not samples:
            continue

        inferred = _infer(samples, positions, penalty,
                          min(deadline, time.monotonic() + 0.40))
        if inferred is None:
            if len(group) > 1:
                bad.append(gi)
            continue
        p, forward_domains, reverse_domains = inferred
        q = _inverse(p)
        for button, sign in group:
            result[button] = (p if sign == 1 else q,
                              forward_domains if sign == 1 else reverse_domains)

    if bad:
        new_groups = []
        for gi, group in enumerate(groups):
            if gi in bad:
                new_groups.extend([[(button, 1)] for button, sign in group])
            else:
                new_groups.append(group)
        memory["groups"] = new_groups
        return _models(memory, positions, penalty, deadline)
    return result


def _move_state(state, permutation, split):
    a = tuple(sorted(permutation[i] for i in state[:split]))
    b = tuple(sorted(permutation[i] for i in state[split:]))
    return a + b


def _search(start, goal, models, split, deadline):
    if start == goal:
        return []

    actions = sorted(models)
    permutations = [models[a][0] for a in actions]
    inverses = [_inverse(p) for p in permutations]

    forward = {start: (None, None)}
    backward = {goal: (None, None)}
    front_f = [start]
    front_b = [goal]
    meet = None
    expansions = 0

    while front_f and front_b:
        if len(forward) + len(backward) > 220000:
            break
        if time.monotonic() >= deadline:
            break

        from_forward = len(front_f) <= len(front_b)
        frontier = front_f if from_forward else front_b
        own = forward if from_forward else backward
        other = backward if from_forward else forward
        perms = permutations if from_forward else inverses
        next_frontier = []

        for state in frontier:
            for k, permutation in enumerate(perms):
                nxt = _move_state(state, permutation, split)
                if nxt == state or nxt in own:
                    continue
                own[nxt] = (state, actions[k])
                if nxt in other:
                    meet = nxt
                    break
                next_frontier.append(nxt)
            if meet is not None:
                break
            expansions += 1
            if expansions % 1024 == 0 and time.monotonic() >= deadline:
                return None
        if meet is not None:
            break
        if from_forward:
            front_f = next_frontier
        else:
            front_b = next_frontier

    if meet is None:
        return None

    left = []
    state = meet
    while forward[state][0] is not None:
        parent, action = forward[state]
        left.append(action)
        state = parent
    left.reverse()

    right = []
    state = meet
    while backward[state][0] is not None:
        parent, action = backward[state]
        right.append(action)
        state = parent
    return left + right


def _optimistic_probe(state, goal, models, split):
    """Find an informative first step when an inferred map is incomplete."""
    best = None
    for lo, hi in ((0, split), (split, len(state))):
        wanted = set(goal[lo:hi])
        occupied = set(state[lo:hi])
        wanted -= occupied
        if not wanted:
            continue

        for source in state[lo:hi]:
            if source in goal[lo:hi]:
                continue
            queue = deque([(source, None, 0)])
            seen = {source}
            while queue:
                node, first, depth = queue.popleft()
                if node in wanted:
                    candidate = (depth, first)
                    if best is None or candidate < best:
                        best = candidate
                    break
                if best is not None and depth >= best[0]:
                    continue
                for action in sorted(models):
                    for nxt in models[action][1][node]:
                        if nxt not in seen:
                            seen.add(nxt)
                            queue.append((nxt, action if first is None else first,
                                          depth + 1))
    return best[1] if best is not None else None


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 8.3
    tiles, targets, buttons, scale = _read_image(obs)

    if not buttons:
        return ("RESET", None, None) if "RESET" in available else ("ACTION6", 32, 32)

    fingerprint = tuple((b["colour"], b["row"], b["col"]) for b in buttons)
    if memory.get("fingerprint") != fingerprint:
        memory.clear()
        memory["fingerprint"] = fingerprint
        memory["buttons"] = buttons
        memory["groups"] = _make_groups(buttons, scale)
        memory["samples"] = []
        memory["targets"] = targets
        memory["positions"] = set(tiles)
        memory["uses"] = [0] * len(buttons)
        memory["plan"] = []

    for colour in (11, 12):
        memory["targets"][colour].update(targets[colour])
        memory["positions"].update(memory["targets"][colour])
    memory["positions"].update(tiles)

    pending = memory.pop("pending", None)
    prediction_ok = False
    if pending is not None:
        action, before = pending
        memory["samples"].append((action, before, dict(tiles)))
        prediction_ok = tiles == memory.pop("expected", None)
    if not prediction_ok:
        memory["plan"] = []

    positions = sorted(memory["positions"], key=lambda p: (p[1], p[0]))
    index = {pos: i for i, pos in enumerate(positions)}

    first_goals = tuple(sorted(index[p] for p, c in tiles.items() if c == 11))
    second_goals = tuple(sorted(index[p] for p, c in tiles.items() if c == 12))
    state = first_goals + second_goals
    split = len(first_goals)
    goal = (
        tuple(sorted(index[p] for p in memory["targets"][11]))
        + tuple(sorted(index[p] for p in memory["targets"][12]))
    )
    counts_match = (len(first_goals) == len(memory["targets"][11])
                    and len(second_goals) == len(memory["targets"][12]))

    def emit(action, permutation=None, remaining=None):
        memory["pending"] = (action, dict(tiles))
        memory["uses"][action] += 1
        memory["last"] = action
        memory["plan"] = remaining or []
        if permutation is not None:
            expected = {}
            for pos, colour in tiles.items():
                expected[positions[permutation[index[pos]]]] = colour
            memory["expected"] = expected
        else:
            memory.pop("expected", None)
        b = memory["buttons"][action]
        return "ACTION6", int(b["row"]), int(b["col"])

    if prediction_ok and memory["plan"]:
        action = memory["plan"][0]
        old_positions = memory.get("model_positions")
        old_models = memory.get("cached_models", {})
        if old_positions == positions and action in old_models:
            return emit(action, old_models[action][0], memory["plan"][1:])

    sampled_buttons = {sample[0] for sample in memory["samples"]}
    # One experiment per visually paired control is generally enough to
    # establish its support and its direction.
    for group in memory["groups"]:
        if not any(button in sampled_buttons for button, sign in group):
            primary = next((button for button, sign in group if sign == 1),
                           group[0][0])
            return emit(primary)

    models = _models(memory, positions, 4.5, deadline)
    memory["model_positions"] = positions
    memory["cached_models"] = models

    if counts_match and models:
        if state == goal:
            # Goal checking happens only after a button click.
            for action in sorted(models):
                if _move_state(state, models[action][0], split) == state:
                    return emit(action, models[action][0])

        plan = _search(state, goal, models, split,
                       min(deadline, time.monotonic() + 2.8))
        if plan:
            return emit(plan[0], models[plan[0]][0], plan[1:])

        # A different cycle preference can resolve a repeated-colour ambiguity.
        if time.monotonic() + 0.8 < deadline:
            alternative = _models(memory, positions, 8.0, deadline)
            plan = _search(state, goal, alternative, split,
                           min(deadline, time.monotonic() + 2.0))
            if plan:
                memory["cached_models"] = alternative
                return emit(plan[0], alternative[plan[0]][0], plan[1:])

        action = _optimistic_probe(state, goal, models, split)
        if action is not None:
            return emit(action, models[action][0])

    # More observations distinguish identical-colour tiles. Prefer an
    # under-tested control that actually moves a goal or changes the board.
    candidates = []
    for action in range(len(buttons)):
        uses = memory["uses"][action]
        if action in models:
            permutation, domains = models[action]
            moves_goal = any(permutation[i] != i for i in state)
            uncertainty = sum(len(domains[i]) > 1 for i in state)
            changes = sum(
                tiles.get(positions[i], 4) != tiles.get(positions[permutation[i]], 4)
                for i in range(len(positions))
            )
            priority = uses + (0 if moves_goal else 2) - 0.25 * uncertainty
            if changes == 0:
                priority += 2
        else:
            priority = uses - 2
        if action == memory.get("last"):
            priority += 0.1
        candidates.append((priority, action))

    action = min(candidates)[1]
    permutation = models[action][0] if action in models else None
    return emit(action, permutation)

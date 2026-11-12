import numpy as np
import math
import time
import heapq
import itertools
from collections import deque, Counter
from array import array


# Directions use the game's action ordering.
_DX = (0, 0, -1, 1)
_DY = (-1, 1, 0, 0)


def _wall_palette(obs):
    result = np.zeros((64, 64), dtype=np.int16)
    for y0, y1 in ((1, 32), (32, 63)):
        for x0, x1 in ((0, 32), (32, 64)):
            values = obs[y0:y1, x0:x1].ravel()
            counts = np.bincount(values, minlength=16)
            counts[5] = 0
            color = int(np.argmax(counts))
            result[0 if y0 == 1 else 32:
                   32 if y1 == 32 else 64, x0:x1] = color
    return result


def _geometry(obs, palette):
    mask = ((obs == 10) | (obs == 1)) & (obs != palette)
    seen = np.zeros((64, 64), dtype=bool)
    rectangles = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        points = []
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = yy + dy, xx + dx
                if (0 <= ny < 64 and 0 <= nx < 64 and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys = [p[0] for p in points]
        xs = [p[1] for p in points]
        top, left = min(ys), min(xs)
        height, width = max(ys) - top + 1, max(xs) - left + 1
        short = min(height, width)
        if (2 <= short <= 8 and
                len(points) >= height * width * .95):
            rectangles.append((short, left, top))
    if rectangles:
        counts = Counter(r[0] for r in rectangles)
        scale = max(counts, key=lambda k: (counts[k], k))
        good = [r for r in rectangles if r[0] == scale]
        ox = Counter(r[1] % scale for r in good).most_common(1)[0][0]
        oy = Counter(r[2] % scale for r in good).most_common(1)[0][0]
    else:
        scale, ox, oy = 4, 2, 2
    # Extra complete padding cells are harmless: they are parsed as walls.
    width = (64 - ox) // scale
    height = (64 - oy) // scale
    return scale, ox, oy, width, height


def _observe(obs, memory):
    palette = _wall_palette(obs)
    if "geometry" not in memory:
        memory["geometry"] = _geometry(obs, palette)
    scale, ox, oy, width, height = memory["geometry"]
    size = width * height

    tokens = set()
    token_colors = {}
    blocks = set()
    selected = None
    hazards_now = set()
    walls_now = set()
    colored = {}
    covered = set()

    for y in range(height):
        for x in range(width):
            p = y * width + x
            sy, sx = oy + y * scale, ox + x * scale
            tile = obs[sy:sy + scale, sx:sx + scale]
            pal = palette[sy:sy + scale, sx:sx + scale]
            normalized = np.where(tile == pal, 0, tile)
            area = tile.size
            center = int(tile[scale // 2, scale // 2])

            is_token = False
            for color in (10, 1):
                if np.count_nonzero(normalized == color) >= .70 * area:
                    tokens.add(p)
                    token_colors[p] = color
                    covered.add(p)
                    is_token = True
                    break
            if is_token:
                continue

            border = np.concatenate((tile[0], tile[-1],
                                     tile[1:-1, 0], tile[1:-1, -1]))
            if (scale >= 3 and center in (9, 11) and
                    np.count_nonzero(border == 5) >= .9 * len(border)):
                blocks.add(p)
                covered.add(p)
                if center == 11:
                    selected = p
                continue

            yy, xx = np.indices(tile.shape)
            odd = ((yy + sy + xx + sx) & 1) != 0
            even_values = tile[~odd]
            if even_values.size and np.any(odd):
                color = Counter(map(int, even_values)).most_common(1)[0][0]
                if (color != 5 and
                        np.mean(tile[odd] == 5) >= .9 and
                        np.mean(even_values == color) >= .9):
                    hazards_now.add(p)
                    continue

            if center in (12, 14, 15):
                if (np.count_nonzero(tile == center) >= .8 * area and
                        np.count_nonzero(normalized == 0) < .5 * area):
                    colored[p] = center
                    continue

            if np.count_nonzero(normalized == 0) >= .5 * area:
                walls_now.add(p)

    old = memory.get("env")
    if old is None:
        walls, hazards, gates, switches = set(), set(), {}, {}
        old_signature = None
    else:
        walls = set(old["walls"])
        hazards = set(old["hazards"])
        gates = dict(old["gates"])
        switches = dict(old["switches"])
        old_signature = old["signature"]

    color_bit = {14: 1, 12: 2, 15: 4}

    visited = set()
    for p, color in colored.items():
        if p in visited:
            continue
        component = []
        queue = [p]
        visited.add(p)
        while queue:
            q = queue.pop()
            component.append(q)
            qx, qy = q % width, q // width
            for dx, dy in zip(_DX, _DY):
                nx, ny = qx + dx, qy + dy
                r = ny * width + nx
                if (0 <= nx < width and 0 <= ny < height and
                        r not in visited and colored.get(r) == color):
                    visited.add(r)
                    queue.append(r)
        bit = color_bit[color]
        if len(component) >= 2:
            for q in component:
                gates[q] = bit
                switches.pop(q, None)
        else:
            q = component[0]
            if q not in gates:
                switches[q] = bit

    # A player can obscure the middle or the end of a three-cell gate.
    for p, color in colored.items():
        bit = color_bit[color]
        for delta in (1, width):
            for start in (p - 2 * delta, p - delta, p):
                cells = (start, start + delta, start + 2 * delta)
                if not all(0 <= q < size for q in cells):
                    continue
                if delta == 1 and cells[0] // width != cells[-1] // width:
                    continue
                if (sum(colored.get(q) == color for q in cells) >= 2 and
                        all(colored.get(q) == color or q in tokens
                            for q in cells)):
                    for q in cells:
                        gates[q] = bit
                        switches.pop(q, None)

    hazards.update(hazards_now)
    for p in range(size):
        if p in covered:
            continue
        if p in walls_now:
            walls.add(p)
        else:
            walls.discard(p)
    walls.difference_update(gates)
    walls.difference_update(switches)
    walls.difference_update(hazards)
    walls.difference_update(tokens)
    walls.difference_update(blocks)

    gate_array = [0] * size
    switch_array = [0] * size
    all_gate_bits = 0
    closed_bits = 0
    for p, bit in gates.items():
        gate_array[p] = bit
        all_gate_bits |= bit
        if p in colored and color_bit[colored[p]] == bit:
            closed_bits |= bit
    for p, bit in switches.items():
        switch_array[p] = bit
    opened = all_gate_bits & ~closed_bits

    neighbors = [[0] * size for _ in range(4)]
    for p in range(size):
        x, y = p % width, p // width
        for a in range(4):
            nx, ny = x + _DX[a], y + _DY[a]
            q = ny * width + nx
            neighbors[a][p] = (
                q if 0 <= nx < width and 0 <= ny < height and q not in walls
                else p
            )

    signature = (width, height, frozenset(walls), frozenset(hazards),
                 tuple(sorted(gates.items())), tuple(sorted(switches.items())))
    env = {
        "width": width, "height": height, "size": size,
        "walls": walls, "hazards": hazards,
        "gates": gates, "switches": switches,
        "gate_array": gate_array, "switch_array": switch_array,
        "neighbors": neighbors, "signature": signature,
    }
    memory["env"] = env
    return (env, tokens, token_colors, blocks, selected, opened,
            old_signature is not None and old_signature != signature)


def _initial_hypotheses(tokens, colors, broad=False):
    positions = sorted(tokens)
    result = []
    n = len(positions)
    if n >= 4:
        for perm in itertools.permutations(positions[:4]):
            result.append((tuple(perm), ()))
    elif n == 3:
        possible_ghosts = [p for p in positions if colors.get(p) == 10]
        if not possible_ghosts:
            possible_ghosts = positions
        for ghost in possible_ghosts:
            active = [p for p in positions if p != ghost]
            for i, j in itertools.permutations(range(4), 2):
                p = [-1] * 4
                p[i], p[j] = active
                result.append((tuple(p), (ghost,)))
    elif n == 2:
        choices = [(0, 1), (1, 0)]
        if broad:
            choices += [ij for ij in itertools.permutations(range(4), 2)
                        if ij not in choices]
        for i, j in choices:
            p = [-1] * 4
            p[i], p[j] = positions
            result.append((tuple(p), ()))
    elif n == 1:
        result.append(((-1, -1, -1, -1), tuple(positions)))
    return result


def _open_from_players(players, env):
    opened = 0
    switches = env["switch_array"]
    for p in players:
        if p >= 0:
            opened |= switches[p]
    return opened


def _token_step(players, action, blocks, opened, env):
    q = list(players)
    active = [i for i, p in enumerate(players) if p >= 0]
    neighbors = env["neighbors"]
    gates = env["gate_array"]
    hazards = env["hazards"]
    width = env["width"]

    for i in active:
        a = action
        if action >= 2 and (i & 1):
            a ^= 1
        if action < 2 and (i & 2):
            a ^= 1
        old = players[i]
        dest = neighbors[a][old]
        if dest in blocks or (gates[dest] & ~opened):
            dest = old
        if dest in hazards:
            return None
        q[i] = dest

    # The engine also merges horizontal crossings/following moves.
    if action >= 2:
        for ii, i in enumerate(active):
            for j in active[ii + 1:]:
                pi, pj = players[i], players[j]
                if (abs(pi - pj) == 1 and pi // width == pj // width and
                        (q[i] == pj or q[j] == pi)):
                    x = ((q[i] % width) + (q[j] % width)) // 2
                    y = ((q[i] // width) + (q[j] // width)) // 2
                    q[i] = q[j] = y * width + x

    groups = {}
    for i in active:
        groups.setdefault(q[i], []).append(i)
    merged = []
    for location, members in groups.items():
        if len(members) >= 2:
            merged.append(location)
            for i in members[:2]:
                q[i] = -1
            for i in members[2:]:
                q[i] = players[i]
    return tuple(q), tuple(merged)


def _block_destination(position, action, players, blocks, opened, env):
    dest = env["neighbors"][action][position]
    if (dest in env["hazards"] or dest in env["switches"] or
            (env["gate_array"][dest] & ~opened) or
            (dest in blocks and dest != position) or
            dest in players):
        return position
    return dest


def _predict(hypothesis, action, click, blocks, selected, opened, env):
    players, ghosts = hypothesis
    new_blocks = set(blocks)
    if action == 6:
        return hypothesis, frozenset(new_blocks)
    if action not in (1, 2, 3, 4):
        return hypothesis, frozenset(new_blocks)
    if selected is not None:
        dest = _block_destination(
            selected, action - 1, players, blocks, opened, env)
        new_blocks.discard(selected)
        new_blocks.add(dest)
        return hypothesis, frozenset(new_blocks)
    result = _token_step(players, action - 1, blocks, opened, env)
    if result is None:
        return None
    new_players, merged = result
    return ((new_players, tuple(sorted(set(ghosts + merged)))),
            frozenset(new_blocks))


def _visible(hypothesis):
    players, ghosts = hypothesis
    return frozenset([p for p in players if p >= 0] + list(ghosts))


def _pair_table(env, blocked, kind, deadline):
    size, width = env["size"], env["width"]
    cells = [p for p in range(size) if p not in blocked]
    m = len(cells)
    index = [-1] * size
    for i, p in enumerate(cells):
        index[p] = i
    if not m:
        return m, index, array("h")

    moves = [[0] * m for _ in range(4)]
    for a in range(4):
        for i, p in enumerate(cells):
            x, y = p % width, p // width
            nx, ny = x + _DX[a], y + _DY[a]
            q = ny * width + nx
            if (0 <= nx < width and 0 <= ny < env["height"] and
                    q not in blocked):
                moves[a][i] = index[q]
            else:
                moves[a][i] = i

    second = []
    for a in range(4):
        aa = a
        if a >= 2 and (kind & 1):
            aa ^= 1
        if a < 2 and (kind & 2):
            aa ^= 1
        second.append(moves[aa])

    inv1 = [[[] for _ in range(m)] for _ in range(4)]
    inv2 = [[[] for _ in range(m)] for _ in range(4)]
    for a in range(4):
        for i in range(m):
            inv1[a][moves[a][i]].append(i)
            inv2[a][second[a][i]].append(i)

    distance = array("h", [-1]) * (m * m)
    queue = deque()
    for i in range(m):
        k = i * m + i
        distance[k] = 0
        queue.append(k)

    for i, p in enumerate(cells):
        for q in (p - 1, p + 1):
            if not (0 <= q < size and q // width == p // width):
                continue
            j = index[q]
            if j < 0:
                continue
            if any(moves[a][i] == j or second[a][j] == i
                   for a in (2, 3)):
                k = i * m + j
                if distance[k] < 0:
                    distance[k] = 1
                    queue.append(k)

    count = 0
    while queue:
        k = queue.popleft()
        i, j = divmod(k, m)
        nd = distance[k] + 1
        for a in range(4):
            for u in inv1[a][i]:
                base = u * m
                for v in inv2[a][j]:
                    kk = base + v
                    if distance[kk] < 0:
                        distance[kk] = nd
                        queue.append(kk)
        count += 1
        if count % 4096 == 0 and time.perf_counter() > deadline:
            return None
    return m, index, distance


def _make_heuristic(env, players, blocks, movable, memory, deadline):
    blocked = frozenset(
        env["walls"] | env["hazards"] | (set() if movable else set(blocks)))
    active = [i for i, p in enumerate(players) if p >= 0]
    kinds = {i ^ j for i, j in itertools.combinations(active, 2)}
    cache = memory.setdefault("distance_cache", {})
    if len(cache) > 12:
        cache.clear()
    tables = {}
    for kind in kinds:
        key = (env["width"], env["height"], blocked, kind)
        if key not in cache and time.perf_counter() < deadline:
            table = _pair_table(env, blocked, kind, deadline)
            if table is not None:
                cache[key] = table
        tables[kind] = cache.get(key)

    width = env["width"]
    hcache = {}

    def pair(i, j, p):
        table = tables.get(i ^ j)
        if table is not None:
            m, index, distance = table
            a, b = index[p[i]], index[p[j]]
            if a >= 0 and b >= 0:
                d = distance[a * m + b]
                return float(d) if d >= 0 else 1000000.0
        a, b = p[i], p[j]
        return (abs(a % width - b % width) +
                abs(a // width - b // width)) / 2.0

    def heuristic(p):
        if p in hcache:
            return hcache[p]
        ids = [i for i, q in enumerate(p) if q >= 0]
        if not ids:
            value = 0.0
        elif len(ids) == 2:
            value = pair(ids[0], ids[1], p)
        elif len(ids) == 4:
            a, b, c, d = ids
            values = []
            for i, j, k, l in ((a, b, c, d),
                               (a, c, b, d),
                               (a, d, b, c)):
                x, y = pair(i, j, p), pair(k, l, p)
                values.append(max(x, y) + .20 * min(x, y))
            value = min(values)
        else:
            value = 1000000.0
        if len(hcache) < 200000:
            hcache[p] = value
        return value

    return heuristic


def _search(env, players, blocks, selected, opened, movable,
            memory, deadline):
    block_tuple = tuple(sorted(blocks))
    count_blocks = len(block_tuple)
    mode = block_tuple.index(selected) if selected in blocks else -1
    root = tuple(players) + block_tuple + (mode, opened)

    heuristic = _make_heuristic(
        env, players, blocks, movable, memory, deadline)
    root_h = heuristic(tuple(players))
    if root_h >= 1000000:
        return None, False
    if all(p < 0 for p in players):
        return [(5, None)], True

    weight = 1.65 if sum(p >= 0 for p in players) == 4 else 1.35
    heap = [(weight * root_h, root_h, 0, 0, root)]
    distance = {root: 0}
    parent = {}
    serial = 0
    best = root
    best_key = (root_h, 0)
    expanded = 0

    def reconstruct(state):
        chunks = []
        while state != root:
            previous, edge = parent[state]
            chunks.append(edge)
            state = previous
        answer = []
        for chunk in reversed(chunks):
            answer.extend(chunk)
        return answer

    while heap:
        _, h, g, _, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        p = state[:4]
        if all(q < 0 for q in p):
            return reconstruct(state), True
        if (h, g) < best_key:
            best_key, best = (h, g), state

        if expanded % 128 == 0:
            if (time.perf_counter() >= deadline or
                    len(distance) > 260000):
                break
        expanded += 1

        bs = state[4:4 + count_blocks]
        bset = set(bs)
        old_mode, old_open = state[-2:]
        successors = []

        for a in range(4):
            result = _token_step(p, a, bset, old_open, env)
            if result is None:
                continue
            np_, _ = result
            new_open = _open_from_players(np_, env)
            if np_ == p and new_open == old_open:
                continue
            edge = []
            if old_mode != -1:
                target = next((q for q in p if q >= 0), None)
                if target is None:
                    continue
                edge.append((6, target))
            edge.append((a + 1, None))
            ns = np_ + bs + (-1, new_open)
            successors.append((ns, edge))

        if movable:
            for bi, bp in enumerate(bs):
                for a in range(4):
                    dest = _block_destination(
                        bp, a, p, bset, old_open, env)
                    if dest == bp:
                        continue
                    new_bs = list(bs)
                    new_bs[bi] = dest
                    edge = []
                    if old_mode != bi:
                        edge.append((6, bp))
                    edge.append((a + 1, None))
                    new_open = _open_from_players(p, env)
                    ns = p + tuple(new_bs) + (bi, new_open)
                    successors.append((ns, edge))

        for ns, edge in successors:
            ng = g + len(edge)
            if ng >= distance.get(ns, 10 ** 9) or ng > 145:
                continue
            nh = heuristic(ns[:4])
            if nh >= 1000000:
                continue
            distance[ns] = ng
            parent[ns] = (state, edge)
            serial += 1
            heapq.heappush(
                heap, (ng + weight * nh, nh, ng, serial, ns))

    if best != root:
        return reconstruct(best), False
    return None, False


def _articulation_risks(cells, env):
    """Ordered pairs separated by removing each cell."""
    width = env["width"]
    adjacency = {}
    for p in cells:
        x, y = p % width, p // width
        adjacency[p] = [
            (y + dy) * width + x + dx
            for dx, dy in zip(_DX, _DY)
            if (0 <= x + dx < width and
                0 <= y + dy < env["height"] and
                (y + dy) * width + x + dx in cells)
        ]

    risks = {}
    unseen = set(cells)
    while unseen:
        seed = next(iter(unseen))
        component = {seed}
        queue = [seed]
        unseen.remove(seed)
        while queue:
            p = queue.pop()
            for q in adjacency[p]:
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    queue.append(q)
        n = len(component)
        for removed in component:
            if len(adjacency[removed]) <= 1:
                risks[removed] = 0
                continue
            remaining = component - {removed}
            squares = 0
            while remaining:
                q = remaining.pop()
                queue = [q]
                count = 1
                while queue:
                    p = queue.pop()
                    for r in adjacency[p]:
                        if r in remaining:
                            remaining.remove(r)
                            queue.append(r)
                            count += 1
                squares += count * count
            risks[removed] = (n - 1) ** 2 - squares
    return risks, adjacency


def _parking_plan(env, players, ghosts, blocks, selected,
                  opened, memory, deadline):
    if not blocks:
        return None
    forbidden = set(env["walls"]) | set(env["hazards"])
    forbidden.update(p for p, bit in env["gates"].items()
                     if bit & ~opened)
    choices = []
    recent = memory.setdefault("parking_targets", {})

    for bp in sorted(blocks):
        if time.perf_counter() > deadline:
            break
        cells = set(range(env["size"])) - forbidden - (set(blocks) - {bp})
        risks, adjacency = _articulation_risks(cells, env)
        old_risk = risks.get(bp, 0)
        if old_risk <= 0:
            continue

        queue = deque([bp])
        previous = {bp: None}
        distances = {bp: 0}
        while queue:
            p = queue.popleft()
            if distances[p] >= 24:
                continue
            for a in range(4):
                q = env["neighbors"][a][p]
                if (q == p or q in previous or q not in cells or
                        q in env["switches"] or q in players):
                    continue
                previous[q] = (p, a)
                distances[q] = distances[p] + 1
                queue.append(q)

        for target, steps in distances.items():
            if target == bp or recent.get(target, 0) >= 2:
                continue
            new_risk = risks.get(target, old_risk)
            if new_risk >= old_risk:
                continue
            cost = steps + (0 if selected == bp else 1) + 1
            improvement = old_risk - new_risk
            leaf_bonus = .1 if len(adjacency.get(target, [])) <= 1 else 0
            score = improvement / (cost + 2) + leaf_bonus
            path = []
            p = target
            while p != bp:
                old, a = previous[p]
                path.append((a + 1, None))
                p = old
            path.reverse()
            if selected != bp:
                path.insert(0, (6, bp))
            group_target = next((p for p in players if p >= 0),
                                ghosts[0] if ghosts else None)
            if group_target is not None:
                path.append((6, group_target))
            choices.append((score, -cost, target, path))

    if not choices:
        return None
    _, _, target, path = max(choices, key=lambda item: item[:3])
    recent[target] = recent.get(target, 0) + 1
    return path


def _choose_probe(hypotheses, blocks, selected, opened, env):
    if len(hypotheses) <= 1:
        return None
    if selected is not None:
        target = next((p for p in hypotheses[0][0] if p >= 0), None)
        return (6, target) if target is not None else None

    best = None
    for action in (4, 1, 3, 2):
        outcomes = Counter()
        moving = 0
        for hypothesis in hypotheses:
            prediction = _predict(
                hypothesis, action, None, blocks, None, opened, env)
            if prediction is None:
                key = ("hazard",)
            else:
                key = tuple(sorted(_visible(prediction[0])))
                moving += key != tuple(sorted(_visible(hypothesis)))
            outcomes[key] += 1
        if len(outcomes) <= 1:
            continue
        n = len(hypotheses)
        entropy = -sum((v / n) * math.log(v / n) for v in outcomes.values())
        score = entropy + .001 * moving
        if best is None or score > best[0]:
            best = (score, (action, None))
    return best[1] if best else None


def act(obs, available, memory):
    started = time.perf_counter()
    obs = np.asarray(obs)
    (env, tokens, colors, blocks, selected, opened,
     map_changed) = _observe(obs, memory)

    hypotheses = []
    primary_survived = False
    predictions = memory.pop("predictions", None)
    if predictions is not None:
        seen = set()
        for hypothesis, expected_blocks, primary in predictions:
            if _visible(hypothesis) == frozenset(tokens) and expected_blocks == frozenset(blocks):
                if hypothesis not in seen:
                    hypotheses.append(hypothesis)
                    seen.add(hypothesis)
                primary_survived |= primary

    if not hypotheses:
        if predictions is not None:
            memory["mismatches"] = memory.get("mismatches", 0) + 1
        hypotheses = _initial_hypotheses(
            tokens, colors, broad=memory.get("mismatches", 0) > 0)
        memory["plan"] = []
    elif not primary_survived:
        memory["plan"] = []

    if map_changed:
        memory["plan"] = []

    if not hypotheses:
        return ("ACTION5" if "ACTION5" in available else available[0],
                None, None)

    plan = memory.setdefault("plan", [])
    action = None

    if not plan and memory.get("probes", 0) < 4:
        probe = _choose_probe(hypotheses, blocks, selected, opened, env)
        if probe is not None:
            action = probe
            memory["probes"] = memory.get("probes", 0) + 1

    if action is None and not plan:
        players, ghosts = hypotheses[0]
        if all(p < 0 for p in players):
            plan = [(5, None)]
        else:
            total_deadline = started + 8.4
            fixed_deadline = min(
                total_deadline, time.perf_counter() + (2.2 if blocks else 7.8))
            fixed_plan, solved = _search(
                env, players, blocks, selected, opened, False,
                memory, fixed_deadline)

            if solved:
                plan = fixed_plan
            else:
                parking = None
                if blocks and time.perf_counter() < total_deadline - .2:
                    parking = _parking_plan(
                        env, players, ghosts, blocks, selected, opened,
                        memory, min(total_deadline - .1,
                                    time.perf_counter() + 1.0))
                if parking:
                    plan = parking
                elif blocks and time.perf_counter() < total_deadline - .2:
                    joint_plan, joint_solved = _search(
                        env, players, blocks, selected, opened, True,
                        memory, total_deadline)
                    if joint_plan:
                        plan = joint_plan if joint_solved else joint_plan[:10]
                    elif fixed_plan:
                        plan = fixed_plan[:8]
                elif fixed_plan:
                    plan = fixed_plan[:10]
        memory["plan"] = plan or []

    if action is None and memory["plan"]:
        action = memory["plan"].pop(0)

    if action is None:
        # Model-guided exploration when a bounded search did not finish.
        players, ghosts = hypotheses[0]
        if selected is not None:
            target = next((p for p in players if p >= 0),
                          next(iter(tokens), 0))
            action = (6, target)
        else:
            visits = memory.setdefault("fallback_visits", {})
            candidates = []
            for a in (4, 1, 3, 2):
                prediction = _predict(
                    hypotheses[0], a, None, blocks, selected, opened, env)
                if prediction is None:
                    continue
                key = prediction[0][0]
                changed = key != players
                score = -visits.get(key, 0) + (1 if changed else -3)
                candidates.append((score, a, key))
            if candidates:
                _, a, key = max(candidates)
                visits[key] = visits.get(key, 0) + 1
                action = (a, None)
            else:
                action = (5, None)

    number, click = action
    name = "ACTION" + str(number)
    if name not in available:
        alternatives = [a for a in available if a != "RESET"]
        name = alternatives[0] if alternatives else available[0]
        number = int(name[6:]) if name.startswith("ACTION") else 0
        click = None
        memory["plan"] = []

    predictions = []
    for i, hypothesis in enumerate(hypotheses):
        prediction = _predict(
            hypothesis, number, click, blocks, selected, opened, env)
        if prediction is not None:
            new_hypothesis, expected_blocks = prediction
            predictions.append((new_hypothesis, expected_blocks, i == 0))
    memory["predictions"] = predictions

    if number == 6:
        if click is None:
            click = next(iter(tokens), 0)
        scale, ox, oy, width, height = memory["geometry"]
        row = oy + (click // width) * scale + scale // 2
        col = ox + (click % width) * scale + scale // 2
        return name, int(min(63, max(0, row))), int(min(63, max(0, col)))
    return name, None, None

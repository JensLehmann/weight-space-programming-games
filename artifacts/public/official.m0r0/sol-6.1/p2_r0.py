import numpy as np
import itertools
import heapq
import time
from collections import deque, Counter
from math import gcd

_SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_COLOR_BITS = {14: 1, 12: 2, 15: 4}
_INF = 65535


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        xmin = xmax = x
        ymin = ymax = y
        count = 0
        while stack:
            yy, xx = stack.pop()
            count += 1
            xmin, xmax = min(xmin, xx), max(xmax, xx)
            ymin, ymax = min(ymin, yy), max(ymax, yy)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = yy + dy, xx + dx
                if (0 <= ny < h and 0 <= nx < w
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append((xmin, ymin, xmax - xmin + 1,
                       ymax - ymin + 1, count))
    return result


def _geometry(obs):
    corners = {int(obs[1, 1]), int(obs[1, 62]),
               int(obs[62, 1]), int(obs[62, 62])}
    candidates = []
    for color in (10, 1):
        if color == 1 and color in corners:
            continue
        for x, y, w, h, count in _components(obs == color):
            if count == w * h and 2 <= min(w, h) <= 8 and max(w, h) <= 24:
                candidates.append((x, y, w, h))
        if candidates:
            break

    if candidates:
        d = 0
        for _, _, w, h in candidates:
            d = gcd(d, gcd(w, h))
        if not 2 <= d <= 8:
            d = Counter(min(w, h) for _, _, w, h in candidates).most_common(1)[0][0]
        s = d
        rx = Counter(x % s for x, _, _, _ in candidates).most_common(1)[0][0]
        ry = Counter(y % s for _, y, _, _ in candidates).most_common(1)[0][0]
    else:
        interiors = []
        for color in (9, 11):
            if color in corners:
                continue
            for x, y, w, h, count in _components(obs == color):
                if w == h and count == w * h and 1 <= w <= 6:
                    interiors.append((x - 1, y - 1, w + 2))
        if interiors:
            s = Counter(z[2] for z in interiors).most_common(1)[0][0]
            rx = Counter(z[0] % s for z in interiors).most_common(1)[0][0]
            ry = Counter(z[1] % s for z in interiors).most_common(1)[0][0]
        else:
            s, rx, ry = 4, 2, 2

    w = (64 - 2 * rx) // s
    h = (64 - 2 * ry) // s
    return s, rx, ry, w, h


def _read(obs, memory):
    if "geometry" not in memory:
        memory["geometry"] = _geometry(obs)
    s, ox, oy, w, h = memory["geometry"]
    n = w * h

    # The recolored wall/padding pattern can change after a pair meets.
    corner = [int(obs[1, 1]), int(obs[1, 62]),
              int(obs[62, 1]), int(obs[62, 62])]
    yy, xx = np.indices((64, 64))
    expected = np.empty((64, 64), dtype=np.int16)
    expected[(yy < 32) & (xx < 32)] = corner[0]
    expected[(yy < 32) & (xx >= 32)] = corner[1]
    expected[(yy >= 32) & (xx < 32)] = corner[2]
    expected[(yy >= 32) & (xx >= 32)] = corner[3]

    old = memory.get("world")
    gates = list(old["gates"]) if old else [0] * n
    switches = list(old["switches"]) if old else [0] * n
    traps = list(old["traps"]) if old else [False] * n
    walls = [False] * n
    agents = {}
    blocks = []
    selected_cell = None
    colored = {c: set() for c in _COLOR_BITS}

    for gy in range(h):
        for gx in range(w):
            p = gy * w + gx
            y, x = oy + gy * s, ox + gx * s
            tile = obs[y:y+s, x:x+s]
            ep = expected[y:y+s, x:x+s]
            valid = np.ones(tile.shape, dtype=bool)
            if y == 0:
                valid[0, :] = False
            if y + s == 64:
                valid[-1, :] = False
            vals = tile[valid]
            if not len(vals):
                continue

            # Movable blocks have a background-colored outline.
            if s >= 3:
                inner = tile[1:-1, 1:-1]
                border = np.concatenate((tile[0], tile[-1],
                                         tile[1:-1, 0], tile[1:-1, -1]))
                if (inner.size and np.all(inner == inner.flat[0])
                        and int(inner.flat[0]) in (9, 11)
                        and np.all(border == 5)):
                    blocks.append(p)
                    if int(inner.flat[0]) == 11:
                        selected_cell = p
                    continue

            # Hazards are rendered with an alternating background mask.
            colors = set(map(int, vals))
            if 5 in colors and len(colors) == 2:
                local_y, local_x = np.indices(tile.shape)
                parity = ((local_y + y + local_x + x) & 1) == 1
                other = next(c for c in colors if c != 5)
                if (np.all(tile[parity & valid] == 5)
                        and np.all(tile[(~parity) & valid] == other)):
                    traps[p] = True
                    continue

            is_wall = bool(np.all(vals == ep[valid]))
            uniform = bool(np.all(vals == vals[0]))
            color = int(vals[0]) if uniform else -1

            if uniform and color in (10, 1) and not is_wall:
                agents[p] = color
            elif uniform and color in _COLOR_BITS and not is_wall:
                colored[color].add(p)
            elif is_wall:
                walls[p] = True

    # Length-three colored strips are doors; isolated colored tiles are buttons.
    for color, cells in colored.items():
        bit = _COLOR_BITS[color]
        door_cells = set()
        for p in cells:
            x, y = p % w, p // w
            for dx, dy in ((1, 0), (0, 1)):
                run = []
                xx0, yy0 = x, y
                while 0 <= xx0 < w and 0 <= yy0 < h:
                    q = yy0 * w + xx0
                    if q not in cells:
                        break
                    run.append(q)
                    xx0 += dx
                    yy0 += dy
                if len(run) >= 3:
                    door_cells.update(run)

        # Also preserve known door identities when part of a strip is obscured.
        door_cells.update(p for p in cells if gates[p] == bit)
        remaining = cells - door_cells
        while remaining:
            p = remaining.pop()
            comp = {p}
            stack = [p]
            while stack:
                q = stack.pop()
                x, y = q % w, q // w
                for dx, dy in _DXY:
                    nx, ny = x + dx, y + dy
                    r = ny * w + nx
                    if 0 <= nx < w and 0 <= ny < h and r in remaining:
                        remaining.remove(r)
                        comp.add(r)
                        stack.append(r)
            if len(comp) >= 2 and not any(switches[q] == bit for q in comp):
                door_cells.update(comp)

        for p in cells:
            if p in door_cells:
                gates[p] = bit
                switches[p] = 0
            elif not gates[p]:
                switches[p] = bit

    for p in range(n):
        if gates[p] or switches[p] or traps[p] or p in agents or p in blocks:
            walls[p] = False

    # Match visually identical blocks to their previous identities.
    previous = memory.get("blocks")
    blocks.sort()
    if previous and len(previous) == len(blocks):
        if len(blocks) <= 6:
            def distance(order):
                return sum(abs(a % w - b % w) + abs(a // w - b // w)
                           for a, b in zip(previous, order))
            blocks = list(min(itertools.permutations(blocks), key=distance))
        else:
            unused = set(blocks)
            ordered = []
            for a in previous:
                b = min(unused, key=lambda q: abs(a % w - q % w)
                        + abs(a // w - q // w))
                ordered.append(b)
                unused.remove(b)
            blocks = ordered

    mode = blocks.index(selected_cell) if selected_cell in blocks else -1
    closed = 0
    for color, cells in colored.items():
        bit = _COLOR_BITS[color]
        if any(gates[p] == bit for p in cells):
            closed |= bit

    all_bits = 0
    for bit in gates:
        all_bits |= bit

    world = dict(w=w, h=h, walls=walls, gates=gates, switches=switches,
                 traps=traps, all_bits=all_bits)
    static_sig = (tuple(walls), tuple(traps))
    full_sig = (static_sig, tuple(gates), tuple(switches))
    changed = memory.get("full_sig") != full_sig
    if memory.get("static_sig") != static_sig:
        memory.pop("pdb", None)
    memory["static_sig"] = static_sig
    memory["full_sig"] = full_sig
    memory["world"] = world
    return world, agents, tuple(blocks), mode, closed, changed


def _gate_state(world, agents):
    pressed = 0
    switches = world["switches"]
    for p in agents:
        if p >= 0:
            pressed |= switches[p]
    return world["all_bits"] & ~pressed


def _agent_step(world, agents, blocks, closed, action):
    w, h = world["w"], world["h"]
    walls, gates, traps = world["walls"], world["gates"], world["traps"]
    occupied = set(blocks)
    dx, dy = _DXY[action - 1]
    moved = list(agents)
    ids = [i for i, p in enumerate(agents) if p >= 0]

    for i in ids:
        p = agents[i]
        sx, sy = _SIGNS[i]
        x, y = p % w + dx * sx, p // w + dy * sy
        if not (0 <= x < w and 0 <= y < h):
            continue
        q = y * w + x
        if walls[q] or q in occupied or (gates[q] & closed):
            continue
        if traps[q]:
            return None
        moved[i] = q

    # Crossing horizontally adjacent pieces also makes a pair.
    for k, i in enumerate(ids):
        if moved[i] < 0:
            continue
        for j in ids[k+1:]:
            p, q = agents[i], agents[j]
            if p // w == q // w and abs(p - q) == 1:
                if moved[i] == q or moved[j] == p:
                    r = (moved[i] + moved[j]) // 2
                    moved[i] = moved[j] = r

    groups = {}
    for i in ids:
        groups.setdefault(moved[i], []).append(i)
    paired_at = []
    for p, group in groups.items():
        if len(group) >= 2:
            paired_at.append(p)
            moved[group[0]] = moved[group[1]] = -1
            for i in group[2:]:
                moved[i] = agents[i]

    moved = tuple(moved)
    return moved, _gate_state(world, moved), tuple(paired_at)


def _block_step(world, agents, blocks, mode, closed, action):
    if mode < 0 or mode >= len(blocks):
        return None
    w, h = world["w"], world["h"]
    p = blocks[mode]
    dx, dy = _DXY[action - 1]
    x, y = p % w + dx, p // w + dy
    if not (0 <= x < w and 0 <= y < h):
        return None
    q = y * w + x
    if (world["walls"][q] or world["traps"][q]
            or world["switches"][q] or (world["gates"][q] & closed)
            or q in agents
            or any(q == b for i, b in enumerate(blocks) if i != mode)):
        return None
    b = list(blocks)
    b[mode] = q
    return tuple(b), _gate_state(world, agents)


def _initial_hypotheses(observed, mode):
    ones = sorted(p for p, c in observed.items() if c == 1)
    tens = sorted(p for p, c in observed.items() if c == 10)
    cells = sorted(observed)
    result = []

    if mode >= 0 and ones:
        active, ghosts = ones, frozenset(tens)
    else:
        active, ghosts = cells, frozenset()

    if len(active) == 4:
        for order in itertools.permutations(active):
            result.append((tuple(order), ghosts))
    elif len(active) == 3 and not ghosts:
        for ghost in active:
            rest = [p for p in active if p != ghost]
            for i, j in itertools.combinations(range(4), 2):
                for order in (rest, rest[::-1]):
                    a = [-1] * 4
                    a[i], a[j] = order
                    result.append((tuple(a), frozenset((ghost,))))
    elif len(active) == 2:
        if ghosts:
            for i, j in itertools.combinations(range(4), 2):
                for order in (active, active[::-1]):
                    a = [-1] * 4
                    a[i], a[j] = order
                    result.append((tuple(a), ghosts))
        else:
            result = [(tuple(active), ghosts), (tuple(active[::-1]), ghosts)]
    elif len(active) == 1:
        # Normally this observation occurs only at completion.
        result = [((active[0], -1), ghosts)]
    return list(dict.fromkeys(result))


def _matches(hyp, observed, mode):
    agents, ghosts = hyp
    visible = {p for p in agents if p >= 0} | set(ghosts)
    if visible != set(observed):
        return False
    if mode >= 0:
        active = {p for p in agents if p >= 0}
        if any(c == 1 and p not in active for p, c in observed.items()):
            return False
    return True


def _pair_type(i, j):
    sx = _SIGNS[i][0] != _SIGNS[j][0]
    sy = _SIGNS[i][1] != _SIGNS[j][1]
    return "D" if sx and sy else ("H" if sx else "V")


def _build_pdb(world, types, deadline):
    w, h = world["w"], world["h"]
    free = [p for p in range(w*h)
            if not world["walls"][p] and not world["traps"][p]]
    index = [-1] * (w*h)
    for i, p in enumerate(free):
        index[p] = i
    k = len(free)
    if not k or k > 450:
        return {"index": index, "k": k, "dist": {}}

    moves = [[0] * k for _ in range(4)]
    for i, p in enumerate(free):
        for a, (dx, dy) in enumerate(_DXY):
            x, y = p % w + dx, p // w + dy
            q = y * w + x
            moves[a][i] = (index[q] if 0 <= x < w and 0 <= y < h
                           and index[q] >= 0 else i)

    distances = {}
    mirrors = {"H": (0, 1, 3, 2),
               "V": (1, 0, 2, 3),
               "D": (1, 0, 3, 2)}
    for kind in types:
        if time.perf_counter() > deadline:
            break
        size = k*k
        reverse = [[] for _ in range(size)]
        dist = [_INF] * size
        seeds = []
        for i in range(k):
            dist[i*k+i] = 0
        mirror = mirrors[kind]
        aborted = False

        for i, p in enumerate(free):
            if i % 12 == 0 and time.perf_counter() > deadline:
                aborted = True
                break
            base = i*k
            for j, q in enumerate(free):
                if i == j:
                    continue
                source = base+j
                adjacent = p // w == q // w and abs(p-q) == 1
                wins = False
                for a in range(4):
                    ni, nj = moves[a][i], moves[mirror[a]][j]
                    if ni == nj or (adjacent and (ni == j or nj == i)):
                        wins = True
                    else:
                        target = ni*k+nj
                        if target != source:
                            reverse[target].append(source)
                if wins:
                    dist[source] = 1
                    seeds.append(source)
        if aborted:
            break

        queue = deque(seeds)
        while queue:
            q = queue.popleft()
            nd = dist[q] + 1
            for p in reverse[q]:
                if nd < dist[p]:
                    dist[p] = nd
                    queue.append(p)
        distances[kind] = dist

    return {"index": index, "k": k, "dist": distances}


def _heuristic_function(world, pdb):
    w = world["w"]
    index, k, distances = pdb["index"], pdb["k"], pdb["dist"]
    cache = {}

    def pair(i, p, j, q):
        kind = _pair_type(i, j)
        if kind in distances and index[p] >= 0 and index[q] >= 0:
            return distances[kind][index[p]*k + index[q]]
        dx = abs(p % w - q % w)
        dy = abs(p // w - q // w)
        rx = 2 if _SIGNS[i][0] != _SIGNS[j][0] else 1
        ry = 2 if _SIGNS[i][1] != _SIGNS[j][1] else 1
        return (dx + rx - 1)//rx + (dy + ry - 1)//ry

    def heuristic(agents):
        if agents in cache:
            return cache[agents]
        active = [(i, p) for i, p in enumerate(agents) if p >= 0]
        if not active:
            value = 0
        elif len(active) == 2:
            (i, p), (j, q) = active
            value = pair(i, p, j, q)
        elif len(active) == 4:
            def d(a, b):
                i, p = active[a]
                j, q = active[b]
                return pair(i, p, j, q)
            value = min(max(d(0, 1), d(2, 3)),
                        max(d(0, 2), d(1, 3)),
                        max(d(0, 3), d(1, 2)))
        else:
            value = 0
        if len(cache) < 250000:
            cache[agents] = value
        return value

    return heuristic


def _search(world, initial, heuristic, deadline, joint=True,
            ignore_blocks=False, node_limit=220000, weight=1.7):
    start_h = heuristic(initial[0])
    if start_h >= _INF:
        return None
    heap = [(weight*start_h, start_h, 0, 0, initial)]
    best = {initial: 0}
    parent = {}
    serial = 1
    expansions = 0

    while heap and expansions < node_limit:
        if expansions % 128 == 0 and time.perf_counter() > deadline:
            return None
        _, _, _, cost, state = heapq.heappop(heap)
        if best.get(state) != cost:
            continue
        agents, blocks, mode, closed = state
        if all(p < 0 for p in agents):
            path = []
            while state != initial:
                state, action = parent[state]
                path.append(action)
            path.reverse()
            return path

        expansions += 1
        successors = []

        if mode < 0:
            for action in range(1, 5):
                r = _agent_step(world, agents,
                                () if ignore_blocks else blocks,
                                closed, action)
                if r is not None:
                    a, g, _ = r
                    if a != agents or g != closed:
                        successors.append(((a, blocks, -1, g), action))
            if joint and not ignore_blocks:
                for i in range(len(blocks)):
                    successors.append(((agents, blocks, i, closed), ("s", i)))
        else:
            successors.append(((agents, blocks, -1, closed), ("s", -1)))
            if joint and not ignore_blocks:
                for action in range(1, 5):
                    r = _block_step(world, agents, blocks, mode, closed, action)
                    if r is not None:
                        b, g = r
                        successors.append(((agents, b, mode, g), action))
                for i in range(len(blocks)):
                    if i != mode:
                        successors.append(((agents, blocks, i, closed), ("s", i)))

        for nxt, action in successors:
            nc = cost + 1
            if nc >= best.get(nxt, 10**9):
                continue
            hh = heuristic(nxt[0])
            if hh >= _INF:
                continue
            hh += int(nxt[2] >= 0)
            best[nxt] = nc
            parent[nxt] = (state, action)
            heapq.heappush(heap, (nc + weight*hh, hh, serial, nc, nxt))
            serial += 1
    return None


def _parking_plan(world, state, optimistic_plan):
    agents, blocks, mode, closed = state
    w, h = world["w"], world["h"]
    a, g = agents, closed
    usage = Counter(p for p in a if p >= 0)
    first_use = {}
    for step, action in enumerate(optimistic_plan):
        if not isinstance(action, int):
            continue
        r = _agent_step(world, a, (), g, action)
        if r is None:
            return None
        a, g, paired = r
        for p in list(a) + list(paired):
            if p >= 0:
                usage[p] += 1
                first_use.setdefault(p, step)

    needed = [i for i, p in enumerate(blocks) if p in usage]
    needed.sort(key=lambda i: first_use.get(blocks[i], -1))
    for i in needed:
        start = blocks[i]
        forbidden = set(p for p in agents if p >= 0)
        forbidden.update(p for j, p in enumerate(blocks) if j != i)
        queue = deque([start])
        previous = {start: None}
        depth = {start: 0}
        while queue:
            p = queue.popleft()
            for action, (dx, dy) in enumerate(_DXY, 1):
                x, y = p % w + dx, p // w + dy
                if not (0 <= x < w and 0 <= y < h):
                    continue
                q = y*w+x
                if (q in previous or q in forbidden
                        or world["walls"][q] or world["traps"][q]
                        or world["switches"][q]
                        or (world["gates"][q] & closed)):
                    continue
                previous[q] = (p, action)
                depth[q] = depth[p]+1
                queue.append(q)

        destinations = [p for p in previous if p != start]
        if not destinations:
            continue
        destination = min(destinations,
                          key=lambda p: (usage[p] != 0, usage[p], depth[p]))
        if usage[destination] >= usage[start] and usage[destination]:
            continue
        path = []
        p = destination
        while p != start:
            p, action = previous[p]
            path.append(action)
        path.reverse()
        prefix = [] if mode == i else [("s", i)]
        return prefix + path + [("s", -1)]
    return None


def _execute_prediction(world, state, ghosts, action):
    agents, blocks, mode, closed = state
    if isinstance(action, tuple):
        return (agents, blocks, action[1], closed), ghosts
    if action not in (1, 2, 3, 4):
        return state, ghosts
    if mode >= 0:
        r = _block_step(world, agents, blocks, mode, closed, action)
        if r is None:
            return state, ghosts
        b, g = r
        return (agents, b, mode, g), ghosts
    r = _agent_step(world, agents, blocks, closed, action)
    if r is None:
        return state, ghosts
    a, g, paired = r
    return (a, blocks, mode, g), ghosts | frozenset(paired)


def _emit(memory, world, state, hyp, action, available):
    agents, blocks, mode, closed = state
    if isinstance(action, tuple):
        name = "ACTION6"
        i = action[1]
        if i >= 0:
            cell = blocks[i]
        else:
            cell = next((p for p in agents if p >= 0 and p not in blocks), None)
            if cell is None:
                cell = next((p for p in range(world["w"]*world["h"])
                             if p not in blocks), 0)
        s, ox, oy, _, _ = memory["geometry"]
        row = oy + (cell // world["w"])*s + s//2
        col = ox + (cell % world["w"])*s + s//2
    else:
        name = "ACTION" + str(action)
        row = col = None

    if name not in available:
        name = "ACTION5" if "ACTION5" in available else available[0]
        action = 5
        row = col = None

    predicted = []
    for a, ghosts in memory.get("hyps", [hyp]):
        ns, ng = _execute_prediction(world, (a, blocks, mode, closed),
                                    ghosts, action)
        predicted.append((ns[0], ng))
    memory["hyps"] = list(dict.fromkeys(predicted))

    ns, ng = _execute_prediction(world, state, hyp[1], action)
    memory["pending"] = (ns, ng)
    memory["blocks"] = ns[1]
    return name, row, col


def act(obs, available, memory):
    start_time = time.perf_counter()
    deadline = start_time + 8.6
    obs = np.asarray(obs)
    world, observed, blocks, mode, closed, changed = _read(obs, memory)

    hypotheses = [h for h in memory.get("hyps", [])
                  if _matches(h, observed, mode)]
    if not hypotheses:
        hypotheses = _initial_hypotheses(observed, mode)
        memory["plan"] = []
        memory.pop("pending", None)
    if not hypotheses:
        memory["plan"] = []
        return ("ACTION5" if "ACTION5" in available else available[0],
                None, None)
    memory["hyps"] = hypotheses

    # Continue a plan only when its predicted state actually occurred.
    pending = memory.get("pending")
    chosen = None
    if pending is not None:
        predicted_state, ghosts = pending
        ph = (predicted_state[0], ghosts)
        if (ph in hypotheses and predicted_state[1] == blocks
                and predicted_state[2] == mode and not changed):
            chosen = ph
        else:
            memory["plan"] = []
    if chosen is None:
        chosen = hypotheses[0]

    if memory.get("plan") and chosen is not None:
        action = memory["plan"].pop(0)
        state = (chosen[0], blocks, mode, closed)
        return _emit(memory, world, state, chosen, action, available)

    # Resolve indistinguishable mirror identities with informative movements.
    probes = memory.get("probes", 0)
    if len(hypotheses) > 1 and probes < 4 and mode < 0:
        best_action = None
        best_score = None
        for action in range(1, 5):
            groups = Counter()
            invalid = 0
            for agents, ghosts in hypotheses:
                r = _agent_step(world, agents, blocks, closed, action)
                if r is None:
                    invalid += 1
                    continue
                a, _, paired = r
                visible = frozenset(p for p in a if p >= 0) | ghosts | frozenset(paired)
                groups[visible] += 1
            if invalid or len(groups) < 2:
                continue
            score = sum(v*v for v in groups.values())
            if best_score is None or score < best_score:
                best_score, best_action = score, action
        if best_action is not None:
            memory["probes"] = probes + 1
            memory["plan"] = []
            state = (chosen[0], blocks, mode, closed)
            return _emit(memory, world, state, chosen, best_action, available)

    types = {"H"}
    if any(len(a) == 4 for a, _ in hypotheses):
        types.update(("V", "D"))
    pdb = memory.get("pdb")
    if pdb is None or not types.issubset(pdb["dist"]):
        pdb = _build_pdb(world, sorted(types),
                         min(deadline - 1.0, time.perf_counter() + 3.0))
        memory["pdb"] = pdb
    heuristic = _heuristic_function(world, pdb)

    # Prefer an identity assignment with a feasible relaxed pairing.
    if chosen not in hypotheses or heuristic(chosen[0]) >= _INF:
        chosen = min(hypotheses, key=lambda h: heuristic(h[0]))
    state = (chosen[0], blocks, mode, closed)

    remaining = max(0.1, deadline - time.perf_counter())
    if blocks:
        search_deadline = time.perf_counter() + remaining * 0.63
        plan = _search(world, state, heuristic, search_deadline,
                       joint=True, node_limit=150000, weight=1.9)
    else:
        plan = _search(world, state, heuristic, deadline,
                       joint=False, node_limit=300000, weight=1.35)

    if plan is None and blocks and time.perf_counter() < deadline - 0.15:
        optimistic = _search(world, state, heuristic, deadline - 0.08,
                             joint=False, ignore_blocks=True,
                             node_limit=180000, weight=1.45)
        if optimistic is not None:
            park = _parking_plan(world, state, optimistic)
            if park:
                plan = park
            else:
                # Use the initial safe part of the relaxed plan.
                safe = []
                trial = state
                ghosts = chosen[1]
                for action in optimistic:
                    if isinstance(action, int) and trial[2] < 0:
                        real = _agent_step(world, trial[0], trial[1],
                                           trial[3], action)
                        relaxed = _agent_step(world, trial[0], (),
                                              trial[3], action)
                        if real is None or relaxed is None or real[0] != relaxed[0]:
                            break
                    trial, ghosts = _execute_prediction(world, trial, ghosts, action)
                    safe.append(action)
                    if len(safe) >= 8:
                        break
                if safe:
                    plan = safe

    if plan:
        action = plan[0]
        memory["plan"] = plan[1:]
        return _emit(memory, world, state, chosen, action, available)

    # Bounded-search fallback: avoid repeating the same joint configuration.
    memory["plan"] = []
    if mode >= 0:
        return _emit(memory, world, state, chosen, ("s", -1), available)

    visits = memory.setdefault("visits", {})
    key = (chosen[0], blocks, closed)
    visits[key] = visits.get(key, 0) + 1
    candidates = []
    for action in range(1, 5):
        r = _agent_step(world, chosen[0], blocks, closed, action)
        if r is None:
            continue
        a, g, _ = r
        if a == chosen[0] and g == closed:
            continue
        score = heuristic(a) + 3 * visits.get((a, blocks, g), 0)
        candidates.append((score, action))

    if candidates:
        action = min(candidates)[1]
    elif blocks:
        # If every piece is pinned, select a movable obstacle.
        movable = []
        for i in range(len(blocks)):
            for action in range(1, 5):
                r = _block_step(world, chosen[0], blocks, i, closed, action)
                if r is not None:
                    movable.append(i)
                    break
        action = ("s", movable[0]) if movable else 5
    else:
        action = 5

    return _emit(memory, world, state, chosen, action, available)

import numpy as np
import heapq
import math
from collections import deque
from itertools import permutations

# Shape masks, in their unrotated orientations.
_SHAPES = [
    [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
    [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
]
_COLORS = [12, 9, 14, 8]
_MASKS = [
    [np.rot90(np.array(s, dtype=bool), -r) for r in range(4)]
    for s in _SHAPES
]
_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
_ACTIONS = [(1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0)]


def _cell(x, y):
    return y * 12 + x


def _xy(p):
    return p % 12, p // 12


def _decode(a):
    for c, color in enumerate(_COLORS):
        mask = a == color
        if np.count_nonzero(mask) < 5:
            continue
        for s in range(6):
            for r in range(4):
                if np.array_equal(mask, _MASKS[s][r]):
                    return (s, c, r)
    return None


def _inside(rect, x, y):
    a, b, c, d = rect[:4]
    return a <= x <= b and c <= y <= d and (
        x == a or x == b or y == c or y == d
    )


def _move_model(m):
    a, b, c, d, x, y, direction = m
    for q in (direction, (direction - 1) % 4,
              (direction + 1) % 4, (direction + 2) % 4):
        dx, dy = _DIRS[q]
        xx, yy = x + dx, y + dy
        if _inside(m, xx, yy):
            return (a, b, c, d, xx, yy, q)
    return m


def _new_models(old, new):
    x0, y0 = _xy(old)
    x1, y1 = _xy(new)
    if abs(x1 - x0) + abs(y1 - y0) != 1:
        return []
    result = set()
    for a in range(min(x0, x1) + 1):
        for b in range(max(x0, x1), 12):
            for c in range(min(y0, y1) + 1):
                for d in range(max(y0, y1), 11):
                    rect = (a, b, c, d)
                    if not _inside(rect, x0, y0):
                        continue
                    if not _inside(rect, x1, y1):
                        continue
                    for direction in range(4):
                        m = rect + (x0, y0, direction)
                        mm = _move_model(m)
                        if mm[4:6] == (x1, y1):
                            result.add(mm)
    return list(result)


def _model_score(m, walls):
    a, b, c, d = m[:4]
    width, height = b - a, d - c
    score = 2 * (width + height)
    if width and height:
        score += 1.5
    for y in range(c, d + 1):
        for x in range(a, b + 1):
            if _inside(m, x, y):
                if walls[_cell(x, y)] == 1:
                    score += 8
                elif walls[_cell(x, y)] < 0:
                    score += 0.15
    return score


def _select_model(candidates, walls):
    if not candidates:
        return None
    return min(candidates, key=lambda m: (_model_score(m, walls), m))


def _observe(obs, memory):
    # Locate the 5x5 player.
    pos = None
    for y in range(11):
        for x in range(12):
            px, py = 4 + 5 * x, 5 * y
            a = obs[py:py + 5, px:px + 5]
            if a.shape == (5, 5):
                if np.all(a[:2] == 12) and np.all(a[2:] == 9):
                    pos = _cell(x, y)
                    break
        if pos is not None:
            break
    if pos is None:
        pos = memory.get("last_pos", _cell(6, 9))

    attr = _decode(obs[55:61:2, 3:9:2])
    if attr is None:
        attr = tuple(memory.get("last_attr", (5, 1, 0)))

    energy = int(np.count_nonzero(obs[61, 13:55] == 11))
    lives = sum(int(obs[61, x] == 8) for x in (56, 59, 62))
    fog = np.count_nonzero(obs[5:55, 9:59] == 5) > 200

    walls = memory.get("walls")
    if walls is None:
        walls = np.full(144, -1, dtype=np.int8)
        # These cells are occupied by the fixed outer frame/UI.
        for y in range(12):
            walls[_cell(0, y)] = 1
        for x in range(12):
            walls[_cell(x, 11)] = 1
        memory["walls"] = walls

    ax, ay = _xy(pos)
    center_x, center_y = 4 + 5 * ax + 1.5, 5 * ay + 1.5

    def visible(px, py, w=5, h=5):
        if not fog:
            return True
        return all(
            (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 400
            for xx in (px, px + w - 1)
            for yy in (py, py + h - 1)
        )

    found_goals = {}
    found_batteries = set()
    features = [[], [], []]  # shape, colour, rotation
    fully_visible = set()

    shape_pad = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]], bool)
    rot_black = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], bool)
    rot_blue = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], bool)
    color_pad = np.array([[9, 14, 14], [9, 0, 8], [12, 12, 8]])
    battery = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)

    for y in range(11):
        for x in range(12):
            p = _cell(x, y)
            px, py = 4 + 5 * x, 5 * y
            a = obs[py:py + 5, px:px + 5]
            if a.shape != (5, 5):
                continue
            full = visible(px, py)
            if full:
                fully_visible.add(p)
            nwall = int(np.count_nonzero(a == 4))
            if nwall >= 6:
                walls[p] = 1
            elif full:
                walls[p] = 0

            q = a[1:4, 1:4]
            target = _decode(q)
            if target is not None:
                # Targets sit inside a dark 7x7 socket.
                if px > 0 and py > 0:
                    socket = obs[py - 1:py + 6, px - 1:px + 6]
                    if socket.shape == (7, 7):
                        border = np.concatenate(
                            (socket[0], socket[-1],
                             socket[1:-1, 0], socket[1:-1, -1])
                        )
                        if np.count_nonzero((border == 5) | (border == 0)) >= 18:
                            found_goals[p] = target
                            walls[p] = 0

            if np.array_equal(q == 11, battery):
                found_batteries.add(p)
            if np.array_equal(q, color_pad):
                features[1].append(p)
            elif np.array_equal(q == 0, shape_pad) and not np.any(q == 1):
                features[0].append(p)
            elif (np.array_equal(q == 0, rot_black) and
                  np.array_equal(q == 1, rot_blue)):
                features[2].append(p)

    walls[pos] = 0

    goals = memory.setdefault("goals", {})
    for p, target in found_goals.items():
        goals[p] = (target, False)
    for p, value in list(goals.items()):
        target, done = value
        if done:
            continue
        if p in fully_visible and p not in found_goals:
            if p == pos and attr == target:
                goals[p] = (target, True)
            elif p != pos:
                goals[p] = (target, True)

    batteries = memory.setdefault("batteries", {})
    for p in found_batteries:
        batteries[p] = True
    for p in list(batteries):
        if p in fully_visible and p not in found_batteries:
            batteries[p] = False

    # Blue launcher lines point away from their backing wall.
    pushers = memory.setdefault("pushers", {})
    for py in range(55):
        for px in range(4, 60):
            if obs[py, px] != 1:
                continue

            if px + 5 <= 64 and np.all(obs[py, px:px + 5] == 1):
                if px > 0 and obs[py, px - 1] == 1:
                    continue
                above = np.count_nonzero(obs[py - 1, px:px + 5] == 4) if py else 0
                below = np.count_nonzero(obs[py + 1, px:px + 5] == 4)
                if max(above, below) >= 3:
                    dy = 1 if above > below else -1
                    sx, sy = px, py if dy == 1 else py - 4
                    ex, ey = sx, sy + dy
                    if (ex - 4) % 5 == 0 and ey % 5 == 0:
                        cx, cy = (ex - 4) // 5, ey // 5
                        if 0 <= cx < 12 and 0 <= cy < 11:
                            pushers[_cell(cx, cy)] = (0, dy)

            if py + 5 <= 55 and np.all(obs[py:py + 5, px] == 1):
                if py and obs[py - 1, px] == 1:
                    continue
                left = np.count_nonzero(obs[py:py + 5, px - 1] == 4)
                right = np.count_nonzero(obs[py:py + 5, px + 1] == 4)
                if max(left, right) >= 3:
                    dx = 1 if left > right else -1
                    sx, sy = px if dx == 1 else px - 4, py
                    ex, ey = sx + dx, sy
                    if (ex - 4) % 5 == 0 and ey % 5 == 0:
                        cx, cy = (ex - 4) // 5, ey // 5
                        if 0 <= cx < 12 and 0 <= cy < 11:
                            pushers[_cell(cx, cy)] = (dx, 0)

    previous_pos = memory.get("last_pos")
    previous_lives = memory.get("last_lives", lives)
    died = lives < previous_lives
    advanced = previous_pos is not None and pos != previous_pos and not died

    old_energy = memory.get("last_energy")
    if old_energy is not None and not died and energy < old_energy:
        diff = old_energy - energy
        if diff in (1, 2):
            memory["decrement"] = diff
    decrement = memory.get("decrement", 1)

    if died:
        memory["devices"] = []
        memory["plan"] = []
        memory["expect"] = None
        for p, (target, _) in list(goals.items()):
            goals[p] = (target, p not in found_goals and p in fully_visible)

    devices = memory.setdefault("devices", [])
    unused = [set(f) for f in features]
    seen_devices = set()

    for dev in devices:
        kind = dev["kind"]
        old = dev["pos"]
        model = dev.get("model")
        predicted_model = _move_model(model) if advanced and model else model
        predicted = (
            _cell(predicted_model[4], predicted_model[5])
            if predicted_model else old
        )
        new = None
        if unused[kind]:
            xx, yy = _xy(predicted)
            new = min(unused[kind], key=lambda p:
                      abs(p % 12 - xx) + abs(p // 12 - yy))
            unused[kind].remove(new)
            seen_devices.add(dev["id"])

        candidates = dev.get("candidates", [])
        if advanced and candidates:
            candidates = [_move_model(m) for m in candidates]
        if new is not None and candidates:
            nx, ny = _xy(new)
            candidates = [m for m in candidates if m[4:6] == (nx, ny)]

        if new is not None and advanced and new != old and not candidates:
            candidates = _new_models(old, new)

        if candidates:
            model = _select_model(candidates, walls)
            dev["model"] = model
            dev["pos"] = _cell(model[4], model[5])
        else:
            dev["model"] = None
            dev["pos"] = new if new is not None else predicted
        dev["candidates"] = candidates
        if new is not None:
            dev["pos"] = new

    for kind in range(3):
        for p in sorted(unused[kind]):
            ident = memory.get("next_device_id", 0)
            memory["next_device_id"] = ident + 1
            devices.append({
                "id": ident, "kind": kind, "pos": p,
                "model": None, "candidates": []
            })
            seen_devices.add(ident)

    return pos, attr, energy, lives, decrement, fully_visible, seen_devices


def _graph(memory):
    walls = memory["walls"]
    goals = memory["goals"]
    pushers = memory["pushers"]
    stops = {p for p in goals}
    graph = [[] for _ in range(144)]
    transitions = [[] for _ in range(144)]

    for p in range(132):
        if walls[p] == 1:
            continue
        x, y = _xy(p)
        for action, dx, dy in _ACTIONS:
            xx, yy = x + dx, y + dy
            if not (0 <= xx < 12 and 0 <= yy < 11):
                continue
            entry = _cell(xx, yy)
            if walls[entry] == 1:
                continue
            end = entry
            pushed = False
            if entry in pushers:
                ux, uy = pushers[entry]
                for k in range(1, 12):
                    tx, ty = xx + k * ux, yy + k * uy
                    if not (0 <= tx < 12 and 0 <= ty < 12):
                        break
                    q = _cell(tx, ty)
                    if walls[q] == 1 or q in stops:
                        if k > 1:
                            end = _cell(xx + (k - 1) * ux,
                                        yy + (k - 1) * uy)
                            pushed = True
                        break
            transitions[p].append((action, entry, end, pushed))
            graph[p].append(end)

    distances = []
    for start in range(144):
        ds = [999] * 144
        ds[start] = 0
        queue = deque([start])
        while queue:
            p = queue.popleft()
            nd = ds[p] + 1
            for q in graph[p]:
                if ds[q] == 999:
                    ds[q] = nd
                    queue.append(q)
        distances.append(ds)
    return transitions, distances


def _device_timeline(devices, horizon=130):
    states = []
    for dev in devices:
        states.append(dev["model"])
    positions = tuple(dev["pos"] for dev in devices)
    seen = {}
    timeline = []
    prefix, period = 0, 1

    for t in range(horizon + 1):
        key = tuple(states), positions
        if key in seen:
            prefix = seen[key]
            period = t - prefix
            break
        seen[key] = t
        timeline.append(positions)
        next_states, next_positions = [], []
        for i, dev in enumerate(devices):
            m = states[i]
            if m is not None:
                m = _move_model(m)
                next_positions.append(_cell(m[4], m[5]))
            else:
                next_positions.append(positions[i])
            next_states.append(m)
        states = next_states
        positions = tuple(next_positions)
    else:
        prefix, period = len(timeline) - 1, 1

    def time_key(t):
        if t < len(timeline):
            return t
        return prefix + (t - prefix) % period

    return timeline, prefix, period, time_key


def _explore(pos, memory, transitions):
    walls = memory["walls"]
    queue = deque([pos])
    parent = {pos: None}
    best = None
    best_score = -1e9

    while queue:
        p = queue.popleft()
        x, y = _xy(p)
        unknown = 0
        for yy in range(max(0, y - 3), min(11, y + 4)):
            for xx in range(max(1, x - 3), min(12, x + 4)):
                unknown += int(walls[_cell(xx, yy)] < 0)
        depth = 0 if parent[p] is None else parent[p][2]
        score = unknown / (depth + 2)
        if p != pos and score > best_score:
            best, best_score = p, score
        for action, entry, end, pushed in transitions[p]:
            if walls[entry] != 0 or walls[end] != 0 or end in parent:
                continue
            parent[end] = (p, action, depth + 1)
            queue.append(end)

    if best is None:
        choices = transitions[pos]
        if choices:
            return choices[0][0]
        return 1
    while parent[best][0] != pos:
        best = parent[best][0]
    return parent[best][1]


def _search(pos, attr, energy, decrement, memory, allowed):
    transitions, dist = _graph(memory)
    devices = memory["devices"]
    goals = [
        (p, tuple(target))
        for p, (target, done) in memory["goals"].items() if not done
    ]
    if not goals:
        return [], _explore(pos, memory, transitions)

    batteries = sorted(p for p, active in memory["batteries"].items() if active)
    battery_index = {p: i for i, p in enumerate(batteries)}
    goal_index = {p: i for i, (p, _) in enumerate(goals)}
    all_goals = (1 << len(goals)) - 1
    timeline, prefix, period, time_key = _device_timeline(devices)

    def tk(t):
        return t if t < prefix else prefix + (t - prefix) % period

    effects = []
    for positions in timeline:
        at = {}
        for i, p in enumerate(positions):
            counts = at.setdefault(p, [0, 0, 0])
            counts[devices[i]["kind"]] += 1
        effects.append(at)

    station_sets = [set() for _ in range(3)]
    for positions in timeline:
        for i, p in enumerate(positions):
            station_sets[devices[i]["kind"]].add(p)

    to_station = [[999] * 144 for _ in range(3)]
    station_between = [[999] * 3 for _ in range(3)]
    for k in range(3):
        if station_sets[k]:
            for p in range(144):
                to_station[k][p] = min(dist[p][q] for q in station_sets[k])
        for j in range(3):
            if station_sets[k] and station_sets[j]:
                station_between[k][j] = min(
                    dist[p][q] for p in station_sets[k] for q in station_sets[j]
                )

    routes = []
    for gp, target in goals:
        station_to_goal = [
            min((dist[q][gp] for q in station_sets[k]), default=999)
            for k in range(3)
        ]
        table = [[999] * 8 for _ in range(144)]
        for p in range(144):
            table[p][0] = dist[p][gp]
            for mask in range(1, 8):
                kinds = [k for k in range(3) if mask & (1 << k)]
                value = 999
                for order in permutations(kinds):
                    v = to_station[order[0]][p] + station_to_goal[order[-1]]
                    for a, b in zip(order, order[1:]):
                        v += station_between[a][b]
                    value = min(value, v)
                table[p][mask] = value
        routes.append(table)

    mst = {}
    for completed in range(1 << len(goals)):
        remaining = [i for i in range(len(goals))
                     if not completed & (1 << i)]
        if len(remaining) < 2:
            mst[completed] = 0
            continue
        used = {remaining[0]}
        total = 0
        while len(used) < len(remaining):
            value, chosen = min(
                (min(dist[goals[a][0]][goals[b][0]],
                     dist[goals[b][0]][goals[a][0]]), b)
                for a in used for b in remaining if b not in used
            )
            total += value
            used.add(chosen)
        mst[completed] = total

    def heuristic(p, s, c, r, completed):
        if completed == all_goals:
            return 0
        best = 999
        for i, (_, target) in enumerate(goals):
            if completed & (1 << i):
                continue
            ds = (target[0] - s) % 6
            dc = (target[1] - c) % 4
            dr = (target[2] - r) % 4
            mask = int(ds > 0) | (int(dc > 0) << 1) | (int(dr > 0) << 2)
            h = max(routes[i][p][mask], ds + dc + dr)
            best = min(best, h)
        return best + mst[completed]

    start = (pos, attr[0], attr[1], attr[2], 0, 0, 0, energy)
    nodes = [start]
    parents = [(-1, None)]
    depths = [0]
    serial = 0
    h0 = heuristic(pos, *attr, 0)
    heap = [(h0, 0, serial, 0)]
    labels = {}
    best_node = 0
    best_value = h0
    goal_node = None
    expansions = 0
    maximum_energy = max(42, memory.get("max_energy", 42))

    def label_key(state):
        p, s, c, r, gm, bm, t, e = state
        return (p, s, c, r, gm, bm, tk(t))

    labels[label_key(start)] = [(0, energy)]

    while heap and expansions < 85000:
        _, g, _, ni = heapq.heappop(heap)
        state = nodes[ni]
        p, s, c, r, gm, bm, t, e = state
        if g != depths[ni]:
            continue
        if gm == all_goals:
            goal_node = ni
            break
        if t >= 115:
            continue
        expansions += 1
        nt = t + 1
        ti = time_key(nt)
        at = effects[ti]
        devpos = timeline[ti]

        for action, entry, end, pushed in transitions[p]:
            if action not in allowed:
                continue
            gi = goal_index.get(entry)
            if gi is not None and not gm & (1 << gi):
                if (s, c, r) != goals[gi][1]:
                    continue

            ns, nc, nr = s, c, r
            nbm = bm
            ne = e
            recharged = False
            counts = at.get(entry)
            if counts:
                ns = (ns + counts[0]) % 6
                nc = (nc + counts[1]) % 4
                nr = (nr + counts[2]) % 4
            bi = battery_index.get(entry)
            if bi is not None and not nbm & (1 << bi):
                nbm |= 1 << bi
                ne = maximum_energy
                recharged = True
            if not recharged:
                ne -= decrement

            if pushed:
                counts = at.get(end)
                if counts:
                    ns = (ns + counts[0]) % 6
                    nc = (nc + counts[1]) % 4
                    nr = (nr + counts[2]) % 4
                bi = battery_index.get(end)
                if bi is not None and not nbm & (1 << bi):
                    nbm |= 1 << bi
                    ne = maximum_energy

            ngm = gm
            gi = goal_index.get(end)
            if gi is not None and not pushed and (ns, nc, nr) == goals[gi][1]:
                ngm |= 1 << gi
            if ne < 0 and ngm != all_goals:
                continue

            new = (end, ns, nc, nr, ngm, nbm, nt, ne)
            ng = g + 1
            key = label_key(new)
            old_labels = labels.get(key, [])
            if any(gg <= ng and ee >= ne for gg, ee in old_labels):
                continue
            labels[key] = [
                (gg, ee) for gg, ee in old_labels
                if not (ng <= gg and ne >= ee)
            ] + [(ng, ne)]

            h = heuristic(end, ns, nc, nr, ngm)
            serial += 1
            new_index = len(nodes)
            nodes.append(new)
            depths.append(ng)
            prediction = (
                action, end, (ns, nc, nr),
                tuple((dev["id"], devpos[i]) for i, dev in enumerate(devices)),
                max(0, ne)
            )
            parents.append((ni, prediction))
            heapq.heappush(heap, (ng + 1.08 * h, ng, serial, new_index))

            value = h + 0.025 * ng
            if value < best_value:
                best_value = value
                best_node = new_index

    endpoint = goal_node if goal_node is not None else best_node
    plan = []
    while endpoint:
        parent, prediction = parents[endpoint]
        plan.append(prediction)
        endpoint = parent
    plan.reverse()
    if plan:
        return plan, None

    # No useful complete route under the present movement hypothesis.
    # Move toward a known recharge pad or an unexplored region.
    reachable_batteries = [p for p in batteries if dist[pos][p] < 999]
    if reachable_batteries:
        target = min(reachable_batteries, key=lambda p: dist[pos][p])
        choices = [
            (dist[end][target], action)
            for action, entry, end, pushed in transitions[pos]
            if action in allowed
        ]
        if choices:
            return [], min(choices)[1]
    return [], _explore(pos, memory, transitions)


def act(obs, available, memory):
    obs = np.asarray(obs)
    pos, attr, energy, lives, decrement, visible, seen = _observe(obs, memory)
    memory["max_energy"] = max(memory.get("max_energy", 42), energy)

    allowed = {
        i for i in range(1, 5)
        if "ACTION" + str(i) in available
    }
    if not allowed:
        name = "RESET" if "RESET" in available else available[0]
        return name, None, None

    expected = memory.get("expect")
    valid_plan = False
    if expected is not None:
        _, ep, ea, ed, ee = expected
        actual_devices = {d["id"]: d["pos"] for d in memory["devices"]}
        valid_plan = (
            ep == pos and tuple(ea) == attr and ee == energy and
            lives == memory.get("last_lives", lives) and
            all(ident not in seen or actual_devices.get(ident) == p
                for ident, p in ed)
        )

    plan = memory.get("plan", []) if valid_plan else []
    if not plan:
        plan, fallback = _search(
            pos, attr, energy, decrement, memory, allowed
        )
    else:
        fallback = None

    if plan and plan[0][0] in allowed:
        prediction = plan.pop(0)
        action = prediction[0]
        memory["expect"] = prediction
        memory["plan"] = plan
    else:
        action = fallback if fallback in allowed else min(allowed)
        memory["expect"] = None
        memory["plan"] = []

    # A safety valve for persistent perception/model failures.
    signature = (pos, attr, energy, lives)
    recent = memory.setdefault("recent", [])
    recent.append(signature)
    if len(recent) > 14:
        del recent[:-14]

    memory["last_pos"] = pos
    memory["last_attr"] = attr
    memory["last_energy"] = energy
    memory["last_lives"] = lives

    if len(recent) >= 10 and len(set(recent[-10:])) <= 2:
        if "RESET" in available and memory.get("resets", 0) < 1:
            memory.clear()
            memory["resets"] = 1
            return "RESET", None, None

    return "ACTION" + str(action), None, None

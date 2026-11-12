import math
import time
import heapq
from collections import deque

SHAPES = [
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 1)),
]
COLORS = (12, 9, 14, 8)
DIRECTIONS = (
    (0, -1, "ACTION1"),
    (0, 1, "ACTION2"),
    (-1, 0, "ACTION3"),
    (1, 0, "ACTION4"),
)


def rotate(a):
    return tuple(tuple(a[2 - x][y] for x in range(3)) for y in range(3))


PATTERNS = {}
for s, pattern in enumerate(SHAPES):
    for r in range(4):
        PATTERNS[pattern] = (s, r)
        pattern = rotate(pattern)


def index(x, y):
    return y * 12 + x


def xy(i):
    return i % 12, i // 12


def decode_pattern(values, background):
    colors = [int(v) for row in values for v in row if int(v) != background]
    if not colors or colors[0] not in COLORS:
        return None
    color = colors[0]
    if any(v != color for v in colors):
        return None
    mask = tuple(tuple(int(int(v) == color) for v in row) for row in values)
    shape_rotation = PATTERNS.get(mask)
    if shape_rotation is None:
        return None
    return shape_rotation[0], COLORS.index(color), shape_rotation[1]


def find_player(obs):
    for gy in range(12):
        y = 5 * gy
        for gx in range(12):
            x = 4 + 5 * gx
            if (bool((obs[y:y + 2, x:x + 5] == 12).all()) and
                    bool((obs[y + 2:y + 5, x:x + 5] == 9).all())):
                return index(gx, gy)
    return None


def read_avatar(obs):
    values = [
        [int(obs[55 + 2 * r, 3 + 2 * c]) for c in range(3)]
        for r in range(3)
    ]
    return decode_pattern(values, 5)


def read_cell(obs, gx, gy):
    x, y = 4 + 5 * gx, 5 * gy
    tile = obs[y:y + 5, x:x + 5]
    middle = [
        [int(obs[y + 1 + r, x + 1 + c]) for c in range(3)]
        for r in range(3)
    ]

    if int((tile == 5).sum()) >= 12:
        goal = decode_pattern(middle, 5)
        if goal is not None:
            return ("G", goal)

    if int((tile == 4).sum()) >= 7:
        return ("W", None)

    if middle == [[11, 11, 11], [11, 3, 11], [11, 11, 11]]:
        return ("E", None)

    if middle == [[9, 14, 14], [9, 0, 8], [12, 12, 8]]:
        return ("C", None)

    black = tuple(tuple(int(v == 0) for v in row) for row in middle)
    white = tuple(tuple(int(v == 1) for v in row) for row in middle)

    if (black == ((0, 1, 0), (0, 1, 1), (0, 0, 0)) and
            white == ((0, 0, 0), (1, 0, 0), (0, 1, 0))):
        return ("R", None)

    if (black == ((1, 0, 0), (0, 1, 1), (0, 1, 0)) and
            not any(any(row) for row in white)):
        return ("S", None)

    return (".", None)


def update_map(obs, player, memory):
    saved = memory.get("board")
    if not isinstance(saved, list) or len(saved) != 144:
        board = [("?", None) for _ in range(144)]
    else:
        board = [tuple(cell) for cell in saved]

    fog = int((obs[:55, 4:64] == 5).sum()) > 350
    px, py = xy(player)
    visible_switches = {"S": [], "C": [], "R": []}

    for gy in range(12):
        for gx in range(12):
            i = index(gx, gy)
            if i == player:
                if board[i][0] in ("G", "E", "W", "?"):
                    board[i] = (".", None)
                continue

            if fog:
                cx, cy = 4 + 5 * gx + 2, 5 * gy + 2
                player_x, player_y = 4 + 5 * px + 1.5, 5 * py + 1.5
                if math.hypot(cx - player_x, cy - player_y) > 18:
                    continue

            cell = read_cell(obs, gx, gy)
            board[i] = cell
            if cell[0] in visible_switches:
                visible_switches[cell[0]].append(i)

    memory["board"] = board
    return board, visible_switches, fog


def read_pushers(obs):
    result = {}
    for gy in range(12):
        for gx in range(12):
            x, y = 4 + 5 * gx, 5 * gy
            tile = obs[y:y + 5, x:x + 5]
            if gy > 0 and bool((tile[0, :] == 1).all()):
                result[index(gx, gy - 1)] = (0, -1)
            if gy < 11 and bool((tile[4, :] == 1).all()):
                result[index(gx, gy + 1)] = (0, 1)
            if gx > 0 and bool((tile[:, 0] == 1).all()):
                result[index(gx - 1, gy)] = (-1, 0)
            if gx < 11 and bool((tile[:, 4] == 1).all()):
                result[index(gx + 1, gy)] = (1, 0)
    return result


def move(pos, dx, dy, board, pushers):
    x, y = xy(pos)
    x += dx
    y += dy
    if not (0 <= x < 12 and 0 <= y < 12):
        return None
    dest = index(x, y)
    if board[dest][0] == "W":
        return None

    if dest in pushers:
        pdx, pdy = pushers[dest]
        last = dest
        tx, ty = x, y
        found_stop = False
        for _ in range(11):
            tx += pdx
            ty += pdy
            if not (0 <= tx < 12 and 0 <= ty < 12):
                break
            p = index(tx, ty)
            if board[p][0] in ("W", "G"):
                found_stop = True
                break
            last = p
        if found_stop:
            dest = last
    return dest


def find_cycle(history):
    if len(history) < 6:
        return None
    for period in range(2, min(20, len(history) - 2) + 1):
        checks = min(period + 2, len(history) - period)
        if checks < 3:
            continue
        good = True
        for k in range(1, checks + 1):
            a = history[-k]
            b = history[-k - period]
            if a is None or b is None or a != b:
                good = False
                break
        if good:
            cycle = history[-period:]
            if None not in cycle and len(set(cycle)) > 1:
                return cycle
    return None


def update_tracks(visible, player, memory):
    tracks = memory.get("tracks", {})
    for kind in ("S", "C", "R"):
        track = tracks.get(kind, {"history": [], "moving": False})
        hist = list(track.get("history", []))
        positions = visible[kind]

        if len(positions) == 1:
            p = positions[0]
        elif len(positions) == 0 and hist:
            cycle = find_cycle(hist)
            if cycle is not None:
                predicted = cycle[0]
                p = predicted if predicted == player else None
            elif hist[-1] == player:
                p = player
            else:
                p = None
        else:
            p = None

        previous = hist[-1] if hist else None
        if p is not None and previous is not None and p != previous:
            track["moving"] = True

        hist.append(p)
        track["history"] = hist[-64:]
        tracks[kind] = track

    memory["tracks"] = tracks
    return tracks


def make_switch_schedule(board, tracks):
    fixed = {
        kind: {i for i, cell in enumerate(board) if cell[0] == kind}
        for kind in ("S", "C", "R")
    }
    cycles = {}

    for kind in ("S", "C", "R"):
        track = tracks.get(kind, {})
        hist = track.get("history", [])
        if not track.get("moving"):
            continue
        cycle = find_cycle(hist)
        if cycle is not None:
            cycles[kind] = cycle
            # Old positions can remain under the player's sprite.
            fixed[kind].difference_update(p for p in hist if p is not None)
        elif hist and hist[-1] is not None:
            fixed[kind] = {hist[-1]}

    period = 1
    for cycle in cycles.values():
        period = math.lcm(period, len(cycle))
        if period > 120:
            period = 120
            break

    schedule = []
    for t in range(period):
        entries = {}
        for kind in ("S", "C", "R"):
            positions = set(fixed[kind])
            if kind in cycles:
                cycle = cycles[kind]
                # The final history entry is time zero.
                positions.add(cycle[(t - 1) % len(cycle)])
            for p in positions:
                entries.setdefault(p, []).append(kind)
        schedule.append(entries)
    return schedule, period


def make_neighbors(board, pushers):
    neighbors = [[] for _ in range(144)]
    for p in range(144):
        if board[p][0] == "W":
            continue
        for dx, dy, action in DIRECTIONS:
            dest = move(p, dx, dy, board, pushers)
            if dest is not None:
                neighbors[p].append((dest, action))
    return neighbors


def distances_to(target, neighbors):
    reverse = [[] for _ in range(144)]
    for p in range(144):
        for q, _ in neighbors[p]:
            reverse[q].append(p)

    dist = [10000] * 144
    dist[target] = 0
    queue = deque([target])
    while queue:
        q = queue.popleft()
        for p in reverse[q]:
            if dist[p] > dist[q] + 1:
                dist[p] = dist[q] + 1
                queue.append(p)
    return dist


def plan(board, player, avatar, energy, decrement, pushers, tracks):
    goals = [i for i, cell in enumerate(board) if cell[0] == "G"]
    refills = [i for i, cell in enumerate(board) if cell[0] == "E"]
    if not goals:
        return None

    neighbors = make_neighbors(board, pushers)
    schedule, period = make_switch_schedule(board, tracks)
    goal_bits = {p: 1 << i for i, p in enumerate(goals)}
    refill_bits = {p: 1 << i for i, p in enumerate(refills)}
    all_goals = (1 << len(goals)) - 1
    goal_targets = {p: tuple(board[p][1]) for p in goals}
    goal_dist = {g: distances_to(g, neighbors) for g in goals}

    switch_positions = {"S": set(), "C": set(), "R": set()}
    for frame in schedule:
        for p, kinds in frame.items():
            for kind in kinds:
                switch_positions[kind].add(p)

    switch_dist = {}
    for kind, positions in switch_positions.items():
        switch_dist[kind] = [distances_to(p, neighbors) for p in positions]

    def heuristic(pos, shape, color, rotation, done):
        remaining = [g for g in goals if not (done & goal_bits[g])]
        if not remaining:
            return 0

        best = 10000
        for g in remaining:
            ts, tc, tr = goal_targets[g]
            needed = (
                ("S", (ts - shape) % 6),
                ("C", (tc - color) % 4),
                ("R", (tr - rotation) % 4),
            )
            h = goal_dist[g][pos]
            for kind, count in needed:
                if count:
                    positions = list(switch_positions[kind])
                    dists = switch_dist[kind]
                    if positions:
                        via = min(
                            dists[j][pos] + goal_dist[g][p]
                            for j, p in enumerate(positions)
                        )
                        h = max(h, via + max(0, count - 1) * 2)
                    else:
                        h += 20 * count
            best = min(best, h)
        return best

    start = (player, avatar[0], avatar[1], avatar[2], 0, 0, energy, 0)
    start_h = heuristic(player, *avatar, 0)
    queue = [(start_h, 0, 0, start, None)]
    serial = 0
    pareto = {}
    best_partial = None
    best_partial_key = None
    deadline = time.monotonic() + 2.0
    expanded = 0

    while queue and expanded < 110000:
        if expanded % 256 == 0 and time.monotonic() > deadline:
            break

        _, cost, _, state, first = heapq.heappop(queue)
        pos, shape, color, rotation, done, used, left, phase = state
        key = (pos, shape, color, rotation, done, used, phase)
        frontier = pareto.get(key, [])

        if any(c <= cost and e >= left for c, e in frontier):
            continue
        pareto[key] = [
            (c, e) for c, e in frontier
            if not (c >= cost and e <= left)
        ]
        pareto[key].append((cost, left))
        expanded += 1

        if done == all_goals:
            return first

        if first is not None:
            h = heuristic(pos, shape, color, rotation, done)
            partial_key = (-done.bit_count(), h, cost)
            if best_partial_key is None or partial_key < best_partial_key:
                best_partial_key = partial_key
                best_partial = first

        next_phase = (phase + 1) % period
        switch_frame = schedule[next_phase]

        for dest, action in neighbors[pos]:
            ns, nc, nr = shape, color, rotation
            ndone, nused = done, used

            if dest in goal_bits and not (done & goal_bits[dest]):
                if (ns, nc, nr) != goal_targets[dest]:
                    continue
                ndone |= goal_bits[dest]

            for kind in switch_frame.get(dest, ()):
                if kind == "S":
                    ns = (ns + 1) % 6
                elif kind == "C":
                    nc = (nc + 1) % 4
                else:
                    nr = (nr + 1) % 4

            if dest in refill_bits and not (used & refill_bits[dest]):
                nused |= refill_bits[dest]
                new_energy = 42
            else:
                new_energy = left - decrement
                if new_energy < 0:
                    if ndone != all_goals:
                        continue
                    new_energy = 0

            next_state = (
                dest, ns, nc, nr, ndone, nused, new_energy, next_phase
            )
            next_first = first if first is not None else action
            next_cost = cost + 1
            h = heuristic(dest, ns, nc, nr, ndone)
            serial += 1
            heapq.heappush(
                queue,
                (next_cost + 1.2 * h, next_cost, serial,
                 next_state, next_first),
            )

    return best_partial


def shortest_action(player, targets, board, pushers, avoid_goals=True):
    if not targets:
        return None, 10000
    targets = set(targets)
    queue = deque([(player, 0, None)])
    seen = {player}

    while queue:
        pos, dist, first = queue.popleft()
        if pos in targets and first is not None:
            return first, dist

        for dx, dy, action in DIRECTIONS:
            dest = move(pos, dx, dy, board, pushers)
            if dest is None or dest in seen:
                continue
            if avoid_goals and board[dest][0] == "G" and dest not in targets:
                continue
            seen.add(dest)
            queue.append((dest, dist + 1, first or action))

    return None, 10000


def fallback(board, player, avatar, energy, decrement, pushers, memory):
    refills = [i for i, cell in enumerate(board) if cell[0] == "E"]
    refill_action, refill_distance = shortest_action(
        player, refills, board, pushers
    )
    if refill_action is not None and energy <= decrement * refill_distance:
        return refill_action

    goals = [i for i, cell in enumerate(board) if cell[0] == "G"]
    if goals:
        goal = min(goals, key=lambda g: abs(xy(g)[0] - xy(player)[0]) +
                                      abs(xy(g)[1] - xy(player)[1]))
        target = tuple(board[goal][1])
        for kind, current, desired in (
            ("S", avatar[0], target[0]),
            ("C", avatar[1], target[1]),
            ("R", avatar[2], target[2]),
        ):
            if current != desired:
                switches = [i for i, cell in enumerate(board) if cell[0] == kind]
                action, _ = shortest_action(player, switches, board, pushers)
                if action is not None:
                    return action
        action, _ = shortest_action(
            player, [goal], board, pushers, avoid_goals=False
        )
        if action is not None:
            return action

    visits = memory.get("visits")
    if not isinstance(visits, list) or len(visits) != 144:
        visits = [0] * 144
    visits[player] += 1
    memory["visits"] = visits

    queue = deque([(player, 0, None)])
    seen = {player}
    best, best_score = None, -10 ** 9

    while queue:
        pos, dist, first = queue.popleft()
        x, y = xy(pos)
        unknown = 0
        for yy in range(max(0, y - 3), min(12, y + 4)):
            for xx in range(max(0, x - 3), min(12, x + 4)):
                if board[index(xx, yy)][0] == "?":
                    unknown += 1

        if first is not None:
            score = 2 * unknown - dist - 4 * visits[pos]
            if score > best_score:
                best_score, best = score, first

        for dx, dy, action in DIRECTIONS:
            dest = move(pos, dx, dy, board, pushers)
            if dest is None or dest in seen or board[dest][0] == "G":
                continue
            seen.add(dest)
            queue.append((dest, dist + 1, first or action))
    return best


def act(obs, available, memory):
    player = find_player(obs)
    avatar = read_avatar(obs)

    if player is None:
        player = memory.get("player")
    if player is None:
        action = "ACTION1" if "ACTION1" in available else available[0]
        return action, None, None

    if avatar is None:
        avatar = memory.get("avatar", (0, 0, 0))

    board, visible_switches, fog = update_map(obs, player, memory)
    tracks = update_tracks(visible_switches, player, memory)
    pushers = read_pushers(obs)

    energy = sum(int(obs[61, x]) == 11 for x in range(13, 55))
    old_energy = memory.get("energy")
    if old_energy is not None:
        difference = old_energy - energy
        if difference in (1, 2):
            memory["decrement"] = difference
    decrement = memory.get("decrement", 1)

    memory["player"] = player
    memory["avatar"] = avatar
    memory["energy"] = energy

    action = plan(board, player, avatar, energy, decrement, pushers, tracks)

    refill_positions = [i for i, cell in enumerate(board) if cell[0] == "E"]
    recharge_action, recharge_distance = shortest_action(
        player, refill_positions, board, pushers
    )
    if (recharge_action is not None and
            energy <= decrement * recharge_distance):
        action = recharge_action

    if action is None:
        action = fallback(
            board, player, avatar, energy, decrement, pushers, memory
        )

    if action not in available:
        action = next(
            (a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
             if a in available),
            available[0],
        )
    return action, None, None

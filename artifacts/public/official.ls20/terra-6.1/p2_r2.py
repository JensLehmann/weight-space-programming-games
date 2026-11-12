import heapq
from collections import deque

import numpy as np


_SHAPES = [
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 1)),
]
_COLORS = (12, 9, 14, 8)
_DIRS = (
    (0, -5, "ACTION1"),
    (0, 5, "ACTION2"),
    (-5, 0, "ACTION3"),
    (5, 0, "ACTION4"),
)
_TRACK_DIRS = ((0, 5), (5, 0), (0, -5), (-5, 0))


def _rotate_cw(a):
    return tuple(tuple(a[2 - x][y] for x in range(3)) for y in range(3))


_PATTERNS = {}
for _shape, _base in enumerate(_SHAPES):
    _p = _base
    for _rotation in range(4):
        _PATTERNS[_p] = (_shape, _rotation)
        _p = _rotate_cw(_p)


def _icon(a, background):
    colors = {
        int(v) for row in a for v in row if int(v) != background
    }
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color not in _COLORS:
        return None
    mask = tuple(
        tuple(1 if int(v) == color else 0 for v in row)
        for row in a
    )
    result = _PATTERNS.get(mask)
    if result is None:
        return None
    return result[0], _COLORS.index(color), result[1]


def _inside(x, y):
    return (
        4 <= x <= 59
        and 0 <= y <= 55
        and (x - 4) % 5 == 0
        and y % 5 == 0
    )


def _distance(a, b):
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) // 5


def _find_player(frame, old=None):
    candidates = []
    for y in range(0, 56, 5):
        for x in range(4, 60, 5):
            tile = frame[y:y + 5, x:x + 5]
            score = int(np.count_nonzero(tile[:2] == 12))
            score += int(np.count_nonzero(tile[2:] == 9))
            if score >= 21:
                distance = 0 if old is None else _distance((x, y), old)
                candidates.append((-score, distance, x, y))
    if not candidates:
        return old
    candidates.sort()
    return candidates[0][2], candidates[0][3]


def _switch_kind(a):
    t = tuple(tuple(int(v) for v in row) for row in a)
    if t == ((3, 0, 3), (1, 0, 0), (3, 1, 3)):
        return 2
    if t == ((0, 3, 3), (3, 0, 0), (3, 0, 3)):
        return 0
    if t == ((9, 14, 14), (9, 0, 8), (12, 12, 8)):
        return 1
    return None


def _is_refill(a):
    t = tuple(tuple(int(v) for v in row) for row in a)
    return t == ((11, 11, 11), (11, 3, 11), (11, 11, 11))


def _track_next(track, pos, direction, board):
    if not track.get("moving") or direction == (0, 0):
        return pos, direction

    key = (pos[0], pos[1], direction[0], direction[1])
    known = track.get("edges", {}).get(key)
    if known is not None:
        nxt = tuple(known)
        return nxt, (nxt[0] - pos[0], nxt[1] - pos[1])

    bad = track.get("bad", set())
    seen = track.get("seen", set())
    line = track.get("line")

    try:
        index = _TRACK_DIRS.index(direction)
    except ValueError:
        return pos, direction

    order = (
        index,
        (index - 1) % 4,
        (index + 1) % 4,
        (index + 2) % 4,
    )

    if line is not None:
        # A reversal observed on a straight track rules out side turns.
        order = (index, (index + 2) % 4)

    for i in order:
        dx, dy = _TRACK_DIRS[i]
        nxt = (pos[0] + dx, pos[1] + dy)
        if nxt in bad or not _inside(*nxt):
            continue
        if line == "horizontal" and dy:
            continue
        if line == "vertical" and dx:
            continue
        if board.get(nxt) == 1 and nxt not in seen:
            continue
        return nxt, (dx, dy)

    return pos, direction


def _update_tracks(memory, detections, player, board):
    tracks = memory.setdefault("tracks", [])
    used = set()

    for track in tracks:
        old = tuple(track["pos"])
        direction = tuple(track.get("direction", (0, 0)))
        predicted, _ = _track_next(track, old, direction, board)
        choices = []

        for i, (kind, pos) in enumerate(detections):
            if i in used or kind != track["kind"]:
                continue
            distance = _distance(old, pos)
            if distance <= 2:
                preference = 0 if track.get("moving") and pos == predicted else 1
                choices.append((preference, distance, i, pos))

        if choices:
            choices.sort()
            _, distance, i, pos = choices[0]
            used.add(i)

            if pos != old and distance == 1:
                new_direction = (pos[0] - old[0], pos[1] - old[1])
                track["moving"] = True
                track.setdefault("seen", set()).add(old)
                track["seen"].add(pos)

                if direction != (0, 0):
                    key = (old[0], old[1], direction[0], direction[1])
                    track.setdefault("edges", {})[key] = pos

                    if new_direction != direction:
                        track.setdefault("bad", set()).add(
                            (old[0] + direction[0], old[1] + direction[1])
                        )
                        if new_direction == (-direction[0], -direction[1]):
                            seen = track["seen"]
                            if len({p[1] for p in seen}) == 1:
                                track["line"] = "horizontal"
                            elif len({p[0] for p in seen}) == 1:
                                track["line"] = "vertical"

                track["direction"] = new_direction

            track["pos"] = pos
            track["missing"] = 0

        elif track.get("moving"):
            # The switch may be hidden beneath the player or in fog.
            if predicted == player or _distance(old, player) > 1:
                track["pos"] = predicted
            elif _distance(old, player) == 1:
                track["pos"] = player
            track["missing"] = track.get("missing", 0) + 1

    for i, (kind, pos) in enumerate(detections):
        if i not in used:
            tracks.append({
                "kind": kind,
                "pos": pos,
                "direction": (0, 0),
                "moving": False,
                "edges": {},
                "seen": {pos},
                "bad": set(),
                "missing": 0,
            })


def _switch_schedule(tracks, board, horizon):
    schedule = [dict() for _ in range(horizon + 1)]
    dynamic = False

    for track in tracks:
        kind = track["kind"]
        pos = tuple(track["pos"])
        direction = tuple(track.get("direction", (0, 0)))
        moving = bool(track.get("moving"))
        dynamic |= moving

        for t in range(1, horizon + 1):
            if moving:
                pos, direction = _track_next(track, pos, direction, board)
            schedule[t].setdefault(pos, []).append(kind)

    return schedule, dynamic


def _push_destination(pos, pushes, board, goals):
    direction = pushes.get(pos)
    if direction is None:
        return pos

    dx, dy = direction
    x, y = pos
    last = pos
    for _ in range(11):
        x += dx
        y += dy
        if not _inside(x, y):
            break
        if board.get((x, y)) == 1 or (x, y) in goals:
            return last
        last = (x, y)

    # A pusher needs a known stopping wall to activate.
    return pos


def _search(start, attributes, energy, decrement, capacity, board, goals,
            refills, pushes, tracks):
    if not goals:
        return None

    horizon = 75
    schedule, dynamic = _switch_schedule(tracks, board, horizon)
    refill_list = list(refills)[:16]
    refill_bit = {p: 1 << i for i, p in enumerate(refill_list)}
    goal_items = list(goals.items())

    switch_positions = {0: [], 1: [], 2: []}
    for track in tracks:
        switch_positions[track["kind"]].append(tuple(track["pos"]))

    def heuristic(pos, attrs):
        best = 10 ** 6
        for goal_pos, goal_attrs in goal_items:
            h = _distance(pos, goal_pos)
            for kind in range(3):
                if attrs[kind] != goal_attrs[kind] and switch_positions[kind]:
                    d = min(
                        _distance(pos, switch_pos)
                        + _distance(switch_pos, goal_pos)
                        for switch_pos in switch_positions[kind]
                    )
                    h = max(h, d)
            best = min(best, h)
        return best

    initial = (
        start[0], start[1],
        attributes[0], attributes[1], attributes[2],
        0, 0, energy,
    )
    queue = []
    serial = 0
    heapq.heappush(
        queue,
        (1.4 * heuristic(start, attributes), 0, serial, initial, None),
    )
    best_energy = {}
    expanded = 0

    while queue and expanded < 70000:
        _, cost, _, state, first_action = heapq.heappop(queue)
        x, y, shape, color, rotation, used, t, left = state

        key = (
            x, y, shape, color, rotation, used,
            t if dynamic else 0,
        )
        if best_energy.get(key, -1) >= left:
            continue
        best_energy[key] = left
        expanded += 1

        if t >= horizon:
            continue

        next_t = t + 1
        switches = schedule[next_t]

        for dx, dy, action in _DIRS:
            dest = (x + dx, y + dy)
            if not _inside(*dest) or board.get(dest) == 1:
                continue

            attrs = (shape, color, rotation)
            target = goals.get(dest)
            if target is not None and attrs != target:
                continue

            ns, nc, nr = attrs
            for kind in switches.get(dest, ()):
                if kind == 0:
                    ns = (ns + 1) % 6
                elif kind == 1:
                    nc = (nc + 1) % 4
                else:
                    nr = (nr + 1) % 4

            nused = used
            nleft = left
            bit = refill_bit.get(dest, 0)
            if bit and not (nused & bit):
                nused |= bit
                nleft = capacity
            else:
                nleft -= decrement
                if nleft < 0:
                    continue

            final = _push_destination(dest, pushes, board, goals)
            if final != dest:
                for kind in switches.get(final, ()):
                    if kind == 0:
                        ns = (ns + 1) % 6
                    elif kind == 1:
                        nc = (nc + 1) % 4
                    else:
                        nr = (nr + 1) % 4
                bit = refill_bit.get(final, 0)
                if bit and not (nused & bit):
                    nused |= bit
                    nleft = capacity

            first = action if first_action is None else first_action
            if target is not None and final == dest:
                return first

            nstate = (
                final[0], final[1], ns, nc, nr,
                nused, next_t, nleft,
            )
            nkey = (
                final[0], final[1], ns, nc, nr, nused,
                next_t if dynamic else 0,
            )
            if best_energy.get(nkey, -1) >= nleft:
                continue

            serial += 1
            h = heuristic(final, (ns, nc, nr))
            heapq.heappush(
                queue,
                (cost + 1 + 1.4 * h, cost + 1, serial, nstate, first),
            )

    return None


def _explore(start, board, goals, attrs, tracks, refills, energy):
    queue = deque([(start, None, 0)])
    seen = {start}
    candidates = []
    switch_cells = {tuple(track["pos"]) for track in tracks}

    while queue:
        pos, first, distance = queue.popleft()

        if pos != start:
            if pos not in board:
                unknown = 0
                for dx, dy, _ in _DIRS:
                    nxt = (pos[0] + dx, pos[1] + dy)
                    if _inside(*nxt) and nxt not in board:
                        unknown += 1
                candidates.append((distance - 0.35 * unknown, first))
            elif pos in refills and energy < 15:
                candidates.append((distance - 4, first))
            elif pos in switch_cells:
                candidates.append((distance + 8, first))

        for dx, dy, action in _DIRS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or not _inside(*nxt) or board.get(nxt) == 1:
                continue
            if nxt in goals and goals[nxt] != attrs:
                continue
            seen.add(nxt)
            queue.append((
                nxt,
                action if first is None else first,
                distance + 1,
            ))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return None


def _alternate_action(player, attrs, board, goals, visits, previous, available):
    choices = []
    for dx, dy, action in _DIRS:
        if action not in available:
            continue
        pos = (player[0] + dx, player[1] + dy)
        if not _inside(*pos) or board.get(pos) == 1:
            continue
        if pos in goals and goals[pos] != attrs:
            continue
        penalty = visits.get(pos, 0)
        if pos == previous:
            penalty += 3
        choices.append((penalty, action))
    if not choices:
        return None
    choices.sort()
    return choices[0][1]


def act(obs, available, memory):
    frame = np.asarray(obs)

    capacity = 0
    energy = 0
    for x in range(13, 55):
        value = int(frame[61, x])
        if value not in (3, 11):
            break
        capacity += 1
        if value == 11:
            energy += 1
    if capacity == 0:
        capacity, energy = 42, 42

    # A partial starting state can be normalized to the level's clean start.
    if not memory.get("initialized"):
        memory["initialized"] = True
        if energy < capacity:
            memory["resets"] = 1
            return "RESET", None, None

    old_player = memory.get("player")
    if old_player is not None:
        old_player = tuple(old_player)

    player = _find_player(frame, old_player)
    if player is None:
        action = next(
            (name for _, _, name in _DIRS if name in available),
            available[0],
        )
        return action, None, None

    ui = [
        [int(frame[55 + 2 * y, 3 + 2 * x]) for x in range(3)]
        for y in range(3)
    ]
    attributes = _icon(ui, 5)
    if attributes is None:
        attributes = tuple(memory.get("attributes", (0, 0, 0)))

    previous_energy = memory.get("energy")
    decrement = memory.get("decrement", 2)
    if previous_energy is not None:
        difference = previous_energy - energy
        if difference in (1, 2):
            decrement = difference

    lives = sum(int(frame[61, x]) == 8 for x in (56, 59, 62))
    previous_lives = memory.get("lives")

    board = memory.setdefault("board", {})
    goals = memory.setdefault("goals", {})
    refills = memory.setdefault("refills", set())
    pushes = memory.setdefault("pushes", {})
    all_goals = memory.setdefault("all_goals", {})
    all_refills = memory.setdefault("all_refills", set())

    if previous_lives is not None and lives < previous_lives:
        goals.clear()
        goals.update(all_goals)
        refills.clear()
        refills.update(all_refills)
        memory["tracks"] = []
        memory["states_seen"] = {}
        old_player = None

    if old_player is not None and old_player != player:
        if player in goals:
            old_attrs = tuple(memory.get("attributes", attributes))
            if old_attrs == tuple(goals[player]):
                del goals[player]
                memory["states_seen"] = {}
        if player in refills:
            refills.discard(player)

    last_action = memory.get("last_action")
    if old_player == player and last_action:
        for dx, dy, name in _DIRS:
            if name == last_action:
                attempted = (player[0] + dx, player[1] + dy)
                if (
                    _inside(*attempted)
                    and attempted not in pushes
                    and attempted not in goals
                ):
                    board[attempted] = 1
                break

    # In fog, distant pixels are replaced by color 5.
    distant = 0
    distant_black = 0
    cx, cy = player[0] + 1.5, player[1] + 1.5
    for y in range(0, 55, 3):
        for x in range(4, 64, 3):
            if (x - cx) ** 2 + (y - cy) ** 2 > 22 ** 2:
                distant += 1
                if int(frame[y, x]) == 5:
                    distant_black += 1
    if distant and distant_black > 0.85 * distant:
        memory["fog"] = True
    fog = bool(memory.get("fog", False))

    def fully_visible(x, y):
        if not fog:
            return True
        return all(
            (xx - cx) ** 2 + (yy - cy) ** 2 <= 20 ** 2
            for xx, yy in (
                (x, y), (x + 4, y),
                (x, y + 4), (x + 4, y + 4),
            )
        )

    detections = []
    for y in range(0, 56, 5):
        for x in range(4, 60, 5):
            pos = (x, y)
            if pos == player or not fully_visible(x, y):
                continue

            tile = frame[y:y + 5, x:x + 5]
            center = tile[1:4, 1:4]
            border = (
                list(tile[0, :]) + list(tile[4, :])
                + list(tile[1:4, 0]) + list(tile[1:4, 4])
            )

            if all(int(v) == 5 for v in border):
                icon = _icon(center, 5)
                if icon is not None:
                    goals[pos] = icon
                    all_goals[pos] = icon
                    board[pos] = 0
                    continue

            kind = _switch_kind(center)
            if kind is not None:
                detections.append((kind, pos))
            if _is_refill(center):
                refills.add(pos)
                all_refills.add(pos)

            board[pos] = 1 if int(np.count_nonzero(tile == 4)) >= 10 else 0

    # Detect the thin pusher walls and their directions.
    for y in range(0, 56, 5):
        for x in range(4, 63):
            if not fully_visible(min(x, 59), y):
                continue
            if all(int(frame[y + k, x]) == 1 for k in range(5)):
                if x > 0 and int(frame[y + 2, x - 1]) == 4:
                    pos = (x + 1, y)
                    if _inside(*pos):
                        pushes[pos] = (5, 0)
                if x < 63 and int(frame[y + 2, x + 1]) == 4:
                    pos = (x - 5, y)
                    if _inside(*pos):
                        pushes[pos] = (-5, 0)

    for x in range(4, 60, 5):
        for y in range(0, 59):
            if not fully_visible(x, min(y, 55)):
                continue
            if all(int(frame[y, x + k]) == 1 for k in range(5)):
                if y > 0 and int(frame[y - 1, x + 2]) == 4:
                    pos = (x, y + 1)
                    if _inside(*pos):
                        pushes[pos] = (0, 5)
                if y < 63 and int(frame[y + 1, x + 2]) == 4:
                    pos = (x, y - 5)
                    if _inside(*pos):
                        pushes[pos] = (0, -5)

    _update_tracks(memory, detections, player, board)
    tracks = memory["tracks"]

    action = _search(
        player, attributes, energy, decrement, capacity,
        board, goals, refills, pushes, tracks,
    )
    if action is None:
        action = _explore(
            player, board, goals, attributes, tracks, refills, energy,
        )

    if action is None and energy <= decrement * 2:
        if memory.get("acted", 0) > 0 and memory.get("resets", 0) < 2:
            resets = memory.get("resets", 0) + 1
            memory.clear()
            memory["initialized"] = True
            memory["resets"] = resets
            return "RESET", None, None

    if action not in available:
        action = next(
            (name for _, _, name in _DIRS if name in available),
            available[0],
        )

    # If the same full visible state recurs, try a different route.
    track_state = tuple(sorted(
        (
            track["kind"],
            tuple(track["pos"]),
            tuple(track.get("direction", (0, 0))),
        )
        for track in tracks
    ))
    fingerprint = (
        player, attributes, track_state,
        tuple(sorted(goals)), tuple(sorted(refills)),
    )
    states_seen = memory.setdefault("states_seen", {})
    count = states_seen.get(fingerprint, 0) + 1
    states_seen[fingerprint] = count

    visits = memory.setdefault("visits", {})
    visits[player] = visits.get(player, 0) + 1
    if count >= 3:
        alternative = _alternate_action(
            player, attributes, board, goals, visits, old_player, available,
        )
        if alternative is not None:
            action = alternative
        states_seen[fingerprint] = 0

    memory["player"] = player
    memory["attributes"] = attributes
    memory["energy"] = energy
    memory["decrement"] = decrement
    memory["lives"] = lives
    memory["last_action"] = action
    memory["acted"] = memory.get("acted", 0) + 1

    return action, None, None

import heapq
from collections import deque
from functools import lru_cache

import numpy as np


_SHAPES = (
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 1)),
)
_COLORS = (12, 9, 14, 8)
_DIRS = (
    (0, -5, "ACTION1"),
    (0, 5, "ACTION2"),
    (-5, 0, "ACTION3"),
    (5, 0, "ACTION4"),
)


def _rotate(mask):
    return tuple(
        tuple(mask[2 - x][y] for x in range(3))
        for y in range(3)
    )


_MASKS = []
for _shape in _SHAPES:
    _rotations = []
    for _ in range(4):
        _rotations.append(_shape)
        _shape = _rotate(_shape)
    _MASKS.append(tuple(_rotations))
_MASKS = tuple(_MASKS)


def _matches(mask):
    return [
        (s, r)
        for s in range(6)
        for r in range(4)
        if _MASKS[s][r] == mask
    ]


def _indicator(frame):
    a = frame[55:61:2, 3:9:2]
    if a.shape != (3, 3):
        return [], None

    values = [int(v) for v in a.flat if int(v) in _COLORS]
    if not values:
        return [], None

    color = max(set(values), key=values.count)
    mask = tuple(
        tuple(int(a[y, x]) == color for x in range(3))
        for y in range(3)
    )
    return _matches(mask), _COLORS.index(color)


def _player(frame):
    for y in range(0, 60, 5):
        for x in range(4, 60, 5):
            a = frame[y:y + 5, x:x + 5]
            if (
                a.shape == (5, 5)
                and np.all(a[:2] == 12)
                and np.all(a[2:] == 9)
            ):
                return (x, y)
    return None


def _goal_at(a):
    if a.shape != (5, 5):
        return None

    if not (
        np.all(a[0, :] == 5)
        and np.all(a[4, :] == 5)
        and np.all(a[:, 0] == 5)
        and np.all(a[:, 4] == 5)
    ):
        return None

    inner = a[1:4, 1:4]
    colors = [int(v) for v in inner.flat if int(v) in _COLORS]
    if not colors or len(set(colors)) != 1:
        return None

    color = colors[0]
    if any(int(v) not in (5, color) for v in inner.flat):
        return None

    mask = tuple(
        tuple(int(inner[y, x]) == color for x in range(3))
        for y in range(3)
    )
    choices = _matches(mask)
    if not choices:
        return None
    return (_COLORS.index(color), choices)


def _tile(frame, x, y):
    a = frame[y:y + 5, x:x + 5]
    if a.shape != (5, 5):
        return "?"

    if _goal_at(a) is not None:
        return "G"

    if (
        int(a[1, 1]) == 0
        and int(a[2, 2]) == 0
        and int(a[2, 3]) == 0
        and int(a[3, 2]) == 0
    ):
        return "S"

    if (
        int(a[1, 2]) == 0
        and int(a[2, 1]) == 1
        and int(a[2, 2]) == 0
        and int(a[2, 3]) == 0
        and int(a[3, 2]) == 1
    ):
        return "R"

    if (
        int(a[1, 1]) == 9
        and int(a[1, 2]) == 14
        and int(a[1, 3]) == 14
        and int(a[2, 1]) == 9
        and int(a[2, 2]) == 0
        and int(a[2, 3]) == 8
        and int(a[3, 1]) == 12
        and int(a[3, 2]) == 12
        and int(a[3, 3]) == 8
    ):
        return "C"

    ring = a[1:4, 1:4]
    if (
        ring.shape == (3, 3)
        and all(
            int(ring[yy, xx]) == 11
            for yy in range(3)
            for xx in range(3)
            if (yy, xx) != (1, 1)
        )
    ):
        return "P"

    if int(np.count_nonzero(a == 4)) >= 6:
        return "#"
    return "."


def _meter(frame):
    row = [int(v) for v in frame[61, 13:55]]
    active = sum(v == 11 for v in row)
    valid = [i for i, v in enumerate(row) if v in (3, 11)]
    if not valid:
        return 42, 42
    return active, max(valid) + 1


def _lives(frame):
    # Each life indicator is a 2x2 block, rendered independently of fog.
    values = []
    for x in (56, 59, 62):
        block = frame[61:63, x:x + 2]
        if block.shape != (2, 2) or not np.all(block == block[0, 0]):
            return None
        v = int(block[0, 0])
        if v not in (3, 8):
            return None
        values.append(v)
    return sum(v == 8 for v in values)


def _in_bounds(p):
    x, y = p
    return 4 <= x <= 59 and 0 <= y <= 55


def _track_step(rec, pos, direction, board):
    x, y = pos
    dx, dy = direction

    edge = rec.get("edges", {}).get((x, y, dx, dy))
    if edge is not None:
        return (edge[0], edge[1]), (edge[2], edge[3])

    if dx == 0 and dy == 0:
        return pos, direction

    nxt = (x + dx, y + dy)
    if not _in_bounds(nxt) or board.get(nxt) == "#":
        dx, dy = -dx, -dy
        nxt = (x + dx, y + dy)
        if not _in_bounds(nxt) or board.get(nxt) == "#":
            return pos, (dx, dy)
    return nxt, (dx, dy)


def _read_board(frame, player, memory):
    board = memory.setdefault("board", {})
    goals = memory.setdefault("goals", {})
    done = memory.setdefault("done", set())
    dyn = memory.setdefault("dynamic", {})
    old_positions = memory.get("symbol_positions", {})

    fog = int(np.count_nonzero(frame[:55] == 5)) > 1600
    found = {"S": [], "C": [], "R": []}

    for y in range(0, 60, 5):
        for x in range(4, 60, 5):
            p = (x, y)
            if p == player:
                continue

            if fog:
                dist2 = (
                    (x + 2 - (player[0] + 1.5)) ** 2
                    + (y + 2 - (player[1] + 1.5)) ** 2
                )
                if dist2 > 17.5 ** 2:
                    continue

            kind = _tile(frame, x, y)
            board[p] = kind

            if kind in found:
                found[kind].append(p)

            if kind == "G" and p not in done:
                goal = _goal_at(frame[y:y + 5, x:x + 5])
                if goal is not None:
                    goals[p] = goal
            elif p in goals and kind != "G" and p in done:
                goals.pop(p, None)

    positions = {}
    for kind in ("S", "C", "R"):
        visible = found[kind]
        previous = old_positions.get(kind, [])
        rec = dyn.get(kind)

        if rec is None and len(previous) == 1 and len(visible) == 1:
            a, b = previous[0], visible[0]
            delta = (b[0] - a[0], b[1] - a[1])
            if a != b and abs(delta[0]) + abs(delta[1]) == 5:
                rec = {"pos": b, "dir": delta, "edges": {}}
                dyn[kind] = rec
                if board.get(a) == kind:
                    board[a] = "."

        if rec is not None:
            old = rec["pos"]
            olddir = rec["dir"]

            if visible:
                new = min(
                    visible,
                    key=lambda q: abs(q[0] - old[0]) + abs(q[1] - old[1]),
                )
            else:
                predicted, _ = _track_step(rec, old, olddir, board)
                new = predicted if predicted == player else old

            delta = (new[0] - old[0], new[1] - old[1])
            if abs(delta[0]) + abs(delta[1]) == 5:
                rec["edges"][(old[0], old[1], olddir[0], olddir[1])] = (
                    new[0], new[1], delta[0], delta[1]
                )
                rec["pos"] = new
                rec["dir"] = delta
                if old != new and board.get(old) == kind:
                    board[old] = "."
            elif new != old:
                rec["pos"] = new

            positions[kind] = [rec["pos"]]
        else:
            places = list(visible)
            for p in previous:
                if p == player and p not in places:
                    places.append(p)
            positions[kind] = places

    memory["symbol_positions"] = positions
    return board, goals, dyn, fog


def _pushers(frame, player, board, memory, fog):
    push = memory.setdefault("pushers", {})

    for y in range(0, 60, 5):
        for x in range(4, 60, 5):
            p = (x, y)
            if board.get(p) == "#" or p == player:
                continue

            if fog:
                dist2 = (x + 2 - player[0]) ** 2 + (y + 2 - player[1]) ** 2
                if dist2 > 17 ** 2:
                    continue

            direction = None
            if y > 0 and np.all(frame[y - 1, x:x + 5] == 1):
                direction = (0, 5)
            elif y + 5 < 64 and np.all(frame[y + 5, x:x + 5] == 1):
                direction = (0, -5)
            elif x > 0 and np.all(frame[y:y + 5, x - 1] == 1):
                direction = (5, 0)
            elif x + 5 < 64 and np.all(frame[y:y + 5, x + 5] == 1):
                direction = (-5, 0)

            if direction is not None:
                push[p] = direction
            elif p in push:
                del push[p]

    return push


def _push_destination(p, push, board, goals):
    if p not in push:
        return p

    dx, dy = push[p]
    q = p
    for _ in range(11):
        nxt = (q[0] + dx, q[1] + dy)
        if not _in_bounds(nxt):
            return p
        if board.get(nxt) == "#" or nxt in goals:
            return q
        q = nxt
    return p


def _trajectories(dyn, board, horizon):
    result = {}
    for kind, rec in dyn.items():
        pos, direction = rec["pos"], rec["dir"]
        seq = [pos]
        for _ in range(horizon):
            pos, direction = _track_step(rec, pos, direction, board)
            seq.append(pos)
        result[kind] = seq
    return result


def _search(
    player, shape, color, rot, energy, maximum, decrement,
    board, goals, dyn, push, denied, available
):
    if not goals:
        return None

    goal_items = []
    for p, (gc, options) in goals.items():
        opts = [
            (s, r) for s, r in options
            if (p, s, r) not in denied
        ]
        if opts:
            goal_items.append((p, gc, opts))

    if not goal_items:
        return None

    pickups = [p for p, k in board.items() if k == "P"]
    pickups = sorted(
        pickups,
        key=lambda p: abs(p[0] - player[0]) + abs(p[1] - player[1]),
    )[:12]
    pickup_bit = {p: 1 << i for i, p in enumerate(pickups)}

    static = {"S": set(), "C": set(), "R": set()}
    for p, kind in board.items():
        if kind in static and kind not in dyn:
            static[kind].add(p)

    horizon = 90
    traj = _trajectories(dyn, board, horizon)
    moving = bool(traj)
    allowed = set(available)
    directions = [
        (dx, dy, action)
        for dx, dy, action in _DIRS
        if action in allowed
    ]

    @lru_cache(maxsize=100000)
    def heuristic(x, y, s, c, r):
        best = 10 ** 6
        for (gx, gy), gc, opts in goal_items:
            dist = (abs(gx - x) + abs(gy - y)) // 5
            for gs, gr in opts:
                changes = (
                    (gs - s) % 6
                    + (gc - c) % 4
                    + (gr - r) % 4
                )
                best = min(best, dist + 2.0 * changes)
        return best

    start = (
        player[0], player[1], shape, color, rot,
        0, energy, 0, None, None
    )
    heap = [
        (heuristic(player[0], player[1], shape, color, rot), 0, start)
    ]
    serial = 0
    best_energy = {}
    expansions = 0
    max_expansions = 70000 if not moving else 45000

    while heap and expansions < max_expansions:
        _, _, node = heapq.heappop(heap)
        x, y, s, c, r, used, e, t, first, first_state = node
        key = (x, y, s, c, r, used, t if moving else 0)

        if best_energy.get(key, -1000) >= e:
            continue
        best_energy[key] = e
        expansions += 1

        if t >= horizon:
            continue

        nt = t + 1

        for dx, dy, action in directions:
            dest = (x + dx, y + dy)
            if not _in_bounds(dest) or board.get(dest) == "#":
                continue

            def apply_switches(p, ss, cc, rr):
                for kind in ("S", "C", "R"):
                    if kind in traj:
                        present = p == traj[kind][nt]
                    else:
                        present = p in static[kind]

                    if present:
                        if kind == "S":
                            ss = (ss + 1) % 6
                        elif kind == "C":
                            cc = (cc + 1) % 4
                        else:
                            rr = (rr + 1) % 4
                return ss, cc, rr

            ns, nc, nr = apply_switches(dest, s, c, r)

            if dest in goals:
                gc, opts = goals[dest]
                if (
                    nc != gc
                    or (ns, nr) not in opts
                    or (dest, ns, nr) in denied
                ):
                    continue

            nu = used
            ne = e - decrement
            bit = pickup_bit.get(dest, 0)
            if bit and not (nu & bit):
                nu |= bit
                ne = maximum

            final = _push_destination(dest, push, board, goals)
            if final != dest:
                ns, nc, nr = apply_switches(final, ns, nc, nr)
                bit = pickup_bit.get(final, 0)
                if bit and not (nu & bit):
                    nu |= bit
                    ne = maximum

            f = action if first is None else first
            fs = (ns, nr) if first is None else first_state

            if final in goals:
                gc, opts = goals[final]
                if (
                    nc == gc
                    and (ns, nr) in opts
                    and (final, ns, nr) not in denied
                ):
                    return f, fs, dest
                continue

            if ne < 0:
                continue

            nk = (
                final[0], final[1], ns, nc, nr, nu,
                nt if moving else 0
            )
            if best_energy.get(nk, -1000) >= ne:
                continue

            h = heuristic(final[0], final[1], ns, nc, nr)
            unknown_penalty = 0.5 if final not in board else 0.0
            serial += 1
            child = (
                final[0], final[1], ns, nc, nr,
                nu, ne, nt, f, fs
            )
            heapq.heappush(
                heap,
                (nt + 1.15 * h + unknown_penalty, serial, child),
            )

    return None


def _explore(player, board, goals, push, visits, available):
    allowed = set(available)
    queue = deque([(player, None, 0)])
    seen = {player}
    best = None

    while queue and len(seen) < 180:
        p, first, distance = queue.popleft()

        if p != player:
            unknown_neighbors = 0
            for dx, dy, _ in _DIRS:
                q = (p[0] + dx, p[1] + dy)
                if _in_bounds(q) and q not in board:
                    unknown_neighbors += 1

            score = visits.get(p, 0) * 5 + distance - 5 * unknown_neighbors
            if board.get(p) in ("S", "C", "R"):
                score -= 1
            if best is None or score < best[0]:
                best = (score, first)

        for dx, dy, action in _DIRS:
            if action not in allowed:
                continue
            q = (p[0] + dx, p[1] + dy)
            if not _in_bounds(q) or board.get(q) == "#" or q in goals:
                continue

            q = _push_destination(q, push, board, goals)
            if q not in seen:
                seen.add(q)
                queue.append((q, action if first is None else first, distance + 1))

    return best[1] if best else None


def act(obs, available, memory):
    frame = np.asarray(obs)
    player = _player(frame)

    if player is None:
        for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if action in available:
                return action, None, None
        return "ACTION1", None, None

    lives = _lives(frame)
    old_lives = memory.get("lives")

    # Losing a life restores goals, pickups, switches and the player's
    # starting properties. Discard the old model before reading this frame.
    if (
        lives is not None
        and old_lives is not None
        and lives < old_lives
    ):
        decrement = memory.get("decrement")
        resets = memory.get("resets", 0)
        memory.clear()
        if decrement is not None:
            memory["decrement"] = decrement
        memory["resets"] = resets

    if lives is not None:
        memory["lives"] = lives

    candidates, color = _indicator(frame)
    if color is None:
        color = memory.get("color", 0)
    if not candidates:
        candidates = [memory.get("shape_rotation", (0, 0))]

    previous = memory.get("last")
    denied = memory.setdefault("denied", set())
    old_board = memory.setdefault("board", {})
    old_goals = memory.setdefault("goals", {})

    predicted = memory.get("predicted_shape_rotation")
    old_sr = memory.get("shape_rotation")
    if predicted in candidates:
        sr = predicted
    elif old_sr in candidates:
        sr = old_sr
    else:
        sr = candidates[0]

    if previous is not None:
        old_player, action, attempted, old_energy, old_max = previous

        if player == old_player and attempted is not None:
            if attempted in old_goals:
                gc, options = old_goals[attempted]
                old_color = memory.get("color")

                # A rejected matching-looking goal can distinguish otherwise
                # visually identical internal shape/rotation combinations.
                if (
                    old_sr is not None
                    and old_color == gc
                    and old_sr in options
                ):
                    denied.add((attempted, old_sr[0], old_sr[1]))
                    alternatives = [
                        v for v in candidates
                        if v != old_sr and (attempted, v[0], v[1]) not in denied
                    ]
                    if alternatives:
                        sr = alternatives[0]
            elif attempted not in memory.get("pushers", {}):
                old_board[attempted] = "#"

        if player != old_player and player in old_goals:
            memory.setdefault("done", set()).add(player)
            old_goals.pop(player, None)

    energy, maximum = _meter(frame)
    decrement = memory.get("decrement", 2)

    if previous is not None:
        _, _, _, old_energy, _ = previous
        diff = old_energy - energy
        if diff in (1, 2):
            decrement = diff
            memory["decrement"] = diff

    memory["shape_rotation"] = sr
    memory["color"] = color

    board, goals, dyn, fog = _read_board(frame, player, memory)
    push = _pushers(frame, player, board, memory, fog)
    visits = memory.setdefault("visits", {})
    visits[player] = visits.get(player, 0) + 1

    if player not in board:
        board[player] = "."

    result = _search(
        player, sr[0], color, sr[1], energy, maximum, decrement,
        board, goals, dyn, push, denied, available
    )

    if result is None and decrement == 2 and "decrement" not in memory:
        result = _search(
            player, sr[0], color, sr[1], energy, maximum, 1,
            board, goals, dyn, push, denied, available
        )

    if result is not None:
        action, predicted_sr, _ = result
    else:
        action = _explore(player, board, goals, push, visits, available)
        predicted_sr = sr

    if action is None:
        resets = memory.get("resets", 0)
        if "RESET" in available and resets < 2:
            memory.clear()
            memory["resets"] = resets + 1
            return "RESET", None, None

        action = next(
            (a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
             if a in available),
            "ACTION1",
        )

    delta = next(
        ((dx, dy) for dx, dy, a in _DIRS if a == action),
        (0, 0),
    )
    attempted = (player[0] + delta[0], player[1] + delta[1])
    memory["last"] = (player, action, attempted, energy, maximum)
    memory["predicted_shape_rotation"] = predicted_sr
    return action, None, None

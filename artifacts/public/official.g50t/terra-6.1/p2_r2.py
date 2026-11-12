import numpy as np
from collections import deque

# Actions are indexed 0..3 internally.
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _center(cell):
    return 4 + 6 * (cell % 10), 4 + 6 * (cell // 10)


def _cell(x, y):
    if (x - 4) % 6 or (y - 4) % 6:
        return None
    i, j = (x - 4) // 6, (y - 4) // 6
    if 0 <= i < 10 and 0 <= j < 10:
        return j * 10 + i
    return None


def _neighbor(cell, direction):
    x, y = cell % 10, cell // 10
    dx, dy = DIRS[direction]
    x, y = x + dx, y + dy
    if 0 <= x < 10 and 0 <= y < 10:
        return y * 10 + x
    return None


def _find_tokens(frame, color):
    found = []
    for j in range(1, 10):
        y = 4 + 6 * j
        for i in range(10):
            x = 4 + 6 * i
            a = frame[y - 2:y + 3, x - 2:x + 3]
            if a.shape == (5, 5) and a[2, 2] == 5:
                if int(np.count_nonzero(a == color)) == 24:
                    found.append(j * 10 + i)
    return found


def _goal_patterns():
    a = np.full((9, 9), 5, dtype=np.int8)
    a[1, 1:8] = 9
    a[7, 1:8] = 9
    a[2:7, 7] = 9
    a[4, 4] = 9
    return [np.rot90(a, -r)[1:8, 1:8].copy() for r in range(4)]


def _door_patterns(color):
    a = np.full((7, 7), 5, dtype=np.int8)
    a[1:6, 1:6] = color
    if color == 8:
        a[1, 2] = 5
        a[1, 4] = 5
    a[6, 3] = color
    return [np.rot90(a, -r).copy() for r in range(4)]


def _components(frame, color):
    labels = [[-1] * 64 for _ in range(64)]
    component = 0
    for y in range(7, 63):
        for x in range(64):
            if int(frame[y, x]) != color or labels[y][x] != -1:
                continue
            labels[y][x] = component
            q = deque([(x, y)])
            while q:
                px, py = q.popleft()
                for dx, dy in DIRS:
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < 64 and 7 <= ny < 63
                            and labels[ny][nx] == -1
                            and int(frame[ny, nx]) == color):
                        labels[ny][nx] = component
                        q.append((nx, ny))
            component += 1
    return labels


def _build_world(frame, player):
    goals = _goal_patterns()
    patterns = {8: _door_patterns(8), 11: _door_patterns(11)}
    board = set()
    doors = []
    plates = []
    portals = []
    goal = None

    for j in range(1, 10):
        y = 4 + 6 * j
        for i in range(10):
            x = 4 + 6 * i
            c = j * 10 + i
            patch = frame[y - 3:y + 4, x - 3:x + 4]
            if patch.shape != (7, 7):
                continue

            corners = (patch[0, 0], patch[0, 6],
                       patch[6, 0], patch[6, 6])
            if all(int(v) != 0 for v in corners):
                if int(np.count_nonzero(patch)) >= 24:
                    board.add(c)

            if any(np.array_equal(patch, g) for g in goals):
                goal = c
                board.add(c)

            is_door = False
            for color in (8, 11):
                for rotation, pattern in enumerate(patterns[color]):
                    if np.array_equal(patch, pattern):
                        # Sprite rotations 0, 90, 180, 270 move
                        # down, left, up, right respectively.
                        direction = (1, 2, 0, 3)[rotation]
                        doors.append({
                            "base": c,
                            "open": _neighbor(c, direction),
                            "color": color,
                        })
                        board.add(c)
                        is_door = True
                        break
                if is_door:
                    break

            if is_door:
                continue

            inner = frame[y - 1:y + 2, x - 1:x + 2]
            color = int(frame[y, x])
            if color in (8, 11, 15) and np.all(inner == color):
                plates.append((c, color))
                board.add(c)

            if (np.all(patch[0, :] == 15)
                    and np.all(patch[6, :] == 15)
                    and np.all(patch[:, 0] == 15)
                    and np.all(patch[:, 6] == 15)):
                portals.append(c)
                board.add(c)

    board.add(player)
    if goal is None:
        # A partial match is useful if a moving sprite obscures the goal.
        best = (0, None)
        for c in board:
            x, y = _center(c)
            patch = frame[y - 3:y + 4, x - 3:x + 4]
            if patch.shape != (7, 7):
                continue
            for g in goals:
                score = int(np.count_nonzero((g == 9) & (patch == 9)))
                if score > best[0]:
                    best = (score, c)
        if best[0] >= 10:
            goal = best[1]

    labels = {color: _components(frame, color) for color in (8, 11, 15)}
    groups_by_key = {}

    def group_for(color, component):
        key = (color, component)
        if key not in groups_by_key:
            groups_by_key[key] = {
                "color": color,
                "plates": [],
                "doors": [],
                "portals": [],
            }
        return groups_by_key[key]

    for c, color in plates:
        x, y = _center(c)
        group_for(color, labels[color][y][x])["plates"].append(c)

    for k, door in enumerate(doors):
        x, y = _center(door["base"])
        color = door["color"]
        comp = labels[color][y][x]
        group_for(color, comp)["doors"].append(k)

    for c in portals:
        x, y = _center(c)
        comp = labels[15][y - 3][x]
        group_for(15, comp)["portals"].append(c)

    groups = list(groups_by_key.values())

    # A token can obscure a short section of wire. If a color has exactly
    # one plate-bearing network, attach otherwise isolated outputs to it.
    for color in (8, 11, 15):
        sources = [g for g in groups if g["color"] == color and g["plates"]]
        if len(sources) == 1:
            source = sources[0]
            for g in groups:
                if g is not source and g["color"] == color and not g["plates"]:
                    source["doors"].extend(g["doors"])
                    source["portals"].extend(g["portals"])
                    g["doors"] = []
                    g["portals"] = []

    groups = [g for g in groups if g["plates"]]
    for g in groups:
        bits = 0
        for k in g["doors"]:
            bits |= 1 << k
        g["bits"] = bits

    # The first HUD marker is yellow; remaining markers are blue.
    charges = 0
    for i in range(8):
        x = 1 + 4 * i
        if x < 64 and int(frame[1, x]) in (1, 2, 9):
            charges += 1
        else:
            break
    charges = max(1, charges)

    return {
        "board": board,
        "doors": doors,
        "groups": groups,
        "goal": goal,
        "start": player,
        "charges": charges,
    }


def _read_mask(frame, world, previous):
    mask = previous
    for k, door in enumerate(world["doors"]):
        color = door["color"]

        def visible_at(c):
            if c is None:
                return False
            x, y = _center(c)
            patch = frame[y - 2:y + 3, x - 2:x + 3]
            return patch.shape == (5, 5) and np.count_nonzero(patch == color) >= 18

        closed = visible_at(door["base"])
        opened = visible_at(door["open"])
        if opened and not closed:
            mask |= 1 << k
        elif closed and not opened:
            mask &= ~(1 << k)
    return mask


def _occupied(world, mask):
    result = set()
    for k, door in enumerate(world["doors"]):
        c = door["open"] if mask & (1 << k) else door["base"]
        if c is not None:
            result.add(c)
    return result


def _simulate(world, scripts, state, action):
    player, ghosts, mask, t = state
    blocked = _occupied(world, mask)
    destination = _neighbor(player, action)
    if destination not in world["board"] or destination in blocked:
        return None

    old = (player,) + ghosts
    new = [destination]
    moving = [True]

    for i, pos in enumerate(ghosts):
        nxt = pos
        moved = False
        if t < len(scripts[i]):
            candidate = _neighbor(pos, scripts[i][t])
            if candidate in world["board"] and candidate not in blocked:
                nxt = candidate
                moved = True
        new.append(nxt)
        moving.append(moved)

    stationary = {old[i] for i in range(len(old)) if not moving[i]}
    old_positions = set(old)
    new_positions = set(new)
    mid_mask = mask
    arrivals = []

    for group in world["groups"]:
        plate_cells = set(group["plates"])
        was = bool(old_positions & plate_cells)
        middle = bool(stationary & plate_cells)
        final = bool(new_positions & plate_cells)

        if group["color"] != 11 and was and not middle:
            mid_mask &= ~group["bits"]
        if not middle and final:
            arrivals.append(group)

    # A closing gate can crush the player even if it reopens immediately.
    newly_closed = mask & ~mid_mask
    for k, door in enumerate(world["doors"]):
        if newly_closed & (1 << k) and new[0] == door["base"]:
            return None

    final_mask = mid_mask
    for group in arrivals:
        if group["color"] == 11:
            final_mask ^= group["bits"]
        else:
            final_mask |= group["bits"]

        ends = group["portals"]
        if len(ends) == 2:
            a, b = ends
            new = [b if p == a else a if p == b else p for p in new]

    max_t = max((len(s) for s in scripts), default=0)
    return (new[0], tuple(new[1:]), final_mask, min(t + 1, max_t))


def _path(previous, state):
    result = []
    while previous[state] is not None:
        parent, action = previous[state]
        result.append((action, state))
        state = parent
    result.reverse()
    return result


def _phase_search(world, scripts, initial, budget):
    previous = {initial: None}
    queue = deque([initial])
    candidates = {}
    best = initial
    goal = world["goal"]
    plate_cells = {p for g in world["groups"] for p in g["plates"]}
    local_limit = 20000
    expanded = 0

    def distance(c):
        if goal is None:
            return 0
        return abs(c % 10 - goal % 10) + abs(c // 10 - goal // 10)

    while queue and budget[0] > 0 and expanded < local_limit:
        state = queue.popleft()
        budget[0] -= 1
        expanded += 1
        player, ghosts, _, _ = state

        if goal is not None and player == goal:
            return _path(previous, state), [], _path(previous, best)

        if distance(player) < distance(best[0]):
            best = state

        if (player in plate_cells and player not in ghosts
                and player not in candidates and state != initial):
            candidates[player] = state

        for action in range(4):
            nxt = _simulate(world, scripts, state, action)
            if nxt is not None and nxt not in previous:
                previous[nxt] = (state, action)
                queue.append(nxt)

    choices = []
    for cell, state in candidates.items():
        path = _path(previous, state)
        choices.append((len(path), distance(cell), cell, path))
    choices.sort(key=lambda item: (item[0], item[1]))
    return None, choices, _path(previous, best)


def _make_plan(world, scripts, record, initial):
    budget = [70000]
    fallback = [None]

    def solve(tapes, prior_record, state, depth):
        solution, choices, toward = _phase_search(world, tapes, state, budget)
        if solution is not None:
            return solution

        if fallback[0] is None and toward:
            fallback[0] = toward

        if depth >= world["charges"] - 1:
            return None

        for _, _, _, prefix in choices[:8]:
            tape = list(prior_record) + [a for a, _ in prefix]
            if not tape:
                continue
            if tape in tapes:
                continue

            new_tapes = tapes + [tape]
            start = world["start"]
            reset_state = (start, tuple([start] * len(new_tapes)), 0, 0)

            if fallback[0] is None:
                fallback[0] = prefix + [(4, reset_state)]

            rest = solve(new_tapes, [], reset_state, depth + 1)
            if rest is not None:
                return prefix + [(4, reset_state)] + rest
            if budget[0] <= 0:
                break
        return None

    plan = solve(scripts, record, initial, len(scripts))
    return plan if plan is not None else (fallback[0] or [])


def _match_ghosts(predicted, visible, player):
    """Use visible ghost locations to correct a mismatched prediction."""
    predicted = list(predicted)
    if not predicted:
        return []
    remaining = list(visible)
    result = [None] * len(predicted)

    for i, pos in enumerate(predicted):
        if pos in remaining:
            result[i] = pos
            remaining.remove(pos)

    for i, pos in enumerate(predicted):
        if result[i] is not None:
            continue
        if remaining:
            nearest = min(
                remaining,
                key=lambda c: abs(c % 10 - pos % 10) + abs(c // 10 - pos // 10),
            )
            result[i] = nearest
            remaining.remove(nearest)
        elif pos == player:
            result[i] = player
        else:
            # Multiple copies can overlap and render as a single token.
            result[i] = pos if pos in visible else player
    return result


def act(obs, available, memory):
    frame = np.asarray(obs)
    available = set(available)

    # Force a level reset only when RESET is explicitly available. The
    # harmless first action prevents an initial reset from being a full reset.
    if not memory:
        if "RESET" in available:
            memory["boot"] = "reset"
            return ("ACTION5", None, None)
        memory["boot"] = "initialize"

    if memory.get("boot") == "reset":
        memory["boot"] = "initialize"
        return ("RESET", None, None)

    players = _find_tokens(frame, 9)
    if not players:
        # Animations or a terminal frame should normally not be presented,
        # but avoid indexing an absent token.
        return ("ACTION5" if "ACTION5" in available else "ACTION1", None, None)
    player = players[0]

    if memory.get("boot") == "initialize":
        world = _build_world(frame, player)
        memory.clear()
        memory["world"] = world
        memory["scripts"] = []
        memory["record"] = []
        memory["ghosts"] = []
        memory["mask"] = 0
        memory["plan"] = []
        memory["pending"] = None

    world = memory["world"]
    pending = memory.get("pending")
    invalidated = False

    if pending is not None:
        action, old_player, expected = pending
        exp_player, exp_ghosts, exp_mask, _ = expected

        if action == 4:
            tape = list(memory["record"])
            if tape:
                memory["scripts"].append(tape)
            memory["record"] = []
            if len(memory["scripts"]) >= world["charges"]:
                memory["scripts"] = []
        elif player != old_player or player == exp_player:
            memory["record"].append(action)

        memory["ghosts"] = list(exp_ghosts)
        memory["mask"] = exp_mask
        if player != exp_player:
            invalidated = True

        actual_mask = _read_mask(frame, world, memory["mask"])
        if actual_mask != memory["mask"]:
            memory["mask"] = actual_mask
            invalidated = True

        visible_ghosts = _find_tokens(frame, 2)
        predicted_visible = set(memory["ghosts"]) - {player}
        if set(visible_ghosts) != predicted_visible:
            memory["ghosts"] = _match_ghosts(
                memory["ghosts"], visible_ghosts, player
            )
            invalidated = True

        memory["pending"] = None

    if invalidated:
        memory["plan"] = []

    scripts = memory["scripts"]
    ghosts = memory["ghosts"]
    if len(ghosts) != len(scripts):
        ghosts = (ghosts + [world["start"]] * len(scripts))[:len(scripts)]
        memory["ghosts"] = ghosts

    if not memory["plan"]:
        max_t = max((len(s) for s in scripts), default=0)
        state = (
            player,
            tuple(ghosts),
            memory["mask"],
            min(len(memory["record"]), max_t),
        )
        memory["plan"] = _make_plan(
            world, scripts, memory["record"], state
        )

    if memory["plan"]:
        action, expected = memory["plan"].pop(0)
        name = "ACTION5" if action == 4 else NAMES[action]
        if name in available:
            memory["pending"] = (action, player, expected)
            return (name, None, None)

    # Conservative fallback: take a legal-looking move toward the goal.
    blocked = _occupied(world, memory["mask"])
    goal = world["goal"]
    choices = []
    for action in range(4):
        nxt = _neighbor(player, action)
        if nxt in world["board"] and nxt not in blocked and NAMES[action] in available:
            distance = 0 if goal is None else (
                abs(nxt % 10 - goal % 10) + abs(nxt // 10 - goal // 10)
            )
            choices.append((distance, action, nxt))
    if choices:
        _, action, nxt = min(choices)
        max_t = max((len(s) for s in scripts), default=0)
        state = (player, tuple(ghosts), memory["mask"],
                 min(len(memory["record"]), max_t))
        expected = _simulate(world, scripts, state, action)
        if expected is not None:
            memory["pending"] = (action, player, expected)
        return (NAMES[action], None, None)

    if memory["record"] and "ACTION5" in available:
        n = len(scripts) + 1
        expected = (world["start"], tuple([world["start"]] * n), 0, 0)
        memory["pending"] = (4, player, expected)
        return ("ACTION5", None, None)
    return ("ACTION1", None, None)

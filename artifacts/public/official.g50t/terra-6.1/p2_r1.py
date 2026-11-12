from collections import deque
import numpy as np

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def find_player(a):
    for row in range(10):
        y = 4 + 6 * row
        for col in range(10):
            x = 4 + 6 * col
            patch = a[y - 2:y + 3, x - 2:x + 3]
            if (patch.shape == (5, 5)
                    and int(patch[2, 2]) == 5
                    and int(np.count_nonzero(patch == 9)) >= 23):
                return row * 10 + col
    return None


def components(a):
    labels = [[-1] * 64 for _ in range(64)]
    colors = []
    for y in range(64):
        for x in range(64):
            color = int(a[y, x])
            if color not in (8, 11, 15) or labels[y][x] >= 0:
                continue
            group = len(colors)
            colors.append(color)
            labels[y][x] = group
            queue = deque([(x, y)])
            while queue:
                xx, yy = queue.popleft()
                for dx, dy in DIRS:
                    nx, ny = xx + dx, yy + dy
                    if (0 <= nx < 64 and 0 <= ny < 64
                            and labels[ny][nx] < 0
                            and int(a[ny, nx]) == color):
                        labels[ny][nx] = group
                        queue.append((nx, ny))
    return labels, colors


def parse_frame(obs, banned_edges=()):
    a = np.asarray(obs)
    labels, colors = components(a)
    centers = [
        (4 + 6 * (i % 10), 4 + 6 * (i // 10))
        for i in range(100)
    ]

    floor = set()
    for i, (x, y) in enumerate(centers):
        if int(a[y, x]) == 0:
            continue
        votes = sum(
            int(a[y + dy, x + dx]) != 0
            for dx in (-2, 2)
            for dy in (-2, 2)
        )
        if votes >= 3:
            floor.add(i)

    cyan = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 8, 5, 8, 5, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 8, 8, 8, 8, 8, 5],
        [5, 5, 5, 8, 5, 5, 5],
    ])
    purple = np.array([
        [5, 5, 5, 5, 5, 5, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 11, 11, 11, 11, 11, 5],
        [5, 5, 5, 11, 5, 5, 5],
    ])

    templates = []
    for is_purple, base in ((False, cyan), (True, purple)):
        for rotation in range(4):
            templates.append(
                (is_purple, rotation, np.rot90(base, -rotation))
            )

    doors = []
    door_cells = set()
    for i, (x, y) in enumerate(centers):
        patch = a[y - 3:y + 4, x - 3:x + 4]
        if patch.shape != (7, 7):
            continue
        for is_purple, rotation, template in templates:
            if not np.array_equal(patch, template):
                continue
            group = labels[y][x]
            if group < 0:
                continue
            dx, dy = ((0, 1), (-1, 0), (0, -1), (1, 0))[rotation]
            col, row = i % 10, i // 10
            nc, nr = col + dx, row + dy
            opened = nr * 10 + nc if 0 <= nc < 10 and 0 <= nr < 10 else -1
            doors.append((i, opened, 1 << group, is_purple))
            door_cells.add(i)
            floor.add(i)
            break

    goal_template = np.full((9, 9), -1, dtype=int)
    goal_template[1, 1:8] = 9
    goal_template[7, 1:8] = 9
    goal_template[2:7, 2:7] = 5
    goal_template[2:7, 7] = 9
    goal_template[4, 4] = 9
    goal_templates = [np.rot90(goal_template, -k) for k in range(4)]

    goals = set()
    for i, (x, y) in enumerate(centers):
        if x < 4 or y < 4 or x + 4 >= 64 or y + 4 >= 64:
            continue
        patch = a[y - 4:y + 5, x - 4:x + 5]
        for template in goal_templates:
            mask = template >= 0
            if np.array_equal(patch[mask], template[mask]):
                goals.add(i)
                floor.add(i)
                break

    player = None
    enemies = []
    for i, (x, y) in enumerate(centers):
        patch = a[y - 2:y + 3, x - 2:x + 3]
        if patch.shape != (5, 5) or int(patch[2, 2]) != 5:
            continue
        if int(np.count_nonzero(patch == 9)) >= 23:
            player = i
            floor.add(i)
        elif int(np.count_nonzero(patch == 14)) >= 23:
            enemies.append((i, 0))
            floor.add(i)

    plates = {}
    plate_groups = {}
    for i, (x, y) in enumerate(centers):
        if i in door_cells:
            continue
        color = int(a[y, x])
        if color not in (8, 11, 15):
            continue
        patch = a[y - 1:y + 2, x - 1:x + 2]
        if patch.shape != (3, 3) or not np.all(patch == color):
            continue
        group = labels[y][x]
        if group < 0:
            continue
        plates[i] = plates.get(i, 0) | (1 << group)
        plate_groups[group] = color
        floor.add(i)

    boxes = {}
    for i, (x, y) in enumerate(centers):
        patch = a[y - 3:y + 4, x - 3:x + 4]
        if patch.shape != (7, 7):
            continue
        if (np.all(patch[0, :] == 15)
                and np.all(patch[6, :] == 15)
                and np.all(patch[:, 0] == 15)
                and np.all(patch[:, 6] == 15)):
            group = labels[y - 3][x]
            if group >= 0:
                boxes.setdefault(group, []).append(i)
                floor.add(i)

    teleports = []
    for group, cells in boxes.items():
        if len(cells) == 2 and group in plate_groups:
            teleports.append((1 << group, cells[0], cells[1]))

    purple_bits = 0
    for _, _, bit, is_purple in doors:
        if is_purple:
            purple_bits |= bit

    slots = 0
    for k in range(8):
        x = 1 + 4 * k
        if x + 2 >= 64:
            break
        patch = a[1:4, x:x + 3]
        if patch.shape != (3, 3):
            break
        count = np.count_nonzero(
            (patch == 1) | (patch == 2) | (patch == 9)
        )
        if int(count) >= 7:
            slots += 1
        else:
            break
    slots = max(1, slots)

    banned = {(int(pos), int(action)) for pos, action in banned_edges}
    neighbors = {}
    for pos in floor:
        col, row = pos % 10, pos // 10
        result = []
        for action, (dx, dy) in enumerate(DIRS):
            nc, nr = col + dx, row + dy
            nxt = nr * 10 + nc
            if (0 <= nc < 10 and 0 <= nr < 10
                    and nxt in floor and (pos, action) not in banned):
                result.append(nxt)
            else:
                result.append(-1)
        neighbors[pos] = tuple(result)

    return {
        "neighbors": neighbors,
        "doors": doors,
        "plates": plates,
        "teleports": teleports,
        "purple_bits": purple_bits,
        "player": player,
        "goals": goals,
        "enemies": tuple(enemies),
        "slots": slots,
    }


def active_bits(model, player, ghosts, enemies):
    plates = model["plates"]
    bits = plates.get(player, 0)
    for pos in ghosts:
        bits |= plates.get(pos, 0)
    for pos, _ in enemies:
        if pos >= 0:
            bits |= plates.get(pos, 0)
    return bits


def blocked_cells(model, active, latch):
    blocked = set()
    for closed, opened, bit, is_purple in model["doors"]:
        on = bool((latch if is_purple else active) & bit)
        pos = opened if on else closed
        if pos >= 0:
            blocked.add(pos)
    return blocked


def transition(model, state, action, tracks):
    player, ghosts, enemies, tick, latch = state
    neighbors = model["neighbors"]
    old_active = active_bits(model, player, ghosts, enemies)
    old_blocked = blocked_cells(model, old_active, latch)

    dest = neighbors.get(player, (-1, -1, -1, -1))[action]
    if dest < 0 or dest in old_blocked:
        return None

    new_ghosts = []
    for index, pos in enumerate(ghosts):
        if tick < len(tracks[index]):
            direction = tracks[index][tick]
            nxt = neighbors.get(pos, (-1, -1, -1, -1))[direction]
            if nxt >= 0 and nxt not in old_blocked:
                pos = nxt
        new_ghosts.append(pos)
    ghosts = tuple(new_ghosts)

    # The enemy's hidden track is approximated by traversable cells.
    # It prefers its current direction, then its two sides, then reverse.
    enemy_to_action = (1, 3, 0, 2)
    new_enemies = []
    for pos, heading in enemies:
        if pos < 0:
            new_enemies.append((pos, heading))
            continue
        chosen, chosen_heading = pos, heading
        for h in (heading, (heading - 1) % 4,
                  (heading + 1) % 4, (heading + 2) % 4):
            nxt = neighbors.get(pos, (-1, -1, -1, -1))[enemy_to_action[h]]
            if nxt >= 0 and nxt not in old_blocked:
                chosen, chosen_heading = nxt, h
                break
        new_enemies.append((chosen, chosen_heading))
    enemies = tuple(new_enemies)

    new_active = active_bits(model, dest, ghosts, enemies)
    rising = new_active & ~old_active
    new_latch = latch ^ (rising & model["purple_bits"])

    for bit, left, right in model["teleports"]:
        if not rising & bit:
            continue
        if dest == left:
            dest = right
        elif dest == right:
            dest = left
        ghosts = tuple(
            right if pos == left else left if pos == right else pos
            for pos in ghosts
        )
        enemies = tuple(
            (right if pos == left else left if pos == right else pos, heading)
            for pos, heading in enemies
        )

    final_active = active_bits(model, dest, ghosts, enemies)
    final_blocked = blocked_cells(model, final_active, new_latch)

    # A gate that closes on an enemy destroys it.
    closing = set()
    for closed, opened, bit, is_purple in model["doors"]:
        was_open = bool((latch if is_purple else old_active) & bit)
        is_open = bool((new_latch if is_purple else final_active) & bit)
        if was_open and not is_open:
            closing.add(closed)

    if dest in closing:
        return None
    if dest in final_blocked:
        return None
    if any(pos in closing for pos in ghosts):
        return None
    if any(pos in final_blocked for pos in ghosts):
        return None

    enemies = tuple(
        (-1, heading) if pos in closing else (pos, heading)
        for pos, heading in enemies
    )
    if any(pos == dest for pos, _ in enemies if pos >= 0):
        return None

    max_tick = max((len(track) for track in tracks), default=0)
    return (dest, ghosts, enemies, min(tick + 1, max_tick), new_latch)


def reconstruct(pred, state):
    path = []
    while pred[state] is not None:
        previous, action = pred[state]
        path.append(action)
        state = previous
    path.reverse()
    return path


def search(model, tracks, start_latch, start_enemies, limit, collect):
    player = model["player"]
    ghosts = tuple(player for _ in tracks)
    start = (player, ghosts, start_enemies, 0, start_latch)
    queue = deque([start])
    pred = {start: None}
    candidates = []
    per_plate = {}
    expanded = 0

    while queue and expanded < limit:
        state = queue.popleft()
        expanded += 1
        pos = state[0]

        if pos in model["goals"]:
            return reconstruct(pred, state), candidates, expanded

        if collect and pos in model["plates"] and state != start:
            signature = (state[1], state[2], state[4])
            seen = per_plate.setdefault(pos, set())
            if len(seen) < 2 and signature not in seen:
                seen.add(signature)
                candidates.append((reconstruct(pred, state), state))

        for action in range(4):
            nxt = transition(model, state, action, tracks)
            if nxt is not None and nxt not in pred:
                pred[nxt] = (state, action)
                queue.append(nxt)

    return None, candidates, expanded


def solve(model):
    if model["player"] is None or not model["goals"]:
        return None

    branches = deque([((), 0, model["enemies"], [])])
    visited = set()
    total_expanded = 0
    branches_tried = 0

    while branches and total_expanded < 120000 and branches_tried < 24:
        tracks, latch, enemies, prefix = branches.popleft()
        key = (tracks, latch, enemies)
        if key in visited:
            continue
        visited.add(key)
        branches_tried += 1

        can_record = len(tracks) + 1 < model["slots"]
        limit = min(18000, 120000 - total_expanded)
        solution, candidates, expanded = search(
            model, tracks, latch, enemies, limit, can_record
        )
        total_expanded += expanded

        if solution is not None:
            return prefix + [ACTIONS[a] for a in solution]

        if not can_record:
            continue

        candidates.sort(key=lambda item: len(item[0]))
        for recording, end_state in candidates[:12]:
            if not recording:
                continue
            new_tracks = tracks + (tuple(recording),)
            end_active = active_bits(
                model, end_state[0], end_state[1], end_state[2]
            )
            new_latch = latch ^ (end_active & model["purple_bits"])
            end_enemies = end_state[2]

            # Rewinding restores enemy positions while preserving headings.
            rewound_enemies = tuple(
                (start_pos, end_enemies[i][1])
                for i, (start_pos, _) in enumerate(enemies)
            )
            new_prefix = (
                prefix + [ACTIONS[a] for a in recording] + ["ACTION5"]
            )
            branches.append(
                (new_tracks, new_latch, rewound_enemies, new_prefix)
            )

    return None


def fallback(model):
    start = model["player"]
    if start is None or not model["goals"]:
        return "ACTION1"

    blocked = blocked_cells(model, 0, 0)
    queue = deque([start])
    first = {start: None}

    while queue:
        pos = queue.popleft()
        if pos in model["goals"] and first[pos] is not None:
            return ACTIONS[first[pos]]
        for action, nxt in enumerate(model["neighbors"].get(pos, ())):
            if nxt < 0 or nxt in blocked or nxt in first:
                continue
            first[nxt] = action if pos == start else first[pos]
            queue.append(nxt)

    for action, nxt in enumerate(model["neighbors"].get(start, ())):
        if nxt >= 0 and nxt not in blocked:
            return ACTIONS[action]
    return "ACTION1"


def choose_available(action, available):
    if not available or action in available:
        return action
    for candidate in ACTIONS:
        if candidate in available:
            return candidate
    if "ACTION5" in available:
        return "ACTION5"
    return action


def act(obs, available, memory):
    available = set(available or [])
    a = np.asarray(obs)

    if "phase" not in memory:
        memory["banned"] = []
        memory["failures"] = 0
        if "RESET" in available and "ACTION5" in available:
            memory["phase"] = "reset"
            return ("ACTION5", None, None)
        memory["phase"] = "plan"

    if memory["phase"] == "reset":
        memory["phase"] = "plan"
        memory.pop("last_move", None)
        return ("RESET", None, None)

    last_move = memory.pop("last_move", None)
    if last_move is not None:
        current = find_player(a)
        previous, direction = last_move
        if current is not None and current == previous:
            failures = int(memory.get("failures", 0))
            if failures < 4 and "RESET" in available:
                banned = memory.setdefault("banned", [])
                edge = [int(previous), int(direction)]
                if edge not in banned:
                    banned.append(edge)
                memory["failures"] = failures + 1
                memory["phase"] = "plan"
                memory.pop("plan", None)
                return ("RESET", None, None)

    if memory["phase"] == "plan":
        model = parse_frame(a, memory.get("banned", []))
        plan = solve(model)
        if plan:
            memory["plan"] = plan
            memory["phase"] = "execute"
        else:
            memory["phase"] = "fallback"
            action = choose_available(fallback(model), available)
            if action in ACTIONS and model["player"] is not None:
                memory["last_move"] = [
                    model["player"], ACTIONS.index(action)
                ]
            return (action, None, None)

    if memory["phase"] == "execute":
        plan = memory.get("plan", [])
        if plan:
            action = choose_available(plan.pop(0), available)
            if action in ACTIONS:
                player = find_player(a)
                if player is not None:
                    memory["last_move"] = [player, ACTIONS.index(action)]
            return (action, None, None)
        memory["phase"] = "fallback"

    model = parse_frame(a, memory.get("banned", []))
    action = choose_available(fallback(model), available)
    if action in ACTIONS and model["player"] is not None:
        memory["last_move"] = [model["player"], ACTIONS.index(action)]
    return (action, None, None)

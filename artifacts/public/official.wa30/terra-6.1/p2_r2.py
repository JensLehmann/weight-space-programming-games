from collections import deque

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _neighbor(p, d):
    x, y = p % 16, p // 16
    dx, dy = DIRS[d]
    x += dx
    y += dy
    if 0 <= x < 16 and 0 <= y < 16:
        return y * 16 + x
    return None


def _adjacent(a, b):
    return abs(a % 16 - b % 16) + abs(a // 16 - b // 16) == 1


def _read_board(obs, memory):
    terrain = {int(k): v for k, v in memory.get("terrain", {}).items()}
    player = None
    facing = 0
    boxes = {}
    helpers = []
    enemies = []

    for y in range(16):
        for x in range(16):
            p = y * 16 + x
            tile = obs[4 * y:min(4 * y + 4, 63), 4 * x:4 * x + 4]
            if tile.shape[0] < 3:
                continue
            center = int(tile[1, 1])

            if center == 14:
                player = p
                if all(int(tile[0, i]) == 0 for i in range(4)):
                    facing = 0
                elif all(int(tile[i, 0]) == 0 for i in range(3)):
                    facing = 2
                elif all(int(tile[i, 3]) == 0 for i in range(3)):
                    facing = 3
                else:
                    facing = 1
            elif center == 9:
                boxes[p] = int(tile[0, 0])
            elif center == 12:
                helpers.append(p)
            elif center == 15:
                enemies.append(p)
            elif center == 5:
                terrain[p] = "w"
            elif center == 2:
                fence = (
                    int(tile[0, 0]) == 2
                    and int(tile[0, 1]) != 2
                    and int(tile[1, 0]) != 2
                    and int(tile[2, 3]) != 2
                )
                if fence:
                    terrain[p] = "f"
                elif (tile == 9).any():
                    terrain[p] = "g"
                elif terrain.get(p) not in ("g", "f"):
                    terrain[p] = "q"

    candidates = {p for p, t in terrain.items() if t in ("g", "q")}
    seen = set()
    for start in list(candidates):
        if start in seen:
            continue
        component = []
        queue = deque([start])
        seen.add(start)
        has_goal = False
        while queue:
            p = queue.popleft()
            component.append(p)
            if terrain.get(p) == "g":
                has_goal = True
            for d in range(4):
                q = _neighbor(p, d)
                if q in candidates and q not in seen:
                    seen.add(q)
                    queue.append(q)
        if has_goal:
            for p in component:
                terrain[p] = "g"
            xs = [p % 16 for p in component]
            ys = [p // 16 for p in component]
            for yy in range(min(ys), max(ys) + 1):
                for xx in range(min(xs), max(xs) + 1):
                    p = yy * 16 + xx
                    if p not in terrain and (
                        p in boxes or p == player
                        or p in helpers or p in enemies
                    ):
                        terrain[p] = "g"

    memory["terrain"] = terrain
    goals = {p for p, t in terrain.items() if t == "g"}
    red = {p for p, t in terrain.items() if t == "q"}
    walls = {p for p, t in terrain.items() if t == "w"}
    fences = {p for p, t in terrain.items() if t == "f"}

    player_box = None
    owners = {}
    for b, border in boxes.items():
        if border == 0 and player is not None and _adjacent(player, b):
            player_box = b
            owners[b] = ("player", player)
        elif border in (5, 3):
            possible = []
            for e in enemies:
                if _adjacent(e, b):
                    possible.append(("enemy", e))
            for h in helpers:
                if _adjacent(h, b):
                    possible.append(("helper", h))
            if possible:
                owners[b] = possible[0]

    return (
        player, facing, boxes, helpers, enemies, goals, red,
        walls, fences, player_box, owners
    )


def _first_step_bfs(start, targets, blocked, fences):
    if start in targets:
        return start
    queue = deque([start])
    first = {start: start}
    while queue:
        p = queue.popleft()
        for d in (2, 3, 0, 1):
            q = _neighbor(p, d)
            if q is None or q in first or q in blocked or q in fences:
                continue
            first[q] = q if p == start else first[p]
            if q in targets:
                return first[q]
            queue.append(q)
    return None


def _pair_first_step(actor, box, targets, blocked, fences):
    if box in targets:
        return actor
    ox = box % 16 - actor % 16
    oy = box // 16 - actor // 16
    queue = deque([actor])
    first = {actor: actor}
    while queue:
        p = queue.popleft()
        bx, by = p % 16 + ox, p // 16 + oy
        for d in (2, 3, 0, 1):
            np = _neighbor(p, d)
            if np is None or np in first or np in blocked or np in fences:
                continue
            dx, dy = DIRS[d]
            nx, ny = bx + dx, by + dy
            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue
            nb = ny * 16 + nx
            if nb in blocked:
                continue
            first[np] = np if p == actor else first[p]
            if nb in targets:
                return first[np]
            queue.append(np)
    return None


def _exact_helper_path(actor, box, targets, occupied, fences):
    # Reproduce the helper's BFS, which tests against the original
    # occupied positions even while exploring hypothetical moves.
    if box in targets:
        return True
    ox = box % 16 - actor % 16
    oy = box // 16 - actor // 16
    occupied = set(occupied) | {actor, box}
    queue = deque([actor])
    seen = {actor}

    while queue:
        p = queue.popleft()
        bx, by = p % 16 + ox, p // 16 + oy
        for d in (2, 3, 0, 1):
            np = _neighbor(p, d)
            if np is None or np in seen or np in fences:
                continue
            dx, dy = DIRS[d]
            nx, ny = bx + dx, by + dy
            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue
            nb = ny * 16 + nx
            if np in occupied and np != box:
                continue
            if nb in occupied and nb != actor:
                continue
            if nb in targets:
                return True
            seen.add(np)
            queue.append(np)
    return False


def _predict_enemy(enemy, boxes, helpers, enemies, red, walls, fences,
                   owners, horizon=28):
    current_boxes = set(boxes)
    other_actors = set(helpers) | (set(enemies) - {enemy})
    held = None
    for b, owner in owners.items():
        if owner == ("enemy", enemy):
            held = b
            break

    positions = [enemy]
    box_history = [set(current_boxes)]
    pos = enemy

    for _ in range(horizon):
        if held is not None:
            if held in red:
                held = None
            else:
                blocked = walls | other_actors | (current_boxes - {held})
                next_pos = _pair_first_step(pos, held, red, blocked, fences)
                if next_pos is not None and next_pos != pos:
                    dx = next_pos % 16 - pos % 16
                    dy = next_pos // 16 - pos // 16
                    new_box = (held // 16 + dy) * 16 + held % 16 + dx
                    current_boxes.remove(held)
                    current_boxes.add(new_box)
                    held = new_box
                    pos = next_pos
        else:
            eligible = []
            for b in sorted(current_boxes):
                owner = owners.get(b)
                if b in red:
                    continue
                if owner and owner[0] == "enemy" and owner[1] != enemy:
                    continue
                eligible.append(b)

            nearby = [b for b in eligible if _adjacent(pos, b)]
            if nearby:
                held = nearby[0]
            else:
                targets = set()
                for b in eligible:
                    for d in range(4):
                        q = _neighbor(b, d)
                        if q is not None:
                            targets.add(q)
                blocked = walls | other_actors | current_boxes
                nxt = _first_step_bfs(pos, targets, blocked, fences)
                if nxt is not None:
                    pos = nxt

        positions.append(pos)
        box_history.append(set(current_boxes))
    return positions, box_history


def _intercept(player, facing, enemy, boxes, helpers, enemies, red,
               walls, fences, owners):
    positions, box_history = _predict_enemy(
        enemy, boxes, helpers, enemies, red, walls, fences, owners
    )
    horizon = len(positions) - 1
    fixed_actors = set(helpers) | (set(enemies) - {enemy})
    start = (player, facing, 0)
    queue = deque([start])
    first_action = {start: None}

    while queue:
        p, face, t = queue.popleft()
        first = first_action[(p, face, t)]
        if _neighbor(p, face) == positions[t]:
            return (first if first is not None else "ACTION5", t + 1)
        if t >= horizon:
            continue

        blocked = walls | fixed_actors | box_history[t] | {positions[t]}
        for d in range(4):
            q = _neighbor(p, d)
            if q is None or q in blocked or q in fences:
                q = p
            if q == positions[t + 1]:
                continue
            state = (q, d, t + 1)
            if state not in first_action:
                first_action[state] = (
                    first if first is not None else ACTIONS[d]
                )
                queue.append(state)
    return None


def _plan_box(player, facing, box, already_held, destinations,
              walls, fences, other_boxes, actors):
    if not destinations:
        return None

    blocked = walls | other_boxes | actors
    start = (player, facing, bool(already_held))
    queue = deque([start])
    first_action = {start: None}
    distance = {start: 0}

    while queue:
        p, face, held = queue.popleft()
        state = (p, face, held)
        first = first_action[state]
        dist = distance[state]

        if held:
            b = _neighbor(p, face)
            if b is None:
                continue
            if b in destinations:
                return (first if first is not None else "ACTION5", dist + 1)

            for d in range(4):
                np = _neighbor(p, d)
                nb = _neighbor(b, d)
                if np is None or nb is None:
                    continue
                if np in blocked or np in fences or nb in blocked:
                    continue
                ns = (np, face, True)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else ACTIONS[d]
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)
        else:
            if _neighbor(p, face) == box:
                ns = (p, face, True)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else "ACTION5"
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)

            for d in range(4):
                np = _neighbor(p, d)
                if np is None or np in blocked or np in fences or np == box:
                    np = p
                ns = (np, d, False)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else ACTIONS[d]
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)
    return None


def _plan_reorient(player, facing, box, already_held, destinations,
                   walls, fences, other_boxes, actors, limit=100000):
    """
    Search over player position, crate position, facing, and attachment.
    Unlike _plan_box, this can release and pick up from another side.
    """
    if not destinations:
        return None

    blocked = walls | other_boxes | actors
    start = (player, box, facing, bool(already_held))
    queue = deque([start])
    first_action = {start: None}
    distance = {start: 0}

    while queue and len(first_action) <= limit:
        p, b, face, held = queue.popleft()
        state = (p, b, face, held)
        first = first_action[state]
        dist = distance[state]

        if held:
            if b in destinations:
                return (first if first is not None else "ACTION5", dist + 1)

            # Release the crate to change its carrying orientation.
            ns = (p, b, face, False)
            if ns not in first_action:
                first_action[ns] = (
                    first if first is not None else "ACTION5"
                )
                distance[ns] = dist + 1
                queue.append(ns)

            for d in range(4):
                np = _neighbor(p, d)
                nb = _neighbor(b, d)
                if np is None or nb is None:
                    continue
                if np in blocked or np in fences or nb in blocked:
                    continue
                ns = (np, nb, face, True)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else ACTIONS[d]
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)
        else:
            if _neighbor(p, face) == b:
                ns = (p, b, face, True)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else "ACTION5"
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)

            for d in range(4):
                np = _neighbor(p, d)
                if np is None or np in blocked or np in fences or np == b:
                    np = p
                ns = (np, b, d, False)
                if ns not in first_action:
                    first_action[ns] = (
                        first if first is not None else ACTIONS[d]
                    )
                    distance[ns] = dist + 1
                    queue.append(ns)
    return None


def _approach_box(player, facing, box, blocked, fences):
    start = (player, facing)
    queue = deque([start])
    first = {start: None}
    while queue:
        p, face = queue.popleft()
        action = first[(p, face)]
        if _neighbor(p, face) == box:
            return action if action is not None else "ACTION5"
        for d in range(4):
            q = _neighbor(p, d)
            if q is None or q in blocked or q in fences or q == box:
                q = p
            state = (q, d)
            if state not in first:
                first[state] = action if action is not None else ACTIONS[d]
                queue.append(state)
    return None


def _handoff_destinations(box, helpers, goals, walls, fences,
                          other_boxes, enemies, bad):
    if not helpers or not goals:
        return set()

    result = set()
    free_goals = goals - other_boxes
    helper_set = set(helpers)
    enemy_set = set(enemies)

    for f in fences:
        if f in bad or f in walls or f in other_boxes or f in enemy_set:
            continue
        for h in helpers:
            other_helpers = helper_set - {h}
            base = walls | other_boxes | enemy_set | other_helpers
            for d in range(4):
                side = _neighbor(f, d)
                if side is None or side in fences or side in base:
                    continue
                if side != h:
                    step = _first_step_bfs(h, {side}, base | {f}, fences)
                    if step is None:
                        continue
                if _exact_helper_path(side, f, free_goals, base, fences):
                    result.add(f)
                    break
            if f in result:
                break
    return result


def _ban_fence_side(box, helper, fences, bad):
    if box not in fences:
        return
    if box % 16 != helper % 16:
        directions = (0, 1)
    else:
        directions = (2, 3)
    bad.add(box)
    for d in directions:
        p = box
        while True:
            p = _neighbor(p, d)
            if p is None or p not in fences:
                break
            bad.add(p)


def _stage_carried_box(player, box, walls, fences, other_boxes, actors):
    blocked = walls | other_boxes | actors
    ox = box % 16 - player % 16
    oy = box // 16 - player // 16
    queue = deque([player])
    first = {player: None}

    while queue:
        p = queue.popleft()
        bx, by = p % 16 + ox, p // 16 + oy
        if not (0 <= bx < 16 and 0 <= by < 16):
            continue
        b = by * 16 + bx
        if b not in fences:
            return first[p], b
        for d in range(4):
            np = _neighbor(p, d)
            nb = _neighbor(b, d)
            if np is None or nb is None:
                continue
            if np in first or np in blocked or np in fences or nb in blocked:
                continue
            first[np] = first[p] if first[p] is not None else ACTIONS[d]
            queue.append(np)
    return None


def _idle(player, facing, boxes, helpers, enemies, walls, fences):
    front = _neighbor(player, facing)
    if front in enemies:
        return "ACTION5"
    if front not in boxes:
        return "ACTION5"
    occupied = set(boxes) | set(helpers) | set(enemies) | walls | fences
    for d in range(4):
        q = _neighbor(player, d)
        if q is None or q in occupied:
            return ACTIONS[d]
    return "ACTION1"


def act(obs, available, memory):
    (player, facing, boxes, helpers, enemies, goals, red, walls, fences,
     player_box, owners) = _read_board(obs, memory)

    if player is None:
        return ("ACTION1", None, None)

    box_positions = set(boxes)
    actors = set(helpers) | set(enemies)
    bad = {int(p) for p in memory.get("bad_fences", [])}

    # Reclaim a crate from a helper that has no route to any goal.
    if player_box is None:
        stuck = []
        for b, owner in owners.items():
            if owner[0] != "helper" or b in goals:
                continue
            h = owner[1]
            occupied = walls | box_positions | actors
            if not _exact_helper_path(h, b, goals, occupied, fences):
                stuck.append((b, h))

        if stuck:
            b, h = min(
                stuck,
                key=lambda pair: (
                    abs(pair[0] % 16 - player % 16)
                    + abs(pair[0] // 16 - player // 16)
                )
            )
            _ban_fence_side(b, h, fences, bad)
            memory["bad_fences"] = sorted(bad)
            blocked = walls | (box_positions - {b}) | actors
            action = _approach_box(player, facing, b, blocked, fences)
            if action is not None:
                return (action, None, None)

    if player_box is None and enemies:
        best = None
        for enemy in enemies:
            plan = _intercept(
                player, facing, enemy, boxes, helpers, enemies, red,
                walls, fences, owners
            )
            if plan is not None and (best is None or plan[1] < best[1]):
                best = plan
        if best is not None:
            return (best[0], None, None)

    if player_box is not None:
        other_boxes = box_positions - {player_box}

        stage = memory.get("stage")
        if stage is not None:
            stage = int(stage)
            if player_box == stage:
                memory.pop("stage", None)
                return ("ACTION5", None, None)
            plan = _plan_box(
                player, facing, player_box, True, {stage},
                walls, fences, other_boxes, actors
            )
            if plan is not None:
                return (plan[0], None, None)
            memory.pop("stage", None)

        plan = _plan_box(
            player, facing, player_box, True, goals,
            walls, fences, other_boxes, actors
        )
        handoffs = set()
        if plan is None and helpers:
            handoffs = _handoff_destinations(
                player_box, helpers, goals, walls, fences,
                other_boxes, enemies, bad
            )
            plan = _plan_box(
                player, facing, player_box, True, handoffs,
                walls, fences, other_boxes, actors
            )
        if plan is not None:
            return (plan[0], None, None)

        # A new orientation may make an otherwise unreachable goal or
        # handoff possible.
        targets = (goals - other_boxes) | handoffs
        plan = _plan_reorient(
            player, facing, player_box, True, targets,
            walls, fences, other_boxes, actors
        )
        if plan is None and bad and helpers:
            relaxed = _handoff_destinations(
                player_box, helpers, goals, walls, fences,
                other_boxes, enemies, set()
            )
            plan = _plan_reorient(
                player, facing, player_box, True, relaxed,
                walls, fences, other_boxes, actors
            )
        if plan is not None:
            return (plan[0], None, None)

        staged = _stage_carried_box(
            player, player_box, walls, fences, other_boxes, actors
        )
        if staged is not None:
            action, destination = staged
            if action is None:
                return ("ACTION5", None, None)
            memory["stage"] = destination
            return (action, None, None)
        return ("ACTION5", None, None)

    memory.pop("stage", None)

    best = None
    previous_target = memory.get("target")
    candidates = []
    for b in box_positions:
        if b in goals:
            continue
        owner = owners.get(b)
        if owner and owner[0] == "helper":
            continue
        candidates.append(b)

    for b in candidates:
        plan = _plan_box(
            player, facing, b, False, goals,
            walls, fences, box_positions - {b}, actors
        )
        if plan is not None:
            score = plan[1] - (2 if b == previous_target else 0)
            if best is None or score < best[0]:
                best = (score, plan[0], b)

    handoffs_by_box = {}
    if best is None and helpers:
        for b in candidates:
            if b in fences:
                continue
            handoffs = _handoff_destinations(
                b, helpers, goals, walls, fences,
                box_positions - {b}, enemies, bad
            )
            handoffs_by_box[b] = handoffs
            plan = _plan_box(
                player, facing, b, False, handoffs,
                walls, fences, box_positions - {b}, actors
            )
            if plan is not None:
                score = plan[1] - (2 if b == previous_target else 0)
                if best is None or score < best[0]:
                    best = (score, plan[0], b)

    if best is not None:
        memory["target"] = best[2]
        return (best[1], None, None)

    # More expensive fallback: permit intermediate releases and
    # reattachments to change the crate's orientation.
    ordered = sorted(
        candidates,
        key=lambda b: (
            0 if b == previous_target else 1,
            abs(b % 16 - player % 16) + abs(b // 16 - player // 16)
        )
    )
    for b in ordered:
        other_boxes = box_positions - {b}
        handoffs = handoffs_by_box.get(b)
        if handoffs is None and helpers:
            handoffs = _handoff_destinations(
                b, helpers, goals, walls, fences,
                other_boxes, enemies, bad
            )
        targets = (goals - other_boxes) | (handoffs or set())
        plan = _plan_reorient(
            player, facing, b, False, targets,
            walls, fences, other_boxes, actors
        )
        if plan is not None:
            memory["target"] = b
            return (plan[0], None, None)

    # A previously bad fence may become usable after other crates move.
    if bad and helpers:
        for b in ordered:
            other_boxes = box_positions - {b}
            handoffs = _handoff_destinations(
                b, helpers, goals, walls, fences,
                other_boxes, enemies, set()
            )
            plan = _plan_reorient(
                player, facing, b, False, handoffs,
                walls, fences, other_boxes, actors
            )
            if plan is not None:
                memory["target"] = b
                return (plan[0], None, None)

    memory.pop("target", None)

    # Move aside if the player's current position blocks a helper.
    for b, owner in owners.items():
        if owner[0] != "helper" or b in goals:
            continue
        h = owner[1]
        occupied = walls | box_positions | actors
        if _exact_helper_path(h, b, goals, occupied, fences):
            if not _exact_helper_path(
                h, b, goals, occupied | {player}, fences
            ):
                blocked = occupied | fences
                choices = []
                for d in range(4):
                    q = _neighbor(player, d)
                    if q is not None and q not in blocked:
                        distance = (
                            abs(q % 16 - b % 16)
                            + abs(q // 16 - b // 16)
                        )
                        choices.append((distance, d))
                if choices:
                    _, d = max(choices)
                    return (ACTIONS[d], None, None)

    action = _idle(player, facing, boxes, helpers, enemies, walls, fences)
    if action not in available and available:
        action = next((a for a in available if a != "RESET"), available[0])
    return (action, None, None)

from collections import deque
import heapq

_NEIGH = []
for _p in range(256):
    _x, _y = _p % 16, _p // 16
    _NEIGH.append((
        _p - 16 if _y > 0 else -1,
        _p + 16 if _y < 15 else -1,
        _p - 1 if _x > 0 else -1,
        _p + 1 if _x < 15 else -1,
    ))

_OPPOSITE = (2, 1, 4, 3)


def _emit(memory, action, held=False):
    memory["last_action"] = action
    memory["last_held"] = bool(held)
    return ("ACTION%d" % action, None, None)


def _find_goals(base):
    allowed = {}
    evidence = set()

    for p, tile in base.items():
        height = len(tile) // 4
        masks = 0
        for mask in range(16):
            good = True
            for j in range(height):
                for i in range(4):
                    border = (
                        (mask & 1 and j == 0) or
                        (mask & 2 and j == 3) or
                        (mask & 4 and i == 0) or
                        (mask & 8 and i == 3)
                    )
                    if tile[j * 4 + i] != (9 if border else 2):
                        good = False
                        break
                if not good:
                    break
            if good:
                masks |= 1 << mask
        allowed[p] = masks
        if 9 in tile and masks:
            evidence.add(p)

    if not evidence:
        return set()

    prefix = [[0] * 17 for _ in range(17)]
    for y in range(16):
        for x in range(16):
            prefix[y + 1][x + 1] = (
                prefix[y][x + 1] + prefix[y + 1][x]
                - prefix[y][x] + (1 if y * 16 + x in evidence else 0)
            )

    candidates = []
    for y0 in range(16):
        for y1 in range(y0, 16):
            for x0 in range(16):
                for x1 in range(x0, 16):
                    count = (
                        prefix[y1 + 1][x1 + 1] - prefix[y0][x1 + 1]
                        - prefix[y1 + 1][x0] + prefix[y0][x0]
                    )
                    if not count:
                        continue

                    good = True
                    unknown = 0
                    covered = []
                    for y in range(y0, y1 + 1):
                        for x in range(x0, x1 + 1):
                            p = y * 16 + x
                            mask = (
                                (1 if y == y0 else 0) |
                                (2 if y == y1 else 0) |
                                (4 if x == x0 else 0) |
                                (8 if x == x1 else 0)
                            )
                            if p in allowed:
                                if not (allowed[p] & (1 << mask)):
                                    good = False
                                    break
                                if p in evidence:
                                    covered.append(p)
                            else:
                                unknown += 1
                        if not good:
                            break

                    if good:
                        area = (x1 - x0 + 1) * (y1 - y0 + 1)
                        candidates.append(
                            (covered, unknown, area, x0, y0, x1, y1)
                        )

    remaining = set(evidence)
    goals = set()
    while remaining:
        best = None
        best_key = None
        for item in candidates:
            covered, unknown, area, x0, y0, x1, y1 = item
            gain = sum(p in remaining for p in covered)
            if not gain:
                continue
            key = (-gain, unknown, area)
            if best_key is None or key < best_key:
                best_key = key
                best = item
        if best is None:
            break
        covered, _, _, x0, y0, x1, y1 = best
        remaining.difference_update(covered)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                goals.add(y * 16 + x)
    return goals


def _read_board(obs, memory):
    base = memory.setdefault("base", {})
    walls = memory.setdefault("walls", set())
    fences = memory.setdefault("fences", set())
    old_goals = memory.get("goals", set())

    boxes = {}
    green = []
    purple = []
    player = None
    visual_face = memory.get("face", 1)
    changed = "goals" not in memory

    for y in range(16):
        height = 3 if y == 15 else 4
        for x in range(16):
            p = y * 16 + x
            tile = [
                int(obs[y * 4 + j, x * 4 + i])
                for j in range(height)
                for i in range(4)
            ]
            colors = set(tile)

            if 14 in colors:
                player = p
                top = sum(tile[i] == 0 for i in range(4))
                left = sum(tile[j * 4] == 0 for j in range(height))
                right = sum(tile[j * 4 + 3] == 0 for j in range(height))
                bottom = sum(
                    tile[(height - 1) * 4 + i] == 0 for i in range(4)
                )
                if top == 4:
                    visual_face = 1
                elif right == height:
                    visual_face = 4
                elif left == height:
                    visual_face = 3
                elif height == 4 and bottom == 4:
                    visual_face = 2
                elif height == 3 and 0 not in colors:
                    visual_face = 2
                continue

            if (tile[5] == 9 and tile[6] == 9 and
                    tile[9] == 9 and tile[10] == 9 and
                    tile[0] in (0, 3, 4, 5)):
                boxes[p] = tile[0]
                continue

            if colors == {12}:
                green.append(p)
                continue
            if 15 in colors and colors.issubset({11, 15}):
                purple.append(p)
                continue
            if colors == {5}:
                walls.add(p)
                continue
            if colors == {1, 2}:
                fences.add(p)
                continue

            if p not in base:
                base[p] = tuple(tile)
                if 9 in colors or 2 in colors or p in old_goals:
                    changed = True

    face = visual_face
    last = memory.get("last_action")
    if last in (1, 2, 3, 4) and not memory.get("last_held", False):
        face = last
    elif last is not None and memory.get("last_held", False):
        face = memory.get("face", visual_face)
    memory["face"] = face

    if changed:
        memory["goals"] = _find_goals(base)

    return (
        player, face, boxes, green, purple,
        walls, fences, memory.get("goals", set())
    )


def _walk_search(start, face, obstacles, fences):
    initial = (start, face)
    previous = {initial: None}
    distance = {initial: 0}
    queue = deque([initial])

    while queue:
        p, facing = queue.popleft()
        for d in range(1, 5):
            q = _NEIGH[p][d - 1]
            if q < 0 or q in obstacles or q in fences:
                q = p
            state = (q, d)
            if state not in previous:
                previous[state] = ((p, facing), d)
                distance[state] = distance[(p, facing)] + 1
                queue.append(state)
    return previous, distance


def _first_action(previous, state):
    first = None
    while previous[state] is not None:
        state, first = previous[state]
    return first


def _pair_search(player, box, obstacles, fences, destinations):
    if box in destinations:
        return 0, None, box

    dx = box % 16 - player % 16
    dy = box // 16 - player // 16
    if abs(dx) + abs(dy) != 1:
        return None

    blocked = set(obstacles)
    blocked.discard(player)
    blocked.discard(box)

    queue = deque([player])
    seen = {player: (0, None)}

    while queue:
        p = queue.popleft()
        bx = p % 16 + dx
        by = p // 16 + dy
        if not (0 <= bx < 16 and 0 <= by < 16):
            continue
        b = by * 16 + bx
        dist, first = seen[p]

        for d in range(1, 5):
            q = _NEIGH[p][d - 1]
            nb = _NEIGH[b][d - 1]
            if q < 0 or nb < 0:
                continue
            if q in blocked or q in fences or nb in blocked:
                continue
            if q in seen:
                continue
            f = d if first is None else first
            if nb in destinations:
                return dist + 1, f, nb
            seen[q] = (dist + 1, f)
            queue.append(q)
    return None


def _regrip_search(player, box, holding, face, obstacles, fences,
                   destinations, limit=90000):
    """
    Search over both walking and carrying states. A crate may be released,
    approached from another side, and reattached.
    """
    if not destinations:
        return None

    blocked = set(obstacles)
    blocked.discard(player)
    blocked.discard(box)

    start = (player, box, 1 if holding else 0)
    best = {start: 0}
    heap = [(0, start, 0)]
    expanded = 0

    def push(cost, state, first):
        old = best.get(state)
        if old is None or cost < old:
            best[state] = cost
            heapq.heappush(heap, (cost, state, first))

    while heap and expanded < limit:
        cost, state, first = heapq.heappop(heap)
        if best.get(state) != cost:
            continue
        expanded += 1
        p, b, held = state

        if held and b in destinations:
            return cost, first, b

        if held:
            # Release the crate.
            push(cost + 1, (p, b, 0), first or 5)

            for d in range(1, 5):
                q = _NEIGH[p][d - 1]
                nb = _NEIGH[b][d - 1]
                if q < 0 or nb < 0:
                    continue
                if q in blocked or q in fences or nb in blocked:
                    continue
                push(cost + 1, (q, nb, 1), first or d)

        else:
            for d in range(1, 5):
                q = _NEIGH[p][d - 1]
                if q < 0:
                    continue
                if q == b:
                    # Turn toward the blocked crate, then attach. If this
                    # is the initial state and we already face it, attach now.
                    if state == start and face == d:
                        push(cost + 1, (p, b, 1), first or 5)
                    else:
                        push(cost + 2, (p, b, 1), first or d)
                elif q not in blocked and q not in fences:
                    push(cost + 1, (q, b, 0), first or d)

    return None


def _green_reach(green, walls, fences):
    seen = set(green)
    queue = deque(green)
    while queue:
        p = queue.popleft()
        for q in _NEIGH[p]:
            if q >= 0 and q not in seen and q not in walls and q not in fences:
                seen.add(q)
                queue.append(q)
    return seen


def _safe_stages(box, boxes, green_reach, walls, fences, goals):
    if not green_reach or not goals:
        return set()

    other_boxes = set(boxes)
    other_boxes.discard(box)
    free_goals = set(goals) - other_boxes
    if not free_goals:
        return set()

    blocked = set(walls) | other_boxes
    result = set()
    for stage in fences:
        if stage in other_boxes:
            continue
        neighbors = [
            q for q in _NEIGH[stage]
            if q >= 0 and q in green_reach and q not in blocked
        ]
        if not neighbors:
            continue

        possible = False
        unsafe = False
        for agent in neighbors:
            path = _pair_search(agent, stage, blocked, fences, free_goals)
            if path is None:
                unsafe = True
            else:
                possible = True

        if possible and not unsafe:
            result.add(stage)
    return result


def _distance(a, b):
    return abs(a % 16 - b % 16) + abs(a // 16 - b // 16)


def _parking_action(player, boxes, green, purple, walls, fences,
                    goals, previous, walk_dist):
    if not green and not purple:
        return None

    dynamic = set(boxes) | set(green) | set(purple)
    terrain = set(walls) | set(fences)

    def danger(p):
        cost = 0
        if p in goals:
            cost += 18
        elif any(q in goals for q in _NEIGH[p] if q >= 0):
            cost += 3
        if any(_distance(p, g) <= 1 for g in green):
            cost += 8
        if any(_distance(p, b) <= 1 for b in boxes):
            cost += 5
        if any(_distance(p, e) <= 1 for e in purple):
            cost += 5

        exits = sum(q >= 0 and q not in terrain for q in _NEIGH[p])
        if exits <= 2:
            cost += 7
        elif exits == 3:
            cost += 1
        return cost

    best_score = danger(player) * 4
    best_state = None

    for state, distance in walk_dist.items():
        p, _ = state
        if p in dynamic:
            continue
        score = danger(p) * 4 + distance
        if score < best_score:
            best_score = score
            best_state = state

    if best_state is not None:
        return _first_action(previous, best_state)
    return None


def _wait_action(player, face, boxes, purple, walls, fences, green):
    front = _NEIGH[player][face - 1]
    if front not in boxes and front not in purple:
        return 5

    occupied = set(boxes) | set(purple) | set(green) | set(walls) | set(fences)
    for d in range(1, 5):
        q = _NEIGH[player][d - 1]
        if q >= 0 and q not in occupied:
            return d
    for d in range(1, 5):
        q = _NEIGH[player][d - 1]
        if q < 0 or q in occupied:
            return d
    return 1


def act(obs, available, memory):
    (player, face, boxes, green, purple,
     walls, fences, goals) = _read_board(obs, memory)

    if player is None:
        return _emit(memory, 1)

    obstacles = set(walls) | set(boxes) | set(green) | set(purple)

    # Detect when autonomous agents and crates have stopped moving.
    progress_signature = (
        tuple(sorted(boxes.items())), tuple(sorted(green)), tuple(sorted(purple))
    )
    if progress_signature == memory.get("progress_signature"):
        memory["still"] = memory.get("still", 0) + 1
    else:
        memory["still"] = 0
        memory["progress_signature"] = progress_signature
    stalled = memory["still"] >= 5

    held = None
    for q in _NEIGH[player]:
        if q >= 0 and boxes.get(q) == 0:
            held = q
            break

    green_reach = _green_reach(green, walls, fences) if green else set()
    stage_cache = {}

    def stages_for(box):
        if box not in stage_cache:
            stage_cache[box] = _safe_stages(
                box, boxes, green_reach, walls, fences, goals
            )
        return stage_cache[box]

    if held is not None:
        if held in goals:
            memory.pop("task", None)
            return _emit(memory, 5, True)

        free_goals = set(goals) - (set(boxes) - {held})
        path = _pair_search(player, held, obstacles, fences, free_goals)
        if path is not None:
            if path[0] == 0:
                return _emit(memory, 5, True)
            return _emit(memory, path[1], True)

        stages = stages_for(held)
        if stages:
            path = _pair_search(player, held, obstacles, fences, stages)
            if path is not None:
                if path[0] == 0:
                    memory.setdefault("staged", set()).add(held)
                    return _emit(memory, 5, True)
                return _emit(memory, path[1], True)

        # The current carrying orientation is inadequate. Search for a
        # release and re-grip sequence before giving up on the crate.
        path = _regrip_search(
            player, held, True, face, obstacles, fences,
            free_goals | stages
        )
        if path is not None and path[1]:
            return _emit(memory, path[1], True)
        return _emit(memory, 5, True)

    for d in range(1, 5):
        if _NEIGH[player][d - 1] in purple:
            memory["purple_chase"] = 0
            if face == d:
                return _emit(memory, 5)
            return _emit(memory, d)

    previous, walk_dist = _walk_search(player, face, obstacles, fences)

    cooldown = memory.get("purple_cooldown", 0)
    if cooldown:
        memory["purple_cooldown"] = cooldown - 1
    else:
        best_enemy = None
        for enemy in purple:
            for d in range(1, 5):
                approach = _NEIGH[enemy][_OPPOSITE[d - 1] - 1]
                if approach < 0:
                    continue
                state = (approach, d)
                if state in walk_dist:
                    candidate = (walk_dist[state], enemy, state)
                    if best_enemy is None or candidate < best_enemy:
                        best_enemy = candidate

        if best_enemy is not None and best_enemy[0] <= 16:
            chase = memory.get("purple_chase", 0) + 1
            memory["purple_chase"] = chase

            signature = (player, tuple(purple), tuple(sorted(boxes)))
            recent = memory.setdefault("chase_recent", [])
            repeated = signature in recent
            recent.append(signature)
            if len(recent) > 10:
                del recent[0]

            if chase <= 10 and not repeated:
                move = _first_action(previous, best_enemy[2])
                return _emit(memory, move if move is not None else 5)

            memory["purple_chase"] = 0
            memory["purple_cooldown"] = 10
            memory["chase_recent"] = []
        else:
            memory["purple_chase"] = 0
            memory["chase_recent"] = []

    staged = memory.setdefault("staged", set())
    staged.intersection_update(boxes.keys())

    green_handled = set()
    for box, color in boxes.items():
        if color not in (3, 5):
            continue
        for agent in green:
            if box in _NEIGH[agent]:
                free_goals = set(goals) - (set(boxes) - {box})
                if box in goals or _pair_search(
                        agent, box, obstacles, fences, free_goals) is not None:
                    green_handled.add(box)
                    break

    best = None
    previous_task = memory.get("task")

    for box in boxes:
        if box in goals:
            continue
        if not stalled and (box in green_handled or (box in staged and green)):
            continue

        free_goals = set(goals) - (set(boxes) - {box})
        for d in range(1, 5):
            approach = _NEIGH[box][_OPPOSITE[d - 1] - 1]
            if approach < 0:
                continue
            state = (approach, d)
            if state not in walk_dist:
                continue

            path = _pair_search(approach, box, obstacles, fences, free_goals)
            is_stage = False
            if path is None and fences and green:
                stages = stages_for(box)
                if stages:
                    path = _pair_search(
                        approach, box, obstacles, fences, stages
                    )
                    is_stage = path is not None
            if path is None:
                continue

            score = walk_dist[state] + path[0] + 2
            if is_stage:
                destination = path[2]
                remaining = (
                    min(_distance(destination, g) for g in goals)
                    if goals else 8
                )
                score += 5 + remaining
            if previous_task == box:
                score -= 3

            key = (score, walk_dist[state], box, d)
            if best is None or key < best[0]:
                best = (key, box, state)

    if best is not None:
        _, box, state = best
        memory["task"] = box
        move = _first_action(previous, state)
        return _emit(memory, move if move is not None else 5)

    # More expensive fallback: plan a re-grip, including recovery of a
    # stalled crate held by an autonomous agent.
    candidates = []
    for box in boxes:
        if box in goals:
            continue
        if not stalled and (box in green_handled or (box in staged and green)):
            continue

        reachable = [
            walk_dist[(q, d)]
            for d in range(1, 5)
            for q in [_NEIGH[box][_OPPOSITE[d - 1] - 1]]
            if q >= 0 and (q, d) in walk_dist
        ]
        if reachable:
            candidates.append((min(reachable), box))
    candidates.sort()

    regrip_best = None
    for _, box in candidates[:4]:
        free_goals = set(goals) - (set(boxes) - {box})
        stages = stages_for(box) if fences and green else set()
        path = _regrip_search(
            player, box, False, face, obstacles, fences,
            free_goals | stages
        )
        if path is not None and path[1]:
            penalty = 5 if path[2] in stages and path[2] not in goals else 0
            key = (path[0] + penalty, box)
            if regrip_best is None or key < regrip_best[0]:
                regrip_best = (key, box, path[1])

    if regrip_best is not None:
        _, box, move = regrip_best
        memory["task"] = box
        staged.discard(box)
        return _emit(memory, move)

    memory.pop("task", None)
    move = _parking_action(
        player, boxes, green, purple, walls, fences,
        goals, previous, walk_dist
    )
    if move is not None:
        return _emit(memory, move)

    return _emit(
        memory,
        _wait_action(player, face, boxes, purple, walls, fences, green)
    )

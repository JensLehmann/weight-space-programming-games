from collections import deque
import heapq

_DX = (0, 0, -1, 1)
_DY = (-1, 1, 0, 0)
_OPP = (1, 0, 3, 2)

_NEIGH = []
for _p in range(256):
    _x, _y = _p % 16, _p // 16
    _NEIGH.append(tuple(
        (_y + _DY[d]) * 16 + (_x + _DX[d])
        if 0 <= _x + _DX[d] < 16 and 0 <= _y + _DY[d] < 16
        else -1
        for d in range(4)
    ))


def _patterns(height):
    result = {}
    for mask in range(16):
        vals = []
        for y in range(height):
            for x in range(4):
                edge = (
                    ((mask & 1) and x == 0)
                    or ((mask & 2) and x == 3)
                    or ((mask & 4) and y == 0)
                    or ((mask & 8) and y == 3)
                )
                vals.append(9 if edge else 2)
        key = tuple(vals)
        result[key] = result.get(key, 0) | (1 << mask)
    return result


_PATTERNS = {3: _patterns(3), 4: _patterns(4)}


def _read_board(obs):
    possible = [-2] * 256
    evidence = set()
    boxes = {}
    walls = set()
    hazards = set()
    blues = []
    purples = []
    player = None
    facing = 0

    for ty in range(16):
        height = 3 if ty == 15 else 4
        for tx in range(16):
            p = ty * 16 + tx
            vals = tuple(
                int(obs[ty * 4 + y, tx * 4 + x])
                for y in range(height)
                for x in range(4)
            )
            colors = set(vals)

            if 14 in colors:
                player = p
                if all(
                    int(obs[ty * 4, tx * 4 + x]) == 0
                    for x in range(4)
                ):
                    facing = 0
                elif height == 4 and all(
                    int(obs[ty * 4 + 3, tx * 4 + x]) == 0
                    for x in range(4)
                ):
                    facing = 1
                elif all(
                    int(obs[ty * 4 + y, tx * 4]) == 0
                    for y in range(height)
                ):
                    facing = 2
                elif all(
                    int(obs[ty * 4 + y, tx * 4 + 3]) == 0
                    for y in range(height)
                ):
                    facing = 3
                else:
                    facing = 1
                possible[p] = -1
                continue

            if all(
                int(obs[ty * 4 + y, tx * 4 + x]) == 9
                for y in (1, 2)
                for x in (1, 2)
            ):
                boxes[p] = int(obs[ty * 4, tx * 4])
                possible[p] = -1
                continue

            if 12 in colors:
                blues.append(p)
                possible[p] = -1
                continue

            if 15 in colors:
                purples.append(p)
                possible[p] = -1
                continue

            if colors == {5}:
                walls.add(p)
                continue

            masks = _PATTERNS[height].get(vals, 0)
            if masks:
                possible[p] = masks
                if 9 in colors:
                    evidence.add(p)
                continue

            if 2 in colors and 1 in colors:
                hazards.add(p)

    return (
        possible, evidence, boxes, walls, hazards,
        blues, purples, player, facing
    )


def _infer_goals(possible, evidence):
    candidates = set()

    for p in evidence:
        x, y = p % 16, p // 16
        masks = possible[p]

        for m in range(16):
            if not (masks & (1 << m)):
                continue

            left, right = bool(m & 1), bool(m & 2)
            top, bottom = bool(m & 4), bool(m & 8)
            if not (left or right) or not (top or bottom):
                continue

            if left and right:
                xs = [(x, x)]
            elif left:
                xs = [(x, r) for r in range(x + 1, 16)]
            else:
                xs = [(l, x) for l in range(x)]

            if top and bottom:
                ys = [(y, y)]
            elif top:
                ys = [(y, b) for b in range(y + 1, 16)]
            else:
                ys = [(t, y) for t in range(y)]

            for x0, x1 in xs:
                for y0, y1 in ys:
                    candidates.add((x0, y0, x1, y1))

    valid = []
    for x0, y0, x1, y1 in candidates:
        cells = []
        ev = set()
        unknown = 0
        good = True

        for y in range(y0, y1 + 1):
            if not good:
                break
            for x in range(x0, x1 + 1):
                p = y * 16 + x
                cells.append(p)
                v = possible[p]

                if v == -1:
                    unknown += 1
                    continue
                if v < 0:
                    good = False
                    break

                mask = (
                    (1 if x == x0 else 0)
                    | (2 if x == x1 else 0)
                    | (4 if y == y0 else 0)
                    | (8 if y == y1 else 0)
                )
                if not (v & (1 << mask)):
                    good = False
                    break
                if p in evidence:
                    ev.add(p)

        if good and ev:
            valid.append((-len(ev), unknown, len(cells), cells, ev))

    valid.sort(key=lambda item: (item[0], item[1], item[2]))
    goals = set()
    used_evidence = set()

    for _, _, _, cells, ev in valid:
        if ev & used_evidence or any(p in goals for p in cells):
            continue
        goals.update(cells)
        used_evidence.update(ev)

    return goals


def _walk_paths(start, direction, blocked):
    paths = {(start, direction): ()}
    q = deque([(start, direction)])

    while q:
        p, old_d = q.popleft()
        path = paths[(p, old_d)]

        for d in range(4):
            n = _NEIGH[p][d]
            if n < 0 or n in blocked:
                n = p
            state = (n, d)
            if state not in paths:
                paths[state] = path + (d + 1,)
                q.append(state)

    return paths


def _distance(a, b):
    return abs(a % 16 - b % 16) + abs(a // 16 - b // 16)


def _pair_path(person, box, destinations, blocked, hazards):
    if box in destinations:
        return ()

    blocked = set(blocked)
    blocked.discard(person)
    blocked.discard(box)

    dx = box % 16 - person % 16
    dy = box // 16 - person // 16
    if abs(dx) + abs(dy) != 1:
        return None

    q = deque([person])
    paths = {person: ()}

    while q:
        p = q.popleft()
        bx, by = p % 16 + dx, p // 16 + dy
        if not (0 <= bx < 16 and 0 <= by < 16):
            continue

        b = by * 16 + bx
        path = paths[p]

        for d in (2, 3, 0, 1):
            np = _NEIGH[p][d]
            nb = _NEIGH[b][d]
            if np < 0 or nb < 0:
                continue
            if np in blocked or nb in blocked or np in hazards:
                continue
            if np in paths:
                continue

            new_path = path + (d,)
            if nb in destinations:
                return new_path

            paths[np] = new_path
            q.append(np)

    return None


def _plan_box(player, facing, target, holding, goals, fixed, hazards,
              allow_regrip=False):
    fixed = set(fixed)
    fixed.discard(target)
    fixed.discard(player)
    usable_goals = set(goals) - fixed

    if not usable_goals:
        return None

    def heuristic(box):
        return min(_distance(box, g) for g in usable_goals) + 1

    heap = []
    best = {}
    serial = 0

    def add(box, direction, cost, path):
        nonlocal serial
        key = (box, direction)
        if cost >= best.get(key, 10**9):
            return
        best[key] = cost
        serial += 1
        heapq.heappush(
            heap,
            (cost + heuristic(box), cost, serial, box, direction, path)
        )

    if holding:
        for d in range(4):
            if _NEIGH[player][d] == target:
                add(target, d, 0, ())
                break
    else:
        paths = _walk_paths(player, facing, fixed | hazards | {target})
        for d in range(4):
            stand = _NEIGH[target][_OPP[d]]
            if stand < 0:
                continue
            path = paths.get((stand, d))
            if path is not None:
                add(target, d, len(path) + 1, path + (5,))

    expansions = 0
    limit = 350 if allow_regrip else 1100

    while heap and expansions < limit:
        _, cost, _, box, d, path = heapq.heappop(heap)
        if best.get((box, d)) != cost:
            continue
        expansions += 1

        if box in usable_goals:
            return path + (5,)

        person = _NEIGH[box][_OPP[d]]
        if person < 0:
            continue

        for move in range(4):
            new_box = _NEIGH[box][move]
            new_person = _NEIGH[person][move]

            if new_box < 0 or new_person < 0:
                continue
            if new_box in fixed or new_person in fixed:
                continue
            if new_person in hazards:
                continue

            add(new_box, d, cost + 1, path + (move + 1,))

        if allow_regrip:
            paths = _walk_paths(person, d, fixed | hazards | {box})

            for nd in range(4):
                if nd == d:
                    continue
                stand = _NEIGH[box][_OPP[nd]]
                if stand < 0:
                    continue
                walk = paths.get((stand, nd))
                if walk is None:
                    continue
                segment = (5,) + walk + (5,)
                add(box, nd, cost + len(segment), path + segment)

    return None


def _enemy_plan(player, facing, enemies, blocked):
    if not enemies:
        return None

    paths = _walk_paths(player, facing, blocked)
    best = None

    for enemy in enemies:
        for d in range(4):
            stand = _NEIGH[enemy][_OPP[d]]
            if stand < 0:
                continue
            path = paths.get((stand, d))
            if path is None:
                continue
            candidate = path + (5,)
            if best is None or len(candidate) < len(best):
                best = candidate

    return best


def _helper_reach(blues, walls, hazards):
    blocked = walls | hazards
    seen = set(blues)
    q = deque(blues)

    while q:
        p = q.popleft()
        for n in _NEIGH[p]:
            if n >= 0 and n not in blocked and n not in seen:
                seen.add(n)
                q.append(n)

    return seen


def _handoff_cells(blues, walls, hazards, boxes, goals):
    if not blues or not hazards or not goals:
        return set()

    reach = _helper_reach(blues, walls, hazards)
    occupied_boxes = set(boxes)
    blocked = set(walls) | occupied_boxes
    free_goals = set(goals) - occupied_boxes
    result = set()

    for h in hazards:
        if h in occupied_boxes:
            continue

        for n in _NEIGH[h]:
            if n < 0 or n not in reach or n in blocked:
                continue
            if _pair_path(n, h, free_goals, blocked, hazards) is not None:
                result.add(h)
                break

    return result


def _helper_pairs(boxes, blues, player, facing):
    result = []
    used = set()

    for box in sorted(boxes, key=lambda p: (0 if boxes[p] == 5 else 1, p)):
        border = boxes[box]
        if border not in (3, 5):
            continue

        candidates = [
            b for b in blues if b not in used and box in _NEIGH[b]
        ]
        if not candidates:
            continue

        if border == 3 and _NEIGH[player][facing] != box:
            continue

        blue = candidates[0]
        used.add(blue)
        result.append((blue, box))

    return result


def _choose_plan(player, facing, held, targets, destinations, fixed,
                 hazards, previous_target, regrip=False):
    if not targets or not destinations:
        return None, None

    def priority(p):
        estimate = _distance(player, p) + min(
            _distance(p, g) for g in destinations
        )
        if p == previous_target:
            estimate -= 6
        return estimate

    ordered = sorted(targets, key=priority)
    ordered = ordered[:3 if regrip else 5]

    best_plan = None
    best_target = None
    best_score = 10**9

    for target in ordered:
        plan = _plan_box(
            player, facing, target, held == target,
            destinations, fixed, hazards, regrip
        )
        if plan:
            score = len(plan) - (6 if target == previous_target else 0)
            if score < best_score:
                best_score = score
                best_plan = plan
                best_target = target

    return best_plan, best_target


def _effect(action, player, facing, target, holding, fixed, hazards):
    """Predict one player action for a particular target box."""
    blocked = set(fixed)
    blocked.discard(player)
    blocked.discard(target)

    if action == 5:
        if holding:
            return player, facing, target, False
        if _NEIGH[player][facing] != target:
            return None
        return player, facing, target, True

    d = action - 1
    if not 0 <= d < 4:
        return None

    new_player = _NEIGH[player][d]

    if holding:
        new_box = _NEIGH[target][d]
        if (
            new_player < 0 or new_box < 0
            or new_player in blocked or new_box in blocked
            or new_player in hazards
        ):
            return None
        return new_player, facing, new_box, True

    if new_player < 0 or new_player in blocked or new_player in hazards:
        new_player = player
    elif new_player == target:
        new_player = player

    return new_player, d, target, False


def _make_trace(plan, player, facing, target, holding, fixed, hazards):
    trace = []
    p, d, t, h = player, facing, target, holding

    for action in plan:
        effect = _effect(action, p, d, t, h, fixed, hazards)
        if effect is None:
            break
        np, nd, nt, nh = effect
        trace.append((action, p, d, t, int(h), np, nd, nt, int(nh)))
        p, d, t, h = np, nd, nt, nh

    return trace


def _follow_trace(memory, player, facing, held, boxes, fixed, hazards):
    trace = memory.get("trace")
    if not trace:
        return None

    step = trace[0]
    action, p, d, target, holding, np, nd, nt, nh = step

    if (
        player != p or facing != d or target not in boxes
        or (held == target) != bool(holding)
        or (held is not None and held != target)
    ):
        memory.pop("trace", None)
        return None

    effect = _effect(
        action, player, facing, target, bool(holding), fixed, hazards
    )
    if effect != (np, nd, nt, bool(nh)):
        memory.pop("trace", None)
        return None

    memory["trace"] = trace[1:]
    memory["target"] = nt
    return action


def _wait_action(player, facing, boxes, purples, fixed, hazards):
    front = _NEIGH[player][facing]
    if front not in boxes and front not in purples:
        return 5

    for d in range(4):
        n = _NEIGH[player][d]
        if d != facing and (n < 0 or n in fixed or n in hazards):
            return d + 1

    for d in range(4):
        n = _NEIGH[player][d]
        if n >= 0 and n not in fixed and n not in hazards:
            return d + 1

    return 5


def act(obs, available, memory):
    (possible, evidence, boxes, walls, visible_hazards,
     blues, purples, player, facing) = _read_board(obs)

    if player is None:
        return ("ACTION5", None, None)

    goals = set(memory.get("goals", []))
    signature = tuple(possible)
    if memory.get("terrain_signature") != signature:
        goals.update(_infer_goals(possible, evidence))
        memory["goals"] = sorted(goals)
        memory["terrain_signature"] = signature

    hazards = set(memory.get("hazards", []))
    hazards.update(visible_hazards)
    memory["hazards"] = sorted(hazards)

    held = None
    for p, border in boxes.items():
        if border == 0 and p in _NEIGH[player]:
            held = p
            break

    front = _NEIGH[player][facing]
    if held is None and front in purples:
        memory.pop("trace", None)
        return ("ACTION5", None, None)

    if held is not None and held in goals:
        memory.pop("trace", None)
        return ("ACTION5", None, None)

    fixed = set(walls) | set(boxes) | set(blues) | set(purples)
    fixed.discard(player)
    free_goals = goals - set(boxes)

    # A carried box on a suitable barrier tile should be released.
    # Exclude the held box from occupancy when assessing the handoff.
    if held is not None and held in hazards and blues and free_goals:
        other_boxes = set(boxes)
        other_boxes.discard(held)
        valid_handoffs = _handoff_cells(
            blues, walls, hazards, other_boxes, goals
        )
        if held in valid_handoffs:
            memory.pop("trace", None)
            return ("ACTION5", None, None)

    action = _follow_trace(
        memory, player, facing, held, boxes, fixed, hazards
    )
    if action is not None:
        name = "ACTION%d" % action
        if name in available:
            return (name, None, None)

    helper_pairs = _helper_pairs(boxes, blues, player, facing)
    productive = set()
    player_blocking = []

    for blue, box in helper_pairs:
        if box in goals:
            productive.add(box)
            continue

        path = _pair_path(
            blue, box, free_goals, fixed | {player}, hazards
        )
        if path is not None:
            productive.add(box)
            continue

        path_without_player = _pair_path(
            blue, box, free_goals, fixed, hazards
        )
        if path_without_player is not None:
            productive.add(box)
            player_blocking.append((blue, box))

    if held is None and player_blocking:
        for d in range(4):
            n = _NEIGH[player][d]
            if n < 0 or n in fixed or n in hazards:
                continue

            good = True
            for blue, box in player_blocking:
                if _pair_path(
                    blue, box, free_goals, fixed | {n}, hazards
                ) is None:
                    good = False
                    break

            if good:
                name = "ACTION%d" % (d + 1)
                if name in available:
                    return (name, None, None)

    # Keep pursuing the same opponent when it moves one tile.
    chosen_enemy = None
    old_enemy = memory.get("enemy")
    if purples:
        if old_enemy is not None:
            nearby = [e for e in purples if _distance(e, old_enemy) <= 1]
            if nearby:
                chosen_enemy = min(nearby, key=lambda e: _distance(e, old_enemy))
        if chosen_enemy is None:
            chosen_enemy = min(purples, key=lambda e: _distance(player, e))
        memory["enemy"] = chosen_enemy
    else:
        memory.pop("enemy", None)

    if held is None and chosen_enemy is not None:
        enemy_path = _enemy_plan(
            player, facing, [chosen_enemy], fixed | hazards
        )
        if enemy_path and len(enemy_path) <= 35:
            name = "ACTION%d" % enemy_path[0]
            if name in available:
                return (name, None, None)

    previous_target = memory.get("target")
    previous_boxes = set(memory.get("previous_boxes", []))

    if previous_target is not None and previous_target not in boxes:
        new_boxes = set(boxes) - previous_boxes
        moved = [p for p in new_boxes if p in _NEIGH[previous_target]]
        if len(moved) == 1:
            previous_target = moved[0]

    memory["previous_boxes"] = sorted(boxes)

    if held is not None:
        targets = [held]
    else:
        targets = [
            p for p in boxes if p not in goals and p not in productive
        ]

    plan, target = _choose_plan(
        player, facing, held, targets, free_goals, fixed,
        hazards, previous_target, False
    )

    if plan is None:
        plan, target = _choose_plan(
            player, facing, held, targets, free_goals, fixed,
            hazards, previous_target, True
        )

    if plan is None and hazards and blues and free_goals:
        handoffs = _handoff_cells(blues, walls, hazards, boxes, goals)

        plan, target = _choose_plan(
            player, facing, held, targets, handoffs, fixed,
            hazards, previous_target, False
        )

        if plan is None:
            plan, target = _choose_plan(
                player, facing, held, targets, handoffs, fixed,
                hazards, previous_target, True
            )

    if plan:
        memory["target"] = target
        trace = _make_trace(
            plan, player, facing, target, held == target, fixed, hazards
        )
        if trace:
            memory["trace"] = trace[1:]
            action = trace[0][0]
        else:
            memory.pop("trace", None)
            action = plan[0]

        name = "ACTION%d" % action
        if name in available:
            return (name, None, None)

    if held is not None:
        memory.pop("trace", None)
        return ("ACTION5", None, None)

    if purples:
        enemy_path = _enemy_plan(player, facing, purples, fixed | hazards)
        if enemy_path:
            name = "ACTION%d" % enemy_path[0]
            if name in available:
                return (name, None, None)

    action = _wait_action(player, facing, boxes, purples, fixed, hazards)
    name = "ACTION%d" % action
    if name not in available:
        name = "ACTION5"
    return (name, None, None)

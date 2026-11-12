import heapq
import time
from collections import deque

W = 22
N = W * W

SHAPES = {
    "s": ((0, 0),),
    "v": ((0, 0), (0, 1)),
    "h": ((0, 0), (1, 0)),
    "q": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "b": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "x": ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
}
DIMS = {
    "s": (3, 3), "v": (3, 6), "h": (6, 3),
    "q": (6, 6), "b": (6, 6), "x": (9, 9),
}
DIRS = (
    (0, -1, "ACTION1"),
    (0, 1, "ACTION2"),
    (-1, 0, "ACTION3"),
    (1, 0, "ACTION4"),
)


def make_masks():
    result = {}
    for kind, cells in SHAPES.items():
        masks = [0] * N
        for p in range(N):
            x, y = p % W, p // W
            mask = 0
            for dx, dy in cells:
                if x + dx >= W or y + dy >= W:
                    mask = 0
                    break
                mask |= 1 << ((y + dy) * W + x + dx)
            masks[p] = mask
        result[kind] = masks
    return result


MASKS = make_masks()


def read_pieces(obs):
    candidates = []
    for kind in ("q", "v", "h", "s"):
        w, h = DIMS[kind]
        cw = 2 if w == 6 else 1
        ch = 2 if h == 6 else 1
        mx = (w - cw) // 2
        my = (h - ch) // 2
        for y in range(64 - h):
            for x in range(65 - w):
                marker = int(obs[y + my, x + mx])
                if marker not in (0, 4, 5):
                    continue
                ok = True
                for yy in range(h):
                    for xx in range(w):
                        v = int(obs[y + yy, x + xx])
                        if mx <= xx < mx + cw and my <= yy < my + ch:
                            if v != marker:
                                ok = False
                                break
                        elif v not in (0, 14):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((kind, x, y, marker))

    if not candidates:
        return [], None, None

    votes = {}
    for kind, x, y, marker in candidates:
        key = (x % 3, y % 3)
        w, h = DIMS[kind]
        votes[key] = votes.get(key, 0) + w * h
    rx, ry = max(votes, key=votes.get)

    candidates = [
        p for p in candidates if p[1] % 3 == rx and p[2] % 3 == ry
    ]
    candidates.sort(key=lambda p: -(DIMS[p[0]][0] * DIMS[p[0]][1]))

    occupied = set()
    squares = []
    for kind, x, y, marker in candidates:
        w, h = DIMS[kind]
        pixels = {(x + xx, y + yy) for yy in range(h) for xx in range(w)}
        if pixels & occupied:
            continue
        occupied.update(pixels)
        squares.append((kind, x, y, marker))

    crosses = []
    for y in range(ry, 55, 3):
        for x in range(rx, 56, 3):
            ok = True
            for yy in range(9):
                for xx in range(9):
                    if 3 <= xx < 6 or 3 <= yy < 6:
                        if int(obs[y + yy, x + xx]) != 11:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                crosses.append(("x", x, y, None))

    bombs = []
    used = set()
    for y in range(ry, 58, 3):
        for x in range(rx, 59, 3):
            if any((x + xx, y + yy) in used
                   for yy in range(6) for xx in range(6)):
                continue
            if not all(int(obs[y + yy, x + xx]) in (12, 13)
                       for yy in range(6) for xx in range(6)):
                continue
            reds = [(xx, yy) for yy in range(6) for xx in range(6)
                    if int(obs[y + yy, x + xx]) == 12]
            stage = min(5, len(reds) // 6)
            angle = None
            if reds and stage:
                if all(yy < stage for xx, yy in reds):
                    angle = 0
                elif all(xx >= 6 - stage for xx, yy in reds):
                    angle = 90
                elif all(yy >= 6 - stage for xx, yy in reds):
                    angle = 180
                elif all(xx < stage for xx, yy in reds):
                    angle = 270
            bombs.append(("b", x, y, (stage, angle)))
            used.update((x + xx, y + yy)
                        for yy in range(6) for xx in range(6))

    squares.sort(key=lambda p: (p[2], p[1], p[0]))
    crosses.sort(key=lambda p: (p[2], p[1]))
    bombs.sort(key=lambda p: (p[2], p[1]))
    return squares + crosses + bombs, rx, ry


def target_pattern(kind):
    if kind == "x":
        rows = (
            "...#####...",
            "...#...#...",
            "...#...#...",
            "####...####",
            "#.........#",
            "#.........#",
            "#.........#",
            "####...####",
            "...#...#...",
            "...#...#...",
            "...#####...",
        )
        return [(x, y) for y, row in enumerate(rows)
                for x, c in enumerate(row) if c == "#"]

    w, h = DIMS[kind]
    w += 2
    h += 2
    return [(x, y) for y in range(h) for x in range(w)
            if x == 0 or y == 0 or x == w - 1 or y == h - 1]


def target_box(target):
    kind, x, y = target[:3]
    w, h = DIMS[kind]
    return (x - 1, y - 1, x + w + 1, y + h + 1)


def boxes_overlap(a, b):
    return (a[0] < b[2] and b[0] < a[2] and
            a[1] < b[3] and b[1] < a[3])


def read_targets(obs, rx, ry):
    found = []
    for kind in ("x", "q", "v", "h", "s"):
        pattern = target_pattern(kind)
        w, h = DIMS[kind]
        w += 2
        h += 2
        for ay in range(ry, 64 - h + 1, 3):
            y = ay - 1
            if y < 0 or y + h > 63:
                continue
            for ax in range(rx, 65 - w + 1, 3):
                x = ax - 1
                if x < 0 or x + w > 64:
                    continue
                count = 0
                valid = True
                for dx, dy in pattern:
                    v = int(obs[y + dy, x + dx])
                    if v == 4:
                        count += 1
                    elif v not in (0, 5, 11, 12, 13, 14):
                        valid = False
                        break
                if valid and count >= max(8, int(len(pattern) * 0.70)):
                    found.append((kind, ax, ay, count, len(pattern)))

    found.sort(key=lambda t: (-(t[3] / t[4]), -t[3], -t[4]))
    result = []
    for kind, x, y, count, total in found:
        box = target_box((kind, x, y))
        if any(boxes_overlap(box, target_box(t)) for t in result):
            continue
        result.append((kind, x, y, count, total))
    return result


def read_terrain(obs, rx, ry):
    walls = 0
    ice = 0
    for cy in range(W):
        py = ry + 3 * cy
        for cx in range(W):
            px = rx + 3 * cx
            bit = 1 << (cy * W + cx)
            if px < 0 or py < 0 or px + 2 >= 64 or py + 2 >= 63:
                walls |= bit
                continue
            wall = False
            frozen = False
            for yy in range(py, py + 3):
                for xx in range(px, px + 3):
                    v = int(obs[yy, xx])
                    wall |= v == 2
                    frozen |= v == 15
            if wall:
                walls |= bit
            if frozen:
                ice |= bit
    return walls, ice


def signature(kinds, state):
    n = len(kinds)
    positions = state[:n]
    active = state[n]
    objects = sorted((kinds[i], positions[i]) for i in range(n))
    return ([list(v) for v in objects] +
            [["active", kinds[active], positions[active]],
             ["stage", state[n + 1]]])


def signature_key(kinds, state):
    n = len(kinds)
    positions = state[:n]
    active = state[n]
    objects = tuple(sorted((kinds[i], positions[i]) for i in range(n)))
    return repr((objects, kinds[active], positions[active], state[n + 1]))


def distance_map(kind, goal, walls):
    masks = MASKS[kind]
    if not (0 <= goal < N) or not masks[goal] or masks[goal] & walls:
        return {}
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        x, y = p % W, p // W
        for dx, dy, _ in DIRS:
            xx, yy = x + dx, y + dy
            if not (0 <= xx < W and 0 <= yy < W):
                continue
            q = yy * W + xx
            if q not in dist and masks[q] and not (masks[q] & walls):
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def search(kinds, start, goal_kind, goal_pos, walls, ice, angles,
           rx, ry, locked, deadline, node_limit, recent):
    n = len(kinds)
    masks = [MASKS[k] for k in kinds]
    matching = [i for i, k in enumerate(kinds) if k == goal_kind]
    if not matching:
        return None, False

    dmap = distance_map(goal_kind, goal_pos, walls)

    def distance(p):
        if p in dmap:
            return dmap[p]
        return (abs(p % W - goal_pos % W) +
                abs(p // W - goal_pos // W) + 8)

    def heuristic(state):
        pos = state[:n]
        active = state[n]
        best = 1000000.0
        for i in matching:
            d = float(distance(pos[i]))
            if goal_kind == "x":
                ap = pos[active]
                approach = max(
                    0,
                    abs(ap % W - pos[i] % W) +
                    abs(ap // W - pos[i] // W) - 3
                )
                d += 0.55 * approach
            elif active != i:
                d += 1.0
            best = min(best, d)
        return best

    def reached(state):
        return any(state[i] == goal_pos for i in matching)

    def shifted(p, dx, dy):
        x, y = p % W + dx, p // W + dy
        if 0 <= x < W and 0 <= y < W:
            return y * W + x
        return -1

    def push(pos, i, dx, dy, visiting):
        if i in visiting:
            return False
        q = shifted(pos[i], dx, dy)
        if q < 0:
            return False
        mask = masks[i][q]
        if not mask or mask & walls:
            return False
        visiting.add(i)
        hits = [j for j in range(n)
                if j != i and mask & masks[j][pos[j]]]
        for j in hits:
            if not push(pos, j, dx, dy, visiting):
                visiting.remove(i)
                return False
        pos[i] = q
        visiting.remove(i)
        return True

    bombs = [i for i, k in enumerate(kinds) if k == "b"]
    blast_order = (
        [i for i, k in enumerate(kinds) if k == "x"] +
        bombs +
        [i for i, k in enumerate(kinds) if k in ("s", "v", "h", "q")]
    )

    def blast_mask(p, angle, length):
        x, y = p % W, p // W
        if angle == 0:
            cells = [(x + a, y + 2 + b)
                     for a in range(2) for b in range(length)]
        elif angle == 180:
            cells = [(x + a, y - 1 - b)
                     for a in range(2) for b in range(length)]
        elif angle == 90:
            cells = [(x - 1 - b, y + a)
                     for a in range(2) for b in range(length)]
        else:
            cells = [(x + 2 + b, y + a)
                     for a in range(2) for b in range(length)]
        mask = 0
        for xx, yy in cells:
            if 0 <= xx < W and 0 <= yy < W:
                mask |= 1 << (yy * W + xx)
        return mask

    def explode(pos):
        moved = {}
        for length in (1, 2, 3):
            for bi in bombs:
                angle = angles.get(bi, 0)
                bm = blast_mask(pos[bi], angle, length)
                dx, dy = {
                    0: (0, 1), 90: (-1, 0),
                    180: (0, -1), 270: (1, 0)
                }.get(angle, (0, 1))
                for j in blast_order:
                    if bm & masks[j][pos[j]]:
                        before = pos[:]
                        if not push(pos, j, dx, dy, set()):
                            pos[:] = before
                        moved[j] = (dx, dy)

        for _ in range(30):
            changed = False
            for j in blast_order:
                if j in moved and masks[j][pos[j]] & ice:
                    dx, dy = moved[j]
                    before = pos[:]
                    if push(pos, j, dx, dy, set()):
                        changed = True
                    else:
                        pos[:] = before
            if not changed:
                break

    def move(state, dx, dy):
        pos = list(state[:n])
        active = state[n]
        stage = state[n + 1]
        q = shifted(pos[active], dx, dy)
        dest = masks[active][q] if q >= 0 else 0

        if dest and not (dest & (walls | ice)):
            hits = [j for j in range(n)
                    if j != active and dest & masks[j][pos[j]]]
            if not hits:
                pos[active] = q
            else:
                for step in range(30):
                    all_done = True
                    for j in hits:
                        if step >= 5 and not (masks[j][pos[j]] & ice):
                            continue
                        before = pos[:]
                        if push(pos, j, dx, dy, set()):
                            all_done = False
                        else:
                            pos[:] = before
                    if all_done:
                        break

        if bombs:
            stage += 1
            if stage >= 6:
                stage = 0
                explode(pos)

        for i, p in locked.items():
            if pos[i] != p:
                return None

        result = tuple(pos) + (active, stage)
        return None if result == state else result

    def neighbours(state):
        for dx, dy, name in DIRS:
            ns = move(state, dx, dy)
            if ns is not None:
                yield ns, (name, None, None)

        active = state[n]
        for i, kind in enumerate(kinds):
            if i == active or kind not in ("s", "v", "h", "q"):
                continue
            p = state[i]
            px = rx + 3 * (p % W)
            py = ry + 3 * (p // W)
            ns = state[:n] + (i, state[n + 1])
            yield ns, ("ACTION6", py + 1, px + 1)

    h0 = heuristic(start)
    heap = [(h0, 0, 0, start)]
    best_g = {start: 0}
    parent = {}
    serial = 0
    best_state = start
    best_h = h0
    expanded = 0
    finish = None

    while heap and expanded < node_limit and time.monotonic() < deadline:
        _, g, _, state = heapq.heappop(heap)
        if g != best_g.get(state):
            continue
        if reached(state):
            finish = state
            break
        expanded += 1
        for ns, action in neighbours(state):
            ng = g + 1
            if ng >= best_g.get(ns, 10**9):
                continue
            best_g[ns] = ng
            parent[ns] = (state, action)
            h = heuristic(ns)
            if h < best_h and signature_key(kinds, ns) not in recent:
                best_h = h
                best_state = ns
            serial += 1
            heapq.heappush(heap, (ng + 1.3 * h, ng, serial, ns))

    solved = finish is not None
    if finish is None:
        if best_state == start:
            return None, False
        finish = best_state

    steps = []
    while finish != start:
        previous, action = parent[finish]
        steps.append((action, finish))
        finish = previous
    steps.reverse()
    return steps, solved


def act(obs, available, memory):
    started = time.monotonic()
    pieces, rx, ry = read_pieces(obs)
    if not pieces or rx is None:
        if "RESET" in available:
            return ("RESET", None, None)
        return ("ACTION1", None, None)

    kinds = [p[0] for p in pieces]
    positions = []
    active = None
    stage = 0
    for i, (kind, x, y, marker) in enumerate(pieces):
        positions.append(((y - ry) // 3) * W + (x - rx) // 3)
        if kind in ("s", "v", "h", "q") and marker == 0:
            active = i
        if kind == "b":
            stage = marker[0]

    if active is None:
        active = next((i for i, k in enumerate(kinds)
                       if k in ("s", "v", "h", "q")), None)
    if active is None:
        return ("ACTION1", None, None)

    state = tuple(positions) + (active, stage)
    sig = signature(kinds, state)
    alignment = [rx, ry]

    previous_bombs = memory.get("bombs", [])
    new_bombs = []
    angles = {}
    newly_learned = False
    unknown_bomb = False
    for i, (kind, x, y, marker) in enumerate(pieces):
        if kind != "b":
            continue
        angle = marker[1]
        old = None
        if previous_bombs:
            old = min(previous_bombs,
                      key=lambda b: abs(b[0] - x) + abs(b[1] - y))
            if abs(old[0] - x) + abs(old[1] - y) > 18:
                old = None
        if angle is None and old is not None:
            angle = old[2]
        elif marker[1] is not None and (old is None or old[2] is None):
            newly_learned = True
        if angle is None:
            unknown_bomb = True
            angles[i] = 0
        else:
            angles[i] = int(angle)
        new_bombs.append([x, y, angle])
    memory["bombs"] = new_bombs

    current_key = signature_key(kinds, state)
    recent_list = memory.get("recent", [])
    recent_list.append(current_key)
    recent_list = recent_list[-50:]
    memory["recent"] = recent_list
    recent = set(recent_list)

    plan = memory.get("plan")
    if plan and memory.get("expected") == sig and not newly_learned:
        action, next_sig = plan.pop(0)
        memory["plan"] = plan
        memory["expected"] = next_sig
        if action[0] in available:
            return tuple(action)

    memory["plan"] = []
    memory["expected"] = None

    walls, ice = read_terrain(obs, rx, ry)
    if memory.get("alignment") == alignment:
        walls |= int(memory.get("walls", 0))
        ice |= int(memory.get("ice", 0))
    memory["alignment"] = alignment
    memory["walls"] = walls
    memory["ice"] = ice

    detected = read_targets(obs, rx, ry)
    if memory.get("target_alignment") != alignment:
        targets = [(k, x, y) for k, x, y, count, total in detected]
    else:
        targets = [tuple(t) for t in memory.get("targets", [])]
        for k, x, y, count, total in detected:
            candidate = (k, x, y)
            if count != total or candidate in targets:
                continue
            if not any(boxes_overlap(target_box(candidate), target_box(t))
                       for t in targets):
                targets.append(candidate)
    memory["target_alignment"] = alignment
    memory["targets"] = [list(t) for t in targets]

    if unknown_bomb and stage == 0:
        ap = positions[active]
        for dx, dy, name in DIRS:
            xx, yy = ap % W + dx, ap // W + dy
            if not (0 <= xx < W and 0 <= yy < W):
                return (name, None, None)
            q = yy * W + xx
            mask = MASKS[kinds[active]][q]
            if not mask or mask & (walls | ice):
                return (name, None, None)
        return ("ACTION1", None, None)

    goals = []
    for kind, x, y in targets:
        gx = (x - rx) // 3
        gy = (y - ry) // 3
        if 0 <= gx < W and 0 <= gy < W:
            goals.append((kind, gy * W + gx))

    solved_indices = {}
    for kind, goal in goals:
        for i, k in enumerate(kinds):
            if k == kind and positions[i] == goal:
                solved_indices[i] = goal
                break

    tasks = []
    for kind, goal in goals:
        matches = [i for i, k in enumerate(kinds) if k == kind]
        if not matches or any(positions[i] == goal for i in matches):
            continue
        estimate = min(
            abs(positions[i] % W - goal % W) +
            abs(positions[i] // W - goal // W) +
            (0 if i == active or kind == "x" else 1)
            for i in matches
        )
        tasks.append((estimate, kind, goal))
    tasks.sort()

    if not tasks:
        return ("ACTION1", None, None)

    previous_task = memory.get("task")
    if previous_task:
        wanted = (previous_task[0], previous_task[1])
        tasks.sort(key=lambda t: (
            0 if (t[1], t[2]) == wanted else 1, t[0]
        ))

    final_steps = None
    chosen_task = None
    best_partial = None
    best_partial_task = None
    deadline = started + 8.2

    # Try to preserve solved goals, but always retry without that restriction
    # before committing to a partial path.
    passes = (True, False) if solved_indices else (False,)
    for protect in passes:
        if time.monotonic() >= deadline:
            break
        locked = solved_indices if protect else {}
        for _, kind, goal in tasks:
            now = time.monotonic()
            if now >= deadline:
                break
            if protect:
                allowance = min(1.3, deadline - now)
                node_limit = 35000
            else:
                allowance = min(4.5, deadline - now)
                node_limit = 140000
            steps, solved = search(
                kinds, state, kind, goal, walls, ice, angles,
                rx, ry, locked, now + allowance, node_limit, recent
            )
            if not steps:
                continue
            if solved:
                final_steps = steps
                chosen_task = (kind, goal)
                break
            if best_partial is None:
                best_partial = steps
                best_partial_task = (kind, goal)
        if final_steps:
            break

    if final_steps is None:
        final_steps = best_partial
        chosen_task = best_partial_task

    if not final_steps:
        memory["task"] = None
        for i, (kind, x, y, _) in enumerate(pieces):
            if i != active and kind in ("s", "v", "h", "q"):
                return ("ACTION6", y + 1, x + 1)
        return ("ACTION1", None, None)

    memory["task"] = list(chosen_task)
    encoded = [
        [list(action), signature(kinds, predicted)]
        for action, predicted in final_steps
    ]
    action, next_sig = encoded.pop(0)
    memory["plan"] = encoded
    memory["expected"] = next_sig
    if action[0] not in available:
        return ("ACTION1", None, None)
    return tuple(action)

import time
import heapq
from collections import deque


_DIRS = (
    ("ACTION1", 0, -3),
    ("ACTION2", 0, 3),
    ("ACTION3", -3, 0),
    ("ACTION4", 3, 0),
)
_DELTA = {a: (dx, dy) for a, dx, dy in _DIRS}


def _piece_mask(kind, w, h, pos, cache):
    key = (kind, w, h, pos)
    if key in cache:
        return cache[key]

    x, y = pos % 64, pos // 64
    if x < 0 or y < 0 or x + w > 64 or y + h > 63:
        cache[key] = None
        return None

    mask = 0
    if kind == "C":
        for yy in range(9):
            if 3 <= yy < 6:
                mask |= ((1 << 9) - 1) << ((y + yy) * 64 + x)
            else:
                mask |= 7 << ((y + yy) * 64 + x + 3)
    else:
        bits = (1 << w) - 1
        for yy in range(h):
            mask |= bits << ((y + yy) * 64 + x)

    cache[key] = mask
    return mask


def _read_pieces(obs):
    candidates = []

    for h, w in ((6, 6), (6, 3), (3, 6), (3, 3)):
        ys = (1,) if h == 3 else (2, 3)
        xs = (1,) if w == 3 else (2, 3)

        for y in range(64 - h):
            for x in range(65 - w):
                marker = int(obs[y + ys[0], x + xs[0]])
                if marker not in (0, 4, 5):
                    continue
                if any(int(obs[y + yy, x + xx]) != marker
                       for yy in ys for xx in xs):
                    continue

                good = True
                n14 = 0
                for yy in range(h):
                    for xx in range(w):
                        if yy in ys and xx in xs:
                            continue
                        v = int(obs[y + yy, x + xx])
                        border = (
                            yy == 0 or yy == h - 1
                            or xx == 0 or xx == w - 1
                        )
                        if v == 14:
                            n14 += 1
                        elif not (border and v == 0):
                            good = False
                            break
                    if not good:
                        break
                if good and n14:
                    candidates.append(
                        ("P", w, h, x, y, marker == 0, n14)
                    )

    candidates.sort(
        key=lambda p: (-(p[1] * p[2]), -p[6], p[4], p[3])
    )
    pieces = []
    occupied = set()

    for kind, w, h, x, y, selected, _ in candidates:
        cells = {
            (x + xx, y + yy) for yy in range(h) for xx in range(w)
        }
        if cells & occupied:
            continue
        pieces.append((kind, w, h, x, y, selected))
        occupied.update(cells)

    for y in range(55):
        for x in range(56):
            if int(obs[y, x + 3]) != 11 or int(obs[y + 4, x]) != 11:
                continue
            good = True
            cells = []
            for yy in range(9):
                for xx in range(9):
                    if 3 <= xx < 6 or 3 <= yy < 6:
                        if int(obs[y + yy, x + xx]) != 11:
                            good = False
                            break
                        cells.append((x + xx, y + yy))
                if not good:
                    break
            if good and not (set(cells) & occupied):
                pieces.append(("C", 9, 9, x, y, False))
                occupied.update(cells)

    for y in range(58):
        for x in range(59):
            if int(obs[y, x]) not in (12, 13):
                continue
            cells = {
                (x + xx, y + yy) for yy in range(6) for xx in range(6)
            }
            if cells & occupied:
                continue
            if all(
                int(obs[y + yy, x + xx]) in (12, 13)
                for yy in range(6) for xx in range(6)
            ):
                pieces.append(("B", 6, 6, x, y, False))
                occupied.update(cells)

    pieces.sort(key=lambda p: (p[0], p[2], p[1], p[4], p[3]))
    return pieces


_CROSS_OUTLINE = [
    (x, y)
    for y, row in enumerate((
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
    ))
    for x, ch in enumerate(row) if ch == "#"
]


def _read_goals(obs, pieces):
    covered = set()
    for kind, w, h, x, y, _ in pieces:
        for yy in range(h):
            for xx in range(w):
                if kind != "C" or 3 <= xx < 6 or 3 <= yy < 6:
                    covered.add((x + xx, y + yy))

    templates = [("C", 11, 11, _CROSS_OUTLINE)]
    for h, w in ((8, 8), (8, 5), (5, 8), (5, 5)):
        border = [
            (xx, yy)
            for yy in range(h) for xx in range(w)
            if yy == 0 or yy == h - 1 or xx == 0 or xx == w - 1
        ]
        templates.append(("P", w, h, border))

    found = []
    for kind, w, h, border in templates:
        for y in range(64 - h):
            for x in range(65 - w):
                visible = 0
                missing = 0
                for xx, yy in border:
                    if int(obs[y + yy, x + xx]) == 4:
                        visible += 1
                    elif (x + xx, y + yy) in covered:
                        missing += 1
                    else:
                        break
                else:
                    if visible >= max(6, (len(border) + 1) // 2):
                        found.append((
                            kind, w - 2, h - 2, x + 1, y + 1,
                            visible, missing, x, y, w, h
                        ))

    found.sort(key=lambda g: (-g[5], g[6], -(g[9] * g[10])))
    goals = []
    used = set()
    for g in found:
        kind, pw, ph, gx, gy, _, _, x, y, w, h = g
        if kind == "C":
            cells = {(x + xx, y + yy) for xx, yy in _CROSS_OUTLINE}
        else:
            cells = {
                (x + xx, y + yy)
                for yy in range(h) for xx in range(w)
                if yy == 0 or yy == h - 1 or xx == 0 or xx == w - 1
            }
        if len(cells & used) > 2:
            continue
        goals.append((kind, pw, ph, gx, gy))
        used.update(cells)
    return goals


def _terrain(obs, pieces, memory):
    old = memory.get("terrain")
    if not isinstance(old, list) or len(old) != 64:
        old = [[-1] * 64 for _ in range(64)]

    hidden = set()
    for kind, w, h, x, y, _ in pieces:
        for yy in range(h):
            for xx in range(w):
                if kind != "C" or 3 <= xx < 6 or 3 <= yy < 6:
                    hidden.add((x + xx, y + yy))

    grid = [list(row) for row in old]
    for y in range(63):
        for x in range(64):
            if (x, y) in hidden:
                continue
            v = int(obs[y, x])
            if v == 2:
                grid[y][x] = 2
            elif v == 15:
                grid[y][x] = 1
            elif v in (1, 4):
                if grid[y][x] == -1 or v == 1:
                    grid[y][x] = 0

    for y in range(63):
        x = 0
        while x < 64:
            if grid[y][x] != -1:
                x += 1
                continue
            start = x
            while x < 64 and grid[y][x] == -1:
                x += 1
            if (
                start > 0 and x < 64 and x - start <= 9
                and grid[y][start - 1] == 1 and grid[y][x] == 1
            ):
                for xx in range(start, x):
                    grid[y][xx] = 1

    for x in range(64):
        y = 0
        while y < 63:
            if grid[y][x] != -1:
                y += 1
                continue
            start = y
            while y < 63 and grid[y][x] == -1:
                y += 1
            if (
                start > 0 and y < 63 and y - start <= 9
                and grid[start - 1][x] == 1 and grid[y][x] == 1
            ):
                for yy in range(start, y):
                    grid[yy][x] = 1

    memory["terrain"] = grid
    walls = 0
    ice = 0
    for y in range(63):
        for x in range(64):
            if grid[y][x] == 2:
                walls |= 1 << (y * 64 + x)
            elif grid[y][x] == 1:
                ice |= 1 << (y * 64 + x)
    walls |= ((1 << 64) - 1) << (63 * 64)
    return walls, ice


def _signature(pieces, positions, selected):
    items = []
    for i, (kind, w, h, _, _, _) in enumerate(pieces):
        p = positions[i]
        items.append((kind, w, h, p % 64, p // 64))
    items.sort()
    selected_pos = positions[selected] if selected is not None else -1
    return tuple(items), selected_pos


def _observed_signature(pieces):
    positions = tuple(p[4] * 64 + p[3] for p in pieces)
    selected = next(
        (i for i, p in enumerate(pieces) if p[0] == "P" and p[5]),
        None
    )
    return _signature(pieces, positions, selected)


def _normalise_signature(sig):
    if sig is None:
        return None
    try:
        return tuple(tuple(x) for x in sig[0]), int(sig[1])
    except (TypeError, ValueError, IndexError):
        return None


def _reconstruct_path(pred, end):
    path = []
    while pred[end] is not None:
        prev, action = pred[end]
        path.append(action)
        end = prev
    path.reverse()
    return path


def _bomb_info(obs, pieces, memory):
    """
    Infer orientation and countdown from coloured bomb stripes.
    Orientation is the direction in which each bomb pushes.
    """
    previous = memory.get("bombs", [])
    used = set()
    info = []
    phases = []

    for piece in pieces:
        if piece[0] != "B":
            continue
        x, y = piece[3], piece[4]
        red = {
            (xx, yy)
            for yy in range(6) for xx in range(6)
            if int(obs[y + yy, x + xx]) == 12
        }
        direction = None
        phase = None

        if red:
            for k in range(1, 6):
                patterns = (
                    ((0, 3), {(xx, yy) for yy in range(k)
                              for xx in range(6)}),
                    ((0, -3), {(xx, yy) for yy in range(6 - k, 6)
                               for xx in range(6)}),
                    ((-3, 0), {(xx, yy) for xx in range(6 - k, 6)
                               for yy in range(6)}),
                    ((3, 0), {(xx, yy) for xx in range(k)
                              for yy in range(6)}),
                )
                for d, pattern in patterns:
                    if red == pattern:
                        direction, phase = d, k
                        break
                if direction is not None:
                    break
        else:
            phase = 0

        if direction is None:
            best = None
            for j, old in enumerate(previous):
                if j in used:
                    continue
                ox, oy, dx, dy = old
                distance = abs(x - ox) + abs(y - oy)
                if best is None or distance < best[0]:
                    best = (distance, j, (dx, dy))
            if best is not None and best[0] <= 18:
                used.add(best[1])
                direction = best[2]

        if phase is not None:
            phases.append(phase)
        info.append((x, y, direction))

    memory["bombs"] = [
        [x, y, direction[0], direction[1]]
        for x, y, direction in info if direction is not None
    ]
    phase = max(set(phases), key=phases.count) if phases else 0
    return phase, [entry[2] for entry in info]


def _rect_mask(x, y, w, h):
    if w <= 0 or h <= 0:
        return 0
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(64, x + w), min(63, y + h)
    if x0 >= x1 or y0 >= y1:
        return 0
    bits = (1 << (x1 - x0)) - 1
    result = 0
    for yy in range(y0, y1):
        result |= bits << (yy * 64 + x0)
    return result


def _solve_bombs(pieces, goals, walls, ice, phase, bomb_dirs, deadline):
    """
    Timed single-player search. The state includes all piece positions and
    the shared bomb countdown.
    """
    specs = [(p[0], p[1], p[2]) for p in pieces]
    players = [i for i, s in enumerate(specs) if s[0] == "P"]
    bombs = [i for i, s in enumerate(specs) if s[0] == "B"]
    if len(players) != 1 or not bombs or len(bombs) != len(bomb_dirs):
        return None
    if any(d is None for d in bomb_dirs):
        return None

    player = players[0]
    positions = tuple(p[4] * 64 + p[3] for p in pieces)
    goal_specs = [
        (kind, w, h, y * 64 + x) for kind, w, h, x, y in goals
    ]
    relevant = []
    for kind, w, h, gp in goal_specs:
        ids = [i for i, spec in enumerate(specs) if spec == (kind, w, h)]
        if ids:
            relevant.append((kind, gp, ids))
    if not relevant:
        return None

    unsolved = [
        g for g in relevant if not any(positions[i] == g[1] for i in g[2])
    ]
    if not unsolved:
        return None

    # First solve an unsatisfied player goal, then pursue other pieces.
    player_goals = [g for g in unsolved if g[0] == "P"]
    if player_goals:
        target = min(
            player_goals,
            key=lambda g: (
                abs(positions[player] % 64 - g[1] % 64)
                + abs(positions[player] // 64 - g[1] // 64)
            )
        )
    else:
        target = min(
            unsolved,
            key=lambda g: min(
                abs(positions[i] % 64 - g[1] % 64)
                + abs(positions[i] // 64 - g[1] // 64)
                for i in g[2]
            )
        )

    target_kind, target_pos, target_ids = target
    cache = {}
    n = len(pieces)

    def mask(i, pos):
        k, w, h = specs[i]
        return _piece_mask(k, w, h, pos, cache)

    def distance(a, b):
        return abs(a % 64 - b % 64) + abs(a // 64 - b // 64)

    def heuristic(pos):
        if target_kind == "P":
            return distance(pos[player], target_pos) / 3.0
        best = 10000.0
        for i in target_ids:
            piece_dist = distance(pos[i], target_pos) / 3.0
            approach = max(0, distance(pos[player], pos[i]) - 9) / 3.0
            value = 2.0 * piece_dist + 0.85 * approach
            if value < best:
                best = value
        return best

    def move_one(out, i, dx, dy, visiting):
        if i in visiting:
            return False
        old = out[i]
        nx = old % 64 + dx
        ny = old // 64 + dy
        if nx < 0 or ny < 0 or nx >= 64 or ny >= 63:
            return False
        new = ny * 64 + nx
        m = mask(i, new)
        if m is None or m & walls:
            return False

        visiting.add(i)
        for j in range(n):
            if j == i:
                continue
            mj = mask(j, out[j])
            if mj is not None and m & mj:
                if not move_one(out, j, dx, dy, visiting):
                    visiting.remove(i)
                    return False
        out[i] = new
        visiting.remove(i)
        return True

    def transition(pos, t, dx, dy):
        out = list(pos)
        old = out[player]
        nx = old % 64 + dx
        ny = old // 64 + dy

        if 0 <= nx < 64 and 0 <= ny < 63:
            new = ny * 64 + nx
            m = mask(player, new)
            if m is not None and not (m & (walls | ice)):
                hits = []
                for j in range(n):
                    if j == player:
                        continue
                    mj = mask(j, out[j])
                    if mj is not None and m & mj:
                        hits.append(j)

                if not hits:
                    out[player] = new
                else:
                    # A push continues for five ticks, or longer on ice.
                    for first in hits:
                        for tick in range(60):
                            current = mask(first, out[first])
                            if tick >= 5 and current is not None:
                                if not (current & ice):
                                    break
                            if not move_one(out, first, dx, dy, set()):
                                break

        t += 1
        if t == 6:
            t = 0
            moved_by_blast = {}

            for wave in range(1, 4):
                reach = 3 * wave
                for bi, direction in zip(bombs, bomb_dirs):
                    bx, by = out[bi] % 64, out[bi] // 64
                    bdx, bdy = direction
                    if bdy < 0:
                        blast = _rect_mask(bx, by - reach, 6, reach)
                    elif bdy > 0:
                        blast = _rect_mask(bx, by + 6, 6, reach)
                    elif bdx < 0:
                        blast = _rect_mask(bx - reach, by, reach, 6)
                    else:
                        blast = _rect_mask(bx + 6, by, reach, 6)

                    for j in range(n):
                        mj = mask(j, out[j])
                        if mj is not None and blast & mj:
                            move_one(out, j, bdx, bdy, set())
                            moved_by_blast[j] = (bdx, bdy)

            for _ in range(60):
                changed = False
                for j, (bdx, bdy) in moved_by_blast.items():
                    mj = mask(j, out[j])
                    if mj is not None and mj & ice:
                        if move_one(out, j, bdx, bdy, set()):
                            changed = True
                if not changed:
                    break

        return tuple(out), t

    root = (positions, phase)
    h0 = heuristic(positions)
    best_partial = root
    best_partial_score = h0
    best_g = {root: 0}
    parent = {}
    serial = 0
    heap = [(2.4 * h0, 0, serial, root)]
    expanded = 0
    end = None

    while heap and time.monotonic() < deadline:
        _, cost, _, state = heapq.heappop(heap)
        if cost != best_g.get(state):
            continue
        pos, t = state

        if any(pos[i] == target_pos for i in target_ids):
            end = state
            break

        h = heuristic(pos)
        if state != root and h + 0.015 * cost < best_partial_score:
            best_partial_score = h + 0.015 * cost
            best_partial = state

        expanded += 1
        if expanded > 45000:
            break

        for action, dx, dy in _DIRS:
            newpos, newphase = transition(pos, t, dx, dy)
            ns = (newpos, newphase)
            ng = cost + 1
            if ng >= best_g.get(ns, 10**9):
                continue
            best_g[ns] = ng
            parent[ns] = (state, action)
            serial += 1
            hh = heuristic(newpos)
            heapq.heappush(heap, (ng + 2.4 * hh, ng, serial, ns))

    if end is None:
        end = best_partial
    if end == root:
        return None

    states = []
    current = end
    while current != root:
        prev, action = parent[current]
        states.append((action, current))
        current = prev
    states.reverse()

    plan = []
    for action, state in states:
        sig = _signature(pieces, state[0], player)
        plan.append((action, None, None, sig))
    return plan


def _solve(pieces, goals, walls, ice, history, deadline):
    n = len(pieces)
    specs = [(p[0], p[1], p[2]) for p in pieces]
    positions = tuple(p[4] * 64 + p[3] for p in pieces)
    active = next(
        (i for i, p in enumerate(pieces) if p[0] == "P" and p[5]),
        None
    )
    if active is None:
        active = next(
            (i for i, p in enumerate(pieces) if p[0] == "P"), None
        )
    if active is None or not goals:
        return None

    goal_specs = [(g[0], g[1], g[2], g[4] * 64 + g[3]) for g in goals]
    matching = []
    for kind, w, h, _ in goal_specs:
        ids = [i for i, s in enumerate(specs) if s == (kind, w, h)]
        if not ids:
            return None
        matching.append(ids)

    mask_cache = {}

    def pm(i, pos):
        kind, w, h = specs[i]
        return _piece_mask(kind, w, h, pos, mask_cache)

    def heuristic(pos):
        unsolved = 0
        distance = 0
        for gi, (_, _, _, gp) in enumerate(goal_specs):
            gx, gy = gp % 64, gp // 64
            best = 1000
            for i in matching[gi]:
                p = pos[i]
                d = abs(p % 64 - gx) + abs(p // 64 - gy)
                if d < best:
                    best = d
            if best:
                unsolved += 1
                distance += best / 3.0
        return unsolved * 8.0 + distance, unsolved

    def do_push(pos, first, dx, dy):
        out = list(pos)

        def move_one(i, visiting):
            if i in visiting:
                return False
            old = out[i]
            nx, ny = old % 64 + dx, old // 64 + dy
            if nx < 0 or ny < 0 or nx >= 64 or ny >= 63:
                return False
            new = ny * 64 + nx
            m = pm(i, new)
            if m is None or (m & walls):
                return False
            visiting.add(i)
            for j in range(n):
                if j == i:
                    continue
                mj = pm(j, out[j])
                if mj is not None and (m & mj):
                    if not move_one(j, visiting):
                        visiting.remove(i)
                        return False
            out[i] = new
            visiting.remove(i)
            return True

        moved = False
        for tick in range(100):
            current = pm(first, out[first])
            if tick >= 5 and current is not None and not (current & ice):
                break
            if not move_one(first, set()):
                break
            moved = True
        return tuple(out) if moved else None

    root = (positions, active)
    h0, u0 = heuristic(positions)
    serial = 0
    heap = [(3.0 * h0, 0, serial, root)]
    best_g = {root: 0}
    parent = {}
    best_state = root
    best_partial = (u0, h0, 0)
    expanded = 0
    history = set(history)

    while heap and time.monotonic() < deadline:
        _, cost, _, state = heapq.heappop(heap)
        if cost != best_g.get(state):
            continue
        pos, sel = state
        hval, unsolved = heuristic(pos)
        if unsolved == 0:
            best_state = state
            break

        partial_key = (unsolved, hval, cost)
        if state != root and cost <= 55 and partial_key < best_partial:
            best_partial = partial_key
            best_state = state

        expanded += 1
        if expanded > 3500:
            break

        for i in range(n):
            if specs[i][0] != "P":
                continue

            start = pos[i]
            other_masks = [
                pm(j, pos[j]) if j != i else None for j in range(n)
            ]
            pred = {start: None}
            dist = {start: 0}
            queue = deque([start])
            click_cost = 0 if i == sel else 1

            while queue:
                here = queue.popleft()
                d0 = dist[here]
                px, py = here % 64, here // 64

                for gi, (gkind, gw, gh, gp) in enumerate(goal_specs):
                    if specs[i] != (gkind, gw, gh):
                        continue
                    if here != gp or here == start:
                        continue
                    if any(pos[j] == gp for j in matching[gi] if j != i):
                        continue

                    newpos = list(pos)
                    newpos[i] = here
                    newpos = tuple(newpos)
                    ns = (newpos, i)
                    ng = cost + click_cost + d0
                    if ng < best_g.get(ns, 10**9):
                        sig = repr(_signature(pieces, newpos, i))
                        if sig not in history:
                            best_g[ns] = ng
                            parent[ns] = (
                                state, i, _reconstruct_path(pred, here), None
                            )
                            hh, _ = heuristic(newpos)
                            serial += 1
                            heapq.heappush(
                                heap, (ng + 3.0 * hh, ng, serial, ns)
                            )

                for action, dx, dy in _DIRS:
                    nx, ny = px + dx, py + dy
                    if nx < 0 or ny < 0 or nx >= 64 or ny >= 63:
                        continue
                    nxt = ny * 64 + nx
                    m = pm(i, nxt)
                    if m is None or m & (walls | ice):
                        continue

                    hits = []
                    for j, mj in enumerate(other_masks):
                        if mj is not None and m & mj:
                            hits.append(j)

                    if not hits:
                        if nxt not in pred:
                            pred[nxt] = (here, action)
                            dist[nxt] = d0 + 1
                            queue.append(nxt)
                    else:
                        base = list(pos)
                        base[i] = here
                        pushed = do_push(tuple(base), hits[0], dx, dy)
                        if pushed is None or pushed == tuple(base):
                            continue
                        ns = (pushed, i)
                        ng = cost + click_cost + d0 + 1
                        if ng >= best_g.get(ns, 10**9):
                            continue
                        sig = repr(_signature(pieces, pushed, i))
                        if sig in history:
                            continue
                        best_g[ns] = ng
                        parent[ns] = (
                            state, i, _reconstruct_path(pred, here), action
                        )
                        hh, _ = heuristic(pushed)
                        serial += 1
                        heapq.heappush(
                            heap, (ng + 3.0 * hh, ng, serial, ns)
                        )

            if time.monotonic() >= deadline:
                break

    if best_state == root:
        return None

    segments = []
    state = best_state
    while state != root:
        prev, i, path, push_action = parent[state]
        segments.append((prev, state, i, path, push_action))
        state = prev
    segments.reverse()

    plan = []
    for prev, end, i, path, push_action in segments:
        cur = list(prev[0])
        selected = prev[1]
        if selected != i:
            p = cur[i]
            w, h = specs[i][1], specs[i][2]
            selected = i
            sig = _signature(pieces, tuple(cur), selected)
            plan.append((
                "ACTION6", p // 64 + h // 2, p % 64 + w // 2, sig
            ))

        for action in path:
            dx, dy = _DELTA[action]
            cur[i] += dy * 64 + dx
            sig = _signature(pieces, tuple(cur), selected)
            plan.append((action, None, None, sig))

        if push_action is not None:
            cur = list(end[0])
            sig = _signature(pieces, tuple(cur), selected)
            plan.append((push_action, None, None, sig))

    return plan


def act(obs, available, memory):
    pieces = _read_pieces(obs)
    observed = _observed_signature(pieces)
    expected = _normalise_signature(memory.get("expected"))
    plan = memory.get("plan", [])

    if expected is not None and expected != observed:
        plan = []
        memory["plan"] = []

    history = memory.get("history", [])
    key = repr(observed)
    if not history or history[-1] != key:
        history.append(key)
    memory["history"] = history[-300:]

    if plan:
        action, row, col, sig = plan.pop(0)
        if action in available:
            memory["plan"] = plan
            memory["expected"] = sig
            return action, row, col
        memory["plan"] = []

    detected_goals = _read_goals(obs, pieces)
    stored_goals = memory.get("goals")
    if stored_goals is None:
        goals = detected_goals
    else:
        goals = [tuple(g) for g in stored_goals]
        for goal in detected_goals:
            if goal not in goals:
                goals.append(goal)
    memory["goals"] = goals

    walls, ice = _terrain(obs, pieces, memory)
    bombs = [p for p in pieces if p[0] == "B"]
    players = [p for p in pieces if p[0] == "P"]

    if bombs:
        phase, directions = _bomb_info(obs, pieces, memory)

        # One ordinary move reveals stripes when the initial bomb orientation
        # is not yet observable.
        if len(players) == 1 and any(d is None for d in directions):
            memory["expected"] = None
            memory["plan"] = []
            if "ACTION1" in available:
                return "ACTION1", None, None

        if len(players) == 1 and all(d is not None for d in directions):
            deadline = time.monotonic() + 7.5
            plan = _solve_bombs(
                pieces, goals, walls, ice, phase, directions, deadline
            )
            if plan:
                action, row, col, sig = plan.pop(0)
                if action in available:
                    memory["plan"] = plan
                    memory["expected"] = sig
                    return action, row, col

    deadline = time.monotonic() + 7.5
    plan = _solve(pieces, goals, walls, ice, memory["history"], deadline)
    if plan:
        action, row, col, sig = plan.pop(0)
        if action in available:
            memory["plan"] = plan
            memory["expected"] = sig
            return action, row, col

    memory["expected"] = None
    memory["plan"] = []
    count = int(memory.get("fallback", 0))
    memory["fallback"] = count + 1
    choices = [a for a, _, _ in _DIRS if a in available]
    if choices:
        return choices[count % len(choices)], None, None
    if "RESET" in available:
        return "RESET", None, None
    return available[0], None, None

import heapq
import itertools
from collections import Counter, deque

import numpy as np


def _cell(x, y):
    return y * 64 + x


def _xy(p):
    return p % 64, p // 64


def _locate_grid(obs):
    a = np.asarray(obs)
    hx = np.count_nonzero(a[1:63, 1:] != a[1:63, :-1], axis=0).astype(float)
    hy = np.count_nonzero(a[1:63, :] != a[:62, :], axis=1).astype(float)
    hx[31] = 0
    hy[31] = 0
    total = float(hx.sum() + hy.sum()) + 1.0
    best = None

    for scale in range(2, 9):
        bx = by = None
        for off in range(scale):
            gx = sum(hx[i - 1] for i in range(1, 64) if i % scale == off)
            gy = sum(hy[i - 1] for i in range(1, 63) if i % scale == off)
            if bx is None or gx > bx[0]:
                bx = (gx, off)
            if by is None or gy > by[0]:
                by = (gy, off)
        score = (total - 1.0 - bx[0] - by[0]) / total + 0.08 / scale
        if best is None or score < best[0]:
            best = (score, scale, bx[1], by[1])

    return best[1], best[2], best[3]


def _read_board(obs, memory):
    a = np.asarray(obs, dtype=np.int16)
    if "grid" not in memory:
        memory["grid"] = _locate_grid(a)

    scale, ox, oy = memory["grid"]
    width = (64 - ox) // scale
    height = (64 - oy) // scale
    corners = (
        int(a[1, 1]), int(a[1, 62]),
        int(a[62, 1]), int(a[62, 62]),
    )

    tiles = dict(memory.get("tiles", {}))
    players = {}
    buttons = []
    selected = -1

    for y in range(height):
        sy = oy + y * scale
        for x in range(width):
            sx = ox + x * scale
            p = _cell(x, y)
            block = a[sy:sy + scale, sx:sx + scale]
            if block.shape != (scale, scale):
                continue

            yy = np.arange(sy, sy + scale)[:, None]
            xx = np.arange(sx, sx + scale)[None, :]
            expected = np.where(
                yy < 32,
                np.where(xx < 32, corners[0], corners[1]),
                np.where(xx < 32, corners[2], corners[3]),
            )
            valid = (yy != 0) & (yy != 63)
            is_wall = bool(np.all((block == expected) | ~valid))
            center = int(block[scale // 2, scale // 2])
            vals, counts = np.unique(block, return_counts=True)
            count = {int(v): int(c) for v, c in zip(vals, counts)}

            is_button = False
            if center in (9, 11) and scale >= 3:
                inner = block[1:-1, 1:-1]
                border = np.concatenate(
                    (block[0, :], block[-1, :],
                     block[1:-1, 0], block[1:-1, -1])
                )
                if inner.size and np.all(inner == center):
                    if np.count_nonzero(border == 5) >= max(1, len(border) - 2):
                        is_button = True

            if is_button:
                buttons.append(p)
                if center == 11:
                    selected = p
                if p not in tiles:
                    tiles[p] = ("floor", 5)
                continue

            if is_wall:
                tiles[p] = ("wall", 0)
                continue

            if center in (1, 10) and count.get(center, 0) >= scale * scale - 1:
                players[p] = center
                if p not in tiles:
                    tiles[p] = ("floor", 5)
                continue

            if count.get(5, 0) >= scale * scale - 1:
                if tiles.get(p, (None,))[0] != "gate":
                    tiles[p] = ("floor", 5)
                continue

            if len(vals) >= 2 and 5 in count:
                checker = np.fromfunction(
                    lambda iy, ix: ((iy + sy + ix + sx) % 2) == 1,
                    (scale, scale),
                    dtype=int,
                )
                if np.all(block[checker] == 5) and np.any(block[~checker] != 5):
                    tiles[p] = ("trap", 0)
                    continue

            if len(vals) == 1:
                color = int(vals[0])
                if color == 5:
                    if tiles.get(p, (None,))[0] != "gate":
                        tiles[p] = ("floor", 5)
                else:
                    prior = tiles.get(p)
                    if prior and prior[0] == "gate" and prior[1] == color:
                        tiles[p] = prior
                    else:
                        tiles[p] = ("mark", color)
            elif p not in tiles:
                tiles[p] = ("floor", 5)

    marks = {
        p: color for p, (kind, color) in tiles.items()
        if kind in ("mark", "gate")
    }
    gate_cells = {
        p for p, (kind, _) in tiles.items() if kind == "gate"
    }

    for p, color in list(marks.items()):
        x, y = _xy(p)
        for dx, dy in ((1, 0), (0, 1)):
            run = []
            for k in range(-2, 3):
                xx, yy = x + k * dx, y + k * dy
                if 0 <= xx < width and 0 <= yy < height:
                    q = _cell(xx, yy)
                    run.append(q if marks.get(q) == color else None)
                else:
                    run.append(None)
            for start in range(3):
                part = run[start:start + 3]
                if all(q is not None for q in part):
                    gate_cells.update(part)

    for p in gate_cells:
        if p in marks:
            tiles[p] = ("gate", marks[p])

    walls, traps = set(), set()
    gates, switches = {}, {}
    for p, (kind, color) in tiles.items():
        if kind == "wall":
            walls.add(p)
        elif kind == "trap":
            traps.add(p)
        elif kind == "gate":
            gates[p] = color
        elif kind == "mark":
            switches[p] = color

    memory["tiles"] = tiles
    board = {
        "width": width,
        "height": height,
        "walls": walls,
        "traps": traps,
        "gates": gates,
        "switches": switches,
    }
    return board, players, tuple(sorted(buttons)), selected


def _visible_positions(players):
    out = set()
    for p in players:
        if p >= 0:
            out.add(p)
        elif p <= -2:
            out.add(-p - 2)
    return out


def _initial_candidates(observed, buttons, selected):
    positions = sorted(observed)
    n = len(positions)
    candidates = []

    if n == 2:
        for a, b in itertools.permutations(positions):
            candidates.append(((a, b, -1, -1), buttons, selected))
    elif n == 4:
        for perm in itertools.permutations(positions):
            candidates.append((tuple(perm), buttons, selected))
    elif n == 3:
        for fixed in positions:
            active = [p for p in positions if p != fixed]
            for r1, r2 in itertools.permutations(range(4), 2):
                other = [i for i in range(4) if i not in (r1, r2)]
                ps = [-1] * 4
                ps[r1], ps[r2] = active
                ps[other[0]] = ps[other[1]] = -fixed - 2
                candidates.append((tuple(ps), buttons, selected))
    elif n == 1:
        p = positions[0]
        candidates.append(((p, p, -1, -1), buttons, selected))

    if selected >= 0:
        filtered = [
            s for s in candidates
            if _matches(s, observed, buttons, selected)
        ]
        if filtered:
            candidates = filtered
    return candidates


def _matches(state, observed, buttons, selected):
    ps, bs, sel = state
    if tuple(bs) != tuple(buttons) or sel != selected:
        return False
    if _visible_positions(ps) != set(observed):
        return False
    if selected >= 0:
        active = {p for p in ps if p >= 0}
        fixed = {-p - 2 for p in ps if p <= -2}
        for p, color in observed.items():
            if color == 1 and p not in active:
                return False
            if color == 10 and p in active and p not in fixed:
                return False
    return True


def _inside(p, board):
    x, y = _xy(p)
    return 0 <= x < board["width"] and 0 <= y < board["height"]


def _gate_open(players, board):
    return {
        board["switches"][p]
        for p in players
        if p >= 0 and p in board["switches"]
    }


def _step(state, action, board):
    players, buttons, selected = state

    if isinstance(action, (tuple, list)):
        target = action[1]
        if target in buttons:
            return players, buttons, target
        return players, buttons, -1

    dx, dy = {
        1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)
    }[action]
    opened = _gate_open(players, board)

    def blocked(p):
        if not _inside(p, board) or p in board["walls"]:
            return True
        color = board["gates"].get(p)
        return color is not None and color not in opened

    if selected >= 0:
        x, y = _xy(selected)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < board["width"] and 0 <= ny < board["height"]):
            return state
        dest = _cell(nx, ny)
        active = {p for p in players if p >= 0}
        if (blocked(dest) or dest in buttons or dest in active
                or dest in board["switches"] or dest in board["traps"]):
            return state
        bs = list(buttons)
        if selected not in bs:
            return state
        bs.remove(selected)
        bs.append(dest)
        return players, tuple(sorted(bs)), dest

    old = list(players)
    new = list(players)
    signs = ((1, 1), (-1, 1), (1, -1), (-1, -1))

    for i, p in enumerate(players):
        if p < 0:
            continue
        x, y = _xy(p)
        nx = x + dx * signs[i][0]
        ny = y + dy * signs[i][1]
        if not (0 <= nx < board["width"] and 0 <= ny < board["height"]):
            continue
        dest = _cell(nx, ny)
        if not blocked(dest) and dest not in buttons:
            new[i] = dest

    if any(p in board["traps"] for p in new if p >= 0):
        return None

    active_indices = [i for i, p in enumerate(old) if p >= 0]
    for ai, i in enumerate(active_indices):
        for j in active_indices[ai + 1:]:
            xi, yi = _xy(old[i])
            xj, yj = _xy(old[j])
            if abs(xi - xj) == 1 and yi == yj:
                if new[i] == old[j] or new[j] == old[i]:
                    ni, nj = _xy(new[i]), _xy(new[j])
                    meet = _cell(
                        (ni[0] + nj[0]) // 2,
                        (ni[1] + nj[1]) // 2,
                    )
                    new[i] = new[j] = meet

    groups = {}
    for i, p in enumerate(new):
        if p >= 0:
            groups.setdefault(p, []).append(i)
    for p, indices in groups.items():
        if len(indices) >= 2:
            for i in indices[:2]:
                new[i] = -p - 2
            for i in indices[2:]:
                new[i] = old[i]

    return tuple(new), buttons, -1


def _distance(a, b):
    ax, ay = _xy(a)
    bx, by = _xy(b)
    return abs(ax - bx) + abs(ay - by)


def _heuristic(state):
    ps, _, selected = state
    active = [p for p in ps if p >= 0]
    if not active:
        return 0
    if len(active) == 2:
        h = (_distance(active[0], active[1]) + 1) // 2
    elif len(active) == 4:
        a, b, c, d = active
        h = min(
            max((_distance(a, b) + 1) // 2, (_distance(c, d) + 1) // 2),
            max((_distance(a, c) + 1) // 2, (_distance(b, d) + 1) // 2),
            max((_distance(a, d) + 1) // 2, (_distance(b, c) + 1) // 2),
        )
    else:
        h = 1
    if selected >= 0:
        h += 1
    return h


def _search(start, board, allow_buttons=False, limit=16000):
    counter = itertools.count()
    h0 = _heuristic(start)
    heap = [(1.35 * h0, h0, next(counter), 0, start)]
    gscore = {start: 0}
    parent = {}
    best = start
    best_h = h0
    expanded = 0

    while heap and expanded < limit:
        _, _, _, g, state = heapq.heappop(heap)
        if g != gscore.get(state):
            continue
        expanded += 1
        h = _heuristic(state)
        if h < best_h:
            best_h, best = h, state
        if h == 0:
            best = state
            break

        _, buttons, selected = state
        actions = [1, 2, 3, 4]
        if allow_buttons:
            if selected >= 0:
                actions.append(("click", -1))
            else:
                actions.extend(("click", p) for p in buttons)

        for action in actions:
            nxt = _step(state, action, board)
            if nxt is None or nxt == state:
                continue
            ng = g + 1
            if ng >= gscore.get(nxt, 10**9):
                continue
            gscore[nxt] = ng
            parent[nxt] = (state, action)
            nh = _heuristic(nxt)
            heapq.heappush(
                heap, (ng + 1.35 * nh, nh, next(counter), ng, nxt)
            )

    solved = _heuristic(best) == 0
    path = []
    while best != start and best in parent:
        prev, action = parent[best]
        path.append(action)
        best = prev
    path.reverse()
    return path, solved


def _clear_route_plan(state, board):
    players, buttons, selected = state
    if selected >= 0 or not buttons:
        return None

    relaxed = (players, (), -1)
    path, solved = _search(relaxed, board, False, 20000)
    if not solved or not path:
        return None

    visited = set(_visible_positions(players))
    order = {}
    cur = relaxed
    for t, action in enumerate(path):
        cur = _step(cur, action, board)
        if cur is None:
            return None
        for p in _visible_positions(cur[0]):
            visited.add(p)
            if p in buttons and p not in order:
                order[p] = t

    if not order:
        return None

    best = None
    for button in sorted(order, key=order.get):
        initial = _step(state, ("click", button), board)
        queue = deque([(initial, [])])
        seen = {initial}

        while queue:
            current, moves = queue.popleft()
            position = current[2]
            if moves and position not in visited:
                key = (len(moves), order[button], position)
                if best is None or key < best[0]:
                    best = (key, button, moves)
                break

            if len(moves) >= 20:
                continue
            for action in (1, 2, 3, 4):
                nxt = _step(current, action, board)
                if nxt is None or nxt == current or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, moves + [action]))

    if best is None:
        return None
    _, button, moves = best
    return [("click", button)] + moves + [("click", -1)]


def _information_action(candidates, board):
    if len(candidates) <= 1 or candidates[0][2] >= 0:
        return None

    best = None
    for action in (1, 2, 3, 4):
        outcomes = []
        safe = True
        distance = 0
        for state in candidates:
            nxt = _step(state, action, board)
            if nxt is None:
                safe = False
                break
            outcomes.append(tuple(sorted(_visible_positions(nxt[0]))))
            distance += _heuristic(nxt)
        if not safe:
            continue
        counts = Counter(outcomes)
        if len(counts) <= 1:
            continue
        uncertainty = sum(v * v for v in counts.values())
        key = (uncertainty, distance, action)
        if best is None or key < best[0]:
            best = (key, action)
    return None if best is None else best[1]


def _display_click(p, memory):
    scale, ox, oy = memory["grid"]
    x, y = _xy(p)
    col = ox + x * scale + scale // 2
    row = oy + y * scale + scale // 2
    return max(0, min(63, row)), max(0, min(63, col))


def _normalize_state(state):
    return tuple(state[0]), tuple(state[1]), int(state[2])


def act(obs, available, memory):
    board, observed, buttons, selected = _read_board(obs, memory)

    candidates = []
    for raw in memory.get("predicted", []):
        state = _normalize_state(raw)
        if _matches(state, observed, buttons, selected):
            candidates.append(state)

    matched_prediction = bool(candidates)
    if not candidates:
        candidates = _initial_candidates(observed, buttons, selected)
    candidates = list(dict.fromkeys(candidates))

    plan = list(memory.get("plan", [])) if matched_prediction else []
    action = None

    if plan and candidates:
        proposed = plan.pop(0)
        if isinstance(proposed, list):
            proposed = tuple(proposed)
        nxt = _step(candidates[0], proposed, board)
        if nxt is not None and nxt != candidates[0]:
            action = proposed
        else:
            plan = []

    if action is None and candidates and selected < 0:
        action = _information_action(candidates, board)

    if action is None and candidates:
        chosen = None
        found_path = None
        partial = None

        if selected < 0:
            for state in candidates[:3]:
                path, solved = _search(state, board, False, 16000)
                if solved and path:
                    chosen, found_path = state, path
                    break
                if partial is None and path:
                    partial = (state, path)

            if found_path is None and buttons:
                for state in candidates[:3]:
                    path = _clear_route_plan(state, board)
                    if path:
                        chosen, found_path = state, path
                        break

        if found_path is None:
            for state in candidates[:2]:
                path, solved = _search(state, board, True, 30000)
                if solved and path:
                    chosen, found_path = state, path
                    break
                if partial is None and path:
                    partial = (state, path)

        if found_path:
            candidates = [chosen]
            action = found_path[0]
            plan = found_path[1:]
        elif selected >= 0:
            action = ("click", -1)
        elif partial is not None:
            candidates = [partial[0]]
            action = partial[1][0]
            plan = partial[1][1:]

    if action is None:
        for candidate_action in (1, 3, 4, 2):
            if not candidates:
                action = candidate_action
                break
            nxt = _step(candidates[0], candidate_action, board)
            if nxt is not None and nxt != candidates[0]:
                action = candidate_action
                break
    if action is None:
        action = 1

    predicted = []
    for state in candidates:
        nxt = _step(state, action, board)
        if nxt is not None:
            predicted.append(nxt)
    memory["predicted"] = list(dict.fromkeys(predicted))
    memory["plan"] = plan

    if isinstance(action, (tuple, list)):
        target = action[1]
        if target < 0:
            # A visible player is guaranteed to be inside the camera viewport.
            # Clicking it deselects a block without selecting another one.
            if observed:
                target = next(iter(observed))
            else:
                target = next(
                    (
                        p for p, tile in memory["tiles"].items()
                        if tile[0] == "floor" and p not in buttons
                    ),
                    buttons[0] if buttons else 0,
                )
        row, col = _display_click(target, memory)
        if "ACTION6" in available:
            return "ACTION6", row, col
        return "ACTION1", None, None

    name = "ACTION%d" % action
    if name not in available:
        name = next(
            (a for a in available if a.startswith("ACTION") and a != "ACTION6"),
            "ACTION1",
        )
    return name, None, None

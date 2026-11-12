import heapq
import numpy as np


E, W, B, G, R, D, Y, H, T = ".", "#", "b", "g", "r", "d", "y", "h", "t"

_LAYOUT_TEXT = [
    """
wwwwwwwwwww
wwwwwwwwwww
wwwwwwwwwww
wwwwwwwwwww
wwwwwwwwwww
mmmmmmmmmmm
oo       oo
oo       oo
oo       oo
oo       oo
oo       oo
oo       oo
oo n     oo
ooxxxxxxxoo
ooxxxxxxxoo
oo   o   oo
oo   o   oo
oovvvo   oo
oooooo   oo
ooxxxx   oo
ooxxxx   oo
ooxoooxxxoo
oo       oo
oo       oo
oo    vvvoo
oooooxooooo
oooooxooooo
oo       oo
oo      voo
ooxxxoooooo
oo       oo
oov vvvvvoo
oooxooooooo
oooxxxxxxoo
oooxooooxoo
oooxooooxoo
oo       oo
oovvv    oo
ooooo    oo
oo     xxoo
ooxxxxxxxoo
oo       oo
oo   +vvvoo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
""",
    """
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
o + g    oo
o   o    oo
oooooooo oo
oooooooo oo
oooooooo oo
oouuuuuu oo
oo    22 oo
oo       oo
oo n     oo
oooooogoooo
oo       oo
oo222ooo oo
oo     o oo
oovvv    oo
oooooo oooo
oooooo oooo
oouuuu   oo
oo       oo
oo2222122oo
oo       oo
oo       oo
oooo vvvvoo
oooo oooooo
oooo     oo
ooooooo  oo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
oooooooogoo
ooooooooooo
""",
    """
ooooooooooo
ooooooooooo
ooooooooooo
ooooooooooo
o         o
o         o
o  n      o
ooooo   ooo
o         o
o y       o
o         o
ovvvvvvv  o
oooooooo  o
o      1  o
o      oooo
o         o
o111111111o
o         o
o      oooo
o      o +o
o  y   o1oo
o         o
o         o
o         o
o         o
o         o
o         o
o         o
ov       vo
oov     voo
ooov   vooo
oooo   oooo
oooovvvoooo
ooooooooooo
ooooooooooo
ooooooooooo
ooooogooooo
ooooooooooo
ooooooooooo
"""
]

_TRANSLATE = {
    " ": E, "n": E, "o": W, "w": W, "m": W,
    "x": B, "g": G, "1": R, "2": D, "y": Y,
    "v": H, "u": H, "+": T,
}
_LAYOUTS = [
    [[_TRANSLATE.get(c, E) for c in line]
     for line in text.strip("\n").splitlines()[::-1]]
    for text in _LAYOUT_TEXT
]


def act(obs, available, memory):
    transparent = {E, D}
    clickable = {B, G, R, D, Y}

    ys, xs = np.nonzero(obs == 9)
    if not len(xs):
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        return (available[0], None, None)

    px = int(xs.min()) // 6
    player_top = int(ys.min()) - 1
    gravity = -1 if player_top >= 31 else 1

    board = memory.setdefault("board", {})
    visits = memory.setdefault("visits", {})
    previous = memory.get("previous")

    def classify(x, top):
        x0 = 6 * x
        if top < 0 or top + 5 >= 63 or x0 + 1 >= 64:
            return None
        a = obs[top + 1:top + 6, x0 + 1:min(64, x0 + 6)]
        counts = np.bincount(a.ravel(), minlength=16)

        if counts[9]:
            if counts[12] >= 2:
                return D if counts[12] < 12 else R
            return None
        if counts[7] >= 3:
            return T
        if counts[14] >= 6:
            return B
        if counts[8] >= 6:
            return G
        if counts[12] >= 10:
            return R
        if counts[12] >= 2:
            return D
        if counts[15] >= 4:
            if counts[11] or counts[0]:
                return H
            if counts[3] >= 1 and counts[15] >= 14:
                return Y
            return W
        if counts[10] >= max(3, counts[5]):
            return E
        if counts[5] + counts[3] >= max(3, a.size // 2):
            return W
        return None

    local = {}
    for k in range(-12, 13):
        top = player_top + 6 * k
        if top < 0 or top + 5 >= 63:
            continue
        for x in range(11):
            tile = classify(x, top)
            if tile is not None:
                local[(x, k)] = tile

    def identify_layout():
        best = None
        special_types = {B, G, R, D, Y, H, T}
        for index, layout in enumerate(_LAYOUTS):
            for anchor in range(len(layout)):
                good = bad = special = 0
                for (x, k), observed in local.items():
                    yy = anchor + k
                    if not (0 <= yy < len(layout)) or x >= len(layout[yy]):
                        continue
                    expected = layout[yy][x]
                    if observed == expected:
                        good += 1
                        if observed in special_types:
                            special += 1
                    else:
                        bad += 1
                if good + bad < 45 or special < 4:
                    continue
                if good < 0.82 * (good + bad):
                    continue
                score = good - 2.5 * bad + 2.0 * special
                if best is None or score > best[0]:
                    best = (score, index, anchor)
        return best

    def register_unknown(prev):
        old = np.asarray(prev.get("frame", []), dtype=np.int8)
        old_y = prev["y"]
        direction = prev.get("direction", -1)
        old_top = prev["top"]
        if old.shape != (64, 64):
            return old_y + 12 * direction

        old_px = prev["x"]
        click = prev.get("click")
        best = None

        for distance in range(12):
            dy = direction * distance
            shift = 6 * dy + old_top - player_top
            old_start = max(0, shift)
            old_end = min(63, 63 + shift)
            if old_end - old_start < 7:
                continue

            new_start = old_start - shift
            new_end = old_end - shift
            a = old[old_start:old_end, :]
            b = obs[new_start:new_end, :]

            old_rows = np.arange(old_start, old_end)[:, None]
            new_rows = np.arange(new_start, new_end)[:, None]
            cols = np.arange(64)[None, :]

            mask = np.ones(a.shape, dtype=bool)
            mask &= ~(
                (old_rows >= old_top) & (old_rows < old_top + 7)
                & (cols >= 6 * old_px) & (cols < 6 * old_px + 7)
            )
            mask &= ~(
                (new_rows >= player_top) & (new_rows < player_top + 7)
                & (cols >= 6 * px) & (cols < 6 * px + 7)
            )
            if click is not None:
                cr, cc = click
                mask &= ~(
                    (old_rows >= cr - 4) & (old_rows <= cr + 4)
                    & (cols >= cc - 4) & (cols <= cc + 4)
                )

            special = (
                (a == 3) | (a == 7) | (a == 8) | (a == 12)
                | (a == 14) | (a == 15)
                | (b == 3) | (b == 7) | (b == 8) | (b == 12)
                | (b == 14) | (b == 15)
            )
            weights = np.where(special, 3.0, 1.0) * mask
            total = float(weights.sum())
            if total < 100:
                continue
            mismatch = float((weights * (a != b)).sum()) / total
            score = 1.0 - mismatch + 0.001 * min(old_end - old_start, 40)
            if best is None or score > best[0]:
                best = (score, dy, mismatch)

        if best is not None and best[2] < 0.12:
            return old_y + best[1]

        best_match = None
        for distance in range(49):
            candidate = old_y + direction * distance
            good = bad = special = 0
            for (x, k), tile in local.items():
                old_tile = board.get((x, candidate + k))
                if old_tile is None:
                    continue
                if old_tile == tile:
                    good += 1
                    if tile not in (E, W):
                        special += 1
                else:
                    bad += 1
            if good + bad < 15 or good < 0.8 * (good + bad):
                continue
            score = good - 2 * bad + 2 * special
            if best_match is None or score > best_match[0]:
                best_match = (score, candidate)
        if best_match is not None:
            return best_match[1]
        return old_y + 20 * direction

    if previous is None:
        py = 0
        match = identify_layout()
        if match is not None:
            _, index, anchor = match
            for yy, row in enumerate(_LAYOUTS[index]):
                for xx, tile in enumerate(row):
                    board[(xx, yy - anchor)] = tile
            memory["known_layout"] = index
    elif previous.get("unknown"):
        py = register_unknown(previous)
        memory["plan"] = []
    else:
        py = previous["expected_y"]
        if px != previous["expected_x"] or gravity != previous["direction"]:
            memory["plan"] = []

    for (x, k), tile in local.items():
        board[(x, py + k)] = tile
    if (px, py) not in board:
        board[(px, py)] = E

    visit_key = (px, py, gravity)
    visits[visit_key] = visits.get(visit_key, 0) + 1

    def base_at(x, y):
        if x < 0 or x >= 11:
            return W
        return board.get((x, y))

    def tile_at(x, y, changes):
        return changes.get((x, y), base_at(x, y))

    def freeze(changes):
        return tuple(sorted((x, y, v) for (x, y), v in changes.items()))

    def thaw(packed):
        return {(x, y): v for x, y, v in packed}

    def visible(y, player_y, direction):
        pt = 36 if direction == -1 else 26
        row = pt + 6 * (y - player_y) + 3
        return 0 <= row < 63

    def fall(x, y, direction, changes):
        for _ in range(100):
            ny = y + direction
            tile = tile_at(x, ny, changes)
            if tile is None:
                return x, ny, False, True, False
            if tile in transparent:
                y = ny
                continue
            if tile == T:
                return x, ny, True, False, False
            if tile == H:
                return x, y, False, False, True
            return x, y, False, False, False
        return x, y, False, True, False

    def transition(x, y, direction, changes, name, target):
        new = dict(changes)
        nx, ny, nd = x, y, direction
        won = unknown = dead = False

        if name in ("ACTION3", "ACTION4"):
            nx = x + (-1 if name == "ACTION3" else 1)
            tile = tile_at(nx, y, changes)
            if tile == T:
                return nx, y, direction, new, True, False, False
            if tile not in transparent:
                return None
            nx, ny, won, unknown, dead = fall(nx, y, direction, new)
            return nx, ny, nd, new, won, unknown, dead

        if name != "ACTION6" or target is None:
            return None

        cx, cy = target
        if not 0 <= cx < 11:
            return None
        tile = tile_at(cx, cy, changes)
        if tile not in clickable:
            return None
        if tile != G and not visible(cy, y, direction):
            return None

        if tile == G:
            new[(cx, cy)] = E
            nd = -direction
            below = tile_at(x, y + nd, new)
            if below not in (W, B, G, R, Y):
                if below is None:
                    nx, ny, unknown = x, y + nd, True
                else:
                    nx, ny, won, unknown, dead = fall(x, y + nd, nd, new)
        elif tile == B:
            new[(cx, cy)] = E
        elif tile == R:
            new[(cx, cy)] = D
        elif tile == D:
            new[(cx, cy)] = R
        elif tile == Y:
            new[(cx, cy)] = E
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (cx + dx, cy + dy)
                if q != (x, y) and tile_at(q[0], q[1], new) == E:
                    new[q] = Y

        if tile != G and (cx, cy) == (x, y + direction):
            nx, ny, won, unknown, dead = fall(cx, cy, direction, new)

        return nx, ny, nd, new, won, unknown, dead

    def execute(name, target, remaining):
        if name not in available:
            return None
        result = transition(px, py, gravity, {}, name, target)
        if result is None:
            return None
        nx, ny, nd, changes, won, unknown, dead = result
        if dead:
            return None

        click = None
        if name == "ACTION6":
            tx, ty = target
            row = player_top + 6 * (ty - py) + 3
            col = 6 * tx + 3
            click = [int(row), int(col)]

        for pos, value in changes.items():
            board[pos] = value

        memory["plan"] = remaining if not unknown else []
        memory["previous"] = {
            "x": px,
            "y": py,
            "top": player_top,
            "expected_x": nx,
            "expected_y": ny,
            "direction": nd,
            "unknown": unknown,
            "frame": obs.tolist(),
            "click": click,
        }
        if click is not None:
            return (name, click[0], click[1])
        return (name, None, None)

    plan = memory.get("plan", [])
    if plan:
        name, target = plan[0]
        if target is not None:
            target = tuple(target)
        result = execute(name, target, plan[1:])
        if result is not None:
            return result
        memory["plan"] = []

    goals = [pos for pos, tile in board.items() if tile == T]
    missing_cache = {}

    def missing_view(y, direction):
        cache_key = (y, direction)
        if cache_key in missing_cache:
            return missing_cache[cache_key]
        lo, hi = (-6, 3) if direction == -1 else (-4, 5)
        missing = sum(
            (xx, yy) not in board
            for yy in range(y + lo, y + hi + 1)
            for xx in range(1, 10)
        ) / 9.0
        missing_cache[cache_key] = missing
        return missing

    initial_missing = missing_view(py, gravity)

    def heuristic(x, y, direction):
        if goals:
            return min(
                0.65 * abs(x - gx) + 0.32 * abs(y - gy)
                for gx, gy in goals
            )
        gain = max(0.0, missing_view(y, direction) - initial_missing)
        return -1.6 * gain - 0.12 * abs(y - py)

    def candidates(x, y, direction, changes):
        result = []
        if "ACTION3" in available:
            result.append(("ACTION3", None))
        if "ACTION4" in available:
            result.append(("ACTION4", None))
        if "ACTION6" not in available:
            return result

        cells = {
            (x - 1, y), (x + 1, y),
            (x - 1, y + direction),
            (x, y + direction),
            (x + 1, y + direction),
        }

        for (cx, cy), tile in board.items():
            current = changes.get((cx, cy), tile)
            if current == G:
                cells.add((cx, cy))
            elif (
                current == Y and abs(cx - x) <= 6
                and visible(cy, y, direction)
            ):
                cells.add((cx, cy))

        for (cx, cy), tile in changes.items():
            if tile == G:
                cells.add((cx, cy))
            elif (
                tile == Y and abs(cx - x) <= 6
                and visible(cy, y, direction)
            ):
                cells.add((cx, cy))

        for pos in sorted(cells):
            if tile_at(pos[0], pos[1], changes) in clickable:
                result.append(("ACTION6", pos))
        return result

    start = (px, py, gravity, ())
    parents = {start: None}
    costs = {start: 0}
    counter = 0
    heap = [(heuristic(px, py, gravity), 0, counter, start)]
    best_state = None
    best_terminal_path = None
    best_score = float("inf")
    solved_state = None
    solved_final = None
    processed = 0
    max_nodes = 22000 if goals else 14000
    max_depth = 75 if goals else 24

    def path_to(state):
        actions = []
        while parents[state] is not None:
            parent, action = parents[state]
            actions.append(action)
            state = parent
        actions.reverse()
        return actions

    while heap and processed < max_nodes:
        _, cost, _, state = heapq.heappop(heap)
        if cost != costs.get(state):
            continue
        processed += 1
        x, y, direction, packed = state
        changes = thaw(packed)

        if cost:
            count = visits.get((x, y, direction), 0)
            value = heuristic(x, y, direction) + 0.22 * cost + 0.5 * count
            if value < best_score:
                best_score = value
                best_state = state
                best_terminal_path = None

        if cost >= max_depth:
            continue

        for name, target in candidates(x, y, direction, changes):
            result = transition(x, y, direction, changes, name, target)
            if result is None:
                continue
            nx, ny, nd, new, won, unknown, dead = result
            if dead:
                continue

            action = (name, target)
            if won:
                solved_state = state
                solved_final = action
                break

            new_cost = cost + 1
            next_state = (nx, ny, nd, freeze(new))

            if unknown:
                count = visits.get((nx, ny, nd), 0)
                value = (
                    heuristic(nx, ny, nd)
                    + 0.22 * new_cost + 0.5 * count + 0.35
                )
                if value < best_score:
                    best_score = value
                    best_state = None
                    best_terminal_path = path_to(state) + [action]
                continue

            if new_cost >= costs.get(next_state, 10**9):
                continue
            costs[next_state] = new_cost
            parents[next_state] = (state, action)
            counter += 1
            weight = 1.8 if goals else 1.0
            priority = new_cost + weight * heuristic(nx, ny, nd)
            heapq.heappush(heap, (priority, new_cost, counter, next_state))

        if solved_final is not None:
            break

    if solved_final is not None:
        chosen = path_to(solved_state) + [solved_final]
    elif best_terminal_path is not None:
        chosen = best_terminal_path
    elif best_state is not None:
        chosen = path_to(best_state)
    else:
        chosen = []

    if chosen:
        name, target = chosen[0]
        result = execute(name, target, chosen[1:])
        if result is not None:
            return result

    for name, target in candidates(px, py, gravity, {}):
        result = execute(name, target, [])
        if result is not None:
            return result

    if "RESET" in available and not memory.get("reset_attempted"):
        memory.clear()
        memory["reset_attempted"] = True
        return ("RESET", None, None)

    for name in ("ACTION3", "ACTION4"):
        if name in available:
            return (name, None, None)
    return (available[0], None, None)

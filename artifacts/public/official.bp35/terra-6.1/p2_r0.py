from collections import deque

import numpy as np


EMPTY, WALL, BREAK, GREEN = ".", "W", "B", "G"
SOLID, GHOST, YELLOW, SPIKE, GOAL, PLAYER = "C", "P", "Y", "S", "T", "@"


def classify(block):
    counts = np.bincount(block.ravel().astype(np.int64), minlength=16)
    n = block.size
    if counts[9] >= 2:
        return PLAYER
    if counts[7] >= 2:
        return GOAL
    if counts[14] >= max(3, n // 5):
        return BREAK
    if counts[8] >= max(3, n // 5):
        return GREEN
    if counts[12] >= max(2, n // 10):
        return SOLID if counts[12] >= n * 0.5 else GHOST
    if counts[15] >= max(3, n // 5):
        return YELLOW if counts[15] >= n * 0.68 else SPIKE
    if counts[5] >= n * 0.55:
        return WALL
    return EMPTY


def read_view(obs, player_top):
    view = {}
    for dy in range(-12, 13):
        top = player_top + 6 * dy
        if top < 0 or top + 5 >= 63:
            continue
        for x in range(11):
            left = 6 * x
            if left + 1 >= 64:
                continue
            block = obs[top + 1:top + 6, left + 1:min(left + 6, 64)]
            if block.size:
                view[(x, dy)] = classify(block)
    return view


def view_key(view, px, gravity):
    parts = [str(px), str(gravity)]
    for x, y in sorted(view):
        t = view[(x, y)]
        if t == PLAYER:
            t = "@"
        parts.append(t)
    return "".join(parts)


def register_view(view, board, pending):
    old_y = int(pending.get("old_y", 0))
    direction = int(pending.get("direction", -1))
    bound = pending.get("bound")
    if bound is None:
        bound = old_y + direction
    bound = int(bound)

    best = None
    for py in range(old_y - 65, old_y + 66):
        if direction > 0 and py < bound:
            continue
        if direction < 0 and py > bound:
            continue

        score = 0.0
        total = 0.0
        overlap = 0
        for (x, dy), t in view.items():
            if t == PLAYER:
                continue
            old = board.get((x, py + dy))
            if old is None:
                continue
            overlap += 1
            weight = 3.0 if t not in (EMPTY, WALL) else 1.0
            score += weight if old == t else -1.3 * weight
            total += weight

        if overlap < 15:
            continue
        candidate = (score, -abs(py - bound), py, total)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is not None and best[0] > 0.25 * best[3]:
        return best[2]

    if board:
        ys = [p[1] for p in board]
        if direction > 0:
            return max(bound, max(ys) + 7)
        return min(bound, min(ys) - 7)
    return bound


def action_key(action, target, py, player_top):
    if action == "ACTION6":
        tx, ty = target
        return "6:%d:%d" % (player_top + 6 * (ty - py) + 3, 6 * tx + 3)
    return action


def search(board, px, py, gravity, player_top, banned):
    positions = list(board)
    bits = {p: 1 << i for i, p in enumerate(positions)}
    greens = [p for p, t in board.items() if t == GREEN]
    known_ys = [y for x, y in board if 0 <= x <= 9]
    if not known_ys:
        return None

    min_y, max_y = min(known_ys), max(known_ys)
    goal_known = GOAL in board.values()

    def tile(x, y, removed, toggled, added):
        if x < 0 or x > 10:
            return WALL
        p = (x, y)
        t = board.get(p)
        if t is None:
            return None
        bit = bits[p]
        if added & bit:
            return YELLOW
        if removed & bit:
            return EMPTY
        if toggled & bit:
            if t == SOLID:
                return GHOST
            if t == GHOST:
                return SOLID
        return t

    def drop(x, y, g, removed, toggled, added):
        for _ in range(120):
            t = tile(x, y + g, removed, toggled, added)
            if t is None:
                return "unknown", y + g
            if t in (EMPTY, GHOST):
                y += g
            elif t == GOAL:
                return "win", y + g
            elif t == SPIKE:
                return "dead", y
            else:
                return "normal", y
        return "unknown", y + g

    def visible(x, y, g, tx, ty, is_first):
        top = player_top if is_first else (36 if g < 0 else 26)
        row = top + 6 * (ty - y) + 3
        col = 6 * tx + 3
        return 0 <= row < 63 and 0 <= col < 64

    initial = (px, py, gravity, 0, 0, 0)
    queue = deque([(initial, None, 0)])
    seen = {initial}
    best_frontier = None
    best_unknown = None
    examined = 0

    while queue and examined < 9000:
        state, first_action, depth = queue.popleft()
        x, y, g, removed, toggled, added = state
        examined += 1
        if depth >= 24:
            continue

        is_first = first_action is None
        outcomes = []

        for dx, action in ((-1, "ACTION3"), (1, "ACTION4")):
            nx = x + dx
            t = tile(nx, y, removed, toggled, added)
            if t == GOAL:
                outcomes.append(
                    ("win", action, None, nx, y, g,
                     removed, toggled, added, None)
                )
            elif t in (EMPTY, GHOST):
                status, ny = drop(nx, y, g, removed, toggled, added)
                outcomes.append(
                    (status, action, None, nx, ny, g,
                     removed, toggled, added, ny)
                )

        candidates = {
            (x, y + g),
            (x - 1, y),
            (x + 1, y),
            (x - 1, y + g),
            (x + 1, y + g),
        }
        for gx, gy in greens:
            if visible(x, y, g, gx, gy, is_first):
                candidates.add((gx, gy))

        for tx, ty in sorted(candidates):
            if not visible(x, y, g, tx, ty, is_first):
                continue
            t = tile(tx, ty, removed, toggled, added)
            if t not in (BREAK, GREEN, SOLID, GHOST, YELLOW):
                continue

            bit = bits.get((tx, ty))
            if bit is None:
                continue
            nr, nt, na, ng = removed, toggled, added, g

            if t in (BREAK, GREEN):
                nr |= bit
            elif t in (SOLID, GHOST):
                nt ^= bit
            else:
                if na & bit:
                    na &= ~bit
                else:
                    nr |= bit
                for ax, ay in (
                    (tx - 1, ty), (tx + 1, ty),
                    (tx, ty - 1), (tx, ty + 1)
                ):
                    if (ax, ay) == (x, y):
                        continue
                    if tile(ax, ay, nr, nt, na) == EMPTY:
                        neighbour_bit = bits.get((ax, ay))
                        if neighbour_bit is not None:
                            na |= neighbour_bit

            status, ny, bound = "normal", y, None

            if t == GREEN:
                ng = -g
                support = tile(x, y + ng, removed, toggled, added)
                if support is None:
                    status, ny, bound = "unknown", y + ng, y + ng
                elif support in (EMPTY, GHOST):
                    status, ny = drop(x, y + ng, ng, nr, nt, na)
                    bound = ny
                elif support in (SPIKE, GOAL):
                    continue
            elif tx == x and ty == y + g:
                if t == GHOST:
                    continue
                status, ny = drop(x, y + g, g, nr, nt, na)
                bound = ny

            outcomes.append(
                (status, "ACTION6", (tx, ty), x, ny, ng,
                 nr, nt, na, bound)
            )

        for outcome in outcomes:
            status, action, target, nx, ny, ng, nr, nt, na, bound = outcome
            if status == "dead":
                continue

            if first_action is None:
                if action_key(action, target, py, player_top) in banned:
                    continue
                first = {
                    "action": action,
                    "target": target,
                    "x": nx,
                    "y": None if status == "unknown" else ny,
                    "gravity": ng,
                    "removed": nr,
                    "toggled": nt,
                    "added": na,
                    "bound": bound if status == "unknown" else None,
                }
            else:
                first = first_action

            if status == "win":
                return first, positions, bits

            if status == "unknown":
                if best_unknown is None:
                    best_unknown = first
                continue

            next_state = (nx, ny, ng, nr, nt, na)
            if next_state in seen:
                continue
            seen.add(next_state)

            if ng < 0:
                low, high = ny - 6, ny + 3
            else:
                low, high = ny - 4, ny + 5

            if low < min_y or high > max_y:
                if best_frontier is None:
                    best_frontier = first
                if not goal_known:
                    return first, positions, bits

            queue.append((next_state, first, depth + 1))

    chosen = best_frontier if best_frontier is not None else best_unknown
    if chosen is None:
        return None
    return chosen, positions, bits


def yellow_growth(board, px, py, gravity, player_top, banned):
    """Grow a yellow block toward an unsafe landing above spikes."""
    gaps = []
    for direction in (-1, 1):
        for distance in range(1, 10):
            x = px + direction * distance
            if x < 0 or x > 10:
                break
            if board.get((x, py)) not in (EMPTY, GHOST):
                break

            y = py
            for _ in range(30):
                t = board.get((x, y + gravity))
                if t in (EMPTY, GHOST):
                    y += gravity
                else:
                    if t == SPIKE:
                        gaps.append((distance, x, y))
                    break
            if y != py:
                break

    if not gaps:
        return None

    _, target_x, target_y = min(gaps)
    choices = []

    for (x, y), t in board.items():
        if t != YELLOW:
            continue
        row = player_top + 6 * (y - py) + 3
        col = 6 * x + 3
        if not (0 <= row < 63 and 0 <= col < 64):
            continue
        if (x, y) == (px, py + gravity):
            continue
        if action_key("ACTION6", (x, y), py, player_top) in banned:
            continue

        old_dist = abs(x - target_x) + abs(y - target_y)
        for nx, ny in (
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1)
        ):
            if (nx, ny) == (px, py):
                continue
            if board.get((nx, ny)) != EMPTY:
                continue
            new_dist = abs(nx - target_x) + abs(ny - target_y)
            if new_dist < old_dist:
                choices.append((new_dist, old_dist, x, y))

    if not choices:
        return None
    _, _, x, y = min(choices)
    return (x, y)


def fallback_move(board, px, py, gravity, player_top, banned, available):
    for dx, action in ((-1, "ACTION3"), (1, "ACTION4")):
        if action not in available or action in banned:
            continue
        x = px + dx
        if board.get((x, py)) not in (EMPTY, GHOST, GOAL):
            continue
        y = py
        safe = True
        for _ in range(30):
            t = board.get((x, y + gravity))
            if t in (EMPTY, GHOST):
                y += gravity
            elif t == SPIKE:
                safe = False
                break
            else:
                break
        if safe:
            return action
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    available = set(available)

    ys, xs = np.where(obs[:63] == 9)
    if len(xs) == 0:
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        if "ACTION3" in available:
            return "ACTION3", None, None
        return next(iter(available)), None, None

    px = int(xs.min()) // 6
    player_top = int(ys.min()) - 1
    gravity = -1 if player_top >= 31 else 1
    view = read_view(obs, player_top)

    board = {}
    for entry in memory.get("board", []):
        if len(entry) == 3:
            x, y, t = entry
            board[(int(x), int(y))] = t

    pending = memory.get("pending")
    if not board or not pending:
        py = 0
    elif pending.get("expected_y") is not None:
        if pending.get("expected_x") == px:
            py = int(pending["expected_y"])
        else:
            py = int(pending.get("old_y", 0))
    else:
        py = register_view(view, board, pending)

    for (x, dy), t in view.items():
        p = (x, py + dy)
        if t == PLAYER:
            if p not in board:
                board[p] = EMPTY
        else:
            board[p] = t

    key = view_key(view, px, gravity)
    tried = memory.setdefault("tried", {})
    counts = tried.get(key, {})
    banned = {a for a, count in counts.items() if count >= 2}

    # A choice that produced exactly the same observation is a no-op.
    if memory.get("last_key") == key:
        last_action = memory.get("last_action")
        if last_action:
            banned.add(last_action)

    result = search(board, px, py, gravity, player_top, banned)

    if result is None and "ACTION6" in available:
        target = yellow_growth(board, px, py, gravity, player_top, banned)
        if target is not None:
            akey = action_key("ACTION6", target, py, player_top)
            counts = tried.setdefault(key, {})
            counts[akey] = counts.get(akey, 0) + 1
            memory["last_key"] = key
            memory["last_action"] = akey
            memory["board"] = [[x, y, t] for (x, y), t in board.items()]
            memory["pending"] = {
                "old_y": py,
                "expected_y": py,
                "expected_x": px,
                "direction": gravity,
            }
            if len(tried) > 75:
                del tried[next(iter(tried))]
            tx, ty = target
            return "ACTION6", int(player_top + 6 * (ty - py) + 3), int(6 * tx + 3)

    if result is None:
        move = fallback_move(
            board, px, py, gravity, player_top, banned, available
        )
        if move is not None:
            counts = tried.setdefault(key, {})
            counts[move] = counts.get(move, 0) + 1
            memory["last_key"] = key
            memory["last_action"] = move
            memory["board"] = [[x, y, t] for (x, y), t in board.items()]
            memory["pending"] = {
                "old_y": py,
                "expected_y": None,
                "expected_x": px + (-1 if move == "ACTION3" else 1),
                "old_x": px,
                "direction": gravity,
                "bound": py,
            }
            if len(tried) > 75:
                del tried[next(iter(tried))]
            return move, None, None

        if banned:
            result = search(board, px, py, gravity, player_top, set())

    if result is None:
        if "RESET" in available and not memory.get("used_reset"):
            memory.clear()
            memory["used_reset"] = True
            return "RESET", None, None
        if "ACTION3" in available:
            return "ACTION3", None, None
        if "ACTION4" in available:
            return "ACTION4", None, None
        return next(iter(available)), None, None

    first, positions, bits = result
    action = first["action"]
    target = first["target"]

    if action not in available:
        if "ACTION3" in available:
            return "ACTION3", None, None
        if "ACTION4" in available:
            return "ACTION4", None, None
        return next(iter(available)), None, None

    akey = action_key(action, target, py, player_top)
    counts = tried.setdefault(key, {})
    counts[akey] = counts.get(akey, 0) + 1
    memory["last_key"] = key
    memory["last_action"] = akey
    if len(tried) > 75:
        del tried[next(iter(tried))]

    removed = first["removed"]
    toggled = first["toggled"]
    added = first["added"]

    for p in positions:
        bit = bits[p]
        base = board[p]
        if added & bit:
            board[p] = YELLOW
        elif removed & bit:
            board[p] = EMPTY
        elif toggled & bit:
            if base == SOLID:
                board[p] = GHOST
            elif base == GHOST:
                board[p] = SOLID

    memory["board"] = [[x, y, t] for (x, y), t in board.items()]
    memory["pending"] = {
        "old_y": py,
        "expected_y": first["y"],
        "expected_x": first["x"],
        "direction": first["gravity"],
        "bound": first["bound"],
    }

    if action == "ACTION6":
        tx, ty = target
        row = player_top + 6 * (ty - py) + 3
        col = 6 * tx + 3
        if 0 <= row < 63 and 0 <= col < 64:
            return action, int(row), int(col)

    return action, None, None

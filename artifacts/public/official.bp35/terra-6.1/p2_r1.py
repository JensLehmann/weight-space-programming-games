from collections import deque
import numpy as np


def _player(frame):
    ys, xs = np.where(frame[:63, :] == 9)
    if len(xs) == 0:
        return None
    return int(xs.min()) // 6, int(ys.min()) - 1


def _tile(frame, x, top):
    x0 = x * 6
    if top < 0 or top + 6 > 63 or x0 < 0 or x0 >= 64:
        return None
    a = frame[top + 1:top + 6, x0 + 1:min(x0 + 6, 64)]
    if a.size < 10:
        return None

    c = np.bincount(a.ravel().astype(np.int32), minlength=16)
    if c[14] >= 5:
        return "R"
    if c[8] >= 5:
        return "G"
    if c[12] >= 10:
        return "P"
    if c[12] >= 3:
        return "D"
    if c[15] >= 15:
        return "Y"
    if c[15] >= 4 and c[11] >= 1:
        return "S"
    if c[7] >= 3:
        return "T"
    if c[5] >= max(10, int(a.size * 0.55)):
        return "W"
    if c[10] >= max(5, int(a.size * 0.3)):
        return "."
    return None


def _estimate_row(old, new, old_player, new_player, old_y,
                  direction, can_fall, click_screen):
    ox, otop = old_player
    nx, ntop = new_player
    if not can_fall:
        return old_y

    best_d = 0
    best_score = -1e30

    for amount in range(13):
        d = direction * amount
        shift = otop - ntop + 6 * d
        lo = max(0, shift)
        hi = min(63, 63 + shift)
        if hi - lo < 10:
            continue

        aa = old[lo:hi, :]
        bb = new[lo - shift:hi - shift, :]
        mask = np.ones(aa.shape, dtype=bool)
        rows = np.arange(lo, hi)[:, None]
        cols = np.arange(64)[None, :]

        mask &= ~(
            (rows >= otop - 1) & (rows <= otop + 7) &
            (cols >= ox * 6 - 1) & (cols <= ox * 6 + 7)
        )
        new_rows = rows - shift
        mask &= ~(
            (new_rows >= ntop - 1) & (new_rows <= ntop + 7) &
            (cols >= nx * 6 - 1) & (cols <= nx * 6 + 7)
        )

        if click_screen is not None:
            cy, cx = click_screen
            mask &= ~(
                (rows >= cy - 9) & (rows <= cy + 9) &
                (cols >= cx - 9) & (cols <= cx + 9)
            )
        if not np.any(mask):
            continue

        rare_a = np.isin(aa, (7, 8, 12, 14, 15))
        rare_b = np.isin(bb, (7, 8, 12, 14, 15))
        weight = 1.0 + 2.0 * (rare_a | rare_b)
        value = np.where(aa == bb, 1.0, -2.0) * weight
        score = float(value[mask].sum() / weight[mask].sum())
        score += 0.04 * (hi - lo) / 63.0
        score -= 0.002 * amount

        if score > best_score:
            best_score = score
            best_d = d
    return old_y + best_d


def _read_board(frame, px, top, py, board):
    for r in range(-12, 13):
        screen_top = top + 6 * r
        if screen_top < 0 or screen_top + 6 > 63:
            continue
        for x in range(11):
            if x == px and r == 0:
                board.setdefault((x, py), ".")
                continue
            kind = _tile(frame, x, screen_top)
            if kind is not None:
                board[(x, py + r)] = kind


def _visible(x, y, px, py, gravity):
    top = 36 if gravity == -1 else 26
    row = top + 6 * (y - py) + 3
    return 0 <= row <= 62 and 0 <= x * 6 + 3 <= 63


def _action_key(action):
    name, target = action
    if target is None:
        return name
    return "%s:%d,%d" % (name, target[0], target[1])


def _safe_action(frame, px, top, py, gravity, action):
    """Reject actions with a visible spike in their immediate fall path."""
    name, target = action

    def tile_at(x, dy):
        return _tile(frame, x, top + 6 * dy)

    def scan(x, start_dy, direction):
        dy = start_dy
        for _ in range(20):
            t = tile_at(x, dy)
            if t is None:
                return True
            if t == "S":
                return False
            if t == "T":
                return True
            if t not in (".", "D"):
                return True
            dy += direction
        return True

    if name in ("ACTION3", "ACTION4"):
        dx = -1 if name == "ACTION3" else 1
        tx = px + dx
        if not 0 <= tx <= 10:
            return False
        t = tile_at(tx, 0)
        if t == "T":
            return True
        if t is not None and t not in (".", "D"):
            return False
        return scan(tx, gravity, gravity)

    if name != "ACTION6" or target is None:
        return True

    tx, ty = target
    dy = ty - py
    t = tile_at(tx, dy)
    if t not in ("R", "G", "P", "D", "Y"):
        return False

    if t == "G":
        ng = -gravity
        ahead = tile_at(px, ng)
        if ahead == "S":
            return False
        if ahead in ("W", "R", "G", "P", "Y", "T"):
            return True
        return scan(px, ng, ng)

    if tx == px and dy == gravity:
        next_tile = tile_at(px, 2 * gravity)
        if next_tile == "S":
            return False
        if t == "Y" and next_tile == ".":
            # The expanding block creates a new block here.
            return True
        return scan(px, 2 * gravity, gravity)

    return True


def _solve(board, px, py, gravity, visits, extended=False,
           frontier_visits=None, banned=None):
    if frontier_visits is None:
        frontier_visits = {}
    if banned is None:
        banned = set()

    goals = [p for p, t in board.items() if t == "T"]
    goal = (
        min(goals, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        if goals else None
    )

    def lookup(x, y, changes):
        return changes.get((x, y), board.get((x, y)))

    def pack(changes):
        return tuple(sorted((x, y, t) for (x, y), t in changes.items()))

    def first_penalty(first):
        if first is None:
            return 0
        key = "%d,%d,%d,%s,%s" % (
            px, py, gravity, first[0], first[1]
        )
        limit = 16 if extended else 6
        return min(limit, visits.get(key, 0) * 2)

    def frontier_score(first, depth, x, y, g):
        score = float(depth + first_penalty(first))
        if goal is None:
            if y > py:
                score += 5
        else:
            score += 0.45 * abs(x - goal[0]) + 0.2 * abs(y - goal[1])
        if extended:
            key = "%d,%d,%d" % (x, y, g)
            score += min(30, 7 * frontier_visits.get(key, 0))
        return score

    def fall(x, y, g, changes):
        for _ in range(80):
            ny = y + g
            t = lookup(x, ny, changes)
            if t is None:
                return "frontier", x, ny
            if t in (".", "D"):
                y = ny
                continue
            if t == "T":
                return "win", x, ny
            if t == "S":
                return "dead", x, y
            return "land", x, y
        return "frontier", x, y + g

    start = (px, py, gravity, ())
    queue = deque([(start, 0, None, ())])
    seen = {start}
    best_frontier = None
    best_frontier_path = None
    best_frontier_key = None
    best_frontier_score = float("inf")
    best_progress = None
    best_progress_path = None
    best_progress_score = 0.0

    nodes = 0
    max_nodes = 14000 if extended else 9000
    max_depth = 28 if extended else 22

    while queue and nodes < max_nodes:
        (x, y, g, packed), depth, first, path = queue.popleft()
        nodes += 1
        if depth >= max_depth:
            continue
        if best_frontier is not None:
            slack = 2 if extended else 3
            if depth > best_frontier_score + slack:
                continue

        changes = {(a, b): t for a, b, t in packed}

        def consider(action, status, xx, yy, gg, new_changes):
            nonlocal best_frontier, best_frontier_path
            nonlocal best_frontier_key, best_frontier_score
            nonlocal best_progress, best_progress_path, best_progress_score

            if depth == 0 and _action_key(action) in banned:
                return None

            initial = first if first is not None else action
            nd = depth + 1
            if extended:
                new_path = path + ((x, y, g, action[0], action[1]),)
            else:
                new_path = ()

            if status == "win":
                if extended:
                    return list(new_path), None
                return initial
            if status == "dead":
                return None
            if status == "frontier":
                s = frontier_score(initial, nd, xx, yy, gg)
                if s < best_frontier_score:
                    best_frontier_score = s
                    best_frontier = initial
                    best_frontier_path = new_path
                    best_frontier_key = "%d,%d,%d" % (xx, yy, gg)
                return None

            new_packed = pack(new_changes)
            state = (xx, yy, gg, new_packed)
            if state not in seen:
                seen.add(state)
                queue.append((state, nd, initial, new_path))

            if goal is not None:
                old_dist = abs(px - goal[0]) + abs(py - goal[1])
                new_dist = abs(xx - goal[0]) + abs(yy - goal[1])
                progress = old_dist - new_dist
            else:
                progress = py - yy

            quality = 3.0 * progress - nd - first_penalty(initial)
            if quality > best_progress_score:
                best_progress_score = quality
                best_progress = initial
                best_progress_path = new_path
            return None

        for dx, name in ((-1, "ACTION3"), (1, "ACTION4")):
            tx = x + dx
            if not 0 <= tx <= 10:
                continue
            t = lookup(tx, y, changes)
            action = (name, None)

            if t is None:
                result = consider(action, "frontier", tx, y, g, changes)
            elif t == "T":
                result = consider(action, "win", tx, y, g, changes)
            elif t in (".", "D"):
                status, xx, yy = fall(tx, y, g, changes)
                result = consider(action, status, xx, yy, g, changes)
            else:
                continue
            if result is not None:
                return result

        candidates = set()
        for dx in (-1, 0, 1):
            for dy in (-2, -1, 0, 1, 2):
                if dx != 0 or dy != 0:
                    candidates.add((x + dx, y + dy))

        for pos, t in board.items():
            current = changes.get(pos, t)
            if current == "G":
                candidates.add(pos)
            elif extended and current == "Y":
                candidates.add(pos)
            elif (
                extended and current in ("P", "D", "R") and
                abs(pos[0] - x) <= 4 and abs(pos[1] - y) <= 5
            ):
                candidates.add(pos)

        if extended:
            for pos, t in changes.items():
                if t in ("G", "Y"):
                    candidates.add(pos)

        ordered = sorted(
            candidates,
            key=lambda p: (abs(p[0] - x) + abs(p[1] - y), p[1], p[0])
        )
        for tx, ty in ordered:
            if not _visible(tx, ty, x, y, g):
                continue
            t = lookup(tx, ty, changes)
            if t not in ("R", "G", "P", "D", "Y"):
                continue

            action = ("ACTION6", (tx, ty))
            nc = changes.copy()
            ng = g

            if t == "R":
                nc[(tx, ty)] = "."
            elif t == "P":
                nc[(tx, ty)] = "D"
            elif t == "D":
                nc[(tx, ty)] = "P"
            elif t == "G":
                nc[(tx, ty)] = "."
                ng = -g
            else:
                nc[(tx, ty)] = "."
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    q = (tx + dx, ty + dy)
                    if q != (x, y) and lookup(q[0], q[1], nc) == ".":
                        nc[q] = "Y"

            status, xx, yy = "land", x, y
            if t == "G":
                ahead = lookup(x, y + ng, nc)
                if ahead is None:
                    status, xx, yy = "frontier", x, y + ng
                elif ahead == "T":
                    status, xx, yy = "win", x, y + ng
                elif ahead in (".", "D"):
                    status, xx, yy = fall(x, y, ng, nc)
                elif ahead == "S":
                    status = "dead"
            elif tx == x and ty == y + g:
                status, xx, yy = fall(tx, ty, g, nc)

            result = consider(action, status, xx, yy, ng, nc)
            if result is not None:
                return result

    if extended:
        if best_frontier_path:
            return list(best_frontier_path), best_frontier_key
        if best_progress_path:
            return list(best_progress_path), None
        return None
    return best_frontier or best_progress


def act(obs, available, memory):
    frame = np.asarray(obs, dtype=np.int8)
    player = _player(frame)
    if player is None:
        if "RESET" in available:
            return "RESET", None, None
        return available[0], None, None

    px, top = player
    board = {}
    for key, value in memory.get("board", {}).items():
        try:
            x, y = key.split(",")
            board[(int(x), int(y))] = value
        except (ValueError, AttributeError):
            pass

    previous = memory.get("previous")
    if previous:
        old = np.asarray(previous["frame"], dtype=np.int8)
        old_player = (previous["px"], previous["top"])
        py = _estimate_row(
            old, frame, old_player, player, previous["py"],
            previous["direction"], previous["can_fall"],
            previous.get("click_screen")
        )
        gravity = memory.get("gravity", -1)
    else:
        py = 0
        gravity = 1 if abs(top - 26) < abs(top - 36) else -1

    _read_board(frame, px, top, py, board)
    visits = memory.setdefault("visits", {})
    history = memory.setdefault("history", [])
    history.append([px, py, gravity])
    if len(history) > 12:
        del history[:-12]

    if not memory.get("extended", False) and len(history) >= 8:
        if len({tuple(s) for s in history[-8:]}) <= 2:
            memory["extended"] = True
            memory["plan"] = []

    choice = None
    plan = memory.get("plan", [])
    if plan:
        entry = plan[0]
        if (int(entry[0]), int(entry[1]), int(entry[2])) == (px, py, gravity):
            target = tuple(entry[4]) if entry[4] is not None else None
            candidate = (entry[3], target)
            if _safe_action(frame, px, top, py, gravity, candidate):
                choice = candidate
                memory["plan"] = plan[1:]
            else:
                memory["plan"] = []
        else:
            memory["plan"] = []

    banned = set()
    frontier_visits = memory.setdefault("frontiers", {})
    extended = memory.get("extended", False)

    for _ in range(8):
        if choice is not None:
            break

        if extended:
            result = _solve(
                board, px, py, gravity, visits, True,
                frontier_visits, banned
            )
            if result is not None:
                new_plan, frontier_key = result
                if new_plan:
                    entry = new_plan[0]
                    target = tuple(entry[4]) if entry[4] is not None else None
                    candidate = (entry[3], target)
                    if _safe_action(frame, px, top, py, gravity, candidate):
                        choice = candidate
                        memory["plan"] = [list(e) for e in new_plan[1:]]
                        if frontier_key is not None:
                            frontier_visits[frontier_key] = (
                                frontier_visits.get(frontier_key, 0) + 1
                            )
                        break
                    banned.add(_action_key(candidate))
                    continue
        else:
            candidate = _solve(
                board, px, py, gravity, visits, False,
                frontier_visits, banned
            )
            if candidate is not None:
                if _safe_action(frame, px, top, py, gravity, candidate):
                    choice = candidate
                    break
                banned.add(_action_key(candidate))
                continue

            extended = True
            memory["extended"] = True
            continue
        break

    if choice is None:
        candidates = []
        for dy in (gravity, -gravity, 0):
            for dx in (0, -1, 1):
                pos = (px + dx, py + dy)
                if (
                    board.get(pos) in ("R", "P", "Y", "G")
                    and _visible(pos[0], pos[1], px, py, gravity)
                ):
                    candidates.append(("ACTION6", pos))

        if extended:
            for pos, kind in sorted(board.items()):
                if kind in ("G", "Y", "P", "D", "R") and _visible(
                    pos[0], pos[1], px, py, gravity
                ):
                    candidates.append(("ACTION6", pos))

        candidates.extend([("ACTION3", None), ("ACTION4", None)])
        candidates.sort(
            key=lambda a: visits.get(
                "%d,%d,%d,%s,%s" % (px, py, gravity, a[0], a[1]), 0
            )
        )
        for candidate in candidates:
            if _action_key(candidate) in banned:
                continue
            if _safe_action(frame, px, top, py, gravity, candidate):
                choice = candidate
                break

    if choice is None:
        choice = ("RESET", None)

    name, target = choice
    if name not in available:
        if "RESET" in available:
            name, target = "RESET", None
        else:
            name, target = available[0], None

    visit_key = "%d,%d,%d,%s,%s" % (px, py, gravity, name, target)
    visits[visit_key] = visits.get(visit_key, 0) + 1

    row = None
    col = None
    can_fall = name in ("ACTION3", "ACTION4")
    click_screen = None
    direction = gravity

    if name == "ACTION6" and target is not None:
        tx, ty = target
        col = max(0, min(63, int(tx * 6 + 3)))
        row = max(0, min(62, int(top + 6 * (ty - py) + 3)))
        click_screen = (row, col)
        kind = board.get((tx, ty))

        if kind == "G":
            board[(tx, ty)] = "."
            gravity = -gravity
            direction = gravity
            can_fall = True
        elif kind in ("R", "P", "D", "Y"):
            can_fall = tx == px and ty == py + gravity
            if kind == "R":
                board[(tx, ty)] = "."
            elif kind == "P":
                board[(tx, ty)] = "D"
            elif kind == "D":
                board[(tx, ty)] = "P"
            else:
                board[(tx, ty)] = "."
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    q = (tx + dx, ty + dy)
                    if q != (px, py) and board.get(q) == ".":
                        board[q] = "Y"

    if name == "RESET":
        memory.clear()
        return "RESET", None, None

    memory["gravity"] = gravity
    memory["board"] = {"%d,%d" % p: t for p, t in board.items()}
    memory["previous"] = {
        "frame": frame.tolist(),
        "px": px,
        "top": top,
        "py": py,
        "direction": direction,
        "can_fall": can_fall,
        "click_screen": click_screen,
    }
    return name, row, col

import heapq
import time

import numpy as np


DIRS = (
    (0, -1, "ACTION1"),
    (0, 1, "ACTION2"),
    (-1, 0, "ACTION3"),
    (1, 0, "ACTION4"),
)

WHITE = np.ones((6, 6), dtype=np.int8)
WHITE[0, :] = WHITE[5, :] = 0
WHITE[:, 0] = WHITE[:, 5] = 0
TRAIN = np.where(WHITE == 0, 11, 12)


def make_state(pieces, trains, pins):
    return (
        tuple(sorted((x, y, c) for (x, y), c in pieces.items())),
        tuple(sorted(trains)),
        tuple(sorted(pins)),
    )


def find_origin(frame):
    best_score = -1
    best = (0, 0)
    for oy in range(6):
        for ox in range(6):
            score = 0
            for y in range(oy, 59, 6):
                if y == 0:
                    continue
                for x in range(ox, 59, 6):
                    patch = frame[y:y + 6, x:x + 6]
                    a = int(np.count_nonzero(patch == WHITE))
                    b = int(np.count_nonzero(patch == TRAIN))
                    if max(a, b) >= 15:
                        score += max(a, b)
            if score > best_score:
                best_score = score
                best = (ox, oy)
    return best


def shift_score(old, new, dx, dy):
    x0, x1 = max(0, -dx), min(64, 64 - dx)
    y0, y1 = max(1, 1 - dy), min(64, 64 - dy)
    if x1 <= x0 or y1 <= y0:
        return -1.0
    a = old[y0:y1, x0:x1]
    b = new[y0 + dy:y1 + dy, x0 + dx:x1 + dx]

    # Ignore moving green pieces and orange vehicles. Retain other
    # distinctive colors as well as the static board and rail artwork.
    va = ((a == 0) | (a == 1) | (a == 5) | (a == 7) |
          (a == 8) | (a == 9) | (a == 10) | (a == 15))
    vb = ((b == 0) | (b == 1) | (b == 5) | (b == 7) |
          (b == 8) | (b == 9) | (b == 10) | (b == 15))
    valid = va & vb
    denom = (
        int(np.count_nonzero(valid & (a != 10))) +
        int(np.count_nonzero(valid & (b != 10)))
    )
    if denom < 30:
        return -1.0
    same = int(np.count_nonzero(valid & (a == b) & (a != 10)))
    overlap = (x1 - x0) * (y1 - y0)
    return 2.0 * same / denom + 0.006 * overlap / 4096.0


def screen_shift(old, new, may_scroll):
    if not may_scroll:
        return 0, 0
    zero = shift_score(old, new, 0, 0)
    if zero > 0.998:
        return 0, 0
    best = (zero, 0, 0)
    for dx in range(-52, 53):
        if dx:
            score = shift_score(old, new, dx, 0) - 0.00004 * abs(dx)
            if score > best[0]:
                best = (score, dx, 0)
    for dy in range(-40, 41):
        if dy:
            score = shift_score(old, new, 0, dy) - 0.00004 * abs(dy)
            if score > best[0]:
                best = (score, 0, dy)
    if best[0] < zero + 0.012:
        return 0, 0
    return best[1], best[2]


def scan(frame, memory):
    if "origin" not in memory:
        memory["origin"] = find_origin(frame)
        memory["board"] = set()
        memory["rails"] = set()
        memory["seen"] = set()
        memory["state"] = ((), (), ())
    elif "previous_frame" in memory:
        old = np.asarray(memory["previous_frame"], dtype=np.int8)
        dx, dy = screen_shift(old, frame, memory.pop("may_scroll", False))
        ox, oy = memory["origin"]
        memory["origin"] = (ox + dx, oy + dy)
        if dx or dy:
            memory["scrolled"] = True

    ox, oy = memory["origin"]
    board = memory["board"]
    rails = memory["rails"]
    seen = memory["seen"]
    old_pieces, old_trains, old_pins = memory["state"]
    pieces = {(x, y): c for x, y, c in old_pieces}
    trains = set(old_trains)
    pins = set(old_pins)

    gx0 = (5 - ox) // 6
    gx1 = (58 - ox) // 6
    gy0 = (6 - oy) // 6
    gy1 = (58 - oy) // 6

    for gy in range(gy0, gy1 + 1):
        sy = oy + 6 * gy
        for gx in range(gx0, gx1 + 1):
            sx = ox + 6 * gx
            pos = (gx, gy)
            patch = frame[sy:sy + 6, sx:sx + 6]
            sw = int(np.count_nonzero(patch == WHITE))
            st = int(np.count_nonzero(patch == TRAIN))
            center = patch[1:5, 1:5]
            count, color = max(
                (int(np.count_nonzero(center == c)), c)
                for c in (14, 8, 9)
            )
            selected = count >= 5 and np.count_nonzero(patch == 3) >= 8
            is_white = sw >= 15 and sw >= st
            is_train = st >= 15 and st > sw
            if selected and not is_train:
                is_white = True

            seen.add(pos)
            pieces.pop(pos, None)
            trains.discard(pos)
            pins.discard(pos)

            if is_white:
                board.add(pos)
            elif not is_train:
                board.discard(pos)

            if is_train:
                trains.add(pos)
                rails.add(pos)
            elif not is_white:
                if np.count_nonzero(patch[2:4, 2:4] == 5) >= 2:
                    rails.add(pos)
                else:
                    rails.discard(pos)

            if is_white or is_train:
                if count >= 5:
                    pieces[pos] = color
                elif (np.count_nonzero(patch == 15) >= 4 and
                      np.count_nonzero(patch == 7) >= 1):
                    pins.add(pos)

    # Allow a train to explore one cell past an observed rail endpoint.
    for gy in range(gy0, gy1 + 1):
        sy = oy + 6 * gy
        if 0 <= sy + 2 < 64:
            if frame[sy + 2, 63] == 5:
                rails.add(((63 - ox) // 6 + 1, gy))
            if frame[sy + 2, 0] == 5:
                rails.add(((0 - ox) // 6 - 1, gy))
    for gx in range(gx0, gx1 + 1):
        sx = ox + 6 * gx
        if 0 <= sx + 2 < 64:
            if frame[63, sx + 2] == 5:
                rails.add((gx, (63 - oy) // 6 + 1))
            if frame[1, sx + 2] == 5:
                rails.add((gx, (1 - oy) // 6 - 1))

    state = make_state(pieces, trains, pins)
    memory["state"] = state
    memory["previous_frame"] = frame.copy()

    if any(c == 9 for c in pieces.values()):
        memory["has_blue"] = True
        memory["complex"] = True
    if pins or any(c == 8 for c in pieces.values()):
        memory["complex"] = True
    if any(c == 8 for c in pieces.values()):
        memory["has_red"] = True

    active_count = sum(
        1 for c in pieces.values()
        if not (memory.get("has_blue", False) and c == 9)
    )
    memory["max_active"] = max(memory.get("max_active", 0), active_count)

    intro = {(x, y) for y in (1, 2, 3) for x in range(1, 8)}
    intro |= {(x, y) for y in (4, 5, 6, 7) for x in range(5, 8)}
    if board == intro:
        memory.setdefault("danger", set()).update({(1, 2), (3, 2), (6, 1)})

    return state


def moves(state, board, rails, danger):
    piece_list, train_list, pin_list = state
    pieces = {(x, y): c for x, y, c in piece_list}
    trains = set(train_list)
    pins = set(pin_list)
    floors = board | trains

    for (x, y), color in pieces.items():
        for dx, dy, _ in DIRS:
            mid = (x + dx, y + dy)
            dst = (x + 2 * dx, y + 2 * dy)
            if mid not in pieces and mid not in pins:
                continue
            if dst not in floors or dst in pieces or dst in pins or dst in danger:
                continue
            new_pieces = dict(pieces)
            del new_pieces[(x, y)]
            if (mid in new_pieces and new_pieces[mid] == color
                    and color != 9):
                del new_pieces[mid]
            new_pieces[dst] = color
            yield (
                ("jump", x, y, dst[0], dst[1]),
                make_state(new_pieces, trains, pins),
                2,
            )

    for dx, dy, name in DIRS:
        new_trains = set(trains)
        new_pieces = dict(pieces)
        new_pins = set(pins)
        moved = False
        valid = True
        ordered = sorted(
            trains, key=lambda p: p[0] * dx + p[1] * dy, reverse=True
        )
        for pos in ordered:
            dst = (pos[0] + dx, pos[1] + dy)
            if dst in new_trains or dst not in rails:
                continue
            if pos in new_pieces and (dst in new_pieces or dst in new_pins):
                valid = False
                break
            if pos in new_pins and (dst in new_pins or dst in new_pieces):
                valid = False
                break
            new_trains.remove(pos)
            new_trains.add(dst)
            if pos in new_pieces:
                new_pieces[dst] = new_pieces.pop(pos)
            if pos in new_pins:
                new_pins.remove(pos)
                new_pins.add(dst)
            moved = True
        if moved and valid:
            yield (
                ("rail", name),
                make_state(new_pieces, new_trains, new_pins),
                1,
            )


def next_state(state, action, memory):
    for candidate, nxt, _ in moves(
        state, memory["board"], memory["rails"], memory.get("danger", set())
    ):
        if candidate == action:
            return nxt
    return None


def visible(pos, origin):
    sx = origin[0] + 6 * pos[0]
    sy = origin[1] + 6 * pos[1]
    return 0 <= sx <= 58 and 1 <= sy <= 58


def count_pieces(state, has_blue):
    return sum(1 for _, _, c in state[0] if not (has_blue and c == 9))


def frontiers(board, rails, seen):
    return [
        (x, y) for x, y in board | rails
        if any((x + dx, y + dy) not in seen for dx, dy, _ in DIRS)
    ]


def frontier_distance(state, frontier, has_blue):
    if not frontier:
        return 0
    return sum(
        min(abs(x - a) + abs(y - b) for a, b in frontier)
        for x, y, c in state[0]
        if not (has_blue and c == 9)
    )


def progress(state, has_blue):
    # Blue pieces are reusable obstacles; prioritize moving green pieces.
    xs = [x for x, _, c in state[0] if c == 14]
    if not xs:
        xs = [x for x, _, c in state[0] if not (has_blue and c == 9)]
    if not xs:
        return 0, 0, 0
    return min(xs), sum(xs), max(xs)


def plan(memory, deadline):
    start = memory["state"]
    board, rails = memory["board"], memory["rails"]
    danger = memory.get("danger", set())
    origin = memory["origin"]
    has_blue = memory.get("has_blue", False)
    complex_board = memory.get("complex", False)
    bans = memory.setdefault("bans", {})
    visits = memory.setdefault("visits", {})
    frontier = frontiers(board, rails, memory["seen"])
    initial_count = count_pieces(start, has_blue)
    red_goal = (
        memory.get("has_red", False) and
        (memory.get("max_active", 0) >= 3 or memory.get("scrolled", False))
    )

    serial = 0
    heap = [(0.0, 0, serial, start)]
    costs = {start: 0}
    parent = {}
    best = None
    best_quality = None
    expanded = 0

    while heap and expanded < 22000 and time.monotonic() < deadline:
        _, cost, _, state = heapq.heappop(heap)
        if costs.get(state) != cost:
            continue
        expanded += 1
        n = count_pieces(state, has_blue)

        if state != start:
            if complex_board:
                least, total, farthest = progress(state, has_blue)
                quality = (
                    n, -least, -total, -farthest,
                    visits.get(state, 0), cost
                )
            else:
                quality = (
                    n, visits.get(state, 0),
                    frontier_distance(state, frontier, has_blue), cost
                )
            if best_quality is None or quality < best_quality:
                best_quality = quality
                best = state

            previous, last_action = parent[state]
            if last_action[0] == "jump":
                if n <= 1 and initial_count > 1:
                    best = state
                    break
                if red_goal and n == 2:
                    best = state
                    break

        forbidden = bans.get(state, set())
        for action, nxt, step_cost in moves(state, board, rails, danger):
            if action in forbidden:
                continue
            if state == start and action[0] == "jump":
                if not (visible((action[1], action[2]), origin) and
                        visible((action[3], action[4]), origin)):
                    continue

            new_cost = cost + step_cost
            if new_cost >= costs.get(nxt, 10**9):
                continue
            costs[nxt] = new_cost
            parent[nxt] = (state, action)
            serial += 1
            n2 = count_pieces(nxt, has_blue)

            if complex_board:
                least, total, farthest = progress(nxt, has_blue)
                priority = (
                    20 * n2 + new_cost
                    - 0.7 * least - 0.12 * total - 0.08 * farthest
                )
            else:
                distance = frontier_distance(nxt, frontier, has_blue)
                priority = 16 * n2 + new_cost + 0.12 * distance
            heapq.heappush(heap, (priority, new_cost, serial, nxt))

    if best is None:
        return []
    path = []
    while best != start:
        previous, action = parent[best]
        path.append(action)
        best = previous
    path.reverse()
    return path


def restart_prompt(frame):
    return np.count_nonzero(frame[49:64, 2:15] == 15) >= 36


def begin_undo(memory, available):
    history = memory.setdefault("history", [])
    if "ACTION7" not in available or not history:
        return None
    before, action = history.pop()
    memory.setdefault("bans", {}).setdefault(before, set()).add(action)
    memory["plan"] = []
    memory.pop("pending", None)
    memory.pop("last_transition", None)
    memory["undo_remaining"] = 1 if action[0] == "jump" else 0
    memory["may_scroll"] = True
    return ("ACTION7", None, None)


def act(obs, available, memory):
    frame = np.asarray(obs, dtype=np.int8)
    deadline = time.monotonic() + 1.5
    memory.setdefault("history", [])
    memory.setdefault("visits", {})
    memory.setdefault("plan", [])

    # The selection outline obscures the piece. Use the saved state
    # until the destination click has been issued.
    pending = memory.get("pending")
    if pending is not None:
        action, before, predicted = pending
        ox, oy = memory["origin"]
        sx = ox + 6 * action[3]
        sy = oy + 6 * action[4]
        memory.pop("pending", None)

        if 0 <= sx <= 58 and 1 <= sy <= 58 and "ACTION6" in available:
            memory["history"].append((before, action))
            memory["state"] = predicted
            memory["visits"][predicted] = memory["visits"].get(predicted, 0) + 1
            memory["last_transition"] = (before, action, predicted)
            memory["may_scroll"] = True
            if memory["plan"] and memory["plan"][0] == action:
                memory["plan"].pop(0)
            return ("ACTION6", sy + 2, sx + 2)

        memory["plan"] = []
        memory.setdefault("bans", {}).setdefault(before, set()).add(action)

    state = scan(frame, memory)

    if memory.get("undo_remaining", 0) > 0:
        memory["undo_remaining"] -= 1
        memory["may_scroll"] = True
        if "ACTION7" in available:
            return ("ACTION7", None, None)

    if restart_prompt(frame):
        result = begin_undo(memory, available)
        if result is not None:
            return result
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)

    transition = memory.pop("last_transition", None)
    if transition is not None:
        before, action, predicted = transition
        if state == before:
            memory.setdefault("bans", {}).setdefault(before, set()).add(action)
            memory["plan"] = []
        elif state != predicted:
            memory["plan"] = []

    path = memory["plan"]
    if path:
        action = path[0]
        if next_state(state, action, memory) is None:
            path = []
        elif action[0] == "jump":
            if not (visible((action[1], action[2]), memory["origin"]) and
                    visible((action[3], action[4]), memory["origin"])):
                path = []

    if not path:
        path = plan(memory, deadline)
        memory["plan"] = path

    if not path:
        result = begin_undo(memory, available)
        if result is not None:
            return result
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        for name in available:
            if name != "ACTION6":
                return (name, None, None)
        return ("ACTION6", 32, 32)

    action = path[0]
    predicted = next_state(state, action, memory)

    if action[0] == "jump":
        if predicted is None or "ACTION6" not in available:
            memory["plan"] = []
            for name in available:
                if name != "ACTION6":
                    return (name, None, None)
            return ("ACTION6", 32, 32)
        ox, oy = memory["origin"]
        sx = ox + 6 * action[1]
        sy = oy + 6 * action[2]
        memory["pending"] = (action, state, predicted)
        return ("ACTION6", sy + 2, sx + 2)

    name = action[1]
    if predicted is None or name not in available:
        memory["plan"] = []
        for candidate in available:
            if candidate != "ACTION6":
                return (candidate, None, None)
        return ("ACTION6", 32, 32)

    carried_green = any(
        c == 14 and (x, y) in state[1]
        for x, y, c in state[0]
    )
    memory["may_scroll"] = carried_green
    memory["history"].append((state, action))
    memory["state"] = predicted
    memory["visits"][predicted] = memory["visits"].get(predicted, 0) + 1
    memory["last_transition"] = (state, action, predicted)
    memory["plan"].pop(0)
    return (name, None, None)

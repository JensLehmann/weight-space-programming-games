import numpy as np
import heapq
from collections import deque


# Pixels on the two diagonals do not participate in the win condition.
_CELLS = tuple(
    (r, c) for r in range(10) for c in range(10)
    if r != c and r + c != 9
)

_RING = (
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
)

# x, y, rotation, diagonal
_POSES = (
    (25, 24, 180, False),
    (33, 21, 0, True),
    (38, 32, 270, False),
    (33, 40, 90, True),
    (25, 45, 0, False),
    (14, 40, 180, True),
    (17, 32, 90, False),
    (14, 21, 270, True),
)

# x, y, rotation
_SMALL_POSES = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}

_OP_ORIENTATION = (0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6)


def _make_masks():
    masks = [0] * 12
    for k, (r, c) in enumerate(_CELLS):
        predicates = (
            r < 5,
            c >= r,
            c >= 5,
            c >= 9 - r,
            r >= 5,
            c <= r,
            c < 5,
            c <= 9 - r,
            r < 3 and 3 <= c < 7,
            3 <= r < 7 and c >= 7,
            r >= 7 and 3 <= c < 7,
            3 <= r < 7 and c < 3,
        )
        for i, inside in enumerate(predicates):
            if inside:
                masks[i] |= 1 << k
    return tuple(masks)


_MASKS = _make_masks()


def _navigation(available):
    lookup = {p: i for i, p in enumerate(_RING)}
    transitions = [[] for _ in range(8)]
    deltas = ((1, -1, 0), (2, 1, 0), (3, 0, -1), (4, 0, 1))

    for i, (r, c) in enumerate(_RING):
        for action, dr, dc in deltas:
            if "ACTION%d" % action not in available:
                continue
            p = (max(0, min(2, r + dr)), max(0, min(2, c + dc)))
            j = lookup.get(p, i)
            if j != i:
                transitions[i].append((j, action))

    distances = [[10000] * 8 for _ in range(8)]
    first = [[None] * 8 for _ in range(8)]
    for start in range(8):
        distances[start][start] = 0
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j, action in transitions[i]:
                if distances[start][j] == 10000:
                    distances[start][j] = distances[start][i] + 1
                    first[start][j] = (
                        action if i == start else first[start][i]
                    )
                    queue.append(j)
    return distances, first, transitions


def _tool_templates():
    horizontal = np.full((9, 14), -1, dtype=np.int8)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[1:8, 1:13] = 15
    horizontal[8, :] = 2

    diagonal = np.full((17, 17), -1, dtype=np.int8)
    red_rows = (
        (6,), (5, 6, 7), (4, 5, 7, 8), (3, 4, 8, 9),
        (2, 3, 9, 10), (1, 2, 10, 11), (0, 1, 11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16),
        (14, 15), (13, 14), (12, 13), (11, 12),
        (10, 11), (10,),
    )
    fill_rows = {
        2: (6, 7), 3: (5, 8), 4: (4, 9), 5: (3, 10),
        6: (2, 11), 7: (2, 12), 8: (3, 13), 9: (4, 14),
        10: (5, 15), 11: (6, 14), 12: (7, 13),
        13: (8, 12), 14: (9, 11),
    }
    for r, (a, b) in fill_rows.items():
        diagonal[r, a:b] = 15
    for r, columns in enumerate(red_rows):
        diagonal[r, list(columns)] = 2

    small = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=np.int8)
    return horizontal, diagonal, small


def _read_palette(obs):
    palette = {}
    selected = None
    for r in range(60):
        for c in range(60):
            if (obs[r, c] != 4 or obs[r, c + 4] != 4
                    or obs[r + 4, c] != 4 or obs[r + 4, c + 4] != 4):
                continue
            box = obs[r:r + 5, c:c + 5]
            color = int(box[2, 2])
            if color == 4:
                continue
            if not (np.all(box[0, :] == 4)
                    and np.all(box[4, :] == 4)
                    and np.all(box[:, 0] == 4)
                    and np.all(box[:, 4] == 4)
                    and np.all(box[1:4, 1:4] == color)):
                continue
            palette[color] = (r + 2, c + 2)
            if r + 5 < 64 and np.all(obs[r + 5, c:c + 5] == 0):
                selected = color

    if not palette:
        # Conservative fallback for the canonical two-color toolbar.
        palette = {0: (4, 37), 15: (4, 43)}
    return palette, selected


def _read_tools(obs, memory, selected):
    horizontal, diagonal, small = _tool_templates()
    best_score = -1.0
    orientation = int(memory.get("_orientation", 0))
    color = selected

    for i, (x, y, rotation, is_diagonal) in enumerate(_POSES):
        base = diagonal if is_diagonal else horizontal
        template = np.rot90(base, -(rotation // 90))
        h, w = template.shape
        region = obs[y:y + h, x:x + w]
        score = float(np.mean(region[template == 2] == 2))
        if score > best_score:
            best_score = score
            orientation = i
            values = region[template == 15]
            color = int(np.bincount(values.astype(int), minlength=16).argmax())

    if best_score < 0.55:
        orientation = int(memory.get("_orientation", 0))
        color = selected if selected is not None else memory.get("_color", 15)

    small_present = None
    small_click = None
    if orientation in _SMALL_POSES:
        x, y, rotation = _SMALL_POSES[orientation]
        template = np.rot90(small, -(rotation // 90))
        h, w = template.shape
        region = obs[y:y + h, x:x + w]
        small_present = bool(np.mean(region[template == 2] == 2) > 0.8)
        small_click = (y + h // 2, x + w // 2)

    return orientation, int(color), small_present, small_click


def _read_boards(obs, palette):
    membership = np.isin(obs, list(set(palette) | {0}))
    integral = np.pad(membership.astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    sums = (integral[10:, 10:] - integral[:-10, 10:]
            - integral[10:, :-10] + integral[:-10, :-10])
    candidates = [tuple(map(int, p)) for p in np.argwhere(sums == 100)]

    target_pos = (3, 3)
    canvas_pos = (34, 27)
    if candidates:
        target_pos = min(
            candidates, key=lambda p: (p[0] - 3) ** 2 + (p[1] - 3) ** 2
        )
        others = [p for p in candidates if p != target_pos]
        if others:
            canvas_pos = min(
                others, key=lambda p: (p[0] - 34) ** 2 + (p[1] - 27) ** 2
            )

    tr, tc = target_pos
    cr, cc = canvas_pos
    target = tuple(int(obs[tr + r, tc + c]) for r, c in _CELLS)
    board = tuple(int(obs[cr + r, cc + c]) for r, c in _CELLS)
    return target, board, target_pos, canvas_pos


def _plan(target, board, colors, start_orientation, start_color,
          distances, operation_ids):
    color_masks = {}
    wrong = 0
    for k, color in enumerate(target):
        color_masks[color] = color_masks.get(color, 0) | (1 << k)
        if board[k] != color:
            wrong |= 1 << k

    # Reverse state: finalized pixels, earliest stroke's position and color.
    root = (0, 8, -1)
    costs = {root: 0}
    parents = {}
    queue = [(0, root)]
    options_cache = {}
    best_cost = 10 ** 9
    best_end = None

    while queue:
        cost, state = heapq.heappop(queue)
        if cost != costs.get(state):
            continue
        if cost >= best_cost:
            break

        covered, previous_orientation, previous_color = state

        if previous_orientation != 8 and (wrong & ~covered) == 0:
            entry_distance = distances[start_orientation][previous_orientation]
            total = cost + entry_distance + (start_color != previous_color)
            if entry_distance < 10000 and total < best_cost:
                best_cost = total
                best_end = state
            continue

        options = options_cache.get(covered)
        if options is None:
            options = []
            for op in operation_ids:
                mask = _MASKS[op]
                exposed = mask & ~covered
                if not exposed:
                    continue
                bit = (exposed & -exposed).bit_length() - 1
                color = target[bit]
                if color not in colors:
                    continue
                if exposed & ~color_masks[color]:
                    continue
                options.append((op, color, covered | mask))
            options_cache[covered] = options

        for op, color, new_covered in options:
            orientation = _OP_ORIENTATION[op]
            edge = 1
            if previous_orientation != 8:
                distance = distances[orientation][previous_orientation]
                if distance >= 10000:
                    continue
                edge += distance + (color != previous_color)

            new_cost = cost + edge
            new_state = (new_covered, orientation, color)
            if new_cost >= best_cost or new_cost >= costs.get(new_state, 10 ** 9):
                continue
            costs[new_state] = new_cost
            parents[new_state] = (state, op, color)
            heapq.heappush(queue, (new_cost, new_state))

    if best_end is None:
        return None

    # Following reverse-search parents gives chronological stroke order.
    result = []
    state = best_end
    while state != root:
        previous, op, color = parents[state]
        result.append((op, color))
        state = previous
    return result


def _fallback(available, memory):
    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    for name in ("ACTION5", "ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if name in available:
            return name, None, None
    return "ACTION6", 4, 4


def act(obs, available, memory):
    available = set(available)
    obs = np.asarray(obs)

    palette, selected = _read_palette(obs)
    orientation, color, small_visible, small_click = _read_tools(
        obs, memory, selected
    )
    target, board, target_pos, canvas_pos = _read_boards(obs, palette)

    if memory.get("_target") != target:
        memory.pop("_small_enabled", None)
    memory["_target"] = target

    if small_visible is not None:
        memory["_small_enabled"] = small_visible

    # A diagonal position hides the small tool. Assume it exists until a
    # cardinal position lets us directly check; replan if it does not.
    small_enabled = memory.get("_small_enabled", True)
    memory["_orientation"] = orientation
    memory["_color"] = color

    distances, first_actions, transitions = _navigation(available)
    operation_ids = []
    if "ACTION5" in available:
        operation_ids.extend(range(8))
    if small_enabled and "ACTION6" in available:
        operation_ids.extend(range(8, 12))

    colors = set(palette) if "ACTION6" in available else {color}
    signature = (
        target, target_pos, canvas_pos,
        tuple(sorted(palette.items())),
        tuple(operation_ids), tuple(sorted(available)),
    )

    if (memory.get("_signature") != signature
            or memory.get("_expected") != board
            or not memory.get("_plan")):
        plan = _plan(
            target, board, colors, orientation, color,
            distances, operation_ids
        )
        if not plan:
            return _fallback(available, memory)
        memory["_plan"] = plan
        memory["_signature"] = signature
        memory["_expected"] = board

    op, desired_color = memory["_plan"][0]
    desired_orientation = _OP_ORIENTATION[op]

    # Move before changing color, so entering a cardinal position can reveal
    # the presence or absence of the small tool without an unnecessary click.
    if orientation != desired_orientation:
        action = first_actions[orientation][desired_orientation]
        if action is None:
            return _fallback(available, memory)
        for destination, candidate_action in transitions[orientation]:
            if candidate_action == action:
                memory["_orientation"] = destination
                break
        return "ACTION%d" % action, None, None

    if color != desired_color:
        if desired_color not in palette or "ACTION6" not in available:
            return _fallback(available, memory)
        row, col = palette[desired_color]
        memory["_color"] = desired_color
        return "ACTION6", int(row), int(col)

    if op >= 8 and not small_visible:
        memory.pop("_plan", None)
        return _fallback(available, memory)

    expected = list(board)
    mask = _MASKS[op]
    for k in range(len(_CELLS)):
        if mask & (1 << k):
            expected[k] = desired_color
    memory["_expected"] = tuple(expected)
    memory["_plan"].pop(0)

    if op < 8:
        return "ACTION5", None, None

    row, col = small_click
    return "ACTION6", int(row), int(col)

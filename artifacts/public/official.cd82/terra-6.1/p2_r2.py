import heapq
from functools import lru_cache

import numpy as np


# Basket positions, in clockwise order around the canvas.
_POS = (
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
)
_BASKET_LAYOUT = (
    (25, 24, 180),
    (33, 21, 0),
    (38, 32, 270),
    (33, 40, 90),
    (25, 45, 0),
    (14, 40, 180),
    (17, 32, 90),
    (14, 21, 270),
)
_STAMP_LAYOUT = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}


def _rotate(a, degrees):
    return np.rot90(a, (-degrees // 90) % 4)


def _make_templates():
    horizontal = np.full((9, 14), -1, dtype=int)
    horizontal[0, 0] = horizontal[0, 13] = 2
    horizontal[1:8, 0] = 2
    horizontal[1:8, 13] = 2
    horizontal[1:8, 1:13] = 15
    horizontal[8, :] = 2

    diagonal = np.full((17, 17), -1, dtype=int)
    diagonal[0, 6] = 2
    for r in range(1, 7):
        left, right = 6 - r, 6 + r
        diagonal[r, left:left + 2] = 2
        diagonal[r, right - 1:right + 1] = 2
        if left + 2 < right - 1:
            diagonal[r, left + 2:right - 1] = 15
    for r in range(7, 11):
        left, right = r - 5, r + 4
        diagonal[r, left:right + 1] = 15
        diagonal[r, right + 1:right + 3] = 2
    for r in range(11, 16):
        left, right = r - 5, 24 - r
        if left <= right:
            diagonal[r, left:right + 1] = 15
        diagonal[r, 25 - r:27 - r] = 2
    diagonal[16, 10] = 2

    stamp = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=int)

    baskets = []
    for i, (_, _, rotation) in enumerate(_BASKET_LAYOUT):
        base = horizontal if i % 2 == 0 else diagonal
        baskets.append(_rotate(base, rotation))
    stamps = {
        i: _rotate(stamp, spec[2])
        for i, spec in _STAMP_LAYOUT.items()
    }
    return baskets, stamps


_BASKETS, _STAMPS = _make_templates()


def _make_masks():
    masks = [0] * 12
    all_cells = 0
    for r in range(10):
        for c in range(10):
            if r == c or r + c == 9:
                continue
            bit = 1 << (10 * r + c)
            all_cells |= bit
            conditions = (
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
            for i, yes in enumerate(conditions):
                if yes:
                    masks[i] |= bit
    return tuple(masks), all_cells


_MASKS, _ALL_CELLS = _make_masks()


def _operation_position(op):
    return op if op < 8 else 2 * (op - 8)


def _distance(a, b):
    d = abs(a - b)
    return min(d, 8 - d)


def _find_basket(obs):
    best = None
    for i, template in enumerate(_BASKETS):
        x, y, _ = _BASKET_LAYOUT[i]
        h, w = template.shape
        patch = obs[y:y + h, x:x + w]
        if patch.shape != template.shape:
            continue

        red = template == 2
        fill = template == 15
        values = patch[fill].astype(int)
        if values.size:
            color = int(np.bincount(values, minlength=16).argmax())
        else:
            color = 15
        matches = int(np.count_nonzero(patch[red] == 2))
        matches += int(np.count_nonzero(values == color))
        total = int(np.count_nonzero(red) + values.size)
        score = matches / max(1, total)
        if best is None or score > best[0]:
            best = (score, i, color)

    if best is None:
        return 0, 15
    return best[1], best[2]


def _stamp_present(obs, position):
    x, y, _ = _STAMP_LAYOUT[position]
    template = _STAMPS[position]
    h, w = template.shape
    patch = obs[y:y + h, x:x + w]
    if patch.shape != template.shape:
        return False
    red = template == 2
    return bool(np.all(patch[red] == 2))


def _find_palette(obs):
    palette = {}
    # Palette squares have a 4-colored border and a uniform 3x3 center.
    for y in range(0, 5):
        for x in range(18, 60):
            a = obs[y:y + 5, x:x + 5]
            if a.shape != (5, 5):
                continue
            if not (
                np.all(a[0, :] == 4)
                and np.all(a[4, :] == 4)
                and np.all(a[:, 0] == 4)
                and np.all(a[:, 4] == 4)
            ):
                continue
            color = int(a[2, 2])
            if np.all(a[1:4, 1:4] == color):
                palette[color] = (y + 2, x + 2)
    return palette


def _plan(obs, palette, start_position, start_color, allow_stamp):
    target = obs[3:13, 3:13]
    canvas = obs[34:44, 27:37]

    color_bits = [0] * 16
    cell_colors = [0] * 100
    mismatch = 0
    for r in range(10):
        for c in range(10):
            if r == c or r + c == 9:
                continue
            idx = 10 * r + c
            bit = 1 << idx
            color = int(target[r, c])
            cell_colors[idx] = color
            color_bits[color] |= bit
            if int(canvas[r, c]) != color:
                mismatch |= bit

    allowed_colors = set(palette)
    allowed_ops = range(12 if allow_stamp else 8)

    @lru_cache(maxsize=None)
    def options(covered):
        result = []
        for op in allowed_ops:
            remaining = _MASKS[op] & ~covered
            if not remaining:
                continue
            first = (remaining & -remaining).bit_length() - 1
            color = cell_colors[first]
            if color not in allowed_colors:
                continue
            if remaining & ~color_bits[color]:
                continue
            new_covered = covered | _MASKS[op]
            result.append((op, color, new_covered))

        # Prefer operations that explain more currently incorrect pixels.
        result.sort(
            key=lambda item: (
                (item[2] & mismatch).bit_count(),
                (item[2] ^ covered).bit_count(),
            ),
            reverse=True,
        )
        return tuple(result)

    if mismatch == 0:
        # A paint action is still needed to trigger the game's win check.
        candidates = options(0)
        if not candidates:
            return []
        op, color, _ = min(
            candidates,
            key=lambda item: (
                _distance(start_position, _operation_position(item[0]))
                + (item[1] != start_color),
                item[0],
            ),
        )
        return [[op, color]]

    @lru_cache(maxsize=None)
    def feasible(covered):
        if not (mismatch & ~covered):
            return ()
        for op, color, new_covered in options(covered):
            rest = feasible(new_covered)
            if rest is not None:
                # This order is last operation to first operation.
                return ((op, color),) + rest
        return None

    reverse_solution = feasible(0)
    if reverse_solution is None:
        return None

    best_plan = list(reversed(reverse_solution))

    def plan_cost(plan):
        p, color = start_position, start_color
        cost = 0
        for op, next_color in plan:
            q = _operation_position(op)
            cost += _distance(p, q) + (color != next_color) + 1
            p, color = q, next_color
        return cost

    best_cost = plan_cost(best_plan)

    # Improve the feasible plan with a bounded shortest-path search.
    # Search is backwards: the first selected operation is the last paint.
    start = (0, -1, -1)
    distances = {start: 0}
    parents = {}
    queue = [(0, 0, start)]
    serial = 0
    expanded = 0

    while queue and expanded < 60000:
        cost, _, state = heapq.heappop(queue)
        if cost != distances.get(state):
            continue
        if cost >= best_cost:
            break
        expanded += 1
        covered, position, color = state

        if position != -1 and not (mismatch & ~covered):
            total = (
                cost
                + _distance(start_position, position)
                + (start_color != color)
            )
            if total < best_cost:
                candidate = []
                s = state
                while s != start:
                    previous, op, op_color = parents[s]
                    candidate.append((op, op_color))
                    s = previous
                best_plan = candidate
                best_cost = total
            continue

        for op, next_color, new_covered in options(covered):
            next_position = _operation_position(op)
            extra = 1
            if position != -1:
                extra += _distance(position, next_position)
                extra += color != next_color
            new_cost = cost + extra
            if new_cost >= best_cost:
                continue
            new_state = (new_covered, next_position, next_color)
            if new_cost < distances.get(new_state, 10**9):
                distances[new_state] = new_cost
                parents[new_state] = (state, op, next_color)
                serial += 1
                heapq.heappush(queue, (new_cost, serial, new_state))

    return [[int(op), int(color)] for op, color in best_plan]


def _move_toward(position, destination):
    clockwise = (destination - position) % 8
    counterclockwise = (position - destination) % 8
    if clockwise <= counterclockwise:
        next_position = (position + 1) % 8
    else:
        next_position = (position - 1) % 8

    r, c = _POS[position]
    nr, nc = _POS[next_position]
    if nr < r:
        return "ACTION1"
    if nr > r:
        return "ACTION2"
    if nc < c:
        return "ACTION3"
    return "ACTION4"


def act(obs, available, memory):
    obs = np.asarray(obs)
    position, selected_color = _find_basket(obs)
    palette = _find_palette(obs)

    # On cardinal positions the stamp, if present in this level, is visible.
    if position % 2 == 0:
        present = _stamp_present(obs, position)
        memory["cd82_stamp"] = present
        if not present:
            plan = memory.get("cd82_plan")
            index = memory.get("cd82_index", 0)
            if plan and any(op >= 8 for op, _ in plan[index:]):
                memory.pop("cd82_plan", None)
                memory.pop("cd82_index", None)

    plan = memory.get("cd82_plan")
    index = memory.get("cd82_index", 0)

    if not plan or index >= len(plan):
        # If the episode began at a diagonal, assume stamps are available;
        # this is checked as soon as a cardinal position is reached.
        allow_stamp = memory.get("cd82_stamp", True)
        plan = _plan(obs, palette, position, selected_color, allow_stamp)
        if plan is None:
            if "RESET" in available and not memory.get("cd82_reset"):
                memory["cd82_reset"] = True
                memory.pop("cd82_stamp", None)
                return "RESET", None, None
            return "ACTION5", None, None
        if not plan:
            return "ACTION5", None, None
        memory["cd82_plan"] = plan
        memory["cd82_index"] = 0
        index = 0

    op, color = plan[index]
    destination = _operation_position(op)

    if position != destination:
        return _move_toward(position, destination), None, None

    if selected_color != color:
        if color in palette:
            row, col = palette[color]
            return "ACTION6", int(row), int(col)
        memory.pop("cd82_plan", None)
        return "ACTION5", None, None

    memory["cd82_index"] = index + 1
    if op < 8:
        return "ACTION5", None, None

    x, y, _ = _STAMP_LAYOUT[position]
    template = _STAMPS[position]
    row = y + template.shape[0] // 2
    col = x + template.shape[1] // 2
    return "ACTION6", int(row), int(col)

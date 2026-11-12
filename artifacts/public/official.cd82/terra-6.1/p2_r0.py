import numpy as np
from functools import lru_cache

# Tool positions, in clockwise order.
_POS = [
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
]
_BASKETS = [
    ("h", 25, 24, 180),
    ("d", 33, 21, 0),
    ("h", 38, 32, 270),
    ("d", 33, 40, 90),
    ("h", 25, 45, 0),
    ("d", 14, 40, 180),
    ("h", 17, 32, 90),
    ("d", 14, 21, 270),
]
_NOZZLES = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}

# Only the outline colour is needed for identifying the active tool.
_H = np.full((9, 14), -1, dtype=int)
_H[0, 0] = _H[0, 13] = 2
_H[1:8, 0] = 2
_H[1:8, 13] = 2
_H[8, :] = 2

_D = np.full((17, 17), -1, dtype=int)
_DIAGONAL_OUTLINE = [
    [6], [5, 6, 7], [4, 5, 7, 8], [3, 4, 8, 9],
    [2, 3, 9, 10], [1, 2, 10, 11], [0, 1, 11, 12],
    [12, 13], [13, 14], [14, 15], [15, 16], [14, 15],
    [13, 14], [12, 13], [11, 12], [10, 11], [10],
]
for _r, _cols in enumerate(_DIAGONAL_OUTLINE):
    for _c in _cols:
        _D[_r, _c] = 2

_N = np.array([
    [2, 15, 15, 15, 15, 2],
    [2, 15, 15, 15, 15, 2],
    [2, 15, 15, 15, 15, 2],
    [2, 2, 0, 0, 2, 2],
    [-1, 2, 2, 2, 2, -1],
], dtype=int)


def _rot(a, degrees):
    return np.rot90(a, -(degrees // 90))


def _find_position(frame):
    best = None
    for i, (kind, x, y, rotation) in enumerate(_BASKETS):
        template = _rot(_H if kind == "h" else _D, rotation)
        rr, cc = np.nonzero(template == 2)
        seen = frame[y + rr, x + cc]
        missing = int(np.count_nonzero(seen != 2))
        # Prefer an exact match; normalize for different outline lengths.
        score = missing / len(rr)
        if best is None or score < best[0]:
            best = (score, i)
    return best[1]


def _has_nozzle(frame, position):
    x, y, rotation = _NOZZLES[position]
    template = _rot(_N, rotation)
    rr, cc = np.nonzero(template == 2)
    return bool(np.all(frame[y + rr, x + cc] == 2))


def _palette(frame):
    buttons = []
    y = 2
    for x in range(18, 60):
        patch = frame[y:y + 5, x:x + 5]
        if patch.shape != (5, 5):
            continue
        if not (np.all(patch[0, :] == 4) and
                np.all(patch[4, :] == 4) and
                np.all(patch[:, 0] == 4) and
                np.all(patch[:, 4] == 4)):
            continue
        color = int(patch[2, 2])
        if not np.all(patch[1:4, 1:4] == color):
            continue
        buttons.append((x, y, color))

    colors = {}
    selected = None
    for x, y, color in buttons:
        colors.setdefault(color, (y + 2, x + 2))
        if y + 5 < 64 and np.all(frame[y + 5, x:x + 5] == 0):
            selected = color
    if selected is None:
        selected = 15 if 15 in colors else (buttons[0][2] if buttons else 0)
    return colors, selected


def _distance(a, b):
    return min((b - a) % 8, (a - b) % 8)


def _move_action(a, b):
    """One shortest movement step from position a towards position b."""
    if a == b:
        return None, a
    clockwise = (b - a) % 8
    counterclockwise = (a - b) % 8
    nxt = (a + 1) % 8 if clockwise <= counterclockwise else (a - 1) % 8
    r0, c0 = _POS[a]
    r1, c1 = _POS[nxt]
    if r1 < r0:
        action = "ACTION1"
    elif r1 > r0:
        action = "ACTION2"
    elif c1 < c0:
        action = "ACTION3"
    else:
        action = "ACTION4"
    return action, nxt


def _stroke_masks(with_nozzle):
    """Return (pixel mask, tool position, narrow-tool flag) for each stroke."""
    masks = [0] * (12 if with_nozzle else 8)
    bit = 0
    for r in range(10):
        for c in range(10):
            if c == r or c == 9 - r:
                continue
            tests = [
                r < 5,             # top half
                c >= r,            # upper/right diagonal
                c >= 5,            # right half
                c >= 9 - r,        # lower/right diagonal
                r >= 5,            # bottom half
                c <= r,            # lower/left diagonal
                c < 5,             # left half
                c <= 9 - r,        # upper/left diagonal
            ]
            if with_nozzle:
                tests += [
                    r < 3 and 3 <= c < 7,
                    3 <= r < 7 and c >= 7,
                    r >= 7 and 3 <= c < 7,
                    3 <= r < 7 and c < 3,
                ]
            for i, yes in enumerate(tests):
                if yes:
                    masks[i] |= 1 << bit
            bit += 1

    ops = [(masks[i], i, False) for i in range(8)]
    if with_nozzle:
        ops.extend((masks[8 + j], p, True)
                   for j, p in enumerate((0, 2, 4, 6)))
    return ops


def _plan(frame, position, selected, palette, with_nozzle):
    target = frame[3:13, 3:13]
    canvas = frame[34:44, 27:37]
    ops = _stroke_masks(with_nozzle)

    target_masks = [0] * 16
    target_colors = []
    mismatch = 0
    bit = 0
    for r in range(10):
        for c in range(10):
            if c == r or c == 9 - r:
                continue
            color = int(target[r, c])
            target_colors.append(color)
            target_masks[color] |= 1 << bit
            if int(canvas[r, c]) != color:
                mismatch |= 1 << bit
            bit += 1

    def stroke_color(uncovered):
        if not uncovered:
            return None
        low = uncovered & -uncovered
        color = target_colors[low.bit_length() - 1]
        if color not in palette or (uncovered & ~target_masks[color]):
            return None
        return color

    # Normally a matching canvas already caused a level transition. If it
    # did not, make a harmless stroke to trigger the simulator's goal check.
    if mismatch == 0:
        choices = []
        for i, (mask, p, narrow) in enumerate(ops):
            color = stroke_color(mask)
            if color is not None:
                cost = _distance(position, p) + (color != selected) + 1
                choices.append((cost, i, color))
        if choices:
            _, i, color = min(choices)
            strokes = ((i, color),)
        else:
            # No harmless stroke exists. Make one stroke and replan.
            if selected not in palette:
                return []
            strokes = ((0, selected),)
    else:
        inf = 1000

        @lru_cache(None)
        def min_strokes(covered):
            if not (mismatch & ~covered):
                return 0
            best = inf
            for mask, _, _ in ops:
                uncovered = mask & ~covered
                if stroke_color(uncovered) is None:
                    continue
                nxt = covered | mask
                if not (mismatch & ~nxt):
                    return 1
                result = min_strokes(nxt)
                if result + 1 < best:
                    best = result + 1
            return best

        needed = min_strokes(0)
        if needed >= inf:
            return []

        @lru_cache(None)
        def best_route(covered, prev_op, prev_color):
            remaining = min_strokes(covered)
            if remaining == 0:
                if prev_op < 0:
                    return 0, ()
                p = ops[prev_op][1]
                return (_distance(position, p) +
                        int(selected != prev_color)), ()

            best_cost = inf
            best_sequence = ()
            for i, (mask, p, _) in enumerate(ops):
                uncovered = mask & ~covered
                color = stroke_color(uncovered)
                if color is None:
                    continue
                nxt = covered | mask
                if min_strokes(nxt) != remaining - 1:
                    continue

                connection = 0
                if prev_op >= 0:
                    connection = (_distance(p, ops[prev_op][1]) +
                                  int(color != prev_color))
                sub_cost, sub_sequence = best_route(nxt, i, color)
                cost = 1 + connection + sub_cost
                if cost < best_cost:
                    best_cost = cost
                    # Reverse search selects the final stroke first.
                    best_sequence = sub_sequence + ((i, color),)
            return best_cost, best_sequence

        _, strokes = best_route(0, -1, -1)

    actions = []
    p = position
    color = selected
    for i, wanted_color in strokes:
        _, dest, narrow = ops[i]
        while p != dest:
            action, p = _move_action(p, dest)
            actions.append((action, None, None))

        if color != wanted_color:
            row, col = palette[wanted_color]
            actions.append(("ACTION6", row, col))
            color = wanted_color

        if narrow:
            x, y, rotation = _NOZZLES[dest]
            template = _rot(_N, rotation)
            row = y + template.shape[0] // 2
            col = x + template.shape[1] // 2
            actions.append(("ACTION6", row, col))
        else:
            actions.append(("ACTION5", None, None))
    return actions


def act(obs, available, memory):
    frame = np.asarray(obs)
    fingerprint = tuple(int(v) for v in frame[3:13, 3:13].flat)

    if memory.get("target") != fingerprint:
        memory.clear()
        memory["target"] = fingerprint

    queue = memory.get("queue")
    if queue:
        return tuple(queue.pop(0))

    position = _find_position(frame)

    # At a diagonal position the narrow tool, if present, is hidden.
    # Move once to a cardinal position to inspect it.
    if position % 2:
        destination = (position - 1) % 8
        action, _ = _move_action(position, destination)
        return action, None, None

    with_nozzle = _has_nozzle(frame, position)
    palette, selected = _palette(frame)
    actions = _plan(frame, position, selected, palette, with_nozzle)

    if actions:
        memory["queue"] = actions[1:]
        return tuple(actions[0])

    # A reset is preferable to repeatedly issuing an unproductive action
    # if the frame could not be decoded or no plan was found.
    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    return "ACTION5", None, None

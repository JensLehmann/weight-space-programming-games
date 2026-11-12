import numpy as np
from functools import lru_cache

# Brush positions, clockwise from the top.
_POSITIONS = (
    (25, 24, 180), (33, 21, 0),
    (38, 32, 270), (33, 40, 90),
    (25, 45, 0), (14, 40, 180),
    (17, 32, 90), (14, 21, 270),
)
_CW_ACTIONS = (4, 2, 2, 3, 3, 1, 1, 4)
_CCW_ACTIONS = (3, 3, 1, 1, 4, 4, 2, 2)

_MINI_LAYOUT = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}

_CELLS = tuple(
    (r, c)
    for r in range(10)
    for c in range(10)
    if c != r and c != 9 - r
)


def _distance(a, b):
    return min((a - b) % 8, (b - a) % 8)


def _movement(a, b):
    result = []
    clockwise = (b - a) % 8
    counterclockwise = (a - b) % 8
    if clockwise <= counterclockwise:
        for _ in range(clockwise):
            result.append(("ACTION%d" % _CW_ACTIONS[a], None, None))
            a = (a + 1) % 8
    else:
        for _ in range(counterclockwise):
            result.append(("ACTION%d" % _CCW_ACTIONS[a], None, None))
            a = (a - 1) % 8
    return result


def _templates():
    horizontal = np.full((9, 14), 15, dtype=int)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[0, 1:13] = -1
    horizontal[8, :] = 2

    rows = (
        "......2..........",
        ".....222.........",
        "....22f22........",
        "...22fff22.......",
        "..22fffff22......",
        ".22fffffff22.....",
        "22fffffffff22....",
        "..ffffffffff22...",
        "...ffffffffff22..",
        "....ffffffffff22.",
        ".....ffffffffff22",
        "......ffffffff22.",
        ".......ffffff22..",
        "........ffff22...",
        ".........ff22....",
        "..........22.....",
        "..........2......",
    )
    diagonal = np.array([
        [-1 if v == "." else 15 if v == "f" else 2 for v in row]
        for row in rows
    ], dtype=int)

    mini = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=int)
    return horizontal, diagonal, mini


def _rotate(template, rotation):
    return np.rot90(template, k=(-rotation // 90) % 4)


def _read_palette(obs):
    palette = {}
    selected = None
    # The palette occupies the upper UI area.
    for y in range(14):
        for x in range(60):
            block = obs[y:y + 5, x:x + 5]
            if not (
                np.all(block[0, :] == 4)
                and np.all(block[4, :] == 4)
                and np.all(block[:, 0] == 4)
                and np.all(block[:, 4] == 4)
            ):
                continue
            colour = int(block[2, 2])
            if not np.all(block[1:4, 1:4] == colour):
                continue
            palette.setdefault(colour, (y + 2, x + 2))
            if np.all(obs[y + 5, x:x + 5] == 0):
                selected = colour
    return palette, selected


def _find_squares(obs, palette):
    # Solid 10x10 components identify the reference and painting canvas.
    colours = [c for c in palette if c not in (2, 3, 4, 5)]
    if not colours:
        colours = [0, 8, 9, 11, 12, 14, 15]
    mask = np.isin(obs, colours)
    seen = np.zeros((64, 64), dtype=bool)
    squares = []

    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(y, x)]
        count = 0
        ymin = ymax = y
        xmin = xmax = x

        while stack:
            r, c = stack.pop()
            count += 1
            ymin, ymax = min(ymin, r), max(ymax, r)
            xmin, xmax = min(xmin, c), max(xmax, c)
            for nr, nc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                if (0 <= nr < 64 and 0 <= nc < 64
                        and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        if count == 100 and ymax - ymin == 9 and xmax - xmin == 9:
            squares.append((ymin, xmin))

    return squares


def _read_images(obs, palette):
    squares = _find_squares(obs, palette)
    reference_candidates = [
        p for p in squares if p[0] < 18 and p[1] < 18
    ]
    if reference_candidates:
        reference = min(
            reference_candidates,
            key=lambda p: abs(p[0] - 3) + abs(p[1] - 3)
        )
    elif squares:
        reference = min(squares, key=lambda p: p[0] + p[1])
    else:
        reference = (3, 3)

    canvas_candidates = [p for p in squares if p != reference]
    if canvas_candidates:
        canvas = min(
            canvas_candidates,
            key=lambda p: abs(p[0] - 34) + abs(p[1] - 27)
        )
    else:
        canvas = (34, 27)

    ry, rx = reference
    cy, cx = canvas
    return (
        obs[ry:ry + 10, rx:rx + 10].copy(),
        obs[cy:cy + 10, cx:cx + 10].copy(),
        reference,
    )


def _read_brush(obs, selected):
    horizontal, diagonal, mini = _templates()
    best = None

    for position, (x, y, rotation) in enumerate(_POSITIONS):
        template = _rotate(
            horizontal if position % 2 == 0 else diagonal, rotation
        )
        h, w = template.shape
        block = obs[y:y + h, x:x + w]
        outline = template == 2
        fill = template == 15
        outline_score = float(np.mean(block[outline] == 2))
        counts = np.bincount(block[fill].astype(int), minlength=16)
        colour = int(np.argmax(counts))
        fill_score = float(counts[colour]) / int(np.count_nonzero(fill))
        score = 2.0 * outline_score + fill_score

        if best is None or score > best[0]:
            best = (score, position, colour)

    _, position, colour = best
    if selected is not None:
        # The brush fill is normally conclusive; the marker is a backup.
        if best[0] < 2.8:
            colour = selected

    has_mini = None
    if position in _MINI_LAYOUT:
        x, y, rotation = _MINI_LAYOUT[position]
        template = _rotate(mini, rotation)
        expected = template.copy()
        expected[expected == 15] = colour
        h, w = expected.shape
        block = obs[y:y + h, x:x + w]
        visible = expected >= 0
        has_mini = bool(np.array_equal(block[visible], expected[visible]))

    return position, colour, has_mini


def _operations(has_mini):
    operations = []

    for position in range(8):
        bits = 0
        for bit, (r, c) in enumerate(_CELLS):
            applies = (
                r < 5 if position == 0 else
                c >= r if position == 1 else
                c >= 5 if position == 2 else
                c >= 9 - r if position == 3 else
                r >= 5 if position == 4 else
                c <= r if position == 5 else
                c < 5 if position == 6 else
                c <= 9 - r
            )
            if applies:
                bits |= 1 << bit
        operations.append((position, False, bits))

    if has_mini:
        for position in (0, 2, 4, 6):
            bits = 0
            for bit, (r, c) in enumerate(_CELLS):
                applies = (
                    r < 3 and 3 <= c < 7 if position == 0 else
                    3 <= r < 7 and c >= 7 if position == 2 else
                    r >= 7 and 3 <= c < 7 if position == 4 else
                    3 <= r < 7 and c < 3
                )
                if applies:
                    bits |= 1 << bit
            operations.append((position, True, bits))

    return operations


def _plan(target, canvas, palette, start_position, start_colour, has_mini):
    operations = _operations(has_mini)
    targets = [int(target[r, c]) for r, c in _CELLS]
    full = (1 << len(_CELLS)) - 1
    mismatch = 0
    colour_bits = {}

    for bit, (r, c) in enumerate(_CELLS):
        colour = targets[bit]
        colour_bits[colour] = colour_bits.get(colour, 0) | (1 << bit)
        if int(canvas[r, c]) != colour:
            mismatch |= 1 << bit

    usable = set(palette)
    choices = {}
    infinity = 10**9

    @lru_cache(None)
    def legal(remaining):
        result = []
        for index, (_, _, mask) in enumerate(operations):
            intersection = remaining & mask
            if not intersection:
                continue
            bit = (intersection & -intersection).bit_length() - 1
            colour = targets[bit]
            if (colour in usable
                    and intersection & ~colour_bits[colour] == 0):
                result.append((index, colour, remaining & ~mask))
        return tuple(result)

    @lru_cache(None)
    def solve(remaining, next_position, next_colour):
        if remaining & mismatch == 0:
            if next_position < 0:
                return 0
            return (_distance(start_position, next_position)
                    + int(start_colour != next_colour))

        best = infinity
        best_choice = None
        for index, colour, new_remaining in legal(remaining):
            position = operations[index][0]
            extra = 1
            if next_position >= 0:
                extra += (_distance(position, next_position)
                          + int(colour != next_colour))
            cost = extra + solve(new_remaining, position, colour)
            if cost < best:
                best = cost
                best_choice = (index, colour, new_remaining)

        if best_choice is not None:
            choices[(remaining, next_position, next_colour)] = best_choice
        return best

    if mismatch == 0:
        # A harmless paint action is needed to invoke the success check.
        candidates = legal(full)
        if not candidates:
            return None
        index, colour, _ = min(
            candidates,
            key=lambda item: (
                _distance(start_position, operations[item[0]][0])
                + int(start_colour != item[1]) + 1
            )
        )
        strokes = [(index, colour)]
    else:
        if solve(full, -1, -1) >= infinity:
            return None

        reverse_strokes = []
        remaining, next_position, next_colour = full, -1, -1
        while remaining & mismatch:
            index, colour, new_remaining = choices[
                (remaining, next_position, next_colour)
            ]
            reverse_strokes.append((index, colour))
            remaining = new_remaining
            next_position = operations[index][0]
            next_colour = colour
        strokes = reverse_strokes[::-1]

    actions = []
    position, colour = start_position, start_colour

    for index, new_colour in strokes:
        destination, is_mini, _ = operations[index]
        actions.extend(_movement(position, destination))
        position = destination

        if new_colour != colour:
            row, col = palette[new_colour]
            actions.append(("ACTION6", int(row), int(col)))
            colour = new_colour

        if is_mini:
            x, y, rotation = _MINI_LAYOUT[position]
            h, w = _rotate(_templates()[2], rotation).shape
            actions.append(("ACTION6", y + h // 2, x + w // 2))
        else:
            actions.append(("ACTION5", None, None))

    return actions


def act(obs, available, memory):
    obs = np.asarray(obs)
    available = set(available)

    saved = memory.get("cd82_plan")
    if saved and saved["actions"]:
        ry, rx = saved["reference"]
        fingerprint = tuple(
            int(v) for v in obs[ry:ry + 10, rx:rx + 10].flat
        )
        if fingerprint == saved["target"]:
            action = tuple(saved["actions"].pop(0))
            if action[0] in available:
                return action
        memory.pop("cd82_plan", None)

    palette, selected = _read_palette(obs)
    target, canvas, reference = _read_images(obs, palette)
    position, colour, has_mini = _read_brush(obs, selected)

    if has_mini is None:
        # Diagonal positions hide the small brush. One move reveals whether
        # this level supports small rectangular stamps.
        neighbours = ((position - 1) % 8, (position + 1) % 8)
        destination = min(neighbours, key=lambda p: _distance(p, 0))
        action = _movement(position, destination)[0]
        if action[0] in available:
            return action

    actions = _plan(
        target, canvas, palette, position, colour, bool(has_mini)
    )

    if actions:
        memory["cd82_plan"] = {
            "actions": actions[1:],
            "reference": reference,
            "target": tuple(int(v) for v in target.flat),
        }
        if actions[0][0] in available:
            return actions[0]
        memory.pop("cd82_plan", None)

    # Normally unreachable: recover if the image could not be interpreted.
    if "RESET" in available:
        memory.pop("cd82_plan", None)
        return ("RESET", None, None)
    if "ACTION5" in available:
        return ("ACTION5", None, None)
    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if name in available:
            return (name, None, None)
    if "ACTION6" in available:
        return ("ACTION6", 4, 4)
    return ("RESET", None, None)

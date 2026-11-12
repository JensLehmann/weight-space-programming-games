import numpy as np
from collections import deque
from functools import lru_cache

INF = 10**8

# Clockwise basket positions, starting at the top.
POSITIONS = (
    (25, 24, 180), (33, 21, 0),
    (38, 32, 270), (33, 40, 90),
    (25, 45, 0), (14, 40, 180),
    (17, 32, 90), (14, 21, 270),
)
GRID_POSITIONS = (
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
)
OP_ORIENTATION = tuple(range(8)) + (0, 2, 4, 6)
RELEVANT = tuple(
    r * 10 + c
    for r in range(10)
    for c in range(10)
    if r != c and r + c != 9
)
FULL_MASK = sum(1 << i for i in RELEVANT)

SMALL_TOOL = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}


def _make_masks():
    masks = [0] * 12
    for i in RELEVANT:
        r, c = divmod(i, 10)
        conditions = (
            r < 5,
            c >= r,
            c >= 5,
            r + c >= 9,
            r >= 5,
            c <= r,
            c < 5,
            r + c <= 9,
            r < 3 and 3 <= c < 7,
            3 <= r < 7 and c >= 7,
            r >= 7 and 3 <= c < 7,
            3 <= r < 7 and c < 3,
        )
        for j, condition in enumerate(conditions):
            if condition:
                masks[j] |= 1 << i
    return tuple(masks)


MASKS = _make_masks()


def _make_templates():
    horizontal = np.full((9, 14), 15, dtype=int)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[8, :] = 2
    horizontal[0, 1:13] = -1

    diagonal = np.full((17, 17), -1, dtype=int)
    for r in range(7):
        lo, hi = 6 - r, 6 + r
        diagonal[r, lo:hi + 1] = 2
        diagonal[r, lo + 2:hi - 1] = 15
    for r in range(7, 15):
        lo = r - 5
        hi = r + 6 if r <= 10 else 26 - r
        diagonal[r, lo:hi - 1] = 15
        diagonal[r, hi - 1:hi + 1] = 2
    diagonal[15, 10:12] = 2
    diagonal[16, 10] = 2

    baskets = []
    for i, (_, _, rotation) in enumerate(POSITIONS):
        base = horizontal if i % 2 == 0 else diagonal
        baskets.append(np.rot90(base, -(rotation // 90)).copy())

    small = np.full((5, 6), 15, dtype=int)
    small[:, 0] = 2
    small[:, 5] = 2
    small[3] = [2, 2, 0, 0, 2, 2]
    small[4] = [-1, 2, 2, 2, 2, -1]
    tools = {
        p: np.rot90(small, -(rotation // 90)).copy()
        for p, (_, _, rotation) in SMALL_TOOL.items()
    }
    return baskets, tools


BASKET_TEMPLATES, TOOL_TEMPLATES = _make_templates()


def _read_palette(a):
    palette = {}
    selected = None

    for yy, xx in np.argwhere(a[:-4, :-4] == 4):
        y, x = int(yy), int(xx)
        if a[y + 4, x + 4] != 4:
            continue
        block = a[y:y + 5, x:x + 5]
        if not (
            np.all(block[0] == 4)
            and np.all(block[4] == 4)
            and np.all(block[:, 0] == 4)
            and np.all(block[:, 4] == 4)
        ):
            continue
        color = int(block[2, 2])
        if not np.all(block[1:4, 1:4] == color):
            continue
        if color not in palette:
            palette[color] = (y + 2, x + 2)
        if y + 5 < 64 and np.all(a[y + 5, x:x + 5] == 0):
            selected = color

    return palette, selected


def _read_images(a, palette):
    # Locate solid 10x10 image areas. The canonical positions also
    # provide reliable fallbacks for uniformly background-colored areas.
    colors = set(palette) | {0, 15}
    good = np.isin(a, list(colors)).astype(np.int32)
    integral = np.pad(good, ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    sums = (
        integral[10:, 10:] - integral[:-10, 10:]
        - integral[10:, :-10] + integral[:-10, :-10]
    )
    candidates = [
        (int(x), int(y)) for y, x in np.argwhere(sums == 100)
    ]

    target_candidates = [
        (x, y) for x, y in candidates if x < 18 and y < 18
    ]
    if target_candidates:
        tx, ty = min(
            target_candidates,
            key=lambda p: (p[0] - 3)**2 + (p[1] - 3)**2,
        )
    else:
        tx, ty = 3, 3

    canvas_candidates = [
        (x, y) for x, y in candidates
        if (x >= 18 or y >= 18) and (x, y) != (tx, ty)
    ]
    if canvas_candidates:
        cx, cy = min(
            canvas_candidates,
            key=lambda p: (p[0] - 27)**2 + (p[1] - 34)**2,
        )
    else:
        cx, cy = 27, 34

    target_pixels = a[ty:ty + 10, tx:tx + 10].reshape(-1)
    canvas_pixels = a[cy:cy + 10, cx:cx + 10].reshape(-1)

    target = [-1] * 100
    canvas = [-1] * 100
    for i in RELEVANT:
        target[i] = int(target_pixels[i])
        canvas[i] = int(canvas_pixels[i])
    return tuple(target), tuple(canvas)


def _read_basket(a, marked_color, memory):
    best_score = -1.0
    position = int(memory.get("cd82_position", 0))
    color = int(memory.get("cd82_color", 15))

    for p, (x, y, _) in enumerate(POSITIONS):
        template = BASKET_TEMPLATES[p]
        h, w = template.shape
        patch = a[y:y + h, x:x + w]
        rim = template == 2
        score = float(np.mean(patch[rim] == 2))
        if score > best_score:
            best_score = score
            position = p
            values = patch[template == 15]
            color = int(np.bincount(values, minlength=16).argmax())

    if marked_color is not None:
        color = marked_color

    if best_score < 0.65:
        position = int(memory.get("cd82_position", position))

    # A diagonal basket hides the small tool, so retain our knowledge.
    tool = memory.get("cd82_tool", True)
    if position in SMALL_TOOL and best_score > 0.8:
        x, y, _ = SMALL_TOOL[position]
        template = TOOL_TEMPLATES[position]
        h, w = template.shape
        patch = a[y:y + h, x:x + w]
        tool = bool(np.mean(patch[template == 2] == 2) > 0.9)
        memory["cd82_tool"] = tool

    return position, color, bool(tool)


def _routes(available):
    index = {p: i for i, p in enumerate(GRID_POSITIONS)}
    deltas = (
        ("ACTION1", -1, 0), ("ACTION2", 1, 0),
        ("ACTION3", 0, -1), ("ACTION4", 0, 1),
    )
    edges = [[] for _ in range(8)]
    for p, (r, c) in enumerate(GRID_POSITIONS):
        for action, dr, dc in deltas:
            if action not in available:
                continue
            q = index.get((r + dr, c + dc))
            if q is not None:
                edges[p].append((q, action))

    distances = [[INF] * 8 for _ in range(8)]
    first = [[None] * 8 for _ in range(8)]
    for start in range(8):
        distances[start][start] = 0
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q, action in edges[p]:
                if distances[start][q] == INF:
                    distances[start][q] = distances[start][p] + 1
                    first[start][q] = (
                        action if p == start else first[start][p]
                    )
                    queue.append(q)
    return distances, first, edges


def _plan(target, canvas, position, color, palette, tool,
          available, distances):
    allowed_colors = {color}
    if "ACTION6" in available:
        allowed_colors.update(palette)

    operations = []
    if "ACTION5" in available:
        operations.extend(range(8))
    if tool and "ACTION6" in available:
        operations.extend(range(8, 12))

    color_masks = {}
    mismatch = 0
    for i in RELEVANT:
        c = target[i]
        color_masks[c] = color_masks.get(c, 0) | (1 << i)
        if canvas[i] != c:
            mismatch |= 1 << i

    @lru_cache(None)
    def options(remaining):
        result = []
        for op in operations:
            exposed = remaining & MASKS[op]
            if not exposed:
                continue
            i = (exposed & -exposed).bit_length() - 1
            c = target[i]
            if c not in allowed_colors:
                continue
            if exposed & ~color_masks[c]:
                continue
            result.append((remaining & ~MASKS[op], op, c))
        return tuple(result)

    @lru_cache(None)
    def solve(remaining, next_position, next_color):
        # Everything still constrained already matches the starting canvas.
        if not (remaining & mismatch):
            if next_position < 0:
                return 0, ()
            return (
                distances[position][next_position]
                + int(color != next_color),
                (),
            )

        best_cost, best_sequence = INF, ()
        for earlier, op, c in options(remaining):
            p = OP_ORIENTATION[op]
            transition = 1
            if next_position >= 0:
                transition += distances[p][next_position]
                transition += int(c != next_color)

            cost, sequence = solve(earlier, p, c)
            cost += transition
            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence + ((op, c),)

        return best_cost, best_sequence

    cost, sequence = solve(FULL_MASK, -1, -1)
    if cost >= INF:
        return None

    # Equality is checked after stamping, not after ordinary movement.
    if not sequence:
        best = None
        for _, op, c in options(FULL_MASK):
            p = OP_ORIENTATION[op]
            value = distances[position][p] + int(color != c) + 1
            if best is None or value < best[0]:
                best = (value, op, c)
        if best is None or best[0] >= INF:
            return None
        sequence = ((best[1], best[2]),)

    return list(sequence)


def act(obs, available, memory):
    a = np.asarray(obs, dtype=int)
    available = {
        str(action).upper().split(".")[-1] for action in available
    }

    palette, marked_color = _read_palette(a)
    target, canvas = _read_images(a, palette)
    position, color, tool = _read_basket(a, marked_color, memory)
    distances, first, edges = _routes(available)

    memory["cd82_position"] = position
    memory["cd82_color"] = color

    invalid_plan = (
        memory.get("cd82_target") != target
        or memory.get("cd82_expected") != canvas
        or memory.get("cd82_plan_tool") != tool
        or not memory.get("cd82_plan")
    )
    if invalid_plan:
        memory["cd82_plan"] = _plan(
            target, canvas, position, color, palette, tool,
            available, distances,
        )
        memory["cd82_plan_tool"] = tool

    memory["cd82_target"] = target
    memory["cd82_expected"] = canvas
    plan = memory.get("cd82_plan")

    if not plan:
        # Recovery if an unusual observation could not be reconstructed.
        if "RESET" in available and not memory.get("cd82_reset_used"):
            memory["cd82_reset_used"] = True
            memory.pop("cd82_plan", None)
            return "RESET", None, None
        for action in ("ACTION1", "ACTION2", "ACTION3",
                       "ACTION4", "ACTION5"):
            if action in available:
                return action, None, None
        if "ACTION6" in available:
            return "ACTION6", 0, 0
        return "RESET", None, None

    op, desired_color = plan[0]
    desired_position = OP_ORIENTATION[op]

    if position != desired_position:
        action = first[position][desired_position]
        if action is not None:
            for q, edge_action in edges[position]:
                if edge_action == action:
                    memory["cd82_position"] = q
                    break
            return action, None, None

    if color != desired_color:
        if desired_color in palette and "ACTION6" in available:
            row, col = palette[desired_color]
            memory["cd82_color"] = desired_color
            return "ACTION6", int(row), int(col)
        memory.pop("cd82_plan", None)
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION1", None, None

    predicted = list(canvas)
    mask = MASKS[op]
    for i in RELEVANT:
        if mask & (1 << i):
            predicted[i] = desired_color
    memory["cd82_expected"] = tuple(predicted)
    memory["cd82_plan"] = plan[1:]

    if op < 8:
        return "ACTION5", None, None

    x, y, _ = SMALL_TOOL[desired_position]
    h, w = TOOL_TEMPLATES[desired_position].shape
    return "ACTION6", int(y + h // 2), int(x + w // 2)

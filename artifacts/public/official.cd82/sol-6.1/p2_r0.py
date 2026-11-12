import heapq
from collections import deque
import numpy as np


def _geometry():
    # Basket orientations, in clockwise order.
    positions = [
        (25, 24, 180), (33, 21, 0),
        (38, 32, 270), (33, 40, 90),
        (25, 45, 0), (14, 40, 180),
        (17, 32, 90), (14, 21, 270),
    ]

    horizontal = np.full((9, 14), -1, dtype=np.int8)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[1:8, 1:13] = 15
    horizontal[8, :] = 2

    diagonal = np.full((17, 17), -1, dtype=np.int8)
    diagonal[0, 6] = 2
    for r in range(1, 7):
        lo, hi = 6 - r, 6 + r
        for c in range(lo, hi + 1):
            diagonal[r, c] = (
                2 if c < lo + 2 or c > hi - 2 else 15
            )
    for r in range(7, 15):
        lo = r - 5
        hi = r + 6 if r <= 10 else 26 - r
        diagonal[r, lo:hi + 1] = 15
        diagonal[r, hi - 1:hi + 1] = 2
    diagonal[15, 10:12] = 2
    diagonal[16, 10] = 2

    templates = []
    for i, (x, y, rotation) in enumerate(positions):
        base = horizontal if i % 2 == 0 else diagonal
        template = np.rot90(base, k=-(rotation // 90))
        templates.append((x, y, template))

    small_base = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=np.int8)

    small_templates = {}
    small_specs = {
        0: (29, 18, 180),
        2: (48, 36, 270),
        4: (29, 55, 0),
        6: (11, 36, 90),
    }
    for p, (x, y, rotation) in small_specs.items():
        template = np.rot90(small_base, k=-(rotation // 90))
        small_templates[p] = (x, y, template)

    cells = [
        (r, c) for r in range(10) for c in range(10)
        if r != c and r + c != 9
    ]

    masks = [0] * 12
    for bit, (r, c) in enumerate(cells):
        conditions = [
            r < 5,             # Top half
            c >= r,            # Upper-right diagonal stroke
            c >= 5,            # Right half
            c >= 9 - r,        # Lower-right diagonal stroke
            r >= 5,            # Bottom half
            c <= r,            # Lower-left diagonal stroke
            c < 5,             # Left half
            c <= 9 - r,        # Upper-left diagonal stroke
            r < 3 and 3 <= c < 7,
            3 <= r < 7 and c >= 7,
            r >= 7 and 3 <= c < 7,
            3 <= r < 7 and c < 3,
        ]
        for i, condition in enumerate(conditions):
            if condition:
                masks[i] |= 1 << bit

    op_positions = list(range(8)) + [0, 2, 4, 6]
    return templates, small_templates, cells, masks, op_positions


def _read_palette(image):
    candidates = []
    for y in range(60):
        for x in range(60):
            if image[y, x] != 4 or image[y, x + 4] != 4:
                continue
            patch = image[y:y + 5, x:x + 5]
            if not (
                np.all(patch[0, :] == 4)
                and np.all(patch[4, :] == 4)
                and np.all(patch[:, 0] == 4)
                and np.all(patch[:, 4] == 4)
            ):
                continue
            color = int(patch[2, 2])
            if np.all(patch[1:4, 1:4] == color):
                candidates.append((x, y, color))

    # Prefer the palette in the upper control panel.
    upper = [item for item in candidates if item[1] <= 10 and item[0] >= 18]
    if upper:
        candidates = upper

    palette = {}
    for x, y, color in sorted(candidates, key=lambda item: (item[1], item[0])):
        if color not in palette:
            palette[color] = (y + 2, x + 2)
    return palette


def _read_boards(image, cells):
    # Solid 10x10 coloured patches distinguish the target and canvas
    # from the background, controls, and basket outlines.
    good = (
        (image != 2) & (image != 3)
        & (image != 4) & (image != 5)
    )
    integral = np.pad(
        good.astype(np.int16), ((1, 0), (1, 0))
    ).cumsum(axis=0).cumsum(axis=1)
    totals = (
        integral[10:, 10:] - integral[:-10, 10:]
        - integral[10:, :-10] + integral[:-10, :-10]
    )
    ys, xs = np.nonzero(totals == 100)
    candidates = [(int(x), int(y)) for y, x in zip(ys, xs)]

    target_xy = (3, 3)
    canvas_xy = (27, 34)

    if candidates:
        target_xy = min(
            candidates,
            key=lambda xy: (xy[0] - 3) ** 2 + (xy[1] - 3) ** 2,
        )
        remaining = [xy for xy in candidates if xy != target_xy]
        if remaining:
            canvas_xy = min(
                remaining,
                key=lambda xy: (xy[0] - 27) ** 2 + (xy[1] - 34) ** 2,
            )

    tx, ty = target_xy
    cx, cy = canvas_xy
    target = [int(image[ty + r, tx + c]) for r, c in cells]
    board = [int(image[cy + r, cx + c]) for r, c in cells]
    return target, board


def _read_basket(image, templates, small_templates):
    best_score = (-1.0, -1)
    orientation = 0
    color = 15

    for p, (x, y, template) in enumerate(templates):
        h, w = template.shape
        patch = image[y:y + h, x:x + w]
        outline = template == 2
        count = int(np.count_nonzero(patch[outline] == 2))
        score = (count / int(np.count_nonzero(outline)), count)

        if score > best_score:
            best_score = score
            orientation = p
            core = patch[template == 15]
            if core.size:
                color = int(np.bincount(core.astype(int), minlength=16).argmax())

    small_present = None
    if orientation in small_templates:
        x, y, template = small_templates[orientation]
        h, w = template.shape
        patch = image[y:y + h, x:x + w]
        outline = template == 2
        small_present = bool(np.all(patch[outline] == 2))

    return orientation, color, small_present


def _movement_paths(available):
    coordinates = [
        (0, 1), (0, 2), (1, 2), (2, 2),
        (2, 1), (2, 0), (1, 0), (0, 0),
    ]
    lookup = {xy: p for p, xy in enumerate(coordinates)}
    edges = [[] for _ in range(8)]

    for p, (r, c) in enumerate(coordinates):
        destinations = [
            (max(0, r - 1), c),
            (min(2, r + 1), c),
            (r, max(0, c - 1)),
            (r, min(2, c + 1)),
        ]
        for action_id, xy in enumerate(destinations, 1):
            name = "ACTION" + str(action_id)
            q = lookup.get(xy)
            if name in available and q is not None and q != p:
                edges[p].append((q, name))

    distances = [[999] * 8 for _ in range(8)]
    first_actions = [[None] * 8 for _ in range(8)]

    for start in range(8):
        distances[start][start] = 0
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q, action in edges[p]:
                if distances[start][q] == 999:
                    distances[start][q] = distances[start][p] + 1
                    first_actions[start][q] = (
                        action if p == start else first_actions[start][p]
                    )
                    queue.append(q)

    return distances, first_actions


def _plan(target, board, start_position, start_color, masks,
          op_positions, distances, palette, available, small_enabled):
    n = len(target)
    full = (1 << n) - 1
    differences = 0
    color_masks = {}
    for bit, (wanted, actual) in enumerate(zip(target, board)):
        color_masks[wanted] = color_masks.get(wanted, 0) | (1 << bit)
        if wanted != actual:
            differences |= 1 << bit

    allowed_colors = {start_color}
    if "ACTION6" in available:
        allowed_colors.update(palette)

    allowed_ops = []
    if "ACTION5" in available:
        allowed_ops.extend(range(8))
    if "ACTION6" in available and small_enabled is not False:
        allowed_ops.extend(range(8, 12))

    # A state describes the still-uncovered target cells and the first
    # orientation/colour of the already constructed suffix.
    root = (full, 8, -1)
    costs = {root: 0}
    parents = {}
    heap = [(0, full, 8, -1)]
    valid_cache = {}
    best_cost = float("inf")
    best_state = None

    while heap:
        cost, unresolved, next_position, next_color = heapq.heappop(heap)
        state = (unresolved, next_position, next_color)
        if costs.get(state) != cost:
            continue
        if cost >= best_cost:
            break

        # Require at least one stroke, since victory is checked after painting.
        if next_position != 8 and not (unresolved & differences):
            total = (
                cost + distances[start_position][next_position]
                + int(start_color != next_color)
            )
            if total < best_cost:
                best_cost = total
                best_state = state
            continue

        options = valid_cache.get(unresolved)
        if options is None:
            options = []
            for operation in allowed_ops:
                exposed = unresolved & masks[operation]
                if not exposed:
                    continue
                bit = (exposed & -exposed).bit_length() - 1
                color = target[bit]
                if color not in allowed_colors:
                    continue
                if exposed & ~color_masks[color]:
                    continue
                new_unresolved = unresolved & ~masks[operation]
                options.append((operation, color, new_unresolved))
            valid_cache[unresolved] = options

        for operation, color, new_unresolved in options:
            position = op_positions[operation]
            if next_position == 8:
                extra = 1
            else:
                extra = (
                    1 + distances[position][next_position]
                    + int(color != next_color)
                )
            new_cost = cost + extra
            if new_cost >= best_cost:
                continue

            new_state = (new_unresolved, position, color)
            if new_cost < costs.get(new_state, float("inf")):
                costs[new_state] = new_cost
                parents[new_state] = (state, operation, color)
                heapq.heappush(
                    heap, (new_cost, new_unresolved, position, color)
                )

    if best_state is None:
        return None

    operations = []
    state = best_state
    while state != root:
        suffix, operation, color = parents[state]
        operations.append([operation, color])
        state = suffix
    return operations


def act(obs, available, memory):
    image = np.asarray(obs, dtype=np.int16)
    available = set(available)
    templates, small_templates, cells, masks, op_positions = _geometry()

    palette = _read_palette(image)
    target, board = _read_boards(image, cells)
    position, color, observed_small = _read_basket(
        image, templates, small_templates
    )
    distances, first_actions = _movement_paths(available)

    same_target = memory.get("target") == target
    if not same_target:
        memory.clear()

    if observed_small is not None:
        memory["small_enabled"] = observed_small
    small_enabled = memory.get("small_enabled")

    operations = memory.get("operations")
    reusable = (
        same_target
        and memory.get("expected_board") == board
        and bool(operations)
    )
    if reusable and small_enabled is False:
        reusable = not any(operation >= 8 for operation, _ in operations)

    if not reusable:
        operations = _plan(
            target, board, position, color, masks, op_positions,
            distances, palette, available, small_enabled,
        )

    memory["target"] = target
    memory["expected_board"] = board[:]
    memory["operations"] = operations or []

    if not operations:
        # Normally only reached if an observation could not be decoded.
        memory.pop("operations", None)
        if "RESET" in available:
            return "RESET", None, None
        if "ACTION5" in available:
            return "ACTION5", None, None
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if name in available:
                return name, None, None
        if "ACTION6" in available and palette:
            row, col = next(iter(palette.values()))
            return "ACTION6", int(row), int(col)
        return "RESET", None, None

    operation, wanted_color = operations[0]
    wanted_position = op_positions[operation]

    if color != wanted_color:
        click = palette.get(wanted_color)
        if click is not None and "ACTION6" in available:
            row, col = click
            return "ACTION6", int(row), int(col)

    if position != wanted_position:
        action = first_actions[position][wanted_position]
        if action is not None:
            return action, None, None
        memory.pop("operations", None)
        return "RESET", None, None

    # Predict the resulting off-diagonal canvas and consume this stroke.
    expected = board[:]
    bits = masks[operation]
    while bits:
        low = bits & -bits
        bit = low.bit_length() - 1
        expected[bit] = wanted_color
        bits ^= low
    memory["expected_board"] = expected
    memory["operations"] = operations[1:]

    if operation < 8:
        return "ACTION5", None, None

    x, y, template = small_templates[position]
    h, w = template.shape
    return "ACTION6", int(y + h // 2), int(x + w // 2)

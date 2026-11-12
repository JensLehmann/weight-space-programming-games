import heapq
import numpy as np


# Basket positions, in clockwise order: top, top-right, right, ...
_STATIONS = (
    (25, 24, 180),
    (33, 21, 0),
    (38, 32, 270),
    (33, 40, 90),
    (25, 45, 0),
    (14, 40, 180),
    (17, 32, 90),
    (14, 21, 270),
)
_RING = ((0, 1), (0, 2), (1, 2), (2, 2),
         (2, 1), (2, 0), (1, 0), (0, 0))
_MASK_POS = tuple(range(8)) + (0, 2, 4, 6)
_SMALL_STATIONS = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}


def _geometry():
    horizontal = np.full((9, 14), 15, dtype=np.int8)
    horizontal[0, 1:-1] = -1
    horizontal[:, 0] = 2
    horizontal[:, -1] = 2
    horizontal[-1, :] = 2

    diagonal = np.full((17, 17), -1, dtype=np.int8)
    for r in range(17):
        if r <= 6:
            left, right = 6 - r, 6 + r
            diagonal[r, left:right + 1] = 2
            if r >= 2:
                diagonal[r, left + 2:right - 1] = 15
        elif r <= 10:
            left, right = r - 5, r + 6
            diagonal[r, left:right - 1] = 15
            diagonal[r, right - 1:right + 1] = 2
        elif r <= 14:
            left, right = r - 5, 26 - r
            diagonal[r, left:right - 1] = 15
            diagonal[r, right - 1:right + 1] = 2
        elif r == 15:
            diagonal[r, 10:12] = 2
        else:
            diagonal[r, 10] = 2

    small = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=np.int8)

    baskets = []
    for p, (x, y, rotation) in enumerate(_STATIONS):
        base = horizontal if p % 2 == 0 else diagonal
        t = np.rot90(base, -(rotation // 90))
        baskets.append((x, y, t, t == 2, t == 15))

    tools = {}
    for p, (x, y, rotation) in _SMALL_STATIONS.items():
        tools[p] = (x, y, np.rot90(small, -(rotation // 90)))

    r, c = np.mgrid[:10, :10]
    relevant = (r != c) & (r + c != 9)
    conditions = [
        r < 5,
        c >= r,
        c >= 5,
        c >= 9 - r,
        r >= 5,
        c <= r,
        c < 5,
        c <= 9 - r,
        (r < 3) & (c >= 3) & (c < 7),
        (r >= 3) & (r < 7) & (c >= 7),
        (r >= 7) & (c >= 3) & (c < 7),
        (r >= 3) & (r < 7) & (c < 3),
    ]
    raster_masks = [condition & relevant for condition in conditions]
    bit_masks = [
        sum(1 << int(i) for i in np.flatnonzero(mask))
        for mask in raster_masks
    ]
    return baskets, tools, relevant, raster_masks, bit_masks


def _palette(a):
    """Return colour -> (click row, click column)."""
    corners = (
        (a[:-4, :-4] == 4) &
        (a[:-4, 4:] == 4) &
        (a[4:, :-4] == 4) &
        (a[4:, 4:] == 4)
    )
    result = {}
    for y, x in np.argwhere(corners):
        y, x = int(y), int(x)
        patch = a[y:y + 5, x:x + 5]
        if not (
            np.all(patch[0] == 4) and
            np.all(patch[-1] == 4) and
            np.all(patch[:, 0] == 4) and
            np.all(patch[:, -1] == 4)
        ):
            continue
        colour = int(patch[2, 2])
        if np.all(patch[1:4, 1:4] == colour):
            result.setdefault(colour, (y + 2, x + 2))
    return result


def _find_images(a):
    """Locate isolated 10x10 images; canonical coordinates are fallbacks."""
    occupied = a != 5
    seen = np.zeros((64, 64), dtype=bool)
    rectangles = []

    for sy, sx in np.argwhere(occupied):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        stack = [(sy, sx)]
        count = 0
        ymin = ymax = sy
        xmin = xmax = sx

        while stack:
            y, x = stack.pop()
            count += 1
            ymin, ymax = min(ymin, y), max(ymax, y)
            xmin, xmax = min(xmin, x), max(xmax, x)
            for ny, nx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if (0 <= ny < 64 and 0 <= nx < 64 and
                        occupied[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))

        if count == 100 and xmax - xmin == 9 and ymax - ymin == 9:
            rectangles.append((xmin, ymin))

    target_xy = (3, 3)
    canvas_xy = (27, 34)
    if len(rectangles) >= 2:
        canvas_xy = min(
            rectangles, key=lambda q: abs(q[0] - 27) + abs(q[1] - 34)
        )
        target_xy = min(
            (q for q in rectangles if q != canvas_xy),
            key=lambda q: abs(q[0] - 3) + abs(q[1] - 3)
        )
    elif rectangles:
        x, y = rectangles[0]
        if x < 18 and y < 18:
            target_xy = (x, y)
        else:
            canvas_xy = (x, y)
    return target_xy, canvas_xy


def _read_basket(a, geometry):
    baskets, tools = geometry[:2]
    best_score = -1.0
    position, colour = 0, 15

    for p, (x, y, template, outline, interior) in enumerate(baskets):
        h, w = template.shape
        patch = a[y:y + h, x:x + w]
        score = float(np.mean(patch[outline] == 2))
        if score > best_score:
            best_score = score
            position = p
            values = patch[interior].astype(np.int64)
            colour = int(np.bincount(values, minlength=16).argmax())

    has_small = None
    if position in tools:
        x, y, template = tools[position]
        h, w = template.shape
        patch = a[y:y + h, x:x + w]
        has_small = bool(
            np.all(patch[template == 2] == 2) and
            np.all(patch[template == 0] == 0)
        )
    return position, colour, has_small


def _distance(a, b):
    d = abs(a - b)
    return min(d, 8 - d)


def _plan(target, canvas, start_pos, start_colour,
          bit_masks, allow_small, palette_colours):
    """
    Reverse A*: covered pixels have already been assigned to later strokes.
    The remaining part of a candidate last stroke must be monochromatic.
    """
    t = target.reshape(-1)
    b = canvas.reshape(-1)
    target_bits = {}
    bad_bits = {}
    mismatch = 0

    for i in range(100):
        r, c = divmod(i, 10)
        if r == c or r + c == 9:
            continue
        bit = 1 << i
        colour = int(t[i])
        target_bits[colour] = target_bits.get(colour, 0) | bit
        if int(b[i]) != colour:
            mismatch |= bit
            bad_bits[colour] = bad_bits.get(colour, 0) | bit

    selectable = set(palette_colours)
    usable = selectable | {start_colour}
    bad_groups = tuple(bad_bits.values())
    number_masks = 12 if allow_small else 8

    # State: (union of later stroke masks, first suffix position, suffix colour)
    initial = (0, -1, -1)
    costs = {initial: 0}
    parents = {}
    serial = 0
    queue = [(len(bad_groups), 0, serial, initial)]

    while queue:
        _, cost, _, state = heapq.heappop(queue)
        if costs.get(state) != cost:
            continue

        covered, suffix_pos, suffix_colour = state
        if suffix_pos >= 0 and not (mismatch & ~covered):
            if suffix_colour == start_colour or suffix_colour in selectable:
                answer = []
                node = state
                while node != initial:
                    previous, mask_id, colour = parents[node]
                    answer.append((mask_id, colour))
                    node = previous
                return answer

        for mask_id in range(number_masks):
            mask = bit_masks[mask_id]
            exposed = mask & ~covered
            if not exposed:
                continue

            first_pixel = (exposed & -exposed).bit_length() - 1
            colour = int(t[first_pixel])
            if colour not in usable:
                continue
            if exposed & ~target_bits[colour]:
                continue

            position = _MASK_POS[mask_id]
            extra = 1
            if suffix_pos >= 0:
                # In forward order this new stroke precedes the suffix.
                if colour != suffix_colour and suffix_colour not in selectable:
                    continue
                extra += _distance(position, suffix_pos)
                extra += int(colour != suffix_colour)

            new_cost = cost + extra
            new_covered = covered | mask
            new_state = (new_covered, position, colour)
            if new_cost >= costs.get(new_state, float("inf")):
                continue

            costs[new_state] = new_cost
            parents[new_state] = (state, mask_id, colour)

            # Movement/selection connection plus a lower bound on extra paints.
            heuristic = _distance(start_pos, position)
            heuristic += int(start_colour != colour)
            heuristic += sum(bool(bits & ~new_covered) for bits in bad_groups)

            serial += 1
            heapq.heappush(
                queue,
                (new_cost + heuristic, new_cost, serial, new_state)
            )

    return None


def _move_towards(position, destination):
    clockwise = (destination - position) % 8
    step = 1 if clockwise <= 4 else -1
    nxt = (position + step) % 8
    r0, c0 = _RING[position]
    r1, c1 = _RING[nxt]
    if r1 < r0:
        return "ACTION1"
    if r1 > r0:
        return "ACTION2"
    if c1 < c0:
        return "ACTION3"
    return "ACTION4"


def act(obs, available, memory):
    a = np.asarray(obs)
    if "_geometry" not in memory:
        memory["_geometry"] = _geometry()
    geometry = memory["_geometry"]
    _, tools, relevant, raster_masks, bit_masks = geometry

    if "_image_positions" not in memory:
        memory["_image_positions"] = _find_images(a)
    (tx, ty), (bx, by) = memory["_image_positions"]

    target = a[ty:ty + 10, tx:tx + 10]
    canvas = a[by:by + 10, bx:bx + 10]
    goal_key = tuple(map(int, target[relevant]))
    board_key = tuple(map(int, canvas[relevant]))

    if memory.get("_goal") != goal_key:
        memory["_goal"] = goal_key
        memory["_plan"] = []
        memory.pop("_has_small", None)

    palette = _palette(a)
    position, colour, observed_small = _read_basket(a, geometry)
    if observed_small is not None:
        memory["_has_small"] = observed_small

    plan = memory.get("_plan", [])
    valid = bool(plan) and memory.get("_expected") == board_key
    if valid and memory.get("_has_small") is False:
        valid = all(mask_id < 8 for mask_id, _ in plan)

    if not valid:
        plan = _plan(
            target, canvas, position, colour, bit_masks,
            memory.get("_has_small") is not False,
            palette.keys()
        )
        memory["_plan"] = plan or []
        memory["_expected"] = board_key

    if not plan:
        # Normally unreachable for a legal puzzle. If tool availability is
        # still unknown, expose a cardinal station to resolve it.
        if position % 2:
            action = _move_towards(position, (position - 1) % 8)
            if action in available:
                return action, None, None
        if "ACTION5" in available:
            return "ACTION5", None, None
        action = next((s for s in available if s != "RESET"), available[0])
        if action == "ACTION6":
            return action, 4, 4
        return action, None, None

    mask_id, wanted_colour = plan[0]
    destination = _MASK_POS[mask_id]

    if position != destination:
        return _move_towards(position, destination), None, None

    if colour != wanted_colour:
        if wanted_colour in palette:
            row, col = palette[wanted_colour]
            return "ACTION6", row, col
        # Discard a stale plan if its required colour is no longer selectable.
        memory["_plan"] = []
        return "ACTION5", None, None

    # Predict the next canvas and consume this stroke. Future observations
    # validate the prediction before any remaining plan is reused.
    predicted = canvas.copy()
    predicted[raster_masks[mask_id]] = wanted_colour
    memory["_expected"] = tuple(map(int, predicted[relevant]))
    memory["_plan"] = plan[1:]

    if mask_id < 8:
        return "ACTION5", None, None

    x, y, template = tools[destination]
    h, w = template.shape
    return "ACTION6", y + h // 2, x + w // 2

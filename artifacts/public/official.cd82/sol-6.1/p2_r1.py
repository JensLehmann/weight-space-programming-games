import numpy as np
from collections import deque
from heapq import heappush, heappop


# Basket positions and rotations from the game's movement mechanics.
_BASKETS = [
    (25, 24, 180),
    (33, 21, 0),
    (38, 32, 270),
    (33, 40, 90),
    (25, 45, 0),
    (14, 40, 180),
    (17, 32, 90),
    (14, 21, 270),
]

_RING_COORDS = [
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
]

_PATCH_LAYOUT = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}


def _rotate(a, rotation):
    return np.rot90(a, -(rotation // 90))


def _make_templates():
    horizontal = np.full((9, 14), -1, dtype=int)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[1:8, 1:13] = 15
    horizontal[8, :] = 2

    diagonal = np.full((17, 17), -1, dtype=int)
    diagonal[0, 6] = 2
    for r in range(1, 7):
        left, right = 6 - r, 6 + r
        diagonal[r, left:left + 2] = 2
        diagonal[r, right - 1:right + 1] = 2
        diagonal[r, left + 2:right - 1] = 15
    for r in range(7, 15):
        left = r - 5
        right = r + 6 if r <= 10 else 26 - r
        diagonal[r, left:right - 1] = 15
        diagonal[r, right - 1:right + 1] = 2
    diagonal[15, 10:12] = 2
    diagonal[16, 10] = 2

    patch = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ], dtype=int)

    baskets = []
    for i, (_, _, rotation) in enumerate(_BASKETS):
        base = horizontal if i % 2 == 0 else diagonal
        baskets.append(_rotate(base, rotation))

    patches = {
        p: _rotate(patch, layout[2])
        for p, layout in _PATCH_LAYOUT.items()
    }
    return baskets, patches


_BASKET_TEMPLATES, _PATCH_TEMPLATES = _make_templates()


def _make_routes():
    lookup = {rc: i for i, rc in enumerate(_RING_COORDS)}
    moves = [
        ("ACTION1", -1, 0),
        ("ACTION2", 1, 0),
        ("ACTION3", 0, -1),
        ("ACTION4", 0, 1),
    ]
    routes = {}
    for start in range(8):
        q = deque([(start, ())])
        seen = {start}
        while q:
            node, path = q.popleft()
            routes[start, node] = path
            r, c = _RING_COORDS[node]
            for action, dr, dc in moves:
                nr = min(2, max(0, r + dr))
                nc = min(2, max(0, c + dc))
                nxt = lookup.get((nr, nc))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + (action,)))
    return routes


_ROUTES = _make_routes()
_DIST = {
    pair: len(path) for pair, path in _ROUTES.items()
}

# The game's goal ignores both diagonals.
_CELLS = [
    (r, c) for r in range(10) for c in range(10)
    if r != c and r + c != 9
]
_FULL = (1 << len(_CELLS)) - 1


def _make_paint_masks():
    predicates = [
        lambda r, c: r < 5,                 # North half
        lambda r, c: c >= r,                # Northeast triangle
        lambda r, c: c >= 5,                # East half
        lambda r, c: c >= 9 - r,            # Southeast triangle
        lambda r, c: r >= 5,                # South half
        lambda r, c: c <= r,                # Southwest triangle
        lambda r, c: c < 5,                 # West half
        lambda r, c: c <= 9 - r,            # Northwest triangle
        lambda r, c: r < 3 and 3 <= c < 7,  # North patch
        lambda r, c: 3 <= r < 7 and c >= 7, # East patch
        lambda r, c: r >= 7 and 3 <= c < 7, # South patch
        lambda r, c: 3 <= r < 7 and c < 3,  # West patch
    ]
    masks = []
    for predicate in predicates:
        mask = 0
        for k, (r, c) in enumerate(_CELLS):
            if predicate(r, c):
                mask |= 1 << k
        masks.append(mask)
    return masks


_PAINT_MASKS = _make_paint_masks()


def _read_palette(obs):
    palette = {}
    for y in range(0, 13):
        for x in range(18, 60):
            block = obs[y:y + 5, x:x + 5]
            if block.shape != (5, 5):
                continue
            if not (
                np.all(block[0, :] == 4)
                and np.all(block[4, :] == 4)
                and np.all(block[:, 0] == 4)
                and np.all(block[:, 4] == 4)
            ):
                continue
            color = int(block[2, 2])
            if np.all(block[1:4, 1:4] == color):
                palette[color] = (y + 2, x + 2)
    return palette


def _read_basket(obs):
    best_score = -1.0
    best_position = 0
    best_color = 15

    for position, template in enumerate(_BASKET_TEMPLATES):
        x, y, _ = _BASKETS[position]
        h, w = template.shape
        region = obs[y:y + h, x:x + w]
        border = template == 2
        interior = template == 15

        border_score = float(np.mean(region[border] == 2))
        values = region[interior].astype(int)
        counts = np.bincount(values, minlength=16)
        color = int(np.argmax(counts))
        uniformity = float(counts[color]) / max(1, len(values))
        score = border_score + 0.15 * uniformity

        if score > best_score:
            best_score = score
            best_position = position
            best_color = color

    return best_position, best_color


def _has_patch(obs, position):
    if position not in _PATCH_LAYOUT:
        return False
    x, y, _ = _PATCH_LAYOUT[position]
    template = _PATCH_TEMPLATES[position]
    h, w = template.shape
    region = obs[y:y + h, x:x + w]
    return bool(
        np.all(region[template == 2] == 2)
        and np.all(region[template == 0] == 0)
    )


def _find_square(obs, allowed, default, x_bounds, y_bounds):
    """Locate an unframed 10x10 picture, preferring the canonical location."""
    dy, dx = default
    eligible = np.isin(obs, list(allowed))

    if np.all(eligible[dy:dy + 10, dx:dx + 10]):
        return dy, dx

    integral = np.pad(
        eligible.astype(int), ((1, 0), (1, 0))
    ).cumsum(0).cumsum(1)

    best = None
    best_score = None
    for y in range(y_bounds[0], y_bounds[1] + 1):
        for x in range(x_bounds[0], x_bounds[1] + 1):
            total = (
                integral[y + 10, x + 10]
                - integral[y, x + 10]
                - integral[y + 10, x]
                + integral[y, x]
            )
            if total != 100:
                continue

            surround = 0
            if y > 0:
                surround += int(np.sum(obs[y - 1, x:x + 10] == 5))
            if y + 10 < 64:
                surround += int(np.sum(obs[y + 10, x:x + 10] == 5))
            if x > 0:
                surround += int(np.sum(obs[y:y + 10, x - 1] == 5))
            if x + 10 < 64:
                surround += int(np.sum(obs[y:y + 10, x + 10] == 5))

            score = (surround, -abs(y - dy) - abs(x - dx))
            if best_score is None or score > best_score:
                best_score = score
                best = (y, x)

    return best if best is not None else default


def _solve(target, canvas, start_position, start_color, palette, has_patch):
    """
    Reverse-layer A*: a paint can be the last remaining layer only when
    every still-visible target pixel in its mask has the same colour.
    """
    values = [int(target[r, c]) for r, c in _CELLS]
    color_masks = {}
    wrong = 0

    for k, (r, c) in enumerate(_CELLS):
        color = values[k]
        color_masks[color] = color_masks.get(color, 0) | (1 << k)
        if int(canvas[r, c]) != color:
            wrong |= 1 << k

    masks = _PAINT_MASKS if has_patch else _PAINT_MASKS[:8]
    allowed_colors = set(palette)
    transition_cache = {}

    def transitions(remaining):
        cached = transition_cache.get(remaining)
        if cached is not None:
            return cached

        result = []
        for index, mask in enumerate(masks):
            visible = remaining & mask
            if not visible:
                continue
            bit_index = (visible & -visible).bit_length() - 1
            color = values[bit_index]
            if color not in allowed_colors:
                continue
            if visible & color_masks[color] != visible:
                continue

            position = index if index < 8 else 2 * (index - 8)
            result.append((
                remaining & ~mask, position, color, index
            ))
        transition_cache[remaining] = result
        return result

    def heuristic(remaining, position, color):
        return (
            _DIST[start_position, position]
            + int(start_color != color)
            + int(bool(remaining & wrong))
        )

    root = (_FULL, -1, -1)
    costs = {root: 0}
    parent = {}
    heap = [(1, 0, root)]

    while heap:
        _, cost, key = heappop(heap)
        if costs.get(key) != cost:
            continue

        remaining, next_position, next_color = key
        if next_position >= 0 and not (remaining & wrong):
            operations = []
            while key != root:
                previous, index, color = parent[key]
                operations.append((index, color))
                key = previous
            # Following parents runs from earliest paint to latest paint.
            return operations

        for new_remaining, position, color, index in transitions(remaining):
            extra = 1
            if next_position >= 0:
                extra += _DIST[position, next_position]
                extra += int(color != next_color)

            new_cost = cost + extra
            new_key = (new_remaining, position, color)
            if new_cost >= costs.get(new_key, float("inf")):
                continue

            costs[new_key] = new_cost
            parent[new_key] = (key, index, color)
            estimate = new_cost + heuristic(new_remaining, position, color)
            heappush(heap, (estimate, new_cost, new_key))

    return None


def _compile_actions(operations, position, color, palette):
    actions = []
    for index, wanted_color in operations:
        wanted_position = index if index < 8 else 2 * (index - 8)

        for action in _ROUTES[position, wanted_position]:
            actions.append((action, None, None))
        position = wanted_position

        if color != wanted_color:
            row, col = palette[wanted_color]
            actions.append(("ACTION6", row, col))
            color = wanted_color

        if index < 8:
            actions.append(("ACTION5", None, None))
        else:
            x, y, _ = _PATCH_LAYOUT[position]
            template = _PATCH_TEMPLATES[position]
            h, w = template.shape
            actions.append(("ACTION6", y + h // 2, x + w // 2))

    return actions


def act(obs, available, memory):
    queued = memory.get("_cd82_actions")
    index = memory.get("_cd82_index", 0)

    if queued is not None and index < len(queued):
        memory["_cd82_index"] = index + 1
        action, row, col = queued[index]
        return action, row, col

    memory.pop("_cd82_actions", None)
    memory.pop("_cd82_index", None)

    obs = np.asarray(obs)
    palette = _read_palette(obs)
    position, color = _read_basket(obs)

    # A diagonal position hides the optional small painter. Move to a
    # neighbouring cardinal position to observe whether it exists.
    if "_cd82_has_patch" not in memory:
        if position % 2:
            neighbors = [(position - 1) % 8, (position + 1) % 8]
            destination = min(neighbors, key=lambda p: _DIST[p, 0])
            action = _ROUTES[position, destination][0]
            return action, None, None
        memory["_cd82_has_patch"] = _has_patch(obs, position)

    has_patch = bool(memory["_cd82_has_patch"])
    allowed_colors = set(palette) | {0, 15}

    ty, tx = _find_square(
        obs, allowed_colors, (3, 3), (0, 6), (0, 6)
    )
    cy, cx = _find_square(
        obs, allowed_colors, (34, 27), (18, 54), (18, 53)
    )

    target = obs[ty:ty + 10, tx:tx + 10]
    canvas = obs[cy:cy + 10, cx:cx + 10]

    operations = _solve(
        target, canvas, position, color, palette, has_patch
    )

    if operations is not None:
        actions = _compile_actions(
            operations, position, color, palette
        )
        if actions:
            memory["_cd82_actions"] = actions
            memory["_cd82_index"] = 1
            return actions[0]

    # Recovery for an unexpectedly unreadable observation.
    memory.clear()
    if "RESET" in available:
        return "RESET", None, None
    return "ACTION5", None, None

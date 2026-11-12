import heapq
import numpy as np


# Brush positions around the canvas, clockwise from the top.
_POSITIONS = [
    (0, 1), (0, 2), (1, 2), (2, 2),
    (2, 1), (2, 0), (1, 0), (0, 0),
]
_BASKETS = [
    (25, 24, 180), (33, 21, 0), (38, 32, 270), (33, 40, 90),
    (25, 45, 0), (14, 40, 180), (17, 32, 90), (14, 21, 270),
]
_GADGETS = {
    0: (29, 18, 180),
    2: (48, 36, 270),
    4: (29, 55, 0),
    6: (11, 36, 90),
}


def _rot(a, degrees):
    return np.rot90(a, -(degrees // 90))


def _basket_position(obs):
    horizontal = np.full((9, 14), -1, dtype=int)
    horizontal[:, 0] = 2
    horizontal[:, 13] = 2
    horizontal[8, :] = 2

    diagonal = np.full((17, 17), -1, dtype=int)
    red_x = [
        [6], [5, 6, 7], [4, 5, 7, 8], [3, 4, 8, 9],
        [2, 3, 9, 10], [1, 2, 10, 11], [0, 1, 11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16],
        [14, 15], [13, 14], [12, 13], [11, 12],
        [10, 11], [10],
    ]
    for y, xs in enumerate(red_x):
        for x in xs:
            diagonal[y, x] = 2

    best = None
    best_score = (-1.0, -1)
    for p, (x, y, rotation) in enumerate(_BASKETS):
        template = _rot(horizontal if p % 2 == 0 else diagonal, rotation)
        h, w = template.shape
        mask = template == 2
        total = int(mask.sum())
        matches = int(np.count_nonzero(obs[y:y+h, x:x+w][mask] == 2))
        score = (matches / total, matches)
        if score > best_score:
            best_score = score
            best = p
    return best if best_score[0] >= 0.9 else None


def _has_gadget(obs, position):
    if position not in _GADGETS:
        return False
    gadget = np.array([
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 15, 15, 15, 15, 2],
        [2, 2, 0, 0, 2, 2],
        [-1, 2, 2, 2, 2, -1],
    ])
    x, y, rotation = _GADGETS[position]
    gadget = _rot(gadget, rotation)
    h, w = gadget.shape
    fixed = (gadget == 2) | (gadget == 0)
    return bool(np.all(obs[y:y+h, x:x+w][fixed] == gadget[fixed]))


def _palette(obs):
    colors = {}
    selected = None
    for x in range(18, 60):
        if not np.all(obs[2, x:x+5] == 4):
            continue
        if not np.all(obs[6, x:x+5] == 4):
            continue
        if not np.all(obs[2:7, x] == 4):
            continue
        if not np.all(obs[2:7, x+4] == 4):
            continue
        color = int(obs[4, x+2])
        if not np.all(obs[3:6, x+1:x+4] == color):
            continue
        colors[color] = (4, x + 2)
        if np.all(obs[7, x:x+5] == 0):
            selected = color
    return colors, selected


def _stroke_masks(target, canvas, have_gadget):
    # The game ignores both diagonals when checking the result.
    cells = [(i, j) for i in range(10) for j in range(10)
             if i != j and i + j != 9]
    count = 12 if have_gadget else 8
    masks = [0] * count
    target_colors = []
    need = 0

    for bit, (i, j) in enumerate(cells):
        b = 1 << bit
        c = int(target[i, j])
        target_colors.append(c)
        if int(canvas[i, j]) != c:
            need |= b

        tests = [
            i < 5, j >= i, j >= 5, j >= 9 - i,
            i >= 5, j <= i, j < 5, j <= 9 - i,
        ]
        if have_gadget:
            tests += [
                i < 3 and 3 <= j < 7,
                3 <= i < 7 and j >= 7,
                i >= 7 and 3 <= j < 7,
                3 <= i < 7 and j < 3,
            ]
        for k, yes in enumerate(tests):
            if yes:
                masks[k] |= b

    color_bits = {}
    for bit, color in enumerate(target_colors):
        color_bits[color] = color_bits.get(color, 0) | (1 << bit)
    return masks, target_colors, color_bits, need


def _direction(stroke):
    return stroke if stroke < 8 else (0, 2, 4, 6)[stroke - 8]


def _distance(a, b):
    d = abs(a - b)
    return min(d, 8 - d)


def _plan(target, canvas, position, selected, palette, have_gadget):
    masks, cell_colors, color_bits, need = _stroke_masks(
        target, canvas, have_gadget
    )
    n = len(masks)
    size = 1 << n

    # covered[s] is the set of pixels painted by strokes in subset s.
    covered = [0] * size
    for s in range(1, size):
        low = s & -s
        k = low.bit_length() - 1
        covered[s] = covered[s ^ low] | masks[k]

    def stroke_color(s, k):
        # Color of k when it was added as the earliest reverse stroke.
        remaining = masks[k] & ~covered[s ^ (1 << k)]
        bit = (remaining & -remaining).bit_length() - 1
        return cell_colors[bit]

    candidates_cache = {}

    def candidates(s):
        if s in candidates_cache:
            return candidates_cache[s]
        result = []
        for k in range(n):
            if s & (1 << k):
                continue
            remaining = masks[k] & ~covered[s]
            if not remaining:
                continue
            bit = (remaining & -remaining).bit_length() - 1
            color = cell_colors[bit]
            if color not in palette:
                continue
            if remaining & ~color_bits[color]:
                continue
            result.append((k, color))
        candidates_cache[s] = result
        return result

    # Reverse Dijkstra: each added stroke precedes all strokes already chosen.
    # The state records the subset and the earliest stroke in that subset.
    root = (0, -1)
    dist = {root: 0}
    parent = {}
    heap = [(0, 0, -1)]
    best_cost = float("inf")
    best_node = None

    while heap:
        cost, s, last = heapq.heappop(heap)
        node = (s, last)
        if dist.get(node) != cost:
            continue
        if cost >= best_cost:
            break

        if last >= 0:
            last_color = stroke_color(s, last)
            if not (need & ~covered[s]):
                total = (
                    cost
                    + _distance(position, _direction(last))
                    + (selected != last_color)
                )
                if total < best_cost:
                    best_cost = total
                    best_node = node
        else:
            last_color = None

        for k, color in candidates(s):
            extra = 1
            if last >= 0:
                extra += _distance(_direction(k), _direction(last))
                extra += (color != last_color)
            new_cost = cost + extra
            if new_cost >= best_cost:
                continue
            new_s = s | (1 << k)
            new_node = (new_s, k)
            if new_cost < dist.get(new_node, float("inf")):
                dist[new_node] = new_cost
                parent[new_node] = node
                heapq.heappush(heap, (new_cost, new_s, k))

    if best_node is None:
        return None

    strokes = []
    node = best_node
    while node != root:
        s, k = node
        strokes.append((k, stroke_color(s, k)))
        node = parent[node]

    actions = []
    current_pos = position
    current_color = selected

    for k, color in strokes:
        destination = _direction(k)
        clockwise = (destination - current_pos) % 8
        counterclockwise = (current_pos - destination) % 8
        step = 1 if clockwise <= counterclockwise else -1

        while current_pos != destination:
            next_pos = (current_pos + step) % 8
            r1, c1 = _POSITIONS[current_pos]
            r2, c2 = _POSITIONS[next_pos]
            if r2 < r1:
                action = "ACTION1"
            elif r2 > r1:
                action = "ACTION2"
            elif c2 < c1:
                action = "ACTION3"
            else:
                action = "ACTION4"
            actions.append((action, None, None))
            current_pos = next_pos

        if current_color != color:
            row, col = palette[color]
            actions.append(("ACTION6", row, col))
            current_color = color

        if k < 8:
            actions.append(("ACTION5", None, None))
        else:
            x, y, rotation = _GADGETS[destination]
            if rotation in (90, 270):
                width, height = 5, 6
            else:
                width, height = 6, 5
            actions.append(("ACTION6", y + height // 2, x + width // 2))

    return actions


def act(obs, available, memory):
    queue = memory.get("queue")
    if queue:
        return tuple(queue.pop(0))

    position = _basket_position(obs)
    if position is None:
        return ("RESET", None, None)

    # Diagonal brushes can obscure part of the canvas. Move once to reveal it.
    if position % 2:
        reveal = {
            1: "ACTION3",
            3: "ACTION1",
            5: "ACTION4",
            7: "ACTION2",
        }
        return (reveal[position], None, None)

    palette, selected = _palette(obs)
    if not palette or selected is None:
        return ("RESET", None, None)

    target = obs[3:13, 3:13]
    canvas = obs[34:44, 27:37]
    have_gadget = _has_gadget(obs, position)
    actions = _plan(
        target, canvas, position, selected, palette, have_gadget
    )

    if not actions:
        return ("RESET", None, None)

    memory["queue"] = actions[1:]
    return actions[0]

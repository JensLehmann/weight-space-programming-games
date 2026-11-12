import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def node(kind, *inputs, name=None, body=None, **fields):
        ident = fresh() if name is None else name
        nd = {"id": ident, "op": kind, "in": list(inputs)}
        nd.update(fields)
        (graph if body is None else body).append(nd)
        return ident

    def linear(x, w, b=None, spatial=False, name=None, body=None):
        ident = fresh() if name is None else name
        w = np.asarray(w, dtype=np.float32)
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        wk = ident + "_w"
        params[wk] = w
        fields = {"w": wk}
        if b is not None:
            bk = ident + "_b"
            params[bk] = np.asarray(b, dtype=np.float32)
            fields["b"] = bk
        return node(
            "conv" if spatial else "dense",
            x, name=ident, body=body, **fields
        )

    def complement(x, n=1, body=None):
        return linear(
            x, -np.eye(n, dtype=np.float32),
            np.ones(n, dtype=np.float32), body=body
        )

    one = linear("avail", np.zeros((1, 8)), [1])

    # Numeric colour image and useful individual colour planes.
    colour = linear("x0", np.arange(16).reshape(1, 16), spatial=True)
    reverse_colour = linear(colour, [[-1]], [15], spatial=True)
    red = node("slice", "x0", start=2, stop=3)
    border_colour = node("slice", "x0", start=4, stop=5)

    # Align the canonical canvas with the target:
    # target (3,3), canvas (27,34), in (x,y) coordinates.
    aligned_canvas = colour
    shifts = [(3, 3)] * 8 + [(3, 0)] * 2 + [(1, 0)]
    for dy, dx in shifts:
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        w[0, 0, 3 + dy, 3 + dx] = 1
        aligned_canvas = linear(aligned_canvas, w, spatial=True)

    difference = node("sub", colour, aligned_canvas)
    difference = node("abs", difference)
    mismatch = node("min2", difference, one)

    # Fixed geometric predicates a*row + b*column + d >= 0.
    # A margin and clipping make their outputs exactly binary.
    bounds = [
        (1, 0, -3), (-1, 0, 12),
        (0, 1, -3), (0, -1, 12),
    ]
    off_diagonals = [
        (1, -1, -1), (-1, 1, -1),
        (1, 1, -16), (-1, -1, 14),
    ]
    extra = [
        [(-1, 0, 7)],                         # top half
        [(-1, 1, 0)],                         # NE triangle
        [(0, 1, -8)],                         # right half
        [(1, 1, -15)],                        # SE triangle
        [(1, 0, -8)],                         # bottom half
        [(1, -1, 0)],                         # SW triangle
        [(0, -1, 7)],                         # left half
        [(-1, -1, 15)],                       # NW triangle
        [(-1, 0, 5), (0, 1, -6), (0, -1, 9)],
        [(1, 0, -6), (-1, 0, 9), (0, 1, -10)],
        [(1, 0, -10), (0, 1, -6), (0, -1, 9)],
        [(1, 0, -6), (-1, 0, 9), (0, -1, 5)],
    ]

    predicates = []
    predicate_index = {}
    for p in bounds + off_diagonals + [p for es in extra for p in es]:
        if p not in predicate_index:
            predicate_index[p] = len(predicates)
            predicates.append(p)

    pw = np.zeros((len(predicates), 3), dtype=np.float32)
    pb = np.zeros(len(predicates), dtype=np.float32)
    for k, (a, b, d) in enumerate(predicates):
        pw[k, 0] = 252 * a
        pw[k, 1] = 252 * b
        pb[k] = 4 * d + 2

    predicates_image = linear("coords", pw, pb, spatial=True)
    predicates_image = node("relu", predicates_image)
    predicates_image = node("min2", predicates_image, one)

    # Channel 0 is the 80-pixel comparison domain.
    # Channels 1..12 are the brush masks on that domain.
    mw = np.zeros((13, len(predicates)), dtype=np.float32)
    mb = np.zeros(13, dtype=np.float32)
    for k in range(13):
        constraints = [] if k == 0 else extra[k - 1]
        for p in bounds + off_diagonals + constraints:
            mw[k, predicate_index[p]] += 1
        # Four bounds and two off-diagonal ORs contribute six.
        mb[k] = -5 - len(constraints)

    masks = linear(predicates_image, mw, mb, spatial=True)
    masks = node("relu", masks)
    domain = node("slice", masks, start=0, stop=1)
    brushes = node("slice", masks, start=1, stop=13)

    # Pixel gates for basket orientation, its fill colour,
    # small-brush availability, and small-brush click centres.
    direction_points = [
        (24, 31), (21, 39), (38, 46), (46, 49),
        (53, 31), (56, 24), (38, 17), (31, 14),
    ]
    fill_points = [
        (28, 31), (29, 41), (38, 42), (48, 41),
        (49, 31), (48, 22), (38, 21), (29, 22),
    ]
    small_border_points = [
        (18, 31), (41, 48), (55, 29), (36, 15),
    ]
    small_click_points = [
        (20, 32), (39, 50), (57, 32), (39, 13),
    ]
    points = (
        direction_points + fill_points
        + small_border_points + small_click_points
    )

    qw = np.zeros((2 * len(points), 3), dtype=np.float32)
    qb = np.zeros(2 * len(points), dtype=np.float32)
    for k, (row, col) in enumerate(points):
        qw[2 * k, 0] = 63
        qw[2 * k + 1, 1] = 63
        qb[2 * k] = -row
        qb[2 * k + 1] = -col

    distances = linear("coords", qw, qb, spatial=True)
    distances = node("abs", distances)
    gw = np.zeros((len(points), 2 * len(points)), dtype=np.float32)
    for k in range(len(points)):
        gw[k, 2 * k:2 * k + 2] = -4
    point_gates = linear(
        distances, gw, np.full(len(points), 2), spatial=True
    )
    point_gates = node("relu", point_gates)
    point_gates = node("min2", point_gates, one)

    direction_gates = node("slice", point_gates, start=0, stop=8)
    fill_gates = node("slice", point_gates, start=8, stop=16)
    small_border_gates = node("slice", point_gates, start=16, stop=20)
    small_click_gates = node("slice", point_gates, start=20, stop=24)

    direction = node("gmax", node("mul", direction_gates, red))
    fills = node("gmax", node("mul", fill_gates, colour))
    current_colour = linear(
        node("vmul", direction, fills), np.ones((1, 8))
    )

    small_visible = node(
        "gmax", node("mul", small_border_gates, red)
    )
    cardinal_w = np.zeros((4, 8), dtype=np.float32)
    cardinal_w[np.arange(4), [0, 2, 4, 6]] = 1
    cardinal_direction = linear(direction, cardinal_w)
    is_cardinal = linear(cardinal_direction, np.ones((1, 4)))
    brush_present = linear(
        node("vmul", cardinal_direction, small_visible),
        np.ones((1, 4))
    )
    old_memory = node("vmul", "mem", complement(is_cardinal))
    node("vadd", brush_present, old_memory, name="mem_next")

    enable_w = np.zeros((12, 1), dtype=np.float32)
    enable_w[8:, 0] = 1
    enable_b = np.r_[np.ones(8), np.zeros(4)]
    enabled_brushes = linear("mem_next", enable_w, enable_b)

    # Reverse painting. The loop peels one feasible stencil per pass.
    # Its carried instruction is the deepest still-required paint.
    body = []
    rem = "peel_remaining"
    chosen = "peel_chosen"
    chosen_colour = "peel_colour"

    region = node("mul", brushes, rem, body=body)
    maximum = node(
        "gmax", node("mul", region, colour, body=body), body=body
    )
    reverse_maximum = node(
        "gmax",
        node("mul", region, reverse_colour, body=body),
        body=body
    )
    populated = node("gmax", region, body=body)

    total_extrema = node("vadd", maximum, reverse_maximum, body=body)
    uniform = linear(
        total_extrema, -np.eye(12), np.full(12, 16), body=body
    )
    uniform = node("vrelu", uniform, body=body)
    feasible = node("vmul", uniform, populated, body=body)
    feasible = node("vmul", feasible, enabled_brushes, body=body)

    area = node("gmean", region, body=body)
    scores = linear(
        area, np.eye(12) * 409600,
        np.arange(12, 0, -1), body=body
    )
    scores = node("vmul", scores, feasible, body=body)

    # Exact winner selection for integer-valued scores.
    compare_w = np.zeros((144, 12), dtype=np.float32)
    reduce_w = np.zeros((12, 144), dtype=np.float32)
    for i in range(12):
        for j in range(12):
            k = 12 * i + j
            compare_w[k, j] += 1
            compare_w[k, i] -= 1
            reduce_w[i, k] = -1

    comparisons = linear(scores, compare_w, body=body)
    comparisons = node("vrelu", comparisons, body=body)
    winner = linear(comparisons, reduce_w, np.ones(12), body=body)
    winner = node("vrelu", winner, body=body)
    winner = node("vmul", winner, feasible, body=body)

    required = node(
        "gmax", node("mul", rem, mismatch, body=body), body=body
    )
    selection = node("vmul", winner, required, body=body)
    selected_any = linear(selection, np.ones((1, 12)), body=body)
    keep_previous = complement(selected_any, body=body)

    next_chosen = node(
        "vadd", selection,
        node("vmul", chosen, keep_previous, body=body),
        body=body
    )
    new_colour = linear(
        node("vmul", selection, maximum, body=body),
        np.ones((1, 12)), body=body
    )
    next_colour = node(
        "vadd", new_colour,
        node("vmul", chosen_colour, keep_previous, body=body),
        body=body
    )

    removed = linear(
        node("mul", brushes, selection, body=body),
        np.ones((1, 12)), spatial=True, body=body
    )
    next_remaining = node(
        "relu", node("sub", rem, removed, body=body), body=body
    )

    graph.append({
        "id": "peel",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": rem, "init": domain},
            {"id": chosen, "zeros": [12]},
            {"id": chosen_colour, "zeros": [1]},
        ],
        "body": body,
        "next": {
            rem: next_remaining,
            chosen: next_chosen,
            chosen_colour: next_colour,
        },
    })

    plan = "peel." + chosen
    paint_colour = "peel." + chosen_colour
    residual = node(
        "gmax", node("mul", "peel." + rem, mismatch)
    )
    solved_plan = complement(residual)

    orientation_w = np.zeros((8, 12), dtype=np.float32)
    orientation_w[:, :8] = np.eye(8)
    for k, d in enumerate([0, 2, 4, 6]):
        orientation_w[d, 8 + k] = 1
    desired_direction = linear(plan, orientation_w)
    desired_direction = node("vmul", desired_direction, solved_plan)

    # If capability is unknown and peeling is blocked, visit the
    # adjacent cardinal position to reveal the small brush.
    fallback_w = np.zeros((8, 8), dtype=np.float32)
    for d in range(8):
        fallback_w[2 * (d // 2), d] = 1
    fallback = node("vmul", linear(direction, fallback_w), residual)
    desired_direction = node("vadd", desired_direction, fallback)

    # Pair current and desired ring positions.
    pair_w = np.zeros((64, 16), dtype=np.float32)
    navigation_w = np.zeros((4, 64), dtype=np.float32)
    at_goal_w = np.zeros((1, 64), dtype=np.float32)
    ring = [
        (0, 1), (0, 2), (1, 2), (2, 2),
        (2, 1), (2, 0), (1, 0), (0, 0),
    ]
    for current in range(8):
        for target in range(8):
            k = 8 * current + target
            pair_w[k, current] = 1
            pair_w[k, 8 + target] = 1
            if current == target:
                at_goal_w[0, k] = 1
            else:
                turn = 1 if (target - current) % 8 <= 4 else -1
                neighbour = (current + turn) % 8
                r0, c0 = ring[current]
                r1, c1 = ring[neighbour]
                if r1 < r0:
                    action = 1
                elif r1 > r0:
                    action = 2
                elif c1 < c0:
                    action = 3
                else:
                    action = 4
                navigation_w[action - 1, k] = 1

    pairs = linear(
        node("vconcat", direction, desired_direction),
        pair_w, -np.ones(64)
    )
    pairs = node("vrelu", pairs)
    navigation = linear(pairs, navigation_w)
    at_goal = linear(pairs, at_goal_w)

    small_w = np.zeros((1, 12), dtype=np.float32)
    small_w[0, 8:] = 1
    use_small = linear(plan, small_w)
    small_paint = node("vmul", at_goal, use_small)
    small_paint = node("vmul", small_paint, solved_plan)

    # Exact scalar inequality test for selected versus desired colour.
    colour_delta = linear(
        node("vconcat", current_colour, paint_colour),
        [[1, -1], [-1, 1]]
    )
    colour_delta = node("vrelu", colour_delta)
    colour_distance = linear(colour_delta, [[1, 1]])
    same_colour = node("vrelu", complement(colour_distance))
    change_colour = complement(same_colour)
    change_colour = node("vmul", change_colour, solved_plan)
    no_colour_change = complement(change_colour)

    controls = node(
        "vconcat", navigation, at_goal, use_small,
        residual, change_colour, small_paint
    )
    output_w = np.zeros((8, 9), dtype=np.float32)
    output_b = np.zeros(8, dtype=np.float32)
    output_b[0] = -10
    output_b[7] = -10
    for a in range(1, 5):
        output_w[a, a - 1] = 1
    output_w[5, 4] = 1
    output_w[5, 5] = -1
    output_w[5, 6] = -1
    output_w[6, 7] = 2
    output_w[6, 8] = 1
    linear(controls, output_w, output_b, name="logits")

    # Palette centres are recognized by their complete 5x5 border.
    ring_kernel = np.zeros((1, 1, 5, 5), dtype=np.float32)
    ring_kernel[0, 0, 0, :] = 1
    ring_kernel[0, 0, 4, :] = 1
    ring_kernel[0, 0, :, 0] = 1
    ring_kernel[0, 0, :, 4] = 1
    palette_centres = linear(
        border_colour, ring_kernel, [-15], spatial=True
    )
    palette_centres = node("relu", palette_centres)

    palette_error = node("abs", node("sub", colour, paint_colour))
    palette_click = node(
        "sub",
        linear(palette_centres, [[100]], spatial=True),
        palette_error
    )

    small_selection = node("vslice", plan, start=8, stop=12)
    brush_click = linear(
        node("mul", small_click_gates, small_selection),
        np.full((1, 4), 100), spatial=True
    )
    node(
        "add",
        node("mul", palette_click, change_colour),
        node("mul", brush_click, no_colour_change),
        name="click"
    )

    return {
        "graph": graph,
        "memory_size": 1,
        "params": params,
    }

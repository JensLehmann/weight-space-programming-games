import numpy as np


def build():
    params = {}
    graph = []
    G = graph
    serial = 0
    pserial = 0

    def node(op, ins=None, ident=None, **fields):
        nonlocal serial
        if ident is None:
            ident = "n%d" % serial
            serial += 1
        nd = {"id": ident, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(fields)
        G.append(nd)
        return ident

    def parameter(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def conv(x, w, b=None, ident=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], ident, **fields)

    def dense(x, w, b=None, ident=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], ident, **fields)

    def sl(x, start, stop, vector=False):
        return node(
            "vslice" if vector else "slice",
            [x], start=start, stop=stop
        )

    def const(value):
        return dense("aprev", np.zeros((1, 8)), [value])

    one = const(1.0)
    zero = const(0.0)

    # ------------------------------------------------------------
    # Coordinate predicates and exact spatial masks.
    #
    # A predicate (a,b,d) means a*row + b*column + d >= 0.
    # Two ReLUs implement a saturated step with a half-pixel margin.
    # A mask is an AND of groups. Each group is either one predicate
    # or two mutually exclusive predicates implementing an OR.
    # ------------------------------------------------------------
    predicates = []
    pred_index = {}

    def pred(t):
        t = tuple(t)
        if t not in pred_index:
            pred_index[t] = len(predicates)
            predicates.append(t)
        return pred_index[t]

    def rect(r0, r1, c0, c1):
        return [
            [(1, 0, -r0)],
            [(-1, 0, r1)],
            [(0, 1, -c0)],
            [(0, -1, c1)],
        ]

    offdiag = rect(3, 12, 3, 12) + [
        [(1, -1, -1), (-1, 1, -1)],
        [(1, 1, -16), (-1, -1, 14)],
    ]

    descriptions = []
    labels = {}

    def add_mask(name, groups):
        labels[name] = len(descriptions)
        descriptions.append(groups)
        for group in groups:
            for t in group:
                pred(t)

    # The eight large painting regions, in simulator orientation order.
    large_conditions = [
        (-1, 0, 7),       # top half
        (-1, 1, 0),       # column >= row
        (0, 1, -8),       # right half
        (1, 1, -15),      # lower-right anti-diagonal triangle
        (1, 0, -8),       # bottom half
        (1, -1, 0),       # row >= column
        (0, -1, 7),       # left half
        (-1, -1, 15),     # upper-left anti-diagonal triangle
    ]
    for i, t in enumerate(large_conditions):
        add_mask("region%d" % i, offdiag + [[t]])

    # Small-tool regions: top, right, bottom, left.
    small_rects = [
        (3, 5, 6, 9),
        (6, 9, 10, 12),
        (10, 12, 6, 9),
        (6, 9, 3, 5),
    ]
    for i, bounds in enumerate(small_rects):
        add_mask("region%d" % (8 + i), offdiag + rect(*bounds))

    # Each anchor is red in exactly one resting basket orientation.
    basket_anchors = [
        (24, 32), (21, 39), (39, 46), (46, 49),
        (53, 32), (56, 24), (39, 17), (31, 14),
    ]
    for i, (r, c) in enumerate(basket_anchors):
        add_mask("basket%d" % i, rect(r, r, c, c))

    small_red_anchors = [
        (18, 31), (38, 52), (55, 29), (39, 11),
    ]
    for i, (r, c) in enumerate(small_red_anchors):
        add_mask("tool_present%d" % i, rect(r, r, c, c))

    small_clicks = [
        (20, 32), (39, 50), (57, 32), (39, 13),
    ]
    for i, (r, c) in enumerate(small_clicks):
        add_mask("tool_click%d" % i, rect(r, r, c, c))

    add_mask("palette_row", rect(4, 4, 18, 62))
    add_mask("constraints", offdiag)

    N = len(predicates)
    W = np.zeros((2 * N, 3), dtype=np.float32)
    B = np.zeros(2 * N, dtype=np.float32)
    for i, (a, b, d) in enumerate(predicates):
        W[i, 0] = W[N + i, 0] = 252.0 * a
        W[i, 1] = W[N + i, 1] = 252.0 * b
        B[i] = 4.0 * (d + 0.5)
        B[N + i] = B[i] - 1.0
    raw = conv("coords", W, B)
    positive = node("relu", [raw])
    predicates01 = node(
        "sub", [sl(positive, 0, N), sl(positive, N, 2 * N)]
    )

    W = np.zeros((len(descriptions), N), dtype=np.float32)
    B = np.zeros(len(descriptions), dtype=np.float32)
    for i, groups in enumerate(descriptions):
        for group in groups:
            for t in group:
                W[i, pred_index[tuple(t)]] += 1.0
        B[i] = 1.0 - len(groups)
    masks = node("relu", [conv(predicates01, W, B)])

    region_masks = sl(masks, 0, 12)
    basket_masks = sl(masks, labels["basket0"], labels["basket0"] + 8)
    tool_present_masks = sl(
        masks, labels["tool_present0"], labels["tool_present0"] + 4
    )
    tool_click_masks = sl(
        masks, labels["tool_click0"], labels["tool_click0"] + 4
    )
    palette_mask = sl(
        masks, labels["palette_row"], labels["palette_row"] + 1
    )
    constraint_mask = sl(
        masks, labels["constraints"], labels["constraints"] + 1
    )

    # ------------------------------------------------------------
    # Read basket orientation and small-tool availability.
    # Memory codes: 0 = unknown, 1 = absent, 2 = present.
    # ------------------------------------------------------------
    red = sl("x0", 2, 3)
    orientation = node(
        "gmax", [node("mul", [basket_masks, red])]
    )
    tool_bits = node(
        "gmax", [node("mul", [tool_present_masks, red])]
    )
    tool_present = dense(tool_bits, np.ones((1, 4)))
    even = dense(
        orientation,
        np.asarray([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)
    )
    odd = dense(even, [[-1]], [1])

    mem_minus_one = node("vrelu", [dense("mem", [[1]], [-1])])
    one_minus_mem = node("vrelu", [dense("mem", [[-1]], [1])])
    unknown_or_present = node(
        "vadd", [mem_minus_one, one_minus_mem]
    )
    diagonal_tool = node("vmul", [odd, unknown_or_present])
    tool_enabled = node("vadd", [tool_present, diagonal_tool])

    code = dense(tool_present, [[1]], [1])
    even_code = node("vmul", [even, code])
    old_code = node("vmul", [odd, "mem"])
    node("vadd", [even_code, old_code], ident="mem_next")

    # Enable the four small masks only when that tool is available.
    W = np.zeros((12, 1), dtype=np.float32)
    W[8:, 0] = 1
    B = np.zeros(12, dtype=np.float32)
    B[:8] = 1
    enable_regions = dense(tool_enabled, W, B)
    usable_masks = node("mul", [region_masks, enable_regions])

    # ------------------------------------------------------------
    # Align the current canvas with the target coordinates.
    # Only a scalar colour image needs to be shifted.
    # Canvas origin minus target origin = (31 rows, 24 columns).
    # ------------------------------------------------------------
    numeric = conv(
        "x0", np.arange(16, dtype=np.float32).reshape(1, 16)
    )

    shift_diag = np.zeros((1, 1, 7, 7), dtype=np.float32)
    shift_diag[0, 0, 6, 6] = 1
    shift_diag_name = parameter(shift_diag)
    G.append({
        "id": "align_diagonal",
        "op": "loop",
        "T": 8,
        "state": [{"id": "align_d_state", "init": numeric}],
        "body": [{
            "id": "align_d_step",
            "op": "conv",
            "in": ["align_d_state"],
            "w": shift_diag_name,
        }],
        "next": {"align_d_state": "align_d_step"},
    })

    shift_rows = np.zeros((1, 1, 7, 7), dtype=np.float32)
    shift_rows[0, 0, 6, 3] = 1
    shift_rows_name = parameter(shift_rows)
    G.append({
        "id": "align_rows",
        "op": "loop",
        "T": 2,
        "state": [{
            "id": "align_r_state",
            "init": "align_diagonal.align_d_state",
        }],
        "body": [{
            "id": "align_r_step",
            "op": "conv",
            "in": ["align_r_state"],
            "w": shift_rows_name,
        }],
        "next": {"align_r_state": "align_r_step"},
    })

    shift_last = np.zeros((1, 1, 3, 3), dtype=np.float32)
    shift_last[0, 0, 2, 1] = 1
    aligned_canvas = conv(
        "align_rows.align_r_state", shift_last
    )
    difference = node("abs", [node("sub", [numeric, aligned_canvas])])
    excess = node("relu", [conv(difference, [[1]], [-1])])
    wrong_pixels = node("sub", [difference, excess])

    # Replicate region masks and target colour planes into 12*16 channels.
    W = np.zeros((192, 12), dtype=np.float32)
    for r in range(12):
        W[16 * r:16 * (r + 1), r] = 1
    repeated_masks = conv(usable_masks, W)

    W = np.zeros((192, 16), dtype=np.float32)
    for r in range(12):
        W[16 * r:16 * (r + 1)] = np.eye(16, dtype=np.float32)
    repeated_colours = conv("x0", W)
    regional_target = node("mul", [repeated_masks, repeated_colours])

    # ------------------------------------------------------------
    # Reverse peeling planner.
    #
    # presence[r,c] is whether colour c occurs in the still-constrained
    # part of region r. A region is peelable iff exactly one colour occurs.
    # A triangular dense map selects the first peelable region.
    #
    # The output is overwritten while any remaining constraint disagrees
    # with the current canvas. Thus it retains the earliest needed stroke
    # of the resulting forward plan.
    # ------------------------------------------------------------
    body = []
    outer_G = G
    G = body

    constrained_target = node("mul", [regional_target, "rev_left"])
    presence = node("gmax", [constrained_target])

    W = np.zeros((12, 192), dtype=np.float32)
    for r in range(12):
        W[r, 16 * r:16 * (r + 1)] = 1
    colour_counts = dense(presence, W)
    single_factor = node(
        "vrelu", [dense(colour_counts, -np.eye(12), np.full(12, 2))]
    )

    W = np.zeros((192, 12), dtype=np.float32)
    for r in range(12):
        W[16 * r:16 * (r + 1), r] = 1
    repeated_factor = dense(single_factor, W)
    valid_pairs = node("vmul", [presence, repeated_factor])

    W = np.zeros((12, 192), dtype=np.float32)
    for r in range(12):
        W[r, 16 * r:16 * (r + 1)] = 1
    eligible = dense(valid_pairs, W)

    W = np.eye(12, dtype=np.float32)
    for r in range(12):
        W[r, :r] = -1
    selected_region = node("vrelu", [dense(eligible, W)])

    W = np.zeros((192, 12), dtype=np.float32)
    for r in range(12):
        W[16 * r:16 * (r + 1), r] = 1
    repeated_selection = dense(selected_region, W)
    selected_pair = node("vmul", [valid_pairs, repeated_selection])

    selected_spatial = node("mul", [usable_masks, selected_region])
    selected_union = conv(selected_spatial, np.ones((1, 12)))
    left_next = node(
        "relu", [node("sub", ["rev_left", selected_union])]
    )

    wrong_here = node(
        "gmax", [node("mul", [wrong_pixels, "rev_left"])]
    )
    # OR(wrong_here, first_iteration).
    both = node("vmul", [wrong_here, "rev_first"])
    either_sum = node("vadd", [wrong_here, "rev_first"])
    negative_both = dense(both, [[-1]])
    update_gate = node("vadd", [either_sum, negative_both])
    keep_gate = dense(update_gate, [[-1]], [1])
    kept_output = node("vmul", ["rev_out", keep_gate])
    fresh_output = node("vmul", [selected_pair, update_gate])
    output_next = node("vadd", [kept_output, fresh_output])

    G = outer_G
    G.append({
        "id": "peel",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "rev_left", "init": constraint_mask},
            {"id": "rev_out", "zeros": [192]},
            {"id": "rev_first", "init": one},
        ],
        "body": body,
        "next": {
            "rev_left": left_next,
            "rev_out": output_next,
            "rev_first": zero,
        },
    })

    chosen = "peel.rev_out"

    W = np.zeros((12, 192), dtype=np.float32)
    for r in range(12):
        W[r, 16 * r:16 * (r + 1)] = 1
    wanted_region = dense(chosen, W)

    W = np.zeros((16, 192), dtype=np.float32)
    for r in range(12):
        W[:, 16 * r:16 * (r + 1)] = np.eye(16, dtype=np.float32)
    wanted_colour = dense(chosen, W)
    wanted_small = sl(wanted_region, 8, 12, vector=True)
    small_goal = dense(wanted_small, np.ones((1, 4)))

    # ------------------------------------------------------------
    # Read the selected palette colour from the black underline.
    # Palette borders (colour 4) are excluded.
    # ------------------------------------------------------------
    black = sl("x0", 0, 1)
    underline_shift = np.zeros((1, 1, 7, 7), dtype=np.float32)
    underline_shift[0, 0, 6, 3] = 1
    underline_at_palette = conv(black, underline_shift)
    selected_area = node("mul", [palette_mask, underline_at_palette])
    selected_colours = node(
        "gmax", [node("mul", ["x0", selected_area])]
    )
    W = np.eye(16, dtype=np.float32)
    W[4, 4] = 0
    current_colour = dense(selected_colours, W)

    colour_product = node("vmul", [current_colour, wanted_colour])
    colour_equal = dense(colour_product, np.ones((1, 16)))
    colour_different = dense(colour_equal, [[-1]], [1])

    # Palette click heat map.
    palette_colour_match = node("mul", ["x0", wanted_colour])
    palette_score = conv(palette_colour_match, np.ones((1, 16)))
    palette_score = node("mul", [palette_score, palette_mask])
    palette_click = node("mul", [palette_score, colour_different])

    # Small-tool click heat map.
    small_score = node("mul", [tool_click_masks, wanted_small])
    small_score = conv(small_score, np.ones((1, 4)))
    small_click = node("mul", [small_score, colour_equal])
    node("add", [palette_click, small_click], ident="click")

    # ------------------------------------------------------------
    # Navigation on the eight-position ring.
    # Combine current orientation and desired region into 96 Boolean
    # current/goal pairs, then decode a shortest ring direction.
    # ------------------------------------------------------------
    current_and_goal = node(
        "vconcat", [orientation, wanted_region]
    )
    W = np.zeros((96, 20), dtype=np.float32)
    for i in range(8):
        for r in range(12):
            q = 12 * i + r
            W[q, i] = 1
            W[q, 8 + r] = 1
    joint = node(
        "vrelu", [dense(current_and_goal, W, np.full(96, -1))]
    )

    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    counter_action = [3, 3, 1, 1, 4, 4, 2, 2]
    region_orientation = list(range(8)) + [0, 2, 4, 6]

    W = np.zeros((8, 96), dtype=np.float32)
    for i in range(8):
        for r, goal in enumerate(region_orientation):
            delta = (goal - i) % 8
            if delta:
                action = (
                    clockwise_action[i] if delta <= 4
                    else counter_action[i]
                )
                W[action, 12 * i + r] = 20
    navigation_logits = dense(joint, W)

    # At an unknown diagonal start, first visit a neighbouring cardinal
    # position so that tool availability is unambiguous.
    unknown_diagonal = node("vmul", [odd, one_minus_mem])
    W = np.zeros((8, 8), dtype=np.float32)
    for i in (1, 3, 5, 7):
        W[clockwise_action[i], i] = 50
    probe_logits = dense(orientation, W)
    probe_logits = node("vmul", [probe_logits, unknown_diagonal])

    action_features = node(
        "vconcat",
        [navigation_logits, probe_logits, colour_different, small_goal]
    )
    W = np.zeros((8, 18), dtype=np.float32)
    W[:, :8] = np.eye(8, dtype=np.float32)
    W[:, 8:16] = np.eye(8, dtype=np.float32)
    W[6, 16] = 10
    W[6, 17] = 2
    B = np.asarray([-100, 0, 0, 0, 0, 1, 0, -100], dtype=np.float32)
    dense(action_features, W, B, ident="logits")

    return {
        "graph": graph,
        "memory_size": 1,
        "params": params,
    }

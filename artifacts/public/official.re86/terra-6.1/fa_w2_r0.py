import numpy as np


def build():
    graph = []
    params = {}
    colors = [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    C = len(colors)
    K = 24
    M = K + 4

    def par(name, value):
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(name, op, inputs, **fields):
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        graph.append(d)
        return name

    def conv(name, source, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node(name, "conv", [source], **fields)

    def dense(name, source, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return node(name, "dense", [source], **fields)

    def sl(name, source, a, b):
        return node(name, "slice", [source], start=a, stop=b)

    def vsl(name, source, a, b):
        return node(name, "vslice", [source], start=a, stop=b)

    def affine(name, source, scale, bias):
        par(name + "_w", [[[[scale]]]])
        par(name + "_b", [bias])
        return conv(name, source, name + "_w", name + "_b")

    def vaffine(name, source, scale, bias):
        par(name + "_w", [[scale]])
        par(name + "_b", [bias])
        return dense(name, source, name + "_w", name + "_b")

    def clamp_vector(name, source, size):
        par(name + "_shift_w", np.eye(size, dtype=np.float32))
        par(name + "_shift_b", -np.ones(size, dtype=np.float32))
        shifted = dense(
            name + "_shift", source, name + "_shift_w", name + "_shift_b"
        )
        excess = node(name + "_excess", "vrelu", [shifted])
        par(name + "_neg_w", -np.eye(size, dtype=np.float32))
        neg = dense(name + "_neg", excess, name + "_neg_w")
        return node(name, "vadd", [source, neg])

    one = sl("one", "coords", 2, 3)
    xy = sl("xy", "coords", 0, 2)
    black = sl("black", "x0", 0, 1)
    fours = sl("fours", "x0", 4, 5)

    cw = np.zeros((C, 16, 1, 1), np.float32)
    for i, color in enumerate(colors):
        cw[i, color, 0, 0] = 1.0
    par("color_w", cw)
    color_planes = conv("color_planes", "x0", "color_w")
    par("sum_colors_w", np.ones((1, C, 1, 1), np.float32))

    # Recognize visible 3x3 marker borders.
    rw = np.full((1, 1, 3, 3), 2.0, np.float32)
    rw[0, 0, 1, 1] = 0.0
    par("ring_w", rw)
    par("ring_b", [-15.0])
    ring0 = conv("ring0", fours, "ring_w", "ring_b")
    ring1 = node("ring1", "relu", [ring0])
    ring = node("ring", "min2", [ring1, one])

    # Encode each marker's position and colour into a float.
    bw = np.zeros((C, 3, 1, 1), np.float32)
    bw[:, 0, 0, 0] = 63.0 / 128.0
    bw[:, 1, 0, 0] = 63.0 / 8192.0
    bb = np.array(
        [1.0 + color / 131072.0 for color in colors], np.float32
    )
    par("base_w", bw)
    par("base_b", bb)
    bases = conv("bases", "coords", "base_w", "base_b")

    dw = np.zeros((C, C, 1, 1), np.float32)
    for i in range(C):
        dw[i, i, 0, 0] = -262144.0
    par("decode_w", dw)
    par("decode_b", np.ones(C, np.float32))

    old_goals = None
    for i in range(K):
        code = vsl("old_code_%d" % i, "mem", i, i + 1)
        diff = node("old_diff_%d" % i, "sub", [bases, code])
        adiff = node("old_abs_%d" % i, "abs", [diff])
        eq0 = conv("old_eq0_%d" % i, adiff, "decode_w", "decode_b")
        eq = node("old_eq_%d" % i, "relu", [eq0])
        if old_goals is None:
            old_goals = eq
        else:
            old_goals = node(
                "old_union_%d" % i, "max2", [old_goals, eq]
            )

    visible_goals = node("visible_goals", "mul", [color_planes, ring])
    goals = node("goals", "max2", [old_goals, visible_goals])
    coded = node("coded", "mul", [bases, goals])
    remaining = conv("coded_sum", coded, "sum_colors_w")
    par("extract_eq_w", [[[[-262144.0]]]])
    par("extract_eq_b", [1.0])
    codes = []
    for i in range(K):
        best = node("best_code_%d" % i, "gmax", [remaining])
        codes.append(best)
        diff = node("extract_diff_%d" % i, "sub", [remaining, best])
        adiff = node("extract_abs_%d" % i, "abs", [diff])
        eq0 = conv(
            "extract_eq0_%d" % i,
            adiff,
            "extract_eq_w",
            "extract_eq_b",
        )
        eq = node("extract_eq_%d" % i, "relu", [eq0])
        remove = node("extract_remove_%d" % i, "mul", [remaining, eq])
        remaining = node(
            "extract_remaining_%d" % i, "sub", [remaining, remove]
        )
    new_codes = node("new_codes", "vconcat", codes)

    par("ones3_w", np.ones((1, 1, 3, 3), np.float32))
    par("ones7_w", np.ones((1, 1, 7, 7), np.float32))
    nw = np.ones((1, 1, 3, 3), np.float32)
    nw[0, 0, 1, 1] = 0.0
    par("neighbor_w", nw)

    all_goals = conv("all_goals", goals, "sum_colors_w")
    goal_area0 = conv("goal_area0", all_goals, "ones3_w")
    goal_area = node("goal_area", "min2", [goal_area0, one])
    visible_area0 = conv("visible_area0", ring, "ones3_w")
    visible_area = node("visible_area", "min2", [visible_area0, one])
    marker_area = node("marker_area", "max2", [goal_area, visible_area])
    not_marker = node("not_marker", "sub", [one, marker_area])

    occupancy = conv("occupancy", color_planes, "sum_colors_w")
    solid_count = conv("solid_count", occupancy, "ones3_w")
    solid0 = affine("solid0", solid_count, 2.0, -17.0)
    solid1 = node("solid1", "relu", [solid0])
    solid = node("solid", "min2", [solid1, one])
    box_area0 = conv("box_area0", solid, "ones7_w")
    box_area = node("box_area", "min2", [box_area0, one])
    not_box = node("not_box", "sub", [one, box_area])
    foreground_mask = node(
        "foreground_mask", "mul", [not_marker, not_box]
    )
    foreground = node(
        "foreground", "mul", [color_planes, foreground_mask]
    )

    center_weighted = node("center_weighted", "mul", [xy, black])
    center_mean = node("center_mean", "gmean", [center_weighted])
    par("center_w", 4096.0 * np.eye(2, dtype=np.float32))
    center = dense("center", center_mean, "center_w")
    delta = node("delta", "sub", [xy, center])
    dr = sl("dr", delta, 0, 1)
    dc = sl("dc", delta, 1, 2)
    adr = node("adr", "abs", [dr])
    adc = node("adc", "abs", [dc])
    distance = node("distance", "add", [adr, adc])
    cheb = node("cheb", "max2", [adr, adc])
    near = affine("near", distance, -0.5, 1.0)

    # Identify the selected figure's colour.
    weighted_fg = node("weighted_fg", "mul", [foreground, near])
    color_best = node("color_best", "gmax", [weighted_fg])
    all_weighted = conv("all_weighted", weighted_fg, "sum_colors_w")
    overall_best = node("overall_best", "gmax", [all_weighted])
    par("neg_best_w", -np.ones((C, 1), np.float32))
    par("neg_best_b", np.full(C, 0.001, np.float32))
    neg_best = dense("neg_best", overall_best, "neg_best_w", "neg_best_b")
    color_margin = node("color_margin", "vadd", [color_best, neg_best])
    color_positive = node("color_positive", "vrelu", [color_margin])
    par("color_gate_w", 1000.0 * np.eye(C, dtype=np.float32))
    color_gate0 = dense("color_gate0", color_positive, "color_gate_w")
    color_gate = clamp_vector("color_gate", color_gate0, C)
    selected_planes = node(
        "selected_planes", "mul", [foreground, color_gate]
    )
    selected_fg = conv("selected_fg", selected_planes, "sum_colors_w")
    actual_planes = node("actual_planes", "mul", [color_planes, color_gate])
    actual_selected = conv("actual_selected", actual_planes, "sum_colors_w")

    # Remember local horizontal, vertical and diagonal geometry.
    lw = np.zeros((3, 1, 3, 3), np.float32)
    lw[0, 0, 1, 0] = lw[0, 0, 1, 2] = 1.0
    lw[1, 0, 0, 1] = lw[1, 0, 2, 1] = 1.0
    for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        lw[2, 0, r, c] = 1.0
    par("local_w", lw)
    local0 = conv("local0", selected_fg, "local_w")
    local1 = node("local1", "mul", [local0, black])
    local2 = node("local2", "min2", [local1, one])
    current_type = node("current_type", "gmax", [local2])

    reset_w = -np.ones((1, 8), np.float32)
    reset_w[0, 5] = 0.0
    par("type_reset_w", reset_w)
    par("type_reset_b", [1.0])
    reset_type = dense(
        "reset_type", "aprev", "type_reset_w", "type_reset_b"
    )
    par("complement_w", [[-1.0]])
    par("complement_b", [1.0])
    keep_type = dense(
        "keep_type", reset_type, "complement_w", "complement_b"
    )
    old_type = vsl("old_type", "mem", K, K + 3)
    fresh_type = node("fresh_type", "vmul", [current_type, reset_type])
    kept_type = node("kept_type", "vmul", [old_type, keep_type])
    figure_type = node("figure_type", "vadd", [fresh_type, kept_type])

    # Count consecutive cycles. After several cycles, allow rearrangement.
    old_cycles = vsl("old_cycles", "mem", K + 3, K + 4)
    cycles_plus = vaffine("cycles_plus", old_cycles, 1.0, 1.0)
    prev_cycle = vsl("prev_cycle", "aprev", 5, 6)
    cycles = node("cycles", "vmul", [cycles_plus, prev_cycle])
    nudge0 = vaffine("nudge0", cycles, 1.0, -3.0)
    nudge1 = node("nudge1", "vrelu", [nudge0])
    nudge = clamp_vector("nudge", nudge1, 1)
    node("mem_next", "vconcat", [new_codes, figure_type, cycles])

    hf = vsl("hf", figure_type, 0, 1)
    vf = vsl("vf", figure_type, 1, 2)
    df = vsl("df", figure_type, 2, 3)
    par("cardinal_sum_w", [[1.0, 1.0, 0.0]])
    cardinal_sum = dense("cardinal_sum", figure_type, "cardinal_sum_w")
    missing0 = dense(
        "missing0", cardinal_sum, "complement_w", "complement_b"
    )
    missing = node("missing", "vrelu", [missing0])
    not_diag = dense("not_diag", df, "complement_w", "complement_b")
    hollow_gate = node("hollow_gate", "vmul", [not_diag, missing])

    row0 = affine("row0", adr, -1000.0, 1.0)
    row_mask = node("row_mask", "relu", [row0])
    col0 = affine("col0", adc, -1000.0, 1.0)
    col_mask = node("col_mask", "relu", [col0])
    diag_delta = node("diag_delta", "sub", [adr, adc])
    diag_abs = node("diag_abs", "abs", [diag_delta])
    diag0 = affine("diag0", diag_abs, -1000.0, 1.0)
    diag_mask = node("diag_mask", "relu", [diag0])
    hmask = node("hmask", "mul", [row_mask, hf])
    vmask = node("vmask", "mul", [col_mask, vf])
    cross_mask = node("cross_mask", "max2", [hmask, vmask])
    diag_part = node("diag_part", "mul", [diag_mask, df])
    cross_part = node("cross_part", "mul", [cross_mask, not_diag])
    line_geometry = node(
        "line_geometry", "add", [diag_part, cross_part]
    )

    near_diff = node("near_diff", "sub", [near, overall_best])
    near_abs = node("near_abs", "abs", [near_diff])
    nearest0 = affine("nearest0", near_abs, -1000.0, 1.0)
    nearest1 = node("nearest1", "relu", [nearest0])
    nearest = node("nearest", "mul", [nearest1, selected_fg])
    seed0 = node("seed0", "max2", [nearest, black])
    seed = node("seed", "mul", [seed0, line_geometry])
    allowed0 = node("allowed0", "max2", [selected_fg, marker_area])
    allowed1 = node("allowed1", "max2", [allowed0, black])
    allowed = node("allowed", "mul", [allowed1, line_geometry])

    graph.append({
        "id": "component",
        "op": "loop",
        "T": 72,
        "state": [{"id": "component_state", "init": seed}],
        "body": [
            {
                "id": "component_expand",
                "op": "conv",
                "in": ["component_state"],
                "w": "ones3_w",
            },
            {
                "id": "component_next",
                "op": "min2",
                "in": ["component_expand", allowed],
            },
        ],
        "next": {"component_state": "component_next"},
    })

    # Estimate hollow diamond or square outlines from radial histograms.
    R = 30
    N = 2 * R
    radial_input = node("radial_input", "concat", [distance, cheb])
    radial_w = np.zeros((N, 2, 1, 1), np.float32)
    radial_b = np.zeros(N, np.float32)
    for kind in range(2):
        for j in range(R):
            i = kind * R + j
            radial_w[i, kind, 0, 0] = 63.0
            radial_b[i] = -(j + 1)
    par("radial_w", radial_w)
    par("radial_b", radial_b)
    radial_diff = conv("radial_diff", radial_input, "radial_w", "radial_b")
    radial_abs = node("radial_abs", "abs", [radial_diff])
    eq_w = np.zeros((N, N, 1, 1), np.float32)
    for i in range(N):
        eq_w[i, i, 0, 0] = -2.0
    par("radial_eq_w", eq_w)
    par("radial_eq_b", np.ones(N, np.float32))
    bins0 = conv("bins0", radial_abs, "radial_eq_w", "radial_eq_b")
    bins = node("bins", "relu", [bins0])
    hits = node("radial_hits", "mul", [bins, selected_fg])
    means = node("radial_means", "gmean", [hits])
    par("radial_count_w", 4096.0 * np.eye(N, dtype=np.float32))
    par(
        "radial_count_b",
        np.array([(N - i) * 0.0001 for i in range(N)], np.float32),
    )
    counts = dense(
        "radial_counts", means, "radial_count_w", "radial_count_b"
    )

    arw = np.zeros((N, 3, 1, 1), np.float32)
    acw = np.zeros((N, 3, 1, 1), np.float32)
    arb = np.zeros(N, np.float32)
    acb = np.zeros(N, np.float32)
    for kind in range(2):
        for j in range(R):
            i = kind * R + j
            arw[i, 0, 0, 0] = 63.0
            acw[i, 1, 0, 0] = 63.0
            arb[i] = -(j + 1)
            acb[i] = -(kind + 1)
    par("anchor_row_w", arw)
    par("anchor_row_b", arb)
    par("anchor_col_w", acw)
    par("anchor_col_b", acb)
    ar0 = conv("ar0", "coords", "anchor_row_w", "anchor_row_b")
    ac0 = conv("ac0", "coords", "anchor_col_w", "anchor_col_b")
    ara = node("ara", "abs", [ar0])
    aca = node("aca", "abs", [ac0])
    ar1 = conv("ar1", ara, "radial_eq_w", "radial_eq_b")
    ac1 = conv("ac1", aca, "radial_eq_w", "radial_eq_b")
    ar = node("ar", "relu", [ar1])
    ac = node("ac", "relu", [ac1])
    anchors = node("anchors", "mul", [ar, ac])
    anchor_scores = node("anchor_scores", "mul", [anchors, counts])
    par("sum_radial_w", np.ones((1, N, 1, 1), np.float32))
    radial_score_map = conv(
        "radial_score_map", anchor_scores, "sum_radial_w"
    )
    radial_best = node("radial_best", "gmax", [radial_score_map])
    best_diff = node("radial_best_diff", "sub", [radial_score_map, radial_best])
    best_abs = node("radial_best_abs", "abs", [best_diff])
    pick0 = affine("radial_pick0", best_abs, -1000.0, 1.0)
    pick1 = node("radial_pick1", "relu", [pick0])
    all_anchors = conv("all_anchors", anchors, "sum_radial_w")
    pick = node("radial_pick", "mul", [pick1, all_anchors])
    picked_xy = node("picked_xy", "mul", [pick, xy])
    picked_coords = node("picked_coords", "gmax", [picked_xy])
    radius = vsl("radius", picked_coords, 0, 1)
    kind_coord = vsl("kind_coord", picked_coords, 1, 2)
    kind = vaffine("kind", kind_coord, 63.0, 0.0)
    is_square = vaffine("is_square", kind, 1.0, -1.0)
    is_diamond = vaffine("is_diamond", kind, -1.0, 2.0)
    diamond_diff = node("diamond_diff", "sub", [distance, radius])
    diamond_abs = node("diamond_abs", "abs", [diamond_diff])
    diamond0 = affine("diamond0", diamond_abs, -1000.0, 1.0)
    diamond = node("diamond", "relu", [diamond0])
    square_diff = node("square_diff", "sub", [cheb, radius])
    square_abs = node("square_abs", "abs", [square_diff])
    square0 = affine("square0", square_abs, -1000.0, 1.0)
    square = node("square", "relu", [square0])
    diamond_part = node("diamond_part", "mul", [diamond, is_diamond])
    square_part = node("square_part", "mul", [square, is_square])
    outline = node("outline", "add", [diamond_part, square_part])
    hollow_pattern = node("hollow_pattern", "mul", [outline, hollow_gate])
    pattern = node(
        "pattern", "max2", ["component.component_state", hollow_pattern]
    )

    selected_goals = node("selected_goals", "mul", [goals, color_gate])
    goal_map = conv("goal_map", selected_goals, "sum_colors_w")
    uncovered = node("uncovered", "mul", [goal_map, ring])
    owned_pixels = node("owned_pixels", "mul", [actual_selected, pattern])
    neighbors0 = conv("owned_neighbors0", owned_pixels, "neighbor_w")
    neighbors = node("owned_neighbors", "min2", [neighbors0, one])
    owned = node("owned", "mul", [goal_map, neighbors])
    eligible = node("eligible", "max2", [uncovered, owned])
    other = node("other", "sub", [goal_map, eligible])
    other_credit = affine("other_credit", other, 0.85, 0.0)
    scoring_goals = node("scoring_goals", "add", [eligible, other_credit])
    owned_mean = node("owned_mean", "gmean", [owned])
    par("scale4096_w", [[4096.0]])
    current_score = dense("current_score", owned_mean, "scale4096_w")

    # Search translations at increments of three pixels.
    for name, dy, dx in [
        ("shift_left_w", 0, -3),
        ("shift_right_w", 0, 3),
        ("shift_up_w", -3, 0),
        ("shift_down_w", 3, 0),
    ]:
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3 - dy, 3 - dx] = 1.0
        par(name, w)
    par("score_scale_w", 4096.0 * np.eye(4, dtype=np.float32))
    signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    goal_y = {1: scoring_goals, -1: scoring_goals}
    cursor_y = {1: black, -1: black}
    previous_acc = None

    for row in range(21):
        states = []
        for q, (sx, sy) in enumerate(signs):
            states.append({
                "id": "scan_g_%d_%d" % (row, q),
                "init": goal_y[sy],
            })
            states.append({
                "id": "scan_p_%d_%d" % (row, q),
                "init": cursor_y[sy],
            })
        acc_id = "scan_acc_%d" % row
        if previous_acc is None:
            states.append({"id": acc_id, "zeros": [4, 64, 64]})
        else:
            states.append({"id": acc_id, "init": previous_acc})

        gs = ["scan_g_%d_%d" % (row, q) for q in range(4)]
        ps = ["scan_p_%d_%d" % (row, q) for q in range(4)]
        prefix = "scan_%d_" % row
        body = [
            {"id": prefix + "goals", "op": "concat", "in": gs},
            {"id": prefix + "cursors", "op": "concat", "in": ps},
            {
                "id": prefix + "overlap",
                "op": "mul",
                "in": [prefix + "goals", pattern],
            },
            {
                "id": prefix + "mean",
                "op": "gmean",
                "in": [prefix + "overlap"],
            },
            {
                "id": prefix + "score",
                "op": "dense",
                "in": [prefix + "mean"],
                "w": "score_scale_w",
            },
            {
                "id": prefix + "deposit",
                "op": "mul",
                "in": [prefix + "cursors", prefix + "score"],
            },
            {
                "id": prefix + "acc_next",
                "op": "add",
                "in": [acc_id, prefix + "deposit"],
            },
        ]
        next_states = {acc_id: prefix + "acc_next"}
        for q, (sx, sy) in enumerate(signs):
            gn = prefix + "g_next_%d" % q
            pn = prefix + "p_next_%d" % q
            body.append({
                "id": gn,
                "op": "conv",
                "in": [gs[q]],
                "w": "shift_left_w" if sx > 0 else "shift_right_w",
            })
            body.append({
                "id": pn,
                "op": "conv",
                "in": [ps[q]],
                "w": "shift_right_w" if sx > 0 else "shift_left_w",
            })
            next_states[gs[q]] = gn
            next_states[ps[q]] = pn

        loop_id = "scan_loop_%d" % row
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 21,
            "state": states,
            "body": body,
            "next": next_states,
        })
        previous_acc = loop_id + "." + acc_id

        if row < 20:
            next_goal_y = {}
            next_cursor_y = {}
            for sy in (1, -1):
                suffix = "pos" if sy > 0 else "neg"
                next_goal_y[sy] = conv(
                    "goal_y_%d_%s" % (row, suffix),
                    goal_y[sy],
                    "shift_up_w" if sy > 0 else "shift_down_w",
                )
                next_cursor_y[sy] = conv(
                    "cursor_y_%d_%s" % (row, suffix),
                    cursor_y[sy],
                    "shift_down_w" if sy > 0 else "shift_up_w",
                )
            goal_y = next_goal_y
            cursor_y = next_cursor_y

    planes = [
        sl("score_plane_%d" % i, previous_acc, i, i + 1)
        for i in range(4)
    ]
    score01 = node("score01", "max2", planes[:2])
    score23 = node("score23", "max2", planes[2:])
    score_map = node("score_map", "max2", [score01, score23])
    penalty = affine("penalty", distance, 63.0 * 0.015 / 3.0, 0.0)
    quality = node("quality", "sub", [score_map, penalty])
    improvement0 = node("improvement0", "sub", [quality, current_score])
    exploration = vaffine("exploration", nudge, 0.85, -0.1)
    improvement1 = node("improvement1", "add", [improvement0, exploration])
    improvement = node("improvement", "relu", [improvement1])

    up0 = affine("up0", dr, -63.0, -0.25)
    down0 = affine("down0", dr, 63.0, -0.25)
    left0 = affine("left0", dc, -63.0, -0.25)
    right0 = affine("right0", dc, 63.0, -0.25)
    values = []
    for name, raw in [
        ("up", up0),
        ("down", down0),
        ("left", left0),
        ("right", right0),
    ]:
        positive = node(name + "_positive", "relu", [raw])
        mask = node(name + "_mask", "min2", [positive, one])
        value_map = node(name + "_value_map", "mul", [improvement, mask])
        values.append(node(name + "_value", "gmax", [value_map]))
    direction_vector = node("direction_vector", "vconcat", values)

    # Pick a paint colour among boxes actually present.
    box_color_planes = node("box_color_planes", "mul", [color_planes, solid])
    box_present = node("box_present", "gmax", [box_color_planes])
    uncovered_colors = node("uncovered_colors", "mul", [goals, ring])
    color_goal_mean = node("color_goal_mean", "gmean", [uncovered_colors])
    par("goal_count_w", 4096.0 * np.eye(C, dtype=np.float32))
    par(
        "goal_count_b",
        np.array([(C - i) * 0.0001 for i in range(C)], np.float32),
    )
    goal_count0 = dense(
        "goal_count0", color_goal_mean, "goal_count_w", "goal_count_b"
    )
    goal_count = node("goal_count", "vmul", [goal_count0, box_present])

    caw = np.zeros((C, 3, 1, 1), np.float32)
    cab = np.zeros(C, np.float32)
    for i in range(C):
        caw[i, 1, 0, 0] = 63.0
        cab[i] = -(i + 1)
    par("color_anchor_w", caw)
    par("color_anchor_b", cab)
    cad = conv(
        "color_anchor_diff", "coords", "color_anchor_w", "color_anchor_b"
    )
    caa = node("color_anchor_abs", "abs", [cad])
    caew = np.zeros((C, C, 1, 1), np.float32)
    for i in range(C):
        caew[i, i, 0, 0] = -2.0
    par("color_anchor_eq_w", caew)
    par("color_anchor_eq_b", np.ones(C, np.float32))
    ca0 = conv(
        "color_anchor0", caa, "color_anchor_eq_w", "color_anchor_eq_b"
    )
    color_anchors = node("color_anchors", "relu", [ca0])
    color_score_planes = node(
        "color_score_planes", "mul", [color_anchors, goal_count]
    )
    color_score_map = conv(
        "color_score_map", color_score_planes, "sum_colors_w"
    )
    best_goal_count = node("best_goal_count", "gmax", [color_score_map])
    par("neg_goal_count_w", -np.ones((C, 1), np.float32))
    par("neg_goal_count_b", np.full(C, 0.00005, np.float32))
    neg_goal_count = dense(
        "neg_goal_count",
        best_goal_count,
        "neg_goal_count_w",
        "neg_goal_count_b",
    )
    goal_margin = node("goal_margin", "vadd", [goal_count, neg_goal_count])
    goal_positive = node("goal_positive", "vrelu", [goal_margin])
    par("goal_gate_w", 20000.0 * np.eye(C, dtype=np.float32))
    goal_gate0 = dense("goal_gate0", goal_positive, "goal_gate_w")
    goal_gate = clamp_vector("goal_gate", goal_gate0, C)
    paint_planes = node("paint_planes", "mul", [color_planes, goal_gate])
    paint_color = conv("paint_color", paint_planes, "sum_colors_w")
    paint_mask = node("paint_mask", "mul", [paint_color, solid])
    paint_exists = node("paint_exists", "gmax", [paint_mask])
    paint_xy = node("paint_xy", "mul", [paint_mask, xy])
    paint_center = node("paint_center", "gmax", [paint_xy])
    paint_delta = node("paint_delta", "vconcat", [paint_center, center])
    par(
        "paint_dir_w",
        np.array([
            [-63.0, 0.0, 63.0, 0.0],
            [63.0, 0.0, -63.0, 0.0],
            [0.0, -63.0, 0.0, 63.0],
            [0.0, 63.0, 0.0, -63.0],
        ], np.float32),
    )
    paint_dir0 = dense("paint_dir0", paint_delta, "paint_dir_w")
    paint_dir = node("paint_dir", "vrelu", [paint_dir0])
    has_matching = node("has_matching", "gmax", [goal_map])
    no_matching = dense(
        "no_matching", has_matching, "complement_w", "complement_b"
    )
    paint_gate = node("paint_gate", "vmul", [no_matching, paint_exists])
    paint_action = node("paint_action", "vmul", [paint_dir, paint_gate])
    combined = node("combined", "vadd", [direction_vector, paint_action])

    # Penalize immediate reversals without forbidding them.
    reversal_w = np.zeros((4, 8), np.float32)
    reversal_w[0, 2] = -0.35
    reversal_w[1, 1] = -0.35
    reversal_w[2, 4] = -0.35
    reversal_w[3, 3] = -0.35
    par("reversal_w", reversal_w)
    reversal = dense("reversal", "aprev", "reversal_w")
    final_direction = node("final_direction", "vadd", [combined, reversal])

    out_w = np.zeros((8, 4), np.float32)
    for i in range(4):
        out_w[i + 1, i] = 1.0
    out_b = np.array(
        [-100.0, 0.0, 0.0, 0.0, 0.0, 0.01, -100.0, -100.0],
        np.float32,
    )
    par("out_w", out_w)
    par("out_b", out_b)
    dense("logits", final_direction, "out_w", "out_b")

    par("click_w", np.zeros((1, 1, 1, 1), np.float32))
    conv("click", one, "click_w")
    return {"graph": graph, "memory_size": M, "params": params}

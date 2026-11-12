import numpy as np


def build():
    graph = []
    params = {}
    serial = 0

    def node(name, op, inputs=None, **fields):
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        graph.append(nd)
        return name

    def parameter(value):
        nonlocal serial
        name = "p%d" % serial
        serial += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(name, source, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node(name, "conv", [source], **fields)

    def dense(name, source, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node(name, "dense", [source], **fields)

    def ss(name, source, start, stop):
        return node(name, "slice", [source], start=start, stop=stop)

    def vs(name, source, start, stop):
        return node(name, "vslice", [source], start=start, stop=stop)

    def clip01(name, source, size):
        dense(name + "_neg", source, -np.eye(size), np.ones(size))
        node(name + "_relu", "vrelu", [name + "_neg"])
        dense(name, name + "_relu", -np.eye(size), np.ones(size))
        return name

    colors = [0, 6, 7, 8, 9, 10, 11, 12, 14]
    nc = len(colors)
    # Per color: decrease, increase, rotate.
    na = 3 * nc
    memory_size = 48 + 3 * na

    ss("one", "coords", 2, 3)
    ss("color13", "x0", 13, 14)

    # Empty targets have four color-13 arms and a non-13 center.
    wt = np.zeros((1, 16, 3, 3), np.float32)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        wt[0, 13, 1 + dy, 1 + dx] = 2
    conv("target_raw", "x0", wt, [-7])
    node("target_cross", "relu", ["target_raw"])
    node("target_difference", "sub", ["target_cross", "color13"])
    node("target", "relu", ["target_difference"])

    # Occupied targets have a color-13 center and four color-13 arms.
    wcross = np.zeros((1, 16, 3, 3), np.float32)
    wcross[0, 13, 1, 1] = 2
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        wcross[0, 13, 1 + dy, 1 + dx] = 2
    conv("occupied_raw", "x0", wcross, [-9])
    node("occupied", "relu", ["occupied_raw"])
    conv("occupied_neighborhood_raw", "occupied",
         np.ones((1, 1, 3, 3), np.float32))
    node("occupied_neighborhood", "min2",
         ["occupied_neighborhood_raw", "one"])

    # Detect marker centers surrounded by each bar color.
    wd = np.zeros((nc, 16, 3, 3), np.float32)
    for i, color in enumerate(colors):
        wd[i, 13, 1, 1] = 8
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            wd[i, color, 1 + dy, 1 + dx] = 2
    conv("dot_raw", "x0", wd, np.full(nc, -13))
    node("dot_relu", "relu", ["dot_raw"])
    node("dot_candidates", "min2", ["dot_relu", "one"])
    node("dot_difference", "sub",
         ["dot_candidates", "occupied_neighborhood"])
    node("dots", "relu", ["dot_difference"])
    node("dot_present", "gmax", ["dots"])

    conv("any_dot_raw", "dots",
         np.ones((1, nc, 1, 1), np.float32))
    node("any_dot", "min2", ["any_dot_raw", "one"])

    # Protect colors near completed markers from unguided exploration.
    wnear = np.zeros((nc, 16, 5, 5), np.float32)
    for i, color in enumerate(colors):
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            wnear[i, color, 2 + dy, 2 + dx] = 1
    conv("near_colors", "x0", wnear)
    node("occupied_colors", "mul", ["near_colors", "occupied"])
    node("occupied_present_raw", "gmax", ["occupied_colors"])
    node("protected_sum", "vadd",
         ["dot_present", "occupied_present_raw"])
    clip01("protected", "protected_sum", nc)

    # Detect colored bars' pivot stripes. Groups: left, right, top, bottom.
    wp = np.zeros((4 * nc, 16, 3, 3), np.float32)
    for i, color in enumerate(colors):
        for group in range(4):
            out = group * nc + i
            wp[out, 3, 1, 1] = 2
            if group < 2:
                wp[out, 3, 0, 1] = 2
                wp[out, 3, 2, 1] = 2
                dy, dx = 0, 1 if group == 0 else -1
            else:
                wp[out, 3, 1, 0] = 2
                wp[out, 3, 1, 2] = 2
                dy, dx = (1 if group == 2 else -1), 0
            wp[out, color, 1 + dy, 1 + dx] = 2
            wp[out, 13, 1 + dy, 1 + dx] = 2
    conv("pivot_raw", "x0", wp, np.full(4 * nc, -7))
    node("pivots", "relu", ["pivot_raw"])

    # Fields describing the directions of empty targets from each pixel.
    node("target_columns", "colmax", ["target"])
    node("target_rows", "rowmax", ["target"])
    node("seeds", "concat", [
        "target", "target", "target", "target",
        "target_columns", "target_columns", "target_rows", "target_rows"
    ])

    wf = np.zeros((8, 8, 3, 3), np.float32)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for channel in range(8):
        dy, dx = directions[channel % 4]
        wf[channel, channel, 1 + dy, 1 + dx] = 1
    flow_w = parameter(wf)
    flow_b = parameter(np.full(8, -1.0 / 128, np.float32))
    graph.append({
        "id": "target_flow",
        "op": "loop",
        "T": 63,
        "state": [{"id": "target_state", "init": "seeds"}],
        "body": [
            {"id": "target_step", "op": "conv",
             "in": ["target_state"], "w": flow_w, "b": flow_b},
            {"id": "target_step_relu", "op": "relu",
             "in": ["target_step"]},
            {"id": "target_step_next", "op": "max2",
             "in": ["target_step_relu", "seeds"]},
        ],
        "next": {"target_state": "target_step_next"},
    })
    node("field_difference", "sub",
         ["target_flow.target_state", "seeds"])
    node("fields", "relu", ["field_difference"])

    # Four marker-local direction scores. Exact alignment is stronger than
    # projected alignment.
    wdir = np.zeros((4, 8, 1, 1), np.float32)
    for i in range(4):
        wdir[i, i, 0, 0] = 1
        wdir[i, i + 4, 0, 0] = 0.20
    conv("direction_fields", "fields", wdir)
    node("marker_drive", "mul", ["direction_fields", "any_dot"])

    # Traversable pixels for propagating marker guidance along connected
    # bars and connectors, but not through background, walls, or UI frames.
    wconduct = np.zeros((1, 16, 1, 1), np.float32)
    for color in [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        wconduct[0, color, 0, 0] = 1
    conv("conductive", "x0", wconduct)

    # One iteration takes a max over four shifted neighbors, attenuates
    # the signal, and retains the original marker source.
    wshift = np.zeros((16, 4, 3, 3), np.float32)
    for group, (dy, dx) in enumerate(
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ):
        for direction in range(4):
            wshift[group * 4 + direction, direction, 1 + dy, 1 + dx] = 1
    shift_w = parameter(wshift)
    shift_b = parameter(np.full(16, -0.003, np.float32))
    graph.append({
        "id": "bar_flow",
        "op": "loop",
        "T": 63,
        "state": [{"id": "bar_state", "init": "marker_drive"}],
        "body": [
            {"id": "bar_shifts", "op": "conv", "in": ["bar_state"],
             "w": shift_w, "b": shift_b},
            {"id": "bar_shift_relu", "op": "relu", "in": ["bar_shifts"]},
            {"id": "bar_s0", "op": "slice", "in": ["bar_shift_relu"],
             "start": 0, "stop": 4},
            {"id": "bar_s1", "op": "slice", "in": ["bar_shift_relu"],
             "start": 4, "stop": 8},
            {"id": "bar_s2", "op": "slice", "in": ["bar_shift_relu"],
             "start": 8, "stop": 12},
            {"id": "bar_s3", "op": "slice", "in": ["bar_shift_relu"],
             "start": 12, "stop": 16},
            {"id": "bar_m01", "op": "max2", "in": ["bar_s0", "bar_s1"]},
            {"id": "bar_m23", "op": "max2", "in": ["bar_s2", "bar_s3"]},
            {"id": "bar_m", "op": "max2", "in": ["bar_m01", "bar_m23"]},
            {"id": "bar_masked", "op": "mul", "in": ["bar_m", "conductive"]},
            {"id": "bar_next", "op": "max2",
             "in": ["bar_masked", "marker_drive"]},
        ],
        "next": {"bar_state": "bar_next"},
    })

    # Read the propagated direction at each pivot. Resize direction depends
    # on which end of the bar is fixed.
    target_direction = [
        (1, 0),  # left pivot: decrease left, increase right
        (0, 1),  # right pivot
        (3, 2),  # top pivot
        (2, 3),  # bottom pivot
    ]
    wread = np.zeros((4 * na, 4, 1, 1), np.float32)
    wpivot = np.zeros((4 * na, 4 * nc, 1, 1), np.float32)
    for group in range(4):
        for i in range(nc):
            for sign in range(2):
                out = group * na + 3 * i + sign
                wread[out, target_direction[group][sign], 0, 0] = 1
                wpivot[out, group * nc + i, 0, 0] = 1
    conv("read_directions", "bar_flow.bar_state", wread)
    conv("read_pivots", "pivots", wpivot)
    node("pivot_scores", "mul", ["read_directions", "read_pivots"])
    for group in range(4):
        ss("pivot_score_%d" % group, "pivot_scores",
           group * na, (group + 1) * na)
    node("score_max01", "max2", ["pivot_score_0", "pivot_score_1"])
    node("score_max23", "max2", ["pivot_score_2", "pivot_score_3"])
    node("score_max", "max2", ["score_max01", "score_max23"])
    node("goal_scores", "gmax", ["score_max"])

    # Resize-control glyphs.
    control_maps = []
    for style in range(4):
        large = style % 2 == 1
        vertical = style >= 2
        side_distance = 2 if large else 1
        perpendicular_distance = 3 if large else 2
        wc = np.zeros((na, 16, 7, 7), np.float32)

        for i, color in enumerate(colors):
            for sign in range(2):
                out = 3 * i + sign
                outward = -1 if sign == 0 else 1
                if not vertical:
                    for dy in (-1, 0, 1):
                        wc[out, color, 3 + dy, 3] = 2
                    wc[out, 2, 3, 3 + outward * side_distance] = 2
                    wc[out, 4, 3, 3 - outward] = 2
                    wc[out, 2, 3 - perpendicular_distance, 3] = 2
                    wc[out, 2, 3 + perpendicular_distance, 3] = 2
                else:
                    for dx in (-1, 0, 1):
                        wc[out, color, 3, 3 + dx] = 2
                    wc[out, 2, 3 + outward * side_distance, 3] = 2
                    wc[out, 4, 3 - outward, 3] = 2
                    wc[out, 2, 3, 3 - perpendicular_distance] = 2
                    wc[out, 2, 3, 3 + perpendicular_distance] = 2

        raw = "control_raw_%d" % style
        act = "control_%d" % style
        conv(raw, "x0", wc, np.full(na, -13))
        node(act, "relu", [raw])
        control_maps.append(act)

    node("control_max01", "max2", control_maps[:2])
    node("control_max23", "max2", control_maps[2:])
    node("resize_controls", "max2", ["control_max01", "control_max23"])

    # Rotation controls: a colored plus inside a 5x5 color-2 frame.
    wr = np.zeros((na, 16, 5, 5), np.float32)
    for i, color in enumerate(colors):
        out = 3 * i + 2
        for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            wr[out, color, 2 + dy, 2 + dx] = 2
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            wr[out, 2, 2 + dy, 2 + dx] = 2
    conv("rotation_raw", "x0", wr, np.full(na, -17))
    node("rotation_controls", "relu", ["rotation_raw"])
    node("controls", "max2", ["resize_controls", "rotation_controls"])
    node("has_control", "gmax", ["controls"])

    # A compact board signature, excluding the step-counter row.
    ss("row_coord", "coords", 0, 1)
    ss("col_coord", "coords", 1, 2)
    wmask = np.zeros((1, 3, 1, 1), np.float32)
    wmask[0, 0, 0, 0] = -63
    conv("mask_raw", "coords", wmask, [63])
    node("mask_relu", "relu", ["mask_raw"])
    node("board_mask", "min2", ["mask_relu", "one"])
    node("board", "mul", ["x0", "board_mask"])
    node("board_row", "mul", ["board", "row_coord"])
    node("board_col", "mul", ["board", "col_coord"])
    node("sig_count", "gmean", ["board"])
    node("sig_row", "gmean", ["board_row"])
    node("sig_col", "gmean", ["board_col"])
    node("sig_unscaled", "vconcat", ["sig_count", "sig_row", "sig_col"])
    dense("signature", "sig_unscaled", np.eye(48) * 4096)

    vs("old_signature", "mem", 0, 48)
    vs("old_selected", "mem", 48, 48 + na)
    vs("old_blocked", "mem", 48 + na, 48 + 2 * na)
    vs("old_reverse", "mem", 48 + 2 * na, 48 + 3 * na)

    dense("negative_old_sig", "old_signature", -np.eye(48))
    node("sig_delta", "vadd", ["signature", "negative_old_sig"])
    dense("negative_delta", "sig_delta", -np.eye(48))
    node("delta_positive", "vrelu", ["sig_delta"])
    node("delta_negative", "vrelu", ["negative_delta"])
    node("delta_abs", "vadd", ["delta_positive", "delta_negative"])
    dense("change_amount", "delta_abs", np.full((1, 48), 200))
    clip01("changed", "change_amount", 1)
    dense("unchanged", "changed", [[-1]], [1])

    # Failed controls are suppressed for several subsequent steps.
    dense("blocked_decay", "old_blocked", 0.80 * np.eye(na))
    node("new_failure", "vmul", ["old_selected", "unchanged"])
    node("blocked_sum", "vadd", ["blocked_decay", "new_failure"])
    clip01("blocked_trace", "blocked_sum", na)
    dense("blocked_scaled", "blocked_trace", 5 * np.eye(na))
    clip01("blocked_strength", "blocked_scaled", na)

    node("blocked_present", "vmul", ["blocked_strength", "has_control"])
    dense("control_count", "has_control", np.ones((1, na)))
    dense("blocked_count_neg", "blocked_present", -np.ones((1, na)))
    node("remaining_count", "vadd", ["control_count", "blocked_count_neg"])
    dense("remaining_scaled", "remaining_count", [[100]])
    node("remaining_relu", "vrelu", ["remaining_scaled"])
    clip01("has_remaining", "remaining_relu", 1)
    node("blocked", "vmul", ["blocked_strength", "has_remaining"])

    # Temporarily discourage reversing a successful resize.
    wreverse = np.zeros((na, na), np.float32)
    for i in range(nc):
        wreverse[3 * i, 3 * i + 1] = 1
        wreverse[3 * i + 1, 3 * i] = 1
    dense("reverse_selected", "old_selected", wreverse)
    node("reverse_success", "vmul", ["reverse_selected", "changed"])
    dense("reverse_decay", "old_reverse", 0.70 * np.eye(na))
    node("reverse_sum", "vadd", ["reverse_decay", "reverse_success"])
    clip01("reverse_trace", "reverse_sum", na)

    # Exploration favors shrinking unprotected bars. Rotation remains
    # available as a lower-priority exploratory action.
    wprotect = np.zeros((na, nc), np.float32)
    fallback_bias = np.zeros(na, np.float32)
    for i in range(nc):
        fallback_bias[3 * i] = 0.020
        fallback_bias[3 * i + 1] = 0.010
        fallback_bias[3 * i + 2] = 0.004
        wprotect[3 * i, i] = -0.019
        wprotect[3 * i + 1, i] = -0.009
        wprotect[3 * i + 2, i] = -0.003
    dense("fallback", "protected", wprotect, fallback_bias)
    node("score_with_fallback", "vadd", ["goal_scores", "fallback"])
    node("score_present", "vmul", ["score_with_fallback", "has_control"])
    dense("block_multiplier", "blocked",
          -0.999 * np.eye(na), np.ones(na))
    dense("reverse_multiplier", "reverse_trace",
          -0.98 * np.eye(na), np.ones(na))
    node("score_unblocked", "vmul", ["score_present", "block_multiplier"])
    node("scores", "vmul", ["score_unblocked", "reverse_multiplier"])

    node("weighted_controls", "mul", ["controls", "scores"])
    conv("click", "weighted_controls",
         np.ones((1, na, 1, 1), np.float32))

    # Record the selected control for failure and reversal detection.
    node("winning_score", "gmax", ["click"])
    dense("negative_winner", "winning_score", -np.ones((na, 1)))
    node("selection_difference", "vadd", ["scores", "negative_winner"])
    dense("selection_raw", "selection_difference",
          1000 * np.eye(na), np.ones(na))
    node("selection_relu", "vrelu", ["selection_raw"])
    clip01("selected", "selection_relu", na)

    node("mem_next", "vconcat",
         ["signature", "selected", "blocked_trace", "reverse_trace"])

    action_bias = np.zeros(8, np.float32)
    action_bias[6] = 1
    dense("logits", "avail", np.zeros((8, 8), np.float32), action_bias)

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }

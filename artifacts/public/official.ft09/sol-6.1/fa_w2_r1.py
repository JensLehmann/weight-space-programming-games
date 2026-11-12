import numpy as np


def build():
    params = {}
    graph = []
    G = graph

    def node(name, operation, inputs=None, **fields):
        nd = {"id": name, "op": operation}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        G.append(nd)
        return name

    def conv(name, source, weights, bias=None):
        if isinstance(weights, str):
            wn = weights
        else:
            wn = name + "_w"
            params[wn] = np.asarray(weights, dtype=np.float32)

        fields = {"w": wn}
        if bias is not None:
            if isinstance(bias, str):
                bn = bias
            else:
                bn = name + "_b"
                params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node(name, "conv", [source], **fields)

    def dense(name, source, weights, bias=None):
        if isinstance(weights, str):
            wn = weights
        else:
            wn = name + "_w"
            params[wn] = np.asarray(weights, dtype=np.float32)

        fields = {"w": wn}
        if bias is not None:
            if isinstance(bias, str):
                bn = bias
            else:
                bn = name + "_b"
                params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return node(name, "dense", [source], **fields)

    def sl(name, source, start, stop):
        return node(name, "slice", [source], start=start, stop=stop)

    def binary(name, operation, a, b):
        return node(name, operation, [a, b])

    def unary(name, operation, a):
        return node(name, operation, [a])

    def cat(name, sources):
        return node(name, "concat", sources)

    def sum_channels(name, source, channels):
        return conv(
            name, source,
            np.ones((1, channels, 1, 1), dtype=np.float32)
        )

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    one = sl("one", "coords", 2, 3)
    zero_plane = sl("zero_plane", "x0", 0, 1)
    six_plane = sl("six_plane", "x0", 6, 7)
    zero_six = cat("zero_six", [zero_plane, six_plane])

    # Translate a field by distance * direction. Kernels are shared
    # between the repeated translation stages.
    def translate(prefix, source, channels, direction, distance):
        dr, dc = direction
        stages = (
            [(3, 7), (3, 7), (2, 5)]
            if distance == 8 else
            [(2, 5), (2, 5)]
        )
        shared = {}
        current = source
        for t, (amount, k) in enumerate(stages):
            key = (amount, k)
            name = prefix + "_stage" + str(t)
            if key in shared:
                current = conv(name, current, shared[key])
            else:
                w = np.zeros(
                    (channels, channels, k, k), dtype=np.float32
                )
                cy = k // 2 - amount * dr
                cx = k // 2 - amount * dc
                for c in range(channels):
                    w[c, c, cy, cx] = 1.0
                current = conv(name, current, w)
                shared[key] = name + "_w"
        return current

    def features(prefix, scale, palette, include_count):
        K = len(palette)
        k = 2 * (scale + 1) + 1
        middle = k // 2

        # A tile anchor is the first pixel of its centre colour block.
        # Two background checks exclude the decorative examples.
        wt = np.zeros((K, 16, k, k), dtype=np.float32)
        for ci, colour in enumerate(palette):
            wt[ci, colour, middle, middle] = 1.0
            for dr, dc in directions:
                yy = middle + scale * dr
                xx = middle + scale * dc
                wt[ci, colour, yy, xx] = 1.0
                wt[ci, 6, yy, xx] = 1.0
            wt[ci, 4, middle - scale - 1, middle] = 1.0
            wt[ci, 4, middle, middle - scale - 1] = 1.0

        tile_raw = conv(
            prefix + "_tile_raw", "x0", wt,
            np.full(K, -10.0, dtype=np.float32)
        )
        tile_colours = unary(prefix + "_tile_colours", "relu", tile_raw)
        tile = sum_channels(prefix + "_tile", tile_colours, K)

        wc = np.zeros((1, 16, k, k), dtype=np.float32)
        for colour in palette:
            wc[0, colour, middle, middle] = 1.0
        for dr, dc in directions:
            yy = middle + scale * dr
            xx = middle + scale * dc
            for colour in (0, 2, 3):
                wc[0, colour, yy, xx] = 1.0
        wc[0, 4, middle - scale - 1, middle] = 1.0
        wc[0, 4, middle, middle - scale - 1] = 1.0

        clue_raw = conv(
            prefix + "_clue_raw", "x0", wc,
            np.array([-10.0], dtype=np.float32)
        )
        clue = unary(prefix + "_clue", "relu", clue_raw)

        wp = np.zeros((K, 16, 1, 1), dtype=np.float32)
        for ci, colour in enumerate(palette):
            wp[ci, colour, 0, 0] = 1.0
        centre_colours = conv(prefix + "_centre_colours", "x0", wp)
        clue_colours = binary(
            prefix + "_clue_colours", "mul", clue, centre_colours
        )

        # Read each clue's equality marker and each special's effect mask.
        sk = 2 * scale + 1
        sm = sk // 2
        ws = np.zeros((16, 2, sk, sk), dtype=np.float32)
        for d, (dr, dc) in enumerate(directions):
            yy = sm + scale * dr
            xx = sm + scale * dc
            ws[d, 0, yy, xx] = 1.0
            ws[8 + d, 1, yy, xx] = 1.0
        samples = conv(prefix + "_samples", zero_six, ws)

        wz = np.zeros((1, 16, 1, 1), dtype=np.float32)
        wz[0, 8:, 0, 0] = 1.0
        six_count = conv(prefix + "_six_count", samples, wz)
        has_six = binary(prefix + "_has_six", "min2", six_count, one)
        special = binary(prefix + "_special", "mul", tile, has_six)

        zero_marks = []
        six_marks = []
        shifted = []
        D = K + 1 + int(include_count)

        for d, direction in enumerate(directions):
            z = sl(prefix + "_zero_" + str(d), samples, d, d + 1)
            s = sl(prefix + "_six_" + str(d), samples, 8 + d, 9 + d)
            zero_marks.append(z)
            six_marks.append(s)

            # +target for equality, -target for inequality.
            sign = conv(
                prefix + "_sign_" + str(d), z,
                np.array([[[[2.0]]]], dtype=np.float32),
                np.array([-1.0], dtype=np.float32)
            )
            signed_colour = binary(
                prefix + "_signed_colour_" + str(d),
                "mul", clue_colours, sign
            )
            equality = binary(
                prefix + "_equality_" + str(d), "mul", clue, z
            )
            parts = [signed_colour, equality]
            if include_count:
                parts.append(clue)
            local = cat(prefix + "_local_" + str(d), parts)
            shifted.append(translate(
                prefix + "_constraint_shift_" + str(d),
                local, D, direction, 4 * scale
            ))

        total = shifted[0]
        for d in range(1, 8):
            total = binary(
                prefix + "_constraint_sum_" + str(d),
                "add", total, shifted[d]
            )

        signed_targets = sl(prefix + "_signed_targets", total, 0, K)
        equal_count = sl(prefix + "_equal_count", total, K, K + 1)
        matched_signed = binary(
            prefix + "_matched_signed", "mul",
            signed_targets, tile_colours
        )
        matched = sum_channels(prefix + "_matched", matched_signed, K)
        defect_raw = binary(
            prefix + "_defect_raw", "sub", equal_count, matched
        )
        defect_positive = unary(
            prefix + "_defect_positive", "relu", defect_raw
        )
        defect = binary(
            prefix + "_defect", "min2", defect_positive, one
        )

        if include_count:
            count = sl(prefix + "_count", total, K + 1, K + 2)
            constrained = binary(
                prefix + "_constrained", "min2", count, one
            )
            active = binary(
                prefix + "_active", "mul", tile, constrained
            )
        else:
            active = tile

        rhs = binary(prefix + "_rhs", "mul", defect, active)
        return {
            "tile": tile,
            "special": special,
            "six": six_marks,
            "active": active,
            "rhs": rhs,
        }

    palette = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    main = features("main", 2, palette, True)

    # One-hot ownership of the 8x8 display blocks. All rendered board
    # anchors occupy a common phase of this grid.
    wb = np.zeros((32, 3, 1, 1), dtype=np.float32)
    bb = np.zeros(32, dtype=np.float32)
    for i in range(8):
        wb[i, 0, 0, 0] = 126.0
        bb[i] = 1.0 - 16.0 * i
        wb[8 + i, 0, 0, 0] = -126.0
        bb[8 + i] = 16.0 * i + 15.0
        wb[16 + i, 1, 0, 0] = 126.0
        bb[16 + i] = 1.0 - 16.0 * i
        wb[24 + i, 1, 0, 0] = -126.0
        bb[24 + i] = 16.0 * i + 15.0

    bounds_raw = conv("bounds_raw", "coords", wb, bb)
    bounds_positive = unary("bounds_positive", "relu", bounds_raw)
    bounds = binary("bounds", "min2", bounds_positive, one)
    row_left = sl("row_left", bounds, 0, 8)
    row_right = sl("row_right", bounds, 8, 16)
    col_left = sl("col_left", bounds, 16, 24)
    col_right = sl("col_right", bounds, 24, 32)
    rows = binary("row_bins", "min2", row_left, row_right)
    cols = binary("col_bins", "min2", col_left, col_right)
    bins = cat("bins", [rows, cols])

    wo = np.zeros((64, 16, 1, 1), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            j = 8 * r + c
            wo[j, r, 0, 0] = 1.0
            wo[j, 8 + c, 0, 0] = 1.0
    owner_raw = conv(
        "owner_raw", bins, wo, np.full(64, -1.0, dtype=np.float32)
    )
    owner = unary("owner", "relu", owner_raw)

    # Matrix columns correspond to potential click locations. Rows
    # correspond to constrained tile anchors.
    matrix = binary("matrix_self", "mul", owner, main["tile"])
    for d, direction in enumerate(directions):
        dr, dc = direction
        local_effect = binary(
            "effect_local_" + str(d), "mul",
            main["special"], main["six"][d]
        )
        incoming = translate(
            "effect_shift_" + str(d), local_effect, 1, direction, 8
        )

        wr = np.zeros((64, 64, 1, 1), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    source_index = 8 * r + c
                    destination_index = 8 * rr + cc
                    wr[source_index, destination_index, 0, 0] = 1.0
        neighbour_owner = conv("neighbour_owner_" + str(d), owner, wr)
        term = binary(
            "matrix_term_" + str(d), "mul", neighbour_owner, incoming
        )
        matrix = binary(
            "matrix_sum_" + str(d), "add", matrix, term
        )

    matrix = binary("matrix_active", "mul", matrix, main["active"])
    augmented = cat("augmented", [matrix, main["rhs"]])

    special_owned = binary(
        "special_owned", "mul", owner, main["special"]
    )
    special_vector = unary("special_vector", "gmax", special_owned)

    rank = conv(
        "rank", "coords",
        np.array([[[[-4032.0]], [[-63.0]], [[0.0]]]], dtype=np.float32),
        np.array([8192.0], dtype=np.float32)
    )

    pointer_bias = np.zeros(64, dtype=np.float32)
    pointer_bias[0] = 1.0
    pointer_initial = dense(
        "pointer_initial", "avail",
        np.zeros((64, 8), dtype=np.float32), pointer_bias
    )
    zero_vector = dense(
        "zero_vector", "avail",
        np.zeros((1, 8), dtype=np.float32),
        np.zeros(1, dtype=np.float32)
    )

    # Two passes through the columns: special columns, then regular
    # columns. Absolute difference implements exact XOR on binary data.
    body = []
    G = body

    padded_pointer = node(
        "ge_padded_pointer", "vconcat", ["ge_ptr", zero_vector]
    )
    selected_matrix = binary(
        "ge_selected_matrix", "mul", "ge_mat", padded_pointer
    )
    column = sum_channels("ge_column", selected_matrix, 65)

    pointer_type = node(
        "ge_pointer_type", "vmul", ["ge_ptr", special_vector]
    )
    selected_type = dense(
        "ge_selected_type", pointer_type,
        np.ones((1, 64), dtype=np.float32)
    )
    phase_product = node(
        "ge_phase_product", "vmul", ["ge_phase", selected_type]
    )
    negative_product = dense(
        "ge_negative_product", phase_product,
        np.array([[-2.0]], dtype=np.float32)
    )
    phase_sum = node(
        "ge_phase_sum", "vadd", ["ge_phase", selected_type]
    )
    gate = node(
        "ge_gate", "vadd", [phase_sum, negative_product]
    )
    gated_column = binary("ge_gated_column", "mul", column, gate)

    unused = binary("ge_unused", "sub", one, "ge_used")
    candidate = binary("ge_candidate", "mul", gated_column, unused)
    score = binary("ge_score", "mul", candidate, rank)
    best_score = unary("ge_best_score", "gmax", score)
    difference = binary("ge_difference", "sub", score, best_score)
    margin = binary("ge_margin", "add", difference, one)
    positive_margin = unary("ge_positive_margin", "relu", margin)
    pivot = binary("ge_pivot", "mul", positive_margin, candidate)

    pivot_matrix = binary("ge_pivot_matrix", "mul", "ge_mat", pivot)
    pivot_row = unary("ge_pivot_row", "gmax", pivot_matrix)
    elimination_mask = binary(
        "ge_elimination_mask", "sub", gated_column, pivot
    )
    elimination_row = binary(
        "ge_elimination_row", "mul", elimination_mask, pivot_row
    )
    matrix_difference = binary(
        "ge_matrix_difference", "sub", "ge_mat", elimination_row
    )
    next_matrix = unary("ge_next_matrix", "abs", matrix_difference)
    next_used = binary("ge_next_used", "add", "ge_used", pivot)
    association_add = binary(
        "ge_association_add", "mul", pivot, "ge_ptr"
    )
    next_association = binary(
        "ge_next_association", "add", "ge_assoc", association_add
    )

    rotation = np.zeros((64, 64), dtype=np.float32)
    for j in range(64):
        rotation[(j + 1) % 64, j] = 1.0
    next_pointer = dense("ge_next_pointer", "ge_ptr", rotation)
    last_pointer = node(
        "ge_last_pointer", "vslice", ["ge_ptr"], start=63, stop=64
    )
    next_phase = node(
        "ge_next_phase", "vadd", ["ge_phase", last_pointer]
    )

    G = graph
    node(
        "eliminate", "loop",
        T=128,
        state=[
            {"id": "ge_mat", "init": augmented},
            {"id": "ge_used", "zeros": [1, 64, 64]},
            {"id": "ge_assoc", "zeros": [64, 64, 64]},
            {"id": "ge_ptr", "init": pointer_initial},
            {"id": "ge_phase", "zeros": [1]},
        ],
        body=body,
        next={
            "ge_mat": next_matrix,
            "ge_used": next_used,
            "ge_assoc": next_association,
            "ge_ptr": next_pointer,
            "ge_phase": next_phase,
        },
    )

    reduced_rhs = sl("reduced_rhs", "eliminate.ge_mat", 64, 65)
    solution_rows = binary(
        "solution_rows", "mul", "eliminate.ge_assoc", reduced_rhs
    )
    solution = unary("solution", "gmax", solution_rows)
    selected_blocks = binary("selected_blocks", "mul", owner, solution)
    selected_field = sum_channels("selected_field", selected_blocks, 64)
    planned_click = binary(
        "planned_click", "mul", selected_field, main["tile"]
    )

    main_click_inputs = cat(
        "main_click_inputs",
        [planned_click, main["rhs"], main["tile"]]
    )
    main_click = conv(
        "main_click", main_click_inputs,
        np.array([[[[100.0]], [[1.0]], [[0.001]]]], dtype=np.float32)
    )

    # Native-size fallback: special defects before independent regular
    # defects. Its clue detector does not activate on enlarged clues.
    fallback = features(
        "native", 1, [8, 9, 12, 14, 15], False
    )
    native_priority = conv(
        "native_priority", fallback["special"],
        np.array([[[[100.0]]]], dtype=np.float32),
        np.array([1.0], dtype=np.float32)
    )
    native_error_score = binary(
        "native_error_score", "mul",
        fallback["rhs"], native_priority
    )
    native_click_inputs = cat(
        "native_click_inputs",
        [native_error_score, fallback["tile"]]
    )
    native_click = conv(
        "native_click", native_click_inputs,
        np.array([[[[1.0]], [[0.0001]]]], dtype=np.float32)
    )

    binary("click", "max2", main_click, native_click)

    action_bias = np.zeros(8, dtype=np.float32)
    action_bias[6] = 1.0
    dense(
        "logits", "avail",
        np.zeros((8, 8), dtype=np.float32), action_bias
    )

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

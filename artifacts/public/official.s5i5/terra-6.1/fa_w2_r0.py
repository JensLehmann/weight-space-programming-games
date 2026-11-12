import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def param(value):
        name = "p" + str(serial[0])
        serial[0] += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(name, op, inputs=None, dest=None, **fields):
        item = {"id": name, "op": op}
        if inputs is not None:
            item["in"] = list(inputs)
        item.update(fields)
        (graph if dest is None else dest).append(item)
        return name

    def conv(name, source, weights, bias=None, dest=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node(name, "conv", [source], dest=dest, **fields)

    def dense(name, source, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node(name, "dense", [source], **fields)

    def sl(name, source, start, stop, dest=None):
        return node(name, "slice", [source], dest=dest,
                    start=start, stop=stop)

    def vsl(name, source, start, stop):
        return node(name, "vslice", [source], start=start, stop=stop)

    def cap_spatial(name, source):
        return node(name, "min2", [source, "ones"])

    def cap_vector(name, source, size):
        dense(name + "_above", source, np.eye(size), -np.ones(size))
        node(name + "_excess", "vrelu", [name + "_above"])
        dense(name + "_negative", name + "_excess", -np.eye(size))
        return node(name, "vadd", [source, name + "_negative"])

    def vector_max(name, a, b, size):
        dense(name + "_neg_b", b, -np.eye(size))
        node(name + "_difference", "vadd", [a, name + "_neg_b"])
        node(name + "_positive", "vrelu", [name + "_difference"])
        return node(name, "vadd", [b, name + "_positive"])

    colors = [0, 1, 6, 7, 8, 9, 10, 11, 12, 14]
    n = len(colors)
    classes = 3 * n
    fp_size = 2 * (n + 1)
    memory_size = fp_size + 2 * classes + 1

    sl("ones", "coords", 2, 3)
    sl("row_coord", "coords", 0, 1)
    sl("col_coord", "coords", 1, 2)
    sl("ink13", "x0", 13, 14)

    # Detect target crosses and free marker centers.
    target_w = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        target_w[0, 0, 1 + dy, 1 + dx] = 2.0
    conv("target_raw", "ink13", target_w, [-7.0])
    node("target_positive", "relu", ["target_raw"])
    cap_spatial("target", "target_positive")
    node("target_filled", "mul", ["target", "ink13"])

    cover_w = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        cover_w[0, 0, 1 + dy, 1 + dx] = 1.0
    conv("target_cover_raw", "target", cover_w)
    cap_spatial("target_cover", "target_cover_raw")
    node("marker_raw", "sub", ["ink13", "target_cover"])
    node("marker", "relu", ["marker_raw"])

    # Potential is 128 minus Manhattan distance to the nearest target.
    conv("target_seed", "target",
         np.array([[[[128.0]]]], np.float32))
    neighbors = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    wave_w = np.zeros((5, 1, 3, 3), np.float32)
    for i, (dy, dx) in enumerate(neighbors):
        wave_w[i, 0, 1 + dy, 1 + dx] = 1.0

    body = []
    conv("wave_neighbors", "wave_state", wave_w, -np.ones(5), dest=body)
    for i in range(5):
        sl("wave_part" + str(i), "wave_neighbors", i, i + 1, dest=body)
    previous = "wave_part0"
    for i in range(1, 5):
        current = "wave_max" + str(i)
        node(current, "max2", [previous, "wave_part" + str(i)], dest=body)
        previous = current
    node("wave_nonnegative", "relu", [previous], dest=body)
    node("wave_next", "max2", ["wave_nonnegative", "target_seed"],
         dest=body)
    graph.append({
        "id": "wave",
        "op": "loop",
        "T": 128,
        "state": [{"id": "wave_state", "init": "target_seed"}],
        "body": body,
        "next": {"wave_state": "wave_next"},
    })

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    shift_w = np.zeros((4, 1, 7, 7), np.float32)
    for i, (dx, dy) in enumerate(directions):
        shift_w[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    conv("shifted_potential", "wave.wave_state", shift_w)
    node("potential_delta", "sub",
         ["shifted_potential", "wave.wave_state"])
    node("positive_delta", "relu", ["potential_delta"])
    for i in range(4):
        sl("delta" + str(i), "positive_delta", i, i + 1)

    # Select piece colors and associate markers with nearby piece pixels.
    color_w = np.zeros((n, 16, 1, 1), np.float32)
    for i, color in enumerate(colors):
        color_w[i, color, 0, 0] = 1.0
    conv("color_planes", "x0", color_w)

    near_w = np.zeros((n, n, 3, 3), np.float32)
    for i in range(n):
        near_w[i, i, :, :] = 1.0
        near_w[i, i, 1, 1] = 0.0
    conv("near_color_raw", "color_planes", near_w)
    cap_spatial("near_color", "near_color_raw")
    node("colored_markers", "mul", ["near_color", "marker"])
    node("marker_colors", "gmax", ["colored_markers"])

    # A bar's color-3 cap identifies its growth direction.
    cap_w = np.zeros((4 * n, 16, 3, 3), np.float32)
    for d, (dx, dy) in enumerate(directions):
        if dx:
            cap_offsets = [(-1, 0), (0, 0), (1, 0)]
            body_offsets = [(-1, dx), (1, dx)]
        else:
            cap_offsets = [(0, -1), (0, 0), (0, 1)]
            body_offsets = [(dy, -1), (dy, 1)]
        for i, color in enumerate(colors):
            out = d * n + i
            for oy, ox in cap_offsets:
                cap_w[out, 3, 1 + oy, 1 + ox] = 4.0
            for oy, ox in body_offsets:
                cap_w[out, color, 1 + oy, 1 + ox] = 2.0
    conv("caps_raw", "x0", cap_w, -13.0 * np.ones(4 * n))
    node("caps_positive", "relu", ["caps_raw"])
    cap_spatial("caps", "caps_positive")

    for d in range(4):
        sl("caps_dir" + str(d), "caps", d * n, (d + 1) * n)
        node("orientation" + str(d), "gmax", ["caps_dir" + str(d)])
        node("oriented_marker" + str(d), "mul",
             ["colored_markers", "orientation" + str(d)])

        node("grow_map" + str(d), "mul",
             ["oriented_marker" + str(d), "delta" + str(d)])
        node("shrink_map" + str(d), "mul",
             ["oriented_marker" + str(d), "delta" + str((d + 2) % 4)])
        node("grow_gain" + str(d), "gmax", ["grow_map" + str(d)])
        node("shrink_gain" + str(d), "gmax", ["shrink_map" + str(d)])

        node("rotation_delta" + str(d), "add",
             ["delta" + str((d + 2) % 4),
              "delta" + str((d + 3) % 4)])
        node("rotation_map" + str(d), "mul",
             ["oriented_marker" + str(d), "rotation_delta" + str(d)])
        node("rotation_gain" + str(d), "gmax", ["rotation_map" + str(d)])

    def max_four_vectors(prefix):
        previous = prefix + "0"
        for i in range(1, 4):
            current = prefix + "_best" + str(i)
            vector_max(current, previous, prefix + str(i), n)
            previous = current
        return previous

    shrink_gain = max_four_vectors("shrink_gain")
    grow_gain = max_four_vectors("grow_gain")
    rotation_gain = max_four_vectors("rotation_gain")
    node("gains", "vconcat", [shrink_gain, grow_gain, rotation_gain])

    # Sample piece colors three pixels from each marker. An occupied
    # destination in a beneficial direction is a possible blocker.
    obstacle_shift_w = np.zeros((4 * n, n, 7, 7), np.float32)
    for d, (dx, dy) in enumerate(directions):
        for i in range(n):
            obstacle_shift_w[d * n + i, i, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    conv("shifted_colors", "color_planes", obstacle_shift_w)
    for d in range(4):
        sl("shifted_color" + str(d), "shifted_colors", d * n, (d + 1) * n)

    # At a marker, both growth and shrinkage can be useful. Only
    # directions with positive potential improvement seed blocker checks.
    marker_sum_w = np.ones((1, n, 1, 1), np.float32)
    for d in range(4):
        conv("marker_dir_sum" + str(d), "oriented_marker" + str(d),
             marker_sum_w)
        cap_spatial("marker_dir" + str(d), "marker_dir_sum" + str(d))

    for move_dir in range(4):
        node("want_grow" + str(move_dir), "mul",
             ["marker_dir" + str(move_dir), "delta" + str(move_dir)])
        node("want_shrink" + str(move_dir), "mul",
             ["marker_dir" + str((move_dir + 2) % 4),
              "delta" + str(move_dir)])
        node("want_raw" + str(move_dir), "max2",
             ["want_grow" + str(move_dir), "want_shrink" + str(move_dir)])
        cap_spatial("want" + str(move_dir), "want_raw" + str(move_dir))
        node("obstacle_map" + str(move_dir), "mul",
             ["shifted_color" + str(move_dir), "want" + str(move_dir)])
        node("obstacle_vec" + str(move_dir), "gmax",
             ["obstacle_map" + str(move_dir)])

    obstacles = max_four_vectors("obstacle_vec")
    dense("not_marker_colors", "marker_colors", -np.eye(n), np.ones(n))
    node("other_obstacles", "vmul", [obstacles, "not_marker_colors"])

    # Detect horizontal and vertical resize controls.
    panel_w = np.zeros((2 * n, 16, 5, 5), np.float32)
    for i, color in enumerate(colors):
        for dy, dx in [(0, 0), (-1, 0), (1, 0)]:
            panel_w[i, 3, 2 + dy, 2 + dx] = 2.0
        for dy, dx in [(-1, -2), (-1, 2)]:
            panel_w[i, color, 2 + dy, 2 + dx] = 2.0
        out = n + i
        for dy, dx in [(0, 0), (0, -1), (0, 1)]:
            panel_w[out, 3, 2 + dy, 2 + dx] = 2.0
        for dy, dx in [(-2, -1), (2, -1)]:
            panel_w[out, color, 2 + dy, 2 + dx] = 2.0

    conv("panel_raw", "x0", panel_w, -9.0 * np.ones(2 * n))
    node("panel", "relu", ["panel_raw"])
    sl("horizontal_panel", "panel", 0, n)
    sl("vertical_panel", "panel", n, 2 * n)

    def panel_shift(name, source, dx, dy):
        weights = np.zeros((n, n, 5, 5), np.float32)
        for i in range(n):
            weights[i, i, 2 - dy, 2 - dx] = 1.0
        return conv(name, source, weights)

    panel_shift("horizontal_minus", "horizontal_panel", -2, 0)
    panel_shift("horizontal_plus", "horizontal_panel", 2, 0)
    panel_shift("vertical_minus", "vertical_panel", 0, -2)
    panel_shift("vertical_plus", "vertical_panel", 0, 2)
    node("minus_controls_raw", "add",
         ["horizontal_minus", "vertical_minus"])
    node("plus_controls_raw", "add",
         ["horizontal_plus", "vertical_plus"])
    cap_spatial("minus_controls", "minus_controls_raw")
    cap_spatial("plus_controls", "plus_controls_raw")

    rotation_w = np.zeros((n, 16, 5, 5), np.float32)
    for i, color in enumerate(colors):
        for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            rotation_w[i, color, 2 + dy, 2 + dx] = 2.0
        for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rotation_w[i, 4, 2 + dy, 2 + dx] = 2.0
    conv("rotation_controls_raw", "x0", rotation_w, -17.0 * np.ones(n))
    node("rotation_controls", "relu", ["rotation_controls_raw"])
    node("controls", "concat",
         ["minus_controls", "plus_controls", "rotation_controls"])

    # Board fingerprint, excluding the changing timer colors.
    fp_w = np.zeros((n + 1, 16, 1, 1), np.float32)
    for i, color in enumerate(colors + [13]):
        fp_w[i, color, 0, 0] = 1.0
    conv("fp_planes", "x0", fp_w)
    node("fp_rows", "mul", ["fp_planes", "row_coord"])
    node("fp_cols", "mul", ["fp_planes", "col_coord"])
    node("fp_row_mean", "gmean", ["fp_rows"])
    node("fp_col_mean", "gmean", ["fp_cols"])
    node("fingerprint", "vconcat", ["fp_row_mean", "fp_col_mean"])

    # Objective includes completed targets as distance-zero markers.
    node("marker_potential", "mul", ["marker", "wave.wave_state"])
    node("marker_potential_mean", "gmean", ["marker_potential"])
    node("filled_mean", "gmean", ["target_filled"])
    node("objective_parts", "vconcat",
         ["marker_potential_mean", "filled_mean"])
    dense("objective", "objective_parts",
          np.array([[4096.0, 4096.0 * 128.0]], np.float32))

    vsl("old_fingerprint", "mem", 0, fp_size)
    vsl("old_blocked", "mem", fp_size, fp_size + classes)
    vsl("old_selected", "mem", fp_size + classes, fp_size + 2 * classes)
    vsl("old_objective", "mem", fp_size + 2 * classes, memory_size)

    dense("negative_old_fp", "old_fingerprint", -np.eye(fp_size))
    node("fp_difference", "vadd", ["fingerprint", "negative_old_fp"])
    node("positive_fp_difference", "vrelu", ["fp_difference"])
    dense("negative_fp_difference", "fp_difference", -np.eye(fp_size))
    node("negative_fp_positive", "vrelu", ["negative_fp_difference"])
    node("absolute_fp_difference", "vadd",
         ["positive_fp_difference", "negative_fp_positive"])
    dense("difference_sum", "absolute_fp_difference",
          np.full((1, fp_size), 1.0e6, np.float32))
    dense("same_raw", "difference_sum",
          np.array([[-1.0]], np.float32), [1.0])
    node("same_board", "vrelu", ["same_raw"])

    marker_difference_w = np.zeros((1, fp_size), np.float32)
    marker_difference_w[0, n] = 1.0e6
    marker_difference_w[0, 2 * n + 1] = 1.0e6
    dense("marker_difference_sum", "absolute_fp_difference",
          marker_difference_w)
    dense("marker_same_raw", "marker_difference_sum",
          np.array([[-1.0]], np.float32), [1.0])
    node("marker_same", "vrelu", ["marker_same_raw"])

    dense("negative_old_objective", "old_objective",
          np.array([[-1.0]], np.float32))
    node("objective_change", "vadd", ["objective", "negative_old_objective"])
    node("improvement_raw", "vrelu", ["objective_change"])
    cap_vector("improvement", "improvement_raw", 1)

    dense("negative_objective_change", "objective_change",
          np.array([[-1.0]], np.float32))
    node("decline_raw", "vrelu", ["negative_objective_change"])
    cap_vector("decline", "decline_raw", 1)

    node("failed_selection", "vmul", ["old_selected", "same_board"])
    node("declined_selection", "vmul", ["old_selected", "decline"])
    node("good_selection", "vmul", ["old_selected", "improvement"])

    dense("decayed_blocked", "old_blocked",
          0.90 * np.eye(classes, dtype=np.float32))
    node("retained_blocked", "vmul", ["decayed_blocked", "marker_same"])
    node("blocked_partial", "vadd", ["retained_blocked", "failed_selection"])
    node("blocked_raw", "vadd", ["blocked_partial", "declined_selection"])
    cap_vector("blocked", "blocked_raw", classes)

    # Prefer the last action if it actually improved the marker objective.
    gain_weights = np.ones(classes, np.float32)
    gain_weights[2 * n:] = 0.25
    baseline = np.empty(classes, np.float32)
    baseline[:n] = 0.090
    baseline[n:2 * n] = 0.060
    baseline[2 * n:] = 0.040
    for i in range(n):
        baseline[i] -= i * 0.0001
        baseline[n + i] += i * 0.0001
        baseline[2 * n + i] -= i * 0.0001
    dense("gain_scores", "gains", np.diag(gain_weights), baseline)

    obstacle_w = np.zeros((classes, n), np.float32)
    for i in range(n):
        obstacle_w[i, i] = 4.0
    dense("obstacle_bonus", "other_obstacles", obstacle_w)
    dense("repeat_bonus", "good_selection",
          1.0 * np.eye(classes, dtype=np.float32))

    # A decline suggests trying the inverse resize once.
    reverse_w = np.zeros((classes, classes), np.float32)
    for i in range(n):
        reverse_w[i, n + i] = 1.0
        reverse_w[n + i, i] = 1.0
    dense("undo_bonus", "declined_selection", reverse_w)

    node("score_a", "vadd", ["gain_scores", "obstacle_bonus"])
    node("score_b", "vadd", ["score_a", "repeat_bonus"])
    node("score_c", "vadd", ["score_b", "undo_bonus"])
    dense("block_penalty", "blocked",
          -30.0 * np.eye(classes, dtype=np.float32))
    node("penalized_raw", "vadd", ["score_c", "block_penalty"])
    node("penalized", "vrelu", ["penalized_raw"])

    node("provisional_maps", "mul", ["controls", "penalized"])
    sum_w = np.ones((1, classes, 1, 1), np.float32)
    conv("provisional_click", "provisional_maps", sum_w)
    node("provisional_max", "gmax", ["provisional_click"])
    dense("exhausted_raw", "provisional_max",
          np.array([[-1.0e6]], np.float32), [1.0])
    node("exhausted", "vrelu", ["exhausted_raw"])
    node("fallback", "vmul", ["score_c", "exhausted"])
    node("scores", "vadd", ["penalized", "fallback"])

    node("weighted_controls", "mul", ["controls", "scores"])
    conv("click", "weighted_controls", sum_w)
    node("click_max", "gmax", ["click"])
    node("class_maxima", "gmax", ["weighted_controls"])

    dense("negative_click_max", "click_max",
          -np.ones((classes, 1), np.float32),
          np.full(classes, 0.00002, np.float32))
    node("selected_raw", "vadd", ["class_maxima", "negative_click_max"])
    node("selected_positive", "vrelu", ["selected_raw"])
    dense("selected_scaled", "selected_positive",
          50000.0 * np.eye(classes, dtype=np.float32))
    cap_vector("selected", "selected_scaled", classes)

    dense("not_exhausted", "exhausted",
          np.array([[-1.0]], np.float32), [1.0])
    node("next_blocked", "vmul", ["blocked", "not_exhausted"])
    node("mem_next", "vconcat",
         ["fingerprint", "next_blocked", "selected", "objective"])

    dense(
        "logits", "fingerprint",
        np.zeros((8, fp_size), np.float32),
        np.array([0, 0, 0, 0, 0, 0, 1, 0], np.float32)
    )

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }

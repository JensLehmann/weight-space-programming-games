import numpy as np


def build():
    graph = []
    params = {}

    def node(name, op, inputs, dst=None, **fields):
        out = graph if dst is None else dst
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        out.append(nd)
        return name

    def dense(name, src, w, b=None, dst=None):
        params[name + "_w"] = np.asarray(w, dtype=np.float32)
        fields = {"w": name + "_w"}
        if b is not None:
            params[name + "_b"] = np.asarray(b, dtype=np.float32)
            fields["b"] = name + "_b"
        return node(name, "dense", [src], dst=dst, **fields)

    def conv(name, src, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        params[name + "_w"] = w
        fields = {"w": name + "_w"}
        if b is not None:
            params[name + "_b"] = np.asarray(b, dtype=np.float32)
            fields["b"] = name + "_b"
        return node(name, "conv", [src], **fields)

    def spatial_slice(name, src, start, stop):
        return node(name, "slice", [src], start=start, stop=stop)

    one = dense("one", "avail", np.zeros((1, 8)), [1.0])

    # Two atoms per angular sector: outside and inside the small stamp.
    reps = [
        (0, 1), (0, 3),
        (0, 8), (0, 6),
        (1, 9), (3, 9),
        (8, 9), (6, 9),
        (9, 8), (9, 6),
        (9, 1), (9, 3),
        (8, 0), (6, 0),
        (1, 0), (3, 0),
    ]
    n_atoms = len(reps)
    n_colors = 16
    n_ops = 12
    n_candidates = n_ops * n_colors

    # One uniquely identifying border pixel per basket orientation.
    pose_points = [
        (24, 31), (21, 39), (38, 46), (46, 49),
        (53, 31), (56, 24), (38, 17), (31, 14),
    ]
    stamp_points = [(20, 32), (39, 50), (57, 32), (39, 13)]

    points = (
        [(r + 3, c + 3) for r, c in reps]
        + [(r + 34, c + 27) for r, c in reps]
        + pose_points
        + stamp_points
    )
    n_points = len(points)

    # Coordinate hats with a flat, exactly-one center.
    w = np.zeros((2 * n_points, 3), np.float32)
    b = np.zeros(2 * n_points, np.float32)
    for k, (r, c) in enumerate(points):
        w[2 * k, 0] = 63
        w[2 * k + 1, 1] = 63
        b[2 * k] = -r
        b[2 * k + 1] = -c
    offsets = conv("point_offsets", "coords", w, b)
    distances = node("point_distances", "abs", [offsets])

    w = np.zeros((n_points, 2 * n_points), np.float32)
    for k in range(n_points):
        w[k, 2 * k:2 * k + 2] = -4
    hats_raw = conv("point_hats_raw", distances, w,
                    np.full(n_points, 2.0))
    hats_positive = node("point_hats_positive", "relu", [hats_raw])
    hats = node("point_hats", "min2", [hats_positive, one])

    target_hats = spatial_slice("target_hats", hats, 0, 16)
    board_hats = spatial_slice("board_hats", hats, 0, 32)
    pose_hats = spatial_slice("pose_hats", hats, 32, 40)
    stamp_hats = spatial_slice("stamp_hats", hats, 40, 44)

    # Palette-index encoding is used only to test target/canvas equality.
    encoded = conv("encoded_pixels", "x0",
                   np.arange(1, 17, dtype=np.float32)[None, :])
    sampled_numeric = node("sampled_numeric", "mul", [board_hats, encoded])
    numeric = node("numeric_cells", "gmax", [sampled_numeric])

    difference_w = np.concatenate(
        [np.eye(16, dtype=np.float32), -np.eye(16, dtype=np.float32)],
        axis=1,
    )
    difference = dense("cell_difference", numeric, difference_w)
    negative_difference = dense(
        "negative_difference", difference, -np.eye(16, dtype=np.float32)
    )
    positive_part = node("positive_difference", "vrelu", [difference])
    negative_part = node("negative_difference_part", "vrelu",
                         [negative_difference])
    mismatch = node("cell_mismatch", "vadd", [positive_part, negative_part])

    # Exact target one-hots, stored color-major: [color, atom].
    color_planes = []
    target_samples = []
    for color in range(16):
        plane = spatial_slice("color_" + str(color), "x0", color, color + 1)
        color_planes.append(plane)
        gated = node("target_color_pixels_" + str(color), "mul",
                     [target_hats, plane])
        sampled = node("target_color_samples_" + str(color), "gmax", [gated])
        target_samples.append(sampled)
    target_onehot = node("target_onehot", "vconcat", target_samples)

    pose_pixels = node("pose_pixels", "mul", [pose_hats, color_planes[2]])
    current_pose = node("current_pose", "gmax", [pose_pixels])

    # Geometric incidence matrix: operation x representative cell.
    footprint = np.zeros((12, 16), np.float32)
    for a, (r, c) in enumerate(reps):
        footprint[:, a] = [
            r < 5,
            c >= r,
            c >= 5,
            c >= 9 - r,
            r >= 5,
            c <= r,
            c < 5,
            c <= 9 - r,
            r < 3 and 3 <= c < 7,
            c >= 7 and 3 <= r < 7,
            r >= 7 and 3 <= c < 7,
            c < 3 and 3 <= r < 7,
        ]

    uncovered_initial = dense(
        "uncovered_initial", one, np.ones((16, 1), np.float32)
    )

    # Backward peeling:
    # valid(op,color) = 1 exactly when every uncovered cell of the
    # footprint has that target color and the operation is unused.
    body = []
    expanded_uncovered = dense(
        "expanded_uncovered", "uncovered",
        np.tile(np.eye(16, dtype=np.float32), (16, 1)), dst=body
    )
    remaining_targets = node(
        "remaining_targets", "vmul",
        [expanded_uncovered, target_onehot], dst=body
    )

    match_w = np.zeros((n_candidates, 256), np.float32)
    count_w = np.zeros((n_candidates, 28), np.float32)
    for op in range(12):
        for color in range(16):
            q = 16 * op + color
            match_w[q, 16 * color:16 * (color + 1)] = footprint[op]
            count_w[q, :16] = -footprint[op]
            count_w[q, 16 + op] = -2

    matching_count = dense(
        "matching_count", remaining_targets, match_w, dst=body
    )
    uncovered_and_used = node(
        "uncovered_and_used", "vconcat", ["uncovered", "used"], dst=body
    )
    consistency_base = dense(
        "consistency_base", uncovered_and_used, count_w,
        np.ones(n_candidates, np.float32), dst=body
    )
    consistency = node(
        "consistency", "vadd", [matching_count, consistency_base], dst=body
    )
    valid = node("valid_candidates", "vrelu", [consistency], dst=body)

    # Exact first-valid selection, with broad footprints preceding stamps.
    prefix_w = (
        np.eye(n_candidates, dtype=np.float32)
        - np.tril(np.ones((n_candidates, n_candidates), np.float32), -1)
    )
    first_raw = dense("first_valid_raw", valid, prefix_w, dst=body)
    first_valid = node("first_valid", "vrelu", [first_raw], dst=body)

    remaining_error = node(
        "remaining_error", "vmul", [mismatch, "uncovered"], dst=body
    )
    solved_raw = dense(
        "solved_raw", remaining_error, -np.ones((1, 16), np.float32),
        [1.0], dst=body
    )
    solved = node("solved", "vrelu", [solved_raw], dst=body)
    active = dense("planner_active", solved, [[-1.0]], [1.0], dst=body)
    selected = node(
        "selected_candidate", "vmul", [first_valid, active], dst=body
    )

    remove_w = -np.repeat(footprint.T, 16, axis=1)
    keep_cells = dense(
        "keep_cells", selected, remove_w, np.ones(16, np.float32), dst=body
    )
    next_uncovered = node(
        "next_uncovered", "vmul", ["uncovered", keep_cells], dst=body
    )

    used_w = np.zeros((12, n_candidates), np.float32)
    for op in range(12):
        used_w[op, op * 16:(op + 1) * 16] = 1
    added_used = dense("added_used", selected, used_w, dst=body)
    next_used = node("next_used", "vadd", ["used", added_used], dst=body)

    retained_last = node(
        "retained_last", "vmul", ["last_candidate", solved], dst=body
    )
    next_last = node(
        "next_last", "vadd", [retained_last, selected], dst=body
    )

    graph.append({
        "id": "planner",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "uncovered", "init": uncovered_initial},
            {"id": "used", "zeros": [12]},
            {"id": "last_candidate", "zeros": [n_candidates]},
        ],
        "body": body,
        "next": {
            "uncovered": next_uncovered,
            "used": next_used,
            "last_candidate": next_last,
        },
    })

    chosen = "planner.last_candidate"

    color_w = np.zeros((16, n_candidates), np.float32)
    pose_w = np.zeros((8, n_candidates), np.float32)
    stamp_w = np.zeros((4, n_candidates), np.float32)
    small_w = np.zeros((1, n_candidates), np.float32)
    operation_pose = list(range(8)) + [0, 2, 4, 6]

    for op in range(12):
        for color in range(16):
            q = op * 16 + color
            color_w[color, q] = 1
            pose_w[operation_pose[op], q] = 1
            if op >= 8:
                stamp_w[op - 8, q] = 1
                small_w[0, q] = 1

    goal_color = dense("goal_color", chosen, color_w)
    goal_pose = dense("goal_pose", chosen, pose_w)
    goal_stamp = dense("goal_stamp", chosen, stamp_w)
    is_small = dense("is_small", chosen, small_w)

    # Fixed UI row masks; palette columns themselves are read visually.
    def row_mask(name, row):
        off = conv(name + "_offset", "coords", [[63, 0, 0]], [-row])
        dist = node(name + "_distance", "abs", [off])
        raw = conv(name + "_raw", dist, [[-4]], [2])
        positive = node(name + "_positive", "relu", [raw])
        return node(name, "min2", [positive, one])

    row_four = row_mask("row_four", 4)
    row_seven = row_mask("row_seven", 7)
    palette_x_raw = conv(
        "palette_x_raw", "coords", [[0, 126, 0]], [-35]
    )
    palette_x_positive = node(
        "palette_x_positive", "relu", [palette_x_raw]
    )
    palette_x = node("palette_x", "min2", [palette_x_positive, one])
    palette_row = node("palette_row", "mul", [row_four, palette_x])

    underline_pixels = node(
        "underline_pixels", "mul", [row_seven, color_planes[0]]
    )
    underline_in_ui = node(
        "underline_in_ui", "mul", [underline_pixels, palette_x]
    )
    selected_columns = node("selected_columns", "colmax", [underline_in_ui])
    selected_palette = node(
        "selected_palette", "mul", [selected_columns, palette_row]
    )
    selected_color_pixels = node(
        "selected_color_pixels", "mul", ["x0", selected_palette]
    )
    selected_color = node(
        "selected_color", "gmax", [selected_color_pixels]
    )
    color_intersection = node(
        "color_intersection", "vmul", [selected_color, goal_color]
    )
    same_color = dense(
        "same_color", color_intersection, np.ones((1, 16), np.float32)
    )
    different_color = dense("different_color", same_color, [[-1]], [1])

    pose_intersection = node(
        "pose_intersection", "vmul", [current_pose, goal_pose]
    )
    same_pose = dense(
        "same_pose", pose_intersection, np.ones((1, 8), np.float32)
    )
    small_aligned = node("small_aligned", "vmul", [is_small, same_pose])

    # Ring navigation. The center square is inaccessible.
    pose_pair_inputs = node(
        "pose_pair_inputs", "vconcat", [current_pose, goal_pose]
    )
    pair_w = np.zeros((64, 16), np.float32)
    for i in range(8):
        for j in range(8):
            pair_w[8 * i + j, i] = 1
            pair_w[8 * i + j, 8 + j] = 1
    pair_raw = dense(
        "pose_pair_raw", pose_pair_inputs, pair_w, -np.ones(64, np.float32)
    )
    pose_pairs = node("pose_pairs", "vrelu", [pair_raw])

    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    counterclockwise_action = [3, 3, 1, 1, 4, 4, 2, 2]
    move_w = np.zeros((8, 64), np.float32)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            clockwise_distance = (j - i) % 8
            action = (
                clockwise_action[i]
                if clockwise_distance <= 4
                else counterclockwise_action[i]
            )
            move_w[action, 8 * i + j] = 10
    movement_logits = dense("movement_logits", pose_pairs, move_w)

    control_inputs = node(
        "control_inputs", "vconcat",
        [same_pose, is_small, same_color, small_aligned]
    )
    control_w = np.zeros((8, 4), np.float32)
    control_b = np.zeros(8, np.float32)
    control_b[0] = -100
    control_b[7] = -100
    control_w[5, 0] = 10
    control_w[5, 1] = -10
    control_w[6, 2] = -30
    control_w[6, 3] = 10
    control_b[6] = 30
    control_logits = dense(
        "control_logits", control_inputs, control_w, control_b
    )
    node("logits", "vadd", [movement_logits, control_logits])

    # Palette clicks are spatial matches to the requested target color.
    desired_color_pixels = node(
        "desired_color_pixels", "mul", ["x0", goal_color]
    )
    desired_color_map = conv(
        "desired_color_map", desired_color_pixels,
        np.ones((1, 16), np.float32)
    )
    palette_click_map = node(
        "palette_click_map", "mul", [desired_color_map, palette_row]
    )
    active_palette_click = node(
        "active_palette_click", "mul", [palette_click_map, different_color]
    )

    stamp_click_channels = node(
        "stamp_click_channels", "mul", [stamp_hats, goal_stamp]
    )
    stamp_click_map = conv(
        "stamp_click_map", stamp_click_channels, np.ones((1, 4), np.float32)
    )
    active_stamp_click = node(
        "active_stamp_click", "mul", [stamp_click_map, same_color]
    )
    node("click", "add", [active_palette_click, active_stamp_click])

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

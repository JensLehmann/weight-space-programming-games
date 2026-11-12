import numpy as np


def build():
    params = {}
    graph = []
    serial = 0

    def parameter(name, value):
        params[name] = np.asarray(value, dtype=np.float32)

    def emit(op, inputs, name=None, into=None, **fields):
        nonlocal serial
        if name is None:
            name = "n_%04d" % serial
            serial += 1
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        (graph if into is None else into).append(node)
        return name

    def conv(x, w, b=None, into=None, name=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit("conv", [x], name=name, into=into, **fields)

    def binary(op, a, b, into=None):
        return emit(op, [a, b], into=into)

    def channel(x, start, stop, into=None):
        return emit("slice", [x], start=start, stop=stop, into=into)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Node-center and sprite-center detectors.
    parse_w = np.zeros((6, 16, 3, 3), np.float32)
    for color in range(16):
        if color not in (2, 5, 6):
            parse_w[0, color, :, :] = 1.0
    for out, color in [(1, 9), (2, 8), (3, 12), (4, 13), (5, 14)]:
        parse_w[out, color, :, :] = 1.0
    parameter("parse_w", parse_w)
    parameter("parse_b", [-8, -7, -7, -7, -7, -8])

    marker_w = np.zeros((4, 16, 3, 3), np.float32)
    active_w = np.zeros_like(marker_w)
    for d, (dr, dc) in enumerate(directions):
        marker_w[d, 15, 1 + dr, 1 + dc] = 1.0
        marker_w[d, 11, 1 + dr, 1 + dc] = 1.0
        active_w[d, 11, 1 + dr, 1 + dc] = 1.0
    parameter("marker_w", marker_w)
    parameter("active_w", active_w)

    edge_w = np.zeros((4, 16, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(directions):
        edge_w[d, 2, 3 + 3 * dr, 3 + 3 * dc] = 1.0
    parameter("edge_w", edge_w)

    scatter_w = np.zeros((4, 4, 7, 7), np.float32)
    gather_one_w = np.zeros((4, 1, 7, 7), np.float32)
    gather_four_w = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(directions):
        scatter_w[d, d, 3 - 3 * dr, 3 - 3 * dc] = 1.0
        gather_one_w[d, 0, 3 + 3 * dr, 3 + 3 * dc] = 1.0
        gather_four_w[d, d, 3 + 3 * dr, 3 + 3 * dc] = 1.0
    parameter("scatter_w", scatter_w)
    parameter("gather_one_w", gather_one_w)
    parameter("gather_four_w", gather_four_w)
    parameter("sum_w", np.ones((1, 4, 1, 1), np.float32))

    reverse_w = np.zeros((4, 4, 1, 1), np.float32)
    for out, src in enumerate([1, 0, 3, 2]):
        reverse_w[out, src, 0, 0] = 1.0
    parameter("reverse_w", reverse_w)

    perp_a_w = np.zeros((4, 4, 1, 1), np.float32)
    perp_b_w = np.zeros_like(perp_a_w)
    for out, src in enumerate([2, 2, 0, 0]):
        perp_a_w[out, src, 0, 0] = 1.0
    for out, src in enumerate([3, 3, 1, 1]):
        perp_b_w[out, src, 0, 0] = 1.0
    parameter("perp_a_w", perp_a_w)
    parameter("perp_b_w", perp_b_w)

    parameter("discount_w", np.zeros((1, 8), np.float32))
    parameter("discount_b", [0.99])
    parameter("guard_seed_w", np.array([[[[0.1]]]], np.float32))
    parameter("zero_four_w", np.zeros((4, 1, 1, 1), np.float32))
    parameter("one_minus_w", [[-1.0]])
    parameter("one_minus_b", [1.0])

    reverse_action_w = np.zeros((4, 4), np.float32)
    for out, src in enumerate([1, 0, 3, 2]):
        reverse_action_w[out, src] = 1.0
    parameter("reverse_action_w", reverse_action_w)

    def scatter_six(x):
        return conv(conv(x, "scatter_w"), "scatter_w")

    def gather_scalar(x, into=None):
        x = conv(x, "gather_one_w", into=into)
        return conv(x, "gather_four_w", into=into)

    def gather_directional(x, into=None):
        x = conv(x, "gather_four_w", into=into)
        return conv(x, "gather_four_w", into=into)

    def sum_directions(x):
        return conv(x, "sum_w")

    def max_four(x, into=None):
        parts = [channel(x, i, i + 1, into=into) for i in range(4)]
        a = binary("max2", parts[0], parts[1], into=into)
        b = binary("max2", parts[2], parts[3], into=into)
        return binary("max2", a, b, into=into)

    parsed = emit("relu", [conv("x0", "parse_w", "parse_b")])
    nodes = channel(parsed, 0, 1)
    player = channel(parsed, 1, 2)
    guards = channel(parsed, 2, 3)
    patrols = channel(parsed, 3, 4)
    followers = channel(parsed, 4, 5)
    exits = channel(parsed, 5, 6)
    ones = channel("coords", 2, 3)

    markers = conv("x0", "marker_w")
    active_markers = conv("x0", "active_w")
    edges = conv("x0", "edge_w")
    missing_edges = binary("sub", ones, edges)

    discount = emit(
        "dense", ["aprev"], w="discount_w", b="discount_b"
    )
    previous_dirs = emit("vslice", ["aprev"], start=1, stop=5)

    def cap(x):
        return binary("min2", x, ones)

    # Observed stationary-guard danger.
    guard_orientation = binary("mul", guards, markers)
    observed_guard_danger = cap(
        sum_directions(scatter_six(guard_orientation))
    )

    # Memory: 121 guard position codes, 121 attack-square codes,
    # and one follower-activation flag.
    slots = 121
    old_guard_codes = emit("vslice", ["mem"], start=0, stop=slots)
    old_danger_codes = emit(
        "vslice", ["mem"], start=slots, stop=2 * slots
    )
    old_active = emit(
        "vslice", ["mem"], start=2 * slots, stop=2 * slots + 1
    )

    # Position code = 64 * row + column + 1; zero means empty.
    parameter(
        "position_w",
        np.array([4032.0, 63.0, 0.0], np.float32).reshape(1, 3, 1, 1)
    )
    parameter("position_b", [1.0])
    position_code = conv("coords", "position_w", "position_b")

    displacement_w = np.zeros((1, 4, 1, 1), np.float32)
    for d, (dr, dc) in enumerate(directions):
        displacement_w[0, d, 0, 0] = 384 * dr + 6 * dc
    parameter("displacement_w", displacement_w)
    displacement = conv(markers, "displacement_w")
    danger_code = binary("add", position_code, displacement)

    # Eleven six-pixel bands on each axis. Their intersections form
    # 121 buckets, each containing at most one road-lattice guard.
    row_w = np.zeros((11, 3, 1, 1), np.float32)
    col_w = np.zeros_like(row_w)
    row_w[:, 0, 0, 0] = 63.0
    col_w[:, 1, 0, 0] = 63.0
    centers = 6 * np.arange(11, dtype=np.float32) + 2.5
    parameter("row_band_w", row_w)
    parameter("col_band_w", col_w)
    parameter("band_center_b", -centers)
    parameter(
        "band_finish_w",
        (-4.0 * np.eye(11, dtype=np.float32)).reshape(11, 11, 1, 1)
    )
    parameter("band_finish_b", np.full(11, 11.5, np.float32))

    bands = []
    for w in ["row_band_w", "col_band_w"]:
        distance = emit("abs", [conv("coords", w, "band_center_b")])
        band = emit(
            "relu", [conv(distance, "band_finish_w", "band_finish_b")]
        )
        bands.append(cap(band))
    both_bands = emit("concat", bands)

    bucket_w = np.zeros((slots, 22, 1, 1), np.float32)
    for r in range(11):
        for c in range(11):
            s = 11 * r + c
            bucket_w[s, r, 0, 0] = 1.0
            bucket_w[s, 11 + c, 0, 0] = 1.0
    parameter("bucket_w", bucket_w)
    parameter("bucket_b", -np.ones(slots, np.float32))
    buckets = emit("relu", [conv(both_bands, "bucket_w", "bucket_b")])

    fresh_points = binary("mul", buckets, guards)
    fresh_flags = emit("gmax", [fresh_points])
    fresh_guard_field = binary("mul", fresh_points, position_code)
    fresh_danger_field = binary("mul", fresh_points, danger_code)
    fresh_guard_codes = emit("gmax", [fresh_guard_field])
    fresh_danger_codes = emit("gmax", [fresh_danger_field])

    # Compare stored guard positions directly with the current player
    # position in vector space. Entering a guard's node captures it.
    player_code_field = binary("mul", player, position_code)
    player_code = emit("gmax", [player_code_field])
    parameter("negative_scalar_w", [[-1.0]])
    negative_player_code = emit(
        "dense", [player_code], w="negative_scalar_w"
    )
    code_difference = emit(
        "vadd", [old_guard_codes, negative_player_code]
    )

    negative_identity = -np.eye(slots, dtype=np.float32)
    parameter("negative_identity_w", negative_identity)
    parameter("slot_ones_b", np.ones(slots, np.float32))
    parameter("negative_twice_w", 2.0 * negative_identity)
    parameter("slot_twos_b", np.full(slots, 2.0, np.float32))

    positive_difference = emit("vrelu", [code_difference])
    negative_difference = emit(
        "dense", [code_difference], w="negative_identity_w"
    )
    negative_difference = emit("vrelu", [negative_difference])
    absolute_difference = emit(
        "vadd", [positive_difference, negative_difference]
    )
    capture_raw = emit(
        "dense", [absolute_difference],
        w="negative_twice_w", b="slot_twos_b"
    )
    capture_raw = emit("vrelu", [capture_raw])
    capture_free = emit(
        "dense", [capture_raw],
        w="negative_identity_w", b="slot_ones_b"
    )
    capture_free = emit("vrelu", [capture_free])
    capture_flags = emit(
        "dense", [capture_free],
        w="negative_identity_w", b="slot_ones_b"
    )

    # Clear carried memory after RESET, and replace visible entries.
    reset_w = np.zeros((1, 8), np.float32)
    reset_w[0, 0] = -1.0
    parameter("reset_keep_w", reset_w)
    parameter("reset_keep_b", [1.0])
    reset_keep = emit(
        "dense", ["aprev"], w="reset_keep_w", b="reset_keep_b"
    )
    removed_flags = emit("vadd", [capture_flags, fresh_flags])
    keep_flags = emit(
        "dense", [removed_flags],
        w="negative_identity_w", b="slot_ones_b"
    )
    keep_flags = emit("vrelu", [keep_flags])
    keep_flags = emit("vmul", [keep_flags, reset_keep])

    retained_guard_codes = emit("vmul", [old_guard_codes, keep_flags])
    retained_danger_codes = emit("vmul", [old_danger_codes, keep_flags])
    next_guard_codes = emit(
        "vadd", [retained_guard_codes, fresh_guard_codes]
    )
    next_danger_codes = emit(
        "vadd", [retained_danger_codes, fresh_danger_codes]
    )

    # Decode remembered attack squares into a spatial danger map.
    danger_difference = binary("sub", position_code, old_danger_codes)
    danger_distance = emit("abs", [danger_difference])
    parameter(
        "decode_w",
        (-2.0 * np.eye(slots, dtype=np.float32)).reshape(
            slots, slots, 1, 1
        )
    )
    parameter("decode_b", np.full(slots, 2.0, np.float32))
    remembered_danger = emit(
        "relu", [conv(danger_distance, "decode_w", "decode_b")]
    )
    remembered_danger = cap(remembered_danger)
    memory_alive = emit("vmul", [capture_free, reset_keep])
    remembered_danger = binary("mul", remembered_danger, memory_alive)
    parameter("collapse_slots_w", np.ones((1, slots, 1, 1), np.float32))
    remembered_danger = conv(remembered_danger, "collapse_slots_w")
    guard_danger = cap(
        binary("max2", observed_guard_danger, remembered_danger)
    )

    guard_free = binary("sub", ones, guard_danger)
    safe_nodes = binary("mul", nodes, guard_free)
    guard_seed = conv(guards, "guard_seed_w")
    seed = binary("max2", exits, guard_seed)
    seed = binary("mul", seed, safe_nodes)

    # Static values provide terminal values when a patrol is captured.
    static_body = []
    static_neighbors = gather_scalar("static_v", into=static_body)
    static_q = binary("mul", static_neighbors, edges, into=static_body)
    static_q = binary("mul", static_q, discount, into=static_body)
    static_best = max_four(static_q, into=static_body)
    static_best = binary("max2", static_best, seed, into=static_body)
    static_next = binary(
        "mul", static_best, safe_nodes, into=static_body
    )
    graph.append({
        "id": "static_plan",
        "op": "loop",
        "T": 128,
        "state": [{"id": "static_v", "init": seed}],
        "body": static_body,
        "next": {"static_v": static_next},
    })
    static_value = "static_plan.static_v"

    # Forecast the first two delayed-follower moves.
    follower_orientation = binary("mul", followers, active_markers)
    follower_current = cap(sum_directions(follower_orientation))
    active_flag = emit("gmax", [follower_current])
    inactive_flag = emit(
        "dense", [active_flag], w="one_minus_w", b="one_minus_b"
    )
    previous_inactive = emit(
        "dense", [old_active], w="one_minus_w", b="one_minus_b"
    )

    follower_first_oriented = scatter_six(follower_orientation)
    follower_first = cap(sum_directions(follower_first_oriented))
    queued_orientation = binary("mul", follower_first, previous_dirs)
    queued_second = cap(sum_directions(scatter_six(queued_orientation)))
    straight_second = cap(
        sum_directions(scatter_six(follower_first_oriented))
    )
    queued_second = binary("mul", queued_second, old_active)
    straight_second = binary("mul", straight_second, previous_inactive)
    follower_second = cap(binary("add", queued_second, straight_second))

    emit(
        "vconcat",
        [next_guard_codes, next_danger_codes, active_flag],
        name="mem_next"
    )

    # Patrol motion, including endpoint reversals.
    horizon = 26
    patrol_orientation = binary("mul", patrols, markers)
    patrol_current = cap(sum_directions(patrol_orientation))
    current_patrols = []
    future_dangers = []

    for t in range(horizon):
        current_patrols.append(patrol_current)
        moved = scatter_six(patrol_orientation)
        continuing = binary("mul", moved, edges)
        reversing = binary("mul", moved, missing_edges)
        reversing = conv(reversing, "reverse_w")
        patrol_orientation = binary("add", continuing, reversing)
        patrol_current = cap(sum_directions(patrol_orientation))

        danger = patrol_current
        if t == 0:
            danger = cap(binary("add", danger, follower_first))
        elif t == 1:
            danger = cap(binary("add", danger, follower_second))
        future_dangers.append(danger)

    def heading_backup(q):
        a = conv(q, "perp_a_w")
        b = conv(q, "perp_b_w")
        best = binary("max2", q, a)
        best = binary("max2", best, b)
        reverse = conv(q, "reverse_w")
        reverse = binary("mul", reverse, inactive_flag)
        return binary("max2", best, reverse)

    # Finite-horizon, directional reachability backups.
    value = conv(ones, "zero_four_w")
    first_q = None

    for t in range(horizon - 1, -1, -1):
        capture_terminal = binary(
            "mul", static_value, current_patrols[t]
        )
        destination = binary("max2", value, capture_terminal)
        future_free = binary("sub", ones, future_dangers[t])
        future_free = binary("mul", future_free, safe_nodes)
        destination = binary("mul", destination, future_free)

        q = gather_directional(destination)
        q = binary("mul", q, edges)
        q = binary("mul", q, discount)

        if t == 0:
            first_q = q
        else:
            best = heading_backup(q)
            best = binary("max2", best, seed)
            value = binary("mul", best, safe_nodes)

    immediate_danger = cap(
        binary("add", guard_danger, future_dangers[0])
    )
    directional_risk = gather_scalar(immediate_danger)

    reverse_action = emit(
        "dense", [previous_dirs], w="reverse_action_w"
    )
    ongoing_follow = emit("vmul", [active_flag, old_active])
    forbidden_reverse = emit("vmul", [reverse_action, ongoing_follow])

    score_inputs = emit(
        "concat", [first_q, edges, missing_edges, directional_risk]
    )
    score_w = np.zeros((4, 16, 1, 1), np.float32)
    for d in range(4):
        score_w[d, d, 0, 0] = 1.0
        score_w[d, 4 + d, 0, 0] = 0.001
        score_w[d, 8 + d, 0, 0] = -2.0
        score_w[d, 12 + d, 0, 0] = -2.0
    parameter("score_w", score_w)

    scores = conv(score_inputs, "score_w")
    parameter("reverse_penalty_w", -2.0 * np.eye(4, dtype=np.float32))
    reverse_penalty = emit(
        "dense", [forbidden_reverse], w="reverse_penalty_w"
    )
    scores = binary("add", scores, reverse_penalty)
    player_scores = binary("mul", scores, player)
    pooled_scores = emit("gmean", [player_scores])

    logit_w = np.zeros((8, 4), np.float32)
    for d in range(4):
        logit_w[d + 1, d] = 4096.0
    logit_b = np.full(8, -10.0, np.float32)
    logit_b[0] = -1.5
    logit_b[1:5] = 0.0
    parameter("logit_w", logit_w)
    parameter("logit_b", logit_b)
    emit(
        "dense", [pooled_scores], name="logits",
        w="logit_w", b="logit_b"
    )

    parameter("click_w", np.zeros((1, 1, 1, 1), np.float32))
    conv(ones, "click_w", name="click")

    return {
        "graph": graph,
        "memory_size": 243,
        "params": params,
    }

import numpy as np


def build():
    graph = []
    params = {}
    node_count = 0
    param_count = 0

    def node(op, inputs, name=None, **fields):
        nonlocal node_count
        if name is None:
            name = "n%d" % node_count
            node_count += 1
        graph.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def param(value):
        nonlocal param_count
        name = "p%d" % param_count
        param_count += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], name=name, **fields)

    def unary(op, x, name=None):
        return node(op, [x], name=name)

    def binary(op, x, y, name=None):
        return node(op, [x, y], name=name)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vcat(*xs):
        return node("vconcat", xs)

    def spatial_scale(x, scale, bias=0.0, name=None):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
            name,
        )

    def vector_scale(x, scale, bias=0.0):
        return dense(x, [[scale]], [bias])

    def reduce_channels(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), dtype=np.float32))

    def spatial_sum(x):
        return vector_scale(unary("gmean", x), 4096.0)

    def vclip01(x):
        a = unary("vrelu", x)
        b = unary("vrelu", vector_scale(x, 1.0, -1.0))
        return dense(vcat(a, b), [[1.0, -1.0]])

    def sclip01(x, one):
        return binary("min2", unary("relu", x), one)

    def vec_not(x):
        return dense(x, [[-1.0]], [1.0])

    def mix_spatial(a, b, gate):
        return binary(
            "add",
            binary("mul", a, vec_not(gate)),
            binary("mul", b, gate),
        )

    def mix_vector(a, b, gate):
        return binary(
            "vadd",
            binary("vmul", a, vec_not(gate)),
            binary("vmul", b, gate),
        )

    one = dense("aprev", np.zeros((1, 8), dtype=np.float32), [1.0])
    xy = conv(
        "coords",
        np.array(
            [
                [[[63.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[63.0]], [[0.0]]],
            ],
            dtype=np.float32,
        ),
    )
    yy = sl(xy, 0, 1)
    xx = sl(xy, 1, 2)

    # Handle arms: selected handles are black, others grey.
    arm_w = np.zeros((2, 16, 5, 5), dtype=np.float32)
    arm_offsets = [
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (0, -2), (0, -1), (0, 1), (0, 2),
    ]
    for out, colour in enumerate((0, 3)):
        for dy, dx in arm_offsets:
            arm_w[out, colour, 2 + dy, 2 + dx] = 1.0
    arms = unary("relu", conv("x0", arm_w, [-7.5, -7.5]))
    arms = binary("add", arms, arms)
    active = sl(arms, 0, 1)
    inactive = sl(arms, 1, 2)
    all_handles = binary("max2", active, inactive)

    allowed = [4, 7, 8, 9, 11, 12, 13, 14, 15]
    marker_w = np.zeros((16, 16, 1, 1), dtype=np.float32)
    for c in allowed:
        marker_w[c, c, 0, 0] = 1.0
    markers = conv("x0", marker_w)
    handles_by_colour = binary("mul", markers, all_handles)
    active_by_colour = binary("mul", markers, active)
    active_colour = unary("gmax", active_by_colour)

    # Single-colour balls and goal outlines.
    ball_w = np.zeros((16, 16, 3, 3), dtype=np.float32)
    for c in range(16):
        ball_w[c, 6, 1, 1] += 1.0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ball_w[c, c, 1 + dy, 1 + dx] += 1.0
    balls = unary("relu", conv("x0", ball_w, np.full(16, -4.5)))
    balls = binary("add", balls, balls)

    ring = [
        (-3, -1), (-3, 1), (-2, -2), (-2, 2),
        (-1, -3), (-1, 3), (1, -3), (1, 3),
        (2, -2), (2, 2), (3, -1), (3, 1),
    ]
    goal_w = np.zeros((16, 16, 7, 7), dtype=np.float32)
    for c in range(16):
        for dy, dx in ring:
            goal_w[c, c, 3 + dy, 3 + dx] = 1.0
    goals = unary("relu", conv("x0", goal_w, np.full(16, -10.5)))
    goals = binary("add", goals, goals)
    goals = binary("min2", goals, one)

    goal_present = unary("gmax", goals)
    ball_present = unary("gmax", balls)
    eligible = binary("vmul", goal_present, ball_present)
    active_eligible = binary("vmul", active_colour, eligible)
    normal_good = vclip01(dense(active_eligible, np.ones((1, 16))))

    normal_group = reduce_channels(
        binary("mul", handles_by_colour, active_colour), 16
    )
    normal_active = reduce_channels(active_by_colour, 16)
    normal_ball = reduce_channels(binary("mul", balls, active_colour), 16)
    normal_goal = reduce_channels(binary("mul", goals, active_colour), 16)
    normal_eligible = reduce_channels(
        binary("mul", handles_by_colour, eligible), 16
    )

    colours = [7, 8, 9, 11, 12, 13, 14, 15]
    nc = len(colours)
    footprint = np.ones((5, 5), dtype=np.float32)
    footprint[0, 0] = 0.0
    footprint[0, 4] = 0.0
    footprint[4, 0] = 0.0
    footprint[4, 4] = 0.0

    # Filled ball or pickup footprint.
    shape_w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for c in [0, 6] + colours:
        shape_w[0, c] = footprint
    filled_shape = unary("relu", conv("x0", shape_w, [-20.5]))
    filled_shape = binary("add", filled_shape, filled_shape)

    colour4_handles = sl(handles_by_colour, 4, 5)
    colour4_active = sl(active_by_colour, 4, 5)
    composite_mode = unary("gmax", colour4_handles)

    seed_area = sclip01(
        conv(colour4_active, np.ones((1, 1, 7, 7), dtype=np.float32)),
        one,
    )
    line = sl("x0", 1, 2)
    flood_seed = binary("mul", seed_area, line)

    # Bridge tether segments hidden by a ball.
    ball_bridge = sclip01(
        conv(filled_shape, np.ones((1, 1, 7, 7), dtype=np.float32)),
        one,
    )

    # Bridge tether segments hidden by goal and decorative rings.
    generic_ring_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for c in [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for dy, dx in ring:
            generic_ring_w[0, c, 3 + dy, 3 + dx] = 1.0
    generic_ring = unary("relu", conv("x0", generic_ring_w, [-9.5]))
    generic_ring = sclip01(binary("add", generic_ring, generic_ring), one)
    ring_bridge = sclip01(
        conv(generic_ring, np.ones((1, 1, 7, 7), dtype=np.float32)),
        one,
    )
    flood_mask = binary("max2", line, ball_bridge)
    flood_mask = binary("max2", flood_mask, ring_bridge)

    flood_w = param(np.ones((1, 1, 3, 3), dtype=np.float32))
    graph.append({
        "id": "flood",
        "op": "loop",
        "T": 64,
        "state": [{"id": "flood_state", "init": flood_seed}],
        "body": [
            {
                "id": "flood_conv",
                "op": "conv",
                "in": ["flood_state"],
                "w": flood_w,
            },
            {
                "id": "flood_cap",
                "op": "min2",
                "in": ["flood_conv", one],
            },
            {
                "id": "flood_allowed",
                "op": "mul",
                "in": ["flood_cap", flood_mask],
            },
            {
                "id": "flood_next",
                "op": "max2",
                "in": ["flood_state", "flood_allowed"],
            },
        ],
        "next": {"flood_state": "flood_next"},
    })
    reached_area = sclip01(
        conv("flood.flood_state", np.ones((1, 1, 7, 7), dtype=np.float32)),
        one,
    )
    composite_group = binary(
        "max2",
        binary("mul", colour4_handles, reached_area),
        colour4_active,
    )
    composite_count = spatial_sum(composite_group)
    composite_sum_x = spatial_sum(binary("mul", composite_group, xx))
    composite_sum_y = spatial_sum(binary("mul", composite_group, yy))

    def centroid_axis(coord, total, count):
        t = binary("sub", total, binary("mul", coord, count))
        lower = binary("add", t, one)
        upper = binary("sub", count, t)
        return sclip01(binary("min2", lower, upper), one)

    centroid_x = centroid_axis(xx, composite_sum_x, composite_count)
    centroid_y = centroid_axis(yy, composite_sum_y, composite_count)
    centroid_map = binary("mul", centroid_x, centroid_y)
    composite_ball = binary("mul", centroid_map, filled_shape)
    composite_ball_present = unary("gmax", composite_ball)
    composite_ball_x = spatial_sum(binary("mul", composite_ball, xx))
    composite_ball_y = spatial_sum(binary("mul", composite_ball, yy))

    # A completed goal is replaced by a black ring.
    black_ring_w = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for dy, dx in ring:
        black_ring_w[0, 0, 3 + dy, 3 + dx] = 1.0
    black_ring = unary("relu", conv("x0", black_ring_w, [-11.5]))
    black_ring = binary("add", black_ring, black_ring)
    done_raw = unary("gmax", binary("mul", composite_ball, black_ring))
    done = vclip01(vector_scale(done_raw, 2.0, -1.0))

    # Read absorbed colours from the ball footprint.
    owned_w = np.zeros((nc, 16, 5, 5), dtype=np.float32)
    for i, c in enumerate(colours):
        owned_w[i, c] = footprint
    owned_counts = conv("x0", owned_w)
    owned = unary("gmax", binary("mul", owned_counts, composite_ball))
    owned_positive = unary("vrelu", owned)
    owned_excess = unary(
        "vrelu", dense(owned, np.eye(nc), -np.ones(nc))
    )
    owned = dense(
        vcat(owned_positive, owned_excess),
        np.concatenate([np.eye(nc), -np.eye(nc)], axis=1),
    )
    owned_count = dense(owned, np.ones((1, nc)))
    complete_colours = vclip01(vector_scale(owned_count, 1.0, -1.0))

    # Split-colour rings represented by ordered pairs of half colours.
    half_w = np.zeros((2 * nc, 16, 7, 7), dtype=np.float32)
    for i, c in enumerate(colours):
        for dy, dx in ring:
            if dx < 0:
                half_w[i, c, 3 + dy, 3 + dx] = 1.0
            else:
                half_w[nc + i, c, 3 + dy, 3 + dx] = 1.0
    halves = unary("relu", conv("x0", half_w, np.full(2 * nc, -5.5)))
    halves = binary("add", halves, halves)
    left_halves = sl(halves, 0, nc)
    right_halves = sl(halves, nc, 2 * nc)

    pairs = [(i, j) for i in range(nc) for j in range(nc) if i != j]
    npairs = len(pairs)
    left_expand = np.zeros((npairs, nc, 1, 1), dtype=np.float32)
    right_expand = np.zeros((npairs, nc, 1, 1), dtype=np.float32)
    outside_w = np.zeros((npairs, nc), dtype=np.float32)
    pair_to_colour = np.zeros((nc, npairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        left_expand[k, i, 0, 0] = 1.0
        right_expand[k, j, 0, 0] = 1.0
        for q in range(nc):
            if q != i and q != j:
                outside_w[k, q] = 1.0
        pair_to_colour[i, k] = 1.0
        pair_to_colour[j, k] = 1.0

    pair_maps = binary(
        "mul",
        conv(left_halves, left_expand),
        conv(right_halves, right_expand),
    )
    pair_present = unary("gmax", pair_maps)
    outside_owned = dense(owned, outside_w)
    compatible = unary(
        "vrelu",
        dense(outside_owned, -np.eye(npairs), np.ones(npairs)),
    )
    compatible_present = binary("vmul", pair_present, compatible)

    colour_needed_raw = dense(compatible_present, pair_to_colour)
    colour_needed_positive = unary("vrelu", colour_needed_raw)
    colour_needed_excess = unary(
        "vrelu",
        dense(colour_needed_raw, np.eye(nc), -np.ones(nc)),
    )
    colour_needed = dense(
        vcat(colour_needed_positive, colour_needed_excess),
        np.concatenate([np.eye(nc), -np.eye(nc)], axis=1),
    )
    unowned = dense(owned, -np.eye(nc), np.ones(nc))
    colour_needed = binary("vmul", colour_needed, unowned)

    # Two orientations of a half-coloured pickup sprite.
    left_pixels = np.array([
        [-1, 1, 0, 0, -1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [-1, 1, 1, 0, -1],
    ])
    right_pixels = np.array([
        [-1, 0, 1, 1, -1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [-1, 0, 0, 1, -1],
    ])
    pickup_w = np.zeros((2 * nc, 16, 5, 5), dtype=np.float32)
    for i, c in enumerate(colours):
        for orientation, template in enumerate((left_pixels, right_pixels)):
            out = orientation * nc + i
            for ry in range(5):
                for rx in range(5):
                    v = template[ry, rx]
                    if v == 0:
                        pickup_w[out, 0, ry, rx] = 1.0
                    elif v == 1:
                        pickup_w[out, c, ry, rx] = 1.0
    pickups = unary("relu", conv("x0", pickup_w, np.full(2 * nc, -20.5)))
    pickups = binary("add", pickups, pickups)
    pickups = binary("max2", sl(pickups, 0, nc), sl(pickups, nc, 2 * nc))

    # Exclude the current and other inferred ball centres.
    all4_count = spatial_sum(colour4_handles)
    all4_sum_x = spatial_sum(binary("mul", colour4_handles, xx))
    all4_sum_y = spatial_sum(binary("mul", colour4_handles, yy))
    other_count = dense(vcat(all4_count, composite_count), [[1.0, -1.0]])
    other_sum_x = dense(vcat(all4_sum_x, composite_sum_x), [[1.0, -1.0]])
    other_sum_y = dense(vcat(all4_sum_y, composite_sum_y), [[1.0, -1.0]])
    other_centroid = binary(
        "mul",
        centroid_axis(xx, other_sum_x, other_count),
        centroid_axis(yy, other_sum_y, other_count),
    )
    exclude_centres = binary("max2", composite_ball, other_centroid)
    pickup_candidates = reduce_channels(
        binary("mul", pickups, colour_needed), nc
    )
    pickup_candidates = sclip01(pickup_candidates, one)
    pickup_candidates = binary(
        "mul", pickup_candidates, binary("sub", one, exclude_centres)
    )
    pickup_candidates = binary(
        "mul", pickup_candidates, vec_not(complete_colours)
    )

    matching_goals = reduce_channels(
        binary("mul", pair_maps, compatible_present), npairs
    )
    matching_goals = sclip01(matching_goals, one)
    matching_goals = binary("mul", matching_goals, complete_colours)
    composite_targets = binary("max2", pickup_candidates, matching_goals)
    composite_target_present = unary("gmax", composite_targets)
    composite_target_present = binary(
        "vmul", composite_target_present, vec_not(done)
    )

    # Nearest valid target, with row-major tie breaking.
    target_distance = binary(
        "add",
        unary("abs", binary("sub", xx, composite_ball_x)),
        unary("abs", binary("sub", yy, composite_ball_y)),
    )
    target_scores = binary(
        "add",
        spatial_scale(target_distance, -1.0),
        spatial_scale(binary("sub", one, composite_targets), -1000.0),
    )
    target_best = unary("gmax", target_scores)
    target_winners = sclip01(
        spatial_scale(
            unary("abs", binary("sub", target_scores, target_best)),
            -2.0,
            1.0,
        ),
        one,
    )
    target_y = dense(
        unary(
            "gmax",
            binary("mul", target_winners, spatial_scale(yy, -1.0, 63.0)),
        ),
        [[-1.0]],
        [63.0],
    )
    chosen_row = sclip01(
        spatial_scale(unary("abs", binary("sub", yy, target_y)), -1.0, 1.0),
        one,
    )
    row_winners = binary("mul", target_winners, chosen_row)
    target_x = dense(
        unary(
            "gmax",
            binary("mul", row_winners, spatial_scale(xx, -1.0, 63.0)),
        ),
        [[-1.0]],
        [63.0],
    )

    group_handles = mix_spatial(normal_group, composite_group, composite_mode)
    active_map = mix_spatial(normal_active, colour4_active, composite_mode)
    group_ball = mix_spatial(normal_ball, composite_ball, composite_mode)
    normal_goal_x = spatial_sum(binary("mul", normal_goal, xx))
    normal_goal_y = spatial_sum(binary("mul", normal_goal, yy))
    goal_x = mix_vector(normal_goal_x, target_x, composite_mode)
    goal_y = mix_vector(normal_goal_y, target_y, composite_mode)
    eligible_handles = mix_spatial(
        normal_eligible, colour4_handles, composite_mode
    )
    composite_good = binary(
        "vmul", composite_ball_present, composite_target_present
    )
    good = mix_vector(normal_good, composite_good, composite_mode)
    has_target = mix_vector(normal_good, composite_target_present, composite_mode)

    count = spatial_sum(group_handles)
    handle_x = spatial_sum(binary("mul", active_map, xx))
    handle_y = spatial_sum(binary("mul", active_map, yy))
    sum_x = spatial_sum(binary("mul", group_handles, xx))
    sum_y = spatial_sum(binary("mul", group_handles, yy))
    ball_x = spatial_sum(binary("mul", group_ball, xx))
    ball_y = spatial_sum(binary("mul", group_ball, yy))
    other_x = dense(vcat(sum_x, handle_x), [[1.0, -1.0]])
    other_y = dense(vcat(sum_y, handle_y), [[1.0, -1.0]])

    desired_x = dense(
        vcat(binary("vmul", count, goal_x), other_x), [[1.0, -1.0]]
    )
    desired_y = dense(
        vcat(binary("vmul", count, goal_y), other_y), [[1.0, -1.0]]
    )
    distance = binary(
        "add",
        unary("abs", binary("sub", xx, desired_x)),
        unary("abs", binary("sub", yy, desired_y)),
    )
    raw_score = spatial_scale(distance, -1.0)
    current_score = unary(
        "gmax",
        binary(
            "add",
            raw_score,
            spatial_scale(binary("sub", one, active_map), -1000.0),
        ),
    )

    # Reject candidate clicks overlapping walls or existing handles.
    wall = sl("x0", 2, 3)
    diamond = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for dy0, dx0 in [
        (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
        (-1, -1), (-1, 0), (-1, 1),
        (1, -1), (1, 0), (1, 1), (-2, 0), (2, 0),
    ]:
        diamond[0, 0, 2 + dy0, 2 + dx0] = 1.0
    wall_block = sclip01(conv(wall, diamond), one)
    handle_box = sclip01(
        conv(all_handles, np.ones((1, 1, 5, 5), dtype=np.float32)), one
    )
    ylo = sclip01(spatial_scale(yy, 1.0, -1.0), one)
    yhi = sclip01(spatial_scale(yy, -1.0, 62.0), one)
    xlo = sclip01(spatial_scale(xx, 1.0, -1.0), one)
    xhi = sclip01(spatial_scale(xx, -1.0, 62.0), one)
    inside = binary("mul", binary("mul", ylo, yhi), binary("mul", xlo, xhi))
    outside = binary("sub", one, inside)

    # Reject candidates that place the ball on a visible hazard.
    hazard = sl("x0", 10, 11)
    hazard_for_ball = sclip01(
        conv(hazard, footprint.reshape(1, 1, 5, 5)), one
    )

    def source_indicators(axis):
        w = np.zeros((64, 3, 1, 1), dtype=np.float32)
        w[:, axis, 0, 0] = -63.0
        z = conv("coords", w, np.arange(64, dtype=np.float32))
        z = unary("abs", z)
        return unary("relu", binary("sub", one, z))

    def destination_indicators(coord, other):
        coeff = -np.arange(64, dtype=np.float32).reshape(64, 1)
        base = dense(count, coeff, np.ones(64, dtype=np.float32))
        t1 = binary("add", binary("add", coord, other), base)
        count_plus_one = vector_scale(count, 1.0, 1.0)
        t2 = binary("sub", count_plus_one, t1)
        return sclip01(binary("min2", t1, t2), one)

    source_x = source_indicators(1)
    dest_x = destination_indicators(xx, other_x)
    hx = unary("rowmax", binary("mul", hazard_for_ball, source_x))
    hx = reduce_channels(binary("mul", hx, dest_x), 64)
    source_y = source_indicators(0)
    dest_y = destination_indicators(yy, other_y)
    hy = unary("colmax", binary("mul", hx, source_y))
    hazard_candidate = reduce_channels(binary("mul", hy, dest_y), 64)
    hazard_candidate = sclip01(
        spatial_scale(hazard_candidate, 4.0, -1.0), one
    )

    blocked = binary("max2", wall_block, handle_box)
    blocked = binary("max2", blocked, outside)
    blocked = binary("max2", blocked, hazard_candidate)
    candidate_score = binary(
        "add", raw_score, spatial_scale(blocked, -1000.0)
    )
    best_score = unary("gmax", candidate_score)
    improvement = dense(
        vcat(best_score, current_score), [[1.0, -1.0]], [-0.1]
    )
    advance = binary(
        "vmul", vclip01(vector_scale(improvement, 10.0)), good
    )

    near = vclip01(dense(vcat(current_score, count), [[1.0, 1.5]]))
    same_group_switch = binary(
        "vmul", binary("vmul", good, vec_not(near)), has_target
    )
    other_group_switch = vec_not(same_group_switch)
    same_candidates = unary("relu", binary("sub", group_handles, active_map))
    other_candidates = unary(
        "relu", binary("sub", eligible_handles, group_handles)
    )
    selection_mask = binary(
        "add",
        binary("mul", same_candidates, same_group_switch),
        binary("mul", other_candidates, other_group_switch),
    )
    selection_mask = binary("min2", selection_mask, one)

    dir_x = unary("vtanh", dense(vcat(goal_x, ball_x), [[0.3, -0.3]]))
    dir_y = unary("vtanh", dense(vcat(goal_y, ball_y), [[0.3, -0.3]]))
    projection = binary(
        "add", binary("mul", xx, dir_x), binary("mul", yy, dir_y)
    )
    selection_score = binary(
        "add",
        spatial_scale(binary("sub", one, selection_mask), -1000.0),
        spatial_scale(projection, -0.1),
    )
    binary(
        "add",
        binary("mul", candidate_score, advance),
        binary("mul", selection_score, vec_not(advance)),
        name="click",
    )

    logits_bias = np.zeros(8, dtype=np.float32)
    logits_bias[6] = 1.0
    dense(
        "aprev",
        np.zeros((8, 8), dtype=np.float32),
        logits_bias,
        "logits",
    )

    return {"graph": graph, "memory_size": 0, "params": params}

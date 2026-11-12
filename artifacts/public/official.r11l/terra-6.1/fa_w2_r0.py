import numpy as np


def build():
    graph = []
    params = {}
    next_id = 0
    next_param = 0

    def uid(prefix="n"):
        nonlocal next_id
        next_id += 1
        return "%s%d" % (prefix, next_id)

    def parameter(value):
        nonlocal next_param
        next_param += 1
        name = "p%d" % next_param
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def op(name, inputs, **fields):
        ident = uid()
        nd = {"id": ident, "op": name, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return ident

    def conv(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return op("conv", [x], **fields)

    def dense(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return op("dense", [x], **fields)

    def affine(x, scale, bias=0.0):
        return conv(x, [[[[scale]]]], [bias])

    def vaffine(x, scale, bias=0.0):
        return dense(x, [[scale]], [bias])

    def unary(name, x):
        return op(name, [x])

    def binary(name, x, y):
        return op(name, [x, y])

    def add(x, y):
        return binary("add", x, y)

    def sub(x, y):
        return binary("sub", x, y)

    def mul(x, y):
        return binary("mul", x, y)

    def maximum(x, y):
        return binary("max2", x, y)

    def vadd(x, y):
        return op("vadd", [x, y])

    def vmul(x, y):
        return op("vmul", [x, y])

    def sl(x, start, stop):
        return op("slice", [x], start=start, stop=stop)

    def gmax(x):
        return op("gmax", [x])

    def clip01(x):
        positive = unary("relu", x)
        below_one = unary("relu", affine(positive, -1.0, 1.0))
        return affine(below_one, -1.0, 1.0)

    def vclip01(x):
        positive = unary("vrelu", x)
        below_one = unary("vrelu", vaffine(positive, -1.0, 1.0))
        return vaffine(below_one, -1.0, 1.0)

    def vone_minus(x):
        return vaffine(x, -1.0, 1.0)

    def threshold(x, cutoff):
        return clip01(affine(x, 2.0, -2.0 * cutoff))

    def union(items):
        result = items[0]
        for item in items[1:]:
            result = maximum(result, item)
        return result

    def coord_of(mask, coordinate_plane):
        return vaffine(op("gmean", [mul(mask, coordinate_plane)]), 4096.0)

    def distance_to(coordinate_plane, value):
        return unary("abs", sub(coordinate_plane, value))

    row = affine(sl("coords", 0, 1), 63.0)
    col = affine(sl("coords", 1, 2), 63.0)
    one = sl("coords", 2, 3)

    tie = add(
        affine(row, -1.0 / 128.0),
        affine(col, -1.0 / 8192.0),
    )

    def choose(mask, score):
        ranked = add(add(score, tie), affine(mask, 256.0))
        best = gmax(ranked)
        delta = sub(ranked, best)
        winner = clip01(affine(delta, 100000.0, 1.0))
        return mul(winner, mask)

    def nearest(mask, r, c):
        dist = add(distance_to(row, r), distance_to(col, c))
        return choose(mask, affine(dist, -1.0))

    planes = [sl("x0", i, i + 1) for i in range(16)]

    # Selected handle arms are colour 0; inactive arms are colour 3.
    handle_colours = [15, 14, 4]
    cross = [
        (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3),
        (4, 2),
    ]
    handle_weights = np.zeros((6, 16, 5, 5), dtype=np.float32)
    for selected in range(2):
        arm_colour = 0 if selected == 0 else 3
        for k, centre_colour in enumerate(handle_colours):
            channel = 3 * selected + k
            for r, c in cross:
                handle_weights[channel, arm_colour, r, c] = 1.0
            handle_weights[channel, centre_colour, 2, 2] = 1.0

    handle_scores = conv("x0", handle_weights)
    handles = []
    for k in range(6):
        detected = threshold(sl(handle_scores, k, k + 1), 10.5)
        handles.append(mul(detected, planes[handle_colours[k % 3]]))

    active = handles[:3]
    inactive = handles[3:]
    active_flags = [gmax(x) for x in active]
    active_map = union(active)
    active_row = coord_of(active_map, row)
    active_col = coord_of(active_map, col)

    # Flood through connector lines to associate black handles and balls.
    walk = union([
        planes[0], planes[1], planes[3], planes[4],
        planes[8], planes[9], planes[11], planes[14], planes[15],
    ])
    flood_id = uid("flood")
    flood_state = uid("state")
    body = []
    outer_graph = graph
    graph = body
    expanded = clip01(conv(
        flood_state, np.ones((1, 1, 3, 3), dtype=np.float32)
    ))
    flood_next = mul(expanded, walk)
    graph = outer_graph
    graph.append({
        "id": flood_id,
        "op": "loop",
        "T": 128,
        "state": [{"id": flood_state, "init": active_map}],
        "body": body,
        "next": {flood_state: flood_next},
    })
    connected = flood_id + "." + flood_state

    same_inactive_parts = [
        mul(inactive[0], active_flags[0]),
        mul(inactive[1], active_flags[1]),
        mul(mul(inactive[2], active_flags[2]), connected),
    ]
    same_inactive = union(same_inactive_parts)
    same_active_parts = [
        mul(active[0], active_flags[0]),
        mul(active[1], active_flags[1]),
        mul(mul(active[2], active_flags[2]), connected),
    ]
    group_handles = union(same_active_parts + same_inactive_parts)
    group_count = vaffine(op("gmean", [group_handles]), 4096.0)
    has_same_inactive = gmax(same_inactive)

    ellipse = [
        (r, c) for r in range(5) for c in range(5)
        if not (r in (0, 4) and c in (0, 4))
    ]

    ball_weights = np.zeros((3, 16, 5, 5), dtype=np.float32)
    for k, colour in enumerate((15, 14)):
        for r, c in ellipse:
            if (r, c) != (2, 2):
                ball_weights[k, colour, r, c] = 1.0
        ball_weights[k, 6, 2, 2] = 1.0
    for r, c in ellipse:
        for colour in (0, 8, 9, 11, 14):
            ball_weights[2, colour, r, c] = 1.0

    ball_scores = conv("x0", ball_weights)
    purple_ball = mul(
        threshold(sl(ball_scores, 0, 1), 17.5), planes[6]
    )
    green_ball = mul(
        threshold(sl(ball_scores, 1, 2), 17.5), planes[6]
    )
    black_ball = threshold(sl(ball_scores, 2, 3), 18.5)
    ball_candidates = union([
        mul(purple_ball, active_flags[0]),
        mul(green_ball, active_flags[1]),
        mul(mul(black_ball, active_flags[2]), connected),
    ])
    visible_ball = nearest(ball_candidates, active_row, active_col)
    visible_ball_flag = gmax(visible_ball)

    # Reciprocal of the integer handle count, for counts 1 through 16.
    reciprocal_terms = []
    for k in range(1, 17):
        difference = vaffine(group_count, 1.0, -float(k))
        absolute = vadd(
            unary("vrelu", difference),
            unary("vrelu", vaffine(difference, -1.0)),
        )
        reciprocal_terms.append(vclip01(vaffine(absolute, -1.0, 1.0)))
    reciprocal_inputs = op("vconcat", reciprocal_terms)
    reciprocal = dense(
        reciprocal_inputs,
        [[1.0 / k for k in range(1, 17)]],
    )

    mean_row = vmul(coord_of(group_handles, row), reciprocal)
    mean_col = vmul(coord_of(group_handles, col), reciprocal)

    def floor_coordinate(value):
        weights = np.full((63, 1), 20.0, dtype=np.float32)
        bias = np.asarray(
            [-20.0 * k + 1.2 for k in range(1, 64)],
            dtype=np.float32,
        )
        steps = unary("vrelu", dense(value, weights, bias))
        below = unary("vrelu", dense(
            steps,
            -np.eye(63, dtype=np.float32),
            np.ones(63, dtype=np.float32),
        ))
        return dense(
            below,
            -np.ones((1, 63), dtype=np.float32),
            [63.0],
        )

    inferred_row = floor_coordinate(mean_row)
    inferred_col = floor_coordinate(mean_col)
    inferred_ball = nearest(one, inferred_row, inferred_col)
    multiple_handles = vclip01(vaffine(group_count, 1.0, -1.0))
    coloured_group = vclip01(vadd(active_flags[0], active_flags[1]))
    use_inferred = vmul(
        vmul(vone_minus(visible_ball_flag), multiple_handles),
        coloured_group,
    )
    ball = union([visible_ball, mul(inferred_ball, use_inferred)])
    ball_row = coord_of(ball, row)
    ball_col = coord_of(ball, col)

    # Goal-ring templates.
    ring_offsets = [
        (-3, -1), (-3, 1),
        (-2, -2), (-2, 2),
        (-1, -3), (-1, 3),
        (1, -3), (1, 3),
        (2, -2), (2, 2),
        (3, -1), (3, 1),
    ]
    ring_weights = np.zeros((4, 16, 7, 7), dtype=np.float32)
    for dr, dc in ring_offsets:
        r, c = dr + 3, dc + 3
        ring_weights[0, 15, r, c] = 1.0
        ring_weights[1, 14, r, c] = 1.0
        ring_weights[2, 9 if dc < 0 else 8, r, c] = 1.0
        ring_weights[3, 14 if dc < 0 else 11, r, c] = 1.0

    ring_scores = conv("x0", ring_weights)
    rings = [
        threshold(sl(ring_scores, i, i + 1), 8.5)
        for i in range(4)
    ]
    ring_present = [gmax(x) for x in rings]
    black_ring_present = gmax(union([rings[2], rings[3]]))

    pigment_colours = [9, 8, 14, 11]
    pigment_weights = np.zeros((4, 16, 5, 5), dtype=np.float32)
    for k, colour in enumerate(pigment_colours):
        for r, c in ellipse:
            pigment_weights[k, colour, r, c] = 1.0
    pigment_counts = conv("x0", pigment_weights)
    pigment_flags = [
        gmax(mul(clip01(sl(pigment_counts, i, i + 1)), ball))
        for i in range(4)
    ]
    p9, p8, p14, p11 = pigment_flags

    pickup_patterns = [
        [
            [-1, 9, 0, 0, -1],
            [9, 9, 0, 0, 0],
            [9, 9, 0, 0, 0],
            [9, 9, 9, 0, 0],
            [-1, 9, 9, 0, -1],
        ],
        [
            [-1, 0, 8, 8, -1],
            [0, 0, 8, 8, 8],
            [0, 0, 8, 8, 8],
            [0, 0, 0, 8, 8],
            [-1, 0, 0, 8, -1],
        ],
        [
            [-1, 14, 0, 0, -1],
            [14, 14, 0, 0, 0],
            [14, 14, 0, 0, 0],
            [14, 14, 14, 0, 0],
            [-1, 14, 14, 0, -1],
        ],
        [
            [-1, 0, 11, 11, -1],
            [0, 0, 11, 11, 11],
            [0, 0, 11, 11, 11],
            [0, 0, 0, 11, 11],
            [-1, 0, 0, 11, -1],
        ],
    ]
    pickup_weights = np.zeros((4, 16, 5, 5), dtype=np.float32)
    for k, pattern in enumerate(pickup_patterns):
        for r in range(5):
            for c in range(5):
                colour = pattern[r][c]
                if colour >= 0:
                    pickup_weights[k, colour, r, c] = 1.0
    pickup_scores = conv("x0", pickup_weights)

    eligibility = [
        vmul(vmul(vone_minus(p9), vone_minus(p14)), vone_minus(p11)),
        vmul(vmul(vone_minus(p8), vone_minus(p14)), vone_minus(p11)),
        vmul(vmul(vone_minus(p14), vone_minus(p9)), vone_minus(p8)),
        vmul(vmul(vone_minus(p11), vone_minus(p9)), vone_minus(p8)),
    ]
    pickup_parts = []
    for i in range(4):
        pickup = threshold(sl(pickup_scores, i, i + 1), 19.5)
        pickup_parts.append(
            mul(mul(pickup, eligibility[i]), active_flags[2])
        )
    pickups = union(pickup_parts)

    ball_distance = add(
        distance_to(row, ball_row), distance_to(col, ball_col)
    )
    away_from_ball = clip01(affine(ball_distance, 1.0, -4.0))
    pickups = mul(pickups, away_from_ball)
    pickup_target = nearest(pickups, ball_row, ball_col)

    goal_candidates = union([
        mul(rings[0], active_flags[0]),
        mul(rings[1], active_flags[1]),
        mul(mul(rings[2], active_flags[2]), vmul(p9, p8)),
        mul(mul(rings[3], active_flags[2]), vmul(p14, p11)),
    ])
    goal_target = nearest(goal_candidates, ball_row, ball_col)

    goal_footprint = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for r in range(7):
        for c in range(7):
            if (
                (r in (0, 6) and c in (2, 3, 4))
                or (r in (1, 5) and 1 <= c <= 5)
                or (r in (2, 3, 4))
            ):
                goal_footprint[0, 0, r, c] = 1.0
    ball_footprint = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for r, c in ellipse:
        ball_footprint[0, 0, r, c] = 1.0
    goal_pixels = conv(goal_target, goal_footprint)
    goal_overlap = conv(goal_pixels, ball_footprint)
    goal_solved = gmax(mul(clip01(goal_overlap), ball))
    unsolved_goal = mul(goal_target, vone_minus(goal_solved))

    target_candidates = union([pickup_target, unsolved_goal])
    target = nearest(target_candidates, ball_row, ball_col)
    has_target = gmax(target)
    target_row = coord_of(target, row)
    target_col = coord_of(target, col)

    # Choose a safe ball endpoint reachable by moving the selected handle.
    has_hazard = gmax(planes[10])
    hazard_pixels = maximum(planes[10], mul(planes[2], has_hazard))
    danger = conv(hazard_pixels, ball_footprint)
    safe_ball = unary("relu", affine(danger, -1.0, 1.0))

    delta_row = sub(row, ball_row)
    delta_col = sub(col, ball_col)
    predicted_click_row = add(active_row, mul(delta_row, group_count))
    predicted_click_col = add(active_col, mul(delta_col, group_count))

    row_low = clip01(affine(predicted_click_row, 1.0, -1.0))
    row_high = clip01(affine(predicted_click_row, -1.0, 62.0))
    col_low = clip01(affine(predicted_click_col, 1.0, -1.0))
    col_high = clip01(affine(predicted_click_col, -1.0, 62.0))
    reachable = mul(mul(row_low, row_high), mul(col_low, col_high))
    waypoint_mask = mul(reachable, safe_ball)
    waypoint = nearest(waypoint_mask, target_row, target_col)
    waypoint_row = coord_of(waypoint, row)
    waypoint_col = coord_of(waypoint, col)

    desired_row = vadd(
        active_row,
        vmul(group_count, vadd(waypoint_row, vaffine(ball_row, -1.0))),
    )
    desired_col = vadd(
        active_col,
        vmul(group_count, vadd(waypoint_col, vaffine(ball_col, -1.0))),
    )

    wall_weights = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for r, c in cross + [(2, 2)]:
        wall_weights[0, 0, r, c] = 1.0
    wall_hits = conv(planes[2], wall_weights)
    wall_free = unary("relu", affine(wall_hits, -1.0, 1.0))

    all_handles = union(active + inactive)
    handle_boxes = conv(
        all_handles, np.ones((1, 1, 5, 5), dtype=np.float32)
    )
    outside_handle_boxes = unary(
        "relu", affine(handle_boxes, -1.0, 1.0)
    )
    inside_row = mul(
        clip01(affine(row, 1.0, -1.0)),
        clip01(affine(row, -1.0, 62.0)),
    )
    inside_col = mul(
        clip01(affine(col, 1.0, -1.0)),
        clip01(affine(col, -1.0, 62.0)),
    )
    valid_click = mul(
        mul(wall_free, outside_handle_boxes),
        mul(inside_row, inside_col),
    )
    move_click = nearest(valid_click, desired_row, desired_col)

    direction_row = vadd(target_row, vaffine(ball_row, -1.0))
    direction_col = vadd(target_col, vaffine(ball_col, -1.0))

    def vector_sign(x):
        positive = vclip01(unary("vrelu", x))
        negative = vclip01(unary("vrelu", vaffine(x, -1.0)))
        return vadd(positive, vaffine(negative, -1.0))

    sign_row = vector_sign(direction_row)
    sign_col = vector_sign(direction_col)
    projection = add(mul(row, sign_row), mul(col, sign_col))
    same_click = choose(same_inactive, affine(projection, -1.0))

    disconnected = affine(connected, -1.0, 1.0)
    other_handles = union([
        mul(mul(inactive[0], vone_minus(active_flags[0])), ring_present[0]),
        mul(mul(inactive[1], vone_minus(active_flags[1])), ring_present[1]),
        mul(mul(inactive[2], disconnected), black_ring_present),
    ])
    has_other = gmax(other_handles)
    other_click = nearest(other_handles, active_row, active_col)

    waypoint_motion = gmax(mul(
        waypoint,
        add(distance_to(row, ball_row), distance_to(col, ball_col)),
    ))
    can_move = vclip01(waypoint_motion)

    switch_same = vmul(
        has_same_inactive,
        vclip01(vadd("mem", vone_minus(can_move))),
    )
    choose_same = vmul(has_target, switch_same)
    choose_other = vmul(vone_minus(has_target), has_other)
    selecting = vclip01(vadd(choose_same, choose_other))
    moving = vone_minus(selecting)

    click_result = union([
        mul(move_click, moving),
        mul(same_click, choose_same),
        mul(other_click, choose_other),
    ])
    graph.append({"id": "click", "op": "relu", "in": [click_result]})
    graph.append({
        "id": "mem_next",
        "op": "dense",
        "in": [selecting],
        "w": parameter([[-1.0]]),
        "b": parameter([1.0]),
    })

    # A partially coloured black ball with no compatible target is likely
    # irrecoverable because pickups cannot be removed or transferred.
    partial_first_pair = vadd(
        vmul(p9, vone_minus(p8)),
        vmul(p8, vone_minus(p9)),
    )
    partial_second_pair = vadd(
        vmul(p14, vone_minus(p11)),
        vmul(p11, vone_minus(p14)),
    )
    partial = vclip01(vadd(partial_first_pair, partial_second_pair))
    reset_needed = vmul(
        vmul(active_flags[2], partial),
        vone_minus(has_target),
    )

    logits_weights = np.zeros((8, 1), dtype=np.float32)
    logits_weights[0, 0] = 2.0
    logits_bias = np.zeros(8, dtype=np.float32)
    logits_bias[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [reset_needed],
        "w": parameter(logits_weights),
        "b": parameter(logits_bias),
    })

    return {
        "graph": graph,
        "memory_size": 1,
        "params": params,
    }

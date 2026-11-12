import numpy as np


def build():
    graph = []
    params = {}
    nc = 0
    pc = 0

    def param(value):
        nonlocal pc
        name = "p%d" % pc
        pc += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs=None, name=None, **fields):
        nonlocal nc
        if name is None:
            name = "n%d" % nc
            nc += 1
        item = {"id": name, "op": op}
        if inputs is not None:
            item["in"] = list(inputs)
        item.update(fields)
        graph.append(item)
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], name=name, **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def neg(x, size):
        return dense(x, -np.eye(size, dtype=np.float32))

    def one_minus(x):
        return dense(x, [[-1.0]], [1.0])

    def clip(x, size):
        positive = node("vrelu", [x])
        excess = dense(
            x,
            np.eye(size, dtype=np.float32),
            -np.ones(size, dtype=np.float32),
        )
        return va(positive, neg(node("vrelu", [excess]), size))

    def absvec(x, size):
        return va(node("vrelu", [x]), node("vrelu", [neg(x, size)]))

    def choose(old, new, gate):
        return va(vm(old, one_minus(gate)), vm(new, gate))

    def nonzero(x, size):
        total = dense(absvec(x, size), np.ones((1, size), np.float32))
        return clip(dense(total, [[1.0]], [-1.0]), 1)

    def first(mask, priority, one):
        score = mul(mask, priority)
        maximum = node("gmax", [score])
        equal = node("relu", [add(sub(score, maximum), one)])
        return mul(mask, equal)

    def point_coordinates(marker, row, col):
        y = node("gmax", [mul(marker, row)])
        x = node("gmax", [mul(marker, col)])
        return node("vconcat", [y, x])

    def connector_coordinates(mask, priority, one, row, col):
        kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
        kernel[0, 0, 1, 1] = 1
        kernel[0, 0, 0, 1] = -1
        kernel[0, 0, 1, 0] = -1
        corners = node("relu", [conv(mask, kernel)])
        a = first(corners, priority, one)
        b = first(sub(corners, a), priority, one)
        return node("vconcat", [
            point_coordinates(a, row, col),
            point_coordinates(b, row, col),
        ])

    def vector_difference(points):
        return dense(points, [
            [-1, 0, 1, 0],
            [0, -1, 0, 1],
        ])

    def target_delta(target, source):
        both = node("vconcat", [target, source])
        return dense(both, [
            [1, 0, -1, 0],
            [0, 1, 0, -1],
        ])

    def canonical_direction(direction):
        score = dense(direction, [[100.0, 1.0]])
        positive = clip(score, 1)
        sign = dense(positive, [[2.0]], [-1.0])
        return vm(direction, sign)

    def signed_endpoint(points, direction):
        score = dense(direction, [[100.0, 1.0]])
        positive = clip(score, 1)
        a = vsl(points, 0, 2)
        b = vsl(points, 2, 4)
        return va(vm(a, positive), vm(b, one_minus(positive)))

    def rotation_and_delta(points, direction, target):
        actual = vector_difference(points)
        desired = canonical_direction(direction)
        rotate = nonzero(va(actual, neg(desired, 2)), 2)
        endpoint = signed_endpoint(points, direction)
        return rotate, target_delta(target, endpoint)

    # Background and foreground masks.
    one = sl("coords", 2, 3)
    origin = node("relu", [
        conv("coords", [[-63.0, -63.0, 1.0]])
    ])
    background_vector = node("gmax", [mul("x0", origin)])
    background_pixels = mul("x0", background_vector)
    background = conv(
        background_pixels, np.ones((1, 16), dtype=np.float32)
    )

    row = conv("coords", [[63.0, 0.0, 0.0]])
    col = conv("coords", [[0.0, 63.0, 0.0]])
    non_ui = node("min2", [row, one])
    foreground = mul(sub(one, background), non_ui)

    black = mul(sl("x0", 0, 1), foreground)
    grey = mul(sl("x0", 4, 5), foreground)
    blue = mul(sl("x0", 8, 9), foreground)
    green = mul(sl("x0", 3, 4), foreground)
    has_grey = node("gmax", [grey])

    # Seed and flood-fill the selected piece.
    non_grey = sub(foreground, grey)
    masked_seed = mul(sub(sub(non_grey, blue), green), has_grey)
    unmasked_seed = mul(black, one_minus(has_grey))
    seed = add(masked_seed, unmasked_seed)

    unmasked_allowed = add(add(black, blue), green)
    allowed = add(
        mul(non_grey, has_grey),
        mul(unmasked_allowed, one_minus(has_grey)),
    )
    selected_present = node("gmax", [seed])

    kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kernel[0, 0, 1, 1] = 1
    kernel[0, 0, 0, 1] = 1
    kernel[0, 0, 2, 1] = 1
    kernel[0, 0, 1, 0] = 1
    kernel[0, 0, 1, 2] = 1
    flood_w = param(kernel)
    graph.append({
        "id": "fill",
        "op": "loop",
        "T": 128,
        "state": [{"id": "fill_state", "init": seed}],
        "body": [
            {
                "id": "fill_dilate",
                "op": "conv",
                "in": ["fill_state"],
                "w": flood_w,
            },
            {
                "id": "fill_next",
                "op": "min2",
                "in": ["fill_dilate", allowed],
            },
        ],
        "next": {"fill_state": "fill_next"},
    })
    selected = "fill.fill_state"

    priority = conv("coords", [[-4032.0, -63.0, 8192.0]])
    connector_mask = mul(add(blue, green), selected)
    observed = connector_coordinates(
        connector_mask, priority, one, row, col
    )

    # Memory:
    # 0:5 phases; 5:9 anchor; 9:13 anchor box; 13:17 second piece;
    # 17:19 second-piece click; 19:23 plan; 23:27 selected endpoints;
    # 27:31 previous selected box; 31 rotation counter.
    phase_bits = vsl("mem", 0, 5)
    phase0 = dense(phase_bits, [[-1, -1, -1, -1, -1]], [1])
    phase1 = vsl(phase_bits, 0, 1)
    phase2 = vsl(phase_bits, 1, 2)
    phase3 = vsl(phase_bits, 2, 3)
    phase4 = vsl(phase_bits, 3, 4)
    phase5 = vsl(phase_bits, 4, 5)

    old_anchor = vsl("mem", 5, 9)
    old_box = vsl("mem", 9, 13)
    old_b = vsl("mem", 13, 17)
    old_pointer = vsl("mem", 17, 19)
    old_plan = vsl("mem", 19, 23)
    old_source = vsl("mem", 23, 27)
    previous_box = vsl("mem", 27, 31)
    previous_rotations = vsl("mem", 31, 32)

    # Update remembered endpoints after movement; re-read after rotation
    # or selection.
    shift_w = np.zeros((4, 8), dtype=np.float32)
    shift_w[0, 1] = shift_w[2, 1] = -3
    shift_w[0, 2] = shift_w[2, 2] = 3
    shift_w[1, 3] = shift_w[3, 3] = -3
    shift_w[1, 4] = shift_w[3, 4] = 3
    predicted = va(old_source, dense("aprev", shift_w))

    refresh_w = np.zeros((1, 8), dtype=np.float32)
    refresh_w[0, 5] = 1
    refresh_w[0, 6] = 1
    refresh = clip(va(dense("aprev", refresh_w), phase0), 1)
    source = choose(predicted, observed, refresh)

    # Bounding box of the visible selected piece.
    inv_row = sub(conv("coords", [[0, 0, 64]]), row)
    inv_col = sub(conv("coords", [[0, 0, 64]]), col)
    max_y = node("gmax", [mul(selected, row)])
    max_x = node("gmax", [mul(selected, col)])
    min_y = dense(node("gmax", [mul(selected, inv_row)]), [[-1]], [64])
    min_x = dense(node("gmax", [mul(selected, inv_col)]), [[-1]], [64])
    current_box = node("vconcat", [min_y, max_y, min_x, max_x])

    unselected = sub(foreground, selected)
    other_present = node("gmax", [unselected])
    other_marker = first(unselected, priority, one)
    pointer_now = point_coordinates(other_marker, row, col)

    capture_anchor = vm(phase0, vm(selected_present, other_present))
    anchor = choose(old_anchor, source, capture_anchor)
    box = choose(old_box, current_box, capture_anchor)
    pointer = choose(old_pointer, pointer_now, capture_anchor)

    # Look outside the anchor box for a third piece.
    by0 = vsl(box, 0, 1)
    by1 = vsl(box, 1, 2)
    bx0 = vsl(box, 2, 3)
    bx1 = vsl(box, 3, 4)

    def spatial_ge(a, b):
        return node("min2", [
            node("relu", [add(sub(a, b), one)]), one
        ])

    inside_box = mul(
        mul(spatial_ge(row, by0), spatial_ge(by1, row)),
        mul(spatial_ge(col, bx0), spatial_ge(bx1, col)),
    )
    third_mask = mul(unselected, sub(one, inside_box))
    third_present = node("gmax", [third_mask])
    capture_b = vm(phase1, third_present)
    remembered_b = choose(old_b, source, capture_b)

    # Enumerate quarter-turn combinations for a three-piece loop.
    plan_input = node("vconcat", [anchor, remembered_b, source])
    attributes_w = np.zeros((64, 12), dtype=np.float32)
    errors_w = np.zeros((32, 12), dtype=np.float32)
    rotations = [
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        np.array([[0, 1], [-1, 0]], dtype=np.float32),
        np.array([[-1, 0], [0, -1]], dtype=np.float32),
        np.array([[0, -1], [1, 0]], dtype=np.float32),
    ]
    identity = np.eye(2, dtype=np.float32)

    for i in range(4):
        for j in range(4):
            k = 4 * i + j
            rb = rotations[i]
            rc = rotations[j]
            for d in range(2):
                attributes_w[4 * k + d, 4:6] = -rb[d]
                attributes_w[4 * k + d, 6:8] = rb[d]
                attributes_w[4 * k + 2 + d, 8:10] = -rc[d]
                attributes_w[4 * k + 2 + d, 10:12] = rc[d]
                errors_w[2 * k + d, 0:2] = identity[d]
                errors_w[2 * k + d, 2:4] = -identity[d]
                errors_w[2 * k + d, 4:6] = rb[d]
                errors_w[2 * k + d, 6:8] = -rb[d]
                errors_w[2 * k + d, 8:10] = -rc[d]
                errors_w[2 * k + d, 10:12] = rc[d]

    attributes = dense(plan_input, attributes_w)
    errors = absvec(dense(plan_input, errors_w), 32)
    score_w = np.zeros((16, 32), dtype=np.float32)
    for k in range(16):
        score_w[k, 2 * k:2 * k + 2] = 32
    scores = dense(errors, score_w, np.arange(16, dtype=np.float32))

    # Minimum-score tournament with deterministic candidate priority.
    minimum = scores
    count = 16
    while count > 1:
        half = count // 2
        even_w = np.zeros((half, count), dtype=np.float32)
        odd_w = np.zeros((half, count), dtype=np.float32)
        for k in range(half):
            even_w[k, 2 * k] = 1
            odd_w[k, 2 * k + 1] = 1
        even = dense(minimum, even_w)
        odd = dense(minimum, odd_w)
        absolute = absvec(va(even, neg(odd, half)), half)
        joined = node("vconcat", [even, odd, absolute])
        min_w = np.zeros((half, 3 * half), dtype=np.float32)
        for k in range(half):
            min_w[k, k] = 0.5
            min_w[k, half + k] = 0.5
            min_w[k, 2 * half + k] = -0.5
        minimum = dense(joined, min_w)
        count = half

    difference_from_minimum = va(scores, neg(minimum, 1))
    winner_raw = dense(
        difference_from_minimum,
        -np.eye(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
    )
    winner = node("vrelu", [winner_raw])
    expand_w = np.zeros((64, 16), dtype=np.float32)
    for k in range(16):
        expand_w[4 * k:4 * k + 4, k] = 1
    expanded = dense(winner, expand_w)
    selected_attributes = vm(attributes, expanded)
    collect_w = np.zeros((4, 64), dtype=np.float32)
    for k in range(16):
        for d in range(4):
            collect_w[d, 4 * k + d] = 1
    candidate_plan = dense(selected_attributes, collect_w)
    plan = choose(old_plan, candidate_plan, phase3)

    # Two-piece alignment with either endpoint pairing.
    dual_geometry = node("vconcat", [anchor, source])
    dual_errors = dense(dual_geometry, [
        [1, 0, -1, 0, -1, 0, 1, 0],
        [0, 1, 0, -1, 0, -1, 0, 1],
        [-1, 0, 1, 0, -1, 0, 1, 0],
        [0, -1, 0, 1, 0, -1, 0, 1],
    ])
    dual_abs = absvec(dual_errors, 4)
    dual_costs = dense(dual_abs, [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    direct = clip(dense(dual_costs, [[-1, 1]], [1]), 1)
    cost_difference = dense(dual_costs, [[1, -1]])
    absolute_cost_difference = absvec(cost_difference, 1)
    cost_features = node("vconcat", [dual_costs, absolute_cost_difference])
    min_cost = dense(cost_features, [[0.5, 0.5, -0.5]])
    dual_rotate = clip(dense(min_cost, [[1]], [-1]), 1)

    a1 = vsl(anchor, 0, 2)
    a2 = vsl(anchor, 2, 4)
    dual_target = va(vm(a1, direct), vm(a2, one_minus(direct)))
    dual_delta = target_delta(dual_target, vsl(source, 0, 2))
    dual_at_target = one_minus(nonzero(dual_delta, 2))
    dual_rotate = clip(va(dual_rotate, dual_at_target), 1)

    # Three-piece alignment.
    b_direction = vsl(plan, 0, 2)
    c_direction = vsl(plan, 2, 4)
    c_rotate, c_delta = rotation_and_delta(source, c_direction, a1)
    b_rotate, b_delta = rotation_and_delta(source, b_direction, a2)
    c_done = vm(one_minus(c_rotate), one_minus(nonzero(c_delta, 2)))
    b_at_target = one_minus(nonzero(b_delta, 2))
    b_rotate = clip(va(b_rotate, b_at_target), 1)

    initial_mask = add(
        mul(unselected, selected_present),
        mul(foreground, one_minus(selected_present)),
    )
    initial_click = mul(initial_mask, priority)
    third_click = mul(third_mask, priority)

    pointer_y = vsl(pointer, 0, 1)
    pointer_x = vsl(pointer, 1, 2)
    dy_pointer = node("abs", [sub(row, pointer_y)])
    dx_pointer = node("abs", [sub(col, pointer_x)])
    pointer_click = sub(
        conv("coords", [[0, 0, 8192]]),
        add(dy_pointer, dx_pointer),
    )

    use_third = vm(phase1, third_present)
    use_pointer = vm(phase4, c_done)
    normal_click = add(
        add(mul(initial_click, phase0), mul(third_click, use_third)),
        mul(pointer_click, use_pointer),
    )

    use_dual = vm(phase1, one_minus(third_present))
    delta = va(
        va(vm(dual_delta, use_dual), vm(c_delta, phase4)),
        vm(b_delta, phase5),
    )
    rotate = va(
        va(vm(dual_rotate, use_dual), phase2),
        va(phase3, va(vm(c_rotate, phase4), vm(b_rotate, phase5))),
    )
    click_action = va(phase0, va(use_third, use_pointer))
    action_features = node("vconcat", [delta, rotate, click_action])

    action_w = np.zeros((8, 4), dtype=np.float32)
    action_b = np.zeros(8, dtype=np.float32)
    action_b[0] = -2000
    action_w[1, 0] = -1
    action_w[2, 0] = 1
    action_w[3, 1] = -1
    action_w[4, 1] = 1
    action_w[5, 2] = 1000
    action_b[5] = -500
    action_w[6, 3] = 1000
    action_b[6] = -500
    action_b[7] = -2000
    normal_logits = dense(action_features, action_w, action_b)

    # Detect an unchanged selected box after a directional action, or
    # eight consecutive rotations. Recover by reset if it is available;
    # otherwise click another visible piece and clear the plan.
    previous_move_w = np.zeros((1, 8), dtype=np.float32)
    previous_move_w[0, 1:5] = 1
    previous_move = dense("aprev", previous_move_w)
    box_change = va(current_box, neg(previous_box, 4))
    blocked = vm(
        vm(previous_move, selected_present),
        one_minus(nonzero(box_change, 4)),
    )

    previous_rotate = vsl("aprev", 5, 6)
    rotation_count = vm(
        dense(previous_rotations, [[1]], [1]),
        previous_rotate,
    )
    rotation_stall = clip(dense(rotation_count, [[1]], [-7]), 1)
    trouble = clip(va(blocked, rotation_stall), 1)

    reset_available = vsl("avail", 0, 1)
    recovery_possible = clip(va(reset_available, other_present), 1)
    recovery = vm(trouble, recovery_possible)
    reset_recovery = vm(recovery, reset_available)
    click_recovery = vm(recovery, one_minus(reset_available))

    recovery_features = node(
        "vconcat", [reset_recovery, click_recovery]
    )
    recovery_w = np.zeros((8, 2), dtype=np.float32)
    recovery_w[0, 0] = 1000
    recovery_w[6, 1] = 1000
    recovery_logits = dense(
        recovery_features,
        recovery_w,
        -500 * np.ones(8, dtype=np.float32),
    )
    combined_logits = va(
        vm(normal_logits, one_minus(recovery)),
        vm(recovery_logits, recovery),
    )
    dense(combined_logits, np.eye(8, dtype=np.float32), name="logits")

    recovery_click = mul(unselected, priority)
    click_map = add(
        mul(normal_click, one_minus(recovery)),
        mul(recovery_click, recovery),
    )
    node("mul", [click_map, one], name="click")

    next_phase1 = va(capture_anchor, vm(phase1, one_minus(third_present)))
    next_phase2 = use_third
    next_phase3 = phase2
    next_phase4 = va(phase3, vm(phase4, one_minus(c_done)))
    next_phase5 = va(phase5, use_pointer)
    next_phases = node("vconcat", [
        next_phase1, next_phase2, next_phase3, next_phase4, next_phase5
    ])
    normal_memory = node("vconcat", [
        next_phases, anchor, box, remembered_b, pointer, plan, source,
        current_box, rotation_count,
    ])
    node("vmul", [normal_memory, one_minus(recovery)], name="mem_next")

    return {
        "graph": graph,
        "memory_size": 32,
        "params": params,
    }

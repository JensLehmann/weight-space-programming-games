import numpy as np


def build():
    graph = []
    params = {}
    node_no = 0
    param_no = 0

    def param(value):
        nonlocal param_no
        name = "p%d" % param_no
        param_no += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        nonlocal node_no
        name = "n%d" % node_no
        node_no += 1
        entry = {"id": name, "op": op, "in": list(inputs)}
        entry.update(fields)
        graph.append(entry)
        return name

    def spatial_affine(x, scale=1.0, bias=0.0):
        w = param(np.array([[[[scale]]]], dtype=np.float32))
        b = param(np.array([bias], dtype=np.float32))
        return node("conv", [x], w=w, b=b)

    def vector_affine(x, scale=1.0, bias=0.0):
        w = param(np.array([[scale]], dtype=np.float32))
        b = param(np.array([bias], dtype=np.float32))
        return node("dense", [x], w=w, b=b)

    def vsub(a, b):
        return node("vadd", [a, vector_affine(b, -1.0)])

    def vmul(a, b):
        return node("vmul", [a, b])

    def clip01(x):
        positive = node("vrelu", [x])
        above_one = node("vrelu", [vector_affine(x, 1.0, -1.0)])
        return vsub(positive, above_one)

    def sum_spatial(x, count):
        w = param(np.ones((1, count, 1, 1), dtype=np.float32))
        return node("conv", [x], w=w)

    def sum_vector(x, count):
        w = param(np.ones((1, count), dtype=np.float32))
        return node("dense", [x], w=w)

    def add_all(values):
        result = values[0]
        for value in values[1:]:
            result = node("vadd", [result, value])
        return result

    def max_spatial(values):
        result = values[0]
        for value in values[1:]:
            result = node("max2", [result, value])
        return result

    def integer_equal(x, value):
        w = param(np.ones((3, 1), dtype=np.float32))
        b = param(np.array(
            [1.0 - value, -float(value), -1.0 - value],
            dtype=np.float32,
        ))
        h = node("dense", [x], w=w, b=b)
        h = node("vrelu", [h])
        w2 = param(np.array([[1.0, -2.0, 1.0]], dtype=np.float32))
        return node("dense", [h], w=w2)

    row = node("slice", ["coords"], start=0, stop=1)
    col = node("slice", ["coords"], start=1, stop=2)
    spatial_one = node("slice", ["coords"], start=2, stop=3)

    below_raw = spatial_affine(row, 63.0, -52.0)
    below = node("min2", [node("relu", [below_raw]), spatial_one])
    board_mask = node("sub", [spatial_one, below])

    head_template = np.array([
        [6, 6, 6, 6, 6, 6],
        [6, 4, 4, 4, 4, 6],
        [6, 4, 6, 6, 4, -1],
        [6, 4, 6, 6, 4, -1],
        [6, 4, 4, 4, 4, 6],
        [6, 6, 6, 6, 6, 6],
    ], dtype=np.int32)

    def detect_head(rotation, selected):
        weights = np.zeros((2, 16, 7, 7), dtype=np.float32)
        biases = np.zeros(2, dtype=np.float32)
        for out, border in enumerate((6, 15)):
            template = head_template.copy()
            template[template == 6] = border
            if selected:
                template[template == 4] = 0
            template = np.rot90(template, k=-(rotation // 90))
            count = 0
            for y in range(6):
                for x in range(6):
                    color = int(template[y, x])
                    if color >= 0:
                        weights[out, color, y + 1, x + 1] = 2.0
                        count += 1
            biases[out] = 1.0 - 2.0 * count
        raw = node(
            "conv", ["x0"], w=param(weights), b=param(biases)
        )
        return sum_spatial(node("relu", [raw]), 2)

    selected_display_head = detect_head(0, True)
    unselected_display_head = detect_head(0, False)
    goal_head = node("mul", [selected_display_head, below])
    click_map = node("mul", [unselected_display_head, below])
    zero_map = node("sub", [spatial_one, spatial_one])
    graph.append({
        "id": "click",
        "op": "add",
        "in": [click_map, zero_map],
    })
    click_available = clip01(node("gmax", [click_map]))

    rotations = (0, 90, 180, 270)
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    board_heads = []
    orientation = []
    connected_maps = []

    stripe = np.full((6, 6), -1, dtype=np.int32)
    stripe[2] = np.array([2, 1, 1, 2, 1, 1])
    stripe[3] = np.array([1, 1, 2, 1, 1, 2])
    stripe_mask = np.zeros((6, 6), dtype=np.int32)
    stripe_mask[2, 0] = 1
    stripe_mask[3, 0] = 1
    stripe_mask[2, 5] = 1
    stripe_mask[3, 5] = 1

    for branch, (rotation, direction) in enumerate(
        zip(rotations, directions)
    ):
        head = detect_head(rotation, True)
        head = node("mul", [head, board_mask])
        board_heads.append(head)
        orientation.append(clip01(node("gmax", [head])))

        rotated_stripe = np.rot90(stripe, k=-(rotation // 90))
        rotated_mask = np.rot90(stripe_mask, k=-(rotation // 90))
        weights = np.zeros((1, 16, 7, 7), dtype=np.float32)
        for y in range(6):
            for x in range(6):
                if rotated_mask[y, x]:
                    color = int(rotated_stripe[y, x])
                    weights[0, color, y + 1, x + 1] = 1.0
        raw = node(
            "conv",
            ["x0"],
            w=param(weights),
            b=param(np.array([-3.0], dtype=np.float32)),
        )
        segment = node("relu", [raw])

        dx, dy = direction
        shift_array = np.zeros((1, 1, 7, 7), dtype=np.float32)
        shift_array[0, 0, 3 - 3 * dy, 3 - 3 * dx] = 1.0
        shift = param(shift_array)

        loop_id = "arm%d" % branch
        front = "front%d" % branch
        seen = "seen%d" % branch
        a = "stepa%d" % branch
        b = "stepb%d" % branch
        c = "newfront%d" % branch
        d = "newseen%d" % branch
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 10,
            "state": [
                {"id": front, "init": head},
                {"id": seen, "zeros": [1, 64, 64]},
            ],
            "body": [
                {"id": a, "op": "conv", "in": [front], "w": shift},
                {"id": b, "op": "conv", "in": [a], "w": shift},
                {"id": c, "op": "mul", "in": [b, segment]},
                {"id": d, "op": "max2", "in": [seen, c]},
            ],
            "next": {front: c, seen: d},
        })
        connected_maps.append(loop_id + "." + seen)

    connected = max_spatial(connected_maps)
    board_head = max_spatial(board_heads)

    horizontal = node("vadd", [orientation[0], orientation[2]])
    vertical = node("vadd", [orientation[1], orientation[3]])
    axis_h = node("mul", [row, horizontal])
    axis_v = node("mul", [col, vertical])
    axis = node("add", [axis_h, axis_v])
    head_axis = node("gmax", [node("mul", [board_head, axis])])
    head_row = node("gmax", [node("mul", [board_head, row])])
    head_col = node("gmax", [node("mul", [board_head, col])])

    rail2 = node("slice", ["x0"], start=2, stop=3)
    rail3 = node("slice", ["x0"], start=3, stop=4)
    rail = node("add", [rail2, rail3])

    def rail_sample(dx, dy):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        w[0, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        shift = param(w)
        sampled = node("conv", [rail], w=shift)
        sampled = node("conv", [sampled], w=shift)
        return clip01(node("gmax", [
            node("mul", [board_head, sampled])
        ]))

    rail_down = rail_sample(0, 1)
    rail_up = rail_sample(0, -1)
    rail_right = rail_sample(1, 0)
    rail_left = rail_sample(-1, 0)
    rail_positive = clip01(node("vadd", [
        vmul(horizontal, rail_down),
        vmul(vertical, rail_right),
    ]))
    rail_negative = clip01(node("vadd", [
        vmul(horizontal, rail_up),
        vmul(vertical, rail_left),
    ]))

    colors = (7, 8, 9, 10, 11, 12, 13, 14, 15)
    block_weights = np.zeros(
        (len(colors), 16, 7, 7), dtype=np.float32
    )
    for i, color in enumerate(colors):
        for ky, kx in ((2, 2), (2, 5), (5, 2), (5, 5)):
            block_weights[i, color, ky, kx] = 1.0
    block_bias = np.full(len(colors), -3.0, dtype=np.float32)
    blocks_raw = node(
        "conv", ["x0"], w=param(block_weights), b=param(block_bias)
    )
    blocks = node("relu", [blocks_raw])
    any_block = sum_spatial(blocks, len(colors))

    carried_map = node("mul", [any_block, connected])
    carried_count = vector_affine(node("gmean", [carried_map]), 4096.0)
    segment_count = vector_affine(node("gmean", [connected]), 4096.0)

    goal_shift_array = np.zeros((1, 1, 7, 7), dtype=np.float32)
    goal_shift_array[0, 0, 3, 0] = 1.0
    goal_shift = param(goal_shift_array)

    pointer = goal_head
    selected_pointer = None
    first_goal_pointer = None
    for i in range(8):
        pointer = node("conv", [pointer], w=goal_shift)
        pointer = node("conv", [pointer], w=goal_shift)
        if i == 0:
            first_goal_pointer = pointer
        gate = integer_equal(carried_count, i)
        gated = node("mul", [pointer, gate])
        if selected_pointer is None:
            selected_pointer = gated
        else:
            selected_pointer = node("add", [selected_pointer, gated])

    desired_at_goal = node("mul", [blocks, selected_pointer])
    desired_color = node("gmax", [desired_at_goal])
    matching_blocks = node("mul", [blocks, desired_color])
    matching_any = sum_spatial(matching_blocks, len(colors))
    not_connected = node("sub", [spatial_one, connected])
    free_matching = node("mul", [matching_any, not_connected])
    free_matching = node("mul", [free_matching, board_mask])
    target_present = clip01(node("gmax", [free_matching]))

    # Select a nearby matching block using both coordinates.
    drow = node("abs", [node("sub", [row, head_row])])
    dcol = node("abs", [node("sub", [col, head_col])])
    full_distance = node("add", [drow, dcol])
    candidate_score = node("sub", [free_matching, full_distance])
    best_score = node("gmax", [candidate_score])
    score_delta = node("abs", [
        node("sub", [candidate_score, best_score])
    ])
    nearest = node("relu", [
        spatial_affine(score_delta, -63.0, 1.0)
    ])
    nearest = node("mul", [nearest, free_matching])
    target_axis = node("gmax", [node("mul", [nearest, axis])])
    target_row = node("gmax", [node("mul", [nearest, row])])
    target_col = node("gmax", [node("mul", [nearest, col])])

    # Check the first carried colour against the first requested colour.
    first_score = node("sub", [carried_map, full_distance])
    best_first_score = node("gmax", [first_score])
    first_delta = node("abs", [
        node("sub", [first_score, best_first_score])
    ])
    first_nearest = node("relu", [
        spatial_affine(first_delta, -63.0, 1.0)
    ])
    first_nearest = node("mul", [first_nearest, carried_map])
    first_color = node("gmax", [
        node("mul", [blocks, first_nearest])
    ])
    first_goal_color = node("gmax", [
        node("mul", [blocks, first_goal_pointer])
    ])
    first_match = clip01(sum_vector(
        vmul(first_color, first_goal_color), len(colors)
    ))
    first_present = clip01(node("gmax", [carried_map]))

    previous_count = node("vslice", ["mem"], start=0, stop=1)
    previous_extend = node("vslice", ["mem"], start=1, stop=2)
    phase = node("vslice", ["mem"], start=2, stop=3)
    previous_scrub = node("vslice", ["mem"], start=3, stop=4)
    previous_axis = node("vslice", ["mem"], start=4, stop=5)
    previous_park = node("vslice", ["mem"], start=5, stop=6)
    previous_segments = node("vslice", ["mem"], start=6, stop=7)

    one = vector_affine(carried_count, 0.0, 1.0)
    dirty = vmul(first_present, vsub(one, first_match))
    scrub_start = vmul(dirty, vsub(one, previous_scrub))
    scrub_retract = vmul(dirty, previous_scrub)

    count_delta = vsub(carried_count, previous_count)
    count_abs = node("vadd", [
        node("vrelu", [count_delta]),
        node("vrelu", [vector_affine(count_delta, -1.0)]),
    ])
    unchanged = node("vrelu", [
        vector_affine(count_abs, -1.0, 1.0)
    ])
    continuing = vmul(previous_extend, unchanged)

    prev_actions = [
        node("vslice", ["aprev"], start=i, stop=i + 1)
        for i in range(8)
    ]
    backward_action = add_all([
        vmul(orientation[0], prev_actions[3]),
        vmul(orientation[1], prev_actions[1]),
        vmul(orientation[2], prev_actions[4]),
        vmul(orientation[3], prev_actions[2]),
    ])
    forward_action = add_all([
        vmul(orientation[0], prev_actions[4]),
        vmul(orientation[1], prev_actions[2]),
        vmul(orientation[2], prev_actions[3]),
        vmul(orientation[3], prev_actions[1]),
    ])

    lost = vmul(
        clip01(vsub(previous_count, carried_count)), backward_action
    )
    lost = vmul(lost, vsub(one, previous_scrub))
    lost = vmul(lost, vsub(one, dirty))

    phase1 = integer_equal(phase, 1)
    phase2 = integer_equal(phase, 2)
    phase4 = integer_equal(phase, 4)
    recovering = clip01(node("vadd", [phase1, phase2]))

    segment_delta = vsub(segment_count, previous_segments)
    segment_abs = node("vadd", [
        node("vrelu", [segment_delta]),
        node("vrelu", [vector_affine(segment_delta, -1.0)]),
    ])
    segments_unchanged = node("vrelu", [
        vector_affine(segment_abs, -1.0, 1.0)
    ])
    blocked_extension = vmul(previous_extend, segments_unchanged)
    blocked_extension = vmul(blocked_extension, unchanged)
    blocked_extension = vmul(blocked_extension, forward_action)
    blocked_extension = vmul(blocked_extension, target_present)
    blocked_extension = vmul(blocked_extension, vsub(one, dirty))
    blocked_extension = vmul(blocked_extension, vsub(one, phase4))

    # The detour state remains active only while the carried count is
    # unchanged. Its first action moves perpendicular to the arm.
    bridge = vmul(phase4, unchanged)
    bridge = vmul(bridge, target_present)
    exceptional = clip01(add_all([
        lost, recovering, dirty, blocked_extension, bridge
    ]))
    ordinary = vsub(one, exceptional)

    extra_segments = vsub(segment_count, carried_count)
    retract_needed = clip01(extra_segments)
    no_retract = vsub(one, retract_needed)

    axis_steps = vector_affine(vsub(target_axis, head_axis), 63.0)
    increase_needed = clip01(axis_steps)
    decrease_needed = clip01(vector_affine(axis_steps, -1.0))
    aligned = vsub(
        one, clip01(node("vadd", [increase_needed, decrease_needed]))
    )

    not_continuing = vsub(one, continuing)
    ready = vmul(no_retract, aligned)
    start_extension = vmul(not_continuing, ready)
    extension = clip01(node("vadd", [continuing, start_extension]))
    extension = vmul(extension, target_present)
    extension = vmul(extension, ordinary)

    retract = vmul(not_continuing, retract_needed)
    retract = vmul(retract, target_present)
    retract = vmul(retract, ordinary)
    retract = clip01(node("vadd", [retract, scrub_retract]))

    align_gate = vmul(not_continuing, no_retract)
    align_gate = vmul(align_gate, target_present)
    align_gate = vmul(align_gate, ordinary)
    increase = vmul(align_gate, increase_needed)
    decrease = vmul(align_gate, decrease_needed)

    no_target = vsub(one, target_present)
    no_click = vsub(one, click_available)
    fallback_retract = vmul(no_target, no_click)
    fallback_retract = vmul(fallback_retract, retract_needed)
    fallback_retract = vmul(fallback_retract, ordinary)
    retract = clip01(node("vadd", [retract, fallback_retract]))

    toward_negative = clip01(vector_affine(head_axis, 63.0, -32.0))
    toward_positive = vsub(one, toward_negative)
    recovery_increase = vmul(recovering, toward_positive)
    recovery_decrease = vmul(recovering, toward_negative)

    scrub_increase = vmul(scrub_start, rail_positive)
    scrub_decrease = vmul(scrub_start, vsub(one, rail_positive))
    scrub_decrease = vmul(scrub_decrease, rail_negative)
    no_scrub_side = vsub(
        one, clip01(node("vadd", [scrub_increase, scrub_decrease]))
    )
    scrub_immediate_retract = vmul(scrub_start, no_scrub_side)
    retract = clip01(node("vadd", [retract, scrub_immediate_retract]))

    # Project the target distance onto the arm direction.
    row_forward = vsub(target_row, head_row)
    col_forward = vsub(target_col, head_col)
    signed_forward = add_all([
        vmul(orientation[0], col_forward),
        vmul(orientation[1], row_forward),
        vmul(orientation[2], vector_affine(col_forward, -1.0)),
        vmul(orientation[3], vector_affine(row_forward, -1.0)),
    ])
    forward_pixels = vector_affine(signed_forward, 63.0)
    tip_pixels = vector_affine(segment_count, 6.0)
    bridge_more = clip01(vsub(forward_pixels, tip_pixels))
    bridge_extend = vmul(bridge, bridge_more)
    bridge_approach = vmul(bridge, vsub(one, bridge_more))
    bridge_increase = vmul(bridge_approach, increase_needed)
    bridge_decrease = vmul(bridge_approach, decrease_needed)

    detour_increase = vmul(blocked_extension, rail_positive)
    detour_decrease = vmul(
        blocked_extension, vsub(one, rail_positive)
    )
    detour_decrease = vmul(detour_decrease, rail_negative)
    detour_started = clip01(node("vadd", [
        detour_increase, detour_decrease
    ]))

    extension = clip01(node("vadd", [extension, bridge_extend]))

    park_request = vmul(no_target, click_available)
    park_request = vmul(park_request, ordinary)
    axis_change = vsub(head_axis, previous_axis)
    axis_change_abs = node("vadd", [
        node("vrelu", [axis_change]),
        node("vrelu", [vector_affine(axis_change, -1.0)]),
    ])
    moved = clip01(vector_affine(axis_change_abs, 126.0))
    park_stuck = vmul(previous_park, vsub(one, moved))
    can_park = vmul(rail_positive, vsub(one, park_stuck))
    park_move = vmul(park_request, can_park)
    switch = vmul(park_request, vsub(one, can_park))

    increase = clip01(add_all([
        increase,
        recovery_increase,
        scrub_increase,
        park_move,
        detour_increase,
        bridge_increase,
    ]))
    decrease = clip01(add_all([
        decrease,
        recovery_decrease,
        scrub_decrease,
        detour_decrease,
        bridge_decrease,
    ]))

    up = add_all([
        vmul(orientation[0], decrease),
        vmul(orientation[1], retract),
        vmul(orientation[2], decrease),
        vmul(orientation[3], extension),
    ])
    down = add_all([
        vmul(orientation[0], increase),
        vmul(orientation[1], extension),
        vmul(orientation[2], increase),
        vmul(orientation[3], retract),
    ])
    left = add_all([
        vmul(orientation[0], retract),
        vmul(orientation[1], decrease),
        vmul(orientation[2], extension),
        vmul(orientation[3], decrease),
    ])
    right = add_all([
        vmul(orientation[0], extension),
        vmul(orientation[1], increase),
        vmul(orientation[2], retract),
        vmul(orientation[3], increase),
    ])

    next_phase = add_all([
        lost,
        vector_affine(phase1, 2.0),
        vector_affine(detour_started, 4.0),
        vector_affine(bridge_extend, 4.0),
    ])
    memory = node("vconcat", [
        carried_count,
        extension,
        next_phase,
        dirty,
        head_axis,
        park_move,
        segment_count,
    ])
    graph.append({
        "id": "mem_next",
        "op": "dense",
        "in": [memory],
        "w": param(np.eye(7, dtype=np.float32)),
    })

    decisions = node(
        "vconcat", [up, down, left, right, switch, lost]
    )
    output_weights = np.zeros((8, 6), dtype=np.float32)
    output_weights[1, 0] = 1.0
    output_weights[2, 1] = 1.0
    output_weights[3, 2] = 1.0
    output_weights[4, 3] = 1.0
    output_weights[6, 4] = 1.0
    output_weights[7, 5] = 1.0
    output_bias = np.array(
        [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        dtype=np.float32,
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [decisions],
        "w": param(output_weights),
        "b": param(output_bias),
    })

    return {"graph": graph, "memory_size": 7, "params": params}

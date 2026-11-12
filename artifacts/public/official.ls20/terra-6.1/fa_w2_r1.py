import numpy as np


def build():
    graph = []
    params = {}
    node_counter = 0
    param_counter = 0

    def param(value):
        nonlocal param_counter
        name = "p%d" % param_counter
        param_counter += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def op(kind, *inputs, **fields):
        nonlocal node_counter
        name = "v%d" % node_counter
        node_counter += 1
        graph.append({"id": name, "op": kind, "in": list(inputs), **fields})
        return name

    def conv(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return op("conv", x, **fields)

    def dense(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return op("dense", x, **fields)

    def sl(x, a, b):
        return op("slice", x, start=a, stop=b)

    def vsl(x, a, b):
        return op("vslice", x, start=a, stop=b)

    def cat(*xs):
        return op("concat", *xs)

    def vcat(*xs):
        return op("vconcat", *xs)

    def add(a, b):
        return op("add", a, b)

    def sub(a, b):
        return op("sub", a, b)

    def mul(a, b):
        return op("mul", a, b)

    def maximum(a, b):
        return op("max2", a, b)

    def minimum(a, b):
        return op("min2", a, b)

    def relu(x):
        return op("relu", x)

    def scalar_inverse(x):
        return dense(x, [[-1.0]], [1.0])

    def spatial_linear(x, scale, bias=0.0):
        return conv(x, np.array([[[[scale]]]], dtype=np.float32), [bias])

    one = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)

    # Broad region containing the playable board.
    rw = np.zeros((4, 3, 1, 1), dtype=np.float32)
    rw[0, 1, 0, 0] = 63.0
    rw[1, 1, 0, 0] = -63.0
    rw[2, 0, 0, 0] = 63.0
    rw[3, 0, 0, 0] = -63.0
    rp = relu(conv("coords", rw, [-4.0, 58.0, 1.0, 54.0]))
    rectangle = minimum(
        minimum(
            minimum(sl(rp, 0, 1), sl(rp, 1, 2)),
            minimum(sl(rp, 2, 3), sl(rp, 3, 4)),
        ),
        one,
    )

    # Locate the centre of the 5x5 player.
    pw = np.zeros((1, 16, 5, 5), dtype=np.float32)
    pw[0, 12, 0:2, :] = 2.0
    pw[0, 9, 2:5, :] = 2.0
    player = mul(relu(conv("x0", pw, [-49.0])), rectangle)

    # Glyph-colour occupancy.
    palette = [12, 9, 14, 8]
    cw = np.zeros((4, 16, 1, 1), dtype=np.float32)
    for i, colour in enumerate(palette):
        cw[i, colour, 0, 0] = 1.0
    colours = conv("x0", cw)
    occupancy = conv(colours, np.ones((1, 4, 1, 1), dtype=np.float32))

    # Sample the nine pixels of the enlarged glyph in the lower-left HUD.
    aw = np.zeros((2, 3, 1, 1), dtype=np.float32)
    aw[0, 0, 0, 0] = 63.0
    aw[1, 1, 0, 0] = 63.0
    delta = op("abs", conv("coords", aw, [-57.0, -5.0]))
    anchor = relu(
        conv(delta, -np.ones((1, 2, 1, 1), dtype=np.float32), [1.0])
    )

    uiw = np.zeros((9, 1, 7, 7), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            uiw[3 * r + c, 0, 1 + 2 * r, 1 + 2 * c] = 1.0
    ui_bits = op("gmax", mul(conv(occupancy, uiw), anchor))

    uicw = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for k in range(4):
        for r in range(3):
            for c in range(3):
                uicw[k, k, 1 + 2 * r, 1 + 2 * c] = 1.0
    ui_colour = op(
        "gmax", mul(minimum(conv(colours, uicw), one), anchor)
    )

    # Detect goal centres by their black ring and coloured glyph.
    goal_input = cat(sl("x0", 5, 6), occupancy)
    gw = np.zeros((2, 2, 5, 5), dtype=np.float32)
    for r in range(5):
        for c in range(5):
            if r in (0, 4) or c in (0, 4):
                gw[0, 0, r, c] = 2.0
            if 1 <= r <= 3 and 1 <= c <= 3:
                gw[1, 1, r, c] = 2.0
    gt = minimum(relu(conv(goal_input, gw, [-31.0, -7.0])), one)
    goal = mul(mul(sl(gt, 0, 1), sl(gt, 1, 2)), rectangle)

    gbw = np.zeros((9, 1, 3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            gbw[3 * r + c, 0, r, c] = 1.0
    goal_bits = conv(occupancy, gbw)

    matches = []
    for turns in range(4):
        permutation = np.rot90(
            np.arange(9).reshape(3, 3), k=-turns
        ).reshape(-1)
        matrix = np.zeros((9, 9), dtype=np.float32)
        for out_index, in_index in enumerate(permutation):
            matrix[out_index, in_index] = 1.0
        rotated = dense(ui_bits, matrix)
        difference = op("abs", sub(goal_bits, rotated))
        match = relu(
            conv(
                difference,
                -np.ones((1, 9, 1, 1), dtype=np.float32),
                [1.0],
            )
        )
        matches.append(match)

    same_shape = maximum(
        maximum(matches[0], matches[1]),
        maximum(matches[2], matches[3]),
    )

    gcw = np.zeros((4, 4, 3, 3), dtype=np.float32)
    for k in range(4):
        gcw[k, k, :, :] = 1.0
    goal_colour = minimum(conv(colours, gcw), one)
    colour_difference = op("abs", sub(goal_colour, ui_colour))
    same_colour = relu(
        conv(
            colour_difference,
            -np.ones((1, 4, 1, 1), dtype=np.float32),
            [1.0],
        )
    )

    shape_goal = mul(goal, same_shape)
    shape_colour_goal = mul(shape_goal, same_colour)
    exact_goal = mul(mul(goal, matches[0]), same_colour)
    goal_flags = vcat(
        op("gmax", goal),
        op("gmax", shape_goal),
        op("gmax", shape_colour_goal),
        op("gmax", exact_goal),
    )
    # Channels: exact goal, rotation switch, colour switch, shape switch.
    need = dense(
        goal_flags,
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
        ],
    )

    def pattern_detector(pattern):
        w = np.zeros((1, 16, 3, 3), dtype=np.float32)
        for r in range(3):
            for c in range(3):
                w[0, pattern[r][c], r, c] = 2.0
        return mul(relu(conv("x0", w, [-17.0])), rectangle)

    shape_switch = pattern_detector(
        [[0, 3, 3], [3, 0, 0], [3, 0, 3]]
    )
    colour_switch = pattern_detector(
        [[9, 14, 14], [9, 0, 8], [12, 12, 8]]
    )
    rotation_switch = pattern_detector(
        [[3, 0, 3], [1, 0, 0], [3, 1, 3]]
    )

    # Remember switch positions when they are obscured by the player.
    remembered_maps = []
    memory_parts = []
    for i, marker in enumerate(
        [shape_switch, colour_switch, rotation_switch]
    ):
        present = op("gmax", marker)
        old_row = vsl("mem", 2 * i, 2 * i + 1)
        old_col = vsl("mem", 2 * i + 1, 2 * i + 2)
        new_row = op("gmax", mul(marker, row))
        new_col = op("gmax", mul(marker, col))
        old_pair = vcat(old_row, old_col)
        new_pair = vcat(new_row, new_col)
        memory_parts.append(
            op(
                "vadd",
                op("vmul", new_pair, present),
                op("vmul", old_pair, scalar_inverse(present)),
            )
        )

        dr = op("abs", sub(row, old_row))
        dc = op("abs", sub(col, old_col))
        remembered = relu(
            conv(
                cat(dr, dc),
                np.array([[[[-63.0]], [[-63.0]]]], dtype=np.float32),
                [1.0],
            )
        )
        remembered = mul(
            mul(remembered, rectangle), scalar_inverse(present)
        )
        remembered_maps.append(maximum(marker, remembered))

    shape_map, colour_map, rotation_map = remembered_maps

    # Recharge marker and low-counter override.
    bw = np.zeros((1, 16, 3, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            bw[0, 3 if (r == 1 and c == 1) else 11, r, c] = 2.0
    booster = mul(relu(conv("x0", bw, [-17.0])), rectangle)
    booster_present = op("gmax", booster)

    row61_weights = np.zeros((1, 3, 1, 1), dtype=np.float32)
    row61_weights[0, 0, 0, 0] = 63.0
    row61_delta = op("abs", conv("coords", row61_weights, [-61.0]))
    row61 = relu(spatial_linear(row61_delta, -1.0, 1.0))
    bar_pixels = mul(sl("x0", 11, 12), row61)
    bar_count = dense(op("gmean", bar_pixels), [[4096.0]])
    low_parts = op(
        "vrelu",
        dense(bar_count, [[-1.0], [-1.0]], [20.0, 19.0]),
    )
    low = dense(low_parts, [[1.0, -1.0]])
    use_booster = op("vmul", low, booster_present)
    no_booster = scalar_inverse(use_booster)

    rotation_need = vsl(need, 1, 2)
    colour_need = vsl(need, 2, 3)
    shape_need = vsl(need, 3, 4)

    normal_target = maximum(
        maximum(exact_goal, mul(rotation_map, rotation_need)),
        maximum(mul(colour_map, colour_need), mul(shape_map, shape_need)),
    )
    selected_target = maximum(
        mul(normal_target, no_booster),
        mul(booster, use_booster),
    )

    shape_active = op("vmul", shape_need, no_booster)
    colour_active = op("vmul", colour_need, no_booster)
    rotation_active = op("vmul", rotation_need, no_booster)
    unwanted = maximum(
        maximum(
            mul(shape_map, scalar_inverse(shape_active)),
            mul(colour_map, scalar_inverse(colour_active)),
        ),
        mul(rotation_map, scalar_inverse(rotation_active)),
    )
    switch_penalty = spatial_linear(unwanted, -0.6, 1.0)

    wall = sl("x0", 4, 5)
    locked_goal = sub(goal, exact_goal)
    free = relu(sub(sub(one, wall), locked_goal))
    passability = mul(mul(free, rectangle), switch_penalty)
    source = maximum(selected_target, spatial_linear(rectangle, 1.0e-5))
    initial = mul(source, passability)

    # Five-pixel shifts. Direction order is up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift3 = np.zeros((4, 1, 7, 7), dtype=np.float32)
    shift2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    shift3_four = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for k, (dy, dx) in enumerate(directions):
        shift3[k, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        shift2[k, k, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        shift3_four[k, k, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    shift3_name = param(shift3)
    shift2_name = param(shift2)
    shift3_four_name = param(shift3_four)
    discount_name = param(np.array([[[[0.98]]]], dtype=np.float32))
    sum4_name = param(np.ones((1, 4, 1, 1), dtype=np.float32))

    # Detect pusher faces. For a pusher directed d, the blue face is
    # three pixels behind the centre of the adjacent entry tile, with
    # colour 4 immediately behind the face.
    facew = np.zeros((4, 16, 3, 3), dtype=np.float32)
    for k, (dy, dx) in enumerate(directions):
        facew[k, 1, 1, 1] = 2.0
        facew[k, 4, 1 - dy, 1 - dx] = 2.0
    face_pixels = relu(conv("x0", facew, [-3.0] * 4))

    face_shift = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for k, (dy, dx) in enumerate(directions):
        face_shift[k, k, 3 - 3 * dy, 3 - 3 * dx] = 1.0
    triggers = mul(conv(face_pixels, face_shift), rectangle)

    # The simulator's pusher scans for a wall or goal in its direction.
    stop_map = minimum(add(wall, goal), one)
    graph.append(
        {
            "id": "stop_shift3",
            "op": "conv",
            "in": [stop_map],
            "w": shift3_name,
        }
    )
    graph.append(
        {
            "id": "next_stop",
            "op": "conv",
            "in": ["stop_shift3"],
            "w": shift2_name,
        }
    )
    no_next_stop = sub(one, "next_stop")

    # Mark whether a stop is within the simulator's 11-cell scan.
    graph.append(
        {
            "id": "stop_scan",
            "op": "loop",
            "T": 10,
            "state": [{"id": "seen_state", "init": "next_stop"}],
            "body": [
                {
                    "id": "seen_shift3",
                    "op": "conv",
                    "in": ["seen_state"],
                    "w": shift3_four_name,
                },
                {
                    "id": "seen_shift5",
                    "op": "conv",
                    "in": ["seen_shift3"],
                    "w": shift2_name,
                },
                {
                    "id": "seen_next",
                    "op": "max2",
                    "in": ["next_stop", "seen_shift5"],
                },
            ],
            "next": {"seen_state": "seen_next"},
        }
    )
    active_triggers = mul(triggers, "stop_scan.seen_state")
    any_trigger = minimum(
        conv(active_triggers, np.ones((1, 4, 1, 1), dtype=np.float32)),
        one,
    )
    no_trigger = sub(one, any_trigger)

    # Value iteration with a parallel four-direction endpoint field.
    # endpoint[d,x] becomes the value of the last cell before the first
    # stop in direction d. Entering a pusher uses that endpoint value.
    graph.append(
        {
            "id": "wave",
            "op": "loop",
            "T": 128,
            "state": [
                {"id": "wave_state", "init": initial},
                {"id": "endpoint_state", "zeros": [4, 64, 64]},
            ],
            "body": [
                {
                    "id": "endpoint_selected",
                    "op": "mul",
                    "in": ["endpoint_state", active_triggers],
                },
                {
                    "id": "endpoint_sum",
                    "op": "conv",
                    "in": ["endpoint_selected"],
                    "w": sum4_name,
                },
                {
                    "id": "ordinary_value",
                    "op": "mul",
                    "in": ["wave_state", no_trigger],
                },
                {
                    "id": "effective_value",
                    "op": "add",
                    "in": ["ordinary_value", "endpoint_sum"],
                },
                {
                    "id": "wave_shift3",
                    "op": "conv",
                    "in": ["effective_value"],
                    "w": shift3_name,
                },
                {
                    "id": "wave_shift5",
                    "op": "conv",
                    "in": ["wave_shift3"],
                    "w": shift2_name,
                },
                {
                    "id": "wave_n",
                    "op": "slice",
                    "in": ["wave_shift5"],
                    "start": 0,
                    "stop": 1,
                },
                {
                    "id": "wave_s",
                    "op": "slice",
                    "in": ["wave_shift5"],
                    "start": 1,
                    "stop": 2,
                },
                {
                    "id": "wave_w",
                    "op": "slice",
                    "in": ["wave_shift5"],
                    "start": 2,
                    "stop": 3,
                },
                {
                    "id": "wave_e",
                    "op": "slice",
                    "in": ["wave_shift5"],
                    "start": 3,
                    "stop": 4,
                },
                {
                    "id": "wave_ns",
                    "op": "max2",
                    "in": ["wave_n", "wave_s"],
                },
                {
                    "id": "wave_we",
                    "op": "max2",
                    "in": ["wave_w", "wave_e"],
                },
                {
                    "id": "wave_best",
                    "op": "max2",
                    "in": ["wave_ns", "wave_we"],
                },
                {
                    "id": "wave_discounted",
                    "op": "conv",
                    "in": ["wave_best"],
                    "w": discount_name,
                },
                {
                    "id": "wave_reached",
                    "op": "max2",
                    "in": [initial, "wave_discounted"],
                },
                {
                    "id": "wave_next",
                    "op": "mul",
                    "in": ["wave_reached", passability],
                },
                {
                    "id": "endpoint_shift3",
                    "op": "conv",
                    "in": ["endpoint_state"],
                    "w": shift3_four_name,
                },
                {
                    "id": "endpoint_shift5",
                    "op": "conv",
                    "in": ["endpoint_shift3"],
                    "w": shift2_name,
                },
                {
                    "id": "endpoint_here",
                    "op": "mul",
                    "in": ["wave_state", "next_stop"],
                },
                {
                    "id": "endpoint_far",
                    "op": "mul",
                    "in": ["endpoint_shift5", no_next_stop],
                },
                {
                    "id": "endpoint_next",
                    "op": "add",
                    "in": ["endpoint_here", "endpoint_far"],
                },
            ],
            "next": {
                "wave_state": "wave_next",
                "endpoint_state": "endpoint_next",
            },
        }
    )

    # Read the effective values of the four neighbouring tiles.
    endpoint_selected = mul("wave.endpoint_state", active_triggers)
    endpoint_sum = conv(
        endpoint_selected, np.ones((1, 4, 1, 1), dtype=np.float32)
    )
    ordinary_value = mul("wave.wave_state", no_trigger)
    effective = add(ordinary_value, endpoint_sum)
    graph.append(
        {
            "id": "read_shift3",
            "op": "conv",
            "in": [effective],
            "w": shift3_name,
        }
    )
    graph.append(
        {
            "id": "read_shift5",
            "op": "conv",
            "in": ["read_shift3"],
            "w": shift2_name,
        }
    )
    direction_values = op("gmax", mul("read_shift5", player))

    logit_weights = np.zeros((8, 4), dtype=np.float32)
    for i in range(4):
        logit_weights[i + 1, i] = 1.0
    logit_bias = [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
    graph.append(
        {
            "id": "logits",
            "op": "dense",
            "in": [direction_values],
            "w": param(logit_weights),
            "b": param(logit_bias),
        }
    )
    graph.append(
        {
            "id": "click",
            "op": "slice",
            "in": ["coords"],
            "start": 2,
            "stop": 3,
        }
    )
    graph.append({"id": "mem_next", "op": "vconcat", "in": memory_parts})

    return {"graph": graph, "memory_size": 6, "params": params}

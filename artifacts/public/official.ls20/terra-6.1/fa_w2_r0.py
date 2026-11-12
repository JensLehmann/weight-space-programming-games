import numpy as np


def build():
    graph = []
    params = {}
    ni = 0
    pi = 0

    def param(value):
        nonlocal pi
        name = "p%d" % pi
        pi += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        nonlocal ni
        name = "n%d" % ni
        ni += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def ss(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def clip_s(x, one):
        return node("min2", [node("relu", [x]), one])

    def clip_v(x, size):
        upper = dense(
            x,
            np.eye(size, dtype=np.float32),
            -np.ones(size, dtype=np.float32),
        )
        return node("vadd", [
            node("vrelu", [x]),
            dense(
                node("vrelu", [upper]),
                -np.eye(size, dtype=np.float32),
            ),
        ])

    def inv(x):
        return dense(x, [[-1.0]], [1.0])

    def abs_v(x, size):
        neg = dense(x, -np.eye(size, dtype=np.float32))
        return node("vadd", [
            node("vrelu", [x]),
            node("vrelu", [neg]),
        ])

    def max4(parts):
        a = node("max2", [parts[0], parts[1]])
        b = node("max2", [parts[2], parts[3]])
        return node("max2", [a, b])

    one = ss("coords", 2, 3)
    row = ss("coords", 0, 1)
    col = ss("coords", 1, 2)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Player position.
    player_w = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for y in range(5):
        for x in range(5):
            player_w[0, 12 if y < 2 else 9, y, x] = 2.0
    player = node("relu", [conv("x0", player_w, [-49.0])])

    # Playable board region and visible walls.
    domain_w = np.zeros((4, 3, 1, 1), dtype=np.float32)
    domain_w[0, 1, 0, 0] = 63.0
    domain_w[1, 1, 0, 0] = -63.0
    domain_w[2, 0, 0, 0] = 63.0
    domain_w[3, 0, 0, 0] = -63.0
    domain = node("relu", [
        conv("coords", domain_w, [-5.0, 57.0, -1.0, 58.0])
    ])
    d = [ss(domain, i, i + 1) for i in range(4)]
    domain = node("min2", [d[0], d[1]])
    domain = node("min2", [domain, d[2]])
    domain = node("min2", [domain, d[3]])
    domain = node("min2", [domain, one])
    wall = ss("x0", 4, 5)
    base_open = node("mul", [node("sub", [one, wall]), domain])

    # Five-pixel cardinal shifts, implemented as shifts of three and two.
    neighbor3 = np.zeros((4, 1, 7, 7), dtype=np.float32)
    neighbor2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        neighbor3[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        neighbor2[i, i, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    neighbor3_name = param(neighbor3)
    neighbor2_name = param(neighbor2)

    shapes = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]
    colors = [12, 9, 14, 8]

    glyphs = {}
    for shape_id, shape in enumerate(shapes):
        base = np.asarray(shape, dtype=np.int8)
        for rotation in range(4):
            bits = tuple(int(v) for v in np.rot90(base, -rotation).ravel())
            for color_id, color in enumerate(colors):
                glyphs[(color, bits)] = (shape_id, rotation, color_id)

    keys = list(glyphs)
    labels = [glyphs[k] for k in keys]
    ntypes = len(keys)

    goal_w = np.zeros((ntypes, 16, 5, 5), dtype=np.float32)
    ui_w = np.zeros((ntypes, 16, 5, 5), dtype=np.float32)
    for i, (color, bits) in enumerate(keys):
        bits = np.asarray(bits, dtype=np.int8).reshape(3, 3)
        for y in range(5):
            for x in range(5):
                if y in (0, 4) or x in (0, 4):
                    goal_w[i, 5, y, x] = 2.0
        for y in range(3):
            for x in range(3):
                c = color if bits[y, x] else 5
                goal_w[i, c, y + 1, x + 1] = 2.0
                ui_w[i, c, 2 * y, 2 * x] = 2.0

    goals = node("relu", [
        conv("x0", goal_w, np.full(ntypes, -49.0))
    ])
    ui = node("relu", [
        conv("x0", ui_w, np.full(ntypes, -17.0))
    ])

    ui_xy_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    ui_xy_w[0, 0, 0, 0] = 63.0
    ui_xy_w[1, 1, 0, 0] = 63.0
    ui_xy = node("abs", [
        conv("coords", ui_xy_w, [-57.0, -5.0])
    ])
    ui_mask_w = np.zeros((1, 2, 1, 1), dtype=np.float32)
    ui_mask_w[0, :, 0, 0] = -2.0
    ui_mask = clip_s(conv(ui_xy, ui_mask_w, [2.0]), one)
    ui = node("mul", [ui, ui_mask])

    code_w = np.arange(
        1, ntypes + 1, dtype=np.float32
    ).reshape(1, ntypes, 1, 1)
    goal_code_map = conv(goals, code_w)
    ui_code_map = conv(ui, code_w)
    all_goals = node("min2", [goal_code_map, one])

    # Flood from the player without crossing an unsolved goal. A goal is
    # eligible if one of its neighboring cells is reached.
    reach_open = node("mul", [
        base_open, node("sub", [one, all_goals])
    ])
    graph.append({
        "id": "reach_wave",
        "op": "loop",
        "T": 48,
        "state": [{"id": "reach_state", "init": player}],
        "body": [
            {"id": "reach_shift3", "op": "conv",
             "in": ["reach_state"], "w": neighbor3_name},
            {"id": "reach_shift2", "op": "conv",
             "in": ["reach_shift3"], "w": neighbor2_name},
            {"id": "reach_u", "op": "slice",
             "in": ["reach_shift2"], "start": 0, "stop": 1},
            {"id": "reach_d", "op": "slice",
             "in": ["reach_shift2"], "start": 1, "stop": 2},
            {"id": "reach_l", "op": "slice",
             "in": ["reach_shift2"], "start": 2, "stop": 3},
            {"id": "reach_r", "op": "slice",
             "in": ["reach_shift2"], "start": 3, "stop": 4},
            {"id": "reach_ud", "op": "max2",
             "in": ["reach_u", "reach_d"]},
            {"id": "reach_lr", "op": "max2",
             "in": ["reach_l", "reach_r"]},
            {"id": "reach_neighbors", "op": "max2",
             "in": ["reach_ud", "reach_lr"]},
            {"id": "reach_keep", "op": "max2",
             "in": ["reach_state", "reach_neighbors"]},
            {"id": "reach_next", "op": "mul",
             "in": ["reach_keep", reach_open]},
        ],
        "next": {"reach_state": "reach_next"},
    })
    reach_neighbors = node(
        "conv", ["reach_wave.reach_state"], w=neighbor3_name
    )
    reach_neighbors = node(
        "conv", [reach_neighbors], w=neighbor2_name
    )
    reach_neighbors = max4([
        ss(reach_neighbors, i, i + 1) for i in range(4)
    ])
    reachable_codes = node("mul", [goal_code_map, reach_neighbors])
    reachable_max = node("gmax", [reachable_codes])
    has_reachable = clip_v(dense(reachable_max, [[100.0]]), 1)
    fallback_codes = node("mul", [goal_code_map, inv(has_reachable)])
    chosen_codes = node("max2", [reachable_codes, fallback_codes])
    goal_code = node("gmax", [chosen_codes])
    ui_code = node("gmax", [ui_code_map])

    # Decode the selected goal and UI glyphs into attributes.
    decode_a = np.ones((3 * ntypes, 1), dtype=np.float32)
    decode_b = np.zeros(3 * ntypes, dtype=np.float32)
    decode_c = np.zeros((ntypes, 3 * ntypes), dtype=np.float32)
    for i in range(ntypes):
        k = i + 1
        decode_b[3 * i:3 * i + 3] = [-(k - 1), -k, -(k + 1)]
        decode_c[i, 3 * i:3 * i + 3] = [1.0, -2.0, 1.0]
    da = param(decode_a)
    db = param(decode_b)
    dc = param(decode_c)

    def decode(x):
        y = node("dense", [x], w=da, b=db)
        y = node("vrelu", [y])
        return node("dense", [y], w=dc)

    goal_type = decode(goal_code)
    ui_type = decode(ui_code)

    feature_w = np.zeros((14, ntypes), dtype=np.float32)
    for i, (shape_id, rotation, color_id) in enumerate(labels):
        feature_w[shape_id, i] = 1.0
        feature_w[6 + rotation, i] = 1.0
        feature_w[10 + color_id, i] = 1.0
    fw = param(feature_w)
    goal_features = node("dense", [goal_type], w=fw)
    ui_features = node("dense", [ui_type], w=fw)
    features = node("vconcat", [ui_features, goal_features])

    delta_w = np.zeros((28, 28), dtype=np.float32)
    for i in range(14):
        delta_w[i, i] = 1.0
        delta_w[i, 14 + i] = -1.0
        delta_w[14 + i, i] = -1.0
        delta_w[14 + i, 14 + i] = 1.0
    delta = node("vrelu", [dense(features, delta_w)])

    mismatch_w = np.zeros((3, 28), dtype=np.float32)
    for i in list(range(0, 6)) + list(range(14, 20)):
        mismatch_w[0, i] = 0.5
    for i in list(range(10, 14)) + list(range(24, 28)):
        mismatch_w[1, i] = 0.5
    for i in list(range(6, 10)) + list(range(20, 24)):
        mismatch_w[2, i] = 0.5
    mismatch = dense(delta, mismatch_w)
    shape_bad = vs(mismatch, 0, 1)
    color_bad = vs(mismatch, 1, 2)
    rotation_bad = vs(mismatch, 2, 3)

    shape_good = inv(shape_bad)
    color_good = inv(color_bad)
    rotation_good = inv(rotation_bad)
    use_shape = shape_bad
    use_color = node("vmul", [shape_good, color_bad])
    shape_color_good = node("vmul", [shape_good, color_good])
    use_rotation = node("vmul", [shape_color_good, rotation_bad])
    use_goal = node("vmul", [shape_color_good, rotation_good])

    tile_patterns = [
        [[0, 3, 3], [3, 0, 0], [3, 0, 3]],
        [[9, 14, 14], [9, 0, 8], [12, 12, 8]],
        [[3, 0, 3], [1, 0, 0], [3, 1, 3]],
        [[11, 11, 11], [11, 3, 11], [11, 11, 11]],
    ]
    tile_w = np.zeros((4, 16, 3, 3), dtype=np.float32)
    for i, pattern in enumerate(tile_patterns):
        for y in range(3):
            for x in range(3):
                tile_w[i, pattern[y][x], y, x] = 2.0
    tiles = node("relu", [
        conv("x0", tile_w, np.full(4, -17.0))
    ])
    observed = [ss(tiles, i, i + 1) for i in range(3)]
    refill = ss(tiles, 3, 4)

    positions = []
    presence = []
    for tile in observed:
        r = node("gmax", [node("mul", [tile, row])])
        c = node("gmax", [node("mul", [tile, col])])
        positions.extend([dense(r, [[63.0]]), dense(c, [[63.0]])])
        count = dense(node("gmean", [tile]), [[4096.0]], [-1.0])
        unique = dense(abs_v(count, 1), [[-2.0]], [1.0])
        presence.append(node("vrelu", [unique]))
    positions = node("vconcat", positions)
    presence = node("vconcat", presence)

    previous_positions = vs("mem", 0, 6)
    previous_presence = vs("mem", 6, 9)
    position_delta = node("vadd", [
        positions,
        dense(previous_positions, -np.eye(6, dtype=np.float32)),
    ])

    predict3 = np.zeros((4, 1, 7, 7), dtype=np.float32)
    predict2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        predict3[i, 0, 3 - 3 * dy, 3 - 3 * dx] = 1.0
        predict2[i, i, 2 - 2 * dy, 2 - 2 * dx] = 1.0
    predict3_name = param(predict3)
    predict2_name = param(predict2)

    predicted = []
    moving_flags = []
    for i, tile in enumerate(observed):
        dy = vs(position_delta, 2 * i, 2 * i + 1)
        dx = vs(position_delta, 2 * i + 1, 2 * i + 2)
        direction_input = node("vconcat", [dy, dy, dx, dx])
        direction_input = dense(
            direction_input,
            np.eye(4, dtype=np.float32),
            [5.0, -5.0, 5.0, -5.0],
        )
        direction = dense(
            abs_v(direction_input, 4),
            -0.4 * np.eye(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        direction = node("vrelu", [direction])
        valid = node("vmul", [
            vs(presence, i, i + 1),
            vs(previous_presence, i, i + 1),
        ])
        direction = node("vmul", [direction, valid])

        shifted = node("conv", [tile], w=predict3_name)
        shifted = node("conv", [shifted], w=predict2_name)
        shifted = node("mul", [shifted, direction])
        projected = max4([ss(shifted, j, j + 1) for j in range(4)])

        moving = dense(direction, np.ones((1, 4), dtype=np.float32))
        moving = clip_v(moving, 1)
        moving_flags.append(moving)
        original_scale = dense(moving, [[-0.15]], [1.0])
        original = node("mul", [tile, original_scale])
        predicted.append(node("max2", [original, projected]))

    goal_diff = node("abs", [
        node("sub", [goal_code_map, goal_code])
    ])
    selected_goal = node("relu", [
        node("sub", [one, goal_diff])
    ])
    selected_goal = node("mul", [selected_goal, all_goals])

    source_parts = [
        node("mul", [predicted[0], use_shape]),
        node("mul", [predicted[1], use_color]),
        node("mul", [predicted[2], use_rotation]),
        node("mul", [selected_goal, use_goal]),
    ]
    source = max4(source_parts)

    allowed_goal = node("mul", [selected_goal, use_goal])
    blocked_goal = node("sub", [all_goals, allowed_goal])
    open_map = node("mul", [
        base_open, node("sub", [one, blocked_goal])
    ])

    # Distance estimates for the objective and energy refills.
    initial = node("concat", [source, refill])
    shift3 = np.zeros((8, 2, 7, 7), dtype=np.float32)
    shift2 = np.zeros((8, 8, 5, 5), dtype=np.float32)
    for direction, (dy, dx) in enumerate(offsets):
        for channel in range(2):
            out = 2 * direction + channel
            shift3[out, channel, 3 + 3 * dy, 3 + 3 * dx] = 1.0
            shift2[out, out, 2 + 2 * dy, 2 + 2 * dx] = 0.97
    shift3_name = param(shift3)
    shift2_name = param(shift2)

    graph.append({
        "id": "distance_wave",
        "op": "loop",
        "T": 48,
        "state": [{"id": "distance_state", "init": initial}],
        "body": [
            {"id": "distance_shift3", "op": "conv",
             "in": ["distance_state"], "w": shift3_name},
            {"id": "distance_shift2", "op": "conv",
             "in": ["distance_shift3"], "w": shift2_name},
            {"id": "distance_up", "op": "slice",
             "in": ["distance_shift2"], "start": 0, "stop": 2},
            {"id": "distance_down", "op": "slice",
             "in": ["distance_shift2"], "start": 2, "stop": 4},
            {"id": "distance_left", "op": "slice",
             "in": ["distance_shift2"], "start": 4, "stop": 6},
            {"id": "distance_right", "op": "slice",
             "in": ["distance_shift2"], "start": 6, "stop": 8},
            {"id": "distance_ud", "op": "max2",
             "in": ["distance_up", "distance_down"]},
            {"id": "distance_lr", "op": "max2",
             "in": ["distance_left", "distance_right"]},
            {"id": "distance_neighbors", "op": "max2",
             "in": ["distance_ud", "distance_lr"]},
            {"id": "distance_keep", "op": "max2",
             "in": ["distance_state", "distance_neighbors"]},
            {"id": "distance_next", "op": "mul",
             "in": ["distance_keep", open_map]},
        ],
        "next": {"distance_state": "distance_next"},
    })

    scores = node("gmax", [
        node("mul", ["distance_wave.distance_state", player])
    ])
    objective_score = vs(scores, 0, 1)
    refill_score = vs(scores, 1, 2)

    bar_w = np.zeros((3, 3, 1, 1), dtype=np.float32)
    bar_w[0, 0, 0, 0] = 63.0
    bar_w[1, 1, 0, 0] = 63.0
    bar_w[2, 1, 0, 0] = -63.0
    bar_coords = conv("coords", bar_w, [-61.0, -12.0, 55.0])
    bar_y = node("abs", [ss(bar_coords, 0, 1)])
    bar_y = clip_s(conv(bar_y, np.asarray([[[[-2.0]]]]), [2.0]), one)
    bar_left = clip_s(ss(bar_coords, 1, 2), one)
    bar_right = clip_s(ss(bar_coords, 2, 3), one)
    bar_mask = node("mul", [node("mul", [bar_y, bar_left]), bar_right])
    bar_pixels = node("mul", [bar_mask, ss("x0", 11, 12)])
    energy = dense(node("gmean", [bar_pixels]), [[4096.0]])

    values = np.asarray(
        [0.97 ** max(0.0, i / 2.0 - 2.0) for i in range(43)],
        dtype=np.float32,
    )
    slopes = values[1:] - values[:-1]
    hinges = dense(
        energy,
        np.ones((42, 1), dtype=np.float32),
        -np.arange(42, dtype=np.float32),
    )
    hinges = node("vrelu", [hinges])
    threshold_w = np.zeros((1, 42), dtype=np.float32)
    threshold_w[0, 0] = slopes[0]
    threshold_w[0, 1:] = slopes[1:] - slopes[:-1]
    threshold = dense(hinges, threshold_w, [values[0]])
    need_refill = clip_v(
        dense(
            node("vconcat", [threshold, objective_score]),
            [[100.0, -100.0]],
        ),
        1,
    )
    refill_reachable = clip_v(dense(refill_score, [[100.0]]), 1)
    use_refill = node("vmul", [need_refill, refill_reachable])
    route_source = node("max2", [
        node("mul", [source, inv(use_refill)]),
        node("mul", [refill, use_refill]),
    ])

    # Penalize transformations that are not the present target.
    unwanted_parts = []
    for tile, projected, use in zip(
        observed, predicted, [use_shape, use_color, use_rotation]
    ):
        location = node("max2", [tile, projected])
        unwanted_parts.append(node("mul", [location, inv(use)]))
    unwanted = node("max2", [unwanted_parts[0], unwanted_parts[1]])
    unwanted = node("max2", [unwanted, unwanted_parts[2]])
    tile_factor = conv(
        unwanted, np.asarray([[[[-0.7]]]], dtype=np.float32), [1.0]
    )

    # Detect directional pusher stripes.
    stripe_w = np.zeros((8, 16, 7, 7), dtype=np.float32)
    stripe_specs = [
        ("h", 3), ("h", -2),
        ("h", -3), ("h", 2),
        ("v", 3), ("v", -2),
        ("v", -3), ("v", 2),
    ]
    for i, (axis, offset) in enumerate(stripe_specs):
        for t in range(-2, 3):
            if axis == "h":
                stripe_w[i, 1, 3 + offset, 3 + t] = 2.0
            else:
                stripe_w[i, 1, 3 + t, 3 + offset] = 2.0
    stripes = node("relu", [
        conv("x0", stripe_w, np.full(8, -9.0))
    ])
    pusher_parts = []
    for i in range(4):
        a = ss(stripes, 2 * i, 2 * i + 1)
        b = ss(stripes, 2 * i + 1, 2 * i + 2)
        pusher_parts.append(node("max2", [a, b]))
    pushers = node("mul", [node("concat", pusher_parts), open_map])
    pusher_parts = [ss(pushers, i, i + 1) for i in range(4)]
    any_pusher = max4(pusher_parts)
    ordinary = node("sub", [one, any_pusher])

    slide_open = node("mul", [
        base_open, node("sub", [one, all_goals])
    ])
    next_open = node("conv", [slide_open], w=neighbor3_name)
    next_open = node("conv", [next_open], w=neighbor2_name)
    next_closed = node("sub", [one, next_open])

    slide3 = np.zeros((4, 4, 7, 7), dtype=np.float32)
    slide2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        slide3[i, i, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        slide2[i, i, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    slide3_name = param(slide3)
    slide2_name = param(slide2)

    discount_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    discount_w[0, 2, 0, 0] = 0.97
    discount = conv("coords", discount_w)

    graph.append({
        "id": "route_wave",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": "route_state", "init": route_source},
            {"id": "slide_state", "zeros": [4, 64, 64]},
        ],
        "body": [
            {"id": "slide_shift3", "op": "conv",
             "in": ["slide_state"], "w": slide3_name},
            {"id": "slide_shift2", "op": "conv",
             "in": ["slide_shift3"], "w": slide2_name},
            {"id": "slide_continue", "op": "mul",
             "in": ["slide_shift2", next_open]},
            {"id": "slide_stop", "op": "mul",
             "in": ["route_state", next_closed]},
            {"id": "slide_next", "op": "add",
             "in": ["slide_continue", "slide_stop"]},
            {"id": "route_plain", "op": "mul",
             "in": ["route_state", ordinary]},
            {"id": "route_pushed", "op": "mul",
             "in": ["slide_state", pushers]},
            {"id": "route_p0", "op": "slice",
             "in": ["route_pushed"], "start": 0, "stop": 1},
            {"id": "route_p1", "op": "slice",
             "in": ["route_pushed"], "start": 1, "stop": 2},
            {"id": "route_p2", "op": "slice",
             "in": ["route_pushed"], "start": 2, "stop": 3},
            {"id": "route_p3", "op": "slice",
             "in": ["route_pushed"], "start": 3, "stop": 4},
            {"id": "route_pm0", "op": "max2",
             "in": ["route_p0", "route_p1"]},
            {"id": "route_pm1", "op": "max2",
             "in": ["route_p2", "route_p3"]},
            {"id": "route_pm2", "op": "max2",
             "in": ["route_pm0", "route_pm1"]},
            {"id": "route_entry0", "op": "max2",
             "in": ["route_plain", "route_pm2"]},
            {"id": "route_entry1", "op": "mul",
             "in": ["route_entry0", open_map]},
            {"id": "route_entry", "op": "mul",
             "in": ["route_entry1", tile_factor]},
            {"id": "route_shift3", "op": "conv",
             "in": ["route_entry"], "w": neighbor3_name},
            {"id": "route_shift2", "op": "conv",
             "in": ["route_shift3"], "w": neighbor2_name},
            {"id": "route_discount", "op": "mul",
             "in": ["route_shift2", discount]},
            {"id": "route_u", "op": "slice",
             "in": ["route_discount"], "start": 0, "stop": 1},
            {"id": "route_d", "op": "slice",
             "in": ["route_discount"], "start": 1, "stop": 2},
            {"id": "route_l", "op": "slice",
             "in": ["route_discount"], "start": 2, "stop": 3},
            {"id": "route_r", "op": "slice",
             "in": ["route_discount"], "start": 3, "stop": 4},
            {"id": "route_ud", "op": "max2",
             "in": ["route_u", "route_d"]},
            {"id": "route_lr", "op": "max2",
             "in": ["route_l", "route_r"]},
            {"id": "route_neighbors", "op": "max2",
             "in": ["route_ud", "route_lr"]},
            {"id": "route_keep", "op": "max2",
             "in": ["route_state", "route_neighbors"]},
            {"id": "route_next", "op": "mul",
             "in": ["route_keep", open_map]},
        ],
        "next": {
            "route_state": "route_next",
            "slide_state": "slide_next",
        },
    })

    route_plain = node("mul", ["route_wave.route_state", ordinary])
    route_pushed = node("mul", ["route_wave.slide_state", pushers])
    pushed = max4([ss(route_pushed, i, i + 1) for i in range(4)])
    route_entry = node("max2", [route_plain, pushed])
    route_entry = node("mul", [route_entry, open_map])
    route_entry = node("mul", [route_entry, tile_factor])

    directions = node("conv", [route_entry], w=neighbor3_name)
    directions = node("conv", [directions], w=neighbor2_name)
    action_scores = node("gmax", [node("mul", [directions, player])])

    fallback = node("conv", [open_map], w=neighbor3_name)
    fallback = node("conv", [fallback], w=neighbor2_name)
    fallback = node("gmax", [node("mul", [fallback, player])])
    fallback = dense(fallback, 0.0001 * np.eye(4, dtype=np.float32))
    action_scores = node("vadd", [action_scores, fallback])

    # Discourage immediate reversals while observed targets are moving.
    moving_sum = node("vconcat", moving_flags)
    moving_any = clip_v(
        dense(moving_sum, np.ones((1, 3), dtype=np.float32)), 1
    )
    reverse_w = np.zeros((4, 8), dtype=np.float32)
    reverse_w[0, 2] = -0.05
    reverse_w[1, 1] = -0.05
    reverse_w[2, 4] = -0.05
    reverse_w[3, 3] = -0.05
    reversal = dense("aprev", reverse_w)
    reversal = node("vmul", [reversal, moving_any])
    action_scores = node("vadd", [action_scores, reversal])

    # Remember failed directions while the player stays in the same cell.
    player_row = dense(
        node("gmax", [node("mul", [player, row])]), [[63.0]]
    )
    player_col = dense(
        node("gmax", [node("mul", [player, col])]), [[63.0]]
    )
    player_pos = node("vconcat", [player_row, player_col])
    previous_player = vs("mem", 9, 11)
    player_delta = node("vadd", [
        player_pos,
        dense(previous_player, -np.eye(2, dtype=np.float32)),
    ])
    distance = dense(abs_v(player_delta, 2), [[1.0, 1.0]])
    unchanged = node("vrelu", [
        dense(distance, [[-2.0]], [1.0])
    ])
    unchanged = node("vmul", [unchanged, vs("mem", 11, 12)])
    bans = node("vadd", [vs("mem", 12, 16), vs("aprev", 1, 5)])
    bans = clip_v(bans, 4)
    bans = node("vmul", [bans, unchanged])
    action_scores = node("vadd", [
        action_scores,
        dense(bans, -2.0 * np.eye(4, dtype=np.float32)),
    ])

    alive = node("gmax", [one])
    memory_value = node("vconcat", [
        positions, presence, player_pos, alive, bans
    ])
    graph.append({
        "id": "mem_next",
        "op": "dense",
        "in": [memory_value],
        "w": param(np.eye(16, dtype=np.float32)),
    })

    logits_w = np.zeros((8, 4), dtype=np.float32)
    for i in range(4):
        logits_w[i + 1, i] = 1.0
    logits_b = np.asarray(
        [-10.0, 0.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
        dtype=np.float32,
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [action_scores],
        "w": param(logits_w),
        "b": param(logits_b),
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": ["coords"],
        "w": param(np.zeros((1, 3, 1, 1), dtype=np.float32)),
    })

    return {"graph": graph, "memory_size": 16, "params": params}

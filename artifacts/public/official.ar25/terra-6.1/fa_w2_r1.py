import numpy as np


def build():
    graph = []
    params = {}
    counter = 0

    def name(prefix="n"):
        nonlocal counter
        counter += 1
        return "%s_%d" % (prefix, counter)

    def param(value):
        key = name("p")
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def emit(op, inputs, dest=None, target=None, **fields):
        if target is None:
            target = graph
        ident = dest if dest is not None else name()
        target.append({"id": ident, "op": op, "in": list(inputs), **fields})
        return ident

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return emit("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return emit("dense", [x], **fields)

    def sslice(x, start, stop):
        return emit("slice", [x], start=start, stop=stop)

    def vslice(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def affine(x, scale=1.0, bias=0.0):
        return dense(x, [[scale]], [bias])

    def vneg(x):
        return affine(x, -1.0)

    def vsub(a, b):
        return emit("vadd", [a, vneg(b)])

    def vone_minus(x):
        return affine(x, -1.0, 1.0)

    def vclip(x):
        a = emit("vrelu", [x])
        b = emit("vrelu", [affine(a, 1.0, -1.0)])
        return vsub(a, b)

    def vmul(a, b):
        return emit("vmul", [a, b])

    def gmax(x):
        return emit("gmax", [x])

    def count(x):
        return affine(emit("gmean", [x]), 4096.0)

    cy = sslice("coords", 0, 1)
    cx = sslice("coords", 1, 2)
    ones = sslice("coords", 2, 3)

    def coordinate_masks(coord):
        w = np.full((21, 1, 1, 1), 63.0, dtype=np.float32)
        b = -np.arange(1, 64, 3, dtype=np.float32)
        d = emit("abs", [conv(coord, w, b)])
        w2 = np.zeros((21, 21, 1, 1), dtype=np.float32)
        for i in range(21):
            w2[i, i, 0, 0] = -100.0
        z = emit("relu", [conv(d, w2, np.ones(21, dtype=np.float32))])
        w3 = np.zeros((21, 21, 1, 1), dtype=np.float32)
        for i in range(21):
            w3[i, i, 0, 0] = 2.0
        return emit("min2", [conv(z, w3), ones])

    xm = coordinate_masks(cx)
    ym = coordinate_masks(cy)
    sum21_param = param(np.ones((1, 21, 1, 1), dtype=np.float32))
    xgrid = emit("conv", [xm], w=sum21_param)
    ygrid = emit("conv", [ym], w=sum21_param)
    lattice = emit("mul", [xgrid, ygrid])
    onevec = gmax(lattice)
    zerovec = affine(onevec, 0.0)

    c0 = sslice("x0", 0, 1)
    c5 = sslice("x0", 5, 6)
    c9 = sslice("x0", 9, 10)
    c10 = sslice("x0", 10, 11)
    c11 = sslice("x0", 11, 12)

    corner = np.zeros((1, 1, 3, 3), dtype=np.float32)
    corner[0, 0, 0, 0] = 1.0
    actual = emit("mul", [conv(c5, corner), lattice])
    goals = emit("mul", [c11, lattice])
    not_actual = emit("sub", [ones, actual])

    def axis_detector(reduction):
        run = emit(reduction, [c10])
        z = conv(run, np.ones((1, 1, 1, 1), dtype=np.float32), [-18.0])
        z = emit("relu", [z])
        z = emit("min2", [z, ones])
        return emit("mul", [z, lattice])

    vertical = axis_detector("colsum")
    horizontal = axis_detector("rowsum")
    vp = gmax(vertical)
    hp = gmax(horizontal)

    def marker_count(axis, color):
        z = emit("mul", [axis, color])
        z = emit("mul", [z, not_actual])
        return count(z)

    def at_least_five(value):
        return vclip(affine(value, 2.0, -8.0))

    vzeros = marker_count(vertical, c0)
    hzeros = marker_count(horizontal, c0)
    vnines = marker_count(vertical, c9)
    hnines = marker_count(horizontal, c9)
    vselected = at_least_five(vzeros)
    hselected = at_least_five(hzeros)
    vmovable = at_least_five(emit("vadd", [vzeros, vnines]))
    hmovable = at_least_five(emit("vadd", [hzeros, hnines]))

    goal_count = count(goals)
    goal_x = count(emit("mul", [goals, cx]))
    goal_y = count(emit("mul", [goals, cy]))
    axis_x = gmax(emit("mul", [vertical, cx]))
    axis_y = gmax(emit("mul", [horizontal, cy]))
    xerr = vsub(goal_x, vmul(goal_count, axis_x))
    yerr = vsub(goal_y, vmul(goal_count, axis_y))
    half_cell = affine(goal_count, 1.5 / 63.0)

    def direction_flags(err, present, movable):
        enabled = vmul(present, movable)
        positive = vclip(affine(vsub(err, half_cell), 1000.0))
        negative = vclip(affine(vsub(vneg(err), half_cell), 1000.0))
        return vmul(positive, enabled), vmul(negative, enabled)

    xpos, xneg = direction_flags(xerr, vp, vmovable)
    ypos, yneg = direction_flags(yerr, hp, hmovable)
    xneed = vclip(emit("vadd", [xpos, xneg]))
    yneed = vclip(emit("vadd", [ypos, yneg]))
    xpick = xneed
    ypick = vmul(yneed, vone_minus(xneed))
    axis_need = vclip(emit("vadd", [xpick, ypick]))

    # Visually identify the selected piece after a click.
    axis_selected = vclip(emit("vadd", [vselected, hselected]))
    seed = emit("mul", [actual, c0])
    seed = emit("mul", [seed, vone_minus(axis_selected)])
    selected_colors = emit("max2", [c0, c11])
    allowed = emit("mul", [actual, selected_colors])
    allowed = emit("mul", [allowed, vone_minus(axis_selected)])

    flood_kernel = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for dy, dx in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        flood_kernel[0, 0, 3 + dy, 3 + dx] = 1.0
    flood_weight = param(flood_kernel)

    flood_id = name("flood")
    flood_state = name("state")
    flood_conv = name("body")
    flood_next = name("body")
    graph.append({
        "id": flood_id,
        "op": "loop",
        "T": 42,
        "state": [{"id": flood_state, "init": seed}],
        "body": [
            {
                "id": flood_conv,
                "op": "conv",
                "in": [flood_state],
                "w": flood_weight,
            },
            {
                "id": flood_next,
                "op": "min2",
                "in": [flood_conv, allowed],
            },
        ],
        "next": {flood_state: flood_next},
    })
    visual_selected = flood_id + "." + flood_state
    visual_present = gmax(visual_selected)

    inv_cx = emit("sub", [ones, cx])
    inv_cy = emit("sub", [ones, cy])
    visual_x = vone_minus(gmax(emit("mul", [visual_selected, inv_cx])))
    visual_y = vone_minus(gmax(emit("mul", [visual_selected, inv_cy])))

    def coordinate_match(coord, value):
        d = emit("abs", [emit("sub", [coord, value])])
        z = conv(
            d,
            -np.full((1, 1, 1, 1), 100.0, dtype=np.float32),
            [1.0],
        )
        z = emit("relu", [z])
        z = conv(z, np.full((1, 1, 1, 1), 2.0, dtype=np.float32))
        return emit("min2", [z, ones])

    visual_anchor = emit("mul", [
        coordinate_match(cx, visual_x),
        coordinate_match(cy, visual_y),
    ])
    visual_anchor = emit("mul", [visual_anchor, lattice])

    # Offset banks: positive-sampling shifts extract a template; negative
    # sampling shifts reconstruct it at an anchor.
    sx_extract = np.zeros((1, 1, 7, 7), dtype=np.float32)
    sy_extract = np.zeros((1, 1, 7, 7), dtype=np.float32)
    sx_draw = np.zeros((1, 1, 7, 7), dtype=np.float32)
    sy_draw = np.zeros((1, 1, 7, 7), dtype=np.float32)
    sx_extract[0, 0, 3, 6] = 1.0
    sy_extract[0, 0, 6, 3] = 1.0
    sx_draw[0, 0, 3, 0] = 1.0
    sy_draw[0, 0, 0, 3] = 1.0
    sx_extract_p = param(sx_extract)
    sy_extract_p = param(sy_extract)
    sx_draw_p = param(sx_draw)
    sy_draw_p = param(sy_draw)

    def offset_bank(image, sx, sy):
        columns = [image]
        for _ in range(6):
            columns.append(emit("conv", [columns[-1]], w=sx))
        table = [[None] * 7 for _ in range(7)]
        for x in range(7):
            table[0][x] = columns[x]
            for y in range(1, 7):
                table[y][x] = emit("conv", [table[y - 1][x]], w=sy)
        return emit("concat", [
            table[y][x] for y in range(7) for x in range(7)
        ])

    visual_offsets = offset_bank(
        visual_selected, sx_extract_p, sy_extract_p
    )
    visual_template = gmax(emit("mul", [visual_offsets, visual_anchor]))

    old_template = vslice("mem", 0, 49)
    old_x = vslice("mem", 49, 50)
    old_y = vslice("mem", 50, 51)
    old_present = vslice("mem", 51, 52)
    old_other_rows = vslice("mem", 52, 73)

    previous_move = dense(
        "aprev",
        np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=np.float32),
    )
    keep = vmul(old_present, previous_move)
    refresh = vone_minus(keep)

    dx = dense(
        "aprev",
        np.asarray([[0, 0, 0, -3.0 / 63.0, 3.0 / 63.0, 0, 0, 0]],
                   dtype=np.float32),
    )
    dy = dense(
        "aprev",
        np.asarray([[0, -3.0 / 63.0, 3.0 / 63.0, 0, 0, 0, 0, 0]],
                   dtype=np.float32),
    )
    predicted_x = emit("vadd", [old_x, dx])
    predicted_y = emit("vadd", [old_y, dy])
    min_x = emit("vadd", [
        vmul(predicted_x, keep), vmul(visual_x, refresh)
    ])
    min_y = emit("vadd", [
        vmul(predicted_y, keep), vmul(visual_y, refresh)
    ])
    template = emit("vadd", [
        vmul(old_template, keep), vmul(visual_template, refresh)
    ])
    selected_present = vclip(emit("vadd", [
        keep, vmul(visual_present, refresh)
    ]))

    anchor = emit("mul", [
        coordinate_match(cx, min_x),
        coordinate_match(cy, min_y),
    ])
    anchor = emit("mul", [anchor, lattice])
    anchor_offsets = offset_bank(anchor, sx_draw_p, sy_draw_p)
    selected_channels = emit("mul", [anchor_offsets, template])
    selected = conv(
        selected_channels, np.ones((1, 49, 1, 1), dtype=np.float32)
    )
    selected = emit("min2", [selected, ones])
    selected = emit("mul", [selected, selected_present])

    # Decode the other pieces from 21 exact 21-bit row integers.
    decode_id = name("decode")
    residual_state = name("state")
    power_state = name("state")
    position_state = name("state")
    image_state = name("state")
    initial_power = affine(onevec, 0.0, float(2 ** 20))
    initial_position = affine(onevec, 0.0, 61.0 / 63.0)
    body = []

    def db(op, *inputs, **fields):
        return emit(op, inputs, target=body, **fields)

    neg_power = db("dense", power_state, w=param([[-1.0]]))
    diff = db("vadd", residual_state, neg_power)
    plus_one = db("vadd", diff, onevec)
    positive = db("vrelu", plus_one)
    above_one = db("vrelu", diff)
    neg_above = db(
        "dense", above_one,
        w=param(-np.eye(21, dtype=np.float32))
    )
    bits = db("vadd", positive, neg_above)
    weighted = db("vmul", bits, power_state)
    neg_weighted = db(
        "dense", weighted,
        w=param(-np.eye(21, dtype=np.float32))
    )
    next_residual = db("vadd", residual_state, neg_weighted)
    next_power = db("dense", power_state, w=param([[0.5]]))
    next_position = db(
        "dense", position_state,
        w=param([[1.0]]), b=param([-3.0 / 63.0])
    )
    row_bits = db("mul", ym, bits)
    row_image = db("conv", row_bits, w=sum21_param)
    coordinate_difference = db("sub", cx, position_state)
    coordinate_difference = db("abs", coordinate_difference)
    column = db(
        "conv", coordinate_difference,
        w=param(np.full((1, 1, 1, 1), -100.0, dtype=np.float32)),
        b=param([1.0]),
    )
    column = db("relu", column)
    column = db(
        "conv", column,
        w=param(np.full((1, 1, 1, 1), 2.0, dtype=np.float32))
    )
    column = db("min2", column, ones)
    decoded_column = db("mul", row_image, column)
    next_image = db("max2", image_state, decoded_column)

    graph.append({
        "id": decode_id,
        "op": "loop",
        "T": 21,
        "state": [
            {"id": residual_state, "init": old_other_rows},
            {"id": power_state, "init": initial_power},
            {"id": position_state, "init": initial_position},
            {"id": image_state, "zeros": [1, 64, 64]},
        ],
        "body": body,
        "next": {
            residual_state: next_residual,
            power_state: next_power,
            position_state: next_position,
            image_state: next_image,
        },
    })
    remembered_other = decode_id + "." + image_state
    observed_other = emit("relu", [emit("sub", [actual, selected])])
    other = emit("add", [
        emit("mul", [remembered_other, keep]),
        emit("mul", [observed_other, refresh]),
    ])
    other_present = gmax(other)

    # Encode the current other-piece image for the next step.
    other_columns = emit("mul", [other, xm])
    powers = np.asarray(
        [2 ** i for i in range(21)], dtype=np.float32
    ).reshape(1, 21, 1, 1)
    weighted_columns = conv(other_columns, powers)
    row_sum = emit("rowsum", [weighted_columns])
    encoded_rows = gmax(emit("mul", [row_sum, ym]))
    emit(
        "vconcat",
        [template, min_x, min_y, selected_present, encoded_rows],
        dest="mem_next",
    )

    v_axis_vector = gmax(emit("mul", [vertical, xm]))
    h_axis_vector = gmax(emit("mul", [horizontal, ym]))

    reflection = np.zeros((441, 21, 1, 1), dtype=np.float32)
    for dest in range(21):
        for axis in range(21):
            source = 2 * axis - dest
            if 0 <= source < 21:
                reflection[dest * 21 + axis, source, 0, 0] = 1.0
    reflection_param = param(reflection)

    group = np.zeros((21, 441, 1, 1), dtype=np.float32)
    for dest in range(21):
        group[dest, dest * 21:(dest + 1) * 21, 0, 0] = 1.0
    group_param = param(group)

    repeat = np.zeros((441, 21), dtype=np.float32)
    for dest in range(21):
        for axis in range(21):
            repeat[dest * 21 + axis, axis] = 1.0
    repeat_param = param(repeat)
    minus_one = affine(onevec, 0.0, -1.0)

    def reflection_loop(initial, use_sum):
        loop_id = name("reflect")
        image_s = name("state")
        flag_s = name("state")
        body = []

        def b(op, *inputs, **fields):
            return emit(op, inputs, target=body, **fields)

        negflag = b("vmul", flag_s, minus_one)
        invflag = b("vadd", onevec, negflag)
        mx = b("mul", xm, flag_s)
        my = b("mul", ym, invflag)
        masks = b("add", mx, my)

        vx = b("vmul", v_axis_vector, flag_s)
        hy = b("vmul", h_axis_vector, invflag)
        axis_vector = b("vadd", vx, hy)

        source = b("mul", image_s, masks)
        rows = b("rowmax", source)
        cols = b("colmax", source)
        rows = b("mul", rows, flag_s)
        cols = b("mul", cols, invflag)
        source = b("add", rows, cols)

        samples = b("conv", source, w=reflection_param)
        repeated = b("dense", axis_vector, w=repeat_param)
        samples = b("mul", samples, repeated)
        destinations = b("conv", samples, w=group_param)
        destinations = b("mul", destinations, masks)
        reflected = b("conv", destinations, w=sum21_param)

        if use_sum:
            va = b("mul", vertical, flag_s)
            ha = b("mul", horizontal, invflag)
            axis_cells = b("add", va, ha)
            off_axis = b("sub", ones, axis_cells)
            reflected = b("mul", reflected, off_axis)
            result = b("add", image_s, reflected)
        else:
            result = b("max2", image_s, reflected)

        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 2,
            "state": [
                {"id": image_s, "init": initial},
                {"id": flag_s, "init": onevec},
            ],
            "body": body,
            "next": {
                image_s: result,
                flag_s: zerovec,
            },
        })
        return loop_id + "." + image_s

    other_union = reflection_loop(other, False)
    uncovered = emit("mul", [goals, emit("sub", [ones, other_union])])
    goal_field = reflection_loop(uncovered, True)

    max_x = gmax(emit("mul", [selected, cx]))
    max_y = gmax(emit("mul", [selected, cy]))
    width = vsub(max_x, min_x)
    height = vsub(max_y, min_y)

    field_offsets = offset_bank(goal_field, sx_extract_p, sy_extract_p)
    matched = emit("mul", [field_offsets, template])
    score = conv(matched, np.ones((1, 49, 1, 1), dtype=np.float32))

    limit_x = affine(width, -1.0, 61.0 / 63.0)
    limit_y = affine(height, -1.0, 61.0 / 63.0)

    def valid_coordinate(coord, limit):
        d = emit("sub", [limit, coord])
        z = conv(
            d,
            np.full((1, 1, 1, 1), 100.0, dtype=np.float32),
            [1.0],
        )
        return emit("min2", [emit("relu", [z]), ones])

    valid = emit("mul", [
        valid_coordinate(cx, limit_x),
        valid_coordinate(cy, limit_y),
    ])
    valid = emit("mul", [valid, lattice])
    invalid = emit("sub", [ones, valid])
    penalty = conv(
        invalid, np.full((1, 1, 1, 1), -1000.0, dtype=np.float32)
    )

    dx_abs = emit("abs", [emit("sub", [cx, min_x])])
    dy_abs = emit("abs", [emit("sub", [cy, min_y])])
    distance = emit("add", [dx_abs, dy_abs])
    distance_penalty = conv(
        distance, np.full((1, 1, 1, 1), -0.02, dtype=np.float32)
    )
    tie_input = emit("concat", [cx, cy])
    tie_break = conv(
        tie_input,
        np.asarray([[[[0.0001]], [[0.000031]]]], dtype=np.float32),
    )
    heat = emit("add", [score, penalty])
    heat = emit("add", [heat, distance_penalty])
    heat = emit("add", [heat, tie_break])
    best_score = gmax(heat)

    gap = emit("sub", [best_score, heat])
    best_mask = conv(
        gap,
        np.full((1, 1, 1, 1), -100000000.0, dtype=np.float32),
        [1.0],
    )
    best_mask = emit("relu", [best_mask])
    best_mask = emit("mul", [best_mask, valid])
    target_x = gmax(emit("mul", [best_mask, cx]))
    target_y = gmax(emit("mul", [best_mask, cy]))

    current_score = gmax(emit("mul", [heat, anchor]))
    improvement = vsub(best_score, current_score)
    at_best = vclip(affine(improvement, -100.0, 1.0))
    switch_piece = vmul(at_best, other_present)

    piece_mode = vone_minus(axis_need)
    piece_move = vmul(
        piece_mode,
        vmul(selected_present, vone_minus(switch_piece)),
    )
    need_piece_click = vmul(
        piece_mode,
        vclip(emit("vadd", [
            vone_minus(selected_present),
            vmul(selected_present, switch_piece),
        ])),
    )
    need_v_click = vmul(xpick, vone_minus(vselected))
    need_h_click = vmul(ypick, vone_minus(hselected))
    click_needed = vclip(emit("vadd", [
        need_piece_click,
        emit("vadd", [need_v_click, need_h_click]),
    ]))

    axis_up = vmul(ypick, vmul(hselected, yneg))
    axis_down = vmul(ypick, vmul(hselected, ypos))
    axis_left = vmul(xpick, vmul(vselected, xneg))
    axis_right = vmul(xpick, vmul(vselected, xpos))
    axis_moves = emit(
        "vconcat", [axis_up, axis_down, axis_left, axis_right]
    )
    axis_moves = dense(axis_moves, np.eye(4, dtype=np.float32) * 200.0)

    ddx = vsub(target_x, min_x)
    ddy = vsub(target_y, min_y)
    directions = emit("vconcat", [vneg(ddy), ddy, vneg(ddx), ddx])
    directions = dense(directions, np.eye(4, dtype=np.float32) * 100.0)

    inertia_w = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
        inertia_w[i, i + 1] = 0.01
    for i, reverse in enumerate([2, 1, 4, 3]):
        inertia_w[i, reverse] = -0.02
    directions = emit("vadd", [directions, dense("aprev", inertia_w)])
    directions = emit("vmul", [directions, piece_move])
    move_logits = emit("vadd", [axis_moves, directions])
    click_logit = affine(click_needed, 300.0)
    combined = emit("vconcat", [move_logits, click_logit])

    output_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[6, 4] = 1.0
    output_b = np.zeros(8, dtype=np.float32)
    output_b[[0, 5, 7]] = -1000.0
    emit(
        "dense",
        [combined],
        dest="logits",
        w=param(output_w),
        b=param(output_b),
    )

    clean = emit("mul", [not_actual, emit("sub", [ones, goals])])
    vmap = emit("mul", [vertical, clean])
    vmap = emit("mul", [vmap, emit("sub", [ones, horizontal])])
    vmap = emit("mul", [vmap, need_v_click])
    hmap = emit("mul", [horizontal, clean])
    hmap = emit("mul", [hmap, emit("sub", [ones, vertical])])
    hmap = emit("mul", [hmap, need_h_click])
    axis_map = conv(
        emit("add", [vmap, hmap]),
        np.full((1, 1, 1, 1), 10.0, dtype=np.float32),
    )

    candidate_other = emit("mul", [other, selected_present])
    candidate_any = emit("mul", [actual, vone_minus(selected_present)])
    piece_map = emit("max2", [candidate_other, candidate_any])
    piece_map = emit("mul", [piece_map, need_piece_click])
    emit("add", [axis_map, piece_map], dest="click")

    return {
        "graph": graph,
        "memory_size": 73,
        "params": params,
    }

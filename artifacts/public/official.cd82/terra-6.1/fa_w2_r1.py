import numpy as np


def build():
    graph = []
    params = {}
    counts = [0, 0]

    def fresh():
        counts[0] += 1
        return "n_%d" % counts[0]

    def param(value):
        counts[1] += 1
        name = "p_%d" % counts[1]
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(kind, inputs=None, dest=None, out=None, **fields):
        if dest is None:
            dest = graph
        name = out if out is not None else fresh()
        item = {"id": name, "op": kind}
        if inputs is not None:
            item["in"] = list(inputs)
        item.update(fields)
        dest.append(item)
        return name

    def op(kind, *inputs, dest=None):
        return node(kind, inputs, dest=dest)

    def conv(x, w, b=None, dest=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], dest=dest, **fields)

    def dense(x, w, b=None, dest=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], dest=dest, **fields)

    def sl(x, start, stop, dest=None):
        return node("slice", [x], dest=dest, start=start, stop=stop)

    def vsl(x, start, stop, dest=None):
        return node("vslice", [x], dest=dest, start=start, stop=stop)

    one_s = sl("coords", 2, 3)
    two_s = op("add", one_s, one_s)
    one_v = op("gmax", one_s)

    constants = {}
    for value in (
        1.0, -1.0, 2.0, 100.0, 256.0, 4096.0,
        10000.0, 1.0 / 256.0,
    ):
        constants[value] = dense(one_v, [[value]])

    def vs(x, value, dest=None):
        return op("vmul", x, constants[float(value)], dest=dest)

    def vsub(a, b, dest=None):
        return op("vadd", a, vs(b, -1.0, dest), dest=dest)

    def vone_minus(a, dest=None):
        return vsub(constants[1.0], a, dest)

    def vclip01(a, dest=None):
        lo = op("vrelu", a, dest=dest)
        hi = op("vrelu", vsub(a, constants[1.0], dest), dest=dest)
        return vsub(lo, hi, dest)

    def point_maps(points):
        n = len(points)
        wr = np.zeros((n, 3, 1, 1), np.float32)
        wc = np.zeros_like(wr)
        wr[:, 0, 0, 0] = 126.0
        wc[:, 1, 0, 0] = 126.0
        br = -2.0 * np.asarray([p[0] for p in points], np.float32)
        bc = -2.0 * np.asarray([p[1] for p in points], np.float32)
        rr = op("abs", conv("coords", wr, br))
        cc = op("abs", conv("coords", wc, bc))
        distance = op("add", rr, cc)
        distance = op("add", distance, distance)
        peak = op("relu", op("sub", two_s, distance))
        excess = op("relu", op("sub", peak, one_s))
        return op("sub", peak, excess)

    def v_equal_scalar(x, values):
        n = len(values)
        w = np.concatenate(
            [np.ones((n, 1), np.float32),
             -np.ones((n, 1), np.float32)],
            axis=0,
        )
        values = np.asarray(values, np.float32)
        b = np.concatenate([-values, values])
        diff = op("vrelu", dense(x, w, b))
        w2 = np.zeros((n, 2 * n), np.float32)
        for i in range(n):
            w2[i, i] = -1.0
            w2[i, n + i] = -1.0
        return op("vrelu", dense(diff, w2, np.ones(n, np.float32)))

    def vector_max(x, n, dest):
        # Pairwise max(a, b) = a + relu(b - a).
        while n > 1:
            m = (n + 1) // 2
            wa = np.zeros((m, n), np.float32)
            wd = np.zeros((m, n), np.float32)
            for i in range(m):
                a = 2 * i
                wa[i, a] = 1.0
                if a + 1 < n:
                    wd[i, a] = -1.0
                    wd[i, a + 1] = 1.0
            base = dense(x, wa, dest=dest)
            diff = dense(x, wd, dest=dest)
            x = op("vadd", base, op("vrelu", diff, dest=dest), dest=dest)
            n = m
        return x

    # The game ignores the target's two diagonals.
    cells = [
        (r, c)
        for r in range(10)
        for c in range(10)
        if r != c and r + c != 9
    ]
    target_points = point_maps([(r + 3, c + 3) for r, c in cells])

    # Eight broad brush shapes, then four narrow brush shapes.
    masks = np.zeros((12, len(cells)), np.float32)
    for j, (r, c) in enumerate(cells):
        masks[:, j] = np.asarray(
            [
                r < 5,
                c >= r,
                c >= 5,
                c >= 9 - r,
                r >= 5,
                c <= r,
                c < 5,
                c <= 9 - r,
                r < 3 and 3 <= c < 7,
                3 <= r < 7 and c >= 7,
                r >= 7 and 3 <= c < 7,
                3 <= r < 7 and c < 3,
            ],
            np.float32,
        )

    shape_maps = conv(target_points, masks[:, :, None, None])
    all_points = conv(
        target_points, np.ones((1, len(cells), 1, 1), np.float32)
    )

    # Translate the canvas to the target's location.
    colour_w = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    colour_image = conv("x0", colour_w)

    def shift_kernel(dr, dc):
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3 + dr, 3 + dc] = 1.0
        return w

    shift33 = param(shift_kernel(3, 3))
    shift30 = param(shift_kernel(3, 0))
    shift10 = param(shift_kernel(1, 0))
    aligned = colour_image
    for _ in range(8):
        aligned = node("conv", [aligned], w=shift33)
    for _ in range(2):
        aligned = node("conv", [aligned], w=shift30)
    aligned = node("conv", [aligned], w=shift10)

    difference = op("abs", op("sub", colour_image, aligned))
    difference = op(
        "sub",
        op("relu", difference),
        op("relu", op("sub", difference, one_s)),
    )

    # A distinct red pixel for each of the eight broad brush positions.
    position_points = [
        (24, 32), (31, 49), (39, 46), (46, 49),
        (53, 32), (46, 14), (39, 17), (31, 14),
    ]
    pos_maps = point_maps(position_points)
    red = sl("x0", 2, 3)
    position = op("gmax", op("mul", pos_maps, red))

    narrow_sensor_points = [(20, 29), (36, 50), (57, 29), (36, 13)]
    narrow_maps = point_maps(narrow_sensor_points)
    narrow_seen = op("gmax", op("mul", narrow_maps, red))

    cardinal_w = np.zeros((4, 8), np.float32)
    for i, p in enumerate((0, 2, 4, 6)):
        cardinal_w[i, p] = 1.0
    cardinal_pos = dense(position, cardinal_w)
    narrow_at_pos = op("vmul", narrow_seen, cardinal_pos)
    narrow_present = dense(
        narrow_at_pos, np.ones((1, 4), np.float32)
    )
    cardinal = dense(cardinal_pos, np.ones((1, 4), np.float32))

    # Candidate channels index (brush shape, colour).
    shape_expand_w = np.zeros((192, 12, 1, 1), np.float32)
    colour_expand_w = np.zeros((192, 16, 1, 1), np.float32)
    for s in range(12):
        for c in range(16):
            q = 16 * s + c
            shape_expand_w[q, s, 0, 0] = 1.0
            colour_expand_w[q, c, 0, 0] = 1.0
    candidate_shapes = conv(shape_maps, shape_expand_w)
    candidate_colours = conv("x0", colour_expand_w)
    candidate_pixels = op("mul", candidate_shapes, candidate_colours)

    sum_shapes_w = param(np.ones((1, 12, 1, 1), np.float32))

    expand12 = np.zeros((192, 12), np.float32)
    for s in range(12):
        expand12[16 * s:16 * s + 16, s] = 1.0
    expand12_p = param(expand12)

    group_w = np.zeros((12, 192), np.float32)
    for s in range(12):
        group_w[s, 16 * s:16 * s + 16] = 1.0
    group_p = param(group_w)

    encode_w = np.zeros((2, 192), np.float32)
    for s in range(12):
        for c in range(16):
            q = 16 * s + c
            encode_w[0, q] = s + 1
            encode_w[1, q] = c + 1
    encode_p = param(encode_w)

    # Sixteen two-component entries provide room for all twelve brushes.
    stack_size = 32
    push_w = np.zeros((stack_size, stack_size), np.float32)
    for i in range(2, stack_size):
        push_w[i, i - 2] = 1.0
    push_p = param(push_w)

    pop_w = np.zeros((stack_size, stack_size), np.float32)
    for i in range(stack_size - 2):
        pop_w[i, i + 2] = 1.0
    pop_p = param(pop_w)

    insert_w = np.zeros((stack_size, 2), np.float32)
    insert_w[0, 0] = 1.0
    insert_w[1, 1] = 1.0
    insert_p = param(insert_w)
    sum_winner_p = param(np.ones((1, 192), np.float32))

    priority = dense(
        one_v,
        (np.arange(192, dtype=np.float32) / 128.0).reshape(192, 1),
    )
    allowed_w = np.zeros((192, 1), np.float32)
    allowed_b = np.ones(192, np.float32)
    for s in range(8, 12):
        allowed_w[16 * s:16 * s + 16, 0] = 1.0
        allowed_b[16 * s:16 * s + 16] = 0.0
    allowed = dense(narrow_present, allowed_w, allowed_b)

    # Reverse planner: peel a uniform visible region each iteration.
    body = []
    uncovered = "uncovered_state"
    stack = "stack_state"

    total_sp = op("mul", shape_maps, uncovered, dest=body)
    total12 = vs(op("gmean", total_sp, dest=body), 4096.0, body)
    total192 = node("dense", [total12], dest=body, w=expand12_p)

    matching_sp = op("mul", candidate_pixels, uncovered, dest=body)
    matching192 = vs(
        op("gmean", matching_sp, dest=body), 4096.0, body
    )

    mismatch_sp = op("mul", total_sp, difference, dest=body)
    mismatch12 = vs(
        op("gmean", mismatch_sp, dest=body), 4096.0, body
    )
    mismatch192 = node(
        "dense", [mismatch12], dest=body, w=expand12_p
    )

    remaining_sp = op("mul", uncovered, difference, dest=body)
    remaining = vs(
        op("gmean", remaining_sp, dest=body), 4096.0, body
    )
    active = vclip01(remaining, body)

    bad = vsub(total192, matching192, body)
    uniform = op("vrelu", vone_minus(bad, body), dest=body)
    present = vclip01(total192, body)
    valid = op("vmul", uniform, present, dest=body)
    valid = op("vmul", valid, allowed, dest=body)

    score = op(
        "vadd",
        constants[10000.0],
        vs(mismatch192, 100.0, body),
        dest=body,
    )
    score = op("vadd", score, vs(total192, 2.0, body), dest=body)
    score = op("vadd", score, priority, dest=body)
    score = op("vmul", score, valid, dest=body)

    best = vector_max(score, 192, body)
    gap = vsub(score, best, body)
    gap = op("vadd", gap, constants[1.0 / 256.0], dest=body)
    winner = vs(op("vrelu", gap, dest=body), 256.0, body)
    winner = op("vmul", winner, valid, dest=body)
    winner = op("vmul", winner, active, dest=body)

    selected_shapes = node(
        "dense", [winner], dest=body, w=group_p
    )
    selected_sp = op(
        "mul", shape_maps, selected_shapes, dest=body
    )
    selected_mask = node(
        "conv", [selected_sp], dest=body, w=sum_shapes_w
    )
    removed = op("mul", uncovered, selected_mask, dest=body)
    new_uncovered = op("sub", uncovered, removed, dest=body)

    win_count = node(
        "dense", [winner], dest=body, w=sum_winner_p
    )
    encoded = node("dense", [winner], dest=body, w=encode_p)
    shifted_stack = node("dense", [stack], dest=body, w=push_p)
    inserted = node("dense", [encoded], dest=body, w=insert_p)
    proposed_stack = op("vadd", shifted_stack, inserted, dest=body)
    keep_old = op(
        "vmul", stack, vone_minus(win_count, body), dest=body
    )
    take_new = op("vmul", proposed_stack, win_count, dest=body)
    new_stack = op("vadd", keep_old, take_new, dest=body)

    graph.append({
        "id": "peel_loop",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": uncovered, "init": all_points},
            {"id": stack, "zeros": [stack_size]},
        ],
        "body": body,
        "next": {
            uncovered: new_uncovered,
            stack: new_stack,
        },
    })

    old_plan = vsl("mem", 0, stack_size)
    initialized = vsl("mem", stack_size, stack_size + 1)
    previous_paint = vsl("mem", stack_size + 1, stack_size + 2)

    new_plan_part = op(
        "vmul", "peel_loop.stack_state", vone_minus(initialized)
    )
    old_plan_part = op("vmul", old_plan, initialized)
    plan0 = op("vadd", new_plan_part, old_plan_part)

    shifted_plan = node("dense", [plan0], w=pop_p)
    plan = op(
        "vadd",
        op("vmul", plan0, vone_minus(previous_paint)),
        op("vmul", shifted_plan, previous_paint),
    )
    init_next = op(
        "vadd",
        initialized,
        op("vmul", vone_minus(initialized), cardinal),
    )
    preflight = vone_minus(init_next)

    shape_id = vsl(plan, 0, 1)
    colour_id = vsl(plan, 1, 2)
    shape_onehot = v_equal_scalar(shape_id, list(range(1, 13)))
    colour_onehot = v_equal_scalar(colour_id, list(range(1, 17)))

    # Identify the palette by its colour-4 borders.
    border_w = np.zeros((1, 1, 5, 5), np.float32)
    border_w[0, 0, 0, 2] = 1.0
    border_w[0, 0, 4, 2] = 1.0
    border_w[0, 0, 2, 0] = 1.0
    border_w[0, 0, 2, 4] = 1.0
    border_count = conv(sl("x0", 4, 5), border_w, [-3.0])
    border_ok = op("relu", border_count)

    row_w = np.zeros((1, 3, 1, 1), np.float32)
    row_w[0, 0, 0, 0] = 126.0
    row_dist = op("abs", conv("coords", row_w, [-8.0]))
    row_dist = op("add", row_dist, row_dist)
    row_peak = op("relu", op("sub", two_s, row_dist))
    row_mark = op(
        "sub", row_peak, op("relu", op("sub", row_peak, one_s))
    )
    palette_centres = op("mul", border_ok, row_mark)
    desired_colour_pixels = op("mul", "x0", colour_onehot)
    desired_colour_map = conv(
        desired_colour_pixels, np.ones((1, 16, 1, 1), np.float32)
    )
    palette_click = op("mul", palette_centres, desired_colour_map)

    underline_shift = np.zeros((1, 1, 7, 7), np.float32)
    underline_shift[0, 0, 6, 3] = 1.0
    underline = conv(sl("x0", 0, 1), underline_shift)
    selected = op("gmax", op("mul", palette_click, underline))

    goal_w = np.zeros((8, 12), np.float32)
    for s in range(8):
        goal_w[s, s] = 1.0
    for s, p in enumerate((0, 2, 4, 6), start=8):
        goal_w[p, s] = 1.0
    goal = dense(shape_onehot, goal_w)

    cur_expand = np.zeros((64, 8), np.float32)
    goal_expand = np.zeros((64, 8), np.float32)
    for a in range(8):
        for b in range(8):
            q = 8 * a + b
            cur_expand[q, a] = 1.0
            goal_expand[q, b] = 1.0
    pairs = op(
        "vmul", dense(position, cur_expand), dense(goal, goal_expand)
    )

    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    inverse_action = {1: 2, 2: 1, 3: 4, 4: 3}
    move_w = np.zeros((4, 64), np.float32)
    hit_w = np.zeros((1, 64), np.float32)
    for a in range(8):
        for b in range(8):
            q = 8 * a + b
            if a == b:
                hit_w[0, q] = 1.0
            elif (b - a) % 8 <= 4:
                move_w[clockwise_action[a] - 1, q] = 1.0
            else:
                edge = clockwise_action[(a - 1) % 8]
                move_w[inverse_action[edge] - 1, q] = 1.0

    moves = dense(pairs, move_w)
    at_goal = dense(pairs, hit_w)

    # A diagonal starting position first moves to a cardinal position,
    # where the optional narrow brush can be detected.
    pre_w = np.zeros((4, 8), np.float32)
    for p, action in ((1, 3), (3, 1), (5, 1), (7, 4)):
        pre_w[action - 1, p] = 1.0
    pre_moves = op("vmul", dense(position, pre_w), preflight)
    normal_moves = op(
        "vmul", moves, op("vmul", selected, init_next)
    )
    final_moves = op("vadd", normal_moves, pre_moves)

    stripe_w = np.zeros((1, 12), np.float32)
    stripe_w[0, 8:12] = 1.0
    is_narrow = dense(shape_onehot, stripe_w)
    ready = op("vmul", op("vmul", selected, at_goal), init_next)
    stamp = op("vmul", ready, vone_minus(is_narrow))
    narrow_click_needed = op("vmul", ready, is_narrow)
    palette_click_needed = op("vmul", vone_minus(selected), init_next)
    click_needed = op("vadd", palette_click_needed, narrow_click_needed)

    narrow_click_points = [(20, 32), (39, 50), (57, 32), (39, 13)]
    narrow_click_maps = point_maps(narrow_click_points)
    narrow_choice = vsl(shape_onehot, 8, 12)
    narrow_click_sp = op("mul", narrow_click_maps, narrow_choice)
    narrow_click_map = conv(
        narrow_click_sp, np.ones((1, 4, 1, 1), np.float32)
    )
    click_map = op(
        "add",
        op("mul", palette_click, palette_click_needed),
        op("mul", narrow_click_map, narrow_click_needed),
    )
    node("mul", [click_map, one_s], out="click")

    action_features = op("vconcat", final_moves, stamp, click_needed)
    action_w = np.zeros((8, 6), np.float32)
    for i in range(4):
        action_w[i + 1, i] = 2.0
    action_w[5, 4] = 2.0
    action_w[6, 5] = 2.0
    bias = np.asarray([-10, -1, -1, -1, -1, -1, -1, -10], np.float32)
    node(
        "dense",
        [action_features],
        out="logits",
        w=param(action_w),
        b=param(bias),
    )

    paint_this_step = op("vadd", stamp, narrow_click_needed)
    memory_out = op("vconcat", plan, init_next, paint_this_step)
    node(
        "dense",
        [memory_out],
        out="mem_next",
        w=param(np.eye(stack_size + 2, dtype=np.float32)),
    )

    return {
        "graph": graph,
        "memory_size": stack_size + 2,
        "params": params,
    }

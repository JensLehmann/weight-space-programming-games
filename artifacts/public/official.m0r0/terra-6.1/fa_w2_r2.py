import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]

    def name(prefix="n"):
        serial[0] += 1
        return "%s%d" % (prefix, serial[0])

    def par(value):
        n = name("p")
        params[n] = np.asarray(value, dtype=np.float32)
        return n

    def node(op, inputs, out=None, **kw):
        n = out or name()
        d = {"id": n, "op": op, "in": list(inputs)}
        d.update(kw)
        graph.append(d)
        return n

    def conv(x, w, b=None, out=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("conv", [x], out, **kw)

    def dense(x, w, b=None, out=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("dense", [x], out, **kw)

    def sl(x, a, b, out=None):
        return node("slice", [x], out, start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def op(kind, *xs):
        return node(kind, xs)

    def cat(xs, out=None):
        return node("concat", xs, out)

    def vc(xs, out=None):
        return node("vconcat", xs, out)

    def lin(xs, weights, bias=0.0, out=None):
        return dense(
            vc(xs),
            np.asarray([weights], dtype=np.float32),
            np.asarray([bias], dtype=np.float32),
            out,
        )

    def spatial_affine(x, weight=1.0, bias=0.0):
        return conv(
            x,
            np.asarray([[[[weight]]]], dtype=np.float32),
            np.asarray([bias], dtype=np.float32),
        )

    def clip_s(x):
        a = op("relu", x)
        b = op("relu", spatial_affine(x, 1.0, -1.0))
        return op("sub", a, b)

    def clip_v(x):
        a = op("vrelu", x)
        b = op("vrelu", lin([x], [1.0], -1.0))
        return lin([a, b], [1.0, -1.0])

    def neg_v(x):
        return lin([x], [-1.0])

    def one_minus(x):
        return lin([x], [-1.0], 1.0)

    def abs_v(x):
        return lin([op("vrelu", x), op("vrelu", neg_v(x))],
                   [1.0, 1.0])

    def mul_v(a, b):
        return op("vmul", a, b)

    def max_s(xs):
        z = xs[0]
        for x in xs[1:]:
            z = op("max2", z, x)
        return z

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def shift_kernel(channels, offsets):
        w = np.zeros((channels * len(offsets), channels, 3, 3),
                     dtype=np.float32)
        for c in range(channels):
            for j, (dy, dx) in enumerate(offsets):
                w[c * len(offsets) + j, c, 1 - dy, 1 - dx] = 1.0
        return w

    def corner(column):
        w = np.zeros((2, 3, 1, 1), dtype=np.float32)
        w[0, 0, 0, 0] = 63.0
        w[1, 1, 0, 0] = 63.0
        q = conv("coords", w, [-1.0, -float(column)])
        q = op("abs", q)
        q = conv(q, np.array([[[[-1.0]], [[-1.0]]]]), [1.0])
        q = op("relu", q)
        return op("gmax", op("mul", "x0", q))

    bg = op("vadd", corner(1), corner(62))
    one = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    invrow = op("sub", one, row)
    invcol = op("sub", one, col)

    def not_background(c):
        return one_minus(clip_v(lin([vs(bg, c, c + 1)], [2.0], -0.5)))

    bg_pixels = op("mul", "x0", bg)
    wall = conv(bg_pixels, np.full((1, 16, 1, 1), 2.0), [-0.5])
    wall = clip_s(wall)
    free0 = op("sub", one, wall)

    token1 = op("mul", sl("x0", 1, 2), not_background(1))
    token10 = op("mul", sl("x0", 10, 11), not_background(10))
    token = op("add", token1, token10)

    count = lin([op("gmean", token)], [4096.0])
    pieces = []
    for k in range(1, 8):
        d = float(2 * (2 * k + 1))
        pieces.append(clip_v(lin([count], [1.0 / d],
                                 -2.0 * k * k / d)))
    scale = lin(pieces, [1.0] * len(pieces), 1.0)
    step = lin([scale], [1.0 / 63.0])

    bounds = op("gmax", cat([
        op("mul", token, row),
        op("mul", token, col),
        op("mul", token, invrow),
        op("mul", token, invcol),
    ]))
    maxy, maxx = vs(bounds, 0, 1), vs(bounds, 1, 2)
    miny = one_minus(vs(bounds, 2, 3))
    minx = one_minus(vs(bounds, 3, 4))
    midy = lin([miny, maxy], [0.5, 0.5])
    midx = lin([minx, maxx], [0.5, 0.5])
    separation = lin([maxx, minx, scale], [63.0, -63.0, -1.0], 1.0)
    axis_x = clip_v(lin([separation], [2.0], -0.5))
    axis_y = one_minus(axis_x)
    midpoint = lin([mul_v(axis_x, midx), mul_v(axis_y, midy)],
                   [1.0, 1.0])
    along = op("add", op("mul", col, axis_x), op("mul", row, axis_y))
    split = op("sub", op("mul", one, midpoint), along)
    split = clip_s(spatial_affine(split, 63.0, 0.5))
    m1 = op("mul", token, split)
    m2 = op("mul", token, op("sub", one, split))
    markers = cat([m1, m2])

    def position(mask):
        z = op("gmax", cat([
            op("mul", mask, invrow),
            op("mul", mask, invcol),
        ]))
        return dense(z, -np.eye(2), np.ones(2))

    pos = vc([position(m1), position(m2)])
    px1 = vs(pos, 1, 2)
    px2 = vs(pos, 3, 4)

    core9 = op("mul", sl("x0", 9, 10), not_background(9))
    core11 = op("mul", sl("x0", 11, 12), not_background(11))
    core9 = op("mul", core9, free0)
    core11 = op("mul", core11, free0)
    selected = clip_v(lin([op("gmax", core11)], [2.0], -0.5))

    def expand_core(x):
        z = conv(x, np.ones((1, 1, 3, 3)))
        return clip_s(spatial_affine(z, 2.0))

    blocks9 = expand_core(core9)
    block11 = expand_core(core11)
    blocks = op("max2", blocks9, block11)

    colors = cat([
        op("mul", sl("x0", 14, 15), free0),
        op("mul", sl("x0", 12, 13), free0),
        op("mul", sl("x0", 15, 16), free0),
    ])

    def shift3(x, dy, dx):
        w = np.zeros((3, 3, 7, 7), dtype=np.float32)
        for c in range(3):
            w[c, c, 3 - dy, 3 - dx] = 1.0
        return conv(x, w)

    seeds = []
    for dy, dx in directions:
        q = shift3(colors, dy * 3, dx * 3)
        q = shift3(q, dy * 3, dx * 3)
        seeds.append(op("mul", q, colors))
    gate_seed = max_s(seeds)

    gate_shift_w = par(shift_kernel(3, directions))
    gate_body = [{
        "id": "gate_shift", "op": "conv", "in": ["gate_state"],
        "w": gate_shift_w
    }]
    gate_parts = []
    for c in range(3):
        items = []
        for d in range(4):
            n = "gate_part_%d_%d" % (c, d)
            gate_body.append({
                "id": n, "op": "slice", "in": ["gate_shift"],
                "start": 4 * c + d, "stop": 4 * c + d + 1
            })
            items.append(n)
        z = items[0]
        for d in range(1, 4):
            n = "gate_max_%d_%d" % (c, d)
            gate_body.append({"id": n, "op": "max2",
                              "in": [z, items[d]]})
            z = n
        gate_parts.append(z)
    gate_body.extend([
        {"id": "gate_join", "op": "concat", "in": gate_parts},
        {"id": "gate_union", "op": "max2",
         "in": ["gate_state", "gate_join"]},
        {"id": "gate_next", "op": "mul",
         "in": ["gate_union", colors]},
    ])
    graph.append({
        "id": "gate_loop", "op": "loop", "T": 12,
        "state": [{"id": "gate_state", "init": gate_seed}],
        "body": gate_body, "next": {"gate_state": "gate_next"}
    })
    gates = [sl("gate_loop.gate_state", c, c + 1) for c in range(3)]
    all_gates = max_s(gates)
    switches = [
        op("relu", op("sub", sl(colors, c, c + 1), gates[c]))
        for c in range(3)
    ]

    fixed_obstacle = op("max2", wall, all_gates)
    real_obstacle = op("max2", fixed_obstacle, blocks)
    real_free = op("sub", one, real_obstacle)
    optimistic_free = op("sub", one, fixed_obstacle)

    wave_shift = par(shift_kernel(1, directions))
    wave_decay_w = par(np.ones((1, 1, 1, 1)))
    wave_decay_b = par(np.array([-0.004], dtype=np.float32))
    binary_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    binary_kernel[0, 0, 1, 1] = 1.0
    binary_kernel[0, 0, 0, 1] = 1.0
    binary_kernel[0, 0, 2, 1] = 1.0
    binary_kernel[0, 0, 1, 0] = 1.0
    binary_kernel[0, 0, 1, 2] = 1.0
    binary_w = par(binary_kernel)

    def distance_wave(source, free, tag, iterations=180):
        state = tag + "_state"
        shifted = tag + "_shift"
        parts = [tag + "_p%d" % i for i in range(4)]
        body = [{
            "id": shifted, "op": "conv", "in": [state], "w": wave_shift
        }]
        for i in range(4):
            body.append({
                "id": parts[i], "op": "slice", "in": [shifted],
                "start": i, "stop": i + 1
            })
        a, b, c = tag + "_a", tag + "_b", tag + "_c"
        dec, rel = tag + "_dec", tag + "_rel"
        union, masked, nxt = tag + "_union", tag + "_masked", tag + "_next"
        body.extend([
            {"id": a, "op": "max2", "in": [parts[0], parts[1]]},
            {"id": b, "op": "max2", "in": [parts[2], parts[3]]},
            {"id": c, "op": "max2", "in": [a, b]},
            {"id": dec, "op": "conv", "in": [c],
             "w": wave_decay_w, "b": wave_decay_b},
            {"id": rel, "op": "relu", "in": [dec]},
            {"id": union, "op": "max2", "in": [state, rel]},
            {"id": masked, "op": "mul", "in": [union, free]},
            {"id": nxt, "op": "max2", "in": [masked, source]},
        ])
        loop_id = tag + "_loop"
        graph.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": [{"id": state, "init": source}],
            "body": body, "next": {state: nxt}
        })
        return loop_id + "." + state

    def binary_wave(source, free, tag, iterations=180):
        state = tag + "_state"
        a, b, c = tag + "_sum", tag + "_clip", tag + "_next"
        loop_id = tag + "_loop"
        graph.append({
            "id": loop_id, "op": "loop", "T": iterations,
            "state": [{"id": state, "init": source}],
            "body": [
                {"id": a, "op": "conv", "in": [state], "w": binary_w},
                {"id": b, "op": "min2", "in": [a, one]},
                {"id": c, "op": "mul", "in": [b, free]},
            ],
            "next": {state: c}
        })
        return loop_id + "." + state

    wave1 = distance_wave(m1, optimistic_free, "dist1")
    wave2 = distance_wave(m2, optimistic_free, "dist2")
    real_reach = binary_wave(m1, real_free, "real")
    connected = clip_v(lin(
        [op("gmax", op("mul", real_reach, m2))], [2.0], -0.5
    ))

    needs = []
    for c in range(3):
        exists = clip_v(lin([op("gmax", gates[c])], [2.0], -0.5))
        needs.append(mul_v(exists, one_minus(connected)))

    switch_target = max_s([
        op("mul", switches[c], needs[c]) for c in range(3)
    ])
    switch_wave = distance_wave(switch_target, real_free, "switchdist")

    shifted = conv(markers, shift_kernel(2, directions))
    repeat_w = np.zeros((8, 2, 1, 1), dtype=np.float32)
    for c in range(2):
        for d in range(4):
            repeat_w[4 * c + d, c, 0, 0] = 1.0
    marker_repeat = conv(markers, repeat_w)
    edges = op("relu", op("sub", shifted, marker_repeat))
    blocked = op("gmax", op("mul", edges, real_obstacle))
    can = dense(blocked, -np.eye(8), np.ones(8))
    cs = [vs(can, i, i + 1) for i in range(8)]

    other_wave = cat([wave2, wave1])
    other8 = conv(other_wave, repeat_w)
    edge_values = op("gmax", op("mul", edges, other8))
    base_values = op("gmax", op("mul", markers, other_wave))
    repeat_v = np.zeros((8, 2), dtype=np.float32)
    for c in range(2):
        repeat_v[4 * c:4 * c + 4, c] = 1.0
    base8 = dense(base_values, repeat_v)
    negative_base8 = dense(base8, -np.eye(8))
    progress = op(
        "vmul", can, op("vadd", edge_values, negative_base8)
    )
    progress = dense(progress, np.eye(8) * 150.0)
    ps = [vs(progress, i, i + 1) for i in range(8)]

    sw8 = conv(switch_wave, np.ones((8, 1, 1, 1)))
    sw_edges = op("gmax", op("mul", edges, sw8))
    sw_base2 = op("gmax", op("mul", markers, switch_wave))
    sw_base8 = dense(sw_base2, repeat_v)
    negative_sw_base8 = dense(sw_base8, -np.eye(8))
    sw_progress = op(
        "vmul", can, op("vadd", sw_edges, negative_sw_base8)
    )
    sw_progress = dense(sw_progress, np.eye(8) * 450.0)
    ss = [vs(sw_progress, i, i + 1) for i in range(8)]

    old_pos = vs("mem", 0, 4)
    old_orientation = vs("mem", 4, 5)
    displacement = lin(
        [px1, vs(old_pos, 1, 2), px2, vs(old_pos, 3, 4)],
        [63.0, -63.0, -63.0, 63.0]
    )
    previous_horizontal = dense(
        "aprev",
        np.array([[0, 0, 0, -1, 1, 0, 0, 0]], dtype=np.float32)
    )
    observed = mul_v(displacement, previous_horizontal)
    positive = clip_v(observed)
    negative = clip_v(neg_v(observed))
    sign = lin([positive, negative], [1.0, -1.0])
    sign_abs = lin([positive, negative], [1.0, 1.0])
    orientation = lin(
        [sign, mul_v(old_orientation, one_minus(sign_abs))], [1.0, 1.0]
    )
    assumed = lin([orientation, abs_v(orientation)], [1.0, -1.0], 1.0)
    plus = lin([assumed], [0.5], 0.5)
    minus = one_minus(plus)

    def pair_scores(values):
        up = lin([values[0], values[4]], [1.0, 1.0])
        down = lin([values[1], values[5]], [1.0, 1.0])
        left_a = lin([values[2], values[7]], [1.0, 1.0])
        left_b = lin([values[3], values[6]], [1.0, 1.0])
        left = lin([mul_v(plus, left_a), mul_v(minus, left_b)],
                   [1.0, 1.0])
        right = lin([mul_v(plus, left_b), mul_v(minus, left_a)],
                    [1.0, 1.0])
        return [up, down, left, right]

    route_scores = pair_scores(ps)
    switch_scores = pair_scores(ss)
    raw_scores = [
        lin([route_scores[i], switch_scores[i]], [1.0, 1.0])
        for i in range(4)
    ]

    zero = lin([step], [0.0])
    negstep = neg_v(step)
    signed = mul_v(step, assumed)
    negsigned = neg_v(signed)
    c1_right = lin([mul_v(plus, cs[3]), mul_v(minus, cs[2])], [1, 1])
    c2_right = lin([mul_v(plus, cs[6]), mul_v(minus, cs[7])], [1, 1])
    c1_left = lin([mul_v(plus, cs[2]), mul_v(minus, cs[3])], [1, 1])
    c2_left = lin([mul_v(plus, cs[7]), mul_v(minus, cs[6])], [1, 1])
    deltas = [
        vc([mul_v(negstep, cs[0]), zero, mul_v(negstep, cs[4]), zero]),
        vc([mul_v(step, cs[1]), zero, mul_v(step, cs[5]), zero]),
        vc([zero, mul_v(negsigned, c1_left),
            zero, mul_v(signed, c2_left)]),
        vc([zero, mul_v(signed, c1_right),
            zero, mul_v(negsigned, c2_right)]),
    ]

    history = vs("mem", 5, 133)
    neg_history = dense(history, -np.eye(128))
    repeat_pos = np.tile(np.eye(4, dtype=np.float32), (32, 1))
    group_sum = np.zeros((32, 128), dtype=np.float32)
    for i in range(32):
        group_sum[i, 4 * i:4 * i + 4] = 1.0
    history_weights = np.array(
        [[1.4 * 0.96 ** i for i in range(32)]], dtype=np.float32
    )

    normal_scores = []
    for raw, delta in zip(raw_scores, deltas):
        candidate = op("vadd", pos, delta)
        difference = op("vadd", dense(candidate, repeat_pos), neg_history)
        magnitude = op(
            "vadd", op("vrelu", difference),
            op("vrelu", dense(difference, -np.eye(128)))
        )
        distances = dense(magnitude, group_sum)
        matches = op("vrelu", dense(
            distances, -63.0 * np.eye(32), np.ones(32)
        ))
        revisit = dense(matches, history_weights)
        delta_abs = op(
            "vadd", op("vrelu", delta),
            op("vrelu", dense(delta, -np.eye(4)))
        )
        moving = clip_v(dense(delta_abs, np.full((1, 4), 63.0)))
        normal_scores.append(
            lin([raw, revisit, moving], [1.0, -1.0, 2.0], -2.0)
        )

    route_base = op("gmax", op("mul", m1, wave2))
    path_sum = op("add", wave1, wave2)
    path_margin = op("sub", path_sum, op("mul", one, route_base))
    path_margin = op("relu", spatial_affine(path_margin, 1.0, -0.94))

    # Exclude the most recently moved block from the next selection.
    last_y = vs("mem", 197, 198)
    last_x = vs("mem", 198, 199)
    dy = op("sub", row, op("mul", one, last_y))
    dx = op("sub", col, op("mul", one, last_x))
    upper_y = op("sub", op("mul", one, step), dy)
    upper_x = op("sub", op("mul", one, step), dx)
    inside_y = op("mul",
                  clip_s(spatial_affine(dy, 63.0, 0.5)),
                  clip_s(spatial_affine(upper_y, 63.0, -0.5)))
    inside_x = op("mul",
                  clip_s(spatial_affine(dx, 63.0, 0.5)),
                  clip_s(spatial_affine(upper_x, 63.0, -0.5)))
    last_block_mask = op("mul", inside_y, inside_x)

    block_click = op("mul", core9, path_margin)
    block_click = op("mul", block_click, op("sub", one, last_block_mask))
    block_candidate = clip_v(lin([op("gmax", block_click)], [100.0]))
    should_select = mul_v(
        mul_v(one_minus(connected), block_candidate),
        clip_v(lin([route_base], [10.0]))
    )
    should_select = mul_v(should_select, one_minus(selected))

    block_obstacle = max_s([
        wall, blocks9, all_gates, token,
        sl(colors, 0, 1), sl(colors, 1, 2), sl(colors, 2, 3)
    ])
    block_shifted = conv(block11, shift_kernel(1, directions))
    block_repeated = conv(block11, np.ones((4, 1, 1, 1)))
    block_edges = op("relu", op("sub", block_shifted, block_repeated))
    block_blocked = op("gmax", op("mul", block_edges, block_obstacle))
    block_can = dense(block_blocked, -np.eye(4), np.ones(4))
    bcs = [vs(block_can, i, i + 1) for i in range(4)]

    block_pos = position(block11)
    block_deltas = [
        vc([mul_v(negstep, bcs[0]), zero]),
        vc([mul_v(step, bcs[1]), zero]),
        vc([zero, mul_v(negstep, bcs[2])]),
        vc([zero, mul_v(step, bcs[3])]),
    ]
    path4 = conv(path_sum, np.ones((4, 1, 1, 1)))
    block_path = op("gmax", op("mul", block_edges, path4))
    block_history = vs("mem", 133, 197)
    neg_bhist = dense(block_history, -np.eye(64))
    repeat_block = np.tile(np.eye(2, dtype=np.float32), (32, 1))
    block_groups = np.zeros((32, 64), dtype=np.float32)
    for i in range(32):
        block_groups[i, 2 * i:2 * i + 2] = 1.0
    block_weights = np.array(
        [[2.0 * 0.96 ** i for i in range(32)]], dtype=np.float32
    )
    block_scores = []
    for i in range(4):
        candidate = op("vadd", block_pos, block_deltas[i])
        diff = op("vadd", dense(candidate, repeat_block), neg_bhist)
        magnitude = op(
            "vadd", op("vrelu", diff),
            op("vrelu", dense(diff, -np.eye(64)))
        )
        distances = dense(magnitude, block_groups)
        matches = op("vrelu", dense(
            distances, -63.0 * np.eye(32), np.ones(32)
        ))
        revisit = dense(matches, block_weights)
        block_scores.append(lin(
            [vs(block_path, i, i + 1), revisit, bcs[i]],
            [-1.0, -1.0, 30.0], -30.0
        ))

    age = vs("mem", 199, 200)
    timeout = clip_v(lin([age], [1.0], -4.0))
    release = clip_v(lin([connected, timeout], [1.0, 1.0]))
    deselect = mul_v(selected, release)

    click_map = op(
        "add",
        op("mul", block_click, one_minus(selected)),
        op("mul", m1, selected)
    )
    sl(click_map, 0, 1, out="click")

    combined = []
    for i in range(4):
        combined.append(lin(
            [mul_v(normal_scores[i], one_minus(selected)),
             mul_v(block_scores[i], selected)],
            [1.0, 1.0]
        ))
    click_flag = lin([should_select, deselect], [1.0, 1.0])
    output_in = vc(combined + [click_flag])
    output_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[6, 4] = 500.0
    output_b = np.array(
        [-1000, 0, 0, 0, 0, -1000, -200, -1000], dtype=np.float32
    )
    dense(output_in, output_w, output_b, out="logits")

    block_mem_pos = op("vmul", block_pos, selected)
    last_pos = vs("mem", 197, 199)
    next_last_pos = op(
        "vadd", block_mem_pos,
        op("vmul", last_pos, one_minus(selected))
    )
    next_age = mul_v(selected, lin([age], [1.0], 1.0))
    vc([
        pos,
        orientation,
        pos,
        vs(history, 0, 124),
        block_mem_pos,
        vs(block_history, 0, 62),
        next_last_pos,
        next_age,
    ], out="mem_next")

    return {"graph": graph, "memory_size": 200, "params": params}

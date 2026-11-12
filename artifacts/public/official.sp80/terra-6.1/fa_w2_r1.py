import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]
    shift_params = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def add(op, inputs, name=None, **fields):
        name = name or fresh()
        graph.append({"id": name, "op": op, "in": inputs, **fields})
        return name

    def par(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def dense(x, w, b=None, name=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return add("dense", [x], name=name, **fields)

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return add("conv", [x], **fields)

    def vslice(x, i):
        return add("vslice", [x], start=i, stop=i + 1)

    def spatial_slice(x, i):
        return add("slice", [x], start=i, stop=i + 1)

    def vscale(x, a, bias=0.0):
        return dense(x, [[a]], [bias])

    def vneg(x):
        return vscale(x, -1)

    def vsub(a, b):
        return add("vadd", [a, vneg(b)])

    def vabs(x):
        return add("vadd", [add("vrelu", [x]), add("vrelu", [vneg(x)])])

    def clip01(x):
        return vsub(
            add("vrelu", [x]),
            add("vrelu", [vscale(x, 1, -1)]),
        )

    def spatial_scale(x, a, bias=0.0):
        return conv(x, np.array([[[[a]]]], dtype=np.float32), [bias])

    def shift(x, dr, dc):
        while dr or dc:
            rr = max(-3, min(3, dr))
            cc = max(-3, min(3, dc))
            key = (rr, cc)
            if key not in shift_params:
                w = np.zeros((1, 1, 7, 7), dtype=np.float32)
                w[0, 0, 3 - rr, 3 - cc] = 1
                shift_params[key] = par(w)
            x = add("conv", [x], w=shift_params[key])
            dr -= rr
            dc -= cc
        return x

    selected = spatial_slice("x0", 9)
    other_bars = spatial_slice("x0", 8)
    cups = spatial_slice("x0", 11)
    water = spatial_slice("x0", 6)
    background = spatial_slice("x0", 12)
    row = spatial_slice("coords", 0)
    col = spatial_slice("coords", 1)

    stat_maps = [
        selected,
        add("mul", [selected, row]),
        add("mul", [selected, col]),
        water,
        add("mul", [water, row]),
        add("mul", [water, col]),
        cups,
        add("mul", [cups, row]),
        add("mul", [cups, col]),
    ]
    stat_mean = add("gmean", [add("concat", stat_maps)])
    factors = [
        4096, 4096 * 63, 4096 * 63,
        4096, 4096 * 63, 4096 * 63,
        4096, 4096 * 63, 4096 * 63,
    ]
    stats = dense(stat_mean, np.diag(factors))
    ns, sy, sx, nw, wy, wx, nc, cy, cx = [
        vslice(stats, i) for i in range(9)
    ]

    # Memory: initialization flag, previous selected x/y moments,
    # and number of previous spills.
    old_flag = vslice("mem", 0)
    old_sx = vslice("mem", 1)
    old_sy = vslice("mem", 2)
    old_trials = vslice("mem", 3)
    previous_spill = vslice("aprev", 5)
    trials = add("vadd", [old_trials, previous_spill])
    one = dense(old_flag, [[0]], [1])
    add("vconcat", [one, sx, sy, trials], name="mem_next")

    first = clip01(add("vadd", [
        add("vrelu", [vscale(old_flag, -1, 1)]),
        previous_spill,
    ]))
    later = add("vrelu", [vscale(first, -1, 1)])

    prev_moves = add("vslice", ["aprev"], start=1, stop=5)
    prev_move = dense(prev_moves, [[1, 1, 1, 1]])
    displacement = add("vadd", [
        vabs(vsub(sx, old_sx)),
        vabs(vsub(sy, old_sy)),
    ])
    unchanged = clip01(vscale(displacement, -0.1, 1))
    blocked = add("vmul", [prev_move, unchanged])
    not_blocked = add("vrelu", [vscale(blocked, -1, 1)])

    # Odd/even spill count, for alternating the fallback edge.
    trial_hinges = add(
        "vrelu",
        [dense(trials, [[1], [1], [1], [1], [1]], [0, -1, -2, -3, -4])],
    )
    flip = clip01(dense(trial_hinges, [[1, -2, 2, -2, 1]]))
    not_flip = add("vrelu", [vscale(flip, -1, 1)])

    axis_w = np.zeros((2, 1, 5, 5), dtype=np.float32)
    axis_w[0, 0, 2, [0, 2, 4]] = 1
    axis_w[1, 0, [0, 2, 4], 2] = 1
    axis_raw = conv(selected, axis_w, [-2.5, -2.5])
    axis_positive = add("relu", [axis_raw])
    axis_max = add("gmax", [axis_positive])
    axis_flags = dense(axis_max, [[2, 0], [0, 2]])
    hf = vslice(axis_flags, 0)
    vf = vslice(axis_flags, 1)
    horizontal = add("vrelu", [vsub(hf, vf)])
    vertical = add("vrelu", [vsub(vf, hf)])
    valid_axis = add("vadd", [horizontal, vertical])
    invalid_axis = add("vrelu", [vscale(valid_axis, -1, 1)])

    classes = [
        (s, length)
        for s in (1, 2, 3, 4)
        for length in (3, 4, 5, 6)
    ]
    thresholds = []
    for s, length in classes:
        n = length * s * s
        thresholds.extend([n - 0.5, n, n + 0.5])
    cw = np.ones((len(thresholds), 1), dtype=np.float32)
    cb = -np.asarray(thresholds, dtype=np.float32)
    class_hinges = add("vrelu", [dense(ns, cw, cb)])
    combine = np.zeros((len(classes), len(thresholds)), dtype=np.float32)
    for i in range(len(classes)):
        combine[i, 3 * i:3 * i + 3] = [2, -4, 2]
    class_flags = dense(class_hinges, combine)

    def make_axis(horizontal_axis, axis_flag):
        axis_coord = col if horizontal_axis else row
        travel_coord = row if horizontal_axis else col
        selected_axis_sum = sx if horizontal_axis else sy
        selected_travel_sum = sy if horizontal_axis else sx
        water_travel_sum = wy if horizontal_axis else wx
        cup_travel_sum = cy if horizontal_axis else cx

        source_difference = vsub(
            add("vmul", [water_travel_sum, nc]),
            add("vmul", [cup_travel_sum, nw]),
        )
        source_sign = add("vtanh", [source_difference])

        opening_lines = {}
        for s in (1, 2, 3, 4):
            if horizontal_axis:
                left = shift(cups, 0, s)
                right = shift(cups, 0, -s)
            else:
                left = shift(cups, s, 0)
                right = shift(cups, -s, 0)
            gap_input = add("concat", [left, right, background])
            gap = conv(
                gap_input,
                np.asarray([[[[1]], [[1]], [[1]]]], dtype=np.float32),
                [-2.5],
            )
            gap = spatial_scale(add("relu", [gap]), 2)
            opening_lines[s] = add(
                "colmax" if horizontal_axis else "rowmax", [gap]
            )

        goal_maps = []
        for s, length in classes:
            line = opening_lines[s]
            distance = s * (length + 1)
            if horizontal_axis:
                shifted = shift(line, 0, distance)
                pair = add("min2", [line, shifted])
                midpoint = shift(pair, 0, -(distance // 2))
            else:
                shifted = shift(line, distance, 0)
                pair = add("min2", [line, shifted])
                midpoint = shift(pair, -(distance // 2), 0)
            goal_maps.append(midpoint)

        goals = add("concat", goal_maps)
        goal_means = add("gmean", [goals])
        goal_moments = add("gmean", [add("mul", [goals, axis_coord])])

        count_scale = np.diag([64.0 / s for s, _ in classes])
        moment_scale = np.diag([64.0 * 63.0 / s for s, _ in classes])
        pair_counts = dense(goal_means, count_scale)
        pair_centres = dense(goal_moments, moment_scale)
        matched_count = add("vmul", [pair_counts, class_flags])
        matched_centres = add("vmul", [pair_centres, class_flags])
        ones = np.ones((1, len(classes)), dtype=np.float32)
        pair_count = dense(matched_count, ones)
        pair_target = dense(matched_centres, ones)

        edge_positions = []
        for s, length in classes:
            offset = (64 % s) // 2
            edge_positions.append(offset + (s * length - 1) / 2.0)
        left_edge = dense(class_flags, [edge_positions])
        right_edge = vsub(vscale(ns, 0, 63), left_edge)
        relative_side = vsub(selected_axis_sum, vscale(ns, 31.5))
        near_right = add("vsigmoid", [relative_side])
        near_left = add("vrelu", [vscale(near_right, -1, 1)])
        choose_right = add("vadd", [
            add("vmul", [near_right, not_flip]),
            add("vmul", [near_left, flip]),
        ])
        choose_left = add("vrelu", [vscale(choose_right, -1, 1)])
        edge_target = add("vadd", [
            add("vmul", [left_edge, choose_left]),
            add("vmul", [right_edge, choose_right]),
        ])
        no_pair = add("vrelu", [vscale(pair_count, -1, 1)])
        target = add("vadd", [
            pair_target,
            add("vmul", [edge_target, no_pair]),
        ])

        error = vsub(add("vmul", [target, ns]), selected_axis_sum)
        scale_values = [s * 0.5 + 0.1 for s, _ in classes]
        tolerance = add("vmul", [dense(class_flags, [scale_values]), ns])
        positive = add("vrelu", [vsub(error, tolerance)])
        negative = add("vrelu", [vsub(vneg(error), tolerance)])
        positive = add("vtanh", [vscale(positive, 0.1)])
        negative = add("vtanh", [vscale(negative, 0.1)])

        position = spatial_scale(travel_coord, 63)
        relative = add(
            "sub",
            [add("mul", [position, ns]), selected_travel_sum],
        )
        oriented = add("mul", [relative, source_sign])
        margin = vscale(ns, -2)
        toward_source = add("sigmoid", [add("add", [oriented, margin])])
        toward_cups = add(
            "sigmoid",
            [add("add", [spatial_scale(oriented, -1), margin])],
        )
        preference = add(
            "add",
            [add("mul", [travel_coord, source_sign]), vscale(ns, 0, 2)],
        )
        source_click = add(
            "mul",
            [other_bars, add("mul", [toward_source, preference])],
        )
        next_click = add(
            "mul",
            [other_bars, add("mul", [toward_cups, preference])],
        )
        chosen_map = add("add", [
            add("mul", [source_click, first]),
            add("mul", [next_click, later]),
        ])
        chosen_map = add("mul", [chosen_map, axis_flag])

        source_exists = clip01(add("gmax", [source_click]))
        force_click = add(
            "vmul",
            [add("vmul", [source_exists, first]), axis_flag],
        )
        allow_move = add("vrelu", [vscale(force_click, -1, 1)])
        move_gate = add(
            "vmul",
            [add("vmul", [axis_flag, allow_move]), not_blocked],
        )
        positive = add("vmul", [positive, move_gate])
        negative = add("vmul", [negative, move_gate])
        return negative, positive, chosen_map, force_click

    left, right, hclick, hforce = make_axis(True, horizontal)
    up, down, vclick, vforce = make_axis(False, vertical)

    mirror_click = add("mul", [other_bars, invalid_axis])
    click_map = add("add", [
        add("add", [hclick, vclick]),
        mirror_click,
    ])
    add("add", [click_map, spatial_scale(click_map, 0)], name="click")
    click_strength = add("gmax", [click_map])
    force = add("vadd", [hforce, vforce])

    decision = add(
        "vconcat",
        [up, down, left, right, click_strength, force],
    )
    out_w = np.zeros((8, 6), dtype=np.float32)
    out_w[1, 0] = 10
    out_w[2, 1] = 10
    out_w[3, 2] = 10
    out_w[4, 3] = 10
    out_w[6, 4] = 1
    out_w[6, 5] = 20
    dense(
        decision,
        out_w,
        [-1, 0, 0, 0, 0, 0.5, 0, -1],
        name="logits",
    )

    return {"graph": graph, "memory_size": 4, "params": params}

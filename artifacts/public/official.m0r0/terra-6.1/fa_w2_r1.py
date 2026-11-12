import numpy as np


def build():
    graph = []
    params = {}
    ni = [0]
    pi = [0]

    def node(op, *inputs, **fields):
        name = "n%d" % ni[0]
        ni[0] += 1
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        graph.append(d)
        return name

    def param(value):
        name = "p%d" % pi[0]
        pi[0] += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", x, **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", x, **fields)

    def sl(x, a, b):
        return node("slice", x, start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", x, start=a, stop=b)

    def add(a, b):
        return node("add", a, b)

    def sub(a, b):
        return node("sub", a, b)

    def mul(a, b):
        return node("mul", a, b)

    def va(a, b):
        return node("vadd", a, b)

    def vm(a, b):
        return node("vmul", a, b)

    def rel(x):
        return node("relu", x)

    def vr(x):
        return node("vrelu", x)

    def gm(x):
        return node("gmax", x)

    def vnot(x):
        return dense(x, [[-1.0]], [1.0])

    def clamp01(x, n):
        lo = vr(x)
        hi = vr(dense(x, np.eye(n, dtype=np.float32), -np.ones(n)))
        return va(lo, dense(hi, -np.eye(n, dtype=np.float32)))

    def vmax(a, b, n):
        difference = va(a, dense(b, -np.eye(n, dtype=np.float32)))
        return va(b, vr(difference))

    def coord_eq(axis, value):
        w = np.zeros((1, 3, 1, 1), np.float32)
        w[0, axis, 0, 0] = 63.0
        z = conv("coords", w, [-float(value)])
        z = node("abs", z)
        return rel(conv(z, [[[[-1.0]]]], [1.0]))

    one = sl("coords", 2, 3)

    # Identify the two wall colours from the letterbox.
    row1 = coord_eq(0, 1)
    sample_left = mul(row1, coord_eq(1, 1))
    sample_right = mul(row1, coord_eq(1, 62))
    wall_colours = va(
        gm(mul("x0", sample_left)),
        gm(mul("x0", sample_right))
    )
    walls = conv(
        mul("x0", wall_colours),
        np.ones((1, 16, 1, 1), np.float32)
    )
    walls = node("min2", walls, one)

    edge = add(coord_eq(0, 0), coord_eq(0, 63))
    edge = add(edge, coord_eq(1, 0))
    edge = add(edge, coord_eq(1, 63))
    edge = node("min2", edge, one)

    player_w = np.zeros((1, 16, 1, 1), np.float32)
    player_w[0, 1, 0, 0] = 1.0
    player_w[0, 10, 0, 0] = 1.0
    players = conv("x0", player_w)
    selected = gm(sl("x0", 1, 2))
    not_selected = vnot(selected)

    # Blocks have outlined interiors: 9 normally, 11 when selected.
    box9 = sl("x0", 9, 10)
    box11 = mul(sl("x0", 11, 12), selected)
    box_pixels = node("max2", box9, box11)
    box_area = conv(box_pixels, np.ones((1, 1, 3, 3), np.float32))
    box_area = node("min2", box_area, one)

    # Detect long gates separately for each of the three switch colours.
    colour_w = np.zeros((3, 16, 1, 1), np.float32)
    for c, colour in enumerate((12, 14, 15)):
        colour_w[c, colour, 0, 0] = 1.0
    coloured = conv("x0", colour_w)

    line_w = np.zeros((6, 3, 7, 7), np.float32)
    for c in range(3):
        line_w[2 * c, c, 3, :] = 2.0
        line_w[2 * c + 1, c, :, 3] = 2.0
    lines = rel(conv(coloured, line_w, np.full(6, -13.0)))

    expand_w = np.zeros((3, 6, 7, 7), np.float32)
    for c in range(3):
        expand_w[c, 2 * c, :, :] = 1.0
        expand_w[c, 2 * c + 1, :, :] = 1.0
    expanded = conv(lines, expand_w)
    expanded = node("min2", expanded, one)
    gate_colours = node("min2", expanded, coloured)
    gate_present = gm(gate_colours)
    gates = conv(gate_colours, np.ones((1, 3, 1, 1), np.float32))
    gates = node("min2", gates, one)

    switches_by_colour = rel(sub(sub(coloured, gate_colours), walls))
    switches_by_colour = node("min2", switches_by_colour, one)
    eligible_switches = mul(switches_by_colour, gate_present)
    switch_seed = conv(
        eligible_switches, np.ones((1, 3, 1, 1), np.float32)
    )
    switch_seed = node("min2", switch_seed, one)

    static_obstacles = add(add(walls, edge), gates)
    open_free = rel(sub(one, static_obstacles))
    free = rel(sub(open_free, box_area))
    switch_seed = mul(switch_seed, free)

    # Player cell corners, followed by leftmost/rightmost selection.
    corner_w = np.zeros((1, 1, 3, 3), np.float32)
    corner_w[0, 0, 1, 1] = 3.0
    corner_w[0, 0, 0, 1] = -1.0
    corner_w[0, 0, 1, 0] = -1.0
    corners = rel(conv(players, corner_w, [-2.0]))

    def extreme_corner(sign):
        w = np.zeros((1, 3, 1, 1), np.float32)
        w[0, 0, 0, 0] = sign * 63.0
        w[0, 1, 0, 0] = sign * 64.0 * 63.0
        priority = conv("coords", w, [5000.0])
        values = mul(priority, corners)
        best = gm(values)
        difference = node("abs", sub(values, best))
        return rel(conv(difference, [[[[-1.0]]]], [1.0]))

    left_seed = extreme_corner(-1.0)
    right_seed = extreme_corner(1.0)
    seeds = node("concat", left_seed, right_seed)

    five_test = rel(
        conv(players, np.ones((1, 1, 5, 5), np.float32) * 2.0, [-49.0])
    )
    is_five = gm(five_test)
    scale = dense(is_five, [[1.0]], [4.0])

    def flood(name, init, passable, channels):
        w = np.zeros((channels, channels, 3, 3), np.float32)
        for c in range(channels):
            for y, x in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)):
                w[c, c, y, x] = 1.0
        pw = param(w)
        wave = name + "_wave"
        accum = name + "_accum"
        cv = name + "_conv"
        clipped = name + "_clipped"
        next_wave = name + "_next_wave"
        next_accum = name + "_next_accum"
        graph.append({
            "id": name,
            "op": "loop",
            "T": 256,
            "state": [
                {"id": wave, "init": init},
                {"id": accum, "zeros": [channels, 64, 64]},
            ],
            "body": [
                {"id": cv, "op": "conv", "in": [wave], "w": pw},
                {"id": clipped, "op": "min2", "in": [cv, one]},
                {"id": next_wave, "op": "mul", "in": [clipped, passable]},
                {"id": next_accum, "op": "add", "in": [accum, next_wave]},
            ],
            "next": {wave: next_wave, accum: next_accum},
        })
        return name + "." + wave, name + "." + accum

    reach, fields = flood("real_flood", seeds, free, 2)
    _, optimistic_fields = flood("optimistic_flood", seeds, open_free, 2)
    _, switch_field = flood("switch_flood", switch_seed, free, 1)

    swap_w = np.zeros((2, 2, 1, 1), np.float32)
    swap_w[0, 1, 0, 0] = 1.0
    swap_w[1, 0, 0, 0] = 1.0
    other_fields = conv(fields, swap_w)
    current_values = gm(mul(seeds, other_fields))
    disconnected = vr(dense(current_values, [[-0.1, -0.1]], [1.0]))
    connected = vnot(disconnected)

    optimistic_other = conv(optimistic_fields, swap_w)
    optimistic_current = gm(mul(seeds, optimistic_other))
    optimistic_connected = clamp01(
        dense(optimistic_current, [[0.1, 0.1]]), 1
    )
    optimistic_sum = conv(
        optimistic_fields, np.ones((1, 2, 1, 1), np.float32)
    )
    optimum = gm(mul(optimistic_sum, left_seed))
    route = rel(add(
        sub(optimistic_sum, optimum),
        dense(optimum, [[0.0]], [3.0])
    ))
    route = node("min2", route, one)

    # A bridge block touches both currently reachable regions.
    near_w = np.zeros((2, 2, 7, 7), np.float32)
    near_w[0, 0, :, :] = 1.0
    near_w[1, 1, :, :] = 1.0
    near = conv(reach, near_w)
    near = node("min2", near, one)
    near_left = sl(near, 0, 1)
    near_right = sl(near, 1, 2)
    bridge_area = mul(near_left, near_right)
    bridge_boxes = mul(bridge_area, box9)
    has_bridge = gm(bridge_boxes)
    route_boxes = mul(route, box9)
    has_route_box = gm(route_boxes)

    route_fallback = vm(
        vnot(has_bridge),
        vm(optimistic_connected, has_route_box)
    )
    has_target = clamp01(va(has_bridge, route_fallback), 1)
    select_block = vm(vm(not_selected, disconnected), has_target)

    target_map = add(
        mul(bridge_boxes, has_bridge),
        mul(route_boxes, route_fallback)
    )

    # Memory layout:
    #   0:96    last 24 joint player positions
    #   96      horizontal orientation evidence
    #   97:113  last 8 selected-block positions
    #   113:119 extreme remembered switch ranks, two per colour
    history_size = 96
    box_history_start = 97
    switch_memory_start = 113
    memory_size = 119

    selected_bridge = gm(mul(bridge_area, box11))
    selected_value = gm(mul(box11, optimistic_sum))
    off_route = vr(dense(
        node("vconcat", optimum, selected_value),
        [[1.0, -1.0]], [-2.0]
    ))
    off_route = clamp01(off_route, 1)
    leave_if_off = vm(vnot(selected_bridge), off_route)
    leave_block = clamp01(va(connected, leave_if_off), 1)

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def shifted_destinations(source, channels):
        out_channels = channels * 4
        w3 = np.zeros((out_channels, channels, 7, 7), np.float32)
        w1 = np.zeros((out_channels, out_channels, 3, 3), np.float32)
        w2 = np.zeros((out_channels, out_channels, 5, 5), np.float32)
        for c in range(channels):
            for d, (dy, dx) in enumerate(directions):
                k = 4 * c + d
                w3[k, c, 3 - 3 * dy, 3 - 3 * dx] = 1.0
                w1[k, k, 1 - dy, 1 - dx] = 1.0
                w2[k, k, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        first = conv(source, w3)
        four = conv(first, w1)
        five = conv(first, w2)
        return add(mul(four, vnot(is_five)), mul(five, is_five))

    box_destinations = shifted_destinations(box11, 1)
    all_switches = conv(
        switches_by_colour, np.ones((1, 3, 1, 1), np.float32)
    )
    box_free = rel(sub(sub(free, players), all_switches))
    box_valid = gm(mul(box_destinations, box_free))
    any_box_move = clamp01(
        dense(box_valid, np.ones((1, 4), np.float32)), 1
    )
    leave_block = clamp01(va(leave_block, vnot(any_box_move)), 1)
    deselect_block = vm(selected, leave_block)
    click_flag = va(select_block, deselect_block)

    priority_w = np.zeros((1, 3, 1, 1), np.float32)
    priority_w[0, 0, 0, 0] = -1.0
    priority_w[0, 1, 0, 0] = -1.0
    priority = conv("coords", priority_w, [10.0])
    target_click = mul(target_map, priority)
    player_click = mul(sl("x0", 1, 2), dense(selected, [[100.0]]))
    click_map = add(mul(target_click, not_selected), player_click)
    graph.append({"id": "click", "op": "relu", "in": [click_map]})

    destinations = shifted_destinations(seeds, 2)
    valid = gm(mul(destinations, free))
    invalid = dense(valid, -np.eye(8, dtype=np.float32), np.ones(8))

    field_repeat = np.zeros((8, 2, 1, 1), np.float32)
    vector_repeat = np.zeros((8, 2), np.float32)
    for i in range(8):
        field_repeat[i, i // 4, 0, 0] = 1.0
        vector_repeat[i, i // 4] = 1.0

    moved_pair = gm(mul(destinations, conv(other_fields, field_repeat)))
    current_pair8 = dense(current_values, vector_repeat)
    pair_values = va(moved_pair, vm(current_pair8, invalid))

    switch_field8 = conv(
        switch_field, np.ones((8, 1, 1, 1), np.float32)
    )
    moved_switch = gm(mul(destinations, switch_field8))
    current_switch = gm(mul(seeds, switch_field))
    current_switch8 = dense(current_switch, vector_repeat)
    switch_values = va(moved_switch, vm(current_switch8, invalid))
    switch_mode = vm(disconnected, gm(switch_seed))
    candidate_values = va(
        vm(pair_values, vnot(switch_mode)),
        vm(switch_values, switch_mode)
    )

    row_coord = sl("coords", 0, 1)
    col_coord = sl("coords", 1, 2)
    px = gm(mul(seeds, col_coord))
    py = gm(mul(seeds, row_coord))
    raw_pos = node("vconcat", px, py)
    pos_w = np.zeros((4, 4), np.float32)
    pos_w[0, 0] = 63.0
    pos_w[1, 2] = 63.0
    pos_w[2, 1] = 63.0
    pos_w[3, 3] = 63.0
    pos = dense(raw_pos, pos_w)

    # Infer whether ACTION3 moves the leftmost player left or right.
    orient_input = node("vconcat", pos, "mem", "aprev")
    delta_w = np.zeros((1, 4 + memory_size + 8), np.float32)
    delta_w[0, 0] = 0.25
    delta_w[0, 2] = -0.25
    delta_w[0, 4] = -0.25
    delta_w[0, 6] = 0.25
    delta = dense(orient_input, delta_w)
    previous_sign_w = np.zeros((1, 8), np.float32)
    previous_sign_w[0, 4] = 1.0
    previous_sign_w[0, 3] = -1.0
    change = vm(delta, dense("aprev", previous_sign_w))
    evidence_raw = va(vsl("mem", 96, 97), change)
    ep = vr(dense(evidence_raw, [[1.0]], [1.0]))
    em = vr(dense(evidence_raw, [[1.0]], [-1.0]))
    evidence = dense(node("vconcat", ep, em), [[1.0, -1.0]], [-1.0])
    orient_plus = dense(
        node("vconcat",
             vr(dense(evidence, [[1.0]], [1.0])),
             vr(evidence)),
        [[1.0, -1.0]]
    )
    orient_minus = vnot(orient_plus)

    plus_w = np.zeros((4, 8), np.float32)
    minus_w = np.zeros((4, 8), np.float32)
    for w in (plus_w, minus_w):
        w[0, 0], w[0, 4] = 1.03, 0.97
        w[1, 1], w[1, 5] = 1.03, 0.97
    plus_w[2, 2], plus_w[2, 7] = 1.03, 0.97
    plus_w[3, 3], plus_w[3, 6] = 1.03, 0.97
    minus_w[2, 3], minus_w[2, 6] = 1.03, 0.97
    minus_w[3, 2], minus_w[3, 7] = 1.03, 0.97

    scores = va(
        vm(dense(candidate_values, plus_w), orient_plus),
        vm(dense(candidate_values, minus_w), orient_minus)
    )
    mobility = va(
        vm(dense(valid, plus_w), orient_plus),
        vm(dense(valid, minus_w), orient_minus)
    )
    mobility = clamp01(mobility, 4)
    scores = va(scores, dense(
        mobility, 25.0 * np.eye(4, dtype=np.float32),
        -25.0 * np.ones(4)
    ))

    # Remember the first and last observed switch position of each colour.
    switch_corner_w = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        switch_corner_w[c, c, 1, 1] = 3.0
        switch_corner_w[c, c, 0, 1] = -1.0
        switch_corner_w[c, c, 1, 0] = -1.0
    switch_corners = rel(conv(
        switches_by_colour, switch_corner_w, [-2.0, -2.0, -2.0]
    ))
    rank_w = np.zeros((1, 3, 1, 1), np.float32)
    rank_w[0, 0, 0, 0] = 63.0 * 64.0
    rank_w[0, 1, 0, 0] = 63.0
    rank = conv("coords", rank_w, [1.0])
    inverse_rank = conv(rank, [[[[-1.0]]]], [5000.0])
    visible_max_rank = gm(mul(switch_corners, rank))
    visible_min_rank = gm(mul(switch_corners, inverse_rank))
    observed_ranks = node("vconcat", visible_max_rank, visible_min_rank)
    old_ranks = vsl("mem", switch_memory_start, switch_memory_start + 6)
    remembered_ranks = vmax(observed_ranks, old_ranks, 6)

    player_ranks = gm(mul(seeds, rank))
    player_inverse_ranks = gm(mul(seeds, inverse_rank))
    rank_inputs = node("vconcat", player_ranks, player_inverse_ranks)
    rank_repeat = np.zeros((12, 4), np.float32)
    remembered_repeat = np.zeros((12, 6), np.float32)
    for p in range(2):
        for c in range(3):
            for extreme in range(2):
                j = p * 6 + c * 2 + extreme
                rank_repeat[j, p + 2 * extreme] = 1.0
                remembered_repeat[j, c + 3 * extreme] = -1.0
    rank_difference = va(
        dense(rank_inputs, rank_repeat),
        dense(remembered_ranks, remembered_repeat)
    )
    negative_one = dense(is_five, [[0.0]], [-1.0])
    rank_abs = va(vr(rank_difference), vr(vm(rank_difference, negative_one)))
    rank_match = vr(dense(
        rank_abs, -np.eye(12, dtype=np.float32), np.ones(12)
    ))

    # A held switch is one whose colour currently has no visible gate.
    gate_repeat = np.zeros((12, 3), np.float32)
    for p in range(2):
        for c in range(3):
            for extreme in range(2):
                gate_repeat[p * 6 + c * 2 + extreme, c] = -1.0
    gate_absent = dense(gate_present, gate_repeat, np.ones(12))
    held_matches = vm(rank_match, gate_absent)
    held_sum_w = np.zeros((2, 12), np.float32)
    held_sum_w[0, 0:6] = 1.0
    held_sum_w[1, 6:12] = 1.0
    held = clamp01(dense(held_matches, held_sum_w), 2)
    held8 = dense(held, vector_repeat)
    moving_held = vm(valid, held8)
    held_cost = va(
        vm(dense(moving_held, plus_w), orient_plus),
        vm(dense(moving_held, minus_w), orient_minus)
    )
    scores = va(scores, dense(
        held_cost, -15.0 * np.eye(4, dtype=np.float32)
    ))

    # Predict joint positions and penalize recent repetitions.
    move_plus = np.zeros((16, 8), np.float32)
    move_minus = np.zeros((16, 8), np.float32)
    for w in (move_plus, move_minus):
        w[1, 0] = -1.0
        w[3, 4] = -1.0
        w[5, 1] = 1.0
        w[7, 5] = 1.0
    move_plus[8, 2] = -1.0
    move_plus[10, 7] = 1.0
    move_plus[12, 3] = 1.0
    move_plus[14, 6] = -1.0
    move_minus[8, 3] = 1.0
    move_minus[10, 6] = -1.0
    move_minus[12, 2] = -1.0
    move_minus[14, 7] = 1.0

    displacement = va(
        vm(dense(valid, move_plus), orient_plus),
        vm(dense(valid, move_minus), orient_minus)
    )
    displacement = vm(displacement, scale)
    repeat_pos = np.zeros((16, 4), np.float32)
    for a in range(4):
        for k in range(4):
            repeat_pos[4 * a + k, k] = 1.0
    predictions = va(dense(pos, repeat_pos), displacement)

    history = vsl("mem", 0, history_size)
    pred_expand = np.zeros((384, 16), np.float32)
    hist_expand = np.zeros((384, history_size), np.float32)
    for a in range(4):
        for h in range(24):
            for k in range(4):
                j = (a * 24 + h) * 4 + k
                pred_expand[j, a * 4 + k] = 1.0
                hist_expand[j, h * 4 + k] = -1.0
    difference = va(
        dense(predictions, pred_expand),
        dense(history, hist_expand)
    )
    absolute_difference = va(
        vr(difference), vr(vm(difference, negative_one))
    )
    sum4 = np.zeros((96, 384), np.float32)
    for i in range(96):
        sum4[i, 4 * i:4 * i + 4] = 1.0
    distance = dense(absolute_difference, sum4)
    matches = vr(dense(
        distance, -2.0 * np.eye(96, dtype=np.float32), np.ones(96)
    ))
    tabu_w = np.zeros((4, 96), np.float32)
    for a in range(4):
        for h in range(24):
            tabu_w[a, a * 24 + h] = -(12.0 - 0.25 * h)
    scores = va(scores, dense(matches, tabu_w))

    reverse_w = np.zeros((4, 8), np.float32)
    reverse_w[0, 2] = -1.0
    reverse_w[1, 1] = -1.0
    reverse_w[2, 4] = -1.0
    reverse_w[3, 3] = -1.0
    scores = va(scores, dense("aprev", reverse_w))

    # Selected-block movement: favor valid off-route destinations and
    # penalize recent block positions.
    box_dest_value = gm(mul(box_destinations, optimistic_sum))
    box_off = vr(va(
        dense(optimum, np.ones((4, 1), np.float32)),
        dense(box_dest_value, -np.eye(4, dtype=np.float32))
    ))
    box_off = vm(box_off, box_valid)
    box_scores = va(
        dense(box_valid, 100.0 * np.eye(4, dtype=np.float32),
              [4.0, 3.0, 1.0, 2.0]),
        dense(box_off, 2.0 * np.eye(4, dtype=np.float32))
    )

    box_x = gm(mul(box11, col_coord))
    box_y = gm(mul(box11, row_coord))
    box_pos = dense(node("vconcat", box_x, box_y),
                    63.0 * np.eye(2, dtype=np.float32))
    box_dest_x = gm(mul(box_destinations, col_coord))
    box_dest_y = gm(mul(box_destinations, row_coord))
    box_raw = node("vconcat", box_dest_x, box_dest_y)
    box_prediction_w = np.zeros((8, 8), np.float32)
    for a in range(4):
        box_prediction_w[2 * a, a] = 63.0
        box_prediction_w[2 * a + 1, 4 + a] = 63.0
    box_predictions = dense(box_raw, box_prediction_w)
    box_history = vsl("mem", box_history_start, box_history_start + 16)
    bp_w = np.zeros((64, 8), np.float32)
    bh_w = np.zeros((64, 16), np.float32)
    for a in range(4):
        for h in range(8):
            for k in range(2):
                j = (a * 8 + h) * 2 + k
                bp_w[j, 2 * a + k] = 1.0
                bh_w[j, 2 * h + k] = -1.0
    box_difference = va(
        dense(box_predictions, bp_w),
        dense(box_history, bh_w)
    )
    box_abs = va(
        vr(box_difference), vr(vm(box_difference, negative_one))
    )
    sum2 = np.zeros((32, 64), np.float32)
    for i in range(32):
        sum2[i, 2 * i:2 * i + 2] = 1.0
    box_distance = dense(box_abs, sum2)
    box_matches = vr(dense(
        box_distance, -2.0 * np.eye(32, dtype=np.float32), np.ones(32)
    ))
    box_tabu_w = np.zeros((4, 32), np.float32)
    for a in range(4):
        for h in range(8):
            box_tabu_w[a, a * 8 + h] = -(30.0 - 2.0 * h)
    box_scores = va(box_scores, dense(box_matches, box_tabu_w))

    box_reverse_w = np.zeros((4, 8), np.float32)
    box_reverse_w[0, 2] = -5.0
    box_reverse_w[1, 1] = -5.0
    box_reverse_w[2, 4] = -5.0
    box_reverse_w[3, 3] = -5.0
    box_scores = va(box_scores, dense("aprev", box_reverse_w))
    box_mode = vm(selected, vnot(leave_block))

    action_w = np.zeros((8, 4), np.float32)
    for a in range(4):
        action_w[a + 1, a] = 1.0
    normal_logits = dense(
        vm(scores, not_selected),
        action_w,
        [-1000.0, 0.0, 0.0, 0.0, 0.0, -1000.0, -100.0, -1000.0]
    )
    box_logits = vm(dense(box_scores, action_w), box_mode)
    click_w = np.zeros((8, 1), np.float32)
    click_w[6, 0] = 2000.0
    click_logits = dense(click_flag, click_w)
    logits = va(va(normal_logits, box_logits), click_logits)
    graph.append({
        "id": "logits",
        "op": "vadd",
        "in": [logits, dense(click_flag, np.zeros((8, 1), np.float32))]
    })

    shifted_history = node("vconcat", pos, vsl("mem", 0, 92))
    next_history = va(
        vm(history, selected),
        vm(shifted_history, not_selected)
    )
    shifted_box_history = node(
        "vconcat", box_pos, vsl("mem", box_history_start, box_history_start + 14)
    )
    next_box_history = va(
        vm(box_history, not_selected),
        vm(shifted_box_history, selected)
    )
    next_memory = node(
        "vconcat", next_history, evidence, next_box_history, remembered_ranks
    )
    graph.append({
        "id": "mem_next",
        "op": "vadd",
        "in": [
            next_memory,
            dense(click_flag, np.zeros((memory_size, 1), np.float32))
        ]
    })

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }

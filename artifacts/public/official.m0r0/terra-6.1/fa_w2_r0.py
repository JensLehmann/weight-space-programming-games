import numpy as np


def build():
    graph = []
    params = {}
    ni = 0
    pi = 0

    def param(value):
        nonlocal pi
        pi += 1
        key = "p%d" % pi
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def op(kind, *inputs, **fields):
        nonlocal ni
        ni += 1
        key = "n%d" % ni
        item = {"id": key, "op": kind, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return key

    def sl(x, a, b):
        return op("slice", x, start=a, stop=b)

    def vs(x, a, b):
        return op("vslice", x, start=a, stop=b)

    def cat(*xs):
        return op("concat", *xs)

    def vc(*xs):
        return op("vconcat", *xs)

    def conv1(x, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        fields = {"w": param(w[:, :, None, None])}
        if b is not None:
            fields["b"] = param(b)
        return op("conv", x, **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return op("dense", x, **fields)

    def saff(x, channels, scale=1.0, bias=0.0):
        return conv1(
            x,
            np.eye(channels, dtype=np.float32) * scale,
            np.full(channels, bias, dtype=np.float32),
        )

    def add(x, y):
        return op("add", x, y)

    def sub(x, y):
        return op("sub", x, y)

    def mul(x, y):
        return op("mul", x, y)

    def va(x, y):
        return op("vadd", x, y)

    def vm(x, y):
        return op("vmul", x, y)

    def relu(x):
        return op("relu", x)

    def vr(x):
        return op("vrelu", x)

    seed = vs("mem", 0, 1)
    scalar_cache = {}

    def scalar(value):
        value = float(value)
        if value not in scalar_cache:
            scalar_cache[value] = dense(seed, [[0.0]], [value])
        return scalar_cache[value]

    def vf(x, scale=1.0, bias=0.0):
        if scale != 1.0:
            x = vm(x, scalar(scale))
        if bias != 0.0:
            x = va(x, scalar(bias))
        return x

    def vabs(x):
        return va(vr(x), vr(vf(x, -1.0)))

    def vclip(x):
        positive = vr(x)
        return va(positive, vf(vr(vf(positive, 1.0, -1.0)), -1.0))

    def vmax(x, y):
        return va(y, vr(va(x, vf(y, -1.0))))

    def veq(x, value):
        return vclip(vf(vabs(vf(x, 1.0, -float(value))), -2.0, 1.0))

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)

    def sclip(x):
        return op("min2", relu(x), one)

    def point(coord, target):
        d = saff(coord, 1, 63.0, -float(target))
        return relu(saff(op("abs", d), 1, -1.0, 1.0))

    # Border colours identify the wall palette.
    corner_y = op("max2", point(row, 1), point(row, 62))
    corner_x = op("max2", point(col, 1), point(col, 62))
    corner_mask = mul(corner_y, corner_x)
    border = vclip(vf(op("gmax", mul("x0", corner_mask)), 2.0, -1.0))

    wall_weights = np.ones(16, dtype=np.float32)
    wall_weights[[5, 9, 10]] = 0.0
    wall_colours = dense(border, np.diag(wall_weights))
    walls = sclip(
        conv1(mul("x0", wall_colours), np.ones((1, 16), np.float32))
    )

    # Detect long coloured gates and separate isolated switches.
    colours = cat(sl("x0", 12, 13), sl("x0", 14, 15), sl("x0", 15, 16))
    colour_border = vc(vs(border, 12, 13), vs(border, 14, 15),
                       vs(border, 15, 16))
    colours = mul(colours, vf(colour_border, -1.0, 1.0))

    run_kernel = np.zeros((6, 3, 7, 7), dtype=np.float32)
    for c in range(3):
        run_kernel[c, c, 3, :] = 1.0
        run_kernel[c + 3, c, :, 3] = 1.0
    runs = op(
        "conv", colours, w=param(run_kernel),
        b=param(np.full(6, -5.5, np.float32))
    )
    runs = sclip(saff(relu(runs), 6, 2.0, 0.0))
    run_union = op("max2", sl(runs, 0, 3), sl(runs, 3, 6))
    dilation = np.zeros((3, 3, 7, 7), dtype=np.float32)
    for c in range(3):
        dilation[c, c, :, :] = 1.0
    gate_regions = sclip(op("conv", run_union, w=param(dilation)))
    gates = sclip(
        conv1(mul(gate_regions, colours), np.ones((1, 3), np.float32))
    )
    switches = mul(colours, saff(gate_regions, 3, -1.0, 1.0))

    box_kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
    boxes = sclip(op("conv", sl("x0", 9, 10), w=param(box_kernel)))
    obstacle = op("max2", walls, op("max2", gates, boxes))
    passable = saff(obstacle, 1, -1.0, 1.0)

    selected_pixels = sl("x0", 1, 2)
    selected = vm(op("gmax", selected_pixels),
                  vf(vs(border, 1, 2), -1.0, 1.0))
    players = op("max2", sl("x0", 10, 11), mul(selected_pixels, selected))
    passable = op("max2", passable, players)

    # Memory: 32 recent joint positions, phase, box stage, switch
    # coordinates, and a 16-state exploration clock.
    H = 32
    history = vs("mem", 0, 4 * H)
    previous = vs("mem", 0, 4)
    phase = vs("mem", 128, 129)
    box_stage = vs("mem", 129, 130)
    old_switches = vs("mem", 130, 136)
    clock = vs("mem", 136, 152)
    phase0 = veq(phase, 0)
    phase1 = veq(phase, 1)
    phase2 = veq(phase, 2)

    # Initially split markers spatially; subsequently track by proximity.
    inverse_col = saff(col, 1, -1.0, 1.0)
    inverse_row = saff(row, 1, -1.0, 1.0)
    xmax = op("gmax", mul(players, col))
    xmin_inv = op("gmax", mul(players, inverse_col))
    ymax = op("gmax", mul(players, row))
    ymin_inv = op("gmax", mul(players, inverse_row))
    xstats = vc(xmax, xmin_inv)
    ystats = vc(ymax, ymin_inv)
    xmid = dense(xstats, [[0.5, -0.5]], [0.5])
    ymid = dense(ystats, [[0.5, -0.5]], [0.5])
    xspan = dense(xstats, [[63.0, 63.0]], [-63.0])
    use_x = vclip(vf(xspan, 1.0, -6.0))
    left_mask = sclip(saff(sub(xmid, col), 1, 63.0, 0.5))
    top_mask = sclip(saff(sub(ymid, row), 1, 63.0, 0.5))
    initial_mask = add(mul(left_mask, use_x),
                       mul(top_mask, vf(use_x, -1.0, 1.0)))

    old_ax = vs(previous, 0, 1)
    old_ay = vs(previous, 1, 2)
    old_bx = vs(previous, 2, 3)
    old_by = vs(previous, 3, 4)
    da = add(op("abs", sub(col, old_ax)), op("abs", sub(row, old_ay)))
    db = add(op("abs", sub(col, old_bx)), op("abs", sub(row, old_by)))
    tracked_mask = sclip(saff(sub(db, da), 1, 32.0, 0.5))
    assignment = add(mul(initial_mask, phase0),
                     mul(tracked_mask, vf(phase0, -1.0, 1.0)))
    tentative_a = mul(players, assignment)
    tentative_b = mul(players, saff(assignment, 1, -1.0, 1.0))

    def position_of(marker):
        return op("gmax", mul(marker, col)), op("gmax", mul(marker, row))

    tax, tay = position_of(tentative_a)
    tbx, tby = position_of(tentative_b)

    # Probe ACTION3 to infer the horizontal assignment.
    probe_delta = dense(vc(tax, old_ax, tbx, old_bx),
                        [[1.0, -1.0, -1.0, 1.0]])
    probe_positive = vclip(vf(probe_delta, 1000.0))
    swap = vm(vm(probe_positive, phase1), vs("aprev", 3, 4))
    a = add(mul(tentative_a, vf(swap, -1.0, 1.0)),
            mul(tentative_b, swap))
    b = add(mul(tentative_b, vf(swap, -1.0, 1.0)),
            mul(tentative_a, swap))
    ax, ay = position_of(a)
    bx, by = position_of(b)
    position = vc(ax, ay, bx, by)

    width = op("gmax", op("rowsum", a))
    height = op("gmax", op("colsum", a))
    extent = va(width, vf(vr(dense(vc(width, height), [[1.0, -1.0]])), -1.0))
    scale5 = vclip(vf(extent, 1.0, -4.0))
    scale4 = vf(scale5, -1.0, 1.0)
    step_size = vf(scale5, 1.0 / 63.0, 4.0 / 63.0)

    # Remember switch positions if covered by a player.
    switch_sources = []
    next_switches = []
    for c in range(3):
        visible = sl(switches, c, c + 1)
        new_x = op("gmax", mul(visible, col))
        new_y = op("gmax", mul(visible, row))
        old_x = vs(old_switches, 2 * c, 2 * c + 1)
        old_y = vs(old_switches, 2 * c + 1, 2 * c + 2)
        has_old = vclip(vf(old_x, 64.0))
        cached_x = va(old_x, vm(vf(has_old, -1.0, 1.0), new_x))
        cached_y = va(old_y, vm(vf(has_old, -1.0, 1.0), new_y))
        next_switches.extend([cached_x, cached_y])

        exists = vclip(vf(cached_x, 64.0))
        dx = op("abs", sub(col, cached_x))
        dy = op("abs", sub(row, cached_y))
        mx = sclip(saff(dx, 1, -63.0, 4.0))
        my = sclip(saff(dy, 1, -63.0, 4.0))
        cached = mul(mul(mx, my), exists)
        switch_sources.append(op("max2", visible, cached))

    sources = cat(a, b, *switch_sources)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    wave_kernel = np.zeros((20, 5, 3, 3), dtype=np.float32)
    for d, (dy, dx) in enumerate(directions):
        for c in range(5):
            wave_kernel[5 * d + c, c, 1 + dy, 1 + dx] = 1.0

    graph.append({
        "id": "wave",
        "op": "loop",
        "T": 240,
        "state": [{"id": "wave_state", "init": sources}],
        "body": [
            {"id": "wave_shift", "op": "conv", "in": ["wave_state"],
             "w": param(wave_kernel)},
            {"id": "wave_u", "op": "slice", "in": ["wave_shift"],
             "start": 0, "stop": 5},
            {"id": "wave_d", "op": "slice", "in": ["wave_shift"],
             "start": 5, "stop": 10},
            {"id": "wave_l", "op": "slice", "in": ["wave_shift"],
             "start": 10, "stop": 15},
            {"id": "wave_r", "op": "slice", "in": ["wave_shift"],
             "start": 15, "stop": 20},
            {"id": "wave_ud", "op": "max2", "in": ["wave_u", "wave_d"]},
            {"id": "wave_lr", "op": "max2", "in": ["wave_l", "wave_r"]},
            {"id": "wave_near", "op": "max2",
             "in": ["wave_ud", "wave_lr"]},
            {"id": "wave_cost", "op": "conv", "in": ["wave_near"],
             "w": param(np.eye(5, dtype=np.float32)[:, :, None, None]),
             "b": param(np.full(5, -1.0 / 512.0, dtype=np.float32))},
            {"id": "wave_positive", "op": "relu", "in": ["wave_cost"]},
            {"id": "wave_best", "op": "max2",
             "in": ["wave_state", "wave_positive"]},
            {"id": "wave_next", "op": "mul",
             "in": ["wave_best", passable]},
        ],
        "next": {"wave_state": "wave_next"},
    })

    fields = "wave.wave_state"
    field_a = sl(fields, 0, 1)
    field_b = sl(fields, 1, 2)
    base_a = op("gmax", mul(a, field_b))
    base_b = op("gmax", mul(b, field_a))
    pair_quality = va(base_a, base_b)
    connected = vclip(vf(pair_quality, 4.0))
    disconnected = vf(connected, -1.0, 1.0)

    # Shift field samples by either four or five display pixels.
    shift_input = cat(fields, passable)
    kernel3 = np.zeros((24, 6, 7, 7), dtype=np.float32)
    for d, (dy, dx) in enumerate(directions):
        for c in range(6):
            kernel3[6 * d + c, c, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    shift3 = op("conv", shift_input, w=param(kernel3))

    extra_kernel = np.zeros((48, 24, 5, 5), dtype=np.float32)
    for si, extra in enumerate((1, 2)):
        for d, (dy, dx) in enumerate(directions):
            for c in range(6):
                extra_kernel[si * 24 + d * 6 + c, d * 6 + c,
                             2 + extra * dy, 2 + extra * dx] = 1.0
    shifted = op("conv", shift3, w=param(extra_kernel))

    def projected(si, component):
        return cat(*[
            sl(shifted, si * 24 + d * 6 + component,
               si * 24 + d * 6 + component + 1)
            for d in range(4)
        ])

    reverse = np.zeros((4, 4), dtype=np.float32)
    for i, j in enumerate((0, 1, 3, 2)):
        reverse[i, j] = 1.0

    def evaluate(si):
        open_map = projected(si, 5)
        open_a = vclip(op("gmax", mul(open_map, a)))
        open_b_local = vclip(op("gmax", mul(open_map, b)))
        blocked_a = vf(open_a, -1.0, 1.0)
        blocked_b = vf(open_b_local, -1.0, 1.0)
        qa = va(op("gmax", mul(projected(si, 1), a)),
                vm(blocked_a, base_a))
        qb = va(op("gmax", mul(projected(si, 0), b)),
                vm(blocked_b, base_b))
        pair = va(qa, dense(qb, reverse))
        open_b = dense(open_b_local, reverse)

        button_scores = []
        for c in range(3):
            field = sl(fields, c + 2, c + 3)
            ca0 = op("gmax", mul(field, a))
            cb0 = op("gmax", mul(field, b))
            sample = projected(si, c + 2)
            ca = va(op("gmax", mul(sample, a)), vm(blocked_a, ca0))
            cb = va(op("gmax", mul(sample, b)), vm(blocked_b, cb0))
            button_scores.append(vmax(ca, dense(cb, reverse)))
        return pair, open_a, open_b, button_scores

    pair4, oa4, ob4, buttons4 = evaluate(0)
    pair5, oa5, ob5, buttons5 = evaluate(1)

    def choose(v4, v5):
        return va(vm(v4, scale4), vm(v5, scale5))

    scores = choose(pair4, pair5)
    open_a = choose(oa4, oa5)
    open_b = choose(ob4, ob5)

    # Switches are more attractive when the markers are disconnected.
    button_weight = vf(disconnected, 0.9, 0.25)
    for c in range(3):
        scores = va(scores, vm(choose(buttons4[c], buttons5[c]), button_weight))

    any_move = vclip(va(open_a, open_b))
    scores = va(scores, vf(any_move, 0.1002, -0.1))

    # Predict successor positions.
    repeat = np.zeros((16, 4), dtype=np.float32)
    for d in range(4):
        for k in range(4):
            repeat[4 * d + k, k] = 1.0
    repeated_position = dense(position, repeat)

    movement = vm(vc(open_a, open_b), step_size)
    movement_weights = np.zeros((16, 8), dtype=np.float32)
    for d, (dy, dx) in enumerate(directions):
        movement_weights[4 * d, d] = dx
        movement_weights[4 * d + 1, d] = dy
        movement_weights[4 * d + 2, 4 + d] = -dx
        movement_weights[4 * d + 3, 4 + d] = dy
    candidates = va(repeated_position, dense(movement, movement_weights))

    # A geometric fallback helps when no distance field reaches the
    # other marker.
    geometric_weights = np.zeros((8, 16), dtype=np.float32)
    for d in range(4):
        geometric_weights[2 * d, 4 * d] = 1.0
        geometric_weights[2 * d, 4 * d + 2] = -1.0
        geometric_weights[2 * d + 1, 4 * d + 1] = 1.0
        geometric_weights[2 * d + 1, 4 * d + 3] = -1.0
    geom_components = vabs(dense(candidates, geometric_weights))
    geom_sum = np.zeros((4, 8), dtype=np.float32)
    for d in range(4):
        geom_sum[d, 2 * d:2 * d + 2] = 1.0
    geometric_distance = dense(geom_components, geom_sum)
    scores = va(scores, vm(vf(geometric_distance, -0.10), disconnected))

    # Compare each candidate with the 32 recent joint positions. The
    # factored construction avoids large diagonal parameter arrays.
    expand = np.zeros((4 * H, 4), dtype=np.float32)
    for d in range(4):
        expand[d * H:(d + 1) * H, d] = 1.0
    expand_param = param(expand)
    distance_parts = []
    for k in range(4):
        select_candidate = np.zeros((4, 16), dtype=np.float32)
        for d in range(4):
            select_candidate[d, 4 * d + k] = 1.0
        candidate_coord = dense(candidates, select_candidate)
        expanded = op("dense", candidate_coord, w=expand_param)

        select_history = np.zeros((H, 4 * H), dtype=np.float32)
        for h in range(H):
            select_history[h, 4 * h + k] = 1.0
        history_coord = dense(history, select_history)
        history_repeated = vc(history_coord, history_coord,
                              history_coord, history_coord)
        distance_parts.append(vabs(va(expanded, vf(history_repeated, -1.0))))

    distances = distance_parts[0]
    for part in distance_parts[1:]:
        distances = va(distances, part)
    matches = vclip(vf(distances, -500.0, 1.0))
    penalty_weights = np.zeros((4, 4 * H), dtype=np.float32)
    for d in range(4):
        for h in range(H):
            penalty_weights[d, d * H + h] = -0.025 / (1.0 + 0.08 * h)
    scores = va(scores, dense(matches, penalty_weights))

    # Penalize the preceding action when neither marker actually moved.
    actual_delta = vabs(va(position, vf(previous, -1.0)))
    actual_distance = dense(actual_delta, np.ones((1, 4), np.float32))
    unchanged = vclip(vf(actual_distance, -500.0, 1.0))
    unchanged = vm(unchanged, vf(phase0, -1.0, 1.0))
    previous_direction = vs("aprev", 1, 5)
    scores = va(scores, vm(previous_direction, vf(unchanged, -0.12)))

    rotation = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        rotation[(i + 1) % 16, i] = 1.0
    rotated = dense(clock, rotation)
    clock_empty = vr(dense(clock, -np.ones((1, 16), np.float32), [1.0]))
    clock_seed = np.zeros((16, 1), dtype=np.float32)
    clock_seed[0, 0] = 1.0
    next_clock = va(rotated, dense(clock_empty, clock_seed))

    preferences = np.zeros((4, 16), dtype=np.float32)
    order = (0, 0, 3, 3, 1, 1, 2, 2, 0, 3, 1, 2, 0, 1, 3, 2)
    for i, action in enumerate(order):
        preferences[action, i] = 0.0004
    scores = va(scores, dense(next_clock, preferences))

    # Box fallback: select a central box, move right and up, deselect.
    box_present = op("gmax", sl("x0", 9, 10))
    stage0 = veq(box_stage, 0)
    stage1 = veq(box_stage, 1)
    stage2 = veq(box_stage, 2)
    stage3 = veq(box_stage, 3)
    not_selected = vf(selected, -1.0, 1.0)
    start_box = vm(vm(vm(vm(disconnected, box_present), stage0), phase2),
                   not_selected)
    manual_deselect = vm(selected, stage0)
    probe = vm(phase0, not_selected)
    next_stage = va(box_stage, va(va(start_box, stage1), va(stage2, stage3)))
    phase_increment = vm(vf(phase2, -1.0, 1.0),
                         vf(manual_deselect, -1.0, 1.0))
    next_phase = va(phase, phase_increment)

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [
            vc(position, vs(history, 0, 4 * (H - 1))),
            next_phase,
            next_stage,
            vc(*next_switches),
            next_clock,
        ],
    })

    logits_input = vc(scores, start_box, stage1, stage2, stage3,
                      manual_deselect, probe)
    output_weights = np.zeros((8, 10), dtype=np.float32)
    output_bias = np.full(8, -20.0, dtype=np.float32)
    for i in range(4):
        output_weights[i + 1, i] = 1.0
        output_bias[i + 1] = 0.0
    output_weights[1, 6] = 100.0
    output_weights[4, 5] = 100.0
    output_weights[3, 9] = 50.0
    output_weights[6, 4] = 100.0
    output_weights[6, 7] = 100.0
    output_weights[6, 8] = 100.0
    output_bias[6] = -10.0
    graph.append({
        "id": "logits", "op": "dense", "in": [logits_input],
        "w": param(output_weights), "b": param(output_bias),
    })

    centrality = saff(op("abs", saff(col, 1, 1.0, -0.5)), 1, -0.5, 1.0)
    box_click = mul(sl("x0", 9, 10), centrality)
    deselect_click = sl("x0", 1, 2)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [
            mul(box_click, start_box),
            mul(deselect_click, va(manual_deselect, stage3)),
        ],
    })

    return {"graph": graph, "memory_size": 152, "params": params}

import numpy as np


def build():
    graph, params = [], {}
    sink = graph
    serial = 0
    pserial = 0

    def uid(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def par(a):
        nonlocal pserial
        pserial += 1
        name = "p" + str(pserial)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = uid()
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        sink.append(d)
        return name

    def conv(x, w, b=None):
        f = {"w": par(w)}
        if b is not None:
            f["b"] = par(b)
        return node("conv", [x], **f)

    def dense(x, w, b=None):
        f = {"w": par(w)}
        if b is not None:
            f["b"] = par(b)
        return node("dense", [x], **f)

    def lin(x, a, b=0.0):
        return dense(x, [[a]], [b])

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

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

    def vr(x):
        return node("vrelu", [x])

    def gm(x):
        return node("gmax", [x])

    one_s = sl("coords", 2, 3)
    one_v = gm(one_s)

    def sclip(x):
        return node("min2", [node("relu", [x]), one_s])

    def vclip(x):
        return va(vr(x), lin(vr(lin(x, 1.0, -1.0)), -1.0))

    def vabs(x):
        return va(vr(x), vr(lin(x, -1.0)))

    def veq(x, c):
        return vclip(lin(vabs(lin(x, 1.0, -c)), -4.0, 2.0))

    def affine(x, weights, bias=0.0):
        return conv(
            x,
            np.asarray(weights, np.float32).reshape(1, -1, 1, 1),
            [bias],
        )

    def seq(x, value):
        return sclip(affine(
            node("abs", [affine(x, [1.0], -value)]),
            [-4.0], 2.0,
        ))

    # One channel per interior cell, with a single nonzero centre pixel.
    cw = np.zeros((20, 3, 1, 1), np.float32)
    cb = np.zeros(20, np.float32)
    for i in range(10):
        cw[i, 0, 0, 0] = 63.0
        cb[i] = -(7 + 5 * i)
        cw[10 + i, 1, 0, 0] = 63.0
        cb[10 + i] = -(11 + 5 * i)
    tents = sclip(conv(
        node("abs", [conv("coords", cw, cb)]),
        (-4.0 * np.eye(20, dtype=np.float32))[:, :, None, None],
        np.full(20, 2.0, np.float32),
    ))
    bw = np.zeros((100, 20, 1, 1), np.float32)
    for i in range(10):
        for j in range(10):
            bw[10 * i + j, i, 0, 0] = 1.0
            bw[10 * i + j, 10 + j, 0, 0] = 1.0
    basis = node("relu", [
        conv(tents, bw, np.full(100, -1.0, np.float32))
    ])
    sum100_conv = par(np.ones((1, 100, 1, 1), np.float32))
    gain100 = par(4096.0 * np.eye(100, dtype=np.float32))
    centres = node("conv", [basis], w=sum100_conv)

    palette = cat(*(sl("x0", c, c + 1) for c in [12, 9, 14, 8]))
    coloured = conv(palette, np.ones((1, 4, 1, 1), np.float32))
    black, grey = sl("x0", 0, 1), sl("x0", 5, 6)
    blue, wall = sl("x0", 1, 2), sl("x0", 4, 5)

    shapes = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]
    patterns = [
        np.rot90(np.asarray(a, np.float32), -r).copy()
        for a in shapes for r in range(4)
    ]
    pw = np.zeros((24, 1, 3, 3), np.float32)
    uw = np.zeros((24, 1, 5, 5), np.float32)
    pb = np.zeros(24, np.float32)
    for k, a in enumerate(patterns):
        pw[k, 0] = 2.0 * a - 1.0
        uw[k, 0, ::2, ::2] = 2.0 * a - 1.0
        pb[k] = 1.0 - a.sum()
    goal_patterns = node("relu", [conv(coloured, pw, pb)])
    ui_patterns = node("relu", [conv(coloured, uw, pb)])

    uiw = np.zeros((2, 3, 1, 1), np.float32)
    uiw[0, 0, 0, 0] = uiw[1, 1, 0, 0] = 63.0
    ui_delta = node("abs", [conv("coords", uiw, [-57.0, -5.0])])
    ui_point = sclip(affine(ui_delta, [-4.0, -4.0], 2.0))
    ui_region = sclip(affine(ui_delta, [-1.0, -1.0], 7.0))
    ui_features = vcat(
        gm(mul(ui_patterns, ui_point)),
        gm(mul(palette, ui_region)),
    )
    sw = np.zeros((96, 28), np.float32)
    for p in range(24):
        for c in range(4):
            sw[4 * p + c, p] = 1.0
            sw[4 * p + c, 24 + c] = 1.0
    current_state = vr(dense(
        ui_features, sw, np.full(96, -1.0, np.float32)
    ))

    player_w = np.zeros((1, 16, 5, 5), np.float32)
    player_w[0, 12, :2, :] = 1.0
    player_w[0, 9, 2:, :] = 1.0
    player_mask = mul(
        node("relu", [conv("x0", player_w, [-24.0])]),
        centres,
    )

    ring_w = np.zeros((1, 1, 7, 7), np.float32)
    ring_w[0, 0, 0, :] = ring_w[0, 0, 6, :] = 1.0
    ring_w[0, 0, :, 0] = ring_w[0, 0, :, 6] = 1.0
    ring = node("relu", [conv(grey, ring_w, [-23.0])])
    pattern_present = conv(
        goal_patterns, np.ones((1, 24, 1, 1), np.float32)
    )
    goal_observed = sclip(affine(
        cat(pattern_present, ring), [1.0, 1.0], -1.0
    ))
    pcw = np.zeros((4, 4, 3, 3), np.float32)
    for c in range(4):
        pcw[c, c] = 1.0
    goal_colours = sclip(conv(palette, pcw))
    code_w = np.zeros((1, 28, 1, 1), np.float32)
    for p in range(24):
        code_w[0, p, 0, 0] = 4.0 * p
    for c in range(4):
        code_w[0, 24 + c, 0, 0] = float(c)
    goal_extra = mul(
        conv(cat(goal_patterns, goal_colours), code_w, [15.0]),
        goal_observed,
    )

    def binary_marker(a):
        a = np.asarray(a, np.float32)
        return node("relu", [
            conv(black, (2.0 * a - 1.0)[None, None],
                 [1.0 - a.sum()])
        ])

    shape_marker = binary_marker(
        [[1, 0, 0], [0, 1, 1], [0, 1, 0]]
    )
    rotation_marker = binary_marker(
        [[0, 1, 0], [0, 1, 1], [0, 0, 0]]
    )
    color_w = np.zeros((1, 16, 3, 3), np.float32)
    picture = [[9, 14, 14], [9, 0, 8], [12, 12, 8]]
    for i in range(3):
        for j in range(3):
            color_w[0, picture[i][j], i, j] = 1.0
    colour_marker = node("relu", [conv("x0", color_w, [-8.0])])

    battery_w = np.ones((1, 1, 3, 3), np.float32)
    battery_w[0, 0, 1, 1] = 0.0
    battery_marker = node("relu", [
        conv(sl("x0", 11, 12), battery_w, [-7.0])
    ])
    push_w = np.zeros((4, 1, 7, 7), np.float32)
    push_w[0, 0, 1:6, 0] = 1.0
    push_w[1, 0, 1:6, 6] = 1.0
    push_w[2, 0, 0, 1:6] = 1.0
    push_w[3, 0, 6, 1:6] = 1.0
    push_observed = node("relu", [
        conv(blue, push_w, np.full(4, -4.0, np.float32))
    ])

    non_grey_w = np.ones((1, 16, 1, 1), np.float32)
    non_grey_w[0, 5, 0, 0] = 0.0
    visibility = sclip(conv(
        conv(
            conv("x0", non_grey_w),
            np.ones((1, 1, 7, 7), np.float32),
        ),
        np.ones((1, 1, 3, 3), np.float32),
    ))

    old_cells = vsl("mem", 0, 100)
    old_map = node("conv", [mul(basis, old_cells)], w=sum100_conv)
    observed_code = affine(
        cat(wall, shape_marker, colour_marker, rotation_marker,
            battery_marker, push_observed, goal_extra),
        [-2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0],
        1.0,
    )

    # Preserve covered modifiers, unless their moving sprite is seen elsewhere.
    preserve = None
    for code, marker in zip(
        [2, 3, 4], [shape_marker, colour_marker, rotation_marker]
    ):
        unseen_marker = lin(gm(mul(marker, centres)), -1.0, 1.0)
        term = mul(mul(seq(old_map, float(code)), player_mask),
                   unseen_marker)
        preserve = term if preserve is None else add(preserve, term)
    preserve = sclip(preserve)
    update_visibility = mul(
        visibility, affine(preserve, [-1.0], 1.0)
    )
    updated_map = add(
        mul(observed_code, update_visibility),
        mul(old_map, affine(update_visibility, [-1.0], 1.0)),
    )
    next_cells = node(
        "dense", [node("gmean", [mul(basis, updated_map)])],
        w=gain100,
    )
    cell_map = mul(updated_map, centres)

    def code_mask(k):
        return mul(seq(cell_map, float(k)), centres)

    stations = [code_mask(2), code_mask(3), code_mask(4)]
    batteries, unknown = code_mask(5), code_mask(0)
    pushes = [code_mask(k) for k in range(6, 10)]
    push_total = conv(
        cat(*pushes), np.ones((1, 4, 1, 1), np.float32)
    )
    all_goals = mul(sclip(affine(cell_map, [1.0], -15.0)), centres)
    goal_exists = gm(all_goals)
    goal_rank = mul(
        conv(
            "coords",
            np.asarray([-63.0, -4032.0, 0.0],
                       np.float32).reshape(1, 3, 1, 1),
            [10000.0],
        ),
        all_goals,
    )
    selected_goal = mul(
        sclip(affine(
            node("abs", [sub(goal_rank, gm(goal_rank))]),
            [-4.0], 2.0,
        )),
        all_goals,
    )
    selected_code = gm(mul(cell_map, selected_goal))

    dw = np.zeros((192, 1), np.float32)
    db = np.zeros(192, np.float32)
    fw = np.zeros((96, 192), np.float32)
    for s in range(96):
        dw[2 * s, 0], dw[2 * s + 1, 0] = 1.0, -1.0
        db[2 * s], db[2 * s + 1] = -(16.0 + s), 16.0 + s
        fw[s, 2 * s:2 * s + 2] = -1.0
    decoded_goal = vr(dense(
        vr(dense(selected_code, dw, db)),
        fw, np.ones(96, np.float32),
    ))
    no_goal = lin(goal_exists, -1.0, 1.0)
    goal_state = va(vm(decoded_goal, goal_exists),
                    vm(current_state, no_goal))
    goal_source = add(selected_goal, mul(unknown, no_goal))

    floor = mul(sclip(affine(cell_map, [1.0], 1.0)), centres)
    transit_floor = mul(floor, affine(all_goals, [-1.0], 1.0))
    ordinary_floor = mul(
        transit_floor, affine(push_total, [-1.0], 1.0)
    )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift3_w = np.zeros((4, 1, 7, 7), np.float32)
    shift3_names, shift2_names = [], []
    for d, (dy, dx) in enumerate(directions):
        shift3_w[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        a = np.zeros((1, 1, 7, 7), np.float32)
        a[0, 0, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        shift3_names.append(par(a))
        a = np.zeros((1, 1, 5, 5), np.float32)
        a[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        shift2_names.append(par(a))
    shift3_all = par(shift3_w)
    minus_one = par([-1.0])

    def shift(x, d):
        return node("conv", [
            node("conv", [x], w=shift3_names[d])
        ], w=shift2_names[d])

    def neighbours(x):
        a = node("conv", [x], w=shift3_all)
        return [
            node("conv", [sl(a, d, d + 1)],
                 w=shift2_names[d], b=minus_one)
            for d in range(4)
        ]

    row_pixel = affine(sl("coords", 0, 1), [63.0])
    col_pixel = affine(sl("coords", 1, 2), [63.0])
    coordinate_fields = [row_pixel, row_pixel, col_pixel, col_pixel]
    neighbour_free = [shift(transit_floor, d) for d in range(4)]
    terminal, scan_init = [], []
    for d in range(4):
        end_here = mul(transit_floor, sub(centres, neighbour_free[d]))
        terminal.append(mul(coordinate_fields[d], end_here))
        scan_init.append(mul(coordinate_fields[d], transit_floor))

    scan_id = uid("scan")
    scan_states = [uid("ss") for _ in range(4)]
    body = []
    sink = body
    scan_next = {
        scan_states[d]: add(
            mul(shift(scan_states[d], d), neighbour_free[d]),
            terminal[d],
        )
        for d in range(4)
    }
    sink = graph
    graph.append({
        "id": scan_id, "op": "loop", "T": 12,
        "state": [
            {"id": scan_states[d], "init": scan_init[d]}
            for d in range(4)
        ],
        "body": body, "next": scan_next,
    })
    endpoints = [scan_id + "." + s for s in scan_states]

    landing_rank = None
    for p, d in enumerate([3, 2, 1, 0]):
        if p < 2:
            rank = affine(cat(row_pixel, endpoints[d]), [64.0, 1.0])
        else:
            rank = affine(cat(endpoints[d], col_pixel), [64.0, 1.0])
        term = mul(rank, pushes[p])
        landing_rank = term if landing_rank is None else add(landing_rank, term)

    landing_codes = np.asarray([
        64.0 * (7 + 5 * i) + (11 + 5 * j)
        for i in range(10) for j in range(10)
    ], np.float32)
    landing_delta = node("abs", [
        conv(landing_rank,
             np.ones((100, 1, 1, 1), np.float32),
             -landing_codes)
    ])
    landing_matches = sclip(conv(
        landing_delta,
        (-4.0 * np.eye(100, dtype=np.float32))[:, :, None, None],
        np.full(100, 2.0, np.float32),
    ))
    landing_lookup = mul(landing_matches, push_total)

    masks = stations + [goal_source, batteries, unknown]
    seeds = [affine(m, [128.0]) for m in masks]
    cap = lin(one_v, 0.0, 128.0)
    route_id = uid("routes")
    route_states = [uid("rs") for _ in range(6)]
    body = []
    sink = body
    route_next = {}
    for k in range(6):
        state = route_states[k]
        ns = neighbours(state)
        best = ns[0]
        for q in ns[1:]:
            best = node("max2", [best, q])
        walking = mul(node("relu", [best]), ordinary_floor)
        walking = node("max2", [walking, seeds[k]])
        cell_values = node(
            "dense", [node("gmean", [mul(state, basis)])],
            w=gain100,
        )
        forced = node(
            "conv", [mul(landing_lookup, cell_values)],
            w=sum100_conv,
        )
        route_next[state] = node("min2", [
            node("max2", [walking, forced]), cap
        ])
    sink = graph
    graph.append({
        "id": route_id, "op": "loop", "T": 52,
        "state": [
            {"id": route_states[k], "init": seeds[k]}
            for k in range(6)
        ],
        "body": body, "next": route_next,
    })
    fields = [route_id + "." + s for s in route_states]

    def negative_distance(field, location, reentry=False):
        value = gm(mul(field, location))
        unreachable = vr(lin(value, -1.0, 1.0))
        result = va(lin(value, 1.0, -128.0),
                    lin(unreachable, -1024.0))
        if reentry:
            result = lin(vr(lin(result, -1.0, -2.0)), -1.0, -2.0)
        return result

    minus_two = lin(one_v, 0.0, -2.0)
    station_costs = [
        [
            minus_two if i == j else
            negative_distance(fields[j], stations[i])
            for j in range(3)
        ] for i in range(3)
    ]
    goal_costs = [
        negative_distance(fields[3], stations[i])
        for i in range(3)
    ]
    player_costs = [
        negative_distance(fields[j], player_mask, True)
        for j in range(3)
    ]
    player_goal_cost = negative_distance(fields[3], player_mask)
    player_battery_cost = negative_distance(fields[4], player_mask)

    reward = vm(goal_state, lin(one_v, 0.0, 512.0))
    macro_init = [vr(va(reward, c)) for c in goal_costs]
    permutations = []
    for operation in range(3):
        w = np.zeros((96, 96), np.float32)
        for s in range(96):
            p, colour = divmod(s, 4)
            shape, rotation = divmod(p, 4)
            if operation == 0:
                shape = (shape + 1) % 6
            elif operation == 1:
                colour = (colour + 1) % 4
            else:
                rotation = (rotation + 1) % 4
            w[s, (shape * 4 + rotation) * 4 + colour] = 1.0
        permutations.append(par(w))
    negative96 = par(-np.eye(96, dtype=np.float32))

    def vmax96(a, b):
        return va(vr(va(a, node("dense", [b], w=negative96))), b)

    macro_id = uid("macro")
    macro_states = [uid("ms") for _ in range(3)]
    body = []
    sink = body
    transformed = [
        node("dense", [macro_states[j]], w=permutations[j])
        for j in range(3)
    ]
    macro_next = {}
    for i in range(3):
        best = macro_init[i]
        for j in range(3):
            best = vmax96(best, va(transformed[j], station_costs[i][j]))
        macro_next[macro_states[i]] = best
    sink = graph
    graph.append({
        "id": macro_id, "op": "loop", "T": 20,
        "state": [
            {"id": macro_states[i], "init": macro_init[i]}
            for i in range(3)
        ],
        "body": body, "next": macro_next,
    })
    sum96 = par(np.ones((1, 96), np.float32))

    def read_current(v):
        return node("dense", [vm(v, current_state)], w=sum96)

    raw_scores = [read_current(vr(va(reward, player_goal_cost)))]
    for j in range(3):
        value = node(
            "dense", [macro_id + "." + macro_states[j]],
            w=permutations[j],
        )
        raw_scores.append(read_current(vr(va(value, player_costs[j]))))

    def vmax_scalar(a, b):
        return va(vr(va(a, lin(b, -1.0))), b)

    raw_best = raw_scores[0]
    for q in raw_scores[1:]:
        raw_best = vmax_scalar(raw_best, q)
    fallback = vclip(lin(raw_best, -1.0, 1.0))
    scores = [
        lin(q, 1.0, b)
        for q, b in zip(raw_scores, [.03, .02, .01, 0.0])
    ]
    best_score = scores[0]
    for q in scores[1:]:
        best_score = vmax_scalar(best_score, q)
    winners = [
        vclip(lin(vabs(va(q, lin(best_score, -1.0))), -500.0, 2.0))
        for q in scores
    ]

    bar_w = np.zeros((2, 3, 1, 1), np.float32)
    bar_w[0, 0, 0, 0] = bar_w[1, 1, 0, 0] = 63.0
    bar_delta = node("abs", [
        conv("coords", bar_w, [-61.5, -33.5])
    ])
    bar = mul(
        sclip(affine(sl(bar_delta, 0, 1), [-1.0], 1.5)),
        sclip(affine(sl(bar_delta, 1, 2), [-1.0], 21.5)),
    )
    steps = lin(
        node("gmean", [mul(sl("x0", 11, 12), bar)]), 2048.0
    )
    previous_steps = vsl("mem", 100, 101)
    decrement = va(previous_steps, lin(steps, -1.0))
    budget = lin(
        vm(steps, lin(veq(decrement, 1.0), .5, .5)),
        1.0, 1.0,
    )

    remaining_route = lin(raw_best, -1.0, 512.0)
    needs_charge = vclip(lin(
        va(remaining_route, lin(budget, -1.0)), 1.0, 1.0
    ))
    next_costs = [player_goal_cost] + player_costs
    next_target_cost = vm(winners[0], next_costs[0])
    for i in range(1, 4):
        next_target_cost = va(
            next_target_cost, vm(winners[i], next_costs[i])
        )

    # Prefer a battery before a distant waypoint, not before a short reentry.
    battery_before_waypoint = vclip(lin(
        va(player_battery_cost, lin(next_target_cost, -1.0)),
        2.0, -0.5,
    ))
    spare = va(budget, player_battery_cost)
    urgent = vclip(lin(spare, -2.0, 1.0))
    reachable = vclip(lin(spare, 2.0, 1.0))
    charge_reason = vclip(va(battery_before_waypoint, urgent))
    new_charge = vm(vm(needs_charge, charge_reason), reachable)

    recharged = vclip(lin(
        va(steps, lin(previous_steps, -1.0)), 1.0, -1.0
    ))
    prior_charge = vsl("mem", 101, 102)
    retained_charge = vm(
        vm(prior_charge, lin(recharged, -1.0, 1.0)),
        reachable,
    )
    choose_battery = vm(
        vclip(va(new_charge, retained_charge)), gm(batteries)
    )
    no_battery = lin(choose_battery, -1.0, 1.0)
    normal_choice = vm(lin(fallback, -1.0, 1.0), no_battery)

    target_fields = [fields[3], fields[0], fields[1], fields[2]]
    navigation = mul(target_fields[0], vm(winners[0], normal_choice))
    for i in range(1, 4):
        navigation = add(
            navigation, mul(target_fields[i], vm(winners[i], normal_choice))
        )
    navigation = add(navigation, mul(fields[5], vm(fallback, no_battery)))
    navigation = add(navigation, mul(fields[4], choose_battery))
    movement_scores = gm(mul(cat(*neighbours(navigation)), player_mask))

    action_w = np.zeros((8, 4), np.float32)
    action_b = np.full(8, -1000000.0, np.float32)
    for d in range(4):
        action_w[1 + d, d] = 1.0
        action_b[1 + d] = 0.0
    graph.append({
        "id": "logits", "op": "dense", "in": [movement_scores],
        "w": par(action_w), "b": par(action_b),
    })
    graph.append({
        "id": "click", "op": "conv", "in": [one_s],
        "w": par(np.zeros((1, 1, 1, 1), np.float32)),
    })
    graph.append({
        "id": "mem_next", "op": "vconcat",
        "in": [next_cells, steps, choose_battery],
    })
    return {"graph": graph, "memory_size": 102, "params": params}

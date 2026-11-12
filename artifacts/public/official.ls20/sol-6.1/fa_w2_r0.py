import numpy as np


def build():
    P = {}
    graph = []
    active = graph
    serial = [0]
    parameter_cache = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = fresh("p")
            parameter_cache[key] = name
            P[name] = a
        return parameter_cache[key]

    def node(op, inputs, **fields):
        name = fresh()
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        active.append(d)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def spatial_linear(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
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

    def relu(a):
        return node("relu", [a])

    def vr(a):
        return node("vrelu", [a])

    def neg(a, size):
        return dense(a, -np.eye(size, dtype=np.float32))

    def vsub(a, b, size):
        return va(a, neg(b, size))

    def vmax(a, b, size):
        return va(vr(vsub(a, b, size)), b)

    def vmax_many(xs, size):
        z = xs[0]
        for x in xs[1:]:
            z = vmax(z, x, size)
        return z

    def constant(value):
        return dense(
            "aprev",
            np.zeros((1, 8), np.float32),
            np.array([value], np.float32),
        )

    zero = constant(0)
    one = constant(1)
    two = constant(2)
    four = constant(4)
    ten = constant(10)
    c200 = constant(200)
    cm200 = constant(-200)
    cm2 = constant(-2)
    c4096 = constant(4096)
    c65 = constant(65)
    c150 = constant(150)

    def cap(x):
        return node("min2", [relu(x), one])

    def vcap(x):
        z = vr(x)
        return vsub(z, vr(va(z, constant_minus_one)), 1)

    constant_minus_one = constant(-1)

    def binary_above(x, threshold, slope=4.0):
        z = spatial_linear(
            x, [[slope]], [(-threshold) * slope]
        )
        return cap(z)

    def masked_pool(x, mask):
        penalty = spatial_linear(mask, [[1000.0]], [-1000.0])
        return node("gmax", [add(x, penalty)])

    def read_cells(x):
        z = mul(cell_masks, x)
        z = node("gmean", [z])
        return dense(z, np.eye(110, dtype=np.float32) * 4096.0)

    # Exact, saturated coordinate masks for the 10-by-11 playable lattice.
    rows = list(range(7, 58, 5))
    cols = list(range(11, 57, 5))
    row_values = rows + [61]
    col_values = cols + [5]
    nc = len(row_values) + len(col_values)

    w = np.zeros((nc, 3, 1, 1), np.float32)
    b = np.zeros(nc, np.float32)
    for i, r in enumerate(row_values):
        w[i, 0, 0, 0] = 63
        b[i] = -r
    for j, c in enumerate(col_values):
        k = len(row_values) + j
        w[k, 1, 0, 0] = 63
        b[k] = -c
    coordinate_difference = conv("coords", w, b)
    coordinate_abs = node("abs", [coordinate_difference])
    coordinate_tents = cap(sub(one, coordinate_abs))

    cell_w = np.zeros((110, nc), np.float32)
    for i in range(11):
        for j in range(10):
            q = i * 10 + j
            cell_w[q, i] = 4
            cell_w[q, len(row_values) + j] = 4
    cell_masks = cap(
        spatial_linear(coordinate_tents, cell_w, np.full(110, -6))
    )
    board = spatial_linear(cell_masks, np.ones((1, 110), np.float32))

    ui_w = np.zeros((1, nc), np.float32)
    ui_w[0, 10] = 4
    ui_w[0, len(row_values) + 10] = 4
    ui_point = cap(spatial_linear(coordinate_tents, ui_w, [-6]))

    timer_row = cap(mul(sl(coordinate_tents, 11, 12), two))

    old_codes = vs("mem", 0, 110)
    old_visits = vs("mem", 110, 220)
    old_steps = vs("mem", 220, 221)
    old_decrement = vs("mem", 221, 222)

    old_code_map = spatial_linear(
        mul(cell_masks, old_codes), np.ones((1, 110), np.float32)
    )
    visit_map = spatial_linear(
        mul(cell_masks, old_visits), np.ones((1, 110), np.float32)
    )

    # Palette projections.
    palette_order = [12, 9, 14, 8]
    cw = np.zeros((4, 16, 1, 1), np.float32)
    for i, c in enumerate(palette_order):
        cw[i, c, 0, 0] = 1
    colours = conv("x0", cw)
    ink = spatial_linear(colours, np.ones((1, 4), np.float32))
    blue = sl("x0", 1, 2)
    wall_colour = sl("x0", 4, 5)
    black = sl("x0", 0, 1)
    fog_colour = sl("x0", 5, 6)
    recharge_colour = sl("x0", 11, 12)

    # Player centre: the fixed 5x5, two-colour actor rectangle.
    pw = np.zeros((1, 16, 5, 5), np.float32)
    pw[0, 12, :2, :] = 2
    pw[0, 9, 2:, :] = 2
    player = mul(relu(conv("x0", pw, [-49])), board)

    # Shape templates, ordered by (shape, clockwise rotation).
    bases = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]
    patterns = [
        np.rot90(np.asarray(base, np.float32), -r)
        for base in bases
        for r in range(4)
    ]

    gw = np.zeros((24, 1, 3, 3), np.float32)
    uw = np.zeros((24, 1, 5, 5), np.float32)
    tb = np.zeros(24, np.float32)
    for i, pattern in enumerate(patterns):
        signed = 4 * pattern - 2
        gw[i, 0] = signed
        for r in range(3):
            for c in range(3):
                uw[i, 0, 2 * r, 2 * c] = signed[r, c]
        tb[i] = 1 - 2 * pattern.sum()

    goal_orientations = mul(relu(conv(ink, gw, tb)), board)
    goal_observed = spatial_linear(
        goal_orientations, np.ones((1, 24), np.float32)
    )

    current_orientations = node(
        "gmax", [mul(relu(conv(ink, uw, tb)), ui_point)]
    )

    colour_samples = np.zeros((4, 4, 5, 5), np.float32)
    for i in range(4):
        for r in (0, 2, 4):
            for c in (0, 2, 4):
                colour_samples[i, i, r, c] = 1
    current_colours = node(
        "gmax",
        [mul(node("min2", [conv(colours, colour_samples), one]), ui_point)],
    )

    expand_orientation = np.zeros((96, 24), np.float32)
    expand_colour = np.zeros((96, 4), np.float32)
    for orient in range(24):
        for colour in range(4):
            q = 4 * orient + colour
            expand_orientation[q, orient] = 1
            expand_colour[q, colour] = 1
    current_state = vm(
        dense(current_orientations, expand_orientation),
        dense(current_colours, expand_colour),
    )

    goal_colour_w = np.zeros((4, 4, 3, 3), np.float32)
    for i in range(4):
        goal_colour_w[i, i] = 1
    goal_colour_presence = node(
        "min2", [conv(colours, goal_colour_w), one]
    )
    goal_colour_code = spatial_linear(
        goal_colour_presence,
        np.arange(4, dtype=np.float32)[None, :] / 128.0,
    )
    goal_orientation_code = spatial_linear(
        goal_orientations,
        (7.0 + np.arange(24, dtype=np.float32) * 4.0 / 128.0)[None, :],
    )
    goal_observation_code = mul(
        add(goal_orientation_code, goal_colour_code), goal_observed
    )

    # Transformation and refill detectors.
    tw = np.zeros((3, 16, 3, 3), np.float32)
    t_bias = np.zeros(3, np.float32)

    shape_black = [(0, 0), (1, 1), (1, 2), (2, 1)]
    for r, c in shape_black:
        tw[0, 0, r, c] = 2
    t_bias[0] = -7

    colour_pattern = [
        [9, 14, 14],
        [9, 0, 8],
        [12, 12, 8],
    ]
    for r in range(3):
        for c in range(3):
            tw[1, colour_pattern[r][c], r, c] = 2
    t_bias[1] = -17

    for r, c in [(0, 1), (1, 1), (1, 2)]:
        tw[2, 0, r, c] = 2
    for r, c in [(1, 0), (2, 1)]:
        tw[2, 1, r, c] = 2
    t_bias[2] = -9

    observed_transforms = mul(relu(conv("x0", tw, t_bias)), board)

    rw = np.zeros((1, 1, 3, 3), np.float32)
    rw[0, 0] = 2
    rw[0, 0, 1, 1] = -2
    observed_recharge = mul(
        relu(conv(recharge_colour, rw, [-15])), board
    )

    # Pusher direction order agrees with actions 1..4.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    push_w = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(directions):
        if dr:
            for offset in (-1, 0, 1):
                push_w[d, 0, 3 - 3 * dr, 3 + offset] = 2
        else:
            for offset in (-1, 0, 1):
                push_w[d, 0, 3 + offset, 3 - 3 * dc] = 2
    observed_pushers = mul(
        relu(conv(blue, push_w, np.full(4, -5))), board
    )

    # Cell codes:
    # 0 unknown; 1 floor; 2 wall; 3/4/5 transformations; 6 refill;
    # 7 + state/128 goal; 8..11 forced transport direction.
    observed_features = cat(
        wall_colour, observed_transforms, observed_recharge, observed_pushers
    )
    observation_base = spatial_linear(
        observed_features,
        [[1, 2, 3, 4, 5, 7, 8, 9, 10]],
        [1],
    )
    observed_code = add(
        observation_base,
        sub(goal_observation_code, goal_observed),
    )

    observed_known = node(
        "min2", [add(sub(one, fog_colour), goal_observed), one]
    )
    observed_known = mul(
        mul(observed_known, board), sub(one, player)
    )
    old_unknown = sub(one, binary_above(old_code_map, 0.5, 2))
    code_map = add(
        add(
            old_code_map,
            mul(sub(observed_code, old_code_map), observed_known),
        ),
        mul(player, old_unknown),
    )

    new_codes = read_cells(code_map)
    new_visits = va(old_visits, read_cells(player))

    # Decode remembered tile kinds.
    type_values = np.array([2, 3, 4, 5, 6, 8, 9, 10, 11], np.float32)
    type_diff = spatial_linear(
        code_map, np.ones((9, 1), np.float32), -type_values
    )
    type_masks = relu(sub(one, mul(node("abs", [type_diff]), ten)))
    wall = sl(type_masks, 0, 1)
    transforms = [sl(type_masks, i, i + 1) for i in range(1, 4)]
    refill = sl(type_masks, 4, 5)
    pushers = sl(type_masks, 5, 9)
    push_union = spatial_linear(pushers, np.ones((1, 4), np.float32))

    goal_low = binary_above(code_map, 6.75, 4)
    goal_high = cap(spatial_linear(code_map, [[-128]], [7.75 * 128]))
    goals = mul(goal_low, goal_high)

    known = binary_above(code_map, 0.5, 2)
    passable = mul(mul(board, known), sub(one, wall))

    # Split the two visible/remembered goals deterministically.
    rank = spatial_linear("coords", [[-4032, -63, 0]])
    best_rank = node(
        "gmax",
        [add(rank, spatial_linear(goals, [[10000]], [-10000]))],
    )
    rank_error = node("abs", [sub(rank, best_rank)])
    first_goal = mul(
        cap(sub(two, mul(rank_error, four))), goals
    )
    second_goal = sub(goals, first_goal)
    goal_masks = [first_goal, second_goal]

    goal_states = []
    goal_seeds = []
    goal_indices = 7.0 + np.arange(96, dtype=np.float32) / 128.0
    for gm in goal_masks:
        gc = masked_pool(code_map, gm)
        positive = vr(
            dense(
                gc,
                np.full((96, 1), 256, np.float32),
                -256 * goal_indices,
            )
        )
        negative = vr(
            dense(
                gc,
                np.full((96, 1), -256, np.float32),
                256 * goal_indices,
            )
        )
        absolute = va(positive, negative)
        unit = vr(
            dense(
                absolute,
                -np.eye(96, dtype=np.float32),
                np.ones(96, np.float32),
            )
        )
        goal_states.append(unit)
        goal_seeds.append(va(vm(unit, c200), cm200))

    # Two small convolutions implement exact five-pixel cardinal shifts.
    def shift_kernel(channels, distance, direction, outputs=None):
        if outputs is None:
            outputs = channels
        k = 5 if distance == 2 else 7
        w = np.zeros((outputs, channels, k, k), np.float32)
        dr, dc = direction
        for i in range(min(outputs, channels)):
            w[i, i, k // 2 + dr * distance, k // 2 + dc * distance] = 1
        return w

    def shift5(x, channels, direction):
        return conv(
            conv(x, shift_kernel(channels, 2, direction)),
            shift_kernel(channels, 3, direction),
        )

    # Locate the landing cell of each forced transport lane.
    wall_and_border = add(wall, sub(one, board))
    neighbour_wall_w = np.zeros((4, 1, 5, 5), np.float32)
    neighbour_wall_w2 = np.zeros((4, 4, 7, 7), np.float32)
    lane_w1 = np.zeros((4, 4, 5, 5), np.float32)
    lane_w2 = np.zeros((4, 4, 7, 7), np.float32)
    for i, (dr, dc) in enumerate(directions):
        neighbour_wall_w[i, 0, 2 + 2 * dr, 2 + 2 * dc] = 1
        neighbour_wall_w2[i, i, 3 + 3 * dr, 3 + 3 * dc] = 1
        lane_w1[i, i, 2 + 2 * dr, 2 + 2 * dc] = 1
        lane_w2[i, i, 3 + 3 * dr, 3 + 3 * dc] = 1

    neighbour_wall = conv(
        conv(wall_and_border, neighbour_wall_w), neighbour_wall_w2
    )
    cell_ids = spatial_linear(
        cell_masks, np.arange(1, 111, dtype=np.float32)[None, :]
    )
    terminal_lane = mul(mul(cell_ids, neighbour_wall), passable)

    lane_state = fresh("lane_state")
    lane_loop = fresh("lane_loop")
    lane_body = []
    saved_active = active
    active = lane_body
    lane_shifted = conv(conv(lane_state, lane_w1), lane_w2)
    lane_next = mul(
        node("max2", [lane_shifted, terminal_lane]), passable
    )
    active = saved_active
    active.append({
        "id": lane_loop,
        "op": "loop",
        "T": 11,
        "state": [{"id": lane_state, "init": terminal_lane}],
        "body": lane_body,
        "next": {lane_state: lane_next},
    })
    lane_result = lane_loop + "." + lane_state
    landing_id = spatial_linear(
        mul(lane_result, pushers), np.ones((1, 4), np.float32)
    )
    landing_difference = spatial_linear(
        landing_id,
        np.ones((110, 1), np.float32),
        -np.arange(1, 111, dtype=np.float32),
    )
    landing_selector = mul(
        relu(sub(one, node("abs", [landing_difference]))),
        push_union,
    )

    # Exploration targets: least-visited known frontier cells.
    unknown = mul(board, sub(one, known))
    unknown_neighbours = [
        shift5(unknown, 1, d) for d in directions
    ]
    adjacent_unknown = spatial_linear(
        cat(*unknown_neighbours), np.ones((1, 4), np.float32)
    )
    frontier = mul(passable, node("min2", [adjacent_unknown, one]))
    frontier_rank = add(
        spatial_linear(visit_map, [[-1]]),
        spatial_linear(frontier, [[10000]], [-10000]),
    )
    frontier_best = node("gmax", [frontier_rank])
    frontier_error = node("abs", [sub(frontier_rank, frontier_best)])
    exploration = mul(
        relu(sub(one, frontier_error)), frontier
    )

    # Sources: shape, colour, rotation, two goals, refill, exploration.
    sources_list = transforms + goal_masks + [refill, exploration]
    sources = cat(*sources_list)
    checkpoints = spatial_linear(
        cat(*(transforms + goal_masks)),
        np.ones((1, 5), np.float32),
    )
    relay_common = sub(one, checkpoints)
    relay = node(
        "min2",
        [add(cat(*([relay_common] * 7)), sources), one],
    )

    # Cardinal, lattice-to-lattice diffusion. The composed support includes
    # five-pixel cardinal neighbours but no five-pixel diagonal neighbours.
    cross2 = np.zeros((1, 1, 5, 5), np.float32)
    cross3 = np.zeros((1, 1, 7, 7), np.float32)
    cross2[0, 0, 2, 2] = 1
    cross3[0, 0, 3, 3] = 1
    for dr, dc in directions:
        cross2[0, 0, 2 + 2 * dr, 2 + 2 * dc] = 1
        cross3[0, 0, 3 + 3 * dr, 3 + 3 * dc] = 1

    reached_state = fresh("reached")
    accumulated_state = fresh("accumulated")
    bfs_loop = fresh("routing")
    bfs_body = []
    saved_active = active
    active = bfs_body

    relayed = mul(reached_state, relay)
    ordinary_parts = []
    transport_parts = []
    for i in range(7):
        channel = sl(relayed, i, i + 1)
        ordinary_parts.append(conv(conv(channel, cross2), cross3))

        sampled = node("gmean", [mul(cell_masks, channel)])
        sampled = dense(
            sampled, np.eye(110, dtype=np.float32) * 4096
        )
        transported = spatial_linear(
            mul(landing_selector, sampled),
            np.ones((1, 110), np.float32),
        )
        transport_parts.append(transported)

    ordinary = node("min2", [cat(*ordinary_parts), one])
    received = add(
        mul(ordinary, sub(one, push_union)),
        cat(*transport_parts),
    )
    reached_next = node(
        "max2",
        [mul(node("min2", [received, one]), passable), sources],
    )
    accumulated_next = add(accumulated_state, reached_next)

    active = saved_active
    active.append({
        "id": bfs_loop,
        "op": "loop",
        "T": 64,
        "state": [
            {"id": reached_state, "init": sources},
            {"id": accumulated_state, "init": sources},
        ],
        "body": bfs_body,
        "next": {
            reached_state: reached_next,
            accumulated_state: accumulated_next,
        },
    })

    reached = bfs_loop + "." + reached_state
    accumulated = bfs_loop + "." + accumulated_state
    distance_values = sub(
        sub(accumulated, c65),
        mul(sub(one, reached), c200),
    )

    pair_costs = [masked_pool(distance_values, t) for t in transforms]
    player_costs = masked_pool(distance_values, player)

    # Abstract dynamic programming over the 96 observable attribute states.
    # State index = ((shape * 4 + rotation) * 4 + colour).
    permutation_weights = []
    for kind in range(3):
        w = np.zeros((96, 96), np.float32)
        for s in range(6):
            for r in range(4):
                for c in range(4):
                    q = (s * 4 + r) * 4 + c
                    ns, nr, nc_ = s, r, c
                    if kind == 0:
                        ns = (s + 1) % 6
                    elif kind == 1:
                        nc_ = (c + 1) % 4
                    else:
                        nr = (r + 1) % 4
                    nq = (ns * 4 + nr) * 4 + nc_
                    w[q, nq] = 1
        permutation_weights.append(w)

    goal_terms = []
    macro_costs = []
    for i in range(3):
        cost = pair_costs[i]
        goal_terms.append(vmax(
            va(goal_seeds[0], vs(cost, 3, 4)),
            va(goal_seeds[1], vs(cost, 4, 5)),
            96,
        ))
        macro_costs.append([
            cm2 if i == j else vs(cost, j, j + 1)
            for j in range(3)
        ])

    macro_states = [fresh("macro_state") for _ in range(3)]
    macro_loop = fresh("attribute_plan")
    macro_body = []
    saved_active = active
    active = macro_body

    after_events = [
        dense(macro_states[j], permutation_weights[j])
        for j in range(3)
    ]
    macro_next = []
    for i in range(3):
        options = [goal_terms[i]] + [
            va(after_events[j], macro_costs[i][j])
            for j in range(3)
        ]
        macro_next.append(vmax_many(options, 96))

    active = saved_active
    active.append({
        "id": macro_loop,
        "op": "loop",
        "T": 20,
        "state": [
            {"id": macro_states[i], "init": goal_terms[i]}
            for i in range(3)
        ],
        "body": macro_body,
        "next": {
            macro_states[i]: macro_next[i] for i in range(3)
        },
    })

    def dot_state(v):
        return dense(
            vm(v, current_state), np.ones((1, 96), np.float32)
        )

    bonuses = [
        dot_state(dense(
            macro_loop + "." + macro_states[j],
            permutation_weights[j],
        ))
        for j in range(3)
    ] + [dot_state(s) for s in goal_seeds]

    complete_options = [
        va(bonuses[j], vs(player_costs, j, j + 1))
        for j in range(5)
    ]
    complete_value = vmax_many(complete_options, 1)
    exploration_gate = vcap(
        vsub(neg(complete_value, 1), c150, 1)
    )

    # Read the timer and infer whether the level spends one or two units.
    timer_pixels = mul(recharge_colour, timer_row)
    steps = vm(node("gmean", [timer_pixels]), c4096)
    spent = vr(vsub(old_steps, steps, 1))
    decrement = vmax(vmax(old_decrement, spent, 1), one, 1)
    decrement = vsub(
        decrement, vr(vsub(decrement, two, 1)), 1
    )

    refill_distance = neg(vs(player_costs, 5, 6), 1)
    refill_need = vsub(
        va(vm(decrement, va(refill_distance, two)), four),
        steps,
        1,
    )
    refill_gate = vm(
        vcap(refill_need), node("gmax", [refill])
    )

    checkpoint_maps = cat(*(transforms + goal_masks))
    action_scores = []
    for direction in directions:
        neighbour_values = masked_pool(
            shift5(distance_values, 7, direction), player
        )
        neighbour_flags = masked_pool(
            shift5(checkpoint_maps, 5, direction), player
        )

        route_options = [
            va(bonuses[j], vs(neighbour_values, j, j + 1))
            for j in range(5)
        ]
        normal_score = vmax_many(route_options, 1)

        flags = [vs(neighbour_flags, j, j + 1) for j in range(5)]
        flag_sum = flags[0]
        event_score = vm(flags[0], bonuses[0])
        for j in range(1, 5):
            flag_sum = va(flag_sum, flags[j])
            event_score = va(event_score, vm(flags[j], bonuses[j]))

        normal_score = va(
            vm(normal_score, vsub(one, flag_sum, 1)),
            event_score,
        )
        explore_score = vs(neighbour_values, 6, 7)
        score = va(
            vm(normal_score, vsub(one, exploration_gate, 1)),
            vm(explore_score, exploration_gate),
        )
        refill_score = vs(neighbour_values, 5, 6)
        score = va(
            vm(score, vsub(one, refill_gate, 1)),
            vm(refill_score, refill_gate),
        )

        # A mismatched goal physically blocks entry, even while exploring.
        goal_block = zero
        for j in range(2):
            mismatch = vcap(neg(bonuses[3 + j], 1))
            goal_block = va(
                goal_block, vm(flags[3 + j], mismatch)
            )
        score = va(score, vm(goal_block, constant(-1000)))
        action_scores.append(score)

    action_vector = vcat(*action_scores)
    output_w = np.zeros((8, 4), np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1
    output_b = np.full(8, -10000, np.float32)
    output_b[1:5] = 0
    logits_value = dense(action_vector, output_w, output_b)
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [logits_value],
        "w": parameter(np.eye(8, dtype=np.float32)),
    })

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [new_codes, new_visits, steps, decrement],
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": ["coords"],
        "w": parameter(np.zeros((1, 3, 1, 1), np.float32)),
    })

    return {
        "graph": graph,
        "memory_size": 222,
        "params": P,
    }

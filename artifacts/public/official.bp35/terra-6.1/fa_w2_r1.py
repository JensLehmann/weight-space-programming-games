import numpy as np


def build():
    graph = []
    params = {}
    counter = 0
    pcounter = 0

    def name(prefix="n"):
        nonlocal counter
        result = "%s%d" % (prefix, counter)
        counter += 1
        return result

    def param(value):
        nonlocal pcounter
        result = "p%d" % pcounter
        pcounter += 1
        params[result] = np.asarray(value, dtype=np.float32)
        return result

    def node(op, inputs, **fields):
        result = name()
        item = {"id": result, "op": op, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return result

    def conv(x, w, bias=None):
        fields = {"w": w}
        if bias is not None:
            fields["b"] = bias
        return node("conv", [x], **fields)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def spatial_linear(x, weight, bias=0.0):
        w = param(np.array([[[[weight]]]], dtype=np.float32))
        return conv(x, w, param([bias]))

    def vector_linear(x, weight, bias=0.0):
        return node(
            "dense",
            [x],
            w=param([[weight]]),
            b=param([bias]),
        )

    def spatial_clip01(x):
        x = unary("relu", x)
        excess = unary("relu", spatial_linear(x, 1.0, -1.0))
        return binary("sub", x, excess)

    def vector_clip01(x):
        x = unary("vrelu", x)
        excess = unary("vrelu", vector_linear(x, 1.0, -1.0))
        return binary("vadd", x, vector_linear(excess, -1.0))

    def color(c):
        return sl("x0", c, c + 1)

    def kernel(dy, dx):
        w = np.zeros((1, 1, 7, 7), dtype=np.float32)
        w[0, 0, 3 + dy, 3 + dx] = 1.0
        return param(w)

    kup = kernel(-3, 0)
    kdown = kernel(3, 0)
    kleft = kernel(0, -3)
    kright = kernel(0, 3)

    def shift6(x, w):
        return conv(conv(x, w), w)

    offsets = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    first_shift = np.zeros((4, 1, 7, 7), dtype=np.float32)
    sum_shift = np.zeros((1, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        first_shift[i, 0, 3 + dy, 3 + dx] = 1.0
        sum_shift[0, i, 3 + dy, 3 + dx] = 1.0
    first_shift = param(first_shift)
    sum_shift = param(sum_shift)

    one = sl("coords", 2, 3)
    row = sl("coords", 0, 1)

    # Player center for both possible facing directions.
    wp = np.zeros((2, 16, 7, 7), dtype=np.float32)
    for dy in (-1, 0):
        for dx in (-1, 0):
            wp[0, 9, 3 + dy, 3 + dx] = 1.0
        for dx in (0, 1):
            wp[1, 9, 3 + dy, 3 + dx] = 1.0
    wp[0, 11, 3, 4] = 1.0
    wp[1, 11, 3, 2] = 1.0
    player_parts = unary(
        "relu",
        conv("x0", param(wp), param([-4.5, -4.5])),
    )
    merge = np.zeros((1, 2, 1, 1), dtype=np.float32)
    merge[0, :, 0, 0] = 2.0
    player = conv(player_parts, param(merge))

    # Camera alignment distinguishes the current gravity direction.
    player_row = unary("gmax", binary("mul", player, row))
    gravity_down = vector_clip01(
        vector_linear(player_row, -63.0 / 5.0, 34.0 / 5.0)
    )
    gravity_up = vector_linear(gravity_down, -1.0, 1.0)

    # Generate all visible tile centers from the player's lattice point.
    graph.append({
        "id": "LATTICE",
        "op": "loop",
        "T": 20,
        "state": [{"id": "lattice_state", "init": player}],
        "body": [
            {
                "id": "lattice_first",
                "op": "conv",
                "in": ["lattice_state"],
                "w": first_shift,
            },
            {
                "id": "lattice_second",
                "op": "conv",
                "in": ["lattice_first"],
                "w": sum_shift,
            },
            {
                "id": "lattice_join",
                "op": "max2",
                "in": ["lattice_state", "lattice_second"],
            },
            {
                "id": "lattice_next",
                "op": "min2",
                "in": ["lattice_join", one],
            },
        ],
        "next": {"lattice_state": "lattice_next"},
    })
    lattice = "LATTICE.lattice_state"

    # The bottom row is an action counter, not game geometry.
    board_rows = spatial_clip01(spatial_linear(row, -63.0, 61.5))
    lattice = binary("mul", lattice, board_rows)

    air = color(10)
    player_color = color(9)
    normal = color(14)
    purple = color(12)
    orange_or_spike = color(15)
    goal_color = color(7)
    green_color = color(8)
    black = color(0)

    # Spike centers have a black pixel two rows above or below.
    black_near = binary(
        "max2",
        conv(black, kernel(-2, 0)),
        conv(black, kernel(2, 0)),
    )
    spike = binary("min2", orange_or_spike, black_near)
    orange = binary("sub", orange_or_spike, spike)

    # Solid purple has purple immediately above its center; the
    # translucent form has a transparent pixel there.
    purple_solid = binary(
        "min2",
        purple,
        conv(purple, kernel(-1, 0)),
    )
    purple_ghost = binary("sub", purple, purple_solid)

    solid = binary("add", normal, purple_solid)
    solid = binary("add", solid, orange)
    solid = binary("mul", solid, lattice)

    free = binary("add", air, player_color)
    free = binary("add", free, purple_ghost)
    free = binary("add", free, goal_color)
    free = binary("min2", free, one)
    free = binary("mul", free, lattice)

    goal = binary("mul", goal_color, lattice)
    green = binary("mul", green_color, lattice)
    spike = binary("mul", spike, lattice)
    valid = binary("min2", binary("add", free, solid), one)

    # A spike or goal is also an unsupported cell: entering that shaft
    # forces a fall rather than permitting a sideways action.
    fall_through = binary("min2", binary("add", free, spike), one)
    fall_up = shift6(fall_through, kup)
    fall_down = shift6(fall_through, kdown)
    support_up = binary("sub", one, fall_up)
    support_down = binary("sub", one, fall_down)

    green_exists = unary("gmax", green)
    solid_cost = solid

    # The visible gravity-facing edge is an exploration target.
    # A visible goal is much more valuable.
    top_edge = spatial_clip01(spatial_linear(row, -63.0, 5.0))
    bottom_edge = spatial_clip01(spatial_linear(row, 63.0, -57.0))
    up_source = binary(
        "max2",
        spatial_linear(top_edge, 70.0),
        spatial_linear(goal, 150.0),
    )
    down_source = binary(
        "max2",
        spatial_linear(bottom_edge, 55.0),
        spatial_linear(goal, 150.0),
    )
    up_source = binary("mul", up_source, valid)
    down_source = binary("mul", down_source, valid)

    # Both value fields are updated simultaneously. Entering a solid
    # costs a click; horizontal movement costs a step. A virtual cell
    # may move sideways only if supported in that gravity direction.
    def paired_value_loop(loop_id, up_seed, down_seed, iterations=56):
        body = []
        up_state = name("state")
        down_state = name("state")

        def bn(op, inputs, **fields):
            result = name("body")
            item = {"id": result, "op": op, "in": list(inputs)}
            item.update(fields)
            body.append(item)
            return result

        def update(state, support, seed, vertical_index):
            after_cost = bn("sub", [state, solid_cost])
            enter = bn("relu", [after_cost])
            enter = bn("mul", [enter, valid])

            first = bn("conv", [enter], w=first_shift)
            shifted = []
            for i, w in enumerate((kup, kdown, kleft, kright)):
                part = bn("slice", [first], start=i, stop=i + 1)
                shifted.append(bn("conv", [part], w=w))

            horizontal = bn("max2", [shifted[2], shifted[3]])
            horizontal = bn("sub", [horizontal, one])
            horizontal = bn("relu", [horizontal])
            horizontal = bn("mul", [horizontal, support])
            best = bn("max2", [horizontal, shifted[vertical_index]])
            best = bn("max2", [best, seed])
            return bn("mul", [best, valid])

        up_next = update(up_state, support_up, up_seed, 0)
        down_next = update(down_state, support_down, down_seed, 1)
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [
                {"id": up_state, "init": up_seed},
                {"id": down_state, "init": down_seed},
            ],
            "body": body,
            "next": {
                up_state: up_next,
                down_state: down_next,
            },
        })
        return loop_id + "." + up_state, loop_id + "." + down_state

    base_up, base_down = paired_value_loop(
        "BASE_VALUES", up_source, down_source
    )

    # One switch can be used from a supported cell. The opposite field
    # already accounts for the forced fall after the switch.
    up_switch = unary("relu", spatial_linear(base_down, 1.0, -2.0))
    up_switch = binary("mul", up_switch, support_up)
    up_switch = binary("mul", up_switch, green_exists)
    up_switch = binary("mul", up_switch, valid)

    down_switch = unary("relu", spatial_linear(base_up, 1.0, -2.0))
    down_switch = binary("mul", down_switch, support_down)
    down_switch = binary("mul", down_switch, green_exists)
    down_switch = binary("mul", down_switch, valid)

    final_up_seed = binary("max2", up_source, up_switch)
    final_down_seed = binary("max2", down_source, down_switch)
    value_up, value_down = paired_value_loop(
        "FINAL_VALUES", final_up_seed, final_down_seed
    )

    current_value = binary(
        "add",
        binary("mul", value_up, gravity_up),
        binary("mul", value_down, gravity_down),
    )
    opposite_base = binary(
        "add",
        binary("mul", base_down, gravity_up),
        binary("mul", base_up, gravity_down),
    )

    # Value after opening a tile, if it is solid.
    enter_value = unary("relu", binary("sub", current_value, solid_cost))
    enter_value = binary("mul", enter_value, valid)
    side_value = unary("relu", spatial_linear(enter_value, 1.0, -1.0))

    # Masks on the four neighboring tile centers.
    player_left = shift6(player, kright)
    player_right = shift6(player, kleft)
    player_above = shift6(player, kdown)
    player_below = shift6(player, kup)
    player_forward = binary(
        "add",
        binary("mul", player_above, gravity_up),
        binary("mul", player_below, gravity_down),
    )

    left_move_mask = binary("mul", player_left, free)
    right_move_mask = binary("mul", player_right, free)
    left_score = unary("gmax", binary("mul", left_move_mask, side_value))
    right_score = unary("gmax", binary("mul", right_move_mask, side_value))
    left_exists = unary("gmax", left_move_mask)
    right_exists = unary("gmax", right_move_mask)

    # Click a solid neighbor, or click a switch when the opposite-gravity
    # plan is more valuable. The spatial scores also form the click map.
    side_neighbors = binary("max2", player_left, player_right)
    side_solid = binary("mul", side_neighbors, solid)
    side_click = binary("mul", side_solid, side_value)

    forward_solid = binary("mul", player_forward, solid)
    forward_click = binary("mul", forward_solid, enter_value)
    block_click = binary("max2", side_click, forward_click)

    switch_score_map = unary(
        "relu",
        spatial_linear(opposite_base, 1.0, -2.0),
    )
    switch_score = unary(
        "gmax",
        binary("mul", switch_score_map, player),
    )
    switch_click = binary("mul", green, switch_score)
    graph.append({
        "id": "click",
        "op": "max2",
        "in": [block_click, switch_click],
    })
    click_score = unary("gmax", "click")
    click_exists = vector_clip01(vector_linear(click_score, 10.0))

    features = node(
        "vconcat",
        [
            left_score,
            right_score,
            click_score,
            left_exists,
            right_exists,
            click_exists,
            "aprev",
        ],
    )

    weights = np.zeros((8, 14), dtype=np.float32)
    bias = np.full(8, -10000.0, dtype=np.float32)

    weights[3, 0] = 1.0
    weights[3, 3] = 100.0
    weights[3, 6 + 4] = -0.1
    bias[3] = -100.0

    weights[4, 1] = 1.0
    weights[4, 4] = 100.0
    weights[4, 6 + 3] = -0.1
    bias[4] = -100.0

    weights[6, 2] = 1.0
    weights[6, 5] = 100.0
    bias[6] = -100.0

    # Undo is a fallback if no visible move or useful click is available.
    bias[7] = -1.0

    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": param(weights),
        "b": param(bias),
    })

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

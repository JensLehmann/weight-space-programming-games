import numpy as np


def build():
    graph = []
    params = {}
    channels = {}
    serial = [0]
    parameter_serial = [0]
    kernel_cache = {}
    shift_cache = {}
    nodes = graph

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        parameter_serial[0] += 1
        name = "p" + str(parameter_serial[0])
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, C=None, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        nodes.append(nd)
        if C is None:
            C = max(channels[x] for x in inputs)
        channels[name] = C
        return name

    channels.update(x0=16, coords=3, avail=8, aprev=8)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], int(w.shape[0]), name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], int(w.shape[0]), name, **fields)

    def affine(x, a=1.0, b=0.0):
        C = channels[x]
        w = np.zeros((C, C, 1, 1), np.float32)
        for i in range(C):
            w[i, i, 0, 0] = a
        return conv(x, w, np.full(C, b, np.float32))

    def vaffine(x, a=1.0, b=0.0):
        C = channels[x]
        return dense(
            x,
            np.eye(C, dtype=np.float32) * a,
            np.full(C, b, np.float32),
        )

    def add(x, y):
        return node("add", [x, y])

    def sub(x, y):
        return node("sub", [x, y])

    def mul(x, y):
        return node("mul", [x, y])

    def maximum(x, y):
        return node("max2", [x, y])

    def minimum(x, y):
        return node("min2", [x, y])

    def absolute(x):
        return node("abs", [x])

    def gmax(x):
        return node("gmax", [x], channels[x])

    def plane(image, c):
        return node("slice", [image], 1, start=c, stop=c + 1)

    def palette(image, colours):
        w = np.zeros((4, 16, 1, 1), np.float32)
        for c in colours:
            w[:, c, 0, 0] = 1.0
        return conv(image, w)

    # Banks are down, up, right, left.
    # A shift reads the input at the stated offset from its output cell.
    def unit_shift(x, axis, sign):
        C = channels[x]
        key = (C, axis, sign)
        if key not in kernel_cache:
            w = np.zeros((C, C, 3, 3), np.float32)
            for c in range(C):
                d = c % 4
                if axis == "g":
                    offsets = (
                        (sign, 0),
                        (-sign, 0),
                        (0, sign),
                        (0, -sign),
                    )
                elif axis == "t":
                    offsets = (
                        (0, sign),
                        (0, sign),
                        (sign, 0),
                        (sign, 0),
                    )
                elif axis == "row":
                    offsets = ((sign, 0),) * 4
                else:
                    raise ValueError(axis)
                dr, dc = offsets[d]
                w[c, c, 1 + dr, 1 + dc] = 1.0
            kernel_cache[key] = parameter(w)
        return node("conv", [x], C, w=kernel_cache[key])

    def shift(x, dg=0, dt=0):
        key = (x, dg, dt)
        if key in shift_cache:
            return shift_cache[key]
        if dg:
            s = 1 if dg > 0 else -1
            previous = shift(x, dg - s, dt)
            result = unit_shift(previous, "g", s)
        elif dt:
            s = 1 if dt > 0 else -1
            previous = shift(x, 0, dt - s)
            result = unit_shift(previous, "t", s)
        else:
            result = x
        shift_cache[key] = result
        return result

    def axis_pool(x, axis):
        C = channels[x]
        rm = node("rowmax", [x], C)
        cm = node("colmax", [x], C)
        joined = node("concat", [rm, cm], 2 * C)
        w = np.zeros((C, 2 * C, 1, 1), np.float32)
        for c in range(C):
            vertical = (c % 4) < 2
            use_column = vertical if axis == "g" else not vertical
            w[c, c + (C if use_column else 0), 0, 0] = 1.0
        return conv(joined, w)

    # Repair the UI-overwritten first row.
    w = np.zeros((1, 3, 1, 1), np.float32)
    w[0, 0, 0, 0] = -126.0
    top = node(
        "relu",
        [conv("coords", w, np.ones(1, np.float32))],
        1,
    )
    not_top = affine(top, -1.0, 1.0)
    row_next = unit_shift("x0", "row", 1)
    image = add(mul("x0", not_top), mul(row_next, top))

    w = np.zeros((4, 3, 1, 1), np.float32)
    w[:, 2, 0, 0] = 1.0
    ones = conv("coords", w)
    zero = affine(ones, 0.0, 0.0)

    def clip01(x):
        return minimum(node("relu", [x]), ones)

    def positive(x):
        return clip01(affine(x, 2.0, -0.5))

    def equal(x, y):
        return clip01(affine(absolute(sub(x, y)), -4.0, 2.0))

    def greater(x, y):
        return positive(sub(x, y))

    def ge(x, y):
        return sub(ones, greater(y, x))

    def signed_clip(x):
        return minimum(maximum(x, affine(ones, -1.0)), ones)

    # Longitudinal coordinates increase in the gravity direction.
    w = np.zeros((4, 3, 1, 1), np.float32)
    w[0, 0, 0, 0] = 63.0
    w[1, 0, 0, 0] = -63.0
    w[2, 1, 0, 0] = 63.0
    w[3, 1, 0, 0] = -63.0
    g = conv("coords", w, np.array([0, 63, 0, 63], np.float32))

    w = np.zeros((4, 3, 1, 1), np.float32)
    w[0:2, 1, 0, 0] = 63.0
    w[2:4, 0, 0, 0] = 63.0
    t = conv("coords", w)
    gp1 = affine(g, 1.0, 1.0)
    tp1 = affine(t, 1.0, 1.0)
    upstream_value = affine(g, -1.0, 65.0)

    yellow = palette(image, [4])
    colour11 = palette(image, [11])
    colour14 = palette(image, [14])

    # Yellow crown pixels distinguish player bases from goal markers.
    yellow_up = shift(yellow, dg=-2)
    player11 = mul(colour11, yellow_up)
    player14 = mul(colour14, yellow_up)
    player_base = add(player11, player14)
    orientation = gmax(player_base)
    O = mul(ones, orientation)

    def field(v):
        return mul(O, v)

    def propagate(seed, mask, axis="t"):
        nonlocal nodes
        loop_id = fresh("L")
        state_id = fresh("s")
        channels[state_id] = 4
        old_nodes = nodes
        nodes = []
        if axis == "t":
            a = unit_shift(state_id, "t", -1)
            b = unit_shift(state_id, "t", 1)
            nxt = mul(maximum(maximum(state_id, a), b), mask)
        else:
            a = unit_shift(
                state_id, "g", -1 if axis == "down" else 1
            )
            nxt = mul(maximum(state_id, a), mask)
        body = nodes
        nodes = old_nodes
        nodes.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: nxt},
        })
        result = loop_id + "." + state_id
        channels[result] = 4
        return result

    # Thin longitudinal strips identify walls.
    raw_wall = mul(palette(image, [5, 1, 12]), O)
    wall_seed = mul(
        raw_wall,
        mul(
            sub(O, shift(raw_wall, dt=-4)),
            sub(O, shift(raw_wall, dt=4)),
        ),
    )
    wall_band = clip01(axis_pool(wall_seed, "g"))
    open_lane = sub(O, wall_band)

    bridge_pixels = mul(palette(image, [1, 12]), O)
    bridge_band = clip01(axis_pool(bridge_pixels, "g"))
    floor_pixels = mul(palette(image, [5]), open_lane)

    # Ignore black brackets surrounding enabled bridges.
    cyan = mul(palette(image, [12]), O)
    bracket_near = clip01(add(
        add(shift(cyan, dt=-2), shift(cyan, dt=2)),
        add(shift(cyan, dt=-4), shift(cyan, dt=4)),
    ))
    clean_black = mul(
        mul(palette(image, [0]), open_lane),
        sub(O, bracket_near),
    )

    button_pixels = mul(palette(image, [9]), O)
    button_over_block = clip01(maximum(
        shift(clean_black, dt=-4),
        shift(clean_black, dt=4),
    ))
    black = add(
        clean_black,
        mul(mul(button_pixels, open_lane), button_over_block),
    )

    # Small-board rendering doubles a two-pixel step to four pixels.
    nine = plane(image, 9)
    nine3 = nine
    for _ in range(3):
        nine3 = unit_shift(nine3, "row", 1)
    small_scalar = gmax(mul(nine, nine3))
    small = dense(small_scalar, np.ones((4, 1), np.float32))
    inverse_step = dense(
        small_scalar,
        np.full((4, 1), -1.0 / 12.0, np.float32),
        np.full(4, 1.0 / 3.0, np.float32),
    )
    margin = dense(
        small_scalar,
        np.full((4, 1), 2.0, np.float32),
        np.full(4, 6.0, np.float32),
    )

    # Recover each block's movable surface and fixed endpoint.
    first_black = mul(black, sub(O, shift(black, dg=-1)))
    last_black = mul(black, sub(O, shift(black, dg=1)))
    front_seed = mul(upstream_value, first_black)
    end_seed = mul(gp1, last_black)
    front_value = propagate(front_seed, black, "down")
    block_end = propagate(end_seed, black, "up")
    block_height_coordinate = affine(front_value, -1.0, 65.0)

    player_edge = mul(
        player_base, sub(O, shift(player_base, dg=1))
    )
    player_height_seed = axis_pool(mul(gp1, player_edge), "g")
    player_height_lane = propagate(player_height_seed, open_lane)
    occupied_block = mul(
        black, equal(block_height_coordinate, player_height_lane)
    )

    floor_limit = propagate(mul(gp1, floor_pixels), O, "down")
    wall_front_value = axis_pool(
        mul(upstream_value, mul(raw_wall, wall_band)), "g"
    )
    wall_start = mul(
        affine(wall_front_value, -1.0, 65.0), wall_band
    )
    wall_limit_seed = mul(
        maximum(
            shift(wall_start, dt=-4),
            shift(wall_start, dt=4),
        ),
        open_lane,
    )
    wall_limit = propagate(wall_limit_seed, open_lane)

    occupied_floor_margin = mul(
        mul(occupied_block, positive(floor_limit)), margin
    )
    growth_limit = maximum(
        wall_limit, add(floor_limit, occupied_floor_margin)
    )

    # Total available block volume supplies a conservative reachability bound.
    t_zero = equal(t, zero)
    lane_left_edge = mul(
        open_lane,
        clip01(add(shift(wall_band, dt=-1), t_zero)),
    )
    volume_mean = node(
        "gmean", [mul(black, lane_left_edge)], 4
    )
    volume = dense(
        volume_mean,
        np.eye(4, dtype=np.float32) * 4096.0,
    )

    # Extend block metadata through free space upstream of each block.
    free = mul(
        mul(open_lane, sub(O, floor_pixels)),
        sub(O, black),
    )
    boundary = mul(
        equal(g, affine(ones, 0.0, 63.0)), free
    )
    boundary_occupied = equal(
        player_height_lane, affine(ones, 0.0, 64.0)
    )
    boundary_limit = maximum(
        wall_limit,
        add(
            floor_limit,
            mul(
                mul(boundary_occupied, positive(floor_limit)),
                margin,
            ),
        ),
    )
    support_seeds = [
        add(front_value, boundary),
        add(block_end, affine(boundary, 64.0)),
        add(
            mul(growth_limit, black),
            mul(boundary_limit, boundary),
        ),
    ]

    def upstream_metadata(seeds):
        nonlocal nodes
        loop_id = fresh("L")
        state_ids = [fresh("s") for _ in seeds]
        for s in state_ids:
            channels[s] = 4
        old_nodes = nodes
        nodes = []
        next_ids = []
        for s, seed in zip(state_ids, seeds):
            nxt = add(seed, mul(free, unit_shift(s, "g", 1)))
            next_ids.append(nxt)
        body = nodes
        nodes = old_nodes
        nodes.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [
                {"id": s, "init": seed}
                for s, seed in zip(state_ids, seeds)
            ],
            "body": body,
            "next": dict(zip(state_ids, next_ids)),
        })
        outputs = [loop_id + "." + s for s in state_ids]
        for x in outputs:
            channels[x] = 4
        return outputs

    support_value, support_end, support_limit = upstream_metadata(
        support_seeds
    )
    support_height = affine(support_value, -1.0, 65.0)
    support_capacity_limit = maximum(
        support_limit, sub(support_end, volume)
    )

    # Goals propagate through bridge-connected components.
    mobility = add(open_lane, mul(wall_band, bridge_band))

    def goal_maps(colour, player):
        goal_pixels = mul(sub(colour, player), wall_band)
        h_wall = axis_pool(mul(gp1, goal_pixels), "g")
        t_wall = axis_pool(mul(tp1, goal_pixels), "g")
        h_seed = mul(
            maximum(
                shift(h_wall, dt=-4),
                shift(h_wall, dt=4),
            ),
            open_lane,
        )
        t_seed = mul(
            maximum(
                shift(t_wall, dt=-4),
                shift(t_wall, dt=4),
            ),
            open_lane,
        )
        component_priority = propagate(t_seed, mobility)
        local_priority = propagate(t_seed, open_lane)
        chosen_h_seed = mul(
            h_seed,
            mul(
                equal(t_seed, component_priority),
                positive(t_seed),
            ),
        )
        chosen_height = propagate(chosen_h_seed, mobility)
        return component_priority, local_priority, chosen_height

    priority11, local11, goal_height11 = goal_maps(
        colour11, player11
    )
    priority14, local14, goal_height14 = goal_maps(
        colour14, player14
    )

    goal_at_player = add(
        mul(goal_height11, player11),
        mul(goal_height14, player14),
    )
    priority_at_player = add(
        mul(priority11, player11),
        mul(priority14, player14),
    )
    local_at_player = add(
        mul(local11, player11),
        mul(local14, player14),
    )
    satisfied = mul(
        equal(gp1, goal_at_player),
        equal(local_at_player, priority_at_player),
    )
    unsatisfied = mul(player_edge, sub(O, satisfied))
    player_priority = mul(tp1, unsatisfied)
    greatest_player_priority = gmax(player_priority)
    active_point = mul(
        unsatisfied,
        mul(
            equal(player_priority, greatest_player_priority),
            positive(player_priority),
        ),
    )

    active_height = gmax(mul(gp1, active_point))
    active_t = gmax(mul(t, active_point))
    target_height = gmax(mul(goal_at_player, active_point))
    target_priority = gmax(
        mul(priority_at_player, active_point)
    )
    target_t = vaffine(target_priority, 1.0, -1.0)

    active_colour11 = gmax(mul(player11, active_point))
    active_colour14 = gmax(mul(player14, active_point))
    chosen_local_goal = add(
        mul(local11, active_colour11),
        mul(local14, active_colour14),
    )

    active_end = gmax(
        mul(shift(support_end, dg=1), active_point)
    )
    active_limit = gmax(
        mul(shift(support_capacity_limit, dg=1), active_point)
    )
    adjacent_goal = equal(
        local_at_player, priority_at_player
    )
    final_possible = mul(
        ge(field(target_height), field(active_limit)),
        ge(field(active_end), field(target_height)),
    )
    final_vector = gmax(mul(
        active_point, mul(adjacent_goal, final_possible)
    ))
    final_mode = field(final_vector)
    transit_mode = sub(O, final_mode)

    active_lane_seed = axis_pool(active_point, "g")
    active_lane = propagate(active_lane_seed, open_lane)

    # Consider bridges on the next wall toward the selected goal.
    wall_adjacent = mul(
        wall_band,
        clip01(maximum(
            shift(active_lane, dt=-4),
            shift(active_lane, dt=4),
        )),
    )
    route_right = greater(field(target_t), field(active_t))
    route_left = greater(field(active_t), field(target_t))
    correct_side = add(
        mul(greater(t, field(active_t)), route_right),
        mul(greater(field(active_t), t), route_left),
    )

    def across_candidate(x):
        downstream = shift(x, dg=1)
        return add(
            mul(shift(downstream, dt=4), route_right),
            mul(shift(downstream, dt=-4), route_left),
        )

    neighbour_height_candidates = across_candidate(support_height)
    neighbour_end_candidates = across_candidate(support_end)
    neighbour_limit_candidates = across_candidate(
        support_capacity_limit
    )
    neighbour_value_candidates = across_candidate(support_value)
    neighbour_goal_candidates = across_candidate(chosen_local_goal)

    bridge_edge = mul(
        bridge_pixels,
        sub(O, shift(bridge_pixels, dg=1)),
    )
    feasible_active = mul(
        ge(gp1, field(active_limit)),
        ge(field(active_end), gp1),
    )
    feasible_neighbour = mul(
        positive(neighbour_value_candidates),
        mul(
            ge(gp1, neighbour_limit_candidates),
            ge(neighbour_end_candidates, gp1),
        ),
    )
    bridge_candidates = mul(
        mul(mul(wall_adjacent, bridge_edge), correct_side),
        mul(
            mul(feasible_active, feasible_neighbour),
            transit_mode,
        ),
    )

    neighbour_adjacent_goal = equal(
        neighbour_goal_candidates, field(target_priority)
    )
    neighbour_can_finish = mul(
        ge(field(target_height), neighbour_limit_candidates),
        ge(neighbour_end_candidates, field(target_height)),
    )
    bad_neighbour = mul(
        neighbour_adjacent_goal,
        sub(O, neighbour_can_finish),
    )

    bridge_cost = add(
        absolute(sub(field(active_height), gp1)),
        absolute(sub(neighbour_height_candidates, gp1)),
    )
    bridge_cost = add(
        bridge_cost, affine(bad_neighbour, 100.0)
    )
    bridge_score = mul(
        affine(bridge_cost, -1.0, 1000.0),
        bridge_candidates,
    )
    selected_bridge = mul(
        bridge_candidates,
        equal(bridge_score, gmax(bridge_score)),
    )
    bridge_height = gmax(mul(gp1, selected_bridge))
    neighbour_height = gmax(mul(
        neighbour_height_candidates, selected_bridge
    ))
    neighbour_end = gmax(mul(
        neighbour_end_candidates, selected_bridge
    ))

    selected_wall = axis_pool(selected_bridge, "g")
    neighbour_lane_seed = mul(
        sub(open_lane, active_lane),
        clip01(maximum(
            shift(selected_wall, dt=-4),
            shift(selected_wall, dt=4),
        )),
    )
    neighbour_lane = propagate(neighbour_lane_seed, open_lane)

    # Positive pressure requests growth; negative pressure requests shrinkage.
    desired_active_height = add(
        mul(field(target_height), final_mode),
        mul(field(bridge_height), transit_mode),
    )
    active_error = signed_clip(mul(
        sub(field(active_height), desired_active_height),
        inverse_step,
    ))
    neighbour_error = mul(
        signed_clip(mul(
            sub(
                field(neighbour_height),
                field(bridge_height),
            ),
            inverse_step,
        )),
        transit_mode,
    )

    active_constraint = mul(
        active_lane,
        equal(support_height, field(active_height)),
    )
    neighbour_constraint = mul(
        mul(neighbour_lane, transit_mode),
        equal(support_height, field(neighbour_height)),
    )
    direct_pressure = add(
        mul(active_constraint, active_error),
        mul(neighbour_constraint, neighbour_error),
    )

    neighbour_t = gmax(mul(t, neighbour_lane))

    # Endpoint distance identifies separate blocks in the same lane,
    # without making an emptied block look artificially far away.
    def weak_potential(reference_t, reference_end, error):
        distance = add(
            absolute(sub(t, field(reference_t))),
            absolute(sub(support_end, field(reference_end))),
        )
        proximity = clip01(
            affine(distance, -1.0 / 128.0, 1.0)
        )
        return affine(mul(proximity, error), 0.04)

    weak_pressure = add(
        weak_potential(active_t, active_end, active_error),
        weak_potential(
            neighbour_t, neighbour_end, neighbour_error
        ),
    )

    active_nonzero = positive(absolute(active_error))
    neighbour_nonzero = positive(absolute(neighbour_error))
    active_driven = mul(active_constraint, active_nonzero)
    neighbour_driven = mul(
        neighbour_constraint, neighbour_nonzero
    )
    unconstrained = mul(
        positive(support_value),
        sub(O, clip01(add(active_driven, neighbour_driven))),
    )
    pressure = add(
        direct_pressure, mul(weak_pressure, unconstrained)
    )

    # Completed subtargets may relay resources, but are less attractive
    # than unconstrained blocks when an alternative transfer is available.
    protected = clip01(add(
        mul(active_constraint, sub(O, active_nonzero)),
        mul(neighbour_constraint, sub(O, neighbour_nonzero)),
    ))

    # Clicking a button grows its own block and shrinks the opposite one.
    small_field = field(small)
    large_field = sub(O, small_field)

    def scaled_shift(x, sign, factor):
        return add(
            mul(
                shift(x, dt=sign * 3 * factor),
                large_field,
            ),
            mul(
                shift(x, dt=sign * 4 * factor),
                small_field,
            ),
        )

    wall_left = scaled_shift(wall_band, -1, 1)
    wall_right = scaled_shift(wall_band, 1, 1)

    def source_at_button(x):
        return add(
            mul(wall_left, scaled_shift(x, -1, 2)),
            mul(wall_right, scaled_shift(x, 1, 2)),
        )

    source_pressure = source_at_button(pressure)
    source_height = source_at_button(support_height)
    source_end = source_at_button(support_end)
    source_value = source_at_button(support_value)

    source_can_shrink = mul(
        positive(source_value),
        greater(source_end, source_height),
    )
    destination_can_grow = greater(
        support_height, support_limit
    )
    has_button_wall = clip01(add(wall_left, wall_right))
    legal_button = mul(
        mul(button_pixels, has_button_wall),
        mul(source_can_shrink, destination_can_grow),
    )

    preservation_cost = affine(
        add(protected, source_at_button(protected)),
        0.125,
    )
    improvement = sub(
        sub(pressure, source_pressure), preservation_cost
    )
    button_score = mul(
        affine(improvement, 100.0, 1000.0),
        legal_button,
    )

    bridge_ready = mul(
        mul(
            equal(
                field(active_height), field(bridge_height)
            ),
            equal(
                field(neighbour_height), field(bridge_height)
            ),
        ),
        transit_mode,
    )
    transfer_score = affine(
        mul(selected_bridge, bridge_ready), 5000.0
    )
    click_banks = maximum(button_score, transfer_score)
    conv(
        click_banks,
        np.ones((1, 4, 1, 1), np.float32),
        name="click",
    )

    logits_bias = np.full(8, -10.0, np.float32)
    logits_bias[6] = 10.0
    dense(
        "avail",
        np.zeros((8, 8), np.float32),
        logits_bias,
        name="logits",
    )

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

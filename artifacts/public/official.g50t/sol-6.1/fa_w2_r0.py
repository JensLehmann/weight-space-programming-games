import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]
    memory_size = 23

    def name(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(value):
        key = name("p")
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def node(op, inputs, out=None, **fields):
        key = out if out is not None else name()
        item = {"id": key, "op": op, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return key

    def conv(x, weights, bias=None, out=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", [x], out=out, **fields)

    def dense(x, weights, bias=None, out=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", [x], out=out, **fields)

    def ss(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs, out=None):
        return node("vconcat", xs, out=out)

    def affine(x, scale, bias=0.0, channels=1):
        weights = np.eye(channels, dtype=np.float32)
        weights = weights[:, :, None, None] * scale
        return conv(x, weights, np.full(channels, bias, np.float32))

    def va(x, scale, bias=0.0):
        return dense(x, [[scale]], [bias])

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def relu(x):
        return node("relu", [x])

    def vrelu(x):
        return node("vrelu", [x])

    def inverse(x):
        return va(x, -1.0, 1.0)

    def gmax(x):
        return node("gmax", [x])

    one = ss("coords", 2, 3)
    row = ss("coords", 0, 1)
    col = ss("coords", 1, 2)

    def clip(x):
        return node("min2", [relu(x), one])

    def vclip(x):
        lower = vrelu(x)
        excess = vrelu(va(x, 1.0, -1.0))
        return vadd(lower, va(excess, -1.0))

    def channel_max(x, channels):
        result = ss(x, 0, 1)
        for i in range(1, channels):
            result = maximum(result, ss(x, i, i + 1))
        return result

    # Reset only after an action has been taken, avoiding the engine's
    # zero-action full-reset case. Preserve the reset bit across RESET.
    previous_action_sum = dense(
        "aprev", np.ones((1, 8), np.float32)
    )
    first_step = inverse(previous_action_sum)
    previous_reset = vs("aprev", 0, 1)
    reset_done = vs("mem", 22, 23)
    reset_available = vs("avail", 0, 1)

    reset_request = vmul(inverse(first_step), inverse(reset_done))
    reset_request = vmul(reset_request, reset_available)
    next_reset_done = vclip(vadd(reset_done, reset_request))

    effective_memory = vmul(
        vs("mem", 0, 22), inverse(previous_reset)
    )
    episode_start = vclip(vadd(first_step, previous_reset))

    def actor_detector(colour):
        weights = np.zeros((1, 16, 5, 5), np.float32)
        weights[0, colour] = 1.0
        weights[0, colour, 2, 2] = 0.0
        weights[0, 5, 2, 2] = 1.0
        return relu(conv("x0", weights, [-24.0]))

    player = actor_detector(9)
    enemy = actor_detector(14)

    # Goal centre: an isolated orange pixel in a black local patch.
    weights = np.zeros((1, 16, 3, 3), np.float32)
    weights[0, 5] = 1.0
    weights[0, 5, 1, 1] = 0.0
    weights[0, 9, 1, 1] = 1.0
    goal = relu(conv("x0", weights, [-8.0]))

    # Broad filled regions are floor; thin decorative wires are not.
    weights = np.zeros((1, 16, 7, 7), np.float32)
    weights[0, 1:] = 1.0
    broad_floor = clip(conv("x0", weights, [-37.0]))

    weights = np.zeros((1, 16, 3, 3), np.float32)
    weights[0, 1:] = 1.0
    local_floor = relu(conv("x0", weights, [-8.0]))
    floor = mul(broad_floor, local_floor)
    floor = maximum(floor, maximum(player, goal))

    colours = [8, 11, 15]
    weights3 = np.zeros((3, 16, 3, 3), np.float32)
    weights5 = np.zeros((3, 16, 5, 5), np.float32)
    weights1 = np.zeros((3, 16, 1, 1), np.float32)
    for i, colour in enumerate(colours):
        weights3[i, colour] = 1.0
        weights5[i, colour] = 1.0
        weights1[i, colour, 0, 0] = 1.0

    count3 = conv("x0", weights3)
    count5 = conv("x0", weights5)
    wires = conv("x0", weights1)

    # Plates have a solid 3x3 core but not a solid 5x5 face.
    plate_core = relu(affine(count3, 1.0, -8.0, 3))
    plate_extent = clip(affine(count5, -1.0, 14.0, 3))
    plates = mul(plate_core, plate_extent)
    plate_any = channel_max(plates, 3)

    gate_count = channel_max(ss(count5, 0, 2), 2)
    gates = clip(affine(gate_count, 1.0, -18.0))
    legal_floor = mul(floor, relu(affine(gates, -1.0, 1.0)))

    # Teleport-pad centres, including occupied pads.
    weights = np.zeros((1, 16, 7, 7), np.float32)
    for r in range(7):
        for c in range(7):
            if r in (0, 6) or c in (0, 6):
                weights[0, 15, r, c] = 1.0
    weights[0, 5, 3, 3] = 1.0
    pads = relu(conv("x0", weights, [-24.0]))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift_parameters = []
    for dr, dc in directions:
        weights = np.zeros((1, 1, 7, 7), np.float32)
        weights[0, 0, 3 + 3 * dr, 3 + 3 * dc] = 1.0
        shift_parameters.append(parameter(weights))

    decrement_weight = parameter(np.ones((1, 1, 1, 1), np.float32))
    decrement_bias = parameter([-1.0])
    portal_weight = parameter([[1.0]])
    portal_bias = parameter([-2.0])

    def neighbours(x):
        parts = []
        for key in shift_parameters:
            first = node("conv", [x], w=key)
            parts.append(node("conv", [first], w=key))
        return cat(*parts)

    def distance(seed_mask, allowed, teleport=False):
        seed = affine(seed_mask, 120.0)
        loop_id = name("wave")
        state_id = name("state")
        body = []

        def bn(op, inputs, **fields):
            key = name("body")
            item = {"id": key, "op": op, "in": list(inputs)}
            item.update(fields)
            body.append(item)
            return key

        parts = []
        for key in shift_parameters:
            first = bn("conv", [state_id], w=key)
            parts.append(bn("conv", [first], w=key))

        propagated = parts[0]
        for part in parts[1:]:
            propagated = bn("max2", [propagated, part])
        propagated = bn(
            "conv", [propagated],
            w=decrement_weight, b=decrement_bias
        )
        propagated = bn("relu", [propagated])
        propagated = bn("mul", [propagated, allowed])
        propagated = bn("max2", [propagated, state_id])

        if teleport:
            pad_values = bn("mul", [state_id, pads])
            best_pad = bn("gmax", [pad_values])
            best_pad = bn(
                "dense", [best_pad],
                w=portal_weight, b=portal_bias
            )
            best_pad = bn("vrelu", [best_pad])
            across_pads = bn("mul", [pads, best_pad])
            propagated = bn("max2", [propagated, across_pads])

        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 100,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: propagated},
        })
        return loop_id + "." + state_id

    from_player = distance(player, legal_floor)
    walking_goal = distance(goal, legal_floor)
    optimistic_goal = distance(goal, floor, teleport=True)
    goal_reachable = gmax(clip(mul(walking_goal, player)))
    goal_unreachable = inverse(goal_reachable)

    # Gate frontiers request their connected control circuits.
    near_player = channel_max(neighbours(from_player), 4)
    frontier = mul(gates, clip(near_player))
    frontier = mul(frontier, clip(optimistic_goal))
    importance_seed = mul(add(near_player, optimistic_goal), frontier)
    importance_seed = conv(
        importance_seed, np.ones((1, 1, 7, 7), np.float32)
    )

    useful_pads = mul(pads, clip(walking_goal))
    pad_request = vmul(gmax(useful_pads), goal_unreachable)
    pad_seed = mul(affine(pads, 10.0), pad_request)
    pad_seed = conv(pad_seed, np.ones((1, 1, 7, 7), np.float32))
    wire_seed = mul(add(importance_seed, pad_seed), wires)

    wire_weights = np.zeros((12, 3, 3, 3), np.float32)
    for d, (dr, dc) in enumerate(directions):
        for c in range(3):
            wire_weights[3 * d + c, c, 1 + dr, 1 + dc] = 1.0
    wire_shift = parameter(wire_weights)

    loop_id = name("circuit")
    state_id = name("wirestate")
    body = []

    def wn(op, inputs, **fields):
        key = name("wirebody")
        item = {"id": key, "op": op, "in": list(inputs)}
        item.update(fields)
        body.append(item)
        return key

    shifted = wn("conv", [state_id], w=wire_shift)
    pieces = [
        wn("slice", [shifted], start=3 * i, stop=3 * i + 3)
        for i in range(4)
    ]
    propagated = pieces[0]
    for piece in pieces[1:]:
        propagated = wn("max2", [propagated, piece])
    propagated = wn("max2", [propagated, state_id])
    propagated = wn("mul", [propagated, wires])

    graph.append({
        "id": loop_id,
        "op": "loop",
        "T": 160,
        "state": [{"id": state_id, "init": wire_seed}],
        "body": body,
        "next": {state_id: propagated},
    })
    wire_values = loop_id + "." + state_id

    # Detect an initially occupied plate using two consecutive thin-wire
    # samples outside the player's sprite.
    vertical_weights = np.zeros((3, 3, 3, 3), np.float32)
    horizontal_weights = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        vertical_weights[c, c, 1, 1] = 1.0
        vertical_weights[c, c, 1, 0] = -1.0
        vertical_weights[c, c, 1, 2] = -1.0
        horizontal_weights[c, c, 1, 1] = 1.0
        horizontal_weights[c, c, 0, 1] = -1.0
        horizontal_weights[c, c, 2, 1] = -1.0

    vertical_wire = relu(conv(wires, vertical_weights))
    horizontal_wire = relu(conv(wires, horizontal_weights))

    rays = []
    for dr, dc in directions:
        source = vertical_wire if dr else horizontal_wire
        weights = np.zeros((3, 3, 7, 7), np.float32)
        for c in range(3):
            weights[c, c, 3 + 3 * dr, 3 + 3 * dc] = 1.0
        shifted_three = conv(source, weights)

        samples = []
        for offset in (1, 2):
            weights = np.zeros((3, 3, 5, 5), np.float32)
            for c in range(3):
                weights[c, c, 2 + offset * dr, 2 + offset * dc] = 1.0
            samples.append(conv(shifted_three, weights))
        rays.append(mul(samples[0], samples[1]))

    contact_map = rays[0]
    for ray in rays[1:]:
        contact_map = maximum(contact_map, ray)
    contact_map = mul(contact_map, player)

    current_switch = gmax(channel_max(contact_map, 3))
    current_pressure = gmax(maximum(
        ss(contact_map, 0, 1), ss(contact_map, 2, 3)
    ))
    initial_switch = vmul(current_switch, episode_start)
    initial_pressure = vmul(current_pressure, episode_start)

    player_row = gmax(mul(player, row))
    player_col = gmax(mul(player, col))

    # UI indicates whether another recording slot is available.
    weights = np.zeros((1, 16, 3, 3), np.float32)
    weights[0, 9] = 1.0
    weights[0, 9, 1, 1] = 0.0
    weights[0, 0, 1, 1] = 1.0
    active_badge = relu(conv("x0", weights, [-8.0]))

    ui_row = relu(affine(
        node("abs", [affine(row, 1.0, -2.0 / 63.0)]),
        -1024.0, 1.0
    ))
    active_badge = mul(active_badge, ui_row)
    active_column = gmax(mul(active_badge, col))

    weights = np.zeros((1, 16, 3, 3), np.float32)
    weights[0, [1, 2, 9]] = 1.0
    badges = mul(clip(conv("x0", weights, [-7.0])), ui_row)
    last_column = gmax(mul(badges, col))
    remaining_slots = vadd(last_column, va(active_column, -1.0))
    can_record = vclip(va(remaining_slots, 63.0 / 4.0))

    old_row = vs(effective_memory, 0, 1)
    old_col = vs(effective_memory, 1, 2)
    old_kind = vs(effective_memory, 2, 3)
    old_lock = vs(effective_memory, 3, 4)

    def location_mask(r, c):
        dr = node("abs", [sub(row, r)])
        dc = node("abs", [sub(col, c)])
        return relu(affine(add(dr, dc), -1024.0, 1.0))

    old_target = location_mask(old_row, old_col)
    arrived = vmul(gmax(mul(old_target, player)), old_lock)

    used_mask = affine(one, 0.0)
    history = []
    for i in range(6):
        base = 4 + 3 * i
        r = vs(effective_memory, base, base + 1)
        c = vs(effective_memory, base + 1, base + 2)
        valid = vs(effective_memory, base + 2, base + 3)
        history.append((r, c, valid))
        used_mask = maximum(
            used_mask, mul(location_mask(r, c), valid)
        )

    unused = relu(affine(used_mask, -1.0, 1.0))
    arrival_event = vclip(vadd(arrived, initial_switch))
    pressure_event = vclip(vadd(
        vmul(arrived, old_kind), initial_pressure
    ))
    ready_record = vmul(pressure_event, can_record)

    raw_candidates = mul(plate_any, unused)
    raw_candidates = mul(raw_candidates, clip(optimistic_goal))
    reachable_mask = clip(from_player)
    reachable_candidates = mul(raw_candidates, reachable_mask)
    any_reachable = gmax(reachable_candidates)
    eligibility = add(
        reachable_mask,
        mul(relu(affine(reachable_mask, -1.0, 1.0)),
            inverse(any_reachable)),
    )
    candidates = mul(raw_candidates, eligibility)

    plate_importance = conv(
        mul(wire_values, plates),
        np.ones((1, 3, 1, 1), np.float32)
    )
    rank_weights = np.zeros((1, 3, 1, 1), np.float32)
    rank_weights[0, 0, 0, 0] = 0.25 * 64.0 * 63.0 / 4096.0
    rank_weights[0, 1, 0, 0] = 0.25 * 63.0 / 4096.0
    rank = conv("coords", rank_weights)

    score = conv(
        cat(plate_importance, from_player, optimistic_goal, rank),
        np.array([0.01, 0.004, 0.002, 1.0], np.float32)
          .reshape(1, 4, 1, 1),
        [1.0],
    )
    score = mul(score, candidates)
    best_score = gmax(score)
    difference = node("abs", [sub(score, best_score)])
    selected = mul(
        relu(affine(difference, -1.0e6, 1.0)), candidates
    )

    selected_valid = gmax(selected)
    selected_row = gmax(mul(selected, row))
    selected_col = gmax(mul(selected, col))
    pressure_plates = maximum(ss(plates, 0, 1), ss(plates, 2, 3))
    selected_kind = gmax(mul(selected, pressure_plates))

    keep = vmul(old_lock, inverse(arrived))
    keep = vmul(keep, goal_unreachable)
    replace = inverse(keep)

    def blend(old, new):
        return vadd(vmul(old, keep), vmul(new, replace))

    target_row = blend(old_row, selected_row)
    target_col = blend(old_col, selected_col)
    target_kind = blend(old_kind, selected_kind)
    switch_target = add(mul(old_target, keep), mul(selected, replace))
    switch_valid = vclip(vadd(
        keep, vmul(selected_valid, replace)
    ))

    entry_candidates = mul(pads, clip(from_player))
    entry_candidates = mul(entry_candidates, goal_unreachable)
    entry_score = mul(
        add(affine(from_player, 0.01, 1.0), rank),
        entry_candidates,
    )
    entry_best = gmax(entry_score)
    entry_gap = node("abs", [sub(entry_score, entry_best)])
    entry = mul(
        relu(affine(entry_gap, -1.0e6, 1.0)), entry_candidates
    )
    entry_valid = gmax(entry)

    navigation_target = add(
        switch_target, mul(entry, inverse(switch_valid))
    )
    navigation_valid = vclip(vadd(
        switch_valid, vmul(entry_valid, inverse(switch_valid))
    ))
    target_distance = distance(navigation_target, floor)

    objective_field = add(
        mul(target_distance, navigation_valid),
        mul(optimistic_goal, inverse(navigation_valid)),
    )
    route_field = add(
        mul(walking_goal, goal_reachable),
        mul(objective_field, goal_unreachable),
    )

    route_neighbours = neighbours(route_field)
    optimistic_neighbours = neighbours(optimistic_goal)
    legal_neighbours = neighbours(legal_floor)

    route_score = add(
        affine(route_neighbours, 100.0, 1.0, 4),
        affine(optimistic_neighbours, 0.01, 0.0, 4),
    )
    route_score = mul(route_score, legal_neighbours)
    route_score = add(
        route_score, affine(legal_neighbours, 1000.0, -1000.0, 4)
    )

    enemy_neighbours = channel_max(neighbours(enemy), 4)
    danger = maximum(enemy, enemy_neighbours)
    danger_neighbours = neighbours(danger)
    route_score = sub(
        route_score, affine(danger_neighbours, 3000.0, 0.0, 4)
    )

    sampled_scores = node("gmean", [mul(route_score, player)])
    features = vcat(sampled_scores, ready_record, reset_request)
    output_weights = np.zeros((8, 6), np.float32)
    output_bias = np.full(8, -1.0e8, np.float32)

    for i in range(4):
        output_weights[i + 1, i] = 4096.0
        output_bias[i + 1] = 0.0

    output_weights[5, 4] = 2.0e7
    output_bias[5] = -1.0e7
    output_weights[0, 5] = 3.0e7
    output_bias[0] = -1.0e7
    dense(features, output_weights, output_bias, out="logits")

    next_lock = vmul(switch_valid, goal_unreachable)
    next_lock = vmul(next_lock, inverse(ready_record))
    next_lock = vmul(next_lock, inverse(arrival_event))

    no_event = inverse(arrival_event)
    constant_one = va(arrival_event, 0.0, 1.0)
    next_history = []
    for i in range(6):
        incoming = (
            (player_row, player_col, constant_one)
            if i == 0 else history[i - 1]
        )
        for previous, new in zip(history[i], incoming):
            next_history.append(vadd(
                vmul(previous, no_event),
                vmul(new, arrival_event),
            ))

    vcat(
        target_row, target_col, target_kind, next_lock,
        *next_history, next_reset_done, out="mem_next"
    )
    conv(one, np.zeros((1, 1, 1, 1), np.float32), out="click")

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": params,
    }

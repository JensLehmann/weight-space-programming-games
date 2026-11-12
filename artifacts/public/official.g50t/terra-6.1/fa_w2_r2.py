import numpy as np


def build():
    graph = []
    params = {}
    serial = 0
    pserial = 0

    def ident(prefix="n"):
        nonlocal serial
        serial += 1
        return f"{prefix}{serial}"

    def param(value):
        nonlocal pserial
        pserial += 1
        name = f"p{pserial}"
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = ident()
        graph.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def unary(op, x):
        return node(op, [x])

    def binary(op, a, b):
        return node(op, [a, b])

    def vcat(*xs):
        return node("vconcat", xs)

    def scalar_affine(x, scale=1.0, bias=0.0):
        return dense(x, [[scale]], [bias])

    def one_minus(x):
        return scalar_affine(x, -1.0, 1.0)

    def clamp01(x):
        positive = unary("vrelu", x)
        above = unary("vrelu", scalar_affine(x, 1.0, -1.0))
        return dense(vcat(positive, above), [[1.0, -1.0]])

    def indicator(x, value):
        diff = scalar_affine(x, 1.0, -float(value))
        neg = scalar_affine(diff, -1.0)
        absolute = binary("vadd", unary("vrelu", diff), unary("vrelu", neg))
        return unary("vrelu", scalar_affine(absolute, -1.0, 1.0))

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    background = sl("x0", 0, 1)
    hud_blue = sl("x0", 1, 2)

    # Player marker.
    wp = np.zeros((1, 16, 5, 5), dtype=np.float32)
    wp[0, 5, 2, 2] = 1
    for dr, dc in [
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (0, -2), (0, -1), (0, 1), (0, 2),
    ]:
        wp[0, 9, 2 + dr, 2 + dc] = 1
    player = unary("relu", conv("x0", wp, [-8.0]))
    player_xy = unary(
        "gmax", binary("mul", sl("coords", 0, 2), player)
    )

    # Goal marker.
    wg = np.zeros((1, 16, 3, 3), dtype=np.float32)
    wg[0, 9, 1, 1] = 1
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        wg[0, 5, 1 + dr, 1 + dc] = 1
    goal = unary("relu", conv("x0", wg, [-4.0]))

    # Floor approximation: reject thin wires and gaps.
    wf = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for dr, dc in [
        (0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
        (-3, -3), (-3, 3), (3, -3), (3, 3),
    ]:
        wf[0, 0, 3 + dr, 3 + dc] = -1
    floor = unary("relu", conv(background, wf, [1.0]))

    # Door centres.
    wd = np.zeros((2, 16, 7, 7), dtype=np.float32)
    for out, colour in enumerate((8, 11)):
        wd[out, colour, 3, 3] = 5
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            wd[out, colour, 3 + dr, 3 + dc] = 1
    door_parts = unary("relu", conv("x0", wd, [-7.0, -7.0]))
    door_sum = conv(door_parts, np.ones((1, 2, 1, 1), np.float32))
    door = binary("min2", door_sum, one)
    closed_floor = unary("relu", binary("sub", floor, door))

    # Blue pressure, purple toggle, and white teleport switches.
    ws = np.zeros((3, 16, 3, 3), dtype=np.float32)
    for out, colour in enumerate((8, 11, 15)):
        ws[out, colour, :, :] = 1
    switch_parts = unary("relu", conv("x0", ws, [-8.0] * 3))
    blue_switch = sl(switch_parts, 0, 1)
    white_switch = sl(switch_parts, 2, 3)
    switch_sum = conv(
        switch_parts, np.ones((1, 3, 1, 1), dtype=np.float32)
    )
    switches = binary("min2", switch_sum, one)
    switches = unary("relu", binary("sub", switches, door))
    blue_switch = unary("relu", binary("sub", blue_switch, door))
    switches = binary("mul", switches, closed_floor)
    blue_switch = binary("mul", blue_switch, closed_floor)
    white_switch = binary("mul", white_switch, closed_floor)

    # White outlined teleport pads.
    wpad = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for r in range(7):
        for c in range(7):
            if r == 0 or r == 6 or c == 0 or c == 6:
                wpad[0, 15, r, c] = 1
    pads = unary("relu", conv("x0", wpad, [-23.0]))
    pads = binary("mul", pads, closed_floor)
    pads_present = clamp01(unary("gmax", pads))

    # Enemy marker.
    we = np.zeros((1, 16, 3, 3), dtype=np.float32)
    we[0, 5, 1, 1] = 1
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        we[0, 14, 1 + dr, 1 + dc] = 1
    enemy = unary("relu", conv("x0", we, [-4.0]))

    # Memory:
    # 0:16 destination predictions; 16:24 visited switches; 24 count
    # 25:29 recent positions; 29 teleport phase; 30:54 blocked edges
    # 54 delay state; 55:59 saved delay direction.
    prev_switches = vsl("mem", 0, 4)
    prev_blue = vsl("mem", 4, 8)
    prev_white = vsl("mem", 8, 12)
    prev_pad = vsl("mem", 12, 16)
    saved_switches = [
        vsl("mem", 16 + 2 * i, 18 + 2 * i) for i in range(4)
    ]
    count = vsl("mem", 24, 25)
    previous_xy = vsl("mem", 25, 27)
    previous2_xy = vsl("mem", 27, 29)
    phase = vsl("mem", 29, 30)
    records = [
        vsl("mem", 30 + 6 * i, 36 + 6 * i) for i in range(4)
    ]
    delay_state = vsl("mem", 54, 55)
    delay_direction = vsl("mem", 55, 59)
    prev_direction = vsl("aprev", 1, 5)

    def coordinate_match(saved):
        saved_row = vsl(saved, 0, 1)
        saved_col = vsl(saved, 1, 2)
        dr = unary("abs", binary("sub", row, saved_row))
        dc = unary("abs", binary("sub", col, saved_col))
        w = np.array([[[[-63.0]]]], dtype=np.float32)
        mr = unary("relu", conv(dr, w, [1.0]))
        mc = unary("relu", conv(dc, w, [1.0]))
        return binary("mul", mr, mc)

    occupied = [
        clamp01(scalar_affine(count, 1.0, -float(i)))
        for i in range(4)
    ]

    for saved, occ in zip(saved_switches, occupied):
        used = binary("min2", coordinate_match(saved), occ)
        switches = unary("relu", binary("sub", switches, used))
        blue_switch = unary("relu", binary("sub", blue_switch, used))
        white_switch = unary("relu", binary("sub", white_switch, used))

    def entered(predictions):
        return dense(
            binary("vmul", predictions, prev_direction),
            np.ones((1, 4), dtype=np.float32),
        )

    entered_switch = entered(prev_switches)
    entered_blue = entered(prev_blue)
    entered_white = entered(prev_white)
    entered_pad = entered(prev_pad)

    # Available rewind slots appear as colour 1 in the top HUD.
    row_weight = unary(
        "relu",
        conv(row, np.array([[[[-63.0]]]], np.float32), [4.0]),
    )
    hud_value = unary("gmax", binary("mul", hud_blue, row_weight))
    slot_available = clamp01(hud_value)
    pressure = clamp01(binary("vadd", entered_blue, entered_white))
    rewind = binary("vmul", pressure, slot_available)
    white_rewind = binary("vmul", entered_white, slot_available)

    def position_difference(a, b):
        diff = dense(
            vcat(a, b),
            [[1, 0, -1, 0], [0, 1, 0, -1]],
        )
        neg = dense(diff, -np.eye(2, dtype=np.float32))
        absolute = binary("vadd", unary("vrelu", diff), unary("vrelu", neg))
        return absolute

    absdiff = position_difference(player_xy, previous_xy)
    same_position = unary(
        "vrelu", dense(absdiff, [[-63.0, -63.0]], [1.0])
    )
    absdiff2 = position_difference(player_xy, previous2_xy)
    same_two_ago = unary(
        "vrelu", dense(absdiff2, [[-63.0, -63.0]], [1.0])
    )
    had_direction = dense(
        prev_direction, np.ones((1, 4), dtype=np.float32)
    )

    # A teleport changes position by more than one six-pixel step.
    jump = clamp01(dense(absdiff, [[63.0, 63.0]], [-6.5]))
    jump = binary("vmul", jump, had_direction)
    phase_started = clamp01(binary("vadd", phase, white_rewind))
    next_phase = binary("vmul", phase_started, one_minus(jump))

    # Six-pixel cardinal shifts.
    shift1 = np.zeros((4, 1, 7, 7), dtype=np.float32)
    shift2 = np.zeros((4, 4, 7, 7), dtype=np.float32)
    directions = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    for i, (dr, dc) in enumerate(directions):
        shift1[i, 0, 3 + dr, 3 + dc] = 1.0
        shift2[i, i, 3 + dr, 3 + dc] = 0.99
    shift1_name = param(shift1)
    shift2_name = param(shift2)

    def shifted_four(x):
        a = node("conv", [x], w=shift1_name)
        return node("conv", [a], w=shift2_name)

    def max_four(x):
        parts = [sl(x, i, i + 1) for i in range(4)]
        m = binary("max2", parts[0], parts[1])
        m = binary("max2", m, parts[2])
        return binary("max2", m, parts[3])

    def predict_neighbors(spatial):
        neighbors = shifted_four(spatial)
        prediction = unary("gmax", binary("mul", neighbors, player))
        return dense(prediction, np.eye(4, dtype=np.float32) / 0.99)

    switch_predictions = predict_neighbors(switches)
    blue_predictions = predict_neighbors(blue_switch)
    white_predictions = predict_neighbors(white_switch)
    pad_predictions = predict_neighbors(pads)

    # Before the first blue switch in a pad puzzle, insert two short
    # back-and-forth movements. The recorded ghost reaches the switch
    # later, leaving time to pass the door on the next replay.
    blue_adjacent = clamp01(
        dense(blue_predictions, np.ones((1, 4), dtype=np.float32))
    )
    delay_trigger = binary("vmul", pads_present, blue_adjacent)
    delay_trigger = binary("vmul", delay_trigger, one_minus(occupied[0]))
    delay_trigger = binary("vmul", delay_trigger, indicator(delay_state, 0))
    delay_trigger = binary("vmul", delay_trigger, had_direction)
    delay_trigger = binary("vmul", delay_trigger, slot_available)

    d1 = indicator(delay_state, 1)
    d2 = indicator(delay_state, 2)
    d3 = indicator(delay_state, 3)
    delay_active = clamp01(
        binary("vadd", delay_trigger, binary("vadd", d1, binary("vadd", d2, d3)))
    )

    reverse_w = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    reverse_prev = dense(prev_direction, reverse_w)
    reverse_delay = dense(delay_direction, reverse_w)

    delay_move = binary("vmul", reverse_prev, delay_trigger)
    delay_move = binary(
        "vadd", delay_move,
        binary("vmul", delay_direction, binary("vadd", d1, d3))
    )
    delay_move = binary("vadd", delay_move, binary("vmul", reverse_delay, d2))

    delay_increment = clamp01(
        binary("vadd", delay_trigger, binary("vadd", d1, binary("vadd", d2, d3)))
    )
    next_delay_state = binary("vadd", delay_state, delay_increment)
    next_delay_direction = binary(
        "vadd",
        binary("vmul", delay_direction, one_minus(delay_trigger)),
        binary("vmul", prev_direction, delay_trigger),
    )

    blocked = binary("vmul", same_position, had_direction)
    oscillated = binary(
        "vmul",
        same_two_ago,
        binary("vmul", one_minus(same_position), had_direction),
    )
    oscillated = binary("vmul", oscillated, one_minus(phase))
    oscillated = binary("vmul", oscillated, one_minus(delay_active))
    blocked_dir = binary("vmul", prev_direction, blocked)
    oscillation_dir = binary("vmul", reverse_prev, oscillated)
    event_dir = binary("vadd", blocked_dir, oscillation_dir)
    edge_event = clamp01(binary("vadd", blocked, oscillated))

    # Directional masks for recently blocked moves.
    blocked_map = binary("mul", coordinate_match(player_xy), event_dir)
    for rec in records:
        pos = vsl(rec, 0, 2)
        dirs = vsl(rec, 2, 6)
        rec_map = binary("mul", coordinate_match(pos), dirs)
        blocked_map = binary("max2", blocked_map, rec_map)
    blocked_map = binary("min2", blocked_map, one)
    edge_allowed = unary("relu", binary("sub", one, blocked_map))

    def propagate(seed, allowed):
        loop_id = ident("loop")
        state_id = ident("state")
        body = []

        def body_node(op, inputs, **fields):
            name = ident("body")
            body.append({
                "id": name, "op": op, "in": list(inputs), **fields
            })
            return name

        a = body_node("conv", [state_id], w=shift1_name)
        b = body_node("conv", [a], w=shift2_name)
        b = body_node("mul", [b, edge_allowed])
        parts = [
            body_node("slice", [b], start=i, stop=i + 1)
            for i in range(4)
        ]
        m = body_node("max2", [parts[0], parts[1]])
        m = body_node("max2", [m, parts[2]])
        m = body_node("max2", [m, parts[3]])
        m = body_node("mul", [m, allowed])
        updated = body_node("max2", [state_id, m])
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 96,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: updated},
        })
        return f"{loop_id}.{state_id}"

    goal_seed = binary("mul", goal, floor)
    normal_targets = binary("mul", switches, one_minus(phase))
    pad_targets = binary("mul", pads, phase)
    intermediate_seed = binary("add", normal_targets, pad_targets)

    goal_closed = propagate(goal_seed, closed_floor)
    intermediate_closed = propagate(intermediate_seed, closed_floor)
    goal_open = propagate(goal_seed, floor)

    def reached(field):
        value = unary("gmax", binary("mul", field, player))
        return clamp01(scalar_affine(value, 20.0))

    goal_reachable = reached(goal_closed)
    intermediate_reachable = reached(intermediate_closed)
    use_goal = binary("vmul", goal_reachable, one_minus(phase))
    use_intermediate = binary(
        "vmul", one_minus(use_goal), intermediate_reachable
    )
    use_open = binary(
        "vmul", one_minus(use_goal), one_minus(intermediate_reachable)
    )

    f_goal = binary("mul", goal_closed, use_goal)
    f_intermediate = binary("mul", intermediate_closed, use_intermediate)
    f_open = binary("mul", goal_open, use_open)
    selected = binary("add", binary("add", f_goal, f_intermediate), f_open)
    selected = binary("mul", selected, closed_floor)
    action_fields = binary("mul", shifted_four(selected), edge_allowed)
    action_scores = unary("gmax", binary("mul", action_fields, player))

    # Small local fallback prevents all-zero logits from repeatedly
    # selecting the first direction.
    local_fields = binary("mul", shifted_four(closed_floor), edge_allowed)
    local_scores = unary("gmax", binary("mul", local_fields, player))
    local_scores = dense(local_scores, 0.02 * np.eye(4, dtype=np.float32))
    action_scores = binary("vadd", action_scores, local_scores)

    # Discourage moves near the enemy.
    enemy_near = binary("max2", enemy, max_four(shifted_four(enemy)))
    danger_directions = shifted_four(enemy_near)
    danger_scores = unary("gmax", binary("mul", danger_directions, player))
    danger_factor = dense(
        danger_scores,
        -0.85 * np.eye(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    action_scores = binary("vmul", action_scores, danger_factor)
    action_scores = binary(
        "vadd", action_scores,
        dense(delay_move, 3.0 * np.eye(4, dtype=np.float32))
    )

    # Save newly encountered switches.
    next_saved = []
    for i in range(4):
        free = one_minus(occupied[i])
        preceding = (
            occupied[i - 1] if i else scalar_affine(count, 0.0, 1.0)
        )
        write = binary(
            "vmul", entered_switch, binary("vmul", free, preceding)
        )
        kept = binary("vmul", saved_switches[i], one_minus(write))
        new = binary("vmul", player_xy, write)
        next_saved.append(binary("vadd", kept, new))

    count_added = binary("vadd", count, entered_switch)
    excess = unary("vrelu", scalar_affine(count_added, 1.0, -4.0))
    next_count = dense(vcat(count_added, excess), [[1.0, -1.0]])

    # FIFO blocked-edge memory, cleared during rewinds or delay.
    def decayed_record(rec):
        pos = vsl(rec, 0, 2)
        dirs = dense(
            vsl(rec, 2, 6), 0.90 * np.eye(4, dtype=np.float32)
        )
        return vcat(pos, dirs)

    candidate_record = vcat(player_xy, event_dir)
    next_records = []
    clear_records = clamp01(binary("vadd", rewind, delay_active))
    keep_records = one_minus(clear_records)
    for i in range(4):
        old = decayed_record(records[i])
        incoming = (
            candidate_record if i == 0 else decayed_record(records[i - 1])
        )
        retained = binary("vmul", old, one_minus(edge_event))
        inserted = binary("vmul", incoming, edge_event)
        merged = binary("vadd", retained, inserted)
        next_records.append(binary("vmul", merged, keep_records))

    memory_value = vcat(
        switch_predictions,
        blue_predictions,
        white_predictions,
        pad_predictions,
        *next_saved,
        next_count,
        player_xy,
        previous_xy,
        next_phase,
        *next_records,
        next_delay_state,
        next_delay_direction,
    )
    graph.append({
        "id": "mem_next",
        "op": "dense",
        "in": [memory_value],
        "w": param(np.eye(59, dtype=np.float32)),
    })

    output_w = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        output_w[i + 1, i] = 1.0
    output_w[5, 4] = 5.0
    output_b = np.array(
        [-10.0, 0.0, 0.0, 0.0, 0.0, -0.5, -10.0, -10.0],
        dtype=np.float32,
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [vcat(action_scores, rewind)],
        "w": param(output_w),
        "b": param(output_b),
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": [one],
        "w": param(np.zeros((1, 1, 1, 1), dtype=np.float32)),
    })

    return {"graph": graph, "memory_size": 59, "params": params}

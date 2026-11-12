import numpy as np


def build():
    graph = []
    params = {}
    ni = [0]
    pi = [0]

    def param(value):
        name = "p%d" % pi[0]
        pi[0] += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(operation, inputs, name=None, **fields):
        if name is None:
            name = "n%d" % ni[0]
            ni[0] += 1
        item = {"id": name, "op": operation, "in": list(inputs)}
        item.update(fields)
        graph.append(item)
        return name

    def op(operation, *inputs):
        return node(operation, inputs)

    def conv(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("conv", [x], **fields)

    def dense(x, weights, bias=None, name=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("dense", [x], name=name, **fields)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def sa(x, channels, scale=1.0, bias=0.0):
        w = np.zeros((channels, channels, 1, 1), np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = scale
        return conv(x, w, np.full(channels, bias, np.float32))

    def va(x, size, scale=1.0, bias=0.0):
        w = np.eye(size, dtype=np.float32) * np.float32(scale)
        return dense(x, w, np.full(size, bias, np.float32))

    def vmax1(x):
        # Clamp a nonnegative scalar vector to at most one.
        return op(
            "vadd",
            x,
            va(op("vrelu", va(x, 1, 1.0, -1.0)), 1, -1.0),
        )

    def max4(x):
        return op(
            "max2",
            op("max2", sl(x, 0, 1), sl(x, 1, 2)),
            op("max2", sl(x, 2, 3), sl(x, 3, 4)),
        )

    def sum4(x):
        return conv(x, np.ones((1, 4, 1, 1), np.float32))

    def sumv4(x):
        return dense(x, np.ones((1, 4), np.float32))

    def detect(x, weights, thresholds):
        raw = conv(x, weights, -np.asarray(thresholds, np.float32))
        return sa(op("relu", raw), len(thresholds), 2.0)

    def shift_weights(classes):
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        first = np.zeros((classes * 4, classes, 7, 7), np.float32)
        later = np.zeros(
            (classes * 4, classes * 4, 7, 7), np.float32
        )
        for c in range(classes):
            for d, (dy, dx) in enumerate(dirs):
                j = 4 * c + d
                first[j, c, 3 + 3 * dy, 3 + 3 * dx] = 1.0
                later[j, j, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        return first, later

    def shift_six_twelve(x, classes):
        first, later = shift_weights(classes)
        p1 = param(first)
        p2 = param(later)
        s3 = node("conv", [x], w=p1)
        s6 = node("conv", [s3], w=p2)
        s9 = node("conv", [s6], w=p2)
        s12 = node("conv", [s9], w=p2)
        return s6, s12

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    ones = sl("coords", 2, 3)
    one = op("gmax", ones)

    def const(values):
        return dense(
            one, np.asarray(values, np.float32).reshape(-1, 1)
        )

    # Features at the local (2, 2) anchor of a six-pixel tile.
    w = np.zeros((7, 16, 7, 7), np.float32)

    def put(output, color, y, x):
        w[output, color, y - 2 + 3, x - 2 + 3] = 1.0

    dot = [
        (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3),
    ]
    for output, color in [(0, 14), (1, 8), (2, 9)]:
        for y, x in dot:
            put(output, color, y, x)

    for x in range(1, 5):
        put(3, 15, 0, x)
        put(3, 15, 3, x)
    for y in (1, 2):
        put(3, 15, y, 1)
        put(3, 15, y, 4)
    for y, x in [(1, 3), (2, 2), (2, 3)]:
        put(3, 7, y, x)

    for y in range(1, 5):
        for x in range(1, 5):
            put(4, 1, y, x)
            put(4, 12, y, x)

    for y, x in [(0, 0), (0, 5), (5, 0), (5, 5)]:
        put(5, 11, y, x)

    for y in (2, 3):
        for x in (2, 3):
            put(6, 5, y, x)

    features = detect(
        "x0", w, [11.5, 11.5, 11.5, 14.5, 15.5, 3.5, 3.5]
    )
    green = sl(features, 0, 1)
    red = sl(features, 1, 2)
    blue = sl(features, 2, 3)
    box = sl(features, 3, 4)
    carrier = sl(features, 5, 6)
    central_rail = sl(features, 6, 7)

    pegs = op("add", op("add", green, red), blue)
    occupied = op("add", pegs, box)
    empty_position = op("relu", sa(occupied, 1, -1.0, 1.0))

    # Rail edges can remain visible beneath occupied carriers.
    ew = np.zeros((2, 16, 7, 7), np.float32)
    for y in (2, 3):
        for x in (0, 5):
            ew[0, 5, y - 2 + 3, x - 2 + 3] = 1.0
    for y in (0, 5):
        for x in (2, 3):
            ew[1, 5, y - 2 + 3, x - 2 + 3] = 1.0
    edge_rail = detect("x0", ew, [3.5, 3.5])
    rail = op(
        "max2",
        central_rail,
        op("max2", sl(edge_rail, 0, 1), sl(edge_rail, 1, 2)),
    )
    terrain = op("max2", rail, carrier)

    s6, s12 = shift_six_twelve(features, 7)
    mid_green = sl(s6, 0, 4)
    mid_red = sl(s6, 4, 8)
    mid_blue = sl(s6, 8, 12)
    mid_box = sl(s6, 12, 16)
    mid_empty = sl(s6, 16, 20)
    mid_carrier = sl(s6, 20, 24)
    mid_rail = sl(s6, 24, 28)

    target_green = sl(s12, 0, 4)
    target_red = sl(s12, 4, 8)
    target_blue = sl(s12, 8, 12)
    target_empty = sl(s12, 16, 20)
    target_carrier = sl(s12, 20, 24)

    middle = op(
        "add",
        op("add", mid_green, mid_red),
        op("add", mid_blue, mid_box),
    )
    mid_pegs = op("add", op("add", mid_green, mid_red), mid_blue)
    target_pegs = op(
        "add", op("add", target_green, target_red), target_blue
    )

    # Memory: four jump records, plan flag, failed flag, exploration
    # timer and phase, carrier moments, stall timer, undo countdown,
    # and previous green/red count.
    history = vsl("mem", 0, 16)
    plan_ty = vsl("mem", 0, 1)
    plan_tx = vsl("mem", 1, 2)
    plan_sy = vsl("mem", 2, 3)
    plan_sx = vsl("mem", 3, 4)
    plan_present = vsl("mem", 16, 17)
    failed_flag = vsl("mem", 17, 18)
    timer = vsl("mem", 18, 19)
    phase_memory = vsl("mem", 19, 23)
    previous_carrier_stats = vsl("mem", 23, 26)
    stall = vsl("mem", 26, 27)
    undo_count = vsl("mem", 27, 28)
    previous_piece_count = vsl("mem", 28, 29)

    previous_direction = dense(
        "aprev",
        np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], np.float32),
    )

    row_offsets = const([-12.0 / 63.0, 0, 12.0 / 63.0, 0])
    col_offsets = const([0, 12.0 / 63.0, 0, -12.0 / 63.0])
    destination_row = op("add", row, row_offsets)
    destination_col = op("add", col, col_offsets)

    def match(y, x, expected_y, expected_x, channels):
        distance = op(
            "add",
            op("abs", op("sub", y, expected_y)),
            op("abs", op("sub", x, expected_x)),
        )
        return op("relu", sa(distance, channels, -63.0, 1.0))

    reverse_ban = None
    for i in range(4):
        old_ty = vsl("mem", 4 * i, 4 * i + 1)
        old_tx = vsl("mem", 4 * i + 1, 4 * i + 2)
        old_sy = vsl("mem", 4 * i + 2, 4 * i + 3)
        old_sx = vsl("mem", 4 * i + 3, 4 * i + 4)
        at_target = match(row, col, old_ty, old_tx, 1)
        to_source = match(
            destination_row, destination_col, old_sy, old_sx, 4
        )
        pair = op("mul", at_target, to_source)
        reverse_ban = (
            pair if reverse_ban is None
            else op("max2", reverse_ban, pair)
        )

    failed_source = match(row, col, plan_sy, plan_sx, 1)
    failed_target = match(
        destination_row, destination_col, plan_ty, plan_tx, 4
    )
    failed_ban = op(
        "mul", op("mul", failed_source, failed_target), failed_flag
    )
    banned = op("max2", reverse_ban, failed_ban)

    legal_sum = op("add", op("add", middle, pegs), target_empty)
    legal_raw = op("relu", sa(legal_sum, 4, 1.0, -2.5))
    legal_raw = sa(legal_raw, 4, 2.0)
    legal = op("relu", op("sub", legal_raw, sa(banned, 4, 2.0)))
    any_legal = op("gmax", max4(legal))

    capture = op(
        "add",
        op("mul", green, mid_green),
        op("mul", red, mid_red),
    )
    nearby = sum4(middle)
    _, nearby_destination = shift_six_twelve(nearby, 1)
    remaining_nearby = op("sub", nearby_destination, capture)

    score = sa(capture, 4, 5.0, 10.0)
    score = op("add", score, sa(remaining_nearby, 4, 0.6))
    score = op("add", score, sa(target_carrier, 4, 2.0))
    score = op("sub", score, sa(blue, 1, 2.0))
    score = op(
        "add",
        score,
        op("add", sa(row, 1, 0.129024), sa(col, 1, 0.002016)),
    )
    score = op("add", score, const([0.0, 0.15, 0.30, 0.45]))
    score = op("mul", score, legal)
    source_click = max4(score)

    best_score = op("gmax", source_click)
    difference = op("sub", score, best_score)
    winner = op("relu", sa(difference, 4, 1.0, 0.00001))
    winner = sa(winner, 4, 100000.0)
    winner = op("mul", winner, legal)

    new_ty = sumv4(op("gmax", op("mul", winner, destination_row)))
    new_tx = sumv4(op("gmax", op("mul", winner, destination_col)))
    new_sy = sumv4(op("gmax", op("mul", winner, row)))
    new_sx = sumv4(op("gmax", op("mul", winner, col)))
    new_plan = op("vconcat", new_ty, new_tx, new_sy, new_sx)

    # Destination marker.
    mw = np.zeros((1, 16, 7, 7), np.float32)
    marker_pixels = [
        (1, 2), (1, 3),
        (2, 1), (2, 4),
        (3, 1), (3, 4),
        (4, 2), (4, 3),
    ]
    for y, x in marker_pixels:
        mw[0, 2, y - 2 + 3, x - 2 + 3] = 1.0
    marker = detect("x0", mw, [7.5])
    any_marker = op("gmax", marker)
    marker_distance = op(
        "add",
        op("abs", op("sub", row, plan_ty)),
        op("abs", op("sub", col, plan_tx)),
    )
    marker_distance = op("mul", marker_distance, plan_present)
    marker_click = op(
        "mul", marker, sa(marker_distance, 1, -64.0, 200.0)
    )

    # Failure/reset key.
    kw = np.zeros((1, 16, 7, 7), np.float32)
    for y in (1, 2):
        for x in range(2, 8):
            kw[0, 15, y - 2 + 3, x - 4 + 3] = 1.0
    key = detect("x0", kw, [11.5])
    any_key = op("gmax", key)
    key_click = sa(key, 1, 1000.0)
    click_base = op("max2", source_click, marker_click)
    node("max2", [click_base, key_click], name="click")

    # Rail goals: carried peg, empty carrier, carried box.
    source_goal_directions = op("mul", middle, target_empty)
    source_goal_directions = op(
        "relu",
        op("sub", source_goal_directions, sa(reverse_ban, 4, 2.0)),
    )
    source_goal = max4(source_goal_directions)
    allowed_source = op("max2", empty_position, carrier)
    source_goal = op(
        "mul", op("mul", source_goal, terrain), allowed_source
    )

    destination_goal = max4(op("mul", middle, target_pegs))
    destination_goal = op(
        "mul", op("mul", destination_goal, terrain), empty_position
    )

    rw = np.zeros((4, 4, 1, 1), np.float32)
    for i, j in enumerate([2, 3, 0, 1]):
        rw[i, j, 0, 0] = 1.0
    opposite_empty = conv(mid_empty, rw)
    box_goal = max4(op("mul", mid_pegs, opposite_empty))
    box_goal = op(
        "mul", op("mul", box_goal, terrain), allowed_source
    )

    goals = op("concat", source_goal, destination_goal, box_goal)
    seeds = sa(goals, 3, 60.0)

    # Recurrent distance-to-goal wave along visible rail cells.
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    wave_weights = []
    for dy, dx in directions:
        ww = np.zeros((3, 3, 7, 7), np.float32)
        for c in range(3):
            ww[c, c, 3 + 3 * dy, 3 + 3 * dx] = 1.0
        wave_weights.append(param(ww))

    outer_graph = graph
    graph = []
    neighbors = []
    for weight in wave_weights:
        half = node("conv", ["wave"], w=weight)
        neighbors.append(node("conv", [half], w=weight))
    neighbor_max = op(
        "max2",
        op("max2", neighbors[0], neighbors[1]),
        op("max2", neighbors[2], neighbors[3]),
    )
    propagated = op("relu", sa(neighbor_max, 3, 1.0, -1.0))
    propagated = op("max2", propagated, seeds)
    next_wave = op("mul", propagated, terrain)
    body = graph
    graph = outer_graph
    graph.append({
        "id": "routing",
        "op": "loop",
        "T": 32,
        "state": [{"id": "wave", "init": seeds}],
        "body": body,
        "next": {"wave": next_wave},
    })

    next_terrain = op("max2", mid_rail, mid_carrier)
    no_next_carrier = op("relu", sa(mid_carrier, 4, -1.0, 1.0))
    possible = op(
        "mul", op("mul", carrier, next_terrain), no_next_carrier
    )

    # Keep an unoccupied carrier visible. A green-carrying platform is
    # allowed to move toward the edge because it can scroll the board.
    next_row = op(
        "add", row, const([-6.0 / 63.0, 0, 6.0 / 63.0, 0])
    )
    next_col = op(
        "add", col, const([0, 6.0 / 63.0, 0, -6.0 / 63.0])
    )
    row_low = op("min2", op("relu", sa(next_row, 4, 63.0, -1.0)), one)
    row_high = op("min2", op("relu", sa(next_row, 4, -63.0, 61.0)), one)
    col_low = op("min2", op("relu", sa(next_col, 4, 63.0, -1.0)), one)
    col_high = op("min2", op("relu", sa(next_col, 4, -63.0, 61.0)), one)
    safe = op("mul", op("mul", row_low, row_high), op("mul", col_low, col_high))
    safe_or_green = op("min2", op("add", safe, green), one)
    possible = op("mul", possible, safe_or_green)
    possible_vector = op("gmax", possible)
    any_possible = op("gmax", max4(possible))

    peg_role = op("mul", carrier, pegs)
    box_role = op("mul", carrier, box)
    empty_role = op("mul", carrier, empty_position)
    roles = op("concat", peg_role, empty_role, box_role)

    route_benefits = []
    for d, weight in enumerate(wave_weights):
        half = node("conv", ["routing.wave"], w=weight)
        full = node("conv", [half], w=weight)
        improvement = op("relu", op("sub", full, "routing.wave"))
        relevant = op("mul", improvement, roles)
        relevant_sum = conv(
            relevant, np.ones((1, 3, 1, 1), np.float32)
        )
        relevant_sum = op(
            "mul", relevant_sum, sl(possible, d, d + 1)
        )
        route_benefits.append(op("gmax", relevant_sum))
    route_vector = op("vconcat", *route_benefits)

    # Periodic exploration preference.
    phase_sum = dense(phase_memory, np.ones((1, 4), np.float32))
    phase_missing = op("vrelu", va(phase_sum, 1, -1.0, 1.0))
    initial_phase = dense(
        phase_missing,
        np.asarray([[1.0], [0.0], [0.0], [0.0]], np.float32),
    )
    phase = op("vadd", phase_memory, initial_phase)
    advanced_timer = op("vadd", timer, previous_direction)
    wrapped = op("vrelu", va(advanced_timer, 1, 1.0, -5.0))
    next_timer = op("vadd", advanced_timer, va(wrapped, 1, -6.0))
    rotation = np.zeros((4, 4), np.float32)
    for i in range(4):
        rotation[(i + 1) % 4, i] = 1.0
    rotated_phase = dense(phase, rotation)
    next_phase = op(
        "vadd",
        op("vmul", phase, va(wrapped, 1, -1.0, 1.0)),
        op("vmul", rotated_phase, wrapped),
    )

    preference = np.zeros((4, 8), np.float32)
    action_ids = [1, 4, 2, 3]
    opposite_ids = [2, 3, 1, 4]
    for i in range(4):
        preference[i, action_ids[i]] = 0.6
        preference[i, opposite_ids[i]] = -0.4
    momentum = dense(
        "aprev",
        preference,
        np.asarray([0.03, 0.04, 0.02, 0.01], np.float32),
    )
    exploration = op(
        "vadd", op("vadd", momentum, phase), const([2, 2, 2, 2])
    )
    exploration = op("vmul", exploration, possible_vector)
    direction_logits = op("vadd", exploration, va(route_vector, 4, 100.0))

    # Discourage repeating a direction that left carrier moments unchanged.
    current_carrier_stats = op(
        "vconcat",
        op("gmean", op("mul", carrier, row)),
        op("gmean", op("mul", carrier, col)),
        op("gmean", carrier),
    )
    current_carrier_stats = va(current_carrier_stats, 3, 4096.0)
    delta = op(
        "vadd", current_carrier_stats, va(previous_carrier_stats, 3, -1.0)
    )
    abs_delta = op(
        "vadd", op("vrelu", delta), op("vrelu", va(delta, 3, -1.0))
    )
    delta_sum = dense(abs_delta, np.ones((1, 3), np.float32))
    unchanged = op("vrelu", va(delta_sum, 1, -50.0, 1.0))
    blocked = op("vmul", unchanged, previous_direction)

    last_direction_weights = np.zeros((4, 8), np.float32)
    for i, action_id in enumerate(action_ids):
        last_direction_weights[i, action_id] = 1.0
    last_direction = dense("aprev", last_direction_weights)
    blocked_direction = op("vmul", last_direction, blocked)
    direction_logits = op(
        "vadd", direction_logits, va(blocked_direction, 4, -10000.0)
    )

    no_legal = va(any_legal, 1, -1.0, 1.0)
    no_marker = va(any_marker, 1, -1.0, 1.0)
    no_possible = va(any_possible, 1, -1.0, 1.0)
    no_key = va(any_key, 1, -1.0, 1.0)
    dead = op(
        "vmul",
        op("vmul", op("vmul", no_legal, no_marker), no_possible),
        no_key,
    )
    previous_undo = vsl("aprev", 7, 8)
    undo_marker = op(
        "vmul", op("vmul", any_marker, previous_undo), failed_flag
    )

    # After 48 steps without a visible green/red reduction, undo five
    # actions and blacklist the most recently planned jump.
    piece_count = va(op("gmean", op("add", green, red)), 1, 4096.0)
    count_drop = op(
        "vrelu",
        op("vadd", previous_piece_count, va(piece_count, 1, -1.0)),
    )
    no_drop = op("vrelu", va(count_drop, 1, -2.0, 1.0))
    undo_active = vmax1(op("vrelu", undo_count))
    stall_trigger = vmax1(op("vrelu", va(stall, 1, 1.0, -47.0)))
    stall_trigger = op(
        "vmul", stall_trigger, va(undo_active, 1, -1.0, 1.0)
    )
    stall_trigger = op("vmul", stall_trigger, no_key)
    backtracking = op("vadd", undo_active, stall_trigger)
    no_backtracking = va(backtracking, 1, -1.0, 1.0)

    next_undo_count = op(
        "vrelu",
        op(
            "vadd",
            op("vadd", undo_count, va(stall_trigger, 1, 5.0)),
            const([-1.0]),
        ),
    )
    next_stall = op(
        "vmul",
        op("vmul", op("vadd", stall, one), no_drop),
        no_backtracking,
    )

    click_logit = va(op("vadd", any_legal, any_marker), 1, 10000.0)
    click_logit = op("vadd", click_logit, va(any_key, 1, 20000.0))
    undo_logit = va(undo_marker, 1, 15000.0, 1.0)
    undo_logit = op("vadd", undo_logit, va(backtracking, 1, 30000.0))

    logit_inputs = op(
        "vconcat", direction_logits, click_logit, undo_logit
    )
    out_w = np.zeros((8, 6), np.float32)
    out_b = np.full(8, -100.0, np.float32)
    out_w[1, 0] = 1.0
    out_w[2, 2] = 1.0
    out_w[3, 3] = 1.0
    out_w[4, 1] = 1.0
    out_w[6, 4] = 1.0
    out_w[7, 5] = 1.0
    out_b[[1, 2, 3, 4, 6, 7]] = 0.0
    dense(logit_inputs, out_w, out_b, name="logits")

    # Save plans only when the actor is actually selecting a source.
    update = op(
        "vmul",
        op("vmul", op("vmul", any_legal, no_marker), no_key),
        no_backtracking,
    )
    keep = va(update, 1, -1.0, 1.0)
    older_history = vsl(history, 0, 12)
    pushed_history = op("vconcat", new_plan, older_history)
    next_history = op(
        "vadd",
        op("vmul", pushed_history, update),
        op("vmul", history, keep),
    )
    next_present = op(
        "vadd", update, op("vmul", plan_present, keep)
    )
    failed_product = op("vmul", failed_flag, dead)
    failed_union = op(
        "vadd",
        op("vadd", failed_flag, dead),
        va(failed_product, 1, -1.0),
    )
    failed_union = op(
        "vadd",
        op("vadd", failed_union, stall_trigger),
        va(op("vmul", failed_union, stall_trigger), 1, -1.0),
    )
    next_failed = op("vmul", failed_union, keep)

    node(
        "vconcat",
        [
            next_history,
            next_present,
            next_failed,
            next_timer,
            next_phase,
            current_carrier_stats,
            next_stall,
            next_undo_count,
            piece_count,
        ],
        name="mem_next",
    )

    return {
        "graph": graph,
        "memory_size": 29,
        "params": params,
    }

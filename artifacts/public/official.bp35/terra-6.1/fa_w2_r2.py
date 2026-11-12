import numpy as np


def build():
    graph = []
    params = {}
    seq = [0]

    def fresh():
        seq[0] += 1
        return "n%d" % seq[0]

    def emit(dst, op, inputs=None, **fields):
        ident = fresh()
        item = {"id": ident, "op": op}
        if inputs is not None:
            item["in"] = inputs
        item.update(fields)
        dst.append(item)
        return ident

    def par(key, value):
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def conv(dst, x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit(dst, "conv", [x], **fields)

    def dense(dst, x, w, b=None):
        fields = {"w": w}
        if b is not None:
            fields["b"] = b
        return emit(dst, "dense", [x], **fields)

    def sl(dst, x, a, b):
        return emit(dst, "slice", [x], start=a, stop=b)

    def vs(dst, x, a, b):
        return emit(dst, "vslice", [x], start=a, stop=b)

    def add(dst, a, b):
        return emit(dst, "add", [a, b])

    def sub(dst, a, b):
        return emit(dst, "sub", [a, b])

    def mul(dst, a, b):
        return emit(dst, "mul", [a, b])

    def maximum(dst, a, b):
        return emit(dst, "max2", [a, b])

    def minimum(dst, a, b):
        return emit(dst, "min2", [a, b])

    def relu(dst, x):
        return emit(dst, "relu", [x])

    def gmax(dst, x):
        return emit(dst, "gmax", [x])

    def cat(dst, *xs):
        return emit(dst, "concat", list(xs))

    def vcat(dst, *xs):
        return emit(dst, "vconcat", list(xs))

    def vadd(dst, a, b):
        return emit(dst, "vadd", [a, b])

    def vmul(dst, a, b):
        return emit(dst, "vmul", [a, b])

    row = sl(graph, "coords", 0, 1)
    col = sl(graph, "coords", 1, 2)
    one = sl(graph, "coords", 2, 3)

    wall = sl(graph, "x0", 5, 6)
    goal = sl(graph, "x0", 7, 8)
    green = sl(graph, "x0", 8, 9)
    empty_color = sl(graph, "x0", 10, 11)
    toggle = sl(graph, "x0", 12, 13)
    brick = sl(graph, "x0", 14, 15)
    color15 = sl(graph, "x0", 15, 16)

    # Player center.
    w = np.zeros((1, 16, 7, 7), np.float32)
    for dy in (-2, -1, 0, 1):
        w[0, 9, 3 + dy, 3] = 1
    par("player_w", w)
    par("player_b", [-3.5])
    par("double_w", [[[[2.0]]]])
    player = relu(graph, conv(graph, "x0", "player_w", "player_b"))
    player = conv(graph, player, "double_w")

    # Yellow blocks and spikes share color 15 at their centers.
    w = np.zeros((2, 16, 7, 7), np.float32)
    w[0, 3, 1, 5] = 1
    w[1, 12, 3, 4] = 1
    par("inspect_w", w)
    inspect = conv(graph, "x0", "inspect_w")
    yellow = mul(graph, color15, sl(graph, inspect, 0, 1))
    spike = sub(graph, color15, yellow)
    solid_toggle = mul(graph, toggle, sl(graph, inspect, 1, 2))
    dotted_toggle = sub(graph, toggle, solid_toggle)
    destructible = add(graph, brick, solid_toggle)
    destructible = add(graph, destructible, yellow)

    # Lattice aligned with the player.
    wr = np.zeros((1, 1, 7, 7), np.float32)
    wr[0, 0, 0, 3] = 1
    wr[0, 0, 6, 3] = 1
    wc = np.zeros((1, 1, 7, 7), np.float32)
    wc[0, 0, 3, 0] = 1
    wc[0, 0, 3, 6] = 1
    par("lattice_row_w", wr)
    par("lattice_col_w", wc)

    rows = emit(graph, "rowmax", [player])
    cols = emit(graph, "colmax", [player])
    rs, cs, lid = fresh(), fresh(), fresh()
    body = []
    rr = conv(body, rs, "lattice_row_w")
    rr = conv(body, rr, "lattice_row_w")
    rr = minimum(body, maximum(body, rr, rs), one)
    cc = conv(body, cs, "lattice_col_w")
    cc = conv(body, cc, "lattice_col_w")
    cc = minimum(body, maximum(body, cc, cs), one)
    graph.append({
        "id": lid,
        "op": "loop",
        "T": 11,
        "state": [{"id": rs, "init": rows}, {"id": cs, "init": cols}],
        "body": body,
        "next": {rs: rr, cs: cc},
    })
    lattice = mul(graph, lid + "." + rs, lid + "." + cs)

    par("bar_w", [[[[63.0]]]])
    par("bar_b", [-62.0])
    bar = relu(graph, conv(graph, row, "bar_w", "bar_b"))
    screen = sub(graph, one, bar)
    lattice = mul(graph, lattice, screen)

    # Current gravity, inferred from the camera-relative player row.
    player_row = gmax(graph, mul(graph, player, row))
    par("gravity_w", [[500.0]])
    par("gravity_b", [-270.0])
    normal = emit(
        graph, "vsigmoid",
        [dense(graph, player_row, "gravity_w", "gravity_b")]
    )
    par("invert_v_w", [[-1.0]])
    par("invert_v_b", [1.0])
    reverse = dense(graph, normal, "invert_v_w", "invert_v_b")
    active_pair = vcat(graph, normal, reverse)
    opposite_pair = vcat(graph, reverse, normal)

    # A short cooldown after an observed gravity flip.
    previous_normal = vs(graph, "mem", 0, 1)
    old_cooldown = vs(graph, "mem", 1, 2)
    previous_click = vs(graph, "aprev", 6, 7)
    par("neg_v_w", [[-1.0]])
    neg_previous = dense(graph, previous_normal, "neg_v_w")
    neg_normal = dense(graph, normal, "neg_v_w")
    delta1 = emit(graph, "vrelu", [vadd(graph, normal, neg_previous)])
    delta2 = emit(graph, "vrelu", [vadd(graph, previous_normal, neg_normal)])
    changed = vmul(graph, vadd(graph, delta1, delta2), previous_click)
    par("cool_start_w", [[8.0]])
    par("cool_decay_w", [[1.0]])
    par("cool_decay_b", [-1.0])
    start_cool = dense(graph, changed, "cool_start_w")
    decay = dense(graph, old_cooldown, "cool_decay_w", "cool_decay_b")
    decay = emit(graph, "vrelu", [decay])
    cooldown = vadd(graph, start_cool, decay)
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [normal, cooldown],
    })

    # Directional vertical shifts by six pixels.
    w = np.zeros((2, 2, 7, 7), np.float32)
    w[0, 0, 0, 3] = 1
    w[1, 1, 6, 3] = 1
    par("vertical_w", w)

    w = np.zeros((2, 2, 7, 7), np.float32)
    w[0, 0, 6, 3] = 1
    w[1, 1, 0, 3] = 1
    par("vertical_reverse_w", w)

    def vertical(dst, x):
        x = conv(dst, x, "vertical_w")
        return conv(dst, x, "vertical_w")

    def vertical_reverse(dst, x):
        x = conv(dst, x, "vertical_reverse_w")
        return conv(dst, x, "vertical_reverse_w")

    # Empty cells in a gravity ray.
    fall_open = sub(graph, one, wall)
    fall_open = sub(graph, fall_open, brick)
    fall_open = sub(graph, fall_open, solid_toggle)
    fall_open = sub(graph, fall_open, color15)
    fall_open = sub(graph, fall_open, green)
    fall_open = sub(graph, fall_open, goal)
    fall_open = relu(graph, fall_open)
    fall_open = mul(graph, fall_open, lattice)

    spike_grid = mul(graph, spike, lattice)
    spike_pair = cat(graph, spike_grid, spike_grid)

    # For gravity flips, an unobserved continuation is also considered
    # unsafe. Ordinary movement may still explore beyond the screen.
    par("edge_w", np.array([[[[-63.0]]], [[[63.0]]]], np.float32))
    par("edge_b", [6.0, -57.0])
    edge = relu(graph, conv(graph, row, "edge_w", "edge_b"))
    edge = minimum(graph, edge, one)
    edge = mul(graph, edge, fall_open)
    uncertain_seed = maximum(graph, spike_pair, edge)

    ds, us, danger_loop = fresh(), fresh(), fresh()
    body = []
    dn = vertical(body, ds)
    dn = mul(body, dn, fall_open)
    dn = maximum(body, dn, spike_pair)
    un = vertical(body, us)
    un = mul(body, un, fall_open)
    un = maximum(body, un, uncertain_seed)
    graph.append({
        "id": danger_loop,
        "op": "loop",
        "T": 11,
        "state": [
            {"id": ds, "init": spike_pair},
            {"id": us, "init": uncertain_seed},
        ],
        "body": body,
        "next": {ds: dn, us: un},
    })
    danger = danger_loop + "." + ds
    uncertain = danger_loop + "." + us
    danger_next = vertical(graph, danger)

    # Virtual entry to a breakable block is safe only if the ensuing fall
    # is safe.
    virtual_danger = mul(graph, danger_next, destructible)
    virtual_danger = maximum(graph, virtual_danger, danger)
    safe = sub(graph, one, virtual_danger)

    passable = sub(graph, one, wall)
    passable = sub(graph, passable, spike)
    passable = sub(graph, passable, green)
    passable = relu(graph, passable)
    passable = mul(graph, passable, lattice)
    plan_passable = mul(graph, passable, safe)

    fall_pair = cat(graph, fall_open, fall_open)
    goal_grid = mul(graph, goal, lattice)
    goal_pair = cat(graph, goal_grid, goal_grid)
    fall_destination = add(graph, fall_pair, goal_pair)
    fall_destination = vertical(graph, fall_destination)
    fall_destination = minimum(graph, fall_destination, one)
    supported = sub(graph, one, fall_destination)
    supported = mul(graph, supported, lattice)

    # Progress and screen-edge exploration targets.
    par("progress_w", np.array([[[[-30.0]]], [[[30.0]]]], np.float32))
    par("progress_offset_w", [[30.0], [-30.0]])
    par("progress_offset_b", [10.0, 10.0])
    progress = conv(graph, row, "progress_w")
    progress = add(
        graph, progress,
        dense(graph, player_row, "progress_offset_w", "progress_offset_b")
    )

    par("frontier_w", np.array([[[[-400.0]]], [[[400.0]]]], np.float32))
    par("frontier_b", [400.0 * 7.0 / 63.0, -400.0 * 56.0 / 63.0])
    frontier = emit(
        graph, "sigmoid",
        [conv(graph, row, "frontier_w", "frontier_b")]
    )
    par(
        "frontier_scale",
        np.array([[[[70.0]], [[0.0]]], [[[0.0]], [[70.0]]]], np.float32)
    )
    frontier = conv(graph, frontier, "frontier_scale")
    seed = relu(graph, add(graph, progress, frontier))
    seed = mul(graph, seed, supported)

    par(
        "goal_scale",
        np.array([[[[200.0]], [[0.0]]], [[[0.0]], [[200.0]]]], np.float32)
    )
    seed = add(graph, seed, conv(graph, goal_pair, "goal_scale"))
    seed = mul(graph, seed, plan_passable)

    par("brick_cost_w", [[[[1.3]]]])
    par("toggle_cost_w", [[[[1.5]]]])
    par("yellow_cost_w", [[[[3.5]]]])
    cost = add(graph, one, conv(graph, brick, "brick_cost_w"))
    cost = add(graph, cost, conv(graph, solid_toggle, "toggle_cost_w"))
    cost = add(graph, cost, conv(graph, yellow, "yellow_cost_w"))

    break_pair = cat(graph, destructible, destructible)
    next_break = vertical(graph, break_pair)

    # Horizontal planner shifts.
    w1 = np.zeros((4, 2, 7, 7), np.float32)
    w2 = np.zeros((4, 4, 7, 7), np.float32)
    for direction, dx in enumerate((-3, 3)):
        for channel in range(2):
            k = 2 * direction + channel
            w1[k, channel, 3, 3 + dx] = 1
            w2[k, k, 3, 3 + dx] = 1
    par("horizontal_first_w", w1)
    par("horizontal_second_w", w2)

    state, planner = fresh(), fresh()
    body = []
    entered = relu(body, sub(body, state, cost))
    h = conv(body, entered, "horizontal_first_w")
    h = conv(body, h, "horizontal_second_w")
    horizontal = maximum(body, sl(body, h, 0, 2), sl(body, h, 2, 4))
    breaking = vertical(body, entered)
    breaking = mul(body, breaking, next_break)
    supported_choice = maximum(body, horizontal, breaking)
    supported_choice = mul(body, supported_choice, supported)
    falling = vertical(body, state)
    falling = mul(body, falling, fall_destination)
    new_value = maximum(body, supported_choice, falling)
    new_value = maximum(body, new_value, seed)
    new_value = mul(body, new_value, plan_passable)
    graph.append({
        "id": planner,
        "op": "loop",
        "T": 48,
        "state": [{"id": state, "init": seed}],
        "body": body,
        "next": {state: new_value},
    })
    value = planner + "." + state

    par("sum_pair_w", [[[[1.0]], [[1.0]]]])
    active_value = mul(graph, value, active_pair)
    active_value = conv(graph, active_value, "sum_pair_w")
    base_value = gmax(graph, mul(graph, active_value, player))

    active_danger = mul(graph, danger, active_pair)
    active_danger = conv(graph, active_danger, "sum_pair_w")
    active_safe = sub(graph, one, active_danger)
    after_break_danger = mul(graph, danger_next, active_pair)
    after_break_danger = conv(graph, after_break_danger, "sum_pair_w")
    after_break_safe = sub(graph, one, after_break_danger)

    # Four adjacent player-cell markers and a reusable four-way dilation.
    w1 = np.zeros((4, 1, 7, 7), np.float32)
    w2 = np.zeros((4, 4, 7, 7), np.float32)
    directions = [(0, 3), (0, -3), (3, 0), (-3, 0)]
    for i, (dy, dx) in enumerate(directions):
        w1[i, 0, 3 + dy, 3 + dx] = 1
        w2[i, i, 3 + dy, 3 + dx] = 1
    par("marker_first_w", w1)
    par("marker_second_w", w2)
    par("sum_four_w", [[[[1.0]], [[1.0]], [[1.0]], [[1.0]]]])

    def adjacent(dst, x):
        a = conv(dst, x, "marker_first_w")
        return conv(dst, a, "marker_second_w")

    def dilate(dst, x):
        a = adjacent(dst, x)
        a = conv(dst, a, "sum_four_w")
        return minimum(dst, a, one)

    markers = adjacent(graph, player)
    left_marker = sl(graph, markers, 0, 1)
    right_marker = sl(graph, markers, 1, 2)
    up_marker = sl(graph, markers, 2, 3)
    down_marker = sl(graph, markers, 3, 4)
    forward_marker = add(
        graph,
        mul(graph, up_marker, normal),
        mul(graph, down_marker, reverse),
    )
    side_markers = maximum(graph, left_marker, right_marker)

    improvement = relu(graph, sub(graph, active_value, base_value))

    open_now = sub(graph, one, wall)
    open_now = sub(graph, open_now, brick)
    open_now = sub(graph, open_now, solid_toggle)
    open_now = sub(graph, open_now, color15)
    open_now = sub(graph, open_now, green)
    open_now = relu(graph, open_now)
    open_now = mul(graph, open_now, lattice)
    open_now = mul(graph, open_now, active_safe)

    left_valid = mul(graph, left_marker, open_now)
    right_valid = mul(graph, right_marker, open_now)
    left_gain = gmax(graph, mul(graph, improvement, left_valid))
    right_gain = gmax(graph, mul(graph, improvement, right_valid))

    local_candidates = mul(graph, destructible, lattice)
    local_candidates = mul(graph, local_candidates, after_break_safe)
    local_candidates = mul(
        graph, local_candidates, maximum(graph, side_markers, forward_marker)
    )
    par("local_bonus_w", [[[[1.0]]]])
    par("local_bonus_b", [0.25])
    local_value = conv(graph, improvement, "local_bonus_w", "local_bonus_b")
    local_click = mul(graph, local_value, local_candidates)
    local_score = gmax(graph, local_click)

    # A gravity flip is rejected if the opposite fall reaches a spike or
    # leaves the observed region without a known landing.
    both_at_player = gmax(graph, mul(graph, value, player))
    other_at_player = vmul(graph, both_at_player, opposite_pair)
    par("other_value_w", [[1.0, 1.0]])
    other_value = dense(graph, other_at_player, "other_value_w")
    neg_base = dense(graph, base_value, "neg_v_w")
    flip_gain = vadd(graph, other_value, neg_base)
    par("flip_penalty_w", [[-1.5]])
    par("flip_penalty_b", [-5.0])
    flip_gain = vadd(
        graph, flip_gain,
        dense(graph, cooldown, "flip_penalty_w", "flip_penalty_b")
    )
    flip_gain = emit(graph, "vrelu", [flip_gain])

    opposite_uncertain = mul(graph, uncertain, opposite_pair)
    opposite_uncertain = conv(graph, opposite_uncertain, "sum_pair_w")
    flip_unsafe = gmax(graph, mul(graph, opposite_uncertain, player))
    par("safe_v_w", [[-1.0]])
    par("safe_v_b", [1.0])
    flip_safe = dense(graph, flip_unsafe, "safe_v_w", "safe_v_b")
    flip_gain = vmul(graph, flip_gain, flip_safe)
    green_grid = mul(graph, green, lattice)
    green_click = mul(graph, green_grid, flip_gain)
    green_score = gmax(graph, green_click)

    meaningful = vcat(graph, left_gain, right_gain, local_score, green_score)
    par("fallback_gate_w", [[-1.0, -1.0, -1.0, -1.0]])
    par("fallback_gate_b", [1.0])
    fallback_gate = dense(
        graph, meaningful, "fallback_gate_w", "fallback_gate_b"
    )
    fallback_gate = emit(graph, "vrelu", [fallback_gate])

    # Find dangerous horizontal entries and the cell immediately beneath
    # them in the active gravity direction.
    danger_entry = mul(graph, active_danger, side_markers)
    danger_entry = mul(graph, danger_entry, open_now if False else lattice)
    danger_entry = mul(graph, danger_entry, sub(graph, one, wall))
    entry_pair = cat(graph, danger_entry, danger_entry)
    support_pair = vertical_reverse(graph, entry_pair)
    support_pair = mul(graph, support_pair, active_pair)
    support_target = conv(graph, support_pair, "sum_pair_w")
    support_target = minimum(graph, support_target, one)
    support_target = mul(graph, support_target, lattice)

    # A dotted toggle can be made solid remotely to create a landing.
    support_toggle = mul(graph, support_target, dotted_toggle)
    support_toggle = mul(graph, support_toggle, fallback_gate)
    par("toggle_support_scale", [[[[1.5]]]])
    support_toggle = conv(graph, support_toggle, "toggle_support_scale")

    # Clicking a yellow block creates yellow blocks in empty neighboring
    # cells. Prefer a source next to a needed support cell.
    empty_target = mul(graph, support_target, empty_color)
    yellow_sources = dilate(graph, empty_target)
    yellow_sources = mul(graph, yellow_sources, yellow)
    yellow_sources = mul(graph, yellow_sources, lattice)
    yellow_sources = mul(graph, yellow_sources, fallback_gate)

    # Do not remove the block immediately in the gravity direction when
    # that would expose the player to a known spike.
    unsafe_forward = mul(graph, forward_marker, after_break_danger)
    allowed_source = sub(graph, one, unsafe_forward)
    yellow_sources = mul(graph, yellow_sources, allowed_source)
    par("yellow_support_scale", [[[[1.2]]]])
    yellow_sources = conv(graph, yellow_sources, "yellow_support_scale")

    # A weak ordinary-block fallback is preferable to repeating an invalid
    # movement, but is subordinate to safe movement and support creation.
    par("brick_fallback_w", [[[[0.2]]]])
    brick_fallback = conv(graph, brick, "brick_fallback_w")
    brick_fallback = mul(graph, brick_fallback, lattice)
    brick_fallback = mul(graph, brick_fallback, fallback_gate)
    brick_fallback = mul(graph, brick_fallback, allowed_source)

    click_map = maximum(graph, local_click, green_click)
    click_map = maximum(graph, click_map, support_toggle)
    click_map = maximum(graph, click_map, yellow_sources)
    click_map = maximum(graph, click_map, brick_fallback)
    graph.append({"id": "click", "op": "relu", "in": [click_map]})
    click_score = gmax(graph, "click")

    par("default_move_w", [[[[0.5]]]])
    left_default = gmax(graph, conv(graph, left_valid, "default_move_w"))
    right_default = gmax(graph, conv(graph, right_valid, "default_move_w"))
    left_score = vadd(graph, left_gain, left_default)
    right_score = vadd(graph, right_gain, right_default)

    features = vcat(graph, left_score, right_score, click_score)
    action_w = np.zeros((8, 3), np.float32)
    action_w[3, 0] = 1
    action_w[4, 1] = 1
    action_w[6, 2] = 1
    action_b = np.full(8, -1000.0, np.float32)
    action_b[3] = action_b[4] = action_b[6] = 0
    par("action_w", action_w)
    par("action_b", action_b)
    action_logits = dense(graph, features, "action_w", "action_b")

    previous_w = np.zeros((8, 8), np.float32)
    previous_w[3, 4] = -0.05
    previous_w[4, 3] = -0.05
    par("previous_action_w", previous_w)
    previous_penalty = dense(graph, "aprev", "previous_action_w")
    graph.append({
        "id": "logits",
        "op": "vadd",
        "in": [action_logits, previous_penalty],
    })

    return {"graph": graph, "memory_size": 2, "params": params}

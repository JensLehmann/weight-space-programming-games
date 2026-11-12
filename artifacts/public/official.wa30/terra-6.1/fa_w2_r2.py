import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]
    constants = {}

    def name(prefix):
        counter[0] += 1
        return prefix + str(counter[0])

    def parameter(value):
        p = name("p")
        params[p] = np.asarray(value, dtype=np.float32)
        return p

    def node(op, inputs, body=None, **fields):
        n = name("n")
        item = {"id": n, "op": op, "in": list(inputs)}
        item.update(fields)
        (graph if body is None else body).append(item)
        return n

    def conv(x, w, b=None, body=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], body=body, **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def scalar(value):
        value = float(value)
        if value not in constants:
            constants[value] = dense(
                "aprev",
                np.zeros((1, 8), dtype=np.float32),
                np.array([value], dtype=np.float32),
            )
        return constants[value]

    def sl(x, start, stop, body=None):
        return node("slice", [x], body=body, start=start, stop=stop)

    def add(x, y):
        return node("add", [x, y])

    def sub(x, y):
        return node("sub", [x, y])

    def mul(x, y):
        return node("mul", [x, y])

    def maximum(x, y):
        return node("max2", [x, y])

    def vadd(x, y):
        return node("vadd", [x, y])

    def vmul(x, y):
        return node("vmul", [x, y])

    def spatial_affine(x, channels, scale, bias=0.0):
        w = np.zeros((channels, channels, 1, 1), dtype=np.float32)
        for i in range(channels):
            w[i, i, 0, 0] = scale
        return conv(x, w, np.full(channels, bias, dtype=np.float32))

    def vector_affine(x, scale, bias=0.0):
        if scale != 1.0:
            x = vmul(x, scalar(scale))
        if bias != 0.0:
            x = vadd(x, scalar(bias))
        return x

    def spatial_clip(x):
        return node("min2", [node("relu", [x]), one])

    def vector_clip(x):
        positive = node("vrelu", [x])
        excess = node("vrelu", [vector_affine(x, 1.0, -1.0)])
        return vadd(positive, vector_affine(excess, -1.0))

    def sum_channels(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), dtype=np.float32))

    def max4(x, body=None):
        a = sl(x, 0, 1, body)
        b = sl(x, 1, 2, body)
        c = sl(x, 2, 3, body)
        d = sl(x, 3, 4, body)
        ab = node("max2", [a, b], body=body)
        cd = node("max2", [c, d], body=body)
        return node("max2", [ab, cd], body=body)

    def max_list(xs):
        result = xs[0]
        for x in xs[1:]:
            result = maximum(result, x)
        return result

    one = sl("coords", 2, 3)
    colors = [sl("x0", i, i + 1) for i in range(16)]

    # Direction order: up, down, left, right.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    wa = np.zeros((4, 1, 5, 5), dtype=np.float32)
    wb = np.zeros((4, 4, 5, 5), dtype=np.float32)
    for k, (dy, dx) in enumerate(directions):
        wa[k, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        wb[k, k, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    shift_a = parameter(wa)
    shift_b = parameter(wb)
    cost = parameter(np.full(4, -1.0, dtype=np.float32))

    def shift4(x):
        a = node("conv", [x], w=shift_a)
        return node("conv", [a], w=shift_b)

    def wave(seed, free, carry_bonus=None):
        loop = name("loop")
        state = name("state")
        body = []
        a = node("conv", [state], body=body, w=shift_a)
        b = node("conv", [a], body=body, w=shift_b, b=cost)
        if carry_bonus is not None:
            b = node("add", [b, carry_bonus], body=body)
        neighbors = max4(b, body=body)
        best = node("max2", [state, neighbors], body=body)
        nxt = node("mul", [best, free], body=body)
        graph.append({
            "id": loop,
            "op": "loop",
            "T": 128,
            "state": [{"id": state, "init": seed}],
            "body": body,
            "next": {state: nxt},
        })
        return loop + "." + state

    # Player anchor.
    wp = np.zeros((1, 1, 5, 5), dtype=np.float32)
    wp[0, 0, 1:5, 1:5] = 1.0
    player_count = conv(colors[14], wp)
    player_full = spatial_clip(
        spatial_affine(player_count, 1, 2.0, -23.0)
    )

    # Row 63 is hidden by the step counter.
    row_delta = conv(
        sl("coords", 0, 1),
        np.array([[[[63.0]]]], dtype=np.float32),
        np.array([-61.0], dtype=np.float32),
    )
    bottom_anchor = spatial_clip(
        spatial_affine(node("abs", [row_delta]), 1, -1.0, 1.0)
    )
    player_partial = spatial_clip(
        spatial_affine(player_count, 1, 2.0, -15.0)
    )
    player = maximum(player_full, mul(player_partial, bottom_anchor))

    # Crate and crate-holder border detectors.
    wc = np.zeros((1, 16, 5, 5), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            if r in (1, 2) and c in (1, 2):
                wc[0, 9, r + 1, c + 1] = 1.0
            else:
                for color in (0, 3, 4, 5):
                    wc[0, color, r + 1, c + 1] = 1.0
    crate_raw = conv("x0", wc, np.array([-11.5], dtype=np.float32))
    crate = spatial_clip(
        spatial_affine(node("relu", [crate_raw]), 1, 2.0)
    )

    wborder = np.zeros((2, 16, 5, 5), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            if r in (0, 3) or c in (0, 3):
                wborder[0, 0, r + 1, c + 1] = 1.0
                wborder[1, 5, r + 1, c + 1] = 1.0
    borders = conv(
        "x0", wborder, np.array([-7.5, -7.5], dtype=np.float32)
    )
    borders = node("relu", [borders])
    black_border = spatial_clip(
        spatial_affine(sl(borders, 0, 1), 1, 2.0)
    )
    grey_border = spatial_clip(
        spatial_affine(sl(borders, 1, 2), 1, 2.0)
    )
    own_crate = mul(crate, black_border)
    npc_crate = mul(crate, grey_border)
    other_crates = sub(crate, own_crate)

    # Hatched barrier cells.
    wh = np.zeros((1, 16, 5, 5), dtype=np.float32)
    hatch = [
        [2, 1, 2, 2],
        [1, 2, 2, 2],
        [2, 2, 2, 1],
        [2, 2, 1, 2],
    ]
    for r in range(4):
        for c in range(4):
            wh[0, hatch[r][c], r + 1, c + 1] = 1.0
    hazard_raw = conv("x0", wh, np.array([-15.5], dtype=np.float32))
    hazard = spatial_clip(
        spatial_affine(node("relu", [hazard_raw]), 1, 2.0)
    )

    # Recognize a crate covering a straight section of hatched barrier.
    hazard_neighbors = shift4(hazard)
    hazard_vertical = mul(
        sl(hazard_neighbors, 0, 1), sl(hazard_neighbors, 1, 2)
    )
    hazard_horizontal = mul(
        sl(hazard_neighbors, 2, 3), sl(hazard_neighbors, 3, 4)
    )
    covered_barrier = mul(
        crate, maximum(hazard_vertical, hazard_horizontal)
    )

    # Grow visible goal regions through their 9/2 pixels.
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 1, 1] = 1.0
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 2, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0
    cross_w = parameter(cross)

    near_two = node("conv", [colors[2]], w=cross_w)
    near_two = node("min2", [near_two, one])
    target_seed = mul(colors[9], near_two)
    target_material = add(colors[2], colors[9])

    flood = name("loop")
    flood_state = name("state")
    body = []
    neighbors = node("conv", [flood_state], body=body, w=cross_w)
    neighbors = node("min2", [neighbors, one], body=body)
    flood_next = node("mul", [neighbors, target_material], body=body)
    graph.append({
        "id": flood,
        "op": "loop",
        "T": 64,
        "state": [{"id": flood_state, "init": target_seed}],
        "body": body,
        "next": {flood_state: flood_next},
    })
    visible_target = flood + "." + flood_state

    # Infer some covered goal cells from adjacent goal interior.
    target_interior = mul(visible_target, colors[2])
    wi = np.zeros((1, 1, 7, 7), dtype=np.float32)
    wi[0, 0, 1, 3] = 1.0
    wi[0, 0, 6, 3] = 1.0
    wi[0, 0, 3, 1] = 1.0
    wi[0, 0, 3, 6] = 1.0
    inferred = spatial_clip(conv(target_interior, wi))
    observed_goal = maximum(visible_target, mul(crate, inferred))
    observed_goal = mul(observed_goal, sub(one, hazard))

    # One-hot masks for 256 grid-cell anchors.
    row_w = np.zeros((16, 3, 1, 1), dtype=np.float32)
    col_w = np.zeros((16, 3, 1, 1), dtype=np.float32)
    row_w[:, 0, 0, 0] = 63.0
    col_w[:, 1, 0, 0] = 63.0
    axis_bias = -np.asarray(
        [4 * i + 1 for i in range(16)], dtype=np.float32
    )
    row_dist = node("abs", [conv("coords", row_w, axis_bias)])
    col_dist = node("abs", [conv("coords", col_w, axis_bias)])
    rows = node(
        "relu", [spatial_affine(row_dist, 16, -1.0, 1.0)]
    )
    cols = node(
        "relu", [spatial_affine(col_dist, 16, -1.0, 1.0)]
    )
    axes = node("concat", [rows, cols])
    grid_w = np.zeros((256, 32, 1, 1), dtype=np.float32)
    for r in range(16):
        for c in range(16):
            i = 16 * r + c
            grid_w[i, r, 0, 0] = 1.0
            grid_w[i, 16 + c, 0, 0] = 1.0
    grid_mask = node(
        "relu",
        [conv(axes, grid_w, np.full(256, -1.0, dtype=np.float32))],
    )

    newly_seen = node("gmax", [mul(grid_mask, observed_goal)])
    newly_seen = vector_clip(vector_affine(newly_seen, 2.0, -1.0))
    goal_memory = vector_clip(vadd("mem", newly_seen))
    graph.append({
        "id": "mem_next",
        "op": "vadd",
        "in": [goal_memory, scalar(0.0)],
    })
    remembered = sum_channels(mul(grid_mask, goal_memory), 256)
    goal = spatial_clip(spatial_affine(remembered, 1, 2.0, -1.0))
    crate_on_goal = mul(crate, goal)

    # Goal distance, with occupied goals excluded.
    available_goal = mul(goal, sub(one, other_crates))
    crate_free = sub(one, colors[5])
    goal_seed = mul(spatial_affine(available_goal, 1, 180.0), crate_free)
    goal_value = wave(goal_seed, crate_free)

    # Identify the carried crate by its black border and offset.
    carry_dirs = node("gmax", [mul(shift4(own_crate), player)])
    carrying = dense(carry_dirs, np.ones((1, 4), dtype=np.float32))
    carrying = vector_clip(carrying)
    not_carrying = vector_affine(carrying, -1.0, 1.0)

    base_block = max_list([
        colors[5], colors[12], colors[15], other_crates
    ])
    player_block = maximum(base_block, hazard)
    seek_free = sub(one, player_block)

    shifted_block = shift4(base_block)
    held_block = sum_channels(mul(shifted_block, carry_dirs), 4)
    carry_free = sub(one, maximum(player_block, held_block))
    free = add(
        mul(seek_free, not_carrying),
        mul(carry_free, carrying),
    )

    # Eligible crates and purple rivals.
    eligible = mul(crate, sub(one, crate_on_goal))
    eligible = mul(eligible, sub(one, npc_crate))
    eligible = mul(eligible, sub(one, own_crate))
    eligible = mul(eligible, sub(one, covered_barrier))
    crate_objective = mul(eligible, goal_value)
    rival_objective = spatial_affine(colors[15], 1, 230.0)
    objective = maximum(crate_objective, rival_objective)
    objective_neighbors = shift4(objective)
    seek_seed = mul(max4(objective_neighbors), seek_free)

    # Carrying potential: prefer positions that bring the crate closer
    # to a free goal, even if the player cannot reach the goal itself.
    carry_potential = sum_channels(mul(shift4(goal_value), carry_dirs), 4)
    carry_seed = mul(carry_potential, carry_free)
    seed = add(
        mul(seek_seed, not_carrying),
        mul(carry_seed, carrying),
    )
    carry_bonus = vector_affine(carrying, 0.75)
    value = wave(seed, free, carry_bonus=carry_bonus)

    shifted_value = shift4(value)
    move_values = node("gmax", [mul(shifted_value, player)])
    current_value = node("gmax", [mul(value, player)])
    best_move = node("gmax", [mul(max4(shifted_value), player)])

    # Explicitly interact with the best adjacent undelivered object.
    adjacent_values = node("gmax", [mul(objective_neighbors, player)])
    best_adjacent = node(
        "gmax", [mul(max4(objective_neighbors), player)]
    )
    comparison = vadd(
        adjacent_values, vector_affine(best_adjacent, -1.0, 0.5)
    )
    good_adjacent = vector_clip(vector_affine(comparison, 2.0))
    good_adjacent = vmul(good_adjacent, vector_clip(adjacent_values))
    good_adjacent = vmul(good_adjacent, not_carrying)

    # Read the player's black facing edge.
    wheads = np.zeros((4, 1, 5, 5), dtype=np.float32)
    for q in range(4):
        wheads[0, 0, 1, q + 1] = 1.0
        wheads[1, 0, 4, q + 1] = 1.0
        wheads[2, 0, q + 1, 1] = 1.0
        wheads[3, 0, q + 1, 4] = 1.0
    head_counts = conv(colors[0], wheads)
    head_full = spatial_clip(
        spatial_affine(head_counts, 4, 2.0, -7.0)
    )
    head_partial = spatial_clip(
        spatial_affine(head_counts, 4, 2.0, -5.0)
    )
    heads = maximum(head_full, mul(head_partial, bottom_anchor))
    facing_visual = node("gmax", [mul(heads, player)])
    previous_direction = node("vslice", ["aprev"], start=1, stop=5)
    facing = vector_clip(vadd(facing_visual, previous_direction))
    ready = dense(
        vmul(good_adjacent, facing),
        np.ones((1, 4), dtype=np.float32),
    )

    # Release on a goal or at the best reachable carrying position.
    own_on_goal = node("gmax", [mul(own_crate, goal)])
    local_stop = vector_clip(
        vector_affine(
            vadd(current_value, vector_affine(best_move, -1.0)),
            4.0,
            0.2,
        )
    )
    drop = vmul(vector_clip(vadd(own_on_goal, local_stop)), carrying)
    interact = vadd(ready, drop)

    # A small reversal penalty breaks otherwise equal movement ties.
    reverse_w = np.zeros((4, 8), dtype=np.float32)
    reverse_w[0, 2] = -0.05
    reverse_w[1, 1] = -0.05
    reverse_w[2, 4] = -0.05
    reverse_w[3, 3] = -0.05
    reversal = dense("aprev", reverse_w)

    moves = vadd(move_values, vector_affine(good_adjacent, 300.0))
    moves = vadd(moves, reversal)
    features = node("vconcat", [moves, interact])
    wo = np.zeros((8, 5), dtype=np.float32)
    for i in range(4):
        wo[i + 1, i] = 1.0
    wo[5, 4] = 1000.0
    bo = np.array(
        [-10000.0, 0.0, 0.0, 0.0, 0.0, -10.0, -10000.0, -10000.0],
        dtype=np.float32,
    )
    logits = dense(features, wo, bo)
    graph.append({
        "id": "logits",
        "op": "vadd",
        "in": [logits, scalar(0.0)],
    })

    zero_click = spatial_affine(one, 1, 0.0)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [zero_click, zero_click],
    })

    return {"graph": graph, "memory_size": 256, "params": params}

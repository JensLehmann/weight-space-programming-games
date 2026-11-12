import numpy as np


def build():
    graph = []
    params = {}
    count = [0]

    def fresh(prefix):
        count[0] += 1
        return "%s_%d" % (prefix, count[0])

    def param(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = fresh("n")
        graph.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def unary(op, x):
        return node(op, [x])

    def binary(op, a, b):
        return node(op, [a, b])

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

    def ss(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def vcat(*xs):
        return node("vconcat", xs)

    one = ss("coords", 2, 3)

    def clip(x):
        return binary("min2", unary("relu", x), one)

    def vclip(x, n):
        x = unary("vrelu", x)
        x = dense(x, -np.eye(n, dtype=np.float32),
                  np.ones(n, dtype=np.float32))
        x = unary("vrelu", x)
        return dense(x, -np.eye(n, dtype=np.float32),
                     np.ones(n, dtype=np.float32))

    def vnot(x, n=1):
        return dense(x, -np.eye(n, dtype=np.float32),
                     np.ones(n, dtype=np.float32))

    def sum_channels(x, n):
        return conv(x, np.ones((1, n, 1, 1), dtype=np.float32))

    # Playable area and static walls.
    bounds_w = np.zeros((4, 3, 1, 1), dtype=np.float32)
    bounds_w[0, 0, 0, 0] = 126
    bounds_w[1, 0, 0, 0] = -126
    bounds_w[2, 1, 0, 0] = 126
    bounds_w[3, 1, 0, 0] = -126
    bounds = clip(conv(
        "coords", bounds_w,
        np.array([-13, 105, -21, 113], dtype=np.float32)
    ))
    rect = ss(bounds, 0, 1)
    for i in range(1, 4):
        rect = binary("min2", rect, ss(bounds, i, i + 1))

    wall = ss("x0", 4, 5)
    black = ss("x0", 5, 6)
    blue = ss("x0", 11, 12)
    base_free = binary("mul", rect, binary("sub", one, wall))
    position_planes = ss("coords", 0, 2)

    # Player body.
    pw = np.zeros((1, 16, 5, 5), dtype=np.float32)
    pw[0, 12, :2, :] = 2
    pw[0, 9, 2:, :] = 2
    player = clip(conv("x0", pw, np.array([-49], dtype=np.float32)))
    player_pos = unary("gmax", binary("mul", position_planes, player))

    # Glyph templates.
    colors = [12, 9, 14, 8]
    ow = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for color in colors:
        ow[0, color, 0, 0] = 1
    occupied = conv("x0", ow)

    shapes = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]
    patterns = []
    for shape in shapes:
        a = np.asarray(shape, dtype=np.float32)
        for r in range(4):
            patterns.append(np.rot90(a, -r).copy())

    gw = np.zeros((24, 1, 3, 3), dtype=np.float32)
    gb = np.zeros(24, dtype=np.float32)
    for i, pat in enumerate(patterns):
        gw[i, 0] = 2 * (2 * pat - 1)
        gb[i] = 1 - 2 * pat.sum()
    glyphs = clip(conv(occupied, gw, gb))

    # Read the scaled HUD icon.
    sample = [0, 2, 4]
    iw = np.zeros((24, 1, 7, 7), dtype=np.float32)
    ib = np.zeros(24, dtype=np.float32)
    for i, pat in enumerate(patterns):
        for r in range(3):
            for c in range(3):
                iw[i, 0, sample[r], sample[c]] = (
                    2 if pat[r, c] else -2
                )
        ib[i] = 1 - 2 * pat.sum()
    icons = clip(conv(occupied, iw, ib))

    point_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    point_w[0, 0, 0, 0] = 63
    point_w[1, 1, 0, 0] = 63
    point_delta = unary("abs", conv(
        "coords", point_w, np.array([-58, -6], dtype=np.float32)
    ))
    point = unary("relu", conv(
        point_delta,
        np.full((1, 2, 1, 1), -2, dtype=np.float32),
        np.array([1], dtype=np.float32)
    ))
    icon = unary("gmax", binary("mul", icons, point))

    icw = np.zeros((4, 16, 7, 7), dtype=np.float32)
    for i, color in enumerate(colors):
        for r in sample:
            for c in sample:
                icw[i, color, r, c] = 2
    icon_colors = clip(conv(
        "x0", icw, -np.ones(4, dtype=np.float32)
    ))
    icon_color = unary("gmax", binary("mul", icon_colors, point))

    shape_matrix = np.zeros((6, 24), dtype=np.float32)
    rotation_matrix = np.zeros((4, 24), dtype=np.float32)
    shape_expand = np.zeros((24, 6), dtype=np.float32)
    rotation_expand = np.zeros((24, 4), dtype=np.float32)
    for i in range(24):
        shape_matrix[i // 4, i] = 1
        rotation_matrix[i % 4, i] = 1
        shape_expand[i, i // 4] = 1
        rotation_expand[i, i % 4] = 1
    icon_shape = dense(icon, shape_matrix)
    icon_rotation = dense(icon, rotation_matrix)

    # Goal frame and goal attributes.
    ring_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    ring_w[0, 0, 0, :] = 1
    ring_w[0, 0, 6, :] = 1
    ring_w[0, 0, 1:6, 0] = 1
    ring_w[0, 0, 1:6, 6] = 1
    ring = clip(conv(
        black, ring_w, np.array([-17.5], dtype=np.float32)
    ))
    goal_glyphs = binary("mul", glyphs, ring)
    any_goal = clip(sum_channels(goal_glyphs, 24))

    gcw = np.zeros((4, 16, 3, 3), dtype=np.float32)
    for i, color in enumerate(colors):
        gcw[i, color, :, :] = 2
    goal_colors = clip(conv(
        "x0", gcw, -np.ones(4, dtype=np.float32)
    ))
    goal_shape = clip(sum_channels(binary(
        "mul", goal_glyphs, dense(icon_shape, shape_expand)
    ), 24))
    goal_rotation = clip(sum_channels(binary(
        "mul", goal_glyphs, dense(icon_rotation, rotation_expand)
    ), 24))
    goal_color = clip(sum_channels(binary(
        "mul", goal_colors, icon_color
    ), 4))
    shape_color = binary("min2", goal_shape, goal_color)
    exact_goal = binary("min2", shape_color, goal_rotation)

    has_shape = unary("gmax", goal_shape)
    has_shape_color = unary("gmax", shape_color)
    has_exact = unary("gmax", exact_goal)

    # Shape, color and rotation switches.
    switch_patterns = [
        [[0, 3, 3], [3, 0, 0], [3, 0, 3]],
        [[9, 14, 14], [9, 0, 8], [12, 12, 8]],
        [[3, 0, 3], [1, 0, 0], [3, 1, 3]],
    ]
    sw = np.zeros((3, 16, 3, 3), dtype=np.float32)
    for i, pat in enumerate(switch_patterns):
        for r in range(3):
            for c in range(3):
                sw[i, pat[r][c], r, c] = 2
    switches = clip(conv(
        "x0", sw, -17 * np.ones(3, dtype=np.float32)
    ))

    # Memory layout:
    # icon 0:24, color 24:28, player position 28:30,
    # covered-switch identity 30:33, active 33, timer 34,
    # switch positions 35:41, switch visibility 41:44.
    old_icon = vs("mem", 0, 24)
    old_color = vs("mem", 24, 28)
    old_pos = vs("mem", 28, 30)
    old_switch = vs("mem", 30, 33)
    active = vs("mem", 33, 34)
    old_timer = vs("mem", 34, 35)
    old_switch_pos = vs("mem", 35, 41)
    old_switch_seen = vs("mem", 41, 44)

    old_shape = dense(old_icon, shape_matrix)
    old_rotation = dense(old_icon, rotation_matrix)

    def equal_onehot(a, b, n):
        return dense(
            binary("vmul", a, b),
            np.ones((1, n), dtype=np.float32)
        )

    changed = vcat(
        vnot(equal_onehot(icon_shape, old_shape, 6)),
        vnot(equal_onehot(icon_color, old_color, 4)),
        vnot(equal_onehot(icon_rotation, old_rotation, 4)),
    )
    changed = binary("vmul", changed, active)

    diff_w = np.zeros((2, 4), dtype=np.float32)
    diff_w[0, 0] = 63
    diff_w[0, 2] = -63
    diff_w[1, 1] = 63
    diff_w[1, 3] = -63
    diff = dense(vcat(player_pos, old_pos), diff_w)
    diff_pos = unary("vrelu", diff)
    diff_neg = unary("vrelu", dense(
        diff, -np.eye(2, dtype=np.float32)
    ))
    diff_abs = binary("vadd", diff_pos, diff_neg)
    moved = vclip(dense(
        diff_abs, np.ones((1, 2), dtype=np.float32)
    ), 1)
    retained = binary("vmul", old_switch, vnot(moved))
    remembered_switch = vclip(binary("vadd", changed, retained), 3)
    switches = binary("max2", switches, binary(
        "mul", player, remembered_switch
    ))
    switches = binary("mul", switches, rect)

    need_shape = vnot(has_shape)
    need_color = unary("vrelu", binary(
        "vadd", has_shape,
        dense(has_shape_color, np.array([[-1]], dtype=np.float32))
    ))
    need_rotation = unary("vrelu", binary(
        "vadd", has_shape_color,
        dense(has_exact, np.array([[-1]], dtype=np.float32))
    ))

    rw = np.ones((1, 1, 3, 3), dtype=np.float32)
    rw[0, 0, 1, 1] = 0
    recharge = clip(conv(
        blue, 2 * rw, np.array([-15], dtype=np.float32)
    ))
    recharge = binary("mul", recharge, rect)

    # Visible pusher lines, in direction order up/down/left/right.
    pusher_w = np.zeros((4, 16, 7, 7), dtype=np.float32)
    pusher_w[0, 1, 6, 1:6] = 2
    pusher_w[1, 1, 0, 1:6] = 2
    pusher_w[2, 1, 1:6, 6] = 2
    pusher_w[3, 1, 1:6, 0] = 2
    pushers = clip(conv(
        "x0", pusher_w, -9 * np.ones(4, dtype=np.float32)
    ))
    pushers = binary("mul", pushers, base_free)
    any_pusher = clip(sum_channels(pushers, 4))
    no_pusher = binary("sub", one, any_pusher)

    locked_goal = unary("relu", binary("sub", any_goal, exact_goal))
    nav_free = binary(
        "mul", base_free, binary("sub", one, locked_goal)
    )
    push_free = binary(
        "mul", base_free, binary("sub", one, any_goal)
    )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shift3 = np.zeros((4, 1, 7, 7), dtype=np.float32)
    shift2 = np.zeros((4, 4, 5, 5), dtype=np.float32)
    group3 = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(directions):
        shift3[i, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        shift2[i, i, 2 + 2 * dy, 2 + 2 * dx] = 1
        group3[i, i, 3 + 3 * dy, 3 + 3 * dx] = 1

    shift3_p = param(shift3)
    shift2_p = param(shift2)
    group3_p = param(group3)
    sum4_p = param(np.ones((1, 4, 1, 1), dtype=np.float32))
    dec_w = param(np.ones((1, 1, 1, 1), dtype=np.float32))
    dec_b = param(np.array([-1], dtype=np.float32))

    def shift_four(x):
        a = node("conv", [x], w=shift3_p)
        return node("conv", [a], w=shift2_p)

    # Predict one step of motion for each visible switch. A stationary
    # switch keeps its current target cell.
    switch_positions = []
    switch_seen = []
    for i in range(3):
        s = ss(switches, i, i + 1)
        switch_positions.append(unary(
            "gmax", binary("mul", position_planes, s)
        ))
        switch_seen.append(unary("gmax", s))
    switch_pos_vector = vcat(*switch_positions)
    switch_seen_vector = vcat(*switch_seen)

    motion_w = np.zeros((12, 12), dtype=np.float32)
    for i in range(3):
        row = 2 * i
        col = row + 1
        old_row = 6 + row
        old_col = 6 + col
        for j, coordinate, old_coordinate, sign in [
            (0, row, old_row, -1),
            (1, row, old_row, 1),
            (2, col, old_col, -1),
            (3, col, old_col, 1),
        ]:
            motion_w[4 * i + j, coordinate] = 12.6 * sign
            motion_w[4 * i + j, old_coordinate] = -12.6 * sign
    motion = vclip(dense(
        vcat(switch_pos_vector, old_switch_pos), motion_w
    ), 12)

    predicted = []
    opposite = [1, 0, 3, 2]
    for i in range(3):
        seen_both = binary(
            "vmul", vs(switch_seen_vector, i, i + 1),
            vs(old_switch_seen, i, i + 1)
        )
        seen_both = binary("vmul", seen_both, active)
        m = binary("vmul", vs(motion, 4 * i, 4 * i + 4), seen_both)
        moving = vclip(dense(
            m, np.ones((1, 4), dtype=np.float32)
        ), 1)
        shifted = shift_four(ss(switches, i, i + 1))
        reorder = np.zeros((4, 4), dtype=np.float32)
        for j in range(4):
            reorder[opposite[j], j] = 1
        selector = dense(m, reorder)
        advanced = sum_channels(binary("mul", shifted, selector), 4)
        stationary = binary(
            "mul", ss(switches, i, i + 1), vnot(moving)
        )
        future = clip(binary("add", advanced, stationary))
        predicted.append(binary("mul", future, rect))

    target = exact_goal
    target = binary("max2", target, binary(
        "mul", predicted[0], need_shape
    ))
    target = binary("max2", target, binary(
        "mul", predicted[1], need_color
    ))
    target = binary("max2", target, binary(
        "mul", predicted[2], need_rotation
    ))
    target = binary("mul", target, rect)

    # Pusher endpoints are open cells immediately before a stopping wall.
    ahead_free = shift_four(push_free)
    endpoints = binary(
        "mul", push_free, binary("sub", one, ahead_free)
    )

    def routing(seed_map):
        seed = conv(
            seed_map, np.array([[[[128]]]], dtype=np.float32)
        )
        loop_id = fresh("route")
        value_state = fresh("value")
        push_state = fresh("pushvalue")

        normal = fresh("normal")
        selected_push = fresh("selected")
        selected_sum = fresh("sum")
        effective = fresh("effective")
        sh3 = fresh("shift3")
        sh2 = fresh("shift2")
        ds = [fresh("direction") for _ in range(4)]
        m1, m2, m3 = fresh("max"), fresh("max"), fresh("max")
        dec = fresh("dec")
        rel = fresh("relu")
        passable = fresh("pass")
        new_value = fresh("newvalue")
        ps3 = fresh("pushshift3")
        ps2 = fresh("pushshift2")
        propagated = fresh("propagated")
        endpoint_value = fresh("endpoint")
        new_push = fresh("newpush")

        body = [
            {"id": normal, "op": "mul",
             "in": [value_state, no_pusher]},
            {"id": selected_push, "op": "mul",
             "in": [push_state, pushers]},
            {"id": selected_sum, "op": "conv",
             "in": [selected_push], "w": sum4_p},
            {"id": effective, "op": "add",
             "in": [normal, selected_sum]},
            {"id": sh3, "op": "conv",
             "in": [effective], "w": shift3_p},
            {"id": sh2, "op": "conv",
             "in": [sh3], "w": shift2_p},
        ]
        for i in range(4):
            body.append({
                "id": ds[i], "op": "slice", "in": [sh2],
                "start": i, "stop": i + 1
            })
        body.extend([
            {"id": m1, "op": "max2", "in": [ds[0], ds[1]]},
            {"id": m2, "op": "max2", "in": [ds[2], ds[3]]},
            {"id": m3, "op": "max2", "in": [m1, m2]},
            {"id": dec, "op": "conv", "in": [m3],
             "w": dec_w, "b": dec_b},
            {"id": rel, "op": "relu", "in": [dec]},
            {"id": passable, "op": "mul", "in": [rel, nav_free]},
            {"id": new_value, "op": "max2", "in": [seed, passable]},
            {"id": ps3, "op": "conv",
             "in": [push_state], "w": group3_p},
            {"id": ps2, "op": "conv",
             "in": [ps3], "w": shift2_p},
            {"id": propagated, "op": "mul",
             "in": [ps2, push_free]},
            {"id": endpoint_value, "op": "mul",
             "in": [value_state, endpoints]},
            {"id": new_push, "op": "max2",
             "in": [propagated, endpoint_value]},
        ])
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 64,
            "state": [
                {"id": value_state, "init": seed},
                {"id": push_state, "zeros": [4, 64, 64]},
            ],
            "body": body,
            "next": {
                value_state: new_value,
                push_state: new_push,
            },
        })
        value = loop_id + "." + value_state
        push_value = loop_id + "." + push_state
        effective_value = binary(
            "add",
            binary("mul", value, no_pusher),
            sum_channels(binary("mul", push_value, pushers), 4)
        )
        return value, effective_value

    target_value, target_effective = routing(target)
    recharge_value, recharge_effective = routing(recharge)

    target_at_player = unary(
        "gmax", binary("mul", target_value, player)
    )
    recharge_at_player = unary(
        "gmax", binary("mul", recharge_value, player)
    )
    target_distance = dense(
        target_at_player, np.array([[-1]], dtype=np.float32),
        np.array([128], dtype=np.float32)
    )
    recharge_distance = dense(
        recharge_at_player, np.array([[-1]], dtype=np.float32),
        np.array([128], dtype=np.float32)
    )

    # Timer bar.
    row_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    row_w[0, 0, 0, 0] = 63
    row_delta = unary("abs", conv(
        "coords", row_w, np.array([-61], dtype=np.float32)
    ))
    row_mask = unary("relu", conv(
        row_delta, np.array([[[[-2]]]], dtype=np.float32),
        np.array([1], dtype=np.float32)
    ))
    col_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    col_w[0, 1, 0, 0] = 126
    col_w[1, 1, 0, 0] = -126
    col_bounds = clip(conv(
        "coords", col_w, np.array([-25, 109], dtype=np.float32)
    ))
    col_mask = binary(
        "min2", ss(col_bounds, 0, 1), ss(col_bounds, 1, 2)
    )
    bar = binary("mul", blue, binary("mul", row_mask, col_mask))
    timer = dense(
        unary("gmean", bar), np.array([[4096]], dtype=np.float32)
    )

    # Infer one- versus two-unit decrement from the preceding action.
    delta = dense(
        vcat(old_timer, timer),
        np.array([[1, -1]], dtype=np.float32),
        np.array([-1], dtype=np.float32)
    )
    delta_pos = unary("vrelu", delta)
    delta_neg = unary("vrelu", dense(
        delta, np.array([[-1]], dtype=np.float32)
    ))
    delta_abs = binary("vadd", delta_pos, delta_neg)
    one_per_action = binary(
        "vmul", vnot(vclip(delta_abs, 1)), active
    )
    factor = dense(
        one_per_action, np.array([[0.5]], dtype=np.float32),
        np.array([0.5], dtype=np.float32)
    )
    moves_left = binary("vmul", timer, factor)

    # Recharge when the target is out of reach, or when a reachable
    # recharge is close to becoming unreachable.
    target_urgent = vclip(dense(
        vcat(target_distance, moves_left),
        np.array([[1, -1]], dtype=np.float32),
        np.array([1], dtype=np.float32)
    ), 1)
    recharge_urgent = vclip(dense(
        vcat(recharge_distance, moves_left),
        np.array([[1, -1]], dtype=np.float32),
        np.array([2], dtype=np.float32)
    ), 1)
    need_recharge = vclip(binary(
        "vadd", target_urgent, recharge_urgent
    ), 1)
    can_recharge = vclip(dense(
        vcat(moves_left, recharge_distance),
        np.array([[1, -1]], dtype=np.float32),
        np.array([0.5], dtype=np.float32)
    ), 1)
    use_recharge = binary("vmul", need_recharge, can_recharge)

    chosen = binary(
        "add",
        binary("mul", target_effective, vnot(use_recharge)),
        binary("mul", recharge_effective, use_recharge)
    )
    neighbor_values = shift_four(chosen)
    action_values = unary(
        "gmax", binary("mul", neighbor_values, player)
    )

    logits_w = np.zeros((8, 4), dtype=np.float32)
    for i in range(4):
        logits_w[i + 1, i] = 1
    logits_b = np.array(
        [-1000, 0, 0, 0, 0, -1000, -1000, -1000],
        dtype=np.float32
    )
    graph.append({
        "id": "logits", "op": "dense", "in": [action_values],
        "w": param(logits_w), "b": param(logits_b)
    })
    graph.append({
        "id": "click", "op": "conv", "in": [one],
        "w": param(np.zeros((1, 1, 1, 1), dtype=np.float32))
    })

    active_next = dense(
        icon, np.zeros((1, 24), dtype=np.float32),
        np.ones(1, dtype=np.float32)
    )
    memory_values = vcat(
        icon, icon_color, player_pos, remembered_switch,
        active_next, timer, switch_pos_vector, switch_seen_vector
    )
    graph.append({
        "id": "mem_next", "op": "dense", "in": [memory_values],
        "w": param(np.eye(44, dtype=np.float32))
    })

    return {"graph": graph, "memory_size": 44, "params": params}

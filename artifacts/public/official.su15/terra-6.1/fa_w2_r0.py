import numpy as np


def build():
    graph = []
    params = {}
    serial = 0
    pserial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return f"{prefix}_{serial}"

    def param(a):
        nonlocal pserial
        pserial += 1
        name = f"p_{pserial}"
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = fresh()
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

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def spatial_linear(x, scale=1.0, bias=0.0):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def spatial_clip(x):
        return binary("min2", unary("relu", x), one)

    def spatial_sum(x, channels):
        return conv(x, np.ones((1, channels, 1, 1), dtype=np.float32))

    def vector_clip(x, n):
        positive = unary("vrelu", x)
        excess = dense(positive, np.eye(n), -np.ones(n))
        excess = unary("vrelu", excess)
        neg_excess = dense(excess, -np.eye(n))
        return binary("vadd", positive, neg_excess)

    def vector_not(x, n):
        return dense(x, -np.eye(n), np.ones(n))

    def vector_sub(x, y, n):
        xy = node("vconcat", [x, y])
        return dense(xy, np.concatenate([np.eye(n), -np.eye(n)], axis=1))

    def vector_total(x, n):
        return dense(x, np.ones((1, n), dtype=np.float32))

    def vector_flag(x, n):
        return vector_clip(vector_total(x, n), 1)

    def highest_only(x, n):
        upper = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            upper[i, i + 1:] = 1.0
        higher_exists = vector_clip(dense(x, upper), n)
        return binary("vmul", x, vector_not(higher_exists, n))

    def counts(x, n):
        return dense(
            unary("gmean", x),
            np.eye(n, dtype=np.float32) * 4096.0,
        )

    def select_channels(x, weights, n):
        return spatial_sum(binary("mul", x, weights), n)

    one = node("slice", ["coords"], start=2, stop=3)

    ycoord = conv(
        "coords",
        np.array([[[[63.0]], [[0.0]], [[0.0]]]], dtype=np.float32),
    )
    xcoord = conv(
        "coords",
        np.array([[[[0.0]], [[63.0]], [[0.0]]]], dtype=np.float32),
    )

    play_lo = spatial_clip(spatial_linear(ycoord, 1.0, -9.0))
    play_hi = spatial_clip(spatial_linear(ycoord, -1.0, 63.0))
    play = binary("mul", play_lo, play_hi)

    head_y = spatial_clip(spatial_linear(ycoord, -1.0, 9.0))
    head_x = spatial_clip(spatial_linear(xcoord, 1.0, -23.0))
    head = binary("mul", head_y, head_x)

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    planes = [
        node("slice", ["x0"], start=c, stop=c + 1)
        for c in colors
    ]
    fruit_colors = node("concat", planes)

    # Four corners and top/left boundaries identify each square fruit.
    w_a = np.zeros((27, 9, 7, 7), dtype=np.float32)
    w_b = np.zeros((27, 27, 7, 7), dtype=np.float32)
    remainders = []

    for j, size in enumerate(sizes):
        q = size - 1
        a = min(max(q - 3, 0), 3)
        b = max(q - 6, 0)
        r = q - a - b
        remainders.append(r)

        shifts_a = [(0, a), (a, 0), (a, a)]
        shifts_b = [(0, b), (b, 0), (b, b)]
        for group in range(3):
            channel = 3 * j + group
            dy, dx = shifts_a[group]
            w_a[channel, j, 3 + dy, 3 + dx] = 1.0
            dy, dx = shifts_b[group]
            w_b[channel, channel, 3 + dy, 3 + dx] = 1.0

    shifted_a = conv(fruit_colors, w_a)
    shifted_b = conv(shifted_a, w_b)
    fruit_checks = node("concat", [fruit_colors, shifted_b])

    w_check = np.zeros((9, 36, 7, 7), dtype=np.float32)
    for j, r in enumerate(remainders):
        w_check[j, j, 3, 3] += 1.0
        w_check[j, j, 2, 3] -= 1.0
        w_check[j, j, 3, 2] -= 1.0
        w_check[j, 9 + 3 * j, 3, 3 + r] += 1.0
        w_check[j, 9 + 3 * j + 1, 3 + r, 3] += 1.0
        w_check[j, 9 + 3 * j + 2, 3 + r, 3 + r] += 1.0

    fruit_tl = unary(
        "relu",
        conv(fruit_checks, w_check, -3.5 * np.ones(9)),
    )
    fruit_tl = conv(
        fruit_tl,
        (2.0 * np.eye(9, dtype=np.float32))[:, :, None, None],
    )

    w_center_a = np.zeros((9, 9, 7, 7), dtype=np.float32)
    w_center_b = np.zeros((9, 9, 7, 7), dtype=np.float32)
    for j, size in enumerate(sizes):
        half = size // 2
        a = min(half, 3)
        b = half - a
        w_center_a[j, j, 3 - a, 3 - a] = 1.0
        w_center_b[j, j, 3 - b, 3 - b] = 1.0

    fruit = conv(conv(fruit_tl, w_center_a), w_center_b)
    fruit_play = binary("mul", fruit, play)
    fruit_head = binary("mul", fruit, head)

    glyphs = [
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
    ]
    glyph_colors = [7, 14, 13]
    w_glyph = np.zeros((3, 16, 7, 7), dtype=np.float32)
    b_glyph = np.zeros(3, dtype=np.float32)

    for j, pattern in enumerate(glyphs):
        positives = 0
        for yy in range(4):
            for xx in range(5):
                bit = pattern[yy][xx]
                positives += bit
                w_glyph[j, glyph_colors[j], 3 + yy - 2, 3 + xx - 2] = (
                    1.0 if bit else -1.0
                )
        b_glyph[j] = -(positives - 0.5)

    enemy = unary("relu", conv("x0", w_glyph, b_glyph))
    enemy = conv(
        enemy,
        (2.0 * np.eye(3, dtype=np.float32))[:, :, None, None],
    )
    enemy_play = binary("mul", enemy, play)
    enemy_head = binary("mul", enemy, head)

    # Detect the outer ring of the goal, allowing its center to be covered.
    color9 = node("slice", ["x0"], start=9, stop=10)
    shifts = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    w_hshift = np.zeros((9, 1, 7, 7), dtype=np.float32)
    for k, (dy, dx) in enumerate(shifts):
        w_hshift[k, 0, 3 + dy, 3 + dx] = 1.0
    hshift = conv(color9, w_hshift)

    goal_rows = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    w_goal = np.zeros((1, 9, 7, 7), dtype=np.float32)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if abs(dx) <= 2 and abs(dy) <= 2:
                continue
            inside = abs(dx) <= goal_rows[dy + 4]
            value = 1.0 if inside else -1.5
            sy = -2 if dy < -3 else (2 if dy > 3 else 0)
            sx = -2 if dx < -3 else (2 if dx > 3 else 0)
            k = shifts.index((sy, sx))
            w_goal[0, k, 3 + dy - sy, 3 + dx - sx] += value

    goals = spatial_clip(conv(hshift, w_goal, np.array([-31.0])))
    goals = binary("mul", goals, play)
    goal_box = spatial_clip(
        conv(goals, np.ones((1, 1, 7, 7), dtype=np.float32))
    )
    goal_box = spatial_clip(
        conv(goal_box, np.ones((1, 1, 3, 3), dtype=np.float32))
    )
    not_goal_box = binary("sub", one, goal_box)

    fruit_in_goal = binary("mul", fruit_play, goal_box)
    enemy_in_goal = binary("mul", enemy_play, goal_box)
    fruit_outside = binary("mul", fruit_play, not_goal_box)
    enemy_outside = binary("mul", enemy_play, not_goal_box)

    fruit_required = counts(fruit_head, 9)
    enemy_required = counts(enemy_head, 3)
    fruit_done = counts(fruit_in_goal, 9)
    enemy_done = counts(enemy_in_goal, 3)
    fruit_all_count = counts(fruit_play, 9)
    enemy_all_count = counts(enemy_play, 3)
    fruit_out_count = counts(fruit_outside, 9)
    enemy_out_count = counts(enemy_outside, 3)

    fruit_missing = vector_clip(vector_sub(fruit_required, fruit_done, 9), 9)
    enemy_missing = vector_clip(vector_sub(enemy_required, enemy_done, 3), 3)
    fruit_need = vector_flag(fruit_missing, 9)

    fruit_direct = binary(
        "vmul", fruit_missing, vector_clip(fruit_out_count, 9)
    )
    fruit_direct = highest_only(fruit_direct, 9)
    fruit_direct_flag = vector_flag(fruit_direct, 9)

    fruit_pairs = dense(fruit_all_count, np.eye(9), -np.ones(9))
    fruit_pairs = vector_clip(fruit_pairs, 9)
    upper9 = np.zeros((9, 9), dtype=np.float32)
    for i in range(9):
        upper9[i, i + 1:] = 1.0
    fruit_higher_needed = vector_clip(dense(fruit_missing, upper9), 9)
    fruit_merge = binary("vmul", fruit_pairs, fruit_higher_needed)
    fruit_merge = highest_only(fruit_merge, 9)
    fruit_merge = binary(
        "vmul", fruit_merge, vector_not(fruit_direct_flag, 1)
    )

    enemy_direct = binary(
        "vmul", enemy_missing, vector_clip(enemy_out_count, 3)
    )
    enemy_direct = highest_only(enemy_direct, 3)
    enemy_direct_flag = vector_flag(enemy_direct, 3)

    enemy_pairs = dense(enemy_all_count, np.eye(3), -np.ones(3))
    enemy_pairs = vector_clip(enemy_pairs, 3)
    upper3 = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        upper3[i, i + 1:] = 1.0
    enemy_higher_needed = vector_clip(dense(enemy_missing, upper3), 3)
    enemy_merge = binary("vmul", enemy_pairs, enemy_higher_needed)
    enemy_merge = highest_only(enemy_merge, 3)
    enemy_merge = binary(
        "vmul", enemy_merge, vector_not(enemy_direct_flag, 1)
    )

    enemy_merge_flag = vector_flag(enemy_merge, 3)
    no_enemy_merge = vector_not(enemy_merge_flag, 1)
    fruit_direct = binary("vmul", fruit_direct, no_enemy_merge)
    fruit_merge = binary("vmul", fruit_merge, no_enemy_merge)
    enemy_direct = binary(
        "vmul", enemy_direct, vector_not(fruit_need, 1)
    )

    f_direct_map = select_channels(fruit_outside, fruit_direct, 9)
    f_merge_map = select_channels(fruit_play, fruit_merge, 9)
    e_direct_map = select_channels(enemy_outside, enemy_direct, 3)
    e_merge_map = select_channels(enemy_play, enemy_merge, 3)

    selected_fruit = spatial_clip(binary("add", f_direct_map, f_merge_map))
    selected_enemy = spatial_clip(binary("add", e_direct_map, e_merge_map))
    selected = spatial_clip(binary("add", selected_fruit, selected_enemy))

    merge_flags = node("vconcat", [
        vector_flag(fruit_merge, 9),
        vector_flag(enemy_merge, 3),
    ])
    merge_mode = vector_clip(vector_total(merge_flags, 2), 1)
    direct_mode = vector_not(merge_mode, 1)

    # Propagate 130 minus Manhattan distance with shared kernels.
    w_neighbor = np.zeros((4, 1, 3, 3), dtype=np.float32)
    for k, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        w_neighbor[k, 0, 1 + dy, 1 + dx] = 1.0
    p_neighbor = param(w_neighbor)
    p_decay_w = param(np.ones((1, 1, 1, 1), dtype=np.float32))
    p_decay_b = param(np.array([-1.0], dtype=np.float32))

    def distance_field(mask, iterations=126):
        seed = spatial_linear(mask, 130.0, 0.0)
        loop_id = fresh("loop")
        state_id = fresh("state")
        body = []

        def body_node(op, inputs, **fields):
            name = fresh("body")
            body.append({"id": name, "op": op, "in": list(inputs), **fields})
            return name

        neighbors = body_node("conv", [state_id], w=p_neighbor)
        pieces = [
            body_node("slice", [neighbors], start=k, stop=k + 1)
            for k in range(4)
        ]
        spread = pieces[0]
        for piece in pieces[1:]:
            spread = body_node("max2", [spread, piece])
        spread = body_node("conv", [spread], w=p_decay_w, b=p_decay_b)
        spread = body_node("relu", [spread])
        updated = body_node("max2", [state_id, spread])

        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [{"id": state_id, "init": seed}],
            "body": body,
            "next": {state_id: updated},
        })
        return loop_id + "." + state_id

    goal_field = distance_field(goals)

    position_bias = conv(
        "coords",
        np.array([[[[0.4]], [[0.006]], [[0.0]]]], dtype=np.float32),
    )

    def select_one(mask, field):
        score = binary("mul", binary("add", field, position_bias), mask)
        maximum = unary("gmax", score)
        threshold = dense(
            maximum,
            np.array([[1.0]], dtype=np.float32),
            np.array([-0.00002], dtype=np.float32),
        )
        near_max = unary("relu", binary("sub", score, threshold))
        near_max = spatial_clip(spatial_linear(near_max, 100000.0))
        return binary("mul", near_max, mask)

    anchor = select_one(selected, goal_field)
    anchor_field = distance_field(anchor)
    other_selected = unary("relu", binary("sub", selected, anchor))
    peer = select_one(other_selected, anchor_field)

    source_direct = binary("mul", anchor, direct_mode)
    source_merge = binary("mul", peer, merge_mode)
    source = spatial_clip(binary("add", source_direct, source_merge))
    source_field = distance_field(source)

    destination_direct = binary("mul", goal_field, direct_mode)
    destination_merge = binary("mul", anchor_field, merge_mode)
    destination = binary("add", destination_direct, destination_merge)
    reachable = spatial_clip(spatial_linear(source_field, 1.0, -121.0))

    all_fruit_map = spatial_clip(spatial_sum(fruit_play, 9))
    all_enemy_map = spatial_clip(spatial_sum(enemy_play, 3))
    all_objects = spatial_clip(binary("add", all_fruit_map, all_enemy_map))
    selected_objects = spatial_clip(binary("add", selected_fruit, selected_enemy))
    unrelated = unary("relu", binary("sub", all_objects, selected_objects))
    hazard_field = distance_field(unrelated, iterations=14)
    hazard = spatial_clip(spatial_linear(hazard_field, 1.0, -119.0))

    move_value = spatial_linear(destination, 1.0, 20.0)
    move_value = binary("sub", move_value, spatial_linear(hazard, 4.0))
    move_value = binary("mul", move_value, reachable)
    move_value = binary("mul", move_value, play)

    all_field = distance_field(all_objects)
    wait_value = spatial_linear(all_field, -1.0, 131.0)
    wait_value = binary("mul", wait_value, play)

    active = vector_clip(unary("gmax", source), 1)
    ordinary = binary(
        "add",
        binary("mul", move_value, active),
        binary("mul", wait_value, vector_not(active, 1)),
    )

    # Protect fruit on levels without an enemy requirement. On levels
    # requiring enemies, allow pursuit and possible fruit downgrades.
    fruit_field = distance_field(all_fruit_map)
    threat_strength = unary(
        "relu", spatial_linear(fruit_field, 1.0, -108.0)
    )
    threat_strength = binary("mul", threat_strength, all_enemy_map)
    threat = vector_clip(unary("gmax", threat_strength), 1)
    has_enemy_objective = vector_flag(enemy_required, 3)
    threat = binary("vmul", threat, vector_not(has_enemy_objective, 1))

    threatened_enemy = select_one(all_enemy_map, fruit_field)
    threatened_field = distance_field(threatened_enemy)
    enemy_reach = spatial_clip(
        spatial_linear(threatened_field, 1.0, -121.0)
    )
    enemy_reach = binary("mul", enemy_reach, play)

    fruit_exclusion = all_fruit_map
    for _ in range(4):
        fruit_exclusion = spatial_clip(
            conv(
                fruit_exclusion,
                np.ones((1, 1, 7, 7), dtype=np.float32),
            )
        )
    fruit_exclusion = spatial_clip(
        conv(
            fruit_exclusion,
            np.ones((1, 1, 3, 3), dtype=np.float32),
        )
    )
    safe_enemy_reach = binary(
        "mul", enemy_reach, binary("sub", one, fruit_exclusion)
    )
    safe_exists = vector_clip(unary("gmax", safe_enemy_reach), 1)
    chosen_reach = binary(
        "add",
        binary("mul", safe_enemy_reach, safe_exists),
        binary("mul", enemy_reach, vector_not(safe_exists, 1)),
    )
    escape_value = spatial_linear(fruit_field, -1.0, 170.0)
    escape_value = binary("mul", escape_value, chosen_reach)

    planned = binary(
        "add",
        binary("mul", escape_value, threat),
        binary("mul", ordinary, vector_not(threat, 1)),
    )

    # A red objective indicates that too much fruit mass has been lost.
    red_plane = node("slice", ["x0"], start=2, stop=3)
    red_header = binary("mul", red_plane, head)
    bad_state = vector_clip(unary("gmax", red_header), 1)

    # Object-layout fingerprint: weighted count and first spatial moments.
    weighted_fruit = conv(
        fruit_play,
        np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1),
    )
    weighted_enemy = conv(
        enemy_play,
        np.array([12.0, 13.0, 14.0], dtype=np.float32).reshape(1, 3, 1, 1),
    )
    weighted_objects = binary("add", weighted_fruit, weighted_enemy)
    moments = binary("mul", "coords", weighted_objects)
    signature = dense(
        unary("gmean", moments),
        4096.0 * np.eye(3, dtype=np.float32),
    )

    old_signature = node("vslice", ["mem"], start=2, stop=5)
    signature_diff = vector_sub(signature, old_signature, 3)
    signature_negative = dense(signature_diff, -np.eye(3))
    absolute_diff = binary(
        "vadd",
        unary("vrelu", signature_diff),
        unary("vrelu", signature_negative),
    )
    diff_total = dense(
        absolute_diff,
        np.full((1, 3), 10.0, dtype=np.float32),
    )
    unchanged = vector_not(vector_clip(diff_total, 1), 1)

    prev_click = node("vslice", ["aprev"], start=6, stop=7)
    prev_undo = node("vslice", ["aprev"], start=7, stop=8)
    repeat = binary("vmul", unchanged, prev_click)
    avoid_previous = vector_clip(binary("vadd", repeat, prev_undo), 1)

    remembered_y = node("vslice", ["mem"], start=0, stop=1)
    remembered_x = node("vslice", ["mem"], start=1, stop=2)
    coord_y = node("slice", ["coords"], start=0, stop=1)
    coord_x = node("slice", ["coords"], start=1, stop=2)
    dy = unary("abs", binary("sub", coord_y, remembered_y))
    dx = unary("abs", binary("sub", coord_x, remembered_x))
    cheb = binary("max2", dy, dx)
    banned = spatial_clip(spatial_linear(cheb, -63.0, 6.0))
    banned = binary("mul", banned, avoid_previous)
    planned = binary("sub", planned, spatial_linear(banned, 300.0))

    tie_bias = conv(
        "coords",
        np.array(
            [[[[-0.4032]], [[-0.0063]], [[0.4095]]]],
            dtype=np.float32,
        ),
    )
    final_score = binary("add", planned, tie_bias)
    graph.append({"id": "click", "op": "relu", "in": [final_score]})

    # Store the selected click for collision or red-state recovery.
    maximum = unary("gmax", "click")
    threshold = dense(
        maximum,
        np.array([[1.0]], dtype=np.float32),
        np.array([-0.00003], dtype=np.float32),
    )
    winner = unary("relu", binary("sub", "click", threshold))
    winner = spatial_clip(spatial_linear(winner, 100000.0))
    winning_coords = binary("mul", "coords", winner)
    winning_coords = node("slice", [winning_coords], start=0, stop=2)
    new_click = dense(
        unary("gmean", winning_coords),
        4096.0 * np.eye(2, dtype=np.float32),
    )

    old_click = node("vslice", ["mem"], start=0, stop=2)
    keep_old = binary("vmul", old_click, bad_state)
    write_new = binary(
        "vmul", new_click, vector_not(bad_state, 1)
    )
    stored_click = binary("vadd", keep_old, write_new)
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [stored_click, signature],
    })

    # Undo a red state; if undo leaves red, reset.
    red_after_undo = binary("vmul", bad_state, prev_undo)
    logit_input = node("vconcat", [bad_state, red_after_undo])
    logits_w = np.zeros((8, 2), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    logits_w[6, 0] = -2.0
    logits_w[7, 0] = 2.0
    logits_w[0, 1] = 4.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [logit_input],
        "w": param(logits_w),
        "b": param(logits_b),
    })

    return {"graph": graph, "memory_size": 5, "params": params}

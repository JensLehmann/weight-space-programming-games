import numpy as np


def build():
    graph = []
    params = {}
    destinations = [graph]
    serial = [0]
    pserial = [0]
    shift_cache = {}

    def parameter(a):
        name = "p_%05d" % pserial[0]
        pserial[0] += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, inputs, name=None, **fields):
        if name is None:
            name = "n_%05d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destinations[-1].append(nd)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b, name)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def sl(x, start, stop):
        return emit("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def relu(a):
        return emit("relu", [a])

    def maximum(a, b):
        return emit("max2", [a, b])

    def sum_maps(xs):
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def max_maps(xs):
        result = xs[0]
        for x in xs[1:]:
            result = maximum(result, x)
        return result

    def shift_once(x, dy, dx, bias=0.0):
        k = 3 if max(abs(dy), abs(dx)) <= 1 else 7
        key = (k, dy, dx, float(bias))
        if key not in shift_cache:
            w = np.zeros((1, 1, k, k), dtype=np.float32)
            w[0, 0, k // 2 + dy, k // 2 + dx] = 1.0
            fields = {"w": parameter(w)}
            if bias != 0:
                fields["b"] = parameter([bias])
            shift_cache[key] = fields
        return emit("conv", [x], **shift_cache[key])

    def offset(x, dy, dx, bias=0.0):
        while max(abs(dy), abs(dx)) > 3:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            x = shift_once(x, sy, sx)
            dy -= sy
            dx -= sx
        return shift_once(x, dy, dx, bias)

    def sharp_equal(diff, channels):
        eye = np.eye(channels, dtype=np.float32)
        a = relu(linear(
            diff, -2.0 * eye,
            np.full(channels, 1.5, dtype=np.float32)
        ))
        excess = relu(linear(
            a, eye, -np.ones(channels, dtype=np.float32)
        ))
        return sub(a, excess)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    opposite = [1, 0, 3, 2]
    H = 128

    zero = linear(
        "coords", np.zeros((1, 3), dtype=np.float32),
        [0.0], name="click"
    )
    one = sl("coords", 2, 3)
    colour = {
        c: sl("x0", c, c + 1)
        for c in (2, 8, 9, 11, 12, 13, 14, 15)
    }

    # Detect centres of the full 3x3 node blocks.
    node_colours = np.zeros((1, 16), dtype=np.float32)
    node_colours[0, [0, 4, 8, 9, 11, 12, 13, 14, 15]] = 1.0
    node_pixels = linear("x0", node_colours)
    centres = relu(conv(
        node_pixels,
        np.ones((1, 1, 3, 3), dtype=np.float32),
        [-8.0]
    ))

    player = mul(centres, colour[9])
    visible_red = mul(centres, colour[8])
    visible_patrol = mul(centres, colour[12])
    copier = mul(centres, colour[13])
    visible_exit = mul(centres, colour[14])

    marker = add(colour[15], colour[11])
    observed_red_dirs = []
    observed_patrol_dirs = []
    active_dirs = []
    bridges = []
    edges = []

    for dy, dx in directions:
        facing = offset(marker, dy, dx)
        active_facing = offset(colour[11], dy, dx)
        observed_red_dirs.append(mul(visible_red, facing))
        observed_patrol_dirs.append(mul(visible_patrol, facing))
        active_dirs.append(mul(copier, active_facing))
        bridge = offset(colour[2], 3 * dy, 3 * dx)
        bridges.append(bridge)
        edges.append(mul(bridge, centres))

    active = sum_maps(active_dirs)
    inactive = relu(sub(copier, active))

    # Unique integer-valued position labels.
    position_code = linear("coords", [[4032.0, 63.0, 1.0]])
    current_position = emit("gmax", [mul(position_code, player)])

    observed_exit_code = emit("gmax", [
        mul(position_code, visible_exit)
    ])
    exit_seen = emit("gmax", [visible_exit])
    exit_absent = dense(exit_seen, [[-1.0]], [1.0])
    old_exit_code = vsl("mem", 33, 34)
    retained_exit_code = emit("vmul", [exit_absent, old_exit_code])
    exit_code = emit("vadd", [
        observed_exit_code, retained_exit_code
    ])
    exit_difference = emit("abs", [sub(position_code, exit_code)])
    remembered_exit = sharp_equal(exit_difference, 1)
    exit_map = maximum(visible_exit, remembered_exit)

    active_mean = emit("gmean", [active])
    active_count = dense(active_mean, [[4096.0]])
    previous_active_count = vsl("mem", 32, 33)
    count_pair = emit("vconcat", [
        active_count, previous_active_count
    ])
    increase = emit("vrelu", [
        dense(count_pair, [[1.0, -1.0]])
    ])
    increase_excess = emit("vrelu", [
        dense(increase, [[1.0]], [-1.0])
    ])
    negative_excess = dense(increase_excess, [[-1.0]])
    newly_active = emit("vadd", [increase, negative_excess])
    established = dense(newly_active, [[-1.0]], [1.0])

    recent_positions = vsl("mem", 0, 32)
    visit_difference = emit("abs", [
        sub(position_code, recent_positions)
    ])
    visit_matches = sharp_equal(visit_difference, 32)
    visit_penalty = linear(
        visit_matches,
        np.full((1, 32), -0.15, dtype=np.float32)
    )

    same_position = emit("gmax", [
        mul(sl(visit_matches, 0, 1), player)
    ])
    moved_flag = dense(same_position, [[-1.0]], [1.0])
    keep_history_w = np.zeros((1, 8), dtype=np.float32)
    keep_history_w[0, 0] = -1.0
    keep_history = dense("aprev", keep_history_w, [1.0])
    not_player = sub(one, player)

    # Memory layout:
    # 0:32 player positions; 32 active count; 33 exit position;
    # 34:46 red positions; 46:58 red headings;
    # 58:66 patrol positions; 66:74 patrol headings.
    def recall_objects(position_start, heading_start, count):
        positions = vsl(
            "mem", position_start, position_start + count
        )
        headings = vsl(
            "mem", heading_start, heading_start + count
        )
        difference = emit("abs", [sub(position_code, positions)])
        matches = sharp_equal(difference, count)

        # Piecewise-linear exact decoding of heading codes 1..4.
        hinge_w = np.tile(
            np.eye(count, dtype=np.float32), (6, 1)
        )
        hinge_b = -np.repeat(
            np.arange(6, dtype=np.float32), count
        )
        hinges = emit("vrelu", [
            dense(headings, hinge_w, hinge_b)
        ])
        decode_w = np.zeros(
            (4 * count, 6 * count), dtype=np.float32
        )
        for d in range(4):
            for j in range(count):
                decode_w[d * count + j, d * count + j] = 1.0
                decode_w[
                    d * count + j, (d + 1) * count + j
                ] = -2.0
                decode_w[
                    d * count + j, (d + 2) * count + j
                ] = 1.0
        decoded = dense(hinges, decode_w)

        result = []
        for d in range(4):
            heading_mask = vsl(
                decoded, d * count, (d + 1) * count
            )
            obj_map = linear(
                mul(matches, heading_mask),
                np.ones((1, count), dtype=np.float32)
            )
            result.append(mul(obj_map, keep_history))
        return result

    old_red_dirs = recall_objects(34, 46, 12)
    old_patrol_dirs = recall_objects(58, 66, 8)

    # A remembered red enemy remains relevant if another sprite
    # currently covers it. Otherwise absence means it was captured.
    red_occluders = add(visible_patrol, copier)
    red_dirs = [
        maximum(
            observed_red_dirs[d],
            mul(old_red_dirs[d], red_occluders)
        )
        for d in range(4)
    ]
    red = sum_maps(red_dirs)

    # Advance remembered patrols by one successful player move.
    # Remove patrols whose previous node was captured by the player.
    old_kept = []
    old_flipped = []
    for d, (dy, dx) in enumerate(directions):
        uncaptured = mul(old_patrol_dirs[d], not_player)
        advanced = offset(uncaptured, -6 * dy, -6 * dx)
        continuing = mul(advanced, bridges[d])
        old_kept.append(continuing)
        old_flipped.append(sub(advanced, continuing))

    predicted_patrol_dirs = []
    for d in range(4):
        advanced = add(old_kept[d], old_flipped[opposite[d]])
        predicted = add(
            old_patrol_dirs[d],
            mul(sub(advanced, old_patrol_dirs[d]), moved_flag)
        )
        predicted_patrol_dirs.append(predicted)

    patrol_occluders = add(visible_red, copier)
    patrol_dirs = [
        maximum(
            observed_patrol_dirs[d],
            mul(predicted_patrol_dirs[d], patrol_occluders)
        )
        for d in range(4)
    ]
    patrol = sum_maps(patrol_dirs)

    # Compress object maps into a fixed number of coordinate/heading
    # slots, using max reductions and exact coordinate matching.
    def store_objects(dir_maps, occupancy, count):
        stacked = emit("concat", dir_maps)
        remaining_key = mul(position_code, occupancy)
        positions = []
        headings = []
        for _ in range(count):
            position = emit("gmax", [remaining_key])
            difference = emit("abs", [
                sub(position_code, position)
            ])
            selected = sharp_equal(difference, 1)
            selected_headings = emit("gmax", [
                mul(stacked, selected)
            ])
            heading = dense(
                selected_headings, [[1.0, 2.0, 3.0, 4.0]]
            )
            positions.append(position)
            headings.append(heading)
            remaining_key = relu(sub(
                remaining_key, mul(position_code, selected)
            ))
        return (
            emit("vconcat", positions),
            emit("vconcat", headings)
        )

    red_positions, red_headings = store_objects(red_dirs, red, 12)
    patrol_positions, patrol_headings = store_objects(
        patrol_dirs, patrol, 8
    )

    memory_tail = vsl("mem", 0, 31)
    emit(
        "vconcat",
        [
            current_position, memory_tail, active_count, exit_code,
            red_positions, red_headings,
            patrol_positions, patrol_headings
        ],
        name="mem_next"
    )

    red_attacks = sum_maps([
        offset(red_dirs[d], -6 * dy, -6 * dx)
        for d, (dy, dx) in enumerate(directions)
    ])
    allowed = relu(sub(centres, red_attacks))

    goal_features = emit("concat", [exit_map, red, inactive])
    goals = linear(goal_features, [[1000.0, 300.0, 300.0]])

    kill_features = emit("concat", [red, patrol, copier])
    kill_bonus = linear(kill_features, [[4.0, 2.0, 4.0]])
    local_bonus = add(kill_bonus, visit_penalty)

    active_first_by_dir = [
        offset(active_dirs[d], -6 * dy, -6 * dx)
        for d, (dy, dx) in enumerate(directions)
    ]
    active_first = sum_maps(active_first_by_dir)
    active_second_new = sum_maps([
        offset(active_first_by_dir[d], -6 * dy, -6 * dx)
        for d, (dy, dx) in enumerate(directions)
    ])

    established_second_parts = []
    for d, (dy, dx) in enumerate(directions):
        previous_direction = vsl("aprev", d + 1, d + 2)
        advanced = offset(active_first, -6 * dy, -6 * dx)
        established_second_parts.append(
            mul(advanced, previous_direction)
        )
    active_second_established = sum_maps(established_second_parts)
    active_second = add(
        mul(active_second_new, newly_active),
        mul(active_second_established, established)
    )

    # Forecast patrol movement and retain reverse-time hazard maps.
    forecast_body = []
    destinations.append(forecast_body)

    moved = []
    kept = []
    flipped = []
    for d, (dy, dx) in enumerate(directions):
        m = offset("f_e%d" % d, -6 * dy, -6 * dx)
        k = mul(m, bridges[d])
        moved.append(m)
        kept.append(k)
        flipped.append(sub(m, k))

    next_enemies = [
        add(kept[d], flipped[opposite[d]]) for d in range(4)
    ]
    patrol_danger = sum_maps(moved)
    old_patrol_positions = sum_maps(
        ["f_e%d" % d for d in range(4)]
    )
    capture_union = emit("min2", [old_patrol_positions, one])
    forecast_danger = add(patrol_danger, "f_a")

    next_hazard_history = emit(
        "concat",
        [forecast_danger, sl("f_hazard", 0, H - 1)]
    )
    next_capture_history = emit(
        "concat",
        [capture_union, sl("f_capture", 0, H - 1)]
    )
    destinations.pop()

    forecast_states = [
        {"id": "f_e%d" % d, "init": patrol_dirs[d]}
        for d in range(4)
    ] + [
        {"id": "f_hazard", "zeros": [H, 64, 64]},
        {"id": "f_capture", "zeros": [H, 64, 64]},
        {"id": "f_a", "init": active_first},
        {"id": "f_b", "init": active_second},
    ]
    forecast_next = {
        "f_e%d" % d: next_enemies[d] for d in range(4)
    }
    forecast_next.update({
        "f_hazard": next_hazard_history,
        "f_capture": next_capture_history,
        "f_a": "f_b",
        "f_b": zero,
    })
    graph.append({
        "id": "forecast",
        "op": "loop",
        "T": H,
        "state": forecast_states,
        "body": forecast_body,
        "next": forecast_next,
    })

    def safe_destination_value(value, danger, capture_reward):
        value = maximum(value, capture_reward)
        safe = relu(sub(allowed, danger))
        return mul(value, safe)

    def directional_backup(safe_value, d):
        dy, dx = directions[d]
        candidate = relu(offset(
            safe_value, 6 * dy, 6 * dx, bias=-1.0
        ))
        return mul(candidate, edges[d])

    def bellman(value, danger, capture_reward):
        safe_value = safe_destination_value(
            value, danger, capture_reward
        )
        candidates = [
            directional_backup(safe_value, d) for d in range(4)
        ]
        return maximum(goals, max_maps(candidates))

    # Reverse-time Bellman backups, leaving three turns for
    # action-conditioned copier forecasts.
    planning_body = []
    destinations.append(planning_body)
    planning_hazard = sl("d_hazard", 0, 1)
    planning_capture = sl("d_capture", 0, 1)
    planning_capture_reward = linear(planning_capture, [[250.0]])
    next_value = bellman(
        "d_value", planning_hazard, planning_capture_reward
    )
    next_planning_hazards = emit(
        "concat", [sl("d_hazard", 1, H), zero]
    )
    next_planning_captures = emit(
        "concat", [sl("d_capture", 1, H), zero]
    )
    destinations.pop()

    graph.append({
        "id": "planning",
        "op": "loop",
        "T": H - 3,
        "state": [
            {"id": "d_value", "init": goals},
            {"id": "d_hazard", "init": "forecast.f_hazard"},
            {"id": "d_capture", "init": "forecast.f_capture"},
        ],
        "body": planning_body,
        "next": {
            "d_value": next_value,
            "d_hazard": next_planning_hazards,
            "d_capture": next_planning_captures,
        },
    })

    danger3 = sl("planning.d_hazard", 0, 1)
    danger2 = sl("planning.d_hazard", 1, 2)
    danger1 = sl("planning.d_hazard", 2, 3)
    capture2 = linear(sl("planning.d_capture", 0, 1), [[250.0]])
    capture1 = linear(sl("planning.d_capture", 1, 2), [[250.0]])
    capture0 = linear(sl("planning.d_capture", 2, 3), [[250.0]])

    immediate_safe = relu(sub(allowed, danger1))
    action_statistics = []

    for d, (dy, dx) in enumerate(directions):
        copier_third = offset(active_second, -6 * dy, -6 * dx)
        branch_danger3 = add(danger3, copier_third)

        value2 = bellman(
            "planning.d_value", branch_danger3, capture2
        )
        value1 = bellman(value2, danger2, capture1)
        safe_value1 = safe_destination_value(
            value1, danger1, capture0
        )
        action_value = directional_backup(safe_value1, d)

        legal_safe = mul(
            offset(immediate_safe, 6 * dy, 6 * dx), edges[d]
        )
        safety_penalty = linear(
            legal_safe, [[2000.0]], [-2000.0]
        )
        destination_bonus = offset(
            local_bonus, 6 * dy, 6 * dx
        )

        score_map = add(
            add(action_value, destination_bonus),
            safety_penalty
        )
        player_score = mul(score_map, player)
        action_statistics.append(emit("gmean", [player_score]))

    statistics = emit("vconcat", action_statistics)
    output_w = np.zeros((8, 4), dtype=np.float32)
    for d in range(4):
        output_w[d + 1, d] = 4096.0
    output_b = np.full(8, -10000.0, dtype=np.float32)
    output_b[0] = -1500.0
    output_b[1:5] = 0.0
    dense(statistics, output_w, output_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 74,
        "params": params,
    }

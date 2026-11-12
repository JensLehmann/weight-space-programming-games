import numpy as np


def build():
    params = {}
    graph = []
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    serial = [0]
    current = [graph]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, shape=None, **fields):
        name = fresh()
        if shape is None:
            if op in ("gmax", "gmean"):
                shape = ("v", shapes[inputs[0]][1])
            elif op in ("concat", "vconcat"):
                shape = (
                    "s" if op == "concat" else "v",
                    sum(shapes[x][1] for x in inputs),
                )
            elif op in ("slice", "vslice"):
                shape = (
                    shapes[inputs[0]][0],
                    fields["stop"] - fields["start"],
                )
            elif op in ("add", "sub", "mul", "max2", "min2"):
                shape = ("s", max(shapes[x][1] for x in inputs))
            elif op in ("vadd", "vmul"):
                shape = ("v", max(shapes[x][1] for x in inputs))
            else:
                shape = shapes[inputs[0]]
        current[0].append({
            "id": name,
            "op": op,
            "in": list(inputs),
            **fields,
        })
        shapes[name] = shape
        return name

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def sl(x, a, b):
        op = "slice" if shapes[x][0] == "s" else "vslice"
        return node(op, [x], start=a, stop=b)

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], ("s", w.shape[0]), **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], ("v", w.shape[0]), **fields)

    def affine(x, a=1.0, b=0.0):
        c = shapes[x][1]
        if shapes[x][0] == "s":
            w = np.eye(c, dtype=np.float32)[:, :, None, None] * a
            return conv(x, w, np.full(c, b, np.float32))
        return dense(
            x,
            np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32),
        )

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def vmax(a, b):
        return node("max2", [a, b])

    def vmin(a, b):
        return node("min2", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def relu(x):
        return node(
            "relu" if shapes[x][0] == "s" else "vrelu", [x]
        )

    def absolute(x):
        if shapes[x][0] == "s":
            return node("abs", [x])
        return va(relu(x), relu(affine(x, -1)))

    def inverse(x):
        return affine(x, -1, 1)

    one = sl("coords", 2, 3)

    def clip(x):
        if shapes[x][0] == "s":
            return vmin(relu(x), one)
        return inverse(relu(inverse(relu(x))))

    def hard_vector(x):
        return clip(affine(x, 100, -50))

    def gmax(x):
        return node("gmax", [x])

    def shift(x, dy, dx):
        c = shapes[x][1]
        r = max(abs(dy), abs(dx), 1)
        k = 2 * r + 1
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i, r + dy, r + dx] = 1
        return conv(x, w)

    def line_dilate(x, axis, radius):
        c = shapes[x][1]
        k = 2 * radius + 1
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            if axis == 0:
                w[i, i, :, radius] = 1
            else:
                w[i, i, radius, :] = 1
        return clip(conv(x, w))

    def plane_dilate(x):
        c = shapes[x][1]
        w = np.zeros((c, c, 3, 3), np.float32)
        for i in range(c):
            w[i, i, 1, 1] = 1
            w[i, i, 0, 1] = 1
            w[i, i, 2, 1] = 1
            w[i, i, 1, 0] = 1
            w[i, i, 1, 2] = 1
        return clip(conv(x, w))

    def neighborhood_max(x, negative_padding=False):
        c = shapes[x][1]
        source = cat(x, one) if negative_padding else x
        cin = c + int(negative_padding)
        w = np.zeros((4 * c, cin, 3, 3), np.float32)
        offsets = [(0, 1), (2, 1), (1, 0), (1, 2)]
        for d, (r, col_index) in enumerate(offsets):
            for i in range(c):
                w[d * c + i, i, r, col_index] = 1
                if negative_padding:
                    w[d * c + i, c, r, col_index] = 128
        b = (
            np.full(4 * c, -128, np.float32)
            if negative_padding else None
        )
        z = conv(source, w, b)
        result = x
        for d in range(4):
            result = vmax(result, sl(z, d * c, (d + 1) * c))
        return result

    def loop(initial, iterations, body_function):
        loop_id = fresh("L")
        state_ids = []
        state_specs = []
        for value in initial:
            sid = fresh("q")
            shapes[sid] = shapes[value]
            state_ids.append(sid)
            state_specs.append({"id": sid, "init": value})
        outer = current[0]
        body = []
        current[0] = body
        updated = body_function(state_ids)
        current[0] = outer
        outer.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": state_specs,
            "body": body,
            "next": dict(zip(state_ids, updated)),
        })
        outputs = []
        for sid, value in zip(state_ids, initial):
            output = loop_id + "." + sid
            shapes[output] = shapes[value]
            outputs.append(output)
        return outputs

    def masked_max(mask, value, reduction="gmax", floor=-256.0):
        z = mul(mask, affine(value, 1, -floor))
        z = node(reduction, [z])
        return affine(z, 1, floor)

    # Semantic colour planes.
    black = sl("x0", 0, 1)
    blue = sl("x0", 1, 2)
    yellow = sl("x0", 4, 5)
    wall = sl("x0", 5, 6)
    buttons = sl("x0", 9, 10)
    purple = sl("x0", 12, 13)
    bases = cat(sl("x0", 11, 12), sl("x0", 14, 15))
    row = affine(sl("coords", 0, 1), 63)
    col = affine(sl("coords", 1, 2), 63)

    # The timer overwrites row zero.
    board = clip(row)
    yellow = mul(yellow, board)
    near_yellow = clip(conv(
        yellow, np.ones((1, 1, 5, 5), np.float32)
    ))
    players = mul(bases, near_yellow)
    goals = sub(bases, players)
    presence = gmax(players)

    base_sum = conv(bases, np.ones((1, 2, 1, 1), np.float32))
    direction_input = cat(base_sum, yellow)
    w = np.zeros((4, 2, 3, 3), np.float32)
    # Down, up, right, left: yellow lies upstream of the player's base.
    for i, (r, c) in enumerate([(0, 1), (2, 1), (1, 0), (1, 2)]):
        w[i, 0, 1, 1] = 1
        w[i, 1, r, c] = 1
    directions = gmax(relu(conv(
        direction_input, w, -np.ones(4, np.float32)
    )))

    gate = add(blue, purple)
    black_or_button = add(black, buttons)
    goal_sum = conv(goals, np.ones((1, 2, 1, 1), np.float32))
    structural = add(add(wall, gate), goal_sum)

    plans = []
    solved_by_axis = []

    for gravity_axis in (0, 1):
        cross_axis = 1 - gravity_axis
        reduction = "colmax" if gravity_axis == 0 else "rowmax"
        gravity_coord = row if gravity_axis == 0 else col
        cross_coord = col if gravity_axis == 0 else row

        if gravity_axis == 0:
            matrix = np.array([
                [1, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
            ], np.float32)
        else:
            matrix = np.array([
                [0, 0, 1, -1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
            ], np.float32)

        direction_values = dense(directions, matrix)
        sign = sl(direction_values, 0, 1)
        positive = sl(direction_values, 1, 2)
        negative = sl(direction_values, 2, 3)
        axis_active = sl(direction_values, 3, 4)

        signed_coord = mul(gravity_coord, sign)
        upstream_pixel = add(affine(signed_coord, -1), negative)

        def cross_shift(x, amount):
            return shift(
                x,
                amount if cross_axis == 0 else 0,
                amount if cross_axis == 1 else 0,
            )

        # Display-space movement is three pixels, or four when scaled.
        small = gmax(mul(buttons, cross_shift(buttons, 3)))
        unit = affine(small, 1, 3)
        twice_unit = affine(unit, 2)
        unit_squared = vm(unit, unit)

        # Long axial strokes identify dividers.
        w = np.zeros((1, 1, 7, 7), np.float32)
        if gravity_axis == 0:
            w[0, 0, :, 3] = 1
        else:
            w[0, 0, 3, :] = 1
        long_stroke = clip(affine(conv(structural, w), 1, -4))
        divider_band = node(reduction, [long_stroke])
        floors = mul(wall, inverse(divider_band))
        free = mul(
            mul(inverse(divider_band), inverse(floors)), board
        )
        not_free = inverse(free)

        # Thin decorative rails are not platform interiors.
        broad_black = mul(
            black,
            mul(
                cross_shift(black_or_button, -1),
                cross_shift(black_or_button, 1),
            ),
        )
        height_seed = affine(
            mul(broad_black, affine(upstream_pixel, 1, 128)),
            1, -128,
        )
        signed_seed = affine(
            mul(free, affine(signed_coord, 1, 128)), 1, -128
        )
        transverse_seed = affine(
            mul(free, affine(cross_coord, -1, 128)), 1, -128
        )

        wall_pixels = mul(structural, divider_band)
        wall_front = masked_max(
            wall_pixels, upstream_pixel, reduction, -128
        )
        wall_present = node(reduction, [wall_pixels])
        limit_seed = add(
            mul(wall_present, affine(wall_front, -1)),
            affine(wall_present, 128, -128),
        )
        limit_seed = affine(
            mul(free, affine(limit_seed, 1, 128)), 1, -128
        )
        divider_anchor = add(
            mul(wall_present, affine(wall_front, -1)),
            affine(wall_present, 128, -128),
        )
        limit_seed = add(
            mul(limit_seed, free), mul(divider_anchor, not_free)
        )

        initial_fields = cat(
            height_seed, signed_seed, limit_seed, transverse_seed
        )
        own_seed = mul(players, free)

        def chamber_body(states):
            fields, own = states
            propagated = neighborhood_max(fields, True)
            fields_next = add(
                mul(propagated, free),
                mul(initial_fields, not_free),
            )
            own_next = mul(plane_dilate(own), free)
            return [fields_next, own_next]

        fields, own = loop(
            [initial_fields, own_seed], 64, chamber_body
        )
        height = sl(fields, 0, 1)
        downstream_max = sl(fields, 1, 2)
        upper_limit = affine(sl(fields, 2, 3), -1)
        transverse_min_negative = sl(fields, 3, 4)
        lower_limit = sub(affine(downstream_max, -1), positive)
        height = vmax(height, lower_limit)

        player_front = affine(
            masked_max(players, signed_coord), -1
        )
        player_front = va(player_front, affine(positive, -1))

        p_t_max = masked_max(players, cross_coord)
        p_t_min_negative = masked_max(
            players, affine(cross_coord, -1)
        )
        player_t = affine(
            va(p_t_max, affine(p_t_min_negative, -1)), .5
        )
        p_s_max = masked_max(players, signed_coord)
        p_s_min_negative = masked_max(
            players, affine(signed_coord, -1)
        )
        player_s = affine(
            va(p_s_max, affine(p_s_min_negative, -1)), .5
        )

        # Player bases measure their occupied platform exactly.
        occupied_height = add(
            height, mul(own, sub(player_front, height))
        )

        distance_goal = add(
            absolute(sub(cross_coord, player_t)),
            absolute(sub(signed_coord, player_s)),
        )
        goal_score = add(
            affine(distance_goal, -1), affine(goals, 256, -256)
        )
        best_goal = gmax(goal_score)
        goal_selected = mul(
            goals,
            relu(affine(
                absolute(sub(goal_score, best_goal)), -1, 1
            )),
        )
        goal_selected = mul(
            line_dilate(goal_selected, gravity_axis, 2), goals
        )
        goal_front = affine(
            masked_max(goal_selected, signed_coord), -1
        )
        goal_front = va(goal_front, affine(positive, -1))
        goal_t = masked_max(goal_selected, cross_coord)
        goal_s = masked_max(goal_selected, signed_coord)

        adjacent_goal = gmax(mul(
            line_dilate(own, cross_axis, 3), goal_selected
        ))
        exact_height = clip(affine(
            absolute(va(player_front, affine(goal_front, -1))),
            -1, 1,
        ))
        solved = vm(adjacent_goal, exact_height)
        solved_by_axis.append(vm(solved, axis_active))

        # Count movable reservoir material once per chamber.
        corner = mul(
            free,
            mul(
                relu(affine(
                    absolute(sub(signed_coord, downstream_max)),
                    -1, 1,
                )),
                relu(affine(
                    absolute(add(
                        cross_coord, transverse_min_negative
                    )),
                    -1, 1,
                )),
            ),
        )
        volume_map = mul(
            corner, relu(sub(height, lower_limit))
        )
        total_volume = affine(
            node("gmean", [volume_map]), 4096
        )
        own_lower = masked_max(own, lower_limit)
        geometric_capacity = va(own_lower, total_volume)
        capacity_ok = clip(affine(
            va(geometric_capacity, affine(goal_front, -1)),
            1, 1,
        ))
        own_upper = masked_max(own, upper_limit)
        bound_ok = clip(affine(
            va(own_upper, affine(goal_front, -1)), 1, 1
        ))
        use_goal = vm(adjacent_goal, vm(capacity_ok, bound_ok))

        # Otherwise select an adjacent bridge toward the goal.
        adjacent_gate = mul(
            gate, line_dilate(own, cross_axis, 3)
        )
        distance_gate = add(
            absolute(sub(cross_coord, goal_t)),
            absolute(sub(signed_coord, goal_s)),
        )
        gate_score = add(
            affine(distance_gate, -1),
            affine(adjacent_gate, 256, -256),
        )
        best_gate = gmax(gate_score)
        selected_gate = mul(
            adjacent_gate,
            relu(affine(
                absolute(sub(gate_score, best_gate)), -1, 1
            )),
        )

        def gate_body(states):
            return [mul(
                line_dilate(states[0], gravity_axis, 1), gate
            )]

        selected_gate = loop(
            [selected_gate], 24, gate_body
        )[0]
        gate_front = affine(
            masked_max(selected_gate, signed_coord), -1
        )
        gate_front = va(gate_front, affine(positive, -1))
        gate_t = masked_max(selected_gate, cross_coord)

        pair_seed = mul(
            line_dilate(selected_gate, cross_axis, 3), free
        )

        def pair_body(states):
            return [mul(plane_dilate(states[0]), free)]

        pair = loop([pair_seed], 64, pair_body)[0]
        target_region = add(
            mul(own, use_goal), mul(pair, inverse(use_goal))
        )
        desired_height = va(
            vm(goal_front, use_goal),
            vm(gate_front, inverse(use_goal)),
        )
        error = sub(desired_height, occupied_height)
        fixed_positive = mul(target_region, relu(error))
        fixed_negative = mul(
            target_region, relu(affine(error, -1))
        )
        fixed_potential = sub(fixed_positive, fixed_negative)
        fixed_pressure = cat(fixed_positive, fixed_negative)
        pressure_mask = cat(target_region, target_region)

        # Max-plus relaxation routes requests through intermediate chambers.
        def pressure_body(states):
            spread = relu(affine(
                neighborhood_max(states[0]), 1, -.025
            ))
            return [add(
                mul(spread, inverse(pressure_mask)),
                fixed_pressure,
            )]

        pressure = loop(
            [fixed_pressure], 64, pressure_body
        )[0]
        potential = sub(
            sl(pressure, 0, 2), sl(pressure, 2, 4)
        )

        left_wall = cross_shift(wall_present, -3)
        right_wall = cross_shift(wall_present, 3)

        def across(x):
            left6 = cross_shift(cross_shift(x, -3), -3)
            right6 = cross_shift(cross_shift(x, 3), 3)
            left8 = cross_shift(left6, -2)
            right8 = cross_shift(right6, 2)
            left = add(
                mul(left6, inverse(small)), mul(left8, small)
            )
            right = add(
                mul(right6, inverse(small)), mul(right8, small)
            )
            return add(
                mul(left, left_wall), mul(right, right_wall)
            )

        neighbor_height = across(occupied_height)
        neighbor_lower = across(lower_limit)
        neighbor_potential = across(potential)
        neighbor_target = across(target_region)
        neighbor_fixed = across(fixed_potential)

        # A button increases its own height by u and decreases its
        # neighbor by u. Exact decrease in target squared error:
        # 2u(E_own - E_neighbor) - u^2(I_own + I_neighbor).
        target_count = add(target_region, neighbor_target)
        direct_score = sub(
            mul(
                twice_unit,
                sub(fixed_potential, neighbor_fixed),
            ),
            mul(unit_squared, target_count),
        )

        # Pressure is used only when neither endpoint is a target.
        # In particular, it cannot overwrite an already-satisfied target.
        intermediate_edge = inverse(clip(target_count))
        routing_score = mul(
            intermediate_edge,
            mul(
                twice_unit,
                sub(potential, neighbor_potential),
            ),
        )
        transfer_score = add(direct_score, routing_score)

        helpful = clip(affine(relu(transfer_score), 16))
        transfer_score = add(
            transfer_score,
            affine(mul(helpful, target_count), .1),
        )
        aim_t = va(
            vm(goal_t, use_goal),
            vm(gate_t, inverse(use_goal)),
        )
        transfer_score = sub(
            transfer_score,
            affine(absolute(sub(cross_coord, aim_t)), .0001),
        )

        # Reservoir exhaustion and player clearance from transverse floors.
        floor_present = node(reduction, [floors])
        floor_front = affine(
            masked_max(floors, signed_coord, reduction), -1
        )
        floor_front = sub(floor_front, positive)
        floor_ahead = mul(
            floor_present,
            clip(sub(floor_front, occupied_height)),
        )
        floor_bound = add(
            floor_front, mul(own, vm(sign, twice_unit))
        )
        optional_floor_bound = add(
            mul(floor_ahead, floor_bound),
            affine(floor_ahead, -128, 128),
        )
        receiving_limit = vmin(
            upper_limit, optional_floor_bound
        )

        can_grow = clip(sub(receiving_limit, occupied_height))
        can_shrink_neighbor = clip(
            sub(neighbor_height, neighbor_lower)
        )
        button_center = mul(
            buttons,
            mul(
                cross_shift(buttons, -1),
                cross_shift(buttons, 1),
            ),
        )
        partner_exists = clip(add(left_wall, right_wall))
        legal_button = mul(
            button_center,
            mul(
                partner_exists,
                mul(can_grow, can_shrink_neighbor),
            ),
        )
        button_click = add(
            transfer_score,
            affine(legal_button, 100000, -100000),
        )

        pair_error = gmax(mul(
            pair,
            absolute(sub(gate_front, occupied_height)),
        ))
        bridge_ready = clip(affine(pair_error, -1, 1))
        bridge_ready = vm(bridge_ready, inverse(use_goal))
        bridge_cells = mul(selected_gate, bridge_ready)
        bridge_click = affine(
            bridge_cells, 110000, -100000
        )
        plan = vmax(button_click, bridge_click)
        plan = add(
            plan, affine(axis_active, 300000, -300000)
        )
        plans.append(plan)

    # Prioritise an unfinished player colour.
    solved = hard_vector(va(
        solved_by_axis[0], solved_by_axis[1]
    ))
    unfinished = vm(presence, inverse(solved))
    choose_first = hard_vector(sl(unfinished, 0, 1))
    choose_second = vm(
        sl(unfinished, 1, 2), inverse(choose_first)
    )
    colour_choice = vcat(choose_first, choose_second)

    combined = vmax(plans[0], plans[1])
    combined = add(
        combined, affine(colour_choice, 300000, -300000)
    )
    click_result = vmax(
        sl(combined, 0, 1), sl(combined, 1, 2)
    )
    graph[-1]["id"] = "click"
    shapes["click"] = shapes[click_result]

    logits_w = np.zeros((8, 8), np.float32)
    logits_b = np.zeros(8, np.float32)
    logits_b[6] = 1000
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": ["avail"],
        "w": parameter(logits_w),
        "b": parameter(logits_b),
    })

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

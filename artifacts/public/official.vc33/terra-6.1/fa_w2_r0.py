import numpy as np


def build():
    graph = []
    params = {}
    serial = [0]
    pserial = [0]

    def name():
        serial[0] += 1
        return "n%d" % serial[0]

    def param(value):
        pserial[0] += 1
        key = "p%d" % pserial[0]
        params[key] = np.asarray(value, dtype=np.float32)
        return key

    def node(op, inputs, **fields):
        ident = name()
        nd = {"id": ident, "op": op, "in": inputs}
        nd.update(fields)
        graph.append(nd)
        return ident

    def conv(x, weights, bias=None):
        fields = {"w": param(weights)}
        if bias is not None:
            fields["b"] = param(bias)
        return node("conv", [x], **fields)

    def scale(x, a, b=0.0):
        return conv(
            x,
            np.asarray([[[[a]]]], dtype=np.float32),
            np.asarray([b], dtype=np.float32),
        )

    def dense_scalar(xs, weights, bias=0.0):
        v = xs[0] if len(xs) == 1 else node("vconcat", xs)
        return node(
            "dense",
            [v],
            w=param(np.asarray([weights], dtype=np.float32)),
            b=param(np.asarray([bias], dtype=np.float32)),
        )

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def maximum(a, b):
        return node("max2", [a, b])

    def relu(x):
        return node("relu", [x])

    def abs_(x):
        return node("abs", [x])

    def gmax(x):
        return node("gmax", [x])

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    two = scale(one, 2.0)

    color0 = sl("x0", 0, 1)
    color1 = sl("x0", 1, 2)
    color4 = sl("x0", 4, 5)
    color5 = sl("x0", 5, 6)
    color9 = sl("x0", 9, 10)
    color11 = sl("x0", 11, 12)
    color12 = sl("x0", 12, 13)
    color14 = sl("x0", 14, 15)

    colors = node("concat", [color11, color14])
    gates = add(color1, color12)

    def clip01(x):
        return node("min2", [relu(x), one])

    def positive(x, gain=200.0):
        return clip01(scale(x, gain))

    inverse_cache = {}

    def inverse(x):
        if x not in inverse_cache:
            inverse_cache[x] = sub(one, x)
        return inverse_cache[x]

    def center(mask, coordinate):
        hi = gmax(mul(mask, coordinate))
        lo_inverse = gmax(mul(mask, inverse(coordinate)))
        return dense_scalar([hi, lo_inverse], [0.5, -0.5], 0.5)

    def select_nearest(mask, distance):
        score = mul(mask, sub(two, distance))
        best = gmax(score)
        gap = sub(best, score)
        peak = relu(sub(one, scale(gap, 200.0)))
        return mul(mask, peak)

    def flood(initial, mask, axis, steps, channels=1):
        w = np.zeros((channels, channels, 3, 3), dtype=np.float32)
        for c in range(channels):
            if axis == "row":
                w[c, c, :, 1] = 1.0
            elif axis == "col":
                w[c, c, 1, :] = 1.0
            else:
                w[c, c, :, :] = 1.0

        loop_id = name()
        state_id = name()
        conv_id = name()
        next_id = name()
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": steps,
            "state": [{"id": state_id, "init": initial}],
            "body": [
                {
                    "id": conv_id,
                    "op": "conv",
                    "in": [state_id],
                    "w": param(w),
                },
                {
                    "id": next_id,
                    "op": "min2",
                    "in": [conv_id, mask],
                },
            ],
            "next": {state_id: next_id},
        })
        return loop_id + "." + state_id

    def move_two(x, axis, displacement):
        w = np.zeros((1, 1, 5, 5), dtype=np.float32)
        if axis == "row":
            w[0, 0, 2 - displacement, 2] = 1.0
        else:
            w[0, 0, 2, 2 - displacement] = 1.0
        return conv(x, w)

    def move_four(x, axis, direction):
        return move_two(
            move_two(x, axis, 2 * direction),
            axis,
            2 * direction,
        )

    def water_surface(line, gravity_coordinate, gate_center, axis):
        line_mask = clip01(scale(line, 2.0))
        water_line = mul(color0, line_mask)
        dist = abs_(sub(gravity_coordinate, gate_center))
        seed = select_nearest(water_line, dist)
        component = flood(seed, water_line, axis, 64)
        return gmax(mul(component, inverse(gravity_coordinate)))

    branches = []
    directions = [
        ("row", 1, (-1, 0)),
        ("col", 1, (0, -1)),
        ("col", -1, (0, 1)),
    ]

    for gravity_axis, sign, neighbour in directions:
        if gravity_axis == "row":
            gravity_coord = row if sign > 0 else inverse(row)
            cross_coord = col
            cross_axis = "col"
        else:
            gravity_coord = col if sign > 0 else inverse(col)
            cross_coord = row
            cross_axis = "row"

        # Locate the boat base and its colour.
        near_w = np.zeros((1, 1, 3, 3), dtype=np.float32)
        near_w[0, 0, 1 + neighbour[0], 1 + neighbour[1]] = 1.0
        near4 = conv(color4, near_w)
        seed_input = node("concat", [colors, near4])
        seed_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
        seed_w[0, 0, 0, 0] = 1.0
        seed_w[1, 1, 0, 0] = 1.0
        seed_w[:, 2, 0, 0] = 1.0
        seeds = relu(conv(seed_input, seed_w, [-1.0, -1.0]))
        boat2 = flood(seeds, colors, "both", 8, channels=2)

        present_by_color = gmax(seeds)
        matching_colors = mul(colors, present_by_color)
        sum_two = np.ones((1, 2, 1, 1), dtype=np.float32)
        boat = conv(boat2, sum_two)
        matching = conv(matching_colors, sum_two)
        marker = relu(sub(matching, boat))

        active = node("vmul", [gmax(boat), gmax(marker)])
        boat_cross = center(boat, cross_coord)
        marker_cross = center(marker, cross_coord)
        boat_gravity = gmax(mul(boat, gravity_coord))
        marker_gravity = gmax(mul(marker, gravity_coord))

        # Detect a barrier between the boat and marker along the boat's
        # present lane. This can make a gate on the marker wall useful.
        boat_line = relu(
            sub(one, scale(abs_(sub(cross_coord, boat_cross)), 63.0))
        )
        along_a = sub(gravity_coord, boat_gravity)
        along_b = sub(marker_gravity, gravity_coord)
        along = positive(mul(along_a, along_b), 20000.0)
        obstruction = gmax(mul(mul(color5, boat_line), along))

        between_a = sub(cross_coord, boat_cross)
        between_b = sub(marker_cross, cross_coord)
        between = positive(mul(between_a, between_b), 20000.0)

        # Normally ignore a gate coincident with the marker's wall. Allow
        # it if the current lane is obstructed.
        marker_separation = positive(
            scale(abs_(sub(cross_coord, marker_cross)), 1.0, -2.1 / 63.0),
            63.0,
        )
        gate_allowed = maximum(marker_separation, mul(one, obstruction))
        gate_candidates = mul(mul(gates, between), gate_allowed)
        gate_distance = add(
            abs_(sub(cross_coord, boat_cross)),
            scale(abs_(sub(gravity_coord, boat_gravity)), 0.2),
        )
        gate_seed = select_nearest(gate_candidates, gate_distance)
        gate_component = flood(gate_seed, gates, "both", 16)
        gate_exists = gmax(gate_candidates)

        gate_cross = center(gate_component, cross_coord)
        gate_gravity_center = center(gate_component, gravity_coord)
        gate_front = gmax(mul(gate_component, gravity_coord))
        desired_surface = dense_scalar(
            [gate_front], [-1.0], 1.0 - 1.0 / 63.0
        )

        center_line = relu(
            sub(one, scale(abs_(sub(cross_coord, gate_cross)), 63.0))
        )
        left_line = move_four(center_line, cross_axis, -1)
        right_line = move_four(center_line, cross_axis, 1)
        left_surface = water_surface(
            left_line, gravity_coord, gate_gravity_center, gravity_axis
        )
        right_surface = water_surface(
            right_line, gravity_coord, gate_gravity_center, gravity_axis
        )

        left_error = dense_scalar(
            [left_surface, desired_surface], [1.0, -1.0]
        )
        right_error = dense_scalar(
            [right_surface, desired_surface], [1.0, -1.0]
        )
        left_error_map = mul(one, left_error)
        right_error_map = mul(one, right_error)

        lp = positive(left_error_map)
        ln = positive(scale(left_error_map, -1.0))
        rp = positive(right_error_map)
        rn = positive(scale(right_error_map, -1.0))
        left_nonzero = maximum(lp, ln)
        right_nonzero = maximum(rp, rn)

        recipient_left = clip01(add(ln, mul(inverse(left_nonzero), rp)))
        recipient_right = clip01(add(lp, mul(inverse(left_nonzero), rn)))

        side_left = positive(sub(gate_cross, cross_coord), 126.0)
        side_right = positive(sub(cross_coord, gate_cross), 126.0)
        wanted_side = add(
            mul(recipient_left, side_left),
            mul(recipient_right, side_right),
        )
        wanted_buttons = mul(color9, wanted_side)
        gate_button_distance = add(
            abs_(sub(cross_coord, gate_cross)),
            scale(abs_(sub(gravity_coord, gate_gravity_center)), 0.25),
        )
        transfer_click = select_nearest(
            wanted_buttons, gate_button_distance
        )

        both_correct = inverse(maximum(left_nonzero, right_nonzero))
        unlocked = gmax(mul(gate_component, color12))
        click_gate = maximum(both_correct, mul(one, unlocked))
        gate_policy = add(
            mul(gate_component, click_gate),
            mul(transfer_click, inverse(click_gate)),
        )
        gate_policy = mul(gate_policy, gate_exists)

        # Otherwise, change the boat's water level toward the marker.
        boat_button_distance = abs_(sub(cross_coord, boat_cross))
        near_button = select_nearest(color9, boat_button_distance)
        near_component = flood(near_button, color9, cross_axis, 8)
        near_center = center(near_component, cross_coord)

        direction_to_button = dense_scalar(
            [near_center, boat_cross], [1.0, -1.0]
        )
        toward_pair = positive(
            mul(
                mul(one, direction_to_button),
                sub(cross_coord, near_center),
            ),
            20000.0,
        )
        pair_mask = mul(
            mul(color9, inverse(near_component)),
            toward_pair,
        )
        pair_button = select_nearest(
            pair_mask, abs_(sub(cross_coord, near_center))
        )
        pair_exists = gmax(pair_mask)
        needs_pair_v = dense_scalar(
            [marker_gravity, boat_gravity], [1.0, -1.0]
        )
        needs_pair = positive(mul(one, needs_pair_v), 63.0)
        use_pair = mul(needs_pair, pair_exists)
        align_policy = add(
            mul(near_button, inverse(use_pair)),
            mul(pair_button, use_pair),
        )
        align_policy = mul(
            align_policy,
            dense_scalar([gate_exists], [-1.0], 1.0),
        )
        branches.append(mul(add(gate_policy, align_policy), active))

    combined = branches[0]
    for branch in branches[1:]:
        combined = maximum(combined, branch)

    # A compact observation fingerprint, excluding the progress bar.
    colored = add(color11, color14)
    fingerprint_maps = node("concat", [
        mul(color0, row),
        mul(color0, col),
        mul(colored, row),
        mul(colored, col),
        color12,
    ])
    fingerprint = node("gmean", [fingerprint_maps])
    old_fingerprint = vsl("mem", 0, 5)

    fp_now_map = mul(fingerprint_maps, scale(one, 0.0))
    # Broadcast the vector fingerprints into spatial tensors for subtraction.
    current_broadcast = add(fp_now_map, fingerprint)
    previous_broadcast = add(fp_now_map, old_fingerprint)
    fp_difference = abs_(sub(current_broadcast, previous_broadcast))
    fp_sum = conv(
        fp_difference,
        np.ones((1, 5, 1, 1), dtype=np.float32),
    )
    changed = positive(fp_sum, 100000.0)
    valid_previous = vsl("mem", 7, 8)
    failed = mul(inverse(changed), valid_previous)

    old_row = vsl("mem", 5, 6)
    old_col = vsl("mem", 6, 7)
    dr = abs_(sub(row, old_row))
    dc = abs_(sub(col, old_col))

    # Avoid the previous button and the nearby opposite button. Select a
    # different transfer point when the last click had no visible effect.
    far_row = positive(scale(dr, 1.0, -12.5 / 63.0), 63.0)
    far_col = positive(scale(dc, 1.0, -12.5 / 63.0), 63.0)
    alternative_mask = mul(color9, maximum(far_row, far_col))
    alternative_distance = scale(add(dr, dc), 0.5)
    alternative = select_nearest(alternative_mask, alternative_distance)
    alternative_exists = gmax(alternative_mask)
    use_fallback = mul(failed, alternative_exists)
    final_map = add(
        mul(combined, inverse(use_fallback)),
        mul(alternative, use_fallback),
    )

    graph.append({
        "id": "click",
        "op": "max2",
        "in": [final_map, final_map],
    })

    # Recover the decoder's first maximal cell for the next memory vector.
    best_click = gmax(final_map)
    click_gap = sub(best_click, final_map)
    click_peak = relu(sub(one, scale(click_gap, 10000.0)))
    click_peak = mul(click_peak, positive(final_map, 10000.0))
    min_row_inverse = gmax(mul(click_peak, inverse(row)))
    chosen_row = dense_scalar([min_row_inverse], [-1.0], 1.0)
    row_match = relu(
        sub(one, scale(abs_(sub(row, chosen_row)), 126.0))
    )
    first_row_peaks = mul(click_peak, row_match)
    min_col_inverse = gmax(mul(first_row_peaks, inverse(col)))
    chosen_col = dense_scalar([min_col_inverse], [-1.0], 1.0)
    valid = gmax(one)

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [fingerprint, chosen_row, chosen_col, valid],
    })

    features = gmax("x0")
    logits_w = np.zeros((8, 16), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": param(logits_w),
        "b": param(logits_b),
    })

    return {"graph": graph, "memory_size": 8, "params": params}

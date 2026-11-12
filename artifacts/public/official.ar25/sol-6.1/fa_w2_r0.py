import numpy as np


def build():
    graph = []
    params = {}
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return "%s_%04d" % (prefix, counter[0])

    def parameter(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        name = name or fresh("n")
        record = {"id": name, "op": op, "in": list(inputs)}
        record.update(fields)
        graph.append(record)
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def saffine(x, channels, scale, bias=0.0):
        return conv(
            x,
            np.eye(channels, dtype=np.float32) * scale,
            np.full(channels, bias, dtype=np.float32),
        )

    def affine(x, scale, bias=0.0):
        return saffine(x, 1, scale, bias)

    def sslice(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vslice(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

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

    def absolute(x):
        return node("abs", [x])

    def gmax(x):
        return node("gmax", [x])

    def gmean(x):
        return node("gmean", [x])

    def vmul(a, b):
        return node("vmul", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    one = dense("aprev", np.zeros((1, 8)), [1.0])

    def clip01(x):
        return node("min2", [relu(x), one])

    def vclip01(x, channels):
        positive = node("vrelu", [x])
        inverse = dense(
            positive,
            -np.eye(channels, dtype=np.float32),
            np.ones(channels, dtype=np.float32),
        )
        inverse = node("vrelu", [inverse])
        return dense(
            inverse,
            -np.eye(channels, dtype=np.float32),
            np.ones(channels, dtype=np.float32),
        )

    # Canonical scale-three cell lattice.
    centers = 3.0 * np.arange(21, dtype=np.float32) + 1.0
    xy = sslice("coords", 0, 2)
    yy = sslice(xy, 0, 1)
    xx = sslice(xy, 1, 2)

    lattice_w = np.zeros((42, 2), dtype=np.float32)
    lattice_w[:21, 0] = 63.0
    lattice_w[21:, 1] = 63.0
    distances = conv(
        xy, lattice_w, -np.concatenate([centers, centers])
    )
    pulses = clip01(relu(saffine(
        absolute(distances), 42, -4.0, 2.0
    )))
    row_bands = sslice(pulses, 0, 21)
    col_bands = sslice(pulses, 21, 42)
    row_lattice = conv(row_bands, np.ones((1, 21)))
    col_lattice = conv(col_bands, np.ones((1, 21)))
    lattice = mul(row_lattice, col_lattice)

    # Corners reveal real geometry; centers reveal targets and ownership.
    corner_w = np.zeros((3, 16, 3, 3), dtype=np.float32)
    for out, color in enumerate((5, 10, 11)):
        corner_w[out, color, 0, 0] = 1.0
    corners = conv("x0", corner_w)
    corner_shape = sslice(corners, 0, 1)
    corner_axis = sslice(corners, 1, 2)
    corner_goal = sslice(corners, 2, 3)

    zero_color = sslice("x0", 0, 1)
    five_color = sslice("x0", 5, 6)
    nine_color = sslice("x0", 9, 10)
    goal_color = sslice("x0", 11, 12)

    real_shape = mul(lattice, corner_shape)
    axis_pixels = mul(lattice, corner_axis)
    goals = mul(lattice, goal_color)
    uncovered = mul(goals, corner_goal)

    # Long palette-10 lines identify mirrors.
    h_count = node("rowsum", [axis_pixels])
    v_count = node("colsum", [axis_pixels])
    horizontal = mul(
        lattice, clip01(affine(h_count, 1.0, -4.0))
    )
    vertical = mul(
        lattice, clip01(affine(v_count, 1.0, -4.0))
    )
    axis_positions = gmax(cat(
        mul(horizontal, yy), mul(vertical, xx)
    ))
    axis_exists = gmax(cat(horizontal, vertical))

    # Several unobscured dots distinguish selection from a crossing.
    axis_zero = mul(axis_pixels, zero_color)
    axis_marker = mul(
        axis_pixels, clip01(add(zero_color, nine_color))
    )
    selected_h_flag = clip01(affine(
        node("rowsum", [axis_zero]), 1.0, -2.0
    ))
    selected_v_flag = clip01(affine(
        node("colsum", [axis_zero]), 1.0, -2.0
    ))
    movable_h_flag = clip01(affine(
        node("rowsum", [axis_marker]), 1.0, -2.0
    ))
    movable_v_flag = clip01(affine(
        node("colsum", [axis_marker]), 1.0, -2.0
    ))

    selected_horizontal = mul(horizontal, selected_h_flag)
    selected_vertical = mul(vertical, selected_v_flag)
    movable_horizontal = mul(horizontal, movable_h_flag)
    movable_vertical = mul(vertical, movable_v_flag)

    selected_axis = gmax(maximum(
        selected_horizontal, selected_vertical
    ))
    no_selected_axis = dense(selected_axis, [[-1.0]], [1.0])
    shape_seeds = mul(
        mul(real_shape, zero_color), no_selected_axis
    )
    shape_present = gmax(shape_seeds)

    # Division-free comparison to the target centroid.
    goal_count = gmean(goals)
    goal_moments = gmean(mul(goals, xy))
    centroid_difference = sub(goal_moments, mul(xy, goal_count))
    half_cell_count = dense(goal_count, [[1.0 / 42.0]])

    drift = clip01(saffine(
        relu(sub(absolute(centroid_difference), half_cell_count)),
        2, 1.0e6
    ))
    row_drift = sslice(drift, 0, 1)
    col_drift = sslice(drift, 1, 2)
    off_axes = maximum(
        mul(movable_horizontal, row_drift),
        mul(movable_vertical, col_drift),
    )
    any_off_axis = gmax(off_axes)
    no_off_axis = dense(any_off_axis, [[-1.0]], [1.0])

    axis_click_candidates = mul(
        mul(off_axes, axis_pixels), nine_color
    )
    shape_click_candidates = mul(real_shape, nine_color)
    axis_click_present = gmax(axis_click_candidates)
    shape_click_present = gmax(shape_click_candidates)

    conv(
        cat(axis_click_candidates, shape_click_candidates),
        [[100.0, 1.0]],
        name="click",
    )

    direction_w = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    axis_directions = conv(centroid_difference, direction_w)
    axis_directions = clip01(saffine(
        relu(sub(axis_directions, half_cell_count)),
        4, 1.0e6
    ))
    selected_axis_fields = cat(
        selected_horizontal, selected_horizontal,
        selected_vertical, selected_vertical,
    )
    axis_moves = gmax(mul(axis_directions, selected_axis_fields))

    # Reflection indicator banks.
    def reflected_bands(coordinate, position):
        base = conv(
            coordinate,
            np.full((21, 1), 63.0, dtype=np.float32),
            centers,
        )
        shift = dense(
            position, np.full((21, 1), -126.0, dtype=np.float32)
        )
        distance = absolute(add(base, shift))
        return clip01(relu(saffine(distance, 21, -4.0, 2.0)))

    h_position = vslice(axis_positions, 0, 1)
    v_position = vslice(axis_positions, 1, 2)
    h_exists = vslice(axis_exists, 0, 1)
    v_exists = vslice(axis_exists, 1, 2)
    reflected_rows = reflected_bands(yy, h_position)
    reflected_cols = reflected_bands(xx, v_position)

    def reflected_image(image, bands, reflected, reduction, exists):
        separated = node(reduction, [mul(image, bands)])
        result = conv(
            mul(separated, reflected), np.ones((1, 21))
        )
        return mul(result, exists)

    def fold_union(image):
        h_image = reflected_image(
            image, row_bands, reflected_rows, "colsum", h_exists
        )
        result = maximum(image, h_image)
        v_image = reflected_image(
            result, col_bands, reflected_cols, "rowsum", v_exists
        )
        return maximum(result, v_image)

    h_correction = affine(mul(horizontal, h_exists), -0.5, 1.0)
    v_correction = affine(mul(vertical, v_exists), -0.5, 1.0)

    def fold_counts(image):
        # Axis points count once rather than as two identical reflections.
        h_image = reflected_image(
            image, row_bands, reflected_rows, "colsum", h_exists
        )
        result = mul(add(image, h_image), h_correction)
        v_image = reflected_image(
            result, col_bands, reflected_cols, "rowsum", v_exists
        )
        return mul(add(result, v_image), v_correction)

    # Ownership-aware flood fills.
    flood_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for r, c in ((3, 3), (0, 3), (6, 3), (3, 0), (3, 6)):
        flood_w[0, 0, r, c] = 1.0
    flood_parameter = parameter(flood_w)

    def flood(seeds, allowed):
        loop_id = fresh("flood")
        state_id = fresh("state")
        neighbor_id = fresh("neighbor")
        bounded_id = fresh("bounded")
        grown_id = fresh("grown")
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": 32,
            "state": [{"id": state_id, "init": seeds}],
            "body": [
                {
                    "id": neighbor_id, "op": "conv",
                    "in": [state_id], "w": flood_parameter,
                },
                {
                    "id": bounded_id, "op": "min2",
                    "in": [neighbor_id, one],
                },
                {
                    "id": grown_id, "op": "mul",
                    "in": [bounded_id, allowed],
                },
            ],
            "next": {state_id: grown_id},
        })
        return loop_id + "." + state_id

    selected_allowed = mul(
        real_shape, clip01(add(zero_color, goal_color))
    )
    component = flood(shape_seeds, selected_allowed)

    other_seeds = mul(
        real_shape, clip01(add(nine_color, five_color))
    )
    other_allowed = maximum(
        other_seeds, mul(real_shape, goal_color)
    )
    other_component = flood(other_seeds, other_allowed)
    other_coverage = clip01(fold_union(other_component))

    # Favor coverage not already supplied by other shapes.
    marginal_goals = mul(goals, sub(one, other_coverage))
    marginal_goals = maximum(marginal_goals, uncovered)
    folded_goals = fold_counts(goals)
    folded_marginal = fold_counts(marginal_goals)

    component_min_inverse = gmax(mul(component, sub(one, xy)))
    visual_min = dense(
        component_min_inverse,
        -np.eye(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    component_max = gmax(mul(component, xy))

    # Memory: destination (2), current origin (2), tracking-valid (1).
    old_destination = vslice("mem", 0, 2)
    old_origin = vslice("mem", 2, 4)
    old_valid = vslice("mem", 4, 5)
    unchanged_selection = dense(
        "aprev",
        [[-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]],
        [1.0],
    )
    continuing = vmul(
        vmul(old_valid, unchanged_selection), no_off_axis
    )
    shape_enabled = vmul(
        vclip01(vadd(shape_present, continuing), 1), no_off_axis
    )
    replace = dense(continuing, [[-1.0]], [1.0])

    step_w = np.zeros((2, 8), dtype=np.float32)
    step_w[0, 1] = -3.0 / 63.0
    step_w[0, 2] = 3.0 / 63.0
    step_w[1, 3] = -3.0 / 63.0
    step_w[1, 4] = 3.0 / 63.0
    previous_step = dense("aprev", step_w)
    tracked_origin = vadd(old_origin, previous_step)
    current_min = vadd(
        vmul(tracked_origin, continuing),
        vmul(visual_min, replace),
    )

    extent = dense(
        vcat(component_max, visual_min),
        [[1.0, 0.0, -1.0, 0.0],
         [0.0, 1.0, 0.0, -1.0]],
    )
    anchor_distance = conv(
        absolute(sub(xy, visual_min)), [[63.0, 63.0]]
    )
    anchor = clip01(affine(anchor_distance, -4.0, 2.0))

    # Shared shifts create a 12-by-12 dynamic offset bank.
    shift_x_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    shift_x_w[0, 0, 3, 6] = 1.0
    shift_x_parameter = parameter(shift_x_w)

    shift_y_w = np.zeros((12, 12, 7, 7), dtype=np.float32)
    for c in range(12):
        shift_y_w[c, c, 6, 3] = 1.0
    shift_y_parameter = parameter(shift_y_w)

    def shift_bank(image):
        horizontal_images = [image]
        for _ in range(11):
            horizontal_images.append(node(
                "conv", [horizontal_images[-1]], w=shift_x_parameter
            ))
        horizontal_bank = cat(*horizontal_images)
        vertical_images = [horizontal_bank]
        for _ in range(11):
            vertical_images.append(node(
                "conv", [vertical_images[-1]], w=shift_y_parameter
            ))
        return cat(*vertical_images)

    component_bank = shift_bank(component)
    target_bank = shift_bank(folded_goals)
    marginal_bank = shift_bank(folded_marginal)

    template = gmax(mul(component_bank, anchor))
    full_match = conv(
        mul(target_bank, template), np.ones((1, 144))
    )
    marginal_match = conv(
        mul(marginal_bank, template), np.ones((1, 144))
    )
    score = conv(
        cat(full_match, marginal_match), [[0.01, 1.0]]
    )

    # Conservative correction for overlapping copies of the same shape.
    # Touching an axis is allowed; crossing it discounts repeated orbits.
    crossing_margin = 0.5 / 63.0
    before_axis = clip01(saffine(
        sub(axis_positions, xy),
        2, 4096.0, -4096.0 * crossing_margin,
    ))
    after_axis = clip01(saffine(
        sub(add(xy, extent), axis_positions),
        2, 4096.0, -4096.0 * crossing_margin,
    ))
    crossings = mul(
        mul(before_axis, after_axis), axis_exists
    )
    dimension_discounts = saffine(crossings, 2, -0.5, 1.0)
    reflection_discount = mul(
        sslice(dimension_discounts, 0, 1),
        sslice(dimension_discounts, 1, 2),
    )
    score = mul(score, reflection_discount)

    travel_cost = conv(
        absolute(sub(xy, current_min)), [[0.0001, 0.0001]]
    )
    score = sub(score, travel_cost)

    overflow = relu(saffine(
        add(xy, extent), 2, 1.0, -(61.0 / 63.0 + 1.0e-5)
    ))
    overflow_cost = conv(overflow, [[10000.0, 10000.0]])
    non_cell_cost = affine(lattice, -1000.0, 1000.0)
    score = sub(sub(score, overflow_cost), non_cell_cost)

    # Extract a destination using reductions and clipped equality masks.
    best_score = gmax(score)
    winners = relu(affine(sub(best_score, score), -1.0e6, 1.0))
    destination_y_inverse = gmax(mul(winners, sub(one, yy)))
    destination_y = dense(destination_y_inverse, [[-1.0]], [1.0])

    winning_row = clip01(affine(
        absolute(sub(yy, destination_y)), -252.0, 2.0
    ))
    destination_x_inverse = gmax(mul(
        mul(winners, winning_row), sub(one, xx)
    ))
    destination_x = dense(destination_x_inverse, [[-1.0]], [1.0])
    new_destination = vcat(destination_y, destination_x)

    destination = vadd(
        vmul(old_destination, continuing),
        vmul(new_destination, replace),
    )
    node(
        "vconcat",
        [destination, current_min, shape_enabled],
        name="mem_next",
    )

    # Follow the tracked destination, vertically before horizontally.
    shape_direction_w = np.array([
        [-1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -1.0],
    ], dtype=np.float32) * 1024.0
    shape_directions = dense(
        vcat(destination, current_min),
        shape_direction_w,
        np.full(4, -1024.0 * 0.5 / 63.0, dtype=np.float32),
    )
    shape_directions = vmul(
        vclip01(shape_directions, 4), shape_enabled
    )

    decision_features = vcat(
        axis_moves, shape_directions,
        axis_click_present, shape_click_present,
    )
    output_w = np.zeros((8, 10), dtype=np.float32)
    output_b = np.full(8, -100.0, dtype=np.float32)

    for direction, action in enumerate((1, 2, 3, 4)):
        output_w[action, direction] = 100.0
        output_w[action, 4 + direction] = (
            20.0 if direction < 2 else 19.0
        )
        output_b[action] = -3.0

    output_b[5] = 1.0
    output_w[6, 8] = 60.0
    output_w[6, 9] = 10.0
    output_b[6] = -2.0
    dense(decision_features, output_w, output_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 5,
        "params": params,
    }

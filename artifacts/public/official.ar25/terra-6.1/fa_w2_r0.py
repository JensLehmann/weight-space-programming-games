import numpy as np


def build():
    graph = []
    params = {}
    current = graph
    serial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return f"{prefix}{serial}"

    def node(op, *inputs, **fields):
        name = fresh()
        current.append({"id": name, "op": op, "in": list(inputs), **fields})
        return name

    def parameter(value):
        name = fresh("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def conv(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("conv", x, **fields)

    def dense(x, weights, bias=None):
        fields = {"w": parameter(weights)}
        if bias is not None:
            fields["b"] = parameter(bias)
        return node("dense", x, **fields)

    def affine(x, scale=1.0, bias=0.0):
        return dense(x, [[scale]], [bias])

    def spatial_affine(x, scale=1.0, bias=0.0):
        return conv(x, [[[[scale]]]], [bias])

    def vadd(a, b):
        return node("vadd", a, b)

    def vmul(a, b):
        return node("vmul", a, b)

    def vsub(a, b):
        return vadd(a, affine(b, -1.0))

    def inverse(a):
        return affine(a, -1.0, 1.0)

    def clip01(a):
        positive = node("vrelu", a)
        excess = node("vrelu", affine(a, 1.0, -1.0))
        return vsub(positive, excess)

    def indicator(a, threshold=0.5, slope=2.0):
        return clip01(affine(a, slope, -slope * threshold))

    def spatial_clip01(a):
        return node("min2", node("relu", a), one)

    def shift(x, dx, dy):
        if max(abs(dx), abs(dy)) <= 1:
            size, center = 3, 1
        else:
            size, center = 7, 3
        w = np.zeros((1, 1, size, size), dtype=np.float32)
        w[0, 0, center + dy, center + dx] = 1.0
        return conv(x, w)

    def plane(index):
        return node("slice", "x0", start=index, stop=index + 1)

    def coord(index):
        return node("slice", "coords", start=index, stop=index + 1)

    def vsum(items):
        result = items[0]
        for item in items[1:]:
            result = vadd(result, item)
        return result

    def ssum(items):
        result = items[0]
        for item in items[1:]:
            result = node("add", result, item)
        return result

    def select_vec(flag, yes, no):
        return vadd(vmul(flag, yes), vmul(inverse(flag), no))

    one = coord(2)
    cy = coord(0)
    cx = coord(1)
    p0 = plane(0)
    p5 = plane(5)
    p9 = plane(9)
    p10 = plane(10)
    p11 = plane(11)

    def fixed_stripes(axis):
        w = np.zeros((21, 3, 1, 1), dtype=np.float32)
        w[:, axis, 0, 0] = 63.0
        b = -3.0 * np.arange(21, dtype=np.float32)
        distance = node("abs", conv("coords", w, b))
        w2 = np.zeros((21, 21, 1, 1), dtype=np.float32)
        for i in range(21):
            w2[i, i, 0, 0] = -2.0
        return node(
            "relu",
            conv(distance, w2, np.ones(21, dtype=np.float32)),
        )

    xs = fixed_stripes(1)
    ys = fixed_stripes(0)
    sum21 = np.ones((1, 21, 1, 1), dtype=np.float32)
    xgrid = conv(xs, sum21)
    ygrid = conv(ys, sum21)
    grid = node("mul", xgrid, ygrid)
    center_grid = shift(grid, -1, -1)

    goal_center = shift(p11, 1, 1)
    goal = node("mul", grid, node("max2", p11, goal_center))
    uncovered = node("mul", grid, p11)
    black = node("mul", grid, p5)

    def long_line(x, reduction):
        counts = node(reduction, x)
        excess = node("relu", spatial_affine(counts, 1.0, -18.0))
        return node("min2", excess, one)

    vertical = long_line(p10, "colsum")
    horizontal = long_line(p10, "rowsum")
    vertical_grid = node("mul", vertical, grid)
    horizontal_grid = node("mul", horizontal, grid)
    vertical_present = indicator(node("gmax", vertical_grid))
    horizontal_present = indicator(node("gmax", horizontal_grid))

    def axis_coordinate(axis_grid, coordinate):
        return affine(
            node("gmax", node("mul", axis_grid, coordinate)),
            63.0,
        )

    axis_x = axis_coordinate(vertical_grid, cx)
    axis_y = axis_coordinate(horizontal_grid, cy)

    def count_on_line(line, colors):
        z = node("mul", line, center_grid)
        z = node("mul", z, colors)
        return affine(node("gmean", z), 4096.0)

    selectable_colors = node("add", p0, p9)
    v_mobile = indicator(
        count_on_line(vertical, selectable_colors), 4.5, 1.0
    )
    h_mobile = indicator(
        count_on_line(horizontal, selectable_colors), 4.5, 1.0
    )
    v_selected = indicator(count_on_line(vertical, p0), 10.5, 1.0)
    h_selected = indicator(count_on_line(horizontal, p0), 10.5, 1.0)
    v_mobile = vmul(v_mobile, vertical_present)
    h_mobile = vmul(h_mobile, horizontal_present)
    mirror_selected = clip01(vadd(v_selected, h_selected))

    def target_midpoint(coordinate):
        pixel_coordinate = spatial_affine(coordinate, 63.0)
        high = node(
            "gmax",
            node("mul", goal, spatial_affine(pixel_coordinate, 1.0, 1.0)),
        )
        low = node(
            "gmax",
            node("mul", goal, spatial_affine(pixel_coordinate, -1.0, 64.0)),
        )
        return dense(node("vconcat", high, low), [[0.5, -0.5]], [31.5])

    target_x = target_midpoint(cx)
    target_y = target_midpoint(cy)

    def direction_flags(axis, target):
        delta = vsub(axis, target)
        negative = clip01(affine(delta, 2.0, -3.1))
        positive = clip01(affine(delta, -2.0, -3.1))
        return negative, positive

    h_up, h_down = direction_flags(axis_y, target_y)
    v_left, v_right = direction_flags(axis_x, target_x)
    h_need = vmul(h_mobile, clip01(vadd(h_up, h_down)))
    v_need = vmul(v_mobile, clip01(vadd(v_left, v_right)))
    v_active = vmul(inverse(h_need), v_need)
    shape_active = vmul(inverse(h_need), inverse(v_need))

    h_move_up = vmul(vmul(h_mobile, h_up), h_selected)
    h_move_down = vmul(vmul(h_mobile, h_down), h_selected)
    v_move_left = vmul(vmul(v_active, v_left), v_selected)
    v_move_right = vmul(vmul(v_active, v_right), v_selected)
    h_click = vmul(h_need, inverse(h_selected))
    v_click = vmul(v_active, inverse(v_selected))

    # Recover a selected shape from its center marks and goal-covered cells.
    selected_center = shift(p0, 1, 1)
    goal_at_center = shift(p11, 1, 1)
    seed = node("mul", black, selected_center)
    seed = node("mul", seed, inverse(mirror_selected))
    allowed = node(
        "mul", black, node("max2", selected_center, goal_at_center)
    )

    flood_state = fresh("state")
    flood_id = fresh("loop")
    flood_body = []
    old_current = current
    current = flood_body
    flood_kernel = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            flood_kernel[0, 0, dy + 3, dx + 3] = 1.0
    grown = conv(flood_state, flood_kernel)
    grown = node("min2", grown, one)
    grown = node("mul", grown, allowed)
    current = old_current
    graph.append(
        {
            "id": flood_id,
            "op": "loop",
            "T": 16,
            "state": [{"id": flood_state, "init": seed}],
            "body": flood_body,
            "next": {flood_state: grown},
        }
    )
    observed_selected = f"{flood_id}.{flood_state}"

    def min_coordinate(mask, coordinate):
        pixel = spatial_affine(coordinate, 63.0)
        reverse = spatial_affine(pixel, -1.0, 64.0)
        maximum = node("gmax", node("mul", mask, reverse))
        return affine(maximum, -1.0, 64.0)

    def max_coordinate(mask, coordinate):
        pixel = spatial_affine(coordinate, 63.0)
        return node("gmax", node("mul", mask, pixel))

    def relative_stripes(coordinate, anchor, count):
        positions = dense(
            anchor,
            np.ones((count, 1), dtype=np.float32),
            3.0 * np.arange(count, dtype=np.float32),
        )
        pixel = spatial_affine(coordinate, 63.0)
        difference = node("abs", node("sub", pixel, positions))
        w = np.zeros((count, count, 1, 1), dtype=np.float32)
        for i in range(count):
            w[i, i, 0, 0] = -2.0
        return node(
            "relu",
            conv(difference, w, np.ones(count, dtype=np.float32)),
        )

    size = 11
    wx = np.zeros((size * size, size, 1, 1), dtype=np.float32)
    wy = np.zeros_like(wx)
    for dy in range(size):
        for dx in range(size):
            i = dy * size + dx
            wx[i, dx, 0, 0] = 1.0
            wy[i, dy, 0, 0] = 1.0

    def shape_locations(ax, ay):
        rx = relative_stripes(cx, ax, size)
        ry = relative_stripes(cy, ay, size)
        locations = node("mul", conv(rx, wx), conv(ry, wy))
        return rx, ry, locations

    observed_x = min_coordinate(observed_selected, cx)
    observed_y = min_coordinate(observed_selected, cy)
    _, _, observed_locations = shape_locations(observed_x, observed_y)
    observed_coeff = node(
        "gmax", node("mul", observed_locations, observed_selected)
    )

    # Memory: 121 footprint bits, anchor x/y, shape-valid flag,
    # consecutive-click count, taboo x/y, taboo-valid flag.
    saved_coeff = node("vslice", "mem", start=0, stop=121)
    saved_x = node("vslice", "mem", start=121, stop=122)
    saved_y = node("vslice", "mem", start=122, stop=123)
    saved_valid = node("vslice", "mem", start=123, stop=124)
    saved_clicks = node("vslice", "mem", start=124, stop=125)
    saved_taboo_x = node("vslice", "mem", start=125, stop=126)
    saved_taboo_y = node("vslice", "mem", start=126, stop=127)
    saved_taboo_valid = node("vslice", "mem", start=127, stop=128)

    prev_up = node("vslice", "aprev", start=1, stop=2)
    prev_down = node("vslice", "aprev", start=2, stop=3)
    prev_left = node("vslice", "aprev", start=3, stop=4)
    prev_right = node("vslice", "aprev", start=4, stop=5)
    prev_click = node("vslice", "aprev", start=6, stop=7)

    use_saved = vmul(saved_valid, inverse(prev_click))
    moved_x = vadd(saved_x, affine(vsub(prev_right, prev_left), 3.0))
    moved_y = vadd(saved_y, affine(vsub(prev_down, prev_up), 3.0))
    anchor_x = select_vec(use_saved, moved_x, observed_x)
    anchor_y = select_vec(use_saved, moved_y, observed_y)
    coefficients = select_vec(use_saved, saved_coeff, observed_coeff)

    rel_x, rel_y, locations = shape_locations(anchor_x, anchor_y)
    selected = conv(
        node("mul", locations, coefficients),
        np.ones((1, size * size, 1, 1), dtype=np.float32),
    )
    selected = node("min2", selected, one)
    selected_present = indicator(node("gmax", selected))
    black_present = indicator(node("gmax", black))
    other = node("mul", black, node("sub", one, selected))
    other_present = indicator(node("gmax", other))

    click_count = vmul(
        prev_click,
        affine(clip01(affine(saved_clicks, 0.5, 0.5)), 2.0),
    )
    escape_trigger = vmul(
        indicator(click_count, 1.5, 2.0), selected_present
    )
    taboo_valid = clip01(
        vadd(
            escape_trigger,
            vmul(saved_taboo_valid, inverse(prev_click)),
        )
    )
    taboo_x = select_vec(escape_trigger, anchor_x, saved_taboo_x)
    taboo_y = select_vec(escape_trigger, anchor_y, saved_taboo_y)

    graph.append(
        {
            "id": "mem_next",
            "op": "vconcat",
            "in": [
                coefficients,
                anchor_x,
                anchor_y,
                selected_present,
                click_count,
                taboo_x,
                taboo_y,
                taboo_valid,
            ],
        }
    )

    # Reflection preserves values, so it works both for binary masks
    # and for goal-coverage counts.
    def reflect(mask, stripes, coordinate, axis, present, reduction):
        source_channels = node("mul", mask, stripes)
        source_lines = node(reduction, source_channels)
        positions = dense(
            axis,
            np.full((21, 1), 2.0, dtype=np.float32),
            -3.0 * np.arange(21, dtype=np.float32),
        )
        pixel_coordinate = spatial_affine(coordinate, 63.0)
        distance = node("abs", node("sub", pixel_coordinate, positions))
        w = np.zeros((21, 21, 1, 1), dtype=np.float32)
        for i in range(21):
            w[i, i, 0, 0] = -2.0
        destination = node(
            "relu",
            conv(distance, w, np.ones(21, dtype=np.float32)),
        )
        reflected = conv(node("mul", source_lines, destination), sum21)
        return node("mul", reflected, present)

    def mirror_closure(mask):
        rx = reflect(mask, xs, cx, axis_x, vertical_present, "rowmax")
        first = node("max2", mask, rx)
        ry = reflect(
            first, ys, cy, axis_y, horizontal_present, "colmax"
        )
        return node("max2", first, ry)

    selected_coverage = mirror_closure(selected)
    relevant_goal = node(
        "mul", goal, node("max2", uncovered, selected_coverage)
    )

    # Sum reflected goals. Divide by two on a mirror axis, where a
    # reflected cell coincides with its source.
    reflected_x = reflect(
        relevant_goal, xs, cx, axis_x, vertical_present, "rowmax"
    )
    x_counts = node("add", relevant_goal, reflected_x)
    x_axis_factor = spatial_affine(vertical_grid, -0.5, 1.0)
    x_counts = node("mul", x_counts, x_axis_factor)
    reflected_y = reflect(
        x_counts, ys, cy, axis_y, horizontal_present, "colmax"
    )
    projected_goal = node("add", x_counts, reflected_y)
    y_axis_factor = spatial_affine(horizontal_grid, -0.5, 1.0)
    projected_goal = node("mul", projected_goal, y_axis_factor)

    max_x = max_coordinate(selected, cx)
    max_y = max_coordinate(selected, cy)
    width = vsub(max_x, anchor_x)
    height = vsub(max_y, anchor_y)

    # Correlate the cached footprint with every candidate placement.
    goal_shifts = [projected_goal]
    for _ in range(size - 1):
        goal_shifts.append(shift(goal_shifts[-1], 3, 0))
    goal_channels = node("concat", *goal_shifts)

    score_state = fresh("state")
    goal_state = fresh("state")
    coeff_state = fresh("state")
    score_loop_id = fresh("loop")
    score_body = []
    old_current = current
    current = score_body

    row_coeff = node("vslice", coeff_state, start=0, stop=size)
    weighted = node("mul", goal_state, row_coeff)
    row_score = conv(
        weighted, np.ones((1, size, 1, 1), dtype=np.float32)
    )
    next_score = node("add", score_state, row_score)

    shift_rows = np.zeros((size, size, 7, 7), dtype=np.float32)
    for i in range(size):
        shift_rows[i, i, 6, 3] = 1.0
    next_goal = conv(goal_state, shift_rows)

    shift_coeff = np.zeros((size * size, size * size), dtype=np.float32)
    for i in range(size * size - size):
        shift_coeff[i, i + size] = 1.0
    next_coeff = dense(coeff_state, shift_coeff)

    current = old_current
    graph.append(
        {
            "id": score_loop_id,
            "op": "loop",
            "T": size,
            "state": [
                {"id": score_state, "zeros": [1, 64, 64]},
                {"id": goal_state, "init": goal_channels},
                {"id": coeff_state, "init": coefficients},
            ],
            "body": score_body,
            "next": {
                score_state: next_score,
                goal_state: next_goal,
                coeff_state: next_coeff,
            },
        }
    )
    placement_score = f"{score_loop_id}.{score_state}"

    pixel_x = spatial_affine(cx, 63.0)
    pixel_y = spatial_affine(cy, 63.0)
    dx = node("abs", node("sub", pixel_x, anchor_x))
    dy = node("abs", node("sub", pixel_y, anchor_y))
    distance = node("add", dx, dy)
    placement_score = node(
        "sub", placement_score, spatial_affine(distance, 0.003)
    )

    taboo_dx = node("abs", node("sub", pixel_x, taboo_x))
    taboo_dy = node("abs", node("sub", pixel_y, taboo_y))
    taboo_distance = node("add", taboo_dx, taboo_dy)
    taboo_penalty = node(
        "relu", spatial_affine(taboo_distance, -1.0 / 3.0, 4.0)
    )
    taboo_penalty = node("mul", taboo_penalty, taboo_valid)
    placement_score = node("sub", placement_score, taboo_penalty)

    x_limit = affine(width, -1.0, 60.5)
    y_limit = affine(height, -1.0, 60.5)
    x_legal = spatial_clip01(
        spatial_affine(node("sub", x_limit, pixel_x), 2.0)
    )
    y_legal = spatial_clip01(
        spatial_affine(node("sub", y_limit, pixel_y), 2.0)
    )
    legal = node("mul", grid, node("mul", x_legal, y_legal))
    invalid_penalty = spatial_affine(node("sub", one, legal), 100.0)
    legal_score = node("sub", placement_score, invalid_penalty)
    best_score = node("gmax", legal_score)

    anchor_x_mask = node("slice", rel_x, start=0, stop=1)
    anchor_y_mask = node("slice", rel_y, start=0, stop=1)
    anchor_mask = node("mul", anchor_x_mask, anchor_y_mask)
    current_score = affine(
        node("gmean", node("mul", placement_score, anchor_mask)),
        4096.0,
    )
    progress = indicator(vsub(best_score, current_score), 0.005, 200.0)

    difference_from_best = node("abs", node("sub", legal_score, best_score))
    best_cells = node(
        "relu", spatial_affine(difference_from_best, -500.0, 1.0)
    )
    best_cells = node("mul", best_cells, legal)

    def direction_mask(a, b):
        return spatial_clip01(
            spatial_affine(node("sub", a, b), 2.0, -1.0)
        )

    up_mask = direction_mask(anchor_y, pixel_y)
    down_mask = direction_mask(pixel_y, anchor_y)
    left_mask = direction_mask(anchor_x, pixel_x)
    right_mask = direction_mask(pixel_x, anchor_x)
    shape_directions = [
        indicator(node("gmax", node("mul", best_cells, m)), 0.5, 2.0)
        for m in (up_mask, down_mask, left_mask, right_mask)
    ]
    shape_move_active = vmul(
        shape_active, vmul(selected_present, progress)
    )
    shape_directions = [
        vmul(d, shape_move_active) for d in shape_directions
    ]

    first_shape_click = vmul(
        shape_active, vmul(inverse(selected_present), black_present)
    )
    next_shape_click = vmul(
        shape_active,
        vmul(selected_present, vmul(other_present, inverse(progress))),
    )
    shape_click = vadd(first_shape_click, next_shape_click)
    click_flag = vsum([h_click, v_click, shape_click])

    h_click_map = node(
        "mul",
        node("mul", node("mul", grid, p10), horizontal),
        node("sub", one, vertical),
    )
    v_click_map = node(
        "mul",
        node("mul", node("mul", grid, p10), vertical),
        node("sub", one, horizontal),
    )
    click_map = ssum(
        [
            node("mul", h_click_map, h_click),
            node("mul", v_click_map, v_click),
            node("mul", black, first_shape_click),
            node("mul", other, next_shape_click),
        ]
    )
    graph.append({"id": "click", "op": "relu", "in": [click_map]})

    features = node(
        "vconcat",
        h_move_up,
        h_move_down,
        v_move_left,
        v_move_right,
        click_flag,
        *shape_directions,
        "aprev",
    )
    weights = np.zeros((8, 17), dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    bias[[0, 5, 7]] = -10000.0
    for i in range(4):
        weights[1 + i, i] = 10000.0
        weights[1 + i, 5 + i] = 100.0
    weights[6, 4] = 10000.0
    weights[1, 9 + 2] = -0.01
    weights[2, 9 + 1] = -0.01
    weights[3, 9 + 4] = -0.01
    weights[4, 9 + 3] = -0.01

    graph.append(
        {
            "id": "logits",
            "op": "dense",
            "in": [features],
            "w": parameter(weights),
            "b": parameter(bias),
        }
    )

    return {"graph": graph, "memory_size": 128, "params": params}

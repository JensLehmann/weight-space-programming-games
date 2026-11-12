import numpy as np


def build():
    graph = []
    params = {}
    current = graph
    serial = 0
    pserial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(a):
        nonlocal pserial
        pserial += 1
        name = "p" + str(pserial)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        ident = fresh() if name is None else name
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        current.append(nd)
        return ident

    def linear(z, w, b=None, vector=False, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 0:
            w = w.reshape(1, 1)
        elif w.ndim == 1:
            w = w.reshape(1, -1)
        if not vector:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(
                np.broadcast_to(
                    np.asarray(b, dtype=np.float32),
                    (w.shape[0],)
                ).copy()
            )
        return node(
            "dense" if vector else "conv",
            [z], name=name, **fields
        )

    def sl(z, start, stop, vector=False):
        return node(
            "vslice" if vector else "slice",
            [z], start=start, stop=stop
        )

    def cat(*zs, vector=False):
        return node("vconcat" if vector else "concat", zs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def relu(a):
        return node("relu", [a])

    def vmax(a):
        return node("gmax", [a])

    def vmean(a):
        return node("gmean", [a])

    def vl(z, a, b=None):
        return linear(z, a, b, vector=True)

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vr(a):
        return node("vrelu", [a])

    def vn(a):
        return vl(a, -1.0)

    def vs(a, b):
        return va(a, vn(b))

    def vi(a):
        return vl(a, -1.0, 1.0)

    def vabs(a):
        return va(vr(a), vr(vn(a)))

    def vclip(a):
        return vs(vr(a), vr(vl(a, 1.0, -1.0)))

    def blend(flag, a, b):
        return va(vm(flag, a), vm(vi(flag), b))

    def loop_begin(states):
        nonlocal current
        parent = current
        body = []
        current = body
        return parent, body, states

    def loop_end(context, transitions, iterations=21):
        nonlocal current
        parent, body, states = context
        current = parent
        ident = fresh("loop")
        current.append({
            "id": ident,
            "op": "loop",
            "T": iterations,
            "state": states,
            "body": body,
            "next": transitions,
        })
        return {
            s["id"]: ident + "." + s["id"] for s in states
        }

    one = vl("avail", np.zeros((1, 8), np.float32), 1.0)
    zero = vl(one, 0.0)
    spatial_one = sl("coords", 2, 3)
    pixel_coordinates = linear(
        "coords", [[63, 0, 0], [0, 63, 0]]
    )
    row = sl(pixel_coordinates, 0, 1)
    col = sl(pixel_coordinates, 1, 2)

    def bounded_gate(delta, channels=1, slope=4.0, bias=2.0):
        absolute = node("abs", [delta])
        value = linear(
            absolute,
            -slope * np.eye(channels, dtype=np.float32),
            np.full(channels, bias, np.float32)
        )
        return node("min2", [relu(value), one])

    def inverse_spatial(z):
        return linear(z, -1.0, 1.0)

    def line_present(z, operation):
        total = node(operation, [z])
        return node(
            "min2", [relu(linear(total, 1.0, -3.0)), one]
        )

    # Separate solid tile color from the center-pixel UI annotation.
    body_kernel = np.zeros((3, 16, 3, 3), np.float32)
    body_kernel[0, 5, 0, 0] = 1
    body_kernel[1, 5, 0, 0] = 1
    body_kernel[1, 4, 0, 0] = 1
    body_kernel[2, 10, 0, 0] = 1
    body = node("conv", ["x0"], w=parameter(body_kernel))
    source_raw = sl(body, 0, 1)
    coverage_raw = sl(body, 1, 2)
    axis_raw = sl(body, 2, 3)

    center_zero = sl("x0", 0, 1)
    center_nine = sl("x0", 9, 10)
    target_raw = sl("x0", 11, 12)
    movable_marker = add(center_zero, center_nine)
    selected_raw = mul(source_raw, center_zero)
    movable_axis_raw = mul(axis_raw, movable_marker)
    selected_axis_raw = mul(axis_raw, center_zero)

    features_raw = cat(
        target_raw, source_raw, selected_raw, coverage_raw,
        axis_raw, movable_axis_raw, selected_axis_raw
    )

    # Gather the 3-pixel display lattice into a padded internal board.
    def compact_dimension(z, channels, coordinate, reduce_op):
        counter = fresh("s")
        accumulator = fresh("s")
        context = loop_begin([
            {"id": counter, "init": zero},
            {"id": accumulator, "zeros": [channels, 64, 64]},
        ])
        source_position = vl(counter, 3.0, 1.0)
        destination_position = vl(counter, 1.0, 20.0)
        source_gate = bounded_gate(
            sub(coordinate, source_position)
        )
        destination_gate = bounded_gate(
            sub(coordinate, destination_position)
        )
        extracted = node(reduce_op, [mul(z, source_gate)])
        placed = mul(extracted, destination_gate)
        updated = add(accumulator, placed)
        next_counter = vl(counter, 1.0, 1.0)
        outputs = loop_end(
            context,
            {counter: next_counter, accumulator: updated}
        )
        return outputs[accumulator]

    compact_rows = compact_dimension(
        features_raw, 7, row, "colsum"
    )
    features = compact_dimension(
        compact_rows, 7, col, "rowsum"
    )
    target = sl(features, 0, 1)
    source = sl(features, 1, 2)
    selected_seed = sl(features, 2, 3)
    coverage = sl(features, 3, 4)
    axes = sl(features, 4, 5)
    movable_axes = sl(features, 5, 6)
    selected_axes = sl(features, 6, 7)

    offsets = np.arange(20, 41, dtype=np.float32)
    row_masks = bounded_gate(
        linear(row, np.ones((21, 1)), -offsets),
        channels=21
    )
    col_masks = bounded_gate(
        linear(col, np.ones((21, 1)), -offsets),
        channels=21
    )
    row_domain = linear(row_masks, np.ones((1, 21)))
    col_domain = linear(col_masks, np.ones((1, 21)))
    board = mul(row_domain, col_domain)
    first_row = sl(row_masks, 0, 1)
    first_col = sl(col_masks, 0, 1)

    h_line = line_present(axes, "rowsum")
    v_line = line_present(axes, "colsum")
    h_exists = vmax(h_line)
    v_exists = vmax(v_line)
    h_position = vmax(mul(h_line, row))
    v_position = vmax(mul(v_line, col))
    h_movable = vmax(line_present(movable_axes, "rowsum"))
    v_movable = vmax(line_present(movable_axes, "colsum"))
    h_selected = vmax(line_present(selected_axes, "rowsum"))
    v_selected = vmax(line_present(selected_axes, "colsum"))

    # A selected axis can annotate an unselected source with zeroes.
    axis_selected = vclip(va(h_selected, v_selected))
    selected_seed = mul(selected_seed, vi(axis_selected))

    target_count = vl(vmean(target), 4096.0)

    def centroid_location(coordinate, domain):
        moment = vl(vmean(mul(target, coordinate)), 4096.0)
        discrepancy = node(
            "abs",
            [sub(mul(coordinate, target_count), moment)]
        )
        score = linear(discrepancy, -1.0)
        score = add(score, linear(coordinate, -0.001))
        score = add(
            score, linear(domain, 10000.0, -10000.0)
        )
        maximum = vmax(score)
        winner = relu(
            linear(sub(maximum, score), -100000.0, 1.0)
        )
        return vmax(mul(winner, coordinate))

    h_goal = centroid_location(row, row_domain)
    v_goal = centroid_location(col, col_domain)
    h_delta = vs(h_goal, h_position)
    v_delta = vs(v_goal, v_position)

    h_needed = vm(
        h_movable, vclip(vl(vabs(h_delta), 2.0, -0.5))
    )
    v_needed = vm(
        v_movable, vclip(vl(vabs(v_delta), 2.0, -0.5))
    )
    h_stage = h_needed
    v_stage = vm(v_needed, vi(h_needed))
    axis_stage = va(h_stage, v_stage)

    # Recover source pixels whose selection marker is hidden by a target.
    allowed_component = node(
        "max2", [selected_seed, mul(source, target)]
    )
    flood_kernel = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        flood_kernel[0, 0, 1 + dy, 1 + dx] = 1
    flood_weight = parameter(flood_kernel)
    component_state = fresh("s")
    context = loop_begin([
        {"id": component_state, "init": selected_seed}
    ])
    expanded = node(
        "conv", [component_state], w=flood_weight
    )
    expanded = node("min2", [expanded, one])
    expanded = mul(expanded, allowed_component)
    outputs = loop_end(
        context, {component_state: expanded}
    )
    selected = outputs[component_state]
    shape_present = vmax(selected)
    other_source = relu(sub(source, selected))

    current_y = vl(
        vmax(mul(linear(row, -1.0, 64.0), selected)),
        -1.0, 64.0
    )
    current_x = vl(
        vmax(mul(linear(col, -1.0, 64.0), selected)),
        -1.0, 64.0
    )

    shift_weights = {}

    def shift(z, direction, channels=1):
        key = (direction, channels)
        if key not in shift_weights:
            dy, dx = {
                "up": (1, 0),
                "down": (-1, 0),
                "left": (0, 1),
                "right": (0, -1),
            }[direction]
            w = np.zeros(
                (channels, channels, 3, 3), np.float32
            )
            for c in range(channels):
                w[c, c, 1 + dy, 1 + dx] = 1
            shift_weights[key] = parameter(w)
        return node(
            "conv", [z], w=shift_weights[key]
        )

    # Normalize the selected bounding-box origin to internal (20,20).
    norm_state = fresh("s")
    context = loop_begin([
        {"id": norm_state, "init": selected}
    ])
    can_up = vi(vmax(mul(norm_state, first_row)))
    shifted_up = shift(norm_state, "up")
    after_up = add(
        norm_state, mul(sub(shifted_up, norm_state), can_up)
    )
    can_left = vi(vmax(mul(after_up, first_col)))
    shifted_left = shift(after_up, "left")
    normalized_next = add(
        after_up, mul(sub(shifted_left, after_up), can_left)
    )
    outputs = loop_end(
        context, {norm_state: normalized_next}
    )
    normalized = outputs[norm_state]

    height = vmax(
        mul(linear(row, 1.0, -19.0), normalized)
    )
    width = vmax(
        mul(linear(col, 1.0, -19.0), normalized)
    )

    # Dynamic reflection using only fixed gather-and-broadcast operations.
    def reflect(z, channels, coordinate, axis_position, reduce_op):
        counter = fresh("s")
        accumulator = fresh("s")
        context = loop_begin([
            {"id": counter, "init": zero},
            {"id": accumulator, "zeros": [channels, 64, 64]},
        ])
        source_position = vl(counter, 1.0, 20.0)
        destination_position = linear(
            cat(axis_position, counter, vector=True),
            [[2.0, -1.0]], -20.0, vector=True
        )
        source_gate = bounded_gate(
            sub(coordinate, source_position)
        )
        destination_gate = bounded_gate(
            sub(coordinate, destination_position)
        )
        gathered = node(
            reduce_op, [mul(z, source_gate)]
        )
        placed = mul(gathered, destination_gate)
        accumulated = add(accumulator, placed)
        next_counter = vl(counter, 1.0, 1.0)
        outputs = loop_end(
            context,
            {counter: next_counter, accumulator: accumulated}
        )
        return outputs[accumulator]

    horizontal = reflect(
        cat(selected, normalized, other_source),
        3, row, h_position, "colsum"
    )
    h_selected_shape = sl(horizontal, 0, 1)
    h_template = sl(horizontal, 1, 2)
    h_other = sl(horizontal, 2, 3)

    vertical = reflect(
        cat(
            selected, normalized,
            h_selected_shape, h_template,
            other_source, h_other
        ),
        6, col, v_position, "rowsum"
    )
    v_selected_shape = sl(vertical, 0, 1)
    v_template = sl(vertical, 1, 2)
    both_selected_shape = sl(vertical, 2, 3)
    both_template = sl(vertical, 3, 4)
    v_other = sl(vertical, 4, 5)
    both_other = sl(vertical, 5, 6)

    # Infer reflection restrictions from the actual rendered coverage.
    empty_board = mul(
        board, inverse_spatial(coverage)
    )
    h_missing = vmax(mul(h_selected_shape, empty_board))
    v_missing = vmax(mul(v_selected_shape, empty_board))
    allow_h = vm(h_exists, vi(h_missing))
    allow_v = vm(v_exists, vi(v_missing))
    allow_both = vm(allow_h, allow_v)

    h_selected_shape = mul(h_selected_shape, allow_h)
    v_selected_shape = mul(v_selected_shape, allow_v)
    both_selected_shape = mul(
        both_selected_shape, allow_both
    )
    selected_union = add(
        add(selected, h_selected_shape),
        add(v_selected_shape, both_selected_shape)
    )
    selected_union = node(
        "min2", [selected_union, one]
    )

    # Explicitly reconstruct other-source coverage, including pixels
    # shared with the selected source's reflected copies.
    h_other = mul(h_other, h_exists)
    v_other = mul(v_other, v_exists)
    both_other = mul(
        both_other, vm(h_exists, v_exists)
    )
    reconstructed_other = add(
        add(other_source, h_other),
        add(v_other, both_other)
    )
    reconstructed_other = node(
        "min2", [reconstructed_other, one]
    )
    reconstructed_other = mul(
        reconstructed_other, coverage
    )
    exclusive_other = relu(
        sub(coverage, selected_union)
    )
    other_coverage = node(
        "max2", [exclusive_other, reconstructed_other]
    )

    # A modest isolated-target bonus breaks otherwise equal fits.
    neighbor_kernel = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_kernel[0, 0, 1 + dy, 1 + dx] = 1
    neighbors = node(
        "conv", [target], w=parameter(neighbor_kernel)
    )
    isolated = relu(linear(neighbors, -1.0, 1.0))
    target_weights = mul(
        target, linear(isolated, 0.125, 1.0)
    )
    relevant_targets = mul(
        target_weights, linear(other_coverage, -0.5, 1.0)
    )

    h_template = mul(h_template, allow_h)
    v_template = mul(v_template, allow_v)
    both_template = mul(both_template, allow_both)

    # Candidate rows are channels. Reflected copies translate in
    # opposite directions along their reflected dimension.
    rightward_candidates = []
    leftward_candidates = []
    original_y = normalized
    horizontal_y = h_template
    vertical_y = v_template
    both_y = both_template

    for y in range(21):
        rightward_candidates.append(
            node("min2", [add(original_y, horizontal_y), one])
        )
        leftward_candidates.append(
            node("min2", [add(vertical_y, both_y), one])
        )
        if y != 20:
            original_y = shift(original_y, "down")
            horizontal_y = shift(horizontal_y, "up")
            vertical_y = shift(vertical_y, "down")
            both_y = shift(both_y, "up")

    rightward_stack = cat(*rightward_candidates)
    leftward_stack = cat(*leftward_candidates)

    x_distance = node("abs", [sub(col, current_x)])
    y_distance = node("abs", [sub(row, current_y)])
    travel = add(
        linear(x_distance, -0.013),
        linear(y_distance, -0.017)
    )
    tie_cost = add(
        linear(col, -0.00011, 20.0 * 0.00011),
        linear(row, -0.00231, 20.0 * 0.00231)
    )
    x_overflow = relu(
        add(linear(col, 1.0, -41.1), width)
    )
    y_overflow = relu(
        add(linear(row, 1.0, -41.1), height)
    )
    bounds_cost = linear(
        add(x_overflow, y_overflow), -1000.0
    )
    domain_cost = linear(
        board, 10000.0, -10000.0
    )
    placement_cost = add(
        add(travel, tie_cost),
        add(bounds_cost, domain_cost)
    )

    # Evaluate every translation with a 21-column sweep.
    initial_best = linear(
        spatial_one, 0.0, -1000000.0
    )
    sweep_x = fresh("s")
    sweep_right = fresh("s")
    sweep_left = fresh("s")
    sweep_best = fresh("s")
    context = loop_begin([
        {"id": sweep_x, "init": zero},
        {"id": sweep_right, "init": rightward_stack},
        {"id": sweep_left, "init": leftward_stack},
        {"id": sweep_best, "init": initial_best},
    ])

    union = node(
        "min2", [add(sweep_right, sweep_left), one]
    )
    hits = mul(union, relevant_targets)
    counts = linear(
        vmean(hits),
        4096.0 * np.eye(21, dtype=np.float32),
        vector=True
    )
    counts_by_row = mul(row_masks, counts)
    score_field = linear(
        counts_by_row, np.ones((1, 21))
    )
    score_field = add(score_field, placement_cost)
    sweep_position = vl(sweep_x, 1.0, 20.0)
    sweep_column = bounded_gate(
        sub(col, sweep_position)
    )
    score_field = add(
        score_field,
        linear(sweep_column, 1000.0, -1000.0)
    )
    best_next = node(
        "max2", [sweep_best, score_field]
    )
    right_next = shift(
        sweep_right, "right", channels=21
    )
    left_next = shift(
        sweep_left, "left", channels=21
    )
    x_next = vl(sweep_x, 1.0, 1.0)
    outputs = loop_end(context, {
        sweep_x: x_next,
        sweep_right: right_next,
        sweep_left: left_next,
        sweep_best: best_next,
    })
    score_map = outputs[sweep_best]

    best_score = vmax(score_map)
    best_cells = relu(
        linear(sub(best_score, score_map), -1000000.0, 1.0)
    )
    destination_y = vmax(mul(best_cells, row))
    destination_x = vmax(mul(best_cells, col))

    # Memory stores destination, tracked origin, and active-plan flag.
    old_goal_y = sl("mem", 0, 1, vector=True)
    old_goal_x = sl("mem", 1, 2, vector=True)
    old_y = sl("mem", 2, 3, vector=True)
    old_x = sl("mem", 3, 4, vector=True)
    old_valid = sl("mem", 4, 5, vector=True)

    previous_movement = vl(
        "aprev", [[0, 1, 1, 1, 1, 0, 0, 0]]
    )
    previous_dy = vl(
        "aprev", [[0, -1, 1, 0, 0, 0, 0, 0]]
    )
    previous_dx = vl(
        "aprev", [[0, 0, 0, -1, 1, 0, 0, 0]]
    )
    continuing = vm(old_valid, previous_movement)
    tracked_y = va(old_y, previous_dy)
    tracked_x = va(old_x, previous_dx)

    navigation_y = blend(
        continuing, tracked_y, current_y
    )
    navigation_x = blend(
        continuing, tracked_x, current_x
    )
    navigation_goal_y = blend(
        continuing, old_goal_y, destination_y
    )
    navigation_goal_x = blend(
        continuing, old_goal_x, destination_x
    )

    shape_dy = vs(navigation_goal_y, navigation_y)
    shape_dx = vs(navigation_goal_x, navigation_x)
    shape_active = vm(
        vclip(va(shape_present, continuing)),
        vi(axis_stage)
    )
    h_move_active = vm(h_stage, h_selected)
    v_move_active = vm(v_stage, v_selected)

    up = va(
        vm(shape_active, vr(vn(shape_dy))),
        vm(h_move_active, vr(vn(h_delta)))
    )
    down = va(
        vm(shape_active, vr(shape_dy)),
        vm(h_move_active, vr(h_delta))
    )
    left = va(
        vm(shape_active, vr(vn(shape_dx))),
        vm(v_move_active, vr(vn(v_delta)))
    )
    right = va(
        vm(shape_active, vr(shape_dx)),
        vm(v_move_active, vr(v_delta))
    )
    need_click = va(
        vm(h_stage, vi(h_selected)),
        vm(v_stage, vi(v_selected))
    )

    node(
        "vconcat",
        [
            navigation_goal_y, navigation_goal_x,
            navigation_y, navigation_x, shape_active
        ],
        name="mem_next"
    )

    # Click an exposed axis tile away from its intersection.
    h_display = vl(h_position, 3.0, -59.0)
    v_display = vl(v_position, 3.0, -59.0)
    h_band = bounded_gate(
        sub(row, h_display), slope=1.0, bias=2.0
    )
    v_band = bounded_gate(
        sub(col, v_display), slope=1.0, bias=2.0
    )
    visible_axis_color = sl("x0", 10, 11)
    h_click = mul(
        mul(h_band, inverse_spatial(v_band)),
        visible_axis_color
    )
    v_click = mul(
        mul(v_band, inverse_spatial(h_band)),
        visible_axis_color
    )
    click_score = add(
        mul(h_click, h_stage),
        mul(v_click, v_stage)
    )
    linear(click_score, 1.0, name="click")

    decision = cat(
        up, down, left, right, need_click,
        vector=True
    )
    output_weights = np.zeros((8, 5), np.float32)
    for i in range(4):
        output_weights[i + 1, i] = 1.0
    output_weights[6, 4] = 100.0
    output_bias = np.array(
        [-100.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, -100.0],
        dtype=np.float32
    )
    linear(
        decision, output_weights, output_bias,
        vector=True, name="logits"
    )

    return {
        "graph": graph,
        "memory_size": 5,
        "params": params,
    }

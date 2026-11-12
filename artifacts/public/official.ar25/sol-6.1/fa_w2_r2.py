import numpy as np


def build():
    graph, params = [], {}
    sizes = {"x0": 16, "coords": 3, "mem": 5, "aprev": 8, "avail": 8}
    serial = [0]
    pserial = [0]
    constants = {}

    def fresh(prefix="n"):
        k = serial[0]
        serial[0] += 1
        return prefix + str(k)

    def parameter(a):
        name = "p" + str(pserial[0])
        pserial[0] += 1
        params[name] = np.asarray(a, np.float32)
        return name

    def node(op, inputs, size, dest=None, name=None, **fields):
        ident = fresh() if name is None else name
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        (graph if dest is None else dest).append(nd)
        sizes[ident] = size
        return ident

    def conv(x, w, b=None, dest=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], w.shape[0], dest, name, **fields)

    def shared_conv(x, pname, channels, dest=None):
        return node("conv", [x], channels, dest, w=pname)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], w.shape[0], name=name, **fields)

    def const(value):
        value = float(value)
        if value not in constants:
            constants[value] = dense(
                "avail", np.zeros((1, 8), np.float32), [value]
            )
        return constants[value]

    def unary(op, x, dest=None):
        return node(op, [x], sizes[x], dest)

    def binary(op, x, y, dest=None, name=None):
        return node(op, [x, y], max(sizes[x], sizes[y]), dest, name)

    def cat(xs, vector=False, dest=None):
        return node(
            "vconcat" if vector else "concat",
            xs, sum(sizes[x] for x in xs), dest
        )

    def sl(x, a, b, vector=False, dest=None):
        return node(
            "vslice" if vector else "slice",
            [x], b - a, dest, start=a, stop=b
        )

    def pool(op, x):
        return node(op, [x], sizes[x])

    def sa(x, a=1.0, b=0.0, dest=None):
        if a != 1.0:
            x = binary("mul", x, const(a), dest)
        if b != 0.0:
            x = binary("add", x, const(b), dest)
        return x

    def va(x, a=1.0, b=0.0):
        if a != 1.0:
            x = binary("vmul", x, const(a))
        if b != 0.0:
            x = binary("vadd", x, const(b))
        return x

    def sclip(x, dest=None):
        return binary("min2", unary("relu", x, dest), const(1.0), dest)

    def vclip(x):
        return binary(
            "vadd", unary("vrelu", x),
            va(unary("vrelu", va(x, b=-1.0)), -1.0)
        )

    def vabs(x):
        return binary(
            "vadd", unary("vrelu", x),
            unary("vrelu", va(x, -1.0))
        )

    def vstep(x, threshold=0.25, slope=4.0):
        return vclip(va(x, slope, -slope * threshold))

    def seq(x, y):
        return sclip(sa(unary("abs", binary("sub", x, y)), -8.0, 2.0))

    def vmul(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = binary("vmul", out, x)
        return out

    def smul(*xs):
        out = xs[0]
        for x in xs[1:]:
            out = binary("mul", out, x)
        return out

    def blend(a, b, gate):
        return binary(
            "vadd", vmul(a, va(gate, -1.0, 1.0)), vmul(b, gate)
        )

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    gx = sa(col, 21.0, -1.0 / 3.0)
    gy = sa(row, 21.0, -1.0 / 3.0)

    def phase_plane(axis):
        w = np.zeros((21, 3, 1, 1), np.float32)
        w[:, axis, 0, 0] = 63.0
        b = -(3.0 * np.arange(21, dtype=np.float32) + 1.0)
        d = unary("abs", conv("coords", w, b))
        hats = unary("relu", sa(d, -2.0, 1.0))
        total = conv(hats, np.ones((1, 21, 1, 1), np.float32))
        return sclip(sa(total, 2.0))

    phase = smul(phase_plane(0), phase_plane(1))

    # Sample corners to distinguish physical objects from UI marks.
    w = np.zeros((3, 16, 3, 3), np.float32)
    for out, colour in enumerate((5, 10, 4)):
        w[out, colour, 0, 0] = 1.0
    corners = conv("x0", w)
    originals = smul(sl(corners, 0, 1), phase)
    mirror_pixels = smul(sl(corners, 1, 2), phase)
    reflected_pixels = smul(sl(corners, 2, 3), phase)
    occupied = sclip(binary("add", originals, reflected_pixels))
    selected_centers = smul(sl("x0", 0, 1), phase)
    targets = smul(sl("x0", 11, 12), phase)

    eligible = smul(
        originals, sclip(binary("add", selected_centers, targets))
    )
    seeds = smul(originals, selected_centers)

    flood_w = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in ((0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)):
        flood_w[0, 0, 3 + dy, 3 + dx] = 1.0
    flood_p = parameter(flood_w)
    flood_state = fresh("s")
    sizes[flood_state] = 1
    body = []
    expanded = shared_conv(flood_state, flood_p, 1, body)
    flooded = binary("min2", expanded, eligible, body)
    flood_id = fresh("L")
    graph.append({
        "id": flood_id, "op": "loop", "T": 32,
        "state": [{"id": flood_state, "init": seeds}],
        "body": body, "next": {flood_state: flooded},
    })
    selected_raw = flood_id + "." + flood_state
    sizes[selected_raw] = 1

    coordinate_extrema = cat([sa(gx, -1.0), sa(gy, -1.0), gx, gy])

    def extrema(mask):
        z = pool(
            "gmax",
            binary("add", coordinate_extrema, sa(mask, 1000.0, -1000.0))
        )
        return (
            va(sl(z, 0, 1, True), -1.0),
            va(sl(z, 1, 2, True), -1.0),
            sl(z, 2, 3, True),
            sl(z, 3, 4, True),
        )

    tx0, ty0, tx1, ty1 = extrema(targets)
    target_axis_x = va(binary("vadd", tx0, tx1), 0.5)
    target_axis_y = va(binary("vadd", ty0, ty1), 0.5)

    # Three exposed cells suffice to identify a mirror. An orthogonal
    # mirror contributes only one cell. Saturation keeps the resulting
    # masks binary even when shapes hide most of the line.
    vertical_run = node("colsum", [mirror_pixels], 1)
    horizontal_run = node("rowsum", [mirror_pixels], 1)
    vertical_band = sclip(sa(vertical_run, 4.0, -10.0))
    horizontal_band = sclip(sa(horizontal_run, 4.0, -10.0))
    vertical_mask = smul(vertical_band, phase)
    horizontal_mask = smul(horizontal_band, phase)
    has_vertical = pool("gmax", vertical_mask)
    has_horizontal = pool("gmax", horizontal_mask)

    mirror_x = pool(
        "gmax", binary("add", gx, sa(vertical_mask, 1000.0, -1000.0))
    )
    mirror_y = pool(
        "gmax", binary("add", gy, sa(horizontal_mask, 1000.0, -1000.0))
    )

    marker = sclip(binary("add", sl("x0", 0, 1), sl("x0", 9, 10)))
    marker = smul(marker, mirror_pixels)
    movable_v_cells = smul(marker, vertical_mask)
    movable_h_cells = smul(marker, horizontal_mask)
    movable_v = pool("gmax", movable_v_cells)
    movable_h = pool("gmax", movable_h_cells)

    def selected_line(cells):
        count = va(pool("gmean", smul(cells, selected_centers)), 4096.0)
        return vstep(count, threshold=1.5, slope=2.0)

    selected_v = selected_line(movable_v_cells)
    selected_h = selected_line(movable_h_cells)
    selected_mirror = vclip(binary("vadd", selected_v, selected_h))
    selected = smul(selected_raw, va(selected_mirror, -1.0, 1.0))
    selected_found = pool("gmax", selected)
    other_originals = unary("relu", binary("sub", originals, selected))
    sx, sy, ex, ey = extrema(selected)

    axis_dx = binary("vadd", target_axis_x, va(mirror_x, -1.0))
    axis_dy = binary("vadd", target_axis_y, va(mirror_y, -1.0))
    need_v = vmul(
        movable_v, vstep(vabs(axis_dx), threshold=0.55, slope=10.0)
    )
    need_h = vmul(
        movable_h, vstep(vabs(axis_dy), threshold=0.55, slope=10.0)
    )
    axis_needed = vclip(binary("vadd", need_v, need_h))

    def reflect(source, axis_position, horizontal):
        channels = sizes[source]
        index_state, output_state = fresh("i"), fresh("s")
        sizes[index_state] = 1
        sizes[output_state] = channels
        body = []
        coordinate = gy if horizontal else gx
        d1 = unary(
            "abs", binary("sub", coordinate, index_state, body), body
        )
        source_mask = sclip(sa(d1, -8.0, 2.0, body), body)
        scanned = binary("mul", source, source_mask, body)
        projected = node(
            "colmax" if horizontal else "rowmax",
            [scanned], channels, body
        )
        negative_index = binary("vmul", index_state, const(-1.0), body)
        doubled_axis = binary("vmul", axis_position, const(2.0), body)
        destination = binary("vadd", doubled_axis, negative_index, body)
        d2 = unary(
            "abs", binary("sub", coordinate, destination, body), body
        )
        destination_mask = sclip(sa(d2, -8.0, 2.0, body), body)
        painted = binary("mul", projected, destination_mask, body)
        accumulated = binary("max2", output_state, painted, body)
        next_index = binary("vadd", index_state, const(1.0), body)
        loop_id = fresh("L")
        graph.append({
            "id": loop_id, "op": "loop", "T": 21,
            "state": [
                {"id": index_state, "zeros": [1]},
                {"id": output_state, "zeros": [channels, 64, 64]},
            ],
            "body": body,
            "next": {
                index_state: next_index,
                output_state: accumulated,
            },
        })
        result = loop_id + "." + output_state
        sizes[result] = channels
        return result

    def reflection_allowed(prediction, axis_exists):
        missing = smul(prediction, sa(occupied, -1.0, 1.0))
        count = va(pool("gmean", missing), 4096.0)
        contradicted = vstep(count, threshold=0.25, slope=4.0)
        return vmul(axis_exists, va(contradicted, -1.0, 1.0))

    objects = cat([selected, other_originals])
    rx = reflect(objects, mirror_x, False)
    allow_x = reflection_allowed(sl(rx, 0, 1), has_vertical)
    x_gates = cat([allow_x, has_vertical], vector=True)
    objects_x = binary("max2", objects, smul(rx, x_gates))
    ry = reflect(objects_x, mirror_y, True)
    allow_y = reflection_allowed(sl(ry, 0, 1), has_horizontal)
    y_gates = cat([allow_y, has_horizontal], vector=True)
    object_orbits = binary("max2", objects_x, smul(ry, y_gates))

    other_coverage = smul(sl(object_orbits, 1, 2), occupied)
    needed_targets = smul(targets, sa(other_coverage, -1.0, 1.0))
    target_fields = cat([targets, needed_targets])
    off_axis_x = sa(seq(gx, mirror_x), -1.0, 1.0)
    off_axis_y = sa(seq(gy, mirror_y), -1.0, 1.0)
    target_rx = reflect(target_fields, mirror_x, False)
    target_fields_x = binary(
        "add", target_fields, smul(target_rx, allow_x, off_axis_x)
    )
    target_ry = reflect(target_fields_x, mirror_y, True)
    virtual_targets = binary(
        "add", target_fields_x, smul(target_ry, allow_y, off_axis_y)
    )
    weighted_targets = conv(
        virtual_targets,
        np.array([1.0, 100.0], np.float32).reshape(1, 2, 1, 1)
    )

    # Read an object-relative template into a vector.
    local_x = binary("sub", gx, sx)
    local_y = binary("sub", gy, sy)
    offsets = np.arange(11, dtype=np.float32)
    dx = unary(
        "abs", conv(local_x, np.ones((11, 1, 1, 1), np.float32), -offsets)
    )
    dy = unary(
        "abs", conv(local_y, np.ones((11, 1, 1, 1), np.float32), -offsets)
    )
    offset_distances = cat([dx, dy])
    template_w = np.zeros((121, 22, 1, 1), np.float32)
    for r in range(11):
        for c in range(11):
            k = 11 * r + c
            template_w[k, c, 0, 0] = -4.0
            template_w[k, 11 + r, 0, 0] = -4.0
    template_locations = unary(
        "relu", conv(offset_distances, template_w, np.ones(121))
    )
    template_locations = sclip(sa(template_locations, 2.0))
    template = pool("gmax", smul(template_locations, selected))

    shift_x_w = np.zeros((1, 1, 7, 7), np.float32)
    shift_x_w[0, 0, 3, 6] = 1.0
    shift_x_p = parameter(shift_x_w)
    horizontal_shifts = [weighted_targets]
    for _ in range(10):
        horizontal_shifts.append(
            shared_conv(horizontal_shifts[-1], shift_x_p, 1)
        )
    horizontal_bank = cat(horizontal_shifts)

    shift_y_w = np.zeros((11, 11, 7, 7), np.float32)
    for c in range(11):
        shift_y_w[c, c, 6, 3] = 1.0
    shift_y_p = parameter(shift_y_w)
    vertical_shifts = [horizontal_bank]
    for _ in range(10):
        vertical_shifts.append(
            shared_conv(vertical_shifts[-1], shift_y_p, 11)
        )
    bank = cat(vertical_shifts)
    products = smul(bank, template)
    correlation = conv(products, np.ones((1, 121, 1, 1), np.float32))

    width_minus_one = binary("vadd", ex, va(sx, -1.0))
    height_minus_one = binary("vadd", ey, va(sy, -1.0))
    limit_x = va(width_minus_one, -1.0, 20.0)
    limit_y = va(height_minus_one, -1.0, 20.0)
    bad_x = sclip(sa(binary("sub", gx, limit_x), 4.0, -1.0))
    bad_y = sclip(sa(binary("sub", gy, limit_y), 4.0, -1.0))
    valid = smul(
        phase, sa(bad_x, -1.0, 1.0), sa(bad_y, -1.0, 1.0)
    )

    x_end = binary("add", gx, width_minus_one)
    y_end = binary("add", gy, height_minus_one)
    cross_x = smul(
        sclip(sa(binary("sub", mirror_x, gx), 4.0, -1.0)),
        sclip(sa(binary("sub", x_end, mirror_x), 4.0, -1.0)),
        allow_x
    )
    cross_y = smul(
        sclip(sa(binary("sub", mirror_y, gy), 4.0, -1.0)),
        sclip(sa(binary("sub", y_end, mirror_y), 4.0, -1.0)),
        allow_y
    )
    correlation = smul(
        correlation, sa(cross_x, -0.5, 1.0), sa(cross_y, -0.5, 1.0)
    )
    scores = binary("add", correlation, sa(valid, 10000.0, -10000.0))

    best_score = pool("gmax", scores)
    best = smul(
        valid, sclip(sa(binary("sub", scores, best_score), 5.0, 1.0))
    )
    distance = binary(
        "add",
        unary("abs", binary("sub", gx, sx)),
        unary("abs", binary("sub", gy, sy))
    )
    negative_best_distance = pool(
        "gmax",
        binary("add", sa(distance, -1.0), sa(best, 1000.0, -1000.0))
    )
    best_distance = va(negative_best_distance, -1.0)
    nearest = smul(best, seq(distance, best_distance))
    negative_goal_y = pool(
        "gmax",
        binary("add", sa(gy, -1.0), sa(nearest, 1000.0, -1000.0))
    )
    goal_y = va(negative_goal_y, -1.0)
    chosen_row = smul(nearest, seq(gy, goal_y))
    negative_goal_x = pool(
        "gmax",
        binary("add", sa(gx, -1.0), sa(chosen_row, 1000.0, -1000.0))
    )
    goal_x = va(negative_goal_x, -1.0)

    # Memory: active flag, destination x/y, predicted current x/y.
    previous_direction = dense(
        "aprev", np.array([[0, 1, 1, 1, 1, 0, 0, 0]], np.float32)
    )
    continue_motion = vmul(sl("mem", 0, 1, True), previous_direction)
    used_goal_x = blend(goal_x, sl("mem", 1, 2, True), continue_motion)
    used_goal_y = blend(goal_y, sl("mem", 2, 3, True), continue_motion)
    used_x = blend(sx, sl("mem", 3, 4, True), continue_motion)
    used_y = blend(sy, sl("mem", 4, 5, True), continue_motion)

    observed_object = vmul(
        selected_found, va(selected_mirror, -1.0, 1.0)
    )
    object_enabled = vmul(
        blend(observed_object, const(1.0), continue_motion),
        va(axis_needed, -1.0, 1.0)
    )
    object_dx = binary("vadd", used_goal_x, va(used_x, -1.0))
    object_dy = binary("vadd", used_goal_y, va(used_y, -1.0))
    horizontal_move = vstep(vabs(object_dx))
    vertical_enabled = vmul(
        object_enabled, va(horizontal_move, -1.0, 1.0)
    )

    object_up = vmul(vertical_enabled, vstep(va(object_dy, -1.0)))
    object_down = vmul(vertical_enabled, vstep(object_dy))
    object_left = vmul(object_enabled, vstep(va(object_dx, -1.0)))
    object_right = vmul(object_enabled, vstep(object_dx))

    line_up = vmul(selected_h, need_h, vstep(va(axis_dy, -1.0)))
    line_down = vmul(selected_h, need_h, vstep(axis_dy))
    line_left = vmul(selected_v, need_v, vstep(va(axis_dx, -1.0)))
    line_right = vmul(selected_v, need_v, vstep(axis_dx))

    up = binary("vadd", object_up, line_up)
    down = binary("vadd", object_down, line_down)
    left = binary("vadd", object_left, line_left)
    right = binary("vadd", object_right, line_right)

    selected_wrong_line = vclip(binary(
        "vadd", vmul(selected_v, need_v), vmul(selected_h, need_h)
    ))
    should_click = vmul(
        axis_needed, va(selected_wrong_line, -1.0, 1.0)
    )

    controls = cat([up, down, left, right, should_click], vector=True)
    output_w = np.zeros((8, 5), np.float32)
    for action, channel in ((1, 0), (2, 1), (3, 2), (4, 3)):
        output_w[action, channel] = 20.0
    output_w[6, 4] = 30.0
    output_b = np.array(
        [-100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -100.0], np.float32
    )
    dense(controls, output_w, output_b, name="logits")

    moving = vclip(binary(
        "vadd",
        binary("vadd", object_up, object_down),
        binary("vadd", object_left, object_right)
    ))
    next_x = binary(
        "vadd", used_x,
        binary("vadd", object_right, va(object_left, -1.0))
    )
    next_y = binary(
        "vadd", used_y,
        binary("vadd", object_down, va(object_up, -1.0))
    )
    memory_values = cat(
        [moving, used_goal_x, used_goal_y, next_x, next_y], vector=True
    )
    dense(memory_values, np.eye(5, dtype=np.float32), name="mem_next")

    click_v = smul(movable_v_cells, need_v)
    click_h = smul(movable_h_cells, need_h)
    click_scores = binary("add", sa(click_v, 2.0), click_h)
    conv(click_scores, np.ones((1, 1, 1, 1), np.float32), name="click")

    return {"graph": graph, "memory_size": 5, "params": params}

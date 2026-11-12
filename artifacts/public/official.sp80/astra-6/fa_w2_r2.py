import numpy as np


def build():
    N, K, S = 6, 6, 4
    M = 5 * N + 2
    params, cache = {}, {}
    graph = []
    current = graph
    counter = [0]
    shapes = {
        "x0": ("s", 16), "coords": ("s", 3),
        "mem": ("v", M), "aprev": ("v", 8), "avail": ("v", 8),
    }

    def fresh(prefix="n"):
        name = prefix + str(counter[0])
        counter[0] += 1
        return name

    def parameter(value):
        value = np.asarray(value, np.float32)
        key = (value.shape, value.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            params[name] = value
            cache[key] = name
        return cache[key]

    def emit(op, ins=(), name=None, **fields):
        name = fresh() if name is None else name
        node = {"id": name, "op": op, "in": list(ins)}
        node.update(fields)
        current.append(node)
        if op in ("conv", "dense"):
            shape = (
                "s" if op == "conv" else "v",
                int(params[fields["w"]].shape[0]),
            )
        elif op in ("gmax", "gmean"):
            shape = ("v", shapes[ins[0]][1])
        elif op in ("concat", "vconcat"):
            shape = (shapes[ins[0]][0], sum(shapes[z][1] for z in ins))
        elif op in ("slice", "vslice"):
            shape = (shapes[ins[0]][0], fields["stop"] - fields["start"])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = ("s", max(shapes[z][1] for z in ins))
        elif op in ("vadd", "vmul"):
            shape = ("v", max(shapes[z][1] for z in ins))
        else:
            shape = shapes[ins[0]]
        shapes[name] = shape
        return name

    def linear(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        spatial = shapes[x][0] == "s"
        if spatial:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv" if spatial else "dense", [x],
                    name=name, **fields)

    def affine(x, a=1.0, b=0.0):
        c = shapes[x][1]
        return linear(
            x, np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32) if b != 0 else None,
        )

    def const(value, n=1):
        return linear("avail", np.zeros((n, 8), np.float32),
                      np.full(n, value, np.float32))

    def sl(x, start, stop):
        return emit("slice" if shapes[x][0] == "s" else "vslice",
                    [x], start=start, stop=stop)

    def cat(*xs, name=None):
        return emit("concat" if shapes[xs[0]][0] == "s" else "vconcat",
                    xs, name=name)

    def add(a, b):
        op = "vadd" if shapes[a][0] == shapes[b][0] == "v" else "add"
        return emit(op, [a, b])

    def sub(a, b):
        if shapes[a][0] == shapes[b][0] == "v":
            return add(a, affine(b, -1))
        return emit("sub", [a, b])

    def mul(a, b):
        op = "vmul" if shapes[a][0] == shapes[b][0] == "v" else "mul"
        return emit(op, [a, b])

    def relu(x):
        return emit("relu" if shapes[x][0] == "s" else "vrelu", [x])

    def absolute(x):
        if shapes[x][0] == "s":
            return emit("abs", [x])
        return add(relu(x), relu(affine(x, -1)))

    def minimum(a, b):
        return emit("min2", [a, b])

    def maximum(a, b):
        return emit("max2", [a, b])

    def gmax(x):
        return emit("gmax", [x])

    def rowmax(x):
        return emit("rowmax", [x])

    def rowsum(x):
        return emit("rowsum", [x])

    def sum_channels(x):
        return linear(x, np.ones((1, shapes[x][1]), np.float32))

    one = sl("coords", 2, 3)
    zero = affine(one, 0)

    def step(x, threshold=0.0, gain=64.0):
        return minimum(relu(affine(x, gain, -threshold * gain)), one)

    def blend(a, b, gate):
        return add(mul(a, affine(gate, -1, 1)), mul(b, gate))

    def quantized(z, thresholds, coefficients, bias):
        gates = [step(z, t, 1000000) for t in thresholds]
        return linear(cat(*gates), [coefficients], [bias])

    def logistic(z):
        return affine(mul(z, affine(z, -1, 1)), 4)

    def shift_right(x):
        c = shapes[x][1]
        w = np.zeros((c, c, 3, 3), np.float32)
        for j in range(c):
            w[j, j, 1, 0] = 1
        return emit("conv", [x], w=parameter(w))

    def begin_loop(initial, zeros=None):
        nonlocal current
        parent, body = current, []
        current = body
        states, ids = [], []
        for value in initial:
            sid = fresh("s")
            states.append({"id": sid, "init": value})
            shapes[sid] = shapes[value]
            ids.append(sid)
        for dims in zeros or []:
            sid = fresh("s")
            states.append({"id": sid, "zeros": list(dims)})
            shapes[sid] = (
                ("s", dims[0]) if len(dims) == 3 else ("v", dims[0])
            )
            ids.append(sid)
        return parent, body, states, ids

    def end_loop(context, outputs, iterations):
        nonlocal current
        parent, body, states, ids = context
        current = parent
        lid = fresh("L")
        node = {
            "id": lid, "op": "loop", "T": int(iterations),
            "state": states, "body": body,
            "next": dict(zip(ids, outputs)),
        }
        current.append(node)
        result = []
        for sid in ids:
            name = lid + "." + sid
            shapes[name] = shapes[sid]
            result.append(name)
        return result, node

    # Recover canonical coordinates from the rotated counter edge.
    raw_col = affine(sl("coords", 1, 2), 63)
    raw_row = affine(sl("coords", 0, 1), 63)
    ui = add(sl("x0", 14, 15), sl("x0", 0, 1))
    edge_masks = [
        step(affine(raw_row, -1), -0.25),
        step(affine(raw_col, -1), -0.25),
        step(raw_row, 62.75),
        step(raw_col, 62.75),
    ]
    edge_mean = emit(
        "gmean", [cat(*[mul(ui, mask) for mask in edge_masks])]
    )
    flags = relu(linear(
        edge_mean, 256 * np.eye(4, dtype=np.float32),
        np.full(4, -3, np.float32),
    ))
    flags = sub(flags, relu(affine(flags, 1, -1)))

    w = np.zeros((8, 3), np.float32)
    b = np.zeros(8, np.float32)
    w[0, 1], w[1, 0] = 63, 63
    w[2, 0], b[2], w[3, 1] = -63, 63, 63
    w[4, 1], b[4], w[5, 0], b[5] = -63, 63, -63, 63
    w[6, 0], w[7, 1], b[7] = 63, -63, 63
    uv4 = linear("coords", w, b)

    repeat_flags = np.zeros((8, 4), np.float32)
    for j in range(4):
        repeat_flags[2*j:2*j+2, j] = 1
    f8 = linear(flags, repeat_flags)
    combine = np.zeros((2, 8), np.float32)
    combine[0, ::2] = 1
    combine[1, 1::2] = 1
    uv = linear(mul(uv4, f8), combine)
    ext = linear(
        uv, [[1, 0], [0, 1], [-1, 0], [0, -1]],
        [0, 0, 63, 63],
    )

    selected_plane = sl("x0", 9, 10)
    obstacle_colors = cat(
        sl("x0", 8, 9), selected_plane, sl("x0", 15, 16)
    )
    cup_colors = cat(sl("x0", 11, 12), sl("x0", 13, 14))
    source_colors = sl("x0", 6, 7)
    movable_mask = sum_channels(obstacle_colors)
    priority = linear(
        cat(uv, selected_plane), [[-1, -64, 8192]], [8192]
    )

    box_w = np.array([
        [.5, 0, -.5, 0],
        [0, .5, 0, -.5],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], np.float32)
    box_b = np.array([31.5, 31.5, -62, -62], np.float32)
    flood_w = parameter(np.ones((1, 1, 7, 7), np.float32))

    def extract(colors, count, flood_rounds, find_hole):
        remaining = sum_channels(colors)
        context = begin_loop([remaining], [[8 * count]])
        rem, history = context[3]

        score = mul(priority, rem)
        peak = gmax(score)
        seed = minimum(step(sub(score, peak), -0.25, 4), rem)
        seed_color = gmax(mul(colors, seed))
        allowed = minimum(sum_channels(mul(colors, seed_color)), rem)

        component = seed
        for _ in range(flood_rounds):
            component = minimum(
                emit("conv", [component], w=flood_w), allowed
            )

        valid = gmax(component)
        extrema = gmax(mul(ext, component))
        box = mul(linear(extrema, box_w, box_b), valid)

        if find_hole:
            centre = sl(box, 0, 2)
            half_size = affine(sl(box, 2, 4), .5)
            inside = step(
                sub(half_size, absolute(sub(uv, centre))), 0, 4
            )
            rectangle = mul(sl(inside, 0, 1), sl(inside, 1, 2))
            hole = relu(sub(rectangle, component))
            hole_extrema = gmax(mul(ext, hole))
            hole_centre = mul(
                linear(hole_extrema, box_w[:2], box_b[:2]), gmax(hole)
            )
            is_selected = gmax(mul(component, selected_plane))
        else:
            hole_centre = linear(valid, np.zeros((2, 1), np.float32))
            is_selected = linear(valid, np.zeros((1, 1), np.float32))

        features = cat(box, hole_centre, valid, is_selected)
        new_history = cat(sl(history, 8, 8 * count), features)
        result, _ = end_loop(
            context, [sub(rem, component), new_history], count
        )
        return result[1]

    source_hist = extract(source_colors, S, 2, False)
    obstacle_hist = extract(obstacle_colors, N, 8, True)
    cup_hist = extract(cup_colors, K, 4, True)

    def hist_field(hist, count, index):
        a = np.zeros((count, 8 * count), np.float32)
        for j in range(count):
            a[j, 8*j + index] = 1
        return linear(hist, a)

    source_widths = hist_field(source_hist, S, 2)
    scale = sl(source_widths, 0, 1)
    hinges = relu(linear(
        scale, np.ones((7, 1), np.float32),
        -np.arange(1, 8, dtype=np.float32),
    ))
    coefficients = [-.5]
    for j in range(2, 8):
        coefficients.append(1/(j+1) - 2/j + 1/(j-1))
    inv_scale = linear(hinges, [coefficients], [1])

    def fields(hist, count, wanted):
        result = {}
        for j in wanted:
            value = hist_field(hist, count, j)
            result[j] = mul(value, inv_scale) if j < 6 else value
        return result

    ov = fields(obstacle_hist, N, range(8))
    cv = fields(cup_hist, K, range(7))
    sv = fields(source_hist, S, [0, 1, 6])

    logical_uv = mul(uv, inv_scale)
    logical_u = sl(logical_uv, 0, 1)
    logical_v = sl(logical_uv, 1, 2)

    bg_ext = gmax(mul(ext, sl("x0", 12, 13)))
    bounds = linear(bg_ext, [
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], [62.5, .5, .5])
    bounds = mul(bounds, inv_scale)
    board_left = sl(bounds, 0, 1)
    board_right = sl(bounds, 1, 2)
    board_bottom = sl(bounds, 2, 3)
    minimum_top = affine(sl(sv[1], 0, 1), 1, 1.5)

    side_excess = linear(bg_ext, [[-1, 0, 1, 0]])
    side_wall = gmax(step(
        sub(add(zero, side_excess), affine(scale, .5))
    ))
    safe_sides = affine(side_wall, -1, 1)

    # Sparse column slots encode objects; image rows encode independent
    # candidate arrangements.
    object_columns = [2 + 4*j for j in range(N)]
    source_columns = [34 + 4*j for j in range(S)]
    ray_columns = (
        object_columns + [j + 1 for j in object_columns] + source_columns
    )

    def column_masks(columns):
        difference = linear(
            raw_col, np.ones((len(columns), 1), np.float32),
            -np.asarray(columns, np.float32),
        )
        return step(affine(absolute(difference), -1), -0.25, 8)

    omask = column_masks(object_columns)
    smask = column_masks(source_columns)
    rmask = column_masks(ray_columns)

    def field(vector, mask):
        return sum_channels(mul(mask, vector))

    def object_max(values):
        return gmax(sum_channels(mul(values, omask)))

    of = {j: field(ov[j], omask) for j in range(7)}
    cf = {j: field(cv[j], omask) for j in range(7)}
    sf = {j: field(sv[j], smask) for j in (0, 1, 6)}

    ox, oy, ow, oh, ohx, ohy, valid_o = [of[j] for j in range(7)]
    cx, cy, cw, ch, chx, chy, valid_c = [cf[j] for j in range(7)]
    half_w, half_h = affine(ow, .5), affine(oh, .5)
    cup_half_w, cup_half_h = affine(cw, .5), affine(ch, .5)

    corner = mul(
        mul(step(ow, 1.5), step(oh, 1.5)), step(sub(oy, ohy))
    )
    turn_right = mul(corner, step(sub(ohx, ox)))
    turn_left = mul(corner, step(sub(ox, ohx)))
    turning = add(turn_left, turn_right)
    straight = sub(valid_o, turning)
    notch_offset = affine(sub(turn_right, turn_left), .5)
    hit_half_width = add(
        mul(half_w, affine(turning, -1, 1)), affine(turning, .5)
    )
    left_enable = sub(valid_o, turn_right)
    right_enable = sub(valid_o, turn_left)

    def expand_rays(ray_field):
        return rowmax(mul(rmask, ray_field))

    ray_dx_field = add(affine(turn_left, -1), shift_right(turn_right))
    ray_dy_field = add(
        add(sub(valid_o, turn_left), shift_right(sub(valid_o, turn_right))),
        sf[6],
    )
    ray_dx = expand_rays(ray_dx_field)
    ray_dy = expand_rays(ray_dy_field)
    ray_abs_dx = absolute(ray_dx)

    vertical_cup = step(sub(cw, ch))
    horizontal_cup = sub(valid_c, vertical_cup)
    cup_dy = sub(
        mul(vertical_cup, step(sub(cy, chy))),
        mul(vertical_cup, step(sub(chy, cy))),
    )
    cup_dx = sub(
        mul(horizontal_cup, step(sub(cx, chx))),
        mul(horizontal_cup, step(sub(chx, cx))),
    )
    direction_match = mul(
        relu(add(mul(cup_dx, ray_dx), mul(cup_dy, ray_dy))), valid_c
    )
    cup_parallel = add(mul(chy, ray_dy), mul(chx, ray_dx))
    cup_perpendicular = add(mul(chx, ray_dy), mul(chy, ray_abs_dx))
    cup_count = rowsum(valid_c)

    has_side_receiver = gmax(horizontal_cup)
    side_receiver_y = gmax(mul(chy, horizontal_cup))
    corner_goal_y = affine(side_receiver_y, 1, .5)
    upper_splitter_y = affine(side_receiver_y, 1, -1)

    expanded_half_w = rowmax(mul(omask, half_w))
    expanded_half_h = rowmax(mul(omask, half_h))
    off_diagonal = affine(omask, -1, 1)

    # Match previous targets to the current extraction order.
    old_x = sl("mem", 0, N)
    old_y = sl("mem", N, 2*N)
    old_plan_x = sl("mem", 2*N, 3*N)
    old_plan_y = sl("mem", 3*N, 4*N)
    old_selected = sl("mem", 4*N, 5*N)
    clock = sl("mem", 5*N, 5*N+1)
    memory_valid = sl("mem", 5*N+1, 5*N+2)

    screen_delta_w = np.zeros((2, 8), np.float32)
    screen_delta_w[0, 4], screen_delta_w[0, 3] = 1, -1
    screen_delta_w[1, 2], screen_delta_w[1, 1] = 1, -1
    screen_delta = linear("aprev", screen_delta_w)
    ab = linear(flags, [[1, 0, -1, 0], [0, -1, 0, 1]])
    a, b = sl(ab, 0, 1), sl(ab, 1, 2)
    dxs, dys = sl(screen_delta, 0, 1), sl(screen_delta, 1, 2)
    previous_dx = add(mul(a, dxs), mul(b, dys))
    previous_dy = sub(mul(a, dys), mul(b, dxs))
    expected_x = add(old_x, mul(old_selected, previous_dx))
    expected_y = add(old_y, mul(old_selected, previous_dy))

    match_distance = add(
        absolute(sub(ox, expected_x)), absolute(sub(oy, expected_y))
    )
    matches = step(affine(match_distance, -1), -0.25)
    has_match = minimum(sum_channels(matches), one)
    mapped_x = add(
        sum_channels(mul(matches, old_plan_x)),
        mul(ox, affine(has_match, -1, 1)),
    )
    mapped_y = add(
        sum_channels(mul(matches, old_plan_y)),
        mul(oy, affine(has_match, -1, 1)),
    )
    anchor_x = blend(ox, mapped_x, memory_valid)
    anchor_y = blend(oy, mapped_y, memory_valid)

    lower_x = add(board_left, half_w)
    upper_x = sub(board_right, half_w)
    lower_y = add(minimum_top, half_h)
    upper_y = sub(board_bottom, half_h)

    seed_rng = linear("coords", [[.71, .13, 0]], [.08])
    seed_rng = add(seed_rng, affine(clock, .04))
    mutate_rows = step(raw_row, .5)

    # Some population members start with splitters above a sideways
    # receiver and turners aligned with its opening. Others retain the
    # current or remembered arrangement.
    guided_rows = mul(step(raw_row, 15.5), has_side_receiver)
    seed_q = quantized(logistic(seed_rng), [.25, .5, .75], [1, 1, 1], 0)
    guided_bar_y = minimum(add(lower_y, seed_q), upper_splitter_y)
    guided_y = add(
        mul(straight, guided_bar_y), mul(turning, corner_goal_y)
    )
    corner_shift = quantized(
        seed_rng, [.16, .32, .68, .84], [2, 2, 2, 2], -4
    )
    guided_x = add(anchor_x, mul(turning, corner_shift))
    guided_x = mul(
        minimum(maximum(guided_x, lower_x), upper_x), valid_o
    )
    guided_y = mul(
        minimum(maximum(guided_y, lower_y), upper_y), valid_o
    )
    initial_x = blend(anchor_x, guided_x, guided_rows)
    initial_y = blend(anchor_y, guided_y, guided_rows)
    initial_score = affine(one, 0, -100000)

    # Fixed-weight routing and arrangement-refinement circuit.
    context = begin_loop([
        initial_x, initial_y, anchor_x, anchor_y,
        initial_score, zero, seed_rng,
    ])
    px, py, best_x, best_y, best_score, best_win, rng = context[3]

    left_x = mul(sub(affine(px, 1, -.5), half_w), valid_o)
    right_x = mul(add(affine(px, 1, .5), half_w), valid_o)
    top_y = sub(py, half_h)
    left_y = mul(add(affine(top_y, 1, -.5), turn_left), valid_o)
    right_y = mul(add(affine(top_y, 1, -.5), turn_right), valid_o)
    rx = expand_rays(add(add(left_x, shift_right(right_x)), sf[0]))
    ry = expand_rays(add(add(left_y, shift_right(right_y)), sf[1]))

    collision_x = add(px, notch_offset)
    horizontal_distance = absolute(sub(collision_x, rx))
    inside = step(sub(hit_half_width, horizontal_distance))
    distance = sub(add(top_y, turning), ry)
    forward = step(distance, -0.1)
    hits = mul(mul(mul(inside, forward), valid_o), ray_dy)
    hit_distance = affine(
        mul(affine(distance, 1, -128), hits), 1, 128
    )
    nearest = affine(rowmax(affine(hit_distance, -1)), -1)
    edge = mul(step(sub(nearest, hit_distance), -0.1), hits)

    reach = sf[6]
    for _ in range(N):
        incoming = minimum(
            sum_channels(mul(edge, expand_rays(reach))), one
        )
        reach = add(
            add(mul(incoming, left_enable),
                shift_right(mul(incoming, right_enable))),
            sf[6],
        )

    reached_rays = expand_rays(reach)
    obstructed_down = step(affine(nearest, -1, 128), 1)
    terminal = mul(reached_rays, affine(obstructed_down, -1, 1))

    outside = add(
        step(sub(board_left, rx)), step(sub(rx, board_right))
    )
    absorbed = mul(maximum(outside, ray_abs_dx), safe_sides)
    terminal = mul(terminal, affine(absorbed, -1, 1))

    ray_parallel = add(mul(ry, ray_dy), mul(rx, ray_dx))
    ray_perpendicular = add(mul(rx, ray_dy), mul(ry, ray_abs_dx))
    perpendicular_error = absolute(sub(cup_perpendicular, ray_perpendicular))
    ahead = step(sub(cup_parallel, ray_parallel), -0.1)
    eligible = mul(mul(ahead, direction_match), terminal)
    smooth = mul(relu(affine(perpendicular_error, -1/6, 1)), eligible)
    exact = mul(
        step(affine(perpendicular_error, -1), -0.4), eligible
    )

    smooth_power = mul(smooth, smooth)
    smooth_power = mul(smooth_power, smooth_power)
    smooth_coverage = rowsum(
        minimum(sum_channels(smooth_power), valid_c)
    )
    exact_coverage = rowsum(minimum(sum_channels(exact), valid_c))
    smooth_leak = sum_channels(
        mul(terminal, affine(rowmax(smooth), -1, 1))
    )
    exact_leak = sum_channels(
        mul(terminal, affine(rowmax(exact), -1, 1))
    )

    ex = rowmax(mul(omask, px))
    ey = rowmax(mul(omask, py))
    cup_near_x = step(sub(
        affine(add(expanded_half_w, cup_half_w), 1, 1),
        absolute(sub(ex, cx)),
    ))
    cup_near_y = step(sub(
        affine(add(expanded_half_h, cup_half_h), 1, 1),
        absolute(sub(ey, cy)),
    ))
    cup_collision = rowsum(sum_channels(
        mul(mul(mul(cup_near_x, cup_near_y), ov[6]), valid_c)
    ))

    overlap_x = step(sub(
        affine(add(expanded_half_w, half_w), 1, .1),
        absolute(sub(ex, px)),
    ))
    overlap_y = step(sub(
        affine(add(expanded_half_h, half_h), 1, .1),
        absolute(sub(ey, py)),
    ))
    overlap = mul(
        mul(mul(overlap_x, overlap_y), off_diagonal), ov[6]
    )
    overlap = affine(
        rowsum(sum_channels(mul(overlap, valid_o))), .5
    )
    invalid = add(cup_collision, overlap)

    deficit = add(
        add(sub(cup_count, exact_coverage), exact_leak), invalid
    )
    won = step(affine(deficit, -1), -0.1)
    fitness = add(smooth_coverage, affine(exact_coverage, 1.5))
    fitness = sub(fitness, affine(smooth_leak, .15))
    fitness = sub(fitness, affine(exact_leak, .3))
    fitness = sub(fitness, affine(invalid, 20))
    fitness = add(fitness, affine(won, 10))

    preparation_cost = rowsum(add(
        mul(turning, absolute(sub(py, corner_goal_y))),
        mul(straight, relu(sub(py, upper_splitter_y))),
    ))
    preparation_cost = mul(preparation_cost, has_side_receiver)
    fitness = sub(fitness, affine(preparation_cost, .003))

    accept = step(
        sub(fitness, best_score), -0.000001, 1000000
    )
    bx = blend(best_x, px, accept)
    by = blend(best_y, py, accept)
    bf = blend(best_score, fitness, accept)
    bw = blend(best_win, won, accept)

    random_x = logistic(rng)
    random_y = logistic(random_x)
    random_mutate = logistic(random_y)
    delta_x = quantized(
        random_x, [.04, .12, .36, .64, .88, .96],
        [4, 1, 1, 1, 1, 4], -6,
    )
    delta_y = quantized(
        random_y, [.03, .09, .20, .80, .91, .97],
        [3, 2, 1, 1, 2, 3], -6,
    )
    mutation = mul(
        mul(step(random_mutate, .72, 1000000), valid_o), mutate_rows
    )
    proposal_x = add(bx, mul(delta_x, mutation))
    proposal_y = add(by, mul(delta_y, mutation))
    proposal_x = mul(
        minimum(maximum(proposal_x, lower_x), upper_x), valid_o
    )
    proposal_y = mul(
        minimum(maximum(proposal_y, lower_y), upper_y), valid_o
    )

    result, planner_node = end_loop(
        context,
        [proposal_x, proposal_y, bx, by, bf, bw, random_mutate],
        1,
    )
    _, _, final_x, final_y, final_score, final_win, _ = result

    travel = rowsum(mul(
        add(absolute(sub(final_x, ox)), absolute(sub(final_y, oy))),
        valid_o,
    ))
    exploration = mul(
        minimum(travel, one), affine(final_win, -1, 1)
    )
    ranking = add(
        sub(final_score, affine(travel, .0005)),
        affine(exploration, .02),
    )
    tied = step(
        sub(ranking, gmax(ranking)), -0.000001, 1000000
    )
    tie_value = mul(affine(raw_row, 1, 1), tied)
    chosen_row = mul(
        step(affine(absolute(sub(tie_value, gmax(tie_value))), -1), -0.1),
        tied,
    )
    plan_x = gmax(mul(mul(final_x, chosen_row), omask))
    plan_y = gmax(mul(mul(final_y, chosen_row), omask))
    plan_wins = gmax(mul(final_win, chosen_row))

    # Select one unfinished object, with fully saturated binary gates.
    dx = sub(add(zero, plan_x), ov[0])
    dy = sub(add(zero, plan_y), ov[1])
    remaining_distance = add(absolute(dx), absolute(dy))
    unfinished = mul(step(remaining_distance, .35), ov[6])
    selection_bias = linear(
        "avail", np.zeros((N, 8), np.float32),
        -np.arange(N, dtype=np.float32) * .01,
    )
    priorities = add(
        mul(
            unfinished,
            sub(affine(add(zero, ov[7]), 200, 100), remaining_distance),
        ),
        selection_bias,
    )
    chosen = mul(
        step(sub(priorities, object_max(priorities)), -.004, 1000),
        unfinished,
    )
    any_move = object_max(unfinished)
    already_selected = gmax(sum_channels(mul(chosen, ov[7])))

    def chosen_value(value):
        return gmax(sum_channels(mul(chosen, value)))

    chosen_x = chosen_value(ov[0])
    chosen_y = chosen_value(ov[1])
    chosen_w = chosen_value(ov[2])
    chosen_h = chosen_value(ov[3])
    target_x = chosen_value(plan_x)
    target_y = chosen_value(plan_y)

    click_score = sub(
        sub(affine(movable_mask, 100), absolute(sub(logical_u, chosen_x))),
        absolute(sub(logical_v, chosen_y)),
    )
    emit("add", [click_score, zero], name="click")

    move_dx = np.array([0, 0, -1, 1], np.float32)
    move_dy = np.array([-1, 1, 0, 0], np.float32)
    new_x = add(
        zero, linear(chosen_x, np.ones((4, 1), np.float32), move_dx)
    )
    new_y = add(
        zero, linear(chosen_y, np.ones((4, 1), np.float32), move_dy)
    )
    chosen_half_w = affine(chosen_w, .5)
    chosen_half_h = affine(chosen_h, .5)

    near_x = step(sub(
        affine(add(chosen_half_w, cup_half_w), 1, 1),
        absolute(sub(new_x, cx)),
    ))
    near_y = step(sub(
        affine(add(chosen_half_h, cup_half_h), 1, 1),
        absolute(sub(new_y, cy)),
    ))
    blocked = rowmax(mul(mul(near_x, near_y), valid_c))
    blocked = add(
        blocked, relu(sub(add(board_left, chosen_half_w), new_x))
    )
    blocked = add(
        blocked, relu(sub(new_x, sub(board_right, chosen_half_w)))
    )
    blocked = add(
        blocked, relu(sub(add(minimum_top, chosen_half_h), new_y))
    )
    blocked = add(
        blocked, relu(sub(new_y, sub(board_bottom, chosen_half_h)))
    )

    move_cost = affine(add(
        absolute(sub(new_x, target_x)), absolute(sub(new_y, target_y))
    ), -1)
    move_cost = sub(move_cost, affine(blocked, 100))
    previous_canonical = cat(previous_dx, previous_dy)
    continuation = linear(
        previous_canonical, np.stack([move_dx, move_dy], axis=1) * .1
    )
    canonical_logits = gmax(add(move_cost, continuation))

    # Rotate the four movement scores back to displayed action directions.
    permutations = [
        [0, 1, 2, 3], [3, 2, 0, 1],
        [1, 0, 3, 2], [2, 3, 1, 0],
    ]
    direction_w = np.zeros((16, 4), np.float32)
    flag_w = np.zeros((16, 4), np.float32)
    reduce_w = np.zeros((4, 16), np.float32)
    for rotation, permutation in enumerate(permutations):
        for action, canonical in enumerate(permutation):
            index = 4 * rotation + action
            direction_w[index, canonical] = 1
            flag_w[index, rotation] = 1
            reduce_w[action, index] = 1
    screen_logits = linear(
        mul(linear(canonical_logits, direction_w), linear(flags, flag_w)),
        reduce_w,
    )

    enabled = mul(any_move, already_selected)
    action_features = cat(
        screen_logits, enabled, any_move, already_selected, plan_wins
    )
    output_w = np.zeros((8, 8), np.float32)
    for j in range(4):
        output_w[j + 1, j] = 1
        output_w[j + 1, 4] = 200
    output_w[5, 5] = -1100
    output_w[5, 7] = 1100
    output_w[6, 6] = -1100
    linear(
        action_features, output_w,
        [-1000, 0, 0, 0, 0, -1000, 100, -1000],
        name="logits",
    )

    next_clock = logistic(affine(clock, .99, .005))
    cat(
        ov[0], ov[1], plan_x, plan_y, ov[7], next_clock, const(1),
        name="mem_next",
    )

    # Algebraically compose consecutive, single-use linear maps.
    # This is graph compilation, not weight fitting or game evaluation.
    references = {}

    def count_references(nodes):
        for node in nodes:
            if node["op"] == "loop":
                for state in node["state"]:
                    if "init" in state:
                        z = state["init"]
                        references[z] = references.get(z, 0) + 1
                for z in node["next"].values():
                    references[z] = references.get(z, 0) + 1
                count_references(node["body"])
            else:
                for z in node["in"]:
                    references[z] = references.get(z, 0) + 1

    count_references(graph)
    for name in ("logits", "click", "mem_next"):
        references[name] = references.get(name, 0) + 1

    def compose_linear_maps(nodes):
        local = {node["id"]: node for node in nodes}
        removed = set()
        for node in nodes:
            if node["op"] == "loop":
                compose_linear_maps(node["body"])
                continue
            if node["op"] not in ("conv", "dense"):
                continue
            while True:
                predecessor = local.get(node["in"][0])
                if predecessor is None:
                    break
                if predecessor["id"] in removed:
                    break
                if predecessor["op"] != node["op"]:
                    break
                if references.get(predecessor["id"], 0) != 1:
                    break
                w2 = params[node["w"]]
                w1 = params[predecessor["w"]]
                if node["op"] == "conv":
                    if w1.shape[-2:] != (1, 1) or w2.shape[-2:] != (1, 1):
                        break
                    a2, a1 = w2[:, :, 0, 0], w1[:, :, 0, 0]
                else:
                    a2, a1 = w2, w1
                b1 = params.get(
                    predecessor.get("b", ""),
                    np.zeros(a1.shape[0], np.float32),
                )
                b2 = params.get(
                    node.get("b", ""), np.zeros(a2.shape[0], np.float32)
                )
                new_w = a2 @ a1
                new_b = a2 @ b1 + b2
                if node["op"] == "conv":
                    new_w = new_w[:, :, None, None]
                node["w"] = parameter(new_w)
                node["b"] = parameter(new_b)
                node["in"] = predecessor["in"][:]
                removed.add(predecessor["id"])
        nodes[:] = [node for node in nodes if node["id"] not in removed]

    compose_linear_maps(graph)

    used_parameters = set()

    def collect_parameters(nodes):
        for node in nodes:
            if node["op"] == "loop":
                collect_parameters(node["body"])
            else:
                if "w" in node:
                    used_parameters.add(node["w"])
                if "b" in node:
                    used_parameters.add(node["b"])

    collect_parameters(graph)
    final_params = {
        name: value for name, value in params.items()
        if name in used_parameters
    }

    # Count static resources and choose a permissible recurrence length.
    def statistics(nodes):
        macs = evaluations = count = spatial = 0
        for node in nodes:
            count += 1
            evaluations += 1
            op = node["op"]
            if op == "loop":
                bm, be, bn, bs = statistics(node["body"])
                macs += node["T"] * bm
                evaluations += node["T"] * be
                count += bn
                spatial += bs
                for state in node["state"]:
                    if shapes[state["id"]][0] == "s":
                        spatial += shapes[state["id"]][1]
                continue
            kind, width = shapes[node["id"]]
            if kind == "s":
                spatial += width
            if op == "conv":
                macs += int(params[node["w"]].size) * 4096
            elif op == "dense":
                macs += int(params[node["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += width * 4096
            elif op in (
                "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
            ):
                macs += shapes[node["in"][0]][1] * 4096
            elif op in ("vadd", "vmul"):
                macs += width
        return macs, evaluations, count, spatial

    total_m, total_e, _, _ = statistics(graph)
    body_m, body_e, _, _ = statistics(planner_node["body"])
    outside_m = total_m - body_m
    outside_e = total_e - body_e
    planner_node["T"] = int(max(1, min(
        128,
        (1500000000 - outside_m) // max(1, body_m),
        (10000 - outside_e) // max(1, body_e),
    )))

    return {
        "graph": graph,
        "memory_size": M,
        "params": final_params,
    }

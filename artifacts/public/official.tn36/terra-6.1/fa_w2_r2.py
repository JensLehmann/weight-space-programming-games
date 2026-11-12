import numpy as np


def build():
    graph = []
    params = {}
    node_counter = 0
    param_counter = 0

    def fresh():
        nonlocal node_counter
        name = "n%d" % node_counter
        node_counter += 1
        return name

    def param(value):
        nonlocal param_counter
        name = "p%d" % param_counter
        param_counter += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, **fields):
        name = fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        graph.append(nd)
        return name

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

    def vmax(a, b):
        return node("max2", [a, b])

    def affine(x, scale, bias=0.0):
        return node(
            "conv",
            [x],
            w=param(np.array([[[[scale]]]], dtype=np.float32)),
            b=param(np.array([bias], dtype=np.float32)),
        )

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def summed_channels(x, count):
        return node(
            "conv",
            [x],
            w=param(np.ones((1, count, 1, 1), dtype=np.float32)),
        )

    def gmax(x):
        return node("gmax", [x])

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    zero = sub(one, one)
    row_px = affine(row, 63.0)
    col_px = affine(col, 63.0)

    def clip01(x):
        return node("min2", [node("relu", [x]), one])

    def triangle(x):
        return node("relu", [affine(node("abs", [x]), -1.0, 1.0)])

    def lift(v):
        return add(zero, v)

    def vec_affine(v, scale, bias=0.0):
        return dense(
            v,
            np.array([[scale]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def position(mask, coordinate):
        return gmax(mul(mask, coordinate))

    # Locate goal and token templates, excluding the lower icon area.
    scene = mul(
        clip01(affine(row_px, 1.0, -2.0)),
        clip01(affine(row_px, -1.0, 39.0)),
    )

    goal = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    token = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    token_big = np.repeat(np.repeat(token, 2, axis=0), 2, axis=1)

    templates = []
    for k in range(4):
        templates.append((np.rot90(goal, -k), -2, -2))
    for k in range(4):
        templates.append((np.rot90(token, -k), -1, -1))
    for k in range(4):
        rotated = np.rot90(token_big, -k)
        templates.append((rotated[1:7, 1:7], -2, -2))

    tw = np.zeros((12, 1, 7, 7), dtype=np.float32)
    tb = np.zeros(12, dtype=np.float32)
    for i, (template, dy0, dx0) in enumerate(templates):
        for y in range(template.shape[0]):
            for x in range(template.shape[1]):
                tw[i, 0, y + dy0 + 3, x + dx0 + 3] = (
                    2.0 if template[y, x] else -2.0
                )
        tb[i] = 1.0 - 2.0 * float(template.sum())

    colour_b = sl("x0", 11, 12)
    matches = node(
        "relu",
        [node("conv", [colour_b], w=param(tw), b=param(tb))],
    )
    matches = mul(matches, scene)

    goals = sl(matches, 0, 4)
    small_raw = sl(matches, 4, 8)
    big = sl(matches, 8, 12)
    goal_map = summed_channels(goals, 4)
    big_map = summed_channels(big, 4)
    big_present = gmax(big_map)
    no_big = vec_affine(big_present, -1.0, 1.0)
    small = mul(small_raw, no_big)
    small_map = summed_channels(small, 4)

    gx = position(goal_map, col)
    gy = position(goal_map, row)
    sx = position(small_map, col)
    sy = position(small_map, row)
    bx = position(big_map, col)
    by = position(big_map, row)
    gf = gmax(goals)
    sf = gmax(small)
    bf = gmax(big)

    features = node(
        "vconcat",
        [gx, gy, sx, sy, bx, by, big_present, gf, sf, bf],
    )

    fw = np.zeros((3, 19), dtype=np.float32)
    fw[0, 0], fw[0, 2], fw[0, 4], fw[0, 6] = (
        63 / 4, -63 / 4, -63 / 4, 0.5
    )
    fw[1, 1], fw[1, 3], fw[1, 5], fw[1, 6] = (
        63 / 4, -63 / 4, -63 / 4, 0.5
    )
    for k in range(4):
        fw[2, 7 + k] = k
        fw[2, 11 + k] = -k
        fw[2, 15 + k] = -k
    displacement = dense(features, fw)

    signs_w = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    signs = node("vrelu", [dense(displacement, signs_w)])
    right_n = vsl(signs, 0, 1)
    left_n = vsl(signs, 1, 2)
    down_n = vsl(signs, 2, 3)
    up_n = vsl(signs, 3, 4)
    nx = node("vadd", [right_n, left_n])
    ny = node("vadd", [down_n, up_n])
    offset_y = node("vadd", [big_present, nx])
    offset_rot = node("vadd", [offset_y, ny])

    right_flag = clip01(lift(right_n))
    left_flag = clip01(lift(left_n))
    down_flag = clip01(lift(down_n))
    up_flag = clip01(lift(up_n))

    dr = lift(vsl(displacement, 2, 3))
    rot_90 = add(
        triangle(affine(dr, 1.0, -1.0)),
        triangle(affine(dr, 1.0, 3.0)),
    )
    rot_180 = add(
        triangle(affine(dr, 1.0, -2.0)),
        triangle(affine(dr, 1.0, 2.0)),
    )
    rot_270 = add(
        triangle(affine(dr, 1.0, -3.0)),
        triangle(affine(dr, 1.0, 1.0)),
    )

    # Program-bit detectors.
    uw = np.zeros((2, 16, 5, 5), dtype=np.float32)
    for dx in (-1, 0, 1):
        uw[0, 1, 2, 2 + dx] = 2
        uw[0, 5, 2, 2 + dx] = 2
    for dy, dx in ((0, -2), (0, 2), (-1, 0), (1, 0)):
        uw[0, 0, 2 + dy, 2 + dx] = 2
    for dy in (-1, 0, 1):
        uw[1, 1, 2 + dy, 2] = 2
        uw[1, 5, 2 + dy, 2] = 2
    for dy, dx in ((-2, 0), (2, 0), (0, -1), (0, 1)):
        uw[1, 0, 2 + dy, 2 + dx] = 2

    marks = node(
        "relu",
        [
            node(
                "conv",
                ["x0"],
                w=param(uw),
                b=param(np.array([-13, -13], dtype=np.float32)),
            )
        ],
    )
    marks = vmax(sl(marks, 0, 1), sl(marks, 1, 2))
    ui_region = mul(
        mul(
            clip01(affine(row_px, 1.0, -31.0)),
            clip01(affine(row_px, -1.0, 50.0)),
        ),
        clip01(affine(col_px, 1.0, -18.0)),
    )
    marks = mul(marks, ui_region)

    first_y = vec_affine(
        gmax(mul(marks, affine(row, -1.0, 1.0))), -63.0, 63.0
    )
    first_x = vec_affine(
        gmax(mul(marks, affine(col, -1.0, 1.0))), -63.0, 63.0
    )
    column_index = affine(sub(col_px, first_x), 0.2)
    bit_index = affine(sub(row_px, first_y), 1.0 / 3.0)

    def interval_at(index, start, length):
        relative = sub(index, start)
        entered = clip01(affine(relative, 1.0, 1.0))
        remaining = clip01(sub(length, relative))
        return mul(entered, remaining)

    # Direct-route fallback.
    shrink_slot = mul(triangle(column_index), big_present)
    horizontal = interval_at(column_index, big_present, nx)
    vertical = interval_at(column_index, offset_y, ny)
    rotation_slot = triangle(sub(column_index, offset_rot))

    direct_bits = {}
    direct_bits[0] = add(
        add(shrink_slot, mul(horizontal, left_flag)),
        add(vertical, mul(rotation_slot, add(rot_90, rot_180))),
    )
    direct_bits[1] = add(
        add(mul(horizontal, right_flag), mul(vertical, down_flag)),
        mul(rotation_slot, add(rot_180, rot_270)),
    )
    direct_bits[2] = mul(
        rotation_slot, add(add(rot_90, rot_180), rot_270)
    )
    direct_bits[3] = shrink_slot
    direct_bits[5] = mul(vertical, up_flag)

    direct_desired = zero
    for index, value in direct_bits.items():
        selector = triangle(affine(bit_index, 1.0, -float(index)))
        direct_desired = add(direct_desired, mul(selector, value))
    direct_desired = clip01(direct_desired)

    # Infer the playable area. Coordinates refer to token anchors, one
    # pixel down and right from the upper-left corner of a scale-1 token.
    right_board = clip01(affine(lift(gx), 63.0, -32.0))
    left_board = sub(one, right_board)

    right_area = mul(
        mul(
            clip01(affine(col_px, 1.0, -33.0)),
            clip01(affine(col_px, -1.0, 59.0)),
        ),
        mul(
            clip01(affine(row_px, 1.0, -4.0)),
            clip01(affine(row_px, -1.0, 30.0)),
        ),
    )
    center_area = mul(
        mul(
            clip01(affine(col_px, 1.0, -14.0)),
            clip01(affine(col_px, -1.0, 48.0)),
        ),
        mul(
            clip01(affine(row_px, 1.0, -9.0)),
            clip01(affine(row_px, -1.0, 39.0)),
        ),
    )
    board = add(mul(right_board, right_area), mul(left_board, center_area))

    # A scale-1 token occupies a 4x4 square around its anchor.
    wall_w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    wall_w[0, 0, 1:5, 1:5] = 1.0
    wall_count = node("conv", [sl("x0", 6, 7)], w=param(wall_w))
    clear = node("relu", [affine(wall_count, -1.0, 1.0)])
    valid = mul(board, clear)

    # Detect the two phases of the 4x4 checkerboard checkpoint sprite.
    cpw = np.zeros((2, 1, 5, 5), dtype=np.float32)
    for phase in range(2):
        for y in range(4):
            for x in range(4):
                cpw[phase, 0, y + 1, x + 1] = (
                    2.0 if (x + y) % 2 == phase else -2.0
                )
    checkpoints = node(
        "relu",
        [
            node(
                "conv",
                [colour_b],
                w=param(cpw),
                b=param(np.array([-15.0, -15.0], dtype=np.float32)),
            )
        ],
    )
    checkpoints = vmax(sl(checkpoints, 0, 1), sl(checkpoints, 1, 2))
    checkpoints = mul(checkpoints, board)

    # Four-pixel shifts are implemented as two two-pixel convolutions.
    # Direction order: left, right, down, up.
    directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    shift_parameters = {}

    def shift_weights(channels):
        if channels in shift_parameters:
            return shift_parameters[channels]
        w1 = np.zeros((4 * channels, channels, 5, 5), dtype=np.float32)
        w2 = np.zeros(
            (4 * channels, 4 * channels, 5, 5), dtype=np.float32
        )
        for d, (dy, dx) in enumerate(directions):
            for c in range(channels):
                out = d * channels + c
                w1[out, c, 2 - 2 * dy, 2 - 2 * dx] = 1.0
                w2[out, out, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        result = (param(w1), param(w2))
        shift_parameters[channels] = result
        return result

    def wavefront(seed, channels, iterations=24):
        w1, w2 = shift_weights(channels)
        loop_id = fresh()
        state_id = fresh()
        body = []

        def body_node(op, inputs, **fields):
            name = fresh()
            nd = {"id": name, "op": op, "in": list(inputs)}
            nd.update(fields)
            body.append(nd)
            return name

        s1 = body_node("conv", [state_id], w=w1)
        s2 = body_node("conv", [s1], w=w2)
        parts = [
            body_node(
                "slice",
                [s2],
                start=d * channels,
                stop=(d + 1) * channels,
            )
            for d in range(4)
        ]
        best = parts[0]
        for part in parts[1:]:
            best = body_node("max2", [best, part])
        subtract_w = np.eye(channels, dtype=np.float32).reshape(
            channels, channels, 1, 1
        )
        dec = body_node(
            "conv",
            [best],
            w=param(subtract_w),
            b=param(-np.ones(channels, dtype=np.float32)),
        )
        dec = body_node("relu", [dec])
        dec = body_node("mul", [dec, valid])
        updated = body_node("max2", [state_id, dec])
        graph.append(
            {
                "id": loop_id,
                "op": "loop",
                "T": iterations,
                "state": [{"id": state_id, "init": seed}],
                "body": body,
                "next": {state_id: updated},
            }
        )
        return loop_id + "." + state_id

    # Simultaneously compute distances from the token and goal.
    seeds = node("concat", [small_map, goal_map])
    seed_w = np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1) * 32.0
    seeds = node("conv", [seeds], w=param(seed_w))
    distances = wavefront(seeds, 2)
    from_start = sl(distances, 0, 1)
    from_goal = sl(distances, 1, 2)

    # A checkpoint must be reachable in six moves, different from the
    # starting position, and closer to the goal than the start is.
    reachable = clip01(affine(from_start, 1.0, -25.0))
    not_start = clip01(affine(from_start, -1.0, 32.0))
    goal_value_at_start = gmax(mul(small_map, from_goal))
    progress = clip01(sub(from_goal, goal_value_at_start))
    checkpoint_candidates = mul(
        checkpoints, mul(reachable, mul(not_start, progress))
    )

    # Reserve one program slot if reaching the goal also needs rotation.
    rotation_needed = clip01(node("abs", [dr]))
    goal_reachable = clip01(
        sub(affine(from_start, 1.0, -25.0), rotation_needed)
    )
    goal_candidates = mul(goal_map, goal_reachable)
    candidates = clip01(add(checkpoint_candidates, goal_candidates))

    # Select the candidate with the best remaining distance. The small
    # coordinate terms break ties deterministically.
    preference = add(
        from_goal,
        add(
            affine(row_px, -0.01, 0.0),
            affine(col_px, -0.0001, 0.0),
        ),
    )
    candidate_score = mul(candidates, preference)
    best_candidate_score = gmax(candidate_score)
    score_difference = sub(candidate_score, best_candidate_score)
    chosen = mul(
        candidates,
        triangle(affine(score_difference, 100000.0)),
    )
    use_wave = gmax(chosen)

    # Distances to the selected waypoint.
    waypoint_seed = affine(chosen, 32.0)
    to_waypoint = wavefront(waypoint_seed, 1)

    # Follow the gradient for up to six program instructions.
    sw1, sw2 = shift_weights(1)
    pos = small_map
    planned_bits = {0: zero, 1: zero, 2: zero, 3: zero, 5: zero}

    for i in range(6):
        shifted = node("conv", [pos], w=sw1)
        shifted = node("conv", [shifted], w=sw2)
        positions = [sl(shifted, d, d + 1) for d in range(4)]
        values = [
            lift(gmax(mul(mul(p, valid), to_waypoint)))
            for p in positions
        ]
        best_value = values[0]
        for value in values[1:]:
            best_value = vmax(best_value, value)
        current_value = lift(gmax(mul(pos, to_waypoint)))
        moving = clip01(sub(best_value, current_value))

        biased = [
            affine(values[d], 1.0, 0.25 * (3 - d))
            for d in range(4)
        ]
        best_biased = biased[0]
        for value in biased[1:]:
            best_biased = vmax(best_biased, value)
        selected = []
        for value in biased:
            difference = node("abs", [sub(value, best_biased)])
            selected.append(
                mul(moving, clip01(affine(difference, -10.0, 2.0)))
            )

        # left=1, right=2, down=3, up=33
        slot = triangle(affine(column_index, 1.0, -float(i)))
        planned_bits[0] = add(
            planned_bits[0],
            mul(slot, add(add(selected[0], selected[2]), selected[3])),
        )
        planned_bits[1] = add(
            planned_bits[1],
            mul(slot, add(selected[1], selected[2])),
        )
        planned_bits[5] = add(
            planned_bits[5],
            mul(slot, selected[3]),
        )

        next_pos = mul(pos, sub(one, moving))
        for p, select in zip(positions, selected):
            next_pos = add(next_pos, mul(p, select))
        pos = next_pos

    waypoint_is_goal = gmax(mul(chosen, goal_map))
    distance_to_waypoint = vec_affine(
        gmax(mul(chosen, from_start)), -1.0, 32.0
    )
    wave_rot_slot = mul(
        triangle(sub(column_index, distance_to_waypoint)),
        waypoint_is_goal,
    )
    planned_bits[0] = add(
        planned_bits[0], mul(wave_rot_slot, add(rot_90, rot_180))
    )
    planned_bits[1] = add(
        planned_bits[1], mul(wave_rot_slot, add(rot_180, rot_270))
    )
    planned_bits[2] = add(
        planned_bits[2],
        mul(wave_rot_slot, add(add(rot_90, rot_180), rot_270)),
    )

    planned_desired = zero
    for index, value in planned_bits.items():
        selector = triangle(affine(bit_index, 1.0, -float(index)))
        planned_desired = add(planned_desired, mul(selector, value))
    planned_desired = clip01(planned_desired)

    wave_enabled = mul(lift(use_wave), lift(no_big))
    desired = add(
        mul(planned_desired, wave_enabled),
        mul(direct_desired, sub(one, wave_enabled)),
    )
    desired = clip01(desired)

    current_on = sl("x0", 5, 6)
    mismatch = add(
        mul(desired, sub(one, current_on)),
        mul(sub(one, desired), current_on),
    )
    mismatch = mul(mismatch, marks)

    # Run button: a solid 5x5 block of colour 9.
    button_w = np.ones((1, 1, 5, 5), dtype=np.float32) * 2.0
    button = node(
        "relu",
        [
            node(
                "conv",
                [sl("x0", 9, 10)],
                w=param(button_w),
                b=param(np.array([-49.0], dtype=np.float32)),
            )
        ],
    )
    button = mul(button, clip01(affine(row_px, 1.0, -52.0)))
    button = mul(button, clip01(affine(col_px, 1.0, -18.0)))
    button = affine(button, 0.1)

    graph.append({"id": "click", "op": "add", "in": [mismatch, button]})
    logits_w = np.zeros((8, 8), dtype=np.float32)
    logits_b = np.zeros(8, dtype=np.float32)
    logits_b[6] = 1.0
    graph.append(
        {
            "id": "logits",
            "op": "dense",
            "in": ["avail"],
            "w": param(logits_w),
            "b": param(logits_b),
        }
    )

    return {"graph": graph, "memory_size": 0, "params": params}

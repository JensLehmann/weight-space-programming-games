import numpy as np


def build():
    params = {}
    graph = []
    scope = graph
    serial = 0
    parameter_cache = {}
    operation_cache = {}
    constants = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "avail": ("v", 8),
        "aprev": ("v", 8),
        "mem": ("v", 5),
    }

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p_" + str(len(params))
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, ins, name=None, **fields):
        nonlocal serial
        cache = operation_cache.setdefault(id(scope), {})
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        if name is None:
            name = "n_" + str(serial)
            serial += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        scope.append(nd)
        cache[key] = name
        if op == "conv":
            shapes[name] = ("s", params[fields["w"]].shape[0])
        elif op == "dense":
            shapes[name] = ("v", params[fields["w"]].shape[0])
        elif op in ("gmax", "gmean"):
            shapes[name] = ("v", shapes[ins[0]][1])
        elif op in ("concat", "vconcat"):
            shapes[name] = (
                "s" if op == "concat" else "v",
                sum(shapes[i][1] for i in ins),
            )
        elif op in ("slice", "vslice"):
            shapes[name] = (
                shapes[ins[0]][0], fields["stop"] - fields["start"]
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[name] = ("s", max(shapes[i][1] for i in ins))
        elif op in ("vadd", "vmul"):
            shapes[name] = ("v", max(shapes[i][1] for i in ins))
        else:
            shapes[name] = shapes[ins[0]]
        return name

    def conv(x, w, b=None, name=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("conv", [x], name, **f)

    def dense(x, w, b=None, name=None):
        f = {"w": parameter(w)}
        if b is not None:
            f["b"] = parameter(b)
        return node("dense", [x], name, **f)

    def constant(value):
        nonlocal scope
        value = float(value)
        if value not in constants:
            saved = scope
            scope = graph
            constants[value] = dense(
                "avail", np.zeros((1, 8), np.float32), [value]
            )
            scope = saved
        return constants[value]

    def affine(x, a=1.0, b=0.0):
        if a == 1 and b == 0:
            return x
        kind, c = shapes[x]
        w = np.eye(c, dtype=np.float32) * a
        bias = np.full(c, b, np.float32)
        if kind == "s":
            return conv(x, w[:, :, None, None], bias)
        return dense(x, w, bias)

    def add(x, y):
        op = "add" if "s" in (shapes[x][0], shapes[y][0]) else "vadd"
        return node(op, [x, y])

    def mul(x, y):
        op = "mul" if "s" in (shapes[x][0], shapes[y][0]) else "vmul"
        return node(op, [x, y])

    def relu(x):
        return node("relu" if shapes[x][0] == "s" else "vrelu", [x])

    def absolute(x):
        if shapes[x][0] == "s":
            return node("abs", [x])
        return add(relu(x), relu(affine(x, -1)))

    def clip(x):
        z = relu(x)
        if shapes[x][0] == "s":
            return node("min2", [z, constant(1)])
        return add(z, affine(relu(affine(z, 1, -1)), -1))

    def sl(x, a, b):
        return node(
            "slice" if shapes[x][0] == "s" else "vslice",
            [x], start=a, stop=b,
        )

    def cat(xs):
        return node(
            "concat" if shapes[xs[0]][0] == "s" else "vconcat", xs
        )

    def gm(x):
        return node("gmax", [x])

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def maximum(xs):
        z = xs[0]
        for x in xs[1:]:
            z = node("max2", [z, x])
        return z

    def summation(xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def centres(x, values):
        values = np.asarray(values, np.float32)
        d = dense(x, np.ones((len(values), 1), np.float32), -values)
        return clip(affine(absolute(d), -4, 2))

    def ge(x, threshold):
        return clip(affine(x, 2, 1 - 2 * threshold))

    def le(x, threshold):
        return clip(affine(x, -2, 1 + 2 * threshold))

    def shift(x, dx, dy):
        c = shapes[x][1]
        z = x
        for axis, amount in ((0, dx), (1, dy)):
            while amount:
                magnitude = 2 if abs(amount) == 4 else min(3, abs(amount))
                step = int(np.sign(amount)) * magnitude
                k = 2 * magnitude + 1
                w = np.zeros((c, c, k, k), np.float32)
                yy = k // 2 + (step if axis else 0)
                xx = k // 2 + (0 if axis else step)
                for i in range(c):
                    w[i, i, yy, xx] = 1
                z = conv(z, w)
                amount -= step
        return z

    row = conv("coords", np.array([[[[63]], [[0]], [[0]]]], np.float32))
    col = conv("coords", np.array([[[[0]], [[63]], [[0]]]], np.float32))
    blank = conv("coords", np.zeros((1, 3, 1, 1), np.float32))
    invrow = affine(row, -1, 64)
    invcol = affine(col, -1, 64)
    region = mul(
        mul(ge(row, 3), le(row, 40)),
        mul(ge(col, 13), le(col, 62)),
    )

    # The black program panel contains isolated three-pixel strokes.
    bit_w = np.zeros((2, 16, 7, 7), np.float32)
    for orientation in range(2):
        stroke = [(0, -1), (0, 0), (0, 1)]
        surround = (
            [(-1, j) for j in (-1, 0, 1)]
            + [(1, j) for j in (-1, 0, 1)]
            + [(0, -2), (0, 2)]
        )
        if orientation:
            stroke = [(b, a) for a, b in stroke]
            surround = [(b, a) for a, b in surround]
        for y, x in stroke:
            bit_w[orientation, 1, y + 3, x + 3] = 1
            bit_w[orientation, 5, y + 3, x + 3] = 1
        for y, x in surround:
            bit_w[orientation, 0, y + 3, x + 3] = 1

    bit_shapes = relu(conv("x0", bit_w, [-10, -10]))
    bit_region = mul(mul(ge(row, 32), le(row, 49)), ge(col, 13))
    horizontal = mul(sl(bit_shapes, 0, 1), bit_region)
    vertical = mul(sl(bit_shapes, 1, 2), bit_region)
    rightmost = gm(mul(horizontal, col))
    right_panel = ge(add(col, affine(rightmost, -1)), -25)
    horizontal = mul(horizontal, right_panel)
    vertical = mul(vertical, right_panel)

    ymin = affine(gm(mul(horizontal, invrow)), -1, 64)
    ymax = gm(mul(vertical, row))
    xmin = affine(gm(mul(horizontal, invcol)), -1, 64)
    xmax = gm(mul(horizontal, col))
    slots = affine(add(xmax, affine(xmin, -1)), 0.2, 1)
    bit_count = affine(add(ymax, affine(ymin, -1)), 1 / 3, 1)

    codes = [0, 1, 2, 3, 33, 10, 12, 8, 9]
    allowed = [constant(1)] + [
        clip(affine(bit_count, 1, 1 - int(code).bit_length()))
        for code in codes[1:]
    ]

    silhouette_w = np.zeros((1, 16, 1, 1), np.float32)
    colour_w = np.zeros_like(silhouette_w)
    wall_w = np.zeros_like(silhouette_w)
    for c in (8, 9, 11, 15):
        silhouette_w[0, c, 0, 0] = 1
        colour_w[0, c, 0, 0] = c
    for c in (2, 6, 7, 10, 12, 13):
        wall_w[0, c, 0, 0] = 1
    coloured = mul(conv("x0", silhouette_w), region)
    colour_values = mul(conv("x0", colour_w), region)
    walls = mul(conv("x0", wall_w), region)

    shifted_cache = {}

    def cached_shift(x, dx, dy):
        key = (x, dx, dy)
        if key not in shifted_cache:
            if dy:
                z = shift(cached_shift(x, dx, 0), 0, dy)
            else:
                z = shift(x, dx, 0)
            shifted_cache[key] = z
        return shifted_cache[key]

    def wide(x, filters, biases):
        tiles = {}
        for o, filt in enumerate(filters):
            for (dy, dx), value in filt.items():
                ty, tx = (dy + 3) // 7, (dx + 3) // 7
                key = (ty, tx)
                if key not in tiles:
                    tiles[key] = np.zeros(
                        (len(filters), 1, 7, 7), np.float32
                    )
                tiles[key][o, 0, dy - 7 * ty + 3, dx - 7 * tx + 3] += value
        parts = []
        for (ty, tx), kernel in sorted(tiles.items()):
            parts.append(conv(
                cached_shift(x, tx * 7, ty * 7),
                kernel, biases if not parts else None,
            ))
        return summation(parts)

    piece_template = np.ones((4, 4), np.float32)
    piece_template[3, 1:3] = 0
    target_template = np.zeros((6, 6), np.float32)
    target_template[1:5, 0] = 1
    target_template[1:5, 5] = 1
    target_template[4, 2:4] = 1
    target_template[5, :] = 1
    checkpoint_template = np.fromfunction(
        lambda y, x: (x + y) % 2 == 0, (4, 4)
    ).astype(np.float32)

    def sampled(template, scale, rotation, anchor):
        a = np.rot90(template, -rotation)
        filt = {
            (y * scale - anchor, x * scale - anchor):
                (1 if a[y, x] else -2)
            for y in range(a.shape[0])
            for x in range(a.shape[1])
        }
        return filt, 1 - int(a.sum())

    piece_filters, piece_biases = [], []
    target_filters, target_biases = [], []
    checkpoint_filters, checkpoint_biases = [], []
    for scale in range(1, 5):
        for rotation in range(4):
            f, b = sampled(piece_template, scale, rotation, 0)
            piece_filters.append(f)
            piece_biases.append(b)
            f, b = sampled(target_template, scale, rotation, scale)
            target_filters.append(f)
            target_biases.append(b)
        f, b = sampled(checkpoint_template, scale, 0, 0)
        checkpoint_filters.append(f)
        checkpoint_biases.append(b)

    pieces = relu(wide(coloured, piece_filters, piece_biases))
    targets = relu(wide(coloured, target_filters, target_biases))
    checkpoints = relu(wide(
        coloured, checkpoint_filters, checkpoint_biases
    ))

    group_w = np.zeros((4, 16, 1, 1), np.float32)
    rotation_w = np.zeros((4, 16), np.float32)
    for scale in range(4):
        group_w[scale, 4 * scale:4 * scale + 4, 0, 0] = 1
        for rotation in range(4):
            rotation_w[rotation, 4 * scale + rotation] = 1

    def anchors(maps):
        masks = conv(maps, group_w)
        present = clip(gm(masks))
        yy = affine(gm(mul(masks, invrow)), -1, 64)
        xx = affine(gm(mul(masks, invcol)), -1, 64)
        ry = clip(affine(absolute(add(row, affine(yy, -1))), -4, 2))
        rx = clip(affine(absolute(add(col, affine(xx, -1))), -4, 2))
        return mul(mul(ry, rx), present)

    start = anchors(pieces)
    goal = anchors(targets)
    piece_rotation = dense(gm(pieces), rotation_w)
    target_rotation = dense(gm(targets), rotation_w)

    pair_w = np.zeros((16, 8), np.float32)
    delta_w = np.zeros((4, 16), np.float32)
    for a in range(4):
        for b in range(4):
            pair_w[4 * a + b, a] = 1
            pair_w[4 * a + b, 4 + b] = 1
            delta_w[(b - a) % 4, 4 * a + b] = 1
    pairs = relu(dense(
        cat([piece_rotation, target_rotation]), pair_w,
        np.full(16, -1, np.float32),
    ))
    rotation_delta = dense(pairs, delta_w)
    rotate_needed = affine(sl(rotation_delta, 0, 1), -1, 1)

    piece_samples, target_samples = [], []
    for scale in range(1, 5):
        piece_samples.append(cached_shift(colour_values, scale, scale))
        target_samples.append(maximum([
            cached_shift(colour_values, -scale, -scale),
            cached_shift(colour_values, 5 * scale - 1, 5 * scale - 1),
        ]))
    piece_colour = vsum(gm(mul(start, cat(piece_samples))))
    target_colour = vsum(gm(mul(goal, cat(target_samples))))
    colour_difference = absolute(add(
        target_colour, affine(piece_colour, -1)
    ))
    recolour_needed = affine(
        clip(affine(colour_difference, -2, 1)), -1, 1
    )
    terminal_cost = add(rotate_needed, recolour_needed)
    colour_choice = centres(target_colour, [9, 8, 15])

    footprints = [
        {(y, x): 1 for y in range(4 * s) for x in range(4 * s)}
        for s in range(1, 5)
    ]
    free = relu(affine(wide(walls, footprints, [0] * 4), -1, 1))

    def scale_kernel(upward, push=False):
        w = np.zeros((4, 4, 1, 1), np.float32)
        for s in range(4):
            t = s + 1 if upward else max(0, s - 1)
            if t < 4:
                if push:
                    w[t, s, 0, 0] = 1
                else:
                    w[s, t, 0, 0] = 1
        return w

    def candidates(value):
        valid = mul(value, free)
        left = shift(valid, -4, 0)
        right = shift(valid, 4, 0)
        moves = [
            left, right,
            shift(valid, 0, 4), shift(valid, 0, -4),
            shift(right, 4, 0), shift(left, -4, 0),
            conv(valid, scale_kernel(True)),
            conv(valid, scale_kernel(False)),
        ]
        return [value] + [
            mul(relu(affine(x, 1, -1)), allowed[i])
            for i, x in enumerate(moves, 1)
        ]

    # Global distance-to-goal, including changes of scale.
    distance_base = mul(goal, affine(terminal_cost, -1, 100))
    saved = scope
    body = []
    scope = body
    shapes["distance_state"] = ("s", 4)
    distance_next = maximum(
        [distance_base] + candidates("distance_state")
    )
    scope = saved
    graph.append({
        "id": "distance", "op": "loop", "T": 24,
        "state": [{"id": "distance_state", "init": distance_base}],
        "body": body,
        "next": {"distance_state": distance_next},
    })
    shapes["distance.distance_state"] = ("s", 4)

    checkpoint_reward = mul(
        checkpoints, affine("distance.distance_state", 10)
    )
    goal_reward = mul(goal, affine(terminal_cost, -1, 1000))

    def feasible_goal(horizon):
        gate = clip(add(horizon, affine(terminal_cost, -1, 1)))
        return mul(goal_reward, gate)

    bounded_initial = maximum([
        checkpoint_reward, feasible_goal(constant(0))
    ])

    # Store horizons zero through five, without storing action maps.
    saved = scope
    body = []
    scope = body
    shapes["bounded_value"] = ("s", 4)
    shapes["bounded_index"] = ("v", 1)
    for i in range(1, 7):
        shapes["history_" + str(i)] = ("s", 4)
    bounded_index_next = affine("bounded_index", 1, 1)
    bounded_next = maximum(
        candidates("bounded_value")
        + [checkpoint_reward, feasible_goal(bounded_index_next)]
    )
    scope = saved
    states = [
        {"id": "bounded_value", "init": bounded_initial},
        {"id": "bounded_index", "zeros": [1]},
    ] + [
        {"id": "history_" + str(i), "zeros": [4, 64, 64]}
        for i in range(1, 7)
    ]
    next_states = {
        "bounded_value": bounded_next,
        "bounded_index": bounded_index_next,
        "history_1": "bounded_value",
    }
    for i in range(2, 7):
        next_states["history_" + str(i)] = "history_" + str(i - 1)
    graph.append({
        "id": "bounded", "op": "loop", "T": 6,
        "state": states, "body": body, "next": next_states,
    })
    for i in range(1, 7):
        shapes["bounded.history_" + str(i)] = ("s", 4)
    history = cat([
        "bounded.history_" + str(i) for i in range(6, 0, -1)
    ])

    position_plane = conv(start, np.ones((1, 4, 1, 1), np.float32))
    px = gm(mul(position_plane, col))
    py = gm(mul(position_plane, row))
    pscale = dense(
        gm(start), np.arange(1, 5, dtype=np.float32)[None, :]
    )
    xyz = cat([px, py, pscale])
    old_run = sl("mem", 0, 1)
    old_phase = sl("mem", 1, 2)
    unchanged = clip(affine(vsum(absolute(add(
        xyz, affine(sl("mem", 2, 5), -1)
    ))), -256, 1))
    failed = mul(old_run, unchanged)
    cycle = add(
        affine(old_phase, 1, 1),
        affine(relu(affine(old_phase, 1, -1)), -3),
    )
    phase = add(
        mul(old_phase, affine(old_run, -1, 1)),
        mul(cycle, failed),
    )

    horizon_repeat = np.zeros((24, 6), np.float32)
    horizon_reduce = np.zeros((4, 24, 1, 1), np.float32)
    position_repeat = np.zeros((36, 4, 1, 1), np.float32)
    score_reduce = np.zeros((9, 36), np.float32)
    for h in range(6):
        for s in range(4):
            horizon_repeat[4 * h + s, h] = 1
            horizon_reduce[s, 4 * h + s, 0, 0] = 1
    for a in range(9):
        for s in range(4):
            position_repeat[4 * a + s, s, 0, 0] = 1
            score_reduce[a, 4 * a + s] = 1

    pair_difference = np.zeros((81, 9), np.float32)
    aggregate_losses = np.zeros((9, 81), np.float32)
    for a in range(9):
        for b in range(9):
            pair_difference[9 * a + b, b] += 1
            pair_difference[9 * a + b, a] -= 1
            aggregate_losses[a, 9 * a + b] = 1

    bit_matrix = np.zeros((6, 9), np.float32)
    rotation_bits = np.zeros((6, 4), np.float32)
    colour_bits = np.zeros((6, 3), np.float32)
    for a, code in enumerate(codes):
        for bit in range(6):
            bit_matrix[bit, a] = (code >> bit) & 1
    for a, code in enumerate([0, 5, 7, 6]):
        for bit in range(6):
            rotation_bits[bit, a] = (code >> bit) & 1
    for a, code in enumerate([14, 15, 63]):
        for bit in range(6):
            colour_bits[bit, a] = (code >> bit) & 1

    repeat_bits = np.zeros((36, 6), np.float32)
    repeat_slot = np.zeros_like(repeat_bits)
    for slot in range(6):
        for bit in range(6):
            repeat_bits[6 * slot + bit, bit] = 1
            repeat_slot[6 * slot + bit, slot] = 1

    initial_flags = cat([rotate_needed, recolour_needed])
    saved = scope
    body = []
    scope = body
    shapes["roll_position"] = ("s", 4)
    shapes["roll_program"] = ("v", 36)
    shapes["roll_index"] = ("v", 1)
    shapes["roll_flags"] = ("v", 2)

    remaining = add(slots, affine("roll_index", -1))
    horizon = centres(remaining, np.arange(1, 7))
    selected_value = conv(
        mul(history, dense(horizon, horizon_repeat)), horizon_reduce
    )
    qmaps = cat(candidates(selected_value))
    scores = dense(gm(mul(
        qmaps, conv("roll_position", position_repeat)
    )), score_reduce)
    ranked = dense(
        scores, np.eye(9, dtype=np.float32) * 16,
        [0, 3, 2, 4, 5, 7, 6, 1, 8],
    )
    losses = dense(
        relu(dense(ranked, pair_difference)), aggregate_losses
    )
    winner = clip(affine(losses, -1, 1))
    has_route = clip(affine(vsum(mul(winner, scores)), 2))
    at_goal = clip(vsum(gm(mul("roll_position", goal))))
    active = mul(
        clip(add("roll_index", affine(phase, -1, 1))),
        clip(remaining),
    )
    move_gate = mul(active, mul(affine(at_goal, -1, 1), has_route))
    move_choice = mul(winner, move_gate)
    rf = sl("roll_flags", 0, 1)
    cf = sl("roll_flags", 1, 2)
    rotate_now = mul(active, mul(at_goal, rf))
    colour_now = mul(
        active, mul(at_goal, mul(affine(rf, -1, 1), cf))
    )
    instruction = add(
        dense(move_choice, bit_matrix),
        add(
            mul(dense(rotation_delta, rotation_bits), rotate_now),
            mul(dense(colour_choice, colour_bits), colour_now),
        ),
    )
    selected_slot = centres("roll_index", np.arange(6))
    program_next = add(
        "roll_program",
        mul(dense(instruction, repeat_bits),
            dense(selected_slot, repeat_slot)),
    )

    moves = [
        shift("roll_position", 4, 0),
        shift("roll_position", -4, 0),
        shift("roll_position", 0, -4),
        shift("roll_position", 0, 4),
    ]
    moves += [
        shift(moves[1], -4, 0),
        shift(moves[0], 4, 0),
        conv("roll_position", scale_kernel(True, True)),
        conv("roll_position", scale_kernel(False, True)),
    ]
    position_next = summation([
        mul(x, sl(move_choice, i, i + 1))
        for i, x in enumerate(moves, 1)
    ] + [
        mul("roll_position",
            affine(vsum(sl(move_choice, 1, 9)), -1, 1))
    ])
    flags_next = cat([
        add(rf, affine(rotate_now, -1)),
        add(cf, affine(colour_now, -1)),
    ])
    index_next = affine("roll_index", 1, 1)
    scope = saved
    graph.append({
        "id": "roll", "op": "loop", "T": 6,
        "state": [
            {"id": "roll_position", "init": start},
            {"id": "roll_program", "zeros": [36]},
            {"id": "roll_index", "zeros": [1]},
            {"id": "roll_flags", "init": initial_flags},
        ],
        "body": body,
        "next": {
            "roll_position": position_next,
            "roll_program": program_next,
            "roll_index": index_next,
            "roll_flags": flags_next,
        },
    })
    shapes["roll.roll_program"] = ("v", 36)

    cw = np.zeros((6, 3, 1, 1), np.float32)
    rw = np.zeros_like(cw)
    cw[:, 1, 0, 0] = 63
    rw[:, 0, 0, 0] = 63
    column_offsets = conv(
        "coords", cw, -5 * np.arange(6, dtype=np.float32)
    )
    row_offsets = conv(
        "coords", rw, -3 * np.arange(6, dtype=np.float32)
    )
    column_masks = clip(affine(absolute(add(
        column_offsets, affine(xmin, -1)
    )), -4, 2))
    row_masks = clip(affine(absolute(add(
        row_offsets, affine(ymin, -1)
    )), -4, 2))

    columns36 = np.zeros((36, 6, 1, 1), np.float32)
    rows36 = np.zeros_like(columns36)
    strokes36 = np.zeros((36, 2, 1, 1), np.float32)
    for slot in range(6):
        for bit in range(6):
            i = 6 * slot + bit
            columns36[i, slot, 0, 0] = 1
            rows36[i, bit, 0, 0] = 1
            strokes36[i, bit % 2, 0, 0] = 1
    locations = mul(
        mul(conv(column_masks, columns36), conv(row_masks, rows36)),
        conv(cat([horizontal, vertical]), strokes36),
    )
    desired = add(blank, "roll.roll_program")
    mismatch = mul(
        absolute(node("sub", [desired, sl("x0", 5, 6)])), locations
    )
    mismatch_map = conv(
        mismatch, np.ones((1, 36, 1, 1), np.float32)
    )
    ready = clip(affine(gm(mismatch_map), -1, 1))

    run_w = np.zeros((1, 16, 3, 3), np.float32)
    run_w[0, 9] = 1
    run_map = mul(relu(conv("x0", run_w, [-8])), ge(row, 52))
    node("max2", [affine(mismatch_map, 100), run_map], name="click")

    unrestoreable = mul(
        recolour_needed, centres(target_colour, [11])
    )
    missing_piece = clip(affine(vsum(gm(start)), -1, 1))
    reset_need = add(unrestoreable, missing_piece)
    lw = np.zeros((8, 1), np.float32)
    lw[0, 0] = 3
    lb = np.zeros(8, np.float32)
    lb[6] = 1
    dense(reset_need, lw, lb, name="logits")
    node("vconcat", [ready, phase, px, py, pscale], name="mem_next")

    # Remove unused construction intermediates, not game-dependent data.
    def prune(nodes, roots):
        wanted = set(roots)
        kept = []
        for nd in reversed(nodes):
            nid = nd["id"]
            if nd["op"] == "loop":
                if not any(
                    x == nid or x.startswith(nid + ".") for x in wanted
                ):
                    continue
                new_body, external = prune(
                    nd["body"], nd["next"].values()
                )
                nd = dict(nd)
                nd["body"] = new_body
                state_ids = {s["id"] for s in nd["state"]}
                wanted.update(external - state_ids)
                wanted.update(
                    s["init"] for s in nd["state"] if "init" in s
                )
            else:
                if nid not in wanted:
                    continue
                wanted.discard(nid)
                wanted.update(nd["in"])
            kept.append(nd)
        return list(reversed(kept)), wanted

    graph, _ = prune(graph, ["logits", "click", "mem_next"])
    used = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for field in ("w", "b"):
                    if field in nd:
                        used.add(nd[field])

    collect(graph)
    params = {k: v for k, v in params.items() if k in used}
    return {"graph": graph, "memory_size": 5, "params": params}

import numpy as np


def build():
    graph = []
    params = {}
    dims = {"mem": 36, "avail": 8}
    serial = [0]
    pserial = [0]

    def name():
        serial[0] += 1
        return "n%d" % serial[0]

    def param(a):
        pserial[0] += 1
        k = "p%d" % pserial[0]
        params[k] = np.asarray(a, dtype=np.float32)
        return k

    def node(op, inputs, **fields):
        n = name()
        graph.append({"id": n, "op": op, "in": list(inputs), **fields})
        return n

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        n = node("vslice", [x], start=start, stop=stop)
        dims[n] = stop - start
        return n

    def vcat(xs):
        if len(xs) == 1:
            return xs[0]
        n = node("vconcat", xs)
        dims[n] = sum(dims[x] for x in xs)
        return n

    def dense(x, w, b=None, out_id=None):
        w = np.asarray(w, dtype=np.float32)
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(np.asarray(b, dtype=np.float32))
        if out_id is None:
            n = node("dense", [x], **fields)
        else:
            n = out_id
            graph.append({"id": n, "op": "dense", "in": [x], **fields})
        dims[n] = w.shape[0]
        return n

    def linear(xs, weights, bias=0.0):
        return dense(vcat(xs), [weights], [bias])

    def const(v):
        return dense("avail", np.zeros((1, 8), np.float32), [v])

    def vrelu(x):
        n = node("vrelu", [x])
        dims[n] = dims[x]
        return n

    def vadd(a, b):
        n = node("vadd", [a, b])
        dims[n] = max(dims[a], dims[b])
        return n

    def vmul(a, b):
        n = node("vmul", [a, b])
        dims[n] = max(dims[a], dims[b])
        return n

    def inv(a):
        return linear([a], [-1.0], 1.0)

    def cap(a):
        p = vrelu(a)
        q = vrelu(linear([a], [1.0], -1.0))
        return linear([p, q], [1.0, -1.0])

    def logical_or(xs):
        return cap(linear(xs, [1.0] * len(xs)))

    def logical_and(xs):
        return vrelu(linear(xs, [1.0] * len(xs), 1.0 - len(xs)))

    def choose(flag, yes, no):
        return vadd(vmul(flag, yes), vmul(inv(flag), no))

    def conv(x, w, b=None):
        fields = {"w": param(np.asarray(w, dtype=np.float32))}
        if b is not None:
            fields["b"] = param(np.asarray(b, dtype=np.float32))
        return node("conv", [x], **fields)

    def affine1(x, weight, bias=0.0):
        return conv(x, np.array([[[[weight]]]], np.float32), [bias])

    def spatial(op, a, b):
        return node(op, [a, b])

    def gmax(x, channels):
        n = node("gmax", [x])
        dims[n] = channels
        return n

    def gmean(x, channels):
        n = node("gmean", [x])
        dims[n] = channels
        return n

    def vector_difference_flag(a, b, size):
        w = np.zeros((2 * size, 2 * size), dtype=np.float32)
        for i in range(size):
            w[i, i] = 1.0
            w[i, size + i] = -1.0
            w[size + i, i] = -1.0
            w[size + i, size + i] = 1.0
        differences = vrelu(dense(vcat([a, b]), w))
        magnitude = dense(
            differences,
            np.ones((1, 2 * size), np.float32),
            [-0.25],
        )
        return cap(linear([magnitude], [2.0]))

    one = const(1.0)
    zero = const(0.0)

    ref = vsl("mem", 0, 12)
    prev = vsl("mem", 12, 24)
    prev_score = vsl("mem", 24, 25)
    old_label = vsl("mem", 25, 26)
    was_undo = vsl("mem", 26, 27)
    active = vsl("mem", 27, 28)
    run = vsl("mem", 28, 29)
    phase = vsl("mem", 29, 30)
    blocked = vsl("mem", 30, 31)
    prev_junction = vsl("mem", 31, 33)
    prev_corner = vsl("mem", 33, 34)
    prev_premark = vsl("mem", 34, 35)
    parked = vsl("mem", 35, 36)

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    blue = sl("x0", 8, 9)
    red = sl("x0", 14, 15)

    # Goal movement features.
    block_w = np.zeros((2, 16, 3, 3), dtype=np.float32)
    for out, colour in enumerate((11, 12)):
        for dy in (0, 1):
            for dx in (0, 1):
                block_w[out, colour, 1 + dy, 1 + dx] = 1.0
    blocks = node("relu", [conv("x0", block_w, [-3.5, -3.5])])

    row2 = spatial("mul", row, row)
    col2 = spatial("mul", col, col)
    rowcol = spatial("mul", row, col)
    raw_features = vcat([
        gmean(blocks, 2),
        gmean(spatial("mul", blocks, row), 2),
        gmean(spatial("mul", blocks, col), 2),
        gmean(spatial("mul", blocks, row2), 2),
        gmean(spatial("mul", blocks, col2), 2),
        gmean(spatial("mul", blocks, rowcol), 2),
    ])
    feature_w = np.diag(np.array([
        8192.0, 8192.0,
        516096.0, 516096.0,
        516096.0, 516096.0,
        32514048.0, 32514048.0,
        32514048.0, 32514048.0,
        32514048.0, 32514048.0,
    ], dtype=np.float32))
    features = dense(raw_features, feature_w)

    scale_w = np.zeros((2, 16, 3, 3), dtype=np.float32)
    scale_w[0, 11, :, :] = 1.0
    scale_w[1, 12, :, :] = 1.0
    scale_patches = node(
        "relu", [conv("x0", scale_w, [-8.5, -8.5])]
    )
    scale_max = gmax(scale_patches, 2)
    is_scale2 = cap(dense(scale_max, [[2.0, 2.0]]))

    def match_count(scale):
        w = np.zeros((2, 16, 7, 7), dtype=np.float32)
        if scale == 1:
            corners = [(-1, -1), (-1, 2), (2, -1), (2, 2)]
        else:
            corners = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
        for out, colour in enumerate((11, 12)):
            w[out, colour, 3, 3] = 1.0
            for dy, dx in corners:
                w[out, colour, 3 + dy, 3 + dx] = 1.0
        found = node("relu", [conv("x0", w, [-4.5, -4.5])])
        means = gmean(found, 2)
        factor = 8192.0 if scale == 1 else 2048.0
        return dense(means, [[factor, factor]])

    score1 = match_count(1)
    score2 = match_count(2)
    score = choose(is_scale2, score2, score1)

    # Tile anchors at each camera scale.
    colours = (1, 2, 9, 10, 15, 11, 12)
    w1 = np.zeros((7, 16, 3, 3), dtype=np.float32)
    for i, colour in enumerate(colours):
        for dy in (0, 1):
            for dx in (0, 1):
                w1[i, colour, 1 + dy, 1 + dx] += 1.0
        w1[i, colour, 0, 1] -= 1.0
        w1[i, colour, 1, 0] -= 1.0
    anchors1 = node("relu", [conv("x0", w1, [-3.5] * 7)])
    anchors1 = conv(
        anchors1,
        (2.0 * np.eye(7, dtype=np.float32)).reshape(7, 7, 1, 1),
    )

    w2 = np.zeros((7, 16, 5, 5), dtype=np.float32)
    for i, colour in enumerate(colours):
        for dy in range(3):
            for dx in range(3):
                w2[i, colour, 2 + dy, 2 + dx] += 1.0
        w2[i, colour, 1, 2] -= 1.0
        w2[i, colour, 2, 1] -= 1.0
    anchors2 = node("relu", [conv("x0", w2, [-8.5] * 7)])
    anchors2 = conv(
        anchors2,
        (2.0 * np.eye(7, dtype=np.float32)).reshape(7, 7, 1, 1),
    )

    tile1 = conv(anchors1, np.ones((1, 7, 1, 1), np.float32))
    tile2 = conv(anchors2, np.ones((1, 7, 1, 1), np.float32))
    goals1 = sl(anchors1, 5, 7)
    goals2 = sl(anchors2, 5, 7)

    # Four-way diagonal junction detector.
    diagonal_offsets = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    diagonal_w = np.zeros((4, 1, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(diagonal_offsets):
        diagonal_w[i, 0, 3 + dy, 3 + dx] = 1.0

    diag1 = conv(tile1, diagonal_w)
    halfway = conv(tile2, diagonal_w)
    diagonal_w2 = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(diagonal_offsets):
        diagonal_w2[i, i, 3 + dy, 3 + dx] = 1.0
    diag2 = conv(halfway, diagonal_w2)

    junction_w = np.ones((1, 4, 1, 1), dtype=np.float32)
    j1 = node("relu", [conv(diag1, junction_w, [-3.5])])
    j2 = node("relu", [conv(diag2, junction_w, [-3.5])])
    j1 = spatial("min2", affine1(j1, 2.0), one)
    j2 = spatial("min2", affine1(j2, 2.0), one)

    occupied_j1 = gmax(spatial("mul", goals1, j1), 2)
    occupied_j2 = gmax(spatial("mul", goals2, j2), 2)
    junction_now = choose(is_scale2, occupied_j2, occupied_j1)
    rising_w = np.array([
        [2.0, 0.0, -2.0, 0.0],
        [0.0, 2.0, 0.0, -2.0],
    ], dtype=np.float32)
    newly_at_junction = vrelu(
        dense(vcat([junction_now, prev_junction]), rising_w, [-1.0, -1.0])
    )
    junction_event = cap(dense(newly_at_junction, [[1.0, 1.0]]))

    # A rotated blue control has a four-pixel horizontal run at scale 1.
    rotated_w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for dx in (-1, 0, 1, 2):
        rotated_w[0, 0, 2, 2 + dx] = 1.0
    rotated_map = node("relu", [conv(blue, rotated_w, [-3.5])])
    rotated_present = cap(linear([gmax(rotated_map, 1)], [2.0]))
    special = logical_and([rotated_present, inv(is_scale2)])

    # Extended L junction: two tile steps in each of two perpendicular
    # directions. This distinguishes a transfer corner from a 2x2 loop.
    directions = [(-3, 0), (0, 3), (3, 0), (0, -3)]
    near_w = np.zeros((4, 1, 7, 7), dtype=np.float32)
    far_w = np.zeros((4, 4, 7, 7), dtype=np.float32)
    for i, (dy, dx) in enumerate(directions):
        near_w[i, 0, 3 + dy, 3 + dx] = 1.0
        far_w[i, i, 3 + dy, 3 + dx] = 1.0
    near = conv(tile1, near_w)
    far = conv(near, far_w)
    arms = spatial("mul", near, far)
    corner_w = np.zeros((4, 4, 1, 1), dtype=np.float32)
    for i in range(4):
        corner_w[i, i, 0, 0] = 1.0
        corner_w[i, (i + 1) % 4, 0, 0] = 1.0
    corner_pairs = node("relu", [conv(arms, corner_w, [-1.5] * 4)])
    corner_map = conv(
        corner_pairs, np.full((1, 4, 1, 1), 2.0, np.float32)
    )
    corner_map = spatial("min2", corner_map, one)
    corner_goals = gmax(spatial("mul", goals1, corner_map), 2)
    corner_now = cap(dense(corner_goals, [[1.0, 1.0]]))
    corner_rise = cap(linear([corner_now, prev_corner], [2.0, -2.0], -1.0))
    corner_fall = cap(linear([prev_corner, corner_now], [2.0, -2.0], -1.0))
    corner_event = logical_or([corner_rise, corner_fall])

    # Detect a goal directly below a matching marker.
    marker_w = np.zeros((2, 16, 7, 7), dtype=np.float32)
    for i, colour in enumerate((11, 12)):
        for dy, dx in [(-1, -1), (-1, 2), (2, -1), (2, 2)]:
            marker_w[i, colour, 3 + dy, 3 + dx] = 1.0
    markers = node("relu", [conv("x0", marker_w, [-3.5, -3.5])])
    markers = conv(
        markers,
        (2.0 * np.eye(2, dtype=np.float32)).reshape(2, 2, 1, 1),
    )
    above_w = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for i in range(2):
        above_w[i, i, 0, 3] = 1.0
    marker_above = conv(markers, above_w)
    premark_goals = gmax(spatial("mul", goals1, marker_above), 2)
    premark_now = cap(dense(premark_goals, [[1.0, 1.0]]))
    premark_rise = cap(
        linear([premark_now, prev_premark], [2.0, -2.0], -1.0)
    )

    # A falling corner event means the goal has been transferred off
    # the long track; a rising event means it has returned.
    parked_after_fall = choose(corner_fall, one, parked)
    new_parked = choose(corner_rise, zero, parked_after_fall)

    changed = vector_difference_flag(features, prev, 12)
    differs_from_ref = vector_difference_flag(features, ref, 12)
    score_delta = linear([score, prev_score], [1.0, -1.0])
    rise = cap(linear([score_delta], [2.0], -1.0))
    fall = cap(linear([score_delta], [-2.0], -1.0))

    actual_fall = logical_and([active, inv(was_undo), fall])
    normal_drop = logical_and([actual_fall, inv(blocked)])
    drop = vmul(normal_drop, inv(special))

    timeout = cap(linear([run, phase], [2.0, -2.0], -7.0))
    normal_switch = logical_or([
        rise, inv(changed), inv(differs_from_ref), timeout, junction_event
    ])
    long_timeout = cap(linear([run], [1.0], -13.0))
    special_switch = logical_or([
        inv(changed),
        corner_event,
        logical_and([parked, premark_rise]),
        long_timeout,
    ])
    ordinary_switch = choose(special, special_switch, normal_switch)
    switch = logical_and([
        active,
        logical_or([
            was_undo,
            logical_and([inv(drop), ordinary_switch]),
        ]),
    ])
    start_control = logical_or([inv(active), switch])

    # Label each connected blue component by its first row-major pixel.
    priority = conv(
        "coords",
        np.array([[[[-4032.0]], [[-63.0]], [[0.0]]]], np.float32),
        [4096.0],
    )
    seed = spatial("mul", priority, blue)

    shift_w = np.zeros((9, 1, 3, 3), dtype=np.float32)
    k = 0
    for dy in range(3):
        for dx in range(3):
            shift_w[k, 0, dy, dx] = 1.0
            k += 1
    shift_param = param(shift_w)

    loop_id = name()
    state_id = name()
    shifted_id = name()
    body = [{
        "id": shifted_id,
        "op": "conv",
        "in": [state_id],
        "w": shift_param,
    }]
    pieces = []
    for i in range(9):
        s = name()
        body.append({
            "id": s,
            "op": "slice",
            "in": [shifted_id],
            "start": i,
            "stop": i + 1,
        })
        pieces.append(s)
    merged = pieces[0]
    for piece in pieces[1:]:
        n = name()
        body.append({"id": n, "op": "max2", "in": [merged, piece]})
        merged = n
    masked = name()
    body.append({"id": masked, "op": "mul", "in": [merged, blue]})
    graph.append({
        "id": loop_id,
        "op": "loop",
        "T": 24,
        "state": [{"id": state_id, "init": seed}],
        "body": body,
        "next": {state_id: masked},
    })
    labels = loop_id + "." + state_id

    below = spatial("sub", old_label, labels)
    below = node("relu", [affine1(below, 2.0, -1.0)])
    below = spatial("min2", below, one)
    candidate = gmax(spatial("mul", labels, below), 1)
    highest = gmax(labels, 1)
    has_candidate = cap(candidate)
    next_label = vadd(candidate, vmul(inv(has_candidate), highest))
    selected = choose(start_control, next_label, old_label)

    seed_difference = node("abs", [spatial("sub", seed, labels)])
    anchor = node("relu", [affine1(seed_difference, -1.0, 1.0)])
    anchor = spatial("mul", anchor, blue)
    label_difference = node("abs", [spatial("sub", labels, selected)])
    selected_mask = node(
        "relu", [affine1(label_difference, -1.0, 1.0)]
    )
    blue_click = spatial("mul", anchor, selected_mask)
    blue_row = gmax(spatial("mul", blue_click, row), 1)
    blue_col = gmax(spatial("mul", blue_click, col), 1)

    valid_row = node("relu", [affine1(row, 63.0, -1.0)])
    valid_col = node("relu", [affine1(col, 63.0, 0.0)])
    valid_row = spatial("min2", valid_row, one)
    valid_col = spatial("min2", valid_col, one)
    red_mask = spatial("mul", spatial("mul", red, valid_row), valid_col)

    dr = node("abs", [spatial("sub", row, blue_row)])
    dc = node("abs", [spatial("sub", col, blue_col)])
    distance = spatial("add", dr, dc)
    red_preference = affine1(distance, -1.0, 3.0)
    red_click = spatial("mul", red_mask, red_preference)

    blue_part = spatial("mul", blue_click, inv(drop))
    red_part = spatial("mul", red_click, drop)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [blue_part, red_part],
    })

    new_ref = choose(start_control, features, ref)
    incremented_run = linear([run], [1.0], 1.0)
    new_run = choose(start_control, one, incremented_run)

    phase_plus_one = linear([phase], [1.0], 1.0)
    phase_wrap = cap(linear([phase_plus_one], [1.0], -4.0))
    next_phase = linear([phase_plus_one, phase_wrap], [1.0, -5.0])
    new_phase = choose(switch, next_phase, phase)

    after_fall = choose(drop, one, zero)
    new_blocked = choose(actual_fall, after_fall, blocked)
    genuine_rise = logical_and([rise, inv(was_undo)])
    new_blocked = choose(genuine_rise, zero, new_blocked)

    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [
            new_ref, features, score, selected, drop, one,
            new_run, new_phase, new_blocked, junction_now,
            corner_now, premark_now, new_parked,
        ],
    })

    logits_bias = np.zeros(8, dtype=np.float32)
    logits_bias[6] = 1.0
    dense(
        "avail",
        np.zeros((8, 8), dtype=np.float32),
        logits_bias,
        out_id="logits",
    )

    return {"graph": graph, "memory_size": 36, "params": params}

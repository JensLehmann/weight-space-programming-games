import numpy as np


def build():
    graph = []
    params = {}
    parameter_cache = {}
    shift_cache = {}
    serial = [0]
    current = [graph]

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, ins, name=None, **kw):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        d = {"id": name, "op": op, "in": list(ins)}
        d.update(kw)
        current[0].append(d)
        return name

    def conv(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("conv", [x], name=name, **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": par(w)}
        if b is not None:
            kw["b"] = par(b)
        return node("dense", [x], name=name, **kw)

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b, name)

    def affine(x, c, a=1.0, b=0.0):
        return lin(
            x,
            np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32),
        )

    def va(x, n, a=1.0, b=0.0):
        return dense(
            x,
            np.eye(n, dtype=np.float32) * a,
            np.full(n, b, np.float32),
        )

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(xs):
        return node("concat", xs)

    def vcat(xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def vmax(a, b):
        return node("max2", [a, b])

    def vadd(a, b):
        return node("vadd", [a, b])

    def vmul(a, b):
        return node("vmul", [a, b])

    def relu(x):
        return node("relu", [x])

    def vr(x):
        return node("vrelu", [x])

    def pool(x):
        return node("gmax", [x])

    one = dense("avail", np.zeros((1, 8)), [1])
    zero = dense("avail", np.zeros((1, 8)), [0])

    def complement(x):
        return va(x, 1, -1, 1)

    def cap(x):
        return node("min2", [x, one])

    def indicator_vector(x, weights, bias):
        w = np.asarray(weights, np.float32).reshape(1, -1)
        z = dense(
            x,
            np.concatenate([w, w], axis=0),
            [bias, bias - 1],
        )
        return dense(vr(z), [[1, -1]])

    def shift(x, c, dy, dx, chunk=3):
        key = (x, c, int(dy), int(dx), chunk)
        if key in shift_cache:
            return shift_cache[key]
        y = x
        ry, rx = int(dy), int(dx)
        while ry or rx:
            sy = max(-chunk, min(chunk, ry))
            sx = max(-chunk, min(chunk, rx))
            radius = max(abs(sy), abs(sx))
            k = 2 * radius + 1
            w = np.zeros((c, c, k, k), np.float32)
            for i in range(c):
                w[i, i, radius + sy, radius + sx] = 1
            y = conv(y, w)
            ry -= sy
            rx -= sx
        shift_cache[key] = y
        return y

    def sum2(x, c, s, axis):
        dy, dx = (s, 0) if axis == 0 else (0, s)
        return add(x, shift(x, c, dy, dx))

    def sum4(x, c, s, axis):
        z = sum2(x, c, s, axis)
        dy, dx = (2 * s, 0) if axis == 0 else (0, 2 * s)
        return add(z, shift(z, c, dy, dx)), z

    def sum6(x, c, s, axis):
        a, b = sum4(x, c, s, axis)
        dy, dx = (4 * s, 0) if axis == 0 else (0, 4 * s)
        return add(a, shift(b, c, dy, dx))

    # Spatial coordinate gates.
    w = np.zeros((3, 3), np.float32)
    w[0, 0] = 126
    w[1, 0] = -126
    w[2, 1] = 126
    crop_parts = cap(relu(lin("coords", w, [-5, 83, -25])))
    crop = relu(lin(crop_parts, [[1, 1, 1]], [-2]))

    lower = cap(relu(lin("coords", [[126, 0, 0]], [-61])))
    bottom = cap(relu(lin("coords", [[126, 0, 0]], [-99])))

    # Foreground colours and obstacles.
    colours = [11, 9, 8, 15]
    w = np.zeros((2, 16), np.float32)
    w[0, colours] = 1
    w[1, 6] = 1
    foreground = lin("x0", w)

    colour_planes = cat([sl("x0", c, c + 1) for c in colours])
    black = sl("x0", 5, 6)

    S = 4
    player_rot_maps = []
    goal_rot_maps = []
    checkpoint_maps = []
    valid_maps = []
    player_samples = []
    goal_samples = []

    for s in range(1, S + 1):
        # Sum over an s-by-s elementary sprite pixel.
        radius = s - 1
        k = 2 * radius + 1
        w = np.zeros((2, 2, k, k), np.float32)
        for c in range(2):
            w[c, c, radius:radius + s, radius:radius + s] = 1
        q = conv(foreground, w)

        h4_both, h2_both = sum4(q, 2, s, 1)
        v4_both, v2_both = sum4(q, 2, s, 0)
        full4_both, _ = sum4(h4_both, 2, s, 0)

        qc = sl(q, 0, 1)
        h4 = sl(h4_both, 0, 1)
        h2 = sl(h2_both, 0, 1)
        v4 = sl(v4_both, 0, 1)
        v2 = sl(v2_both, 0, 1)
        full4 = sl(full4_both, 0, 1)
        obstacle_count = sl(full4_both, 1, 2)
        valid_maps.append(relu(affine(obstacle_count, 1, -1, 1)))

        # Player notch / goal peg rectangles.
        pegs = [
            shift(h2, 1, 3 * s, s),
            shift(v2, 1, s, 0),
            shift(h2, 1, 0, s),
            shift(v2, 1, s, 3 * s),
        ]

        w = np.zeros((4, 5), np.float32)
        w[:, 0] = 2
        for r in range(4):
            w[r, r + 1] = -4
        p = relu(lin(
            cat([full4] + pegs), w,
            np.full(4, 1 - 28 * s * s),
        ))
        p = mul(p, crop)
        player_rot_maps.append(p)

        # Goal: two walls, end bar, and inner peg.
        h6 = add(h4, shift(h2, 1, 0, 4 * s))
        v6 = add(v4, shift(v2, 1, 4 * s, 0))
        full6 = shift(sum6(h6, 1, s, 0), 1, -s, -s)

        pieces = [
            shift(v4, 1, 0, -s),
            shift(v4, 1, 0, 4 * s),
            shift(h4, 1, -s, 0),
            shift(h4, 1, 4 * s, 0),
            shift(h6, 1, 4 * s, -s),
            shift(v6, 1, -s, -s),
            shift(h6, 1, -s, -s),
            shift(v6, 1, -s, 4 * s),
        ] + pegs + [full6]

        w = np.zeros((4, 13), np.float32)
        for r, indices in enumerate([
            [0, 1, 4, 8],
            [2, 3, 5, 9],
            [0, 1, 6, 10],
            [2, 3, 7, 11],
        ]):
            w[r, indices] = 4
            w[r, 12] = -2
        g = relu(lin(
            cat(pieces), w,
            np.full(4, 1 - 32 * s * s),
        ))
        g = mul(g, crop)
        goal_rot_maps.append(g)

        # Separable checkerboard checkpoint detector.
        a = sub(qc, shift(qc, 1, 0, s))
        a = add(a, shift(a, 1, 0, 2 * s))
        a = sub(a, shift(a, 1, s, 0))
        a = add(a, shift(a, 1, 2 * s, 0))
        cp = relu(affine(a, 1, 2, 1 - 16 * s * s))
        checkpoint_maps.append(mul(cp, crop))

        ps = lin(p, [[1, 1, 1, 1]])
        player_samples.append(shift(ps, 1, -s, -s))
        peg_offsets = [
            (3 * s, s), (s, 0), (0, s), (s, 3 * s),
        ]
        for r, (dy, dx) in enumerate(peg_offsets):
            goal_samples.append(
                shift(sl(g, r, r + 1), 1, -dy, -dx)
            )

    player_all_rot = cat(player_rot_maps)
    goal_all_rot = cat(goal_rot_maps)

    group = np.zeros((S, 4 * S), np.float32)
    rot_group = np.zeros((4, 4 * S), np.float32)
    for s in range(S):
        group[s, 4 * s:4 * s + 4] = 1
        for r in range(4):
            rot_group[r, 4 * s + r] = 1

    player_map = lin(player_all_rot, group)
    goal_map = lin(goal_all_rot, group)
    checkpoints = cat(checkpoint_maps)
    valid = cat(valid_maps)
    player_rotation = dense(pool(player_all_rot), rot_group)
    goal_rotation = dense(pool(goal_all_rot), rot_group)

    ps = lin(cat(player_samples), np.ones((1, S), np.float32))
    gs = lin(cat(goal_samples), np.ones((1, 4 * S), np.float32))
    player_colour = pool(mul(colour_planes, ps))
    goal_colour = pool(mul(colour_planes, gs))

    # Relative orientation decoded to command bits.
    repeat_p = np.zeros((16, 4), np.float32)
    repeat_g = np.zeros((16, 4), np.float32)
    for p in range(4):
        for g in range(4):
            repeat_p[4 * p + g, p] = 1
            repeat_g[4 * p + g, g] = 1
    pairs = vmul(
        dense(player_rotation, repeat_p),
        dense(goal_rotation, repeat_g),
    )
    rw = np.zeros((7, 16), np.float32)
    rotation_codes = [0, 5, 7, 6]
    for p in range(4):
        for g in range(4):
            code = rotation_codes[(g - p) % 4]
            rw[0, 4 * p + g] = float(p != g)
            for bit in range(6):
                rw[bit + 1, 4 * p + g] = (code >> bit) & 1
    rotation_info = dense(pairs, rw)
    rotation_needed = vs(rotation_info, 0, 1)
    rotation_bits = vs(rotation_info, 1, 7)

    same_colour = vmul(player_colour, goal_colour)
    colour_needed = dense(same_colour, -np.ones((1, 4)), [1])
    colour_codes = [0, 14, 15, 63]
    cw = np.array([
        [(code >> bit) & 1 for code in colour_codes]
        for bit in range(6)
    ], np.float32)
    colour_bits = dense(goal_colour, cw)
    K = vadd(rotation_needed, colour_needed)

    k_ge_1 = indicator_vector(K, [1], 0)
    k_ge_2 = indicator_vector(K, [1], -1)
    k_zero = complement(k_ge_1)
    k_eq_1 = dense(vcat([k_ge_1, k_ge_2]), [[1, -1]])

    # Local bit detectors: line colour 1 or 5, surrounded by colour 0.
    w = np.zeros((2, 16, 3, 3), np.float32)
    for r in range(3):
        for c in range(3):
            for out, on_line in [(0, r == 1), (1, c == 1)]:
                if on_line:
                    w[out, 1, r, c] = 2
                    w[out, 5, r, c] = 2
                else:
                    w[out, 0, r, c] = 2
    raw_markers = mul(relu(conv("x0", w, [-17, -17])), lower)
    raw_horizontal = sl(raw_markers, 0, 1)
    raw_vertical = sl(raw_markers, 1, 2)

    # Require complete horizontal/vertical pairs. Decorative bracket
    # edges can match a single line, but not this paired structure.
    horizontal = mul(
        raw_horizontal, shift(raw_vertical, 1, 3, 0)
    )
    vertical = mul(
        raw_vertical, shift(raw_horizontal, 1, -3, 0)
    )
    marker_map = add(horizontal, vertical)

    inverse_xy = lin(
        "coords",
        [[-63, 0, 0], [0, -63, 0]],
        [64, 64],
    )
    minima = pool(mul(horizontal, inverse_xy))
    geom_base = dense(
        minima,
        [[-1, 0], [0, -1], [0, 0]],
        [64, 64, 0],
    )
    N = pool(node("rowsum", [horizontal]))
    geometry = vadd(geom_base, dense(N, [[0], [0], [1]]))
    active = {
        t: indicator_vector(N, [1], 1 - t)
        for t in range(1, 7)
    }

    # Initial pose: row, column, scale.
    player_union = lin(player_map, np.ones((1, S), np.float32))
    xy = pool(mul(player_union, "coords"))
    initial_xy = dense(
        xy,
        [[63, 0, 0], [0, 63, 0], [0, 0, 0]],
    )
    sw = np.zeros((3, S), np.float32)
    sw[2] = np.arange(1, S + 1)
    pose = vadd(initial_xy, dense(pool(player_map), sw))
    has_player = pool(player_union)

    up_scale = np.zeros((S, S), np.float32)
    down_scale = np.zeros((S, S), np.float32)
    for i in range(S - 1):
        up_scale[i, i + 1] = 1
        down_scale[i + 1, i] = 1

    def bellman(value, retain_candidates=False):
        left = shift(value, S, 0, -4, chunk=2)
        right = shift(value, S, 0, 4, chunk=2)
        down = shift(value, S, 4, 0, chunk=2)
        up = shift(value, S, -4, 0, chunk=2)
        grow = lin(value, up_scale)
        shrink = lin(value, down_scale)
        right8 = shift(right, S, 0, 4, chunk=2)
        left8 = shift(left, S, 0, -4, chunk=2)
        candidates = [
            value, left, right, down, up,
            grow, shrink, right8, left8,
        ]
        best = candidates[1]
        for a in candidates[2:]:
            best = vmax(best, a)
        best = vmax(value, affine(best, S, 1, -1))
        best = mul(best, valid)
        packed = cat(candidates) if retain_candidates else None
        return best, packed

    goal_value = mul(goal_map, va(K, 1, -1, 100))

    # Bounded propagation through program-length segments.
    body = []
    current[0] = body
    phase_k = vcat(["phase_state", K])
    before_goal = indicator_vector(phase_k, [1, -1], 1)
    after_goal = indicator_vector(phase_k, [1, -1], 2)
    seed = vmax("value_state", mul(goal_value, before_goal))
    propagated, _ = bellman(seed)
    propagated = vmax(propagated, mul(goal_value, after_goal))
    checkpoint_next = vmax(
        "checkpoint_state", mul(propagated, checkpoints)
    )

    phase_n = vcat(["phase_state", N])
    boundary = indicator_vector(phase_n, [1, -1], 2)
    not_boundary = complement(boundary)
    next_value = add(
        mul(propagated, not_boundary),
        mul(checkpoint_next, boundary),
    )
    next_phase = vmul(va("phase_state", 1, 1, 1), not_boundary)
    current[0] = graph
    graph.append({
        "id": "planner",
        "op": "loop",
        "T": 48,
        "state": [
            {"id": "value_state", "zeros": [S, 64, 64]},
            {"id": "checkpoint_state", "zeros": [S, 64, 64]},
            {"id": "phase_state", "zeros": [1]},
        ],
        "body": body,
        "next": {
            "value_state": next_value,
            "checkpoint_state": checkpoint_next,
            "phase_state": next_phase,
        },
    })

    # Remaining-horizon tables and checkpoint progress preference.
    endpoint_value = affine("planner.checkpoint_state", S, 2, 0)
    final_goal_value = affine(goal_value, S, 2, 0)
    f = vmax(endpoint_value, mul(final_goal_value, k_zero))
    packed_candidates = {}
    for t in range(1, 7):
        f, packed_candidates[t] = bellman(f, True)
        if t == 1:
            permitted = complement(k_ge_2)
            f = vmax(f, mul(final_goal_value, permitted))
        else:
            f = vmax(f, final_goal_value)

    xy_grid = lin("coords", [[63, 0, 0], [0, 63, 0]])

    def pose_map(p):
        neg_xy = dense(p, [[-1, 0, 0], [0, -1, 0]])
        delta = node("abs", [add(xy_grid, neg_xy)])
        xy_margin = lin(delta, [[-1, -1]], [0.5])

        w = np.zeros((2 * S, 3), np.float32)
        b = np.zeros(2 * S, np.float32)
        for i in range(S):
            w[2 * i, 2] = 1
            b[2 * i] = -(i + 1)
            w[2 * i + 1, 2] = -1
            b[2 * i + 1] = i + 1
        deviations = vr(dense(p, w, b))
        w = np.zeros((S, 2 * S), np.float32)
        for i in range(S):
            w[i, 2 * i:2 * i + 2] = -1
        scale_margin = dense(deviations, w)
        return cap(affine(
            relu(add(xy_margin, scale_margin)), S, 4, 0
        ))

    A = 9
    codes = [0, 1, 2, 3, 33, 8, 9, 10, 12]
    bit_weights = np.array([
        [(code >> bit) & 1 for code in codes]
        for bit in range(6)
    ], np.float32)
    motion = np.array([
        [0, 0, 0, 4, -4, 0, 0, 0, 0],
        [0, -4, 4, 0, 0, 0, 0, 8, -8],
        [0, 0, 0, 0, 0, 1, -1, 0, 0],
    ], np.float32)
    gather = np.zeros((A, A * S), np.float32)
    for a in range(A):
        gather[a, a * S:(a + 1) * S] = 1

    # Finite argmax implemented with pairwise ReLU comparisons.
    differences = np.zeros((A * A, A), np.float32)
    reduce_differences = np.zeros((A, A * A), np.float32)
    for i in range(A):
        for j in range(A):
            differences[i * A + j, j] += 1
            differences[i * A + j, i] -= 1
            reduce_differences[i, i * A + j] = -1
    preference = np.arange(A, 0, -1, dtype=np.float32)
    preference[1:] -= 32

    programs = []
    for t in range(6, 0, -1):
        pm = pose_map(pose)
        repeated_pose = cat([pm] * A)
        values = dense(
            pool(mul(packed_candidates[t], repeated_pose)),
            gather,
        )
        scores = dense(
            values, np.eye(A, dtype=np.float32) * 32, preference
        )
        violations = vr(dense(scores, differences))
        chosen = vr(dense(
            violations, reduce_differences,
            np.ones(A, np.float32),
        ))

        if t <= 2:
            on_goal = pool(mul(pm, goal_map))
            on_goal = dense(on_goal, np.ones((1, S), np.float32))
            required = k_ge_1 if t == 1 else k_ge_2
            align = vmul(vmul(on_goal, required), active[t])
            move_gate = vmul(active[t], complement(align))
        else:
            align = zero
            move_gate = active[t]

        selected = vmul(chosen, move_gate)
        bits = dense(selected, bit_weights)
        pose = vadd(pose, dense(selected, motion))

        if t == 2:
            bits = vadd(bits, vmul(rotation_bits, align))
        elif t == 1:
            rf = vmul(vmul(align, rotation_needed), k_eq_1)
            cf = vmul(align, colour_needed)
            bits = vadd(bits, vmul(rotation_bits, rf))
            bits = vadd(bits, vmul(colour_bits, cf))
        programs.append(bits)

    desired_bits = vcat(programs)

    # Trace time t corresponds to program column N-t.
    w = np.zeros((72, 3), np.float32)
    for i in range(36):
        w[2 * i, 0] = 63
        w[2 * i + 1, 1] = 63
    coordinate_copies = lin("coords", w)

    w = np.zeros((72, 3), np.float32)
    b = np.zeros(72, np.float32)
    for j, t in enumerate(range(6, 0, -1)):
        for bit in range(6):
            i = 6 * j + bit
            w[2 * i, 0] = -1
            b[2 * i] = -3 * bit
            w[2 * i + 1, 1] = -1
            w[2 * i + 1, 2] = -5
            b[2 * i + 1] = 5 * t
    centres = dense(geometry, w, b)
    distances = node("abs", [add(coordinate_copies, centres)])
    w = np.zeros((36, 72), np.float32)
    for i in range(36):
        w[i, 2 * i:2 * i + 2] = -1
    slots = cap(affine(
        relu(lin(distances, w, np.full(36, 0.5))),
        36, 4, 0,
    ))

    desired_map = lin(
        mul(slots, desired_bits),
        np.ones((1, 36), np.float32),
    )
    slot_union = cap(lin(slots, np.ones((1, 36), np.float32)))
    clickable_bits = mul(marker_map, slot_union)
    mismatch = mul(
        node("abs", [sub(desired_map, black)]),
        clickable_bits,
    )

    # Round run button: solid seven-by-seven centre.
    w = np.zeros((1, 16, 7, 7), np.float32)
    w[0, 9] = 2
    run_button = mul(relu(conv("x0", w, [-97])), bottom)
    click_features = cat([mismatch, run_button])
    lin(click_features, [[100, 1]], name="click")

    # RESET only when no controllable sprite is detected.
    w = np.zeros((8, 1), np.float32)
    b = np.full(8, -10, np.float32)
    w[0, 0] = -3
    b[0] = 2
    b[6] = 1
    dense(has_player, w, b, name="logits")

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

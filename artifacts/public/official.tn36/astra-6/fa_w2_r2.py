import numpy as np


def build():
    P = {}
    graph = []
    ctx = graph
    shapes = {
        "x0": (True, 16),
        "coords": (True, 3),
        "mem": (False, 6),
        "aprev": (False, 8),
        "avail": (False, 8),
    }
    serial = 0
    pserial = 0
    shift_weights = {}
    shift_cache = {}

    def fresh():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def param(a):
        nonlocal pserial
        pserial += 1
        name = "p%d" % pserial
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins, name=None, **kw):
        ident = name or fresh()
        nd = {"id": ident, "op": op, "in": list(ins)}
        nd.update(kw)
        ctx.append(nd)

        if op == "conv":
            shapes[ident] = (True, P[kw["w"]].shape[0])
        elif op == "dense":
            shapes[ident] = (False, P[kw["w"]].shape[0])
        elif op in ("gmax", "gmean"):
            shapes[ident] = (False, shapes[ins[0]][1])
        elif op in ("concat", "vconcat"):
            shapes[ident] = (
                op == "concat",
                sum(shapes[x][1] for x in ins),
            )
        elif op in ("slice", "vslice"):
            shapes[ident] = (
                op == "slice", kw["stop"] - kw["start"]
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[ident] = (
                True, max(shapes[x][1] for x in ins)
            )
        elif op in ("vadd", "vmul"):
            shapes[ident] = (
                False, max(shapes[x][1] for x in ins)
            )
        else:
            shapes[ident] = shapes[ins[0]]
        return ident

    def conv(x, w, b=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return emit("conv", [x], **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": param(w)}
        if b is not None:
            kw["b"] = param(b)
        return emit("dense", [x], name=name, **kw)

    def cat(xs):
        return emit("concat", xs)

    def vcat(xs):
        return emit("vconcat", xs)

    def slc(x, a, b):
        op = "slice" if shapes[x][0] else "vslice"
        return emit(op, [x], start=a, stop=b)

    def unary(op, x):
        return emit(op, [x])

    def binary(op, x, y, name=None):
        return emit(op, [x, y], name=name)

    def slinear(x, a=1.0, b=0.0):
        c = shapes[x][1]
        w = (np.eye(c, dtype=np.float32) * a)[:, :, None, None]
        return conv(x, w, np.full(c, b, np.float32))

    def vlinear(x, a=1.0, b=0.0):
        c = shapes[x][1]
        return dense(
            x, np.eye(c, dtype=np.float32) * a,
            np.full(c, b, np.float32)
        )

    def const(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense(
            "avail", np.zeros((len(values), 8), np.float32), values
        )

    def clip(x):
        # Exact saturation to [0, 1].
        if shapes[x][0]:
            a = unary("relu", slinear(x, -1, 1))
            return unary("relu", slinear(a, -1, 1))
        a = unary("vrelu", vlinear(x, -1, 1))
        return unary("vrelu", vlinear(a, -1, 1))

    def vabs(x):
        a = unary("vrelu", x)
        b = unary("vrelu", vlinear(x, -1, 0))
        return binary("vadd", a, b)

    def veq(x, centers):
        centers = np.asarray(centers, np.float32)
        d = dense(
            x, np.ones((len(centers), 1), np.float32), -centers
        )
        # Flat unit plateau around each integer, robust to coordinate rounding.
        return clip(vlinear(vabs(d), -4, 2))

    def vsum(xs):
        if len(xs) == 1:
            return xs[0]
        z = vcat(xs)
        return dense(z, np.ones((1, shapes[z][1]), np.float32))

    def max_fields(xs):
        y = xs[0]
        for x in xs[1:]:
            y = binary("max2", y, x)
        return y

    def add_fields(xs):
        y = xs[0]
        for x in xs[1:]:
            y = binary("add", y, x)
        return y

    def chunks(n):
        n = abs(int(n))
        q, r = divmod(n, 3)
        if r == 1 and q:
            return [3] * (q - 1) + [2, 2]
        return [3] * q + ([r] if r else [])

    def shift(x, dx, dy):
        # Output at (y,x) reads the input at (y+dy,x+dx).
        nonlocal shift_cache
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return x
        key = (x, dx, dy)
        if key in shift_cache:
            return shift_cache[key]

        cx, cy = chunks(dx), chunks(dy)
        n = max(len(cx), len(cy))
        cx += [0] * (n - len(cx))
        cy += [0] * (n - len(cy))
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        y = x
        c = shapes[x][1]
        for px, py in zip(cx, cy):
            px, py = px * sx, py * sy
            k = 2 * max(abs(px), abs(py)) + 1
            wk = (c, px, py)
            if wk not in shift_weights:
                w = np.zeros((c, c, k, k), np.float32)
                for i in range(c):
                    w[i, i, k // 2 + py, k // 2 + px] = 1
                shift_weights[wk] = param(w)
            y = emit("conv", [y], w=shift_weights[wk])
        shift_cache[key] = y
        return y

    def make_loop(T, inits, fn):
        nonlocal ctx, shift_cache
        parent, parent_cache = ctx, shift_cache
        state_ids = [fresh() for _ in inits]
        for sid, init in zip(state_ids, inits):
            shapes[sid] = shapes[init]

        body = []
        ctx = body
        shift_cache = {}
        next_ids = fn(state_ids)
        ctx = parent
        shift_cache = parent_cache

        lid = fresh()
        ctx.append({
            "id": lid,
            "op": "loop",
            "T": T,
            "state": [
                {"id": sid, "init": init}
                for sid, init in zip(state_ids, inits)
            ],
            "body": body,
            "next": dict(zip(state_ids, next_ids)),
        })
        outputs = []
        for sid in state_ids:
            name = lid + "." + sid
            shapes[name] = shapes[sid]
            outputs.append(name)
        return outputs

    def plane(weights):
        w = np.zeros((1, 16, 1, 1), np.float32)
        for color, value in weights.items():
            w[0, color, 0, 0] = value
        return conv("x0", w)

    # Coordinates in pixel units.
    row = conv(
        "coords", np.array([[[[63]], [[0]], [[0]]]], np.float32)
    )
    col = conv(
        "coords", np.array([[[[0]], [[63]], [[0]]]], np.float32)
    )

    zero_plane = slc("x0", 0, 1)
    black_plane = slc("x0", 5, 6)
    toggle_ink = plane({1: 1, 5: 1})
    ink = plane({8: 1, 9: 1, 11: 1, 14: 1, 15: 1})
    color_value = plane({9: 1, 8: 2, 15: 3})
    obstacle = slc("x0", 6, 7)

    # Detect the centers of both kinds of program-bit glyph.
    ui_input = cat([toggle_ink, zero_plane])
    uw = np.zeros((2, 2, 5, 5), np.float32)
    for dx in (-1, 0, 1):
        uw[0, 0, 2, 2 + dx] = 1
    for dy in (-1, 1):
        for dx in range(-2, 3):
            uw[0, 1, 2 + dy, 2 + dx] = 1
    for dx in (-2, 2):
        uw[0, 1, 2, 2 + dx] = 1

    for dy in (-1, 0, 1):
        uw[1, 0, 2 + dy, 2] = 1
        for dx in (-2, -1, 1, 2):
            uw[1, 1, 2 + dy, 2 + dx] = 1
    for dy in (-2, 2):
        uw[1, 1, 2 + dy, 2] = 1

    ui = unary("relu", conv(ui_input, uw, [-14, -16]))
    centers = conv(ui, np.ones((1, 2, 1, 1), np.float32))

    def minimum_coordinate(coord):
        inv = slinear(coord, -1, 64)
        m = unary("gmax", binary("mul", centers, inv))
        return vlinear(m, -1, 64)

    left = minimum_coordinate(col)
    top = minimum_coordinate(row)
    right = unary("gmax", binary("mul", centers, col))
    bottom = unary("gmax", binary("mul", centers, row))

    n_float = dense(vcat([left, right]), [[-0.2, 0.2]], [1])
    b_float = dense(
        vcat([top, bottom]), [[-1.0 / 3, 1.0 / 3]], [1]
    )
    n_flags = veq(n_float, range(1, 7))
    n_steps = dense(
        n_flags, np.arange(1, 7, dtype=np.float32)[None, :]
    )

    # Candidate command order: left, right, down, up, left*2,
    # right*2, grow, shrink.
    command_bits = [1, 2, 2, 6, 4, 4, 4, 4]
    legal_vec = clip(dense(
        b_float, np.full((8, 1), 2, np.float32),
        1 - 2 * np.asarray(command_bits, np.float32)
    ))
    legal = [slc(legal_vec, i, i + 1) for i in range(8)]

    # Object origins must lie above the programming panel.
    region = clip(slinear(binary("sub", row, top), -1, -2))

    def partitions(lo, count):
        """Minimum number of seven-pixel tiles; short first tile."""
        number = (count + 6) // 7
        first = count - 7 * (number - 1)
        lengths = [first] + [7] * (number - 1)
        ans = []
        for length in lengths:
            hi = lo + length - 1
            low_center, high_center = hi - 3, lo + 3
            center = min(max(0, low_center), high_center)
            ans.append((lo, hi, center))
            lo = hi + 1
        return ans

    def templates(patterns, scale, origin_offset):
        """Exact signed-template correlation, tiled into legal kernels."""
        mats = [
            np.kron(p, np.ones((scale, scale), np.float32))
            for p in patterns
        ]
        signed = [2 * m - 1 for m in mats]
        n = mats[0].shape[0]
        lo = origin_offset
        tiles = partitions(lo, n)
        parts = []
        for yl, yh, yc in tiles:
            for xl, xh, xc in tiles:
                radius = max(
                    abs(yl - yc), abs(yh - yc),
                    abs(xl - xc), abs(xh - xc)
                )
                k = 2 * radius + 1
                w = np.zeros((len(mats), 1, k, k), np.float32)
                for o, m in enumerate(signed):
                    for yy in range(yl, yh + 1):
                        for xx in range(xl, xh + 1):
                            w[o, 0, yy - yc + radius, xx - xc + radius] = (
                                m[yy - lo, xx - lo]
                            )
                z = conv(ink, w)
                parts.append(shift(z, xc, yc))
        total = add_fields(parts)
        count = np.asarray([m.sum() for m in mats], np.float32)
        z = conv(
            total,
            np.eye(len(mats), dtype=np.float32)[:, :, None, None],
            1 - count,
        )
        return binary("mul", unary("relu", z), region)

    pawn_pattern = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ], np.float32)
    checkpoint_pattern = np.fromfunction(
        lambda y, x: ((x + y) % 2 == 0), (4, 4)
    ).astype(np.float32)
    goal_pattern = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ], np.float32)

    pawn_patterns = [
        np.rot90(pawn_pattern, -r).copy() for r in range(4)
    ] + [checkpoint_pattern]
    goal_patterns = [
        np.rot90(goal_pattern, -r).copy() for r in range(4)
    ]

    pawns, checkpoints, goals = [], [], []
    pawn_rotations, goal_rotations = [], []
    pawn_colors = []

    for s in range(1, 5):
        pd = templates(pawn_patterns, s, 0)
        gd = templates(goal_patterns, s, -s)

        pm = conv(
            pd, np.array([1, 1, 1, 1, 0], np.float32)[None, :, None, None]
        )
        gm = conv(gd, np.ones((1, 4, 1, 1), np.float32))
        pawns.append(pm)
        goals.append(gm)
        checkpoints.append(slc(pd, 4, 5))

        pr = conv(
            pd, np.array([0, 1, 2, 3, 0], np.float32)[None, :, None, None]
        )
        gr = conv(
            gd, np.arange(4, dtype=np.float32)[None, :, None, None]
        )
        pawn_rotations.append(unary("gmax", pr))
        goal_rotations.append(unary("gmax", gr))
        sampled_color = shift(color_value, s, s)
        pawn_colors.append(
            unary("gmax", binary("mul", pm, sampled_color))
        )

    actor_position = cat(pawns)
    checkpoint_map = cat(checkpoints)
    pawn_rotation = vsum(pawn_rotations)
    raw_goal_rotation = vsum(goal_rotations)
    pawn_color = vsum(pawn_colors)

    def square_sums(x, base):
        """Top-left-anchored sums for base, 2*base, 3*base, 4*base."""
        k = 7
        w = np.zeros((1, 1, k, k), np.float32)
        if base == 4:
            w[0, 0, 3:7, 3:7] = 1
            b1 = conv(x, w)
        else:
            # Six samples, offsets -2..3, then re-anchor at the top-left.
            w[0, 0, 1:7, 1:7] = 1
            b1 = shift(conv(x, w), 2, 2)

        h2 = binary("add", b1, shift(b1, base, 0))
        b2 = binary("add", h2, shift(h2, 0, base))
        h3 = binary("add", h2, shift(b1, 2 * base, 0))
        b3 = add_fields([
            h3, shift(h3, 0, base), shift(h3, 0, 2 * base)
        ])
        h4 = binary("add", b2, shift(b2, 2 * base, 0))
        b4 = binary("add", h4, shift(h4, 0, 2 * base))
        return [b1, b2, b3, b4]

    color_boxes = square_sums(color_value, 6)
    goal_colors = []
    for s, gm, box in zip(range(1, 5), goals, color_boxes):
        z = binary("mul", gm, shift(box, -s, -s))
        z = unary("gmax", z)
        goal_colors.append(vlinear(z, 1.0 / (16 * s * s), 0))
    raw_goal_color = vsum(goal_colors)

    # Six memory entries: goal x, y, scale, rotation, colour, validity.
    raw_goal_map = max_fields(goals)
    goal_valid = unary("gmax", raw_goal_map)
    goal_x = unary("gmax", binary("mul", raw_goal_map, col))
    goal_y = unary("gmax", binary("mul", raw_goal_map, row))
    goal_scale_flags = vcat([unary("gmax", g) for g in goals])
    goal_scale = dense(
        goal_scale_flags, np.arange(1, 5, dtype=np.float32)[None, :]
    )
    observed_goal = vcat([
        goal_x, goal_y, goal_scale, raw_goal_rotation,
        raw_goal_color, goal_valid
    ])
    old_gate = vlinear(goal_valid, -1, 1)
    new_memory = binary("vmul", observed_goal, goal_valid)
    old_memory = binary("vmul", "mem", old_gate)
    memory = binary("vadd", new_memory, old_memory, name="mem_next")

    remembered_xy = slc(memory, 0, 2)
    remembered_scale = slc(memory, 2, 3)
    remembered_rotation = slc(memory, 3, 4)
    remembered_color = slc(memory, 4, 5)

    xy_error = unary(
        "abs", binary("sub", cat([col, row]), remembered_xy)
    )
    exact_position = clip(conv(
        xy_error, np.full((1, 2, 1, 1), -4, np.float32), [2]
    ))
    exact_scale = veq(remembered_scale, [1, 2, 3, 4])
    goal_value = slinear(
        binary("mul", exact_position, exact_scale), 1000, 0
    )

    occupancy = cat(square_sums(obstacle, 4))
    free = unary("relu", slinear(occupancy, -1, 1))
    goal_value = binary("mul", goal_value, free)

    # Rotation and colour can be corrected before the positional program.
    rotation_difference = dense(
        vcat([remembered_rotation, pawn_rotation]), [[1, -1]]
    )
    rotation_cases = veq(rotation_difference, range(-3, 4))
    rotation_codes = [5, 7, 6, 0, 5, 7, 6]

    def bits_matrix(codes):
        return np.asarray([
            [(int(code) >> bit) & 1 for code in codes]
            for bit in range(6)
        ], np.float32)

    rotation_program = dense(
        rotation_cases, bits_matrix(rotation_codes)
    )
    needs_rotation = dense(
        rotation_cases, [[1, 1, 1, 0, 1, 1, 1]]
    )

    color_difference = dense(
        vcat([remembered_color, pawn_color]), [[1, -1]]
    )
    same_color = veq(color_difference, [0])
    needs_color = vlinear(same_color, -1, 1)
    goal_color_cases = veq(remembered_color, [0, 1, 2, 3])
    color_program = dense(
        goal_color_cases, bits_matrix([0, 14, 15, 63])
    )
    movement_budget = dense(
        vcat([n_steps, needs_rotation, needs_color]), [[1, -1, -1]]
    )
    movement_budget = unary("vrelu", movement_budget)

    grow_w = np.zeros((4, 4, 1, 1), np.float32)
    shrink_w = np.zeros_like(grow_w)
    for i in range(3):
        grow_w[i, i + 1, 0, 0] = 1
        shrink_w[i + 1, i, 0, 0] = 1

    def candidates(value, cost=True, use_legality=True):
        if cost:
            value = unary("relu", slinear(value, 1, -1))
        ss = [slc(value, i, i + 1) for i in range(4)]
        direction_parts = [[] for _ in range(6)]
        for z in ss:
            l = shift(z, -4, 0)
            r = shift(z, 4, 0)
            d = shift(z, 0, 4)
            u = shift(z, 0, -4)
            ll = shift(l, -4, 0)
            rr = shift(r, 4, 0)
            for bucket, field in zip(
                direction_parts, [l, r, d, u, ll, rr]
            ):
                bucket.append(field)
        result = [cat(xs) for xs in direction_parts]
        result += [conv(value, grow_w), conv(value, shrink_w)]
        if use_legality:
            result = [
                binary("mul", x, gate)
                for x, gate in zip(result, legal)
            ]
        return result

    def bellman(value, seed):
        z = max_fields(candidates(value))
        z = binary("max2", z, seed)
        return binary("mul", z, free)

    # Checkpoint-level dynamic programming. The last six iterations are a
    # frozen-seed finite-horizon pass; history states retain horizons 0..6.
    clock_start = const([1, 0, 0, 0, 0, 0])
    timer_start = const([1] + [0] * 54)
    clock_shift = np.zeros((6, 6), np.float32)
    timer_shift = np.zeros((55, 55), np.float32)
    for i in range(5):
        clock_shift[i + 1, i] = 1
    for i in range(54):
        timer_shift[i + 1, i] = 1
    freeze_w = np.zeros((1, 55), np.float32)
    freeze_w[0, :48] = 1
    force_w = np.zeros((1, 55), np.float32)
    force_w[0, 47] = 1

    def planning_body(st):
        value, seed, clock, timer = st[:4]
        normal_end = dense(
            binary("vmul", clock, n_flags),
            np.ones((1, 6), np.float32)
        )
        not_end = vlinear(normal_end, -1, 1)
        training_phase = dense(timer, freeze_w)
        final_reset = dense(timer, force_w)
        boundary = binary(
            "vadd",
            binary("vmul", normal_end, training_phase),
            binary("vmul", final_reset, not_end),
        )
        not_boundary = vlinear(boundary, -1, 1)

        advanced = bellman(value, seed)
        cp_reward = binary(
            "mul", checkpoint_map,
            unary("relu", slinear(advanced, 1, -10))
        )
        improved_seed = max_fields([seed, goal_value, cp_reward])
        next_seed = binary(
            "add",
            binary("mul", improved_seed, boundary),
            binary("mul", seed, not_boundary),
        )
        next_value = binary(
            "add",
            binary("mul", next_seed, boundary),
            binary("mul", advanced, not_boundary),
        )

        shifted_clock = dense(clock, clock_shift)
        next_clock = binary(
            "vadd",
            binary("vmul", shifted_clock, not_end),
            binary("vmul", clock_start, normal_end),
        )
        next_timer = dense(timer, timer_shift)
        return [
            next_value, next_seed, next_clock, next_timer,
            value, st[4], st[5], st[6], st[7], st[8]
        ]

    planned = make_loop(
        54,
        [goal_value, goal_value, clock_start, timer_start]
        + [goal_value] * 6,
        planning_body,
    )
    terminal = planned[1]
    # H0, H1, ..., H5.
    horizons = cat([planned[i] for i in range(9, 3, -1)])

    horizon_repeat = np.zeros((24, 6), np.float32)
    horizon_sum = np.zeros((4, 24, 1, 1), np.float32)
    for h in range(6):
        for s in range(4):
            horizon_repeat[4 * h + s, h] = 1
            horizon_sum[s, 4 * h + s, 0, 0] = 1

    # Deterministic winner circuit with an exact 1/16 tie-breaking margin.
    pair_w = np.zeros((81, 9), np.float32)
    pair_b = np.zeros(81, np.float32)
    winner_w = np.zeros((9, 81), np.float32)
    for i in range(9):
        for j in range(9):
            k = 9 * i + j
            pair_w[k, j] += 1
            pair_w[k, i] -= 1
            pair_b[k] = (i - j) / 16.0
            winner_w[i, k] = -16

    score_sum = np.zeros((9, 36), np.float32)
    choice_repeat = np.zeros((36, 9), np.float32)
    position_sum = np.zeros((4, 36, 1, 1), np.float32)
    for a in range(9):
        for s in range(4):
            score_sum[a, 4 * a + s] = 1
            choice_repeat[4 * a + s, a] = 1
            position_sum[s, 4 * a + s, 0, 0] = 1

    instruction_codes = [0, 1, 2, 3, 33, 12, 10, 8, 9]
    instruction_bits = bits_matrix(instruction_codes)
    stop_choice = const([1] + [0] * 8)
    zero_scalar = const([0])
    zero_bits = const([0] * 6)

    def extraction_body(st):
        position, remaining, rot_flag, col_flag = st[:4]

        h_flags = veq(remaining, range(1, 7))
        selected_h = binary(
            "mul", horizons, dense(h_flags, horizon_repeat)
        )
        selected_h = conv(selected_h, horizon_sum)

        qs = [terminal] + candidates(selected_h)
        q_stack = cat(qs)
        p_stack = cat([position] * 9)
        scores = dense(
            unary("gmax", binary("mul", q_stack, p_stack)),
            score_sum
        )
        comparisons = unary("vrelu", dense(scores, pair_w, pair_b))
        winner = unary(
            "vrelu", dense(comparisons, winner_w, np.ones(9))
        )

        not_rot = vlinear(rot_flag, -1, 1)
        is_color = binary("vmul", col_flag, not_rot)
        is_move = dense(vcat([rot_flag, is_color]), [[-1, -1]], [1])
        not_move = vlinear(is_move, -1, 1)

        emitted = binary(
            "vadd",
            binary("vmul", dense(winner, instruction_bits), is_move),
            binary(
                "vadd",
                binary("vmul", rotation_program, rot_flag),
                binary("vmul", color_program, is_color),
            ),
        )
        actual_choice = binary(
            "vadd",
            binary("vmul", winner, is_move),
            binary("vmul", stop_choice, not_move),
        )

        scatters = candidates(position, cost=False, use_legality=False)
        # Reverse gather directions to obtain forward position transport.
        transported = cat([
            position,
            scatters[1], scatters[0], scatters[3], scatters[2],
            scatters[5], scatters[4], scatters[7], scatters[6],
        ])
        transported = binary(
            "mul", transported, dense(actual_choice, choice_repeat)
        )
        next_position = conv(transported, position_sum)
        next_remaining = unary(
            "vrelu",
            binary("vadd", remaining, vlinear(is_move, -1, 0))
        )
        next_col = binary("vmul", col_flag, rot_flag)

        return [
            next_position, next_remaining, zero_scalar, next_col,
            emitted, st[4], st[5], st[6], st[7], st[8]
        ]

    extracted = make_loop(
        6,
        [actor_position, movement_budget, needs_rotation, needs_color]
        + [zero_bits] * 6,
        extraction_body,
    )
    program = vcat([extracted[i] for i in range(9, 3, -1)])

    # Convert the six instruction words to their spatial bit locations.
    x_relative = binary("sub", col, left)
    y_relative = binary("sub", row, top)
    slot_distance = conv(
        x_relative, np.ones((6, 1, 1, 1), np.float32),
        -5 * np.arange(6, dtype=np.float32)
    )
    bit_distance = conv(
        y_relative, np.ones((6, 1, 1, 1), np.float32),
        -3 * np.arange(6, dtype=np.float32)
    )
    slots = unary("relu", slinear(unary("abs", slot_distance), -1, 1))
    bit_rows = unary("relu", slinear(unary("abs", bit_distance), -1, 1))

    location_w = np.zeros((36, 12, 1, 1), np.float32)
    for slot in range(6):
        for bit in range(6):
            k = 6 * slot + bit
            location_w[k, slot, 0, 0] = 1
            location_w[k, 6 + bit, 0, 0] = 1
    locations = unary("relu", conv(
        cat([slots, bit_rows]), location_w, -np.ones(36, np.float32)
    ))
    desired = conv(
        binary("mul", locations, program),
        np.ones((1, 36, 1, 1), np.float32)
    )
    discrepancy = unary("abs", binary("sub", desired, black_plane))
    discrepancy = unary("relu", slinear(discrepancy, 2, -1))
    toggle_score = slinear(
        binary("mul", discrepancy, centers), 100, 0
    )

    # A filled 5x5 interior distinguishes the run button from hollow tabs.
    run_w = np.zeros((1, 16, 5, 5), np.float32)
    run_w[0, 9] = 1
    run = unary("relu", conv("x0", run_w, [-24]))
    lower_screen = clip(slinear(row, 1, -50))
    run = binary("mul", run, lower_screen)
    binary("add", toggle_score, run, name="click")

    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 1
    dense(
        "avail", np.zeros((8, 8), np.float32),
        logits_bias, name="logits"
    )

    return {
        "graph": graph,
        "memory_size": 6,
        "params": P,
    }

import numpy as np


def build():
    params = {}
    cache = {}
    graph = []
    current = graph
    serial = 0

    def fresh():
        nonlocal serial
        serial += 1
        return "n%d" % serial

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(params)
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, ins=None, **fields):
        name = fresh()
        nd = {"id": name, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(fields)
        current.append(nd)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def affine(x, w, b=None):
        w = np.asarray(w, np.float32)
        return conv(x, w[:, :, None, None], b)

    def constant(v):
        v = np.asarray(v, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(v), 8), np.float32), v)

    def unary(op, x):
        return node(op, [x])

    def binary(op, x, y):
        return node(op, [x, y])

    def cat(xs, vector=False):
        return node("vconcat" if vector else "concat", xs)

    def sl(x, a, b, vector=False):
        return node("vslice" if vector else "slice",
                    [x], start=a, stop=b)

    def pool(x, op="gmax"):
        return node(op, [x])

    one = constant([1])
    zero_v = constant([0])

    def clip(x):
        return binary("min2", unary("relu", x), one)

    def sinv(x, channels):
        return affine(x, -np.eye(channels, dtype=np.float32),
                      np.ones(channels, np.float32))

    def vinv(x, n):
        return dense(x, -np.eye(n, dtype=np.float32),
                     np.ones(n, np.float32))

    def vscale(x, n, a, b=0):
        return dense(x, np.eye(n, dtype=np.float32) * a,
                     np.full(n, b, np.float32))

    def spatial_scale(x, n, a, b=0):
        return affine(x, np.eye(n, dtype=np.float32) * a,
                      np.full(n, b, np.float32))

    def total(xs):
        z = xs[0]
        for x in xs[1:]:
            z = binary("add", z, x)
        return z

    def vtotal(xs):
        z = xs[0]
        for x in xs[1:]:
            z = binary("vadd", z, x)
        return z

    cw = np.zeros((3, 3), np.float32)
    cw[0, 0] = 63
    cw[1, 1] = 63
    cw[2, 2] = 1
    xy = affine("coords", cw)
    row = sl(xy, 0, 1)
    col = sl(xy, 1, 2)
    zero_s = affine("coords", np.zeros((1, 3), np.float32))

    colour_ids = [11, 9, 8, 15]
    colour_w = np.zeros((4, 16), np.float32)
    for i, c in enumerate(colour_ids):
        colour_w[i, c] = 1
    colours = affine("x0", colour_w)
    ink = affine(colours, np.ones((1, 4), np.float32))

    def row_ge(v):
        return clip(spatial_scale(row, 1, 2, 1 - 2 * v))

    def row_le(v):
        return clip(spatial_scale(row, 1, -2, 1 + 2 * v))

    top = row_le(40)
    board_rows = binary("mul", row_ge(3), top)
    bottom = row_ge(51)

    def equality(coord, value):
        d = unary("abs", binary("sub", coord, value))
        return clip(spatial_scale(d, 1, -4, 2))

    def minimum_coordinate(mask, coord):
        weight = spatial_scale(coord, 1, -1, 64)
        maximum = pool(binary("mul", mask, weight))
        return dense(maximum, [[-1]], [64])

    def locate(mask):
        yy = minimum_coordinate(mask, row)
        xx = minimum_coordinate(mask, col)
        point = binary("mul", equality(row, yy), equality(col, xx))
        point = binary("mul", point, mask)
        return point, xx, yy

    gather4_w = np.zeros((9, 1, 5, 5), np.float32)
    sample_offsets = [
        (dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)
    ]
    for i, (dy, dx) in enumerate(sample_offsets):
        gather4_w[i, 0, 2 + dy, 2 + dx] = 1
    gather4 = conv(ink, gather4_w)

    quadrant_shift = np.zeros((4, 4, 7, 7), np.float32)
    quadrant_offsets = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for i, (dy, dx) in enumerate(quadrant_offsets):
        quadrant_shift[i, i, 3 + dy, 3 + dx] = 1

    def template(coeff, scale):
        coeff = np.asarray(coeff, np.float32)
        positives = float(np.sum(coeff > 0))
        if scale <= 3:
            k = 2 * scale + 1
            w = np.zeros((4, 1, k, k), np.float32)
            for q, (qy, qx) in enumerate(quadrant_offsets):
                for iy in range(3):
                    for ix in range(3):
                        w[q, 0,
                          scale + (iy - 1) * scale,
                          scale + (ix - 1) * scale] = coeff[
                              qy + iy, qx + ix]
            z = conv(ink, w)
        else:
            w = np.zeros((4, 9, 5, 5), np.float32)
            for q, (qy, qx) in enumerate(quadrant_offsets):
                for iy in range(3):
                    for ix in range(3):
                        j = iy * 3 + ix
                        dy, dx = sample_offsets[j]
                        w[q, j, 2 + dy, 2 + dx] = coeff[
                            qy + iy, qx + ix]
            z = conv(gather4, w)
        for _ in range(scale):
            z = conv(z, quadrant_shift)
        z = affine(z, np.full((1, 4), 2, np.float32),
                   [1 - 2 * positives])
        return clip(z)

    player_base = np.ones((4, 4), np.float32)
    player_base[3, 1:3] = 0
    goal_base = np.zeros((6, 6), np.float32)
    goal_base[1:5, 0] = 1
    goal_base[1:5, 5] = 1
    goal_base[4, 2:4] = 1
    goal_base[5, :] = 1

    player_maps = []
    goal_maps = []
    checkpoint_maps = []
    goal_patterns = []

    for scale in range(1, 5):
        for rotation in range(4):
            pp = np.zeros((6, 6), np.float32)
            pp[1:5, 1:5] = np.rot90(player_base, -rotation)
            player_maps.append(template(2 * pp - 1, scale))

            gp = np.rot90(goal_base, -rotation).copy()
            goal_patterns.append((scale, gp))
            goal_maps.append(template(2 * gp - 1, scale))

        cp = np.zeros((6, 6), np.float32)
        for y in range(4):
            for x in range(4):
                cp[y + 1, x + 1] = (
                    1 if (x + y) % 2 == 0 else -1
                )
        checkpoint_maps.append(template(cp, scale))

    def pose(maps):
        stack = cat(maps)
        union = clip(affine(stack, np.ones((1, 16), np.float32)))
        union = binary("mul", union, top)
        point, xx, yy = locate(union)
        flags = pool(binary("mul", stack, point))
        ws = np.zeros((4, 16), np.float32)
        wr = np.zeros((4, 16), np.float32)
        for i in range(16):
            ws[i // 4, i] = 1
            wr[i % 4, i] = 1
        sizes = dense(flags, ws)
        rotations = dense(flags, wr)
        return point, xx, yy, sizes, rotations

    player_point, px, py, player_size, player_rotation = pose(player_maps)
    goal_point, gx, gy, goal_size, goal_rotation = pose(goal_maps)

    player_colour = pool(binary("mul", colours, player_point))

    def shift_small(x, channels, dy, dx):
        z = x
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        for t in range(max(abs(dy), abs(dx))):
            yy = sy if t < abs(dy) else 0
            xx = sx if t < abs(dx) else 0
            w = np.zeros((channels, channels, 3, 3), np.float32)
            for c in range(channels):
                w[c, c, 1 + yy, 1 + xx] = 1
            z = conv(z, w)
        return z

    goal_colour_parts = []
    for m, (scale, gp) in zip(goal_maps, goal_patterns):
        iy, ix = np.argwhere(gp > 0)[0]
        sample = shift_small(
            colours, 4,
            (int(iy) - 1) * scale,
            (int(ix) - 1) * scale
        )
        selected = binary("mul", m, goal_point)
        goal_colour_parts.append(pool(binary("mul", sample, selected)))
    goal_colour = vtotal(goal_colour_parts)

    def joint(a, b, n):
        wa = np.zeros((n * n, n), np.float32)
        wb = np.zeros((n * n, n), np.float32)
        for i in range(n):
            for j in range(n):
                wa[i * n + j, i] = 1
                wb[i * n + j, j] = 1
        return binary("vmul", dense(a, wa), dense(b, wb))

    def command_bits(code):
        return [(code >> i) & 1 for i in range(6)]

    rotation_joint = joint(player_rotation, goal_rotation, 4)
    rotation_bits_w = np.zeros((6, 16), np.float32)
    rotation_needed_w = np.zeros((1, 16), np.float32)
    rotation_codes = [0, 5, 7, 16]
    for a in range(4):
        for b in range(4):
            d = (b - a) % 4
            rotation_bits_w[:, a * 4 + b] = command_bits(rotation_codes[d])
            rotation_needed_w[0, a * 4 + b] = int(d != 0)
    rotation_bits = dense(rotation_joint, rotation_bits_w)
    rotation_needed = dense(rotation_joint, rotation_needed_w)

    colour_joint = joint(player_colour, goal_colour, 4)
    colour_bits_w = np.zeros((6, 16), np.float32)
    colour_needed_w = np.zeros((1, 16), np.float32)
    colour_codes = [0, 14, 15, 63]
    for a in range(4):
        for b in range(4):
            if a != b:
                colour_bits_w[:, a * 4 + b] = command_bits(colour_codes[b])
                colour_needed_w[0, a * 4 + b] = 1
    colour_bits = dense(colour_joint, colour_bits_w)
    colour_needed = dense(colour_joint, colour_needed_w)

    # Detect line switches, then require the structural H/V pair.
    ui_w = np.zeros((2, 16, 3, 3), np.float32)
    for y in range(3):
        for x in range(3):
            for orientation in range(2):
                line = (y == 1) if orientation == 0 else (x == 1)
                if line:
                    ui_w[orientation, 1, y, x] = 1
                    ui_w[orientation, 5, y, x] = 1
                else:
                    ui_w[orientation, 0, y, x] = 1
    ui_raw = conv("x0", ui_w, [-8.5, -8.5])
    ui = clip(spatial_scale(ui_raw, 2, 2))
    horizontal_raw = sl(ui, 0, 1)
    vertical_raw = sl(ui, 1, 2)

    horizontal = binary(
        "mul", horizontal_raw,
        shift_small(vertical_raw, 1, 3, 0)
    )
    vertical = binary(
        "mul", vertical_raw,
        shift_small(horizontal, 1, -3, 0)
    )
    ui_all = binary("max2", horizontal, vertical)

    remaining_columns = node("colmax", [horizontal])
    columns = []
    for _ in range(6):
        xx = minimum_coordinate(remaining_columns, col)
        selected = binary("mul", equality(col, xx), remaining_columns)
        columns.append(selected)
        remaining_columns = clip(binary("sub", remaining_columns, selected))
    column_stack = cat(columns)
    column_available = pool(column_stack)

    first_ui_row = minimum_coordinate(horizontal, row)
    bit_maps = []
    for b in range(6):
        yy = dense(first_ui_row, [[1]], [3 * b])
        bit_maps.append(binary("mul", equality(row, yy), ui_all))
    bit_stack = cat(bit_maps)
    bit_available = pool(bit_stack)
    ui_centres = clip(affine(bit_stack, np.ones((1, 6), np.float32)))

    codes = [1, 2, 3, 33, 10, 12, 8, 9, 0]
    costs = np.array([
        16 + int(c).bit_count() if c else 0 for c in codes
    ], np.float32)
    allow_w = np.zeros((9, 6), np.float32)
    for a, bit in enumerate([0, 1, 1, 5, 3, 3, 3, 3]):
        allow_w[a, bit] = 1
    allow_b = np.zeros(9, np.float32)
    allow_b[8] = 1
    allowed = dense(bit_available, allow_w, allow_b)
    repeat_action = np.zeros((36, 9), np.float32)
    for a in range(9):
        repeat_action[4 * a:4 * a + 4, a] = 1
    allowed36 = dense(allowed, repeat_action)
    negative_cost36 = constant(-np.repeat(costs, 4))

    def rectangle(xx, yy, width, offset):
        xstart = binary("vadd", xx, offset)
        ystart = binary("vadd", yy, offset)
        xend = binary("vadd", xstart, width)
        yend = binary("vadd", ystart, width)
        a = clip(spatial_scale(binary("sub", col, xstart), 1, 2, 1))
        b = clip(spatial_scale(binary("sub", xend, col), 1, 2, -1))
        c = clip(spatial_scale(binary("sub", row, ystart), 1, 2, 1))
        d = clip(spatial_scale(binary("sub", yend, row), 1, 2, -1))
        return binary("mul", binary("mul", a, b), binary("mul", c, d))

    size_values = np.arange(1, 5, dtype=np.float32)[None, :]
    ps = dense(player_size, size_values)
    gs = dense(goal_size, size_values)
    player_box = rectangle(px, py, dense(ps, [[4]]), zero_v)
    goal_box = rectangle(gx, gy, dense(gs, [[6]]), dense(gs, [[-1]]))
    excluded = binary("max2", player_box, goal_box)

    hazard_w = np.zeros((1, 16), np.float32)
    hazard_w[0, [8, 9, 15]] = 1
    hazards = affine("x0", hazard_w)
    hazards = binary("mul", hazards, sinv(excluded, 1))
    hazards = binary("mul", hazards, board_rows)

    wall_w = np.zeros((1, 16), np.float32)
    wall_w[0, [6, 12, 13]] = 1
    occupied = clip(binary("add", affine("x0", wall_w), hazards))

    expand_w = np.zeros((1, 1, 3, 3), np.float32)
    expand_w[0, 0, 1:3, 1:3] = 1
    occupancy = occupied
    blocked = []
    for i in range(1, 16):
        occupancy = clip(conv(occupancy, expand_w))
        if i in (3, 7, 11, 15):
            blocked.append(occupancy)
    valid = sinv(cat(blocked), 4)

    initial_position = binary("mul", player_point, player_size)
    goal_position = binary("mul", goal_point, goal_size)
    goal_seed = spatial_scale(goal_position, 4, 10000)
    goal_seed = binary("mul", goal_seed, valid)

    shift_weights = {}

    def shift2(x, dy, dx):
        key = (dy, dx)
        if key not in shift_weights:
            w = np.zeros((4, 4, 5, 5), np.float32)
            for c in range(4):
                w[c, c, 2 + dy, 2 + dx] = 1
            shift_weights[key] = w
        return conv(x, shift_weights[key])

    def transitions(x, forward=False):
        sign = -1 if forward else 1
        l2 = shift2(x, 0, -2 * sign)
        left = shift2(l2, 0, -2 * sign)
        r2 = shift2(x, 0, 2 * sign)
        right = shift2(r2, 0, 2 * sign)
        d2 = shift2(x, 2 * sign, 0)
        down = shift2(d2, 2 * sign, 0)
        u2 = shift2(x, -2 * sign, 0)
        up = shift2(u2, -2 * sign, 0)
        right8 = shift2(
            shift2(right, 0, 2 * sign), 0, 2 * sign
        )
        left8 = shift2(
            shift2(left, 0, -2 * sign), 0, -2 * sign
        )

        grow_w = np.zeros((4, 4), np.float32)
        shrink_w = np.zeros((4, 4), np.float32)
        if not forward:
            for s in range(3):
                grow_w[s, s + 1] = 1
            shrink_w[0, 0] = 1
            for s in range(1, 4):
                shrink_w[s, s - 1] = 1
        else:
            for s in range(1, 4):
                grow_w[s, s - 1] = 1
            shrink_w[0, 0] = 1
            for s in range(3):
                shrink_w[s, s + 1] = 1

        grow = affine(x, grow_w)
        shrink = affine(x, shrink_w)
        return [left, right, down, up, right8, left8, grow, shrink, x]

    def candidates(x):
        z = cat(transitions(x))
        z = unary("relu", binary("add", z, negative_cost36))
        return binary("mul", z, allowed36)

    def bellman(x):
        z = candidates(x)
        best = sl(z, 0, 4)
        for a in range(1, 9):
            best = binary("max2", best, sl(z, 4 * a, 4 * a + 4))
        return binary("mul", best, valid)

    saved = current
    body = []
    current = body
    global_state = fresh()
    global_next = bellman(global_state)
    current = saved
    global_loop = node(
        "loop", T=28,
        state=[{"id": global_state, "init": goal_seed}],
        body=body, next={global_state: global_next}
    )
    global_value = global_loop + "." + global_state

    checkpoints = binary("mul", cat(checkpoint_maps), top)
    checkpoints = binary("mul", checkpoints, valid)
    checkpoint_reward = unary(
        "relu", spatial_scale(global_value, 4, 4, -32000)
    )
    checkpoint_reward = binary("mul", checkpoint_reward, checkpoints)
    seed = binary("max2", goal_seed, checkpoint_reward)

    initial_history = cat([seed] * 7)
    saved = current
    body = []
    current = body
    finite_state = fresh()
    history_state = fresh()
    finite_next = bellman(finite_state)
    history_next = cat([finite_next, sl(history_state, 0, 24)])
    current = saved
    finite_loop = node(
        "loop", T=6,
        state=[
            {"id": finite_state, "init": seed},
            {"id": history_state, "init": initial_history}
        ],
        body=body,
        next={
            finite_state: finite_next,
            history_state: history_next
        }
    )
    history = finite_loop + "." + history_state

    count_inputs = cat(
        [column_available, rotation_needed, colour_needed], vector=True
    )
    count_w = np.ones((1, 8), np.float32)
    count_w[0, 6:] = -1
    remaining_count = dense(count_inputs, count_w)

    hi = dense(remaining_count, np.ones((7, 1), np.float32),
               -np.arange(7, dtype=np.float32))
    lo = dense(remaining_count, -np.ones((7, 1), np.float32),
               np.arange(7, dtype=np.float32))
    distance = binary("vadd", unary("vrelu", hi), unary("vrelu", lo))
    horizon_initial = unary("vrelu", vscale(distance, 7, -1, 1))

    slot_initial = constant([1, 0, 0, 0, 0, 0])
    program_initial = constant(np.zeros(36, np.float32))

    horizon_select_w = np.zeros((28, 7), np.float32)
    for h in range(7):
        d = max(0, h - 1)
        history_index = 6 - d
        horizon_select_w[
            4 * history_index:4 * history_index + 4, h
        ] = 1

    action_sum_w = np.zeros((9, 36), np.float32)
    for a in range(9):
        action_sum_w[a, 4 * a:4 * a + 4] = 1

    tie_w = np.eye(9, dtype=np.float32)
    for i in range(9):
        tie_w[i, :i] = -1

    bits_w = np.array(
        [command_bits(c) for c in codes], np.float32
    ).T

    slot_lift_w = np.zeros((36, 6), np.float32)
    bits_lift_w = np.zeros((36, 6), np.float32)
    for j in range(6):
        for b in range(6):
            slot_lift_w[6 * j + b, j] = 1
            bits_lift_w[6 * j + b, b] = 1

    slot_advance_w = np.zeros((6, 6), np.float32)
    for j in range(5):
        slot_advance_w[j + 1, j] = 1

    horizon_advance_w = np.zeros((7, 7), np.float32)
    horizon_advance_w[0, 0] = 1
    for h in range(1, 7):
        horizon_advance_w[h - 1, h] = 1

    spatial_action_sum_w = np.zeros((4, 36), np.float32)
    for a in range(9):
        for s in range(4):
            spatial_action_sum_w[s, 4 * a + s] = 1

    noop_choice = constant([0, 0, 0, 0, 0, 0, 0, 0, 1])

    saved = current
    body = []
    current = body
    sim_state = fresh()
    horizon_state = fresh()
    slot_state = fresh()
    program_state = fresh()

    horizon_gate = dense(horizon_state, horizon_select_w)
    selected_history = binary("mul", history, horizon_gate)
    history_sum_w = np.zeros((4, 28), np.float32)
    for j in range(7):
        for s in range(4):
            history_sum_w[s, 4 * j + s] = 1
    value = affine(selected_history, history_sum_w)

    qs = candidates(value)
    repeated_position = cat([sim_state] * 9)
    q = dense(
        pool(binary("mul", qs, repeated_position)), action_sum_w
    )

    q_spatial = binary("add", zero_s, q)
    q_max = sl(q_spatial, 0, 1)
    for a in range(1, 9):
        q_max = binary("max2", q_max, sl(q_spatial, a, a + 1))
    maximum_q = pool(q_max)
    negative_max = dense(
        maximum_q, -np.ones((9, 1), np.float32),
        np.ones(9, np.float32)
    )
    tied = unary("vrelu", binary("vadd", q, negative_max))
    choice = unary("vrelu", dense(tied, tie_w))

    h_zero = sl(horizon_state, 0, 1, vector=True)
    h_nonzero = vinv(h_zero, 1)
    choice = binary(
        "vadd",
        binary("vmul", choice, h_nonzero),
        binary("vmul", noop_choice, h_zero)
    )

    slot0 = sl(slot_state, 0, 1, vector=True)
    slot1 = sl(slot_state, 1, 2, vector=True)
    rotation_gate = binary("vmul", slot0, rotation_needed)
    colour_slot = binary(
        "vadd",
        binary("vmul", slot0, vinv(rotation_needed, 1)),
        binary("vmul", slot1, rotation_needed)
    )
    colour_gate = binary("vmul", colour_slot, colour_needed)
    correction_gate = binary("vadd", rotation_gate, colour_gate)
    movement_gate = vinv(correction_gate, 1)

    active = dense(
        binary("vmul", slot_state, column_available),
        np.ones((1, 6), np.float32)
    )
    movement_active = binary("vmul", movement_gate, active)

    move_bits = dense(choice, bits_w)
    selected_bits = vtotal([
        binary("vmul", move_bits, movement_gate),
        binary("vmul", rotation_bits, rotation_gate),
        binary("vmul", colour_bits, colour_gate)
    ])
    selected_bits = binary("vmul", selected_bits, active)
    slot_bits = binary(
        "vmul",
        dense(slot_state, slot_lift_w),
        dense(selected_bits, bits_lift_w)
    )
    program_next = binary("vadd", program_state, slot_bits)

    future_positions = cat(transitions(sim_state, forward=True))
    choice36 = dense(choice, repeat_action)
    moved_position = affine(
        binary("mul", future_positions, choice36),
        spatial_action_sum_w
    )
    sim_next = binary(
        "add",
        binary("mul", moved_position, movement_active),
        binary("mul", sim_state, vinv(movement_active, 1))
    )

    advanced_horizon = dense(horizon_state, horizon_advance_w)
    horizon_next = binary(
        "vadd",
        binary("vmul", advanced_horizon, movement_active),
        binary("vmul", horizon_state, vinv(movement_active, 1))
    )
    slot_next = dense(slot_state, slot_advance_w)

    current = saved
    plan_loop = node(
        "loop", T=6,
        state=[
            {"id": sim_state, "init": initial_position},
            {"id": horizon_state, "init": horizon_initial},
            {"id": slot_state, "init": slot_initial},
            {"id": program_state, "init": program_initial}
        ],
        body=body,
        next={
            sim_state: sim_next,
            horizon_state: horizon_next,
            slot_state: slot_next,
            program_state: program_next
        }
    )
    program = plan_loop + "." + program_state

    desired_parts = []
    for b in range(6):
        select_w = np.zeros((6, 36), np.float32)
        for j in range(6):
            select_w[j, 6 * j + b] = 1
        desired_columns = dense(program, select_w)
        at_columns = binary("mul", column_stack, desired_columns)
        at_columns = affine(
            at_columns, np.ones((1, 6), np.float32)
        )
        desired_parts.append(binary("mul", at_columns, bit_maps[b]))
    desired = total(desired_parts)

    black_w = np.zeros((1, 16), np.float32)
    black_w[0, 5] = 1
    current_bits = binary(
        "mul", affine("x0", black_w), ui_centres
    )
    mismatch = unary("abs", binary("sub", desired, current_bits))
    mismatch = binary("mul", mismatch, ui_centres)

    run_w = np.zeros((1, 16, 5, 5), np.float32)
    run_w[0, 9, :, :] = 2
    run_button = clip(conv("x0", run_w, [-49]))
    run_button = binary("mul", run_button, bottom)

    click_value = binary(
        "add", spatial_scale(mismatch, 1, 100), run_button
    )
    current.append({
        "id": "click", "op": "relu", "in": [click_value]
    })

    logits_bias = np.full(8, -5, np.float32)
    logits_bias[6] = 5
    current.append({
        "id": "logits",
        "op": "dense",
        "in": ["avail"],
        "w": parameter(np.zeros((8, 8), np.float32)),
        "b": parameter(logits_bias)
    })

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params
    }

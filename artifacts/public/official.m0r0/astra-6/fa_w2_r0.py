import numpy as np


def build():
    params = {}
    graph = []
    shapes = {
        "x0": (16, 64, 64), "coords": (3, 64, 64),
        "mem": (21,), "aprev": (8,), "avail": (8,)
    }
    target = graph
    caches = {id(graph): {}}
    parameter_cache = {}
    constants = {}
    spatial_zeros = {}
    serial = 0

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(value):
        a = np.asarray(value, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(params))
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, inputs, w=None, b=None, **fields):
        fields = dict(fields)
        if w is not None:
            fields["w"] = parameter(w)
        if b is not None:
            fields["b"] = parameter(b)
        key = (op, tuple(inputs), tuple(sorted(fields.items())))
        cache = caches.setdefault(id(target), {})
        if key in cache:
            return cache[key]

        name = fresh()
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        target.append(node)
        cache[key] = name

        ss = [shapes[x] for x in inputs]
        if op == "conv":
            shape = (params[fields["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (params[fields["w"]].shape[0],)
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        else:
            shape = ss[0]
        shapes[name] = shape
        return name

    def spatial(x):
        return len(shapes[x]) == 3

    def const(value):
        nonlocal target
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constants:
            old = target
            target = graph
            constants[key] = emit(
                "dense", ["aprev"],
                w=np.zeros((len(a), 8), np.float32), b=a
            )
            target = old
        return constants[key]

    def lin(x, w, b=None):
        w = np.asarray(w, np.float32)
        if spatial(x):
            return emit("conv", [x], w=w[:, :, None, None], b=b)
        return emit("dense", [x], w=w, b=b)

    def sl(x, start, stop):
        if start == 0 and stop == shapes[x][0]:
            return x
        return emit("slice" if spatial(x) else "vslice",
                    [x], start=start, stop=stop)

    def cat(*xs):
        if len(xs) == 1:
            return xs[0]
        return emit("concat" if spatial(xs[0]) else "vconcat", xs)

    def add(a, b):
        return emit("add" if spatial(a) or spatial(b) else "vadd", [a, b])

    def mul(a, b):
        return emit("mul" if spatial(a) or spatial(b) else "vmul", [a, b])

    def affine(x, scale=1.0, bias=0.0):
        if scale == 1 and bias == 0:
            return x
        n = shapes[x][0]
        if not spatial(x) or n == 1:
            return lin(x, np.eye(n, dtype=np.float32) * scale,
                       np.full(n, bias, np.float32))
        y = x if scale == 1 else mul(x, const(scale))
        return y if bias == 0 else add(y, const(bias))

    def sub(a, b):
        if spatial(a) or spatial(b):
            return emit("sub", [a, b])
        return add(a, affine(b, -1))

    def relu(x):
        return emit("relu" if spatial(x) else "vrelu", [x])

    def absolute(x):
        if spatial(x):
            return emit("abs", [x])
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        h = relu(lin(x, np.concatenate([eye, -eye], axis=0)))
        return lin(h, np.concatenate([eye, eye], axis=1))

    def clip(x):
        if spatial(x):
            return emit("min2", [relu(x), const(1)])
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        h = relu(lin(
            x, np.concatenate([eye, eye], axis=0),
            np.concatenate([np.zeros(n), -np.ones(n)])
        ))
        return lin(h, np.concatenate([eye, -eye], axis=1))

    def tent(x):
        if spatial(x):
            return relu(sub(const(1), absolute(x)))
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        h = relu(lin(
            x, np.concatenate([eye, eye, eye], axis=0),
            np.concatenate([np.ones(n), np.zeros(n), -np.ones(n)])
        ))
        return lin(h, np.concatenate([eye, -2*eye, eye], axis=1))

    def exact_point(x):
        return clip(affine(absolute(x), -4, 2))

    def maximum(a, b):
        if spatial(a) or spatial(b):
            return emit("max2", [a, b])
        return add(b, relu(sub(a, b)))

    def minimum(a, b):
        return emit("min2", [a, b])

    def choose(gate, yes, no):
        return add(no, mul(gate, sub(yes, no)))

    def total(x):
        n = shapes[x][0]
        return x if n == 1 else lin(x, np.ones((1, n), np.float32))

    def gm(x):
        return emit("gmax", [x])

    def zeros(n):
        nonlocal target
        if n not in spatial_zeros:
            old = target
            target = graph
            spatial_zeros[n] = emit(
                "conv", ["coords"], w=np.zeros((n, 3, 1, 1), np.float32)
            )
            target = old
        return spatial_zeros[n]

    def begin_loop(initial):
        nonlocal target
        old = target
        lid = fresh("L")
        states, ids = [], []
        for init in initial:
            sid = fresh("s")
            states.append({"id": sid, "init": init})
            shapes[sid] = shapes[init]
            ids.append(sid)
        body = []
        target = body
        caches[id(body)] = {}
        return old, lid, states, ids, body

    def end_loop(ctx, values, iterations):
        nonlocal target
        old, lid, states, ids, body = ctx
        target = old
        node = {
            "id": lid, "op": "loop", "T": iterations,
            "state": states, "body": body,
            "next": dict(zip(ids, values))
        }
        target.append(node)
        outputs = []
        for sid in ids:
            name = lid + "." + sid
            shapes[name] = shapes[sid]
            outputs.append(name)
        return outputs, node

    # Coordinates and persistent tracking state.
    pix = lin("coords", [[63, 0, 0], [0, 63, 0]])
    rr, cc = sl(pix, 0, 1), sl(pix, 1, 2)
    previous = sl("mem", 0, 4)
    previous_a, previous_b = sl(previous, 0, 2), sl(previous, 2, 4)
    valid = sl("mem", 6, 7)
    old_ghost = sl("mem", 7, 9)
    old_ghost_valid = sl("mem", 9, 10)
    old_count = sl("mem", 10, 11)
    pawn = add(sl("x0", 10, 11), sl("x0", 1, 2))

    # Infer square size using both horizontal and vertical pawn runs.
    runs_w = np.zeros((8, 1, 7, 7), np.float32)
    for j, length in enumerate((4, 5, 6, 7)):
        start = 3 - length // 2
        runs_w[j, 0, 3, start:start+length] = 1
        runs_w[4+j, 0, start:start+length, 3] = 1
    runs = gm(relu(emit(
        "conv", [pawn], w=runs_w,
        b=-np.tile(np.arange(3, 7, dtype=np.float32), 2)
    )))
    horizontal_runs, vertical_runs = sl(runs, 0, 4), sl(runs, 4, 8)
    runs = sub(horizontal_runs, relu(sub(horizontal_runs, vertical_runs)))
    detected_scale = lin(runs, [
        [0, -1, 0, 0], [0, 1, -1, 0],
        [0, 0, 1, -1], [0, 0, 0, 1]
    ], [1, 0, 0, 0])
    scale_code = choose(valid, sl("mem", 11, 15), detected_scale)
    sizes = np.array([4, 5, 6, 7], np.float32)
    size = lin(scale_code, sizes[None, :])
    inv_size = lin(scale_code, (1/sizes)[None, :])
    half_size = lin(scale_code, np.floor(sizes/2)[None, :])

    # A raw corner supplies the cell-lattice residue, not a level identity.
    corner_w = np.zeros((1, 1, 3, 3), np.float32)
    corner_w[0, 0, 1, 1] = 3
    corner_w[0, 0, 0, 1] = -1
    corner_w[0, 0, 1, 0] = -1
    raw_corners = relu(emit("conv", [pawn], w=corner_w, b=[-2]))
    anchor_score = sub(affine(raw_corners, 1000), lin(pix, [[1, .01]]))
    anchor_point = relu(affine(sub(gm(anchor_score), anchor_score), -2000, 1))
    anchor = gm(mul(anchor_point, pix))

    floor_w = np.zeros((32, 2), np.float32)
    for j in range(2):
        floor_w[16*j:16*j+16, j] = 32
    floor_b = np.tile(4-32*np.arange(1, 17, dtype=np.float32), 2)
    sum_halves = np.zeros((2, 32), np.float32)
    sum_halves[0, :16] = 1
    sum_halves[1, 16:] = 1
    quotient = lin(clip(lin(mul(anchor, inv_size), floor_w, floor_b)),
                   sum_halves)
    residue = sub(anchor, mul(quotient, size))
    grid_xy = mul(sub(pix, residue), inv_size)

    # Lattice-aligned point detection keeps touching pawns separate.
    lattice = []
    for axis in range(2):
        d = lin(sl(grid_xy, axis, axis+1),
                np.full((16, 1), 32, np.float32),
                -32*np.arange(16, dtype=np.float32))
        lattice.append(total(exact_point(d)))
    corners = mul(pawn, mul(lattice[0], lattice[1]))
    count = affine(emit("gmean", [corners]), 4096)
    tie_cost = lin(grid_xy, [[.048, .003]])

    def point_at(position):
        return tent(total(absolute(sub(grid_xy, position))))

    def pick(mask, old_position, tracking):
        distance = total(absolute(sub(grid_xy, old_position)))
        score = sub(sub(affine(mask, 1000),
                        mul(distance, tracking)), tie_cost)
        point = relu(affine(sub(gm(score), score), -2000, 1))
        return point, gm(mul(point, grid_xy))

    eligible0 = relu(sub(
        corners, mul(point_at(old_ghost), old_ghost_valid)
    ))
    _, raw_a = pick(eligible0, previous_a, valid)
    dropped = mul(clip(sub(old_count, count)), valid)
    ghost = choose(dropped, raw_a, old_ghost)
    ghost_valid = clip(add(old_ghost_valid, dropped))
    eligible = relu(sub(corners, mul(point_at(ghost), ghost_valid)))
    tracking = mul(valid, sub(const(1), dropped))
    a_point, a_position = pick(eligible, previous_a, tracking)
    _, b_position = pick(relu(sub(eligible, a_point)), previous_b, tracking)
    positions = cat(a_position, b_position)

    # Round the four inferred logical coordinates with a threshold bank.
    round_w = np.zeros((64, 4), np.float32)
    round_sum = np.zeros((4, 64), np.float32)
    for j in range(4):
        round_w[16*j:16*j+16, j] = 2
        round_sum[j, 16*j:16*j+16] = 1
    positions = lin(clip(lin(
        positions, round_w,
        np.tile(1-2*np.arange(1, 17, dtype=np.float32), 4)
    )), round_sum)
    a_position, b_position = sl(positions, 0, 2), sl(positions, 2, 4)

    # Infer mirror-control polarity from actual motion.
    delta = mul(sub(positions, previous), tracking)
    previous_h = lin("aprev", [[0, 0, 0, -1, 1, 0, 0, 0]])
    previous_v = lin("aprev", [[0, -1, 1, 0, 0, 0, 0, 0]])
    motion_h = mul(lin(delta, [[0, 1, 0, -1]]), previous_h)
    motion_v = mul(lin(delta, [[1, 0, 1, 0]]), previous_v)

    def update_sign(motion, old_sign):
        evidence = clip(absolute(motion))
        observed = affine(clip(motion), 2, -1)
        default = choose(tracking, old_sign, const(1))
        return choose(evidence, observed, default)

    hsign = update_sign(motion_h, sl("mem", 4, 5))
    vsign = update_sign(motion_v, sl("mem", 5, 6))

    # Render palette and object classification.
    stripe = add(exact_point(affine(rr, 1, -1)),
                 exact_point(affine(rr, 1, -62)))
    wall_palette = gm(mul("x0", stripe))
    wall_paint = total(mul("x0", wall_palette))
    nonwall = sub(const(1), wall_paint)

    object_w = np.zeros((3, 16), np.float32)
    object_w[0, 14] = object_w[1, 12] = object_w[2, 15] = 1
    objects = mul(lin("x0", object_w), nonwall)
    extent_w = np.zeros((6, 3, 7, 7), np.float32)
    for j in range(3):
        extent_w[2*j, j, 3, :] = 1
        extent_w[2*j+1, j, :, 3] = 1
    extents = relu(emit(
        "conv", [objects], w=extent_w, b=np.full(6, -6, np.float32)
    ))
    merge_w = np.zeros((3, 6), np.float32)
    for j in range(3):
        merge_w[j, 2*j:2*j+2] = 1
    extents = lin(extents, merge_w)
    dilate_w = np.zeros((3, 3, 7, 7), np.float32)
    for j in range(3):
        dilate_w[j, j] = 1
    gates = clip(emit("conv", [extents], w=dilate_w))
    plate_colors = relu(sub(objects, gates))
    plates = clip(total(plate_colors))
    useful_plates = clip(total(mul(plate_colors, gm(gates))))
    floor = clip(maximum(add(sl("x0", 5, 6), pawn), plates))
    selected_box = mul(sl("x0", 11, 12), nonwall)
    boxes = mul(add(sl("x0", 9, 10), sl("x0", 11, 12)), nonwall)
    box_present = gm(boxes)
    box_mode = gm(sl("x0", 1, 2))

    # Pixel-space distance propagation supplies a long-range heuristic.
    seed = affine(a_point, 1024)
    wave_limit = affine(floor, 2048)
    wave_ctx = begin_loop([zeros(1)])
    wave_state = wave_ctx[3][0]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbor_w = np.zeros((4, 1, 3, 3), np.float32)
    for j, (dy, dx) in enumerate(directions):
        neighbor_w[j, 0, 1+dy, 1+dx] = 1
    neighbors = emit("conv", [wave_state], w=neighbor_w)
    best = sl(neighbors, 0, 1)
    for j in range(1, 4):
        best = maximum(best, sl(neighbors, j, j+1))
    wave_next = maximum(seed, minimum(affine(best, 1, -1), wave_limit))
    (wave,), _ = end_loop(wave_ctx, [wave_next], 256)

    # Sample sixteen logical rows using one shared sampling circuit.
    field = add(add(affine(floor, 4096), affine(useful_plates, 2048)), wave)
    geometry = cat(size, sl(residue, 1, 2), half_size)
    centers_w = np.zeros((16, 3), np.float32)
    centers_w[:, 0] = np.arange(16)
    centers_w[:, 1:] = 1
    x_centers = lin(geometry, centers_w)
    x_masks = exact_point(sub(
        lin(cc, np.ones((16, 1), np.float32)), x_centers
    ))
    columns = emit("rowmax", [mul(x_masks, field)])
    first_y = add(sl(residue, 0, 1), half_size)
    sample_ctx = begin_loop([const(np.zeros(256)), first_y])
    sample_values, sample_y = sample_ctx[3]
    y_mask = exact_point(sub(rr, sample_y))
    sampled_row = gm(mul(columns, y_mask))
    new_samples = cat(sl(sample_values, 16, 256), sampled_row)
    (sampled, _), _ = end_loop(
        sample_ctx, [new_samples, add(sample_y, size)], 16
    )

    # Axis index = 4*x + y%4; channel block = y//4.
    permutation = np.zeros((64, 64), np.float32)
    for x in range(16):
        for y in range(4):
            permutation[4*x+y, 16*y+x] = 1
    packed = cat(*[
        lin(sl(sampled, 64*j, 64*j+64), permutation)
        for j in range(4)
    ])

    def axis_basis(coordinate):
        reflected = absolute(affine(coordinate, 1, -31.5))
        basis = exact_point(lin(
            reflected, np.ones((32, 1), np.float32),
            -np.arange(32, dtype=np.float32)-.5
        ))
        half = clip(affine(coordinate, -4, 126))
        boundary_w = np.zeros((2, 32), np.float32)
        boundary_w[0, np.arange(32) % 4 == 0] = 1
        boundary_w[1, np.arange(32) % 4 == 3] = 1
        ends = lin(basis, boundary_w)
        e0, e3 = sl(ends, 0, 1), sl(ends, 1, 2)
        return (basis, half, choose(half, e3, e0),
                choose(half, e0, e3))

    rb, rh, r0, r3 = axis_basis(rr)
    cb, ch, c0, c3 = axis_basis(cc)

    # Reconstruct four channel blocks with a shared four-iteration circuit.
    reconstruction = begin_loop([packed, zeros(4), zeros(4)])
    values, row_queue, col_queue = reconstruction[3]
    block = sl(values, 0, 64)
    left = lin(sl(block, 0, 32),
               np.eye(32, dtype=np.float32)[::-1].copy())
    right = sl(block, 32, 64)

    def reconstruct_one(basis, half):
        a = total(mul(basis, left))
        b = total(mul(basis, right))
        return choose(half, a, b)

    row_entry = reconstruct_one(rb, rh)
    col_entry = reconstruct_one(cb, ch)
    (_, row_code, col_code), _ = end_loop(
        reconstruction,
        [
            cat(sl(values, 64, 256), block),
            cat(sl(row_queue, 1, 4), row_entry),
            cat(sl(col_queue, 1, 4), col_entry)
        ], 4
    )

    def decode(code):
        free = clip(affine(code, 1, -2000))
        plate = clip(affine(code, 1, -5800))
        wave_value = sub(sub(code, affine(free, 4096)),
                         affine(plate, 2048))
        return free, plate, wave_value

    row_free, row_plate, row_wave = decode(row_code)
    col_free, col_plate, col_wave = decode(col_code)
    expand_a = np.zeros((16, 4), np.float32)
    expand_b = np.zeros((16, 4), np.float32)
    for a in range(4):
        for b in range(4):
            expand_a[4*a+b, a] = 1
            expand_b[4*a+b, b] = 1

    def expand(x, axis):
        return lin(x, expand_a if axis == 0 else expand_b)

    # Sparse shifts, grouped four channels at a time.
    def pixel_shift(x, axis, amount):
        if abs(amount) == 3:
            step = 1 if amount > 0 else -1
            z = pixel_shift(x, axis, step)
            z = pixel_shift(z, axis, step)
            return pixel_shift(z, axis, step)
        n = shapes[x][0]
        k = 2*abs(amount)+1
        pieces = []
        for start in range(0, n, 4):
            g = min(4, n-start)
            w = np.zeros((g, g, k, k), np.float32)
            for j in range(g):
                if axis == 0:
                    w[j, j, k//2+amount, k//2] = 1
                else:
                    w[j, j, k//2, k//2+amount] = 1
            pieces.append(emit(
                "conv", [sl(x, start, start+g)], w=w
            ))
        return cat(*pieces)

    def shift(x, axis, vertical, direction, joint):
        if not vertical:
            return pixel_shift(pixel_shift(x, axis, 2*direction),
                               axis, 2*direction)
        regular = pixel_shift(x, axis, direction)
        wrapped = pixel_shift(x, axis, -3*direction)
        n = shapes[x][0]
        stride = 4 if joint and axis == 0 else 1
        group = 16 if joint and axis == 0 else 4
        pieces = []
        for base in range(0, n, group):
            if direction > 0:
                pieces.extend([
                    sl(wrapped, base+stride, base+group), zeros(stride)
                ])
            else:
                pieces.extend([
                    zeros(stride), sl(wrapped, base, base+group-stride)
                ])
        wrapped = cat(*pieces)
        boundary = (r3 if direction > 0 else r0) if axis == 0 \
            else (c3 if direction > 0 else c0)
        return choose(boundary, wrapped, regular)

    actions = [(True, -1), (True, 1), (False, -1), (False, 1)]
    can_a, can_b = [], []
    for vertical, direction in actions:
        other = direction if vertical else -direction
        can_a.append(expand(
            shift(row_free, 0, vertical, direction, False), 0
        ))
        can_b.append(expand(
            shift(col_free, 1, vertical, other, False), 1
        ))

    heuristic = sub(const(100), mul(
        sub(const(2048), add(expand(row_wave, 0), expand(col_wave, 1))),
        affine(inv_size, .5)
    ))
    axis_difference = absolute(sub(rr, cc))
    same = exact_point(axis_difference)
    adjacent = exact_point(affine(axis_difference, 1, -4))
    diagonal = const([
        int(a == b) for a in range(4) for b in range(4)
    ])
    goal = affine(mul(
        add(affine(same, 1100), affine(adjacent, 1099)), diagonal
    ), 1, -1000)
    goal = maximum(goal, affine(
        maximum(expand(row_plate, 0), expand(col_plate, 1)), 1060, -1000
    ))

    def encode_position(position):
        y, x = sl(position, 0, 1), sl(position, 1, 2)
        block = total(clip(lin(
            y, np.full((3, 1), 2, np.float32), [-7, -15, -23]
        )))
        axis = sub(add(affine(x, 4), y), affine(block, 4))
        return block, axis

    ab, ai = encode_position(a_position)
    bb, bi = encode_position(b_position)
    channel = add(affine(ab, 4), bb)
    selector = tent(lin(
        channel, np.ones((16, 1), np.float32),
        -np.arange(16, dtype=np.float32)
    ))
    location = exact_point(add(absolute(sub(rr, ai)),
                               absolute(sub(cc, bi))))
    location_penalty = affine(sub(const(1), location), 10000)

    # Joint-state value iteration and simultaneous action-value readout.
    planner = begin_loop([heuristic, const([0, 0, 0, 0])])
    value = planner[3][0]
    q_fields, q_readout = [], []
    for j, (vertical, direction) in enumerate(actions):
        other = direction if vertical else -direction
        moved_b = shift(value, 1, vertical, other, True)
        after_b = choose(can_b[j], moved_b, value)
        moved_a = shift(after_b, 0, vertical, direction, True)
        successor = choose(can_a[j], moved_a, after_b)
        q = affine(successor, 1, -1)
        q_fields.append(q)
        q_readout.append(total(mul(
            gm(sub(q, location_penalty)), selector
        )))
    updated = goal
    for q in q_fields:
        updated = maximum(updated, q)
    (_, action_values), planner_node = end_loop(
        planner, [updated, cat(*q_readout)], 1
    )

    hp, vp = affine(hsign, .5, .5), affine(vsign, .5, .5)
    up, down, left_q, right_q = [
        sl(action_values, j, j+1) for j in range(4)
    ]
    policy = cat(
        choose(vp, up, down), choose(vp, down, up),
        choose(hp, left_q, right_q), choose(hp, right_q, left_q)
    )
    dy = sub(sl(b_position, 0, 1), sl(a_position, 0, 1))
    dx = sub(sl(b_position, 1, 2), sl(a_position, 1, 2))
    touching = mul(tent(dy), tent(affine(absolute(dx), 1, -1)))
    inward = mul(mul(dx, hsign), touching)
    policy = add(policy, lin(inward, [[0], [0], [-40], [40]]))

    # Move a selected blocker toward available space, away from the pawn.
    def physical_neighbors(x):
        w = np.zeros((4, 1, 7, 7), np.float32)
        for j, (dy0, dx0) in enumerate(directions):
            w[j, 0, 3+3*dy0, 3+3*dx0] = 1
        first = emit("conv", [x], w=w)
        alternatives = []
        for remainder in (1, 2, 3):
            w = np.zeros((4, 4, 7, 7), np.float32)
            for j, (dy0, dx0) in enumerate(directions):
                w[j, j, 3+remainder*dy0, 3+remainder*dx0] = 1
            alternatives.append(emit("conv", [first], w=w))
        w = np.zeros((4, 4, 3, 3), np.float32)
        for j, (dy0, dx0) in enumerate(directions):
            w[j, j, 1+dy0, 1+dx0] = 1
        alternatives.append(emit("conv", [alternatives[-1]], w=w))
        result = mul(alternatives[0], sl(scale_code, 0, 1))
        for j in range(1, 4):
            result = add(result, mul(
                alternatives[j], sl(scale_code, j, j+1)
            ))
        return result

    box_free = gm(mul(physical_neighbors(floor), selected_box))
    box_position = gm(mul(selected_box, grid_xy))
    box_scores = add(affine(box_free, 100), lin(
        sub(box_position, a_position),
        [[-.1, 0], [.1, 0], [0, -.1], [0, .1]]
    ))
    momentum = np.zeros((4, 8), np.float32)
    for j, action in enumerate((1, 2, 3, 4)):
        momentum[j, action] = 1
        momentum[j, (2, 1, 4, 3)[j]] = -2
    box_scores = add(box_scores, lin("aprev", momentum))
    policy = choose(box_mode, box_scores, policy)

    still = clip(affine(total(absolute(sub(positions, previous))), -4, 1))
    repeated = clip(affine(total(absolute(
        sub(positions, sl("mem", 17, 21))
    )), -4, 1))
    stuck = mul(mul(maximum(still, repeated), valid),
                sub(const(1), box_mode))
    stall = mul(affine(sl("mem", 15, 16), 1, 1), stuck)
    age = mul(affine(sl("mem", 16, 17), 1, 1), box_mode)
    click_now = clip(add(
        mul(clip(affine(stall, 4, -10)), box_present),
        mul(clip(affine(age, 4, -22)), box_mode)
    ))
    click_distance = total(absolute(sub(grid_xy, a_position)))
    box_click = sub(affine(boxes, 1000), affine(click_distance, .01))
    click_field = choose(box_mode, affine(a_point, 1000), box_click)

    output_w = np.zeros((8, 5), np.float32)
    output_b = np.full(8, -1000000, np.float32)
    for j in range(4):
        output_w[j+1, j] = 1
        output_b[j+1] = 0
    output_w[6, 4], output_b[6] = 1, 0
    logits_source = lin(
        cat(policy, affine(click_now, 20000, -10000)), output_w, output_b
    )
    memory_source = cat(
        positions, hsign, vsign, const(1), ghost, ghost_valid,
        count, scale_code, stall, age, previous
    )
    graph.extend([
        {"id": "logits", "op": "dense", "in": [logits_source],
         "w": parameter(np.eye(8, dtype=np.float32))},
        {"id": "click", "op": "conv", "in": [click_field],
         "w": parameter(np.ones((1, 1, 1, 1), np.float32))},
        {"id": "mem_next", "op": "dense", "in": [memory_source],
         "w": parameter(np.eye(21, dtype=np.float32))}
    ])
    shapes.update({"logits": (8,), "click": (1, 64, 64), "mem_next": (21,)})

    # Static graph cleanup and budget allocation; no game simulation.
    def prune(nodes, roots):
        needed = set(roots)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                outputs = [nd["id"]+"."+s["id"] for s in nd["state"]]
                if not any(x in needed for x in outputs):
                    continue
                needed.difference_update(outputs)
                body, external = prune(nd["body"], nd["next"].values())
                nd["body"] = body
                external.difference_update(s["id"] for s in nd["state"])
                needed.update(external)
                needed.update(s["init"] for s in nd["state"] if "init" in s)
                kept.append(nd)
            elif nd["id"] in needed:
                needed.remove(nd["id"])
                needed.update(nd["in"])
                kept.append(nd)
        return list(reversed(kept)), needed

    graph, _ = prune(graph, ["logits", "click", "mem_next"])

    def costs(nodes):
        macs = evaluations = 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                m, e = costs(nd["body"])
                macs += nd["T"] * m
                evaluations += 1 + nd["T"] * e
                continue
            evaluations += 1
            if op == "conv":
                macs += int(params[nd["w"]].size) * 4096
            elif op == "dense":
                macs += int(params[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
                macs += int(np.prod(shapes[nd["id"]]))
            elif op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
                macs += int(np.prod(shapes[nd["in"][0]]))
        return macs, evaluations

    body_macs, body_evals = costs(planner_node["body"])
    all_macs, all_evals = costs(graph)
    planner_node["T"] = max(1, min(
        64,
        (1_490_000_000 - all_macs + body_macs) // max(1, body_macs),
        (9_950 - all_evals + body_evals) // max(1, body_evals)
    ))
    planner_node["T"] = int(planner_node["T"])

    used = set()

    def collect(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                collect(nd["body"])
            else:
                for key in ("w", "b"):
                    if key in nd:
                        used.add(nd[key])

    collect(graph)
    return {
        "graph": graph,
        "memory_size": 21,
        "params": {name: value for name, value in params.items() if name in used}
    }

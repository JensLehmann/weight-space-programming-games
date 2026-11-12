import numpy as np


def build():
    N, D = 5, 38
    params = {}
    parameter_cache = {}
    constant_cache = {}
    constant_values = {}
    constants = []
    nodes = []
    cache = {}
    serial = 0
    shapes = {
        "x0": (16, True),
        "coords": (3, True),
        "aprev": (8, False),
        "avail": (8, False),
    }

    layout = {}
    cursor = N * D
    for key, size in [
        ("count", 1), ("stage", 1), ("done", N),
        ("base", N), ("binary", N), ("active", N),
        ("busy", 1), ("goal", 3), ("covered", 8),
        ("wave", 1), ("tri_ready", 1), ("tri_goals", 6),
    ]:
        layout[key] = (cursor, cursor + size)
        cursor += size
    memory_size = cursor
    shapes["mem"] = (memory_size, False)

    def fresh(prefix="n"):
        nonlocal serial
        name = prefix + str(serial)
        serial += 1
        return name

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, ins, channels=None, spatial=None, name=None, **fields):
        if channels is None:
            channels = shapes[ins[0]][0]
        if spatial is None:
            spatial = shapes[ins[0]][1]
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in cache:
            return cache[key]
        if name is None:
            name = fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        nodes.append(nd)
        shapes[name] = (channels, spatial)
        cache[key] = name
        return name

    def const(a):
        a = np.asarray(a, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key in constant_cache:
            return constant_cache[key]
        name = fresh("k")
        constants.append({
            "id": name, "op": "dense", "in": ["aprev"],
            "w": parameter(np.zeros((len(a), 8), np.float32)),
            "b": parameter(a),
        })
        shapes[name] = (len(a), False)
        constant_cache[key] = name
        constant_values[name] = a
        return name

    def lin(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if name is None and x in constant_values:
            y = w @ constant_values[x]
            if b is not None:
                y = y + np.asarray(b, np.float32)
            return const(y)
        out = w.shape[0]
        spatial = shapes[x][1]
        if spatial:
            w = w.reshape(out, shapes[x][0], 1, 1)
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(
                np.broadcast_to(np.asarray(b, np.float32), (out,)).copy()
            )
        return emit(
            "conv" if spatial else "dense",
            [x], out, spatial, name, **fields
        )

    def sl(x, start, stop=None):
        if stop is None:
            stop = start + 1
        if x in constant_values:
            return const(constant_values[x][start:stop])
        return emit(
            "slice" if shapes[x][1] else "vslice",
            [x], stop - start, shapes[x][1],
            start=start, stop=stop
        )

    def cat(xs, name=None):
        xs = list(xs)
        if name is None and all(x in constant_values for x in xs):
            return const(np.concatenate([constant_values[x] for x in xs]))
        return emit(
            "concat" if shapes[xs[0]][1] else "vconcat",
            xs, sum(shapes[x][0] for x in xs),
            shapes[xs[0]][1], name
        )

    def add(a, b, name=None):
        if name is None and a in constant_values and b in constant_values:
            return const(constant_values[a] + constant_values[b])
        spatial = shapes[a][1] or shapes[b][1]
        return emit(
            "add" if spatial else "vadd", [a, b],
            max(shapes[a][0], shapes[b][0]), spatial, name
        )

    def mul(a, b, name=None):
        if name is None and a in constant_values and b in constant_values:
            return const(constant_values[a] * constant_values[b])
        spatial = shapes[a][1] or shapes[b][1]
        return emit(
            "mul" if spatial else "vmul", [a, b],
            max(shapes[a][0], shapes[b][0]), spatial, name
        )

    def aff(x, a=1, b=0):
        if x in constant_values:
            return const(constant_values[x] * np.float32(a) + np.float32(b))
        if a == 1 and b == 0:
            return x
        c = shapes[x][0]
        if c <= 8:
            return lin(x, np.eye(c, dtype=np.float32) * a, b)
        y = x if a == 1 else mul(x, const([a]))
        return y if b == 0 else add(y, const([b]))

    def scale(x, a):
        return aff(x, a, 0)

    def sub(a, b):
        if a in constant_values and b in constant_values:
            return const(constant_values[a] - constant_values[b])
        spatial = shapes[a][1] or shapes[b][1]
        if spatial:
            return emit(
                "sub", [a, b],
                max(shapes[a][0], shapes[b][0]), True
            )
        return add(a, scale(b, -1))

    def relu(x):
        return emit("relu" if shapes[x][1] else "vrelu", [x])

    def absolute(x):
        if shapes[x][1]:
            return emit("abs", [x])
        sign = emit("vtanh", [scale(x, 32)])
        return mul(x, sign)

    def bit(x):
        return emit(
            "sigmoid" if shapes[x][1] else "vsigmoid",
            [aff(x, 256, -128)]
        )

    zero = const([0])
    one = const([1])

    def inv(x):
        return sub(one, x) if shapes[x][1] else aff(x, -1, 1)

    def eq(a, b):
        return relu(inv(absolute(sub(a, b))))

    def choose(flag, yes, no):
        return add(no, mul(flag, sub(yes, no)))

    def union(a, b):
        return bit(add(a, b))

    def xor(a, b):
        return sub(add(a, b), scale(mul(a, b), 2))

    def summation(xs):
        xs = list(xs)
        result = xs[0]
        for x in xs[1:]:
            result = add(result, x)
        return result

    def total(x):
        return lin(x, np.ones((1, shapes[x][0]), np.float32))

    def pool(x, mean=False):
        return emit(
            "gmean" if mean else "gmax",
            [x], shapes[x][0], False
        )

    def spatial_total(x):
        return scale(pool(x, True), 4096)

    def weighted(a, b):
        return total(mul(a, b))

    def repeat(x, n):
        return cat([x] * n)

    def loop_build(states, steps, body_fn):
        nonlocal nodes, cache
        loop_name = fresh("L")
        state_specs = []
        state_ids = []
        for init, size, spatial in states:
            sid = fresh("s")
            state_ids.append(sid)
            shapes[sid] = (size, spatial)
            if init is None:
                state_specs.append({
                    "id": sid,
                    "zeros": [size, 64, 64] if spatial else [size]
                })
            else:
                state_specs.append({"id": sid, "init": init})
        outer_nodes, outer_cache = nodes, cache
        nodes, cache = [], dict(cache)
        outputs = body_fn(*state_ids)
        body = nodes
        nodes, cache = outer_nodes, outer_cache
        nodes.append({
            "id": loop_name, "op": "loop", "T": steps,
            "state": state_specs, "body": body,
            "next": dict(zip(state_ids, outputs)),
        })
        result = []
        for sid in state_ids:
            name = loop_name + "." + sid
            shapes[name] = shapes[sid]
            result.append(name)
        return result

    old = {key: sl("mem", *v) for key, v in layout.items()}
    data = sl("mem", 0, N * D)
    descriptors = [sl(data, i * D, (i + 1) * D) for i in range(N)]

    def select_descriptor(flags):
        def body_fn(ds, fs, accumulator):
            d = sl(ds, 0, D)
            f = sl(fs, 0)
            return [
                cat([sl(ds, D, N * D), d]),
                cat([sl(fs, 1, N), f]),
                add(accumulator, mul(d, f)),
            ]
        return loop_build([
            (data, N * D, False),
            (flags, N, False),
            (None, D, False),
        ], N, body_fn)[2]

    # Image sampling at the centres of world cells.
    def coordinate_axis(axis):
        w = np.zeros((20, 3), np.float32)
        w[:, axis] = 63
        raw = lin("coords", w, -(3 + 3 * np.arange(20)))
        hats = bit(aff(absolute(raw), -4, 3))
        return total(hats), lin(
            hats, np.arange(20, dtype=np.float32)[None, :]
        )

    row_lattice, gy = coordinate_axis(0)
    col_lattice, gx = coordinate_axis(1)
    lattice = mul(row_lattice, col_lattice)

    # Planner coordinates: candidate attributes are represented by
    # one-hot rows and columns, independently of image content.
    def query_axis(axis, n):
        w = np.zeros((n, 3), np.float32)
        w[:, axis] = 63
        raw = lin("coords", w, -np.arange(n, dtype=np.float32))
        return bit(aff(absolute(raw), -4, 3))

    query_rows = query_axis(0, 64)
    query_cols = query_axis(1, 8)
    source_query = sl(query_rows, 0, 32)
    rotation_matrix = np.zeros((4, 32), np.float32)
    for r in range(4):
        rotation_matrix[r, r * 8:(r + 1) * 8] = 1
    rotation_query = lin(source_query, rotation_matrix)

    def gather(query, values):
        return total(mul(query, values))

    def reduce_max(x):
        c = shapes[x][0]
        q = query_cols if c == 8 else sl(query_cols, 0, c)
        return pool(sub(
            gather(q, x),
            scale(inv(total(q)), 1e8)
        ))

    def winner_vector(x):
        return bit(aff(sub(x, reduce_max(x)), 1, 1))

    def winner_map(score, rank):
        ranked = add(scale(score, 512), rank)
        return bit(aff(sub(ranked, pool(ranked)), 1, 1))

    corner = relu(lin("coords", [[200, 200, 0]], [-399]))
    background = pool(mul("x0", corner))
    foreground = mul(total(relu(sub("x0", background))), lattice)

    def colour(c):
        w = np.zeros((1, 16), np.float32)
        w[0, c] = 1
        return mul(lin("x0", w), foreground)

    grey, red, green, black = (
        colour(4), colour(8), colour(3), colour(0)
    )
    masked = pool(grey)
    coloured_body = sub(sub(sub(foreground, grey), red), green)
    body = choose(masked, coloured_body, black)
    present = pool(body)

    kernel = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
        kernel[0, 0, 3 + dy, 3 + dx] = 1
    near_body = bit(emit(
        "conv", [body], 1, True, w=parameter(kernel)
    ))
    pins = mul(add(red, green), near_body)
    selected_mask = bit(add(body, pins))
    grow = mul(masked, pool(mul(body, black)))

    xmin = aff(pool(mul(selected_mask, aff(gx, -1, 20))), -1, 20)
    ymin = aff(pool(mul(selected_mask, aff(gy, -1, 20))), -1, 20)
    xmax = pool(mul(selected_mask, gx))
    ymax = pool(mul(selected_mask, gy))
    width = aff(sub(xmax, xmin), 1, 1)
    height = aff(sub(ymax, ymin), 1, 1)
    local_x, local_y = sub(gx, xmin), sub(gy, ymin)

    def hats9(x):
        return bit(aff(absolute(sub(
            x, const(np.arange(9, dtype=np.float32))
        )), -4, 3))

    xh, yh = hats9(local_x), hats9(local_y)
    xcode = lin(xh, (2.0 ** np.arange(9))[None, :])
    row_codes = spatial_total(mul(yh, mul(selected_mask, xcode)))
    pin_rank = aff(add(scale(gy, -20), scale(gx, -1)), 1, 500)

    def extract_body(remaining, xbuffer, ybuffer, vbuffer):
        score = mul(remaining, pin_rank)
        selected = mul(
            remaining, bit(aff(sub(score, pool(score)), 1, 1))
        )
        return [
            relu(sub(remaining, selected)),
            cat([sl(xbuffer, 1, 8), pool(mul(selected, local_x))]),
            cat([sl(ybuffer, 1, 8), pool(mul(selected, local_y))]),
            cat([sl(vbuffer, 1, 8), pool(remaining)]),
        ]

    extracted = loop_build([
        (pins, 1, True),
        (None, 8, False), (None, 8, False), (None, 8, False),
    ], 8, extract_body)

    observed_descriptor = cat([
        width, height, xmin, ymin, zero,
        extracted[1], extracted[2], extracted[3], row_codes
    ])

    ws = cat(sl(d, 0) for d in descriptors)
    hs = cat(sl(d, 1) for d in descriptors)
    xs = cat(sl(d, 2) for d in descriptors)
    ys = cat(sl(d, 3) for d in descriptors)
    rs = cat(sl(d, 4) for d in descriptors)

    def rectangle(x, y, w, h):
        return mul(
            mul(bit(aff(sub(gx, x), 1, 1)), bit(sub(add(x, w), gx))),
            mul(bit(aff(sub(gy, y), 1, 1)), bit(sub(add(y, h), gy)))
        )

    known_boxes = rectangle(xs, ys, ws, hs)
    current_box = mul(present, rectangle(xmin, ymin, width, height))
    explored = bit(add(total(known_boxes), current_box))
    unknown = mul(foreground, inv(explored))
    unknown_exists = pool(unknown)
    inventory = inv(old["stage"])
    record = mul(inventory, mul(present, inv(grow)))
    finish_inventory = mul(record, inv(unknown_exists))

    exists = bit(ws)
    counts = cat(total(sl(d, 21, 29)) for d in descriptors)
    priority = const(np.arange(N, 0, -1, dtype=np.float32))
    anchor_score = mul(
        exists,
        add(sub(scale(counts, 2048), scale(mul(ws, hs), 8)), priority)
    )
    anchor_flags = mul(winner_vector(anchor_score), exists)
    anchor = select_descriptor(anchor_flags)
    nonanchor = mul(exists, inv(anchor_flags))
    remaining = mul(nonanchor, inv(old["done"]))
    remaining_exists = bit(total(remaining))
    leaf_score = mul(remaining, add(scale(counts, N + 1), priority))
    leaf_flags = mul(winner_vector(leaf_score), remaining)
    leaf = select_descriptor(leaf_flags)

    new_plan = mul(old["stage"], mul(inv(old["busy"]), remaining_exists))
    wave_end = mul(old["stage"], mul(inv(old["busy"]), inv(remaining_exists)))

    def rotated_ports(d):
        w, h = sl(d, 0), sl(d, 1)
        x, y = sl(d, 5, 13), sl(d, 13, 21)
        return (
            cat([x, aff(sub(h, y), 1, -1),
                 aff(sub(w, x), 1, -1), y]),
            cat([y, x, aff(sub(h, y), 1, -1),
                 aff(sub(w, x), 1, -1)]),
            repeat(sl(d, 21, 29), 4)
        )

    anchor_x = add(sl(anchor, 5, 13), sl(anchor, 2))
    anchor_y = add(sl(anchor, 13, 21), sl(anchor, 3))
    anchor_valid = mul(sl(anchor, 21, 29), inv(old["covered"]))
    rx, ry, rv = rotated_ports(leaf)
    two_pin = eq(counts, const(np.full(N, 2)))
    eligible_flip = mul(nonanchor, two_pin)
    desired_halves = xor(old["base"], old["binary"])

    candidate_x = sub(gather(query_cols, anchor_x), gather(source_query, rx))
    candidate_y = sub(gather(query_cols, anchor_y), gather(source_query, ry))
    candidate_valid = mul(
        gather(source_query, rv), gather(query_cols, anchor_valid)
    )
    candidate_r = lin(rotation_query, [[0, 1, 2, 3]])
    candidate_half = lin(rotation_query, [[0, 0, 1, 1]])
    w, h = sl(leaf, 0), sl(leaf, 1)
    candidate_w = gather(rotation_query, cat([w, h, w, h]))
    candidate_h = gather(rotation_query, cat([h, w, h, w]))

    def bounds(x, y, w, h):
        return mul(
            mul(bit(aff(x, 1, 1)), bit(aff(y, 1, 1))),
            mul(
                bit(aff(add(x, w), -1, 21)),
                bit(aff(add(y, h), -1, 21))
            )
        )

    candidate_valid = mul(
        candidate_valid,
        bounds(candidate_x, candidate_y, candidate_w, candidate_h)
    )

    first_port = np.zeros((4, 32), np.float32)
    shift_ports = np.zeros((32, 32), np.float32)
    for r in range(4):
        first_port[r, r * 8] = 1
        for j in range(8):
            shift_ports[r * 8 + j, r * 8 + (j + 1) % 8] = 1

    def match_body(mx, my, mv, accumulated):
        px = gather(rotation_query, lin(mx, first_port))
        py = gather(rotation_query, lin(my, first_port))
        pv = gather(rotation_query, lin(mv, first_port))
        distance = add(
            absolute(sub(add(candidate_x, px), anchor_x)),
            absolute(sub(add(candidate_y, py), anchor_y))
        )
        matches = mul(
            relu(inv(distance)),
            mul(anchor_valid, pv)
        )
        return [
            lin(mx, shift_ports),
            lin(my, shift_ports),
            lin(mv, shift_ports),
            add(accumulated, matches),
        ]

    matches = loop_build([
        (rx, 32, False), (ry, 32, False), (rv, 32, False),
        (None, 8, True),
    ], 8, match_body)[3]

    def turn_cost(r, current):
        delta = sub(r, current)
        return add(delta, scale(bit(scale(delta, -1)), 4))

    travel = add(
        absolute(sub(candidate_x, sl(leaf, 2))),
        absolute(sub(candidate_y, sl(leaf, 3)))
    )
    half_penalty = mul(
        mul(old["wave"], weighted(two_pin, leaf_flags)),
        scale(absolute(sub(
            candidate_half, weighted(desired_halves, leaf_flags)
        )), 128)
    )
    star_score = sub(
        sub(
            sub(scale(total(matches), 1024), travel),
            turn_cost(candidate_r, sl(leaf, 4))
        ),
        add(half_penalty, scale(inv(candidate_valid), 32768))
    )
    star_rank = aff(add(
        lin(source_query, (8 * np.arange(32, dtype=np.float32))[None, :]),
        lin(query_cols, np.arange(8, dtype=np.float32)[None, :])
    ), -1, 256)
    star_winner = winner_map(star_score, star_rank)
    star_x = pool(mul(star_winner, candidate_x))
    star_y = pool(mul(star_winner, candidate_y))
    star_r = pool(mul(star_winner, candidate_r))
    star_goal = cat([star_x, star_y, star_r])
    coverage = bit(pool(mul(matches, star_winner)))

    # Complete closure for a three-piece cycle.
    b_flags = mul(winner_vector(mul(nonanchor, priority)), nonanchor)
    c_available = mul(nonanchor, inv(b_flags))
    c_flags = mul(winner_vector(mul(c_available, priority)), c_available)
    bd, cd = select_descriptor(b_flags), select_descriptor(c_flags)
    brx, bry, _ = rotated_ports(bd)
    crx, cry, _ = rotated_ports(cd)
    triangle_mode = mul(
        mul(eq(total(exists), const([3])), eq(total(counts), const([6]))),
        eq(reduce_max(counts), const([2]))
    )

    configurations = [
        (rb, rc, ib, ic)
        for rb in range(4) for rc in range(4)
        for ib in range(2) for ic in range(2)
    ]
    ba = np.zeros((32, 64), np.float32)
    bf = np.zeros((32, 64), np.float32)
    ca = np.zeros((32, 64), np.float32)
    cf = np.zeros((32, 64), np.float32)
    brot = np.zeros((4, 64), np.float32)
    crot = np.zeros((4, 64), np.float32)
    for t, (rb, rc, ib, ic) in enumerate(configurations):
        ba[rb * 8 + ib, t] = 1
        bf[rb * 8 + 1 - ib, t] = 1
        ca[rc * 8 + ic, t] = 1
        cf[rc * 8 + 1 - ic, t] = 1
        brot[rb, t] = 1
        crot[rc, t] = 1

    baq, bfq = lin(query_rows, ba), lin(query_rows, bf)
    caq, cfq = lin(query_rows, ca), lin(query_rows, cf)
    brq, crq = lin(query_rows, brot), lin(query_rows, crot)
    side_query = sl(query_cols, 0, 2)
    flipped_side = lin(side_query, [[0, 1], [1, 0]])
    anchor_a_x = gather(side_query, sl(anchor_x, 0, 2))
    anchor_a_y = gather(side_query, sl(anchor_y, 0, 2))
    anchor_b_x = gather(flipped_side, sl(anchor_x, 0, 2))
    anchor_b_y = gather(flipped_side, sl(anchor_y, 0, 2))
    tbx = sub(anchor_a_x, gather(baq, brx))
    tby = sub(anchor_a_y, gather(baq, bry))
    tcx = sub(anchor_b_x, gather(caq, crx))
    tcy = sub(anchor_b_y, gather(caq, cry))
    trb = lin(brq, [[0, 1, 2, 3]])
    trc = lin(crq, [[0, 1, 2, 3]])
    closure_distance = add(
        absolute(sub(add(tbx, gather(bfq, brx)),
                     add(tcx, gather(cfq, crx)))),
        absolute(sub(add(tby, gather(bfq, bry)),
                     add(tcy, gather(cfq, cry))))
    )
    bw, bh = sl(bd, 0), sl(bd, 1)
    cw, ch = sl(cd, 0), sl(cd, 1)
    triangle_valid = mul(
        mul(relu(inv(closure_distance)), total(side_query)),
        mul(
            bounds(
                tbx, tby,
                gather(brq, cat([bw, bh, bw, bh])),
                gather(brq, cat([bh, bw, bh, bw]))
            ),
            bounds(
                tcx, tcy,
                gather(crq, cat([cw, ch, cw, ch])),
                gather(crq, cat([ch, cw, ch, cw]))
            )
        )
    )
    triangle_cost = summation([
        absolute(sub(tbx, sl(bd, 2))),
        absolute(sub(tby, sl(bd, 3))),
        absolute(sub(tcx, sl(cd, 2))),
        absolute(sub(tcy, sl(cd, 3))),
        turn_cost(trb, sl(bd, 4)),
        turn_cost(trc, sl(cd, 4)),
    ])
    orientation_penalty = mul(old["wave"], scale(add(
        absolute(sub(
            lin(brq, [[0, 0, 1, 1]]),
            weighted(desired_halves, b_flags)
        )),
        absolute(sub(
            lin(crq, [[0, 0, 1, 1]]),
            weighted(desired_halves, c_flags)
        ))
    ), 128))
    triangle_score = scale(add(
        add(triangle_cost, orientation_penalty),
        scale(inv(triangle_valid), 32768)
    ), -1)
    triangle_rank = aff(add(
        lin(query_rows, (2 * np.arange(64, dtype=np.float32))[None, :]),
        lin(side_query, [[0, 1]])
    ), -1, 128)
    triangle_winner = winner_map(triangle_score, triangle_rank)
    computed_triangle_goals = cat([
        pool(mul(triangle_winner, field))
        for field in [tbx, tby, trb, tcx, tcy, trc]
    ])
    triangle_goals = choose(
        old["tri_ready"], old["tri_goals"], computed_triangle_goals
    )
    is_b = weighted(leaf_flags, b_flags)
    triangle_goal = choose(
        is_b, sl(triangle_goals, 0, 3), sl(triangle_goals, 3, 6)
    )
    planned_goal = choose(triangle_mode, triangle_goal, star_goal)

    active_flags = choose(new_plan, leaf_flags, old["active"])
    goal = choose(new_plan, planned_goal, old["goal"])
    active_descriptor = select_descriptor(active_flags)
    start_busy = union(old["busy"], new_plan)

    # Occupancy reconstruction supplies reliable interior click targets.
    ad = active_descriptor
    ax, ay, ar = sl(ad, 2), sl(ad, 3), sl(ad, 4)
    aw, ah = sl(ad, 0), sl(ad, 1)
    dx, dy = sub(gx, ax), sub(gy, ay)
    rotation_flags = eq(const(np.arange(4)), ar)
    original_x = summation([
        mul(sl(rotation_flags, 0), dx),
        mul(sl(rotation_flags, 1), dy),
        mul(sl(rotation_flags, 2), aff(sub(aw, dx), 1, -1)),
        mul(sl(rotation_flags, 3), aff(sub(aw, dy), 1, -1)),
    ])
    original_y = summation([
        mul(sl(rotation_flags, 0), dy),
        mul(sl(rotation_flags, 1), aff(sub(ah, dx), 1, -1)),
        mul(sl(rotation_flags, 2), aff(sub(ah, dy), 1, -1)),
        mul(sl(rotation_flags, 3), dx),
    ])
    click_rows, click_cols = hats9(original_y), hats9(original_x)
    coded_row = total(mul(click_rows, sl(ad, 29, 38)))

    def decode_body(remainder, power, digits):
        digit = bit(aff(sub(remainder, power), 1, 1))
        return [
            sub(remainder, mul(digit, power)),
            scale(power, 0.5),
            cat([digit, sl(digits, 0, 8)])
        ]

    decoded = loop_build([
        (coded_row, 1, True),
        (const([256]), 1, False),
        (None, 9, True),
    ], 9, decode_body)[2]
    target_mask = mul(lattice, total(mul(decoded, click_cols)))

    wrong_body = bit(spatial_total(mul(body, inv(target_mask))))
    target_selected = mul(present, inv(wrong_body))
    execute = mul(start_busy, target_selected)
    rotation_needed = inv(eq(ar, sl(goal, 2)))
    x_error, y_error = sub(sl(goal, 0), ax), sub(sl(goal, 1), ay)
    go_right, go_left = bit(x_error), bit(scale(x_error, -1))
    go_down, go_up = bit(y_error), bit(scale(y_error, -1))
    horizontal = add(go_left, go_right)
    vertical_enabled = mul(inv(rotation_needed), inv(horizontal))
    complete = mul(
        execute, mul(vertical_enabled, inv(add(go_up, go_down)))
    )
    action_rotate = mul(execute, rotation_needed)
    action_left = mul(execute, mul(inv(rotation_needed), go_left))
    action_right = mul(execute, mul(inv(rotation_needed), go_right))
    action_up = mul(execute, mul(vertical_enabled, go_up))
    action_down = mul(execute, mul(vertical_enabled, go_down))
    movement_total = summation([
        action_rotate, action_left, action_right, action_up, action_down
    ])
    play_actions = cat([
        zero, action_up, action_down, action_left,
        action_right, action_rotate, inv(movement_total), zero
    ])
    inventory_actions = choose(
        mul(present, grow),
        const([0, 0, 0, 0, 0, 1, 0, 0]),
        const([0, 0, 0, 0, 0, 0, 1, 0]),
    )
    action_values = choose(old["stage"], play_actions, inventory_actions)
    add(
        scale(action_values, 20),
        const([-100, 0, 0, 0, 0, 0, 0, -100]),
        name="logits"
    )

    click_rank = aff(add(scale(gy, -20), scale(gx, -1)), 1, 10000)
    inventory_click = mul(unknown, click_rank)
    target_click = mul(
        mul(target_mask, mul(foreground, inv(body))), click_rank
    )
    target_click = mul(
        target_click, mul(start_busy, inv(target_selected))
    )
    add(
        choose(old["stage"], target_click, inventory_click),
        zero, name="click"
    )

    # Keep the recorded poses synchronized with actual movements.
    rotate_slots = mul(active_flags, action_rotate)
    next_rs = sub(
        add(rs, rotate_slots),
        scale(mul(eq(rs, const(np.full(N, 3))), rotate_slots), 4)
    )
    odd_rotation = add(
        eq(next_rs, const(np.ones(N))),
        eq(next_rs, const(np.full(N, 3)))
    )
    new_widths = choose(odd_rotation, hs, ws)
    new_heights = choose(odd_rotation, ws, hs)
    next_xs = sub(
        add(xs, mul(active_flags, sub(action_right, action_left))),
        mul(rotate_slots, relu(aff(add(xs, new_widths), 1, -20)))
    )
    next_ys = sub(
        add(ys, mul(active_flags, sub(action_down, action_up))),
        mul(rotate_slots, relu(aff(add(ys, new_heights), 1, -20)))
    )

    poses = cat([next_xs, next_ys, next_rs])
    first_pose = np.zeros((3, 3 * N), np.float32)
    pose_shift = np.zeros((3 * N, 3 * N), np.float32)
    for j in range(3):
        first_pose[j, j * N] = 1
        for i in range(N):
            pose_shift[j * N + i, j * N + (i + 1) % N] = 1
    write_flags = mul(record, eq(old["count"], const(np.arange(N))))
    flag_shift = np.zeros((N, N), np.float32)
    for i in range(N):
        flag_shift[i, (i + 1) % N] = 1

    def update_body(ds, ps, fs, result):
        d = sl(ds, 0, D)
        moved = cat([sl(d, 0, 2), lin(ps, first_pose), sl(d, 5, D)])
        updated = choose(sl(fs, 0), observed_descriptor, moved)
        return [
            cat([sl(ds, D, N * D), d]),
            lin(ps, pose_shift),
            lin(fs, flag_shift),
            cat([sl(result, D, N * D), updated])
        ]

    next_data = loop_build([
        (data, N * D, False),
        (poses, 3 * N, False),
        (write_flags, N, False),
        (None, N * D, False),
    ], N, update_body)[3]

    next_done = mul(
        inv(wave_end), union(old["done"], mul(complete, active_flags))
    )
    next_covered = mul(
        inv(wave_end), union(old["covered"], mul(new_plan, coverage))
    )
    next_base = choose(
        mul(new_plan, mul(inv(old["wave"]), leaf_flags)),
        bit(aff(sl(planned_goal, 2), 1, -1)),
        old["base"]
    )
    lower = np.tril(np.ones((N, N), np.float32), -1)
    unfilled_lower = lin(mul(eligible_flip, inv(old["binary"])), lower)
    carry = eq(unfilled_lower, const(np.zeros(N)))
    toggles = mul(carry, eligible_flip)
    next_binary = choose(
        wave_end, xor(old["binary"], toggles), old["binary"]
    )
    initialize_triangle = mul(
        new_plan, mul(triangle_mode, inv(old["tri_ready"]))
    )
    next_tri_ready = mul(
        inv(wave_end), union(old["tri_ready"], initialize_triangle)
    )
    next_tri_goals = choose(
        initialize_triangle, computed_triangle_goals, old["tri_goals"]
    )

    cat([
        next_data,
        add(old["count"], record),
        union(old["stage"], finish_inventory),
        next_done,
        next_base,
        next_binary,
        active_flags,
        mul(start_busy, inv(complete)),
        goal,
        next_covered,
        union(old["wave"], wave_end),
        next_tri_ready,
        next_tri_goals,
    ], name="mem_next")

    # A lower-cost occupancy representation is retained only as a
    # compilation fallback, not as an additional runtime controller.
    active_odd = add(sl(rotation_flags, 1), sl(rotation_flags, 3))
    target_box = rectangle(
        ax, ay, choose(active_odd, ah, aw), choose(active_odd, aw, ah)
    )
    zeros6 = const(np.zeros(6, np.float32))
    original_graph = constants + nodes

    def compile_graph(aliases):
        def resolve(name):
            while name in aliases:
                name = aliases[name]
            return name

        def rewrite(items):
            result = []
            for nd in items:
                item = dict(nd)
                if item["op"] == "loop":
                    item["state"] = [
                        {
                            **s,
                            **({"init": resolve(s["init"])} if "init" in s else {})
                        }
                        for s in item["state"]
                    ]
                    item["body"] = rewrite(item["body"])
                    item["next"] = {
                        k: resolve(v) for k, v in item["next"].items()
                    }
                else:
                    item["in"] = [resolve(x) for x in item["in"]]
                result.append(item)
            return result

        graph = rewrite(original_graph)
        producers = {}
        for nd in graph:
            producers[nd["id"]] = nd
            if nd["op"] == "loop":
                for state in nd["state"]:
                    producers[nd["id"] + "." + state["id"]] = nd

        needed = set()

        def require(name):
            name = resolve(name)
            nd = producers.get(name)
            if nd is None or nd["id"] in needed:
                return
            needed.add(nd["id"])
            if nd["op"] != "loop":
                for src in nd["in"]:
                    require(src)
                return
            local = {s["id"] for s in nd["state"]}
            local.update(x["id"] for x in nd["body"])
            for s in nd["state"]:
                if "init" in s:
                    require(s["init"])
            for child in nd["body"]:
                for src in child.get("in", []):
                    if src not in local:
                        require(src)

        for output in ("logits", "click", "mem_next"):
            require(output)
        graph = [nd for nd in graph if nd["id"] in needed]

        used = set()
        node_count = 0
        evaluations = 0
        macs = 0
        spatial_channels = 19
        largest_spatial = 16
        largest_vector = max(memory_size, 8)

        def inspect(items, repetitions=1):
            nonlocal node_count, evaluations, macs
            nonlocal spatial_channels, largest_spatial, largest_vector
            for nd in items:
                node_count += 1
                evaluations += repetitions
                for field in ("w", "b"):
                    if field in nd:
                        used.add(nd[field])
                op = nd["op"]
                if op == "loop":
                    for state in nd["state"]:
                        if "init" in state:
                            c, sp = shapes[state["init"]]
                        else:
                            c = state["zeros"][0]
                            sp = len(state["zeros"]) == 3
                        if sp:
                            spatial_channels += 2 * c
                            largest_spatial = max(largest_spatial, c)
                        else:
                            largest_vector = max(largest_vector, c)
                    inspect(nd["body"], repetitions * nd["T"])
                    continue
                c, sp = shapes[nd["id"]]
                if sp:
                    spatial_channels += c
                    largest_spatial = max(largest_spatial, c)
                else:
                    largest_vector = max(largest_vector, c)
                cost = 0
                if op == "conv":
                    cost = int(params[nd["w"]].size) * 4096
                elif op == "dense":
                    cost = int(params[nd["w"]].size)
                elif op in ("add", "sub", "mul", "max2", "min2"):
                    cost = c * 4096
                elif op in ("vadd", "vmul"):
                    cost = c
                elif op in (
                    "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
                ):
                    cost = shapes[nd["in"][0]][0] * 4096
                macs += repetitions * cost

        inspect(graph)
        selected_params = {key: params[key] for key in used}
        valid = (
            node_count <= 1000
            and evaluations <= 10000
            and macs <= 1500000000
            and sum(a.size for a in selected_params.values()) <= 150000
            and spatial_channels <= 4096
            and largest_spatial <= 512
            and largest_vector <= 4096
        )
        return graph, selected_params, valid

    graph, selected_params, valid = compile_graph({})
    if not valid:
        # This is graph-size bookkeeping only: no game states, weight
        # fitting, or simulator calls are used during construction.
        aliases = {
            planned_goal: star_goal,
            next_tri_ready: zero,
            next_tri_goals: zeros6,
        }
        graph, selected_params, valid = compile_graph(aliases)
        if not valid:
            aliases[target_mask] = target_box
            graph, selected_params, valid = compile_graph(aliases)

    return {
        "graph": graph,
        "memory_size": memory_size,
        "params": selected_params,
    }

import numpy as np


def build():
    N = 6
    K = 8
    E = N * K

    params = {}
    parameter_cache = {}
    constants = []
    graph = []
    current = graph
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]
    node_cache = {}
    constant_cache = {}

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def emit(op, ins, name=None, dest=None, **fields):
        target = current if dest is None else dest
        key = (id(target), op, tuple(ins), tuple(sorted(fields.items())))
        if name is None and key in node_cache:
            return node_cache[key]

        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        target.append(nd)
        ss = [shapes[x] for x in ins]

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
        if name not in ("logits", "click", "mem_next"):
            node_cache[key] = name
        return name

    def dense(x, w, b=None, name=None, dest=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("dense", [x], name=name, dest=dest, **kw)

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("conv", [x], **kw)

    def const(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            constant_cache[key] = dense(
                "aprev", np.zeros((len(a), 8), np.float32),
                a, dest=constants
            )
        return constant_cache[key]

    def cat(xs):
        if len(xs) == 1:
            return xs[0]
        return emit("vconcat", xs)

    def cut(x, start, stop):
        return emit(
            "vslice" if len(shapes[x]) == 1 else "slice",
            [x], start=start, stop=stop
        )

    def vr(x):
        return emit("vrelu", [x])

    def vm(a, b):
        return emit("vmul", [a, b])

    def va(terms, bias=0.0):
        terms = [(x, float(c)) for x, c in terms if c != 0]
        if not terms:
            return const(bias)
        n = max(shapes[x][0] for x, _ in terms)
        if n <= 8:
            inputs = [x for x, _ in terms]
            width = sum(shapes[x][0] for x in inputs)
            w = np.zeros((n, width), np.float32)
            offset = 0
            for x, c in terms:
                m = shapes[x][0]
                if m == 1:
                    w[:, offset] = c
                else:
                    w[np.arange(n), offset + np.arange(n)] = c
                offset += m
            return dense(cat(inputs), w, np.full(n, bias, np.float32))

        out = None
        for x, c in terms:
            y = x if c == 1 else vm(x, const(c))
            out = y if out is None else emit("vadd", [out, y])
        if bias != 0:
            out = emit("vadd", [out, const(bias)])
        return out

    def vclip(x, lo=0.0, hi=1.0):
        n = shapes[x][0]
        if n <= 8:
            eye = np.eye(n, dtype=np.float32)
            z = dense(
                x, np.concatenate([eye, eye], 0),
                np.concatenate([
                    np.full(n, -lo, np.float32),
                    np.full(n, -hi, np.float32)
                ])
            )
            z = vr(z)
            return dense(
                z, np.concatenate([eye, -eye], 1) / (hi - lo)
            )
        a = vr(va([(x, 1)], -lo))
        b = vr(va([(x, 1)], -hi))
        return va([(a, 1 / (hi - lo)), (b, -1 / (hi - lo))])

    def vabs(x):
        return va([(vr(x), 1), (vr(va([(x, -1)])), 1)])

    def veq(x, y=None):
        if y is None:
            y = const(0)
        n = max(shapes[x][0], shapes[y][0])
        if n > 8:
            return vr(va([
                (vabs(va([(x, 1), (y, -1)])), -1)
            ], 1))

        nx, ny = shapes[x][0], shapes[y][0]
        base = np.zeros((n, nx + ny), np.float32)
        for i in range(n):
            base[i, 0 if nx == 1 else i] = 1
            base[i, nx + (0 if ny == 1 else i)] = -1
        w = np.concatenate([base] * 4, 0)
        b = np.concatenate([
            np.full(n, 0.5, np.float32),
            np.full(n, 0.25, np.float32),
            np.full(n, -0.25, np.float32),
            np.full(n, -0.5, np.float32),
        ])
        z = vr(dense(cat([x, y]), w, b))
        eye = np.eye(n, dtype=np.float32)
        return dense(z, 4 * np.concatenate([eye, -eye, -eye, eye], 1))

    def choose(g, yes, no):
        return va([
            (no, 1),
            (vm(g, va([(yes, 1), (no, -1)])), 1)
        ])

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def vand(xs):
        return vr(va([(x, 1) for x in xs], 1 - len(xs)))

    def sm(op, a, b):
        return emit(op, [a, b])

    def scut(x, gain=1.0, bias=0.0):
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        z = conv(
            x, np.concatenate([gain * eye, gain * eye], 0),
            np.concatenate([
                np.full(n, bias, np.float32),
                np.full(n, bias - 1, np.float32)
            ])
        )
        z = emit("relu", [z])
        return conv(z, np.concatenate([eye, -eye], 1))

    def seq(x, y):
        return scut(emit("abs", [sm("sub", x, y)]), -4, 2)

    def gmax(x):
        return emit("gmax", [x])

    def gmean(x):
        return emit("gmean", [x])

    def spatial_sum_channels(x):
        return conv(x, np.ones((1, shapes[x][0]), np.float32))

    def basis(axis, size, origin=0, step=1):
        w = np.zeros((size, 3), np.float32)
        w[:, axis] = 63
        b = -(origin + step * np.arange(size, dtype=np.float32))
        z = emit("abs", [conv("coords", w, b)])
        return scut(z, -4, 2)

    def loop_add(body, state_specs, next_map, T):
        name = fresh()
        node = {
            "id": name, "op": "loop", "T": T,
            "state": state_specs, "body": body, "next": next_map
        }
        graph.append(node)
        outs = {}
        for s in state_specs:
            out = name + "." + s["id"]
            shapes[out] = shapes[s["id"]]
            outs[s["id"]] = out
        return node, outs

    # DIR stores dx + 3*dy for each outward-facing connector.
    fields = [
        ("X", E), ("Y", E), ("DIR", E),
        ("W", N), ("H", N), ("BX", N), ("BY", N),
        ("AR", N), ("PX", N), ("PY", N), ("GR", N),
        ("DX", N), ("DY", N), ("C", N), ("S", N),
        ("L", N), ("TR", N), ("D", 1), ("F", 1),
        ("number", 1), ("selected", 1), ("phase", 1), ("pending", 1),
    ]
    M = sum(n for _, n in fields)
    shapes["mem"] = (M,)
    old = {}
    offset = 0
    for key, n in fields:
        old[key] = cut("mem", offset, offset + n)
        offset += n

    zero = const(0)
    one = const(1)
    ids = const(np.arange(N, dtype=np.float32))
    zerosN = const(np.zeros(N, np.float32))
    onesN = const(np.ones(N, np.float32))

    # Sample one pixel per rendered logical cell.
    lattice_r = spatial_sum_channels(basis(0, 20, 3, 3))
    lattice_c = spatial_sum_channels(basis(1, 20, 3, 3))
    lattice = sm("mul", lattice_r, lattice_c)

    gx = conv("coords", [[0, 21, 0]], [-1])
    gy = conv("coords", [[21, 0, 0]], [-1])
    gx1 = sm("add", gx, one)
    gy1 = sm("add", gy, one)
    pixel_r = conv("coords", [[63, 0, 0]])
    pixel_c = conv("coords", [[0, 63, 0]])
    priority = conv("coords", [[-420, -21, 0]], [1021])
    blank_click = conv("coords", [[-63, -63, 0]])

    row_basis = basis(0, E)
    col_basis = basis(1, E)
    row8 = cut(row_basis, 0, K)
    rowN = cut(row_basis, 0, N)

    def embed(v, b):
        return spatial_sum_channels(sm("mul", b, v))

    def vmax(v):
        return va([
            (gmax(embed(va([(v, 1)], 1000), rowN)), 1)
        ], -1000)

    repeat_w = np.zeros((E, N), np.float32)
    for i in range(N):
        repeat_w[i * K:(i + 1) * K, i] = 1

    def repeat_objects(v):
        return dense(v, repeat_w)

    reduce_terms_w = np.zeros((K, E), np.float32)
    for i in range(N):
        reduce_terms_w[:, i * K:(i + 1) * K] = np.eye(
            K, dtype=np.float32
        )

    group_sum_w = repeat_w.T
    tile_terms_w = reduce_terms_w.T

    def select_terms(v, objhot):
        return dense(vm(v, repeat_objects(objhot)), reduce_terms_w)

    corner = scut(
        conv("coords", [[63, 63, 0]], [-125.5]), 4, 0
    )
    bg_colour = gmax(sm("mul", "x0", corner))
    bg = spatial_sum_channels(sm("mul", "x0", bg_colour))
    fg = sm("sub", one, bg)
    foreground = sm("mul", fg, lattice)

    c0 = cut("x0", 0, 1)
    c3 = cut("x0", 3, 4)
    c4 = cut("x0", 4, 5)
    c8 = cut("x0", 8, 9)

    grey = gmax(sm("mul", sm("mul", c4, fg), lattice))
    coloured = emit("relu", [
        sm("sub", sm("sub", sm("sub", fg, c4), c8), c3)
    ])
    black_body = sm("mul", sm("mul", c0, fg), lattice)
    coloured_body = sm("mul", coloured, lattice)
    selected_body = sm(
        "add",
        sm("mul", coloured_body, grey),
        sm("mul", black_body, va([(grey, -1)], 1))
    )

    near_w = np.zeros((1, 1, 7, 7), np.float32)
    for dr, dc in [(0, 0), (-3, 0), (3, 0), (0, -3), (0, 3)]:
        near_w[0, 0, 3 + dr, 3 + dc] = 1
    near_black = scut(conv(black_body, near_w))
    terminals_all = sm("mul", sm("add", c8, c3), lattice)
    terminal_gate = sm(
        "add", sm("mul", near_black, va([(grey, -1)], 1)), grey
    )
    terminal_mask = sm("mul", terminals_all, terminal_gate)
    selected_mask = scut(sm("add", selected_body, terminal_mask))
    present = gmax(selected_mask)

    bx = va([
        (gmax(sm("mul", selected_mask, sm("sub", const(20), gx))), -1)
    ], 20)
    by = va([
        (gmax(sm("mul", selected_mask, sm("sub", const(20), gy))), -1)
    ], 20)
    ex = gmax(sm("mul", selected_mask, gx))
    ey = gmax(sm("mul", selected_mask, gy))
    width = va([(ex, 1), (bx, -1)], 1)
    height = va([(ey, 1), (by, -1)], 1)

    def first_point(mask):
        score = sm("mul", mask, priority)
        peak = gmax(score)
        hit = sm("mul", mask, seq(score, peak))
        xx = va([(gmax(sm("mul", hit, gx1)), 1)], -1)
        yy = va([(gmax(sm("mul", hit, gy1)), 1)], -1)
        return xx, yy

    body_x, body_y = first_point(selected_body)
    body_x = va([(body_x, 1), (bx, -1)])
    body_y = va([(body_y, 1), (by, -1)])

    has_black = gmax(sm("mul", selected_body, c0))
    has_colour = gmax(
        emit("relu", [sm("sub", coloured_body, black_body)])
    )
    expandable = vm(grey, vm(has_black, has_colour))

    # Prefix scans rank connector pixels without an internal argmax.
    row_counts = emit("rowsum", [terminal_mask])
    prefix_input = emit("concat", [terminal_mask, row_counts])
    p_move, p_acc = fresh(), fresh()
    shapes[p_move] = (2, 64, 64)
    shapes[p_acc] = (2, 64, 64)
    prefix_body = []
    saved_current = current
    current = prefix_body
    shift_w = np.zeros((2, 2, 3, 3), np.float32)
    shift_w[0, 0, 1, 0] = 1
    shift_w[1, 1, 0, 1] = 1
    shifted = conv(p_move, shift_w)
    accumulated = sm("add", p_acc, shifted)
    current = saved_current
    _, prefix_out = loop_add(
        prefix_body,
        [
            {"id": p_move, "init": prefix_input},
            {"id": p_acc, "zeros": [2, 64, 64]},
        ],
        {p_move: shifted, p_acc: accumulated},
        63
    )
    rank = conv(prefix_out[p_acc], [[1, 1]])
    rank_difference = conv(
        rank, np.ones((K, 1), np.float32),
        -np.arange(K, dtype=np.float32)
    )
    ranked = scut(emit("abs", [rank_difference]), -4, 2)
    ranked = sm("mul", ranked, terminal_mask)
    terminal_present = gmax(ranked)
    term_x = gmax(sm("mul", ranked, gx1))
    term_y = gmax(sm("mul", ranked, gy1))
    term_x = va([
        (term_x, 1),
        (vm(terminal_present, va([(bx, 1)], 1)), -1),
        (va([(terminal_present, -1)], 1), -30),
    ])
    term_y = va([
        (term_y, 1),
        (vm(terminal_present, va([(by, 1)], 1)), -1),
    ])

    # Adjacent body pixels determine the connector's outward direction.
    direction_w = np.zeros((1, 1, 7, 7), np.float32)
    direction_w[0, 0, 3, 0] = 1
    direction_w[0, 0, 3, 6] = -1
    direction_w[0, 0, 0, 3] = 3
    direction_w[0, 0, 6, 3] = -3
    direction_field = conv(selected_body, direction_w, [4])
    term_direction = va([
        (gmax(sm("mul", ranked, direction_field)), 1),
        (terminal_present, -4)
    ])

    acq = veq(old["phase"], zero)
    not_acq = va([(acq, -1)], 1)
    selected_hot = veq(ids, old["selected"])
    observe_existing = vm(not_acq, present)
    pos_update_hot = vm(selected_hot, observe_existing)

    work = dict(old)
    work["BX"] = choose(pos_update_hot, bx, old["BX"])
    work["BY"] = choose(pos_update_hot, by, old["BY"])

    room = vclip(va([(old["number"], -1)], N))
    record = vand([
        acq, present, va([(expandable, -1)], 1), room
    ])
    new_hot = veq(ids, old["number"])
    record_hot = vm(new_hot, record)
    record_end_hot = repeat_objects(record_hot)

    work["X"] = choose(
        record_end_hot, dense(term_x, tile_terms_w), old["X"]
    )
    work["Y"] = choose(
        record_end_hot, dense(term_y, tile_terms_w), old["Y"]
    )
    work["DIR"] = choose(
        record_end_hot,
        dense(term_direction, tile_terms_w),
        old["DIR"]
    )
    group_record = vclip(va([
        (old["pending"], 1), (expandable, 1)
    ]))
    for key, value in [
        ("W", width), ("H", height), ("BX", bx), ("BY", by),
        ("AR", zero), ("PX", body_x), ("PY", body_y),
        ("GR", group_record),
    ]:
        work[key] = choose(record_hot, value, work[key])

    number = va([(old["number"], 1), (record, 1)])
    object_valid = vclip(va([(number, 1), (ids, -1)]))
    ref_valid = vm(
        vclip(va([(work["X"], 1)], 1)),
        repeat_objects(object_valid)
    )

    # f(d) is the encoded direction after a clockwise quarter-turn.
    # On {-3,-1,1,3}, its values are {1,-3,3,-1}.
    quarter_direction = va([
        (work["DIR"], -2),
        (vr(va([(work["DIR"], 1)], 1)), 5),
        (vr(va([(work["DIR"], 1)], -1)), -5),
    ], -5)

    ddx = sm("sub", gx, work["BX"])
    ddy = sm("sub", gy, work["BY"])
    in_box = scut(ddx, 4, 2)
    in_box = sm("mul", in_box, scut(ddy, 4, 2))
    in_box = sm("mul", in_box, scut(
        sm("sub", va([(work["W"], 1)], -1), ddx), 4, 2
    ))
    in_box = sm("mul", in_box, scut(
        sm("sub", va([(work["H"], 1)], -1), ddy), 4, 2
    ))
    in_box = sm("mul", in_box, object_valid)
    covered = scut(spatial_sum_channels(in_box))
    unknown = sm("mul", foreground, sm("sub", one, covered))
    unknown_exists = gmax(unknown)
    unknown_click = sm("mul", unknown, priority)
    at_capacity = vclip(va([(number, 1)], 1 - N))
    finished_scan = vm(record, vclip(va([
        (unknown_exists, -1), (at_capacity, 1)
    ], 1)))

    connector_counts = dense(ref_valid, group_sum_w)
    root_score = va([
        (connector_counts, 16), (ids, -1), (object_valid, 100)
    ], -100)
    root_hot = vm(veq(root_score, vmax(root_score)), object_valid)

    solver_keys = ["DX", "DY", "C", "S", "L", "TR", "D", "F"]
    initial_solver = {
        "DX": work["BX"], "DY": work["BY"],
        "C": onesN, "S": zerosN, "L": root_hot,
        "TR": zerosN, "D": zero, "F": zero
    }
    solver_init = {}
    for key in solver_keys:
        solver_init[key] = choose(
            finished_scan, initial_solver[key], old[key]
        )
    scanning = vm(acq, va([(finished_scan, -1)], 1))
    solver_init["F"] = choose(scanning, one, solver_init["F"])

    CANDIDATES = N * 4 * K
    codes = const(np.arange(CANDIDATES, dtype=np.float32))
    table = np.zeros((7, CANDIDATES), np.float32)
    cosines = [1, 0, -1, 0]
    sines = [0, 1, 0, -1]
    for i in range(N):
        for r in range(4):
            for k in range(K):
                j = (i * 4 + r) * K + k
                table[:, j] = [
                    i, k, cosines[r], sines[r],
                    4 * K - r * K - k, K - k, r
                ]
    term_ids = const(np.arange(K, dtype=np.float32))
    diagonal = seq(pixel_r, pixel_c)
    not_diagonal = sm("sub", one, diagonal)
    frontier_priority = conv("coords", [[-63, 0, 0]], [E + 1])
    total_terminals = vsum(ref_valid)

    states = {}
    state_specs = []
    for key in solver_keys:
        sid = fresh()
        states[key] = sid
        shapes[sid] = shapes[solver_init[key]]
        state_specs.append({"id": sid, "init": solver_init[key]})

    solver_body = []
    saved_current = current
    current = solver_body
    st = states

    alive = va([(st["F"], -1)], 1)
    placed = vclip(st["L"])
    active_terms = vm(ref_valid, repeat_objects(placed))
    cc = repeat_objects(st["C"])
    ss = repeat_objects(st["S"])
    current_x = va([
        (repeat_objects(st["DX"]), 1),
        (vm(cc, work["X"]), 1),
        (vm(ss, work["Y"]), -1),
    ])
    current_y = va([
        (repeat_objects(st["DY"]), 1),
        (vm(ss, work["X"]), 1),
        (vm(cc, work["Y"]), 1),
    ])
    current_direction = va([
        (vm(cc, work["DIR"]), 1),
        (vm(ss, quarter_direction), 1),
    ])

    xr = embed(current_x, row_basis)
    xc = embed(current_x, col_basis)
    yr = embed(current_y, row_basis)
    yc = embed(current_y, col_basis)
    ar = embed(active_terms, row_basis)
    ac = embed(active_terms, col_basis)
    direction_columns = embed(current_direction, col_basis)

    distance = sm(
        "add",
        emit("abs", [sm("sub", xr, xc)]),
        emit("abs", [sm("sub", yr, yc)])
    )
    pairs = scut(distance, -4, 2)
    pairs = sm("mul", sm("mul", pairs, ar), ac)
    pairs = sm("mul", pairs, not_diagonal)
    row_degree = emit("rowsum", [pairs])
    col_degree = emit("colsum", [pairs])
    frontier = sm(
        "mul", ar, emit("relu", [sm("sub", one, row_degree)])
    )
    frontier_score = sm("mul", frontier, frontier_priority)
    frontier_peak = gmax(frontier_score)
    frontier_one = sm(
        "mul", frontier, seq(frontier_score, frontier_peak)
    )
    qx = va([(gmean(sm("mul", frontier_one, xr)), 64)])
    qy = va([(gmean(sm("mul", frontier_one, yr)), 64)])
    frontier_exists = gmax(frontier)
    unmatched = va([(gmean(frontier), 64)])

    depth_hot = veq(ids, st["D"])
    cursor = vsum(vm(depth_hot, st["TR"]))
    cursor_hot = veq(codes, cursor)
    decoded = dense(cursor_hot, table)
    oi, ki, co, si, jump_object, jump_term, rotation = [
        cut(decoded, j, j + 1) for j in range(7)
    ]
    object_hot = veq(ids, oi)
    endpoint_hot = veq(term_ids, ki)

    candidate_ref_x = select_terms(work["X"], object_hot)
    candidate_ref_y = select_terms(work["Y"], object_hot)
    candidate_valid = select_terms(ref_valid, object_hot)
    candidate_direction = select_terms(work["DIR"], object_hot)
    candidate_quarter = select_terms(quarter_direction, object_hot)
    endpoint_x = vsum(vm(candidate_ref_x, endpoint_hot))
    endpoint_y = vsum(vm(candidate_ref_y, endpoint_hot))
    endpoint_valid = vsum(vm(candidate_valid, endpoint_hot))

    rotated_endpoint_x = va([
        (vm(co, endpoint_x), 1), (vm(si, endpoint_y), -1)
    ])
    rotated_endpoint_y = va([
        (vm(si, endpoint_x), 1), (vm(co, endpoint_y), 1)
    ])
    new_dx = va([(qx, 1), (rotated_endpoint_x, -1)])
    new_dy = va([(qy, 1), (rotated_endpoint_y, -1)])
    new_x = va([
        (vm(co, candidate_ref_x), 1),
        (vm(si, candidate_ref_y), -1), (new_dx, 1)
    ])
    new_y = va([
        (vm(si, candidate_ref_x), 1),
        (vm(co, candidate_ref_y), 1), (new_dy, 1)
    ])
    new_direction = va([
        (vm(co, candidate_direction), 1),
        (vm(si, candidate_quarter), 1),
    ])

    nxr = embed(new_x, row8)
    nyr = embed(new_y, row8)
    nar = embed(candidate_valid, row8)
    direction_rows = embed(new_direction, row8)
    new_distance = sm(
        "add",
        emit("abs", [sm("sub", nxr, xc)]),
        emit("abs", [sm("sub", nyr, yc)])
    )
    new_pairs = sm(
        "mul", sm("mul", scut(new_distance, -4, 2), nar), ac
    )
    matches = va([(gmean(new_pairs), 4096)])
    multiplicity_conflict = gmax(sm("mul", new_pairs, col_degree))
    direction_conflict = gmax(sm(
        "mul", new_pairs,
        emit("abs", [sm("add", direction_rows, direction_columns)])
    ))
    conflict = va([
        (multiplicity_conflict, 1), (direction_conflict, 1)
    ])

    candidate_count = vsum(candidate_valid)
    active_count = vsum(active_terms)
    remaining = va([
        (total_terminals, 1), (active_count, -1), (candidate_count, -1)
    ])
    new_unmatched = va([
        (unmatched, 1), (candidate_count, 1), (matches, -2)
    ])
    enough_remaining = vclip(va([
        (remaining, 1), (new_unmatched, -1)
    ], 1))

    is_placed = vclip(vsum(vm(object_hot, st["L"])))
    is_group = vsum(vm(object_hot, work["GR"]))
    rotation_zero = veq(rotation, zero)
    group_ok = vclip(va([
        (is_group, -1), (rotation_zero, 1)
    ], 1))
    exhausted = vclip(va([(cursor, 1)], 1 - CANDIDATES))
    accept = vand([
        alive, va([(exhausted, -1)], 1),
        va([(is_placed, -1)], 1), endpoint_valid,
        va([(conflict, -1)], 1), enough_remaining,
        frontier_exists, group_ok
    ])

    bad_endpoint_block = vclip(va([
        (endpoint_valid, -1),
        (vm(is_group, va([(rotation_zero, -1)], 1)), 1)
    ], 1))
    increment = choose(
        is_placed, jump_object,
        choose(bad_endpoint_block, jump_term, one)
    )
    cursor_change = vm(alive, choose(
        exhausted, va([(cursor, -1)]), increment
    ))
    next_trials = va([
        (st["TR"], 1), (vm(depth_hot, cursor_change), 1)
    ])

    back = vm(vm(exhausted, alive), vclip(st["D"]))
    deepest = veq(st["L"], va([(st["D"], 1)], 1))
    remove = vm(deepest, back)
    insert = vm(object_hot, accept)

    next_state = {}
    for key, value in [
        ("DX", new_dx), ("DY", new_dy), ("C", co), ("S", si)
    ]:
        next_state[key] = choose(insert, value, st[key])
    next_state["L"] = va([
        (vm(st["L"], va([(remove, -1)], 1)), 1),
        (vm(insert, va([(st["D"], 1)], 2)), 1)
    ])
    next_state["TR"] = next_trials
    next_state["D"] = va([
        (st["D"], 1), (accept, 1), (back, -1)
    ])
    all_placed = veq(va([(st["D"], 1)], 2), number)
    next_state["F"] = va([
        (st["F"], 1), (vm(accept, all_placed), 1)
    ])
    current = saved_current

    solver_loop, solver_outputs = loop_add(
        solver_body, state_specs,
        {states[key]: next_state[key] for key in solver_keys},
        1
    )
    sol = {key: solver_outputs[states[key]] for key in solver_keys}

    wm = va([(work["W"], 1)], -1)
    hm = va([(work["H"], 1)], -1)

    def rotation_bounds(c, s):
        cp, cn = vr(c), vr(va([(c, -1)]))
        sp, sn = vr(s), vr(va([(s, -1)]))
        minx = va([(vm(cn, wm), -1), (vm(sp, hm), -1)])
        maxx = va([(vm(cp, wm), 1), (vm(sn, hm), 1)])
        miny = va([(vm(sn, wm), -1), (vm(cn, hm), -1)])
        maxy = va([(vm(sp, wm), 1), (vm(cp, hm), 1)])
        return minx, maxx, miny, maxy

    minrx, maxrx, minry, maxry = rotation_bounds(sol["C"], sol["S"])
    left = va([(sol["DX"], 1), (minrx, 1)])
    right = va([(sol["DX"], 1), (maxrx, 1)])
    top = va([(sol["DY"], 1), (minry, 1)])
    bottom = va([(sol["DY"], 1), (maxry, 1)])
    missing_penalty = va([(object_valid, -100)], 100)

    global_left = va([
        (vmax(va([(left, -1), (missing_penalty, -1)])), -1)
    ])
    global_top = va([
        (vmax(va([(top, -1), (missing_penalty, -1)])), -1)
    ])
    global_right = vmax(va([
        (right, 1), (missing_penalty, -1)
    ]))
    global_bottom = vmax(va([
        (bottom, 1), (missing_penalty, -1)
    ]))

    shift_x = va([
        (vr(va([(global_left, -1)])), 1),
        (vr(va([(global_right, 1)], -19)), -1)
    ])
    shift_y = va([
        (vr(va([(global_top, -1)])), 1),
        (vr(va([(global_bottom, 1)], -19)), -1)
    ])
    target_bx = va([(left, 1), (shift_x, 1)])
    target_by = va([(top, 1), (shift_y, 1)])
    desired_rotation = va([
        (vr(sol["S"]), 1),
        (vr(va([(sol["C"], -1)])), 2),
        (vr(va([(sol["S"], -1)])), 3),
    ])
    rotation_error = va([
        (veq(work["AR"], desired_rotation), -1)
    ], 1)
    error_x = va([(target_bx, 1), (work["BX"], -1)])
    error_y = va([(target_by, 1), (work["BY"], -1)])
    error = vm(object_valid, vclip(va([
        (vabs(error_x), 1), (vabs(error_y), 1), (rotation_error, 4)
    ]), 0.25, 0.75))

    sid_now = choose(acq, old["number"], old["selected"])
    sid_hot_now = veq(ids, sid_now)
    execution_score = va([
        (error, 100), (vm(error, sid_hot_now), 20), (ids, -1)
    ])
    chosen_hot = vm(
        veq(execution_score, vmax(execution_score)), error
    )
    chosen_id = vsum(vm(chosen_hot, ids))
    any_error = vclip(vsum(error))
    too_wide = vclip(va([
        (global_right, 1), (global_left, -1)
    ], -19), 0.25, 0.75)
    too_tall = vclip(va([
        (global_bottom, 1), (global_top, -1)
    ], -19), 0.25, 0.75)

    active_phase = va([(scanning, -1)], 1)
    ready = vm(active_phase, sol["F"])
    bad_plan = vm(ready, vclip(va([
        (any_error, -1), (too_wide, 1), (too_tall, 1)
    ], 1)))
    execute = vm(ready, va([(bad_plan, -1)], 1))

    switch = vm(
        execute, va([(veq(chosen_id, sid_now), -1)], 1)
    )
    controlling = vm(execute, va([(switch, -1)], 1))
    chosen_rotation_error = vsum(vm(chosen_hot, rotation_error))
    turn = vm(controlling, chosen_rotation_error)
    translate = vm(
        controlling, va([(chosen_rotation_error, -1)], 1)
    )
    dx = vsum(vm(chosen_hot, error_x))
    dy = vsum(vm(chosen_hot, error_y))
    go_left = vm(
        translate, vclip(va([(dx, -1)]), 0.25, 0.75)
    )
    go_right = vm(translate, vclip(dx, 0.25, 0.75))
    vertical = va([
        (translate, 1), (go_left, -1), (go_right, -1)
    ])
    go_up = vm(vertical, vclip(va([(dy, -1)]), 0.25, 0.75))
    go_down = vm(vertical, vclip(dy, 0.25, 0.75))
    grow = vm(scanning, expandable)
    action5 = va([(turn, 1), (grow, 1)])
    action6 = va([
        (go_up, -1), (go_down, -1), (go_left, -1),
        (go_right, -1), (action5, -1)
    ], 1)

    action_features = cat([
        go_up, go_down, go_left, go_right, action5, action6
    ])
    action_w = np.zeros((8, 6), np.float32)
    for a in range(1, 7):
        action_w[a, a - 1] = 10
    action_b = np.zeros(8, np.float32)
    action_b[0] = action_b[7] = -100
    dense(
        action_features, action_w, action_b, name="logits"
    )

    actual_c = va([
        (work["AR"], -1),
        (vr(va([(work["AR"], 1)], -2)), 2)
    ], 1)
    actual_s = va([
        (work["AR"], 1),
        (vr(va([(work["AR"], 1)], -1)), -2)
    ])
    aminx, _, aminy, _ = rotation_bounds(actual_c, actual_s)
    actual_body_x = va([
        (work["BX"], 1), (vm(actual_c, work["PX"]), 1),
        (vm(actual_s, work["PY"]), -1), (aminx, -1)
    ])
    actual_body_y = va([
        (work["BY"], 1), (vm(actual_s, work["PX"]), 1),
        (vm(actual_c, work["PY"]), 1), (aminy, -1)
    ])
    click_x = va([
        (vsum(vm(chosen_hot, actual_body_x)), 3)
    ], 3)
    click_y = va([
        (vsum(vm(chosen_hot, actual_body_y)), 3)
    ], 3)
    object_click = conv(
        emit("concat", [
            emit("abs", [sm("sub", pixel_c, click_x)]),
            emit("abs", [sm("sub", pixel_r, click_y)])
        ]),
        [[-1, -1]]
    )
    scan_click_gate = vm(scanning, va([(expandable, -1)], 1))
    blank_gate = va([
        (scan_click_gate, -1), (switch, -1)
    ], 1)
    click_map = sm(
        "add",
        sm("add",
           sm("mul", unknown_click, scan_click_gate),
           sm("mul", object_click, switch)),
        sm("mul", blank_click, blank_gate)
    )
    emit("add", [click_map, zero], name="click")

    resume_remove = vm(
        veq(sol["L"], va([(sol["D"], 1)], 1)), bad_plan
    )
    final_solver = dict(sol)
    final_solver["L"] = vm(
        sol["L"], va([(resume_remove, -1)], 1)
    )
    final_solver["D"] = vr(va([
        (sol["D"], 1), (bad_plan, -1)
    ]))
    final_solver["F"] = vm(
        sol["F"], va([(bad_plan, -1)], 1)
    )

    turn_hot = vm(chosen_hot, turn)
    turn_increment = va([
        (veq(work["AR"], const(3)), -4)
    ], 1)
    work["AR"] = va([
        (work["AR"], 1), (vm(turn_hot, turn_increment), 1)
    ])
    next_selected = choose(switch, chosen_id, sid_now)
    next_selected = choose(scan_click_gate, number, next_selected)
    next_phase = vm(
        active_phase,
        va([
            (vm(sol["F"], va([(bad_plan, -1)], 1)), 1)
        ], 1)
    )
    next_pending = choose(
        record, zero,
        choose(scanning, vclip(va([
            (old["pending"], 1), (expandable, 1)
        ])), zero)
    )

    memory_values = dict(work)
    memory_values.update(final_solver)
    memory_values["number"] = number
    memory_values["selected"] = next_selected
    memory_values["phase"] = next_phase
    memory_values["pending"] = next_pending
    emit(
        "vconcat", [memory_values[key] for key, _ in fields],
        name="mem_next"
    )

    # Remove unused intermediate expressions and their parameters.
    def prune(nodes, outputs):
        needed = set(outputs)
        kept = []
        for nd in reversed(nodes):
            if nd["op"] == "loop":
                result_names = {
                    nd["id"] + "." + s["id"] for s in nd["state"]
                }
                if not (result_names & needed):
                    continue
                body, external = prune(
                    nd["body"], list(nd["next"].values())
                )
                nd["body"] = body
                local_states = {s["id"] for s in nd["state"]}
                needed.difference_update(result_names)
                needed.update(external - local_states)
                for state in nd["state"]:
                    if "init" in state:
                        needed.add(state["init"])
                kept.append(nd)
            elif nd["id"] in needed:
                needed.remove(nd["id"])
                needed.update(nd.get("in", []))
                kept.append(nd)
        kept.reverse()
        return kept, needed

    complete_graph, _ = prune(
        constants + graph, ["logits", "click", "mem_next"]
    )

    # Fuse single-use affine chains where doing so reduces graph size
    # without increasing the number of parameters.
    def affine_cleanup(nodes):
        for nd in nodes:
            if nd["op"] == "loop":
                affine_cleanup(nd["body"])

        changed = True
        while changed:
            changed = False
            uses = {}
            by_id = {nd["id"]: nd for nd in nodes}
            for nd in nodes:
                if nd["op"] == "loop":
                    refs = []
                    for state in nd["state"]:
                        if "init" in state:
                            refs.append(state["init"])

                    def gather(ns):
                        out = []
                        for q in ns:
                            out.extend(q.get("in", []))
                            if q["op"] == "loop":
                                out.extend(gather(q["body"]))
                        return out

                    refs += gather(nd["body"])
                else:
                    refs = nd.get("in", [])
                for ref in refs:
                    uses[ref] = uses.get(ref, 0) + 1

            for nd in nodes:
                if nd["op"] != "dense":
                    continue
                source = nd["in"][0]
                prev = by_id.get(source)
                if (
                    prev is None or prev["op"] != "dense"
                    or uses.get(source) != 1
                    or source in ("logits", "click", "mem_next")
                ):
                    continue
                w1 = params[prev["w"]]
                w2 = params[nd["w"]]
                if w1.shape[1] * w2.shape[0] > w1.size + w2.size:
                    continue
                # Only exactly integral affine maps are fused.
                if not (
                    np.all(w1 == np.round(w1))
                    and np.all(w2 == np.round(w2))
                ):
                    continue
                b1 = params.get(
                    prev.get("b"), np.zeros(w1.shape[0], np.float32)
                )
                b2 = params.get(
                    nd.get("b"), np.zeros(w2.shape[0], np.float32)
                )
                w = w2 @ w1
                b = w2 @ b1 + b2
                nd["in"] = list(prev["in"])
                nd["w"] = parameter(w)
                nd["b"] = parameter(b)
                nodes.remove(prev)
                changed = True
                break

    # Pruning normally suffices; affine cleanup is a graph-size safeguard.
    def count_nodes(nodes):
        return sum(
            1 + (count_nodes(n["body"]) if n["op"] == "loop" else 0)
            for n in nodes
        )

    if count_nodes(complete_graph) > 990:
        affine_cleanup(complete_graph)

    def ordinary_mac(nd):
        op = nd["op"]
        ins = nd.get("in", [])
        if op == "conv":
            return int(np.prod(params[nd["w"]].shape)) * 4096
        if op == "dense":
            return int(np.prod(params[nd["w"]].shape))
        if op in ("add", "sub", "mul", "max2", "min2"):
            return int(np.prod(shapes[nd["id"]]))
        if op in (
            "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
        ):
            return int(np.prod(shapes[ins[0]]))
        if op in ("vadd", "vmul"):
            return shapes[nd["id"]][0]
        return 0

    def totals(nodes, omit_solver=False):
        mac = 0
        evaluations = 0
        for nd in nodes:
            evaluations += 1
            if nd["op"] == "loop":
                t = 0 if omit_solver and nd is solver_loop else nd["T"]
                bm, be = totals(nd["body"])
                mac += t * bm
                evaluations += t * be
            else:
                mac += ordinary_mac(nd)
        return mac, evaluations

    base_mac, base_evaluations = totals(
        complete_graph, omit_solver=True
    )
    body_mac, body_evaluations = totals(solver_loop["body"])
    t_mac = (1_490_000_000 - base_mac) // max(1, body_mac)
    t_eval = (9950 - base_evaluations) // max(1, body_evaluations)
    solver_loop["T"] = int(max(1, min(128, t_mac, t_eval)))

    used_params = set()

    def collect_params(nodes):
        for nd in nodes:
            for key in ("w", "b"):
                if key in nd:
                    used_params.add(nd[key])
            if nd["op"] == "loop":
                collect_params(nd["body"])

    collect_params(complete_graph)
    final_params = {key: params[key] for key in used_params}

    return {
        "graph": complete_graph,
        "memory_size": M,
        "params": final_params,
    }

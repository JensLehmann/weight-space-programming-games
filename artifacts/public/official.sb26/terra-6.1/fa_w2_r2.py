import numpy as np


def build():
    graph = []
    params = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", 4),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    serial = [0]
    target = [graph]

    def fresh(prefix="n"):
        serial[0] += 1
        return f"{prefix}{serial[0]}"

    def par(a):
        k = fresh("p")
        params[k] = np.asarray(a, dtype=np.float32)
        return k

    def emit(op, *inputs, **fields):
        k = fresh()
        target[0].append({"id": k, "op": op, "in": list(inputs), **fields})
        if op == "conv":
            shapes[k] = ("s", int(params[fields["w"]].shape[0]))
        elif op == "dense":
            shapes[k] = ("v", int(params[fields["w"]].shape[0]))
        elif op in ("gmax", "gmean"):
            shapes[k] = ("v", shapes[inputs[0]][1])
        elif op in ("slice", "vslice"):
            shapes[k] = (
                shapes[inputs[0]][0],
                fields["stop"] - fields["start"],
            )
        elif op in ("concat", "vconcat"):
            shapes[k] = (
                "s" if op == "concat" else "v",
                sum(shapes[x][1] for x in inputs),
            )
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shapes[k] = ("s", max(shapes[x][1] for x in inputs))
        elif op in ("vadd", "vmul"):
            shapes[k] = ("v", max(shapes[x][1] for x in inputs))
        else:
            shapes[k] = shapes[inputs[0]]
        return k

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return emit("conv", x, **fields)

    def dense(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return emit("dense", x, **fields)

    def affine(x, scale=1.0, bias=0.0):
        kind, n = shapes[x]
        if kind == "s":
            w = (scale * np.eye(n, dtype=np.float32))[:, :, None, None]
            return conv(x, w, np.full(n, bias, dtype=np.float32))
        w = scale * np.eye(n, dtype=np.float32)
        return dense(x, w, np.full(n, bias, dtype=np.float32))

    def add(a, b):
        kind = "add" if "s" in (shapes[a][0], shapes[b][0]) else "vadd"
        return emit(kind, a, b)

    def mul(a, b):
        kind = "mul" if "s" in (shapes[a][0], shapes[b][0]) else "vmul"
        return emit(kind, a, b)

    def neg(x):
        return affine(x, -1.0, 0.0)

    def sub(a, b):
        if "s" in (shapes[a][0], shapes[b][0]):
            return emit("sub", a, b)
        return emit("vadd", a, neg(b))

    def summation(xs):
        y = xs[0]
        for x in xs[1:]:
            y = add(y, x)
        return y

    def relu(x):
        return emit("relu" if shapes[x][0] == "s" else "vrelu", x)

    def absolute(x):
        if shapes[x][0] == "s":
            return emit("abs", x)
        return add(relu(x), relu(neg(x)))

    def clip01(x):
        return sub(x, relu(affine(x, 1.0, -1.0)))

    def spatial_sum(x):
        return conv(
            x,
            np.ones((1, shapes[x][1], 1, 1), dtype=np.float32),
        )

    def vector_sum(x):
        return dense(
            x,
            np.ones((1, shapes[x][1]), dtype=np.float32),
        )

    def near_zero(x):
        d = relu(affine(absolute(x), 1.0, -0.01))
        return relu(affine(d, -2.0, 1.0))

    def equal_scalar(x, value):
        return near_zero(affine(x, 1.0, -float(value)))

    def rank_channels(x, n, first=1):
        w = np.ones((n, 1, 1, 1), dtype=np.float32)
        b = -np.arange(first, first + n, dtype=np.float32)
        return near_zero(conv(x, w, b))

    def vector_onehot_code(x):
        w = np.ones((16, 1), dtype=np.float32)
        b = -np.arange(1, 17, dtype=np.float32)
        return near_zero(dense(x, w, b))

    # Goal outline corners.
    gw = np.zeros((16, 16, 3, 3), dtype=np.float32)
    gb = np.full(16, -5.0, dtype=np.float32)
    for c in range(16):
        gw[c, c, 1, 1] += 1
        gw[c, c, 1, 2] += 1
        gw[c, c, 2, 1] += 1
        gw[c, 5, 1, 0] += 1
        gw[c, 5, 0, 1] += 1
        gw[c, 5, 2, 2] += 1
    gb[5] = -100.0
    goals = relu(conv("x0", gw, gb))
    goal_mask = spatial_sum(goals)

    # Solid tiles and outlined call tiles.
    sw = np.zeros((16, 16, 7, 7), dtype=np.float32)
    rw = np.zeros((16, 16, 7, 7), dtype=np.float32)
    sb = np.full(16, -6.0, dtype=np.float32)
    rb = np.full(16, -6.0, dtype=np.float32)
    for c in range(16):
        for w in (sw, rw):
            w[c, c, 2, 2] += 1
            w[c, c, 2, 3] += 1
            w[c, c, 3, 2] += 1
            w[c, 4, 2, 1] += 1
            w[c, 4, 1, 2] += 1
            w[c, 4, 6, 2] += 1
        sw[c, c, 3, 3] += 1
        rw[c, 4, 3, 3] += 1
    sw[2, 2, 4, 4] += 1
    sb[2] = -7.0
    sb[4] = -100.0
    rb[4] = -100.0
    solids = relu(conv("x0", sw, sb))
    rings = relu(conv("x0", rw, rb))

    # Empty-slot dots, aligned with the tile detections.
    dw = np.zeros((1, 16, 7, 7), dtype=np.float32)
    for yy, xx in ((2, 2), (2, 3), (3, 2), (3, 3)):
        dw[0, 2, yy, xx] += 1
    for yy, xx in ((2, 1), (1, 2), (6, 2), (4, 4)):
        dw[0, 4, yy, xx] += 1
    dots = relu(conv("x0", dw, np.array([-7.0], dtype=np.float32)))
    shift_diag = np.zeros((1, 1, 3, 3), dtype=np.float32)
    shift_diag[0, 0, 2, 2] = 1.0
    dots = conv(dots, shift_diag)

    all_tiles = summation([spatial_sum(solids), spatial_sum(rings), dots])
    row = emit("slice", "coords", start=0, stop=1)
    col = emit("slice", "coords", start=1, stop=2)
    one = emit("slice", "coords", start=2, stop=3)
    row_num = affine(row, 63.0, 0.0)
    col_num = affine(col, 63.0, 0.0)

    lower = sub(
        relu(affine(row, 63.0, -7.0)),
        relu(affine(row, 63.0, -8.0)),
    )
    upper = sub(
        relu(affine(row, -63.0, 53.0)),
        relu(affine(row, -63.0, 52.0)),
    )
    program_region = mul(lower, upper)
    palette_region = sub(
        relu(affine(row, 63.0, -53.0)),
        relu(affine(row, 63.0, -54.0)),
    )

    program = mul(all_tiles, program_region)
    empty_program = mul(dots, program_region)
    solid_palette = mul(solids, palette_region)
    ring_palette = mul(rings, palette_region)

    # Read the enclosing frame colour above each slot.
    up3 = np.zeros((16, 16, 7, 7), dtype=np.float32)
    left3 = np.zeros((16, 16, 7, 7), dtype=np.float32)
    up1 = np.zeros((16, 16, 3, 3), dtype=np.float32)
    left1 = np.zeros((16, 16, 3, 3), dtype=np.float32)
    for c in range(16):
        up3[c, c, 0, 3] = 1.0
        left3[c, c, 3, 0] = 1.0
        up1[c, c, 0, 1] = 1.0
        left1[c, c, 1, 0] = 1.0
    frame_top = conv(conv("x0", up3), up1)
    frame_left = conv(conv("x0", left3), left1)

    # At the first slot, the frame edge is four pixels left and the
    # pixel immediately outside that edge is background. This prevents
    # a same-colour preceding tile from being mistaken for a frame edge.
    background = emit("slice", "x0", start=4, stop=5)
    bg_left3 = np.zeros((1, 1, 7, 7), dtype=np.float32)
    bg_left2 = np.zeros((1, 1, 5, 5), dtype=np.float32)
    bg_left3[0, 0, 3, 0] = 1.0
    bg_left2[0, 0, 2, 0] = 1.0
    outside_left = conv(conv(background, bg_left3), bg_left2)
    frame_starts = mul(
        mul(program, spatial_sum(mul(frame_top, frame_left))),
        outside_left,
    )

    # Raster-order prefix counts.
    grow = emit("rowsum", goal_mask)
    srow = emit("rowsum", program)
    frow = emit("rowsum", frame_starts)
    wl = np.zeros((1, 1, 3, 3), dtype=np.float32)
    wu = np.zeros((1, 1, 3, 3), dtype=np.float32)
    wl[0, 0, 1, 0] = 1.0
    wu[0, 0, 0, 1] = 1.0
    pl = par(wl)
    pu = par(wu)

    graph.append({
        "id": "prefix",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": "gp", "zeros": [1, 64, 64]},
            {"id": "gv", "zeros": [1, 64, 64]},
            {"id": "sp", "zeros": [1, 64, 64]},
            {"id": "sv", "zeros": [1, 64, 64]},
            {"id": "fp", "zeros": [1, 64, 64]},
            {"id": "fv", "zeros": [1, 64, 64]},
        ],
        "body": [
            {"id": "gpc", "op": "conv", "in": ["gp"], "w": pl},
            {"id": "gpn", "op": "add", "in": ["gpc", goal_mask]},
            {"id": "gvc", "op": "conv", "in": ["gv"], "w": pu},
            {"id": "gvn", "op": "add", "in": ["gvc", grow]},
            {"id": "spc", "op": "conv", "in": ["sp"], "w": pl},
            {"id": "spn", "op": "add", "in": ["spc", program]},
            {"id": "svc", "op": "conv", "in": ["sv"], "w": pu},
            {"id": "svn", "op": "add", "in": ["svc", srow]},
            {"id": "fpc", "op": "conv", "in": ["fp"], "w": pl},
            {"id": "fpn", "op": "add", "in": ["fpc", frame_starts]},
            {"id": "fvc", "op": "conv", "in": ["fv"], "w": pu},
            {"id": "fvn", "op": "add", "in": ["fvc", frow]},
        ],
        "next": {
            "gp": "gpn",
            "gv": "gvn",
            "sp": "spn",
            "sv": "svn",
            "fp": "fpn",
            "fv": "fvn",
        },
    })
    for s in ("gp", "gv", "sp", "sv", "fp", "fv"):
        shapes["prefix." + s] = ("s", 1)

    grank = add("prefix.gp", sub("prefix.gv", grow))
    srank = add("prefix.sp", sub("prefix.sv", srow))
    frank = add("prefix.fp", sub("prefix.fv", frow))
    goal_gates = mul(rank_channels(grank, 20), goal_mask)
    slot_gates = mul(rank_channels(srank, 16), program)

    code_w = np.arange(1, 17, dtype=np.float32).reshape(1, 16, 1, 1)
    goal_map = conv(goals, code_w)
    solid_map = conv(solids, code_w)
    ring_map = conv(rings, code_w)
    frame_map = conv(frame_top, code_w)

    goal_codes = emit("gmax", mul(goal_gates, goal_map))
    goal_count = emit("gmax", mul(grank, goal_mask))
    slot_exists = emit("gmax", slot_gates)
    slot_empty = emit("gmax", mul(slot_gates, empty_program))
    slot_solid = emit("gmax", mul(slot_gates, solid_map))
    slot_ring = emit("gmax", mul(slot_gates, ring_map))
    slot_frame = emit("gmax", mul(slot_gates, frank))

    shift_vector = np.zeros((16, 16), dtype=np.float32)
    for i in range(15):
        shift_vector[i, i + 1] = 1.0
    next_frame = dense(slot_frame, shift_vector)
    slot_end = mul(
        clip01(absolute(sub(slot_frame, next_frame))),
        slot_exists,
    )

    def frame_info(number):
        mask = mul(equal_scalar(slot_frame, number), slot_exists)
        count = vector_sum(mask)
        weighted = mul(slot_gates, mask)
        priority = mul(weighted, affine(srank, -1.0, 17.0))
        priority_map = spatial_sum(priority)
        best = emit("gmax", priority_map)
        first_index = affine(best, -1.0, 16.0)
        first_gate = mul(near_zero(sub(priority_map, best)), program)
        first_code = emit("gmax", mul(first_gate, frame_map))
        return first_index, first_code, clip01(count)

    first2, color2, exists2 = frame_info(2)
    first3, color3, exists3 = frame_info(3)

    solid_inventory = dense(
        emit("gmean", solid_palette),
        4096.0 * np.eye(16, dtype=np.float32),
    )
    ring_inventory = dense(
        emit("gmean", ring_palette),
        4096.0 * np.eye(16, dtype=np.float32),
    )
    ring_count2 = vector_sum(
        mul(ring_inventory, vector_onehot_code(color2))
    )
    ring_count3 = vector_sum(
        mul(ring_inventory, vector_onehot_code(color3))
    )

    # Each candidate uses up to two call sites. The coordinate axes
    # encode site and target frame, with 32/33 denoting no call.
    def call_gates(axis, parity):
        w = np.ones((16, 1, 1, 1), dtype=np.float32)
        b = -np.array(
            [2 * j + parity for j in range(16)],
            dtype=np.float32,
        )
        return near_zero(conv(axis, w, b))

    a2 = call_gates(row_num, 0)
    a3 = call_gates(row_num, 1)
    b2 = call_gates(col_num, 0)
    b3 = call_gates(col_num, 1)
    calls2 = add(a2, b2)
    calls3 = add(a3, b3)
    calls = add(calls2, calls3)
    call_codes = add(mul(calls2, color2), mul(calls3, color3))

    static_error = spatial_sum(
        mul(absolute(sub(call_codes, slot_ring)), clip01(slot_ring))
    )
    static_error = add(
        static_error,
        spatial_sum(mul(calls, clip01(slot_solid))),
    )
    static_error = add(
        static_error,
        spatial_sum(mul(calls, affine(slot_exists, -1.0, 1.0))),
    )
    static_error = add(
        static_error,
        spatial_sum(mul(calls2, affine(exists2, -1.0, 1.0))),
    )
    static_error = add(
        static_error,
        spatial_sum(mul(calls3, affine(exists3, -1.0, 1.0))),
    )
    static_error = add(
        static_error,
        spatial_sum(relu(affine(calls, 1.0, -1.0))),
    )
    static_error = summation([
        static_error,
        relu(affine(row_num, 1.0, -33.0)),
        relu(affine(col_num, 1.0, -33.0)),
    ])

    minus_one = affine(one, -1.0, 0.0)
    initial_values = mul(one, slot_solid)

    # Execute candidate programs in parallel over spatial cells.
    body = []
    target[0] = body
    for key, n in (
        ("st_ptr", 1),
        ("st_s1", 1),
        ("st_s2", 1),
        ("st_goal", 1),
        ("st_error", 1),
        ("st_values", 16),
        ("st_seen", 16),
    ):
        shapes[key] = ("s", n)

    pointer_hot = rank_channels("st_ptr", 16, first=0)
    active = spatial_sum(pointer_hot)
    selected2 = spatial_sum(mul(pointer_hot, calls2))
    selected3 = spatial_sum(mul(pointer_hot, calls3))
    is_call = add(selected2, selected3)
    literal = sub(active, is_call)
    at_end = spatial_sum(mul(pointer_hot, slot_end))
    unwind = equal_scalar("st_ptr", -2)

    goal_hot = rank_channels("st_goal", 20, first=0)
    current_goal = spatial_sum(mul(goal_hot, goal_codes))
    old_value = spatial_sum(mul(pointer_hot, "st_values"))
    mismatch = mul(
        clip01(old_value),
        absolute(sub(old_value, current_goal)),
    )
    missing_goal = relu(affine(current_goal, -1.0, 1.0))
    error_next = add(
        "st_error",
        mul(literal, add(mismatch, missing_goal)),
    )

    change = mul(literal, sub(current_goal, old_value))
    values_next = add("st_values", mul(pointer_hot, change))
    seen_next = emit("max2", "st_seen", pointer_hot)
    goal_next = add("st_goal", literal)

    call_start = add(
        mul(selected2, first2),
        mul(selected3, first3),
    )
    return_address = sub(
        affine("st_ptr", 1.0, 1.0),
        mul(at_end, affine("st_ptr", 1.0, 3.0)),
    )
    pop = add(mul(literal, at_end), unwind)
    ptr_next = summation([
        "st_ptr",
        mul(is_call, sub(call_start, "st_ptr")),
        mul(mul(literal, at_end), sub("st_s1", "st_ptr")),
        mul(literal, affine(at_end, -1.0, 1.0)),
        mul(unwind, sub("st_s1", "st_ptr")),
    ])
    s1_next = summation([
        "st_s1",
        mul(is_call, sub(return_address, "st_s1")),
        mul(pop, sub("st_s2", "st_s1")),
    ])
    s2_next = summation([
        "st_s2",
        mul(is_call, sub("st_s1", "st_s2")),
        mul(pop, sub(minus_one, "st_s2")),
    ])

    target[0] = graph
    graph.append({
        "id": "execute",
        "op": "loop",
        "T": 32,
        "state": [
            {"id": "st_ptr", "zeros": [1, 64, 64]},
            {"id": "st_s1", "init": minus_one},
            {"id": "st_s2", "init": minus_one},
            {"id": "st_goal", "zeros": [1, 64, 64]},
            {"id": "st_error", "zeros": [1, 64, 64]},
            {"id": "st_values", "init": initial_values},
            {"id": "st_seen", "zeros": [16, 64, 64]},
        ],
        "body": body,
        "next": {
            "st_ptr": ptr_next,
            "st_s1": s1_next,
            "st_s2": s2_next,
            "st_goal": goal_next,
            "st_error": error_next,
            "st_values": values_next,
            "st_seen": seen_next,
        },
    })
    for key, n in (
        ("st_ptr", 1),
        ("st_s1", 1),
        ("st_s2", 1),
        ("st_goal", 1),
        ("st_error", 1),
        ("st_values", 16),
        ("st_seen", 16),
    ):
        shapes["execute." + key] = ("s", n)

    values = "execute.st_values"
    seen = "execute.st_seen"
    error = summation([
        static_error,
        "execute.st_error",
        absolute(affine("execute.st_ptr", 1.0, 1.0)),
        absolute(sub("execute.st_goal", goal_count)),
    ])

    need_calls2 = spatial_sum(mul(mul(calls2, seen), slot_empty))
    need_calls3 = spatial_sum(mul(mul(calls3, seen), slot_empty))
    error = summation([
        error,
        relu(sub(need_calls2, ring_count2)),
        relu(sub(need_calls3, ring_count3)),
    ])

    inventory_errors = []
    eye = np.eye(16, dtype=np.float32)[:, :, None, None]
    for c in range(16):
        delta = conv(
            values,
            eye,
            np.full(16, -(c + 1), dtype=np.float32),
        )
        equals = near_zero(delta)
        needed = spatial_sum(mul(equals, slot_empty))
        available = emit("vslice", solid_inventory, start=c, stop=c + 1)
        inventory_errors.append(relu(sub(needed, available)))
    error = add(error, summation(inventory_errors))
    valid = near_zero(error)

    # Select the first valid candidate.
    priority = summation([
        affine(one, 0.0, 1.1),
        affine(row_num, -0.001, 0.0),
        affine(col_num, -0.00001, 0.0),
    ])
    score = mul(valid, priority)
    best_score = emit("gmax", score)
    selected = relu(
        affine(absolute(sub(score, best_score)), -1000000.0, 1.0)
    )
    selected = mul(selected, valid)
    solution_exists = clip01(emit("gmax", selected))

    desired_candidate = add(values, mul(call_codes, seen))
    desired = emit("gmax", mul(desired_candidate, selected))
    chosen_calls = emit("gmax", mul(mul(calls, seen), selected))

    needed_values = mul(desired, slot_empty)
    needed_flags = clip01(needed_values)
    priorities = dense(
        needed_flags,
        np.diag(np.arange(16, 0, -1, dtype=np.float32)),
    )
    destination_scores = spatial_sum(mul(slot_gates, priorities))
    best_destination = emit("gmax", destination_scores)
    has_move = clip01(best_destination)
    destination = near_zero(sub(destination_scores, best_destination))
    destination = mul(mul(destination, program), has_move)

    chosen_slot = emit("gmax", mul(slot_gates, destination))
    chosen_color = vector_sum(mul(desired, chosen_slot))
    chosen_is_call = vector_sum(mul(chosen_calls, chosen_slot))
    color_hot = vector_onehot_code(chosen_color)
    solid_source = spatial_sum(mul(solid_palette, color_hot))
    ring_source = spatial_sum(mul(ring_palette, color_hot))
    source = add(
        mul(solid_source, affine(chosen_is_call, -1.0, 1.0)),
        mul(ring_source, chosen_is_call),
    )

    # Memory stores click phase, destination coordinates and recovery state.
    phase = emit("vslice", "mem", start=0, stop=1)
    saved_xy = emit("vslice", "mem", start=1, stop=3)
    recovery = emit("vslice", "mem", start=3, stop=4)
    not_phase = affine(phase, -1.0, 1.0)

    xy_w = np.zeros((2, 3, 1, 1), dtype=np.float32)
    xy_w[0, 0, 0, 0] = 63.0
    xy_w[1, 1, 0, 0] = 63.0
    xy = conv("coords", xy_w)
    destination_xy = emit("gmax", mul(xy, destination))
    xy_error = spatial_sum(absolute(sub(xy, saved_xy)))
    saved_destination = near_zero(xy_error)
    source_click = mul(source, not_phase)
    destination_click = mul(saved_destination, phase)
    graph.append({
        "id": "click",
        "op": "add",
        "in": [source_click, destination_click],
    })

    next_phase = mul(has_move, not_phase)
    failed = mul(affine(solution_exists, -1.0, 1.0), not_phase)
    rec0 = equal_scalar(recovery, 0)
    rec1 = equal_scalar(recovery, 1)
    do_undo = mul(failed, rec0)
    do_reset = mul(failed, rec1)
    next_recovery = summation([recovery, do_undo, do_reset])
    graph.append({
        "id": "mem_next",
        "op": "vconcat",
        "in": [next_phase, destination_xy, next_recovery],
    })

    click_action = add(phase, mul(has_move, not_phase))
    features = emit("vconcat", click_action, do_undo, do_reset)
    action_w = np.zeros((8, 3), dtype=np.float32)
    action_b = np.full(8, -5.0, dtype=np.float32)
    action_b[5] = 1.0
    action_b[6] = 0.0
    action_b[7] = 0.0
    action_b[0] = 0.0
    action_w[6, 0] = 3.0
    action_w[7, 1] = 3.0
    action_w[0, 2] = 3.0
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [features],
        "w": par(action_w),
        "b": par(action_b),
    })

    return {"graph": graph, "memory_size": 4, "params": params}

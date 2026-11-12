import numpy as np


def build():
    F, L, N, K = 5, 10, 18, 20
    S = F * L
    M = 4 * S + K + 10

    G, P = [], {}
    current = G
    serial = 0
    param_cache = {}
    node_caches = {}
    constants = {}
    vector_constants = {}
    sum_weights = {}
    spatial_sum_weights = {}

    def fresh(prefix="n"):
        nonlocal serial
        serial += 1
        return "%s_%d" % (prefix, serial)

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in param_cache:
            name = fresh("p")
            param_cache[key] = name
            P[name] = a
        return param_cache[key]

    def emit(op, ins=None, name=None, where=None, **fields):
        dest = current if where is None else where
        cache = node_caches.setdefault(id(dest), {})
        ins = [] if ins is None else list(ins)
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if name is None:
            if dest is not G:
                global_cache = node_caches.setdefault(id(G), {})
                if key in global_cache:
                    return global_cache[key]
            if key in cache:
                return cache[key]
        node_id = name or fresh()
        nd = {"id": node_id, "op": op, "in": ins}
        nd.update(fields)
        dest.append(nd)
        cache[key] = node_id
        return node_id

    def dense(x, w, b=None, name=None, where=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, where=where, **fields)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    one = dense(
        "avail",
        np.zeros((1, 8), np.float32),
        np.ones(1, np.float32)
    )

    def C(value):
        key = float(value)
        if key not in constants:
            constants[key] = dense(
                one, np.array([[key]], np.float32), where=G
            )
        return constants[key]

    def VC(values):
        a = np.asarray(values, np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key not in vector_constants:
            vector_constants[key] = dense(one, a[:, None], where=G)
        return vector_constants[key]

    def sl(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def neg(a):
        return vm(a, C(-1))

    def vs(a, b):
        return va(a, neg(b))

    def vr(a):
        return emit("vrelu", [a])

    def lin(a, scale=1, bias=0):
        y = a if scale == 1 else vm(a, C(scale))
        return y if bias == 0 else va(y, C(bias))

    def vabs(a):
        return va(vr(a), vr(neg(a)))

    def eq(a, b):
        return vr(vs(C(1), vabs(vs(a, b))))

    def positive(a):
        return vs(vr(a), vr(lin(a, bias=-1)))

    def complement(a):
        return vs(C(1), a)

    def prod(*xs):
        y = xs[0]
        for x in xs[1:]:
            y = vm(y, x)
        return y

    def total(x, n):
        if n not in sum_weights:
            sum_weights[n] = parameter(
                np.ones((1, n), np.float32)
            )
        return emit("dense", [x], w=sum_weights[n])

    def dot(a, b, n):
        return total(vm(a, b), n)

    def blend(g, a, b):
        return va(vm(g, a), vm(complement(g), b))

    def sadd(a, b):
        return emit("add", [a, b])

    def ssub(a, b):
        return emit("sub", [a, b])

    def smul(a, b):
        return emit("mul", [a, b])

    def sr(a):
        return emit("relu", [a])

    def sclip(a):
        return emit("min2", [sr(a), C(1)])

    def spatial_sum(x, channels):
        if channels not in spatial_sum_weights:
            spatial_sum_weights[channels] = parameter(
                np.ones((1, channels, 1, 1), np.float32)
            )
        return emit("conv", [x], w=spatial_sum_weights[channels])

    def spatial_encode(x, values):
        return conv(
            x, np.asarray(values, np.float32).reshape(1, -1, 1, 1)
        )

    def gather_spatial(image, mask):
        return emit("gmax", [smul(image, mask)])

    # Pixel coordinates and a strictly decreasing row-major rank.
    cw = np.zeros((2, 3, 1, 1), np.float32)
    cw[0, 0, 0, 0] = 63
    cw[1, 1, 0, 0] = 63
    xy = conv("coords", cw)
    row = emit("slice", [xy], start=0, stop=1)
    col = emit("slice", [xy], start=1, stop=2)

    rw = np.zeros((1, 3, 1, 1), np.float32)
    rw[0, 0, 0, 0] = -63 * 64 / 4096
    rw[0, 1, 0, 0] = -63 / 4096
    rank = conv("coords", rw, np.array([2], np.float32))

    below = sclip(ssub(row, C(53)))
    instruction_region = sclip(ssub(row, C(9)))
    target_region = sclip(ssub(C(12), row))
    frame_region = smul(
        sclip(ssub(row, C(11))),
        sclip(ssub(C(53), row))
    )

    def point_masks(rows, cols):
        dr = emit("abs", [ssub(row, rows)])
        dc = emit("abs", [ssub(col, cols)])
        distance = sadd(dr, dc)
        return sclip(sadd(smul(distance, C(-4)), C(2)))

    # Item anchors are item origin + (1,1).
    # Channels 0:16 are solids; channels 16:32 are references.
    iw = np.zeros((32, 16, 7, 7), np.float32)
    ib = np.full(32, -15, np.float32)
    for color in range(16):
        if color == 4:
            ib[color] = -100
            ib[16 + color] = -100
            continue
        for dy in range(4):
            for dx in range(4):
                iw[color, color, 3 + dy, 3 + dx] = 1
                if dy in (0, 3) or dx in (0, 3):
                    iw[16 + color, color, 3 + dy, 3 + dx] = 1
                else:
                    iw[16 + color, 4, 3 + dy, 3 + dx] = 1

    items = smul(sr(conv("x0", iw, ib)), instruction_region)
    item_code = spatial_encode(items, np.arange(1, 33))
    pool_counts = vm(
        emit("gmean", [smul(items, below)]), C(4096)
    )

    # Frame anchors are frame origin + (3,3).
    fw = np.zeros((16, 16, 7, 7), np.float32)
    fb = np.full(16, -13, np.float32)
    for color in range(16):
        if color == 4:
            fb[color] = -100
            continue
        for dx in range(7):
            fw[color, color, 0, dx] = 1
        for dy in range(1, 7):
            fw[color, color, dy, 0] = 1
        fw[color, 4, 1, 1] = 1

    frames = smul(sr(conv("x0", fw, fb)), frame_region)
    frame_anchors = spatial_sum(frames, 16)
    frame_color_map = spatial_encode(frames, np.arange(1, 17))

    # Target anchors are target origin + (3,3).
    tw = np.zeros((16, 16, 7, 7), np.float32)
    tb = np.full(16, -35, np.float32)
    for color in range(16):
        if color == 5:
            tb[color] = -100
            continue
        for dy in range(6):
            for dx in range(6):
                if dy in (0, 5) or dx in (0, 5):
                    tw[color, color, dy, dx] = 1
                else:
                    tw[color, 5, dy, dx] = 1

    targets = smul(sr(conv("x0", tw, tb)), target_region)
    target_anchors = spatial_sum(targets, 16)
    target_color_map = spatial_encode(targets, np.arange(1, 17))

    def first_anchor(anchors):
        scores = smul(anchors, rank)
        best = emit("gmax", [scores])
        peak = sr(sadd(smul(ssub(scores, best), C(4096)), C(1)))
        selected = smul(anchors, peak)
        remaining = ssub(anchors, selected)
        return selected, remaining

    # Compact target reader: remove the first remaining anchor and append
    # its color to the target vector.
    target_indices = VC(np.arange(N))
    target_body = []
    current = target_body
    pick, target_remaining = first_anchor("trAnchors")
    symbol = gather_spatial(target_color_map, pick)
    target_write = eq(target_indices, "trIndex")
    target_updated = va("trSymbols", vm(target_write, symbol))
    target_index_updated = lin("trIndex", bias=1)
    current = G

    target_loop = fresh("targets")
    G.append({
        "id": target_loop,
        "op": "loop",
        "T": N,
        "state": [
            {"id": "trAnchors", "init": target_anchors},
            {"id": "trSymbols", "zeros": [N]},
            {"id": "trIndex", "zeros": [1]},
        ],
        "body": target_body,
        "next": {
            "trAnchors": target_remaining,
            "trSymbols": target_updated,
            "trIndex": target_index_updated,
        },
    })
    target_vector = target_loop + ".trSymbols"
    target_length = total(positive(target_vector), N)

    # Compact frame reader, ordered by (row,column).
    frame_indices = VC(np.arange(F))
    frame_body = []
    current = frame_body
    pick, frame_remaining = first_anchor("frAnchors")
    position = gather_spatial(xy, pick)
    frame_symbol = gather_spatial(frame_color_map, pick)
    frame_present = emit("gmax", [pick])
    frame_write = eq(frame_indices, "frIndex")
    updated_rows = va("frRows", vm(frame_write, sl(position, 0, 1)))
    updated_cols = va("frCols", vm(frame_write, sl(position, 1, 2)))
    updated_colors = va("frColors", vm(frame_write, frame_symbol))
    updated_exists = va("frExists", vm(frame_write, frame_present))
    frame_index_updated = lin("frIndex", bias=1)
    current = G

    frame_loop = fresh("frames")
    G.append({
        "id": frame_loop,
        "op": "loop",
        "T": F,
        "state": [
            {"id": "frAnchors", "init": frame_anchors},
            {"id": "frRows", "zeros": [F]},
            {"id": "frCols", "zeros": [F]},
            {"id": "frColors", "zeros": [F]},
            {"id": "frExists", "zeros": [F]},
            {"id": "frIndex", "zeros": [1]},
        ],
        "body": frame_body,
        "next": {
            "frAnchors": frame_remaining,
            "frRows": updated_rows,
            "frCols": updated_cols,
            "frColors": updated_colors,
            "frExists": updated_exists,
            "frIndex": frame_index_updated,
        },
    })

    fr = frame_loop + ".frRows"
    fc = frame_loop + ".frCols"
    colors = frame_loop + ".frColors"
    exists = frame_loop + ".frExists"
    frame_colors = [sl(colors, f, f + 1) for f in range(F)]
    frame_exists = [sl(exists, f, f + 1) for f in range(F)]

    # Expand frame coordinates to instruction-slot coordinates.
    repeat = np.zeros((S, F), np.float32)
    offsets = np.zeros(S, np.float32)
    for f in range(F):
        for j in range(L):
            repeat[f * L + j, f] = 1
            offsets[f * L + j] = 6 * j

    slot_rows = dense(fr, repeat)
    slot_cols = dense(fc, repeat, offsets)
    repeated_colors = dense(colors, repeat)
    repeated_exists = dense(exists, repeat)

    all_slot_masks = point_masks(slot_rows, slot_cols)
    top_masks = point_masks(
        lin(slot_rows, bias=-3), lin(slot_cols, bias=1)
    )
    pixel_code = spatial_encode("x0", np.arange(1, 17))
    sampled_top = gather_spatial(pixel_code, top_masks)
    top_match = eq(sampled_top, repeated_colors)

    prefix_matrix = np.zeros((S, S), np.float32)
    for f in range(F):
        for j in range(L):
            prefix_matrix[f * L + j, f * L:f * L + j + 1] = 1
    missing_prefix = dense(complement(top_match), prefix_matrix)
    valid_slots = vm(
        vr(vs(C(1), missing_prefix)), repeated_exists
    )

    slot_masks = smul(all_slot_masks, valid_slots)
    observed_slots = gather_spatial(item_code, slot_masks)

    # Decode fixed starting instructions.
    initial_present = positive(observed_slots)
    initial_call = positive(lin(observed_slots, bias=-16))
    initial_literal = vs(initial_present, initial_call)
    initial_a = initial_literal
    raw_call_color = lin(observed_slots, bias=-16)
    for f in range(F):
        match = vm(eq(raw_call_color, frame_colors[f]), initial_call)
        initial_a = va(initial_a, vm(match, C(f + 2)))
    initial_b = vm(observed_slots, initial_literal)

    # Allowed reference alternatives, ordered by frame index.
    categories = VC(np.arange(1, 33))
    active_calls = []
    for f in range(F):
        cat = lin(frame_colors[f], bias=16)
        count = dot(pool_counts, eq(categories, cat), 32)
        active_calls.append(vm(positive(count), frame_exists[f]))

    next_alternatives = [C(0)]
    for old_choice in range(1, F + 2):
        nxt = C(0)
        for f in reversed(range(old_choice - 1, F)):
            nxt = blend(active_calls[f], C(f + 2), nxt)
        next_alternatives.append(nxt)
    next_table = emit("vconcat", next_alternatives)

    # Memory: assignments, literal colors, decision stack, fixed mask,
    # return stack, interpreter registers, startup clock and click state.
    old_a = sl("mem", 0, S)
    old_b = sl("mem", S, 2 * S)
    old_d = sl("mem", 2 * S, 3 * S)
    old_fixed = sl("mem", 3 * S, 4 * S)
    old_r = sl("mem", 4 * S, 4 * S + K)
    base = 4 * S + K
    old_regs = [
        sl("mem", base + i, base + i + 1) for i in range(6)
    ]
    clock = sl("mem", base + 6, base + 7)
    phase = sl("mem", base + 7, base + 8)
    saved_row = sl("mem", base + 8, base + 9)
    saved_col = sl("mem", base + 9, base + 10)

    initialize = eq(clock, C(7))
    enabled = positive(lin(clock, bias=-6))
    not_initialize = complement(initialize)

    a_init = blend(initialize, initial_a, old_a)
    b_init = blend(initialize, initial_b, old_b)
    d_init = vm(old_d, not_initialize)
    fixed = blend(initialize, initial_present, old_fixed)
    r_init = vm(old_r, not_initialize)
    reg_init = [vm(x, not_initialize) for x in old_regs]
    mutable = complement(fixed)

    slot_indices = VC(np.arange(S))
    return_indices = VC(np.arange(K))
    call_choices = VC(np.arange(2, F + 2))
    alternative_indices = VC(np.arange(F + 2))
    frame_entry = VC(np.arange(F) * L)
    call_categories = lin(colors, bias=16)

    # Bounded backtracking interpreter. All machine registers are
    # integer-valued; equality and gates use piecewise-linear arithmetic.
    body = []
    current = body
    A, B, D, R = "sA", "sB", "sD", "sR"
    pc, p, h = "sPC", "sP", "sH"
    d, bt, done = "sDepth", "sBT", "sDone"

    pc_hot = eq(slot_indices, pc)
    p_hot = eq(target_indices, p)
    choice = dot(A, pc_hot, S)
    literal_color = dot(B, pc_hot, S)
    current_target = dot(target_vector, p_hot, N)
    valid = dot(valid_slots, pc_hot, S)

    literal_array = eq(A, C(1))
    is_literal = dot(literal_array, pc_hot, S)
    is_empty = eq(choice, C(0))
    call_hot = eq(call_choices, choice)
    is_call = total(call_hot, F)
    call_pc = dot(frame_entry, call_hot, F)
    call_cat = dot(call_categories, call_hot, F)
    call_exists = dot(exists, call_hot, F)

    literal_matches = vm(literal_array, eq(B, literal_color))
    reference_matches = eq(A, choice)
    matching_assignments = va(
        vm(literal_matches, is_literal),
        vm(reference_matches, is_call)
    )
    used = total(vm(matching_assignments, mutable), S)
    query_cat = va(
        vm(literal_color, is_literal), vm(call_cat, is_call)
    )
    available_count = dot(
        pool_counts, eq(categories, query_cat), 32
    )
    stock_ok = complement(positive(vs(used, available_count)))

    literal_ok = vm(stock_ok, eq(literal_color, current_target))
    room = positive(vs(C(K), h))
    call_ok = prod(stock_ok, room, call_exists)
    has_return = positive(h)

    active = prod(enabled, complement(bt), complement(done))
    valid_active = vm(active, valid)
    empty_gate = vm(valid_active, is_empty)
    literal_good = prod(valid_active, is_literal, literal_ok)
    call_good = prod(valid_active, is_call, call_ok)
    return_good = prod(active, complement(valid), has_return)

    bad_literal = vm(is_literal, complement(literal_ok))
    bad_call = vm(is_call, complement(call_ok))
    failure = vm(active, va(
        vm(valid, va(bad_literal, bad_call)),
        vm(complement(valid), complement(has_return))
    ))

    last_decision_hot = eq(slot_indices, lin(d, bias=-1))
    last_slot = dot(D, last_decision_hot, S)
    last_slot_hot = eq(slot_indices, last_slot)
    old_choice = dot(A, last_slot_hot, S)
    alternative_hot = eq(alternative_indices, old_choice)
    next_choice = dot(next_table, alternative_hot, F + 2)
    next_exists = positive(next_choice)
    back = prod(enabled, bt, complement(done), positive(d))
    advance = vm(back, next_exists)
    exhaust = vm(back, complement(next_exists))

    new_a = va(A, vm(pc_hot, empty_gate))
    new_a = va(new_a, prod(
        last_slot_hot, back, vs(next_choice, old_choice)
    ))
    new_b = va(B, prod(
        pc_hot, empty_gate, vs(current_target, literal_color)
    ))

    decision_write = prod(eq(slot_indices, d), empty_gate)
    new_d_stack = va(
        vm(D, complement(decision_write)),
        vm(decision_write, pc)
    )

    return_write = prod(eq(return_indices, h), call_good)
    new_r = va(
        vm(R, complement(return_write)),
        vm(return_write, lin(pc, bias=1))
    )
    return_top = dot(
        R, eq(return_indices, lin(h, bias=-1)), K
    )

    new_pc = va(pc, literal_good)
    new_pc = va(new_pc, vm(call_good, vs(call_pc, pc)))
    new_pc = va(new_pc, vm(return_good, vs(return_top, pc)))
    new_pc = vm(new_pc, complement(back))
    new_p = vm(va(p, literal_good), complement(back))
    new_h = vm(
        vs(va(h, call_good), return_good), complement(back)
    )
    new_depth = vs(va(d, empty_gate), exhaust)
    new_bt = vs(va(bt, failure), advance)
    success = vm(
        literal_good, eq(lin(p, bias=1), target_length)
    )
    new_done = va(done, success)

    current = G
    solver_loop = fresh("solver")
    solver_node = {
        "id": solver_loop,
        "op": "loop",
        "T": 1,
        "state": [
            {"id": "sA", "init": a_init},
            {"id": "sB", "init": b_init},
            {"id": "sD", "init": d_init},
            {"id": "sR", "init": r_init},
            {"id": "sPC", "init": reg_init[0]},
            {"id": "sP", "init": reg_init[1]},
            {"id": "sH", "init": reg_init[2]},
            {"id": "sDepth", "init": reg_init[3]},
            {"id": "sBT", "init": reg_init[4]},
            {"id": "sDone", "init": reg_init[5]},
        ],
        "body": body,
        "next": {
            "sA": new_a,
            "sB": new_b,
            "sD": new_d_stack,
            "sR": new_r,
            "sPC": new_pc,
            "sP": new_p,
            "sH": new_h,
            "sDepth": new_depth,
            "sBT": new_bt,
            "sDone": new_done,
        },
    }
    G.append(solver_node)

    def result(state):
        return solver_loop + "." + state

    final_a = result("sA")
    final_b = result("sB")
    ready = vm(enabled, result("sDone"))

    desired = vm(final_b, eq(final_a, C(1)))
    for f in range(F):
        desired = va(desired, vm(
            eq(final_a, C(f + 2)),
            lin(frame_colors[f], bias=16)
        ))

    required = vm(positive(final_a), valid_slots)
    correct = vm(eq(desired, observed_slots), required)
    mismatch = vm(required, complement(correct))
    has_mismatch = positive(total(mismatch, S))

    lower = np.tril(np.ones((S, S), np.float32), -1)
    preceding = dense(mismatch, lower)
    first_bad = vm(mismatch, vr(vs(C(1), preceding)))
    destination_row = dot(slot_rows, first_bad, S)
    destination_col = dot(slot_cols, first_bad, S)
    desired_category = dot(desired, first_bad, S)

    category_selector = eq(categories, desired_category)
    matching_items = spatial_sum(
        smul(items, category_selector), 32
    )
    correct_positions = spatial_sum(
        smul(slot_masks, correct), S
    )
    source_candidates = sr(ssub(matching_items, correct_positions))
    source_scores = smul(source_candidates, rank)

    destination_peak = point_masks(saved_row, saved_col)
    fallback_peak = point_masks(C(63), C(0))

    install = vm(ready, has_mismatch)
    select_gate = vm(install, complement(phase))
    place_gate = vm(install, phase)
    busy = vs(enabled, ready)

    click_selection = smul(source_scores, select_gate)
    click_placement = smul(destination_peak, place_gate)
    click_wait = smul(fallback_peak, busy)
    emit("add", [
        sadd(click_selection, click_placement), click_wait
    ], name="click")

    # Undo short perturbed prefixes, then test once to clear selection.
    undo_flag = complement(positive(lin(clock, bias=-5)))
    startup_test = eq(clock, C(6))
    run_flag = va(
        startup_test, vm(ready, complement(has_mismatch))
    )
    click_flag = va(busy, install)

    action_features = emit(
        "vconcat", [undo_flag, run_flag, click_flag]
    )
    action_w = np.zeros((8, 3), np.float32)
    action_w[7, 0] = 10
    action_w[5, 1] = 10
    action_w[6, 2] = 10
    action_b = np.full(8, -1000, np.float32)
    action_b[5:8] = 0
    dense(action_features, action_w, action_b, name="logits")

    clock_increment = lin(clock, bias=1)
    next_clock = vs(
        clock_increment, vr(lin(clock_increment, bias=-8))
    )
    next_phase = vm(install, complement(phase))

    emit("vconcat", [
        result("sA"),
        result("sB"),
        result("sD"),
        fixed,
        result("sR"),
        result("sPC"),
        result("sP"),
        result("sH"),
        result("sDepth"),
        result("sBT"),
        result("sDone"),
        next_clock,
        next_phase,
        destination_row,
        destination_col,
    ], name="mem_next")

    # Static graph bookkeeping, not observation-dependent optimization.
    other_evaluations = len(G)
    other_evaluations += (N - 1) * len(target_body)
    other_evaluations += (F - 1) * len(frame_body)
    solver_node["T"] = max(
        1, min(96, (9500 - other_evaluations) // len(body))
    )

    return {
        "graph": G,
        "memory_size": M,
        "params": P,
    }

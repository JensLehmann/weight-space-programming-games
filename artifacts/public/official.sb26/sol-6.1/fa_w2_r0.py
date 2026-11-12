import numpy as np


def build():
    # Maximum supported representation sizes.
    F = 5                 # Functions, including the root
    S = 9                 # Slots per function
    STRIDE = S + 1        # Extra position represents the end of a function
    N = F * STRIDE
    D = 28                # Backtracking decisions
    R = F - 1             # Return-stack entries
    K = 3 + R             # Decision record: pointer, target, choice, returns
    HSIZE = D * K
    CSIZE = 2 + R         # Interpreter: pointer, target, returns
    TARGET = 24
    TYPES = 16 + F
    M = N + HSIZE + CSIZE + 2

    graph = []
    params = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (M,),
        "aprev": (8,),
        "avail": (8,),
    }
    current = graph
    serial = [0]
    parameter_cache = {}

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, ins, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current.append(nd)
        ss = [shapes[i] for i in ins]

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

    def conv(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("conv", [x], name=name, **kw)

    def dense(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return node("dense", [x], name=name, **kw)

    def sl(x, a, b):
        op = "slice" if len(shapes[x]) == 3 else "vslice"
        return node(op, [x], start=a, stop=b)

    def cat(xs):
        op = "concat" if len(shapes[xs[0]]) == 3 else "vconcat"
        return node(op, xs)

    def gm(x):
        return node("gmax", [x])

    def add(a, b):
        return node("vadd", [a, b])

    def mul(a, b):
        return node("vmul", [a, b])

    one_map = sl("coords", 2, 3)
    zero_map = conv("coords", np.zeros((1, 3, 1, 1), np.float32))
    constants = {}
    for value in (-1, 0, 1, 2, 8, 10):
        constants[value] = dense(
            "avail", np.zeros((1, 8), np.float32),
            np.array([value], np.float32)
        )

    def scale(x, value):
        return mul(x, constants[value])

    def sub(a, b):
        return add(a, scale(b, -1))

    def lin(x, weight=1.0, bias=0.0):
        return dense(
            x, np.array([[weight]], np.float32),
            np.array([bias], np.float32)
        )

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def ssum(x):
        return conv(x, np.ones((1, shapes[x][0], 1, 1), np.float32))

    def clip(x):
        # Exact clipping to [0,1], including vector-to-spatial broadcasting.
        r = node("vrelu", [x])
        return gm(node("min2", [one_map, r]))

    def inv(x):
        return sub(constants[1], x)

    def shat(x):
        # Equals one near zero and zero for integer differences of magnitude >=1.
        a = node("abs", [x])
        c = shapes[x][0]
        w = (-4 * np.eye(c, dtype=np.float32))[:, :, None, None]
        z = conv(a, w, np.full(c, 2, np.float32))
        return node("min2", [node("relu", [z]), one_map])

    def vhat(x):
        lifted = node("add", [zero_map, x])
        return gm(shat(lifted))

    def veq(x, value):
        return vhat(lin(x, 1, -value))

    def select(flag, yes, no):
        return add(mul(flag, yes), mul(inv(flag), no))

    def shift(x, dy=0, dx=0):
        # Positive offsets move features down/right.
        c = shapes[x][0]
        k = 1 if dy == 0 and dx == 0 else (
            3 if max(abs(dy), abs(dx)) <= 1 else
            5 if max(abs(dy), abs(dx)) <= 2 else 7
        )
        w = np.zeros((c, c, k, k), np.float32)
        for i in range(c):
            w[i, i, k // 2 - dy, k // 2 - dx] = 1
        return conv(x, w)

    def simple_loop(init, T, make_body):
        nonlocal current
        lid, sid = fresh(), fresh()
        shapes[sid] = shapes[init]
        old = current
        body = []
        current = body
        result = make_body(sid)
        current = old
        nd = {
            "id": lid, "op": "loop", "T": T,
            "state": [{"id": sid, "init": init}],
            "body": body, "next": {sid: result},
        }
        current.append(nd)
        out = lid + "." + sid
        shapes[out] = shapes[init]
        return out

    # ------------------------------------------------------------------
    # Perception.
    # ------------------------------------------------------------------
    colors = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    C = len(colors)
    w = np.zeros((C, 16, 1, 1), np.float32)
    for i, color in enumerate(colors):
        w[i, color, 0, 0] = 1
    color_planes = conv("x0", w)

    # Rectangle top-left corners: height ten, width at least ten.
    down3 = shift(color_planes, -3, 0)
    down6 = shift(down3, -3, 0)
    down9 = shift(down6, -3, 0)
    right3 = shift(color_planes, 0, -3)
    right6 = shift(right3, 0, -3)

    w = np.zeros((C, C, 7, 7), np.float32)
    for i in range(C):
        for dy, dx in ((0, 0), (3, 0), (0, 3)):
            w[i, i, 3 + dy, 3 + dx] += 1
        w[i, i, 2, 3] -= 1
        w[i, i, 3, 2] -= 1
    z = conv(color_planes, w, np.full(C, -5, np.float32))
    for q in (down6, down9, right6):
        z = node("add", [z, q])
    frame_corners = ssum(node("relu", [z]))

    # Target six-pixel rings, distinguished by their color-5 interior.
    w = np.zeros((C, 16, 7, 7), np.float32)
    for i, color in enumerate(colors):
        for dy, dx in (
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (2, 0), (3, 0)
        ):
            w[i, color, 3 + dy, 3 + dx] += 1
        w[i, 5, 4, 4] += 1
        w[i, color, 2, 3] -= 1
        w[i, color, 3, 2] -= 1
    target_corners = ssum(node(
        "relu", [conv("x0", w, np.full(C, -7, np.float32))]
    ))

    # Spatial row-major ranks, using two prefix-scan loops.
    anchors = cat([frame_corners, target_corners])
    hp = simple_loop(
        anchors, 64,
        lambda q: node("add", [shift(q, 0, 1), anchors])
    )
    row_counts = node("rowsum", [anchors])
    vp = simple_loop(
        row_counts, 64,
        lambda q: node("add", [shift(q, 1, 0), row_counts])
    )
    ranks = node("add", [
        node("sub", [hp, anchors]),
        node("sub", [vp, row_counts]),
    ])

    def rank_masks(rank, count, mask):
        w = np.ones((count, 1, 1, 1), np.float32)
        z = conv(rank, w, -np.arange(count, dtype=np.float32))
        return node("mul", [shat(z), mask])

    frame_masks = rank_masks(sl(ranks, 0, 1), F, frame_corners)
    target_masks = rank_masks(sl(ranks, 1, 2), TARGET, target_corners)

    color_value_w = np.arange(1, 17, dtype=np.float32).reshape(1, 16, 1, 1)
    color_value = conv("x0", color_value_w)
    frame_colors = gm(node("mul", [frame_masks, color_value]))
    target_values = gm(node("mul", [target_masks, color_value]))
    target_length = vsum(gm(target_masks))

    # Propagate each frame's identity along its top border.
    border = ssum(color_planes)
    coverage = simple_loop(
        frame_masks, 64,
        lambda q: node("mul", [
            node("max2", [shift(q, 0, 1), frame_masks]),
            border,
        ])
    )
    coverage_at_slots = shift(shift(coverage, 3, 0), 1, 0)
    q = shift(shift(frame_masks, 3, 3), 1, 1)
    chunks = []
    for s in range(S):
        chunks.append(node("mul", [q, coverage_at_slots]))
        if s + 1 < S:
            q = shift(shift(q, 0, 3), 0, 3)
    raw_slots = cat(chunks)

    w = np.zeros((N, F * S, 1, 1), np.float32)
    for f in range(F):
        for s in range(S):
            w[f * STRIDE + s, s * F + f, 0, 0] = 1
    slot_masks = conv(raw_slots, w)
    slot_valid = gm(slot_masks)

    capacity_w = np.zeros((F, N), np.float32)
    for f in range(F):
        capacity_w[f, f * STRIDE:f * STRIDE + S] = 1
    capacities = dense(slot_valid, capacity_w)
    function_exists = clip(capacities)

    # Column-index masks serve as a small, broadcastable abstract tape.
    axis_w = np.zeros((1, 3, 1, 1), np.float32)
    axis_w[0, 1, 0, 0] = 63
    column_axis = conv("coords", axis_w)

    w = np.zeros((N, 3, 1, 1), np.float32)
    w[:, 1, 0, 0] = 63
    column_masks = shat(conv(
        "coords", w, -np.arange(N, dtype=np.float32)
    ))
    target_columns = sl(column_masks, 0, TARGET)
    type_columns = sl(column_masks, 1, TYPES + 1)
    reference_columns = sl(column_masks, 17, 17 + F)

    target_tape = ssum(node("mul", [target_columns, target_values]))

    # Token detections at click coordinates: the 4x4 colored border.
    w = np.zeros((C, 16, 7, 7), np.float32)
    for i, color in enumerate(colors):
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                if dy in (-1, 2) or dx in (-1, 2):
                    w[i, color, 3 + dy, 3 + dx] = 1
    token_features = node(
        "relu", [conv("x0", w, np.full(C, -11, np.float32))]
    )
    token_exists = ssum(token_features)
    token_color = conv(
        token_features,
        np.array([c + 1 for c in colors], np.float32).reshape(1, C, 1, 1)
    )
    background = sl("x0", 4, 5)
    not_background = node("sub", [one_map, background])

    reference_match = shat(node("sub", [token_color, frame_colors]))
    reference_code = conv(
        reference_match,
        np.arange(17, 17 + F, dtype=np.float32).reshape(1, F, 1, 1)
    )
    token_id = node("add", [
        node("mul", [token_color, not_background]),
        node("mul", [
            reference_code,
            node("mul", [token_exists, background]),
        ]),
    ])
    actual_slots = gm(node("mul", [slot_masks, token_id]))

    # Available resources include fixed board tokens plus bottom inventory.
    row_w = np.zeros((1, 3, 1, 1), np.float32)
    row_w[0, 0, 0, 0] = 63
    bottom_region = node("min2", [
        node("relu", [conv("coords", row_w, np.array([-53], np.float32))]),
        one_map,
    ])
    bottom_tokens = node("mul", [bottom_region, token_exists])
    board_positions = ssum(slot_masks)
    resource_positions = node("mul", [
        token_exists,
        node("min2", [
            node("add", [bottom_region, board_positions]), one_map
        ]),
    ])

    type_numbers = dense(
        "avail", np.zeros((TYPES, 8), np.float32),
        np.arange(1, TYPES + 1, dtype=np.float32)
    )
    resource_types = shat(node("sub", [token_id, type_numbers]))
    resource_counts = node("gmean", [
        node("mul", [resource_types, resource_positions])
    ])
    resource_counts = dense(
        resource_counts, 4096 * np.eye(TYPES, dtype=np.float32)
    )
    resource_tape = ssum(node("mul", [type_columns, resource_counts]))

    # Detect the currently selected token's full 6x6 black outline.
    black = sl("x0", 0, 1)
    bd3 = shift(black, -3, 0)
    bd5 = shift(bd3, -2, 0)
    br3 = shift(black, 0, -3)
    br5 = shift(br3, 0, -2)
    bdr5 = shift(shift(bd5, 0, -3), 0, -2)
    w = np.zeros((1, 1, 7, 7), np.float32)
    for dy, dx in ((0, 0), (3, 0), (0, 3)):
        w[0, 0, 3 + dy, 3 + dx] += 1
    w[0, 0, 2, 3] -= 1
    w[0, 0, 3, 2] -= 1
    selected_z = conv(black, w, np.array([-5], np.float32))
    for q in (bd5, br5, bdr5):
        selected_z = node("add", [selected_z, q])
    selected_origin = node("relu", [selected_z])
    selected_exists = gm(selected_origin)
    selected_click = shift(selected_origin, 2, 2)

    # ------------------------------------------------------------------
    # Memory and initialization.
    # ------------------------------------------------------------------
    mem_plan = sl("mem", 0, N)
    mem_hist = sl("mem", N, N + HSIZE)
    mem_cur = sl("mem", N + HSIZE, N + HSIZE + CSIZE)
    mem_mode = sl("mem", M - 2, M - 1)
    mem_boot = sl("mem", M - 1, M)

    boot_done = clip(lin(mem_boot, 1, -7))
    initialize = clip(add(inv(boot_done), clip(scale(mem_mode, -1))))
    keep = inv(initialize)
    initial_plan = select(initialize, actual_slots, mem_plan)
    initial_hist = mul(keep, mem_hist)
    initial_cur = mul(keep, mem_cur)
    initial_mode = mul(keep, mem_mode)

    # Decision-stack linear maps.
    hist_down_w = np.zeros((HSIZE, HSIZE), np.float32)
    hist_up_w = np.zeros((HSIZE, HSIZE), np.float32)
    hist_top_w = np.zeros((K, HSIZE), np.float32)
    hist_insert_w = np.zeros((HSIZE, K), np.float32)
    for k in range(K):
        a = k * D
        hist_top_w[k, a] = 1
        hist_insert_w[a, k] = 1
        for j in range(1, D):
            hist_down_w[a + j, a + j - 1] = 1
            hist_up_w[a + j - 1, a + j] = 1
    stage_insert_w = np.zeros((HSIZE, 1), np.float32)
    stage_insert_w[2 * D, 0] = 1

    frame_index_w = np.zeros((F, N), np.float32)
    slot_index_w = np.zeros((1, N), np.float32)
    for i in range(N):
        frame_index_w[i // STRIDE, i] = 1
        slot_index_w[0, i] = i % STRIDE

    # Interpreter transforms for calls, returns, and restored decisions.
    call_w = np.zeros((CSIZE, CSIZE + 1), np.float32)
    call_b = np.zeros(CSIZE, np.float32)
    call_w[0, CSIZE] = STRIDE
    call_b[0] = -17 * STRIDE
    call_w[1, 1] = 1
    call_w[2, 0] = 1
    call_b[2] = 2
    for j in range(1, R):
        call_w[2 + j, 1 + j] = 1

    return_w = np.zeros((CSIZE, CSIZE), np.float32)
    return_b = np.zeros(CSIZE, np.float32)
    return_w[0, 2] = 1
    return_b[0] = -1
    return_w[1, 1] = 1
    for j in range(R - 1):
        return_w[2 + j, 3 + j] = 1

    restore_w = np.zeros((CSIZE, K), np.float32)
    restore_b = np.zeros(CSIZE, np.float32)
    restore_w[0, 0] = 1
    restore_b[0] = -1
    restore_w[1, 1] = 1
    for j in range(R):
        restore_w[2 + j, 3 + j] = 1

    choice_prefix_w = np.tril(
        np.ones((F + 1, F + 1), np.float32), -1
    )

    # ------------------------------------------------------------------
    # Bounded neural backtracking interpreter.
    # ------------------------------------------------------------------
    solver_id = fresh()
    state_ids = [fresh() for _ in range(4)]
    plan_s, hist_s, cur_s, mode_s = state_ids
    for sid, init in zip(
        state_ids, [initial_plan, initial_hist, initial_cur, initial_mode]
    ):
        shapes[sid] = shapes[init]

    old_current = current
    body = []
    current = body

    parsing = mul(veq(mode_s, 0), boot_done)
    backing = mul(veq(mode_s, 1), boot_done)
    top = dense(hist_s, hist_top_w)
    tp, tg, ts = sl(top, 0, 1), sl(top, 1, 2), sl(top, 2, 3)
    cp, cg = sl(cur_s, 0, 1), sl(cur_s, 1, 2)

    has_decision = clip(tp)
    retry = mul(backing, has_decision)
    query_p = select(backing, lin(tp, 1, -1), cp)
    query_g = select(backing, tg, cg)
    query_last_return = select(
        backing, sl(top, K - 1, K), sl(cur_s, CSIZE - 1, CSIZE)
    )
    stack_space = clip(lin(query_last_return, -1, 1))

    pointer_spatial = shat(node("sub", [column_axis, query_p]))
    pointer = gm(node("mul", [column_masks, pointer_spatial]))
    read_token = vsum(mul(pointer, plan_s))

    # Clear the old decision before considering its next alternative.
    cleared_amount = scale(mul(read_token, retry), -1)
    base_plan = add(plan_s, mul(pointer, cleared_amount))

    goal_spatial = shat(node("sub", [column_axis, query_g]))
    target_id = gm(node("mul", [goal_spatial, target_tape]))

    function_index = dense(pointer, frame_index_w)
    capacity = vsum(mul(function_index, capacities))
    slot_index = dense(pointer, slot_index_w)
    at_end = clip(lin(sub(slot_index, capacity), 1, 1))
    not_end = inv(at_end)
    empty = veq(read_token, 0)

    # Count currently assigned token types using the abstract column tape.
    plan_type_matches = shat(node("sub", [column_axis, base_plan]))
    assigned_tape = ssum(plan_type_matches)
    remaining_tape = node("sub", [resource_tape, assigned_tape])

    literal_position = shat(node("sub", [column_axis, target_id]))
    literal_remaining = gm(node("mul", [literal_position, remaining_tape]))
    literal_available = clip(literal_remaining)

    references_remaining = gm(node("mul", [
        reference_columns, remaining_tape
    ]))
    references_available = mul(
        mul(clip(references_remaining), function_exists), stack_space
    )
    choices = cat([literal_available, references_available])

    # New decisions begin at choice zero; retries begin after the old choice.
    cutoff = sub(mul(backing, ts), parsing)
    permitted_w = -np.ones((F + 1, 1), np.float32)
    permitted = clip(dense(
        cutoff, permitted_w, np.arange(F + 1, dtype=np.float32)
    ))
    choices = mul(choices, permitted)
    preceding_choices = dense(choices, choice_prefix_w)
    first_choice = mul(choices, clip(sub(constants[1], preceding_choices)))
    has_choice = vsum(first_choice)
    choice_number = dense(
        first_choice, np.arange(F + 1, dtype=np.float32)[None, :]
    )
    reference_choice_w = np.zeros((1, F + 1), np.float32)
    reference_choice_w[0, 1:] = np.arange(17, 17 + F, dtype=np.float32)
    chosen_token = add(
        mul(sl(first_choice, 0, 1), target_id),
        dense(first_choice, reference_choice_w)
    )

    new_slot = mul(mul(parsing, not_end), empty)
    push_decision = mul(new_slot, has_choice)
    restore_decision = mul(retry, has_choice)
    pop_decision = mul(retry, inv(has_choice))
    assign = add(push_decision, restore_decision)
    next_plan = add(base_plan, mul(pointer, mul(chosen_token, assign)))

    # Update the decision stack.
    push_fields = cat(
        [lin(cp, 1, 1), cg, choice_number] +
        [sl(cur_s, 2 + j, 3 + j) for j in range(R)]
    )
    pushed_history = add(
        dense(hist_s, hist_down_w),
        dense(push_fields, hist_insert_w)
    )
    popped_history = dense(hist_s, hist_up_w)
    hist_negative = scale(hist_s, -1)
    next_hist = add(
        hist_s,
        add(
            mul(push_decision, add(pushed_history, hist_negative)),
            mul(pop_decision, add(popped_history, hist_negative))
        )
    )
    stage_delta = mul(restore_decision, sub(choice_number, ts))
    next_hist = add(next_hist, dense(stage_delta, stage_insert_w))

    # Execute an existing literal, call, or return.
    occupied = inv(empty)
    is_literal = mul(occupied, clip(lin(read_token, -1, 17)))
    is_reference = mul(occupied, inv(is_literal))
    active_slot = mul(parsing, not_end)
    literal_active = mul(active_slot, is_literal)
    reference_active = mul(active_slot, is_reference)
    literal_match = vhat(sub(read_token, target_id))

    literal_event = mul(literal_active, literal_match)
    call_event = mul(reference_active, stack_space)
    end_event = mul(parsing, at_end)
    has_return = clip(sl(cur_s, 2, 3))
    return_event = mul(end_event, has_return)

    parse_failure = add(
        add(
            mul(new_slot, inv(has_choice)),
            mul(literal_active, inv(literal_match))
        ),
        add(
            mul(reference_active, inv(stack_space)),
            mul(end_event, inv(has_return))
        )
    )
    no_decisions = mul(backing, inv(has_decision))
    success = mul(
        literal_event, vhat(sub(lin(cg, 1, 1), target_length))
    )

    called_cur = dense(cat([cur_s, read_token]), call_w, call_b)
    returned_cur = dense(cur_s, return_w, return_b)
    restored_cur = dense(top, restore_w, restore_b)
    cur_negative = scale(cur_s, -1)
    literal_delta_w = np.zeros((CSIZE, 1), np.float32)
    literal_delta_w[0:2, 0] = 1
    next_cur = add(cur_s, dense(literal_event, literal_delta_w))
    for event, transformed in (
        (call_event, called_cur),
        (return_event, returned_cur),
        (restore_decision, restored_cur),
    ):
        next_cur = add(next_cur, mul(event, add(transformed, cur_negative)))

    next_mode = add(
        mode_s,
        add(
            add(parse_failure, scale(no_decisions, 2)),
            sub(scale(success, 2), restore_decision)
        )
    )

    current = old_current
    solver = {
        "id": solver_id, "op": "loop", "T": 1,
        "state": [
            {"id": plan_s, "init": initial_plan},
            {"id": hist_s, "init": initial_hist},
            {"id": cur_s, "init": initial_cur},
            {"id": mode_s, "init": initial_mode},
        ],
        "body": body,
        "next": {
            plan_s: next_plan, hist_s: next_hist,
            cur_s: next_cur, mode_s: next_mode,
        },
    }
    graph.append(solver)
    outputs = []
    for sid in state_ids:
        out = solver_id + "." + sid
        shapes[out] = shapes[sid]
        outputs.append(out)
    plan, history, interpreter, mode = outputs

    # ------------------------------------------------------------------
    # Placement controller.
    # ------------------------------------------------------------------
    ready = mul(boot_done, veq(mode, 2))
    mismatch = mul(
        inv(vhat(sub(plan, actual_slots))),
        clip(plan)
    )
    prefix_w = np.tril(np.ones((N, N), np.float32), -1)
    first_mismatch = mul(
        mismatch,
        clip(sub(constants[1], dense(mismatch, prefix_w)))
    )
    needs_move = vsum(first_mismatch)
    desired_id = vsum(mul(first_mismatch, plan))

    destination = ssum(node("mul", [slot_masks, first_mismatch]))
    source_match = shat(node("sub", [token_id, desired_id]))
    source = node("mul", [source_match, bottom_tokens])

    edit = mul(ready, needs_move)
    test = mul(ready, inv(needs_move))
    cancel = mul(inv(ready), selected_exists)
    click_action = clip(add(edit, cancel))

    source_flag = mul(edit, inv(selected_exists))
    destination_flag = mul(edit, selected_exists)
    click_map = node("add", [
        node("mul", [source, source_flag]),
        node("mul", [destination, destination_flag]),
    ])
    node("add", [
        click_map, node("mul", [selected_click, cancel])
    ], name="click")

    logit_inputs = cat([click_action, test, constants[1]])
    logit_w = np.zeros((8, 3), np.float32)
    logit_w[6, 0] = 10
    logit_w[5, 1] = 10
    logit_w[7, 2] = 1
    dense(logit_inputs, logit_w, name="logits")

    # Initialization undoes a small number of perturbed starting actions.
    failed = veq(mode, 3)
    boot_increment = lin(mem_boot, 1, 1)
    eight_map = node("mul", [one_map, constants[8]])
    next_boot = mul(
        gm(node("min2", [eight_map, boot_increment])),
        inv(failed)
    )
    keep_mode = mul(boot_done, inv(failed))
    stored_mode = add(
        mul(keep_mode, mode),
        scale(inv(keep_mode), -1)
    )
    node("vconcat", [
        plan, history, interpreter, stored_mode, next_boot
    ], name="mem_next")

    # Choose the number of internal reasoning iterations by static budget
    # bookkeeping only. This does not evaluate observations or the game.
    def node_cost(nd):
        op = nd["op"]
        if op == "loop":
            costs = [node_cost(q) for q in nd["body"]]
            mac = nd["T"] * sum(q[0] for q in costs)
            evaluations = 1 + nd["T"] * sum(q[1] for q in costs)
            return mac, evaluations
        out = shapes[nd["id"]]
        count = int(np.prod(out))
        if op == "conv":
            w = params[nd["w"]]
            mac = int(np.prod(w.shape)) * 4096
        elif op == "dense":
            mac = int(np.prod(params[nd["w"]].shape))
        elif op in (
            "add", "sub", "mul", "max2", "min2", "vadd", "vmul"
        ):
            mac = count
        elif op in (
            "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"
        ):
            mac = int(np.prod(shapes[nd["in"][0]]))
        else:
            mac = 0
        return mac, 1

    body_mac = sum(node_cost(q)[0] for q in body)
    body_evals = sum(node_cost(q)[1] for q in body)
    total_mac = sum(node_cost(q)[0] for q in graph)
    total_evals = sum(node_cost(q)[1] for q in graph)
    base_mac = total_mac - body_mac
    base_evals = total_evals - body_evals
    solver["T"] = int(max(1, min(
        64,
        (1_450_000_000 - base_mac) // max(1, body_mac),
        (9700 - base_evals) // max(1, body_evals),
    )))

    def count_nodes(nodes):
        return sum(
            1 + (count_nodes(q["body"]) if q["op"] == "loop" else 0)
            for q in nodes
        )

    assert M <= 256
    assert sum(a.size for a in params.values()) <= 150_000
    assert count_nodes(graph) <= 1000
    assert sum(s[0] for s in shapes.values() if len(s) == 3) <= 4096
    assert max(s[0] for s in shapes.values() if len(s) == 3) <= 512
    assert max(s[0] for s in shapes.values() if len(s) == 1) <= 4096
    assert sum(node_cost(q)[0] for q in graph) <= 1_500_000_000
    assert sum(node_cost(q)[1] for q in graph) <= 10_000
    assert all(np.isfinite(a).all() for a in params.values())

    return {"graph": graph, "memory_size": M, "params": params}

import numpy as np


def build():
    # Capacities for the symbolic workspace.
    S, F, N, D = 24, 6, 24, 24
    I = 16 + F
    M = 2 * S + D + 7 + 4
    colors = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    C = len(colors)

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
    spatial_channels = [19]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        name = fresh("p")
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins, name=None, **kw):
        ident = name or fresh()
        nd = {"id": ident, "op": op, "in": list(ins)}
        nd.update(kw)
        current.append(nd)

        ss = [shapes[x] for x in ins]
        if op == "conv":
            shape = (params[kw["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (params[kw["w"]].shape[0],)
        elif op in ("concat", "vconcat"):
            shape = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("slice", "vslice"):
            shape = (kw["stop"] - kw["start"],) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(s[0] for s in ss), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
        else:
            shape = ss[0]
        shapes[ident] = shape
        if len(shape) == 3:
            spatial_channels[0] += shape[0]
        return ident

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def cat(xs, vector=False):
        if len(xs) == 1:
            return xs[0]
        return emit("vconcat" if vector else "concat", xs)

    def sl(x, a, b):
        return emit(
            "vslice" if len(shapes[x]) == 1 else "slice",
            [x], start=a, stop=b
        )

    def aff(parts, bias=0.0, out=1, name=None):
        ids, ws = [], []
        for x, coefficient in parts:
            n = shapes[x][0]
            w = np.asarray(coefficient, dtype=np.float32)
            if w.ndim == 0:
                if n == out:
                    w = float(w) * np.eye(out, dtype=np.float32)
                elif n == 1:
                    w = np.full((out, 1), float(w), np.float32)
                elif out == 1:
                    w = np.full((1, n), float(w), np.float32)
                else:
                    raise ValueError("ambiguous affine broadcasting")
            elif w.ndim == 1:
                w = w.reshape(out, n)
            else:
                w = w.reshape(out, n)
            ids.append(x)
            ws.append(w)
        b = np.broadcast_to(
            np.asarray(bias, np.float32), (out,)
        ).copy()
        return dense(cat(ids, True), np.concatenate(ws, axis=1), b, name)

    def constant(value):
        v = np.asarray(value, np.float32).reshape(-1)
        return dense("aprev", np.zeros((len(v), 8), np.float32), v)

    def vr(x):
        return emit("vrelu", [x])

    def relu(x):
        return emit("relu", [x])

    def vm(a, b):
        return emit("vmul", [a, b])

    def va(a, b):
        return emit("vadd", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def vmax(x):
        return emit("gmax", [x])

    def summation(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def slinear(x, coefficients, bias=None):
        w = np.asarray(coefficients, np.float32).reshape(
            1, shapes[x][0], 1, 1
        )
        b = None if bias is None else np.array([bias], np.float32)
        return conv(x, w, b)

    def shift(x, dx=0, dy=0):
        c = shapes[x][0]
        radius = max(abs(dx), abs(dy))
        k = 2 * radius + 1
        w = np.zeros((c, c, k, k), np.float32)
        for j in range(c):
            w[j, j, radius - dy, radius - dx] = 1
        return conv(x, w)

    zero_sp = conv(
        "coords", np.zeros((1, 3, 1, 1), np.float32)
    )
    one_sp = conv(
        "coords", np.array([[[[0]], [[0]], [[1]]]], np.float32)
    )
    one_v = constant([1])
    zero_v = constant([0])
    scale4096 = constant([4096])

    def lift(x):
        return add(zero_sp, x)

    def seq(a, b):
        # Exact equality for the integer-valued symbolic quantities.
        if len(shapes[a]) == 1 and len(shapes[b]) == 1:
            a = lift(a)
        difference = sub(a, b)
        absolute = emit("abs", [difference])
        return relu(sub(one_sp, absolute))

    def veq(a, b):
        return vmax(seq(a, b))

    def scaled_mean(x):
        return emit("gmean", [mul(x, scale4096)])

    def table(one_hot, feature):
        return scaled_mean(mul(one_hot, feature))

    def blend(new, old, gate):
        opposite = aff([(gate, -1)], 1)
        return va(vm(new, gate), vm(old, opposite))

    def declare_state(shape):
        ident = fresh("state")
        shapes[ident] = shape
        if len(shape) == 3:
            spatial_channels[0] += shape[0]
        return ident

    def finish_loop(body, states, nxt, T):
        ident = fresh("loop")
        graph.append({
            "id": ident,
            "op": "loop",
            "T": T,
            "state": states,
            "body": body,
            "next": nxt,
        })
        outputs = []
        for st in states:
            result = ident + "." + st["id"]
            shapes[result] = shapes[st["id"]]
            outputs.append(result)
        return ident, outputs

    # ---------------------------------------------------------------
    # Vision: frame and goal corners.
    # A corner is represented two pixels down/right of its outer corner.
    # ---------------------------------------------------------------
    w = np.zeros((2 * C, 16, 7, 7), np.float32)
    edge = [(-2, dx) for dx in range(-2, 4)]
    edge += [(dy, -2) for dy in range(-1, 4)]
    for j, color in enumerate(colors):
        for background, offset in ((4, 0), (5, C)):
            for dy, dx in edge:
                w[offset + j, color, dy + 3, dx + 3] = 1
            w[offset + j, background, 2, 2] += 1
    corners = relu(conv("x0", w, np.full(2 * C, -11, np.float32)))
    frame_colors_raw = sl(corners, 0, C)
    goal_colors = sl(corners, C, 2 * C)

    frame_anchor = slinear(frame_colors_raw, np.ones(C))
    goal_mask = slinear(goal_colors, np.ones(C))
    goal_code = slinear(
        goal_colors, np.array(colors, np.float32) + 1
    )

    # Extend each frame's first slot to subsequent six-pixel slots.
    gate_w = np.zeros((1, 16, 7, 7), np.float32)
    for color in colors:
        gate_w[0, color, 1, 6] = 1
    top_gate = conv("x0", gate_w)

    body = []
    old_current = current
    current = body
    spread_state = declare_state((1, 64, 64))
    q = shift(shift(spread_state, dx=3), dx=3)
    q = emit("min2", [q, top_gate])
    q = emit("max2", [q, frame_anchor])
    current = old_current
    _, spread_out = finish_loop(
        body,
        [{"id": spread_state, "init": frame_anchor}],
        {spread_state: q}, 8
    )
    slot_mask = shift(spread_out[0], dx=2, dy=2)
    frame_colors = shift(frame_colors_raw, dx=2, dy=2)
    first_slot = slinear(frame_colors, np.ones(C))

    # Raster ranks: inclusive horizontal prefix plus preceding rows.
    masks = cat([slot_mask, goal_mask, first_slot])
    row_totals = emit("rowsum", [masks])
    drives = cat([masks, row_totals])
    prefix_w = np.zeros((6, 6, 3, 3), np.float32)
    for j in range(3):
        prefix_w[j, j, 1, 0] = 1
        prefix_w[j + 3, j + 3, 0, 1] = 1

    body = []
    current = body
    prefix_state = declare_state((6, 64, 64))
    pnext = add(conv(prefix_state, prefix_w), drives)
    current = graph
    _, prefix_out = finish_loop(
        body,
        [{"id": prefix_state, "zeros": [6, 64, 64]}],
        {prefix_state: pnext}, 64
    )
    hprefix = sl(prefix_out[0], 0, 3)
    vprefix = sl(prefix_out[0], 3, 6)
    ranks = sub(add(hprefix, vprefix), row_totals)
    slot_rank = sl(ranks, 0, 1)
    goal_rank = sl(ranks, 1, 2)
    frame_rank = sl(ranks, 2, 3)
    target_length = vmax(goal_rank)

    # Map each frame color to the corresponding callable frame number.
    frame_number_by_color = table(frame_colors, frame_rank)

    # Four-by-four token rings: filled and hollow share these 12 pixels.
    ring_w = np.zeros((C, 16, 5, 5), np.float32)
    perimeter = [
        (dy, dx)
        for dy in range(-1, 3)
        for dx in range(-1, 3)
        if dy in (-1, 2) or dx in (-1, 2)
    ]
    for j, color in enumerate(colors):
        for dy, dx in perimeter:
            ring_w[j, color, dy + 2, dx + 2] = 1
    rings = relu(conv("x0", ring_w, np.full(C, -11, np.float32)))
    background_plane = sl("x0", 4, 5)
    hollow = mul(rings, background_plane)
    solid = sub(rings, hollow)
    terminal_code = slinear(solid, np.array(colors, np.float32) + 1)
    call_numbers = aff(
        [(frame_number_by_color, 1)], 16, out=C
    )
    call_code = slinear(mul(hollow, call_numbers), np.ones(C))
    token_code = add(terminal_code, call_code)
    token_mask = slinear(rings, np.ones(C))

    slot_indices = constant(np.arange(1, S + 1))
    goal_indices = constant(np.arange(1, N + 1))
    frame_indices = constant(np.arange(1, F + 1))
    inventory_indices = constant(np.arange(1, I + 1))
    target_indices = constant(np.arange(N))
    call_indices = constant(np.arange(17, 17 + F))

    slot_hot = mul(seq(slot_rank, slot_indices), slot_mask)
    goal_hot = mul(seq(goal_rank, goal_indices), goal_mask)
    frame_hot = mul(seq(frame_rank, frame_indices), first_slot)
    inventory_hot = mul(
        seq(token_code, inventory_indices), token_mask
    )

    initial_program = table(slot_hot, token_code)
    target_table = table(goal_hot, goal_code)
    entry_table = table(frame_hot, slot_rank)
    inventory_table = scaled_mean(inventory_hot)

    following_slot = shift(shift(slot_mask, dx=-3), dx=-3)
    next_code = mul(add(slot_rank, one_sp), following_slot)
    next_table = table(slot_hot, next_code)

    # Static spatial index banks used by the interpreter's equality gates.
    slot_index_bank = lift(slot_indices)
    target_index_bank = lift(target_indices)
    inventory_index_bank = lift(inventory_indices)
    call_index_bank = lift(call_indices)

    # ---------------------------------------------------------------
    # Persistent workspace.
    # ---------------------------------------------------------------
    offset = 0
    mem_program = sl("mem", offset, offset + S)
    offset += S
    mem_order = sl("mem", offset, offset + S)
    offset += S
    mem_stack = sl("mem", offset, offset + D)
    offset += D
    mem_core = sl("mem", offset, offset + 7)
    offset += 7
    control = sl("mem", offset, offset + 4)
    counter = sl(control, 0, 1)
    phase = sl(control, 1, 2)
    remembered_y = sl(control, 2, 3)
    remembered_x = sl(control, 3, 4)

    # Three undo actions, followed by a test that clears any selection.
    warm = vr(aff([(counter, 1)], -3))
    not_warm = aff([(warm, -1)], 1)
    counter_excess = vr(aff([(counter, 1)], -3))
    next_counter = aff([(counter, 1), (counter_excess, -1)], 1)
    initial_core = constant([1, 0, 0, 0, 0, 0, 0])

    # Make integer predicates in a shared two-layer ReLU circuit.
    def predicate_layer(raw_ids, specifications):
        raw = cat(raw_ids, True)
        width = len(raw_ids)
        hidden_w, hidden_b, output_rows = [], [], []

        def hinge(coefficients, bias):
            row = np.zeros(width, np.float32)
            for k, value in coefficients.items():
                row[k] = value
            hidden_w.append(row)
            hidden_b.append(bias)
            return len(hidden_w) - 1

        for specification in specifications:
            terms = []
            for kind, coefficients, threshold, multiplier in specification:
                if kind == "ge":
                    terms.append((
                        hinge(coefficients, 1 - threshold), multiplier
                    ))
                    terms.append((
                        hinge(coefficients, -threshold), -multiplier
                    ))
                else:
                    terms.append((
                        hinge(coefficients, 1 - threshold), multiplier
                    ))
                    terms.append((
                        hinge(coefficients, -threshold), -2 * multiplier
                    ))
                    terms.append((
                        hinge(coefficients, -1 - threshold), multiplier
                    ))
            output_rows.append(terms)

        hidden = vr(dense(
            raw, np.stack(hidden_w), np.array(hidden_b, np.float32)
        ))
        out_w = np.zeros((len(specifications), len(hidden_w)), np.float32)
        for r, terms in enumerate(output_rows):
            for k, value in terms:
                out_w[r, k] += value
        return dense(hidden, out_w)

    def ge(coefficients, threshold):
        return [("ge", coefficients, threshold, 1)]

    def eq(coefficients, value=0):
        return [("eq", coefficients, value, 1)]

    # ---------------------------------------------------------------
    # Neural backtracking interpreter.
    # Core: pc, target position, assignment depth, stack depth,
    #       backtracking flag, non-output timer, solved flag.
    # ---------------------------------------------------------------
    body = []
    current = body
    program_state = declare_state((S,))
    order_state = declare_state((S,))
    stack_state = declare_state((D,))
    core_state = declare_state((7,))

    pc, tp, depth, sp, back, timer, solved = [
        sl(core_state, j, j + 1) for j in range(7)
    ]
    pc_hot = veq(slot_index_bank, pc)
    target_hot = veq(target_index_bank, tp)
    last_hot = veq(depth, order_state)

    current_type = summation(vm(program_state, pc_hot))
    continuation = summation(vm(next_table, pc_hot))
    target_type = summation(vm(target_table, target_hot))
    last_type = summation(vm(program_state, last_hot))

    # Next alternative after a terminal is call 1; after call j, call j+1.
    next_alternative = aff(
        [(vr(aff([(last_type, 1)], -16)), 1)], 17
    )
    candidate_difference = aff([
        (next_alternative, 1), (target_type, -1)
    ])
    candidate = va(target_type, vm(back, candidate_difference))

    candidate_matches = veq(candidate, program_state)
    count_used = summation(candidate_matches)
    stock_hot = veq(inventory_index_bank, candidate)
    capacity = summation(vm(inventory_table, stock_hot))

    current_call_hot = veq(call_index_bank, current_type)
    call_entry = summation(vm(entry_table, current_call_hot))
    stack_top = sl(stack_state, 0, 1)

    raw = [
        pc, tp, depth, sp, back, timer, solved,
        current_type, target_type, candidate, capacity,
        count_used, target_length
    ]
    specs = [
        eq({0: 1}),                         # 0 pc zero
        eq({7: 1}),                         # 1 empty instruction
        [("ge", {7: 1}, 1, 1),
         ("ge", {7: 1}, 17, -1)],            # 2 terminal
        ge({7: 1}, 17),                     # 3 call
        eq({7: 1, 8: -1}),                  # 4 terminal matches target
        eq({3: 1}),                         # 5 empty stack
        ge({3: 1}, D),                      # 6 full stack
        eq({2: 1}),                         # 7 empty assignment trail
        ge({9: -1}, -(16 + F)),             # 8 candidate is in range
        ge({10: 1, 11: -1}, 1),             # 9 inventory has room
        ge({5: 1}, 40),                     # 10 nonproductive recursion
        ge({1: 1, 12: -1}, 0),              # 11 target finished
        ge({4: 1}, 1),                      # 12 backtracking
        ge({6: 1}, 1),                      # 13 solved
    ]
    predicates = predicate_layer(raw, specs)

    # Conjunctions of predicates and their complements.
    event_conditions = [
        # active execution
        [(12, False), (13, False), (11, False)],
        # finish
        [(12, False), (13, False), (11, True)],
        # assign an unassigned instruction
        [(12, False), (13, False), (11, False),
         (0, False), (1, True), (10, False)],
        # emit a terminal
        [(12, False), (13, False), (11, False),
         (0, False), (2, True), (4, True), (10, False)],
        # enter a call
        [(12, False), (13, False), (11, False),
         (0, False), (3, True), (6, False), (10, False)],
        # return
        [(12, False), (13, False), (11, False),
         (0, True), (5, False), (10, False)],
        # replace last choice with its next call alternative
        [(12, True), (13, False), (7, False), (8, True)],
        # exhaust and remove the last choice
        [(12, True), (13, False), (7, False), (8, False)],
    ]
    ew = np.zeros((len(event_conditions), len(specs)), np.float32)
    eb = np.zeros(len(event_conditions), np.float32)
    for j, conditions in enumerate(event_conditions):
        positives = 0
        for k, positive in conditions:
            ew[j, k] = 1 if positive else -1
            positives += int(positive)
        eb[j] = 1 - positives
    events = vr(dense(predicates, ew, eb))
    active, finish, new, emit_terminal, call, ret, replace, pop = [
        sl(events, j, j + 1) for j in range(8)
    ]
    room = sl(predicates, 9, 10)
    rerun = vm(replace, room)
    failed = aff([
        (active, 1), (new, -1), (emit_terminal, -1),
        (call, -1), (ret, -1)
    ])

    # Update program assignments and the chronological choice trail.
    inserted_value = vm(new, candidate)
    replacing_or_popping = aff([(replace, 1), (pop, 1)])
    removed_value = vm(replacing_or_popping, last_type)
    replacement_value = aff([
        (vm(replace, candidate), 1), (removed_value, -1)
    ])
    program_next = va(
        va(program_state, vm(pc_hot, inserted_value)),
        vm(last_hot, replacement_value)
    )

    new_depth_value = vm(new, aff([(depth, 1)], 1))
    removed_depth_value = vm(pop, depth)
    order_next = va(
        va(order_state, vm(pc_hot, new_depth_value)),
        vm(last_hot, aff([(removed_depth_value, -1)]))
    )
    depth_next = aff([(depth, 1), (new, 1), (pop, -1)])

    def pc_change(event, destination):
        delta = aff([(destination, 1), (pc, -1)])
        return vm(event, delta)

    pc_next = aff([
        (pc, 1),
        (pc_change(call, call_entry), 1),
        (pc_change(emit_terminal, continuation), 1),
        (pc_change(ret, stack_top), 1),
        (pc_change(rerun, one_v), 1),
    ])

    keep_run = aff([(rerun, -1)], 1)
    target_next = va(vm(tp, keep_run), emit_terminal)
    stack_depth_next = vm(
        aff([(sp, 1), (call, 1), (ret, -1)]), keep_run
    )

    down = np.zeros((D, D), np.float32)
    up = np.zeros((D, D), np.float32)
    for j in range(1, D):
        down[j, j - 1] = 1
        up[j - 1, j] = 1
    first_vector = np.zeros((D, 1), np.float32)
    first_vector[0, 0] = 1
    pushed = va(
        dense(stack_state, down),
        dense(continuation, first_vector)
    )
    popped = dense(stack_state, up)
    keep_stack = aff([(call, -1), (ret, -1), (rerun, -1)], 1)
    stack_next = va(
        va(vm(stack_state, keep_stack), vm(pushed, call)),
        vm(popped, ret)
    )

    no_room = aff([(room, -1)], 1)
    back_next = aff([
        (back, 1), (failed, 1),
        (vm(new, no_room), 1), (rerun, -1)
    ])
    keep_timer = aff([(emit_terminal, -1), (rerun, -1)], 1)
    timer_next = vm(aff([(timer, 1), (active, 1)]), keep_timer)
    solved_next = va(solved, finish)
    core_next = cat([
        pc_next, target_next, depth_next, stack_depth_next,
        back_next, timer_next, solved_next
    ], True)

    current = graph
    solver_id, solver_outputs = finish_loop(
        body,
        [
            {"id": program_state, "init": mem_program},
            {"id": order_state, "init": mem_order},
            {"id": stack_state, "init": mem_stack},
            {"id": core_state, "init": mem_core},
        ],
        {
            program_state: program_next,
            order_state: order_next,
            stack_state: stack_next,
            core_state: core_next,
        },
        1
    )
    solution, final_order, final_stack, final_core = solver_outputs
    ready = sl(final_core, 6, 7)
    final_depth = sl(final_core, 2, 3)
    final_back = sl(final_core, 4, 5)
    not_ready = aff([(ready, -1)], 1)
    dead = vm(
        warm,
        vm(vm(veq(final_depth, zero_v), final_back), not_ready)
    )

    # ---------------------------------------------------------------
    # Placement policy.
    # ---------------------------------------------------------------
    desired = slinear(mul(slot_hot, solution), np.ones(S))
    equal_here = seq(desired, token_code)
    difference = emit("abs", [sub(desired, token_code)])
    different = emit("min2", [difference, one_sp])
    nonempty_desired = emit("min2", [desired, one_sp])
    mismatch = mul(mul(slot_mask, different), nonempty_desired)

    priority = conv(
        slot_rank, np.array([[[[-1]]]], np.float32),
        np.array([S + 1], np.float32)
    )
    destination_score = mul(mismatch, priority)
    best_destination = vmax(destination_score)
    destination_mask = mul(
        seq(destination_score, best_destination), mismatch
    )
    needs_move = vmax(mismatch)
    wanted_type = vmax(mul(desired, destination_mask))

    correct_slot = mul(slot_mask, equal_here)
    source_mask = mul(
        mul(token_mask, seq(token_code, wanted_type)),
        sub(one_sp, correct_slot)
    )

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    destination_y = vmax(mul(row, destination_mask))
    destination_x = vmax(mul(col, destination_mask))

    source_priority = conv(
        "coords",
        np.array([[[[1]], [[-0.01]], [[10]]]], np.float32)
    )
    source_score = mul(source_mask, source_priority)

    dy = emit("abs", [sub(row, remembered_y)])
    dx = emit("abs", [sub(col, remembered_x)])
    remembered_score = conv(
        add(dy, dx), np.array([[[[-100]]]], np.float32),
        np.array([10], np.float32)
    )
    idle_score = conv(
        "coords",
        np.array([[[[1]], [[-1]], [[0]]]], np.float32)
    )
    not_phase = aff([(phase, -1)], 1)
    source_gate = vm(ready, not_phase)
    destination_gate = vm(ready, phase)
    click_sum = add(
        add(mul(source_score, source_gate),
            mul(remembered_score, destination_gate)),
        mul(idle_score, not_ready)
    )
    conv(click_sum, np.ones((1, 1, 1, 1), np.float32), name="click")

    no_move = aff([(needs_move, -1)], 1)
    submit = vm(vm(vm(ready, no_move), not_phase), warm)

    logit_features = cat([counter, warm, submit, dead], True)
    lw = np.zeros((8, 4), np.float32)
    lb = np.full(8, -100, np.float32)
    lb[6] = 1
    lb[5] = 10
    lw[5, 1] = -10
    lw[5, 2] = 10
    lb[7] = 40
    lw[7, 0] = -10
    lw[7, 1] = -100
    lw[7, 3] = 200
    dense(logit_features, lw, lb, name="logits")

    # Warm-up observations initialize the fixed program. Afterwards all
    # search state is carried exactly, including a completed solution.
    stored_program = va(
        vm(solution, warm), vm(initial_program, not_warm)
    )
    stored_order = vm(final_order, warm)
    stored_stack = vm(final_stack, warm)
    stored_core = va(
        vm(final_core, warm), vm(initial_core, not_warm)
    )
    alive = aff([(dead, -1)], 1)
    stored_counter = vm(next_counter, alive)
    stored_phase = vm(
        vm(vm(vm(ready, not_phase), needs_move), warm), alive
    )
    emit(
        "vconcat",
        [
            stored_program, stored_order, stored_stack, stored_core,
            stored_counter, stored_phase, destination_y, destination_x
        ],
        name="mem_next"
    )

    # Choose the interpreter unrolling from exact static resource counts.
    # This is graph bookkeeping, not weight fitting or game simulation.
    def node_cost(nd):
        op = nd["op"]
        if op == "loop":
            mac, evaluations = list_cost(nd["body"])
            return nd["T"] * mac, nd["T"] * evaluations + 1
        shape = shapes[nd["id"]]
        entries = int(np.prod(shape))
        if op == "conv":
            mac = int(params[nd["w"]].size) * 4096
        elif op == "dense":
            mac = int(params[nd["w"]].size)
        elif op in ("add", "sub", "mul", "max2", "min2", "vadd", "vmul"):
            mac = entries
        elif op in ("rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
            mac = int(np.prod(shapes[nd["in"][0]]))
        else:
            mac = 0
        return mac, 1

    def list_cost(nodes):
        mac = evaluations = 0
        for nd in nodes:
            m, e = node_cost(nd)
            mac += m
            evaluations += e
        return mac, evaluations

    solver_node = next(nd for nd in graph if nd["id"] == solver_id)
    body_mac, body_evaluations = list_cost(solver_node["body"])
    total_mac, total_evaluations = list_cost(graph)
    base_mac = total_mac - body_mac
    base_evaluations = total_evaluations - body_evaluations

    iterations = min(
        256,
        (1_450_000_000 - base_mac) // max(1, body_mac),
        (9_800 - base_evaluations) // max(1, body_evaluations),
    )
    solver_node["T"] = max(1, int(iterations))

    def count_nodes(nodes):
        return sum(
            1 + (count_nodes(nd["body"]) if nd["op"] == "loop" else 0)
            for nd in nodes
        )

    assert sum(a.size for a in params.values()) <= 150_000
    assert count_nodes(graph) <= 1_000
    assert spatial_channels[0] <= 4_096
    assert all(s[0] <= (512 if len(s) == 3 else 4096)
               for s in shapes.values())
    final_mac, final_evaluations = list_cost(graph)
    assert final_mac <= 1_500_000_000
    assert final_evaluations <= 10_000
    assert shapes["mem_next"] == (M,)

    return {"graph": graph, "memory_size": M, "params": params}

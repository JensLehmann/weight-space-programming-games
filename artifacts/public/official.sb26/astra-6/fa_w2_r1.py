import numpy as np


def build():
    N = 32          # Maximum instruction slots
    L = 24          # Maximum target length
    K = 32          # Interpreter stack capacity
    M = 137

    graph = []
    current = graph
    params = {}
    parameter_cache = {}
    constant_cache = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (M,),
        "aprev": (8,),
        "avail": (8,),
    }
    macs = {}
    evaluations = {}
    serial = 0

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a.copy()
        return parameter_cache[key]

    def emit(op, ins, name=None, dest=None, **fields):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        (current if dest is None else dest).append(nd)

        ss = [shapes[x] for x in ins]
        cost = 0
        if op == "conv":
            w = params[fields["w"]]
            out = (w.shape[0], 64, 64)
            cost = int(w.size) * 4096
        elif op == "dense":
            w = params[fields["w"]]
            out = (w.shape[0],)
            cost = int(w.size)
        elif op in ("slice", "vslice"):
            out = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("concat", "vconcat"):
            out = (sum(s[0] for s in ss),) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            out = (ss[0][0],)
            cost = ss[0][0] * 4096
        elif op in ("rowmax", "colmax", "rowsum", "colsum"):
            out = ss[0]
            cost = out[0] * 4096
        elif op in ("add", "sub", "mul", "max2", "min2"):
            out = (max(s[0] for s in ss), 64, 64)
            cost = out[0] * 4096
        elif op in ("vadd", "vmul"):
            out = (max(s[0] for s in ss),)
            cost = out[0]
        else:
            out = ss[0]

        shapes[name] = out
        macs[name] = cost
        evaluations[name] = 1
        return name

    def dense(x, w, b=None, name=None, dest=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(
                np.asarray(b, np.float32).reshape(w.shape[0])
            )
        return emit("dense", [x], name=name, dest=dest, **fields)

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def conv1(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 1:
            w = w[None, :]
        return conv(x, w[:, :, None, None], b)

    def sl(x, start, stop):
        op = "slice" if len(shapes[x]) == 3 else "vslice"
        return emit(op, [x], start=start, stop=stop)

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def catv(*xs):
        return emit("vconcat", xs)

    ones = sl("coords", 2, 3)
    zeros = conv1(ones, [[0.0]])
    one = emit("gmax", [ones])
    zero = emit("gmax", [zeros])

    def const(values):
        a = np.asarray(values, np.float32).reshape(-1)
        if a.size == 1 and a[0] == 1:
            return one
        if a.size == 1 and a[0] == 0:
            return zero
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            constant_cache[key] = dense(
                one, np.zeros((a.size, 1), np.float32),
                a, dest=graph
            )
        return constant_cache[key]

    def neg(x):
        return vm(x, const([-1]))

    def vs(a, b):
        return va(a, neg(b))

    def affine(x, a=1.0, b=0.0):
        return dense(x, [[a]], [b])

    def complement(x):
        if shapes[x][0] == 1:
            return affine(x, -1, 1)
        return va(one, neg(x))

    def total(x):
        return dense(x, np.ones((1, shapes[x][0]), np.float32))

    def dot(a, b):
        return total(vm(a, b))

    def positive(x):
        # Exact indicator x > 0 for integer-valued x.
        z = emit("add", [zeros, x])
        z = emit("relu", [z])
        z = emit("min2", [z, ones])
        return emit("gmax", [z])

    def eq(a, b):
        # Exact equality for integer-valued vectors, with broadcasting.
        z = emit("sub", [zeros, b])
        z = emit("add", [z, a])
        z = emit("abs", [z])
        z = emit("sub", [ones, z])
        z = emit("relu", [z])
        return emit("gmax", [z])

    def indexed_eq(key, count):
        d = dense(
            key, -np.ones((count, 1), np.float32),
            np.arange(1, count + 1, dtype=np.float32)
        )
        z = emit("add", [zeros, d])
        z = emit("abs", [z])
        z = emit("sub", [ones, z])
        z = emit("relu", [z])
        return emit("gmax", [z])

    def first_hot(flags):
        n = shapes[flags][0]
        w = -np.tril(np.ones((n, n), np.float32), -1)
        gate = dense(flags, w, np.ones(n, np.float32))
        gate = emit("vrelu", [gate])
        return vm(flags, gate)

    def spatial_triangle(d):
        d = emit("abs", [d])
        d = emit("sub", [ones, d])
        return emit("relu", [d])

    def spatial_codes(field, count, mask):
        d = conv1(
            field, np.ones((count, 1), np.float32),
            -np.arange(1, count + 1, dtype=np.float32)
        )
        return emit("mul", [spatial_triangle(d), mask])

    def shift_spatial(x, dy, dx, kernel):
        w = np.zeros((1, 1, kernel, kernel), np.float32)
        w[0, 0, kernel // 2 + dy, kernel // 2 + dx] = 1
        return conv(x, w)

    def add_loop(name, body, states, nxt, T):
        nd = {
            "id": name, "op": "loop", "T": int(T),
            "state": states, "body": body, "next": nxt,
        }
        graph.append(nd)
        for s in states:
            shapes[name + "." + s["id"]] = shapes[s["id"]]
        macs[name] = T * sum(macs[n["id"]] for n in body)
        evaluations[name] = 1 + T * sum(
            evaluations[n["id"]] for n in body
        )
        return nd

    # ---------- Visual glyph parser ----------

    ink_weights = np.ones((1, 16), np.float32)
    ink_weights[0, [0, 2, 4, 5]] = 0
    ink = conv1("x0", ink_weights)
    color = conv1(
        "x0", np.arange(1, 17, dtype=np.float32)[None, :]
    )
    features = emit("concat", [
        ink, sl("x0", 4, 5), sl("x0", 2, 3),
        sl("x0", 5, 6), sl("x0", 0, 1)
    ])

    # Anchors are (sprite.x + 2, sprite.y + 2).
    w = np.zeros((5, 5, 5, 5), np.float32)

    def put(out, channel, dy, dx, value=1):
        w[out, channel, dy + 2, dx + 2] += value

    # Solid 4x4 token.
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            put(0, 0, dy, dx)
    put(0, 0, -2, -1, -1)
    put(0, 0, -1, -2, -1)

    # Hollow 4x4 call token.
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            if dy in (-1, 2) or dx in (-1, 2):
                put(1, 0, dy, dx)
            else:
                put(1, 1, dy, dx)
    put(1, 0, -2, -1, -1)
    put(1, 0, -1, -2, -1)

    # Empty slot's 2x2 marker.
    for dy in (0, 1):
        for dx in (0, 1):
            put(2, 2, dy, dx)
    put(2, 2, -1, 0, -1)
    put(2, 2, 0, -1, -1)

    # Target outline on its dark backing.
    for dy, dx in [(-2, -2), (-2, -1), (-1, -2)]:
        put(3, 0, dy, dx)
    put(3, 3, -1, -1)

    # Black selection outline.
    for dy, dx in [(-2, -2), (-2, -1), (-1, -2)]:
        put(4, 4, dy, dx)

    glyphs = emit("relu", [
        conv(features, w, [-15, -15, -3, -3, -2])
    ])
    solid = sl(glyphs, 0, 1)
    hollow = sl(glyphs, 1, 2)
    empty = sl(glyphs, 2, 3)
    target_mask = sl(glyphs, 3, 4)
    selected_mask = sl(glyphs, 4, 5)
    item = emit("add", [solid, hollow])

    region_w = np.zeros((2, 3), np.float32)
    region_w[:, 0] = [-252, 252]
    regions = conv1("coords", region_w, [210, -214])
    regions = emit("relu", [regions])
    regions = emit("min2", [regions, ones])
    board_region = sl(regions, 0, 1)
    stock_region = sl(regions, 1, 2)

    slot_mask = emit("mul", [
        emit("add", [item, empty]), board_region
    ])
    stock_mask = emit("mul", [item, stock_region])

    token_color = shift_spatial(color, -1, -1, 3)
    call_color = conv1(token_color, [[1]], [16])
    token_code = emit("add", [
        emit("mul", [token_color, solid]),
        emit("mul", [call_color, hollow]),
    ])

    owner_code = shift_spatial(color, -3, 0, 7)
    owner_code = shift_spatial(owner_code, -1, 0, 3)
    target_color = shift_spatial(color, -2, -2, 5)

    # ---------- Row-major object ranks ----------

    row_slots = emit("rowsum", [slot_mask])
    row_targets = emit("rowsum", [target_mask])
    rank_drive = emit("concat", [
        slot_mask, target_mask, row_slots, row_targets
    ])

    rank_body = []
    current = rank_body
    shapes["prefix_reg"] = (4, 64, 64)
    pw = np.zeros((4, 4, 3, 3), np.float32)
    for i in range(4):
        pw[i, i, 1 if i < 2 else 0, 0 if i < 2 else 1] = 1
    ph = conv("prefix_reg", pw)
    pn = emit("add", [ph, rank_drive])
    current = graph
    add_loop(
        "prefix", rank_body,
        [{"id": "prefix_reg", "zeros": [4, 64, 64]}],
        {"prefix_reg": pn}, 64
    )

    pr = "prefix.prefix_reg"
    ranks = emit("add", [sl(pr, 0, 2), sl(pr, 2, 4)])
    ranks = emit("sub", [
        ranks, emit("concat", [row_slots, row_targets])
    ])
    slot_rank = sl(ranks, 0, 1)
    target_rank = sl(ranks, 1, 2)

    slot_table = spatial_codes(slot_rank, N, slot_mask)
    target_table = spatial_codes(target_rank, L, target_mask)

    observed_program = emit("gmax", [
        emit("mul", [slot_table, token_code])
    ])
    owners = emit("gmax", [
        emit("mul", [slot_table, owner_code])
    ])
    targets = emit("gmax", [
        emit("mul", [target_table, target_color])
    ])
    target_length = affine(
        emit("gmean", [target_mask]), 4096, 0
    )

    # Entry address for each possible function color.
    owner_masks = spatial_codes(owner_code, 16, slot_mask)
    function_exists = emit("gmax", [owner_masks])
    reverse_rank = conv1(slot_rank, [[-1]], [N + 1])
    entry_score = emit("gmax", [
        emit("mul", [owner_masks, reverse_rank])
    ])
    entries = vs(
        vm(function_exists, const([N + 1])), entry_score
    )
    call_allowed = catv(const(np.zeros(16)), function_exists)

    inventory_masks = spatial_codes(token_code, 32, stock_mask)
    inventory = vm(
        emit("gmean", [inventory_masks]), const([4096])
    )

    shift_next = np.zeros((N, N), np.float32)
    shift_prev = np.zeros((N, N), np.float32)
    for i in range(N - 1):
        shift_next[i, i + 1] = 1
        shift_prev[i + 1, i] = 1

    next_owner = dense(owners, shift_next)
    prev_owner = dense(owners, shift_prev)
    successors = vm(
        eq(owners, next_owner),
        const(np.arange(2, N + 2, dtype=np.float32))
    )
    first_slot = complement(eq(owners, prev_owner))

    # ---------- Search initialization and persistent registers ----------

    layout = [
        ("P", N), ("A", N), ("R", 32), ("S", K),
        ("pc", 1), ("tp", 1), ("depth", 1), ("sp", 1),
        ("E", 1), ("B", 1), ("F", 1),
    ]
    old = {}
    offset = 0
    for key, size in layout:
        old[key] = sl("mem", offset, offset + size)
        offset += size
    age = sl("mem", offset, offset + 1)
    old_signature = sl("mem", offset + 1, offset + 2)

    ready = positive(affine(age, 1, -5))
    signature = dense(
        observed_program,
        np.arange(1, N + 1, dtype=np.float32)[None, :]
    )
    changed = complement(eq(signature, old_signature))
    restart_changed = vm(
        vm(ready, complement(old["F"])), changed
    )
    initialize = positive(va(
        eq(age, const([6])), restart_changed
    ))
    keep = complement(initialize)

    initial = {
        "P": observed_program,
        "A": neg(positive(observed_program)),
        "R": inventory,
        "S": const(np.zeros(K)),
        "pc": one, "tp": one, "depth": zero, "sp": zero,
        "E": one, "B": zero, "F": zero,
    }

    states = []
    reg = {}
    for key, size in layout:
        value = va(
            vm(old[key], keep), vm(initial[key], initialize)
        )
        sid = "reg_" + key
        states.append({"id": sid, "init": value})
        shapes[sid] = (size,)
        reg[key] = sid

    # ---------- Bounded DFS program synthesizer ----------

    body = []
    current = body
    P, A, R, S = [reg[k] for k in ("P", "A", "R", "S")]
    pc, tp, depth, sp = [reg[k] for k in ("pc", "tp", "depth", "sp")]
    E, B, F = [reg[k] for k in ("E", "B", "F")]

    back_valid = vm(B, positive(depth))
    pc_hot = indexed_eq(pc, N)
    trail_hot = eq(A, depth)
    q = va(vm(pc_hot, E), vm(trail_hot, back_valid))

    old_token = dot(q, P)
    context_owner = dot(q, owners)
    context_first = dot(q, first_slot)
    successor = dot(q, successors)

    target_hot = indexed_eq(tp, L)
    wanted_color = dot(target_hot, targets)

    old_hot = indexed_eq(old_token, 32)
    wanted_hot = indexed_eq(wanted_color, 32)
    self_call_hot = indexed_eq(affine(context_owner, 1, 16), 32)

    restored_stock = va(R, vm(old_hot, back_valid))
    available_stock = positive(restored_stock)
    allowed = va(call_allowed, vm(wanted_hot, E))

    greater = dense(
        old_token, -np.ones((32, 1), np.float32),
        np.arange(1, 33, dtype=np.float32)
    )
    greater = positive(greater)
    not_self_first = complement(vm(self_call_hot, context_first))

    candidates = vm(
        vm(available_stock, allowed),
        vm(greater, not_self_first)
    )
    candidate_hot = first_hot(candidates)
    candidate_exists = total(candidate_hot)
    candidate_id = dense(
        candidate_hot, np.arange(1, 33, dtype=np.float32)[None, :]
    )
    no_candidate = complement(candidate_exists)

    old_present = positive(old_token)
    pc_present = positive(pc)
    empty_active = vm(
        vm(E, pc_present), complement(old_present)
    )
    new_assignment = vm(empty_active, candidate_exists)
    back_alternative = vm(back_valid, candidate_exists)
    back_clear = vm(back_valid, no_candidate)
    assignment = va(new_assignment, back_alternative)

    change = va(new_assignment, back_valid)
    token_delta = vm(vs(candidate_id, old_token), change)
    Pn = va(P, vm(q, token_delta))

    timestamp_delta = vs(
        vm(affine(depth, 1, 1), new_assignment),
        vm(depth, back_clear)
    )
    An = va(A, vm(q, timestamp_delta))
    depth_n = va(depth, vs(new_assignment, back_clear))
    Rn = vs(restored_stock, vm(candidate_hot, assignment))

    effective_hot = va(
        old_hot, vm(candidate_hot, new_assignment)
    )
    literal_hot = sl(effective_hot, 0, 16)
    call_hot = sl(effective_hot, 16, 32)
    literal_present = total(literal_hot)
    call_present = total(call_hot)
    literal_match = dot(effective_hot, wanted_hot)
    call_entry = dot(call_hot, entries)

    stack_nonempty = positive(sp)
    stack_room = positive(affine(sp, -1, K))
    entry_valid = positive(call_entry)

    do_call = vm(
        vm(E, call_present), vm(stack_room, entry_valid)
    )
    do_literal = vm(E, literal_match)

    last_target = complement(positive(vs(target_length, tp)))
    success = vm(do_literal, last_target)

    returning = vm(E, complement(pc_present))
    do_pop = vm(returning, stack_nonempty)
    root_end = vm(returning, complement(stack_nonempty))

    failure = va(
        va(
            vm(empty_active, no_candidate),
            vs(vm(E, literal_present), do_literal)
        ),
        va(vs(vm(E, call_present), do_call), root_end)
    )

    stack_top = sl(S, 0, 1)
    pc_delta = va(
        va(
            vm(vs(successor, pc), do_literal),
            vm(vs(call_entry, pc), do_call)
        ),
        va(
            vm(vs(stack_top, pc), do_pop),
            vm(complement(pc), back_alternative)
        )
    )
    pc_n = va(pc, pc_delta)
    tp_n = va(
        va(tp, do_literal),
        vm(complement(tp), back_alternative)
    )
    sp_n = vs(
        va(sp, vs(do_call, do_pop)),
        vm(sp, back_alternative)
    )

    push_delta = -np.eye(K, dtype=np.float32)
    pop_delta = -np.eye(K, dtype=np.float32)
    for i in range(1, K):
        push_delta[i, i - 1] += 1
    for i in range(K - 1):
        pop_delta[i, i + 1] += 1
    insert = np.zeros((K, 1), np.float32)
    insert[0, 0] = 1

    push_change = va(
        dense(S, push_delta), dense(successor, insert)
    )
    pop_change = dense(S, pop_delta)
    Sn = va(
        S, va(vm(push_change, do_call), vm(pop_change, do_pop))
    )

    En = va(vs(vs(E, failure), success), back_alternative)
    Bn = vs(va(B, failure), back_alternative)
    Fn = va(F, success)

    next_values = {
        "P": Pn, "A": An, "R": Rn, "S": Sn,
        "pc": pc_n, "tp": tp_n, "depth": depth_n, "sp": sp_n,
        "E": En, "B": Bn, "F": Fn,
    }
    current = graph
    search_node = add_loop(
        "search", body, states,
        {reg[k]: next_values[k] for k, _ in layout}, 1
    )
    result = {k: "search." + reg[k] for k, _ in layout}

    # ---------- Piece-selection and placement circuit ----------

    desired_program = result["P"]
    mismatch = vm(
        positive(desired_program),
        complement(eq(observed_program, desired_program))
    )
    destination_hot = first_hot(mismatch)
    need_move = total(destination_hot)
    desired_token = dot(destination_hot, desired_program)

    destination = emit("mul", [slot_table, destination_hot])
    destination = conv1(
        destination, np.ones((1, N), np.float32)
    )

    source_difference = emit("sub", [token_code, desired_token])
    source = emit("mul", [
        spatial_triangle(source_difference), stock_mask
    ])
    matching_selected = emit("gmax", [
        emit("mul", [source, selected_mask])
    ])

    source_click = emit("mul", [
        source, complement(matching_selected)
    ])
    destination_click = emit("mul", [
        destination, matching_selected
    ])
    emit("add", [source_click, destination_click], name="click")

    can_act = vm(result["F"], ready)
    action_features = catv(need_move, can_act)
    action_w = np.zeros((8, 2), np.float32)
    action_b = np.full(8, -20, np.float32)
    action_b[5] = 0
    action_w[6, 0] = 2
    action_b[6] = -1
    action_w[7, 1] = -3
    action_b[7] = 2
    dense(
        action_features, action_w, action_b, name="logits"
    )

    memory_parts = [result[k] for k, _ in layout]
    memory_parts += [affine(age, 1, 1), signature]
    emit("vconcat", memory_parts, name="mem_next")

    # Choose the recurrent work allowance using only static graph costs.
    base_macs = sum(
        macs[n["id"]] for n in graph if n["id"] != "search"
    )
    base_evaluations = sum(
        evaluations[n["id"]]
        for n in graph if n["id"] != "search"
    )
    body_macs = sum(macs[n["id"]] for n in body)
    body_evaluations = sum(evaluations[n["id"]] for n in body)

    iterations = min(
        128,
        int((9800 - base_evaluations - 1) // body_evaluations),
        int((1_450_000_000 - base_macs) // max(1, body_macs)),
    )
    search_node["T"] = max(1, iterations)

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }

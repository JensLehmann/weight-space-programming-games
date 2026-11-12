import numpy as np


def build():
    S = 32                 # Maximum number of program slots
    K = 24                 # Interpreter stack capacity
    M = 3 * S + K + 5

    graph = []
    params = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "mem": ("v", M),
        "aprev": ("v", 8),
        "avail": ("v", 8),
    }
    current = graph
    serial = 0
    parameter_cache = {}

    def par(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            params[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def node(op, ins, shape, **fields):
        nonlocal serial
        name = "n%d" % serial
        serial += 1
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current.append(nd)
        shapes[name] = shape
        return name

    def dense(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("dense", [x], ("v", np.asarray(w).shape[0]), **fields)

    def conv(x, w, b=None):
        fields = {"w": par(w)}
        if b is not None:
            fields["b"] = par(b)
        return node("conv", [x], ("s", np.asarray(w).shape[0]), **fields)

    def cat(*xs):
        kind = shapes[xs[0]][0]
        return node(
            "concat" if kind == "s" else "vconcat",
            xs, (kind, sum(shapes[x][1] for x in xs))
        )

    def sl(x, a, b):
        kind = shapes[x][0]
        return node(
            "slice" if kind == "s" else "vslice",
            [x], (kind, b - a), start=a, stop=b
        )

    def unary(op, x):
        return node(op, [x], shapes[x])

    def binary(op, a, b):
        kind = "s" if "s" in (shapes[a][0], shapes[b][0]) else "v"
        return node(op, [a, b], (kind, max(shapes[a][1], shapes[b][1])))

    def add(a, b):
        return binary("add" if "s" in (shapes[a][0], shapes[b][0])
                      else "vadd", a, b)

    def mul(a, b):
        return binary("mul" if "s" in (shapes[a][0], shapes[b][0])
                      else "vmul", a, b)

    def sub(a, b):
        if "s" in (shapes[a][0], shapes[b][0]):
            return binary("sub", a, b)
        return add(a, affine(b, -1.0))

    def affine(x, scale=1.0, bias=0.0):
        kind, n = shapes[x]
        w = np.eye(n, dtype=np.float32) * scale
        b = np.full(n, bias, np.float32)
        if kind == "s":
            return conv(x, w[:, :, None, None], b)
        return dense(x, w, b)

    def constant(value, n=1):
        return dense(
            "avail", np.zeros((n, 8), np.float32),
            np.full(n, value, np.float32)
        )

    one = constant(1.0)
    zero = constant(0.0)

    def clip(x):
        """Exact clipping to [0,1], using permitted operations."""
        kind, n = shapes[x]
        if kind == "s":
            r = unary("relu", x)
            return binary("min2", r, one)
        w = np.concatenate([
            np.eye(n, dtype=np.float32),
            np.eye(n, dtype=np.float32)
        ], axis=0)
        z = dense(x, w, np.concatenate([
            np.zeros(n, np.float32), -np.ones(n, np.float32)
        ]))
        z = unary("vrelu", z)
        return dense(z, np.concatenate([
            np.eye(n, dtype=np.float32),
            -np.eye(n, dtype=np.float32)
        ], axis=1))

    def inv(x):
        return affine(x, -1.0, 1.0)

    def eqs(a, b):
        """Robust equality for integer-valued spatial codes."""
        d = unary("abs", sub(a, b))
        return clip(affine(d, -10.0, 1.5))

    def eqv(a, b):
        """Integer equality for scalar vectors."""
        d = sub(a, b)
        z = dense(
            d, np.ones((3, 1), np.float32),
            np.array([1.0, 0.0, -1.0], np.float32)
        )
        z = unary("vrelu", z)
        return dense(z, np.array([[1.0, -2.0, 1.0]], np.float32))

    def vmax(x):
        return node("gmax", [x], ("v", shapes[x][1]))

    def mean(x):
        return node("gmean", [x], ("v", shapes[x][1]))

    def total(x):
        return affine(mean(x), 4096.0)

    def vsum(x):
        return dense(x, np.ones((1, shapes[x][1]), np.float32))

    def product(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = mul(z, x)
        return z

    def sumall(*xs):
        z = xs[0]
        for x in xs[1:]:
            z = add(z, x)
        return z

    def shift(x, dy, dx):
        """A shift whose offsets are at most three."""
        n = shapes[x][1]
        w = np.zeros((n, n, 7, 7), np.float32)
        for c in range(n):
            w[c, c, 3 + dy, 3 + dx] = 1.0
        return conv(x, w)

    # ------------------------------------------------------------------
    # Visual parsing. Anchors are sprite positions plus (2,2).
    # Tile codes: 1..16 = solid colour+1, 17..32 = call colour+17.
    # ------------------------------------------------------------------
    colours = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nc = len(colours)
    filters = []
    biases = []

    def detector(entries):
        w = np.zeros((16, 7, 7), np.float32)
        for colour, dy, dx in entries:
            w[colour, dy + 3, dx + 3] += 1.0
        filters.append(w)
        biases.append(-len(entries) + 0.5)

    for c in colours:
        detector([(c, y, x) for y in range(-1, 3)
                  for x in range(-1, 3)])

    for c in colours:
        entries = []
        for y in range(-1, 3):
            for x in range(-1, 3):
                border = y in (-1, 2) or x in (-1, 2)
                entries.append((c if border else 4, y, x))
        detector(entries)

    for c in colours:
        entries = []
        for y in range(-2, 4):
            for x in range(-2, 4):
                border = y in (-2, 3) or x in (-2, 3)
                entries.append((c if border else 5, y, x))
        detector(entries)

    detector([(2, y, x) for y in (0, 1) for x in (0, 1)])

    visual = conv("x0", np.stack(filters), np.array(biases, np.float32))
    visual = affine(unary("relu", visual), 2.0)
    solids = sl(visual, 0, nc)
    calls = sl(visual, nc, 2 * nc)
    targets = sl(visual, 2 * nc, 3 * nc)
    spots = sl(visual, 3 * nc, 3 * nc + 1)

    def weighted_spatial(x, weights):
        return conv(x, np.asarray(weights, np.float32)[None, :, None, None])

    solid_presence = weighted_spatial(solids, np.ones(nc))
    call_presence = weighted_spatial(calls, np.ones(nc))
    target_presence = weighted_spatial(targets, np.ones(nc))
    tile_presence = add(solid_presence, call_presence)
    object_presence = clip(add(tile_presence, spots))

    tile_code = add(
        weighted_spatial(solids, [c + 1 for c in colours]),
        weighted_spatial(calls, [c + 17 for c in colours])
    )
    target_code = weighted_spatial(targets, [c + 1 for c in colours])

    row = sl("coords", 0, 1)
    row_pixels = affine(row, 63.0)
    program_region = product(
        clip(affine(row_pixels, 1.0, -7.5)),
        clip(affine(row_pixels, -1.0, 52.5))
    )
    bank_region = clip(affine(row_pixels, 1.0, -53.5))
    slot_mask = mul(object_presence, program_region)
    bank_mask = mul(tile_presence, bank_region)
    observed_program = mul(tile_code, slot_mask)

    index = conv(
        "coords",
        np.array([[[[4032.0]], [[63.0]], [[0.0]]]], np.float32),
        np.array([1.0], np.float32)
    )
    priority = affine(index, -1.0, 4097.0)

    # The routine colour is on the horizontal frame border four rows up.
    above = shift(shift("x0", -3, 0), -1, 0)
    colour_weights = np.arange(1, 17, dtype=np.float32)
    frame_code = weighted_spatial(above, colour_weights)
    frame_code = mul(frame_code, slot_mask)

    def code_planes(x, count):
        w = np.ones((count, 1, 1, 1), np.float32)
        d = conv(x, w, -np.arange(1, count + 1, dtype=np.float32))
        return clip(affine(unary("abs", d), -10.0, 1.5))

    bank_types = mul(code_planes(tile_code, 32), bank_mask)
    bank_counts = total(bank_types)
    frame_types = mul(code_planes(frame_code, 16), slot_mask)
    frame_exists = vmax(frame_types)

    neighbour_mask = shift(shift(slot_mask, 0, 3), 0, 3)
    neighbour_frame = shift(shift(frame_code, 0, 3), 0, 3)
    next_pointer_map = product(
        affine(index, 1.0, 6.0),
        neighbour_mask,
        eqs(frame_code, neighbour_frame),
        slot_mask
    )
    root = affine(vmax(mul(slot_mask, priority)), -1.0, 4097.0)
    target_length = total(target_presence)

    # Rank anchors in row-major order. The resulting rank maps are used
    # to compress spatial search state into the bounded memory vector.
    rank_body = []
    saved = current
    current = rank_body
    rank_states = []
    for name, init in (
        ("rank_remaining_p", slot_mask),
        ("rank_remaining_t", target_presence),
    ):
        shapes[name] = ("s", 1)
        rank_states.append({"id": name, "init": init})
    for name in ("rank_map_p", "rank_map_t"):
        shapes[name] = ("s", 1)
        rank_states.append({"id": name, "zeros": [1, 64, 64]})
    shapes["rank_counter"] = ("v", 1)
    rank_states.append({"id": "rank_counter", "zeros": [1]})

    counter_next = affine("rank_counter", 1.0, 1.0)
    rank_next = {"rank_counter": counter_next}
    for suffix in ("p", "t"):
        rem = "rank_remaining_" + suffix
        rm = "rank_map_" + suffix
        score = mul(rem, priority)
        highest = vmax(score)
        winner = product(
            clip(affine(sub(score, highest), 2.0, 1.0)), rem
        )
        rank_next[rem] = sub(rem, winner)
        rank_next[rm] = add(rm, mul(winner, counter_next))

    current = saved
    graph.append({
        "id": "Ranks", "op": "loop", "T": S,
        "state": rank_states, "body": rank_body, "next": rank_next
    })
    for st in rank_states:
        shapes["Ranks." + st["id"]] = shapes[st["id"]]

    program_rank = "Ranks.rank_map_p"
    target_rank = "Ranks.rank_map_t"
    rank_planes = mul(code_planes(program_rank, S), slot_mask)

    def expand_memory(v):
        return weighted_spatial(mul(rank_planes, v), np.ones(S))

    def compress_memory(x):
        return total(mul(rank_planes, x))

    mp = sl("mem", 0, S)
    md = sl("mem", S, 2 * S)
    mc = sl("mem", 2 * S, 3 * S)
    ms = sl("mem", 3 * S, 3 * S + K)
    mi = sl("mem", 3 * S + K, 3 * S + K + 1)
    mm = sl("mem", 3 * S + K + 1, 3 * S + K + 2)
    started = sl("mem", 3 * S + K + 2, 3 * S + K + 3)
    phase = sl("mem", 3 * S + K + 3, 3 * S + K + 4)
    stage = sl("mem", 3 * S + K + 4, M)

    started = clip(started)
    phase = clip(phase)
    initial = inv(clip(stage))

    root_stack_w = np.zeros((K, 1), np.float32)
    root_stack_w[0, 0] = 1.0
    root_stack = dense(root, root_stack_w)

    init_p = add(
        mul(expand_memory(mp), started),
        mul(observed_program, inv(started))
    )
    init_d = mul(expand_memory(md), started)
    init_c = mul(expand_memory(mc), started)
    init_stack = add(mul(ms, started), mul(root_stack, inv(started)))
    init_idx = mul(mi, started)
    init_mode = mul(mm, started)

    # ------------------------------------------------------------------
    # Discrete program synthesis.
    #
    # P stores assigned token codes. D stores decision timestamps.
    # C stores the terminal colour required when a decision was created.
    # mode: 0 interpret, 1 backtrack, 3 solved, 4 exhausted.
    # ------------------------------------------------------------------
    search_body = []
    current = search_body
    search_states = [
        {"id": "search_p", "init": init_p},
        {"id": "search_d", "init": init_d},
        {"id": "search_c", "init": init_c},
        {"id": "search_stack", "init": init_stack},
        {"id": "search_idx", "init": init_idx},
        {"id": "search_mode", "init": init_mode},
    ]
    for st in search_states:
        shapes[st["id"]] = shapes[st["init"]]

    P = "search_p"
    D = "search_d"
    C = "search_c"
    stack = "search_stack"
    idx = "search_idx"
    mode = "search_mode"

    normal = eqv(mode, zero)
    back = eqv(mode, one)
    pc = sl(stack, 0, 1)
    tail = sl(stack, 1, K)
    pc_valid = clip(pc)
    tail_valid = clip(vsum(tail))
    overflow = clip(sl(stack, K - 1, K))

    pc_mask = mul(eqs(index, pc), slot_mask)
    fetched = vmax(mul(cat(P, next_pointer_map), pc_mask))
    token = sl(fetched, 0, 1)
    next_pc = sl(fetched, 1, 2)

    target_pointer = affine(idx, 1.0, 1.0)
    target_at = mul(eqs(target_rank, target_pointer), target_presence)
    required = vmax(mul(target_code, target_at))
    done = clip(affine(sub(idx, target_length), 1.0, 1.0))

    token_positive = clip(token)
    is_call = clip(affine(token, 1.0, -16.0))
    is_terminal = sub(token_positive, is_call)
    empty = inv(token_positive)
    matches = eqv(token, required)

    decision_max = vmax(D)
    decisions_exist = clip(decision_max)
    variable_mask = clip(D)
    last_mask = product(eqs(D, decision_max), variable_mask)
    last_token = vmax(mul(P, last_mask))
    last_colour = vmax(mul(C, last_mask))

    # Remove the last decision before computing its alternatives.
    removed_mask = mul(last_mask, back)
    base_p = sub(P, mul(P, removed_mask))
    base_d = sub(D, mul(D, removed_mask))

    choice_mask = add(mul(last_mask, back), mul(pc_mask, inv(back)))
    choice_colour = add(
        mul(last_colour, back), mul(required, inv(back))
    )
    threshold = mul(last_token, back)

    assigned_types = mul(code_planes(base_p, 32), clip(base_d))
    remaining = sub(bank_counts, total(assigned_types))
    available_types = clip(remaining)

    # One terminal alternative, plus all available call alternatives.
    def scalar_one_hot(x, n):
        w = np.ones((3 * n, 1), np.float32)
        b = []
        for j in range(1, n + 1):
            b.extend([1.0 - j, -float(j), -1.0 - j])
        z = unary("vrelu", dense(x, w, np.array(b, np.float32)))
        out_w = np.zeros((n, 3 * n), np.float32)
        for j in range(n):
            out_w[j, 3 * j:3 * j + 3] = [1.0, -2.0, 1.0]
        return dense(z, out_w)

    terminal_alternative = scalar_one_hot(choice_colour, 16)
    allowed = cat(terminal_alternative, frame_exists)

    later = clip(dense(
        threshold, -np.ones((32, 1), np.float32),
        np.arange(1, 33, dtype=np.float32)
    ))
    eligible = product(available_types, allowed, later)

    # Select the first eligible token using prefix inhibition.
    prefix_w = np.eye(32, dtype=np.float32)
    for j in range(32):
        prefix_w[j, :j] = -1.0
    first_choice = unary("vrelu", dense(eligible, prefix_w))
    chosen_token = dense(
        first_choice, np.arange(1, 33, dtype=np.float32)[None, :]
    )
    choice_exists = clip(vsum(first_choice))

    active_normal = product(normal, inv(done))
    fresh_choice = product(active_normal, pc_valid, empty)
    back_choice = mul(back, decisions_exist)
    want_choice = add(fresh_choice, back_choice)
    assign = mul(want_choice, choice_exists)
    fresh_no_choice = product(fresh_choice, inv(choice_exists))
    exhausted = mul(back, inv(decisions_exist))

    consume = product(
        active_normal, pc_valid, is_terminal, matches
    )
    push = product(
        active_normal, pc_valid, is_call, inv(overflow)
    )
    pop = product(active_normal, inv(pc_valid), tail_valid)

    failure = mul(active_normal, add(
        product(inv(pc_valid), inv(tail_valid)),
        mul(pc_valid, add(
            mul(is_terminal, inv(matches)),
            mul(is_call, overflow)
        ))
    ))
    success = mul(normal, done)

    assignment_stamp = add(decision_max, inv(back))
    next_p = add(
        base_p, product(choice_mask, chosen_token, assign)
    )
    next_d = add(
        base_d, product(choice_mask, assignment_stamp, assign)
    )
    next_c = add(
        mul(C, inv(mul(choice_mask, fresh_choice))),
        product(choice_mask, choice_colour, fresh_choice)
    )

    call_colour = affine(token, 1.0, -16.0)
    call_region = product(eqs(frame_code, call_colour), slot_mask)
    call_entry = affine(
        vmax(mul(call_region, priority)), -1.0, 4097.0
    )

    consume_stack = cat(next_pc, tail)
    pop_stack = cat(tail, zero)
    push_stack = cat(call_entry, next_pc, sl(stack, 1, K - 1))

    unchanged_stack = inv(sumall(consume, pop, push, assign))
    next_stack = sumall(
        mul(stack, unchanged_stack),
        mul(consume_stack, consume),
        mul(pop_stack, pop),
        mul(push_stack, push),
        mul(root_stack, assign)
    )
    next_idx = mul(add(idx, consume), inv(assign))

    next_mode = sumall(
        mul(mode, inv(sumall(assign, success, exhausted))),
        failure,
        fresh_no_choice,
        affine(success, 3.0),
        affine(exhausted, 4.0)
    )

    search_next = {
        "search_p": next_p,
        "search_d": next_d,
        "search_c": next_c,
        "search_stack": next_stack,
        "search_idx": next_idx,
        "search_mode": next_mode,
    }

    current = graph
    search_loop = {
        "id": "Search", "op": "loop", "T": 1,
        "state": search_states,
        "body": search_body,
        "next": search_next
    }
    graph.append(search_loop)
    for st in search_states:
        shapes["Search." + st["id"]] = shapes[st["id"]]

    final_p = "Search.search_p"
    final_d = "Search.search_d"
    final_c = "Search.search_c"
    final_stack = "Search.search_stack"
    final_idx = "Search.search_idx"
    final_mode = "Search.search_mode"

    solved = eqv(final_mode, constant(3.0))
    unsat = product(eqv(final_mode, constant(4.0)), inv(initial))
    placing = mul(solved, inv(initial))

    # ------------------------------------------------------------------
    # Plan realization: alternate bank selection and destination clicks.
    # Already-correct slots and unreachable unassigned slots are ignored.
    # ------------------------------------------------------------------
    pending = product(
        clip(final_p),
        inv(eqs(final_p, observed_program)),
        slot_mask
    )
    pending_score = mul(pending, priority)
    pending_max = vmax(pending_score)
    destination = product(
        clip(affine(sub(pending_score, pending_max), 2.0, 1.0)),
        pending
    )
    has_pending = clip(total(destination))
    needed_token = vmax(mul(final_p, destination))

    source = product(eqs(tile_code, needed_token), bank_mask)
    bank_click = mul(source, priority)
    destination_click = mul(destination, priority)
    click_map = product(
        add(mul(bank_click, inv(phase)),
            mul(destination_click, phase)),
        placing,
        has_pending
    )
    node("add", [click_map, zero], ("s", 1), **{})
    graph[-1]["id"] = "click"
    shapes["click"] = ("s", 1)

    run_program = product(placing, inv(has_pending))
    action_features = cat(initial, unsat, run_program)
    action_w = np.zeros((8, 3), np.float32)
    action_b = np.array(
        [-2.0, -100.0, -100.0, -100.0, -100.0, 0.0, 1.0, -1.0],
        np.float32
    )
    action_w[0, :2] = 12.0
    action_w[5, 0] = 9.0
    action_w[5, 2] = 10.0
    action_w[7, 1] = 10.0
    action_w[6, :] = -2.0
    logits_id = dense(action_features, action_w, action_b)
    graph[-1]["id"] = "logits"
    shapes["logits"] = shapes[logits_id]

    # Initial actions clear selection. An exhausted search resets the
    # level, or uses undo if RESET is not exposed by the action mask.
    keep = inv(add(initial, unsat))
    next_phase = product(placing, has_pending, inv(phase))
    next_stage = affine(placing, 1.0, 1.0)

    memory_parts = [
        mul(compress_memory(final_p), keep),
        mul(compress_memory(final_d), keep),
        mul(compress_memory(final_c), keep),
        mul(final_stack, keep),
        mul(final_idx, keep),
        mul(final_mode, keep),
        keep,
        next_phase,
        next_stage,
    ]
    memory_id = cat(*memory_parts)
    graph[-1]["id"] = "mem_next"
    shapes["mem_next"] = shapes[memory_id]

    # Choose the largest fixed iteration count satisfying both runtime
    # budgets. This is bookkeeping over the constructed graph, not game
    # simulation or weight optimization.
    def ordinary_cost(nd):
        op = nd["op"]
        out_shape = shapes[nd["id"]]
        if op == "conv":
            w = params[nd["w"]]
            return int(np.prod(w.shape)) * 4096
        if op == "dense":
            return int(np.prod(params[nd["w"]].shape))
        if op in ("add", "sub", "mul", "max2", "min2"):
            return out_shape[1] * 4096
        if op in ("rowmax", "colmax", "rowsum", "colsum",
                  "gmax", "gmean"):
            return shapes[nd["in"][0]][1] * 4096
        if op in ("vadd", "vmul"):
            return out_shape[1]
        return 0

    def cost(nodes):
        macs = 0
        evaluations = 0
        for nd in nodes:
            if nd["op"] == "loop":
                bm, be = cost(nd["body"])
                macs += nd["T"] * bm
                evaluations += 1 + nd["T"] * be
            else:
                macs += ordinary_cost(nd)
                evaluations += 1
        return macs, evaluations

    # Renamed output nodes need their original shape records as well.
    body_macs, body_evals = cost(search_body)
    base_macs, base_evals = cost(graph)
    other_macs = base_macs - body_macs
    other_evals = base_evals - body_evals

    by_evals = (9900 - other_evals) // max(1, body_evals)
    by_macs = (1490000000 - other_macs) // max(1, body_macs)
    search_loop["T"] = int(max(1, min(256, by_evals, by_macs)))

    assert sum(a.size for a in params.values()) <= 150000
    assert shapes["mem_next"] == ("v", M)

    return {"graph": graph, "memory_size": M, "params": params}

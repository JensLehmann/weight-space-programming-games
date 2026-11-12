import numpy as np


def build():
    N = 32                 # Maximum program slots
    G = 32                 # Maximum target length
    K = 48                 # Interpreter return stack
    M = 3 * N + 32 + K + 8

    graph = []
    current = graph
    params = {}
    cache = {}
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "mem": (M,),
        "aprev": (8,),
        "avail": (8,),
    }
    serial = [0]

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            params[name] = a
            cache[key] = name
        return cache[key]

    def emit(op, xs, name=None, **kw):
        name = name or uid()
        nd = {"id": name, "op": op, "in": list(xs)}
        nd.update(kw)
        current.append(nd)

        if op == "conv":
            shape = (params[kw["w"]].shape[0], 64, 64)
        elif op == "dense":
            shape = (params[kw["w"]].shape[0],)
        elif op in ("concat", "vconcat"):
            shape = (sum(shapes[x][0] for x in xs),) + shapes[xs[0]][1:]
        elif op in ("slice", "vslice"):
            shape = (kw["stop"] - kw["start"],) + shapes[xs[0]][1:]
        elif op in ("gmax", "gmean"):
            shape = (shapes[xs[0]][0],)
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = (max(shapes[x][0] for x in xs), 64, 64)
        elif op in ("vadd", "vmul"):
            shape = (max(shapes[x][0] for x in xs),)
        else:
            shape = shapes[xs[0]]
        shapes[name] = shape
        return name

    def conv(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("conv", [x], name=name, **kw)

    def lin(x, w, b=None, name=None):
        kw = {"w": parameter(w)}
        if b is not None:
            kw["b"] = parameter(b)
        return emit("dense", [x], name=name, **kw)

    def aff(xs, ws, b=None):
        z = xs[0] if len(xs) == 1 else emit("vconcat", xs)
        return lin(z, np.concatenate(ws, axis=1), b)

    def forms(xs, rows, biases=None):
        out = len(rows)
        ws = [np.zeros((out, shapes[x][0]), np.float32) for x in xs]
        for i, row in enumerate(rows):
            for (j, col), value in row.items():
                ws[j][i, col] += value
        if biases is None:
            biases = np.zeros(out, np.float32)
        return aff(xs, ws, biases)

    def vr(x):
        return emit("vrelu", [x])

    def sr(x):
        return emit("relu", [x])

    def va(a, b):
        return emit("vadd", [a, b])

    def vm(a, b):
        return emit("vmul", [a, b])

    def vs(x, start, stop=None):
        return emit("vslice", [x], start=start,
                    stop=start + 1 if stop is None else stop)

    def ss(x, start, stop=None):
        return emit("slice", [x], start=start,
                    stop=start + 1 if stop is None else stop)

    def const(values):
        a = np.asarray(values, np.float32).reshape(-1)
        return lin("avail", np.zeros((len(a), 8), np.float32), a)

    def triangle(x):
        # Exact equality-to-zero indicator for integer inputs.
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        w = np.concatenate([eye, eye, eye], axis=0)
        b = np.concatenate([np.ones(n), np.zeros(n), -np.ones(n)])
        h = vr(lin(x, w, b))
        return lin(h, np.concatenate([eye, -2 * eye, eye], axis=1))

    def hot(x, component, count):
        # One-hot encoding of an integer in 1..count.
        d = shapes[x][0]
        w = np.zeros((3 * count, d), np.float32)
        w[:, component] = 1
        centers = np.arange(1, count + 1, dtype=np.float32)
        b = np.concatenate([1 - centers, -centers, -1 - centers])
        h = vr(lin(x, w, b))
        eye = np.eye(count, dtype=np.float32)
        return lin(h, np.concatenate([eye, -2 * eye, eye], axis=1))

    def positive(x):
        # Clip a nonnegative integer to {0,1}; also maps negative values to 0.
        n = shapes[x][0]
        eye = np.eye(n, dtype=np.float32)
        h = vr(lin(x, np.concatenate([eye, eye], axis=0),
                   np.concatenate([np.zeros(n), -np.ones(n)])))
        return lin(h, np.concatenate([eye, -eye], axis=1))

    def first_positive(x):
        # First positive entry of a nonnegative integer vector.
        n = shapes[x][0]
        before = np.tril(np.ones((n, n), np.float32), -1)
        inclusive = np.tril(np.ones((n, n), np.float32))
        h = vr(lin(x, np.concatenate([-before, -inclusive], axis=0),
                   np.ones(2 * n, np.float32)))
        eye = np.eye(n, dtype=np.float32)
        return lin(h, np.concatenate([eye, -eye], axis=1))

    def complement(x):
        n = shapes[x][0]
        return lin(x, -np.eye(n, dtype=np.float32), np.ones(n, np.float32))

    def spatial_linear(x, weights, bias=None):
        w = np.asarray(weights, np.float32)
        return conv(x, w[:, :, None, None], bias)

    I = np.eye(N, dtype=np.float32)
    I32 = np.eye(32, dtype=np.float32)
    I16 = np.eye(16, dtype=np.float32)

    # ------------------------------------------------------------------
    # Visual front end. Anchors are at sprite origin + (2,2).
    # ------------------------------------------------------------------
    colors = [c for c in range(16) if c not in (4, 5)]
    C = len(colors)
    tile_w = np.zeros((2 * C, 16, 5, 5), np.float32)

    for j, color in enumerate(colors):
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                tile_w[j, color, dy + 2, dx + 2] = 1
                if dy in (-1, 2) or dx in (-1, 2):
                    tile_w[C + j, color, dy + 2, dx + 2] = 1
                else:
                    tile_w[C + j, 4, dy + 2, dx + 2] = 1

    tiles = sr(conv("x0", tile_w, -15 * np.ones(2 * C)))

    codes = np.array(
        [c + 1 for c in colors] + [c + 17 for c in colors],
        np.float32,
    )
    tf = spatial_linear(tiles, np.stack([np.ones(2 * C), codes]))
    occupied = ss(tf, 0)
    tile_code = ss(tf, 1)

    empty_w = np.zeros((1, 16, 3, 3), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            empty_w[0, 2, dy + 1, dx + 1] = 1
    empty_w[0, 2, 0, 1] = -4
    empty_w[0, 2, 1, 0] = -4
    empty = sr(conv("x0", empty_w, [-3]))

    goal_w = np.zeros((C, 16, 7, 7), np.float32)
    for j, color in enumerate(colors):
        for dy in range(-2, 4):
            for dx in range(-2, 4):
                cc = color if dy in (-2, 3) or dx in (-2, 3) else 5
                goal_w[j, cc, dy + 3, dx + 3] = 1
    goal_det = sr(conv("x0", goal_w, -35 * np.ones(C)))
    goal_map = spatial_linear(goal_det, np.ones((1, C)))
    goal_code = spatial_linear(
        goal_det, np.array([[c + 1 for c in colors]], np.float32)
    )

    # Exact, saturated lower-palette band.
    bw = np.zeros((2, 3, 1, 1), np.float32)
    bw[:, 0, 0, 0] = 4 * 63
    bh = sr(conv("coords", bw, [-4 * 53.25, -4 * 53.25 - 1]))
    bottom = spatial_linear(bh, [[1, -1]])
    not_bottom = spatial_linear(bottom, [[-1]], [1])

    slot_map = emit("mul", [
        emit("add", [occupied, empty]), not_bottom
    ])
    inventory_map = emit("mul", [occupied, bottom])

    inventory_w = np.zeros((32, 2 * C), np.float32)
    for j, code in enumerate(codes.astype(int)):
        inventory_w[code - 1, j] = 1
    inventory_channels = spatial_linear(tiles, inventory_w)
    inventory_channels = emit("mul", [inventory_channels, bottom])
    inventory_counts = lin(
        emit("gmean", [inventory_channels]), 4096 * I32
    )
    goal_length = lin(emit("gmean", [goal_map]), [[4096]])

    # Frame color is four rows above each slot anchor.
    color_image = spatial_linear(
        "x0", np.arange(1, 17, dtype=np.float32)[None, :]
    )
    sh7 = np.zeros((1, 1, 7, 7), np.float32)
    sh7[0, 0, 0, 3] = 1
    sh3 = np.zeros((1, 1, 3, 3), np.float32)
    sh3[0, 0, 0, 1] = 1
    frame_code = conv(conv(color_image, sh7), sh3)

    # Row-major ranks, using two parallel prefix sums.
    maps = emit("concat", [goal_map, slot_map])
    row_totals = emit("rowsum", [maps])
    rank_drive = emit("concat", [maps, row_totals])

    rank_state = uid("rs")
    shapes[rank_state] = (4, 64, 64)
    rank_body = []
    current = rank_body
    shift = np.zeros((4, 4, 3, 3), np.float32)
    shift[0, 0, 1, 0] = 1
    shift[1, 1, 1, 0] = 1
    shift[2, 2, 0, 1] = 1
    shift[3, 3, 0, 1] = 1
    rank_next = emit("add", [conv(rank_state, shift), rank_drive])
    current = graph
    rank_id = uid("rank")
    graph.append({
        "id": rank_id,
        "op": "loop",
        "T": 64,
        "state": [{"id": rank_state, "zeros": [4, 64, 64]}],
        "body": rank_body,
        "next": {rank_state: rank_next},
    })
    rank_final = rank_id + "." + rank_state
    shapes[rank_final] = (4, 64, 64)
    rank_features = emit("concat", [rank_final, row_totals])
    rw = np.zeros((2, 6), np.float32)
    rw[0, [0, 2, 4]] = [1, 1, -1]
    rw[1, [1, 3, 5]] = [1, 1, -1]
    ranks = spatial_linear(rank_features, rw)

    def ranked_masks(channel, mask, count):
        r = ss(ranks, channel)
        d = spatial_linear(
            r, np.ones((count, 1)),
            -np.arange(1, count + 1, dtype=np.float32),
        )
        d = emit("abs", [d])
        h = sr(spatial_linear(
            d, -np.eye(count, dtype=np.float32), np.ones(count)
        ))
        return emit("mul", [h, mask])

    goal_masks = ranked_masks(0, goal_map, G)
    slot_masks = ranked_masks(1, slot_map, N)

    goals = emit("gmax", [emit("mul", [goal_masks, goal_code])])
    observed_program = emit("gmax", [
        emit("mul", [slot_masks, tile_code])
    ])
    groups = emit("gmax", [emit("mul", [slot_masks, frame_code])])
    exists = emit("gmax", [slot_masks])

    prev_w = np.zeros((N, N), np.float32)
    next_w = np.zeros((N, N), np.float32)
    for i in range(1, N):
        prev_w[i, i - 1] = 1
        next_w[i - 1, i] = 1

    same_previous = triangle(aff(
        [groups, lin(groups, prev_w)], [I, -I]
    ))
    same_next = triangle(aff(
        [groups, lin(groups, next_w)], [I, -I]
    ))
    starts = vm(exists, complement(same_previous))
    lasts = vm(exists, complement(same_next))

    # ------------------------------------------------------------------
    # Persistent solver state and initialization.
    # ------------------------------------------------------------------
    sizes = [N, N, 32, K, N, 6, 1, 1]
    old = []
    offset = 0
    for size in sizes:
        old.append(vs("mem", offset, offset + size))
        offset += size
    old_p, old_a, old_r, old_s, old_v, old_reg, phase, counter = old

    reuse = positive(lin(counter, [[1]], [-6]))
    ready = positive(lin(counter, [[1]], [-5]))
    fresh = complement(reuse)

    def mixed(a, b):
        return va(vm(a, reuse), vm(b, fresh))

    init_p = mixed(old_p, observed_program)
    init_a = vm(old_a, reuse)
    init_r = mixed(old_r, inventory_counts)
    init_s = mixed(old_s, const(np.full(K, N + 1)))
    init_v = vm(old_v, reuse)
    init_reg = mixed(old_reg, const([1, 1, 0, 0, 0, 0]))

    state_names = [uid("q") for _ in range(6)]
    P, A, R, S, V, reg = state_names
    for name, size in zip(state_names, [N, N, 32, K, N, 6]):
        shapes[name] = (size,)

    body = []
    current = body

    # reg = [program counter, target counter, assignment depth,
    #        backtracking flag, success flag, exhausted flag]
    pc_mask = hot(reg, 0, N)
    goal_pointer = hot(reg, 1, G)
    depth_w = np.zeros((N, 6), np.float32)
    depth_w[:, 2] = -1
    last_assignment = triangle(aff([A, reg], [I, depth_w]))

    products = [
        vm(P, pc_mask),
        vm(goals, goal_pointer),
        vm(P, last_assignment),
        vm(lasts, pc_mask),
        pc_mask,
        vm(V, pc_mask),
    ]
    retrieval_ws = []
    for j, x in enumerate(products):
        w = np.zeros((6, shapes[x][0]), np.float32)
        w[j] = 1
        retrieval_ws.append(w)
    ret = aff(products, retrieval_ws)
    # ret = [instruction, required color, last choice, frame-end,
    #        valid-PC, visited-call-site]

    primitive_hot = hot(ret, 1, 32)
    previous_hot = hot(ret, 2, 32)

    pre = vr(forms(
        [ret, reg, goal_length],
        [
            {(0, 0): -1},
            {(0, 0): 1},
            {(0, 0): 1},
            {(0, 0): 1, (0, 1): -1},
            {(0, 1): 1, (0, 0): -1},
            {(1, 0): -1},
            {(1, 0): 1},
            {(1, 2): 1},
            {(1, 2): 1},
            {(1, 1): 1, (2, 0): -1},
            {(2, 0): 1, (1, 1): -1},
        ],
        [1, -16, -17, 0, 0, 1, -N, 0, -1, 0, 0],
    ))

    info = vr(forms(
        [pre, reg],
        [
            {(0, 0): 1},
            {(0, 1): 1, (0, 2): -1},
            {(0, 0): -1, (0, 1): -1, (0, 2): 1},
            {(0, 3): -1, (0, 4): -1},
            {(0, 5): 1},
            {(0, 6): 1},
            {(0, 7): 1, (0, 8): -1},
            {(0, 9): -1, (0, 10): -1},
            {(1, 3): -1, (1, 4): -1, (1, 5): -1},
            {(1, 3): 1, (1, 4): -1, (1, 5): -1},
        ],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    ))

    events = vr(forms(
        [info, ret],
        [
            {(0, 8): 1, (1, 4): 1, (0, 0): 1},
            {(0, 8): 1, (1, 4): 1, (0, 2): 1, (0, 3): 1},
            {(0, 8): 1, (1, 4): 1, (0, 2): 1, (0, 3): -1},
            {(0, 8): 1, (1, 4): 1, (0, 1): 1, (1, 5): -1},
            {(0, 8): 1, (0, 4): 1},
            {(0, 8): 1, (0, 5): 1},
            {(0, 9): 1, (0, 6): 1},
            {(0, 9): 1, (0, 6): -1},
            {(0, 8): 1, (1, 4): 1, (0, 2): 1,
             (0, 3): 1, (0, 7): 1},
            {(0, 8): 1, (1, 4): 1, (0, 1): 1, (1, 5): 1},
        ],
        [-2, -3, -2, -2, -1, -1, -1, 0, -4, -3],
    ))

    ev = [vs(events, i) for i in range(10)]
    new, primitive, bad_primitive, call, returning = ev[:5]
    ended, back, dead_event, success_event, cycle = ev[5:]

    # Candidate generation: demanded literal, then available calls in order.
    previous_bound = vm(vs(ret, 2), back)
    restored = va(R, vm(previous_hot, back))
    call_counts = vs(restored, 16, 32)

    forbidden = vr(lin(
        previous_bound, np.ones((16, 1), np.float32),
        1 - np.arange(17, 33, dtype=np.float32),
    ))
    call_candidates = first_positive(vr(aff(
        [call_counts, forbidden], [I16, -64 * I16]
    )))

    primitive_count = lin(
        vm(restored, primitive_hot), np.ones((1, 32), np.float32)
    )
    no_primitive = vr(lin(primitive_count, [[-1]], [1]))
    use_primitive = vr(forms(
        [no_primitive, reg],
        [{(0, 0): -1, (1, 3): -1}],
        [1],
    ))
    call_hot = lin(
        call_candidates,
        np.concatenate([np.zeros((16, 16), np.float32), I16], axis=0),
    )
    chosen = va(
        vm(primitive_hot, use_primitive),
        vm(call_hot, complement(use_primitive)),
    )
    choice_summary = lin(chosen, np.stack([
        np.arange(1, 33, dtype=np.float32),
        np.ones(32, np.float32),
    ]))
    choice_value = vs(choice_summary, 0)
    found = vs(choice_summary, 1)
    not_found = complement(found)
    attempt = va(new, back)

    Rn = aff([restored, vm(chosen, attempt)], [I32, -I32])

    write_new = vm(pc_mask, new)
    write_back = vm(last_assignment, back)
    write = va(write_new, write_back)
    Pn = aff(
        [P, vm(P, write), vm(write, choice_value)],
        [I, -I, I],
    )

    grow = vm(new, found)
    shrink = vm(back, not_found)
    depth_plus_one = forms([reg], [{(0, 2): 1}], [1])
    age_plus = vm(pc_mask, vm(grow, depth_plus_one))
    age_minus = vm(A, vm(write_back, not_found))
    An = aff([A, age_minus, age_plus], [I, -I, I])

    # Detect nonproductive recursive cycles between literal emissions.
    keep_visits = forms(
        [primitive, back], [{(0, 0): -1, (1, 0): -1}], [1]
    )
    Vn = vm(va(V, vm(pc_mask, call)), keep_visits)

    next_pc = vm(
        forms([reg], [{(0, 0): 1}], [1]),
        forms([ret], [{(0, 3): -1}], [1]),
    )

    function_w = np.zeros((N, 6), np.float32)
    function_w[:, 0] = -1
    function_match = triangle(aff(
        [groups, ret], [I, function_w], 16 * np.ones(N)
    ))
    function_pc = lin(
        vm(function_match, starts),
        np.arange(1, N + 1, dtype=np.float32)[None, :],
    )

    push_w = -np.eye(K, dtype=np.float32)
    for i in range(1, K):
        push_w[i, i - 1] += 1
    push_pc_w = np.zeros((K, 1), np.float32)
    push_pc_w[0, 0] = 1
    push_delta = aff([S, next_pc], [push_w, push_pc_w])

    pop_w = -np.eye(K, dtype=np.float32)
    for i in range(K - 1):
        pop_w[i, i + 1] += 1
    pop_bias = np.zeros(K, np.float32)
    pop_bias[-1] = N + 1
    pop_delta = lin(S, pop_w, pop_bias)
    reset_delta = lin(
        S, -np.eye(K, dtype=np.float32), np.full(K, N + 1)
    )
    Sn = va(va(va(
        S, vm(push_delta, call)),
        vm(pop_delta, returning)),
        vm(reset_delta, back))

    new_failure = vm(new, not_found)
    failure = forms(
        [bad_primitive, ended, cycle, new_failure],
        [{(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1}],
        [0],
    )
    back_finished = vm(back, found)

    branch_deltas = forms(
        [reg, next_pc, function_pc, S],
        [
            {(1, 0): 1, (0, 0): -1},
            {(2, 0): 1, (0, 0): -1},
            {(3, 0): 1, (0, 0): -1},
            {(0, 0): -1},
            {(0, 1): -1},
        ],
        [0, 0, 0, 1, 1],
    )
    branch_gates = forms(
        [events],
        [
            {(0, 1): 1},
            {(0, 3): 1},
            {(0, 4): 1},
            {(0, 6): 1},
            {(0, 6): 1},
        ],
    )
    changes = vm(branch_deltas, branch_gates)

    regn = forms(
        [reg, changes, grow, shrink, failure, back_finished, events],
        [
            {(0, 0): 1, (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1},
            {(0, 1): 1, (1, 4): 1, (6, 1): 1},
            {(0, 2): 1, (2, 0): 1, (3, 0): -1},
            {(0, 3): 1, (4, 0): 1, (5, 0): -1},
            {(0, 4): 1, (6, 8): 1},
            {(0, 5): 1, (6, 7): 1},
        ],
    )

    current = graph
    solver_id = uid("solver")
    initial = [init_p, init_a, init_r, init_s, init_v, init_reg]
    updates = [Pn, An, Rn, Sn, Vn, regn]
    solver_node = {
        "id": solver_id,
        "op": "loop",
        "T": 1,  # Set from the node-evaluation budget below.
        "state": [
            {"id": name, "init": init}
            for name, init in zip(state_names, initial)
        ],
        "body": body,
        "next": dict(zip(state_names, updates)),
    }
    graph.append(solver_node)

    final_state = []
    for name in state_names:
        full_name = solver_id + "." + name
        shapes[full_name] = shapes[name]
        final_state.append(full_name)
    solved_p, solved_a, solved_r, solved_s, solved_v, solved_reg = final_state

    # ------------------------------------------------------------------
    # Read the synthesized assignment and carry out one click at a time.
    # ------------------------------------------------------------------
    equal_observed = triangle(aff(
        [solved_p, observed_program], [I, -I]
    ))
    is_zero = vr(lin(solved_p, -I, np.ones(N)))
    required = complement(is_zero)
    differences = vm(
        vm(required, complement(equal_observed)), exists
    )
    first_difference = first_positive(differences)

    target_summary = lin(first_difference, np.stack([
        np.arange(1, N + 1, dtype=np.float32),
        np.ones(N, np.float32),
    ]))
    target_index = vs(target_summary, 0)
    need_move = vs(target_summary, 1)
    desired_code = lin(
        vm(first_difference, solved_p), np.ones((1, N), np.float32)
    )

    source_distance = emit("abs", [
        emit("sub", [tile_code, desired_code])
    ])
    source_match = sr(spatial_linear(source_distance, [[-1]], [1]))
    source_click = emit("mul", [source_match, inventory_map])

    destination_distance = emit("abs", [
        emit("sub", [ss(ranks, 1), target_index])
    ])
    destination_match = sr(spatial_linear(
        destination_distance, [[-1]], [1]
    ))
    destination_click = emit("mul", [destination_match, slot_map])

    not_phase = complement(phase)
    emit("add", [
        emit("mul", [source_click, not_phase]),
        emit("mul", [destination_click, phase]),
    ], name="click")

    success_ready = vr(forms(
        [solved_reg, ready], [{(0, 4): 1, (1, 0): 1}], [-1]
    ))
    moving = vm(success_ready, need_move)
    executing = vm(success_ready, complement(need_move))
    policy_input = emit("vconcat", [moving, executing])

    policy_w = np.zeros((8, 2), np.float32)
    policy_b = np.full(8, -100, np.float32)
    policy_w[6, 0] = 5
    policy_w[5, 1] = 5
    policy_b[5] = 0
    policy_b[6] = 0
    policy_b[7] = 1
    lin(policy_input, policy_w, policy_b, name="logits")

    phase_next = vm(moving, not_phase)
    counter_next = lin(
        vr(lin(counter, [[-1]], [6])), [[-1]], [7]
    )
    alive = forms([solved_reg], [{(0, 5): -1}], [1])
    counter_next = vm(counter_next, alive)

    emit("vconcat", [
        solved_p, solved_a, solved_r, solved_s, solved_v, solved_reg,
        phase_next, counter_next,
    ], name="mem_next")

    # Budget bookkeeping only: no observations, simulation, or fitting.
    def evaluations(nodes):
        total = 0
        for nd in nodes:
            if nd["op"] == "loop":
                total += 1 + nd["T"] * evaluations(nd["body"])
            else:
                total += 1
        return total

    def macs(nodes):
        total = 0
        for nd in nodes:
            op = nd["op"]
            if op == "loop":
                total += nd["T"] * macs(nd["body"])
                continue
            out = shapes[nd["id"]]
            if op == "conv":
                total += int(params[nd["w"]].size) * 4096
            elif op == "dense":
                total += int(params[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2",
                        "vadd", "vmul"):
                total += int(np.prod(out))
            elif op in ("rowmax", "colmax", "rowsum", "colsum",
                        "gmax", "gmean"):
                total += int(np.prod(shapes[nd["in"][0]]))
        return total

    body_evals = evaluations(body)
    fixed_evals = evaluations(graph) - body_evals
    body_macs = macs(body)
    fixed_macs = macs(graph) - body_macs

    t_eval = (9800 - fixed_evals) // body_evals
    t_mac = int((1490000000 - fixed_macs) // max(1, body_macs))
    solver_node["T"] = max(1, min(256, t_eval, t_mac))

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }

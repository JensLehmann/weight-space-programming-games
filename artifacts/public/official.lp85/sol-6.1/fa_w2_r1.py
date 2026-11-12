import numpy as np


def build():
    N = 36
    S = N + 1
    K = 4
    M = 6 * N + 2
    graph, params, cache = [], {}, {}
    lengths = {"x0": 16, "coords": 3, "mem": M, "aprev": 8, "avail": 8}
    spatial = {"x0", "coords"}
    serial = [0]
    sink = [graph]

    def fresh():
        serial[0] += 1
        return "n%d" % serial[0]

    def parameter(a):
        a = np.asarray(a, np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p%d" % len(params)
            cache[key] = name
            params[name] = a
        return cache[key]

    def node(op, ins, **fields):
        name = fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        sink[0].append(nd)
        if op in ("conv", "dense"):
            lengths[name] = params[fields["w"]].shape[0]
        elif op in ("concat", "vconcat"):
            lengths[name] = sum(lengths[x] for x in ins)
        elif op in ("slice", "vslice"):
            lengths[name] = fields["stop"] - fields["start"]
        else:
            lengths[name] = max(lengths[x] for x in ins)
        if op in ("conv", "concat", "slice"):
            spatial.add(name)
        elif op not in (
            "dense", "vconcat", "vslice", "vrelu", "vtanh", "vsigmoid",
            "vadd", "vmul", "gmax", "gmean"
        ) and any(x in spatial for x in ins):
            spatial.add(name)
        return name

    def conv(x, w, b=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def sl(x, a, b):
        return node(
            "slice" if x in spatial else "vslice",
            [x], start=a, stop=b
        )

    def const(v):
        v = np.asarray(v, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(v), 8), np.float32), v)

    one = const([1])

    def vsub(a, b):
        minus = dense(b, -np.eye(lengths[b], dtype=np.float32))
        return node("vadd", [a, minus])

    def vsum(xs):
        z = xs[0]
        for x in xs[1:]:
            z = node("vadd", [z, x])
        return z

    def vscale(x, a):
        return node("vmul", [x, const([a])])

    def vclip(x):
        return vsub(
            node("vrelu", [x]),
            node("vrelu", [vsub(x, one)])
        )

    def equal_field(a, b):
        d = node("abs", [node("sub", [a, b])])
        return node("relu", [node("sub", [one, d])])

    def winner_field(score, gain=128):
        best = node("gmax", [score])
        d = node("abs", [node("sub", [score, best])])
        d = node("mul", [d, const([gain])])
        return node("relu", [node("sub", [one, d])])

    def paint(mask, values):
        return conv(
            node("mul", [mask, values]),
            np.ones((1, lengths[mask]), np.float32)
        )

    def scalar_sum(x):
        return dense(x, np.ones((1, lengths[x]), np.float32))

    def vector_hot(index, size):
        i = np.arange(size, dtype=np.float32)
        bias = np.concatenate([1 - i, -i, -1 - i])
        ramps = node("vrelu", [
            dense(index, np.ones((3 * size, 1), np.float32), bias)
        ])
        eye = np.eye(size, dtype=np.float32)
        return dense(ramps, np.concatenate([eye, -2 * eye, eye], 1))

    row = node("mul", [sl("coords", 0, 1), const([63])])
    col = node("mul", [sl("coords", 1, 2), const([63])])

    # Exclude the HUD and the vertical step counter.
    yr = conv("coords", [[63, 0, 0]], [-2])
    xr = conv("coords", [[0, 63, 0]], [-0.5])
    yf = node("min2", [node("relu", [yr]), one])
    xf = node("min2", [node("relu", [xr]), one])
    inside = node("mul", [yf, xf])

    def button_anchor(colour):
        plane = node("mul", [sl("x0", colour, colour + 1), inside])
        w = np.zeros((1, 1, 7, 7), np.float32)
        w[0, 0, 3, 3] = 1
        w[0, 0, 2, :] = -1
        w[0, 0, 3, 2] = -1
        return node("relu", [conv(plane, w)])

    left_anchor = button_anchor(8)
    right_anchor = button_anchor(14)

    # Upper-left corners of solid tiles, at either rendering scale.
    colours = [0, 1, 2, 6, 7, 9, 10, 11, 12, 13, 15]
    C = len(colours)
    tw = np.zeros((2 * C, 16, 7, 7), np.float32)
    tb = np.zeros(2 * C, np.float32)
    for scale_index, side in enumerate((2, 4)):
        for ci, colour in enumerate(colours):
            out = scale_index * C + ci
            for y in range(side):
                for x in range(side):
                    tw[out, colour, 3 + y, 3 + x] = 1
            for x in range(side):
                tw[out, colour, 2, 3 + x] = -1
            for y in range(side):
                tw[out, colour, 3 + y, 2] = -1
            tb[out] = 1 - side * side

    detections = node("relu", [conv("x0", tw, tb)])
    small = sl(detections, 0, C)
    large = sl(detections, C, 2 * C)
    is_large = vclip(scalar_sum(node("gmax", [large])))
    is_small = vsub(one, is_large)
    tile_channels = node("add", [
        node("mul", [small, is_small]),
        node("mul", [large, is_large])
    ])
    tiles = conv(tile_channels, np.ones((1, C), np.float32))
    colour_field = conv(
        "x0", np.arange(16, dtype=np.float32).reshape(1, 16)
    )

    # Prefix scans establish stable, observation-derived site indices.
    def scan(x, dy, dx):
        loop_id, state_id = fresh(), fresh()
        lengths[state_id] = lengths[x]
        spatial.add(state_id)
        body = []
        old_sink = sink[0]
        sink[0] = body
        w = np.zeros((1, 1, 3, 3), np.float32)
        w[0, 0, 1 + dy, 1 + dx] = 1
        shifted = conv(state_id, w)
        nxt = node("add", [shifted, x])
        sink[0] = old_sink
        sink[0].append({
            "id": loop_id, "op": "loop", "T": 63,
            "state": [{"id": state_id, "init": x}],
            "body": body, "next": {state_id: nxt}
        })
        result = loop_id + "." + state_id
        lengths[result] = lengths[x]
        spatial.add(result)
        return result

    def rank_field(anchor):
        rows = node("rowsum", [anchor])
        above = node("sub", [scan(rows, -1, 0), rows])
        inclusive = scan(anchor, 0, -1)
        rank = conv(node("add", [above, inclusive]), [[1]], [-1])
        penalty = conv(anchor, [[-1000]], [1000])
        return rank, node("add", [rank, penalty])

    rank, masked_rank = rank_field(tiles)
    _, right_rank = rank_field(right_anchor)
    site_mask = equal_field(
        masked_rank, const(np.arange(N, dtype=np.float32))
    )
    right_mask = equal_field(
        right_rank, const(np.arange(K, dtype=np.float32))
    )
    present = node("gmax", [site_mask])
    controls = node("gmax", [right_mask])
    number_controls = scalar_sum(controls)
    site_y = node("gmax", [node("mul", [site_mask, row])])
    site_x = node("gmax", [node("mul", [site_mask, col])])
    colours_now = node("gmax", [
        node("mul", [site_mask, colour_field])
    ])

    # Destination brackets are distinct from the movable goal tiles.
    bracket_planes = node("concat", [
        sl("x0", 11, 12), sl("x0", 12, 13)
    ])

    def brackets(offsets):
        w = np.zeros((2, 2, 7, 7), np.float32)
        for c in range(2):
            for dy, dx in offsets:
                w[c, c, 3 + dy, 3 + dx] = 1
        return node("relu", [conv(bracket_planes, w, [-3, -3])])

    brackets_small = brackets([
        (-1, -1), (-1, 2), (2, -1), (2, 2)
    ])
    brackets_large = brackets([
        (-3, -3), (-3, 3), (3, -3), (3, 3)
    ])
    shift = np.zeros((2, 2, 3, 3), np.float32)
    shift[0, 0, 2, 2] = 1
    shift[1, 1, 2, 2] = 1
    brackets_large = conv(brackets_large, shift)
    bracket_field = node("add", [
        node("mul", [brackets_small, is_small]),
        node("mul", [brackets_large, is_large])
    ])
    target_b = node("gmax", [
        node("mul", [site_mask, sl(bracket_field, 0, 1)])
    ])
    target_c = node("gmax", [
        node("mul", [site_mask, sl(bracket_field, 1, 2)])
    ])

    # Pair physical controls by spatial proximity.
    right_y = node("gmax", [node("mul", [right_mask, row])])
    right_x = node("gmax", [node("mul", [right_mask, col])])
    ldy = node("abs", [node("sub", [row, right_y])])
    ldx = node("abs", [node("sub", [col, right_x])])
    left_score = node("mul", [
        node("add", [ldy, ldx]), const([-1])
    ])
    left_score = node("sub", [
        left_score, conv(left_anchor, [[-1000]], [1000])
    ])
    tie = conv("coords", [[-0.4032, -0.0063, 0]])
    left_score = node("add", [left_score, tie])
    paired_left_mask = winner_field(left_score, 65536)

    # Memory: four learned left permutations, history, initial sample,
    # probing clock, and completion flag.
    old_permutations = [
        sl("mem", g * N, (g + 1) * N) for g in range(K)
    ]
    history = sl("mem", 4 * N, 5 * N)
    initial = sl("mem", 5 * N, 6 * N)
    clock = sl("mem", 6 * N, 6 * N + 1)
    old_complete = sl("mem", 6 * N + 1, 6 * N + 2)

    clock_hot = vector_hot(clock, 13)
    learn_w = np.zeros((K, 13), np.float32)
    done_w = np.zeros((K, 13), np.float32)
    for g in range(K):
        learn_w[g, 3 * g:3 * g + 3] = 1
        done_w[g, 3 * g + 3] = 1
    learn_controls = node("vmul", [dense(clock_hot, learn_w), controls])
    learning = scalar_sum(learn_controls)
    done = node("vmul", [
        dense(clock_hot, done_w), vsub(one, old_complete)
    ])

    phase_zero_w = np.zeros((1, 13), np.float32)
    phase_zero_w[0, ::3] = 1
    phase_zero = dense(clock_hot, phase_zero_w)
    reset_history = vsum([
        vsub(one, learning), node("vmul", [learning, phase_zero])
    ])
    history_updated = vsum([vscale(history, 16), colours_now])
    powers = np.array([
        1 if t == 0 else (16, 256, 4096)[(t - 1) % 3]
        for t in range(13)
    ], np.float32).reshape(1, 13)
    growth = dense(clock_hot, powers)
    adjusted = vsub(
        history_updated, node("vmul", [growth, initial])
    )
    signature_field = paint(site_mask, adjusted)

    # Shifted colour histories reveal source-to-destination motion.
    difference = node("abs", [
        node("sub", [signature_field, history])
    ])
    signature_score = node("mul", [difference, const([-10000])])
    dy = node("abs", [node("sub", [row, site_y])])
    dx = node("abs", [node("sub", [col, site_x])])
    distance = node("add", [dy, dx])
    rank_penalty = node("mul", [rank, const([0.02])])
    invalid_penalty = conv(tiles, [[-10000000]], [10000000])
    score = node("sub", [signature_score, distance])
    score = node("sub", [score, rank_penalty])
    score = node("sub", [score, invalid_penalty])
    inferred = node("gmax", [
        node("mul", [winner_field(score), rank])
    ])
    inferred = vsum([
        node("vmul", [present, inferred]),
        node("vmul", [
            vsub(one, present), const(np.arange(N, dtype=np.float32))
        ])
    ])

    permutations = []
    for g in range(K):
        gate = sl(done, g, g + 1)
        permutations.append(vsum([
            node("vmul", [gate, inferred]),
            node("vmul", [vsub(one, gate), old_permutations[g]])
        ]))

    # One-hot axes for an S-by-S joint-state table.
    def fixed_axis(axis):
        thresholds = np.arange(-1, S + 1, dtype=np.float32)
        w = np.zeros((S + 2, 3), np.float32)
        w[:, axis] = 63
        ramps = node("relu", [conv("coords", w, -thresholds)])
        out = np.zeros((S, S + 2), np.float32)
        for i in range(S):
            out[i, i:i + 3] = [1, -2, 1]
        return conv(ramps, out)

    row_mask = fixed_axis(0)
    col_mask = fixed_axis(1)
    valid_rows = node("min2", [
        node("relu", [node("sub", [const([S]), row])]), one
    ])

    append_dummy = np.zeros((S, N), np.float32)
    append_dummy[:N] = np.eye(N, dtype=np.float32)
    dummy_bias = np.zeros(S, np.float32)
    dummy_bias[N] = N
    forward = [
        dense(p, append_dummy, dummy_bias) for p in permutations
    ]
    inverse = []
    for p in forward:
        p_field = paint(row_mask, p)
        matrix = node("mul", [
            equal_field(p_field, col), valid_rows
        ])
        inverse_field = node("colmax", [
            node("mul", [matrix, row])
        ])
        inverse.append(node("gmax", [
            node("mul", [inverse_field, col_mask])
        ]))
    differences = [vsub(f, i) for f, i in zip(forward, inverse)]

    # Extract the movable goals and assign the absent-goal dummy site.
    goal_b = node("mul", [tiles, sl("x0", 11, 12)])
    goal_c = node("mul", [tiles, sl("x0", 12, 13)])
    nb = vscale(node("gmean", [goal_b]), 4096)
    nc = vscale(node("gmean", [goal_c]), 4096)
    b1, c1 = vclip(nb), vclip(nc)
    b2, c2 = vclip(vsub(nb, one)), vclip(vsub(nc, one))
    reversed_rank = node("sub", [const([N]), rank])
    min_b = vsub(const([N]), node("gmax", [
        node("mul", [goal_b, reversed_rank])
    ]))
    min_c = vsub(const([N]), node("gmax", [
        node("mul", [goal_c, reversed_rank])
    ]))
    max_b = node("gmax", [node("mul", [goal_b, rank])])
    max_c = node("gmax", [node("mul", [goal_c, rank])])

    type1_b = b1
    type1_c = node("vmul", [vsub(one, b1), c1])
    type1_d = vsub(one, vsum([type1_b, type1_c]))
    type2_b = b2
    type2_c = node("vmul", [
        vsub(one, b2),
        vsum([
            node("vmul", [b1, c1]),
            node("vmul", [vsub(one, b1), c2])
        ])
    ])
    type2_d = vsub(one, vsum([type2_b, type2_c]))

    position1 = vsum([
        node("vmul", [type1_b, min_b]),
        node("vmul", [type1_c, min_c]),
        vscale(type1_d, N)
    ])
    position2 = vsum([
        node("vmul", [type2_b, max_b]),
        node("vmul", [type2_c, max_c]),
        vscale(type2_d, N)
    ])
    current1 = vector_hot(position1, S)
    current2 = vector_hot(position2, S)

    tb = dense(target_b, append_dummy)
    tc = dense(target_c, append_dummy)
    dummy = np.zeros(S, np.float32)
    dummy[N] = 1
    td = const(dummy)

    def destinations(b, c, d):
        return vsum([
            node("vmul", [b, tb]),
            node("vmul", [c, tc]),
            node("vmul", [d, td])
        ])

    terminal_row = paint(
        row_mask, destinations(type1_b, type1_c, type1_d)
    )
    terminal_col = paint(
        col_mask, destinations(type2_b, type2_c, type2_d)
    )
    off_diagonal = node("sub", [one, equal_field(row, col)])
    terminal = node("mul", [
        node("mul", [terminal_row, terminal_col]), off_diagonal
    ])

    # Monotone Bellman iteration: retained values make alternating
    # direction updates accumulate rather than erase earlier reachability.
    loop_id, value_state, phase_state = fresh(), fresh(), fresh()
    lengths[value_state] = lengths[phase_state] = 1
    spatial.add(value_state)
    body = []
    old_sink = sink[0]
    sink[0] = body

    def gather(value, permutation):
        selected_rows = equal_field(row, permutation)
        selected = node("colmax", [
            node("mul", [value, selected_rows])
        ])
        row_gathered = paint(row_mask, selected)
        selected_cols = equal_field(col, permutation)
        selected = node("rowmax", [
            node("mul", [row_gathered, selected_cols])
        ])
        return paint(col_mask, selected)

    best = None
    for g in range(K):
        chosen = vsum([
            inverse[g],
            node("vmul", [phase_state, differences[g]])
        ])
        candidate = gather(value_state, chosen)
        candidate = node("mul", [candidate, sl(controls, g, g + 1)])
        best = candidate if best is None else node("max2", [best, candidate])
    discounted = node("mul", [best, const([0.98])])
    reached = node("max2", [terminal, discounted])
    next_value = node("max2", [value_state, reached])
    next_phase = dense(phase_state, [[-1]], [1])
    sink[0] = old_sink
    graph.append({
        "id": loop_id, "op": "loop", "T": 96,
        "state": [
            {"id": value_state, "init": terminal},
            {"id": phase_state, "zeros": [1]}
        ],
        "body": body,
        "next": {value_state: next_value, phase_state: next_phase}
    })
    value = loop_id + "." + value_state
    lengths[value] = 1
    spatial.add(value)

    def action_value(p, gate):
        next1 = scalar_sum(node("vmul", [p, current1]))
        next2 = scalar_sum(node("vmul", [p, current2]))
        point = node("mul", [
            equal_field(row, next1), equal_field(col, next2)
        ])
        q = node("gmax", [node("mul", [value, point])])
        return node("vmul", [q, gate])

    qleft, qright = [], []
    for g in range(K):
        gate = sl(controls, g, g + 1)
        qleft.append(action_value(forward[g], gate))
        qright.append(action_value(inverse[g], gate))
    qleft = node("vconcat", qleft)
    qright = node("vconcat", qright)
    has_route = vclip(vscale(
        scalar_sum(node("vconcat", [qleft, qright])), 1000000
    ))

    planning = vsub(one, learning)
    left_scores = vsum([
        vscale(learn_controls, 2),
        node("vmul", [planning, qleft]),
        vscale(controls, 0.000001)
    ])
    right_scores = node("vmul", [planning, qright])
    click_channels = node("concat", [
        node("mul", [right_mask, right_scores]),
        node("mul", [paired_left_mask, left_scores])
    ])
    conv(click_channels, np.ones((1, 2 * K), np.float32))
    graph[-1]["id"] = "click"

    total_probes = vscale(number_controls, 3)
    clock_plus_one = vsum([clock, one])
    excess = node("vrelu", [vsub(clock_plus_one, total_probes)])
    clock_next = vsub(clock_plus_one, excess)
    keep_clock = vclip(vsum([learning, has_route]))
    clock_next = node("vmul", [clock_next, keep_clock])
    complete_now = vclip(vsum([vsub(clock, total_probes), one]))
    complete_next = node("vmul", [complete_now, has_route])

    history_next = vsum([
        node("vmul", [reset_history, colours_now]),
        node("vmul", [vsub(one, reset_history), history_updated])
    ])
    initial_next = vsum([
        node("vmul", [reset_history, colours_now]),
        node("vmul", [vsub(one, reset_history), initial])
    ])
    node("vconcat", permutations + [
        history_next, initial_next, clock_next, complete_next
    ])
    graph[-1]["id"] = "mem_next"

    logits_bias = np.full(8, -1, np.float32)
    logits_bias[6] = 10
    dense("avail", np.zeros((8, 8), np.float32), logits_bias)
    graph[-1]["id"] = "logits"

    return {"graph": graph, "memory_size": M, "params": params}

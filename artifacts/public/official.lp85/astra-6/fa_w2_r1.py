import numpy as np


def build():
    N = 48
    M = 5 * N + 13
    graph = []
    params = {}
    parameter_cache = {}
    constant_cache = {}
    counter = 0
    current = graph

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p%d" % len(params)
            parameter_cache[key] = name
            params[name] = a.copy()
        return parameter_cache[key]

    def node(op, ins=None, **fields):
        nonlocal counter
        name = "n%d" % counter
        counter += 1
        d = {"id": name, "op": op}
        if ins is not None:
            d["in"] = list(ins)
        d.update(fields)
        current.append(d)
        return name

    def conv(x, w, b=None):
        w = np.asarray(w, dtype=np.float32)
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

    def const(values):
        a = np.asarray(values, dtype=np.float32).reshape(-1)
        key = (a.shape, a.tobytes())
        if key not in constant_cache:
            constant_cache[key] = dense(
                "avail", np.zeros((len(a), 8), np.float32), a
            )
        return constant_cache[key]

    def sl(x, start, stop, vector=False):
        return node(
            "vslice" if vector else "slice",
            [x], start=int(start), stop=int(stop)
        )

    def cat(xs, vector=False):
        return node("vconcat" if vector else "concat", xs)

    def aff(x, a=1.0, b=0.0):
        return conv(x, [[a]], [b])

    def va(x, size, a=1.0, b=0.0):
        return dense(
            x, np.eye(size, dtype=np.float32) * a,
            np.full(size, b, np.float32)
        )

    def vneg(x, size):
        return dense(x, -np.eye(size, dtype=np.float32))

    def vclip(x, size):
        lo = node("vrelu", [x])
        hi = node("vrelu", [va(x, size, 1, -1)])
        return node("vadd", [lo, vneg(hi, size)])

    one = const([1])
    halfwidth = const([1.5])
    labels = const(2 * np.arange(N, dtype=np.float32))
    button_labels = const(2 * np.arange(4, dtype=np.float32))

    def eq(a, b):
        z = node("sub", [a, b])
        z = node("abs", [z])
        z = node("sub", [halfwidth, z])
        z = node("relu", [z])
        return node("min2", [z, one])

    def clip_spatial(x):
        return node("min2", [node("relu", [x]), one])

    def sum_channels(x, channels):
        return conv(x, np.ones((1, channels), np.float32))

    def scatter(v, basis):
        return sum_channels(node("mul", [basis, v]), N)

    def extract(x, basis, factor):
        z = node("gmean", [node("mul", [basis, x])])
        return dense(z, np.eye(N, dtype=np.float32) * factor)

    def spatial_sum(xs):
        result = xs[0]
        for x in xs[1:]:
            result = node("add", [result, x])
        return result

    def vector_sum(xs):
        result = xs[0]
        for x in xs[1:]:
            result = node("vadd", [result, x])
        return result

    def vmax(v, size):
        while size > 1:
            out = (size + 1) // 2
            dw = np.zeros((out, size), np.float32)
            bw = np.zeros((out, size), np.float32)
            for j in range(out):
                a = 2 * j
                b = min(a + 1, size - 1)
                dw[j, a] += 1
                dw[j, b] -= 1
                bw[j, b] = 1
            d = node("vrelu", [dense(v, dw)])
            v = node("vadd", [d, dense(v, bw)])
            size = out
        return v

    # Artificial matrix coordinates for tile identities.
    rid = conv("coords", [[126, 0, 0]])
    cid = conv("coords", [[0, 126, 0]])
    rb = eq(rid, labels)
    cb = eq(cid, labels)

    # ---------------------------------------------------------------
    # Visual parsing.
    # ---------------------------------------------------------------
    regular_colors = [0, 1, 2, 6, 7, 9, 10, 13, 15]
    tile_colors = regular_colors + [11, 12]

    w = np.zeros((1, 16), np.float32)
    w[0, regular_colors] = 1
    regular = conv("x0", w)

    w = np.zeros((1, 1, 3, 3), np.float32)
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        w[0, 0, dy + 1, dx + 1] = 1
    large = node("gmax", [
        node("relu", [conv(regular, w, [-4])])
    ])
    small = dense(large, [[-1]], [1])

    w = np.zeros((1, 16), np.float32)
    w[0, tile_colors] = 1
    tile_pixels = conv("x0", w)

    anchor_versions = []
    for step in (1, 2):
        w = np.zeros((1, 1, 5, 5), np.float32)
        for dy, dx in [(0, 0), (0, step), (step, 0), (step, step)]:
            w[0, 0, dy + 2, dx + 2] += 1
        w[0, 0, 1, 2] = -4
        w[0, 0, 2, 1] = -4
        anchor_versions.append(
            node("relu", [conv(tile_pixels, w, [-3])])
        )
    anchors = node("add", [
        node("mul", [anchor_versions[0], small]),
        node("mul", [anchor_versions[1], large])
    ])

    # Right and left arrow components, excluding the UI.
    w = np.zeros((2, 16), np.float32)
    w[0, 14] = 1
    w[1, 8] = 1
    buttons = conv("x0", w)
    ui_y = clip_spatial(conv("coords", [[126, 0, 0]], [-5]))
    ui_x = clip_spatial(conv("coords", [[0, 126, 0]], [-1]))
    buttons = node("mul", [
        buttons, node("mul", [ui_y, ui_x])
    ])

    w = np.zeros((2, 2, 5, 5), np.float32)
    for c in range(2):
        w[c, c, 2, 2] = 1
        w[c, c, 2, 1] = -1
        for dy in (-2, -1):
            for dx in range(-2, 3):
                w[c, c, dy + 2, dx + 2] = -1
    button_anchors = node("relu", [conv(buttons, w)])

    # Fixed target frames surrounding tiles.
    w = np.zeros((2, 16), np.float32)
    w[0, 11] = 1
    w[1, 12] = 1
    marker_pixels = conv("x0", w)

    w = np.zeros((2, 2, 5, 5), np.float32)
    for c in range(2):
        for dy in (-1, 2):
            for dx in (-1, 2):
                w[c, c, dy + 2, dx + 2] = 1
    marker_small = node("relu", [
        conv(marker_pixels, w, [-3, -3])
    ])

    w = np.zeros((2, 2, 3, 3), np.float32)
    for c in range(2):
        w[c, c, 2, 2] = 1
    marker_shift = conv(marker_pixels, w)

    w = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        for dy in (-3, 3):
            for dx in (-3, 3):
                w[c, c, dy + 3, dx + 3] = 1
    marker_large = node("relu", [
        conv(marker_shift, w, [-3, -3])
    ])
    markers = node("add", [
        node("mul", [marker_small, small]),
        node("mul", [marker_large, large])
    ])

    # Row-major ranks for tiles and the two kinds of controls.
    rank_input = cat([anchors, button_anchors])

    w = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        w[c, c, 1, 0] = 1

    parent = current
    body = []
    current = body
    hshift = conv("horizontal_state", w)
    hnext = node("add", [hshift, rank_input])
    current = parent
    hloop = node(
        "loop", T=64,
        state=[{"id": "horizontal_state", "zeros": [3, 64, 64]}],
        body=body,
        next={"horizontal_state": hnext}
    )
    hp = hloop + ".horizontal_state"
    row_totals = node("rowmax", [hp])

    w = np.zeros((3, 3, 3, 3), np.float32)
    for c in range(3):
        w[c, c, 0, 1] = 1

    parent = current
    body = []
    current = body
    vshift = conv("vertical_state", w)
    vnext = node("add", [vshift, row_totals])
    current = parent
    vloop = node(
        "loop", T=64,
        state=[{"id": "vertical_state", "zeros": [3, 64, 64]}],
        body=body,
        next={"vertical_state": vnext}
    )
    vp = vloop + ".vertical_state"

    ranks = node("sub", [node("add", [hp, vp]), row_totals])
    ranks = conv(
        ranks, 2 * np.eye(3, dtype=np.float32), [-2, -2, -2]
    )

    tile_basis = node("mul", [
        eq(sl(ranks, 0, 1), labels), anchors
    ])
    right_basis = node("mul", [
        eq(sl(ranks, 1, 2), button_labels),
        sl(button_anchors, 0, 1)
    ])
    left_basis = node("mul", [
        eq(sl(ranks, 2, 3), button_labels),
        sl(button_anchors, 1, 2)
    ])

    button_planes = []
    for j in range(4):
        button_planes += [
            sl(right_basis, j, j + 1),
            sl(left_basis, j, j + 1)
        ]
    button_basis = cat(button_planes)
    valid_buttons = node("gmax", [button_basis])

    color_image = conv(
        "x0", (2 * np.arange(16, dtype=np.float32))[None, :]
    )
    colors = extract(color_image, tile_basis, 4096)
    ys = extract(rid, tile_basis, 4096)
    xs = extract(cid, tile_basis, 4096)
    target_b = extract(sl(markers, 0, 1), tile_basis, 4096)
    target_c = extract(sl(markers, 1, 2), tile_basis, 4096)

    # ---------------------------------------------------------------
    # Persistent empirical transition model.
    # ---------------------------------------------------------------
    previous_colors = sl("mem", 4 * N, 5 * N, True)
    counts = sl("mem", 5 * N, 5 * N + 4, True)
    previous_choice = sl("mem", 5 * N + 4, 5 * N + 12, True)
    initialized = sl("mem", 5 * N + 12, 5 * N + 13, True)
    initialize = dense(initialized, [[-1]], [1])
    initial_identity = node("vmul", [labels, initialize])

    maps = []
    for j in range(4):
        maps.append(node("vadd", [
            sl("mem", j * N, (j + 1) * N, True),
            initial_identity
        ]))

    def matrices(mapping):
        encoded_columns = scatter(mapping, cb)
        encoded_rows = scatter(mapping, rb)
        return eq(rid, encoded_columns), eq(cid, encoded_rows)

    old_matrices = [matrices(m) for m in maps]

    current_color_rows = scatter(colors, rb)
    previous_color_cols = scatter(previous_colors, cb)
    same_color = eq(current_color_rows, previous_color_cols)

    dy = node("abs", [
        node("sub", [scatter(ys, rb), scatter(ys, cb)])
    ])
    dx = node("abs", [
        node("sub", [scatter(xs, rb), scatter(xs, cb)])
    ])
    distance = node("max2", [dy, dx])

    previous_parts = []
    for j, (p, pt) in enumerate(old_matrices):
        previous_parts += [
            node("mul", [
                p, sl(previous_choice, 2 * j, 2 * j + 1, True)
            ]),
            node("mul", [
                pt, sl(previous_choice, 2 * j + 1, 2 * j + 2, True)
            ])
        ]
    previous_prediction = spatial_sum(previous_parts)

    match_score = conv(
        cat([same_color, previous_prediction, distance, rid, cid]),
        [[1000000, 100000, -256, -8, -0.125]]
    )

    valid_rows = sum_channels(rb, N)
    valid_cols = sum_channels(cb, N)
    free_initial = node("mul", [valid_rows, valid_cols])

    # Parallel greedy bipartite matching between consecutive frames.
    parent = current
    body = []
    current = body
    penalty = aff("matching_free", 4000000, -4000000)
    active_score = node("add", [match_score, penalty])
    row_best = node("rowmax", [active_score])
    col_best = node("colmax", [active_score])
    deficits = node("add", [
        node("sub", [active_score, row_best]),
        node("sub", [active_score, col_best])
    ])
    chosen = clip_spatial(aff(deficits, 8, 1))
    chosen = node("mul", [chosen, "matching_free"])
    used_rows = node("rowmax", [chosen])
    used_cols = node("colmax", [chosen])
    next_free = node("relu", [
        node("sub", [
            node("sub", ["matching_free", used_rows]),
            used_cols
        ])
    ])
    next_matching = node("add", ["matching_result", chosen])
    current = parent
    matching_loop = node(
        "loop", T=N,
        state=[
            {"id": "matching_free", "init": free_initial},
            {"id": "matching_result", "zeros": [1, 64, 64]}
        ],
        body=body,
        next={
            "matching_free": next_free,
            "matching_result": next_matching
        }
    )
    assignment = matching_loop + ".matching_result"

    observed_forward = extract(
        node("colsum", [node("mul", [assignment, rid])]),
        cb, 64
    )
    observed_reverse = extract(
        node("rowsum", [node("mul", [assignment, cid])]),
        rb, 64
    )

    updated_maps = []
    for j in range(4):
        sr = sl(previous_choice, 2 * j, 2 * j + 1, True)
        slf = sl(previous_choice, 2 * j + 1, 2 * j + 2, True)
        used = node("vadd", [sr, slf])
        unused = dense(used, [[-1]], [1])
        updated_maps.append(vector_sum([
            node("vmul", [maps[j], unused]),
            node("vmul", [observed_forward, sr]),
            node("vmul", [observed_reverse, slf])
        ]))

    updated_matrices = [matrices(m) for m in updated_maps]
    action_maps = []
    for m, (p, pt) in zip(updated_maps, updated_matrices):
        inverse = extract(
            node("rowsum", [node("mul", [p, cid])]), rb, 64
        )
        action_maps.extend([m, inverse])
    all_action_maps = cat(action_maps, True)

    # ---------------------------------------------------------------
    # Goal identities and joint-state value iteration.
    # ---------------------------------------------------------------
    goal_b = eq(current_color_rows, const([22]))
    goal_c = eq(current_color_rows, const([24]))
    has_b = node("gmax", [goal_b])
    has_c = node("gmax", [goal_c])
    no_b = dense(has_b, [[-1]], [1])
    both = node("vmul", [has_b, has_c])
    not_both = dense(both, [[-1]], [1])

    primary = node("add", [
        node("mul", [goal_b, has_b]),
        node("mul", [goal_c, no_b])
    ])
    secondary = node("add", [
        node("mul", [goal_c, both]),
        node("mul", [primary, not_both])
    ])

    first_reversed = node("gmax", [
        node("mul", [primary, aff(rid, -1, 2 * N)])
    ])
    first_index = dense(first_reversed, [[-1]], [2 * N])
    second_index = node("gmax", [
        node("mul", [secondary, rid])
    ])
    first_source_col = eq(cid, first_index)
    second_source_row = eq(rid, second_index)

    tb_row = scatter(target_b, rb)
    tb_col = scatter(target_b, cb)
    tc_row = scatter(target_c, rb)
    tc_col = scatter(target_c, cb)

    target_first = node("add", [
        node("mul", [tb_row, has_b]),
        node("mul", [tc_row, no_b])
    ])
    target_primary_col = node("add", [
        node("mul", [tb_col, has_b]),
        node("mul", [tc_col, no_b])
    ])
    target_second = node("add", [
        node("mul", [tc_col, both]),
        node("mul", [target_primary_col, not_both])
    ])
    initial_value = aff(
        node("mul", [target_first, target_second]), 100, 0
    )

    phase_initial = const([1, 0, 0, 0, 0, 0, 0, 0])
    expand_phase = np.zeros((8 * N, 8), np.float32)
    combine_maps = np.zeros((N, 8 * N), np.float32)
    for a in range(8):
        expand_phase[a * N:(a + 1) * N, a] = 1
        combine_maps[:, a * N:(a + 1) * N] = np.eye(
            N, dtype=np.float32
        )
    rotate_phase = np.zeros((8, 8), np.float32)
    for a in range(8):
        rotate_phase[(a + 1) % 8, a] = 1

    parent = current
    body = []
    current = body
    phase_weights = dense("planning_phase", expand_phase)
    selected_mapping = dense(
        node("vmul", [all_action_maps, phase_weights]),
        combine_maps
    )
    next_phase = dense("planning_phase", rotate_phase)

    active_button = dense(
        node("vmul", ["planning_phase", valid_buttons]),
        np.ones((1, 8), np.float32)
    )
    move_cost = dense(active_button, [[-1000]], [1001])

    row_gather_mask = eq(rid, selected_mapping)
    col_gather_mask = eq(cid, selected_mapping)

    gathered = node("colsum", [
        node("mul", ["planning_value", row_gather_mask])
    ])
    gathered = sum_channels(node("mul", [gathered, rb]), N)
    gathered = node("rowsum", [
        node("mul", [gathered, col_gather_mask])
    ])
    gathered = sum_channels(node("mul", [gathered, cb]), N)
    proposal = node("sub", [gathered, move_cost])
    next_value = node("max2", ["planning_value", proposal])
    current = parent
    planning_loop = node(
        "loop", T=256,
        state=[
            {"id": "planning_value", "init": initial_value},
            {"id": "planning_phase", "init": phase_initial}
        ],
        body=body,
        next={
            "planning_value": next_value,
            "planning_phase": next_phase
        }
    )
    value = planning_loop + ".planning_value"

    next_first = []
    next_second = []
    for p, pt in updated_matrices:
        for forward, transpose in ((p, pt), (pt, p)):
            next_first.append(node("rowsum", [
                node("mul", [forward, first_source_col])
            ]))
            next_second.append(node("colsum", [
                node("mul", [transpose, second_source_row])
            ]))
    next_states = node("mul", [
        cat(next_first), cat(next_second)
    ])
    q = node("gmax", [node("mul", [next_states, value])])
    q = node("vmul", [q, valid_buttons])
    best_q = vmax(q, 8)

    # ---------------------------------------------------------------
    # Probe briefly, then follow the empirical value function.
    # ---------------------------------------------------------------
    pair_matrix = np.zeros((4, 8), np.float32)
    for j in range(4):
        pair_matrix[j, 2 * j:2 * j + 2] = 1
    pair_present = vclip(dense(valid_buttons, pair_matrix), 4)
    under_sampled = vclip(va(counts, 4, -1, 3), 4)
    needs_probe = vclip(
        dense(
            node("vmul", [under_sampled, pair_present]),
            np.ones((1, 4), np.float32)
        ), 1
    )
    has_plan = vclip(best_q, 1)
    use_plan = node("vmul", [
        has_plan, dense(needs_probe, [[-1]], [1])
    ])
    use_probe = dense(use_plan, [[-1]], [1])

    probe_pairs = node("vadd", [
        dense(
            under_sampled,
            np.diag(np.array([1000, 990, 980, 970], np.float32))
        ),
        vneg(counts, 4)
    ])
    expand_pairs = np.zeros((8, 4), np.float32)
    left_penalty = np.zeros((8, 8), np.float32)
    for j in range(4):
        expand_pairs[2 * j:2 * j + 2, j] = 1
        left_penalty[2 * j + 1, 2 * j] = -10000
    probe_scores = node("vadd", [
        dense(probe_pairs, expand_pairs),
        dense(valid_buttons, left_penalty)
    ])
    scores = node("vadd", [
        node("vmul", [q, use_plan]),
        node("vmul", [probe_scores, use_probe])
    ])
    scores = dense(
        scores, np.eye(8, dtype=np.float32),
        -np.arange(8, dtype=np.float32) / 32
    )

    shifted_scores = va(scores, 8, 1, 20000)
    click_map = aff(
        sum_channels(
            node("mul", [button_basis, shifted_scores]), 8
        ),
        1, -20000
    )
    best_score = node("gmax", [click_map])

    choice_pre = node("vadd", [
        va(scores, 8, 64, 1),
        dense(best_score, np.full((8, 1), -64, np.float32))
    ])
    choice = node("vmul", [
        vclip(choice_pre, 8), valid_buttons
    ])

    # ---------------------------------------------------------------
    # Budget safeguard.
    #
    # The leftmost UI column fills with colour 5 as actions are spent.
    # Preserve the learned model across RESET, but do not record RESET
    # as a button transition.
    # ---------------------------------------------------------------
    leftmost_column = node("relu", [
        conv("coords", [[0, -63, 0]], [1])
    ])
    spent_pixels = dense(
        node("gmean", [
            node("mul", [sl("x0", 5, 6), leftmost_column])
        ]),
        [[4096]]
    )
    near_exhaustion = vclip(
        dense(spent_pixels, [[1]], [-57]), 1
    )

    # Q=100 means the next click reaches the goal immediately.
    immediate_win = vclip(dense(best_q, [[1]], [-99]), 1)
    actually_finishing = node("vmul", [immediate_win, use_plan])
    reset_request = node("vmul", [
        near_exhaustion,
        dense(actually_finishing, [[-1]], [1])
    ])
    reset_request = node("vmul", [
        reset_request, sl("avail", 0, 1, True)
    ])
    executing_click = dense(reset_request, [[-1]], [1])
    recorded_choice = node("vmul", [choice, executing_click])

    next_counts = node("vadd", [
        counts, dense(recorded_choice, pair_matrix)
    ])

    memory_output = cat(
        updated_maps + [
            colors, next_counts, recorded_choice, one
        ],
        True
    )

    node("vslice", [memory_output], start=0, stop=M)
    current[-1]["id"] = "mem_next"

    node("slice", [click_map], start=0, stop=1)
    current[-1]["id"] = "click"

    logits_weight = np.zeros((8, 1), np.float32)
    logits_weight[0, 0] = 2
    logits_bias = np.zeros(8, np.float32)
    logits_bias[6] = 1
    node(
        "dense", [reset_request],
        w=parameter(logits_weight),
        b=parameter(logits_bias)
    )
    current[-1]["id"] = "logits"

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }

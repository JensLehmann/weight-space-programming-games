import numpy as np


def build():
    N = 48
    K = 4
    graph = []
    params = {}
    active = graph
    serial = [0]
    const_cache = {}
    sum_cache = {}

    def uid(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(value):
        name = uid("p")
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def emit(op, ins=None, name=None, target=None, **fields):
        node = {"id": name or uid(), "op": op}
        if ins is not None:
            node["in"] = list(ins)
        node.update(fields)
        (active if target is None else target).append(node)
        return node["id"]

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit("dense", [x], name=name, **fields)

    def sl(x, start, stop, vector=False):
        return emit("vslice" if vector else "slice", [x],
                    start=start, stop=stop)

    one_image = sl("coords", 2, 3)
    unit = emit("gmax", [one_image])

    def constant(values):
        a = np.asarray(values, dtype=np.float32).reshape(-1)
        key = (a.size, a.tobytes())
        if key not in const_cache:
            # Constants are always top-level, including those requested
            # while constructing a loop body.
            w = parameter(np.zeros((a.size, 1), np.float32))
            b = parameter(a)
            const_cache[key] = emit(
                "dense", [unit], target=graph, w=w, b=b
            )
        return const_cache[key]

    c0 = constant([0])
    c1 = constant([1])
    c2 = constant([2])
    cm1 = constant([-1])
    cm4 = constant([-4])

    def add(a, b):
        return emit("add", [a, b])

    def sub(a, b):
        return emit("sub", [a, b])

    def mul(a, b):
        return emit("mul", [a, b])

    def relu(a):
        return emit("relu", [a])

    def scale(a, value):
        return mul(a, constant([value]))

    def affine(a, weight, bias):
        return add(scale(a, weight), constant([bias]))

    def complement(a):
        return sub(one_image, a)

    def eq_diff(d):
        a = emit("abs", [d])
        a = add(mul(a, cm4), c2)
        a = relu(a)
        return emit("min2", [a, c1])

    def eq(a, b):
        return eq_diff(sub(a, b))

    def vadd(a, b):
        return emit("vadd", [a, b])

    def vmul(a, b):
        return emit("vmul", [a, b])

    def vscale(a, value):
        return vmul(a, constant([value]))

    def vsub(a, b):
        return vadd(a, vmul(b, cm1))

    def vcomp(a):
        return vadd(c1, vmul(a, cm1))

    def vclip(a):
        a = emit("vrelu", [a])
        excess = emit("vrelu", [vadd(a, cm1)])
        return vsub(a, excess)

    def vsum(a, size):
        return dense(a, np.ones((1, size), np.float32))

    def sum_channels(x, count):
        if count not in sum_cache:
            sum_cache[count] = parameter(
                np.ones((1, count, 1, 1), np.float32)
            )
        return emit("conv", [x], w=sum_cache[count])

    def finish_loop(loop_name, states, body, next_values, iterations):
        graph.append({
            "id": loop_name,
            "op": "loop",
            "T": iterations,
            "state": states,
            "body": body,
            "next": next_values,
        })

    # Physical display coordinates.
    wr = np.zeros((1, 3), np.float32)
    wc = np.zeros((1, 3), np.float32)
    wr[0, 0] = 63
    wc[0, 1] = 63
    row_index = conv("coords", wr)
    col_index = conv("coords", wc)

    # Solid tile-square detectors at the two relevant display scales.
    valid_colours = [1, 2, 9, 10, 11, 12, 15]
    w = np.zeros((1, 16), np.float32)
    w[0, valid_colours] = 1
    tile_ink = conv("x0", w)

    def square_detector(size):
        kernel = 3 if size == 2 else 7
        centre = kernel // 2
        w = np.zeros((1, 1, kernel, kernel), np.float32)
        for dy in range(size):
            for dx in range(size):
                w[0, 0, centre + dy, centre + dx] = 1
        w[0, 0, centre - 1, centre] = -(size * size)
        w[0, 0, centre, centre - 1] = -(size * size)
        return relu(conv(tile_ink, w, [1 - size * size]))

    small_tiles = square_detector(2)
    large_tiles = square_detector(4)
    large_scale = emit("gmax", [large_tiles])
    small_scale = vcomp(large_scale)
    tile_anchor = add(mul(small_tiles, small_scale),
                      mul(large_tiles, large_scale))

    # Marker corners, expressed at the corresponding tile anchor.
    def marker_detector(large):
        w = np.zeros((2, 16, 7, 7), np.float32)
        offsets = (-3, 3) if large else (-1, 2)
        for out, colour in enumerate((11, 12)):
            for dy in offsets:
                for dx in offsets:
                    w[out, colour, dy + 3, dx + 3] = 1
        z = relu(conv("x0", w, [-3, -3]))
        if large:
            shift = np.zeros((2, 2, 3, 3), np.float32)
            shift[0, 0, 2, 2] = 1
            shift[1, 1, 2, 2] = 1
            z = conv(z, shift)
        return z

    marker_small = marker_detector(False)
    marker_large = marker_detector(True)
    marker_planes = add(mul(marker_small, small_scale),
                        mul(marker_large, large_scale))
    marker_code = conv(marker_planes, [[11, 12]])

    colour_w = np.zeros((1, 16), np.float32)
    for colour in valid_colours:
        colour_w[0, colour] = colour
    colour_code = conv("x0", colour_w)

    # A bottom-right extremum supplies one clickable anchor per arrow.
    button_w = np.zeros((2, 16), np.float32)
    button_w[0, 14] = 1       # R buttons
    button_w[1, 8] = 1        # L buttons
    button_ink = conv("x0", button_w)

    usable_row = emit("min2", [relu(affine(row_index, 2, -3)), c1])
    usable_col = emit("min2", [relu(affine(col_index, 2, -3)), c1])
    usable_display = mul(usable_row, usable_col)
    button_ink = mul(button_ink, usable_display)

    extrema_w = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        extrema_w[c, c, 3, 3] = 1
        for dy in range(1, 4):
            for dx in range(-3, 4):
                extrema_w[c, c, dy + 3, dx + 3] = -1
        for dx in range(1, 4):
            extrema_w[c, c, 3, dx + 3] = -1
    button_anchors = relu(conv(button_ink, extrema_w))
    r_anchor = sl(button_anchors, 0, 1)
    l_anchor = sl(button_anchors, 1, 2)

    # Raster-order ranks for tiles and for each button colour.
    anchors = emit("concat", [tile_anchor, r_anchor, l_anchor])
    row_counts = emit("rowsum", [anchors])
    prefix_name = uid("prefix")
    su, sh = uid("up"), uid("left")
    prefix_body = []
    active = prefix_body

    up_w = np.zeros((3, 3, 3, 3), np.float32)
    left_w = np.zeros_like(up_w)
    for i in range(3):
        up_w[i, i, 0, 1] = 1
        left_w[i, i, 1, 0] = 1
    next_u = add(conv(su, up_w), row_counts)
    next_h = add(conv(sh, left_w), anchors)

    active = graph
    finish_loop(
        prefix_name,
        [{"id": su, "zeros": [3, 64, 64]},
         {"id": sh, "zeros": [3, 64, 64]}],
        prefix_body, {su: next_u, sh: next_h}, 64
    )
    ranks = add(sub(prefix_name + "." + su, row_counts),
                prefix_name + "." + sh)
    tile_rank = sl(ranks, 0, 1)

    rank_difference = conv(
        tile_rank, np.ones((N, 1), np.float32),
        -np.arange(1, N + 1, dtype=np.float32)
    )
    tile_basis = mul(eq_diff(rank_difference), tile_anchor)

    def extract_board(field):
        return emit("gmax", [mul(tile_basis, field)])

    colours = extract_board(colour_code)
    targets = extract_board(marker_code)
    xpos = extract_board(col_index)
    ypos = extract_board(row_index)

    # Synthetic N-by-N coordinate space used by the matching and planner.
    def synthetic_basis(axis):
        w = np.zeros((N, 3), np.float32)
        w[:, axis] = 63
        difference = conv(
            "coords", w, -np.arange(N, dtype=np.float32)
        )
        return eq_diff(difference)

    row_basis = synthetic_basis(0)
    col_basis = synthetic_basis(1)

    def embed(vector, basis):
        return sum_channels(mul(basis, vector), N)

    current_row = embed(colours, row_basis)
    current_col = embed(colours, col_basis)
    previous_colours = sl("mem", 0, N, vector=True)
    previous_row = embed(previous_colours, row_basis)
    previous_col = embed(previous_colours, col_basis)

    xrow = embed(xpos, row_basis)
    xcol = embed(xpos, col_basis)
    yrow = embed(ypos, row_basis)
    ycol = embed(ypos, col_basis)

    valid_row = emit("min2", [current_row, c1])
    valid_col = emit("min2", [current_col, c1])
    valid_pairs = mul(valid_row, valid_col)

    old_maps = sl("mem", N, 5 * N, vector=True)
    old_seen = sl("mem", 5 * N, 5 * N + K, vector=True)
    last_action = sl("mem", 5 * N + K, 5 * N + K + 8,
                     vector=True)
    old_clock = sl("mem", 5 * N + K + 8, 256, vector=True)

    pair_w = np.concatenate(
        [np.eye(K, dtype=np.float32), np.eye(K, dtype=np.float32)],
        axis=1
    )
    last_loop = dense(last_action, pair_w)
    last_r = vsum(sl(last_action, 0, 4, vector=True), 4)
    last_l = vsum(sl(last_action, 4, 8, vector=True), 4)

    gate_w = np.zeros((K * N, K), np.float32)
    for k in range(K):
        gate_w[k * N:(k + 1) * N, k] = 1
    last_gate = dense(last_loop, gate_w)

    combine_w = np.concatenate(
        [np.eye(N, dtype=np.float32)] * K, axis=1
    )
    selected_old_map = dense(vmul(old_maps, last_gate), combine_w)
    old_map_row = embed(selected_old_map, row_basis)
    selected_known = vsum(vmul(old_seen, last_loop), K)

    # Colour-consistent source/destination matching. L observations are
    # reversed so that memory always stores the R permutation.
    match_r = eq(previous_row, current_col)
    match_l = eq(current_row, previous_col)
    colour_matches = add(mul(match_r, last_r), mul(match_l, last_l))
    permitted = mul(colour_matches, valid_pairs)

    distance = add(emit("abs", [sub(xrow, xcol)]),
                   emit("abs", [sub(yrow, ycol)]))
    retained_edge = mul(eq(old_map_row, col_index), selected_known)

    matching_score = affine(distance, -4096, 1000000)
    matching_score = add(matching_score, scale(retained_edge, 1000000))
    matching_score = sub(matching_score, scale(row_index, 64))
    matching_score = sub(matching_score, col_index)
    matching_score = mul(matching_score, permitted)

    # Greedy one-to-one assignment with a deterministic tie break.
    match_name = uid("matching")
    ss, sm = uid("scores"), uid("matches")
    match_body = []
    active = match_body

    largest = emit("gmax", [ss])
    pick = eq(ss, largest)
    positive = emit("min2", [ss, c1])
    pick = mul(pick, positive)
    used_row = emit("rowmax", [pick])
    used_col = emit("colmax", [pick])
    remaining = mul(complement(used_row), complement(used_col))
    next_scores = mul(ss, remaining)
    next_matches = emit("max2", [sm, pick])

    active = graph
    finish_loop(
        match_name,
        [{"id": ss, "init": matching_score},
         {"id": sm, "zeros": [1, 64, 64]}],
        match_body, {ss: next_scores, sm: next_matches}, N
    )
    assignment = match_name + "." + sm
    new_map_field = emit("rowsum", [mul(assignment, col_index)])
    new_map = emit("gmax", [mul(new_map_field, row_basis)])

    repeated_new_map = emit("vconcat", [new_map] * K)
    stored_maps = vadd(
        vmul(old_maps, vcomp(last_gate)),
        vmul(repeated_new_map, last_gate)
    )
    seen = vclip(vadd(old_seen, last_loop))

    seen_gate = dense(seen, gate_w)
    identity_maps = constant(np.tile(np.arange(N, dtype=np.float32), K))
    usable_maps = vadd(
        vmul(stored_maps, seen_gate),
        vmul(identity_maps, vcomp(seen_gate))
    )

    forward_maps = [
        sl(usable_maps, k * N, (k + 1) * N, vector=True)
        for k in range(K)
    ]
    reverse_maps = []
    for forward in forward_maps:
        field = embed(forward, row_basis)
        permutation = mul(eq(field, col_index), valid_pairs)
        inverse_field = emit("colsum", [mul(permutation, row_index)])
        inverse = emit("gmax", [mul(inverse_field, col_basis)])
        reverse_maps.append(inverse)

    all_maps = emit("vconcat", forward_maps + reverse_maps)

    # Current and desired pair-position sets. Equal-colour goal tiles
    # are interchangeable; a single goal is represented on the diagonal.
    target_row = embed(targets, row_basis)
    target_col = embed(targets, col_basis)

    def type_fields(r, c):
        return (eq(r, constant([11])), eq(c, constant([11])),
                eq(r, constant([12])), eq(c, constant([12])))

    cr11, cc11, cr12, cc12 = type_fields(current_row, current_col)
    tr11, tc11, tr12, tc12 = type_fields(target_row, target_col)

    has11 = emit("gmax", [cr11])
    has12 = emit("gmax", [cr12])
    mixed_types = vmul(has11, has12)
    goal_count = emit("gmax", [
        emit("colsum", [add(cr11, cr12)])
    ])
    multiple_goals = vclip(vadd(goal_count, cm1))
    diagonal = eq(row_index, col_index)
    distinct_factor = complement(mul(diagonal, multiple_goals))

    def pair_positions(r11, c11, r12, c12):
        same = add(mul(r11, c11), mul(r12, c12))
        same = mul(same, distinct_factor)
        mixed = mul(r11, c12)
        return add(mul(same, vcomp(mixed_types)),
                   mul(mixed, mixed_types))

    current_pairs = pair_positions(cr11, cc11, cr12, cc12)
    target_pairs = pair_positions(tr11, tc11, tr12, tc12)
    already_at_pair_target = emit(
        "gmax", [mul(current_pairs, target_pairs)]
    )

    # Rank each colour of button independently and pair equal ranks.
    button_rank_w = np.zeros((8, 3), np.float32)
    button_rank_b = np.zeros(8, np.float32)
    for a in range(8):
        button_rank_w[a, 1 if a < 4 else 2] = 1
        button_rank_b[a] = -(a % 4 + 1)
    button_difference = conv(ranks, button_rank_w, button_rank_b)
    button_basis = eq_diff(button_difference)

    replicate_button_w = np.zeros((8, 2), np.float32)
    replicate_button_w[:4, 0] = 1
    replicate_button_w[4:, 1] = 1
    button_basis = mul(
        button_basis, conv(button_anchors, replicate_button_w)
    )
    button_exists = emit("gmax", [button_basis])

    # Serial asynchronous Bellman updates. Each iteration applies one
    # learned permutation to both axes of the pair-value table.
    planner_name = uid("planner")
    pv, pp, ps = uid("value"), uid("phase"), uid("actionvalues")
    initial_value = scale(target_pairs, 100)
    initial_phase = constant([1, 0, 0, 0, 0, 0, 0, 0])

    select_gate_w = np.zeros((8 * N, 8), np.float32)
    for a in range(8):
        select_gate_w[a * N:(a + 1) * N, a] = 1
    select_sum_w = np.concatenate(
        [np.eye(N, dtype=np.float32)] * 8, axis=1
    )
    rotate_w = np.zeros((8, 8), np.float32)
    for a in range(8):
        rotate_w[(a + 1) % 8, a] = 1

    planner_body = []
    active = planner_body

    selection_gate = dense(pp, select_gate_w)
    selected_map = dense(vmul(all_maps, selection_gate), select_sum_w)
    destination_rows = eq(row_index, selected_map)
    destination_cols = eq(col_index, selected_map)

    first_axis = mul(pv, destination_rows)
    first_axis = emit("colsum", [first_axis])
    first_axis = mul(first_axis, row_basis)
    first_axis = sum_channels(first_axis, N)

    successor_value = mul(first_axis, destination_cols)
    successor_value = emit("rowsum", [successor_value])
    successor_value = mul(successor_value, col_basis)
    successor_value = sum_channels(successor_value, N)

    backed_up = relu(sub(successor_value, c1))
    next_value = emit("max2", [pv, backed_up])
    sampled_value = emit("gmax", [mul(successor_value, current_pairs)])
    next_action_values = vadd(
        vmul(ps, vcomp(pp)), vmul(pp, sampled_value)
    )
    next_phase = dense(pp, rotate_w)

    active = graph
    finish_loop(
        planner_name,
        [{"id": pv, "init": initial_value},
         {"id": pp, "init": initial_phase},
         {"id": ps, "zeros": [8]}],
        planner_body,
        {pv: next_value, pp: next_phase, ps: next_action_values},
        224
    )
    action_values = planner_name + "." + ps

    # A four-phase clock breaks exploration ties without a random source.
    clock_w = np.zeros((4, 4), np.float32)
    clock_w[0, :] = -1
    clock_w[0, 3] += 1
    clock_w[1, 0] = 1
    clock_w[2, 1] = 1
    clock_w[3, 2] = 1
    clock = dense(old_clock, clock_w, [1, 0, 0, 0])
    clock8 = emit("vconcat", [clock, clock])
    seen8 = emit("vconcat", [seen, seen])

    scores = vmul(action_values, vcomp(already_at_pair_target))
    scores = vadd(scores, vscale(vcomp(seen8), 200))
    scores = vadd(scores, vscale(clock8, 0.1))
    # R wins a direction tie; every action has a distinct secondary key.
    scores = vadd(scores, constant(
        [0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.000]
    ))
    scores = vadd(scores, vscale(vcomp(button_exists), -10000))

    # Exact finite-margin winner circuit for the eight action scores.
    comparison_w = np.zeros((64, 8), np.float32)
    for i in range(8):
        for j in range(8):
            comparison_w[8 * i + j, j] += 1000
            comparison_w[8 * i + j, i] -= 1000
    defeats = vclip(dense(scores, comparison_w))
    winner_w = np.zeros((8, 64), np.float32)
    for i in range(8):
        winner_w[i, 8 * i:8 * i + 8] = -1
    winner = emit("vrelu", [
        dense(defeats, winner_w, np.ones(8, np.float32))
    ])

    click_field = mul(button_basis, winner)
    emit("conv", [click_field], name="click",
         w=parameter(np.ones((1, 8, 1, 1), np.float32)))

    logits_b = np.full(8, -1000, np.float32)
    logits_b[6] = 1
    dense(unit, np.zeros((8, 1), np.float32), logits_b, name="logits")

    emit("vconcat", [colours, stored_maps, seen, winner, clock],
         name="mem_next")

    return {
        "graph": graph,
        "memory_size": 256,
        "params": params,
    }

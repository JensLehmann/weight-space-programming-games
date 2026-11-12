import numpy as np


def build():
    N = 40
    G = 4
    BASE = 5 * N + 10
    M = BASE + 4

    params = {}
    graph = []
    out = graph
    serial = [0]
    constants = {}

    def parameter(a):
        name = "p%d" % len(params)
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, inputs, name=None, **fields):
        serial[0] += 1
        name = name or ("n%d" % serial[0])
        node = {"id": name, "op": op, "in": list(inputs)}
        node.update(fields)
        out.append(node)
        return name

    def linear(x, w, b=None, spatial=False, name=None):
        w = np.asarray(w, dtype=np.float32)
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return emit(
            "conv" if spatial else "dense",
            [x], name=name, **fields
        )

    def const(value):
        value = float(value)
        if value not in constants:
            serial[0] += 1
            name = "n%d" % serial[0]
            graph.append({
                "id": name,
                "op": "dense",
                "in": ["aprev"],
                "w": parameter(np.zeros((1, 8))),
                "b": parameter([value]),
            })
            constants[value] = name
        return constants[value]

    def unary(op, x):
        return emit(op, [x])

    def binary(op, a, b):
        return emit(op, [a, b])

    def ss(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def cat(xs):
        return emit("concat", xs)

    def vcat(xs, name=None):
        return emit("vconcat", xs, name=name)

    def affine(x, a=1.0, b=0.0, vector=False):
        if a != 1.0:
            x = binary("vmul" if vector else "mul", x, const(a))
        if b != 0.0:
            x = binary("vadd" if vector else "add", x, const(b))
        return x

    def vsub(a, b):
        return binary("vadd", a, affine(b, -1, vector=True))

    def one_minus(x, vector=False):
        return affine(x, -1, 1, vector)

    def total_channels(x, count):
        return linear(x, np.ones((1, count)), spatial=True)

    def hard_hat(d):
        d = unary("abs", d)
        d = unary("relu", affine(d, -4, 2))
        return binary("min2", d, const(1))

    def clip_vector(x):
        a = unary("vrelu", x)
        b = unary("vrelu", affine(x, 1, -1, True))
        return vsub(a, b)

    def peak(score, gain=256):
        top = unary("gmax", score)
        difference = binary("sub", score, top)
        return unary("relu", affine(difference, gain, 1))

    def vector_hat(t, centers):
        centers = list(centers)
        k = len(centers)
        w = np.ones((3 * k, 1), np.float32)
        b = []
        reduce = np.zeros((k, 3 * k), np.float32)
        for j, center in enumerate(centers):
            b.extend([1 - center, -center, -1 - center])
            reduce[j, 3*j:3*j+3] = [1, -2, 1]
        h = unary("vrelu", linear(t, w, b))
        return linear(h, reduce)

    def vector_windows(t, lo, hi):
        k = len(lo)
        w = np.ones((4 * k, 1), np.float32)
        b = []
        reduce = np.zeros((k, 4 * k), np.float32)
        for j, (a, z) in enumerate(zip(lo, hi)):
            b.extend([1-a, -a, 1-z, -z])
            reduce[j, 4*j:4*j+4] = [1, -1, -1, 1]
        h = unary("vrelu", linear(t, w, b))
        return linear(h, reduce)

    # Persistent permutation models and observation signatures.
    old_maps = vs("mem", 0, G * N)
    old_signature = vs("mem", G * N, (G + 1) * N)
    clock = vs("mem", 5*N, 5*N + 1)
    old_phase = vs("mem", 5*N + 1, 5*N + 9)
    old_ready = vs("mem", 5*N + 9, BASE)

    old_nav_value = vs("mem", BASE, BASE + 1)
    old_nav_direction = vs("mem", BASE + 1, BASE + 2)
    old_meter = vs("mem", BASE + 2, BASE + 3)
    old_small_budget = vs("mem", BASE + 3, BASE + 4)

    first = vector_hat(clock, [0])
    next_clock = affine(clock, 1, 1, True)

    phase_seed = np.zeros((8, 1), np.float32)
    phase_seed[0, 0] = 1
    phase = binary("vadd", old_phase, linear(first, phase_seed))
    next_phase = linear(phase, np.roll(np.eye(8), 1, axis=0))

    py = linear("coords", [[63, 0, 0]], spatial=True)
    px = linear("coords", [[0, 63, 0]], spatial=True)

    tile_w = np.zeros((1, 16), np.float32)
    tile_w[0, [1, 2, 9, 10, 11, 12, 15]] = 1
    tile_ink = linear("x0", tile_w, spatial=True)

    button_w = np.zeros((2, 16), np.float32)
    button_w[0, 14] = 1
    button_w[1, 8] = 1
    button_ink = linear("x0", button_w, spatial=True)

    color_b = ss("x0", 11, 12)
    color_c = ss("x0", 12, 13)

    code_w = np.zeros((1, 16), np.float32)
    for color, code in [
        (1, 1), (2, 2), (9, 3), (10, 4),
        (15, 5), (11, 6), (12, 7)
    ]:
        code_w[0, color] = code
    color_code = linear("x0", code_w, spatial=True)

    # Camera scale.
    w = np.zeros((1, 1, 5, 5), np.float32)
    for dx in [0, 1, 2]:
        w[0, 0, 2, 2+dx] = 1
    scale_test = unary(
        "relu", linear(tile_ink, w, [-2], True)
    )
    scale_two = unary("gmax", scale_test)
    scale_one = one_minus(scale_two, True)

    # Tile top-left anchors, rejecting bracket corners.
    w = np.zeros((2, 1, 7, 7), np.float32)
    for channel, size in enumerate([1, 3]):
        for dy, dx in [
            (0, 0), (0, size), (size, 0), (size, size)
        ]:
            w[channel, 0, 3+dy, 3+dx] += 1
        w[channel, 0, 2, 3] -= 4
        w[channel, 0, 3, 2] -= 4
    anchors = unary(
        "relu", linear(tile_ink, w, [-3, -3], True)
    )
    tile_anchor = binary(
        "add",
        binary("mul", ss(anchors, 0, 1), scale_one),
        binary("mul", ss(anchors, 1, 2), scale_two),
    )

    # One anchor per arrow, including rotated arrows.
    w = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        w[c, c, 3, 3] = 1
        w[c, c, 2, :] = -1
        w[c, c, 3, 2] = -1
    button_anchor = unary(
        "relu", linear(button_ink, w, spatial=True)
    )

    valid_x = binary(
        "min2", unary("relu", affine(px, 2, -1)), const(1)
    )
    valid_y = binary(
        "min2", unary("relu", affine(py, 2, -3)), const(1)
    )
    valid_button = binary("mul", valid_x, valid_y)
    button_anchor = binary("mul", button_anchor, valid_button)
    red_anchor = ss(button_anchor, 0, 1)
    blue_anchor = ss(button_anchor, 1, 2)

    # Row-major indexing using recurrent prefix sums.
    rank_input = cat([tile_anchor, red_anchor])

    def prefix(inp, vertical, tag):
        nonlocal out
        body = []
        out = body
        w = np.zeros((2, 2, 3, 3), np.float32)
        for c in range(2):
            w[c, c, 0 if vertical else 1,
              1 if vertical else 0] = 1
        state = tag + "_state"
        shifted = linear(state, w, spatial=True)
        nxt = binary("add", shifted, inp)
        out = graph
        graph.append({
            "id": tag,
            "op": "loop",
            "T": 63,
            "state": [{"id": state, "init": inp}],
            "body": body,
            "next": {state: nxt},
        })
        return tag + "." + state

    horizontal = prefix(rank_input, False, "HP")
    row_counts = unary("rowsum", rank_input)
    vertical = prefix(row_counts, True, "VP")
    ranks = binary(
        "sub", binary("add", horizontal, vertical), row_counts
    )
    tile_rank = ss(ranks, 0, 1)
    red_rank = ss(ranks, 1, 2)

    def rank_selectors(rank, anchor, count):
        d = linear(
            rank, np.ones((count, 1)),
            -np.arange(1, count+1), True
        )
        h = unary("relu", affine(unary("abs", d), -1, 1))
        return binary("mul", h, anchor)

    tile_select = rank_selectors(tile_rank, tile_anchor, N)
    red_select = rank_selectors(red_rank, red_anchor, G)
    tile_exists = unary("gmax", tile_select)
    red_exists = unary("gmax", red_select)

    tile_x = unary("gmax", binary("mul", tile_select, px))
    tile_y = unary("gmax", binary("mul", tile_select, py))
    red_x = unary("gmax", binary("mul", red_select, px))
    red_y = unary("gmax", binary("mul", red_select, py))

    # Pair red and blue arrows using alignment and distance.
    dx = unary("abs", binary("sub", px, red_x))
    dy = unary("abs", binary("sub", py, red_y))
    distance = binary("add", dx, dy)
    alignment = binary("min2", dx, dy)
    pairing_score = binary(
        "add", affine(distance, -1), affine(alignment, -16)
    )
    pairing_score = binary(
        "add", pairing_score, affine(blue_anchor, 4096, -4096)
    )
    pixel_order = binary("add", affine(py, 64), px)
    pairing_score = binary(
        "add", pairing_score, affine(pixel_order, -1/4096)
    )
    blue_select = binary(
        "mul", peak(pairing_score, 16384), red_exists
    )

    # Target bracket detectors at both camera scales.
    def brackets(large):
        w = np.zeros((2, 16, 7, 7), np.float32)
        offsets = [-3, 3] if large else [-1, 2]
        for channel, color in enumerate([11, 12]):
            for dy in offsets:
                for dx in offsets:
                    w[channel, color, 3+dy, 3+dx] = 1
        result = unary(
            "relu", linear("x0", w, [-3, -3], True)
        )
        if large:
            shift = np.zeros((2, 2, 3, 3), np.float32)
            shift[0, 0, 2, 2] = 1
            shift[1, 1, 2, 2] = 1
            result = linear(result, shift, spatial=True)
        return result

    marks_small = brackets(False)
    marks_large = brackets(True)
    marks = binary(
        "add",
        binary("mul", marks_small, scale_one),
        binary("mul", marks_large, scale_two),
    )
    mark_b = ss(marks, 0, 1)
    mark_c = ss(marks, 1, 2)

    # Temporal colour signatures for permutation acquisition.
    previous_image = total_channels(
        binary("mul", tile_select, old_signature), N
    )
    signature_image = binary(
        "add", color_code, affine(previous_image, 1/8)
    )
    next_signature = unary(
        "gmax", binary("mul", tile_select, signature_image)
    )

    signature_error = unary(
        "abs", binary("sub", signature_image, old_signature)
    )
    matches = unary(
        "relu", affine(signature_error, -8192, 8.192)
    )
    matches = binary("min2", matches, const(1))

    dx = unary("abs", binary("sub", px, tile_x))
    dy = unary("abs", binary("sub", py, tile_y))
    movement_distance = binary("add", dx, dy)
    match_score = binary(
        "add", affine(matches, 256),
        affine(movement_distance, -1)
    )
    match_score = binary(
        "add", match_score, affine(tile_rank, -1/64)
    )
    match_score = binary(
        "add", match_score, affine(tile_anchor, 512, -512)
    )

    destination_mask = peak(match_score, 256)
    inferred_map = unary(
        "gmax", binary("mul", destination_mask, tile_rank)
    )

    site_numbers = linear(
        "aprev", np.zeros((N, 8)), np.arange(1, N+1)
    )
    inferred_map = binary(
        "vadd",
        binary("vmul", inferred_map, tile_exists),
        binary(
            "vmul", site_numbers, one_minus(tile_exists, True)
        ),
    )

    identity_flat = np.tile(
        np.arange(1, N+1), G
    ).astype(np.float32)
    maps = binary(
        "vadd", old_maps, linear(first, identity_flat[:, None])
    )

    save = vector_hat(clock, [5, 10, 15, 20])
    save = binary("vmul", save, red_exists)
    save = binary("vmul", save, one_minus(old_ready, True))
    repeat_group = np.repeat(
        np.eye(G, dtype=np.float32), N, axis=0
    )
    save_flat = linear(save, repeat_group)
    proposed = linear(
        inferred_map, np.tile(np.eye(N), (G, 1))
    )
    maps = binary(
        "vadd", maps,
        binary("vmul", vsub(proposed, maps), save_flat)
    )

    # Select up to two goals, unfilled ones first.
    goal = binary(
        "mul", tile_anchor, binary("add", color_b, color_c)
    )
    aligned = binary(
        "add",
        binary("mul", color_b, mark_b),
        binary("mul", color_c, mark_c),
    )
    priority = binary(
        "add", affine(tile_rank, -1, N+1),
        affine(aligned, -(N+1), N+1)
    )
    priority = binary("mul", priority, goal)

    first_goal = binary("mul", peak(priority, 4), goal)
    remaining_goal = binary("mul", goal, one_minus(first_goal))
    second_priority = binary("mul", priority, remaining_goal)
    second_goal = binary(
        "mul", peak(second_priority, 4), remaining_goal
    )

    r1 = unary("gmax", binary("mul", first_goal, tile_rank))
    r2_raw = unary("gmax", binary("mul", second_goal, tile_rank))
    b1 = unary("gmax", binary("mul", first_goal, color_b))
    b2_raw = unary("gmax", binary("mul", second_goal, color_b))
    has_second = unary("gmax", second_goal)
    missing_second = one_minus(has_second, True)
    r2 = binary(
        "vadd", r2_raw, binary("vmul", r1, missing_second)
    )
    b2 = binary(
        "vadd", b2_raw, binary("vmul", b1, missing_second)
    )

    # Product-state planning grid.
    identity_rows = hard_hat(binary("sub", py, site_numbers))
    identity_cols = hard_hat(binary("sub", px, site_numbers))

    target_b = unary("gmax", binary("mul", tile_select, mark_b))
    target_c = unary("gmax", binary("mul", tile_select, mark_c))

    row_b = total_channels(
        binary("mul", identity_rows, target_b), N
    )
    row_c = total_channels(
        binary("mul", identity_rows, target_c), N
    )
    col_b = total_channels(
        binary("mul", identity_cols, target_b), N
    )
    col_c = total_channels(
        binary("mul", identity_cols, target_c), N
    )

    terminal_rows = binary(
        "add", row_c,
        binary("mul", binary("sub", row_b, row_c), b1)
    )
    terminal_cols = binary(
        "add", col_c,
        binary("mul", binary("sub", col_b, col_c), b2)
    )
    diagonal = hard_hat(binary("sub", py, px))
    distinct = one_minus(binary("mul", diagonal, has_second))
    terminal = binary(
        "mul", binary("mul", terminal_rows, terminal_cols),
        distinct
    )

    current_state = binary(
        "mul",
        hard_hat(binary("sub", py, r1)),
        hard_hat(binary("sub", px, r2)),
    )

    group_start = linear(
        "aprev", np.zeros((G, 8)), [1, 0, 0, 0]
    )
    body = []
    out = body

    group_gate = linear("vi_group", repeat_group)
    selected_maps = binary("vmul", maps, group_gate)
    permutation = linear(
        selected_maps, np.tile(np.eye(N), (1, G))
    )
    permutation_rows = hard_hat(binary("sub", py, permutation))
    permutation_cols = hard_hat(binary("sub", px, permutation))

    def route_axis(value, gather_mask, scatter_mask, reduction):
        x = binary("mul", value, gather_mask)
        x = unary(reduction, x)
        x = binary("mul", x, scatter_mask)
        x = total_channels(x, N)
        return binary("min2", x, const(1))

    forward = route_axis(
        "vi_value", permutation_rows, identity_rows, "colmax"
    )
    forward = route_axis(
        forward, permutation_cols, identity_cols, "rowmax"
    )
    backward = route_axis(
        "vi_value", identity_rows, permutation_rows, "colmax"
    )
    backward = route_axis(
        backward, identity_cols, permutation_cols, "rowmax"
    )

    best_successor = binary("max2", forward, backward)
    next_value = binary(
        "max2", "vi_value", affine(best_successor, 0.97)
    )

    q_forward = unary(
        "gmax", binary("mul", forward, current_state)
    )
    q_backward = unary(
        "gmax", binary("mul", backward, current_state)
    )
    two_q = vcat([q_forward, q_backward])
    spread_q = np.zeros((8, 2), np.float32)
    spread_q[:4, 0] = 1
    spread_q[4:, 1] = 1
    candidate_q = linear(two_q, spread_q)
    q_gate = linear("vi_group", np.tile(np.eye(G), (2, 1)))
    next_q = binary(
        "vadd", "vi_q",
        binary("vmul", vsub(candidate_q, "vi_q"), q_gate)
    )
    next_group = linear(
        "vi_group", np.roll(np.eye(G), 1, axis=0)
    )

    out = graph
    graph.append({
        "id": "PLAN",
        "op": "loop",
        "T": 152,
        "state": [
            {"id": "vi_value", "init": terminal},
            {"id": "vi_q", "zeros": [8]},
            {"id": "vi_group", "init": group_start},
        ],
        "body": body,
        "next": {
            "vi_value": next_value,
            "vi_q": next_q,
            "vi_group": next_group,
        },
    })

    reachable_value = unary(
        "gmax", binary("mul", "PLAN.vi_value", current_state)
    )
    reachable = clip_vector(
        affine(reachable_value, 1000, vector=True)
    )
    ready = clip_vector(binary("vadd", old_ready, reachable))

    probing = vector_windows(
        clock, [0, 5, 10, 15], [5, 10, 15, 20]
    )
    probing = binary("vmul", probing, red_exists)
    probing = binary("vmul", probing, one_minus(ready, True))
    probe_spread = np.zeros((8, 4), np.float32)
    probe_spread[:4, :] = np.eye(4)
    probe_scores = linear(probing, probe_spread)

    scores = binary(
        "vadd", "PLAN.vi_q",
        affine(probe_scores, 4, vector=True)
    )

    # Geometric controller for one goal and one button pair.
    # Value at a tile is 64 minus its local-lattice distance
    # to the matching target bracket.
    nav_target = binary(
        "add",
        binary("mul", mark_b, b1),
        binary("mul", mark_c, one_minus(b1, True)),
    )
    nav_seed = affine(
        binary("mul", nav_target, tile_anchor), 64
    )

    nav_body = []
    out = nav_body

    def spread_three(value, vertical):
        w = np.zeros((3, 1, 7, 7), np.float32)
        for c, offset in enumerate([-3, 0, 3]):
            if vertical:
                w[c, 0, 3+offset, 3] = 1
            else:
                w[c, 0, 3, 3+offset] = 1
        shifted = linear(value, w, spatial=True)
        a = ss(shifted, 0, 1)
        b = ss(shifted, 1, 2)
        c = ss(shifted, 2, 3)
        return binary("max2", binary("max2", a, b), c)

    near_one = spread_three(
        spread_three("nav_field", False), True
    )
    near_two = spread_three(
        spread_three(near_one, False), True
    )
    nav_neighbors = binary(
        "add",
        binary("mul", near_one, scale_one),
        binary("mul", near_two, scale_two),
    )
    nav_candidate = unary(
        "relu", affine(nav_neighbors, 1, -1)
    )
    nav_candidate = binary("mul", nav_candidate, tile_anchor)
    nav_next = binary("max2", "nav_field", nav_candidate)

    out = graph
    graph.append({
        "id": "NAV",
        "op": "loop",
        "T": 40,
        "state": [{"id": "nav_field", "init": nav_seed}],
        "body": nav_body,
        "next": {"nav_field": nav_next},
    })

    nav_value = unary(
        "gmax", binary("mul", "NAV.nav_field", first_goal)
    )
    button_count = linear(red_exists, np.ones((1, G)))
    single_pair = vector_hat(button_count, [1])
    nav_gate = binary(
        "vmul", single_pair, one_minus(has_second, True)
    )
    nav_gate = binary("vmul", nav_gate, clip_vector(nav_value))

    # A decrease in potential means the last direction was wrong.
    away = clip_vector(vsub(old_nav_value, nav_value))
    flip = binary(
        "vmul", away,
        affine(old_nav_direction, -2, 1, True)
    )
    nav_direction = binary("vadd", old_nav_direction, flip)
    previous_reset = vs("aprev", 0, 1)
    nav_initialize = clip_vector(
        binary("vadd", first, previous_reset)
    )
    nav_direction = binary(
        "vadd",
        binary(
            "vmul", nav_direction,
            one_minus(nav_initialize, True)
        ),
        nav_initialize,
    )

    nav_action_w = np.zeros((8, 1), np.float32)
    nav_action_w[0, 0] = 1
    nav_action_w[4, 0] = -1
    nav_action_b = np.zeros(8, np.float32)
    nav_action_b[4] = 1
    nav_action = linear(
        nav_direction, nav_action_w, nav_action_b
    )

    scores = binary(
        "vadd",
        binary("vmul", scores, one_minus(nav_gate, True)),
        binary(
            "vmul", affine(nav_action, 4, vector=True),
            nav_gate
        ),
    )

    # Read the left-edge action meter. A 4-5 pixel advance
    # identifies the short, thirteen-move budget.
    column_zero = unary("relu", affine(px, -1, 1))
    spent_pixels = binary(
        "mul", ss("x0", 5, 6), column_zero
    )
    meter = linear(unary("gmean", spent_pixels), [[4096]])
    meter_delta = vsub(meter, old_meter)
    small_observed = clip_vector(
        affine(meter_delta, 1, -2, True)
    )
    previous_click = vs("aprev", 6, 7)
    small_observed = binary(
        "vmul", small_observed, one_minus(first, True)
    )
    small_observed = binary(
        "vmul", small_observed, previous_click
    )
    small_budget = clip_vector(
        binary("vadd", old_small_budget, small_observed)
    )

    nav_distance = affine(nav_value, -1, 64, True)
    remaining_budget = affine(meter, -13/64, 13, True)
    shortage = clip_vector(
        affine(
            vsub(nav_distance, remaining_budget),
            4, 1, True
        )
    )
    reset_request = binary(
        "vmul", binary("vmul", shortage, small_budget),
        nav_gate
    )

    # Deterministic, weak tie breaking.
    scores = binary(
        "vadd", scores, affine(phase, 0.00001, vector=True)
    )
    fixed_tie = linear(
        "aprev", np.zeros((8, 8)),
        np.arange(8, 0, -1, dtype=np.float32) * 1e-7
    )
    scores = binary("vadd", scores, fixed_tie)
    scores = affine(scores, 1, 2, True)

    button_masks = cat([red_select, blue_select])
    scored_buttons = binary("mul", button_masks, scores)
    click = ss(scored_buttons, 0, 1)
    for j in range(1, 8):
        click = binary(
            "max2", click, ss(scored_buttons, j, j+1)
        )
    linear(click, [[1]], spatial=True, name="click")

    logits_w = np.zeros((8, 1), np.float32)
    logits_w[0, 0] = 2
    logits_b = np.zeros(8, np.float32)
    logits_b[6] = 1
    linear(
        reset_request, logits_w, logits_b, name="logits"
    )

    vcat([
        maps,
        next_signature,
        next_clock,
        next_phase,
        ready,
        nav_value,
        nav_direction,
        meter,
        small_budget,
    ], name="mem_next")

    return {
        "graph": graph,
        "memory_size": M,
        "params": params,
    }

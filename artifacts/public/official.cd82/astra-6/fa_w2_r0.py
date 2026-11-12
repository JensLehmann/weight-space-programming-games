import numpy as np


def build():
    P = {}
    G = []
    serial = [0]
    constants = {}

    def emit(op, ins=None, L=None, name=None, **fields):
        if L is None:
            L = G
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(fields)
        L.append(nd)
        return name

    def parameter(key, value):
        if key not in P:
            P[key] = np.asarray(value, dtype=np.float32)
        return key

    def dense(x, w, b=None, key=None, L=None, name=None):
        if key is None:
            key = "p%d" % len(P)
        fields = {"w": parameter(key + "_w", w)}
        if b is not None:
            fields["b"] = parameter(key + "_b", b)
        return emit("dense", [x], L=L, name=name, **fields)

    def conv(x, w, b=None, key=None, name=None):
        if key is None:
            key = "p%d" % len(P)
        fields = {"w": parameter(key + "_w", w)}
        if b is not None:
            fields["b"] = parameter(key + "_b", b)
        return emit("conv", [x], name=name, **fields)

    def constant(value):
        a = np.asarray(value, np.float32).reshape(-1)
        key = tuple(float(v) for v in a)
        if key not in constants:
            constants[key] = dense(
                "avail", np.zeros((len(a), 8), np.float32), a
            )
        return constants[key]

    def va(a, b, L=None):
        return emit("vadd", [a, b], L=L)

    def vm(a, b, L=None):
        return emit("vmul", [a, b], L=L)

    def scale(a, s, L=None):
        return vm(a, constant(s), L=L)

    def minus_one(a, L=None):
        return va(scale(a, -1, L), constant(1), L)

    def clip01_nonnegative(a, L=None):
        excess = emit("vrelu", [va(a, constant(-1), L)], L=L)
        return va(a, scale(excess, -1, L), L)

    def vs(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def ss(x, start, stop):
        return emit("slice", [x], start=start, stop=stop)

    # Each wedge has an exterior and an inset-tool region.
    atoms = [
        (0, 1), (0, 3),
        (0, 8), (0, 6),
        (1, 9), (3, 9),
        (8, 9), (6, 9),
        (9, 8), (9, 6),
        (9, 1), (9, 3),
        (8, 0), (6, 0),
        (1, 0), (3, 0),
    ]

    orientation_points = [
        (24, 31), (31, 49), (38, 46), (46, 49),
        (53, 31), (46, 14), (38, 17), (31, 14),
    ]
    tool_border_points = [
        (18, 30), (37, 52), (59, 30), (37, 11)
    ]
    tool_click_points = [
        (20, 32), (39, 50), (57, 32), (39, 13)
    ]

    points = (
        [(r + 3, c + 3) for r, c in atoms]
        + [(r + 34, c + 27) for r, c in atoms]
        + orientation_points
        + tool_border_points
        + tool_click_points
    )

    # Exact point masks. Saturation removes coordinate roundoff.
    npoints = len(points)
    w = np.zeros((2 * npoints, 3, 1, 1), np.float32)
    b = np.zeros(2 * npoints, np.float32)
    for i, (r, c) in enumerate(points):
        w[2 * i, 0, 0, 0] = 63
        w[2 * i + 1, 1, 0, 0] = 63
        b[2 * i] = -r
        b[2 * i + 1] = -c
    distances = emit("abs", [conv("coords", w, b)])
    w = np.zeros((npoints, 2 * npoints, 1, 1), np.float32)
    for i in range(npoints):
        w[i, 2 * i:2 * i + 2, 0, 0] = -4
    point_raw = emit(
        "relu", [conv(distances, w, np.full(npoints, 2, np.float32))]
    )
    point_masks = emit("min2", [point_raw, constant(1)])

    features = []
    for i in range(32):
        mask = ss(point_masks, i, i + 1)
        pixels = emit("mul", ["x0", mask])
        features.append(emit("gmax", [pixels]))
    target = emit("vconcat", features[:16])
    board = emit("vconcat", features[16:])

    red = ss("x0", 2, 3)
    red_samples = emit(
        "gmax",
        [emit("mul", [red, ss(point_masks, 32, 44)])]
    )
    orientation = vs(red_samples, 0, 8)
    tool_present = dense(
        vs(red_samples, 8, 12), np.ones((1, 4), np.float32)
    )

    # Palette centres and the underline identifying the selected colour.
    border = ss("x0", 4, 5)
    w = np.zeros((1, 1, 5, 5), np.float32)
    for r, c in [(0, 2), (4, 2), (2, 0), (2, 4)]:
        w[0, 0, r, c] = 1
    centres_raw = emit("relu", [conv(border, w, [-3])])

    w = np.zeros((1, 3, 1, 1), np.float32)
    w[0, 0, 0, 0] = 63
    row_distance = emit("abs", [conv("coords", w, [-4])])
    row_raw = emit(
        "relu",
        [conv(row_distance, np.array([[[[-4]]]], np.float32), [2])]
    )
    row_four = emit("min2", [row_raw, constant(1)])
    centres = emit("mul", [centres_raw, row_four])

    zero_plane = ss("x0", 0, 1)
    w = np.zeros((1, 1, 7, 7), np.float32)
    w[0, 0, 6, 3] = 1
    underline = conv(zero_plane, w)
    selected_centres = emit("mul", [centres, underline])
    selected_colour = emit(
        "gmax", [emit("mul", ["x0", selected_centres])]
    )

    # A[operation, atom] is the analytical paint incidence matrix.
    A = np.zeros((12, 16), np.float32)
    for j, (r, c) in enumerate(atoms):
        A[:, j] = [
            r < 5,
            c >= r,
            c >= 5,
            c >= 9 - r,
            r >= 5,
            c <= r,
            c < 5,
            c <= 9 - r,
            r < 3 and 3 <= c < 7,
            c >= 7 and 3 <= r < 7,
            r >= 7 and 3 <= c < 7,
            c < 3 and 3 <= r < 7,
        ]

    operation_positions = list(range(8)) + [0, 2, 4, 6]
    position_matrix = np.zeros((8, 12), np.float32)
    for i, p in enumerate(operation_positions):
        position_matrix[p, i] = 1

    cap_w = np.zeros((12, 1), np.float32)
    cap_w[8:, 0] = 1
    cap_b = np.zeros(12, np.float32)
    cap_b[:8] = 1
    capability = dense(tool_present, cap_w, cap_b)

    # Initial mismatches, one per atom.
    overlap = vm(target, board)
    match_w = np.zeros((16, 256), np.float32)
    for j in range(16):
        match_w[j, 16 * j:16 * j + 16] = -1
    mismatches = dense(overlap, match_w, np.ones(16, np.float32))
    inverse_target = minus_one(target)

    # ReLU argmax circuit for twelve integer-valued scores.
    pair_w = np.zeros((144, 12), np.float32)
    pair_b = np.zeros(144, np.float32)
    winner_w = np.zeros((12, 144), np.float32)
    for i in range(12):
        for j in range(12):
            q = 12 * i + j
            pair_w[q, j] += 1
            pair_w[q, i] -= 1
            pair_b[q] = float(j < i)
            winner_w[i, q] = -1

    def winner12(x, L=None):
        d = dense(x, pair_w, pair_b, key="competition", L=L)
        d = emit("vrelu", [d], L=L)
        z = dense(
            d, winner_w, np.ones(12, np.float32),
            key="winner", L=L
        )
        return emit("vrelu", [z], L=L)

    repeat_atoms = np.zeros((256, 16), np.float32)
    for j in range(16):
        repeat_atoms[16 * j:16 * j + 16, j] = 1

    candidate_bad = np.zeros((192, 256), np.float32)
    repeat_operations = np.zeros((192, 12), np.float32)
    candidate_sum = np.zeros((12, 192), np.float32)
    candidate_colour = np.zeros((12, 192), np.float32)
    for m in range(12):
        for colour in range(16):
            q = 16 * m + colour
            repeat_operations[q, m] = 1
            candidate_sum[m, q] = 1
            candidate_colour[m, q] = colour
            for j in range(16):
                candidate_bad[q, 16 * j + colour] = -A[m, j]

    route_w = np.zeros((12, 8), np.float32)
    for m, p in enumerate(operation_positions):
        for q in range(8):
            distance = min((p - q) % 8, (q - p) % 8)
            route_w[m, q] = -16 * distance
    route_b = np.array([112 - i for i in range(12)], np.float32)

    # Reverse-layer planning: remove a monochromatic exposed mask,
    # assigning its reverse execution rank and colour.
    B = []
    coverage = dense("p_remaining", A, key="coverage", L=B)
    remaining_bad = vm("p_remaining", mismatches, B)
    bad_coverage = dense(remaining_bad, A, key="coverage", L=B)
    bad_total = dense(
        remaining_bad, np.ones((1, 16), np.float32),
        key="sum_atoms", L=B
    )
    active = clip01_nonnegative(bad_total, B)
    empty = emit("vrelu", [minus_one(coverage, B)], L=B)

    repeated_r = dense(
        "p_remaining", repeat_atoms, key="repeat_atoms", L=B
    )
    masked_bad = vm(repeated_r, inverse_target, B)
    candidate_score = dense(
        masked_bad, candidate_bad, np.ones(192, np.float32),
        key="candidate_bad", L=B
    )
    empty_penalty = dense(
        empty, -repeat_operations, key="empty_penalty", L=B
    )
    valid = emit(
        "vrelu", [va(candidate_score, empty_penalty, B)], L=B
    )
    good = dense(valid, candidate_sum, key="candidate_sum", L=B)
    good = vm(vm(good, capability, B), active, B)

    area_score = va(scale(coverage, 256, B),
                    scale(bad_coverage, 256, B), B)
    route_score = dense(
        "p_last", route_w, route_b, key="route_score", L=B
    )
    scores = vm(good, va(area_score, route_score, B), B)
    chosen = vm(winner12(scores, B), good, B)

    removed = dense(chosen, A.T, key="removed_atoms", L=B)
    remaining_next = vm("p_remaining", minus_one(removed, B), B)
    rank_next = va("p_rank", vm(chosen, "p_time", B), B)

    candidate_chosen = vm(
        valid,
        dense(chosen, repeat_operations,
              key="repeat_operations", L=B),
        B
    )
    colour_next = va(
        "p_colour",
        dense(candidate_chosen, candidate_colour,
              key="candidate_colour", L=B),
        B
    )
    last_next = dense(
        chosen, position_matrix, key="operation_position", L=B
    )
    time_next = va("p_time", constant(1), B)

    emit(
        "loop", name="plan", T=12,
        state=[
            {"id": "p_remaining", "init": constant(np.ones(16))},
            {"id": "p_rank", "zeros": [12]},
            {"id": "p_colour", "zeros": [12]},
            {"id": "p_time", "init": constant(1)},
            {"id": "p_last", "init": orientation},
        ],
        body=B,
        next={
            "p_remaining": remaining_next,
            "p_rank": rank_next,
            "p_colour": colour_next,
            "p_time": time_next,
            "p_last": last_next,
        }
    )

    initialized = vs("mem", 0, 1)
    fresh = minus_one(initialized)
    old_ranks = vs("mem", 1, 13)
    old_colours = vs("mem", 13, 25)
    ranks = va(vm("plan.p_rank", fresh), vm(old_ranks, initialized))
    colours = va(
        vm("plan.p_colour", fresh), vm(old_colours, initialized)
    )

    # A queued stroke can be skipped if it currently changes nothing.
    numeric_w = np.zeros((16, 256), np.float32)
    for j in range(16):
        numeric_w[j, 16 * j:16 * j + 16] = np.arange(16)
    board_numeric = dense(board, numeric_w)
    comparison_input = emit("vconcat", [board_numeric, colours])

    comparison_w = np.zeros((192, 28), np.float32)
    uniform_w = np.zeros((12, 192), np.float32)
    for m in range(12):
        for j in range(16):
            q = 16 * m + j
            comparison_w[q, j] = 1
            comparison_w[q, 16 + m] = -1
            uniform_w[m, q] = -A[m, j]
    difference = dense(comparison_input, comparison_w)
    positive = emit("vrelu", [difference])
    negative = emit("vrelu", [scale(difference, -1)])
    absolute_difference = va(positive, negative)
    already_uniform = emit(
        "vrelu",
        [dense(absolute_difference, uniform_w, np.ones(12))]
    )

    B = []
    first = winner12("s_rank", B)
    skip = vm(first, already_uniform, B)
    skipped_ranks = vm("s_rank", minus_one(skip, B), B)
    emit(
        "loop", name="skip", T=12,
        state=[{"id": "s_rank", "init": ranks}],
        body=B, next={"s_rank": skipped_ranks}
    )
    ranks = "skip.s_rank"

    task = vm(winner12(ranks), clip01_nonnegative(ranks))
    task_exists = dense(task, np.ones((1, 12), np.float32))
    desired_position = dense(
        task, position_matrix, key="operation_position"
    )
    desired_numeric = dense(
        vm(task, colours), np.ones((1, 12), np.float32)
    )

    # Convert the planned scalar colour into an exact one-hot vector.
    hue_w = np.ones((32, 1), np.float32)
    hue_w[16:, 0] = -1
    hue_b = np.concatenate(
        [-np.arange(16), np.arange(16)]
    ).astype(np.float32)
    hue_distance = emit(
        "vrelu", [dense(desired_numeric, hue_w, hue_b)]
    )
    hue_tent_w = np.zeros((16, 32), np.float32)
    for c in range(16):
        hue_tent_w[c, c] = -1
        hue_tent_w[c, 16 + c] = -1
    desired_colour = emit(
        "vrelu", [dense(hue_distance, hue_tent_w, np.ones(16))]
    )

    same_colour = dense(
        vm(desired_colour, selected_colour),
        np.ones((1, 16), np.float32)
    )
    same_position = dense(
        vm(desired_position, orientation),
        np.ones((1, 8), np.float32)
    )
    paint_ready = vm(same_colour, same_position)
    change_colour = vm(minus_one(same_colour), task_exists)

    big_w = np.zeros((1, 12), np.float32)
    big_w[0, :8] = 1
    small_w = np.zeros((1, 12), np.float32)
    small_w[0, 8:] = 1
    big = dense(task, big_w)
    small = dense(task, small_w)
    fire_big = vm(paint_ready, big)
    fire_small = vm(paint_ready, small)

    # Shortest perimeter movement, with a deterministic tie break.
    pair_input = emit("vconcat", [orientation, desired_position])
    movement_pairs_w = np.zeros((64, 16), np.float32)
    movement_out_w = np.zeros((8, 64), np.float32)
    clockwise_action = [4, 2, 2, 3, 3, 1, 1, 4]
    counterclockwise_action = [3, 3, 1, 1, 4, 4, 2, 2]
    for current in range(8):
        for destination in range(8):
            q = 8 * current + destination
            movement_pairs_w[q, current] = 1
            movement_pairs_w[q, 8 + destination] = 1
            if current != destination:
                cw = (destination - current) % 8
                ccw = (current - destination) % 8
                action = (
                    clockwise_action[current]
                    if cw <= ccw
                    else counterclockwise_action[current]
                )
                movement_out_w[action, q] = 10
    movement_pairs = emit(
        "vrelu",
        [dense(pair_input, movement_pairs_w, -np.ones(64))]
    )
    movement_logits = dense(movement_pairs, movement_out_w)

    controls = emit(
        "vconcat", [fire_big, fire_small, change_colour]
    )
    control_w = np.zeros((8, 3), np.float32)
    control_w[5, 0] = 15
    control_w[6, 1] = 15
    control_w[6, 2] = 20
    control_b = np.zeros(8, np.float32)
    control_b[[0, 7]] = -100
    control_logits = dense(controls, control_w, control_b)
    normal_logits = va(movement_logits, control_logits)

    # A diagonal initial observation hides whether the inset tool exists.
    # Move once to a cardinal position before committing the plan.
    odd_w = np.zeros((1, 8), np.float32)
    odd_w[0, [1, 3, 5, 7]] = 1
    odd = dense(orientation, odd_w)
    normalize = vm(fresh, odd)
    commit = minus_one(normalize)

    normalize_w = np.zeros((8, 8), np.float32)
    normalize_w[3, [1, 3]] = 100
    normalize_w[4, [5, 7]] = 100
    normalization_logits = vm(
        dense(orientation, normalize_w), fresh
    )
    emit(
        "vadd", [normal_logits, normalization_logits], name="logits"
    )

    # Palette click map.
    colour_pixels = emit("mul", ["x0", desired_colour])
    colour_match = conv(
        colour_pixels, np.ones((1, 16, 1, 1), np.float32)
    )
    palette_click = emit("mul", [colour_match, centres])
    palette_click = emit("mul", [palette_click, change_colour])

    # Inset-tool click map.
    small_task = vs(task, 8, 12)
    inset_points = ss(point_masks, 44, 48)
    inset_selected = emit("mul", [inset_points, small_task])
    inset_click = conv(
        inset_selected, np.ones((1, 4, 1, 1), np.float32)
    )
    inset_click = emit("mul", [inset_click, same_colour])
    emit("add", [palette_click, inset_click], name="click")

    # Consume a task only when this step actually performs its paint.
    actual_paint = vm(paint_ready, commit)
    consumed = vm(task, actual_paint)
    next_ranks = vm(ranks, minus_one(consumed))
    emit(
        "vconcat", [commit, next_ranks, colours], name="mem_next"
    )

    return {"graph": G, "memory_size": 25, "params": P}

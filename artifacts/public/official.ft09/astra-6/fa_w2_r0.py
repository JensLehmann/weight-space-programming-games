import numpy as np


def build():
    G, P = [], {}
    serial = [0]
    pserial = [0]

    def parameter(a):
        name = "p%d" % pserial[0]
        pserial[0] += 1
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, inputs, target=None, name=None, **fields):
        if name is None:
            name = "n%d" % serial[0]
            serial[0] += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        (G if target is None else target).append(nd)
        return name

    def unary(op, x, target=None):
        return emit(op, [x], target=target)

    def binary(op, x, y, target=None, name=None):
        return emit(op, [x, y], target=target, name=name)

    def conv(x, w, b=None, target=None, name=None):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :, None, None]
        elif w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(np.asarray(b).reshape(-1))
        return emit("conv", [x], target=target, name=name, **fields)

    def dense(x, w, b=None, target=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(np.asarray(b).reshape(-1))
        return emit("dense", [x], target=target, name=name, **fields)

    def sl(x, a, b):
        return emit("slice", [x], start=a, stop=b)

    def vs(x, a, b):
        return emit("vslice", [x], start=a, stop=b)

    def const(values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return dense("avail", np.zeros((len(values), 8)), values)

    def spatial_affine(x, a, b=0.0, target=None):
        return conv(x, [[a]], [b], target=target)

    def clipped(x, target=None):
        return binary(
            "min2", unary("relu", x, target), one, target=target
        )

    def sharp_max_mask(field, target=None):
        maximum = emit("gmax", [field], target=target)
        difference = binary("sub", field, maximum, target=target)
        return clipped(
            spatial_affine(difference, 2.0, 1.0, target),
            target=target,
        )

    directions = [(dy, dx) for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1)]
    colours = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nc = len(colours)

    # Spatial constants and integer-valued display coordinates.
    one = sl("coords", 2, 3)
    half = const([0.5])
    four = const([4.0])
    unit_area = const([1.0 / 4096.0])

    xy = conv("coords", [[63, 0, 0], [0, 63, 0]])
    row = sl(xy, 0, 1)
    col = sl(xy, 1, 2)
    priority = conv("coords", [[-4032, -63, 0]], [4096])

    # Recognize the upper-left pixel of each rendered central 2x2 cell.
    # A missing border beyond the image is treated like background.
    w = np.zeros((1, 16, 7, 7), np.float32)
    for c in colours:
        w[0, c, 3, 3] = 1
    for dy, dx in [(-2, 0), (0, -2)]:
        for c in range(16):
            if c not in (4, 5):
                w[0, c, 3 + dy, 3 + dx] += 1
    for dy, dx in [(-3, 0), (0, -3)]:
        for c in range(16):
            if c != 4:
                w[0, c, 3 + dy, 3 + dx] -= 1
    centres = unary("relu", conv("x0", w, [-2]))

    # Clues have 0/2/3 relation markers around a coloured centre.
    w = np.zeros((1, 16, 5, 5), np.float32)
    for dy, dx in directions:
        if dy or dx:
            for c in (0, 2, 3):
                w[0, c, 2 + 2 * dy, 2 + 2 * dx] = 1
    clue_markers = clipped(conv("x0", w))
    clues = binary("mul", centres, clue_markers)
    playable = binary("sub", centres, clues)

    # Nine stencil channels; the central channel is always one.
    w = np.zeros((9, 16, 5, 5), np.float32)
    b = np.zeros(9, np.float32)
    b[4] = 1
    for d, (dy, dx) in enumerate(directions):
        if dy or dx:
            w[d, 6, 2 + 2 * dy, 2 + 2 * dx] = 1
    marked_stencil = conv("x0", w, b)
    w = np.ones((1, 9), np.float32)
    w[0, 4] = 0
    special_present = binary(
        "min2", conv(marked_stencil, w), one
    )
    special = binary("mul", playable, special_present)
    ordinary = binary("sub", playable, special)

    w = np.zeros((nc, 16), np.float32)
    for i, c in enumerate(colours):
        w[i, c] = 1
    colour_planes = conv("x0", w)
    tile_colours = binary("mul", colour_planes, centres)
    clue_colours = binary("mul", colour_planes, clues)

    numeric_colour = conv(
        "x0", np.arange(16, dtype=np.float32)[None, :]
    )
    current_colour = binary("mul", numeric_colour, playable)
    zero_plane = sl("x0", 0, 1)

    # Route signed constraints from clues to neighbouring tile centres.
    # Channels 0..9: +e_colour for equality, -e_colour for inequality.
    # Channel 10: number of equality constraints.
    routed = []
    channels = nc + 1
    for dy, dx in directions:
        if not (dy or dx):
            continue

        w = np.zeros((1, 1, 5, 5), np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        equality = conv(zero_plane, w)
        sign = spatial_affine(equality, 2, -1)
        signed_colours = binary("mul", clue_colours, sign)
        equality_count = binary("mul", clues, equality)
        packed = emit("concat", [signed_colours, equality_count])

        w = np.zeros((channels, channels, 7, 7), np.float32)
        for c in range(channels):
            w[c, c, 3 - 3 * dy, 3 - 3 * dx] = 1
        shift3 = parameter(w)
        z = emit("conv", [packed], w=shift3)
        z = emit("conv", [z], w=shift3)

        w = np.zeros((channels, channels, 5, 5), np.float32)
        for c in range(channels):
            w[c, c, 2 - 2 * dy, 2 - 2 * dx] = 1
        routed.append(conv(z, w))

    requirements = routed[0]
    for z in routed[1:]:
        requirements = binary("add", requirements, z)

    signed_requirements = sl(requirements, 0, nc)
    equality_count = sl(requirements, nc, nc + 1)
    matches = binary("mul", signed_requirements, colour_planes)
    matches = conv(matches, np.ones((1, nc)))
    errors = clipped(binary("sub", equality_count, matches))
    errors = binary("mul", errors, playable)

    w = -np.ones((1, channels), np.float32)
    w[0, nc] = 2
    constrained = clipped(conv(requirements, w))
    constrained = binary("mul", constrained, playable)

    # Detect larger palettes; retain this information across observations.
    palette = emit("gmax", [tile_colours])
    ptest = dense(palette, -np.ones((2, nc)), [3, 2])
    ptest = unary("vrelu", ptest)
    binary_now = dense(ptest, [[1, -1]])
    old_multi = vs("mem", 85, 86)
    not_old_multi = dense(old_multi, [[-1]], [1])
    use_binary = binary("vmul", binary_now, not_old_multi)
    next_multi = dense(use_binary, [[-1]], [1])

    # A spatial bank identifying the 64 possible 8x8 display bins.
    w = np.zeros((32, 3), np.float32)
    b = np.zeros(32, np.float32)
    for axis in range(2):
        for i in range(8):
            j = axis * 8 + i
            w[j, axis] = 252
            b[j] = 2 - 32 * i
            w[j + 16, axis] = -252
            b[j + 16] = 32 * i + 30
    bands = conv("coords", w, b)
    bands = binary("min2", sl(bands, 0, 16), sl(bands, 16, 32))
    bands = clipped(bands)

    w = np.zeros((64, 16), np.float32)
    for r in range(8):
        for c in range(8):
            w[8 * r + c, r] = 1
            w[8 * r + c, 8 + c] = 1
    bins = unary("relu", conv(bands, w, -np.ones(64)))
    action_cells = binary("mul", bins, playable)

    # Snapshot colours for identifying the ordinary click stencil.
    snapshot = emit("gmax", [binary("mul", bins, current_colour)])
    previous_colours = vs("mem", 0, 64)
    previous_field = binary("mul", bins, previous_colours)
    previous_field = conv(previous_field, np.ones((1, 64)))
    changed = unary("abs", binary("sub", numeric_colour, previous_field))
    changed = binary("min2", changed, one)
    changed = binary("mul", changed, playable)

    previous_row = vs("mem", 64, 65)
    previous_col = vs("mem", 65, 66)
    previous_ordinary = vs("mem", 66, 67)
    old_known = vs("mem", 67, 76)
    old_effects = vs("mem", 76, 85)
    previous_click = vs("aprev", 6, 7)
    can_learn = binary("vmul", previous_ordinary, previous_click)

    dr = binary("sub", row, previous_row)
    dc = binary("sub", col, previous_col)
    dr = conv(dr, np.ones((9, 1)),
              [-8 * dy for dy, dx in directions])
    dc = conv(dc, np.ones((9, 1)),
              [-8 * dx for dy, dx in directions])
    distance = binary("add", unary("abs", dr), unary("abs", dc))
    near = binary("sub", half, distance)
    near = clipped(binary("mul", near, four))
    observed = binary("mul", near, playable)
    observed = binary("mul", observed, can_learn)

    learn_mask = emit("gmax", [observed])
    measurements = emit("gmax", [binary("mul", observed, changed)])
    retain = dense(learn_mask, -np.eye(9), np.ones(9))
    known = binary(
        "vadd", learn_mask, binary("vmul", retain, old_known)
    )
    effects = binary(
        "vadd", measurements, binary("vmul", retain, old_effects)
    )

    w = np.zeros((9, 9), np.float32)
    w[4, 4] = -1
    b = np.zeros(9, np.float32)
    b[4] = 1
    default_effect = dense(known, w, b)
    ordinary_stencil = binary("vadd", effects, default_effect)

    stencil_field = binary(
        "add",
        binary("mul", ordinary, ordinary_stencil),
        binary("mul", special, marked_stencil),
    )

    # Construct the click-effect matrix.
    # Spatial positions are equation rows; channels are action columns.
    action_rows = emit("gmax", [binary("mul", action_cells, row)])
    action_cols = emit("gmax", [binary("mul", action_cells, col)])
    relative_rows = binary("sub", row, action_rows)
    relative_cols = binary("sub", col, action_cols)

    e9 = np.zeros(9, np.float32)
    e9[0] = 1
    initial_direction = const(e9)

    B = []
    offsets = dense(
        "if_dir",
        np.asarray([[8 * dy for dy, dx in directions],
                    [8 * dx for dy, dx in directions]]),
        target=B,
    )
    off_r = emit("vslice", [offsets], target=B, start=0, stop=1)
    off_c = emit("vslice", [offsets], target=B, start=1, stop=2)

    rr = binary("sub", relative_rows, off_r, target=B)
    cc = binary("sub", relative_cols, off_c, target=B)
    dist = binary(
        "add", unary("abs", rr, B), unary("abs", cc, B), target=B
    )
    near = binary("sub", half, dist, target=B)
    near = clipped(binary("mul", near, four, target=B), target=B)

    selected_stencil = binary(
        "mul", stencil_field, "if_dir", target=B
    )
    selected_stencil = conv(
        selected_stencil, np.ones((1, 9)), target=B
    )
    action_effect = binary(
        "mul", action_cells, selected_stencil, target=B
    )
    action_effect = emit("gmax", [action_effect], target=B)
    contribution = binary("mul", near, action_effect, target=B)
    accumulated = binary("add", "if_A", contribution, target=B)

    shift9 = np.zeros((9, 9), np.float32)
    for i in range(8):
        shift9[i + 1, i] = 1
    next_direction = dense("if_dir", shift9, target=B)

    G.append({
        "id": "influence",
        "op": "loop",
        "T": 9,
        "state": [
            {"id": "if_A", "zeros": [64, 64, 64]},
            {"id": "if_dir", "init": initial_direction},
        ],
        "body": B,
        "next": {"if_A": accumulated, "if_dir": next_direction},
    })
    matrix = binary("mul", "influence.if_A", constrained)

    # Gauss-Jordan elimination over GF(2).
    # XOR of Boolean tensors is implemented by abs(a-b).
    e64 = np.zeros(64, np.float32)
    e64[0] = 1
    initial_column = const(e64)

    B = []
    coefficient = binary("mul", "gj_A", "gj_c", target=B)
    coefficient = conv(coefficient, np.ones((1, 64)), target=B)
    unused = binary("sub", one, "gj_R", target=B)
    candidates = binary("mul", coefficient, unused, target=B)
    pivot_scores = binary("mul", candidates, priority, target=B)
    pivot = sharp_max_mask(pivot_scores, target=B)
    pivot = binary("mul", pivot, candidates, target=B)

    pivot_row = binary("mul", "gj_A", pivot, target=B)
    pivot_row = emit("gmax", [pivot_row], target=B)
    pivot_rhs = binary("mul", "gj_b", pivot, target=B)
    pivot_rhs = emit("gmax", [pivot_rhs], target=B)

    eliminate_here = binary("sub", coefficient, pivot, target=B)
    matrix_delta = binary(
        "mul", eliminate_here, pivot_row, target=B
    )
    next_matrix = unary(
        "abs", binary("sub", "gj_A", matrix_delta, target=B), B
    )
    rhs_delta = binary("mul", eliminate_here, pivot_rhs, target=B)
    next_rhs = unary(
        "abs", binary("sub", "gj_b", rhs_delta, target=B), B
    )
    next_used = binary("add", "gj_R", pivot, target=B)
    pivot_assignment = binary("mul", pivot, "gj_c", target=B)
    next_assignment = binary(
        "add", "gj_P", pivot_assignment, target=B
    )

    shift64 = np.zeros((64, 64), np.float32)
    for i in range(63):
        shift64[i + 1, i] = 1
    next_column = dense("gj_c", shift64, target=B)

    G.append({
        "id": "eliminate",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": "gj_A", "init": matrix},
            {"id": "gj_b", "init": errors},
            {"id": "gj_R", "zeros": [1, 64, 64]},
            {"id": "gj_P", "zeros": [64, 64, 64]},
            {"id": "gj_c", "init": initial_column},
        ],
        "body": B,
        "next": {
            "gj_A": next_matrix,
            "gj_b": next_rhs,
            "gj_R": next_used,
            "gj_P": next_assignment,
            "gj_c": next_column,
        },
    })

    A = "eliminate.gj_A"
    rhs = "eliminate.gj_b"
    used = "eliminate.gj_R"
    assignment = "eliminate.gj_P"

    solution = emit("gmax", [binary("mul", assignment, rhs)])
    solution_map = binary("mul", action_cells, solution)
    solution_map = conv(solution_map, np.ones((1, 64)))

    has_solution = emit("gmax", [solution_map])
    no_solution = dense(has_solution, [[-1]], [1])
    binary_rank = binary(
        "add", solution_map, binary("mul", errors, no_solution)
    )

    # Improve the canonical solution using a free-column/nullspace move.
    # A selected improvement strictly reduces the remaining basis plan.
    pivot_columns = emit("gmax", [assignment])
    free_columns = dense(pivot_columns, -np.eye(64), np.ones(64))

    alternative_rhs = unary("abs", binary("sub", A, rhs))
    alternative_rhs = binary("mul", alternative_rhs, used)
    alternative_cost = emit("gmean", [alternative_rhs])
    alternative_cost = binary("vadd", alternative_cost, unit_area)
    basis_cost = emit("gmean", [binary("mul", rhs, used)])

    cost_at_actions = binary("mul", action_cells, alternative_cost)
    improvement = binary("sub", basis_cost, cost_at_actions)
    improvement = unary("relu", improvement)
    improvement = binary("mul", improvement, free_columns)
    improvement = binary("mul", improvement, action_cells)
    improvement = conv(
        improvement, np.full((1, 64), 16.0 * 4096.0)
    )
    binary_rank = binary("add", binary_rank, improvement)

    rank = binary(
        "add",
        binary("mul", binary_rank, use_binary),
        binary("mul", errors, next_multi),
    )

    # Separate integer rank selection from positional tie-breaking.
    best_rank = sharp_max_mask(rank)
    best_rank = binary("mul", best_rank, playable)
    positional_score = binary("mul", best_rank, priority)
    special_score = binary("mul", best_rank, special)
    special_score = spatial_affine(special_score, 5000)
    binary(
        "add", positional_score, special_score, name="click"
    )

    chosen = sharp_max_mask("click")
    chosen = binary("mul", chosen, best_rank)
    chosen_row = emit("gmax", [binary("mul", chosen, row)])
    chosen_col = emit("gmax", [binary("mul", chosen, col)])
    chosen_ordinary = emit(
        "gmax", [binary("mul", chosen, ordinary)]
    )

    emit(
        "vconcat",
        [snapshot, chosen_row, chosen_col, chosen_ordinary,
         known, effects, next_multi],
        name="mem_next",
    )

    logits_bias = np.full(8, -10, np.float32)
    logits_bias[0] = -5
    logits_bias[6] = 10
    dense(
        "avail", np.zeros((8, 8)), logits_bias, name="logits"
    )

    return {"graph": G, "memory_size": 86, "params": P}

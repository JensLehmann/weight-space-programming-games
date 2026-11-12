import numpy as np


def build():
    graph = []
    params = {}
    nodes = graph
    serial = 0

    def fresh():
        nonlocal serial
        serial += 1
        return "n_%d" % serial

    def emit(op, ins, name=None, **fields):
        name = fresh() if name is None else name
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        nodes.append(nd)
        return name

    def conv(src, weight, bias=None, name=None):
        name = fresh() if name is None else name
        if isinstance(weight, str):
            wn = weight
        else:
            w = np.asarray(weight, dtype=np.float32)
            if w.ndim == 2:
                w = w[:, :, None, None]
            wn = name + "_w"
            params[wn] = w
        fields = {"w": wn}
        if bias is not None:
            bn = name + "_b"
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return emit("conv", [src], name=name, **fields)

    def dense(src, weight, bias=None, name=None):
        name = fresh() if name is None else name
        wn = name + "_w"
        params[wn] = np.asarray(weight, dtype=np.float32)
        fields = {"w": wn}
        if bias is not None:
            bn = name + "_b"
            params[bn] = np.asarray(bias, dtype=np.float32)
            fields["b"] = bn
        return emit("dense", [src], name=name, **fields)

    def sl(src, start, stop):
        return emit("slice", [src], start=start, stop=stop)

    def vs(src, start, stop):
        return emit("vslice", [src], start=start, stop=stop)

    def unary(op, src):
        return emit(op, [src])

    def binary(op, a, b):
        return emit(op, [a, b])

    def cat(xs):
        return emit("concat", xs)

    def spatial_sum(xs):
        return conv(cat(xs), np.ones((1, len(xs)), np.float32))

    def vneg(src, size):
        return dense(src, -np.eye(size, dtype=np.float32))

    def vone_minus(src, size):
        return dense(
            src, -np.eye(size, dtype=np.float32),
            np.ones(size, np.float32)
        )

    def vclip01(src, size):
        r = unary("vrelu", src)
        excess = unary(
            "vrelu",
            dense(r, np.eye(size, dtype=np.float32),
                  -np.ones(size, np.float32))
        )
        return binary("vadd", r, vneg(excess, size))

    def vabs(src, size):
        return binary(
            "vadd", unary("vrelu", src),
            unary("vrelu", vneg(src, size))
        )

    def shift8(src, channels, dy, dx):
        # Output at (y,x) reads input at (y+8*dy,x+8*dx).
        w = np.zeros((channels, channels, 7, 7), np.float32)
        for c in range(channels):
            w[c, c, 3 + 3 * dy, 3 + 3 * dx] = 1
        a = conv(src, w)
        b = conv(a, a + "_w")  # Reuse the three-pixel shift.
        w2 = np.zeros((channels, channels, 5, 5), np.float32)
        for c in range(channels):
            w2[c, c, 2 + 2 * dy, 2 + 2 * dx] = 1
        return conv(b, w2)

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    palette = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    one = conv("coords", [[0, 0, 1]])
    rank = conv("coords", [[4032, 63, 0]])
    xy = sl("coords", 0, 2)

    # Feature channels:
    # 0: possible variable colour; 1: clue marking;
    # 2: special-effect marking; 3: not background; 4: colour ID.
    wf = np.zeros((5, 16), np.float32)
    wf[0, palette] = 1
    wf[1, [0, 2, 3]] = 1
    wf[2, 6] = 1
    wf[3, :] = 1
    wf[3, 4] = 0
    wf[4, :] = np.arange(16, dtype=np.float32)
    features = conv("x0", wf)
    structural = sl(features, 0, 4)
    colour = sl(features, 4, 5)
    six = sl(features, 2, 3)
    zero = sl("x0", 0, 1)
    markings = cat([zero, six])

    # Detect the upper-left pixel of each tile's central 2x2 block.
    # Negative boundary tests also treat off-screen padding correctly.
    wd = np.zeros((2, 4, 7, 7), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = 3 + 2 * dy, 3 + 2 * dx
            wd[0, 0, yy, xx] += 1
            wd[0, 2, yy, xx] += 1
            if dy or dx:
                wd[1, 1, yy, xx] += 1
    wd[0, 0, 3, 3] += 1
    wd[1, 0, 3, 3] += 1
    for out in range(2):
        wd[out, 3, 0, 3] -= 1
        wd[out, 3, 3, 0] -= 1
    detections = unary("relu", conv(structural, wd, [-9, -8]))
    tile = sl(detections, 0, 1)
    clue = sl(detections, 1, 2)
    clue_colour = binary("mul", clue, colour)

    # Transport each clue's eight requirements to the affected tiles.
    violations = []
    adjacent_clues = []
    special_marks = []
    for dy, dx in directions:
        w = np.zeros((2, 2, 5, 5), np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        w[1, 1, 2 + 2 * dy, 2 + 2 * dx] = 1
        samples = conv(markings, w)
        requires_equal = binary("mul", clue, sl(samples, 0, 1))
        special_marks.append(sl(samples, 1, 2))

        transported = shift8(
            cat([clue_colour, clue, requires_equal]), 3, -dy, -dx
        )
        target_colour = sl(transported, 0, 1)
        present = sl(transported, 1, 2)
        equal_required = sl(transported, 2, 3)

        distance = unary("abs", binary("sub", colour, target_colour))
        equal_now = unary("relu", conv(distance, [[-1]], [1]))
        wrong = unary("abs", binary("sub", equal_now, equal_required))
        violations.append(binary("mul", wrong, present))
        adjacent_clues.append(present)

    bad = binary(
        "mul",
        binary("min2", spatial_sum(violations), one),
        tile
    )
    constrained = binary(
        "mul",
        binary("min2", spatial_sum(adjacent_clues), one),
        tile
    )
    special = binary(
        "mul",
        binary("min2", spatial_sum(special_marks), one),
        tile
    )
    ordinary = binary("sub", tile, special)

    # Neighbour observations serve both matrix construction and memory.
    tile_colour = binary("mul", tile, colour)
    cell_data = cat([constrained, tile_colour])
    neighbour_constraints = []
    neighbour_colours = []
    for dy, dx in directions:
        shifted = shift8(cell_data, 2, dy, dx)
        neighbour_constraints.append(sl(shifted, 0, 1))
        neighbour_colours.append(sl(shifted, 1, 2))
    local_colours = cat(neighbour_colours)

    # Memory:
    # [previous row, previous column, previous ordinary-tile flag,
    #  eight previous neighbour colours, eight learned effect bits].
    previous_xy = vs("mem", 0, 2)
    previous_ordinary = vs("mem", 2, 3)
    previous_colours = vs("mem", 3, 11)
    old_effects = vs("mem", 11, 19)

    previous_distance = unary("abs", binary("sub", xy, previous_xy))
    previous_location = unary(
        "relu", conv(previous_distance, [[-126, -126]], [1])
    )
    observed_colours = unary(
        "gmax", binary("mul", local_colours, previous_location)
    )
    colour_change = binary(
        "vadd", observed_colours, vneg(previous_colours, 8)
    )
    changed = vclip01(vabs(colour_change, 8), 8)
    previously_present = vclip01(previous_colours, 8)
    known = binary("vmul", previously_present, previous_ordinary)
    effects = binary(
        "vadd",
        binary("vmul", old_effects, vone_minus(known, 8)),
        binary("vmul", changed, known)
    )

    # Exact, margin-separated membership in the 8x8 tile-slot grid.
    wg = np.zeros((32, 3), np.float32)
    bg = np.zeros(32, np.float32)
    for axis in range(2):
        for b in range(8):
            i = 2 * (axis * 8 + b)
            wg[i, axis] = 126
            wg[i + 1, axis] = 126
            bg[i] = 1 - 16 * b
            bg[i + 1] = 1 - 16 * (b + 1)
    gates = binary("min2", unary("relu", conv("coords", wg, bg)), one)

    wb = np.zeros((16, 32), np.float32)
    for i in range(16):
        wb[i, 2 * i] = 1
        wb[i, 2 * i + 1] = -1
    bands = conv(gates, wb)

    ws = np.zeros((64, 16), np.float32)
    for r in range(8):
        for c in range(8):
            ws[8 * r + c, r] = 1
            ws[8 * r + c, 8 + c] = 1
    slots = unary("relu", conv(bands, ws, -np.ones(64, np.float32)))
    basis = binary("mul", slots, tile)

    # Matrix channels index equations; spatial positions index clicks.
    matrix = binary("mul", basis, constrained)
    rhs = unary("gmax", binary("mul", basis, bad))

    for d, (dy, dx) in enumerate(directions):
        learned = vs(effects, d, d + 1)
        effect = binary(
            "add", special_marks[d], binary("mul", ordinary, learned)
        )
        active_effect = binary("mul", effect, neighbour_constraints[d])
        weighted_basis = binary("mul", basis, active_effect)

        permutation = np.zeros((64, 64), np.float32)
        for r in range(8):
            for c in range(8):
                rr, cc = r + dy, c + dx
                if 0 <= rr < 8 and 0 <= cc < 8:
                    permutation[8 * rr + cc, 8 * r + c] = 1
        contribution = conv(weighted_basis, permutation)
        matrix = binary("add", matrix, contribution)

    # Gauss-Jordan elimination over GF(2).
    # Binary XOR is abs(a-b), so all carried matrix entries stay exact.
    first = np.zeros(64, np.float32)
    first[0] = 1
    pointer_init = dense("avail", np.zeros((64, 8), np.float32), first)

    body = []
    nodes = body

    astate = "a_state"
    bstate = "b_state"
    kstate = "k_state"

    selected_row_channels = binary("mul", astate, kstate)
    selected_row = conv(
        selected_row_channels, np.ones((1, 64), np.float32)
    )
    normal_entry = binary("mul", selected_row, ordinary)

    # Prefer ordinary pivot columns; leave useful special columns free.
    pivot_score = conv(
        cat([selected_row, normal_entry]), [[16384, 8192]]
    )
    pivot_score = binary("sub", pivot_score, rank)
    best_pivot_score = unary("gmax", pivot_score)
    pivot_delta = binary("sub", pivot_score, best_pivot_score)
    pivot_selector = unary("relu", conv(pivot_delta, [[2]], [1]))
    pivot_selector = binary("mul", pivot_selector, selected_row)

    pivot_column = unary(
        "gmax", binary("mul", astate, pivot_selector)
    )
    eliminate_rows = binary(
        "vmul", pivot_column, vone_minus(kstate, 64)
    )
    correction = binary("mul", selected_row, eliminate_rows)
    next_matrix = unary("abs", binary("sub", astate, correction))

    pivot_rhs = dense(
        binary("vmul", bstate, kstate),
        np.ones((1, 64), np.float32)
    )
    rhs_correction = binary("vmul", eliminate_rows, pivot_rhs)
    rhs_sign = dense(
        rhs_correction, -2 * np.eye(64, dtype=np.float32),
        np.ones(64, np.float32)
    )
    next_rhs = binary(
        "vadd", binary("vmul", bstate, rhs_sign), rhs_correction
    )

    advance = np.zeros((64, 64), np.float32)
    for i in range(63):
        advance[i + 1, i] = 1
    next_pointer = dense(kstate, advance)

    nodes = graph
    graph.append({
        "id": "eliminate",
        "op": "loop",
        "T": 64,
        "state": [
            {"id": astate, "init": matrix},
            {"id": bstate, "init": rhs},
            {"id": kstate, "init": pointer_init},
        ],
        "body": body,
        "next": {
            astate: next_matrix,
            bstate: next_rhs,
            kstate: next_pointer,
        },
    })

    # Gain is the exact decrease in transformed RHS Hamming weight.
    # Free columns can therefore be chosen when they beat a pivot click.
    error_sign = dense(
        "eliminate.b_state", 2 * np.eye(64, dtype=np.float32),
        -np.ones(64, np.float32)
    )
    gain = conv(
        binary("mul", "eliminate.a_state", error_sign),
        np.ones((1, 64), np.float32)
    )
    positive_gain = vclip01(unary("gmax", gain), 1)

    # Count visible palette colours, excluding the bottom status bar.
    palette_planes = cat([sl("x0", c, c + 1) for c in palette])
    nonbar = binary(
        "min2",
        unary("relu", conv("coords", [[-126, 0, 0]], [125])),
        one
    )
    palette_present = unary(
        "gmax", binary("mul", palette_planes, nonbar)
    )
    more_than_two = vclip01(
        dense(
            palette_present, np.ones((1, len(palette)), np.float32),
            [-2]
        ),
        1
    )
    use_linear = binary(
        "vmul", positive_gain, vone_minus(more_than_two, 1)
    )
    priority = binary(
        "add",
        binary("mul", gain, use_linear),
        binary("mul", bad, vone_minus(use_linear, 1))
    )

    # Integer-separated priorities and a deterministic positional tie-break.
    conv(
        cat([priority, tile, special, rank]),
        [[100000, 10000, 5000, -1]],
        name="click"
    )

    # Reconstruct the decoded click inside the graph for the next memory.
    click_max = unary("gmax", "click")
    click_delta = binary("sub", "click", click_max)
    selected_click = unary("relu", conv(click_delta, [[2]], [1]))

    next_xy = unary("gmax", binary("mul", xy, selected_click))
    next_ordinary = unary(
        "gmax", binary("mul", ordinary, selected_click)
    )
    next_colours = unary(
        "gmax", binary("mul", local_colours, selected_click)
    )
    emit(
        "vconcat", [next_xy, next_ordinary, next_colours, effects],
        name="mem_next"
    )

    action_bias = np.zeros(8, np.float32)
    action_bias[6] = 10
    dense(
        "avail", np.zeros((8, 8), np.float32),
        action_bias, name="logits"
    )

    return {
        "graph": graph,
        "memory_size": 19,
        "params": params,
    }

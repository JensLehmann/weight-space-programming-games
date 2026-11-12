import numpy as np


def build():
    P = {}
    G = []
    shapes = {
        "x0": (16, 64, 64),
        "coords": (3, 64, 64),
        "avail": (8,),
        "aprev": (8,),
    }
    serial = 0
    pserial = 0
    spatial_total = 19
    mac_total = 0
    evaluations = 0
    destination = G
    multiplier = 1

    def parameter(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def node(op, inputs, name=None, **fields):
        nonlocal serial, spatial_total, mac_total, evaluations
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destination.append(nd)

        ss = [shapes[x] for x in inputs]
        cost = 0
        if op == "conv":
            w = P[fields["w"]]
            shape = (w.shape[0], 64, 64)
            cost = int(np.prod(w.shape)) * 4096
        elif op == "dense":
            w = P[fields["w"]]
            shape = (w.shape[0],)
            cost = int(np.prod(w.shape))
        elif op == "concat":
            shape = (sum(s[0] for s in ss), 64, 64)
        elif op == "vconcat":
            shape = (sum(s[0] for s in ss),)
        elif op in ("slice", "vslice"):
            shape = (fields["stop"] - fields["start"],) + ss[0][1:]
        elif op in ("gmax", "gmean"):
            shape = (ss[0][0],)
            cost = ss[0][0] * 4096
        elif op in ("add", "sub", "mul", "min2", "max2"):
            shape = (max(s[0] for s in ss), 64, 64)
            cost = shape[0] * 4096
        elif op in ("vadd", "vmul"):
            shape = (max(s[0] for s in ss),)
            cost = shape[0]
        else:
            shape = ss[0]

        shapes[name] = shape
        if len(shape) == 3:
            spatial_total += shape[0]
        mac_total += multiplier * cost
        evaluations += multiplier
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def channel(x, i):
        return node("slice", [x], start=i, stop=i + 1)

    def total(xs):
        z = xs[0]
        for x in xs[1:]:
            z = node("add", [z, x])
        return z

    def sum_channels(x):
        c = shapes[x][0]
        return conv(x, np.ones((1, c, 1, 1), np.float32))

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    colours = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Numeric colour at every pixel.
    colour = conv(
        "x0",
        np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1),
    )

    # A tile anchor is the upper-left pixel of its central 2x2 block.
    # All 36 tile pixels must be its centre colour or marker colour 6.
    # The centre must have the centre colour, and the upper/left margins
    # must be background 4. This also excludes the tutorial illustrations.
    tw = np.zeros((len(colours), 16, 7, 7), np.float32)
    for o, c in enumerate(colours):
        tw[o, c, 1:7, 1:7] = 1
        tw[o, 6, 1:7, 1:7] = 1
        tw[o, c, 3, 3] += 1
        tw[o, 4, 0, 3] = 1
        tw[o, 4, 3, 0] = 1
    tile_channels = node(
        "relu",
        [conv("x0", tw, -38 * np.ones(len(colours), np.float32))],
    )
    tiles = sum_channels(tile_channels)

    mw = np.zeros((1, 16, 7, 7), np.float32)
    mw[0, 6, 1:7, 1:7] = 1
    marker_count = conv("x0", mw)
    special = node("min2", [marker_count, tiles])

    # Clue anchors: eight surrounding symbols, a coloured centre, and
    # background outside the tile. Symbol 0 means equality; 2/3 mean
    # inequality, exactly as in the simulator's goal test.
    cw = np.zeros((1, 16, 7, 7), np.float32)
    for dr, dc in directions:
        for c in (0, 2, 3):
            cw[0, c, 3 + 2 * dr, 3 + 2 * dc] = 1
    for c in colours:
        cw[0, c, 3, 3] = 1
    cw[0, 4, 0, 3] = 1
    cw[0, 4, 3, 0] = 1
    clues = node("relu", [conv("x0", cw, np.array([-10], np.float32))])
    clue_colour = node("mul", [clues, colour])

    # Sample both clue symbols and patterned-switch marker pixels.
    sw = np.zeros((16, 16, 5, 5), np.float32)
    for d, (dr, dc) in enumerate(directions):
        sw[d, 0, 2 + 2 * dr, 2 + 2 * dc] = 1
        sw[8 + d, 6, 2 + 2 * dr, 2 + 2 * dc] = 1
    samples = conv("x0", sw)
    zero_samples = [channel(samples, i) for i in range(8)]
    six_samples = [channel(samples, 8 + i) for i in range(8)]

    errors = []
    presences = []
    for d, (dr, dc) in enumerate(directions):
        z = node("concat", [clue_colour, clues, zero_samples[d]])

        # Transport the clue's requirement to its neighbour, eight
        # display pixels away. Three small shifts implement that distance.
        for distance in (3, 3, 2):
            k = 2 * distance + 1
            w = np.zeros((3, 3, k, k), np.float32)
            for c in range(3):
                w[c, c,
                  distance - dr * distance,
                  distance - dc * distance] = 1
            z = conv(z, w)

        required_colour = channel(z, 0)
        present = channel(z, 1)
        wants_equal = channel(z, 2)

        delta = node("abs", [node("sub", [colour, required_colour])])
        equal = node(
            "relu",
            [conv(delta, -np.ones((1, 1, 1, 1), np.float32),
                  np.ones(1, np.float32))],
        )
        wrong = node("abs", [node("sub", [equal, wants_equal])])
        errors.append(node("mul", [wrong, present]))
        presences.append(present)

    mismatches = node("mul", [total(errors), tiles])
    special_rhs = node("min2", [mismatches, special])
    constrained_special = node("min2", [total(presences), special])

    # One-hot spatial bank for the 8x8 possible board cells.
    # Quarter-pixel margins make these indicators exactly binary despite
    # float32 coordinate rounding.
    bw = np.zeros((256, 3, 1, 1), np.float32)
    bb = np.zeros(256, np.float32)
    for cell in range(64):
        r, c = divmod(cell, 8)
        q = 4 * cell
        bw[q, 0, 0, 0] = -63
        bb[q] = 8 * r - 0.25
        bw[q + 1, 0, 0, 0] = 63
        bb[q + 1] = -(8 * r + 7) - 0.25
        bw[q + 2, 1, 0, 0] = -63
        bb[q + 2] = 8 * c - 0.25
        bw[q + 3, 1, 0, 0] = 63
        bb[q + 3] = -(8 * c + 7) - 0.25

    outside = node("relu", [conv("coords", bw, bb)])
    rw = np.zeros((64, 256, 1, 1), np.float32)
    for cell in range(64):
        rw[cell, 4 * cell:4 * cell + 4, 0, 0] = -4
    bank = node(
        "relu",
        [conv(outside, rw, np.ones(64, np.float32))],
    )

    # Matrix columns are possible patterned-switch clicks; matrix rows
    # live at constrained patterned-switch anchors.
    special_columns = node(
        "gmax", [node("mul", [bank, special])],
    )
    matrix_terms = [node("mul", [bank, special_columns])]

    for d, (dr, dc) in enumerate(directions):
        local_marker = node("mul", [six_samples[d], special])
        column_flags = node(
            "gmax", [node("mul", [bank, local_marker])],
        )

        permutation = np.zeros((64, 64, 1, 1), np.float32)
        for column in range(64):
            r, c = divmod(column, 8)
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                permutation[column, 8 * rr + cc, 0, 0] = 1

        row_regions = conv(bank, permutation)
        matrix_terms.append(node("mul", [row_regions, column_flags]))

    matrix = node("mul", [total(matrix_terms), constrained_special])

    rank_w = np.array([1.0, 1.0 / 128.0, 1.0],
                      np.float32).reshape(1, 3, 1, 1)
    rank = conv("coords", rank_w)
    one = channel("coords", 2)

    # Fixed 64-column Gauss-Jordan elimination over GF(2).
    # Binary XOR is abs(a-b); an outer product is broadcast multiplication.
    body = []
    states = [
        {"id": "aa", "init": matrix},
        {"id": "rr", "init": special_rhs},
        {"id": "used_rows", "zeros": [1, 64, 64]},
        {"id": "pivot_columns", "zeros": [64]},
        {"id": "column_counter", "zeros": [1]},
    ]
    for s in states:
        shape = shapes[s["init"]] if "init" in s else tuple(s["zeros"])
        shapes[s["id"]] = shape
        if len(shape) == 3:
            spatial_total += shape[0]

    destination = body
    multiplier = 64

    # Exact one-hot column selector from an integer iteration counter:
    # relu(t-j+1) - 2 relu(t-j) + relu(t-j-1).
    iw = np.ones((192, 1), np.float32)
    ib = np.empty(192, np.float32)
    ow = np.zeros((64, 192), np.float32)
    for j in range(64):
        ib[3 * j:3 * j + 3] = [1 - j, -j, -1 - j]
        ow[j, 3 * j:3 * j + 3] = [1, -2, 1]
    ramps = node("vrelu", [dense("column_counter", iw, ib)])
    column_hot = dense(ramps, ow)

    column_values = sum_channels(node("mul", ["aa", column_hot]))
    unused = node("sub", [one, "used_rows"])
    eligible = node("mul", [column_values, unused])

    pivot_scores = node("mul", [eligible, rank])
    highest = node("gmax", [pivot_scores])
    gap = node("abs", [node("sub", [pivot_scores, highest])])
    at_highest = node(
        "relu",
        [conv(gap, np.array([[[[-20000.0]]]], np.float32),
              np.ones(1, np.float32))],
    )
    pivot = node("mul", [at_highest, eligible])

    pivot_row = node("gmax", [node("mul", ["aa", pivot])])
    pivot_rhs = node("gmax", [node("mul", ["rr", pivot])])
    factor = node("sub", [column_values, pivot])

    outer = node("mul", [factor, pivot_row])
    next_matrix = node("abs", [node("sub", ["aa", outer])])
    rhs_outer = node("mul", [factor, pivot_rhs])
    next_rhs = node("abs", [node("sub", ["rr", rhs_outer])])
    next_used = node("add", ["used_rows", pivot])

    exists = node("gmax", [pivot])
    marked_column = node("vmul", [column_hot, exists])
    next_pivots = node("vadd", ["pivot_columns", marked_column])
    next_counter = dense(
        "column_counter",
        np.ones((1, 1), np.float32),
        np.ones(1, np.float32),
    )

    destination = G
    multiplier = 1
    G.append({
        "id": "elimination",
        "op": "loop",
        "T": 64,
        "state": states,
        "body": body,
        "next": {
            "aa": next_matrix,
            "rr": next_rhs,
            "used_rows": next_used,
            "pivot_columns": next_pivots,
            "column_counter": next_counter,
        },
    })
    for s in states:
        shapes["elimination." + s["id"]] = shapes[s["id"]]

    # Free variables are zero. In reduced form each pivot column has
    # exactly one surviving row, so a global max extracts its RHS.
    solution = node(
        "gmax",
        [node("mul", ["elimination.aa", "elimination.rr"])],
    )
    solution = node(
        "vmul", [solution, "elimination.pivot_columns"],
    )
    solution_map = sum_channels(node("mul", [bank, solution]))
    planned_special = node("mul", [solution_map, special])

    # Finish patterned-switch corrections before independent ordinary
    # corrections. A small coordinate term gives deterministic tie-breaking.
    features = node(
        "concat", [planned_special, mismatches, tiles, "coords"],
    )
    click_w = np.array(
        [100.0, 10.0, 1.0, -0.0001, -0.000001, 0.0],
        np.float32,
    ).reshape(1, 6, 1, 1)
    conv(features, click_w, np.array([-100.0], np.float32), name="click")

    action_bias = np.zeros(8, np.float32)
    action_bias[6] = 1
    dense(
        "avail",
        np.zeros((8, 8), np.float32),
        action_bias,
        name="logits",
    )

    assert sum(a.size for a in P.values()) <= 150000
    assert spatial_total <= 4096
    assert mac_total <= 1500000000
    assert len(G) + len(body) <= 1000
    assert evaluations + 1 <= 10000
    assert all(np.isfinite(a).all() for a in P.values())

    return {"graph": G, "memory_size": 0, "params": P}

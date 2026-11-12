import numpy as np


def build():
    P = {}
    graph = []
    current = graph
    serial = 0
    pserial = 0

    def param(a):
        nonlocal pserial
        name = "p%d" % pserial
        pserial += 1
        P[name] = np.asarray(a, dtype=np.float32)
        return name

    def emit(op, ins=None, name=None, **fields):
        nonlocal serial
        if name is None:
            name = "n%d" % serial
            serial += 1
        nd = {"id": name, "op": op}
        if ins is not None:
            nd["in"] = list(ins)
        nd.update(fields)
        current.append(nd)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return emit("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return emit("dense", [x], name=name, **fields)

    def point(x, w, b=None, name=None):
        w = np.atleast_2d(np.asarray(w, np.float32))
        return conv(x, w[:, :, None, None], b, name)

    def scalar(x, a, b=0.0, name=None):
        return point(x, [[a]], [b], name)

    def unary(op, x):
        return emit(op, [x])

    def binary(op, a, b):
        return emit(op, [a, b])

    def cat(xs):
        return emit("concat", xs)

    def vcat(xs, name=None):
        return emit("vconcat", xs, name=name)

    def sl(x, start, stop):
        return emit("slice", [x], start=start, stop=stop)

    def vsl(x, start, stop):
        return emit("vslice", [x], start=start, stop=stop)

    def add(a, b):
        return binary("add", a, b)

    def sub(a, b):
        return binary("sub", a, b)

    def mul(a, b):
        return binary("mul", a, b)

    def va(a, b):
        return binary("vadd", a, b)

    def vm(a, b):
        return binary("vmul", a, b)

    def relu(x):
        return unary("relu", x)

    def absolute(x):
        return unary("abs", x)

    def gm(x):
        return unary("gmax", x)

    def vs_const(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense("avail", np.zeros((len(values), 8)), values)

    def vneg(x, n):
        return dense(x, -np.eye(n, dtype=np.float32))

    def vone_minus(x, n):
        return dense(x, -np.eye(n, dtype=np.float32),
                     np.ones(n, dtype=np.float32))

    def clip01(x, channels=1):
        a = relu(x)
        b = point(
            x, np.eye(channels, dtype=np.float32),
            -np.ones(channels, dtype=np.float32)
        )
        return sub(a, relu(b))

    def sum_maps(xs):
        out = xs[0]
        for x in xs[1:]:
            out = add(out, x)
        return out

    def sample_sum(x, include_center, bias=0.0):
        w = np.zeros((1, 1, 5, 5), np.float32)
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                if include_center or dy != 0 or dx != 0:
                    w[0, 0, dy + 2, dx + 2] = 1
        return conv(x, w, [bias])

    def shift(x, channels, dy, dx, distance):
        k = 2 * distance + 1
        w = np.zeros((channels, channels, k, k), np.float32)
        for c in range(channels):
            w[c, c, distance + dy * distance,
              distance + dx * distance] = 1
        return conv(x, w)

    one = sl("coords", 2, 3)

    # A unique, stable pixel key, also used for remembering the last click.
    pixel_key = point("coords", [[4032.0, 63.0, 1.0]])

    def colour_plane(c):
        w = np.zeros((1, 16), np.float32)
        w[0, c] = 1
        return point("x0", w)

    colour = point("x0", [list(range(16))])
    zero = colour_plane(0)
    six = colour_plane(6)
    background = colour_plane(4)

    palette_w = np.zeros((1, 16), np.float32)
    palette_w[0, [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 1
    palette = point("x0", palette_w)

    pattern_w = np.zeros((1, 16), np.float32)
    pattern_w[0, [0, 2, 3]] = 1
    pattern_pixel = point("x0", pattern_w)

    # At scale two, sprite-centre pixels see their 3x3 template at
    # offsets -2, 0, +2. The surrounding moat is at offsets +/-4.
    acceptable = add(palette, six)
    filled = relu(sample_sum(acceptable, True, -8))

    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    w1 = np.zeros((4, 1, 5, 5), np.float32)
    w2 = np.zeros((1, 4, 5, 5), np.float32)
    for j, (dy, dx) in enumerate(cardinal):
        w1[j, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        w2[0, j, 2 + 2 * dy, 2 + 2 * dx] = 1
    moat_first = conv(background, w1)
    moat = relu(conv(moat_first, w2, [-3]))

    centre_ok = mul(palette, moat)
    tile = mul(filled, centre_ok)
    clue_shape = relu(sample_sum(pattern_pixel, False, -7))
    clue = mul(clue_shape, centre_ok)

    has_six = clip01(sample_sum(six, False))
    special = mul(tile, has_six)
    normal = sub(tile, special)

    # Field order: playable tile, colour, special tile, clue, zero, six.
    fields = cat([tile, colour, special, clue, zero, six])
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),             (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]
    offsets = [(0, 0)] + directions
    offset_index = {d: i for i, d in enumerate(offsets)}

    neighbours = []
    violations = []
    adjacent_clues = []

    for dy, dx in directions:
        s3 = shift(fields, 6, dy, dx, 3)
        s6 = shift(s3, 6, dy, dx, 3)
        s8 = shift(s6, 6, dy, dx, 2)

        nt = sl(s8, 0, 1)
        nc = sl(s8, 1, 2)
        ns = sl(s8, 2, 3)
        nq = sl(s8, 3, 4)
        relation_zero = sl(s6, 4, 5)
        effect_six = sl(s6, 5, 6)

        neighbours.append({
            "tile": nt,
            "colour": nc,
            "special": ns,
            "clue": nq,
            "six": effect_six,
        })
        adjacent_clues.append(nq)

        same = relu(sub(one, absolute(sub(colour, nc))))
        product = mul(same, relation_zero)
        # XOR(same, relation_zero): zero means equality is required.
        mismatch = point(
            cat([same, relation_zero, product]),
            [[1.0, 1.0, -2.0]]
        )
        violations.append(mul(mismatch, nq))

    bad = mul(tile, clip01(sum_maps(violations)))
    constrained = mul(tile, clip01(sum_maps(adjacent_clues)))

    local_colours = cat(
        [colour] + [n["colour"] for n in neighbours]
    )
    local_tiles = cat(
        [tile] + [n["tile"] for n in neighbours]
    )

    # Memory:
    #   0:9   learned ordinary-action footprint
    #   9:18  which footprint entries have been observed
    #   18    preceding click's pixel key
    #   19    preceding click was on an ordinary tile
    #   20:29 preceding local colours
    #   29:38 preceding local tile-presence flags
    old_bits = vsl("mem", 0, 9)
    old_known = vsl("mem", 9, 18)
    old_key = vsl("mem", 18, 19)
    old_normal = vsl("mem", 19, 20)
    old_colours = vsl("mem", 20, 29)
    old_valid = vsl("mem", 29, 38)

    old_position = relu(
        scalar(absolute(sub(pixel_key, old_key)), -8.0, 1.0)
    )
    now_colours = gm(mul(local_colours, old_position))
    now_valid = gm(mul(local_tiles, old_position))

    difference = va(now_colours, vneg(old_colours, 9))
    difference_abs = va(
        unary("vrelu", difference),
        unary("vrelu", vneg(difference, 9))
    )
    unchanged = unary("vrelu", vone_minus(difference_abs, 9))
    changed = vone_minus(unchanged, 9)

    observed = vm(vm(old_valid, now_valid), old_normal)
    not_observed = vone_minus(observed, 9)
    new_bits = va(vm(old_bits, not_observed), vm(changed, observed))
    new_known = va(
        old_known, vm(observed, vone_minus(old_known, 9))
    )

    prior_values = np.zeros(9, np.float32)
    prior_values[0] = 1
    prior = vs_const(prior_values)
    footprint = va(
        new_bits, vm(prior, vone_minus(new_known, 9))
    )
    footprint_parts = [vsl(footprint, i, i + 1) for i in range(9)]

    # Assign one action variable to each 8x8 display bin.
    # The centre pixels of any one tile lie in the same bin.
    def bins(axis):
        w = np.zeros((8, 3), np.float32)
        w[:, axis] = 63
        centres = 8 * np.arange(8, dtype=np.float32) + 3.5
        distance = absolute(point("coords", w, -centres))
        raw = point(
            distance, -4 * np.eye(8, dtype=np.float32),
            np.full(8, 16, np.float32)
        )
        return clip01(raw, 8)

    row_bins = bins(0)
    col_bins = bins(1)

    wr = np.zeros((64, 8), np.float32)
    wc = np.zeros((64, 8), np.float32)
    for r in range(8):
        for c in range(8):
            wr[8 * r + c, r] = 1
            wc[8 * r + c, c] = 1

    slots = mul(point(row_bins, wr), point(col_bins, wc))
    slot_index = point(
        slots, np.arange(64, dtype=np.float32)[None, :]
    )
    pivot_priority = scalar(slot_index, -1, 64)

    # Matrix rows live at tile-centre pixels; its 64 channels are
    # action variables. Ordinary effects use the learned footprint.
    self_effect = add(mul(normal, footprint_parts[0]), special)
    matrix = mul(slots, self_effect)

    for (dy, dx), n in zip(directions, neighbours):
        neighbour_normal = sub(n["tile"], n["special"])
        reverse = offset_index[(-dy, -dx)]
        ordinary_effect = mul(
            neighbour_normal, footprint_parts[reverse]
        )
        special_effect = mul(n["tile"], n["six"])
        effect = add(ordinary_effect, special_effect)

        permutation = np.zeros((64, 64), np.float32)
        for r in range(8):
            for c in range(8):
                rr, cc = r + dy, c + dx
                if 0 <= rr < 8 and 0 <= cc < 8:
                    permutation[8 * rr + cc, 8 * r + c] = 1

        neighbour_slots = point(slots, permutation)
        matrix = add(matrix, mul(neighbour_slots, effect))

    matrix = mul(matrix, constrained)

    # Fixed Gauss-Jordan elimination over GF(2).
    # For binary tensors, abs(a-b) is XOR.
    body = []
    current = body

    A = "gs_A"
    rhs = "gs_rhs"
    used = "gs_used"
    assignment = "gs_assignment"
    clock = "gs_clock"

    column_location = relu(
        sub(one, absolute(sub(slot_index, clock)))
    )
    column_vector = gm(mul(slots, column_location))
    coefficient = point(
        mul(A, column_vector), np.ones((1, 64), np.float32)
    )

    eligible = mul(coefficient, sub(one, used))
    row_score = mul(eligible, pivot_priority)
    maximum = gm(row_score)
    equal_maximum = relu(
        sub(one, absolute(sub(row_score, maximum)))
    )
    pick = mul(eligible, equal_maximum)

    pivot_row = gm(mul(A, pick))
    pivot_rhs = gm(mul(rhs, pick))
    eliminate = sub(coefficient, pick)

    next_A = absolute(sub(A, mul(eliminate, pivot_row)))
    next_rhs = absolute(sub(rhs, mul(eliminate, pivot_rhs)))
    next_used = add(used, pick)
    next_clock = dense(clock, [[1.0]], [1.0])
    next_assignment = add(
        assignment, mul(pick, next_clock)
    )

    current = graph
    emit(
        "loop",
        name="elimination",
        T=64,
        state=[
            {"id": A, "init": matrix},
            {"id": rhs, "init": bad},
            {"id": used, "zeros": [1, 64, 64]},
            {"id": assignment, "zeros": [1, 64, 64]},
            {"id": clock, "zeros": [1]},
        ],
        body=body,
        next={
            A: next_A,
            rhs: next_rhs,
            used: next_used,
            assignment: next_assignment,
            clock: next_clock,
        },
    )

    final_rhs = "elimination." + rhs
    final_assignment = "elimination." + assignment

    variable_numbers = vs_const(np.arange(1, 65, dtype=np.float32))
    ones64 = vs_const(np.ones(64, np.float32))
    assigned_variables = relu(
        sub(
            ones64,
            absolute(sub(final_assignment, variable_numbers))
        )
    )
    solution = gm(mul(assigned_variables, final_rhs))
    solution_map = point(
        mul(slots, solution), np.ones((1, 64), np.float32)
    )
    candidate = mul(tile, solution_map)

    # If an initially unknown footprint leaves no selected correction,
    # make a direct corrective click and observe its actual effects.
    any_candidate = gm(candidate)
    fallback = mul(bad, sub(one, any_candidate))
    candidate = binary("max2", candidate, fallback)

    # Any selected solution variable may be executed. Prefer special
    # actions, then use a deterministic row-major order.
    click_priority = point(
        cat([pixel_key, special]),
        [[-1.0, 16384.0]],
        [8192.0]
    )
    click_score = mul(candidate, click_priority)
    emit("add", [click_score, tile], name="click")

    # Reconstruct the decoder's unique winner for transition memory.
    winning_score = gm("click")
    chosen = relu(
        scalar(absolute(sub("click", winning_score)), -4.0, 1.0)
    )
    next_key = gm(mul(pixel_key, chosen))
    next_normal = gm(mul(normal, chosen))
    next_colours = gm(mul(local_colours, chosen))
    next_valid = gm(mul(local_tiles, chosen))

    vcat(
        [
            new_bits, new_known, next_key, next_normal,
            next_colours, next_valid,
        ],
        name="mem_next",
    )

    logits_bias = np.full(8, -10.0, np.float32)
    logits_bias[6] = 10.0
    dense(
        "avail", np.zeros((8, 8), np.float32),
        logits_bias, name="logits"
    )

    return {
        "graph": graph,
        "memory_size": 38,
        "params": P,
    }

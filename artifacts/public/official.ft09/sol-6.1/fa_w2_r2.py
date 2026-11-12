import numpy as np


def build():
    params = {}
    graph = []
    node_counter = 0
    param_counter = 0

    def parameter(value):
        nonlocal param_counter
        name = "p%d" % param_counter
        param_counter += 1
        params[name] = np.asarray(value, dtype=np.float32)
        return name

    def node(op, inputs, target=None, ident=None, **fields):
        nonlocal node_counter
        if target is None:
            target = graph
        if ident is None:
            ident = "n%d" % node_counter
            node_counter += 1
        nd = {"id": ident, "op": op, "in": list(inputs)}
        nd.update(fields)
        target.append(nd)
        return ident

    def conv(x, w, b=None, target=None, ident=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], target, ident, **fields)

    def dense(x, w, b=None, target=None, ident=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], target, ident, **fields)

    def sl(x, a, b, target=None):
        return node("slice", [x], target, start=a, stop=b)

    def unary(op, x, target=None):
        return node(op, [x], target)

    def binary(op, a, b, target=None):
        return node(op, [a, b], target)

    def cat(xs, target=None):
        return node("concat", xs, target)

    def vcat(xs, target=None):
        return node("vconcat", xs, target)

    def pointwise(x, weights, bias=None, target=None):
        w = np.asarray(weights, np.float32).reshape(1, -1, 1, 1)
        b = None if bias is None else [bias]
        return conv(x, w, b, target)

    constant_plane = sl("coords", 2, 3)
    one = unary("gmax", constant_plane)

    def constant(value):
        return dense(one, np.zeros((1, 1), np.float32), [value])

    zero = constant(0)
    minus_one = constant(-1)
    minus_two = constant(-2)
    two = constant(2)
    four = constant(4)
    minus_four = constant(-4)
    area = constant(4096)
    special_penalty = constant(4096000)
    score_offset = constant(100000)

    def clip_spatial(x, target=None):
        excess = unary("relu", binary("sub", x, one, target), target)
        return binary("sub", x, excess, target)

    # A logical sprite pixel occupies a 2x2 display block.
    # Recognition is aligned to the upper-left pixel of its centre block.
    colours = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nc = len(colours)
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    detector = np.zeros((2 * nc + 2, 16, 7, 7), np.float32)
    detector_bias = np.full(2 * nc + 2, -10, np.float32)
    detector_bias[-1] = 0

    for q, colour in enumerate(colours):
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                detector[q, colour, 3 + dy, 3 + dx] = 1
        detector[q, 4, 0, 3] = 1
        detector[q, 4, 3, 0] = 1

        out = nc + q
        detector[out, colour, 3, 3] = 1
        for dx, dy in directions:
            detector[out, colour, 3 + 2 * dy, 3 + 2 * dx] = 1
            detector[out, 6, 3 + 2 * dy, 3 + 2 * dx] = 1
        detector[out, 4, 0, 3] = 1
        detector[out, 4, 3, 0] = 1

    clue_channel = 2 * nc
    for colour in colours:
        detector[clue_channel, colour, 3, 3] = 1
    for dx, dy in directions:
        for colour in (0, 2, 3):
            detector[clue_channel, colour,
                     3 + 2 * dy, 3 + 2 * dx] = 1
    detector[clue_channel, 4, 0, 3] = 1
    detector[clue_channel, 4, 3, 0] = 1

    for dx, dy in directions:
        detector[-1, 6, 3 + 2 * dy, 3 + 2 * dx] = 1

    detected = unary("relu", conv("x0", detector, detector_bias))
    ordinary_channels = sl(detected, 0, nc)
    coupled_channels = sl(detected, nc, 2 * nc)
    clue = sl(detected, clue_channel, clue_channel + 1)
    has_six = clip_spatial(
        sl(detected, clue_channel + 1, clue_channel + 2)
    )
    coupled_channels = binary("mul", coupled_channels, has_six)

    ordinary = pointwise(ordinary_channels, np.ones(nc))
    coupled = pointwise(coupled_channels, np.ones(nc))
    tiles = binary("add", ordinary, coupled)

    colour_number = conv(
        "x0", np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    )
    clue_number = binary("mul", colour_number, clue)

    def shift_eight(x, channels, dx, dy):
        w7 = np.zeros((channels, channels, 7, 7), np.float32)
        w5 = np.zeros((channels, channels, 5, 5), np.float32)
        for c in range(channels):
            w7[c, c, 3 - 3 * dy, 3 - 3 * dx] = 1
            w5[c, c, 2 - 2 * dy, 2 - 2 * dx] = 1
        p7 = parameter(w7)
        p5 = parameter(w5)
        y = node("conv", [x], w=p7)
        y = node("conv", [y], w=p7)
        return node("conv", [y], w=p5)

    violations = []
    constraint_fields = []
    for dx, dy in directions:
        w = np.zeros((1, 16, 5, 5), np.float32)
        w[0, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
        equal_symbol = binary("mul", conv("x0", w), clue)

        message = cat([clue_number, clue, equal_symbol])
        message = shift_eight(message, 3, dx, dy)
        neighbour_colour = sl(message, 0, 1)
        exists = sl(message, 1, 2)
        requires_equal = sl(message, 2, 3)

        difference = unary(
            "abs", binary("sub", colour_number, neighbour_colour)
        )
        equal = unary("relu", pointwise(difference, [-1], 1))
        matching_present = binary("mul", exists, equal)
        matching_required = binary("mul", requires_equal, equal)
        violation = pointwise(
            cat([matching_present, requires_equal, matching_required]),
            [1, 1, -2]
        )
        violations.append(violation)
        constraint_fields.append(exists)

    wrong = clip_spatial(pointwise(cat(violations), np.ones(8)))
    constrained = clip_spatial(
        pointwise(cat(constraint_fields), np.ones(8))
    )
    wrong = binary("mul", wrong, tiles)

    constrained_ordinary = binary("mul", constrained, ordinary)
    constrained_coupled = binary("mul", constrained, coupled)
    wrong_ordinary = binary("mul", wrong, ordinary)
    wrong_coupled = binary("mul", wrong, coupled)

    position_w = np.zeros((1, 3, 1, 1), np.float32)
    position_w[0, 0, 0, 0] = 64 * 63
    position_w[0, 1, 0, 0] = 63
    position = conv("coords", position_w)

    # Rank coupled centres in row-major order.
    remaining = coupled
    ranked = []
    for _ in range(8):
        ranking_score = pointwise(
            cat([remaining, position]), [10000, -1], -10000
        )
        peak = unary("gmax", ranking_score)
        at_peak = unary(
            "relu",
            pointwise(binary("sub", ranking_score, peak), [1], 1)
        )
        selected = binary("mul", at_peak, remaining)
        ranked.append(selected)
        remaining = binary("sub", remaining, selected)

    ranked_maps = cat(ranked)

    marker_w = np.zeros((8, 16, 5, 5), np.float32)
    for d, (dx, dy) in enumerate(directions):
        marker_w[d, 6, 2 + 2 * dy, 2 + 2 * dx] = 1
    marker_fields = conv("x0", marker_w)

    influences = ranked_maps
    for d, (dx, dy) in enumerate(directions):
        marker = sl(marker_fields, d, d + 1)
        triggered = binary("mul", ranked_maps, marker)
        shifted = shift_eight(triggered, 8, dx, dy)
        influences = binary("add", influences, shifted)

    # Six subset bits run in parallel; two run across four loop iterations.
    bits = np.zeros((64, 8), np.float32)
    for option in range(64):
        for bit in range(6):
            bits[option, bit] = (option >> bit) & 1

    subset_w = parameter(bits.reshape(64, 8, 1, 1))
    base_influences = node("conv", [influences], w=subset_w)
    base_click_maps = node("conv", [ranked_maps], w=subset_w)
    base_click_counts = binary(
        "vmul", unary("gmean", base_click_maps), area
    )

    influence_six = sl(influences, 6, 7)
    influence_seven = sl(influences, 7, 8)
    exists_six = unary("gmax", ranked[6])
    exists_seven = unary("gmax", ranked[7])

    anchors = dense(
        one, np.zeros((64, 1), np.float32),
        np.arange(64, dtype=np.float32)
    )
    anchor_distance = unary("abs", binary("sub", position, anchors))
    anchor_score = binary(
        "add", binary("mul", anchor_distance, minus_four), two
    )
    option_locations = binary(
        "min2", unary("relu", anchor_score), one
    )

    body = []

    def bu(op, x):
        return unary(op, x, body)

    def bb(op, a, b):
        return binary(op, a, b, body)

    def vector_abs(x):
        negative = bb("vmul", x, minus_one)
        return bb("vadd", bu("vrelu", x), bu("vrelu", negative))

    folded_index = vector_abs(bb("vadd", "batch_index", minus_two))
    folded_index = vector_abs(bb("vadd", folded_index, minus_one))
    low_bit = bb(
        "vadd", one, bb("vmul", folded_index, minus_one)
    )

    high_a = bu("vrelu", bb("vadd", "batch_index", minus_one))
    high_b = bu("vrelu", bb("vadd", "batch_index", minus_two))
    high_bit = bb("vadd", high_a, bb("vmul", high_b, minus_one))

    extra_six = bb("mul", influence_six, low_bit)
    extra_seven = bb("mul", influence_seven, high_bit)
    extra = bb("add", extra_six, extra_seven)
    counts = bb("add", base_influences, extra)

    # Exact integer parity for the possible toggle counts 0 through 8.
    parity_fold = bu("abs", bb("sub", counts, four))
    parity_fold = bu("abs", bb("sub", parity_fold, two))
    parity_fold = bu("abs", bb("sub", parity_fold, one))
    parity = bb("sub", one, parity_fold)

    projected_wrong = bu("abs", bb("sub", wrong, parity))
    ordinary_errors = bb(
        "mul", projected_wrong, constrained_ordinary
    )
    coupled_errors = bb(
        "mul", projected_wrong, constrained_coupled
    )
    ordinary_cost = bb("vmul", bu("gmean", ordinary_errors), area)
    coupled_cost = bb(
        "vmul", bu("gmean", coupled_errors), special_penalty
    )

    extra_clicks = bb(
        "vadd",
        bb("vmul", exists_six, low_bit),
        bb("vmul", exists_seven, high_bit)
    )
    click_cost = bb("vadd", base_click_counts, extra_clicks)
    cost = bb(
        "vadd", bb("vadd", ordinary_cost, coupled_cost), click_cost
    )
    scores = bb(
        "vadd", bb("vmul", cost, minus_one), score_offset
    )

    spatial_scores = bb("mul", option_locations, scores)
    reduced_scores = pointwise(
        spatial_scores, np.ones(64), target=body
    )
    batch_best = bu("gmax", reduced_scores)

    winner_difference = bb(
        "vadd", scores, bb("vmul", batch_best, minus_one)
    )
    winners = bu("vrelu", bb("vadd", winner_difference, one))
    selected_low_bits = dense(
        winners, bits[:, :6].T, target=body
    )
    batch_choices = vcat(
        [selected_low_bits, low_bit, high_bit], body
    )

    improvement = bb(
        "vadd", batch_best, bb("vmul", "best_value", minus_one)
    )
    positive_improvement = bu("vrelu", improvement)
    better = bb(
        "vadd",
        positive_improvement,
        bb(
            "vmul",
            bu("vrelu", bb("vadd", improvement, minus_one)),
            minus_one
        )
    )
    not_worse = bb(
        "vadd",
        bu("vrelu", bb("vadd", improvement, one)),
        bb("vmul", positive_improvement, minus_one)
    )
    retain_old = bb("vadd", one, bb("vmul", better, minus_one))
    next_choices = bb(
        "vadd",
        bb("vmul", "chosen_union", retain_old),
        bb("vmul", batch_choices, not_worse)
    )
    next_best = bb("vadd", "best_value", positive_improvement)
    next_index = bb("vadd", "batch_index", one)

    graph.append({
        "id": "plan",
        "op": "loop",
        "T": 4,
        "state": [
            {"id": "batch_index", "init": zero},
            {"id": "best_value", "init": zero},
            {"id": "chosen_union", "zeros": [8]},
        ],
        "body": body,
        "next": {
            "batch_index": next_index,
            "best_value": next_best,
            "chosen_union": next_choices,
        },
    })

    choice_excess = unary(
        "vrelu", binary("vadd", "plan.chosen_union", minus_one)
    )
    chosen_bits = binary(
        "vadd", "plan.chosen_union",
        binary("vmul", choice_excess, minus_one)
    )
    chosen_centres = binary("mul", ranked_maps, chosen_bits)
    chosen_centres = pointwise(chosen_centres, np.ones(8))

    click_features = cat([
        chosen_centres, wrong_ordinary, wrong_coupled, tiles
    ])
    conv(
        click_features,
        np.asarray([1000, 1, 2, 0.001], np.float32).reshape(1, 4, 1, 1),
        ident="click"
    )

    action_bias = np.zeros(8, np.float32)
    action_bias[6] = 1
    dense(
        one, np.zeros((8, 1), np.float32),
        action_bias, ident="logits"
    )

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

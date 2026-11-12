import numpy as np


def build():
    graph = []
    params = {}
    counter = 0
    param_counter = 0
    target = graph
    shift_weights = {}

    def node(op, inputs=None, **fields):
        nonlocal counter
        name = "n%d" % counter
        counter += 1
        nd = {"id": name, "op": op}
        if inputs is not None:
            nd["in"] = list(inputs)
        nd.update(fields)
        target.append(nd)
        return name

    def parameter(a):
        nonlocal param_counter
        name = "p%d" % param_counter
        param_counter += 1
        params[name] = np.asarray(a, dtype=np.float32)
        return name

    def conv(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], **fields)

    def spatial_affine(x, scale=1.0, bias=0.0):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def vector_affine(x, scale=1.0, bias=0.0):
        return dense(
            x,
            np.array([[scale]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def op(name, a, b=None):
        return node(name, [a] if b is None else [a, b])

    def spatial_equal(a, b):
        d = op("abs", op("sub", a, b))
        return op("relu", spatial_affine(d, -1.0, 1.0))

    def vector_equal(a, b, offset=0.0):
        ab = node("vconcat", [a, b])
        z = dense(
            ab,
            np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32),
            np.array([-offset, offset], dtype=np.float32),
        )
        z = node("vrelu", [z])
        z = dense(
            z,
            np.array([[-1.0, -1.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        return node("vrelu", [z])

    def shift_once(x, dy, dx):
        key = (dy, dx)
        if key not in shift_weights:
            w = np.zeros((1, 1, 7, 7), dtype=np.float32)
            w[0, 0, 3 + dy, 3 + dx] = 1.0
            shift_weights[key] = parameter(w)
        return node("conv", [x], w=shift_weights[key])

    def shift(x, dy=0, dx=0):
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            x = shift_once(x, sy, sx)
            dy -= sy
            dx -= sx
        return x

    def gmax(x):
        return node("gmax", [x])

    def gsum(x):
        return vector_affine(node("gmean", [x]), 4096.0, 0.0)

    def sum_spatial(xs):
        y = xs[0]
        for x in xs[1:]:
            y = op("add", y, x)
        return y

    def sum_vectors(xs):
        y = xs[0]
        for x in xs[1:]:
            y = node("vadd", [y, x])
        return y

    def vproduct(a, b):
        return node("vmul", [a, b])

    def forward_direction(current_id, desired_id, current_known, desired_known):
        differences = (1, 2, 3, -4, -5, -6)
        cases = [
            vector_equal(desired_id, current_id, float(d))
            for d in differences
        ]
        forward_known = sum_vectors(cases)
        both_known = vproduct(current_known, desired_known)
        unknown = vector_affine(both_known, -1.0, 1.0)
        return node(
            "vadd",
            [vproduct(both_known, forward_known), unknown],
        )

    one_spatial = sl("coords", 2, 3)
    row = spatial_affine(sl("coords", 0, 1), 63.0, 0.0)
    col = spatial_affine(sl("coords", 1, 2), 63.0, 0.0)
    black = sl("x0", 5, 6)
    color3 = sl("x0", 3, 4)
    color0 = sl("x0", 0, 1)

    # Detect glyph centers from the colored 7x7 border.
    colors = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    border_w = np.zeros((len(colors), 16, 7, 7), dtype=np.float32)
    for j, c in enumerate(colors):
        border_w[j, c, 0, :] = 1
        border_w[j, c, 6, :] = 1
        border_w[j, c, 1:6, 0] = 1
        border_w[j, c, 1:6, 6] = 1
    border = conv(
        "x0", border_w, np.full(len(colors), -23.5, np.float32)
    )
    border = node("relu", [border])
    tile = conv(
        border,
        np.full((1, len(colors), 1, 1), 2.0, dtype=np.float32),
    )

    # Two rotation-invariant fingerprints for equality comparisons.
    def signature(which):
        weights = np.zeros((4, 1, 5, 5), dtype=np.float32)
        base = np.zeros((5, 5), dtype=np.float32)
        for y in range(5):
            for x in range(5):
                i = 5 * y + x
                if which == 0:
                    base[y, x] = (
                        1 + (41 * i * i + 73 * i + 27 * x + 19) % 137
                    )
                else:
                    base[y, x] = (
                        1 + (31 * i * i + 59 * i + 43 * y + 11) % 149
                    )
        for r in range(4):
            weights[r, 0] = np.rot90(base, r)
        z = conv(black, weights)
        parts = [sl(z, r, r + 1) for r in range(4)]
        result = parts[0]
        for p in parts[1:]:
            result = op("max2", result, p)
        return result

    h1 = signature(0)
    h2 = signature(1)

    # Recognize known glyph-cycle positions, allowing all four rotations.
    patterns = [
        ("00100", "00100", "01110", "00100", "11111"),
        ("00001", "00101", "11111", "10100", "10000"),
        ("11111", "10100", "10000", "10100", "11111"),
        ("00100", "11111", "10101", "10101", "00100"),
        ("10001", "11111", "00100", "11111", "10001"),
        ("11111", "00100", "11100", "10000", "11111"),
        ("10001", "11111", "10001", "10001", "11011"),
        ("10000", "11110", "10010", "10010", "11111"),
        ("11111", "10001", "11101", "10101", "11111"),
        ("00111", "00101", "11111", "10100", "11100"),
        ("01110", "01010", "11111", "10001", "11111"),
        ("11111", "10001", "10001", "11111", "00100"),
        ("11110", "10011", "10001", "11001", "01111"),
        ("00100", "11111", "10101", "11111", "00100"),
        ("10101", "10000", "10101", "10000", "11111"),
        ("11101", "10000", "00000", "00001", "10111"),
        ("10111", "00100", "00100", "00100", "11101"),
        ("00100", "10101", "00100", "10101", "00100"),
        ("11011", "10001", "10101", "10001", "11011"),
        ("10111", "10101", "11101", "00000", "10101"),
        ("10001", "00000", "10001", "11011", "01010"),
    ]
    p_arrays = [
        np.array(
            [[int(c) for c in row_text] for row_text in pat],
            dtype=np.float32,
        )
        for pat in patterns
    ]
    recognizer_w = np.zeros(
        (4 * len(patterns), 1, 5, 5), dtype=np.float32
    )
    recognizer_b = np.zeros(4 * len(patterns), dtype=np.float32)
    for r in range(4):
        for i, pat in enumerate(p_arrays):
            q = np.rot90(pat, r)
            j = r * len(patterns) + i
            recognizer_w[j, 0] = 2.0 * q - 1.0
            recognizer_b[j] = 1.0 - q.sum()
    matches = node("relu", [conv(black, recognizer_w, recognizer_b)])
    groups = [
        sl(matches, r * len(patterns), (r + 1) * len(patterns))
        for r in range(4)
    ]
    best = groups[0]
    for group in groups[1:]:
        best = op("max2", best, group)
    compress = np.zeros((2, len(patterns), 1, 1), dtype=np.float32)
    for i in range(len(patterns)):
        compress[0, i, 0, 0] = float(i % 7 + 1)
        compress[1, i, 0, 0] = 1.0
    recognized = conv(best, compress)
    glyph_id = sl(recognized, 0, 1)
    count = sl(recognized, 1, 2)
    known = node(
        "relu",
        [op("min2", count, spatial_affine(count, -1.0, 2.0))],
    )

    # Separate the upper rules from the two lower strings.
    col0 = node("relu", [spatial_affine(col, -1.0, 1.0)])
    panel_rows = node("rowmax", [op("mul", color3, col0)])
    non_panel = op("sub", one_spatial, panel_rows)
    bottom_tile = op("mul", tile, panel_rows)
    top_tile = op("mul", tile, non_panel)

    inv_row = spatial_affine(row, -1.0, 64.0)
    first_bottom_y = vector_affine(
        gmax(op("mul", bottom_tile, inv_row)), -1.0, 64.0
    )
    last_bottom_y = gmax(op("mul", bottom_tile, row))
    input_tiles = op("mul", bottom_tile, spatial_equal(row, first_bottom_y))
    output_tiles = op("mul", bottom_tile, spatial_equal(row, last_bottom_y))

    inv_col = spatial_affine(col, -1.0, 64.0)
    input_left = vector_affine(
        gmax(op("mul", input_tiles, inv_col)), -1.0, 64.0
    )
    output_left = vector_affine(
        gmax(op("mul", output_tiles, inv_col)), -1.0, 64.0
    )

    # Find the left edge of a selector bar, and require its matching lower bar.
    bar_w = np.zeros((1, 1, 5, 5), dtype=np.float32)
    bar_w[0, 0, 2, :] = 1.0
    bar = node("relu", [conv(color0, bar_w, np.array([-4.5], np.float32))])
    bar = spatial_affine(bar, 2.0, 0.0)
    bar_start = node("relu", [op("sub", bar, shift(bar, dx=-1))])
    paired_bar = op("mul", bar_start, shift(bar_start, dy=12))
    marker = shift(paired_bar, dy=-6)
    selected_top = op("mul", top_tile, marker)
    selected_bottom = op("mul", output_tiles, marker)
    top_selected = gmax(selected_top)
    selected_top_y = gmax(op("mul", selected_top, row))
    selected_top_x = gmax(op("mul", selected_top, col))
    selected_bottom_x = gmax(op("mul", selected_bottom, col))

    # Anchor each rule at its last left-hand glyph.
    gap = shift(color3, dx=5)
    right1 = shift(top_tile, dx=10)
    right2 = shift(top_tile, dx=17)
    right3 = shift(top_tile, dx=24)
    left1 = shift(top_tile, dx=-7)
    left2 = op("mul", left1, shift(top_tile, dx=-14))
    anchor = op("mul", op("mul", top_tile, gap), right1)

    left_zero = op("sub", one_spatial, left1)
    left_one = op("sub", left1, left2)
    left_two = left2

    def left_fields(field):
        minus1 = shift(field, dx=-7)
        minus2 = shift(field, dx=-14)
        first = sum_spatial([
            op("mul", left_zero, field),
            op("mul", left_one, minus1),
            op("mul", left_two, minus2),
        ])
        second = sum_spatial([
            op("mul", left_one, field),
            op("mul", left_two, minus1),
        ])
        return first, second, field

    rule_h1 = left_fields(h1)
    rule_h2 = left_fields(h2)
    right_h1 = [shift(h1, dx=d) for d in (10, 17, 24)]
    right_h2 = [shift(h2, dx=d) for d in (10, 17, 24)]
    right_id = [shift(glyph_id, dx=d) for d in (10, 17, 24)]
    right_known = [shift(known, dx=d) for d in (10, 17, 24)]
    right_present = [right1, right2, right3]

    left_len = sum_spatial([one_spatial, left1, left2])
    right_len = sum_spatial([one_spatial, right2, right3])
    priority = conv(
        node("concat", [row, col]),
        np.array([[[[-64.0]], [[-1.0]]]], dtype=np.float32),
        np.array([4096.0], dtype=np.float32),
    )

    zero = dense(
        "avail",
        np.zeros((1, 8), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )

    # Ordinary mode: walk the input and translated output in parallel.
    loop_id = "translation"
    states = [
        {"id": "scan_in", "init": input_left},
        {"id": "scan_out", "init": output_left},
        {"id": "answer_h1", "init": zero},
        {"id": "answer_h2", "init": zero},
        {"id": "answer_id", "init": zero},
        {"id": "answer_known", "init": zero},
        {"id": "answer_found", "init": zero},
    ]
    body = []
    target = body

    queries1 = []
    queries2 = []
    query_valid = []
    for offset in (0, 7, 14):
        qx = vector_affine("scan_in", 1.0, float(offset))
        qmask = op("mul", input_tiles, spatial_equal(col, qx))
        query_valid.append(gmax(qmask))
        queries1.append(gmax(op("mul", qmask, h1)))
        queries2.append(gmax(op("mul", qmask, h2)))

    comparisons = []
    for i in range(3):
        e1 = spatial_equal(rule_h1[i], queries1[i])
        e2 = spatial_equal(rule_h2[i], queries2[i])
        comparisons.append(op("mul", op("mul", e1, e2), query_valid[i]))

    optional_second = op(
        "max2", op("sub", one_spatial, left1), comparisons[1]
    )
    optional_third = op(
        "max2", op("sub", one_spatial, left2), comparisons[2]
    )
    candidates = op(
        "mul",
        op("mul", op("mul", anchor, comparisons[0]), optional_second),
        optional_third,
    )
    best_priority = gmax(op("mul", candidates, priority))
    chosen = op("mul", candidates, spatial_equal(priority, best_priority))

    selected_offsets = []
    for i, offset in enumerate((0, 7, 14)):
        at_offset = vector_equal(selected_bottom_x, "scan_out", float(offset))
        exists = gmax(op("mul", chosen, right_present[i]))
        selected_offsets.append(vproduct(at_offset, exists))

    def picked(fields):
        pieces = []
        for i in range(3):
            value = gmax(op("mul", chosen, fields[i]))
            pieces.append(vproduct(selected_offsets[i], value))
        return sum_vectors(pieces)

    got_h1 = picked(right_h1)
    got_h2 = picked(right_h2)
    got_id = picked(right_id)
    got_known = picked(right_known)
    got_found = sum_vectors(selected_offsets)

    advance_in = vector_affine(gmax(op("mul", chosen, left_len)), 7.0, 0.0)
    advance_out = vector_affine(gmax(op("mul", chosen, right_len)), 7.0, 0.0)
    next_in = node("vadd", ["scan_in", advance_in])
    next_out = node("vadd", ["scan_out", advance_out])
    next_h1 = node("vadd", ["answer_h1", got_h1])
    next_h2 = node("vadd", ["answer_h2", got_h2])
    next_id = node("vadd", ["answer_id", got_id])
    next_known = node("vadd", ["answer_known", got_known])
    next_found = node("vadd", ["answer_found", got_found])

    target = graph
    graph.append({
        "id": loop_id,
        "op": "loop",
        "T": 8,
        "state": states,
        "body": body,
        "next": {
            "scan_in": next_in,
            "scan_out": next_out,
            "answer_h1": next_h1,
            "answer_h2": next_h2,
            "answer_id": next_id,
            "answer_known": next_known,
            "answer_found": next_found,
        },
    })

    current_h1 = gmax(op("mul", selected_bottom, h1))
    current_h2 = gmax(op("mul", selected_bottom, h2))
    current_id = gmax(op("mul", selected_bottom, glyph_id))
    current_known = gmax(op("mul", selected_bottom, known))

    desired_h1 = loop_id + ".answer_h1"
    desired_h2 = loop_id + ".answer_h2"
    desired_id = loop_id + ".answer_id"
    desired_known = loop_id + ".answer_known"
    found = loop_id + ".answer_found"

    equal = vproduct(
        vector_equal(current_h1, desired_h1),
        vector_equal(current_h2, desired_h2),
    )
    unequal = vector_affine(equal, -1.0, 1.0)
    forward = forward_direction(
        current_id, desired_id, current_known, desired_known
    )
    backward = vector_affine(forward, -1.0, 1.0)
    ordinary1 = vproduct(vproduct(found, unequal), backward)
    ordinary2 = vproduct(vproduct(found, unequal), forward)
    ordinary4 = node(
        "vadd",
        [vproduct(found, equal), vector_affine(found, -1.0, 1.0)],
    )

    # Altered mode: align each rule group, in reading order, with the
    # corresponding segment of the fixed lower strings.
    first_left_x = op(
        "sub",
        col,
        spatial_affine(op("add", left1, left2), 7.0, 0.0),
    )
    first_right_x = spatial_affine(col, 1.0, 10.0)
    same_selected_row = spatial_equal(row, selected_top_y)
    left_match = op(
        "mul",
        op("mul", anchor, same_selected_row),
        spatial_equal(first_left_x, selected_top_x),
    )
    right_match = op(
        "mul",
        op("mul", anchor, same_selected_row),
        spatial_equal(first_right_x, selected_top_x),
    )
    chosen_rule = op("max2", left_match, right_match)
    selected_is_left = gmax(left_match)
    selected_is_right = gmax(right_match)
    rule_found = gmax(chosen_rule)
    selected_priority = gmax(op("mul", chosen_rule, priority))

    earlier = node("relu", [op("sub", priority, selected_priority)])
    earlier = op("min2", earlier, one_spatial)
    earlier = op("mul", earlier, anchor)
    preceding_left = gsum(op("mul", earlier, left_len))
    preceding_right = gsum(op("mul", earlier, right_len))
    desired_left_x = node(
        "vadd", [input_left, vector_affine(preceding_left, 7.0, 0.0)]
    )
    desired_right_x = node(
        "vadd", [output_left, vector_affine(preceding_right, 7.0, 0.0)]
    )
    desired_left_mask = op(
        "mul",
        op("mul", input_tiles, spatial_equal(col, desired_left_x)),
        selected_is_left,
    )
    desired_right_mask = op(
        "mul",
        op("mul", output_tiles, spatial_equal(col, desired_right_x)),
        selected_is_right,
    )
    desired_mask = op("max2", desired_left_mask, desired_right_mask)

    alt_current_h1 = gmax(op("mul", selected_top, h1))
    alt_current_h2 = gmax(op("mul", selected_top, h2))
    alt_current_id = gmax(op("mul", selected_top, glyph_id))
    alt_current_known = gmax(op("mul", selected_top, known))
    alt_desired_h1 = gmax(op("mul", desired_mask, h1))
    alt_desired_h2 = gmax(op("mul", desired_mask, h2))
    alt_desired_id = gmax(op("mul", desired_mask, glyph_id))
    alt_desired_known = gmax(op("mul", desired_mask, known))
    alt_valid = vproduct(rule_found, gmax(desired_mask))

    alt_equal = vproduct(
        vector_equal(alt_current_h1, alt_desired_h1),
        vector_equal(alt_current_h2, alt_desired_h2),
    )
    alt_unequal = vector_affine(alt_equal, -1.0, 1.0)
    alt_forward = forward_direction(
        alt_current_id,
        alt_desired_id,
        alt_current_known,
        alt_desired_known,
    )
    alt_backward = vector_affine(alt_forward, -1.0, 1.0)
    altered1 = vproduct(vproduct(alt_valid, alt_unequal), alt_backward)
    altered2 = vproduct(vproduct(alt_valid, alt_unequal), alt_forward)
    altered4 = node(
        "vadd",
        [vproduct(alt_valid, alt_equal), vector_affine(alt_valid, -1.0, 1.0)],
    )

    not_top = vector_affine(top_selected, -1.0, 1.0)

    def choose_mode(ordinary, altered):
        return node(
            "vadd",
            [vproduct(not_top, ordinary), vproduct(top_selected, altered)],
        )

    score1 = choose_mode(ordinary1, altered1)
    score2 = choose_mode(ordinary2, altered2)
    score4 = choose_mode(ordinary4, altered4)

    scores = node("vconcat", [score1, score2, score4])
    logits_w = np.zeros((8, 3), dtype=np.float32)
    logits_w[1, 0] = 1.0
    logits_w[2, 1] = 1.0
    logits_w[4, 2] = 1.0
    logits_b = np.array(
        [-100, 0, 0, -100, 0, -100, -100, -100], dtype=np.float32
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [scores],
        "w": parameter(logits_w),
        "b": parameter(logits_b),
    })
    graph.append({
        "id": "click",
        "op": "slice",
        "in": ["x0"],
        "start": 0,
        "stop": 1,
    })

    return {"graph": graph, "memory_size": 0, "params": params}

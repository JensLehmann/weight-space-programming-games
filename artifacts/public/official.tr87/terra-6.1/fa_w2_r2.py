import numpy as np


def build():
    glyphs = [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],
        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],
        ["10101", "10000", "10101", "10000", "11111"],
        ["11101", "10000", "00000", "00001", "10111"],
        ["10111", "00100", "00100", "00100", "11101"],
        ["00100", "10101", "00100", "10101", "00100"],
        ["11011", "10001", "10101", "10001", "11011"],
        ["10111", "10101", "11101", "00000", "10101"],
        ["10001", "00000", "10001", "11011", "01010"],
    ]

    params = {}
    graph = []
    target = graph
    counter = [0]
    affine_cache = {}
    shift_cache = {}

    def add(op, inputs, **fields):
        name = "n%d" % counter[0]
        counter[0] += 1
        target.append({"id": name, "op": op, "in": inputs, **fields})
        return name

    def sl(x, a, b):
        return add("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return add("vslice", [x], start=a, stop=b)

    def binary(op, a, b):
        return add(op, [a, b])

    def mul(a, b):
        return binary("mul", a, b)

    def sub(a, b):
        return binary("sub", a, b)

    def plus(a, b):
        return binary("add", a, b)

    def vmax(x):
        return add("gmax", [x])

    def vmul(a, b):
        return add("vmul", [a, b])

    def vplus(a, b):
        return add("vadd", [a, b])

    def affine(x, scale, bias):
        key = (float(scale), float(bias))
        if key not in affine_cache:
            i = len(affine_cache)
            w, b = "aff_w_%d" % i, "aff_b_%d" % i
            params[w] = np.array([[[[scale]]]], dtype=np.float32)
            params[b] = np.array([bias], dtype=np.float32)
            affine_cache[key] = (w, b)
        w, b = affine_cache[key]
        return add("conv", [x], w=w, b=b)

    def clip01(x):
        return binary("min2", x, ones)

    def present(x):
        return clip01(add("relu", [x]))

    def spatial_eq(a, b, scale=2.0):
        d = add("abs", [sub(a, b)])
        return clip01(add("relu", [affine(d, -scale, 2.0)]))

    def rank_eq(a, b):
        d = add("abs", [sub(a, b)])
        return clip01(add("relu", [affine(d, -8192.0, 1.0)]))

    def shift_unit(x, axis, offset):
        key = (axis, int(offset))
        if key not in shift_cache:
            name = "shift_%s_%s%d" % (
                axis, "p" if offset >= 0 else "m", abs(offset)
            )
            w = np.zeros((1, 1, 7, 7), dtype=np.float32)
            if axis == "x":
                w[0, 0, 3, 3 + offset] = 1.0
            else:
                w[0, 0, 3 + offset, 3] = 1.0
            params[name] = w
            shift_cache[key] = name
        return add("conv", [x], w=shift_cache[key])

    def shift(x, axis, distance):
        while distance:
            d = min(3, abs(distance))
            if distance < 0:
                d = -d
            x = shift_unit(x, axis, d)
            distance -= d
        return x

    def select_score(candidate):
        weighted = mul(candidate, priority)
        best = vmax(weighted)
        return mul(candidate, rank_eq(priority, best))

    def length_flags(chosen, second, third):
        p0 = vmax(chosen)
        p1 = vmax(mul(chosen, present(second)))
        p2 = vmax(mul(chosen, present(third)))
        values = add("vconcat", [p0, p1, p2])
        f = add("dense", [values], w="length_w")
        return [vsl(f, i, i + 1) for i in range(3)]

    def shifted_pointers(p):
        p7 = shift(p, "x", -7)
        p14 = shift(p7, "x", -7)
        p21 = shift(p14, "x", -7)
        return p7, p14, p21

    def advance(shifts, flags):
        return plus(
            plus(mul(shifts[0], flags[0]), mul(shifts[1], flags[1])),
            mul(shifts[2], flags[2]),
        )

    black = sl("x0", 5, 6)
    ca = sl("x0", 10, 11)
    cb = sl("x0", 7, 8)
    cc = sl("x0", 11, 12)
    template_input = add("concat", [black, ca, cb, cc])
    ones = sl("coords", 2, 3)
    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one_v = vmax(ones)

    match_w = np.zeros((84, 4, 7, 7), dtype=np.float32)
    for g, rows in enumerate(glyphs):
        mask = np.array(
            [[int(c) for c in r] for r in rows], dtype=np.int8
        )
        family_channel = 1 + g // 7
        for rotation in range(4):
            rotated = np.rot90(mask, rotation)
            colors = np.full((7, 7), family_channel, dtype=np.int8)
            colors[1:6, 1:6] = np.where(
                rotated == 1, 0, family_channel
            )
            for y in range(7):
                for x in range(7):
                    match_w[
                        4 * g + rotation, colors[y, x], y, x
                    ] = 2.0

    params["match_w"] = match_w
    params["match_b"] = np.full(84, -97.0, dtype=np.float32)
    raw = add("conv", [template_input], w="match_w", b="match_b")
    raw = add("relu", [raw])

    merge_w = np.zeros((21, 84, 1, 1), dtype=np.float32)
    for g in range(21):
        merge_w[g, 4 * g:4 * g + 4, 0, 0] = 1.0
    params["merge_w"] = merge_w
    hits = add("conv", [raw], w="merge_w")
    hits = binary("min2", hits, ones)

    params["id_w"] = np.arange(1, 22, dtype=np.float32).reshape(
        1, 21, 1, 1
    )
    params["count_w"] = np.ones((1, 21, 1, 1), dtype=np.float32)
    ids = add("conv", [hits], w="id_w")
    hit_count = add("conv", [hits], w="count_w")
    hit_map = present(hit_count)

    background2 = sl("x0", 2, 3)
    rule_rows = add("rowmax", [background2])
    lower_rows = sub(ones, rule_rows)
    rule_ids = mul(ids, rule_rows)
    lower_hits = mul(hit_map, lower_rows)

    inv_row = sub(ones, row)
    top_inv = vmax(mul(lower_hits, inv_row))
    bottom_y = vmax(mul(lower_hits, row))

    params["one_minus_w"] = np.array([[-1.0]], dtype=np.float32)
    params["one_minus_b"] = np.array([1.0], dtype=np.float32)
    top_y = add(
        "dense", [top_inv], w="one_minus_w", b="one_minus_b"
    )

    top_row_mask = spatial_eq(row, top_y, 100.0)
    bottom_row_mask = spatial_eq(row, bottom_y, 100.0)
    top_hits = mul(lower_hits, top_row_mask)
    bottom_hits = mul(lower_hits, bottom_row_mask)
    top_ids = mul(ids, top_hits)
    bottom_ids = mul(ids, bottom_hits)

    inv_col = sub(ones, col)
    top_left_inv = vmax(mul(top_hits, inv_col))
    bottom_left_inv = vmax(mul(bottom_hits, inv_col))
    top_left = add(
        "dense", [top_left_inv], w="one_minus_w", b="one_minus_b"
    )
    bottom_left = add(
        "dense", [bottom_left_inv], w="one_minus_w", b="one_minus_b"
    )
    top_ptr = mul(top_hits, spatial_eq(col, top_left, 100.0))
    bottom_ptr = mul(bottom_hits, spatial_eq(col, bottom_left, 100.0))

    zero = sl("x0", 0, 1)
    cursor_cols = add("colmax", [zero])
    selected_bottom = mul(bottom_hits, cursor_cols)
    selected_bottom_id = vmax(mul(bottom_ids, selected_bottom))
    edit_mode = vmax(mul(zero, rule_rows))

    # Align rule glyphs to the center of each connecting gap.
    left0 = shift(rule_ids, "x", -5)
    left1 = shift(left0, "x", -7)
    left2 = shift(left1, "x", -7)
    right0 = shift(rule_ids, "x", 5)
    right1 = shift(right0, "x", 7)
    right2 = shift(right1, "x", 7)
    color3 = sl("x0", 3, 4)
    anchor = mul(mul(present(left0), present(right0)), color3)

    params["priority_w"] = np.array(
        [[[[-1.0]], [[-1.0 / 64.0]], [[0.0]]]],
        dtype=np.float32,
    )
    params["priority_b"] = np.array([2.0], dtype=np.float32)
    priority = add(
        "conv", ["coords"], w="priority_w", b="priority_b"
    )

    params["length_w"] = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    top_next1 = shift(top_ids, "x", 7)
    top_next2 = shift(top_next1, "x", 7)

    # Follow normal translation rules, carrying pointers in both strings.
    normal_body = []
    target = normal_body

    in0 = vmax(mul(top_ids, "normal_in"))
    in1 = vmax(mul(top_next1, "normal_in"))
    in2 = vmax(mul(top_next2, "normal_in"))

    no_left1 = sub(ones, present(left1))
    no_left2 = sub(ones, present(left2))
    m1 = mul(spatial_eq(left0, in0), no_left1)
    m2 = mul(
        mul(spatial_eq(left1, in0), spatial_eq(left0, in1)),
        no_left2,
    )
    m3 = mul(
        mul(spatial_eq(left2, in0), spatial_eq(left1, in1)),
        spatial_eq(left0, in2),
    )
    m1 = mul(m1, anchor)
    m2 = mul(m2, anchor)
    m3 = mul(m3, anchor)
    candidates = plus(plus(m1, m2), m3)
    chosen = select_score(candidates)

    input_flags = [
        vmax(mul(chosen, m1)),
        vmax(mul(chosen, m2)),
        vmax(mul(chosen, m3)),
    ]
    output_flags = length_flags(chosen, right1, right2)

    rhs0 = vmax(mul(chosen, right0))
    rhs1 = vmax(mul(chosen, right1))
    rhs2 = vmax(mul(chosen, right2))

    input_shifts = shifted_pointers("normal_in")
    output_shifts = shifted_pointers("normal_out")
    next_input = advance(input_shifts, input_flags)
    next_output = advance(output_shifts, output_flags)

    at0 = vmax(mul("normal_out", selected_bottom))
    at1 = vmax(mul(output_shifts[0], selected_bottom))
    at2 = vmax(mul(output_shifts[1], selected_bottom))
    found = vplus(
        vplus(vmul(at0, rhs0), vmul(at1, rhs1)),
        vmul(at2, rhs2),
    )
    next_expected = vplus("normal_expected", found)

    target = graph
    graph.append({
        "id": "Normal",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "normal_in", "init": top_ptr},
            {"id": "normal_out", "init": bottom_ptr},
            {"id": "normal_expected", "zeros": [1]},
        ],
        "body": normal_body,
        "next": {
            "normal_in": next_input,
            "normal_out": next_output,
            "normal_expected": next_expected,
        },
    })

    # A bracket must be present on both sides of the same glyph row.
    # This avoids confusing brackets on vertically adjacent rule rows.
    cursor_above = binary(
        "max2", shift(zero, "y", -5), shift(zero, "y", -6)
    )
    cursor_below = binary(
        "max2", shift(zero, "y", 5), shift(zero, "y", 6)
    )
    cursor_rule = mul(cursor_above, cursor_below)
    cursor_left = shift(cursor_rule, "x", -5)
    cursor_right = shift(cursor_rule, "x", 5)

    params["three_w"] = np.array([[3.0]], dtype=np.float32)
    initial_priority = add("dense", [one_v], w="three_w")

    # Editable-rule heuristic: align rules in reading order to segments
    # of the two lower strings.
    edit_body = []
    target = edit_body

    gap = sub("edit_prev_priority", priority)
    eligible = clip01(
        add("relu", [affine(gap, 8192.0, -0.5)])
    )
    edit_candidates = mul(anchor, eligible)
    edit_chosen = select_score(edit_candidates)
    edit_next_priority = vmax(mul(edit_chosen, priority))

    edit_input_flags = length_flags(edit_chosen, left1, left2)
    edit_output_flags = length_flags(edit_chosen, right1, right2)

    edit_input_value = vmax(mul(top_ids, "edit_in"))
    edit_output_value = vmax(mul(bottom_ids, "edit_out"))

    first_left = plus(
        plus(
            mul(left0, edit_input_flags[0]),
            mul(left1, edit_input_flags[1]),
        ),
        mul(left2, edit_input_flags[2]),
    )
    selected_left_rule = mul(edit_chosen, cursor_left)
    selected_right_rule = mul(edit_chosen, cursor_right)
    left_selected = vmax(selected_left_rule)
    right_selected = vmax(selected_right_rule)

    left_current = vmax(mul(selected_left_rule, first_left))
    right_current = vmax(mul(selected_right_rule, right0))
    current_piece = vplus(left_current, right_current)
    expected_piece = vplus(
        vmul(left_selected, edit_input_value),
        vmul(right_selected, edit_output_value),
    )
    edit_next_current = vplus("edit_current", current_piece)
    edit_next_expected = vplus("edit_expected", expected_piece)

    edit_input_shifts = shifted_pointers("edit_in")
    edit_output_shifts = shifted_pointers("edit_out")
    edit_next_input = advance(edit_input_shifts, edit_input_flags)
    edit_next_output = advance(edit_output_shifts, edit_output_flags)

    target = graph
    graph.append({
        "id": "Edit",
        "op": "loop",
        "T": 12,
        "state": [
            {"id": "edit_in", "init": top_ptr},
            {"id": "edit_out", "init": bottom_ptr},
            {"id": "edit_prev_priority", "init": initial_priority},
            {"id": "edit_current", "zeros": [1]},
            {"id": "edit_expected", "zeros": [1]},
        ],
        "body": edit_body,
        "next": {
            "edit_in": edit_next_input,
            "edit_out": edit_next_output,
            "edit_prev_priority": edit_next_priority,
            "edit_current": edit_next_current,
            "edit_expected": edit_next_expected,
        },
    })

    normal_mode = add(
        "dense", [edit_mode], w="one_minus_w", b="one_minus_b"
    )
    current = vplus(
        vmul(normal_mode, selected_bottom_id),
        vmul(edit_mode, "Edit.edit_current"),
    )
    expected = vplus(
        vmul(normal_mode, "Normal.normal_expected"),
        vmul(edit_mode, "Edit.edit_expected"),
    )
    values = add("vconcat", [current, expected])

    # Convert numeric glyph IDs into one-hot vectors.
    hat_w = np.zeros((126, 2), dtype=np.float32)
    hat_b = np.zeros(126, dtype=np.float32)
    for which in range(2):
        for g in range(21):
            t = g + 1
            for k, threshold in enumerate((t - 1, t, t + 1)):
                j = which * 63 + g * 3 + k
                hat_w[j, which] = 1.0
                hat_b[j] = -float(threshold)
    params["hat_w"] = hat_w
    params["hat_b"] = hat_b
    hinges = add("dense", [values], w="hat_w", b="hat_b")
    hinges = add("vrelu", [hinges])

    hot_w = np.zeros((42, 126), dtype=np.float32)
    for which in range(2):
        for g in range(21):
            j = which * 63 + g * 3
            hot_w[which * 21 + g, j:j + 3] = [1.0, -2.0, 1.0]
    params["hot_w"] = hot_w
    hot = add("dense", [hinges], w="hot_w")

    pair_w = np.zeros((441, 42), dtype=np.float32)
    for c in range(21):
        for e in range(21):
            p = c * 21 + e
            pair_w[p, c] = 1.0
            pair_w[p, 21 + e] = 1.0
    params["pair_w"] = pair_w
    params["pair_b"] = np.full(441, -1.5, dtype=np.float32)
    pairs = add("dense", [hot], w="pair_w", b="pair_b")
    pairs = add("vrelu", [pairs])

    # Select the shorter direction around the seven-glyph cycle.
    action_w = np.zeros((8, 441), dtype=np.float32)
    for c in range(21):
        for e in range(21):
            if c // 7 != e // 7:
                continue
            delta = (e % 7 - c % 7) % 7
            p = c * 21 + e
            if 1 <= delta <= 3:
                action_w[2, p] = 2.0
            elif 4 <= delta <= 6:
                action_w[1, p] = 2.0
    params["action_w"] = action_w
    params["action_b"] = np.array(
        [-1.0, 0.0, 0.0, -1.0, 0.1, -1.0, -1.0, -1.0],
        dtype=np.float32,
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [pairs],
        "w": "action_w",
        "b": "action_b",
    })
    graph.append({
        "id": "click",
        "op": "slice",
        "in": ["x0"],
        "start": 0,
        "stop": 1,
    })

    return {"graph": graph, "memory_size": 0, "params": params}

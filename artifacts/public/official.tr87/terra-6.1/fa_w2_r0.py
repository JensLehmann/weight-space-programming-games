import numpy as np


def build():
    graph = []
    params = {}
    active = [graph]
    serial = [0]
    shift_weights = {}
    shift_nodes = {}

    def fresh(prefix="n"):
        serial[0] += 1
        return "%s_%d" % (prefix, serial[0])

    def param(value):
        p = fresh("p")
        params[p] = np.asarray(value, dtype=np.float32)
        return p

    def node(kind, inputs, **fields):
        n = fresh()
        d = {"id": n, "op": kind, "in": list(inputs)}
        d.update(fields)
        active[0].append(d)
        return n

    def conv(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("conv", [x], **fields)

    def dense(x, w, b=None):
        fields = {"w": param(w)}
        if b is not None:
            fields["b"] = param(b)
        return node("dense", [x], **fields)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def mul(a, b):
        return node("mul", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mx(a, b):
        return node("max2", [a, b])

    def relu(x):
        return node("relu", [x])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vr(x):
        return node("vrelu", [x])

    def vscale(x, scale, bias=0.0):
        return dense(
            x,
            np.array([[scale]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def vsub(a, b):
        return va(a, vscale(b, -1.0))

    def vone_minus(x):
        return vscale(x, -1.0, 1.0)

    def vclamp(x):
        y = vr(x)
        return vsub(y, vr(vscale(y, 1.0, -1.0)))

    def vhard(x):
        y = vscale(x, 100.0, -50.0)
        return vclamp(y)

    def vmax(a, b):
        return va(a, vr(vsub(b, a)))

    def vsum(xs):
        result = xs[0]
        for x in xs[1:]:
            result = va(result, x)
        return result

    def smax(xs):
        result = xs[0]
        for x in xs[1:]:
            result = mx(result, x)
        return result

    def spatial_linear(x, scale, bias=0.0):
        return conv(
            x,
            np.array([[[[scale]]]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
        )

    def one_minus(x):
        return spatial_linear(x, -1.0, 1.0)

    def hard(x):
        a = relu(spatial_linear(x, 100.0, -50.0))
        b = relu(spatial_linear(x, 100.0, -51.0))
        return sub(a, b)

    def sum_channels(x, count):
        return conv(x, np.ones((1, count, 1, 1), dtype=np.float32))

    def shift_weight(dy, dx):
        key = (dy, dx)
        if key not in shift_weights:
            w = np.zeros((1, 1, 7, 7), dtype=np.float32)
            w[0, 0, 3 - dy, 3 - dx] = 1.0
            shift_weights[key] = param(w)
        return shift_weights[key]

    def shift(x, dy, dx):
        key = (id(active[0]), x, dy, dx)
        if key in shift_nodes:
            return shift_nodes[key]
        y = x
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            y = node("conv", [y], w=shift_weight(sy, sx))
            dy -= sy
            dx -= sx
        shift_nodes[key] = y
        return y

    def at(x, dx):
        return shift(x, 0, -dx)

    def eq_spatial_scalar(x, scalar, scale=126.0):
        difference = node("abs", [sub(x, scalar)])
        return relu(spatial_linear(difference, -scale, 1.0))

    def eq_scalars(a, b):
        d = vsub(a, b)
        absolute = va(vr(d), vr(vscale(d, -1.0)))
        return vr(vscale(absolute, -126.0, 1.0))

    # Locate the 7x7 tile frames.
    tile_w = np.zeros((1, 16, 1, 1), dtype=np.float32)
    for colour in range(6, 16):
        tile_w[0, colour, 0, 0] = 1.0
    tile = conv("x0", tile_w)
    edge = relu(sub(tile, shift(tile, 1, 0)))
    run_start = relu(sub(edge, shift(edge, 0, 1)))

    seven_w = np.zeros((1, 1, 7, 7), dtype=np.float32)
    seven_w[0, 0, 3, :] = 1.0
    seven = shift(conv(edge, seven_w), 0, -3)
    run_start = node(
        "min2",
        [run_start, relu(spatial_linear(seven, 1.0, -6.0))],
    )

    spread_id = fresh("spread")
    spread_state = fresh("state")
    b1, b2, b3, b4, b5 = [fresh("body") for _ in range(5)]
    graph.append({
        "id": spread_id,
        "op": "loop",
        "T": 9,
        "state": [{"id": spread_state, "init": run_start}],
        "body": [
            {"id": b1, "op": "conv", "in": [spread_state],
             "w": shift_weight(0, 3)},
            {"id": b2, "op": "conv", "in": [b1],
             "w": shift_weight(0, 3)},
            {"id": b3, "op": "conv", "in": [b2],
             "w": shift_weight(0, 1)},
            {"id": b4, "op": "mul", "in": [b3, edge]},
            {"id": b5, "op": "max2", "in": [spread_state, b4]},
        ],
        "next": {spread_state: b5},
    })
    anchors = shift(spread_id + "." + spread_state, 3, 3)

    row = sl("coords", 0, 1)
    col = sl("coords", 1, 2)
    one = sl("coords", 2, 3)
    invrow = sub(one, row)
    invcol = sub(one, col)
    colour3 = sl("x0", 3, 4)
    colour0 = sl("x0", 0, 1)
    black = sl("x0", 5, 6)

    # The message panel has colour 3 at the left edge.
    first_col = relu(spatial_linear(col, -63.0, 1.0))
    panel = node("rowmax", [mul(colour3, first_col)])
    bottom = mul(anchors, panel)
    upper = mul(anchors, one_minus(panel))

    first_bottom_row = node("gmax", [mul(bottom, invrow)])
    last_bottom_row = node("gmax", [mul(bottom, row)])
    input_anchors = hard(
        mul(bottom, eq_spatial_scalar(invrow, first_bottom_row))
    )
    output_anchors = hard(
        mul(bottom, eq_spatial_scalar(row, last_bottom_row))
    )

    input_first_inv = node("gmax", [mul(input_anchors, invcol)])
    output_first_inv = node("gmax", [mul(output_anchors, invcol)])
    input_first = vone_minus(input_first_inv)
    output_first = vone_minus(output_first_inv)

    # Cursor brackets surround either an output tile or an editable rule.
    cursor_bars = mul(shift(colour0, 6, 0), shift(colour0, -6, 0))
    selected_output = mul(output_anchors, cursor_bars)
    group_first = mul(upper, one_minus(at(upper, -7)))
    selected_upper = mul(group_first, cursor_bars)
    selected = mx(selected_output, selected_upper)
    selected_output_col = node("gmax", [mul(selected_output, col)])
    selected_output_exists = node("gmax", [selected_output])
    altered = node("gmax", [selected_upper])
    ordinary = vone_minus(altered)

    # Extract each glyph in four orientations.
    indices = np.arange(25).reshape(5, 5)
    features = []
    for k in range(4):
        w = np.zeros((25, 1, 5, 5), dtype=np.float32)
        for j, src in enumerate(np.rot90(indices, k=k).reshape(-1)):
            yy, xx = divmod(int(src), 5)
            w[j, 0, yy, xx] = 1.0
        features.append(conv(black, w))

    # Locate rules with one, two, or three tiles on either side.
    rules = {}
    for k in (1, 2, 3):
        left = group_first
        left = mul(left, at(upper, 7 * (k - 1)))
        left = mul(left, one_minus(at(upper, 7 * k)))
        left = mul(left, at(colour3, 7 * (k - 1) + 4))
        right_offset = 7 * (k - 1) + 10
        left = mul(left, at(upper, right_offset))
        for r in (1, 2, 3):
            m = mul(left, at(upper, right_offset + 7 * (r - 1)))
            m = mul(m, one_minus(at(upper, right_offset + 7 * r)))
            rules[(k, r)] = m

    priority_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    priority_w[0, 0, 0, 0] = -(63.0 * 64.0) / 4096.0
    priority_w[0, 1, 0, 0] = -63.0 / 4096.0
    priority = conv(
        "coords", priority_w, np.array([1.0], dtype=np.float32)
    )

    # Walk input and output positions in parallel.
    loop_id = fresh("parse")
    p_state = fresh("position")
    q_state = fresh("position")
    used_state = fresh("used")
    found_state = fresh("found")
    goal_state = fresh("goal")
    body = []
    active[0] = body

    p_positions = [
        hard(mul(
            input_anchors,
            eq_spatial_scalar(col, vscale(p_state, 1.0, j / 9.0)),
        ))
        for j in range(3)
    ]
    q_positions = [
        hard(mul(
            output_anchors,
            eq_spatial_scalar(col, vscale(q_state, 1.0, j / 9.0)),
        ))
        for j in range(3)
    ]

    def glyph_match(query_position):
        query = node("gmax", [mul(features[0], query_position)])
        exists = node("gmax", [query_position])
        matches = []
        for f in features:
            difference = node("abs", [sub(f, query)])
            distance = sum_channels(difference, 25)
            matches.append(relu(spatial_linear(distance, -1.0, 1.0)))
        return mul(smax(matches), exists)

    input_matches = [glyph_match(p) for p in p_positions]
    output_match = glyph_match(q_positions[0])

    def frame_colour_match(position):
        frame_position = shift(position, -3, 0)
        colour = node("gmax", [mul("x0", frame_position)])
        agreement = sum_channels(mul("x0", colour), 16)
        return shift(agreement, 3, 0)

    input_colour_match = frame_colour_match(p_positions[0])
    output_colour_match = frame_colour_match(q_positions[0])

    # For editable rules, use equal-glyph run lengths as an assignment clue.
    pi1 = node("gmax", [mul(input_matches[0], p_positions[1])])
    pi2 = node("gmax", [mul(input_matches[0], p_positions[2])])
    qo1 = node("gmax", [mul(output_match, q_positions[1])])
    qo2 = node("gmax", [mul(output_match, q_positions[2])])
    input_lengths = {
        1: vone_minus(pi1),
        2: vm(pi1, vone_minus(pi2)),
        3: vm(pi1, pi2),
    }
    output_lengths = {
        1: vone_minus(qo1),
        2: vm(qo1, vone_minus(qo2)),
        3: vm(qo1, qo2),
    }
    p_exists = node("gmax", [p_positions[0]])
    q_exists = node("gmax", [q_positions[0]])
    alt_valid = vm(altered, vm(p_exists, q_exists))

    free = one_minus(mul(used_state, altered))
    candidates = []
    for k in (1, 2, 3):
        normal_match = input_matches[0]
        for j in range(1, k):
            normal_match = mul(normal_match, at(input_matches[j], 7 * j))
        normal_match = mul(normal_match, input_colour_match)
        for r in (1, 2, 3):
            offset = 7 * (k - 1) + 10
            editable_match = mul(
                input_colour_match, at(output_colour_match, offset)
            )
            editable_match = mul(
                editable_match,
                vm(alt_valid, vm(input_lengths[k], output_lengths[r])),
            )
            eligible = mx(mul(normal_match, ordinary), editable_match)
            candidates.append(mul(rules[(k, r)], mul(eligible, free)))

    # Convert near-binary matches back to exact binary values before
    # comparing priorities and updating the carried positions.
    candidates = hard(smax(candidates))
    best = node("gmax", [mul(candidates, priority)])
    chosen = hard(
        mul(candidates, eq_spatial_scalar(priority, best, 8192.0))
    )

    chosen_kr = {
        (k, r): mul(chosen, rules[(k, r)])
        for k in (1, 2, 3)
        for r in (1, 2, 3)
    }
    chosen_k = {
        k: smax([chosen_kr[(k, r)] for r in (1, 2, 3)])
        for k in (1, 2, 3)
    }
    chosen_r = {
        r: smax([chosen_kr[(k, r)] for k in (1, 2, 3)])
        for r in (1, 2, 3)
    }

    normal_targets = []
    for j in range(3):
        cursor_j = vm(
            selected_output_exists,
            vhard(eq_scalars(
                selected_output_col, vscale(q_state, 1.0, j / 9.0)
            )),
        )
        for k in (1, 2, 3):
            eligible = smax([
                chosen_kr[(k, r)] for r in (1, 2, 3) if r > j
            ])
            offset = 7 * (k - 1) + 10 + 7 * j
            normal_targets.append(mul(shift(eligible, 0, offset), cursor_j))
    normal_target = mul(smax(normal_targets), ordinary)

    left_selected = []
    right_selected = []
    for k in (1, 2, 3):
        left_selected.append(
            node("gmax", [mul(chosen_k[k], selected_upper)])
        )
        offset = 7 * (k - 1) + 10
        right_selected.append(
            node("gmax", [
                mul(shift(chosen_k[k], 0, offset), selected_upper)
            ])
        )
    left_flag = vm(altered, vsum(left_selected))
    right_flag = vm(altered, vsum(right_selected))
    editable_target = mx(
        mul(p_positions[0], left_flag),
        mul(q_positions[0], right_flag),
    )
    target = hard(mx(normal_target, editable_target))
    target_exists = node("gmax", [target])
    target_pixels = node("gmax", [mul(features[0], target)])
    not_found = vone_minus(found_state)
    next_goal = va(goal_state, vm(target_pixels, not_found))
    next_found = va(found_state, vm(target_exists, not_found))

    p_increment = vsum([
        vscale(node("gmax", [chosen_k[k]]), k / 9.0)
        for k in (1, 2, 3)
    ])
    q_increment = vsum([
        vscale(node("gmax", [chosen_r[r]]), r / 9.0)
        for r in (1, 2, 3)
    ])
    next_p = va(p_state, p_increment)
    next_q = va(q_state, q_increment)
    next_used = mx(used_state, mul(chosen, altered))

    active[0] = graph
    graph.append({
        "id": loop_id,
        "op": "loop",
        "T": 10,
        "state": [
            {"id": p_state, "init": input_first},
            {"id": q_state, "init": output_first},
            {"id": used_state, "zeros": [1, 64, 64]},
            {"id": found_state, "zeros": [1]},
            {"id": goal_state, "zeros": [25]},
        ],
        "body": body,
        "next": {
            p_state: next_p,
            q_state: next_q,
            used_state: next_used,
            found_state: next_found,
            goal_state: next_goal,
        },
    })

    goal = loop_id + "." + goal_state
    found = loop_id + "." + found_state
    current_orientations = [
        node("gmax", [mul(f, selected)]) for f in features
    ]

    # Exact rotation-invariant equality.
    negative_identity_25 = param(-np.eye(25, dtype=np.float32))
    negative_goal = node("dense", [goal], w=negative_identity_25)
    equality_scores = []
    for current in current_orientations:
        difference = va(current, negative_goal)
        negative_difference = node(
            "dense", [difference], w=negative_identity_25
        )
        absolute = va(vr(difference), vr(negative_difference))
        distance = dense(
            absolute, np.ones((1, 25), dtype=np.float32)
        )
        equality_scores.append(vr(vscale(distance, -1.0, 1.0)))
    equal = equality_scores[0]
    for e in equality_scores[1:]:
        equal = vmax(equal, e)
    equal = vhard(equal)

    # Known glyph families permit choosing the shorter cycling direction.
    # Unknown or ambiguous glyphs use the forward direction.
    templates = [
        [
            "..5../..5../.555./..5../55555",
            "....5/..5.5/55555/5.5../5....",
            "55555/5.5../5..../5.5../55555",
            "..5../55555/5.5.5/5.5.5/..5..",
            "5...5/55555/..5../55555/5...5",
            "55555/..5../555../5..../55555",
            "5...5/55555/5...5/5...5/55.55",
        ],
        [
            "5..../5555./5..5./5..5./55555",
            "55555/5...5/555.5/5.5.5/55555",
            "..555/..5.5/55555/5.5../555..",
            ".555./.5.5./55555/5...5/55555",
            "55555/5...5/5...5/55555/..5..",
            "5555./5..55/5...5/55..5/.5555",
            "..5../55555/5.5.5/55555/..5..",
        ],
        [
            "5.5.5/5..../5.5.5/5..../55555",
            "555.5/5..../...../....5/5.555",
            "5.555/..5../..5../..5../555.5",
            "..5../5.5.5/..5../5.5.5/..5..",
            "55.55/5...5/5.5.5/5...5/55.55",
            "5.555/5.5.5/555.5/...../5.5.5",
            "5...5/...../5...5/55.55/.5.5.",
        ],
    ]
    classifier_w = np.zeros((84, 25), dtype=np.float32)
    classifier_b = np.zeros(84, dtype=np.float32)
    for group in range(3):
        for index in range(7):
            base = np.array([
                [1.0 if c == "5" else 0.0 for c in line]
                for line in templates[group][index].split("/")
            ], dtype=np.float32)
            for rotation in range(4):
                i = (group * 7 + index) * 4 + rotation
                pattern = np.rot90(base, k=rotation).reshape(-1)
                classifier_w[i] = 2.0 * pattern - 1.0
                classifier_b[i] = -float(pattern.sum()) + 0.5
    classifier_wp = param(classifier_w)
    classifier_bp = param(classifier_b)

    group_w = np.zeros((21, 84), dtype=np.float32)
    for i in range(21):
        group_w[i, 4 * i:4 * i + 4] = 2.0
    group_wp = param(group_w)
    neg_eye_21 = param(-np.eye(21, dtype=np.float32))
    identity_21 = param(np.eye(21, dtype=np.float32))
    minus_ones_21 = param(-np.ones(21, dtype=np.float32))

    def classify(vector):
        raw = node(
            "dense", [vector], w=classifier_wp, b=classifier_bp
        )
        summed = node("dense", [vr(raw)], w=group_wp)
        excess = vr(node(
            "dense", [summed], w=identity_21, b=minus_ones_21
        ))
        return va(summed, node("dense", [excess], w=neg_eye_21))

    current_class = classify(current_orientations[0])
    goal_class = classify(goal)
    type_w = np.zeros((3, 21), dtype=np.float32)
    index_w = np.zeros((1, 21), dtype=np.float32)
    count_w = np.ones((1, 21), dtype=np.float32)
    for group in range(3):
        type_w[group, 7 * group:7 * group + 7] = 1.0
        index_w[0, 7 * group:7 * group + 7] = np.arange(1, 8)
    type_wp = param(type_w)
    index_wp = param(index_w)
    count_wp = param(count_w)

    current_type = node("dense", [current_class], w=type_wp)
    goal_type = node("dense", [goal_class], w=type_wp)
    same_type = dense(
        vm(current_type, goal_type),
        np.ones((1, 3), dtype=np.float32),
    )
    current_count = node("dense", [current_class], w=count_wp)
    goal_count = node("dense", [goal_class], w=count_wp)

    def exactly_one(x):
        d = vscale(x, 1.0, -1.0)
        absolute = va(vr(d), vr(vscale(d, -1.0)))
        return vr(vscale(absolute, -1.0, 1.0))

    unambiguous = vm(exactly_one(current_count), exactly_one(goal_count))
    same_type = vm(same_type, unambiguous)
    current_index = node("dense", [current_class], w=index_wp)
    goal_index = node("dense", [goal_class], w=index_wp)
    delta = vsub(goal_index, current_index)

    negative_short = vsub(
        vclamp(vscale(delta, 2.0, 7.0)),
        vclamp(vscale(delta, 2.0, 1.0)),
    )
    positive_wrap = vclamp(vscale(delta, 2.0, -7.0))
    go_backward = vm(same_type, va(negative_short, positive_wrap))

    needs_change = vm(found, vone_minus(equal))
    backward_score = vm(needs_change, go_backward)
    forward_score = vm(needs_change, vone_minus(go_backward))
    action_inputs = node("vconcat", [backward_score, forward_score])
    action_w = np.zeros((8, 2), dtype=np.float32)
    action_b = np.full(8, -2.0, dtype=np.float32)
    action_w[1, 0] = 1.0
    action_w[2, 1] = 1.0
    action_b[1] = 0.0
    action_b[2] = 0.0
    action_b[4] = 0.5
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [action_inputs],
        "w": param(action_w),
        "b": param(action_b),
    })
    graph.append({
        "id": "click",
        "op": "slice",
        "in": ["x0"],
        "start": 0,
        "stop": 1,
    })

    return {"graph": graph, "memory_size": 0, "params": params}

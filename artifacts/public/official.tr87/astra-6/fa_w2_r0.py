import numpy as np


def build():
    P = {}
    parameter_cache = {}
    shapes = {
        "x0": ("s", 16),
        "coords": ("s", 3),
        "avail": ("v", 8),
        "aprev": ("v", 8),
    }
    graph = []
    current = graph
    global_cache = {}
    local_cache = global_cache
    serial = [0]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(P))
            parameter_cache[key] = name
            P[name] = a
        return parameter_cache[key]

    def node(op, ins, **fields):
        nonlocal current, local_cache
        key = (op, tuple(ins), tuple(sorted(fields.items())))
        if key in global_cache:
            return global_cache[key]
        if key in local_cache:
            return local_cache[key]

        name = fresh()
        nd = {"id": name, "op": op, "in": list(ins)}
        nd.update(fields)
        current.append(nd)
        local_cache[key] = name

        if op == "conv":
            shape = ("s", P[fields["w"]].shape[0])
        elif op == "dense":
            shape = ("v", P[fields["w"]].shape[0])
        elif op in ("gmax", "gmean"):
            shape = ("v", shapes[ins[0]][1])
        elif op in ("concat", "vconcat"):
            shape = (shapes[ins[0]][0],
                     sum(shapes[x][1] for x in ins))
        elif op in ("slice", "vslice"):
            shape = (shapes[ins[0]][0],
                     fields["stop"] - fields["start"])
        elif op in ("add", "sub", "mul", "max2", "min2"):
            shape = ("s", max(shapes[x][1] for x in ins))
        elif op in ("vadd", "vmul"):
            shape = ("v", max(shapes[x][1] for x in ins))
        else:
            shape = shapes[ins[0]]
        shapes[name] = shape
        return name

    def linear(x, w, b=None):
        w = np.asarray(w, np.float32)
        spatial = shapes[x][0] == "s"
        if spatial and w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv" if spatial else "dense", [x], **fields)

    def affine(x, a=1.0, b=0.0):
        n = shapes[x][1]
        return linear(x, np.eye(n, dtype=np.float32) * a,
                      np.full(n, b, np.float32))

    def const(values):
        nonlocal current, local_cache
        values = np.asarray(values, np.float32).reshape(-1)
        old_current, old_cache = current, local_cache
        current, local_cache = graph, global_cache
        result = linear("avail", np.zeros((len(values), 8), np.float32),
                        values)
        current, local_cache = old_current, old_cache
        return result

    def cat(*xs):
        return node("concat" if shapes[xs[0]][0] == "s" else "vconcat",
                    xs)

    def sl(x, start, stop=None):
        if stop is None:
            stop = start + 1
        return node("slice" if shapes[x][0] == "s" else "vslice",
                    [x], start=start, stop=stop)

    def add(a, b):
        spatial = shapes[a][0] == "s" or shapes[b][0] == "s"
        return node("add" if spatial else "vadd", [a, b])

    def mul(a, b):
        spatial = shapes[a][0] == "s" or shapes[b][0] == "s"
        return node("mul" if spatial else "vmul", [a, b])

    def sub(a, b):
        if shapes[a][0] == "s" or shapes[b][0] == "s":
            return node("sub", [a, b])
        return add(a, affine(b, -1))

    def relu(x):
        return node("relu" if shapes[x][0] == "s" else "vrelu", [x])

    def inv(x):
        return affine(x, -1, 1)

    def clip(x):
        if shapes[x][0] == "s":
            return node("min2", [relu(x), one])
        return sub(relu(x), relu(affine(x, 1, -1)))

    def equal(a, b):
        d = sub(a, b)
        if shapes[d][0] == "s":
            magnitude = node("abs", [d])
        else:
            magnitude = add(relu(d), relu(affine(d, -1)))
        return relu(affine(magnitude, -1, 1))

    def ge(a, b):
        return clip(affine(sub(a, b), 1, 1))

    def total(x):
        return linear(x, np.ones((1, shapes[x][1]), np.float32))

    def pool(x, maximum=False):
        return node("gmax" if maximum else "gmean", [x])

    def spatial_sum(x):
        return affine(pool(x), 4096)

    def shift(x, dy=0, dx=0):
        channels = shapes[x][1]
        while dy or dx:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            radius = max(abs(sy), abs(sx))
            k = 2 * radius + 1
            w = np.zeros((channels, channels, k, k), np.float32)
            for c in range(channels):
                w[c, c, radius + sy, radius + sx] = 1
            x = linear(x, w)
            dy -= sy
            dx -= sx
        return x

    def vshift(x, distance):
        n = shapes[x][1]
        w = np.zeros((n, n), np.float32)
        for i in range(n):
            j = i + distance
            if 0 <= j < n:
                w[i, j] = 1
        return linear(x, w)

    def first_true(x):
        n = shapes[x][1]
        previous = linear(x, np.tril(np.ones((n, n), np.float32), -1))
        return relu(sub(x, previous))

    loop_nodes = []

    def loop(T, specifications, body_function):
        nonlocal current, local_cache
        outer_current, outer_cache = current, local_cache
        states = []
        state_ids = []
        for spec in specifications:
            sid = fresh("s")
            state_ids.append(sid)
            entry = {"id": sid}
            entry.update(spec)
            states.append(entry)
            if "init" in spec:
                shapes[sid] = shapes[spec["init"]]
            else:
                z = spec["zeros"]
                shapes[sid] = ("s", z[0]) if len(z) == 3 else ("v", z[0])

        body = []
        current, local_cache = body, {}
        next_ids = body_function(state_ids)
        current, local_cache = outer_current, outer_cache

        lid = fresh("L")
        nd = {
            "id": lid, "op": "loop", "T": int(T),
            "state": states, "body": body,
            "next": dict(zip(state_ids, next_ids)),
        }
        current.append(nd)
        loop_nodes.append(nd)
        results = []
        for sid in state_ids:
            name = lid + "." + sid
            shapes[name] = shapes[sid]
            results.append(name)
        return results

    zero = const([0])
    one_v = const([1])
    one = sl("coords", 2, 3)

    # Binary foreground templates, grouped A, B, C.
    patterns = [
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

    black = sl("x0", 5)
    grey = sl("x0", 3)
    families = cat(sl("x0", 10), sl("x0", 7), sl("x0", 11))

    filters = []
    biases = []
    for pattern in patterns:
        a = np.array([[int(c) for c in row] for row in pattern],
                     dtype=np.float32)
        for rotation in range(4):
            r = np.rot90(a, rotation)
            filters.append((2 * r - 1)[None])
            biases.append(1 - r.sum())
    detected_rotations = relu(linear(
        black, np.stack(filters), np.asarray(biases, np.float32)))

    combine = np.zeros((21, 84), np.float32)
    for j in range(21):
        combine[j, 4*j:4*j+4] = 1
    detected_shapes = node(
        "min2", [linear(detected_rotations, combine), one])

    border = np.ones((7, 7), np.float32)
    border[1:6, 1:6] = 0
    border_weights = np.zeros((3, 3, 7, 7), np.float32)
    for j in range(3):
        border_weights[j, j] = border
    valid_borders = relu(linear(
        families, border_weights, np.full(3, -23, np.float32)))
    expand_family = np.zeros((21, 3), np.float32)
    for j in range(21):
        expand_family[j, j // 7] = 1
    glyphs = node("min2", [
        detected_shapes, linear(valid_borders, expand_family)
    ])

    encoding = np.stack([
        np.ones(21, np.float32),
        np.arange(1, 22, dtype=np.float32),
        np.repeat(np.arange(1, 4, dtype=np.float32), 7),
    ])
    encoded = linear(glyphs, encoding)
    present, code, family = [sl(encoded, j) for j in range(3)]

    top_row = node("rowmax", [sl("x0", 2)])
    bottom_present = mul(present, inv(top_row))
    selected = mul(present, shift(sl("x0", 0), -6, 0))
    packed = cat(present, code, family, selected)

    def parts(x):
        return [sl(x, j) for j in range(4)]

    p0 = parts(packed)
    lm1_tensor = shift(packed, 0, -7)
    lm2_tensor = shift(lm1_tensor, 0, -7)
    rp0_tensor = shift(packed, 0, 10)
    rp1_tensor = shift(rp0_tensor, 0, 7)
    rp2_tensor = shift(rp1_tensor, 0, 7)
    lm1, lm2 = parts(lm1_tensor), parts(lm2_tensor)
    rp0, rp1, rp2 = parts(rp0_tensor), parts(rp1_tensor), parts(rp2_tensor)

    anchors = mul(
        mul(mul(present, rp0[0]), shift(grey, 0, 5)), top_row)

    left_has_two = lm1[0]
    left_has_three = mul(lm1[0], lm2[0])
    left_length = add(add(present, left_has_two), left_has_three)
    left_first = add(
        add(code, mul(left_has_two, sub(lm1[1], code))),
        mul(left_has_three, sub(lm2[1], lm1[1])))
    left_second = mul(
        left_has_two,
        add(code, mul(lm2[0], sub(lm1[1], code))))
    left_third = mul(left_has_three, code)

    right_length = add(
        add(rp0[0], rp1[0]), mul(rp1[0], rp2[0]))
    right_first = rp0[1]
    right_second = rp1[1]
    right_third = mul(rp1[0], rp2[1])

    row_coordinate = affine(sl("coords", 0), 63)
    last_bottom_row = pool(mul(row_coordinate, bottom_present), True)
    output_rows = clip(affine(sub(row_coordinate, last_bottom_row), 1, 4))
    input_cells = mul(bottom_present, inv(output_rows))
    output_cells = mul(bottom_present, output_rows)

    anchor_row_counts = node("rowsum", [anchors])
    prefix_drive = cat(anchors, anchor_row_counts, bottom_present)
    prefix_weights = np.zeros((3, 3, 3, 3), np.float32)
    prefix_weights[0, 0, 1, 0] = 1
    prefix_weights[1, 1, 0, 1] = 1
    prefix_weights[2, 2, 1, 0] = 1

    def prefix_body(ss):
        return [add(linear(ss[0], prefix_weights), prefix_drive)]

    prefix = loop(
        64, [{"zeros": [3, 64, 64]}], prefix_body)[0]
    anchor_rank = affine(
        sub(add(sl(prefix, 0), sl(prefix, 1)), anchor_row_counts),
        1, -1)
    bottom_rank = affine(sl(prefix, 2), 1, -1)

    K = 8
    N = 9
    WIDE = 27

    anchor_masks = mul(
        equal(anchor_rank, const(np.arange(K))), anchors)
    bottom_masks = mul(
        equal(bottom_rank, const(np.arange(N))), bottom_present)

    def extract(feature, masks):
        return affine(pool(mul(feature, masks)), 4096)

    top_features = [
        left_length, right_length,
        left_first, left_second, left_third,
        right_first, right_second, right_third,
        p0[3], rp0[3], p0[2], rp0[2],
    ]
    (LL, RL, A, B, C, D, E, F,
     selected_left, selected_right, LF, RF) = [
        extract(f, anchor_masks) for f in top_features
    ]

    source = extract(mul(code, input_cells), bottom_masks)
    destination = extract(mul(code, output_cells), bottom_masks)
    selected_output = extract(mul(selected, output_cells), bottom_masks)
    source_family = pool(mul(family, input_cells), True)
    destination_family = pool(mul(family, output_cells), True)
    source_length = spatial_sum(input_cells)
    destination_length = spatial_sum(output_cells)

    exists = clip(LL)
    bp, cp, ep, fp = [clip(x) for x in (B, C, E, F)]
    b_absent, c_absent = inv(bp), inv(cp)

    def read_with_index(values, index):
        return total(mul(values, index))

    # A deterministic finite-string transducer. Repeated application also
    # handles dictionaries whose visible translation is composed in two stages.
    def translate(values, iterations):
        slots = [values, vshift(values, 1), vshift(values, 2)]
        position_numbers = const(np.arange(WIDE))
        write_numbers = [
            const(np.arange(WIDE) - j) for j in range(3)
        ]

        def body(ss):
            p, q, result = ss
            at_p = equal(position_numbers, p)
            v0, v1, v2 = [read_with_index(x, at_p) for x in slots]
            m0 = equal(A, v0)
            m1 = add(mul(equal(B, v1), bp), b_absent)
            m2 = add(mul(equal(C, v2), cp), c_absent)
            matches = mul(mul(mul(m0, m1), m2), exists)
            choice = first_true(matches)

            consumed = total(mul(choice, LL))
            produced = total(mul(choice, RL))
            r0, r1, r2 = [
                total(mul(choice, x)) for x in (D, E, F)
            ]
            writes = [
                mul(equal(nums, q), symbol)
                for nums, symbol in zip(write_numbers, (r0, r1, r2))
            ]
            new_result = add(result, add(add(writes[0], writes[1]), writes[2]))
            return [add(p, consumed), add(q, produced), new_result]

        return loop(iterations, [
            {"zeros": [1]}, {"zeros": [1]}, {"zeros": [WIDE]}
        ], body)[2]

    padded_source = cat(source, const(np.zeros(WIDE - N)))
    translated_once = translate(padded_source, N)
    translated_twice = translate(translated_once, 12)
    first_direct = sl(translated_once, 0)
    family_low = affine(destination_family, 7, -6)
    family_high = affine(destination_family, 7)
    use_direct = mul(ge(first_direct, family_low), ge(family_high, first_direct))
    normal_target = sl(add(
        mul(translated_once, use_direct),
        mul(translated_twice, inv(use_direct))), 0, N)

    # Relative cyclic differences are unchanged when an entire rule side
    # is advanced through its seven-symbol alphabet.
    def cyclic_difference(a, b):
        d = sub(a, b)
        return add(d, affine(clip(affine(d, -1)), 7))

    template_relative = [
        cyclic_difference(B, A), cyclic_difference(C, A),
        cyclic_difference(E, D), cyclic_difference(F, D),
    ]
    source_windows = [
        source,
        cyclic_difference(vshift(source, 1), source),
        cyclic_difference(vshift(source, 2), source),
    ]
    destination_windows = [
        destination,
        cyclic_difference(vshift(destination, 1), destination),
        cyclic_difference(vshift(destination, 2), destination),
    ]
    optional_present = [bp, cp, ep, fp]
    optional_absent = [inv(x) for x in optional_present]
    family_compatible = mul(
        mul(equal(LF, source_family), equal(RF, destination_family)),
        exists)

    # Each DFS frame is [input position, output position, next rule,
    #                    assigned left heads, assigned right heads].
    # Stack movement is performed only with slices, concatenations and gates.
    frame_width = 3 + 2 * K
    stack_width = frame_width * 11
    indices_n = const(np.arange(N))
    indices_k = const(np.arange(K))
    rule_successors = np.arange(1, K + 1, dtype=np.float32)[None]
    empty_frame = const(np.zeros(frame_width))
    negative_LL = affine(LL, -1)
    negative_RL = affine(RL, -1)

    def constraint_body(ss):
        stack = ss[0]
        p = sl(stack, 0)
        q = sl(stack, 1)
        next_rule = sl(stack, 2)
        assigned_l = sl(stack, 3, 3 + K)
        assigned_r = sl(stack, 3 + K, frame_width)

        ip = equal(indices_n, p)
        iq = equal(indices_n, q)
        x0, x1, x2 = [read_with_index(x, ip) for x in source_windows]
        y0, y1, y2 = [read_with_index(x, iq) for x in destination_windows]

        structural = []
        for actual, expected, used, absent in zip(
                (x1, x2, y1, y2), template_relative,
                optional_present, optional_absent):
            structural.append(add(mul(equal(expected, actual), used), absent))

        assigned_l_used = clip(assigned_l)
        assigned_r_used = clip(assigned_r)
        same_l = equal(assigned_l, x0)
        same_r = equal(assigned_r, y0)
        agree_l = add(same_l, inv(assigned_l_used))
        agree_r = add(same_r, inv(assigned_r_used))

        # Reuse an already assigned input head rather than introduce an
        # ambiguous competing rule.
        head_hits = mul(same_l, assigned_l_used)
        head_seen = clip(total(head_hits))
        unique_head = add(head_hits, inv(head_seen))

        remaining_input = affine(sub(source_length, p), 1, 1)
        remaining_output = affine(sub(destination_length, q), 1, 1)
        fits_input = clip(add(negative_LL, remaining_input))
        fits_output = clip(add(negative_RL, remaining_output))
        not_tried = ge(indices_k, next_rule)

        feasible = family_compatible
        for factor in structural + [
                agree_l, agree_r, unique_head,
                fits_input, fits_output, not_tried]:
            feasible = mul(feasible, factor)

        choice = first_true(feasible)
        can_advance = total(choice)
        next_p = add(p, total(mul(choice, LL)))
        next_q = add(q, total(mul(choice, RL)))
        new_l = add(assigned_l, mul(choice, sub(x0, assigned_l)))
        new_r = add(assigned_r, mul(choice, sub(y0, assigned_r)))
        parent_next_rule = linear(choice, rule_successors)

        child = cat(next_p, next_q, zero, new_l, new_r)
        parent = cat(p, q, parent_next_rule, assigned_l, assigned_r)
        old_tail = sl(stack, frame_width, stack_width - frame_width)
        pushed = cat(child, parent, old_tail)
        popped = cat(sl(stack, frame_width, stack_width), empty_frame)

        complete = equal(p, source_length)
        running = inv(complete)
        push_gate = mul(running, can_advance)
        pop_gate = mul(running, inv(can_advance))
        next_stack = add(
            add(mul(pushed, push_gate), mul(popped, pop_gate)),
            mul(stack, complete))
        return [next_stack]

    solved_stack = loop(
        32, [{"zeros": [stack_width]}], constraint_body)[0]
    constraint_loop = loop_nodes[-1]
    altered_l = sl(solved_stack, 3, 3 + K)
    altered_r = sl(solved_stack, 3 + K, frame_width)

    def selected_error(target, actual, selector):
        nonempty = clip(target)
        wrong = mul(inv(equal(target, actual)), nonempty)
        error = total(mul(sub(target, actual), selector))
        selected_wrong = total(mul(wrong, selector))
        return error, selected_wrong

    normal_error, normal_wrong = selected_error(
        normal_target, destination, selected_output)
    left_error, left_wrong = selected_error(
        altered_l, A, selected_left)
    right_error, right_wrong = selected_error(
        altered_r, D, selected_right)
    altered_error = add(left_error, right_error)
    altered_wrong = add(left_wrong, right_wrong)

    editable_mode = clip(add(total(selected_left), total(selected_right)))
    error = add(
        mul(normal_error, inv(editable_mode)),
        mul(altered_error, editable_mode))
    wrong = clip(add(
        mul(normal_wrong, inv(editable_mode)),
        mul(altered_wrong, editable_mode)))

    positive_short = mul(ge(error, const([1])), ge(const([3]), error))
    wrapped_positive = ge(const([-4]), error)
    increment = clip(add(positive_short, wrapped_positive))
    decrement_action = mul(wrong, inv(increment))
    increment_action = mul(wrong, increment)
    advance_action = inv(wrong)

    controls = cat(decrement_action, increment_action, advance_action)
    output_weights = np.zeros((8, 3), np.float32)
    output_weights[1, 0] = 10
    output_weights[2, 1] = 10
    output_weights[4, 2] = 5
    output_bias = np.full(8, -20, np.float32)
    output_bias[[1, 2, 4]] = 0
    action_values = linear(controls, output_weights, output_bias)

    # Required, fixed output identifiers.
    graph.append({
        "id": "logits", "op": "dense", "in": [action_values],
        "w": parameter(np.eye(8, dtype=np.float32))
    })
    shapes["logits"] = ("v", 8)
    graph.append({
        "id": "click", "op": "conv", "in": [one],
        "w": parameter(np.zeros((1, 1, 1, 1), np.float32))
    })
    shapes["click"] = ("s", 1)

    # Choose the bounded solver's iteration count analytically from the
    # constructed graph's operation counts, not from observations or trials.
    def statistics(nodes):
        evaluations = 0
        macs = 0
        for nd in nodes:
            op = nd["op"]
            evaluations += 1
            if op == "loop":
                e, m = statistics(nd["body"])
                evaluations += nd["T"] * e
                macs += nd["T"] * m
                continue
            if op == "conv":
                macs += int(P[nd["w"]].size) * 4096
            elif op == "dense":
                macs += int(P[nd["w"]].size)
            elif op in ("add", "sub", "mul", "max2", "min2"):
                macs += shapes[nd["id"]][1] * 4096
            elif op in ("vadd", "vmul"):
                macs += shapes[nd["id"]][1]
            elif op in (
                    "rowmax", "colmax", "rowsum", "colsum", "gmax", "gmean"):
                macs += shapes[nd["in"][0]][1] * 4096
        return evaluations, macs

    constraint_loop["T"] = 1
    all_e, all_m = statistics(graph)
    body_e, body_m = statistics(constraint_loop["body"])
    base_e, base_m = all_e - body_e, all_m - body_m
    by_evaluations = (9800 - base_e) // max(1, body_e)
    by_macs = (1450000000 - base_m) // max(1, body_m)
    constraint_loop["T"] = int(max(
        1, min(64, by_evaluations, by_macs)))

    return {
        "graph": graph,
        "memory_size": 0,
        "params": P,
    }

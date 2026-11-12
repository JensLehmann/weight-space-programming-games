import numpy as np


def build():
    graph = []
    params = {}
    destinations = [graph]
    cache = {}
    serial = [0]

    def fresh(prefix="n"):
        serial[0] += 1
        return prefix + str(serial[0])

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in cache:
            name = "p" + str(len(params))
            params[name] = a
            cache[key] = name
        return cache[key]

    def node(op, inputs, **fields):
        name = fresh()
        nd = {"id": name, "op": op, "in": list(inputs)}
        nd.update(fields)
        destinations[-1].append(nd)
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

    def relu(x):
        return node("relu", [x])

    def vr(x):
        return node("vrelu", [x])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def sa(a, b):
        return node("add", [a, b])

    def ss(a, b):
        return node("sub", [a, b])

    def sm(a, b):
        return node("mul", [a, b])

    def vc(*xs):
        return node("vconcat", xs)

    def sc(*xs):
        return node("concat", xs)

    def vs(x, start, stop):
        return node("vslice", [x], start=start, stop=stop)

    def sl(x, start, stop):
        return node("slice", [x], start=start, stop=stop)

    def vmax(x):
        return node("gmax", [x])

    def vaff(x, n, scale=1.0, bias=0.0):
        return dense(
            x,
            np.eye(n, dtype=np.float32) * scale,
            np.full(n, bias, dtype=np.float32),
        )

    def vneg(x, n):
        return dense(x, -np.eye(n, dtype=np.float32))

    def vsub(a, b, n):
        return va(a, vneg(b, n))

    def vabs(x, n):
        return va(vr(x), vr(vneg(x, n)))

    def veq(a, b, n):
        return vr(vaff(vabs(vsub(a, b, n), n), n, -1, 1))

    def vcap(x, n):
        return vsub(x, vr(vaff(x, n, 1, -1)), n)

    def vpositive(x, n):
        return vcap(vr(x), n)

    def vwrap7(x, n):
        return va(x, vaff(vpositive(vneg(x, n), n), n, 7, 0))

    def broadcast(x, n):
        return dense(x, np.ones((n, 1), np.float32))

    def total(x, n):
        return dense(x, np.ones((1, n), np.float32))

    def const(values):
        values = np.asarray(values, np.float32).reshape(-1)
        return dense(
            "avail",
            np.zeros((len(values), 8), np.float32),
            values,
        )

    def saff(x, channels, scale=1.0, bias=0.0):
        w = np.eye(channels, dtype=np.float32)
        w = w[:, :, None, None] * scale
        return conv(x, w, np.full(channels, bias, np.float32))

    def scap(x, channels):
        return ss(x, relu(saff(x, channels, 1, -1)))

    def ssharp(x):
        return scap(relu(saff(x, 1, 100, -50)), 1)

    def spatial_equal(a, b):
        difference = node("abs", [ss(a, b)])
        return ssharp(relu(saff(difference, 1, -4, 1)))

    def shift(x, channels, dy=0, dx=0):
        # Output at (r,c) reads input at (r+dy,c+dx).
        for axis, amount in ((0, dy), (1, dx)):
            while amount:
                step = max(-3, min(3, amount))
                k = 2 * abs(step) + 1
                w = np.zeros((channels, channels, k, k), np.float32)
                rr = k // 2 + (step if axis == 0 else 0)
                cc = k // 2 + (step if axis == 1 else 0)
                for ch in range(channels):
                    w[ch, ch, rr, cc] = 1
                x = conv(x, w)
                amount -= step
        return x

    zero = const([0])

    # Known glyph alphabet; rotations are enumerated analytically.
    glyph_strings = [
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

    colour_w = np.zeros((4, 16, 1, 1), np.float32)
    for out_ch, colour in enumerate([5, 10, 7, 11]):
        colour_w[out_ch, colour, 0, 0] = 1
    visual = conv("x0", colour_w)

    filters = []
    owners = []
    for identity, rows_text in enumerate(glyph_strings):
        silhouette = np.array(
            [[int(c) for c in row] for row in rows_text], np.int8
        )
        seen = set()
        for rotation in range(4):
            rot = np.rot90(silhouette, rotation)
            signature = rot.tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            black = np.zeros((7, 7), np.float32)
            black[1:6, 1:6] = rot
            filt = np.zeros((4, 7, 7), np.float32)
            filt[0] = black
            filt[1 + identity // 7] = 1 - black
            filters.append(filt)
            owners.append(identity)

    matches = relu(conv(
        visual,
        np.asarray(filters, np.float32),
        np.full(len(filters), -48, np.float32),
    ))
    combine = np.zeros((21, len(filters), 1, 1), np.float32)
    for j, owner in enumerate(owners):
        combine[owner, j, 0, 0] = 1
    identities = conv(matches, combine)

    # Tile-centre fields: glyph number, alphabet, presence.
    code_w = np.zeros((3, 21, 1, 1), np.float32)
    for j in range(21):
        code_w[0, j, 0, 0] = j % 7 + 1
        code_w[1, j, 0, 0] = j // 7 + 1
        code_w[2, j, 0, 0] = 1
    codes = conv(identities, code_w)
    presence = sl(codes, 2, 3)

    key_w = np.array([4032, 63, 0], np.float32).reshape(1, 3, 1, 1)
    pixel_key = conv("coords", key_w)
    tile_key = sm(pixel_key, presence)
    tile_fields = sc(codes, tile_key)

    row_w = np.array([63, 0, 0], np.float32).reshape(1, 3, 1, 1)
    rows = conv("coords", row_w)
    reverse_rows = saff(rows, 1, -1, 64)
    positive_key = saff(pixel_key, 1, -1, 8192)

    # Find the table/string separator from nearly solid padding rows.
    padding = sl("x0", 3, 4)
    padding_count = node("rowsum", [padding])
    solid_padding = scap(relu(saff(padding_count, 1, 1, -60)), 1)
    boundary_reverse = vmax(sm(solid_padding, reverse_rows))
    boundary = dense(
        boundary_reverse,
        np.array([[-1]], np.float32),
        np.array([64], np.float32),
    )
    lower_region = scap(relu(ss(rows, boundary)), 1)
    lower_presence = sm(presence, lower_region)

    low_reverse = vmax(sm(lower_presence, reverse_rows))
    input_row = dense(
        low_reverse,
        np.array([[-1]], np.float32),
        np.array([64], np.float32),
    )
    output_row = vmax(sm(lower_presence, rows))
    input_candidates = sm(
        lower_presence, spatial_equal(rows, input_row)
    )
    output_candidates = sm(
        lower_presence, spatial_equal(rows, output_row)
    )

    # Rule-gap centres: three padding pixels with background above/below.
    gap_w = np.zeros((1, 16, 3, 3), np.float32)
    gap_w[0, 3, 1, :] = 1
    gap_w[0, 2, 0, :] = 1
    gap_w[0, 2, 2, :] = 1
    gaps = relu(conv("x0", gap_w, np.array([-8], np.float32)))

    left1 = shift(tile_fields, 4, dx=-5)
    right1 = shift(tile_fields, 4, dx=5)
    left2 = sm(shift(left1, 4, dx=-7), sl(left1, 2, 3))
    right2 = sm(shift(right1, 4, dx=7), sl(right1, 2, 3))
    left3 = sm(shift(left2, 4, dx=-7), sl(left2, 2, 3))
    right3 = sm(shift(right2, 4, dx=7), sl(right2, 2, 3))

    p2 = sl(left2, 2, 3)
    p3 = sl(left3, 2, 3)
    left_start = sa(
        sa(left3, sm(left2, saff(p3, 1, -1, 1))),
        sm(left1, saff(p2, 1, -1, 1)),
    )
    left_second = sa(sm(left2, p3), sm(left1, ss(p2, p3)))
    left_third = sm(left1, p3)
    rule_fields = sc(
        left_start, left_second, left_third,
        right1, right2, right3,
    )

    # Upper-left bracket corner identifies the selected side or glyph.
    corner_w = np.zeros((1, 16, 5, 5), np.float32)
    for rr, cc in [(2, 2), (2, 3), (2, 4), (3, 2)]:
        corner_w[0, 0, rr, cc] = 1
    corner_w[0, 0, 2, 1] = -1
    corner = relu(conv(
        "x0", corner_w, np.array([-3], np.float32)
    ))
    selected_centre = shift(corner, 1, dy=-6, dx=-2)
    selected_key = vmax(sm(selected_centre, pixel_key))
    selected_number = vmax(sm(selected_centre, sl(codes, 0, 1)))
    top_region = saff(lower_region, 1, -1, 1)
    alter_mode = vmax(sm(selected_centre, top_region))
    ordinary_mode = vaff(alter_mode, 1, -1, 1)

    def enumerate_fields(features, candidates, channels, count):
        loop_id = fresh("read")
        candidate_state = fresh("cand")
        sequence_state = fresh("seq")
        body = []
        destinations.append(body)

        best_key = vmax(sm(candidate_state, positive_key))
        chosen = sm(
            candidate_state, spatial_equal(positive_key, best_key)
        )
        values = vmax(sm(features, chosen))
        retained = vs(sequence_state, channels, channels * count)
        next_sequence = vc(retained, values)
        next_candidates = ss(candidate_state, chosen)

        destinations.pop()
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": count,
            "state": [
                {"id": candidate_state, "init": candidates},
                {"id": sequence_state, "zeros": [channels * count]},
            ],
            "body": body,
            "next": {
                candidate_state: next_candidates,
                sequence_state: next_sequence,
            },
        })
        return loop_id + "." + sequence_state

    rule_record = enumerate_fields(rule_fields, gaps, 24, 6)
    input_record = enumerate_fields(tile_fields, input_candidates, 4, 9)
    output_record = enumerate_fields(tile_fields, output_candidates, 4, 9)

    def extract(x, width, indices):
        w = np.zeros((len(indices), width), np.float32)
        for j, index in enumerate(indices):
            w[j, index] = 1
        return dense(x, w)

    def bottom_field(record, field):
        return extract(record, 36, [4 * j + field for j in range(9)])

    in_num = bottom_field(input_record, 0)
    in_family = bottom_field(input_record, 1)
    out_num = bottom_field(output_record, 0)
    out_family = bottom_field(output_record, 1)
    out_presence = bottom_field(output_record, 2)
    out_keys = bottom_field(output_record, 3)

    def rule_field(side, field):
        return extract(
            rule_record, 144,
            [24 * r + 12 * side + 4 * slot + field
             for r in range(6) for slot in range(3)],
        )

    ln = rule_field(0, 0)
    lf = rule_field(0, 1)
    lp = rule_field(0, 2)
    rn = rule_field(1, 0)
    rf = rule_field(1, 1)
    rp = rule_field(1, 2)

    left_keys = extract(
        rule_record, 144, [24 * j + 3 for j in range(6)]
    )
    right_keys = extract(
        rule_record, 144, [24 * j + 15 for j in range(6)]
    )

    sum_slots = np.zeros((6, 18), np.float32)
    first_slots = np.zeros((6, 18), np.float32)
    repeat_rules = np.zeros((18, 6), np.float32)
    query_to_rules = np.zeros((18, 3), np.float32)
    collapse_rules = np.zeros((3, 18), np.float32)
    for r in range(6):
        first_slots[r, 3 * r] = 1
        for s in range(3):
            sum_slots[r, 3 * r + s] = 1
            repeat_rules[3 * r + s, r] = 1
            query_to_rules[3 * r + s, s] = 1
            collapse_rules[s, 3 * r + s] = 1

    left_lengths = dense(lp, sum_slots)
    right_lengths = dense(rp, sum_slots)
    rule_valid = vpositive(left_lengths, 6)
    left_first_num = dense(ln, first_slots)
    right_first_num = dense(rn, first_slots)

    pad_w = np.zeros((27, 9), np.float32)
    pad_w[:9] = np.eye(9, dtype=np.float32)
    input_num27 = dense(in_num, pad_w)
    input_family27 = dense(in_family, pad_w)
    output_num27 = dense(out_num, pad_w)
    output_family27 = dense(out_family, pad_w)

    def pointer_selector(pointer, offset):
        difference = dense(
            pointer,
            -np.ones((27, 1), np.float32),
            np.arange(27, dtype=np.float32) - offset,
        )
        return vr(vaff(vabs(difference, 27), 27, -1, 1))

    def query_string(numbers, families, pointer):
        ns, fs = [], []
        for offset in range(3):
            sel = pointer_selector(pointer, offset)
            ns.append(total(vm(numbers, sel), 27))
            fs.append(total(vm(families, sel), 27))
        return vc(*ns), vc(*fs)

    def match_side(qn, qf, numbers, families, present, modular):
        q_numbers = dense(qn, query_to_rules)
        q_families = dense(qf, query_to_rules)
        if modular:
            differences = vwrap7(vsub(q_numbers, numbers, 18), 18)
            first_difference = dense(differences, first_slots)
            repeated_difference = dense(first_difference, repeat_rules)
            good_numbers = veq(differences, repeated_difference, 18)
        else:
            good_numbers = veq(q_numbers, numbers, 18)
        good_families = veq(q_families, families, 18)
        good = vm(good_numbers, good_families)
        errors = vm(vaff(good, 18, -1, 1), present)
        error_count = dense(errors, sum_slots)
        compatible = vr(vaff(error_count, 6, -1, 1))
        return vm(compatible, rule_valid)

    prefix_w = np.tril(np.ones((6, 6), np.float32), -1)

    def first_match(matches):
        return vr(vsub(matches, dense(matches, prefix_w), 6))

    def chosen_length(selection, lengths):
        return total(vm(selection, lengths), 6)

    def selected_slots(values, selection):
        repeated = dense(selection, repeat_rules)
        return dense(vm(values, repeated), collapse_rules)

    def rewrite(numbers, families, iterations):
        loop_id = fresh("rewrite")
        ip = fresh("ip")
        op = fresh("op")
        nums = fresh("numbers")
        fams = fresh("families")
        body = []
        destinations.append(body)

        qn, qf = query_string(numbers, families, ip)
        compatible = match_side(qn, qf, ln, lf, lp, False)
        selection = first_match(compatible)
        chosen_numbers = selected_slots(rn, selection)
        chosen_families = selected_slots(rf, selection)
        consume = chosen_length(selection, left_lengths)
        produce = chosen_length(selection, right_lengths)

        new_nums, new_fams = nums, fams
        for offset in range(3):
            position = pointer_selector(op, offset)
            value_n = vs(chosen_numbers, offset, offset + 1)
            value_f = vs(chosen_families, offset, offset + 1)
            new_nums = va(new_nums, vm(position, value_n))
            new_fams = va(new_fams, vm(position, value_f))
        next_ip = va(ip, consume)
        next_op = va(op, produce)

        destinations.pop()
        graph.append({
            "id": loop_id,
            "op": "loop",
            "T": iterations,
            "state": [
                {"id": ip, "init": zero},
                {"id": op, "init": zero},
                {"id": nums, "zeros": [27]},
                {"id": fams, "zeros": [27]},
            ],
            "body": body,
            "next": {
                ip: next_ip, op: next_op,
                nums: new_nums, fams: new_fams,
            },
        })
        return loop_id + "." + nums, loop_id + "." + fams

    translated1, family1 = rewrite(input_num27, input_family27, 9)
    translated2, family2 = rewrite(translated1, family1, 27)

    direct = veq(vs(family1, 0, 1), vs(out_family, 0, 1), 1)
    indirect = vaff(direct, 1, -1, 1)
    desired_output = va(
        vm(vs(translated1, 0, 9), direct),
        vm(vs(translated2, 0, 9), indirect),
    )

    # This shared constant must be top-level, not loop-local.
    zeros6 = broadcast(zero, 6)

    # Constraint pass for layouts where rule sides are editable.
    alter_loop = fresh("constraints")
    aip = fresh("aip")
    aop = fresh("aop")
    left_targets_state = fresh("lt")
    right_targets_state = fresh("rt")
    body = []
    destinations.append(body)

    aqn, aqf = query_string(input_num27, input_family27, aip)
    bqn, bqf = query_string(output_num27, output_family27, aop)
    left_compatible = match_side(aqn, aqf, ln, lf, lp, True)
    right_compatible = match_side(bqn, bqf, rn, rf, rp, True)

    repeated_a = broadcast(vs(aqn, 0, 1), 6)
    repeated_b = broadcast(vs(bqn, 0, 1), 6)
    unused = veq(left_targets_state, zeros6, 6)
    same_left = veq(left_targets_state, repeated_a, 6)
    same_right = veq(right_targets_state, repeated_b, 6)
    reuse_allowed = vcap(va(unused, vm(same_left, same_right)), 6)
    candidates = vm(vm(left_compatible, right_compatible), reuse_allowed)
    assignment = first_match(candidates)

    next_left_targets = va(
        left_targets_state,
        vm(assignment, vsub(repeated_a, left_targets_state, 6)),
    )
    next_right_targets = va(
        right_targets_state,
        vm(assignment, vsub(repeated_b, right_targets_state, 6)),
    )
    next_aip = va(aip, chosen_length(assignment, left_lengths))
    next_aop = va(aop, chosen_length(assignment, right_lengths))

    destinations.pop()
    graph.append({
        "id": alter_loop,
        "op": "loop",
        "T": 9,
        "state": [
            {"id": aip, "init": zero},
            {"id": aop, "init": zero},
            {"id": left_targets_state, "zeros": [6]},
            {"id": right_targets_state, "zeros": [6]},
        ],
        "body": body,
        "next": {
            aip: next_aip,
            aop: next_aop,
            left_targets_state: next_left_targets,
            right_targets_state: next_right_targets,
        },
    })

    assigned_left = alter_loop + "." + left_targets_state
    assigned_right = alter_loop + "." + right_targets_state
    unused_left = veq(assigned_left, zeros6, 6)
    unused_right = veq(assigned_right, zeros6, 6)
    desired_left = va(assigned_left, vm(unused_left, left_first_num))
    desired_right = va(assigned_right, vm(unused_right, right_first_num))
    desired_top = vc(desired_left, desired_right)
    current_top = vc(left_first_num, right_first_num)
    top_keys = vc(left_keys, right_keys)
    top_valid = vc(rule_valid, rule_valid)

    selected_bottom_mask = veq(out_keys, broadcast(selected_key, 9), 9)
    selected_top_mask = veq(top_keys, broadcast(selected_key, 12), 12)
    bottom_target = total(vm(desired_output, selected_bottom_mask), 9)
    top_target = total(vm(desired_top, selected_top_mask), 12)
    selected_target = va(
        vm(bottom_target, ordinary_mode),
        vm(top_target, alter_mode),
    )

    bottom_errors = vm(
        vaff(veq(desired_output, out_num, 9), 9, -1, 1),
        out_presence,
    )
    top_errors = vm(
        vaff(veq(desired_top, current_top, 12), 12, -1, 1),
        top_valid,
    )
    any_bottom_error = vpositive(total(bottom_errors, 9), 1)
    any_top_error = vpositive(total(top_errors, 12), 1)
    any_error = va(
        vm(any_bottom_error, ordinary_mode),
        vm(any_top_error, alter_mode),
    )

    current_error = vaff(
        veq(selected_target, selected_number, 1), 1, -1, 1
    )
    delta = vwrap7(vsub(selected_target, selected_number, 1), 1)
    increment_direction = vsub(
        vr(vaff(delta, 1, -1, 4)),
        vr(vaff(delta, 1, -1, 3)),
        1,
    )
    increment = vm(current_error, increment_direction)
    decrement = vsub(current_error, increment, 1)

    # Success is checked only after a glyph edit. Already-correct states
    # receive a reversible one-step edit, followed by restoration.
    complete = vaff(any_error, 1, -1, 1)
    decrement = va(decrement, complete)
    advance = vaff(current_error, 1, -1, 1)

    action_features = vc(decrement, increment, advance)
    action_w = np.zeros((8, 3), np.float32)
    action_w[1, 0] = 10
    action_w[2, 1] = 10
    action_w[4, 2] = 5
    action_b = np.array(
        [-100, 0, 0, -100, 0, -100, -100, -100], np.float32
    )
    graph.append({
        "id": "logits",
        "op": "dense",
        "in": [action_features],
        "w": parameter(action_w),
        "b": parameter(action_b),
    })
    graph.append({
        "id": "click",
        "op": "conv",
        "in": ["coords"],
        "w": parameter(np.zeros((1, 3, 1, 1), np.float32)),
    })

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

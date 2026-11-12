import itertools
import numpy as np


def build():
    graph = []
    root = graph
    params = {}
    parameter_cache = {}
    scalar_cache = {}
    serial = 0

    def uid(prefix="n"):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def parameter(value):
        a = np.asarray(value, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "p" + str(len(params))
            parameter_cache[key] = name
            params[name] = a
        return parameter_cache[key]

    def node(op, inputs, name=None, **fields):
        name = uid() if name is None else name
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        graph.append(d)
        return name

    def conv(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return node("conv", [x], name=name, **fields)

    def dense(x, w, b=None, name=None):
        w = np.asarray(w, np.float32)
        fields = {"w": parameter(w)}
        if b is not None:
            b = np.asarray(b, np.float32)
            if b.ndim == 0:
                b = np.full(w.shape[0], b, np.float32)
            fields["b"] = parameter(b)
        return node("dense", [x], name=name, **fields)

    def const(value):
        nonlocal graph
        value = float(value)
        if value not in scalar_cache:
            old_graph = graph
            graph = root
            scalar_cache[value] = dense(
                "avail", np.zeros((1, 8), np.float32), [value]
            )
            graph = old_graph
        return scalar_cache[value]

    one = const(1)
    minus_one = const(-1)

    def sl(x, a, b):
        return node("slice", [x], start=a, stop=b)

    def vsl(x, a, b):
        return node("vslice", [x], start=a, stop=b)

    def cat(*xs):
        return node("concat", xs)

    def vcat(*xs):
        return node("vconcat", xs)

    def add(a, b):
        return node("add", [a, b])

    def sub(a, b):
        return node("sub", [a, b])

    def mul(a, b):
        return node("mul", [a, b])

    def va(a, b):
        return node("vadd", [a, b])

    def vm(a, b):
        return node("vmul", [a, b])

    def vn(a):
        return vm(a, minus_one)

    def vs(a, b):
        return va(a, vn(b))

    def vr(a):
        return node("vrelu", [a])

    def sr(a):
        return node("relu", [a])

    def sa(a):
        return node("abs", [a])

    def vmax(a, b):
        return va(b, vr(vs(a, b)))

    def vabs(a):
        return va(vr(a), vr(vn(a)))

    def affine_s(x, a=1, b=0):
        if a != 1:
            x = mul(x, const(a))
        if b != 0:
            x = add(x, const(b))
        return x

    def affine_v(x, a=1, b=0):
        if a != 1:
            x = vm(x, const(a))
        if b != 0:
            x = va(x, const(b))
        return x

    def clip_s(x):
        return node("min2", [sr(x), one])

    def clip_v(x):
        return vs(vr(x), vr(va(x, const(-1))))

    def equal_s(a, b):
        return sr(affine_s(sa(sub(a, b)), -1, 1))

    def equal_v(a, b):
        return vr(affine_v(vabs(vs(a, b)), -1, 1))

    def gmax(x):
        return node("gmax", [x])

    def sample(x, dy=0, dx=0):
        # Output at (y,x) reads the input at (y+dy,x+dx).
        while dy != 0 or dx != 0:
            sy = max(-3, min(3, dy))
            sx = max(-3, min(3, dx))
            w = np.zeros((1, 1, 7, 7), np.float32)
            w[0, 0, 3 + sy, 3 + sx] = 1
            x = conv(x, w)
            dy -= sy
            dx -= sx
        return x

    def loop(T, initial, body_function):
        nonlocal graph
        loop_name = uid("L")
        states = {}
        specifications = []
        for key, init in initial.items():
            state_name = uid("s")
            states[key] = state_name
            spec = {"id": state_name}
            if isinstance(init, str):
                spec["init"] = init
            else:
                spec["zeros"] = list(init)
            specifications.append(spec)

        outer_graph = graph
        graph = []
        next_values = body_function(states)
        body = graph
        graph = outer_graph
        graph.append({
            "id": loop_name,
            "op": "loop",
            "T": T,
            "state": specifications,
            "body": body,
            "next": {
                states[key]: next_values[key] for key in states
            },
        })
        return {
            key: loop_name + "." + states[key] for key in states
        }

    # ------------------------------------------------------------------
    # Geometry: tile centers, bottom strings, connectors and selection.
    # ------------------------------------------------------------------

    tile_w = np.zeros((1, 16), np.float32)
    tile_w[0, 6:16] = 1
    tile = conv("x0", tile_w)

    start_w = np.zeros((1, 1, 3, 3), np.float32)
    start_w[0, 0, 1, 1] = 1
    start_w[0, 0, 0, 1] = -1
    start_w[0, 0, 1, 0] = -1
    corner = sr(conv(tile, start_w))

    border_w = np.zeros((1, 1, 7, 7), np.float32)
    border_w[0, 0, 0, :] = 1
    border_w[0, 0, :, 0] = 1
    border = sr(conv(tile, border_w, -12))
    seeds = mul(sample(corner, -3, -3), border)

    def centers_body(s):
        moved = mul(sample(s["c"], 0, -7), border)
        return {"c": node("max2", [s["c"], moved])}

    centers = loop(8, {"c": seeds}, centers_body)["c"]

    background_two = sl("x0", 2, 3)
    bottom_region = sub(one, node("rowmax", [background_two]))
    top_region = sub(one, bottom_region)
    bottom_centers = mul(centers, bottom_region)
    top_centers = mul(centers, top_region)

    xcoord = conv("coords", [[0, 63, 0]])
    ycoord = conv("coords", [[63, 0, 0]])
    rule_priority = conv("coords", [[-4032, -63, 0]], 5000)

    row_priority = mul(affine_s(ycoord, -1, 64), bottom_centers)
    source_row_score = gmax(row_priority)
    source_centers = mul(
        sr(add(sub(row_priority, source_row_score), one)),
        bottom_centers,
    )
    target_centers = sub(bottom_centers, source_centers)
    source_row = node("rowmax", [source_centers])
    target_row = node("rowmax", [target_centers])

    def first_x(mask):
        score = gmax(mul(affine_s(xcoord, -1, 64), mask))
        return dense(score, [[-1]], [64])

    source_x = first_x(source_centers)
    target_x = first_x(target_centers)

    def near_x(position):
        difference = sa(sub(xcoord, position))
        return clip_s(affine_s(difference, -4, 2))

    def cursor(position, row, mask=None):
        result = mul(near_x(position), row)
        if mask is not None:
            result = mul(result, mask)
        return result

    color_three = sl("x0", 3, 4)
    bridge_w = np.zeros((1, 1, 3, 3), np.float32)
    bridge_w[0, 0, 1, :] = 1
    bridge_w[0, 0, 0, 1] = -3
    bridge_w[0, 0, 2, 1] = -3
    bridges = mul(sr(conv(color_three, bridge_w, -2)), top_region)

    color_zero = sl("x0", 0, 1)
    bracket_w = np.zeros((1, 1, 3, 3), np.float32)
    for y, x in [(1, 1), (1, 2), (2, 1)]:
        bracket_w[0, 0, y, x] = 1
    for y, x in [(2, 2), (1, 0), (0, 1)]:
        bracket_w[0, 0, y, x] = -1
    bracket_corner = sr(conv(color_zero, bracket_w, -2))
    selected = mul(sample(bracket_corner, -6, -2), centers)
    editable_mode = gmax(mul(selected, top_region))

    # ------------------------------------------------------------------
    # Rotation-invariant fingerprints.
    # ------------------------------------------------------------------

    black = sl("x0", 5, 6)
    fingerprint_weights = np.array(
        [2 ** i if i < 15 else 19 * 2 ** (i - 15)
         for i in range(25)],
        dtype=np.float32,
    ).reshape(5, 5)

    rotations = np.stack(
        [np.rot90(fingerprint_weights, k) for k in range(4)]
    )[:, None, :, :]
    hashes4 = conv(black, rotations)
    fingerprint = node("min2", [sl(hashes4, 0, 1), sl(hashes4, 1, 2)])
    fingerprint = node("min2", [fingerprint, sl(hashes4, 2, 3)])
    fingerprint = node("min2", [fingerprint, sl(hashes4, 3, 4)])

    family_pixels = conv(
        "x0", np.arange(16, dtype=np.float32)[None, :]
    )
    family_border = sample(family_pixels, -3, 0)
    family_centers = mul(family_border, centers)
    codes = mul(
        add(fingerprint, affine_s(family_border, 65536)),
        centers,
    )

    # Catalogue is used only to choose a short direction around a
    # seven-symbol cycle. Translation itself uses visual fingerprints.
    patterns = [
        "00100 00100 01110 00100 11111",
        "00001 00101 11111 10100 10000",
        "11111 10100 10000 10100 11111",
        "00100 11111 10101 10101 00100",
        "10001 11111 00100 11111 10001",
        "11111 00100 11100 10000 11111",
        "10001 11111 10001 10001 11011",

        "10000 11110 10010 10010 11111",
        "11111 10001 11101 10101 11111",
        "00111 00101 11111 10100 11100",
        "01110 01010 11111 10001 11111",
        "11111 10001 10001 11111 00100",
        "11110 10011 10001 11001 01111",
        "00100 11111 10101 11111 00100",

        "10101 10000 10101 10000 11111",
        "11101 10000 00000 00001 10111",
        "10111 00100 00100 00100 11101",
        "00100 10101 00100 10101 00100",
        "11011 10001 10101 10001 11011",
        "10111 10101 11101 00000 10101",
        "10001 00000 10001 11011 01010",
    ]
    known_codes = []
    known_ids = []
    for i, text in enumerate(patterns):
        p = np.array(
            [[int(c) for c in row] for row in text.split()],
            dtype=np.float32,
        )
        h = min(
            float(np.sum(p * np.rot90(fingerprint_weights, k)))
            for k in range(4)
        )
        family = [10, 7, 11][i // 7]
        known_codes.append(h + 65536 * family)
        known_ids.append(i % 7 + 1)

    def identify(code_map):
        differences = conv(
            code_map,
            np.ones((21, 1), np.float32),
            -np.asarray(known_codes, np.float32),
        )
        matches = sr(affine_s(sa(differences), -1, 1))
        return conv(
            matches, np.asarray(known_ids, np.float32)[None, :]
        )

    ids = identify(codes)

    # Data sampled at each bridge:
    # left symbols nearest first, right symbols left-to-right;
    # six existence flags; two cycle ids; two selection flags.
    offsets = [-5, -12, -19, 5, 12, 19]
    bridge_codes = [sample(codes, 0, d) for d in offsets]
    bridge_exists = [sample(centers, 0, d) for d in offsets]
    left_selection = add(
        add(sample(selected, 0, -5), sample(selected, 0, -12)),
        sample(selected, 0, -19),
    )
    right_selection = sample(selected, 0, 5)
    rule_features = cat(
        *(bridge_codes + bridge_exists + [
            sample(ids, 0, -5),
            sample(ids, 0, 5),
            left_selection,
            right_selection,
        ])
    )

    packed_insert = np.zeros((48, 8), np.float32)
    packed_insert[:8] = np.eye(8, dtype=np.float32)
    packed_shift = np.zeros((48, 48), np.float32)
    packed_shift[8:, :40] = np.eye(40, dtype=np.float32)

    pack_projection = np.zeros((8, 16), np.float32)
    pack_projection[0, 0] = 1
    pack_projection[1, 3] = 1
    pack_projection[2, 6:9] = 1
    pack_projection[3, 9:12] = 1
    pack_projection[4, 14] = 1
    pack_projection[5, 15] = 1
    pack_projection[6, 12] = 1
    pack_projection[7, 13] = 1

    data_projection = np.zeros((5, 16), np.float32)
    data_projection[:3, 3:6] = np.eye(3, dtype=np.float32)
    data_projection[3, 6:9] = 1
    data_projection[4, 9:12] = 1

    length_projection = np.zeros((3, 16), np.float32)
    length_projection[0, 6] = 1
    length_projection[0, 7] = -1
    length_projection[1, 7] = 1
    length_projection[1, 8] = -1
    length_projection[2, 8] = 1

    def make_rewrite_map(input_codes, input_centers, collect=False):
        backwards = cat(
            input_codes,
            sample(input_codes, 0, -7),
            sample(input_codes, 0, -14),
        )
        initial = {
            "remaining": bridges,
            "used": [1, 64, 64],
            "data": [5, 64, 64],
        }
        if collect:
            initial["packed"] = [48]

        def body(s):
            scores = mul(s["remaining"], rule_priority)
            best = gmax(scores)
            chosen = mul(
                sr(add(sub(scores, best), one)),
                s["remaining"],
            )
            features = gmax(mul(rule_features, chosen))
            lhs = vsl(features, 0, 3)
            exists = vsl(features, 6, 9)
            mismatch = mul(
                node("min2", [sa(sub(backwards, lhs)), one]),
                exists,
            )
            matching_end = mul(
                sr(conv(mismatch, -np.ones((1, 3), np.float32), 1)),
                input_centers,
            )
            lengths = dense(features, length_projection)
            by_length = mul(matching_end, lengths)
            matching_start = add(
                add(
                    sl(by_length, 0, 1),
                    sample(sl(by_length, 1, 2), 0, 7),
                ),
                sample(sl(by_length, 2, 3), 0, 14),
            )
            accepted = sr(sub(matching_start, s["used"]))
            new_data = mul(accepted, dense(features, data_projection))
            result = {
                "remaining": sub(s["remaining"], chosen),
                "used": node("max2", [s["used"], matching_start]),
                "data": add(s["data"], new_data),
            }
            if collect:
                packed_rule = dense(features, pack_projection)
                result["packed"] = va(
                    dense(s["packed"], packed_shift),
                    dense(packed_rule, packed_insert),
                )
            return result

        return loop(6, initial, body)

    first_map = make_rewrite_map(codes, centers, collect=True)
    packed_rules = first_map["packed"]

    def translate(rewrite_map, input_mask, input_row, input_x):
        initial = {
            "input_x": input_x,
            "output_x": target_x,
            "output": [1, 64, 64],
        }

        def body(s):
            here = cursor(s["input_x"], input_row, input_mask)
            data = gmax(mul(rewrite_map, here))
            rhs = vsl(data, 0, 3)
            lhs_length = vsl(data, 3, 4)
            rhs_length = vsl(data, 4, 5)
            positions = dense(
                s["output_x"],
                np.ones((3, 1), np.float32),
                [0, 7, 14],
            )
            output_sites = mul(near_x(positions), target_row)
            emitted = conv(
                mul(output_sites, rhs), np.ones((1, 3), np.float32)
            )
            return {
                "input_x": va(s["input_x"], vm(lhs_length, const(7))),
                "output_x": va(s["output_x"], vm(rhs_length, const(7))),
                "output": node("max2", [s["output"], emitted]),
            }

        return loop(9, initial, body)["output"]

    expected_once = translate(
        first_map["data"], source_centers, source_row, source_x
    )
    virtual_centers = clip_s(expected_once)
    virtual_codes = add(mul(codes, top_region), expected_once)
    virtual_all_centers = add(top_centers, virtual_centers)

    second_map = make_rewrite_map(
        virtual_codes, virtual_all_centers, collect=False
    )
    expected_twice = translate(
        second_map["data"], virtual_centers, target_row, target_x
    )

    first_target_site = cursor(target_x, target_row, target_centers)
    first_expected_code = gmax(mul(expected_once, first_target_site))
    target_family = gmax(mul(family_centers, first_target_site))

    family_thresholds = dense(
        first_expected_code,
        np.ones((15, 1), np.float32),
        -65536 * np.arange(1, 16, dtype=np.float32),
    )
    first_family = dense(
        clip_v(family_thresholds), np.ones((1, 15), np.float32)
    )
    use_once = equal_v(first_family, target_family)
    expected = add(
        mul(expected_once, use_once),
        mul(expected_twice, vs(one, use_once)),
    )
    expected_ids = identify(expected)

    # ------------------------------------------------------------------
    # Editable-rule circuit. Six rule slots, three-symbol sides.
    # All 6! orderings are evaluated as vector lanes.
    # ------------------------------------------------------------------

    next_code = sample(codes, 0, 7)
    next_next_code = sample(codes, 0, 14)
    same_next = equal_s(codes, next_code)
    same_next_next = equal_s(next_code, next_next_code)
    run_lengths = mul(
        add(add(one, same_next), mul(same_next, same_next_next)),
        centers,
    )

    def read_bottom(mask, start):
        positions = dense(
            start,
            np.ones((9, 1), np.float32),
            7 * np.arange(9, dtype=np.float32),
        )
        sites = mul(near_x(positions), mask)
        return vcat(
            gmax(mul(sites, codes)),
            gmax(mul(sites, run_lengths)),
            gmax(mul(sites, ids)),
        )

    bottom_source = read_bottom(source_centers, source_x)
    bottom_target = read_bottom(target_centers, target_x)

    permutations = list(itertools.permutations(range(6)))
    lanes = len(permutations)

    get_left_lengths = np.zeros((6, 48), np.float32)
    get_right_lengths = np.zeros((6, 48), np.float32)
    for i in range(6):
        get_left_lengths[i, 8 * i + 2] = 1
        get_right_lengths[i, 8 * i + 3] = 1
    left_lengths = dense(packed_rules, get_left_lengths)
    right_lengths = dense(packed_rules, get_right_lengths)

    source_prefixes = []
    target_prefixes = []
    for rule in range(6):
        w = np.zeros((lanes, 6), np.float32)
        for lane, perm in enumerate(permutations):
            before = perm[:perm.index(rule)]
            for j in before:
                w[lane, j] = 1
        source_prefixes.append(dense(left_lengths, w))
        target_prefixes.append(dense(right_lengths, w))

    # Piecewise-linear lookup tables: at integral positions these
    # exactly retrieve code, repeated-run length and cycle id.
    expand = np.zeros((3 * lanes, 3), np.float32)
    for f in range(3):
        expand[f * lanes:(f + 1) * lanes, f] = 1

    coefficient_matrices = []
    c0 = np.zeros((3, 27), np.float32)
    c1 = np.zeros((3, 27), np.float32)
    for f in range(3):
        c0[f, 9 * f] = 1
        c1[f, 9 * f] = -1
        c1[f, 9 * f + 1] = 1
    coefficient_matrices.extend([c0, c1])
    for j in range(1, 10):
        w = np.zeros((3, 27), np.float32)
        for f in range(3):
            for k, value in [(j - 1, 1), (j, -2), (j + 1, 1)]:
                if 0 <= k < 9:
                    w[f, 9 * f + k] += value
        coefficient_matrices.append(w)

    def coefficients(bottom):
        return [
            dense(dense(bottom, w), expand)
            for w in coefficient_matrices
        ]

    source_coefficients = coefficients(bottom_source)
    target_coefficients = coefficients(bottom_target)

    def lookup(prefix, cs):
        q = vcat(prefix, prefix, prefix)
        result = va(cs[0], vm(q, cs[1]))
        for j in range(1, 10):
            hinge = vr(va(q, const(-j)))
            result = va(result, vm(hinge, cs[j + 1]))
        return (
            vsl(result, 0, lanes),
            vsl(result, lanes, 2 * lanes),
            vsl(result, 2 * lanes, 3 * lanes),
        )

    drop_rule = np.zeros((48, 48), np.float32)
    drop_rule[:40, 8:] = np.eye(40, dtype=np.float32)

    search_initial = {
        "rules": packed_rules,
        "cost": [lanes],
        "desired_code": [lanes],
        "desired_id": [lanes],
    }
    for i in range(6):
        search_initial["u" + str(i)] = source_prefixes[i]
        search_initial["v" + str(i)] = target_prefixes[i]

    def editable_body(s):
        rule = vsl(s["rules"], 0, 8)
        fields = [vsl(rule, i, i + 1) for i in range(8)]
        left_result = lookup(s["u0"], source_coefficients)
        right_result = lookup(s["v0"], target_coefficients)

        total_cost = s["cost"]
        desired_code = s["desired_code"]
        desired_id = s["desired_id"]

        for side, result in enumerate([left_result, right_result]):
            wanted_code, available_run, wanted_id = result
            current_code = fields[side]
            length = fields[2 + side]
            is_selected = fields[4 + side]
            current_id = fields[6 + side]

            bad_run = clip_v(vs(length, available_run))
            mismatch = vs(one, equal_v(wanted_code, current_code))
            distance = vabs(vs(wanted_id, current_id))
            distance = vs(
                distance,
                vr(affine_v(distance, 2, -7)),
            )
            cost = va(
                vm(bad_run, const(1000)),
                va(vm(distance, const(4)), mismatch),
            )
            total_cost = va(total_cost, cost)
            desired_code = va(
                desired_code, vm(wanted_code, is_selected)
            )
            desired_id = va(
                desired_id, vm(wanted_id, is_selected)
            )

        result = {
            "rules": dense(s["rules"], drop_rule),
            "cost": total_cost,
            "desired_code": desired_code,
            "desired_id": desired_id,
        }
        for i in range(6):
            result["u" + str(i)] = s["u" + str((i + 1) % 6)]
            result["v" + str(i)] = s["v" + str((i + 1) % 6)]
        return result

    editable = loop(6, search_initial, editable_body)

    tie_break = dense(
        one,
        np.zeros((lanes, 1), np.float32),
        -np.arange(lanes, dtype=np.float32),
    )
    scores = va(vm(editable["cost"], const(-1024)), tie_break)

    padding_count = 1024 - lanes
    padding = dense(
        one,
        np.zeros((padding_count, 1), np.float32),
        np.full(padding_count, -16000000, np.float32),
    )
    reduced = vcat(scores, padding)
    width = 1024
    while width > 1:
        half = width // 2
        reduced = vmax(
            vsl(reduced, 0, half),
            vsl(reduced, half, width),
        )
        width = half
    winner = vr(va(vs(scores, reduced), one))
    sum_lanes = np.ones((1, lanes), np.float32)

    editable_code = dense(
        vm(winner, editable["desired_code"]), sum_lanes
    )
    editable_id = dense(
        vm(winner, editable["desired_id"]), sum_lanes
    )
    editable_cost = dense(vm(winner, editable["cost"]), sum_lanes)

    # ------------------------------------------------------------------
    # Controller.
    # ------------------------------------------------------------------

    current_code = gmax(mul(codes, selected))
    current_id = gmax(mul(ids, selected))
    ordinary_code = gmax(mul(expected, selected))
    ordinary_id = gmax(mul(expected_ids, selected))
    desired_code = va(ordinary_code, editable_code)
    desired_id = va(ordinary_id, editable_id)

    mismatch = clip_v(vabs(vs(desired_code, current_code)))

    ordinary_bad = gmax(mul(
        node("min2", [sa(sub(expected, codes)), one]),
        target_centers,
    ))
    ordinary_complete = vs(one, ordinary_bad)
    editable_complete = vr(vs(one, editable_cost))
    force_check = va(
        vm(vs(one, editable_mode), ordinary_complete),
        vm(editable_mode, editable_complete),
    )
    rotate = vmax(mismatch, force_check)

    delta = vs(desired_id, current_id)
    increment = va(
        vs(clip_v(delta), clip_v(va(delta, const(-3)))),
        clip_v(affine_v(delta, -1, -3)),
    )
    rotate_increment = vm(rotate, increment)
    navigate = vs(one, mismatch)

    controller_features = vcat(
        rotate, rotate_increment, navigate
    )
    output_w = np.zeros((8, 3), np.float32)
    output_b = np.full(8, -100, np.float32)
    output_w[1] = [2, -1, 0]
    output_w[2] = [1, 1, 0]
    output_w[4] = [0, 0, 1]
    output_b[[1, 2, 4]] = 0
    dense(
        controller_features, output_w, output_b, name="logits"
    )
    node("slice", ["x0"], name="click", start=0, stop=1)

    return {
        "graph": root,
        "memory_size": 0,
        "params": params,
    }

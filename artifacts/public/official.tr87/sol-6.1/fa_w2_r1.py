import numpy as np


def build():
    graph = []
    params = {}
    shapes = {}
    parameter_cache = {}
    linear_cache = {}
    shift_cache = {}
    present_cache = {}
    bank_cache = {}
    counter = [0]
    current = [graph]

    def parameter(a):
        a = np.asarray(a, dtype=np.float32)
        key = (a.shape, a.tobytes())
        if key not in parameter_cache:
            name = "w%d" % len(params)
            params[name] = a
            parameter_cache[key] = name
        return parameter_cache[key]

    def node(op, inputs, channels, spatial=True, name=None, **fields):
        if name is None:
            name = "n%d" % counter[0]
            counter[0] += 1
        d = {"id": name, "op": op, "in": list(inputs)}
        d.update(fields)
        current[0].append(d)
        shapes[name] = (spatial, channels)
        return name

    def conv(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("conv", [x], w.shape[0], True, name, **fields)

    def dense(x, w, b=None, name=None):
        fields = {"w": parameter(w)}
        if b is not None:
            fields["b"] = parameter(b)
        return node("dense", [x], w.shape[0], False, name, **fields)

    def linear(x, a=1.0, b=0.0):
        key = (x, float(a), float(b))
        if key in linear_cache:
            return linear_cache[key]
        is_spatial, c = shapes[x]
        w = np.eye(c, dtype=np.float32) * a
        bias = np.full(c, b, dtype=np.float32) if b != 0 else None
        if is_spatial:
            result = conv(x, w[:, :, None, None], bias)
        else:
            result = dense(x, w, bias)
        linear_cache[key] = result
        return result

    def binary(op, a, b):
        sa, ca = shapes[a]
        sb, cb = shapes[b]
        is_spatial = sa or sb
        if not is_spatial:
            if op == "sub":
                return binary("add", a, linear(b, -1))
            op = {"add": "vadd", "mul": "vmul"}[op]
        return node(op, [a, b], max(ca, cb), is_spatial)

    def add(a, b):
        return binary("add", a, b)

    def sub(a, b):
        return binary("sub", a, b)

    def mul(a, b):
        return binary("mul", a, b)

    def total(xs):
        xs = list(xs)
        if not xs:
            return zero
        r = xs[0]
        for z in xs[1:]:
            r = add(r, z)
        return r

    def relu(x):
        s, c = shapes[x]
        return node("relu" if s else "vrelu", [x], c, s)

    def spatial(x):
        return x if shapes[x][0] else mul(one, x)

    def absolute(x):
        x = spatial(x)
        return node("abs", [x], shapes[x][1], True)

    def pool(x):
        x = spatial(x)
        return node("gmax", [x], shapes[x][1], False)

    def vslice(x, start, stop):
        return node("vslice", [x], stop - start, False,
                    start=start, stop=stop)

    def sslice(x, start, stop):
        return node("slice", [x], stop - start, True,
                    start=start, stop=stop)

    def complement(x):
        return linear(x, -1, 1)

    def present(x):
        if x not in present_cache:
            sx = spatial(x)
            present_cache[x] = node(
                "min2", [sx, one], shapes[sx][1], True)
        return present_cache[x]

    def equal(a, b, scale=1.0):
        if isinstance(b, (int, float)):
            d = linear(spatial(a), 1, -b)
        else:
            d = sub(spatial(a), b)
        return relu(linear(absolute(d), -scale, 1))

    def choose(m, a, b):
        return add(mul(m, a), mul(complement(m), b))

    def reduce_channels(x, weights=None):
        s, c = shapes[x]
        if weights is None:
            weights = np.ones(c, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32).reshape(1, c)
        return conv(x, w[:, :, None, None]) if s else dense(x, w)

    def constant_vector(values):
        v = np.asarray(values, dtype=np.float32)
        return dense(vone, np.zeros((len(v), 1), np.float32), v)

    def shift(x, dx=0, dy=0):
        key = (x, dx, dy)
        if key in shift_cache:
            return shift_cache[key]
        if dx == 0 and dy == 0:
            return x
        c = shapes[x][1]
        sx = max(-3, min(3, dx))
        sy = max(-3, min(3, dy))
        w = np.zeros((c, c, 7, 7), np.float32)
        for k in range(c):
            w[k, k, 3 + sy, 3 + sx] = 1
        q = conv(x, w)
        q = shift(q, dx - sx, dy - sy)
        shift_cache[key] = q
        return q

    def bank(x):
        if x not in bank_cache:
            bank_cache[x] = total([
                present(x),
                present(relu(linear(x, 1, -7))),
                present(relu(linear(x, 1, -14))),
            ])
        return bank_cache[x]

    def modulo_equal(a, b):
        d = absolute(sub(spatial(a), b))
        return add(
            relu(linear(d, -1, 1)),
            relu(linear(absolute(linear(d, 1, -7)), -1, 1))
        )

    def pick(match):
        score = mul(match, priority)
        best = pool(score)
        winner = mul(match, equal(score, best, 32768))
        return winner, pool(match)

    def gather(mask, value):
        return pool(mul(mask, value))

    shapes["x0"] = (True, 16)
    shapes["coords"] = (True, 3)
    shapes["aprev"] = (False, 8)
    shapes["avail"] = (False, 8)

    y = sslice("coords", 0, 1)
    x = sslice("coords", 1, 2)
    one = sslice("coords", 2, 3)
    zero = conv(one, np.zeros((1, 1, 1, 1), np.float32),
                name="click")
    vone = pool(one)
    priority = total([linear(y, -1, 2), linear(x, -1 / 128)])
    ink = sslice("x0", 5, 6)
    padding = sslice("x0", 3, 4)
    black = sslice("x0", 0, 1)

    # A1..A7, B1..B7, C1..C7.
    templates = [
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
    gw, gb = [], []
    for rows in templates:
        a = np.array([[int(z) for z in row] for row in rows],
                     dtype=np.float32)
        for rotation in range(4):
            ar = np.rot90(a, rotation)
            gw.append((2 * ar - 1)[None])
            gb.append(-float(ar.sum()) + 0.5)
    hits = linear(relu(conv(
        ink, np.stack(gw), np.array(gb, np.float32))), 2)

    orientation_sum = np.zeros((21, 84, 1, 1), np.float32)
    for k in range(21):
        orientation_sum[k, 4 * k:4 * k + 4, 0, 0] = 1
    hits = present(conv(hits, orientation_sum))

    background_weights = np.zeros((3, 16, 1, 1), np.float32)
    for k, colour in enumerate([10, 7, 11]):
        background_weights[k, colour, 0, 0] = 1
    backgrounds = conv("x0", background_weights)
    perimeter = np.ones((7, 7), np.float32)
    perimeter[1:6, 1:6] = 0
    pw = np.zeros((3, 3, 7, 7), np.float32)
    for k in range(3):
        pw[k, k] = perimeter
    borders = linear(relu(conv(
        backgrounds, pw, np.full(3, -23.5, np.float32))), 2)

    bw = np.zeros((21, 3, 1, 1), np.float32)
    for k in range(21):
        bw[k, k // 7, 0, 0] = 1
    glyphs = node("min2", [hits, conv(borders, bw)], 21, True)
    code = reduce_channels(glyphs, np.arange(1, 22))
    tile = present(reduce_channels(glyphs))

    # The colour-3 region at the left edge separates sequences from rules.
    left_edge = relu(linear(x, -63, 1))
    lower = node("rowmax", [mul(padding, left_edge)], 1, True)
    upper = complement(lower)
    bottom_tiles = mul(tile, lower)
    top_tiles = mul(tile, upper)
    source_score = mul(bottom_tiles, linear(y, -1, 1))
    source = mul(bottom_tiles,
                 equal(source_score, pool(source_score), 32768))
    target = sub(bottom_tiles, source)

    # Inclusive horizontal prefix count ranks each sequence's symbols.
    rw = np.zeros((1, 1, 3, 3), np.float32)
    rw[0, 0, 1, 0] = 1
    graph.append({
        "id": "Ranks", "op": "loop", "T": 64,
        "state": [{"id": "rank_state", "zeros": [1, 64, 64]}],
        "body": [
            {"id": "rank_shift", "op": "conv",
             "in": ["rank_state"], "w": parameter(rw)},
            {"id": "rank_new", "op": "add",
             "in": ["rank_shift", bottom_tiles]},
        ],
        "next": {"rank_state": "rank_new"}
    })
    shapes["Ranks.rank_state"] = (True, 1)

    N = 9
    slot_numbers = constant_vector(np.arange(1, N + 1))
    slots = equal("Ranks.rank_state", slot_numbers)
    source_tokens = pool(mul(slots, mul(source, code)))
    target_tokens = pool(mul(slots, mul(target, code)))
    target_bank = bank(vslice(target_tokens, 0, 1))

    bracket = shift(black, 0, -6)
    selected_top = mul(top_tiles, bracket)
    selected_bottom = mul(target, bracket)
    altered = pool(selected_top)

    # Connector pixels identify the end of a LHS and start of a RHS.
    last_left = mul(top_tiles, shift(padding, 4, 0))
    first_right = mul(top_tiles, shift(padding, -4, 0))
    left_members = mul(top_tiles, total([
        last_left, shift(last_left, 7), shift(last_left, 14)
    ]))
    right_members = mul(top_tiles, total([
        first_right, shift(first_right, -7), shift(first_right, -14)
    ]))
    left_codes = mul(code, left_members)
    right_codes = mul(code, right_members)
    previous1 = shift(left_codes, -7)
    previous2 = shift(left_codes, -14)
    lf3 = present(previous2)
    lf2 = sub(present(previous1), lf3)
    lf1 = complement(present(previous1))

    lhs = [
        total([mul(lf1, code), mul(lf2, previous1),
               mul(lf3, previous2)]),
        add(mul(lf2, code), mul(lf3, previous1)),
        mul(lf3, code),
    ]
    lhs_length = total([lf1, linear(lf2, 2), linear(lf3, 3)])
    rhs = [
        shift(right_codes, 10),
        shift(right_codes, 17),
        shift(right_codes, 24),
    ]
    rhs[1] = mul(rhs[1], present(rhs[0]))
    rhs[2] = mul(rhs[2], present(rhs[1]))
    rhs_length = total([present(z) for z in rhs])
    lhs_hash = total([
        lhs[0], linear(lhs[1], 32), linear(lhs[2], 1024)
    ])
    direct = equal(bank(rhs[0]), target_bank)

    # Single-symbol second-stage translation table.
    alphabet = constant_vector(np.arange(1, 22))
    table_match = mul(equal(lhs[0], alphabet), last_left)
    table_winner, _ = pick(table_match)
    table = [gather(table_winner, z) for z in rhs]

    translated = []
    translated_lengths = []
    for r in rhs:
        categorical = equal(r, alphabet)
        fields = [
            reduce_channels(mul(categorical, table[k]))
            for k in range(3)
        ]
        translated.append(fields)
        translated_lengths.append(total([present(z) for z in fields]))

    a, b, c = translated
    al, bl, cl = translated_lengths
    a_flags = {k: equal(al, k) for k in range(1, 4)}
    ab = add(al, bl)
    ab_flags = {k: equal(ab, k) for k in range(2, 7)}

    packed = []
    for j in range(9):
        terms = []
        if j < 3:
            terms.append(a[j])
        for k in range(1, 4):
            if 0 <= j - k < 3:
                terms.append(mul(a_flags[k], b[j - k]))
        for k in range(2, 7):
            if 0 <= j - k < 3:
                terms.append(mul(ab_flags[k], c[j - k]))
        packed.append(total(terms))

    effective_rhs = []
    for j in range(9):
        if j < 3:
            effective_rhs.append(choose(direct, rhs[j], packed[j]))
        else:
            effective_rhs.append(mul(complement(direct), packed[j]))
    effective_length = choose(
        direct, rhs_length, total(translated_lengths))

    pointer_cache = {}

    def pointer_shift(p, amount):
        key = (p, amount)
        if key not in pointer_cache:
            w = np.zeros((N, N), np.float32)
            if amount == 0:
                w[:] = np.eye(N, dtype=np.float32)
            elif amount < N:
                for k in range(N - amount):
                    w[k + amount, k] = 1
            pointer_cache[key] = dense(p, w)
        return pointer_cache[key]

    def read_token(tokens, p, offset=0):
        return reduce_channels(mul(tokens, pointer_shift(p, offset)))

    def indicators(length, count):
        if shapes[length][0]:
            length = pool(length)
        d = dense(length, np.ones((count, 1), np.float32),
                  -np.arange(1, count + 1, dtype=np.float32))
        av = add(relu(d), relu(linear(d, -1)))
        return relu(linear(av, -1, 1))

    def advance(p, length, count):
        flags = indicators(length, count)
        return total([
            mul(pointer_shift(p, k), vslice(flags, k - 1, k))
            for k in range(1, count + 1)
        ])

    def exact_lhs(samples):
        h1 = samples[0]
        h2 = add(h1, linear(samples[1], 32))
        h3 = add(h2, linear(samples[2], 1024))
        wanted = total([
            mul(lf1, h1), mul(lf2, h2), mul(lf3, h3)
        ])
        return equal(lhs_hash, wanted)

    def structural(fields, samples):
        result = equal(bank(fields[0]), bank(samples[0]))
        for j in range(1, 3):
            needed = present(fields[j])
            relative_match = modulo_equal(
                sub(fields[j], fields[0]),
                sub(samples[j], samples[0]))
            relative_match = mul(relative_match, present(samples[j]))
            result = mul(result, add(
                complement(needed), mul(needed, relative_match)))
        return result

    initial_pointer = constant_vector([1] + [0] * (N - 1))
    initial_output = constant_vector([0] * N)

    # Cache entries produced inside a loop must remain loop-local.
    scoped_caches = [
        linear_cache, shift_cache, present_cache,
        bank_cache, pointer_cache
    ]
    outer_cache_snapshots = [dict(cache) for cache in scoped_caches]

    body = []
    current[0] = body
    for name in ["src_pointer", "dst_pointer", "output_state"]:
        shapes[name] = (False, N)
    shapes["goal_left"] = (True, 1)
    shapes["goal_right"] = (True, 1)

    sv = [read_token(source_tokens, "src_pointer", k)
          for k in range(3)]
    tv = [read_token(target_tokens, "dst_pointer", k)
          for k in range(3)]

    ordinary_match = exact_lhs(sv)
    altered_match = mul(structural(lhs, sv), structural(rhs, tv))
    used = present("goal_left")
    compatible = mul(
        equal("goal_left", sv[0]), equal("goal_right", tv[0]))
    allowed = add(complement(used), mul(used, compatible))
    altered_match = mul(altered_match, allowed)
    matches = mul(
        last_left, choose(altered, altered_match, ordinary_match))
    winner, found = pick(matches)

    left_len = gather(winner, lhs_length)
    raw = [gather(winner, z) for z in rhs]
    raw_len = gather(winner, rhs_length)
    base_output = [gather(winner, z) for z in effective_rhs]
    base_length = gather(winner, effective_length)
    is_direct = gather(winner, direct)

    new_goal_left = node("max2", [
        "goal_left", mul(mul(winner, sv[0]), altered)
    ], 1, True)
    new_goal_right = node("max2", [
        "goal_right", mul(mul(winner, tv[0]), altered)
    ], 1, True)

    # Also allow a second-stage grouped rule to consume the RHS
    # of this rule, optionally followed by one adjacent source rule.
    continuation_pointer = advance("src_pointer", left_len, 3)
    continuation = [
        read_token(source_tokens, continuation_pointer, k)
        for k in range(3)
    ]

    prefix_match = equal(lhs[0], raw[0])
    for j in range(1, 3):
        needed = pool(present(raw[j]))
        prefix_match = mul(prefix_match, add(
            complement(needed), mul(needed, equal(lhs[j], raw[j]))))
    long_enough = present(relu(linear(
        sub(lhs_length, raw_len), 1, 1)))
    converter_match = mul(last_left, mul(
        prefix_match, mul(long_enough, direct)))
    converter, converter_exists = pick(converter_match)

    converter_lhs = [gather(converter, z) for z in lhs]
    converter_rhs = [gather(converter, z) for z in rhs]
    converter_lhs_len = gather(converter, lhs_length)
    converter_rhs_len = gather(converter, rhs_length)
    remaining = sub(converter_lhs_len, raw_len)
    rem_zero = pool(equal(remaining, 0))
    raw_one = pool(equal(raw_len, 1))
    raw_two = pool(equal(raw_len, 2))
    missing1 = add(
        mul(raw_one, converter_lhs[1]),
        mul(raw_two, converter_lhs[2]))
    missing2 = mul(raw_one, converter_lhs[2])

    partner_match = mul(last_left, exact_lhs(continuation))
    partner_match = mul(partner_match, equal(rhs[0], missing1))
    partner_match = mul(partner_match, equal(rhs[1], missing2))
    partner_match = mul(partner_match, equal(rhs_length, remaining))
    partner, partner_exists = pick(partner_match)
    partner_left_len = gather(partner, lhs_length)

    composition_valid = mul(
        converter_exists,
        mul(complement(is_direct), add(rem_zero, partner_exists)))
    composition_valid = mul(composition_valid, complement(altered))

    output_fields = []
    for j in range(9):
        if j < 3:
            output_fields.append(choose(
                composition_valid, converter_rhs[j], base_output[j]))
        else:
            output_fields.append(mul(
                complement(composition_valid), base_output[j]))
    output_length = choose(
        composition_valid, converter_rhs_len, base_length)
    consumed_left = add(
        left_len, mul(composition_valid, partner_left_len))

    next_src = advance("src_pointer", consumed_left, 6)
    next_dst = advance("dst_pointer", output_length, 9)
    emitted = total([
        mul(pointer_shift("dst_pointer", j), output_fields[j])
        for j in range(9)
    ])
    next_output = add("output_state", emitted)

    current[0] = graph
    for cache, snapshot in zip(scoped_caches, outer_cache_snapshots):
        cache.clear()
        cache.update(snapshot)

    graph.append({
        "id": "Program", "op": "loop", "T": N,
        "state": [
            {"id": "src_pointer", "init": initial_pointer},
            {"id": "dst_pointer", "init": initial_pointer},
            {"id": "output_state", "init": initial_output},
            {"id": "goal_left", "init": zero},
            {"id": "goal_right", "init": zero},
        ],
        "body": body,
        "next": {
            "src_pointer": next_src,
            "dst_pointer": next_dst,
            "output_state": next_output,
            "goal_left": new_goal_left,
            "goal_right": new_goal_right,
        }
    })
    shapes["Program.output_state"] = (False, N)
    shapes["Program.goal_left"] = (True, 1)
    shapes["Program.goal_right"] = (True, 1)

    desired_bottom = reduce_channels(
        mul(slots, "Program.output_state"))
    desired_bottom = mul(desired_bottom, target)

    ordinary_goal = gather(selected_bottom, desired_bottom)
    ordinary_current = gather(selected_bottom, code)
    ordinary_difference = mul(
        sub(ordinary_goal, ordinary_current),
        pool(present(ordinary_goal)))
    ordinary_errors = mul(
        target, mul(present(desired_bottom),
                    present(absolute(sub(desired_bottom, code)))))
    ordinary_any_error = pool(ordinary_errors)

    # Align the first-glyph bracket with the last-glyph LHS anchor.
    selected_l = mul(last_left, total([
        selected_top,
        shift(selected_top, -7),
        shift(selected_top, -14),
    ]))
    selected_r = mul(selected_top, first_right)
    right_goals_at_start = shift("Program.goal_right", -10)
    alter_goal = add(
        gather(selected_l, "Program.goal_left"),
        gather(selected_r, right_goals_at_start))
    alter_current = add(
        gather(selected_l, lhs[0]), gather(selected_r, code))
    alter_difference = mul(
        sub(alter_goal, alter_current), pool(present(alter_goal)))

    left_errors = mul(
        present("Program.goal_left"),
        present(absolute(sub("Program.goal_left", lhs[0]))))
    right_errors = mul(
        present("Program.goal_right"),
        present(absolute(sub("Program.goal_right", rhs[0]))))
    alter_any_error = pool(node(
        "max2", [left_errors, right_errors], 1, True))

    difference = choose(
        altered, alter_difference, ordinary_difference)
    any_error = choose(
        altered, alter_any_error, ordinary_any_error)

    # Piecewise-linear shortest direction on the seven-symbol cycle.
    relay_w = np.array([[1], [1], [1], [1], [-1], [-1]],
                       dtype=np.float32)
    relay_b = np.array([1, -1, -3, -4, -3, -4],
                       dtype=np.float32)
    hinges = relu(dense(difference, relay_w, relay_b))
    direction = dense(
        hinges,
        np.array([[1, -1, -2, 2, 2, -2]], dtype=np.float32),
        np.array([-1], dtype=np.float32))

    final_features = node(
        "vconcat", [direction, any_error], 2, False)
    action_w = np.zeros((8, 2), np.float32)
    action_b = np.full(8, -10, np.float32)
    action_w[1, 0] = -2
    action_b[1] = 0
    action_w[2] = [2, -3]
    action_b[2] = 3
    action_b[4] = 1
    dense(final_features, action_w, action_b, name="logits")

    return {
        "graph": graph,
        "memory_size": 0,
        "params": params,
    }

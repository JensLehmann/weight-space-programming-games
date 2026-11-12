import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128),
        "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}

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
    glyphs = [
        np.array([[int(c) for c in row] for row in rows], np.int32)
        for rows in patterns
    ]

    # Rotation-invariant orbit-population hinges.
    orbits = []
    for r, c in [(0, 1), (1, 1), (0, 2), (1, 2), (2, 1)]:
        orbit = []
        for _ in range(4):
            orbit.append((r, c))
            r, c = -c, r
        orbits.append(orbit)

    for o, orbit in enumerate(orbits):
        for t in range(4):
            ch = 4 * o + t
            for r, c in orbit:
                p["W1"][ch, 5, r + 2, c + 2] = 1
            p["b1"][ch] = -t

    # Tile-border checks.
    p["W1"][20, 5, 0, :] = 1
    p["W1"][21, 5, 4, :] = 1
    p["W1"][22, 5, :, 0] = 1
    p["W1"][23, 5, :, 4] = 1

    p["W1"][24, 17, 2, 2] = 63
    p["W1"][25, 7, 0, 0] = 1
    p["W1"][26, 11, 0, 0] = 1
    p["W1"][27, 0, 2, :] = 1
    p["W1"][28, 16, 2, 2] = 63
    p["W1"][29, 0, 2, :] = 1
    p["W1"][29, 0, 3, 0] = 1
    p["b1"][29] = -5
    p["W1"][30, 2, 2, 2] = 1
    p["W1"][31, 5, 2, 2] = 1

    for idx, glyph in enumerate(glyphs):
        family = idx // 7
        bias = 1.0
        for o, orbit in enumerate(orbits):
            q = sum(int(glyph[r + 2, c + 2]) for r, c in orbit)
            p["W2"][idx, 4 * o, 1, 1] += 1
            if q < 4:
                p["W2"][idx, 4 * o + q, 1, 1] -= 2
            bias -= q

        p["W2"][idx, 20, 0, 1] = -1
        p["W2"][idx, 21, 2, 1] = -1
        p["W2"][idx, 22, 1, 0] = -1
        p["W2"][idx, 23, 1, 2] = -1

        if family == 0:
            p["W2"][idx, 25, 0, 0] = -1
            p["W2"][idx, 26, 0, 0] = -1
        else:
            p["W2"][idx, 24 + family, 0, 0] = 1
            bias -= 1

        if glyph[2, 2]:
            p["W2"][idx, 31, 1, 1] = 1
            bias -= 1
        else:
            p["W2"][idx, 31, 1, 1] = -1
        p["b2"][idx] = bias

    for ch, weight, bias in [
        (21, 1, -32), (22, -1, 39),
        (23, 1, -49), (24, -1, 50),
    ]:
        p["W2"][ch, 28, 1, 1] = weight
        p["b2"][ch] = bias

    # A slightly right-shifted cursor band still includes aligned
    # symbols, and includes the next input symbol on staggered rows.
    p["W2"][25, 27, 1, 0] = 1
    p["b2"][25] = -1
    p["W2"][27, 27, 1, 0] = 1
    p["b2"][27] = -2
    p["W2"][26, 30, 1, 1] = 1

    coord_eps = 1e-4
    p["W2"][28, 29, 1, 1] = 1
    p["W2"][28, 28, 1, 1] = coord_eps
    p["b2"][28] = -0.5

    p["W2"][29, 24, 1, 1] = 1
    p["W2"][30, 24, 1, 1] = 1
    p["b2"][30] = -33

    p["W2"][31, 29, 1, 1] = 1
    p["W2"][31, 24, 1, 1] = coord_eps
    p["b2"][31] = -0.5

    def cursor_gate(ch, scale):
        p["W3"][ch, 64 + 25] += scale
        p["W3"][ch, 64 + 27] -= scale

    for j in range(7):
        for family in range(3):
            idx = 7 * family + j
            p["W3"][j, idx] = 64 * (family + 1)
            packed = 1 + 8 * family
            p["W3"][7 + j, idx] = packed
            p["W3"][14 + j, idx] = packed

        p["W3"][j, 21] = -192

        cursor_gate(7 + j, 17)
        p["W3"][7 + j, 22:24] = -17
        p["b3"][7 + j] = -17

        cursor_gate(14 + j, 17)
        p["W3"][14 + j, 24] = -17
        p["b3"][14 + j] = -17

    # Only A/B glyphs terminate rule records. In particular, an
    # internal C glyph is not mistaken for the result of a C rule.
    for family in range(2):
        for j in range(7):
            idx = 7 * family + j
            p["W3"][21, idx] = 8 * family + j + 1
            p["W3"][22, idx] = 1
    p["W3"][23, 26] = 1

    gate = 20000.0
    for ch, source, left in [
        (24, True, True), (25, True, False),
        (26, False, True), (27, False, False),
    ]:
        for family in range(3):
            for j in range(7):
                p["W3"][ch, 7 * family + j] = gate + j + 1
        p["W3"][ch, 29] = -100 if left else 100
        p["b3"][ch] = -gate + (6300 if left else 0)
        if source:
            p["W3"][ch, 22:24] = -gate
        else:
            p["W3"][ch, 24] = -gate

    # The middle reader accepts the centered odd-row position and
    # the right middle position of an even row.
    for ch, source in [(28, True), (29, False)]:
        for family in range(3):
            for j in range(7):
                p["W3"][ch, 7 * family + j] = gate + j + 1
        p["W3"][ch, 29] = 100
        p["W3"][ch, 30] = -200
        p["b3"][ch] = -gate - 3100
        if source:
            p["W3"][ch, 22:24] = -gate
        else:
            p["W3"][ch, 24] = -gate

    p["W3"][30, 28] = 2
    p["W3"][31, 31] = 2

    # Horizontal dictionary-key transport and record emission.
    for j in range(7):
        p["Wh"][j, j, 1, 0] = 1
        p["Uh"][j, j] = 1
        p["Uh"][j, 23] = -10000

        p["Wh"][7 + j, j, 1, 0] = 1
        p["Uh"][7 + j, 21] = 1
        p["Uh"][7 + j, 22] = 10000
        p["bh"][7 + j] = -10064

    # Cursor-to-selected-glyph wave.
    p["Wh"][14, 14, 0, 1] = 1
    p["Uh"][14, 30] = 8
    p["bh"][14] = -1
    p["Wh"][15, 14, 0, 1] = 100
    p["Uh"][15, 21] = 1
    p["Uh"][15, 22] = 10000
    p["bh"][15] = -10200

    next_z = 0

    def add_z(weights=None, bias=0.0):
        nonlocal next_z
        i = next_z
        next_z += 1
        if weights is not None:
            p["Wz"][i] = weights
        p["bz"][i] = bias
        return i

    def e(i, size=128):
        v = np.zeros(size, np.float32)
        v[i] = 1
        return v

    def gf(ch, mean=False):
        return 64 + ch if mean else 32 + ch

    input_sum = np.zeros(96, np.float32)
    output_sum = np.zeros(96, np.float32)
    input_sum[39:46] = 1
    output_sum[46:53] = 1

    def record(j, lower=False):
        v = np.zeros(96, np.float32)
        if lower:
            v[23 + j] = 4096
            v[7 + j] = -1
        else:
            v[7 + j] = 1
        return v

    # Ordinary one-symbol dictionary lookup.
    first_target = np.zeros(128, np.float32)
    for j in range(7):
        base = record(j)
        base -= 8 * input_sum
        base -= 2048 * input_sum
        base[gf(7 + j)] += 2048
        units = [
            add_z(base, 8 - knot)
            for knot in [0, 1, 23, 24]
        ]
        for unit, coefficient in zip(units, [1, 0, -24, 23]):
            first_target[unit] += coefficient

    constant = e(add_z(bias=1))

    # Centered-string endpoint decoder.
    def endpoints(left_ch, right_ch):
        left = e(add_z(e(gf(left_ch), 96), -4600))
        right = e(add_z(e(gf(right_ch), 96), -4500))
        count = 5 * constant.copy()
        previous_left, previous_right = 4600, 4500
        summed = e(gf(left_ch), 96) + e(gf(right_ch), 96)

        for n in range(6, 9):
            boundary = 5600 + 700 * n
            a = add_z(summed / 50, -(boundary - 100) / 50)
            b = add_z(summed / 50, -(boundary - 50) / 50)
            step = e(a) - e(b)
            parity = n % 2
            left_base = 2800 + 350 * n + 50 * parity
            right_base = 2800 + 350 * n - 50 * parity
            left -= (left_base - previous_left) * step
            right -= (right_base - previous_right) * step
            count += step
            previous_left, previous_right = left_base, right_base

        return left, right, count

    src_first, src_last, src_count = endpoints(24, 25)
    dst_first, dst_last, dst_count = endpoints(26, 27)
    src_middle = e(add_z(e(gf(28), 96)))
    dst_middle = e(add_z(e(gf(29), 96)))

    def total_ids(left_ch, right_ch, count):
        v = np.zeros(96, np.float32)
        v[gf(left_ch, True)] = 2048
        v[gf(right_ch, True)] = 2048
        return e(add_z(v)) - 3150 * count

    src_total = total_ids(24, 25, src_count)
    dst_total = total_ids(26, 27, dst_count)
    src_residual = src_total - src_first - src_last - 2 * src_middle
    dst_residual = dst_total - dst_first - dst_last - 2 * dst_middle

    def coordinate_step(channel, low, high):
        scale = 1.0 / (2 * coord_eps)
        v = e(gf(channel), 96) * (scale / (high - low))
        a = add_z(v, (-scale - low) / (high - low))
        b = add_z(v, (-scale - high) / (high - low))
        return e(a) - e(b)

    lower_rule_row = coordinate_step(30, 15, 17)
    right_rule_column = coordinate_step(31, 31, 33)
    noneditable = coordinate_step(30, 31, 33)
    rule_ordinal = 2 * lower_rule_row + right_rule_column

    v = e(15, 96)
    v[gf(30)] = -800
    current_code = e(add_z(v, 800))
    output_family = e(add_z(output_sum, -1))
    input_family = e(add_z(input_sum, -1))

    side_a = add_z(v, 800 - 7.5)
    side_b = add_z(v, 800 - 8.5)
    selected_side = e(side_a) - e(side_b)

    # C-rule values: strip either the one-occurrence or two-occurrence
    # key header, without embedding any particular output glyph.
    c_values = []
    for j in range(7):
        value = np.zeros(128, np.float32)
        for header in [128, 320]:
            units = [
                add_z(record(j), -header - knot)
                for knot in [0, 16, 32]
            ]
            for u, coefficient in zip(units, [1, -2, 1]):
                value[u] += coefficient
        c_values.append(value)

    # Difference between the other output record and the largest one.
    c_deltas = []
    for j in range(7):
        delta = record(j, lower=True) - record(j)
        c_deltas.append(e(add_z(delta, 1000)) - 1000 * constant)

    # Cursor coordinate in units of a seven-pixel output cell.
    cursor_x_g = e(gf(31), 96) / (2 * coord_eps)
    cursor_x_bias = -1.0 / (2 * coord_eps)
    cursor_index_g = cursor_x_g / 7
    cursor_index_bias = (cursor_x_bias - 10) / 7
    cursor_index = (
        e(add_z(cursor_x_g, cursor_x_bias)) - 10 * constant
    ) / 7

    ordinary_query = np.zeros(96, np.float32)
    for j in range(7):
        ordinary_query[gf(7 + j)] = (j + 1) / 17.0

    first_query = e(gf(24), 96)
    last_query = e(gf(25), 96)
    middle_query = e(gf(28), 96)

    c_query = np.zeros(128, np.float32)

    def isolated_query(position, query, query_bias=0.0):
        base = 100 * cursor_index_g + query
        bias = 100 * (cursor_index_bias - position) + query_bias
        units = [add_z(base, bias - t) for t in [0, 20, 40]]
        for u, coefficient in zip(units, [1, -2, 1]):
            c_query[u] += coefficient

    def adjacent_queries(position, query, query_bias=0.0):
        address = 100 * cursor_index_g
        bias = 100 * (cursor_index_bias - position)
        a = add_z(address + query, bias + query_bias)
        b = add_z(address, bias)
        c = add_z(address + query, bias - 150 + query_bias)
        d = add_z(address, bias - 150)
        c_query[:] += e(a) - e(b) - e(c) + e(d)

    # Contraction/expansion alignment fast path.
    isolated_query(0, first_query, -5600)
    isolated_query(1, ordinary_query)
    adjacent_queries(2, middle_query)
    adjacent_queries(4, ordinary_query)
    isolated_query(6, last_query, -5600)

    assert next_z <= 128

    p["bo"][:] = -10000
    p["bo"][1] = 0
    p["bo"][2] = 0
    p["bo"][4] = 10.25

    def add_error_mem(i, coefficient):
        p["Wo"][2, 128 + i] += coefficient
        p["Wo"][1, 128 + i] -= coefficient

    sharp = 16.0
    small = 0.002
    direct_query = first_target - output_family

    for j in range(1, 8):
        a, b = 2 * (j - 1), 2 * (j - 1) + 1
        p["Wmz"][a] = sharp * direct_query
        p["Wmz"][b] = sharp * direct_query
        p["bm"][a] = sharp * (-j + 0.5)
        p["bm"][b] = sharp * (-j - 0.5)
        add_error_mem(a, j / 2)
        add_error_mem(b, -j / 2)
        p["Wo"][4, 128 + a] -= 5
        p["Wo"][4, 128 + b] += 5

    current_id = current_code - output_family
    p["Wo"][2, :128] -= current_id
    p["Wo"][1, :128] += current_id

    # Paired tanh units gate approximately linear payloads.
    c_guard = 16 * constant - input_family
    for j in range(1, 8):
        a = 14 + 2 * (j - 1)
        b = a + 1
        address = c_query - j * constant + 10 * c_guard
        p["Wmz"][a] = (
            8 * address + small * (c_values[j - 1] - current_code)
        )
        p["Wmz"][b] = 8 * address
        add_error_mem(a, 50 / small)
        add_error_mem(b, -50 / small)

        a = 28 + 2 * (j - 1)
        b = a + 1
        address = (
            c_query - j * constant
            + 10 * (cursor_index - 6 * constant)
            + 1000 * c_guard
        )
        p["Wmz"][a] = 8 * address + small * c_deltas[j - 1]
        p["Wmz"][b] = 8 * address
        add_error_mem(a, 50 / small)
        add_error_mem(b, -50 / small)

    # Editable paired-rule controller.
    source_roles = [src_first, src_residual, src_middle, src_last]
    target_roles = [dst_first, dst_middle, dst_residual, dst_last]
    address = 8 * rule_ordinal + selected_side + 100 * noneditable

    unit = 42
    for rule in range(4):
        for side, desired in [
            (0, source_roles[rule]),
            (1, target_roles[rule]),
        ]:
            a, b = unit, unit + 1
            unit += 2
            error = desired - current_code
            p["Wmz"][a] = sharp * address + small * error
            p["Wmz"][b] = sharp * address
            p["bm"][a] = -sharp * (8 * rule + side) + small * 8 * side
            p["bm"][b] = -sharp * (8 * rule + side)
            add_error_mem(a, 50 / small)
            add_error_mem(b, -50 / small)

    p["bo"][4] += 20
    p["Wo"][4, :128] -= 20 * noneditable
    p["Wo"][4, :128] += (20 / 16) * input_family

    return p

import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5),
        "b1": (32,),
        "W2": (32, 32, 3, 3),
        "b2": (32,),
        "W3": (32, 96),
        "b3": (32,),
        "Wh": (16, 16, 3, 3),
        "Uh": (16, 32),
        "bh": (16,),
        "Wz": (128, 96),
        "bz": (128,),
        "Wm": (64, 64),
        "Wmz": (64, 128),
        "Wma": (64, 8),
        "bm": (64,),
        "Wo": (8, 192),
        "bo": (8,),
        "Wc": (1, 48),
        "bc": (1,),
    }
    p = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    denominator = 65536.0
    c_offset = 1.0 / 64.0
    auxiliary_scale = 1.0e-6

    # Rotation-invariant orbit fingerprint.
    def orbit(y, x):
        ay, ax = abs(y), abs(x)
        if ay == 0 and ax == 0:
            return 0
        if max(ay, ax) == 1:
            return 1 if min(ay, ax) == 0 else 2
        if min(ay, ax) == 0:
            return 3
        if ay == 2 and ax == 2:
            return 4
        return 5

    orbit_weights = (1.0, 3.0, 7.0, 13.0, 23.0, 41.0)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            W1[0, 5, dy + 2, dx + 2] = (
                orbit_weights[orbit(dy, dx)] / denominator
            )

    corners = [(-2, -2), (-2, 2), (2, 2), (2, -2)]
    for k in range(4):
        for dy, dx in (corners[k], corners[(k + 1) % 4]):
            W1[k + 1, 5, dy + 2, dx + 2] = 1.0
        b1[k + 1] = -1.0

    tile_colors = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    lines = [
        [(-2, x) for x in range(-2, 3)],
        [(2, x) for x in range(-2, 3)],
        [(y, -2) for y in range(-2, 3)],
        [(y, 2) for y in range(-2, 3)],
    ]
    for k, line in enumerate(lines):
        for dy, dx in line:
            for color in tile_colors:
                W1[5 + k, color, dy + 2, dx + 2] = 1.0

    # Family probes.
    W1[9, 10, 0, 0] = 1.0
    W1[9, 11, 0, 0] = 1.0
    W1[10, 7, 0, 0] = 1.0
    W1[11, 11, 0, 0] = 1.0
    W1[12, 10, 0, 0] = 1.0

    # Selection-bracket endpoint.
    for dx in range(-2, 3):
        W1[13, 0, 0, dx + 2] = 1.0
    W1[13, 0, 1, 0] = 1.0
    b1[13] = -5.0

    W1[15, 17, 2, 2] = 1.0

    def hinge(channel, coordinate, sign, threshold):
        W1[channel, coordinate, 2, 2] = 63.0 * sign
        b1[channel] = -sign * threshold

    hinge(17, 16, +1, 11)
    hinge(18, 16, -1, 12)
    hinge(19, 16, +1, 20)
    hinge(20, 16, -1, 21)
    hinge(21, 16, +1, 33)

    hinge(22, 17, +1, 26)
    hinge(23, 17, -1, 27)

    hinge(24, 16, -1, 36)
    hinge(25, 16, +1, 49)
    hinge(26, 16, -1, 50)
    hinge(27, 16, +1, 59)

    hinge(28, 17, +1, 39)
    hinge(29, 17, -1, 40)

    # Exact orbit-count separators for four C-family glyph classes.
    # These recognize appearance; destinations are still read from
    # the displayed rules.
    def c_separator(channel, coefficients, threshold):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                W1[channel, 5, dy + 2, dx + 2] = (
                    coefficients[orbit(dy, dx)]
                )
                for color in range(16):
                    if color not in (5, 11):
                        W1[channel, color, dy + 2, dx + 2] = -10.0
        b1[channel] = -float(threshold)

    c_separator(14, (0, -3, 0, 3, 1, -1), 11)  # C1
    c_separator(16, (1, 0, 0, 0, -1, 0), 0)    # C4
    c_separator(30, (0, 0, 0, 0, 0, 1), 7)     # C5
    c_separator(31, (0, 3, 0, 0, 1, -3), 3)    # C3

    border_terms = [
        (5, -1, -1), (5, -1, 1),
        (6, 1, -1), (6, 1, 1),
        (7, -1, -1), (7, 1, -1),
        (8, -1, 1), (8, 1, 1),
    ]

    def glyph_feature(out, gates=(), family=None):
        b2[out] = 1.0 - 80.0
        W2[out, 0, 1, 1] = 1.0
        for c in range(1, 5):
            W2[out, c, 1, 1] = 61.0 / denominator
        W2[out, 11, 0, 0] += c_offset

        for c, dy, dx in border_terms:
            W2[out, c, dy + 1, dx + 1] += 2.0
        for c in gates:
            W2[out, c, 1, 1] -= 2.0

        if family is not None:
            W2[out, family, 0, 0] += 2.0
            b2[out] -= 2.0

    row_gates = [(17,), (18, 19), (20, 21)]
    for r in range(6):
        band, half = divmod(r, 2)
        glyph_feature(
            r, row_gates[band] + (22 + half,), family=9
        )
        glyph_feature(
            6 + r, row_gates[band] + (28 + half,), family=10
        )

    glyph_feature(12, (24, 25))
    glyph_feature(13, (26, 27))

    W2[14, 13, 0, 1] = 1.0

    glyph_feature(15, (21, 28), family=12)
    glyph_feature(16, (21, 29), family=12)

    # Six spatial bins for the lower strings.
    boundaries = (17, 24, 31, 38, 45)
    for k, boundary in enumerate(boundaries):
        lo, hi = 17 + 2 * k, 18 + 2 * k
        W2[lo, 15, 1, 1] = -63.0
        b2[lo] = float(boundary)
        W2[hi, 15, 1, 1] = 63.0
        b2[hi] = -float(boundary - 1)

    # Sparse, border-validated class markers on the lower source row.
    for out, source in ((27, 14), (28, 16), (29, 30), (30, 31)):
        W2[out, source, 1, 1] = 1.0
        b2[out] = -40.0 * 32.0
        for c, dy, dx in border_terms:
            W2[out, c, dy + 1, dx + 1] += 32.0
        W2[out, 24, 1, 1] = -32.0
        W2[out, 25, 1, 1] = -32.0
    W2[30, 15, 1, 1] += 63.0 / 128.0

    # Axial rule fields. Tiny A-destination fields share the B fields.
    # Their row gate is derived from the corresponding source field.
    for r in range(6):
        W3[r, r] = 1.0
        _, half = divmod(r, 2)
        out = 6 + r
        W3[out, out] = 1.0
        W3[out, 15 + half] = auxiliary_scale
        W3[out, 32 + r] = auxiliary_scale
        b3[out] = -1.5 * auxiliary_scale

    for slot in range(6):
        for out, source in ((12 + slot, 12), (18 + slot, 13)):
            W3[out, source] = 1.0
            if slot:
                W3[out, 17 + 2 * (slot - 1)] = -2.0
            if slot < 5:
                W3[out, 18 + 2 * slot] = -2.0

    W3[24, 64 + 14] = 1.0
    W3[25, 32 + 14] = 1.0

    W3[27, 13] = 1.0
    W3[27, 64 + 14] = 2.0
    b3[27] = -2.0

    W3[28, 27] = 0.5
    W3[29, 28] = 1.0
    W3[30, 29] = 1.0
    W3[31, 30] = 1.0

    def recurrent(out, source, dy=0, dx=0, weight=1.0):
        Wh[out, source, dy + 1, dx + 1] += weight

    # Effective source-prefix length. The C-family adjustment is a
    # palindrome-compression/singleton-expansion alignment heuristic.
    recurrent(0, 0, dx=-1)
    Uh[0, 12:18] = 1.0
    Uh[0, 28] = -1.0
    Uh[0, 29] = 1.0
    Uh[0, 30] = -1.0

    recurrent(1, 1, dx=-1)
    Uh[1, 18:24] = 1.0

    recurrent(2, 1)
    Uh[2, 24] = 16.0
    bh[2] = -16.0

    # Broadcast the selected output ordinal across its row.
    recurrent(3, 2)
    recurrent(3, 3, dx=-1, weight=-1.0)
    recurrent(3, 4, dx=-1, weight=-1.0)
    recurrent(4, 3, dx=-1)
    recurrent(4, 4, dx=-1)

    recurrent(5, 2)
    recurrent(5, 5, dx=1, weight=-1.0)
    recurrent(5, 6, dx=1, weight=-1.0)
    recurrent(6, 5, dx=1)
    recurrent(6, 6, dx=1)

    recurrent(7, 7, dy=1)
    for c in (3, 4, 5, 6):
        recurrent(7, c)
    recurrent(7, 2, weight=-1.0)

    # Interval rather than point matching of source/output ordinals.
    recurrent(8, 0)
    recurrent(8, 7, weight=-1.0)
    Uh[8, 12:18] = -1.0
    Uh[8, 28] = 1.0
    Uh[8, 29] = -1.0
    Uh[8, 30] = 1.0
    bh[8] = 0.3

    recurrent(9, 7)
    recurrent(9, 0, weight=-1.0)
    bh[9] = -0.35

    Uh[10, 12:18] = 1.0
    recurrent(10, 8, weight=-20.0)
    recurrent(10, 9, weight=-20.0)

    # Upper selection row and selected upper-group identity.
    Uh[12, 25] = 1.0
    recurrent(13, 12, dy=-1)
    recurrent(14, 13, dy=-1)

    Uh[11, :12] = 1.0
    Uh[11, 24] = 2.0
    recurrent(11, 14, dy=-1, weight=2.0)
    bh[11] = -4.0

    for r in range(6):
        Uh[15, r] = float(2 * r + 1)
        Uh[15, 6 + r] = float(2 * r + 2)
    Uh[15, 24] = 16.0
    recurrent(15, 14, dy=-1, weight=16.0)
    bh[15] = -32.0

    # g = [h_max, h_mean, f3_max, f3_mean].
    def expression(entries=(), constant=0.0):
        v = np.zeros(96, dtype=np.float32)
        for index, coefficient in entries:
            v[index] += coefficient
        return v, float(constant)

    def difference(a, b):
        return a[0] - b[0], a[1] - b[1]

    def put_z(index, expr, offset=0.0):
        p["Wz"][index] = expr[0]
        p["bz"][index] = expr[1] + offset

    def put_abs_pair(index, expr, tolerance):
        put_z(index, expr, -tolerance)
        put_z(index + 1, (-expr[0], -expr[1]), -tolerance)

    query_source = expression(((10, 1.0),))
    query_output = expression(((32 + 27, 1.0),))
    query_upper = expression(((11, 1.0),))
    tolerance = 1.0e-5

    for r in range(6):
        lhs = expression(((32 + r, 1.0),))
        rhs_b = expression(((32 + 6 + r, 1.0),))
        rhs_a = expression((
            (32 + 6 + r, 1.0 / auxiliary_scale),
            (32 + r, -1.0),
        ), 1.5)

        put_abs_pair(
            4 * r, difference(query_source, lhs), tolerance
        )
        put_abs_pair(
            4 * r + 2, difference(query_output, rhs_b), tolerance
        )
        put_abs_pair(
            24 + 2 * r, difference(query_output, rhs_a), tolerance
        )

        outer = expression((
            (64 + r, 2048.0),
            (32 + r, -0.5),
        ))
        put_abs_pair(
            36 + 2 * r, difference(query_source, outer), 2.0e-5
        )

    # Rule-editing group selection and lower-string binding.
    for group in range(12):
        selector = expression((
            (15, 1.0),
            (11, -float(group + 1)),
        ))
        put_abs_pair(48 + 2 * group, selector, 1.0e-4)

        rule, side = divmod(group, 2)
        entries = []
        for earlier in range(rule):
            channel = earlier if side == 0 else 6 + earlier
            entries.append((64 + channel, 4096.0))
        put_z(
            72 + group,
            expression(entries, 1.0 + 16.0 * side),
        )

    candidates = (
        [(0, slot) for slot in range(1, 6)] +
        [(1, slot) for slot in range(1, 6)]
    )
    for candidate, (family, slot) in enumerate(candidates):
        channel = 12 + 6 * family + slot
        lower = expression(((32 + channel, 1.0),))
        put_abs_pair(
            84 + 2 * candidate,
            difference(query_upper, lower),
            tolerance,
        )

    source_count = expression(
        ((64 + c, 4096.0) for c in range(12, 18))
    )
    output_count = expression(
        ((64 + c, 4096.0) for c in range(18, 24))
    )
    put_z(104, source_count)
    put_z(105, output_count)
    put_z(107, query_upper)

    # Detect the later member of a repeated C3 source group.
    later = expression((
        (2, 7.0),
        (32 + 31, -128.0),
    ), 158.0)
    later = (
        later[0] + 3.5 * source_count[0] - 7.0 * output_count[0],
        later[1],
    )
    put_z(106, (-later[0], -later[1]))

    c3_code = 1.0 + c_offset + 451.0 / denominator
    put_abs_pair(
        108,
        expression(((10, 1.0),), -c3_code),
        tolerance,
    )

    # Triangular joint-equality tests for the second RHS symbol.
    # The source fingerprint is lexicographically weighted, so these
    # bumps require both the rule source and its second target.
    bump_width = 1.0 / 16384.0
    for r in range(6):
        joint = expression((
            (10, 64.0),
            (32 + r, -63.0),
            (32 + 27, 1.0),
            (64 + 6 + r, -4096.0 / auxiliary_scale),
            (32 + 6 + r, 1.0 / auxiliary_scale),
        ), -1.5)
        base = 110 + 3 * r
        put_z(base, joint, bump_width)
        put_z(base + 1, joint)
        put_z(base + 2, joint, -bump_width)

    Wm, Wmz = p["Wm"], p["Wmz"]
    Wma, bm = p["Wma"], p["bm"]

    # Direct and palindromic-rule match indicators.
    for r in range(6):
        for unit, source_pair, target_pair in (
            (r, 4 * r, 4 * r + 2),
            (6 + r, 4 * r, 24 + 2 * r),
            (12 + r, 36 + 2 * r, 24 + 2 * r),
        ):
            Wmz[unit, source_pair:source_pair + 2] = -1.0e7
            Wmz[unit, target_pair:target_pair + 2] = -1.0e7
            Wmz[unit, 107] = -100.0
            bm[unit] = 5.0
            if unit >= 6:
                Wm[unit, 61] = -20.0
                bm[unit] -= 20.0

    # Cache an upper group's associated lower-string ordinal.
    cache_start = 18
    cache_scale = 0.001
    for group in range(12):
        unit = cache_start + group
        Wmz[unit, 48 + 2 * group] = 1.0e7
        Wmz[unit, 49 + 2 * group] = 1.0e7
        Wmz[unit, 72 + group] = cache_scale

    rank_gain = 100.0
    mismatch_gain = 1.0e12

    for candidate, (family, slot) in enumerate(candidates):
        u = 30 + 3 * candidate
        d0, d1 = 84 + 2 * candidate, 85 + 2 * candidate
        count_z = 104 + family

        low = slot - 2.6 + 16.0 * family
        high = slot - 1.9 + 16.0 * family

        Wm[u, cache_start:cache_start + 12] = (
            rank_gain / cache_scale
        )
        Wmz[u, count_z] = -0.5 * rank_gain
        bm[u] = -rank_gain * (11.0 / cache_scale + low)

        Wm[u + 1, cache_start:cache_start + 12] = (
            -rank_gain / cache_scale
        )
        Wmz[u + 1, count_z] = 0.5 * rank_gain
        bm[u + 1] = rank_gain * (
            11.0 / cache_scale + high
        )

        for unit in (u, u + 1):
            Wmz[unit, d0] = -mismatch_gain
            Wmz[unit, d1] = -mismatch_gain

        Wmz[u + 2, d0] = mismatch_gain
        Wmz[u + 2, d1] = mismatch_gain
        bm[u + 2] = -5.0

    # Upper-mode cache refresh.
    Wmz[60, 107] = 20.0
    Wma[60, 4] = 20.0
    bm[60] = -30.0

    # Later-member flag.
    Wmz[61, 106] = -20.0
    Wmz[61, 108:110] = -1.0e7
    Wmz[61, 107] = -100.0
    bm[61] = 5.0

    # Refresh that flag after navigating onto a repeated-source token.
    Wma[62, 4] = 10.0
    Wmz[62, 108:110] = -1.0e7
    Wmz[62, 107] = -100.0
    bm[62] = -5.0

    # Joint second-destination match.
    for r in range(6):
        base = 110 + 3 * r
        Wmz[63, base] = 20.0 / bump_width
        Wmz[63, base + 1] = -40.0 / bump_width
        Wmz[63, base + 2] = 20.0 / bump_width
    Wm[63, 61] = 10.0
    Wmz[63, 107] = -100.0
    bm[63] = -15.0

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -1.0e6

    bo[1] = -5.0
    Wo[1, 107] = 10.0

    bo[2] = 30.0
    Wo[2, 128 + 60] = 10.0
    Wo[2, 128 + 62] = 20.0

    bo[4] = 230.0
    Wo[4, 128:128 + 18] = 10.0
    Wo[4, 128 + 30:128 + 60] = 10.0
    Wo[4, 128 + 60] = -50.0
    Wo[4, 128 + 63] = 10.0

    return p

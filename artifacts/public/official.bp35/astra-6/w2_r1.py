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

    def template(out, samples):
        for colour, dy, dx in samples:
            p["W1"][out, colour, dy + 2, dx + 2] += 1
        p["b1"][out] = 1 - len(samples)

    # Tile-centre detectors.
    template(0, [
        (5, 0, 0), (3, -2, -1), (3, 1, -2), (3, 2, 1)
    ])
    for out, colour in [(1, 14), (2, 8), (3, 12), (4, 15)]:
        template(out, [
            (colour, 0, 0),
            (3, -2, -2), (3, -2, 2),
            (3, 2, -2), (3, 2, 2),
        ])

    # Background samples distinguish a ghost from a filled purple tile.
    template(5, [
        (12, 0, 0),
        (12, -1, -1), (12, -1, 1),
        (12, 1, -1), (12, 1, 1),
        (10, -1, 0), (10, 1, 0),
        (10, 0, -1), (10, 0, 1),
    ])
    template(6, [
        (15, 0, 0), (0, 2, 0), (5, 0, -2), (5, 0, 2)
    ])
    template(7, [
        (15, 0, 0), (0, -2, 0), (5, 0, -2), (5, 0, 2)
    ])
    template(8, [
        (7, 0, 0), (7, -1, 0), (7, -2, 0),
        (5, 1, 0), (5, 0, -1), (5, 0, 1),
    ])
    template(9, [
        (9, 0, -1), (9, -2, -1),
        (9, 1, -1), (5, 2, -1),
    ])

    # Folded coordinate bases: triangular waves with period twelve.
    for j in range(11):
        p["W1"][10 + j, 17, 2, 2] = 63
        p["b1"][10 + j] = -6 * j
    for j in range(10):
        p["W1"][21 + j, 16, 2, 2] = 63
        p["b1"][21 + j] = -6 * (j + 1)
    p["W1"][31, 16, 2, 2] = 63

    def w2(out, inp, value=1.0, dy=0, dx=0):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    thresholds = (2.25, 2.75, 3.25, 3.75)
    for out, threshold in enumerate(thresholds):
        for j in range(11):
            coefficient = 1 if j == 0 else (-2 if j % 2 else 2)
            w2(out, 10 + j, coefficient)
        p["b2"][out] = -threshold

    for out, threshold in zip(range(4, 8), thresholds):
        w2(out, 31)
        for j in range(1, 11):
            w2(out, 20 + j, -2 if j % 2 else 2)
        p["b2"][out] = -threshold

    # Symbol planes.
    for out, inp in [
        (8, 0), (9, 1), (10, 2), (11, 3),
        (12, 4), (14, 8), (15, 5),
    ]:
        w2(out, inp)
    w2(13, 6)
    w2(13, 7)

    w2(16, 9, dx=1)       # Player centre.
    w2(17, 9, dx=-1)      # Second spatial query marker.

    w2(18, 9, 100, dx=1)
    w2(18, 10)
    p["b2"][18] = -100
    w2(19, 31)

    # Saturated binary gravity flags at the player.
    for out, bias in [(20, -132), (21, -133)]:
        w2(out, 9, 100, dx=1)
        w2(out, 31)
        p["b2"][out] = bias
    for out, bias in [(22, -68), (23, -69)]:
        w2(out, 9, 100, dx=1)
        w2(out, 31, -1)
        p["b2"][out] = bias

    w2(24, 10)           # Column coordinate.
    w2(25, 10)
    p["b2"][25] = -62    # Clipped right boundary.
    w2(26, 0, dy=-1)
    w2(27, 0, dy=1)

    for inp in range(10):
        w2(28, inp)
        w2(29, inp)
    p["b2"][29] = -1
    w2(30, 31)
    p["b2"][30] = -33
    w2(31, 31)
    p["b2"][31] = -35

    def f3(out, local=(), row=(), col=(), bias=0.0):
        for inp, value in local:
            p["W3"][out, inp] += value
        for inp, value in row:
            p["W3"][out, 32 + inp] += value
        for inp, value in col:
            p["W3"][out, 64 + inp] += value
        p["b3"][out] = bias

    hat_x = [(0, 2), (1, -2), (2, -2), (3, 2)]
    phase_up = [(4, 2), (5, -2), (6, -2), (7, 2)]

    f3(0, local=hat_x)
    f3(1, row=[(28, 1), (29, -1)])
    f3(2, local=[
        (8, 1), (10, 1), (12, 1), (13, 1), (25, 1)
    ])
    f3(3, local=[
        (8, 1), (9, 1), (10, 1),
        (11, 1), (12, 1), (15, 1),
    ])
    f3(4, local=[(14, 1)])
    f3(5, local=[(19, -1)], bias=1)
    f3(6, local=[(19, 1)], bias=-62)
    f3(7, local=[(16, 1)])
    f3(8, local=[(17, 1)])
    f3(9, local=[(20, 1), (21, -1)])
    f3(10, local=[(22, 1), (23, -1)])

    # Supporting removable tile, exactly six pixels along gravity.
    f3(11,
       local=[(9, 1), (11, 1), (19, 1), (30, -2)],
       col=[(20, 1), (21, -1)], bias=-34)
    f3(12,
       local=[(9, 1), (11, 1), (19, 1), (31, -2)],
       col=[(22, 1), (23, -1)], bias=-36)

    # A spatial tent around the player's x coordinate. Its combination
    # is nonzero only for immediately adjacent removable side blocks.
    side_gate = 128.0
    side_local = [(9, side_gate), (11, side_gate), (24, -1)]
    side_row = [(16, side_gate), (18, 1)]
    f3(13, local=side_local, row=side_row,
       bias=7 - 2 * side_gate)
    f3(22, local=side_local, row=side_row,
       bias=-2 * side_gate)
    f3(23, local=side_local, row=side_row,
       bias=-7 - 2 * side_gate)

    f3(14, local=[(10, 1)])

    # Expansion fallback prefers useful heights and rightward growth.
    f3(15, local=[
        (12, 100), (24, 0.001),
        (19, 0.01), (30, -0.02),
    ], bias=-99.33)

    # Small exploration rewards for newly reachable immutable floors.
    f3(16, local=[(26, 8), (19, -0.002)], bias=-8 + 0.068)
    f3(17, local=[(27, 8), (19, 0.002)], bias=-8 - 0.068)

    # Tile-row phase identifies gravity even away from the player.
    # Prepare ghost platforms ahead, but not those left behind.
    ghost_gate = 128.0
    f3(18, local=[
        (15, ghost_gate), (19, -0.01)
    ] + [(i, ghost_gate * v) for i, v in phase_up],
       bias=-2 * ghost_gate + 0.39)
    f3(19, local=[
        (15, ghost_gate), (19, 0.01)
    ] + [(i, -ghost_gate * v) for i, v in phase_up],
       bias=-ghost_gate - 0.29)

    f3(20, local=[(12, 1)])
    f3(21, local=[(18, 1)])

    def wh(out, inp, value=1.0, dy=0, dx=0):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Capped support-distance transforms.
    for positive, excess, dy in [(0, 1, -1), (2, 3, 1)]:
        for out, bias in [(positive, -1), (excess, -8)]:
            wh(out, positive, 1, dy=dy)
            wh(out, excess, -1, dy=dy)
            p["Uh"][out, 3] = 8
            p["bh"][out] = bias

    horizontal_discount = 0.48
    vertical_discount = 0.98
    gate = 128.0

    # Each propagation component is represented as
    # relu(s) - relu(s-1), so represented values remain bounded.
    # Both members of every pair use exactly the same inputs.
    for base, support, dy, frontier, floor_reward in [
        (4, 0, -1, 5, 16),
        (8, 2, 1, 6, 17),
    ]:
        hp, he, vp, ve = range(base, base + 4)
        sources = [(hp, 1), (he, -1), (vp, 1), (ve, -1)]

        for out, cap_bias in [(hp, 0), (he, -1)]:
            for source, sign in sources:
                wh(out, source, sign * horizontal_discount, dx=-1)
                wh(out, source, sign * horizontal_discount, dx=1)
            wh(out, support, gate)
            wh(out, support + 1, -gate)
            p["Uh"][out, 0] = -gate
            p["Uh"][out, 1] = 8 * gate
            p["Uh"][out, 2] = -gate
            p["Uh"][out, 3] = -7 * gate
            p["bh"][out] = -8 * gate + cap_bias

        for out, cap_bias in [(vp, 0), (ve, -1)]:
            for source, sign in sources:
                wh(out, source, sign * vertical_discount, dy=dy)
            p["Uh"][out, 0] = gate
            p["Uh"][out, 2] = -gate
            p["Uh"][out, 4] = 8
            p["Uh"][out, frontier] = 1
            p["Uh"][out, floor_reward] = 1
            p["bh"][out] = -gate + cap_bias

    # Query left/right using separate spatial markers and max/mean
    # decoding; query the vertical component at the player directly.
    query_gate = 512.0
    query_offset = 64.0
    for lr, vertical, base in [(12, 13, 4), (14, 15, 8)]:
        for source, sign in [
            (base, 1), (base + 1, -1),
            (base + 2, 1), (base + 3, -1),
        ]:
            wh(lr, source, sign * horizontal_discount, dx=-1)
        p["bh"][lr] = -query_gate
        p["Uh"][lr, 7] = query_gate + query_offset
        p["Uh"][lr, 8] = query_gate

        wh(vertical, base + 2, 1)
        wh(vertical, base + 3, -1)
        p["bh"][vertical] = -query_gate
        p["Uh"][vertical, 7] = query_gate

    def hm(ch):
        return ch

    def ha(ch):
        return 16 + ch

    def fm(ch):
        return 32 + ch

    def fa(ch):
        return 64 + ch

    next_z = 0

    def unit(action, terms, bias=0.0, gain=1.0):
        nonlocal next_z
        z = next_z
        next_z += 1
        for index, coefficient in terms:
            p["Wz"][z, index] += coefficient
        p["bz"][z] = bias
        if action is not None:
            p["Wo"][action, z] += gain
        return z

    for lr, vertical, mode, supporting in [
        (12, 13, 9, 11),
        (14, 15, 10, 12),
    ]:
        unit(3, [(hm(lr), 1), (fm(mode), 64)],
             bias=-query_offset - 64)
        unit(4, [
            (ha(lr), 4096), (hm(lr), -1), (fm(mode), 64)
        ], bias=-64)

        unit(6, [
            (hm(vertical), 4),
            (fm(mode), 64),
            (fm(supporting), 64),
            (fm(14), -64),
        ], bias=-128 - 0.00003)

    # Gravity reversal is lower priority than useful ordinary progress.
    for opposite_vertical, mode in [(15, 9), (13, 10)]:
        unit(6, [
            (hm(opposite_vertical), 0.35),
            (fm(mode), 64),
            (fm(14), 64),
        ], bias=-128)

    unit(6, [(fm(14), 0.00012)])
    unit(6, [
        (fa(13), 4096 * 32),
        (fa(22), -4096 * 64),
        (fa(23), 4096 * 32),
    ])

    unit(6, [(fm(18), 1000)])
    unit(6, [(fm(19), 1000)])

    unit(6, [
        (fm(20), 0.0003),
        (fm(14), -64),
        (fm(11), -64),
        (fm(12), -64),
    ])

    zx = unit(None, [(fm(21), 1)])

    # Remember x with a nearly linear tanh. Paired tanh windows detect
    # a previous horizontal action that failed to change x.
    p["Wmz"][0, zx] = 0.001
    for first, previous_action, penalized, opposite in [
        (1, 3, 3, 4),
        (3, 4, 4, 3),
    ]:
        for mem_index, offset in [(first, 2), (first + 1, -2)]:
            p["Wm"][mem_index, 0] = 4000
            p["Wmz"][mem_index, zx] = -4
            p["Wma"][mem_index, previous_action] = 800
            p["bm"][mem_index] = -800 + offset

        p["Wo"][penalized, 128 + first] = -2
        p["Wo"][penalized, 128 + first + 1] = 2
        p["Wo"][opposite, 128 + first] = 0.0001
        p["Wo"][opposite, 128 + first + 1] = -0.0001

    p["bo"][:] = -1000000
    p["bo"][3] = 0
    p["bo"][4] = 0.00001
    p["bo"][6] = -0.00001

    # Click priorities match the high-priority preparatory operations.
    p["Wc"][0, 18] = 100000
    p["Wc"][0, 19] = 100000
    p["Wc"][0, 13] = 4096
    p["Wc"][0, 22] = -8192
    p["Wc"][0, 23] = 4096
    p["Wc"][0, 14] = 1024
    p["Wc"][0, 11] = 256
    p["Wc"][0, 12] = 256
    p["Wc"][0, 15] = 64

    return p

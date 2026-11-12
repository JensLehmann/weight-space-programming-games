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
    p = {name: np.zeros(shape, dtype=np.float32)
         for name, shape in shapes.items()}

    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    sizes = [1, 2, 3, 4, 5, 7, 8, 9, 10]
    codes = [9 - k for k in range(9)]

    def w1(out, inp, dy, dx, value):
        p["W1"][out, inp, dy + 2, dx + 2] += value

    def w2(out, inp, dy, dx, value):
        p["W2"][out, inp, dy + 1, dx + 1] += value

    def wh(out, inp, dy, dx, value):
        p["Wh"][out, inp, dy + 1, dx + 1] += value

    # Positive integer kernels make the following clipped dilations exact.
    local_kernel = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 6, 4, 2],
        [3, 6, 9, 6, 3],
        [2, 4, 6, 4, 2],
        [1, 2, 3, 2, 1],
    ], dtype=np.float32)

    # W1 0:18: paired cumulative-color dilations.
    # Their difference is a binary radius-two neighborhood indicator.
    for k in range(9):
        for color in colors[:k + 1]:
            p["W1"][2 * k, color] = local_kernel
            p["W1"][2 * k + 1, color] = local_kernel
        p["b1"][2 * k + 1] = -1.0

    # W1 18,19: difference is exactly one in the playable rows.
    # The half-integer margins avoid putting a row on a ReLU boundary.
    w1(18, 16, 0, 0, 126.0)
    w1(19, 16, 0, 0, 126.0)
    p["b1"][18] = -18.5
    p["b1"][19] = -19.5

    # W1 20:29: square-corner detectors, anchored one pixel
    # down/right from the corner. The extra exterior tests reject
    # rounded goal markers and hollow hazards.
    for k, color in enumerate(colors):
        out = 20 + k
        if k == 0:
            w1(out, color, -1, -1, 1.0)
        else:
            for dy, dx in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
                w1(out, color, dy, dx, 1.0)
            for dy, dx in [(-2, -1), (-1, -2),
                           (-2, 0), (0, -2)]:
                w1(out, color, dy, dx, -4.0)
            p["b1"][out] = -3.0

    # W1 29: excludes the small legend at the left of the header.
    w1(29, 17, 0, 0, -63.0)
    p["b1"][29] = 25.0

    # W1 30: the rounded goal marker's leading corner.
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, -1)]:
        w1(30, 9, dy, dx, 1.0)
    for dy, dx in [(-1, 0), (0, -1), (-1, 1)]:
        w1(30, 9, dy, dx, -4.0)
    p["b1"][30] = -3.0

    # W1 31: hollow hazard heads, including all three hazard colors.
    for color in (7, 14, 13):
        for dy, dx in [(0, 0), (1, -1), (1, 1)]:
            w1(31, color, dy, dx, 1.0)
        for dy, dx in [(1, 0), (-1, 0)]:
            w1(31, color, dy, dx, -4.0)
    p["b1"][31] = -2.0

    def playable_gate(out, strength, bias):
        w2(out, 18, 0, 0, strength)
        w2(out, 19, 0, 0, -strength)
        p["b2"][out] = bias

    # W2 0:18: radius-three cumulative neighborhoods, clipped to one.
    # Summing their differences encodes the lowest nearby tier.
    for k in range(9):
        for out in (2 * k, 2 * k + 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    w2(out, 2 * k, dy, dx, 1.0)
                    w2(out, 2 * k + 1, dy, dx, -1.0)
            playable_gate(
                out, 40.0,
                -40.0 - float(out == 2 * k + 1)
            )

    # W2 18,19,24: body-piece indicator, inverse-tier code, mass.
    # For large squares the anchor is slightly above/left of the
    # geometric center; it remains inside the square.
    for k in range(9):
        radius = min(sizes[k] // 2, 2)
        shift = 1 - radius
        w2(18, 20 + k, shift, shift, 1.0)
        w2(19, 20 + k, shift, shift, float(codes[k]))
        w2(24, 20 + k, shift, shift, float(2 ** k))
    playable_gate(18, 40.0, -40.0)
    playable_gate(19, 40.0, -40.0)
    playable_gate(24, 1024.0, -1024.0)

    # W2 20,25: requested tier and requested mass in the header.
    for k in range(9):
        w2(20, 20 + k, 0, 0, float(codes[k]))
        w2(25, 20 + k, 0, 0, float(2 ** k))
    for out, strength in ((20, 40.0), (25, 1024.0)):
        w2(out, 18, 0, 0, -strength)
        w2(out, 19, 0, 0, strength)
        w2(out, 29, 0, 0, -strength)

    # W2 21,22: playable goal and hazard seeds.
    w2(21, 30, -1, -1, 1.0)
    playable_gate(21, 40.0, -40.0)
    w2(22, 31, 0, 0, 1.0)
    playable_gate(22, 40.0, -40.0)

    # W2 23: area-normalized short-range piece repulsion.
    # Exclude the colors that also form large goal/hazard graphics.
    inv_area = [1.0 / (s * s) for s in sizes[:6]]
    for k in range(6):
        coefficient = inv_area[k]
        if k < 5:
            coefficient -= inv_area[k + 1]
        coefficient *= 1.5 / 9.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w2(23, 2 * k, dy, dx, coefficient)
    playable_gate(23, 1024.0, -1024.0)

    # W2 26: invalid bottom-row click penalty.
    w2(26, 18, 0, 0, 0.5)
    p["b2"][26] = -52.75

    # f3 channels:
    # 0 candidate presence, 1 candidate inverse-tier code,
    # 2 body indicator, 3 body code, 4 row-pooled minimum-tier seed,
    # 5 column-pooled requested-tier seed, 6 goals, 7 hazards,
    # 8 local repulsion, 9 mass, 10 required mass, 11 bottom penalty.
    p["W3"][0, 16] = 1.0
    p["W3"][0, 17] = -1.0
    for k in range(9):
        p["W3"][1, 2 * k] = 1.0
        p["W3"][1, 2 * k + 1] = -1.0

    for out, inp in [(2, 18), (3, 19), (6, 21), (7, 22),
                     (8, 23), (9, 24), (10, 25), (11, 26)]:
        p["W3"][out, inp] = 1.0
    p["W3"][4, 32 + 19] = 1.0
    p["W3"][5, 64 + 20] = 1.0

    # h0:h4: vertical max broadcast of the lowest present tier.
    # F = h2 + h3. The paired comparator computes max(up, down),
    # and the next stage also takes max with the local seed.
    for src in (2, 3):
        wh(0, src, -1, 0, 1.0)
        wh(0, src, 1, 0, -1.0)
        wh(1, src, 1, 0, 1.0)
    wh(2, 0, 0, 0, 1.0)
    wh(2, 1, 0, 0, 1.0)
    p["Uh"][2, 4] = -1.0
    p["Uh"][3, 4] = 1.0

    # h4:h8: horizontal max broadcast of the header requirement.
    for src in (6, 7):
        wh(4, src, 0, -1, 1.0)
        wh(4, src, 0, 1, -1.0)
        wh(5, src, 0, 1, 1.0)
    wh(6, 4, 0, 0, 1.0)
    wh(6, 5, 0, 0, 1.0)
    p["Uh"][6, 5] = -1.0
    p["Uh"][7, 5] = 1.0

    # h8 is zero exactly when the lowest tier is below the target.
    p["bh"][8] = 1.0
    for src in (6, 7):
        wh(8, src, 0, 0, 1.0)
    for src in (2, 3):
        wh(8, src, 0, 0, -1.0)

    # h9: click eligibility near a globally lowest-tier piece.
    p["Uh"][9, 0] = 1.0
    p["Uh"][9, 1] = 1.0
    for src in (2, 3):
        wh(9, src, 0, 0, -1.0)

    # h10: merging sources, only below the displayed target.
    p["Uh"][10, 2] = 1.0
    p["Uh"][10, 3] = 1.0
    for src in (2, 3):
        wh(10, src, 0, 0, -1.0)
    wh(10, 8, 0, 0, -32.0)

    def inertial_field(current, previous):
        # Heavy-ball relaxation of a discrete spatial potential.
        wh(current, current, 0, 0, 0.10)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            wh(current, current, dy, dx, 0.45)
        wh(current, previous, 0, 0, -0.90)
        wh(previous, current, 0, 0, 1.0)

    # h11,h12: merging potential. Noneligible pieces are sinks.
    inertial_field(11, 12)
    wh(11, 10, 0, 0, 2.0)
    p["Uh"][11, 2] = -1.0

    # h13,h14: goal potential.
    inertial_field(13, 14)
    p["Uh"][13, 6] = 1.0

    # h15: eligible spatial click score.
    wh(15, 9, 0, 0, 512.0)
    wh(15, 11, 0, 0, 3.0)
    wh(15, 13, 0, 0, 2.0)
    p["Uh"][15, 8] = -1.0
    p["bh"][15] = -496.0

    p["Wc"][0, 32 + 15] = 1.0
    p["Wc"][0, 11] = -1024.0

    # Global recovery tests. g contains f3 maxima starting at 32,
    # and f3 means starting at 64.
    p["Wz"][0, 32 + 5] = -1.0
    p["bz"][0] = 1.0

    p["Wz"][1, 64 + 10] = 4096.0
    p["Wz"][1, 64 + 9] = -4096.0
    p["bz"][1] = -0.1

    # Normally click. Prefer RESET for recovery if available,
    # otherwise ACTION7 provides the game's undo mechanism.
    p["bo"][6] = 1.0
    p["bo"][0] = -1.0
    p["bo"][7] = -2.0
    for unit in (0, 1):
        p["Wo"][0, unit] = 1025.0
        p["Wo"][7, unit] = 1024.0

    return p

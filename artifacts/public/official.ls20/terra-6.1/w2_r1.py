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

    W1, b1 = p["W1"], p["b1"]
    W2, b2 = p["W2"], p["b2"]
    W3, b3 = p["W3"], p["b3"]
    Wz, bz = p["Wz"], p["bz"]
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    glyph_colors = (8, 9, 12, 14)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Goal glyph and enlarged UI glyph samples.
    for j in range(9):
        dy, dx = divmod(j, 3)
        dy -= 1
        dx -= 1
        for color in glyph_colors:
            W1[j, color, 2 + dy, 2 + dx] = 1.0
            W1[9 + j, color, 2 + 2 * dy, 2 + 2 * dx] = 1.0

    # Goal detector.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) == 2 or abs(dx) == 2:
                W1[18, 5, 2 + dy, 2 + dx] = 10.0
            else:
                for color in glyph_colors:
                    W1[18, color, 2 + dy, 2 + dx] = 1.0
    b1[18] = -164.0

    # Player detector.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            color = 12 if dy < 0 else 9
            W1[19, color, 2 + dy, 2 + dx] = 2.0
    b1[19] = -49.0

    # Rotation switch.
    for dy, dx, color in (
        (-1, 0, 0),
        (0, -1, 1),
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
    ):
        W1[20, color, 2 + dy, 2 + dx] = 2.0
    b1[20] = -9.0

    # Shape switch.
    for dy, dx in ((-1, -1), (0, 0), (0, 1), (1, 0)):
        W1[31, 0, 2 + dy, 2 + dx] = 2.0
    b1[31] = -7.0

    # Select the fixed center of the UI icon.
    for out, coord, sign, threshold in (
        (21, 16, 1.0, 57.0 / 63.0),
        (22, 16, -1.0, 57.0 / 63.0),
        (23, 17, 1.0, 5.0 / 63.0),
        (24, 17, -1.0, 5.0 / 63.0),
    ):
        W1[out, coord, 2, 2] = sign
        b1[out] = -sign * threshold

    W1[25, 16, 2, 2] = 1.0
    W1[26, 17, 2, 2] = 1.0

    for i, (dy, dx) in enumerate(directions):
        W1[27 + i, 4, 2 + 2 * dy, 2 + 2 * dx] = 1.0

    for j in range(9):
        W2[j, j, 1, 1] = 1.0
        W2[9 + j, 9 + j, 1, 1] = 1.0
        for c in (21, 22, 23, 24):
            W2[9 + j, c, 1, 1] = -126.0

    # Difference of channels 18 and 19 is the goal marker.
    W2[18, 18, 1, 1] = 1.0
    W2[19, 18, 1, 1] = 1.0
    b2[19] = -1.0

    W2[20, 19, 1, 1] = 1.0
    W2[21, 20, 1, 1] = 1.0
    W2[21, 31, 1, 1] = 1.0
    W2[22, 25, 1, 1] = 1.0
    W2[23, 26, 1, 1] = 1.0

    for i, (dy, dx) in enumerate(directions):
        W2[24 + i, 19, 1 - dy, 1 - dx] = 1.0
        W2[24 + i, 27 + i, 1, 1] = 1.0
        b2[24 + i] = -1.0

    for j in range(9):
        W3[j, j] = 1.0
        W3[j, 18] = 2.0
        W3[j, 19] = -2.0
        b3[j] = -2.0
        W3[9 + j, 9 + j] = 1.0

    # Goal, player, and switch coordinates.
    for out, coordinate in ((18, 22), (19, 23)):
        W3[out, coordinate] = 1.0
        W3[out, 18] = 2.0
        W3[out, 19] = -2.0
        b3[out] = -2.0

    for out, coordinate in ((20, 22), (21, 23)):
        W3[out, coordinate] = 1.0
        W3[out, 20] = 2.0
        b3[out] = -2.0

    for out, coordinate in ((22, 22), (23, 23)):
        W3[out, coordinate] = 1.0
        W3[out, 21] = 2.0
        b3[out] = -2.0

    for i in range(4):
        W3[24 + i, 24 + i] = 1.0

    def feature(channel):
        return 32 + channel

    goal_deltas = (
        (20, 18),
        (18, 20),
        (21, 19),
        (19, 21),
    )
    switch_deltas = (
        (20, 22),
        (22, 20),
        (21, 23),
        (23, 21),
    )

    base_glyphs = [
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
    ]

    # Goal-directed terms are gated on an exact glyph match.
    unit = 0
    for base in base_glyphs:
        base = np.asarray(base, dtype=np.int8)
        for rotation in range(4):
            pattern = np.rot90(base, -rotation).reshape(-1)
            ones = int(pattern.sum())
            for action_index, (positive, negative) in enumerate(goal_deltas):
                for j, bit in enumerate(pattern):
                    sign = 1.0 if bit else -1.0
                    Wz[unit, feature(j)] = sign
                    Wz[unit, feature(9 + j)] = sign
                Wz[unit, feature(positive)] += 0.5
                Wz[unit, feature(negative)] -= 0.5
                bz[unit] = -2.0 * ones + 0.0005
                Wo[1 + action_index, unit] = 1.0
                unit += 1

    for i, (positive, negative) in enumerate(switch_deltas):
        u = 96 + i
        Wz[u, feature(positive)] = 0.5
        Wz[u, feature(negative)] = -0.5
        Wo[1 + i, u] = 0.01

    for i in range(4):
        u = 100 + i
        Wz[u, feature(24 + i)] = 1.0
        Wo[1 + i, u] = -2.0

    # Coordinates and diagonal projections for the history comparison.
    Wz[104, feature(20)] = 1.0
    Wz[105, feature(21)] = 1.0
    Wz[106, feature(20)] = 1.0
    Wz[106, feature(21)] = 1.0
    Wz[107, feature(20)] = 1.0
    Wz[107, feature(21)] = -1.0
    bz[107] = 1.0

    # Three previous positions, centered and scaled to keep tanh linear.
    Wmz[0, 104] = 0.1
    Wmz[1, 105] = 0.1
    bm[0] = -0.05
    bm[1] = -0.05
    Wm[2, 0] = 1.0
    Wm[3, 1] = 1.0
    Wm[4, 2] = 1.0
    Wm[5, 3] = 1.0

    # Two diagonal projections give a strong penalty for revisiting the
    # same cell and a weaker penalty for a diagonal alignment.
    k = 80.0
    epsilon = 0.025
    history_weight = 0.00018
    cell = 5.0 / 63.0

    for history in range(3):
        for direction, (dy, dx) in enumerate(directions):
            for projection in range(2):
                offset = (dy + dx if projection == 0 else dy - dx) * cell
                zcoord = 106 if projection == 0 else 107
                for side in range(2):
                    row = 6 + (((history * 4 + direction) * 2
                                + projection) * 2 + side)
                    Wmz[row, zcoord] = k
                    Wm[row, 2 * history] = -10.0 * k
                    Wm[row, 2 * history + 1] = (
                        -10.0 * k if projection == 0 else 10.0 * k
                    )
                    shift = epsilon if side == 0 else -epsilon
                    bm[row] = k * (offset + shift - 1.0)
                    Wo[1 + direction, 128 + row] = (
                        -history_weight if side == 0 else history_weight
                    )

    # Original immediate-reversal penalty.
    for i in range(4):
        Wma[54 + i, 1 + i] = 1.0
    opposites = (1, 0, 3, 2)
    for i in range(4):
        Wo[1 + i, 128 + 54 + opposites[i]] = -0.002

    bo[0] = -100.0
    bo[5:] = -100.0
    return p

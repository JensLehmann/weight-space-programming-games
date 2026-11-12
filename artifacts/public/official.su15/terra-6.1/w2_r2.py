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
    Wc = p["Wc"]

    # Fruit markers.
    colors = [10, 6, 15, 11, 12, 8, 9, 7, 14]
    W1[0, 10, 2, 2] = 1.0
    for channel, color in enumerate(colors[1:], 1):
        for dy, dx in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            W1[channel, color, 2 + dy, 2 + dx] = 1.0
        for dy, dx in [(-2, -1), (-1, -2), (-2, 0), (0, -2)]:
            W1[channel, color, 2 + dy, 2 + dx] = -1.0
        b1[channel] = -3.0

    # Goal corner, including a red goal after mass has been lost.
    corner = [
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    count = 0
    for y, row in enumerate(corner):
        for x, value in enumerate(row):
            weight = 1.0 if value else -1.0
            W1[9, 9, y, x] = weight
            W1[9, 2, y, x] = weight
            count += value
    b1[9] = -(count - 1)

    # Invalid click rows.
    W1[10, 16, 2, 2] = -63.0
    b1[10] = 10.0
    W1[11, 16, 2, 2] = 63.0
    b1[11] = -62.0

    # Enemy templates.
    enemy_patterns = [
        (12, 7, [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]),
        (13, 14, [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]),
        (14, 13, [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]),
    ]
    for channel, color, pattern in enemy_patterns:
        count = 0
        for y, row in enumerate(pattern):
            for x, value in enumerate(row):
                W1[channel, color, y + 1, x] = 1.0 if value else -1.0
                count += value
        b1[channel] = -(count - 1)

    # Local occupancy for same-color pair detection.
    pair_colors = [10, 6, 15, 11, 12, 8]
    pair_sizes = [1, 2, 3, 4, 5, 7]
    for i, (color, size) in enumerate(zip(pair_colors, pair_sizes)):
        W1[15 + i, color, :, :] = 1.0 / float(size * size)

    # Header-red and coordinate masks.
    W1[21, 2, 2, 2] = 1.0
    W1[22, 16, 2, 2] = 63.0
    b1[22] = -9.0
    W1[23, 17, 2, 2] = -63.0
    b1[23] = 24.0

    # f2[0]: approximate movable-object centers.
    W2[0, 0, 1, 1] = 1.0
    for channel in (1, 2, 3, 4, 5, 7, 8):
        shift = 1 if channel >= 3 else 0
        W2[0, channel, 1 - shift, 1 - shift] = 1.0
    for channel in (12, 13, 14):
        W2[0, channel, 0, 1] = 1.0
    W2[0, 10, 1, 1] = -100.0
    W2[0, 11, 1, 1] = -100.0

    # Goal marker and invalid-row masks.
    W2[1, 9, 0, 0] = 1.0
    W2[1, 10, 1, 1] = -100.0
    W2[1, 11, 1, 1] = -100.0
    W2[2, 10, 1, 1] = 1.0
    W2[3, 11, 1, 1] = 1.0

    for i in range(len(pair_colors)):
        W2[4 + i, 15 + i, :, :] = 1.0
        W2[4 + i, 10, 1, 1] = -100.0
        W2[4 + i, 11, 1, 1] = -100.0
        b2[4 + i] = -9.0

    W2[10, 21, 1, 1] = 1.0
    W2[10, 22, 1, 1] = -10.0

    W2[11, 9, :, :] = 1.0
    W2[11, 10, 1, 1] = -100.0
    W2[11, 11, 1, 1] = -100.0

    # Header requests a fruit of level at least 2.
    for channel in range(2, 9):
        W2[12, channel, 1, 1] = 1.0
    W2[12, 22, 1, 1] = -10.0
    W2[12, 23, 1, 1] = -10.0

    # Level-0 and level-1 fruit markers.
    W2[15, 0, 1, 1] = 1.0
    W2[15, 1, 1, 1] = 1.0
    W2[15, 10, 1, 1] = -100.0
    W2[15, 11, 1, 1] = -100.0

    W3[0, 0] = 1.0
    W3[1, 32 + 1] = 1.0
    W3[2, 64 + 1] = 1.0
    W3[3, 2] = 1.0
    W3[4, 3] = 1.0
    for i in range(len(pair_colors)):
        W3[5, 4 + i] = 1.0
    W3[6, 10] = 1.0
    W3[7, 32 + 11] = 1.0
    W3[7, 64 + 11] = 1.0
    b3[7] = -1.0

    W3[8, 64 + 12] = 1.0
    W3[10, 15] = 1.0
    W3[11, 1] = 1.0

    # Directional goal waves.
    for channel, source, ky, kx in [
        (0, 1, 0, 1),
        (1, 1, 2, 1),
        (2, 2, 1, 0),
        (3, 2, 1, 2),
    ]:
        Wh[channel, channel, ky, kx] = 1.0
        Uh[channel, source] = 64.0
        bh[channel] = -1.0

    # Bounded diffusion around objects and goals.
    for channel in (4, 7, 15):
        for ky in range(3):
            for kx in range(3):
                if (ky, kx) != (1, 1):
                    Wh[channel, channel, ky, kx] = 0.12

    Uh[4, 0] = 1.0
    Uh[4, 7] = -1.0
    Wh[4, 14, 1, 1] = -1.0

    Uh[7, 11] = 1.0

    # Clip object influence into a candidate region.
    Wh[5, 4, 1, 1] = 1.0
    bh[5] = -0.035
    Wh[6, 4, 1, 1] = 1.0
    bh[6] = -0.036

    # Spread the header flag horizontally, then clip it to approximately 1.
    Wh[8, 8, 1, 0] = 0.49
    Wh[8, 8, 1, 2] = 0.49
    Uh[8, 8] = 10000.0
    Wh[10, 8, 1, 1] = 1.0
    bh[10] = -1.0
    Wh[11, 8, 1, 1] = 1.0
    bh[11] = -2.0

    # Mark undersized fruit near a goal when a larger fruit is requested.
    Uh[14, 10] = 2.0
    Wh[14, 7, 1, 1] = 1.0
    Wh[14, 10, 1, 1] = 1.0
    Wh[14, 11, 1, 1] = -1.0
    bh[14] = -3.1
    Wh[15, 14, 1, 1] = 1.0

    # Candidate membership, goal proximity, pair preference, and
    # suppression near undersized fruit already at a goal.
    Wc[0, 32 + 5] = 100000.0
    Wc[0, 32 + 6] = -100000.0
    for channel in range(4):
        Wc[0, 32 + channel] = 0.3
    Wc[0, 1] = -18.9
    Wc[0, 2] = -18.9
    Wc[0, 3] = -10000.0
    Wc[0, 4] = -10000.0
    Wc[0, 5] = 30.0
    Wc[0, 32 + 15] = -40.0

    # Reset when the objective display indicates insufficient fruit mass.
    p["Wz"][0, 32 + 6] = 1.0
    p["Wo"][0, 0] = 2.0
    p["bo"][6] = 1.0
    return p

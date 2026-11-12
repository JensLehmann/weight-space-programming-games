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

    # Approximate visible support.
    for color in (2, 6, 7, 8, 9, 11, 12, 13, 14, 15):
        W1[0, color, 2, 2] = 1.0

    # Isolated 2x2 goal.
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            W1[1, 11, dy + 2, dx + 2] = (
                1.0 if 0 <= dy <= 1 and 0 <= dx <= 1 else -1.0
            )
    b1[1] = -3.0

    # Player detectors shifted toward each candidate position.
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i, (dy, dx) in enumerate(shifts):
        for yy in (0, 1):
            for xx in (0, 1):
                W1[2 + i, 14, 2 + dy + yy, 2 + dx + xx] = 1.0
        b1[2 + i] = -3.0

    # Two yellow checkerboard orientations.
    patterns = (
        ((9, 4), (4, 9)),
        ((4, 9), (9, 4)),
    )
    for i, pattern in enumerate(patterns):
        for dy in range(2):
            for dx in range(2):
                W1[6 + i, pattern[dy][dx], 2 + dy, 2 + dx] = 1.0
        b1[6 + i] = -3.0

    # Yellow and blue control interiors.
    W1[8, 9, :, :] = 1.0
    b1[8] = -24.0
    W1[9, 8, :, :] = 1.0
    b1[9] = -24.0

    W1[10, 13, 2, 2] = 1.0
    W1[11, 4, 2, 2] = 1.0

    for dy in (0, 1):
        for dx in (0, 1):
            W1[12, 14, 2 + dy, 2 + dx] = 1.0
    b1[12] = -3.0

    W1[13, 16, 2, 2] = 1.0
    W1[14, 17, 2, 2] = 1.0

    # Fully supported 2x2 positions, empty positions, and pivots.
    for dy in (0, 1):
        for dx in (0, 1):
            W2[0, 0, 1 + dy, 1 + dx] = 1.0
            W2[7, 11, 1 + dy, 1 + dx] = 1.0
            W2[8, 10, 1 + dy, 1 + dx] = 1.0
    b2[0] = -3.0
    b2[7] = -3.0
    b2[8] = -3.0

    W2[1, 1, 1, 1] = 1.0
    for i, (dy, dx) in enumerate(shifts):
        W2[2 + i, 2 + i, 1 + dy, 1 + dx] = 1.0

    W2[6, 6, 1, 1] = 1.0
    W2[6, 7, 1, 1] = 1.0
    W2[9, 8, 1, 1] = 1.0
    W2[10, 9, 1, 1] = 1.0

    # Coordinates at player and goal.
    W2[11, 12, 1, 1] = 1.0
    W2[11, 13, 1, 1] = 1.0
    b2[11] = -1.0
    W2[12, 12, 1, 1] = 1.0
    W2[12, 14, 1, 1] = 1.0
    b2[12] = -1.0
    W2[13, 1, 1, 1] = 1.0
    W2[13, 13, 1, 1] = 1.0
    b2[13] = -1.0
    W2[14, 1, 1, 1] = 1.0
    W2[14, 14, 1, 1] = 1.0
    b2[14] = -1.0

    # A checkerboard at one of the candidate player positions.
    W2[15, 6, 1, 1] = 1.0
    W2[15, 7, 1, 1] = 1.0
    for i, (dy, dx) in enumerate(shifts):
        W2[15, 2 + i, 1 + dy, 1 + dx] = 1.0
    b2[15] = -1.0

    W3[0, 0] = 1.0
    W3[0, 6] = 1.0
    W3[1, 1] = 1.0
    for i in range(4):
        W3[2 + i, 2 + i] = 1.0

    W3[6, 6] = 1.0
    W3[7, 7] = 1.0
    W3[8, 8] = 1.0
    W3[9, 9] = 1.0
    W3[10, 10] = 1.0

    W3[11, 6] = 1.0
    for i in range(4):
        W3[11, 2 + i] = 1.0
    b3[11] = -1.0

    for i in range(4):
        W3[12 + i, 2 + i] = 1.0
        W3[12 + i, 6] = 1.0
        b3[12 + i] = -1.0

        W3[16 + i, 2 + i] = 1.0
        W3[16 + i, 7] = 1.0
        b3[16 + i] = -1.0

        W3[20 + i, 2 + i] = 1.0
        W3[20 + i, 0] = 1.0
        b3[20 + i] = -1.0

    W3[24, 11] = 1.0
    W3[25, 12] = 1.0
    W3[26, 13] = 1.0
    W3[27, 14] = 1.0

    # Broadcast the adjacent-checkerboard signal along its row and column.
    W3[28, 32 + 15] = 1.0
    W3[28, 64 + 15] = 1.0

    # Estimated possible bridge positions around each pivot.
    Wh[0, 0, 0, 1] = 0.45
    Wh[0, 0, 2, 1] = 0.45
    Uh[0, 8] = 1.0

    Wh[1, 1, 1, 0] = 0.45
    Wh[1, 1, 1, 2] = 0.45
    Uh[1, 8] = 1.0

    Wh[2, 0, 1, 1] = -6.0
    Wh[2, 1, 1, 1] = -6.0
    Uh[2, 0] = -1.0
    bh[2] = 1.0

    # Max-like propagation of a discounted goal potential.
    Wh[3, 9, 0, 1] = 1.0
    Wh[3, 9, 2, 1] = -1.0
    Wh[4, 9, 2, 1] = 1.0

    Wh[5, 9, 1, 0] = 1.0
    Wh[5, 9, 1, 2] = -1.0
    Wh[6, 9, 1, 2] = 1.0

    Wh[7, 3, 1, 1] = 1.0
    Wh[7, 4, 1, 1] = 1.0
    Wh[7, 5, 1, 1] = -1.0
    Wh[7, 6, 1, 1] = -1.0

    Wh[8, 5, 1, 1] = 1.0
    Wh[8, 6, 1, 1] = 1.0

    Wh[9, 7, 1, 1] = 0.99
    Wh[9, 8, 1, 1] = 0.99
    Wh[9, 2, 1, 1] = -4.0
    Uh[9, 1] = 1.0
    bh[9] = -0.005

    for i in range(4):
        Wh[10 + i, 9, 1, 1] = 1.0
        Uh[10 + i, 2 + i] = 64.0
        bh[10 + i] = -64.0

    # Carry the checker signal to yellow-control pixels.
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        Wh[14, 14, 1 + dy, 1 + dx] = 0.24
    Uh[14, 28] = 1.0

    Wh[15, 14, 1, 1] = 1.0
    Uh[15, 9] = 32.0
    bh[15] = -32.0

    Wz, bz = p["Wz"], p["bz"]
    for i in range(4):
        Wz[i, 10 + i] = 1.0
        Wz[i, 32 + 12 + i] = -64.0
        Wz[i, 32 + 16 + i] = -64.0
        Wz[i, 32 + 20 + i] = 0.02

        Wz[4 + i, 10 + i] = 1.0
        Wz[4 + i, 32 + 12 + i] = 64.0
        Wz[4 + i, 32 + 16 + i] = 64.0
        bz[4 + i] = -64.0

    Wz[20, 32 + 24] = 1.0
    Wz[21, 32 + 25] = 1.0
    Wz[30, 32 + 11] = 1.0

    coordinate_pairs = (
        (26, 24), (24, 26), (27, 25), (25, 27)
    )
    for i, (positive, negative) in enumerate(coordinate_pairs):
        Wz[24 + i, 32 + positive] = 0.2
        Wz[24 + i, 32 + negative] = -0.2
        Wz[24 + i, 32 + 20 + i] = 1.0
        bz[24 + i] = -1.0

    # Remember player coordinates and detect an immediately blocked move.
    p["Wmz"][0, 20] = 0.01
    p["Wmz"][1, 21] = 0.01

    for i in range(4):
        axis = 0 if i < 2 else 1
        zcoord = 20 + axis
        u = 2 + 3 * i

        p["Wmz"][u, zcoord] = 315.0
        p["Wm"][u, axis] = -31500.0
        p["Wma"][u, 1 + i] = 2.0
        p["bm"][u] = -1.0

        p["Wmz"][u + 1, zcoord] = -315.0
        p["Wm"][u + 1, axis] = 31500.0
        p["Wma"][u + 1, 1 + i] = 2.0
        p["bm"][u + 1] = -1.0

        p["Wma"][u + 2, 1 + i] = 2.0
        p["bm"][u + 2] = -1.0

    Wo, bo = p["Wo"], p["bo"]
    bo[0] = -100.0
    bo[5] = -100.0
    bo[7] = -100.0
    bo[6] = -0.02

    for i in range(4):
        action = 1 + i
        Wo[action, i] = 1.0
        Wo[action, 24 + i] = 1.0
        bo[action] = 0.001 - 1.5 * np.tanh(1.0)

        u = 2 + 3 * i
        Wo[action, 128 + u] = -1.5
        Wo[action, 128 + u + 1] = -1.5
        Wo[action, 128 + u + 2] = 1.5

        Wo[6, 4 + i] = 1.0

    Wo[6, 30] = 0.10

    # Yellow when a checkerboard is adjacent; otherwise blue.
    p["Wc"][0, 10] = 1e-7
    p["Wc"][0, 9] = 1e-8
    p["Wc"][0, 32 + 15] = 1.0

    return p

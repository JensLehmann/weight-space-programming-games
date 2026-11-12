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
    p = {k: np.zeros(s, dtype=np.float32) for k, s in shapes.items()}
    W1, b1 = p["W1"], p["b1"]

    def piece_detector(channel, dy, dx, center_colors):
        for color in center_colors:
            W1[channel, color, 2 + dy, 2 + dx] = 18
        for ry in range(-1, 2):
            for rx in range(-1, 2):
                if ry or rx:
                    W1[channel, 0, 2 + dy + ry, 2 + dx + rx] = 2
                    W1[channel, 14, 2 + dy + ry, 2 + dx + rx] = 2
        b1[channel] = -33

    piece_detector(0, 0, 0, (0,))
    piece_detector(4, 0, 0, (4, 5))

    # A square goal border may be partly covered by a piece.
    for y in range(5):
        for x in range(5):
            if y in (0, 4) or x in (0, 4):
                for color in (0, 4, 14):
                    W1[1, color, y, x] = 2
    b1[1] = -31

    # Dilated permanent-wall mask.
    for y in range(1, 4):
        for x in range(1, 4):
            W1[2, 2, y, x] = 1
            W1[3, 2, y, x] = 1
    b1[3] = -1

    W1[5, 17, 2, 2] = 1
    W1[6, 16, 2, 2] = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(directions):
        piece_detector(7 + i, -dy, -dx, (0,))

    W2, b2 = p["W2"], p["b2"]
    W2[0, 0, 1, 1] = 1

    # Unoccupied goal and selected piece on goal.
    W2[1, 1, 1, 1] = 1
    W2[1, 0, 1, 1] = -1
    W2[1, 4, 1, 1] = -1
    W2[2, 1, 1, 1] = 1
    W2[2, 0, 1, 1] = 1
    b2[2] = -1

    W2[3, 4, 1, 1] = 1
    W2[4, 2, 1, 1] = 1
    W2[4, 3, 1, 1] = -1

    # Unselected piece not already on a goal.
    W2[5, 4, 1, 1] = 1
    W2[5, 1, 1, 1] = -1

    # Selected-piece coordinates.
    W2[6, 0, 1, 1] = 1
    W2[6, 5, 1, 1] = 1
    b2[6] = -1
    W2[7, 0, 1, 1] = 1
    W2[7, 6, 1, 1] = 1
    b2[7] = -1

    for i, (dy, dx) in enumerate(directions):
        W2[8 + i, 7 + i, 1 - dy, 1 - dx] = 1
    W2[12, 5, 1, 1] = 1

    W3, b3 = p["W3"], p["b3"]
    for i in range(12):
        W3[i, i] = 1

    # Open goals seed the field, with a preference for rightward goals.
    W3[12, 1] = 1
    W3[12, 12] = 0.4
    b3[12] = -0.4

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def field_weight(out, offset, value):
        ky, kx = offset
        Wh[out, 0, ky, kx] += value
        Wh[out, 7, ky, kx] += value

    north, south, west, east = (0, 1), (2, 1), (1, 0), (1, 2)

    # ReLU identities implement a four-neighbor maximum.
    field_weight(1, north, 1)
    field_weight(1, south, -1)
    field_weight(2, east, 1)
    field_weight(2, west, -1)
    field_weight(3, south, 1)
    field_weight(4, west, 1)
    Wh[5, 1, 1, 1] = 1
    Wh[5, 3, 1, 1] = 1
    Wh[5, 2, 1, 1] = -1
    Wh[5, 4, 1, 1] = -1
    Wh[6, 2, 1, 1] = 1
    Wh[6, 4, 1, 1] = 1

    Wh[0, 5, 1, 1] = 1
    Wh[0, 6, 1, 1] = 1
    Uh[0, 4] = -2
    Uh[0, 1] = -2
    bh[0] = -0.002
    Uh[7, 12] = 1

    # Read the field at candidate destinations three pixels away.
    for i, (dy, dx) in enumerate(directions):
        field_weight(8 + i, (1 + dy, 1 + dx), 1)
        Uh[8 + i, 8 + i] = 2
        bh[8 + i] = -2

    Wz = p["Wz"]
    for i in range(4):
        Wz[i, 8 + i] = 1
    Wz[4, 32 + 2] = 1
    Wz[5, 32 + 6] = 1
    Wz[6, 32 + 7] = 1

    # Store the current selected-piece coordinates.
    p["Wmz"][0, 5] = 0.1
    p["Wmz"][1, 6] = 0.1

    # Compare current and previous coordinates to detect failed moves.
    step = 0.1 * 3.0 / 63.0
    scale = 400.0
    tests = [
        (1, 6, 1),
        (1, 6, -1),
        (0, 5, 1),
        (0, 5, -1),
    ]
    for i, (mem_coord, z_coord, sign) in enumerate(tests):
        j = 2 + i
        p["Wm"][j, mem_coord] = -scale * sign
        p["Wmz"][j, z_coord] = scale * sign * 0.1
        p["Wma"][j, 1 + i] = scale * 0.3
        p["bm"][j] = scale * (step / 2.0 - 0.3)

    # Keep the preceding direction for a small reversal penalty.
    for i in range(4):
        p["Wma"][8 + i, 1 + i] = 1

    for i in range(4):
        p["Wo"][1 + i, i] = 1
    opposite = [1, 0, 3, 2]
    for i in range(4):
        p["Wo"][1 + i, 128 + 8 + opposite[i]] = -0.004

    # Click another piece if the selected one is solved or stuck.
    p["Wo"][6, 4] = 5
    for i in range(4):
        p["Wo"][6, 128 + 2 + i] = 4
    p["bo"][6] = 12

    p["bo"][0] = -100
    p["bo"][5] = -100
    p["bo"][7] = -100

    p["Wc"][0, 5] = 1
    return p

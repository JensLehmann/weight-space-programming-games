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
    Wh, Uh = p["Wh"], p["Uh"]

    colours = (6, 7, 8, 9, 10, 11, 12, 14)
    diagonals = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    # First layer: colour, marker, target, cap and slider features.
    for i, colour in enumerate(colours):
        W1[i, colour, 2, 2] = 1
    W1[8, 13, 2, 2] = 1

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        W1[9, 13, 2 + dy, 2 + dx] = 1
    b1[9] = -3

    for ch, threshold in ((10, 5), (11, 6)):
        W1[ch, 13, 2, 2] = 4
        for dy, dx in diagonals:
            for colour in colours:
                W1[ch, colour, 2 + dy, 2 + dx] += 1
            W1[ch, 13, 2 + dy, 2 + dx] -= 4
        b1[ch] = -threshold

    cap_specs = (
        ((0, -1), (0, 1), (1, 0)),
        ((0, -1), (0, 1), (-1, 0)),
        ((-1, 0), (1, 0), (0, 1)),
        ((-1, 0), (1, 0), (0, -1)),
    )
    for ch, (side1, side2, interior) in enumerate(cap_specs, start=12):
        W1[ch, 3, 2, 2] = 1
        W1[ch, 3, 2 + side1[0], 2 + side1[1]] = 1
        W1[ch, 3, 2 + side2[0], 2 + side2[1]] = 1
        for colour in colours + (13,):
            W1[ch, colour, 2 + interior[0], 2 + interior[1]] = 1
        b1[ch] = -3

    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for ch, (dy, dx) in enumerate(directions, start=16):
        for colour in colours:
            W1[ch, colour, 2, 2] = 1
        W1[ch, 4, 2 + dy, 2 + dx] = 1
        W1[ch, 3, 2 + 2 * dy, 2 + 2 * dx] = 1
        b1[ch] = -2

    W1[20, 16, 2, 2] = 1
    W1[21, 17, 2, 2] = 1
    W1[22, 16, 2, 2] = 63
    b1[22] = -62
    W1[23, 3, 2, 2] = 1

    # Marker colour. Suppress a marker already coincident with a target.
    for i in range(8):
        W2[i, 8, 1, 1] = 4
        W2[i, 9, 1, 1] = -4
        for dy, dx in diagonals:
            W2[i, i, 1 + dy, 1 + dx] += 1
            W2[i, 8, 1 + dy, 1 + dx] -= 4
        b2[i] = -5

    W2[8, 10, 1, 1] = 1
    W2[8, 11, 1, 1] = -1
    W2[8, 9, 1, 1] = -1

    # Slider colour and side.
    for i in range(8):
        W2[9 + i, i, 1, 1] = 2
        for ch in range(16, 20):
            W2[9 + i, ch, 1, 1] = 1
        b2[9 + i] = -2

    W2[17, 16, 1, 1] = 1
    W2[17, 18, 1, 1] = 1
    W2[18, 17, 1, 1] = 1
    W2[18, 19, 1, 1] = 1

    # Target position encodings.
    for out, coord, sign, bias in (
        (19, 21, 1, -1),
        (20, 21, -1, 0),
        (21, 20, 1, -1),
        (22, 20, -1, 0),
    ):
        W2[out, 9, 1, 1] = 2
        W2[out, coord, 1, 1] = sign
        b2[out] = bias

    for i in range(4):
        W2[23 + i, 12 + i, 1, 1] = 1
    W2[27, 20, 1, 1] = 1
    W2[28, 21, 1, 1] = 1

    # Remaining-step bar: the largest column with colour 3 on row 63.
    W2[29, 22, 1, 1] = 2
    W2[29, 23, 1, 1] = 2
    W2[29, 21, 1, 1] = 1
    b2[29] = -4
    W2[30, 22, 1, 1] = 1

    for i in range(8):
        W3[i, i] = 1
        W3[8 + i, 9 + i] = 1
    W3[16, 17] = 1
    W3[17, 18] = 1
    W3[18, 17] = -1
    W3[18, 18] = -1
    b3[18] = 1

    # Direct aligned-target requests.
    requests = (
        (19, 25, 28, True, True),
        (20, 26, 28, False, True),
        (21, 23, 27, True, False),
        (22, 24, 27, False, False),
        (20, 25, 28, False, True),
        (19, 26, 28, True, True),
        (22, 23, 27, False, False),
        (21, 24, 27, True, False),
    )
    for out, (goal, cap, coord, positive, horizontal) in enumerate(
        requests, start=19
    ):
        axial_offset = 32 if horizontal else 64
        W3[out, axial_offset + goal] = 63
        W3[out, axial_offset + cap] = 128
        W3[out, 8] = 128
        W3[out, coord] = -63 if positive else 63
        b3[out] = -256 - (63 if positive else 126)

    W3[27, 32 + 29] = 1
    W3[28, 32 + 30] = 1

    # Three hinges form a short triangular pulse on the timer row.
    for ch, threshold in ((29, 61.0), (30, 59.0), (31, 57.0)):
        W3[ch, 32 + 29] = -63
        W3[ch, 32 + 30] = 100
        b3[ch] = threshold - 100

    marker_gate = np.float32(1024)
    button_gate = np.float32(1e24)

    for i in range(8):
        for direction in range(2):
            h = 2 * i + direction

            for dy in range(3):
                for dx in range(3):
                    if dy != 1 or dx != 1:
                        Wh[h, h, dy, dx] = np.float32(0.17)

            goal_channels = (
                range(23, 27) if direction == 0 else range(19, 23)
            )
            for ch in goal_channels:
                Uh[h, ch] = 1

            for j in range(8):
                Uh[h, j] -= marker_gate
            Uh[h, i] += marker_gate

            for j in range(8):
                if j != i:
                    Uh[h, 8 + j] = -button_gate
            Uh[h, 17 if direction == 0 else 16] = -button_gate

    # Early colour-12 decrease, then a brief colour-8 decrease window.
    scale = np.float32(10000)
    h12_decrease = 2 * colours.index(12)
    Uh[h12_decrease, 27] += 63 * scale
    Uh[h12_decrease, 28] -= 61 * scale

    h8_decrease = 2 * colours.index(8)
    Uh[h8_decrease, 29] += scale
    Uh[h8_decrease, 30] -= 2 * scale
    Uh[h8_decrease, 31] += scale

    # On the first actor step only, reset if the bar is already shorter
    # than a full bar. A previous click changes the memory sign.
    p["Wz"][0, 32 + 27] = -1
    p["bz"][0] = 1
    p["bm"][0] = 1
    p["Wma"][0, 6] = -2
    p["Wo"][0, 0] = 100
    p["Wo"][0, 128] = 100
    p["bo"][0] = -77
    p["bo"][6] = 1

    p["Wc"][0, 18] = -button_gate
    p["Wc"][0, 32:48] = 1

    return p

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

    # Endpoints next to a black selected body. Green overlaps are
    # included so that a partly matched pair retains its bounds.
    for out, bias in ((0, -25.0), (1, -26.0)):
        W1[out, 0, :, :] = 1.0
        W1[out, 8, 2, 2] = 25.0
        W1[out, 3, 2, 2] = 25.0
        b1[out] = bias

    W1[2, 8, 2, 2] = 1.0
    W1[2, 3, 2, 2] = 1.0
    W1[3, 16, 2, 2] = 1.0
    W1[4, 16, 2, 2] = -1.0
    b1[4] = 1.0
    W1[5, 17, 2, 2] = 1.0
    W1[6, 17, 2, 2] = -1.0
    b1[6] = 1.0

    # Grey indicator, excluding the progress bar at row zero.
    for out, bias in ((7, -2.0), (8, -2.02)):
        W1[out, 4, 2, 2] = 2.0
        W1[out, 16, 2, 2] = 1.0
        b1[out] = bias

    W1[9, 12, 2, 2] = 1.0
    W1[9, 14, 2, 2] = 1.0
    W1[10, 3, 2, 2] = 1.0
    W1[11, 8, 2, 2] = 1.0

    W2, b2 = p["W2"], p["b2"]

    # Complete endpoint blocks after local black-body detection.
    for out, bias in ((0, -10.0), (1, -11.0)):
        W2[out, 0, :, :] = 1.0
        W2[out, 1, :, :] = -1.0
        W2[out, 2, 1, 1] = 10.0
        b2[out] = bias

    for out, inp in (
        (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
        (8, 9), (9, 10), (10, 11),
    ):
        W2[out, inp, 1, 1] = 1.0

    W2[7, 7, 1, 1] = 50.0
    W2[7, 8, 1, 1] = -50.0

    W3 = p["W3"]
    W3[0, 2] = 1.0                 # Blue or green endpoint.
    W3[1, 0] = 1.0                 # Endpoint near black body.
    W3[1, 1] = -1.0
    W3[2, 3] = 1.0                 # Row.
    W3[3, 4] = 1.0                 # Reversed row.
    W3[4, 5] = 1.0                 # Column.
    W3[5, 6] = 1.0                 # Reversed column.
    W3[6, 7] = 1.0                 # Grey click candidate.
    W3[7, 8] = 1.0                 # Unmasked click candidate.
    W3[8, 9] = 1.0                 # Green overlap.
    W3[9, 32 + 2] = 1.0            # Endpoint row maximum.
    W3[10, 64 + 2] = 1.0           # Endpoint column maximum.
    W3[11, 32 + 7] = 1.0           # Grey row maximum.
    W3[12, 10] = 1.0               # Blue only.

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Propagate evidence that grey-masked pieces are present.
    Wh[0, 0, 0, 1] = 1.0
    Wh[1, 1, 2, 1] = 1.0
    Uh[0, 11] = 0.01
    Uh[1, 11] = 0.01

    for out, bias in ((2, -0.005), (3, -0.015)):
        Wh[out, 0, 1, 1] = 1.0
        Wh[out, 1, 1, 1] = 1.0
        bh[out] = bias

    # Four endpoint-coordinate extrema, with separate paths for
    # grey-masked and black-selected renderings.
    for j in range(4):
        grey = 4 + j
        black = 8 + j

        Wh[grey, 2, 1, 1] = 10000.0
        Wh[grey, 3, 1, 1] = -10000.0
        Uh[grey, 0] = 100.0
        Uh[grey, 2 + j] = 1.0
        bh[grey] = -200.0

        Wh[black, 2, 1, 1] = -10000.0
        Wh[black, 3, 1, 1] = 10000.0
        Uh[black, 1] = 100.0
        Uh[black, 2 + j] = 1.0
        bh[black] = -100.0

    # Directional proximity fields for the click map.
    Wh[12, 12, 0, 1] = 0.95
    Wh[13, 13, 2, 1] = 0.95
    Wh[14, 14, 1, 0] = 0.95
    Wh[15, 15, 1, 2] = 0.95
    Uh[12, 9] = 1.0
    Uh[13, 9] = 1.0
    Uh[14, 10] = 1.0
    Uh[15, 10] = 1.0

    Wz, bz = p["Wz"], p["bz"]

    # Hinge features encode the four extrema on the 20-cell grid.
    for j in range(4):
        for i in range(20):
            unit = 20 * j + i
            Wz[unit, 4 + j] = 1.0
            Wz[unit, 8 + j] = 1.0
            bz[unit] = -(4.0 + 3.0 * i) / 63.0
        Wz[80 + j, 4 + j] = 1.0
        Wz[80 + j, 8 + j] = 1.0

    Wz[84, 64 + 8] = 4096.0 / 9.0
    Wz[85, 32 + 12] = 1.0

    Wm, Wmz = p["Wm"], p["Wmz"]
    Wma, bm = p["Wma"], p["bm"]

    # Encode each coordinate rank into five temporary memory bits.
    for j in range(4):
        for bit in range(5):
            index = 5 * j + bit
            values = [float((i >> bit) & 1) for i in range(20)]
            slopes = [
                21.0 * (values[i + 1] - values[i])
                for i in range(19)
            ]
            coeffs = [slopes[0]]
            coeffs.extend(slopes[i] - slopes[i - 1]
                          for i in range(1, 19))
            coeffs.append(0.0)
            for i, coefficient in enumerate(coeffs):
                Wmz[index, 20 * j + i] = 10.0 * coefficient
            bm[index] = -5.0

    # On the step after clicking, latch the previous observation's
    # encoded target bounds.
    for i in range(20):
        saved = 20 + i
        Wm[saved, saved] = 20.0
        Wm[saved, i] = 5.0
        Wma[saved, 6] = 30.0
        bm[saved] = -10.0

    Wm[40, 40] = 6.0
    Wma[40, 6] = 12.0
    bm[40] = -3.0

    # Compare width-minus-height of target and selected bounds.
    for unit, sign in ((41, 1.0), (42, -1.0)):
        for j, axis_sign in (
            (0, -1.0), (1, -1.0), (2, 1.0), (3, 1.0)
        ):
            for bit in range(5):
                saved = 20 + 5 * j + bit
                Wm[unit, saved] = (
                    30.0 * sign * axis_sign *
                    (3.0 / 126.0) * (2 ** bit)
                )
            Wmz[unit, 80 + j] = -30.0 * sign * axis_sign
        bm[unit] = -0.6

    Wma[43, 5] = 5.0
    Wma[45, 6] = 5.0

    # Two green blocks without blue can indicate a swapped pair.
    Wmz[44, 84] = 6.0
    Wmz[44, 85] = -12.0
    bm[44] = -9.0
    Wm[46, 44] = 5.0

    Wo, bo = p["Wo"], p["bo"]
    bo[0] = -20.0
    bo[7] = -20.0

    # Move toward the remembered target's bounding-box centre.
    for action, a, b, direction in (
        (1, 0, 1, -1.0),
        (2, 0, 1, 1.0),
        (3, 2, 3, -1.0),
        (4, 2, 3, 1.0),
    ):
        Wo[action, 80 + a] = -5.0 * direction
        Wo[action, 80 + b] = 5.0 * direction
        for bit in range(5):
            weight = (
                5.0 * direction * (3.0 / 126.0) * (2 ** bit)
            )
            Wo[action, 128 + 20 + 5 * a + bit] = weight
            Wo[action, 128 + 20 + 5 * b + bit] = -weight

    bo[5] = 10.0
    Wo[5, 128 + 41] = 3.5
    Wo[5, 128 + 42] = 3.5
    Wo[5, 128 + 43] = -6.0
    Wo[5, 128 + 44] = 2.5
    Wo[5, 128 + 45] = -6.0
    Wo[5, 128 + 46] = 5.0

    # Click initially, then operate on the selected piece.
    Wo[6, 128 + 40] = -20.0

    p["Wc"][0, 6] = 100.0
    p["Wc"][0, 7] = 60.0
    for channel in range(12, 16):
        p["Wc"][0, 32 + channel] = 1.0

    return p

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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Pass the colour planes through to f3.
    for c in range(16):
        p["W1"][c, c, 2, 2] = 1.0
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0

    # Local detectors for a prospective step onto a yellow checkerboard
    # or onto the currently absent axis of a blue bridge.
    for i, (dy, dx) in enumerate(directions):
        ch = 16 + i
        p["W1"][ch, 14, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        p["W1"][ch, 4, 2, 2] = 1.0
        p["W1"][ch, 9, 2, 3] = 1.0
        p["b1"][ch] = -2.0

        ch = 20 + i
        oy, ox = dx, -dy
        p["W1"][ch, 14, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        p["W1"][ch, 4, 2, 2] = 1.0
        p["W1"][ch, 13,
                2 - 2 * dy + 2 * oy,
                2 - 2 * dx + 2 * ox] = 1.0
        p["b1"][ch] = -2.0

    # Detect transparent halves of yellow or pink checkerboards.
    p["W1"][24, 4, 2, 2] = 1.0
    p["W1"][24, 9, 2, 3] = 1.0
    p["W1"][24, 15, 2, 3] = 1.0
    p["b1"][24] = -1.0

    # Coordinates for recording the player's position.
    p["W1"][25, 16, 2, 2] = 1.0
    p["W1"][26, 17, 2, 2] = 1.0

    # Player masks shifted to the left and right destinations.
    p["W1"][28, 14, 2, 4] = 1.0
    p["W1"][29, 14, 2, 0] = 1.0

    for c in list(range(16, 27)) + [28, 29]:
        p["W2"][c, c, 1, 1] = 1.0

    for c in range(16, 24):
        p["W3"][c, c] = 1.0
    p["W3"][28, 28] = 1.0
    p["W3"][29, 29] = 1.0

    # Large horizontal runs identify the yellow and blue buttons.
    for ch, colour in ((30, 9), (31, 8)):
        for dx in (-2, 0, 2):
            p["W1"][ch, colour, 2, 2 + dx] = 1.0
        p["b1"][ch] = -2.0
        for dx in (-1, 0, 1):
            p["W2"][ch, ch, 1, 1 + dx] = 1.0
        p["b2"][ch] = -2.0

    # Goal pixels are colour 11 in a board column.
    p["W3"][24, 11] = 1.0
    p["W3"][24, 64 + 4] = 1.0
    p["b3"][24] = -1.0

    # Approximate blocked background, allowing checkerboards and
    # the row and column through a bridge pivot.
    p["W3"][25, 4] = 1.0
    p["W3"][25, 32 + 13] = -1.0
    p["W3"][25, 64 + 13] = -1.0
    p["W3"][25, 24] = -1.0

    p["W3"][26, 30] = 1.0
    p["W3"][26, 64 + 4] = -1.0
    p["W3"][27, 31] = 1.0
    p["W3"][27, 64 + 4] = -1.0

    # Player row and column, zero away from the player.
    p["W3"][30, 14] = 1.0
    p["W3"][30, 25] = 1.0
    p["b3"][30] = -1.0
    p["W3"][31, 14] = 1.0
    p["W3"][31, 26] = 1.0
    p["b3"][31] = -1.0

    # Broadcast a bridge-switch request and retain it at blue buttons.
    for dy, dx in directions:
        p["Wh"][5, 5, 1 + dy, 1 + dx] = 0.249
    for ch in range(20, 24):
        p["Uh"][5, ch] = 1.0
    p["Wh"][3, 5, 1, 1] = 1.0
    p["Uh"][3, 27] = 64.0
    p["bh"][3] = -64.0

    # Route field: four two-pixel shifts, followed by synchronized
    # pairwise maxima. Goal pixels are fixed sources.
    for i, (dy, dx) in enumerate(directions):
        p["Wh"][6 + i, 15, 1 + dy, 1 + dx] = 1.0

    p["Wh"][10, 6, 0, 1] = 1.0
    p["Wh"][10, 7, 2, 1] = -1.0
    p["Wh"][11, 8, 1, 0] = 1.0
    p["Wh"][11, 9, 1, 2] = -1.0
    p["Wh"][12, 7, 2, 1] = 1.0
    p["Wh"][13, 9, 1, 2] = 1.0

    p["Wh"][14, 10, 1, 1] = 1.0
    p["Wh"][14, 12, 1, 1] = 1.0
    p["Wh"][14, 11, 1, 1] = -1.0
    p["Wh"][14, 13, 1, 1] = -1.0

    p["Wh"][4, 11, 1, 1] = 1.0
    p["Wh"][4, 13, 1, 1] = 1.0

    # Remove incoming feedback at the source.
    p["Uh"][14, 24] = -1000.0
    p["Uh"][4, 24] = -1000.0

    p["Wh"][15, 14, 1, 1] = 1.0
    p["Wh"][15, 4, 1, 1] = 1.0
    p["Uh"][15, 24] = 100.0
    p["bh"][15] = -1.0
    for ch in (0, 1, 3, 5, 25):
        p["Uh"][15, ch] = -1000.0

    # Vertical readings at the player.
    for i, (dy, dx) in enumerate(directions[:2]):
        p["Wh"][i, 6 + i, 1 + dy, 1 + dx] = 1.0
        p["Uh"][i, 14] = 128.0
        p["bh"][i] = -128.0

    # Pack the left and right readings into one channel. The right
    # destination has an additional offset of 128.
    p["Wh"][2, 15, 1, 1] = 1.0
    p["Uh"][2, 28] = 128.0
    p["Uh"][2, 29] = 256.0
    p["bh"][2] = -128.0

    # Decode directional scores from spatial maxima and mean.
    p["Wz"][0, 0] = 1.0
    p["Wz"][1, 1] = 1.0
    p["Wz"][2, 16 + 2] = 1024.0
    p["Wz"][2, 2] = -1.0
    p["Wz"][3, 2] = 1.0
    p["bz"][3] = -128.0

    # Scores for switch requests, gated by their local detectors.
    for i in range(4):
        for base, detector in ((4, 16 + i), (8, 20 + i)):
            z = base + i
            if i < 2:
                p["Wz"][z, i] = 1.0
                offset = 0.0
            elif i == 2:
                p["Wz"][z, 16 + 2] = 1024.0
                p["Wz"][z, 2] = -1.0
                offset = 0.0
            else:
                p["Wz"][z, 2] = 1.0
                offset = -128.0
            p["Wz"][z, 32 + detector] = 100.0
            p["bz"][z] = offset - 99.5

    # Position features for failed-move detection.
    p["Wz"][15, 32 + 30] = 1.0
    p["Wz"][16, 32 + 31] = 1.0
    p["Wmz"][0, 15] = 0.01
    p["Wmz"][1, 16] = 0.01

    for i, (zcoord, mcoord, sign, action) in enumerate((
        (15, 0, 1.0, 1),
        (15, 0, -1.0, 2),
        (16, 1, 1.0, 3),
        (16, 1, -1.0, 4),
    )):
        unit = 2 + i
        p["Wmz"][unit, zcoord] = 100.0 * sign
        p["Wm"][unit, mcoord] = -10000.0 * sign
        p["Wma"][unit, action] = 200.0
        p["bm"][unit] = -199.0

    p["bo"][:] = -1_000_000.0
    for i in range(4):
        p["bo"][1 + i] = 0.0
        p["Wo"][1 + i, i] = 1.0
        p["Wo"][1 + i, 128 + 2 + i] = -5.0

    p["bo"][6] = 5.0
    for i in range(4, 12):
        p["Wo"][6, i] = 1.0

    # Yellow by default; a broadcast bridge request selects blue.
    p["Wc"][0, 26] = 1.0
    p["Wc"][0, 32 + 3] = 1_000_000.0

    return p

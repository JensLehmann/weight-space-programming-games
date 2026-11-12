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
    Wz, bz = p["Wz"], p["bz"]
    Wmz, Wma, bm = p["Wmz"], p["Wma"], p["bm"]
    Wo, bo, Wc = p["Wo"], p["bo"], p["Wc"]

    # Large player, anchored one pixel down and right from top-left.
    for orient in range(4):
        for dy in range(4):
            for dx in range(4):
                if orient == 0:
                    color = 10 if dy < 2 else 9
                elif orient == 1:
                    color = 9 if dy < 2 else 10
                elif orient == 2:
                    color = 10 if dx < 2 else 9
                else:
                    color = 9 if dx < 2 else 10
                W1[orient, color, dy + 1, dx + 1] = 1.0
        b1[orient] = -15.5

    # Small player, with boundary checks.
    for orient in range(4):
        ch = 4 + orient
        for dy in range(2):
            for dx in range(2):
                if orient == 0:
                    color = 10 if dy == 0 else 9
                elif orient == 1:
                    color = 9 if dy == 0 else 10
                elif orient == 2:
                    color = 10 if dx == 0 else 9
                else:
                    color = 9 if dx == 0 else 10
                W1[ch, color, dy + 1, dx + 1] += 1.0

        if orient < 2:
            boundary = [
                (dy, dx) for dy in (-1, 0) for dx in (-2, 1)
            ]
        else:
            boundary = [
                (dy, dx) for dy in (-2, 1) for dx in (-1, 0)
            ]
        for dy, dx in boundary:
            for color in (9, 10):
                W1[ch, color, dy + 2, dx + 2] -= 1.0
        b1[ch] = -3.5

    # Teleport icon.
    W1[8, 11, 2, 2] = 1.0
    W1[8, 3, 1, 2] = 1.0
    b1[8] = -1.5

    # Goal corners.
    for ch, (dy, dx) in zip((9, 10), ((-1, 1), (1, -1))):
        W1[ch, 10, 2, 2] = 1.0
        W1[ch, 9, 2 + dy, 2] = 1.0
        W1[ch, 9, 2, 2 + dx] = 1.0
        b1[ch] = -2.5

    obstacle_colors = (3, 4, 5, 6, 12, 13)

    # Large and small collision footprints.
    for color in obstacle_colors:
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                W1[11, color, dy + 2, dx + 2] = 1.0
        for dy in (-1, 0):
            for dx in (-1, 0):
                W1[12, color, dy + 2, dx + 2] = 1.0

    # Shrink icon and selected board pixels.
    W1[13, 15, 2, 2] = 1.0
    W1[13, 3, 1, 2] = 1.0
    b1[13] = -1.5
    W1[14, 14, 2, 2] = 1.0

    # Distances from spell-board cell centers.
    for i, value in enumerate((50, 55, 60)):
        pos = 15 + 2 * i
        neg = pos + 1
        W1[pos, 16, 2, 2] = 63.0
        b1[pos] = -float(value)
        W1[neg, 16, 2, 2] = -63.0
        b1[neg] = float(value)

    for i, value in enumerate((25, 30, 35)):
        pos = 21 + 2 * i
        neg = pos + 1
        W1[pos, 17, 2, 2] = 63.0
        b1[pos] = -float(value)
        W1[neg, 17, 2, 2] = -63.0
        b1[neg] = float(value)

    W1[27, 16, 2, 2] = 1.0
    W1[28, 17, 2, 2] = 1.0

    # Fire icon.
    W1[29, 6, 2, 2] = 1.0
    W1[29, 3, 1, 2] = 1.0
    b1[29] = -1.5

    # Top-left interior corner of either colored switch.
    W1[30, 13, 2, 2] = 1.0
    W1[30, 12, 2, 2] = 1.0
    W1[30, 6, 2, 1] = 1.0
    W1[30, 6, 1, 2] = 1.0
    b1[30] = -2.5

    # Black spell-guide pixels.
    W1[31, 0, 2, 2] = 1.0

    # Player and goal.
    for ch in range(4):
        W2[0, ch, 1, 1] = 2.0
        W2[1, 4 + ch, 1, 1] = 2.0
    for ch in (9, 10):
        W2[2, ch, 1, 1] = 2.0

    W2[3, 12, 1, 1] = -1.0
    b2[3] = 1.0

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for k, (dy, dx) in enumerate(directions):
        for ch in range(8):
            W2[4 + k, ch, 1 - dy, 1 - dx] = 2.0

    W2[8, 13, 1, 1] = 2.0

    # Cross-pattern differences and black-guide responses.
    cross = {(0, 1), (1, 0), (1, 2), (2, 1)}
    for row in range(3):
        for col in range(3):
            ch = 9 + 3 * row + col
            for f in (
                15 + 2 * row, 16 + 2 * row,
                21 + 2 * col, 22 + 2 * col,
            ):
                W2[ch, f, 1, 1] = -1.0
            if (row, col) in cross:
                b2[ch] = 1.0
                W2[ch, 14, 1, 1] = -1.0
            else:
                W2[ch, 14, 1, 1] = 1.0
            W2[ch, 31, 1, 1] = 20.0

    # Expand each switch source by one pixel in every direction.
    for dy in range(3):
        for dx in range(3):
            W2[18, 30, dy, dx] = 2.0

    # Switch coordinates, gated by its detector.
    W2[19, 28, 1, 1] = 63.0
    W2[19, 30, 1, 1] = 200.0
    b2[19] = -100.0

    W2[20, 28, 1, 1] = -63.0
    W2[20, 30, 1, 1] = 200.0
    b2[20] = -37.0

    W2[21, 27, 1, 1] = 63.0
    W2[21, 30, 1, 1] = 200.0
    b2[21] = -100.0

    W2[22, 27, 1, 1] = -63.0
    W2[22, 30, 1, 1] = 200.0
    b2[22] = -37.0

    W2[23, 27, 1, 1] = 63.0
    W2[24, 28, 1, 1] = 63.0

    # Large-footprint clearance at each two-pixel destination.
    for k, (dy, dx) in enumerate(directions):
        W2[25 + k, 11, 1 + dy, 1 + dx] = -1.0
        b2[25 + k] = 1.0

    W2[29, 8, 1, 1] = 2.0
    W2[30, 29, 1, 1] = 2.0

    # Any selected cell in the spell board.
    W2[31, 14, 1, 1] = 1.0
    for f in (16, 19, 22, 25):
        W2[31, f, 1, 1] = -2.0

    for ch in range(18):
        W3[ch, ch] = 1.0

    # Switches are wave sources, and their expanded source pixels are
    # treated as traversable for propagation.
    W3[2, 18] = 1.0
    W3[3, 18] = 1.0

    # Axial switch alignment at the player.
    for ch in range(18, 22):
        W3[ch, 0] = 100.0
        W3[ch, 1] = 100.0
        b3[ch] = -100.0

    W3[18, 64 + 22] = 1.0
    W3[18, 23] = 1.0
    b3[18] -= 63.0

    W3[19, 64 + 21] = 1.0
    W3[19, 23] = -1.0

    W3[20, 32 + 20] = 1.0
    W3[20, 24] = 1.0
    b3[20] -= 63.0

    W3[21, 32 + 19] = 1.0
    W3[21, 24] = -1.0

    W3[22, 18] = 1.0
    W3[26, 31] = 1.0

    # Large-player destination veto.
    for k in range(4):
        if k < 2:
            W3[27 + k, 64 + 0] = 1.0
        else:
            W3[27 + k, 32 + 0] = 1.0
        W3[27 + k, 25 + k] = -1.0

    W3[31, 29] = 1.0

    # Three-stage four-neighbor max wave. Small source injection and
    # decay keep values bounded while preserving a distance gradient.
    Wh[1, 0, 0, 1] = 1.0
    Wh[1, 0, 2, 1] = -1.0
    Wh[2, 0, 1, 0] = 1.0
    Wh[2, 0, 1, 2] = -1.0
    Wh[3, 0, 2, 1] = 1.0
    Wh[4, 0, 1, 2] = 1.0

    Wh[5, 1, 1, 1] = 1.0
    Wh[5, 3, 1, 1] = 1.0
    Wh[5, 2, 1, 1] = -1.0
    Wh[5, 4, 1, 1] = -1.0
    Wh[6, 2, 1, 1] = 1.0
    Wh[6, 4, 1, 1] = 1.0

    Wh[0, 5, 1, 1] = 0.98
    Wh[0, 6, 1, 1] = 0.98
    Uh[0, 2] = 0.02
    Uh[0, 3] = 2.0
    bh[0] = -2.0

    # Sample two pixels away, with the large-footprint veto.
    for k, (dy, dx) in enumerate(directions):
        Wh[8 + k, 0, 1 + dy, 1 + dx] = 1.0
        Uh[8 + k, 4 + k] = 4.0
        Uh[8 + k, 27 + k] = -4.0
        bh[8 + k] = -4.0

    # Propagate a saturated small-player signal to the teleport icon.
    for dy, dx in directions:
        Wh[12, 14, 1 + dy, 1 + dx] = 1.0
        Wh[13, 14, 1 + dy, 1 + dx] = 1.0
    Uh[12, 1] = 4.0
    Uh[13, 1] = 4.0
    bh[13] = -1.0
    Wh[14, 12, 1, 1] = 1.0
    Wh[14, 13, 1, 1] = -1.0

    Wh[15, 14, 1, 1] = 1.0
    Uh[15, 31] = 10.0
    bh[15] = -10.0

    for k in range(4):
        Wz[k, 8 + k] = 1.0
        Wo[1 + k, k] = 1.0

    # Shrink when large and the shrink icon is present.
    Wz[4, 32] = 1.0
    Wz[4, 32 + 8] = 1.0
    bz[4] = -1.5
    Wo[6, 4] = 200.0

    # Clamp directional switch-alignment scores.
    for k in range(4):
        a = 10 + 2 * k
        Wz[a, 32 + 18 + k] = 1.0
        Wz[a + 1, 32 + 18 + k] = 1.0
        bz[a + 1] = -0.5
        Wo[1 + k, a] += 10.0
        Wo[1 + k, a + 1] -= 10.0

    Wz[20, 32 + 26] = 1.0
    Wz[21, 32 + 22] = 1.0

    # Fire after facing an aligned switch, or continue a partial pattern.
    for k in range(4):
        a = 10 + 2 * k
        Wmz[k, a] = 6.0
        Wmz[k, a + 1] = -6.0
        Wmz[k, 20] = 4.0
        Wmz[k, 21] = 4.0
        Wma[k, 1 + k] = 4.0
        bm[k] = -10.5
        Wo[6, 128 + k] = 20.0

    # Attempt teleport if small and no wave reaches a candidate.
    Wz[25, 32 + 1] = 2.0
    Wz[25, 32 + 31] = 2.0
    for k in range(4):
        Wz[25, 8 + k] = -10.0
    bz[25] = -3.0
    Wo[6, 25] = 40.0

    bo[0] = -100.0
    bo[6] = 64.0

    for row in range(3):
        for col in range(3):
            ch = 9 + 3 * row + col
            if (row, col) in cross:
                priority = 4.0 - (3 * row + col) * 0.1
            else:
                priority = 10.0 - (3 * row + col) * 0.1
            Wc[0, ch] = priority

    Wc[0, 32 + 15] = 30.0

    return p

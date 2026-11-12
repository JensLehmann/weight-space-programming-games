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
    W2 = p["W2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wz, bz = p["Wz"], p["bz"]
    Wm, Wmz, bm = p["Wm"], p["Wmz"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    colors = (8, 9, 12, 14)
    corners = ((-1, -1), (-1, 2), (2, -1), (2, 2))

    for i, color in enumerate(colors):
        # Free board blocks, unmarked goal blocks, marked goal blocks,
        # and blocks captured at a perpendicular cable.
        for channel in (i, 4 + i, 8 + i, 18 + i):
            for dy, dx in corners:
                W1[channel, color, dy + 2, dx + 2] = 1.0

        W1[i, color, 2, 2] = 1.0
        W1[i, 2, 2, 0] = -2.0
        b1[i] = -4.0

        W1[4 + i, color, 2, 2] = 1.0
        b1[4 + i] = -4.0

        W1[8 + i, 0, 2, 2] = 1.0
        b1[8 + i] = -4.0

        # Selected horizontal cable at the block's left edge and
        # perpendicular cable at its top edge.
        W1[18 + i, 2, 2, 0] = 1.0
        W1[18 + i, 2, 0, 2] = 1.0
        W1[18 + i, 3, 0, 2] = 1.0
        b1[18 + i] = -5.0

    # Selected horizontal-cable stripe detectors.
    W1[12, 1, 2, 2] = 1.0
    W1[12, 2, 3, 2] = 1.0
    b1[12] = -1.0

    W1[13, 2, 2, 2] = 1.0
    W1[13, 1, 3, 2] = 1.0
    b1[13] = -1.0

    # Selected head detector.
    for dy, dx in corners:
        W1[14, 0, dy + 2, dx + 2] = 1.0
    for dy in (0, 1):
        for dx in (0, 1):
            W1[14, 6, dy + 2, dx + 2] = 1.0
            W1[14, 15, dy + 2, dx + 2] = 1.0
    b1[14] = -7.0

    # Board/goal-strip masks and column coordinate.
    W1[15, 16, 2, 2] = 63.0
    b1[15] = -52.0
    W1[16, 16, 2, 2] = -63.0
    b1[16] = 55.0
    W1[17, 17, 2, 2] = 1.0

    for i in range(4):
        W2[i, i, 1, 1] = 1.0
        W2[i, 15, 1, 1] = -1.0

        W2[4 + i, 4 + i, 1, 1] = 1.0
        W2[4 + i, 16, 1, 1] = -1.0

        W2[8 + i, 8 + i, 1, 1] = 1.0
        W2[8 + i, 16, 1, 1] = -1.0

        W2[18 + i, 18 + i, 1, 1] = 1.0
        W2[18 + i, 15, 1, 1] = -1.0

    W2[12, 12, 1, 1] = 1.0
    W2[12, 13, 1, 1] = 1.0
    W2[12, 15, 1, 1] = -1.0

    W2[13, 14, 1, 1] = 1.0
    W2[13, 15, 1, 1] = -1.0
    W2[14, 17, 1, 1] = 1.0

    # Broadcast head and free-block indicators across their rows.
    W3[0, 32 + 13] = 1.0
    for i in range(4):
        W3[1 + i, 32 + i] = 1.0

        # Larger value corresponds to a goal further left.
        W3[5 + i, 4 + i] = 64.0
        W3[5 + i, 14] = -63.0

        W3[9 + i, 8 + i] = 1.0

        W3[14 + i, 32 + 13] = 1.0
        W3[14 + i, 32 + i] = 1.0
        b3[14 + i] = -1.0

        W3[18 + i, 18 + i] = 1.0

    W3[13, 12] = 1.0

    # Propagate block-row information up and down, sampling it
    # at the selected head's row.
    for i in range(4):
        down = 4 * i
        up = down + 1
        above_head = down + 2
        below_head = down + 3

        Wh[down, down, 0, 1] = 1.0
        Uh[down, 1 + i] = 1.0

        Wh[up, up, 2, 1] = 1.0
        Uh[up, 1 + i] = 1.0

        Wh[above_head, down, 0, 1] = 1.0
        Uh[above_head, 0] = 64.0
        bh[above_head] = -64.0

        Wh[below_head, up, 2, 1] = 1.0
        Uh[below_head, 0] = 64.0
        bh[below_head] = -64.0

    # g layout: h max, h mean, f3 max, f3 mean.
    comparator = {}
    unit = 0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            comparator[i, j] = unit
            Wz[unit, 32 + 5 + j] = 1.0
            Wz[unit, 32 + 5 + i] = -1.0
            bz[unit] = 0.5
            unit += 1

    for i in range(4):
        Wz[20 + 3 * i, 4 * i + 2] = 1.0
        Wz[21 + 3 * i, 4 * i + 3] = 1.0
        Wz[22 + 3 * i, 32 + 14 + i] = 1.0

    # Approximate compact-cable indicator.
    Wz[41, 64 + 13] = -4096.0
    for i in range(4):
        Wz[41, 64 + 9 + i] = 8192.0
    bz[41] = 2.0

    # Count marked goal blocks.
    for i in range(4):
        Wz[42, 64 + 9 + i] = 4096.0

    # Captured block on a perpendicular cable.
    for i in range(4):
        Wz[43, 32 + 18 + i] = 1.0

    # Previous marked-block count.
    Wmz[0, 42] = 0.01

    # Retraction latch: a new mark sets it; a compact cable clears it.
    Wm[1, 1] = 4.0
    Wm[1, 0] = -4000.0
    Wmz[1, 42] = 40.0
    Wmz[1, 41] = -100.0
    bm[1] = 0.5

    # Gate directional evidence by the leftmost unmatched goal color.
    for i in range(4):
        for direction in range(3):
            m = 2 + 3 * i + direction
            Wmz[m, 20 + 3 * i + direction] = 8.0
            for j in range(4):
                if i != j:
                    Wmz[m, comparator[i, j]] = -100.0
            bm[m] = -4.0

    for i in range(4):
        Wo[1, 128 + 2 + 3 * i] = 1.0
        Wo[2, 128 + 3 + 3 * i] = 1.0
        Wo[4, 128 + 4 + 3 * i] = 1.0

    bo[1] = 4.0
    bo[2] = 4.0
    bo[4] = 4.001

    Wo[3, 128 + 1] = 5.0
    bo[3] = 1.0

    # Move a captured block down while it remains on a crossing cable.
    Wo[2, 43] = 10.0

    return p

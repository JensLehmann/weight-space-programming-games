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

    w, b = p["W1"], p["b1"]

    # Anchor for the first horizontal instruction bit.
    w[0, 1, 2, 2] = 1
    w[0, 5, 2, 2] = 1
    w[0, 0, 2, 1] = 1
    w[0, 0, 1, 2] = 1
    w[0, 0, 0, 2] = -1
    w[0, 0, 2, 3] = -1
    b[0] = -2

    w[1, 1, 2, 2] = 1
    w[2, 9, 2, 2] = 1

    # Lower-screen mask and column coordinate.
    w[3, 16, 2, 2] = 1
    w[4, 16, 2, 2] = 1
    b[3] = -50 / 63
    b[4] = -51 / 63
    w[5, 17, 2, 2] = 1

    # Column masks.
    w[6, 17, 2, 2] = -63
    w[7, 17, 2, 2] = -63
    b[6], b[7] = 42, 41
    w[8, 17, 2, 2] = -63
    w[9, 17, 2, 2] = -63
    b[8], b[9] = 37, 36

    # Upside-down piece cue.
    for dy in (0, 1):
        for dx in (-1, 0, 1, 2):
            w[10, 11, 2 + dy, 2 + dx] = 1
    for dx in (-1, 2):
        w[10, 11, 1, 2 + dx] = 1
    for dx in (0, 1):
        w[10, 11, 1, 2 + dx] = -1
    b[10] = -9

    # Enlarged piece cue.
    w[11, 11, :, :] = 1
    b[11] = -24

    # Vertical instruction bits, off and on.
    for out, color in ((12, 1), (13, 5)):
        for dy in (0, 1, 2):
            w[out, color, 2 + dy, 2] = 1
        w[out, 0, 2, 1] = 1
        w[out, 0, 2, 3] = 1
        w[out, 0, 1, 2] = 1
        b[out] = -5

    # Selected horizontal bit.
    for dx in (0, 1, 2):
        w[14, 5, 2, 2 + dx] = 1
    w[14, 0, 2, 1] = 1
    w[14, 0, 1, 2] = 1
    w[14, 0, 3, 2] = 1
    b[14] = -5

    # Three hinges per row form exact integer-row indicators.
    rows = (35, 41, 47, 39, 45)
    for i, row in enumerate(rows):
        for j, threshold in enumerate((row - 1, row, row + 1)):
            ch = 15 + 3 * i + j
            w[ch, 16, 2, 2] = 63
            b[ch] = -threshold

    # Solid patch inside the run button.
    w[30, 9, :, :] = 1
    b[30] = -24

    w, b = p["W2"], p["b2"]

    w[0, 0, 1, 1] = 1
    w[0, 1, 1, 1] = 1
    b[0] = -1

    w[1, 0, 0, 1] = 1
    w[2, 1, 1, 1] = 1
    w[3, 2, 1, 1] = 1
    w[4, 3, 1, 1] = 63
    w[4, 4, 1, 1] = -63
    w[5, 5, 1, 1] = 1
    w[6, 6, 1, 1] = 1
    w[6, 7, 1, 1] = -1
    w[7, 8, 1, 1] = 1
    w[7, 9, 1, 1] = -1
    w[8, 10, 1, 1] = 1
    w[9, 11, 1, 1] = 1
    w[10, 12, 1, 1] = 1
    w[11, 13, 1, 1] = 1
    w[12, 14, 1, 1] = 1

    for i in range(15):
        w[13 + i, 15 + i, 1, 1] = 1
    w[28, 30, 1, 1] = 1

    w, b = p["W3"], p["b3"]
    w[0, 0] = 1
    w[1, 1] = 1
    w[2, 2] = 1

    # Button interior AND lower-screen mask.
    w[3, 28] = 1
    w[3, 4] = 1
    b[3] = -1
    w[4, 5] = 1

    # Broadcast visual cues along their source rows.
    w[5, 32 + 8] = 1
    w[6, 32 + 9] = 1
    w[7, 6] = 1
    w[8, 7] = 1

    def row_gated(out, feature, row_index):
        w[out, feature] = 1
        base = 13 + 3 * row_index
        w[out, base] = 1
        w[out, base + 1] = -2
        w[out, base + 2] = 1
        b[out] = -1

    row_gated(9, 10, 1)
    row_gated(10, 10, 2)
    row_gated(11, 11, 0)
    row_gated(12, 11, 1)
    row_gated(13, 11, 2)
    row_gated(14, 12, 3)
    row_gated(15, 12, 4)

    uh, wh, bh = p["Uh"], p["Wh"], p["bh"]

    # Propagate and clip the piece-orientation and size cues.
    uh[2, 5] = 1
    wh[2, 2, 0, 1] = 1
    wh[3, 2, 1, 1] = 1
    bh[3] = -1

    uh[4, 6] = 1
    wh[4, 4, 0, 1] = 1
    wh[5, 4, 1, 1] = 1
    bh[5] = -1

    uh[6, 7] = 1
    wh[6, 4, 1, 1] = 1
    wh[6, 5, 1, 1] = -1
    bh[6] = -1

    uh[1, 1] = 1

    # Instruction-bit edits.
    wh[0, 1, 0, 0] = 1
    uh[0, 2] = 1
    wh[0, 2, 1, 1] = -1
    wh[0, 3, 1, 1] = 1
    wh[0, 6, 1, 1] = -1
    bh[0] = -1

    uh[7, 10] = 1
    wh[7, 2, 1, 1] = 1
    wh[7, 3, 1, 1] = -1
    bh[7] = -1

    uh[8, 9] = 1
    uh[8, 8] = 1
    wh[8, 4, 1, 1] = 1
    wh[8, 5, 1, 1] = -1
    bh[8] = -2

    uh[9, 11] = 1
    wh[9, 2, 1, 1] = 1
    wh[9, 3, 1, 1] = -1
    bh[9] = -1

    uh[10, 11] = 1
    wh[10, 6, 1, 1] = 1
    bh[10] = -1

    uh[12, 8] = 1
    wh[12, 4, 1, 1] = 1
    wh[12, 5, 1, 1] = -1
    bh[12] = -1
    uh[11, 12] = 1
    wh[11, 12, 1, 1] = -1

    uh[13, 13] = 1
    wh[13, 2, 1, 1] = -1
    wh[13, 3, 1, 1] = 1

    # Edits outrank the run button; column breaks ties.
    wc = p["Wc"][0]
    wc[0] = 100
    wc[3] = 5
    wc[4] = 0.01
    wc[14] = 100
    wc[15] = 100
    for channel in (0, 7, 8, 9, 10, 11, 13):
        wc[32 + channel] = 100

    # A slow episode counter and a latch that prevents repeated resets.
    p["Wm"][0, 0] = 1
    p["bm"][0] = 0.01
    p["Wm"][1, 1] = 2
    p["Wma"][1, 0] = 4
    p["Wo"][0, 128] = 20
    p["Wo"][0, 129] = -20
    p["bo"][0] = -4.6

    p["bo"][6] = 1
    return p

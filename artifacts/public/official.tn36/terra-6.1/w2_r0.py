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
    W2, b2 = p["W2"], p["b2"]
    W3 = p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def stroke(ch, colour, vertical):
        if vertical:
            for dy in (-1, 0, 1):
                W1[ch, colour, 2 + dy, 2] = 1
            for dx, dy in ((-1, 0), (1, 0), (0, -2), (0, 2)):
                W1[ch, 0, 2 + dy, 2 + dx] = 1
        else:
            for dx in (-1, 0, 1):
                W1[ch, colour, 2, 2 + dx] = 1
            for dx, dy in ((-2, 0), (2, 0), (0, -1), (0, 1)):
                W1[ch, 0, 2 + dy, 2 + dx] = 1
        b1[ch] = -6

    stroke(0, 1, False)
    stroke(1, 5, False)
    stroke(2, 1, True)
    stroke(3, 5, True)

    def row_below(ch, value):
        W1[ch, 16, 2, 2] = -63
        b1[ch] = value

    def row_above(ch, value):
        W1[ch, 16, 2, 2] = 63
        b1[ch] = -value

    def col_below(ch, value):
        W1[ch, 17, 2, 2] = -63
        b1[ch] = value

    def col_above(ch, value):
        W1[ch, 17, 2, 2] = 63
        b1[ch] = -value

    bit_rows = (33, 36, 39, 42, 45, 48)
    for i, row in enumerate(bit_rows):
        row_below(4 + 2 * i, row)
        row_above(5 + 2 * i, row)

    col_below(16, 32)
    col_above(17, 61)
    W1[18, 17, 2, 2] = 63
    col_below(19, 2)
    col_above(20, 2)
    W1[21, 2, 2, 2] = 1

    W1[22, 9, :, :] = 1
    b1[22] = -24

    W1[23, 6, 2, 2] = 1

    W1[24, 11, 1:5, 2:5] = 1
    b1[24] = -11

    W1[25, 11, 1:5, 1:5] = 1
    W1[25, 11, 0, 1:5] = -1
    b1[25] = -15

    row_below(26, 24)
    row_above(27, 24)
    row_below(30, 10)
    row_above(31, 10)

    for bit in range(6):
        for colour_state in range(2):
            out = 2 * bit + colour_state
            source = (0 if bit % 2 == 0 else 2) + colour_state
            W2[out, source, 1, 1] = 1
            W2[out, 4 + 2 * bit, 1, 1] = -3
            W2[out, 5 + 2 * bit, 1, 1] = -3
            W2[out, 16, 1, 1] = -3
            W2[out, 17, 1, 1] = -3

    W2[12, 0, 1, 1] = 1
    W2[12, 2, 1, 1] = 1
    W2[12, 4, 1, 1] = -3
    W2[12, 15, 1, 1] = -3

    W2[13, 21, 1, 1] = 1
    W2[13, 4, 2, 1] = -3
    W2[13, 5, 2, 1] = -3
    W2[13, 19, 1, 1] = -3
    W2[13, 20, 1, 1] = -3

    W2[14, 22, 1, 1] = 1

    W2[15, 23, 1, 1] = 1
    W2[15, 26, 1, 1] = -3
    W2[15, 27, 1, 1] = -3
    W2[15, 16, 1, 1] = -3

    W2[16, 25, 1, 1] = 1
    W2[16, 16, 1, 1] = -3
    W2[16, 5, 1, 1] = -3

    for out in (17, 18, 19):
        W2[out, 24, 1, 1] = 1
        W2[out, 16, 1, 1] = -3
        W2[out, 5, 1, 1] = -3
    W2[17, 26, 1, 1] = -3
    W2[18, 27, 1, 1] = -3
    W2[18, 30, 1, 1] = -3
    W2[19, 31, 1, 1] = -3

    boundaries = (36, 41, 46, 51, 56)
    greater, less = {}, {}
    for i, value in enumerate(boundaries):
        g = 20 + 2 * i
        l = g + 1
        greater[value], less[value] = g, l
        W2[g, 18, 1, 1] = 1
        b2[g] = -value
        W2[l, 18, 1, 1] = -1
        b2[l] = value

    W3[0, 12] = 1

    def mismatch(out, code, lower=None, upper=None):
        for bit in range(6):
            state = 0 if (code & (1 << bit)) else 1
            W3[out, 2 * bit + state] = 1
        if lower is not None:
            W3[out, less[lower]] = -3
        if upper is not None:
            W3[out, greater[upper]] = -3

    mismatch(1, 33)

    mismatch(2, 9, upper=36)
    mismatch(3, 1, lower=36, upper=41)
    mismatch(4, 3, lower=41)

    mismatch(5, 2, upper=51)
    mismatch(6, 33, lower=51)

    mismatch(7, 33, upper=51)
    mismatch(8, 2, lower=51, upper=56)
    mismatch(9, 0, lower=56)

    mismatch(10, 1, upper=46)
    mismatch(11, 3, lower=46, upper=51)
    mismatch(12, 1, lower=51, upper=56)
    mismatch(13, 0, lower=56)

    for i, source in enumerate((13, 16, 15, 17, 18, 19)):
        W3[14 + i, 32 + source] = 1

    W3[20, 14] = 1

    for i in range(6):
        Wh[i, i, 0, 1] = 1
        Uh[i, 14 + i] = 1

    Uh[6, 0] = 1
    Wh[6, 0, 1, 1] = -1

    Uh[7, 1] = 2
    Wh[7, 0, 1, 1] = 1
    Wh[7, 1, 1, 1] = -1
    Wh[7, 2, 1, 1] = -1
    bh[7] = -2

    for f in (2, 3, 4):
        Uh[8, f] = 2
    Wh[8, 1, 1, 1] = 1
    Wh[8, 2, 1, 1] = -1
    bh[8] = -2

    for out, fs, stage in (
        (9, (5, 6), 3),
        (10, (7, 8, 9), 4),
        (11, (10, 11, 12, 13), 5),
    ):
        for f in fs:
            Uh[out, f] = 2
        Wh[out, 2, 1, 1] = 8
        Wh[out, 1, 1, 1] = -8
        Wh[out, stage, 1, 1] = 1
        bh[out] = -10

    for h in range(6, 12):
        p["Wc"][0, 32 + h] = 2
    p["Wc"][0, 20] = 1
    p["bo"][6] = 1

    return p

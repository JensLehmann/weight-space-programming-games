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

    colors = (6, 8, 9, 11, 12, 14, 15)
    W1, b1 = p["W1"], p["b1"]
    W2 = p["W2"]
    W3, b3 = p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    def pixel(unit, color, dy, dx, weight):
        W1[unit, color, dy + 2, dx + 2] += weight

    def template(unit, requirements):
        for color, dy, dx in requirements:
            pixel(unit, color, dy, dx, 2.0)
        b1[unit] = 1.0 - 2.0 * len(requirements)

    outside = (
        [(-2, dx) for dx in range(-2, 3)]
        + [(dy, -2) for dy in range(-1, 3)]
    )

    for k, color in enumerate(colors):
        req = [
            (color, dy, dx)
            for dy in range(-1, 3)
            for dx in range(-1, 3)
        ]
        req += [(4, dy, dx) for dy, dx in outside]
        template(k, req)

        req = []
        for dy in range(-1, 3):
            for dx in range(-1, 3):
                rim = dy in (-1, 2) or dx in (-1, 2)
                req.append((color if rim else 4, dy, dx))
        req += [(4, dy, dx) for dy, dx in outside]
        template(7 + k, req)

        req = [(color, dy, dx) for dy, dx in outside]
        req += [
            (5, dy, dx)
            for dy in range(-1, 3)
            for dx in range(-1, 3)
        ]
        template(14 + k, req)

    template(21, [
        (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1),
        (4, -1, 0), (4, -1, 1),
        (4, 0, -1), (4, 1, -1), (4, -1, -1),
    ])
    template(22, [(0, dy, dx) for dy, dx in outside])

    pixel(23, 0, -2, 0, 1.0)
    W1[24, 16, 2, 2] = 1.0
    W1[25, 17, 2, 2] = 1.0
    W1[26, 18, 2, 2] = 1.0

    W1[27, 16, 2, 2] = 1.0
    b1[27] = -52.0 / 63.0
    W1[28, 16, 2, 2] = -1.0
    b1[28] = 54.0 / 63.0

    def f2(out, inp, weight=1.0, dy=0, dx=0):
        W2[out, inp, dy + 1, dx + 1] += weight

    for k in range(7):
        f2(k, k)
        f2(k, 28, -63.0)
        f2(7 + k, 14 + k)

    for k in range(7):
        f2(14, k)
        f2(15, 7 + k)
    f2(14, 27, -63.0)
    f2(15, 27, -63.0)
    f2(16, 21)
    f2(16, 27, -63.0)

    f2(17, 22)
    f2(18, 23, dy=-1)
    f2(18, 27, -63.0)

    # One-row upward shift prevents the phase signal from also
    # rewarding the selectable palette anchors themselves.
    f2(19, 22, dy=1)

    for k, color in enumerate(colors):
        f2(20, k)
        f2(20, 7 + k)
        f2(21, k, float(color))
        f2(21, 7 + k, float(color))
    f2(20, 28, -63.0)
    f2(21, 28, -16.0 * 63.0)

    f2(22, 24)
    f2(23, 25)
    f2(24, 26)

    for k in range(7):
        f2(26, 7 + k)
    f2(26, 28, -63.0)

    for k, color in enumerate(colors):
        f2(27, 14 + k, float(color))
        f2(29, 14 + k)

    f2(30, 22)
    f2(30, 27, -63.0)

    def local(out, inp, weight=1.0):
        W3[out, inp] += weight

    def rowmax(out, inp, weight=1.0):
        W3[out, 32 + inp] += weight

    def colmax(out, inp, weight=1.0):
        W3[out, 64 + inp] += weight

    local(0, 16)
    rowmax(0, 18)
    b3[0] = -1.0

    local(1, 14)
    local(1, 15, 2.0)
    rowmax(1, 18, 4.0)
    b3[1] = -4.0

    local(2, 16)
    local(3, 20)
    local(4, 21)
    colmax(5, 29)
    colmax(6, 27)

    for k in range(7):
        colmax(8 + k, 7 + k)
        rowmax(8 + k, k)
        b3[8 + k] = -1.0

    # Critical correction: broadcast selection horizontally before
    # the recurrent block carries it upward.
    rowmax(15, 19)

    local(16, 22)
    local(17, 23)
    local(18, 17)
    local(19, 30)
    local(20, 26)

    def rec(out, inp, weight=1.0, dy=0, dx=0):
        Wh[out, inp, dy + 1, dx + 1] += weight

    # Prefix holes / eligible remaining target colours.
    rec(0, 0, dx=-1)
    Uh[0, 0] = 1.0
    for k in range(7):
        Uh[0, 8 + k] = 1.0

    # Contributions before the first root hole.
    Uh[1, 1] = 1.0
    rec(1, 0, -4.0)

    # Vertical transport of root-prefix contributions.
    rec(2, 2, dy=-1)
    rec(2, 1)

    # Target suffix and prefix ranks.
    rec(3, 3, dx=1)
    Uh[3, 5] = 1.0
    rec(6, 6, dx=-1)
    Uh[6, 5] = 1.0

    # Sum contributions across the current row.
    rec(4, 4, dx=-1)
    rec(4, 2)
    rec(5, 5, dx=1)
    rec(5, 2)

    # Absolute difference between emitted-prefix length and target rank.
    rec(7, 4)
    rec(7, 5)
    rec(7, 2, -1.0)
    rec(7, 6, -1.0)
    Uh[7, 5] = 1.0

    rec(8, 4, -1.0)
    rec(8, 5, -1.0)
    rec(8, 2)
    rec(8, 6)
    Uh[8, 5] = -1.0

    Uh[9, 6] = 1.0
    rec(9, 7, -16.0)
    rec(9, 8, -16.0)

    # Once the root's expansion reaches the target length, route the
    # first target colour still represented by a solid palette token.
    for k, color in enumerate(colors):
        Uh[10, 8 + k] = float(color + 32)
    Uh[10, 5] = 16.0
    bh[10] = -16.0
    rec(10, 0, -16.0)
    rec(10, 4, 16.0)
    rec(10, 5, 16.0)
    rec(10, 2, -16.0)
    rec(10, 3, -16.0)
    rec(10, 6, -16.0)

    rec(11, 11, dx=-1)
    rec(11, 9)
    rec(11, 10)
    rec(12, 12, dx=1)
    rec(12, 9)
    rec(12, 10)

    K = 32.0
    for out, sign in ((13, 1.0), (14, -1.0)):
        rec(out, 11, sign)
        rec(out, 12, sign)
        rec(out, 9, -sign)
        rec(out, 10, -sign)
        Uh[out, 4] = -sign
        Uh[out, 3] = -K
        bh[out] = K

    # Uniform selection signal at every column above the palette.
    rec(15, 15, dy=1)
    Uh[15, 15] = 1.0

    p["Wz"][0, 32 + 2] = 1.0
    p["Wz"][1, 32 + 18] = 1.0
    p["Wz"][2, 32 + 19] = 1.0

    p["Wma"][0, 5] = 3.0

    p["bo"][:] = -100.0
    p["bo"][5] = 3.0
    p["bo"][6] = 2.0
    p["Wo"][6, 0] = 40.0
    p["Wo"][6, 1] = 40.0

    p["bo"][0] = -30.0
    p["Wo"][0, 128] = 80.0
    p["Wo"][0, 2] = 200.0
    p["bo"][7] = -30.0
    p["Wo"][7, 128] = 60.0

    wc = p["Wc"][0]
    wc[2] = 64.0
    wc[3] = 32.0
    wc[16] = -4.032
    wc[17] = -0.063
    wc[20] = 0.125
    wc[32 + 13] = -1.0
    wc[32 + 14] = -1.0
    wc[32 + 15] = 100.0

    return p

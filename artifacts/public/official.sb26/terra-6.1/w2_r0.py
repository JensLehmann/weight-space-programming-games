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

    # Goal-colour signals weighted by leftmost occurrence.
    for i, color in enumerate(colors):
        W1[i, color, 2, 2] = 10
        for other in range(16):
            if other not in (4, 5):
                W1[i, other, 2, 1] -= 10
        W1[i, 16, 2, 2] = -189
        W1[i, 17, 2, 2] = -1
        b1[i] = -5

    # Solid or hollow four-by-four tile anchors.
    for i, color in enumerate(colors):
        c = 7 + i
        for dy, dx in ((0, -1), (0, 0), (0, 1), (0, 2),
                       (1, -1), (1, 2)):
            W1[c, color, 2 + dy, 2 + dx] += 1
        W1[c, color, 2, 0] -= 1
        W1[c, color, 1, 2] -= 1
        b1[c] = -5

    # Top-left pixel of an empty spot's red marker.
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        W1[14, 2, 2 + dy, 2 + dx] += 1
    W1[14, 2, 2, 1] -= 10
    W1[14, 2, 1, 2] -= 10
    b1[14] = -3

    # Selection outline around a coloured inventory tile.
    for dy, dx in ((0, -1), (-1, 0), (-1, -1)):
        W1[15, 0, 2 + dy, 2 + dx] = 1
    for color in colors:
        W1[15, color, 2, 2] = 1
    b1[15] = -3

    # Difference of these ramps is one on inventory rows.
    W1[16, 16, 2, 2] = 63
    b1[16] = -54
    W1[17, 16, 2, 2] = 63
    b1[17] = -55

    W1[18, 16, 2, 2] = 1
    W1[19, 17, 2, 2] = 1

    # A colour-3 pixel at the right screen edge indicates a spent bar
    # when restricted to the energy-bar row in the next layer.
    W1[27, 3, 2, 2] = 1
    W1[27, 18, 2, 3] = -1
    W1[28, 16, 2, 2] = 63
    b1[28] = -52
    W1[29, 16, 2, 2] = 63
    b1[29] = -53

    W2, b2 = p["W2"], p["b2"]

    for i in range(7):
        W2[i, i, 1, 1] = 1
        W2[7 + i, 7 + i, 1, 1] = 1
        W2[7 + i, 16, 1, 1] = 1
        W2[7 + i, 17, 1, 1] = -1
        b2[7 + i] = -1

    # Ignore empty inventory positions.
    W2[14, 14, 1, 1] = 1
    W2[14, 16, 1, 1] = -1
    W2[14, 17, 1, 1] = 1

    W2[15, 15, 1, 1] = 1
    W2[15, 16, 1, 1] = 1
    W2[15, 17, 1, 1] = -1
    b2[15] = -1

    # Slot priority: leftward first, with a lower-right bias.
    W2[16, 14, 1, 1] = 1
    W2[16, 19, 1, 1] = -0.5
    W2[16, 16, 1, 1] = -1
    W2[16, 17, 1, 1] = 1

    W2[17, 14, 1, 1] = 1
    W2[17, 18, 1, 1] = 1
    W2[17, 19, 1, 1] = 1
    W2[17, 16, 1, 1] = -1
    W2[17, 17, 1, 1] = 1
    b2[17] = -1 - 70 / 63

    # Spent energy indicator.
    W2[27, 27, 1, 1] = 1
    W2[27, 28, 1, 1] = 1
    W2[27, 29, 1, 1] = -1
    b2[27] = -1

    W3 = p["W3"]
    for i in range(7):
        W3[i, 32 + i] = 1
        W3[7 + i, 7 + i] = 1
    W3[14, 14] = 1
    W3[15, 32 + 15] = 1
    W3[17, 16] = 1
    W3[18, 17] = 1
    W3[19, 27] = 1

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # Propagate goal-colour priorities vertically.
    for i in range(7):
        Wh[i, i, 0, 1] = 0.5
        Wh[i, i, 2, 1] = 0.5
        Uh[i, i] = 1

    # Matching inventory scores, suppressed after selection.
    for i in range(7):
        c = 7 + i
        Wh[c, i, 1, 1] = 1
        Wh[c, 15, 1, 1] = -1000
        Uh[c, c] = 100
        bh[c] = -100

    Wh[15, 15, 0, 1] = 0.5
    Wh[15, 15, 2, 1] = 0.5
    Uh[15, 15] = 1

    Wz, bz = p["Wz"], p["bz"]
    Wz[0, 32 + 14] = 1
    Wz[1, 32 + 15] = 1
    for i in range(7):
        Wz[2, 7 + i] = 1e8
        Wz[3, 7 + i] = 1e8
    bz[3] = -1

    # Clip the combined selected/spent indication to one.
    Wz[4, 32 + 15] = 1
    Wz[4, 32 + 19] = 1
    Wz[5, 32 + 15] = 1
    Wz[5, 32 + 19] = 1
    bz[5] = -1

    # First-step flag. It changes sign after the initial invocation.
    p["bm"][0] = 1
    p["Wm"][0, 0] = 5
    p["bm"][1] = 5
    p["Wm"][1, 0] = -10

    p["bo"][:] = -100
    p["bo"][0] = -4.5
    p["Wo"][0, 4] = 4
    p["Wo"][0, 5] = -4
    p["Wo"][0, 128 + 1] = 2

    p["bo"][5] = 0
    p["bo"][6] = -3
    p["Wo"][6, 0] = 2
    p["Wo"][6, 1] = 2
    p["Wo"][6, 2] = 2
    p["Wo"][6, 3] = -2

    # Before selection, matching tiles dominate. Afterwards, rank slots.
    p["Wc"][0, 17] = 1e-9
    p["Wc"][0, 18] = 0.5e-9
    for i in range(7):
        p["Wc"][0, 32 + 7 + i] = 1

    return p

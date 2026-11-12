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

    # Player pixels adjacent to a colored strip.
    for out, horizontal in ((0, True), (1, False)):
        W1[out, 4, 2, 2] = 10.0
        b1[out] = -10.0
        for color in (11, 14):
            for d in (-2, -1, 1, 2):
                if horizontal:
                    W1[out, color, 2, 2 + d] = 1.0
                else:
                    W1[out, color, 2 + d, 2] = 1.0

    for color in (1, 5, 12):
        W1[2, color, 2, 2] = 1.0
    W1[3, 9, 2, 2] = 1.0
    W1[4, 0, 2, 2] = 1.0
    W1[5, 12, 2, 2] = 1.0

    W2, b2 = p["W2"], p["b2"]

    W2[0, 0, 1, 1] = 1.0
    W2[1, 0, 1, 1] = 1.0
    b2[1] = -1.0
    W2[2, 1, 1, 1] = 1.0
    W2[3, 1, 1, 1] = 1.0
    b2[3] = -1.0
    W2[4, 2, 1, 1] = 1.0
    W2[5, 3, 1, 1] = 1.0
    W2[6, 4, 1, 1] = 1.0
    W2[7, 5, 1, 1] = 1.0

    W3 = p["W3"]

    # Axial projections of the player.
    W3[0, 32 + 0] = 1.0
    W3[0, 32 + 1] = -1.0
    W3[1, 64 + 2] = 1.0
    W3[1, 64 + 3] = -1.0
    W3[6, 64 + 0] = 1.0
    W3[6, 64 + 1] = -1.0
    W3[7, 32 + 2] = 1.0
    W3[7, 32 + 3] = -1.0

    W3[2, 4] = 1.0
    W3[3, 5] = 1.0
    W3[4, 32 + 6] = 1.0
    W3[5, 64 + 6] = 1.0
    W3[8, 7] = 1.0

    Uh, bh, Wh = p["Uh"], p["bh"], p["Wh"]

    # Directional waves from the player.
    for h in (0, 1):
        Uh[h, 0] = 1.0
        bh[h] = -0.015
    for h in (2, 3):
        Uh[h, 1] = 1.0
        bh[h] = -0.015

    Wh[0, 0, 0, 1] = 1.0
    Wh[1, 1, 2, 1] = 1.0
    Wh[2, 2, 1, 0] = 1.0
    Wh[3, 3, 1, 2] = 1.0

    # Floor-presence sources and directional wall propagation.
    Uh[4, 4] = 1.0
    Uh[5, 5] = 1.0
    for h in (6, 7, 8, 9):
        Uh[h, 2] = 10.0
        bh[h] = -10.0

    Wh[6, 4, 0, 1] = 1.0
    Wh[6, 6, 0, 1] = 1.0
    Wh[7, 4, 2, 1] = 1.0
    Wh[7, 7, 2, 1] = 1.0
    Wh[8, 5, 1, 0] = 1.0
    Wh[8, 8, 1, 0] = 1.0
    Wh[9, 5, 1, 2] = 1.0
    Wh[9, 9, 1, 2] = 1.0

    # Button scores require a green pixel and a donor-side floor.
    for h in (10, 11, 12, 13):
        Uh[h, 3] = 1000.0
        bh[h] = -1010.2

    # Horizontal-gravity transfers toward the player's lane.
    Wh[10, 6, 0, 1] = 10.0
    Wh[10, 1, 1, 1] = 1.0
    Wh[11, 7, 2, 1] = 10.0
    Wh[11, 0, 1, 1] = 1.0

    # Vertical-gravity transfers: score both sides relative to the player.
    Wh[12, 8, 1, 0] = 10.0
    Wh[12, 2, 1, 1] = 1.0
    Wh[12, 3, 1, 1] = 1.0
    bh[12] = -1009.8

    Wh[13, 9, 1, 2] = 10.0
    Wh[13, 2, 1, 1] = 1.0
    Wh[13, 3, 1, 1] = 1.0
    bh[13] = -1010.4

    # Aligned active switches, biased against immediately crossing back.
    Uh[14, 8] = 100.0
    Uh[14, 6] = 1.0
    bh[14] = -100.5
    Wh[14, 0, 1, 1] = 1.0
    Wh[14, 1, 1, 1] = -2.0

    Uh[15, 8] = 100.0
    Uh[15, 7] = 1.0
    bh[15] = -100.5
    Wh[15, 2, 1, 1] = 1.0
    Wh[15, 3, 1, 1] = -2.0

    p["bo"][6] = 10.0

    p["Wc"][0, 3] = 10.0
    for h in (10, 11, 12, 13):
        p["Wc"][0, 32 + h] = 10.0
    p["Wc"][0, 32 + 14] = 500.0
    p["Wc"][0, 32 + 15] = 500.0

    return p

import numpy as np

SHAPES = {
    "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
    "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
    "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
    "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
}

ROW, COL = 16, 17


def build():
    p = {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}
    W1, b1 = p["W1"], p["b1"]
    C = 2  # centre tap; tap [C+dy, C+dx]

    # ---- vertical-gravity features ----
    W1[0, 9, C, C] = 1.0
    W1[0, COL, C, C] = -1.0                 # is9 - col
    for c, cw in ((1, -1.0), (3, 0.0)):     # hole right-edge detector
        W1[c, 11, C, C + 1] = 1.0
        W1[c, 11, C + 1, C] = 1.0
        W1[c, 11, C - 1, C] = 1.0
        W1[c, 11, C, C] = -2.0
        W1[c, COL, C, C] = cw
        b1[c] = -1.0
    W1[2, 9, C, C] = 1.0                    # is9

    # ---- horizontal-gravity features ----
    W1[4, 9, C, C] = 1.0
    W1[4, ROW, C, C] = -1.0                 # is9 - row
    for c, rw in ((5, -1.0), (6, 0.0)):     # rotated hole detector
        W1[c, 11, C + 1, C] = 1.0
        W1[c, 11, C, C - 1] = 1.0
        W1[c, 11, C, C + 1] = 1.0
        W1[c, 11, C, C] = -2.0
        W1[c, ROW, C, C] = rw
        b1[c] = -1.0

    # ---- gravity orientation: liquid(6) pixel adjacent to source(4) ----
    W1[7, 6, C, C] = 2.0
    W1[7, 4, C - 1, C] = 1.0
    W1[7, 4, C + 1, C] = 1.0
    b1[7] = -2.0                            # vertical source
    W1[8, 6, C, C] = 2.0
    W1[8, 4, C, C - 1] = 1.0
    W1[8, 4, C, C + 1] = 1.0
    b1[8] = -2.0                            # horizontal source

    ident = [0, 1, 2, 4, 5]
    clip = [3, 6, 7, 8]
    W2, b2, W3, b3 = p["W2"], p["b2"], p["W3"], p["b3"]
    for c in ident:
        W2[c, c, 1, 1] = 1.0
        W3[c, c] = 1.0
    for c in clip:                          # min(x,1) = relu(1 - relu(1 - x))
        W2[c, c, 1, 1] = -1.0
        b2[c] = 1.0
        W3[c, c] = -1.0
        b3[c] = 1.0

    F = 32  # offset of f3.max in g
    M0, M1, H9, GV = F + 0, F + 1, F + 2, F + 3
    M4, M5, GH, HV, HH = F + 4, F + 5, F + 6, F + 7, F + 8
    Wz, bz = p["Wz"], p["bz"]
    # z0: move screen-left
    Wz[0, M1] = 63.0; Wz[0, M0] = -63.0
    Wz[0, H9] = 100.0; Wz[0, GV] = 100.0; Wz[0, HV] = 100.0
    bz[0] = -302.5
    # z1: move screen-right
    Wz[1, M0] = 63.0; Wz[1, M1] = -63.0
    Wz[1, H9] = 100.0; Wz[1, GV] = 100.0; Wz[1, HV] = 100.0
    bz[1] = -300.5
    # z2: move screen-up
    Wz[2, M5] = 63.0; Wz[2, M4] = -63.0
    Wz[2, H9] = 100.0; Wz[2, GH] = 100.0; Wz[2, HH] = 100.0
    bz[2] = -302.5
    # z3: move screen-down
    Wz[3, M4] = 63.0; Wz[3, M5] = -63.0
    Wz[3, H9] = 100.0; Wz[3, GH] = 100.0; Wz[3, HH] = 100.0
    bz[3] = -300.5

    Wo, bo = p["Wo"], p["bo"]
    bo[:] = -10.0
    Wo[1, 2] = 10.0; bo[1] = 0.0   # ACTION1 up
    Wo[2, 3] = 10.0; bo[2] = 0.0   # ACTION2 down
    Wo[3, 0] = 10.0; bo[3] = 0.0   # ACTION3 left
    Wo[4, 1] = 10.0; bo[4] = 0.0   # ACTION4 right
    bo[5] = 1.0                    # ACTION5 pour when aligned
    return p

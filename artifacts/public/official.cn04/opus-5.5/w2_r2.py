import numpy as np


def build():
    f = np.float32
    p = {
        "W1": np.zeros((32, 19, 5, 5), f), "b1": np.zeros(32, f),
        "W2": np.zeros((32, 32, 3, 3), f), "b2": np.zeros(32, f),
        "W3": np.zeros((32, 96), f), "b3": np.zeros(32, f),
        "Wh": np.zeros((16, 16, 3, 3), f), "Uh": np.zeros((16, 32), f), "bh": np.zeros(16, f),
        "Wz": np.zeros((128, 96), f), "bz": np.zeros(128, f),
        "Wm": np.zeros((64, 64), f), "Wmz": np.zeros((64, 128), f),
        "Wma": np.zeros((64, 8), f), "bm": np.zeros(64, f),
        "Wo": np.zeros((8, 192), f), "bo": np.zeros(8, f),
        "Wc": np.zeros((1, 48), f), "bc": np.zeros(1, f),
    }
    W1, b1, W2, b2, W3, b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]

    # ---- layer 1 ----
    colours = [c for c in range(16) if c not in (3, 8)]
    cross = [(2, 2), (0, 2), (1, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 3), (2, 4)]
    for i, c in enumerate(colours):
        for (u, v) in cross:
            W1[2 * i, c, u, v] = 1.0
            W1[2 * i + 1, c, u, v] = 1.0
        b1[2 * i + 1] = -1.0
    W1[28, 8, 1:4, 1:4] = 1.0
    b1[28] = -8.0
    W1[29, 16, 2, 2] = 1.0
    W1[30, 17, 2, 2] = 1.0
    W1[31, 0, 2, 2] = 1.0
    W1[31, 18, 0, 2] = 1.0
    W1[31, 18, 1, 2] = 1.0
    b1[31] = -2.0

    # ---- layer 2 ----
    for ch, bias in ((0, -16.0), (1, -17.0)):
        for i, c in enumerate(colours):
            w = 9.0 if c == 0 else -1.0
            W2[ch, 2 * i, 1, 1] = w
            W2[ch, 2 * i + 1, 1, 1] = -w
        W2[ch, 28, 1, 1] = 10.0
        b2[ch] = bias
    W2[2, 28, 1, 1] = 1.0
    W2[3, 28, 1, 1] = 1.0
    W2[3, 0, 1, 1] = -1.0
    W2[3, 1, 1, 1] = 1.0
    W2[4, 29, 1, 1] = 1.0
    W2[5, 30, 1, 1] = 1.0
    W2[6, 31, 1, 1] = 1.0

    # ---- layer 3 ----
    mS = {0: 1.0, 1: -1.0}
    mO = {3: 1.0}

    def ext(ch, mask, coord, sign):
        for k, w in mask.items():
            W3[ch, k] += 2.0 * w
        W3[ch, coord] += sign
        b3[ch] = -2.0 if sign > 0 else -1.0

    ext(0, mS, 4, 1.0); ext(1, mS, 4, -1.0); ext(2, mS, 5, 1.0); ext(3, mS, 5, -1.0)
    for k, w in mS.items():
        W3[4, k] += w
    ext(5, mO, 4, 1.0); ext(6, mO, 4, -1.0); ext(7, mO, 5, 1.0); ext(8, mO, 5, -1.0)
    W3[9, 3] = 1.0
    W3[10, 6] = 1.0

    # ---- decision ----
    Wz, bz = p["Wz"], p["bz"]
    G = lambda k: 32 + k
    A, B, C, D, pS = G(0), G(1), G(2), G(3), G(4)
    OA, OB, OC, OD, pO, pZ = G(5), G(6), G(7), G(8), G(9), G(10)
    meanZ = 64 + 10
    for row, sgn, (a1, a2, s1, s2) in ((0, 1.0, (OA, OB, A, B)), (1, -1.0, (OA, OB, A, B)),
                                       (2, 1.0, (OC, OD, C, D)), (3, -1.0, (OC, OD, C, D))):
        Wz[row, a1] += 63.0 * sgn
        Wz[row, a2] += -63.0 * sgn
        Wz[row, s1] += -63.0 * sgn
        Wz[row, s2] += 63.0 * sgn
        Wz[row, pS] += 200.0
        Wz[row, pO] += 200.0
        Wz[row, pZ] += 200.0
        bz[row] = -603.0
    # click: select another sprite (only with a real black body present)
    Wz[4, pO] = 5.0; Wz[4, pS] = -10.0; Wz[4, meanZ] = 100.0; bz[4] = -5.5
    # covered-target detector
    Wz[6, pS] = 1.0; Wz[6, pZ] = 1.0; Wz[6, pO] = -2.0; bz[6] = -1.0

    # memory unit 0: previous action was ACTION5
    p["Wma"][0, 5] = 5.0
    p["bm"][0] = -2.5

    Wo, bo = p["Wo"], p["bo"]
    Wo[2, 0] = 10.0     # down
    Wo[1, 1] = 10.0     # up
    Wo[4, 2] = 10.0     # right
    Wo[3, 3] = 10.0     # left
    Wo[6, 4] = 10000.0  # click other connector
    Wo[3, 6] = 2.0      # covered -> alternate with left moves
    Wo[3, 128 + 0] = 2.0
    bo[3] = -2.0
    bo[0] = -5.0
    bo[5] = 0.1
    bo[7] = -5.0

    p["Wc"][0, 9] = 1.0
    return p

import numpy as np


def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)), "Wma": np.zeros((64, 8)),
        "bm": np.zeros(64), "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    W1, b1 = p["W1"], p["b1"]

    def s(o, col, w, r=1):
        for ky in range(2 - r, 3 + r):
            for kx in range(2 - r, 3 + r):
                W1[o, col, ky, kx] += w

    def cen(o, col, w):
        W1[o, col, 2, 2] += w

    def orth(o, col, w):
        for ky, kx in ((1, 2), (3, 2), (2, 1), (2, 3)):
            W1[o, col, ky, kx] += w

    # 0 srcA: free box ring
    cen(0, 4, 10); cen(0, 3, 10); s(0, 9, 1); s(0, 2, -10); b1[0] = -10
    # 2 target border
    cen(2, 9, 10); s(2, 2, 1); s(2, 4, -10); s(2, 3, -10); s(2, 0, -10); b1[2] = -10
    # 3-5 walls
    cen(3, 5, 1); cen(4, 12, 1); cen(5, 15, 1)
    # 6 G faced free box
    cen(6, 3, 10); s(6, 2, -10); b1[6] = -9
    # 7 C carry flag
    cen(7, 0, 10); s(7, 9, 1); s(7, 14, -10); b1[7] = -10
    # 8 R: carried box ring touching target interior
    cen(8, 0, 10); orth(8, 2, 5); s(8, 9, 1); s(8, 14, -100); b1[8] = -15
    # masks
    dirs = [((3, 2), (4, 2)), ((1, 2), (0, 2)), ((2, 3), (2, 4)), ((2, 1), (2, 0))]
    for d, (o1, o2) in enumerate(dirs):
        o = 9 + d
        W1[o, 14, o1[0], o1[1]] += 1
        W1[o, 0, o1[0], o1[1]] += 1
        W1[o, 14, o2[0], o2[1]] += 1
        W1[o, 14, 2, 2] += -2
        W1[o, 0, 2, 2] += -2
        b1[o] = -1
        o = 13 + d
        W1[o, 0, o1[0], o1[1]] += 1
        W1[o, 9, o2[0], o2[1]] += 1
        W1[o, 9, 2, 2] += -1
        W1[o, 14, 2, 2] += -1
        W1[o, 0, 2, 2] += -1
        b1[o] = -1
    # 17 blocking tiles
    cen(17, 2, 10); s(17, 1, 1); b1[17] = -10
    # 18 any uncarried box ring
    cen(18, 4, 1); cen(18, 3, 1)
    # 19 target interior
    cen(19, 2, 10); s(19, 1, -10); b1[19] = -9
    # 24 uncarried box ring touching target interior
    cen(24, 4, 10); cen(24, 3, 10); orth(24, 2, 1); s(24, 1, -10); b1[24] = -10

    W2 = p["W2"]
    for i in range(25):
        if i != 20:
            W2[i, i, 1, 1] = 1.0
    for ky in range(3):
        for kx in range(3):
            W2[20, 24, ky, kx] = 1.0
            W2[0, 24, ky, kx] = -10.0
            W2[6, 24, ky, kx] = -10.0
    for i in range(25):
        p["W3"][i, i] = 1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    c = 0.24
    S = 0.1
    for f in (0, 1):
        Wh[f, f, 0, 1] = c; Wh[f, f, 2, 1] = c
        Wh[f, f, 1, 0] = c; Wh[f, f, 1, 2] = c
    Uh[0, 0] = S
    for ch in (3, 4, 5, 17, 20):
        Uh[0, ch] = -10.0
    Uh[1, 2] = S; Uh[1, 19] = S
    for ch in (3, 4, 5, 17, 18):
        Uh[1, ch] = -10.0
    for d in range(4):
        Wh[2 + d, 0, 1, 1] = 1.0
        Uh[2 + d, 9 + d] = 10.0
        bh[2 + d] = -10.0
        Wh[6 + d, 1, 1, 1] = 1.0
        Uh[6 + d, 13 + d] = 10.0
        bh[6 + d] = -10.0

    Wz, Wo, bo = p["Wz"], p["Wo"], p["bo"]
    F = 32
    for d in range(4):
        Wz[d, 2 + d] = 1.0
        Wz[d, F + 7] = -100.0
        Wz[4 + d, 6 + d] = 1.0
        Wo[1 + d, d] = 1.0
        Wo[1 + d, 4 + d] = 1.0
    Wz[8, F + 6] = 1.0
    Wz[8, F + 7] = -100.0
    Wz[9, F + 8] = 1.0
    Wo[5, 8] = 10.0
    Wo[5, 9] = 10.0
    bo[0] = -1.0
    bo[6] = -1.0
    bo[7] = -1.0

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

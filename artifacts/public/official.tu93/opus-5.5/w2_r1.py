import numpy as np

def build():
    W1 = np.zeros((32, 19, 5, 5), np.float32); b1 = np.zeros(32, np.float32)
    W2 = np.zeros((32, 32, 3, 3), np.float32); b2 = np.zeros(32, np.float32)
    W3 = np.zeros((32, 96), np.float32); b3 = np.zeros(32, np.float32)
    Wh = np.zeros((16, 16, 3, 3), np.float32)
    Uh = np.zeros((16, 32), np.float32); bh = np.zeros(16, np.float32)
    Wz = np.zeros((128, 96), np.float32); bz = np.zeros(128, np.float32)
    Wm = np.zeros((64, 64), np.float32); Wmz = np.zeros((64, 128), np.float32)
    Wma = np.zeros((64, 8), np.float32); bm = np.zeros(64, np.float32)
    Wo = np.zeros((8, 192), np.float32); bo = np.zeros(8, np.float32)
    Wc = np.zeros((1, 48), np.float32); bc = np.zeros(1, np.float32)

    def s1(c, col, dy, dx, v):
        W1[c, col, dy + 2, dx + 2] += v

    for col in [0, 2, 4, 8, 9, 11, 12, 13, 14, 15]:
        s1(0, col, 0, 0, 1.0)          # track
    s1(1, 14, 0, 0, 1.0)               # exit
    for col in [8, 11, 12, 13, 15]:
        s1(14, col, 0, 0, 1.0)         # enemy body

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    body = [8, 12, 13]
    for i, (dy, dx) in enumerate(dirs):
        py, px = (0, 1) if dx == 0 else (1, 0)
        for c, extra in ((2 + i, False), (10 + i, True)):
            for P in (9, 4):
                s1(c, P, -2 * dy, -2 * dx, 1.0)
                s1(c, P, -2 * dy + py, -2 * dx + px, 1.0)
                s1(c, P, -2 * dy - py, -2 * dx - px, 1.0)
                s1(c, P, -dy, -dx, -3.0)
            if extra:
                s1(c, 2, 0, 0, 1.0)
                b1[c] = -3.0
            else:
                b1[c] = -2.0
        c = 6 + i
        s1(c, 15, 0, 0, 1.0)
        s1(c, 11, 0, 0, 1.0)
        for col in body:
            s1(c, col, -dy, -dx, 1.0)
            s1(c, col, -2 * dy, -2 * dx, 1.0)
            s1(c, col, py, px, 1.0)
            s1(c, col, -py, -px, 1.0)
        b1[c] = -4.0
        W2[c, c, 1, 1] = 1.0
        W2[c, c, 1 + py, 1 + px] = 1.0
        W2[c, c, 1 - py, 1 - px] = 1.0

    for c in range(15):
        if not (6 <= c <= 9):
            W2[c, c, 1, 1] = 1.0
        W3[c, c] = 1.0

    BIG = 1000.0
    nb = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    # Normal reach R (h0,h1)
    for out, off in ((0, 0.0), (1, 1.0)):
        Uh[out, 1] = 6.0; Uh[out, 0] = BIG; Uh[out, 14] = 5.0
        bh[out] = -BIG - 5.0 - off
        for (oy, ox) in nb:
            Wh[out, 0, oy + 1, ox + 1] = 6.0
            Wh[out, 1, oy + 1, ox + 1] = -6.0
            Wh[out, 13, oy + 1, ox + 1] = 1.0
            Wh[out, 14, oy + 1, ox + 1] = -1.0
        Wh[out, 11, 1, 1] = -BIG
        Wh[out, 12, 1, 1] = BIG
    # Danger reach Q (h13,h14)
    for out, off in ((13, 0.0), (14, 1.0)):
        Uh[out, 0] = BIG
        bh[out] = -2 * BIG - off
        for (oy, ox) in nb:
            Wh[out, 0, oy + 1, ox + 1] = 1.0
            Wh[out, 1, oy + 1, ox + 1] = -1.0
            Wh[out, 13, oy + 1, ox + 1] = 1.0
            Wh[out, 14, oy + 1, ox + 1] = -1.0
        Wh[out, 11, 1, 1] = BIG
        Wh[out, 12, 1, 1] = -BIG
    # Zind pair (h11,h12)
    bh[12] = -1.0
    for i in range(4):
        Wh[11, 7 + i, 1, 1] = 1.0
        Wh[12, 7 + i, 1, 1] = 1.0
    # Distance counter
    Wh[2, 2, 1, 1] = 1.0
    Wh[2, 0, 1, 1] = 1.0; Wh[2, 1, 1, 1] = -1.0
    Wh[2, 13, 1, 1] = 1.0; Wh[2, 14, 1, 1] = -1.0
    for i, (dy, dx) in enumerate(dirs):
        Uh[3 + i, 2 + i] = BIG; bh[3 + i] = -BIG
        Wh[3 + i, 2, 1, 1] = 1.0
        Uh[7 + i, 6 + i] = 8.0; bh[7 + i] = -1.0
        Wh[7 + i, 7 + i, 1 - dy, 1 - dx] = 1.0

    for i in range(4):
        Wz[i, 3 + i] = 1.0
        Wz[4 + i, 32 + 10 + i] = 1.0
        Wo[1 + i, i] = 1.0
        Wo[1 + i, 4 + i] = 0.5
    bo[0] = -5.0
    bo[5:] = -5.0

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3,
            "Wh": Wh, "Uh": Uh, "bh": bh, "Wz": Wz, "bz": bz,
            "Wm": Wm, "Wmz": Wmz, "Wma": Wma, "bm": bm,
            "Wo": Wo, "bo": bo, "Wc": Wc, "bc": bc}

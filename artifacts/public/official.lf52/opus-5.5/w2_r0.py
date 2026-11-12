import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    G, GR, P = 14, 2, 15
    W1, b1, W2, b2, W3, b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]
    W1[0, G, 2, 2] = 1.0            # green centre
    W1[1, GR, 2, 2] = 1.0           # gray centre
    W1[6, GR, 3, 2] = 1.0           # gray one row below
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for d, (dr, dc) in enumerate(dirs):
        W1[2 + d, G, 2, 2] = 1.0
        W1[2 + d, G, 2 + dr, 2 + dc] = -1.0
        W1[2 + d, G, 2 + 2 * dr, 2 + 2 * dc] = -1.0
        W1[7 + d, G, 2 + 2 * dr, 2 + 2 * dc] = 1.0
        W1[7 + d, P, 2 + 2 * dr, 2 + 2 * dc] = 1.0
        W1[11 + d, G, 2, 2] = 1.0
        W1[11 + d, G, 2 + dr, 2 + dc] = -1.0
        W1[11 + d, P, 2 + 2 * dr, 2 + 2 * dc] = 1.0
        b1[11 + d] = -1.0
        W2[d, 2 + d, 1, 1] = 1.0
        W2[d, 7 + d, 1 + dr, 1 + dc] = 1.0
        b2[d] = -1.0
        W2[6 + d, 11 + d, 1, 1] = 1.0
    # progress-bar boundary detectors at row 0
    pairs = [(1, 0), (2, 1), (3, 2), (4, 3), (10, 4)]
    for k, (L, R) in enumerate(pairs):
        c = 16 + k
        W1[c, L, 2, 2] = 1.0
        W1[c, R, 2, 3] = 1.0
        W1[c, 16, 2, 2] = -126.0
        b1[c] = -1.0
        for j in range(3):
            W2[11, c, 1, j] = 1.0   # 3-column-wide cursor
    W2[4, 1, 1, 1] = 1.0
    W2[4, 6, 1, 1] = 1.0
    b2[4] = -1.0                    # gray vertical pair (landing marker)
    W2[5, 0, 1, 1] = 1.0            # green copy
    for i in range(10):
        W3[i, i] = 1.0
    # cursor bonus: green AND colmax(cursor stripe)
    W3[10, 5] = 1.0
    W3[10, 64 + 11] = 1.0
    b3[10] = -1.0
    Wc = p["Wc"]
    Wc[0, 0:4] = 2.0
    Wc[0, 6:10] = 2.0
    Wc[0, 4] = 50.0
    Wc[0, 5] = 1.0
    Wc[0, 10] = 20.0
    p["bo"][6] = 1.0
    return p

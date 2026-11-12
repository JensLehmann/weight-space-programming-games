import numpy as np


def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32),
        "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32),
        "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32),
        "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32),
        "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32),
        "Wz": np.zeros((128, 96), np.float32),
        "bz": np.zeros(128, np.float32),
        "Wm": np.zeros((64, 64), np.float32),
        "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32),
        "bm": np.zeros(64, np.float32),
        "Wo": np.zeros((8, 192), np.float32),
        "bo": np.zeros(8, np.float32),
        "Wc": np.zeros((1, 48), np.float32),
        "bc": np.zeros(1, np.float32),
    }
    W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
    W1[0, 4, 2, 2] = 1.0; W1[0, 18, 1, 2] = 1.0; b1[0] = -1.0
    W1[1, 9, 2, 2] = 1.0
    W1[2, 0, 2, 2] = 1.0
    W1[3, 5, 2, 2] = 1.0
    W1[4, 4, :, :] = 1.0
    W1[5, 16, 2, 2] = -63.0; b1[5] = 4.0
    W1[6, 5, 2, 2] = 1.0; W1[6, 9, 2, 2] = 1.0
    W1[7, 12, 2, 2] = 1.0                                        # is12 active pipe
    for c in range(4):
        W2[c, c, 1, 1] = 1.0
    W2[4, 4, :, :] = 1.0
    W2[4, 2, 1, 1] = 300.0
    W2[4, 5, 1, 1] = -300.0
    b2[4] = -300.0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, (dr, dc) in enumerate(dirs):
        W2[5 + di, 3, 1 + dr, 1 + dc] = 1.0
        W2[9 + di, 6, 1 + dr, 1 + dc] = 1.0
    W2[13, 7, 1, 1] = 1.0
    for c in range(14):
        p["W3"][c, c] = 1.0
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Bf = 1e7
    Wh[0, 0, :, :] = 0.999 / 9.0
    Uh[0, 4] = 1.0; Uh[0, 2] = Bf; Uh[0, 1] = Bf; bh[0] = -Bf
    Wh[1, 0, 1, 1] = -1e30; bh[1] = 1.0
    Wh[13, 1, 1, 1] = 1.0; Uh[13, 2] = 1.0; bh[13] = -1.0
    Bg = 1e4
    for di, (dr, dc) in enumerate(dirs):
        t = 2 + di
        Wh[t, t, 1 + dr, 1 + dc] = 1.0
        Wh[t, 13, 1 + dr, 1 + dc] = 1.0 + Bg
        Uh[t, 9 + di] = Bg
        bh[t] = -Bg
        g = 6 + di
        Wh[g, t, 1, 1] = 1.0
        Uh[g, 1] = Bg
        Uh[g, 5 + di] = Bg
        bh[g] = -2 * Bg
        Wh[12, g, 1, 1] = -1e3
    bh[12] = 1.0
    Wh[10, 10, :, :] = 0.998 / 9.0; Uh[10, 0] = 1.0
    # h14: active pipe adjacent to piece container
    Bq = 1e5
    Uh[14, 13] = 4 * Bq
    for (dr, dc) in dirs:
        Wh[14, 1, 1 + dr, 1 + dc] = -Bq
    bh[14] = -4 * Bq + 3.5 * Bq
    C, Ca, Bb = 2000.0, 1000.0, 1e4
    Wh[11, 10, 1, 1] = 1.0
    Wh[11, 12, 1, 1] = -C
    Wh[11, 1, 1, 1] = -Ca
    Wh[11, 14, 1, 1] = 1.0
    Uh[11, 1] = Bb
    bh[11] = C + Ca - Bb
    p["Wc"][0, 32 + 11] = 1.0
    p["bo"][6] = 10.0
    return p

import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32), "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32), "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32), "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32), "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32),
        "Wz": np.zeros((128, 96), np.float32), "bz": np.zeros(128, np.float32),
        "Wm": np.zeros((64, 64), np.float32), "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32), "bm": np.zeros(64, np.float32),
        "Wo": np.zeros((8, 192), np.float32), "bo": np.zeros(8, np.float32),
        "Wc": np.zeros((1, 48), np.float32), "bc": np.zeros(1, np.float32),
    }
    W1, b1, W2, b2, W3, b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    # ---------------- layer 1 ----------------
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        W1[0, 11, dy + 2, dx + 2] = 1.0
    for dy, dx in [(2, 2), (2, -1), (-1, 2), (-1, -1)]:
        W1[0, 4, dy + 2, dx + 2] = 1.0
    b1[0] = -7.0
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            W1[1, 11, dy + 2, dx + 2] = 1.0
    b1[1] = -15.0
    W1[2, 11, 2, 2] = 1.0
    W1[3, 8, 2, 2] = 1.0
    W1[4, 14, 2, 2] = 1.0
    W1[4, 14, 3, 2] = 1.0
    W1[4, 14, 2, 3] = 1.0
    b1[4] = -2.0
    W1[5, 11, :, :] = 1.0
    b1[5] = -9.0
    W1[6, 17, 2, 2] = 63.0; b1[6] = -31.0
    W1[7, 17, 2, 2] = 63.0; b1[7] = -32.0
    W1[8, 16, 2, 2] = 63.0; b1[8] = -31.0
    W1[9, 16, 2, 2] = 63.0; b1[9] = -32.0
    # ch10: unfilled timer-bar pixel (colour 14 in column 0)
    W1[10, 14, 2, 2] = 1.0
    W1[10, 17, 2, 2] = -63.0

    # ---------------- layer 2 ----------------
    W2[0, 2, 1, 1] = 1.0
    W2[0, 0, :, :] = -2.0
    W2[0, 5, :, :] = -1.0
    W2[2, 3, 1, 1] = 1.0
    W2[3, 4, 1, 1] = 1.0
    quad = {4: (1, -1, -1.0), 5: (-1, -1, 0.0), 6: (1, 1, -2.0), 7: (-1, 1, -1.0)}
    for c, (sr, sb, bias) in quad.items():
        W2[c, 0, 1, 1] = 1.0
        W2[c, 1, 1, 1] = 1.0
        W2[c, 6, 1, 1] = sr
        W2[c, 7, 1, 1] = -sr
        W2[c, 8, 1, 1] = sb
        W2[c, 9, 1, 1] = -sb
        b2[c] = bias
    W2[8, 10, 1, 1] = 1.0  # timer-unfilled pixel

    # ---------------- layer 3 ----------------
    for k in [0, 2, 3, 4, 5, 6, 7]:
        W3[k, k] = 1.0
    # f3[8]: button pixel lying in a row the timer has not reached yet
    W3[8, 32 + 8] = 1.0
    W3[8, 2] = 1.0
    W3[8, 3] = 1.0
    b3[8] = -1.0

    # ---------------- recurrent block ----------------
    flood_offsets = {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (0, -1), (0, 1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)],
        4: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0)],
        6: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0)],
    }
    for a, offs in flood_offsets.items():
        b = a + 1
        for dy, dx in offs:
            for o in (a, b):
                Wh[o, a, dy + 1, dx + 1] = 1.0
                Wh[o, b, dy + 1, dx + 1] = -1.0
        Uh[a, 0] = 1.0
        Uh[b, 0] = 1.0
        bh[b] = -1.0

    Wh[14, 14, 1, 1] = 1.0
    bh[14] = 1.0
    Wh[15, 14, 1, 1] = -1.0
    bh[15] = 66.0

    seeds = {8: (4, 1, 1), 9: (5, -1, 1), 10: (6, 1, -1), 11: (7, -1, -1)}
    for s, (gq, sA, sC) in seeds.items():
        Wh[s, 0, 1, 1] = sA
        Wh[s, 1, 1, 1] = -sA
        Wh[s, 2, 1, 1] = -sA
        Wh[s, 3, 1, 1] = sA
        Wh[s, 4, 1, 1] = sC
        Wh[s, 5, 1, 1] = -sC
        Wh[s, 6, 1, 1] = -sC
        Wh[s, 7, 1, 1] = sC
        Wh[s, 15, 1, 1] = -3.0
        Uh[s, gq] = 4.0
        bh[s] = -4.0

    for o in (12, 13):
        Wh[o, 12, :, :] = 1.0
        Wh[o, 13, :, :] = -1.0
        for s in (8, 9, 10, 11):
            Wh[o, s, 1, 1] = 1.0
        Uh[o, 2] = -10.0
    bh[13] = -1.0

    # ---------------- click map ----------------
    p["Wc"][0, 2] = 1.0          # left button
    p["Wc"][0, 3] = 0.5          # right button base
    p["Wc"][0, 8] = 2.0          # timer sweep bonus on not-yet-passed button rows
    p["Wc"][0, 32 + 12] = 1.0    # flag
    p["Wc"][0, 32 + 13] = -1.0

    # ---------------- action: always click ----------------
    p["bo"][6] = 1.0
    return p

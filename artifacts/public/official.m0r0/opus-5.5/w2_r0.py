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
    W1, b1, W3, b3 = p["W1"], p["b1"], p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    # ---- f1 ----
    W1[0, 10, 2, 2] = 1.0
    W1[1, 5, 2, 2] = 1.0; W1[1, 10, 2, 2] = 1.0
    for k, (dr, dc) in enumerate(dirs):
        W1[2 + k, 10, 2 + dr, 2 + dc] = 1.0
        W1[6 + k, 5, 2 + dr, 2 + dc] = 1.0
        W1[6 + k, 10, 2 + dr, 2 + dc] = 1.0
    W1[10, 10, 2, 3] = 1.0; W1[10, 10, 2, 2] = -1.0
    W1[11, 17, 2, 2] = 63.0; W1[11, 10, 2, 2] = 100.0; b1[11] = -99.0
    W1[12, 17, 2, 2] = -63.0; W1[12, 10, 2, 2] = 100.0; b1[12] = -36.0
    for c in range(13):
        p["W2"][c, c, 1, 1] = 1.0
    # ---- f3 ---- 0 Free,1 Bs,2 P,3-6 edge,7,8 coords
    W3[0, 1] = 1.0
    W3[1, 64 + 10] = 1.0
    W3[2, 0] = 1.0
    for k in range(4):
        W3[3 + k, 0] = 1.0; W3[3 + k, 6 + k] = 1.0; W3[3 + k, 2 + k] = -1.0; b3[3 + k] = -1.0
    W3[7, 11] = 1.0; W3[8, 12] = 1.0
    # ---- recurrence ----
    Wh[0, 0, 1, 2] = 1.0; Uh[0, 1] = 1.0               # r: count of Bs columns to the right
    Wh[1, 0, 1, 1] = 1.0; Uh[1, 2] = 1.0; bh[1] = -1.0  # SL
    cross = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
    for (u, v, c) in ((2, 3, 4), (5, 6, 7)):
        for ch in (u, v):
            for (i, j) in cross:
                Wh[ch, u, i, j] = 1.0; Wh[ch, v, i, j] = -1.0
            if u == 2:
                Wh[ch, 1, 1, 1] = 2.0
            else:
                Uh[ch, 2] = 2.0; Wh[ch, 1, 1, 1] = -2.0
            Uh[ch, 0] = 50.0; bh[ch] = -50.0
        bh[v] -= 1.0
        Wh[c, c, 1, 1] = 1.0; Wh[c, u, 1, 1] = 1.0; Wh[c, v, 1, 1] = -1.0
    KG = 1000.0
    cA, cB = 4, 7
    # (channel, label 'A'/'B', dir index)
    gains = [(8, 'A', 0), (9, 'B', 0), (10, 'A', 1), (11, 'B', 1),
             (12, 'A', 3), (13, 'B', 2), (14, 'A', 2), (15, 'B', 3)]
    for ch, lab, k in gains:
        cnt = cB if lab == 'A' else cA
        d = dirs[k]
        Wh[ch, cnt, 1 + d[0], 1 + d[1]] += 1.0
        Wh[ch, cnt, 1, 1] -= 1.0
        Uh[ch, 3 + k] += KG
        if lab == 'A':
            Wh[ch, 1, 1, 1] += KG
        else:
            Uh[ch, 2] += KG; Wh[ch, 1, 1, 1] -= KG
        bh[ch] = 1.0 - 2 * KG
    # ---- z ----
    Wz = p["Wz"]
    Wz[0, 8] = Wz[0, 9] = 1.0       # up
    Wz[1, 10] = Wz[1, 11] = 1.0     # down
    Wz[2, 12] = Wz[2, 13] = 1.0     # converge
    Wz[3, 14] = Wz[3, 15] = 1.0     # diverge
    Wz[4, 32 + 7] = 1.0; Wz[5, 32 + 8] = 1.0
    # ---- memory ----
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    S = 2000.0; G = 100.0
    Wmz[0, 4] = 1.0 / S; Wmz[0, 5] = 1.0 / S
    for u, sg, a in ((28, 1.0, 4), (29, -1.0, 4), (30, 1.0, 3), (31, -1.0, 3)):
        Wmz[u, 4] = 2.0 * sg; Wmz[u, 5] = 2.0 * sg
        Wm[u, 0] = -2.0 * S * sg
        Wma[u, a] = G
        bm[u] = -G - 4.0
    # latch s: +1 => ACTION4 converges
    Wm[34, 34] = 4.0; bm[34] = 1.0
    Wm[34, 28] = -5.0; Wm[34, 31] = -5.0
    Wm[34, 29] = 5.0; Wm[34, 30] = 5.0
    # gated horizontal values
    for u, zi, sg in ((50, 2, 1.0), (51, 3, -1.0), (52, 3, 1.0), (53, 2, -1.0)):
        Wmz[u, zi] = 0.1
        Wm[u, 34] = 20.0 * sg
        bm[u] = -20.0
    for j in range(1, 5):
        Wma[43 + j, j] = 3.0; bm[43 + j] = -1.5
    # ---- logits ----
    Wo, bo = p["Wo"], p["bo"]
    M = 128
    Wo[1, 0] = 1.0
    Wo[2, 1] = 1.0
    Wo[4, M + 50] = 10.0; Wo[4, M + 51] = 10.0; bo[4] = 10.0
    Wo[3, M + 52] = 10.0; Wo[3, M + 53] = 10.0; bo[3] = 10.0
    rev = {1: 2, 2: 1, 3: 4, 4: 3}
    for j in range(1, 5):
        Wo[j, M + 43 + j] += 0.1
        Wo[rev[j], M + 43 + j] += -0.2
    bo[0] = bo[5] = bo[6] = bo[7] = -100.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

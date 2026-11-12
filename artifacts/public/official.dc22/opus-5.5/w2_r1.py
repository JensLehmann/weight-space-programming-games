import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)), "Wma": np.zeros((64, 8)),
        "bm": np.zeros(64),
        "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
    W1[0, 14, 2, 2] = 1.0
    W1[1, 11, 2, 2] = 1.0
    W1[2, 17, 2, 2] = 126.0; b1[2] = -79.0
    W1[3, 17, 2, 2] = 1.0
    W1[4, 16, 2, 2] = 1.0
    for ch, col in ((5, 9), (6, 15)):
        W1[ch, col, 2, 2] = 1; W1[ch, col, 3, 3] = 1
        W1[ch, col, 2, 3] = -1; W1[ch, col, 3, 2] = -1; b1[ch] = -1
    W1[7, 9, 2, 2] = 1.0
    W1[8, 15, 2, 2] = 1.0
    W1[10, 17, 2, 2] = 126.0; b1[10] = -80.0
    W2[0, 0, 1, 1] = 1; W2[0, 3, 1, 1] = 1; b2[0] = -1
    W2[1, 0, 1, 1] = 1; W2[1, 4, 1, 1] = 1; b2[1] = -1
    W2[2, 1, 1, 1] = 1; W2[2, 2, 1, 1] = -1; W2[2, 3, 1, 1] = 1; b2[2] = -1
    W2[3, 1, 1, 1] = 1; W2[3, 2, 1, 1] = -1; W2[3, 4, 1, 1] = 1; b2[3] = -1
    W2[4, 0, 1, 1] = 1
    W2[5, 5, 1, 1] = 1
    W2[6, 6, 1, 1] = 1
    for ch, src in ((7, 7), (8, 8)):
        W2[ch, src, 1, 1] = 1; W2[ch, 2, 1, 1] = 1; W2[ch, 10, 1, 1] = -1; b2[ch] = -1
    for c in range(9):
        p["W3"][c, c] = 1.0
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    Wh[0, 0, :, :] = 0.08; Uh[0, 4] = 1.0
    for g, t, u, chk, btn in ((1, 2, 3, 5, 7), (4, 5, 6, 6, 8)):
        Wh[g, 0, 1, 1] = 1.0; Uh[g, chk] = 1000.0; bh[g] = -1000.0
        Wh[t, t, :, :] = 0.11; Wh[t, g, 1, 1] = 1.0
        Wh[u, t, 1, 1] = 1.0; Uh[u, btn] = 1e4; bh[u] = -1e4
    p["Wc"][0, 32 + 3] = 1.0; p["Wc"][0, 32 + 6] = 1.0
    K = 4096.0 * 63.0 / 4.0
    Wz, bz = p["Wz"], p["bz"]
    PX, PY, GX, GY = 64, 65, 66, 67
    Wz[0, GX] = K;  Wz[0, PX] = -K
    Wz[1, PX] = K;  Wz[1, GX] = -K
    Wz[2, GY] = K;  Wz[2, PY] = -K
    Wz[3, PY] = K;  Wz[3, GY] = -K
    # z4: bridge adjacency (h max of gate channels 1 and 4)
    Wz[4, 1] = 100.0; Wz[4, 4] = 100.0; bz[4] = -0.5
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    for i in range(4):
        Wmz[i, i] = 0.001
    for u, acts in ((4, (3, 4)), (5, (1, 2))):
        for i in range(4):
            Wmz[u, i] = 5.0
            Wm[u, i] = -5000.0
        for a in acts:
            Wma[u, a] = 1000.0
        bm[u] = 5.0 - 1000.0
    Wma[6, 1] = 6.0; Wma[6, 2] = -6.0; Wm[6, 6] = 3.0
    Wma[7, 4] = 6.0; Wma[7, 3] = -6.0; Wm[7, 7] = 3.0
    Wma[8, 6] = 4.0; Wm[8, 8] = 1.5; bm[8] = -0.6
    Wo, bo = p["Wo"], p["bo"]
    M = 128
    Wo[1, 3] = 1.0; Wo[2, 2] = 1.0; Wo[3, 1] = 1.0; Wo[4, 0] = 1.0
    Wo[3, M + 4] = -500.0; Wo[4, M + 4] = -500.0; bo[3] -= 500.0; bo[4] -= 500.0
    Wo[1, M + 5] = -500.0; Wo[2, M + 5] = -500.0; bo[1] -= 500.0; bo[2] -= 500.0
    Wo[1, M + 6] = 6.0; Wo[2, M + 6] = -6.0
    Wo[4, M + 7] = 6.0; Wo[3, M + 7] = -6.0
    Wo[6, M + 4] = 250.0; Wo[6, M + 5] = 250.0
    Wo[6, 4] = 400.0
    Wo[6, M + 8] = -1000.0
    bo[6] = -400.0 - 1000.0 * 0.97
    for a in (0, 5, 7):
        bo[a] = -1e4
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

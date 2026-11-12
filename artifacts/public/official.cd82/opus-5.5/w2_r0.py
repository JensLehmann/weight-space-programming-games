import numpy as np

def build():
    P = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)), "Wma": np.zeros((64, 8)),
        "bm": np.zeros(64), "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    W1, b1, W2, b2, W3 = P["W1"], P["b1"], P["W2"], P["b2"], P["W3"]

    def lin(u, a, b, c):  # f1 unit relu(a*row + b*col + c)
        W1[u, 16, 2, 2] = 63.0 * a
        W1[u, 17, 2, 2] = 63.0 * b
        b1[u] = c

    # ---- f1 ----
    for c in range(16):
        W1[0, c, 2, 2] = -100.0 if c in (2, 3, 4, 5) else c + 1.0
    lin(1, 1, 0, 0); lin(2, 0, 1, 0)                 # row, col
    lin(3, 1, -1, 0); lin(4, -1, 1, 0)               # target diag
    lin(5, 1, 1, -15); lin(6, -1, -1, 15)            # target anti
    lin(7, 1, -1, -7); lin(8, -1, 1, 7)              # canvas diag
    lin(9, 1, 1, -70); lin(10, -1, -1, 70)           # canvas anti
    W1[11, 2, 2, 2] = 1.0                            # colour 2
    lin(12, -1, 0, 15.5); lin(13, 1, 0, -27.5); lin(14, -1, 0, 48.5)
    lin(15, 0, 1, -21.5); lin(16, 0, -1, 41.5)
    # selected colour (interior pixel above border above underline)
    for c in range(16):
        W1[17, c, 1, 2] = c + 1.0
    W1[17, 3, 1, 2] += -20; W1[17, 4, 1, 2] += -20
    W1[17, 4, 3, 2] += 20; W1[17, 0, 4, 2] += 20
    W1[17, 4, 2, 2] += -20; W1[17, 3, 2, 2] += -20
    b1[17] = -40
    # next swatch click
    W1[18, 4, 2, 2] = 2; W1[18, 4, 1, 2] = 2; W1[18, 3, 2, 1] = 2
    W1[18, 3, 3, 1] = 2; W1[18, 0, 3, 0] = 2; b1[18] = -9
    # swatch top-left fallback
    W1[19, 4, 2, 2] = 2; W1[19, 3, 1, 2] = 2; W1[19, 4, 3, 2] = 2; b1[19] = -5
    lin(20, 1, 1, -24.5)                             # outside target area
    lin(21, -1, 0, 33.5); lin(22, 1, 0, -43.5)       # canvas box gates
    lin(23, 0, -1, 26.5); lin(24, 0, 1, -36.5)

    # ---- f2 ----
    def hp(ch, a, b, c):
        W2[ch, 1, 1, 1] = a; W2[ch, 2, 1, 1] = b; b2[ch] = c
    hp(0, 1, 0, -7.5); hp(1, 1, -1, -0.5); hp(2, 0, -1, 7.5); hp(3, -1, -1, 14.5)
    hp(4, -1, 0, 7.5); hp(5, -1, 1, -0.5); hp(6, 0, 1, -7.5); hp(7, 1, 1, -15.5)
    for ch in range(8):
        W2[ch, 20, 1, 1] = 40.0
    hp(8, 1, 0, -38.5); hp(9, 1, -1, -7.5); hp(10, 0, -1, 31.5); hp(11, -1, -1, 69.5)
    hp(12, -1, 0, 38.5); hp(13, -1, 1, 6.5); hp(14, 0, 1, -31.5); hp(15, 1, 1, -70.5)
    for ch in range(8, 16):
        for u in (21, 22, 23, 24):
            W2[ch, u, 1, 1] = 40.0
    for ch, (ua, ub) in zip((16, 17, 18, 19), ((3, 4), (5, 6), (7, 8), (9, 10))):
        W2[ch, ua, 1, 1] = -1.5; W2[ch, ub, 1, 1] = -1.5; b2[ch] = 1.5
    W2[20, 0, 1, 1] = 1.0
    W2[21, 17, 1, 1] = 1.0
    for k, u in ((22, 13), (23, 14), (24, 15), (25, 16)):
        W2[k, 11, 1, 1] = 1.0; W2[k, 12, 1, 1] = -2.0; W2[k, u, 1, 1] = -2.0
    W2[26, 18, 1, 1] = 1.0; W2[27, 19, 1, 1] = 1.0

    # ---- f3 ----
    for p in range(8):
        W3[p, 20] = 1.0; W3[p, p] = -40.0; W3[p, 16] = -40.0; W3[p, 17] = -40.0
        q = 8 + p
        W3[q, 20] = 1.0; W3[q, q] = -40.0; W3[q, 18] = -40.0; W3[q, 19] = -40.0
    for dst, src in ((16, 21), (17, 22), (18, 23), (19, 24), (20, 25), (21, 26), (22, 27)):
        W3[dst, src] = 1.0

    # ---- z ----
    Wz, bz = P["Wz"], P["bz"]
    MX = lambda k: 32 + k
    SM = lambda k: 64 + k
    T, B, L, R = MX(17), MX(18), MX(19), MX(20)
    S = MX(16)
    posdef = [({T: 1, L: -1, R: -1}, 0), ({T: 1, R: 1}, -1), ({R: 1, T: -1, B: -1}, 0),
              ({B: 1, R: 1}, -1), ({B: 1, L: -1, R: -1}, 0), ({B: 1, L: 1}, -1),
              ({L: 1, T: -1, B: -1}, 0), ({T: 1, L: 1}, -1)]
    for p, (w, bb) in enumerate(posdef):
        for k, v in w.items():
            Wz[p, k] = v
        bz[p] = bb

    def pair(u, terms, off):
        for k, v in terms.items():
            Wz[u, k] += v; Wz[u + 1, k] += v
        bz[u] = off - 0.25; bz[u + 1] = off - 0.75

    for p in range(8):
        base = 8 + 12 * p
        Mt, St, Mc, Sc = MX(p), SM(p), MX(8 + p), SM(8 + p)
        pair(base + 0, {Mt: 40.0, St: -4096.0}, 0.0)
        pair(base + 2, {Mc: 1.0, Mt: -1.0}, 0.0)
        pair(base + 4, {Mt: 1.0, Mc: -1.0}, 0.0)
        pair(base + 6, {Mc: 40.0, Sc: -4096.0}, 0.0)
        pair(base + 8, {S: 1.0, Mt: -1.0}, 0.0)
        pair(base + 10, {Mt: 1.0, S: -1.0}, 0.0)

    # ---- gates (tanh, stateless) ----
    K = 10.0
    Wmz, bm = P["Wmz"], P["bm"]
    for p in range(8):
        base = 8 + 12 * p
        for u, smw, bias in ((p, -6.0, -4.5), (8 + p, 6.0, -10.5)):
            Wmz[u, p] = 4 * K
            Wmz[u, base + 0] += -12 * K; Wmz[u, base + 1] += 12 * K
            for i, sg in ((2, 1), (3, -1), (4, 1), (5, -1), (6, 1), (7, -1)):
                Wmz[u, base + i] += 2 * sg * K
            for i, sg in ((8, 1), (9, -1), (10, 1), (11, -1)):
                Wmz[u, base + i] += 2 * sg * smw * K
            bm[u] = bias * K

    # ---- logits ----
    Wo, bo = P["Wo"], P["bo"]
    move = {0: 4, 1: 2, 2: 2, 3: 3, 4: 3, 5: 1, 6: 1, 7: 4}
    for p, a in move.items():
        Wo[a, p] = 1.0
    for p in range(8):
        Wo[5, 128 + p] = 2.0
        Wo[6, 128 + 8 + p] = 2.0
    bo[5] = 16.0; bo[6] = 16.0
    bo[0] = -100.0; bo[7] = -100.0; bo[1] = 0.01

    # ---- click map ----
    P["Wc"][0, 21] = 10.0
    P["Wc"][0, 22] = 1.0

    return {k: np.asarray(v, np.float32) for k, v in P.items()}

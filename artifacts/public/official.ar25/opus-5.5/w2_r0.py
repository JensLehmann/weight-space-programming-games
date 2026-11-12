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
    W1[0, 11, 2, 2] = 1; W1[0, 18, 2, 3] = 1; b1[0] = -1
    W1[1, 5, 2, 2] = 1; W1[1, 18, 3, 2] = 1; b1[1] = -1
    W1[2, 4, 2, 2] = 1
    W1[3, 16, 2, 2] = 1; W1[4, 17, 2, 2] = 1; W1[5, 18, 2, 2] = 1
    for (i, j) in [(0, 2), (2, 2), (4, 2)]:
        W1[6, 10, i, j] = 2
    b1[6] = -5
    for (i, j) in [(2, 0), (2, 2), (2, 4)]:
        W1[7, 10, i, j] = 2
    b1[7] = -5
    W1[8, 0, 2, 2] = 2; W1[8, 5, 2, 1] = 2; W1[8, 5, 2, 3] = 2; b1[8] = -5
    for c, v in [(11, 5), (12, 4), (15, 3), (8, 2), (14, 1)]:
        W1[9, c, 2, 2] = v
    W1[9, 18, 2, 3] = -20
    W1[10, 0, 2, 2] = 2; W1[10, 10, 2, 0] = 2; W1[10, 10, 2, 4] = 2; b1[10] = -5
    W1[11, 0, 2, 2] = 1
    W1[12, 5, 2, 2] = 1; W1[12, 4, 2, 2] = 1; W1[12, 18, 3, 2] = 1; b1[12] = -1
    W1[13, 0, 2, 2] = 2; W1[13, 10, 0, 2] = 2; W1[13, 10, 4, 2] = 2; b1[13] = -5

    W2, b2 = p["W2"], p["b2"]
    for m, k in [(0, 0), (2, 4), (11, 12), (12, 20)]:
        W2[k, m, 1, 1] = 1; W2[k, 3, 1, 1] = 1; b2[k] = -1
        W2[k + 1, m, 1, 1] = 1; W2[k + 1, 3, 1, 1] = -1
        W2[k + 2, m, 1, 1] = 1; W2[k + 2, 4, 1, 1] = 1; b2[k + 2] = -1
        W2[k + 3, m, 1, 1] = 1; W2[k + 3, 4, 1, 1] = -1
    W2[8, 6, 1, 1] = 1; W2[9, 7, 1, 1] = 1; W2[10, 8, 1, 1] = 1; W2[11, 1, 1, 1] = 1
    W2[16, 2, 1, 1] = 1; W2[17, 9, 1, 1] = 1
    W2[18, 10, 1, 1] = 1; W2[19, 13, 1, 1] = 1
    for i in range(24):
        p["W3"][i, i] = 1.0

    G = 32
    Bmr, Bnr, Bmc, Bnc = G + 0, G + 1, G + 2, G + 3
    Rmr, Rnr, Rmc, Rnc = G + 4, G + 5, G + 6, G + 7
    V, H, SEL, P = G + 8, G + 9, G + 10, G + 11
    Smr, Snr, Smc, Snc = G + 12, G + 13, G + 14, G + 15
    REX, MH, MV = G + 16, G + 18, G + 19
    Umr, Unr, Umc, Unc = G + 20, G + 21, G + 22, G + 23
    BARMEAN = 64 + 17

    def dvec(a, b, c, d):
        v = np.zeros(96); v[a] = 63; v[b] = -63; v[c] = -63; v[d] = 63
        return v
    Dx = dvec(Bmc, Bnc, Rmc, Rnc); Dy = dvec(Bmr, Bnr, Rmr, Rnr)
    Sx = dvec(Bmc, Bnc, Smc, Snc); Sy = dvec(Bmr, Bnr, Smr, Snr)
    Ux = dvec(Bmc, Bnc, Umc, Unc); Uy = dvec(Bmr, Bnr, Umr, Unr)
    Px = dvec(Umc, Unc, Smc, Snc); Py = dvec(Umr, Unr, Smr, Snr)

    Wz, bz = p["Wz"], p["bz"]
    def unit(i, vec, gates, bias):
        Wz[i] = vec
        for idx, w in gates:
            Wz[i, idx] += w
        bz[i] = bias
    pm = [(MH, -100), (MV, -100), (REX, 100)]
    unit(0, Dx, pm + [(V, 100)], -203)
    unit(1, -Dx, pm + [(V, 100)], -203)
    unit(2, Dx, pm + [(V, -100)], -103)
    unit(3, -Dx, pm + [(V, -100)], -103)
    unit(4, Dy, pm + [(H, 100)], -203)
    unit(5, -Dy, pm + [(H, 100)], -203)
    unit(6, Dy, pm + [(H, -100)], -103)
    unit(7, -Dy, pm + [(H, -100)], -103)
    unit(8, Ux, [(MV, 100), (REX, 100)], -206)
    unit(9, -Ux, [(MV, 100), (REX, 100)], -206)
    unit(10, Uy, [(MH, 100), (REX, 100)], -206)
    unit(11, -Uy, [(MH, 100), (REX, 100)], -206)
    Wz[12, P] = 1; Wz[12, SEL] = -1; Wz[12, MH] = -1; Wz[12, MV] = -1
    Wz[13, BARMEAN] = 4096.0
    unit(14, Sx, [(REX, -100), (MH, -100), (MV, -100)], -3)
    unit(15, -Sx, [(REX, -100), (MH, -100), (MV, -100)], -3)
    unit(16, Sy, [(REX, -100), (MV, -100), (MH, -100)], -3)
    unit(17, -Sy, [(REX, -100), (MV, -100), (MH, -100)], -3)
    # mirror selected, no reflections: move mirror toward pieces
    unit(18, Py, [(MH, 100), (REX, -100)], -103)
    unit(19, -Py, [(MH, 100), (REX, -100)], -103)
    unit(20, Px, [(MV, 100), (REX, -100)], -103)
    unit(21, -Px, [(MV, 100), (REX, -100)], -103)

    s, K = 3e-5, 10.0
    Wm, Wmz, bm, Wma = p["Wm"], p["Wmz"], p["bm"], p["Wma"]
    Wmz[0, 13] = s
    Wm[1, 0] = K / s; Wmz[1, 13] = -K; bm[1] = K / 2
    Wm[2, 0] = -K / s; Wmz[2, 13] = K; bm[2] = K / 2
    for k in range(4):
        Wma[3 + k, 1 + k] = 3.0

    Wo, bo = p["Wo"], p["bo"]
    amap = {0: 3, 1: 4, 2: 4, 3: 3, 4: 1, 5: 2, 6: 2, 7: 1,
            8: 4, 9: 3, 10: 2, 11: 1, 14: 4, 15: 3, 16: 2, 17: 1,
            18: 2, 19: 1, 20: 4, 21: 3}
    for u, a in amap.items():
        Wo[a, u] = 1.0
    Wo[5, 12] = 1000.0
    Wo[5, 128 + 1] = 100.0
    Wo[5, 128 + 2] = 100.0
    rev = {1: 2, 2: 1, 3: 4, 4: 3}
    for a, r in rev.items():
        Wo[r, 128 + 3 + (a - 1)] = -14.0
    bo[:] = [-10.0, 0.0, 0.0, 0.0, 0.0, 0.2, -10.0, -10.0]

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

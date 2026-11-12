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
    W1, b1, W2, b2 = P["W1"], P["b1"], P["W2"], P["b2"]
    # ---- layer 1 (5x5) ----
    W1[0, 8, 2, 2] = 1.0                 # is8
    W1[1, 0, :, :] = 1.0                 # n0
    W1[2, 0, :, :] = 1.0; b1[2] = -1.0   # relu(n0-1)
    W1[3, 16, 2, 2] = 1.0                # row
    W1[4, 17, 2, 2] = 1.0                # col
    W1[5, 4, 2, 2] = 1.0                 # is4
    W1[6, 4, 0, 2] = 1.0
    W1[7, 4, 4, 2] = 1.0
    W1[8, 4, 2, 0] = 1.0
    W1[9, 4, 2, 4] = 1.0
    W1[10, 16, 2, 2] = -63.0; b1[10] = 1.0   # row0 indicator
    # ---- layer 2 (3x3) ----
    W2[0, 0, 1, 1] = 1.0
    W2[1, 1, :, :] = 1.0; W2[1, 2, :, :] = -1.0
    W2[2, 1, :, :] = 1.0; W2[2, 2, :, :] = -1.0; b2[2] = -1.0
    W2[3, 3, 1, 1] = 1.0
    W2[4, 4, 1, 1] = 1.0
    W2[5, 5, 1, 1] = 1.0
    W2[6, 10, 1, 1] = 1.0
    for o, bb in ((7, 0.0), (8, -1.0), (9, -2.0)):
        W2[o, 6, 0, 1] = 1; W2[o, 7, 2, 1] = 1; W2[o, 8, 1, 0] = 1; W2[o, 9, 1, 2] = 1
        b2[o] = bb
    # ---- layer 3 (1x1, axial inputs at 32+c rowmax, 64+c colmax) ----
    W3, b3 = P["W3"], P["b3"]
    W3[0, 0] = 1; W3[0, 1] = 1; W3[0, 2] = -1; b3[0] = -1      # sel
    W3[1, 0] = 1; W3[1, 1] = -1; W3[1, 2] = 1                  # oth8 (= all8 in grey)
    W3[2, 3] = 1                                               # row
    W3[3, 4] = 1                                               # col
    def ep_terms(r):
        W3[r, 5] = 1; W3[r, 7] = 1; W3[r, 8] = -2; W3[r, 9] = 1; W3[r, 6] = -2
    ep_terms(4); b3[4] = -1                                    # ep
    W3[5, 0] = 1                                               # all8
    W3[6, 5] = 1; W3[6, 6] = -1                                # grey flag source
    ep_terms(7); W3[7, 32 + 0] = 1; b3[7] = -2                 # epR: ep in row with an 8
    ep_terms(8); W3[8, 64 + 0] = 1; b3[8] = -2                 # epC: ep in col with an 8
    # ---- h: coordinate-weighted masks ----
    Uh, bh = P["Uh"], P["bh"]
    for k, m in enumerate([0, 1, 4]):
        o = 4 * k
        Uh[o + 0, 2] = 1; Uh[o + 0, m] = 1; bh[o + 0] = -1
        Uh[o + 1, 2] = -1; Uh[o + 1, m] = 1
        Uh[o + 2, 3] = 1; Uh[o + 2, m] = 1; bh[o + 2] = -1
        Uh[o + 3, 3] = -1; Uh[o + 3, m] = 1
    Uh[12, 3] = 1; Uh[12, 7] = 1; bh[12] = -1     # col * epR
    Uh[13, 3] = -1; Uh[13, 7] = 1                 # (1-col) * epR
    Uh[14, 2] = 1; Uh[14, 8] = 1; bh[14] = -1     # row * epC
    Uh[15, 2] = -1; Uh[15, 8] = 1                 # (1-row) * epC
    Wz, bz = P["Wz"], P["bz"]
    GI = 32 + 6
    EXR, EXC = 32 + 7, 32 + 8
    e = 3.5 / 63.0
    ea = 1.5 / 63.0

    def mode(base, s, o, exs, exo, grey):
        Rs, Qs, Cs, Ps = 4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3
        Ro, Qo, Co, Po = 4 * o, 4 * o + 1, 4 * o + 2, 4 * o + 3
        dv = {Ro: 1, Qo: -1, Rs: -1, Qs: 1}
        dh = {Co: 1, Po: -1, Cs: -1, Ps: 1}
        ad = {Cs: 1, Ps: 1, Rs: -1, Qs: -1, Co: -1, Po: -1, Ro: 1, Qo: 1}
        specs = [(dv, 1, e), (dv, -1, e), (dh, 1, e), (dh, -1, e), (ad, 1, ea), (ad, -1, ea)]
        for j, (terms, sign, thr) in enumerate(specs):
            r = base + j
            for i, w in terms.items():
                Wz[r, i] += sign * w
            Wz[r, exs] += 5.0; Wz[r, exo] += 5.0
            bz[r] = -10.0 - thr
            if grey:
                Wz[r, GI] += 5.0; bz[r] -= 5.0
                Wz[r, EXR] -= 5.0; Wz[r, EXC] -= 5.0   # only when no aligned target
            else:
                Wz[r, GI] -= 5.0

    mode(0, 0, 1, 32, 33, False)
    mode(6, 1, 2, 33, 36, True)
    Wz[12, 32] = 1; Wz[12, 33] = -1; Wz[12, GI] = -1
    Wz[13, 37] = 1; Wz[13, 36] = -1; Wz[13, GI] = 1; bz[13] = -1
    Wz[14, 33] = 1; Wz[14, 32] = -1; Wz[14, GI] = -1
    Wz[15, 36] = 1; Wz[15, 37] = -1; Wz[15, GI] = 1; bz[15] = -1
    Wz[16, GI] = 1.0
    # aligned-target moves (grey)
    dhR = {12: 1, 13: -1, 6: -1, 7: 1}
    dvC = {14: 1, 15: -1, 4: -1, 5: 1}
    for r, terms, sign, ex in ((17, dhR, 1, EXR), (18, dhR, -1, EXR),
                               (19, dvC, 1, EXC), (20, dvC, -1, EXC)):
        for i, w in terms.items():
            Wz[r, i] += sign * w
        Wz[r, ex] += 5.0; Wz[r, GI] += 5.0; Wz[r, 33] += 5.0
        bz[r] = -15.0 - e
    # memory
    P["Wmz"][0, 12] = 8.0; P["Wmz"][0, 13] = 8.0
    P["Wma"][0, 3] = 8.0
    P["bm"][0] = -12.0
    P["Wmz"][1, 16] = 8.0
    P["Wma"][1, 5] = 8.0
    P["bm"][1] = -12.0
    # ---- logits ----
    Wo, bo = P["Wo"], P["bo"]
    M = 1000.0
    for base in (0, 6):
        Wo[2, base + 0] = M
        Wo[1, base + 1] = M
        Wo[4, base + 2] = M
        Wo[3, base + 3] = M
    Wo[4, 17] = 2000.0; Wo[3, 18] = 2000.0
    Wo[2, 19] = 1500.0; Wo[1, 20] = 1500.0
    Wo[5, 4] = 1e5; Wo[5, 5] = 1e5
    Wo[3, 12] = 10.0; Wo[3, 13] = 10.0
    Wo[1, 128] = 10.0; bo[1] = 10.0
    Wo[6, 14] = 50.0; Wo[6, 15] = 50.0
    Wo[6, 129] = 5.0
    bo[5] = 1.0
    bo[0] = -10.0; bo[7] = -10.0
    # ---- click map ----
    P["Wc"][0, 1] = 2.0
    P["Wc"][0, 4] = 1.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}

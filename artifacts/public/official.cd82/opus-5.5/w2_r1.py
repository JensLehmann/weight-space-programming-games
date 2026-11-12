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
    for c in range(16):
        W1[0, c, 2, 2] = c + 1
    D = [
        ("t_r_lo", -1, 0, 3), ("t_r_hi", 1, 0, -12), ("t_c_lo", 0, -1, 3), ("t_c_hi", 0, 1, -12),
        ("t_s0", 1, 0, -7), ("t_s4", -1, 0, 8), ("t_s6", 0, 1, -7), ("t_s2", 0, -1, 8),
        ("t_s1", 1, -1, 1), ("t_s5", -1, 1, 1), ("t_s3", -1, -1, 16), ("t_s7", 1, 1, -14),
        ("c_r_lo", -1, 0, 34), ("c_r_hi", 1, 0, -43), ("c_c_lo", 0, -1, 27), ("c_c_hi", 0, 1, -36),
        ("c_s0", 1, 0, -38), ("c_s4", -1, 0, 39), ("c_s6", 0, 1, -31), ("c_s2", 0, -1, 32),
        ("c_s1", 1, -1, -6), ("c_s5", -1, 1, 8), ("c_s3", -1, -1, 71), ("c_s7", 1, 1, -69),
    ]
    ch = {}
    for k, (n, a, b, c) in enumerate(D, start=1):
        ch[n] = k
        W1[k, 16, 2, 2] = 63.0 * a
        W1[k, 17, 2, 2] = 63.0 * b
        b1[k] = c
    W1[25, 2, 2, 2] = 1.0                      # colour 2
    for c in range(16):                         # selected code
        W1[26, c, 0, 2] = c + 1
    W1[26, 4, 0, 2] -= 17.0                     # ignore columns under swatch border
    W1[26, 0, 2, 2] += 17.0
    W1[26, 3, 3, 2] += 17.0
    b1[26] = -34.0
    W1[27, 0, 4, 2] = 1.0
    W1[28, 18, 2, 2] = 1.0
    W1[28, 3, 2, 2] = -1.0
    W1[28, 4, 2, 2] = -1.0
    W1[29, 17, 2, 2] = -63.0
    b1[29] = 18.0

    for q in range(8):
        for pre, o, mode in (("t", q, "A"), ("t", 8 + q, "B"), ("c", 16 + q, "A")):
            names = [pre + "_r_lo", pre + "_r_hi", pre + "_c_lo", pre + "_c_hi", pre + "_s%d" % q]
            if mode == "A":
                W2[o, 0, 1, 1] = 1.0
            else:
                W2[o, 0, 1, 1] = -1.0
                b2[o] = 17.0
            for n in names:
                W2[o, ch[n], 1, 1] = -34.0
    W2[24, 25, 1, 1] = 15.0; W2[24, ch["t_r_hi"], 1, 1] = -1.0
    W2[25, 25, 1, 1] = 100.0; W2[25, ch["c_r_hi"], 1, 1] = 1.0; b2[25] = -106.0
    W2[26, 25, 1, 1] = 100.0; W2[26, ch["c_c_lo"], 1, 1] = 1.0; b2[26] = -104.0
    W2[27, 25, 1, 1] = 100.0; W2[27, ch["c_c_hi"], 1, 1] = 1.0; b2[27] = -105.0
    for o, src, g in ((28, 26, 32.0), (29, 27, 2.0), (30, 28, 2.0)):
        W2[o, src, 1, 1] = 1.0
        W2[o, ch["t_r_hi"], 1, 1] = -g
        W2[o, 29, 1, 1] = -g
    for k in range(31):
        P["W3"][k, k] = 1.0

    P["Uh"][0, 29] = 1.0
    P["Wh"][0, 0, 1, 0] = 1.0
    P["Uh"][1, 30] = 1.0
    P["bh"][1] = -5.0
    P["Wh"][1, 0, 1, 1] = 1.0
    P["Wc"][0, 30] = 0.5
    P["Wc"][0, 32 + 1] = 2.0

    Wz, bz = P["Wz"], P["bz"]
    MX, MN = 32, 64
    s_idx = MX + 28
    for q in range(8):
        n = 50.0 if q % 2 == 0 else 45.0
        sc = 4096.0 / n
        t = MX + q
        Wz[q, MX + q] = 1.0; Wz[q, MX + 8 + q] = 1.0; bz[q] = -17.0
        base = 8 + 4 * q
        Wz[base, MX + 16 + q] = 1.0; Wz[base, t] = -1.0
        Wz[base + 1, MX + 16 + q] = -1.0; Wz[base + 1, t] = 1.0
        Wz[base + 2, MN + 16 + q] = sc; Wz[base + 2, t] = -1.0
        Wz[base + 3, MN + 16 + q] = -sc; Wz[base + 3, t] = 1.0
        Wz[40 + 2 * q, t] = 1.0; Wz[40 + 2 * q, s_idx] = -1.0
        Wz[41 + 2 * q, t] = -1.0; Wz[41 + 2 * q, s_idx] = 1.0
    for k in range(4):
        Wz[56 + 2 * k, MX + 24 + k] = 1.0
        Wz[57 + 2 * k, MX + 24 + k] = 1.0
        bz[57 + 2 * k] = -1.0

    pos = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0)}

    def move(p, q):
        if p == q:
            return 5
        r, c = pos[p]
        rt, ct = pos[q]
        if r != rt:
            nr = r + (1 if rt > r else -1)
            if (nr, c) != (1, 1):
                return 2 if rt > r else 1
        if c != ct:
            nc = c + (1 if ct > c else -1)
            if (r, nc) != (1, 1):
                return 4 if ct > c else 3
        if r != rt:
            return 4
        return 1

    Wmz, bm, Wo, bo = P["Wmz"], P["bm"], P["Wo"], P["bo"]
    G, PEN, MG = 10.0, 4000.0, 60.0
    for p in range(8):
        r, c = pos[p]
        coef = [0.0] * 4
        nreq = 0
        if r == 0: coef[0], coef[1] = 1, -1; nreq += 1
        elif r == 2: coef[0], coef[1] = -1, 1; nreq += 1
        else: coef[0], coef[1] = -1, -1
        if c == 0: coef[2], coef[3] = 1, -1; nreq += 1
        elif c == 2: coef[2], coef[3] = -1, 1; nreq += 1
        else: coef[2], coef[3] = -1, -1
        for q in range(8):
            u = p * 8 + q
            for k in range(4):
                Wmz[u, 8 + 4 * q + k] = G * MG
            Wmz[u, q] = -G * PEN
            Wmz[u, 40 + 2 * q] = -G * PEN
            Wmz[u, 41 + 2 * q] = -G * PEN
            for k in range(4):
                Wmz[u, 56 + 2 * k] += G * PEN * coef[k]
                Wmz[u, 57 + 2 * k] -= G * PEN * coef[k]
            bm[u] = G * (-PEN * nreq - 1.0)
            a = move(p, q)
            w = 2.0 if p == q else 3.0 ** (-q)
            Wo[a, 128 + u] += w
            bo[a] += w
    bo[0] -= 1.0
    bo[7] -= 1.0
    bo[6] += 1e-4
    return {k: np.asarray(v, np.float32) for k, v in P.items()}

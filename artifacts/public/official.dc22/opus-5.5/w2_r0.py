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
    W1, b1, W2, b2, W3, b3 = P["W1"], P["b1"], P["W2"], P["b2"], P["W3"], P["b3"]
    W1[0, 14, 2, 2] = 1.0
    W1[1, 11, 2, 2] = 1.0
    W1[2, 17, 2, 2] = 63.0; b1[2] = -35.0
    W1[3, 16, 2, 2] = 1.0
    W1[4, 17, 2, 2] = 1.0
    W1[5, 17, 2, 2] = 63.0; b1[5] = -37.0
    W1[7, 17, 2, 2] = 63.0; b1[7] = -38.0
    for c in range(16):
        if c not in (0, 3, 4, 5):
            W1[6, c, 2, 2] = 1.0
    W2[0, 0, 1, 1] = 1.0
    W2[1, 1, 1, 1] = 1.0; W2[1, 2, 1, 1] = -1.0
    W2[2, 3, 1, 1] = 1.0
    W2[3, 4, 1, 1] = 1.0
    W2[4, 6, 1, 1] = 1.0; W2[4, 5, 1, 1] = 1.0; W2[4, 7, 1, 1] = -1.0; b2[4] = -1.0
    W3[0, 2] = 1; W3[0, 0] = 1; b3[0] = -1
    W3[1, 3] = 1; W3[1, 0] = 1; b3[1] = -1
    W3[2, 2] = 1; W3[2, 1] = 1; b3[2] = -1
    W3[3, 3] = 1; W3[3, 1] = 1; b3[3] = -1
    W3[4, 4] = 1
    Wz = P["Wz"]; K = 252.0
    Wz[0, 35] = K; Wz[0, 33] = -K
    Wz[1, 33] = K; Wz[1, 35] = -K
    Wz[2, 34] = K; Wz[2, 32] = -K
    Wz[3, 32] = K; Wz[3, 34] = -K
    Wz[4, 33] = 1.0
    Wz[5, 32] = 1.0
    Wm, Wmz, Wma, bm = P["Wm"], P["Wmz"], P["Wma"], P["bm"]
    s0 = 0.05
    Wmz[0, 4] = s0
    Wmz[1, 5] = s0
    G, H = 6000.0, 500.0
    px = s0 / 63.0
    spec = {1: (5, 1, -1), 2: (5, 1, 1), 3: (4, 0, -1), 4: (4, 0, 1)}
    for a, (zi, mi, s) in spec.items():
        u = 1 + a
        Wmz[u, zi] = -s * s0 * G
        Wm[u, mi] = s * G
        Wma[u, a] = H
        bm[u] = G * px - H
        Wm[5 + a, u] = 20.0
        bm[5 + a] = -10.0
    Wo, bo = P["Wo"], P["bo"]
    Wo[4, 0] = 1; Wo[3, 1] = 1; Wo[2, 2] = 1; Wo[1, 3] = 1
    cw = {1: 4, 4: 2, 2: 3, 3: 1}
    ccw = {1: 3, 4: 1, 2: 4, 3: 2}

    def add(target, unit, amount):
        Wo[target, 128 + unit] += amount / 2.0
        bo[target] += amount / 2.0

    c = 700.0
    for a in range(1, 5):
        q = 1 + a
        add(a, q, -500.0)
        add(cw[a], q, 3.0)
        add(ccw[a], q, 1.0)
        add(6, q, c)
        add(6, 5 + a, c)
    bo[0] += -1000.0; bo[5] += -1000.0; bo[7] += -1000.0; bo[6] += -1000.0
    P["Wc"][0, 4] = 1.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}

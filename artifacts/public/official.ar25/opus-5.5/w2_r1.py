import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float64) for k, s in shapes.items()}

    # ---- f1 ----
    W1, b1 = p["W1"], p["b1"]
    for di in range(3):
        for dj in range(3):
            W1[0, 11, 1 + di, 1 + dj] = 0.25
    W1[0, 11, 2, 2] = 1.0
    b1[0] = -1.5                               # uncovered-target block pixel
    W1[1, 17, 2, 2] = 63.0; b1[1] = -62.0      # column 63 detector
    W1[2, 16, 2, 2] = 63.0; b1[2] = -62.0      # row 63 detector
    W1[3, 5, 2, 2] = 1.0; W1[3, 4, 2, 2] = 1.0 # cover (piece / reflection)

    # ---- f2 ----
    W2 = p["W2"]
    W2[0, 0, 1, 1] = 4.0; W2[0, 1, 1, 1] = -8.0; W2[0, 2, 1, 1] = -8.0
    W2[1, 3, 1, 1] = 1.0; W2[1, 1, 1, 1] = -2.0; W2[1, 2, 1, 1] = -2.0

    # ---- f3 ----
    W3, b3 = p["W3"], p["b3"]
    W3[0, 0] = 1.0
    W3[1, 0] = 1.0; b3[1] = -1.0               # Tm = f3[0]-f3[1]
    W3[2, 1] = 1.0                             # cover C

    # ---- recurrent: reach front + linear and quadratic accumulators ----
    Uh, bh, Wh = p["Uh"], p["bh"], p["Wh"]
    cross = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    for (ki, kj) in cross:
        Wh[0, 0, ki, kj] = 1.0; Wh[0, 1, ki, kj] = -1.0
        Wh[1, 0, ki, kj] = 1.0; Wh[1, 1, ki, kj] = -1.0
    Uh[0, 2] = 1.0
    Uh[1, 2] = 1.0; bh[1] = -1.0
    # Acc (h2): accumulates reach indicator at uncovered-target pixels
    Wh[2, 2, 1, 1] = 1.0; Wh[2, 0, 1, 1] = 1.0; Wh[2, 1, 1, 1] = -1.0
    Uh[2, 0] = 1000.0; Uh[2, 1] = -1000.0; bh[2] = -1000.0
    # Acc2 (h3): accumulates Acc -> ~ (T-d)^2/2
    Wh[3, 3, 1, 1] = 1.0; Wh[3, 2, 1, 1] = 1.0
    Uh[3, 0] = 1000.0; Uh[3, 1] = -1000.0; bh[3] = -1000.0

    # ---- z ----
    Wz = p["Wz"]
    lam = 300.0
    Wz[0, 16 + 2] = 1.0                        # mean(Acc)
    Wz[0, 16 + 3] = 1.0 / 64.0                 # mean(Acc2)/64
    Wz[1, 64 + 0] = lam; Wz[1, 64 + 1] = -lam  # lam*mean(Tm)

    # ---- memory ----
    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    eps = 1e-6
    Ge = 65536.0
    G = Ge / eps
    K = 2e7
    dlt = 8.0
    Wmz[0, 0] = eps; Wmz[0, 1] = -eps          # m0 = tanh(eps*S)
    acts = [1, 2, 3, 4, 5]
    for j in acts:                             # D_j: last move/switch
        Wm[j, j] = 5.0
        for k in acts:
            Wma[j, k] = 10.0 if k == j else -10.0
        bm[j] = -2.5
    for j in range(1, 5):                      # R_j: prev move j and improved
        u = 5 + j
        Wma[u, j] = K
        bm[u] = -K - dlt
        Wmz[u, 0] = Ge; Wmz[u, 1] = -Ge
        Wm[u, 0] = -G
    u = 10                                     # In: prev was a move and got worse
    for j in range(1, 5):
        Wma[u, j] = K
    bm[u] = -K - dlt
    Wmz[u, 0] = -Ge; Wmz[u, 1] = Ge
    Wm[u, 0] = G

    # ---- logits (indicator I = (m+1)/2) ----
    Wo, bo = p["Wo"], p["bo"]
    M = 128

    def add(a, unit, w):
        Wo[a, M + unit] += w / 2.0
        bo[a] += w / 2.0

    prevD = {3: 1, 2: 3, 4: 2, 5: 4, 1: 5}
    for a in acts:
        add(a, prevD[a], 1.0)
    for a in range(1, 5):
        add(a, 5 + a, 3.0)
    add(7, 10, 2.0)
    bo[1] += 0.1
    bo[0] = -100.0; bo[6] = -100.0

    return {k: v.astype(np.float32) for k, v in p.items()}

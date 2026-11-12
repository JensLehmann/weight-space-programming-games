import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, W3 = p["W1"], p["b1"], p["W2"], p["W3"]
    A = 10  # walker colour
    W1[0, A, 2, 2] = 1.0
    W1[2, 17, 2, 2] = 1.0; W1[2, A, 2, 2] = 1.0; b1[2] = -1.0   # col on walker
    W1[3, A, 2, 2] = 1.0; W1[3, 17, 2, 2] = -1.0                # 1-col on walker
    W1[4, 16, 2, 2] = 1.0; W1[4, A, 2, 2] = 1.0; b1[4] = -1.0   # row on walker
    W1[5, A, 2, 2] = 1.0; W1[5, 16, 2, 2] = -1.0                # 1-row on walker
    for c in range(6):
        W2[c, c, 1, 1] = 1.0
    for c in range(4):
        W3[3 + c, 2 + c] = 1.0
    Wz = p["Wz"]
    for i in range(4):
        Wz[i, 35 + i] = 1.0   # z0..3 = xmax, 1-xmin, ymax, 1-ymin

    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    K = 10.0; M = 200.0; S = 0.05; R = 1.0 / S
    for i in range(4):
        Wmz[i, i] = S
    # NU (12): prev up and no vertical change ; ND (13): prev down and no change
    for u, a, s in ((12, 1, 1.0), (13, 2, -1.0)):
        Wma[u, a] = K * M
        bm[u] = K * (-M + 1.5)
        Wmz[u, 2] += s * 63 * K; Wm[u, 2] -= s * 63 * K * R
        Wmz[u, 3] -= s * 63 * K; Wm[u, 3] += s * 63 * K * R
    # previous-action copies (16..19 for actions 1..4)
    for a in range(1, 5):
        Wma[15 + a, a] = 2 * K
        bm[15 + a] = -K
    # chaotic reservoir 20..59 (fixed-seed pseudo-random generator)
    rng = np.random.default_rng(12345)
    N = 40
    sl = slice(20, 60)
    Wm[sl, sl] = 3.0 * rng.standard_normal((N, N)) / np.sqrt(N)
    bm[sl] = rng.standard_normal(N)
    Wma[sl, :] = 1.5 * rng.standard_normal((N, 8))
    Wmz[sl, 0:4] = 4.0 * rng.standard_normal((N, 4))

    Wo, bo = p["Wo"], p["bo"]
    mi = lambda j: 128 + j
    Wo[1:5, 128 + 20:128 + 60] = 0.35 * rng.standard_normal((4, N))

    def add01(r, j, w):  # add w * (0/1 version of memory unit j) to logit r
        Wo[r, mi(j)] += 0.5 * w
        bo[r] += 0.5 * w

    for a in range(1, 5):
        add01(a, 15 + a, 1.0)          # momentum
    for a, b in ((1, 2), (2, 1), (3, 4), (4, 3)):
        add01(a, 15 + b, -1.5)         # avoid undoing last move
    add01(1, 12, -8.0)                 # up was blocked
    add01(2, 13, -8.0)                 # down was blocked
    bo[0] = -1000.0; bo[5] = -1000.0; bo[6] = -1000.0; bo[7] = -1000.0

    return {k: np.asarray(v, np.float32) for k, v in p.items()}

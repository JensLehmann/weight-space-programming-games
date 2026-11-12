import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32),
        "bh": (16,), "Wz": (128, 96), "bz": (128,), "Wm": (64, 64),
        "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,), "Wo": (8, 192),
        "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ---- layer 1: detectors ----
    W1[0, 11, 2, 2] += 1
    for dy, dx in nb:
        for col in (11, 4, 5):
            W1[0, col, 2 + dy, 2 + dx] += 1
    b1[0] = -4
    W1[1, 4, 2, 2] += 1
    W1[1, 11, 2, 2] += 1
    for dy, dx in nb:
        W1[1, 4, 2 + dy, 2 + dx] += 1
    b1[1] = -4
    for d in range(-2, 3):
        W1[2, 10, 2 + d, 2] += 1
        W1[3, 10, 2, 2 + d] += 1
    b1[2] = -4
    b1[3] = -4
    W1[4, 5, 2, 1] += 1; W1[4, 0, 2, 2] += 1; W1[4, 5, 2, 3] += 1
    b1[4] = -2
    W1[5, 0, 2, 1] += 1; W1[5, 10, 2, 2] += 1; W1[5, 0, 2, 4] += 1
    b1[5] = -2
    W1[6, 0, 1, 2] += 1; W1[6, 10, 2, 2] += 1; W1[6, 0, 4, 2] += 1
    b1[6] = -2
    W1[7, 16, 2, 2] = 1
    W1[8, 17, 2, 2] = 1
    b1[9] = 1
    for col in (0, 9, 11):
        W1[10, col, 2, 2] += 1
    for dy, dx in nb:
        W1[10, 5, 2 + dy, 2 + dx] += 1
    b1[10] = -4

    # ---- layer 2 ----
    def w2(o, i, v):
        W2[o, i, 1, 1] += v

    def gate(o, D, coord, inv):
        w2(o, D, 1)
        if inv:
            w2(o, coord, -1)
        else:
            w2(o, coord, 1)
            b2[o] = -1

    ROW, COL = 7, 8
    for base, D in ((0, 0), (4, 1), (15, 10)):
        gate(base, D, COL, False)
        gate(base + 1, D, COL, True)
        gate(base + 2, D, ROW, False)
        gate(base + 3, D, ROW, True)
    for o, i in ((8, 0), (9, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), (23, 10)):
        w2(o, i, 1)
    gate(19, 2, COL, False)
    gate(20, 2, COL, True)
    gate(21, 3, ROW, False)
    gate(22, 3, ROW, True)

    for c in range(24):
        p["W3"][c, c] = 1.0

    # ---- z units ----
    gi = lambda c: 32 + c
    EX = {0: 1, 1: -1, 4: -1, 5: 1}
    EY = {2: 1, 3: -1, 6: -1, 7: 1}
    EXM = {0: 1, 1: -1, 19: -1, 20: 1}
    EYM = {2: 1, 3: -1, 21: -1, 22: 1}
    PT, YP, V, H, P, HMS, VMS = 8, 9, 10, 11, 12, 13, 14
    A1, A2, A3, A4 = 1, 2, 3, 4
    K = 8.0
    units = [
        (EX, +1, {P: 1, V: 1, PT: 1, YP: 1}, 4, 0.05, A3),
        (EX, -1, {P: 1, V: 1, PT: 1, YP: 1}, 4, 0.05, A4),
        (EX, +1, {P: 1, V: -1, PT: 1, YP: 1}, 3, 0.05, A4),
        (EX, -1, {P: 1, V: -1, PT: 1, YP: 1}, 3, 0.05, A3),
        (EY, +1, {P: 1, H: 1, PT: 1, YP: 1}, 4, 0.05, A1),
        (EY, -1, {P: 1, H: 1, PT: 1, YP: 1}, 4, 0.05, A2),
        (EY, +1, {P: 1, H: -1, PT: 1, YP: 1}, 3, 0.05, A2),
        (EY, -1, {P: 1, H: -1, PT: 1, YP: 1}, 3, 0.05, A1),
        (EXM, +1, {VMS: 1, PT: 1}, 2, 0.07, A4),
        (EXM, -1, {VMS: 1, PT: 1}, 2, 0.07, A3),
        (EYM, +1, {HMS: 1, PT: 1}, 2, 0.07, A2),
        (EYM, -1, {HMS: 1, PT: 1}, 2, 0.07, A1),
    ]
    Wz, bz, Wo, bo = p["Wz"], p["bz"], p["Wo"], p["bo"]
    for u, (err, s, cond, cnt, th, act) in enumerate(units):
        for c, v in err.items():
            Wz[u, gi(c)] += s * v
        for c, v in cond.items():
            Wz[u, gi(c)] += K * v
        bz[u] = -th - K * cnt
        Wo[act, u] = 10.0

    bo[:] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.001, -1.0, -1.0], np.float32)

    # ---- memory: repeat counters, wall guard, anti-reversal ----
    Wm, Wma, bm = p["Wm"], p["Wma"], p["bm"]
    opp = {1: 2, 2: 1, 3: 4, 4: 3}
    for k, j in enumerate((1, 2, 3, 4)):
        # counter m_k
        Wm[k, k] = 1.0
        for a in range(8):
            Wma[k, a] = 0.05 if a == j else -3.0
        # threshold n_k (from previous counter)
        Wm[4 + k, k] = 20.0
        bm[4 + k] = -11.0
        Wo[j, 128 + 4 + k] += -15.0
        bo[j] += -15.0 * 1.0  # with n=-1: -15*(-1) -15 = 0
        # previous-action flag r_k
        Wma[8 + k, j] = 3.0
        Wo[opp[j], 128 + 8 + k] += -20.0
    return p

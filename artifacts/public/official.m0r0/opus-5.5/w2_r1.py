import numpy as np

def build():
    K = 20.0
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
    W1[0, 10, 2, 2] = 1.0
    W1[1, 10, 2, 2] = 2.0; W1[1, 17, 2, 2] = 1.0; b1[1] = -2.0   # col on piece
    W1[2, 10, 2, 2] = 2.0; W1[2, 17, 2, 2] = -1.0; b1[2] = -1.0  # 1-col on piece
    W1[3, 10, 2, 2] = 2.0; W1[3, 16, 2, 2] = 1.0; b1[3] = -2.0   # row on piece
    W1[4, 10, 2, 2] = 2.0; W1[4, 16, 2, 2] = -1.0; b1[4] = -1.0  # 1-row on piece
    for c in range(5):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0

    Wz = p["Wz"]
    # z6..z9 = extremes q_k (f3 max of channels 1..4)
    for k in range(4):
        Wz[6 + k, 32 + 1 + k] = 1.0

    Wm, Wmz, Wma, bm = p["Wm"], p["Wmz"], p["Wma"], p["bm"]
    Wo, bo = p["Wo"], p["bo"]

    # m0..m3: stored extremes
    for k in range(4):
        Wmz[k, 6 + k] = 0.1
    # m4..m7: previous action indicator (+-1) for actions 1..4
    P = {}
    for i, a in enumerate([1, 2, 3, 4]):
        u = 4 + i
        Wma[u, a] = 2 * K
        bm[u] = -K
        P[a] = u

    nxt = {1: 3, 3: 2, 2: 4, 4: 1}
    # movement detectors m8..m39
    u = 8
    for a in [1, 2, 3, 4]:
        for k in range(4):
            for s in (1.0, -1.0):
                Wma[u, a] = 400.0
                Wm[u, k] = 3000.0 * s
                Wmz[u, 6 + k] = -300.0 * s
                bm[u] = -409.0
                # momentum on a: +2u+2 per detector  (4*M)
                Wo[a, 128 + u] += 2.0; bo[a] += 2.0
                # suppress turn while moving: -6*M
                Wo[nxt[a], 128 + u] += -3.0; bo[nxt[a]] += -3.0
                u += 1
    for a in [1, 2, 3, 4]:
        # stuck penalty on a: -4*(P+1)/2
        Wo[a, 128 + P[a]] += -2.0; bo[a] += -2.0
        # stuck turn bonus: +6*(P+1)/2
        Wo[nxt[a], 128 + P[a]] += 3.0; bo[nxt[a]] += 3.0

    # ring m40..m59
    R0, N = 40, 20
    for i in range(1, N):
        Wm[R0 + i, R0 + i - 1] = K
        bm[R0 + i] = -0.5 * K
    Wm[R0, R0 + N - 1] = K
    bm[R0] = 0.5 * K
    Wma[R0, :] = -K

    st = 13579
    for i in range(N):
        st = (st * 1103515245 + 12345) % (2 ** 31)
        d = 1 + (st >> 16) % 4
        kick = (i == 0) or ((st >> 20) % 5 < 2)
        c = 2.5 if kick else 0.25
        Wo[d, 128 + R0 + i] += c
        bo[d] += c

    bo[0] += -6.0; bo[5] += -6.0; bo[6] += -6.0; bo[7] += -6.0

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

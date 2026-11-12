import numpy as np

A_SYMS = [
    "00100 00100 01110 00100 11111",
    "00001 00101 11111 10100 10000",
    "11111 10100 10000 10100 11111",
    "00100 11111 10101 10101 00100",
    "10001 11111 00100 11111 10001",
    "11111 00100 11100 10000 11111",
    "10001 11111 10001 10001 11011",
]


def build():
    f = np.float32
    P = {
        "W1": np.zeros((32, 19, 5, 5), f), "b1": np.zeros(32, f),
        "W2": np.zeros((32, 32, 3, 3), f), "b2": np.zeros(32, f),
        "W3": np.zeros((32, 96), f), "b3": np.zeros(32, f),
        "Wh": np.zeros((16, 16, 3, 3), f), "Uh": np.zeros((16, 32), f), "bh": np.zeros(16, f),
        "Wz": np.zeros((128, 96), f), "bz": np.zeros(128, f),
        "Wm": np.zeros((64, 64), f), "Wmz": np.zeros((64, 128), f),
        "Wma": np.zeros((64, 8), f), "bm": np.zeros(64, f),
        "Wo": np.zeros((8, 192), f), "bo": np.zeros(8, f),
        "Wc": np.zeros((1, 48), f), "bc": np.zeros(1, f),
    }
    M = 32768.0
    K = 32768.0
    wts = [1.0, 5.0, 25.0, 125.0, 625.0, 3200.0, 16384.0]

    # rotation orbits of 5x5 grid
    ids, orb = {}, {}
    for r in range(5):
        for c in range(5):
            pts = [(r, c)]
            for _ in range(3):
                a, b = pts[-1]
                pts.append((b, 4 - a))
            key = min(pts)
            if key not in ids:
                ids[key] = len(ids)
            orb[(r, c)] = ids[key]
    vals = []
    for s in A_SYMS:
        m = [[int(ch) for ch in row] for row in s.split()]
        vals.append(sum(wts[orb[(r, c)]] * m[r][c] for r in range(5) for c in range(5)))

    # ---- layer 1 ----
    W1 = P["W1"]
    for ch in (10, 7, 11):
        W1[0, ch, 0, :] = 1
        W1[1, ch, 4, :] = 1
        W1[2, ch, :, 0] = 1
        W1[3, ch, :, 4] = 1
    W1[4, 10, 0, 0] = 1      # A family corner
    W1[5, 0, 2, 2] = 1       # cursor colour 0
    W1[6, 2, 2, 2] = 1       # colour 2 (top background)
    for r in range(5):
        for c in range(5):
            W1[7 + orb[(r, c)], 5, r, c] = 1   # orbit counts

    # ---- layer 2 ----
    W2, b2 = P["W2"], P["b2"]

    def ring(j, c):
        for (u, p, q) in [(0, 0, 0), (0, 0, 2), (1, 2, 0), (1, 2, 2),
                          (2, 0, 0), (2, 2, 0), (3, 0, 2), (3, 2, 2)]:
            W2[j, u, p, q] += c

    def hashc(j):
        for o in range(7):
            W2[j, 7 + o, 1, 1] += wts[o]

    th = sorted(set(v + d for v in vals for d in (-1, 0, 1)))
    assert len(th) <= 21
    ui = {}
    for n, t in enumerate(th):
        ui[t] = n
        hashc(n)
        ring(n, M)
        W2[n, 4, 0, 0] += M
        b2[n] = -t - 41 * M
    HB, CUR, C2 = 21, 22, 23
    hashc(HB)
    ring(HB, M)
    W2[HB, 4, 0, 0] += -M
    b2[HB] = -40 * M
    W2[CUR, 5, 1, 1] = 1
    W2[C2, 6, 1, 1] = 1

    # ---- layer 3 (input: f2 | rowmax | colmax) ----
    W3, b3 = P["W3"], P["b3"]
    RM, CM = 32, 64
    for k, v in enumerate(vals):
        tri = [(ui[v - 1], 1.0), (ui[v], -2.0), (ui[v + 1], 1.0)]
        for (u, c) in tri:
            W3[k, u] += c
            W3[8 + k, u] += c
        W3[k, CM + CUR] = 1; W3[k, RM + C2] = -1; b3[k] = -1      # S_k
        W3[8 + k, RM + C2] = 1; b3[8 + k] = -1                     # L_k
    W3[7, HB] = 1; W3[7, CM + CUR] = K; W3[7, RM + C2] = -K; b3[7] = -K   # T
    W3[15, HB] = 1; W3[15, RM + C2] = K; b3[15] = -K                      # RB

    # ---- recurrent scan ----
    Wh, Uh, bh = P["Wh"], P["Uh"], P["bh"]
    for k in range(7):
        Wh[k, k, 1, 0] = 1
        Uh[k, 8 + k] = 1
        Uh[k, 15] = -10
        Wh[7 + k, k, 1, 0] = K
        Uh[7 + k, 15] = 1
        bh[7 + k] = -K

    # ---- decision ----
    Wz, bz = P["Wz"], P["bz"]
    for k in range(7):
        Wz[2 * k, 7 + k] = 1; Wz[2 * k, 32 + 7] = -1
        Wz[2 * k, 32 + k] = K; bz[2 * k] = -K
        Wz[2 * k + 1, 7 + k] = -1; Wz[2 * k + 1, 32 + 7] = 1
        Wz[2 * k + 1, 32 + k] = K; bz[2 * k + 1] = -K

    P["Wo"][2, 0:14] = 10
    P["bo"][:] = np.array([-100, -100, -5, -100, 0, -100, -100, -100], f)
    return {k: np.asarray(v, np.float32) for k, v in P.items()}

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

    def w1(o, c, dr, dc, v):
        W1[o, c, 2 + dr, 2 + dc] += v

    N3 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    # ch0: obstacle colours
    for c in (3, 4, 5, 12, 15):
        w1(0, c, 0, 0, 1)
    # ch1/2: sandwich detectors (blocker tiles)
    w1(1, 1, 0, 0, 1); w1(1, 2, 0, -1, 1); w1(1, 2, 0, 1, 1); b1[1] = -2
    w1(2, 1, 0, 0, 1); w1(2, 2, -1, 0, 1); w1(2, 2, 1, 0, 1); b1[2] = -2
    # ch3: box border pixel (4/3), raw
    w1(3, 4, 0, 0, 1); w1(3, 3, 0, 0, 1)
    # ch4: row 63
    w1(4, 16, 0, 0, 63); b1[4] = -62
    # ch5: target source: centre 2 with 9 neighbours
    for c in range(16):
        if c != 2:
            w1(5, c, 0, 0, -10)
    for dr, dc in N3:
        if dr or dc:
            w1(5, 9, dr, dc, 1)
    # ch6: carried-box L corner of colour 0
    w1(6, 0, 0, 0, 1); w1(6, 0, 1, 0, 1); w1(6, 0, 0, 1, 1); b1[6] = -2
    # ch22: clean 2 (no colour 1 in 3x3); ch23: 0 not near 14
    w1(22, 2, 0, 0, 1); w1(23, 0, 0, 0, 1)
    for dr, dc in N3:
        w1(22, 1, dr, dc, -1)
        w1(23, 14, dr, dc, -1)
    # ch8: facing border colour 3 (raw)
    w1(8, 3, 0, 0, 1)
    vs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    others = [c for c in range(16) if c not in (0, 9)]
    for d, (vr, vc) in enumerate(vs):
        wr, wc = -vr, -vc
        o = 9 + d  # player outline marker
        w1(o, 14, wr, wc, 1); w1(o, 0, wr, wc, 1)
        w1(o, 14, 2 * wr, 2 * wc, 1)
        w1(o, 14, 0, 0, -1); w1(o, 0, 0, 0, -1)
        b1[o] = -1
        # carried-box pure-9 pixel at p - v
        o = 13 + d
        w1(o, 9, wr, wc, 1)
        for dr, dc in N3:
            for c in others:
                w1(o, c, wr + dr, wc + dc, -1)
    # ch17: faced anti-helper border colour 11
    w1(17, 11, 0, 0, 1)

    # layer 2
    W2, b2 = p["W2"], p["b2"]
    W2[0, 0, 1, 1] = 1
    W2[0, 1, :, :] = 1
    W2[0, 2, :, :] = 1
    W2[3, 3, 1, 1] = 1; W2[3, 4, 1, 1] = -2; W2[3, 22, :, :] = -2
    W2[8, 8, 1, 1] = 1; W2[8, 22, :, :] = -1
    for k in (5, 6, 9, 10, 11, 12, 17):
        W2[k, k, 1, 1] = 1
    for d, (vr, vc) in enumerate(vs):
        W2[13 + d, 13 + d, 1 - vr, 1 - vc] = 1
    # ch7: carried-box 0 with clean 2 to its left/up/up-left
    W2[7, 22, 1, 0] = 0.2; W2[7, 22, 0, 1] = 0.2; W2[7, 22, 0, 0] = 0.2
    W2[7, 23, 1, 1] = 1; b2[7] = -1
    for k in (0, 3) + tuple(range(5, 18)):
        p["W3"][k, k] = 1

    # recurrent fields
    a, S, B = 0.248, 1.0, 1000.0
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    for ch in (0, 1):
        Wh[ch, ch, 0, 1] = Wh[ch, ch, 2, 1] = Wh[ch, ch, 1, 0] = Wh[ch, ch, 1, 2] = a
    Uh[0, 3] = S + B; Uh[0, 0] = -B
    Uh[1, 5] = S; Uh[1, 0] = -B
    for d, (vr, vc) in enumerate(vs):
        Wh[2 + d, 0, 1, 1] = 1; Uh[2 + d, 9 + d] = B; bh[2 + d] = -B
        Wh[6 + d, 1, 1 + vr, 1 + vc] = 1; Uh[6 + d, 13 + d] = B; bh[6 + d] = -B

    # readout: g = [hmax 0-15, hmean 16-31, f3max 32-63, f3mean 64-95]
    K = 1e4
    Wz, bz = p["Wz"], p["bz"]
    Cidx = 32 + 6
    for d in range(4):
        Wz[d, 2 + d] = 1; Wz[d, Cidx] = -K
        Wz[4 + d, 6 + d] = 1; Wz[4 + d, Cidx] = K; bz[4 + d] = -K
    Wz[8, 64 + 8] = 4096; Wz[8, Cidx] = -K; bz[8] = -7
    Wz[9, 64 + 8] = 4096; Wz[9, Cidx] = -K; bz[9] = -8
    Wz[10, 32 + 7] = 5; Wz[10, Cidx] = 10; bz[10] = -10
    Wz[11, 32 + 7] = 5; Wz[11, Cidx] = 10; bz[11] = -11
    Wz[12, 32 + 17] = 1; Wz[12, Cidx] = -K

    Wo, bo = p["Wo"], p["bo"]
    for d in range(4):
        Wo[1 + d, d] = 1
        Wo[1 + d, 4 + d] = 1
    Wo[5, 8] = 1e5; Wo[5, 9] = -1e5
    Wo[5, 10] = 1e5; Wo[5, 11] = -1e5
    Wo[5, 12] = 1e5
    bo[0] = -10; bo[5] = -1; bo[6] = -10; bo[7] = -10

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

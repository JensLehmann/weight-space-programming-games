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
    W1, b1, W2, b2, W3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"]
    W1[0, 0, 2, 2] = 1.0                      # is0
    W1[1, 9, 2, 2] = 1.0                      # is9
    W1[2, 12, 2, 2] = 1.0                     # is12
    for c in (11, 14):
        W1[3, c, 2, 2] = 1.0
        W1[4, c, 2, 2] = 1.0
        for o in (5, 6, 7, 8):
            W1[o, c, 2, 2] = 1.0
        W1[9, c, 2, 2] = 1.0
    W1[4, 4, :, :] = -10.0                    # marker: C with no 4 nearby
    W1[5, 4, 2, 1] = 1.0; b1[5] = -1.0
    W1[6, 4, 2, 3] = 1.0; b1[6] = -1.0
    W1[7, 4, 1, 2] = 1.0; b1[7] = -1.0
    W1[8, 4, 3, 2] = 1.0; b1[8] = -1.0
    W1[9, 4, :, :] = 0.1; b1[9] = -1.0
    # live button (only at 9 pixels)
    W1[12, 9, 2, 2] = 3.0; W1[12, 0, :, :] = 0.1; b1[12] = -3.0
    # dead button
    W1[13, 9, 2, 2] = 1.0; W1[13, 0, :, :] = -10.0
    # wall
    W1[14, 5, 2, 2] = 1.0
    for c in list(range(10)) + [13, 14]:
        W2[c, c, 1, 1] = 1.0
        W3[c, c] = 1.0
    # saturate live-button indicator
    W2[12, 12, 1, 1] = 10.0
    W2[15, 12, 1, 1] = 10.0; b2[15] = -1.0
    W3[12, 12] = 1.0; W3[12, 15] = -1.0
    # shifted marker copies for strict directional counters
    W2[16, 4, 1, 0] = 1.0                     # marker from left
    W2[17, 4, 1, 2] = 1.0                     # marker from right
    W2[18, 4, 0, 1] = 1.0                     # marker from above
    W2[19, 4, 2, 1] = 1.0                     # marker from below
    W3[16, 64 + 16] = 1.0
    W3[17, 64 + 17] = 1.0
    W3[18, 32 + 18] = 1.0
    W3[19, 32 + 19] = 1.0

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    # strict directional marker counters
    Wh[0, 0, 1, 0] = 1.0; Uh[0, 16] = 1.0     # E_L
    Wh[3, 3, 1, 2] = 1.0; Uh[3, 17] = 1.0     # E_R
    Wh[4, 4, 0, 1] = 1.0; Uh[4, 18] = 1.0     # E_U
    Wh[5, 5, 2, 1] = 1.0; Uh[5, 19] = 1.0     # E_D
    for g, e, f in ((6, 0, 5), (7, 3, 6), (8, 4, 7), (9, 5, 8)):
        Wh[g, e, 1, 1] = 1.0
        Uh[g, f] = 100.0
        bh[g] = -100.0
    # own container flood F = h10 - h11
    for o, off in ((10, 0.0), (11, -1.0)):
        Wh[o, 10, :, :] = 1.0
        Wh[o, 11, :, :] = -1.0
        for g in (6, 7, 8, 9):
            Wh[o, g, 1, 1] = 1.0
        Uh[o, 0] = 100.0
        Uh[o, 9] = 1000.0
        bh[o] = -100.0 + off
    # own button O = h12 - h13
    for o, off in ((12, 0.0), (13, -1.0)):
        Wh[o, 10, :, :] = 1.0
        Wh[o, 11, :, :] = -1.0
        Wh[o, 12, :, :] = 1.0
        Wh[o, 13, :, :] = -1.0
        Uh[o, 1] = 100.0
        bh[o] = -100.0 + off
    # clock -> 10
    Wh[14, 14, 1, 1] = 0.9
    bh[14] = 1.0
    # late refill flood G = h1 - h2
    K = 1.0e4
    for o, off in ((1, 0.0), (2, -1.0)):
        Wh[o, 1, :, :] = 1.0
        Wh[o, 2, :, :] = -1.0
        Uh[o, 14] = 1000.0                    # wall medium
        Uh[o, 13] = 1000.0                    # dead button medium
        Uh[o, 12] = 1000.0                    # live button seed
        Wh[o, 12, 1, 1] = -2000.0             # suppress own buttons
        Wh[o, 13, 1, 1] = 2000.0
        Wh[o, 14, 1, 1] = K                   # clock gate
        bh[o] = -1000.0 - 10.0 * K + off
    # target: G on dead-button pixels only
    Wh[15, 1, 1, 1] = 1.0
    Wh[15, 2, 1, 1] = -1.0
    Uh[15, 13] = 1000.0
    bh[15] = -1000.0

    # click map over [f3 (32), h (16)]
    p["Wc"][0, 32 + 15] = 200.0
    p["Wc"][0, 32 + 12] = 100.0
    p["Wc"][0, 32 + 13] = -100.0
    p["Wc"][0, 2] = 30.0
    p["Wc"][0, 1] = 1.0
    p["bo"][6] = 1.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

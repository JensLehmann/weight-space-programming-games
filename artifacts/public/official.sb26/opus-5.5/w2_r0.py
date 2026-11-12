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
    colors = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    NC = len(colors)

    # ---- layer 1 ----
    for k, c in enumerate(colors):
        W1[k, c, 2, 2] = 1.0                 # is colour c
    W1[11, 0, 2, 2] = 1.0                    # black
    for a in (2, 3):                         # solid red 2x2 anchor
        for b in (2, 3):
            W1[12, 2, a, b] = 1.0
    b1[12] = -3.0
    R = 63.0
    W1[13, 16, 2, 2] = -R; b1[13] = 8.0      # top gate r<=7
    W1[14, 16, 2, 2] = -R; b1[14] = 7.0
    W1[15, 16, 2, 2] = -R; b1[15] = 53.0     # mid gate r<=52
    W1[16, 16, 2, 2] = -R; b1[16] = 52.0
    W1[17, 16, 2, 2] = R;  b1[17] = -56.0    # bottom gate r>=57
    W1[18, 16, 2, 2] = R;  b1[18] = -57.0
    W1[19, 16, 2, 2] = 1.0                   # row
    W1[20, 17, 2, 2] = 1.0                   # col

    # ---- layer 2 ----
    for k in range(NC):
        W2[k, k, 1, 1] = 4.0                 # target leftness
        W2[k, 13, 1, 1] = 4.0
        W2[k, 14, 1, 1] = -4.0
        W2[k, 20, 1, 1] = -3.6
        b2[k] = -4.0
        W2[11 + k, k, 1, 1] = 1.0            # palette pixel of colour c
        W2[11 + k, 17, 1, 1] = 1.0
        W2[11 + k, 18, 1, 1] = -1.0
        b2[11 + k] = -1.0
    W2[22, 11, 1, 1] = 1.0; W2[22, 17, 1, 1] = 1.0; W2[22, 18, 1, 1] = -1.0; b2[22] = -1.0
    W2[23, 12, 1, 1] = 1.0; W2[23, 15, 1, 1] = 1.0; W2[23, 16, 1, 1] = -1.0; b2[23] = -1.0
    W2[24, 19, 1, 1] = 1.0                   # row
    W2[25, 20, 1, 1] = 1.0                   # col
    W2[26, 17, 1, 1] = 1.0; W2[26, 18, 1, 1] = -1.0   # bottom band

    # ---- layer 3 (axial) ----
    for k in range(NC):
        W3[k, 32 + k] = 1.0                  # rowmax leftness
        W3[11 + k, 11 + k] = 1.0             # palette pixel
    W3[22, 32 + 22] = 1.0                    # rowmax selection
    W3[23, 23] = 1.0                         # slot
    W3[24, 24] = 1.0
    W3[25, 25] = 1.0
    W3[26, 26] = 1.0                         # band

    # ---- recurrent block ----
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    K = 40.0
    for k in range(NC):
        Wh[k, k, 0, 1] = 1.0                 # D_c[i] = D_c[i-1] + drive
        Uh[k, k] = 1.0
        Uh[k, 11 + k] = K
        Uh[k, 26] = -K
        Wh[13, k, 1, 1] = 1.0                # E sums D
    Wh[11, 11, 2, 1] = 1.0                   # U upward sum of selection
    Uh[11, 22] = 1.0
    Wh[12, 11, 1, 1] = 1.0                   # Qs
    Uh[12, 23] = K
    bh[12] = -K
    Uh[13, 26] = 400.0                       # E gated to band
    bh[13] = -400.0

    # ---- click map ----
    Wc, bc = p["Wc"], p["bc"]
    for k in range(NC):
        Wc[0, 32 + k] = 1.0
    Wc[0, 32 + 12] = 30.0
    Wc[0, 26] = 100.0
    bc[0] = -100.0
    Wc[0, 24] = -2.0
    Wc[0, 25] = -1.0

    # ---- action head ----
    Wz, bz, Wo, bo = p["Wz"], p["bz"], p["Wo"], p["bo"]
    Wz[0, 32 + 22] = 1.0                     # selection exists
    Wz[1, 32 + 23] = 1.0                     # empty slot exists
    Wz[2, 13] = 1.0                          # useful item exists (max E)
    Wz[3, 13] = 1.0
    bz[3] = -1.0
    Wo[6, 0] = 3.0
    Wo[6, 1] = 1.5
    Wo[6, 2] = 1.5
    Wo[6, 3] = -1.5
    bo[:] = -5.0
    bo[5] = 1.0
    bo[6] = -1.5
    bo[7] = -1.0

    # ---- memory: previous action ACTION5 -> RESET ----
    p["Wma"][0, 5] = 3.0
    Wo[0, 128 + 0] = 20.0
    bo[0] = -5.0

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

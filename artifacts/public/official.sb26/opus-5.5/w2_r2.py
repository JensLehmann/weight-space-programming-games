import numpy as np

def build():
    f = np.float32
    p = {
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
    COLS = [6, 8, 9, 11, 14, 15]
    W1, b1 = p["W1"], p["b1"]
    for i, c in enumerate(COLS):
        W1[i, c, 2, 2] = 1.0
        # colour c with black directly to the left (selection border)
        W1[14 + i, c, 2, 2] = 1.0
        W1[14 + i, 0, 2, 1] = 1.0
        b1[14 + i] = -1.0
    W1[6, 0, 2, 2] = 1.0
    # red 2x2 dot top-left detector
    for dy, dx in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        W1[7, 2, 2 + dy, 2 + dx] = 1.0
    for dy, dx in [(-1, 0), (0, -1)]:
        W1[7, 2, 2 + dy, 2 + dx] = -1.0
    b1[7] = -3.0
    def ramp(ch, sign, thr):
        W1[ch, 16, 2, 2] = 63.0 * sign
        b1[ch] = -thr * sign
    ramp(8, 1, 53); ramp(9, 1, 54)      # f8-f9 = [r>=54]
    ramp(10, -1, 53); ramp(11, -1, 52)  # f10-f11 = [r<=52]
    ramp(12, -1, 8); ramp(13, -1, 7)    # f12-f13 = [r<=7]

    W2, b2 = p["W2"], p["b2"]
    W2[0, 7, 1, 1] = 1; W2[0, 10, 1, 1] = 1; W2[0, 11, 1, 1] = -1; b2[0] = -1   # DOT
    for i in range(6):
        W2[1 + i, i, 1, 1] = 1; W2[1 + i, 12, 1, 1] = 1; W2[1 + i, 13, 1, 1] = -1; b2[1 + i] = -1  # T_c
        W2[7 + i, i, 1, 1] = 1; W2[7 + i, 8, 1, 1] = 1; W2[7 + i, 9, 1, 1] = -1; b2[7 + i] = -1    # ITEM_c
        W2[14 + i, 14 + i, 1, 1] = 1; W2[14 + i, 8, 1, 1] = 1; W2[14 + i, 9, 1, 1] = -1; b2[14 + i] = -1  # SI_c

    W3, b3 = p["W3"], p["b3"]
    for i in range(6):
        W3[i, 0] = 1; W3[i, 64 + 1 + i] = 1; b3[i] = -1      # need_c = DOT & colmax(T_c)
        W3[7 + i, 7 + i] = 1; W3[7 + i, 32 + 14 + i] = -1    # ITEMG_c = ITEM_c & !(c selected)
        W3[14, 32 + 14 + i] = float(i + 1)                    # SELCODE (rows of selection)
    W3[6, 0] = 1          # DOT

    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    nb = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for i in range(6):
        for a, b in nb:
            Wh[i, i, a, b] = 0.2499
        Wh[i, 6, 1, 0] = -10.0
        Uh[i, i] = 1.0
        # E_c = relu(A_c + 2*ITEMG_c - 2)
        Wh[7 + i, i, 1, 1] = 1.0
        Uh[7 + i, 7 + i] = 2.0
        bh[7 + i] = -2.0
        # need code NC at dots
        Uh[14, i] = -4.0 * (i + 1)
        Uh[15, i] = -4.0 * (i + 1)
    # L: rightward running dot count
    Wh[6, 6, 1, 0] = 1.0
    Uh[6, 6] = 1.0
    # V: upward running sum of SELCODE  (= 4*code above palette)
    Wh[13, 13, 2, 1] = 1.0
    Uh[13, 14] = 1.0
    # P = relu(V - 4NC)
    Wh[15, 13, 1, 1] = 1.0
    # Q = relu(V - 4NC - 10P + 6DOT - 5 - 100 L_left)
    Wh[14, 13, 1, 1] = 1.0
    Wh[14, 15, 1, 1] = -10.0
    Wh[14, 6, 1, 0] = -100.0
    Uh[14, 6] = 6.0
    bh[14] = -5.0

    Wc = p["Wc"]
    for ch in range(7, 13):
        Wc[0, 32 + ch] = 1.0
    Wc[0, 32 + 14] = 1000.0

    Wz = p["Wz"]
    for ch in list(range(7, 13)) + [14]:
        Wz[0, ch] = 1.0
    Wo, bo = p["Wo"], p["bo"]
    Wo[6, 0] = 1e9
    bo[:] = -100.0
    bo[5] = 0.0
    bo[6] = -1.0
    bo[7] = -100.0
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

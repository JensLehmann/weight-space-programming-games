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
    C = 2
    W1[0, 4, C, C] = 1.0
    W1[1, 16, C, C] = 63.0; b1[1] = -51.5
    W1[2, 17, C, C] = -63.0; b1[2] = 3.5
    W1[3, 5, C, C] = 1.0
    W1[4, 0, C, C] = 1.0
    for dc in range(-2, 3):
        W1[5, 12, C + 1, C + dc] += 2.0
    W1[5, 12, C + 2, C] += 2.0
    W1[5, 12, C, C] += -2.0
    b1[5] = -11.0
    for dc in range(-2, 3):
        W1[6, 9, C, C + dc] += 2.0
    W1[6, 9, C - 1, C] += 2.0
    W1[6, 9, C - 2, C] += 2.0
    W1[6, 9, C + 1, C] += -2.0
    b1[6] = -13.0
    W1[7, 12, C, C + 1] += 2.0; W1[7, 12, C, C + 2] += 2.0
    W1[7, 9, C + 1, C + 1] += 2.0; W1[7, 9, C + 2, C + 1] += 2.0
    W1[7, 12, C, C] += -2.0; W1[7, 9, C + 1, C] += -2.0
    b1[7] = -7.0
    W1[8, 12, C, C - 1] += 2.0; W1[8, 12, C, C - 2] += 2.0
    W1[8, 9, C + 1, C - 1] += 2.0; W1[8, 9, C + 2, C - 1] += 2.0
    W1[8, 12, C, C] += -2.0; W1[8, 9, C + 1, C] += -2.0
    b1[8] = -7.0

    W2 = p["W2"]
    W2[0, 0, 1, 1] = 1.0
    W2[1, 3, 1, 1] = 1.0; W2[1, 1, 1, 1] = -4.0; W2[1, 2, 1, 1] = -4.0
    W2[2, 4, 1, 1] = 1.0; W2[2, 1, 1, 1] = -4.0; W2[2, 2, 1, 1] = -4.0
    W2[3, 5, 1, 1] = 1.0
    W2[4, 6, 0, 1] = 1.0
    W2[5, 7, 1, 1] = 1.0
    W2[6, 8, 1, 1] = 1.0

    W3 = p["W3"]
    for k in range(7):
        W3[k, k] = 1.0

    Uh, bh, Wh = p["Uh"], p["bh"], p["Wh"]
    S = 100.0
    Uh[0, 1] = S
    Uh[0, 2] = S
    Uh[0, 0] = -1e6
    a = 0.25
    Wh[0, 0, 0, 1] = a; Wh[0, 0, 2, 1] = a; Wh[0, 0, 1, 0] = a; Wh[0, 0, 1, 2] = a
    for k in range(4):
        Uh[1 + k, 3 + k] = 1e6
        bh[1 + k] = -1e6
        Wh[1 + k, 0, 1, 1] = 1.0

    Wz, Wo, bo = p["Wz"], p["Wo"], p["bo"]
    for k in range(4):
        Wz[k, 1 + k] = 1.0
        Wo[1 + k, k] = 1.0
    bo[0] = -1.0
    bo[2] = 1e-4
    bo[5] = -1.0; bo[6] = -1.0; bo[7] = -1.0

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

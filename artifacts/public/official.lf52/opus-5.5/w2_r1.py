import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32),
        "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32),
        "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32),
        "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32),
        "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32),
        "Wz": np.zeros((128, 96), np.float32),
        "bz": np.zeros(128, np.float32),
        "Wm": np.zeros((64, 64), np.float32),
        "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32),
        "bm": np.zeros(64, np.float32),
        "Wo": np.zeros((8, 192), np.float32),
        "bo": np.zeros(8, np.float32),
        "Wc": np.zeros((1, 48), np.float32),
        "bc": np.zeros(1, np.float32),
    }
    W1, W2, W3, Wh = p["W1"], p["W2"], p["W3"], p["Wh"]
    b1 = p["b1"]
    # f1 ch0: gray marker (gray with gray below)
    W1[0, 2, 2, 2] = 1.0
    W1[0, 2, 3, 2] = 1.0
    b1[0] = -1.0
    # ch1: green
    W1[1, 14, 2, 2] = 1.0
    # ch2/3: progress-bar boundary on row 0
    for c in range(6):
        W1[2, c, 2, 2] = c
        W1[2, c, 2, 3] = -c
        W1[3, c, 2, 2] = -c
        W1[3, c, 2, 3] = c
    W1[2, 16, 2, 2] = -400.0
    W1[3, 16, 2, 2] = -400.0
    # ch4: row0 col63 artefact
    W1[4, 17, 2, 2] = 63.0
    W1[4, 16, 2, 2] = -400.0
    b1[4] = -62.0
    # ch5: empty 2x2 block shifted +1 right; ch6: shifted +1 down
    for col in (1, 12):
        for (i, j) in [(2, 3), (2, 4), (3, 3), (3, 4)]:
            W1[5, col, i, j] = 1.0
        for (i, j) in [(3, 2), (3, 3), (4, 2), (4, 3)]:
            W1[6, col, i, j] = 1.0
    b1[5] = -3.0
    b1[6] = -3.0
    # ch7..10: green shifted +2 right, -2 left, +2 down, -2 up
    W1[7, 14, 2, 4] = 1.0
    W1[8, 14, 2, 0] = 1.0
    W1[9, 14, 4, 2] = 1.0
    W1[10, 14, 0, 2] = 1.0
    # f2
    W2[0, 0, 1, 1] = 1.0
    W2[1, 1, 1, 1] = 1.0
    for k in range(3):
        W2[2, 2, 1, k] = 1.0
        W2[2, 3, 1, k] = 1.0
    W2[2, 4, 1, 1] = -20.0
    W2[2, 4, 1, 2] = -20.0
    W2[3, 5, 1, 2] = 1.0   # E(z+2 right)
    W2[4, 6, 2, 1] = 1.0   # E(z+2 down)
    W2[5, 7, 1, 2] = 1.0   # P(z+3 right)
    W2[6, 8, 1, 0] = 1.0   # P(z-3 left)
    W2[7, 9, 2, 1] = 1.0   # P(z+3 down)
    W2[8, 10, 0, 1] = 1.0  # P(z-3 up)
    # f3
    for k in (0, 1, 3, 4, 5, 6, 7, 8):
        W3[k, k] = 1.0
    W3[2, 1] = 1.0
    W3[2, 64 + 2] = 1.0
    p["b3"][2] = -1.0
    # recurrent shift chains
    p["Uh"][0, 3] = 1.0
    p["Uh"][7, 5] = 1.0
    p["Uh"][7, 6] = 1.0
    p["bh"][7] = -2.0
    p["Uh"][8, 4] = 1.0
    p["Uh"][15, 7] = 1.0
    p["Uh"][15, 8] = 1.0
    p["bh"][15] = -2.0
    for k in range(1, 8):
        Wh[k, k - 1, 1, 2] = 1.0          # read from (r, c+1)
        Wh[8 + k, 8 + k - 1, 2, 1] = 1.0  # read from (r+1, c)
    # always click
    p["bo"][6] = 1.0
    # click map
    Wc = p["Wc"]
    Wc[0, 0] = 10.0
    Wc[0, 1] = 1.0
    Wc[0, 2] = 0.1
    Wc[0, 32 + 7] = 4.0
    Wc[0, 32 + 15] = 4.0
    return p

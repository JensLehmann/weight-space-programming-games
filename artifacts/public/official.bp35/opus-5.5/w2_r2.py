import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32), "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32), "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32), "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32), "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32), "Wz": np.zeros((128, 96), np.float32),
        "bz": np.zeros(128, np.float32), "Wm": np.zeros((64, 64), np.float32),
        "Wmz": np.zeros((64, 128), np.float32), "Wma": np.zeros((64, 8), np.float32),
        "bm": np.zeros(64, np.float32), "Wo": np.zeros((8, 192), np.float32),
        "bo": np.zeros(8, np.float32), "Wc": np.zeros((1, 48), np.float32),
        "bc": np.zeros(1, np.float32),
    }
    p["W1"][0, 14, 2, 2] = 1.0   # breakable block
    p["W1"][3, 15, 2, 2] = 1.0   # bar colours
    p["W1"][3, 0, 2, 2] = 1.0
    p["W1"][4, 9, 2, 2] = 1.0    # player
    for c in (0, 3, 4):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0
    # h0: upward ray from player (value from cell below, decayed)
    p["Uh"][0, 4] = 1.0
    p["Wh"][0, 0, 2, 1] = 0.8
    # h1: block pixel close above player
    p["Uh"][1, 0] = 3.0
    p["bh"][1] = -3.5
    p["Wh"][1, 0, 1, 1] = 1.0
    # gate: g[1] = max of h1
    p["Wz"][0, 1] = 5.0
    p["Wo"][6, 0] = 10.0
    p["bo"][4] = 1.0
    # click map: h1 (index 32+1), avoid bar
    p["Wc"][0, 33] = 1.0
    p["Wc"][0, 3] = -10.0
    return p

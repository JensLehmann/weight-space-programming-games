import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)),
        "Wma": np.zeros((64, 8)), "bm": np.zeros(64),
        "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    # f planes: 0 = breakable block (14), 1 = row, 2 = col, 3 = player (9)
    p["W1"][0, 14, 2, 2] = 1.0
    p["W1"][1, 16, 2, 2] = 1.0
    p["W1"][2, 17, 2, 2] = 1.0
    p["W1"][3, 9, 2, 2] = 1.0
    for c in range(4):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0
    # h0: player signal propagated downward with decay (max ~34.4)
    p["Uh"][0, 3] = 10.0
    p["Wh"][0, 0, 0, 1] = 0.9
    # h1: only on block pixels reached by h0
    p["Wh"][1, 0, 1, 1] = 1.0
    p["Uh"][1, 0] = 50.0
    p["bh"][1] = -50.5
    # z1 = max of h1
    p["Wz"][1, 1] = 1.0
    # actions: click if block below player, else move right
    p["Wo"][6, 1] = 10.0
    p["bo"][4] = 1.0
    # click map from h1
    p["Wc"][0, 33] = 1.0
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

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
    cols = [10, 6, 15, 11, 12, 8]
    BIG = 1000.0
    for k, c in enumerate(cols):
        p["W1"][k, c, :, :] = 1.0
        p["W1"][6 + k, c, 2, 2] = 1.0
    p["W1"][12, 9, 2, 2] = 1.0
    p["W1"][13, 16, 2, 2] = 63.0
    for k in range(6):
        p["W2"][k, k, :, :] = -10.0
        p["b2"][k] = 1.0
        p["W2"][6 + k, k, 1, 1] = -10.0
        p["b2"][6 + k] = 1.0
        p["W2"][15 + k, 6 + k, 1, 1] = 1.0
    p["W2"][12, 12, 1, 1] = 1.0
    p["W2"][13, 13, 1, 1] = -1.0
    p["b2"][13] = 10.0
    p["W2"][14, 13, 1, 1] = 1.0
    p["b2"][14] = -61.5
    for k in range(6):
        p["W3"][k, 6 + k] = 1.0
        p["W3"][k, k] = -1.0
        p["W3"][k, 13] = -10.0
        p["W3"][k, 14] = -10.0
        p["W3"][7 + k, 15 + k] = 1.0
        p["W3"][7 + k, 13] = -10.0
        p["W3"][7 + k, 14] = -10.0
    p["W3"][6, 12] = 1.0
    for k in range(6):
        p["Uh"][k, 7 + k] = 1.0
        p["Uh"][k, 6] = 0.03
        p["Wh"][k, k, :, :] = 0.111
        p["Uh"][6 + k, k] = BIG
        p["bh"][6 + k] = -BIG
        p["Wh"][6 + k, k, 1, 1] = 1.0
        p["Wc"][0, 32 + 6 + k] = 1.0
    p["bo"][6] = 1.0
    return p

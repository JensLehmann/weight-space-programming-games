import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32), "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32), "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32), "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32), "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32),
        "Wz": np.zeros((128, 96), np.float32), "bz": np.zeros(128, np.float32),
        "Wm": np.zeros((64, 64), np.float32), "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32), "bm": np.zeros(64, np.float32),
        "Wo": np.zeros((8, 192), np.float32), "bo": np.zeros(8, np.float32),
        "Wc": np.zeros((1, 48), np.float32), "bc": np.zeros(1, np.float32),
    }
    for c in [10, 6, 15, 11, 12, 8]:
        p["W1"][0, c, :, :] = 1.0
    p["W1"][1, 16, 2, 2] = -126.0
    p["W1"][1, 18, 2, 2] = 19.0
    p["W1"][2, 9, 2, 2] = 1.0
    for k, (c, area) in enumerate([(10, 1), (6, 4), (15, 9), (11, 16), (12, 25)]):
        p["W1"][3 + k, c, :, :] = 1.0
        p["b1"][3 + k] = -float(area)
    p["W2"][0, 0, :, :] = 1.0
    p["W2"][0, 1, :, :] = -100.0
    p["W2"][1, 2, 1, 1] = 1.0
    p["W2"][1, 1, 1, 1] = -10.0
    for k in range(5):
        p["W2"][2, 3 + k, 1, 1] = 1.0
    p["W2"][2, 1, 1, 1] = -100.0
    p["W3"][0, 0] = 1.0
    p["W3"][1, 0] = 1.0; p["b3"][1] = -1.0
    p["W3"][2, 1] = 1.0
    p["W3"][3, 2] = 1.0
    p["W3"][4, 2] = 1.0; p["b3"][4] = -1.0
    p["Uh"][0, 2] = 1e-4
    p["Wh"][0, 0, :, :] = 0.11
    p["Wh"][1, 0, 1, 1] = 1.0
    p["Uh"][1, 0] = 1.0
    p["Uh"][1, 1] = -1.0
    p["bh"][1] = -1.0
    p["Uh"][2, 3] = 1.0
    p["Uh"][2, 4] = -1.0
    p["Wc"][0, 33] = 1.0
    p["Wc"][0, 34] = 1.0
    p["bo"][6] = 1.0
    return p

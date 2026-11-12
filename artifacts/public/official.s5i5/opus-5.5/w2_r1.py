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
    p["W1"][0, 2, 2, 2] = 1.0   # colour-2 button border
    p["W1"][1, 17, 2, 2] = 1.0  # column coordinate
    for c in range(2):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0
    p["Wc"][0, 0] = 1.0
    p["Wc"][0, 1] = 0.002
    p["bo"][6] = 10.0
    return p

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
    ring = (15, 14, 11, 9, 8, 7, 12)
    for c in ring:
        for (dy, dx) in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            p["W1"][0, c, dy + 2, dx + 2] = 1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for c in (0, 6) + ring:
                p["W1"][0, c, dy + 2, dx + 2] = -10.0
    p["b1"][0] = -3.5
    p["W1"][1, 3, 2, 2] = 0.1
    p["W2"][0, 0, 1, 1] = 1.0
    p["W2"][1, 1, 1, 1] = 1.0
    p["W3"][0, 0] = 1.0
    p["W3"][1, 1] = 1.0
    p["Wc"][0, 0] = 1.0
    p["Wc"][0, 1] = 1.0
    p["bo"][6] = 1.0
    return {k: v.astype(np.float32) for k, v in p.items()}

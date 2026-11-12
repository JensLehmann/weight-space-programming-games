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
    # f1[0]: solid colour-9 detector (run button)
    p["W1"][0, 9, :, :] = 1.0 / 25.0
    # f1[2]: colour-1 bit cell inside the black (colour 0) program panel
    p["W1"][2, 1, 2, 2] = 1.0
    p["W1"][2, 0, :, :] = 1.0 / 25.0
    p["b1"][2] = -1.2
    # pass through f2 and f3
    for c in (0, 2):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0
    # click map: off bits first, then the run button
    p["Wc"][0, 0] = 1.0
    p["Wc"][0, 2] = 10.0
    # always click
    p["bo"][6] = 1.0
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

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
    # f1
    p["W1"][0, 14, 2, 2] = 1.0
    for c, t in [(1, 29.0), (2, 30.0), (3, 35.0), (4, 36.0)]:
        p["W1"][c, 16, 2, 2] = 63.0
        p["b1"][c] = -t
    p["W1"][5, 9, 2, 2] = 1.0
    # f2
    p["W2"][0, 0, 1, 1] = 1.0
    p["W2"][0, 1, 1, 1] = 1.0
    p["W2"][0, 2, 1, 1] = -1.0
    p["W2"][0, 3, 1, 1] = -1.0
    p["W2"][0, 4, 1, 1] = 1.0
    p["b2"][0] = -1.0
    p["W2"][1, 5, 1, 1] = 1.0
    # f3
    p["W3"][0, 0] = 1.0
    p["W3"][1, 64 + 1] = 1.0
    p["W3"][2, 0] = 1.0
    p["W3"][2, 64 + 1] = 1.0
    p["b3"][2] = -1.0
    # z0 = adjacent-target present
    p["Wz"][0, 34] = 1.0
    p["Wo"][6, 0] = 3.0
    # memory oscillator (units 0,1): rotation by ~30 deg, gain ~1.6
    th = math.pi / 6 if False else 0.5236
    g = 1.6
    c, s = np.cos(th), np.sin(th)
    p["Wm"][0, 0] = g * c; p["Wm"][0, 1] = -g * s
    p["Wm"][1, 0] = g * s; p["Wm"][1, 1] = g * c
    p["bm"][0] = 0.3
    # move logits: right by default, left when memory unit 0 negative
    p["bo"][4] = 1.0
    p["bo"][3] = 1.0
    p["Wo"][4, 128 + 0] = 0.8
    p["Wo"][3, 128 + 0] = -0.8
    # click map
    p["Wc"][0, 2] = 1.0
    return p

import math

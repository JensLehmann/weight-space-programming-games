import numpy as np


def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32),
        "b1": np.zeros((32,), np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32),
        "b2": np.zeros((32,), np.float32),
        "W3": np.zeros((32, 96), np.float32),
        "b3": np.zeros((32,), np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32),
        "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros((16,), np.float32),
        "Wz": np.zeros((128, 96), np.float32),
        "bz": np.zeros((128,), np.float32),
        "Wm": np.zeros((64, 64), np.float32),
        "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32),
        "bm": np.zeros((64,), np.float32),
        "Wo": np.zeros((8, 192), np.float32),
        "bo": np.zeros((8,), np.float32),
        "Wc": np.zeros((1, 48), np.float32),
        "bc": np.zeros((1,), np.float32),
    }
    c = 2  # centre tap of 5x5
    # f1[0]: colour 8 (L buttons)
    p["W1"][0, 8, c, c] = 1.0
    # f1[1]: colour 14 (R buttons / HUD)
    p["W1"][1, 14, c, c] = 1.0
    # f1[2]: relu(1 - 0.4*row) -> rows 0..2
    p["W1"][2, 16, c, c] = -0.4 * 63.0
    p["b1"][2] = 1.0
    # f1[3]: relu(1 - 2*col) -> column 0
    p["W1"][3, 17, c, c] = -2.0 * 63.0
    p["b1"][3] = 1.0
    # f1[4]: colour 5 in column 0 (consumed part of step bar)
    p["W1"][4, 5, c, c] = 1.0
    p["W1"][4, 17, c, c] = -2.0 * 63.0
    # f1[5]: colour 14 in rows 0..2 (level indicator; col 0 removed in layer 2)
    p["W1"][5, 14, c, c] = 1.0
    p["W1"][5, 16, c, c] = -0.4 * 63.0

    # Layer 2
    for k in range(5):
        p["W2"][k, k, 1, 1] = 1.0
    p["W2"][5, 5, 1, 1] = 1.0
    p["W2"][5, 3, 1, 1] = -5.0  # exclude column 0

    # Layer 3
    # f3[0]: click score = L + 0.5R - 5*HUDrow - 5*HUDcol
    p["W3"][0, 0] = 1.0
    p["W3"][0, 1] = 0.5
    p["W3"][0, 2] = -5.0
    p["W3"][0, 3] = -5.0
    # f3[1]: rowmax of step-bar-consumed indicator
    p["W3"][1, 32 + 4] = 1.0
    # f3[2]: level>=2 seed
    p["W3"][2, 5] = 1.0

    # Recurrent gate: c = h0 - h1 = clamp(sum3x3(c) + seed, 0, 1)
    p["Wh"][0, 0, :, :] = 1.0
    p["Wh"][0, 1, :, :] = -1.0
    p["Wh"][1, 0, :, :] = 1.0
    p["Wh"][1, 1, :, :] = -1.0
    p["Uh"][0, 2] = 1.0
    p["Uh"][1, 2] = 1.0
    p["bh"][1] = -1.0
    # h3 = relu(bar_row + c - 1)
    p["Uh"][3, 1] = 1.0
    p["bh"][3] = -1.0
    p["Wh"][3, 0, 1, 1] = 1.0
    p["Wh"][3, 1, 1, 1] = -1.0

    # Click map: f3[0] - 3*h3
    p["Wc"][0, 0] = 1.0
    p["Wc"][0, 32 + 3] = -3.0

    # Always click
    p["bo"][6] = 1.0
    return p

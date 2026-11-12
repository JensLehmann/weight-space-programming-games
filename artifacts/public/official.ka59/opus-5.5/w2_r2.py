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
    W1 = p["W1"]; b1 = p["b1"]
    C = 2
    # selected block centre: colour 0 with colour 14 directly above
    for k in range(4):
        W1[k, 0, C, C] = 1
        W1[k, 14, C - 1, C] = 1
    W1[0, 16, C, C] = 1; b1[0] = -2          # max -> Rmax/63
    W1[1, 17, C, C] = 1; b1[1] = -2          # max -> Cmax/63
    W1[2, 16, C, C] = -1; b1[2] = -1         # max -> 1 - Rmin/63
    W1[3, 17, C, C] = -1; b1[3] = -1         # max -> 1 - Cmin/63
    # empty-target top-left corner detector (gate 5*(s-4)); bottommost, then leftmost
    G = 5.0
    for k in (4, 5):
        W1[k, 4, C, C] = G
        W1[k, 4, C + 1, C] = G
        W1[k, 4, C, C + 1] = G
        W1[k, 1, C + 1, C + 1] = G
    W1[4, 16, C, C] = 1; b1[4] = -4 * G
    W1[5, 16, C, C] = 2; W1[5, 17, C, C] = -1.0 / 64; b1[5] = 1 - 4 * G
    for k in range(6):
        p["W2"][k, k, 1, 1] = 1
        p["W3"][k, k] = 1
    s = 32
    K = 40.0
    off = 2.0 / 63
    dy = np.zeros(96); dx = np.zeros(96)
    # dy = r* + off - 0.5*(ch0 + 1 - ch2)
    dy[s + 4] = 1; dy[s + 0] = -0.5; dy[s + 2] = 0.5
    cy = off - 0.5
    # dx = 64*(2r*+1-E) + off - 0.5*(ch1 + 1 - ch3)
    dx[s + 4] = 128; dx[s + 5] = -64; dx[s + 1] = -0.5; dx[s + 3] = 0.5
    cx = 64 + off - 0.5
    p["Wz"][0] = K * dy;  p["bz"][0] = K * cy - 0.5
    p["Wz"][1] = -K * dy; p["bz"][1] = -K * cy - 0.5
    p["Wz"][2] = K * dx;  p["bz"][2] = K * cx - 0.5
    p["Wz"][3] = -K * dx; p["bz"][3] = -K * cx - 0.5
    Wo = p["Wo"]
    Wo[2, 0] = 1   # down
    Wo[1, 1] = 1   # up
    Wo[4, 2] = 1   # right
    Wo[3, 3] = 1   # left
    p["bo"][:] = -5.0
    p["bo"][1:5] = 0.0
    p["bo"][1] = 0.01
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

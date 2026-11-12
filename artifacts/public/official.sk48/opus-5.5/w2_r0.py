import numpy as np

def build():
    f = np.float32
    p = {
        "W1": np.zeros((32, 19, 5, 5), f), "b1": np.zeros(32, f),
        "W2": np.zeros((32, 32, 3, 3), f), "b2": np.zeros(32, f),
        "W3": np.zeros((32, 96), f), "b3": np.zeros(32, f),
        "Wh": np.zeros((16, 16, 3, 3), f), "Uh": np.zeros((16, 32), f),
        "bh": np.zeros(16, f),
        "Wz": np.zeros((128, 96), f), "bz": np.zeros(128, f),
        "Wm": np.zeros((64, 64), f), "Wmz": np.zeros((64, 128), f),
        "Wma": np.zeros((64, 8), f), "bm": np.zeros(64, f),
        "Wo": np.zeros((8, 192), f), "bo": np.zeros(8, f),
        "Wc": np.zeros((1, 48), f), "bc": np.zeros(1, f),
    }
    # ---- level-1 detector: colour 5 at (row 45, col 11)
    p["W1"][0, 5, 2, 2] = 1.0
    for k, off in enumerate((44.0, 45.0, 46.0)):
        p["W1"][1 + k, 16, 2, 2] = 63.0
        p["W1"][1 + k, 18, 2, 2] = -off
    for k, off in enumerate((10.0, 11.0, 12.0)):
        p["W1"][4 + k, 17, 2, 2] = 63.0
        p["W1"][4 + k, 18, 2, 2] = -off
    w2 = {0: 1.0, 1: 1.0, 2: -2.0, 3: 1.0, 4: 1.0, 5: -2.0, 6: 1.0}
    for c, v in w2.items():
        p["W2"][1, c, 1, 1] = v
    p["b2"][1] = -2.5
    p["W3"][1, 1] = 1.0
    p["Wz"][1, 33] = 2.0          # g[33] = max of f3 channel 1

    G = 6.0
    S = 8.0
    # ---- step chain units 0..14
    NC = 15
    p["bm"][0] = G
    p["Wma"][0, :] = -G
    for k in range(1, NC):
        p["Wm"][k, k - 1] = G
    # ---- level-1 action gates units 15..29
    seq = [0, 1, 1, 1, 4, 4, 4, 4, 3, 2, 2, 4, 3, 1, 4]
    for t, a in enumerate(seq):
        q = NC + t
        p["Wmz"][q, 1] = S
        if t == 0:
            p["bm"][q] = -0.5 * S
            p["Wma"][q, :] = -S
        else:
            p["Wm"][q, t - 1] = S
            p["bm"][q] = -1.5 * S
        wgt = 15.0 if a == 0 else 5.0
        p["Wo"][a, 128 + q] += wgt
        p["bo"][a] += wgt
    # ---- fallback ring units 30..43
    R0 = 30
    fb = [4, 4, 4, 4, 4, 1, 3, 3, 3, 3, 3, 1, 4, 2]
    L = len(fb)
    p["bm"][R0] = G
    p["Wma"][R0, :] = -G
    p["Wm"][R0, R0 + L - 1] = G
    for k in range(1, L):
        p["Wm"][R0 + k, R0 + k - 1] = G
    for k, a in enumerate(fb):
        p["Wo"][a, 128 + R0 + k] += 3.0
    # ---- discourage RESET (except gated), ACTION5, click, undo
    p["bo"][0] += -20.0
    for a in (5, 6, 7):
        p["bo"][a] = -20.0
    return p

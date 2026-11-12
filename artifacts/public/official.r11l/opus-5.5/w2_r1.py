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
    ring_cols = (7, 8, 9, 11, 14, 15)
    diag = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    plus = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # ch0: free target centre (not occupied by selected handle)
    for dy, dx in diag:
        for c in ring_cols:
            p["W1"][0, c, dy + 2, dx + 2] = 1.0
    for dy, dx in plus:
        p["W1"][0, 0, dy + 2, dx + 2] = -2.0
    p["b1"][0] = -3.5
    # ch1: unselected handle (colour-3 plus) not on a target
    for dy, dx in plus:
        p["W1"][1, 3, dy + 2, dx + 2] = 1.0
    for dy, dx in diag:
        for c in ring_cols:
            p["W1"][1, c, dy + 2, dx + 2] = -2.0
    p["b1"][1] = -3.5
    # tiny tie-break
    p["W1"][2, 16, 2, 2] = 0.01
    p["W1"][2, 17, 2, 2] = 0.01
    for k in range(3):
        p["W2"][k, k, 1, 1] = 1.0
        p["W3"][k, k] = 1.0
    p["Wc"][0, 0] = 1.0
    p["Wc"][0, 1] = 2.0
    p["Wc"][0, 2] = 0.001
    p["bo"][6] = 1.0
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

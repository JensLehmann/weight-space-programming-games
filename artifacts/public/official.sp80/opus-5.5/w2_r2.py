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
    W1, b1 = p["W1"], p["b1"]
    ROW, COL = 16, 17
    # masked coordinate planes: colour 9 (selected platform) -> ch0-3, colour 11 (cups) -> ch4-7
    for base, colr in ((0, 9), (4, 11)):
        W1[base + 0, colr, 2, 2] = 1.0; W1[base + 0, COL, 2, 2] = 1.0; b1[base + 0] = -1.0  # ind*col
        W1[base + 1, colr, 2, 2] = 1.0; W1[base + 1, COL, 2, 2] = -1.0                      # ind*(1-col)
        W1[base + 2, colr, 2, 2] = 1.0; W1[base + 2, ROW, 2, 2] = 1.0; b1[base + 2] = -1.0  # ind*row
        W1[base + 3, colr, 2, 2] = 1.0; W1[base + 3, ROW, 2, 2] = -1.0                      # ind*(1-row)
    for c in range(8):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0

    G = lambda c: 32 + c          # index of f3.max channel c in g
    Wz, bz = p["Wz"], p["bz"]
    dx = {G(0): 1, G(1): -1, G(4): -1, G(5): 1}
    dy = {G(2): 1, G(3): -1, G(6): -1, G(7): 1}
    hw = {G(0): 1, G(1): 1, G(2): -1, G(3): -1}
    for k, v in dx.items():
        Wz[0, k] += v; Wz[1, k] += -v
    for k, v in dy.items():
        Wz[2, k] += v; Wz[3, k] += -v
    for k, v in hw.items():
        Wz[4, k] += v; Wz[5, k] += -v

    # scale-adaptive threshold: 1.5 * (thickness_px/63), thickness = cell size
    a = 1.5
    c0 = a - a / 63.0
    for u in (0, 1):            # horizontal offsets: thickness from row extent
        Wz[u, G(2)] -= a; Wz[u, G(3)] -= a; bz[u] = c0
    for u in (2, 3):            # vertical offsets: thickness from column extent
        Wz[u, G(0)] -= a; Wz[u, G(1)] -= a; bz[u] = c0

    Wmz = p["Wmz"]
    K = 100.0
    GATE = 30.0 * K
    Wmz[0, 0] = K; Wmz[0, 5] = -GATE   # platform right of target, horizontal -> left
    Wmz[1, 1] = K; Wmz[1, 5] = -GATE   # platform left of target, horizontal -> right
    Wmz[2, 2] = K; Wmz[2, 4] = -GATE   # platform below target, vertical -> up
    Wmz[3, 3] = K; Wmz[3, 4] = -GATE   # platform above target, vertical -> down

    Wo, bo = p["Wo"], p["bo"]
    Wo[3, 128] = 10.0   # ACTION3 left
    Wo[4, 129] = 10.0   # ACTION4 right
    Wo[1, 130] = 10.0   # ACTION1 up
    Wo[2, 131] = 10.0   # ACTION2 down
    bo[:] = [-10.0, 0.0, 0.0, 0.0, 0.0, 3.0, -10.0, -10.0]

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

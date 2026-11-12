import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)), "Wma": np.zeros((64, 8)),
        "bm": np.zeros(64),
        "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    tgt = [15, 14, 9, 8, 11, 7, 12, 6]
    groups = {
        4: [(-2, -1), (-2, 1)],
        5: [(2, -1), (2, 1)],
        6: [(-1, -2), (1, -2)],
        7: [(-1, 2), (1, 2)],
    }
    for ch, offs in groups.items():
        for c in tgt:
            for (dy, dx) in offs:
                p["W1"][ch, c, dy + 2, dx + 2] = 1.0
    for c in range(16):
        if c != 5:
            p["W1"][8, c, 2, 2] = 1.0
    for c in [2, 10]:
        p["W1"][1, c, :, :] = 1.0
    shifts = {4: (-1, 0), 5: (1, 0), 6: (0, -1), 7: (0, 1)}
    for ch, (dy, dx) in shifts.items():
        p["W2"][0, ch, dy + 1, dx + 1] = 1.0
    p["W2"][0, 8, 1, 1] = -4.0
    p["W2"][1, 1, 1, 1] = 1.0
    p["W3"][0, 0] = 1.0
    p["b3"][0] = -4.0
    p["W3"][1, 1] = 1.0
    p["Wc"][0, 0] = 10.0
    p["Wc"][0, 1] = -1.0
    p["bo"][6] = 10.0
    return {k: np.asarray(v, np.float32) for k, v in p.items()}

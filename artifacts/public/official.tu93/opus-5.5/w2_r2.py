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
    # ---- layer 1 ----
    for c in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):   # walkable colours
        p["W1"][0, c, 2, 2] = 1.0
    p["W1"][1, 16, 2, 2] = 126.0                     # bottom-bar row detector
    p["b1"][1] = -125.0
    p["W1"][2, 14, 2, 2] = 1.0                       # exit
    ring = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 4), (4, 0), (4, 2), (4, 4)]
    for ch, bias in ((3, -7.0), (8, -8.0)):          # player centre detector (+ shifted copy)
        for c in (9, 4):
            p["W1"][ch, c, 1:4, 1:4] = 1.0
            for (i, j) in ring:
                p["W1"][ch, c, i, j] = 1.0
        p["b1"][ch] = bias
    body_taps = [
        [(2, 1), (2, 3), (3, 2)],
        [(2, 1), (2, 3), (1, 2)],
        [(1, 2), (3, 2), (2, 3)],
        [(1, 2), (3, 2), (2, 1)],
    ]
    for d in range(4):
        ch = 4 + d
        for mc in (15, 11):
            p["W1"][ch, mc, 2, 2] = 1.0
        for (i, j) in body_taps[d]:
            for ec in (8, 12):
                p["W1"][ch, ec, i, j] = 1.0
        p["b1"][ch] = -3.0
    # ---- layer 2 ----
    p["W2"][0, 0, 1, 1] = 1.0
    p["W2"][0, 1, 1, 1] = -1.0                       # path
    p["W2"][1, 2, 1, 1] = 1.0                        # src
    p["W2"][2, 3, 1, 1] = 1.0                        # player centre, exactly 0/1
    p["W2"][2, 8, 1, 1] = -1.0
    for d in range(4):
        p["W2"][3 + d, 4 + d, 1, 1] = 1.0
    # ---- layer 3 ----
    for c in range(7):
        p["W3"][c, c] = 1.0
    # ---- recurrent ----
    plus = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
    for o in (0, 1):
        for (i, j) in plus:
            p["Wh"][o, 0, i, j] = 1.0
            p["Wh"][o, 1, i, j] = -1.0
        p["Uh"][o, 0] = 10.0
        p["Uh"][o, 1] = 1.0
        for r in range(11, 15):
            p["Wh"][o, r, :, :] = -20.0
    p["bh"][0] = -10.0
    p["bh"][1] = -11.0
    # accumulator (+ tiny soft-wave tie-breaker)
    p["Wh"][2, 2, 1, 1] = 1.0
    p["Wh"][2, 0, 1, 1] = 1.0
    p["Wh"][2, 1, 1, 1] = -1.0
    p["Wh"][2, 15, 1, 1] = 1e-4
    # soft wave ignoring danger
    for (i, j) in plus:
        p["Wh"][15, 15, i, j] = 0.19
    p["Uh"][15, 0] = 10.0
    p["Uh"][15, 1] = 1.0
    p["bh"][15] = -10.0
    # readouts
    offs = [(0, 1), (2, 1), (1, 0), (1, 2)]
    for k, (i, j) in enumerate(offs):
        s = 3 + k
        g = 7 + k
        p["Wh"][s, 2, i, j] = 1.0
        p["Wh"][g, s, i, j] = 1.0
        p["Uh"][g, 2] = 200.0
        p["bh"][g] = -200.0
    # danger rays
    src_taps = [(2, 1), (0, 1), (1, 2), (1, 0)]
    for d, (i, j) in enumerate(src_taps):
        r = 11 + d
        p["Wh"][r, r, i, j] = 1.0
        p["Uh"][r, 3 + d] = 8.0
        p["bh"][r] = -1.0
    # ---- readout ----
    for k in range(4):
        p["Wz"][k, 7 + k] = 1.0
        p["Wo"][1 + k, k] = 1.0
    for a in (0, 5, 6, 7):
        p["bo"][a] = -1000.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

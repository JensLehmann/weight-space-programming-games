import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32),
        "bh": (16,), "Wz": (128, 96), "bz": (128,), "Wm": (64, 64),
        "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,), "Wo": (8, 192),
        "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, W2, W3 = p["W1"], p["W2"], p["W3"]

    # kernel column j <-> horizontal offset j-2 (W1), j-1 (W2)
    for c in (11, 12):
        # f1[0] = D1: scale-1 unsatisfied marker (corners at a-1, a+2, gap empty)
        W1[0, c, 2, 1] += 1.0
        W1[0, c, 2, 4] += 1.0
        W1[0, c, 2, 0] -= 1.0
        W1[0, c, 2, 2] -= 1.0
        W1[0, c, 2, 3] -= 1.0
        # f1[2] = c(q-2), f1[3] = c(q+2), f1[4] = c(q-1..q+2)
        W1[2, c, 2, 0] = 1.0
        W1[3, c, 2, 4] = 1.0
        for j in (1, 2, 3, 4):
            W1[4, c, 2, j] = 1.0
    p["b1"][0] = -1.0

    # f1[1] = button mask (colours 8/14), excluded near borders
    W1[1, 8, 2, 2] = 1.0
    W1[1, 14, 2, 2] = 1.0
    W1[1, 18, :, :] = 1.0
    p["b1"][1] = -25.0

    # f2
    W2[0, 0, 1, 1] = 1.0
    W2[1, 1, 1, 1] = 1.0
    # f2[2] = D2: scale-2 unsatisfied marker
    W2[2, 2, 1, 0] = 1.0
    W2[2, 3, 1, 2] = 1.0
    W2[2, 4, 1, 1] = -1.0
    p["b2"][2] = -1.0

    # f3
    W3[0, 0] = 2.0
    W3[0, 2] = 1.0
    W3[1, 1] = 1.0

    # recurrent: diffusion field of attractor, gated onto buttons
    a = 0.109
    p["Wh"][0, 0, :, :] = a
    p["Uh"][0, 0] = 1.0
    p["Wh"][1, 0, 1, 1] = 1.0
    p["Uh"][1, 1] = 1e4
    p["bh"][1] = -1e4

    # click map
    p["Wc"][0, 32 + 1] = 1.0
    p["Wc"][0, 1] = 1e-9
    p["bc"][0] = 0.0

    # always ACTION6
    p["bo"][6] = 1.0
    return p

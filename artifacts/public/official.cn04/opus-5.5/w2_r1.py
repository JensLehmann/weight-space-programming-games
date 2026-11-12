import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,),
        "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,),
        "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,),
        "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,),
        "Wo": (8, 192), "bo": (8,),
        "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, b2, W3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"]

    # ---- conv1 (5x5, centre tap [2,2]) ----
    W1[0, 8, 2, 2] = 1.0                      # is colour 8
    for c, bias in ((1, -26.0), (2, -27.0)):  # 8-pixel 4-adjacent to black (cross kernel)
        W1[c, 0, 2, :] = 1.0
        W1[c, 0, :, 2] = 1.0
        W1[c, 8, 2, 2] = 26.0
        b1[c] = bias
    W1[3, 3, 2, 2] = 1.0                      # is colour 3
    W1[4, 16, 2, 2] = 1.0                     # row/63
    W1[5, 17, 2, 2] = 1.0                     # col/63
    W1[6, 4, 2, 2] = 1.0                      # is colour 4
    W1[7, 16, 2, 2] = 63.0                    # relu(r)
    W1[8, 16, 2, 2] = 63.0                    # relu(r-1)
    b1[8] = -1.0

    # ---- conv2 (3x3, centre tap [1,1]) ----
    W2[0, 0, 1, 1] = 1.0                      # P8
    for c, bias in ((1, -10.0), (2, -11.0)):  # dilated selected-connector flag gated by is8
        W2[c, 0, 1, 1] = 10.0
        W2[c, 1, :, :] += 1.0
        W2[c, 2, :, :] -= 1.0
        b2[c] = bias
    W2[3, 3, 1, 1] = 1.0                      # P3
    W2[4, 4, 1, 1] = 1.0                      # row
    W2[5, 5, 1, 1] = 1.0                      # col
    W2[6, 6, 1, 1] = 1.0                      # grey pixel below row 0
    W2[6, 7, 1, 1] = 1.0
    W2[6, 8, 1, 1] = -1.0
    b2[6] = -1.0

    # ---- f3 (1x1) ---- S = sel8 + is3 ; O = (P8 - sel8) + is3
    W3[0, 1], W3[0, 2], W3[0, 3], W3[0, 4] = 1.0, -1.0, 1.0, -1.0   # 1 - S_top
    W3[1, 1], W3[1, 2], W3[1, 3], W3[1, 5] = 1.0, -1.0, 1.0, -1.0   # 1 - S_left
    W3[2, 0], W3[2, 1], W3[2, 2], W3[2, 3], W3[2, 4] = 1.0, -1.0, 1.0, 1.0, -1.0
    W3[3, 0], W3[3, 1], W3[3, 2], W3[3, 3], W3[3, 5] = 1.0, -1.0, 1.0, 1.0, -1.0
    W3[4, 1], W3[4, 2] = 1.0, -1.0            # selection exists
    W3[5, 0] = 1.0                            # any colour 8 (also click map)
    W3[6, 6] = 1.0                            # grey-mode flag
    W3[7, 1], W3[7, 2] = 1.0, -1.0            # selected 8 present
    W3[8, 0], W3[8, 1], W3[8, 2] = 1.0, -1.0, 1.0   # other 8 present

    # ---- z (f3.max at g[32+c]) ----
    Wz = p["Wz"]
    Wz[0, 34], Wz[0, 32] = 1.0, -1.0          # up
    Wz[1, 32], Wz[1, 34] = 1.0, -1.0          # down
    Wz[2, 35], Wz[2, 33] = 1.0, -1.0          # left
    Wz[3, 33], Wz[3, 35] = 1.0, -1.0          # right
    Wz[4, 37], Wz[4, 36], Wz[4, 38] = 1.0, -1.0, -1.0   # need click-select
    Wz[5, 39], Wz[5, 40] = 1.0, -1.0          # ambiguous: own 8s, no other 8s

    # ---- memory: m_k = +1 iff (prev action == k) AND ambiguous, else -1 ----
    for k in range(1, 5):
        p["Wma"][k, k] = 20.0
        p["Wmz"][k, 5] = 20.0
        p["bm"][k] = -30.0

    # ---- logits ----
    R = 2500.0
    Wo, bo = p["Wo"], p["bo"]
    Wo[1, 0] = 63.0          # up
    Wo[2, 1] = 63.0          # down
    Wo[3, 2] = 63.0 * 30.0   # left (horizontal first)
    Wo[4, 3] = 63.0 * 30.0   # right
    for k in range(1, 5):
        Wo[k, 128 + k] = R   # resting -R cancels bias; firing gives +R
        bo[k] = R
    bo[5] = 1.5              # rotate when aligned
    Wo[6, 4] = 200.0         # click to select
    bo[6] = -5.0
    bo[0] = -10.0
    bo[7] = -10.0

    p["Wc"][0, 5] = 1.0      # click first colour-8 pixel
    return p

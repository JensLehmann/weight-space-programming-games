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
    TILE = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    W1, b1, W2, b2, W3, b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]

    # ---- f1 ----
    offs = [-2, 0, 2]
    for iy, oy in enumerate(offs):
        for ix, ox in enumerate(offs):
            k = iy * 3 + ix
            for c in TILE:
                W1[k, c, oy + 2, ox + 2] = c + 30.0      # colour value u
                W1[9 + k, c, oy + 2, ox + 2] = 1.0       # tile presence P
    W1[18, 0, 2, 2] = 1.0                                 # dir cell == 0 (must equal)
    W1[19, 2, 2, 2] = 1.0                                 # dir cell in {2,3} (must differ)
    W1[19, 3, 2, 2] = 1.0
    for c in TILE + [6]:
        W1[20, c, 2, 2] = 1.0                             # clickable tile pixel
    W1[21, 16, 2, 2] = 200.0                              # row 63 (timer bar)
    b1[21] = -200.0 * 62.5 / 63.0

    def kidx(o2y, o2x):
        return (o2y // 2 + 1) * 3 + (o2x // 2 + 1)

    # per axis component s: (cp_o1, cp_o2, tp_o1, tp_o2, dir_o1)
    AX = {1: (-1, -2, 0, 2, -1), 0: (0, 0, 1, 0, 0), -1: (1, 2, 0, -2, 1)}
    dirs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

    for j, (dy, dx) in enumerate(dirs):
        ay, ax = AX[dy], AX[dx]
        cp_k = kidx(ay[1], ax[1]); cp_ky, cp_kx = ay[0] + 1, ax[0] + 1
        tp_k = kidx(ay[3], ax[3]); tp_ky, tp_kx = ay[2] + 1, ax[2] + 1
        d_ky, d_kx = ay[4] + 1, ax[4] + 1
        n1, n2, g = 3 * j, 3 * j + 1, 3 * j + 2
        for out, sgn in ((n1, 1.0), (n2, -1.0)):
            W2[out, cp_k, cp_ky, cp_kx] += sgn
            W2[out, tp_k, tp_ky, tp_kx] += -sgn
            W2[out, 18, d_ky, d_kx] += 100.0
            W2[out, 19, d_ky, d_kx] += 100.0
            b2[out] = -100.0
        W2[g, 19, d_ky, d_kx] += 1.0
        W2[g, 9 + cp_k, cp_ky, cp_kx] += 1.0
        W2[g, 9 + tp_k, tp_ky, tp_kx] += 1.0
        b2[g] = -2.0
        # ---- f3 ----
        W3[n1, n1] = 1.0; W3[n1, n2] = 1.0; W3[n1, g] = -100.0          # V0a
        W3[n2, n1] = 1.0; W3[n2, n2] = 1.0; W3[n2, g] = -100.0          # V0b
        b3[n2] = -15.0
        W3[g, g] = 1.0; W3[g, n1] = -1.0; W3[g, n2] = -1.0              # V2
        # ---- drive & rays ----
        p["Uh"][j, n1] = 1.0
        p["Uh"][j, n2] = -3.0
        p["Uh"][j, g] = 1.0
        p["Wh"][j, j, 1 - dy, 1 - dx] = 0.6
        p["Wc"][0, 32 + j] = 1.0

    # tile indicator through f2/f3
    W2[24, 20, 1, 1] = 1.0
    W2[24, 21, 1, 1] = -10.0
    W3[24, 24] = 1.0
    p["Wc"][0, 24] = 1000.0

    p["bo"][6] = 1.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

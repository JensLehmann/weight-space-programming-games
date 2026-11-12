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
    K = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    code = {c: i + 1 for i, c in enumerate(K)}
    fs = [-2, 0, 2]
    offs = [(fy, fx) for fy in fs for fx in fs]
    kidx = {o: i for i, o in enumerate(offs)}

    # ---- f1 ----
    W1 = p["W1"]; b1 = p["b1"]
    for k, (fy, fx) in enumerate(offs):
        for c in K:
            W1[2 * k, c, 2 + fy, 2 + fx] = code[c]
            W1[2 * k + 1, c, 2 + fy, 2 + fx] = 1.0
    W1[18, 0, 2, 2] = 1.0            # is zero marker
    W1[19, 2, 2, 2] = 1.0            # nonzero marker (2 or 3)
    W1[19, 3, 2, 2] = 1.0
    W1[20, 16, 2, 2] = 63.0          # bottom row indicator
    b1[20] = -62.0

    # ---- f2 ----
    W2 = p["W2"]
    dec = {-2: (0, -2), 2: (0, 2), 0: (0, 0), 3: (1, 2), -3: (-1, -2)}
    def axis(comp):
        if comp == 1:
            return (-2, 3)
        if comp == -1:
            return (2, -3)
        return (0, 2)
    dirs = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
    for di, (dx, dy) in enumerate(dirs):
        ocx, onx = axis(dx)
        ocy, ony = axis(dy)
        ecx, fcx = dec[ocx]; ecy, fcy = dec[ocy]
        enx, fnx = dec[onx]; eny, fny = dec[ony]
        kC = kidx[(fcy, fcx)]; kN = kidx[(fny, fnx)]
        # D+
        W2[3 * di, 2 * kC, 1 + ecy, 1 + ecx] += 1.0
        W2[3 * di, 2 * kN, 1 + eny, 1 + enx] -= 1.0
        # D-
        W2[3 * di + 1, 2 * kC, 1 + ecy, 1 + ecx] -= 1.0
        W2[3 * di + 1, 2 * kN, 1 + eny, 1 + enx] += 1.0
        # V
        W2[3 * di + 2, 2 * kC + 1, 1 + ecy, 1 + ecx] += 1.0
        W2[3 * di + 2, 2 * kN + 1, 1 + eny, 1 + enx] += 1.0
    W2[24, 18, 1, 1] = 1.0                    # Z at anchor
    W2[25, 19, 1, 1] = 1.0                    # NZ at anchor
    W2[26, 2 * kidx[(0, 0)] + 1, 1, 1] = 1.0  # isK here
    W2[27, 20, 1, 1] = 1.0                    # bottom row

    # ---- f3 ----
    W3 = p["W3"]; b3 = p["b3"]
    M = 10.0
    for di in range(8):
        W3[di, 3 * di] = 1.0; W3[di, 3 * di + 1] = 1.0
        W3[di, 3 * di + 2] = M; W3[di, 24] = M
        b3[di] = -3 * M
        W3[8 + di, 3 * di] = -1.0; W3[8 + di, 3 * di + 1] = -1.0
        W3[8 + di, 3 * di + 2] = M; W3[8 + di, 25] = M
        b3[8 + di] = 1.0 - 3 * M
    W3[16, 26] = 1.0
    W3[17, 27] = 1.0

    # ---- recurrent rays ----
    Uh = p["Uh"]; Wh = p["Wh"]
    for di, (dx, dy) in enumerate(dirs):
        Uh[di, di] = 1.0
        Uh[di, 8 + di] = 1.0
        Wh[di, di, 1 - dy, 1 - dx] = 0.5

    # ---- click map ----
    P = 1000.0
    Wc = p["Wc"]
    for di in range(8):
        Wc[0, 32 + di] = 1.0
    Wc[0, 16] = P
    Wc[0, 17] = -2 * P
    p["bc"][0] = -P

    # ---- action: always click ----
    p["bo"][6] = 1.0

    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

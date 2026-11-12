import numpy as np


def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, b2, W3, b3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"], p["b3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]

    dirs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    tile = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    v = {k: 1.0 + 0.01 * i for i, k in enumerate(tile)}
    border = [0, 2, 3]
    bgc = [4, 5]
    K, G, Gp, Gz, M, alpha = 100.0, 200.0, 1000.0, 100.0, 1000.0, 0.5
    CH_Z, CH_TC, CH_N6 = 24, 25, 26

    # ---------- layer 1 ----------
    for n, (dy, dx) in enumerate(dirs):
        # C_n: code of pixel p-2d (constraint centre)
        for k in tile:
            W1[n, k, 2 - 2 * dy, 2 - 2 * dx] = v[k]
        # T_n: code of tile pixel at p + d (+2 perpendicular for orthogonal)
        if dy != 0 and dx != 0:
            py, px = 0, 0
        elif dy == 0:
            py, px = 1, 0
        else:
            py, px = 0, 1
        oy, ox = dy + 2 * py, dx + 2 * px
        for k in tile:
            W1[8 + n, k, 2 + oy, 2 + ox] = v[k]
        # CI_n: border colour at p, tile colour at p-2d, background outward
        for k in border:
            W1[16 + n, k, 2, 2] = 1.0
        for k in tile:
            W1[16 + n, k, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        comps = []
        if dy:
            comps.append((dy, 0))
        if dx:
            comps.append((0, dx))
        for (cy, cx) in comps:
            for k in bgc:
                W1[16 + n, k, 2 + cy, 2 + cx] = 1.0
        b1[16 + n] = -(1.0 + len(comps))
    W1[CH_Z, 0, 2, 2] = 1.0
    for k in tile + [6]:
        W1[CH_TC, k, 2, 2] = 1.0
    W1[CH_N6, 6, 1:4, 1:4] = 1.0

    # ---------- layer 2 (at q; constraint side at q-d, tile side at q+d) ----------
    for n, (dy, dx) in enumerate(dirs):
        a = (1 - dy, 1 - dx)
        bpos = (1 + dy, 1 + dx)
        # pos_n
        W2[n, n, a[0], a[1]] += -K
        W2[n, 8 + n, bpos[0], bpos[1]] += K
        W2[n, 16 + n, a[0], a[1]] += -G
        b2[n] = G
        # neg_n
        W2[8 + n, n, a[0], a[1]] += K
        W2[8 + n, 8 + n, bpos[0], bpos[1]] += -K
        W2[8 + n, 16 + n, a[0], a[1]] += -G
        b2[8 + n] = G
        # Zb_n
        W2[16 + n, CH_Z, a[0], a[1]] = 1.0
        W2[16 + n, 16 + n, a[0], a[1]] = 1.0
        b2[16 + n] = -1.0
    W2[24, CH_TC, 1, 1] = -1.0
    b2[24] = 1.0
    W2[25, CH_N6, :, :] = -1.0
    b2[25] = 1.0

    # ---------- layer 3 (1x1) ----------
    for n in range(8):
        # wE: zero border but colours differ
        W3[n, n] = 1.0
        W3[n, 8 + n] = 1.0
        W3[n, 16 + n] = Gp
        b3[n] = -Gp
        # wD: nonzero border but colours equal
        W3[8 + n, n] = -1.0
        W3[8 + n, 8 + n] = -1.0
        W3[8 + n, 16 + n] = -Gz
        b3[8 + n] = 1.0
        # wE2: spurious huge mismatch (no tile under the zero border)
        W3[16 + n, n] = 1.0
        W3[16 + n, 8 + n] = 1.0
        W3[16 + n, 16 + n] = Gp
        b3[16 + n] = -20.0 - Gp
    W3[24, 24] = 1.0  # notTile
    W3[25, 25] = 1.0  # notNTi

    # ---------- recurrent rays ----------
    for n, (dy, dx) in enumerate(dirs):
        Uh[n, n] = 1.0
        Uh[n, 8 + n] = 1.0
        Uh[n, 16 + n] = -2.0
        Wh[n, n, 1 - dy, 1 - dx] = alpha  # read R(p - d): propagate along +d
        Wh[8, n, 1, 1] = 1.0
        Wh[9, n, 1, 1] = 1.0
    Uh[8, 24] = -M
    Uh[9, 24] = -M
    Uh[9, 25] = -M

    # ---------- click map & action ----------
    p["Wc"][0, 32 + 8] = 1.0
    p["Wc"][0, 32 + 9] = 2.0
    p["bo"][6] = 1.0
    return p

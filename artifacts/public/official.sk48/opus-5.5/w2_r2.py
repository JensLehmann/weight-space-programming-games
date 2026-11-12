import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float64) for k, s in shapes.items()}

    # f1/f3 channels: colour0 count,row,col ; colour1 count,row,col
    for ci, c in enumerate((0, 1)):
        ch = 3 * ci
        p["W1"][ch, c, 2, 2] = 1.0
        p["W1"][ch + 1, c, 2, 2] = 1.0
        p["W1"][ch + 1, 16, 2, 2] = 1.0
        p["b1"][ch + 1] = -1.0
        p["W1"][ch + 2, c, 2, 2] = 1.0
        p["W1"][ch + 2, 17, 2, 2] = 1.0
        p["b1"][ch + 2] = -1.0
    for c in range(6):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0

    # z0..5: deviations of exact sums from fresh level-1 values
    M = 4096.0 * 63.0
    feats = [(1, 1164.0), (2, 432.0), (5, 404.0)]
    for i, (c, v) in enumerate(feats):
        p["Wz"][2 * i, 64 + c] = M
        p["bz"][2 * i] = -v
        p["Wz"][2 * i + 1, 64 + c] = -M
        p["bz"][2 * i + 1] = v
    # z6..8: means of colour-1 features (fallback change detection)
    for k in range(3):
        p["Wz"][6 + k, 64 + 3 + k] = 1.0

    K = 1e4
    # ---- undo phase: units 0..7 ----
    NU = 8
    p["Wma"][0, :] = -K
    p["bm"][0] = 0.5 * K
    for j in range(1, NU):
        p["Wm"][j, j - 1] = K
        p["bm"][j] = -0.5 * K
    for j in range(NU):
        p["Wo"][7, 128 + j] += 50.0
        p["bo"][7] += 50.0

    # ---- scripted plan for level 1: units 8..27 ----
    script = [1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 3, 3, 1, 4, 4]
    S0 = NU
    p["Wm"][S0, NU - 1] = K
    p["Wmz"][S0, 0:6] = -K
    p["bm"][S0] = -0.5 * K
    for t in range(1, len(script)):
        p["Wm"][S0 + t, S0 + t - 1] = K
        p["bm"][S0 + t] = -0.5 * K
    for t, a in enumerate(script):
        p["Wo"][a, 128 + S0 + t] += 50.0
        p["bo"][a] += 50.0
    base = S0 + len(script)          # 28

    # ---- fallback explorer ----
    moves = (1, 2, 3, 4)
    succ = {4: 1, 1: 3, 3: 2, 2: 4}
    for idx, i in enumerate(moves):
        jc = base + idx               # 28..31
        p["Wma"][jc, i] = 5.0
        p["Wo"][succ[i], 128 + jc] += 1.0
    st = base + 4                     # 32..34 storage
    for k in range(3):
        p["Wmz"][st + k, 6 + k] = 1.0
    KD, K2, theta = 1e6, 1e6, 3e-5
    j = st + 3                        # 35..58 detectors
    for i in moves:
        for k in range(3):
            for s in (1.0, -1.0):
                p["Wmz"][j, 6 + k] = s * KD
                p["Wm"][j, st + k] = -s * KD
                p["Wma"][j, i] = K2
                p["bm"][j] = -KD * theta - K2
                p["Wo"][i, 128 + j] += 2.0
                p["bo"][i] += 2.0
                j += 1
    assert j <= 64

    p["bo"][4] += 0.5
    for a in (0, 5, 6):
        p["bo"][a] = -1000.0
    p["bo"][7] += -30.0

    return {k: v.astype(np.float32) for k, v in p.items()}

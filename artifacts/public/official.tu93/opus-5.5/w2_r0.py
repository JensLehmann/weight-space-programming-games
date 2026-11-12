import numpy as np

def build():
    shapes = {"W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
              "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
              "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
              "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,)}
    P = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, W3, b3 = P["W1"], P["b1"], P["W2"], P["W3"], P["b3"]
    Wh, Uh, bh, Wz, Wo, bo = P["Wh"], P["Uh"], P["bh"], P["Wz"], P["Wo"], P["bo"]

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    walk_cols = [0, 2, 14, 9, 4, 8, 12, 13, 15, 11]
    enemy_cols = [8, 12, 13]
    dot_cols = [15, 11]
    player_cols = [9, 4]

    # ---- conv1 ----
    for c in walk_cols:
        W1[0, c, 2, 2] = 1.0
    W1[1, 14, 2, 2] = 1.0
    for c in enemy_cols + dot_cols:
        W1[14, c, 2, 2] = 1.0
    for f, (fr, fc) in enumerate(dirs):
        pr, pc = (0, 1) if fr != 0 else (1, 0)
        for s in (1, 2):
            o = 2 + 2 * f + (s - 1)
            for t in (-1, 0, 1):
                r, cc = -s * fr + t * pr, -s * fc + t * pc
                for dc in dot_cols:
                    W1[o, dc, r + 2, cc + 2] += 1.0
                for u in range(s):
                    r2, c2 = -u * fr + t * pr, -u * fc + t * pc
                    for ec in enemy_cols:
                        W1[o, ec, r2 + 2, c2 + 2] -= 2.0
    for d, (dr, dc) in enumerate(dirs):
        pr, pc = (0, 1) if dr != 0 else (1, 0)
        o = 10 + d
        for t in (-1, 0, 1):
            for k in (2, 1):
                r, cc = -k * dr + t * pr, -k * dc + t * pc
                for pcol in player_cols:
                    W1[o, pcol, r + 2, cc + 2] += 1.0
            for pcol in player_cols:
                W1[o, pcol, t * pr + 2, t * pc + 2] -= 10.0
        b1[o] = -5.0

    # ---- conv2 ----
    W2[0, 0, 1, 1] = 1.0
    W2[1, 1, 1, 1] = 1.0
    W2[11, 14, 1, 1] = 1.0
    for d, (dr, dc) in enumerate(dirs):
        W2[3 + d, 10 + d, 1 - dr, 1 - dc] = 1.0      # pn_d
        W2[7 + d, 2 + 2 * d, 1, 1] += 1.0            # Y_d (all enemies)
        W2[7 + d, 3 + 2 * d, 1, 1] += 1.0
        W2[7 + d, 3 + 2 * d, 1 - dr, 1 - dc] += 1.0

    # ---- f3 ----
    W3[0, 1] = 1.0
    W3[1, 0] = -1.0; b3[1] = 1.0                      # wall
    for d in range(4):
        W3[2 + d, 3 + d] = 1.0                        # pn_d
        W3[6 + d, 3 + d] = 1.0                        # open_d
        W3[6 + d, 0] = 1.0
        b3[6 + d] = -1.0
        W3[10 + d, 7 + d] = 1.0                       # Y_d
    W3[14, 11] = 1.0                                  # E

    # ---- recurrence ----
    S, BIG = 1e33, 1e37
    Wh[0, 0, :, :] = 0.11
    Uh[0, 0] = S
    Uh[0, 1] = -BIG
    Wh[0, 11, 1, 1] = -1e36
    Wh[0, 6, 1, 1] = 1.0
    for d in range(4):
        Wh[1 + d, 0, 1, 1] = 1.0
        Uh[1 + d, 2 + d] = BIG
        bh[1 + d] = -BIG
    Wh[5, 0, :, :] = 0.11
    Wh[5, 5, :, :] = 0.11
    Wh[5, 12, 1, 1] = -BIG
    Uh[5, 1] = -BIG
    Wh[6, 5, :, :] = 0.11
    Uh[6, 14] = BIG
    bh[6] = -BIG
    for f, (fr, fc) in enumerate(dirs):
        Wh[7 + f, 7 + f, 1 - fr, 1 - fc] = 0.5
        Uh[7 + f, 10 + f] = 1.0
        Wh[11, 7 + f, 1, 1] = 20.0
        Wh[12, 7 + f, 1, 1] = -20.0
    bh[11] = -3.2
    bh[12] = 4.2

    # ---- readout ----
    for d in range(4):
        Wz[d, 1 + d] = 1.0
        Wz[4 + d, 32 + 6 + d] = 1.0
        Wo[1 + d, d] = 1.0
        Wo[1 + d, 4 + d] = 1e-30
    bo[0] = -1.0
    bo[5] = -1.0; bo[6] = -1.0; bo[7] = -1.0
    return P

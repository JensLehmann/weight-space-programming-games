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
    W1, b1, W2, b2, W3 = p["W1"], p["b1"], p["W2"], p["b2"], p["W3"]
    Wh, Uh, bh = p["Wh"], p["Uh"], p["bh"]
    nb = [c for c in range(16) if c not in (4, 5)]
    items = [1] + list(range(6, 16))
    # ---- layer 1 (5x5, centre (2,2), left neighbour (2,1)) ----
    for c in nb:
        W1[0, c, 2, 2] = 1.0          # NB
        W1[1, c, 2, 1] = 1.0          # NB left
        W1[2, c, 2, 2] = float(c)     # colour value
    W1[3, 2, 2, 2] = 1.0              # RED
    W1[4, 2, 2, 1] = 1.0              # RED left
    W1[5, 0, 2, 2] = 1.0              # BLACK
    for c in items:
        W1[6, c, 2, 2] = 1.0          # ITEM colour
    ramps = {7: 0, 8: 1, 9: 2, 10: 57, 11: 58, 12: 59, 13: 54, 14: 55, 15: 52, 16: 53}
    for ch, a in ramps.items():
        W1[ch, 16, 2, 2] = 63.0
        b1[ch] = -float(a)
    W1[17, 16, 2, 2] = 1.0            # row coord
    W1[18, 17, 2, 2] = 1.0            # col coord
    # ---- layer 2 (3x3 centre only) ----
    def w2(o, i, v):
        W2[o, i, 1, 1] += v
    def tent1(o, s=1.0):
        w2(o, 7, s); w2(o, 8, -2 * s); w2(o, 9, s)
    def tent58(o, s=1.0):
        w2(o, 10, s); w2(o, 11, -2 * s); w2(o, 12, s)
    # 0 Em2: target left edge on row 1
    w2(0, 0, 1); w2(0, 1, -1); tent1(0); b2[0] = -1
    # 1 TC2: target colour on row 1
    w2(1, 2, 1); tent1(1, 16.0); b2[1] = -16
    # 2 Dm2: palette hole left edge on row 58
    w2(2, 3, 1); w2(2, 4, -1); tent58(2); b2[2] = -1
    # 3 Sel2: black at rows >= 55
    w2(3, 5, 1); w2(3, 13, 1); w2(3, 14, -1); b2[3] = -1
    # 4 PI2: palette item pixel row 58
    w2(4, 6, 1); tent58(4); b2[4] = -1
    # 5 CV2
    w2(5, 2, 1)
    # 6 DOT2: red at rows <= 52
    w2(6, 3, 1); w2(6, 15, -1); w2(6, 16, 1)
    # 7,8 coords
    w2(7, 17, 1); w2(8, 18, 1)
    # ---- layer 3 ----
    for o in range(4):
        W3[o, 64 + o] = 1.0           # column max broadcasts
    for o in range(4, 9):
        W3[o, o] = 1.0
    # ---- recurrent block ----
    C, L, R = (1, 1), (1, 0), (1, 2)
    def wh(o, i, pos, v):
        Wh[o, i, pos[0], pos[1]] += v
    wh(0, 0, L, 1); Uh[0, 0] = 1                       # Ct
    wh(1, 1, L, 1); Uh[1, 2] = 1                       # Lk
    wh(2, 2, R, 1); Uh[2, 2] = 1                       # Rk
    for j, o in enumerate((3, 4, 5)):                  # E_j
        wh(o, 0, C, 1); wh(o, 1, L, -1); wh(o, 2, C, -1); bh[o] = -float(j)
    wh(6, 3, C, 16); wh(6, 4, C, -32); wh(6, 5, C, 16); Uh[6, 1] = 1; bh[6] = -16  # X
    wh(7, 7, L, 1); wh(7, 6, C, 1)                     # Lp
    wh(8, 8, R, 1); wh(8, 6, C, 1)                     # Rp
    wh(9, 7, L, -1 / 6); wh(9, 8, C, -1 / 6); Uh[9, 5] = 1        # P1
    wh(10, 7, L, 1 / 6); wh(10, 8, C, 1 / 6); Uh[10, 5] = -1      # P2
    wh(11, 11, L, 1); Uh[11, 3] = 10                   # Ls
    wh(12, 12, R, 1); Uh[12, 3] = 10                   # Rs
    wh(13, 11, L, 1); wh(13, 12, C, 1)                 # Q
    Uh[13, 6] = 100; Uh[13, 7] = -30; Uh[13, 8] = -4; bh[13] = -100
    # ---- click map ----
    p["Wc"][0, 4] = 15.0
    p["Wc"][0, 32 + 9] = -10.0
    p["Wc"][0, 32 + 10] = -10.0
    p["Wc"][0, 32 + 13] = 10.0
    p["bc"][0] = -5.0
    # ---- action head ----
    p["Wz"][0, 32 + 6] = -1.0   # max DOT
    p["Wz"][0, 32 + 3] = -1.0   # max Sel
    p["bz"][0] = 1.0
    p["Wo"][5, 0] = 10.0
    p["bo"][6] = 5.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}

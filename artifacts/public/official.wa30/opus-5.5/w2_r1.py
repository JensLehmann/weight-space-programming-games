import numpy as np

def build():
    P = {
        "W1": np.zeros((32, 19, 5, 5)), "b1": np.zeros(32),
        "W2": np.zeros((32, 32, 3, 3)), "b2": np.zeros(32),
        "W3": np.zeros((32, 96)), "b3": np.zeros(32),
        "Wh": np.zeros((16, 16, 3, 3)), "Uh": np.zeros((16, 32)), "bh": np.zeros(16),
        "Wz": np.zeros((128, 96)), "bz": np.zeros(128),
        "Wm": np.zeros((64, 64)), "Wmz": np.zeros((64, 128)), "Wma": np.zeros((64, 8)),
        "bm": np.zeros(64), "Wo": np.zeros((8, 192)), "bo": np.zeros(8),
        "Wc": np.zeros((1, 48)), "bc": np.zeros(1),
    }
    W1, b1, W2, W3, b3 = P["W1"], P["b1"], P["W2"], P["W3"], P["b3"]
    # ---- layer 1 (5x5, centre 2) ----
    W1[0, 2, 2, 2] = 1.0; W1[0, 1, 1:4, 1:4] = -1.0          # c0 target interior T
    W1[1, 4, 2, 2] = 1.0; W1[1, 3, 2, 2] = 1.0               # c1 box border (4/3)
    W1[2, 16, 2, 2] = 64.0; b1[2] = -63.0                    # c2 bottom row
    W1[3, 5, 2, 2] = 1.0                                     # c3 wall colour 5
    W1[4, 2, 2, 2] = 10.0; W1[4, 1, 1:4, 1:4] += 1.0; b1[4] = -10.0  # c4 holey blocker
    W1[5, 14, 2, 2] = 1.0                                    # c5 player body
    W1[6, 3, 2, 2] = 1.0                                     # c6 facing border
    for k, b in ((7, -4.0), (8, -5.0)):                      # c7/c8 carried interior
        W1[k, 9, 2, 2] = 1.0; W1[k, 0, 1:4, 1:4] += 1.0; b1[k] = b
    W1[9, 11, 2, 2] = 1.0                                    # c9 facing adversary
    W1[10, 2, 1:4, 1:4] = 1.0; W1[10, 1, 1:4, 1:4] = -3.0    # c10 target-2 nearby
    # ---- layer 2 (3x3, centre 1) ----
    W2[0, 0, 1, 1] = 1.0                                     # d0 T
    W2[2, 3, 1, 1] = 1.0; W2[2, 4, 1, 1] = 1.0               # d2 walls (5 + holey)
    W2[3, 1, 1, 1] = 1.0; W2[3, 2, 1, 1] = -2.0              # d3 all box borders
    W2[4, 5, 1, 1] = 1.0                                     # d4 player
    W2[5, 6, 1, 1] = 1.0                                     # d5 facing (raw)
    W2[6, 7, 1, 1] = 1.0; W2[6, 8, 1, 1] = -1.0              # d6 carry mask E
    W2[7, 10] = 1.0                                          # d7 target-2 count near
    W2[8, 9, 1, 1] = 1.0                                     # d8 adversary facing
    W2[9, 0] = 1.0                                           # d9 dilated T
    W2[11, 3, 1, 1] = 1.0                                    # d11 colour-5 walls only
    # ---- layer 3 ----
    for k in (0, 2, 3, 4, 5, 6, 8, 11):
        W3[k, k] = 1.0
    W3[7, 6] = 100.0; W3[7, 7] = 1.0; b3[7] = -100.0         # carried box on target
    W3[9, 3] = 100.0; W3[9, 9] = 1.0; b3[9] = -100.0         # on-target seed (border & near T)
    # ---- recurrent ----
    Wh, Uh, bh = P["Wh"], P["Uh"], P["bh"]
    BIG = 1e4
    NB = [(0, 1), (2, 1), (1, 0), (1, 2)]
    for ch in (0, 1):
        for (a, b) in NB:
            Wh[ch, ch, a, b] = 0.249
    # h0 box field: sources = box borders, walls clamp, on-target boxes clamp
    Uh[0, 3] = 1.0; Uh[0, 2] = -BIG; Wh[0, 14, 1, 1] = -1e6
    # h1 target field: sources T, obstacles colour-5 walls and boxes
    Uh[1, 0] = 1.0; Uh[1, 11] = -BIG; Uh[1, 3] = -BIG
    grads = [((0, 1), (2, 1)), ((2, 1), (0, 1)), ((1, 0), (1, 2)), ((1, 2), (1, 0))]
    for i, (pos, neg) in enumerate(grads):
        for src, base, mask in ((0, 2, 4), (1, 6, 6), (1, 10, 4)):
            k = base + i
            Wh[k, src, pos[0], pos[1]] = 1.0
            Wh[k, src, neg[0], neg[1]] = -1.0
            Uh[k, mask] = BIG; bh[k] = -BIG
    # h14 on-target flood fill along box borders
    for (a, b) in NB:
        Wh[14, 14, a, b] = 0.25
    Uh[14, 9] = 1.0; Uh[14, 3] = BIG; bh[14] = -BIG
    # h15 grabbable facing = facing & not on target
    Uh[15, 5] = 1.0; Wh[15, 14, 1, 1] = -1e6
    # ---- readout ----
    Wz, bz = P["Wz"], P["bz"]; s = 0.01
    for i in range(4):
        Wz[i, 2 + i] = s; Wz[i, 32 + 6] = -100.0             # seek box (not carrying)
        Wz[4 + i, 6 + i] = s                                 # carried box -> target
        Wz[12 + i, 10 + i] = s; Wz[12 + i, 32 + 6] = 100.0; bz[12 + i] = -100.0
    Wz[8, 32 + 6] = 1.0      # carrying
    Wz[9, 15] = 1.0          # grabbable facing (h15 max)
    Wz[10, 32 + 8] = 1.0     # facing adversary
    Wz[11, 32 + 7] = 1.0     # carried box on target
    # ---- memory ----
    Wmz, Wma, bm = P["Wmz"], P["Wma"], P["bm"]
    Wmz[1, 11] = 10.0; bm[1] = -5.0                                   # release
    Wmz[2, 9] = 10.0; Wmz[2, 8] = -10.0; bm[2] = -5.0                 # grab
    Wmz[3, 10] = 10.0; bm[3] = -5.0                                   # kill adversary
    for j in (1, 2, 3):
        Wma[j, 5] = -1e4
    # ---- logits ----
    Wo, bo = P["Wo"], P["bo"]
    for i in range(4):
        Wo[1 + i, i] = 1.0; Wo[1 + i, 4 + i] = 1.0; Wo[1 + i, 12 + i] = 1.0
    for j in (1, 2, 3):
        Wo[5, 128 + j] = 50.0
    bo[5] = 99.0
    bo[0] = -1000.0; bo[6] = -1000.0; bo[7] = -1000.0
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}

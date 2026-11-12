import numpy as np

def build():
    shapes = {"W1":(32,19,5,5),"b1":(32,),"W2":(32,32,3,3),"b2":(32,),"W3":(32,96),"b3":(32,),
              "Wh":(16,16,3,3),"Uh":(16,32),"bh":(16,),"Wz":(128,96),"bz":(128,),"Wm":(64,64),
              "Wmz":(64,128),"Wma":(64,8),"bm":(64,),"Wo":(8,192),"bo":(8,),"Wc":(1,48),"bc":(1,)}
    P = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    W1, b1, W2, W3 = P["W1"], P["b1"], P["W2"], P["W3"]

    def feat(o, d, conds, c, x):
        # conds: list of (pos, colours); core computed at pos c, output located at pos x
        for pos, cols in conds:
            off = pos - c
            r, cc = off * d[0], off * d[1]
            assert abs(r) <= 2 and abs(cc) <= 2
            for col in cols:
                W1[o, col, 2 + r, 2 + cc] += 1.0
        b1[o] = -(len(conds) - 1)
        s = c - x
        W2[o, o, 1 + s * d[0], 1 + s * d[1]] = 1.0
        W3[o, o] = 1.0

    G = [14]
    BH = [0, 11]
    Ecol = [1, 12]
    dirs = {"R": (0, 1), "D": (1, 0), "L": (0, -1), "U": (-1, 0)}

    def sets(name):
        if name in ("R", "L"):
            return dict(B2=BH, M1g=[14], M1o=[8, 9, 15], M2=[14, 8, 9, 15], B7=BH)
        if name == "D":
            return dict(B2=[0, 11, 15], M1g=[14], M1o=[8, 9, 7], M2=[14, 8, 9, 5], B7=BH)
        return dict(B2=BH, M1g=[14], M1o=[8, 9, 5], M2=[14, 8, 9, 7], B7=[0, 11, 15])

    # exact directions R, D: Lg, Lo, R features
    for k, nm in enumerate(["R", "D"]):
        d = dirs[nm]; s = sets(nm); base = 3 * k
        feat(base, d, [(0, G), (1, BH), (2, s["B2"]), (3, s["M1g"])], 1, 1)
        feat(base + 1, d, [(0, G), (1, BH), (2, s["B2"]), (3, s["M1o"])], 1, 1)
        feat(base + 2, d, [(6, s["M2"]), (7, s["B7"]), (8, BH), (9, Ecol)], 7, 6)
    # partial directions L, U
    for k, nm in enumerate(["L", "U"]):
        d = dirs[nm]; s = sets(nm); base = 6 + 2 * k
        feat(base, d, [(0, G), (1, BH), (2, s["B2"]), (3, s["M1g"])], 1, 1)
        feat(base + 1, d, [(0, G), (1, BH), (2, s["B2"]), (3, s["M1o"])], 1, 1)

    # gray target ring (not on row 0)
    W1[10, 2, 2, 2] = 1.0
    for col in range(16):
        W1[10, col, 1, 2] += 1.0
    b1[10] = -1.0
    W2[10, 10, 1, 1] = 1.0; W3[10, 10] = 1.0
    # any green pixel
    W1[11, 14, 2, 2] = 1.0
    W2[11, 11, 1, 1] = 1.0; W3[11, 11] = 1.0

    # recurrent shift chains + AND
    Uh, bh, Wh = P["Uh"], P["bh"], P["Wh"]
    for k, nm in enumerate(["R", "D"]):
        d = dirs[nm]; o = 6 * k; base = 3 * k
        Uh[o, base + 2] = 1.0                       # h0 = R
        for j in range(1, 5):
            Wh[o + j, o + j - 1, 1 + d[0], 1 + d[1]] = 1.0
        J = o + 5
        Wh[J, o + 4, 1 + d[0], 1 + d[1]] = 1.0
        Uh[J, base] = 1.0
        Uh[J, base + 1] = 0.85
        bh[J] = -1.0

    # click map
    Wc = P["Wc"]
    Wc[0, 10] = 100.0
    Wc[0, 32 + 5] = 20.0
    Wc[0, 32 + 11] = 20.0
    Wc[0, 0] = 2.0; Wc[0, 1] = 1.7
    Wc[0, 3] = 2.0; Wc[0, 4] = 1.7
    Wc[0, 6] = 10.0; Wc[0, 7] = 8.5
    Wc[0, 8] = 10.0; Wc[0, 9] = 8.5
    Wc[0, 11] = 1.0

    # memory ring counter (phases 0..L-1) plus "started" flag S
    Lr, S, K = 20, 20, 10.0
    Wm, bm = P["Wm"], P["bm"]
    bm[S] = K
    Wm[0, Lr - 1] = K; Wm[0, S] = -2 * K; bm[0] = 2 * K
    for j in range(1, Lr):
        Wm[j, j - 1] = K; Wm[j, S] = K; bm[j] = -K

    P["bo"][6] = 1.0
    Wo = P["Wo"]
    for ph, a in zip([4, 9, 14, 19], [1, 3, 2, 4]):
        Wo[a, 128 + ph] = 5.0
    return P

import numpy as np

def build():
    G = []; P = {}
    f32 = lambda a: np.asarray(a, np.float32)
    def nd(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); G.append(d); return i
    def cv(i, x, W, b=None):
        P["W" + i] = f32(W); kw = {"w": "W" + i}
        if b is not None:
            P["b" + i] = f32(b); kw["b"] = "b" + i
        return nd(i, "conv", [x], **kw)
    def dn(i, x, W, b=None):
        P["W" + i] = f32(W); kw = {"w": "W" + i}
        if b is not None:
            P["b" + i] = f32(b); kw["b"] = "b" + i
        return nd(i, "dense", [x], **kw)
    def sl(i, x, a, b): return nd(i, "slice", [x], start=a, stop=b)
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    one = dn("one", "aprev", np.zeros((1, 8)), [1.0])
    dn("m1", "aprev", np.zeros((1, 8)), [-1.0])

    W = np.zeros((32, 3, 1, 1)); b = np.zeros(32)
    for k in range(16):
        W[k, 0, 0, 0] = 63; b[k] = -(4 * k + 1.5)
        W[16 + k, 1, 0, 0] = 63; b[16 + k] = -(4 * k + 1.5)
    cv("Bpre", "coords", W, b)
    nd("Babs", "abs", ["Bpre"])
    cv("Bl", "Babs", -4 * np.eye(32)[:, :, None, None], 8 * np.ones(32))
    nd("Br", "relu", ["Bl"])
    nd("B", "min2", ["Br", one])
    nd("CB", "concat", ["coords", "B"])
    W = np.zeros((2, 35, 1, 1))
    W[0, 0, 0, 0] = 63; W[1, 1, 0, 0] = 63
    for k in range(16):
        W[0, 3 + k, 0, 0] = -4 * k; W[1, 19 + k, 0, 0] = -4 * k
    cv("PHI", "CB", W)
    W = np.zeros((256, 32, 1, 1)); W2 = np.zeros((256, 32, 1, 1))
    for c in range(256):
        W[c, c // 16, 0, 0] = 1; W2[c, 16 + c % 16, 0, 0] = 1
    cv("R256", "B", W); cv("C256", "B", W2)
    nd("M256", "mul", ["R256", "C256"])

    def mask(i, K, ax):
        WA = np.zeros((7 * K, 2, 1, 1)); bA = np.zeros(7 * K)
        WB = np.zeros((7 * K, 2, 1, 1)); bB = np.zeros(7 * K)
        for m in range(K):
            for ai in range(7):
                a = ai - 3; c = m * 7 + ai
                WA[c, ax, 0, 0] = 2; bA[c] = 2 * a + 1
                WB[c, ax, 0, 0] = -2; bB[c] = 2 * (3 - a) + 1
        cv(i + "A", "PHI", WA, bA); cv(i + "B", "PHI", WB, bB)
        nd(i + "Ar", "relu", [i + "A"]); nd(i + "Br", "relu", [i + "B"])
        nd(i + "m", "min2", [i + "Ar", i + "Br"])
        return nd(i, "min2", [i + "m", one])

    def pool(i, X, K):
        cur = X
        for ax in (0, 1):
            W = np.zeros((7 * K, K, 7, 7)); S = np.zeros((K, 7 * K, 1, 1))
            for m in range(K):
                for ai in range(7):
                    a = ai - 3
                    if ax == 0: W[m * 7 + ai, m, 3 + a, 3] = 1
                    else: W[m * 7 + ai, m, 3, 3 + a] = 1
                    S[m, m * 7 + ai, 0, 0] = 1
            t = i + "x%d" % ax
            cv(t + "s", cur, W)
            mk = mask(t + "k", K, ax)
            nd(t + "p", "mul", [t + "s", mk])
            cv(t + "u", t + "p", S)
            cur = nd(t + "o", "min2", [t + "u", one])
        return cur

    groups = [[9], [4, 3], [5, 12, 15, 11], [15, 11], [0], [14], [2], [1], [3], [11], [12]]
    W = np.zeros((11, 16, 1, 1))
    for g, cols in enumerate(groups):
        for c in cols: W[g, c, 0, 0] = 1
    cv("CL", "x0", W)
    sl("X8", "CL", 0, 8)
    x9 = sl("x9", "CL", 0, 1); x2 = sl("x2", "CL", 6, 7)
    x3 = sl("x3", "CL", 8, 9); x11 = sl("x11", "CL", 9, 10)
    x12 = sl("x12", "CL", 10, 11)
    PC = pool("PA", "X8", 8)
    names = ["cp9", "cpB", "cpW", "cpE", "cp0", "cp14", "cp2", "cp1"]
    for j, n in enumerate(names): sl(n, PC, j, j + 1)
    cp12 = pool("P12", x12, 1)

    W = np.zeros((1, 8, 1, 1)); W[0, 1] = W[0, 2] = W[0, 4] = 1
    cv("nscs", PC, W)
    nd("nsc", "min2", ["nscs", one])
    nd("nsci", "sub", [one, "nsc"])
    nd("seed", "mul", [x9, "nsci"])
    nd("Areg", "add", [x2, "seed"])
    P["Wfl"] = f32(np.ones((1, 1, 3, 3)))
    G.append({"id": "FL", "op": "loop", "T": 10,
              "state": [{"id": "fs", "init": "seed"}],
              "body": [{"id": "fc", "op": "conv", "in": ["fs"], "w": "Wfl"},
                       {"id": "fn", "op": "min2", "in": ["fc", "Areg"]}],
              "next": {"fs": "fn"}})
    nd("f2", "mul", ["FL.fs", x2])
    nd("f2d", "conv", ["f2"], w="Wfl")
    nd("f2m", "min2", ["f2d", one])
    cpT = pool("PT", "f2m", 1)

    nd("b21", "mul", ["cp2", "cp1"]); nd("n9", "sub", [one, "cp9"])
    nd("blk", "mul", ["b21", "n9"])

    cv("blk2", "blk", 2 * np.ones((1, 1, 1, 1)))
    nd("seenS", "max2", [cpT, "blk2"])
    nd("Tm", "mul", ["M256", "seenS"]); nd("Tvec", "gmax", ["Tm"])
    nd("Mm", "mul", ["M256", "mem"])
    cv("memS", "Mm", np.ones((1, 256, 1, 1)))
    nd("mneg", "vmul", ["mem", "m1"]); nd("md", "vadd", ["Tvec", "mneg"])
    nd("mr", "vrelu", ["md"]); nd("mem_next", "vadd", ["mem", "mr"])
    nd("ms1", "sub", ["memS", one]); nd("msa", "abs", ["ms1"])
    nd("msb", "sub", [one, "msa"]); nd("Tmem", "relu", ["msb"])
    nd("Hmem", "relu", ["ms1"])
    nd("Tcell", "max2", [cpT, "Tmem"]); nd("Hcell", "max2", ["blk", "Hmem"])

    nd("boxnc", "mul", ["cp9", "cpB"])
    nd("bs1", "add", ["cpW", "boxnc"]); nd("bs2", "add", ["bs1", "Hcell"])
    nd("Blocked", "min2", ["bs2", one]); nd("F", "sub", [one, "Blocked"])

    # ---- region reachability from empty targets (helper-passable cells) ----
    nd("ET", "mul", ["Tcell", "F"])
    nd("Pas0", "add", ["F", cp12]); nd("Pas", "min2", ["Pas0", one])
    cr = np.zeros((1, 1, 3, 3)); cr[0, 0, 1, :] = 1; cr[0, 0, :, 1] = 1
    P["Wcr"] = f32(cr)
    G.append({"id": "RL", "op": "loop", "T": 160,
              "state": [{"id": "r", "init": "ET"}],
              "body": [{"id": "rc", "op": "conv", "in": ["r"], "w": "Wcr"},
                       {"id": "rm", "op": "min2", "in": ["rc", one]},
                       {"id": "rp", "op": "mul", "in": ["rm", "Pas"]},
                       {"id": "rn", "op": "max2", "in": ["rp", "ET"]}],
              "next": {"r": "rn"}})
    nd("dR", "conv", ["RL.r"], w="Wcr")
    nd("dR1", "min2", ["dR", one])
    nd("hgp", "mul", ["dR1", "Hcell"])
    Hg = pool("PH", "hgp", 1)
    nd("THg0", "add", ["Tcell", Hg]); nd("THg", "min2", ["THg0", one])
    nd("nTHg", "sub", [one, "THg"])

    nd("Cb", "mul", ["cp0", "cp9"]); nd("kv", "gmax", ["Cb"])
    nd("GA0", "mul", ["boxnc", "nTHg"]); nd("GA", "add", ["GA0", "cpE"])

    def sh4(i, M):
        W1 = np.zeros((4, 1, 7, 7)); W2 = np.zeros((4, 4, 3, 3))
        for k, (di, dj) in enumerate(DIRS):
            W1[k, 0, 3 + 3 * di, 3 + 3 * dj] = 1; W2[k, k, 1 + di, 1 + dj] = 1
        cv(i + "a", M, W1); return cv(i, i + "a", W2)
    sh4("ShCb", "Cb")
    nd("dm", "mul", ["ShCb", "cp14"]); nd("dvec", "gmax", ["dm"])
    def Sd(i, M):
        sh4(i + "h", M); nd(i + "m", "mul", [i + "h", "dvec"])
        return cv(i, i + "m", np.ones((1, 4, 1, 1)))
    nd("FBx0", "add", ["F", "Hcell"]); nd("FBx", "min2", ["FBx0", one])
    Sd("SF", "FBx"); nd("FB", "mul", ["F", "SF"])
    nd("TF1", "mul", ["Tcell", "F"])
    cv("Hh", Hg, 0.5 * np.ones((1, 1, 1, 1)))
    nd("TF", "max2", ["TF1", "Hh"])
    Sd("STF", "TF"); nd("GB", "mul", ["STF", "F"])
    nd("dG", "sub", ["GB", "GA"]); nd("dGk", "mul", ["dG", "kv"]); nd("Gm", "add", ["GA", "dGk"])
    nd("dF", "sub", ["FB", "F"]); nd("dFk", "mul", ["dF", "kv"]); nd("Fe", "add", ["F", "dFk"])

    gam = 0.97
    Wv = np.zeros((4, 1, 3, 3))
    for k, (di, dj) in enumerate(DIRS): Wv[k, 0, 1 + di, 1 + dj] = gam
    P["WVc"] = f32(Wv)
    G.append({"id": "VL", "op": "loop", "T": 224,
              "state": [{"id": "V", "init": "Gm"}],
              "body": [{"id": "Vc", "op": "conv", "in": ["V"], "w": "WVc"},
                       {"id": "V0", "op": "slice", "in": ["Vc"], "start": 0, "stop": 1},
                       {"id": "V1", "op": "slice", "in": ["Vc"], "start": 1, "stop": 2},
                       {"id": "V2", "op": "slice", "in": ["Vc"], "start": 2, "stop": 3},
                       {"id": "V3", "op": "slice", "in": ["Vc"], "start": 3, "stop": 4},
                       {"id": "Va", "op": "max2", "in": ["V0", "V1"]},
                       {"id": "Vb", "op": "max2", "in": ["V2", "V3"]},
                       {"id": "Vm", "op": "max2", "in": ["Va", "Vb"]},
                       {"id": "Vp", "op": "mul", "in": ["Vm", "Fe"]},
                       {"id": "Vn", "op": "max2", "in": ["Vp", "Gm"]}],
              "next": {"V": "Vn"}})

    Ws = np.zeros((4, 1, 3, 3))
    for k, (di, dj) in enumerate(DIRS): Ws[k, 0, 1 - di, 1 - dj] = 1
    cv("stc", "cp14", Ws)
    nd("std", "sub", ["stc", "cp14"]); nd("st", "relu", ["std"])
    nd("stV", "mul", ["st", "VL.V"]); nd("sc", "gmax", ["stV"])

    nd("g3", "mul", [x3, "nTHg"]); nd("grab", "gmax", ["g3"])
    nd("ef", "gmax", [x11])
    nd("dp", "mul", ["Cb", "THg"]); nd("drop", "gmax", ["dp"])
    nd("ga", "vadd", ["grab", "ef"])
    dn("omk", "kv", [[-1.0]], [1.0])
    nd("gak", "vmul", ["ga", "omk"]); nd("drk", "vmul", ["drop", "kv"])
    nd("a5", "vadd", ["gak", "drk"])
    nd("lv", "vconcat", ["sc", "a5"])
    Wo = np.zeros((8, 5))
    for k in range(4): Wo[1 + k, k] = 10
    Wo[5, 4] = 20
    dn("logits", "lv", Wo, [-100, 0, 0, 0, 0, 0, -100, -100])
    cv("click", "cp14", np.zeros((1, 1, 1, 1)), [0.0])
    return {"graph": G, "memory_size": 256, "params": P}

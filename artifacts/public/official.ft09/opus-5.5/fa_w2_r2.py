import numpy as np

def build():
    P = {}
    G = []
    dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

    def shift_w(C, k, sr, sc):
        w = np.zeros((C, C, k, k), np.float32)
        r = k // 2
        for c in range(C):
            w[c, c, r - sr, r - sc] = 1.0
        return w

    # F: [v, m0, m23, zc, bg4, bad]
    WF = np.zeros((6, 16, 1, 1), np.float32)
    for c in range(16):
        WF[0, c, 0, 0] = c
    WF[1, 0, 0, 0] = 1
    WF[2, 2, 0, 0] = 1
    WF[2, 3, 0, 0] = 1
    for c in (0, 2, 3):
        WF[3, c, 0, 0] = 1
    WF[4, 4, 0, 0] = 1
    for c in (0, 2, 3, 4, 5, 6):
        WF[5, c, 0, 0] = 1
    P["WF"] = WF
    G.append({"id": "F", "op": "conv", "in": ["x0"], "w": "WF"})
    G.append({"id": "F4", "op": "slice", "in": ["F"], "start": 0, "stop": 4})

    s6s, s8s = [], []
    for i, (dr, dc) in enumerate(dirs):
        P["Ws3_%d" % i] = shift_w(4, 7, 3 * dr, 3 * dc)
        P["Ws2_%d" % i] = shift_w(4, 5, 2 * dr, 2 * dc)
        G.append({"id": "S3_%d" % i, "op": "conv", "in": ["F4"], "w": "Ws3_%d" % i})
        G.append({"id": "S6_%d" % i, "op": "conv", "in": ["S3_%d" % i], "w": "Ws3_%d" % i})
        G.append({"id": "S8_%d" % i, "op": "conv", "in": ["S6_%d" % i], "w": "Ws2_%d" % i})
        s6s.append("S6_%d" % i)
        s8s.append("S8_%d" % i)
    G.append({"id": "X", "op": "concat", "in": ["F"] + s6s + s8s})  # 70 ch

    WD = np.zeros((8, 70, 1, 1), np.float32)
    for d in range(8):
        WD[d, 38 + 4 * d, 0, 0] = 1
        WD[d, 0, 0, 0] = -1
    P["WD"] = WD
    G.append({"id": "D", "op": "conv", "in": ["X"], "w": "WD"})
    G.append({"id": "AD", "op": "abs", "in": ["D"]})
    WE = np.zeros((8, 8, 1, 1), np.float32)
    for d in range(8):
        WE[d, d, 0, 0] = -1
    P["WE"] = WE
    P["bE"] = np.ones(8, np.float32)
    G.append({"id": "Ep", "op": "conv", "in": ["AD"], "w": "WE", "b": "bE"})
    G.append({"id": "E", "op": "relu", "in": ["Ep"]})
    G.append({"id": "Y", "op": "concat", "in": ["X", "E"]})  # 78 ch

    WP = np.zeros((16, 78, 1, 1), np.float32)
    bP = np.zeros(16, np.float32)
    for d in range(8):
        WP[2 * d, 6 + 4 * d + 1, 0, 0] = 1
        WP[2 * d, 38 + 4 * d + 3, 0, 0] = -1
        WP[2 * d, 70 + d, 0, 0] = -1
        WP[2 * d + 1, 6 + 4 * d + 2, 0, 0] = 1
        WP[2 * d + 1, 38 + 4 * d + 3, 0, 0] = -1
        WP[2 * d + 1, 70 + d, 0, 0] = 1
        bP[2 * d + 1] = -1
    P["WP"] = WP
    P["bP"] = bP
    G.append({"id": "Pp", "op": "conv", "in": ["Y"], "w": "WP", "b": "bP"})
    G.append({"id": "R", "op": "relu", "in": ["Pp"]})

    # plain clickable tile test
    WU = np.zeros((2, 6, 7, 7), np.float32)
    WU[0, 0, 3 - 2, 3] = 1   # v at q-(2,0)
    WU[1, 4, 3 - 3, 3] = 1   # bg4 at q-(3,0)
    P["WU"] = WU
    G.append({"id": "U", "op": "conv", "in": ["F"], "w": "WU"})
    G.append({"id": "Z", "op": "concat", "in": ["F", "U"]})  # 8 ch
    WDT = np.zeros((1, 8, 1, 1), np.float32)
    WDT[0, 0, 0, 0] = 1
    WDT[0, 6, 0, 0] = -1
    P["WDT"] = WDT
    G.append({"id": "DT", "op": "conv", "in": ["Z"], "w": "WDT"})
    G.append({"id": "ADT", "op": "abs", "in": ["DT"]})
    P["WET"] = -np.ones((1, 1, 1, 1), np.float32)
    P["bET"] = np.ones(1, np.float32)
    G.append({"id": "EqTp", "op": "conv", "in": ["ADT"], "w": "WET", "b": "bET"})
    G.append({"id": "EqT", "op": "relu", "in": ["EqTp"]})

    # NTi detection: count colour-6 cells at +-2 offsets around q
    WA = np.zeros((1, 16, 5, 5), np.float32)
    for i in (0, 2, 4):
        for j in (0, 2, 4):
            if not (i == 2 and j == 2):
                WA[0, 6, i, j] = 1
    P["WA"] = WA
    G.append({"id": "A6", "op": "conv", "in": ["x0"], "w": "WA"})
    G.append({"id": "Z2", "op": "concat", "in": ["Z", "EqT", "A6"]})  # 10 ch

    WT = np.zeros((1, 10, 1, 1), np.float32)
    WT[0, 8, 0, 0] = 1    # uniform
    WT[0, 7, 0, 0] = 1    # background above
    WT[0, 5, 0, 0] = -1   # bad colour
    P["WT"] = WT
    P["bT"] = -np.ones(1, np.float32)
    G.append({"id": "Tvp", "op": "conv", "in": ["Z2"], "w": "WT", "b": "bT"})
    G.append({"id": "Tv", "op": "relu", "in": ["Tvp"]})

    WN = np.zeros((1, 10, 1, 1), np.float32)
    WN[0, 9, 0, 0] = 1
    WN[0, 7, 0, 0] = 8
    WN[0, 5, 0, 0] = -8
    P["WN"] = WN
    P["bN"] = np.array([-8.0], np.float32)
    G.append({"id": "Nvp", "op": "conv", "in": ["Z2"], "w": "WN", "b": "bN"})
    G.append({"id": "Nr", "op": "relu", "in": ["Nvp"]})
    P["WNc"] = -np.ones((1, 1, 1, 1), np.float32)
    P["bNc"] = np.ones(1, np.float32)
    G.append({"id": "Ncp", "op": "conv", "in": ["Nr"], "w": "WNc", "b": "bNc"})
    G.append({"id": "Nc", "op": "relu", "in": ["Ncp"]})

    G.append({"id": "Q", "op": "concat", "in": ["R", "Tv", "Nc"]})  # 18 ch
    WC = np.zeros((2, 18, 1, 1), np.float32)
    WC[0, :16, 0, 0] = 1
    WC[0, 16, 0, 0] = 10
    WC[1, :16, 0, 0] = 50
    WC[1, 17, 0, 0] = -1000
    P["WC"] = WC
    P["bC"] = np.array([-10.0, 0.0], np.float32)
    G.append({"id": "clickp", "op": "conv", "in": ["Q"], "w": "WC", "b": "bC"})
    G.append({"id": "clickr", "op": "relu", "in": ["clickp"]})
    P["WS"] = np.ones((1, 2, 1, 1), np.float32)
    G.append({"id": "click", "op": "conv", "in": ["clickr"], "w": "WS"})

    G.append({"id": "gm", "op": "gmax", "in": ["click"]})
    P["WL"] = np.zeros((8, 1), np.float32)
    bL = np.zeros(8, np.float32)
    bL[6] = 1.0
    P["bL"] = bL
    G.append({"id": "logits", "op": "dense", "in": ["gm"], "w": "WL", "b": "bL"})

    return {"graph": G, "memory_size": 0, "params": P}

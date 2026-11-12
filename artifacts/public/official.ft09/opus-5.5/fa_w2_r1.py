import numpy as np

def build():
    K = [1, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    P = {}
    W = np.zeros((13, 16, 5, 5), np.float32)
    for d, (dy, dx) in enumerate(D):
        W[d, 0, 2 + 2 * dy, 2 + 2 * dx] = 1.0
        for c in (0, 2, 3):
            W[8, c, 2 + 2 * dy, 2 + 2 * dx] = 1.0
    for c in K:
        W[9, c, 2, 2] = float(c)
        W[10, c, 2, 2] = 1.0
        W[11, c, 2, 4] = float(c)
    W[12, 6, 2, 4] = 1.0
    P["Wc1"] = W

    WB = np.zeros((1, 13, 1, 1), np.float32); WB[0, 8] = 1; WB[0, 10] = 1
    P["WB"] = WB; P["bB"] = np.array([-8.0], np.float32)
    Wdv = np.zeros((1, 13, 1, 1), np.float32); Wdv[0, 9] = 1; Wdv[0, 11] = -1
    P["Wdv"] = Wdv
    P["Wn1"] = -np.ones((1, 1, 1, 1), np.float32); P["bn1"] = np.ones(1, np.float32)

    # shift stages: read [B, v, z_d] from q - 8d
    Ws1 = np.zeros((24, 10, 7, 7), np.float32)
    Ws2 = np.zeros((24, 24, 7, 7), np.float32)
    Ws3 = np.zeros((24, 24, 5, 5), np.float32)
    SelB = np.zeros((8, 24, 1, 1), np.float32)
    SelV = np.zeros((8, 24, 1, 1), np.float32)
    SelZ = np.zeros((8, 24, 1, 1), np.float32)
    for d, (dy, dx) in enumerate(D):
        srcs = [0, 1, 2 + d]
        for k in range(3):
            o = 3 * d + k
            Ws1[o, srcs[k], 3 - 3 * dy, 3 - 3 * dx] = 1.0
            Ws2[o, o, 3 - 3 * dy, 3 - 3 * dx] = 1.0
            Ws3[o, o, 2 - 2 * dy, 2 - 2 * dx] = 1.0
        SelB[d, 3 * d] = 1.0
        SelV[d, 3 * d + 1] = 1.0
        SelZ[d, 3 * d + 2] = 1.0
    P["Ws1"] = Ws1; P["Ws2"] = Ws2; P["Ws3"] = Ws3
    P["SelB"] = SelB; P["SelV"] = SelV; P["SelZ"] = SelZ
    P["Wn8"] = (-np.eye(8, dtype=np.float32)).reshape(8, 8, 1, 1)
    P["bn8"] = np.ones(8, np.float32)
    P["Wsum"] = np.ones((1, 8, 1, 1), np.float32)
    P["Wp"] = np.full((1, 1, 1, 1), 10.0, np.float32); P["bp"] = np.ones(1, np.float32)
    P["Wl"] = np.zeros((8, 1), np.float32)
    bl = np.zeros(8, np.float32); bl[6] = 1.0
    P["bl"] = bl

    g = [
        {"id": "c1", "op": "conv", "in": ["x0"], "w": "Wc1"},
        {"id": "Bl", "op": "conv", "in": ["c1"], "w": "WB", "b": "bB"},
        {"id": "B", "op": "relu", "in": ["Bl"]},
        {"id": "dv", "op": "conv", "in": ["c1"], "w": "Wdv"},
        {"id": "adv", "op": "abs", "in": ["dv"]},
        {"id": "e1l", "op": "conv", "in": ["adv"], "w": "Wn1", "b": "bn1"},
        {"id": "e1", "op": "relu", "in": ["e1l"]},
        {"id": "six", "op": "slice", "in": ["c1"], "start": 12, "stop": 13},
        {"id": "inK", "op": "slice", "in": ["c1"], "start": 10, "stop": 11},
        {"id": "v", "op": "slice", "in": ["c1"], "start": 9, "stop": 10},
        {"id": "z", "op": "slice", "in": ["c1"], "start": 0, "stop": 8},
        {"id": "s", "op": "add", "in": ["e1", "six"]},
        {"id": "T", "op": "min2", "in": ["inK", "s"]},
        {"id": "N6", "op": "min2", "in": ["six", "T"]},
        {"id": "X1", "op": "concat", "in": ["B", "v", "z"]},
        {"id": "s1", "op": "conv", "in": ["X1"], "w": "Ws1"},
        {"id": "s2", "op": "conv", "in": ["s1"], "w": "Ws2"},
        {"id": "s3", "op": "conv", "in": ["s2"], "w": "Ws3"},
        {"id": "Bs", "op": "conv", "in": ["s3"], "w": "SelB"},
        {"id": "vb", "op": "conv", "in": ["s3"], "w": "SelV"},
        {"id": "zs", "op": "conv", "in": ["s3"], "w": "SelZ"},
        {"id": "dd", "op": "sub", "in": ["vb", "v"]},
        {"id": "ad", "op": "abs", "in": ["dd"]},
        {"id": "el", "op": "conv", "in": ["ad"], "w": "Wn8", "b": "bn8"},
        {"id": "e", "op": "relu", "in": ["el"]},
        {"id": "xz", "op": "sub", "in": ["zs", "e"]},
        {"id": "xa", "op": "abs", "in": ["xz"]},
        {"id": "m1", "op": "min2", "in": ["xa", "Bs"]},
        {"id": "m2", "op": "min2", "in": ["m1", "T"]},
        {"id": "vs", "op": "conv", "in": ["m2"], "w": "Wsum"},
        {"id": "pf", "op": "conv", "in": ["N6"], "w": "Wp", "b": "bp"},
        {"id": "click", "op": "mul", "in": ["vs", "pf"]},
        {"id": "gm", "op": "gmax", "in": ["click"]},
        {"id": "logits", "op": "dense", "in": ["gm"], "w": "Wl", "b": "bl"},
    ]
    return {"graph": g, "memory_size": 0, "params": P}

import numpy as np

def build():
    P = {}
    Wp = np.zeros((5, 3, 1, 1), np.float32)
    Wp[0, :, 0, 0] = [-63, 0, 63]
    Wp[1, :, 0, 0] = [0, -63, 63]
    Wp[2, :, 0, 0] = [0, 126, -125]
    Wp[3, :, 0, 0] = [63, 0, 0]
    Wp[4, :, 0, 0] = [0, 63, 0]
    P["Wp"] = Wp
    Wb1 = np.zeros((4, 2, 1, 1), np.float32)
    Wb1[0, 0] = 1; Wb1[1, 0] = 1; Wb1[2, 1] = 1; Wb1[3, 1] = 1
    P["Wb1"] = Wb1
    P["bb1"] = np.array([-7, -8, -7, -8], np.float32)
    Wb2 = np.zeros((2, 4, 1, 1), np.float32)
    Wb2[0, 0] = 1; Wb2[0, 1] = -1; Wb2[1, 2] = 1; Wb2[1, 3] = -1
    P["Wb2"] = Wb2

    NU = 23
    rows1 = []
    def r(coef, b):
        w = np.zeros(NU, np.float32)
        for k, c in coef.items():
            w[k] = c
        rows1.append((w, b))
    def pair(coef, b):
        r(coef, b); r(coef, b - 1)
    pair({2: 2, 0: -2}, -2.5)
    pair({0: 2, 2: -2}, -2.5)
    pair({3: 2, 1: -2}, -2.5)
    pair({1: 2, 3: -2}, -2.5)
    pair({4: 0.1}, -2.0)
    pair({2: 1, 3: 1}, 0.0)
    pair({9: 1}, -6.0)
    pair({10: 1}, -6.0)
    pair({0: -1, 5: 1, 7: 2}, -69.0)
    pair({0: 1, 5: -1, 7: -2}, 61.0)
    pair({1: -1, 6: 1, 8: 2}, -69.0)
    pair({1: 1, 6: -1, 8: -2}, 61.0)
    r({11: 1, 13: -1}, 0); r({11: -1, 13: 1}, 0)
    r({12: 1, 14: -1}, 0); r({12: -1, 14: 1}, 0)
    r({16: 1, 17: 1, 18: 1, 19: 1}, 0)
    P["W1"] = np.stack([w for w, b in rows1]).astype(np.float32)
    P["b1"] = np.array([b for w, b in rows1], np.float32)
    n1 = len(rows1)
    W2 = np.zeros((14, n1), np.float32)
    for k in range(12):
        W2[k, 2 * k] = 1; W2[k, 2 * k + 1] = -1
    W2[12, 24:28] = 1
    W2[13, 28] = 1
    P["W2"] = W2
    W3 = np.zeros((14, 14), np.float32); b3 = np.zeros(14, np.float32)
    for i, d in enumerate([0, 1, 2, 3]):
        W3[2 * i, d] = 1; W3[2 * i, 4] = -1; W3[2 * i, 6] = -1; W3[2 * i, 7] = -1; W3[2 * i, 5] = 1; b3[2 * i] = -1
        W3[2 * i + 1, d] = 1; W3[2 * i + 1, 4] = 1; W3[2 * i + 1, 6] = -1; W3[2 * i + 1, 7] = -1; W3[2 * i + 1, 5] = 1; b3[2 * i + 1] = -2
    W3[8, 8] = 1; W3[8, 6] = 1; b3[8] = -1
    W3[9, 9] = 1; W3[9, 6] = 1; b3[9] = -1
    W3[10, 10] = 1; W3[10, 7] = 1; b3[10] = -1
    W3[11, 11] = 1; W3[11, 7] = 1; b3[11] = -1
    W3[12, 13] = 1; W3[12, 12] = -1
    W3[13, 5] = -1; W3[13, 6] = -1; W3[13, 7] = -1; b3[13] = 1
    P["W3"] = W3; P["b3"] = b3
    Wo = np.zeros((8, 14), np.float32); bo = np.zeros(8, np.float32)
    bo[0] = -5; bo[6] = -5; bo[7] = -5
    Wo[1, 2] = 2; Wo[1, 1] = 2; Wo[1, 9] = 2
    Wo[2, 0] = 2; Wo[2, 3] = 2; Wo[2, 8] = 2
    Wo[3, 4] = 1; Wo[3, 7] = 1; Wo[3, 11] = 2
    Wo[4, 6] = 1; Wo[4, 5] = 1; Wo[4, 10] = 2
    Wo[5, 12] = 20; Wo[5, 13] = 3; bo[5] = 0.5
    P["Wo"] = Wo; P["bo"] = bo

    g = [
        {"id": "pc", "op": "conv", "in": ["coords"], "w": "Wp"},
        {"id": "pos", "op": "slice", "in": ["pc"], "start": 0, "stop": 2},
        {"id": "colm", "op": "slice", "in": ["pc"], "start": 2, "stop": 3},
        {"id": "pos2", "op": "slice", "in": ["pc"], "start": 3, "stop": 5},
        {"id": "colr", "op": "relu", "in": ["colm"]},
        {"id": "x11", "op": "slice", "in": ["x0"], "start": 11, "stop": 12},
        {"id": "tm", "op": "sub", "in": ["x11", "colr"]},
        {"id": "t", "op": "relu", "in": ["tm"]},
        {"id": "x4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "tp", "op": "mul", "in": ["t", "pos"]},
        {"id": "gt", "op": "gmax", "in": ["tp"]},
        {"id": "yp", "op": "mul", "in": ["x4", "pos"]},
        {"id": "gy", "op": "gmax", "in": ["yp"]},
        {"id": "x10", "op": "slice", "in": ["x0"], "start": 10, "stop": 11},
        {"id": "rs", "op": "rowsum", "in": ["x10"]},
        {"id": "cs", "op": "colsum", "in": ["x10"]},
        {"id": "hm", "op": "gmax", "in": ["rs"]},
        {"id": "tq", "op": "mul", "in": ["t", "pos2"]},
        {"id": "tmx", "op": "gmax", "in": ["tq"]},
        {"id": "rc", "op": "concat", "in": ["rs", "cs"]},
        {"id": "bc1", "op": "conv", "in": ["rc"], "w": "Wb1", "b": "bb1"},
        {"id": "bc1r", "op": "relu", "in": ["bc1"]},
        {"id": "hv", "op": "conv", "in": ["bc1r"], "w": "Wb2"},
        {"id": "hvp", "op": "mul", "in": ["hv", "pos"]},
        {"id": "mvh", "op": "gmax", "in": ["hvp"]},
        {"id": "xz", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        {"id": "hb", "op": "slice", "in": ["hv"], "start": 0, "stop": 1},
        {"id": "vb", "op": "slice", "in": ["hv"], "start": 1, "stop": 2},
        {"id": "dh", "op": "mul", "in": ["xz", "hb"]},
        {"id": "dhr", "op": "rowsum", "in": ["dh"]},
        {"id": "shc", "op": "gmax", "in": ["dhr"]},
        {"id": "dv", "op": "mul", "in": ["xz", "vb"]},
        {"id": "dvc", "op": "colsum", "in": ["dv"]},
        {"id": "svc", "op": "gmax", "in": ["dvc"]},
        {"id": "zp", "op": "mul", "in": ["xz", "pos"]},
        {"id": "z0", "op": "gmax", "in": ["zp"]},
        {"id": "v", "op": "vconcat", "in": ["gt", "gy", "hm", "tmx", "mvh", "shc", "svc", "z0"]},
        {"id": "u", "op": "vconcat", "in": ["v", "mem", "aprev"]},
        {"id": "a1", "op": "dense", "in": ["u"], "w": "W1", "b": "b1"},
        {"id": "a1r", "op": "vrelu", "in": ["a1"]},
        {"id": "a2", "op": "dense", "in": ["a1r"], "w": "W2"},
        {"id": "a3", "op": "dense", "in": ["a2"], "w": "W3", "b": "b3"},
        {"id": "a3r", "op": "vrelu", "in": ["a3"]},
        {"id": "logits", "op": "dense", "in": ["a3r"], "w": "Wo", "b": "bo"},
        {"id": "mem_next", "op": "vslice", "in": ["v"], "start": 11, "stop": 13},
        {"id": "click", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
    ]
    return {"graph": g, "memory_size": 2, "params": P}

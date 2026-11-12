import numpy as np

def build():
    f = np.float32
    dirs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    P = {}
    Wv = np.zeros((1, 16, 1, 1), f)
    for c in range(16):
        Wv[0, c, 0, 0] = c + 1
    P["Wv"] = Wv
    Wa = np.zeros((9, 16, 5, 5), f); ba = np.zeros(9, f)
    for i, (dy, dx) in enumerate(dirs):
        for c in (0, 2, 3):
            Wa[0, c, 2 + 2 * dy, 2 + 2 * dx] = 1
        Wa[1 + i, 0, 2 + 2 * dy, 2 + 2 * dx] = 1
    for c in (0, 2, 3, 4, 5):
        Wa[0, c, 2, 2] = -8
    ba[0] = -7
    P["Wa"] = Wa; P["ba"] = ba
    Wz = np.zeros((8, 9, 1, 1), f); bz = -np.ones(8, f)
    for i in range(8):
        Wz[i, 0, 0, 0] = 1; Wz[i, 1 + i, 0, 0] = 1
    P["Wz"] = Wz; P["bz"] = bz
    W1 = np.zeros((24, 18, 7, 7), f)
    W2 = np.zeros((24, 24, 7, 7), f)
    W3 = np.zeros((24, 24, 5, 5), f)
    for i, (dy, dx) in enumerate(dirs):
        k1 = (3 - 3 * dy, 3 - 3 * dx)
        W1[3 * i, 0, k1[0], k1[1]] = 1
        W1[3 * i + 1, 10 + i, k1[0], k1[1]] = 1
        W1[3 * i + 2, 9, k1[0], k1[1]] = 1
        for j in range(3):
            W2[3 * i + j, 3 * i + j, k1[0], k1[1]] = 1
            W3[3 * i + j, 3 * i + j, 2 - 2 * dy, 2 - 2 * dx] = 1
    P["W1"] = W1; P["W2"] = W2; P["W3"] = W3
    Wd = np.zeros((8, 25, 1, 1), f)
    for i in range(8):
        Wd[i, 3 * i + 2, 0, 0] = 1; Wd[i, 24, 0, 0] = -1
    P["Wd"] = Wd
    We = np.zeros((8, 8, 1, 1), f)
    for i in range(8):
        We[i, i, 0, 0] = -1
    P["We"] = We; P["be"] = np.ones(8, f)
    Wp = np.zeros((16, 32, 1, 1), f)
    for i in range(8):
        Wp[i, 3 * i, 0, 0] = 1; Wp[i, 24 + i, 0, 0] = 1
        Wp[8 + i, 3 * i + 1, 0, 0] = 1; Wp[8 + i, 24 + i, 0, 0] = 1
    P["Wp"] = Wp; P["bp"] = -np.ones(16, f)
    Wvi = np.zeros((1, 40, 1, 1), f)
    for i in range(8):
        Wvi[0, 3 * i + 1, 0, 0] = 1
        Wvi[0, 24 + i, 0, 0] = 1
        Wvi[0, 32 + i, 0, 0] = -2
    P["Wvi"] = Wvi
    Wvc = np.zeros((4, 1, 5, 5), f)
    for i, (cy, cx) in enumerate(corners):
        Wvc[i, 0, 2, 2] = 1; Wvc[i, 0, 2 + 2 * cy, 2 + 2 * cx] = -1
    P["Wvc"] = Wvc
    Wg = np.zeros((2, 16, 7, 7), f)
    Wg[0, 4, 3, 0] = 1
    for c in (0, 4, 5):
        Wg[0, c, 3, 3] -= 1
    Wg[1, 6, 3, 5] = 1
    P["Wg"] = Wg
    Wcf = np.zeros((2, 6, 1, 1), f)
    Wcf[0, 0:4, 0, 0] = -1; Wcf[0, 4, 0, 0] = 1; Wcf[1, 5, 0, 0] = 10
    P["Wcf"] = Wcf; P["bcf"] = np.array([0, 1], f)
    Wl = np.zeros((8, 8), f)
    Wl[6, 6] = 1.0
    P["Wl"] = Wl

    g = [
        {"id": "v", "op": "conv", "in": ["x0"], "w": "Wv"},
        {"id": "A", "op": "conv", "in": ["x0"], "w": "Wa", "b": "ba"},
        {"id": "sA", "op": "relu", "in": ["A"]},
        {"id": "zl", "op": "conv", "in": ["sA"], "w": "Wz", "b": "bz"},
        {"id": "zs", "op": "relu", "in": ["zl"]},
        {"id": "src", "op": "concat", "in": ["sA", "v", "zs"]},
        {"id": "sh1", "op": "conv", "in": ["src"], "w": "W1"},
        {"id": "sh2", "op": "conv", "in": ["sh1"], "w": "W2"},
        {"id": "sh3", "op": "conv", "in": ["sh2"], "w": "W3"},
        {"id": "cat1", "op": "concat", "in": ["sh3", "v"]},
        {"id": "Dl", "op": "conv", "in": ["cat1"], "w": "Wd"},
        {"id": "aD", "op": "abs", "in": ["Dl"]},
        {"id": "el", "op": "conv", "in": ["aD"], "w": "We", "b": "be"},
        {"id": "e", "op": "relu", "in": ["el"]},
        {"id": "cat2", "op": "concat", "in": ["sh3", "e"]},
        {"id": "pl", "op": "conv", "in": ["cat2"], "w": "Wp", "b": "bp"},
        {"id": "pr", "op": "relu", "in": ["pl"]},
        {"id": "cat3", "op": "concat", "in": ["sh3", "pr"]},
        {"id": "viol", "op": "conv", "in": ["cat3"], "w": "Wvi"},
        {"id": "vc", "op": "conv", "in": ["v"], "w": "Wvc"},
        {"id": "avc", "op": "abs", "in": ["vc"]},
        {"id": "g2", "op": "conv", "in": ["x0"], "w": "Wg"},
        {"id": "cat4", "op": "concat", "in": ["avc", "g2"]},
        {"id": "cf", "op": "conv", "in": ["cat4"], "w": "Wcf", "b": "bcf"},
        {"id": "cfr", "op": "relu", "in": ["cf"]},
        {"id": "cell", "op": "slice", "in": ["cfr"], "start": 0, "stop": 1},
        {"id": "fac", "op": "slice", "in": ["cfr"], "start": 1, "stop": 2},
        {"id": "m1", "op": "mul", "in": ["cell", "viol"]},
        {"id": "click", "op": "mul", "in": ["m1", "fac"]},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wl"},
    ]
    return {"graph": g, "memory_size": 0, "params": P}

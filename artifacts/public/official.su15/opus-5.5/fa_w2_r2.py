import numpy as np

def build():
    P = {}
    d = 0.01
    W0 = np.zeros((8, 16, 1, 1))
    for i, c in enumerate([10, 6, 15, 11, 12, 8]):
        W0[i, c] = 1
    for c in (7, 13, 14):
        W0[6, c] = 1
    W0[7, 9] = 1
    P["W0"] = W0
    Wr = np.zeros((2, 3, 1, 1)); Wr[0, 0] = 1000; Wr[1, 0] = 1000
    P["Wr"] = Wr
    P["br"] = np.array([-1000 * 9.5 / 63, -1000 * 62.5 / 63])
    P["Wdz"] = np.zeros((1, 8)); P["bdz"] = np.array([d])
    P["Wlz"] = np.zeros((8, 8)); bl = np.zeros(8); bl[6] = 1.0; P["blz"] = bl
    Wsh = np.zeros((2, 1, 3, 3)); Wsh[0, 0, 1, 0] = 1; Wsh[1, 0, 1, 2] = 1
    Wsv = np.zeros((2, 1, 3, 3)); Wsv[0, 0, 0, 1] = 1; Wsv[1, 0, 2, 1] = 1
    P["Wsh"] = Wsh; P["Wsv"] = Wsv
    P["Wnr"] = np.full((1, 1, 1, 1), 1000.0); P["bnr"] = np.array([-1000 * (1 - 1.5 * d)])
    Wd = np.zeros((7, 7, 7, 7))
    for a in range(7):
        for b in range(7):
            if (a - 3) ** 2 + (b - 3) ** 2 <= 13:
                for c in range(7):
                    Wd[c, c, a, b] = 1
    P["Wd"] = Wd
    Wn = np.zeros((1, 7, 1, 1)); Wn[0, :6] = 1; P["Wn"] = Wn
    P["bm1"] = np.array([-1.0])
    Wa = np.zeros((6, 6, 3, 3)); Wb = np.zeros((6, 6, 3, 3)); Wt = np.zeros((6, 6, 7, 7))
    for c in range(6):
        Wa[c, c, 1, 1] = 2; Wa[c, c, 1, 0] = -2; Wa[c, c, 0, 1] = -2
        Wb[c, c] = 1
        for a in (0, 3, 6):
            for b in (0, 3, 6):
                Wt[c, c, a, b] = 1
    P["Wa"] = Wa; P["ba"] = np.full(6, -1.0)
    P["Wb"] = Wb; P["Wt"] = Wt; P["bt"] = np.full(6, -1.0)
    P["b9"] = np.full(6, -9.0)
    P["W2m"] = np.full((1, 6, 1, 1), 1.0 / 9)
    P["Ws6"] = np.ones((1, 6, 1, 1))
    Wc = np.zeros((1, 8, 1, 1)); Wc[0, :, 0, 0] = [10, -4, -3, 1, 20, 8, -8, 8]
    P["Wc"] = Wc; P["bc"] = np.array([-20.0])

    g = [
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "c0", "op": "conv", "in": ["x0"], "w": "W0"},
        {"id": "rr", "op": "conv", "in": ["coords"], "w": "Wr", "b": "br"},
        {"id": "rr1", "op": "relu", "in": ["rr"]},
        {"id": "rm", "op": "min2", "in": ["rr1", "one"]},
        {"id": "rm0", "op": "slice", "in": ["rm"], "start": 0, "stop": 1},
        {"id": "rm1", "op": "slice", "in": ["rm"], "start": 1, "stop": 2},
        {"id": "mask", "op": "sub", "in": ["rm0", "rm1"]},
        {"id": "fm", "op": "mul", "in": ["c0", "mask"]},
        {"id": "goal", "op": "slice", "in": ["fm"], "start": 7, "stop": 8},
        {"id": "dv", "op": "dense", "in": ["avail"], "w": "Wdz", "b": "bdz"},
        {"id": "G", "op": "loop", "T": 64,
         "state": [{"id": "h", "init": "goal"}],
         "body": [
             {"id": "s1", "op": "conv", "in": ["h"], "w": "Wsh"},
             {"id": "s1a", "op": "slice", "in": ["s1"], "start": 0, "stop": 1},
             {"id": "s1b", "op": "slice", "in": ["s1"], "start": 1, "stop": 2},
             {"id": "q1", "op": "max2", "in": ["h", "s1a"]},
             {"id": "q2", "op": "max2", "in": ["q1", "s1b"]},
             {"id": "s2", "op": "conv", "in": ["q2"], "w": "Wsv"},
             {"id": "s2a", "op": "slice", "in": ["s2"], "start": 0, "stop": 1},
             {"id": "s2b", "op": "slice", "in": ["s2"], "start": 1, "stop": 2},
             {"id": "q3", "op": "max2", "in": ["q2", "s2a"]},
             {"id": "q4", "op": "max2", "in": ["q3", "s2b"]},
             {"id": "q5", "op": "sub", "in": ["q4", "dv"]},
             {"id": "hn", "op": "max2", "in": ["q5", "goal"]}],
         "next": {"h": "hn"}},
        {"id": "nr", "op": "conv", "in": ["G.h"], "w": "Wnr", "b": "bnr"},
        {"id": "nr1", "op": "relu", "in": ["nr"]},
        {"id": "nr2", "op": "min2", "in": ["nr1", "one"]},
        {"id": "nf", "op": "sub", "in": ["one", "nr2"]},
        {"id": "fe", "op": "slice", "in": ["fm"], "start": 0, "stop": 7},
        {"id": "fe2", "op": "mul", "in": ["fe", "nf"]},
        {"id": "d1", "op": "conv", "in": ["fe2"], "w": "Wd"},
        {"id": "d1m", "op": "min2", "in": ["d1", "one"]},
        {"id": "d2", "op": "conv", "in": ["d1m"], "w": "Wd"},
        {"id": "D", "op": "min2", "in": ["d2", "one"]},
        {"id": "N", "op": "conv", "in": ["D"], "w": "Wn"},
        {"id": "Nm", "op": "min2", "in": ["N", "one"]},
        {"id": "En", "op": "slice", "in": ["D"], "start": 6, "stop": 7},
        {"id": "e1", "op": "conv", "in": ["fe"], "w": "Wd"},
        {"id": "e1m", "op": "min2", "in": ["e1", "one"]},
        {"id": "e2", "op": "conv", "in": ["e1m"], "w": "Wd"},
        {"id": "Da", "op": "min2", "in": ["e2", "one"]},
        {"id": "Nx0", "op": "conv", "in": ["Da"], "w": "Wn", "b": "bm1"},
        {"id": "Nx", "op": "relu", "in": ["Nx0"]},
        {"id": "fr", "op": "slice", "in": ["fm"], "start": 0, "stop": 6},
        {"id": "an", "op": "conv", "in": ["fr"], "w": "Wa", "b": "ba"},
        {"id": "an1", "op": "relu", "in": ["an"]},
        {"id": "k1", "op": "conv", "in": ["an1"], "w": "Wb"},
        {"id": "k2", "op": "conv", "in": ["k1"], "w": "Wt", "b": "bt"},
        {"id": "mg", "op": "relu", "in": ["k2"]},
        {"id": "Mg", "op": "conv", "in": ["mg"], "w": "Ws6"},
        {"id": "k2u", "op": "conv", "in": ["k1"], "w": "Wt"},
        {"id": "k3", "op": "conv", "in": ["k2u"], "w": "Wt", "b": "b9"},
        {"id": "k3r", "op": "relu", "in": ["k3"]},
        {"id": "m2s", "op": "conv", "in": ["k3r"], "w": "W2m"},
        {"id": "M2", "op": "min2", "in": ["m2s", "one"]},
        {"id": "feat", "op": "concat", "in": ["Nm", "N", "En", "G.h", "mask", "Mg", "Nx", "M2"]},
        {"id": "click", "op": "conv", "in": ["feat"], "w": "Wc", "b": "bc"},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wlz", "b": "blz"},
    ]
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

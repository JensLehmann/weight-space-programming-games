import numpy as np

def build():
    P = {}
    f = np.float32
    Wsel = np.zeros((3, 16, 1, 1), f)
    for i, c in enumerate([12, 14, 15]):
        Wsel[i, c] = 1.0
    P["Wsel"] = Wsel
    P["Wr1"] = np.array([63.0, 0.0, -1.0], f).reshape(1, 3, 1, 1)
    Wm12 = np.zeros((2, 4, 1, 1), f)
    Wm12[0, :, 0, 0] = [-1.0, 0.0, -63.0, 1.0]
    Wm12[1, :, 0, 0] = [-1.0, 0.0, 63.0, -62.0]
    P["Wm12"] = Wm12
    Whl = np.zeros((2, 3, 1, 1), f)
    Whl[0, :, 0, 0] = [0.0, -1e6, 5e5]
    Whl[1, :, 0, 0] = [0.0, 1e6, -5e5]
    P["Whl"] = Whl
    Wp = np.zeros((1, 22, 3, 3), f)
    Wp[0, 5, 1, 1] = 1.0
    Wp[0, 10, 1, 1] = 1.0
    Wp[0, 19:22, 1, 1] = 1.0
    Wp[0, 9, :, :] = -4.0
    P["Wp"] = Wp
    P["Wp2"] = Wp[:, :19].copy()
    We = np.zeros((2, 3, 1, 1), f)
    We[0, 0, 0, 0] = -63.0; We[0, 2, 0, 0] = 1.0
    We[1, 0, 0, 0] = 63.0;  We[1, 2, 0, 0] = -62.0
    P["We"] = We
    P["Wg"] = np.array([0.99, -4.0, -4.0], f).reshape(1, 3, 1, 1)
    P["Wk"] = np.array([-1e6, 1e6], f).reshape(2, 1, 1, 1)
    Wsh = np.zeros((8, 2, 3, 3), f)
    for k, (a, b) in enumerate([(0, 1), (2, 1), (1, 0), (1, 2)]):
        for c in range(2):
            Wsh[2 * k + c, c, a, b] = 1.0
    P["Wsh"] = Wsh
    WE = np.zeros((8, 2, 3, 3), f)
    for c in range(2):
        for d, (a, b) in enumerate([(2, 1), (0, 1), (1, 2), (1, 0)]):
            WE[4 * c + d, c, a, b] = 1.0
            WE[4 * c + d, c, 1, 1] = -1.0
    P["WE"] = WE
    WD = np.zeros((8, 2, 1, 1), f)
    WD[0:4, 1] = 1.0
    WD[4:8, 0] = 1.0
    P["WD"] = WD
    WR = np.zeros((8, 2, 1, 1), f)
    WR[0:4, 0] = 1.0
    WR[4:8, 1] = 1.0
    P["WR"] = WR
    P["Wq"] = (-1.0 / 0.99) * np.eye(8, dtype=f)
    P["bq"] = np.ones(8, f)
    Wa4 = np.zeros((4, 8), f)
    for i, j in enumerate([0, 1, 3, 2]):
        Wa4[i, j] = 1.0
    P["Wa4"] = Wa4
    Wb4 = np.zeros((4, 8), f)
    for i, j in enumerate([4, 5, 6, 7]):
        Wb4[i, j] = 1.0
    P["Wb4"] = Wb4
    P["Wnd"] = np.array([[0, 0, 1, -1]], f)
    P["Wvt"] = np.array([[1, 0, 0, 0, 1, 0, 0, 0]], f)
    Wdf = np.zeros((1, 13), f)
    Wdf[0, 0] = 1; Wdf[0, 1] = -1; Wdf[0, 3] = -1; Wdf[0, 4] = 1
    P["Wdf"] = Wdf
    Wsg = np.zeros((1, 13), f)
    Wsg[0, 9] = 1.0; Wsg[0, 8] = -1.0
    P["Wsg"] = Wsg
    P["Wh2"] = np.array([[4e4, 3.0, 0.0, 0.0]], f)
    P["Wcd"] = np.array([[0, 0, -1, 1, 0, 0, 1, -1]], f)
    p = 0.02
    K = 10.0
    c0 = 0.015
    Wo = np.zeros((8, 30), f)
    bo = np.full(8, -100.0, f)
    # z: cvec 0..7, hd 8, rv 9..16, nb 17..20, hnb 21, aprev 22..29
    Wo[1, 0] = 1; Wo[1, 4] = 1; Wo[1, 9 + 2] = -p; Wo[1, 17] = -K; Wo[1, 22 + 2] = -c0; bo[1] = 0
    Wo[2, 1] = 1; Wo[2, 5] = 1; Wo[2, 9 + 1] = -p; Wo[2, 18] = -K; Wo[2, 22 + 1] = -c0; bo[2] = 0
    for j in (2, 3, 6, 7):
        Wo[3, j] = 0.5; Wo[4, j] = 0.5
    Wo[3, 8] = -0.5; Wo[3, 9 + 4] = -p; Wo[3, 22 + 4] = -c0; bo[3] = 0
    Wo[3, 19] = -0.5 * K; Wo[3, 20] = -0.5 * K; Wo[3, 21] = 0.5 * K
    Wo[4, 8] = 0.5;  Wo[4, 9 + 3] = -p; Wo[4, 22 + 3] = -c0; bo[4] = 0
    Wo[4, 19] = -0.5 * K; Wo[4, 20] = -0.5 * K; Wo[4, 21] = -0.5 * K
    P["Wo"] = Wo
    P["bo"] = bo
    P["Wcl"] = np.zeros((1, 1, 1, 1), f)

    g = [
        {"id": "xc", "op": "concat", "in": ["x0", "coords"]},
        {"id": "X3", "op": "conv", "in": ["x0"], "w": "Wsel"},
        {"id": "r1", "op": "conv", "in": ["coords"], "w": "Wr1"},
        {"id": "a1", "op": "abs", "in": ["r1"]},
        {"id": "ac", "op": "concat", "in": ["a1", "coords"]},
        {"id": "m12l", "op": "conv", "in": ["ac"], "w": "Wm12"},
        {"id": "m12", "op": "relu", "in": ["m12l"]},
        {"id": "mA", "op": "slice", "in": ["m12"], "start": 0, "stop": 1},
        {"id": "mB", "op": "slice", "in": ["m12"], "start": 1, "stop": 2},
        {"id": "X3a", "op": "mul", "in": ["X3", "mA"]},
        {"id": "wl", "op": "gmax", "in": ["X3a"]},
        {"id": "X3b", "op": "mul", "in": ["X3", "mB"]},
        {"id": "wr", "op": "gmax", "in": ["X3b"]},
        {"id": "hll", "op": "conv", "in": ["coords"], "w": "Whl"},
        {"id": "hsg", "op": "sigmoid", "in": ["hll"]},
        {"id": "Lh", "op": "slice", "in": ["hsg"], "start": 0, "stop": 1},
        {"id": "Rh", "op": "slice", "in": ["hsg"], "start": 1, "stop": 2},
        {"id": "wA", "op": "mul", "in": ["Lh", "wl"]},
        {"id": "wB", "op": "mul", "in": ["Rh", "wr"]},
        {"id": "wh", "op": "add", "in": ["wA", "wB"]},
        {"id": "X3w", "op": "mul", "in": ["X3", "wh"]},
        {"id": "gp", "op": "sub", "in": ["X3", "X3w"]},
        {"id": "xg", "op": "concat", "in": ["xc", "gp"]},
        {"id": "ppre", "op": "conv", "in": ["xg"], "w": "Wp"},
        {"id": "edgl", "op": "conv", "in": ["coords"], "w": "We"},
        {"id": "edg", "op": "relu", "in": ["edgl"]},
        {"id": "pc", "op": "concat", "in": ["ppre", "edg"]},
        {"id": "pgl", "op": "conv", "in": ["pc"], "w": "Wg"},
        {"id": "Pg", "op": "relu", "in": ["pgl"]},
        {"id": "ppre2", "op": "conv", "in": ["xc"], "w": "Wp2"},
        {"id": "pc2", "op": "concat", "in": ["ppre2", "edg"]},
        {"id": "pgl2", "op": "conv", "in": ["pc2"], "w": "Wg"},
        {"id": "Pb", "op": "relu", "in": ["pgl2"]},
        {"id": "S", "op": "slice", "in": ["x0"], "start": 10, "stop": 11},
        {"id": "col", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "cnt", "op": "gmean", "in": ["S"]},
        {"id": "Sc", "op": "mul", "in": ["S", "col"]},
        {"id": "mcol", "op": "gmean", "in": ["Sc"]},
        {"id": "ccol", "op": "mul", "in": ["col", "cnt"]},
        {"id": "dif", "op": "sub", "in": ["ccol", "mcol"]},
        {"id": "dl", "op": "conv", "in": ["dif"], "w": "Wk"},
        {"id": "sg", "op": "sigmoid", "in": ["dl"]},
        {"id": "LR", "op": "mul", "in": ["sg", "S"]},
        {"id": "H", "op": "loop", "T": 256,
         "state": [{"id": "h", "init": "LR"}],
         "body": [
             {"id": "hs", "op": "conv", "in": ["h"], "w": "Wsh"},
             {"id": "s0", "op": "slice", "in": ["hs"], "start": 0, "stop": 2},
             {"id": "s1", "op": "slice", "in": ["hs"], "start": 2, "stop": 4},
             {"id": "s2", "op": "slice", "in": ["hs"], "start": 4, "stop": 6},
             {"id": "s3", "op": "slice", "in": ["hs"], "start": 6, "stop": 8},
             {"id": "m01", "op": "max2", "in": ["s0", "s1"]},
             {"id": "m23", "op": "max2", "in": ["s2", "s3"]},
             {"id": "mm", "op": "max2", "in": ["m01", "m23"]},
             {"id": "mg", "op": "mul", "in": ["mm", "Pg"]},
             {"id": "hn", "op": "max2", "in": ["mg", "LR"]},
         ],
         "next": {"h": "hn"}},
        {"id": "El", "op": "conv", "in": ["LR"], "w": "WE"},
        {"id": "E", "op": "relu", "in": ["El"]},
        {"id": "DD", "op": "conv", "in": ["H.h"], "w": "WD"},
        {"id": "ED", "op": "mul", "in": ["E", "DD"]},
        {"id": "vals", "op": "gmax", "in": ["ED"]},
        {"id": "LR8", "op": "conv", "in": ["LR"], "w": "WR"},
        {"id": "LD", "op": "mul", "in": ["LR8", "DD"]},
        {"id": "V8", "op": "gmax", "in": ["LD"]},
        {"id": "EP", "op": "mul", "in": ["E", "Pb"]},
        {"id": "pas", "op": "gmax", "in": ["EP"]},
        {"id": "q", "op": "dense", "in": ["pas"], "w": "Wq", "b": "bq"},
        {"id": "stay", "op": "vmul", "in": ["q", "V8"]},
        {"id": "qv", "op": "vadd", "in": ["vals", "stay"]},
        {"id": "A4", "op": "dense", "in": ["q"], "w": "Wa4"},
        {"id": "B4", "op": "dense", "in": ["q"], "w": "Wb4"},
        {"id": "nb", "op": "vmul", "in": ["A4", "B4"]},
        {"id": "ndf", "op": "dense", "in": ["nb"], "w": "Wnd"},
        {"id": "Vtot", "op": "dense", "in": ["V8"], "w": "Wvt"},
        {"id": "rv", "op": "vmul", "in": ["aprev", "Vtot"]},
        {"id": "LRc", "op": "mul", "in": ["LR", "col"]},
        {"id": "pos", "op": "gmean", "in": ["LRc"]},
        {"id": "v", "op": "vconcat", "in": ["pos", "mem", "aprev"]},
        {"id": "diff", "op": "dense", "in": ["v"], "w": "Wdf"},
        {"id": "sgn", "op": "dense", "in": ["v"], "w": "Wsg"},
        {"id": "ev", "op": "vmul", "in": ["diff", "sgn"]},
        {"id": "em", "op": "vconcat", "in": ["ev", "mem"]},
        {"id": "hp", "op": "dense", "in": ["em"], "w": "Wh2"},
        {"id": "hnew", "op": "vtanh", "in": ["hp"]},
        {"id": "mem_next", "op": "vconcat", "in": ["hnew", "pos"]},
        {"id": "hnb", "op": "vmul", "in": ["hnew", "ndf"]},
        {"id": "cd", "op": "dense", "in": ["qv"], "w": "Wcd"},
        {"id": "hd", "op": "vmul", "in": ["hnew", "cd"]},
        {"id": "z", "op": "vconcat", "in": ["qv", "hd", "rv", "nb", "hnb", "aprev"]},
        {"id": "logits", "op": "dense", "in": ["z"], "w": "Wo", "b": "bo"},
        {"id": "click", "op": "conv", "in": ["Pg"], "w": "Wcl"},
    ]
    return {"graph": g, "memory_size": 3, "params": P}

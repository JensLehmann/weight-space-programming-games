import numpy as np

def build():
    P = {}
    def diag(k2d, n=2):
        w = np.zeros((n, n) + k2d.shape, np.float32)
        for i in range(n):
            w[i, i] = k2d
        return w
    k = np.zeros((3, 3)); k[1, 1] = k[1, 2] = k[2, 1] = k[2, 2] = 1
    P["We2"] = diag(k); P["be2"] = np.full(2, -3.0)
    k = np.zeros((3, 3)); k[1, 1] = k[1, 0] = k[0, 1] = k[0, 0] = -1
    P["Wc2"] = diag(k); P["bone2"] = np.ones(2)
    P["We3"] = diag(np.ones((3, 3))); P["be3"] = np.full(2, -8.0)
    P["Wc3"] = diag(-np.ones((3, 3)))
    Wt = np.zeros((1, 16, 3, 3))
    for c in [1, 2, 9, 10, 11, 12, 15]:
        Wt[0, c] = 1.0
    P["Wt"] = Wt; P["bt"] = np.array([-8.0])
    P["Wneg"] = (-np.eye(2)).reshape(2, 2, 1, 1)
    Wtm = np.zeros((1, 16, 3, 3))
    for c in [1, 2, 9, 10, 15]:
        Wtm[0, c] = 1.0
    P["Wtm"] = Wtm
    P["Wgm"] = np.ones((1, 2, 3, 3))
    P["Wdil"] = diag(np.ones((3, 3)))
    P["W300"] = diag(np.array([[300.0]]))
    Wsh = np.zeros((6, 2, 3, 3)); Wsv = np.zeros((6, 2, 3, 3))
    for s in range(3):
        for c in range(2):
            Wsh[s * 2 + c, c, 1, s] = 1.0
            Wsv[s * 2 + c, c, s, 1] = 1.0
    P["Wsh"] = Wsh; P["Wsv"] = Wsv
    P["Wid"] = diag(np.array([[1.0]])); P["bm1"] = np.full(2, -1.0)
    P["We"] = np.array([[1.0, 1.0]])
    # improvement detection; mem = [e_prev, L, p]
    P["Wd"] = np.array([[1e5, -1e5, 0.0, 0.0]]); P["bd"] = np.array([-1.0])
    P["Wb"] = np.array([[-1.0]]); P["bb"] = np.array([1.0])
    P["WT"] = np.array([[0.0, 1.0, 0.0, 1.0]]); P["bT"] = np.array([-0.5])
    Wc = np.zeros((4, 3, 1, 1))
    Wc[0, 0] = 63; Wc[1, 1] = 63
    Wc[2, 1] = -63; Wc[2, 2] = 1
    Wc[3, 0] = -63; Wc[3, 2] = 2
    P["Wco"] = Wc
    P["Wkeep"] = np.array([0.0, 0.0, -1.0, -1.0]).reshape(1, 4, 1, 1); P["bkeep"] = np.array([1.0])
    Ws = np.zeros((2, 16, 1, 1)); Ws[0, 8] = 1; Ws[1, 14] = 1
    P["Wsel"] = Ws
    K = np.zeros((7, 7)); K[0:3, :] = -10.0; K[3, 0:3] = -10.0; K[3, 3] = 1.0
    P["Wan"] = diag(K)
    P["Wsum"] = np.ones((1, 2, 1, 1))
    P["Wr"] = np.array([4032.0, 63.0, 0.0]).reshape(1, 3, 1, 1)
    P["Wrev"] = np.array([-8064.0, -126.0, 4095.0]).reshape(1, 3, 1, 1)
    P["W4"] = np.array([4.0]).reshape(1, 1, 1, 1)
    P["Wq"] = np.array([8192.0, -1.0]).reshape(1, 2, 1, 1)
    P["Wcl"] = np.array([8192.0, 8192.0, -1.0, -1e5, -1e5]).reshape(1, 5, 1, 1)
    P["Wh1"] = np.array([[-1.0]]); P["bh1"] = np.array([1.0])
    P["Wdq"] = np.array([[-1.0, 1.0]])
    P["WL"] = np.array([[-1.0, 1.0]]); P["bL"] = np.array([8192.0])
    P["Wpn"] = np.array([[1.0, 1.0, -2.0]])
    P["Wl"] = np.zeros((8, 1)); bl = np.zeros(8); bl[6] = 1.0; P["bl"] = bl

    g = [
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "P", "op": "slice", "in": ["x0"], "start": 11, "stop": 13},
        {"id": "e2c", "op": "conv", "in": ["P"], "w": "We2", "b": "be2"},
        {"id": "e2", "op": "relu", "in": ["e2c"]},
        {"id": "c2c", "op": "conv", "in": ["e2"], "w": "Wc2", "b": "bone2"},
        {"id": "n2", "op": "relu", "in": ["c2c"]},
        {"id": "e3c", "op": "conv", "in": ["P"], "w": "We3", "b": "be3"},
        {"id": "e3", "op": "relu", "in": ["e3c"]},
        {"id": "c3c", "op": "conv", "in": ["e3"], "w": "Wc3", "b": "bone2"},
        {"id": "n3", "op": "relu", "in": ["c3c"]},
        {"id": "tc", "op": "conv", "in": ["x0"], "w": "Wt", "b": "bt"},
        {"id": "tr", "op": "relu", "in": ["tc"]},
        {"id": "f", "op": "gmax", "in": ["tr"]},
        {"id": "dn", "op": "sub", "in": ["n3", "n2"]},
        {"id": "fd", "op": "mul", "in": ["dn", "f"]},
        {"id": "ncov", "op": "add", "in": ["n2", "fd"]},
        {"id": "Gm", "op": "conv", "in": ["ncov"], "w": "Wneg", "b": "bone2"},
        {"id": "src", "op": "sub", "in": ["P", "Gm"]},
        {"id": "tmc", "op": "conv", "in": ["x0"], "w": "Wtm"},
        {"id": "gmc", "op": "conv", "in": ["Gm"], "w": "Wgm"},
        {"id": "msum", "op": "add", "in": ["tmc", "gmc"]},
        {"id": "mask", "op": "min2", "in": ["msum", "one"]},
        {"id": "sdc", "op": "conv", "in": ["src"], "w": "Wdil"},
        {"id": "sdm", "op": "min2", "in": ["sdc", "one"]},
        {"id": "sd1", "op": "mul", "in": ["sdm", "mask"]},
        {"id": "seed", "op": "conv", "in": ["sd1"], "w": "W300"},
        {"id": "W", "op": "loop", "T": 200,
         "state": [{"id": "V", "init": "seed"}],
         "body": [
             {"id": "Lh", "op": "conv", "in": ["V"], "w": "Wsh"},
             {"id": "La", "op": "slice", "in": ["Lh"], "start": 0, "stop": 2},
             {"id": "Lb", "op": "slice", "in": ["Lh"], "start": 2, "stop": 4},
             {"id": "Lc", "op": "slice", "in": ["Lh"], "start": 4, "stop": 6},
             {"id": "Lm1", "op": "max2", "in": ["La", "Lb"]},
             {"id": "Lmh", "op": "max2", "in": ["Lm1", "Lc"]},
             {"id": "Lv", "op": "conv", "in": ["Lmh"], "w": "Wsv"},
             {"id": "Ld", "op": "slice", "in": ["Lv"], "start": 0, "stop": 2},
             {"id": "Le", "op": "slice", "in": ["Lv"], "start": 2, "stop": 4},
             {"id": "Lf", "op": "slice", "in": ["Lv"], "start": 4, "stop": 6},
             {"id": "Lm2", "op": "max2", "in": ["Ld", "Le"]},
             {"id": "Lmx", "op": "max2", "in": ["Lm2", "Lf"]},
             {"id": "Ldec", "op": "conv", "in": ["Lmx"], "w": "Wid", "b": "bm1"},
             {"id": "Ldr", "op": "relu", "in": ["Ldec"]},
             {"id": "Ldm", "op": "mul", "in": ["Ldr", "mask"]},
             {"id": "Vn", "op": "max2", "in": ["Ldm", "seed"]}],
         "next": {"V": "Vn"}},
        {"id": "pr", "op": "mul", "in": ["Gm", "W.V"]},
        {"id": "E", "op": "gmean", "in": ["pr"]},
        {"id": "e", "op": "dense", "in": ["E"], "w": "We"},
        {"id": "em", "op": "vconcat", "in": ["e", "mem"]},
        {"id": "dl", "op": "dense", "in": ["em"], "w": "Wd", "b": "bd"},
        {"id": "g1", "op": "vrelu", "in": ["dl"]},
        {"id": "bl0", "op": "dense", "in": ["g1"], "w": "Wb", "b": "bb"},
        {"id": "bad", "op": "vrelu", "in": ["bl0"]},
        {"id": "tb", "op": "vconcat", "in": ["mem", "bad"]},
        {"id": "T", "op": "dense", "in": ["tb"], "w": "WT", "b": "bT"},
        {"id": "p", "op": "vslice", "in": ["mem"], "start": 2, "stop": 3},
        {"id": "pc", "op": "conv", "in": ["coords"], "w": "Wco"},
        {"id": "pcr", "op": "relu", "in": ["pc"]},
        {"id": "pen", "op": "slice", "in": ["pcr"], "start": 2, "stop": 4},
        {"id": "kc", "op": "conv", "in": ["pcr"], "w": "Wkeep", "b": "bkeep"},
        {"id": "keep", "op": "relu", "in": ["kc"]},
        {"id": "sel", "op": "conv", "in": ["x0"], "w": "Wsel"},
        {"id": "bmk", "op": "mul", "in": ["sel", "keep"]},
        {"id": "anc0", "op": "conv", "in": ["bmk"], "w": "Wan"},
        {"id": "anc1", "op": "relu", "in": ["anc0"]},
        {"id": "anc", "op": "conv", "in": ["anc1"], "w": "Wsum"},
        {"id": "r0", "op": "conv", "in": ["coords"], "w": "Wr"},
        {"id": "rq", "op": "conv", "in": ["coords"], "w": "Wrev"},
        {"id": "rqp", "op": "mul", "in": ["rq", "p"]},
        {"id": "r", "op": "add", "in": ["r0", "rqp"]},
        {"id": "rd", "op": "sub", "in": ["r", "T"]},
        {"id": "rd4", "op": "conv", "in": ["rd"], "w": "W4"},
        {"id": "hs", "op": "relu", "in": ["rd4"]},
        {"id": "h", "op": "min2", "in": ["hs", "one"]},
        {"id": "ah", "op": "mul", "in": ["anc", "h"]},
        {"id": "ar", "op": "mul", "in": ["anc", "r"]},
        {"id": "ahr", "op": "mul", "in": ["ah", "r"]},
        {"id": "cq1", "op": "concat", "in": ["ah", "ahr"]},
        {"id": "m1", "op": "conv", "in": ["cq1"], "w": "Wq"},
        {"id": "q1", "op": "gmax", "in": ["m1"]},
        {"id": "cq2", "op": "concat", "in": ["anc", "ar"]},
        {"id": "m2", "op": "conv", "in": ["cq2"], "w": "Wq"},
        {"id": "q2", "op": "gmax", "in": ["m2"]},
        {"id": "hq0", "op": "dense", "in": ["q1"], "w": "Wh1", "b": "bh1"},
        {"id": "hq1", "op": "vrelu", "in": ["hq0"]},
        {"id": "hb", "op": "dense", "in": ["hq1"], "w": "Wh1", "b": "bh1"},
        {"id": "qq", "op": "vconcat", "in": ["q1", "q2"]},
        {"id": "dq", "op": "dense", "in": ["qq"], "w": "Wdq"},
        {"id": "mq", "op": "vmul", "in": ["dq", "hb"]},
        {"id": "q2m", "op": "vconcat", "in": ["q2", "mq"]},
        {"id": "Ln", "op": "dense", "in": ["q2m"], "w": "WL", "b": "bL"},
        {"id": "pw", "op": "vmul", "in": ["p", "hq1"]},
        {"id": "pcat", "op": "vconcat", "in": ["p", "hq1", "pw"]},
        {"id": "pn", "op": "dense", "in": ["pcat"], "w": "Wpn"},
        {"id": "mem_next", "op": "vconcat", "in": ["e", "Ln", "pn"]},
        {"id": "cat", "op": "concat", "in": ["ah", "anc", "ar", "pen"]},
        {"id": "click", "op": "conv", "in": ["cat"], "w": "Wcl"},
        {"id": "logits", "op": "dense", "in": ["e"], "w": "Wl", "b": "bl"},
    ]
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 3, "params": params}

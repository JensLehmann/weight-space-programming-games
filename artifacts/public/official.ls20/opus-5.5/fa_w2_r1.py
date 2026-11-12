import numpy as np

def build():
    P = {}; G = []
    def n(id_, op, ins, **kw):
        d = {"id": id_, "op": op, "in": ins}; d.update(kw); G.append(d); return d
    def par(name, arr):
        P[name] = np.asarray(arr, np.float32); return name
    n("one", "slice", ["coords"], start=2, stop=3)
    w = np.zeros((1, 16, 1, 1)); w[0, 4] = -1
    n("free0", "conv", ["x0"], w=par("Wfree", w), b=par("b_one", [1.0]))
    w = np.zeros((1, 16, 3, 3)); w[0, 0] = -1; w[0, 1, 1, 1] = 1
    n("pu0", "conv", ["x0"], w=par("Wpu", w))
    n("pu1", "relu", ["pu0"])
    n("pu2", "conv", ["pu1"], w=par("Wpd", np.ones((1, 1, 7, 7))))
    n("pu3", "min2", ["pu2", "one"])
    n("fr1", "sub", ["free0", "pu3"])
    n("free", "relu", ["fr1"])
    w = np.zeros((1, 16, 1, 1)); w[0, 5] = -1
    n("nf", "conv", ["x0"], w=par("Wnf", w), b="b_one")
    cols = [8, 9, 12, 14]
    w = np.zeros((4, 16, 1, 1))
    for i, c in enumerate(cols): w[i, c] = 1
    n("cs", "conv", ["x0"], w=par("Wcs", w))
    w = np.zeros((1, 3, 1, 1)); w[0, 0] = -63; w[0, 2] = 52
    n("rmr", "conv", ["coords"], w=par("Wrm", w))
    n("rmm", "min2", ["rmr", "one"])
    n("rm", "relu", ["rmm"])
    w = np.zeros((1, 16, 5, 5)); w[0, 12, 0:2, :] = 1; w[0, 9, 2:5, :] = 1
    n("pcr", "conv", ["x0"], w=par("Wpc", w), b=par("bpc", [-24.0]))
    n("Pc", "relu", ["pcr"])
    w = np.zeros((1, 16, 7, 7)); w[0, 5] = 1; w[0, 5, 2:5, 2:5] = 0
    n("dr", "conv", ["x0"], w=par("Wgd", w), b=par("bgd", [-34.0]))
    n("dr2", "relu", ["dr"])
    n("dr3", "min2", ["dr2", "one"])
    n("D", "mul", ["dr3", "rm"])
    w = np.zeros((9, 1, 3, 3))
    for a in range(3):
        for b in range(3): w[a * 3 + b, 0, a, b] = 1
    n("nfs", "conv", ["nf"], w=par("Wsh9", w))
    n("gbm", "mul", ["nfs", "D"]); n("gb", "gmax", ["gbm"])
    w = np.zeros((4, 4, 3, 3))
    for i in range(4): w[i, i] = 1
    n("cwin", "conv", ["cs"], w=par("Wcw", w))
    n("gcm", "mul", ["cwin", "D"]); n("gc", "gmax", ["gcm"])
    w = np.zeros((18, 3, 1, 1))
    for a in range(3):
        for b in range(3):
            k = a * 3 + b
            w[k, 0] = 63; w[k, 2] = -(55 + 2 * a)
            w[9 + k, 1] = 63; w[9 + k, 2] = -(3 + 2 * b)
    n("ind0", "conv", ["coords"], w=par("Wind", w))
    n("ind1", "abs", ["ind0"])
    w = np.zeros((9, 18, 1, 1))
    for k in range(9): w[k, k] = -1; w[k, 9 + k] = -1
    n("ind2", "conv", ["ind1"], w=par("Wind2", w), b=par("bind2", np.ones(9)))
    n("Ind", "relu", ["ind2"])
    n("hbm", "mul", ["Ind", "nf"]); n("hb", "gmax", ["hbm"])
    n("hreg", "conv", ["Ind"], w=par("Whr", np.ones((1, 9, 1, 1))))
    n("hcm", "mul", ["cs", "hreg"]); n("hc", "gmax", ["hcm"])
    n("kv", "vconcat", ["hb", "hc", "gb", "gc"])
    n("kq", "dense", ["kv"], w=par("Wsq", 10 * np.eye(26)), b=par("bsq", -5 * np.ones(26)))
    n("ks", "vsigmoid", ["kq"])
    w = np.zeros((26, 26))
    for i in range(13):
        w[i, i] = 1; w[i, 13 + i] = -1; w[13 + i, i] = -1; w[13 + i, 13 + i] = 1
    n("kd", "dense", ["ks"], w=par("Wdf", w))
    n("kr", "vrelu", ["kd"])
    n("mis", "dense", ["kr"], w=par("Wsum", np.ones((1, 26))))
    n("mt0", "dense", ["mis"], w=par("Wmt", [[-30.0]]), b=par("bmt", [15.0]))
    n("match", "vsigmoid", ["mt0"])
    n("blk", "slice", ["x0"], start=0, stop=1)
    n("Ms", "mul", ["blk", "rm"])
    n("gd", "sub", ["D", "Ms"])
    n("gdm", "mul", ["gd", "match"])
    n("src", "add", ["Ms", "gdm"])
    n("S", "conv", ["src"], w=par("W1000", [[[[1000.0]]]]))
    w = np.zeros((4, 1, 3, 3)); w[0, 0, 0, 1] = 1; w[1, 0, 2, 1] = 1; w[2, 0, 1, 0] = 1; w[3, 0, 1, 2] = 1
    par("Wnb", w); par("Wid", [[[[1.0]]]]); par("bm1", [-1.0])
    body = [
        {"id": "nb", "op": "conv", "in": ["v"], "w": "Wnb"},
        {"id": "n0", "op": "slice", "in": ["nb"], "start": 0, "stop": 1},
        {"id": "n1", "op": "slice", "in": ["nb"], "start": 1, "stop": 2},
        {"id": "n2", "op": "slice", "in": ["nb"], "start": 2, "stop": 3},
        {"id": "n3", "op": "slice", "in": ["nb"], "start": 3, "stop": 4},
        {"id": "m01", "op": "max2", "in": ["n0", "n1"]},
        {"id": "m23", "op": "max2", "in": ["n2", "n3"]},
        {"id": "mm", "op": "max2", "in": ["m01", "m23"]},
        {"id": "dec", "op": "conv", "in": ["mm"], "w": "Wid", "b": "bm1"},
        {"id": "mx", "op": "max2", "in": ["dec", "S"]},
        {"id": "mf", "op": "mul", "in": ["mx", "free"]},
        {"id": "vn", "op": "relu", "in": ["mf"]},
    ]
    G.append({"id": "L", "op": "loop", "T": 200, "state": [{"id": "v", "zeros": [1, 64, 64]}],
              "body": body, "next": {"v": "vn"}})
    w = np.zeros((4, 1, 7, 7)); w[0, 0, 0, 3] = 1; w[1, 0, 6, 3] = 1; w[2, 0, 3, 0] = 1; w[3, 0, 3, 6] = 1
    n("vs", "conv", ["L.v"], w=par("Wdir", w))
    n("vsm", "mul", ["vs", "Pc"])
    n("sc", "gmax", ["vsm"])
    n("sa", "vconcat", ["sc", "aprev"])
    w = np.zeros((8, 12)); bb = np.full(8, -1e4)
    rev = {0: 2, 1: 1, 2: 4, 3: 3}
    for d in range(4):
        w[1 + d, d] = 1.0; w[1 + d, 4 + rev[d]] = 0.3; bb[1 + d] = 0.0
    n("logits", "dense", ["sa"], w=par("Wo", w), b=par("bo", bb))
    n("click", "conv", ["x0"], w=par("Wc", np.zeros((1, 16, 1, 1))))
    return {"graph": G, "memory_size": 0, "params": P}

import numpy as np

def build():
    P = {}
    Whc = np.zeros((3, 16, 7, 7), np.float32)
    for k in range(3):
        Whc[0, 11, 3, k] = 1.0
        Whc[1, 11, 3, 4 + k] = 1.0
        Whc[2, 11, k, 3] = 1.0
        Whc[2, 11, 4 + k, 3] = 1.0
    P["Whc"] = Whc
    Wh1 = np.zeros((1, 19, 1, 1), np.float32)
    Wh1[0, 0:3, 0, 0] = 2.0
    Wh1[0, 3 + 12, 0, 0] = 2.0
    P["Wh1"] = Wh1
    P["bh1"] = np.array([-7.0], np.float32)
    Wmk = np.zeros((1, 16, 5, 5), np.float32)
    Wmk[0, 9, 2, 4] = 1.0; Wmk[0, 9, 2, 3] = -1.0
    Wmk[0, 9, 2, 0] = 1.0; Wmk[0, 9, 2, 1] = -1.0
    P["Wmk"] = Wmk
    P["W64"] = np.array([[64.0]], np.float32)
    P["bm1"] = np.array([-1.0], np.float32)
    Ws4 = np.zeros((1, 16, 7, 7), np.float32)
    Ws4[0, 9, 3, 3] = 1.0
    Ws4[0, 9, 6, 3] = 1.0
    P["Ws4"] = Ws4
    P["Wsr"] = np.zeros((2, 2, 3, 3), np.float32)
    P["Wsl"] = np.zeros((2, 2, 3, 3), np.float32)
    P["W3r"] = np.zeros((2, 2, 7, 7), np.float32)
    P["W3l"] = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        P["Wsr"][c, c, 1, 0] = 1.0
        P["Wsl"][c, c, 1, 2] = 1.0
        P["W3r"][c, c, 3, 0] = 1.0
        P["W3l"][c, c, 3, 6] = 1.0
    P["Wneg"] = np.array([[-1.0]], np.float32)
    P["bone"] = np.array([1.0], np.float32)
    W8 = np.zeros((1, 16, 1, 1), np.float32)
    W8[0, 8, 0, 0] = 1.0
    W8[0, 15, 0, 0] = 1.0
    P["W8"] = W8
    P["Wpos"] = np.array([4032.0, 63.0, 0.0], np.float32).reshape(1, 3, 1, 1)
    P["Wst"] = np.array([-2.0], np.float32).reshape(1, 1, 1, 1)
    P["Wdd"] = np.array([-1.0, -4096.0], np.float32).reshape(1, 2, 1, 1)
    P["bdd"] = np.array([8200.0], np.float32)
    P["Wns"] = np.array([[-64.0 / 12.0]], np.float32)
    P["bns"] = np.array([64.0 / 3.0], np.float32)
    P["Wdn"] = np.array([[2.0, -2.0]], np.float32)
    P["bdn"] = np.array([3.0], np.float32)
    P["Wcal"] = np.array([[1.0, -1.0]], np.float32)
    P["Wc6"] = np.array([[1.0, -1.0, -1.0, -1.0]], np.float32)
    P["Wmn"] = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], np.float32)
    Wo = np.zeros((8, 7), np.float32)
    Wo[5, 4] = 10.0; Wo[5, 6] = 0.01
    Wo[4, 1] = 2.0
    Wo[3, 2] = 2.0
    Wo[6, 3] = 2.0; Wo[6, 5] = 4.0
    P["Wo"] = Wo
    P["bo"] = np.array([-5, -5, -5, 0, 0, 0.5, -1, -5], np.float32)

    def match(p, mk, sp):
        return [
            {"id": p + "a", "op": "mul", "in": [mk, "H"]},
            {"id": p + "g", "op": "gmean", "in": [p + "a"]},
            {"id": p + "d", "op": "dense", "in": [p + "g"], "w": "W64", "b": "bm1"},
            {"id": p + "r", "op": "vrelu", "in": [p + "d"]},
            {"id": p + "b", "op": "mul", "in": [sp, "Lq"]},
            {"id": p + "gb", "op": "gmax", "in": [p + "b"]},
            {"id": p + "m", "op": "vmul", "in": [p + "r", p + "gb"]},
        ], p + "m"

    g = [
        {"id": "oneS", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "hc3", "op": "conv", "in": ["x0"], "w": "Whc"},
        {"id": "hcm", "op": "min2", "in": ["hc3", "oneS"]},
        {"id": "hcat", "op": "concat", "in": ["hcm", "x0"]},
        {"id": "hpre", "op": "conv", "in": ["hcat"], "w": "Wh1", "b": "bh1"},
        {"id": "hole", "op": "relu", "in": ["hpre"]},
        {"id": "H", "op": "colmax", "in": ["hole"]},
        {"id": "mkc", "op": "conv", "in": ["x0"], "w": "Wmk"},
        {"id": "mkr", "op": "relu", "in": ["mkc"]},
        {"id": "Mk", "op": "colmax", "in": ["mkr"]},
        {"id": "p9", "op": "slice", "in": ["x0"], "start": 9, "stop": 10},
        {"id": "Sp", "op": "colmax", "in": ["p9"]},
        {"id": "p6", "op": "slice", "in": ["x0"], "start": 6, "stop": 7},
        {"id": "Lq", "op": "colmax", "in": ["p6"]},
        {"id": "MS", "op": "concat", "in": ["Mk", "Sp"]},
        {"id": "s4c", "op": "conv", "in": ["x0"], "w": "Ws4", "b": "bm1"},
        {"id": "s4r", "op": "relu", "in": ["s4c"]},
        {"id": "s4", "op": "gmax", "in": ["s4r"]},
    ]
    n0, m0 = match("z0", "Mk", "Sp")
    g += n0

    body = [
        {"id": "R3", "op": "conv", "in": ["Rs"], "w": "W3r"},
        {"id": "R4", "op": "conv", "in": ["R3"], "w": "Wsr"},
        {"id": "Rd", "op": "sub", "in": ["R4", "R3"]},
        {"id": "Rds", "op": "mul", "in": ["Rd", "s4"]},
        {"id": "Rn", "op": "add", "in": ["R3", "Rds"]},
        {"id": "L3", "op": "conv", "in": ["Ls"], "w": "W3l"},
        {"id": "L4", "op": "conv", "in": ["L3"], "w": "Wsl"},
        {"id": "Ld", "op": "sub", "in": ["L4", "L3"]},
        {"id": "Lds", "op": "mul", "in": ["Ld", "s4"]},
        {"id": "Ln", "op": "add", "in": ["L3", "Lds"]},
        {"id": "Rmk", "op": "slice", "in": ["Rn"], "start": 0, "stop": 1},
        {"id": "Rsp", "op": "slice", "in": ["Rn"], "start": 1, "stop": 2},
        {"id": "Lmk", "op": "slice", "in": ["Ln"], "start": 0, "stop": 1},
        {"id": "Lsp", "op": "slice", "in": ["Ln"], "start": 1, "stop": 2},
    ]
    nr, mR = match("zr", "Rmk", "Rsp")
    nl, mL = match("zl", "Lmk", "Lsp")
    body += nr + nl
    body += [
        {"id": "nf", "op": "dense", "in": ["f"], "w": "Wneg", "b": "bone"},
        {"id": "aR", "op": "vmul", "in": [mR, "nf"]},
        {"id": "sRn", "op": "vadd", "in": ["sR", "aR"]},
        {"id": "aL", "op": "vmul", "in": [mL, "nf"]},
        {"id": "sLn", "op": "vadd", "in": ["sL", "aL"]},
        {"id": "tot", "op": "vadd", "in": [mR, mL]},
        {"id": "c1", "op": "dense", "in": ["tot"], "w": "Wneg", "b": "bone"},
        {"id": "c2", "op": "vrelu", "in": ["c1"]},
        {"id": "c3", "op": "dense", "in": ["c2"], "w": "Wneg", "b": "bone"},
        {"id": "af", "op": "vmul", "in": ["nf", "c3"]},
        {"id": "fn", "op": "vadd", "in": ["f", "af"]},
    ]
    g.append({"id": "LP", "op": "loop", "T": 20,
              "state": [{"id": "Rs", "init": "MS"},
                        {"id": "Ls", "init": "MS"},
                        {"id": "f", "zeros": [1]},
                        {"id": "sR", "zeros": [1]},
                        {"id": "sL", "zeros": [1]}],
              "body": body,
              "next": {"Rs": "Rn", "Ls": "Ln", "f": "fn", "sR": "sRn", "sL": "sLn"}})
    g += [
        {"id": "x8s", "op": "conv", "in": ["x0"], "w": "W8"},
        {"id": "has8", "op": "gmax", "in": ["x8s"]},
        {"id": "kk", "op": "vslice", "in": ["mem"], "start": 0, "stop": 1},
        {"id": "nn", "op": "vslice", "in": ["mem"], "start": 1, "stop": 2},
        {"id": "lqm", "op": "gmean", "in": ["Lq"]},
        {"id": "fac", "op": "dense", "in": ["s4"], "w": "Wns", "b": "bns"},
        {"id": "Ns", "op": "vmul", "in": ["lqm", "fac"]},
        {"id": "kN", "op": "vconcat", "in": ["kk", "Ns"]},
        {"id": "e0", "op": "dense", "in": ["kN"], "w": "Wdn", "b": "bdn"},
        {"id": "e1", "op": "vrelu", "in": ["e0"]},
        {"id": "e2", "op": "dense", "in": ["e1"], "w": "Wneg", "b": "bone"},
        {"id": "e3", "op": "vrelu", "in": ["e2"]},
        {"id": "done", "op": "dense", "in": ["e3"], "w": "Wneg", "b": "bone"},
        {"id": "Ad", "op": "vmul", "in": [m0, "done"]},
        {"id": "aAd", "op": "vconcat", "in": [m0, "Ad"]},
        {"id": "cal", "op": "dense", "in": ["aAd"], "w": "Wcal"},
        {"id": "cv", "op": "vconcat", "in": ["has8", m0, "LP.sR", "LP.sL"]},
        {"id": "c6l", "op": "dense", "in": ["cv"], "w": "Wc6"},
        {"id": "c6", "op": "vrelu", "in": ["c6l"]},
        {"id": "mv", "op": "vconcat", "in": ["kk", "nn", "cal", "c6"]},
        {"id": "mem_next", "op": "dense", "in": ["mv"], "w": "Wmn"},
        {"id": "v", "op": "vconcat", "in": [m0, "LP.sR", "LP.sL", "has8", "Ad", "cal", "nn"]},
        {"id": "logits", "op": "dense", "in": ["v"], "w": "Wo", "b": "bo"},
        {"id": "pos", "op": "conv", "in": ["coords"], "w": "Wpos"},
        {"id": "p9pos", "op": "mul", "in": ["p9", "pos"]},
        {"id": "selp", "op": "gmax", "in": ["p9pos"]},
        {"id": "dpos", "op": "sub", "in": ["pos", "selp"]},
        {"id": "st0", "op": "conv", "in": ["dpos"], "w": "Wst", "b": "bone"},
        {"id": "st1", "op": "relu", "in": ["st0"]},
        {"id": "st", "op": "min2", "in": ["st1", "oneS"]},
        {"id": "dst", "op": "concat", "in": ["dpos", "st"]},
        {"id": "val", "op": "conv", "in": ["dst"], "w": "Wdd", "b": "bdd"},
        {"id": "click", "op": "mul", "in": ["x8s", "val"]},
    ]
    return {"graph": g, "memory_size": 2, "params": P}

import numpy as np

def build():
    P = {}
    w = np.zeros((2, 19, 1, 1))
    for c in range(16):
        if c not in (5, 6):
            w[0, c] = 1
    w[1, 16] = -63.0; w[1, 18] = 63.0
    P["Wpc"] = w
    w = np.zeros((1, 16, 1, 1)); w[0, 9] = 1; w[0, 4] = 1
    P["Wpm"] = w
    w = np.zeros((1, 16, 1, 1))
    for c in (8, 15, 11):
        w[0, c] = 1
    P["Wem"] = w
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def detw(ecols):
        w = np.zeros((4, 16, 3, 3))
        for d, (fr, fc) in enumerate(dirs):
            pr, pcc = fc, fr
            w[d, 15, 1, 1] = 1; w[d, 11, 1, 1] = 1
            for col in ecols:
                w[d, col, 1 - fr, 1 - fc] += 1
                w[d, col, 1 + pr, 1 + pcc] += 1
                w[d, col, 1 - pr, 1 - pcc] += 1
                w[d, col, 1 + fr, 1 + fc] -= 1
        return w
    P["Wdet8"] = detw((8,)); P["Wdetc"] = detw((12, 13)); P["bdet"] = np.full(4, -3.0)
    w = np.zeros((4, 4, 7, 7))
    for d, (fr, fc) in enumerate(dirs):
        w[d, d, 3 - 3 * fr, 3 - 3 * fc] = 1
    P["WB"] = w
    w = np.zeros((1, 4, 7, 7))
    for d, (fr, fc) in enumerate(dirs):
        pr, pcc = fc, fr
        for t in (1, 2, 3):
            for s in (-1, 0, 1):
                w[0, d, 3 - t * fr - s * pr, 3 - t * fc - s * pcc] = 1
    P["WC"] = w
    eps = 1.0 / 1024
    w = np.zeros((16, 2, 5, 5)); wm = np.zeros((8, 1, 5, 5)); k = 0
    for (dr, dc) in dirs:
        for o in (1, 2):
            for c in range(2):
                w[2 * k + c, c, 2 + dr * o, 2 + dc * o] = 1
            wm[k, 0, 2 + dr * o, 2 + dc * o] = 1
            k += 1
    P["Wsh"] = w; P["bsh"] = np.full(16, -eps)
    P["Wshm"] = wm; P["bshm"] = np.full(8, -eps)
    w = np.zeros((8, 2, 7, 7)); wa = np.zeros((4, 1, 7, 7)); wb = np.zeros((8, 4, 7, 7))
    for d, (dr, dc) in enumerate(dirs):
        wa[d, 0, 3 + 3 * dr, 3 + 3 * dc] = 1
        for c in range(2):
            w[2 * d + c, c, 3 + 3 * dr, 3 + 3 * dc] = 1
            wb[2 * d + c, d, 3 + 3 * dr, 3 + 3 * dc] = 1
    P["Wsc"] = w; P["Wsa"] = wa; P["Wsb"] = wb
    w = np.zeros((8, 8))
    for d in range(4):
        w[1 + d, 2 * d + 1] = 1.0
        w[1 + d, 2 * d] = 1e-4
    P["Wo"] = w; P["bo"] = np.array([-1, 0, 0, 0, 0, -1, -1, -1], float)

    body = [
        {"id": "v1", "op": "slice", "in": ["v"], "start": 1, "stop": 2},
        {"id": "vm", "op": "mul", "in": ["v1", "notD"]},
        {"id": "sh", "op": "conv", "in": ["v"], "w": "Wsh", "b": "bsh"},
        {"id": "shm", "op": "conv", "in": ["vm"], "w": "Wshm", "b": "bshm"},
    ]
    for i in range(8):
        body.append({"id": "s%d" % i, "op": "slice", "in": ["sh"], "start": 2 * i, "stop": 2 * i + 2})
        body.append({"id": "t%d" % i, "op": "slice", "in": ["shm"], "start": i, "stop": i + 1})
    for p in ("s", "t"):
        body += [
            {"id": p + "m01", "op": "max2", "in": [p + "0", p + "1"]},
            {"id": p + "m23", "op": "max2", "in": [p + "2", p + "3"]},
            {"id": p + "m45", "op": "max2", "in": [p + "4", p + "5"]},
            {"id": p + "m67", "op": "max2", "in": [p + "6", p + "7"]},
            {"id": p + "ma", "op": "max2", "in": [p + "m01", p + "m23"]},
            {"id": p + "mb", "op": "max2", "in": [p + "m45", p + "m67"]},
            {"id": p + "mc", "op": "max2", "in": [p + "ma", p + "mb"]},
        ]
    body += [
        {"id": "mc0", "op": "slice", "in": ["smc"], "start": 0, "stop": 1},
        {"id": "mc1", "op": "slice", "in": ["smc"], "start": 1, "stop": 2},
        {"id": "e1", "op": "mul", "in": ["mc1", "DE"]},
        {"id": "a1", "op": "max2", "in": ["tmc", "e1"]},
        {"id": "n1", "op": "concat", "in": ["mc0", "a1"]},
        {"id": "md", "op": "max2", "in": ["n1", "src"]},
        {"id": "vn", "op": "mul", "in": ["md", "P2"]},
    ]
    g = [
        {"id": "xc", "op": "concat", "in": ["x0", "coords"]},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "pc", "op": "conv", "in": ["xc"], "w": "Wpc"},
        {"id": "pc0", "op": "slice", "in": ["pc"], "start": 0, "stop": 1},
        {"id": "pc1", "op": "slice", "in": ["pc"], "start": 1, "stop": 2},
        {"id": "pass0", "op": "min2", "in": ["pc0", "pc1"]},
        {"id": "detl", "op": "conv", "in": ["x0"], "w": "Wdet8", "b": "bdet"},
        {"id": "det", "op": "relu", "in": ["detl"]},
        {"id": "Bd", "op": "conv", "in": ["det"], "w": "WB"},
        {"id": "C1", "op": "conv", "in": ["det"], "w": "WC"},
        {"id": "C2", "op": "conv", "in": ["Bd"], "w": "WC"},
        {"id": "Dc", "op": "min2", "in": ["C2", "one"]},
        {"id": "notD", "op": "sub", "in": ["one", "Dc"]},
        {"id": "Em", "op": "conv", "in": ["x0"], "w": "Wem"},
        {"id": "Ds", "op": "add", "in": ["C1", "C2"]},
        {"id": "DEs", "op": "add", "in": ["Ds", "Em"]},
        {"id": "DE", "op": "min2", "in": ["DEs", "one"]},
        {"id": "detlc", "op": "conv", "in": ["x0"], "w": "Wdetc", "b": "bdet"},
        {"id": "detc", "op": "relu", "in": ["detlc"]},
        {"id": "Bdc", "op": "conv", "in": ["detc"], "w": "WB"},
        {"id": "C2c", "op": "conv", "in": ["Bdc"], "w": "WC"},
        {"id": "Pc", "op": "min2", "in": ["C2c", "one"]},
        {"id": "nPc", "op": "sub", "in": ["one", "Pc"]},
        {"id": "n3", "op": "conv", "in": ["nPc"], "w": "Wsa"},
        {"id": "n6", "op": "conv", "in": ["n3"], "w": "Wsb"},
        {"id": "P2", "op": "concat", "in": ["pass0", "pass0"]},
        {"id": "src", "op": "slice", "in": ["x0"], "start": 14, "stop": 15},
        {"id": "H", "op": "loop", "T": 232,
         "state": [{"id": "v", "zeros": [2, 64, 64]}],
         "body": body, "next": {"v": "vn"}},
        {"id": "sc", "op": "conv", "in": ["H.v"], "w": "Wsc"},
        {"id": "click", "op": "conv", "in": ["x0"], "w": "Wpm"},
        {"id": "pr0", "op": "mul", "in": ["sc", "click"]},
        {"id": "pr", "op": "mul", "in": ["pr0", "n6"]},
        {"id": "gm", "op": "gmax", "in": ["pr"]},
        {"id": "logits", "op": "dense", "in": ["gm"], "w": "Wo", "b": "bo"},
    ]
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

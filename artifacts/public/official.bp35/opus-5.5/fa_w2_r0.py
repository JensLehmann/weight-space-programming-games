import numpy as np

def build():
    g = [
        {"id": "rmk0", "op": "conv", "in": ["coords"], "w": "Wrm", "b": "brm"},
        {"id": "rmk1", "op": "relu", "in": ["rmk0"]},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "rmask", "op": "min2", "in": ["rmk1", "one"]},
        {"id": "p0", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        {"id": "s0", "op": "mul", "in": ["p0", "rmask"]},
        {"id": "spk", "op": "colmax", "in": ["s0"]},
        {"id": "cl", "op": "conv", "in": ["x0"], "w": "Wc", "b": "bc"},
        {"id": "spk10", "op": "conv", "in": ["spk"], "w": "Wsp"},
        {"id": "cl2", "op": "sub", "in": ["cl", "spk10"]},
        {"id": "click", "op": "relu", "in": ["cl2"]},
        {"id": "gt", "op": "gmax", "in": ["click"]},
        {"id": "p9", "op": "slice", "in": ["x0"], "start": 9, "stop": 10},
        {"id": "p8", "op": "slice", "in": ["x0"], "start": 8, "stop": 9},
        {"id": "cc", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "pc", "op": "mul", "in": ["p9", "cc"]},
        {"id": "colpos", "op": "gmax", "in": ["pc"]},
        {"id": "gc", "op": "mul", "in": ["p8", "cc"]},
        {"id": "t8", "op": "gmax", "in": ["gc"]},
        {"id": "has8", "op": "gmax", "in": ["p8"]},
        {"id": "hp", "op": "vmul", "in": ["has8", "colpos"]},
        {"id": "dv", "op": "vconcat", "in": ["t8", "hp"]},
        {"id": "d", "op": "dense", "in": ["dv"], "w": "Wdd"},
        {"id": "td", "op": "vtanh", "in": ["d"]},
        {"id": "v", "op": "vconcat", "in": ["colpos", "mem"]},
        {"id": "d2", "op": "dense", "in": ["v"], "w": "Wd"},
        {"id": "d2r", "op": "vrelu", "in": ["d2"]},
        {"id": "st0", "op": "dense", "in": ["d2r"], "w": "Ws", "b": "bs"},
        {"id": "stuck", "op": "vrelu", "in": ["st0"]},
        {"id": "fv", "op": "vconcat", "in": ["stuck", "aprev"]},
        {"id": "f0", "op": "dense", "in": ["fv"], "w": "Wf", "b": "bf"},
        {"id": "flip", "op": "vrelu", "in": ["f0"]},
        {"id": "r", "op": "vslice", "in": ["mem"], "start": 1, "stop": 2},
        {"id": "rf", "op": "vmul", "in": ["r", "flip"]},
        {"id": "rv", "op": "vconcat", "in": ["r", "flip", "rf"]},
        {"id": "rn", "op": "dense", "in": ["rv"], "w": "Wr"},
        {"id": "nr", "op": "dense", "in": ["rn"], "w": "Wnr", "b": "bnr"},
        {"id": "tdm", "op": "vmul", "in": ["td", "nr"]},
        {"id": "mem_next", "op": "vconcat", "in": ["colpos", "rn"]},
        {"id": "lv", "op": "vconcat", "in": ["gt", "rn", "tdm"]},
        {"id": "logits", "op": "dense", "in": ["lv"], "w": "Wl", "b": "bl"},
    ]
    Wrm = np.zeros((1, 3, 1, 1), np.float32); Wrm[0, 0, 0, 0] = -63.0
    brm = np.array([62.0], np.float32)
    Wsp = np.array([[[[10.0]]]], np.float32)
    Wc = np.zeros((1, 16, 7, 7), np.float32)
    Wc[0, 14, 3, 3] = 4.0
    Wc[0, 9, 4:7, 3] = 1.0
    bc = np.array([-4.5], np.float32)
    Wdd = np.array([[30.0, -30.0]], np.float32)
    Wd = np.array([[1, -1, 0], [-1, 1, 0]], np.float32)
    Ws = np.array([[-200.0, -200.0]], np.float32)
    bs = np.array([1.0], np.float32)
    Wf = np.zeros((1, 9), np.float32)
    Wf[0, 0] = 1; Wf[0, 4] = 1; Wf[0, 5] = 1
    bf = np.array([-1.0], np.float32)
    Wr = np.array([[1, 1, -2]], np.float32)
    Wnr = np.array([[-1.0]], np.float32)
    bnr = np.array([1.0], np.float32)
    Wl = np.zeros((8, 3), np.float32)
    bl = np.full(8, -20.0, np.float32)
    Wl[6, 0] = 20.0; bl[6] = 0.0
    Wl[4, 1] = -10.0; bl[4] = 5.0; Wl[4, 2] = 15.0
    Wl[3, 1] = 10.0; bl[3] = -5.0; Wl[3, 2] = -15.0
    return {"graph": g, "memory_size": 2,
            "params": {"Wrm": Wrm, "brm": brm, "Wsp": Wsp, "Wc": Wc, "bc": bc,
                       "Wdd": Wdd, "Wd": Wd, "Ws": Ws, "bs": bs, "Wf": Wf,
                       "bf": bf, "Wr": Wr, "Wnr": Wnr, "bnr": bnr,
                       "Wl": Wl, "bl": bl}}

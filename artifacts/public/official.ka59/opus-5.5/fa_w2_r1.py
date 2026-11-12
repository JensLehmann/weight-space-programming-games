import numpy as np

def build():
    g = [
        {"id": "mraw", "op": "conv", "in": ["coords"], "w": "Wm"},
        {"id": "mrel", "op": "relu", "in": ["mraw"]},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "mask", "op": "min2", "in": ["mrel", "one"]},
        {"id": "r", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "c", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "negc", "op": "conv", "in": ["coords"], "w": "Wn"},
        {"id": "c0", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        {"id": "c4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "c5", "op": "slice", "in": ["x0"], "start": 5, "stop": 6},
        {"id": "x14", "op": "slice", "in": ["x0"], "start": 14, "stop": 15},
        {"id": "A0", "op": "mul", "in": ["c0", "mask"]},
        {"id": "A4", "op": "mul", "in": ["c4", "mask"]},
        {"id": "A5", "op": "mul", "in": ["c5", "mask"]},
        {"id": "N14", "op": "conv", "in": ["x14"], "w": "K3"},
        {"id": "nbl", "op": "conv", "in": ["N14"], "w": "Wu", "b": "bu"},
        {"id": "nb", "op": "relu", "in": ["nbl"]},
        {"id": "U", "op": "mul", "in": ["A4", "nb"]},
        {"id": "A0r", "op": "mul", "in": ["A0", "r"]},
        {"id": "A0c", "op": "mul", "in": ["A0", "c"]},
        {"id": "rS", "op": "gmax", "in": ["A0r"]},
        {"id": "cS", "op": "gmax", "in": ["A0c"]},
        {"id": "Un", "op": "mul", "in": ["U", "negc"]},
        {"id": "gg", "op": "gmax", "in": ["Un"]},
        {"id": "d", "op": "sub", "in": ["gg", "negc"]},
        {"id": "da", "op": "abs", "in": ["d"]},
        {"id": "sl", "op": "conv", "in": ["da"], "w": "Ws", "b": "bs"},
        {"id": "s", "op": "relu", "in": ["sl"]},
        {"id": "E", "op": "mul", "in": ["A4", "s"]},
        {"id": "Er", "op": "mul", "in": ["E", "r"]},
        {"id": "gEr", "op": "gmean", "in": ["Er"]},
        {"id": "nE", "op": "gmean", "in": ["E"]},
        {"id": "t1", "op": "vmul", "in": ["rS", "nE"]},
        {"id": "D4", "op": "conv", "in": ["A4"], "w": "W4"},
        {"id": "D4d", "op": "slice", "in": ["D4"], "start": 0, "stop": 4},
        {"id": "D4a", "op": "mul", "in": ["D4d", "A0"]},
        {"id": "v4", "op": "gmax", "in": ["D4a"]},
        {"id": "vL", "op": "vslice", "in": ["v4"], "start": 0, "stop": 1},
        {"id": "vR", "op": "vslice", "in": ["v4"], "start": 1, "stop": 2},
        {"id": "vU", "op": "vslice", "in": ["v4"], "start": 2, "stop": 3},
        {"id": "vD", "op": "vslice", "in": ["v4"], "start": 3, "stop": 4},
        {"id": "p1", "op": "vmul", "in": ["vL", "vR"]},
        {"id": "p2", "op": "vmul", "in": ["vU", "vD"]},
        {"id": "f", "op": "vmul", "in": ["p1", "p2"]},
        {"id": "feat", "op": "vconcat", "in": ["gEr", "t1", "gg", "cS", "f"]},
        {"id": "lg0", "op": "dense", "in": ["feat"], "w": "Wo", "b": "bo"},
        {"id": "pm", "op": "vconcat", "in": ["rS", "cS", "mem"]},
        {"id": "dl", "op": "dense", "in": ["pm"], "w": "Wd"},
        {"id": "dr", "op": "vrelu", "in": ["dl"]},
        {"id": "stl", "op": "dense", "in": ["dr"], "w": "Wsum", "b": "bsum"},
        {"id": "stuck", "op": "vrelu", "in": ["stl"]},
        {"id": "sa", "op": "vmul", "in": ["stuck", "aprev"]},
        {"id": "pen", "op": "dense", "in": ["sa"], "w": "Wpen"},
        {"id": "logits", "op": "vadd", "in": ["lg0", "pen"]},
        {"id": "mem_next", "op": "vconcat", "in": ["rS", "cS"]},
        {"id": "S", "op": "slice", "in": ["D4"], "start": 4, "stop": 5},
        {"id": "infl", "op": "conv", "in": ["S"], "w": "Wi", "b": "bi"},
        {"id": "inf", "op": "relu", "in": ["infl"]},
        {"id": "infc", "op": "min2", "in": ["inf", "one"]},
        {"id": "penc", "op": "conv", "in": ["infc"], "w": "Wp", "b": "bp"},
        {"id": "candl", "op": "conv", "in": ["N14"], "w": "Wc", "b": "bc"},
        {"id": "cand", "op": "relu", "in": ["candl"]},
        {"id": "A45", "op": "add", "in": ["A5", "A4"]},
        {"id": "cs", "op": "mul", "in": ["A45", "cand"]},
        {"id": "click", "op": "mul", "in": ["cs", "penc"]},
    ]
    def w1(v):
        return np.full((1, 1, 1, 1), v, np.float32)
    Wm = np.zeros((1, 3, 1, 1), np.float32); Wm[0, 0, 0, 0] = -63.0; Wm[0, 2, 0, 0] = 63.0
    Wn = np.zeros((1, 3, 1, 1), np.float32); Wn[0, 1, 0, 0] = -1.0; Wn[0, 2, 0, 0] = 1.0
    K3 = np.ones((1, 1, 3, 3), np.float32)
    W4 = np.zeros((5, 1, 7, 7), np.float32)
    for o, pts in enumerate([[(3, 1), (3, 0)], [(3, 5), (3, 6)], [(1, 3), (0, 3)], [(5, 3), (6, 3)]]):
        for a, b in pts:
            W4[o, 0, a, b] = 1.0
            W4[4, 0, a, b] = 1.0
    kr, kc = 800000.0, 1000.0
    off = 1.0 + 2.0 / 63.0
    Wo = np.zeros((8, 5), np.float32)
    bo = np.full((8,), -1.0, np.float32)
    Wo[1] = [-kr, kr, 0, 0, 0]; bo[1] = 0.0
    Wo[2] = [kr, -kr, 0, 0, 0]; bo[2] = 0.0
    Wo[3] = [0, 0, kc, kc, 0]; bo[3] = -kc * off
    Wo[4] = [0, 0, -kc, -kc, 0]; bo[4] = kc * off
    Wo[6, 4] = 1e9
    Wd = np.array([[1, 0, -1, 0], [-1, 0, 1, 0], [0, 1, 0, -1], [0, -1, 0, 1]], np.float32)
    Wsum = np.full((1, 4), -1e4, np.float32)
    bsum = np.ones((1,), np.float32)
    Wpen = (-1e13 * np.eye(8)).astype(np.float32)
    params = {
        "Wm": Wm, "Wn": Wn, "K3": K3,
        "Wu": w1(-1.0), "bu": np.ones((1,), np.float32),
        "Ws": w1(-100.0), "bs": np.ones((1,), np.float32),
        "W4": W4, "Wo": Wo, "bo": bo,
        "Wd": Wd, "Wsum": Wsum, "bsum": bsum, "Wpen": Wpen,
        "Wi": w1(1.0), "bi": np.full((1,), -3.0, np.float32),
        "Wp": w1(-0.6), "bp": np.ones((1,), np.float32),
        "Wc": w1(1.0), "bc": np.full((1,), -4.0, np.float32),
    }
    return {"graph": g, "memory_size": 2, "params": params}

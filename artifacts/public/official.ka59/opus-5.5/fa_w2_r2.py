import numpy as np

def build():
    P = {}
    P["Wmk"] = np.array([-63.0, 0.0, 63.0], np.float32).reshape(1, 3, 1, 1)
    k = 1e9
    Wo = np.zeros((8, 6), np.float32)
    Wo[1, 0] = -k; Wo[2, 0] = k; Wo[3, 1] = -k; Wo[4, 1] = k
    Wo[1, 2] = -400.0; Wo[2, 3] = -400.0; Wo[3, 4] = -400.0; Wo[4, 5] = -400.0
    P["Wo"] = Wo
    bo = np.full(8, -1e6, np.float32); bo[1:5] = 0.0; bo[6] = -1e5
    P["bo"] = bo
    Wa = np.zeros((8, 8), np.float32)
    Wa[2, 1] = -60.0; Wa[1, 2] = -60.0; Wa[4, 3] = -60.0; Wa[3, 4] = -60.0
    P["Wa"] = Wa
    P["Wc"] = np.zeros((1, 3, 1, 1), np.float32)
    P["Wneg"] = np.array([[-1.0]], np.float32)
    Ww = np.zeros((1, 16, 1, 1), np.float32); Ww[0, 15] = 1.0; Ww[0, 2] = 1.0
    P["Ww"] = Ww
    Wd = np.zeros((4, 1, 7, 7), np.float32)
    for d in range(2, 4):
        Wd[0, 0, 3 - d, 3] = 1.0
        Wd[1, 0, 3 + d, 3] = 1.0
        Wd[2, 0, 3, 3 - d] = 1.0
        Wd[3, 0, 3, 3 + d] = 1.0
    P["Wd"] = Wd
    g = [
        {"id": "mk", "op": "conv", "in": ["coords"], "w": "Wmk"},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "row", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "col", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "mask", "op": "min2", "in": ["mk", "one"]},
        {"id": "p0", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        {"id": "p4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "a0", "op": "mul", "in": ["p0", "mask"]},
        {"id": "a4", "op": "mul", "in": ["p4", "mask"]},
        {"id": "a0r", "op": "mul", "in": ["a0", "row"]},
        {"id": "a0c", "op": "mul", "in": ["a0", "col"]},
        {"id": "a4r", "op": "mul", "in": ["a4", "row"]},
        {"id": "a4c", "op": "mul", "in": ["a4", "col"]},
        {"id": "st", "op": "concat", "in": ["a0r", "a0c", "a0", "a4r", "a4c", "a4"]},
        {"id": "v", "op": "gmean", "in": ["st"]},
        {"id": "onev", "op": "gmean", "in": ["one"]},
        {"id": "mone", "op": "dense", "in": ["onev"], "w": "Wneg"},
        {"id": "R0", "op": "vslice", "in": ["v"], "start": 0, "stop": 1},
        {"id": "C0", "op": "vslice", "in": ["v"], "start": 1, "stop": 2},
        {"id": "N0", "op": "vslice", "in": ["v"], "start": 2, "stop": 3},
        {"id": "R4", "op": "vslice", "in": ["v"], "start": 3, "stop": 4},
        {"id": "C4", "op": "vslice", "in": ["v"], "start": 4, "stop": 5},
        {"id": "N4", "op": "vslice", "in": ["v"], "start": 5, "stop": 6},
        {"id": "y1", "op": "vmul", "in": ["R4", "N0"]},
        {"id": "y2", "op": "vmul", "in": ["R0", "N4"]},
        {"id": "x1", "op": "vmul", "in": ["C4", "N0"]},
        {"id": "x2", "op": "vmul", "in": ["C0", "N4"]},
        {"id": "pr", "op": "vconcat", "in": ["y1", "x1"]},
        {"id": "ng", "op": "vconcat", "in": ["y2", "x2"]},
        {"id": "ngm", "op": "vmul", "in": ["ng", "mone"]},
        {"id": "d", "op": "vadd", "in": ["pr", "ngm"]},
        {"id": "wl", "op": "conv", "in": ["x0"], "w": "Ww"},
        {"id": "wlm", "op": "mul", "in": ["wl", "mask"]},
        {"id": "sh", "op": "conv", "in": ["wlm"], "w": "Wd"},
        {"id": "bl", "op": "mul", "in": ["sh", "a0"]},
        {"id": "blv", "op": "gmax", "in": ["bl"]},
        {"id": "dd", "op": "vconcat", "in": ["d", "blv"]},
        {"id": "lg0", "op": "dense", "in": ["dd"], "w": "Wo", "b": "bo"},
        {"id": "pa", "op": "dense", "in": ["aprev"], "w": "Wa"},
        {"id": "logits", "op": "vadd", "in": ["lg0", "pa"]},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc"},
    ]
    return {"graph": g, "memory_size": 0, "params": P}

import numpy as np

def build():
    g = [
        {"id": "mc", "op": "conv", "in": ["coords"], "w": "Wmask"},
        {"id": "mr", "op": "relu", "in": ["mc"]},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "mask", "op": "min2", "in": ["mr", "one"]},
        {"id": "row", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "col", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "c0", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        {"id": "c4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "p0", "op": "mul", "in": ["c0", "mask"]},
        {"id": "p4", "op": "mul", "in": ["c4", "mask"]},
        {"id": "pr", "op": "rowmax", "in": ["p0"]},
        {"id": "pc", "op": "colmax", "in": ["p0"]},
        {"id": "prr", "op": "mul", "in": ["pr", "row"]},
        {"id": "pcc", "op": "mul", "in": ["pc", "col"]},
        {"id": "prm", "op": "gmax", "in": ["prr"]},
        {"id": "pcm", "op": "gmax", "in": ["pcc"]},
        {"id": "dyr", "op": "sub", "in": ["row", "prm"]},
        {"id": "dxc", "op": "sub", "in": ["col", "pcm"]},
        {"id": "ady", "op": "abs", "in": ["dyr"]},
        {"id": "adx", "op": "abs", "in": ["dxc"]},
        {"id": "dist", "op": "add", "in": ["ady", "adx"]},
        {"id": "dk", "op": "conv", "in": ["dist"], "w": "Wk"},
        {"id": "wt", "op": "sigmoid", "in": ["dk"]},
        {"id": "w4", "op": "mul", "in": ["wt", "p4"]},
        {"id": "wy", "op": "mul", "in": ["dyr", "w4"]},
        {"id": "wx", "op": "mul", "in": ["dxc", "w4"]},
        {"id": "cat", "op": "concat", "in": ["wy", "wx"]},
        {"id": "gv", "op": "gmean", "in": ["cat"]},
        {"id": "logits", "op": "dense", "in": ["gv"], "w": "Wo", "b": "bo"},
        {"id": "click", "op": "conv", "in": ["mask"], "w": "Wc"},
    ]
    Wmask = np.zeros((1, 3, 1, 1), np.float32)
    Wmask[0, 0, 0, 0] = -126.0
    Wmask[0, 2, 0, 0] = 125.0
    Wk = np.full((1, 1, 1, 1), -12.0, np.float32)
    S = 1e12
    Wo = np.zeros((8, 2), np.float32)
    Wo[1] = [-S, 0]
    Wo[2] = [S, 0]
    Wo[3] = [0, -S]
    Wo[4] = [0, S]
    bo = np.full((8,), -1e30, np.float32)
    bo[1:5] = 0.0
    Wc = np.zeros((1, 1, 1, 1), np.float32)
    return {"graph": g, "memory_size": 0,
            "params": {"Wmask": Wmask, "Wk": Wk, "Wo": Wo, "bo": bo, "Wc": Wc}}

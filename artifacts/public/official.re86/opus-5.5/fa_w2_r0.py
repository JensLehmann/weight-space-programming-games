import numpy as np

def build():
    seq = [1, 1, 1, 4, 4, 4, 5, 2, 2, 2, 3, 3, 3, 5]
    n = len(seq)
    S = np.zeros((n, n), np.float32)
    for i in range(n):
        S[(i + 1) % n, i] = 1.0
    Wsum = -np.ones((1, n), np.float32)
    bsum = np.array([1.0], np.float32)
    E0 = np.zeros((n, 1), np.float32); E0[0, 0] = 1.0
    WL = np.zeros((8, n), np.float32)
    for i, a in enumerate(seq):
        WL[a, i] = 4.0
    bL = np.array([-20.0, 0, 0, 0, 0, 0, -20.0, -20.0], np.float32)
    graph = [
        {"id": "s", "op": "dense", "in": ["mem"], "w": "Wsum", "b": "bsum"},
        {"id": "s0", "op": "dense", "in": ["s"], "w": "E0"},
        {"id": "cur", "op": "vadd", "in": ["mem", "s0"]},
        {"id": "logits", "op": "dense", "in": ["cur"], "w": "WL", "b": "bL"},
        {"id": "nx", "op": "dense", "in": ["cur"], "w": "S", "b": "bS"},
        {"id": "mem_next", "op": "vsigmoid", "in": ["nx"]},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc", "b": "bc"},
    ]
    params = {
        "Wsum": Wsum, "bsum": bsum, "E0": E0,
        "WL": WL, "bL": bL,
        "S": S * 40.0, "bS": np.full((n,), -20.0, np.float32),
        "Wc": np.zeros((1, 3, 1, 1), np.float32),
        "bc": np.zeros((1,), np.float32),
    }
    return {"graph": graph, "memory_size": n, "params": params}

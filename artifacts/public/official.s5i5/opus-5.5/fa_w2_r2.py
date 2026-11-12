import numpy as np

def build():
    graph = [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},
        {"id": "base", "op": "conv", "in": ["x"], "w": "Wc", "b": "bc"},
        {"id": "rowp", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "pen0", "op": "conv", "in": ["rowp"], "w": "Wp", "b": "bp"},
        {"id": "pen", "op": "relu", "in": ["pen0"]},
        {"id": "colp", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "a", "op": "dense", "in": ["mem"], "w": "Wa", "b": "ba"},
        {"id": "bias", "op": "mul", "in": ["colp", "a"]},
        {"id": "c1", "op": "add", "in": ["base", "bias"]},
        {"id": "click", "op": "sub", "in": ["c1", "pen"]},
        {"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wn", "b": "bn"},
        {"id": "logits", "op": "dense", "in": ["mem"], "w": "Wl", "b": "bl"},
    ]
    Wc = np.zeros((1, 19, 1, 1), np.float32)
    Wc[0, 4, 0, 0] = 1.0
    bc = np.zeros((1,), np.float32)
    Wp = np.array([[[[630.0]]]], np.float32)
    bp = np.array([-625.0], np.float32)
    Wa = np.array([[0.9]], np.float32)
    ba = np.array([-0.45], np.float32)
    Wn = np.array([[-1.0]], np.float32)
    bn = np.array([1.0], np.float32)
    Wl = np.zeros((8, 1), np.float32)
    bl = np.zeros((8,), np.float32)
    bl[6] = 5.0
    params = {"Wc": Wc, "bc": bc, "Wp": Wp, "bp": bp, "Wa": Wa, "ba": ba,
              "Wn": Wn, "bn": bn, "Wl": Wl, "bl": bl}
    return {"graph": graph, "memory_size": 1, "params": params}

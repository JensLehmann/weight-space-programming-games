import numpy as np

def build():
    graph = [
        {"id": "g", "op": "gmean", "in": ["x0"]},
        {"id": "logits", "op": "dense", "in": ["g"], "w": "Wl", "b": "bl"},
        {"id": "xc", "op": "concat", "in": ["x0", "coords"]},
        {"id": "a", "op": "conv", "in": ["xc"], "w": "Wa"},
        {"id": "c9", "op": "slice", "in": ["x0"], "start": 9, "stop": 10},
        {"id": "ct", "op": "conv", "in": ["coords"], "w": "Wct"},
        {"id": "m", "op": "mul", "in": ["c9", "ct"]},
        {"id": "click", "op": "add", "in": ["a", "m"]},
    ]
    Wl = np.zeros((8, 16), np.float32)
    bl = np.zeros(8, np.float32)
    bl[6] = 5.0
    Wa = np.zeros((1, 19, 3, 3), np.float32)
    Wa[0, 1, 1, 1] = 5.0
    Wa[0, 2, :, :] = -5.0
    Wct = np.zeros((1, 3, 1, 1), np.float32)
    Wct[0, 0, 0, 0] = 0.5
    Wct[0, 1, 0, 0] = 1.0
    Wct[0, 2, 0, 0] = 1.0
    return {"graph": graph, "memory_size": 0,
            "params": {"Wl": Wl, "bl": bl, "Wa": Wa, "Wct": Wct}}

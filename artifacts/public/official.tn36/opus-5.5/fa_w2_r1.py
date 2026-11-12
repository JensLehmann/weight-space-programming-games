import numpy as np

def build():
    Wc = np.zeros((1, 19, 1, 1), np.float32)
    Wc[0, 1, 0, 0] = 3.0      # colour 1: un-set instruction bars -> click first
    Wc[0, 9, 0, 0] = 1.0      # colour 9: run button
    Wc[0, 16, 0, 0] = 0.5     # prefer lower rows (avoid timer bar)
    Wc[0, 17, 0, 0] = 1.0     # prefer right columns (avoid slot frame)
    bc = np.zeros((1,), np.float32)
    Wl = np.zeros((8, 8), np.float32)
    bl = np.zeros((8,), np.float32)
    bl[6] = 1.0
    graph = [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},
        {"id": "click", "op": "conv", "in": ["x"], "w": "Wc", "b": "bc"},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wl", "b": "bl"},
    ]
    return {"graph": graph, "memory_size": 0,
            "params": {"Wc": Wc, "bc": bc, "Wl": Wl, "bl": bl}}

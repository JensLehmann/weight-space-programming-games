import numpy as np

def build():
    graph = [
        {"id": "m1", "op": "dense", "in": ["mem"], "w": "Wm", "b": "bm"},
        {"id": "mem_next", "op": "vtanh", "in": ["m1"]},
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},
        {"id": "base", "op": "conv", "in": ["x"], "w": "Wc", "b": "bc"},
        {"id": "colp", "op": "conv", "in": ["x"], "w": "Wp"},
        {"id": "colm", "op": "mul", "in": ["colp", "mem_next"]},
        {"id": "gpre", "op": "conv", "in": ["coords"], "w": "Wg"},
        {"id": "g", "op": "relu", "in": ["gpre"]},
        {"id": "gn", "op": "conv", "in": ["g"], "w": "Wn"},
        {"id": "s1", "op": "add", "in": ["base", "colm"]},
        {"id": "click", "op": "add", "in": ["s1", "gn"]},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wl", "b": "bl"},
    ]
    Wm = np.array([[-3.0]], np.float32)
    bm = np.array([1.0], np.float32)
    Wc = np.zeros((1, 19, 1, 1), np.float32)
    Wc[0, 4, 0, 0] = 1.0
    Wc[0, 2, 0, 0] = 0.3
    Wc[0, 16, 0, 0] = -0.3
    bc = np.zeros((1,), np.float32)
    Wp = np.zeros((1, 19, 1, 1), np.float32)
    Wp[0, 17, 0, 0] = 0.6
    Wg = np.zeros((1, 3, 1, 1), np.float32)
    Wg[0, 0, 0, 0] = 63.0
    Wg[0, 2, 0, 0] = -62.5
    Wn = np.full((1, 1, 1, 1), -20.0, np.float32)
    Wl = np.zeros((8, 8), np.float32)
    bl = np.zeros((8,), np.float32)
    bl[6] = 1.0
    return {"graph": graph, "memory_size": 1,
            "params": {"Wm": Wm, "bm": bm, "Wc": Wc, "bc": bc, "Wp": Wp,
                       "Wg": Wg, "Wn": Wn, "Wl": Wl, "bl": bl}}

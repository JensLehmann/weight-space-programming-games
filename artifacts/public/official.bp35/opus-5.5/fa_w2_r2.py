import numpy as np

def build():
    Wl = np.zeros((8, 16), np.float32)
    Wl[6, 14] = 3.0
    bl = np.array([-5.0, -5.0, -5.0, 0.3, 0.5, -5.0, -1.0, -2.0], np.float32)
    Wc = np.zeros((1, 19, 1, 1), np.float32)
    Wc[0, 14, 0, 0] = 4.0
    Wc[0, 15, 0, 0] = -10.0
    Wc[0, 16, 0, 0] = 2.0
    bc = np.zeros((1,), np.float32)
    graph = [
        {"id": "gm", "op": "gmax", "in": ["x0"]},
        {"id": "logits", "op": "dense", "in": ["gm"], "w": "Wl", "b": "bl"},
        {"id": "xc", "op": "concat", "in": ["x0", "coords"]},
        {"id": "click", "op": "conv", "in": ["xc"], "w": "Wc", "b": "bc"},
    ]
    return {"graph": graph, "memory_size": 0,
            "params": {"Wl": Wl, "bl": bl, "Wc": Wc, "bc": bc}}

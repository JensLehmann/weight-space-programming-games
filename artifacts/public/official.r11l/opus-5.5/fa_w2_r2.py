import numpy as np

def build():
    graph = [
        {"id": "g", "op": "gmean", "in": ["x0"]},
        {"id": "logits", "op": "dense", "in": ["g"], "w": "Wl", "b": "bl"},
        {"id": "ring", "op": "conv", "in": ["x0"], "w": "Wr", "b": "br"},
        {"id": "cb", "op": "conv", "in": ["coords"], "w": "Wq"},
        {"id": "click", "op": "add", "in": ["ring", "cb"]},
    ]
    Wl = np.zeros((8, 16), np.float32)
    bl = np.zeros(8, np.float32)
    bl[6] = 1.0
    Wr = np.zeros((1, 16, 7, 7), np.float32)
    for c, v in [(15, 1.0), (14, 0.9), (11, 0.8), (8, 0.8), (9, 0.8)]:
        Wr[0, c, 1:6, 1:6] = v
        Wr[0, c, 3, 3] = -3.0
    Wr[0, 6, :, :] = -30.0
    for c in (2, 10):
        Wr[0, c, 3, 3] = -10.0
    br = np.zeros(1, np.float32)
    Wq = np.zeros((1, 3, 1, 1), np.float32)
    Wq[0, 0, 0, 0] = 0.01
    Wq[0, 1, 0, 0] = 0.005
    return {"graph": graph, "memory_size": 0,
            "params": {"Wl": Wl, "bl": bl, "Wr": Wr, "br": br, "Wq": Wq}}

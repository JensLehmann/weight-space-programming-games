import numpy as np

def build():
    graph = [
        {"id": "g", "op": "gmean", "in": ["x0"]},
        {"id": "logits", "op": "dense", "in": ["g"], "w": "Wo", "b": "bo"},
        {"id": "score", "op": "conv", "in": ["x0"], "w": "Wc"},
        {"id": "r63", "op": "conv", "in": ["coords"], "w": "Wr"},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "mask", "op": "min2", "in": ["r63", "one"]},
        {"id": "click", "op": "mul", "in": ["score", "mask"]},
    ]
    Wo = np.zeros((8, 16), np.float32)
    bo = np.array([-10, -5, -5, -5, -5, -5, 5, -1], np.float32)
    Wc = np.zeros((1, 16, 1, 1), np.float32)
    Wc[0, 14, 0, 0] = 1.0   # green piece
    Wc[0, 8, 0, 0] = 0.9    # red piece
    Wc[0, 2, 0, 0] = 3.0    # gray jump-target ring
    Wr = np.zeros((1, 3, 1, 1), np.float32)
    Wr[0, 0, 0, 0] = 63.0   # row index -> mask row 0
    return {"graph": graph, "memory_size": 0,
            "params": {"Wo": Wo, "bo": bo, "Wc": Wc, "Wr": Wr}}

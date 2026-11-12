import numpy as np

def build():
    graph = [
        {"id": "x9", "op": "slice", "in": ["x0"], "start": 9, "stop": 10},
        {"id": "click", "op": "conv", "in": ["x9"], "w": "Wc", "b": "bc"},
        {"id": "g", "op": "gmean", "in": ["x0"]},
        {"id": "logits", "op": "dense", "in": ["g"], "w": "Wo", "b": "bo"},
    ]
    Wc = np.ones((1, 1, 3, 3), np.float32)
    bc = np.zeros((1,), np.float32)
    Wo = np.zeros((8, 16), np.float32)
    bo = np.zeros((8,), np.float32)
    bo[6] = 1.0
    return {"graph": graph, "memory_size": 0,
            "params": {"Wc": Wc, "bc": bc, "Wo": Wo, "bo": bo}}

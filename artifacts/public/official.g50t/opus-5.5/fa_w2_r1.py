import numpy as np

def build():
    graph = [
        {"id": "g", "op": "gmean", "in": ["x0"]},
        {"id": "l1", "op": "dense", "in": ["g"], "w": "Wo", "b": "bo"},
        {"id": "l2", "op": "dense", "in": ["aprev"], "w": "Wa"},
        {"id": "logits", "op": "vadd", "in": ["l1", "l2"]},
        {"id": "click", "op": "conv", "in": ["x0"], "w": "Wc", "b": "bc"},
    ]
    Wo = np.zeros((8, 16), np.float32)
    bo = np.array([-9.0, 0.3, 0.5, 0.1, 1.0, -9.0, -9.0, -9.0], np.float32)
    Wa = np.zeros((8, 8), np.float32)
    # cycle: A4 -> A2 -> A1 -> A4
    Wa[2, 4] = 3.0
    Wa[1, 2] = 3.0
    Wa[4, 1] = 3.0
    Wc = np.zeros((1, 16, 1, 1), np.float32)
    bc = np.zeros((1,), np.float32)
    return {"graph": graph, "memory_size": 0,
            "params": {"Wo": Wo, "bo": bo, "Wa": Wa, "Wc": Wc, "bc": bc}}

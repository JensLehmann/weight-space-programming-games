import numpy as np

def build():
    graph = [
        {"id": "logits", "op": "dense", "in": ["aprev"], "w": "Wa", "b": "ba"},
        {"id": "click", "op": "conv", "in": ["x0"], "w": "Wc", "b": "bc"},
    ]
    Wa = np.zeros((8, 8), np.float32)
    ba = np.zeros(8, np.float32)
    ba[0] = -100.0
    ba[6] = -100.0
    ba[7] = -100.0
    ba[1] = 1.0
    trans = {1: 4, 4: 2, 2: 3, 3: 5, 5: 1}
    for p, n in trans.items():
        Wa[n, p] = 5.0
    Wc = np.zeros((1, 16, 1, 1), np.float32)
    bc = np.zeros(1, np.float32)
    return {"graph": graph, "memory_size": 0,
            "params": {"Wa": Wa, "ba": ba, "Wc": Wc, "bc": bc}}

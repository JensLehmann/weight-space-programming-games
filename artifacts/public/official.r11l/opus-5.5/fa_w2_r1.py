import numpy as np

def build():
    graph = [
        {"id": "F", "op": "conv", "in": ["x0"], "w": "Wf"},
        {"id": "G", "op": "concat", "in": ["x0", "F"]},
        {"id": "S", "op": "conv", "in": ["G"], "w": "Ws"},
        {"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wm", "b": "bm"},
        {"id": "v", "op": "vconcat", "in": ["mem", "mem_next"]},
        {"id": "P", "op": "mul", "in": ["S", "v"]},
        {"id": "click", "op": "conv", "in": ["P"], "w": "Wc"},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wo", "b": "bo"},
    ]
    ring = [8, 9, 11, 14, 15]
    Wf = np.zeros((4, 16, 7, 7), np.float32)
    Wf[0, 6] = 1.0
    for c in ring:
        Wf[1, c] = 1.0
    Wf[2, 0] = 1.0
    Wf[2, 3] = 1.0
    Wf[3, 10] = 1.0
    Ws = np.zeros((2, 20, 1, 1), np.float32)
    for c in ring:
        Ws[0, c, 0, 0] = 1.0
    Ws[0, 16, 0, 0] = -5.0
    Ws[0, 18, 0, 0] = -5.0
    Ws[0, 19, 0, 0] = -5.0
    Ws[1, 3, 0, 0] = 1.0
    Ws[1, 17, 0, 0] = -0.3
    Wm = -np.ones((1, 1), np.float32)
    bm = np.ones((1,), np.float32)
    Wc = np.ones((1, 2, 1, 1), np.float32)
    Wo = np.zeros((8, 8), np.float32)
    bo = np.zeros((8,), np.float32)
    bo[6] = 10.0
    return {"graph": graph, "memory_size": 1,
            "params": {"Wf": Wf, "Ws": Ws, "Wm": Wm, "bm": bm,
                       "Wc": Wc, "Wo": Wo, "bo": bo}}

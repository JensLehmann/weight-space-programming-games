import numpy as np

def build():
    M = 4
    W = np.zeros((8, 8), np.float32)
    W[4, 4] = 1.5; W[2, 4] = 1.2; W[2, 2] = 1.5; W[4, 2] = 1.2
    W[3, 2] = 0.8; W[1, 3] = 1.0; W[1, 1] = 1.0; W[4, 1] = 1.3
    b = np.array([-10.0, 0.4, 0.8, 0.3, 1.0, -10.0, -10.0, -10.0], np.float32)
    Wmem = np.zeros((M, M), np.float32)
    for i in range(M):
        Wmem[(i + 1) % M, i] = 1.0
    bm = np.array([1.0, 0, 0, 0], np.float32) * 0.0
    Wlm = np.zeros((8, M), np.float32)
    Wlm[2, 0] = 0.3; Wlm[4, 2] = 0.3
    Wc = np.zeros((1, 3, 1, 1), np.float32)
    bc = np.zeros((1,), np.float32)
    graph = [
        {"id": "l1", "op": "dense", "in": ["aprev"], "w": "Wl", "b": "bl"},
        {"id": "mm", "op": "dense", "in": ["mem"], "w": "Wmem", "b": "bm"},
        {"id": "mem_next", "op": "vtanh", "in": ["mm"]},
        {"id": "l2", "op": "dense", "in": ["mem"], "w": "Wlm"},
        {"id": "logits", "op": "vadd", "in": ["l1", "l2"]},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc", "b": "bc"},
    ]
    return {"graph": graph, "memory_size": M,
            "params": {"Wl": W, "bl": b, "Wmem": Wmem, "bm": bm,
                       "Wlm": Wlm, "Wc": Wc, "bc": bc}}

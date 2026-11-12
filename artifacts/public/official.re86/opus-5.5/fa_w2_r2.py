import numpy as np

def build():
    M = 16
    graph = [
        {"id": "ma", "op": "vconcat", "in": ["mem", "aprev"]},
        {"id": "mem_next", "op": "dense", "in": ["ma"], "w": "Wm", "b": "bm"},
        {"id": "logits", "op": "dense", "in": ["mem_next"], "w": "Wl", "b": "bl"},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc", "b": "bc"},
    ]
    Wm = np.zeros((M, M + 8), np.float32)
    for i in range(M):
        Wm[(i + 1) % M, i] = 1.0
    Wm[0, M:M + 8] = -1.0
    bm = np.zeros(M, np.float32)
    bm[0] = 1.0
    sched = [1, 1, 1, 4, 4, 4, 2, 2, 2, 3, 3, 3, 5, 1, 4, 2]
    Wl = np.zeros((8, M), np.float32)
    for p, a in enumerate(sched):
        Wl[a, p] = 1.0
    bl = np.zeros(8, np.float32)
    params = {
        "Wm": Wm, "bm": bm, "Wl": Wl, "bl": bl,
        "Wc": np.zeros((1, 3, 1, 1), np.float32),
        "bc": np.zeros(1, np.float32),
    }
    return {"graph": graph, "memory_size": M, "params": params}

import numpy as np

def build():
    K = 64
    Ws = -np.ones((1, K), np.float32)
    bs = np.ones((1,), np.float32)
    We = np.zeros((K, 1), np.float32); We[0, 0] = 1.0
    Wsh = np.zeros((K, K), np.float32)
    for t in range(K - 1):
        Wsh[t + 1, t] = 1.0
    Wl = np.zeros((8, K), np.float32)
    bl = np.zeros((8,), np.float32)
    Wv = np.zeros((2, K), np.float32)
    clicks = {1: (50, 30), 2: (55, 25), 3: (55, 35), 4: (60, 30)}
    for t in range(K):
        if t in clicks:
            Wl[6, t] = 1.0
            r, c = clicks[t]
            Wv[0, t] = 2.0 * r / 63.0
            Wv[1, t] = 2.0 * c / 63.0
        else:
            Wl[3, t] = 1.0
    graph = [
        {"id": "s", "op": "dense", "in": ["mem"], "w": "Ws", "b": "bs"},
        {"id": "e", "op": "dense", "in": ["s"], "w": "We"},
        {"id": "sh", "op": "dense", "in": ["mem"], "w": "Wsh"},
        {"id": "mem_next", "op": "vadd", "in": ["sh", "e"]},
        {"id": "logits", "op": "dense", "in": ["mem_next"], "w": "Wl", "b": "bl"},
        {"id": "v", "op": "dense", "in": ["mem_next"], "w": "Wv"},
        {"id": "vr", "op": "vslice", "in": ["v"], "start": 0, "stop": 1},
        {"id": "vc", "op": "vslice", "in": ["v"], "start": 1, "stop": 2},
        {"id": "R", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "C", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "a", "op": "mul", "in": ["R", "vr"]},
        {"id": "bq", "op": "mul", "in": ["C", "vc"]},
        {"id": "rr", "op": "mul", "in": ["R", "R"]},
        {"id": "cc", "op": "mul", "in": ["C", "C"]},
        {"id": "s1", "op": "add", "in": ["a", "bq"]},
        {"id": "s2", "op": "sub", "in": ["s1", "rr"]},
        {"id": "click", "op": "sub", "in": ["s2", "cc"]},
    ]
    return {"graph": graph, "memory_size": K,
            "params": {"Ws": Ws, "bs": bs, "We": We, "Wsh": Wsh,
                       "Wl": Wl, "bl": bl, "Wv": Wv}}

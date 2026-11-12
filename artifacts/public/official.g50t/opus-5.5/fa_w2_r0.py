import numpy as np

def build():
    seq = [4, 4, 4, 2, 2, 2, 3, 3, 2, 2, 2, 4, 4, 4, 1, 1]
    K = len(seq)
    L = np.zeros((8, K), np.float32)
    for p, a in enumerate(seq):
        L[a, p] = 5.0
    bl = np.zeros(8, np.float32)
    bl[4] = 1.0
    bl[0] = -30.0; bl[5] = -30.0; bl[6] = -30.0; bl[7] = -30.0
    P = np.zeros((K, K), np.float32)
    for p in range(K):
        P[(p + 1) % K, p] = 1.0
    Ws = np.zeros((K, 8), np.float32)
    Ws[1, :] = -1.0
    bs = np.zeros(K, np.float32)
    bs[1] = 1.0
    Wc = np.zeros((1, 3, 1, 1), np.float32)
    graph = [
        {"id": "logits", "op": "dense", "in": ["mem"], "w": "L", "b": "bl"},
        {"id": "sh", "op": "dense", "in": ["mem"], "w": "P"},
        {"id": "st", "op": "dense", "in": ["aprev"], "w": "Ws", "b": "bs"},
        {"id": "mem_next", "op": "vadd", "in": ["sh", "st"]},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc"},
    ]
    return {"graph": graph, "memory_size": K,
            "params": {"L": L, "bl": bl, "P": P, "Ws": Ws, "bs": bs, "Wc": Wc}}

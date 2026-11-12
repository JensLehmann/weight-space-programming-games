import numpy as np

def build():
    # 8 undos (rewind perturbations to level start), then the hand-derived level-1 solution:
    # U U U, R R R R, L L, D D, R R, L L, U, R R
    seq = [7] * 8 + [1, 1, 1, 4, 4, 4, 4, 3, 3, 2, 2, 4, 4, 3, 3, 1, 4, 4]
    L = len(seq)

    Wsum = np.ones((1, L), np.float32)
    We0 = np.zeros((L, 1), np.float32)
    We0[0, 0] = -1.0
    be0 = np.zeros((L,), np.float32)
    be0[0] = 1.0

    Wact = np.zeros((8, L), np.float32)
    for k, a in enumerate(seq):
        Wact[a, k] = 10.0

    Wshift = np.zeros((L, L), np.float32)
    for k in range(L):
        Wshift[(k + 1) % L, k] = 1.0

    graph = [
        {"id": "s", "op": "dense", "in": ["mem"], "w": "Wsum",
         "comment": "sum of phase register (0 at episode start)"},
        {"id": "t", "op": "dense", "in": ["s"], "w": "We0", "b": "be0",
         "comment": "e0*(1-s)"},
        {"id": "p", "op": "vadd", "in": ["mem", "t"],
         "comment": "current phase one-hot"},
        {"id": "logits", "op": "dense", "in": ["p"], "w": "Wact"},
        {"id": "mem_next", "op": "dense", "in": ["p"], "w": "Wshift",
         "comment": "advance phase cyclically"},
        {"id": "click", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
    ]
    params = {"Wsum": Wsum, "We0": We0, "be0": be0, "Wact": Wact, "Wshift": Wshift}
    return {"graph": graph, "memory_size": L, "params": params}

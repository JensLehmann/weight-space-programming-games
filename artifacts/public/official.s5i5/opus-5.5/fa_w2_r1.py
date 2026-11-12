import numpy as np

def build():
    graph = [
        {"id": "c4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "c2", "op": "slice", "in": ["x0"], "start": 2, "stop": 3},
        {"id": "rs", "op": "rowsum", "in": ["c2"]},
        {"id": "rsw", "op": "conv", "in": ["rs"], "w": "Wrs", "b": "brs"},
        {"id": "m", "op": "mul", "in": ["c4", "rsw"]},
        {"id": "colc", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "colb", "op": "conv", "in": ["colc"], "w": "Wcol"},
        {"id": "click", "op": "add", "in": ["m", "colb"]},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wl", "b": "bl"},
    ]
    Wrs = np.full((1, 1, 1, 1), 0.05, np.float32)
    brs = np.ones((1,), np.float32)
    Wcol = np.full((1, 1, 1, 1), -0.2, np.float32)
    Wl = np.eye(8, dtype=np.float32)
    bl = np.zeros((8,), np.float32)
    bl[6] = 1.0
    return {"graph": graph, "memory_size": 0,
            "params": {"Wrs": Wrs, "brs": brs, "Wcol": Wcol, "Wl": Wl, "bl": bl}}

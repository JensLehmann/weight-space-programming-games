import numpy as np

def build():
    M = 8
    Wm = np.zeros((M, M), np.float32)
    for k in range(M - 1):
        Wm[k + 1, k] = 1.0
    Wm[M - 1, M - 1] = 1.0
    bm = np.zeros(M, np.float32); bm[0] = 1.0

    Wl = np.zeros((8, M), np.float32); bl = np.zeros(8, np.float32)
    bl[6] = 10.0; Wl[6, 4] = -10.0
    bl[3] = 5.0

    Wr = np.zeros((2, M), np.float32); br = np.zeros(2, np.float32)
    br[0] = 50.0 / 63; Wr[0, 1] = 5.0 / 63; Wr[0, 3] = 5.0 / 63
    br[1] = 30.0 / 63; Wr[1, 1] = -5.0 / 63; Wr[1, 2] = 10.0 / 63; Wr[1, 3] = -5.0 / 63

    Wn = -np.ones((1, 1, 1, 1), np.float32)

    graph = [
        {"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wm", "b": "bm"},
        {"id": "logits", "op": "dense", "in": ["mem"], "w": "Wl", "b": "bl"},
        {"id": "rc", "op": "dense", "in": ["mem"], "w": "Wr", "b": "br"},
        {"id": "rv", "op": "vslice", "in": ["rc"], "start": 0, "stop": 1},
        {"id": "cv", "op": "vslice", "in": ["rc"], "start": 1, "stop": 2},
        {"id": "rowc", "op": "slice", "in": ["coords"], "start": 0, "stop": 1},
        {"id": "colc", "op": "slice", "in": ["coords"], "start": 1, "stop": 2},
        {"id": "dr", "op": "sub", "in": ["rowc", "rv"]},
        {"id": "dc", "op": "sub", "in": ["colc", "cv"]},
        {"id": "adr", "op": "abs", "in": ["dr"]},
        {"id": "adc", "op": "abs", "in": ["dc"]},
        {"id": "dist", "op": "add", "in": ["adr", "adc"]},
        {"id": "click", "op": "conv", "in": ["dist"], "w": "Wn"},
    ]
    return {"graph": graph, "memory_size": M,
            "params": {"Wm": Wm, "bm": bm, "Wl": Wl, "bl": bl,
                       "Wr": Wr, "br": br, "Wn": Wn}}

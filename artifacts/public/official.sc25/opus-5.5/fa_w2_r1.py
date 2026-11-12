import numpy as np

def build():
    graph = [
        {"id": "x", "op": "concat", "in": ["x0", "coords"]},
        {"id": "cl", "op": "conv", "in": ["x"], "w": "Wcol", "b": "bcol"},
        {"id": "clr", "op": "relu", "in": ["cl"]},
        {"id": "xc", "op": "concat", "in": ["x", "clr"]},
        {"id": "click", "op": "conv", "in": ["xc"], "w": "Wc"},
        {"id": "gc", "op": "gmax", "in": ["click"]},
        {"id": "lc", "op": "dense", "in": ["gc"], "w": "Wgc"},
        {"id": "la", "op": "dense", "in": ["aprev"], "w": "Wa", "b": "ba"},
        {"id": "logits", "op": "vadd", "in": ["la", "lc"]},
    ]
    Wcol = np.zeros((1, 19, 1, 1), np.float32)
    Wcol[0, 17, 0, 0] = 1.0          # col/63
    bcol = np.array([-0.8], np.float32)
    Wc = np.zeros((1, 20, 1, 1), np.float32)
    Wc[0, 0, 0, 0] = 1.0             # black hint cells
    Wc[0, 19, 0, 0] = -10.0          # strongly suppress right-edge UI bar
    Wgc = np.zeros((8, 1), np.float32)
    Wgc[6, 0] = 6.0
    Wa = np.zeros((8, 8), np.float32)
    nxt = {3: 1, 1: 4, 4: 2, 2: 3}
    for p, n in nxt.items():
        Wa[n, p] = 3.0
    Wa[3, 6] = 3.0
    Wa[3, 0] = 3.0
    ba = np.array([-5.0, 1.0, 1.0, 2.0, 1.0, -5.0, -1.0, -5.0], np.float32)
    return {"graph": graph, "memory_size": 0,
            "params": {"Wcol": Wcol, "bcol": bcol, "Wc": Wc, "Wgc": Wgc,
                       "Wa": Wa, "ba": ba}}

import numpy as np

def build():
    P = {}
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    Wm = np.zeros((1, 16, 3, 3), np.float32)
    for c in (14, 8, 9):
        for (i, j) in ((1, 1), (1, 2), (2, 1), (2, 2)):
            Wm[0, c, i, j] = 1.0
    P["Wm"] = Wm
    P["bm"] = np.array([-3.0], np.float32)
    W = np.zeros((16, 17, 1, 1), np.float32)
    for d in range(4):
        W[0 * 4 + d, 16, 0, 0] = 1.0
        W[1 * 4 + d, 16, 0, 0] = 1.0
        W[2 * 4 + d, 1, 0, 0] = 1.0
        W[2 * 4 + d, 12, 0, 0] = 1.0
        W[3 * 4 + d, 15, 0, 0] = 1.0
    P["W16"] = W
    S = np.zeros((16, 16, 7, 7), np.float32)
    for c in range(16):
        dx, dy = dirs[c % 4]
        S[c, c, 3 + 3 * dy, 3 + 3 * dx] = 1.0
    P["S"] = S
    Wf = np.zeros((16, 64, 1, 1), np.float32)
    for d in range(4):
        Wf[d, 0 + d] = 1; Wf[d, 16 + 4 + d] = 1; Wf[d, 32 + 8 + d] = 1
        Wf[4 + d, 0 + d] = 1; Wf[4 + d, 16 + 12 + d] = 1; Wf[4 + d, 32 + 8 + d] = 1
        Wf[8 + d, 48 + 2] = 1; Wf[8 + d, 16 + 4 + d] = 1; Wf[8 + d, 32 + d] = 1
        Wf[12 + d, 48 + 2] = 1; Wf[12 + d, 16 + 12 + d] = 1; Wf[12 + d, 32 + d] = 1
    P["Wf"] = Wf
    P["bf"] = np.full((16,), -2.0, np.float32)
    Wr = np.zeros((1, 3, 1, 1), np.float32)
    Wr[0, 0, 0, 0] = -63.0
    Wr[0, 2, 0, 0] = 1.0
    P["Wr"] = Wr
    Wc = np.zeros((1, 34, 1, 1), np.float32)
    for d in range(4):
        Wc[0, 16 + d, 0, 0] = 2.0      # Vp
        Wc[0, 20 + d, 0, 0] = 0.5      # Vu
        Wc[0, 24 + d, 0, 0] = 10.0     # Hp
        Wc[0, 28 + d, 0, 0] = 3.0      # Hu
    Wc[0, 32, 0, 0] = -100.0
    Wc[0, 33, 0, 0] = 0.05
    P["Wc"] = Wc
    # logits from v = [gmax F (16), aprev (8), one (1)]
    Wl = np.zeros((8, 25), np.float32)
    for d in range(4):
        Wl[6, 0 + d] = 10.0
        Wl[6, 8 + d] = 10.0
        Wl[6, 12 + d] = 3.0
        Wl[6, 4 + d] = 2.0
    Wl[6, 16 + 6] = -3.0
    for a in (1, 2, 3, 4):
        Wl[a, 24] = 1.0
    # arrow cycle: 1 -> 4 -> 2 -> 3 -> 1
    nxt = {1: 4, 4: 2, 2: 3, 3: 1}
    for a, b in nxt.items():
        Wl[b, 16 + a] = 0.5
    P["Wl"] = Wl
    graph = [
        {"id": "ml", "op": "conv", "in": ["x0"], "w": "Wm", "b": "bm"},
        {"id": "M", "op": "relu", "in": ["ml"]},
        {"id": "xm", "op": "concat", "in": ["x0", "M"]},
        {"id": "X", "op": "conv", "in": ["xm"], "w": "W16"},
        {"id": "A1", "op": "conv", "in": ["X"], "w": "S"},
        {"id": "A2", "op": "conv", "in": ["A1"], "w": "S"},
        {"id": "A3", "op": "conv", "in": ["A2"], "w": "S"},
        {"id": "A4", "op": "conv", "in": ["A3"], "w": "S"},
        {"id": "cat", "op": "concat", "in": ["X", "A2", "A4", "x0"]},
        {"id": "fl", "op": "conv", "in": ["cat"], "w": "Wf", "b": "bf"},
        {"id": "F", "op": "relu", "in": ["fl"]},
        {"id": "rl", "op": "conv", "in": ["coords"], "w": "Wr"},
        {"id": "rz", "op": "relu", "in": ["rl"]},
        {"id": "cc", "op": "concat", "in": ["x0", "F", "rz", "M"]},
        {"id": "click", "op": "conv", "in": ["cc"], "w": "Wc"},
        {"id": "gF", "op": "gmax", "in": ["F"]},
        {"id": "g1", "op": "gmax", "in": ["rz"]},
        {"id": "v", "op": "vconcat", "in": ["gF", "aprev", "g1"]},
        {"id": "logits", "op": "dense", "in": ["v"], "w": "Wl"},
    ]
    return {"graph": graph, "memory_size": 0, "params": P}

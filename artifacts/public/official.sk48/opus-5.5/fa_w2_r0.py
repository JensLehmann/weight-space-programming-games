import numpy as np


def build():
    N = 256
    shape = [('Tm', 3), ('E', 4), ('R', 3), ('Tp', 2), ('E', 3), ('R', 2), ('Tm', 1), ('E', 2)]

    def macro(m):
        return [m[k] for k, r in shape for _ in range(r)]

    variants = [
        {'Tm': 1, 'Tp': 2, 'E': 4, 'R': 3},   # head facing right (level-1 solution)
        {'Tm': 2, 'Tp': 1, 'E': 4, 'R': 3},   # right, mirrored
        {'Tm': 3, 'Tp': 4, 'E': 2, 'R': 1},   # facing down
        {'Tm': 4, 'Tp': 3, 'E': 2, 'R': 1},   # down, mirrored
        {'Tm': 1, 'Tp': 2, 'E': 3, 'R': 4},   # facing left
        {'Tm': 3, 'Tp': 4, 'E': 1, 'R': 2},   # facing up
    ]
    sched = [7] * 7
    for m in variants:
        sched += macro(m) + [7] * 20
    pad = [4, 4, 3, 1, 4, 4, 3, 2, 4]
    k = 0
    while len(sched) < N:
        sched.append(pad[k % len(pad)])
        k += 1
    sched = sched[:N]

    # phase extraction: e0 at episode start (mem = 0), else e_k
    Wp = np.eye(N, dtype=np.float32)
    Wp[0, :] -= 1.0
    bp = np.zeros(N, np.float32)
    bp[0] = 1.0
    # cyclic shift e_k -> e_{k+1}
    Ws = np.zeros((N, N), np.float32)
    for i in range(N):
        Ws[(i + 1) % N, i] = 1.0

    Wa = np.zeros((8, N), np.float32)
    for i, a in enumerate(sched):
        Wa[a, i] = 10.0
    ba = np.array([-50.0, 0.4, 0.3, 0.2, 0.5, -50.0, -50.0, -1.0], np.float32)

    Wc = np.zeros((1, 3, 1, 1), np.float32)
    bc = np.zeros(1, np.float32)

    graph = [
        {"id": "p", "op": "dense", "in": ["mem"], "w": "Wp", "b": "bp",
         "comment": "phase one-hot"},
        {"id": "mem_next", "op": "dense", "in": ["p"], "w": "Ws",
         "comment": "advance phase"},
        {"id": "logits", "op": "dense", "in": ["p"], "w": "Wa", "b": "ba",
         "comment": "scheduled macro action"},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc", "b": "bc",
         "comment": "unused constant click map"},
    ]
    params = {"Wp": Wp, "bp": bp, "Ws": Ws, "Wa": Wa, "ba": ba, "Wc": Wc, "bc": bc}
    return {"graph": graph, "memory_size": N, "params": params}

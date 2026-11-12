import numpy as np

def build():
    ring = [(0,2),(0,4),(1,1),(1,5),(2,0),(2,6),(4,0),(4,6),(5,1),(5,5),(6,2),(6,4)]
    excl = (0,1,2,3,5,10)
    Wr = np.zeros((16,16,7,7), np.float32)
    for c in range(16):
        if c in excl:
            continue
        for (i,j) in ring:
            Wr[c,c,i,j] = 1.0
    Wsel = np.zeros((16,16,3,3), np.float32)
    for c in range(1,16):
        Wsel[c,c,1,1] = 1.0
        for (i,j) in [(0,1),(1,0),(1,2),(2,1)]:
            Wsel[c,0,i,j] = 1.0
    bsel = np.full((16,), -4.5, np.float32)
    Wsum = np.ones((1,16,1,1), np.float32)
    Wfb = np.full((1,16,1,1), 0.01, np.float32)
    Km = np.zeros((1,1,7,7), np.float32); Km[0,0,3,6] = 1.0
    Kp = np.zeros((1,1,7,7), np.float32); Kp[0,0,3,0] = 1.0
    Wc = np.eye(4, dtype=np.float32); Wc[0,:] -= 1.0
    bc = np.array([1,0,0,0], np.float32)
    Wsh = np.zeros((4,4), np.float32)
    for k in range(4):
        Wsh[(k+1)%4, k] = 1.0
    params = {
        "Wr": Wr, "Wsel": Wsel, "bsel": bsel, "Wsum": Wsum, "Wfb": Wfb,
        "Km": Km, "Kp": Kp, "Wc": Wc, "bc": bc, "Wsh": Wsh,
        "Wl": np.zeros((8,8), np.float32),
        "bl": np.array([0,0,0,0,0,0,1,0], np.float32),
    }
    graph = [
        {"id": "R", "op": "conv", "in": ["x0"], "w": "Wr"},
        {"id": "S", "op": "conv", "in": ["x0"], "w": "Wsel", "b": "bsel"},
        {"id": "Sr", "op": "relu", "in": ["S"]},
        {"id": "g", "op": "gmax", "in": ["Sr"]},
        {"id": "Rg", "op": "mul", "in": ["R", "g"]},
        {"id": "Tsel", "op": "conv", "in": ["Rg"], "w": "Wsum"},
        {"id": "Tany", "op": "conv", "in": ["R"], "w": "Wfb"},
        {"id": "T", "op": "add", "in": ["Tsel", "Tany"]},
        {"id": "Tm", "op": "conv", "in": ["T"], "w": "Km"},
        {"id": "Tp", "op": "conv", "in": ["T"], "w": "Kp"},
        {"id": "X3", "op": "slice", "in": ["x0"], "start": 3, "stop": 4},
        {"id": "cur", "op": "dense", "in": ["mem"], "w": "Wc", "b": "bc"},
        {"id": "mem_next", "op": "dense", "in": ["cur"], "w": "Wsh"},
        {"id": "c0", "op": "vslice", "in": ["cur"], "start": 0, "stop": 1},
        {"id": "c1", "op": "vslice", "in": ["cur"], "start": 1, "stop": 2},
        {"id": "c2", "op": "vslice", "in": ["cur"], "start": 2, "stop": 3},
        {"id": "c3", "op": "vslice", "in": ["cur"], "start": 3, "stop": 4},
        {"id": "c13", "op": "vadd", "in": ["c1", "c3"]},
        {"id": "a0", "op": "mul", "in": ["Tm", "c0"]},
        {"id": "a2", "op": "mul", "in": ["Tp", "c2"]},
        {"id": "a1", "op": "mul", "in": ["X3", "c13"]},
        {"id": "a02", "op": "add", "in": ["a0", "a2"]},
        {"id": "click", "op": "add", "in": ["a02", "a1"]},
        {"id": "logits", "op": "dense", "in": ["aprev"], "w": "Wl", "b": "bl"},
    ]
    return {"graph": graph, "memory_size": 4, "params": params}

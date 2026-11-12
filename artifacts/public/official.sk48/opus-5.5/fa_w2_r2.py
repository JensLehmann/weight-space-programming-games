import numpy as np

def build():
    E, R, Pa, Pb = (1, 0), (-1, 0), (0, 1), (0, -1)
    pattern = [E, E, E, E, Pa, Pa, R, E, Pb, R, R, Pa, E, E, E,
               Pb, Pb, Pb, E, R, Pa, Pa, E, R, R, Pb, E, E, Pa, R]
    P = len(pattern)
    # level-1 script: 6x undo, then the hand-derived solution
    script = [7] * 6 + [1, 1, 1, 4, 4, 4, 4, 3, 2, 2, 4, 3, 1, 4] + [4]
    Q = len(script)

    p = {}
    Wm = np.zeros((2, 3, 1, 1)); Wm[0, 0, 0, 0] = -126.0; Wm[0, 2, 0, 0] = 105.0
    Wm[1, 0, 0, 0] = -126.0; Wm[1, 2, 0, 0] = 104.0
    p["Wm"] = Wm
    Wu = np.zeros((4, 6))
    for i, j in enumerate([5, 3, 4, 2]): Wu[i, j] = 4096.0
    Wv = np.zeros((4, 6))
    for i, j in enumerate([0, 1, 0, 1]): Wv[i, j] = 4096.0
    p["Wu"] = Wu; p["Wv"] = Wv
    p["Wd"] = np.array([[10, -10, 0, 0], [0, 0, 10, -10], [-10, 10, 0, 0], [0, 0, -10, 10]], float)
    p["bd"] = -np.ones(4)
    p["Wq"] = 5.0 * np.array([[1, 0, -1, 0], [0, 1, 0, -1]], float)
    p["Ws"] = -np.ones((1, P)); p["bs"] = np.ones(1)
    We = np.zeros((P, 1)); We[0, 0] = 1.0; p["We"] = We
    Wp = np.zeros((P, P))
    for i in range(P): Wp[(i + 1) % P, i] = 1.0
    p["Wp"] = Wp
    Wab = np.zeros((2, P))
    for j, (a, b) in enumerate(pattern): Wab[0, j] = a; Wab[1, j] = b
    p["Wab"] = Wab
    p["Wx"] = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    p["Wy"] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
    Wl = np.zeros((8, 4))
    Wl[1] = [0, -1, -1, 0]; Wl[2] = [0, 1, 1, 0]
    Wl[3] = [-1, 0, 0, -1]; Wl[4] = [1, 0, 0, 1]
    p["Wl"] = Wl
    p["Wid"] = np.eye(8)
    p["bl"] = np.array([-20, 0.001, 0.002, 0.003, 0.004, -20, -20, -10], float)
    p["Wc"] = np.zeros((1, 3, 1, 1))
    # gate
    p["Wg"] = np.array([[-0.05 * 4096.0]]); p["bg"] = np.array([65.0])
    p["Wn"] = np.array([[-1.0]])
    # script register
    p["Ws2"] = -np.ones((1, Q))
    We2 = np.zeros((Q, 1)); We2[0, 0] = 1.0; p["We2"] = We2
    Wp2 = np.zeros((Q, Q))
    for i in range(Q): Wp2[min(i + 1, Q - 1), i] = 1.0
    p["Wp2"] = Wp2
    Wsc = np.zeros((8, Q))
    for j, a in enumerate(script): Wsc[a, j] = 30.0
    p["Wsc"] = Wsc

    g = [
        {"id": "mrow", "op": "conv", "in": ["coords"], "w": "Wm"},
        {"id": "mr", "op": "relu", "in": ["mrow"]},
        {"id": "ma", "op": "slice", "in": ["mr"], "start": 0, "stop": 1},
        {"id": "mb", "op": "slice", "in": ["mr"], "start": 1, "stop": 2},
        {"id": "mask", "op": "sub", "in": ["ma", "mb"]},
        {"id": "A", "op": "slice", "in": ["x0"], "start": 0, "stop": 2},
        {"id": "AM", "op": "mul", "in": ["A", "mask"]},
        {"id": "RC", "op": "slice", "in": ["coords"], "start": 0, "stop": 2},
        {"id": "A0", "op": "slice", "in": ["AM"], "start": 0, "stop": 1},
        {"id": "A1", "op": "slice", "in": ["AM"], "start": 1, "stop": 2},
        {"id": "P0", "op": "mul", "in": ["A0", "RC"]},
        {"id": "P1", "op": "mul", "in": ["A1", "RC"]},
        {"id": "F", "op": "concat", "in": ["AM", "P0", "P1"]},
        {"id": "G", "op": "gmean", "in": ["F"]},
        {"id": "U", "op": "dense", "in": ["G"], "w": "Wu"},
        {"id": "V", "op": "dense", "in": ["G"], "w": "Wv"},
        {"id": "Qp", "op": "vmul", "in": ["U", "V"]},
        {"id": "Dpn", "op": "dense", "in": ["Qp"], "w": "Wd", "b": "bd"},
        {"id": "Rz", "op": "vrelu", "in": ["Dpn"]},
        {"id": "dirl", "op": "dense", "in": ["Rz"], "w": "Wq"},
        {"id": "dir", "op": "vtanh", "in": ["dirl"]},
        {"id": "A4", "op": "slice", "in": ["x0"], "start": 4, "stop": 5},
        {"id": "A4m", "op": "mul", "in": ["A4", "mask"]},
        {"id": "g4", "op": "gmean", "in": ["A4m"]},
        {"id": "gl", "op": "dense", "in": ["g4"], "w": "Wg", "b": "bg"},
        {"id": "gate", "op": "vsigmoid", "in": ["gl"]},
        {"id": "ngate", "op": "dense", "in": ["gate"], "w": "Wn", "b": "bs"},
        {"id": "mc", "op": "vslice", "in": ["mem"], "start": 0, "stop": P},
        {"id": "ms", "op": "vslice", "in": ["mem"], "start": P, "stop": P + Q},
        {"id": "S", "op": "dense", "in": ["mc"], "w": "Ws", "b": "bs"},
        {"id": "E0", "op": "dense", "in": ["S"], "w": "We"},
        {"id": "cur", "op": "vadd", "in": ["mc", "E0"]},
        {"id": "nxc", "op": "dense", "in": ["cur"], "w": "Wp"},
        {"id": "S2", "op": "dense", "in": ["ms"], "w": "Ws2", "b": "bs"},
        {"id": "E2", "op": "dense", "in": ["S2"], "w": "We2"},
        {"id": "cur2", "op": "vadd", "in": ["ms", "E2"]},
        {"id": "nxs", "op": "dense", "in": ["cur2"], "w": "Wp2"},
        {"id": "mem_next", "op": "vconcat", "in": ["nxc", "nxs"]},
        {"id": "ab", "op": "dense", "in": ["cur"], "w": "Wab"},
        {"id": "Xs", "op": "dense", "in": ["dir"], "w": "Wx"},
        {"id": "Ys", "op": "dense", "in": ["ab"], "w": "Wy"},
        {"id": "T", "op": "vmul", "in": ["Xs", "Ys"]},
        {"id": "swl", "op": "dense", "in": ["T"], "w": "Wl"},
        {"id": "sl", "op": "dense", "in": ["cur2"], "w": "Wsc"},
        {"id": "a1", "op": "vmul", "in": ["sl", "gate"]},
        {"id": "a2", "op": "vmul", "in": ["swl", "ngate"]},
        {"id": "asum", "op": "vadd", "in": ["a1", "a2"]},
        {"id": "logits", "op": "dense", "in": ["asum"], "w": "Wid", "b": "bl"},
        {"id": "click", "op": "conv", "in": ["coords"], "w": "Wc"},
    ]
    p = {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}
    return {"graph": g, "memory_size": P + Q, "params": p}

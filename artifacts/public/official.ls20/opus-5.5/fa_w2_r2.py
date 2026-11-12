import numpy as np

def build():
    P = {}
    K = 1000.0
    w = np.zeros((4, 3, 1, 1), np.float32)
    w[0, :, 0, 0] = [-K*63, 0, K*59.5]
    w[1, :, 0, 0] = [0, K*63, -K*3.5]
    w[2, :, 0, 0] = [K*63, 0, -K*52.5]
    w[3, :, 0, 0] = [0, -K*63, K*11.5]
    P["Wpm"] = w
    w = np.zeros((1, 16, 1, 1), np.float32)
    for c in [0, 1, 3, 5, 8, 9, 11, 12, 14]:
        w[0, c, 0, 0] = 1.0
    P["Ww"] = w
    w = np.zeros((1, 16, 1, 1), np.float32)
    w[0, 5, 0, 0] = 1.0
    w[0, 11, 0, 0] = 0.95
    P["Wg"] = w
    w = np.zeros((1, 16, 1, 1), np.float32)
    for c in [0, 1, 8, 14]:
        w[0, c, 0, 0] = 1.0
    P["Wmod"] = w
    w = np.zeros((4, 1, 3, 3), np.float32)
    w[0, 0, 0, 1] = 1; w[1, 0, 2, 1] = 1; w[2, 0, 1, 0] = 1; w[3, 0, 1, 2] = 1
    P["Wsh"] = w
    P["bsh"] = np.full((4,), -1.0/1024, np.float32)
    w = np.zeros((1, 16, 5, 5), np.float32)
    w[0, 12, 0:2, :] = 1.0
    w[0, 9, 2:5, :] = 1.0
    P["Wa"] = w
    P["ba"] = np.array([-24.0], np.float32)
    w = np.zeros((4, 1, 7, 7), np.float32)
    w[0, 0, 0, 3] = 1; w[1, 0, 6, 3] = 1; w[2, 0, 3, 0] = 1; w[3, 0, 3, 6] = 1
    P["Wn"] = w
    P["Wd"] = np.array([[-4096.0, 1.0, 0.0]], np.float32)
    P["bd"] = np.array([-0.5], np.float32)
    P["Wt"] = np.array([[3.0, 0.0, 3.0]], np.float32)
    P["Wm0"] = np.array([[4096.0]], np.float32)
    P["Wsf"] = np.array([[-0.98]], np.float32)
    P["bsf"] = np.array([0.98], np.float32)
    P["Wdf"] = np.array([[1,0,-1,0],[0,1,0,-1],[-1,0,1,0],[0,-1,0,1]], np.float32)
    P["Wsm"] = np.full((1, 4), -200.0, np.float32)
    P["bsm"] = np.array([1.0], np.float32)
    wo = np.zeros((8, 20), np.float32)
    for d in range(4):
        wo[1+d, d] = 1000.0
        wo[1+d, 12+1+d] = -5000.0
    rev = {1: 2, 2: 1, 3: 4, 4: 3}
    for a, r in rev.items():
        wo[r, 4+a] = -8.0
    P["Wo"] = wo
    bo = np.full((8,), -1e4, np.float32)
    bo[1:5] = 0.0
    P["bo"] = bo
    P["Wc"] = np.ones((1, 1, 1, 1), np.float32)

    def body(p):
        return [
            {"id": p+"c", "op": "conv", "in": [p+"v"], "w": "Wsh", "b": "bsh"},
            {"id": p+"s0", "op": "slice", "in": [p+"c"], "start": 0, "stop": 1},
            {"id": p+"s1", "op": "slice", "in": [p+"c"], "start": 1, "stop": 2},
            {"id": p+"s2", "op": "slice", "in": [p+"c"], "start": 2, "stop": 3},
            {"id": p+"s3", "op": "slice", "in": [p+"c"], "start": 3, "stop": 4},
            {"id": p+"m1", "op": "max2", "in": [p+"s0", p+"s1"]},
            {"id": p+"m2", "op": "max2", "in": [p+"s2", p+"s3"]},
            {"id": p+"m3", "op": "max2", "in": [p+"m1", p+"m2"]},
            {"id": p+"m4", "op": "max2", "in": [p+"m3", p+"v"]},
            {"id": p+"n", "op": "mul", "in": [p+"m4", "Wp"]},
        ]

    g = [
        {"id": "pm", "op": "conv", "in": ["coords"], "w": "Wpm"},
        {"id": "ps", "op": "sigmoid", "in": ["pm"]},
        {"id": "q0", "op": "slice", "in": ["ps"], "start": 0, "stop": 1},
        {"id": "q1", "op": "slice", "in": ["ps"], "start": 1, "stop": 2},
        {"id": "q2", "op": "slice", "in": ["ps"], "start": 2, "stop": 3},
        {"id": "q3", "op": "slice", "in": ["ps"], "start": 3, "stop": 4},
        {"id": "M1", "op": "mul", "in": ["q0", "q1"]},
        {"id": "T1", "op": "mul", "in": ["q2", "q3"]},
        {"id": "MT", "op": "mul", "in": ["M1", "T1"]},
        {"id": "M", "op": "sub", "in": ["M1", "MT"]},
        {"id": "w0", "op": "conv", "in": ["x0"], "w": "Ww"},
        {"id": "Wp", "op": "mul", "in": ["w0", "M"]},
        {"id": "g0", "op": "conv", "in": ["x0"], "w": "Wg"},
        {"id": "gm", "op": "mul", "in": ["g0", "M"]},
        {"id": "mc", "op": "conv", "in": ["x0"], "w": "Wmod"},
        {"id": "modm", "op": "mul", "in": ["mc", "M"]},
        {"id": "cnt", "op": "gmean", "in": ["modm"]},
        {"id": "mA", "op": "vslice", "in": ["mem"], "start": 0, "stop": 2},
        {"id": "mP", "op": "vslice", "in": ["mem"], "start": 2, "stop": 4},
        {"id": "u", "op": "vconcat", "in": ["cnt", "mA"]},
        {"id": "dl", "op": "dense", "in": ["u"], "w": "Wd", "b": "bd"},
        {"id": "dr", "op": "vrelu", "in": ["dl"]},
        {"id": "u2", "op": "vconcat", "in": ["dr", "mA"]},
        {"id": "tl", "op": "dense", "in": ["u2"], "w": "Wt"},
        {"id": "m1n", "op": "vtanh", "in": ["tl"]},
        {"id": "m0n", "op": "dense", "in": ["cnt"], "w": "Wm0"},
        {"id": "sf", "op": "dense", "in": ["m1n"], "w": "Wsf", "b": "bsf"},
        {"id": "ms", "op": "mul", "in": ["modm", "sf"]},
        {"id": "src", "op": "add", "in": ["gm", "ms"]},
        {"id": "L1", "op": "loop", "T": 256,
         "state": [{"id": "La_v", "init": "src"}],
         "body": body("La_"), "next": {"La_v": "La_n"}},
        {"id": "L2", "op": "loop", "T": 256,
         "state": [{"id": "Lb_v", "init": "L1.La_v"}],
         "body": body("Lb_"), "next": {"Lb_v": "Lb_n"}},
        {"id": "anc", "op": "conv", "in": ["x0"], "w": "Wa", "b": "ba"},
        {"id": "A", "op": "relu", "in": ["anc"]},
        {"id": "cA", "op": "mul", "in": ["A", "coords"]},
        {"id": "p3", "op": "gmax", "in": ["cA"]},
        {"id": "pos", "op": "vslice", "in": ["p3"], "start": 0, "stop": 2},
        {"id": "pp", "op": "vconcat", "in": ["pos", "mP"]},
        {"id": "df", "op": "dense", "in": ["pp"], "w": "Wdf"},
        {"id": "dfr", "op": "vrelu", "in": ["df"]},
        {"id": "sl", "op": "dense", "in": ["dfr"], "w": "Wsm", "b": "bsm"},
        {"id": "same", "op": "vrelu", "in": ["sl"]},
        {"id": "pen", "op": "vmul", "in": ["same", "aprev"]},
        {"id": "mem_next", "op": "vconcat", "in": ["m0n", "m1n", "pos"]},
        {"id": "nb", "op": "conv", "in": ["L2.Lb_v"], "w": "Wn"},
        {"id": "nA", "op": "mul", "in": ["nb", "A"]},
        {"id": "qv", "op": "gmax", "in": ["nA"]},
        {"id": "qa", "op": "vconcat", "in": ["qv", "aprev", "pen"]},
        {"id": "logits", "op": "dense", "in": ["qa"], "w": "Wo", "b": "bo"},
        {"id": "click", "op": "conv", "in": ["A"], "w": "Wc"},
    ]
    return {"graph": g, "memory_size": 4, "params": P}

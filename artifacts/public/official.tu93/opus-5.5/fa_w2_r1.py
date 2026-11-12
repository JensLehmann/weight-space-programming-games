import numpy as np

def build():
    D = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    P = {}
    f = np.float32
    W_cb = np.zeros((5, 16, 1, 1), f)
    for c in (15, 11): W_cb[0, c] = 1
    W_cb[1, 8] = 1
    for c in (8, 12, 13): W_cb[2, c] = 1
    for c in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15): W_cb[3, c] = 1
    for c in (9, 4): W_cb[4, c] = 1
    P["W_cb"] = W_cb
    W_ed = np.zeros((8, 3, 3, 3), f)
    for t in range(2):
        for d, (dy, dx) in enumerate(D):
            o = 4 * t + d
            W_ed[o, 0, 1, 1] = 1
            W_ed[o, 1 + t, 1 - dy, 1 - dx] += 1
            W_ed[o, 1 + t, 1 + dy, 1 + dx] -= 1
    P["W_ed"] = W_ed
    P["b_ed"] = -np.ones(8, f)
    W_A = np.zeros((8, 8, 7, 7), f)
    for t in range(2):
        for d, (dy, dx) in enumerate(D):
            W_A[4 * t + d, 4 * t + d, 3 - 3 * dy, 3 - 3 * dx] = 1
    P["W_A"] = W_A
    W_H = np.zeros((2, 8, 7, 7), f)
    for t in range(2):
        for d, (dy, dx) in enumerate(D):
            py, px = abs(dx), abs(dy)
            for s1 in (1, 2, 3):
                for s in (-1, 0, 1):
                    W_H[t, 4 * t + d, 3 - (s1 * dy + s * py), 3 - (s1 * dx + s * px)] = 1
    P["W_H"] = W_H
    W_r = np.zeros((1, 3, 1, 1), f)
    W_r[0, 0] = 126.0
    W_r[0, 2] = -125.0
    P["W_r"] = W_r
    W_neg = np.zeros((1, 2, 1, 1), f)
    W_neg[0, 0] = -1.0
    P["W_neg"] = W_neg
    P["b_one"] = np.ones(1, f)
    W_bm = np.zeros((1, 5, 1, 1), f)
    W_bm[0, 1] = 0.9
    P["W_bm"] = W_bm
    g = 0.97
    W_l = np.zeros((8, 2, 7, 7), f)
    for d, (dy, dx) in enumerate(D):
        for c in range(2):
            W_l[2 * d + c, c, 3 + 3 * dy, 3 + 3 * dx] = g
    P["W_l"] = W_l
    W_s = np.zeros((8, 2, 7, 7), f)
    for k in range(2):
        for d, (dy, dx) in enumerate(D):
            W_s[4 * k + d, k, 3 + 3 * dy, 3 + 3 * dx] = 1
    P["W_s"] = W_s
    W_hb = np.zeros((4, 1, 7, 7), f)
    W_hc = np.zeros((4, 4, 7, 7), f)
    for d, (dy, dx) in enumerate(D):
        W_hb[d, 0, 3 + 3 * dy, 3 + 3 * dx] = 1
        W_hc[d, d, 3 + 3 * dy, 3 + 3 * dx] = 1
    P["W_hb"] = W_hb
    P["W_hc"] = W_hc
    W_o = np.zeros((8, 12), f)
    for d in range(4):
        W_o[1 + d, d] = 1e6
        W_o[1 + d, 4 + d] = 1.0
        W_o[1 + d, 8 + d] = -1e7
    P["W_o"] = W_o
    b_o = np.zeros(8, f)
    b_o[0] = -1e9
    P["b_o"] = b_o
    W_rev = np.zeros((8, 8), f)
    opp = {1: 2, 2: 1, 3: 4, 4: 3}
    for a, o in opp.items():
        W_rev[o, a] = -8e4
    P["W_rev"] = W_rev
    P["W_c"] = np.ones((1, 1, 1, 1), f)

    graph = [
        {"id": "cb", "op": "conv", "in": ["x0"], "w": "W_cb"},
        {"id": "eb", "op": "slice", "in": ["cb"], "start": 0, "stop": 3},
        {"id": "pm", "op": "slice", "in": ["cb"], "start": 3, "stop": 4},
        {"id": "pl", "op": "slice", "in": ["cb"], "start": 4, "stop": 5},
        {"id": "bm", "op": "conv", "in": ["cb"], "w": "W_bm"},
        {"id": "ed0", "op": "conv", "in": ["eb"], "w": "W_ed", "b": "b_ed"},
        {"id": "ed", "op": "relu", "in": ["ed0"]},
        {"id": "A", "op": "conv", "in": ["ed"], "w": "W_A"},
        {"id": "Hr", "op": "conv", "in": ["A"], "w": "W_H"},
        {"id": "Hall", "op": "slice", "in": ["Hr"], "start": 1, "stop": 2},
        {"id": "r63l", "op": "conv", "in": ["coords"], "w": "W_r"},
        {"id": "r63", "op": "relu", "in": ["r63l"]},
        {"id": "mrl", "op": "sub", "in": ["pm", "r63"]},
        {"id": "mraw", "op": "relu", "in": ["mrl"]},
        {"id": "hneg", "op": "conv", "in": ["Hr"], "w": "W_neg", "b": "b_one"},
        {"id": "hs", "op": "relu", "in": ["hneg"]},
        {"id": "msafe", "op": "mul", "in": ["mraw", "hs"]},
        {"id": "M2", "op": "concat", "in": ["msafe", "mraw"]},
        {"id": "src", "op": "slice", "in": ["x0"], "start": 14, "stop": 15},
        {"id": "src2", "op": "concat", "in": ["src", "src"]},
        {"id": "L", "op": "loop", "T": 256,
         "state": [{"id": "v", "zeros": [2, 64, 64]}],
         "body": [
             {"id": "lc", "op": "conv", "in": ["v"], "w": "W_l"},
             {"id": "s1", "op": "slice", "in": ["lc"], "start": 0, "stop": 2},
             {"id": "s2", "op": "slice", "in": ["lc"], "start": 2, "stop": 4},
             {"id": "s3", "op": "slice", "in": ["lc"], "start": 4, "stop": 6},
             {"id": "s4", "op": "slice", "in": ["lc"], "start": 6, "stop": 8},
             {"id": "m12", "op": "max2", "in": ["s1", "s2"]},
             {"id": "m34", "op": "max2", "in": ["s3", "s4"]},
             {"id": "mx", "op": "max2", "in": ["m12", "m34"]},
             {"id": "ls", "op": "max2", "in": ["mx", "src2"]},
             {"id": "ln", "op": "mul", "in": ["ls", "M2"]},
             {"id": "vr", "op": "slice", "in": ["v"], "start": 1, "stop": 2},
             {"id": "bon", "op": "mul", "in": ["vr", "bm"]},
             {"id": "lb", "op": "max2", "in": ["ln", "bon"]},
         ],
         "next": {"v": "lb"}},
        {"id": "vs8", "op": "conv", "in": ["L.v"], "w": "W_s"},
        {"id": "vp", "op": "mul", "in": ["vs8", "pl"]},
        {"id": "gv", "op": "gmax", "in": ["vp"]},
        {"id": "hb", "op": "conv", "in": ["Hall"], "w": "W_hb"},
        {"id": "hc", "op": "conv", "in": ["hb"], "w": "W_hc"},
        {"id": "hp", "op": "mul", "in": ["hc", "pl"]},
        {"id": "gh", "op": "gmax", "in": ["hp"]},
        {"id": "g", "op": "vconcat", "in": ["gv", "gh"]},
        {"id": "lg", "op": "dense", "in": ["g"], "w": "W_o", "b": "b_o"},
        {"id": "rev", "op": "dense", "in": ["aprev"], "w": "W_rev"},
        {"id": "logits", "op": "vadd", "in": ["lg", "rev"]},
        {"id": "click", "op": "conv", "in": ["pl"], "w": "W_c"},
    ]
    return {"graph": graph, "memory_size": 0, "params": P}

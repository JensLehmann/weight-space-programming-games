import numpy as np


def build():
    D = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def put(w, o, i, dr, dc, val):
        c = w.shape[-1] // 2
        w[o, i, c + dr, c + dc] += val

    P = {}
    NF = 9
    Wf = np.zeros((NF, 16, 1, 1), np.float32)
    for col in (0, 2, 4, 8, 9, 11, 12, 13, 14, 15):
        Wf[0, col] = 1
        Wf[8, col] = 0.01
    Wf[1, 14] = 1
    for col in (8, 12, 13):
        Wf[2, col] = 1
    Wf[3, 15] = 1
    Wf[3, 11] = 1
    Wf[4, 13] = 1
    Wf[5, 9] = 1
    Wf[5, 4] = 1
    Wf[6, 8] = 0.9
    Wf[7, 15] = 1
    P["Wf"] = Wf

    Wdir = np.zeros((4, NF, 3, 3), np.float32)
    Wdir13 = np.zeros((4, NF, 3, 3), np.float32)
    for d, (dr, dc) in enumerate(D):
        put(Wdir, d, 3, 0, 0, 1.0)
        put(Wdir, d, 2, -dr, -dc, 1.0)
        put(Wdir, d, 2, dr, dc, -1.0)
        put(Wdir13, d, 7, 0, 0, 1.0)
        put(Wdir13, d, 4, -dr, -dc, 1.0)
        put(Wdir13, d, 4, dr, dc, -1.0)
    P["Wdir"] = Wdir
    P["Wdir13"] = Wdir13
    P["bdir"] = -np.ones(4, np.float32)

    Wa = np.zeros((4, 4, 7, 7), np.float32)
    Wsh2 = np.zeros((4, 4, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(D):
        put(Wa, d, d, -3 * dr, -3 * dc, 1.0)
        put(Wsh2, d, d, 3 * dr, 3 * dc, 1.0)
    P["Wa"] = Wa
    P["Wsh2"] = Wsh2
    Wb = np.zeros((1, 4, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(D):
        for er in (-1, 0, 1):
            for ec in (-1, 0, 1):
                put(Wb, 0, d, -2 * dr + er, -2 * dc + ec, 1.0)
    P["Wb"] = Wb

    a = 0.98
    Wsh = np.zeros((8, 1, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(D):
        put(Wsh, d, 0, dr, dc, a)
        put(Wsh, 4 + d, 0, 3 * dr, 3 * dc, a ** 3)
    P["Wsh"] = Wsh

    Wsc = np.zeros((4, 1, 7, 7), np.float32)
    for d, (dr, dc) in enumerate(D):
        put(Wsc, d, 0, 3 * dr, 3 * dc, 1.0)
    P["Wsc"] = Wsc

    P["Wneg"] = -np.eye(4, dtype=np.float32)
    P["bone"] = np.ones(4, np.float32)
    P["W99"] = 0.99 * np.eye(4, dtype=np.float32)
    P["b01"] = np.full(4, 0.01, np.float32)

    Wl = np.zeros((8, 4), np.float32)
    bl = np.full(8, -10.0, np.float32)
    for d in range(4):
        Wl[1 + d, d] = 1000.0
        bl[1 + d] = 0.0
    P["Wl"] = Wl
    P["bl"] = bl

    body = [{"id": "s", "op": "conv", "in": ["v"], "w": "Wsh"}]
    for k in range(8):
        body.append({"id": "s%d" % k, "op": "slice", "in": ["s"], "start": k, "stop": k + 1})
    body += [
        {"id": "m01", "op": "max2", "in": ["s0", "s1"]},
        {"id": "m23", "op": "max2", "in": ["s2", "s3"]},
        {"id": "m45", "op": "max2", "in": ["s4", "s5"]},
        {"id": "m67", "op": "max2", "in": ["s6", "s7"]},
        {"id": "ma", "op": "max2", "in": ["m01", "m23"]},
        {"id": "mb", "op": "max2", "in": ["m45", "m67"]},
        {"id": "mc", "op": "max2", "in": ["ma", "mb"]},
        {"id": "mw", "op": "mul", "in": ["mc", "walk"]},
        {"id": "vn", "op": "max2", "in": ["mw", "seedall"]},
    ]

    graph = [
        {"id": "feat", "op": "conv", "in": ["x0"], "w": "Wf"},
        {"id": "walkraw", "op": "slice", "in": ["feat"], "start": 0, "stop": 1},
        {"id": "seed", "op": "slice", "in": ["feat"], "start": 1, "stop": 2},
        {"id": "pl", "op": "slice", "in": ["feat"], "start": 5, "stop": 6},
        {"id": "red", "op": "slice", "in": ["feat"], "start": 6, "stop": 7},
        {"id": "wfloor", "op": "slice", "in": ["feat"], "start": 8, "stop": 9},
        {"id": "seedall", "op": "max2", "in": ["seed", "red"]},
        {"id": "Fpre", "op": "conv", "in": ["feat"], "w": "Wdir", "b": "bdir"},
        {"id": "Fd", "op": "relu", "in": ["Fpre"]},
        {"id": "dA", "op": "conv", "in": ["Fd"], "w": "Wa"},
        {"id": "F13pre", "op": "conv", "in": ["feat"], "w": "Wdir13", "b": "bdir"},
        {"id": "Fd13", "op": "relu", "in": ["F13pre"]},
        {"id": "t1", "op": "conv", "in": ["Fd13"], "w": "Wa"},
        {"id": "t2", "op": "conv", "in": ["t1"], "w": "Wa"},
        {"id": "t3", "op": "conv", "in": ["t2"], "w": "Wa"},
        {"id": "dsum", "op": "add", "in": ["dA", "t3"]},
        {"id": "dB", "op": "conv", "in": ["dsum"], "w": "Wb"},
        {"id": "wsub", "op": "sub", "in": ["walkraw", "dB"]},
        {"id": "wrel", "op": "relu", "in": ["wsub"]},
        {"id": "walk", "op": "max2", "in": ["wrel", "wfloor"]},
        {"id": "V", "op": "loop", "T": 200,
         "state": [{"id": "v", "init": "seedall"}],
         "body": body, "next": {"v": "vn"}},
        {"id": "sc", "op": "conv", "in": ["V.v"], "w": "Wsc"},
        {"id": "scp", "op": "mul", "in": ["sc", "pl"]},
        {"id": "g", "op": "gmax", "in": ["scp"]},
        {"id": "dz1", "op": "conv", "in": ["dB"], "w": "Wsc"},
        {"id": "dz2", "op": "conv", "in": ["dz1"], "w": "Wsh2"},
        {"id": "dzp", "op": "mul", "in": ["dz2", "pl"]},
        {"id": "dg", "op": "gmax", "in": ["dzp"]},
        {"id": "fl", "op": "dense", "in": ["dg"], "w": "Wneg", "b": "bone"},
        {"id": "f", "op": "vrelu", "in": ["fl"]},
        {"id": "fac", "op": "dense", "in": ["f"], "w": "W99", "b": "b01"},
        {"id": "gs", "op": "vmul", "in": ["g", "fac"]},
        {"id": "logits", "op": "dense", "in": ["gs"], "w": "Wl", "b": "bl"},
        {"id": "click", "op": "slice", "in": ["feat"], "start": 0, "stop": 1},
    ]
    return {"graph": graph, "memory_size": 0,
            "params": {k: np.asarray(v, np.float32) for k, v in P.items()}}

import numpy as np, math

def build():
    P = {}
    Wm = np.zeros((6, 6), np.float32); bm = np.zeros(6, np.float32)
    Rs = []
    for k, per in enumerate([7.3, 11.7, 17.9]):
        th = 2 * math.pi / per
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        Rs.append(R)
        Wm[2*k:2*k+2, 2*k:2*k+2] = R
        bm[2*k] = 1.0
    P["Wm"] = Wm; P["bm"] = bm
    cfix = np.linalg.solve(np.eye(2) - Rs[2], np.array([1.0, 0.0]))
    rows = []
    for ax in (0, 1):
        for j in range(11):
            rows.append((ax, 6.0 * j))
        for j in range(1, 7):
            rows.append((ax, 9.0 * j))
    nK = len(rows)
    Wk = np.zeros((nK, 3, 1, 1), np.float32)
    for i, (ax, off) in enumerate(rows):
        Wk[i, ax, 0, 0] = 63.0
        Wk[i, 2, 0, 0] = -off
    P["Wk"] = Wk
    WF = np.zeros((6, nK, 1, 1), np.float32)
    for ax in (0, 1):
        base = ax * 17
        WF[ax, base, 0, 0] = 1.0 / 63.0
        WF[2 + ax, base, 0, 0] = 1.0 / 6.0
        for j in range(1, 11):
            WF[2 + ax, base + j, 0, 0] = 2.0 * ((-1) ** j) / 6.0
        WF[4 + ax, base, 0, 0] = 1.0 / 9.0
        for j in range(1, 7):
            WF[4 + ax, base + 10 + j, 0, 0] = 2.0 * ((-1) ** j) / 9.0
    P["WF"] = WF
    P["Ws"] = np.ones((1, 6, 1, 1), np.float32)
    P["bs"] = np.array([40.0], np.float32)
    # colour features
    Wf = np.zeros((8, 16, 1, 1), np.float32)
    for c in (14, 8): Wf[0, c] = 1.0
    for c in (14, 8, 15, 7): Wf[1, c] = 1.0
    for c in (1, 12): Wf[2, c] = 1.0
    Wf[3, 11] = 1.0
    Wf[4, 15] = 1.0
    Wf[5, 2] = 1.0
    Wf[6, 9] = 1.0
    Wf[7, 0] = 1.0; Wf[7, 1] = 1.0
    P["Wf"] = Wf
    P["Wbox"] = np.ones((1, 1, 3, 3), np.float32)
    P["Wone"] = np.ones((1, 1, 1, 1), np.float32)
    P["bneg"] = np.array([-1.0], np.float32)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    Wsa = np.zeros((4, 1, 7, 7), np.float32)
    Wsb = np.zeros((4, 4, 7, 7), np.float32)
    for i, (dr, dc) in enumerate(dirs):
        Wsa[i, 0, 3 + 3*dr, 3 + 3*dc] = 1.0
        Wsb[i, i, 3 + 3*dr, 3 + 3*dc] = 1.0
    P["Wsa"] = Wsa; P["Wsb"] = Wsb
    P["Wj"] = np.ones((1, 4, 1, 1), np.float32)
    Wt = np.zeros((4, 8, 1, 1), np.float32)
    for d in range(4):
        Wt[d, d] = 1.0; Wt[d, 4 + d] = 1.0
    P["Wt"] = Wt; P["bt"] = np.ones(4, np.float32)
    Wr = np.zeros((4, 3, 1, 1), np.float32); br = np.zeros(4, np.float32)
    Wr[0, 0] = 63.0
    Wr[1, 0] = 63.0; br[1] = -1.0
    Wr[2, 0] = 63.0; br[2] = -57.5
    Wr[3, 1] = -63.0; br[3] = 8.5
    P["Wr"] = Wr; P["br"] = br
    P["Wrp"] = np.array([1.0, -1.0], np.float32).reshape(1, 2, 1, 1)
    P["Wc"] = np.array([1e6, 1000.0, 10.0, 0.02, 10.0, 0.02], np.float32).reshape(1, 6, 1, 1)
    Wl = np.zeros((8, 10), np.float32); bl = np.zeros(8, np.float32)
    Wl[6, 6] = 400.0; Wl[6, 7] = 100.0; Wl[6, 8] = 100.0; bl[6] = -10.0
    bl[7] = 2.5
    bl[0] = -100.0; bl[5] = -100.0
    for i, a in enumerate((1, 2, 3, 4)):
        ang = i * math.pi / 2
        u = np.array([math.cos(ang), math.sin(ang)])
        Wl[a, 4] = u[0]; Wl[a, 5] = u[1]
        Wl[a, 9] = 10.0
        bl[a] = -float(u @ cfix) - 10.0
    P["Wl"] = Wl; P["bl"] = bl
    def sl(i, a, b, s):
        return {"id": i, "op": "slice", "in": [a], "start": b, "stop": s}
    g = [
        {"id": "mem_next", "op": "dense", "in": ["mem"], "w": "Wm", "b": "bm"},
        {"id": "kl", "op": "conv", "in": ["coords"], "w": "Wk"},
        {"id": "kr", "op": "relu", "in": ["kl"]},
        {"id": "F", "op": "conv", "in": ["kr"], "w": "WF"},
        {"id": "S", "op": "mul", "in": ["F", "mem_next"]},
        {"id": "sc", "op": "conv", "in": ["S"], "w": "Ws", "b": "bs"},
        {"id": "ft", "op": "conv", "in": ["x0"], "w": "Wf"},
        sl("G", "ft", 0, 1), sl("Mb", "ft", 1, 2), sl("L", "ft", 2, 3),
        sl("Y", "ft", 3, 4), sl("PU", "ft", 4, 5), sl("GR", "ft", 5, 6),
        sl("BL", "ft", 6, 7), sl("W01", "ft", 7, 8),
        {"id": "box", "op": "conv", "in": ["W01"], "w": "Wbox"},
        {"id": "boxm", "op": "conv", "in": ["box"], "w": "Wone", "b": "bneg"},
        {"id": "boxr", "op": "relu", "in": ["boxm"]},
        {"id": "ind", "op": "sub", "in": ["box", "boxr"]},
        {"id": "Bp", "op": "mul", "in": ["BL", "ind"]},
        {"id": "M", "op": "add", "in": ["Mb", "Bp"]},
        {"id": "Ls1", "op": "conv", "in": ["L"], "w": "Wsa"},
        {"id": "Ls2", "op": "conv", "in": ["Ls1"], "w": "Wsb"},
        {"id": "A", "op": "mul", "in": ["Ls2", "M"]},
        {"id": "A1", "op": "conv", "in": ["A"], "w": "Wsb"},
        {"id": "J", "op": "conv", "in": ["A1"], "w": "Wsb"},
        {"id": "Jsum", "op": "conv", "in": ["J"], "w": "Wj"},
        {"id": "JP", "op": "mul", "in": ["G", "Jsum"]},
        {"id": "Gs1", "op": "conv", "in": ["G"], "w": "Wsa"},
        {"id": "Gs", "op": "conv", "in": ["Gs1"], "w": "Wsb"},
        {"id": "Npeg", "op": "conv", "in": ["Gs"], "w": "Wj"},
        {"id": "Nn1", "op": "conv", "in": ["Npeg"], "w": "Wsa"},
        {"id": "Nn2", "op": "conv", "in": ["Nn1"], "w": "Wsb"},
        {"id": "Nn3", "op": "conv", "in": ["Nn2"], "w": "Wsb"},
        {"id": "Nn", "op": "conv", "in": ["Nn3"], "w": "Wsb"},
        {"id": "TN", "op": "concat", "in": ["Gs", "Nn"]},
        {"id": "T", "op": "conv", "in": ["TN"], "w": "Wt", "b": "bt"},
        {"id": "V4", "op": "mul", "in": ["J", "T"]},
        sl("V0", "V4", 0, 1), sl("V1", "V4", 1, 2), sl("V2", "V4", 2, 3), sl("V3", "V4", 3, 4),
        {"id": "Va", "op": "max2", "in": ["V0", "V1"]},
        {"id": "Vb", "op": "max2", "in": ["V2", "V3"]},
        {"id": "V", "op": "max2", "in": ["Va", "Vb"]},
        {"id": "JPV", "op": "mul", "in": ["G", "V"]},
        {"id": "rc", "op": "conv", "in": ["coords"], "w": "Wr", "b": "br"},
        {"id": "rr", "op": "relu", "in": ["rc"]},
        sl("rr0", "rr", 0, 2),
        {"id": "rowpos", "op": "conv", "in": ["rr0"], "w": "Wrp"},
        sl("ra", "rr", 2, 3), sl("rb", "rr", 3, 4),
        {"id": "reg", "op": "mul", "in": ["ra", "rb"]},
        {"id": "GRp", "op": "mul", "in": ["GR", "rowpos"]},
        {"id": "IC", "op": "mul", "in": ["PU", "reg"]},
        {"id": "GN", "op": "mul", "in": ["GRp", "Npeg"]},
        {"id": "GS", "op": "mul", "in": ["GRp", "sc"]},
        {"id": "JS", "op": "mul", "in": ["JP", "sc"]},
        {"id": "cc", "op": "concat", "in": ["IC", "GRp", "GN", "GS", "JPV", "JS"]},
        {"id": "click", "op": "conv", "in": ["cc"], "w": "Wc"},
        {"id": "fg", "op": "gmax", "in": ["IC"]},
        {"id": "gg", "op": "gmax", "in": ["GRp"]},
        {"id": "jg", "op": "gmax", "in": ["JP"]},
        {"id": "yg", "op": "gmax", "in": ["Y"]},
        {"id": "lv", "op": "vconcat", "in": ["mem_next", "fg", "gg", "jg", "yg"]},
        {"id": "logits", "op": "dense", "in": ["lv"], "w": "Wl", "b": "bl"},
    ]
    return {"graph": g, "memory_size": 6, "params": P}

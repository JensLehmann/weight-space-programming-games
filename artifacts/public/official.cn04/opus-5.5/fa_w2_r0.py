import numpy as np

def build():
    P = {}
    def k1(name, mat):
        m = np.array(mat, np.float32)
        P[name] = m.reshape(m.shape[0], m.shape[1], 1, 1)
    k1("W_rm", [[63, 0, -1], [63, 0, -2]])
    k1("W_rmc", [[1, -1]])
    P["W_bg"] = (100.0 * np.eye(16)).astype(np.float32)
    P["b_bg"] = np.full(16, -40.0, np.float32)
    k1("W_sum16", [[1.0] * 16])
    ext = np.zeros((4, 16), np.float32)
    ext[0, 0] = 1; ext[1, 3] = 1; ext[2, 4] = 1; ext[3, 8] = 1
    k1("W_ext", ext)
    P["W_g"] = np.array([[40000.0]], np.float32)
    P["b_g"] = np.array([-20.0], np.float32)
    k1("W_gs", [[1, 0, -1, -1, -1], [1, -1, -1, 0, -1]])
    K = np.zeros((7, 7), np.float32)
    K[3, :] = 1; K[:, 3] = 1
    wx = np.zeros((4, 2, 7, 7), np.float32)
    wx[0, 0] = K; wx[1, 0] = K; wx[2, 1] = K; wx[3, 1] = K
    P["W_x"] = wx
    P["b_x"] = np.array([0.0, -1.0, 0.0, -1.0], np.float32)
    k1("W_adj", [[1, -1, 0, 0], [0, 0, 1, -1]])
    dirs = [(-3, 0), (3, 0), (0, -3), (0, 3)]
    rep = np.zeros((8, 2), np.float32)
    sh = np.zeros((8, 2, 7, 7), np.float32)
    ln = np.zeros((2, 8, 7, 7), np.float32)
    for b in range(2):
        for k, (dy, dx) in enumerate(dirs):
            c = b * 4 + k
            rep[c, b] = 1
            sh[c, b, 3 + dy, 3 + dx] = 1
            ln[b, c, 3 + dy, 3 + dx] = 1
    k1("W_rep", rep)
    P["W_sh"] = sh
    P["W_ln"] = ln
    k1("W_D", [[1, -1, 2, -2], [1, -1, 2, -2], [-1, 1, -2, 2], [-1, 1, -2, 2]])
    P["b_D"] = np.array([0.0, -1.0, 0.0, -1.0], np.float32)
    k1("W_cl", [[1, -1, 0, 0], [0, 0, 1, -1]])
    E = np.zeros((3, 1, 7, 7), np.float32)
    for c in range(3):
        E[c, 0, 0, 3] = 1; E[c, 0, 6, 3] = 1; E[c, 0, 3, 0] = 1; E[c, 0, 3, 6] = 1
    P["W_e"] = E
    P["b_e"] = np.array([0.0, -1.0, -2.0], np.float32)
    k1("W_e2", [[1, -2, 1]])
    st = np.zeros((10, 6), np.float32)
    st[0, 0] = 1
    st[1, 0] = -1; st[1, 2] = 1
    st[2, 1] = 1
    st[3, 1] = -1; st[3, 2] = 1
    st[4, 3] = 1
    st[5, 3] = -1; st[5, 5] = 1
    st[6, 4] = 1
    st[7, 4] = -1; st[7, 5] = 1
    st[8, 2] = 1
    st[9, 5] = 1
    k1("W_st", st)
    lin = np.array([
        [-1, 1, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 1, -1, 0, 0],
        [-1, -1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 1, 1, 0, 0]], np.float32) * 10.5
    P["W_lin"] = lin
    P["W_mis"] = np.array([[-10.0]], np.float32)
    P["b_mis"] = np.array([-1.0], np.float32)
    wo = np.zeros((8, 5), np.float32)
    bo = np.zeros(8, np.float32)
    bo[0] = -100.0
    wo[1, 0] = -1; wo[2, 0] = 1
    wo[3, 1] = -1; wo[4, 1] = 1
    wo[5, 2] = 1; bo[5] = 0.6
    wo[6, 4] = 20; wo[6, 3] = -40
    bo[7] = -100.0
    P["W_out"] = wo
    P["b_out"] = bo
    k1("W_click", [[1.0]])

    g = [
        {"id": "rmc", "op": "conv", "in": ["coords"], "w": "W_rm"},
        {"id": "rmr", "op": "relu", "in": ["rmc"]},
        {"id": "rowm", "op": "conv", "in": ["rmr"], "w": "W_rmc"},
        {"id": "cmean", "op": "gmean", "in": ["x0"]},
        {"id": "bgl", "op": "dense", "in": ["cmean"], "w": "W_bg", "b": "b_bg"},
        {"id": "bgw", "op": "vsigmoid", "in": ["bgl"]},
        {"id": "xw", "op": "mul", "in": ["x0", "bgw"]},
        {"id": "bg", "op": "conv", "in": ["xw"], "w": "W_sum16"},
        {"id": "rb", "op": "mul", "in": ["rowm", "bg"]},
        {"id": "nb", "op": "sub", "in": ["rowm", "rb"]},
        {"id": "Pc", "op": "conv", "in": ["x0"], "w": "W_ext"},
        {"id": "Q", "op": "mul", "in": ["Pc", "rowm"]},
        {"id": "q0", "op": "slice", "in": ["Q"], "start": 0, "stop": 1},
        {"id": "q4", "op": "slice", "in": ["Q"], "start": 2, "stop": 3},
        {"id": "q8", "op": "slice", "in": ["Q"], "start": 3, "stop": 4},
        {"id": "x4n", "op": "mul", "in": ["q4", "nb"]},
        {"id": "gsc", "op": "gmean", "in": ["x4n"]},
        {"id": "gl", "op": "dense", "in": ["gsc"], "w": "W_g", "b": "b_g"},
        {"id": "gv", "op": "vsigmoid", "in": ["gl"]},
        {"id": "cat1", "op": "concat", "in": ["nb", "Q"]},
        {"id": "gsp", "op": "conv", "in": ["cat1"], "w": "W_gs"},
        {"id": "gsr", "op": "relu", "in": ["gsp"]},
        {"id": "gsel", "op": "slice", "in": ["gsr"], "start": 0, "stop": 1},
        {"id": "oth", "op": "slice", "in": ["gsr"], "start": 1, "stop": 2},
        {"id": "dif", "op": "sub", "in": ["gsel", "q0"]},
        {"id": "gd", "op": "mul", "in": ["dif", "gv"]},
        {"id": "sel", "op": "add", "in": ["q0", "gd"]},
        {"id": "cat2", "op": "concat", "in": ["sel", "oth"]},
        {"id": "xc", "op": "conv", "in": ["cat2"], "w": "W_x", "b": "b_x"},
        {"id": "xr", "op": "relu", "in": ["xc"]},
        {"id": "adj", "op": "conv", "in": ["xr"], "w": "W_adj"},
        {"id": "rep", "op": "conv", "in": ["cat2"], "w": "W_rep"},
        {"id": "sh1", "op": "conv", "in": ["cat2"], "w": "W_sh"},
        {"id": "M", "op": "mul", "in": ["rep", "sh1"]},
        {"id": "lsc", "op": "conv", "in": ["M"], "w": "W_ln"},
        {"id": "cat3", "op": "concat", "in": ["adj", "lsc"]},
        {"id": "Dc", "op": "conv", "in": ["cat3"], "w": "W_D", "b": "b_D"},
        {"id": "Dr", "op": "relu", "in": ["Dc"]},
        {"id": "cl", "op": "conv", "in": ["Dr"], "w": "W_cl"},
        {"id": "cs", "op": "slice", "in": ["cl"], "start": 0, "stop": 1},
        {"id": "ct", "op": "slice", "in": ["cl"], "start": 1, "stop": 2},
        {"id": "sn", "op": "mul", "in": ["q8", "cs"]},
        {"id": "tn", "op": "mul", "in": ["q8", "ct"]},
        {"id": "ds", "op": "sub", "in": ["q8", "sn"]},
        {"id": "dsg", "op": "mul", "in": ["ds", "gv"]},
        {"id": "s", "op": "add", "in": ["sn", "dsg"]},
        {"id": "ec", "op": "conv", "in": ["x4n"], "w": "W_e", "b": "b_e"},
        {"id": "er", "op": "relu", "in": ["ec"]},
        {"id": "n1", "op": "conv", "in": ["er"], "w": "W_e2"},
        {"id": "ep", "op": "mul", "in": ["n1", "x4n"]},
        {"id": "dt", "op": "sub", "in": ["ep", "tn"]},
        {"id": "dtg", "op": "mul", "in": ["dt", "gv"]},
        {"id": "t", "op": "add", "in": ["tn", "dtg"]},
        {"id": "sc", "op": "mul", "in": ["s", "coords"]},
        {"id": "tc", "op": "mul", "in": ["t", "coords"]},
        {"id": "stc", "op": "concat", "in": ["sc", "tc"]},
        {"id": "stat", "op": "conv", "in": ["stc"], "w": "W_st"},
        {"id": "gs", "op": "gmax", "in": ["stat"]},
        {"id": "lin", "op": "dense", "in": ["gs"], "w": "W_lin"},
        {"id": "dyx", "op": "vslice", "in": ["lin"], "start": 0, "stop": 2},
        {"id": "osv", "op": "vslice", "in": ["lin"], "start": 2, "stop": 3},
        {"id": "otv", "op": "vslice", "in": ["lin"], "start": 3, "stop": 4},
        {"id": "prod", "op": "vmul", "in": ["osv", "otv"]},
        {"id": "hs", "op": "vslice", "in": ["gs"], "start": 8, "stop": 9},
        {"id": "ht", "op": "vslice", "in": ["gs"], "start": 9, "stop": 10},
        {"id": "hst", "op": "vmul", "in": ["hs", "ht"]},
        {"id": "mv", "op": "vmul", "in": ["dyx", "hst"]},
        {"id": "misl", "op": "dense", "in": ["prod"], "w": "W_mis", "b": "b_mis"},
        {"id": "mis", "op": "vrelu", "in": ["misl"]},
        {"id": "feat", "op": "vconcat", "in": ["mv", "mis", "hs", "ht"]},
        {"id": "logits", "op": "dense", "in": ["feat"], "w": "W_out", "b": "b_out"},
        {"id": "clk", "op": "sub", "in": ["oth", "sel"]},
        {"id": "click", "op": "conv", "in": ["clk"], "w": "W_click"},
    ]
    return {"graph": g, "memory_size": 0, "params": P}

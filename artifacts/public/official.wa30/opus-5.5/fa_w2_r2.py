import numpy as np

def build():
    G = []; P = {}
    def prm(n, a):
        P[n] = np.asarray(a, np.float32); return n
    def nd(id, op, ins, L=None, **kw):
        d = {"id": id, "op": op, "in": ins}; d.update(kw)
        (G if L is None else L).append(d); return id
    def const(id, vals):
        vals = np.asarray(vals, np.float32)
        prm(id + "_w", np.zeros((len(vals), 8))); prm(id + "_b", vals)
        return nd(id, "dense", ["avail"], w=id + "_w", b=id + "_b")
    K1 = const("K1", [1.0]); K25 = const("K25", [2.5]); K2 = const("K2", [2.0]); K100 = const("K100", [100.0])
    W = np.zeros((32, 3, 1, 1))
    for r in range(16):
        W[r, 0, 0, 0] = 63; W[r, 2, 0, 0] = -(4 * r + 1.5)
        W[16 + r, 1, 0, 0] = 63; W[16 + r, 2, 0, 0] = -(4 * r + 1.5)
    nd("RC", "conv", ["coords"], w=prm("W_rc", W))
    nd("RCa", "abs", ["RC"]); nd("RCs", "sub", [K25, "RCa"])
    nd("RCr", "relu", ["RCs"]); nd("RCm", "min2", ["RCr", K1])
    Wi = np.zeros((256, 32, 1, 1))
    for r in range(16):
        for c in range(16):
            Wi[r * 16 + c, r] = 1; Wi[r * 16 + c, 16 + c] = 1
    nd("INDc", "conv", ["RCm"], w=prm("W_ind", Wi), b=prm("b_ind", -np.ones(256)))
    nd("IND", "relu", ["INDc"])
    nd("rok0", "conv", ["coords"], w=prm("W_rok", np.array([-63, 0, 63.]).reshape(1, 3, 1, 1)))
    nd("rowok", "min2", ["rok0", K1])
    groups = [[3, 4], [3, 4, 5, 11, 12, 15], [14], [2, 9], [9], [0, 3, 4, 5], [2], [1], [0], [11], [3], [15, 11], [12], [5, 11, 12, 15]]
    Wm = np.zeros((len(groups), 16, 1, 1))
    for i, g in enumerate(groups):
        for c in g: Wm[i, c] = 1
    nd("mk0", "conv", ["x0"], w=prm("W_mk", Wm)); nd("mk", "mul", ["mk0", "rowok"])
    sl = [nd("sl%d" % i, "slice", ["mk"], start=i, stop=i + 1) for i in range(len(groups))]
    cross = np.zeros((1, 1, 3, 3)); cross[0, 0, 0, 1] = cross[0, 0, 2, 1] = cross[0, 0, 1, 0] = cross[0, 0, 1, 2] = 1
    nd("n2", "conv", [sl[6]], w=prm("W_cross", cross)); nd("n2p", "mul", ["n2", sl[7]])
    nd("hc", "conv", ["n2p"], w=prm("W_h2", np.ones((1, 1, 1, 1))), b=prm("b_h2", [-2.]))
    nd("hole0", "relu", ["hc"]); nd("hole", "min2", ["hole0", K1]); nd("mOp", "add", [sl[1], "hole"])
    prm("W_sum", np.ones((1, 256, 1, 1)))
    def CP(n, m):
        nd(n + "_a", "mul", ["IND", m]); nd(n + "_g", "gmax", [n + "_a"])
        nd(n + "_r", "mul", ["IND", n + "_g"]); nd(n + "_s", "conv", [n + "_r"], w="W_sum")
        return nd(n, "min2", [n + "_s", K1])
    Bx = CP("Bx", sl[0]); Ob = CP("Ob", sl[1]); Op = CP("Op", "mOp"); Pc = CP("Pc", sl[2])
    prm("W_ones3", np.ones((1, 1, 3, 3)))
    nd("bb", "conv", [sl[5]], w="W_ones3"); nd("seed0", "sub", [sl[4], "bb"]); nd("seed", "relu", ["seed0"])
    body = []
    nd("tc", "conv", ["t"], body, w="W_ones3"); nd("ts", "add", ["tc", "seed"], body)
    nd("tm", "min2", ["ts", K1], body); nd("tn", "mul", ["tm", sl[3]], body)
    G.append({"id": "TF", "op": "loop", "T": 16, "state": [{"id": "t", "init": "seed"}], "body": body, "next": {"t": "tn"}})
    nd("hole2", "mul", ["hole", K2]); nd("code", "add", ["TF.t", "hole2"])
    nd("tv_a", "mul", ["IND", "code"]); nd("tv", "gmax", ["tv_a"])
    nd("oms", "dense", ["aprev"], w=prm("W_oms", -np.ones((1, 8))), b=prm("b_oms", [1.]))
    nd("vv", "vmul", ["tv", "oms"]); nd("mem_next", "vadd", ["mem", "vv"])
    nd("Mm_a", "mul", ["IND", "mem_next"]); nd("Mm", "conv", ["Mm_a"], w="W_sum")
    nd("Hs0", "conv", ["Mm"], w=prm("W_hs", [[[[4.]]]]), b=prm("b_hs", [-6.]))
    nd("Hs1", "relu", ["Hs0"]); nd("Hs", "min2", ["Hs1", K1])
    nd("Hs2", "mul", ["Hs", K2]); nd("Tc", "sub", ["Mm", "Hs2"])
    nd("NT", "sub", [K1, "Tc"]); nd("FA", "sub", [K1, Op]); nd("Fb", "sub", [K1, Ob])
    nd("NHs", "sub", [K1, "Hs"])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    Wp4 = np.zeros((4, 1, 3, 3)); Wq1 = np.zeros((4, 1, 5, 5)); Wq2 = np.zeros((4, 4, 5, 5))
    for k, (dy, dx) in enumerate(dirs):
        Wp4[k, 0, 1 + dy, 1 + dx] = 0.98
        Wq1[k, 0, 2 + 2 * dy, 2 + 2 * dx] = 1; Wq2[k, k, 2 + 2 * dy, 2 + 2 * dx] = 1
    prm("W_p4", Wp4)
    body = []
    nd("rc", "conv", ["r"], body, w="W_p4")
    for k in range(4): nd("rs%d" % k, "slice", ["rc"], body, start=k, stop=k + 1)
    nd("rm1", "max2", ["rs0", "rs1"], body); nd("rm2", "max2", ["rs2", "rs3"], body)
    nd("rm3", "max2", ["rm1", "rm2"], body); nd("rf", "mul", ["rm3", "FA"], body)
    nd("rn", "max2", ["rf", sl[12]], body)
    G.append({"id": "R", "op": "loop", "T": 128, "state": [{"id": "r", "init": sl[12]}], "body": body, "next": {"r": "rn"}})
    nd("Rb0", "mul", ["R.r", K100]); nd("Rb", "min2", ["Rb0", K1])
    prm("W_s4", np.ones((1, 4, 1, 1)))
    nd("q1", "conv", ["Rb"], w=prm("W_q1", Wq1)); nd("q2", "conv", ["q1"], w=prm("W_q2", Wq2))
    nd("q3", "conv", ["q2"], w="W_s4"); nd("HS", "min2", ["q3", K1])
    nd("HSH", "mul", ["HS", "Hs"]); nd("TH0", "add", ["Tc", "HSH"]); nd("TH", "min2", ["TH0", K1])
    nd("BxT", "mul", [Bx, "Tc"])
    nd("ob2a", "add", [sl[13], "BxT"]); nd("ob2", "min2", ["ob2a", K1]); nd("Fb2", "sub", [K1, "ob2"])
    nd("op2a", "add", ["ob2", "Hs"]); nd("op2", "min2", ["op2a", K1]); nd("FA2", "sub", [K1, "op2"])
    nd("SA0", "mul", [Bx, "NT"]); nd("BN", "mul", ["SA0", "NHs"])
    Wa = np.zeros((20, 5, 5, 5)); Wb = np.zeros((20, 20, 5, 5)); Wf = np.zeros((4, 1, 3, 3))
    Wp = np.zeros((8, 2, 3, 3)); Wd = np.zeros((16, 4, 3, 3))
    for k, (dy, dx) in enumerate(dirs):
        for j in range(5):
            s = -1 if j == 2 else 1
            Wa[4 * j + k, j, 2 + 2 * s * dy, 2 + 2 * s * dx] = 1
            Wb[4 * j + k, 4 * j + k, 2 + 2 * s * dy, 2 + 2 * s * dx] = 1
        Wf[k, 0, 1 - dy, 1 - dx] = 1
        for c in range(2): Wp[k * 2 + c, c, 1 + dy, 1 + dx] = 0.98
        for c in range(4): Wd[4 * k + c, c, 1 + dy, 1 + dx] = 0.98
    nd("X3", "concat", ["Fb", "TH", sl[2], "Fb2", "BN"])
    nd("sh1", "conv", ["X3"], w=prm("W_sa", Wa)); nd("sh2", "conv", ["sh1"], w=prm("W_sb", Wb))
    nd("Fbs", "slice", ["sh2"], start=0, stop=4); nd("Ts", "slice", ["sh2"], start=4, stop=8)
    nd("Ss", "slice", ["sh2"], start=8, stop=12); nd("Fb2s", "slice", ["sh2"], start=12, stop=16)
    nd("BNs", "slice", ["sh2"], start=16, stop=20)
    nd("V4", "mul", ["Fbs", "FA"]); nd("G4", "mul", ["V4", "Ts"])
    nd("V4n", "mul", ["Fb2s", "FA2"]); nd("G4n", "mul", ["V4n", "Ts"])
    prm("W_d", Wd)
    body = []
    nd("dc", "conv", ["d"], body, w="W_d")
    for k in range(4): nd("ds%d" % k, "slice", ["dc"], body, start=4 * k, stop=4 * k + 4)
    nd("dm1", "max2", ["ds0", "ds1"], body); nd("dm2", "max2", ["ds2", "ds3"], body)
    nd("dm3", "max2", ["dm1", "dm2"], body); nd("df", "mul", ["dm3", "V4n"], body)
    nd("dn", "max2", ["df", "G4n"], body)
    G.append({"id": "D", "op": "loop", "T": 256, "state": [{"id": "d", "init": "G4n"}], "body": body, "next": {"d": "dn"}})
    nd("S", "mul", ["BNs", "D.d"])
    for k in range(4): nd("Sk%d" % k, "slice", ["S"], start=k, stop=k + 1)
    nd("Sm1", "max2", ["Sk0", "Sk1"]); nd("Sm2", "max2", ["Sk2", "Sk3"]); nd("Sm", "max2", ["Sm1", "Sm2"])
    nd("SA2", "add", ["Sm", sl[11]]); nd("SA", "min2", ["SA2", K1])
    nd("fc", "conv", [sl[2]], w=prm("W_f", Wf)); nd("fm", "mul", ["fc", sl[8]]); nd("f", "gmax", ["fm"])
    nd("V4f", "mul", ["V4", "f"]); nd("VB", "conv", ["V4f"], w="W_s4")
    nd("G4f", "mul", ["G4", "f"]); nd("GB", "conv", ["G4f"], w="W_s4")
    nd("srcs", "concat", ["SA", "GB"]); nd("frees", "concat", ["FA", "VB"])
    prm("W_p", Wp)
    body = []
    nd("pc", "conv", ["v"], body, w="W_p")
    for k in range(4): nd("ps%d" % k, "slice", ["pc"], body, start=2 * k, stop=2 * k + 2)
    nd("pm1", "max2", ["ps0", "ps1"], body); nd("pm2", "max2", ["ps2", "ps3"], body)
    nd("pm3", "max2", ["pm1", "pm2"], body); nd("pf", "mul", ["pm3", "frees"], body)
    nd("vn", "max2", ["pf", "srcs"], body)
    G.append({"id": "H", "op": "loop", "T": 256, "state": [{"id": "v", "init": "srcs"}], "body": body, "next": {"v": "vn"}})
    nd("FAf", "slice", ["H.v"], start=0, stop=1); nd("FBf", "slice", ["H.v"], start=1, stop=2)
    nd("mAs", "mul", ["Ss", "FAf"]); nd("mA", "gmax", ["mAs"])
    nd("mBs", "mul", ["Ss", "FBf"]); nd("mB", "gmax", ["mBs"])
    nd("g0", "gmean", [sl[8]])
    nd("hl", "dense", ["g0"], w=prm("W_hl", [[5 * 4096.]]), b=prm("b_hl", [-50.]))
    nd("h", "vsigmoid", ["hl"])
    nd("omh", "dense", ["h"], w=prm("W_omh", [[-1.]]), b=prm("b_omh", [1.]))
    nd("mvA", "vmul", ["mA", "omh"]); nd("mvB", "vmul", ["mB", "h"]); nd("mv", "vadd", ["mvA", "mvB"])
    nd("bm", "mul", ["S", sl[2]]); nd("bg", "gmax", ["bm"])
    nd("bl", "dense", ["bg"], w=prm("W_bl", 5000 * np.eye(4)), b=prm("b_bl", -10 * np.ones(4)))
    nd("bs", "vsigmoid", ["bl"]); nd("bon", "vmul", ["bs", "omh"])
    nd("fb", "vmul", ["bg", "f"])
    nd("gl", "dense", ["fb"], w=prm("W_gl", 5000 * np.ones((1, 4))), b=prm("b_gl", [-10.]))
    nd("g3", "vsigmoid", ["gl"])
    nd("a11", "gmax", [sl[9]])
    nd("NPc", "sub", [K1, Pc]); nd("ho1", "mul", [sl[8], "NPc"]); nd("ho2", "mul", ["ho1", "TH"])
    nd("hot", "gmax", ["ho2"])
    nd("nr0", "dense", ["mB"], w=prm("W_nr", -1000. * np.ones((1, 4))), b=prm("b_nr", [1.]))
    nd("nr", "vrelu", ["nr0"]); nd("nrh", "vmul", ["nr", "h"])
    nd("trA", "vadd", ["a11", "g3"]); nd("trA2", "vmul", ["trA", "omh"]); nd("trB", "vmul", ["hot", "h"])
    nd("tr1", "vadd", ["trA2", "trB"]); nd("tr", "vadd", ["tr1", "nrh"])
    nd("z", "vconcat", ["mv", "tr", "bon"])
    Wo = np.zeros((8, 9)); bo = np.full(8, -10.)
    for k in range(4):
        Wo[1 + k, k] = 1; Wo[1 + k, 5 + k] = 1.2; bo[1 + k] = 0
    Wo[5, 4] = 3; bo[5] = -0.5
    nd("logits", "dense", ["z"], w=prm("W_o", Wo), b=prm("b_o", bo))
    nd("click", "conv", ["rowok"], w=prm("W_c", np.zeros((1, 1, 1, 1))))
    return {"graph": G, "memory_size": 256, "params": P}

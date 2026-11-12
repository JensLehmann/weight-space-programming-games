import numpy as np

def build():
    P = {}
    g = []
    def nd(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); g.append(d)

    M = 16
    nd("row", "slice", ["coords"], start=0, stop=1)
    P["RMw"] = np.full((1, 1, 1, 1), 1000.0)
    nd("rmc", "conv", ["row"], w="RMw")
    nd("rmask", "tanh", ["rmc"])
    nd("Y0", "slice", ["x0"], start=4, stop=5)
    nd("Y", "mul", ["Y0", "rmask"])
    nd("W5", "slice", ["x0"], start=5, stop=6)
    nd("B9", "slice", ["x0"], start=9, stop=10)
    nd("C12", "slice", ["x0"], start=12, stop=13)
    nd("cr", "slice", ["coords"], start=0, stop=2)
    Tw = np.zeros((1, 16, 1, 1)); Tw[0, 11] = 1; Tw[0, 14] = 1
    P["Tw"] = Tw
    nd("T", "conv", ["x0"], w="Tw")
    P["N4w"] = np.full((1, 1, 5, 5), 10.0)
    nd("n4c", "conv", ["Y"], w="N4w")
    nd("n4", "tanh", ["n4c"])
    nd("B", "mul", ["T", "n4"])
    nd("Mk", "sub", ["T", "B"])

    nd("nB", "gmean", ["B"]); nd("nM", "gmean", ["Mk"]); nd("nY", "gmean", ["Y"])
    nd("Bc", "mul", ["B", "cr"]); nd("Bx", "gmean", ["Bc"])
    nd("Mc", "mul", ["Mk", "cr"]); nd("Mx", "gmean", ["Mc"])
    nd("Yc", "mul", ["Y", "cr"]); nd("Yx", "gmean", ["Yc"])

    K = 1e13; eps = 1e-10
    Wd0 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], dtype=np.float64)
    P["Wp"] = K * Wd0; P["Wn"] = -K * Wd0
    P["be"] = np.full(2, -K * eps); P["Wc2"] = Wd0.copy()

    def dsign(tag, u):
        nd("dp" + tag, "dense", [u], w="Wp", b="be")
        nd("rp" + tag, "vrelu", ["dp" + tag])
        nd("tp" + tag, "vtanh", ["rp" + tag])
        nd("dn" + tag, "dense", [u], w="Wn", b="be")
        nd("rn" + tag, "vrelu", ["dn" + tag])
        nd("tn" + tag, "vtanh", ["rn" + tag])
        nd("cc" + tag, "vconcat", ["tp" + tag, "tn" + tag])
        nd("s" + tag, "dense", ["cc" + tag], w="Wc2")
        return "s" + tag

    nd("u1a", "vmul", ["Mx", "nB"]); nd("u1b", "vmul", ["Bx", "nM"])
    nd("u1", "vconcat", ["u1a", "u1b"])
    sMB = dsign("MB", "u1")
    nd("u2a", "vmul", ["Yx", "nB"]); nd("u2b", "vmul", ["Bx", "nY"])
    nd("u2", "vconcat", ["u2a", "u2b"])
    sYB = dsign("YB", "u2")
    nd("gp", "vmul", [sMB, sYB])
    P["Gw"] = np.full((1, 2), 5.0)
    nd("gs", "dense", ["gp"], w="Gw"); nd("sg", "vtanh", ["gs"])
    nd("ax", "vmul", [sYB, sYB])
    P["Pm"] = np.array([[0.0, 1.0], [1.0, 0.0]])
    nd("aw", "dense", ["ax"], w="Pm")

    nd("Ac", "mul", ["cr", "nY"])
    nd("A", "sub", ["Ac", "Yx"])
    Fw = np.zeros((2, 2, 1, 1)); Fw[0, 0] = -1e5; Fw[1, 1] = -1e5
    P["Fw"] = Fw
    nd("fdc", "conv", ["A"], w="Fw"); nd("fd", "tanh", ["fdc"])
    nd("absA", "abs", ["A"])
    P["Dw"] = np.full((1, 2, 1, 1), 100.0)
    nd("dd", "conv", ["absA"], w="Dw")

    # 16-state clock
    qv = np.zeros((1, M)); gv = np.zeros((1, M)); tv = np.zeros((1, M)); ev = np.zeros((1, M))
    targets = [0.05, 0.6, 0.15, 0.45, 0.3, 0.75]
    for i in range(M):
        if i < 10:
            qv[0, i] = (-1.0 if i % 2 == 0 else 1.0) + 1.0
        else:
            qv[0, i] = 1.0
            gv[0, i] = -1.0
            tv[0, i] = targets[i - 10]
            ev[0, i] = -20.0
    P["Qw"] = qv; P["Qb"] = np.array([-1.0])
    nd("q", "dense", ["mem"], w="Qw", b="Qb")
    nd("dterm", "mul", ["dd", "q"])
    P["Gdw"] = gv; P["Gdb"] = np.array([1.0])
    nd("gd", "dense", ["mem"], w="Gdw", b="Gdb")
    P["Tv"] = tv
    nd("tt", "dense", ["mem"], w="Tv")
    nd("ddt", "sub", ["dd", "tt"])
    nd("addt", "abs", ["ddt"])
    P["Ev"] = ev
    nd("evv", "dense", ["mem"], w="Ev")
    nd("Et", "mul", ["addt", "evv"])

    nd("Amc", "mul", ["cr", "nM"])
    nd("Am", "sub", ["Amc", "Mx"])
    nd("mdc", "conv", ["Am"], w="Fw"); nd("md", "tanh", ["mdc"])
    nd("fm", "mul", ["fd", "md"])
    nd("fmw", "mul", ["fm", "aw"])
    P["GSw"] = np.full((1, 2, 1, 1), -80.0); P["GSb"] = np.array([20.0])
    nd("gsc", "conv", ["fmw"], w="GSw", b="GSb")
    nd("Gt", "mul", ["gsc", "C12"])

    WL = np.zeros((4, 1, 7, 7))
    for k in range(1, 4):
        WL[0, 0, 3 + k, 3] = 5.0
        WL[1, 0, 3 - k, 3] = 5.0
        WL[2, 0, 3, 3 + k] = 5.0
        WL[3, 0, 3, 3 - k] = 5.0
    P["WLw"] = WL
    nd("wlc", "conv", ["W5"], w="WLw"); nd("wl", "tanh", ["wlc"])
    WD = np.zeros((2, 4, 1, 1))
    WD[0, 0] = 1; WD[0, 1] = -1; WD[1, 2] = 1; WD[1, 3] = -1
    P["WDw"] = WD
    nd("wd", "conv", ["wl"], w="WDw")
    nd("pr", "mul", ["wd", "fd"])
    nd("prw", "mul", ["pr", "aw"])
    P["Sw"] = np.full((1, 2, 1, 1), -10.0)
    nd("S10", "conv", ["prw"], w="Sw")
    nd("Ds0", "mul", ["S10", "sg"])
    nd("Ds", "mul", ["Ds0", "gd"])
    nd("pre0", "add", ["Ds", "dterm"])
    nd("pre", "add", ["pre0", "Et"])
    P["Pw"] = np.ones((1, 1, 1, 1)); P["Pb"] = np.array([50.0])
    nd("pre2", "conv", ["pre"], w="Pw", b="Pb")
    nd("cb", "mul", ["pre2", "B9"])
    nd("click", "add", ["cb", "Gt"])

    Perm = np.zeros((M, M))
    for i in range(M):
        Perm[(i + 1) % M, i] = 1.0
    e1 = np.zeros(M); e1[1] = 1.0
    P["Mw"] = Perm - np.outer(e1, np.ones(M)); P["Mb"] = e1
    nd("mem_next", "dense", ["mem"], w="Mw", b="Mb")
    P["Lw"] = np.zeros((8, M)); Lb = np.zeros(8); Lb[6] = 1.0
    P["Lb"] = Lb
    nd("logits", "dense", ["mem"], w="Lw", b="Lb")

    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": M, "params": params}

import numpy as np

def build():
    P = {}
    g = []
    def n(id_, op, ins, **kw):
        d = {"id": id_, "op": op, "in": ins}; d.update(kw); g.append(d)

    n("x", "concat", ["x0", "coords"])
    WZ = np.zeros((1, 19, 1, 1)); WZ[0, 0] = 1.0; WZ[0, 16] = 126.0
    P["WZ"] = WZ; P["bZ"] = np.array([-1.0])
    n("zc", "conv", ["x"], w="WZ", b="bZ"); n("zr", "relu", ["zc"])
    n("x0s", "slice", ["x0"], start=0, stop=1)
    n("Z", "min2", ["zr", "x0s"])
    W4 = np.zeros((1, 19, 1, 1)); W4[0, 4] = 1.0; W4[0, 16] = 126.0
    P["W4"] = W4
    n("q4c", "conv", ["x"], w="W4", b="bZ"); n("q4r", "relu", ["q4c"])
    n("x4", "slice", ["x0"], start=4, stop=5)
    n("x4m", "min2", ["q4r", "x4"])
    n("x8", "slice", ["x0"], start=8, stop=9)
    n("x3", "slice", ["x0"], start=3, stop=4)
    WM = np.zeros((1, 3, 1, 1)); WM[0, 0] = 63.0
    P["WM"] = WM
    n("rm1", "conv", ["coords"], w="WM", b="bZ")
    n("rma", "abs", ["rm1"])
    P["WNeg"] = -np.ones((1, 1, 1, 1)); P["bOne"] = np.array([1.0])
    n("rmb", "conv", ["rma"], w="WNeg", b="bOne")
    n("rowm", "relu", ["rmb"])
    n("x0m", "mul", ["x0", "rowm"])
    n("bvec", "gmax", ["x0m"])
    n("Bx", "mul", ["x0", "bvec"])
    P["WS16"] = np.ones((1, 16, 1, 1))
    n("B", "conv", ["Bx"], w="WS16")
    n("xb", "concat", ["x0", "B"])
    WO = np.zeros((1, 17, 1, 1))
    for c in range(16):
        if c not in (0, 3, 4, 8):
            WO[0, c] = 1.0
    WO[0, 16] = -1.0
    P["WO"] = WO
    n("Oraw", "conv", ["xb"], w="WO")
    n("O", "relu", ["Oraw"])
    n("zox", "concat", ["Z", "O", "x8"])
    WN = np.zeros((1, 3, 7, 7))
    nb = [(0, 3), (6, 3), (3, 0), (3, 6)]
    for (a, b) in nb:
        WN[0, 0, a, b] = 1.0; WN[0, 1, a, b] = -4.0
    WN[0, 2, 3, 3] = 1.0
    P["WN"] = WN
    n("sc", "conv", ["zox"], w="WN", b="bZ")
    n("sr", "relu", ["sc"])
    n("Sng", "min2", ["sr", "x8"])
    n("Tng", "sub", ["x8", "Sng"])
    WE = np.zeros((1, 1, 7, 7))
    for (a, b) in nb:
        WE[0, 0, a, b] = 1.0
    P["WE"] = WE
    n("n4", "conv", ["x4m"], w="WE", b="bZ")
    n("n4a", "abs", ["n4"])
    n("n4b", "conv", ["n4a"], w="WNeg", b="bOne")
    n("n4r", "relu", ["n4b"])
    n("Eg", "min2", ["n4r", "x4m"])
    n("gz", "gmean", ["Z"])
    P["WG0"] = np.array([[-4096.0]]); P["bG0"] = np.array([18.0])
    n("gzl", "dense", ["gz"], w="WG0", b="bG0")
    n("G", "vsigmoid", ["gzl"])
    n("g4", "gmean", ["x4m"]); n("gO", "gmean", ["O"])
    n("gv", "vconcat", ["gz", "g4", "gO"])
    P["WNS"] = np.array([[-4096.0, -4096.0, 0.0], [-4096.0, 0.0, -4096.0]])
    P["bNS"] = np.array([4.5, 4.5])
    n("nsl", "dense", ["gv"], w="WNS", b="bNS")
    n("nss", "vsigmoid", ["nsl"])

    n("dS", "sub", ["x8", "Sng"]); n("GdS", "mul", ["dS", "G"])
    n("S0", "add", ["Sng", "GdS"]); n("S", "add", ["S0", "x3"])
    n("dT", "sub", ["Eg", "Tng"]); n("GdT", "mul", ["dT", "G"])
    n("T0", "add", ["Tng", "GdT"]); n("T", "add", ["T0", "x3"])

    WR = np.zeros((2, 3, 1, 1)); WR[0, 0] = 63.0; WR[1, 1] = 63.0
    P["WR"] = WR
    n("R", "conv", ["coords"], w="WR")
    n("RR", "mul", ["R", "R"])
    n("SR", "mul", ["S", "R"]); n("SRR", "mul", ["S", "RR"])
    n("TR", "mul", ["T", "R"]); n("TRR", "mul", ["T", "RR"])
    n("mom", "concat", ["S", "SR", "SRR", "T", "TR", "TRR"])
    n("gm", "gmean", ["mom"])
    pairs = [(6, 0), (1, 5), (7, 0), (2, 5), (0, 5), (0, 3), (1, 1), (0, 4), (2, 2),
             (5, 8), (6, 6), (5, 9), (7, 7)]
    WU1 = np.zeros((13, 10)); WV1 = np.zeros((13, 10))
    for k, (u, v) in enumerate(pairs):
        WU1[k, u] = 4096.0; WV1[k, v] = 4096.0
    P["WU1"] = WU1; P["WV1"] = WV1
    n("U1", "dense", ["gm"], w="WU1"); n("V1", "dense", ["gm"], w="WV1")
    n("pr", "vmul", ["U1", "V1"])
    WQ = np.zeros((5, 13))
    WQ[0, 0] = 1; WQ[0, 1] = -1
    WQ[1, 2] = 1; WQ[1, 3] = -1
    WQ[2, 4] = 1
    WQ[3, 5] = 1; WQ[3, 6] = -1; WQ[3, 7] = -1; WQ[3, 8] = 1
    WQ[4, 9] = 1; WQ[4, 10] = -1; WQ[4, 11] = -1; WQ[4, 12] = 1
    P["WQ"] = WQ
    n("q", "dense", ["pr"], w="WQ")
    n("qa", "vslice", ["q"], start=3, stop=4)
    n("qb", "vslice", ["q"], start=4, stop=5)
    n("oo", "vmul", ["qa", "qb"])
    n("h", "vconcat", ["q", "oo"])
    WA = np.zeros((4, 6)); WA[0, 0] = 1; WA[1, 0] = -1; WA[2, 1] = 1; WA[3, 1] = -1
    P["WA"] = WA
    n("a1p", "dense", ["h"], w="WA")
    n("a1", "vrelu", ["a1p"])
    n("ha", "vconcat", ["h", "a1", "G"])              # 11
    beta = 0.1; th = 1.6
    WP = np.zeros((7, 11)); bP = np.zeros(7)
    WP[0, 5] = -1e-3; bP[0] = -10.0; WP[0, 10] = -30.0
    WP[1, 6] = beta; WP[1, 7] = beta; WP[1, 2] = -th * beta
    WP[2, 8] = beta; WP[2, 9] = beta; WP[2, 2] = -th * beta
    WP[3, 0] = -beta; WP[4, 0] = beta
    WP[5, 1] = -beta; WP[6, 1] = beta
    P["WP"] = WP; P["bP"] = bP
    n("p2", "dense", ["ha"], w="WP", b="bP")
    n("s", "vsigmoid", ["p2"])
    WGa = np.zeros((4, 7)); WGa[0, 1] = 1; WGa[1, 1] = 1; WGa[2, 2] = 1; WGa[3, 2] = 1
    WD = np.zeros((4, 7)); WD[0, 3] = 1; WD[1, 4] = 1; WD[2, 5] = 1; WD[3, 6] = 1
    P["WGa"] = WGa; P["WD"] = WD
    n("Ug", "dense", ["s"], w="WGa"); n("Vg", "dense", ["s"], w="WD")
    n("mv", "vmul", ["Ug", "Vg"])
    n("sm", "vconcat", ["s", "mv", "nss"])
    WL = np.zeros((8, 13)); bL = np.array([-100.0, 0, 0, 0, 0, 2.0, -100.0, -100.0])
    WL[1, 7] = 5.0; WL[2, 8] = 5.0; WL[3, 9] = 4.0; WL[4, 10] = 4.0; WL[5, 0] = 10.0
    WL[6, 11] = 120.0; WL[6, 12] = 120.0
    P["WL"] = WL; P["bL"] = bL
    n("logits", "dense", ["sm"], w="WL", b="bL")
    n("cin", "concat", ["O", "x4m", "x8"])
    WC = np.zeros((1, 3, 1, 1)); WC[0, 0] = 1.0; WC[0, 1] = 1.0; WC[0, 2] = 2.0
    P["WC"] = WC
    n("click", "conv", ["cin"], w="WC")

    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

import numpy as np

def build():
    P = {}
    g = []

    def n(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}
        d.update(kw)
        g.append(d)

    n("one", "slice", ["coords"], start=2, stop=3)
    W = np.zeros((2, 3, 1, 1)); W[0, 0] = 63; W[1, 0] = 63; W[1, 2] = -1
    P["Wrm"] = W
    n("rmp", "conv", ["coords"], w="Wrm")
    n("rmr", "relu", ["rmp"])
    P["Wrd"] = np.array([1.0, -1.0]).reshape(1, 2, 1, 1)
    n("rowmask", "conv", ["rmr"], w="Wrd")
    n("f4", "slice", ["x0"], start=4, stop=5)
    n("f4m", "mul", ["f4", "rowmask"])

    W = np.zeros((4, 1, 3, 3))
    W[0, 0, 1, 0] = 1; W[1, 0, 1, 2] = 1; W[2, 0, 0, 1] = 1; W[3, 0, 2, 1] = 1
    P["Wnb"] = W
    n("nb4", "conv", ["f4m"], w="Wnb")
    W = np.zeros((1, 16, 1, 1)); W[0, 11] = 1; W[0, 14] = 1
    P["Wq"] = W
    n("q", "conv", ["x0"], w="Wq")
    n("qn", "mul", ["nb4", "q"])
    n("flags", "gmax", ["qn"])

    P["Wd5"] = np.ones((1, 1, 5, 5))
    n("d4", "conv", ["f4m"], w="Wd5")
    n("near4", "min2", ["d4", "one"])
    W = np.zeros((2, 16, 1, 1)); W[0, 11] = 1; W[1, 14] = 1
    P["Wc2"] = W
    n("c2", "conv", ["x0"], w="Wc2")
    n("P", "mul", ["c2", "near4"])
    n("G", "sub", ["c2", "P"])
    W = np.zeros((4, 3, 1, 1))
    W[0, :, 0, 0] = [0, 1, 0]; W[1, :, 0, 0] = [0, -1, 1]
    W[2, :, 0, 0] = [1, 0, 0]; W[3, :, 0, 0] = [-1, 0, 1]
    P["Wpj"] = W
    n("proj", "conv", ["coords"], w="Wpj")

    K = 2000.0
    P["Ws"] = np.array([[1, 1, 1, 1, -1, -1, -1, -1]], dtype=np.float64)
    P["Wkr"] = np.array([[K]]); P["Wkf"] = np.array([[-K]])
    P["bk"] = np.array([-K * 0.5 / 63.0])
    for ci, c in ((0, "11"), (1, "14")):
        n("P" + c, "slice", ["P"], start=ci, stop=ci + 1)
        n("G" + c, "slice", ["G"], start=ci, stop=ci + 1)
        n("pp" + c, "mul", ["P" + c, "proj"])
        n("pg" + c, "mul", ["G" + c, "proj"])
        n("gp" + c, "gmax", ["pp" + c])
        n("gg" + c, "gmax", ["pg" + c])
        n("fp" + c, "vmul", ["flags", "gp" + c])
        n("fg" + c, "vmul", ["flags", "gg" + c])
        n("sv" + c, "vconcat", ["fp" + c, "fg" + c])
        n("s" + c, "dense", ["sv" + c], w="Ws")
        n("r" + c, "dense", ["s" + c], w="Wkr", b="bk")
        n("rs" + c, "vsigmoid", ["r" + c])
        n("f" + c, "dense", ["s" + c], w="Wkf", b="bk")
        n("fs" + c, "vsigmoid", ["f" + c])
    n("presP", "gmax", ["P"])
    n("presG", "gmax", ["G"])
    n("gate", "vmul", ["presP", "presG"])
    n("rise", "vconcat", ["rs11", "rs14"])
    n("fall", "vconcat", ["fs11", "fs14"])
    n("riseg", "vmul", ["rise", "gate"])
    n("fallg", "vmul", ["fall", "gate"])
    P["Wsc"] = np.diag([2.0, 1.0])
    n("riseW", "dense", ["riseg"], w="Wsc")
    n("fallW", "dense", ["fallg"], w="Wsc")

    W = np.zeros((2, 2, 5, 5)); W[0, 0] = 1; W[1, 1] = 1
    P["Wd5b"] = W
    n("seedd", "conv", ["P"], w="Wd5b")
    n("seedm", "min2", ["seedd", "one"])
    n("blk", "slice", ["x0"], start=0, stop=1)
    n("seed", "mul", ["seedm", "blk"])
    W = np.zeros((1, 16, 1, 1)); W[0, 0] = 1; W[0, 9] = 1
    P["Wpas"] = W
    n("pas", "conv", ["x0"], w="Wpas")
    W = np.zeros((2, 2, 3, 3)); W[0, 0] = 1; W[1, 1] = 1
    P["Wfl"] = W
    g.append({"id": "FL", "op": "loop", "T": 64,
              "state": [{"id": "r", "init": "seed"}],
              "body": [{"id": "rc", "op": "conv", "in": ["r"], "w": "Wfl"},
                       {"id": "rmn", "op": "min2", "in": ["rc", "one"]},
                       {"id": "rn", "op": "mul", "in": ["rmn", "pas"]}],
              "next": {"r": "rn"}})

    n("wall", "slice", ["x0"], start=5, stop=6)
    n("B", "slice", ["x0"], start=9, stop=10)
    n("KR", "mul", ["blk", "FL.r"])
    n("KN", "sub", ["blk", "KR"])
    n("Kst", "concat", ["KR", "KN"])
    prim = {0: (0, 1), 1: (2, 1), 2: (1, 0), 3: (1, 2)}
    lat = {0: [(1, 0), (1, 2)], 1: [(1, 0), (1, 2)], 2: [(0, 1), (2, 1)], 3: [(0, 1), (2, 1)]}
    Wsd = np.zeros((16, 4, 3, 3)); Wpr = np.zeros((16, 16, 3, 3)); Wbp = np.zeros((16, 16, 3, 3))
    Wgv = np.zeros((16, 4)); Wa = np.zeros((4, 16, 1, 1))
    for v in range(4):
        for d in range(4):
            ch = v * 4 + d
            a, b = prim[d]
            Wsd[ch, v, a, b] = 1; Wpr[ch, ch, a, b] = 1; Wbp[ch, ch, a, b] = 1
            for (a2, b2) in lat[d]:
                Wpr[ch, ch, a2, b2] = 1
            if d < 2:
                Wgv[ch, 0] = 1; Wgv[ch, 1] = 1
            else:
                Wgv[ch, 2] = 1; Wgv[ch, 3] = 1
            Wa[v, ch] = 1
    P["Wsd"] = Wsd; P["Wpr"] = Wpr; P["Wbp"] = Wbp; P["Wgv"] = Wgv; P["Wa"] = Wa
    n("seedc", "conv", ["Kst"], w="Wsd")
    n("seedw", "mul", ["seedc", "wall"])
    g.append({"id": "DL", "op": "loop", "T": 12,
              "state": [{"id": "d", "init": "seedw"}],
              "body": [{"id": "dc", "op": "conv", "in": ["d"], "w": "Wpr"},
                       {"id": "ds", "op": "add", "in": ["dc", "seedc"]},
                       {"id": "dm", "op": "min2", "in": ["ds", "one"]},
                       {"id": "dn", "op": "mul", "in": ["dm", "wall"]}],
              "next": {"d": "dn"}})
    n("bp", "conv", ["DL.d"], w="Wbp")
    n("bpm", "min2", ["bp", "one"])
    n("Bs", "mul", ["bpm", "B"])
    n("gv", "dense", ["flags"], w="Wgv")
    n("BsG", "mul", ["Bs", "gv"])
    n("A", "conv", ["BsG"], w="Wa")
    n("Am", "min2", ["A", "one"])
    n("aR", "slice", ["Am"], start=0, stop=2)
    n("aN", "slice", ["Am"], start=2, stop=4)
    n("X1", "mul", ["aN", "FL.r"])
    n("X2", "sub", ["aN", "X1"])
    n("aRR", "mul", ["aR", "FL.r"])
    n("X3", "sub", ["aR", "aRR"])
    n("cr", "concat", ["X1", "X2"])
    n("cf", "concat", ["X3", "X2"])
    W = np.zeros((2, 4, 1, 1)); W[0, 0] = 2; W[0, 2] = 1; W[1, 1] = 2; W[1, 3] = 1
    P["Wrt"] = W
    W = np.zeros((2, 4, 1, 1)); W[0, 0] = 2; W[0, 2] = 0.5; W[1, 1] = 2; W[1, 3] = 0.5
    P["Wft"] = W
    n("rt", "conv", ["cr"], w="Wrt")
    n("ft", "conv", ["cf"], w="Wft")
    n("tr", "mul", ["rt", "riseW"])
    n("tf", "mul", ["ft", "fallW"])

    # goal adjacency to own container
    W = np.zeros((2, 2, 7, 7)); W[0, 0] = 1; W[1, 1] = 1
    P["Wd7"] = W
    n("Rd", "conv", ["FL.r"], w="Wd7")
    n("Rdm", "min2", ["Rd", "one"])
    n("bR", "rowmax", ["Rdm"])
    n("bC", "colmax", ["Rdm"])
    n("GbR", "mul", ["G", "bR"])
    n("GbC", "mul", ["G", "bC"])
    n("ginR", "gmax", ["GbR"])
    n("ginC", "gmax", ["GbC"])
    P["Wgc"] = np.array([[1.0, 1.0, 0.0, 0.0]])
    P["Wgr"] = np.array([[0.0, 0.0, 1.0, 1.0]])
    n("gc", "dense", ["flags"], w="Wgc")
    n("gr", "dense", ["flags"], w="Wgr")
    n("ginRg", "vmul", ["ginR", "gc"])
    n("ginCg", "vmul", ["ginC", "gr"])
    n("gin", "vadd", ["ginRg", "ginCg"])
    P["Wng"] = -np.eye(2); P["bng"] = np.ones(2)
    n("ng", "dense", ["gin"], w="Wng", b="bng")
    n("npc", "vmul", ["ng", "presP"])
    P["Wo2"] = np.ones((1, 2))
    n("nps", "dense", ["npc"], w="Wo2")

    # ---- pipe alignment ----
    n("gg2", "vconcat", ["gr", "gc"])
    P["Wsum2"] = np.ones((1, 2, 1, 1))
    def latd(name, src, r):
        k = 2 * r + 1
        W = np.zeros((2, 1, k, k)); W[0, 0, r, :] = 1; W[1, 0, :, r] = 1
        P["Wl" + name] = W
        n(name + "a", "conv", [src], w="Wl" + name)
        n(name + "b", "mul", [name + "a", "gg2"])
        n(name + "c", "conv", [name + "b"], w="Wsum2")
        n(name, "min2", [name + "c", "one"])
    def band(name, src):
        n(name + "a", "colmax", [src])
        n(name + "b", "rowmax", [src])
        n(name + "c", "concat", [name + "a", name + "b"])
        n(name + "d", "mul", [name + "c", "gg2"])
        n(name, "conv", [name + "d"], w="Wsum2")
    W = np.zeros((1, 16, 1, 1)); W[0, 1] = 1; W[0, 12] = 1
    P["Wpp"] = W
    n("PP", "conv", ["x0"], w="Wpp")
    n("p1", "slice", ["x0"], start=1, stop=2)
    n("p12", "slice", ["x0"], start=12, stop=13)
    P["Wrs"] = np.ones((1, 2, 1, 1))
    n("Rsum", "conv", ["FL.r"], w="Wrs")
    latd("Rl", "Rsum", 1)
    band("Rb", "Rl")
    n("RbP", "mul", ["Rb", "PP"])
    latd("RbD", "RbP", 3)
    n("adjP", "mul", ["RbD", "PP"])
    W = np.zeros((2, 3, 1, 1)); W[0, 1] = 1; W[1, 0] = 1
    P["Wlc"] = W
    n("lc2", "conv", ["coords"], w="Wlc")
    n("lc2g", "mul", ["lc2", "gg2"])
    n("lco", "conv", ["lc2g"], w="Wsum2")
    P["Wgs"] = np.ones((1, 2, 1, 1))
    n("Gs", "conv", ["G"], w="Wgs")
    n("Gsm", "min2", ["Gs", "one"])
    n("Gl", "mul", ["Gsm", "lco"])
    n("glv", "gmax", ["Gl"])
    n("dlt", "sub", ["lco", "glv"])
    n("adl", "abs", ["dlt"])
    n("cl", "sub", ["one", "adl"])
    n("cla", "add", ["cl", "one"])
    n("sc", "mul", ["cla", "adjP"])
    n("scm", "gmax", ["sc"])
    n("scd", "sub", ["sc", "scm"])
    P["Wsg"] = np.array([K]).reshape(1, 1, 1, 1); P["bsg"] = np.array([K * 0.004])
    n("scz", "conv", ["scd"], w="Wsg", b="bsg")
    n("scs", "sigmoid", ["scz"])
    n("sel", "mul", ["scs", "adjP"])
    n("s1", "mul", ["sel", "p1"])
    n("s12", "mul", ["sel", "p12"])
    n("pt", "mul", ["s12", "nps"])
    n("nPP", "sub", ["one", "PP"])
    latd("Ls", "s1", 1)
    n("L", "mul", ["Ls", "nPP"])
    P["Wd3"] = np.ones((1, 1, 3, 3))
    n("Lea", "conv", ["s1"], w="Wd3")
    n("Leb", "min2", ["Lea", "one"])
    n("Le", "mul", ["Leb", "nPP"])
    band("Lb", "L")
    n("Lbm", "min2", ["Lb", "one"])
    n("sS", "mul", ["blk", "L"])
    n("sA", "mul", ["blk", "Le"])
    n("sB", "mul", ["blk", "Lbm"])
    n("s3", "concat", ["sS", "sA", "sB"])
    W = np.zeros((3, 3, 3, 3))
    for i in range(3):
        W[i, i] = 1
    P["Wf3"] = W
    g.append({"id": "F3", "op": "loop", "T": 64,
              "state": [{"id": "f", "init": "s3"}],
              "body": [{"id": "fc", "op": "conv", "in": ["f"], "w": "Wf3"},
                       {"id": "fm", "op": "min2", "in": ["fc", "one"]},
                       {"id": "fn", "op": "mul", "in": ["fm", "pas"]}],
              "next": {"f": "fn"}})
    P["Wgw"] = np.array([0.0, -1.0, 1.0]).reshape(1, 3, 1, 1)
    P["Wsh"] = np.array([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    n("grz", "conv", ["F3.f"], w="Wgw")
    n("grw", "relu", ["grz"])
    n("shr", "conv", ["F3.f"], w="Wsh")
    P["Wav"] = np.ones((1, 4, 1, 1))
    n("avs", "conv", ["Am"], w="Wav")
    n("av", "min2", ["avs", "one"])
    n("gb", "mul", ["grw", "av"])
    n("al", "sub", ["gb", "shr"])
    n("alB", "mul", ["al", "B"])
    n("alg", "mul", ["alB", "nps"])

    n("tb", "concat", ["tr", "tf", "pt", "Am", "B", "alg"])
    P["Wclk"] = np.array([1, 1, 1, 1, 3, 0.01, 0.01, 0.01, 0.01, 0.001, 6],
                         dtype=np.float64).reshape(1, 11, 1, 1)
    n("click", "conv", ["tb"], w="Wclk")

    P["Wlog"] = np.zeros((8, 4))
    bl = np.zeros(8); bl[6] = 1.0
    P["blog"] = bl
    n("logits", "dense", ["flags"], w="Wlog", b="blog")

    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}
